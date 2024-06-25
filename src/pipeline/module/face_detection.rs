use opencv::core::{self, Mat, MatTraitConst, Scalar, Size, CV_32FC3, CV_32FC1, CV_32F, MatTraitConstManual, MatExprTraitConst, copy_to, CV_8UC3, CV_8UC1};
use opencv::imgproc::{INTER_LINEAR, resize, cvt_color, COLOR_BGR2RGB, COLOR_RGBA2RGB, COLOR_BGRA2GRAY, COLOR_BGR2GRAY};
use std::collections::HashMap;
use std::ffi::c_float;
use std::ops::Mul;
use crate::processing::generate_anchors::{AnchorConfig, Config, generate_anchors_fpn2};
use crate::triton_client::client::TritonInferenceClient;
use anyhow::{Error, Result};
use bytemuck::cast_slice;
use image::ImageError::Parameter;
use ndarray::{Array, Array2, Array3, Array4, Axis, Dim, IntoDimension, s};
use opencv::imgcodecs::imwrite;
use opencv::imgproc;
use opencv::imgproc::ColorConversionCodes::COLOR_GRAY2BGR;
use opencv::prelude::Boxed;
use crate::processing::bbox_transform::clip_boxes;
use crate::processing::nms::nms;
use crate::rcnn::anchors::anchors;
use crate::triton_client::client::triton::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use crate::triton_client::client::triton::{InferParameter, InferTensorContents, ModelConfigRequest, ModelInferRequest};
use crate::triton_client::client::triton::infer_parameter::ParameterChoice;

pub struct RetinaFaceDetection {
    triton_infer_client: TritonInferenceClient,
    image_size: (i32, i32),
    use_landmarks: bool,
    confidence_threshold: f32,
    iou_threshold: f32,
    fpn_keys: Vec<String>,
    _feat_stride_fpn: Vec<i32>,
    anchor_cfg: HashMap<String, AnchorConfig>,
    _anchors_fpn: HashMap<String, Array2<f32>>,
    _num_anchors: HashMap<String, usize>,
    pixel_means: Vec<f32>,
    pixel_stds: Vec<f32>,
    pixel_scale: f32,
    bbox_stds: Vec<f32>,
    landmark_std: f32,
}

impl RetinaFaceDetection {
    pub async fn new(
        triton_host: &str,
        triton_port: &str,
        image_size: (i32, i32),
        max_batch_size: i32,
        confidence_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Self, Error> {
        let triton_infer_client = match TritonInferenceClient::new(triton_host, triton_port).await {
            Ok(triton_infer_client) => triton_infer_client,
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        let mut fpn_keys = vec![];
        let _feat_stride_fpn = vec![32, 16, 8];
        let _ratio = vec![1.0];

        let mut anchor_cfg = HashMap::new();
        anchor_cfg.insert(
            "32".to_string(), AnchorConfig {
                scales: vec![32.0, 16.0],
                base_size: 16,
                ratios: _ratio.clone(),
                allowed_border: 9999
            }
        );
        anchor_cfg.insert(
            "16".to_string(),
            AnchorConfig {
                base_size: 16,
                ratios: _ratio.clone(),
                scales: vec![8.0, 4.0],
                allowed_border: 9999,
            },
        );
        anchor_cfg.insert(
            "8".to_string(), AnchorConfig {
                scales: vec![2.0, 1.0],
                base_size: 16,
                ratios: _ratio.clone(),
                allowed_border: 9999
            }
        );
        let config = Config {
            rpn_anchor_cfg: anchor_cfg.clone(),
        };

        for s in &_feat_stride_fpn {
            fpn_keys.push(format!("stride{}", s));
        }

        let dense_anchor = false;

        let _anchors_fpn = fpn_keys
            .iter()
            .zip(generate_anchors_fpn2(dense_anchor, Some(&config)))
            .map(|(k, v)| (k.clone(), v))
            .collect::<HashMap<_, _>>();



        let _num_anchors = _anchors_fpn
            .iter()
            .map(|(k, v)| (k.clone(), v.clone().shape()[0]))
            .collect::<HashMap<_, _>>();

        // println!("_anchors_fpn: {:?}", _anchors_fpn);
        // println!("_num_anchors: {:?}", _num_anchors);

        let pixel_means = vec![0.0, 0.0, 0.0];
        let pixel_stds = vec![1.0, 1.0, 1.0];
        let pixel_scale = 1.0;

        Ok(RetinaFaceDetection {
            triton_infer_client,
            image_size,
            use_landmarks: true,
            confidence_threshold,
            iou_threshold,
            fpn_keys,
            _feat_stride_fpn,
            anchor_cfg,
            _anchors_fpn,
            _num_anchors,
            pixel_means,
            pixel_stds,
            pixel_scale,
            bbox_stds: vec![1.0, 1.0, 1.0, 1.0],
            landmark_std: 1.0,
        })
    }

    pub fn _preprocess(&self, img: &Mat) -> Result<(Mat, f32), Error> {

        let img_shape = match img.size() {
            Ok(img_shape) => img_shape,
            Err(e) => return Err(Error::from(e))
        };

        let im_ratio = img_shape.height as f32 / img_shape.width as f32;
        let model_ratio = self.image_size.1 as f32 / self.image_size.0 as f32;

        let (new_width, new_height) = if im_ratio > model_ratio {
            let new_height = self.image_size.1;
            let new_width = (new_height as f32 / im_ratio) as i32;
            (new_width, new_height)
        } else {
            let new_width = self.image_size.0;
            let new_height = (new_width as f32 * im_ratio) as i32;
            (new_width, new_height)
        };

        let det_scale = new_height as f32 / img_shape.height as f32;

        let mut resized_img = Mat::default();

        match resize(&img, &mut resized_img, Size::new(new_width, new_height), 0.0, 0.0, INTER_LINEAR) {
            Ok(_) => {},
            Err(e) => return Err(Error::from(e))
        }

        //imwrite("./resized.png", &resized_img, &core::Vector::default()).unwrap();


        let mut det_img = match Mat::new_rows_cols_with_default(self.image_size.1, self.image_size.0, core::CV_8UC3, Scalar::all(0.0)) {
            Ok(det_img) => det_img,
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        let  mut roi = match Mat::roi_mut(&mut det_img, core::Rect::new(0, 0, new_width, new_height)) {
            Ok(roi) => roi,
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        match  resized_img.copy_to(&mut roi) {
            Ok(_) => {},
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        //imwrite("./det_img.png", &det_img, &core::Vector::default()).unwrap();

        Ok((det_img, det_scale))
    }


    pub async fn _forward(&self, img: &Mat) { //  -> Result<(Array2<f32>, Option<Array3<f32>>), Error>
        let mut im = img.clone();
        imwrite("./im.png", &im, &core::Vector::default()).unwrap();

        let im_info = im.size().unwrap();
        let rows = im_info.height;
        let cols = im_info.width;

        print!("{:?} {:?}", &rows, &cols);
        let mut im_tensor = Array4::<f32>::zeros((1, 3, rows as usize, cols as usize));

        // Convert the image to float and normalize it
        for i in 0..3 {
            for y in 0..rows {
                for x in 0..cols {
                    let pixel_value = im.at_2d::<core::Vec3b>(y, x).unwrap()[2 - i];
                    im_tensor[[0, i, y as usize, x as usize]] = (pixel_value as f32 / self.pixel_scale - self.pixel_means[2 - i]) / self.pixel_stds[2 - i];
                }
            }
        }

        // println!("im_tensor {:?}", im_tensor.clone());
        let vec = im_tensor.into_raw_vec();

        let mut param : HashMap<String,  InferParameter> = HashMap::new();
        param.insert("max_batch_size".to_string(), InferParameter { parameter_choice: Option::from(ParameterChoice::Int64Param(1)) });

        let models = self.triton_infer_client
            .model_config(ModelConfigRequest {
                name: "face_detection_retina".to_string(),
                version: "".to_string(),
            }).await.unwrap();

        let cfg = models.config.unwrap();

        let mut cfg_outputs = Vec::<InferRequestedOutputTensor>::new();

        for out_cfg in cfg.output.iter() {
            cfg_outputs.push(InferRequestedOutputTensor { name: out_cfg.name.to_string(), parameters: Default::default() })
        }

        println!("out {:?}", &cfg_outputs);

        let model_request = ModelInferRequest{
            model_name: "face_detection_retina".to_string(),
            model_version: "".to_string(),
            id: "".to_string(),
            parameters: Default::default(),
            inputs: vec![InferInputTensor {
                name: "data".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![1, 3, 640, 640],
                parameters: param,
                contents: Option::from(InferTensorContents {
                    bool_contents: vec![],
                    int_contents: vec![],
                    int64_contents: vec![],
                    uint_contents: vec![],
                    uint64_contents: vec![],
                    fp32_contents: vec,
                    fp64_contents: vec![],
                    bytes_contents: vec![],
                }),
            }],
            outputs: cfg_outputs.clone(),
            raw_input_contents: vec![],
        };

        let mut model_out = match self.triton_infer_client.model_infer(model_request).await {
            Ok(model_out) => model_out,
            Err(e) => {
                println!("{:?}", e);
                return //Err(Error::from(e))
            }
        };

        let mut net_out: Vec<Array4<f32>> = vec![Array4::zeros([1,1,1,1]); cfg_outputs.len()];

        for (idx, output) in &mut model_out.outputs.iter_mut().enumerate() {
            let dims = &output.shape;
            let dimensions: [usize; 4] = [
                dims[0] as usize,
                dims[1] as usize,
                dims[2] as usize,
                dims[3] as usize,
            ];
            let u8_array: &[u8] = &model_out.raw_output_contents[idx];
            let f_array = u8_to_f32_vec(u8_array);
            let array4_f32: Array4<f32> = Array4::from_shape_fn(dimensions.into_dimension(), |(_, _, _, idx)| {
                f_array[idx]
            });
            let result_index = cfg_outputs.iter().position(|r| *r.name == output.name).unwrap();
            net_out.insert(result_index, array4_f32);
            // println!("out {:?} {:?}", array4_f32, model_out.model_name);
            // net_out.push(array4_f32);
        }


        // let mut proposals_list = Vec::new();
        // let mut scores_list = Vec::new();
        // let mut landmarks_list = Vec::new();

        let mut sym_idx = 0;

        for s in &self._feat_stride_fpn {
            let stride = *s as usize;
            // println!("sym_idx {:?}", &net_out[sym_idx + 1]);



            let mut scores = &net_out[sym_idx];

            // let sliced_scores = scores.slice(s![.., self._num_anchors[&format!("stride{}", stride)].., .., ..]).to_owned();
            println!("scores {:?}", &scores);


            // let mut bbox_deltas = &net_out[sym_idx + 1];
            //
            // let height = bbox_deltas.shape()[2];
            // let width = bbox_deltas.shape()[3];

            // let A = self._num_anchors[&format!("stride{}", stride)];
            // println!("height, width {:?} {:?}", bbox_deltas.shape(), bbox_deltas.shape());
            // let K = height * width;
            // println!("K {:?}", &K);
            // let anchors_fpn = &self._anchors_fpn[&format!("stride{}", stride)];
            // let anchor_plane = anchors(height, width, stride, anchors_fpn);
            // let anchors_shape = anchor_plane.into_shape((K * &A, 4)).unwrap();
            // println!("anchors_shape {:?}", &anchors_shape);
            //
            // let permuted_scores = scores.clone().permuted_axes([0, 2, 3, 1]).unwrap();
            // println!("permuted_scores {:?}", &permuted_scores);
            //---------------------------------------------------------------------------------------
            //     let bbox_deltas = bbox_deltas.permuted_axes([0, 2, 3, 1]);
            //     let bbox_pred_len = bbox_deltas.dim().3 / &A;
            //     let bbox_deltas = bbox_deltas.into_shape((-1, bbox_pred_len))?;
            //
            //     let mut bbox_deltas = bbox_deltas.to_owned();
            //     for i in (0..4).step_by(4) {
            //         bbox_deltas.slice_mut(s![.., i]).mul_assign(self.bbox_stds[i]);
            //         bbox_deltas.slice_mut(s![.., i + 1]).mul_assign(self.bbox_stds[i + 1]);
            //         bbox_deltas.slice_mut(s![.., i + 2]).mul_assign(self.bbox_stds[i + 2]);
            //         bbox_deltas.slice_mut(s![.., i + 3]).mul_assign(self.bbox_stds[i + 3]);
            //     }
            //     let proposals = self.bbox_pred(&anchors, &bbox_deltas);
            //     clip_boxes(proposals, &im_info);
            //
            //     let scores_ravel = scores.view().iter().copied().collect::<Vec<_>>();
            //     let order: Vec<usize> = scores_ravel.iter().enumerate().filter(|(_, &s)| s >= self.confidence_threshold).map(|(i, _)| i).collect();
            //     let proposals = proposals.select(Axis(0), &order);
            //     let scores = scores.select(Axis(0), &order);
            //
            //     proposals_list.push(proposals);
            //     scores_list.push(scores);
            //
            //     if self.use_landmarks {
            //         let landmark_deltas = net_out[sym_idx + 2];
            //         let landmark_pred_len = landmark_deltas.dim().1 / A;
            //         let landmark_deltas = landmark_deltas.permuted_axes([0, 2, 3, 1]).into_shape((-1, 5, landmark_pred_len / 5))?;
            //         let mut landmark_deltas = landmark_deltas.to_owned();
            //         landmark_deltas *= self.landmark_std;
            //         let landmarks = self.landmark_pred(&anchors, &landmark_deltas);
            //         let landmarks = landmarks.select(Axis(0), &order);
            //         landmarks_list.push(landmarks);
            //     }
            //
            break;
            if self.use_landmarks {
                sym_idx += 3;
            } else {
                sym_idx += 2;
            }
        }


            //
            // let proposals = ndarray::stack(Axis(0), &proposals_list.iter().map(|a| a.view()).collect::<Vec<_>>())?;
            // let landmarks = if self.use_landmarks {
            //     let landmarks = ndarray::stack(Axis(0), &landmarks_list.iter().map(|a| a.view()).collect::<Vec<_>>())?;
            //     Some(landmarks)
            // } else {
            //     None
            // };
            //
            // if proposals.is_empty() {
            //     let landmarks = if self.use_landmarks {
            //         Some(Array::zeros((0, 5, 2)))
            //     } else {
            //         None
            //     };
            //     return Ok((Array::zeros((0, 5)), landmarks));
            // }
            //
            // let scores = ndarray::stack(Axis(0), &scores_list.iter().map(|a| a.view()).collect::<Vec<_>>())?;
            // let scores_ravel = scores.view().iter().copied().collect::<Vec<_>>();
            // let order = scores_ravel.into_iter().enumerate().sorted_by(|a, b| b.1.partial_cmp(&a.1).unwrap()).map(|(i, _)| i).collect::<Vec<_>>();
            // let proposals = proposals.select(Axis(0), &order);
            // let scores = scores.select(Axis(0), &order);
            //
            // let pre_det = ndarray::concatenate![Axis(1), proposals.slice(s![.., ..4]), scores];
            // let keep = nms(&pre_det, self.iou_threshold);
            // let det = ndarray::concatenate![Axis(1), pre_det.select(Axis(0), &keep), proposals.slice(s![.., 4..])];
            // let landmarks = landmarks.map(|landmarks| landmarks.select(Axis(0), &keep));
            //
            // Ok((det, landmarks))

    }
}


fn u8_to_f32_vec(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(TryInto::try_into)
        .map(Result::unwrap)
        .map(f32::from_le_bytes)
        .collect()
}

fn convert_vecs<T, U>(v: Vec<T>) -> Vec<U>
    where
        T: Into<U>,
{
    v.into_iter().map(Into::into).collect()
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use opencv::core::Mat;
    use crate::pipeline::module::face_detection::RetinaFaceDetection;
    use crate::processing::generate_anchors::generate_anchors_fpn2;
    use crate::utils::utils::byte_data_to_opencv;

    #[tokio::test]
    async fn test_retina_face_detection() {
        let retina_face_detection = match RetinaFaceDetection::new(
            "",
            "8603",
            (640, 640),
            1,
            0.7,
            0.45,
        ).await
        {
            Ok(retina_face_detection)  => retina_face_detection,
            Err(e) => {
                println!("{:?}", e);
                return
            }
        };
    }

    #[tokio::test]
    async fn test_preprocess() {
        let retina_face_detection = match RetinaFaceDetection::new(
            "",
            "",
            (640, 640),
            1,
            0.7,
            0.45,
        ).await
        {
            Ok(retina_face_detection)  => retina_face_detection,
            Err(e) => {
                println!("{:?}", e);
                return
            }
        };

        let im_bytes: &[u8] = include_bytes!("");
        let image = byte_data_to_opencv(im_bytes).unwrap();

        let (preprocessed_img, scale) = retina_face_detection._preprocess(&image).unwrap();

        println!("{:?}", &preprocessed_img);
        println!("Preprocessing done with scale: {}", scale);

        retina_face_detection._forward(&preprocessed_img).await;
    }


    #[tokio::test]
    async fn test_forward() {
    }
}
