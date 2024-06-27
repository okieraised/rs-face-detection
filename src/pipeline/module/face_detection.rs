use opencv::core::{self, Mat, MatTraitConst, Scalar, Size, MatTraitConstManual, MatExprTraitConst};
use opencv::imgproc::{INTER_LINEAR, resize};
use std::collections::HashMap;
use std::ops::{Mul, MulAssign};
use crate::processing::generate_anchors::{AnchorConfig, Config, generate_anchors_fpn2};
use crate::triton_client::client::TritonInferenceClient;
use anyhow::{Error, Result};
use ndarray::{Array, Array2, Array3, Array4, ArrayBase, Axis, concatenate, Dim, IntoDimension, Ix, Ix2, Ix3, OwnedRepr, s, ShapeBuilder};
use opencv::imgcodecs::imwrite;
use opencv::prelude::Boxed;
use crate::processing::bbox_transform::clip_boxes;
use crate::processing::nms::nms;
use crate::rcnn::anchors::anchors;
use crate::triton_client::client::triton::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use crate::triton_client::client::triton::{InferParameter, InferTensorContents, ModelConfigRequest, ModelInferRequest};
use crate::triton_client::client::triton::infer_parameter::ParameterChoice;
use crate::utils::utils::{argsort_descending, reorder_2d, reorder_3d, vstack_2d, vstack_3d};

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
            // println!("cfg {:?} {:?}", &out_cfg.name, &out_cfg.dims);
            cfg_outputs.push(InferRequestedOutputTensor { name: out_cfg.name.to_string(), parameters: Default::default() })
        }

        // println!("out {:?}", &cfg_outputs);

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

            let array4_f32: Array4<f32> = Array4::from_shape_vec(dimensions.into_dimension(), f_array).unwrap();
            let result_index = cfg_outputs.iter().position(|r| *r.name == output.name).unwrap();

            // println!("result_index {:?}", &result_index);
            net_out[result_index] =  array4_f32;
            // println!("out {:?} {:?}", array4_f32, model_out.model_name);
            // net_out.push(array4_f32);
        }

        // println!("net_out {:?} {:?}", &net_out, net_out.len());
        let mut proposals_list: Vec<Array2<f32>> = Vec::new();
        let mut scores_list: Vec<Array<f32, Dim<[Ix; 2]>>> = Vec::new();
        let mut landmarks_list: Vec<Array<f32, Ix3>> = Vec::new();

        let mut sym_idx = 0;

        for s in &self._feat_stride_fpn {
            let stride = *s as usize;
            // println!("s: {:?}", stride);

            // println!("sym_idx {:?}", &net_out[sym_idx + 1]);



            let mut scores = net_out[sym_idx].to_owned();
            // println!("scores {:?}",  scores);
            let sliced_scores = &scores.slice(s![.., self._num_anchors[&format!("stride{}", stride)].., .., ..]).to_owned();

            // println!("scores {:?}",  sliced_scores);


            let mut bbox_deltas = net_out[sym_idx + 1].to_owned();
            println!("bbox_deltas dim {:?}", &bbox_deltas.dim());

            // println!("bbox_deltas {:?}", &bbox_deltas);

            let height = bbox_deltas.shape()[2];
            let width = bbox_deltas.shape()[3];

            let A = self._num_anchors[&format!("stride{}", stride)];
            let K = height * width;
            // println!("height, width {:?} {:?}", &bbox_deltas.shape(), &bbox_deltas.shape());
            println!("K {:?}", &K);


            let anchors_fpn = &self._anchors_fpn[&format!("stride{}", stride)];
            let anchor_plane = anchors(height, width, stride, anchors_fpn);
            // println!("anchor_plane {:?}", &anchor_plane);
            let anchors_reshape = anchor_plane.into_shape((K * &A, 4)).unwrap();
            // println!("anchors_reshape {:?}", &anchors_reshape);

            let transposed_scores = sliced_scores.clone().permuted_axes([0, 2, 3, 1]);
            let scores_shape = transposed_scores.shape();
            let mut scores_dim: usize = 1;
            for dim in scores_shape {
                scores_dim *= dim;
            }
            let flattened_scores: Vec<f32> = transposed_scores.iter().cloned().collect();
            let arr_scores = Array::from(flattened_scores);
            // Calculate the new shape for reshaping
            // Reshape the array to (-1, 1)
            let reshaped_scores = arr_scores.into_shape((scores_dim, 1)).unwrap();
            // println!("reshaped_scores {:?}", &reshaped_scores);

            let bbox_deltas_transposed = bbox_deltas.permuted_axes([0, 2, 3, 1]);
            // println!("bbox_deltas {:?}", &bbox_deltas);

            let bbox_pred_len = bbox_deltas_transposed.dim().3 / &A;
            println!("bbox_pred_len {:?}", &bbox_pred_len);


            let bbox_deltas_shape = bbox_deltas_transposed.shape();
            println!("bbox_deltas_shape {:?}", &bbox_deltas_shape);

            let mut bbox_deltas_dim: usize = 1;
            for dim in bbox_deltas_shape {
                bbox_deltas_dim *= dim;
            }

            let flattened_bbox_deltas: Vec<f32> = bbox_deltas_transposed.iter().cloned().collect();
            let arr_bbox_deltas = Array::from(flattened_bbox_deltas);
            // println!("len arr_bbox_deltas {:?}", &arr_bbox_deltas.len());

            let mut bbox_deltas_reshaped = arr_bbox_deltas.into_shape(((bbox_deltas_dim) / &bbox_pred_len, bbox_pred_len)).unwrap();
            // println!("bbox_deltas_reshaped {:?}", &bbox_deltas_reshaped);

            for i in (0..4).step_by(4) {
                bbox_deltas_reshaped.slice_mut(s![.., i]).mul_assign(self.bbox_stds[i]);
                bbox_deltas_reshaped.slice_mut(s![.., i + 1]).mul_assign(self.bbox_stds[i + 1]);
                bbox_deltas_reshaped.slice_mut(s![.., i + 2]).mul_assign(self.bbox_stds[i + 2]);
                bbox_deltas_reshaped.slice_mut(s![.., i + 3]).mul_assign(self.bbox_stds[i + 3]);
            }
            // println!("bbox_deltas_reshaped {:?}", &bbox_deltas_reshaped);

            let mut proposals = self.bbox_pred(anchors_reshape.clone(), bbox_deltas_reshaped.to_owned());
            // println!("proposals {:?}", &proposals);
            // println!("im_info {:?}", &im_info);
            clip_boxes(&mut proposals, (im_info.height as usize, im_info.width as usize));
            // println!("clip_boxes {:?}", &proposals);

            let scores_ravel = reshaped_scores.view().iter().copied().collect::<Vec<_>>();
            // println!("scores_ravel {:?}", &scores_ravel);
            let order: Vec<usize> = scores_ravel.iter().enumerate().filter(|(_, &s)| s >= self.confidence_threshold).map(|(i, _)| i).collect();
            // println!("order {:?}", &order);
            let selected_proposals = proposals.select(Axis(0), &order);
            // println!("proposals {:?}", &selected_proposals);
            let selected_scores = reshaped_scores.select(Axis(0), &order);
            // println!("selected_scores {:?}", &selected_scores);

            proposals_list.push(selected_proposals);
            scores_list.push(selected_scores);

            if self.use_landmarks {
                let landmark_deltas = net_out[sym_idx + 2].to_owned();
                // println!("landmark_deltas {:?}", &landmark_deltas);
                let landmark_pred_len = landmark_deltas.dim().1 / A;
                // println!("landmark_pred_len {:?}", &landmark_pred_len);
                let transposed_landmark_deltas = &landmark_deltas.permuted_axes([0, 2, 3, 1]);

                let landmark_deltas_shape = transposed_landmark_deltas.shape();
                let mut landmark_deltas_dim: usize = 1;
                for dim in landmark_deltas_shape {
                    landmark_deltas_dim *= dim;
                }

                let flattened_landmark_deltas: Vec<f32> = transposed_landmark_deltas.iter().cloned().collect();
                let arr_landmark_deltas = Array::from(flattened_landmark_deltas);
                let mut reshaped_landmark_deltas = arr_landmark_deltas.into_shape((landmark_deltas_dim / landmark_pred_len, 5, landmark_pred_len / 5)).unwrap();
                // println!("reshaped_landmark_deltas {:?}", &reshaped_landmark_deltas);

                reshaped_landmark_deltas *= self.landmark_std;
                // println!("reshaped_landmark_deltas {:?}", &reshaped_landmark_deltas);

                let landmarks = self.landmark_pred(anchors_reshape, reshaped_landmark_deltas);
                // println!("landmarks {:?}", &landmarks);
                let selected_landmarks = landmarks.select(Axis(0), &order);
                // println!("selected_landmarks {:?}", &selected_landmarks);
                landmarks_list.push(selected_landmarks);
            }

            // break;
            if self.use_landmarks {
                sym_idx += 3;
            } else {
                sym_idx += 2;
            }
        }

        // println!("proposals_list {:?}", &proposals_list);
        let proposals = vstack_2d(proposals_list);

        let mut landmarks: Option<Array3<f32>> = None;

        // println!("proposals {:?}", &proposals);
        // println!("proposals.dim()[0] {:?}", &proposals.dim().0);
        if proposals.dim().0 == 0 {
            if self.use_landmarks {
                let det: Array2<f32> = Array2::zeros((0, 5));
                landmarks = Some(Array3::zeros((0, 5, 2)));
                //return Ok((det, landmarks));
            }
        }

        let score_stack = vstack_2d(scores_list);
        // println!("score_stack {:?}", &score_stack);
        let score_stack_ravel = score_stack.view().iter().copied().collect::<Vec<_>>();
        // println!("score_stack_ravel {:?}", &score_stack_ravel);

        let order = argsort_descending(&score_stack_ravel);
        // println!("order {:?}", &order);

        let selected_proposals = reorder_2d(proposals, &order);
        // println!("selected_proposals {:?}", &selected_proposals);

        let selected_score = reorder_2d(score_stack, &order);
        // println!("selected_score {:?}", &selected_score);

        if self.use_landmarks {
            let landmarks_stack = vstack_3d(landmarks_list);
            // println!("landmarks_stack {:?}", &landmarks_stack);

            landmarks = Some(reorder_3d(landmarks_stack, &order));
            // println!("selected_landmarks {:?}", &selected_landmarks);
        }


        let pre_det = concatenate![Axis(1), selected_proposals.slice(s![.., 0..4]).to_owned(), selected_score];
        // println!("pre_det {:?}", &pre_det);

        let keep = nms(&pre_det, self.iou_threshold);
        // println!("keep {:?}", &keep);

        let mut det = concatenate![Axis(1), pre_det, selected_proposals.slice(s![.., 4..]).to_owned()];
        // println!("det {:?}", &det);




        let selected_rows = &keep.iter()
            .map(|&i| det.slice(s![i, ..]).to_owned())
            .collect::<Vec<_>>();

        let new_det_shape = (selected_rows.len(), det.shape()[1]);


        // println!("det {:?}", &det);

        let det = Array2::from_shape_vec(
            new_det_shape,
            selected_rows.into_iter().flat_map(| row| row.iter().cloned()).collect()
        ).unwrap();

        println!("det_selected {:?}", &det);

        if self.use_landmarks {
            let selected_landmarks = &keep.iter()
                .map(|&i| landmarks.clone().unwrap().slice(s![i, .., ..]).to_owned())
                .collect::<Vec<_>>();

            // Convert the vector of arrays back into a 3D array
            let new_landmarks_shape = (selected_landmarks.len(), landmarks.clone().unwrap().shape()[1], landmarks.clone().unwrap().shape()[2]);
            landmarks = Some(Array3::from_shape_vec(
                new_landmarks_shape,
                selected_landmarks.into_iter().flat_map(|array| array.iter().cloned()).collect()
            ).unwrap());


            println!("landmarks {:?}", &landmarks);
        }
        // Ok((det, landmarks))

    }


    fn bbox_pred(&self, boxes: ArrayBase<OwnedRepr<f32>, Ix2>, box_deltas: ArrayBase<OwnedRepr<f32>, Ix2>) -> Array2<f32> {
        if boxes.shape()[0] == 0 {
            return Array2::zeros((0, box_deltas.shape()[1]));
        }

        let boxes = boxes.mapv(|x| x as f32);
        let widths = &boxes.slice(s![.., 2]) - &boxes.slice(s![.., 0]) + 1.0;
        let heights = &boxes.slice(s![.., 3]) - &boxes.slice(s![.., 1]) + 1.0;
        let ctr_x = &boxes.slice(s![.., 0]) + 0.5 * (&widths - 1.0);
        let ctr_y = &boxes.slice(s![.., 1]) + 0.5 * (&heights - 1.0);

        let dx = box_deltas.slice(s![.., 0..1]);
        let dy = box_deltas.slice(s![.., 1..2]);
        let dw = box_deltas.slice(s![.., 2..3]);
        let dh = box_deltas.slice(s![.., 3..4]);

        let pred_ctr_x = &dx * &widths.clone().insert_axis(Axis(1)) + &ctr_x.insert_axis(Axis(1));
        let pred_ctr_y = &dy * &heights.clone().insert_axis(Axis(1)) + &ctr_y.insert_axis(Axis(1));
        let pred_w = dw.mapv(f32::exp) * &widths.insert_axis(Axis(1));
        let pred_h = dh.mapv(f32::exp) * &heights.insert_axis(Axis(1));

        let mut pred_boxes = Array2::<f32>::zeros(box_deltas.raw_dim());

        pred_boxes.slice_mut(s![.., 0..1]).assign(&(&pred_ctr_x - 0.5 * (&pred_w - 1.0)));
        pred_boxes.slice_mut(s![.., 1..2]).assign(&(&pred_ctr_y - 0.5 * (&pred_h - 1.0)));
        pred_boxes.slice_mut(s![.., 2..3]).assign(&(&pred_ctr_x + 0.5 * (&pred_w - 1.0)));
        pred_boxes.slice_mut(s![.., 3..4]).assign(&(&pred_ctr_y + 0.5 * (&pred_h - 1.0)));

        if box_deltas.shape()[1] > 4 {
            pred_boxes.slice_mut(s![.., 4..]).assign(&box_deltas.slice(s![.., 4..]));
        }

        pred_boxes
    }

    fn landmark_pred(&self, boxes: ArrayBase<OwnedRepr<f32>, Ix2>, landmark_deltas: ArrayBase<OwnedRepr<f32>, Ix3>) -> ArrayBase<OwnedRepr<f32>, Ix3> {
        if boxes.shape()[0] == 0 {
            return Array3::zeros((0, landmark_deltas.shape()[1], landmark_deltas.shape()[2]));
        }

        let boxes = boxes.mapv(|x| x as f32);
        let widths = &boxes.slice(s![.., 2]) - &boxes.slice(s![.., 0]) + 1.0;
        let heights = &boxes.slice(s![.., 3]) - &boxes.slice(s![.., 1]) + 1.0;
        let ctr_x = &boxes.slice(s![.., 0]) + 0.5 * (&widths - 1.0);
        let ctr_y = &boxes.slice(s![.., 1]) + 0.5 * (&heights - 1.0);

        let mut pred = landmark_deltas.clone();

        for i in 0..5 {
            pred.slice_mut(s![.., i, 0]).assign(&(&landmark_deltas.slice(s![.., i, 0]) * &widths + &ctr_x));
            pred.slice_mut(s![.., i, 1]).assign(&(&landmark_deltas.slice(s![.., i, 1]) * &heights + &ctr_y));
        }

        pred
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
