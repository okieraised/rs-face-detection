use opencv::core::{self, Mat, MatTraitConst, Scalar, Size, CV_32FC3, CV_32FC1, CV_32F};
use opencv::imgproc::{INTER_LINEAR, resize, cvt_color, COLOR_BGR2RGB, COLOR_RGBA2RGB, COLOR_BGRA2GRAY};
use std::collections::HashMap;
use std::ffi::c_float;
use crate::processing::generate_anchors::{AnchorConfig, Config, generate_anchors_fpn2};
use crate::triton_client::client::TritonInferenceClient;
use anyhow::{Error, Result};
use ndarray::{Array, Array2, Array3, Array4, Axis, s};
use opencv::prelude::Boxed;
use crate::processing::bbox_transform::clip_boxes;
use crate::processing::nms::nms;
use crate::rcnn::anchors::anchors;
use crate::triton_client::client::triton::ModelInferRequest;

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
    _num_anchors: HashMap<String, Array2<f32>>,
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
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<HashMap<_, _>>();

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


        let mut det_img = match Mat::new_rows_cols_with_default(self.image_size.1, self.image_size.0, core::CV_8UC3, Scalar::all(0.0)) {
            Ok(det_img) => det_img,
            Err(e) => {
                return Err(Error::from(e))
            }
        };


        let mut roi = match Mat::roi(&mut det_img, core::Rect::new(0, 0, new_width, new_height)) {
            Ok(roi) => roi,
            Err(e) => {
                return Err(Error::from(e))
            }
        };

       match  resized_img.copy_to(&mut roi.clone_pointee()) {
           Ok(_) => {},
           Err(e) => {
               return Err(Error::from(e))
           }
       };

        Ok((det_img, det_scale))
    }


    pub async fn _forward(&self, img: &Mat) { //  -> Result<(Array2<f32>, Option<Array3<f32>>), Error>
        // let mut proposals_list = Vec::new();
        // let mut scores_list = Vec::new();
        // let mut landmarks_list = Vec::new();

        // let mut im = img.clone();
        let mut temp = img.clone();

        match img.convert_to(&mut temp, CV_32F, 1.0, 0.0) {
            Ok(_) => {},
            Err(e) => {
                println!("{:?}", e);
                return //Err(Error::from(e))
            }
        };

        // let mut temp = im.clone();
        let mut im =  Mat::default();

        cvt_color(&mut temp.clone(), &mut im, COLOR_BGRA2GRAY, 0).unwrap();
        println!("immm {:?}", &im);


        let im_info = [im.rows() as f32, im.cols() as f32];
        let mut im_tensor = Array4::<f32>::zeros((1, 3, im.rows() as usize, im.cols() as usize));

        for i in 0..3 {
            let channel = (2 - i) as usize;
            let (mean, std) = (self.pixel_means[channel], self.pixel_stds[channel]);
            for x in 0..im.rows() {
                for y in 0..im.cols() {
                    im_tensor[(0, i, x as usize, y as usize)] = ((im.at_2d::<f32>(x, y).unwrap() / self.pixel_scale - mean) / std);
                }
            }
        }

        println!("{:?}", im_tensor)


        // let model_request = ModelInferRequest{
        //     model_name: "".to_string(),
        //     model_version: "".to_string(),
        //     id: "".to_string(),
        //     parameters: Default::default(),
        //     inputs: vec![],
        //     outputs: vec![],
        //     raw_input_contents: vec![im_tensor.into_raw_vec()],
        // };
        //
        // let net_out = match self.triton_infer_client.model_infer(model_request).await {
        //     Ok(net_out) => net_out,
        //     Err(e) => {
        //         return Err(Error::from(e))
        //     }
        // };
        //
        // let mut sym_idx = 0;
        //
        // for s in &self._feat_stride_fpn {
        //     let stride = *s as usize;
        //     let scores = &net_out.raw_output_contents[sym_idx];
        //     let bbox_deltas = &net_out.raw_output_contents[sym_idx + 1];
        //
        //     let height = bbox_deltas.dim().2;
        //     let width = bbox_deltas.dim().3;
        //
        //     let A = self._num_anchors[&format!("stride{}", stride)].clone().into_raw_vec().to_owned();
        //     let K = height * width;
        //     let anchors_fpn = &self._anchors_fpn[&format!("stride{}", stride)];
        //     let anchors = anchors(height, width, stride, anchors_fpn);
        //     let anchors = anchors.into_shape((K * &A, 4))?;
        //
        //     let scores = scores.permuted_axes([0, 2, 3, 1]).into_shape((-1, 1))?;
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
        //     if self.use_landmarks {
        //         sym_idx += 3;
        //     } else {
        //         sym_idx += 2;
        //     }
        // }
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

        let im_bytes: &[u8] = include_bytes!("/home/tripg/Documents/repo/rs-faceid-pipeline/test_data/anderson.jpg");
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
