use anyhow::Error;
use ndarray::{Array2, Array4, IntoDimension, s};
use opencv::core::{Mat, MatTraitConst, Size};
use opencv::imgproc::{COLOR_BGR2RGB, cvt_color, INTER_LINEAR, resize};
use crate::triton_client::client::triton::{InferTensorContents, ModelConfigResponse, ModelInferRequest};
use crate::triton_client::client::triton::model_infer_request::InferInputTensor;
use crate::triton_client::client::TritonInferenceClient;
use crate::utils::utils::u8_to_f32_vec;

#[derive(Debug)]
pub(crate) struct FaceQualityAssessment {
    triton_infer_client: TritonInferenceClient,
    triton_model_config: ModelConfigResponse,
    model_name: String,
    image_size: (i32, i32),
    threshold: f32,
    batch_size: i32,
}


impl FaceQualityAssessment {
    pub async fn new(
        triton_infer_client: TritonInferenceClient,
        triton_model_config: ModelConfigResponse,
        model_name: String,
        image_size: (i32, i32),
        batch_size: i32,
        threshold: f32,
    ) -> Result<Self, Error> {
        Ok(FaceQualityAssessment {
            triton_infer_client,
            triton_model_config,
            model_name,
            image_size,
            threshold,
            batch_size,
        })
    }

    pub async fn call(&self, images: &[Mat], is_debug: Option<bool>) -> Result<(Vec<f32>, Vec<i32>), Error>{

        let debug = is_debug.unwrap_or(false);

        let batch_size = images.len();
        let mut scores: Vec<f32> = vec![];
        let mut idxs: Vec<i32> = vec![];

        for i in 0..batch_size {
            let mut resized_image = Mat::default();
            match resize(&images[i], &mut resized_image,Size::new(self.image_size.0, self.image_size.1), 0.0, 0.0, INTER_LINEAR){
                Ok(_) => {},
                Err(e) => return Err(Error::from(e)),
            }

            let mut rgb_img = Mat::default();

            match cvt_color(&resized_image, &mut rgb_img, COLOR_BGR2RGB, 0) {
                Ok(_) => {}
                Err(e) => return Err(Error::from(e)),
            };

            let im_info = match rgb_img.size() {
                Ok(im_info) => {im_info}
                Err(e) => return Err(Error::from(e))
            };
            let rows = im_info.height;
            let cols = im_info.width;

            let mut im_tensor = Array4::<f32>::zeros((1, rows as usize, cols as usize, 3));

            // Convert the image to float and normalize it
            for i in 0..3 {
                for y in 0..rows {
                    for x in 0..cols {
                        let pixel_value = rgb_img.at_2d::<opencv::core::Vec3b>(y, x).unwrap()[i];
                        im_tensor[[0, y as usize, x as usize, i]] = (pixel_value as f32 - 127.5) * 0.00784313725;
                    }
                }
            }
            if debug {
                println!("face_quality_assessment - im_tensor: {:?}", im_tensor)
            }

            let transposed_tensor = im_tensor.permuted_axes([0, 3, 1, 2]);
            if debug {
                println!("face_quality_assessment - transposed_tensor: {:?}", transposed_tensor)
            }

            let flattened_vec: Vec<f32> = transposed_tensor.iter().cloned().collect();

            let model_cfg = match &self.triton_model_config.config {
                None => {
                    return Err(Error::msg("face_quality_assessment - face quality assessment model config is empty"))
                }
                Some(model_cfg) => {model_cfg}
            };

            let model_request = ModelInferRequest{
                model_name: self.model_name.to_owned(),
                model_version: "".to_string(),
                id: "".to_string(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: model_cfg.input[0].name.to_string(),
                    datatype: model_cfg.input[0].data_type().as_str_name()[5..].to_uppercase(),
                    shape: model_cfg.input[0].dims.to_owned(),
                    parameters: Default::default(),
                    contents: Option::from(InferTensorContents {
                        bool_contents: vec![],
                        int_contents: vec![],
                        int64_contents: vec![],
                        uint_contents: vec![],
                        uint64_contents: vec![],
                        fp32_contents: flattened_vec,
                        fp64_contents: vec![],
                        bytes_contents: vec![],
                    }),
                }],
                outputs: Default::default(),
                raw_input_contents: vec![],
            };

            let mut model_out = match self.triton_infer_client.model_infer(model_request).await {
                Ok(model_out) => model_out,
                Err(e) => {
                    return Err(Error::from(e))
                }
            };

            let mut net_out: Vec<Array2<f32>> = vec![];

            for (idx, output) in &mut model_out.outputs.iter_mut().enumerate() {
                let dims = &output.shape;
                let dimensions: [usize; 2] = [
                    dims[0] as usize,
                    dims[1] as usize,
                ];
                let u8_array: &[u8] = &model_out.raw_output_contents[idx];
                let f_array = u8_to_f32_vec(u8_array);

                let array2_f32: Array2<f32> = match Array2::from_shape_vec(dimensions.into_dimension(), f_array) {
                    Ok(array2_f32) => {array2_f32}
                    Err(e) => {
                        return Err(Error::from(e))
                    }
                };
                net_out.push(array2_f32);
            }

            let score = net_out[0].slice(s![0, 0]).into_scalar().to_owned();
            let predict = if score > self.threshold {
                1
            } else {
                0
            };

            idxs.push(predict);
            scores.push(score);
        }

        if debug {
            println!("face_quality_assessment - idxs: {:?}", idxs);
            println!("face_quality_assessment - scores: {:?}", scores);
        }

        Ok((scores, idxs))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, s};
    use crate::pipeline::module::face_alignment::FaceAlignment;
    use crate::pipeline::module::face_antispoofing::FaceAntiSpoofing;
    use crate::pipeline::module::face_detection::RetinaFaceDetection;
    use crate::pipeline::module::face_quality_assessment::FaceQualityAssessment;
    use crate::pipeline::module::face_selection::FaceSelection;
    use crate::triton_client::client::triton::{ModelConfigRequest, ModelConfigResponse};
    use crate::triton_client::client::TritonInferenceClient;
    use crate::utils::utils::byte_data_to_opencv;

    // #[tokio::test]
    // async fn test_face_antispoofing() {
    //     let triton_host = "";
    //     let triton_port = "";
    //     let im_bytes: &[u8] = include_bytes!("");
    //     let image = byte_data_to_opencv(im_bytes).unwrap();
    //
    //     let triton_infer_client = match TritonInferenceClient::new(triton_host, triton_port).await {
    //         Ok(triton_infer_client) => triton_infer_client,
    //         Err(e) => {
    //             println!("{:?}", e);
    //             return
    //         }
    //     };
    //
    //     let model_name = "face_detection_retina".to_string();
    //
    //     let face_detection_model_config = match triton_infer_client
    //         .model_config(ModelConfigRequest {
    //             name: model_name.to_owned(),
    //             version: "".to_string(),
    //         }).await {
    //         Ok(model_config_resp) => {model_config_resp}
    //         Err(e) => {
    //             println!("{:?}", e);
    //             return
    //         }
    //     };
    //
    //     let retina_face_detection = match RetinaFaceDetection::new(
    //         triton_infer_client.clone(),
    //         face_detection_model_config,
    //         model_name,
    //         (640, 640),
    //         1,
    //         0.7,
    //         0.45,
    //     ).await
    //     {
    //         Ok(retina_face_detection)  => retina_face_detection,
    //         Err(e) => {
    //             println!("{:?}", e);
    //             return
    //         }
    //     };
    //     let (preprocessed_img, scale) = retina_face_detection.call(&image, Some(false)).await.unwrap();
    //     let face_selection = FaceSelection::new(0.3, 0.3, 0.1, 0.0075).await;
    //     let (selected_face_box, selected_face_point) = face_selection.call(&image, preprocessed_img, scale, Some(false), None).unwrap();
    //
    //
    //     let model_names: Vec<String> = vec![
    //         "miniFAS_4".to_string(),
    //         "miniFAS_2_7".to_string(),
    //         "miniFAS_2".to_string(),
    //         "miniFAS_1".to_string(),
    //     ];
    //
    //
    //     let mut model_antispoofing_config: Vec<ModelConfigResponse> = vec![];
    //
    //     for model_name in &model_names {
    //         let face_antispoofing_model_config = match triton_infer_client
    //             .model_config(ModelConfigRequest {
    //                 name: model_name.to_owned(),
    //                 version: "".to_string(),
    //             }).await {
    //             Ok(model_config_resp) => {model_config_resp}
    //             Err(e) => {
    //                 println!("{:?}", e);
    //                 return
    //             }
    //         };
    //         model_antispoofing_config.push(face_antispoofing_model_config)
    //     }
    //
    //     let face_antispoofing = match FaceAntiSpoofing::new(
    //         triton_infer_client.clone(),
    //         model_antispoofing_config.clone(),
    //         model_names.clone(),
    //         vec![
    //             (80, 80),
    //             (80, 80),
    //             (256, 256),
    //             (128, 128),
    //         ],
    //         vec![4.0, 2.7, 2.0, 1.0],
    //         1,
    //         0.55).await {
    //         Ok(face_antispoofing) => {face_antispoofing}
    //         Err(e) => {
    //             println!("{:?}", e);
    //             return
    //         }
    //     };
    //
    //     if let Some(ref _selected_face_box) = selected_face_box {
    //         let sliced_boxes = _selected_face_box.slice(s![..4]);
    //         let boxes: Array1<f32> = sliced_boxes.to_owned();
    //         let result = face_antispoofing.call(vec![&image],vec![&boxes], None).await.unwrap();
    //
    //         println!("result: {:?}", result);
    //
    //         let standard_landmarks = Array2::from(vec![
    //             [38.2946, 51.6963],
    //             [73.5318, 51.5014],
    //             [56.0252, 71.7366],
    //             [41.5493, 92.3655],
    //             [70.7299, 92.2041],
    //         ]);
    //
    //         let face_alignment = FaceAlignment::new((112, 112), standard_landmarks);
    //         let aligned_img = match face_alignment.call(&image, selected_face_box.clone(), selected_face_point, Some(false)) {
    //             Ok(aligned_img) => {aligned_img}
    //             Err(e) => {
    //                 println!("{:?}", e);
    //                 return
    //             }
    //         };
    //
    //         let model_name = "face_quality_assetment".to_string();
    //
    //         let face_quality_assessment_model_config = match triton_infer_client
    //             .model_config(ModelConfigRequest {
    //                 name: model_name.to_owned(),
    //                 version: "".to_string(),
    //             }).await {
    //             Ok(model_config_resp) => {model_config_resp}
    //             Err(e) => {
    //                 println!("{:?}", e);
    //                 return
    //             }
    //         };
    //
    //
    //         let face_quality_assessment = FaceQualityAssessment::new(
    //             triton_infer_client.clone(),
    //             face_quality_assessment_model_config,
    //             "face_quality_assetment".to_string(),
    //             (112, 112),
    //             1,
    //             55.0).await.unwrap();
    //
    //         let (scores, idxs) = match face_quality_assessment.call(&vec![aligned_img], None).await {
    //             Ok( (scores, idxs)) => {
    //                 (scores, idxs)
    //             }
    //             Err(e) => {
    //                 println!("{:?}", e);
    //                 return
    //             }
    //         };
    //
    //         println!("{:?} {:?}", scores, idxs)
    //     }
    //
    // }
}
