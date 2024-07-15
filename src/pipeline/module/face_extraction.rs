use std::cmp;
use std::future::Future;
use anyhow::Error;
use ndarray::{Array2, Array3, Array4, IntoDimension, s};
use opencv::core::{Mat, MatTraitConst, Size};
use opencv::imgproc::{COLOR_BGR2RGB, cvt_color, INTER_LINEAR, resize};
use crate::triton_client::client::triton::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use crate::triton_client::client::triton::{InferTensorContents, ModelConfigRequest, ModelConfigResponse, ModelInferRequest};
use crate::triton_client::client::TritonInferenceClient;
use crate::utils::utils::{normalize_outputs, u8_to_f32_vec};

#[derive(Debug)]
pub struct FaceExtraction {
    triton_infer_client: TritonInferenceClient,
    triton_model_config: ModelConfigResponse,
    model_name: String,
    image_size: (i32, i32),
    batch_size: i32
}

impl FaceExtraction {
    pub async fn new(
        triton_infer_client: TritonInferenceClient,
        triton_model_config: ModelConfigResponse,
        model_name: String,
        image_size: (i32, i32),
        batch_size: i32,
    ) -> Result<Self, Error> {
            Ok(FaceExtraction {
                triton_infer_client,
                triton_model_config,
                model_name,
                image_size,
                batch_size,
            })
    }

    fn _preprocess(&self, imgs: &[Mat], is_debug: Option<bool>) -> Result<Array4<f32>, Error> {
        let num_batches = cmp::max((imgs.len() as f32 / self.batch_size as f32).ceil() as usize, 1);
        let mut preprocessed_images = Array4::<f32>::zeros((num_batches, 3usize, self.image_size.1 as usize, self.image_size.0 as usize));

        for (idx, output) in imgs.iter().enumerate() {
            let mut resized_img = Mat::default();
            match resize(&imgs[idx], &mut resized_img, Size::new(self.image_size.0, self.image_size.1), 0.0, 0.0, INTER_LINEAR) {
                Ok(_) => {},
                Err(e) => return Err(Error::from(e))
            }

            let mut converted_img = Mat::default();
            let _ = match cvt_color(&resized_img, &mut converted_img, COLOR_BGR2RGB, 0) {
                Ok(_) => {}
                Err(e) => return Err(Error::from(e))
            };

            let im_info = match converted_img.size() {
                Ok(im_info) => {im_info}
                Err(e) => return Err(Error::from(e))
            };

            let rows = im_info.height;
            let cols = im_info.width;


            let mut im_tensor = Array3::<f32>::zeros((rows as usize, cols as usize, 3));
            for i in 0..3 {
                for y in 0..rows {
                    for x in 0..cols {
                        let pixel_value = converted_img.at_2d::<opencv::core::Vec3b>(y, x).unwrap()[i];
                        im_tensor[[y as usize, x as usize, i]] = (pixel_value as f32 - 127.5) * 0.0078125;
                    }
                }
            }
            let transposed_tensors = im_tensor.permuted_axes([2, 0, 1]);
            preprocessed_images.index_axis_mut(ndarray::Axis(0), idx).assign(&transposed_tensors);
        }
        Ok(preprocessed_images)
    }

    pub async fn call(&self, imgs: &[Mat], is_debug: Option<bool>) -> Result<Vec<Array2<f32>>, Error> {
        let debug = is_debug.unwrap_or(false);

        let preprocessed_images = match self._preprocess(imgs, is_debug) {
            Ok(preprocessed_images) => {preprocessed_images}
            Err(e) => return Err(Error::from(e))
        };


        let model_cfg = match &self.triton_model_config.config {
            None => {
                return Err(Error::msg("face_extraction - face extraction model config is empty"))
            }
            Some(model_cfg) => {model_cfg}
        };


        let mut outputs: Vec<Vec<Array2<f32>>> = vec![];

        for i in (0..preprocessed_images.dim().0).step_by(self.batch_size as usize) {
            let batch = preprocessed_images.slice(s![i..i + self.batch_size as usize, .., .., ..]);
            let mut im_tensor: Vec<Array4<f32>> = vec![];
            im_tensor.push(batch.to_owned());
            let mut flattened_vec: Vec<f32> = Vec::new();
            for array in im_tensor {
                flattened_vec.extend(array.iter());
            }

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

            outputs.push(net_out);

        }

        let normalized_outputs = normalize_outputs(outputs);
        Ok(normalized_outputs)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2};
    use crate::pipeline::module::face_alignment::FaceAlignment;
    use crate::pipeline::module::face_detection::RetinaFaceDetection;
    use crate::pipeline::module::face_extraction::FaceExtraction;
    use crate::pipeline::module::face_quality::FaceQuality;
    use crate::pipeline::module::face_selection::FaceSelection;
    use crate::triton_client::client::triton::ModelConfigRequest;
    use crate::triton_client::client::TritonInferenceClient;
    use crate::utils::utils::byte_data_to_opencv;

    #[tokio::test]
    async fn test_face_extraction() {
        // let triton_host = "";
        // let triton_port = "";
        // let im_bytes: &[u8] = include_bytes!("");
        // let image = byte_data_to_opencv(im_bytes).unwrap();
        //
        // let triton_infer_client = match TritonInferenceClient::new(triton_host, triton_port).await {
        //     Ok(triton_infer_client) => triton_infer_client,
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // let model_name = "face_detection_retina".to_string();
        //
        // let face_detection_model_config = match triton_infer_client
        //     .model_config(ModelConfigRequest {
        //         name: model_name.to_owned(),
        //         version: "".to_string(),
        //     }).await {
        //     Ok(model_config_resp) => {model_config_resp}
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // // Query face quality model config
        // let face_quality_model_config = match triton_infer_client
        //     .model_config(ModelConfigRequest {
        //         name: "face_quality".to_string(),
        //         version: "".to_string(),
        //     }).await {
        //     Ok(model_config_resp) => {model_config_resp}
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // // Query face extraction model config
        // let face_extraction_model_config = match triton_infer_client
        //     .model_config(ModelConfigRequest {
        //         name: "face_identification".to_string(),
        //         version: "".to_string(),
        //     }).await {
        //     Ok(model_config_resp) => {model_config_resp}
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // let retina_face_detection = match RetinaFaceDetection::new(
        //     triton_infer_client.clone(),
        //     face_detection_model_config,
        //     model_name,
        //     (640, 640),
        //     1,
        //     0.7,
        //     0.45,
        // ).await
        // {
        //     Ok(retina_face_detection)  => retina_face_detection,
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // let (detections, key_points) = retina_face_detection.call(&image, Some(true)).await.unwrap();
        //
        // let face_selection = FaceSelection::new(0.3, 0.3, 0.1, 0.0075).await;
        // let (selected_face_box, selected_face_point)= match face_selection.call(&image, detections, key_points, Some(false), None) {
        //     Ok((selected_face_box, selected_face_point)) => {(selected_face_box, selected_face_point)}
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // let standard_landmarks = Array2::from(vec![
        //     [38.2946, 51.6963],
        //     [73.5318, 51.5014],
        //     [56.0252, 71.7366],
        //     [41.5493, 92.3655],
        //     [70.7299, 92.2041],
        // ]);
        //
        // let face_alignment = FaceAlignment::new((112, 112), standard_landmarks);
        // let aligned_face_image= match face_alignment.call(&image, selected_face_box, selected_face_point, Some(true)) {
        //     Ok(aligned_face_image) => {aligned_face_image}
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // let face_quality = match FaceQuality::new(
        //     triton_infer_client.clone(),
        //     face_quality_model_config,
        //     "face_quality".to_string(),
        //     (112, 112),
        //     0.5,
        // ).await
        // {
        //     Ok(face_quality)  => face_quality,
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // let aligned_img_arr = &[aligned_face_image];
        //
        // let (quality_score, quality_class) = match face_quality.call(aligned_img_arr, Some(true)).await {
        //     Ok((quality_score, quality_class)) => {(quality_score, quality_class)}
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // let face_extraction = match FaceExtraction::new(
        //     triton_infer_client.clone(),
        //     face_extraction_model_config,
        //     "face_identification".to_string(),
        //     (112, 112),
        //     1,
        // ).await {
        //     Ok(face_extraction) => {face_extraction}
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // let facial_feature = match face_extraction.call(aligned_img_arr, Some(true)).await {
        //     Ok(facial_feature) => {facial_feature}
        //     Err(e) => {
        //         println!("{:?}", e);
        //         return
        //     }
        // };
        //
        // println!("{:?}", facial_feature);

    }
}