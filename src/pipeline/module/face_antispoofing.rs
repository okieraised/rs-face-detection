use anyhow::Error;
use ndarray::{Array1, Array3, Array4, s};
use opencv::core::{Mat, MatTraitConst, Rect, Size};
use opencv::imgproc::{COLOR_RGB2BGR, cvt_color, INTER_LINEAR, resize};
use crate::triton_client::client::triton::ModelConfigResponse;
use crate::triton_client::client::TritonInferenceClient;

#[derive(Debug)]
pub(crate) struct FaceAntiSpoofing {
    triton_infer_client: TritonInferenceClient,
    triton_model_config: ModelConfigResponse,
    model_name: Vec<String>,
    scales: Vec<f32>,
    image_sizes: Vec<(i32, i32)>,
    threshold: f32,
    batch_size: i32,
}

struct CropParams {
    org_img: Mat,
    bbox: Rect,
    scale: f32,
    out_w: i32,
    out_h: i32,
    crop: bool,
}



impl FaceAntiSpoofing {
    pub async fn new(
        triton_infer_client: TritonInferenceClient,
        triton_model_config: ModelConfigResponse,
        model_name: Vec<String>,
        image_sizes: Vec<(i32, i32)>,
        scales: Vec<f32>,
        batch_size: i32,
        threshold: f32,
    ) -> Result<Self, Error> {
        Ok(FaceAntiSpoofing {
            triton_infer_client,
            triton_model_config,
            model_name,
            scales,
            image_sizes,
            threshold,
            batch_size,
        })
    }

    async fn call(&self, imgs: Vec<&Mat>, face_boxes: Vec<&Array1<f32>>, is_debug: Option<bool>) {

        let debug = is_debug.unwrap_or(false);

        let mut list_image_scales: &mut Vec<Vec<Mat>> = &mut Vec::with_capacity(face_boxes.len());
        let mut list_weight_scales: &mut Vec<Vec<f32>> = &mut Vec::with_capacity(face_boxes.len());


        for (image, face_box) in imgs.into_iter().zip(face_boxes) {
            let mut converted_image = Mat::default();
            cvt_color(&image, &mut converted_image, COLOR_RGB2BGR, 0).unwrap();
            let (tmps, weights) = self._get_scale_image(image, face_box).unwrap();
            for (i, _) in self.scales.clone().into_iter().enumerate() {
                list_image_scales.insert(i, vec![tmps.clone()[i].clone()]).to_owned();
                list_weight_scales.insert(i, vec![weights.clone()[i]]).to_owned();
            }
        }

        for (i, _) in self.scales.iter().enumerate() {
            let preprocessed_images = self._preprocess(&list_image_scales[i], i).unwrap();

            for i in (0..preprocessed_images.shape()[0]).step_by(self.batch_size as usize) {
                let x = preprocessed_images.slice(s![i..i + self.batch_size as usize, .., .., ..]);
                println!("x: {:?}", x);
            }


        }
    }

    fn _preprocess(&self, images: &Vec<Mat>, index: usize) -> Result<Array4<f32>, Error>{
        let batch_input_size = (
            f32::max((images.len() as f32 / self.batch_size as f32).ceil(), 1.0) * self.batch_size as f32
            ) as i32;

        let mut preprocessed_images = Array4::<f32>::zeros((batch_input_size as usize, 3, self.image_sizes[index].1 as usize, self.image_sizes[index].0 as usize));

        for (i, img) in images.into_iter().enumerate() {
            let mut img_brg = Mat::default();

            match cvt_color(&img, &mut img_brg, COLOR_RGB2BGR, 0) {
                Ok(_) => {}
                Err(e) => return Err(Error::from(e))
            };

            let im_info = match img_brg.size() {
                Ok(im_info) => {im_info}
                Err(e) => return Err(Error::from(e))
            };
            let rows = im_info.height;
            let cols = im_info.width;

            let mut im_tensor = Array3::<f32>::zeros((rows as usize, cols as usize, 3));
            for i in 0..3 {
                for y in 0..rows {
                    for x in 0..cols {
                        let pixel_value = img_brg.at_2d::<opencv::core::Vec3b>(y, x).unwrap()[i];
                        im_tensor[[y as usize, x as usize, 2-i]] = pixel_value as f32;
                    }
                }
            }

            let transposed_tensors = im_tensor.permuted_axes([2, 0, 1]);
            preprocessed_images.slice_mut(s![i,..,..,..]).assign(&transposed_tensors);
        }

        Ok(preprocessed_images)
    }

    fn _get_scale_image(&self, image: &Mat, face_box: &Array1<f32>) -> Result<(Vec<Mat>, Vec<f32>), opencv::Error> {

        let det_xmin = face_box[0];
        let det_ymin = face_box[1];
        let det_xmax = face_box[2];
        let det_ymax = face_box[3];
        let det_height = det_ymax - det_ymin;
        let c_x = (det_xmin + det_xmax) / 2.0;


        let left = (c_x - 0.47 * det_height) as i32;
        let right = (c_x + 0.47 * det_height) as i32;

        let top = det_ymin;
        let bottom = det_ymax;
        let bbox = Rect::new(left as i32, top as i32, (right - left + 1) as i32, (bottom - top + 1.0) as i32);

        let mut crops = Vec::new();
        let mut weights = Vec::new();

        for (i, scale) in self.scales.clone().into_iter().enumerate() {

            let out_w = self.image_sizes[i].0;
            let out_h = self.image_sizes[i].1;

            let param = CropParams {
                org_img: image.clone(),
                bbox,
                scale,
                out_w,
                out_h,
                crop: true,
            };

            let (crop, weight) = self._crop_image(param).unwrap();
            crops.push(crop);
            weights.push(weight);
        }

        Ok((crops, weights))
    }


    fn _crop_image(&self, param: CropParams) -> Result<(Mat, f32), Error> {

        if !param.crop {
            let mut dst_img = Mat::default();

            let _ = match resize(&param.org_img, &mut dst_img, Size::new(param.out_w, param.out_h), 0.0, 0.0, INTER_LINEAR) {
                Ok(_) => {}
                Err(e) => {
                    return Err(Error::from(e))
                }
            };
            return Ok((dst_img, 1.0));
        }

        let src_h = param.org_img.rows();
        let src_w = param.org_img.cols();

        let (left_top_x, left_top_y, right_bottom_x, right_bottom_y, weight) = self._get_new_box(src_w, src_h, param.bbox, param.scale);

        println!("{:?} {:?} {:?} {:?} {:?}", left_top_x, left_top_y, right_bottom_x, right_bottom_y, weight);

        let roi = Rect::new(left_top_x, left_top_y, right_bottom_x - left_top_x + 1, right_bottom_y - left_top_y + 1);
        let img = match Mat::roi(&param.org_img, roi) {
            Ok(img) => {img}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        let mut dst_img = Mat::default();
        match resize(&img, &mut dst_img, Size::new(param.out_w, param.out_h), 0.0, 0.0, INTER_LINEAR) {
            Ok(_) => {}
            Err(e) => {
                return Err(Error::from(e))
            }
        };

        Ok((dst_img, weight))
    }

    fn _get_new_box(&self, src_w: i32, src_h: i32, bbox: Rect, scale_ori: f32) -> (i32, i32, i32, i32, f32) {
        let x = bbox.x;
        let y = bbox.y;
        let box_w = bbox.width;
        let box_h = bbox.height;
        let scale = f32::min((src_h as f32 - 1.0) / box_h as f32, f32::min((src_w as f32 - 1.0) / box_w as f32, scale_ori));

        let new_width = box_w as f32 * scale;
        let new_height = box_h as f32 * scale;
        let (center_x, center_y) = (box_w as f32 / 2.0 + x as f32, box_h as f32 / 2.0 + y as f32);

        let mut left_top_x = center_x - new_width / 2.0;
        let mut left_top_y = center_y - new_height / 2.0;
        let mut right_bottom_x = center_x + new_width / 2.0;
        let mut right_bottom_y = center_y + new_height / 2.0;

        if left_top_x < 0.0 {
            right_bottom_x -= left_top_x;
            left_top_x = 0.0;
        }

        if left_top_y < 0.0 {
            right_bottom_y -= left_top_y;
            left_top_y = 0.0;
        }

        if right_bottom_x > src_w as f32 - 1.0 {
            left_top_x -= right_bottom_x - src_w as f32 + 1.0;
            right_bottom_x = src_w as f32 - 1.0;
        }

        if right_bottom_y > src_h as f32 - 1.0 {
            left_top_y -= right_bottom_y - src_h as f32 + 1.0;
            right_bottom_y = src_h as f32 - 1.0;
        }

        (
            left_top_x as i32,
            left_top_y as i32,
            right_bottom_x as i32,
            right_bottom_y as i32,
            scale / scale_ori,
        )
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Error;
    use ndarray::{Array1, s};
    use crate::pipeline::module::face_antispoofing::FaceAntiSpoofing;
    use crate::pipeline::module::face_detection::RetinaFaceDetection;
    use crate::pipeline::module::face_selection::FaceSelection;
    use crate::triton_client::client::triton::ModelConfigRequest;
    use crate::triton_client::client::TritonInferenceClient;
    use crate::utils::utils::byte_data_to_opencv;

    #[tokio::test]
    async fn test_face_antispoofing() {
        let triton_host = "";
        let triton_port = "";
        let im_bytes: &[u8] = include_bytes!("");
        let image = byte_data_to_opencv(im_bytes).unwrap();

        let triton_infer_client = match TritonInferenceClient::new(triton_host, triton_port).await {
            Ok(triton_infer_client) => triton_infer_client,
            Err(e) => {
                println!("{:?}", e);
                return
            }
        };

        let model_name = "face_detection_retina".to_string();

        let face_detection_model_config = match triton_infer_client
            .model_config(ModelConfigRequest {
                name: model_name.to_owned(),
                version: "".to_string(),
            }).await {
            Ok(model_config_resp) => {model_config_resp}
            Err(e) => {
                println!("{:?}", e);
                return
            }
        };

        let retina_face_detection = match RetinaFaceDetection::new(
            triton_infer_client.clone(),
            face_detection_model_config,
            model_name,
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
        let (preprocessed_img, scale) = retina_face_detection.call(&image, Some(false)).await.unwrap();
        let face_selection = FaceSelection::new(0.3, 0.3, 0.1, 0.0075).await;
        let (selected_face_box, selected_face_point) = face_selection.call(&image, preprocessed_img, scale, Some(false), None).unwrap();



        let face_antispoofing = match FaceAntiSpoofing::new(
            triton_infer_client.clone(),
            Default::default(),
            vec![],
            vec![
                (80, 80),
                (80, 80),
                (256, 256),
                (128, 128),
            ],
            vec![4.0, 2.7, 2.0, 1.0],
            1,
            0.55).await {
            Ok(face_antispoofing) => {face_antispoofing}
            Err(e) => {
                println!("{:?}", e);
                return
            }
        };

        if let Some(_selected_face_box) = selected_face_box {

            let sliced_boxes = _selected_face_box.slice(s![..4]);
            let boxes: Array1<f32> = sliced_boxes.to_owned();

            println!("boxes: {:?}", boxes);

            face_antispoofing.call(vec![&image],vec![&boxes], None).await;
        }

    }
}
