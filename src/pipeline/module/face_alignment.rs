extern crate opencv;
extern crate ndarray;
extern crate ndarray_linalg as linalg;

use crate::utils::utils::array2_to_mat;
use opencv::core::{self, Mat, Rect, Scalar, Size, BORDER_CONSTANT, ToInputOutputArray, Range};
use opencv::imgproc::{warp_affine, resize, INTER_LINEAR};
use opencv::prelude::{*, MatTraitConst};
use opencv::calib3d::{estimate_affine_partial_2d, LMEDS};
use opencv::imgcodecs::*;
use anyhow::{Error, Result};
use ndarray::{Array1, Array2};

struct FaceAlignment {
    image_size: (i32, i32),
    standard_landmarks: Array2<f32>,
}

impl FaceAlignment {
    /// new creates new FaceAlignment instance.
    pub fn new(image_size: (i32, i32), standard_landmarks: Array2<f32>) -> Self {
        FaceAlignment {
            image_size,
            standard_landmarks,
        }
    }

    /// call aligns the face image
    /// Converts landmarks and standard_landmarks to Mat using Mat::from_slice_2d.
    /// Uses estimate_affine_partial_2d to estimate the affine transformation matrix.
    /// If the transformation matrix is None, it uses the bounding box or center crop logic.
    /// Uses warp_affine to apply the affine transformation to align the face.
    pub fn _process(&self, img: &Mat, bbox: Option<Array1<f32>>, landmarks: Option<Array2<f32>>, is_debug: Option<bool>) -> Result<Mat, Error> {

        let debug = is_debug.unwrap_or(false);

        let img_shape = match img.size() {
            Ok(img_shape) => img_shape,
            Err(e) => return Err(Error::from(e))
        };

        let standard_landmarks_mat = match array2_to_mat(&self.standard_landmarks) {
            Ok(standard_landmarks_mat) => {standard_landmarks_mat}
            Err(e) => return Err(e.into()),
        };
        let mut landmarks_mat = Mat::default();
        if let Some(_landmarks) = landmarks {
            landmarks_mat = match array2_to_mat(&_landmarks) {
                Ok(landmarks_mat) => {landmarks_mat}
                Err(e) => return Err(Error::from(e))
            };
        }

        let mut inliers = Mat::default();

        let transformation_matrix = match estimate_affine_partial_2d(
            &landmarks_mat,
            &standard_landmarks_mat,
            &mut inliers,
            LMEDS,
            3.0,
            2000,
            0.99,
            10,
        ) {
            Ok(transformation_matrix) => {transformation_matrix},
            Err(e) => return Err(e.into()),
        };

        if debug {
            println!("transformation_matrix: {:?}", transformation_matrix);
        }

        if transformation_matrix.empty() {
            let mut det: Array1<f32> = Array1::zeros(4);
            if bbox.is_none() {
                det = Array1::zeros(4);
                det[0] = img_shape.width.to_owned() as f32 * 0.0625;
                det[1] = img_shape.height as f32 * 0.0625;
                det[2] = img_shape.width as f32 - det[0];
                det[3] = img_shape.height as f32 - det[1];
            } else {
                det = bbox.unwrap();
            }

            let margin: f32 = 44.0;
            let mut bb: Array1<f32> = Array1::zeros(4);
            bb[0] = f32::max(det[0] - margin / 2.0, 0.0);
            bb[1] = f32::max(det[1] - margin / 2.0, 0.0);
            bb[2] = f32::max(det[2] + margin / 2.0, img_shape.width as f32);
            bb[3] = f32::max(det[1] + margin / 2.0, img_shape.height as f32);

            let x0 = bb[0] as i32;
            let y0 = bb[1] as i32;
            let x1 = bb[2] as i32;
            let y1 = bb[3] as i32;
            let width = x1 - x0;
            let height = y1 - y0;

            let rect = Rect::new(x0, y0, width, height);
            let mut cropped_img = img.clone();
            let roi = match Mat::roi_mut(&mut cropped_img, rect) {
                Ok(roi) => {roi}
                Err(e) => return Err(e.into()),
            };

            let mut resized_image = Mat::default();
            match resize(
                &roi,
                &mut resized_image,
                Size::new(self.image_size.0, self.image_size.1),
                0.0,
                0.0,
                INTER_LINEAR,
            ){
                Ok(_) => {
                    if debug {
                        match imwrite("./aligned.jpg", &resized_image, &Default::default()) {
                            Ok(_) => {}
                            Err(e) => return Err(e.into()),
                        };
                    }
                    Ok(resized_image)
                },
                Err(e) => return Err(e.into()),
            }
        } else {
            let mut aligned_image = Mat::default();
            match warp_affine(
                &img,
                &mut aligned_image,
                &transformation_matrix,
                Size::new(self.image_size.0, self.image_size.1),
                INTER_LINEAR,
                BORDER_CONSTANT,
                Scalar::default())
            {
                Ok(_) => {
                    if debug {
                        match imwrite("./aligned.jpg", &aligned_image, &Default::default()) {
                            Ok(_) => {}
                            Err(e) => return Err(e.into()),
                        };
                    }
                    Ok(aligned_image)

                },
                Err(e) => return Err(e.into()),
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{Array2};
    use crate::pipeline::module::face_alignment::FaceAlignment;
    use crate::pipeline::module::face_detection::RetinaFaceDetection;
    use crate::pipeline::module::face_selection::FaceSelection;
    use crate::utils::utils::byte_data_to_opencv;

    #[tokio::test]
    async fn test_face_alignment() {
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

        let (preprocessed_img, scale) = retina_face_detection._preprocess(&image, Some(false)).unwrap();

        let predicted_output = retina_face_detection._forward(&preprocessed_img, None).await.unwrap();

        let (detections, key_points) = retina_face_detection._postprocess(predicted_output, scale, None).await;

        let face_selection = FaceSelection::new(0.3, 0.3, 0.1, 0.0075).await;

        let face_selection_result = match face_selection._process(&image, detections, key_points, Some(false), None) {
            Ok(selection) => {selection}
            Err(e) => {
                println!("{:?}", e);
                return
            }
        };

        let (selected_face_box, selected_face_point) = face_selection_result;

        let standard_landmarks = Array2::from(vec![
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ]);

        let face_alignment = FaceAlignment::new((112, 112), standard_landmarks);
        face_alignment._process(&image, selected_face_box, selected_face_point, Some(true));
    }
}