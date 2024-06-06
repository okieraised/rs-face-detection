extern crate opencv;
extern crate ndarray;
extern crate ndarray_linalg as linalg;

use opencv::core::{self, Mat, Rect, Scalar, Size, BORDER_CONSTANT};
use opencv::imgproc::{warp_affine, resize, INTER_LINEAR};
use opencv::prelude::{*, MatTraitConst};
use opencv::calib3d::{estimate_affine_partial_2d, LMEDS};
use opencv::imgcodecs::*;
use anyhow::{Error, Result};

struct FaceAlignment {
    image_size: (i32, i32),
    standard_landmarks: Vec<[f32; 2]>,
}

impl FaceAlignment {
    /// new creates new FaceAlignment instance.
    pub fn new(image_size: (i32, i32), standard_landmarks: Vec<[f32; 2]>) -> Self {
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
    pub fn call(&self, image: &Mat, bbox: Option<Rect>, landmarks: Vec<[f32; 2]>, debug: Option<bool>) -> Result<Mat, Error> {

        let mut write_output_to_file = false;
        if !debug.is_none() && debug.unwrap() {
            write_output_to_file = true
        }

        let landmarks_mat = match Mat::from_slice_2d(&landmarks) {
            Ok(landmarks_mat) => landmarks_mat,
            Err(e) => return Err(e.into()),
        };

        let standard_landmarks_mat = match Mat::from_slice_2d(&self.standard_landmarks) {
            Ok(standard_landmarks_mat) => standard_landmarks_mat,
            Err(e) => return Err(e.into()),
        };

        let mut transformation_matrix = Mat::default();
        match estimate_affine_partial_2d(
            &landmarks_mat,
            &standard_landmarks_mat,
            &mut core::no_array(),
            LMEDS,
            3.0,
            2000,
            0.99,
            10,
        ) {
            Ok(_) => {},
            Err(e) => return Err(e.into()),

        };

        if transformation_matrix.empty() {
            let det = if let Some(bbox) = bbox {
                bbox
            } else {
                Rect::new(
                    (image.cols() as f32 * 0.0625) as i32,
                    (image.rows() as f32 * 0.0625) as i32,
                    (image.cols() as f32 * (1.0 - 0.125)) as i32,
                    (image.rows() as f32 * (1.0 - 0.125)) as i32,
                )
            };

            let margin = 44;
            let bb = Rect::new(
                (det.x - margin / 2).max(0),
                (det.y - margin / 2).max(0),
                (det.width + margin / 2).min(image.cols()),
                (det.height + margin / 2).min(image.rows()),
            );

            // let ret = Mat::roi(&Mat::clone(image), bb)?.clone_pointee();


            let ret = match Mat::roi(&Mat::clone(image), bb) {
                Ok(ret) => {
                    ret.clone_pointee()
                },
                Err(e) => return Err(e.into()),
            };


            let mut resized_image = Mat::default();

            match resize(
                &ret,
                &mut resized_image,
                Size::new(self.image_size.0, self.image_size.1),
                0.0,
                0.0,
                INTER_LINEAR,
            ){
                Ok(_) => {},
                Err(e) => return Err(e.into()),

            };

            if write_output_to_file{
                match imwrite("./face_align.png", &resized_image, &core::Vector::default()) {
                    Ok(_) => {},
                    Err(e) => return Err(e.into()),

                };
            }

            Ok(resized_image)
        } else {
            let mut aligned_image = Mat::default();

            match warp_affine(
                &image,
                &mut aligned_image,
                &transformation_matrix,
                Size::new(self.image_size.0, self.image_size.1),
                INTER_LINEAR,
                BORDER_CONSTANT,
                Scalar::default())
            {
                Ok(_) => {},
                Err(e) => return Err(e.into()),

            };

            if write_output_to_file{
                match imwrite("./face_align.png", &aligned_image, &core::Vector::default()) {
                    Ok(_) => {},
                    Err(e) => return Err(e.into()),

                };
            }
            Ok(aligned_image)
        }
    }
}



#[cfg(test)]
mod tests {
    use opencv::core::{Rect};
    use crate::pipeline::module::face_align::FaceAlignment;

    #[test]
    fn test_face_align() {
        let standard_landmarks = vec![
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ];

        let face_alignment = FaceAlignment::new((112, 112), standard_landmarks);
        let image = opencv::imgcodecs::imread("/home/tripg/Documents/repo/rs-faceid-pipeline/test_data/anderson.jpg", opencv::imgcodecs::IMREAD_COLOR).unwrap();
        let bbox = Some(Rect::new(0, 0, 100, 100));
        let landmarks: Vec<[f32; 2]> = vec![
            [30.0, 40.0],
            [70.0, 40.0],
            [50.0, 60.0],
            [35.0, 80.0],
            [65.0, 80.0],
        ];

        let result = match face_alignment.call(&image, bbox, landmarks, None)  {
            Ok(result) => {
                println!("Face aligned successfully {:?}", result);
            },
            Err(e) => println!("Face aligned successfully {:?}", e)
        };

    }

}