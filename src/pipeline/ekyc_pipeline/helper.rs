// use std::ops::Mul;
// use ndarray::{Array1, Array2};
//
// #[derive(Debug)]
// pub struct FaceHelper {
//     face_size: (i32, i32),
//     face_template: Array2<f32>,
//     fas_size: (i32, i32),
//     // fas_template: Array2<f32>,
//     fa_size: (i32, i32),
//     // fa_template: Array2<f32>,
// }
//
// impl FaceHelper {
//
//     pub fn new(in_face_size: Option<(i32, i32)>, in_face_template: Option<Array2<f32>>, in_fas_size: Option<(i32, i32)>, in_fas_template: Option<Array2<f32>>, in_fa_size: Option<(i32, i32)>, in_fa_template: Option<Array2<f32>>) -> Self {
//
//         let mut face_size: (i32, i32) = (112, 112);
//         let mut fas_size: (i32, i32) = (224, 224);
//         let mut fa_size: (i32, i32) = (128/4*3, 128);
//
//
//         if let Some(_face_size) = in_face_size {
//             face_size = _face_size;
//         };
//
//         if let Some(_fas_size) = in_fas_size {
//             fas_size = _fas_size;
//         };
//
//         if let Some(_fa_size) = in_fa_size {
//             fa_size.0 = _fa_size.0/4*3;
//             fa_size.1 = _fa_size.1;
//         };
//
//         let mut face_template: Array2<f32> = Array2::from(vec![
//             [38.2946, 51.6963],
//             [73.5318, 51.5014],
//             [56.0252, 71.7366],
//             [41.5493, 92.3655],
//             [70.7299, 92.2041],
//         ]);
//
//         // let vector: Array1<f32> =Array1::from_vec(vec![face_size.0 as f32 /112.0, face_size.1 as f32 / 112.0]);
//         // let x_face_template = face_template.dot(&vector); //.into_shape((2, 5)).unwrap();
//
//
//         if let Some(_face_template) = in_face_template {
//             face_template = _face_template;
//         }
//
//         println!("{:?}", face_template);
//
//         FaceHelper {
//             face_size,
//             face_template,
//             fas_size,
//             fa_size,
//         }
//     }
// }
//
//
// #[cfg(test)]
// mod tests {
//     use crate::pipeline::ekyc_pipeline::helper::FaceHelper;
//
//     #[test]
//     fn test_new_face_helper() {
//         FaceHelper::new(None, None,None, None,None,None);
//     }
//
// }