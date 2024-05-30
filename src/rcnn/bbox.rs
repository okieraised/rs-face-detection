// use ndarray::{Array2, ArrayView2};
//
// pub fn bbox_overlaps(
//     boxes: ArrayView2<f32>,
//     query_boxes: ArrayView2<f32>,
// ) -> Array2<f32> {
//     let n = boxes.shape()[0];
//     let k = query_boxes.shape()[0];
//     let mut overlaps = Array2::<f32>::zeros((n, k));
//
//     for k_idx in 0..k {
//         let box_area =
//             (query_boxes[(k_idx, 2)] - query_boxes[(k_idx, 0)] + 1.0)
//                 * (query_boxes[(k_idx, 3)] - query_boxes[(k_idx, 1)] + 1.0);
//
//         for n_idx in 0..n {
//             let iw = (boxes[(n_idx, 2)].min(query_boxes[(k_idx, 2)])
//                 - boxes[(n_idx, 0)].max(query_boxes[(k_idx, 0)])
//                 + 1.0)
//                 .max(0.0);
//
//             if iw > 0.0 {
//                 let ih = (boxes[(n_idx, 3)].min(query_boxes[(k_idx, 3)])
//                     - boxes[(n_idx, 1)].max(query_boxes[(k_idx, 1)])
//                     + 1.0)
//                     .max(0.0);
//
//                 if ih > 0.0 {
//                     let ua = ((
//                         (boxes[(n_idx, 2)] - boxes[(n_idx, 0)] + 1.0)
//                             * (boxes[(n_idx, 3)] - boxes[(n_idx, 1)] + 1.0)
//                             + box_area
//                             - iw * ih,
//                     ))
//                         .max(1.0e-10);
//
//                     overlaps[(n_idx, k_idx)] = iw * ih / ua;
//                 }
//             }
//         }
//     }
//     overlaps
// }
//
// #[cfg(test)]
// mod tests {
//
//     #[test]
//     fn test_bbox_overlaps() {
//     }
// }