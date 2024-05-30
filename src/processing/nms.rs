use ndarray::{array, Array1, Array2, Axis, Dim, s};
use ndarray::SliceInfoElem::Slice;
use std::cmp::Ordering;

// greedily select boxes with high confidence and overlap with current maximum <= thresh
//rule out overlap >= thresh
//   :param dets: [[x1, y1, x2, y2 score]]
//    :param thresh: retain overlap < thresh
//    :return: indexes to keep
// fn nms(dets: Array2<f32>, thresh: f32) -> Vec<usize> {
//     let x1 = dets.slice(s![.., 0]);
//     let y1 = dets.slice(s![.., 1]);
//     let x2 = dets.slice(s![.., 2]);
//     let y2 = dets.slice(s![.., 3]);
//     let scores = dets.slice(s![.., 4]);
//
//     let areas: Array1<f32> = (&x2 - &x1 + 1.0) * (&y2 - &y1 + 1.0);
//     let mut order: Vec<usize> = (0..scores.len()).collect();
//     order.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
//     let mut keep = Vec::new();
//
//     while !order.is_empty() {
//         let i = order[0];
//         keep.push(i);
//
//         let xx1 = x1[i].max(x1.select(s![order[1..].to_vec()], &[]));
//         let yy1 = y1[i].max(y1.select(s![order[1..].to_vec()], &[]));
//         let xx2 = x2[i].min(x2.select(s![order[1..].to_vec()], &[]));
//         let yy2 = y2[i].min(y2.select(s![order[1..].to_vec()], &[]));
//
//         let w = (xx2 - &xx1 + 1.0).mapv(|v| v.max(0.0));
//         let h = (yy2 - &yy1 + 1.0).mapv(|v| v.max(0.0));
//         let inter = &w * &h;
//         let ovr = &inter / (&areas[i] + areas.select(s![order[1..].to_vec()], &[]) - &inter);
//
//         let mut inds: Vec<usize> = Vec::new();
//         for (idx, &ovr_value) in ovr.iter().enumerate() {
//             if ovr_value <= thresh {
//                 inds.push(idx + 1);
//             }
//         }
//
//         order = inds.iter().map(|&x| order[x]).collect();
//     }
//
//     keep
// }



// fn nms(dets: Array2<f32>, thresh: f32) -> Vec<f32> {
//     let x1 = dets.slice(s![.., 0]);
//     let y1 = dets.slice(s![.., 1]);
//     let x2 = dets.slice(s![.., 2]);
//     let y2 = dets.slice(s![.., 3]);
//     let scores = dets.slice(s![.., 4]);
//
//     let areas: Array1<f32> = (&x2 - &x1 + 1.0) * (&y2 - &y1 + 1.0);
//     let mut order: Array1<usize> = (0..scores.len()).collect();
//     order.into_raw_vec().sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));
//
//     let mut keep = Vec::new();
//
//     while !order.is_empty() {
//         let i = order[0];
//         keep.push(i);
//
//         let xx1 = x1[i].max(&x1.select(Axis(0), &order[1..]));
//         let yy1 = y1[i].max(&y1.select(Axis(0), &order[1..]));
//         let xx2 = x2[i].min(&x2.select(Axis(0), &order[1..]));
//         let yy2 = y2[i].min(&y2.select(Axis(0), &order[1..]));
//
//         let w = (&xx2 - &xx1 + 1.0).mapv(|v| v.max(0.0));
//         let h = (&yy2 - &yy1 + 1.0).mapv(|v| v.max(0.0));
//         let inter = &w * &h;
//         let ovr = &inter / (&areas[i] + &areas.select(Axis(0), &order[1..]) - &inter);
//
//         let mut inds = Vec::new();
//         for (idx, &ovr_value) in ovr.iter().enumerate() {
//             if ovr_value <= thresh {
//                 inds.push(idx + 1);
//             }
//         }
//
//         order = inds.iter().map(|&x| order[x]).collect();
//     }
//
//     keep
// }
//
//
// #[cfg(test)]
// mod tests {
//     use ndarray::array;
//     use crate::processing::nms::nms;
//
//     #[test]
//     fn test_nms() {
//         let dets = array![
//             [100.0, 100.0, 210.0, 210.0, 0.72],
//             [250.0, 250.0, 420.0, 420.0, 0.8],
//             [220.0, 220.0, 320.0, 330.0, 0.92],
//             [100.0, 100.0, 210.0, 210.0, 0.6]
//         ];
//
//         let thresh = 0.4;
//         let keep = nms(dets, thresh);
//
//         println!("Kept indices: {:?}", keep);
//     }
//
// }