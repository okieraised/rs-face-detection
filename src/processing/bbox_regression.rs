// use ndarray::prelude::*;
// use ndarray::{Array, Array2};
// // use rcnn::bbox::{bbox_overlaps};
// use std::collections::HashMap;
// use crate::rcnn;
//
// const BBOX_REGRESSION_THRES: f32 = 0.0;
//
// #[derive(Debug, Clone)]
// struct Config {
//     train: TrainConfig,
// }
//
// #[derive(Debug, Clone)]
// struct TrainConfig {
//     bbox_normalization_precomputed: bool,
//     bbox_means: Vec<f32>,
//     bbox_stds: Vec<f32>,
//     bbox_weights: Array1<f32>,
// }
//
//
//
// #[derive(Debug, Clone)]
// struct RoidbEntry {
//     boxes: Array2<f32>,
//     max_overlaps: Array1<f32>,
//     max_classes: Array1<usize>,
//     bbox_targets: Option<Array2<f32>>,
// }
//
// /// compute_bbox_regression_targets: given rois, overlaps, gt labels, compute bounding box regression targets
// ///
// /// Description.
// ///
// /// * `rois` - roidb[i]['boxes'] k * 4.
// /// * `overlaps` - roidb[i]['max_overlaps'] k * 1
// /// * `labels` - roidb[i]['max_classes'] k * 1
// /// * `overlaps` - Text about bar.
// /// * `return` - targets[i][class, dx, dy, dw, dh] k * 5.
// fn compute_bbox_regression_targets(
//     rois: &Array2<f32>,
//     overlaps: &Array1<f32>,
//     labels: &Array1<usize>,
// ) -> Array2<f32> {
//     let rois = rois.mapv(|x| x as f32);
//
//     if rois.nrows() != overlaps.len() {
//         println!("bbox regression: len(rois) != len(overlaps)");
//     }
//
//     let gt_inds: Vec<usize> = overlaps.iter()
//         .enumerate()
//         .filter(|(_, &overlap)| overlap == 1.0)
//         .map(|(i, _)| i)
//         .collect();
//
//     if gt_inds.is_empty() {
//         println!("bbox regression: len(gt_inds) == 0");
//     }
//
//     let ex_inds: Vec<usize> = overlaps.iter()
//         .enumerate()
//         .filter(|(_, &overlap)| overlap >= BBOX_REGRESSION_THRES)
//         .map(|(i, _)| i)
//         .collect();
//
//     let ex_rois = rois.select(Axis(0), &ex_inds);
//     let gt_rois = rois.select(Axis(0), &gt_inds);
//     let ex_gt_overlaps = bbox_overlaps(<ArrayView2<f32>>::from(&ex_rois), <ArrayView2<f32>>::from(&gt_rois));
//
//     let gt_assignment: Vec<usize> = ex_gt_overlaps.outer_iter()
//         .map(|row| row.argmax().unwrap())
//         .collect();
//
//     let assigned_gt_rois: Array2<f32> = gt_assignment.iter()
//         .map(|&i| gt_rois.row(i))
//         .collect::<Array2<f32>>();
//
//     let targets = Array2::<f32>::zeros((rois.nrows(), 5));
//     let mut targets = targets.clone();
//     for &i in &ex_inds {
//         targets[[i, 0]] = labels[i] as f32;
//         let ex_roi = ex_rois.row(i).to_owned();
//         let gt_roi = assigned_gt_rois.row(i).to_owned();
//         let trans = bbox_transform(&ex_roi, &gt_roi);
//         for j in 0..4 {
//             targets[[i, j + 1]] = trans[j];
//         }
//     }
//
//     targets
// }
//
// fn add_bbox_regression_targets(roidb: &mut Vec<RoidbEntry>, config: &Config) -> (Array1<f32>, Array1<f32>) {
//     println!("bbox regression: add bounding box regression targets");
//     assert!(!roidb.is_empty());
//     assert!(roidb[0].max_classes.len() > 0);
//
//     let num_images = roidb.len();
//     let num_classes = roidb[0].max_classes.len();
//     for im_i in 0..num_images {
//         let rois = &roidb[im_i].boxes;
//         let max_overlaps = &roidb[im_i].max_overlaps;
//         let max_classes = &roidb[im_i].max_classes;
//         roidb[im_i].bbox_targets = Some(compute_bbox_regression_targets(
//             rois, max_overlaps, max_classes,
//         ));
//     }
//
//     let (means, stds) = if config.train.bbox_normalization_precomputed {
//         let means = Array2::from_shape_vec(
//             (num_classes, 4),
//             config.train.bbox_means.clone().repeat(num_classes),
//         ).unwrap();
//         let stds = Array2::from_shape_vec(
//             (num_classes, 4),
//             config.train.bbox_stds.clone().repeat(num_classes),
//         ).unwrap();
//         (means, stds)
//     } else {
//         let mut class_counts = Array2::<f32>::zeros((num_classes, 1)) + 1e-14;
//         let mut sums = Array2::<f32>::zeros((num_classes, 4));
//         let mut squared_sums = Array2::<f32>::zeros((num_classes, 4));
//         for im_i in 0..num_images {
//             if let Some(targets) = &roidb[im_i].bbox_targets {
//                 for cls in 1..num_classes {
//                     let cls_indexes: Vec<usize> = targets
//                         .outer_iter()
//                         .enumerate()
//                         .filter(|(_, target)| target[0] as usize == cls)
//                         .map(|(idx, _)| idx)
//                         .collect();
//                     if !cls_indexes.is_empty() {
//                         class_counts[[cls, 0]] += cls_indexes.len() as f32;
//                         for &idx in &cls_indexes {
//                             sums.row_mut(cls).assign(&(&sums.row(cls) + &targets.slice(s![idx, 1..])));
//                             squared_sums.row_mut(cls).assign(&(&squared_sums.row(cls) + &targets.slice(s![idx, 1..]).mapv(|x| x * x)));
//                         }
//                     }
//                 }
//             }
//         }
//         let means = &sums / &class_counts;
//         let stds = (&squared_sums / &class_counts - &means.mapv(|x| x * x)).mapv(|x| x.sqrt());
//         (means, stds)
//     };
//
//     for im_i in 0..num_images {
//         if let Some(targets) = &mut roidb[im_i].bbox_targets {
//             for cls in 1..num_classes {
//                 let cls_indexes: Vec<usize> = targets
//                     .outer_iter()
//                     .enumerate()
//                     .filter(|(_, target)| target[0] as usize == cls)
//                     .map(|(idx, _)| idx)
//                     .collect();
//                 for &idx in &cls_indexes {
//                     let mut target = targets.slice_mut(s![idx, 1..]);
//                     target -= &means.row(cls);
//                     target /= &stds.row(cls);
//                 }
//             }
//         }
//     }
//
//     (means.into_shape((num_classes * 4,)).unwrap(), stds.into_shape((num_classes * 4,)).unwrap())
// }
//
// /// expand_bbox_regression_targets
// /// This function takes bounding box target data and expands it
// /// such that each class has its own set of bounding box regression targets.
// /// Expand from 5 to 4 * num_classes; only the right class has non-zero bbox regression targets
// /// :param bbox_targets_data: [k * 5]
// /// :param num_classes: number of classes
// /// :return: bbox target processed [k * 4 num_classes]
// /// bbox_weights ! only foreground boxes have bbox regression computation!
// fn expand_bbox_regression_targets(
//     bbox_targets_data: &Array2<f32>,
//     num_classes: usize,
//     bbox_weights: &Array1<f32>,
// ) -> (Array2<f32>, Array2<f32>) {
//     let classes = bbox_targets_data.column(0).to_owned();
//     let k = classes.len();
//     let mut bbox_targets = Array2::<f32>::zeros((k, 4 * num_classes));
//     let mut bbox_weights = Array2::<f32>::zeros((k, 4 * num_classes));
//
//     for (index, &cls) in classes.iter().enumerate() {
//         if cls > 0.0 {
//             let cls = cls as usize;
//             let start = 4 * cls;
//             let end = start + 4;
//             bbox_targets.slice_mut(s![index, start..end])
//                 .assign(&bbox_targets_data.slice(s![index, 1..5]));
//             bbox_weights.slice_mut(s![index, start..end])
//                 .assign(&bbox_weights.view());
//         }
//     }
//
//     (bbox_targets, bbox_weights)
// }
//
// #[cfg(test)]
// mod test {
//     use ndarray::array;
//     use crate::processing::bbox_regression::{compute_bbox_regression_targets, Config, expand_bbox_regression_targets, TrainConfig};
//
//     #[test]
//     fn test_compute_bbox_regression_target() {
//         let rois = array![[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]].into_dyn();
//         let overlaps = array![0.9, 1.0].into_dyn();
//         let labels = array![1, 2].into_dyn();
//         let bbox_regression_thresh = 0.5;
//
//         let targets = compute_bbox_regression_targets(&rois, &overlaps, &labels, bbox_regression_thresh);
//         println!("Targets: {:?}", targets);
//     }
//
//     #[test]
//     fn test_expand_bbox_regression_target() {
//         let bbox_targets_data = array![
//             [1.0, 0.1, 0.2, 0.3, 0.4],
//             [2.0, 0.2, 0.3, 0.4, 0.5],
//             [0.0, 0.0, 0.0, 0.0, 0.0],
//             [1.0, 0.3, 0.4, 0.5, 0.6]
//         ];
//
//         let num_classes = 3;
//         let config = Config {
//             train: TrainConfig {
//                 bbox_normalization_precomputed: false,
//                 bbox_means: vec![],
//                 bbox_stds: vec![],
//                 bbox_weights: array![1.0, 1.0, 1.0, 1.0],
//             },
//         };
//
//         let (bbox_targets, bbox_weights) = expand_bbox_regression_targets(
//             &bbox_targets_data,
//             num_classes,
//             &config.train.bbox_weights,
//         );
//
//         println!("BBox Targets: {:?}", bbox_targets);
//         println!("BBox Weights: {:?}", bbox_weights);
//     }
// }