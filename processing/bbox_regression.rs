use ndarray::prelude::*;
use ndarray::{Array1, Array2, s, ArrayView2};
use rcnn::bbox::{bbox_overlaps};
use std::collections::HashMap;
use crate::processing::bbox_transform::nonlinear_transform;
use crate::rcnn;

const BBOX_REGRESSION_THRES: f32 = 0.0;

#[derive(Debug, Clone)]
struct Config {
    train: TrainConfig,
}

#[derive(Debug, Clone)]
struct TrainConfig {
    bbox_normalization_precomputed: bool,
    bbox_means: Vec<f32>,
    bbox_stds: Vec<f32>,
    bbox_weights: Array1<f32>,
}



#[derive(Debug, Clone)]
struct RoidbEntry {
    boxes: Array2<f32>,
    max_overlaps: Array1<f32>,
    max_classes: Array1<usize>,
    bbox_targets: Option<Array2<f32>>,
}


// fn argmax(array: &[f32]) -> usize {
//     if array.is_empty() {
//         panic!("Cannot find argmax of an empty array");
//     }
//
//     let mut max_index = 0;
//     let mut max_value = array[0];
//
//     for (i, &value) in array.iter().enumerate().skip(1) {
//         if value > max_value {
//             max_value = value;
//             max_index = i;
//         }
//     }
//
//     max_index
// }
//
// fn compute_bbox_regression_targets(
//     rois: &Array2<f32>,
//     overlaps: &Array1<f32>,
//     labels: &Array1<f32>,
//     bbox_regression_thresh: f32
// ) -> Array2<f32> {
//     // Ensure ROIs are floats (already ensured by type)
//
//     // Sanity check
//     if rois.nrows() != overlaps.len() {
//         eprintln!("bbox regression: len(rois) != len(overlaps)");
//     }
//
//     // Indices of ground-truth ROIs
//     let gt_inds: Vec<usize> = overlaps.iter()
//         .enumerate()
//         .filter(|&(_, &overlap)| overlap == 1.0)
//         .map(|(i, _)| i)
//         .collect();
//     if gt_inds.is_empty() {
//         eprintln!("bbox regression: len(gt_inds) == 0");
//     }
//
//     // Indices of examples for which we try to make predictions
//     let ex_inds: Vec<usize> = overlaps.iter()
//         .enumerate()
//         .filter(|&(_, &overlap)| overlap >= bbox_regression_thresh)
//         .map(|(i, _)| i)
//         .collect();
//
//     // Get IoU overlap between each ex ROI and gt ROI
//     let ex_rois = rois.select(Axis(0), &ex_inds);
//     let gt_rois = rois.select(Axis(0), &gt_inds);
//     let ex_gt_overlaps = bbox_overlaps(&ex_rois, &gt_rois);
//
//     // Find which gt ROI each ex ROI has max overlap with
//     let gt_assignment: Vec<usize> = ex_gt_overlaps.axis_iter(Axis(1))
//         .map(|row| argmax(row.as_slice().unwrap()))
//         .collect();
//
//     let gt_rois_assigned = gt_assignment.iter()
//         .map(|&idx| gt_rois.slice(s![idx, ..]))
//         .collect::<Vec<_>>();
//
//     // Prepare target array
//     let mut targets = Array2::<f32>::zeros((rois.nrows(), 5));
//
//     // Set the labels
//     for &ex_idx in &ex_inds {
//         targets[[ex_idx, 0]] = labels[ex_idx];
//     }
//
//     // Set the bbox regression targets
//     for (i, &ex_idx) in ex_inds.iter().enumerate() {
//         let ex_roi = rois.slice(s![ex_idx, ..]).to_owned();
//         let gt_roi = gt_rois_assigned[i].to_owned();
//         let ex_roi_2d = ex_roi.clone().into_shape((4, 1)).unwrap();
//         let gt_roi_2d = gt_roi.clone().into_shape((4, 1)).unwrap();
//         let bbox_trans = nonlinear_transform(&ex_roi_2d, &gt_roi_2d);
//         targets.slice_mut(s![ex_idx, 1..]).assign(&bbox_trans);
//     }
//
//     targets
// }
//
// #[cfg(test)]
// mod test {
//     use ndarray::array;
//     use crate::processing::bbox_regression::{compute_bbox_regression_targets};
//
//     #[test]
//     fn test_compute_bbox_regression_target() {
//         let rois = array![
//             [10.0, 20.0, 50.0, 60.0],
//             [15.0, 25.0, 55.0, 65.0]
//         ];
//         let overlaps = array![0.5, 1.0];
//         let labels = array![1.0, 2.0];
//         let bbox_regression_thresh = 0.5;
//
//         let targets = compute_bbox_regression_targets(&rois, &overlaps, &labels, bbox_regression_thresh);
//         println!("Targets:\n{:?}", targets);
//     }

    // #[test]
    // fn test_expand_bbox_regression_target() {
    //     let bbox_targets_data = array![
    //         [1.0, 0.1, 0.2, 0.3, 0.4],
    //         [2.0, 0.2, 0.3, 0.4, 0.5],
    //         [0.0, 0.0, 0.0, 0.0, 0.0],
    //         [1.0, 0.3, 0.4, 0.5, 0.6]
    //     ];
    //
    //     let num_classes = 3;
    //     let config = Config {
    //         train: TrainConfig {
    //             bbox_normalization_precomputed: false,
    //             bbox_means: vec![],
    //             bbox_stds: vec![],
    //             bbox_weights: array![1.0, 1.0, 1.0, 1.0],
    //         },
    //     };
    //
    //     let (bbox_targets, bbox_weights) = expand_bbox_regression_targets(
    //         &bbox_targets_data,
    //         num_classes,
    //         &config.train.bbox_weights,
    //     );
    //
    //     println!("BBox Targets: {:?}", bbox_targets);
    //     println!("BBox Weights: {:?}", bbox_weights);
    // }
// }



// fn compute_bbox_regression_targets(
//     rois: &Array2<f32>,
//     overlaps: &Array1<f32>,
//     labels: &Array1<f32>,
//     bbox_regression_thresh: f32
// ) -> Array2<f32> {
//     // Ensure ROIs are floats (already ensured by type)
//
//     // Sanity check
//     assert!(rois.nrows() == overlaps.len());
//
//     if rois.nrows() != overlaps.len() {
//         eprintln!("bbox regression: len(rois) != len(overlaps)");
//     }
//
//     // Indices of ground-truth ROIs
//     let gt_inds: Vec<usize> = overlaps.iter()
//         .enumerate()
//         .filter(|&(_, &overlap)| overlap == 1.0)
//         .map(|(i, _)| i)
//         .collect();
//     assert!(!gt_inds.is_empty());
//     if gt_inds.is_empty() {
//         eprintln!("bbox regression: len(gt_inds) == 0");
//     }
//
//     // Indices of examples for which we try to make predictions
//     let ex_inds: Vec<usize> = overlaps.iter()
//         .enumerate()
//         .filter(|&(_, &overlap)| overlap >= bbox_regression_thresh)
//         .map(|(i, _)| i)
//         .collect();
//
//     // Get IoU overlap between each ex ROI and gt ROI
//     let ex_rois = rois.select(Axis(0), &ex_inds);
//     let gt_rois = rois.select(Axis(0), &gt_inds);
//     let ex_gt_overlaps = bbox_overlaps(&ex_rois, &gt_rois);
//
//     // Find which gt ROI each ex ROI has max overlap with
//     let gt_assignment: Vec<usize> = ex_gt_overlaps.axis_iter(Axis(1))
//         .map(|row| argmax(row))
//         .collect();
//
//     let gt_rois_assigned = gt_inds.iter()
//         .enumerate()
//         .map(|(i, &idx)| gt_rois.slice(s![idx, ..]))
//         .collect::<Vec<_>>();
//
//     // Prepare target array
//     let mut targets = Array2::<f32>::zeros((rois.nrows(), 5));
//
//     // Set the labels
//     for &ex_idx in &ex_inds {
//         targets[[ex_idx, 0]] = labels[ex_idx];
//     }
//
//     // Set the bbox regression targets
//     for (i, &ex_idx) in ex_inds.iter().enumerate() {
//         let ex_roi = rois.slice(s![ex_idx, ..]);
//         let gt_roi = rois.slice(s![gt_assignment[i], ..]);
//         let bbox_trans = nonlinear_transform(&ex_roi.to_owned(), &gt_roi.to_owned());
//         targets.slice_mut(s![ex_idx, 1..]).assign(&bbox_trans);
//     }
//
//     targets
// }

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
//         let trans = nonlinear_transform(&ex_roi, &gt_roi);
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
