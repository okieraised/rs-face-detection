// Used the ndarray crate to handle multi-dimensional arrays. The logic of the Python function is preserved,
// iterating over width, height, and the number of base anchors to calculate the resulting anchors.
// The function returns a Array4<f32>.

use ndarray::{Array2, Array4};

fn anchors(height: usize, width: usize, stride: usize, base_anchors: Array2<f32>) -> Array4<f32> {
    let a = base_anchors.shape()[0];
    let mut all_anchors = Array4::<f32>::zeros((height, width, a, 4));

    for iw in 0..width {
        let sw = iw * stride;
        for ih in 0..height {
            let sh = ih * stride;
            for k in 0..a {
                all_anchors[[ih, iw, k, 0]] = base_anchors[[k, 0]] + sw as f32;
                all_anchors[[ih, iw, k, 1]] = base_anchors[[k, 1]] + sh as f32;
                all_anchors[[ih, iw, k, 2]] = base_anchors[[k, 2]] + sw as f32;
                all_anchors[[ih, iw, k, 3]] = base_anchors[[k, 3]] + sh as f32;
            }
        }
    }
    all_anchors
}
