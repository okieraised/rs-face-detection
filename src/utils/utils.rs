use opencv::core::{self, Mat, MatTraitConst};
use opencv::imgcodecs::{imdecode, IMREAD_UNCHANGED};
use opencv::imgproc::{cvt_color, COLOR_RGBA2RGB, COLOR_GRAY2RGB};
use anyhow::{Error, Result};

pub fn byte_data_to_opencv(im_bytes: &[u8]) -> Result<Mat, Error> {
    // Convert bytes to Mat
    let img_as_mat = match Mat::from_slice(im_bytes) {
        Ok(img_as_mat) => img_as_mat,
        Err(e) => {
            return Err(Error::from(e))
        }
    };

    // Decode the image
    let opencv_img = match imdecode(&img_as_mat, IMREAD_UNCHANGED) {
        Ok(opencv_img) => opencv_img,
        Err(e) => {
            return Err(Error::from(e))
        }
    };

    // Check the number of channels and convert if necessary
    let opencv_img = match opencv_img.channels() {
        4 => {
            let mut rgb_img = Mat::default();
            match cvt_color(&opencv_img, &mut rgb_img, COLOR_RGBA2RGB, 0) {
                Ok(_) => {},
                Err(e) => {
                    return Err(Error::from(e))
                }
            };
            rgb_img
        }
        2 => {
            let mut rgb_img = Mat::default();
            match cvt_color(&opencv_img, &mut rgb_img, COLOR_GRAY2RGB, 0) {
                Ok(_) => {
                },
                Err(e) => {
                    return Err(Error::from(e))
                }
            };
            rgb_img
        }
        _ => opencv_img,
    };

    Ok(opencv_img)
}


#[cfg(test)]
mod tests {
    use crate::utils::utils::byte_data_to_opencv;

    #[test]
    fn test_nms() {
        let im_bytes: &[u8] = include_bytes!("../../test_data/anderson.jpg");

        match byte_data_to_opencv(im_bytes) {
            Ok(img) => println!("Image loaded and converted successfully {:?}", img),
            Err(e) => eprintln!("Failed to load and convert image: {}", e),
        }
    }

}