use image::{DynamicImage, ImageFormat, RgbImage};
use std::io::Cursor;
use std::error::Error;

pub fn byte_data_to_image(im_bytes: &[u8]) -> Result<DynamicImage, Box<dyn Error>> {
    // Decode the image bytes into an image::DynamicImage
    let img = image::load(Cursor::new(im_bytes), ImageFormat::from_extension("jpg").unwrap())?;

    // Convert the image to RGB if it is RGBA or Grayscale
    let img = match img {
        DynamicImage::ImageRgba8(img) => DynamicImage::ImageRgba8(img).into_rgb8(),
        DynamicImage::ImageLuma8(img) => DynamicImage::ImageLuma8(img).into_rgb8(),
        _ => RgbImage::from(img),
    };

    Ok(DynamicImage::from(img))
}

#[cfg(test)]
mod tests {
    use crate::utils::utils::byte_data_to_image;

    #[test]
    fn test_nms() {
        let im_bytes: &[u8] = include_bytes!("../../test_data/anderson.jpg");

        match byte_data_to_image(im_bytes) {
            Ok(img) => println!("Image loaded and converted successfully"),
            Err(e) => eprintln!("Failed to load and convert image: {}", e),
        }
    }

}