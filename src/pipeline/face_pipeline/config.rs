use ndarray::Array2;

#[derive(Debug)]
pub enum FaceQualityClass {
    Bad = 0,
    Good = 1,
    WearingMask = 2,
    WearingSunGlasses = 3,
}

#[derive(Debug)]
pub struct FaceDetectionConfig {
    pub model_name: String,
    pub timeout: i32,
    pub image_size: (i32, i32),
    pub max_batch_size: i32,
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
}

impl FaceDetectionConfig {
    pub(crate) fn new() -> Self {
        FaceDetectionConfig {
            model_name: "face_detection_retina".to_string(),
            timeout: 20,
            image_size: (640, 640),
            max_batch_size: 1,
            confidence_threshold: 0.7,
            iou_threshold: 0.45,
        }
    }
}


#[derive(Debug)]
pub struct FaceAlignmentConfig {
    pub image_size: (i32, i32),
    pub standard_landmarks: Array2<f32>,
}

impl FaceAlignmentConfig {
    pub fn new() -> Self {
        FaceAlignmentConfig {
            image_size: (112, 112),
            standard_landmarks: Array2::from(vec![
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ]),
        }
    }
}

#[derive(Debug)]
pub struct FaceIdentificationConfig {
    pub model_name: String,
    pub timeout: i32,
    pub image_size: (i32, i32),
    pub batch_size: i32,
}

impl FaceIdentificationConfig {
    pub fn new() -> Self {
        FaceIdentificationConfig {
            model_name: "face_identification".to_string(),
            timeout: 20,
            image_size: (112, 112),
            batch_size: 1,
        }
    }
}

#[derive(Debug)]
pub struct FaceQualityConfig {
    pub model_name: String,
    pub timeout: i32,
    pub image_size: (i32, i32),
    pub batch_size: i32,
    pub threshold: f32,
}

impl FaceQualityConfig {
    pub fn new() -> Self {
        FaceQualityConfig {
            model_name: "face_quality".to_string(),
            timeout: 20,
            image_size: (112, 112),
            batch_size: 1,
            threshold: 0.5,
        }
    }
}

#[derive(Debug)]
pub struct FaceSelectionConfig {
    pub margin_center_left_ratio: f32,
    pub margin_center_right_ratio: f32,
    pub margin_edge_ratio: f32,
    pub minimum_face_ratio: f32,
    pub minimum_width_height_ratio: f32,
    pub maximum_width_height_ratio: f32,
}

impl FaceSelectionConfig {
    pub fn new() -> Self {
        FaceSelectionConfig {
            margin_center_left_ratio: 0.3,
            margin_center_right_ratio: 0.3,
            margin_edge_ratio: 0.1,
            minimum_face_ratio: 0.0075,
            minimum_width_height_ratio: 0.65,
            maximum_width_height_ratio: 1.1,
        }
    }
}
