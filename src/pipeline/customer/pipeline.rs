#[derive(Debug)]
enum FaceQualityClass {
    Bad = 0,
    Good = 1,
    WearingMask = 2,
    WearingSunGlasses = 3,
}

#[derive(Debug)]
struct RetinaFaceDetectionConfig {
    model_name: String,
    timeout: u32,
    image_size: (u32, u32),
    max_batch_size: u32,
    confidence_threshold: f32,
    iou_threshold: f32,
}

impl RetinaFaceDetectionConfig {
    fn new(model_name: String) -> Self {
        RetinaFaceDetectionConfig {
            model_name,
            timeout: 20,
            image_size: (640, 640),
            max_batch_size: 1,
            confidence_threshold: 0.7,
            iou_threshold: 0.45,
        }
    }
}


#[derive(Debug)]
struct FaceAlignConfig {
    image_size: (u32, u32),
    standard_landmarks: Vec<[f32; 2]>,
}

impl FaceAlignConfig {
    fn new() -> Self {
        FaceAlignConfig {
            image_size: (112, 112),
            standard_landmarks: vec![
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
        }
    }
}

#[derive(Debug)]
struct ArcFaceRecognitionConfig {
    model_name: String,
    timeout: u32,
    image_size: (u32, u32),
    batch_size: u32,
}

impl ArcFaceRecognitionConfig {
    fn new(model_name: String) -> Self {
        ArcFaceRecognitionConfig {
            model_name,
            timeout: 20,
            image_size: (112, 112),
            batch_size: 1,
        }
    }
}

#[derive(Debug)]
struct FaceQualityConfig {
    model_name: String,
    timeout: u32,
    image_size: (u32, u32),
    batch_size: u32,
    threshold: f32,
}

impl FaceQualityConfig {
    fn new(model_name: String) -> Self {
        FaceQualityConfig {
            model_name,
            timeout: 20,
            image_size: (112, 112),
            batch_size: 1,
            threshold: 0.5,
        }
    }
}

#[derive(Debug)]
struct FaceSelectionConfig {
    margin_center_left_ratio: f32,
    margin_center_right_ratio: f32,
    margin_edge_ratio: f32,
    minimum_face_ratio: f32,
    minimum_width_height_ratio: f32,
    maximum_width_height_ratio: f32,
}

impl FaceSelectionConfig {
    fn new() -> Self {
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
