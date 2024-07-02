mod rcnn;
mod processing;
pub mod triton_client;
pub mod pipeline;
mod utils;

#[cfg(test)]
mod tests {
    use crate::triton_client::client::TritonInferenceClient;

    #[tokio::test]
    async fn test_pipeline() {
        let triton_host = "";
        let triton_port = "";

        let im_bytes: &[u8] = include_bytes!("");


        let triton_infer_client = match TritonInferenceClient::new(triton_host, triton_port).await {
            Ok(triton_infer_client) => triton_infer_client,
            Err(e) => {
                println!("{:?}", e);
                return
            }
        };

        let pipeline = match crate::pipeline::face_pipeline::pipeline::FacePipeline::new(triton_host, triton_port, Some(true), Some(false)).await {
            Ok(pipeline) => {pipeline}
            Err(e) => {
                println!("{:?}", e);
                return
            }
        };

        let _ = pipeline.extract(im_bytes).await;
    }
}
