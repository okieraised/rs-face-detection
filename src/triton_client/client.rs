use tonic::transport::Channel;
use tonic::Request;

pub mod triton {
    tonic::include_proto!("inference");
}

use triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use triton::ModelInferRequest;



async fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Create a channel to the Triton server
    let channel = Channel::from_static("http://[::1]:8001")
        .connect()
        .await?;

    // Create an inference client
    let mut client = GrpcInferenceServiceClient::new(channel);

    // Prepare the inference request
    let request = ModelInferRequest {
        model_name: "your_model_name".into(),
        inputs: vec![],
        outputs: vec![],
        ..Default::default()
    };

    // Send the request
    let response = client.model_infer(Request::new(request)).await?;

    println!("Response: {:?}", response);

    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {}", e);
    }
}
