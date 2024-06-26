use anyhow::Error;
use anyhow::Result;
use std::ptr::null;
use std::time;
use tonic::transport::Channel;
use tonic::Request;

pub mod triton {
    tonic::include_proto!("inference");
}

use triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use triton::{ModelInferRequest, ModelReadyRequest, ModelConfigRequest};
use crate::triton_client::client::triton::{CudaSharedMemoryRegisterRequest, CudaSharedMemoryRegisterResponse,
                                           CudaSharedMemoryStatusRequest, CudaSharedMemoryStatusResponse,
                                           CudaSharedMemoryUnregisterRequest, CudaSharedMemoryUnregisterResponse,
                                           ModelConfigResponse, ModelInferResponse,
                                           ModelMetadataRequest, ModelMetadataResponse,
                                           ModelReadyResponse, ModelStatisticsRequest,
                                           ModelStatisticsResponse, RepositoryIndexRequest,
                                           RepositoryIndexResponse, RepositoryModelLoadRequest,
                                           RepositoryModelLoadResponse, RepositoryModelUnloadRequest,
                                           RepositoryModelUnloadResponse, ServerLiveRequest,
                                           ServerLiveResponse, ServerMetadataRequest,
                                           ServerMetadataResponse, ServerReadyRequest,
                                           ServerReadyResponse, SystemSharedMemoryRegisterRequest,
                                           SystemSharedMemoryRegisterResponse, SystemSharedMemoryStatusRequest,
                                           SystemSharedMemoryStatusResponse, SystemSharedMemoryUnregisterRequest,
                                           SystemSharedMemoryUnregisterResponse, TraceSettingRequest,
                                           TraceSettingResponse};

#[derive(Debug, Clone)]
pub struct TritonInferenceClient {
    c: GrpcInferenceServiceClient<Channel>,
}

macro_rules! wrap_method_with_args {
    ($doc:literal, $name:ident, $req_type:ty, $resp_type:ty) => {
        #[doc=$doc]
        pub async fn $name(&self, req: $req_type) -> Result<$resp_type, Error> {
            let response = self.c.clone().$name(tonic::Request::new(req)).await?;
            Ok(response.into_inner())
        }
    };
}

macro_rules! wrap_method_no_args {
    ($doc:literal, $name:ident, $req_type:ty, $resp_type:ty) => {
        #[doc=$doc]
        pub async fn $name(&self) -> Result<$resp_type, Error> {
            let req: $req_type = Default::default();
            let response = self.c.clone().$name(tonic::Request::new(req)).await?;
            Ok(response.into_inner())
        }
    };
}

impl TritonInferenceClient {
    async fn new(host: &str, port: &str) -> Self {
        let channel_url = format!("{}:{}", host, port);
        let channel = Channel::from_shared(channel_url).expect("url must be valid")
            .connect()
            .await.unwrap();

        let client = GrpcInferenceServiceClient::new(channel);

        TritonInferenceClient {
            c: client,
        }
    }

    wrap_method_no_args!(
        "Check liveness of the inference server.",
        server_live,
        ServerLiveRequest,
        ServerLiveResponse
    );

    wrap_method_no_args!(
        "Check readiness of the inference server.",
        server_ready,
        ServerReadyRequest,
        ServerReadyResponse
    );

    wrap_method_with_args!(
        "Check readiness of a model in the inference server.",
        model_ready,
        ModelReadyRequest,
        ModelReadyResponse
    );

    wrap_method_no_args!(
        "Get server metadata.",
        server_metadata,
        ServerMetadataRequest,
        ServerMetadataResponse
    );

    wrap_method_with_args!(
        "Get model metadata.",
        model_metadata,
        ModelMetadataRequest,
        ModelMetadataResponse
    );

    wrap_method_with_args!(
        "Perform inference using specific model.",
        model_infer,
        ModelInferRequest,
        ModelInferResponse
    );

    wrap_method_with_args!(
        "Get model configuration.",
        model_config,
        ModelConfigRequest,
        ModelConfigResponse
    );

    wrap_method_with_args!(
        "Get the cumulative inference statistics for a model.",
        model_statistics,
        ModelStatisticsRequest,
        ModelStatisticsResponse
    );

    wrap_method_with_args!(
        "Get the index of model repository contents.",
        repository_index,
        RepositoryIndexRequest,
        RepositoryIndexResponse
    );

    wrap_method_with_args!(
        "Load or reload a model from a repository.",
        repository_model_load,
        RepositoryModelLoadRequest,
        RepositoryModelLoadResponse
    );

    wrap_method_with_args!(
        "Unload a model.",
        repository_model_unload,
        RepositoryModelUnloadRequest,
        RepositoryModelUnloadResponse
    );

    wrap_method_with_args!(
        "Get the status of all registered system-shared-memory regions.",
        system_shared_memory_status,
        SystemSharedMemoryStatusRequest,
        SystemSharedMemoryStatusResponse
    );

    wrap_method_with_args!(
        "Register a system-shared-memory region.",
        system_shared_memory_register,
        SystemSharedMemoryRegisterRequest,
        SystemSharedMemoryRegisterResponse
    );

    wrap_method_with_args!(
        "Unregister a system-shared-memory region.",
        system_shared_memory_unregister,
        SystemSharedMemoryUnregisterRequest,
        SystemSharedMemoryUnregisterResponse
    );

    wrap_method_with_args!(
        "Get the status of all registered CUDA-shared-memory regions.",
        cuda_shared_memory_status,
        CudaSharedMemoryStatusRequest,
        CudaSharedMemoryStatusResponse
    );

    wrap_method_with_args!(
        "Register a CUDA-shared-memory region.",
        cuda_shared_memory_register,
        CudaSharedMemoryRegisterRequest,
        CudaSharedMemoryRegisterResponse
    );

    wrap_method_with_args!(
        "Unregister a CUDA-shared-memory region.",
        cuda_shared_memory_unregister,
        CudaSharedMemoryUnregisterRequest,
        CudaSharedMemoryUnregisterResponse
    );

    wrap_method_with_args!(
        "Update and get the trace setting of the Triton server.",
        trace_setting,
        TraceSettingRequest,
        TraceSettingResponse
    );

}

#[cfg(test)]
mod tests {
    use crate::triton_client::client::TritonInferenceClient;

    #[tokio::test]
    async fn test_new() {
        let mut client = TritonInferenceClient::new("http://10.124.71.246", "8604").await;
    }
}
