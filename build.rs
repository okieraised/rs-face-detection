fn main() {
    tonic_build::compile_protos("triton-proto/model_config.proto").unwrap();
    tonic_build::compile_protos("triton-proto/grpc_service.proto").unwrap();
}