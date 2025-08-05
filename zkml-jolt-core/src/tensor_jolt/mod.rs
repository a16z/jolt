pub mod execution_trace;
pub mod instruction;
// pub mod tensor_heap;
pub mod r1cs;

#[cfg(test)]
mod tests {
    use onnx_tracer::{custom_addsubmul_model, tensor::Tensor};

    #[test]
    fn test_custom_addsubmul() {
        // --- Preprocessing ---
        let custom_addsubmul_model = custom_addsubmul_model();
        let program_bytecode = onnx_tracer::decode_model(custom_addsubmul_model.clone());

        // --- Proving ---
        let input = Tensor::new(Some(&[10, 20, 30, 40]), &[1, 4]).unwrap();
        let execution_trace = onnx_tracer::execution_trace(custom_addsubmul_model, &input);
        println!("Execution trace: {execution_trace:#?}");
        // let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
        //     JoltSNARK::prove(pp.clone(), execution_trace);

        // // --- Verification ---
        // snark.verify((&pp).into()).unwrap();
    }
}
