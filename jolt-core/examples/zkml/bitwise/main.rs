use ark_bn254::{Bn254, Fr};
use jolt_core::jolt_onnx::trace::onnx::{JoltONNXDevice, ONNXParser};
use jolt_core::jolt_onnx::vm::onnx_vm::ONNXJoltVM;
use jolt_core::poly::commitment::hyperkzg::HyperKZG;
use jolt_core::utils::transcript::KeccakTranscript;

type PCS = HyperKZG<Bn254, KeccakTranscript>;
type F = Fr;
type ProofTranscript = KeccakTranscript;

fn main() {
    // Setup model and get trace (input for proving)
    let model_path = "./onnx/bitwise_test.onnx";
    let graph = ONNXParser::load_model(model_path).unwrap();
    let trace = graph.trace(); // TODO: make this more opaque to the user

    // Generate preprocessing
    println!("Generating preprocessing...");
    let pp = ONNXJoltVM::<F, PCS, ProofTranscript>::prover_preprocess(1 << 20);

    // Prove
    println!("Proving SNARK...");
    let io = JoltONNXDevice::new(graph.input_count as u64, graph.output_count as u64);
    let (snark, commitments, verifier_io, _) =
        ONNXJoltVM::<F, PCS, ProofTranscript>::prove(io, trace, pp.clone());

    // Verify
    println!("Verifying SNARK...");
    snark
        .verify(pp.shared, commitments, verifier_io, None)
        .unwrap();
    println!("SNARK verified successfully!");
}
