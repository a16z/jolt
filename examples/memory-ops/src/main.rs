use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_memory_ops(target_dir);

    let prover_preprocessing = guest::preprocess_prover_memory_ops(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_memory_ops(&prover_preprocessing);

    let prove = guest::build_prover_memory_ops(program, prover_preprocessing);
    let verify = guest::build_verifier_memory_ops(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof) = prove();
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify(output, proof);

    println!(
        "outputs: {} {} {} {}",
        output.0, output.1, output.2, output.3
    );
    println!("valid: {is_valid}");
}
