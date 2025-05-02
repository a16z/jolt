use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_muldiv(target_dir);

    let prover_preprocessing = guest::preprocess_prover_muldiv(&program);
    let verifier_preprocessing = guest::preprocess_verifier_muldiv(&program);

    let prove = guest::build_prover_muldiv(program, prover_preprocessing);
    let verify = guest::build_verifier_muldiv(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof) = prove(12031293, 17, 92);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify(12031293, 17, 92, output, proof);

    println!("output: {output}");
    println!("valid: {is_valid}");
}
