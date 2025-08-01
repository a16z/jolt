use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_failing_verifier(target_dir);

    // let prover_preprocessing = guest::preprocess_prover_failing_verifier(&mut program);
    // let verifier_preprocessing =
    //     guest::verifier_preprocessing_from_prover_failing_verifier(&prover_preprocessing);

    // let prove = guest::build_prover_failing_verifier(program, prover_preprocessing);
    // let verify = guest::build_verifier_failing_verifier(verifier_preprocessing);
    let now = Instant::now();
    // let (output, proof, program_io) = prove();
    let analysis = guest::analyze_failing_verifier().trace;
    println!("Program cycle count: {:?}", analysis.len());
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    // let is_valid = verify(output, program_io.panic, proof);
    // println!("int to string valid: {is_valid}");
}
