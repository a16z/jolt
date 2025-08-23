use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let _program = guest::compile_verifier(target_dir);

    // let prover_preprocessing = guest::preprocess_prover_verifier(&mut program);
    // let verifier_preprocessing =
    //     guest::verifier_preprocessing_from_prover_verifier(&prover_preprocessing);

    // let prove = guest::build_prover_verifier(program, prover_preprocessing);
    // let verify = guest::build_verifier_verifier(verifier_preprocessing);
    let now = Instant::now();
    // let (output, proof, program_io) = prove();
    guest::trace_verifier_to_file("/tmp/jolt.trace");
    println!("Trace runtime: {} s", now.elapsed().as_secs_f64());
    // let is_valid = verify(output, program_io.panic, proof);
    // println!("int to string valid: {is_valid}");
}
