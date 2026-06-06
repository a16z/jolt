use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_field_inline_eq_poly(target_dir);

    let shared_preprocessing = guest::preprocess_shared_field_inline_eq_poly(&mut program)
        .expect("field-inline preprocessing should succeed");
    let prover_preprocessing =
        guest::preprocess_prover_field_inline_eq_poly(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_field_inline_eq_poly(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
        None,
    );

    let prove = guest::build_prover_field_inline_eq_poly(program, prover_preprocessing);
    let verify = guest::build_verifier_field_inline_eq_poly(verifier_preprocessing);

    let input = 7u32;
    let native_output = guest::field_inline_eq_poly(input);
    let (output, proof, program_io) = prove(input);
    let is_valid = verify(input, output, program_io.panic, proof);

    assert_eq!(output, native_output);
    info!("output: {output}");
    info!("valid: {is_valid}");
}
