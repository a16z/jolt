use jolt_sdk::serialize_and_print_size;
use jolt_sdk::DoryContext;
use jolt_sdk::DoryGlobals;
use jolt_sdk::DoryLayout;
#[cfg(feature = "zk")]
use jolt_sdk::PrivateInput;
use jolt_sdk::UntrustedAdvice;
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();
    let layout = match std::env::var("FIB2_DORY_LAYOUT")
        .ok()
        .map(|v| v.to_ascii_lowercase())
        .as_deref()
    {
        Some("cycle") | Some("cyclemajor") => DoryLayout::CycleMajor,
        Some("address") | Some("addressmajor") | None => DoryLayout::AddressMajor,
        Some(other) => panic!(
            "invalid FIB2_DORY_LAYOUT='{other}', expected one of: cycle, cyclemajor, address, addressmajor"
        ),
    };

    DoryGlobals::initialize_context(1, 1, DoryContext::Main, Some(layout))
        .expect("failed to set Dory layout");
    info!("Using Dory layout: {layout:?}");

    let save_to_disk = std::env::args().any(|arg| arg == "--save");
    let target_dir = "/tmp/jolt-guest-targets";

    let mut program = guest::compile_fib(target_dir);
    let shared_preprocessing = guest::preprocess_shared_fib(&mut program);
    let prover_preprocessing = guest::preprocess_prover_fib(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    #[cfg(feature = "zk")]
    let blindfold_setup = Some(prover_preprocessing.blindfold_setup());
    #[cfg(not(feature = "zk"))]
    let blindfold_setup = None;
    let verifier_preprocessing =
        guest::preprocess_verifier_fib(shared_preprocessing, verifier_setup, blindfold_setup);

    if save_to_disk {
        serialize_and_print_size(
            "Verifier Preprocessing",
            "/tmp/jolt_fib2_verifier_preprocessing.dat",
            &verifier_preprocessing,
        )
        .expect("Could not serialize preprocessing.");
    }

    let prove_fib = guest::build_prover_fib(program, prover_preprocessing);
    let verify_fib = guest::build_verifier_fib(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, io_device) = prove_fib(50);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    if save_to_disk {
        serialize_and_print_size("Proof", "/tmp/jolt_fib2_proof.bin", &proof)
            .expect("Could not serialize proof.");
        serialize_and_print_size("io_device", "/tmp/jolt_fib2_io_device.bin", &io_device)
            .expect("Could not serialize io_device.");
    }

    let is_valid = verify_fib(50, output, io_device.panic, proof);
    info!("output: {output}");
    info!("valid: {is_valid}");

    #[cfg(feature = "zk")]
    {
        let mut private_program = guest::compile_fib_with_private_input(target_dir);
        let private_shared_preprocessing =
            guest::preprocess_shared_fib_with_private_input(&mut private_program);
        let private_prover_preprocessing =
            guest::preprocess_prover_fib_with_private_input(private_shared_preprocessing.clone());
        let private_verifier_setup = private_prover_preprocessing.generators.to_verifier_setup();
        let private_verifier_preprocessing = guest::preprocess_verifier_fib_with_private_input(
            private_shared_preprocessing,
            private_verifier_setup,
            Some(private_prover_preprocessing.blindfold_setup()),
        );

        let prove_fib_with_private_input = guest::build_prover_fib_with_private_input(
            private_program,
            private_prover_preprocessing,
        );
        let verify_fib_with_private_input =
            guest::build_verifier_fib_with_private_input(private_verifier_preprocessing);

        let now = Instant::now();
        let (private_output, private_proof, private_io_device) =
            prove_fib_with_private_input(50, PrivateInput::new(0u32));
        info!(
            "Prover runtime with private input: {} s",
            now.elapsed().as_secs_f64()
        );

        let private_valid = verify_fib_with_private_input(
            50,
            private_output,
            private_io_device.panic,
            private_proof,
        );
        info!("output with private input: {private_output}");
        info!("valid with private input: {private_valid}");
    }

    let mut advice_program = guest::compile_fib_with_large_advice_input(target_dir);
    let advice_shared_preprocessing =
        guest::preprocess_shared_fib_with_large_advice_input(&mut advice_program);
    let advice_prover_preprocessing =
        guest::preprocess_prover_fib_with_large_advice_input(advice_shared_preprocessing.clone());
    let advice_verifier_setup = advice_prover_preprocessing.generators.to_verifier_setup();
    #[cfg(feature = "zk")]
    let advice_blindfold_setup = Some(advice_prover_preprocessing.blindfold_setup());
    #[cfg(not(feature = "zk"))]
    let advice_blindfold_setup = None;
    let advice_verifier_preprocessing = guest::preprocess_verifier_fib_with_large_advice_input(
        advice_shared_preprocessing,
        advice_verifier_setup,
        advice_blindfold_setup,
    );

    let prove_fib_with_large_advice = guest::build_prover_fib_with_large_advice_input(
        advice_program,
        advice_prover_preprocessing,
    );
    let verify_fib_with_large_advice =
        guest::build_verifier_fib_with_large_advice_input(advice_verifier_preprocessing);

    let advice_payload = vec![7u8; 65536];
    let advice_input = UntrustedAdvice::new(advice_payload.as_slice());
    let advice_input_bytes = jolt_sdk::postcard::to_stdvec(&advice_input)
        .expect("failed to serialize advice input")
        .len();

    let now = Instant::now();
    let (advice_output, advice_proof, advice_io_device) =
        prove_fib_with_large_advice(50, advice_input);
    info!(
        "Prover runtime with large advice input: {} s",
        now.elapsed().as_secs_f64()
    );

    let advice_trace_length = advice_proof.trace_length as usize;
    assert!(
        advice_input_bytes > advice_trace_length,
        "expected advice input bytes ({advice_input_bytes}) to exceed trace length ({advice_trace_length})",
    );

    let advice_valid =
        verify_fib_with_large_advice(50, advice_output, advice_io_device.panic, advice_proof);
    info!("output with large advice input: {advice_output}");
    info!("valid with large advice input: {advice_valid}");
    info!("advice input bytes: {advice_input_bytes}");
    info!("trace length with large advice input: {advice_trace_length}");
}
