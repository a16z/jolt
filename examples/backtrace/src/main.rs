#[cfg(all(feature = "nostd", feature = "std"))]
compile_error!("Enable only one of `nostd` or `std` to avoid guest feature unification.");

#[cfg(any(feature = "nostd", feature = "std"))]
use std::env;

#[cfg(any(feature = "nostd", feature = "std"))]
use std::time::Instant;
#[cfg(any(feature = "nostd", feature = "std"))]
use tracing::info;

fn main() {
    tracing_subscriber::fmt::init();

    #[cfg(any(feature = "nostd", feature = "std"))]
    let should_panic = env_flag("JOLT_BT_TRIGGER").unwrap_or(true);
    #[cfg(any(feature = "nostd", feature = "std"))]
    let target_dir = "/tmp/jolt-guest-targets";

    #[cfg(feature = "nostd")]
    run_nostd(target_dir, should_panic);

    #[cfg(feature = "std")]
    run_std(target_dir, should_panic);

    #[cfg(not(any(feature = "nostd", feature = "std")))]
    {
        eprintln!("Enable feature `nostd` or `std`");
        std::process::exit(1);
    }
}

#[cfg(any(feature = "nostd", feature = "std"))]
fn env_flag(key: &str) -> Option<bool> {
    env::var(key).ok().map(|v| {
        let v = v.trim().to_ascii_lowercase();
        !(v.is_empty() || v == "0" || v == "false" || v == "off")
    })
}

#[cfg(feature = "nostd")]
fn run_nostd(target_dir: &str, should_panic: bool) {
    info!("mode=nostd should_panic={}", should_panic);

    let trace_enabled = env_flag("JOLT_BACKTRACE").unwrap_or(false);
    let trace_file = format!("{target_dir}/backtrace-nostd.trace");

    let mut program = guest_nostd::compile_panic_backtrace_nostd(target_dir);

    let shared_preprocessing = guest_nostd::preprocess_shared_panic_backtrace_nostd(&mut program);
    let prover_preprocessing =
        guest_nostd::preprocess_prover_panic_backtrace_nostd(shared_preprocessing.clone());
    let verifier_preprocessing = guest_nostd::preprocess_verifier_panic_backtrace_nostd(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    let prove = guest_nostd::build_prover_panic_backtrace_nostd(program, prover_preprocessing);
    let verify = guest_nostd::build_verifier_panic_backtrace_nostd(verifier_preprocessing);

    if trace_enabled && should_panic {
        info!("nostd trace -> {}", trace_file);
        guest_nostd::trace_panic_backtrace_nostd_to_file(&trace_file, should_panic);
    }

    let now = Instant::now();
    let (output, proof, program_io) = prove(should_panic);
    info!("nostd prover runtime: {} s", now.elapsed().as_secs_f64());
    info!("nostd output: {:?}", output);
    info!("nostd panicked: {}", program_io.panic);

    let is_valid = verify(should_panic, output, program_io.panic, proof);
    info!("nostd proof valid: {}", is_valid);

    assert!(is_valid, "nostd backtrace proof verification failed");
    assert_eq!(program_io.panic, should_panic, "nostd panic flag mismatch");
}

#[cfg(feature = "std")]
fn run_std(target_dir: &str, should_panic: bool) {
    info!("mode=std should_panic={}", should_panic);

    let trace_enabled = env_flag("JOLT_BACKTRACE").unwrap_or(false);
    let trace_file = format!("{target_dir}/backtrace-std.trace");

    let mut program = guest_std::compile_panic_backtrace_std(target_dir);

    let shared_preprocessing = guest_std::preprocess_shared_panic_backtrace_std(&mut program);
    let prover_preprocessing =
        guest_std::preprocess_prover_panic_backtrace_std(shared_preprocessing.clone());
    let verifier_preprocessing = guest_std::preprocess_verifier_panic_backtrace_std(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    let prove = guest_std::build_prover_panic_backtrace_std(program, prover_preprocessing);
    let verify = guest_std::build_verifier_panic_backtrace_std(verifier_preprocessing);

    if trace_enabled && should_panic {
        info!("std trace -> {}", trace_file);
        guest_std::trace_panic_backtrace_std_to_file(&trace_file, should_panic);
    }

    let now = Instant::now();
    let (output, proof, program_io) = prove(should_panic);
    info!("std prover runtime: {} s", now.elapsed().as_secs_f64());
    info!("std output: {:?}", output);
    info!("std panicked: {}", program_io.panic);

    let is_valid = verify(should_panic, output, program_io.panic, proof);
    info!("std proof valid: {}", is_valid);

    assert!(is_valid, "std backtrace proof verification failed");
    assert_eq!(program_io.panic, should_panic, "std panic flag mismatch");
}
