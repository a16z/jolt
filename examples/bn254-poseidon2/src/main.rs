//! Poseidon2 BN254 t=3 driver — two prove/verify paths behind a CLI flag:
//!
//! - `--backend inline` (default): every Fr op dispatches to the native FR
//!   coprocessor (FieldMul/Add/Sub/Inv/Mov + the FR Twist sumchecks).
//! - `--backend native`: software `ark_bn254::Fr` (Montgomery multiplication
//!   compiled to ordinary RV64IMAC). No FR coprocessor used.
//!
//! Both paths exercise the modular Bolt backend (`#[jolt::provable(backend =
//! "modular")]`) end-to-end. Used for correctness round-trip and as the
//! standard FR-coprocessor benchmark.

use std::time::Instant;

use clap::{Parser, ValueEnum};
use tracing::info;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Backend {
    /// Inline FR coprocessor — `jolt-inlines-bn254-fr` lowers each Fr op to
    /// a native FieldOp instruction. Default.
    Inline,
    /// Software `ark_bn254::Fr` — Montgomery multiplication runs as ordinary
    /// RV64IMAC cycles. No FR coprocessor.
    Native,
}

#[derive(Parser, Debug)]
#[command(about = "BN254 Poseidon2 (t=3) prove + verify, two backends")]
struct Args {
    #[arg(long, value_enum, default_value_t = Backend::Inline)]
    backend: Backend,
}

pub fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let target_dir = "/tmp/jolt-guest-targets";
    let s0: [u64; 4] = [1, 0, 0, 0];
    let s1: [u64; 4] = [2, 0, 0, 0];
    let s2: [u64; 4] = [3, 0, 0, 0];

    let (output, prove_secs, verify_secs, valid, label) = match args.backend {
        Backend::Inline => {
            let mut program = inline_guest::compile_bn254_poseidon2_inline(target_dir);
            let prove_start = Instant::now();
            let (output, bundle) =
                inline_guest::prove_bn254_poseidon2_inline(&mut program, s0, s1, s2)
                    .expect("modular prove succeeds on bn254_poseidon2_inline");
            let prove_secs = prove_start.elapsed().as_secs_f64();
            let verify_start = Instant::now();
            let verify_result = inline_guest::verify_bn254_poseidon2_inline(&bundle, &mut program);
            let verify_secs = verify_start.elapsed().as_secs_f64();
            (
                output,
                prove_secs,
                verify_secs,
                verify_result,
                "inline FR coprocessor",
            )
        }
        Backend::Native => {
            let mut program = native_guest::compile_bn254_poseidon2_native(target_dir);
            let prove_start = Instant::now();
            let (output, bundle) =
                native_guest::prove_bn254_poseidon2_native(&mut program, s0, s1, s2)
                    .expect("modular prove succeeds on bn254_poseidon2_native");
            let prove_secs = prove_start.elapsed().as_secs_f64();
            let verify_start = Instant::now();
            let verify_result = native_guest::verify_bn254_poseidon2_native(&bundle, &mut program);
            let verify_secs = verify_start.elapsed().as_secs_f64();
            (
                output,
                prove_secs,
                verify_secs,
                verify_result,
                "software ark_bn254::Fr",
            )
        }
    };

    let valid_ok = valid.is_ok();
    info!("=== bn254-poseidon2 (modular Bolt backend, {label}) ===");
    info!("prove time : {prove_secs:.3} s");
    info!("verify time: {verify_secs:.3} s");
    info!("output[0]  : {:?}", output[0]);
    info!("output[1]  : {:?}", output[1]);
    info!("output[2]  : {:?}", output[2]);
    info!("valid      : {valid_ok}");

    if let Err(err) = valid {
        info!("verify error: {err:?}");
        std::process::exit(1);
    }
}
