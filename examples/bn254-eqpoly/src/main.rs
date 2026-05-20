//! BN254 equality-polynomial MLE evaluation driver — two prove/verify paths
//! behind a CLI flag:
//!
//! - `--backend inline` (default): every Fr op dispatches to the native FR
//!   coprocessor (FieldMul/Add/Sub + the FR Twist sumchecks).
//! - `--backend native`: software `ark_bn254::Fr` (Montgomery multiplication
//!   compiled to ordinary RV64IMAC).
//!
//! Both compute `eq(r, x) = ∏ᵢ (rᵢ·xᵢ + (1−rᵢ)·(1−xᵢ))` over deterministic
//! r, x ∈ Fr^32 generated from a `seed: u64`. Used as the standard FR-coprocessor
//! benchmark in place of the previous Poseidon2 example, because eqpoly is a
//! clean tight-chain Fr workload with no round-constant tables or MDS structure
//! to muddy the signal.

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
#[command(about = "BN254 eq(r,x) MLE evaluation prove + verify, two backends")]
struct Args {
    #[arg(long, value_enum, default_value_t = Backend::Inline)]
    backend: Backend,
    #[arg(long, default_value_t = 0x5EED_C0DEu64)]
    seed: u64,
}

pub fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let target_dir = "/tmp/jolt-guest-targets";
    let seed = args.seed;

    let (output, prove_secs, verify_secs, valid, label) = match args.backend {
        Backend::Inline => {
            let mut program = inline_guest::compile_bn254_eqpoly_inline(target_dir);
            let prove_start = Instant::now();
            let (output, bundle) = inline_guest::prove_bn254_eqpoly_inline(&mut program, seed)
                .expect("modular prove succeeds on bn254_eqpoly_inline");
            let prove_secs = prove_start.elapsed().as_secs_f64();
            let verify_start = Instant::now();
            let verify_result = inline_guest::verify_bn254_eqpoly_inline(&bundle, &mut program);
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
            let mut program = native_guest::compile_bn254_eqpoly_native(target_dir);
            let prove_start = Instant::now();
            let (output, bundle) = native_guest::prove_bn254_eqpoly_native(&mut program, seed)
                .expect("modular prove succeeds on bn254_eqpoly_native");
            let prove_secs = prove_start.elapsed().as_secs_f64();
            let verify_start = Instant::now();
            let verify_result = native_guest::verify_bn254_eqpoly_native(&bundle, &mut program);
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
    info!("=== bn254-eqpoly (modular Bolt backend, {label}) ===");
    info!("seed       : {seed:#018x}");
    info!("prove time : {prove_secs:.3} s");
    info!("verify time: {verify_secs:.3} s");
    info!("eq(r, x)   : {output:?}");
    info!("valid      : {valid_ok}");

    if let Err(err) = valid {
        info!("verify error: {err:?}");
        std::process::exit(1);
    }
}
