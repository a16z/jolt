//! End-to-end test: real muldiv guest → protocol.jolt → prove → verify.
//!
//! Exercises the full pipeline with a real RISC-V program:
//!   1. Compile + trace the muldiv guest via jolt-host
//!   2. Generate protocol.jolt from the ground-truth example (matching params)
//!   3. Link the Module<PolynomialId> to the CPU backend
//!   4. `prove()` with real trace data
//!   5. `verify()` the proof
//!
//! Currently limited to Stage 1 (Outer Spartan) — virtual polynomials for
//! Stage 2+ (product accumulator, read-write checking) are not yet computed
//! by `prove()`.
#![allow(clippy::print_stderr)]

use std::process::Command;

use common::constants::RAM_START_ADDRESS;
use jolt_compiler::module::Module;
use jolt_compiler::{Op, VerifierOp};
use jolt_compute::link;
use jolt_cpu::CpuBackend;
use jolt_field::Fr;
use jolt_host::{BytecodePreprocessing, Program};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_transcript::MockTranscript;
use jolt_verifier::{verify, JoltVerifyingKey, OneHotConfig, ProverConfig, ReadWriteConfig};
use jolt_witness::PolynomialId;
use jolt_zkvm::prove::prove;

type MockPCS = MockCommitmentScheme<Fr>;

/// Generate a `Module<PolynomialId>` by running the ground-truth example
/// with the given trace parameters.
fn build_protocol_module(
    log_t: usize,
    log_k_bytecode: usize,
    log_k_ram: usize,
) -> Module<PolynomialId> {
    let tmp_path = format!(
        "/tmp/jolt_muldiv_e2e_{log_t}_{log_k_bytecode}_{log_k_ram}.jolt"
    );

    let output = Command::new("cargo")
        .args([
            "run",
            "--example",
            "jolt_core_module",
            "-p",
            "jolt-compiler",
            "-q",
            "--",
            "--log-t",
            &log_t.to_string(),
            "--log-k-bytecode",
            &log_k_bytecode.to_string(),
            "--log-k-ram",
            &log_k_ram.to_string(),
            "--emit",
            &tmp_path,
        ])
        .output()
        .expect("failed to run jolt_core_module example");

    assert!(
        output.status.success(),
        "jolt_core_module failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = std::fs::read(&tmp_path).expect("failed to read protocol binary");
    Module::from_bytes(&bytes)
}

/// Truncate a module to only include Stage 1 (Outer Spartan).
///
/// Removes all prover ops from `BeginStage { index: 1 }` onward and
/// all verifier ops from the second `BeginStage` onward. This allows
/// testing Stage 1 in isolation without needing virtual polynomials
/// that Stage 2+ depends on.
fn truncate_to_stage1(module: &mut Module<PolynomialId>) {
    // Prover: cut at BeginStage { index: 1 }
    if let Some(pos) = module
        .prover
        .ops
        .iter()
        .position(|op| matches!(op, Op::BeginStage { index: 1 }))
    {
        module.prover.ops.truncate(pos);
    }

    // Verifier: cut at the second BeginStage
    let mut stage_count = 0;
    if let Some(pos) = module.verifier.ops.iter().position(|op| {
        if matches!(op, VerifierOp::BeginStage) {
            stage_count += 1;
        }
        stage_count > 1
    }) {
        module.verifier.ops.truncate(pos);
    }
}

/// Full end-to-end: trace real muldiv → prove → verify (Stage 1 only).
#[test]
#[ignore = "requires muldiv-guest build (run with --ignored)"]
fn muldiv_prove_verify() {
    // 1. Compile + decode + trace the muldiv guest
    let mut program = Program::new("muldiv-guest");
    let (bytecode_raw, _init_mem, _program_size, entry_address) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, trace, _, io_device) = program.trace(&inputs, &[], &[]);

    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, entry_address);
    let memory_layout = &io_device.memory_layout;

    // 2. Derive trace parameters
    let trace_length = trace.len().next_power_of_two();
    let log_t = trace_length.trailing_zeros() as usize;
    let bytecode_k = bytecode.code_size;
    let log_k_bytecode = bytecode_k.trailing_zeros() as usize;
    let ram_k = 1usize << 20; // 1M entries
    let log_k_ram = ram_k.trailing_zeros() as usize;

    eprintln!(
        "muldiv trace: {} cycles (padded to {trace_length}), log_t={log_t}, bytecode_k={bytecode_k} (log={log_k_bytecode}), ram_k={ram_k} (log={log_k_ram})",
        trace.len(),
    );

    // 3. Build protocol module and truncate to Stage 1
    let mut module = build_protocol_module(log_t, log_k_bytecode, log_k_ram);
    truncate_to_stage1(&mut module);

    let backend = CpuBackend;
    let executable = link::<PolynomialId, CpuBackend, Fr>(module, &backend);

    // 4. Build ProverConfig
    let one_hot = OneHotConfig::new(log_t);
    let rw_config = ReadWriteConfig::new(log_t, log_k_ram);
    let config = ProverConfig {
        trace_length,
        ram_k,
        bytecode_k,
        one_hot_config: one_hot,
        rw_config,
        memory_start: RAM_START_ADDRESS,
        memory_end: RAM_START_ADDRESS + ram_k as u64,
        entry_address,
        io_hash: [0u8; 32],
    };

    // 5. Build trace data and prove
    let trace_data = jolt_zkvm::prove::TraceData {
        trace: &trace,
        bytecode: &bytecode,
        memory_layout,
    };
    let pcs_setup = ();
    let mut transcript = MockTranscript::<Fr>::default();

    let proof = prove::<_, _, _, _, MockPCS>(
        &executable,
        &trace_data,
        &backend,
        &pcs_setup,
        &mut transcript,
        config,
    );

    eprintln!(
        "proof: {} stages, {} commitments",
        proof.stage_proofs.len(),
        proof.commitments.len(),
    );

    // 6. Verify
    let vk = JoltVerifyingKey::<PolynomialId, Fr, MockPCS>::from_module(
        &executable.module,
        (),
    );
    verify(&vk, &proof, &[0u8; 32]).expect("proof should verify");
}
