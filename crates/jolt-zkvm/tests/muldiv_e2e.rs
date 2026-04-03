//! End-to-end test: real muldiv guest → protocol.jolt → prove → verify.
//!
//! Exercises the full pipeline with a real RISC-V program:
//!   1. Compile + trace the muldiv guest via jolt-host
//!   2. Generate protocol.jolt from the ground-truth example (matching params)
//!   3. Link the Module to the CPU backend
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
use jolt_field::{Field, Fr};
use jolt_host::{BytecodePreprocessing, Program};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_r1cs::R1csKey;
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::{
    verify, JoltVerifyingKey, OneHotConfig, ProverConfig, ReadWriteConfig, TRANSCRIPT_LABEL,
};
use jolt_zkvm::prove::prove;

type MockPCS = MockCommitmentScheme<Fr>;

/// Generate a `Module` by running the ground-truth example
/// with the given trace parameters.
fn build_protocol_module(log_t: usize, log_k_bytecode: usize, log_k_ram: usize) -> Module {
    let tmp_path = format!("/tmp/jolt_muldiv_e2e_{log_t}_{log_k_bytecode}_{log_k_ram}.jolt");

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
fn truncate_to_stage1(module: &mut Module) {
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
    let executable = link::<CpuBackend, Fr>(module, &backend);

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
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        heap_size: memory_layout.heap_size,
        inputs: io_device.inputs.clone(),
        outputs: io_device.outputs.clone(),
        panic: io_device.panic,
    };

    // 5. Build trace data and prove
    let trace_data = jolt_zkvm::prove::TraceData {
        trace: &trace,
        bytecode: &bytecode,
        memory_layout,
    };
    let pcs_setup = ();
    let mut transcript = Blake2bTranscript::<Fr>::new(TRANSCRIPT_LABEL);

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

    // Debug: dump stage proof structure
    for (si, sp) in proof.stage_proofs.iter().enumerate() {
        eprintln!(
            "  stage {si}: {} round_polys, {} evals",
            sp.round_polys.round_polynomials.len(),
            sp.evals.len(),
        );
        // Check if the claimed sum matches round_poly_1(0) + round_poly_1(1)
        if sp.round_polys.round_polynomials.len() > 1 {
            let rp = &sp.round_polys.round_polynomials[1]; // first remaining round poly
            let sum = rp.evaluate(Fr::from_u64(0)) + rp.evaluate(Fr::from_u64(1));
            eprintln!(
                "    rp[1](0)+rp[1](1) = {sum}    (this is what sumcheck checks against combined_claim)"
            );
            eprintln!("    evals[0] (uniskip_eval) = {}", sp.evals[0]);
        }
    }

    // 6. Verify
    let r1cs_key = R1csKey::new(
        jolt_r1cs::constraints::rv64::rv64_constraints::<Fr>(),
        trace_length,
    );
    let vk = JoltVerifyingKey::<Fr, MockPCS>::new(&executable.module, (), r1cs_key);
    verify(&vk, &proof, &[0u8; 32]).expect("proof should verify");
}
