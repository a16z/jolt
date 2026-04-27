//! End-to-end smoke test for the BN254 Fr coprocessor v2 SDK guest.
//!
//! Drives `bn254-fr-poseidon2-sdk-guest` through the full modular prover
//! (`jolt_zkvm::prove::prove`) — mirrors
//! `jolt-bench/src/stacks/modular.rs::build_modular_setup` with three
//! FR-specific changes:
//!   1. `Program::trace_two_pass_advice` runs the compute_advice ELF first
//!      so the FR advice tape is populated before the proof trace,
//!   2. `extract_trace` and `DerivedSource::with_field_reg` are threaded
//!      with the drained `FieldRegEvent` stream,
//!   3. Bytecode flags + lookup-table indices route through the modular
//!      `static_circuit_flags` / `lookup_table_kind` (jolt-core's `Flags`
//!      impl panics on FR `Instruction` variants),
//!   4. `FieldRegInc` dense polynomial is overwritten post-`finish()` with
//!      `new − old` deltas (CycleInput::dense uses `i128`, can't hold a
//!      256-bit Fr delta).
//!
//! ## Status (as of 2026-04-24)
//!
//! Trace + witness extraction succeed (~36k real cycles, ~16k FR events,
//! padded to log_t=16). The modular prover starts but is **SIGKILL'd**
//! mid-prove on macOS — likely OOM around the FR Twist commit phase
//! (FieldRegRa one-hot at K_FR=16 plus the K_FR×T derived polys for
//! Val/Wa/RaRs1/RaRs2 add ≥128 MB of ModFr scratch on top of the rest).
//!
//! The test is `#[ignore]` by default. Once the FR prover memory profile
//! is tightened (or the smoke is run on a higher-memory host), this test
//! will produce the first real prover wall-time number for an FR program.
#![allow(non_snake_case, clippy::print_stderr, clippy::too_many_arguments)]

use std::process::Command;
use std::time::Instant;

use common::constants::{ONEHOT_CHUNK_THRESHOLD_LOG_T, RAM_START_ADDRESS};
use jolt_compiler::module::Module;
use jolt_compiler::{Op, VerifierOp};
use jolt_compute::link;
use jolt_cpu::CpuBackend;
use jolt_dory::DoryScheme;
use jolt_host::{
    extract_trace, lookup_table_kind, static_circuit_flags, static_instruction_flags,
    BytecodePreprocessing, CycleRow, Program as HostProgram,
};
use jolt_instructions::flags::InstructionFlags as ModInstructionFlags;
use jolt_instructions::{InterleavedBitsMarker as ModInterleavedBits, LookupTableKind};
use jolt_r1cs::{constraints::rv64, R1csKey, R1csSource};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::{OneHotConfig, ProverConfig, ReadWriteConfig, TRANSCRIPT_LABEL};
use jolt_witness::bytecode_raf::{BytecodeData, BytecodeEntry};
use jolt_witness::derived::{
    DerivedSource, FieldRegConfig, InstructionFlags, RamConfig, RegisterAccessData,
};
use jolt_witness::preprocessed::PreprocessedSource;
use jolt_witness::provider::ProverData;
use jolt_witness::{
    field_reg_inc_polynomial, FieldRegEvent, PolynomialConfig, PolynomialId, Polynomials,
};
use jolt_zkvm::prove::prove as modular_prove;
use jolt_zkvm::runtime::prefix_suffix::LookupTraceData;
use num_traits::{One, Zero};

type ModFr = jolt_field::Fr;

// `field_reg_inc_polynomial` (re-exported from `jolt_witness::field_reg`) is
// the canonical helper. Used below post-`polys.finish()` to populate
// `FieldRegInc`. See specs/fr-v2-audit.md C11.

/// Drop-in copy of the bench's `build_protocol_module` — shells out to the
/// `jolt_core_module` example to emit the protocol binary, then loads it.
fn build_protocol_module(log_t: usize, log_k_bytecode: usize, log_k_ram: usize) -> Module {
    let tmp_path = format!(
        "/tmp/jolt_p2_sdk_module_{log_t}_{log_k_bytecode}_{log_k_ram}.jolt"
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

fn truncate_after_stage(module: &mut Module, num_stages: usize) {
    let mut in_final_stage = false;
    if let Some(pos) = module.prover.ops.iter().position(|op| {
        if let Op::BeginStage { index } = op {
            if *index >= num_stages {
                return true;
            }
            in_final_stage = *index + 1 == num_stages;
        }
        if in_final_stage {
            if let Op::BatchRoundBegin { batch, .. } = op {
                let bdef = &module.prover.batched_sumchecks[batch.0];
                if bdef.instances.iter().all(|inst| inst.phases.is_empty()) {
                    return true;
                }
            }
        }
        false
    }) {
        module.prover.ops.truncate(pos);
    }

    let mut stage_count = 0;
    if let Some(pos) = module.verifier.ops.iter().position(|op| {
        if matches!(op, VerifierOp::BeginStage) {
            stage_count += 1;
        }
        stage_count > num_stages
    }) {
        module.verifier.ops.truncate(pos);
    }
}

#[test]
#[ignore = "exercises the full FR modular prover; run with --no-capture"]
fn poseidon2_sdk_modular_prove_smoke() {
    let s0: [u64; 4] = [1, 0, 0, 0];
    let s1: [u64; 4] = [2, 0, 0, 0];
    let s2: [u64; 4] = [3, 0, 0, 0];
    let inputs =
        postcard::to_stdvec(&(s0, s1, s2)).expect("postcard encode poseidon2 inputs");

    let mut program = HostProgram::new("bn254-fr-poseidon2-sdk-guest");
    // jolt-host's `Program::new` defaults are stack=4 KiB, heap=32 MiB —
    // ignoring the SDK guest's `#[jolt::provable]` macro args (stack=64K,
    // heap=128K). The 32 MiB default produces ram_k=2^22 in the modular
    // prover, and `derived::ram_val/ram_combined_ra/ram_ra_indicator`
    // each allocate `Vec<F>(k * T)` = ~16 GB, OOMing the kernel. Use the
    // macro's actual values so ram_k stays tractable.
    let _ = program
        .set_stack_size(65536)
        .set_heap_size(2 * 1024 * 1024)
        .set_max_input_size(8192);

    // Two-pass trace must run BEFORE decode(): decode() calls build() (no
    // features), which populates `self.elf` and short-circuits the
    // subsequent `build_with_features("compute_advice")` so the advice
    // ELF never gets built and Pass 2 panics with "Failed to read from
    // advice tape".
    let trace_start = Instant::now();
    let (_, trace, final_memory, io_device, tracer_fr_events) =
        program.trace_two_pass_advice(&inputs, &[], &[]);
    let trace_ms = trace_start.elapsed().as_millis();

    let (bytecode_raw, init_mem, _program_size, entry_address) = program.decode();

    // Convert tracer FieldRegEvent → witness FieldRegEvent (identical fields,
    // distinct types to avoid pulling tracer into jolt-witness).
    let fr_events: Vec<FieldRegEvent> = tracer_fr_events
        .iter()
        .map(|e| FieldRegEvent {
            cycle: e.cycle_index,
            slot: e.slot,
            old: e.old,
            new: e.new,
        })
        .collect();

    let real_cycles = trace
        .iter()
        .filter(|c| !matches!(c, tracer::instruction::Cycle::NoOp))
        .count();
    eprintln!(
        "[poseidon2-sdk-e2e] trace: {} real cycles ({} total, padded to 2^{}), {} FR events, traced in {} ms",
        real_cycles,
        trace.len(),
        trace.len().next_power_of_two().trailing_zeros(),
        fr_events.len(),
        trace_ms,
    );

    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, entry_address);
    let memory_layout = io_device.memory_layout.clone();
    let trace_length = trace.len().next_power_of_two();
    let log_t = trace_length.trailing_zeros() as usize;
    let bytecode_k = bytecode.code_size;
    let log_k_bytecode = bytecode_k.trailing_zeros() as usize;

    let lowest_addr = memory_layout.get_lowest_address();
    let highest_addr = RAM_START_ADDRESS + memory_layout.get_total_memory_size();

    // Tight ram_K: max(remapped trace addresses, bytecode end), pow-2 ceiling.
    let min_bc_addr = init_mem.iter().map(|(a, _)| *a).min().unwrap_or(lowest_addr);
    let max_bc_addr = init_mem.iter().map(|(a, _)| *a).max().unwrap_or(lowest_addr)
        + (common::constants::BYTES_PER_INSTRUCTION as u64 - 1);
    let num_bc_words = max_bc_addr.div_ceil(8) - min_bc_addr / 8 + 1;
    let bytecode_start_remapped = if min_bc_addr >= lowest_addr && min_bc_addr != 0 {
        (min_bc_addr - lowest_addr) / 8
    } else {
        0
    };
    let trace_max_remapped: u64 = trace
        .iter()
        .filter_map(|cycle| {
            let addr = cycle.ram_access_address()?;
            if addr == 0 || addr < lowest_addr {
                None
            } else {
                Some((addr - lowest_addr) / 8)
            }
        })
        .max()
        .unwrap_or(0);
    let ram_k_min = trace_max_remapped.max(bytecode_start_remapped + num_bc_words + 1);
    let ram_k = (ram_k_min as usize).next_power_of_two();
    let log_k_ram = ram_k.trailing_zeros() as usize;

    eprintln!(
        "[poseidon2-sdk-e2e] params: log_t={log_t}, log_k_bytecode={log_k_bytecode}, log_k_ram={log_k_ram}, ram_k={ram_k}"
    );
    eprintln!(
        "[poseidon2-sdk-e2e] addr probe: lowest=0x{lowest_addr:x}, highest=0x{highest_addr:x}, \
         trace_max_remapped={trace_max_remapped}, bytecode_start_remapped={bytecode_start_remapped}, \
         num_bc_words={num_bc_words}, ram_k_min={ram_k_min}, K×T (Fr)={} GB",
        (ram_k as u64 * trace_length as u64 * 32) / (1 << 30)
    );

    // PCS setup sized for `log_t` (no ceiling — this test runs only at the
    // measured trace length, unlike the bench's --log-t argument).
    let max_log_k_chunk = if log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T { 4 } else { 8 };
    let pcs_setup_start = Instant::now();
    let pcs_prover_setup = DoryScheme::setup_prover(max_log_k_chunk + log_t);
    eprintln!(
        "[poseidon2-sdk-e2e] PCS setup: {} ms",
        pcs_setup_start.elapsed().as_millis()
    );

    let mut module = build_protocol_module(log_t, log_k_bytecode, log_k_ram);
    truncate_after_stage(&mut module, 8);
    let backend = CpuBackend;
    let executable = link::<CpuBackend, ModFr>(module, &backend);

    let one_hot = OneHotConfig::new(log_t);
    let log_k_chunk = one_hot.log_k_chunk as usize;
    let rw_config = ReadWriteConfig::new(log_t, log_k_ram);

    let remap = |addr: u64| ((addr - lowest_addr) / 8) as usize;
    let config = ProverConfig {
        trace_length,
        ram_k,
        bytecode_k,
        one_hot_config: one_hot,
        rw_config,
        memory_start: RAM_START_ADDRESS,
        memory_end: highest_addr,
        entry_address,
        io_hash: [0u8; 32],
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        heap_size: memory_layout.heap_size,
        inputs: io_device.inputs.clone(),
        outputs: {
            let mut out = io_device.outputs.clone();
            out.truncate(out.iter().rposition(|&b| b != 0).map_or(0, |pos| pos + 1));
            out
        },
        panic: io_device.panic,
        ram_lowest_address: lowest_addr,
        input_word_offset: remap(memory_layout.input_start),
        output_word_offset: remap(memory_layout.output_start),
        panic_word_offset: remap(memory_layout.panic),
        termination_word_offset: remap(memory_layout.termination),
    };

    let poly_config = PolynomialConfig::new(log_k_chunk, 128, log_k_bytecode, log_k_ram);
    let matrices = rv64::rv64_constraints::<ModFr>();
    let r1cs_key = R1csKey::new(matrices, trace_length);

    let (cycle_inputs, r1cs_witness, instruction_flag_data) = extract_trace::<_, ModFr>(
        &trace,
        trace_length,
        &bytecode,
        &memory_layout,
        r1cs_key.num_vars_padded,
        &fr_events,
    );

    // Per-cycle register/lookup data.
    let mut rd_indices = vec![None; trace_length];
    let mut rs1_indices = vec![None; trace_length];
    let mut rs2_indices = vec![None; trace_length];
    let mut lookup_keys = vec![0u128; trace_length];
    let mut table_kind_indices: Vec<Option<usize>> = vec![None; trace_length];
    let mut table_indices: Vec<Option<usize>> = vec![None; trace_length];
    let mut is_interleaved = vec![true; trace_length];
    for (t, cycle) in trace.iter().enumerate() {
        if let Some((reg, _, _)) = cycle.rd_write() {
            rd_indices[t] = Some(reg as usize);
        }
        if let Some((reg, _)) = cycle.rs1_read() {
            rs1_indices[t] = Some(reg as usize);
        }
        if let Some((reg, _)) = cycle.rs2_read() {
            rs2_indices[t] = Some(reg as usize);
        }
        lookup_keys[t] = cycle.lookup_index();
        let kind = cycle.lookup_table_index();
        if let Some(idx) = kind {
            assert!(idx < LookupTableKind::COUNT);
        }
        table_kind_indices[t] = kind;
        // Modular path: identical to `cycle.lookup_table_index()` since both
        // route through `lookup_table_kind` over the tracer Instruction. Use
        // the index directly — FR cycles return None (no integer lookup),
        // matching what the protocol module expects.
        table_indices[t] = kind;
        is_interleaved[t] = ModInterleavedBits::is_interleaved_operands(&cycle.circuit_flags());
    }
    let reg_access = RegisterAccessData {
        rd_indices,
        rs1_indices,
        rs2_indices,
    };
    let lookup_trace = LookupTraceData {
        lookup_keys,
        table_kind_indices,
        is_interleaved: is_interleaved.clone(),
    };
    let is_raf: Vec<bool> = is_interleaved.iter().map(|&b| !b).collect();
    let lookup_flags = jolt_witness::derived::LookupFlagData {
        table_indices,
        is_raf,
    };

    let mut polys = Polynomials::<ModFr>::new(poly_config);
    polys.push(&cycle_inputs);
    polys.finish();

    // Overwrite all-zero pre-allocated FieldRegInc with actual deltas.
    // CycleInput::dense uses i128 (RdInc/RamInc fit), but Fr deltas are
    // 256-bit so the dense witness slot stays at 0 during push() and is
    // populated here from the FieldRegEvent stream. Required for any
    // FR-active program — see audit C11.
    let field_reg_inc = field_reg_inc_polynomial::<ModFr>(&fr_events, trace_length);
    let _ = polys.insert(PolynomialId::FieldRegInc, field_reg_inc);

    // SDK Poseidon2 has no advice consumers — both advice slots stay zero.
    let _ = polys.insert(
        PolynomialId::UntrustedAdvice,
        vec![ModFr::zero(); trace_length],
    );
    let _ = polys.insert(
        PolynomialId::TrustedAdvice,
        vec![ModFr::zero(); trace_length],
    );

    let (initial_ram_state, final_ram_state) =
        jolt_host::ram::build_ram_states(&init_mem, &final_memory, &io_device, ram_k);

    let bc_entries: Vec<BytecodeEntry<ModFr>> = bytecode
        .bytecode
        .iter()
        .map(|instruction| {
            let instr = instruction.normalize();
            // Modular flags path — handles all opcodes including FR via
            // `with_isa_struct!`'s field arms (jolt-core's Flags impl panics
            // on FR variants).
            let circuit_flags = static_circuit_flags(instruction);
            let instr_flags = static_instruction_flags(instruction);
            let lookup_table = lookup_table_kind(instruction).map(|k| k as usize);
            BytecodeEntry {
                address: jolt_field::Field::from_u64(instr.address as u64),
                imm: jolt_field::Field::from_i128(instr.operands.imm),
                circuit_flags: circuit_flags.to_vec(),
                rd: instr.operands.rd,
                rs1: instr.operands.rs1,
                rs2: instr.operands.rs2,
                lookup_table,
                is_interleaved: ModInterleavedBits::is_interleaved_operands(&circuit_flags),
                is_branch: instr_flags[ModInstructionFlags::Branch],
                left_is_rs1: instr_flags[ModInstructionFlags::LeftOperandIsRs1Value],
                left_is_pc: instr_flags[ModInstructionFlags::LeftOperandIsPC],
                right_is_rs2: instr_flags[ModInstructionFlags::RightOperandIsRs2Value],
                right_is_imm: instr_flags[ModInstructionFlags::RightOperandIsImm],
                is_noop: instr_flags[ModInstructionFlags::IsNoop],
            }
        })
        .collect();

    let mut pc_indices = Vec::with_capacity(trace_length);
    for cycle in &trace {
        pc_indices.push(bytecode.get_pc(cycle));
    }
    pc_indices.resize(trace_length, 0);

    let bytecode_data = BytecodeData {
        pc_indices,
        entries: bc_entries,
        entry_index: bytecode.entry_bytecode_index(),
        num_lookup_tables: jolt_compiler::params::NUM_LOOKUP_TABLES,
    };

    // Build per-cycle FR bytecode snapshots — required by FieldRegConfig
    // so the FR Twist materializers (FieldRegRaRs1/2, Wa, Val, gather)
    // produce non-zero outputs. Uses the same `fr_meta()` path as
    // extract_trace.
    let fr_bytecode: Vec<jolt_witness::FrCycleBytecode> =
        trace.iter().map(|c| c.fr_meta()).collect();
    let mut fr_bytecode_padded = fr_bytecode;
    fr_bytecode_padded.resize(trace_length, jolt_witness::FrCycleBytecode::default());

    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let derived = DerivedSource::new(&r1cs_witness, trace_length, r1cs_key.num_vars_padded)
        .with_ram(RamConfig {
            ram_k: config.ram_k,
            lowest_addr: config.ram_lowest_address,
            initial_state: initial_ram_state.clone(),
            final_state: final_ram_state,
        })
        .with_instruction_flags(InstructionFlags {
            is_noop: instruction_flag_data.is_noop,
            left_is_rs1: instruction_flag_data.left_is_rs1,
            left_is_pc: instruction_flag_data.left_is_pc,
            right_is_rs2: instruction_flag_data.right_is_rs2,
            right_is_imm: instruction_flag_data.right_is_imm,
        })
        .with_register_access(reg_access)
        .with_lookup_flags(lookup_flags)
        .with_field_reg(FieldRegConfig {
            bytecode: fr_bytecode_padded,
            events: fr_events,
        });

    let mut preprocessed = PreprocessedSource::new();
    preprocessed.populate_ram(&config, &initial_ram_state);
    {
        use jolt_field::Field;
        let k = bytecode_data.entries.len();
        let pc_idx_poly: Vec<ModFr> = bytecode_data
            .pc_indices
            .iter()
            .map(|&i| ModFr::from_u64(i as u64))
            .collect();
        preprocessed.insert(PolynomialId::BytecodePcIndex, pc_idx_poly);
        let pc_0 = bytecode_data.pc_indices[0];
        let mut entry_trace = vec![ModFr::zero(); k];
        entry_trace[pc_0] = ModFr::one();
        preprocessed.insert(PolynomialId::BytecodeEntryTrace, entry_trace);
        let mut entry_expected = vec![ModFr::zero(); k];
        entry_expected[bytecode_data.entry_index] = ModFr::one();
        preprocessed.insert(PolynomialId::BytecodeEntryExpected, entry_expected);
        bytecode_data.populate_preprocessed(&mut preprocessed);
    }

    let mut provider = ProverData::new(&mut polys, r1cs, derived, preprocessed)
        .with_lookup_trace(lookup_trace);
    let mut transcript = Blake2bTranscript::<ModFr>::new(TRANSCRIPT_LABEL);

    let prove_start = Instant::now();
    let zkvm_proof = modular_prove::<_, _, _, DoryScheme>(
        &executable,
        &mut provider,
        &CpuBackend,
        &pcs_prover_setup,
        &mut transcript,
        config,
    );
    let prove_ms = prove_start.elapsed().as_millis();

    eprintln!("\n=== Poseidon2 BN254 t=3 modular prover (SDK guest) ===");
    eprintln!("  trace_length      : {trace_length} (log_t = {log_t})");
    eprintln!("  real cycles       : {real_cycles}");
    eprintln!("  FR events         : {}", zkvm_proof.commitments.len());
    eprintln!("  prove_ms          : {prove_ms} ms");
    eprintln!("  proof commitments : {}", zkvm_proof.commitments.len());
    eprintln!();
}
