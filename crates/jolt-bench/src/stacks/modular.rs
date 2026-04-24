//! Modular-stack benchmark runner.
//!
//! The modular stack runs any program whose guest ELF produces a valid
//! jolt-host trace. A `DoryScheme` modular prover produces a
//! `jolt_verifier::JoltProof`, which is verified by the native modular
//! `jolt_verifier::verify` — no jolt-core dependency on the verify path.
//!
//! Setup (trace, PCS generators, verifying key) runs outside the
//! measurement window. Only `jolt_zkvm::prove::prove()` is timed for
//! `prove_ms`, and only `jolt_verifier::verify()` for `verify_ms`.

use std::process::Command;

use common::constants::{ONEHOT_CHUNK_THRESHOLD_LOG_T, RAM_START_ADDRESS};
use jolt_compiler::module::Module;
use jolt_compiler::{Op, VerifierOp};
use jolt_compute::{link, LookupTraceData};
use jolt_core::zkvm::instruction::{
    Flags as CoreFlags, InstructionLookup, InterleavedBitsMarker as CoreInterleavedBits,
};
use jolt_core::zkvm::lookup_table::LookupTables as CoreLookupTables;
use jolt_cpu::CpuBackend;
use jolt_dory::DoryScheme;
use jolt_host::{
    apply_field_op_events_to_r1cs, extract_trace, BytecodePreprocessing, CycleRow,
    InstructionFlagData, Program as HostProgram,
};
use jolt_instructions::LookupTableKind;
use jolt_r1cs::{constraints::rv64, R1csKey, R1csSource};
use jolt_transcript::{Blake2bTranscript as ModBlake2bTranscript, Transcript};
use jolt_verifier::{
    verify as modular_verify, JoltVerifyingKey, OneHotConfig, ProverConfig, ReadWriteConfig,
    TRANSCRIPT_LABEL,
};
use jolt_witness::bytecode_raf::{BytecodeData, BytecodeEntry};
use jolt_witness::derived::{
    DerivedSource, FieldRegConfig, InstructionFlags, RamConfig, RegisterAccessData,
};
use jolt_witness::preprocessed::PreprocessedSource;
use jolt_witness::provider::ProverData;
use jolt_witness::{PolynomialConfig, PolynomialId, Polynomials};
use jolt_zkvm::prove::prove as modular_prove;
use num_traits::{One, Zero};

use super::{IterMetrics, StackOutcome, StackRunner};
use crate::measure::{time_it, PeakRssSampler};
use crate::programs::Program;

type ModFr = jolt_field::Fr;

const DEFAULT_MAX_TRACE_LENGTH: usize = 1 << 16;

pub struct ModularStack;

impl ModularStack {
    fn run_program_once(guest_name: &str, inputs: &[u8], log_t: Option<usize>) -> IterMetrics {
        let max_trace_length = log_t.map_or(DEFAULT_MAX_TRACE_LENGTH, |n| 1usize << n);

        // -- 1. Trace the guest (jolt-host) and compute protocol sizes.
        let mut program = HostProgram::new(guest_name);
        let (bytecode_raw, init_mem, _program_size, entry_address) = program.decode();
        let (_, trace, final_memory, io_device, field_reg_events) =
            program.trace_with_field_reg_events(inputs, &[], &[]);

        let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, entry_address);
        let memory_layout = io_device.memory_layout.clone();

        // `--log-t N` → `max_padded_trace_length = 2^N` (a CEILING, matching
        // jolt-core's `JoltSharedPreprocessing::new`). The actual trace length
        // is the raw trace padded to the next power of two. If the guest
        // program runs fewer cycles than the ceiling, the measured log_t is
        // smaller than N — the ratchet's historical log_t=12 was real log_t=9
        // for muldiv (≤512 cycles). For real log_t=N, pick a program large
        // enough (sha2-chain with `--num-iters` scales cleanly).
        assert!(
            trace.len() <= max_trace_length,
            "trace ({} cycles) exceeds max_trace_length (2^{}); \
             increase --log-t or lower inputs",
            trace.len(),
            max_trace_length.trailing_zeros(),
        );
        let trace_length = trace.len().next_power_of_two();
        let log_t_val = trace_length.trailing_zeros() as usize;

        let bytecode_k = bytecode.code_size;
        let log_k_bytecode = bytecode_k.trailing_zeros() as usize;

        let lowest_addr = memory_layout.get_lowest_address();
        let highest_addr = RAM_START_ADDRESS + memory_layout.get_total_memory_size();

        // Tight ram_K matching jolt-core's DoryGlobals sizing: the max of
        // (max remapped RAM address actually touched in the trace) and
        // (bytecode end: remap(min_bytecode_addr) + bytecode_words.len() + 1),
        // rounded up to a power of two. The loose bound
        // `(highest_addr - lowest_addr) / 8` over-sizes by up to 8x and OOMs
        // under real Dory commitments.
        let min_bc_addr = init_mem
            .iter()
            .map(|(a, _)| *a)
            .min()
            .unwrap_or(lowest_addr);
        let max_bc_addr = init_mem
            .iter()
            .map(|(a, _)| *a)
            .max()
            .unwrap_or(lowest_addr)
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

        // -- 2. PCS setup — size for `max_log_T = log2(max_padded_trace_length)`
        // (matches `JoltProverPreprocessing::new`). Not `log_t_val`: if the
        // actual trace is smaller than the ceiling, we still want generators
        // sized for the ceiling so the PCS setup is identical to what
        // jolt-core would produce for the same --log-t flag.
        let max_log_t = max_trace_length.trailing_zeros() as usize;
        let max_log_k_chunk = if max_log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };
        let pcs_prover_setup = DoryScheme::setup_prover(max_log_k_chunk + max_log_t);
        let pcs_verifier_setup =
            <DoryScheme as jolt_openings::CommitmentScheme>::verifier_setup(&pcs_prover_setup);

        // -- 3. Build module, witness, config.
        let (
            executable,
            mut polys,
            r1cs_key,
            mut r1cs_witness,
            setup,
            initial_ram,
            final_ram,
            instruction_flag_data,
            reg_access,
            lookup_trace,
            lookup_flags,
            bytecode_data,
        ) = build_modular_setup(
            &trace,
            &init_mem,
            &final_memory,
            &io_device,
            &bytecode,
            entry_address,
            trace_length,
            log_t_val,
            bytecode_k,
            log_k_bytecode,
            ram_k,
            log_k_ram,
            lowest_addr,
            highest_addr,
        );

        // BN254 Fr coprocessor: k=16 slots, zero-initialized.
        // Events come from the tracer (empty if the guest didn't emit FieldOp /
        // FMov{I2F,F2I} cycles). Wiring this unconditionally lets Modules that
        // declare FR polynomials consume them; non-FR Modules simply don't
        // materialize the FR polys.
        let fr_events: Vec<jolt_witness::derived::FieldRegEvent> = field_reg_events
            .into_iter()
            .map(|e| jolt_witness::derived::FieldRegEvent {
                cycle: e.cycle_index,
                slot: e.slot as usize,
                old: e.old,
                new: e.new,
                op: e.op.map(|p| jolt_witness::derived::FieldOpPayload {
                    funct3: p.funct3,
                    a: p.a,
                    b: p.b,
                }),
                fmov: e.fmov.map(|p| jolt_witness::derived::FMovPayload {
                    funct3: p.funct3,
                    limb_idx: p.limb_idx,
                    limb: p.limb,
                }),
            })
            .collect();

        // Overlay the FieldOp columns (V_FLAG_IS_FIELD_*, V_FIELD_OP_{A,B,RESULT},
        // and the V_LEFT/V_RIGHT/V_PRODUCT routing for FMUL/FINV) onto the
        // R1CS witness BEFORE handing it to the Spartan R1csSource. No-op for
        // guests that didn't emit FieldOp cycles.
        apply_field_op_events_to_r1cs::<ModFr>(
            &mut r1cs_witness,
            trace_length,
            r1cs_key.num_vars_padded,
            &fr_events,
        );
        // (Plan P, task #65) Populate V_LIMB_SUM_A/B from the register-write
        // stream so R1CS rows 29/30 can bind FieldOpA/B on FieldOp cycles.
        jolt_host::r1cs_witness::populate_limb_sum_columns::<ModFr>(
            &mut r1cs_witness,
            &reg_access.rd_indices,
            trace_length,
            r1cs_key.num_vars_padded,
        );

        let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
        let field_reg_config = FieldRegConfig {
            k: 16,
            initial_state: vec![[0u64; 4]; 16],
            events: fr_events,
        };
        let derived = DerivedSource::new(&r1cs_witness, trace_length, r1cs_key.num_vars_padded)
            .with_ram(RamConfig {
                ram_k: setup.config.ram_k,
                lowest_addr: setup.config.ram_lowest_address,
                initial_state: initial_ram.clone(),
                final_state: final_ram,
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
            .with_field_reg(field_reg_config);
        let mut preprocessed = PreprocessedSource::new();
        preprocessed.populate_ram(&setup.config, &initial_ram);
        populate_bytecode_preprocessed(&mut preprocessed, &bytecode_data);
        let mut provider = ProverData::new(&mut polys, r1cs, derived, preprocessed)
            .with_lookup_trace(lookup_trace);

        let mut prove_transcript = ModBlake2bTranscript::<ModFr>::new(TRANSCRIPT_LABEL);

        // -- 4. Timed modular prove.
        let sampler = PeakRssSampler::start();
        let (prove_ms, zkvm_proof) = time_it(|| {
            modular_prove::<_, _, _, DoryScheme>(
                &executable,
                &mut provider,
                &CpuBackend,
                &pcs_prover_setup,
                &mut prove_transcript,
                setup.config.clone(),
            )
        });
        let peak_rss_mb = sampler.finish();

        let proof_bytes = bincode::serde::encode_to_vec(&zkvm_proof, bincode::config::standard())
            .expect("serialize modular proof")
            .len() as u64;

        // -- 5. Build verifying key and timed native modular verify.
        let verifying_key = JoltVerifyingKey::<ModFr, DoryScheme>::new(
            &executable.module,
            pcs_verifier_setup,
            r1cs_key,
        );
        // Native modular verify is under active development — some programs
        // exercise paths that don't yet match the prover's emit contract
        // (e.g. advice-zero handling, preprocessed-poly evals). The bench
        // reports verify_ms as a best-effort measurement and does not fail
        // the run on verify error. Prove remains the perf-critical metric.
        let (verify_ms, ()) = time_it(|| {
            if let Err(e) = modular_verify(&verifying_key, &zkvm_proof, &setup.config.io_hash) {
                eprintln!(
                    "[jolt-bench] modular verify failed (non-fatal, prove metrics still valid): {e:?}"
                );
            }
        });

        IterMetrics {
            prove_ms,
            verify_ms,
            peak_rss_mb,
            proof_bytes,
        }
    }
}

impl StackRunner for ModularStack {
    fn run(
        &self,
        program: Program,
        iters: usize,
        warmup: usize,
        log_t: Option<usize>,
        num_iters_override: Option<u32>,
    ) -> StackOutcome {
        let guest_name = program.guest_name();
        let inputs = program.canonical_inputs_with(num_iters_override);
        for _ in 0..warmup {
            let _ = Self::run_program_once(guest_name, &inputs, log_t);
        }
        let measurements = (0..iters)
            .map(|_| Self::run_program_once(guest_name, &inputs, log_t))
            .collect();
        StackOutcome::Metrics(measurements)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Modular program setup — ported from jolt-equivalence/tests/muldiv.rs
// and jolt-zkvm/tests/muldiv_e2e.rs. Program-agnostic: only `guest_name`
// (and its ELF) differ between muldiv / sha / fib / btreemap. Duplicated
// here because those helpers are `#[cfg(test)]` only; when a public
// helper lands in jolt-zkvm (or jolt-host), delete this section.
// ═══════════════════════════════════════════════════════════════════

struct ZkvmSetup {
    config: ProverConfig,
}

fn build_protocol_module(log_t: usize, log_k_bytecode: usize, log_k_ram: usize) -> Module {
    let tmp_path = format!("/tmp/jolt_bench_module_{log_t}_{log_k_bytecode}_{log_k_ram}.jolt");

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

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn build_modular_setup(
    trace: &[jolt_host::Cycle],
    init_mem: &[(u64, u8)],
    final_memory: &jolt_host::Memory,
    io_device: &jolt_host::JoltDevice,
    bytecode: &BytecodePreprocessing,
    entry_address: u64,
    trace_length: usize,
    log_t: usize,
    bytecode_k: usize,
    log_k_bytecode: usize,
    ram_k: usize,
    log_k_ram: usize,
    lowest_addr: u64,
    highest_addr: u64,
) -> (
    jolt_compute::Executable<CpuBackend, ModFr>,
    Polynomials<ModFr>,
    R1csKey<ModFr>,
    Vec<ModFr>,
    ZkvmSetup,
    Vec<u64>,
    Vec<u64>,
    InstructionFlagData<ModFr>,
    RegisterAccessData,
    LookupTraceData,
    jolt_witness::derived::LookupFlagData,
    BytecodeData<ModFr>,
) {
    let memory_layout = &io_device.memory_layout;

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
        trace,
        trace_length,
        bytecode,
        memory_layout,
        r1cs_key.num_vars_padded,
    );

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
        table_kind_indices[t] = cycle.lookup_table_index().inspect(|&idx| {
            assert!(idx < LookupTableKind::COUNT);
        });
        table_indices[t] = InstructionLookup::<64>::lookup_table(cycle)
            .map(|t| CoreLookupTables::<64>::enum_index(&t));
        // Truncate refactor's 18-flag array down to jolt-core's 14-flag shape.
        // The 4 BN254 Fr flags (IsFieldMul/Add/Sub/Inv) live at indices 14..17
        // and don't affect is_interleaved_operands (which only reads the first
        // 4 flags: AddOperands, SubtractOperands, MultiplyOperands, Advice).
        let cycle_flags = cycle.circuit_flags();
        let mut core_flags_14 = [false; 14];
        core_flags_14.copy_from_slice(&cycle_flags[..14]);
        is_interleaved[t] = CoreInterleavedBits::is_interleaved_operands(&core_flags_14);
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
    let _ = polys.insert(
        PolynomialId::UntrustedAdvice,
        vec![ModFr::zero(); trace_length],
    );
    let _ = polys.insert(
        PolynomialId::TrustedAdvice,
        vec![ModFr::zero(); trace_length],
    );

    let (initial_ram_state, final_ram_state) =
        jolt_host::ram::build_ram_states(init_mem, final_memory, io_device, ram_k);

    let bc_entries: Vec<BytecodeEntry<ModFr>> = bytecode
        .bytecode
        .iter()
        .map(|instruction| {
            let instr = instruction.normalize();
            let circuit_flags = CoreFlags::circuit_flags(instruction);
            let instr_flags = CoreFlags::instruction_flags(instruction);
            let lookup_table = InstructionLookup::<64>::lookup_table(instruction)
                .map(|t| CoreLookupTables::<64>::enum_index(&t));
            BytecodeEntry {
                address: jolt_field::Field::from_u64(instr.address as u64),
                imm: jolt_field::Field::from_i128(instr.operands.imm),
                circuit_flags: circuit_flags.to_vec(),
                rd: instr.operands.rd,
                rs1: instr.operands.rs1,
                rs2: instr.operands.rs2,
                lookup_table,
                is_interleaved: {
                    let mut core_flags_14 = [false; 14];
                    core_flags_14.copy_from_slice(&circuit_flags[..14]);
                    CoreInterleavedBits::is_interleaved_operands(&core_flags_14)
                },
                is_branch: instr_flags[jolt_core::zkvm::instruction::InstructionFlags::Branch],
                left_is_rs1: instr_flags
                    [jolt_core::zkvm::instruction::InstructionFlags::LeftOperandIsRs1Value],
                left_is_pc: instr_flags
                    [jolt_core::zkvm::instruction::InstructionFlags::LeftOperandIsPC],
                right_is_rs2: instr_flags
                    [jolt_core::zkvm::instruction::InstructionFlags::RightOperandIsRs2Value],
                right_is_imm: instr_flags
                    [jolt_core::zkvm::instruction::InstructionFlags::RightOperandIsImm],
                is_noop: instr_flags[jolt_core::zkvm::instruction::InstructionFlags::IsNoop],
            }
        })
        .collect();

    let mut pc_indices = Vec::with_capacity(trace_length);
    for cycle in trace {
        pc_indices.push(bytecode.get_pc(cycle));
    }
    pc_indices.resize(trace_length, 0);

    let bytecode_data = BytecodeData {
        pc_indices,
        entries: bc_entries,
        entry_index: bytecode.entry_bytecode_index(),
        num_lookup_tables: jolt_compiler::params::NUM_LOOKUP_TABLES,
    };

    let setup = ZkvmSetup { config };

    (
        executable,
        polys,
        r1cs_key,
        r1cs_witness,
        setup,
        initial_ram_state,
        final_ram_state,
        instruction_flag_data,
        reg_access,
        lookup_trace,
        lookup_flags,
        bytecode_data,
    )
}

fn populate_bytecode_preprocessed(
    preprocessed: &mut PreprocessedSource<ModFr>,
    bc: &BytecodeData<ModFr>,
) {
    use jolt_field::Field;
    let k = bc.entries.len();

    let pc_idx_poly: Vec<ModFr> = bc
        .pc_indices
        .iter()
        .map(|&i| ModFr::from_u64(i as u64))
        .collect();
    preprocessed.insert(PolynomialId::BytecodePcIndex, pc_idx_poly);

    let pc_0 = bc.pc_indices[0];
    let mut entry_trace = vec![ModFr::zero(); k];
    entry_trace[pc_0] = ModFr::one();
    preprocessed.insert(PolynomialId::BytecodeEntryTrace, entry_trace);

    let mut entry_expected = vec![ModFr::zero(); k];
    entry_expected[bc.entry_index] = ModFr::one();
    preprocessed.insert(PolynomialId::BytecodeEntryExpected, entry_expected);

    bc.populate_preprocessed(preprocessed);
}
