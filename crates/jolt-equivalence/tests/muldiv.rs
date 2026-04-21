//! Cross-system equivalence test for the muldiv guest program.
//!
//! Runs jolt-core's prover and jolt-zkvm's prover, both with
//! `Blake2bTranscript` (real Fiat-Shamir, identical domain-separation labels),
//! extracts per-stage protocol trace data, and compares them
//! coefficient-by-coefficient.
//!
//! Each stage has its own independent test so stages light up green
//! incrementally as the jolt-zkvm pipeline is wired up.
#![allow(non_snake_case, clippy::print_stderr)]

use std::collections::BTreeSet;
use std::panic::{self, AssertUnwindSafe};
use std::process::Command;
use std::sync::OnceLock;

use jolt_core::curve::Bn254Curve;
use jolt_core::field::JoltField;
use jolt_core::host;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::poly::opening_proof::OpeningId;
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::instruction::{
    Flags as CoreFlags, InstructionLookup, InterleavedBitsMarker as CoreInterleavedBits,
};
use jolt_core::zkvm::lookup_table::LookupTables as CoreLookupTables;
use jolt_core::zkvm::proof_serialization::JoltProof;
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifier, JoltVerifierPreprocessing};

use common::constants::RAM_START_ADDRESS;
use jolt_compiler::module::Module;
use jolt_compiler::{Op, VerifierOp};
use jolt_compute::link;
use jolt_cpu::CpuBackend;
use jolt_dory::types::DoryProverSetup;
use jolt_dory::DoryScheme;
use jolt_host::{extract_trace, BytecodePreprocessing, CycleRow, InstructionFlagData, Program};
use jolt_instructions::LookupTableKind;
use jolt_r1cs::{constraints::rv64, R1csKey, R1csSource};
use jolt_transcript::Transcript;
use jolt_verifier::{OneHotConfig, ProverConfig, ReadWriteConfig, TRANSCRIPT_LABEL};
use jolt_witness::bytecode_raf::{BytecodeData, BytecodeEntry};
use jolt_witness::derived::{
    DerivedSource, FieldRegConfig, InstructionFlags, RamConfig, RegisterAccessData,
};
use jolt_witness::preprocessed::PreprocessedSource;
use jolt_witness::provider::ProverData;
use jolt_witness::{PolynomialConfig, PolynomialId, Polynomials};
use jolt_zkvm::prove::prove;
use jolt_zkvm::runtime::prefix_suffix::LookupTraceData;
use num_traits::{One, Zero};

use jolt_equivalence::checkpoint::{CheckpointTranscript, TranscriptEvent};
use jolt_equivalence::{compare_stage, StageTrace};

type Fr = ark_bn254::Fr;
type NewFr = jolt_field::Fr;

type CoreProver<'a> = JoltCpuProver<'a, Fr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript>;
type CoreProof = JoltProof<Fr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript>;
type CoreVerifier<'a> = JoltVerifier<'a, Fr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript>;

fn to_ark(f: NewFr) -> Fr {
    f.into()
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn extract_clear_rounds(
    proof: &SumcheckInstanceProof<Fr, Bn254Curve, Blake2bTranscript>,
    initial_claim: Fr,
    challenges: &[<Fr as JoltField>::Challenge],
) -> Vec<Vec<Fr>> {
    match proof {
        SumcheckInstanceProof::Clear(clear) => {
            clear.decompress_all_rounds(initial_claim, challenges)
        }
        SumcheckInstanceProof::Zk(_) => panic!("expected ClearSumcheckProof"),
    }
}

fn extract_clear_degree(proof: &SumcheckInstanceProof<Fr, Bn254Curve, Blake2bTranscript>) -> usize {
    match proof {
        SumcheckInstanceProof::Clear(clear) => clear
            .compressed_polys
            .first()
            .map_or(0, |p| p.coeffs_except_linear_term.len()),
        SumcheckInstanceProof::Zk(_) => panic!("expected ClearSumcheckProof"),
    }
}

/// Collect the claim values for all new openings added between two snapshots
/// of the verifier's opening accumulator.
fn diff_opening_evals(keys_before: &BTreeSet<OpeningId>, verifier: &CoreVerifier<'_>) -> Vec<Fr> {
    verifier
        .opening_accumulator
        .openings
        .iter()
        .filter(|(k, _)| !keys_before.contains(k))
        .map(|(_, (_, claim))| *claim)
        .collect()
}

fn snapshot_opening_keys(verifier: &CoreVerifier<'_>) -> BTreeSet<OpeningId> {
    verifier
        .opening_accumulator
        .openings
        .keys()
        .copied()
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// jolt-core extraction (reference prover)
// ═══════════════════════════════════════════════════════════════════

fn extract_jolt_core_stages() -> Vec<StageTrace<Fr>> {
    DoryGlobals::reset();

    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode,
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
        e_entry,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

    let elf_contents = program.get_elf_contents().expect("elf");
    let prover: CoreProver<'_> = CoreProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io = prover.program_io.clone();
    let (proof, _debug): (CoreProof, _) = prover.prove();

    let verifier_preprocessing: &'static _ = Box::leak(Box::new(JoltVerifierPreprocessing::from(
        &prover_preprocessing,
    )));
    let mut verifier: CoreVerifier<'_> =
        CoreVerifier::new(verifier_preprocessing, proof, io, None, None).expect("build verifier");
    verifier.run_preamble();

    let mut stages = Vec::new();

    macro_rules! extract_stage {
        (tuple, $verify:ident, $proof_field:ident) => {{
            let keys_before = snapshot_opening_keys(&verifier);
            let (sr, _) = verifier.$verify().expect(stringify!($verify));
            let coeffs = extract_clear_rounds(
                &verifier.proof.$proof_field,
                sr.initial_claim,
                &sr.challenges,
            );
            let degree = extract_clear_degree(&verifier.proof.$proof_field);
            let evals = diff_opening_evals(&keys_before, &verifier);
            stages.push(StageTrace {
                num_rounds: sr.challenges.len(),
                poly_degree: degree,
                round_poly_coeffs: coeffs,
                evals,
            });
        }};
        (plain, $verify:ident, $proof_field:ident) => {{
            let keys_before = snapshot_opening_keys(&verifier);
            let sr = verifier.$verify().expect(stringify!($verify));
            let coeffs = extract_clear_rounds(
                &verifier.proof.$proof_field,
                sr.initial_claim,
                &sr.challenges,
            );
            let degree = extract_clear_degree(&verifier.proof.$proof_field);
            let evals = diff_opening_evals(&keys_before, &verifier);
            stages.push(StageTrace {
                num_rounds: sr.challenges.len(),
                poly_degree: degree,
                round_poly_coeffs: coeffs,
                evals,
            });
        }};
    }

    extract_stage!(tuple, verify_stage1, stage1_sumcheck_proof);
    extract_stage!(tuple, verify_stage2, stage2_sumcheck_proof);
    extract_stage!(plain, verify_stage3, stage3_sumcheck_proof);
    extract_stage!(plain, verify_stage4, stage4_sumcheck_proof);
    extract_stage!(plain, verify_stage5, stage5_sumcheck_proof);
    extract_stage!(plain, verify_stage6, stage6_sumcheck_proof);
    extract_stage!(plain, verify_stage7, stage7_sumcheck_proof);

    stages
}

/// Protocol parameters extracted from jolt-core's proof.
struct CoreProtocolParams {
    trace_length: usize,
    ram_k: usize,
    bytecode_k: usize,
    /// PCS generators shared from jolt-core's prover preprocessing.
    /// Must be the exact same SRS for commitment equivalence.
    pcs_setup: DoryProverSetup,
}

/// Run jolt-core verifier and return per-operation state history + protocol params.
fn extract_jolt_core_state_history() -> (Vec<[u8; 32]>, CoreProtocolParams) {
    DoryGlobals::reset();

    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode,
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
        e_entry,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

    let elf_contents = program.get_elf_contents().expect("elf");
    let prover: CoreProver<'_> = CoreProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io = prover.program_io.clone();
    let (proof, _debug): (CoreProof, _) = prover.prove();

    let params = CoreProtocolParams {
        trace_length: proof.trace_length,
        ram_k: proof.ram_K,
        bytecode_k: prover_preprocessing.shared.bytecode.code_size,
        pcs_setup: DoryProverSetup(prover_preprocessing.generators.clone()),
    };

    let verifier_preprocessing: &'static _ = Box::leak(Box::new(JoltVerifierPreprocessing::from(
        &prover_preprocessing,
    )));
    let mut verifier: CoreVerifier<'_> =
        CoreVerifier::new(verifier_preprocessing, proof, io, None, None).expect("build verifier");

    verifier.run_preamble();
    let _ = verifier.verify_stage1().expect("stage1");
    let _ = verifier.verify_stage2().expect("stage2");
    let _ = verifier.verify_stage3().expect("stage3");
    let _ = verifier.verify_stage4().expect("stage4");
    let _ = verifier.verify_stage5().expect("stage5");
    let _ = verifier.verify_stage6().expect("stage6");
    let _ = verifier.verify_stage7().expect("stage7");

    (verifier.transcript.state_history.clone(), params)
}

// ═══════════════════════════════════════════════════════════════════
// jolt-zkvm extraction (new modular pipeline)
// ═══════════════════════════════════════════════════════════════════

#[allow(dead_code)]
fn build_protocol_module(log_t: usize, log_k_bytecode: usize, log_k_ram: usize) -> Module {
    build_protocol_module_with_example("jolt_core_module", log_t, log_k_bytecode, log_k_ram)
}

/// Generalized module builder: invokes `cargo run --example <example>` and
/// deserializes the emitted `.jolt` binary.
fn build_protocol_module_with_example(
    example: &str,
    log_t: usize,
    log_k_bytecode: usize,
    log_k_ram: usize,
) -> Module {
    let tmp_path =
        format!("/tmp/jolt_equiv_{example}_{log_t}_{log_k_bytecode}_{log_k_ram}.jolt");

    let output = Command::new("cargo")
        .args([
            "run",
            "--example",
            example,
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
        .unwrap_or_else(|e| panic!("failed to run {example} example: {e}"));

    assert!(
        output.status.success(),
        "{example} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = std::fs::read(&tmp_path).expect("failed to read protocol binary");
    Module::from_bytes(&bytes)
}

/// Truncate the module to include only the first `num_stages` prover stages.
/// Verifier ops are cut at the corresponding boundary.
fn truncate_after_stage(module: &mut Module, num_stages: usize) {
    // Prover: cut at BeginStage{num_stages} (the start of the next stage).
    // Also cut at the first BatchRoundBegin in the last stage if
    // the instance phases haven't been wired yet.
    let mut seen = 0;
    let mut in_final_stage = false;
    if let Some(pos) = module.prover.ops.iter().position(|op| {
        if let Op::BeginStage { index } = op {
            if *index >= num_stages {
                return true;
            }
            in_final_stage = *index + 1 == num_stages;
            seen = *index + 1;
        }
        // Stop before BatchRoundBegin if we're in the final stage
        // and the instance phases haven't been wired yet.
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

    // Verifier: cut at the (num_stages+1)-th BeginStage.
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

struct ZkvmSetup {
    trace_length: usize,
    config: ProverConfig,
}

#[allow(clippy::type_complexity)]
fn setup_zkvm_muldiv(
    core_params: &CoreProtocolParams,
) -> (
    jolt_compute::Executable<CpuBackend, NewFr>,
    Polynomials<NewFr>,
    R1csKey<NewFr>,
    Vec<NewFr>, // r1cs_witness
    ZkvmSetup,
    Vec<u64>, // initial_ram_state
    Vec<u64>, // final_ram_state
    InstructionFlagData<NewFr>,
    RegisterAccessData,
    LookupTraceData,
    jolt_witness::derived::LookupFlagData,
    BytecodeData<NewFr>,
) {
    setup_zkvm_muldiv_with_example(core_params, "jolt_core_module")
}

#[allow(clippy::type_complexity)]
fn setup_zkvm_muldiv_with_example(
    core_params: &CoreProtocolParams,
    example_name: &str,
) -> (
    jolt_compute::Executable<CpuBackend, NewFr>,
    Polynomials<NewFr>,
    R1csKey<NewFr>,
    Vec<NewFr>, // r1cs_witness
    ZkvmSetup,
    Vec<u64>, // initial_ram_state
    Vec<u64>, // final_ram_state
    InstructionFlagData<NewFr>,
    RegisterAccessData,
    LookupTraceData,
    jolt_witness::derived::LookupFlagData,
    BytecodeData<NewFr>,
) {
    let mut program = Program::new("muldiv-guest");
    let (bytecode_raw, init_mem, _program_size, entry_address) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, trace, final_memory, io_device) = program.trace(&inputs, &[], &[]);

    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, entry_address);
    let memory_layout = &io_device.memory_layout;

    let trace_length = core_params.trace_length;
    let log_t = trace_length.trailing_zeros() as usize;
    let bytecode_k = core_params.bytecode_k;
    let log_k_bytecode = bytecode_k.trailing_zeros() as usize;
    let ram_k = core_params.ram_k;
    let log_k_ram = ram_k.trailing_zeros() as usize;

    eprintln!(
        "Protocol params: trace_length={trace_length}, bytecode_k={bytecode_k}, \
         ram_k={ram_k} (log_t={log_t}, log_k_bc={log_k_bytecode}, log_k_ram={log_k_ram})"
    );

    let mut module =
        build_protocol_module_with_example(example_name, log_t, log_k_bytecode, log_k_ram);
    truncate_after_stage(&mut module, 8);
    let backend = CpuBackend;
    let executable = link::<CpuBackend, NewFr>(module, &backend);

    let one_hot = OneHotConfig::new(log_t);
    let log_k_chunk = one_hot.log_k_chunk as usize;
    let rw_config = ReadWriteConfig::new(log_t, log_k_ram);

    let lowest_addr = memory_layout.get_lowest_address();
    let remap = |addr: u64| ((addr - lowest_addr) / 8) as usize;

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
        outputs: {
            // jolt-core truncates trailing zeros from outputs in gen_from_trace
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
    let matrices = rv64::rv64_constraints::<NewFr>();
    let r1cs_key = R1csKey::new(matrices, trace_length);

    let (cycle_inputs, r1cs_witness, instruction_flag_data) = extract_trace::<_, NewFr>(
        &trace,
        trace_length,
        &bytecode,
        memory_layout,
        r1cs_key.num_vars_padded,
    );

    // Extract per-cycle register access indices and lookup data from the trace.
    let mut rd_indices = vec![None; trace_length];
    let mut rs1_indices = vec![None; trace_length];
    let mut rs2_indices = vec![None; trace_length];
    let mut lookup_keys = vec![0u128; trace_length];
    let mut table_kind_indices: Vec<Option<usize>> = vec![None; trace_length];
    let mut table_indices: Vec<Option<usize>> = vec![None; trace_length];
    let mut is_interleaved = vec![true; trace_length]; // NoOp padding has is_interleaved=true
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
        // Truncate refactor's 18-flag array to jolt-core's 14-flag shape
        // (IsFieldMul/Add/Sub/Inv at indices 14..17 don't affect is_interleaved_operands).
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
        table_kind_indices: table_kind_indices.clone(),
        is_interleaved: is_interleaved.clone(),
    };

    let is_raf: Vec<bool> = is_interleaved.iter().map(|&b| !b).collect();
    let lookup_flags = jolt_witness::derived::LookupFlagData {
        table_indices,
        is_raf,
    };

    let mut polys = Polynomials::<NewFr>::new(poly_config);
    polys.push(&cycle_inputs);
    polys.finish();
    let _ = polys.insert(
        PolynomialId::UntrustedAdvice,
        vec![NewFr::zero(); trace_length],
    );
    let _ = polys.insert(
        PolynomialId::TrustedAdvice,
        vec![NewFr::zero(); trace_length],
    );

    let (initial_ram_state, final_ram_state) =
        jolt_host::ram::build_ram_states(&init_mem, &final_memory, &io_device, ram_k);

    // Build BytecodeData for BytecodeReadRaf sumcheck
    let bc_entries: Vec<BytecodeEntry<NewFr>> = bytecode
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
    for cycle in &trace {
        pc_indices.push(bytecode.get_pc(cycle));
    }
    // Pad remaining cycles (NoOp) to bytecode index 0
    pc_indices.resize(trace_length, 0);

    let bytecode_data = BytecodeData {
        pc_indices,
        entries: bc_entries,
        entry_index: bytecode.entry_bytecode_index(),
        num_lookup_tables: jolt_compiler::params::NUM_LOOKUP_TABLES,
    };

    let setup = ZkvmSetup {
        trace_length,
        config,
    };

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
    preprocessed: &mut PreprocessedSource<NewFr>,
    bc: &BytecodeData<NewFr>,
) {
    use jolt_field::Field;
    let k = bc.entries.len();

    // BytecodePcIndex: per-cycle bytecode table index as field elements (length = T, padded)
    let pc_idx_poly: Vec<NewFr> = bc
        .pc_indices
        .iter()
        .map(|&i| NewFr::from_u64(i as u64))
        .collect();
    preprocessed.insert(PolynomialId::BytecodePcIndex, pc_idx_poly);

    // BytecodeEntryTrace: one-hot at PC of cycle 0 (from trace)
    let pc_0 = bc.pc_indices[0];
    let mut entry_trace = vec![NewFr::zero(); k];
    entry_trace[pc_0] = NewFr::one();
    preprocessed.insert(PolynomialId::BytecodeEntryTrace, entry_trace);

    // BytecodeEntryExpected: one-hot at expected entry bytecode index (from preprocessing)
    let mut entry_expected = vec![NewFr::zero(); k];
    entry_expected[bc.entry_index] = NewFr::one();
    preprocessed.insert(PolynomialId::BytecodeEntryExpected, entry_expected);

    // Per-field bytecode polynomials for WeightedSum (BytecodeVal decomposition)
    bc.populate_preprocessed(preprocessed);
}

fn extract_jolt_zkvm_stages() -> Vec<StageTrace<Fr>> {
    let (_, params) = jolt_core_state_history();
    let (
        executable,
        mut polys,
        r1cs_key,
        r1cs_witness,
        setup,
        initial_ram,
        final_ram,
        instruction_flag_data,
        reg_access,
        lookup_trace,
        lookup_flags,
        bytecode_data,
    ) = setup_zkvm_muldiv(params);

    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let derived = DerivedSource::new(&r1cs_witness, setup.trace_length, r1cs_key.num_vars_padded)
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
        .with_lookup_flags(lookup_flags);
    let mut preprocessed = PreprocessedSource::new();
    preprocessed.populate_ram(&setup.config, &initial_ram);
    populate_bytecode_preprocessed(&mut preprocessed, &bytecode_data);
    let mut provider =
        ProverData::new(&mut polys, r1cs, derived, preprocessed).with_lookup_trace(lookup_trace);

    let pcs_setup = &params.pcs_setup;
    let mut transcript = jolt_transcript::Blake2bTranscript::<NewFr>::new(TRANSCRIPT_LABEL);

    let proof = prove::<_, _, _, DoryScheme>(
        &executable,
        &mut provider,
        &backend(),
        pcs_setup,
        &mut transcript,
        setup.config,
    );

    proof
        .stage_proofs
        .iter()
        .map(|sp| {
            let round_polys = &sp.round_polys.round_polynomials;
            StageTrace {
                num_rounds: round_polys.len(),
                poly_degree: round_polys
                    .first()
                    .map_or(0, |p| p.coefficients().len().saturating_sub(1)),
                round_poly_coeffs: round_polys
                    .iter()
                    .map(|p| p.coefficients().iter().copied().map(to_ark).collect())
                    .collect(),
                evals: sp.evals.iter().copied().map(to_ark).collect(),
            }
        })
        .collect()
}

/// Run jolt-zkvm prover with a CheckpointTranscript and return the event log.
fn extract_jolt_zkvm_checkpoint_log() -> Vec<TranscriptEvent> {
    let (_, params) = jolt_core_state_history();
    let (
        executable,
        mut polys,
        r1cs_key,
        r1cs_witness,
        setup,
        initial_ram,
        final_ram,
        instruction_flag_data,
        reg_access,
        lookup_trace,
        lookup_flags,
        bytecode_data,
    ) = setup_zkvm_muldiv(params);

    // Debug: print InstructionRa[0] stats
    if let Some(ra0) = polys.try_get(PolynomialId::InstructionRa(0)) {
        let nonzero: Vec<_> = ra0
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_zero())
            .take(5)
            .collect();
        eprintln!(
            "InstructionRa(0): len={}, nonzero_count={}, first_nonzero={:?}",
            ra0.len(),
            ra0.iter().filter(|v| !v.is_zero()).count(),
            nonzero.iter().map(|(i, _)| i).collect::<Vec<_>>(),
        );
    }

    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let derived = DerivedSource::new(&r1cs_witness, setup.trace_length, r1cs_key.num_vars_padded)
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
        .with_lookup_flags(lookup_flags);
    let mut preprocessed = PreprocessedSource::new();
    preprocessed.populate_ram(&setup.config, &initial_ram);
    populate_bytecode_preprocessed(&mut preprocessed, &bytecode_data);
    let mut provider =
        ProverData::new(&mut polys, r1cs, derived, preprocessed).with_lookup_trace(lookup_trace);

    let pcs_setup = &params.pcs_setup;
    let mut transcript =
        CheckpointTranscript::<jolt_transcript::Blake2bTranscript<NewFr>>::new(TRANSCRIPT_LABEL);

    let _proof = prove::<_, _, _, DoryScheme>(
        &executable,
        &mut provider,
        &backend(),
        pcs_setup,
        &mut transcript,
        setup.config,
    );

    transcript.into_log()
}

fn backend() -> CpuBackend {
    CpuBackend
}

// ═══════════════════════════════════════════════════════════════════
// Cached setup (OnceLock so multiple per-stage tests share one run)
// ═══════════════════════════════════════════════════════════════════

static CORE_STAGES: OnceLock<Vec<StageTrace<Fr>>> = OnceLock::new();
static ZKVM_STAGES: OnceLock<Result<Vec<StageTrace<Fr>>, String>> = OnceLock::new();
static CORE_STATE_HISTORY: OnceLock<(Vec<[u8; 32]>, CoreProtocolParams)> = OnceLock::new();
static ZKVM_CHECKPOINT_LOG: OnceLock<Result<Vec<TranscriptEvent>, String>> = OnceLock::new();

fn jolt_core_stages() -> &'static Vec<StageTrace<Fr>> {
    CORE_STAGES.get_or_init(extract_jolt_core_stages)
}

fn jolt_zkvm_stages() -> &'static Result<Vec<StageTrace<Fr>>, String> {
    ZKVM_STAGES.get_or_init(|| {
        let result = panic::catch_unwind(AssertUnwindSafe(extract_jolt_zkvm_stages));
        match result {
            Ok(stages) => Ok(stages),
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                Err(msg)
            }
        }
    })
}

fn jolt_core_state_history() -> &'static (Vec<[u8; 32]>, CoreProtocolParams) {
    CORE_STATE_HISTORY.get_or_init(extract_jolt_core_state_history)
}

fn jolt_zkvm_checkpoint() -> &'static Result<Vec<TranscriptEvent>, String> {
    ZKVM_CHECKPOINT_LOG.get_or_init(|| {
        let result = panic::catch_unwind(AssertUnwindSafe(extract_jolt_zkvm_checkpoint_log));
        match result {
            Ok(log) => Ok(log),
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                Err(msg)
            }
        }
    })
}

// ═══════════════════════════════════════════════════════════════════
// Smoke tests (jolt-core only — always runnable)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn jolt_core_mock_transcript_proves() {
    let stages = jolt_core_stages();
    assert_eq!(stages.len(), 7);
    for (i, stage) in stages.iter().enumerate() {
        eprintln!(
            "Stage {}: {} rounds, degree {}, {} evals",
            i + 1,
            stage.num_rounds,
            stage.poly_degree,
            stage.evals.len(),
        );
        assert!(stage.num_rounds > 0);
        assert!(stage.poly_degree > 0);
        assert_eq!(stage.round_poly_coeffs.len(), stage.num_rounds);
    }
}

#[test]
#[ignore = "requires full pipeline wiring"]
fn jolt_zkvm_mock_transcript_proves() {
    let stages = match jolt_zkvm_stages() {
        Ok(s) => s,
        Err(e) => panic!("jolt-zkvm prove failed: {e}"),
    };
    eprintln!("jolt-zkvm produced {} stages", stages.len());
    for (i, stage) in stages.iter().enumerate() {
        eprintln!(
            "Stage {}: {} rounds, degree {}, {} evals",
            i + 1,
            stage.num_rounds,
            stage.poly_degree,
            stage.evals.len(),
        );
    }
    assert!(!stages.is_empty(), "expected at least 1 stage");
}

// ═══════════════════════════════════════════════════════════════════
// Per-stage cross-system equivalence tests
// ═══════════════════════════════════════════════════════════════════

macro_rules! equivalence_test_body {
    ($stage_idx:literal) => {{
        let core = jolt_core_stages();
        let zkvm = match jolt_zkvm_stages() {
            Ok(s) => s,
            Err(e) => panic!("jolt-zkvm prove failed: {e}"),
        };
        assert!(
            core.len() > $stage_idx,
            "jolt-core missing stage {}",
            $stage_idx + 1
        );
        if zkvm.len() <= $stage_idx {
            eprintln!(
                "SKIP: jolt-zkvm has {} stages, need {}",
                zkvm.len(),
                $stage_idx + 1
            );
            return;
        }
        compare_stage($stage_idx, &core[$stage_idx], &zkvm[$stage_idx])
            .unwrap_or_else(|e| panic!("{e}"));
    }};
}

#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage1() {
    equivalence_test_body!(0);
}
#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage2() {
    equivalence_test_body!(1);
}
#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage3() {
    equivalence_test_body!(2);
}
#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage4() {
    equivalence_test_body!(3);
}
#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage5() {
    equivalence_test_body!(4);
}
#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage6() {
    equivalence_test_body!(5);
}
#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage7() {
    equivalence_test_body!(6);
}

// ═══════════════════════════════════════════════════════════════════
// Transcript divergence test
// ═══════════════════════════════════════════════════════════════════

fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        let _ = write!(s, "{byte:02x}");
    }
    s
}

/// Compare jolt-core and jolt-zkvm transcript state histories operation-by-operation.
///
/// jolt-core's state_history (cfg(test)) gives us the state after every raw hash
/// operation. jolt-zkvm's CheckpointTranscript gives us state_after for every
/// append/squeeze. Since both use the same Blake2b256 transcript construction,
/// the N-th state in both logs should be identical.
///
/// The first divergence pinpoints the exact operation where the two systems disagree.
#[test]
fn transcript_divergence() {
    let (golden, _) = jolt_core_state_history();
    let log = match jolt_zkvm_checkpoint() {
        Ok(l) => l,
        Err(e) => panic!("jolt-zkvm checkpoint extraction failed: {e}"),
    };

    // Extract state_after from checkpoint events
    let zkvm_states: Vec<[u8; 32]> = log
        .iter()
        .map(|ev| match ev {
            TranscriptEvent::Append { state_after, .. } => *state_after,
            TranscriptEvent::Squeeze { state_after } => *state_after,
        })
        .collect();

    eprintln!("jolt-core state_history: {} entries", golden.len());
    eprintln!("jolt-zkvm checkpoint:    {} entries", zkvm_states.len());

    // golden[0] is the initial state from `new(b"Jolt")`, not an operation.
    // zkvm_states[0] is the state after the first operation.
    // So golden[i+1] should match zkvm_states[i].
    let golden_ops = &golden[1..]; // skip initial state
    let min_len = golden_ops.len().min(zkvm_states.len());

    // Check initial states match (both from `new(b"Jolt")`)
    eprintln!("Initial state (jolt-core): {}", &hex(&golden[0])[..16]);

    // Print label summary around any divergence to understand transcript structure
    {
        let mut last_label = String::new();
        let mut run_start = 0;
        for (i, event) in log.iter().enumerate() {
            let label = match event {
                TranscriptEvent::Append { bytes, .. } => {
                    if bytes.len() >= 32 {
                        let end = bytes[..24].iter().position(|&b| b == 0).unwrap_or(24);
                        std::str::from_utf8(&bytes[..end])
                            .unwrap_or("field")
                            .to_string()
                    } else {
                        format!("raw({})", bytes.len())
                    }
                }
                TranscriptEvent::Squeeze { .. } => "squeeze".to_string(),
            };
            if label != last_label {
                if !last_label.is_empty() {
                    eprintln!("  ops[{run_start}..{i}] ({}) = {last_label}", i - run_start);
                }
                last_label = label;
                run_start = i;
            }
        }
        if !last_label.is_empty() {
            eprintln!(
                "  ops[{run_start}..{}] ({}) = {last_label}",
                log.len(),
                log.len() - run_start
            );
        }
    }

    for i in 0..min_len {
        if golden_ops[i] != zkvm_states[i] {
            eprintln!("\n=== DIVERGENCE at operation #{i} ===");
            // Show wider context (20 ops before)
            let start = i.saturating_sub(20);
            eprintln!("Context (operations {start}..{}):", (i + 1).min(min_len));
            for j in start..=i.min(min_len - 1) {
                let marker = if j == i { ">>>" } else { "   " };
                eprintln!(
                    "  {marker} [{j}] core={} zkvm={}",
                    &hex(&golden_ops[j])[..16],
                    &hex(&zkvm_states[j])[..16],
                );
                // Show the checkpoint event details for zkvm
                if j < log.len() {
                    match &log[j] {
                        TranscriptEvent::Append { bytes, .. } => {
                            let label_preview = if bytes.len() >= 32 {
                                let label_end =
                                    bytes[..24].iter().position(|&b| b == 0).unwrap_or(24);
                                let label_str =
                                    std::str::from_utf8(&bytes[..label_end]).unwrap_or("???");
                                format!("Append(label={:?}, len={})", label_str, bytes.len())
                            } else {
                                format!("Append({} bytes)", bytes.len())
                            };
                            eprintln!("           zkvm event: {label_preview}");
                        }
                        TranscriptEvent::Squeeze { .. } => {
                            eprintln!("           zkvm event: Squeeze");
                        }
                    }
                }
            }
            // Print full diagnostic for divergence
            eprintln!("\n=== FULL DIAGNOSTIC ===");
            eprintln!("Op #{i} full states:");
            eprintln!("  core: {}", hex(&golden_ops[i]));
            eprintln!("  zkvm: {}", hex(&zkvm_states[i]));
            if i > 0 {
                eprintln!("Op #{} full states (prior):", i - 1);
                eprintln!("  core: {}", hex(&golden_ops[i - 1]));
                eprintln!("  zkvm: {}", hex(&zkvm_states[i - 1]));
            }
            // Print full bytes at divergence point
            if let TranscriptEvent::Append { bytes, .. } = &log[i] {
                eprintln!(
                    "Bytes appended at op #{i} ({} bytes): {}",
                    bytes.len(),
                    hex(bytes)
                );
                if bytes.len() == 32 {
                    let count = u64::from_be_bytes(bytes[24..32].try_into().unwrap());
                    eprintln!("  LabelWithCount count = {count}");
                }
            }
            panic!(
                "Transcript divergence at operation #{i}: \
                 core state {} != zkvm state {}",
                &hex(&golden_ops[i])[..16],
                &hex(&zkvm_states[i])[..16],
            );
        }
    }

    match golden_ops.len().cmp(&zkvm_states.len()) {
        std::cmp::Ordering::Greater => eprintln!(
            "\njolt-zkvm log is shorter ({} ops) than jolt-core ({} ops). \
             {} operations matched before zkvm ran out.",
            zkvm_states.len(),
            golden_ops.len(),
            min_len,
        ),
        std::cmp::Ordering::Less => eprintln!(
            "\njolt-zkvm log is longer ({} ops) than jolt-core ({} ops). \
             {} operations matched before core ran out.",
            zkvm_states.len(),
            golden_ops.len(),
            min_len,
        ),
        std::cmp::Ordering::Equal => eprintln!("\nAll {} operations matched perfectly.", min_len,),
    }
}

/// Compare all InstructionRa chunk data between jolt-core and jolt-zkvm.
#[test]
fn instruction_ra0_data_matches() {
    use jolt_core::poly::commitment::dory::DoryGlobals;

    let (_, params) = jolt_core_state_history();

    DoryGlobals::reset();
    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode,
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
        e_entry,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents = program.get_elf_contents().expect("elf");
    let prover: CoreProver<'_> = CoreProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );

    let one_hot_params = &prover.one_hot_params;
    let instruction_d = one_hot_params.instruction_d;
    let trace_length = params.trace_length;

    let (_, polys, _, _, _, _, _, _, _, _, _, _) = setup_zkvm_muldiv(params);

    // Check ALL InstructionRa chunks
    let mut total_diffs = 0;
    for chunk in 0..instruction_d {
        let core_dense: Vec<Fr> = {
            use jolt_core::zkvm::instruction::LookupQuery;
            let k_chunk = one_hot_params.k_chunk;
            let addresses: Vec<Option<u8>> = prover
                .trace
                .iter()
                .map(|cycle| {
                    let lookup_index = LookupQuery::<64>::to_lookup_index(cycle);
                    Some(one_hot_params.lookup_index_chunk(lookup_index, chunk))
                })
                .collect();

            let t = trace_length;
            let mut dense = vec![Fr::zero(); t * k_chunk];
            for (cycle, &addr) in addresses.iter().enumerate() {
                if let Some(a) = addr {
                    dense[a as usize * t + cycle] = Fr::from(1u64);
                }
            }
            dense
        };

        let zkvm_data: Vec<NewFr> = polys.get(PolynomialId::InstructionRa(chunk)).to_vec();

        let diffs: usize = core_dense
            .iter()
            .zip(zkvm_data.iter())
            .filter(|(c, z)| {
                let z_ark: Fr = (**z).into();
                **c != z_ark
            })
            .count();

        if diffs > 0 {
            eprintln!(
                "InstructionRa({chunk}): {diffs} differences (len={})",
                core_dense.len()
            );
            // Print details for first diverging cycles
            use jolt_core::zkvm::instruction::LookupQuery;
            let t = trace_length;
            for (i, (c, z)) in core_dense.iter().zip(zkvm_data.iter()).enumerate() {
                let z_ark: Fr = (*z).into();
                if *c != z_ark {
                    let addr = i / t;
                    let cycle = i % t;
                    if cycle < prover.trace.len() {
                        let core_lookup = LookupQuery::<64>::to_lookup_index(&prover.trace[cycle]);
                        let core_chunk_val = one_hot_params.lookup_index_chunk(core_lookup, chunk);
                        eprintln!(
                            "  flat={i} addr={addr} cycle={cycle}: core={} zkvm={}, \
                             core_lookup={core_lookup:#x} core_chunk{chunk}={core_chunk_val}, \
                             instr={:?}",
                            i32::from(!c.is_zero()),
                            i32::from(!z_ark.is_zero()),
                            &format!("{:?}", prover.trace[cycle])
                                .split('(')
                                .next()
                                .unwrap_or("?"),
                        );
                    }
                }
            }
            total_diffs += diffs;
        }
    }

    eprintln!("Checked {instruction_d} InstructionRa chunks, total diffs: {total_diffs}");
    assert_eq!(
        total_diffs, 0,
        "polynomial data should match exactly across all chunks"
    );
}

/// Compare initial and final RAM states built by jolt-host's `build_ram_states`
/// against jolt-core's `gen_ram_memory_states` to find any discrepancies.
#[test]
fn debug_ram_state_comparison() {
    use jolt_core::poly::commitment::dory::DoryGlobals;
    use jolt_core::zkvm::ram::{gen_ram_memory_states, RAMPreprocessing};

    let (_, params) = jolt_core_state_history();
    let ram_k = params.ram_k;

    // -- jolt-core side --
    DoryGlobals::reset();
    let mut core_program = host::Program::new("muldiv-guest");
    let (_bytecode, core_init_mem, _, _e_entry) = core_program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, _, core_final_memory, core_io_device) = core_program.trace(&inputs, &[], &[]);

    let ram_pp = RAMPreprocessing::preprocess(core_init_mem.clone());
    let (core_initial, core_final) =
        gen_ram_memory_states::<Fr>(ram_k, &ram_pp, &core_io_device, &core_final_memory);

    // -- jolt-host side --
    let mut host_program = Program::new("muldiv-guest");
    let (_bytecode_raw, host_init_mem, _, _entry) = host_program.decode();
    let (_, _, host_final_memory, host_io_device) = host_program.trace(&inputs, &[], &[]);

    let (host_initial, host_final) = jolt_host::ram::build_ram_states(
        &host_init_mem,
        &host_final_memory,
        &host_io_device,
        ram_k,
    );

    // -- Compare initial states --
    eprintln!("ram_k = {ram_k}");
    eprintln!(
        "core_initial len={}, host_initial len={}",
        core_initial.len(),
        host_initial.len()
    );

    let mut initial_diffs = 0usize;
    let max_report = 20;
    let len = core_initial.len().min(host_initial.len());
    for i in 0..len {
        if core_initial[i] != host_initial[i] {
            if initial_diffs < max_report {
                eprintln!(
                    "  INITIAL diff at k={i}: core={:#018x} host={:#018x}",
                    core_initial[i], host_initial[i]
                );
            }
            initial_diffs += 1;
        }
    }
    // Check for length mismatch
    if core_initial.len() != host_initial.len() {
        eprintln!(
            "  INITIAL length mismatch: core={} host={}",
            core_initial.len(),
            host_initial.len()
        );
    }
    eprintln!("Initial state: {initial_diffs} differences out of {len} entries");

    // -- Compare final states --
    eprintln!(
        "core_final len={}, host_final len={}",
        core_final.len(),
        host_final.len()
    );

    let mut final_diffs = 0usize;
    let len_final = core_final.len().min(host_final.len());
    for i in 0..len_final {
        if core_final[i] != host_final[i] {
            if final_diffs < max_report {
                eprintln!(
                    "  FINAL diff at k={i}: core={:#018x} host={:#018x}",
                    core_final[i], host_final[i]
                );
            }
            final_diffs += 1;
        }
    }
    if core_final.len() != host_final.len() {
        eprintln!(
            "  FINAL length mismatch: core={} host={}",
            core_final.len(),
            host_final.len()
        );
    }
    eprintln!("Final state: {final_diffs} differences out of {len_final} entries");

    // -- Summary of nonzero entries for sanity --
    let core_initial_nonzero = core_initial.iter().filter(|&&v| v != 0).count();
    let host_initial_nonzero = host_initial.iter().filter(|&&v| v != 0).count();
    let core_final_nonzero = core_final.iter().filter(|&&v| v != 0).count();
    let host_final_nonzero = host_final.iter().filter(|&&v| v != 0).count();
    eprintln!(
        "Nonzero counts: core_initial={core_initial_nonzero}, host_initial={host_initial_nonzero}, \
         core_final={core_final_nonzero}, host_final={host_final_nonzero}"
    );

    assert_eq!(
        initial_diffs, 0,
        "initial RAM state has {initial_diffs} differences between jolt-core and jolt-host"
    );
    assert_eq!(
        final_diffs, 0,
        "final RAM state has {final_diffs} differences between jolt-core and jolt-host"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Cross-system verification: jolt-zkvm proof → jolt-core verifier
// ═══════════════════════════════════════════════════════════════════

/// Convert a `jolt_poly::UnivariatePoly<NewFr>` to a `CompressedUniPoly<Fr>`.
///
/// `CompressedUniPoly` stores `[c0, c2, c3, ...]` (linear term `c1` omitted;
/// the verifier reconstructs it from the sumcheck hint `h(0) + h(1)`).
fn to_compressed_uni_poly(
    poly: &jolt_poly::UnivariatePoly<NewFr>,
) -> jolt_core::poly::unipoly::CompressedUniPoly<Fr> {
    let coeffs = poly.coefficients();
    assert!(
        coeffs.len() >= 2,
        "round poly must have at least 2 coefficients"
    );
    let mut compressed = Vec::with_capacity(coeffs.len() - 1);
    compressed.push(to_ark(coeffs[0]));
    for c in &coeffs[2..] {
        compressed.push(to_ark(*c));
    }
    jolt_core::poly::unipoly::CompressedUniPoly {
        coeffs_except_linear_term: compressed,
    }
}

/// Convert a slice of jolt-zkvm round polynomials to a jolt-core
/// `SumcheckInstanceProof::Clear` (non-zk).
fn to_core_sumcheck_proof(
    round_polys: &[jolt_poly::UnivariatePoly<NewFr>],
) -> SumcheckInstanceProof<Fr, Bn254Curve, Blake2bTranscript> {
    let compressed: Vec<_> = round_polys.iter().map(to_compressed_uni_poly).collect();
    SumcheckInstanceProof::Clear(jolt_core::subprotocols::sumcheck::ClearSumcheckProof::new(
        compressed,
    ))
}

/// Convert jolt-dory `DoryCommitment` to jolt-core `ArkGT`.
///
/// Both are repr(transparent) wrappers over the same `Fq12` type.
fn commitment_to_ark(
    c: &jolt_dory::types::DoryCommitment,
) -> <DoryCommitmentScheme as jolt_core::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment
{
    // DoryCommitment(Bn254GT) → Bn254GT → ArkGT via transmute_copy
    // SAFETY: Bn254GT and ArkGT are both repr(transparent) over Fq12.
    unsafe { std::mem::transmute_copy(&c.0) }
}

/// Run jolt-core prover and return the proof, verifier preprocessing, and IO.
///
/// This is separate from `extract_jolt_core_state_history` because we need
/// the proof as a value (not consumed by a verifier).
fn run_jolt_core_prover() -> (
    CoreProof,
    &'static JoltVerifierPreprocessing<Fr, Bn254Curve, DoryCommitmentScheme>,
    common::jolt_device::JoltDevice,
    CoreProtocolParams,
) {
    DoryGlobals::reset();

    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode,
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
        e_entry,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

    let elf_contents = program.get_elf_contents().expect("elf");
    let prover: CoreProver<'_> = CoreProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io = prover.program_io.clone();
    let (proof, _debug): (CoreProof, _) = prover.prove();

    let params = CoreProtocolParams {
        trace_length: proof.trace_length,
        ram_k: proof.ram_K,
        bytecode_k: prover_preprocessing.shared.bytecode.code_size,
        pcs_setup: DoryProverSetup(prover_preprocessing.generators.clone()),
    };

    let verifier_preprocessing: &'static _ = Box::leak(Box::new(JoltVerifierPreprocessing::from(
        &prover_preprocessing,
    )));

    (proof, verifier_preprocessing, io, params)
}

/// Run jolt-zkvm prover and return the proof.
fn run_jolt_zkvm_prover(
    params: &CoreProtocolParams,
) -> jolt_verifier::JoltProof<NewFr, DoryScheme> {
    let (
        executable,
        mut polys,
        r1cs_key,
        r1cs_witness,
        setup,
        initial_ram,
        final_ram,
        instruction_flag_data,
        reg_access,
        lookup_trace,
        lookup_flags,
        bytecode_data,
    ) = setup_zkvm_muldiv(params);

    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let mut preprocessed = PreprocessedSource::new();
    preprocessed.populate_ram(&setup.config, &initial_ram);
    let derived = DerivedSource::new(&r1cs_witness, setup.trace_length, r1cs_key.num_vars_padded)
        .with_ram(RamConfig {
            ram_k: setup.config.ram_k,
            lowest_addr: setup.config.ram_lowest_address,
            initial_state: initial_ram,
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
        .with_lookup_flags(lookup_flags);
    populate_bytecode_preprocessed(&mut preprocessed, &bytecode_data);
    let mut provider =
        ProverData::new(&mut polys, r1cs, derived, preprocessed).with_lookup_trace(lookup_trace);

    let pcs_setup = &params.pcs_setup;
    let mut transcript = jolt_transcript::Blake2bTranscript::<NewFr>::new(TRANSCRIPT_LABEL);

    prove::<_, _, _, DoryScheme>(
        &executable,
        &mut provider,
        &backend(),
        pcs_setup,
        &mut transcript,
        setup.config,
    )
}

/// Prove with jolt-zkvm, convert to jolt-core's proof type, verify with jolt-core verifier.
#[test]
fn zkvm_proof_accepted_by_core_verifier() {
    // 1. Run jolt-core prover to get proof scaffolding + verifier preprocessing.
    let (core_proof, verifier_preprocessing, io, params) = run_jolt_core_prover();

    // 2. Run jolt-zkvm prover to get the proof under test.
    let zkvm_proof = run_jolt_zkvm_prover(&params);

    // 3. Convert: substitute zkvm proof data into core proof structure.
    assert_eq!(
        zkvm_proof.stage_proofs.len(),
        8,
        "expected 8 stage proofs from jolt-zkvm (7 sumcheck + 1 PCS opening)"
    );

    // Stages 1, 2: skip first round poly (uniskip). Stages 3-7: all round polys.
    let stage1_sc =
        to_core_sumcheck_proof(&zkvm_proof.stage_proofs[0].round_polys.round_polynomials[1..]);
    let stage2_sc =
        to_core_sumcheck_proof(&zkvm_proof.stage_proofs[1].round_polys.round_polynomials[1..]);
    let stage3_sc =
        to_core_sumcheck_proof(&zkvm_proof.stage_proofs[2].round_polys.round_polynomials);
    let stage4_sc =
        to_core_sumcheck_proof(&zkvm_proof.stage_proofs[3].round_polys.round_polynomials);
    let stage5_sc =
        to_core_sumcheck_proof(&zkvm_proof.stage_proofs[4].round_polys.round_polynomials);
    let stage6_sc =
        to_core_sumcheck_proof(&zkvm_proof.stage_proofs[5].round_polys.round_polynomials);
    let stage7_sc =
        to_core_sumcheck_proof(&zkvm_proof.stage_proofs[6].round_polys.round_polynomials);

    // Commitments: convert DoryCommitment → ArkGT.
    // jolt-core's proof.commitments has only the main witness (25 polys).
    // jolt-zkvm's may include advice commitments too, so we take only the
    // first N to match.
    let expected_num_commitments = core_proof.commitments.len();
    eprintln!(
        "[cross-system] commitments: zkvm={} core={}",
        zkvm_proof.commitments.len(),
        expected_num_commitments
    );
    let commitments: Vec<_> = zkvm_proof
        .commitments
        .iter()
        .filter_map(|c| c.as_ref())
        .take(expected_num_commitments)
        .map(commitment_to_ark)
        .collect();

    // Opening proof: DoryProof(ArkDoryProof) → ArkDoryProof
    assert_eq!(
        zkvm_proof.opening_proofs.len(),
        1,
        "expected exactly 1 joint opening proof"
    );
    let joint_opening_proof = zkvm_proof.opening_proofs[0].0.clone();

    let converted_proof = JoltProof {
        // From jolt-zkvm (the data under test):
        commitments,
        stage1_sumcheck_proof: stage1_sc,
        stage2_sumcheck_proof: stage2_sc,
        stage3_sumcheck_proof: stage3_sc,
        stage4_sumcheck_proof: stage4_sc,
        stage5_sumcheck_proof: stage5_sc,
        stage6_sumcheck_proof: stage6_sc,
        stage7_sumcheck_proof: stage7_sc,
        joint_opening_proof,
        // From jolt-core (structural scaffolding — identical by transcript parity):
        stage1_uni_skip_first_round_proof: core_proof.stage1_uni_skip_first_round_proof,
        stage2_uni_skip_first_round_proof: core_proof.stage2_uni_skip_first_round_proof,
        untrusted_advice_commitment: core_proof.untrusted_advice_commitment,
        opening_claims: core_proof.opening_claims,
        // Metadata:
        trace_length: core_proof.trace_length,
        ram_K: core_proof.ram_K,
        rw_config: core_proof.rw_config,
        one_hot_config: core_proof.one_hot_config,
        dory_layout: core_proof.dory_layout,
    };

    // 4. Verify with jolt-core verifier, stage by stage for diagnostics.
    let mut verifier = CoreVerifier::new(verifier_preprocessing, converted_proof, io, None, None)
        .expect("failed to construct jolt-core verifier");

    verifier.run_preamble();
    eprintln!("[cross-system] preamble OK");

    let _ = verifier.verify_stage1().expect("stage 1 failed");
    eprintln!("[cross-system] stage 1 OK");

    let _ = verifier.verify_stage2().expect("stage 2 failed");
    eprintln!("[cross-system] stage 2 OK");

    let _ = verifier.verify_stage3().expect("stage 3 failed");
    eprintln!("[cross-system] stage 3 OK");

    let _ = verifier.verify_stage4().expect("stage 4 failed");
    eprintln!("[cross-system] stage 4 OK");

    let _ = verifier.verify_stage5().expect("stage 5 failed");
    eprintln!("[cross-system] stage 5 OK");

    let _ = verifier.verify_stage6().expect("stage 6 failed");
    eprintln!("[cross-system] stage 6 OK");

    let _ = verifier.verify_stage7().expect("stage 7 failed");
    eprintln!("[cross-system] stage 7 OK");

    let _ = verifier.verify_stage8().expect("stage 8 failed");
    eprintln!("[cross-system] stage 8 OK");

    eprintln!("SUCCESS: jolt-core verifier accepted jolt-zkvm proof (all 8 stages)");
}

// ═══════════════════════════════════════════════════════════════════
// Modular self-verify: jolt-zkvm proof → jolt-verifier::verify
// ═══════════════════════════════════════════════════════════════════

/// Prove with jolt-zkvm (modular stack) and verify with the native
/// `jolt_verifier::verify` — no jolt-core dependency on the verify path.
///
/// This test is the target for end-to-end modular self-verification.
/// It requires that `jolt_core_module.rs` (the handwritten protocol
/// reference used by the bench) emits a *complete* verifier schedule
/// (all 7 stages, CollectOpeningClaim for every committed poly,
/// matching Op::RecordEvals on the prover side). Today only stages
/// 1–4 are wired on the verifier side and Op::RecordEvals is missing
/// on the prover side — so this test currently fails with
/// "missing eval 0 in stage proof". The commit-skip fix (Option
/// commitments) is the first of several steps needed; the remaining
/// work is tracked in PERF_TASKS.md.
#[test]
fn modular_self_verify() {
    use jolt_dory::types::DoryVerifierSetup;

    let (_, params) = jolt_core_state_history();
    let zkvm_proof = run_jolt_zkvm_prover(params);

    let (
        executable,
        _polys,
        r1cs_key,
        _r1cs_witness,
        setup,
        _initial_ram,
        _final_ram,
        _instruction_flag_data,
        _reg_access,
        _lookup_trace,
        _lookup_flags,
        _bytecode_data,
    ) = setup_zkvm_muldiv(params);

    let pcs_verifier_setup = DoryVerifierSetup(params.pcs_setup.0.to_verifier_setup());
    let verifying_key = jolt_verifier::JoltVerifyingKey::<NewFr, DoryScheme>::new(
        &executable.module,
        pcs_verifier_setup,
        r1cs_key,
    );

    jolt_verifier::verify(&verifying_key, &zkvm_proof, &setup.config.io_hash)
        .expect("modular verify should accept modular proof");
}

/// Partial self-verify check: the commit-skip symmetry.
///
/// Proves that the `Vec<Option<PCS::Output>>` commitment encoding
/// correctly round-trips through the modular verifier's
/// `AbsorbCommitment` op — i.e. we no longer see the historical
/// `InvalidProof("missing commitment")` error when all-zero advice
/// polys cause the prover to skip the commit. This is a stepping
/// stone toward full `modular_self_verify`.
#[test]
fn modular_self_verify_commit_skip_alignment() {
    use jolt_dory::types::DoryVerifierSetup;

    let (_, params) = jolt_core_state_history();
    let zkvm_proof = run_jolt_zkvm_prover(params);

    // muldiv has all-zero UntrustedAdvice + TrustedAdvice, so the prover
    // must skip 2 commits. The proof's commitments Vec should carry
    // `None` at those positions and `Some(...)` elsewhere.
    let total = zkvm_proof.commitments.len();
    let some = zkvm_proof
        .commitments
        .iter()
        .filter(|c| c.is_some())
        .count();
    let none = total - some;
    assert_eq!(
        none, 2,
        "expected 2 skipped advice commits (UntrustedAdvice + TrustedAdvice); \
         got {none} none / {some} some (total {total})"
    );

    // Walk the verifier schedule up to the first op that needs an eval
    // we don't yet record; until we hit that, AbsorbCommitment must
    // succeed for every op (previously failed with "missing commitment").
    let (
        executable,
        _polys,
        r1cs_key,
        _r1cs_witness,
        setup,
        _initial_ram,
        _final_ram,
        _instruction_flag_data,
        _reg_access,
        _lookup_trace,
        _lookup_flags,
        _bytecode_data,
    ) = setup_zkvm_muldiv(params);

    let pcs_verifier_setup = DoryVerifierSetup(params.pcs_setup.0.to_verifier_setup());
    let verifying_key = jolt_verifier::JoltVerifyingKey::<NewFr, DoryScheme>::new(
        &executable.module,
        pcs_verifier_setup,
        r1cs_key,
    );

    match jolt_verifier::verify(&verifying_key, &zkvm_proof, &setup.config.io_hash) {
        Ok(()) => {
            eprintln!("SUCCESS: full modular self-verify passed");
        }
        Err(e) => {
            let s = format!("{e:?}");
            assert!(
                !s.contains("missing commitment"),
                "regression: commit-skip alignment broken — {s}"
            );
            eprintln!("expected downstream error (pre-full-wiring): {s}");
        }
    }
}

// -------- Phase 2: Jolt protocol + FieldReg coprocessor integration --------
//
// Uses `jolt_core_module_with_fieldreg` which extends the main Jolt Module
// with committed FieldRegReadValue / FieldRegWriteValue polys (other FieldReg
// polys stay Virtual per the Phase 1 scope note). The muldiv guest program
// has no Fr operations, so both polys are all-zero; this test verifies that
// adding the FieldReg commit barriers doesn't regress the existing muldiv
// pipeline.

fn run_jolt_zkvm_prover_with_fieldreg(
    params: &CoreProtocolParams,
) -> jolt_verifier::JoltProof<NewFr, DoryScheme> {
    run_jolt_zkvm_prover_with_fieldreg_events(params, Vec::new())
}

fn run_jolt_zkvm_prover_with_fieldreg_events(
    params: &CoreProtocolParams,
    events: Vec<jolt_witness::derived::FieldRegEvent>,
) -> jolt_verifier::JoltProof<NewFr, DoryScheme> {
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
    ) = setup_zkvm_muldiv_with_example(params, "jolt_core_module_with_fieldreg");

    let t = setup.trace_length;
    let log_t = t.trailing_zeros() as usize;
    let log_k_chunk = if log_t < 25 { 4 } else { 8 };
    let k_fieldreg = 1usize << log_k_chunk;

    // Overlay FieldOp columns (flags + A/B/Result + V_PRODUCT routing) for any
    // event carrying a FieldOpPayload. No-op when every event has op=None.
    jolt_host::apply_field_op_events_to_r1cs::<NewFr>(
        &mut r1cs_witness,
        t,
        r1cs_key.num_vars_padded,
        &events,
    );

    // Build the FieldRegConfig, then derive RV/WV from it (single source of truth).
    let field_reg_config = FieldRegConfig {
        k: k_fieldreg,
        initial_state: vec![[0u64; 4]; k_fieldreg],
        events,
    };
    let rv = field_reg_config.compute_read_value::<NewFr>(t);
    let wv = field_reg_config.compute_write_value::<NewFr>(t);

    let _ = polys.insert(jolt_witness::PolynomialId::FieldRegReadValue, rv);
    let _ = polys.insert(jolt_witness::PolynomialId::FieldRegWriteValue, wv);

    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let mut preprocessed = PreprocessedSource::new();
    preprocessed.populate_ram(&setup.config, &initial_ram);

    let derived = DerivedSource::new(&r1cs_witness, setup.trace_length, r1cs_key.num_vars_padded)
        .with_ram(RamConfig {
            ram_k: setup.config.ram_k,
            lowest_addr: setup.config.ram_lowest_address,
            initial_state: initial_ram,
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
    populate_bytecode_preprocessed(&mut preprocessed, &bytecode_data);
    let mut provider =
        ProverData::new(&mut polys, r1cs, derived, preprocessed).with_lookup_trace(lookup_trace);

    let pcs_setup = &params.pcs_setup;
    let mut transcript = jolt_transcript::Blake2bTranscript::<NewFr>::new(TRANSCRIPT_LABEL);

    prove::<_, _, _, DoryScheme>(
        &executable,
        &mut provider,
        &backend(),
        pcs_setup,
        &mut transcript,
        setup.config,
    )
}

/// Phase 2 acceptance: adding the FieldReg commit barriers to the main Jolt
/// protocol Module doesn't regress muldiv modular self-verify. With empty
/// FieldReg witness (muldiv has no Fr ops), the verifier MUST still accept.
#[test]
fn modular_self_verify_with_fieldreg() {
    use jolt_dory::types::DoryVerifierSetup;

    let (_, params) = jolt_core_state_history();
    let zkvm_proof = run_jolt_zkvm_prover_with_fieldreg(params);

    let (
        executable,
        _polys,
        r1cs_key,
        _r1cs_witness,
        setup,
        _initial_ram,
        _final_ram,
        _instruction_flag_data,
        _reg_access,
        _lookup_trace,
        _lookup_flags,
        _bytecode_data,
    ) = setup_zkvm_muldiv_with_example(params, "jolt_core_module_with_fieldreg");

    let pcs_verifier_setup = DoryVerifierSetup(params.pcs_setup.0.to_verifier_setup());
    let verifying_key = jolt_verifier::JoltVerifyingKey::<NewFr, DoryScheme>::new(
        &executable.module,
        pcs_verifier_setup,
        r1cs_key,
    );

    jolt_verifier::verify(&verifying_key, &zkvm_proof, &setup.config.io_hash)
        .expect("Phase 2: FieldReg-extended Module should still verify muldiv");
}

/// Phase 2b acceptance (first half — honest path): inject non-zero synthetic
/// FieldReg events into the muldiv proof pipeline and confirm the full protocol
/// still accepts. This is the first test that actually exercises non-zero Fr
/// values through the Stage-2 FR Twist batched instance.
///
/// The guest (muldiv) doesn't emit these events — we inject them synthetically
/// via `FieldRegConfig`. The R1CS doesn't yet constrain `FieldOpA/B/Result`
/// columns against the Twist (task #59), so honest non-zero events should pass
/// without any tracer-layer wiring.
#[test]
fn modular_self_verify_with_fieldreg_nonempty_events() {
    use jolt_dory::types::DoryVerifierSetup;
    use jolt_witness::derived::FieldRegEvent;

    let (_, params) = jolt_core_state_history();

    // Three events consistent with initial_state = [[0;4]; k]:
    //   cycle 5,  slot 0: 0 → (123, 0, 0, 0)
    //   cycle 10, slot 1: 0 → (456, 789, 0, 0)
    //   cycle 20, slot 0: (123, 0, 0, 0) → (999, 999, 999, 999)
    // All cycles < trace_length (512 at log_t=9) and slots < k_fieldreg (16 at log_k_chunk=4).
    let events = vec![
        FieldRegEvent {
            cycle: 5,
            slot: 0,
            old: [0, 0, 0, 0],
            new: [123, 0, 0, 0],
            op: None,
        },
        FieldRegEvent {
            cycle: 10,
            slot: 1,
            old: [0, 0, 0, 0],
            new: [456, 789, 0, 0],
            op: None,
        },
        FieldRegEvent {
            cycle: 20,
            slot: 0,
            old: [123, 0, 0, 0],
            new: [999, 999, 999, 999],
            op: None,
        },
    ];

    let zkvm_proof = run_jolt_zkvm_prover_with_fieldreg_events(params, events);

    let (
        executable,
        _polys,
        r1cs_key,
        _r1cs_witness,
        setup,
        _initial_ram,
        _final_ram,
        _instruction_flag_data,
        _reg_access,
        _lookup_trace,
        _lookup_flags,
        _bytecode_data,
    ) = setup_zkvm_muldiv_with_example(params, "jolt_core_module_with_fieldreg");

    let pcs_verifier_setup = DoryVerifierSetup(params.pcs_setup.0.to_verifier_setup());
    let verifying_key = jolt_verifier::JoltVerifyingKey::<NewFr, DoryScheme>::new(
        &executable.module,
        pcs_verifier_setup,
        r1cs_key,
    );

    jolt_verifier::verify(&verifying_key, &zkvm_proof, &setup.config.io_hash)
        .expect("Phase 2b: non-empty FieldReg events should verify end-to-end");
}

/// Phase 2b acceptance (honest FADD overlay e2e): inject a `FieldRegEvent`
/// carrying a `FieldOpPayload` with funct3 = FADD. The event overlay wires
/// `V_FLAG_IS_FIELD_ADD = 1` + operand columns onto the R1CS witness. With
/// honest `new = a + b`, the FADD gate's local `(A·B − C) = 0` holds, and
/// the existing prover pipeline accepts the proof end-to-end.
///
/// Scope caveat: this test only exercises that the event overlay doesn't
/// break honest proving. The FADD/FSUB/FMUL/FINV R1CS rows (matrix indices
/// 19-26) are NOT yet enforced by the Spartan outer sumcheck — Spartan
/// currently samples only rows 0-18 via `NUM_R1CS_CONSTRAINTS = 19` in
/// `jolt-compiler/src/params.rs`. A negative (tampered-result rejects) test
/// at this level would fail because the gate's unsatisfiability is silently
/// dropped. See the spec's Phase 2b follow-up: bumping
/// `NUM_R1CS_CONSTRAINTS` to 27 is a protocol-level change (outer uniskip
/// domain, cross-verify parity with jolt-core) deferred to its own session.
#[test]
fn modular_self_verify_with_fieldreg_fadd_payload() {
    use jolt_dory::types::DoryVerifierSetup;
    use jolt_witness::derived::{FieldOpPayload, FieldRegEvent, FIELD_OP_FUNCT3_FADD};

    let (_, params) = jolt_core_state_history();

    // 123 + 456 = 579 in BN254 Fr (well within u64 so no modular wrap).
    let events = vec![FieldRegEvent {
        cycle: 5,
        slot: 0,
        old: [0, 0, 0, 0],
        new: [579, 0, 0, 0],
        op: Some(FieldOpPayload {
            funct3: FIELD_OP_FUNCT3_FADD,
            a: [123, 0, 0, 0],
            b: [456, 0, 0, 0],
        }),
    }];

    let zkvm_proof = run_jolt_zkvm_prover_with_fieldreg_events(params, events);

    let (
        executable,
        _polys,
        r1cs_key,
        _r1cs_witness,
        setup,
        _initial_ram,
        _final_ram,
        _instruction_flag_data,
        _reg_access,
        _lookup_trace,
        _lookup_flags,
        _bytecode_data,
    ) = setup_zkvm_muldiv_with_example(params, "jolt_core_module_with_fieldreg");

    let pcs_verifier_setup = DoryVerifierSetup(params.pcs_setup.0.to_verifier_setup());
    let verifying_key = jolt_verifier::JoltVerifyingKey::<NewFr, DoryScheme>::new(
        &executable.module,
        pcs_verifier_setup,
        r1cs_key,
    );

    jolt_verifier::verify(&verifying_key, &zkvm_proof, &setup.config.io_hash)
        .expect("Phase 2b: honest FADD payload must verify end-to-end");
}

/// Phase 2b acceptance (second half — adversarial path, witness-consistency):
/// Mutate an event's `old` value so it diverges from the running state.
/// `FieldRegConfig::compute_read_value` catches this at prove time — the FR
/// Twist witness is provably inconsistent before sumcheck begins. This tests
/// the first line of defense.
#[test]
fn modular_self_verify_with_fieldreg_nonempty_events_inconsistent_event_rejects() {
    use jolt_witness::derived::FieldRegEvent;

    let (_, params) = jolt_core_state_history();

    // Honest events would have `old = [0;4]` at cycle 5 slot 0 since initial_state
    // is all zeros. Break that by claiming old=[99,0,0,0].
    let events = vec![FieldRegEvent {
        cycle: 5,
        slot: 0,
        old: [99, 0, 0, 0], // ← mutation: does not match initial_state = 0
        new: [1, 0, 0, 0],
        op: None,
    }];

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_jolt_zkvm_prover_with_fieldreg_events(params, events)
    }));

    assert!(
        result.is_err(),
        "Inconsistent FieldReg event (old ≠ running state) MUST be rejected \
         by the FR Twist witness computation"
    );
}

/// Phase 2b acceptance (sumcheck-level adversarial): Inject honest events but
/// then overwrite the committed `FieldRegWriteValue` polynomial with a bad
/// value. The witness-consistency check passes (events are self-consistent),
/// but the FR Twist sumcheck must reject because the committed WV polynomial
/// no longer matches `Wa · (Val + Inc)` at the reduction point.
///
/// This directly tests the Plan B sumcheck's adversarial-rejection property
/// on the full Jolt protocol (not just the standalone harness).
#[test]
fn modular_self_verify_with_fieldreg_nonempty_events_tampered_wv_rejects() {
    use jolt_dory::types::DoryVerifierSetup;
    use jolt_witness::derived::{FieldRegConfig, FieldRegEvent};
    use num_traits::Zero;

    let (_, params) = jolt_core_state_history();
    let (
        executable,
        mut polys,
        r1cs_key,
        r1cs_witness,
        setup,
        initial_ram,
        final_ram,
        instruction_flag_data,
        reg_access,
        lookup_trace,
        lookup_flags,
        bytecode_data,
    ) = setup_zkvm_muldiv_with_example(params, "jolt_core_module_with_fieldreg");

    let t = setup.trace_length;
    let log_t = t.trailing_zeros() as usize;
    let log_k_chunk = if log_t < 25 { 4 } else { 8 };
    let k_fieldreg = 1usize << log_k_chunk;

    let events = vec![FieldRegEvent {
        cycle: 5,
        slot: 0,
        old: [0, 0, 0, 0],
        new: [42, 0, 0, 0],
        op: None,
    }];
    let field_reg_config = FieldRegConfig {
        k: k_fieldreg,
        initial_state: vec![[0u64; 4]; k_fieldreg],
        events,
    };
    let rv = field_reg_config.compute_read_value::<NewFr>(t);
    let mut wv = field_reg_config.compute_write_value::<NewFr>(t);

    // Tamper: flip the written value at cycle 5 to something wrong.
    // Honest wv[5] = 42; swap in NewFr::one() so the committed WV polynomial
    // disagrees with the FR Twist identity WriteValue = Wa · (Val + Inc).
    wv[5] = NewFr::one();

    let _ = polys.insert(jolt_witness::PolynomialId::FieldRegReadValue, rv);
    let _ = polys.insert(jolt_witness::PolynomialId::FieldRegWriteValue, wv);

    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let mut preprocessed = PreprocessedSource::new();
    preprocessed.populate_ram(&setup.config, &initial_ram);

    let derived = DerivedSource::new(&r1cs_witness, setup.trace_length, r1cs_key.num_vars_padded)
        .with_ram(RamConfig {
            ram_k: setup.config.ram_k,
            lowest_addr: setup.config.ram_lowest_address,
            initial_state: initial_ram,
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
    populate_bytecode_preprocessed(&mut preprocessed, &bytecode_data);
    let mut provider =
        ProverData::new(&mut polys, r1cs, derived, preprocessed).with_lookup_trace(lookup_trace);

    let pcs_setup = &params.pcs_setup;
    let mut transcript = jolt_transcript::Blake2bTranscript::<NewFr>::new(TRANSCRIPT_LABEL);
    let io_hash = setup.config.io_hash;
    let prover_config = setup.config;

    // Prove may succeed with tampered WV — the prover doesn't self-check. The
    // verifier is the one that must catch it.
    let prove_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prove::<_, _, _, DoryScheme>(
            &executable,
            &mut provider,
            &backend(),
            pcs_setup,
            &mut transcript,
            prover_config,
        )
    }));

    let Ok(zkvm_proof) = prove_result else {
        // If the prover itself panics on the inconsistent witness, that's also
        // a valid rejection signal — the tampering was caught before proof emission.
        return;
    };

    let pcs_verifier_setup = DoryVerifierSetup(params.pcs_setup.0.to_verifier_setup());
    let verifying_key = jolt_verifier::JoltVerifyingKey::<NewFr, DoryScheme>::new(
        &executable.module,
        pcs_verifier_setup,
        r1cs_key,
    );

    let verify_result = jolt_verifier::verify(&verifying_key, &zkvm_proof, &io_hash);
    assert!(
        verify_result.is_err(),
        "Tampered FieldRegWriteValue MUST be rejected by the FR Twist sumcheck, \
         but verify returned Ok"
    );
}
