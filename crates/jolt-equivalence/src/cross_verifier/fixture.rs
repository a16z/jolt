//! Honest fixture builder for the cross-verifier soundness suite.
//!
//! Produces a paired (modular proof, jolt-core scaffold, both verifier
//! setups) bundle from the `muldiv-guest` workload. Cached via
//! `OnceLock` so multiple tests in the same binary share a single
//! ~5–10s prove run.
//!
//! The setup mirrors the live path used by `muldiv.rs::modular_self_verify`
//! and `zkvm_proof_accepted_by_core_verifier`. Some helpers are duplicated
//! from `muldiv.rs` for now — a follow-up commit will dedupe.

#![allow(non_snake_case, clippy::print_stderr)]

use std::process::Command;
use std::sync::OnceLock;

use ark_bn254::Fr as ArkFr;
use jolt_core::curve::Bn254Curve;
use jolt_core::host;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::proof_serialization::JoltProof as CoreJoltProof;
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifier, JoltVerifierPreprocessing};

use common::constants::RAM_START_ADDRESS;
use jolt_compiler::module::Module;
use jolt_compiler::PolynomialId;
use jolt_compute::link;
use jolt_cpu::CpuBackend;
use jolt_dory::types::DoryProverSetup;
use jolt_dory::DoryScheme;
use jolt_field::Fr as NewFr;
use jolt_host::{extract_trace, BytecodePreprocessing, CycleRow, InstructionFlagData, Program};
use jolt_instructions::LookupTableKind;
use jolt_r1cs::{constraints::rv64, R1csKey, R1csSource};
use jolt_transcript::Transcript;
use jolt_verifier::{JoltProof as ModularJoltProof, JoltVerifyingKey};
use jolt_verifier::{OneHotConfig, ProverConfig, ReadWriteConfig, TRANSCRIPT_LABEL};
use jolt_witness::bytecode_raf::{BytecodeData, BytecodeEntry};
use jolt_witness::derived::{DerivedSource, InstructionFlags, RamConfig, RegisterAccessData};
use jolt_witness::preprocessed::PreprocessedSource;
use jolt_witness::provider::ProverData;
use jolt_witness::{PolynomialConfig, Polynomials};
use jolt_zkvm::prove::prove;
use jolt_zkvm::runtime::prefix_suffix::LookupTraceData;
use num_traits::Zero;

use jolt_core::zkvm::instruction::{
    Flags as CoreFlags, InstructionLookup, InterleavedBitsMarker as CoreInterleavedBits,
};
use jolt_core::zkvm::lookup_table::LookupTables as CoreLookupTables;

use super::conversion::CoreScaffold;

type CoreProver<'a> = JoltCpuProver<'a, ArkFr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript>;
type CoreProof = CoreJoltProof<ArkFr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript>;
type CoreVerifier<'a> =
    JoltVerifier<'a, ArkFr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript>;

/// Cached honest fixture: one modular proof, one core scaffold, both
/// verifier setups.
pub struct HonestFixture {
    /// The modular prover's output — the canonical proof under test.
    pub modular_proof: ModularJoltProof<NewFr, DoryScheme>,
    /// Jolt-core proof scaffold for the modular→core conversion.
    pub core_scaffold: CoreScaffold,
    /// Jolt-core verifier preprocessing (leaked to `'static` to live
    /// for the test binary lifetime).
    pub core_verifier_preprocessing:
        &'static JoltVerifierPreprocessing<ArkFr, Bn254Curve, DoryCommitmentScheme>,
    /// Public IO captured from the program execution.
    pub core_io: common::jolt_device::JoltDevice,
    /// Modular verifier key (leaked to `'static`).
    pub modular_verifying_key: &'static JoltVerifyingKey<NewFr, DoryScheme>,
    /// Expected IO hash (zero for muldiv-guest).
    pub io_hash: [u8; 32],
}

/// Cached protocol parameters extracted from jolt-core's prover.
pub struct CoreProtocolParams {
    pub trace_length: usize,
    pub ram_k: usize,
    pub bytecode_k: usize,
    /// PCS generators — must be the exact same SRS for commitment equivalence.
    pub pcs_setup: DoryProverSetup,
}

static FIXTURE: OnceLock<HonestFixture> = OnceLock::new();

/// Get-or-build the cached honest fixture.
pub fn fixture() -> &'static HonestFixture {
    FIXTURE.get_or_init(build_honest_fixture)
}

fn build_honest_fixture() -> HonestFixture {
    let (core_proof, core_verifier_preprocessing, core_io, params) = run_core_prover();
    let modular_proof = run_modular_prover(&params);

    let core_scaffold = CoreScaffold::from_core_proof(&core_proof);

    // Build the modular verifying key once and leak it.
    let pcs_verifier_setup =
        jolt_dory::types::DoryVerifierSetup(params.pcs_setup.0.to_verifier_setup());

    // Modular setup: rebuild executable + r1cs_key for the verifying key.
    let (executable, _polys, r1cs_key, _, _setup, _, _, _, _, _, _, _) = setup_muldiv(&params);
    let modular_verifying_key: &'static _ =
        Box::leak(Box::new(JoltVerifyingKey::<NewFr, DoryScheme>::new(
            &executable.module,
            pcs_verifier_setup,
            r1cs_key,
        )));

    HonestFixture {
        modular_proof,
        core_scaffold,
        core_verifier_preprocessing,
        core_io,
        modular_verifying_key,
        io_hash: [0u8; 32],
    }
}

fn run_core_prover() -> (
    CoreProof,
    &'static JoltVerifierPreprocessing<ArkFr, Bn254Curve, DoryCommitmentScheme>,
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

fn run_modular_prover(params: &CoreProtocolParams) -> ModularJoltProof<NewFr, DoryScheme> {
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
    ) = setup_muldiv(params);

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
        &CpuBackend,
        pcs_setup,
        &mut transcript,
        setup.config,
    )
}

/// Modular setup output — same shape as `muldiv.rs::setup_zkvm_muldiv`.
pub struct ModularSetup {
    pub trace_length: usize,
    pub config: ProverConfig,
}

#[allow(clippy::type_complexity)]
fn setup_muldiv(
    core_params: &CoreProtocolParams,
) -> (
    jolt_compute::Executable<CpuBackend, NewFr>,
    Polynomials<NewFr>,
    R1csKey<NewFr>,
    Vec<NewFr>,
    ModularSetup,
    Vec<u64>,
    Vec<u64>,
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

    let mut module = build_protocol_module(log_t, log_k_bytecode, log_k_ram);
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
        is_interleaved[t] = CoreInterleavedBits::is_interleaved_operands(&cycle.circuit_flags());
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
                is_interleaved: CoreInterleavedBits::is_interleaved_operands(&circuit_flags),
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
    pc_indices.resize(trace_length, 0);

    let bytecode_data = BytecodeData {
        pc_indices,
        entries: bc_entries,
        entry_index: bytecode.entry_bytecode_index(),
        num_lookup_tables: jolt_compiler::params::NUM_LOOKUP_TABLES,
    };

    let setup = ModularSetup {
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

    let pc_idx_poly: Vec<NewFr> = bc
        .pc_indices
        .iter()
        .map(|&i| NewFr::from_u64(i as u64))
        .collect();
    preprocessed.insert(PolynomialId::BytecodePcIndex, pc_idx_poly);

    let pc_0 = bc.pc_indices[0];
    let mut entry_trace = vec![NewFr::zero(); k];
    entry_trace[pc_0] = NewFr::from_u64(1);
    preprocessed.insert(PolynomialId::BytecodeEntryTrace, entry_trace);

    let mut entry_expected = vec![NewFr::zero(); k];
    entry_expected[bc.entry_index] = NewFr::from_u64(1);
    preprocessed.insert(PolynomialId::BytecodeEntryExpected, entry_expected);

    bc.populate_preprocessed(preprocessed);
}

fn build_protocol_module(log_t: usize, log_k_bytecode: usize, log_k_ram: usize) -> Module {
    let tmp_path = format!("/tmp/jolt_equiv_module_{log_t}_{log_k_bytecode}_{log_k_ram}.jolt");

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
    use jolt_compiler::{Op, VerifierOp};
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

    let _ = seen;
}

/// Verify a core proof using jolt-core's verifier. Returns `Ok(())` on
/// acceptance, error string on rejection.
pub fn verify_with_core(proof: CoreProof, fixture: &HonestFixture) -> Result<(), String> {
    let mut verifier = CoreVerifier::new(
        fixture.core_verifier_preprocessing,
        proof,
        fixture.core_io.clone(),
        None,
        None,
    )
    .map_err(|e| format!("verifier construction: {e:?}"))?;

    verifier.run_preamble();
    let _ = verifier
        .verify_stage1()
        .map_err(|e| format!("stage1: {e:?}"))?;
    let _ = verifier
        .verify_stage2()
        .map_err(|e| format!("stage2: {e:?}"))?;
    let _ = verifier
        .verify_stage3()
        .map_err(|e| format!("stage3: {e:?}"))?;
    let _ = verifier
        .verify_stage4()
        .map_err(|e| format!("stage4: {e:?}"))?;
    let _ = verifier
        .verify_stage5()
        .map_err(|e| format!("stage5: {e:?}"))?;
    let _ = verifier
        .verify_stage6()
        .map_err(|e| format!("stage6: {e:?}"))?;
    let _ = verifier
        .verify_stage7()
        .map_err(|e| format!("stage7: {e:?}"))?;
    let _ = verifier
        .verify_stage8()
        .map_err(|e| format!("stage8: {e:?}"))?;
    Ok(())
}
