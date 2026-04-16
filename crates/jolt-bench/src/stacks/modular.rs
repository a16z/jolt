//! Modular-stack benchmark runner.
//!
//! Today the modular stack is only reachable for `muldiv` via the same
//! prove+verify pattern exercised by the `zkvm_proof_accepted` gate test in
//! `jolt-equivalence`. The runner mirrors that path: a DoryScheme modular
//! prover produces a `jolt_verifier::JoltProof`, which we transplant into
//! a jolt-core `JoltProof` scaffold and verify with the jolt-core verifier.
//! This is not a pure modular verify — see the `verify_note` emitted on
//! each row in the JSON output.
//!
//! Scaffolding runs (jolt-core prover) sit outside the measurement window.
//! Only `jolt_zkvm::prove::prove()` is timed for `prove_ms`, and only
//! `JoltVerifier::verify_stage*` for `verify_ms`.
//!
//! For non-muldiv programs this runner returns an unsupported row because
//! the modular witness-assembly path has no generalized helper today.

use std::process::Command;

use ark_bn254::Fr as ArkFr;
use jolt_core::curve::Bn254Curve;
use jolt_core::field::JoltField;
use jolt_core::host;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::Blake2bTranscript as CoreBlake2bTranscript;
use jolt_core::zkvm::instruction::{
    Flags as CoreFlags, InstructionLookup, InterleavedBitsMarker as CoreInterleavedBits,
};
use jolt_core::zkvm::lookup_table::LookupTables as CoreLookupTables;
use jolt_core::zkvm::proof_serialization::JoltProof as CoreJoltProof;
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifier, JoltVerifierPreprocessing};

use common::constants::RAM_START_ADDRESS;
use jolt_compiler::module::Module;
use jolt_compiler::{Op, VerifierOp};
use jolt_compute::{link, LookupTraceData};
use jolt_cpu::CpuBackend;
use jolt_dory::types::DoryProverSetup;
use jolt_dory::DoryScheme;
use jolt_host::{
    extract_trace, BytecodePreprocessing, CycleRow, InstructionFlagData, Program as HostProgram,
};
use jolt_instructions::LookupTableKind;
use jolt_r1cs::{constraints::rv64, R1csKey, R1csSource};
use jolt_transcript::{Blake2bTranscript as ModBlake2bTranscript, Transcript};
use jolt_verifier::{OneHotConfig, ProverConfig, ReadWriteConfig, TRANSCRIPT_LABEL};
use jolt_witness::bytecode_raf::{BytecodeData, BytecodeEntry};
use jolt_witness::derived::{DerivedSource, InstructionFlags, RamConfig, RegisterAccessData};
use jolt_witness::preprocessed::PreprocessedSource;
use jolt_witness::provider::ProverData;
use jolt_witness::{PolynomialConfig, PolynomialId, Polynomials};
use jolt_zkvm::prove::prove as modular_prove;
use num_traits::{One, Zero};

use super::{IterMetrics, StackOutcome, StackRunner};
use crate::measure::{time_it, PeakRssSampler};
use crate::output::{Run, StackLabel};
use crate::programs::Program;

type ModFr = jolt_field::Fr;
type CoreProver<'a> =
    JoltCpuProver<'a, ArkFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;
type CoreProof = CoreJoltProof<ArkFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;
type CoreVerifierAlias<'a> =
    JoltVerifier<'a, ArkFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;

const MAX_TRACE_LENGTH: usize = 1 << 16;

fn to_ark(f: ModFr) -> ArkFr {
    f.into()
}

pub struct ModularStack;

impl ModularStack {
    fn run_muldiv_once(inputs: &[u8]) -> IterMetrics {
        DoryGlobals::reset();

        // -- 1. Scaffolding run: jolt-core prover. Outside measurement window.
        let mut core_program = host::Program::new("muldiv-guest");
        let (bytecode, init_memory_state, _, e_entry) = core_program.decode();
        let (_, _, _, io_device_core) = core_program.trace(inputs, &[], &[]);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode,
            io_device_core.memory_layout.clone(),
            init_memory_state,
            MAX_TRACE_LENGTH,
            e_entry,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
        let elf_contents = core_program
            .get_elf_contents()
            .expect("muldiv ELF should be available after decode()");
        let core_prover = CoreProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            inputs,
            &[],
            &[],
            None,
            None,
            None,
        );
        let core_io = core_prover.program_io.clone();
        let (core_proof, _debug): (CoreProof, _) = core_prover.prove();

        let core_params = CoreProtocolParams {
            trace_length: core_proof.trace_length,
            ram_k: core_proof.ram_K,
            bytecode_k: prover_preprocessing.shared.bytecode.code_size,
            pcs_setup: DoryProverSetup(prover_preprocessing.generators.clone()),
        };
        let verifier_preprocessing = Box::leak(Box::new(JoltVerifierPreprocessing::from(
            &prover_preprocessing,
        )));

        // -- 2. Modular prover setup (also outside measurement window).
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
        ) = setup_zkvm_muldiv(&core_params, inputs);

        let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
        let derived =
            DerivedSource::new(&r1cs_witness, setup.trace_length, r1cs_key.num_vars_padded)
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
        let mut provider = ProverData::new(&mut polys, r1cs, derived, preprocessed);

        let pcs_setup = &core_params.pcs_setup;
        let mut transcript = ModBlake2bTranscript::<ModFr>::new(TRANSCRIPT_LABEL);

        // -- 3. Timed modular prove.
        let sampler = PeakRssSampler::start();
        let (prove_ms, zkvm_proof) = time_it(|| {
            modular_prove::<_, _, _, DoryScheme>(
                &executable,
                &mut provider,
                &CpuBackend,
                pcs_setup,
                &mut transcript,
                setup.config.clone(),
                Some(lookup_trace),
                Some(bytecode_data),
            )
        });
        let peak_rss_mb = sampler.finish();

        // Serialize modular proof via bincode (the format it was designed for).
        let proof_bytes = bincode::serde::encode_to_vec(&zkvm_proof, bincode::config::standard())
            .expect("serialize modular proof")
            .len() as u64;

        // -- 4. Transplant modular proof into core proof scaffold, verify via core.
        let converted_proof = transplant(core_proof, &zkvm_proof);
        let mut verifier =
            CoreVerifierAlias::new(verifier_preprocessing, converted_proof, core_io, None, None)
                .expect("build core verifier for transplanted modular proof");
        let (verify_ms, ()) = time_it(|| {
            verifier.run_preamble();
            let _ = verifier.verify_stage1().expect("stage 1 verify");
            let _ = verifier.verify_stage2().expect("stage 2 verify");
            let _ = verifier.verify_stage3().expect("stage 3 verify");
            let _ = verifier.verify_stage4().expect("stage 4 verify");
            let _ = verifier.verify_stage5().expect("stage 5 verify");
            let _ = verifier.verify_stage6().expect("stage 6 verify");
            let _ = verifier.verify_stage7().expect("stage 7 verify");
            let _ = verifier.verify_stage8().expect("stage 8 verify");
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
    fn run(&self, program: Program, iters: usize, warmup: usize) -> StackOutcome {
        if program != Program::Muldiv {
            return StackOutcome::Unsupported(Run::unsupported(
                StackLabel::Modular,
                format!(
                    "modular stack currently lacks a generic witness-assembly path; only \
                     muldiv is wired up today (program={})",
                    program.cli_name()
                ),
            ));
        }

        let inputs = program.canonical_inputs();
        for _ in 0..warmup {
            let _ = Self::run_muldiv_once(&inputs);
        }
        let measurements = (0..iters).map(|_| Self::run_muldiv_once(&inputs)).collect();
        StackOutcome::Metrics(measurements)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Modular muldiv setup — ported from jolt-equivalence/tests/muldiv.rs
// (setup_zkvm_muldiv, build_protocol_module, truncate_after_stage,
// populate_bytecode_preprocessed, proof transplant).
//
// Duplicated here because those helpers are `#[cfg(test)]` only. When a
// public helper lands in jolt-zkvm (or jolt-host), delete this section.
// ═══════════════════════════════════════════════════════════════════

struct CoreProtocolParams {
    trace_length: usize,
    ram_k: usize,
    bytecode_k: usize,
    pcs_setup: DoryProverSetup,
}

struct ZkvmSetup {
    trace_length: usize,
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

#[allow(clippy::type_complexity)]
fn setup_zkvm_muldiv(
    core_params: &CoreProtocolParams,
    inputs: &[u8],
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
    let mut program = HostProgram::new("muldiv-guest");
    let (bytecode_raw, init_mem, _program_size, entry_address) = program.decode();
    let (_, trace, final_memory, io_device) = program.trace(inputs, &[], &[]);

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
    let executable = link::<CpuBackend, ModFr>(module, &backend);

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
    let matrices = rv64::rv64_constraints::<ModFr>();
    let r1cs_key = R1csKey::new(matrices, trace_length);

    let (cycle_inputs, r1cs_witness, instruction_flag_data) = extract_trace::<_, ModFr>(
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
    let mut table_kinds: Vec<Option<LookupTableKind>> = vec![None; trace_length];
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
        table_kinds[t] = cycle.lookup_table_index().map(|idx| {
            assert!(idx < LookupTableKind::COUNT);
            // SAFETY: LookupTableKind is #[repr(u8)] with contiguous discriminants
            // 0..COUNT, and idx < COUNT is asserted above.
            unsafe { std::mem::transmute::<u8, LookupTableKind>(idx as u8) }
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
        table_kinds,
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
        jolt_host::ram::build_ram_states(&init_mem, &final_memory, &io_device, ram_k);

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
}

// ── proof transplant: modular → core ─────────────────────────────────────

fn to_compressed_uni_poly(
    poly: &jolt_poly::UnivariatePoly<ModFr>,
) -> jolt_core::poly::unipoly::CompressedUniPoly<ArkFr> {
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

fn to_core_sumcheck_proof(
    round_polys: &[jolt_poly::UnivariatePoly<ModFr>],
) -> SumcheckInstanceProof<ArkFr, Bn254Curve, CoreBlake2bTranscript> {
    let compressed: Vec<_> = round_polys.iter().map(to_compressed_uni_poly).collect();
    SumcheckInstanceProof::Clear(jolt_core::subprotocols::sumcheck::ClearSumcheckProof::new(
        compressed,
    ))
}

fn commitment_to_ark(
    c: &jolt_dory::types::DoryCommitment,
) -> <DoryCommitmentScheme as jolt_core::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment
{
    // SAFETY: DoryCommitment(Bn254GT) and ArkGT are both repr(transparent)
    // over the same Fq12 type.
    unsafe { std::mem::transmute_copy(&c.0) }
}

fn transplant(
    core_proof: CoreProof,
    zkvm_proof: &jolt_verifier::JoltProof<ModFr, DoryScheme>,
) -> CoreProof {
    assert_eq!(
        zkvm_proof.stage_proofs.len(),
        8,
        "expected 8 stage proofs from jolt-zkvm (7 sumcheck + 1 PCS opening)"
    );

    let stage1 =
        to_core_sumcheck_proof(&zkvm_proof.stage_proofs[0].round_polys.round_polynomials[1..]);
    let stage2 =
        to_core_sumcheck_proof(&zkvm_proof.stage_proofs[1].round_polys.round_polynomials[1..]);
    let stage3 = to_core_sumcheck_proof(&zkvm_proof.stage_proofs[2].round_polys.round_polynomials);
    let stage4 = to_core_sumcheck_proof(&zkvm_proof.stage_proofs[3].round_polys.round_polynomials);
    let stage5 = to_core_sumcheck_proof(&zkvm_proof.stage_proofs[4].round_polys.round_polynomials);
    let stage6 = to_core_sumcheck_proof(&zkvm_proof.stage_proofs[5].round_polys.round_polynomials);
    let stage7 = to_core_sumcheck_proof(&zkvm_proof.stage_proofs[6].round_polys.round_polynomials);

    let expected_num_commitments = core_proof.commitments.len();
    let commitments: Vec<_> = zkvm_proof
        .commitments
        .iter()
        .take(expected_num_commitments)
        .map(commitment_to_ark)
        .collect();

    assert_eq!(
        zkvm_proof.opening_proofs.len(),
        1,
        "expected exactly 1 joint opening proof"
    );
    let joint_opening_proof = zkvm_proof.opening_proofs[0].0.clone();

    CoreJoltProof {
        commitments,
        stage1_sumcheck_proof: stage1,
        stage2_sumcheck_proof: stage2,
        stage3_sumcheck_proof: stage3,
        stage4_sumcheck_proof: stage4,
        stage5_sumcheck_proof: stage5,
        stage6_sumcheck_proof: stage6,
        stage7_sumcheck_proof: stage7,
        joint_opening_proof,
        // Moved from core_proof (Claims<F> has no Clone impl, so we consume the proof).
        stage1_uni_skip_first_round_proof: core_proof.stage1_uni_skip_first_round_proof,
        stage2_uni_skip_first_round_proof: core_proof.stage2_uni_skip_first_round_proof,
        untrusted_advice_commitment: core_proof.untrusted_advice_commitment,
        opening_claims: core_proof.opening_claims,
        trace_length: core_proof.trace_length,
        ram_K: core_proof.ram_K,
        rw_config: core_proof.rw_config,
        one_hot_config: core_proof.one_hot_config,
        dory_layout: core_proof.dory_layout,
    }
}

// Keep the JoltField bound reachable — `to_ark` relies on JoltField's blanket
// conversion between ark-bn254 and jolt-field Fr.
#[allow(dead_code)]
const _: fn() = || {
    fn assert_jolt_field<F: JoltField>() {}
    assert_jolt_field::<ArkFr>();
};
