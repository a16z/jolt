//! Core-side public oracle fixture and acceptance checks for equivalence gates.
//!
//! This module runs jolt-core as the temporary reference oracle and exposes the
//! public data needed by the Bolt equivalence tests. It does not patch expected
//! values to compensate for Bolt mismatches.

#![expect(
    clippy::expect_used,
    clippy::too_many_arguments,
    reason = "oracle fixtures should fail fast when reference setup is malformed"
)]
#![expect(
    dead_code,
    reason = "cross-stack assertions against jolt-core are disabled on this branch due to FR coprocessor R1CS divergence"
)]

use std::time::Instant;

use ark_serialize::{CanonicalSerialize, Compress};
use bolt::protocols::jolt::JoltProtocolParams;
use common::constants::{RAM_START_ADDRESS, XLEN};
use common::jolt_device::JoltDevice;
use jolt_core::curve::Bn254Curve;
use jolt_core::host;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme as CoreCommitmentScheme;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::Blake2bTranscript as CoreBlake2bTranscript;
use jolt_core::zkvm::instruction::InstructionLookup;
use jolt_core::zkvm::lookup_table::LookupTables as CoreLookupTables;
use jolt_core::zkvm::proof_serialization::JoltProof as CoreJoltProof;
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifier, JoltVerifierPreprocessing};
use jolt_dory::DoryProverSetup;
use jolt_field::Fr;
use jolt_kernels::stage1::{
    Stage1ExecutionArtifacts, Stage1OuterR1csData, Stage1OuterRv64Data, Stage1Rv64Cycle,
};
use jolt_kernels::stage2::{
    Stage2ExecutionArtifacts, Stage2InstructionLookupCycle, Stage2ProductVirtualCycle,
    Stage2RamAccess, Stage2RamData, Stage2RamOutputLayout,
};
use jolt_kernels::stage3::{Stage3Cycle, Stage3ExecutionArtifacts};
use jolt_kernels::stage4::{Stage4ExecutionArtifacts, Stage4RegisterAccess};
use jolt_kernels::stage6::Stage6WitnessParams;
use jolt_kernels::trace::{
    stage1_rv64_cycles, stage2_instruction_lookup_cycles, stage2_product_virtual_cycles,
    stage2_ram_accesses, stage3_cycles, stage4_register_accesses, stage5_lookup_trace,
    stage6_bytecode_entries,
};
use jolt_r1cs::{constraints::rv64, R1csKey};
use jolt_trace::{extract_trace, BytecodePreprocessing};
use jolt_verifier::JoltStageProof;
use jolt_witness::{CycleInput, Stage6BytecodeEntry};
use strum::EnumCount;

use crate::checks::{
    assert_core_stage1_uniskip_proof_matches_bolt, assert_core_stage2_opening_claims_match_bolt,
    assert_core_stage2_sumcheck_proof_matches_bolt, assert_core_stage3_opening_claims_match_bolt,
    assert_core_stage3_sumcheck_proof_matches_bolt, assert_core_stage6_opening_claims_match_bolt,
    assert_core_stage6_sumcheck_proof_matches_bolt,
};
use crate::commitment_oracle::BoltPreambleSource;
use crate::core_conversion::{
    clone_core_proof, core_proof_with_bolt_evaluation, core_proof_with_bolt_stage1,
    core_proof_with_bolt_stage2, core_proof_with_bolt_stage3, core_proof_with_bolt_stage4,
    core_proof_with_bolt_stage5, core_proof_with_bolt_stage6, core_proof_with_bolt_stage7,
    core_proof_with_full_bolt,
};
use jolt_profiling::{observed_span_names_with_prefix, time_it, PeakRssSampler, PerfMetrics};

pub type CoreFr = ark_bn254::Fr;
pub type CoreCommitment = <DoryCommitmentScheme as CoreCommitmentScheme>::Commitment;
type CoreProver<'a> =
    JoltCpuProver<'a, CoreFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;
pub type CoreProof = CoreJoltProof<CoreFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;
type CoreVerifier<'a> =
    JoltVerifier<'a, CoreFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;
pub type CoreVerifierPreprocessing =
    JoltVerifierPreprocessing<CoreFr, Bn254Curve, DoryCommitmentScheme>;

fn assert_core_verifies_proof(
    fixture: &CoreMuldivCommitmentFixture,
    proof: CoreProof,
    message: &str,
) {
    CoreVerifier::new(
        fixture.verifier_preprocessing,
        proof,
        fixture.io.clone(),
        None,
        None,
    )
    .expect("construct core verifier")
    .verify()
    .expect(message);
}

pub struct CoreMuldivCommitmentFixture {
    pub params: JoltProtocolParams,
    pub(crate) core_metrics: PerfMetrics,
    pub pcs_setup: DoryProverSetup,
    pub proof: CoreProof,
    pub(crate) verifier_preprocessing: &'static CoreVerifierPreprocessing,
    pub io: JoltDevice,
    pub entry_address: u64,
    pub cycle_inputs: Vec<CycleInput>,
    pub r1cs_witness: Vec<Fr>,
    pub rv64_cycles: Vec<Stage1Rv64Cycle>,
    pub product_virtual_cycles: Vec<Stage2ProductVirtualCycle>,
    pub instruction_lookup_cycles: Vec<Stage2InstructionLookupCycle>,
    pub stage3_cycles: Vec<Stage3Cycle>,
    pub stage4_register_accesses: Vec<Stage4RegisterAccess>,
    pub stage5_lookup_indices: Vec<u128>,
    pub stage5_lookup_table_indices: Vec<Option<usize>>,
    pub stage5_is_interleaved_operands: Vec<bool>,
    pub stage6_bytecode_entries: Vec<Stage6BytecodeEntry<Fr>>,
    pub stage6_entry_bytecode_index: usize,
    pub stage6_num_lookup_tables: usize,
    pub ram_accesses: Vec<Stage2RamAccess>,
    pub initial_ram_state: Vec<u64>,
    pub final_ram_state: Vec<u64>,
    pub ram_start_address: u64,
    pub ram_output_layout: Stage2RamOutputLayout,
    pub commitments: Vec<CoreCommitment>,
}

impl CoreMuldivCommitmentFixture {
    pub fn stage2_ram_data(&self) -> Stage2RamData<'_> {
        Stage2RamData {
            log_k: self.params.log_k_ram,
            start_address: self.ram_start_address,
            initial_ram: &self.initial_ram_state,
            final_ram: &self.final_ram_state,
            accesses: &self.ram_accesses,
            output_layout: Some(self.ram_output_layout),
        }
    }

    pub fn r1cs_key(&self) -> R1csKey<Fr> {
        R1csKey::new(rv64::rv64_constraints::<Fr>(), self.proof.trace_length)
    }

    pub fn stage1_outer_rv64_data<'a>(
        &'a self,
        r1cs_key: &'a R1csKey<Fr>,
    ) -> Stage1OuterRv64Data<'a> {
        Stage1OuterRv64Data::new(r1cs_key, &self.r1cs_witness, &self.rv64_cycles)
            .expect("valid RV64-backed stage1 data")
    }

    pub fn stage1_outer_r1cs_data<'a>(
        &'a self,
        r1cs_key: &'a R1csKey<Fr>,
    ) -> Stage1OuterR1csData<'a, Fr> {
        Stage1OuterR1csData::new(r1cs_key, &self.r1cs_witness).expect("valid R1CS witness shape")
    }

    pub fn stage6_witness_params(&self) -> Stage6WitnessParams {
        Stage6WitnessParams {
            trace_len: self.proof.trace_length,
            log_k_chunk: self.params.log_k_chunk,
            log_k_bytecode: self.params.log_k_bytecode,
            log_k_ram: self.params.log_k_ram,
            lookups_ra_virtual_log_k_chunk: self.params.lookups_ra_virtual_log_k_chunk,
            instruction_d: self.params.instruction_d,
            instruction_ra_virtual_d: self.params.instruction_ra_virtual_d,
            bytecode_d: self.params.bytecode_d,
            ram_d: self.params.ram_d,
        }
    }
}

impl BoltPreambleSource for CoreMuldivCommitmentFixture {
    fn program_io(&self) -> &JoltDevice {
        &self.io
    }

    fn preprocessing_digest(&self) -> [u8; 32] {
        self.verifier_preprocessing.shared.digest()
    }

    fn ram_k(&self) -> u64 {
        self.proof.ram_K as u64
    }

    fn trace_length(&self) -> u64 {
        self.proof.trace_length as u64
    }

    fn entry_address(&self) -> u64 {
        self.entry_address
    }

    fn ram_rw_phase1_num_rounds(&self) -> u64 {
        self.proof.rw_config.ram_rw_phase1_num_rounds as u64
    }

    fn ram_rw_phase2_num_rounds(&self) -> u64 {
        self.proof.rw_config.ram_rw_phase2_num_rounds as u64
    }

    fn registers_rw_phase1_num_rounds(&self) -> u64 {
        self.proof.rw_config.registers_rw_phase1_num_rounds as u64
    }

    fn registers_rw_phase2_num_rounds(&self) -> u64 {
        self.proof.rw_config.registers_rw_phase2_num_rounds as u64
    }

    fn log_k_chunk(&self) -> u64 {
        self.proof.one_hot_config.log_k_chunk as u64
    }

    fn lookups_ra_virtual_log_k_chunk(&self) -> u64 {
        self.proof.one_hot_config.lookups_ra_virtual_log_k_chunk as u64
    }

    fn dory_layout(&self) -> u64 {
        self.proof.dory_layout as u64
    }
}

pub fn core_muldiv_commitment_fixture() -> CoreMuldivCommitmentFixture {
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("muldiv inputs");
    core_guest_commitment_fixture("muldiv-guest", inputs, 1 << 16)
}

pub fn core_muldiv_commitment_fixture_at_log_t(log_t: usize) -> CoreMuldivCommitmentFixture {
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("muldiv inputs");
    core_guest_commitment_fixture("muldiv-guest", inputs, 1usize << log_t)
}

pub fn core_sha2_chain_commitment_fixture(log_t: usize) -> CoreMuldivCommitmentFixture {
    let target_cycles = ((1usize << log_t) as f64 * 0.9) as usize;
    let num_iters = std::cmp::max(1, (target_cycles as f64 / 3396.0) as u32);
    let mut inputs = Vec::new();
    inputs.extend(postcard::to_stdvec(&[5u8; 32]).expect("sha2-chain input"));
    inputs.extend(postcard::to_stdvec(&num_iters).expect("sha2-chain iterations"));
    core_guest_commitment_fixture("sha2-chain-guest", inputs, 1usize << log_t)
}

fn core_guest_commitment_fixture(
    guest_name: &str,
    inputs: Vec<u8>,
    max_trace_length: usize,
) -> CoreMuldivCommitmentFixture {
    let setup_start = Instant::now();
    let _core_setup_span = tracing::info_span!("core.setup").entered();

    let mut core_program = host::Program::new(guest_name);
    let (core_bytecode, init_memory_state, _, entry_address) = core_program.decode();
    let core_bytecode_for_bolt = core_bytecode.clone();
    let (_, trace, _, host_io_device) = core_program.trace(&inputs, &[], &[]);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        core_bytecode,
        host_io_device.memory_layout.clone(),
        init_memory_state,
        max_trace_length,
        entry_address,
    )
    .expect("shared preprocessing");
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
    let elf_contents = core_program.get_elf_contents().expect("guest elf");
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
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1_000.0;
    drop(_core_setup_span);
    let io = prover.program_io.clone();
    let initial_ram_state = prover.initial_ram_state.clone();
    let final_ram_state = prover.final_ram_state.clone();
    let core_rss_sampler = PeakRssSampler::start().expect("start core RSS sampler");
    let _core_prove_span = tracing::info_span!("core.prove").entered();
    let (prove_ms, (proof, _debug)) = time_it(|| prover.prove());
    drop(_core_prove_span);
    let peak_rss_mb = core_rss_sampler.finish();
    let proof_bytes = proof.serialized_size(Compress::Yes) as u64;
    let verifier_preprocessing: &'static _ = Box::leak(Box::new(JoltVerifierPreprocessing::from(
        &prover_preprocessing,
    )));
    let core_verifier = CoreVerifier::new(
        verifier_preprocessing,
        clone_core_proof(&proof),
        io.clone(),
        None,
        None,
    )
    .expect("construct core verifier");
    let _core_verify_span = tracing::info_span!("core.verify").entered();
    let (verify_ms, verify_result) = time_it(|| core_verifier.verify());
    drop(_core_verify_span);
    verify_result.expect("core verifier accepts proof");
    let core_metrics = PerfMetrics {
        setup_ms: Some(setup_ms),
        prove_ms: Some(prove_ms),
        verify_ms: Some(verify_ms),
        proof_bytes: Some(proof_bytes),
        peak_rss_mb: Some(peak_rss_mb),
        span_names: observed_span_names_with_prefix("core."),
    };

    let mut padded_trace = trace.clone();
    padded_trace.resize(proof.trace_length, jolt_trace::Cycle::NoOp);
    let product_virtual_cycles = stage2_product_virtual_cycles(&padded_trace, proof.trace_length);
    let instruction_lookup_cycles =
        stage2_instruction_lookup_cycles(&padded_trace, proof.trace_length);
    let ram_accesses = stage2_ram_accesses(&padded_trace, proof.trace_length, |address| {
        remap_address(address, &host_io_device.memory_layout).map(|address| address as usize)
    });
    let ram_start_address = host_io_device.memory_layout.get_lowest_address();
    let ram_output_layout = Stage2RamOutputLayout {
        io_start: remap_address(
            host_io_device.memory_layout.input_start,
            &host_io_device.memory_layout,
        )
        .expect("input start remaps") as usize,
        io_end: remap_address(RAM_START_ADDRESS, &host_io_device.memory_layout)
            .expect("RAM start remaps") as usize,
    };
    let bytecode = BytecodePreprocessing::preprocess(core_bytecode_for_bolt, entry_address);
    let stage6_bytecode_entries = stage6_bytecode_entries(&bytecode, |instruction| {
        InstructionLookup::<XLEN>::lookup_table(instruction)
            .map(|table| CoreLookupTables::<XLEN>::enum_index(&table))
    });
    let stage6_entry_bytecode_index = bytecode.entry_bytecode_index();
    let stage6_num_lookup_tables = CoreLookupTables::<XLEN>::COUNT;
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), proof.trace_length);
    let (cycle_inputs, r1cs_witness, _) = extract_trace::<_, Fr>(
        &trace,
        proof.trace_length,
        &bytecode,
        &host_io_device.memory_layout,
        r1cs_key.num_vars_padded,
        &[], // no FR events in the core oracle path
    );
    let rv64_cycles = stage1_rv64_cycles(&trace, proof.trace_length, &bytecode);
    let stage3_cycles = stage3_cycles(&trace, proof.trace_length, &bytecode);
    let stage4_register_accesses = stage4_register_accesses(&trace, proof.trace_length);
    let stage5_trace = stage5_lookup_trace(&padded_trace, proof.trace_length, |cycle| {
        InstructionLookup::<XLEN>::lookup_table(cycle)
            .map(|table| CoreLookupTables::<XLEN>::enum_index(&table))
    });
    let log_t = proof.trace_length.trailing_zeros() as usize;
    let log_k_bytecode = prover_preprocessing
        .shared
        .bytecode
        .code_size
        .trailing_zeros() as usize;
    let log_k_ram = proof.ram_K.trailing_zeros() as usize;
    let params = JoltProtocolParams::new(log_t, log_k_bytecode, log_k_ram);

    let commitments = proof.commitments.clone();

    CoreMuldivCommitmentFixture {
        params,
        core_metrics,
        pcs_setup: DoryProverSetup(prover_preprocessing.generators.clone()),
        proof,
        verifier_preprocessing,
        io,
        entry_address,
        cycle_inputs,
        r1cs_witness,
        rv64_cycles,
        product_virtual_cycles,
        instruction_lookup_cycles,
        stage3_cycles,
        stage4_register_accesses,
        stage5_lookup_indices: stage5_trace.lookup_indices,
        stage5_lookup_table_indices: stage5_trace.lookup_table_indices,
        stage5_is_interleaved_operands: stage5_trace.is_interleaved_operands,
        stage6_bytecode_entries,
        stage6_entry_bytecode_index,
        stage6_num_lookup_tables,
        ram_accesses,
        initial_ram_state,
        final_ram_state,
        ram_start_address,
        ram_output_layout,
        commitments,
    }
}

pub fn assert_core_accepts_bolt_stage1(
    fixture: &CoreMuldivCommitmentFixture,
    artifacts: &Stage1ExecutionArtifacts<Fr>,
) {
    assert_core_stage1_uniskip_proof_matches_bolt(&fixture.proof, &artifacts.sumchecks[0]);
    let proof = core_proof_with_bolt_stage1(&fixture.proof, artifacts);
    assert_core_verifies_proof(fixture, proof, "jolt-core accepts Bolt Stage 1 proof");
}

pub fn assert_core_accepts_bolt_stage2(
    fixture: &CoreMuldivCommitmentFixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
) {
    let proof = core_proof_with_bolt_stage2(&fixture.proof, stage1_artifacts, stage2_artifacts);
    assert_core_stage2_sumcheck_proof_matches_bolt(&fixture.proof, &stage2_artifacts.sumchecks[1]);
    assert_core_stage2_opening_claims_match_bolt(&fixture.proof, stage2_artifacts);

    assert_core_verifies_proof(fixture, proof, "core accepts Bolt Stage 2");
}

pub(crate) fn assert_core_accepts_bolt_stage3(
    fixture: &CoreMuldivCommitmentFixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
) {
    let proof = core_proof_with_bolt_stage3(
        &fixture.proof,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
    );
    assert_core_stage3_sumcheck_proof_matches_bolt(&fixture.proof, &stage3_artifacts.sumchecks[0]);
    assert_core_stage3_opening_claims_match_bolt(&fixture.proof, stage3_artifacts);

    assert_core_verifies_proof(fixture, proof, "core accepts Bolt Stage 3");
}

pub(crate) fn assert_core_accepts_bolt_stage4(
    fixture: &CoreMuldivCommitmentFixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
) {
    let proof = core_proof_with_bolt_stage4(
        &fixture.proof,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
    );
    assert_core_verifies_proof(fixture, proof, "core accepts Bolt Stage 4");
}

pub(crate) fn assert_core_accepts_bolt_stage5(
    fixture: &CoreMuldivCommitmentFixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
    stage5_proof: &JoltStageProof,
) {
    let proof = core_proof_with_bolt_stage5(
        &fixture.proof,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
        stage5_proof,
    );
    assert_core_verifies_proof(fixture, proof, "core accepts Bolt Stage 5");
}

pub(crate) fn assert_core_accepts_bolt_stage6(
    fixture: &CoreMuldivCommitmentFixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
    stage5_proof: &JoltStageProof,
    stage6_proof: &JoltStageProof,
) {
    let proof = core_proof_with_bolt_stage6(
        &fixture.proof,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
        stage5_proof,
        stage6_proof,
    );
    assert_core_stage6_sumcheck_proof_matches_bolt(&fixture.proof, &stage6_proof.sumchecks[0]);
    assert_core_stage6_opening_claims_match_bolt(&fixture.proof, stage6_proof);
    assert_core_verifies_proof(fixture, proof, "core accepts Bolt Stage 6");
}

pub(crate) fn assert_core_accepts_bolt_stage7(
    fixture: &CoreMuldivCommitmentFixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
    stage5_proof: &JoltStageProof,
    stage6_proof: &JoltStageProof,
    stage7_proof: &JoltStageProof,
) {
    let proof = core_proof_with_bolt_stage7(
        &fixture.proof,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
        stage5_proof,
        stage6_proof,
        stage7_proof,
    );
    assert_core_verifies_proof(fixture, proof, "core accepts Bolt Stage 7");
}

pub(crate) fn assert_core_accepts_bolt_evaluation_proof(
    fixture: &CoreMuldivCommitmentFixture,
    evaluation: &jolt_verifier::JoltEvaluationProof,
) {
    let proof = core_proof_with_bolt_evaluation(&fixture.proof, evaluation);
    assert_core_verifies_proof(fixture, proof, "core accepts Bolt evaluation proof");
}

pub(crate) fn assert_core_accepts_full_bolt_proof(
    fixture: &CoreMuldivCommitmentFixture,
    proof: &jolt_verifier::JoltProof,
    artifacts: &jolt_prover::JoltProverArtifacts,
) {
    let core_proof = core_proof_with_full_bolt(&fixture.proof, proof, artifacts);
    assert_core_verifies_proof(fixture, core_proof, "core accepts full Bolt proof");
}
