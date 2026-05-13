//! Commitment-phase oracle runner for Bolt/Jolt equivalence checks.
//!
//! This module owns the thin commitment bridge used by tests: run the
//! generated real-trace commitment phase, keep a small synthetic transcript
//! bridge for CPU-plan ordering tests, and expose transcript traces in the
//! canonical equivalence shape.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "commitment oracle gates should fail fast on malformed generated artifacts"
)]

use std::borrow::Cow;

use ark_serialize::CanonicalSerialize;
use bolt::protocols::jolt::CommitmentCpuProgram;
use common::jolt_device::JoltDevice;
use jolt_dory::{DoryCommitment, DoryProverSetup, DoryScheme};
use jolt_field::Fr;
use jolt_prover::stages::commitment as generated_commitment;
use jolt_transcript::{Label, LabelWithCount, Transcript, U64Word};
use jolt_verifier::stages::commitment as generated_verifier_commitment;
use jolt_witness::{
    commitment_trace_sources, optional_oracle_data, CommitmentTraceSources, CycleInput,
};

use crate::artifacts::{
    ArtifactSource, CommitmentArtifact, CommitmentTrace, EquivalenceRun, TranscriptTrace,
};
use crate::checkpoint::{CheckpointTranscript, TranscriptEvent};
use crate::core_conversion::{commitment_to_ark, CoreCommitment};
use crate::plan_adapters::{
    leak_generated_commitment_prover_program, leak_generated_commitment_verifier_program,
};

const TRANSCRIPT_LABEL: &[u8] = b"Jolt";

pub type BoltTranscript = CheckpointTranscript<jolt_transcript::Blake2bTranscript<Fr>>;

#[derive(Clone, Debug)]
pub struct CommitmentRecord {
    pub artifact: String,
}

#[derive(Clone, Debug)]
pub struct BoltCommitmentTrace {
    pub commitments: Vec<Option<DoryCommitment>>,
    pub records: Vec<CommitmentRecord>,
    pub log: Vec<TranscriptEvent>,
}

impl BoltCommitmentTrace {
    pub fn commitment_trace(&self) -> CommitmentTrace {
        CommitmentTrace {
            commitments: self
                .records
                .iter()
                .zip(&self.commitments)
                .map(|(record, commitment)| CommitmentArtifact {
                    label: record.artifact.clone(),
                    artifact: record.artifact.clone(),
                    bytes: commitment.as_ref().map(bolt_commitment_bytes),
                })
                .collect(),
        }
    }

    pub fn committed_prefix(&self, len: usize) -> CommitmentTrace {
        CommitmentTrace {
            commitments: self
                .records
                .iter()
                .zip(&self.commitments)
                .filter_map(|(record, commitment)| {
                    commitment.as_ref().map(|commitment| CommitmentArtifact {
                        label: record.artifact.clone(),
                        artifact: record.artifact.clone(),
                        bytes: Some(bolt_commitment_bytes(commitment)),
                    })
                })
                .take(len)
                .collect(),
        }
    }

    pub fn equivalence_run(&self, source: ArtifactSource) -> EquivalenceRun<Fr> {
        let mut run = EquivalenceRun::new(source);
        run.commitments = self.commitment_trace();
        run.transcript = TranscriptTrace {
            events: self.log.clone(),
        };
        run
    }
}

pub fn core_commitment_trace(commitments: &[CoreCommitment], artifact: &str) -> CommitmentTrace {
    CommitmentTrace {
        commitments: commitments
            .iter()
            .map(|commitment| CommitmentArtifact {
                label: artifact.to_owned(),
                artifact: artifact.to_owned(),
                bytes: Some(core_commitment_bytes(commitment)),
            })
            .collect(),
    }
}

macro_rules! generated_commitment_trace_fns {
    ($trace_fn:ident, $artifacts_to_trace_fn:ident, $artifacts:ty) => {
        pub(crate) fn $trace_fn(artifacts: &$artifacts) -> CommitmentTrace {
            CommitmentTrace {
                commitments: artifacts
                    .records
                    .iter()
                    .zip(&artifacts.commitments)
                    .map(|(record, commitment)| CommitmentArtifact {
                        label: record.artifact.to_owned(),
                        artifact: record.artifact.to_owned(),
                        bytes: commitment.as_ref().map(bolt_commitment_bytes),
                    })
                    .collect(),
            }
        }

        fn $artifacts_to_trace_fn(
            artifacts: &$artifacts,
            log: Vec<TranscriptEvent>,
        ) -> BoltCommitmentTrace {
            BoltCommitmentTrace {
                commitments: artifacts.commitments.clone(),
                records: artifacts
                    .records
                    .iter()
                    .map(|record| CommitmentRecord {
                        artifact: record.artifact.to_owned(),
                    })
                    .collect(),
                log,
            }
        }
    };
}

generated_commitment_trace_fns!(
    generated_commitment_trace,
    generated_prover_commitment_artifacts_to_trace,
    generated_commitment::CommitmentArtifacts
);
generated_commitment_trace_fns!(
    generated_verifier_commitment_trace,
    generated_verifier_commitment_artifacts_to_trace,
    generated_verifier_commitment::CommitmentArtifacts
);

fn bolt_commitment_bytes(commitment: &DoryCommitment) -> Vec<u8> {
    core_commitment_bytes(&commitment_to_ark(commitment))
}

fn core_commitment_bytes(commitment: &CoreCommitment) -> Vec<u8> {
    let mut bytes = Vec::new();
    commitment
        .serialize_uncompressed(&mut bytes)
        .expect("commitment serialization");
    bytes
}

pub trait BoltPreambleSource {
    fn program_io(&self) -> &JoltDevice;
    fn preprocessing_digest(&self) -> [u8; 32];
    fn ram_k(&self) -> u64;
    fn trace_length(&self) -> u64;
    fn entry_address(&self) -> u64;
    fn ram_rw_phase1_num_rounds(&self) -> u64;
    fn ram_rw_phase2_num_rounds(&self) -> u64;
    fn registers_rw_phase1_num_rounds(&self) -> u64;
    fn registers_rw_phase2_num_rounds(&self) -> u64;
    fn log_k_chunk(&self) -> u64;
    fn lookups_ra_virtual_log_k_chunk(&self) -> u64;
    fn dory_layout(&self) -> u64;
}

pub fn append_bolt_preamble<T, P>(transcript: &mut T, preamble: &P)
where
    T: Transcript<Challenge = Fr>,
    P: BoltPreambleSource,
{
    let program_io = preamble.program_io();
    let preprocessing_digest = preamble.preprocessing_digest();
    append_bytes(transcript, b"preprocessing_digest", &preprocessing_digest);
    append_u64(
        transcript,
        b"max_input_size",
        program_io.memory_layout.max_input_size,
    );
    append_u64(
        transcript,
        b"max_output_size",
        program_io.memory_layout.max_output_size,
    );
    append_u64(transcript, b"heap_size", program_io.memory_layout.heap_size);
    append_bytes(transcript, b"inputs", &program_io.inputs);
    append_bytes(transcript, b"outputs", &program_io.outputs);
    append_u64(transcript, b"panic", program_io.panic as u64);
    append_u64(transcript, b"ram_K", preamble.ram_k());
    append_u64(transcript, b"trace_length", preamble.trace_length());
    append_u64(transcript, b"entry_address", preamble.entry_address());
    append_u64(
        transcript,
        b"ram_rw_phase1_num_rounds",
        preamble.ram_rw_phase1_num_rounds(),
    );
    append_u64(
        transcript,
        b"ram_rw_phase2_num_rounds",
        preamble.ram_rw_phase2_num_rounds(),
    );
    append_u64(
        transcript,
        b"registers_rw_phase1_num_rounds",
        preamble.registers_rw_phase1_num_rounds(),
    );
    append_u64(
        transcript,
        b"registers_rw_phase2_num_rounds",
        preamble.registers_rw_phase2_num_rounds(),
    );
    append_u64(transcript, b"log_k_chunk", preamble.log_k_chunk());
    append_u64(
        transcript,
        b"lookups_ra_virtual_log_k_chunk",
        preamble.lookups_ra_virtual_log_k_chunk(),
    );
    append_u64(transcript, b"dory_layout", preamble.dory_layout());
}

pub(crate) fn transcript_with_bolt_preamble<P>(preamble: &P) -> BoltTranscript
where
    P: BoltPreambleSource,
{
    let mut transcript = BoltTranscript::new(TRANSCRIPT_LABEL);
    append_bolt_preamble(&mut transcript, preamble);
    transcript
}

pub fn transcript_with_bolt_commitment_trace<P>(
    preamble: &P,
    trace: &BoltCommitmentTrace,
) -> BoltTranscript
where
    P: BoltPreambleSource,
{
    let mut transcript = transcript_with_bolt_preamble(preamble);
    append_bolt_commitment_trace(&mut transcript, trace);
    transcript
}

fn append_u64<T>(transcript: &mut T, label: &'static [u8], value: u64)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&Label(label));
    transcript.append(&U64Word(value));
}

fn append_bytes<T>(transcript: &mut T, label: &'static [u8], bytes: &[u8])
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(label, bytes.len() as u64));
    transcript.append_bytes(bytes);
}

#[derive(Clone, Debug)]
pub(crate) struct GeneratedCommitmentInputStorage {
    sources: CommitmentTraceSources,
}

impl GeneratedCommitmentInputStorage {
    pub(crate) fn from_cycles(cycle_inputs: &[CycleInput]) -> Self {
        Self {
            sources: commitment_trace_sources(cycle_inputs),
        }
    }

    pub(crate) fn sparse_inputs(&self) -> generated_commitment::SparseCommitmentInputs<'_> {
        generated_commitment::SparseCommitmentInputs::new(
            generated_commitment::CommitmentOracleInputs::from_trace_sources(
                &self.sources,
                None,
                None,
            ),
        )
    }
}

pub fn run_generated_bolt_commitment_pair_with_cpu_programs(
    prover_program: &CommitmentCpuProgram,
    verifier_program: &CommitmentCpuProgram,
    setup: &DoryProverSetup,
    cycle_inputs: &[CycleInput],
) -> (BoltCommitmentTrace, BoltCommitmentTrace) {
    run_generated_bolt_commitment_pair_with_cycles(
        leak_generated_commitment_prover_program(prover_program),
        leak_generated_commitment_verifier_program(verifier_program),
        setup,
        cycle_inputs,
    )
}

fn run_generated_synthetic_bolt_commitment_pair(
    prover_program: &'static generated_commitment::CommitmentProverProgramPlan,
    verifier_program: &'static generated_verifier_commitment::CommitmentVerifierProgramPlan,
) -> (BoltCommitmentTrace, BoltCommitmentTrace) {
    let setup = DoryScheme::setup_prover(max_generated_num_vars(prover_program));
    let mut inputs = SyntheticCommitmentInputs;
    let prover_trace = run_generated_bolt_commitment_prover(prover_program, &setup, &mut inputs);
    let verifier_trace =
        run_generated_bolt_commitment_verifier(verifier_program, &prover_trace.commitments);
    (prover_trace, verifier_trace)
}

pub fn run_generated_synthetic_bolt_commitment_pair_with_cpu_programs(
    prover_program: &CommitmentCpuProgram,
    verifier_program: &CommitmentCpuProgram,
) -> (BoltCommitmentTrace, BoltCommitmentTrace) {
    run_generated_synthetic_bolt_commitment_pair(
        leak_generated_commitment_prover_program(prover_program),
        leak_generated_commitment_verifier_program(verifier_program),
    )
}

pub fn run_generated_bolt_commitment_pair_with_cycles(
    prover_program: &'static generated_commitment::CommitmentProverProgramPlan,
    verifier_program: &'static generated_verifier_commitment::CommitmentVerifierProgramPlan,
    setup: &DoryProverSetup,
    cycle_inputs: &[CycleInput],
) -> (BoltCommitmentTrace, BoltCommitmentTrace) {
    let storage = GeneratedCommitmentInputStorage::from_cycles(cycle_inputs);
    let mut inputs = storage.sparse_inputs();
    let prover_trace = run_generated_bolt_commitment_prover(prover_program, setup, &mut inputs);
    let verifier_trace =
        run_generated_bolt_commitment_verifier(verifier_program, &prover_trace.commitments);
    (prover_trace, verifier_trace)
}

fn run_generated_bolt_commitment_prover<I>(
    program: &'static generated_commitment::CommitmentProverProgramPlan,
    setup: &DoryProverSetup,
    inputs: &mut I,
) -> BoltCommitmentTrace
where
    I: generated_commitment::CommitmentInputProvider,
{
    let mut transcript = BoltTranscript::new(TRANSCRIPT_LABEL);
    let artifacts = generated_commitment::prove_commitment_phase_with_program(
        program,
        inputs,
        setup,
        &mut transcript,
    )
    .expect("generated Bolt commitment prover succeeds");
    generated_prover_commitment_artifacts_to_trace(&artifacts, transcript.log().to_vec())
}

fn run_generated_bolt_commitment_verifier(
    program: &'static generated_verifier_commitment::CommitmentVerifierProgramPlan,
    proof_commitments: &[Option<DoryCommitment>],
) -> BoltCommitmentTrace {
    let mut transcript = BoltTranscript::new(TRANSCRIPT_LABEL);
    let artifacts = generated_verifier_commitment::verify_commitment_phase_with_program(
        program,
        proof_commitments,
        &mut transcript,
    )
    .expect("generated Bolt commitment verifier succeeds");
    generated_verifier_commitment_artifacts_to_trace(&artifacts, transcript.log().to_vec())
}

fn append_bolt_commitment_trace<T>(transcript: &mut T, trace: &BoltCommitmentTrace)
where
    T: Transcript<Challenge = Fr>,
{
    for event in &trace.log {
        match event {
            TranscriptEvent::Append { bytes, .. } => transcript.append_bytes(bytes),
            TranscriptEvent::Squeeze { .. } => {
                panic!("commitment transcript trace unexpectedly contains a squeeze")
            }
        }
    }
}

struct SyntheticCommitmentInputs;

impl generated_commitment::CommitmentInputProvider for SyntheticCommitmentInputs {
    fn materialize(&mut self, _oracle: &'static str) -> Option<Cow<'_, [Fr]>> {
        None
    }

    fn materialize_with_num_vars(
        &mut self,
        oracle: &'static str,
        num_vars: usize,
    ) -> Option<Cow<'_, [Fr]>> {
        optional_oracle_data(oracle, num_vars).map(Cow::Owned)
    }
}

fn max_generated_num_vars(program: &generated_commitment::CommitmentProverProgramPlan) -> usize {
    program
        .batch_plans
        .iter()
        .map(|plan| plan.num_vars)
        .chain(program.optional_plans.iter().map(|plan| plan.num_vars))
        .max()
        .unwrap_or(0)
}
