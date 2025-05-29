//! This module provides the main API for the Jolt ONNX zkVM.
#![allow(clippy::field_reassign_with_default)] // TODO: Remove this when all zkVM portions are fully fleshed out

use crate::field::JoltField;
use crate::jolt::instruction::JoltInstructionSet;
use crate::jolt::subtable::JoltSubtableSet;
use crate::jolt::vm::bytecode::{BytecodeRow, BytecodeStuff};
use crate::jolt::vm::read_write_memory::ReadWriteMemoryStuff;
use crate::jolt::vm::timestamp_range_check::TimestampRangeCheckStuff;
use crate::jolt::vm::ProverDebugInfo;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::r1cs::inputs::R1CSStuff;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::MEMORY_OPS_PER_INSTRUCTION;
use common::rv_trace::{MemoryOp, NUM_CIRCUIT_FLAGS};
use instruction_lookups::{
    InstructionLookupStuff, InstructionLookupsPreprocessing, InstructionLookupsProof,
};
use serde::{Deserialize, Serialize};

use super::common::onnx_trace::JoltONNXDevice;
use super::memory_checking::{Initializable, StructuredPolynomialData};
use super::precompiles::PrecompileOperators;

pub mod instruction_lookups;
pub mod onnx_vm;
pub mod precompiles;

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltStuff<T: CanonicalSerialize + CanonicalDeserialize + Sync> {
    pub(crate) bytecode: BytecodeStuff<T>,
    pub(crate) read_write_memory: ReadWriteMemoryStuff<T>,
    pub(crate) instruction_lookups: InstructionLookupStuff<T>,
    pub(crate) timestamp_range_check: TimestampRangeCheckStuff<T>,
    pub(crate) r1cs: R1CSStuff<T>,
}

impl<T: CanonicalSerialize + CanonicalDeserialize + Sync> StructuredPolynomialData<T>
    for JoltStuff<T>
{
    fn read_write_values(&self) -> Vec<&T> {
        // self.bytecode
        //     .read_write_values()
        //     .into_iter()
        //     .chain(self.read_write_memory.read_write_values())
        //     .chain(self.instruction_lookups.read_write_values())
        //     .chain(self.timestamp_range_check.read_write_values())
        //     .chain(self.r1cs.read_write_values())
        //     .collect()
        self.instruction_lookups.read_write_values()
    }

    fn init_final_values(&self) -> Vec<&T> {
        // self.bytecode
        //     .init_final_values()
        //     .into_iter()
        //     .chain(self.read_write_memory.init_final_values())
        //     .chain(self.instruction_lookups.init_final_values())
        //     .chain(self.timestamp_range_check.init_final_values())
        //     .chain(self.r1cs.init_final_values())
        //     .collect()
        self.instruction_lookups.init_final_values()
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        // self.bytecode
        //     .read_write_values_mut()
        //     .into_iter()
        //     .chain(self.read_write_memory.read_write_values_mut())
        //     .chain(self.instruction_lookups.read_write_values_mut())
        //     .chain(self.timestamp_range_check.read_write_values_mut())
        //     .chain(self.r1cs.read_write_values_mut())
        //     .collect()
        self.instruction_lookups.read_write_values_mut()
    }

    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        // self.bytecode
        //     .init_final_values_mut()
        //     .into_iter()
        //     .chain(self.read_write_memory.init_final_values_mut())
        //     .chain(self.instruction_lookups.init_final_values_mut())
        //     .chain(self.timestamp_range_check.init_final_values_mut())
        //     .chain(self.r1cs.init_final_values_mut())
        //     .collect()
        self.instruction_lookups.init_final_values_mut()
    }
}

/// Note –– F: JoltField bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type JoltPolynomials<F: JoltField> = JoltStuff<MultilinearPolynomial<F>>;
/// Note –– PCS: CommitmentScheme bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type JoltCommitments<PCS: CommitmentScheme<ProofTranscript>, ProofTranscript: Transcript> =
    JoltStuff<PCS::Commitment>;

/// A SNARK for correct execution of an ONNX model on a given input.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProof<
    const C: usize,
    const M: usize,
    F,
    PCS,
    InstructionSet,
    Subtables,
    ProofTranscript,
> where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    InstructionSet: JoltInstructionSet,
    Subtables: JoltSubtableSet<F>,
    ProofTranscript: Transcript,
{
    /// The length of the trace, which is the number of steps in the execution trace.
    pub trace_length: usize,
    /// Instruction lookups proof.
    pub instruction_lookups:
        InstructionLookupsProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript>,
}

impl<const C: usize, const M: usize, F, PCS, InstructionSet, Subtables, ProofTranscript>
    JoltProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    InstructionSet: JoltInstructionSet,
    Subtables: JoltSubtableSet<F>,
    ProofTranscript: Transcript,
{
    /// Preprocessing step for the verifier.
    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    pub fn verifier_preprocess(
        max_trace_length: usize,
    ) -> JoltVerifierPreprocessing<C, F, PCS, ProofTranscript> {
        let instruction_lookups_preprocessing =
            InstructionLookupsPreprocessing::preprocess::<M, InstructionSet, Subtables>();
        let max_poly_len: usize = [max_trace_length.next_power_of_two(), M]
            .into_iter()
            .max()
            .unwrap();
        let generators = PCS::setup(max_poly_len);
        JoltVerifierPreprocessing {
            generators,
            instruction_lookups: instruction_lookups_preprocessing,
        }
    }

    /// Preprocessing step for the prover.
    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    pub fn prover_preprocess(
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<C, F, PCS, ProofTranscript> {
        let small_value_lookup_tables = F::compute_lookup_tables();
        F::initialize_lookup_tables(small_value_lookup_tables.clone());
        let shared = Self::verifier_preprocess(max_trace_length);
        JoltProverPreprocessing {
            shared,
            field: small_value_lookup_tables,
        }
    }

    /// Prove the execution trace of an ONNX model.
    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    pub fn prove(
        program_io: JoltONNXDevice,
        trace: Vec<JoltONNXTraceStep<InstructionSet>>,
        preprocessing: JoltProverPreprocessing<C, F, PCS, ProofTranscript>,
    ) -> (
        JoltProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript>,
        JoltCommitments<PCS, ProofTranscript>,
        JoltONNXDevice,
        Option<ProverDebugInfo<F, ProofTranscript>>,
    ) {
        let trace_length = trace.len();
        println!("Trace length: {trace_length}");
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let instruction_polynomials = InstructionLookupsProof::<
            C,
            M,
            F,
            PCS,
            InstructionSet,
            Subtables,
            ProofTranscript,
        >::generate_witness(
            &preprocessing.shared.instruction_lookups, &trace
        );
        let mut opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript> =
            ProverOpeningAccumulator::new();

        // HACK: We still need to implement the other polynomials
        let mut jolt_polynomials = JoltPolynomials::default();

        jolt_polynomials.instruction_lookups = instruction_polynomials;
        let jolt_commitments = commit_jolt_polys::<C, F, PCS, ProofTranscript>(
            &jolt_polynomials,
            &preprocessing.shared,
        );
        jolt_commitments
            .read_write_values()
            .iter()
            .for_each(|value| value.append_to_transcript(&mut transcript));
        jolt_commitments
            .init_final_values()
            .iter()
            .for_each(|value| value.append_to_transcript(&mut transcript));

        // TODO: Bytecode proof

        let instruction_proof = InstructionLookupsProof::prove(
            &preprocessing.shared.generators,
            &mut jolt_polynomials,
            &preprocessing.shared.instruction_lookups,
            &mut opening_accumulator,
            &mut transcript,
        );

        // TODO: Memory proof

        // TODO: Spartan proof

        let jolt_proof = JoltProof {
            trace_length,
            instruction_lookups: instruction_proof,
        };

        // TODO: Batch prove openings

        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript,
            opening_accumulator,
        });
        #[cfg(not(test))]
        let debug_info = None;
        (jolt_proof, jolt_commitments, program_io, debug_info)
    }

    /// Verify the [`JoltProof`]
    #[tracing::instrument(skip_all)]
    pub fn verify(
        self,
        preprocessing: JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        commitments: JoltCommitments<PCS, ProofTranscript>,
        _program_io: JoltONNXDevice,
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript>>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: VerifierOpeningAccumulator<F, PCS, ProofTranscript> =
            VerifierOpeningAccumulator::new();
        #[cfg(test)]
        if let Some(debug_info) = _debug_info {
            transcript.compare_to(debug_info.transcript);
            opening_accumulator
                .compare_to(debug_info.opening_accumulator, &preprocessing.generators);
        }
        commitments
            .read_write_values()
            .iter()
            .for_each(|value| value.append_to_transcript(&mut transcript));
        commitments
            .init_final_values()
            .iter()
            .for_each(|value| value.append_to_transcript(&mut transcript));
        Self::verify_instruction_lookups(
            &preprocessing.instruction_lookups,
            &preprocessing.generators,
            self.instruction_lookups,
            &commitments,
            &mut opening_accumulator,
            &mut transcript,
        )?;
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    fn verify_instruction_lookups<'a>(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        generators: &PCS::Setup,
        proof: InstructionLookupsProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript>,
        commitments: &'a JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        InstructionLookupsProof::verify(
            preprocessing,
            generators,
            proof,
            commitments,
            opening_accumulator,
            transcript,
        )
    }
}

/// Preprocessing for the verifier and prover
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<const C: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    /// Shared generators for the commitment scheme
    pub generators: PCS::Setup,
    /// Preprocessing for instruction lookups
    pub instruction_lookups: InstructionLookupsPreprocessing<C, F>,
}

/// Preprocessing for the prover
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProverPreprocessing<const C: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    /// Shared preprocessing data
    /// that can be used by both the prover and verifier.
    pub shared: JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
    field: F::SmallValueLookupTables,
}

/// Commmits to the Jolt polynomials.
// TODO: Remove this when we have a proper implementation
pub fn commit_jolt_polys<const C: usize, F, PCS, ProofTranscript>(
    jolt_polynomials: &JoltPolynomials<F>,
    preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
) -> JoltCommitments<PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let span = tracing::span!(tracing::Level::INFO, "commit::initialize");
    let _guard = span.enter();
    let mut commitments = JoltCommitments::<PCS, ProofTranscript>::initialize_onnx(preprocessing);
    drop(_guard);
    drop(span);

    // let trace_polys = jolt_polynomials.read_write_values();
    // let trace_commitments = PCS::batch_commit(&trace_polys, &preprocessing.generators);

    // commitments
    //     .read_write_values_mut()
    //     .into_iter()
    //     .zip(trace_commitments)
    //     .for_each(|(dest, src)| *dest = src);

    // let span = tracing::span!(tracing::Level::INFO, "commit::t_final");
    // let _guard = span.enter();
    // commitments.bytecode.t_final = PCS::commit(
    //     &jolt_polynomials.bytecode.t_final,
    //     &preprocessing.generators,
    // );
    // drop(_guard);
    // drop(span);

    let span = tracing::span!(tracing::Level::INFO, "commit::read_write_memory");
    let _guard = span.enter();
    // (
    //     commitments.read_write_memory.v_final,
    //     commitments.read_write_memory.t_final,
    // ) = join_conditional!(
    //     || PCS::commit(
    //         &jolt_polynomials.read_write_memory.v_final,
    //         &preprocessing.generators
    //     ),
    //     || PCS::commit(
    //         &jolt_polynomials.read_write_memory.t_final,
    //         &preprocessing.generators
    //     )
    // );
    commitments.instruction_lookups.final_cts = PCS::batch_commit(
        &jolt_polynomials.instruction_lookups.final_cts,
        &preprocessing.generators,
    );
    drop(_guard);
    drop(span);

    commitments
}

impl<T: CanonicalSerialize + CanonicalDeserialize + Default + Sync> JoltStuff<T> {
    fn initialize_onnx<const C: usize, PCS, ProofTranscript>(
        preprocessing: &JoltVerifierPreprocessing<C, PCS::Field, PCS, ProofTranscript>,
    ) -> Self
    where
        PCS: CommitmentScheme<ProofTranscript>,
        ProofTranscript: Transcript,
    {
        Self {
            bytecode: BytecodeStuff::default(),
            read_write_memory: ReadWriteMemoryStuff::default(),
            instruction_lookups: InstructionLookupStuff::initialize(
                &preprocessing.instruction_lookups,
            ),
            timestamp_range_check: TimestampRangeCheckStuff::default(),
            r1cs: R1CSStuff::default(),
        }
    }
}

/// Execution trace step for the Jolt ONNX VM.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct JoltONNXTraceStep<InstructionSet: JoltInstructionSet> {
    pub instruction_lookup: Option<InstructionSet>,
    pub bytecode_row: BytecodeRow,
    pub memory_ops: [MemoryOp; MEMORY_OPS_PER_INSTRUCTION],
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
    pub precompile: Option<PrecompileOperators>,
}

impl<InstructionSet: JoltInstructionSet> JoltONNXTraceStep<InstructionSet> {
    /// Create a new [`JoltONNXTraceStep`] with default values.
    pub fn no_op() -> Self {
        JoltONNXTraceStep {
            instruction_lookup: None,
            bytecode_row: BytecodeRow::no_op(0),
            memory_ops: [
                MemoryOp::noop_read(),  // rs1
                MemoryOp::noop_read(),  // rs2
                MemoryOp::noop_write(), // rd is write-only
                MemoryOp::noop_read(),  // RAM
            ],
            circuit_flags: [false; NUM_CIRCUIT_FLAGS],
            precompile: None,
        }
    }

    /// Pad the trace to the next power of two length.
    fn pad(trace: &mut Vec<Self>) {
        let unpadded_length = trace.len();
        let padded_length = unpadded_length.next_power_of_two();
        trace.resize(padded_length, Self::no_op());
    }
}
