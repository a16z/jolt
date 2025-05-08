//! This module provides the main API for the Jolt ONNX zkVM.

use super::trace::onnx::JoltONNXDevice;
use crate::field::JoltField;
use crate::jolt::instruction::JoltInstructionSet;
use crate::jolt::vm::bytecode::BytecodeStuff;
use crate::jolt::vm::instruction_lookups::{
    InstructionLookupStuff, InstructionLookupsPreprocessing, InstructionLookupsProof,
};
use crate::jolt::vm::read_write_memory::ReadWriteMemoryStuff;
use crate::jolt::vm::timestamp_range_check::TimestampRangeCheckStuff;
use crate::jolt::vm::{
    JoltCommitments, JoltPolynomials, JoltStuff, JoltTraceStep, ProverDebugInfo,
};
use crate::jolt::{
    instruction::{
        div::DIVInstruction, divu::DIVUInstruction, mulh::MULHInstruction,
        mulhsu::MULHSUInstruction, rem::REMInstruction, remu::REMUInstruction,
        VirtualInstructionSequence,
    },
    subtable::JoltSubtableSet,
    vm::timestamp_range_check::TimestampValidityProof,
};
use crate::lasso::memory_checking::{
    Initializable, MemoryCheckingProver, MemoryCheckingVerifier, StructuredPolynomialData,
};
use crate::msm::icicle;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::{
    ProverOpeningAccumulator, ReducedOpeningProof, VerifierOpeningAccumulator,
};
use crate::r1cs::constraints::R1CSConstraints;
use crate::r1cs::inputs::{ConstraintInput, R1CSPolynomials, R1CSProof, R1CSStuff};
use crate::r1cs::spartan::{self, UniformSpartanProof};
use crate::utils::errors::ProofVerifyError;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use crate::{join_conditional, jolt};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::rv_trace::{MemoryLayout, NUM_CIRCUIT_FLAGS};
use common::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    rv_trace::{ELFInstruction, JoltDevice, MemoryOp},
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};
use strum::EnumCount;

pub mod onnx_vm;

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
    pub trace_length: usize,
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
    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn verifier_preprocess(
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

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn prover_preprocess(
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

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    fn prove(
        program_io: JoltONNXDevice,
        mut trace: Vec<JoltTraceStep<InstructionSet>>,
        mut preprocessing: JoltProverPreprocessing<C, F, PCS, ProofTranscript>,
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

        // TODO: Send commitment to jolt polynomials
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

    #[tracing::instrument(skip_all)]
    fn verify(
        self,
        mut preprocessing: JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
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

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<const C: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub generators: PCS::Setup,
    pub instruction_lookups: InstructionLookupsPreprocessing<C, F>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProverPreprocessing<const C: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub shared: JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
    field: F::SmallValueLookupTables,
}

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
            r1cs: R1CSStuff::initialize(&C),
        }
    }
}
