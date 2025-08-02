//! A state-of-the-art zkVM, called Jolt, which turns almost everything a VM does into reads and writes to memory.
//! This includes the “fetch-decode-execute” logic of the VM.

use crate::jolt::zkvm::{
    bytecode::{BytecodePreprocessing, BytecodeProof},
    instruction_lookups::LookupsProof,
    r1cs::{
        constraints::{JoltONNXConstraints, R1CSConstraints},
        spartan::UniformSpartanProof,
    },
    registers::RegistersTwistProof,
};
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, dory::DoryGlobals},
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use onnx_tracer::trace_types::{ONNXCycle, ONNXInstr};
use onnx_vm::WORD_SIZE;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

pub mod bytecode;
pub mod instruction_lookups;
pub mod onnx_vm;
pub mod r1cs;
pub mod registers;

#[derive(Clone, Serialize, Deserialize)]
pub struct JoltProverPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub shared: JoltSharedPreprocessing,
    field: F::SmallValueLookupTables,
    _p: PhantomData<(ProofTranscript, PCS)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltSharedPreprocessing {
    pub bytecode: BytecodePreprocessing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltVerifierPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub shared: JoltSharedPreprocessing,
    _p: PhantomData<(F, PCS, ProofTranscript)>,
}

impl<F, PCS, ProofTranscript> From<&JoltProverPreprocessing<F, PCS, ProofTranscript>>
    for JoltVerifierPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn from(preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>) -> Self {
        JoltVerifierPreprocessing {
            shared: preprocessing.shared.clone(),
            _p: PhantomData,
        }
    }
}

pub struct JoltSNARK<F, PCS, ProofTranscript>
where
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    F: JoltField,
{
    pub trace_length: usize,
    bytecode: BytecodeProof<F, ProofTranscript>,
    instruction_lookups: LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>,
    registers: RegistersTwistProof<F, ProofTranscript>,
    r1cs: UniformSpartanProof<F, ProofTranscript>,
    _p: PhantomData<PCS>,
}

impl<F, PCS, ProofTranscript> JoltSNARK<F, PCS, ProofTranscript>
where
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    F: JoltField,
{
    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn shared_preprocess(bytecode: Vec<ONNXInstr>) -> JoltSharedPreprocessing {
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(bytecode);
        JoltSharedPreprocessing {
            bytecode: bytecode_preprocessing,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn prover_preprocess(
        bytecode: Vec<ONNXInstr>,
    ) -> JoltProverPreprocessing<F, PCS, ProofTranscript> {
        let small_value_lookup_tables = F::compute_lookup_tables();
        F::initialize_lookup_tables(small_value_lookup_tables.clone());
        let shared = Self::shared_preprocess(bytecode);
        JoltProverPreprocessing {
            shared,
            field: small_value_lookup_tables,
            _p: PhantomData,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    pub fn prove(
        mut preprocessing: JoltProverPreprocessing<F, PCS, ProofTranscript>,
        mut trace: Vec<ONNXCycle>,
    ) -> Self {
        let trace_length = trace.len();
        println!("Trace length: {trace_length}");
        F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));
        // pad trace to the next power of two
        trace.resize(trace.len().next_power_of_two(), ONNXCycle::no_op());
        let padded_trace_length = trace.len();
        let ram_addresses: Vec<usize> = trace.iter().map(|cycle| cycle.td() + 1).collect();
        let ram_K = ram_addresses.iter().max().unwrap().next_power_of_two();
        let K = [
            preprocessing.shared.bytecode.code_size,
            ram_K,
            1 << 16, // K for instruction lookups Shout
        ]
        .into_iter()
        .max()
        .unwrap();
        println!("T = {padded_trace_length}, K = {K}");
        let _guard = DoryGlobals::initialize(K, padded_trace_length);
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: ProverOpeningAccumulator<F, PCS, ProofTranscript> =
            ProverOpeningAccumulator::new();
        let constraint_builder = JoltONNXConstraints::construct_constraints(padded_trace_length);
        let spartan_key = UniformSpartanProof::<F, ProofTranscript>::setup(
            &constraint_builder,
            padded_trace_length,
        );
        transcript.append_scalar(&spartan_key.vk_digest);
        let r1cs = UniformSpartanProof::prove::<PCS>(
            &preprocessing,
            &constraint_builder,
            &spartan_key,
            &trace,
            &mut transcript,
        )
        .ok()
        .unwrap();
        let instruction_proof: LookupsProof<WORD_SIZE, F, PCS, ProofTranscript> =
            LookupsProof::prove(
                &preprocessing,
                &trace,
                &mut opening_accumulator,
                &mut transcript,
            );
        let registers_proof = RegistersTwistProof::prove(
            &preprocessing,
            &trace,
            ram_K,
            &mut opening_accumulator,
            &mut transcript,
        );
        let bytecode_proof =
            BytecodeProof::prove(&preprocessing.shared.bytecode, &trace, &mut transcript);
        JoltSNARK {
            trace_length,
            r1cs,
            registers: registers_proof,
            instruction_lookups: instruction_proof,
            bytecode: bytecode_proof,
            _p: PhantomData,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn verify(
        &self,
        preprocessing: JoltVerifierPreprocessing<F, PCS, ProofTranscript>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: VerifierOpeningAccumulator<F, PCS, ProofTranscript> =
            VerifierOpeningAccumulator::new();
        // Regenerate the uniform Spartan key
        let padded_trace_length = self.trace_length.next_power_of_two();
        let r1cs_builder = JoltONNXConstraints::construct_constraints(padded_trace_length);
        let spartan_key =
            UniformSpartanProof::<F, ProofTranscript>::setup(&r1cs_builder, padded_trace_length);
        transcript.append_scalar(&spartan_key.vk_digest);
        self.r1cs
            .verify::<PCS>(&spartan_key, &mut transcript)
            .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))?;
        self.instruction_lookups
            .verify(&mut opening_accumulator, &mut transcript)?;
        self.registers.verify(
            padded_trace_length,
            &mut opening_accumulator,
            &mut transcript,
        )?;
        self.bytecode.verify(
            &preprocessing.shared.bytecode,
            self.trace_length,
            &mut transcript,
        )?;
        Ok(())
    }
}
