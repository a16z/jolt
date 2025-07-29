//! A state-of-the-art zkVM, called Jolt, which turns almost everything a VM does into reads and writes to memory.
//! This includes the “fetch-decode-execute” logic of the VM.

use crate::jolt::vm::bytecode::{BytecodePreprocessing, BytecodeProof};
use jolt_core::{
    field::JoltField,
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use onnx_tracer::trace_types::{ONNXCycle, ONNXInstr};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

pub mod bytecode;
pub mod onnx_vm;
pub mod r1cs;

#[derive(Clone, Serialize, Deserialize)]
pub struct JoltProverPreprocessing<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    pub shared: JoltSharedPreprocessing,
    field: F::SmallValueLookupTables,
    _transcript: PhantomData<ProofTranscript>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltSharedPreprocessing {
    pub bytecode: BytecodePreprocessing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltVerifierPreprocessing<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    pub shared: JoltSharedPreprocessing,
    _p: PhantomData<(F, ProofTranscript)>,
}

impl<F, ProofTranscript> From<&JoltProverPreprocessing<F, ProofTranscript>>
    for JoltVerifierPreprocessing<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    fn from(preprocessing: &JoltProverPreprocessing<F, ProofTranscript>) -> Self {
        JoltVerifierPreprocessing {
            shared: preprocessing.shared.clone(),
            _p: PhantomData,
        }
    }
}

pub struct JoltSNARK<F, ProofTranscript>
where
    ProofTranscript: Transcript,
    F: JoltField,
{
    pub trace_length: usize,
    bytecode: BytecodeProof<F, ProofTranscript>,
}

impl<F, ProofTranscript> JoltSNARK<F, ProofTranscript>
where
    ProofTranscript: Transcript,
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
    fn prover_preprocess(bytecode: Vec<ONNXInstr>) -> JoltProverPreprocessing<F, ProofTranscript> {
        let small_value_lookup_tables = F::compute_lookup_tables();
        F::initialize_lookup_tables(small_value_lookup_tables.clone());
        let shared = Self::shared_preprocess(bytecode);
        JoltProverPreprocessing {
            shared,
            field: small_value_lookup_tables,
            _transcript: PhantomData,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    pub fn prove(
        mut preprocessing: JoltProverPreprocessing<F, ProofTranscript>,
        mut trace: Vec<ONNXCycle>,
    ) -> Self {
        let trace_length = trace.len();
        println!("Trace length: {trace_length}");
        F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));
        // pad trace to the next power of two
        trace.resize(trace.len().next_power_of_two(), ONNXCycle::no_op());
        let padded_trace_length = trace.len();
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let bytecode_proof =
            BytecodeProof::prove(&preprocessing.shared.bytecode, &trace, &mut transcript);
        JoltSNARK {
            trace_length: padded_trace_length,
            bytecode: bytecode_proof,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn verify(
        &self,
        preprocessing: JoltVerifierPreprocessing<F, ProofTranscript>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        self.bytecode.verify(
            &preprocessing.shared.bytecode,
            self.trace_length,
            &mut transcript,
        )?;
        Ok(())
    }
}
