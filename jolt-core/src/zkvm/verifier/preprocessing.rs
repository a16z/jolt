use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use tracer::instruction::Instruction;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::ram::RAMPreprocessing;
use crate::zkvm::Serializable;

#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
#[cfg(feature = "zk")]
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
#[cfg(feature = "zk")]
use crate::transcripts::Transcript;

#[cfg(feature = "prover")]
use crate::zkvm::prover::JoltProverPreprocessing;

pub(crate) struct StageVerifyResult<F: JoltField> {
    #[allow(dead_code)]
    pub(crate) challenges: Vec<F::Challenge>,
    #[cfg(feature = "zk")]
    pub(crate) batched_output_constraint: Option<OutputClaimConstraint>,
    #[cfg(feature = "zk")]
    pub(crate) output_constraint_challenge_values: Vec<F>,
    #[cfg(feature = "zk")]
    pub(crate) batched_input_constraint: InputClaimConstraint,
    #[cfg(feature = "zk")]
    pub(crate) input_constraint_challenge_values: Vec<F>,
    #[cfg(feature = "zk")]
    pub(crate) uniskip_input_constraint: Option<InputClaimConstraint>,
    #[cfg(feature = "zk")]
    pub(crate) uniskip_input_constraint_challenge_values: Vec<F>,
}

#[cfg(feature = "zk")]
impl<F: JoltField> StageVerifyResult<F> {
    pub(super) fn new(
        challenges: Vec<F::Challenge>,
        batched_output_constraint: Option<OutputClaimConstraint>,
        output_constraint_challenge_values: Vec<F>,
        batched_input_constraint: InputClaimConstraint,
        input_constraint_challenge_values: Vec<F>,
    ) -> Self {
        Self {
            challenges,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
            uniskip_input_constraint: None,
            uniskip_input_constraint_challenge_values: Vec::new(),
        }
    }

    pub(super) fn with_uniskip(
        challenges: Vec<F::Challenge>,
        batched_output_constraint: Option<OutputClaimConstraint>,
        output_constraint_challenge_values: Vec<F>,
        batched_input_constraint: InputClaimConstraint,
        input_constraint_challenge_values: Vec<F>,
        uniskip_input_constraint: InputClaimConstraint,
        uniskip_input_constraint_challenge_values: Vec<F>,
    ) -> Self {
        Self {
            challenges,
            batched_output_constraint,
            output_constraint_challenge_values,
            batched_input_constraint,
            input_constraint_challenge_values,
            uniskip_input_constraint: Some(uniskip_input_constraint),
            uniskip_input_constraint_challenge_values,
        }
    }
}

#[cfg(feature = "zk")]
pub(super) fn batch_output_constraints<F: JoltField, T: Transcript>(
    instances: &[&dyn SumcheckInstanceVerifier<F, T>],
) -> Option<OutputClaimConstraint> {
    let constraints: Vec<Option<OutputClaimConstraint>> = instances
        .iter()
        .map(|instance| instance.get_params().output_claim_constraint())
        .collect();
    OutputClaimConstraint::batch(&constraints, instances.len())
}

#[cfg(feature = "zk")]
pub(super) fn batch_input_constraints<F: JoltField, T: Transcript>(
    instances: &[&dyn SumcheckInstanceVerifier<F, T>],
) -> InputClaimConstraint {
    let constraints: Vec<InputClaimConstraint> = instances
        .iter()
        .map(|instance| instance.get_params().input_claim_constraint())
        .collect();
    InputClaimConstraint::batch_required(&constraints, instances.len())
}

#[cfg(feature = "zk")]
pub(super) fn scale_batching_coefficients<F: JoltField, T: Transcript>(
    batching_coefficients: &[F],
    instances: &[&dyn SumcheckInstanceVerifier<F, T>],
) -> Vec<F> {
    let max_num_rounds = instances.iter().map(|i| i.num_rounds()).max().unwrap_or(0);
    batching_coefficients
        .iter()
        .zip(instances.iter())
        .map(|(coeff, instance)| {
            let scale = max_num_rounds - instance.num_rounds();
            coeff.mul_pow_2(scale)
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct JoltSharedPreprocessing {
    pub bytecode: Arc<BytecodePreprocessing>,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
}

impl CanonicalSerialize for JoltSharedPreprocessing {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.bytecode
            .as_ref()
            .serialize_with_mode(&mut writer, compress)?;
        self.ram.serialize_with_mode(&mut writer, compress)?;
        self.memory_layout
            .serialize_with_mode(&mut writer, compress)?;
        self.max_padded_trace_length
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.bytecode.serialized_size(compress)
            + self.ram.serialized_size(compress)
            + self.memory_layout.serialized_size(compress)
            + self.max_padded_trace_length.serialized_size(compress)
    }
}

impl CanonicalDeserialize for JoltSharedPreprocessing {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let bytecode =
            BytecodePreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let ram = RAMPreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let memory_layout = MemoryLayout::deserialize_with_mode(&mut reader, compress, validate)?;
        let max_padded_trace_length =
            usize::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self {
            bytecode: Arc::new(bytecode),
            ram,
            memory_layout,
            max_padded_trace_length,
        })
    }
}

impl ark_serialize::Valid for JoltSharedPreprocessing {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.bytecode.check()?;
        self.ram.check()?;
        self.memory_layout.check()
    }
}

impl JoltSharedPreprocessing {
    #[tracing::instrument(skip_all, name = "JoltSharedPreprocessing::new")]
    pub fn new(
        bytecode: Vec<Instruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_padded_trace_length: usize,
    ) -> JoltSharedPreprocessing {
        let bytecode = Arc::new(BytecodePreprocessing::preprocess(bytecode));
        let ram = RAMPreprocessing::preprocess(memory_init);
        Self {
            bytecode,
            ram,
            memory_layout,
            max_padded_trace_length,
        }
    }
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing,
}

impl<F, PCS> Serializable for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
}

impl<F, PCS> JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> JoltVerifierPreprocessing<F, PCS> {
    #[tracing::instrument(skip_all, name = "JoltVerifierPreprocessing::new")]
    pub fn new(
        shared: JoltSharedPreprocessing,
        generators: PCS::VerifierSetup,
    ) -> JoltVerifierPreprocessing<F, PCS> {
        Self {
            generators,
            shared: shared.clone(),
        }
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, PCS: CommitmentScheme<Field = F>> From<&JoltProverPreprocessing<F, PCS>>
    for JoltVerifierPreprocessing<F, PCS>
{
    fn from(prover_preprocessing: &JoltProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&prover_preprocessing.generators);
        Self {
            generators,
            shared: prover_preprocessing.shared.clone(),
        }
    }
}
