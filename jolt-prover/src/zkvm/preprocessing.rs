use std::{
    fs::File,
    io::{Read, Write},
    marker::PhantomData,
    path::Path,
};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;

use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::commitment::{
        commitment_scheme::CommitmentScheme, dory::DoryGlobals, pedersen::PedersenGenerators,
    },
    utils::math::Math,
    zkvm::{
        bytecode::chunks::{
            committed_lanes, is_valid_committed_bytecode_chunking_for_len,
            DEFAULT_COMMITTED_BYTECODE_CHUNK_COUNT,
        },
        program::{CommittedProgramProverData, ProgramMetadata, ProgramPreprocessing},
        Serializable,
    },
};
#[cfg(feature = "prover")]
use crate::{
    poly::commitment::commitment_scheme::ZkEvalCommitment, zkvm::prover::JoltProverPreprocessing,
};

#[derive(Debug, Clone)]
pub struct JoltSharedPreprocessing<
    PCS: CommitmentScheme = crate::poly::commitment::dory::DoryCommitmentScheme,
> {
    pub program: ProgramPreprocessing<PCS>,
    pub program_meta: ProgramMetadata,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
    pub bytecode_chunk_count: usize,
}

impl<PCS: CommitmentScheme> JoltSharedPreprocessing<PCS>
where
    PCS::Commitment: CanonicalSerialize,
{
    /// Blake2b-256 digest of the serialized preprocessing, used to bind
    /// the program identity to the Fiat-Shamir transcript.
    pub fn digest(&self) -> [u8; 32] {
        use ark_serialize::CanonicalSerialize;
        use blake2::{digest::consts::U32, Blake2b, Digest};
        let mut buf = Vec::new();
        self.serialize_compressed(&mut buf)
            .expect("serialization cannot fail for in-memory buffer");
        Blake2b::<U32>::digest(&buf).into()
    }
}

impl<PCS: CommitmentScheme> CanonicalSerialize for JoltSharedPreprocessing<PCS>
where
    PCS::Commitment: CanonicalSerialize,
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.program.serialize_with_mode(&mut writer, compress)?;
        self.program_meta
            .serialize_with_mode(&mut writer, compress)?;
        self.memory_layout
            .serialize_with_mode(&mut writer, compress)?;
        self.max_padded_trace_length
            .serialize_with_mode(&mut writer, compress)?;
        self.bytecode_chunk_count
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.program.serialized_size(compress)
            + self.program_meta.serialized_size(compress)
            + self.memory_layout.serialized_size(compress)
            + self.max_padded_trace_length.serialized_size(compress)
            + self.bytecode_chunk_count.serialized_size(compress)
    }
}

impl<PCS: CommitmentScheme> CanonicalDeserialize for JoltSharedPreprocessing<PCS>
where
    PCS::Commitment: CanonicalDeserialize,
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let program = ProgramPreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let program_meta = ProgramMetadata::deserialize_with_mode(&mut reader, compress, validate)?;
        let memory_layout = MemoryLayout::deserialize_with_mode(&mut reader, compress, validate)?;
        let max_padded_trace_length =
            usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let bytecode_chunk_count = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let shared = Self {
            program,
            program_meta,
            memory_layout,
            max_padded_trace_length,
            bytecode_chunk_count,
        };
        if matches!(validate, ark_serialize::Validate::Yes) {
            ark_serialize::Valid::check(&shared)?;
        }
        Ok(shared)
    }
}

impl<PCS: CommitmentScheme> ark_serialize::Valid for JoltSharedPreprocessing<PCS>
where
    PCS::Commitment: ark_serialize::Valid,
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.program.check()?;
        self.program_meta.check()?;
        self.memory_layout.check()?;
        if self.program.is_committed()
            && !is_valid_committed_bytecode_chunking_for_len(
                self.program.bytecode_len(),
                self.bytecode_chunk_count,
            )
        {
            return Err(ark_serialize::SerializationError::InvalidData);
        }
        Ok(())
    }
}

impl<PCS: CommitmentScheme> JoltSharedPreprocessing<PCS> {
    #[tracing::instrument(skip_all, name = "JoltSharedPreprocessing::new")]
    pub fn new(
        program: ProgramPreprocessing<PCS>,
        memory_layout: MemoryLayout,
        max_padded_trace_length: usize,
    ) -> JoltSharedPreprocessing<PCS> {
        Self {
            program_meta: program.meta(),
            program,
            memory_layout,
            max_padded_trace_length,
            bytecode_chunk_count: DEFAULT_COMMITTED_BYTECODE_CHUNK_COUNT,
        }
    }

    #[tracing::instrument(skip_all, name = "JoltSharedPreprocessing::new_committed")]
    pub fn new_committed(
        program: ProgramPreprocessing<PCS>,
        memory_layout: MemoryLayout,
        max_padded_trace_length: usize,
        bytecode_chunk_count: usize,
    ) -> (
        JoltSharedPreprocessing<PCS>,
        CommittedProgramProverData<PCS>,
        PCS::ProverSetup,
    ) {
        let bytecode_len = program.bytecode_len();
        assert!(
            is_valid_committed_bytecode_chunking_for_len(bytecode_len, bytecode_chunk_count),
            "bytecode chunk count ({bytecode_chunk_count}) must be non-zero, a power of two, at \
             most {}, and divide bytecode size ({bytecode_len})",
            crate::zkvm::bytecode::chunks::MAX_COMMITTED_BYTECODE_CHUNK_COUNT,
        );
        let mut shared = Self {
            program_meta: program.meta(),
            program,
            memory_layout,
            max_padded_trace_length,
            bytecode_chunk_count,
        };
        let (max_total_vars, max_log_k_chunk) = shared.compute_max_total_vars(true);
        let generators = PCS::setup_prover(max_total_vars);
        let (committed_program, prover_data) = shared.program.commit(
            &shared.memory_layout,
            &generators,
            shared.bytecode_chunk_count,
            max_log_k_chunk,
        );
        shared.program = committed_program;
        shared.program_meta = shared.program.meta();
        (shared, prover_data, generators)
    }

    pub fn is_committed_mode(&self) -> bool {
        self.program.is_committed()
    }

    pub fn bytecode_size(&self) -> usize {
        self.program_meta.bytecode_len
    }

    #[inline]
    pub fn committed_program_image_num_words(&self) -> usize {
        self.program_meta
            .committed_program_image_num_words(&self.memory_layout)
    }

    #[inline]
    pub(crate) fn precommitted_candidate_total_vars(
        &self,
        include_committed: bool,
        include_trusted_advice: bool,
        include_untrusted_advice: bool,
    ) -> Vec<usize> {
        let mut candidates = Vec::with_capacity(
            include_committed as usize * 2
                + include_trusted_advice as usize
                + include_untrusted_advice as usize,
        );

        if include_trusted_advice {
            let (trusted_sigma, trusted_nu) = DoryGlobals::advice_sigma_nu_from_max_bytes(
                self.memory_layout.max_trusted_advice_size as usize,
            );
            candidates.push(trusted_sigma + trusted_nu);
        }

        if include_untrusted_advice {
            let (untrusted_sigma, untrusted_nu) = DoryGlobals::advice_sigma_nu_from_max_bytes(
                self.memory_layout.max_untrusted_advice_size as usize,
            );
            candidates.push(untrusted_sigma + untrusted_nu);
        }

        if include_committed {
            let chunk_cycle_log_t = (self.bytecode_size() / self.bytecode_chunk_count)
                .next_power_of_two()
                .log_2();
            candidates.push(committed_lanes().log_2() + chunk_cycle_log_t);
            candidates.push(self.committed_program_image_num_words().log_2());
        }

        candidates
    }

    #[inline]
    pub(crate) fn max_total_vars_from_candidates(
        main_total_vars: usize,
        candidates: impl IntoIterator<Item = usize>,
    ) -> usize {
        let mut max_total_vars = main_total_vars;
        for total_vars in candidates {
            max_total_vars = max_total_vars.max(total_vars);
        }
        max_total_vars
    }

    #[inline]
    pub(crate) fn compute_max_total_vars(&self, include_committed: bool) -> (usize, usize) {
        use common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T;
        let max_t_any = self.max_padded_trace_length.next_power_of_two();
        let max_log_t = max_t_any.log_2();
        let max_log_k_chunk = if max_log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };

        let max_total_vars = Self::max_total_vars_from_candidates(
            max_log_k_chunk + max_log_t,
            self.precommitted_candidate_total_vars(include_committed, true, true),
        );

        (max_total_vars, max_log_k_chunk)
    }
}

/// Serializable wrapper around [`PedersenGenerators`] for ZK setup transfer.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BlindfoldSetup<C: JoltCurve>(pub PedersenGenerators<C>);

impl<C: JoltCurve> std::ops::Deref for BlindfoldSetup<C> {
    type Target = PedersenGenerators<C>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<C: JoltCurve> From<BlindfoldSetup<C>> for PedersenGenerators<C> {
    fn from(setup: BlindfoldSetup<C>) -> Self {
        setup.0
    }
}

#[derive(Debug, Clone)]
pub struct JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
{
    _curve: PhantomData<C>,
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing<PCS>,
    pub blindfold_setup: Option<BlindfoldSetup<C>>,
}

impl<F, C, PCS> CanonicalSerialize for JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    PCS::VerifierSetup: CanonicalSerialize,
    PCS::Commitment: CanonicalSerialize,
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.generators.serialize_with_mode(&mut writer, compress)?;
        self.shared.serialize_with_mode(&mut writer, compress)?;
        self.blindfold_setup
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.generators.serialized_size(compress)
            + self.shared.serialized_size(compress)
            + self.blindfold_setup.serialized_size(compress)
    }
}

impl<F, C, PCS> CanonicalDeserialize for JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    PCS::VerifierSetup: CanonicalDeserialize,
    PCS::Commitment: CanonicalDeserialize,
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        Ok(Self {
            _curve: PhantomData,
            generators: PCS::VerifierSetup::deserialize_with_mode(&mut reader, compress, validate)?,
            shared: JoltSharedPreprocessing::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            blindfold_setup: Option::<BlindfoldSetup<C>>::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
        })
    }
}

impl<F, C, PCS> ark_serialize::Valid for JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    PCS::VerifierSetup: ark_serialize::Valid,
    PCS::Commitment: ark_serialize::Valid,
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.generators.check()?;
        self.shared.check()?;
        self.blindfold_setup.check()?;
        Ok(())
    }
}

impl<F, C, PCS> Serializable for JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    PCS::VerifierSetup: CanonicalSerialize + CanonicalDeserialize,
    PCS::Commitment: CanonicalSerialize + CanonicalDeserialize,
{
}

impl<F, C, PCS> JoltVerifierPreprocessing<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    PCS::VerifierSetup: CanonicalSerialize + CanonicalDeserialize,
    PCS::Commitment: CanonicalSerialize + CanonicalDeserialize,
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

impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F>>
    JoltVerifierPreprocessing<F, C, PCS>
{
    #[tracing::instrument(skip_all, name = "JoltVerifierPreprocessing::new")]
    pub fn new(
        mut shared: JoltSharedPreprocessing<PCS>,
        generators: PCS::VerifierSetup,
        blindfold_setup: Option<BlindfoldSetup<C>>,
    ) -> Self {
        shared.program = shared.program.to_verifier_program();
        Self {
            _curve: PhantomData,
            generators,
            shared,
            blindfold_setup,
        }
    }

    #[cfg(feature = "zk")]
    pub fn pedersen_generators(&self, count: usize) -> PedersenGenerators<C> {
        let gens = &self
            .blindfold_setup
            .as_ref()
            .expect("BlindfoldSetup required for ZK mode")
            .0;
        assert!(
            count <= gens.message_generators.len(),
            "Requested {count} Pedersen generators but BlindfoldSetup only has {}",
            gens.message_generators.len()
        );
        PedersenGenerators::new(
            gens.message_generators[..count].to_vec(),
            gens.blinding_generator,
        )
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, C: JoltCurve<F = F>, PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<C>>
    From<&JoltProverPreprocessing<F, C, PCS>> for JoltVerifierPreprocessing<F, C, PCS>
{
    fn from(prover_preprocessing: &JoltProverPreprocessing<F, C, PCS>) -> Self {
        let shared = prover_preprocessing.shared.clone();
        let generators = PCS::setup_verifier(&prover_preprocessing.generators);
        #[cfg(not(feature = "zk"))]
        let blindfold_setup = None;
        #[cfg(feature = "zk")]
        let blindfold_setup = Some(prover_preprocessing.blindfold_setup());
        Self::new(shared, generators, blindfold_setup)
    }
}
