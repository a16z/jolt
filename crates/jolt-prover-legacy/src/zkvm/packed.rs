//! The packed (Akita/lattice) prove path.
//!
//! Mirrors [`super::prover::JoltCpuProver::prove_parts`] with the lattice
//! stage swaps: one physical Akita polynomial `OneHotTrace` replaces the
//! per-polynomial streaming Dory commits, the nine-stage bytecode read-raf
//! discharges the four reduced inc claims through its fused-inc val stages
//! (producing the `FusedInc` opening at the shared 6b cycle point), the
//! lattice booleanity carries the fused-inc columns, stage 7 folds the
//! increment one-hot claims into `HammingWeightClaimReduction`, the
//! stage 8 uses one native grouped Akita opening for the trace and every
//! precommitted object.
//!
//! The prover runs over the `AkitaFp128` newtype (the legacy `JoltField`
//! impl of the same underlying fp128 element the verifier stack uses), so
//! every verifier-native conversion is a newtype unwrap. The transcript IS
//! the verifier-native `jolt_transcript::LegacyBlake2bTranscript` end to
//! end (the legacy `Transcript` vocabulary is implemented directly over it,
//! see `transcripts::verifier_native`), so commitment absorption and opening
//! proofs use the same transcript object as the stage provers: one digest
//! engine, with no state conversions or mirrored transcript interaction.

use std::{collections::BTreeMap, sync::Arc};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use jolt_akita::{AdviceScheduleParams, AkitaSetupParams};
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
    is_valid_committed_program_immediate, INVALID_COMMITTED_PROGRAM_IMMEDIATE,
};
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::lattice::{
    advice_packing_plan, precommitted_packing_plan, OneHotTraceLayoutPlan, OneHotTraceShape,
    PrecommittedPackingShape, PrefixPackedObjectPlan, ONE_HOT_TRACE_LAYOUT,
};
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial};
use jolt_openings::{
    CommitmentScheme as VerifierCommitmentScheme, EvaluationClaim, GroupOpeningClaim,
    PrecommittedClaim, PrecommittedRole, PrefixPackedClaims, TransparentObjectSetup,
};
use jolt_poly::Polynomial;
use jolt_program::preprocess::{JoltProgramPreprocessing, ProgramMetadata};
use jolt_transcript::append_length_prefixed;
use jolt_verifier::config::{
    CommitmentConfig, JoltProtocolConfig, ScalarChallengeEndianness, ZkConfig,
};
use jolt_verifier::preprocessing::{
    CommittedProgramPreprocessing as VerifierCommittedProgramPreprocessing,
    JoltVerifierPreprocessing, ProgramPreprocessing as VerifierProgramPreprocessing,
};
use jolt_verifier::proof::{JoltProof, JoltProofClaims, JoltStageProofs, TracePolynomialOrder};
use jolt_verifier::verifier::absorb_packed_program_commitments;
use jolt_verifier::VerifierError;

use crate::curve::{JoltCurve, JoltGroupElement};
use crate::field::akita::AkitaFp128;
use crate::poly::commitment::commitment_scheme::{
    CommitmentScheme, StreamingCommitmentScheme, ZkEvalCommitment,
};
use crate::poly::commitment::dory::{DoryGlobals, DoryLayout};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::{OpeningAccumulator, SumcheckId};
use crate::subprotocols::booleanity::{
    lattice_booleanity_params, FusedIncColumns, LatticeBooleanityAddressSumcheckProver,
    LatticeBooleanityCycleSumcheckProver,
};
use crate::transcripts::Transcript as LegacyTranscript;
use crate::utils::math::Math;
use crate::zkvm::bytecode::read_raf_checking::{
    BytecodeReadRafAddressSumcheckProver, BytecodeReadRafCycleSumcheckProver,
    BytecodeReadRafSumcheckParams,
};
use crate::zkvm::claim_reductions::{
    AdviceClaimReductionParams, AdviceClaimReductionProver, AdviceKind,
    HammingWeightClaimReductionParams, HammingWeightClaimReductionProver,
    PrecommittedClaimReduction,
};
use crate::zkvm::fiat_shamir_preamble;
use crate::zkvm::instruction_lookups::ra_virtual::{
    InstructionRaSumcheckParams, InstructionRaSumcheckProver as LookupsRaSumcheckProver,
};
use crate::zkvm::packed_witness::{
    pack_one_hot_columns, DigitZeroRow, FusedIncValue, FUSED_INC_BITS,
};
use crate::zkvm::prover::JoltCpuProver;
use crate::zkvm::ram::hamming_booleanity::{
    HammingBooleanitySumcheckParams, HammingBooleanitySumcheckProver,
};
use crate::zkvm::ram::ra_virtual::{RamRaVirtualParams, RamRaVirtualSumcheckProver};
use crate::zkvm::witness::CommittedPolynomial;

pub type AkitaField = jolt_akita::AkitaField;
pub type AkitaScheme = jolt_akita::AkitaScheme;
/// The verifier-native transcript engine the whole packed prove runs over.
pub type AkitaTranscript = jolt_transcript::LegacyBlake2bTranscript<AkitaField>;
/// The packed axis is transparent-only: the vector-commitment parameter is
/// the do-nothing placeholder.
pub type AkitaVc = NoVectorCommitment<AkitaField>;
/// The verifier-native proof the packed prover emits.
pub type AkitaJoltProof = JoltProof<AkitaScheme, AkitaVc>;
/// A group placeholder for the packed prover's curve parameter: the packed
/// axis is transparent-only, so no group operation is ever performed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoGroup;

macro_rules! no_group_op {
    () => {
        panic!("the packed (Akita) axis is transparent-only; no group operations exist")
    };
}

impl std::ops::Add for NoGroup {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        no_group_op!()
    }
}
impl std::ops::Sub for NoGroup {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self {
        no_group_op!()
    }
}
impl std::ops::Neg for NoGroup {
    type Output = Self;
    fn neg(self) -> Self {
        no_group_op!()
    }
}
impl<'a> std::ops::Add<&'a NoGroup> for NoGroup {
    type Output = Self;
    fn add(self, _rhs: &'a Self) -> Self {
        no_group_op!()
    }
}
impl<'a> std::ops::Sub<&'a NoGroup> for NoGroup {
    type Output = Self;
    fn sub(self, _rhs: &'a Self) -> Self {
        no_group_op!()
    }
}
impl std::ops::AddAssign for NoGroup {
    fn add_assign(&mut self, _rhs: Self) {
        no_group_op!()
    }
}
impl std::ops::SubAssign for NoGroup {
    fn sub_assign(&mut self, _rhs: Self) {
        no_group_op!()
    }
}

impl CanonicalSerialize for NoGroup {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        0
    }
}

impl ark_serialize::Valid for NoGroup {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for NoGroup {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        Ok(Self)
    }
}

impl JoltGroupElement for NoGroup {
    type Scalar = AkitaFp128;

    fn zero() -> Self {
        Self
    }

    fn is_zero(&self) -> bool {
        true
    }

    fn double(&self) -> Self {
        no_group_op!()
    }

    fn scalar_mul(&self, _scalar: &Self::Scalar) -> Self {
        no_group_op!()
    }
}

/// The packed prover's curve placeholder — see [`NoGroup`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AkitaNoCurve;

impl JoltCurve for AkitaNoCurve {
    type F = AkitaFp128;
    type G1 = NoGroup;
    type G2 = NoGroup;
    type G1Affine = NoGroup;
    type GT = NoGroup;

    fn g1_generator() -> Self::G1 {
        no_group_op!()
    }

    fn g2_generator() -> Self::G2 {
        no_group_op!()
    }

    fn g1_to_affine(_point: &Self::G1) -> Self::G1Affine {
        no_group_op!()
    }

    fn pairing(_g1: &Self::G1, _g2: &Self::G2) -> Self::GT {
        no_group_op!()
    }

    fn multi_pairing(_g1s: &[Self::G1], _g2s: &[Self::G2]) -> Self::GT {
        no_group_op!()
    }

    fn g1_msm(_bases: &[Self::G1], _scalars: &[Self::F]) -> Self::G1 {
        no_group_op!()
    }

    fn g1_affine_msm(_bases: &[Self::G1Affine], _scalars: &[Self::F]) -> Self::G1 {
        no_group_op!()
    }

    fn g2_msm(_bases: &[Self::G2], _scalars: &[Self::F]) -> Self::G2 {
        no_group_op!()
    }

    fn random_g1<R: rand_core::RngCore>(_rng: &mut R) -> Self::G1 {
        no_group_op!()
    }
}

/// A zero-sized stand-in for the legacy per-polynomial commitment machinery:
/// the Akita path commits one physical OneHotTrace polynomial, so none of
/// these entry points is ever reached.
#[derive(Clone, Debug, Default, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaPackedScheme;

macro_rules! no_per_polynomial_commitment {
    () => {
        panic!("the Akita path commits one physical OneHotTrace polynomial; legacy per-polynomial commitment entry points are unreachable")
    };
}

impl CommitmentScheme for AkitaPackedScheme {
    type Field = AkitaFp128;
    type ProverSetup = AkitaPackedScheme;
    type VerifierSetup = AkitaPackedScheme;
    type Commitment = AkitaPackedScheme;
    type Proof = AkitaPackedScheme;
    type BatchedProof = AkitaPackedScheme;
    type OpeningProofHint = AkitaPackedScheme;

    /// The packed pipeline keeps the established 2^12 trace floor so every
    /// canonical physical object has a supported folded Akita schedule.
    const MIN_PADDED_TRACE_LENGTH: usize = 1 << 12;

    fn setup_prover(_max_num_vars: usize) -> Self::ProverSetup {
        Self
    }

    fn setup_verifier(_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        Self
    }

    fn commit(
        _poly: &MultilinearPolynomial<Self::Field>,
        _setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        no_per_polynomial_commitment!()
    }

    fn batch_commit<U>(
        _polys: &[U],
        _gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        no_per_polynomial_commitment!()
    }

    fn prove<ProofTranscript: LegacyTranscript>(
        _setup: &Self::ProverSetup,
        _poly: &MultilinearPolynomial<Self::Field>,
        _opening_point: &[<Self::Field as crate::field::JoltField>::Challenge],
        _hint: Option<Self::OpeningProofHint>,
        _transcript: &mut ProofTranscript,
    ) -> (Self::Proof, Option<Self::Field>) {
        no_per_polynomial_commitment!()
    }

    fn verify<ProofTranscript: LegacyTranscript>(
        _proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        _opening_point: &[<Self::Field as crate::field::JoltField>::Challenge],
        _opening: &Self::Field,
        _commitment: &Self::Commitment,
    ) -> Result<(), crate::utils::errors::ProofVerifyError> {
        no_per_polynomial_commitment!()
    }

    fn protocol_name() -> &'static [u8] {
        b"akita-packed"
    }
}

impl StreamingCommitmentScheme for AkitaPackedScheme {
    type ChunkState = ();

    fn process_chunk<T: crate::utils::small_scalar::SmallScalar>(
        _setup: &Self::ProverSetup,
        _chunk: &[T],
    ) -> Self::ChunkState {
        no_per_polynomial_commitment!()
    }

    fn process_chunk_onehot(
        _setup: &Self::ProverSetup,
        _onehot_k: usize,
        _chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        no_per_polynomial_commitment!()
    }

    fn aggregate_chunks(
        _setup: &Self::ProverSetup,
        _onehot_k: Option<usize>,
        _tier1_commitments: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        no_per_polynomial_commitment!()
    }
}

impl ZkEvalCommitment<AkitaNoCurve> for AkitaPackedScheme {
    fn eval_commitment(_proof: &Self::Proof) -> Option<NoGroup> {
        None
    }

    fn eval_commitment_gens(_setup: &Self::ProverSetup) -> Option<(NoGroup, NoGroup)> {
        None
    }

    fn eval_commitment_gens_verifier(_setup: &Self::VerifierSetup) -> Option<(NoGroup, NoGroup)> {
        None
    }
}

impl crate::zkvm::proof::ProofField for AkitaFp128 {
    type VerifierField = AkitaField;

    fn into_verifier_field(self) -> AkitaField {
        self.0
    }
}

impl crate::zkvm::proof::ProofCurve<AkitaFp128> for AkitaNoCurve {
    type VerifierVectorCommitment = AkitaVc;
    type VerifierRoundCommitment = NoCommitment;

    fn g1_into_verifier(_commitment: NoGroup) -> NoCommitment {
        NoCommitment
    }

    fn vc_setup_from_prover_blindfold(
        _setup: &crate::poly::commitment::pedersen::PedersenGenerators<Self>,
    ) {
    }
}

/// The transparent setup of a singleton commitment object (advice word
/// objects, including direct program objects): one polynomial at `num_vars`, seeded by the
/// object plan's layout digest — the shared [`TransparentObjectSetup`]
/// convention `akita_verifier_preprocessing` and the modular packed prover
/// re-derive independently, so all sides stay on a single definition.
fn transparent_object_setup(
    num_vars: usize,
    layout_digest: [u8; 32],
) -> Result<
    (
        <AkitaScheme as VerifierCommitmentScheme>::ProverSetup,
        <AkitaScheme as VerifierCommitmentScheme>::VerifierSetup,
    ),
    jolt_openings::OpeningsError,
> {
    <AkitaScheme as TransparentObjectSetup>::transparent_object_setup(num_vars, layout_digest)
}

fn advice_object_setup(
    kind: JoltAdviceKind,
    max_advice_bytes: usize,
) -> Result<<AkitaScheme as VerifierCommitmentScheme>::ProverSetup, VerifierError> {
    let word_vars = (max_advice_bytes / 8).next_power_of_two().log_2();
    let plan = advice_packing_plan(kind, word_vars).map_err(|error| {
        VerifierError::FinalOpeningVerificationFailed {
            reason: error.to_string(),
        }
    })?;
    let (setup, _verifier_setup) =
        transparent_object_setup(plan.packing().packed_num_vars(), plan.layout_digest()).map_err(
            |error| VerifierError::FinalOpeningVerificationFailed {
                reason: error.to_string(),
            },
        )?;
    Ok(setup)
}

fn advice_physical_num_vars(
    kind: JoltAdviceKind,
    max_advice_bytes: usize,
) -> Result<usize, VerifierError> {
    let words = (max_advice_bytes / 8).next_power_of_two();
    let plan = advice_packing_plan(kind, words.log_2()).map_err(|error| {
        VerifierError::FinalOpeningVerificationFailed {
            reason: error.to_string(),
        }
    })?;
    Ok(plan.packing().packed_num_vars())
}

pub fn provision_precommitted_schedules(
    max_untrusted_advice_bytes: usize,
    max_trusted_advice_bytes: usize,
    direct_program_physical_vars: &[usize],
    one_hot_k: usize,
    max_final_num_vars: usize,
) -> Result<(), VerifierError> {
    let untrusted_physical_vars = (max_untrusted_advice_bytes > 0)
        .then(|| advice_physical_num_vars(JoltAdviceKind::Untrusted, max_untrusted_advice_bytes))
        .transpose()?;
    let trusted_physical_vars = (max_trusted_advice_bytes > 0)
        .then(|| advice_physical_num_vars(JoltAdviceKind::Trusted, max_trusted_advice_bytes))
        .transpose()?;
    jolt_akita::schedule_registry::provision_precommitted_for_k(
        untrusted_physical_vars,
        trusted_physical_vars,
        direct_program_physical_vars,
        one_hot_k,
        max_final_num_vars,
    )
    .map(|_| ())
    .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
        reason: error.to_string(),
    })
}

fn grouped_batch_poly_capacity(
    max_untrusted_advice_bytes: usize,
    max_trusted_advice_bytes: usize,
    direct_program_objects: usize,
) -> usize {
    1 + direct_program_objects
        + usize::from(max_untrusted_advice_bytes > 0)
        + usize::from(max_trusted_advice_bytes > 0)
}

/// An advice commitment object: the canonical word polynomial used by
/// both the base advice reductions and the Akita PCS opening.
pub struct AdviceObject {
    pub words: Vec<u64>,
    pub plan: PrefixPackedObjectPlan,
    pub polynomial: Polynomial<AkitaField>,
    pub commitment: <AkitaScheme as jolt_crypto::Commitment>::Output,
    pub hint: <AkitaScheme as VerifierCommitmentScheme>::OpeningHint,
    pub setup: <AkitaScheme as VerifierCommitmentScheme>::ProverSetup,
}

/// Builds the canonical zero-padded advice-word commitment. The setup is derived from the public advice shape
/// with the same fixed seed on both sides (the Akita setup is transparent).
pub fn commit_advice(
    kind: JoltAdviceKind,
    advice_bytes: &[u8],
    max_advice_bytes: usize,
    setup: &<AkitaScheme as VerifierCommitmentScheme>::ProverSetup,
) -> Result<AdviceObject, VerifierError> {
    let commit_failed = |reason: String| VerifierError::FinalOpeningVerificationFailed { reason };

    let words = common::advice::canonical_advice_words(advice_bytes, max_advice_bytes)
        .map_err(|error| commit_failed(error.to_string()))?;
    let word_vars = words.len().log_2();
    let plan =
        advice_packing_plan(kind, word_vars).map_err(|error| commit_failed(error.to_string()))?;
    let physical_vars = plan.packing().packed_num_vars();
    debug_assert_eq!(
        setup.max_num_vars(),
        physical_vars,
        "advice object setup shape must match the dense advice domain"
    );
    let mut evaluations = vec![AkitaField::zero(); 1usize << physical_vars];
    for (evaluation, word) in evaluations.iter_mut().zip(&words) {
        *evaluation = AkitaField::from_u64(*word);
    }
    let polynomial = Polynomial::new(evaluations);

    let (commitment, hint) = <AkitaScheme as VerifierCommitmentScheme>::commit(&polynomial, setup)
        .map_err(|error| commit_failed(error.to_string()))?;
    Ok(AdviceObject {
        words,
        plan,
        polynomial,
        commitment,
        hint,
        setup: setup.clone(),
    })
}

/// Precommits the trusted advice-word polynomial out of band.
/// The caller passes the returned object to the packed prove and its
/// commitment to the verifier. Runs at preprocessing time, so it builds its
/// own object setup.
pub fn commit_trusted_advice(
    trusted_advice_bytes: &[u8],
    max_trusted_advice_bytes: usize,
) -> Result<AdviceObject, VerifierError> {
    let setup = advice_object_setup(JoltAdviceKind::Trusted, max_trusted_advice_bytes)?;
    commit_advice(
        JoltAdviceKind::Trusted,
        trusted_advice_bytes,
        max_trusted_advice_bytes,
        &setup,
    )
}

/// One direct bounded-dense committed-program object.
pub struct DirectProgramObject {
    pub plan: PrefixPackedObjectPlan,
    pub commitment: <AkitaScheme as jolt_crypto::Commitment>::Output,
    pub hint: <AkitaScheme as VerifierCommitmentScheme>::OpeningHint,
}

pub struct DirectProgramObjects {
    pub objects: Vec<DirectProgramObject>,
}

/// Assembles and commits the direct bounded-dense bytecode chunks and program
/// image under their transparent object setups.
pub fn commit_direct_program(
    program: &crate::zkvm::program::FullProgramPreprocessing,
    memory_layout: &common::jolt_device::MemoryLayout,
    bytecode_chunk_count: usize,
) -> Result<DirectProgramObjects, VerifierError> {
    let commit_failed = |reason: String| VerifierError::FinalOpeningVerificationFailed { reason };
    if program
        .bytecode
        .bytecode
        .iter()
        .any(|instruction| !is_valid_committed_program_immediate(instruction.operands.imm))
    {
        return Err(commit_failed(
            INVALID_COMMITTED_PROGRAM_IMMEDIATE.to_owned(),
        ));
    }
    let bytecode_len = program.bytecode_len();
    assert!(
        bytecode_len.is_multiple_of(bytecode_chunk_count),
        "bytecode chunk count must divide bytecode length"
    );
    let log_bytecode_rows = (bytecode_len / bytecode_chunk_count).log_2();
    let image_words_padded = program.committed_program_image_num_words(memory_layout);
    let image_words =
        crate::zkvm::program::build_program_image_words_padded(program, image_words_padded);
    let shape = PrecommittedPackingShape {
        bytecode_chunks: bytecode_chunk_count,
        log_bytecode_rows,
        trace_order: TracePolynomialOrder::CycleMajor,
        program_image_log_words: Some(image_words_padded.log_2()),
    };
    let plan =
        precommitted_packing_plan(&shape).map_err(|error| commit_failed(error.to_string()))?;
    let mut chunks =
        crate::zkvm::bytecode::chunks::build_committed_bytecode_chunk_coeffs_with_layout::<
            AkitaFp128,
        >(
            &program.bytecode.bytecode,
            bytecode_chunk_count,
            DoryLayout::CycleMajor,
        )
        .into_iter()
        .map(|chunk| chunk.into_iter().map(|value| value.0).collect::<Vec<_>>());
    let objects = plan
        .objects()
        .map(|object_plan| {
            let mut evaluations = match object_plan.packing().ids()[0] {
                JoltCommittedPolynomial::BytecodeChunk(_) => chunks.next().ok_or_else(|| {
                    commit_failed("missing direct bytecode chunk witness".to_owned())
                })?,
                JoltCommittedPolynomial::ProgramImageInit => image_words
                    .iter()
                    .map(|word| AkitaField::from_u64(*word))
                    .collect(),
                _ => {
                    return Err(commit_failed(
                        "unexpected direct committed-program object".to_owned(),
                    ))
                }
            };
            evaluations.resize(
                1usize << object_plan.packing().packed_num_vars(),
                AkitaField::default(),
            );
            let witness = Polynomial::new(evaluations);
            let (setup, _verifier_setup) = transparent_object_setup(
                object_plan.packing().packed_num_vars(),
                object_plan.layout_digest(),
            )
            .map_err(|error| commit_failed(error.to_string()))?;
            let (commitment, hint) =
                <AkitaScheme as VerifierCommitmentScheme>::commit(&witness, &setup)
                    .map_err(|error| commit_failed(error.to_string()))?;
            Ok(DirectProgramObject {
                plan: object_plan.clone(),
                commitment,
                hint,
            })
        })
        .collect::<Result<Vec<_>, VerifierError>>()?;
    Ok(DirectProgramObjects { objects })
}

/// The packed sibling of `JoltSharedPreprocessing::new_committed`: marks the
/// program committed (metadata + digest) and assembles/commits the direct
/// program objects instead of per-polynomial commitments.
/// The placeholder per-polynomial structs carry unit commitments and zeroed
/// shape fields — the packed path never reads them; the real direct-program
/// commitment binds via explicit transcript absorption in canonical object
/// order, exactly like the base committed chunk commitments.
pub fn shared_preprocessing_with_direct_program(
    program: crate::zkvm::program::ProgramPreprocessing<AkitaPackedScheme>,
    memory_layout: common::jolt_device::MemoryLayout,
    max_padded_trace_length: usize,
    bytecode_chunk_count: usize,
) -> Result<
    (
        crate::zkvm::preprocessing::JoltSharedPreprocessing<AkitaPackedScheme>,
        crate::zkvm::program::CommittedProgramProverData<AkitaPackedScheme>,
        DirectProgramObjects,
    ),
    VerifierError,
> {
    let crate::zkvm::program::ProgramPreprocessing::Full(full) = program else {
        return Err(VerifierError::FinalOpeningVerificationFailed {
            reason: "packed committed preprocessing starts from a full program".to_string(),
        });
    };
    let direct_program = commit_direct_program(&full, &memory_layout, bytecode_chunk_count)?;
    let meta = full.meta();
    let meta_for_shared = meta.clone();
    let bytecode_len = full.bytecode_len();
    let bytecode_T = bytecode_len / bytecode_chunk_count;
    let committed = crate::zkvm::program::CommittedProgramPreprocessing::<AkitaPackedScheme> {
        meta,
        bytecode_commitments: crate::zkvm::bytecode::TrustedBytecodeCommitments {
            commitments: vec![AkitaPackedScheme; bytecode_chunk_count],
            num_columns: 0,
            log_k_chunk: 0,
            bytecode_chunk_count,
            bytecode_len,
            bytecode_T,
        },
        program_commitments: crate::zkvm::program::TrustedProgramCommitments {
            program_image_commitment: AkitaPackedScheme,
            program_image_num_columns: 0,
            program_image_num_words: full.committed_program_image_num_words(&memory_layout),
        },
    };
    let shared = crate::zkvm::preprocessing::JoltSharedPreprocessing::<AkitaPackedScheme> {
        program_meta: meta_for_shared,
        program: crate::zkvm::program::ProgramPreprocessing::Committed(committed),
        memory_layout,
        max_padded_trace_length,
        bytecode_chunk_count,
    };
    let prover_data = crate::zkvm::program::CommittedProgramProverData::<AkitaPackedScheme> {
        full,
        bytecode_hints: crate::zkvm::bytecode::TrustedBytecodeHints { hints: Vec::new() },
        program_hints: crate::zkvm::program::TrustedProgramHints {
            program_image_hint: AkitaPackedScheme,
        },
    };
    Ok((shared, prover_data, direct_program))
}

/// The packed prover pinned to the Akita stack.
pub type AkitaPackedProver<'a> =
    JoltCpuProver<'a, AkitaFp128, AkitaNoCurve, AkitaPackedScheme, AkitaTranscript>;

impl AkitaPackedProver<'_> {
    /// Akita setup parameters sized to the physical `OneHotTrace` polynomial.
    ///
    /// Provisions this program's grouped trusted-advice rows first: setup
    /// sizing folds them into the matrix capacity, so they must be installed
    /// before the setup this describes is built.
    #[expect(
        clippy::expect_used,
        reason = "consistent with the canonical-layout expects below; a program whose \
                  advice capacity cannot be scheduled is a preprocessing-time invariant break"
    )]
    pub fn one_hot_trace_setup_params(&self) -> AkitaSetupParams {
        let one_hot_trace_shape = self.one_hot_trace_shape();
        let shape = ONE_HOT_TRACE_LAYOUT
            .setup_shape(&one_hot_trace_shape)
            .expect("canonical OneHotTrace layout must exist");
        let layout_digest = ONE_HOT_TRACE_LAYOUT
            .layout_digest(&one_hot_trace_shape)
            .expect("canonical OneHotTrace layout digest must exist");
        let one_hot_k = 1usize << self.one_hot_params.log_k_chunk;
        let max_trusted_advice_bytes =
            self.program_io.memory_layout.max_trusted_advice_size as usize;
        let max_untrusted_advice_bytes =
            self.program_io.memory_layout.max_untrusted_advice_size as usize;
        let direct_program_physical_vars = if self.preprocessing.is_committed_mode() {
            let bytecode_len = self.preprocessing.shared.bytecode_size();
            let chunk_count = self.preprocessing.shared.bytecode_chunk_count;
            let packing = precommitted_packing_plan(&PrecommittedPackingShape {
                bytecode_chunks: chunk_count,
                log_bytecode_rows: (bytecode_len / chunk_count).log_2(),
                trace_order: TracePolynomialOrder::CycleMajor,
                program_image_log_words: Some(
                    self.preprocessing
                        .shared
                        .program
                        .committed_program_image_num_words(&self.program_io.memory_layout)
                        .log_2(),
                ),
            })
            .expect("canonical direct program layout must exist");
            packing
                .objects()
                .map(|object| object.packing().packed_num_vars())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let has_precommitted = max_trusted_advice_bytes > 0
            || max_untrusted_advice_bytes > 0
            || !direct_program_physical_vars.is_empty();
        let advice_schedule = if has_precommitted {
            // The trace this prove uses may be shorter than the program's
            // padded ceiling, but preprocessing must cover every arity a proof
            // of this program can select, so sweep up to the ceiling's arity.
            let max_final_num_vars = ONE_HOT_TRACE_LAYOUT
                .setup_shape(&OneHotTraceShape {
                    log_t: self.preprocessing.shared.max_padded_trace_length.log_2(),
                    ..one_hot_trace_shape
                })
                .expect("the padded-ceiling OneHotTrace layout must exist")
                .num_vars;
            let untrusted_physical_vars = (max_untrusted_advice_bytes > 0)
                .then(|| {
                    advice_physical_num_vars(JoltAdviceKind::Untrusted, max_untrusted_advice_bytes)
                })
                .transpose()
                .expect("untrusted-advice physical arity must derive");
            let trusted_physical_vars = (max_trusted_advice_bytes > 0)
                .then(|| {
                    advice_physical_num_vars(JoltAdviceKind::Trusted, max_trusted_advice_bytes)
                })
                .transpose()
                .expect("trusted-advice physical arity must derive");
            Some(
                AdviceScheduleParams::new(
                    untrusted_physical_vars,
                    trusted_physical_vars,
                    max_final_num_vars,
                )
                .with_direct_program_physical_arities(direct_program_physical_vars.clone()),
            )
        } else {
            None
        };
        AkitaSetupParams::one_hot_only_grouped(
            shape.num_vars,
            shape.num_polys,
            grouped_batch_poly_capacity(
                max_untrusted_advice_bytes,
                max_trusted_advice_bytes,
                direct_program_physical_vars.len(),
            ),
            layout_digest,
            one_hot_k,
            advice_schedule,
        )
    }

    fn one_hot_trace_shape(&self) -> OneHotTraceShape {
        OneHotTraceShape {
            ra_layout: self.ra_layout(),
            log_t: self.trace.len().log_2(),
            log_k_chunk: self.one_hot_params.log_k_chunk,
        }
    }

    fn ra_layout(&self) -> JoltRaPolynomialLayout {
        JoltRaPolynomialLayout::new(
            self.one_hot_params.instruction_d,
            self.one_hot_params.bytecode_d,
            self.one_hot_params.ram_d,
        )
        .expect("Jolt always commits at least one RA polynomial")
    }

    /// Builds the physical prefix-packed `OneHotTrace` polynomial.
    /// Instruction, bytecode, and increment columns omit digit zero; RAM
    /// commits every row (`specs/digit-zero-virtualization.md`).
    #[tracing::instrument(skip_all, name = "assemble_one_hot_trace")]
    fn assemble_one_hot_trace(
        &self,
        plan: &OneHotTraceLayoutPlan,
        fused_inc: &[FusedIncValue],
    ) -> jolt_poly::OneHotPolynomial {
        use crate::zkvm::instruction::LookupQuery;
        use crate::zkvm::ram::remap_address;
        use common::constants::XLEN;
        use rayon::prelude::*;

        let params = &self.one_hot_params;
        let trace = &self.trace;
        let program = self.preprocessing.materialized_program();
        let memory_layout = &self.preprocessing.shared.memory_layout;
        let cycle_data = trace
            .par_iter()
            .map(|cycle| {
                (
                    LookupQuery::<XLEN>::to_lookup_index(cycle),
                    program.get_pc(cycle),
                    remap_address(cycle.ram_access().address() as u64, memory_layout),
                )
            })
            .collect::<Vec<_>>();
        let k = 1usize << params.log_k_chunk;
        let columns = plan
            .packing()
            .ids()
            .par_iter()
            .map(|polynomial| {
                let digit_zero_row = match polynomial {
                    JoltCommittedPolynomial::InstructionRa(_)
                    | JoltCommittedPolynomial::BytecodeRa(_)
                    | JoltCommittedPolynomial::BalancedIncDigit(_)
                    | JoltCommittedPolynomial::BalancedIncCarry => DigitZeroRow::Virtualized,
                    JoltCommittedPolynomial::RamRa(_) => DigitZeroRow::Committed,
                    _ => unreachable!("OneHotTrace plan contains only canonical columns"),
                };
                let indices = cycle_data
                    .iter()
                    .zip(fused_inc)
                    .map(|((lookup_index, pc, ram_address), inc)| {
                        let selected_row = match polynomial {
                            JoltCommittedPolynomial::InstructionRa(index) => {
                                Some(params.lookup_index_chunk(*lookup_index, *index) as usize)
                            }
                            JoltCommittedPolynomial::BytecodeRa(index) => {
                                Some(params.bytecode_pc_chunk(*pc, *index) as usize)
                            }
                            JoltCommittedPolynomial::RamRa(index) => (*ram_address)
                                .map(|address| params.ram_address_chunk(address, *index) as usize),
                            JoltCommittedPolynomial::BalancedIncDigit(index) => {
                                Some(inc.balanced_digit_row(params.log_k_chunk, *index))
                            }
                            JoltCommittedPolynomial::BalancedIncCarry => {
                                Some(inc.balanced_carry_row(params.log_k_chunk))
                            }
                            _ => unreachable!("OneHotTrace plan contains only canonical columns"),
                        };
                        selected_row.map(|row| {
                            u8::try_from(row).expect("OneHotTrace K is at most the u8 row domain")
                        })
                    })
                    .collect::<Vec<_>>();
                (indices, digit_zero_row)
            })
            .collect();
        pack_one_hot_columns(k, plan.packing().slot_capacity(), columns)
    }

    /// The per-cycle fused increments, shared by the inc-column witness
    /// build and OneHotTrace assembly.
    fn fused_inc_values(&self) -> Vec<FusedIncValue> {
        use rayon::prelude::*;

        self.trace
            .par_iter()
            .map(FusedIncValue::from_cycle)
            .collect()
    }

    fn fused_inc_columns(&self, fused_cycles: &[FusedIncValue]) -> FusedIncColumns {
        use rayon::prelude::*;
        use std::sync::Arc;

        let chunk_count = FUSED_INC_BITS / self.one_hot_params.log_k_chunk;
        let width = self.one_hot_params.log_k_chunk;
        let one_hot: Vec<Arc<Vec<Option<u8>>>> = (0..chunk_count)
            .map(|index| {
                Arc::new(
                    fused_cycles
                        .par_iter()
                        .map(|cycle| Some(cycle.balanced_digit_row(width, index) as u8))
                        .collect(),
                )
            })
            .chain(core::iter::once(Arc::new(
                fused_cycles
                    .par_iter()
                    .map(|cycle| Some(cycle.balanced_carry_row(width) as u8))
                    .collect(),
            )))
            .collect();
        let fused: Vec<i128> = fused_cycles.par_iter().map(|cycle| cycle.delta).collect();
        FusedIncColumns { one_hot, fused }
    }

    /// Builds and commits the untrusted advice-word polynomial.
    /// Also materializes the base advice *word* polynomial on `self.advice`
    /// so the shared stage-4/6b/7 advice reduction machinery runs unchanged.
    ///
    /// Does not absorb: the untrusted object is a precommitted batch group, so
    /// it is committed before the trace, while its commitment must still be
    /// absorbed at the canonical position *after* the trace commitment.
    #[tracing::instrument(skip_all, name = "generate_and_commit_untrusted_advice_packed")]
    fn generate_and_commit_untrusted_advice_packed(
        &mut self,
    ) -> Result<Option<AdviceObject>, VerifierError> {
        if self.program_io.untrusted_advice.is_empty() {
            return Ok(None);
        }
        let max_advice_bytes = self.program_io.memory_layout.max_untrusted_advice_size as usize;
        // The object setup depends only on the (preprocessing-time) advice
        // shape; build it once per preprocessing instead of per prove.
        let setup = match self.preprocessing.untrusted_advice_object_setup.get() {
            Some(setup) => setup,
            None => {
                let built = advice_object_setup(JoltAdviceKind::Untrusted, max_advice_bytes)?;
                self.preprocessing
                    .untrusted_advice_object_setup
                    .get_or_init(|| built)
            }
        };
        let object = commit_advice(
            JoltAdviceKind::Untrusted,
            &self.program_io.untrusted_advice,
            max_advice_bytes,
            setup,
        )?;
        self.advice.untrusted_advice_polynomial =
            Some(MultilinearPolynomial::from(object.words.clone()));
        Ok(Some(object))
    }

    #[tracing::instrument(skip_all, name = "prove_stage6a_lattice")]
    fn prove_stage6a_lattice(
        &mut self,
        columns: &FusedIncColumns,
    ) -> (
        crate::subprotocols::sumcheck::SumcheckInstanceProof<
            AkitaFp128,
            AkitaNoCurve,
            AkitaTranscript,
        >,
        BytecodeReadRafSumcheckParams<AkitaFp128>,
        crate::subprotocols::booleanity::LatticeBooleanityCycleInput<AkitaFp128>,
    ) {
        let bytecode_read_raf_params = BytecodeReadRafSumcheckParams::gen(
            &self.preprocessing.shared.program,
            Some(self.preprocessing.materialized_program()),
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let booleanity_params = lattice_booleanity_params(
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let mut bytecode_read_raf = BytecodeReadRafAddressSumcheckProver::initialize(
            bytecode_read_raf_params,
            Arc::clone(&self.trace),
            self.preprocessing.bytecode(),
            &columns.fused,
        );
        let mut booleanity = LatticeBooleanityAddressSumcheckProver::initialize(
            booleanity_params,
            &self.trace,
            &self.preprocessing.materialized_program().bytecode,
            &self.program_io.memory_layout,
            columns.one_hot.clone(),
        );

        let (sumcheck_proof, _r, _claim) =
            self.prove_batched_sumcheck(vec![&mut bytecode_read_raf, &mut booleanity]);

        (
            sumcheck_proof,
            bytecode_read_raf.into_params(),
            booleanity.into_cycle_input(),
        )
    }

    #[tracing::instrument(skip_all, name = "prove_stage6b_lattice")]
    fn prove_stage6b_lattice(
        &mut self,
        bytecode_read_raf_params: BytecodeReadRafSumcheckParams<AkitaFp128>,
        booleanity_cycle_input: crate::subprotocols::booleanity::LatticeBooleanityCycleInput<
            AkitaFp128,
        >,
        fused_inc: Vec<i128>,
    ) -> crate::subprotocols::sumcheck::SumcheckInstanceProof<
        AkitaFp128,
        AkitaNoCurve,
        AkitaTranscript,
    > {
        let ram_hamming_booleanity_params =
            HammingBooleanitySumcheckParams::new(&self.opening_accumulator);
        let ram_ra_virtual_params = RamRaVirtualParams::new(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let lookups_ra_virtual_params = InstructionRaSumcheckParams::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let bytecode_stage_gammas: Vec<Vec<AkitaFp128>> = bytecode_read_raf_params
            .stage_gammas()
            .iter()
            .map(|gammas| gammas.to_vec())
            .collect();
        let mut bytecode_read_raf = BytecodeReadRafCycleSumcheckProver::initialize(
            bytecode_read_raf_params,
            Arc::clone(&self.trace),
            self.preprocessing.bytecode(),
            &self.opening_accumulator,
            fused_inc,
        );
        let mut booleanity = LatticeBooleanityCycleSumcheckProver::initialize(
            booleanity_cycle_input,
            &self.opening_accumulator,
        );
        let mut ram_hamming_booleanity =
            HammingBooleanitySumcheckProver::initialize(ram_hamming_booleanity_params, &self.trace);
        let mut ram_ra_virtual = RamRaVirtualSumcheckProver::initialize(
            ram_ra_virtual_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );
        let mut lookups_ra_virtual =
            LookupsRaSumcheckProver::initialize(lookups_ra_virtual_params, &self.trace);

        // The advice claim-reduction cycle phases join at the bundle's
        // canonical tail, exactly as in the base 6b assembly (the lattice
        // batch has no inc slot — the fused-inc claims are discharged inside
        // the read-raf's fused stages).
        let main_total_vars = self.trace.len().log_2() + self.one_hot_params.log_k_chunk;
        let precommitted_candidates = self.preprocessing.shared.precommitted_candidate_total_vars(
            self.preprocessing.is_committed_mode(),
            self.advice.trusted_advice_polynomial.is_some(),
            self.advice.untrusted_advice_polynomial.is_some(),
        );
        let precommitted_scheduling_reference =
            PrecommittedClaimReduction::<AkitaFp128>::scheduling_reference(
                main_total_vars,
                &precommitted_candidates,
            );
        for (kind, max_size, polynomial) in [
            (
                crate::zkvm::claim_reductions::AdviceKind::Trusted,
                self.program_io.memory_layout.max_trusted_advice_size as usize,
                &self.advice.trusted_advice_polynomial,
            ),
            (
                crate::zkvm::claim_reductions::AdviceKind::Untrusted,
                self.program_io.memory_layout.max_untrusted_advice_size as usize,
                &self.advice.untrusted_advice_polynomial,
            ),
        ] {
            if let Some(polynomial) = polynomial {
                let params = AdviceClaimReductionParams::new(
                    kind,
                    max_size,
                    precommitted_scheduling_reference,
                    &self.opening_accumulator,
                );
                let prover = AdviceClaimReductionProver::initialize(params, polynomial.clone());
                match kind {
                    crate::zkvm::claim_reductions::AdviceKind::Trusted => {
                        self.advice_reduction_prover_trusted = Some(prover)
                    }
                    crate::zkvm::claim_reductions::AdviceKind::Untrusted => {
                        self.advice_reduction_prover_untrusted = Some(prover)
                    }
                }
            }
        }
        // Committed-program mode: the bytecode/program-image claim-reduction
        // cycle phases join after the advice slots (the bundle's canonical
        // tail). `BytecodeClaimReductionParams::new` draws eta internally —
        // after the instruction-RA gamma, matching the lattice verifier.
        if self.preprocessing.is_committed_mode() {
            let bytecode_chunk_count = self.preprocessing.shared.bytecode_chunk_count;
            let bytecode_reduction_params =
                crate::zkvm::claim_reductions::BytecodeClaimReductionParams::new(
                    // The reduction folds one eta slot per STAGED val: the five
                    // base stages plus the store wire (the fused stages dedup
                    // through it and carry no staged val of their own).
                    &bytecode_stage_gammas
                        [..crate::zkvm::bytecode::read_raf_checking::NUM_VAL_CLAIMS]
                        .iter()
                        .map(Vec::as_slice)
                        .collect::<Vec<_>>(),
                    self.preprocessing.shared.bytecode_size(),
                    bytecode_chunk_count,
                    precommitted_scheduling_reference,
                    &self.opening_accumulator,
                    &mut self.transcript,
                );
            let bytecode_chunk_coeffs =
                crate::zkvm::bytecode::chunks::build_committed_bytecode_chunk_coeffs(
                    &self.preprocessing.materialized_program().bytecode.bytecode,
                    bytecode_chunk_count,
                );
            self.bytecode_reduction_prover = Some(
                crate::zkvm::claim_reductions::BytecodeClaimReductionProver::initialize(
                    bytecode_reduction_params,
                    &bytecode_chunk_coeffs,
                ),
            );

            let padded_len_words = self
                .preprocessing
                .shared
                .program
                .committed_program_image_num_words(&self.program_io.memory_layout);
            let program_image_words = crate::zkvm::program::build_program_image_words_padded(
                self.preprocessing.materialized_program(),
                padded_len_words,
            );
            let program_image_reduction_params =
                crate::zkvm::claim_reductions::ProgramImageClaimReductionParams::new(
                    &self.program_io,
                    self.preprocessing.shared.program_meta.min_bytecode_address,
                    padded_len_words,
                    self.one_hot_params.ram_k,
                    precommitted_scheduling_reference,
                    &self.opening_accumulator,
                );
            self.program_image_reduction_prover = Some(
                crate::zkvm::claim_reductions::ProgramImageClaimReductionProver::initialize(
                    program_image_reduction_params,
                    program_image_words,
                ),
            );
        }

        let mut advice_trusted = self.advice_reduction_prover_trusted.take();
        let mut advice_untrusted = self.advice_reduction_prover_untrusted.take();
        let mut bytecode_reduction = self.bytecode_reduction_prover.take();
        let mut program_image_reduction = self.program_image_reduction_prover.take();

        let mut instances: Vec<
            &mut dyn crate::subprotocols::sumcheck_prover::SumcheckInstanceProver<_, _>,
        > = vec![
            &mut bytecode_read_raf,
            &mut booleanity,
            &mut ram_hamming_booleanity,
            &mut ram_ra_virtual,
            &mut lookups_ra_virtual,
        ];
        if let Some(ref mut advice) = advice_trusted {
            instances.push(advice);
        }
        if let Some(ref mut advice) = advice_untrusted {
            instances.push(advice);
        }
        if let Some(ref mut reduction) = bytecode_reduction {
            instances.push(reduction);
        }
        if let Some(ref mut reduction) = program_image_reduction {
            instances.push(reduction);
        }

        let (sumcheck_proof, _r, _claim) =
            self.prove_batched_sumcheck(instances.iter_mut().map(|v| &mut **v as _).collect());

        self.advice_reduction_prover_trusted = advice_trusted;
        self.advice_reduction_prover_untrusted = advice_untrusted;
        self.bytecode_reduction_prover = bytecode_reduction;
        self.program_image_reduction_prover = program_image_reduction;
        sumcheck_proof
    }

    #[tracing::instrument(skip_all, name = "prove_stage7_lattice")]
    fn prove_stage7_lattice(
        &mut self,
        columns: FusedIncColumns,
    ) -> crate::subprotocols::sumcheck::SumcheckInstanceProof<
        AkitaFp128,
        AkitaNoCurve,
        AkitaTranscript,
    > {
        let hw_params = HammingWeightClaimReductionParams::new_lattice(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let hw_prover = HammingWeightClaimReductionProver::initialize_lattice(
            hw_params,
            &self.trace,
            self.preprocessing,
            &self.one_hot_params,
            &columns.one_hot,
        );

        // The advice/committed address phases join at the batch tail
        // (prefix-aligned within it), exactly as in the base stage-7
        // assembly. The Stage 7 batch is address-reduction-sized — wider
        // than the address alignment window the two-phase schedule assumes —
        // so each instance compensates the batch's extra `2^Δ` claim
        // scaling (see `boost_scale_pow_2`).
        use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams as _;
        let mut advice_instances = Vec::new();
        for advice in [
            self.advice_reduction_prover_trusted.take(),
            self.advice_reduction_prover_untrusted.take(),
        ]
        .into_iter()
        .flatten()
        {
            let mut advice = advice;
            if advice.params().precommitted.num_address_phase_rounds() > 0 {
                advice.transition_to_address_phase();
                advice_instances.push(advice);
            }
        }
        let mut bytecode_reduction = self
            .bytecode_reduction_prover
            .take()
            .filter(|prover| prover.params().precommitted.num_address_phase_rounds() > 0);
        if let Some(prover) = bytecode_reduction.as_mut() {
            prover.transition_to_address_phase();
        }
        let mut program_image_reduction = self
            .program_image_reduction_prover
            .take()
            .filter(|prover| prover.params().precommitted.num_address_phase_rounds() > 0);
        if let Some(prover) = program_image_reduction.as_mut() {
            prover.transition_to_address_phase();
        }
        let batch_rounds = [hw_prover.params.num_rounds()]
            .into_iter()
            .chain(
                advice_instances
                    .iter()
                    .map(|advice| advice.params().num_rounds()),
            )
            .chain(
                bytecode_reduction
                    .iter()
                    .map(|prover| prover.params().num_rounds()),
            )
            .chain(
                program_image_reduction
                    .iter()
                    .map(|prover| prover.params().num_rounds()),
            )
            .max()
            .unwrap_or(0);
        let mut instances: Vec<
            Box<dyn crate::subprotocols::sumcheck_prover::SumcheckInstanceProver<_, _>>,
        > = vec![Box::new(hw_prover)];
        for mut advice in advice_instances {
            advice.boost_scale_pow_2(batch_rounds - advice.params().num_rounds());
            instances.push(Box::new(advice));
        }
        if let Some(mut prover) = bytecode_reduction {
            prover.boost_scale_pow_2(batch_rounds - prover.params().num_rounds());
            instances.push(Box::new(prover));
        }
        if let Some(mut prover) = program_image_reduction {
            prover.boost_scale_pow_2(batch_rounds - prover.params().num_rounds());
            instances.push(Box::new(prover));
        }

        let (sumcheck_proof, _r, _claim) =
            self.prove_batched_sumcheck(instances.iter_mut().map(|v| &mut **v as _).collect());
        sumcheck_proof
    }

    /// The `(polynomial, relation)` pair holding a semantic column's final
    /// claim on the accumulator. This covers `OneHotTrace` and the direct
    /// committed-program objects.
    fn leaf_source(
        polynomial: JoltCommittedPolynomial,
    ) -> Result<(CommittedPolynomial, SumcheckId), VerifierError> {
        Ok(match polynomial {
            JoltCommittedPolynomial::BytecodeChunk(chunk) => (
                CommittedPolynomial::BytecodeChunk(chunk),
                SumcheckId::BytecodeClaimReduction,
            ),
            JoltCommittedPolynomial::ProgramImageInit => (
                CommittedPolynomial::ProgramImageInit,
                SumcheckId::ProgramImageClaimReduction,
            ),
            JoltCommittedPolynomial::InstructionRa(index) => (
                CommittedPolynomial::InstructionRa(index),
                SumcheckId::HammingWeightClaimReduction,
            ),
            JoltCommittedPolynomial::BytecodeRa(index) => (
                CommittedPolynomial::BytecodeRa(index),
                SumcheckId::HammingWeightClaimReduction,
            ),
            JoltCommittedPolynomial::RamRa(index) => (
                CommittedPolynomial::RamRa(index),
                SumcheckId::HammingWeightClaimReduction,
            ),
            JoltCommittedPolynomial::BalancedIncDigit(index) => (
                CommittedPolynomial::BalancedIncDigit(index),
                SumcheckId::HammingWeightClaimReduction,
            ),
            JoltCommittedPolynomial::BalancedIncCarry => (
                CommittedPolynomial::BalancedIncCarry,
                SumcheckId::HammingWeightClaimReduction,
            ),
            other => {
                return Err(VerifierError::FinalOpeningBatchFailed {
                    reason: format!("polynomial {other:?} is not a per-proof packed column"),
                })
            }
        })
    }

    /// A packed column's final claim from the accumulator, with the
    /// challenge coordinates unwrapped to verifier-field values.
    fn resolve_leaf_claim(
        &self,
        polynomial: JoltCommittedPolynomial,
    ) -> Result<(Vec<AkitaField>, AkitaField), VerifierError> {
        let (legacy, sumcheck) = Self::leaf_source(polynomial)?;
        let (point, value) = self
            .opening_accumulator
            .try_get_committed_polynomial_opening(legacy, sumcheck)
            .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
                reason: format!("missing final claim for packed column {polynomial:?}"),
            })?;
        Ok((point.r.iter().map(|value| value.0).collect(), value.0))
    }

    /// The fixed-prefix advice claim produced by the retained word-level
    /// advice claim reduction.
    fn packed_advice_claims(
        &self,
        kind: JoltAdviceKind,
        object: &AdviceObject,
    ) -> Result<PrefixPackedClaims<AkitaField>, VerifierError> {
        let batch_failed = |reason: String| VerifierError::FinalOpeningBatchFailed { reason };
        let advice_kind = match kind {
            JoltAdviceKind::Untrusted => AdviceKind::Untrusted,
            JoltAdviceKind::Trusted => AdviceKind::Trusted,
        };
        let (point, value) = self
            .opening_accumulator
            .get_advice_opening(advice_kind, SumcheckId::AdviceClaimReduction)
            .ok_or_else(|| batch_failed("missing final dense advice claim".to_string()))?;
        let logical_point = point.r.iter().map(|value| value.0).collect::<Vec<_>>();
        let claims = BTreeMap::from([(
            match kind {
                JoltAdviceKind::Trusted => JoltCommittedPolynomial::TrustedAdvice,
                JoltAdviceKind::Untrusted => JoltCommittedPolynomial::UntrustedAdvice,
            },
            EvaluationClaim::new(logical_point, value.0),
        )]);
        object
            .plan
            .packed_claims(&claims)
            .map_err(|error| batch_failed(error.to_string()))
    }

    /// Claims for one independently opened committed-program object.
    fn packed_program_claims(
        &self,
        plan: &PrefixPackedObjectPlan,
    ) -> Result<PrefixPackedClaims<AkitaField>, VerifierError> {
        let batch_failed = |reason: String| VerifierError::FinalOpeningBatchFailed { reason };
        let claims = plan
            .packing()
            .ids()
            .iter()
            .map(|polynomial| {
                let (point, value) = self.resolve_leaf_claim(*polynomial)?;
                Ok((*polynomial, EvaluationClaim::new(point, value)))
            })
            .collect::<Result<BTreeMap<_, _>, VerifierError>>()?;
        plan.packed_claims(&claims)
            .map_err(|error| batch_failed(error.to_string()))
    }

    /// The Akita prove pipeline. `object_setup` is the Akita prover setup
    /// sized to OneHotTrace ([`Self::one_hot_trace_setup_params`]);
    /// `trusted_advice` is the precommitted trusted-advice object, passed exactly
    /// when trusted advice exists.
    #[tracing::instrument(skip_all, name = "prove_packed")]
    pub fn prove_packed(
        mut self,
        object_setup: &<AkitaScheme as VerifierCommitmentScheme>::ProverSetup,
        trusted_advice: Option<&AdviceObject>,
        program: Option<&DirectProgramObjects>,
    ) -> Result<AkitaJoltProof, VerifierError> {
        assert_eq!(
            program.is_some(),
            self.preprocessing.is_committed_mode(),
            "committed-program mode and the direct program objects must agree"
        );
        assert_eq!(
            trusted_advice.is_some(),
            !self.program_io.trusted_advice.is_empty(),
            "the precommitted dense trusted-advice object must be passed exactly when trusted advice exists"
        );
        let preprocessing_digest = self.preprocessing.shared.digest();
        fiat_shamir_preamble(
            &self.program_io,
            self.one_hot_params.ram_k,
            self.trace.len(),
            self.preprocessing.shared.program_meta.entry_address,
            &self.rw_config,
            &self.one_hot_params.to_config(),
            DoryLayout::CycleMajor,
            &preprocessing_digest,
            &mut self.transcript,
        );

        // One-hot machinery (RaPolynomial and friends) reads the global trace
        // dimensions; initialize them exactly like the base commit path.
        let main_total_vars = self.trace.len().log_2() + self.one_hot_params.log_k_chunk;
        let _guard = DoryGlobals::initialize_main_with_log_embedding(
            1 << self.one_hot_params.log_k_chunk,
            self.trace.len(),
            main_total_vars,
            Some(DoryLayout::CycleMajor),
        );

        let fused_cycles = self.fused_inc_values();
        let mut fused_inc_columns = self.fused_inc_columns(&fused_cycles);
        let plan = ONE_HOT_TRACE_LAYOUT
            .plan(&self.one_hot_trace_shape())
            .expect("canonical OneHotTrace layout must exist");
        // Both advice objects are precommitted batch groups, so both are
        // committed before the trace: the final commit is conditioned on the
        // frozen profile of every precommitted group.
        let advice_object = self.generate_and_commit_untrusted_advice_packed()?;
        let mut precommitted = Vec::with_capacity(2 + program.map_or(0, |p| p.objects.len()));
        if let Some(object) = advice_object.as_ref() {
            precommitted.push((
                JoltAdviceKind::Untrusted.precommitted_role(),
                &object.commitment,
                &object.hint,
            ));
        }
        if let Some(object) = trusted_advice {
            precommitted.push((
                JoltAdviceKind::Trusted.precommitted_role(),
                &object.commitment,
                &object.hint,
            ));
        }
        if let Some(program) = program {
            for (object_index, object) in program.objects.iter().enumerate() {
                let id = object
                    .plan
                    .packing()
                    .ids()
                    .first()
                    .copied()
                    .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
                        reason: "direct committed-program object has no polynomial id".to_owned(),
                    })?;
                let role = match id {
                    JoltCommittedPolynomial::BytecodeChunk(index) => PrecommittedRole::new_indexed(
                        2 + object_index as u64,
                        b"bytecode_chunk",
                        "bytecode-chunk",
                        index as u64,
                    ),
                    JoltCommittedPolynomial::ProgramImageInit => PrecommittedRole::new(
                        2 + object_index as u64,
                        b"program_image_init",
                        "program-image-init",
                    ),
                    _ => {
                        return Err(VerifierError::FinalOpeningBatchFailed {
                            reason: "unexpected direct committed-program object role".to_owned(),
                        })
                    }
                };
                precommitted.push((role, &object.commitment, &object.hint));
            }
        }
        let one_hot_trace_witness = self.assemble_one_hot_trace(&plan, &fused_cycles);
        let precommitted_hints = precommitted
            .iter()
            .map(|(_, _, hint)| *hint)
            .collect::<Vec<_>>();
        let committed = if precommitted_hints.is_empty() {
            AkitaScheme::commit_one_hot_group_owned(
                object_setup,
                plan.layout_digest(),
                vec![one_hot_trace_witness],
            )
        } else {
            AkitaScheme::commit_one_hot_group_owned_with_precommitted(
                object_setup,
                plan.layout_digest(),
                vec![one_hot_trace_witness],
                &precommitted_hints,
            )
        };
        let (commitment, hint) =
            committed.map_err(|error| VerifierError::FinalOpeningVerificationFailed {
                reason: error.to_string(),
            })?;

        // Absorb the packed commitment objects exactly where and how the
        // verifier's `absorb_commitments` akita arm does. The commit order above
        // is independent of this absorb order.
        append_length_prefixed(&mut self.transcript, b"commitment", &commitment);
        if let Some(object) = advice_object.as_ref() {
            append_length_prefixed(
                &mut self.transcript,
                b"untrusted_advice",
                &object.commitment,
            );
        }
        if let Some(trusted) = trusted_advice {
            append_length_prefixed(&mut self.transcript, b"trusted_advice", &trusted.commitment);
            self.advice.trusted_advice_polynomial =
                Some(MultilinearPolynomial::from(trusted.words.clone()));
        }
        if let Some(program) = program {
            let commitments = program
                .objects
                .iter()
                .map(|object| object.commitment.clone())
                .collect::<Vec<_>>();
            absorb_packed_program_commitments(&commitments, &mut self.transcript);
        }

        let (stage1_uni_skip_first_round_proof, stage1_sumcheck_proof, _r_stage1) =
            self.prove_stage1();
        let (stage2_uni_skip_first_round_proof, stage2_sumcheck_proof, _r_stage2) =
            self.prove_stage2();
        let (stage3_sumcheck_proof, _r_stage3) = self.prove_stage3();
        let (stage4_sumcheck_proof, _r_stage4) = self.prove_stage4();
        let (stage5_sumcheck_proof, _r_stage5) = self.prove_stage5();
        let (stage6a_sumcheck_proof, bytecode_read_raf_params, booleanity_cycle_input) =
            self.prove_stage6a_lattice(&fused_inc_columns);
        let stage6b_sumcheck_proof = self.prove_stage6b_lattice(
            bytecode_read_raf_params,
            booleanity_cycle_input,
            std::mem::take(&mut fused_inc_columns.fused),
        );
        let stage7_sumcheck_proof = self.prove_stage7_lattice(fused_inc_columns);
        let mut common_point: Option<Vec<AkitaField>> = None;
        let mut evaluations = Vec::with_capacity(plan.packing().ids().len());
        for polynomial in plan.packing().ids() {
            let (leaf_point, value) = self.resolve_leaf_claim(*polynomial)?;
            let point = ONE_HOT_TRACE_LAYOUT
                .column_point(*polynomial, self.one_hot_params.log_k_chunk, &leaf_point)
                .map_err(|error| VerifierError::FinalOpeningBatchFailed {
                    reason: error.to_string(),
                })?;
            if let Some(expected) = &common_point {
                if expected != &point {
                    return Err(VerifierError::FinalOpeningBatchFailed {
                        reason: format!(
                            "OneHotTrace column {polynomial:?} does not share the canonical opening point"
                        ),
                    });
                }
            } else {
                common_point = Some(point);
            }
            evaluations.push(value);
        }
        let common_point = common_point.ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "OneHotTrace has no columns".to_string(),
        })?;
        let packed_claims = plan.packed_claims(common_point, evaluations);
        let packed_claim = plan
            .packing()
            .reduce_claims(&packed_claims, &mut self.transcript)
            .map_err(|error| VerifierError::FinalOpeningBatchFailed {
                reason: error.to_string(),
            })?;
        let untrusted_physical = advice_object
            .as_ref()
            .map(|object| {
                let claims = self.packed_advice_claims(JoltAdviceKind::Untrusted, object)?;
                object
                    .plan
                    .packing()
                    .reduce_claims(&claims, &mut self.transcript)
                    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
                        reason: error.to_string(),
                    })
            })
            .transpose()?;
        let trusted_physical = trusted_advice
            .map(|object| {
                let claims = self.packed_advice_claims(JoltAdviceKind::Trusted, object)?;
                object
                    .plan
                    .packing()
                    .reduce_claims(&claims, &mut self.transcript)
                    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
                        reason: error.to_string(),
                    })
            })
            .transpose()?;

        let mut batch_precommitted = Vec::with_capacity(2 + program.map_or(0, |p| p.objects.len()));
        for (role, object, claim) in [
            (
                JoltAdviceKind::Untrusted.precommitted_role(),
                advice_object.as_ref(),
                untrusted_physical.as_ref(),
            ),
            (
                JoltAdviceKind::Trusted.precommitted_role(),
                trusted_advice,
                trusted_physical.as_ref(),
            ),
        ] {
            if let (Some(object), Some(claim)) = (object, claim) {
                batch_precommitted.push((
                    PrecommittedClaim::new(
                        role,
                        GroupOpeningClaim::new(
                            object.commitment.clone(),
                            claim.point.as_slice().to_vec(),
                            vec![claim.value],
                        ),
                    ),
                    object.hint.clone(),
                ));
            }
        }
        if let Some(program) = program {
            for (object_index, object) in program.objects.iter().enumerate() {
                let claims = self.packed_program_claims(&object.plan)?;
                let physical = object
                    .plan
                    .packing()
                    .reduce_claims(&claims, &mut self.transcript)
                    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
                        reason: error.to_string(),
                    })?;
                let id = object
                    .plan
                    .packing()
                    .ids()
                    .first()
                    .copied()
                    .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
                        reason: "direct committed-program object has no polynomial id".to_owned(),
                    })?;
                let role = match id {
                    JoltCommittedPolynomial::BytecodeChunk(index) => PrecommittedRole::new_indexed(
                        2 + object_index as u64,
                        b"bytecode_chunk",
                        "bytecode-chunk",
                        index as u64,
                    ),
                    JoltCommittedPolynomial::ProgramImageInit => PrecommittedRole::new(
                        2 + object_index as u64,
                        b"program_image_init",
                        "program-image-init",
                    ),
                    _ => {
                        return Err(VerifierError::FinalOpeningBatchFailed {
                            reason: "unexpected direct committed-program object role".to_owned(),
                        })
                    }
                };
                batch_precommitted.push((
                    PrecommittedClaim::new(
                        role,
                        GroupOpeningClaim::new(
                            object.commitment.clone(),
                            physical.point.as_slice().to_vec(),
                            vec![physical.value],
                        ),
                    ),
                    object.hint.clone(),
                ));
            }
        }

        let main_group = GroupOpeningClaim::new(
            commitment.clone(),
            packed_claim.point.as_slice().to_vec(),
            vec![packed_claim.value],
        );
        let joint_opening_proof = <AkitaScheme as VerifierCommitmentScheme>::prove_batch(
            object_setup,
            batch_precommitted,
            main_group,
            hint,
            &mut self.transcript,
        )
        .map_err(|error| VerifierError::FinalOpeningBatchFailed {
            reason: error.to_string(),
        })?;

        let claims = crate::zkvm::clear_claims::build_packed_clear_claims(
            self.opening_accumulator
                .openings
                .iter()
                .map(|(id, (_point, claim))| {
                    (crate::zkvm::proof::convert_opening_id(*id), claim.0)
                }),
        )?;

        let stages = JoltStageProofs::<AkitaField, AkitaVc> {
            stage1_uni_skip_first_round_proof: crate::zkvm::proof::convert_uniskip(
                stage1_uni_skip_first_round_proof,
            ),
            stage1_sumcheck_proof: crate::zkvm::proof::convert_sumcheck(stage1_sumcheck_proof),
            stage2_uni_skip_first_round_proof: crate::zkvm::proof::convert_uniskip(
                stage2_uni_skip_first_round_proof,
            ),
            stage2_sumcheck_proof: crate::zkvm::proof::convert_sumcheck(stage2_sumcheck_proof),
            stage3_sumcheck_proof: crate::zkvm::proof::convert_sumcheck(stage3_sumcheck_proof),
            stage4_sumcheck_proof: crate::zkvm::proof::convert_sumcheck(stage4_sumcheck_proof),
            stage5_sumcheck_proof: crate::zkvm::proof::convert_sumcheck(stage5_sumcheck_proof),
            stage6a_sumcheck_proof: crate::zkvm::proof::convert_sumcheck(stage6a_sumcheck_proof),
            stage6b_sumcheck_proof: crate::zkvm::proof::convert_sumcheck(stage6b_sumcheck_proof),
            stage7_sumcheck_proof: crate::zkvm::proof::convert_sumcheck(stage7_sumcheck_proof),
        };

        Ok(JoltProof {
            protocol: JoltProtocolConfig {
                zk: ZkConfig::Transparent,
                commitment: CommitmentConfig::Packed,
                scalar_challenge_endianness: ScalarChallengeEndianness::Little,
            },
            commitments: commitment,
            stages,
            joint_opening_proof,
            untrusted_advice_commitment: advice_object
                .as_ref()
                .map(|object| object.commitment.clone()),
            claims: JoltProofClaims::Clear(claims),
            trace_length: self.trace.len(),
            ram_K: self.one_hot_params.ram_k,
            rw_config: crate::zkvm::proof::convert_read_write_config(self.rw_config.clone()),
            one_hot_config: crate::zkvm::proof::convert_one_hot_config(
                self.one_hot_params.to_config(),
            ),
            trace_polynomial_order: TracePolynomialOrder::CycleMajor,
        })
    }
}

/// The verifier preprocessing for a packed proof: the program preprocessing
/// (full-program mode), the digest, the `OneHotTrace` setup, and the per-object
/// setups derived from the public shapes (transparent setup, fixed seed —
/// the same derivation the prover's object builders use).
pub fn akita_verifier_preprocessing(
    preprocessing: &crate::zkvm::prover::JoltProverPreprocessing<
        AkitaFp128,
        AkitaNoCurve,
        AkitaPackedScheme,
    >,
    akita_verifier_setup: <AkitaScheme as VerifierCommitmentScheme>::VerifierSetup,
    direct_program: Option<&DirectProgramObjects>,
) -> JoltVerifierPreprocessing<AkitaScheme, AkitaVc> {
    let program = match &preprocessing.shared.program {
        crate::zkvm::program::ProgramPreprocessing::Full(full) => {
            VerifierProgramPreprocessing::Full(Arc::new(JoltProgramPreprocessing {
                bytecode: full.bytecode.as_ref().clone(),
                ram: full.ram.clone(),
                memory_layout: preprocessing.shared.memory_layout.clone(),
                max_padded_trace_length: preprocessing.shared.max_padded_trace_length,
            }))
        }
        crate::zkvm::program::ProgramPreprocessing::Committed(committed) => {
            let direct_program = direct_program
                .expect("committed-program mode requires direct program preprocessing");
            VerifierProgramPreprocessing::Committed(VerifierCommittedProgramPreprocessing {
                meta: ProgramMetadata {
                    entry_address: committed.meta.entry_address,
                    min_bytecode_address: committed.meta.min_bytecode_address,
                    entry_bytecode_index: committed.meta.entry_bytecode_index,
                    program_image_len_words: committed.meta.program_image_len_words,
                    bytecode_len: committed.meta.bytecode_len,
                },
                memory_layout: preprocessing.shared.memory_layout.clone(),
                max_padded_trace_length: preprocessing.shared.max_padded_trace_length,
                direct_program_commitments: direct_program
                    .objects
                    .iter()
                    .map(|object| object.commitment.clone())
                    .collect(),
                bytecode_chunk_count: preprocessing.shared.bytecode_chunk_count,
                trace_order: TracePolynomialOrder::CycleMajor,
            })
        }
    };
    let one_hot_k = akita_verifier_setup.one_hot_k();
    let akita_verifier_max_final_num_vars = akita_verifier_setup.max_num_vars();
    let layout = &preprocessing.shared.memory_layout;
    let direct_program_plan = preprocessing.shared.program.is_committed().then(|| {
        let bytecode_len = preprocessing.shared.bytecode_size();
        let bytecode_chunk_count = preprocessing.shared.bytecode_chunk_count;
        precommitted_packing_plan(&PrecommittedPackingShape {
            bytecode_chunks: bytecode_chunk_count,
            log_bytecode_rows: (bytecode_len / bytecode_chunk_count).log_2(),
            trace_order: TracePolynomialOrder::CycleMajor,
            program_image_log_words: Some(
                preprocessing
                    .shared
                    .program
                    .committed_program_image_num_words(layout)
                    .log_2(),
            ),
        })
        .expect("the canonical precommitted packing plan must exist")
    });
    let direct_program_physical_vars = direct_program_plan
        .as_ref()
        .map(|plan| {
            plan.objects()
                .map(|object| object.packing().packed_num_vars())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    provision_precommitted_schedules(
        layout.max_untrusted_advice_size as usize,
        layout.max_trusted_advice_size as usize,
        &direct_program_physical_vars,
        one_hot_k,
        akita_verifier_max_final_num_vars,
    )
    .expect("precommitted grouped schedules must provision for the verifier");

    let mut verifier_preprocessing = JoltVerifierPreprocessing::new(
        program,
        preprocessing.shared.digest(),
        akita_verifier_setup,
        None,
    );
    let advice_setup = |kind: JoltAdviceKind, max_bytes: usize| {
        (max_bytes > 0).then(|| {
            let word_vars = (max_bytes / 8).next_power_of_two().log_2();
            let plan = advice_packing_plan(kind, word_vars)
                .expect("the canonical advice layout must derive");
            let (_, verifier_setup) =
                transparent_object_setup(plan.packing().packed_num_vars(), plan.layout_digest())
                    .expect("the transparent advice-shape setup must derive");
            verifier_setup
        })
    };
    verifier_preprocessing.untrusted_advice_setup = advice_setup(
        JoltAdviceKind::Untrusted,
        layout.max_untrusted_advice_size as usize,
    );
    verifier_preprocessing.trusted_advice_setup = advice_setup(
        JoltAdviceKind::Trusted,
        layout.max_trusted_advice_size as usize,
    );
    if let Some(plan) = direct_program_plan {
        verifier_preprocessing.direct_program_setups = plan
            .objects()
            .map(|object| {
                transparent_object_setup(object.packing().packed_num_vars(), object.layout_digest())
                    .expect("the transparent program-shape setup must derive")
                    .1
            })
            .collect();
    }
    verifier_preprocessing
}

#[cfg(all(test, feature = "host"))]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::host;
    use crate::zkvm::preprocessing::JoltSharedPreprocessing;
    use crate::zkvm::program::ProgramPreprocessing;
    use crate::zkvm::prover::JoltProverPreprocessing;
    use serial_test::serial;

    /// Proves and verifies muldiv end to end over the packed (Akita) stack:
    /// the full-program packed pipeline, one `OneHotTrace` commitment object, and
    /// the joint packed opening.
    #[test]
    #[serial]
    fn muldiv_e2e_akita() {
        crate::poly::commitment::dory::DoryGlobals::reset();
        let mut program = host::Program::new("muldiv-guest");
        let (bytecode, init_memory_state, _, e_entry) = program.decode();
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let program_data =
            ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry).unwrap();
        let shared: JoltSharedPreprocessing<AkitaPackedScheme> =
            JoltSharedPreprocessing::new(program_data, io_device.memory_layout.clone(), 1 << 16);
        let prover_preprocessing = JoltProverPreprocessing::new(shared);
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = AkitaPackedProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        )
        .unwrap();
        let io_device = prover.program_io.clone();
        let setup_params = prover.one_hot_trace_setup_params();
        assert_eq!(setup_params.one_hot_k(), 16);
        let (object_setup, verifier_setup) =
            <AkitaScheme as VerifierCommitmentScheme>::setup(setup_params).unwrap();
        let proof = prover
            .prove_packed(&object_setup, None, None)
            .expect("packed prover should produce a verifier-native proof");

        let verifier_preprocessing =
            akita_verifier_preprocessing(&prover_preprocessing, verifier_setup, None);
        let verify = |proof: &AkitaJoltProof| {
            jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                &verifier_preprocessing,
                &io_device,
                proof,
                None,
            )
        };
        verify(&proof).expect("packed verifier should accept the packed proof");

        // Live tampers on the fused-inc pipeline's claim wires: the fused
        // increment's reduced claim and the hamming-reduction digit/carry
        // finals each participate in a batched output fold — an offset on
        // any of them must be rejected.
        let tamper = |mutate: &dyn Fn(&mut jolt_verifier::proof::ClearProofClaims<AkitaField>)| {
            let mut tampered = proof.clone();
            let jolt_verifier::proof::JoltProofClaims::Clear(claims) = &mut tampered.claims else {
                panic!("packed proofs carry clear claims");
            };
            mutate(claims);
            tampered
        };
        let one = AkitaField::from_u64(1);
        assert!(
            verify(&tamper(&|claims| claims
                .stage6b
                .bytecode_read_raf
                .fused_inc += one))
            .is_err(),
            "tampered read-raf fused-inc opening must be rejected"
        );
        assert!(
            verify(&tamper(&|claims| claims
                .stage7
                .hamming_weight_claim_reduction
                .balanced_inc_digits[0] += one))
            .is_err(),
            "tampered increment digit final must be rejected"
        );
        assert!(
            verify(&tamper(&|claims| claims
                .stage7
                .hamming_weight_claim_reduction
                .balanced_inc_carry += one))
            .is_err(),
            "tampered increment carry final must be rejected"
        );
    }

    /// The large-trace regime at e2e scale: small traces select K = 16 by
    /// the shared toggle, so this pins the K = 256 arm — preset dispatch,
    /// 8-bit lane mapping, and the layout digest — by overriding the
    /// prover's one-hot params before proving (the verifier accepts either
    /// regime at any trace length; the choice is carried by the proof's
    /// one-hot config and bound by the digest).
    #[test]
    #[serial]
    fn muldiv_e2e_akita_forced_k256() {
        crate::poly::commitment::dory::DoryGlobals::reset();
        let mut program = host::Program::new("muldiv-guest");
        let (bytecode, init_memory_state, _, e_entry) = program.decode();
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let program_data =
            ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry).unwrap();
        let shared: JoltSharedPreprocessing<AkitaPackedScheme> =
            JoltSharedPreprocessing::new(program_data, io_device.memory_layout.clone(), 1 << 16);
        let prover_preprocessing = JoltProverPreprocessing::new(shared);
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let mut prover = AkitaPackedProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        )
        .unwrap();
        let forced = crate::zkvm::config::OneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        };
        prover.one_hot_params = crate::zkvm::config::OneHotParams::from_config(
            &forced,
            prover_preprocessing.shared.bytecode_size(),
            prover.one_hot_params.ram_k,
        );
        let io_device = prover.program_io.clone();
        let (object_setup, verifier_setup) =
            <AkitaScheme as VerifierCommitmentScheme>::setup(prover.one_hot_trace_setup_params())
                .unwrap();
        let proof = prover
            .prove_packed(&object_setup, None, None)
            .expect("packed prover should produce a verifier-native proof");

        let verifier_preprocessing =
            akita_verifier_preprocessing(&prover_preprocessing, verifier_setup, None);
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &verifier_preprocessing,
            &io_device,
            &proof,
            None,
        )
        .expect("packed verifier should accept the forced-K256 proof");
    }
}

#[cfg(all(test, feature = "host"))]
mod advice_tests {
    // Force-link inline crates so their `inventory::submit!` entries are
    // retained by the linker (the merkle guest expands sha2 inlines).
    extern crate jolt_inlines_keccak256;
    extern crate jolt_inlines_sha2;

    use super::*;
    use crate::host;
    use crate::zkvm::preprocessing::JoltSharedPreprocessing;
    use crate::zkvm::program::ProgramPreprocessing;
    use crate::zkvm::prover::JoltProverPreprocessing;
    use serial_test::serial;

    /// The packed advice e2e: a guest consuming both advice kinds, proved
    /// over three commitment objects (`OneHotTrace`, untrusted advice, trusted advice), with
    /// per-object tamper rejection.
    #[test]
    #[serial]
    #[expect(clippy::unwrap_used)]
    fn advice_e2e_akita() {
        DoryGlobals::reset();
        let mut program = host::Program::new("merkle-tree-guest");
        let (bytecode, init_memory_state, _, e_entry) = program.decode();

        // Merkle tree with 4 leaves: input=leaf1, trusted=[leaf2, leaf3],
        // untrusted=leaf4.
        let inputs = postcard::to_stdvec(&[5u8; 32].as_slice()).unwrap();
        let untrusted_advice = postcard::to_stdvec(&[8u8; 32]).unwrap();
        let mut trusted_advice = postcard::to_stdvec(&[6u8; 32]).unwrap();
        trusted_advice.extend(postcard::to_stdvec(&[7u8; 32]).unwrap());

        let (_, _, _, io_device) = program.trace(&inputs, &untrusted_advice, &trusted_advice);
        let program_data =
            ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry).unwrap();
        let shared: JoltSharedPreprocessing<AkitaPackedScheme> =
            JoltSharedPreprocessing::new(program_data, io_device.memory_layout.clone(), 1 << 16);
        let prover_preprocessing = JoltProverPreprocessing::new(shared);
        let elf_contents = program.get_elf_contents().expect("elf contents is None");

        let trusted_object = commit_trusted_advice(
            &trusted_advice,
            io_device.memory_layout.max_trusted_advice_size as usize,
        )
        .expect("trusted advice object must commit");

        let prover: AkitaPackedProver<'_> = JoltCpuProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
            None,
            None,
            None,
        )
        .unwrap();
        let io_device = prover.program_io.clone();

        let (object_setup, verifier_setup) =
            <AkitaScheme as VerifierCommitmentScheme>::setup(prover.one_hot_trace_setup_params())
                .expect("the transparent packed setup must derive");
        let trusted_commitment = trusted_object.commitment.clone();
        let proof = prover
            .prove_packed(&object_setup, Some(&trusted_object), None)
            .expect("packed prover should produce a verifier-native proof");
        assert!(proof.untrusted_advice_commitment.is_some());

        let verifier_preprocessing =
            akita_verifier_preprocessing(&prover_preprocessing, verifier_setup, None);
        let verify = |proof: &AkitaJoltProof| {
            jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                &verifier_preprocessing,
                &io_device,
                proof,
                Some(&trusted_commitment),
            )
        };
        verify(&proof).expect("packed verifier should accept the packed proof");
    }

    /// The advice-size boundary e2e: the untrusted advice buffer fills
    /// `max_untrusted_advice_size` exactly, so the byte column carries
    /// non-degenerate lane content on every row (the 32-byte case above is
    /// padding-dominated) and the exact-capacity edge is exercised end to
    /// end. The guest reads only its postcard-encoded leaf prefix; the
    /// remaining filler bytes still enter the committed column.
    #[test]
    #[serial]
    #[expect(clippy::unwrap_used)]
    fn advice_e2e_akita_full_advice() {
        DoryGlobals::reset();
        let mut program = host::Program::new("merkle-tree-guest");
        let (bytecode, init_memory_state, _, e_entry) = program.decode();

        let inputs = postcard::to_stdvec(&[5u8; 32].as_slice()).unwrap();
        let mut trusted_advice = postcard::to_stdvec(&[6u8; 32]).unwrap();
        trusted_advice.extend(postcard::to_stdvec(&[7u8; 32]).unwrap());
        let leaf = postcard::to_stdvec(&[8u8; 32]).unwrap();

        // Fill the advice capacity exactly (the test never overrides the
        // default, and the traced layout below re-confirms the size).
        let max_untrusted = common::constants::DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE as usize;
        let mut untrusted_advice = leaf;
        untrusted_advice
            .extend((untrusted_advice.len()..max_untrusted).map(|index| (index * 31 + 7) as u8));
        assert_eq!(untrusted_advice.len(), max_untrusted);

        let (_, _, _, io_device) = program.trace(&inputs, &untrusted_advice, &trusted_advice);
        assert_eq!(
            io_device.memory_layout.max_untrusted_advice_size as usize,
            max_untrusted
        );
        let program_data =
            ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry).unwrap();
        let shared: JoltSharedPreprocessing<AkitaPackedScheme> =
            JoltSharedPreprocessing::new(program_data, io_device.memory_layout.clone(), 1 << 16);
        let prover_preprocessing = JoltProverPreprocessing::new(shared);
        let elf_contents = program.get_elf_contents().expect("elf contents is None");

        let trusted_object = commit_trusted_advice(
            &trusted_advice,
            io_device.memory_layout.max_trusted_advice_size as usize,
        )
        .expect("trusted advice object must commit");

        let prover: AkitaPackedProver<'_> = JoltCpuProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
            None,
            None,
            None,
        )
        .unwrap();
        let io_device = prover.program_io.clone();

        let (object_setup, verifier_setup) =
            <AkitaScheme as VerifierCommitmentScheme>::setup(prover.one_hot_trace_setup_params())
                .expect("the transparent packed setup must derive");
        let trusted_commitment = trusted_object.commitment.clone();
        let proof = prover
            .prove_packed(&object_setup, Some(&trusted_object), None)
            .expect("packed prover should produce a verifier-native proof");

        let verifier_preprocessing =
            akita_verifier_preprocessing(&prover_preprocessing, verifier_setup, None);
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &verifier_preprocessing,
            &io_device,
            &proof,
            Some(&trusted_commitment),
        )
        .expect("packed verifier should accept the full-advice proof");
    }
}

#[cfg(all(test, feature = "host"))]
mod committed_tests {
    use super::*;
    use crate::host;
    use crate::zkvm::program::ProgramPreprocessing;
    use crate::zkvm::prover::JoltProverPreprocessing;
    use serial_test::serial;

    /// The committed-program packed e2e: the direct bytecode chunks and
    /// program image join the main trace in one native Akita batch.
    fn committed_e2e(bytecode_chunk_count: usize) {
        DoryGlobals::reset();
        let mut program = host::Program::new("muldiv-guest");
        let (bytecode, init_memory_state, _, e_entry) = program.decode();
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let program_data = ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry)
            .expect("program preprocessing");
        let (shared, prover_data, direct_program) = shared_preprocessing_with_direct_program(
            program_data,
            io_device.memory_layout.clone(),
            1 << 16,
            bytecode_chunk_count,
        )
        .expect("packed committed preprocessing");
        let prover_preprocessing =
            JoltProverPreprocessing::new_committed(shared, prover_data, AkitaPackedScheme);
        let elf_contents = program.get_elf_contents().expect("elf contents is None");

        let prover: AkitaPackedProver<'_> = JoltCpuProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        )
        .unwrap();
        let io_device = prover.program_io.clone();

        let (object_setup, verifier_setup) =
            <AkitaScheme as VerifierCommitmentScheme>::setup(prover.one_hot_trace_setup_params())
                .expect("the transparent packed setup must derive");
        let proof = prover
            .prove_packed(&object_setup, None, Some(&direct_program))
            .expect("packed prover should produce a verifier-native proof");

        let verifier_preprocessing = akita_verifier_preprocessing(
            &prover_preprocessing,
            verifier_setup,
            Some(&direct_program),
        );
        let verify = |proof: &AkitaJoltProof| {
            jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                &verifier_preprocessing,
                &io_device,
                proof,
                None,
            )
        };
        verify(&proof).expect("packed verifier should accept the committed packed proof");
    }

    #[test]
    #[serial]
    fn muldiv_e2e_akita_committed_program() {
        committed_e2e(1);
        committed_e2e(2);
    }

    /// Timed sha2-chain prove+verify over the packed (Akita) stack —
    /// `PERF_LOG_T` selects the padded trace target (default 2^20). Ignored:
    /// release-only perf harness, run explicitly and never concurrently with
    /// other jobs.
    #[test]
    #[ignore = "release-only perf harness"]
    #[serial]
    fn sha2_chain_akita_perf() {
        use crate::zkvm::preprocessing::JoltSharedPreprocessing;
        use std::time::Instant;

        const CYCLES_PER_SHA256: f64 = 3396.0;
        let log_t: usize = std::env::var("PERF_LOG_T")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(20);
        let max_trace = 1usize << log_t;
        let iters = std::cmp::max(1, (max_trace as f64 * 0.9 / CYCLES_PER_SHA256) as u32);
        let inputs = [
            postcard::to_stdvec(&[5u8; 32]).unwrap(),
            postcard::to_stdvec(&iters).unwrap(),
        ]
        .concat();
        let mut trusted_advice = vec![0u8; 1 << 23];
        trusted_advice
            .iter_mut()
            .enumerate()
            .for_each(|(index, byte)| *byte = (index.wrapping_mul(31).wrapping_add(7)) as u8);
        // The guest's fixed-size argument consumes this prefix; all remaining
        // bytes still belong to, and are bound by, the trusted-advice object.
        trusted_advice[..32].fill(7);
        // PERF_TRACE=1 dumps a Perfetto (chrome) trace of the run to the
        // repo-root benchmark-runs/perfetto_traces/ directory.
        let _trace_guard = std::env::var("PERF_TRACE").ok().map(|_| {
            use tracing_subscriber::prelude::*;
            let dir = format!(
                "{}/../../benchmark-runs/perfetto_traces",
                env!("CARGO_MANIFEST_DIR")
            );
            std::fs::create_dir_all(&dir).ok();
            let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
                .include_args(true)
                .file(format!("{dir}/sha2-2exp{log_t}-akita.json"))
                .build();
            tracing_subscriber::registry().with(chrome_layer).init();
            guard
        });

        eprintln!("sha2-chain/akita: {iters} iterations, target 2^{log_t}");
        eprintln!(
            "trusted advice: {} bytes ({} u64 words)",
            trusted_advice.len(),
            trusted_advice.len() / 8
        );

        crate::poly::commitment::dory::DoryGlobals::reset();
        let mut program = host::Program::new("sha2-chain-guest");
        program.set_func("sha2_chain");
        program.set_max_trusted_advice_size(trusted_advice.len() as u64);
        let (bytecode, init_memory_state, _, e_entry) = program.decode();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &trusted_advice);
        let program_data =
            ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry).unwrap();
        let shared: JoltSharedPreprocessing<AkitaPackedScheme> =
            JoltSharedPreprocessing::new(program_data, io_device.memory_layout.clone(), max_trace);
        let prover_preprocessing = JoltProverPreprocessing::new(shared);
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = AkitaPackedProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &inputs,
            &[],
            &trusted_advice,
            None,
            None,
            None,
        )
        .unwrap();
        let io_device = prover.program_io.clone();
        eprintln!("trace length: {}", prover.trace.len());
        let setup_params = prover.one_hot_trace_setup_params();
        eprintln!("OneHotTrace one-hot K: {}", setup_params.one_hot_k());
        let setup_start = Instant::now();
        let (object_setup, verifier_setup) =
            <AkitaScheme as VerifierCommitmentScheme>::setup(setup_params).unwrap();
        eprintln!("akita setup: {:.2?}", setup_start.elapsed());

        let commit_start = Instant::now();
        let trusted_object = commit_trusted_advice(
            &trusted_advice,
            io_device.memory_layout.max_trusted_advice_size as usize,
        )
        .expect("trusted advice object must commit");
        eprintln!(
            "akita trusted advice commit: {:.2?}",
            commit_start.elapsed()
        );

        let prove_start = Instant::now();
        let proof = prover
            .prove_packed(&object_setup, Some(&trusted_object), None)
            .expect("packed prover should produce a verifier-native proof");
        eprintln!("akita prove: {:.2?}", prove_start.elapsed());
        // Akita's native field deliberately has no Serde implementation, so
        // the packed proof has no top-level postcard/bincode wire encoding yet.
        // Report the fixed-width field payload plus the already-serialized
        // commitment/opening envelopes; this excludes only container tags and
        // length prefixes.
        let field_elements = format!("{proof:?}").matches("Fp128(").count();
        assert!(field_elements > 0, "Akita field Debug format changed");
        let config = bincode::config::standard();
        let commitment_bytes = bincode::serde::encode_to_vec(&proof.commitments, config)
            .unwrap()
            .len();
        let opening_bytes = bincode::serde::encode_to_vec(&proof.joint_opening_proof, config)
            .unwrap()
            .len();
        let main_commitment_bytes = bincode::serde::encode_to_vec(&proof.commitments, config)
            .unwrap()
            .len();
        let trusted_commitment_bytes =
            bincode::serde::encode_to_vec(&trusted_object.commitment, config)
                .unwrap()
                .len();
        let untrusted_commitment_bytes =
            bincode::serde::encode_to_vec(&proof.untrusted_advice_commitment, config)
                .unwrap()
                .len();
        let metadata_bytes = bincode::serde::encode_to_vec(
            (
                &proof.protocol,
                &proof.trace_length,
                &proof.ram_K,
                &proof.rw_config,
                &proof.one_hot_config,
                &proof.trace_polynomial_order,
            ),
            config,
        )
        .unwrap()
        .len();
        let proof_payload_size = field_elements * 16
            + commitment_bytes
            + opening_bytes
            + untrusted_commitment_bytes
            + metadata_bytes;
        eprintln!(
            "akita proof payload size: {proof_payload_size} bytes ({field_elements} field elements; excludes container framing)"
        );
        eprintln!("akita grouped opening proof size: {opening_bytes} bytes");
        eprintln!("akita main commitment size: {main_commitment_bytes} bytes");
        eprintln!("akita trusted commitment size: {trusted_commitment_bytes} bytes");

        let verifier_preprocessing_start = Instant::now();
        let verifier_preprocessing =
            akita_verifier_preprocessing(&prover_preprocessing, verifier_setup, None);
        eprintln!(
            "akita verifier preprocessing: {:.2?}",
            verifier_preprocessing_start.elapsed()
        );
        let verify_start = Instant::now();
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &verifier_preprocessing,
            &io_device,
            &proof,
            Some(&trusted_object.commitment),
        )
        .expect("packed verifier should accept the packed proof");
        eprintln!("akita verify: {:.2?}", verify_start.elapsed());
    }
}

use jolt_crypto::{Commitment, HomomorphicCommitment, VectorCommitment};
use jolt_field::{CanonicalBytes, JoltField};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};

/// A vector-commitment placeholder for transparent-only protocol
/// configurations that never produce or verify hiding commitments (the
/// packed/lattice Jolt path): the proof model requires *some*
/// [`VectorCommitment`] type parameter, but every zk arm is rejected
/// fail-closed before a commitment could be touched.
pub struct NoVectorCommitment<F>(std::marker::PhantomData<fn() -> F>);

impl<F> Clone for NoVectorCommitment<F> {
    fn clone(&self) -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F> Debug for NoVectorCommitment<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NoVectorCommitment")
    }
}

impl<F> PartialEq for NoVectorCommitment<F> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<F> Eq for NoVectorCommitment<F> {}

/// The (empty) commitment value of [`NoVectorCommitment`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoCommitment;

// `AppendToTranscript` comes from jolt-transcript's blanket impl over
// `CanonicalBytes`: an empty canonical encoding, so absorbing a
// `NoCommitment` is a no-op.
impl CanonicalBytes for NoCommitment {
    const NUM_BYTES: usize = 0;

    fn to_bytes_le(&self, _out: &mut [u8]) {}
}

impl<F: JoltField> HomomorphicCommitment<F> for NoCommitment {
    fn add(_c1: &Self, _c2: &Self) -> Self {
        Self
    }

    fn linear_combine(_c1: &Self, _c2: &Self, _scalar: &F) -> Self {
        Self
    }
}

impl<F: JoltField> Commitment for NoVectorCommitment<F> {
    type Output = NoCommitment;
}

impl<F: JoltField> VectorCommitment for NoVectorCommitment<F> {
    type Field = F;
    type Setup = ();

    fn capacity(_setup: &Self::Setup) -> usize {
        0
    }

    #[expect(
        clippy::panic,
        reason = "transparent-only placeholder; every zk arm is rejected before a commitment could be requested"
    )]
    fn commit(
        _setup: &Self::Setup,
        _values: &[Self::Field],
        _blinding: &Self::Field,
    ) -> Self::Output {
        panic!("NoVectorCommitment never commits: the packed axis is transparent-only")
    }

    fn verify(
        _setup: &Self::Setup,
        _commitment: &Self::Output,
        _values: &[Self::Field],
        _blinding: &Self::Field,
    ) -> bool {
        false
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod advice_object_tests {
    use super::*;

    /// A couple of bytes of advice must stay provable: without the packing
    /// plan's capacity padding, the zero-variable dense domain of a one-word
    /// region has no dense fold schedule and `advice_object_setup` fails
    /// before anything is committed.
    #[test]
    fn byte_sized_advice_region_commits_and_opens() {
        let max_advice_bytes = 8;
        let advice_bytes = [5u8, 7];

        for kind in [JoltAdviceKind::Untrusted, JoltAdviceKind::Trusted] {
            let setup = advice_object_setup(kind, max_advice_bytes).unwrap();
            let AdviceObject {
                plan,
                polynomial,
                commitment,
                hint,
                ..
            } = commit_advice(kind, &advice_bytes, max_advice_bytes, &setup).unwrap();
            let column = match kind {
                JoltAdviceKind::Trusted => JoltCommittedPolynomial::TrustedAdvice,
                JoltAdviceKind::Untrusted => JoltCommittedPolynomial::UntrustedAdvice,
            };
            let logical_vars = plan.logical_num_vars(column).unwrap();
            let selector_vars = plan.packing().selector_num_vars();
            assert!(selector_vars > 0, "tiny advice must pad selector capacity");

            // Logical claim: the dense word polynomial restricted to slot zero.
            let logical_point = (0..logical_vars)
                .map(|index| AkitaField::from_u64(index as u64 + 2))
                .collect::<Vec<_>>();
            let mut physical_point = vec![AkitaField::zero(); selector_vars];
            physical_point.extend_from_slice(&logical_point);
            let value = polynomial.evaluate(&physical_point);

            let claims = std::collections::BTreeMap::from([(
                column,
                EvaluationClaim::new(logical_point, value),
            )]);
            let packed = plan.packed_claims(&claims).unwrap();

            let mut prover_transcript =
                <AkitaTranscript as jolt_transcript::Transcript>::new(b"tiny-advice-object");
            let physical = plan
                .packing()
                .reduce_claims(&packed, &mut prover_transcript)
                .unwrap();
            let proof = <AkitaScheme as VerifierCommitmentScheme>::open(
                &polynomial,
                physical.point.as_slice(),
                physical.value,
                &setup,
                Some(hint),
                &mut prover_transcript,
            )
            .unwrap();

            let (_, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
                jolt_akita::AkitaSetupParams::dense_only(
                    plan.packing().packed_num_vars(),
                    1,
                    plan.layout_digest(),
                ),
            )
            .unwrap();
            let mut verifier_transcript =
                <AkitaTranscript as jolt_transcript::Transcript>::new(b"tiny-advice-object");
            let reduced = plan
                .packing()
                .reduce_claims(&packed, &mut verifier_transcript)
                .unwrap();
            <AkitaScheme as VerifierCommitmentScheme>::verify(
                &commitment,
                reduced.point.as_slice(),
                reduced.value,
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
            )
            .unwrap();
        }
    }
}
