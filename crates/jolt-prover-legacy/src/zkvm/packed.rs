//! The packed (Akita/lattice) prove path.
//!
//! Mirrors [`super::prover::JoltCpuProver::prove_parts`] with the lattice
//! stage swaps: one native Akita commitment group `W_jolt` replaces the
//! per-polynomial streaming Dory commits, the `IncVirtualization` phase runs
//! strictly between stage 5 and the stage-6 address phase, the six-stage
//! bytecode read-raf and the lattice booleanity carry the fused-inc columns,
//! stage 6b gains `FusedIncClaimReduction`, stage 7 folds the increment
//! one-hot claims into `HammingWeightClaimReduction`, the reconstruction phase
//! settles auxiliary advice/bytecode/image columns, and stage 8 uses one native
//! same-point Akita opening for Wjolt plus packed openings for auxiliaries.
//!
//! The prover runs over the `AkitaFp128` newtype (the legacy `JoltField`
//! impl of the same underlying fp128 element the verifier stack uses), so
//! every verifier-native conversion is a newtype unwrap. The transcript IS
//! the verifier-native `jolt_transcript::LegacyBlake2bTranscript` end to
//! end (the legacy `Transcript` vocabulary is implemented directly over it,
//! see `transcripts::verifier_native`), so the verifier-native pieces — the
//! commitment absorbs, `prove_packed_openings` — take the same transcript
//! object the stage provers append to: one digest engine, no state
//! conversions, no hand-mirrored transcript interaction.

use std::sync::Arc;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::lattice::geometry::{word_byte_num_vars, WORD_BYTES};
use jolt_claims::protocols::jolt::lattice::{
    advice_bytes_packing, precommitted_packing, PrecommittedPackingShape, UnsignedIncChunking,
    WJoltLayoutPlan, WJoltShape, W_JOLT_LAYOUT,
};
use jolt_claims::protocols::jolt::{BytecodeRegisterLane, JoltAdviceKind, JoltCommittedPolynomial};
use jolt_openings::{
    prove_packed_openings, CommitmentScheme as VerifierCommitmentScheme, EvaluationClaim,
    PackedProverGroup, PackedProverObject, PrefixPackedStatement, PrefixPacking,
};
use jolt_poly::{MultilinearPoly, OneHotPolynomial};
use jolt_transcript::append_length_prefixed;
use jolt_verifier::config::{CommitmentConfig, JoltProtocolConfig, ZkConfig};
use jolt_verifier::preprocessing::JoltVerifierPreprocessing;
use jolt_verifier::proof::{JoltProof, JoltProofClaims, JoltStageProofs, TracePolynomialOrder};
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
    lattice_booleanity_params, LatticeBooleanityAddressSumcheckProver,
    LatticeBooleanityCycleSumcheckProver, LatticeIncColumns,
};
use crate::transcripts::Transcript as LegacyTranscript;
use crate::utils::math::Math;
use crate::zkvm::bytecode::read_raf_checking::{
    BytecodeReadRafAddressSumcheckProver, BytecodeReadRafCycleSumcheckProver,
    BytecodeReadRafSumcheckParams, LATTICE_N_STAGES,
};
use crate::zkvm::claim_reductions::{
    AdviceClaimReductionParams, AdviceClaimReductionProver, BytecodeReconstructionSumcheckParams,
    BytecodeReconstructionSumcheckProver, FusedIncClaimReductionParams,
    FusedIncClaimReductionProver, HammingWeightClaimReductionParams,
    HammingWeightClaimReductionProver, IncVirtualizationSumcheckParams,
    IncVirtualizationSumcheckProver, PrecommittedClaimReduction,
    ProgramImageReconstructionSumcheckParams, ProgramImageReconstructionSumcheckProver,
    TrustedAdviceReconstructionSumcheckParams, TrustedAdviceReconstructionSumcheckProver,
    UntrustedAdviceReconstructionSumcheckParams, UntrustedAdviceReconstructionSumcheckProver,
};
use crate::zkvm::fiat_shamir_preamble;
use crate::zkvm::instruction_lookups::ra_virtual::{
    InstructionRaSumcheckParams, InstructionRaSumcheckProver as LookupsRaSumcheckProver,
};
use crate::zkvm::packed_witness::{
    assemble_one_hot_members, FusedIncCycle, SparseUnitPolynomial, UNSIGNED_INC_BITS,
};
use crate::zkvm::prover::JoltCpuProver;
use crate::zkvm::ram::hamming_booleanity::{
    HammingBooleanitySumcheckParams, HammingBooleanitySumcheckProver,
};
use crate::zkvm::ram::populate_memory_states;
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
/// A packed commitment object's leaf statement over the Akita stack.
pub type AkitaPackedStatement = PrefixPackedStatement<
    AkitaField,
    JoltCommittedPolynomial,
    <AkitaScheme as jolt_crypto::Commitment>::Output,
>;

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
/// the Akita path commits Wjolt as one native one-hot member group, so none of
/// these entry points is ever reached.
#[derive(Clone, Debug, Default, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaPackedScheme;

macro_rules! no_per_polynomial_commitment {
    () => {
        panic!("the Akita path commits one native Wjolt member group; legacy per-polynomial commitment entry points are unreachable")
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

/// The transparent setup of a singleton commitment object (advice byte
/// columns, `W_prog`): one polynomial at `num_vars`, fixed zero seed — the
/// convention `akita_verifier_preprocessing` re-derives on the verifier
/// side, so the two must stay a single definition.
fn transparent_object_setup(
    num_vars: usize,
) -> Result<
    (
        <AkitaScheme as VerifierCommitmentScheme>::ProverSetup,
        <AkitaScheme as VerifierCommitmentScheme>::VerifierSetup,
    ),
    jolt_openings::OpeningsError,
> {
    <AkitaScheme as VerifierCommitmentScheme>::setup(jolt_akita::AkitaSetupParams::new(
        num_vars, 1, [0u8; 32],
    ))
}

/// A packed advice commitment object (`A_unt` per proof, `A_tru`
/// precommitted): the word polynomial the base advice reductions run over,
/// the byte one-hot column, and its commitment data.
pub struct PackedAdviceObject {
    pub words: Vec<u64>,
    pub byte_column: SparseUnitPolynomial<AkitaField>,
    pub commitment: <AkitaScheme as jolt_crypto::Commitment>::Output,
    pub hint: <AkitaScheme as VerifierCommitmentScheme>::OpeningHint,
    pub setup: <AkitaScheme as VerifierCommitmentScheme>::ProverSetup,
}

/// Builds a packed advice byte commitment object from raw advice bytes: per
/// `(place ‖ word)` row the hot value is the advice byte, zero-padded
/// past the actual advice length — the same zero padding the base word
/// polynomial carries. The setup is derived from the public advice shape
/// with the same fixed seed on both sides (the Akita setup is transparent).
pub fn packed_advice_byte_object(
    advice_bytes: &[u8],
    max_advice_bytes: usize,
) -> Result<PackedAdviceObject, VerifierError> {
    let commit_failed = |reason: String| VerifierError::FinalOpeningVerificationFailed { reason };

    let mut words = vec![0u64; max_advice_bytes / 8];
    populate_memory_states(0, advice_bytes, Some(&mut words), None);

    let word_vars = words.len().next_power_of_two().log_2();
    let cell_vars = word_byte_num_vars(word_vars);
    let limb_bits = WORD_BYTES.log_2();
    let mut one_positions = Vec::with_capacity(WORD_BYTES << word_vars);
    for limb in 0..WORD_BYTES {
        for (word_index, word) in words.iter().enumerate() {
            let byte = (word >> (8 * limb)) as u8 as usize;
            one_positions.push((((byte << limb_bits) | limb) << word_vars) | word_index);
        }
    }
    one_positions.sort_unstable();
    let byte_column = SparseUnitPolynomial::<AkitaField>::new(cell_vars, one_positions);

    let (setup, _verifier_setup) =
        transparent_object_setup(cell_vars).map_err(|error| commit_failed(error.to_string()))?;
    let (commitment, hint) =
        <AkitaScheme as VerifierCommitmentScheme>::commit(&byte_column, &setup)
            .map_err(|error| commit_failed(error.to_string()))?;
    Ok(PackedAdviceObject {
        words,
        byte_column,
        commitment,
        hint,
        setup,
    })
}

/// Precommits the trusted-advice byte one-hot column (`A_tru`) out of band.
/// The caller passes the returned object to the packed prove and its
/// commitment to the verifier.
pub fn commit_trusted_advice_packed(
    trusted_advice_bytes: &[u8],
    max_trusted_advice_bytes: usize,
) -> Result<PackedAdviceObject, VerifierError> {
    packed_advice_byte_object(trusted_advice_bytes, max_trusted_advice_bytes)
}

/// The precommitted `W_prog` commitment object (committed-program mode):
/// the packed sub-column witness (bytecode lanes + program image), its
/// Akita commitment/hint, the shape-exact setup, and the packing shape —
/// assembled and committed once at preprocessing time.
pub struct PackedProgramObject {
    pub shape: PrecommittedPackingShape,
    pub witness: SparseUnitPolynomial<AkitaField>,
    pub commitment: <AkitaScheme as jolt_crypto::Commitment>::Output,
    pub hint: <AkitaScheme as VerifierCommitmentScheme>::OpeningHint,
    pub setup: <AkitaScheme as VerifierCommitmentScheme>::ProverSetup,
}

/// Assembles and commits `W_prog` from the full (public) program: every
/// bytecode sub-column plus the program image, packed per the canonical
/// `precommitted_packing`. The setup is derived from the public program
/// shape with the same fixed seed on both sides (the Akita setup is
/// transparent). The imm lane uses the field's canonical byte width, so
/// negative immediates (`p − |imm|`) reconstruct exactly.
pub fn commit_program_packed(
    program: &crate::zkvm::program::FullProgramPreprocessing,
    memory_layout: &common::jolt_device::MemoryLayout,
    bytecode_chunk_count: usize,
) -> Result<PackedProgramObject, VerifierError> {
    let commit_failed = |reason: String| VerifierError::FinalOpeningVerificationFailed { reason };
    let imm_byte_width = <AkitaFp128 as crate::field::JoltField>::NUM_BYTES;
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
        imm_byte_width,
        program_image_log_words: Some(image_words_padded.log_2()),
    };
    let packing = precommitted_packing(&shape).map_err(|error| commit_failed(error.to_string()))?;
    let one_positions = crate::zkvm::packed_witness::assemble_precommitted_witness::<AkitaFp128>(
        &packing,
        &program.bytecode.bytecode,
        log_bytecode_rows,
        imm_byte_width,
        Some(&image_words),
    )
    .map_err(commit_failed)?;
    let witness = SparseUnitPolynomial::<AkitaField>::new(packing.packed_num_vars, one_positions);
    let (setup, _verifier_setup) = transparent_object_setup(packing.packed_num_vars)
        .map_err(|error| commit_failed(error.to_string()))?;
    let (commitment, hint) = <AkitaScheme as VerifierCommitmentScheme>::commit(&witness, &setup)
        .map_err(|error| commit_failed(error.to_string()))?;
    Ok(PackedProgramObject {
        shape,
        witness,
        commitment,
        hint,
        setup,
    })
}

/// The packed sibling of `JoltSharedPreprocessing::new_committed`: marks the
/// program committed (metadata + digest) and assembles/commits `W_prog`
/// through [`commit_program_packed`] instead of per-polynomial commitments.
/// The placeholder per-polynomial structs carry unit commitments and zeroed
/// shape fields — the packed path never reads them; the real `W_prog`
/// commitment binds via explicit transcript absorption in canonical object
/// order, exactly like the base committed chunk commitments.
pub fn shared_preprocessing_committed_packed(
    program: crate::zkvm::program::ProgramPreprocessing<AkitaPackedScheme>,
    memory_layout: common::jolt_device::MemoryLayout,
    max_padded_trace_length: usize,
    bytecode_chunk_count: usize,
) -> Result<
    (
        crate::zkvm::preprocessing::JoltSharedPreprocessing<AkitaPackedScheme>,
        crate::zkvm::program::CommittedProgramProverData<AkitaPackedScheme>,
        PackedProgramObject,
    ),
    VerifierError,
> {
    let crate::zkvm::program::ProgramPreprocessing::Full(full) = program else {
        return Err(VerifierError::FinalOpeningVerificationFailed {
            reason: "packed committed preprocessing starts from a full program".to_string(),
        });
    };
    let w_prog = commit_program_packed(&full, &memory_layout, bytecode_chunk_count)?;
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
    Ok((shared, prover_data, w_prog))
}

/// The packed prover pinned to the Akita stack.
pub type AkitaPackedProver<'a> =
    JoltCpuProver<'a, AkitaFp128, AkitaNoCurve, AkitaPackedScheme, AkitaTranscript>;

impl AkitaPackedProver<'_> {
    /// Akita setup parameters sized to Wjolt: one commitment
    /// group of per-column one-hot members over the shared cell dimension.
    pub fn wjolt_setup_params(&self) -> jolt_akita::AkitaSetupParams {
        let wjolt_shape = self.wjolt_shape();
        let shape = W_JOLT_LAYOUT
            .setup_shape(&wjolt_shape)
            .expect("canonical Wjolt layout must exist");
        let layout_digest = W_JOLT_LAYOUT
            .layout_digest(&wjolt_shape)
            .expect("canonical Wjolt layout digest must exist");
        jolt_akita::AkitaSetupParams::one_hot_only(
            shape.num_vars,
            shape.num_polys,
            layout_digest,
            1 << self.one_hot_params.log_k_chunk,
        )
    }

    fn wjolt_shape(&self) -> WJoltShape {
        WJoltShape {
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

    /// Per-cycle hot addresses of a committed `Ra` polynomial, mirroring
    /// `CommittedPolynomial::generate_witness`'s one-hot index computation.
    fn packed_ra_indices(&self, polynomial: JoltCommittedPolynomial) -> Option<Vec<Option<usize>>> {
        use crate::zkvm::instruction::LookupQuery;
        use crate::zkvm::ram::remap_address;
        use common::constants::XLEN;
        use rayon::prelude::*;

        let params = &self.one_hot_params;
        let indices = match polynomial {
            JoltCommittedPolynomial::InstructionRa(index) => self
                .trace
                .par_iter()
                .map(|cycle| {
                    let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                    Some(params.lookup_index_chunk(lookup_index, index) as usize)
                })
                .collect(),
            JoltCommittedPolynomial::BytecodeRa(index) => self
                .trace
                .par_iter()
                .map(|cycle| {
                    let pc = self.preprocessing.materialized_program().get_pc(cycle);
                    Some(params.bytecode_pc_chunk(pc, index) as usize)
                })
                .collect(),
            JoltCommittedPolynomial::RamRa(index) => self
                .trace
                .par_iter()
                .map(|cycle| {
                    remap_address(
                        cycle.ram_access().address() as u64,
                        &self.preprocessing.shared.memory_layout,
                    )
                    .map(|address| params.ram_address_chunk(address, index) as usize)
                })
                .collect(),
            _ => return None,
        };
        Some(indices)
    }

    /// The per-cycle fused increments, shared by the inc-column witness
    /// build and Wjolt assembly.
    fn fused_inc_cycles(&self) -> Vec<FusedIncCycle> {
        use rayon::prelude::*;

        self.trace
            .par_iter()
            .map(FusedIncCycle::from_cycle)
            .collect()
    }

    fn lattice_inc_columns(&self, fused_cycles: &[FusedIncCycle]) -> LatticeIncColumns {
        use rayon::prelude::*;

        let chunk_count = UNSIGNED_INC_BITS / self.one_hot_params.log_k_chunk;
        let width = self.one_hot_params.log_k_chunk;
        let hot_lanes: Vec<Vec<u8>> = (0..chunk_count)
            .map(|index| {
                fused_cycles
                    .par_iter()
                    .map(|cycle| cycle.chunk_hot_lane_bits(width, index) as u8)
                    .collect()
            })
            .collect();
        let msb_hot_lanes: Vec<u8> = fused_cycles
            .par_iter()
            .map(|cycle| u8::from(cycle.msb()))
            .collect();
        let fused: Vec<i128> = fused_cycles.par_iter().map(|cycle| cycle.delta).collect();
        LatticeIncColumns {
            hot_lanes,
            msb_hot_lanes,
            fused,
        }
    }

    /// Builds and commits the untrusted-advice byte one-hot column (`A_unt`).
    /// Also materializes the base advice *word* polynomial on `self.advice`
    /// so the shared stage-4/6b/7 advice reduction machinery runs unchanged.
    #[tracing::instrument(skip_all, name = "generate_and_commit_untrusted_advice_packed")]
    fn generate_and_commit_untrusted_advice_packed(
        &mut self,
    ) -> Result<Option<PackedAdviceObject>, VerifierError> {
        if self.program_io.untrusted_advice.is_empty() {
            return Ok(None);
        }
        let object = packed_advice_byte_object(
            &self.program_io.untrusted_advice,
            self.program_io.memory_layout.max_untrusted_advice_size as usize,
        )?;
        append_length_prefixed(
            &mut self.transcript,
            b"untrusted_advice",
            &object.commitment,
        );
        self.advice.untrusted_advice_polynomial =
            Some(MultilinearPolynomial::from(object.words.clone()));
        Ok(Some(object))
    }

    /// Builds the strict one-hot member polynomials of `W_jolt` in the
    /// canonical native-batch member order.
    #[tracing::instrument(skip_all, name = "assemble_witness")]
    fn assemble_witness(
        &self,
        members: &[JoltCommittedPolynomial],
        fused_inc: &[FusedIncCycle],
    ) -> Vec<OneHotPolynomial> {
        let chunking = UnsignedIncChunking::new(self.one_hot_params.log_k_chunk)
            .expect("log_k_chunk divides the 64 unsigned-inc bits");
        assemble_one_hot_members(
            members,
            chunking,
            self.trace.len().log_2(),
            &|polynomial| self.packed_ra_indices(polynomial),
            fused_inc,
        )
        .expect("Wjolt assembly must cover every native member")
    }

    /// The `IncVirtualization` phase: a single-instance batched sumcheck
    /// strictly between stage 5 and the stage-6 address phase.
    fn prove_inc_virtualization_phase(
        &mut self,
    ) -> crate::subprotocols::sumcheck::SumcheckInstanceProof<
        AkitaFp128,
        AkitaNoCurve,
        AkitaTranscript,
    > {
        let params = IncVirtualizationSumcheckParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let mut instance = IncVirtualizationSumcheckProver::initialize(params, self.trace.clone());
        let (proof, _r, _claim) = self.prove_batched_sumcheck(vec![&mut instance]);
        proof
    }

    #[tracing::instrument(skip_all, name = "prove_stage6a_lattice")]
    fn prove_stage6a_lattice(
        &mut self,
        columns: &LatticeIncColumns,
    ) -> (
        crate::subprotocols::sumcheck::SumcheckInstanceProof<
            AkitaFp128,
            AkitaNoCurve,
            AkitaTranscript,
        >,
        BytecodeReadRafSumcheckParams<AkitaFp128>,
        crate::subprotocols::booleanity::LatticeBooleanityCycleInput<AkitaFp128>,
    ) {
        let bytecode_read_raf_params = BytecodeReadRafSumcheckParams::gen_with_stages(
            &self.preprocessing.shared.program,
            Some(self.preprocessing.materialized_program()),
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
            LATTICE_N_STAGES,
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
        );
        let mut booleanity = LatticeBooleanityAddressSumcheckProver::initialize(
            booleanity_params,
            &self.trace,
            &self.preprocessing.materialized_program().bytecode,
            &self.program_io.memory_layout,
            columns.hot_lanes.clone(),
            columns.msb_hot_lanes.clone(),
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
        let fused_inc_params =
            FusedIncClaimReductionParams::new(self.trace.len(), &self.opening_accumulator);
        let mut fused_inc_claim_reduction =
            FusedIncClaimReductionProver::initialize(fused_inc_params, fused_inc);

        // The advice claim-reduction cycle phases join at the bundle's
        // canonical tail, exactly as in the base 6b assembly (the lattice
        // batch has no inc slot — `IncVirtualization` ran between stages 5
        // and 6a).
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
                    &bytecode_stage_gammas
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
            &mut fused_inc_claim_reduction,
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
        columns: LatticeIncColumns,
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
            columns.hot_lanes,
            columns.msb_hot_lanes,
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

    /// The reconstruction phase (the head of the stage-8 region): settles
    /// the completed advice word claims, the bytecode chunk claims, and the
    /// program-image claim against the packed byte/lane columns. Members in
    /// canonical order — untrusted advice, trusted advice, bytecode, image —
    /// matching the verifier's declaration order; runs exactly when any
    /// member is present.
    #[tracing::instrument(skip_all, name = "prove_reconstruction_phase")]
    fn prove_reconstruction_phase(
        &mut self,
        untrusted: Option<&PackedAdviceObject>,
        trusted: Option<&PackedAdviceObject>,
    ) -> Option<
        crate::subprotocols::sumcheck::SumcheckInstanceProof<
            AkitaFp128,
            AkitaNoCurve,
            AkitaTranscript,
        >,
    > {
        let committed_program = self.preprocessing.is_committed_mode();
        if untrusted.is_none() && trusted.is_none() && !committed_program {
            return None;
        }

        let word_vars =
            |object: &PackedAdviceObject| object.words.len().next_power_of_two().log_2();
        let mut untrusted_prover = untrusted.map(|object| {
            let params = UntrustedAdviceReconstructionSumcheckParams::new(
                word_vars(object),
                &self.opening_accumulator,
                &mut self.transcript,
            );
            UntrustedAdviceReconstructionSumcheckProver::initialize(params, &object.words)
        });
        let mut trusted_prover = trusted.map(|object| {
            let params = TrustedAdviceReconstructionSumcheckParams::new(
                word_vars(object),
                &self.opening_accumulator,
            );
            TrustedAdviceReconstructionSumcheckProver::initialize(params, &object.words)
        });
        let mut program_provers = committed_program.then(|| {
            let bytecode_chunk_count = self.preprocessing.shared.bytecode_chunk_count;
            let log_rows =
                (self.preprocessing.shared.bytecode_size() / bytecode_chunk_count).log_2();
            let bytecode_params = BytecodeReconstructionSumcheckParams::new(
                bytecode_chunk_count,
                <AkitaFp128 as crate::field::JoltField>::NUM_BYTES,
                &self.opening_accumulator,
                &mut self.transcript,
            );
            let bytecode_prover = BytecodeReconstructionSumcheckProver::initialize(
                bytecode_params,
                &self.preprocessing.materialized_program().bytecode.bytecode,
                log_rows,
            );
            let image_params =
                ProgramImageReconstructionSumcheckParams::new(&self.opening_accumulator);
            let padded_len_words = self
                .preprocessing
                .shared
                .program
                .committed_program_image_num_words(&self.program_io.memory_layout);
            let image_words = crate::zkvm::program::build_program_image_words_padded(
                self.preprocessing.materialized_program(),
                padded_len_words,
            );
            let image_prover =
                ProgramImageReconstructionSumcheckProver::initialize(image_params, &image_words);
            (bytecode_prover, image_prover)
        });

        let mut instances: Vec<
            &mut dyn crate::subprotocols::sumcheck_prover::SumcheckInstanceProver<_, _>,
        > = Vec::new();
        if let Some(prover) = untrusted_prover.as_mut() {
            instances.push(prover);
        }
        if let Some(prover) = trusted_prover.as_mut() {
            instances.push(prover);
        }
        if let Some((bytecode, image)) = program_provers.as_mut() {
            instances.push(bytecode);
            instances.push(image);
        }

        let (proof, _r, _claim) = self.prove_batched_sumcheck(instances);
        Some(proof)
    }

    /// The legacy `(polynomial, sumcheck)` pair holding a `W_jolt` packed
    /// column's final claim on the accumulator.
    /// The legacy `(polynomial, relation)` pair holding a packed column's
    /// final claim on the accumulator — `W_jolt` members and `W_prog`
    /// sub-columns alike (their variant sets are disjoint).
    fn leaf_source(
        polynomial: JoltCommittedPolynomial,
    ) -> Result<(CommittedPolynomial, SumcheckId), VerifierError> {
        Ok(match polynomial {
            JoltCommittedPolynomial::BytecodeRegisterSelector { chunk, lane } => {
                let lane = match lane {
                    BytecodeRegisterLane::Rs1 => 0,
                    BytecodeRegisterLane::Rs2 => 1,
                    BytecodeRegisterLane::Rd => 2,
                };
                (
                    CommittedPolynomial::BytecodeRegisterSelector(chunk, lane),
                    SumcheckId::BytecodeChunkReconstruction,
                )
            }
            JoltCommittedPolynomial::BytecodeCircuitFlag { chunk, flag } => (
                CommittedPolynomial::BytecodeCircuitFlag(chunk, flag),
                SumcheckId::BytecodeChunkReconstruction,
            ),
            JoltCommittedPolynomial::BytecodeInstructionFlag { chunk, flag } => (
                CommittedPolynomial::BytecodeInstructionFlag(chunk, flag),
                SumcheckId::BytecodeChunkReconstruction,
            ),
            JoltCommittedPolynomial::BytecodeLookupSelector { chunk } => (
                CommittedPolynomial::BytecodeLookupSelector(chunk),
                SumcheckId::BytecodeChunkReconstruction,
            ),
            JoltCommittedPolynomial::BytecodeRafFlag { chunk } => (
                CommittedPolynomial::BytecodeRafFlag(chunk),
                SumcheckId::BytecodeChunkReconstruction,
            ),
            JoltCommittedPolynomial::BytecodeUnexpandedPcBytes { chunk } => (
                CommittedPolynomial::BytecodeUnexpandedPcBytes(chunk),
                SumcheckId::BytecodeChunkReconstruction,
            ),
            JoltCommittedPolynomial::BytecodeImmBytes { chunk } => (
                CommittedPolynomial::BytecodeImmBytes(chunk),
                SumcheckId::BytecodeChunkReconstruction,
            ),
            JoltCommittedPolynomial::ProgramImageBytes => (
                CommittedPolynomial::ProgramImageBytes,
                SumcheckId::ProgramImageReconstruction,
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
            JoltCommittedPolynomial::UnsignedIncChunk(index) => (
                CommittedPolynomial::UnsignedIncChunk(index),
                SumcheckId::HammingWeightClaimReduction,
            ),
            JoltCommittedPolynomial::UnsignedIncMsb => (
                CommittedPolynomial::UnsignedIncMsb,
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

    /// One packed advice object's statement: its single byte-column claim
    /// from the reconstruction phase.
    fn packed_advice_statement(
        &self,
        kind: JoltAdviceKind,
        object: &PackedAdviceObject,
    ) -> Result<(PrefixPacking<JoltCommittedPolynomial>, AkitaPackedStatement), VerifierError> {
        let batch_failed = |reason: String| VerifierError::FinalOpeningBatchFailed { reason };
        let word_vars = object.words.len().next_power_of_two().log_2();
        let packing = advice_bytes_packing(kind, word_vars)
            .map_err(|error| batch_failed(error.to_string()))?;
        let (advice_kind, sumcheck) = match kind {
            JoltAdviceKind::Untrusted => (
                crate::zkvm::claim_reductions::AdviceKind::Untrusted,
                SumcheckId::UntrustedAdviceReconstruction,
            ),
            JoltAdviceKind::Trusted => (
                crate::zkvm::claim_reductions::AdviceKind::Trusted,
                SumcheckId::TrustedAdviceReconstruction,
            ),
        };
        let (point, value) = self
            .opening_accumulator
            .get_advice_opening(advice_kind, sumcheck)
            .ok_or_else(|| batch_failed("missing packed advice byte-column claim".to_string()))?;
        let statement = PrefixPackedStatement::new(
            object.commitment.clone(),
            vec![(
                JoltCommittedPolynomial::advice_bytes(kind),
                EvaluationClaim::new(
                    point.r.iter().map(|value| value.0).collect::<Vec<_>>(),
                    value.0,
                ),
            )],
        );
        Ok((packing, statement))
    }

    /// The `W_prog` statement: one final claim per precommitted sub-column,
    /// pulled from the accumulator at the reconstruction phase's produced
    /// ids.
    fn packed_program_statement(
        &self,
        object: &PackedProgramObject,
    ) -> Result<(PrefixPacking<JoltCommittedPolynomial>, AkitaPackedStatement), VerifierError> {
        let batch_failed = |reason: String| VerifierError::FinalOpeningBatchFailed { reason };
        let packing =
            precommitted_packing(&object.shape).map_err(|error| batch_failed(error.to_string()))?;
        let mut claims = Vec::new();
        for (polynomial, _slot) in &packing {
            let (legacy, sumcheck) = Self::leaf_source(*polynomial)?;
            let (point, value) = self
                .opening_accumulator
                .try_get_committed_polynomial_opening(legacy, sumcheck)
                .ok_or_else(|| {
                    batch_failed(format!(
                        "missing final claim for precommitted column {polynomial:?}"
                    ))
                })?;
            claims.push((
                *polynomial,
                EvaluationClaim::new(
                    point.r.iter().map(|value| value.0).collect::<Vec<_>>(),
                    value.0,
                ),
            ));
        }
        Ok((
            packing,
            AkitaPackedStatement::new(object.commitment.clone(), claims),
        ))
    }

    /// The Akita prove pipeline. `object_setup` is the Akita prover setup
    /// sized to Wjolt ([`Self::wjolt_setup_params`]);
    /// `trusted_advice` is the precommitted `A_tru` object, passed exactly
    /// when trusted advice exists.
    #[tracing::instrument(skip_all, name = "prove_packed")]
    pub fn prove_packed(
        mut self,
        object_setup: &<AkitaScheme as VerifierCommitmentScheme>::ProverSetup,
        trusted_advice: Option<PackedAdviceObject>,
        program: Option<PackedProgramObject>,
    ) -> Result<AkitaJoltProof, VerifierError> {
        assert!(
            !cfg!(feature = "zk"),
            "zk x lattice is rejected fail-closed"
        );
        assert_eq!(
            program.is_some(),
            self.preprocessing.is_committed_mode(),
            "committed-program mode and the W_prog object must agree"
        );
        assert_eq!(
            trusted_advice.is_some(),
            !self.program_io.trusted_advice.is_empty(),
            "the precommitted A_tru object must be passed exactly when trusted advice exists"
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

        let fused_cycles = self.fused_inc_cycles();
        let columns = self.lattice_inc_columns(&fused_cycles);
        let plan = W_JOLT_LAYOUT
            .plan(&self.wjolt_shape())
            .expect("canonical Wjolt layout must exist");
        let WJoltLayoutPlan { members, .. } = &plan;
        let w_jolt_witness = self.assemble_witness(members, &fused_cycles);
        let (commitment, hint) = AkitaScheme::commit_one_hot_group(
            object_setup,
            object_setup.default_layout_digest(),
            &w_jolt_witness,
        )
        .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
            reason: error.to_string(),
        })?;

        // Absorb the packed commitment objects exactly where and how the
        // verifier's `absorb_commitments` akita arm does.
        append_length_prefixed(&mut self.transcript, b"commitment", &commitment);
        let advice_object = self.generate_and_commit_untrusted_advice_packed()?;
        if let Some(trusted) = trusted_advice.as_ref() {
            append_length_prefixed(&mut self.transcript, b"trusted_advice", &trusted.commitment);
            self.advice.trusted_advice_polynomial =
                Some(MultilinearPolynomial::from(trusted.words.clone()));
        }
        if let Some(program) = program.as_ref() {
            append_length_prefixed(
                &mut self.transcript,
                b"w_prog_commitment",
                &program.commitment,
            );
        }

        let (stage1_uni_skip_first_round_proof, stage1_sumcheck_proof, _r_stage1) =
            self.prove_stage1();
        let (stage2_uni_skip_first_round_proof, stage2_sumcheck_proof, _r_stage2) =
            self.prove_stage2();
        let (stage3_sumcheck_proof, _r_stage3) = self.prove_stage3();
        let (stage4_sumcheck_proof, _r_stage4) = self.prove_stage4();
        let (stage5_sumcheck_proof, _r_stage5) = self.prove_stage5();
        let inc_virtualization_proof = self.prove_inc_virtualization_phase();
        let (stage6a_sumcheck_proof, bytecode_read_raf_params, booleanity_cycle_input) =
            self.prove_stage6a_lattice(&columns);
        let stage6b_sumcheck_proof = self.prove_stage6b_lattice(
            bytecode_read_raf_params,
            booleanity_cycle_input,
            columns.fused.clone(),
        );
        let stage7_sumcheck_proof = self.prove_stage7_lattice(columns);
        let reconstruction_proof =
            self.prove_reconstruction_phase(advice_object.as_ref(), trusted_advice.as_ref());

        // Stage 8: Wjolt opens directly as one native same-point batch. Advice
        // and Wprog, when present, remain auxiliary packed objects.
        let mut common_point: Option<Vec<AkitaField>> = None;
        let mut w_jolt_evaluations = Vec::with_capacity(members.len());
        for polynomial in members {
            let (leaf_point, value) = self.resolve_leaf_claim(*polynomial)?;
            let point = W_JOLT_LAYOUT
                .member_point(*polynomial, self.one_hot_params.log_k_chunk, &leaf_point)
                .map_err(|error| VerifierError::FinalOpeningBatchFailed {
                    reason: error.to_string(),
                })?;
            if let Some(expected) = &common_point {
                if expected != &point {
                    return Err(VerifierError::FinalOpeningBatchFailed {
                        reason: format!(
                            "Wjolt member {polynomial:?} does not share the canonical opening point"
                        ),
                    });
                }
            } else {
                common_point = Some(point);
            }
            w_jolt_evaluations.push(value);
        }
        let common_point = common_point.ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "Wjolt has no members".to_string(),
        })?;
        let w_jolt_polys: Vec<&dyn MultilinearPoly<AkitaField>> = w_jolt_witness
            .iter()
            .map(|member| member as &dyn MultilinearPoly<AkitaField>)
            .collect();
        let w_jolt_opening = <AkitaScheme as VerifierCommitmentScheme>::open_batch(
            &w_jolt_polys,
            &common_point,
            &w_jolt_evaluations,
            object_setup,
            hint,
            &mut self.transcript,
        )
        .map_err(|error| VerifierError::FinalOpeningBatchFailed {
            reason: error.to_string(),
        })?;
        let untrusted_statement = advice_object
            .as_ref()
            .map(|object| {
                self.packed_advice_statement(JoltAdviceKind::Untrusted, object)
                    .map(|(packing, statement)| (object, packing, statement))
            })
            .transpose()?;
        let trusted_statement = trusted_advice
            .as_ref()
            .map(|object| {
                self.packed_advice_statement(JoltAdviceKind::Trusted, object)
                    .map(|(packing, statement)| (object, packing, statement))
            })
            .transpose()?;
        let program_statement = program
            .as_ref()
            .map(|object| {
                self.packed_program_statement(object)
                    .map(|(packing, statement)| (object, packing, statement))
            })
            .transpose()?;
        let mut objects: Vec<PackedProverObject<'_, AkitaScheme, JoltCommittedPolynomial>> =
            Vec::new();
        let mut groups = Vec::new();
        for entry in [untrusted_statement.as_ref(), trusted_statement.as_ref()]
            .into_iter()
            .flatten()
        {
            let (object, packing, statement) = entry;
            groups.push(PackedProverGroup::singleton(
                objects.len(),
                Some(object.hint.clone()),
            ));
            objects.push(PackedProverObject {
                packing,
                statement,
                polynomial: &object.byte_column,
                setup: &object.setup,
            });
        }
        if let Some((object, packing, statement)) = program_statement.as_ref() {
            groups.push(PackedProverGroup::singleton(
                objects.len(),
                Some(object.hint.clone()),
            ));
            objects.push(PackedProverObject {
                packing,
                statement,
                polynomial: &object.witness,
                setup: &object.setup,
            });
        }
        let auxiliary = if objects.is_empty() {
            None
        } else {
            Some(
                prove_packed_openings(objects, groups, &mut self.transcript).map_err(|error| {
                    VerifierError::FinalOpeningBatchFailed {
                        reason: error.to_string(),
                    }
                })?,
            )
        };
        let joint_opening_proof = jolt_verifier::proof::AkitaJointOpeningProof {
            w_jolt: w_jolt_opening,
            auxiliary,
        };

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
            inc_virtualization_sumcheck_proof: crate::zkvm::proof::convert_sumcheck(
                inc_virtualization_proof,
            ),
            stage6a_sumcheck_proof: crate::zkvm::proof::convert_sumcheck(stage6a_sumcheck_proof),
            stage6b_sumcheck_proof: crate::zkvm::proof::convert_sumcheck(stage6b_sumcheck_proof),
            stage7_sumcheck_proof: crate::zkvm::proof::convert_sumcheck(stage7_sumcheck_proof),
            reconstruction_sumcheck_proof: reconstruction_proof
                .map(crate::zkvm::proof::convert_sumcheck),
        };

        Ok(JoltProof {
            protocol: JoltProtocolConfig {
                zk: ZkConfig::Transparent,
                commitment: CommitmentConfig::Packed,
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
/// (full-program mode), the digest, the `W_jolt` setup, and the per-object
/// setups derived from the public shapes (transparent setup, fixed seed —
/// the same derivation the prover's object builders use).
pub fn akita_verifier_preprocessing(
    preprocessing: &crate::zkvm::prover::JoltProverPreprocessing<
        AkitaFp128,
        AkitaNoCurve,
        AkitaPackedScheme,
    >,
    akita_verifier_setup: <AkitaScheme as VerifierCommitmentScheme>::VerifierSetup,
    w_prog_commitment: Option<<AkitaScheme as jolt_crypto::Commitment>::Output>,
) -> JoltVerifierPreprocessing<AkitaScheme, AkitaVc> {
    let program = match &preprocessing.shared.program {
        crate::zkvm::program::ProgramPreprocessing::Full(full) => {
            jolt_verifier::preprocessing::ProgramPreprocessing::Full(
                jolt_program::preprocess::JoltProgramPreprocessing {
                    bytecode: full.bytecode.as_ref().clone(),
                    ram: full.ram.clone(),
                    memory_layout: preprocessing.shared.memory_layout.clone(),
                    max_padded_trace_length: preprocessing.shared.max_padded_trace_length,
                },
            )
        }
        crate::zkvm::program::ProgramPreprocessing::Committed(committed) => {
            jolt_verifier::preprocessing::ProgramPreprocessing::Committed(
                jolt_verifier::preprocessing::CommittedProgramPreprocessing {
                    meta: jolt_program::preprocess::ProgramMetadata {
                        entry_address: committed.meta.entry_address,
                        min_bytecode_address: committed.meta.min_bytecode_address,
                        entry_bytecode_index: committed.meta.entry_bytecode_index,
                        program_image_len_words: committed.meta.program_image_len_words,
                        bytecode_len: committed.meta.bytecode_len,
                    },
                    memory_layout: preprocessing.shared.memory_layout.clone(),
                    max_padded_trace_length: preprocessing.shared.max_padded_trace_length,
                    w_prog_commitment: w_prog_commitment
                        .expect("committed-program mode requires the W_prog commitment"),
                    bytecode_chunk_count: preprocessing.shared.bytecode_chunk_count,
                },
            )
        }
    };
    let committed_mode = preprocessing.shared.program.is_committed();
    let mut verifier_preprocessing = JoltVerifierPreprocessing::new(
        program,
        preprocessing.shared.digest(),
        akita_verifier_setup,
        None,
    );
    // The per-kind advice commitment-object setups are derived from the
    // public advice shapes with the same fixed seed the prover uses (the
    // Akita setup is transparent).
    let advice_setup = |max_bytes: usize| {
        (max_bytes > 0).then(|| {
            let word_vars = (max_bytes / 8).next_power_of_two().log_2();
            let (_, verifier_setup) = transparent_object_setup(word_byte_num_vars(word_vars))
                .expect("the transparent advice-shape setup must derive");
            verifier_setup
        })
    };
    let layout = &preprocessing.shared.memory_layout;
    verifier_preprocessing.untrusted_advice_setup =
        advice_setup(layout.max_untrusted_advice_size as usize);
    verifier_preprocessing.trusted_advice_setup =
        advice_setup(layout.max_trusted_advice_size as usize);
    // The W_prog setup is derived from the public program shape (transparent
    // setup, fixed seed) — the same shape `commit_program_packed` committed
    // at.
    if committed_mode {
        let imm_byte_width = <AkitaFp128 as crate::field::JoltField>::NUM_BYTES;
        let bytecode_len = preprocessing.shared.bytecode_size();
        let bytecode_chunk_count = preprocessing.shared.bytecode_chunk_count;
        let shape = PrecommittedPackingShape {
            bytecode_chunks: bytecode_chunk_count,
            log_bytecode_rows: (bytecode_len / bytecode_chunk_count).log_2(),
            imm_byte_width,
            program_image_log_words: Some(
                preprocessing
                    .shared
                    .program
                    .committed_program_image_num_words(&preprocessing.shared.memory_layout)
                    .log_2(),
            ),
        };
        let packing =
            precommitted_packing(&shape).expect("the canonical precommitted packing must exist");
        let (_, program_verifier_setup) = transparent_object_setup(packing.packed_num_vars)
            .expect("the transparent program-shape setup must derive");
        verifier_preprocessing.w_prog_setup = Some(program_verifier_setup);
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
    /// the full-program packed pipeline, one `W_jolt` commitment object, and
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
        );
        let io_device = prover.program_io.clone();
        let setup_params = prover.wjolt_setup_params();
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
        // increment's reduced claim and the hamming-reduction chunk/msb
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
                .fused_inc_claim_reduction
                .fused_inc += one))
            .is_err(),
            "tampered fused-inc reduced claim must be rejected"
        );
        assert!(
            verify(&tamper(&|claims| claims
                .stage7
                .hamming_weight_claim_reduction
                .unsigned_inc_chunks[0] += one))
            .is_err(),
            "tampered unsigned-inc chunk final must be rejected"
        );
        assert!(
            verify(&tamper(&|claims| claims
                .stage7
                .hamming_weight_claim_reduction
                .unsigned_inc_msb += one))
            .is_err(),
            "tampered unsigned-inc msb final must be rejected"
        );
    }

    /// The large-trace regime at e2e scale: small traces select K = 16 by
    /// the shared toggle, so this pins the K = 256 arm — preset dispatch,
    /// 8-bit member mapping, and the layout digest — by overriding the
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
        );
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
            <AkitaScheme as VerifierCommitmentScheme>::setup(prover.wjolt_setup_params()).unwrap();
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
    /// over three commitment objects (`W_jolt`, `A_unt`, `A_tru`), with
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

        let trusted_object = commit_trusted_advice_packed(
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
        );
        let io_device = prover.program_io.clone();

        let (object_setup, verifier_setup) =
            <AkitaScheme as VerifierCommitmentScheme>::setup(prover.wjolt_setup_params())
                .expect("the transparent packed setup must derive");
        let trusted_commitment = trusted_object.commitment.clone();
        let proof = prover
            .prove_packed(&object_setup, Some(trusted_object), None)
            .expect("packed prover should produce a verifier-native proof");
        assert!(proof.untrusted_advice_commitment.is_some());
        assert!(proof.stages.reconstruction_sumcheck_proof.is_some());
        // Wjolt is discharged by its native same-point batch. The two advice
        // commitment objects remain in the auxiliary packed opening.
        let auxiliary = proof
            .joint_opening_proof
            .auxiliary
            .as_ref()
            .expect("advice requires an auxiliary opening");
        assert_eq!(auxiliary.openings.len(), 2);
        assert_eq!(auxiliary.evaluations.len(), 2);

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

        // Per-object tampers: a mutated claimed evaluation breaks that
        // object's native opening; a dropped reconstruction proof breaks the
        // fail-closed presence rule. The two advice objects hold the last two
        // per-object evaluations.
        for object in 0..2 {
            let mut tampered = proof.clone();
            tampered
                .joint_opening_proof
                .auxiliary
                .as_mut()
                .unwrap()
                .evaluations[object] += AkitaField::from_u64(1);
            assert!(
                verify(&tampered).is_err(),
                "tampered object-{object} evaluation must be rejected"
            );
        }
        let mut tampered = proof.clone();
        tampered.stages.reconstruction_sumcheck_proof = None;
        assert!(
            verify(&tampered).is_err(),
            "a dropped reconstruction proof must be rejected"
        );
        // The auxiliary joint opening is mandatory whenever auxiliary
        // objects exist: dropping it must break fail-closed.
        let mut tampered = proof.clone();
        tampered.joint_opening_proof.auxiliary = None;
        assert!(
            verify(&tampered).is_err(),
            "a dropped auxiliary opening proof must be rejected"
        );
    }
}

#[cfg(all(test, feature = "host"))]
mod committed_tests {
    use super::*;
    use crate::host;
    use crate::zkvm::program::ProgramPreprocessing;
    use crate::zkvm::prover::JoltProverPreprocessing;
    use serial_test::serial;

    /// The committed-program packed e2e: `W_prog` joins as the second
    /// commitment object (muldiv carries no advice), with tamper rejection
    /// on its claimed evaluation and a reconstruction wire.
    fn committed_e2e(bytecode_chunk_count: usize) {
        DoryGlobals::reset();
        let mut program = host::Program::new("muldiv-guest");
        let (bytecode, init_memory_state, _, e_entry) = program.decode();
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let program_data = ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry)
            .expect("program preprocessing");
        let (shared, prover_data, w_prog) = shared_preprocessing_committed_packed(
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
        );
        let io_device = prover.program_io.clone();

        let (object_setup, verifier_setup) =
            <AkitaScheme as VerifierCommitmentScheme>::setup(prover.wjolt_setup_params())
                .expect("the transparent packed setup must derive");
        let w_prog_commitment = w_prog.commitment.clone();
        let proof = prover
            .prove_packed(&object_setup, None, Some(w_prog))
            .expect("packed prover should produce a verifier-native proof");
        assert!(proof.stages.reconstruction_sumcheck_proof.is_some());
        // Wjolt is discharged by its native same-point batch; Wprog is the
        // only auxiliary packed object.
        let auxiliary = proof
            .joint_opening_proof
            .auxiliary
            .as_ref()
            .expect("committed-program mode requires an auxiliary opening");
        assert_eq!(auxiliary.openings.len(), 1);
        assert_eq!(auxiliary.evaluations.len(), 1);

        let verifier_preprocessing = akita_verifier_preprocessing(
            &prover_preprocessing,
            verifier_setup,
            Some(w_prog_commitment),
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

        // Tampers: the W_prog claimed evaluation (last object) breaks its
        // native opening; a mutated reconstruction wire breaks the batched
        // output check.
        let mut tampered = proof.clone();
        tampered
            .joint_opening_proof
            .auxiliary
            .as_mut()
            .unwrap()
            .evaluations[0] += AkitaField::from_u64(1);
        assert!(
            verify(&tampered).is_err(),
            "tampered W_prog evaluation must be rejected"
        );
        let mut tampered = proof.clone();
        let jolt_verifier::proof::JoltProofClaims::Clear(claims) = &mut tampered.claims else {
            panic!("packed proofs carry clear claims");
        };
        let bytecode_cell = claims
            .reconstruction
            .bytecode
            .as_mut()
            .expect("committed proofs carry the bytecode reconstruction cell");
        bytecode_cell.pc_bytes[0] += AkitaField::from_u64(1);
        assert!(
            verify(&tampered).is_err(),
            "tampered bytecode reconstruction wire must be rejected"
        );
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

        crate::poly::commitment::dory::DoryGlobals::reset();
        let mut program = host::Program::new("sha2-chain-guest");
        let (bytecode, init_memory_state, _, e_entry) = program.decode();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);
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
            &[],
            None,
            None,
            None,
        );
        let io_device = prover.program_io.clone();
        eprintln!("trace length: {}", prover.trace.len());
        let setup_params = prover.wjolt_setup_params();
        eprintln!("Wjolt one-hot K: {}", setup_params.one_hot_k());
        let setup_start = Instant::now();
        let (object_setup, verifier_setup) =
            <AkitaScheme as VerifierCommitmentScheme>::setup(setup_params).unwrap();
        eprintln!("akita setup: {:.2?}", setup_start.elapsed());

        let prove_start = Instant::now();
        let proof = prover
            .prove_packed(&object_setup, None, None)
            .expect("packed prover should produce a verifier-native proof");
        eprintln!("akita prove: {:.2?}", prove_start.elapsed());

        let verifier_preprocessing =
            akita_verifier_preprocessing(&prover_preprocessing, verifier_setup, None);
        let verify_start = Instant::now();
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &verifier_preprocessing,
            &io_device,
            &proof,
            None,
        )
        .expect("packed verifier should accept the packed proof");
        eprintln!("akita verify: {:.2?}", verify_start.elapsed());
    }
}

use jolt_crypto::{Commitment, HomomorphicCommitment, VectorCommitment};
use jolt_field::{CanonicalBytes, Field, FixedByteSize};
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
impl FixedByteSize for NoCommitment {
    const NUM_BYTES: usize = 0;
}

impl CanonicalBytes for NoCommitment {
    fn to_bytes_le(&self, _out: &mut [u8]) {}
}

impl<F: Field> HomomorphicCommitment<F> for NoCommitment {
    fn add(_c1: &Self, _c2: &Self) -> Self {
        Self
    }

    fn linear_combine(_c1: &Self, _c2: &Self, _scalar: &F) -> Self {
        Self
    }
}

impl<F: Field> Commitment for NoVectorCommitment<F> {
    type Output = NoCommitment;
}

impl<F: Field> VectorCommitment for NoVectorCommitment<F> {
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
