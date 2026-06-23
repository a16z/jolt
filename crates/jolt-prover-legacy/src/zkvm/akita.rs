//! Akita prover-side artifact construction.
//!
//! This module is intentionally feature-gated and keeps Akita-specific setup
//! native to the modular verifier/opening crates. The legacy prover still owns
//! trace execution and sumcheck proving; Akita artifacts are derived from those
//! prover-native inputs without going through deleted core/compat paths.

use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use blake2::{digest::consts::U32, Blake2b, Digest};
use common::constants::XLEN;
use jolt_akita::{AkitaCommitment, AkitaField, AkitaProverHint, AkitaScheme, AkitaSetupParams};
use jolt_claims::protocols::jolt::{
    formulas::{dimensions::TracePolynomialOrder, ra::JoltRaPolynomialLayout},
    lattice_packed_validity_digest,
};
use jolt_field::FromPrimitiveInt;
use jolt_openings::{
    CommitmentScheme as OpeningCommitmentScheme, PackingProverSetup, PackingSetupParams,
    PackingVerifierSetup,
};
use jolt_poly::Polynomial;
use jolt_program::preprocess::ProgramMetadata as VerifierProgramMetadata;
use jolt_verifier::{
    akita::{
        commit_akita_packing_jolt_witness, AkitaCommittedPackedJoltWitness, AkitaJoltProof,
        AkitaPackingBatchProof, AkitaPackingJoltWitnessInput, AkitaPackingProverSetup,
        AkitaPackingVerifierSetup, AkitaPrecommittedOpeningInput, AkitaVerifierPreprocessing,
    },
    config::{IncrementCommitmentMode, JoltProtocolConfig, PcsFamily, ProgramMode},
    stages::{
        stage8::{
            derive_lattice_packed_validity_requirements, derive_lattice_packed_witness_layout,
        },
        CommittedProgramSchedule, PrecommittedSchedule,
    },
    CommittedProgramPreprocessing as VerifierCommittedProgramPreprocessing,
    ProgramPreprocessing as VerifierProgramPreprocessing, VerifierError,
};
use tracer::{build_trace_rows, instruction::Cycle};

use crate::{
    curve::{JoltCurve, JoltGroupElement},
    field::{akita::JoltAkitaField, JoltField},
    poly::{
        commitment::commitment_scheme::{
            CommitmentScheme as ProverCommitmentScheme, StreamingCommitmentScheme, ZkEvalCommitment,
        },
        multilinear_polynomial::MultilinearPolynomial,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, small_scalar::SmallScalar},
    zkvm::{
        bytecode::chunks::{
            committed_bytecode_chunk_cycle_len, committed_lanes, for_each_active_lane_value,
            ActiveLaneValue,
        },
        instruction::LookupQuery,
        preprocessing::JoltSharedPreprocessing,
        program::{build_program_image_words_padded, ProgramPreprocessing},
        prover::JoltCpuProver,
        ram::remap_address,
        ProverDebugInfo,
    },
};

pub type AkitaRV64IMACProver<'a, ProofTranscript = AkitaLegacyBlake2bTranscript> =
    JoltCpuProver<'a, JoltAkitaField, AkitaNoCurve, AkitaNoopCommitmentScheme, ProofTranscript>;

pub type AkitaRV64IMACProverPreprocessing = crate::zkvm::prover::JoltProverPreprocessing<
    JoltAkitaField,
    AkitaNoCurve,
    AkitaNoopCommitmentScheme,
>;

type Blake2b256 = Blake2b<U32>;

#[derive(Clone, Debug)]
pub struct AkitaLegacyBlake2bTranscript {
    state: [u8; 32],
    n_rounds: u32,
}

impl Default for AkitaLegacyBlake2bTranscript {
    fn default() -> Self {
        <Self as jolt_transcript::Transcript>::new(b"")
    }
}

impl AkitaLegacyBlake2bTranscript {
    fn hasher(&self) -> Blake2b256 {
        let mut round_bytes = [0u8; 32];
        round_bytes[28..].copy_from_slice(&self.n_rounds.to_be_bytes());
        Blake2b256::new()
            .chain_update(self.state)
            .chain_update(round_bytes)
    }

    fn challenge_bytes(&mut self, out: &mut [u8]) {
        let mut remaining_len = out.len();
        let mut start = 0;
        while remaining_len > 32 {
            let mut chunk = [0u8; 32];
            self.challenge_bytes32(&mut chunk);
            out[start..start + 32].copy_from_slice(&chunk);
            start += 32;
            remaining_len -= 32;
        }

        let mut full_rand = [0u8; 32];
        self.challenge_bytes32(&mut full_rand);
        out[start..start + remaining_len].copy_from_slice(&full_rand[..remaining_len]);
    }

    fn challenge_bytes32(&mut self, out: &mut [u8; 32]) {
        let hash: [u8; 32] = self.hasher().finalize().into();
        out.copy_from_slice(&hash);
        self.update_state(hash);
    }

    fn update_state(&mut self, new_state: [u8; 32]) {
        self.state = new_state;
        self.n_rounds += 1;
    }

    fn challenge_akita_field(&mut self) -> AkitaField {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        JoltAkitaField::from(u128::from_le_bytes(buf)).into_akita()
    }

    fn challenge_akita_scalar(&mut self) -> AkitaField {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        buf.reverse();
        JoltAkitaField::from_bytes(&buf).into_akita()
    }
}

impl jolt_transcript::Transcript for AkitaLegacyBlake2bTranscript {
    type Challenge = AkitaField;

    fn new(label: &'static [u8]) -> Self {
        assert!(
            label.len() <= jolt_transcript::MAX_LABEL_LEN,
            "label must be at most {} bytes",
            jolt_transcript::MAX_LABEL_LEN
        );

        let mut padded = [0u8; 32];
        padded[..label.len()].copy_from_slice(label);
        let state = Blake2b256::new().chain_update(padded).finalize().into();

        Self { state, n_rounds: 0 }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        let hash: [u8; 32] = self.hasher().chain_update(bytes).finalize().into();
        self.update_state(hash);
    }

    fn challenge(&mut self) -> Self::Challenge {
        self.challenge_akita_field()
    }

    fn challenge_scalar(&mut self) -> Self::Challenge {
        self.challenge_akita_scalar()
    }

    fn state(&self) -> [u8; 32] {
        self.state
    }
}

impl crate::transcripts::Transcript for AkitaLegacyBlake2bTranscript {
    fn new(label: &'static [u8]) -> Self {
        <Self as jolt_transcript::Transcript>::new(label)
    }

    #[cfg(test)]
    fn compare_to(&mut self, _other: Self) {}

    fn raw_append_label(&mut self, label: &'static [u8]) {
        assert!(label.len() <= 32);
        let mut padded = [0u8; 32];
        padded[..label.len()].copy_from_slice(label);
        jolt_transcript::Transcript::append_bytes(self, &padded);
    }

    fn raw_append_bytes(&mut self, bytes: &[u8]) {
        jolt_transcript::Transcript::append_bytes(self, bytes);
    }

    fn raw_append_u64(&mut self, x: u64) {
        let mut packed = [0_u8; 32];
        packed[24..].copy_from_slice(&x.to_be_bytes());
        jolt_transcript::Transcript::append_bytes(self, &packed);
    }

    fn raw_append_scalar<F: JoltField>(&mut self, scalar: &F) {
        let mut buf = vec![];
        scalar.serialize_uncompressed(&mut buf).unwrap();
        buf.reverse();
        jolt_transcript::Transcript::append_bytes(self, &buf);
    }

    fn challenge_u128(&mut self) -> u128 {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        u128::from_le_bytes(buf)
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        self.challenge_scalar_128_bits()
    }

    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        buf.reverse();
        F::from_bytes(&buf)
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        (0..len).map(|_| self.challenge_scalar()).collect()
    }

    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let q: F = self.challenge_scalar();
        let mut q_powers = vec![F::one(); len];
        for i in 1..len {
            q_powers[i] = q_powers[i - 1] * q;
        }
        q_powers
    }

    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge {
        F::Challenge::from(self.challenge_u128())
    }

    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge> {
        (0..len)
            .map(|_| self.challenge_scalar_optimized::<F>())
            .collect()
    }

    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let q: F::Challenge = self.challenge_scalar_optimized::<F>();
        let mut q_powers = vec![F::one(); len];
        for i in 1..len {
            q_powers[i] = q * q_powers[i - 1];
        }
        q_powers
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaNoGroupElement;

impl Add for AkitaNoGroupElement {
    type Output = Self;

    fn add(self, _rhs: Self) -> Self::Output {
        Self
    }
}

impl Add<&AkitaNoGroupElement> for AkitaNoGroupElement {
    type Output = Self;

    fn add(self, _rhs: &AkitaNoGroupElement) -> Self::Output {
        Self
    }
}

impl AddAssign for AkitaNoGroupElement {
    fn add_assign(&mut self, _rhs: Self) {}
}

impl Sub for AkitaNoGroupElement {
    type Output = Self;

    fn sub(self, _rhs: Self) -> Self::Output {
        Self
    }
}

impl Sub<&AkitaNoGroupElement> for AkitaNoGroupElement {
    type Output = Self;

    fn sub(self, _rhs: &AkitaNoGroupElement) -> Self::Output {
        Self
    }
}

impl SubAssign for AkitaNoGroupElement {
    fn sub_assign(&mut self, _rhs: Self) {}
}

impl Neg for AkitaNoGroupElement {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self
    }
}

impl JoltGroupElement for AkitaNoGroupElement {
    type Scalar = JoltAkitaField;

    fn zero() -> Self {
        Self
    }

    fn is_zero(&self) -> bool {
        true
    }

    fn double(&self) -> Self {
        Self
    }

    fn scalar_mul(&self, _scalar: &Self::Scalar) -> Self {
        Self
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AkitaNoCurve;

impl JoltCurve for AkitaNoCurve {
    type F = JoltAkitaField;
    type G1 = AkitaNoGroupElement;
    type G2 = AkitaNoGroupElement;
    type G1Affine = AkitaNoGroupElement;
    type GT = AkitaNoGroupElement;

    fn g1_generator() -> Self::G1 {
        AkitaNoGroupElement
    }

    fn g2_generator() -> Self::G2 {
        AkitaNoGroupElement
    }

    fn g1_to_affine(point: &Self::G1) -> Self::G1Affine {
        *point
    }

    fn pairing(_g1: &Self::G1, _g2: &Self::G2) -> Self::GT {
        AkitaNoGroupElement
    }

    fn multi_pairing(_g1s: &[Self::G1], _g2s: &[Self::G2]) -> Self::GT {
        AkitaNoGroupElement
    }

    fn g1_msm(_bases: &[Self::G1], _scalars: &[Self::F]) -> Self::G1 {
        AkitaNoGroupElement
    }

    fn g1_affine_msm(_bases: &[Self::G1Affine], _scalars: &[Self::F]) -> Self::G1 {
        AkitaNoGroupElement
    }

    fn g2_msm(_bases: &[Self::G2], _scalars: &[Self::F]) -> Self::G2 {
        AkitaNoGroupElement
    }

    fn random_g1<R: rand_core::RngCore>(_rng: &mut R) -> Self::G1 {
        AkitaNoGroupElement
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaNoopCommitmentScheme;

#[derive(Clone, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaNoopSetup {
    pub max_num_vars: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaNoopCommitment;

#[derive(Clone, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaNoopProof;

#[derive(Clone, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct AkitaNoopOpeningProofHint;

impl ProverCommitmentScheme for AkitaNoopCommitmentScheme {
    type Field = JoltAkitaField;
    type ProverSetup = AkitaNoopSetup;
    type VerifierSetup = AkitaNoopSetup;
    type Commitment = AkitaNoopCommitment;
    type Proof = AkitaNoopProof;
    type BatchedProof = Vec<AkitaNoopProof>;
    type OpeningProofHint = AkitaNoopOpeningProofHint;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        AkitaNoopSetup { max_num_vars }
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup.clone()
    }

    fn commit(
        _poly: &MultilinearPolynomial<Self::Field>,
        _setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        (AkitaNoopCommitment, AkitaNoopOpeningProofHint)
    }

    fn batch_commit<U>(
        polys: &[U],
        _gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        polys
            .iter()
            .map(|_| (AkitaNoopCommitment, AkitaNoopOpeningProofHint))
            .collect()
    }

    fn combine_commitments<C: std::borrow::Borrow<Self::Commitment>>(
        _commitments: &[C],
        _coeffs: &[Self::Field],
    ) -> Self::Commitment {
        AkitaNoopCommitment
    }

    fn combine_hints(
        _hints: Vec<Self::OpeningProofHint>,
        _coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        AkitaNoopOpeningProofHint
    }

    fn prove<ProofTranscript: Transcript>(
        _setup: &Self::ProverSetup,
        _poly: &MultilinearPolynomial<Self::Field>,
        _opening_point: &[<Self::Field as JoltField>::Challenge],
        _hint: Option<Self::OpeningProofHint>,
        _transcript: &mut ProofTranscript,
    ) -> (Self::Proof, Option<Self::Field>) {
        (AkitaNoopProof, None)
    }

    fn verify<ProofTranscript: Transcript>(
        _proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        _opening_point: &[<Self::Field as JoltField>::Challenge],
        _opening: &Self::Field,
        _commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"akita-noop"
    }
}

impl StreamingCommitmentScheme for AkitaNoopCommitmentScheme {
    type ChunkState = ();

    fn process_chunk<T: SmallScalar>(_setup: &Self::ProverSetup, _chunk: &[T]) -> Self::ChunkState {
    }

    fn process_chunk_onehot(
        _setup: &Self::ProverSetup,
        _onehot_k: usize,
        _chunk: &[Option<usize>],
    ) -> Self::ChunkState {
    }

    fn aggregate_chunks(
        _setup: &Self::ProverSetup,
        _onehot_k: Option<usize>,
        _tier1_commitments: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        (AkitaNoopCommitment, AkitaNoopOpeningProofHint)
    }
}

impl ZkEvalCommitment<AkitaNoCurve> for AkitaNoopCommitmentScheme {
    fn eval_commitment(_proof: &Self::Proof) -> Option<AkitaNoGroupElement> {
        None
    }

    fn eval_commitment_gens(
        _setup: &Self::ProverSetup,
    ) -> Option<(AkitaNoGroupElement, AkitaNoGroupElement)> {
        None
    }

    fn eval_commitment_gens_verifier(
        _setup: &Self::VerifierSetup,
    ) -> Option<(AkitaNoGroupElement, AkitaNoGroupElement)> {
        None
    }
}

impl
    JoltCpuProver<
        '_,
        JoltAkitaField,
        AkitaNoCurve,
        AkitaNoopCommitmentScheme,
        AkitaLegacyBlake2bTranscript,
    >
{
    #[expect(
        clippy::type_complexity,
        reason = "public prover API returns verifier preprocessing, proof, and optional debug data"
    )]
    pub fn prove_akita(
        self,
    ) -> Result<
        (
            AkitaVerifierPreprocessing,
            AkitaJoltProof,
            Option<
                ProverDebugInfo<
                    JoltAkitaField,
                    AkitaLegacyBlake2bTranscript,
                    AkitaNoopCommitmentScheme,
                >,
            >,
        ),
        VerifierError,
    > {
        let public_io = self.program_io.clone();
        let has_untrusted_advice = !self.program_io.untrusted_advice.is_empty();
        let packed_witness = self.commit_akita_packed_witness()?;
        let precommitted = self.commit_akita_precommitted_program(&packed_witness)?;
        let packed_commitment = packed_witness_commitment(&packed_witness.committed.artifacts)?;
        let untrusted_advice_commitment = has_untrusted_advice.then(|| packed_commitment.clone());
        let placeholder_opening = placeholder_akita_batch_proof(packed_commitment);
        let (proof_parts, debug_info) = self.prove_parts_with_akita(&packed_witness);
        let mut proof = crate::zkvm::proof::akita_proof_parts_into_verifier(
            proof_parts,
            packed_witness.committed.artifacts.protocol,
            packed_witness.committed.artifacts.commitments.clone(),
            placeholder_opening,
            untrusted_advice_commitment,
        )?;
        let opening_inputs = precommitted.opening_inputs();
        jolt_verifier::akita::prove_and_attach_akita_opening_proofs_with_precommitted::<
            AkitaLegacyBlake2bTranscript,
            _,
        >(
            &packed_witness.prover_setup,
            &precommitted.preprocessing,
            &public_io,
            &mut proof,
            None,
            &packed_witness.committed.artifacts,
            &packed_witness.committed.witness,
            &opening_inputs,
        )?;

        Ok((precommitted.preprocessing, proof, debug_info))
    }
}

fn packed_witness_commitment(
    artifacts: &jolt_verifier::akita::AkitaPackingWitnessArtifacts,
) -> Result<AkitaCommitment, VerifierError> {
    artifacts
        .payload()
        .map(|payload| payload.packed_witness.clone())
        .ok_or_else(|| invalid_akita_prover_config("Akita artifacts must carry lattice payload"))
}

fn placeholder_akita_batch_proof(commitment: AkitaCommitment) -> AkitaPackingBatchProof {
    jolt_openings::PackingBatchProof {
        reduction: None,
        native: jolt_akita::AkitaBatchProof {
            commitment,
            statement_bridge: Vec::new(),
            proof_shape: Vec::new(),
            proof: Vec::new(),
        },
    }
}

#[derive(Clone, Debug)]
pub struct AkitaPackedWitnessProverData {
    pub protocol: JoltProtocolConfig,
    pub precommitted: PrecommittedSchedule,
    pub prover_setup: AkitaPackingProverSetup,
    pub verifier_setup: AkitaPackingVerifierSetup,
    pub committed: AkitaCommittedPackedJoltWitness,
}

#[derive(Clone)]
pub struct AkitaPrecommittedProgramProverData {
    pub preprocessing: AkitaVerifierPreprocessing,
    pub opening_inputs: Vec<AkitaOwnedPrecommittedOpening>,
}

impl AkitaPrecommittedProgramProverData {
    pub fn opening_inputs(&self) -> Vec<AkitaPrecommittedOpeningInput<'_>> {
        self.opening_inputs
            .iter()
            .map(AkitaOwnedPrecommittedOpening::as_input)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct AkitaOwnedPrecommittedOpening {
    pub commitment: AkitaCommitment,
    pub polynomials: Vec<Polynomial<AkitaField>>,
    pub hint: AkitaProverHint,
}

impl AkitaOwnedPrecommittedOpening {
    pub fn as_input(&self) -> AkitaPrecommittedOpeningInput<'_> {
        AkitaPrecommittedOpeningInput {
            polynomials: self.polynomials.as_slice(),
            hint: &self.hint,
        }
    }
}

impl<F, C, PCS, ProofTranscript> JoltCpuProver<'_, F, C, PCS, ProofTranscript>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: StreamingCommitmentScheme<Field = F> + ZkEvalCommitment<C>,
    ProofTranscript: Transcript,
{
    pub fn commit_akita_packed_witness(
        &self,
    ) -> Result<AkitaPackedWitnessProverData, VerifierError> {
        if !self.program_io.trusted_advice.is_empty()
            || self.advice.trusted_advice_commitment.is_some()
        {
            return Err(invalid_akita_prover_config(
                "Akita packed witness construction does not support trusted advice yet",
            ));
        }

        let precommitted = derive_precommitted_schedule(
            &self.preprocessing.shared,
            self.trace.len().ilog2() as usize,
            self.one_hot_params.log_k_chunk,
            !self.program_io.untrusted_advice.is_empty(),
        )?;
        let layout = derive_akita_packed_layout(
            &self.preprocessing.shared,
            self.trace.len().ilog2() as usize,
            self.one_hot_params.log_k_chunk,
            self.one_hot_params.instruction_d,
            self.one_hot_params.bytecode_d,
            self.one_hot_params.ram_d,
            &precommitted,
        )?;
        let max_num_vars = layout
            .dimension
            .max(akita_precommitted_program_max_num_vars(
                &self.preprocessing.shared,
            )?);
        let max_num_polys_per_commitment_group =
            require_committed_program(&self.preprocessing.shared.program)?
                .bytecode_commitments
                .bytecode_chunk_count
                .max(1);
        let (prover_setup, verifier_setup) = akita_packing_setup_with_max_num_vars(
            &layout,
            max_num_vars,
            max_num_polys_per_commitment_group,
        );
        let trace_rows = build_trace_rows(
            &self.trace,
            &self.preprocessing.materialized_program().bytecode,
        )
        .map_err(|error| {
            invalid_akita_prover_config(format!("failed to build Akita trace rows: {error}"))
        })?;
        let instruction_lookup_indices = instruction_lookup_indices(&self.trace);
        let remapped_ram_addresses = self
            .trace
            .iter()
            .map(|cycle| {
                remap_address(
                    cycle.ram_access().address() as u64,
                    &self.preprocessing.shared.memory_layout,
                )
            })
            .collect::<Vec<_>>();
        let committed = commit_akita_packing_jolt_witness(
            &prover_setup,
            AkitaPackingJoltWitnessInput {
                layout,
                trace_rows: &trace_rows,
                log_k_chunk: self.one_hot_params.log_k_chunk,
                instruction_lookup_indices: &instruction_lookup_indices,
                remapped_ram_addresses: Some(&remapped_ram_addresses),
                untrusted_advice: (!self.program_io.untrusted_advice.is_empty())
                    .then_some(self.program_io.untrusted_advice.as_slice()),
            },
        )?;

        Ok(AkitaPackedWitnessProverData {
            protocol: committed.artifacts.protocol,
            precommitted,
            prover_setup,
            verifier_setup,
            committed,
        })
    }

    pub fn commit_akita_precommitted_program(
        &self,
        packed_witness: &AkitaPackedWitnessProverData,
    ) -> Result<AkitaPrecommittedProgramProverData, VerifierError>
    where
        PCS::Commitment: ark_serialize::CanonicalSerialize,
    {
        if packed_witness.precommitted.bytecode.is_none()
            || packed_witness.precommitted.program_image.is_none()
        {
            return Err(invalid_akita_prover_config(
                "Akita committed-program openings require bytecode and program-image schedules",
            ));
        }

        let shared = &self.preprocessing.shared;
        let committed = require_committed_program(&shared.program)?;
        let program = self.preprocessing.materialized_program();
        let bytecode_chunk_count = committed.bytecode_commitments.bytecode_chunk_count;
        let bytecode_opening = commit_akita_precommitted_polynomial_group(
            build_akita_bytecode_chunk_polynomials(
                &program.bytecode.bytecode,
                bytecode_chunk_count,
            ),
            &packed_witness.prover_setup.pcs,
        )?;
        let bytecode_chunk_commitments =
            vec![bytecode_opening.commitment.clone(); bytecode_chunk_count];
        let mut opening_inputs = Vec::with_capacity(2);
        opening_inputs.push(bytecode_opening);

        let program_image_polynomial = build_akita_program_image_polynomial(
            program,
            committed.program_commitments.program_image_num_words,
        );
        let program_image_opening = commit_akita_precommitted_polynomial_group(
            vec![program_image_polynomial],
            &packed_witness.prover_setup.pcs,
        )?;
        let program_image_commitment = program_image_opening.commitment.clone();
        opening_inputs.push(program_image_opening);

        let verifier_program =
            VerifierProgramPreprocessing::Committed(VerifierCommittedProgramPreprocessing {
                meta: verifier_program_metadata(&committed.meta),
                memory_layout: shared.memory_layout.clone(),
                max_padded_trace_length: shared.max_padded_trace_length,
                bytecode_chunk_commitments,
                program_image_commitment,
            });
        let preprocessing = AkitaVerifierPreprocessing::new(
            verifier_program,
            shared.digest(),
            packed_witness.verifier_setup.clone(),
            None,
        );

        Ok(AkitaPrecommittedProgramProverData {
            preprocessing,
            opening_inputs,
        })
    }
}

pub fn derive_akita_packed_layout<PCS>(
    shared: &JoltSharedPreprocessing<PCS>,
    log_t: usize,
    log_k_chunk: usize,
    instruction_ra_count: usize,
    bytecode_ra_count: usize,
    ram_ra_count: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<jolt_openings::PackingWitnessLayout, VerifierError>
where
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme,
{
    require_committed_program(&shared.program)?;
    let config = lattice_layout_derivation_config(log_k_chunk, precommitted)?;
    let ra_layout =
        JoltRaPolynomialLayout::new(instruction_ra_count, bytecode_ra_count, ram_ra_count)
            .map_err(|error| invalid_akita_prover_config(error.to_string()))?;
    derive_lattice_packed_witness_layout(&config, log_t, log_k_chunk, ra_layout, precommitted)
}

pub fn derive_precommitted_schedule<PCS>(
    shared: &JoltSharedPreprocessing<PCS>,
    log_t: usize,
    log_k_chunk: usize,
    include_untrusted_advice: bool,
) -> Result<PrecommittedSchedule, VerifierError>
where
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme,
{
    let committed = committed_program_schedule(shared)?;
    PrecommittedSchedule::new(
        TracePolynomialOrder::CycleMajor,
        log_t,
        log_k_chunk,
        None,
        include_untrusted_advice.then_some(shared.memory_layout.max_untrusted_advice_size as usize),
        Some(committed),
    )
    .map_err(|error| VerifierError::InvalidPrecommittedSchedule {
        reason: error.to_string(),
    })
}

pub fn akita_packing_setup(
    layout: &jolt_openings::PackingWitnessLayout,
    max_num_polys_per_commitment_group: usize,
) -> (AkitaPackingProverSetup, AkitaPackingVerifierSetup) {
    akita_packing_setup_with_max_num_vars(
        layout,
        layout.dimension,
        max_num_polys_per_commitment_group,
    )
}

pub fn akita_packing_setup_with_max_num_vars(
    layout: &jolt_openings::PackingWitnessLayout,
    max_num_vars: usize,
    max_num_polys_per_commitment_group: usize,
) -> (AkitaPackingProverSetup, AkitaPackingVerifierSetup) {
    assert!(
        max_num_vars >= layout.dimension,
        "Akita setup max_num_vars ({max_num_vars}) must cover packed layout dimension ({})",
        layout.dimension
    );
    let params = PackingSetupParams {
        pcs: AkitaSetupParams::new(
            max_num_vars,
            max_num_polys_per_commitment_group,
            layout.digest,
        ),
        layout: layout.clone(),
    };
    let (pcs, verifier_pcs) = AkitaScheme::setup(params.pcs);
    (
        PackingProverSetup {
            pcs,
            layout: params.layout.clone(),
        },
        PackingVerifierSetup {
            pcs: verifier_pcs,
            layout: params.layout,
        },
    )
}

fn akita_precommitted_program_max_num_vars<PCS>(
    shared: &JoltSharedPreprocessing<PCS>,
) -> Result<usize, VerifierError>
where
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme,
{
    let committed = require_committed_program(&shared.program)?;
    let chunk_len = committed_lanes()
        * committed_bytecode_chunk_cycle_len(
            committed.meta.bytecode_len,
            committed.bytecode_commitments.bytecode_chunk_count,
        );
    let program_image_len = committed.program_commitments.program_image_num_words;
    Ok(akita_polynomial_num_vars(chunk_len)?.max(akita_polynomial_num_vars(program_image_len)?))
}

fn akita_polynomial_num_vars(len: usize) -> Result<usize, VerifierError> {
    if len == 0 || !len.is_power_of_two() {
        return Err(invalid_akita_prover_config(format!(
            "Akita precommitted polynomial length must be a non-zero power of two, got {len}"
        )));
    }
    Ok(len.ilog2() as usize)
}

fn committed_program_schedule<PCS>(
    shared: &JoltSharedPreprocessing<PCS>,
) -> Result<CommittedProgramSchedule, VerifierError>
where
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme,
{
    let committed = require_committed_program(&shared.program)?;
    let program_image_start_index = shared
        .memory_layout
        .remapped_word_address(committed.meta.min_bytecode_address)
        .map_err(|error| VerifierError::InvalidCommittedProgram {
            reason: error.to_string(),
        })?;

    Ok(CommittedProgramSchedule {
        bytecode_len: committed.meta.bytecode_len,
        bytecode_chunk_count: committed.bytecode_commitments.bytecode_chunk_count,
        program_image_len_words: committed.meta.program_image_len_words,
        program_image_start_index: program_image_start_index as usize,
    })
}

fn require_committed_program<PCS>(
    program: &ProgramPreprocessing<PCS>,
) -> Result<&crate::zkvm::program::CommittedProgramPreprocessing<PCS>, VerifierError>
where
    PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme,
{
    match program {
        ProgramPreprocessing::Committed(committed) => Ok(committed),
        ProgramPreprocessing::Full(_) => Err(invalid_akita_prover_config(
            "Akita lattice mode requires committed program preprocessing",
        )),
    }
}

fn lattice_layout_derivation_config(
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<JoltProtocolConfig, VerifierError> {
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice.program_mode = ProgramMode::Committed;
    config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
    config.lattice.advice.untrusted = precommitted.untrusted_advice.is_some();
    // Layout derivation only needs the lattice feature switches, but verifier
    // config validation also requires concrete packed-witness bindings.
    config.lattice.packed_witness.layout_digest = Some([0; 32]);
    config.lattice.packed_witness.d_pack = Some(0);
    config.lattice.packed_witness.validity_digest = Some([0; 32]);
    config.lattice.packed_witness.validity_digest = Some(lattice_packed_validity_digest(
        &derive_lattice_packed_validity_requirements(&config, log_k_chunk, precommitted)?,
    ));
    Ok(config)
}

fn instruction_lookup_indices(trace: &[Cycle]) -> Vec<u128> {
    trace
        .iter()
        .map(LookupQuery::<XLEN>::to_lookup_index)
        .collect()
}

fn build_akita_bytecode_chunk_polynomials(
    instructions: &[jolt_riscv::JoltInstructionRow],
    chunk_count: usize,
) -> Vec<Polynomial<AkitaField>> {
    let bytecode_len = instructions.len();
    let chunk_cycle_len = committed_bytecode_chunk_cycle_len(bytecode_len, chunk_count);
    let lane_capacity = committed_lanes();
    let mut chunks = (0..chunk_count)
        .map(|_| vec![AkitaField::default(); lane_capacity * chunk_cycle_len])
        .collect::<Vec<_>>();

    for (cycle, instruction) in instructions.iter().enumerate() {
        let chunk_index = cycle / chunk_cycle_len;
        let chunk_cycle = cycle % chunk_cycle_len;
        let coeffs = &mut chunks[chunk_index];
        for_each_active_lane_value::<JoltAkitaField>(instruction, |global_lane, lane_value| {
            let index = TracePolynomialOrder::CycleMajor.address_cycle_to_index(
                global_lane,
                chunk_cycle,
                lane_capacity,
                chunk_cycle_len,
            );
            let value = match lane_value {
                ActiveLaneValue::One => <AkitaField as FromPrimitiveInt>::from_u64(1),
                ActiveLaneValue::Scalar(value) => value.into_akita(),
            };
            coeffs[index] += value;
        });
    }

    chunks.into_iter().map(Polynomial::new).collect()
}

fn build_akita_program_image_polynomial(
    program: &crate::zkvm::program::FullProgramPreprocessing,
    padded_len: usize,
) -> Polynomial<AkitaField> {
    Polynomial::new(
        build_program_image_words_padded(program, padded_len)
            .into_iter()
            .map(<AkitaField as FromPrimitiveInt>::from_u64)
            .collect(),
    )
}

fn commit_akita_precommitted_polynomial_group(
    polynomials: Vec<Polynomial<AkitaField>>,
    setup: &jolt_akita::AkitaProverSetup,
) -> Result<AkitaOwnedPrecommittedOpening, VerifierError> {
    let (commitment, hint) =
        AkitaScheme::commit_group(setup, setup.default_layout_digest, &polynomials).map_err(
            |error| VerifierError::FinalOpeningBatchFailed {
                reason: error.to_string(),
            },
        )?;
    Ok(AkitaOwnedPrecommittedOpening {
        commitment,
        polynomials,
        hint,
    })
}

fn verifier_program_metadata(
    meta: &crate::zkvm::program::ProgramMetadata,
) -> VerifierProgramMetadata {
    VerifierProgramMetadata {
        entry_address: meta.entry_address,
        min_bytecode_address: meta.min_bytecode_address,
        entry_bytecode_index: meta.entry_bytecode_index,
        program_image_len_words: meta.program_image_len_words,
        bytecode_len: meta.bytecode_len,
    }
}

fn invalid_akita_prover_config(reason: impl Into<String>) -> VerifierError {
    VerifierError::InvalidProtocolConfig {
        reason: reason.into(),
    }
}

#[cfg(all(test, feature = "host"))]
mod tests {
    #![expect(
        clippy::expect_used,
        clippy::unwrap_used,
        reason = "tests assert successful prover artifact construction"
    )]

    use serial_test::serial;

    use super::*;
    use crate::{
        host,
        poly::commitment::dory::DoryGlobals,
        zkvm::{
            preprocessing::JoltSharedPreprocessing, program::ProgramPreprocessing,
            prover::JoltProverPreprocessing, RV64IMACProver,
        },
    };
    use jolt_verifier::JoltProofClaims;

    fn muldiv_program_inputs() -> (host::Program, Vec<u8>) {
        let program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
        (program, inputs)
    }

    #[test]
    #[serial]
    fn muldiv_committed_program_builds_akita_packed_witness() {
        DoryGlobals::reset();
        let (mut program, inputs) = muldiv_program_inputs();
        let (bytecode, init_memory_state, _, entry_address) = program.decode();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);
        let program_preprocessing =
            ProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address).unwrap();
        let (shared, committed_program_prover_data, generators) =
            JoltSharedPreprocessing::new_committed(
                program_preprocessing,
                io_device.memory_layout,
                1 << 16,
                1,
            );
        let prover_preprocessing = JoltProverPreprocessing::new_committed(
            shared,
            committed_program_prover_data,
            generators,
        );
        let elf_contents = program
            .get_elf_contents()
            .expect("test program should provide ELF contents");
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        );

        let data = prover
            .commit_akita_packed_witness()
            .expect("Akita packed witness should commit");
        let payload = data
            .committed
            .artifacts
            .payload()
            .expect("Akita artifacts should carry a lattice payload");

        assert_eq!(data.protocol.pcs, PcsFamily::Lattice);
        assert_eq!(
            payload.layout_digest,
            data.committed.artifacts.layout.digest
        );
        assert_eq!(payload.d_pack, data.committed.artifacts.layout.dimension);
        assert!(data.precommitted.bytecode.is_some());
        assert!(data.precommitted.program_image.is_some());

        let precommitted = prover
            .commit_akita_precommitted_program(&data)
            .expect("Akita precommitted program should commit");
        let verifier_committed = precommitted
            .preprocessing
            .program
            .committed()
            .expect("Akita preprocessing should use committed program mode");

        assert_eq!(
            precommitted.preprocessing.pcs_setup.layout.dimension,
            payload.d_pack
        );
        assert!(
            precommitted.preprocessing.pcs_setup.pcs.max_num_vars >= payload.d_pack,
            "Akita setup should cover packed and precommitted opening dimensions"
        );
        assert_eq!(
            verifier_committed.bytecode_chunk_commitments.len(),
            prover.preprocessing.shared.bytecode_chunk_count
        );
        assert_eq!(precommitted.opening_inputs.len(), 2);
        let bytecode_opening = precommitted
            .opening_inputs
            .first()
            .expect("bytecode opening should be first");
        assert_eq!(
            bytecode_opening.polynomials.len(),
            prover.preprocessing.shared.bytecode_chunk_count
        );
        assert!(bytecode_opening
            .hint
            .matches_commitment(&bytecode_opening.commitment));
        for commitment in &verifier_committed.bytecode_chunk_commitments {
            assert_eq!(&bytecode_opening.commitment, commitment);
        }
        let program_image_opening = precommitted
            .opening_inputs
            .last()
            .expect("program image opening should follow bytecode chunks");
        assert_eq!(program_image_opening.polynomials.len(), 1);
        assert_eq!(
            program_image_opening.commitment,
            verifier_committed.program_image_commitment
        );
        assert!(program_image_opening
            .hint
            .matches_commitment(&verifier_committed.program_image_commitment));
        let borrowed_inputs = precommitted.opening_inputs();
        assert_eq!(borrowed_inputs.len(), precommitted.opening_inputs.len());
    }

    #[test]
    #[serial]
    fn muldiv_committed_program_proves_and_verifies_akita() {
        DoryGlobals::reset();
        let (mut program, inputs) = muldiv_program_inputs();
        let (bytecode, init_memory_state, _, entry_address) = program.decode();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);
        let program_preprocessing: ProgramPreprocessing<AkitaNoopCommitmentScheme> =
            ProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address).unwrap();
        let (shared, committed_program_prover_data, generators) =
            JoltSharedPreprocessing::new_committed(
                program_preprocessing,
                io_device.memory_layout.clone(),
                1 << 16,
                1,
            );
        let prover_preprocessing = JoltProverPreprocessing::<
            JoltAkitaField,
            AkitaNoCurve,
            AkitaNoopCommitmentScheme,
        >::new_committed(
            shared, committed_program_prover_data, generators
        );
        let elf_contents = program
            .get_elf_contents()
            .expect("test program should provide ELF contents");
        let prover = AkitaRV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        );

        let (verifier_preprocessing, proof, _) = prover
            .prove_akita()
            .expect("Akita proof should be produced");
        jolt_verifier::akita::verify_akita_clear::<AkitaLegacyBlake2bTranscript>(
            &verifier_preprocessing,
            &io_device,
            &proof,
            None,
            &proof.protocol,
        )
        .expect("Akita proof should verify");

        let mut tampered_claim = proof.clone();
        let JoltProofClaims::Clear(claims) = &mut tampered_claim.claims else {
            panic!("Akita e2e proof should be clear");
        };
        claims.stage7.hamming_weight_claim_reduction.ram_ra[0] += AkitaField::one();
        jolt_verifier::akita::verify_akita_clear::<AkitaLegacyBlake2bTranscript>(
            &verifier_preprocessing,
            &io_device,
            &tampered_claim,
            None,
            &tampered_claim.protocol,
        )
        .expect_err("tampered prover-produced Akita opening claim should reject");

        let mut tampered_opening = proof.clone();
        let first_byte = tampered_opening
            .joint_opening_proof
            .native
            .proof
            .first_mut()
            .expect("prover-produced Akita final opening proof should contain proof bytes");
        *first_byte ^= 1;
        jolt_verifier::akita::verify_akita_clear::<AkitaLegacyBlake2bTranscript>(
            &verifier_preprocessing,
            &io_device,
            &tampered_opening,
            None,
            &tampered_opening.protocol,
        )
        .expect_err("tampered prover-produced Akita final opening proof should reject");
    }
}
