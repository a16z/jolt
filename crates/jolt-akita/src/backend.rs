use std::collections::BTreeSet;
use std::io::Cursor;

use akita_pcs::{
    AkitaDeserialize, AkitaSerialize, AkitaTranscript, CommitmentProver, ComputeBackendSetup,
    CpuBackend,
};
use akita_prover::{AkitaPolyOps, CommittedPolynomials, ProverClaims};
use akita_transcript::Transcript as AkitaNativeTranscript;
use akita_types::{
    BasisMode, CommitmentVerifier, CommittedOpenings, SetupContributionMode, VerifierClaims,
};
use jolt_crypto::Commitment;
use jolt_field::{CanonicalBytes, FixedByteSize};
use jolt_openings::{
    BatchOpeningResult, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError,
    PhysicalView, ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use serde::{Deserialize, Serialize};

use crate::layout::PackedWitnessSource;
use crate::types::{
    append_field_slice, AkitaBatchProof, AkitaCommitInput, AkitaCommitment, AkitaField,
    AkitaHidingCommitment, AkitaProverHint, AkitaProverSetup, AkitaSetupParams, AkitaVerifierSetup,
    NativeCommitment, NativeDensePoly, NativeHint, NativeProof, NativeProofShape, NativeScheme,
    NativeSparsePoly, NativeVerifier, AKITA_D, LAYERZERO_AKITA_REV,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaScheme;

impl AkitaScheme {
    pub fn commit_group(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        polynomials: &[Polynomial<AkitaField>],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        let num_vars = validate_commit_polynomials(setup, polynomials)?;
        let dense = dense_polynomials(polynomials)?;
        let dense_refs = dense.iter().collect::<Vec<_>>();
        commit_native_group(
            setup,
            layout_digest,
            num_vars,
            polynomials.len(),
            dense_refs.as_slice(),
        )
    }

    fn commit_packed_sparse_witness(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        polynomial: &NativeSparsePoly,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        commit_native_group(
            setup,
            layout_digest,
            polynomial.num_vars(),
            1,
            &[polynomial],
        )
    }

    fn commit_packed_dense_source(
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
        polynomial: Polynomial<AkitaField>,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        Self::commit_packed_witness(
            setup,
            AkitaCommitInput {
                layout_digest,
                polynomial,
            },
        )
    }

    pub fn commit_packed_witness(
        setup: &AkitaProverSetup,
        input: AkitaCommitInput,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        validate_setup_layout_digest(setup.default_layout_digest, input.layout_digest)?;
        Self::commit_group(setup, input.layout_digest, &[input.polynomial])
    }

    pub fn commit_packed_source<S>(
        setup: &AkitaProverSetup,
        source: &S,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError>
    where
        S: PackedWitnessSource<AkitaField>,
    {
        let layout = source.layout();
        if layout.dimension > setup.max_num_vars {
            return Err(OpeningsError::PolynomialTooLarge {
                poly_size: layout.dimension,
                setup_max: setup.max_num_vars,
            });
        }
        if let Some(polynomial) = packed_source_sparse_polynomial(source)? {
            return Self::commit_packed_sparse_witness(setup, layout.digest, &polynomial);
        }
        let polynomial = packed_source_polynomial(source)?;
        Self::commit_packed_dense_source(setup, layout.digest, polynomial)
    }
}

fn commit_native_group<P>(
    setup: &AkitaProverSetup,
    layout_digest: [u8; 32],
    num_vars: usize,
    poly_count: usize,
    polynomials: &[P],
) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError>
where
    P: AkitaPolyOps<AkitaField, AKITA_D>,
{
    validate_native_commit_shape(setup, num_vars, poly_count)?;
    if polynomials.len() != poly_count {
        return Err(invalid_batch(format!(
            "Akita native commit received {} polynomials for {poly_count} commitment slots",
            polynomials.len()
        )));
    }
    for polynomial in polynomials {
        if polynomial.num_vars() != num_vars {
            return Err(invalid_batch(format!(
                "Akita native commit mixes {}-variable and {num_vars}-variable polynomials",
                polynomial.num_vars()
            )));
        }
    }

    let (native_commitment, native_hint) =
        NativeScheme::commit(&setup.native, &CpuBackend, &setup.prepared, polynomials)
            .map_err(akita_error)?;
    let commitment = AkitaCommitment {
        layout_digest,
        num_vars,
        poly_count,
        native: serialize_akita(&native_commitment)?,
    };
    Ok((
        commitment.clone(),
        AkitaProverHint {
            commitment,
            native: Some(native_hint),
        },
    ))
}

fn validate_native_commit_shape(
    setup: &AkitaProverSetup,
    num_vars: usize,
    poly_count: usize,
) -> Result<(), OpeningsError> {
    if num_vars > setup.max_num_vars {
        return Err(OpeningsError::PolynomialTooLarge {
            poly_size: num_vars,
            setup_max: setup.max_num_vars,
        });
    }
    if num_vars != setup.max_num_vars {
        return Err(invalid_batch(format!(
            "Akita commitment dimension {num_vars} does not match exact setup dimension {}",
            setup.max_num_vars
        )));
    }
    if poly_count > setup.max_num_polys_per_commitment_group {
        return Err(invalid_batch(format!(
            "Akita commitment group has {poly_count} polynomials but setup supports {}",
            setup.max_num_polys_per_commitment_group
        )));
    }
    Ok(())
}

fn validate_commit_polynomials(
    setup: &AkitaProverSetup,
    polynomials: &[Polynomial<AkitaField>],
) -> Result<usize, OpeningsError> {
    let first = polynomials
        .first()
        .ok_or_else(|| invalid_batch("Akita commitment group must contain a polynomial"))?;
    let num_vars = first.num_vars();
    validate_native_commit_shape(setup, num_vars, polynomials.len())?;
    for polynomial in polynomials {
        if polynomial.num_vars() != num_vars {
            return Err(invalid_batch(format!(
                "Akita commitment group mixes {}-variable and {num_vars}-variable polynomials",
                polynomial.num_vars()
            )));
        }
    }
    Ok(num_vars)
}

pub(crate) fn packed_source_sparse_polynomial<S>(
    source: &S,
) -> Result<Option<NativeSparsePoly>, OpeningsError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let layout = source.layout();
    if layout.cells == 0 {
        return Err(invalid_batch(
            "Akita packed witness layout must contain at least one cell",
        ));
    }
    if layout.dimension >= usize::BITS as usize {
        return Err(invalid_batch(format!(
            "Akita packed witness dimension {} exceeds usize bit width",
            layout.dimension
        )));
    }
    let domain_size = 1usize << layout.dimension;
    if domain_size < AKITA_D {
        return Ok(None);
    }
    if layout.cells > domain_size {
        return Err(invalid_batch(format!(
            "Akita packed witness has {} cells but dimension {} supports {domain_size}",
            layout.cells, layout.dimension
        )));
    }

    let mut seen = BTreeSet::new();
    let mut coeffs = Vec::new();
    let mut result = Ok(());
    source.for_each_nonzero(|rank, value| {
        if result.is_err() {
            return;
        }
        if rank >= layout.cells {
            result = Err(invalid_batch(format!(
                "Akita packed witness source emitted rank {rank} outside {} real cells",
                layout.cells
            )));
            return;
        }
        if !seen.insert(rank) {
            result = Err(invalid_batch(format!(
                "Akita packed witness source emitted rank {rank} more than once"
            )));
            return;
        }
        if value != AkitaField::one() {
            result = Err(invalid_batch(format!(
                "Akita sparse packed witness source emitted non-unit value at rank {rank}"
            )));
            return;
        }
        let akita_rank = jolt_to_akita_index(layout.dimension, rank);
        coeffs.push((akita_rank / AKITA_D, akita_rank % AKITA_D, 1));
    });
    result?;

    NativeSparsePoly::from_signed_coeffs(layout.dimension, domain_size / AKITA_D, coeffs)
        .map(Some)
        .map_err(akita_error)
}

pub(crate) fn packed_source_polynomial<S>(
    source: &S,
) -> Result<Polynomial<AkitaField>, OpeningsError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let layout = source.layout();
    if layout.cells == 0 {
        return Err(invalid_batch(
            "Akita packed witness layout must contain at least one cell",
        ));
    }
    if layout.dimension >= usize::BITS as usize {
        return Err(invalid_batch(format!(
            "Akita packed witness dimension {} exceeds usize bit width",
            layout.dimension
        )));
    }
    let domain_size = 1usize << layout.dimension;
    if layout.cells > domain_size {
        return Err(invalid_batch(format!(
            "Akita packed witness has {} cells but dimension {} supports {domain_size}",
            layout.cells, layout.dimension
        )));
    }

    let mut evals = vec![AkitaField::zero(); domain_size];
    let mut seen = vec![false; layout.cells];
    let mut result = Ok(());
    source.for_each_nonzero(|rank, value| {
        if result.is_err() {
            return;
        }
        if rank >= layout.cells {
            result = Err(invalid_batch(format!(
                "Akita packed witness source emitted rank {rank} outside {} real cells",
                layout.cells
            )));
            return;
        }
        if seen[rank] {
            result = Err(invalid_batch(format!(
                "Akita packed witness source emitted rank {rank} more than once"
            )));
            return;
        }
        if value.is_zero() {
            result = Err(invalid_batch(format!(
                "Akita packed witness source emitted zero at rank {rank}"
            )));
            return;
        }

        seen[rank] = true;
        evals[rank] = value;
    });
    result?;

    Ok(Polynomial::new(evals))
}

impl Commitment for AkitaScheme {
    type Output = AkitaCommitment;
}

impl CommitmentScheme for AkitaScheme {
    type Field = AkitaField;
    type Proof = AkitaBatchProof;
    type ProverSetup = AkitaProverSetup;
    type VerifierSetup = AkitaVerifierSetup;
    type Polynomial = Polynomial<AkitaField>;
    type OpeningHint = AkitaProverHint;
    type SetupParams = AkitaSetupParams;

    #[expect(
        clippy::panic,
        reason = "CommitmentScheme::setup cannot return native Akita setup errors"
    )]
    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        let native = NativeScheme::setup_prover(
            params.max_num_vars,
            params.max_num_polys_per_commitment_group,
        )
        .unwrap_or_else(|err| panic!("Akita setup failed: {err}"));
        let prepared = CpuBackend
            .prepare_setup(&native)
            .unwrap_or_else(|err| panic!("Akita setup preparation failed: {err}"));
        let native_verifier = NativeScheme::setup_verifier(&native);
        let verifier = AkitaVerifierSetup {
            max_num_vars: params.max_num_vars,
            max_num_polys_per_commitment_group: params.max_num_polys_per_commitment_group,
            default_layout_digest: params.default_layout_digest,
            packed_layout: params.packed_layout.clone(),
            native: serialize_akita(&native_verifier)
                .unwrap_or_else(|err| panic!("Akita verifier setup serialization failed: {err}")),
        };
        let prover = AkitaProverSetup {
            max_num_vars: params.max_num_vars,
            max_num_polys_per_commitment_group: params.max_num_polys_per_commitment_group,
            default_layout_digest: params.default_layout_digest,
            packed_layout: params.packed_layout,
            native,
            prepared,
            verifier: verifier.clone(),
        };
        (prover, verifier)
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        prover_setup.verifier.clone()
    }

    #[expect(
        clippy::panic,
        reason = "CommitmentScheme::commit cannot return native Akita commit errors"
    )]
    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        let polynomial = Polynomial::from(polynomial_evaluations(poly));
        Self::commit_group(setup, setup.default_layout_digest, &[polynomial])
            .unwrap_or_else(|err| panic!("Akita commit failed: {err}"))
    }

    #[expect(
        clippy::panic,
        reason = "CommitmentScheme::open cannot return native Akita prove errors"
    )]
    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        let hint = hint.unwrap_or_else(|| Self::commit(poly, setup).1);
        let statement = singleton_statement(hint.commitment.clone(), point, eval);
        <Self as BatchOpeningScheme>::prove_batch(
            setup,
            transcript,
            &statement,
            std::slice::from_ref(poly),
            vec![hint],
        )
        .unwrap_or_else(|err| panic!("Akita open failed: {err}"))
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let statement = singleton_statement(commitment.clone(), point, eval);
        <Self as BatchOpeningScheme>::verify_batch(setup, transcript, &statement, proof).map(|_| ())
    }

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    ) {
        transcript.append(&Label(b"akita_opening_inputs"));
        append_field_slice(transcript, b"akita_opening_point", point);
        eval.append_to_transcript(transcript);
    }
}

impl BatchOpeningScheme for AkitaScheme {
    fn prove_batch<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<Self::Proof, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let dense = dense_polynomials(polynomials)?;
        let dense_refs = dense.iter().collect::<Vec<_>>();
        prove_batch_with_native_polynomials(
            setup,
            transcript,
            statement,
            dense_refs.as_slice(),
            hints,
        )
    }

    fn verify_batch<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let normalized = normalize_clear_batch(statement)?;
        if proof.commitment != normalized.commitment {
            return Err(OpeningsError::VerificationFailed);
        }
        validate_verifier_setup(setup, &normalized.commitment)?;
        bind_verifier_setup_key(setup, transcript);
        bind_batch_statement(statement, &normalized, transcript);
        let mut akita_transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/batch");
        let statement_bridge = bind_jolt_transcript_bridge(transcript, &mut akita_transcript);
        if proof.statement_bridge != statement_bridge {
            return Err(OpeningsError::VerificationFailed);
        }
        bind_proof_bytes(proof, transcript);

        let native_verifier = deserialize_akita::<NativeVerifier>(&setup.native, &())?;
        let native_commitment =
            deserialize_akita::<NativeCommitment>(&proof.commitment.native, &())?;
        let proof_shape = deserialize_akita::<NativeProofShape>(&proof.proof_shape, &())?;
        let native_proof = deserialize_akita::<NativeProof>(&proof.proof, &proof_shape)?;
        let openings = statement
            .claims
            .iter()
            .map(|claim| claim.claim)
            .collect::<Vec<_>>();
        let claims: VerifierClaims<'_, AkitaField, _> = (
            statement.pcs_point.as_slice(),
            vec![CommittedOpenings {
                openings: openings.as_slice(),
                commitment: &native_commitment,
            }],
        );
        NativeScheme::batched_verify(
            &native_proof,
            &native_verifier,
            &mut akita_transcript,
            claims,
            BasisMode::Lagrange,
            SetupContributionMode::Direct,
        )
        .map_err(|_| OpeningsError::VerificationFailed)?;

        Ok(BatchOpeningResult {
            coefficients: normalized.coefficients,
            joint_commitment: normalized.commitment,
            reduced_opening: normalized.reduced_opening,
        })
    }
}

pub(crate) fn prove_batch_with_native_polynomials<T, P, OpeningId, RelationId>(
    setup: &AkitaProverSetup,
    transcript: &mut T,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    polynomials: &[P],
    hints: Vec<AkitaProverHint>,
) -> Result<AkitaBatchProof, OpeningsError>
where
    T: Transcript<Challenge = AkitaField>,
    P: AkitaPolyOps<AkitaField, AKITA_D>,
{
    let normalized = normalize_clear_batch(statement)?;
    validate_native_prover_inputs(setup, &normalized.commitment, polynomials, &hints)?;
    bind_verifier_setup_key(&setup.verifier, transcript);
    bind_batch_statement(statement, &normalized, transcript);
    let mut akita_transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/batch");
    let statement_bridge = bind_jolt_transcript_bridge(transcript, &mut akita_transcript);

    let native_commitment =
        deserialize_akita::<NativeCommitment>(&normalized.commitment.native, &())?;
    let native_hint = hints
        .into_iter()
        .next()
        .and_then(|hint| hint.native)
        .ok_or_else(|| invalid_batch("Akita prover hint is missing native opening data"))?;
    let claims: ProverClaims<'_, AkitaField, P, _, NativeHint> = (
        statement.pcs_point.as_slice(),
        vec![CommittedPolynomials {
            polynomials,
            commitment: &native_commitment,
            hint: native_hint,
        }],
    );

    let native_proof = NativeScheme::batched_prove(
        &setup.native,
        &CpuBackend,
        &setup.prepared,
        claims,
        &mut akita_transcript,
        BasisMode::Lagrange,
        SetupContributionMode::Direct,
    )
    .map_err(akita_error)?;
    let proof_shape = native_proof.shape();
    let proof = AkitaBatchProof {
        commitment: normalized.commitment,
        statement_bridge,
        proof_shape: serialize_akita(&proof_shape)?,
        proof: serialize_akita(&native_proof)?,
    };
    bind_proof_bytes(&proof, transcript);
    Ok(proof)
}

impl ZkOpeningScheme for AkitaScheme {
    type HidingCommitment = AkitaHidingCommitment;
    type Blind = ();

    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit(poly, setup)
    }

    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        let proof = Self::open(poly, point, eval, setup, Some(hint), transcript);
        (
            proof,
            AkitaHidingCommitment {
                eval: field_bytes(eval),
            },
            (),
        )
    }

    fn verify_zk(
        _commitment: &Self::Output,
        _point: &[Self::Field],
        _proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        Err(transparent_zk_error())
    }

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    ) {
        transcript.append(&Label(b"akita_zk_opening_inputs"));
        append_field_slice(transcript, b"akita_zk_opening_point", point);
        hiding_commitment.append_to_transcript(transcript);
    }
}

impl ZkBatchOpeningScheme for AkitaScheme {
    fn prove_batch_zk<T, OpeningId, RelationId>(
        _setup: &Self::ProverSetup,
        _transcript: &mut T,
        _statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        _evals: &[Self::Field],
        _polynomials: &[Self::Polynomial],
        _hints: Vec<Self::OpeningHint>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        Err(transparent_zk_error())
    }

    fn verify_batch_zk<T, OpeningId, RelationId>(
        _setup: &Self::VerifierSetup,
        _transcript: &mut T,
        _statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        _proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output, Self::HidingCommitment>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        Err(transparent_zk_error())
    }
}

struct NormalizedBatch {
    commitment: AkitaCommitment,
    coefficients: Vec<AkitaField>,
    reduced_opening: AkitaField,
}

fn normalize_clear_batch<OpeningId, RelationId>(
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> Result<NormalizedBatch, OpeningsError> {
    if statement.claims.is_empty() {
        return Err(invalid_batch(
            "Akita batch opening requires at least one claim",
        ));
    }
    let commitment = statement.claims[0].commitment.clone();
    if statement.pcs_point.len() != commitment.num_vars {
        return Err(invalid_batch(format!(
            "Akita opening point has {} variables but commitment has {}",
            statement.pcs_point.len(),
            commitment.num_vars
        )));
    }
    if statement.layout_digest != commitment.layout_digest {
        return Err(invalid_batch(
            "Akita direct batch statement layout digest does not match commitment layout digest",
        ));
    }
    if statement.logical_point != statement.pcs_point {
        return Err(invalid_batch(
            "Akita direct batch requires logical point and PCS point to match",
        ));
    }
    let mut coefficients = Vec::with_capacity(statement.claims.len());
    let mut reduced_opening = AkitaField::zero();
    for claim in &statement.claims {
        if claim.commitment != commitment {
            return Err(invalid_batch(
                "Akita batch statement must use exactly one commitment group",
            ));
        }
        if !matches!(claim.view, PhysicalView::Direct) {
            return Err(invalid_batch(
                "Akita native adapter expects direct physical views; use PackedCombine to lower packed views first",
            ));
        }
        coefficients.push(claim.scale);
        reduced_opening += claim.scale * claim.claim;
    }
    if commitment.poly_count != statement.claims.len() {
        return Err(invalid_batch(format!(
            "Akita commitment covers {} polynomials but statement has {} claims",
            commitment.poly_count,
            statement.claims.len()
        )));
    }
    Ok(NormalizedBatch {
        commitment,
        coefficients,
        reduced_opening,
    })
}

fn validate_native_prover_inputs<P>(
    setup: &AkitaProverSetup,
    commitment: &AkitaCommitment,
    polynomials: &[P],
    hints: &[AkitaProverHint],
) -> Result<(), OpeningsError>
where
    P: AkitaPolyOps<AkitaField, AKITA_D>,
{
    validate_setup_shape(
        setup.max_num_vars,
        setup.max_num_polys_per_commitment_group,
        setup.default_layout_digest,
        commitment,
    )?;
    if polynomials.len() != commitment.poly_count {
        return Err(invalid_batch(format!(
            "Akita prover received {} polynomials for {} commitment slots",
            polynomials.len(),
            commitment.poly_count
        )));
    }
    if hints.len() != 1 {
        return Err(invalid_batch(format!(
            "Akita prover expects one grouped commitment hint, got {}",
            hints.len()
        )));
    }
    if !hints[0].matches_commitment(commitment) {
        return Err(invalid_batch(
            "Akita prover hint does not match the statement commitment",
        ));
    }
    for polynomial in polynomials {
        if polynomial.num_vars() != commitment.num_vars {
            return Err(invalid_batch(format!(
                "Akita polynomial has {} variables but commitment has {}",
                polynomial.num_vars(),
                commitment.num_vars
            )));
        }
    }
    Ok(())
}

fn validate_verifier_setup(
    setup: &AkitaVerifierSetup,
    commitment: &AkitaCommitment,
) -> Result<(), OpeningsError> {
    validate_setup_shape(
        setup.max_num_vars,
        setup.max_num_polys_per_commitment_group,
        setup.default_layout_digest,
        commitment,
    )
}

fn validate_setup_shape(
    max_num_vars: usize,
    max_num_polys_per_commitment_group: usize,
    _default_layout_digest: [u8; 32],
    commitment: &AkitaCommitment,
) -> Result<(), OpeningsError> {
    if commitment.num_vars > max_num_vars {
        return Err(OpeningsError::PolynomialTooLarge {
            poly_size: commitment.num_vars,
            setup_max: max_num_vars,
        });
    }
    if commitment.num_vars != max_num_vars {
        return Err(invalid_batch(format!(
            "Akita commitment dimension {} does not match exact setup dimension {max_num_vars}",
            commitment.num_vars
        )));
    }
    if commitment.poly_count > max_num_polys_per_commitment_group {
        return Err(invalid_batch(format!(
            "Akita commitment covers {} polynomials but setup supports {}",
            commitment.poly_count, max_num_polys_per_commitment_group
        )));
    }
    Ok(())
}

fn validate_setup_layout_digest(expected: [u8; 32], actual: [u8; 32]) -> Result<(), OpeningsError> {
    if actual != expected {
        return Err(invalid_batch(
            "Akita commitment layout digest does not match setup key layout digest",
        ));
    }
    Ok(())
}

pub(crate) fn bind_verifier_setup_key<T>(setup: &AkitaVerifierSetup, transcript: &mut T)
where
    T: Transcript<Challenge = AkitaField>,
{
    transcript.append(&Label(b"akita_setup_key"));
    transcript.append_bytes(b"layerzero-akita/fp128/d64full");
    transcript.append(&LabelWithCount(
        b"layerzero_akita_rev",
        LAYERZERO_AKITA_REV.len() as u64,
    ));
    transcript.append_bytes(LAYERZERO_AKITA_REV.as_bytes());
    transcript.append(&U64Word(AKITA_D as u64));
    transcript.append(&U64Word(setup.max_num_vars as u64));
    transcript.append(&U64Word(setup.max_num_polys_per_commitment_group as u64));
    transcript.append_bytes(&setup.default_layout_digest);
    match &setup.packed_layout {
        Some(layout) => {
            transcript.append_bytes(&[1]);
            transcript.append_bytes(&layout.digest);
            transcript.append(&U64Word(layout.dimension as u64));
            transcript.append(&U64Word(layout.cells as u64));
        }
        None => transcript.append_bytes(&[0]),
    }
    transcript.append(&LabelWithCount(
        b"akita_verifier_setup",
        setup.native.len() as u64,
    ));
    transcript.append_bytes(&setup.native);
}

fn bind_batch_statement<OpeningId, RelationId, T>(
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
    normalized: &NormalizedBatch,
    transcript: &mut T,
) where
    T: Transcript<Challenge = AkitaField>,
{
    transcript.append(&Label(b"akita_batch_statement"));
    normalized.commitment.append_to_transcript(transcript);
    transcript.append_bytes(&statement.layout_digest);
    transcript.append_bytes(&normalized.commitment.layout_digest);
    append_field_slice(transcript, b"akita_logical_point", &statement.logical_point);
    append_field_slice(transcript, b"akita_pcs_point", &statement.pcs_point);
    transcript.append(&LabelWithCount(
        b"akita_claims",
        statement.claims.len() as u64,
    ));
    for claim in &statement.claims {
        claim.commitment.append_to_transcript(transcript);
        claim.claim.append_to_transcript(transcript);
        claim.scale.append_to_transcript(transcript);
        transcript.append_bytes(&[0]);
    }
    append_field_slice(transcript, b"akita_coefficients", &normalized.coefficients);
    normalized.reduced_opening.append_to_transcript(transcript);
}

fn bind_jolt_transcript_bridge<T>(
    jolt_transcript: &mut T,
    akita_transcript: &mut AkitaTranscript<AkitaField>,
) -> Vec<u8>
where
    T: Transcript<Challenge = AkitaField>,
{
    let bridge = jolt_transcript.challenge_scalar();
    akita_transcript.append_field(b"jolt_statement_bridge", &bridge);
    field_bytes(bridge)
}

fn bind_proof_bytes<T>(proof: &AkitaBatchProof, transcript: &mut T)
where
    T: Transcript<Challenge = AkitaField>,
{
    transcript.append(&LabelWithCount(
        b"akita_stmt_bridge",
        proof.statement_bridge.len() as u64,
    ));
    transcript.append_bytes(&proof.statement_bridge);
    transcript.append(&LabelWithCount(
        b"akita_proof_shape",
        proof.proof_shape.len() as u64,
    ));
    transcript.append_bytes(&proof.proof_shape);
    transcript.append(&LabelWithCount(b"akita_proof", proof.proof.len() as u64));
    transcript.append_bytes(&proof.proof);
}

fn singleton_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment> {
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: commitment.layout_digest,
        claims: vec![jolt_openings::BatchOpeningClaim {
            id: (),
            relation: (),
            commitment,
            claim: eval,
            view: PhysicalView::Direct,
            scale: AkitaField::one(),
        }],
    }
}

fn dense_polynomials(
    polynomials: &[Polynomial<AkitaField>],
) -> Result<Vec<NativeDensePoly>, OpeningsError> {
    polynomials
        .iter()
        .map(|poly| {
            let evals = jolt_to_akita_evals(poly.num_vars(), poly.evals())?;
            NativeDensePoly::from_field_evals(poly.num_vars(), &evals).map_err(akita_error)
        })
        .collect()
}

fn jolt_to_akita_evals(
    num_vars: usize,
    jolt_evals: &[AkitaField],
) -> Result<Vec<AkitaField>, OpeningsError> {
    let Some(expected) = 1usize.checked_shl(num_vars as u32) else {
        return Err(invalid_batch(format!(
            "Akita polynomial dimension {num_vars} exceeds usize bit width"
        )));
    };
    if jolt_evals.len() != expected {
        return Err(invalid_batch(format!(
            "Akita polynomial has {} evaluations but dimension {num_vars} requires {expected}",
            jolt_evals.len()
        )));
    }
    if num_vars == 0 {
        return Ok(jolt_evals.to_vec());
    }
    let mut akita_evals = vec![AkitaField::zero(); jolt_evals.len()];
    for (jolt_index, &eval) in jolt_evals.iter().enumerate() {
        let akita_index = jolt_to_akita_index(num_vars, jolt_index);
        akita_evals[akita_index] = eval;
    }
    Ok(akita_evals)
}

fn jolt_to_akita_index(num_vars: usize, index: usize) -> usize {
    if num_vars == 0 {
        return index;
    }
    index.reverse_bits() >> (usize::BITS as usize - num_vars)
}

fn polynomial_evaluations<P>(polynomial: &P) -> Vec<AkitaField>
where
    P: MultilinearPoly<AkitaField> + ?Sized,
{
    let capacity = if polynomial.num_vars() < usize::BITS as usize {
        1usize << polynomial.num_vars()
    } else {
        0
    };
    let mut evals = Vec::with_capacity(capacity);
    polynomial.for_each_row(polynomial.num_vars(), &mut |_, row| {
        evals.extend_from_slice(row);
    });
    evals
}

fn serialize_akita<T>(value: &T) -> Result<Vec<u8>, OpeningsError>
where
    T: AkitaSerialize,
{
    let mut bytes = Vec::with_capacity(value.compressed_size());
    value
        .serialize_compressed(&mut bytes)
        .map_err(akita_error)?;
    Ok(bytes)
}

fn deserialize_akita<T>(bytes: &[u8], ctx: &T::Context) -> Result<T, OpeningsError>
where
    T: AkitaDeserialize,
{
    T::deserialize_compressed(Cursor::new(bytes), ctx).map_err(akita_error)
}

fn field_bytes(value: AkitaField) -> Vec<u8> {
    let mut bytes = vec![0u8; AkitaField::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    bytes
}

fn invalid_batch(message: impl Into<String>) -> OpeningsError {
    OpeningsError::InvalidBatch(message.into())
}

fn akita_error(error: impl ToString) -> OpeningsError {
    OpeningsError::InvalidBatch(error.to_string())
}

fn transparent_zk_error() -> OpeningsError {
    OpeningsError::InvalidBatch(
        "Akita native adapter is transparent-only and does not support ZK openings yet".to_owned(),
    )
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

    use super::*;
    use jolt_openings::BatchOpeningClaim;

    #[derive(Default)]
    struct RecordingTranscript {
        bytes: Vec<u8>,
    }

    impl Transcript for RecordingTranscript {
        type Challenge = AkitaField;

        fn new(label: &'static [u8]) -> Self {
            let mut transcript = Self::default();
            transcript.append_bytes(label);
            transcript
        }

        fn append_bytes(&mut self, bytes: &[u8]) {
            self.bytes.extend_from_slice(bytes);
        }

        fn challenge(&mut self) -> Self::Challenge {
            AkitaField::zero()
        }

        fn state(&self) -> [u8; 32] {
            [0; 32]
        }
    }

    #[test]
    fn setup_key_transcript_binds_layerzero_revision() {
        assert!(include_str!("../Cargo.toml").contains(LAYERZERO_AKITA_REV));
        let setup = AkitaVerifierSetup {
            max_num_vars: 4,
            max_num_polys_per_commitment_group: 1,
            default_layout_digest: [7; 32],
            packed_layout: None,
            native: vec![1, 2, 3],
        };
        let mut transcript = RecordingTranscript::new(b"akita-setup-key-test");

        bind_verifier_setup_key(&setup, &mut transcript);

        assert!(contains_subslice(
            &transcript.bytes,
            b"layerzero-akita/fp128/d64full"
        ));
        assert!(contains_subslice(&transcript.bytes, b"layerzero_akita_rev"));
        assert!(contains_subslice(
            &transcript.bytes,
            LAYERZERO_AKITA_REV.as_bytes()
        ));
    }

    #[test]
    fn direct_opening_requires_statement_commitment_layout_digest() {
        let setup_params = AkitaSetupParams::new(1, 1, [7; 32]);
        let (prover_setup, verifier_setup) = AkitaScheme::setup(setup_params);
        let polynomial = Polynomial::new(vec![AkitaField::from_u64(2), AkitaField::from_u64(5)]);
        let commitment_digest = [9; 32];
        let (commitment, hint) = AkitaScheme::commit_group(
            &prover_setup,
            commitment_digest,
            std::slice::from_ref(&polynomial),
        )
        .expect("direct commitment may use its own layout digest");
        assert_eq!(commitment.layout_digest, commitment_digest);

        let point = vec![AkitaField::from_u64(3)];
        let claim = polynomial.evaluate(&point);
        let statement = BatchOpeningStatement {
            logical_point: point.clone(),
            pcs_point: point,
            layout_digest: commitment_digest,
            claims: vec![BatchOpeningClaim {
                id: (),
                relation: (),
                commitment: commitment.clone(),
                claim,
                view: PhysicalView::Direct,
                scale: AkitaField::one(),
            }],
        };
        assert_eq!(statement.layout_digest, commitment.layout_digest);

        let mut prover_transcript = RecordingTranscript::new(b"akita-direct-layout");
        let proof = AkitaScheme::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &[polynomial],
            vec![hint],
        )
        .expect("direct proof should prove");
        assert!(contains_subslice(
            &prover_transcript.bytes,
            &commitment_digest
        ));

        let mut changed_wrapper_statement = statement.clone();
        changed_wrapper_statement.layout_digest = [13; 32];
        let mut verifier_transcript = RecordingTranscript::new(b"akita-direct-layout");
        let _error = AkitaScheme::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &changed_wrapper_statement,
            &proof,
        )
        .expect_err("changed direct statement digest should reject");

        let mut changed_commitment_statement = statement;
        changed_commitment_statement.claims[0]
            .commitment
            .layout_digest = [15; 32];
        let mut verifier_transcript = RecordingTranscript::new(b"akita-direct-layout");
        let _error = AkitaScheme::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &changed_commitment_statement,
            &proof,
        )
        .expect_err("changed direct commitment digest should reject");
    }

    fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    }
}
