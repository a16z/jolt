//! Polynomial commitment scheme (PCS) trait hierarchy.
//!
//! - [`CommitmentScheme`] — commit, open, verify for multilinear polynomials.
//! - [`AdditivelyHomomorphic`] — linear combination of commitments.
//! - [`StreamingCommitment`] — chunked commitment without full materialization.
//! - [`ZkOpeningScheme`] — zero-knowledge commitments and opening proofs.

use std::fmt::Debug;

use jolt_crypto::{Commitment, HomomorphicCommitment};
use jolt_field::{Field, FromPrimitiveInt};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};
use serde::{de::DeserializeOwned, Serialize};

use crate::error::OpeningsError;

/// Verifier statement for a same-point batch opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchOpeningStatement<F, C, OpeningId = (), RelationId = (), Claim = F> {
    pub logical_point: Vec<F>,
    pub pcs_point: Vec<F>,
    pub layout_digest: [u8; 32],
    pub claims: Vec<BatchOpeningClaim<F, C, OpeningId, RelationId, Claim>>,
}

/// One logical opening claim inside a same-point batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchOpeningClaim<F, C, OpeningId = (), RelationId = (), Claim = F> {
    pub id: OpeningId,
    pub relation: RelationId,
    pub commitment: C,
    pub claim: Claim,
    pub view: PhysicalView<F>,
    pub scale: F,
}

/// Physical commitment view used to satisfy a logical opening claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicalView<F> {
    Direct,
    PackedLinear {
        layout_digest: [u8; 32],
        coefficients: Vec<F>,
    },
}

/// PCS-specific reduction data produced by verifying a same-point batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchOpeningResult<F, C, R = F> {
    pub coefficients: Vec<F>,
    pub joint_commitment: C,
    pub reduced_opening: R,
}

/// Commit to f: F^n -> F, then prove f(r) = v for verifier-chosen r.
pub trait CommitmentScheme: Commitment {
    type Field: Field;
    type Proof: Clone + Debug + Eq + Send + Sync + 'static + Serialize + DeserializeOwned;
    type ProverSetup: Clone + Send + Sync;
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;

    type Polynomial: MultilinearPoly<Self::Field> + From<Vec<Self::Field>>;

    /// Auxiliary data from commit reused during opening (e.g. Dory row commitments).
    type OpeningHint: Clone + Send + Sync + Default;

    type SetupParams;

    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup);

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup;

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof;

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    );
}

/// C = Σ s_i · C_i.
pub trait AdditivelyHomomorphic: CommitmentScheme
where
    Self::Output: HomomorphicCommitment<Self::Field>,
{
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output;

    fn combine_hints(
        _hints: Vec<Self::OpeningHint>,
        _scalars: &[Self::Field],
    ) -> Self::OpeningHint {
        Self::OpeningHint::default()
    }
}

/// Same-point batch opening extension for a PCS.
pub trait BatchOpeningScheme: CommitmentScheme {
    fn prove_batch<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<Self::Proof, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;

    fn verify_batch<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;
}

/// ZK same-point batch opening extension mirroring [`ZkOpeningScheme`].
pub trait ZkBatchOpeningScheme: BatchOpeningScheme + ZkOpeningScheme {
    #[expect(
        clippy::type_complexity,
        reason = "ZK batch openings mirror ZkOpeningScheme's proof, hiding commitment, and blind tuple"
    )]
    fn prove_batch_zk<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        evals: &[Self::Field],
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;

    #[expect(
        clippy::type_complexity,
        reason = "ZK batch verification returns the same opening result shape with a hiding commitment"
    )]
    fn verify_batch_zk<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output, Self::HidingCommitment>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>;
}

/// Incremental commitment without full materialization.
pub trait StreamingCommitment: CommitmentScheme {
    type PartialCommitment: Clone + Send + Sync;

    fn begin(setup: &Self::ProverSetup) -> Self::PartialCommitment;

    fn feed(
        partial: &mut Self::PartialCommitment,
        chunk: &[Self::Field],
        setup: &Self::ProverSetup,
    );

    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output;
}

/// Opening proofs that hide the evaluation behind a commitment.
pub trait ZkOpeningScheme: CommitmentScheme {
    type HidingCommitment: Clone
        + Debug
        + Eq
        + Send
        + Sync
        + 'static
        + Serialize
        + DeserializeOwned
        + AppendToTranscript;

    type Blind: Clone + Send + Sync;

    /// Commit in the scheme's ZK/hiding mode.
    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    /// Open a ZK/hiding commitment using the opening hint returned by
    /// [`commit_zk`](Self::commit_zk).
    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind);

    /// Verify a ZK opening proof and return the hiding commitment to the
    /// evaluation that the proof binds internally.
    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError>;

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    );
}

impl<PCS> BatchOpeningScheme for PCS
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
{
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
        validate_batch_prover_inputs::<Self, _, _, _>(statement, polynomials, hints.len())?;
        let reduction = clear_homomorphic_batch_reduction::<Self, _, _, _>(statement, transcript)?;
        let joint_polynomial = combine_polynomials::<Self>(polynomials, &reduction.proof_scalars)?;
        let combined_hint = Self::combine_hints(hints, &reduction.proof_scalars);
        let proof = Self::open(
            &joint_polynomial,
            statement.pcs_point.as_slice(),
            reduction.joint_eval,
            setup,
            Some(combined_hint),
            transcript,
        );
        Self::bind_opening_inputs(
            transcript,
            statement.logical_point.as_slice(),
            &reduction.joint_eval,
        );
        Ok(proof)
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
        validate_batch_statement(statement)?;
        let reduction = clear_homomorphic_batch_reduction::<Self, _, _, _>(statement, transcript)?;
        let joint_commitment =
            combine_commitments::<Self, _, _, _>(statement, &reduction.proof_scalars);
        Self::verify(
            &joint_commitment,
            statement.pcs_point.as_slice(),
            reduction.joint_eval,
            proof,
            setup,
            transcript,
        )?;
        Self::bind_opening_inputs(
            transcript,
            statement.logical_point.as_slice(),
            &reduction.joint_eval,
        );
        Ok(BatchOpeningResult {
            coefficients: reduction.logical_coefficients,
            joint_commitment,
            reduced_opening: reduction.joint_eval,
        })
    }
}

impl<PCS> ZkBatchOpeningScheme for PCS
where
    PCS: AdditivelyHomomorphic + ZkOpeningScheme,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
{
    fn prove_batch_zk<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        evals: &[Self::Field],
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        validate_batch_prover_inputs::<Self, _, _, _>(statement, polynomials, hints.len())?;
        if evals.len() != statement.claims.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "evaluation count {} does not match claim count {}",
                evals.len(),
                statement.claims.len()
            )));
        }

        let reduction =
            hidden_homomorphic_batch_reduction::<Self, _, _, _>(statement, evals, transcript)?;
        let joint_polynomial = combine_polynomials::<Self>(polynomials, &reduction.proof_scalars)?;
        let combined_hint = Self::combine_hints(hints, &reduction.proof_scalars);
        let (proof, hiding_commitment, blind) = Self::open_zk(
            &joint_polynomial,
            statement.pcs_point.as_slice(),
            reduction.joint_eval,
            setup,
            combined_hint,
            transcript,
        );
        Self::bind_zk_opening_inputs(
            transcript,
            statement.logical_point.as_slice(),
            &hiding_commitment,
        );
        Ok((proof, hiding_commitment, blind))
    }

    fn verify_batch_zk<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output, Self::HidingCommitment>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        validate_batch_statement(statement)?;
        let scalars = hidden_homomorphic_batch_scalars::<Self, _, _, _>(statement, transcript)?;
        let joint_commitment =
            combine_commitments::<Self, _, _, _>(statement, &scalars.proof_scalars);
        let hiding_commitment = Self::verify_zk(
            &joint_commitment,
            statement.pcs_point.as_slice(),
            proof,
            setup,
            transcript,
        )?;
        Self::bind_zk_opening_inputs(
            transcript,
            statement.logical_point.as_slice(),
            &hiding_commitment,
        );
        Ok(BatchOpeningResult {
            coefficients: scalars.logical_coefficients,
            joint_commitment,
            reduced_opening: hiding_commitment,
        })
    }
}

struct HomomorphicBatchReduction<F> {
    proof_scalars: Vec<F>,
    logical_coefficients: Vec<F>,
    joint_eval: F,
}

struct HomomorphicBatchScalars<F> {
    proof_scalars: Vec<F>,
    logical_coefficients: Vec<F>,
}

fn clear_homomorphic_batch_reduction<PCS, T, OpeningId, RelationId>(
    statement: &BatchOpeningStatement<PCS::Field, PCS::Output, OpeningId, RelationId>,
    transcript: &mut T,
) -> Result<HomomorphicBatchReduction<PCS::Field>, OpeningsError>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    validate_batch_statement(statement)?;
    let scaled_claims = statement
        .claims
        .iter()
        .map(|claim| claim.claim * claim.scale)
        .collect::<Vec<_>>();
    transcript.append(&LabelWithCount(b"rlc_claims", scaled_claims.len() as u64));
    for claim in &scaled_claims {
        claim.append_to_transcript(transcript);
    }
    let gamma_powers = transcript.challenge_scalar_powers(scaled_claims.len());
    let logical_coefficients = statement
        .claims
        .iter()
        .zip(&gamma_powers)
        .map(|(claim, gamma)| *gamma * claim.scale)
        .collect::<Vec<_>>();
    let joint_eval = gamma_powers
        .iter()
        .zip(&scaled_claims)
        .fold(PCS::Field::from_u64(0), |acc, (gamma, claim)| {
            acc + *gamma * *claim
        });
    Ok(HomomorphicBatchReduction {
        proof_scalars: gamma_powers,
        logical_coefficients,
        joint_eval,
    })
}

fn hidden_homomorphic_batch_reduction<PCS, T, OpeningId, RelationId>(
    statement: &BatchOpeningStatement<PCS::Field, PCS::Output, OpeningId, RelationId, ()>,
    evals: &[PCS::Field],
    transcript: &mut T,
) -> Result<HomomorphicBatchReduction<PCS::Field>, OpeningsError>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let scalars = hidden_homomorphic_batch_scalars::<PCS, _, _, _>(statement, transcript)?;
    let joint_eval = scalars
        .proof_scalars
        .iter()
        .zip(evals)
        .fold(PCS::Field::from_u64(0), |acc, (scalar, eval)| {
            acc + *scalar * *eval
        });
    Ok(HomomorphicBatchReduction {
        proof_scalars: scalars.proof_scalars,
        logical_coefficients: scalars.logical_coefficients,
        joint_eval,
    })
}

fn hidden_homomorphic_batch_scalars<PCS, T, OpeningId, RelationId>(
    statement: &BatchOpeningStatement<PCS::Field, PCS::Output, OpeningId, RelationId, ()>,
    transcript: &mut T,
) -> Result<HomomorphicBatchScalars<PCS::Field>, OpeningsError>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    validate_batch_statement(statement)?;
    let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
    let logical_coefficients = statement
        .claims
        .iter()
        .zip(&gamma_powers)
        .map(|(claim, gamma)| *gamma * claim.scale)
        .collect();
    Ok(HomomorphicBatchScalars {
        proof_scalars: gamma_powers,
        logical_coefficients,
    })
}

fn validate_batch_statement<F, C, OpeningId, RelationId, Claim>(
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId, Claim>,
) -> Result<(), OpeningsError>
where
    F: Field,
{
    if statement.claims.is_empty() {
        return Err(OpeningsError::InvalidBatch(
            "batch opening requires at least one claim".to_owned(),
        ));
    }
    for claim in &statement.claims {
        if !matches!(claim.view, PhysicalView::Direct) {
            return Err(OpeningsError::InvalidBatch(
                "homomorphic batch openings only support direct physical views".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_batch_prover_inputs<PCS, OpeningId, RelationId, Claim>(
    statement: &BatchOpeningStatement<PCS::Field, PCS::Output, OpeningId, RelationId, Claim>,
    polynomials: &[PCS::Polynomial],
    hint_count: usize,
) -> Result<(), OpeningsError>
where
    PCS: CommitmentScheme,
{
    validate_batch_statement(statement)?;
    if polynomials.len() != statement.claims.len() {
        return Err(OpeningsError::InvalidBatch(format!(
            "polynomial count {} does not match claim count {}",
            polynomials.len(),
            statement.claims.len()
        )));
    }
    if hint_count != statement.claims.len() {
        return Err(OpeningsError::InvalidBatch(format!(
            "hint count {hint_count} does not match claim count {}",
            statement.claims.len()
        )));
    }
    for polynomial in polynomials {
        if polynomial.num_vars() != statement.pcs_point.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "polynomial has {} variables but opening point has {}",
                polynomial.num_vars(),
                statement.pcs_point.len()
            )));
        }
    }
    Ok(())
}

fn combine_commitments<PCS, OpeningId, RelationId, Claim>(
    statement: &BatchOpeningStatement<PCS::Field, PCS::Output, OpeningId, RelationId, Claim>,
    scalars: &[PCS::Field],
) -> PCS::Output
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
{
    let commitments = statement
        .claims
        .iter()
        .map(|claim| claim.commitment.clone())
        .collect::<Vec<_>>();
    PCS::combine(&commitments, scalars)
}

fn combine_polynomials<PCS>(
    polynomials: &[PCS::Polynomial],
    scalars: &[PCS::Field],
) -> Result<PCS::Polynomial, OpeningsError>
where
    PCS: CommitmentScheme,
{
    debug_assert_eq!(polynomials.len(), scalars.len());
    let mut combined = Vec::new();
    for (polynomial, scalar) in polynomials.iter().zip(scalars) {
        let evals = polynomial_evaluations(polynomial);
        if combined.is_empty() {
            combined = vec![PCS::Field::from_u64(0); evals.len()];
        } else if combined.len() != evals.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "polynomial evaluation length {} does not match first length {}",
                evals.len(),
                combined.len()
            )));
        }
        for (acc, eval) in combined.iter_mut().zip(evals) {
            *acc += *scalar * eval;
        }
    }
    Ok(PCS::Polynomial::from(combined))
}

fn polynomial_evaluations<F, P>(polynomial: &P) -> Vec<F>
where
    F: Field,
    P: MultilinearPoly<F>,
{
    let mut evals = Vec::with_capacity(1usize << polynomial.num_vars());
    polynomial.for_each_row(polynomial.num_vars(), &mut |_, row| {
        evals.extend_from_slice(row);
    });
    evals
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "tests assert successful batch proof results"
)]
mod tests {
    use super::*;
    use crate::mock::MockCommitmentScheme;
    use jolt_field::{Fr, FromPrimitiveInt, Invertible};
    use jolt_poly::Polynomial;
    use jolt_transcript::Blake2bTranscript;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum OpeningId {
        A,
        B,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum RelationId {
        First,
        Second,
    }

    #[test]
    fn batch_opening_statement_preserves_claim_metadata() {
        let statement = BatchOpeningStatement {
            logical_point: vec![3_u64, 5],
            pcs_point: vec![5_u64, 3],
            layout_digest: [7; 32],
            claims: vec![
                BatchOpeningClaim {
                    id: OpeningId::A,
                    relation: RelationId::First,
                    commitment: 11_u64,
                    claim: 13,
                    view: PhysicalView::Direct,
                    scale: 17,
                },
                BatchOpeningClaim {
                    id: OpeningId::B,
                    relation: RelationId::Second,
                    commitment: 19_u64,
                    claim: 23,
                    view: PhysicalView::PackedLinear {
                        layout_digest: [23; 32],
                        coefficients: vec![29, 31],
                    },
                    scale: 37,
                },
            ],
        };

        assert_eq!(statement.logical_point, vec![3, 5]);
        assert_eq!(statement.pcs_point, vec![5, 3]);
        assert_eq!(statement.layout_digest, [7; 32]);
        assert_eq!(statement.claims[0].id, OpeningId::A);
        assert_eq!(statement.claims[0].relation, RelationId::First);
        assert_eq!(statement.claims[0].claim, 13);
        assert_eq!(statement.claims[1].id, OpeningId::B);
        assert_eq!(statement.claims[1].relation, RelationId::Second);
        assert_eq!(statement.claims[1].claim, 23);
    }

    #[test]
    fn zk_batch_opening_statement_can_omit_claim_payload() {
        let statement: BatchOpeningStatement<_, _, _, _, ()> = BatchOpeningStatement {
            logical_point: vec![1_u64],
            pcs_point: vec![1_u64],
            layout_digest: [0; 32],
            claims: vec![BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::First,
                commitment: 2_u64,
                claim: (),
                view: PhysicalView::Direct,
                scale: 1,
            }],
        };

        let _: () = statement.claims[0].claim;
    }

    #[test]
    fn physical_view_records_packed_layout_and_coefficients() {
        let view = PhysicalView::PackedLinear {
            layout_digest: [41; 32],
            coefficients: vec![43_u64, 47],
        };

        assert!(matches!(view, PhysicalView::PackedLinear { .. }));
        if let PhysicalView::PackedLinear {
            layout_digest,
            coefficients,
        } = view
        {
            assert_eq!(layout_digest, [41; 32]);
            assert_eq!(coefficients, vec![43, 47]);
        }
    }

    type MockPCS = MockCommitmentScheme<Fr>;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn batch_polynomials() -> (Vec<Polynomial<Fr>>, Vec<Fr>) {
        let polynomials = vec![
            Polynomial::new((0..8).map(|value| fr(value + 1)).collect()),
            Polynomial::new((0..8).map(|value| fr(17 + 2 * value)).collect()),
        ];
        let point = vec![fr(2), fr(3), fr(5)];
        (polynomials, point)
    }

    fn clear_batch_statement(
        polynomials: &[Polynomial<Fr>],
        point: &[Fr],
    ) -> BatchOpeningStatement<Fr, <MockPCS as Commitment>::Output, OpeningId, RelationId> {
        let commitments = polynomials
            .iter()
            .map(|polynomial| MockPCS::commit(polynomial.evaluations(), &()).0)
            .collect::<Vec<_>>();
        let first_scale = fr(2);
        let second_scale = fr(5);
        BatchOpeningStatement {
            logical_point: point.to_vec(),
            pcs_point: point.to_vec(),
            layout_digest: [9; 32],
            claims: vec![
                BatchOpeningClaim {
                    id: OpeningId::A,
                    relation: RelationId::First,
                    commitment: commitments[0].clone(),
                    claim: polynomials[0].evaluate(point)
                        * first_scale.inverse().expect("scale is nonzero"),
                    view: PhysicalView::Direct,
                    scale: first_scale,
                },
                BatchOpeningClaim {
                    id: OpeningId::B,
                    relation: RelationId::Second,
                    commitment: commitments[1].clone(),
                    claim: polynomials[1].evaluate(point)
                        * second_scale.inverse().expect("scale is nonzero"),
                    view: PhysicalView::Direct,
                    scale: second_scale,
                },
            ],
        }
    }

    #[test]
    fn homomorphic_batch_opening_roundtrip_clear() {
        let (polynomials, point) = batch_polynomials();
        let statement = clear_batch_statement(&polynomials, &point);
        let hints = vec![(); polynomials.len()];

        let mut prover_transcript = Blake2bTranscript::new(b"batch-clear");
        let proof = <MockPCS as BatchOpeningScheme>::prove_batch(
            &(),
            &mut prover_transcript,
            &statement,
            &polynomials,
            hints,
        )
        .expect("batch proof should be produced");

        let mut verifier_transcript = Blake2bTranscript::new(b"batch-clear");
        let result = <MockPCS as BatchOpeningScheme>::verify_batch(
            &(),
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("batch proof should verify");

        assert_eq!(result.coefficients.len(), statement.claims.len());
        assert_eq!(
            result.reduced_opening,
            result
                .coefficients
                .iter()
                .zip(&statement.claims)
                .fold(Fr::from_u64(0), |acc, (coefficient, claim)| {
                    acc + *coefficient * claim.claim
                })
        );
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn homomorphic_batch_opening_rejects_tampered_clear_claim() {
        let (polynomials, point) = batch_polynomials();
        let statement = clear_batch_statement(&polynomials, &point);

        let mut prover_transcript = Blake2bTranscript::new(b"batch-clear-tampered");
        let proof = <MockPCS as BatchOpeningScheme>::prove_batch(
            &(),
            &mut prover_transcript,
            &statement,
            &polynomials,
            vec![(); polynomials.len()],
        )
        .expect("batch proof should be produced");

        let mut tampered = statement.clone();
        tampered.claims[1].claim += fr(1);

        let mut verifier_transcript = Blake2bTranscript::new(b"batch-clear-tampered");
        let result = <MockPCS as BatchOpeningScheme>::verify_batch(
            &(),
            &mut verifier_transcript,
            &tampered,
            &proof,
        );
        assert!(result.is_err(), "tampered claim should fail");
    }

    #[test]
    fn homomorphic_batch_opening_rejects_packed_view() {
        let (polynomials, point) = batch_polynomials();
        let mut statement = clear_batch_statement(&polynomials, &point);
        statement.claims[0].view = PhysicalView::PackedLinear {
            layout_digest: [3; 32],
            coefficients: vec![fr(1), fr(2)],
        };

        let mut transcript = Blake2bTranscript::new(b"batch-packed");
        let result = <MockPCS as BatchOpeningScheme>::prove_batch(
            &(),
            &mut transcript,
            &statement,
            &polynomials,
            vec![(); polynomials.len()],
        );
        assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
    }

    #[test]
    fn homomorphic_batch_opening_rejects_mismatched_witness_count() {
        let (polynomials, point) = batch_polynomials();
        let statement = clear_batch_statement(&polynomials, &point);

        let mut transcript = Blake2bTranscript::new(b"batch-mismatch");
        let result = <MockPCS as BatchOpeningScheme>::prove_batch(
            &(),
            &mut transcript,
            &statement,
            &polynomials[..1],
            vec![()],
        );
        assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
    }

    fn zk_batch_statement(
        polynomials: &[Polynomial<Fr>],
        point: &[Fr],
    ) -> BatchOpeningStatement<Fr, <MockPCS as Commitment>::Output, OpeningId, RelationId, ()> {
        let commitments = polynomials
            .iter()
            .map(|polynomial| {
                <MockPCS as ZkOpeningScheme>::commit_zk(polynomial.evaluations(), &()).0
            })
            .collect::<Vec<_>>();
        BatchOpeningStatement {
            logical_point: point.to_vec(),
            pcs_point: point.to_vec(),
            layout_digest: [10; 32],
            claims: vec![
                BatchOpeningClaim {
                    id: OpeningId::A,
                    relation: RelationId::First,
                    commitment: commitments[0].clone(),
                    claim: (),
                    view: PhysicalView::Direct,
                    scale: fr(3),
                },
                BatchOpeningClaim {
                    id: OpeningId::B,
                    relation: RelationId::Second,
                    commitment: commitments[1].clone(),
                    claim: (),
                    view: PhysicalView::Direct,
                    scale: fr(7),
                },
            ],
        }
    }

    #[test]
    fn homomorphic_batch_opening_roundtrip_zk() {
        let (polynomials, point) = batch_polynomials();
        let statement = zk_batch_statement(&polynomials, &point);
        let evals = polynomials
            .iter()
            .map(|polynomial| polynomial.evaluate(&point))
            .collect::<Vec<_>>();

        let mut prover_transcript = Blake2bTranscript::new(b"batch-zk");
        let (proof, hiding_commitment, _blind) = <MockPCS as ZkBatchOpeningScheme>::prove_batch_zk(
            &(),
            &mut prover_transcript,
            &statement,
            &evals,
            &polynomials,
            vec![(); polynomials.len()],
        )
        .expect("ZK batch proof should be produced");

        let mut verifier_transcript = Blake2bTranscript::new(b"batch-zk");
        let result = <MockPCS as ZkBatchOpeningScheme>::verify_batch_zk(
            &(),
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("ZK batch proof should verify");

        assert_eq!(result.reduced_opening, hiding_commitment);
        assert_eq!(result.coefficients.len(), statement.claims.len());
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn homomorphic_zk_batch_opening_rejects_eval_count_mismatch() {
        let (polynomials, point) = batch_polynomials();
        let statement = zk_batch_statement(&polynomials, &point);
        let evals = [polynomials[0].evaluate(&point)];

        let mut transcript = Blake2bTranscript::new(b"batch-zk-mismatch");
        let result = <MockPCS as ZkBatchOpeningScheme>::prove_batch_zk(
            &(),
            &mut transcript,
            &statement,
            &evals,
            &polynomials,
            vec![(); polynomials.len()],
        );
        assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
    }
}
