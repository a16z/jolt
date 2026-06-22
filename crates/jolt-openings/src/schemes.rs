//! Polynomial commitment scheme (PCS) trait hierarchy.
//!
//! - [`CommitmentScheme`] — commit, open, verify for multilinear polynomials.
//! - [`AdditivelyHomomorphic`] — linear combination of commitments.
//! - [`StreamingCommitment`] — chunked commitment without full materialization.
//! - [`ZkOpeningScheme`] — zero-knowledge commitments and opening proofs.

use std::fmt::Debug;

use jolt_crypto::{Commitment, HomomorphicCommitment};
use jolt_field::Field;
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{de::DeserializeOwned, Serialize};

use crate::error::OpeningsError;

/// Optional metadata exposed by commitment backends that bind a layout digest
/// into the commitment.
pub trait CommitmentLayoutDigest {
    fn layout_digest(&self) -> Option<[u8; 32]>;
}

impl CommitmentLayoutDigest for u64 {
    fn layout_digest(&self) -> Option<[u8; 32]> {
        None
    }
}

/// Verifier statement for a same-point batch opening.
///
/// `logical_point` is the protocol point where the listed claims originated.
/// `pcs_point` is the single point opened by the backend after any embedding or
/// packed-view reduction. For direct views these are normally equal; packed
/// reductions may keep a logical point for transcript binding while opening the
/// packed commitment at the reduced PCS point.
///
/// `layout_digest` domain-separates the physical statement. Direct statements
/// should use the commitment's own layout digest when the backend exposes one;
/// packed statements should use the canonical packing layout digest that also
/// appears on every packed physical view. The digest binds metadata only: the
/// packed-view relation is proven by the packing reduction, not by this field
/// alone.
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
    /// Multiplier applied when embedding this claim into the batch PCS point.
    /// The verifier checks `claim * scale` against the physical opening.
    pub scale: F,
}

/// Physical commitment view used to satisfy a logical opening claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicalView<F> {
    Direct,
    PackedLinear {
        layout_digest: [u8; 32],
        terms: Vec<PackedLinearTerm<F>>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedLinearTerm<F> {
    pub coefficient: F,
    pub family: PackedFamilyRef,
    pub limb: usize,
    pub symbol: usize,
    pub row_point: Vec<F>,
}

impl<F> PackedLinearTerm<F> {
    pub fn new(coefficient: F, family: PackedFamilyRef, limb: usize, symbol: usize) -> Self {
        Self {
            coefficient,
            family,
            limb,
            symbol,
            row_point: Vec::new(),
        }
    }

    pub fn with_row_point(mut self, row_point: Vec<F>) -> Self {
        self.row_point = row_point;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackedFamilyRef {
    pub namespace: u64,
    pub id: u64,
    pub index: u64,
}

impl PackedFamilyRef {
    pub const fn new(namespace: u64, id: u64, index: u64) -> Self {
        Self {
            namespace,
            id,
            index,
        }
    }
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

    fn packed_term<F>(coefficient: F) -> PackedLinearTerm<F> {
        PackedLinearTerm::new(coefficient, PackedFamilyRef::new(0x6a6f_6c74, 1, 0), 0, 0)
    }

    fn packed_term_at<F>(coefficient: F, symbol: usize) -> PackedLinearTerm<F> {
        PackedLinearTerm::new(
            coefficient,
            PackedFamilyRef::new(0x6a6f_6c74, 1, 0),
            0,
            symbol,
        )
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
                        terms: vec![packed_term(29), packed_term_at(31, 1)],
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
    fn physical_view_records_packed_layout_and_terms() {
        let view = PhysicalView::PackedLinear {
            layout_digest: [41; 32],
            terms: vec![packed_term(43_u64), packed_term_at(47, 1)],
        };

        assert!(matches!(view, PhysicalView::PackedLinear { .. }));
        if let PhysicalView::PackedLinear {
            layout_digest,
            terms,
        } = view
        {
            assert_eq!(layout_digest, [41; 32]);
            assert_eq!(terms[0].coefficient, 43);
            assert_eq!(terms[1].coefficient, 47);
            assert_eq!(terms[1].symbol, 1);
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
            terms: vec![packed_term(fr(1)), packed_term_at(fr(2), 1)],
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
