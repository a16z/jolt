//! Stateless claim and result types for PCS operations.

use jolt_field::Field;
use jolt_poly::Polynomial;

use crate::schemes::{CommitmentSchemeVerifier, ZkOpeningScheme};

/// Prover-side opening claim: polynomial, evaluation point, and claimed value.
#[derive(Clone, Debug)]
pub struct ProverClaim<F: Field, P = Polynomial<F>> {
    pub polynomial: P,
    pub point: Vec<F>,
    pub eval: F,
}

/// Verifier-side opening claim: commitment, point, and claimed value.
#[derive(Clone, Debug)]
pub struct OpeningClaim<F, PCS>
where
    F: Field,
    PCS: CommitmentSchemeVerifier<Field = F>,
{
    pub commitment: PCS::Output,
    pub point: Vec<F>,
    pub eval: F,
}

/// Opening point pair for protocols whose public point order differs from the
/// backend's proof point order.
///
/// `public` is the point that belongs to the surrounding protocol transcript
/// and output relation. `proof` is the coordinate order consumed by the PCS
/// proof algorithm. Most schemes set both fields equal; Dory uses `proof` for
/// its private row-major opening order while Jolt keeps `public` in protocol
/// order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchOpeningPoint<F: Field> {
    pub public: Vec<F>,
    pub proof: Vec<F>,
}

impl<F: Field> BatchOpeningPoint<F> {
    /// Builds a point whose public and proof coordinates are identical.
    pub fn same(point: Vec<F>) -> Self {
        Self {
            public: point.clone(),
            proof: point,
        }
    }
}

/// Prover-side raw term in a source-backed batch opening.
///
/// `eval` is the source's raw evaluation at `point.public`. `eval_scale`
/// describes how that raw evaluation contributes to the PCS output relation.
/// For current Jolt/Dory Stage 8, dense and advice sources use nontrivial
/// embedding factors while RA sources use one.
#[derive(Clone, Debug)]
pub struct ProverBatchOpeningTerm<F: Field, ClaimId, SourceId> {
    pub claim_id: ClaimId,
    pub source_id: SourceId,
    pub point: BatchOpeningPoint<F>,
    pub eval: F,
    pub eval_scale: F,
}

/// Verifier-side raw term in a source-backed batch opening.
#[derive(Clone, Debug)]
pub struct VerifierBatchOpeningTerm<F, PCS, ClaimId, SourceId>
where
    F: Field,
    PCS: CommitmentSchemeVerifier<Field = F>,
{
    pub claim_id: ClaimId,
    pub source_id: SourceId,
    pub commitment: PCS::Output,
    pub point: BatchOpeningPoint<F>,
    pub eval: F,
    pub eval_scale: F,
}

/// A linear coefficient applied to a committed source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearSourceTerm<F: Field, SourceId> {
    pub source_id: SourceId,
    pub coefficient: F,
}

/// The value opened by a source-backed batch-opening proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchOutputValue<F: Field, HidingCommitment> {
    Public(F),
    Hidden(HidingCommitment),
}

impl<F: Field, HidingCommitment> BatchOutputValue<F, HidingCommitment> {
    /// Returns the scalar when this is a transparent output.
    pub fn as_public(&self) -> Option<&F> {
        match self {
            Self::Public(value) => Some(value),
            Self::Hidden(_) => None,
        }
    }

    /// Returns the hiding commitment when this is a ZK output.
    pub fn as_hidden(&self) -> Option<&HidingCommitment> {
        match self {
            Self::Public(_) => None,
            Self::Hidden(commitment) => Some(commitment),
        }
    }
}

/// One output created by a source-backed batch-opening proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenedBatchOutput<F: Field, HidingCommitment> {
    pub point: Vec<F>,
    pub value: BatchOutputValue<F, HidingCommitment>,
}

/// Expression describing how a PCS output is derived from raw claim terms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchOutputExpression<F: Field, ClaimId> {
    /// Linear relation `output = Σ coefficient_i * claim_i`.
    Linear(Vec<(ClaimId, F)>),
}

/// Relation between one opened PCS output and the raw claims supplied by the
/// surrounding protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchOutputRelation<F: Field, ClaimId> {
    pub output_index: usize,
    pub expression: BatchOutputExpression<F, ClaimId>,
}

impl<F: Field, ClaimId> BatchOutputRelation<F, ClaimId> {
    /// Returns the linear terms when this is a linear output relation.
    pub fn linear_terms(&self) -> Option<&[(ClaimId, F)]> {
        match &self.expression {
            BatchOutputExpression::Linear(terms) => Some(terms),
        }
    }
}

/// Public metadata returned by source-backed batch opening.
///
/// The proof itself remains `PCS::BatchProof`; this structure tells protocol
/// code which output values the PCS opened and how those outputs relate to the
/// raw claims it supplied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchOpeningPublic<F: Field, HidingCommitment, ClaimId> {
    pub outputs: Vec<OpenedBatchOutput<F, HidingCommitment>>,
    pub relations: Vec<BatchOutputRelation<F, ClaimId>>,
}

impl<F: Field, HidingCommitment, ClaimId> BatchOpeningPublic<F, HidingCommitment, ClaimId> {
    /// Returns the only linear relation when the batch opening produced exactly
    /// one linear output relation.
    pub fn single_linear_relation(&self) -> Option<&BatchOutputRelation<F, ClaimId>> {
        let [relation] = self.relations.as_slice() else {
            return None;
        };
        Some(relation)
    }
}

/// Prover-only witnesses for ZK source-backed batch-opening outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkBatchOpeningWitness<F: Field, Blind> {
    pub output_values: Vec<F>,
    pub output_blinds: Vec<Blind>,
}

/// Transparent prover result for source-backed batch opening.
#[derive(Clone, Debug)]
pub struct BatchOpeningProverResult<PCS, ClaimId>
where
    PCS: CommitmentSchemeVerifier,
{
    pub proof: PCS::BatchProof,
    pub public: BatchOpeningPublic<PCS::Field, (), ClaimId>,
}

/// ZK prover result for source-backed batch opening.
#[derive(Clone, Debug)]
pub struct ZkBatchOpeningProverResult<PCS, ClaimId>
where
    PCS: ZkOpeningScheme,
{
    pub proof: PCS::BatchProof,
    pub public: BatchOpeningPublic<PCS::Field, PCS::HidingCommitment, ClaimId>,
    pub witness: ZkBatchOpeningWitness<PCS::Field, PCS::Blind>,
}
