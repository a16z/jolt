use jolt_crypto::{Bn254, Bn254G1};
use jolt_field::Fr;
use jolt_hyperkzg::error::HyperKZGError;
use jolt_hyperkzg::{HyperKZGCommitment, HyperKZGProof, VariableBatchKzgProof, VerifierObserver};
use jolt_openings::OpeningsError;
use jolt_r1cs::ConstraintMatrixEvalError;
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::{CompressedSumcheckProof, SumcheckError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Commitment = HyperKZGCommitment<Bn254>;
pub type OpeningProof = HyperKZGProof<Bn254>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VerifierCost {
    pub ec_mul: usize,
    pub ec_add: usize,
    pub pairing_pairs: usize,
    pub fr_mul: usize,
    pub fr_inv: usize,
    pub keccak: usize,
}

impl VerifierObserver for VerifierCost {
    fn ec_mul(&mut self, count: usize) {
        self.ec_mul += count;
    }

    fn ec_add(&mut self, count: usize) {
        self.ec_add += count;
    }

    fn pairing_pairs(&mut self, count: usize) {
        self.pairing_pairs += count;
    }

    fn record_fr_mul(&mut self) {
        self.fr_mul += 1;
    }

    fn record_fr_inv(&mut self) {
        self.fr_inv += 1;
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageProof {
    pub round_polynomials: CompressedSumcheckProof<Fr>,
    pub committed_rounds: Option<CommittedStageProof>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedStageProof {
    pub round_commitments: Vec<Bn254G1>,
    pub round_claims: Vec<Fr>,
    pub sum_at_zero: Fr,
    pub opening: VariableBatchKzgProof<Bn254>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WrapperProof {
    pub public_challenges: Vec<[u8; 16]>,
    pub commitments: Vec<Commitment>,
    pub stages: Vec<StageProof>,
    pub stage_claims: Vec<Vec<Fr>>,
    pub reduced_claims: Vec<Fr>,
    pub opening: OpeningProof,
}

impl WrapperProof {
    /// Scalar/group payload bytes, excluding serde container-length prefixes.
    pub fn payload_bytes(&self) -> usize {
        let round_scalars = self
            .stages
            .iter()
            .flat_map(|stage| &stage.round_polynomials.round_polynomials)
            .map(|round| round.coeffs_except_linear_term().len())
            .sum::<usize>();
        let committed_groups = self
            .stages
            .iter()
            .filter_map(|stage| stage.committed_rounds.as_ref())
            .map(|stage| stage.round_commitments.len() + 3)
            .sum::<usize>();
        let committed_scalars = self
            .stages
            .iter()
            .filter_map(|stage| stage.committed_rounds.as_ref())
            .map(|stage| stage.round_claims.len() + 1)
            .sum::<usize>();
        let opening_scalars = self.opening.v.iter().map(Vec::len).sum::<usize>() + 1;
        let stage_claims = self.stage_claims.iter().map(Vec::len).sum::<usize>();
        16 * self.public_challenges.len()
            + 32 * (self.commitments.len()
                + round_scalars
                + committed_groups
                + committed_scalars
                + stage_claims
                + self.reduced_claims.len()
                + self.opening.com.len()
                + 1
                + opening_scalars)
    }

    /// Exact `bincode::config::standard()` size for this proof's serde shape.
    pub fn bincode_bytes(&self) -> usize {
        let stage_prefixes = self
            .stages
            .iter()
            .map(|stage| {
                varint_bytes(stage.round_polynomials.round_polynomials.len())
                    + stage
                        .round_polynomials
                        .round_polynomials
                        .iter()
                        .map(|round| varint_bytes(round.coeffs_except_linear_term().len()))
                        .sum::<usize>()
            })
            .sum::<usize>();
        let committed_prefixes = self
            .stages
            .iter()
            .map(|stage| match &stage.committed_rounds {
                Some(committed) => {
                    varint_bytes(committed.round_commitments.len())
                        + committed.round_commitments.len() * varint_bytes(32)
                        + varint_bytes(committed.round_claims.len())
                        + 3 * varint_bytes(32)
                }
                None => 0,
            })
            .sum::<usize>();
        self.payload_bytes()
            + varint_bytes(self.public_challenges.len())
            + (self.commitments.len() + self.opening.com.len() + 1) * varint_bytes(32)
            + varint_bytes(self.commitments.len())
            + varint_bytes(self.stages.len())
            + stage_prefixes
            + self.stages.len()
            + committed_prefixes
            + varint_bytes(self.stage_claims.len())
            + self
                .stage_claims
                .iter()
                .map(|claims| varint_bytes(claims.len()))
                .sum::<usize>()
            + varint_bytes(self.reduced_claims.len())
            + varint_bytes(self.opening.com.len())
            + self
                .opening
                .v
                .iter()
                .map(|values| varint_bytes(values.len()))
                .sum::<usize>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageMemberSpec {
    pub rounds: usize,
    pub degree: usize,
    pub offset: usize,
}

pub struct StageMember<'a> {
    pub prover: &'a mut dyn ProveRounds<Fr>,
    pub input_claim: Fr,
    pub degree: usize,
    pub offset: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageResult {
    pub point: Vec<Fr>,
    pub coefficients: Vec<Fr>,
    pub output_claims: Vec<Fr>,
    pub final_claim: Fr,
}

impl StageResult {
    pub fn member_point(
        &self,
        member: usize,
        specs: &[StageMemberSpec],
    ) -> Result<&[Fr], StreamError> {
        let spec = specs.get(member).ok_or(StreamError::StageMemberCount)?;
        let end = spec
            .offset
            .checked_add(spec.rounds)
            .ok_or(StreamError::StageMemberCount)?;
        self.point
            .get(spec.offset..end)
            .ok_or(StreamError::StageMemberCount)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReductionClaim {
    pub polynomial_weights: Vec<Fr>,
    pub point: Vec<Fr>,
    pub value: Fr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorTerm {
    pub coefficient: Fr,
    pub columns: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorStreamStatement {
    pub key_digest: [u8; 32],
    pub rows: usize,
    pub column_count: usize,
    pub k: usize,
    pub row_input_claim: Fr,
    pub row_degree: usize,
    pub stage_a_encoding: StageAEncoding,
    pub terms: Vec<TensorTerm>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssemblyMemberStatement {
    pub input_claim: Fr,
    pub spec: StageMemberSpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColumnId {
    pub group: usize,
    pub slot: usize,
}

impl ColumnId {
    pub fn index(self, packing: usize) -> Result<usize, StreamError> {
        if self.slot >= packing {
            return Err(StreamError::ColumnOutOfRange {
                column: self.slot,
                columns: packing,
            });
        }
        self.group
            .checked_mul(packing)
            .and_then(|base| base.checked_add(self.slot))
            .ok_or(StreamError::PackedLengthOverflow)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineForm {
    pub constant: Fr,
    pub weights: Vec<(ColumnId, Fr)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    pub coefficient: Fr,
    pub factors: Vec<AffineForm>,
}

pub struct TermContext<'a> {
    pub row_point: &'a [Fr],
    pub batching_coefficients: &'a [Fr],
    pub challenges: &'a [Fr],
}

pub trait TermExporter {
    fn terms(&self, context: &TermContext<'_>) -> Vec<Term>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssemblyStatement {
    pub key_digest: [u8; 32],
    pub public_inputs: Vec<Fr>,
    pub rows: usize,
    pub column_count: usize,
    pub k: usize,
    pub members: Vec<AssemblyMemberStatement>,
    /// Column evaluations at stage A's row point needed by the member-final
    /// checks and reduced by stage B into the final packed opening.
    pub factor_columns: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageAEncoding {
    Compressed,
    KzgCommitted,
}

#[derive(Debug, Error)]
pub enum StreamError {
    #[error("no columns")]
    NoColumns,
    #[error("column packing factor must be a nonzero power of two, got {0}")]
    InvalidPacking(usize),
    #[error("column {column} has {actual} rows, expected {expected}")]
    RowCount {
        column: usize,
        expected: usize,
        actual: usize,
    },
    #[error("bit column {column} contains {value} at row {row}")]
    InvalidBit {
        column: usize,
        row: usize,
        value: u8,
    },
    #[error("SRS has {actual} powers, need {required}")]
    SetupTooSmall { required: usize, actual: usize },
    #[error("row count times packing factor overflows usize")]
    PackedLengthOverflow,
    #[error("point has {actual} variables, expected {expected}")]
    PointDimension { expected: usize, actual: usize },
    #[error("stage must contain at least one member")]
    EmptyStage,
    #[error("stage proof member count mismatch")]
    StageMemberCount,
    #[error("stage output claim mismatch")]
    StageOutputClaim,
    #[error("stage padding scale is not invertible")]
    StageScale,
    #[error("stage proof encoding does not match the statement")]
    StageEncoding,
    #[error("column tensor must contain at least one factor")]
    EmptyTensor,
    #[error("column tensor term {term} has arity {actual}, expected {expected}")]
    TensorArity {
        term: usize,
        expected: usize,
        actual: usize,
    },
    #[error("column index {column} is out of range for {columns} columns")]
    ColumnOutOfRange { column: usize, columns: usize },
    #[error("stream shape/proof stage count mismatch")]
    StageCount,
    #[error("stage A output does not equal stage B input")]
    StageLink,
    #[error("opening claim mismatch")]
    OpeningClaim,
    #[error("sumcheck: {0}")]
    Sumcheck(#[from] SumcheckError<Fr>),
    #[error("commitment failed: {0}")]
    Commitment(#[from] OpeningsError),
    #[error("relation check failed: {0}")]
    Relation(#[from] ConstraintMatrixEvalError),
    #[error("HyperKZG: {0}")]
    HyperKzg(#[from] HyperKZGError),
}

fn varint_bytes(value: usize) -> usize {
    match value {
        0..=250 => 1,
        251..=0xffff => 3,
        0x1_0000..=0xffff_ffff => 5,
        _ => 9,
    }
}
