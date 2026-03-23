//! Core types for the protocol graph.
//!
//! These types model the claim dependency DAG of the Jolt SNARK at three levels:
//! - **ClaimGraph** (invariant): polynomials, claims, vertices — the structural truth
//! - **Staging** (choice): how vertices are layered into Fiat-Shamir interactions
//! - **CommitmentStrategy** (choice): how committed polys are grouped and transcript-bound

use std::collections::HashMap;

use crate::claim::ClaimDefinition;
use crate::PolynomialId;

use super::symbolic::{NumVars, SymbolicExpr};

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

macro_rules! id_type {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
        pub struct $name(pub u32);

        impl From<u32> for $name {
            fn from(v: u32) -> Self {
                Self(v)
            }
        }

        impl From<usize> for $name {
            fn from(v: usize) -> Self {
                Self(v as u32)
            }
        }
    };
}

id_type!(ClaimId, "Unique identifier for a claim in the graph.");
id_type!(VertexId, "Unique identifier for a vertex in the graph.");
id_type!(StageId, "Unique identifier for a stage in the staging.");
id_type!(
    CommitmentGroupId,
    "Unique identifier for a commitment group."
);

// ---------------------------------------------------------------------------
// Polynomials
// ---------------------------------------------------------------------------

/// A polynomial in the protocol.
///
/// Committed polynomials have PCS data; virtual polynomials are derived during
/// proving. The `num_vars` determines the evaluation domain size `2^num_vars`.
#[derive(Clone, Debug)]
pub struct Polynomial {
    pub id: PolynomialId,
    pub kind: PolynomialKind,
    pub num_vars: NumVars,
}

/// Whether a polynomial is committed (PCS-backed) or virtual (derived).
#[derive(Clone, Debug)]
pub enum PolynomialKind {
    /// Has a PCS commitment. Must be discharged by a terminal Opening vertex.
    /// The `group` determines which other polynomials this one is committed
    /// alongside — affecting proof structure but not claim flow.
    Committed { group: CommitmentGroupId },
    /// Derived from other data (R1CS witness, trace, composition).
    /// Claims on virtual polys are proven by sumchecks but don't require PCS openings.
    Virtual,
}

// ---------------------------------------------------------------------------
// Commitment groups
// ---------------------------------------------------------------------------

/// A set of polynomials committed together as a single prover message.
///
/// Grouping is a choice — it affects proof structure and opening batch
/// overhead, but not the claim dependency graph.
#[derive(Clone, Debug)]
pub struct CommitmentGroup {
    pub id: CommitmentGroupId,
    pub polynomials: Vec<PolynomialId>,
}

// ---------------------------------------------------------------------------
// Claims
// ---------------------------------------------------------------------------

/// A claim: "polynomial P evaluates to v at point r."
///
/// Claims are the edges of the protocol graph. They're produced by vertices
/// and depended on by downstream vertices.
#[derive(Clone, Debug)]
pub struct Claim {
    pub id: ClaimId,
    pub polynomial: PolynomialId,
    /// Symbolic evaluation point (composed from stage challenge outputs).
    pub point: SymbolicPoint,
}

/// A symbolic evaluation point — composed from challenge outputs of stages.
///
/// All evaluation points in the protocol are compositions of stage challenge
/// points. The graph makes this explicit and dimensionally checkable.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum SymbolicPoint {
    /// The challenge vector output by a stage's Fiat-Shamir interaction.
    Challenges(StageId),
    /// Concatenation: `[point_a || point_b]`. Used when claim reductions
    /// produce a multi-dimensional point (e.g., unified = [r_addr || r_cycle]).
    Concat(Vec<SymbolicPoint>),
    /// A sub-range of another point. Used when Spartan's `r_y` is sliced to
    /// extract `r_cycle = r_y[..log_T]`.
    Slice {
        source: Box<SymbolicPoint>,
        range: VarRange,
    },
}

/// A range of variable indices within a challenge point.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct VarRange {
    pub start: SymbolicExpr,
    pub end: SymbolicExpr,
}

// ---------------------------------------------------------------------------
// Vertices
// ---------------------------------------------------------------------------

/// A vertex in the protocol graph — an atomic proof step.
#[derive(Clone, Debug)]
pub enum Vertex {
    Sumcheck(Box<SumcheckVertex>),
    PointNormalization(PointNormalizationVertex),
    Opening(OpeningVertex),
}

impl Vertex {
    pub fn id(&self) -> VertexId {
        match self {
            Self::Sumcheck(v) => v.id,
            Self::PointNormalization(v) => v.id,
            Self::Opening(v) => v.id,
        }
    }

    /// All claims this vertex depends on (incoming edges).
    pub fn dep_claims(&self) -> &[ClaimId] {
        match self {
            Self::Sumcheck(v) => &v.deps,
            Self::PointNormalization(v) => &v.consumes,
            Self::Opening(v) => std::slice::from_ref(&v.consumes),
        }
    }

    /// All claims this vertex produces (outgoing edges).
    pub fn produced_claims(&self) -> &[ClaimId] {
        match self {
            Self::Sumcheck(v) => &v.produces,
            Self::PointNormalization(v) => &v.produces,
            Self::Opening(_) => &[],
        }
    }
}

/// A single sumcheck instance.
///
/// One vertex, one set of upstream dependencies, one set of leaf claims
/// produced at the stage's challenge point.
#[derive(Clone, Debug)]
pub struct SumcheckVertex {
    pub id: VertexId,
    /// Upstream claims whose eval values this vertex depends on.
    ///
    /// Determines dependency edges for topological ordering.
    /// A claim can appear in multiple vertices' deps (fan-out).
    pub deps: Vec<ClaimId>,
    /// How the input claimed sum is derived from upstream evals.
    pub input: InputClaim,
    /// Leaf claims produced at the stage's challenge point.
    pub produces: Vec<ClaimId>,
    /// The output claim formula (verifier check expression).
    pub formula: ClaimFormula,
    /// Degree of the sumcheck round polynomial.
    pub degree: usize,
    /// Number of sumcheck variables.
    pub num_vars: NumVars,
    /// Public polynomial used for weighting.
    pub weighting: PublicPolynomial,
    /// Variable-binding phases (e.g., address then cycle for multi-phase sumchecks).
    /// Single-phase vertices have one entry.
    pub phases: Vec<Phase>,
}

/// How the input claimed sum is derived.
#[derive(Clone, Debug)]
pub enum InputClaim {
    /// `claimed_sum = f(deps evals, challenges)`. The formula references
    /// eval values from `deps` claims via `OpeningBinding`s.
    Formula {
        formula: ClaimFormula,
        /// How to resolve each challenge variable in the formula.
        /// Indexed by `Var::Challenge(var_id)` — the graph-driven verifier
        /// uses these to map formula challenges to concrete squeezed values.
        challenge_labels: Vec<ChallengeLabel>,
    },
    /// `claimed_sum` is a known constant (e.g., zero for booleanity checks).
    Constant(i64),
}

/// Where a formula's challenge variable gets its value.
#[derive(Clone, Debug)]
pub enum ChallengeLabel {
    /// Resolved from the stage's `pre_squeeze` spec with this label.
    /// For `Scalar` specs, the value is the squeezed scalar.
    /// For `GammaPowers` specs, the value is the base gamma.
    PreSqueeze(&'static str),
    /// Must be supplied externally by the verifier (e.g., public-input-derived
    /// values like initial RAM evaluation).
    External(&'static str),
}

/// Wraps a [`ClaimDefinition`] with graph-level binding metadata.
///
/// The `opening_claims` map formula variable indices to [`ClaimId`]s,
/// replacing the opaque `(polynomial_tag, sumcheck_tag)` pairs in the
/// underlying `ClaimDefinition`.
#[derive(Clone, Debug)]
pub struct ClaimFormula {
    pub definition: ClaimDefinition,
    /// Maps formula `Var::Opening(var_id)` to the upstream [`ClaimId`]
    /// whose eval value it reads.
    pub opening_claims: HashMap<u32, ClaimId>,
}

/// Public polynomial used as weighting in a sumcheck vertex.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PublicPolynomial {
    /// Standard equality polynomial: `eq(r, x)`.
    Eq,
    /// Successor equality polynomial: `eq+1(r, x)`. Used by shift.
    EqPlusOne,
    /// Less-than polynomial: `LT(r, x)`. Used by RegistersValEval.
    Lt,
    /// Precomputed from prior challenge values. The vertex's formula
    /// encodes how to compute it; the prover materializes it.
    Derived,
}

/// A variable-binding phase within a multi-phase sumcheck.
#[derive(Clone, Debug)]
pub struct Phase {
    pub num_vars: NumVars,
    pub variable_group: VariableGroup,
}

/// Which variable group a phase binds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariableGroup {
    /// Cycle dimension (log_T variables).
    Cycle,
    /// Address dimension (log_k variables).
    Address,
}

/// Transforms claims from a short point to a longer point via Lagrange
/// zero-selector scaling.
///
/// `eval_at_unified = eval_at_short × ∏(1 − r_extra_i)`
///
/// Not a proof step — a mathematical identity that the verifier checks.
#[derive(Clone, Debug)]
pub struct PointNormalizationVertex {
    pub id: VertexId,
    pub consumes: Vec<ClaimId>,
    pub produces: Vec<ClaimId>,
    /// The extra dimensions to zero-pad (source of r_extra values).
    pub padding_source: SymbolicPoint,
}

/// Terminal vertex: discharges a committed polynomial claim via PCS.
#[derive(Clone, Debug)]
pub struct OpeningVertex {
    pub id: VertexId,
    pub consumes: ClaimId,
}

// ---------------------------------------------------------------------------
// Stages
// ---------------------------------------------------------------------------

/// A stage: a partition of vertices sharing a single Fiat-Shamir interaction.
///
/// All sumcheck vertices in a stage produce leaf claims at the same challenge
/// point. Vertices within a stage are independent (antichain property).
#[derive(Clone, Debug)]
pub struct Stage {
    pub id: StageId,
    pub vertices: Vec<VertexId>,
    /// The random challenge point produced by this stage.
    pub challenge_point: ChallengePoint,
    /// How vertices are batched into proof artifacts.
    pub batching: Vec<BatchGroup>,
    /// Challenges squeezed from the transcript before the sumcheck runs.
    pub pre_squeeze: Vec<ChallengeSpec>,
}

/// The random challenge point produced by a stage — `num_vars` random field
/// elements derived from the transcript after the prover sends round polys.
#[derive(Clone, Debug)]
pub struct ChallengePoint {
    pub num_vars: NumVars,
}

/// A group of vertices batched under a shared α coefficient.
#[derive(Clone, Debug)]
pub struct BatchGroup {
    pub vertices: Vec<VertexId>,
}

/// Specification of a challenge squeeze operation.
#[derive(Clone, Debug)]
pub enum ChallengeSpec {
    /// Squeeze one scalar from the transcript.
    Scalar { label: &'static str },
    /// Squeeze a vector of `dim` scalars.
    Vector {
        label: &'static str,
        dim: SymbolicExpr,
    },
    /// Squeeze one scalar and compute `n` powers: `[1, γ, γ², ..., γ^{n-1}]`.
    GammaPowers {
        label: &'static str,
        count: SymbolicExpr,
    },
}

/// Terminal stage: discharges all remaining committed polynomial claims via PCS.
#[derive(Clone, Debug)]
pub struct OpeningStage {
    pub vertices: Vec<VertexId>,
    /// The evaluation point shared by all claims (or the primary point when
    /// the PCS internalizes multiple points, as with Dory matrix layout).
    pub point: SymbolicPoint,
    pub reduction: ReductionStrategy,
    pub opening_groups: Vec<OpeningGroup>,
}

/// A batch of claims opened together in one PCS proof.
#[derive(Clone, Debug)]
pub struct OpeningGroup {
    pub vertices: Vec<VertexId>,
    pub source_groups: Vec<CommitmentGroupId>,
}

/// How opening claims are combined before PCS proving.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReductionStrategy {
    /// Random linear combination — standard when all claims share the same point.
    Rlc,
}

// ---------------------------------------------------------------------------
// Top-level graph types
// ---------------------------------------------------------------------------

/// The invariant claim dependency structure — what must be proven.
///
/// Determined entirely by the SNARK's polynomial identities. Does not change
/// when you re-stage, re-batch, or re-order the proof.
#[derive(Clone, Debug)]
pub struct ClaimGraph {
    pub polynomials: Vec<Polynomial>,
    pub claims: Vec<Claim>,
    pub vertices: Vec<Vertex>,
}

impl ClaimGraph {
    pub fn vertex(&self, id: VertexId) -> &Vertex {
        &self.vertices[id.0 as usize]
    }

    pub fn claim(&self, id: ClaimId) -> &Claim {
        &self.claims[id.0 as usize]
    }

    pub fn polynomial(&self, id: PolynomialId) -> Option<&Polynomial> {
        self.polynomials.iter().find(|p| p.id == id)
    }
}

/// A particular scheduling of the claim graph into Fiat-Shamir interactions.
#[derive(Clone, Debug)]
pub struct Staging {
    pub stages: Vec<Stage>,
    pub opening: OpeningStage,
}

/// How committed polynomials are grouped and ordered for the initial
/// commitment phase.
#[derive(Clone, Debug)]
pub struct CommitmentStrategy {
    pub groups: Vec<CommitmentGroup>,
    /// Transcript ordering: groups are appended to the transcript in this
    /// order before any sumcheck stage executes.
    pub transcript_order: Vec<CommitmentGroupId>,
}

/// The complete protocol specification: invariant structure + scheduling choices.
///
/// Both prover and verifier derive from the same `ProtocolGraph`. The prover
/// walks the staging forward (executing sumchecks). The verifier walks the
/// same staging forward (replaying Fiat-Shamir, checking claims).
#[derive(Clone, Debug)]
pub struct ProtocolGraph {
    pub claim_graph: ClaimGraph,
    pub staging: Staging,
    pub commitment: CommitmentStrategy,
}
