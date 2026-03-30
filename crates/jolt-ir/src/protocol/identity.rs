//! Polynomial identities — the pure math layer of the protocol.
//!
//! A [`PolynomialIdentity`] declares WHAT the SNARK proves, with zero
//! scheduling concerns. The compiler transforms identities into the
//! claim graph (executable protocol) via scheduling hints or cost models.
//!
//! This is Level 0 of the progressive lowering pipeline:
//! ```text
//! Protocol (~20 identities)  ← this module
//!     ↓ compiler
//! Claim Graph (~50-100 nodes)
//!     ↓ staging
//! Staged Graph (ProtocolGraph)
//! ```
//!
//! # Design
//!
//! Each identity is a sumcheck statement:
//!
//! ```text
//! Σ_{x ∈ {0,1}^n} w(r,x) · C(f₁(x), ..., fₖ(x)) = claimed_sum
//! ```
//!
//! Where `w` is a public weighting polynomial (eq, eq+1, lt), `C` is the
//! composition formula, and `claimed_sum` is zero, a constant, or a value
//! derived from predecessor identities' opening evaluations.
//!
//! Each identity declares the full set of polynomial evaluations it produces
//! (`produces`). The output formula is encoded as a [`ClaimDefinition`] whose
//! `opening_bindings` reference a subset of those polynomials.

use crate::claim::ClaimDefinition;
use crate::PolynomialId;

use super::symbolic::SymbolicExpr;

/// Unique identifier for a polynomial identity.
///
/// Used as key in scheduling hints to map identities to stages.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct IdentityId(pub u32);

/// A polynomial identity that the SNARK must prove.
///
/// Each identity mechanically produces one sumcheck instance when lowered.
/// The formula, domain, and claim type are fixed at protocol definition
/// time. Scheduling (stage assignment, batching, weighting choice) is
/// determined by the compiler, not here.
#[derive(Clone, Debug)]
pub struct PolynomialIdentity {
    /// Unique identifier.
    pub id: IdentityId,
    /// Human-readable name — used in scheduling hints and diagnostics.
    pub name: &'static str,
    /// All polynomial evaluations this identity produces at its challenge point.
    ///
    /// This is the primary declaration — every polynomial the identity "opens"
    /// must appear here. The formula's `opening_bindings` reference a subset of
    /// these (the polynomials that participate in the composition). The rest are
    /// additional evaluations that downstream identities can consume.
    pub produces: Vec<PolynomialId>,
    /// The output composition formula: what gets summed.
    ///
    /// `output.opening_bindings` identifies which polynomials participate
    /// in the formula. These must be a subset of `produces`.
    pub output: ClaimDefinition,
    /// What the sum equals (input claim).
    pub input: IdentityClaim,
    /// Degree of the sumcheck round polynomial.
    ///
    /// Equals the maximum product degree of the composition formula plus one
    /// for the eq/weighting polynomial. For example:
    /// - `f² - f` (booleanity) → degree 3 (product of 2 inputs + eq)
    /// - `γ·f` (claim reduction) → degree 2 (1 input × challenge-constant + eq)
    /// - `∏ᵢ fᵢ` (d-way product) → degree d+1
    pub degree: usize,
    /// Domain: which variables are summed over.
    pub domain: DomainSpec,
}

/// What a polynomial identity's sum equals (the input claim).
#[derive(Clone, Debug)]
pub enum IdentityClaim {
    /// Sum is zero (zero-check). Booleanity, output checks.
    Zero,

    /// Sum is a constant known at protocol definition time.
    Constant(i64),

    /// Sum equals the same formula as the output, evaluated over openings
    /// from a predecessor identity at its challenge point.
    ///
    /// This is the standard claim-reduction pattern: the output formula is
    /// evaluated at the new point (producing fresh openings), and the input
    /// is the same formula applied to the predecessor's openings. A fresh
    /// γ-challenge is squeezed for the RLC batching.
    ///
    /// Mechanically: `input_claim = output.formula(predecessor_openings, γ)`
    /// where `predecessor_openings[i]` maps to the same polynomials as
    /// `output.opening_bindings[i]` but at the predecessor's point.
    Reduction {
        /// Challenge label for the γ-power RLC.
        gamma_label: &'static str,
    },

    /// Sum equals a formula evaluated over predecessor openings.
    ///
    /// For cases where the input formula differs from the output formula
    /// (e.g., RAM read-write checking: input is `rv + γ·wv`, output is
    /// `c0·ra·val + c1·ra·inc`).
    Predecessor(PredecessorClaim),
}

/// Input claim derived from a formula over predecessor openings.
#[derive(Clone, Debug)]
pub struct PredecessorClaim {
    /// Formula computing the claimed sum from predecessor openings.
    ///
    /// `formula.opening_bindings` identifies which predecessor polynomials
    /// feed the input claim. These must be available from earlier stages.
    pub formula: ClaimDefinition,
    /// Challenge labels for the formula's challenge slots.
    pub challenge_labels: Vec<ChallengeLabel>,
    /// Explicit source identity for opening bindings that would otherwise
    /// be ambiguous.
    ///
    /// When the same polynomial is produced by multiple upstream identities
    /// (e.g., `ExpandedPc` at S1 and S3), the compiler needs to know which
    /// evaluation to wire. Each entry maps a formula `var_id` to the source
    /// `IdentityId` whose produced claim should be used.
    ///
    /// Bindings not listed here use the default resolution (most recent
    /// producer of that polynomial).
    pub source_bindings: Vec<(u32, IdentityId)>,
}

/// How a challenge value is obtained.
#[derive(Clone, Debug)]
pub enum ChallengeLabel {
    /// Freshly squeezed from the Fiat-Shamir transcript.
    PreSqueeze(&'static str),
    /// Derived from prior protocol state (e.g., a scaling factor
    /// computed from previous round parameters).
    External(&'static str),
}

/// Domain specification: which variables the sum ranges over.
#[derive(Clone, Debug)]
pub enum DomainSpec {
    /// Sum over {0,1}^n where n = log2(trace_length). The standard case.
    TraceLength,
    /// Sum over {0,1}^n where n = log2(trace_length) + log2(address_space).
    /// Used by RA polynomial sumchecks that operate in the product domain.
    TraceTimesAddress,
    /// Sum over {0,1}^n where n = log2(address_space). Used by claim reductions
    /// in the address dimension only.
    AddressLength,
    /// Sum over {0,1}^n where n is a symbolic expression of config parameters.
    Symbolic(SymbolicExpr),
}

// ---------------------------------------------------------------------------
// Scheduling hints
// ---------------------------------------------------------------------------

/// Scheduling hints for the compiler.
///
/// Carries exactly the decisions currently baked into `build_jolt_protocol()`,
/// extracted into a data structure. The compiler validates hints against
/// Fiat-Shamir correctness — an invalid hint is rejected, never silently used.
///
/// Migration path:
/// - Phase A (now): hints required, reproduce current hand-tuned behavior
/// - Phase B (later): hints optional, compiler can derive some decisions
/// - Phase C (future): hints deleted, cost model drives all scheduling
#[derive(Clone, Debug, Default)]
pub struct SchedulingHints {
    /// Which identities share a Fiat-Shamir epoch (stage).
    /// `(identity_id, stage_index)` — stage indices are 0-based.
    pub stage_assignment: Vec<(IdentityId, u32)>,

    /// Which sumchecks within a stage batch together under a shared RLC.
    /// `(stage_index, batch_groups)` where each batch group is a list of
    /// identity IDs that share a single batched sumcheck invocation.
    pub batch_groups: Vec<(u32, Vec<Vec<IdentityId>>)>,

    /// Which polynomials commit together and in what transcript order.
    pub commitment_groups: Vec<Vec<PolynomialId>>,

    /// Opening reduction groups (claims that RLC together into one PCS opening).
    pub opening_groups: Vec<Vec<PolynomialId>>,

    /// Per-identity scheduling metadata.
    pub identity_meta: Vec<(IdentityId, IdentityMeta)>,
}

/// Per-identity scheduling metadata that the compiler cannot yet derive.
#[derive(Clone, Debug)]
pub struct IdentityMeta {
    /// How the eq polynomial weights this sumcheck.
    pub weighting: WeightingHint,
    /// Multi-phase decomposition. None = single-phase (most common).
    pub phases: Option<Vec<PhaseHint>>,
}

/// Hint for which public weighting polynomial to use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightingHint {
    /// Standard eq(r, x).
    Eq,
    /// Successor eq+1(r, x).
    EqPlusOne,
    /// Less-than LT(r, x).
    Lt,
    /// Derived from prior challenges (e.g., pre-combined linear combination).
    Derived,
}

/// Hint for a variable-binding phase within a multi-phase sumcheck.
#[derive(Clone, Debug)]
pub struct PhaseHint {
    /// Number of variables bound in this phase.
    pub num_vars: SymbolicExpr,
    /// Whether this phase binds cycle or address variables.
    pub variable_group: PhaseVariableGroup,
}

/// Which variable group a phase binds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseVariableGroup {
    /// Cycle dimension (log_T variables). Binds low-to-high.
    Cycle,
    /// Address dimension (log_k variables). Binds high-to-low.
    Address,
}
