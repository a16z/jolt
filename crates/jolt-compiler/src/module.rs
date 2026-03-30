//! Compiler module types consumed by the prover runtime and verifier.

use serde::{Deserialize, Serialize};

use crate::formula::{BindingOrder, Formula};
use crate::ir::PolyKind;

/// Complete output of the compilation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub polys: Vec<PolyDecl>,
    pub challenges: Vec<ChallengeDecl>,
    pub prover: Schedule,
    pub verifier: VerifierSchedule,
}

/// Polynomial declaration in the compiled module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyDecl {
    pub name: String,
    pub kind: PolyKind,
    /// Total number of field elements (2^num_vars).
    pub num_elements: usize,
}

/// Challenge declaration in the compiled module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeDecl {
    pub name: String,
    pub source: ChallengeSource,
}

/// How a challenge value is determined.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeSource {
    /// Squeezed from transcript after a stage completes.
    FiatShamir { after_stage: usize },
    /// Squeezed within a sumcheck stage after a round polynomial is appended.
    SumcheckRound { stage: usize, round: usize },
    /// Power of another challenge: `challenges[base]^exponent`.
    Power { base: usize, exponent: usize },
    /// From outside the protocol (preprocessing, public input).
    External,
}

/// Prover execution schedule: a flat sequence of ops with compiled kernel defs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub ops: Vec<Op>,
    pub kernels: Vec<KernelDef>,
}

/// Definition of a single sumcheck kernel (compiled by the backend at link time).
///
/// Each kernel captures a composition formula and the provenance of its inputs.
/// The formula is compiled into a backend-specific kernel at link time; input
/// buffers are resolved at execution time by the runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelDef {
    /// The composition formula for compute backend compilation.
    pub formula: Formula,
    /// Data provenance for each formula input. `inputs[i]` describes where
    /// the data for `Formula::Input(i)` comes from at runtime.
    pub inputs: Vec<InputBinding>,
    /// Variable binding direction.
    pub binding_order: BindingOrder,
    /// Total sumcheck rounds for this kernel.
    pub num_rounds: usize,
    /// Composition degree (determines round polynomial size).
    pub degree: usize,
}

/// Data provenance for a kernel input.
///
/// Each variant describes where the runtime obtains buffer data:
/// - [`Provided`](InputBinding::Provided) — loaded from the [`BufferProvider`]
///   (witness data, preprocessed tables). The poly index references `Module.polys`.
/// - Table variants — built on-device from challenge values. The runtime calls
///   the appropriate backend primitive (e.g., `eq_table`) using the
///   challenge values at the given indices. The poly index is the storage slot
///   in `Module.polys` for lifecycle tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputBinding {
    /// Buffer loaded from the provider (witness / preprocessed).
    Provided { poly: usize },
    /// Eq table built on-device: `eq(r, x) = Π(rᵢxᵢ + (1-rᵢ)(1-xᵢ))`.
    /// Challenge indices whose values form the evaluation point.
    EqTable { poly: usize, challenges: Vec<usize> },
    /// Eq-plus-one table: `eq(r, x) · (1 + r_{n-1})`.
    EqPlusOneTable { poly: usize, challenges: Vec<usize> },
    /// Less-than table from challenge points.
    LtTable { poly: usize, challenges: Vec<usize> },
}

impl InputBinding {
    /// The poly slot this binding writes to / reads from in the buffer table.
    pub fn poly(&self) -> usize {
        match self {
            InputBinding::Provided { poly }
            | InputBinding::EqTable { poly, .. }
            | InputBinding::EqPlusOneTable { poly, .. }
            | InputBinding::LtTable { poly, .. } => *poly,
        }
    }
}

/// A single prover operation in the schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Op {
    // --- FS transcript barriers ---
    /// Emit polynomial commitments into the transcript.
    EmitCommitments { polys: Vec<usize> },
    /// Emit sumcheck round polynomial coefficients.
    EmitRoundPoly { kernel: usize, num_coeffs: usize },
    /// Emit scalar evaluations.
    EmitScalars { evals: Vec<usize> },
    /// Squeeze a challenge from the transcript.
    Squeeze { challenge: usize },

    // --- Compute ---
    /// Compute one sumcheck round polynomial.
    ///
    /// Round 0 (`bind_challenge: None`): reduce only.
    /// Rounds 1+ (`bind_challenge: Some(ch)`): fused bind-at-challenge + reduce
    /// in a single device pass.
    SumcheckRound {
        kernel: usize,
        round: usize,
        bind_challenge: Option<usize>,
    },
    /// Extract polynomial evaluation (buffer fully bound → single element).
    Evaluate { poly: usize },
    /// Bind polynomial buffers at a challenge point (post-sumcheck only).
    ///
    /// Used after a sumcheck completes to reduce surviving polynomials
    /// that participate in later stages.
    FinalBind {
        polys: Vec<usize>,
        challenge: usize,
        order: BindingOrder,
    },

    // --- Lifecycle ---
    /// Release a polynomial buffer. Compiler-inserted after last use.
    /// The runtime drops the device buffer, reclaiming memory.
    Release { poly: usize },
}

/// Verifier execution schedule: Fiat-Shamir replay and claim-checking formulas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierSchedule {
    pub stages: Vec<VerifierStage>,
}

/// One verifier stage: a single batched sumcheck verification round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierStage {
    /// Committed polys absorbed before this stage.
    pub commitments: Vec<usize>,
    /// Symbolic formula to compute the batched input claim.
    pub input_claim: ClaimFormula,
    /// Sumcheck verification parameters.
    pub num_rounds: usize,
    pub degree: usize,
    /// Evaluations produced after sumcheck completes.
    pub evaluations: Vec<Evaluation>,
    /// Challenges squeezed after evaluations are absorbed.
    pub post_squeeze: Vec<usize>,
}

/// Symbolic sum-of-products for computing a verifier claim from
/// upstream evaluation values and challenges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimFormula {
    pub terms: Vec<ClaimTerm>,
}

impl ClaimFormula {
    pub fn zero() -> Self {
        Self { terms: vec![] }
    }
}

/// A single term in a claim formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimTerm {
    pub coeff: i128,
    pub factors: Vec<ClaimFactor>,
}

/// Factor in a claim formula term.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClaimFactor {
    /// Value of evaluation `evals[i]` (accumulated across stages).
    Eval(usize),
    /// Value of challenge `challenges[i]`.
    Challenge(usize),
}

/// A polynomial evaluation at a specific point in the verifier schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    /// Poly index in the module's poly table.
    pub poly: usize,
    /// Which vertex produced the evaluation point.
    pub at_vertex: usize,
}

impl Schedule {
    /// Count of FS barrier ops.
    pub fn fs_op_count(&self) -> usize {
        self.ops.iter().filter(|s| s.is_fs()).count()
    }

    /// Count of compute ops.
    pub fn compute_op_count(&self) -> usize {
        self.ops.len() - self.fs_op_count()
    }
}

impl Op {
    /// Whether this op is a Fiat-Shamir transcript operation.
    pub fn is_fs(&self) -> bool {
        matches!(
            self,
            Op::EmitCommitments { .. }
                | Op::EmitRoundPoly { .. }
                | Op::EmitScalars { .. }
                | Op::Squeeze { .. }
        )
    }
}
