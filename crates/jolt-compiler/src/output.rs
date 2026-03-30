//! Compiler output types consumed by jolt-zkvm (prover) and jolt-verifier.

use serde::{Deserialize, Serialize};

use crate::formula::{BindingOrder, CompositionFormula};
use crate::ir::PolyKind;

/// Complete output of the compilation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerOutput {
    pub polys: Vec<PolySpec>,
    pub challenges: Vec<ChallengeSpec>,
    pub schedule: ProverSchedule,
    pub script: VerifierScript,
}

// ---------------------------------------------------------------------------
// Shared metadata
// ---------------------------------------------------------------------------

/// Polynomial metadata in the compiled output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolySpec {
    pub name: String,
    pub kind: PolyKind,
    /// Total number of field elements (2^num_vars).
    pub num_elements: usize,
}

/// Challenge metadata in the compiled output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeSpec {
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

// ---------------------------------------------------------------------------
// Prover schedule
// ---------------------------------------------------------------------------

/// Execution schedule for the prover: a flat sequence of steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverSchedule {
    pub steps: Vec<ProverStep>,
    pub kernels: Vec<KernelSpec>,
}

/// Specification of a single sumcheck kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSpec {
    /// The composition formula for compute backend compilation.
    pub formula: CompositionFormula,
    /// Maps formula `Input(i)` → poly index in `ProverSchedule.polys`.
    pub inputs: Vec<usize>,
    /// How eq weighting is applied.
    pub eq_mode: EqMode,
    /// Variable binding direction.
    pub binding_order: BindingOrder,
    /// Total sumcheck rounds for this kernel.
    pub num_rounds: usize,
    /// Composition degree (determines round polynomial size).
    pub degree: usize,
}

/// How eq polynomial weighting is configured for a kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EqMode {
    /// Eq is one of the kernel's input polynomials (index into `KernelSpec.inputs`).
    AsInput(usize),
    /// Implicit unit weight (no eq factor).
    Unit,
}

/// A single prover operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProverStep {
    // --- FS transcript barriers ---
    /// Absorb polynomial commitments into the transcript.
    AppendCommitments { polys: Vec<usize> },
    /// Absorb sumcheck round polynomial coefficients.
    AppendRoundPoly { kernel: usize, num_coeffs: usize },
    /// Absorb scalar evaluations.
    AppendScalars { evals: Vec<usize> },
    /// Squeeze a challenge from the transcript.
    Squeeze { challenge: usize },

    // --- Compute operations (reorderable between barriers) ---
    /// Materialize a derived polynomial buffer (eq table, etc).
    Materialize { poly: usize },
    /// Compute one sumcheck round polynomial.
    SumcheckRound {
        kernel: usize,
        round: usize,
        num_vars_remaining: usize,
    },
    /// Halve polynomial buffers at a challenge point.
    Bind {
        polys: Vec<usize>,
        challenge: usize,
        order: BindingOrder,
    },
    /// Extract polynomial evaluation (buffer fully bound → single element).
    Evaluate { poly: usize },
}

// ---------------------------------------------------------------------------
// Verifier script
// ---------------------------------------------------------------------------

/// Verification script: the Fiat-Shamir replay and claim-checking formulas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierScript {
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
    pub evaluations: Vec<EvalSpec>,
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

/// Specification of a polynomial evaluation in the proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSpec {
    /// Poly index in the script's poly table.
    pub poly: usize,
    /// Which vertex produced the evaluation point.
    pub at_vertex: usize,
}

impl ProverSchedule {
    /// Count of FS barrier steps.
    pub fn fs_step_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| matches!(
                s,
                ProverStep::AppendCommitments { .. }
                    | ProverStep::AppendRoundPoly { .. }
                    | ProverStep::AppendScalars { .. }
                    | ProverStep::Squeeze { .. }
            ))
            .count()
    }

    /// Count of compute steps.
    pub fn compute_step_count(&self) -> usize {
        self.steps.len() - self.fs_step_count()
    }
}

impl ProverStep {
    /// Whether this step is a Fiat-Shamir transcript operation.
    pub fn is_fs(&self) -> bool {
        matches!(
            self,
            ProverStep::AppendCommitments { .. }
                | ProverStep::AppendRoundPoly { .. }
                | ProverStep::AppendScalars { .. }
                | ProverStep::Squeeze { .. }
        )
    }
}
