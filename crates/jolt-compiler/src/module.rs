//! Compiler module types consumed by the prover runtime and verifier.

use std::fmt::Debug;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::formula::BindingOrder;
use crate::ir::PolyKind;
use crate::kernel_spec::KernelSpec;
use crate::polynomial_id::PolynomialId;

/// Index into `VerifierSchedule::stages`.
///
/// Distinguishes verifier stage indices from staging loop counters.
/// Staging stages that produce no sumcheck (e.g. eval-only) may be
/// skipped, so the staging index and verifier index can diverge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VerifierStageIndex(pub usize);

/// Complete output of the compilation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub polys: Vec<PolyDecl>,
    pub challenges: Vec<ChallengeDecl>,
    pub prover: Schedule,
    pub verifier: VerifierSchedule,
}

impl Module {
    /// Serialize to a compact binary format (`.jolt` protocol binary).
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .expect("module serialization should not fail")
    }

    /// Deserialize from a `.jolt` protocol binary.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let (module, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .expect("invalid protocol.jolt binary");
        module
    }
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
    SumcheckRound {
        stage: VerifierStageIndex,
        round: usize,
    },
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
    /// Batched sumcheck definitions (indexed by `Op::BatchedSumcheckRound::batch`).
    pub batched_sumchecks: Vec<BatchedSumcheckDef>,
}

/// A batched sumcheck stage grouping heterogeneous instances.
///
/// Each instance has its own kernel (formula + inputs) and runs for a
/// subset of the total rounds. Shorter instances are front-loaded: inactive
/// in early rounds, contributing `claim/2` per round until they activate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedSumcheckDef {
    pub instances: Vec<BatchedInstance>,
    /// Per-instance input claim formulas (evaluated by the runtime before
    /// the first round, absorbed into transcript for Fiat-Shamir).
    pub input_claims: Vec<ClaimFormula>,
    pub max_rounds: usize,
    pub max_degree: usize,
}

/// A single instance within a [`BatchedSumcheckDef`].
///
/// An instance may span multiple *phases*, each with its own compiled kernel.
/// For most instances a single phase suffices. Multi-phase instances (e.g.
/// RamReadWriteChecking) transition between kernels mid-sumcheck — the runtime
/// resolves fresh inputs at each phase boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedInstance {
    /// Kernel phases executed sequentially. The sum of all phase `num_rounds`
    /// equals the total round count for this instance.
    pub phases: Vec<InstancePhase>,
    /// Challenge index for this instance's batching coefficient.
    pub batch_coeff: usize,
    /// First round where this instance is active (`max_rounds - num_rounds`).
    pub first_active_round: usize,
}

impl BatchedInstance {
    /// Total number of rounds across all phases.
    pub fn num_rounds(&self) -> usize {
        self.phases.iter().map(|p| p.num_rounds).sum()
    }

    /// Find the phase and its start offset for a given instance-local round.
    ///
    /// Returns `(phase_index, phase_start_round)` where `phase_start_round`
    /// is the first instance-local round of that phase.
    pub fn phase_for_round(&self, instance_round: usize) -> (usize, usize) {
        let mut cumulative = 0;
        for (i, phase) in self.phases.iter().enumerate() {
            if instance_round < cumulative + phase.num_rounds {
                return (i, cumulative);
            }
            cumulative += phase.num_rounds;
        }
        panic!(
            "instance_round {instance_round} exceeds total rounds {}",
            self.num_rounds()
        );
    }
}

/// One phase of a [`BatchedInstance`].
///
/// Each phase has its own compiled kernel. The runtime resolves the phase's
/// kernel inputs when the phase begins and binds them each subsequent round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstancePhase {
    /// Index into `Schedule.kernels`.
    pub kernel: usize,
    /// Number of sumcheck rounds in this phase.
    pub num_rounds: usize,
    /// Scalar captures: at the start of this phase, the runtime reads the
    /// scalar value from each listed polynomial's (now fully-bound) device
    /// buffer and stores it in the corresponding challenge slot.
    ///
    /// This bridges phase boundaries: intermediate values from a prior phase
    /// become challenge constants for the next phase's formula.
    pub scalar_captures: Vec<ScalarCapture>,
    /// When present, this phase uses segmented reduce for mixed-size inputs.
    pub segmented: Option<SegmentedConfig>,
}

/// Captures a scalar value from a bound buffer into a challenge slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarCapture {
    /// Polynomial whose device buffer holds a single scalar (1-element).
    pub poly: PolynomialId,
    /// Challenge index where the scalar value is stored.
    pub challenge: usize,
}

/// Configuration for segmented reduce in a multi-dimensional sumcheck phase.
///
/// When a phase has mixed-size inputs (e.g. T-element cycle-only polynomials
/// alongside T×K-element cycle×address polynomials), the runtime performs a
/// segmented reduce: iterating over outer positions, extracting inner columns
/// from mixed inputs, running the Dense kernel on inner-sized slices, and
/// accumulating with outer eq weights.
///
/// The kernel itself is compiled as Dense over the inner dimension. The
/// segmented structure is runtime-level orchestration, not a backend concern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentedConfig {
    /// Log₂ of the inner segment size (bound in this phase).
    pub inner_num_vars: usize,
    /// Log₂ of the outer segment size (bound in a later phase).
    pub outer_num_vars: usize,
    /// Per kernel input: `true` = inner-only (`2^inner_num_vars` elements),
    /// `false` = full inner×outer (`2^(inner + outer)` elements).
    pub inner_only: Vec<bool>,
    /// Challenge indices for the outer eq table (built once at phase start).
    pub outer_eq_challenges: Vec<usize>,
}

impl Schedule {
    pub fn compute_op_count(&self) -> usize {
        self.ops.iter().filter(|s| s.is_compute()).count()
    }

    pub fn pcs_op_count(&self) -> usize {
        self.ops.iter().filter(|s| s.is_pcs()).count()
    }

    pub fn orchestration_op_count(&self) -> usize {
        self.ops.iter().filter(|s| s.is_orchestration()).count()
    }
}

/// Definition of a single sumcheck kernel (compiled by the backend at link time).
///
/// Combines a [`KernelSpec`] (what the backend compiles) with runtime context
/// (where to get inputs, how many rounds). The spec captures the algorithmic
/// decisions; the rest is orchestration metadata for the runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelDef {
    /// Backend compilation target: formula, iteration pattern, eval grid, binding order.
    pub spec: KernelSpec,
    /// Data provenance for each kernel input. `inputs[i]` describes where
    /// the data for the i-th input comes from at runtime. The first
    /// `spec.formula.num_inputs` entries are formula value columns; any
    /// remaining entries are extra inputs required by the iteration pattern
    /// (e.g., tensor eq buffers, sparse key column).
    pub inputs: Vec<InputBinding>,
    /// Total sumcheck rounds for this kernel.
    pub num_rounds: usize,
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
    Provided { poly: PolynomialId },
    /// Eq table built on-device: `eq(r, x) = Π(rᵢxᵢ + (1−rᵢ)(1−xᵢ))`.
    /// Challenge indices whose values form the evaluation point.
    EqTable {
        poly: PolynomialId,
        challenges: Vec<usize>,
    },
    /// Eq-plus-one table: `eq(r, x) · (1 + r_{n-1})`.
    EqPlusOneTable {
        poly: PolynomialId,
        challenges: Vec<usize>,
    },
    /// Less-than table from challenge points.
    LtTable {
        poly: PolynomialId,
        challenges: Vec<usize>,
    },
    /// Project a T×K source polynomial onto K elements via cycle eq weighting.
    ///
    /// Computes `result[k] = Σ_t eq(r_cycle, t) · source[t * outer_size + k]`
    /// where `r_cycle` comes from the challenge slots at runtime.
    ///
    /// Used when a downstream sumcheck instance needs the cycle-bound
    /// projection of a larger polynomial (e.g., RAM RAF RA from RamCombinedRa).
    EqProject {
        /// Polynomial ID for the projected result (storage/lifecycle).
        poly: PolynomialId,
        /// Source T×K polynomial to project from.
        source: PolynomialId,
        /// Challenge indices forming the cycle eq point.
        challenges: Vec<usize>,
        /// Size of the inner (cycle) dimension.
        inner_size: usize,
        /// Size of the outer (address) dimension.
        outer_size: usize,
    },
}

impl InputBinding {
    /// The poly slot this binding writes to / reads from in the buffer table.
    pub fn poly(&self) -> PolynomialId {
        match self {
            InputBinding::Provided { poly }
            | InputBinding::EqTable { poly, .. }
            | InputBinding::EqPlusOneTable { poly, .. }
            | InputBinding::LtTable { poly, .. }
            | InputBinding::EqProject { poly, .. } => *poly,
        }
    }
}

/// Fiat-Shamir domain-separation tag for transcript operations.
///
/// Each variant maps to a concrete byte string that the runtime absorbs
/// before the payload. Using an enum (not raw strings) ensures the Module
/// is self-contained and tag mismatches are caught at compile time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainSeparator {
    /// Polynomial commitment: `b"commitment"`.
    Commitment,
    /// Untrusted advice commitment: `b"untrusted_advice"`.
    UntrustedAdvice,
    /// Trusted advice commitment: `b"trusted_advice"`.
    TrustedAdvice,
    /// Full univariate-skip polynomial: `b"uniskip_poly"`.
    UniskipPoly,
    /// Compressed sumcheck round polynomial: `b"sumcheck_poly"`.
    SumcheckPoly,
    /// Batched sumcheck input claim: `b"sumcheck_claim"`.
    SumcheckClaim,
    /// Polynomial opening evaluation: `b"opening_claim"`.
    OpeningClaim,
}

impl DomainSeparator {
    /// The concrete byte string for Fiat-Shamir domain separation.
    pub fn as_bytes(&self) -> &'static [u8] {
        match self {
            Self::Commitment => b"commitment",
            Self::UntrustedAdvice => b"untrusted_advice",
            Self::TrustedAdvice => b"trusted_advice",
            Self::UniskipPoly => b"uniskip_poly",
            Self::SumcheckPoly => b"sumcheck_poly",
            Self::SumcheckClaim => b"sumcheck_claim",
            Self::OpeningClaim => b"opening_claim",
        }
    }
}

/// A single prover operation in the schedule.
///
/// Three categories:
/// - **Compute** — dispatched to [`ComputeBackend`] via compiled kernels.
/// - **PCS** — dispatched to [`CommitmentScheme`] (commit, reduce, open).
/// - **Orchestration** — zero-cost host bookkeeping (transcript, lifecycle).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Op {
    // ── Compute (dispatched to ComputeBackend via compiled kernels) ──
    /// Compute one sumcheck round polynomial.
    ///
    /// Round 0 (`bind_challenge: None`): reduce only.
    /// Rounds 1+ (`bind_challenge: Some(ch)`): fused bind-at-challenge + reduce.
    SumcheckRound {
        kernel: usize,
        round: usize,
        bind_challenge: Option<usize>,
    },
    /// One round of a batched sumcheck: dispatch to active instance kernels,
    /// combine round polynomials with batching coefficients.
    BatchedSumcheckRound {
        batch: usize,
        round: usize,
        bind_challenge: Option<usize>,
    },
    /// Extract polynomial evaluation (buffer fully bound → single element).
    Evaluate { poly: PolynomialId },
    /// Bind polynomial buffers at a challenge value (post-sumcheck survivors).
    Bind {
        polys: Vec<PolynomialId>,
        challenge: usize,
        order: BindingOrder,
    },
    /// Project polynomial buffers by evaluating Lagrange basis at a challenge,
    /// collapsing the constraint dimension after a univariate skip round.
    ///
    /// Transforms each buffer from `num_cycles × stride` entries to
    /// `num_cycles × num_groups` entries:
    ///
    /// ```text
    /// result[c * G + g] = scale · Σ_{k=0}^{D-1} L_k(r) · buf[c * stride + offsets[g] + k]
    /// ```
    ///
    /// where `D` = `domain_size`, `G` = `group_offsets.len()`,
    /// `L_k` are Lagrange basis polynomials over the symmetric domain
    /// `{domain_start, …, domain_start + D - 1}`, `r = challenges[challenge]`,
    /// and `scale = L_kernel(challenges[kernel_tau], r)` if `kernel_tau` is set (1 otherwise).
    LagrangeProject {
        polys: Vec<PolynomialId>,
        challenge: usize,
        domain_size: usize,
        domain_start: i64,
        stride: usize,
        group_offsets: Vec<usize>,
        /// When set, all projected values are multiplied by the Lagrange kernel
        /// `L(challenges[kernel_tau], challenges[challenge])` over the projection domain.
        /// This folds the uniskip kernel factor into the projected buffers.
        kernel_tau: Option<usize>,
    },

    // ── PCS (dispatched to CommitmentScheme trait) ──
    /// Commit polynomials, absorb commitments into transcript,
    /// capture raw data and hints for later opening proofs.
    Commit {
        polys: Vec<PolynomialId>,
        tag: DomainSeparator,
        /// Total multilinear variables for the PCS grid.
        /// Polynomials shorter than `2^num_vars` are zero-padded.
        /// Determines Dory matrix dimensions via balanced (sigma, nu) split.
        num_vars: usize,
    },
    /// Commit polynomials via streaming (chunked) PCS.
    ///
    /// Uses `StreamingCommitment::begin/feed/finish` instead of
    /// `CommitmentScheme::commit` — enables large polynomials to be
    /// committed without holding the full evaluation table in a single
    /// contiguous allocation on the PCS side.
    CommitStreaming {
        polys: Vec<PolynomialId>,
        tag: DomainSeparator,
        /// Chunk size (evaluations per feed call).
        chunk_size: usize,
        /// Total multilinear variables for the PCS grid.
        num_vars: usize,
    },
    /// RLC-reduce all accumulated opening claims via transcript challenges.
    ReduceOpenings,
    /// Generate PCS opening proofs for all reduced claims.
    Open,

    // ── Orchestration (zero-cost host bookkeeping) ──
    /// Absorb public instance data into the transcript.
    Preamble,
    /// Begin a new verifier stage (for incremental proof assembly).
    BeginStage { index: usize },
    /// Interpolate round evals → monomial coefficients, absorb into transcript.
    AbsorbRoundPoly {
        kernel: usize,
        num_coeffs: usize,
        tag: DomainSeparator,
    },
    /// Record polynomial evaluations in the stage proof for the verifier.
    ///
    /// Pushes values to `stage.evals` so the verifier can read them via
    /// `VerifierOp::RecordEvals`. Does not touch the transcript.
    RecordEvals { polys: Vec<PolynomialId> },
    /// Absorb polynomial evaluations into the Fiat-Shamir transcript.
    ///
    /// Transcript-only — does not record in the stage proof. Pair with
    /// `RecordEvals` when the verifier also needs the values.
    AbsorbEvals {
        polys: Vec<PolynomialId>,
        tag: DomainSeparator,
    },
    /// Evaluate a [`ClaimFormula`] against current evaluations/challenges and
    /// absorb the resulting scalar into the Fiat-Shamir transcript.
    AbsorbInputClaim {
        formula: ClaimFormula,
        tag: DomainSeparator,
        /// Batch index and instance index within the batch.
        /// Used to initialize the runtime's per-instance claim for
        /// inactive-round constant contributions.
        batch: usize,
        instance: usize,
    },
    /// Squeeze a Fiat-Shamir challenge.
    Squeeze { challenge: usize },
    /// Accumulate a PCS opening claim: (poly data, eval point from stage).
    CollectOpeningClaim {
        poly: PolynomialId,
        at_stage: VerifierStageIndex,
    },
    /// Release a device buffer (GPU memory).
    ReleaseDevice { poly: PolynomialId },
    /// Release host-side polynomial data (provider memory).
    /// Emitted after `ReduceOpenings` when evaluation tables are no longer needed.
    ReleaseHost { polys: Vec<PolynomialId> },
}

impl Op {
    pub fn is_compute(&self) -> bool {
        matches!(
            self,
            Op::SumcheckRound { .. }
                | Op::BatchedSumcheckRound { .. }
                | Op::Evaluate { .. }
                | Op::Bind { .. }
                | Op::LagrangeProject { .. }
        )
    }

    pub fn is_pcs(&self) -> bool {
        matches!(
            self,
            Op::Commit { .. } | Op::CommitStreaming { .. } | Op::ReduceOpenings | Op::Open
        )
    }

    pub fn is_orchestration(&self) -> bool {
        matches!(
            self,
            Op::Preamble
                | Op::BeginStage { .. }
                | Op::AbsorbRoundPoly { .. }
                | Op::RecordEvals { .. }
                | Op::AbsorbEvals { .. }
                | Op::AbsorbInputClaim { .. }
                | Op::Squeeze { .. }
                | Op::CollectOpeningClaim { .. }
                | Op::ReleaseDevice { .. }
                | Op::ReleaseHost { .. }
        )
    }
}

/// Verifier execution schedule: a flat sequence of ops for Fiat-Shamir replay
/// and claim checking.
///
/// The verifier is a generic interpreter: it walks `ops` in order, one match
/// arm per variant, mirroring the prover's flat `Vec<Op>` execution model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierSchedule {
    pub ops: Vec<VerifierOp>,
    /// Total number of challenge slots.
    pub num_challenges: usize,
    /// Total number of polynomial slots (for evaluation tracking).
    pub num_polys: usize,
    /// Total number of sumcheck stages (for preallocating point/eval vectors).
    pub num_stages: usize,
}

/// A single verifier operation in the schedule.
///
/// Mirrors the prover's [`Op`] enum: the verifier walks a flat `Vec<VerifierOp>`
/// in a single match loop. The compiler places each op at the exact position
/// where its data dependencies are satisfied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerifierOp {
    /// Absorb prover config into transcript (matches prover's `Preamble`).
    Preamble,
    /// Advance stage proof cursor; subsequent `VerifySumcheck` and `RecordEvals`
    /// read from this stage proof.
    BeginStage,
    /// Absorb next commitment from proof, store in commitment map.
    AbsorbCommitment {
        poly: PolynomialId,
        tag: DomainSeparator,
    },
    /// Squeeze a Fiat-Shamir challenge.
    Squeeze { challenge: usize },
    /// Absorb a round polynomial from the current stage proof into transcript.
    ///
    /// Reads the next round polynomial (at `round_poly_cursor`) from the stage
    /// proof, absorbs its coefficients into the Fiat-Shamir transcript, and
    /// advances the cursor. Used for uniskip rounds that are not part of the
    /// batched sumcheck verified by [`VerifySumcheck`].
    AbsorbRoundPoly {
        num_coeffs: usize,
        tag: DomainSeparator,
    },
    /// Verify batched sumcheck from current stage proof.
    ///
    /// Computes combined claim from instance `input_claim` formulas, verifies
    /// sumcheck rounds, stores `final_eval` and challenge point for the stage.
    ///
    /// When `batch_challenges` is non-empty, the handler:
    /// 1. Evaluates each instance's `input_claim` formula
    /// 2. Absorbs each claim into transcript with `claim_tag`
    /// 3. Squeezes batch coefficients into `challenges[batch_challenges[i]]`
    /// 4. Combines: `Σ batch_coeff[i] * claim[i] * 2^(max_rounds - num_rounds[i])`
    VerifySumcheck {
        instances: Vec<SumcheckInstance>,
        stage: usize,
        /// Challenge indices for per-instance batching coefficients.
        /// Empty for unbatched stages (scaling uses `2^offset` only).
        batch_challenges: Vec<usize>,
        /// Transcript tag for absorbing input claims before squeezing
        /// batch coefficients. Required when `batch_challenges` is non-empty.
        claim_tag: Option<DomainSeparator>,
    },
    /// Read polynomial evaluations from current stage proof into the global table.
    RecordEvals { evals: Vec<Evaluation> },
    /// Absorb polynomial evaluations into transcript.
    AbsorbEvals {
        polys: Vec<PolynomialId>,
        tag: DomainSeparator,
    },
    /// Verify output: composition formula must equal stored `final_eval`.
    ///
    /// The compiler places this at the exact position where all referenced
    /// evaluations are available, eliminating deferred checks.
    CheckOutput {
        instances: Vec<SumcheckInstance>,
        stage: usize,
        /// When non-empty, each instance's output is multiplied by its batch
        /// coefficient (stored at the given challenge index). Empty for
        /// unbatched sumchecks.
        batch_challenges: Vec<usize>,
    },
    /// Accumulate a PCS opening claim for a committed polynomial.
    CollectOpeningClaim {
        poly: PolynomialId,
        at_stage: VerifierStageIndex,
    },
    /// RLC-reduce all collected claims and verify PCS opening proofs.
    VerifyOpenings,
}

/// A single sumcheck instance within a batched stage.
///
/// The verifier evaluates `input_claim` before the sumcheck to compute
/// this instance's contribution to the combined claimed sum. After the
/// sumcheck, it evaluates `output_check` at the instance's challenge
/// slice to verify the composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SumcheckInstance {
    /// Symbolic formula for this instance's input claim.
    pub input_claim: ClaimFormula,
    /// Composition formula for the output claim check.
    /// Evaluated at the instance's challenge slice (offset by
    /// `max_rounds − num_rounds` into the stage's sumcheck challenges).
    pub output_check: ClaimFormula,
    /// Number of sumcheck rounds for this instance.
    pub num_rounds: usize,
    /// Composition degree (determines round polynomial size).
    pub degree: usize,
    /// How to convert raw sumcheck challenges to the canonical opening point
    /// used when evaluating `output_check`. Applied by the verifier before
    /// formula evaluation. `None` means raw challenges are used as-is.
    pub normalize: Option<PointNormalization>,
}

/// How raw sumcheck challenges are converted to the canonical opening point.
///
/// Sumcheck rounds produce challenges in binding order (LowToHigh = LSB first).
/// The opening point used by output check formulas is typically in big-endian
/// (MSB first) order. This enum specifies the transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PointNormalization {
    /// Reverse the full challenge sequence (LowToHigh → big-endian).
    Reverse,
    /// Multi-segment: split raw challenges into contiguous segments,
    /// reverse within each segment, then concatenate in the specified order.
    ///
    /// Example: RamRW with phase1=25 cycle vars, phase2=20 address vars:
    ///   `sizes = [25, 20], output_order = [1, 0]`
    ///   Result: `[reversed(raw[25..45]) ∥ reversed(raw[0..25])]`
    ///   = `[big-endian address ∥ big-endian cycle]`
    Segments {
        sizes: Vec<usize>,
        output_order: Vec<usize>,
    },
}

/// Univariate-skip first-round verification parameters.
///
/// The uniskip sends a full (uncompressed) polynomial. The verifier absorbs
/// it, squeezes a challenge, evaluates the polynomial at that challenge, and
/// records the result as an opening claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniskipVerify {
    /// Number of full polynomial coefficients (`degree + 1`).
    pub num_coeffs: usize,
    /// Poly index for the output evaluation (stored after verification).
    pub eval_poly: PolynomialId,
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
    /// Value of evaluation `evals[poly_index]` (accumulated across stages).
    Eval(PolynomialId),
    /// Value of challenge `challenges[i]`.
    Challenge(usize),
    /// Single-variable eq between two challenge values:
    /// `eq(challenges[a], challenges[b]) = a*b + (1-a)(1-b)`.
    EqChallengePair { a: usize, b: usize },
    /// Multilinear eq polynomial evaluated at two points:
    /// `eq(r, s) = ∏ᵢ (rᵢ·sᵢ + (1−rᵢ)(1−sᵢ))`
    /// where `r` is formed from challenges at the given indices and
    /// `s` is the sumcheck challenge point from the given stage.
    EqEval {
        challenges: Vec<usize>,
        at_stage: VerifierStageIndex,
    },
    /// Lagrange kernel evaluation `L(τ, r)` over the uniform R1CS constraint
    /// domain. The runtime computes this from the domain size in the R1CS key.
    ///
    /// `L(τ, r) = Σ_k eq(τ, k) × L_k(r)` where `k` ranges over the
    /// constraint domain and `L_k` is the k-th Lagrange basis polynomial.
    LagrangeKernel {
        /// Challenge index for the τ value (e.g. τ_high).
        tau_challenge: usize,
        /// Challenge index for the evaluation point (e.g. uniskip r0).
        at_challenge: usize,
    },
    /// Uniform R1CS matrix–evaluation inner product at a Lagrange point.
    ///
    /// The runtime computes `Σ_k L_k(r0) × (Σ_j M[k][j] × z_j)` where:
    /// - `M` is the A or B matrix from the preprocessed `UniformSpartanKey`
    /// - `L_k(r0)` is the Lagrange basis for constraint `k` at `r0`
    /// - `z_j` are the prover-provided evaluation values at `eval_polys[j]`
    ///
    /// The Module never embeds matrix coefficients — the runtime resolves
    /// them from the R1CS key at verification time.
    UniformR1CSEval {
        matrix: R1CSMatrix,
        /// Poly identifiers whose evaluations form the z-vector.
        eval_polys: Vec<PolynomialId>,
        /// Challenge index for the Lagrange interpolation point (r0).
        at_challenge: usize,
        /// Number of constraints to evaluate (may be less than the full R1CS).
        num_constraints: usize,
    },
    /// Eq evaluation between challenge values and a contiguous **slice** of a
    /// stage's (normalized) sumcheck point.
    ///
    /// Computes `eq(r, s[offset..offset+len])` where `len = challenges.len()`.
    /// This is needed when an output check uses only a portion of the opening
    /// point (e.g. the cycle portion of a combined address×cycle point).
    EqEvalSlice {
        challenges: Vec<usize>,
        at_stage: VerifierStageIndex,
        /// Starting index within the (normalized) sumcheck point.
        offset: usize,
    },
    /// Lagrange kernel `L(τ, r) = Σ_{k=0}^{N-1} L_k(τ) × L_k(r)` over an
    /// explicit domain `{0, 1, ..., domain_size-1}`.
    ///
    /// Generalizes [`LagrangeKernel`] to arbitrary domain sizes (e.g. size 3
    /// for product virtualization, size 10 for R1CS outer).
    LagrangeKernelDomain {
        tau_challenge: usize,
        at_challenge: usize,
        domain_size: usize,
    },
    /// Single Lagrange basis polynomial `L_k(r)` at a challenge value `r`,
    /// over the domain `{0, 1, ..., domain_size-1}`.
    LagrangeWeight {
        challenge: usize,
        domain_size: usize,
        basis_index: usize,
    },
    /// Evaluation of a public/preprocessed polynomial at the current stage's
    /// (normalized) sumcheck point. The runtime resolves the polynomial from
    /// the verifying key by its module poly index.
    PreprocessedPolyEval {
        poly: PolynomialId,
        at_stage: VerifierStageIndex,
    },
    /// Evaluation from the current stage's prover-provided evaluation list,
    /// at the given position. Used in output_check formulas when the same
    /// polynomial is opened at multiple points by different instances within
    /// a single batched stage. Position indexes into the stage's evaluation list.
    StageEval(usize),
}

/// Which matrix of the R1CS relation `Az ∘ Bz = Cz`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum R1CSMatrix {
    A,
    B,
}

/// A polynomial evaluation at a specific point in the verifier schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    /// Poly identifier in the module's poly table.
    pub poly: PolynomialId,
    /// Verifier stage whose sumcheck challenge point is the evaluation point.
    pub at_stage: VerifierStageIndex,
}
