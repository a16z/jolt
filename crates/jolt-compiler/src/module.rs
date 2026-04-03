//! Compiler module types consumed by the prover runtime and verifier.

use std::fmt::Debug;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::formula::BindingOrder;
use crate::ir::PolyKind;
use crate::kernel_spec::KernelSpec;

/// Identity type for polynomials in compiled modules.
///
/// The compiler is protocol-agnostic: it uses `usize` indices internally.
/// Downstream crates instantiate with their own key type (e.g. the
/// zkVM's `PolynomialId` enum) via [`Module::remap`].
pub trait PolyId:
    Copy + Eq + Hash + Ord + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static
{
}

impl PolyId for usize {}

/// Index into `VerifierSchedule::stages`.
///
/// Distinguishes verifier stage indices from staging loop counters.
/// Staging stages that produce no sumcheck (e.g. eval-only) may be
/// skipped, so the staging index and verifier index can diverge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VerifierStageIndex(pub usize);

/// Complete output of the compilation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Module<P: PolyId = usize> {
    pub polys: Vec<PolyDecl>,
    pub challenges: Vec<ChallengeDecl>,
    pub prover: Schedule<P>,
    pub verifier: VerifierSchedule<P>,
}

impl<Q: PolyId> Module<Q> {
    /// Remap all polynomial identifiers using a mapping function.
    ///
    /// Converts `Module<Q>` to `Module<P>` by applying `f` to every
    /// polynomial reference in the prover and verifier schedules.
    /// Polynomial declarations and challenge metadata are unchanged.
    pub fn remap<P: PolyId>(self, f: impl Fn(Q) -> P) -> Module<P> {
        Module {
            polys: self.polys,
            challenges: self.challenges,
            prover: self.prover.remap(&f),
            verifier: self.verifier.remap(&f),
        }
    }

    /// Serialize to a compact binary format (`.jolt` protocol binary).
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .expect("module serialization should not fail")
    }

    /// Deserialize from a `.jolt` protocol binary.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let (module, _) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard())
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
#[serde(bound = "")]
pub struct Schedule<P: PolyId = usize> {
    pub ops: Vec<Op<P>>,
    pub kernels: Vec<KernelDef<P>>,
}

impl<P: PolyId> Schedule<P> {
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

impl<Q: PolyId> Schedule<Q> {
    pub fn remap<P: PolyId>(self, f: &impl Fn(Q) -> P) -> Schedule<P> {
        Schedule {
            ops: self.ops.into_iter().map(|op| op.remap(f)).collect(),
            kernels: self.kernels.into_iter().map(|k| k.remap(f)).collect(),
        }
    }
}

/// Definition of a single sumcheck kernel (compiled by the backend at link time).
///
/// Combines a [`KernelSpec`] (what the backend compiles) with runtime context
/// (where to get inputs, how many rounds). The spec captures the algorithmic
/// decisions; the rest is orchestration metadata for the runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct KernelDef<P: PolyId = usize> {
    /// Backend compilation target: formula, iteration pattern, eval grid, binding order.
    pub spec: KernelSpec,
    /// Data provenance for each kernel input. `inputs[i]` describes where
    /// the data for the i-th input comes from at runtime. The first
    /// `spec.formula.num_inputs` entries are formula value columns; any
    /// remaining entries are extra inputs required by the iteration pattern
    /// (e.g., tensor eq buffers, sparse key column).
    pub inputs: Vec<InputBinding<P>>,
    /// Total sumcheck rounds for this kernel.
    pub num_rounds: usize,
}

impl<Q: PolyId> KernelDef<Q> {
    pub fn remap<P: PolyId>(self, f: &impl Fn(Q) -> P) -> KernelDef<P> {
        KernelDef {
            spec: self.spec,
            inputs: self.inputs.into_iter().map(|b| b.remap(f)).collect(),
            num_rounds: self.num_rounds,
        }
    }
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
#[serde(bound = "")]
pub enum InputBinding<P: PolyId = usize> {
    /// Buffer loaded from the provider (witness / preprocessed).
    Provided { poly: P },
    /// Eq table built on-device: `eq(r, x) = Π(rᵢxᵢ + (1−rᵢ)(1−xᵢ))`.
    /// Challenge indices whose values form the evaluation point.
    EqTable { poly: P, challenges: Vec<usize> },
    /// Eq-plus-one table: `eq(r, x) · (1 + r_{n-1})`.
    EqPlusOneTable { poly: P, challenges: Vec<usize> },
    /// Less-than table from challenge points.
    LtTable { poly: P, challenges: Vec<usize> },
}

impl<P: PolyId> InputBinding<P> {
    /// The poly slot this binding writes to / reads from in the buffer table.
    pub fn poly(&self) -> P {
        match self {
            InputBinding::Provided { poly }
            | InputBinding::EqTable { poly, .. }
            | InputBinding::EqPlusOneTable { poly, .. }
            | InputBinding::LtTable { poly, .. } => *poly,
        }
    }
}

impl<Q: PolyId> InputBinding<Q> {
    pub fn remap<P: PolyId>(self, f: &impl Fn(Q) -> P) -> InputBinding<P> {
        match self {
            InputBinding::Provided { poly } => InputBinding::Provided { poly: f(poly) },
            InputBinding::EqTable { poly, challenges } => InputBinding::EqTable {
                poly: f(poly),
                challenges,
            },
            InputBinding::EqPlusOneTable { poly, challenges } => InputBinding::EqPlusOneTable {
                poly: f(poly),
                challenges,
            },
            InputBinding::LtTable { poly, challenges } => InputBinding::LtTable {
                poly: f(poly),
                challenges,
            },
        }
    }
}

/// Fiat-Shamir domain-separation tag for transcript operations.
///
/// Each variant maps to a concrete byte string that the runtime appends
/// before the payload. Using an enum (not raw strings) ensures the Module
/// is self-contained and tag mismatches are caught at compile time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TranscriptTag {
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

impl TranscriptTag {
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
#[serde(bound = "")]
pub enum Op<P: PolyId = usize> {
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
    /// Extract polynomial evaluation (buffer fully bound → single element).
    Evaluate { poly: P },
    /// Bind polynomial buffers at a challenge value (post-sumcheck survivors).
    Bind {
        polys: Vec<P>,
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
        polys: Vec<P>,
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
        polys: Vec<P>,
        tag: TranscriptTag,
    },
    /// Commit polynomials via streaming (chunked) PCS.
    ///
    /// Uses `StreamingCommitment::begin/feed/finish` instead of
    /// `CommitmentScheme::commit` — enables large polynomials to be
    /// committed without holding the full evaluation table in a single
    /// contiguous allocation on the PCS side.
    CommitStreaming {
        polys: Vec<P>,
        tag: TranscriptTag,
        /// Chunk size (evaluations per feed call).
        chunk_size: usize,
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
        tag: TranscriptTag,
    },
    /// Absorb polynomial evaluations into transcript.
    AbsorbEvals {
        polys: Vec<P>,
        tag: TranscriptTag,
    },
    /// Squeeze a Fiat-Shamir challenge.
    Squeeze { challenge: usize },
    /// Accumulate a PCS opening claim: (poly data, eval point from stage).
    CollectOpeningClaim {
        poly: P,
        at_stage: VerifierStageIndex,
    },
    /// Release a device buffer (GPU memory).
    ReleaseDevice { poly: P },
    /// Release host-side polynomial data (provider memory).
    /// Emitted after `ReduceOpenings` when evaluation tables are no longer needed.
    ReleaseHost { polys: Vec<P> },
}

impl<P: PolyId> Op<P> {
    pub fn is_compute(&self) -> bool {
        matches!(
            self,
            Op::SumcheckRound { .. }
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
                | Op::AbsorbEvals { .. }
                | Op::Squeeze { .. }
                | Op::CollectOpeningClaim { .. }
                | Op::ReleaseDevice { .. }
                | Op::ReleaseHost { .. }
        )
    }
}

impl<Q: PolyId> Op<Q> {
    pub fn remap<P: PolyId>(self, f: &impl Fn(Q) -> P) -> Op<P> {
        match self {
            Op::SumcheckRound {
                kernel,
                round,
                bind_challenge,
            } => Op::SumcheckRound {
                kernel,
                round,
                bind_challenge,
            },
            Op::Evaluate { poly } => Op::Evaluate { poly: f(poly) },
            Op::Bind {
                polys,
                challenge,
                order,
            } => Op::Bind {
                polys: polys.into_iter().map(&f).collect(),
                challenge,
                order,
            },
            Op::LagrangeProject {
                polys,
                challenge,
                domain_size,
                domain_start,
                stride,
                group_offsets,
                kernel_tau,
            } => Op::LagrangeProject {
                polys: polys.into_iter().map(&f).collect(),
                challenge,
                domain_size,
                domain_start,
                stride,
                group_offsets,
                kernel_tau,
            },
            Op::Commit { polys, tag } => Op::Commit {
                polys: polys.into_iter().map(&f).collect(),
                tag,
            },
            Op::CommitStreaming {
                polys,
                tag,
                chunk_size,
            } => Op::CommitStreaming {
                polys: polys.into_iter().map(&f).collect(),
                tag,
                chunk_size,
            },
            Op::ReduceOpenings => Op::ReduceOpenings,
            Op::Open => Op::Open,
            Op::Preamble => Op::Preamble,
            Op::BeginStage { index } => Op::BeginStage { index },
            Op::AbsorbRoundPoly {
                kernel,
                num_coeffs,
                tag,
            } => Op::AbsorbRoundPoly {
                kernel,
                num_coeffs,
                tag,
            },
            Op::AbsorbEvals { polys, tag } => Op::AbsorbEvals {
                polys: polys.into_iter().map(&f).collect(),
                tag,
            },
            Op::Squeeze { challenge } => Op::Squeeze { challenge },
            Op::CollectOpeningClaim { poly, at_stage } => Op::CollectOpeningClaim {
                poly: f(poly),
                at_stage,
            },
            Op::ReleaseDevice { poly } => Op::ReleaseDevice { poly: f(poly) },
            Op::ReleaseHost { polys } => Op::ReleaseHost {
                polys: polys.into_iter().map(&f).collect(),
            },
        }
    }
}

/// Verifier execution schedule: a flat sequence of ops for Fiat-Shamir replay
/// and claim checking.
///
/// The verifier is a generic interpreter: it walks `ops` in order, one match
/// arm per variant, mirroring the prover's flat `Vec<Op>` execution model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierSchedule<P: PolyId = usize> {
    pub ops: Vec<VerifierOp<P>>,
    /// Total number of challenge slots.
    pub num_challenges: usize,
    /// Total number of polynomial slots (for evaluation tracking).
    pub num_polys: usize,
    /// Total number of sumcheck stages (for preallocating point/eval vectors).
    pub num_stages: usize,
}

impl<Q: PolyId> VerifierSchedule<Q> {
    pub fn remap<P: PolyId>(self, f: &impl Fn(Q) -> P) -> VerifierSchedule<P> {
        VerifierSchedule {
            ops: self.ops.into_iter().map(|op| op.remap(f)).collect(),
            num_challenges: self.num_challenges,
            num_polys: self.num_polys,
            num_stages: self.num_stages,
        }
    }
}

/// A single verifier operation in the schedule.
///
/// Mirrors the prover's [`Op`] enum: the verifier walks a flat `Vec<VerifierOp>`
/// in a single match loop. The compiler places each op at the exact position
/// where its data dependencies are satisfied.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum VerifierOp<P: PolyId = usize> {
    /// Absorb prover config into transcript (matches prover's `Preamble`).
    Preamble,
    /// Advance stage proof cursor; subsequent `VerifySumcheck` and `RecordEvals`
    /// read from this stage proof.
    BeginStage,
    /// Absorb next commitment from proof, store in commitment map.
    AbsorbCommitment { poly: P },
    /// Squeeze a Fiat-Shamir challenge.
    Squeeze { challenge: usize },
    /// Absorb a round polynomial from the current stage proof into transcript.
    ///
    /// Reads the next round polynomial (at `round_poly_cursor`) from the stage
    /// proof, absorbs its coefficients into the Fiat-Shamir transcript, and
    /// advances the cursor. Used for uniskip rounds that are not part of the
    /// batched sumcheck verified by [`VerifySumcheck`].
    AbsorbRoundPoly { num_coeffs: usize },
    /// Verify batched sumcheck from current stage proof.
    ///
    /// Computes combined claim from instance `input_claim` formulas, verifies
    /// sumcheck rounds, stores `final_eval` and challenge point for the stage.
    VerifySumcheck {
        instances: Vec<SumcheckInstance<P>>,
        stage: usize,
    },
    /// Read polynomial evaluations from current stage proof into the global table.
    RecordEvals { evals: Vec<Evaluation<P>> },
    /// Absorb polynomial evaluations into transcript.
    AbsorbEvals { polys: Vec<P> },
    /// Verify output: composition formula must equal stored `final_eval`.
    ///
    /// The compiler places this at the exact position where all referenced
    /// evaluations are available, eliminating deferred checks.
    CheckOutput {
        instances: Vec<SumcheckInstance<P>>,
        stage: usize,
    },
    /// Accumulate a PCS opening claim for a committed polynomial.
    CollectOpeningClaim {
        poly: P,
        at_stage: VerifierStageIndex,
    },
    /// RLC-reduce all collected claims and verify PCS opening proofs.
    VerifyOpenings,
}

impl<Q: PolyId> VerifierOp<Q> {
    pub fn remap<P: PolyId>(self, f: &impl Fn(Q) -> P) -> VerifierOp<P> {
        match self {
            VerifierOp::Preamble => VerifierOp::Preamble,
            VerifierOp::BeginStage => VerifierOp::BeginStage,
            VerifierOp::AbsorbCommitment { poly } => VerifierOp::AbsorbCommitment { poly: f(poly) },
            VerifierOp::Squeeze { challenge } => VerifierOp::Squeeze { challenge },
            VerifierOp::AbsorbRoundPoly { num_coeffs } => {
                VerifierOp::AbsorbRoundPoly { num_coeffs }
            }
            VerifierOp::VerifySumcheck { instances, stage } => VerifierOp::VerifySumcheck {
                instances: instances.into_iter().map(|i| i.remap(f)).collect(),
                stage,
            },
            VerifierOp::RecordEvals { evals } => VerifierOp::RecordEvals {
                evals: evals.into_iter().map(|e| e.remap(f)).collect(),
            },
            VerifierOp::AbsorbEvals { polys } => VerifierOp::AbsorbEvals {
                polys: polys.into_iter().map(&f).collect(),
            },
            VerifierOp::CheckOutput { instances, stage } => VerifierOp::CheckOutput {
                instances: instances.into_iter().map(|i| i.remap(f)).collect(),
                stage,
            },
            VerifierOp::CollectOpeningClaim { poly, at_stage } => {
                VerifierOp::CollectOpeningClaim {
                    poly: f(poly),
                    at_stage,
                }
            }
            VerifierOp::VerifyOpenings => VerifierOp::VerifyOpenings,
        }
    }
}

/// A single sumcheck instance within a batched stage.
///
/// The verifier evaluates `input_claim` before the sumcheck to compute
/// this instance's contribution to the combined claimed sum. After the
/// sumcheck, it evaluates `output_check` at the instance's challenge
/// slice to verify the composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SumcheckInstance<P: PolyId = usize> {
    /// Symbolic formula for this instance's input claim.
    pub input_claim: ClaimFormula<P>,
    /// Composition formula for the output claim check.
    /// Evaluated at the instance's challenge slice (offset by
    /// `max_rounds − num_rounds` into the stage's sumcheck challenges).
    pub output_check: ClaimFormula<P>,
    /// Number of sumcheck rounds for this instance.
    pub num_rounds: usize,
    /// Composition degree (determines round polynomial size).
    pub degree: usize,
    /// How to convert raw sumcheck challenges to the canonical opening point
    /// used when evaluating `output_check`. Applied by the verifier before
    /// formula evaluation. `None` means raw challenges are used as-is.
    pub normalize: Option<PointNormalization>,
}

impl<Q: PolyId> SumcheckInstance<Q> {
    pub fn remap<P: PolyId>(self, f: &impl Fn(Q) -> P) -> SumcheckInstance<P> {
        SumcheckInstance {
            input_claim: self.input_claim.remap(f),
            output_check: self.output_check.remap(f),
            num_rounds: self.num_rounds,
            degree: self.degree,
            normalize: self.normalize,
        }
    }
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
#[serde(bound = "")]
pub struct UniskipVerify<P: PolyId = usize> {
    /// Number of full polynomial coefficients (`degree + 1`).
    pub num_coeffs: usize,
    /// Poly index for the output evaluation (stored after verification).
    pub eval_poly: P,
}

impl<Q: PolyId> UniskipVerify<Q> {
    pub fn remap<P: PolyId>(self, f: &impl Fn(Q) -> P) -> UniskipVerify<P> {
        UniskipVerify {
            num_coeffs: self.num_coeffs,
            eval_poly: f(self.eval_poly),
        }
    }
}

/// Symbolic sum-of-products for computing a verifier claim from
/// upstream evaluation values and challenges.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ClaimFormula<P: PolyId = usize> {
    pub terms: Vec<ClaimTerm<P>>,
}

impl<P: PolyId> ClaimFormula<P> {
    pub fn zero() -> Self {
        Self { terms: vec![] }
    }
}

impl<Q: PolyId> ClaimFormula<Q> {
    pub fn remap<P: PolyId>(self, f: &impl Fn(Q) -> P) -> ClaimFormula<P> {
        ClaimFormula {
            terms: self.terms.into_iter().map(|t| t.remap(f)).collect(),
        }
    }
}

/// A single term in a claim formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ClaimTerm<P: PolyId = usize> {
    pub coeff: i128,
    pub factors: Vec<ClaimFactor<P>>,
}

impl<Q: PolyId> ClaimTerm<Q> {
    pub fn remap<P: PolyId>(self, f: &impl Fn(Q) -> P) -> ClaimTerm<P> {
        ClaimTerm {
            coeff: self.coeff,
            factors: self.factors.into_iter().map(|fac| fac.remap(f)).collect(),
        }
    }
}

/// Factor in a claim formula term.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum ClaimFactor<P: PolyId = usize> {
    /// Value of evaluation `evals[poly_index]` (accumulated across stages).
    Eval(P),
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
        eval_polys: Vec<P>,
        /// Challenge index for the Lagrange interpolation point (r0).
        at_challenge: usize,
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
        poly: P,
        at_stage: VerifierStageIndex,
    },
    /// Evaluation from the current stage's prover-provided evaluation list,
    /// at the given position. Used in output_check formulas when the same
    /// polynomial is opened at multiple points by different instances within
    /// a single batched stage. Position indexes into the stage's evaluation list.
    StageEval(usize),
}

impl<Q: PolyId> ClaimFactor<Q> {
    pub fn remap<P: PolyId>(self, f: &impl Fn(Q) -> P) -> ClaimFactor<P> {
        match self {
            ClaimFactor::Eval(poly) => ClaimFactor::Eval(f(poly)),
            ClaimFactor::Challenge(i) => ClaimFactor::Challenge(i),
            ClaimFactor::EqChallengePair { a, b } => ClaimFactor::EqChallengePair { a, b },
            ClaimFactor::EqEval {
                challenges,
                at_stage,
            } => ClaimFactor::EqEval {
                challenges,
                at_stage,
            },
            ClaimFactor::LagrangeKernel {
                tau_challenge,
                at_challenge,
            } => ClaimFactor::LagrangeKernel {
                tau_challenge,
                at_challenge,
            },
            ClaimFactor::UniformR1CSEval {
                matrix,
                eval_polys,
                at_challenge,
            } => ClaimFactor::UniformR1CSEval {
                matrix,
                eval_polys: eval_polys.into_iter().map(f).collect(),
                at_challenge,
            },
            ClaimFactor::EqEvalSlice {
                challenges,
                at_stage,
                offset,
            } => ClaimFactor::EqEvalSlice {
                challenges,
                at_stage,
                offset,
            },
            ClaimFactor::LagrangeKernelDomain {
                tau_challenge,
                at_challenge,
                domain_size,
            } => ClaimFactor::LagrangeKernelDomain {
                tau_challenge,
                at_challenge,
                domain_size,
            },
            ClaimFactor::LagrangeWeight {
                challenge,
                domain_size,
                basis_index,
            } => ClaimFactor::LagrangeWeight {
                challenge,
                domain_size,
                basis_index,
            },
            ClaimFactor::PreprocessedPolyEval { poly, at_stage } => {
                ClaimFactor::PreprocessedPolyEval {
                    poly: f(poly),
                    at_stage,
                }
            }
            ClaimFactor::StageEval(i) => ClaimFactor::StageEval(i),
        }
    }
}

/// Which matrix of the R1CS relation `Az ∘ Bz = Cz`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum R1CSMatrix {
    A,
    B,
}

/// A polynomial evaluation at a specific point in the verifier schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Evaluation<P: PolyId = usize> {
    /// Poly identifier in the module's poly table.
    pub poly: P,
    /// Verifier stage whose sumcheck challenge point is the evaluation point.
    pub at_stage: VerifierStageIndex,
}

impl<Q: PolyId> Evaluation<Q> {
    pub fn remap<P: PolyId>(self, f: &impl Fn(Q) -> P) -> Evaluation<P> {
        Evaluation {
            poly: f(self.poly),
            at_stage: self.at_stage,
        }
    }
}
