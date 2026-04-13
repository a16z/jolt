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
    /// PCS-level number of variables for the opening proof.
    /// When `Some(n)`, the polynomial is zero-padded to `2^n` elements
    /// before PCS opening (e.g. dense cycle-only polys padded to K*T).
    /// When `None`, `num_elements` is used as-is.
    pub committed_num_vars: Option<usize>,
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
    /// Batched sumcheck definitions (indexed by batch index in granular ops).
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
    /// Extra buffers to bind alongside kernel inputs at each round.
    ///
    /// Carry bindings are materialized at phase start and bound with the
    /// same round challenge as the kernel inputs.  They do NOT participate
    /// in kernel evaluation — only in binding.  This bridges multi-phase
    /// instances where a later phase needs the bound-down version of a
    /// buffer that was not part of the earlier phase's formula.
    pub carry_bindings: Vec<InputBinding>,
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

/// Configuration for the Gruen-based booleanity sumcheck.
///
/// Stored in [`Op::BooleanityInit`] and consumed by the runtime to
/// initialize a [`CpuBooleanityState`](jolt_cpu::booleanity::CpuBooleanityState).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooleanityConfig {
    /// RA polynomial IDs to download for G_d / H construction.
    pub ra_poly_ids: Vec<PolynomialId>,
    /// Challenge slot indices for r_address (LE, length = log_k_chunk).
    pub addr_challenges: Vec<usize>,
    /// Challenge slot indices for r_cycle (LE, length = log_t).
    pub cycle_challenges: Vec<usize>,
    /// Challenge slot indices for γ^d (length = total_d).
    pub gamma_powers: Vec<usize>,
    /// Challenge slot indices for γ^{2d} (length = total_d).
    pub gamma_powers_square: Vec<usize>,
    pub log_k_chunk: usize,
    pub log_t: usize,
}

/// Configuration for the fused HammingWeight + Address Reduction sumcheck (Stage 7).
///
/// Stored in [`Op::HwReductionInit`] and consumed by the runtime to
/// initialize a `CpuHwReductionState`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HwReductionConfig {
    /// RA polynomial IDs to download for G_i computation (N total).
    pub ra_poly_ids: Vec<PolynomialId>,
    /// Challenge slot indices for r_cycle (BE, length = log_t).
    /// From the booleanity opening point's cycle portion.
    pub cycle_challenges_be: Vec<usize>,
    /// Challenge slot indices for r_addr_bool (BE, length = log_k_chunk).
    /// Shared across all families.
    pub addr_bool_challenges_be: Vec<usize>,
    /// Per-RA-poly challenge slot indices for r_addr_virt (BE, length = log_k_chunk each).
    pub addr_virt_challenges_be: Vec<Vec<usize>>,
    /// Challenge slot indices for γ^{3i} powers (length = 3*N).
    /// Order: γ^0, γ^1, ..., γ^{3N-1}.
    pub gamma_powers: Vec<usize>,
    /// HammingWeight evaluation challenge slot (for RAM HW claims).
    pub hw_eval_challenge: usize,
    pub instruction_d: usize,
    pub bytecode_d: usize,
    pub ram_d: usize,
    pub log_k_chunk: usize,
    pub log_t: usize,
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
    /// Eq-gather: build eq table from challenges, then gather per-element
    /// values using integer indices from a source polynomial.
    ///
    /// Computes `result[j] = eq(r, index[j])` where `r` is the challenge
    /// point and `index[j]` are per-cycle lookup indices from the provider.
    ///
    /// Used for register write-address indicators (eq(r_address, rd[j]))
    /// and RAM access indicators (eq(r_address, addr[j])).
    EqGather {
        /// Polynomial ID for the gathered result (T elements).
        poly: PolynomialId,
        /// Challenge indices forming the eq point (log₂K entries).
        eq_challenges: Vec<usize>,
        /// Source of per-cycle integer indices (T entries, each in 0..K-1).
        /// The provider materializes this from trace data.
        indices: PolynomialId,
    },
    /// Pushforward of eq polynomial through an index mapping.
    ///
    /// Computes `result[k] = Σ_j eq(r, j) × 1{indices[j] == k}` where
    /// `r` is the challenge point and `indices[j]` maps cycle `j` to an
    /// address-space bin `k`. The result has `output_size` elements.
    ///
    /// Used for BytecodeReadRaf F[stage] tables: each F[s][k] accumulates
    /// the eq weight of all cycles whose PC maps to bytecode index k.
    EqPushforward {
        poly: PolynomialId,
        eq_challenges: Vec<usize>,
        indices: PolynomialId,
        output_size: usize,
    },
    /// Multiply a source polynomial element-wise by a challenge value.
    ///
    /// Computes `result[k] = challenges[challenge]^power × source[k]`.
    /// Used for gamma-weighting preprocessed polynomials (e.g., entry_gamma × f_expected).
    ScaleByChallenge {
        poly: PolynomialId,
        source: PolynomialId,
        challenge: usize,
        power: u8,
    },
    /// Transpose a source polynomial from row-major to column-major layout.
    ///
    /// Source layout: `src[row * cols + col]` (rows × cols elements).
    /// Result layout: `dst[col * rows + row]` (cols × rows elements).
    ///
    /// Used by Booleanity to rearrange RA polynomials from address-major
    /// `[k * T + j]` to cycle-major `[j * K + k]` so that LowToHigh binding
    /// binds address variables first (matching jolt-core's Phase 1 → Phase 2).
    Transpose {
        poly: PolynomialId,
        source: PolynomialId,
        rows: usize,
        cols: usize,
    },
    /// Compute a BytecodeReadRaf Val polynomial for a specific stage.
    ///
    /// Computes `gamma^stage × (Val[stage](k) + raf_contribution(k))` where:
    /// - Val[stage] is a linear combination of bytecode fields weighted by
    ///   powers of `challenges[stage_gamma_base]`
    /// - raf_contribution = gamma^raf_power × k (identity polynomial), if present
    /// - gamma = challenges[gamma_base]
    BytecodeVal {
        poly: PolynomialId,
        /// Stage index (0-4) selects the Val formula.
        stage: u8,
        /// Challenge index for this stage's gamma base (one squeeze → N powers).
        stage_gamma_base: usize,
        /// Number of gamma powers needed for this stage's formula.
        stage_gamma_count: usize,
        /// Challenge index for the overall gamma (shared across all stages).
        gamma_base: usize,
        /// For stages 0/2: power p such that gamma^p × k is added to Val.
        /// Stage 0: raf_gamma_power = Some(5), Stage 2: Some(4), others: None.
        raf_gamma_power: Option<u8>,
        /// For stages 3/4: challenge indices for the r_register eq point.
        /// Used to compute eq(register_index, r_register) per bytecode entry.
        register_eq_challenges: Vec<usize>,
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
            | InputBinding::EqProject { poly, .. }
            | InputBinding::EqGather { poly, .. }
            | InputBinding::EqPushforward { poly, .. }
            | InputBinding::ScaleByChallenge { poly, .. }
            | InputBinding::Transpose { poly, .. }
            | InputBinding::BytecodeVal { poly, .. } => *poly,
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
    /// RAM value check batching gamma: `b"ram_val_check_gamma"`.
    RamValCheckGamma,
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
            Self::RamValCheckGamma => b"ram_val_check_gamma",
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
    /// Compute one sumcheck round polynomial.
    ///
    /// Round 0 (`bind_challenge: None`): reduce only.
    /// Rounds 1+ (`bind_challenge: Some(ch)`): fused bind-at-challenge + reduce.
    SumcheckRound {
        kernel: usize,
        round: usize,
        bind_challenge: Option<usize>,
    },
    /// Initialize a batched sumcheck round: zero the combined accumulator and
    /// update per-instance claims from the previous round's evaluations.
    BatchRoundBegin {
        batch: usize,
        round: usize,
        max_evals: usize,
        bind_challenge: Option<usize>,
    },
    /// Inactive instance contribution: add `coeff * (claim / 2)` to all eval
    /// slots in the combined accumulator, then halve the stored claim.
    BatchInactiveContribution { batch: usize, instance: usize },
    /// Materialize a single kernel input buffer.
    ///
    /// Unconditionally builds/uploads the buffer described by `binding`.
    /// Emitted by the compiler at the exact schedule position where the
    /// buffer is needed — no runtime skip logic.
    Materialize { binding: InputBinding },
    /// Materialize a kernel input, but skip if a buffer of the expected
    /// size already exists (e.g., produced by a prior compute op like
    /// `PrefixSuffixMaterialize`).
    ///
    /// Only used for `Provided` bindings at instance activation where
    /// a cross-instance compute op may have already produced the buffer.
    MaterializeUnlessFresh {
        binding: InputBinding,
        expected_size: usize,
    },
    /// Materialize a kernel input, but only if no buffer exists for this
    /// poly. Used at phase transitions where bound-down buffers from the
    /// previous phase (or other instances) should be preserved.
    MaterializeIfAbsent { binding: InputBinding },
    /// Build the outer eq table for a segmented phase and store it in
    /// the runtime's per-instance segmented state.
    MaterializeSegmentedOuterEq {
        batch: usize,
        instance: usize,
        segmented: SegmentedConfig,
    },
    /// Bind the previous phase's kernel inputs at a challenge.
    /// Emitted at phase transitions (before resolving the new phase's inputs).
    InstanceBindPreviousPhase {
        batch: usize,
        instance: usize,
        kernel: usize,
        challenge: usize,
    },
    /// Capture a scalar from a fully-bound 1-element device buffer into a
    /// challenge slot. Bridges phase boundaries: an intermediate value computed
    /// in one phase becomes a challenge constant for the next phase's formula.
    CaptureScalar {
        poly: PolynomialId,
        challenge: usize,
    },
    /// Standard dense reduce for one instance within a batched round.
    /// Stores the per-instance evaluations for later accumulation.
    InstanceReduce {
        batch: usize,
        instance: usize,
        kernel: usize,
    },
    /// Segmented reduce for one instance (mixed-dimensional inputs).
    /// Uses the outer eq table to weight inner-dimension kernel evaluations.
    InstanceSegmentedReduce {
        batch: usize,
        instance: usize,
        kernel: usize,
        round_within_phase: usize,
        segmented: SegmentedConfig,
    },
    /// Bind kernel inputs for an active instance within a round.
    /// Emitted for rounds after the first within a phase.
    InstanceBind {
        batch: usize,
        instance: usize,
        kernel: usize,
        challenge: usize,
    },
    /// Extrapolate lower-degree instance evals to `max_evals` via interpolation,
    /// then accumulate `coeff * evals[i]` into the combined polynomial.
    BatchAccumulateInstance {
        batch: usize,
        instance: usize,
        max_evals: usize,
        num_evals: usize,
    },
    /// Finalize a batched round: store the combined evaluations as
    /// `last_round_coeffs` for subsequent `AbsorbRoundPoly`.
    BatchRoundFinalize { batch: usize },

    /// Initialize PrefixSuffix state for an instance entering a PS phase.
    PrefixSuffixInit {
        batch: usize,
        instance: usize,
        kernel: usize,
    },
    /// Ingest a challenge into the PrefixSuffix state machine.
    PrefixSuffixBind {
        batch: usize,
        instance: usize,
        challenge: usize,
    },
    /// Compute PrefixSuffix address round evaluations.
    /// Produces 3 evals: `[eval_0, claim - eval_0, eval_2]`.
    PrefixSuffixReduce { batch: usize, instance: usize },
    /// Materialize PrefixSuffix outputs into device buffers and destroy
    /// the PS state. Emitted at the end of a PS phase before transitioning.
    PrefixSuffixMaterialize { batch: usize, instance: usize },

    /// Initialize Gruen-based booleanity state for an instance.
    BooleanityInit {
        batch: usize,
        instance: usize,
        config: BooleanityConfig,
    },
    /// Ingest a challenge into the booleanity state machine.
    BooleanityBind {
        batch: usize,
        instance: usize,
        challenge: usize,
    },
    /// Compute Gruen booleanity round evaluations (4 evals for degree 3).
    BooleanityReduce { batch: usize, instance: usize },
    /// Extract final per-RA-poly evaluations from the booleanity state.
    BooleanityCacheOpenings {
        batch: usize,
        instance: usize,
        ra_poly_ids: Vec<PolynomialId>,
    },

    /// Initialize HW reduction state: compute G_i from RA data + r_cycle,
    /// build eq_bool and eq_virt tables.
    HwReductionInit {
        batch: usize,
        instance: usize,
        config: HwReductionConfig,
    },
    /// Ingest a challenge into the HW reduction state (bind G, eq_bool, eq_virt).
    HwReductionBind {
        batch: usize,
        instance: usize,
        challenge: usize,
    },
    /// Compute HW reduction round evaluations (3 evals for degree 2).
    HwReductionReduce { batch: usize, instance: usize },
    /// Extract final G_i evaluations from the HW reduction state.
    HwReductionCacheOpenings {
        batch: usize,
        instance: usize,
        g_poly_ids: Vec<PolynomialId>,
    },

    /// Extract polynomial evaluation.
    Evaluate { poly: PolynomialId, mode: EvalMode },
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
    /// Interleave-duplicate polynomial buffers: `buf'[2i] = buf'[2i+1] = buf[i]`.
    ///
    /// Extends a polynomial that does not depend on a new low-order variable
    /// (e.g. the streaming variable in the outer Spartan remaining sumcheck).
    /// The resulting buffer is twice as large and ready for standard dense
    /// sumcheck rounds that bind the new variable first (LowToHigh order).
    DuplicateInterleave { polys: Vec<PolynomialId> },
    /// Regroup constraint buffers for the group-split uniskip.
    ///
    /// Transforms Az/Bz from flat layout `[cycle * old_stride + constraint]`
    /// into interleaved layout `[(2 * cycle + group) * new_stride + k]`
    /// where `k` is the constraint index within the group.
    ///
    /// `group_indices[0]` constraints form group 0, `group_indices[1]` form group 1.
    /// Groups are zero-padded to `new_stride`. The group dimension is
    /// INTERLEAVED (group bit at the LOW end / LSB) so that the eq table's
    /// group-selection variable is the first variable bound in LowToHigh order,
    /// matching jolt-core's GruenSplitEqPolynomial layout.
    RegroupConstraints {
        polys: Vec<PolynomialId>,
        /// Indices of original constraints in each group (within `old_stride`).
        group_indices: Vec<Vec<usize>>,
        old_stride: usize,
        new_stride: usize,
        num_cycles: usize,
    },

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

    /// Absorb public instance data into the transcript.
    Preamble,
    /// Begin a new verifier stage (for incremental proof assembly).
    BeginStage { index: usize },
    /// Interpolate round evals → monomial coefficients, absorb into transcript.
    AbsorbRoundPoly {
        num_coeffs: usize,
        tag: DomainSeparator,
        encoding: RoundPolyEncoding,
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
        /// Pre-computed scale: `val * 2^inactive_scale_bits`. The compiler
        /// knows `max_rounds - inst.num_rounds()` at emit time.
        inactive_scale_bits: usize,
    },
    /// Squeeze a Fiat-Shamir challenge.
    Squeeze { challenge: usize },
    /// Compute a derived challenge: `challenges[target] = challenges[base]^exponent`.
    ComputePower {
        target: usize,
        base: usize,
        exponent: u64,
    },
    /// Append a domain separator label (empty payload) to the transcript.
    AppendDomainSeparator { tag: DomainSeparator },
    /// Accumulate a PCS opening claim: (poly data, eval point from stage).
    CollectOpeningClaim {
        poly: PolynomialId,
        at_stage: VerifierStageIndex,
    },
    /// Scale an evaluation by `∏(1 − ch[i])` (Lagrange zero selector).
    /// Used for dense (cycle-only) polynomials whose Dory matrix embedding
    /// includes a `eq(r_addr, 0)` factor.
    ScaleEval {
        poly: PolynomialId,
        factor_challenges: Vec<usize>,
    },
    /// Accumulate a PCS opening claim with an explicit challenge-index point.
    /// Unlike `CollectOpeningClaim`, the point spans multiple stages
    /// (e.g. `[r_address_stage7, r_cycle_stage6]`).
    CollectOpeningClaimAt {
        poly: PolynomialId,
        point_challenges: Vec<usize>,
        /// When set, the polynomial's evaluation table is zero-padded to
        /// `2^committed_num_vars` elements for the RLC combination.
        committed_num_vars: Option<usize>,
    },
    /// Post-proof transcript binding: absorb opening point + joint eval.
    /// Calls `PCS::bind_opening_inputs(transcript, point, eval)`.
    BindOpeningInputs { point_challenges: Vec<usize> },
    /// Evaluate a preprocessed polynomial's MLE at a challenge-derived point.
    ///
    /// Materializes the polynomial from the provider, evaluates the MLE at
    /// `[challenges[i] for i in at_challenges]`, and stores the result in
    /// `state.evaluations[store_as]`. Used for init_eval in RamValCheck.
    EvaluatePreprocessed {
        source: PolynomialId,
        at_challenges: Vec<usize>,
        store_as: PolynomialId,
    },
    /// Release a device buffer (GPU memory).
    ReleaseDevice { poly: PolynomialId },
    /// Release host-side polynomial data (provider memory).
    /// Emitted after `ReduceOpenings` when evaluation tables are no longer needed.
    ReleaseHost { polys: Vec<PolynomialId> },
    /// Copy an evaluation value to a snapshot slot so it survives
    /// later stages that re-evaluate the same polynomial at a new point.
    /// Runtime: `state.evaluations[to] = state.evaluations[from]`.
    SnapshotEval {
        from: PolynomialId,
        to: PolynomialId,
    },
    /// Bind carry buffers for a phase.  These are extra polynomial buffers
    /// that are not kernel inputs but must be bound at the same cadence
    /// so they are the right size when the next phase begins.
    BindCarryBuffers {
        polys: Vec<PolynomialId>,
        challenge: usize,
        order: BindingOrder,
    },
}

/// How to compute and encode the round polynomial for transcript absorption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoundPolyEncoding {
    /// Standard sumcheck: interpolate evaluations at {0, 1, ..., num_coeffs-1}
    /// to monomial coefficients, send compressed (skip c1).
    Compressed,
    /// Univariate skip: convolve composition evaluations with the Lagrange
    /// kernel polynomial, send all coefficients (no compression).
    Uniskip {
        domain_size: usize,
        domain_start: i64,
        tau_challenge: usize,
        zero_base: bool,
    },
}

/// How the runtime should extract a polynomial evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvalMode {
    /// Buffer is fully bound (1 element). Direct scalar read.
    FullyBound,
    /// Buffer has 2 elements after n-1 bind rounds. Interpolate at last
    /// squeezed challenge: `buf[0] + r * (buf[1] - buf[0])`.
    FinalBind,
    /// No buffer — evaluate the last round polynomial at the last squeezed challenge.
    RoundPoly,
}

impl Op {
    pub fn is_compute(&self) -> bool {
        matches!(
            self,
            Op::SumcheckRound { .. }
                | Op::InstanceReduce { .. }
                | Op::InstanceSegmentedReduce { .. }
                | Op::InstanceBind { .. }
                | Op::PrefixSuffixReduce { .. }
                | Op::BooleanityReduce { .. }
                | Op::BooleanityCacheOpenings { .. }
                | Op::HwReductionReduce { .. }
                | Op::HwReductionCacheOpenings { .. }
                | Op::Evaluate { .. }
                | Op::Bind { .. }
                | Op::LagrangeProject { .. }
                | Op::DuplicateInterleave { .. }
                | Op::RegroupConstraints { .. }
        )
    }

    pub fn is_pcs(&self) -> bool {
        matches!(
            self,
            Op::Commit { .. }
                | Op::CommitStreaming { .. }
                | Op::ReduceOpenings
                | Op::Open
                | Op::BindOpeningInputs { .. }
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
                | Op::ComputePower { .. }
                | Op::AppendDomainSeparator { .. }
                | Op::CollectOpeningClaim { .. }
                | Op::ScaleEval { .. }
                | Op::CollectOpeningClaimAt { .. }
                | Op::EvaluatePreprocessed { .. }
                | Op::ReleaseDevice { .. }
                | Op::ReleaseHost { .. }
                | Op::SnapshotEval { .. }
                | Op::BatchRoundBegin { .. }
                | Op::BatchInactiveContribution { .. }
                | Op::Materialize { .. }
                | Op::MaterializeUnlessFresh { .. }
                | Op::MaterializeIfAbsent { .. }
                | Op::MaterializeSegmentedOuterEq { .. }
                | Op::InstanceBindPreviousPhase { .. }
                | Op::CaptureScalar { .. }
                | Op::BatchAccumulateInstance { .. }
                | Op::BatchRoundFinalize { .. }
                | Op::PrefixSuffixInit { .. }
                | Op::PrefixSuffixBind { .. }
                | Op::PrefixSuffixMaterialize { .. }
                | Op::BooleanityInit { .. }
                | Op::BooleanityBind { .. }
                | Op::HwReductionInit { .. }
                | Op::HwReductionBind { .. }
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
    /// Append a domain separator label (empty payload) to the transcript.
    AppendDomainSeparator { tag: DomainSeparator },
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
        /// First integer in the Lagrange domain (symmetric convention: -(N-1)/2).
        domain_start: i64,
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
        domain_start: i64,
    },
    /// Single Lagrange basis polynomial `L_k(r)` at a challenge value `r`,
    /// over the domain `{domain_start, ..., domain_start + domain_size - 1}`.
    LagrangeWeight {
        challenge: usize,
        domain_size: usize,
        domain_start: i64,
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
