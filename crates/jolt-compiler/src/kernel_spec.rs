//! Kernel specification: the compiler's output, the backend's input.
//!
//! [`KernelSpec`] is the boundary between protocol-aware compilation and
//! hardware-aware codegen. The compiler decides WHAT to compute (formula,
//! iteration pattern, evaluation grid). The backend decides HOW (codegen,
//! parallelism, memory layout).

use serde::{Deserialize, Serialize};

use crate::formula::{BindingOrder, Formula};
use crate::module::ChallengeIdx;

/// Full algorithmic description of a sumcheck kernel.
///
/// Produced by the compiler from protocol-level definitions. Consumed by
/// [`ComputeBackend::compile`] to produce backend-native executable code.
///
/// The spec captures:
/// - **Formula**: the sum-of-products composition to evaluate at each position
/// - **Evaluation grid**: how many points, determining round polynomial degree
/// - **Iteration pattern**: dense pairwise or tensor-factored
/// - **Binding order**: which end of the hypercube to bind first
///
/// The compiled kernel bakes ALL of this in. At dispatch time, only the
/// actual buffer data and challenge values are provided.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSpec {
    /// Sum-of-products composition formula.
    pub formula: Formula,
    /// Number of evaluation points on the grid (equals `formula.degree()`).
    ///
    /// Standard grid: `degree` points at `{0, 1, ..., degree-1}`.
    /// Toom-Cook grid: `D` points at `{1, ..., D-1, ∞}`.
    pub num_evals: usize,
    /// How to traverse input data during reduce.
    pub iteration: Iteration,
    /// Variable binding direction.
    pub binding_order: BindingOrder,
    /// Optional Dao-Thaler + Gruen cubic-assembly fast path.
    ///
    /// When `Some`, the runtime may invoke `gruen_segmented_reduce` on
    /// backends that support it, assembling the round cubic via the
    /// prev-claim trick instead of the standard evaluation grid.
    /// Backends that don't recognize the hint fall back to the generic
    /// `segmented_reduce`.
    #[serde(default)]
    pub gruen_hint: Option<GruenHint>,
}

/// Metadata directing the runtime to use the Gruen cubic-assembly fast path
/// on a segmented dense kernel whose formula factors as `eq(w, x) · q(x)`
/// with `q` a specific linear-combo-of-bilinear-product shape.
///
/// For kernel 3 (RAM read-write phase 1):
/// `q(x) = a(x) · ((1 + γ) · b(x) + γ · c(x))`
///
/// Equivalently `q(x) = a(x) · (b(x) + γ · (b(x) + c(x)))`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GruenHint {
    /// Which formula input is the eq-table (length `2^inner_num_vars` after binding).
    pub eq_input: u32,
    /// Per-round challenge indices: `w_current` at round `k` is
    /// `challenges[eq_challenges[k].0]`.
    pub eq_challenges: Vec<ChallengeIdx>,
    /// The `q(x)` factorization baked in as a linear combination.
    pub q_lincombo: LinComboQ,
}

/// `q(x) = a(x) · ((1 + γ) · b(x) + γ · c(x))` shape for [`GruenHint`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinComboQ {
    pub a_input: u32,
    pub b_input: u32,
    pub c_input: u32,
    pub gamma_challenge: ChallengeIdx,
}

/// Data traversal strategy for a sumcheck kernel.
///
/// Determines the iteration pattern used by
/// [`ComputeBackend::reduce`] and [`ComputeBackend::bind`],
/// and the meaning of extra inputs beyond the formula's value columns.
///
/// # Input layout convention
///
/// Inputs passed to `reduce`/`bind` follow a fixed layout:
///
/// | Positions | Contents |
/// |-----------|----------|
/// | `0 .. formula.num_inputs` | Formula value columns (`Input(i)` in the formula) |
/// | After value columns | Extra inputs specified by the `Iteration` variant |
///
/// For [`Dense`](Iteration::Dense): no extra inputs.
/// For [`DenseTensor`](Iteration::DenseTensor): two extra inputs (outer eq, inner eq).
/// For [`Sparse`](Iteration::Sparse): one extra input (sorted u64 key column).
///
/// Instance-type subprotocols (PrefixSuffix, Booleanity, HammingWeightReduction)
/// are handled via [`InstanceConfig`](crate::module::InstanceConfig), not this enum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Iteration {
    /// Dense pairwise: iterate over adjacent pairs in contiguous buffers.
    ///
    /// All inputs are formula value columns. Pair layout depends on
    /// [`BindingOrder`]:
    /// - `LowToHigh`: pairs at `(buf[2i], buf[2i+1])`
    /// - `HighToLow`: pairs at `(buf[i], buf[i + n/2])`
    Dense,

    /// Dense with factored eq weights (split-eq tensor product).
    ///
    /// `eq(x_out, x_in) = outer[x_out] · inner[x_in]`
    ///
    /// Enables cache-friendly nested iteration: the outer loop accumulates
    /// weighted inner sums, reducing intermediate results between outer
    /// iterations. Two extra inputs follow the formula value columns:
    /// `outer_eq` then `inner_eq`.
    DenseTensor,

    /// Dense with Dao-Thaler + Gruen split-eq cubic assembly.
    ///
    /// For `formula = eq(x) · q(x)` where `q(x)` is a deg-2 product of
    /// multilinear factors, the round cubic `s(X) = l(X) · q(X)` is
    /// assembled from `(q(0), q(∞) coefficient)` plus the previous-round
    /// claim — avoiding the 4-point Toom-Cook eval grid. See
    /// <https://eprint.iacr.org/2024/1210.pdf>.
    ///
    /// Two extra inputs follow the formula value columns: `outer_eq` then
    /// `inner_eq` (same layout as [`DenseTensor`](Iteration::DenseTensor)),
    /// but the runtime routes to `reduce_dense_gruen_deg2` instead of the
    /// tensor reducer and assembles the cubic via `gruen_cubic_evals`.
    Gruen,

    /// Sparse merge-join over sorted entries.
    ///
    /// Entries with adjacent keys `(2k, 2k+1)` are paired and composed.
    /// One extra input follows the formula value columns: a sorted `u64`
    /// key column (stored as `DeviceBuffer::U64`).
    ///
    /// Entries present in only one half are paired with checkpoint defaults.
    /// Used for read-write memory checking where the address space is sparse.
    Sparse,

    /// Lagrange-domain evaluation for univariate skip rounds.
    ///
    /// Some inputs are "cycle-indexed" (length T) and others are
    /// "domain-indexed" (length T × stride, accessed at `buf[c*stride + d]`
    /// for cycle c and domain point d). `domain_indexed[j]` is `true` if
    /// formula input `j` is domain-indexed.
    ///
    /// The kernel produces `2K - 1` evaluation sums: K base evaluations
    /// at domain points {domain_start, ..., domain_start + K - 1} and
    /// K - 1 extended evaluations at points beyond the base domain
    /// (needed to interpolate the degree-2(K-1) composition polynomial).
    Domain {
        /// Number of points in the Lagrange domain (K).
        domain_size: usize,
        /// Padded stride between cycles in domain-indexed buffers.
        stride: usize,
        /// First integer in the evaluation domain.
        domain_start: i64,
        /// Which formula inputs are domain-indexed (vs cycle-indexed).
        domain_indexed: Vec<bool>,
        /// Challenge index for the Lagrange kernel evaluation point (τ_high).
        /// Used by `AbsorbRoundPoly` post-processing to convolve with L(τ_high, Y).
        tau_challenge: ChallengeIdx,
        /// Whether base domain evaluations (first K values) should be zeroed
        /// before interpolation. True for R1CS outer sumcheck where Az*Bz = Cz
        /// guarantees t1 vanishes on the base domain. False for product/other
        /// uniskip rounds where base evaluations are non-zero.
        zero_base: bool,
    },
}

impl KernelSpec {
    /// Create a spec that derives `num_evals` from the formula's degree.
    pub fn new(formula: Formula, iteration: Iteration, binding_order: BindingOrder) -> Self {
        let num_evals = formula.degree();
        Self {
            formula,
            num_evals,
            iteration,
            binding_order,
            gruen_hint: None,
        }
    }
}
