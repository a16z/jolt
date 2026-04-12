//! Kernel specification: the compiler's output, the backend's input.
//!
//! [`KernelSpec`] is the boundary between protocol-aware compilation and
//! hardware-aware codegen. The compiler decides WHAT to compute (formula,
//! iteration pattern, evaluation grid). The backend decides HOW (codegen,
//! parallelism, memory layout).

use serde::{Deserialize, Serialize};

use crate::formula::{BindingOrder, Formula};

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

    /// Sparse merge-join over sorted entries.
    ///
    /// Entries with adjacent keys `(2k, 2k+1)` are paired and composed.
    /// One extra input follows the formula value columns: a sorted `u64`
    /// key column (stored as `DeviceBuffer::U64`).
    ///
    /// Entries present in only one half are paired with checkpoint defaults.
    /// Used for read-write memory checking where the address space is sparse.
    Sparse,

    /// Multi-phase prefix-suffix decomposition for instruction lookup sumchecks.
    ///
    /// Decomposes a high-dimensional sumcheck (total_address_bits address +
    /// cycle rounds) into manageable sub-phases. Each sub-phase binds
    /// `chunk_bits` address variables using prefix-suffix MLE decomposition.
    ///
    /// The runtime implements:
    /// - Per sub-phase: build P (prefix) and Q (suffix) polynomials from
    ///   expanding tables and trace data, then evaluate Σ P×Q for each point
    /// - Between sub-phases: update expanding tables and prefix checkpoints
    /// - At address→cycle transition: materialize RA polys and combined_val
    ///
    /// The formula field is ignored — the iteration handles all evaluation.
    PrefixSuffix {
        /// Total address bits in the decomposition (LOG_K_INSTRUCTION = 128).
        total_address_bits: usize,
        /// Bits per sub-phase (LOG_K / num_phases).
        chunk_bits: usize,
        /// Number of sub-phases in the address decomposition (8 or 16).
        num_phases: usize,
        /// Log₂ of the virtual RA polynomial chunk size.
        ra_virtual_log_k_chunk: usize,
        /// Challenge index for γ (instruction read-RAF batching).
        gamma: usize,
        /// Challenge indices for r_reduction (log_T entries, BIG_ENDIAN).
        /// The cycle opening point from the prior InstructionClaimReduction
        /// sumcheck. Used to build eq(r_reduction, j) for cycle weighting
        /// in both the address-phase suffix accumulation and the cycle-phase
        /// eq factor.
        r_reduction: Vec<usize>,
        /// PolynomialIds where materialized RA polys are stored at the
        /// address→cycle transition. The cycle phase kernel reads these
        /// as Provided inputs.
        output_ra_polys: Vec<crate::polynomial_id::PolynomialId>,
        /// PolynomialId where the combined_val polynomial is stored at
        /// the address→cycle transition.
        output_combined_val: crate::polynomial_id::PolynomialId,
    },

    /// Gruen-based booleanity sumcheck.
    ///
    /// Two-phase iteration: Phase 1 binds address variables using G_d
    /// projections and an expanding F table; Phase 2 binds cycle variables
    /// using pre-scaled H polynomials. The formula field is ignored — the
    /// runtime handles all evaluation via `BooleanityInit/Bind/Reduce` ops.
    Booleanity {
        config: crate::module::BooleanityConfig,
    },

    /// Fused HammingWeight + Address Reduction sumcheck (Stage 7).
    ///
    /// Operates on G_i polynomials (cycle-projected RA) with eq_bool and
    /// eq_virt tables. The formula field is ignored — the runtime handles
    /// evaluation via `HwReductionInit/Bind/Reduce` ops.
    HammingWeightReduction {
        config: crate::module::HwReductionConfig,
    },

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
        tau_challenge: usize,
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
        }
    }
}
