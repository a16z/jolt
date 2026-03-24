//! Kernel descriptors for compute backend compilation.
//!
//! A [`KernelDescriptor`] captures the structure of a composition-reduce
//! operation so that each compute backend can compile it into an optimized
//! kernel at setup time. Two shapes are supported:
//!
//! - [`ProductSum`](KernelShape::ProductSum) — explicit fast-path for the
//!   product-of-linear-interpolants pattern that dominates prover time (~80%).
//! - [`Custom`](KernelShape::Custom) — escape hatch using an [`Expr`] for
//!   bespoke compositions (RAM read-write, Spartan outer, etc.).

use crate::Expr;

/// Shape of the composition evaluated at each pair position.
///
/// Two variants: an explicit fast-path for the product-sum pattern
/// (dominant cost in Jolt), and an escape hatch for bespoke compositions.
#[derive(Debug, Clone)]
pub enum KernelShape {
    /// Product of D linear interpolants evaluated on the Toom-Cook grid
    /// $\{1, 2, \ldots, D-1, \infty\}$, summed across multiple product groups.
    ///
    /// At each pair position, produces D evaluations:
    /// $$\text{eval}\[k\] = \sum_{g=0}^{P-1} \prod_{j=0}^{D-1} p_{g,j}(t_k)$$
    ///
    /// where $t_k \in \{1, 2, \ldots, D-1, \infty\}$, $D$ = `num_inputs_per_product`,
    /// and $P$ = `num_products`.
    ///
    /// The Toom-Cook grid enables $O(D \log D)$ multiplications via balanced
    /// binary splitting with extrapolation, vs $O(D^2)$ for the standard grid.
    ///
    /// This covers instruction RA sumchecks ($D \in \{4, 8, 16\}$) and most
    /// claim reductions. GPU backends generate D-specialized kernels with
    /// fully unrolled product evaluation.
    ProductSum {
        /// Number of input buffers per product group ($D$).
        ///
        /// Determines both the product degree and the number of Toom-Cook
        /// evaluations ($D$). Common values: 4, 8, 16, 32.
        num_inputs_per_product: usize,
        /// Number of product groups summed together ($P$).
        num_products: usize,
    },

    /// `opening(0) * opening(1)` — degree 2, 2 inputs.
    ///
    /// Hand-coded kernel that evaluates the product of two linear interpolants
    /// on the standard grid `{0, 2}` (skipping `t=1`). Eliminates stack-VM
    /// dispatch overhead compared to the equivalent `Custom` expression.
    ///
    /// Used by claim reduction stages (S3, S5, S7) where `opening(0) = eq`
    /// and `opening(1) = g` (pre-computed linear combination).
    EqProduct,

    /// `opening(0) * opening(1) * (opening(1) - 1)` — degree 3, 2 inputs.
    ///
    /// Hand-coded kernel for Hamming booleanity checks. Evaluates on the
    /// standard grid `{0, 2, 3}` (skipping `t=1`) without stack-VM dispatch.
    ///
    /// Used by stage 6 to prove that the Hamming weight polynomial is
    /// Boolean-valued on the hypercube.
    HammingBooleanity,

    /// Arbitrary composition defined by a symbolic expression.
    ///
    /// The [`Expr`] references `num_inputs` input buffers via
    /// `Var::Opening(0..num_inputs-1)`. Challenge variables in the expression
    /// are baked in as constants at compile time.
    ///
    /// Used for bespoke compositions that don't fit the `ProductSum` pattern:
    /// RAM read-write checking (phase-based), Spartan outer (multiquadratic),
    /// etc.
    Custom {
        /// The symbolic composition formula.
        expr: Expr,
        /// Number of input buffers the expression reads from.
        num_inputs: usize,
    },
}

impl KernelShape {
    /// Total number of input buffers consumed by this kernel.
    pub fn num_inputs(&self) -> usize {
        match self {
            Self::ProductSum {
                num_inputs_per_product,
                num_products,
            } => num_inputs_per_product * num_products,
            Self::EqProduct => 2,
            Self::HammingBooleanity => 2,
            Self::Custom { num_inputs, .. } => *num_inputs,
        }
    }

    /// Degree of the composition polynomial.
    ///
    /// For `ProductSum`, this equals `num_inputs_per_product` (the product of
    /// D linear functions is degree D). For `Custom`, the caller must specify
    /// it separately in [`KernelDescriptor::degree`].
    pub fn implied_degree(&self) -> Option<usize> {
        match self {
            Self::ProductSum {
                num_inputs_per_product,
                ..
            } => Some(*num_inputs_per_product),
            Self::EqProduct => Some(2),
            Self::HammingBooleanity => Some(3),
            Self::Custom { .. } => None,
        }
    }
}

/// Tensor-product decomposition parameters for split-eq optimization.
///
/// The eq polynomial weight table has a $\sqrt{N}$-decomposition into
/// outer and inner factors:
/// $$\text{eq}(r, x) = \text{eq}_{\text{out}}(r_{\text{out}}, x_{\text{out}})
///     \cdot \text{eq}_{\text{in}}(r_{\text{in}}, x_{\text{in}})$$
///
/// This maps to GPU thread hierarchy: outer index maps to thread groups,
/// inner index maps to threads within a group. On CPU it improves cache
/// locality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorSplit {
    /// Number of outer variables. Outer loop size = $2^{\text{outer\_vars}}$.
    pub outer_vars: usize,
    /// Number of inner variables. Inner loop size = $2^{\text{inner\_vars}}$.
    pub inner_vars: usize,
}

impl TensorSplit {
    /// Total number of variables ($\text{outer\_vars} + \text{inner\_vars}$).
    #[inline]
    pub fn total_vars(&self) -> usize {
        self.outer_vars + self.inner_vars
    }

    /// Constructs a balanced split: $\lceil n/2 \rceil$ outer, $\lfloor n/2 \rfloor$ inner.
    pub fn balanced(num_vars: usize) -> Self {
        let outer = num_vars.div_ceil(2);
        let inner = num_vars - outer;
        Self {
            outer_vars: outer,
            inner_vars: inner,
        }
    }
}

/// Evaluation grid on which the kernel computes composition values.
///
/// Determines which `t` values the kernel evaluates during each sumcheck
/// round, and whether specialized evaluation routines are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalGrid {
    /// Standard grid `{0, 2, 3, ..., degree}`, skipping `t=1`.
    ///
    /// The value at `t=1` is derived from the sumcheck claimed sum.
    /// Used by EqProduct, HammingBooleanity, and Custom kernels.
    Standard {
        /// Composition degree. The grid has `degree` points.
        degree: usize,
    },

    /// Toom-Cook grid `{1, 2, ..., D-1, ∞}` for product-of-interpolants.
    ///
    /// Evaluated via specialized `eval_prod_D` routines using balanced
    /// binary splitting with extrapolation. Used by ProductSum kernels.
    ToomCook {
        /// Number of interpolants per product ($D$).
        d: usize,
    },
}

impl EvalGrid {
    /// Number of evaluation values produced.
    #[inline]
    pub fn num_evals(&self) -> usize {
        match self {
            Self::Standard { degree } => *degree,
            Self::ToomCook { d } => *d,
        }
    }
}

/// How the eq polynomial weight buffer is managed by the evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EqHandling {
    /// Eq buffer is tensor-decomposed into outer × inner factors.
    ///
    /// Maps to GPU thread hierarchy (outer → thread groups, inner → threads).
    /// On CPU, enables cache-friendly blocking.
    TensorSplit(TensorSplit),

    /// Eq buffer is a flat vector of length $2^{n}$.
    Flat,
}

/// Complete description of a composition-reduce kernel.
///
/// Constructed from sumcheck instance definitions at setup time.
/// Each compute backend compiles this into its `CompiledKernel` type
/// via an inherent `compile` method (not through the `ComputeBackend` trait).
///
/// # Examples
///
/// ```
/// use jolt_ir::{KernelDescriptor, KernelShape, TensorSplit};
///
/// // Instruction RA with D=4, 3 product groups
/// let desc = KernelDescriptor {
///     shape: KernelShape::ProductSum {
///         num_inputs_per_product: 4,
///         num_products: 3,
///     },
///     degree: 4,
///     tensor_split: Some(TensorSplit::balanced(20)),
/// };
/// assert_eq!(desc.shape.num_inputs(), 12);
/// assert_eq!(desc.num_evals(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct KernelDescriptor {
    /// The computation performed at each pair position.
    pub shape: KernelShape,

    /// Degree of the composition polynomial.
    ///
    /// For `ProductSum`, this must equal `num_inputs_per_product` and the
    /// kernel produces `degree` evaluations on the Toom-Cook grid.
    /// For `Custom`, this is caller-specified and the kernel produces
    /// `degree` evaluations on the standard grid `{0, 2, ..., degree}`
    /// (skipping `t=1`, which is derived from the sumcheck claim).
    pub degree: usize,

    /// Optional tensor-product decomposition of the weight buffer.
    ///
    /// When present, enables the split-eq optimization: GPU maps outer
    /// variables to thread groups and inner variables to threads within
    /// a group; CPU uses it for cache-friendly blocking.
    ///
    /// When `None`, the weight buffer is flat (standard reduction).
    pub tensor_split: Option<TensorSplit>,
}

impl KernelDescriptor {
    /// Number of evaluation values produced per pair position.
    ///
    /// For `ProductSum`: $D$ (Toom-Cook grid `{1, ..., D-1, ∞}`).
    /// For standard-grid kernels: `degree` (grid `{0, 2, ..., degree}`,
    /// skipping `t=1` which is derived from the sumcheck claim).
    #[inline]
    pub fn num_evals(&self) -> usize {
        match &self.shape {
            KernelShape::ProductSum {
                num_inputs_per_product,
                ..
            } => *num_inputs_per_product,
            KernelShape::EqProduct => 2,         // {0, 2}
            KernelShape::HammingBooleanity => 3, // {0, 2, 3}
            KernelShape::Custom { .. } => self.degree,
        }
    }

    /// Total number of input buffers.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.shape.num_inputs()
    }

    /// Validates that the descriptor is internally consistent.
    ///
    /// Returns `true` if:
    /// - `ProductSum` degree matches `num_inputs_per_product`
    /// - `TensorSplit` total vars is positive (when present)
    /// - `num_inputs_per_product` and `num_products` are non-zero
    pub fn is_valid(&self) -> bool {
        let shape_ok = match &self.shape {
            KernelShape::ProductSum {
                num_inputs_per_product,
                num_products,
            } => {
                *num_inputs_per_product > 0
                    && *num_products > 0
                    && self.degree == *num_inputs_per_product
            }
            KernelShape::EqProduct => self.degree == 2,
            KernelShape::HammingBooleanity => self.degree == 3,
            KernelShape::Custom { num_inputs, .. } => *num_inputs > 0 && self.degree > 0,
        };
        shape_ok && self.tensor_split.is_none_or(|ts| ts.total_vars() > 0)
    }

    /// Evaluation grid for this kernel.
    ///
    /// Determines the set of `t` values at which the composition is evaluated
    /// during each sumcheck round, and whether Toom-Cook routines are used.
    #[inline]
    pub fn eval_grid(&self) -> EvalGrid {
        match &self.shape {
            KernelShape::ProductSum {
                num_inputs_per_product,
                ..
            } => EvalGrid::ToomCook {
                d: *num_inputs_per_product,
            },
            _ => EvalGrid::Standard {
                degree: self.degree,
            },
        }
    }

    /// How the eq polynomial weight buffer should be managed.
    ///
    /// `TensorSplit` enables the split-eq optimization for both GPU and CPU.
    /// `Flat` uses a standard flat buffer.
    #[inline]
    pub fn eq_handling(&self) -> EqHandling {
        match self.tensor_split {
            Some(ts) => EqHandling::TensorSplit(ts),
            None => EqHandling::Flat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExprBuilder;

    #[test]
    fn product_sum_num_inputs() {
        let shape = KernelShape::ProductSum {
            num_inputs_per_product: 4,
            num_products: 3,
        };
        assert_eq!(shape.num_inputs(), 12);
    }

    #[test]
    fn product_sum_implied_degree() {
        let shape = KernelShape::ProductSum {
            num_inputs_per_product: 8,
            num_products: 1,
        };
        assert_eq!(shape.implied_degree(), Some(8));
    }

    #[test]
    fn custom_num_inputs() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let shape = KernelShape::Custom {
            expr: b.build(a * bv),
            num_inputs: 2,
        };
        assert_eq!(shape.num_inputs(), 2);
        assert_eq!(shape.implied_degree(), None);
    }

    #[test]
    fn tensor_split_balanced_even() {
        let ts = TensorSplit::balanced(20);
        assert_eq!(ts.outer_vars, 10);
        assert_eq!(ts.inner_vars, 10);
        assert_eq!(ts.total_vars(), 20);
    }

    #[test]
    fn tensor_split_balanced_odd() {
        let ts = TensorSplit::balanced(15);
        assert_eq!(ts.outer_vars, 8);
        assert_eq!(ts.inner_vars, 7);
        assert_eq!(ts.total_vars(), 15);
    }

    #[test]
    fn descriptor_num_evals() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 4,
                num_products: 2,
            },
            degree: 4,
            tensor_split: None,
        };
        // ProductSum: num_evals = D (Toom-Cook grid)
        assert_eq!(desc.num_evals(), 4);
        assert_eq!(desc.num_inputs(), 8);
    }

    #[test]
    fn descriptor_with_tensor_split() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 16,
                num_products: 1,
            },
            degree: 16,
            tensor_split: Some(TensorSplit::balanced(20)),
        };
        assert!(desc.is_valid());
        assert_eq!(desc.tensor_split.unwrap().outer_vars, 10);
        assert_eq!(desc.tensor_split.unwrap().inner_vars, 10);
    }

    #[test]
    fn descriptor_valid_product_sum() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 4,
                num_products: 3,
            },
            degree: 4,
            tensor_split: None,
        };
        assert!(desc.is_valid());
    }

    #[test]
    fn descriptor_invalid_degree_mismatch() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 4,
                num_products: 3,
            },
            degree: 8, // mismatch
            tensor_split: None,
        };
        assert!(!desc.is_valid());
    }

    #[test]
    fn descriptor_invalid_zero_products() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 4,
                num_products: 0,
            },
            degree: 4,
            tensor_split: None,
        };
        assert!(!desc.is_valid());
    }

    #[test]
    fn descriptor_valid_custom() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let desc = KernelDescriptor {
            shape: KernelShape::Custom {
                expr: b.build(a * a - a),
                num_inputs: 1,
            },
            degree: 2,
            tensor_split: None,
        };
        assert!(desc.is_valid());
    }

    #[test]
    fn descriptor_custom_with_split() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let desc = KernelDescriptor {
            shape: KernelShape::Custom {
                expr: b.build(a * bv),
                num_inputs: 2,
            },
            degree: 2,
            tensor_split: Some(TensorSplit {
                outer_vars: 10,
                inner_vars: 10,
            }),
        };
        assert!(desc.is_valid());
        // Custom: num_evals = degree (standard grid, skipping t=1)
        assert_eq!(desc.num_evals(), 2);
    }

    #[test]
    fn eval_grid_product_sum_is_toom_cook() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 8,
                num_products: 2,
            },
            degree: 8,
            tensor_split: None,
        };
        assert_eq!(desc.eval_grid(), EvalGrid::ToomCook { d: 8 });
        assert_eq!(desc.eval_grid().num_evals(), 8);
    }

    #[test]
    fn eval_grid_eq_product_is_standard() {
        let desc = KernelDescriptor {
            shape: KernelShape::EqProduct,
            degree: 2,
            tensor_split: None,
        };
        assert_eq!(desc.eval_grid(), EvalGrid::Standard { degree: 2 });
        assert_eq!(desc.eval_grid().num_evals(), 2);
    }

    #[test]
    fn eval_grid_hamming_is_standard() {
        let desc = KernelDescriptor {
            shape: KernelShape::HammingBooleanity,
            degree: 3,
            tensor_split: None,
        };
        assert_eq!(desc.eval_grid(), EvalGrid::Standard { degree: 3 });
        assert_eq!(desc.eval_grid().num_evals(), 3);
    }

    #[test]
    fn eval_grid_custom_is_standard() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let desc = KernelDescriptor {
            shape: KernelShape::Custom {
                expr: b.build(a * a * a),
                num_inputs: 1,
            },
            degree: 3,
            tensor_split: None,
        };
        assert_eq!(desc.eval_grid(), EvalGrid::Standard { degree: 3 });
    }

    #[test]
    fn eval_grid_num_evals_matches_descriptor() {
        let shapes: Vec<KernelDescriptor> = vec![
            KernelDescriptor {
                shape: KernelShape::ProductSum {
                    num_inputs_per_product: 4,
                    num_products: 3,
                },
                degree: 4,
                tensor_split: None,
            },
            KernelDescriptor {
                shape: KernelShape::EqProduct,
                degree: 2,
                tensor_split: None,
            },
            KernelDescriptor {
                shape: KernelShape::HammingBooleanity,
                degree: 3,
                tensor_split: None,
            },
        ];
        for desc in &shapes {
            assert_eq!(desc.eval_grid().num_evals(), desc.num_evals());
        }
    }

    #[test]
    fn eq_handling_flat_when_no_split() {
        let desc = KernelDescriptor {
            shape: KernelShape::EqProduct,
            degree: 2,
            tensor_split: None,
        };
        assert_eq!(desc.eq_handling(), EqHandling::Flat);
    }

    #[test]
    fn eq_handling_tensor_when_split() {
        let ts = TensorSplit::balanced(20);
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 4,
                num_products: 3,
            },
            degree: 4,
            tensor_split: Some(ts),
        };
        assert_eq!(desc.eq_handling(), EqHandling::TensorSplit(ts));
    }
}
