use jolt_field::Field;

use crate::builder::ExprBuilder;
use crate::expr::Expr;
use crate::kernel::{KernelDescriptor, KernelShape};
use crate::normalize::{SopValue, SumOfProducts};
use crate::polynomial_id::PolynomialId;

/// Maps an opening variable index to a concrete polynomial identity.
///
/// `var_id` matches `Var::Opening(id)` in the expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningBinding {
    pub var_id: u32,
    pub polynomial: PolynomialId,
}

/// A complete claim definition: expression + binding metadata.
///
/// This is the single source of truth for a sumcheck claim formula. All
/// backends (evaluation, R1CS, Lean4, circuit) consume this structure.
///
/// # Example
///
/// ```
/// use jolt_ir::{ExprBuilder, ClaimDefinition, OpeningBinding, PolynomialId};
///
/// let b = ExprBuilder::new();
/// let h = b.opening(0);
/// let gamma = b.challenge(0);
/// let expr = b.build(gamma * (h * h - h));
///
/// let claim = ClaimDefinition {
///     expr,
///     opening_bindings: vec![
///         OpeningBinding { var_id: 0, polynomial: PolynomialId::HammingWeight },
///     ],
///     num_challenges: 1,
/// };
///
/// assert_eq!(claim.opening_bindings.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct ClaimDefinition {
    pub expr: Expr,
    pub opening_bindings: Vec<OpeningBinding>,
    pub num_challenges: u32,
}

impl ClaimDefinition {
    /// Evaluate the claim expression with concrete opening and challenge values.
    ///
    /// Convenience wrapper around `Expr::evaluate`. The caller is responsible
    /// for ordering `openings` and `challenges` to match the `var_id` indices
    /// in the binding metadata.
    pub fn evaluate<F: Field>(&self, openings: &[F], challenges: &[F]) -> F {
        self.expr.evaluate(openings, challenges)
    }

    /// Convert the claim's output formula into a [`KernelDescriptor`].
    ///
    /// The descriptor describes the inner computation `g(x)` where the full
    /// sumcheck identity is `Σ eq(r, x) · g(x) = claimed_sum`.
    ///
    /// The returned descriptor uses `Opening(0)` for the eq buffer and shifts
    /// the claim's `Opening(i)` to `Opening(i+1)`. Challenge variables are
    /// preserved as-is — the caller supplies concrete values (with the eq
    /// factor removed) at kernel compile time via `compile_with_challenges`.
    ///
    /// Recognizes the following fast-path patterns:
    /// - [`HammingBooleanity`](KernelShape::HammingBooleanity): `eq · h · (h − 1)`
    /// - [`EqProduct`](KernelShape::EqProduct): `eq · g` (single opening per term)
    /// - Falls back to [`Custom`](KernelShape::Custom) for all other expressions.
    pub fn to_kernel_descriptor(&self) -> KernelDescriptor {
        let sop = self.expr.to_sum_of_products();

        if let Some(desc) = Self::try_hamming_booleanity(&sop) {
            return desc;
        }
        if let Some(desc) = Self::try_eq_product(&sop) {
            return desc;
        }

        Self::build_custom_descriptor(&sop, self.opening_bindings.len())
    }

    /// Detect HammingBooleanity: `eq · h · (h − 1)`.
    ///
    /// Pattern in SoP form: exactly 2 terms, 1 distinct opening variable,
    /// one term with `opening²` and the other with `opening`, each scaled
    /// by a challenge factor (the eq weight). The negation in `h² − h`
    /// comes from the runtime challenge values, not the SoP coefficients.
    fn try_hamming_booleanity(sop: &SumOfProducts) -> Option<KernelDescriptor> {
        if sop.terms.len() != 2 {
            return None;
        }

        let opening_factors = |idx: usize| -> Vec<u32> {
            sop.terms[idx]
                .factors
                .iter()
                .filter_map(|f| match f {
                    SopValue::Opening(id) => Some(*id),
                    _ => None,
                })
                .collect()
        };
        let challenge_count = |idx: usize| -> usize {
            sop.terms[idx]
                .factors
                .iter()
                .filter(|f| matches!(f, SopValue::Challenge(_)))
                .count()
        };

        let o0 = opening_factors(0);
        let o1 = opening_factors(1);

        // One term has 2 opening factors (h²), the other has 1 (h)
        let (sq_idx, lin_idx) = if o0.len() == 2 && o1.len() == 1 {
            (0, 1)
        } else if o0.len() == 1 && o1.len() == 2 {
            (1, 0)
        } else {
            return None;
        };

        let sq_openings = opening_factors(sq_idx);
        let lin_openings = opening_factors(lin_idx);

        // All opening references must be the same variable
        if sq_openings[0] != sq_openings[1] || sq_openings[0] != lin_openings[0] {
            return None;
        }

        // Both terms must have exactly 1 challenge factor (the eq weight)
        if challenge_count(sq_idx) != 1 || challenge_count(lin_idx) != 1 {
            return None;
        }

        Some(KernelDescriptor {
            shape: KernelShape::HammingBooleanity,
            degree: 3,
            tensor_split: None,
        })
    }

    /// Detect EqProduct: `eq · g` where every term is linear in openings.
    ///
    /// Pattern in SoP form: every term has exactly 1 opening factor and at
    /// least 1 challenge factor. The caller pre-computes `g = Σ c_i · p_i`
    /// into a single buffer, then uses the EqProduct kernel.
    fn try_eq_product(sop: &SumOfProducts) -> Option<KernelDescriptor> {
        if sop.terms.is_empty() {
            return None;
        }

        for term in &sop.terms {
            let n_openings = term
                .factors
                .iter()
                .filter(|f| matches!(f, SopValue::Opening(_)))
                .count();
            let n_challenges = term
                .factors
                .iter()
                .filter(|f| matches!(f, SopValue::Challenge(_)))
                .count();
            if n_openings != 1 || n_challenges < 1 {
                return None;
            }
        }

        Some(KernelDescriptor {
            shape: KernelShape::EqProduct,
            degree: 2,
            tensor_split: None,
        })
    }

    /// Build a Custom kernel descriptor from the SoP representation.
    ///
    /// The kernel expression is `Opening(0) * inner` where `inner` has the
    /// claim's `Opening(i)` shifted to `Opening(i+1)`.
    fn build_custom_descriptor(sop: &SumOfProducts, num_claim_openings: usize) -> KernelDescriptor {
        let b = ExprBuilder::new();
        let eq = b.opening(0);

        // Rebuild the expression from SoP terms with shifted opening indices
        let mut sum_handle = None;
        for term in &sop.terms {
            let mut product = b.constant(term.coefficient);
            for factor in &term.factors {
                let node = match factor {
                    SopValue::Constant(c) => b.constant(*c),
                    SopValue::Opening(id) => b.opening(*id + 1),
                    SopValue::Challenge(id) => b.challenge(*id),
                };
                product = product * node;
            }
            sum_handle = Some(match sum_handle {
                None => product,
                Some(acc) => acc + product,
            });
        }

        let inner = sum_handle.unwrap_or_else(|| b.zero());
        let kernel_expr = b.build(eq * inner);

        // Degree = max opening factors per term + 1 (for eq)
        let max_opening_degree = sop
            .terms
            .iter()
            .map(|t| {
                t.factors
                    .iter()
                    .filter(|f| matches!(f, SopValue::Opening(_)))
                    .count()
            })
            .max()
            .unwrap_or(0);

        KernelDescriptor {
            shape: KernelShape::Custom {
                expr: kernel_expr,
                num_inputs: num_claim_openings + 1, // +1 for eq
            },
            degree: max_opening_degree + 1,
            tensor_split: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;
    use crate::kernel::KernelShape;

    #[test]
    fn to_kernel_hamming_booleanity() {
        let claim = crate::zkvm::claims::ram::hamming_booleanity();
        let desc = claim.to_kernel_descriptor();
        assert!(matches!(desc.shape, KernelShape::HammingBooleanity));
        assert_eq!(desc.degree, 3);
        assert!(desc.is_valid());
    }

    #[test]
    fn to_kernel_eq_product_registers_reduction() {
        let claim = crate::zkvm::claims::reductions::registers_claim_reduction();
        let desc = claim.to_kernel_descriptor();
        assert!(matches!(desc.shape, KernelShape::EqProduct));
        assert_eq!(desc.degree, 2);
        assert!(desc.is_valid());
    }

    #[test]
    fn to_kernel_eq_product_increment_reduction() {
        let claim = crate::zkvm::claims::reductions::increment_claim_reduction();
        let desc = claim.to_kernel_descriptor();
        assert!(matches!(desc.shape, KernelShape::EqProduct));
        assert_eq!(desc.degree, 2);
        assert!(desc.is_valid());
    }

    #[test]
    fn to_kernel_custom_ram_rw() {
        let claim = crate::zkvm::claims::ram::ram_read_write_checking();
        let desc = claim.to_kernel_descriptor();
        assert!(matches!(desc.shape, KernelShape::Custom { .. }));
        assert_eq!(desc.degree, 3); // eq · ra · val (or eq · ra · inc)
        assert!(desc.is_valid());
        if let KernelShape::Custom { num_inputs, .. } = desc.shape {
            assert_eq!(num_inputs, 4); // eq + 3 openings
        }
    }

    #[test]
    fn to_kernel_custom_ram_ra_virtual() {
        let claim = crate::zkvm::claims::ram::ram_ra_virtual(4);
        let desc = claim.to_kernel_descriptor();
        // c0 · o0 · o1 · o2 · o3: challenge factor means not pure product → Custom
        assert!(matches!(desc.shape, KernelShape::Custom { .. }));
        assert_eq!(desc.degree, 5); // eq + 4 openings
        assert!(desc.is_valid());
    }

    #[test]
    fn to_kernel_eq_product_raf_evaluation() {
        let claim = crate::zkvm::claims::ram::ram_raf_evaluation();
        let desc = claim.to_kernel_descriptor();
        // c0 · ra: 1 challenge, 1 opening → EqProduct
        assert!(matches!(desc.shape, KernelShape::EqProduct));
        assert_eq!(desc.degree, 2);
        assert!(desc.is_valid());
    }

    #[test]
    fn claim_definition_construction() {
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        let claim = ClaimDefinition {
            expr,
            opening_bindings: vec![OpeningBinding {
                var_id: 0,
                polynomial: PolynomialId::HammingWeight,
            }],
            num_challenges: 1,
        };

        assert_eq!(claim.opening_bindings.len(), 1);
        assert_eq!(claim.num_challenges, 1);
    }

    #[test]
    fn claim_with_batching_coefficients() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let alpha = b.challenge(0);
        let expr = b.build(a + alpha * bv);

        let claim = ClaimDefinition {
            expr,
            opening_bindings: vec![
                OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::RamInc,
                },
                OpeningBinding {
                    var_id: 1,
                    polynomial: PolynomialId::RdInc,
                },
            ],
            num_challenges: 1,
        };

        assert_eq!(claim.opening_bindings.len(), 2);
    }

    #[test]
    fn claim_with_no_challenges() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(a * a);

        let claim = ClaimDefinition {
            expr,
            opening_bindings: vec![OpeningBinding {
                var_id: 0,
                polynomial: PolynomialId::RamInc,
            }],
            num_challenges: 0,
        };

        assert_eq!(claim.num_challenges, 0);
    }
}
