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

    /// Compile the claim into a kernel descriptor with materialized challenge values.
    ///
    /// Returns `(descriptor, materialized_values)` where the interpretation of
    /// `materialized_values` depends on the kernel shape:
    /// - **EqProduct**: per-opening pre-combination weights for `g = Σ w_i · poly_i`
    /// - **HammingBooleanity**: `[eq_scale]` — scale factor for the eq buffer
    /// - **Custom**: challenge values indexed by slot for expression baking
    pub fn compile_descriptor<F: Field>(&self, challenges: &[F]) -> (KernelDescriptor, Vec<F>) {
        let sop = self.expr.to_sum_of_products();

        if let Some(desc) = Self::try_hamming_booleanity(&sop) {
            let scale = Self::extract_hamming_eq_scale::<F>(&sop, challenges);
            return (desc, vec![scale]);
        }
        if let Some(desc) = Self::try_eq_product(&sop) {
            let weights = Self::extract_eq_product_coefficients::<F>(
                &sop,
                challenges,
                self.opening_bindings.len(),
            );
            return (desc, weights);
        }

        let challenge_values = Self::extract_custom_challenges::<F>(&sop, challenges);
        let desc = Self::build_custom_descriptor(&sop, self.opening_bindings.len());
        (desc, challenge_values)
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
        let mut has_nonzero = false;
        for term in &sop.terms {
            if term.coefficient == 0 {
                continue;
            }
            has_nonzero = true;
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

        if !has_nonzero {
            return None;
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

    /// Per-opening pre-combination weights for EqProduct claims.
    ///
    /// For each SoP term (exactly 1 opening factor), computes
    /// `coefficient × Π challenges[c_id]` and accumulates by opening index.
    fn extract_eq_product_coefficients<F: Field>(
        sop: &SumOfProducts,
        challenges: &[F],
        num_openings: usize,
    ) -> Vec<F> {
        let mut weights = vec![F::zero(); num_openings];
        for term in &sop.terms {
            if term.coefficient == 0 {
                continue;
            }
            let opening_id = term
                .factors
                .iter()
                .find_map(|f| match f {
                    SopValue::Opening(id) => Some(*id as usize),
                    _ => None,
                })
                .expect("EqProduct term must have an opening factor");

            let mut w = F::from_i128(term.coefficient);
            for factor in &term.factors {
                if let SopValue::Challenge(id) = factor {
                    w *= challenges[*id as usize];
                }
            }
            weights[opening_id] += w;
        }
        weights
    }

    /// Eq scale factor from HammingBooleanity's squared term.
    ///
    /// The `H²` term carries the positive eq factor:
    /// `coefficient × Π challenges[c_id]`.
    fn extract_hamming_eq_scale<F: Field>(sop: &SumOfProducts, challenges: &[F]) -> F {
        let sq_term = sop
            .terms
            .iter()
            .find(|t| {
                t.factors
                    .iter()
                    .filter(|f| matches!(f, SopValue::Opening(_)))
                    .count()
                    == 2
            })
            .expect("HammingBooleanity must have a squared term");

        let mut scale = F::from_i128(sq_term.coefficient);
        for factor in &sq_term.factors {
            if let SopValue::Challenge(id) = factor {
                scale *= challenges[*id as usize];
            }
        }
        scale
    }

    /// Challenge values by slot index for Custom kernels.
    ///
    /// Scans SoP terms for `Challenge(id)` references and returns
    /// concrete values indexed `[0, 1, ..., max_id]`.
    fn extract_custom_challenges<F: Field>(sop: &SumOfProducts, challenges: &[F]) -> Vec<F> {
        let max_id = sop
            .terms
            .iter()
            .flat_map(|t| t.factors.iter())
            .filter_map(|f| match f {
                SopValue::Challenge(id) => Some(*id),
                _ => None,
            })
            .max();

        match max_id {
            Some(max) => (0..=max).map(|i| challenges[i as usize]).collect(),
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;
    use crate::kernel::KernelShape;
    use jolt_field::{Field, Fr};

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

    #[test]
    fn compile_hamming_booleanity_scale() {
        let claim = crate::zkvm::claims::ram::hamming_booleanity();
        let eq_eval = Fr::from_u64(7);
        let challenges = vec![eq_eval, -eq_eval];

        let (desc, values) = claim.compile_descriptor::<Fr>(&challenges);
        assert!(matches!(desc.shape, KernelShape::HammingBooleanity));
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], eq_eval);

        // scale * (H² - H) must match direct evaluation
        let h = Fr::from_u64(5);
        let direct = claim.evaluate::<Fr>(&[h], &challenges);
        let via_compile = values[0] * (h * h - h);
        assert_eq!(direct, via_compile);
    }

    #[test]
    fn compile_registers_eq_product_weights() {
        let claim = crate::zkvm::claims::reductions::registers_claim_reduction();
        let eq_eval = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let gamma_sq = gamma * gamma;
        let challenges = vec![eq_eval, gamma, gamma_sq];

        let (desc, weights) = claim.compile_descriptor::<Fr>(&challenges);
        assert!(matches!(desc.shape, KernelShape::EqProduct));
        assert_eq!(weights.len(), 3);
        assert_eq!(weights[0], eq_eval);
        assert_eq!(weights[1], eq_eval * gamma);
        assert_eq!(weights[2], eq_eval * gamma_sq);

        // Σ w_i * o_i must match direct evaluation
        let openings = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_compile: Fr = weights
            .iter()
            .zip(openings.iter())
            .map(|(w, o)| *w * *o)
            .sum();
        assert_eq!(direct, via_compile);
    }

    #[test]
    fn compile_increment_eq_product_weights() {
        let claim = crate::zkvm::claims::reductions::increment_claim_reduction();
        let c0 = Fr::from_u64(11);
        let c1 = Fr::from_u64(13);
        let challenges = vec![c0, c1];

        let (desc, weights) = claim.compile_descriptor::<Fr>(&challenges);
        assert!(matches!(desc.shape, KernelShape::EqProduct));
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0], c0);
        assert_eq!(weights[1], c1);

        let openings = vec![Fr::from_u64(3), Fr::from_u64(7)];
        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_compile: Fr = weights
            .iter()
            .zip(openings.iter())
            .map(|(w, o)| *w * *o)
            .sum();
        assert_eq!(direct, via_compile);
    }

    #[test]
    fn compile_raf_evaluation_eq_product() {
        let claim = crate::zkvm::claims::ram::ram_raf_evaluation();
        let unmap = Fr::from_u64(13);
        let challenges = vec![unmap];

        let (desc, weights) = claim.compile_descriptor::<Fr>(&challenges);
        assert!(matches!(desc.shape, KernelShape::EqProduct));
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], unmap);
    }

    #[test]
    fn compile_ram_rw_custom_challenges() {
        let claim = crate::zkvm::claims::ram::ram_read_write_checking();
        let c0 = Fr::from_u64(13);
        let c1 = Fr::from_u64(17);
        let challenges = vec![c0, c1];

        let (desc, values) = claim.compile_descriptor::<Fr>(&challenges);
        assert!(matches!(desc.shape, KernelShape::Custom { .. }));
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], c0);
        assert_eq!(values[1], c1);
    }

    #[test]
    fn compile_ram_ra_virtual_custom() {
        let claim = crate::zkvm::claims::ram::ram_ra_virtual(4);
        let eq_eval = Fr::from_u64(7);
        let challenges = vec![eq_eval];

        let (desc, values) = claim.compile_descriptor::<Fr>(&challenges);
        assert!(matches!(desc.shape, KernelShape::Custom { .. }));
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], eq_eval);
    }

    #[test]
    fn compile_instruction_lookups_eq_product() {
        let claim = crate::zkvm::claims::reductions::instruction_lookups_claim_reduction();
        let challenges: Vec<Fr> = (10..=14).map(Fr::from_u64).collect();

        let (desc, weights) = claim.compile_descriptor::<Fr>(&challenges);
        assert!(matches!(desc.shape, KernelShape::EqProduct));
        assert_eq!(weights.len(), 5);

        // Each term: c_i * opening_i → weight[i] = c_i
        for (i, w) in weights.iter().enumerate() {
            assert_eq!(*w, challenges[i]);
        }

        let openings: Vec<Fr> = (1..=5).map(Fr::from_u64).collect();
        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_compile: Fr = weights
            .iter()
            .zip(openings.iter())
            .map(|(w, o)| *w * *o)
            .sum();
        assert_eq!(direct, via_compile);
    }

    #[test]
    fn compile_hamming_weight_reduction_eq_product() {
        let polynomials = vec![
            PolynomialId::InstructionRa(0),
            PolynomialId::BytecodeRa(0),
            PolynomialId::RamRa(0),
        ];
        let claim = crate::zkvm::claims::reductions::hamming_weight_claim_reduction(&polynomials);
        let challenges: Vec<Fr> = vec![Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(13)];

        let (desc, weights) = claim.compile_descriptor::<Fr>(&challenges);
        assert!(matches!(desc.shape, KernelShape::EqProduct));
        assert_eq!(weights.len(), 3);

        let openings: Vec<Fr> = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let direct = claim.evaluate::<Fr>(&openings, &challenges);
        let via_compile: Fr = weights
            .iter()
            .zip(openings.iter())
            .map(|(w, o)| *w * *o)
            .sum();
        assert_eq!(direct, via_compile);
    }
}
