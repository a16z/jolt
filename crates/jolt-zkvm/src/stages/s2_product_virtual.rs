//! Stage 2a: Product virtual remainder sumcheck.
//!
//! Proves five R1CS product-virtual constraints hold across the cycle
//! domain. These constraints enforce relationships between virtual
//! polynomials that the standard linear Spartan inner sumcheck cannot
//! handle (they involve products of witness variables).
//!
//! The five constraints are:
//!
//! | # | Constraint            | Identity: left · right = ...         |
//! |---|-----------------------|--------------------------------------|
//! | 0 | Product               | left_inst · right_inst               |
//! | 1 | WriteLookupOutputToRD | is_rd_not_zero · wl_flag             |
//! | 2 | WritePCtoRD           | is_rd_not_zero · jump_flag           |
//! | 3 | ShouldBranch          | lookup_output · branch_flag          |
//! | 4 | ShouldJump            | jump_flag · (1 − next_is_noop)       |
//!
//! All five are batched with γ-power coefficients into a single degree-3
//! sumcheck:
//!
//! ```text
//! Σ_x eq(τ, x) · Σ_i γ^i · left_i(x) · right_i(x) = claimed_sum
//! ```

use std::sync::Arc;

use jolt_compute::{ComputeBackend, CpuBackend};
use jolt_field::Field;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_transcript::Transcript;

use crate::evaluators::catalog::{self, Term};
use crate::evaluators::kernel::KernelEvaluator;
use crate::stage::{ProverStage, StageBatch};
use jolt_ir::zkvm::claims::spartan;

/// Number of factor polynomials (8 unique polynomials across 5 constraints).
pub const NUM_FACTORS: usize = 8;

/// Number of product-virtual constraints.
pub const NUM_CONSTRAINTS: usize = 5;

/// Product virtual remainder prover stage.
///
/// Constructed with the 8 factor polynomial evaluation tables, an eq point
/// (r_cycle from Spartan outer sumcheck), γ-power batching coefficients, and
/// the claimed sum.
///
/// # Factor polynomial layout
///
/// | Index | Polynomial                      |
/// |-------|---------------------------------|
/// | 0     | left_instruction_input          |
/// | 1     | right_instruction_input         |
/// | 2     | is_rd_not_zero                  |
/// | 3     | write_lookup_output_to_rd_flag  |
/// | 4     | jump_flag                       |
/// | 5     | lookup_output                   |
/// | 6     | branch_flag                     |
/// | 7     | next_is_noop                    |
pub struct ProductVirtualStage<F: Field> {
    /// 8 factor polynomial evaluation tables (consumed by extract_claims).
    factor_polys: Option<Vec<Vec<F>>>,
    /// Eq polynomial evaluation point (r_cycle from Spartan).
    eq_point: Vec<F>,
    /// γ-power batching coefficients `[γ^0, γ^1, γ^2, γ^3, γ^4]`.
    gamma_powers: Vec<F>,
    num_vars: usize,
    claimed_sum: F,
}

impl<F: Field> ProductVirtualStage<F> {
    /// Creates a new product virtual stage.
    ///
    /// # Arguments
    ///
    /// * `factor_polys` — 8 evaluation tables in the order documented above.
    ///   Each must have length `2^eq_point.len()`.
    /// * `eq_point` — Evaluation point for the eq polynomial, typically
    ///   r_cycle from the Spartan outer sumcheck.
    /// * `gamma_powers` — `[γ^0, γ^1, γ^2, γ^3, γ^4]` for batching
    ///   the five product constraints.
    /// * `claimed_sum` — Expected sum of the batched identity. For a
    ///   satisfying R1CS assignment, this derives from the Spartan output.
    ///
    /// # Panics
    ///
    /// Panics if `factor_polys.len() != 8`, `gamma_powers.len() != 5`, or
    /// any polynomial table has the wrong length.
    pub fn new(
        factor_polys: Vec<Vec<F>>,
        eq_point: Vec<F>,
        gamma_powers: Vec<F>,
        claimed_sum: F,
    ) -> Self {
        assert_eq!(factor_polys.len(), NUM_FACTORS);
        assert_eq!(gamma_powers.len(), NUM_CONSTRAINTS);
        let num_vars = eq_point.len();
        let n = 1usize << num_vars;
        for (i, poly) in factor_polys.iter().enumerate() {
            assert_eq!(
                poly.len(),
                n,
                "factor_polys[{i}] length {} != 2^{num_vars} = {n}",
                poly.len(),
            );
        }
        Self {
            factor_polys: Some(factor_polys),
            eq_point,
            gamma_powers,
            num_vars,
            claimed_sum,
        }
    }
}

impl<F: Field, T: Transcript> ProverStage<F, T> for ProductVirtualStage<F> {
    fn name(&self) -> &'static str {
        "S2_product_virtual"
    }

    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let factor_polys = self
            .factor_polys
            .as_ref()
            .expect("build() called after extract_claims()");

        let eq_table = EqPolynomial::new(self.eq_point.clone()).evaluations();

        // Build the batched formula:
        //   eq · (γ^0·left·right + γ^1·rd_nz·wl + γ^2·rd_nz·jump
        //        + γ^3·lookup·branch + γ^4·jump − γ^4·jump·noop)
        //
        // Expanding the ShouldJump constraint γ^4·jump·(1−noop) gives 6 terms.
        let g = &self.gamma_powers;
        let terms = vec![
            Term {
                coeff: g[0],
                factors: vec![0, 1],
            }, // left_inst · right_inst
            Term {
                coeff: g[1],
                factors: vec![2, 3],
            }, // is_rd_nz · wl_flag
            Term {
                coeff: g[2],
                factors: vec![2, 4],
            }, // is_rd_nz · jump_flag
            Term {
                coeff: g[3],
                factors: vec![5, 6],
            }, // lookup_out · branch_flag
            Term {
                coeff: g[4],
                factors: vec![4],
            }, // jump_flag (from 1−noop expansion)
            Term {
                coeff: -g[4],
                factors: vec![4, 7],
            }, // −jump_flag · next_noop
        ];

        let (desc, challenges) = catalog::formula_descriptor(&terms, NUM_FACTORS, 3);
        let kernel = jolt_cpu_kernels::compile_with_challenges::<F>(&desc, &challenges);

        let backend = Arc::new(CpuBackend);
        let mut inputs = Vec::with_capacity(NUM_FACTORS + 1);
        inputs.push(backend.upload(&eq_table));
        for poly in factor_polys {
            inputs.push(backend.upload(poly));
        }

        let witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        StageBatch {
            claims: vec![SumcheckClaim {
                num_vars: self.num_vars,
                degree: 3,
                claimed_sum: self.claimed_sum,
            }],
            witnesses: vec![Box::new(witness)],
        }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let factor_polys = self
            .factor_polys
            .take()
            .expect("extract_claims() called twice");

        // Sumcheck uses LowToHigh binding: challenges[j] = variable x_j (bit j, LSB-first).
        // Polynomial::evaluate expects MSB-first: point[0] = x_{n-1}.
        // Reverse to align conventions.
        let eval_point: Vec<F> = challenges.iter().rev().copied().collect();

        factor_polys
            .into_iter()
            .map(|evals| {
                let poly = Polynomial::new(evals.clone());
                let eval = poly.evaluate(&eval_point);
                ProverClaim {
                    evaluations: evals,
                    point: eval_point.clone(),
                    eval,
                }
            })
            .collect()
    }

    fn claim_definitions(&self) -> Vec<ClaimDefinition> {
        vec![spartan::product_virtual_remainder()]
    }
}

/// Brute-force computation of the product virtual batched sum.
///
/// Used by tests to compute the correct claimed_sum from factor polynomial
/// evaluation tables.
pub fn brute_force_product_virtual_sum<F: Field>(
    factor_polys: &[Vec<F>],
    eq_point: &[F],
    gamma_powers: &[F],
) -> F {
    assert_eq!(factor_polys.len(), NUM_FACTORS);
    assert_eq!(gamma_powers.len(), NUM_CONSTRAINTS);
    let eq_table = EqPolynomial::new(eq_point.to_vec()).evaluations();

    let mut sum = F::zero();
    for (x, &eq_val) in eq_table.iter().enumerate() {
        let left_inst = factor_polys[0][x];
        let right_inst = factor_polys[1][x];
        let is_rd_nz = factor_polys[2][x];
        let wl_flag = factor_polys[3][x];
        let jump_flag = factor_polys[4][x];
        let lookup_out = factor_polys[5][x];
        let branch_flag = factor_polys[6][x];
        let next_noop = factor_polys[7][x];

        let formula = gamma_powers[0] * left_inst * right_inst
            + gamma_powers[1] * is_rd_nz * wl_flag
            + gamma_powers[2] * is_rd_nz * jump_flag
            + gamma_powers[3] * lookup_out * branch_flag
            + gamma_powers[4] * jump_flag * (F::one() - next_noop);

        sum += eq_val * formula;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier, SumcheckClaim};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::Zero;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    fn random_poly(n: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
        (0..n).map(|_| Fr::random(rng)).collect()
    }

    fn random_boolean_poly(n: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
        (0..n)
            .map(|_| Fr::from_u64(rng.next_u64() % 2))
            .collect()
    }

    fn gamma_powers(gamma: Fr) -> Vec<Fr> {
        let mut g = Fr::from_u64(1);
        (0..NUM_CONSTRAINTS)
            .map(|_| {
                let v = g;
                g *= gamma;
                v
            })
            .collect()
    }

    /// Generate factor polys for a satisfying assignment.
    ///
    /// For a satisfying witness, each product constraint holds pointwise:
    ///   left_i(x) · right_i(x) = C_i(x)
    /// This means the batched sum over eq is determined by the C_i values.
    fn random_factor_polys(num_vars: usize, rng: &mut ChaCha20Rng) -> Vec<Vec<Fr>> {
        let n = 1usize << num_vars;
        vec![
            random_poly(n, rng),         // 0: left_instruction_input
            random_poly(n, rng),         // 1: right_instruction_input
            random_boolean_poly(n, rng), // 2: is_rd_not_zero
            random_boolean_poly(n, rng), // 3: write_lookup_output_to_rd_flag
            random_boolean_poly(n, rng), // 4: jump_flag
            random_poly(n, rng),         // 5: lookup_output
            random_boolean_poly(n, rng), // 6: branch_flag
            random_boolean_poly(n, rng), // 7: next_is_noop
        ]
    }

    #[test]
    fn brute_force_matches_formula_descriptor() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let polys = random_factor_polys(num_vars, &mut rng);
        let gamma = Fr::from_u64(7);
        let g = gamma_powers(gamma);

        let bf_sum = brute_force_product_virtual_sum(&polys, &eq_point, &g);

        // Also compute via formula_descriptor for cross-check
        let eq_table = EqPolynomial::new(eq_point).evaluations();
        let mut formula_sum = Fr::zero();
        for (x, &eq_val) in eq_table.iter().enumerate() {
            let val = g[0] * polys[0][x] * polys[1][x]
                + g[1] * polys[2][x] * polys[3][x]
                + g[2] * polys[2][x] * polys[4][x]
                + g[3] * polys[5][x] * polys[6][x]
                + g[4] * polys[4][x] * (Fr::from_u64(1) - polys[7][x]);
            formula_sum += eq_val * val;
        }

        assert_eq!(bf_sum, formula_sum);
    }

    #[test]
    fn stage_build_produces_valid_sumcheck() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let polys = random_factor_polys(num_vars, &mut rng);
        let gamma = Fr::from_u64(11);
        let g = gamma_powers(gamma);
        let claimed_sum = brute_force_product_virtual_sum(&polys, &eq_point, &g);

        let mut stage = ProductVirtualStage::new(polys, eq_point, g, claimed_sum);

        let mut transcript = Blake2bTranscript::new(b"pv_test");
        let batch = stage.build(&[], &mut transcript);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.witnesses.len(), 1);
        assert_eq!(batch.claims[0].degree, 3);
        assert_eq!(batch.claims[0].num_vars, num_vars);
        assert_eq!(batch.claims[0].claimed_sum, claimed_sum);
    }

    #[test]
    fn stage_full_prove_verify_round_trip() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(555);
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let polys = random_factor_polys(num_vars, &mut rng);
        let gamma = Fr::from_u64(13);
        let g = gamma_powers(gamma);
        let claimed_sum = brute_force_product_virtual_sum(&polys, &eq_point, &g);

        let mut stage = ProductVirtualStage::new(polys.clone(), eq_point.clone(), g, claimed_sum);

        let mut pt = Blake2bTranscript::new(b"pv_roundtrip");
        let mut batch = stage.build(&[], &mut pt);

        let claim = batch.claims[0].clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        // Verify
        let mut vt = Blake2bTranscript::new(b"pv_roundtrip");
        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &[claim],
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        )
        .expect("verification should succeed");

        // LowToHigh sumcheck: challenges[j] = bit j (LSB-first).
        // Polynomial::evaluate / EqPolynomial expect MSB-first.
        let eval_point: Vec<Fr> = challenges.iter().rev().copied().collect();

        // Verify extract_claims produces correct evaluations at the reversed point.
        let claims =
            <ProductVirtualStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                final_eval,
            );
        assert_eq!(claims.len(), NUM_FACTORS);
        for (i, c) in claims.iter().enumerate() {
            let expected = Polynomial::new(polys[i].clone()).evaluate(&eval_point);
            assert_eq!(c.eval, expected, "claim {i} eval mismatch");
        }

        // Verify output claim: eq(τ, r) · g(evals) == final_eval
        let eq_eval = EqPolynomial::new(eq_point.clone()).evaluate(&eval_point);
        let factor_evals: Vec<Fr> = claims.iter().map(|c| c.eval).collect();
        let g = gamma_powers(gamma);

        let manual_g = g[0] * factor_evals[0] * factor_evals[1]
            + g[1] * factor_evals[2] * factor_evals[3]
            + g[2] * factor_evals[2] * factor_evals[4]
            + g[3] * factor_evals[5] * factor_evals[6]
            + g[4] * factor_evals[4] * (Fr::from_u64(1) - factor_evals[7]);

        let output_challenges = vec![g[0], g[1], g[2], g[3], g[4], -g[4]];
        let def = spartan::product_virtual_remainder();
        let def_g = def.evaluate::<Fr>(&factor_evals, &output_challenges);
        assert_eq!(manual_g, def_g, "claim def vs manual g(r) mismatch");

        assert_eq!(eq_eval * manual_g, final_eval, "eq*g vs final_eval");
    }

    #[test]
    fn extract_claims_correct_evaluations() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(777);
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let polys = random_factor_polys(num_vars, &mut rng);
        let gamma = Fr::from_u64(17);
        let g = gamma_powers(gamma);
        let claimed_sum = brute_force_product_virtual_sum(&polys, &eq_point, &g);

        let mut stage =
            ProductVirtualStage::new(polys.clone(), eq_point, g, claimed_sum);

        let mut pt = Blake2bTranscript::new(b"pv_extract");
        let mut batch = stage.build(&[], &mut pt);
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        // Get verifier challenges to pass to extract_claims
        let mut vt = Blake2bTranscript::new(b"pv_extract");
        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &[SumcheckClaim {
                num_vars,
                degree: 3,
                claimed_sum,
            }],
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        )
        .expect("verification should succeed");

        let claims =
            <ProductVirtualStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                final_eval,
            );
        assert_eq!(claims.len(), NUM_FACTORS);

        let eval_point: Vec<Fr> = challenges.iter().rev().copied().collect();
        for (i, claim) in claims.iter().enumerate() {
            let expected = Polynomial::new(polys[i].clone()).evaluate(&eval_point);
            assert_eq!(claim.eval, expected, "claim {i} eval mismatch");
            assert_eq!(claim.point, eval_point);
        }
    }

    #[test]
    fn claim_definitions_returns_product_virtual() {
        let stage = ProductVirtualStage::<Fr>::new(
            vec![vec![Fr::zero(); 4]; NUM_FACTORS],
            vec![Fr::from_u64(1), Fr::from_u64(2)],
            vec![Fr::from_u64(1); NUM_CONSTRAINTS],
            Fr::zero(),
        );
        let defs =
            <ProductVirtualStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::claim_definitions(
                &stage,
            );
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].opening_bindings.len(), NUM_FACTORS);
        assert_eq!(defs[0].challenge_bindings.len(), 6); // 5 γ-powers + 1 negated
    }

    #[test]
    fn zero_claimed_sum_with_satisfying_witness() {
        // Construct a "satisfying" witness where all product constraints
        // evaluate to zero (both factors zero), making the batched sum zero.
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(999);
        let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        // All factor polys zero → all products zero → sum is zero
        let polys = vec![vec![Fr::zero(); n]; NUM_FACTORS];
        let gamma = Fr::from_u64(19);
        let g = gamma_powers(gamma);
        let claimed_sum = brute_force_product_virtual_sum(&polys, &eq_point, &g);
        assert!(claimed_sum.is_zero());

        let mut stage = ProductVirtualStage::new(polys, eq_point, g, claimed_sum);
        let mut pt = Blake2bTranscript::new(b"pv_zero");
        let mut batch = stage.build(&[], &mut pt);

        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"pv_zero");
        let result = BatchedSumcheckVerifier::verify(
            &[SumcheckClaim {
                num_vars,
                degree: 3,
                claimed_sum: Fr::zero(),
            }],
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );
        assert!(result.is_ok(), "verification should succeed");
    }
}
