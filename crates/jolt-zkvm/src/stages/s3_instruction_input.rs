//! Stage 3b: Instruction input virtualization sumcheck.
//!
//! Proves that the instruction operand inputs are correctly composed from
//! flag-gated register values, immediates, and PC:
//!
//! $$\sum_j \text{eq}(r_{\text{cycle}}, j) \cdot \bigl(\text{Right}(j) + \gamma \cdot \text{Left}(j)\bigr) = \text{claim}$$
//!
//! where:
//! - $\text{Right}(j) = \text{is\_rs2}(j) \cdot \text{rs2\_v}(j) + \text{is\_imm}(j) \cdot \text{imm}(j)$
//! - $\text{Left}(j) = \text{is\_rs1}(j) \cdot \text{rs1\_v}(j) + \text{is\_pc}(j) \cdot \text{unexpanded\_pc}(j)$
//!
//! Degree 3 (eq × flag × value), `log_T` rounds, HighToLow binding.

use jolt_field::{Field, WithChallenge};
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::{EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_sumcheck::prover::SumcheckCompute;
use jolt_transcript::Transcript;

use crate::stage::{ProverStage, StageBatch};

/// Number of witness polynomials in the instruction input sumcheck.
pub const NUM_INSTR_INPUT_POLYS: usize = 8;

// Polynomial index constants.
const LEFT_IS_RS1: usize = 0;
const RS1_VALUE: usize = 1;
const LEFT_IS_PC: usize = 2;
const UNEXPANDED_PC: usize = 3;
const RIGHT_IS_RS2: usize = 4;
const RS2_VALUE: usize = 5;
const RIGHT_IS_IMM: usize = 6;
const IMM: usize = 7;

/// Instruction input virtualization prover stage.
///
/// Proves that instruction operands are correctly composed from flag-gated
/// register values, immediates, and program counter values.
pub struct InstructionInputStage<F: Field> {
    /// Full-size evaluation tables, preserved for [`extract_claims`].
    ///
    /// Order: [left_is_rs1, rs1_value, left_is_pc, unexpanded_pc,
    ///         right_is_rs2, rs2_value, right_is_imm, imm].
    witness_polys: Option<[Vec<F>; NUM_INSTR_INPUT_POLYS]>,
    /// Challenge point for eq evaluation (from prior stage, big-endian).
    r_cycle: Vec<F>,
    /// Batching coefficient γ.
    gamma: F,
    /// Expected sum.
    claimed_sum: F,
    num_vars: usize,
}

impl<F: Field> InstructionInputStage<F> {
    /// Creates a new instruction input stage.
    ///
    /// # Arguments
    ///
    /// * `witness_polys` — 8 evaluation tables, each of length `2^num_vars`.
    /// * `r_cycle` — Eq challenge point (big-endian).
    /// * `gamma` — Batching coefficient.
    /// * `claimed_sum` — Expected sum from prior claims.
    pub fn new(
        witness_polys: [Vec<F>; NUM_INSTR_INPUT_POLYS],
        r_cycle: Vec<F>,
        gamma: F,
        claimed_sum: F,
    ) -> Self {
        let num_vars = r_cycle.len();
        let expected_len = 1usize << num_vars;
        for (i, poly) in witness_polys.iter().enumerate() {
            assert_eq!(
                poly.len(),
                expected_len,
                "witness_polys[{i}].len() = {} != {expected_len}",
                poly.len(),
            );
        }

        Self {
            witness_polys: Some(witness_polys),
            r_cycle,
            gamma,
            claimed_sum,
            num_vars,
        }
    }
}

impl<F: WithChallenge, T: Transcript> ProverStage<F, T> for InstructionInputStage<F> {
    fn name(&self) -> &'static str {
        "S3_instruction_input"
    }

    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let polys = self
            .witness_polys
            .as_ref()
            .expect("build() called after extract_claims()");

        let eq_table = EqPolynomial::new(self.r_cycle.clone()).evaluations();
        let eq = Polynomial::new(eq_table);

        // Clone witness polys into mutable Polynomial wrappers for binding.
        let witness: [Polynomial<F>; NUM_INSTR_INPUT_POLYS] =
            core::array::from_fn(|i| Polynomial::new(polys[i].clone()));

        let evaluator = InstructionInputWitness {
            eq,
            polys: witness,
            gamma: self.gamma,
            claim: F::zero(),
        };

        StageBatch {
            claims: vec![SumcheckClaim {
                num_vars: self.num_vars,
                degree: 3,
                claimed_sum: self.claimed_sum,
            }],
            witnesses: vec![Box::new(evaluator)],
        }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let polys = self
            .witness_polys
            .take()
            .expect("extract_claims() called twice");

        // HighToLow binding: challenges are MSB-first, no reversal needed.
        let eval_point = challenges.to_vec();

        polys
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
        vec![]
    }
}

/// Sumcheck witness for the instruction input identity (degree 3).
///
/// Formula: `eq(r, x) · [right(x) + γ · left(x)]` where
/// `right = is_rs2 · rs2_v + is_imm · imm` and
/// `left = is_rs1 · rs1_v + is_pc · unexpanded_pc`.
///
/// Uses HighToLow binding via `Polynomial::bind`.
struct InstructionInputWitness<F: WithChallenge> {
    eq: Polynomial<F>,
    polys: [Polynomial<F>; NUM_INSTR_INPUT_POLYS],
    gamma: F,
    claim: F,
}

impl<F: WithChallenge> SumcheckCompute<F> for InstructionInputWitness<F> {
    fn set_claim(&mut self, claim: F) {
        self.claim = claim;
    }

    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let eq_evals = self.eq.evaluations();
        let half = eq_evals.len() / 2;
        let gamma = self.gamma;

        let w: [&[F]; NUM_INSTR_INPUT_POLYS] =
            core::array::from_fn(|i| self.polys[i].evaluations());

        let mut eval_0 = F::zero();
        let mut eval_2 = F::zero();
        let mut eval_3 = F::zero();

        // HighToLow: pair [j] (MSB=0) with [j + half] (MSB=1).
        for j in 0..half {
            let eq_lo = eq_evals[j];
            let eq_hi = eq_evals[j + half];
            let eq_d = eq_hi - eq_lo;

            // Load (lo, delta) for each witness polynomial.
            let l_rs1_lo = w[LEFT_IS_RS1][j];
            let l_rs1_d = w[LEFT_IS_RS1][j + half] - l_rs1_lo;
            let rs1_v_lo = w[RS1_VALUE][j];
            let rs1_v_d = w[RS1_VALUE][j + half] - rs1_v_lo;
            let l_pc_lo = w[LEFT_IS_PC][j];
            let l_pc_d = w[LEFT_IS_PC][j + half] - l_pc_lo;
            let upc_lo = w[UNEXPANDED_PC][j];
            let upc_d = w[UNEXPANDED_PC][j + half] - upc_lo;
            let r_rs2_lo = w[RIGHT_IS_RS2][j];
            let r_rs2_d = w[RIGHT_IS_RS2][j + half] - r_rs2_lo;
            let rs2_v_lo = w[RS2_VALUE][j];
            let rs2_v_d = w[RS2_VALUE][j + half] - rs2_v_lo;
            let r_imm_lo = w[RIGHT_IS_IMM][j];
            let r_imm_d = w[RIGHT_IS_IMM][j + half] - r_imm_lo;
            let imm_lo = w[IMM][j];
            let imm_d = w[IMM][j + half] - imm_lo;

            // t=0: all lo values.
            let right_0 = r_rs2_lo * rs2_v_lo + r_imm_lo * imm_lo;
            let left_0 = l_rs1_lo * rs1_v_lo + l_pc_lo * upc_lo;
            eval_0 += eq_lo * (right_0 + gamma * left_0);

            // t=2: p(2) = lo + 2·delta.
            let eq_2 = eq_lo + eq_d + eq_d;
            let l_rs1_2 = l_rs1_lo + l_rs1_d + l_rs1_d;
            let rs1_v_2 = rs1_v_lo + rs1_v_d + rs1_v_d;
            let l_pc_2 = l_pc_lo + l_pc_d + l_pc_d;
            let upc_2 = upc_lo + upc_d + upc_d;
            let r_rs2_2 = r_rs2_lo + r_rs2_d + r_rs2_d;
            let rs2_v_2 = rs2_v_lo + rs2_v_d + rs2_v_d;
            let r_imm_2 = r_imm_lo + r_imm_d + r_imm_d;
            let imm_2 = imm_lo + imm_d + imm_d;

            let right_2 = r_rs2_2 * rs2_v_2 + r_imm_2 * imm_2;
            let left_2 = l_rs1_2 * rs1_v_2 + l_pc_2 * upc_2;
            eval_2 += eq_2 * (right_2 + gamma * left_2);

            // t=3: p(3) = p(2) + delta.
            let eq_3 = eq_2 + eq_d;
            let r_rs2_3 = r_rs2_2 + r_rs2_d;
            let rs2_v_3 = rs2_v_2 + rs2_v_d;
            let r_imm_3 = r_imm_2 + r_imm_d;
            let imm_3 = imm_2 + imm_d;
            let l_rs1_3 = l_rs1_2 + l_rs1_d;
            let rs1_v_3 = rs1_v_2 + rs1_v_d;
            let l_pc_3 = l_pc_2 + l_pc_d;
            let upc_3 = upc_2 + upc_d;

            let right_3 = r_rs2_3 * rs2_v_3 + r_imm_3 * imm_3;
            let left_3 = l_rs1_3 * rs1_v_3 + l_pc_3 * upc_3;
            eval_3 += eq_3 * (right_3 + gamma * left_3);
        }

        // Degree 3: hint = s(0)+s(1), evals at {0, 2, 3}.
        UnivariatePoly::from_evals_and_hint(self.claim, &[eval_0, eval_2, eval_3])
    }

    fn bind(&mut self, challenge: F::Challenge) {
        let c: F = challenge.into();
        self.eq.bind(c);
        for poly in &mut self.polys {
            poly.bind(c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier};
    use jolt_transcript::Blake2bTranscript;
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// Brute-force computation of the instruction input claimed sum.
    fn brute_force_instr_input_sum(
        polys: &[Vec<Fr>; NUM_INSTR_INPUT_POLYS],
        r_cycle: &[Fr],
        gamma: Fr,
    ) -> Fr {
        let n = polys[0].len();
        let eq_table = EqPolynomial::new(r_cycle.to_vec()).evaluations();

        (0..n)
            .map(|j| {
                let right = polys[RIGHT_IS_RS2][j] * polys[RS2_VALUE][j]
                    + polys[RIGHT_IS_IMM][j] * polys[IMM][j];
                let left = polys[LEFT_IS_RS1][j] * polys[RS1_VALUE][j]
                    + polys[LEFT_IS_PC][j] * polys[UNEXPANDED_PC][j];
                eq_table[j] * (right + gamma * left)
            })
            .sum()
    }

    fn run_instr_input_test(num_vars: usize, seed: u64) {
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        let r_cycle: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let gamma = Fr::random(&mut rng);

        let witness_polys: [Vec<Fr>; NUM_INSTR_INPUT_POLYS] =
            core::array::from_fn(|_| (0..n).map(|_| Fr::random(&mut rng)).collect());

        let claimed_sum = brute_force_instr_input_sum(&witness_polys, &r_cycle, gamma);

        let polys_copy = witness_polys.clone();
        let mut stage = InstructionInputStage::new(witness_polys, r_cycle.clone(), gamma, claimed_sum);

        let mut pt = Blake2bTranscript::new(b"instr_input");
        let mut batch = stage.build(&[], &mut pt);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.claims[0].degree, 3);
        assert_eq!(batch.claims[0].num_vars, num_vars);

        let claims_snapshot = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
        );

        let mut vt = Blake2bTranscript::new(b"instr_input");
        let (final_eval, challenges) = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
        )
        .expect("verification should succeed");

        // Oracle check: final_eval = eq(r_cycle, r) · (right(r) + γ·left(r)).
        let eq_at_r = Polynomial::new(EqPolynomial::new(r_cycle).evaluations())
            .evaluate(&challenges);
        let right_at_r = Polynomial::new(polys_copy[RIGHT_IS_RS2].clone()).evaluate(&challenges)
            * Polynomial::new(polys_copy[RS2_VALUE].clone()).evaluate(&challenges)
            + Polynomial::new(polys_copy[RIGHT_IS_IMM].clone()).evaluate(&challenges)
                * Polynomial::new(polys_copy[IMM].clone()).evaluate(&challenges);
        let left_at_r = Polynomial::new(polys_copy[LEFT_IS_RS1].clone()).evaluate(&challenges)
            * Polynomial::new(polys_copy[RS1_VALUE].clone()).evaluate(&challenges)
            + Polynomial::new(polys_copy[LEFT_IS_PC].clone()).evaluate(&challenges)
                * Polynomial::new(polys_copy[UNEXPANDED_PC].clone()).evaluate(&challenges);
        let expected = eq_at_r * (right_at_r + gamma * left_at_r);
        assert_eq!(final_eval, expected, "oracle check failed");

        // Extract claims and verify evaluations.
        let claims =
            <InstructionInputStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
                &mut stage,
                &challenges,
                final_eval,
            );
        assert_eq!(claims.len(), NUM_INSTR_INPUT_POLYS);
        for (i, claim) in claims.iter().enumerate() {
            let expected_eval = Polynomial::new(polys_copy[i].clone()).evaluate(&challenges);
            assert_eq!(claim.eval, expected_eval, "claim {i} eval mismatch");
            assert_eq!(claim.point, challenges, "claim {i} point mismatch");
        }
    }

    #[test]
    fn instruction_input_even_vars() {
        run_instr_input_test(6, 42);
    }

    #[test]
    fn instruction_input_odd_vars() {
        run_instr_input_test(7, 123);
    }

    #[test]
    fn instruction_input_small() {
        run_instr_input_test(4, 777);
    }

    #[test]
    fn instruction_input_boolean_flags() {
        // Flags are boolean in real traces — verify correctness with {0,1} flags.
        let num_vars = 5;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(999);

        let r_cycle: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let gamma = Fr::random(&mut rng);

        let bool_poly = |rng: &mut ChaCha20Rng| -> Vec<Fr> {
            use rand_core::RngCore;
            (0..n)
                .map(|_| {
                    if rng.next_u32() & 1 == 1 {
                        Fr::one()
                    } else {
                        Fr::zero()
                    }
                })
                .collect()
        };
        let rand_poly =
            |rng: &mut ChaCha20Rng| -> Vec<Fr> { (0..n).map(|_| Fr::random(rng)).collect() };

        let witness_polys: [Vec<Fr>; NUM_INSTR_INPUT_POLYS] = [
            bool_poly(&mut rng), // left_is_rs1
            rand_poly(&mut rng), // rs1_value
            bool_poly(&mut rng), // left_is_pc
            rand_poly(&mut rng), // unexpanded_pc
            bool_poly(&mut rng), // right_is_rs2
            rand_poly(&mut rng), // rs2_value
            bool_poly(&mut rng), // right_is_imm
            rand_poly(&mut rng), // imm
        ];

        let claimed_sum = brute_force_instr_input_sum(&witness_polys, &r_cycle, gamma);

        let mut stage =
            InstructionInputStage::new(witness_polys, r_cycle.clone(), gamma, claimed_sum);

        let mut pt = Blake2bTranscript::new(b"bool_flags");
        let mut batch = stage.build(&[], &mut pt);

        let claims_snapshot = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(
            &batch.claims,
            &mut batch.witnesses,
            &mut pt,
        );

        let mut vt = Blake2bTranscript::new(b"bool_flags");
        let result = BatchedSumcheckVerifier::verify(
            &claims_snapshot,
            &proof,
            &mut vt,
        );
        assert!(result.is_ok(), "boolean-flag verification failed");
    }
}
