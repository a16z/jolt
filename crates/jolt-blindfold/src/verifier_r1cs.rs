//! Verifier R1CS encoding deferred sumcheck verification checks.
//!
//! In committed mode, the sumcheck verifier does not check round polynomial
//! consistency directly. Instead, all checks are encoded as R1CS constraints
//! and proved via Spartan.
//!
//! For each sumcheck round with polynomial $P(X) = c_0 + c_1 X + \ldots + c_d X^d$:
//!
//! 1. **Sum check**: $P(0) + P(1) = 2c_0 + c_1 + \ldots + c_d = \text{running\_sum}$
//! 2. **Eval check**: $P(r_i) = c_0 + r_i c_1 + r_i^2 c_2 + \ldots = \text{next\_running\_sum}$
//!
//! All constraints are linear in the witness variables (coefficients and
//! evaluation outputs). The Fiat-Shamir challenges $r_i$ are baked into
//! the matrix coefficients.

use jolt_field::Field;
use jolt_spartan::SimpleR1CS;

/// Configuration for one sumcheck stage.
///
/// A stage is a sequence of sumcheck rounds with the same polynomial degree.
/// Each stage has an initial claimed sum; the running sum propagates across
/// rounds within the stage.
#[derive(Clone, Debug)]
pub struct StageConfig<F: Field> {
    /// Number of sumcheck rounds in this stage.
    pub num_rounds: usize,
    /// Degree of the round polynomial (e.g., 3 for Spartan outer sumcheck).
    pub degree: usize,
    /// Initial claimed sum for this stage (the first round's running sum).
    pub claimed_sum: F,
}

/// Fiat-Shamir challenges baked into the R1CS matrix coefficients.
///
/// The challenges are flattened across all stages in order:
/// stage 0 round 0, stage 0 round 1, ..., stage 1 round 0, ...
///
/// These values are known to both prover and verifier (derived from the
/// transcript) and are embedded into the A and C matrices as constants.
#[derive(Clone, Debug)]
pub struct BakedPublicInputs<F: Field> {
    /// All round challenges, flattened across stages.
    pub challenges: Vec<F>,
}

/// Builds a verifier R1CS encoding all deferred sumcheck checks.
///
/// Returns a [`SimpleR1CS`] with:
/// - `2 × total_rounds` constraints (sum check + eval per round)
/// - `1 + Σ num_rounds × (degree + 2)` variables
///
/// # Variable layout
///
/// - `Z[0] = 1` — constant variable for baked values
/// - Per round: `degree + 1` coefficient variables, then 1 evaluation variable
///
/// # Panics
///
/// Panics if `baked.challenges.len()` does not equal the total number of rounds.
pub fn build_verifier_r1cs<F: Field>(
    stages: &[StageConfig<F>],
    baked: &BakedPublicInputs<F>,
) -> SimpleR1CS<F> {
    let total_rounds: usize = stages.iter().map(|s| s.num_rounds).sum();
    assert_eq!(
        baked.challenges.len(),
        total_rounds,
        "challenge count must equal total rounds"
    );

    let num_constraints = 2 * total_rounds;
    let num_variables: usize = 1 + stages
        .iter()
        .map(|s| s.num_rounds * (s.degree + 2))
        .sum::<usize>();

    let mut a_entries = Vec::new();
    let mut b_entries = Vec::new();
    let mut c_entries = Vec::new();

    let mut var_counter = 1usize;
    let mut constraint_idx = 0usize;
    let mut challenge_idx = 0usize;

    let one = F::one();
    let two = F::from_u64(2);

    for stage in stages {
        // For the first round, the running sum is the stage's claimed_sum (a constant).
        // For subsequent rounds, it's the previous round's evaluation variable.
        let mut running_sum_var: Option<usize> = None;
        let mut running_sum_const: Option<F> = Some(stage.claimed_sum);

        for _round in 0..stage.num_rounds {
            let coeff_start = var_counter;
            var_counter += stage.degree + 1;
            let eval_var = var_counter;
            var_counter += 1;

            // A: 2·c₀ + c₁ + c₂ + ... + c_d
            a_entries.push((constraint_idx, coeff_start, two));
            for j in 1..=stage.degree {
                a_entries.push((constraint_idx, coeff_start + j, one));
            }
            // B: 1 (the constant variable Z[0])
            b_entries.push((constraint_idx, 0, one));
            // C: running_sum (constant or variable)
            if let Some(val) = running_sum_const {
                c_entries.push((constraint_idx, 0, val));
            } else if let Some(var) = running_sum_var {
                c_entries.push((constraint_idx, var, one));
            }
            constraint_idx += 1;

            // A: c₀ + r·c₁ + r²·c₂ + ... + r^d·c_d
            let r = baked.challenges[challenge_idx];
            let mut r_power = one;
            for j in 0..=stage.degree {
                a_entries.push((constraint_idx, coeff_start + j, r_power));
                r_power *= r;
            }
            // B: 1
            b_entries.push((constraint_idx, 0, one));
            // C: eval_var
            c_entries.push((constraint_idx, eval_var, one));
            constraint_idx += 1;

            // Next round's running sum is this round's evaluation
            running_sum_var = Some(eval_var);
            running_sum_const = None;
            challenge_idx += 1;
        }
    }

    SimpleR1CS::new(
        num_constraints,
        num_variables,
        a_entries,
        b_entries,
        c_entries,
    )
}

/// Assigns witness values from round polynomial coefficients.
///
/// # Arguments
///
/// * `stages` — Stage configurations (for structure: num_rounds, degree).
/// * `baked` — Baked challenges (for computing polynomial evaluations).
/// * `stage_coefficients` — `stage_coefficients[s][r]` contains the polynomial
///   coefficients `[c₀, c₁, ..., c_d]` for stage `s`, round `r`.
///
/// # Returns
///
/// Complete witness vector `Z` with `Z[0] = 1`, coefficient variables, and
/// computed evaluation variables.
///
/// # Panics
///
/// Panics if stage/round counts or coefficient lengths don't match the configs.
pub fn assign_witness<F: Field>(
    stages: &[StageConfig<F>],
    baked: &BakedPublicInputs<F>,
    stage_coefficients: &[Vec<Vec<F>>],
) -> Vec<F> {
    assert_eq!(stages.len(), stage_coefficients.len());

    let num_variables: usize = 1 + stages
        .iter()
        .map(|s| s.num_rounds * (s.degree + 2))
        .sum::<usize>();

    let mut witness = vec![F::zero(); num_variables];
    witness[0] = F::one();

    let mut var_idx = 1usize;
    let mut challenge_idx = 0usize;

    for (s, stage) in stages.iter().enumerate() {
        assert_eq!(stage_coefficients[s].len(), stage.num_rounds);

        for r in 0..stage.num_rounds {
            let coeffs = &stage_coefficients[s][r];
            assert_eq!(
                coeffs.len(),
                stage.degree + 1,
                "round {r} of stage {s}: expected {} coefficients, got {}",
                stage.degree + 1,
                coeffs.len()
            );

            // Assign coefficient variables
            for &c in coeffs {
                witness[var_idx] = c;
                var_idx += 1;
            }

            // Compute P(r_i) via Horner's method and assign eval variable
            let challenge = baked.challenges[challenge_idx];
            let eval = horner_evaluate(coeffs, challenge);
            witness[var_idx] = eval;
            var_idx += 1;

            challenge_idx += 1;
        }
    }

    witness
}

/// Evaluates a polynomial given as coefficient vector at a point via Horner's method.
///
/// For `coeffs = [c₀, c₁, ..., c_d]`, computes `c₀ + c₁·x + c₂·x² + ... + c_d·x^d`.
fn horner_evaluate<F: Field>(coeffs: &[F], x: F) -> F {
    let mut result = F::zero();
    for &c in coeffs.iter().rev() {
        result = result * x + c;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_spartan::R1CS;
    use num_traits::Zero;

    /// Checks that Az∘Bz = Cz for all constraints.
    fn check_satisfaction<F: Field>(r1cs: &SimpleR1CS<F>, witness: &[F]) {
        let (az, bz, cz) = r1cs.multiply_witness(witness);
        for i in 0..r1cs.num_constraints() {
            assert_eq!(
                az[i] * bz[i],
                cz[i],
                "constraint {i} not satisfied: Az*Bz={:?}, Cz={:?}",
                az[i] * bz[i],
                cz[i]
            );
        }
    }

    /// Checks that at least one constraint is violated.
    fn check_violation<F: Field>(r1cs: &SimpleR1CS<F>, witness: &[F]) {
        let (az, bz, cz) = r1cs.multiply_witness(witness);
        let violated = (0..r1cs.num_constraints()).any(|i| az[i] * bz[i] != cz[i]);
        assert!(violated, "expected at least one constraint violation");
    }

    #[test]
    fn single_stage_degree_1() {
        // Degree 1: P(X) = c₀ + c₁·X
        // Round 0: P(0)+P(1) = 2c₀+c₁ = claimed_sum, P(r₀) = c₀+r₀·c₁ = eval₀
        // Round 1: P(0)+P(1) = 2c₀+c₁ = eval₀,       P(r₁) = c₀+r₁·c₁ = eval₁

        let claimed_sum = Fr::from_u64(10);
        let r0 = Fr::from_u64(3);
        let r1 = Fr::from_u64(7);

        let stages = vec![StageConfig {
            num_rounds: 2,
            degree: 1,
            claimed_sum,
        }];
        let baked = BakedPublicInputs {
            challenges: vec![r0, r1],
        };

        let r1cs = build_verifier_r1cs(&stages, &baked);
        assert_eq!(r1cs.num_constraints(), 4);
        assert_eq!(r1cs.num_variables(), 1 + 2 * 3); // 1 + 2 rounds * (1+1 coeffs + 1 eval)

        // Round 0: c₀=3, c₁=4 → sum=2*3+4=10=claimed_sum ✓
        // eval₀ = 3 + 3*4 = 15
        // Round 1: c₀=5, c₁=5 → sum=2*5+5=15=eval₀ ✓
        // eval₁ = 5 + 7*5 = 40
        let coeffs = vec![
            vec![Fr::from_u64(3), Fr::from_u64(4)],
            vec![Fr::from_u64(5), Fr::from_u64(5)],
        ];
        let witness = assign_witness(&stages, &baked, &[coeffs]);
        check_satisfaction(&r1cs, &witness);
    }

    #[test]
    fn single_stage_degree_3() {
        // Degree 3: P(X) = c₀ + c₁X + c₂X² + c₃X³
        // Sum = 2c₀ + c₁ + c₂ + c₃ = claimed_sum
        let claimed_sum = Fr::from_u64(100);
        let r0 = Fr::from_u64(5);

        let stages = vec![StageConfig {
            num_rounds: 1,
            degree: 3,
            claimed_sum,
        }];
        let baked = BakedPublicInputs {
            challenges: vec![r0],
        };

        let r1cs = build_verifier_r1cs(&stages, &baked);
        assert_eq!(r1cs.num_constraints(), 2);
        assert_eq!(r1cs.num_variables(), 6); // 1 const + 4 coeffs + 1 eval

        // c₀=20, c₁=30, c₂=10, c₃=20 → sum = 2*20+30+10+20 = 100 ✓
        // eval = 20 + 30*5 + 10*25 + 20*125 = 20+150+250+2500 = 2920
        let coeffs = vec![vec![
            Fr::from_u64(20),
            Fr::from_u64(30),
            Fr::from_u64(10),
            Fr::from_u64(20),
        ]];
        let witness = assign_witness(&stages, &baked, &[coeffs]);
        check_satisfaction(&r1cs, &witness);
    }

    #[test]
    fn multi_stage_different_degrees() {
        // Stage 0: degree 1, 2 rounds
        // Stage 1: degree 2, 1 round
        let stages = vec![
            StageConfig {
                num_rounds: 2,
                degree: 1,
                claimed_sum: Fr::from_u64(10),
            },
            StageConfig {
                num_rounds: 1,
                degree: 2,
                claimed_sum: Fr::from_u64(50),
            },
        ];
        let baked = BakedPublicInputs {
            challenges: vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(4)],
        };

        let r1cs = build_verifier_r1cs(&stages, &baked);
        assert_eq!(r1cs.num_constraints(), 6); // 2*(2+1)

        // Stage 0, round 0: c₀=3, c₁=4 → sum=10 ✓, eval=3+2*4=11
        // Stage 0, round 1: c₀=4, c₁=3 → sum=11 ✓, eval=4+3*3=13
        // Stage 1, round 0: c₀=10, c₁=20, c₂=10 → sum=2*10+20+10=50 ✓, eval=10+4*20+16*10=250
        let stage0_coeffs = vec![
            vec![Fr::from_u64(3), Fr::from_u64(4)],
            vec![Fr::from_u64(4), Fr::from_u64(3)],
        ];
        let stage1_coeffs = vec![vec![Fr::from_u64(10), Fr::from_u64(20), Fr::from_u64(10)]];
        let witness = assign_witness(&stages, &baked, &[stage0_coeffs, stage1_coeffs]);
        check_satisfaction(&r1cs, &witness);
    }

    #[test]
    fn tampered_coefficient_rejected() {
        let stages = vec![StageConfig {
            num_rounds: 1,
            degree: 1,
            claimed_sum: Fr::from_u64(10),
        }];
        let baked = BakedPublicInputs {
            challenges: vec![Fr::from_u64(3)],
        };

        let r1cs = build_verifier_r1cs(&stages, &baked);

        // Correct: c₀=3, c₁=4 → sum=10 ✓
        let mut witness = assign_witness(
            &stages,
            &baked,
            &[vec![vec![Fr::from_u64(3), Fr::from_u64(4)]]],
        );
        check_satisfaction(&r1cs, &witness);

        // Tamper: change c₀ from 3 to 5, sum becomes 2*5+4=14≠10
        witness[1] = Fr::from_u64(5);
        check_violation(&r1cs, &witness);
    }

    #[test]
    fn wrong_challenge_causes_eval_violation() {
        let stages = vec![StageConfig {
            num_rounds: 1,
            degree: 1,
            claimed_sum: Fr::from_u64(10),
        }];

        // Build R1CS with challenge r=3
        let baked_r1cs = BakedPublicInputs {
            challenges: vec![Fr::from_u64(3)],
        };
        let r1cs = build_verifier_r1cs(&stages, &baked_r1cs);

        // Assign witness with different challenge r=5
        let baked_witness = BakedPublicInputs {
            challenges: vec![Fr::from_u64(5)],
        };
        let witness = assign_witness(
            &stages,
            &baked_witness,
            &[vec![vec![Fr::from_u64(3), Fr::from_u64(4)]]],
        );

        // Sum constraint is fine (doesn't involve challenge),
        // but eval constraint uses different r → violation
        check_violation(&r1cs, &witness);
    }

    #[test]
    fn chain_propagation_across_rounds() {
        // Verify that the running sum chains correctly:
        // round 0 eval feeds round 1 sum check, etc.
        let stages = vec![StageConfig {
            num_rounds: 3,
            degree: 1,
            claimed_sum: Fr::from_u64(20),
        }];
        let baked = BakedPublicInputs {
            challenges: vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(4)],
        };

        let r1cs = build_verifier_r1cs(&stages, &baked);
        assert_eq!(r1cs.num_constraints(), 6);

        // Round 0: c₀=5, c₁=10 → sum=20 ✓, eval=5+2*10=25
        // Round 1: c₀=10, c₁=5 → sum=25 ✓, eval=10+3*5=25
        // Round 2: c₀=10, c₁=5 → sum=25 ✓, eval=10+4*5=30
        let coeffs = vec![
            vec![Fr::from_u64(5), Fr::from_u64(10)],
            vec![Fr::from_u64(10), Fr::from_u64(5)],
            vec![Fr::from_u64(10), Fr::from_u64(5)],
        ];
        let witness = assign_witness(&stages, &baked, &[coeffs]);
        check_satisfaction(&r1cs, &witness);
    }

    #[test]
    fn horner_correctness() {
        // P(X) = 1 + 2X + 3X² + 4X³
        let coeffs = [
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ];
        let x = Fr::from_u64(5);
        // P(5) = 1 + 10 + 75 + 500 = 586
        assert_eq!(horner_evaluate(&coeffs, x), Fr::from_u64(586));
    }

    #[test]
    fn empty_stages() {
        let stages: Vec<StageConfig<Fr>> = vec![];
        let baked = BakedPublicInputs { challenges: vec![] };
        let r1cs = build_verifier_r1cs(&stages, &baked);
        assert_eq!(r1cs.num_constraints(), 0);
        assert_eq!(r1cs.num_variables(), 1);
    }

    #[test]
    fn integration_with_real_sumcheck() {
        use jolt_poly::{Polynomial, UnivariatePoly};
        use jolt_sumcheck::{
            ClearRoundHandler, RoundHandler, SumcheckClaim, SumcheckCompute, SumcheckProver,
        };
        use jolt_transcript::{Blake2bTranscript, Transcript};

        // Define a simple inner-product sumcheck: g(x) = a(x) * b(x)
        // a = [1, 2, 3, 4], b = [5, 6, 7, 8]
        // Sum = 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
        struct IpWitness {
            a: Polynomial<Fr>,
            b: Polynomial<Fr>,
        }

        impl SumcheckCompute<Fr> for IpWitness {
            fn round_polynomial(&self) -> UnivariatePoly<Fr> {
                let half = self.a.evaluations().len() / 2;
                let a = self.a.evaluations();
                let b = self.b.evaluations();
                let mut evals = [Fr::zero(); 3];
                for i in 0..half {
                    let a_lo = a[i];
                    let a_hi = a[i + half];
                    let b_lo = b[i];
                    let b_hi = b[i + half];
                    let a_delta = a_hi - a_lo;
                    let b_delta = b_hi - b_lo;
                    for (t, eval) in evals.iter_mut().enumerate() {
                        let x = Fr::from_u64(t as u64);
                        let a_val = a_lo + x * a_delta;
                        let b_val = b_lo + x * b_delta;
                        *eval += a_val * b_val;
                    }
                }
                let points: Vec<(Fr, Fr)> =
                    (0..3).map(|t| (Fr::from_u64(t as u64), evals[t])).collect();
                UnivariatePoly::interpolate(&points)
            }

            fn bind(&mut self, challenge: Fr) {
                self.a.bind(challenge);
                self.b.bind(challenge);
            }
        }

        // A handler that records challenges
        struct RecordingHandler {
            inner: ClearRoundHandler<Fr>,
            challenges: Vec<Fr>,
            round_polys: Vec<Vec<Fr>>,
        }

        impl RecordingHandler {
            fn new(cap: usize) -> Self {
                Self {
                    inner: ClearRoundHandler::with_capacity(cap),
                    challenges: Vec::with_capacity(cap),
                    round_polys: Vec::with_capacity(cap),
                }
            }
        }

        impl RoundHandler<Fr> for RecordingHandler {
            type Proof = (
                jolt_sumcheck::proof::SumcheckProof<Fr>,
                Vec<Fr>,
                Vec<Vec<Fr>>,
            );

            fn absorb_round_poly(
                &mut self,
                poly: &UnivariatePoly<Fr>,
                transcript: &mut impl Transcript,
            ) {
                self.round_polys.push(poly.coefficients().to_vec());
                self.inner.absorb_round_poly(poly, transcript);
            }

            fn on_challenge(&mut self, challenge: Fr) {
                self.challenges.push(challenge);
            }

            fn finalize(
                self,
            ) -> (
                jolt_sumcheck::proof::SumcheckProof<Fr>,
                Vec<Fr>,
                Vec<Vec<Fr>>,
            ) {
                (self.inner.finalize(), self.challenges, self.round_polys)
            }
        }

        let mut witness = IpWitness {
            a: Polynomial::new(vec![
                Fr::from_u64(1),
                Fr::from_u64(2),
                Fr::from_u64(3),
                Fr::from_u64(4),
            ]),
            b: Polynomial::new(vec![
                Fr::from_u64(5),
                Fr::from_u64(6),
                Fr::from_u64(7),
                Fr::from_u64(8),
            ]),
        };

        let claim = SumcheckClaim {
            num_vars: 2,
            degree: 2,
            claimed_sum: Fr::from_u64(70),
        };

        let mut transcript = Blake2bTranscript::new(b"test");
        let handler = RecordingHandler::new(2);
        let (_proof, challenges, round_polys) = SumcheckProver::prove_with_handler(
            &claim,
            &mut witness,
            &mut transcript,
            |c: u128| Fr::from_u128(c),
            handler,
        );

        // Now build verifier R1CS from the sumcheck stage
        let stages = vec![StageConfig {
            num_rounds: 2,
            degree: 2,
            claimed_sum: Fr::from_u64(70),
        }];
        let baked = BakedPublicInputs {
            challenges: challenges.clone(),
        };

        let r1cs = build_verifier_r1cs(&stages, &baked);
        let w = assign_witness(&stages, &baked, &[round_polys]);
        check_satisfaction(&r1cs, &w);
    }
}
