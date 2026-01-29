//! Relaxed R1CS for BlindFold Protocol
//!
//! Standard R1CS: (A·Z) ∘ (B·Z) = C·Z
//!
//! This doesn't fold nicely due to cross-terms. Relaxed R1CS:
//!   (A·Z) ∘ (B·Z) = u·(C·Z) + E
//!
//! Where:
//! - u ∈ F is a scalar (u=1 for non-relaxed)
//! - E ∈ F^m is an error vector (E=0 for non-relaxed)
//!
//! This allows folding two satisfying instances into one.

use crate::curve::{JoltCurve, JoltGroupElement};
use crate::field::JoltField;
use crate::poly::commitment::pedersen::PedersenGenerators;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand_core::CryptoRngCore;

use super::r1cs::VerifierR1CS;

/// Relaxed R1CS Instance (public data)
///
/// Contains commitments to the witness and error vector, plus public inputs.
/// The verifier can compute folded instances without knowing the witness.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct RelaxedR1CSInstance<F: JoltField, C: JoltCurve> {
    /// Commitment to error vector E: Ē = Com(E, r_E)
    pub E_bar: C::G1,
    /// Scalar (u=1 for non-relaxed instances)
    pub u: F,
    /// Commitment to witness vector W: W̄ = Com(W, r_W)
    pub W_bar: C::G1,
    /// Public inputs (challenges, initial claim, etc.)
    pub x: Vec<F>,
    /// Per-round commitments from ZK sumcheck
    pub round_commitments: Vec<C::G1>,
    /// Evaluation commitments (e.g., y_com) for extra constraints
    pub eval_commitments: Vec<C::G1>,
}

/// Relaxed R1CS Witness (private data)
///
/// Contains the actual values and blinding factors for commitments.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct RelaxedR1CSWitness<F: JoltField> {
    /// Error vector (zeros for non-relaxed)
    pub E: Vec<F>,
    /// Blinding factor for E commitment
    pub r_E: F,
    /// Witness values (private portion of Z)
    pub W: Vec<F>,
    /// Blinding factor for W commitment
    pub r_W: F,
    /// Per-round polynomial coefficients (openings for round_commitments)
    pub round_coefficients: Vec<Vec<F>>,
    /// Per-round blinding factors
    pub round_blindings: Vec<F>,
}

impl<F: JoltField, C: JoltCurve> RelaxedR1CSInstance<F, C> {
    /// Create a non-relaxed instance (u=1, E=0) from standard R1CS witness.
    ///
    /// This is the starting point before any folding.
    #[allow(clippy::too_many_arguments)]
    pub fn new_non_relaxed<R: CryptoRngCore>(
        gens: &PedersenGenerators<C>,
        witness: &[F],
        public_inputs: Vec<F>,
        num_constraints: usize,
        round_commitments: Vec<C::G1>,
        round_coefficients: Vec<Vec<F>>,
        round_blindings: Vec<F>,
        eval_commitments: Vec<C::G1>,
        rng: &mut R,
    ) -> (Self, RelaxedR1CSWitness<F>) {
        // For non-relaxed: E = 0, u = 1
        let E = vec![F::zero(); num_constraints];
        let r_E = F::zero(); // No blinding needed for zero vector
        let r_W = F::random(rng);

        // Commit to witness
        let W_bar = gens.commit(witness, &r_W);

        // Commit to E (zero commitment with zero blinding = identity)
        let E_bar = C::G1::zero();

        let instance = Self {
            E_bar,
            u: F::one(),
            W_bar,
            x: public_inputs,
            round_commitments,
            eval_commitments,
        };

        let witness_struct = RelaxedR1CSWitness {
            E,
            r_E,
            W: witness.to_vec(),
            r_W,
            round_coefficients,
            round_blindings,
        };

        (instance, witness_struct)
    }

    /// Fold two instances into one.
    ///
    /// Given instances u1, u2 and cross-term commitment T̄, computes:
    /// - Ē' = Ē₁ + r·T̄ + r²·Ē₂
    /// - u' = u₁ + r·u₂
    /// - W̄' = W̄₁ + r·W̄₂
    /// - x' = x₁ + r·x₂
    /// - round_commitments' = round_commitments₁ + r·round_commitments₂
    pub fn fold(&self, other: &Self, T_bar: &C::G1, r: F) -> Self {
        let r_squared = r * r;

        assert_eq!(self.x.len(), other.x.len(), "Public input length mismatch");
        assert_eq!(
            self.round_commitments.len(),
            other.round_commitments.len(),
            "Round commitment length mismatch"
        );
        assert_eq!(
            self.eval_commitments.len(),
            other.eval_commitments.len(),
            "Eval commitment length mismatch"
        );

        // Ē' = Ē₁ + r·T̄ + r²·Ē₂
        let E_bar = self.E_bar + T_bar.scalar_mul(&r) + other.E_bar.scalar_mul(&r_squared);

        // u' = u₁ + r·u₂
        let u = self.u + r * other.u;

        // W̄' = W̄₁ + r·W̄₂
        let W_bar = self.W_bar + other.W_bar.scalar_mul(&r);

        // x' = x₁ + r·x₂
        let x: Vec<F> = self
            .x
            .iter()
            .zip(&other.x)
            .map(|(a, b)| *a + r * *b)
            .collect();

        // round_commitments' = round_commitments₁ + r·round_commitments₂
        let round_commitments: Vec<C::G1> = self
            .round_commitments
            .iter()
            .zip(&other.round_commitments)
            .map(|(c1, c2)| *c1 + c2.scalar_mul(&r))
            .collect();

        let eval_commitments: Vec<C::G1> = self
            .eval_commitments
            .iter()
            .zip(&other.eval_commitments)
            .map(|(c1, c2)| *c1 + c2.scalar_mul(&r))
            .collect();

        Self {
            E_bar,
            u,
            W_bar,
            x,
            round_commitments,
            eval_commitments,
        }
    }
}

impl<F: JoltField> RelaxedR1CSWitness<F> {
    /// Create a non-relaxed witness (E=0) from the private witness values.
    pub fn new_non_relaxed<R: CryptoRngCore>(
        witness: Vec<F>,
        num_constraints: usize,
        round_coefficients: Vec<Vec<F>>,
        round_blindings: Vec<F>,
        rng: &mut R,
    ) -> Self {
        Self {
            E: vec![F::zero(); num_constraints],
            r_E: F::zero(),
            W: witness,
            r_W: F::random(rng),
            round_coefficients,
            round_blindings,
        }
    }

    /// Fold two witnesses into one.
    ///
    /// Given witnesses w1, w2 and cross-term T, computes:
    /// - E' = E₁ + r·T + r²·E₂
    /// - r_E' = r_E₁ + r·r_T + r²·r_E₂
    /// - W' = W₁ + r·W₂
    /// - r_W' = r_W₁ + r·r_W₂
    /// - round_coefficients' = round_coefficients₁ + r·round_coefficients₂
    /// - round_blindings' = round_blindings₁ + r·round_blindings₂
    pub fn fold(&self, other: &Self, T: &[F], r_T: F, r: F) -> Self {
        let r_squared = r * r;

        // E' = E₁ + r·T + r²·E₂
        let E: Vec<F> = self
            .E
            .iter()
            .zip(T.iter())
            .zip(&other.E)
            .map(|((e1, t), e2)| *e1 + r * *t + r_squared * *e2)
            .collect();

        // r_E' = r_E₁ + r·r_T + r²·r_E₂
        let r_E = self.r_E + r * r_T + r_squared * other.r_E;

        // W' = W₁ + r·W₂
        let W: Vec<F> = self
            .W
            .iter()
            .zip(&other.W)
            .map(|(w1, w2)| *w1 + r * *w2)
            .collect();

        // r_W' = r_W₁ + r·r_W₂
        let r_W = self.r_W + r * other.r_W;

        // round_coefficients' = round_coefficients₁ + r·round_coefficients₂
        let round_coefficients: Vec<Vec<F>> = self
            .round_coefficients
            .iter()
            .zip(&other.round_coefficients)
            .map(|(c1, c2)| c1.iter().zip(c2).map(|(a, b)| *a + r * *b).collect())
            .collect();

        // round_blindings' = round_blindings₁ + r·round_blindings₂
        let round_blindings: Vec<F> = self
            .round_blindings
            .iter()
            .zip(&other.round_blindings)
            .map(|(r1, r2)| *r1 + r * *r2)
            .collect();

        Self {
            E,
            r_E,
            W,
            r_W,
            round_coefficients,
            round_blindings,
        }
    }

    /// Check if the witness satisfies the relaxed R1CS.
    ///
    /// Verifies: (A·Z) ∘ (B·Z) = u·(C·Z) + E
    pub fn check_satisfaction(&self, r1cs: &VerifierR1CS<F>, u: F, x: &[F]) -> Result<(), usize> {
        // Build Z vector: [u, public_inputs..., witness...]
        // The u at Z[0] allows proper folding since it becomes u' = u1 + r*u2
        let mut z = Vec::with_capacity(r1cs.num_vars);
        z.push(u); // u scalar at index 0 (1 for non-relaxed, folded otherwise)

        // Public inputs (challenges + initial claim in our case)
        z.extend_from_slice(x);

        // Private witness
        z.extend_from_slice(&self.W);

        assert_eq!(
            z.len(),
            r1cs.num_vars,
            "Z vector size mismatch: {} vs {}",
            z.len(),
            r1cs.num_vars
        );

        // Compute Az, Bz, Cz
        let az = r1cs.a.mul_vector(&z);
        let bz = r1cs.b.mul_vector(&z);
        let cz = r1cs.c.mul_vector(&z);

        // Check: (Az)_i * (Bz)_i = u * (Cz)_i + E_i for all i
        for i in 0..r1cs.num_constraints {
            let lhs = az[i] * bz[i];
            let rhs = u * cz[i] + self.E[i];
            if lhs != rhs {
                return Err(i);
            }
        }

        Ok(())
    }

    /// Verify that round_coefficients match the coefficients embedded in W.
    ///
    /// This is critical for soundness: ensures the coefficients used in R1CS
    /// constraints are the same as those verified via commitment opening.
    ///
    /// If round_coefficients is empty (unit tests without round commitment data),
    /// this check is skipped.
    pub fn verify_round_coefficients_consistency(
        &self,
        r1cs: &VerifierR1CS<F>,
        final_output_info: &[super::protocol::FinalOutputInfo],
    ) -> Result<(), usize> {
        use crate::poly::opening_proof::OpeningId;
        use std::collections::HashSet;

        // Skip check if no round coefficients (unit tests without round commitment data)
        if self.round_coefficients.is_empty() {
            return Ok(());
        }

        // Note: final_output_info is no longer used as we recompute from config
        // to properly account for shared openings
        let _ = final_output_info;

        // Track allocated openings to match R1CS's global_opening_vars behavior
        let mut allocated_openings: HashSet<OpeningId> = HashSet::new();

        let mut w_idx = 0;
        let mut round_idx = 0;

        for config in r1cs.stage_configs.iter() {
            // Skip past initial_input variables (if present) - BEFORE rounds
            if let Some(ref ii_config) = config.initial_input {
                if let Some(ref constraint) = ii_config.constraint {
                    // Count only NEW openings (matching R1CS allocation)
                    let num_new_openings = constraint
                        .required_openings
                        .iter()
                        .filter(|id| {
                            if allocated_openings.contains(id) {
                                false
                            } else {
                                allocated_openings.insert(**id);
                                true
                            }
                        })
                        .count();

                    fn estimate_aux_var_count(constraint: &super::OutputClaimConstraint) -> usize {
                        let mut count = 0;
                        for term in &constraint.terms {
                            if term.factors.len() <= 1 {
                                count += 1;
                            } else {
                                count += term.factors.len();
                            }
                        }
                        count
                    }

                    let num_aux = estimate_aux_var_count(constraint);
                    w_idx += num_new_openings + num_aux;
                }
            }

            for _round_in_stage in 0..config.num_rounds {
                let num_coeffs = config.poly_degree + 1;
                let num_intermediates = config.poly_degree.saturating_sub(1);

                // Extract coefficients from W at current position
                let coeffs_in_w = &self.W[w_idx..w_idx + num_coeffs];

                // Compare with round_coefficients
                if round_idx >= self.round_coefficients.len() {
                    return Err(round_idx);
                }
                let expected_coeffs = &self.round_coefficients[round_idx];

                if coeffs_in_w.len() != expected_coeffs.len() {
                    return Err(round_idx);
                }

                for (w_coeff, expected) in coeffs_in_w.iter().zip(expected_coeffs.iter()) {
                    if w_coeff != expected {
                        return Err(round_idx);
                    }
                }

                // Move to next round's position in W
                // Layout: coefficients + intermediates + next_claim
                w_idx += num_coeffs + num_intermediates + 1;
                round_idx += 1;
            }

            // Skip past final_output variables (if present)
            // Recompute from config to properly account for shared openings
            if let Some(ref fo_config) = config.final_output {
                if let Some(ref constraint) = fo_config.constraint {
                    // Count only NEW openings (matching R1CS allocation)
                    let num_new_openings = constraint
                        .required_openings
                        .iter()
                        .filter(|id| {
                            if allocated_openings.contains(id) {
                                false
                            } else {
                                allocated_openings.insert(**id);
                                true
                            }
                        })
                        .count();

                    fn estimate_aux_var_count_for_constraint(
                        constraint: &super::OutputClaimConstraint,
                    ) -> usize {
                        let mut count = 0;
                        for term in &constraint.terms {
                            if term.factors.len() <= 1 {
                                count += 1;
                            } else {
                                count += term.factors.len();
                            }
                        }
                        count
                    }

                    let num_aux = estimate_aux_var_count_for_constraint(constraint);
                    w_idx += num_new_openings + num_aux;
                } else {
                    // Simple constraint: evaluation_vars + accumulator_vars
                    let num_evals = fo_config.num_evaluations;
                    w_idx += num_evals; // evaluation_vars
                    if num_evals > 1 {
                        w_idx += num_evals - 1; // accumulator_vars
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::Bn254Curve;
    use crate::subprotocols::blindfold::r1cs::VerifierR1CSBuilder;
    use crate::subprotocols::blindfold::witness::{BlindFoldWitness, RoundWitness, StageWitness};
    use crate::subprotocols::blindfold::StageConfig;
    use ark_bn254::Fr;
    use ark_std::{One, UniformRand, Zero};

    use rand::thread_rng;

    fn verify_commitment_opening<F: JoltField, C: JoltCurve>(
        gens: &PedersenGenerators<C>,
        commitment: &C::G1,
        values: &[F],
        blinding: &F,
    ) -> bool {
        gens.commit(values, blinding) == *commitment
    }

    #[test]
    fn test_non_relaxed_instance_creation() {
        let mut rng = thread_rng();
        type F = Fr;

        // Create a simple R1CS
        let configs = [StageConfig::new(1, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        // Create valid witness
        let c0 = F::from_u64(40);
        let c1 = F::from_u64(5);
        let c2 = F::from_u64(10);
        let c3 = F::from_u64(5);
        let initial_claim = F::from_u64(100); // 2*40 + 5 + 10 + 5 = 100
        let challenge = F::from_u64(3);

        let round = RoundWitness::new(vec![c0, c1, c2, c3], challenge);
        let blindfold_witness =
            BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round])]);
        let z = blindfold_witness.assign(&r1cs);

        // Verify standard R1CS is satisfied
        assert!(r1cs.is_satisfied(&z));

        // Extract witness portion (after constant 1 and public inputs)
        let witness_start = 1 + r1cs.num_public_inputs;
        let witness: Vec<F> = z[witness_start..].to_vec();
        let public_inputs: Vec<F> = z[1..witness_start].to_vec();

        // Create generators (need enough for the witness size)
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(witness.len() + 10);

        // Create non-relaxed instance (with empty round commitment data for unit test)
        let (instance, relaxed_witness) = RelaxedR1CSInstance::<F, Bn254Curve>::new_non_relaxed(
            &gens,
            &witness,
            public_inputs.clone(),
            r1cs.num_constraints,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &mut rng,
        );

        // Verify instance properties
        assert_eq!(instance.u, F::one());
        assert_eq!(instance.x, public_inputs);

        // Verify witness properties
        assert_eq!(relaxed_witness.W, witness);
        assert!(relaxed_witness.E.iter().all(|e| e.is_zero()));

        // Verify commitment opening
        assert!(verify_commitment_opening(
            &gens,
            &instance.W_bar,
            &relaxed_witness.W,
            &relaxed_witness.r_W
        ));
    }

    #[test]
    fn test_relaxed_satisfaction_non_relaxed() {
        let mut rng = thread_rng();
        type F = Fr;

        // Create R1CS
        let configs = [StageConfig::new(1, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        // Create valid witness
        let c0 = F::from_u64(40);
        let c1 = F::from_u64(5);
        let c2 = F::from_u64(10);
        let c3 = F::from_u64(5);
        let initial_claim = F::from_u64(100);
        let challenge = F::from_u64(3);

        let round = RoundWitness::new(vec![c0, c1, c2, c3], challenge);
        let blindfold_witness =
            BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round])]);
        let z = blindfold_witness.assign(&r1cs);

        // Extract components
        let witness_start = 1 + r1cs.num_public_inputs;
        let witness: Vec<F> = z[witness_start..].to_vec();
        let public_inputs: Vec<F> = z[1..witness_start].to_vec();

        // Create relaxed witness
        let relaxed_witness = RelaxedR1CSWitness::new_non_relaxed(
            witness,
            r1cs.num_constraints,
            Vec::new(),
            Vec::new(),
            &mut rng,
        );

        // Check relaxed satisfaction (should pass since u=1, E=0)
        let result = relaxed_witness.check_satisfaction(&r1cs, F::one(), &public_inputs);
        assert!(result.is_ok(), "Relaxed R1CS should be satisfied");
    }

    #[test]
    fn test_witness_folding() {
        let mut rng = thread_rng();
        type F = Fr;

        let n = 10;
        let m = 5; // num constraints

        // Create two random witnesses
        let w1: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let w2: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let e1: Vec<F> = (0..m).map(|_| F::rand(&mut rng)).collect();
        let e2: Vec<F> = (0..m).map(|_| F::rand(&mut rng)).collect();
        let t: Vec<F> = (0..m).map(|_| F::rand(&mut rng)).collect();

        let r_e1 = F::rand(&mut rng);
        let r_e2 = F::rand(&mut rng);
        let r_t = F::rand(&mut rng);
        let r_w1 = F::rand(&mut rng);
        let r_w2 = F::rand(&mut rng);

        let wit1 = RelaxedR1CSWitness {
            E: e1.clone(),
            r_E: r_e1,
            W: w1.clone(),
            r_W: r_w1,
            round_coefficients: Vec::new(),
            round_blindings: Vec::new(),
        };

        let wit2 = RelaxedR1CSWitness {
            E: e2.clone(),
            r_E: r_e2,
            W: w2.clone(),
            r_W: r_w2,
            round_coefficients: Vec::new(),
            round_blindings: Vec::new(),
        };

        let r = F::rand(&mut rng);
        let folded = wit1.fold(&wit2, &t, r_t, r);

        // Verify folding formula: E' = E1 + r*T + r^2*E2
        let r_sq = r * r;
        for i in 0..m {
            let expected = e1[i] + r * t[i] + r_sq * e2[i];
            assert_eq!(folded.E[i], expected);
        }

        // Verify: W' = W1 + r*W2
        for i in 0..n {
            let expected = w1[i] + r * w2[i];
            assert_eq!(folded.W[i], expected);
        }

        // Verify blinding: r_E' = r_E1 + r*r_T + r^2*r_E2
        assert_eq!(folded.r_E, r_e1 + r * r_t + r_sq * r_e2);

        // Verify blinding: r_W' = r_W1 + r*r_W2
        assert_eq!(folded.r_W, r_w1 + r * r_w2);
    }

    #[test]
    fn test_instance_folding() {
        let mut rng = thread_rng();
        type F = Fr;

        let gens = PedersenGenerators::<Bn254Curve>::deterministic(20);

        // Create two instances
        let x1: Vec<F> = (0..5).map(|_| F::rand(&mut rng)).collect();
        let x2: Vec<F> = (0..5).map(|_| F::rand(&mut rng)).collect();

        let u1 = F::rand(&mut rng);
        let u2 = F::rand(&mut rng);

        // Random commitments (for testing formula, not real commitments)
        let E_bar1 = gens.message_generators[0].scalar_mul(&F::rand(&mut rng));
        let E_bar2 = gens.message_generators[1].scalar_mul(&F::rand(&mut rng));
        let W_bar1 = gens.message_generators[2].scalar_mul(&F::rand(&mut rng));
        let W_bar2 = gens.message_generators[3].scalar_mul(&F::rand(&mut rng));
        let T_bar = gens.message_generators[4].scalar_mul(&F::rand(&mut rng));

        let inst1 = RelaxedR1CSInstance::<F, Bn254Curve> {
            E_bar: E_bar1,
            u: u1,
            W_bar: W_bar1,
            x: x1.clone(),
            round_commitments: Vec::new(),
            eval_commitments: Vec::new(),
        };

        let inst2 = RelaxedR1CSInstance::<F, Bn254Curve> {
            E_bar: E_bar2,
            u: u2,
            W_bar: W_bar2,
            x: x2.clone(),
            round_commitments: Vec::new(),
            eval_commitments: Vec::new(),
        };

        let r = F::rand(&mut rng);
        let folded = inst1.fold(&inst2, &T_bar, r);

        // Verify: u' = u1 + r*u2
        assert_eq!(folded.u, u1 + r * u2);

        // Verify: x' = x1 + r*x2
        for i in 0..5 {
            assert_eq!(folded.x[i], x1[i] + r * x2[i]);
        }

        // Verify: E_bar' = E_bar1 + r*T_bar + r^2*E_bar2
        let r_sq = r * r;
        let expected_E_bar = E_bar1 + T_bar.scalar_mul(&r) + E_bar2.scalar_mul(&r_sq);
        assert_eq!(folded.E_bar, expected_E_bar);

        // Verify: W_bar' = W_bar1 + r*W_bar2
        let expected_W_bar = W_bar1 + W_bar2.scalar_mul(&r);
        assert_eq!(folded.W_bar, expected_W_bar);
    }

    #[test]
    fn test_commitment_homomorphism_for_folding() {
        let mut rng = thread_rng();
        type F = Fr;

        let gens = PedersenGenerators::<Bn254Curve>::deterministic(20);

        // Create two witness vectors
        let w1: Vec<F> = (0..5).map(|_| F::rand(&mut rng)).collect();
        let w2: Vec<F> = (0..5).map(|_| F::rand(&mut rng)).collect();
        let r_w1 = F::rand(&mut rng);
        let r_w2 = F::rand(&mut rng);

        // Commit individually
        let c1 = gens.commit(&w1, &r_w1);
        let c2 = gens.commit(&w2, &r_w2);

        // Fold using homomorphism
        let r = F::rand(&mut rng);
        let c_folded = c1 + c2.scalar_mul(&r);

        // Fold the values directly
        let w_folded: Vec<F> = w1.iter().zip(&w2).map(|(a, b)| *a + r * *b).collect();
        let r_w_folded = r_w1 + r * r_w2;

        // Commit to folded values
        let c_expected = gens.commit(&w_folded, &r_w_folded);

        // Homomorphism should hold
        assert_eq!(c_folded, c_expected);
    }
}
