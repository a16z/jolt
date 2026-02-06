//! Nova Folding Scheme for BlindFold
//!
//! Implements folding operations that combine two relaxed R1CS instance-witness
//! pairs into a single pair while preserving satisfiability.
//!
//! The key insight: if (u₁, w₁) and (u₂, w₂) both satisfy the relaxed R1CS,
//! then their fold (u', w') also satisfies it.

use crate::curve::{JoltCurve, JoltGroupElement};
use crate::field::JoltField;
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::transcripts::Transcript;
use rand_chacha::ChaCha20Rng;
use rand_core::{CryptoRngCore, SeedableRng};

use super::r1cs::VerifierR1CS;
use super::relaxed_r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};
use super::OutputClaimConstraint;

fn estimate_aux_var_count(constraint: &OutputClaimConstraint) -> usize {
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

/// Compute the cross-term T for folding two R1CS instances.
///
/// The cross-term captures the "interaction" between two instances when folded:
///   T = (AZ₁) ∘ (BZ₂) + (AZ₂) ∘ (BZ₁) - u₁*(CZ₂) - u₂*(CZ₁)
///
/// This is needed because when we fold Z' = Z₁ + r*Z₂:
///   (AZ') ∘ (BZ') = (AZ₁ + r*AZ₂) ∘ (BZ₁ + r*BZ₂)
///                 = (AZ₁∘BZ₁) + r*(AZ₁∘BZ₂ + AZ₂∘BZ₁) + r²*(AZ₂∘BZ₂)
///                       ↑                    ↑                  ↑
///                  from (u₁,w₁)         cross-term T        from (u₂,w₂)
pub fn compute_cross_term<F: JoltField>(
    r1cs: &VerifierR1CS<F>,
    z1: &[F],
    u1: F,
    z2: &[F],
    u2: F,
) -> Vec<F> {
    // Compute matrix-vector products
    let az1 = r1cs.a.mul_vector(z1);
    let bz1 = r1cs.b.mul_vector(z1);
    let cz1 = r1cs.c.mul_vector(z1);

    let az2 = r1cs.a.mul_vector(z2);
    let bz2 = r1cs.b.mul_vector(z2);
    let cz2 = r1cs.c.mul_vector(z2);

    // T[i] = (AZ₁)[i]*(BZ₂)[i] + (AZ₂)[i]*(BZ₁)[i] - u₁*(CZ₂)[i] - u₂*(CZ₁)[i]
    (0..r1cs.num_constraints)
        .map(|i| az1[i] * bz2[i] + az2[i] * bz1[i] - u1 * cz2[i] - u2 * cz1[i])
        .collect()
}

/// Sample a random satisfying relaxed R1CS instance-witness pair.
///
/// This is used in the BlindFold protocol to generate a random mask.
/// The key observation is that for relaxed R1CS, we can:
/// 1. Sample random witness W with proper structure (coefficients, intermediates, next_claims)
/// 2. Sample random scalar u
/// 3. Sample random public inputs x
/// 4. Compute E to make it satisfy: E = (AZ) ∘ (BZ) - u*(CZ)
/// 5. Extract round coefficients from W and commit to them
///
/// The resulting (instance, witness) pair is uniformly random but satisfying.
/// Critically, round_coefficients are extracted from W (not generated separately)
/// to ensure consistency after folding.
pub fn sample_random_satisfying_pair<F: JoltField, C: JoltCurve, R: CryptoRngCore>(
    gens: &PedersenGenerators<C>,
    r1cs: &VerifierR1CS<F>,
    eval_commitment_gens: Option<(C::G1, C::G1)>,
    rng: &mut R,
) -> (RelaxedR1CSInstance<F, C>, RelaxedR1CSWitness<F>, Vec<F>) {
    // Build W with proper structure: [coefficients, intermediates, next_claim] per round
    // This ensures round_coefficients extracted from W match the actual coefficients
    let mut W: Vec<F> = Vec::new();
    let mut round_coefficients: Vec<Vec<F>> = Vec::new();
    let mut round_blindings: Vec<F> = Vec::new();
    let mut round_commitments: Vec<C::G1> = Vec::new();
    let mut eval_commitments: Vec<C::G1> = Vec::new();

    // Track allocated openings to match R1CS's global_opening_vars behavior
    let mut allocated_openings: std::collections::HashSet<crate::poly::opening_proof::OpeningId> =
        std::collections::HashSet::new();

    for config in &r1cs.stage_configs {
        // Handle initial_input witness variables (if present) - BEFORE rounds
        // NOTE: This must match the R1CS builder's variable allocation order
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
                let num_aux = estimate_aux_var_count(constraint);

                for _ in 0..(num_new_openings + num_aux) {
                    W.push(F::random(rng));
                }
            }
        }

        for _ in 0..config.num_rounds {
            let num_coeffs = config.poly_degree + 1;
            let num_intermediates = config.poly_degree.saturating_sub(1);

            // Generate random coefficients
            let coeffs: Vec<F> = (0..num_coeffs).map(|_| F::random(rng)).collect();

            // Generate random intermediates
            let intermediates: Vec<F> = (0..num_intermediates).map(|_| F::random(rng)).collect();

            // Generate random next_claim
            let next_claim = F::random(rng);

            // Append to W in order: coefficients, intermediates, next_claim
            W.extend_from_slice(&coeffs);
            W.extend_from_slice(&intermediates);
            W.push(next_claim);

            // Commit to coefficients
            let blinding = F::random(rng);
            let commitment = gens.commit(&coeffs, &blinding);

            round_coefficients.push(coeffs);
            round_blindings.push(blinding);
            round_commitments.push(commitment);
        }

        // Handle final_output witness variables (if present)
        // NOTE: This must match the R1CS builder's variable allocation order
        if let Some(ref fo_config) = config.final_output {
            if let Some(ref constraint) = fo_config.constraint {
                // General constraint: opening_vars (witness) + aux_vars (witness)
                // Challenge vars are public inputs, not witness vars
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
                let num_aux = estimate_aux_var_count(constraint);

                for _ in 0..(num_new_openings + num_aux) {
                    W.push(F::random(rng));
                }
            } else {
                // Simple constraint: evaluation_vars + accumulator_vars
                let num_evals = fo_config.num_evaluations;
                for _ in 0..num_evals {
                    W.push(F::random(rng));
                }
                if num_evals > 1 {
                    for _ in 0..(num_evals - 1) {
                        W.push(F::random(rng));
                    }
                }
            }
        }
    }

    // Append extra constraints after all stages
    for constraint in &r1cs.extra_constraints {
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
        let num_aux = estimate_aux_var_count(constraint);

        for _ in 0..num_new_openings {
            W.push(F::random(rng));
        }

        let output_value = F::random(rng);
        W.push(output_value);

        for _ in 0..num_aux {
            W.push(F::random(rng));
        }

        let blinding = F::random(rng);
        W.push(blinding);

        let (g1_0, h1) = eval_commitment_gens.expect("Missing eval commitment generators");
        let commitment = g1_0.scalar_mul(&output_value) + h1.scalar_mul(&blinding);
        eval_commitments.push(commitment);
    }

    // Sample random public inputs
    let x: Vec<F> = (0..r1cs.num_public_inputs)
        .map(|_| F::random(rng))
        .collect();

    // Sample random scalar u (non-zero to avoid degenerate case)
    let u = loop {
        let candidate = F::random(rng);
        if !candidate.is_zero() {
            break candidate;
        }
    };

    // Build Z vector: [u, public_inputs..., witness...]
    let mut z = Vec::with_capacity(r1cs.num_vars);
    z.push(u);
    z.extend_from_slice(&x);
    z.extend_from_slice(&W);

    assert_eq!(
        z.len(),
        r1cs.num_vars,
        "sample_random_satisfying_pair: z.len()={} != r1cs.num_vars={}. x.len()={}, W.len()={}",
        z.len(),
        r1cs.num_vars,
        x.len(),
        W.len()
    );

    let expected_rounds: usize = r1cs.stage_configs.iter().map(|s| s.num_rounds).sum();
    assert_eq!(
        round_coefficients.len(),
        expected_rounds,
        "sample_random_satisfying_pair: round_coefficients.len()={} != expected_rounds={}",
        round_coefficients.len(),
        expected_rounds
    );

    // Compute E to make the instance satisfy: E = (AZ) ∘ (BZ) - u*(CZ)
    let az = r1cs.a.mul_vector(&z);
    let bz = r1cs.b.mul_vector(&z);
    let cz = r1cs.c.mul_vector(&z);

    let E: Vec<F> = (0..r1cs.num_constraints)
        .map(|i| az[i] * bz[i] - u * cz[i])
        .collect();

    // Sample blinding factors for E and W
    let r_E = F::random(rng);
    let r_W = F::random(rng);

    // Commit to E and W
    let E_bar = gens.commit(&E, &r_E);
    let W_bar = gens.commit(&W, &r_W);

    let instance = RelaxedR1CSInstance {
        E_bar,
        u,
        W_bar,
        x: x.clone(),
        round_commitments,
        eval_commitments,
    };

    let witness = RelaxedR1CSWitness {
        E,
        r_E,
        W,
        r_W,
        round_coefficients,
        round_blindings,
    };

    (instance, witness, z)
}

/// Derive a deterministic RNG seed from transcript state.
fn derive_rng_seed<T: Transcript>(transcript: &mut T) -> [u8; 32] {
    transcript.append_message(b"BlindFold_random_seed");
    let a = transcript.challenge_u128();
    let b = transcript.challenge_u128();
    let mut seed = [0u8; 32];
    seed[..16].copy_from_slice(&a.to_le_bytes());
    seed[16..].copy_from_slice(&b.to_le_bytes());
    seed
}

/// Sample random satisfying pair deterministically from transcript.
///
/// Same as `sample_random_satisfying_pair` but derives randomness from transcript
/// state, allowing verifier to reconstruct the same random instance.
pub fn sample_random_satisfying_pair_deterministic<F: JoltField, C: JoltCurve, T: Transcript>(
    gens: &PedersenGenerators<C>,
    r1cs: &VerifierR1CS<F>,
    eval_commitment_gens: Option<(C::G1, C::G1)>,
    transcript: &mut T,
) -> (RelaxedR1CSInstance<F, C>, RelaxedR1CSWitness<F>, Vec<F>) {
    let seed = derive_rng_seed(transcript);
    let mut rng = ChaCha20Rng::from_seed(seed);
    sample_random_satisfying_pair(gens, r1cs, eval_commitment_gens, &mut rng)
}

/// Sample only the random instance (no witness) deterministically from transcript.
///
/// This is used by the verifier to reconstruct the random instance without
/// needing it in the proof.
pub fn sample_random_instance_deterministic<F: JoltField, C: JoltCurve, T: Transcript>(
    gens: &PedersenGenerators<C>,
    r1cs: &VerifierR1CS<F>,
    eval_commitment_gens: Option<(C::G1, C::G1)>,
    transcript: &mut T,
) -> RelaxedR1CSInstance<F, C> {
    let (instance, _, _) =
        sample_random_satisfying_pair_deterministic(gens, r1cs, eval_commitment_gens, transcript);
    instance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::{Bn254Curve, JoltGroupElement};
    use crate::subprotocols::blindfold::r1cs::VerifierR1CSBuilder;
    use crate::subprotocols::blindfold::witness::{BlindFoldWitness, RoundWitness, StageWitness};
    use crate::subprotocols::blindfold::StageConfig;
    use ark_bn254::Fr;
    use ark_std::{One, UniformRand, Zero};
    use rand::thread_rng;

    #[test]
    fn test_cross_term_computation() {
        type F = Fr;

        // Create R1CS
        let configs = [StageConfig::new(1, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        // Create two valid witnesses
        // Witness 1
        let round1 = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let blindfold_witness1 =
            BlindFoldWitness::new(F::from_u64(100), vec![StageWitness::new(vec![round1])]);
        let z1 = blindfold_witness1.assign(&r1cs);

        // Witness 2
        let round2 = RoundWitness::new(
            vec![
                F::from_u64(50),
                F::from_u64(10),
                F::from_u64(20),
                F::from_u64(10),
            ],
            F::from_u64(5),
        );
        let blindfold_witness2 =
            BlindFoldWitness::new(F::from_u64(140), vec![StageWitness::new(vec![round2])]);
        let z2 = blindfold_witness2.assign(&r1cs);

        // Both should satisfy standard R1CS
        assert!(r1cs.is_satisfied(&z1));
        assert!(r1cs.is_satisfied(&z2));

        // Compute cross-term with u1=u2=1 (non-relaxed)
        let T = compute_cross_term(&r1cs, &z1, F::one(), &z2, F::one());

        // Cross-term should have correct length
        assert_eq!(T.len(), r1cs.num_constraints);

        // Verify the cross-term formula manually for constraint 0
        let az1 = r1cs.a.mul_vector(&z1);
        let bz1 = r1cs.b.mul_vector(&z1);
        let cz1 = r1cs.c.mul_vector(&z1);
        let az2 = r1cs.a.mul_vector(&z2);
        let bz2 = r1cs.b.mul_vector(&z2);
        let cz2 = r1cs.c.mul_vector(&z2);

        let expected_t0 = az1[0] * bz2[0] + az2[0] * bz1[0] - F::one() * cz2[0] - F::one() * cz1[0];
        assert_eq!(T[0], expected_t0);
    }

    #[test]
    fn test_random_satisfying_pair() {
        let mut rng = thread_rng();
        type F = Fr;

        // Create R1CS
        let configs = [StageConfig::new(1, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        // Need enough generators for witness
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.num_vars + 10);

        // Sample random satisfying pair
        let (instance, witness, z) = sample_random_satisfying_pair(&gens, &r1cs, None, &mut rng);

        // Instance should have non-zero u
        assert!(!instance.u.is_zero());

        // Witness should have correct sizes
        assert_eq!(witness.E.len(), r1cs.num_constraints);
        assert_eq!(witness.W.len(), r1cs.num_vars - 1 - r1cs.num_public_inputs);

        // Check relaxed R1CS satisfaction: (AZ) ∘ (BZ) = u*(CZ) + E
        let az = r1cs.a.mul_vector(&z);
        let bz = r1cs.b.mul_vector(&z);
        let cz = r1cs.c.mul_vector(&z);

        for i in 0..r1cs.num_constraints {
            let lhs = az[i] * bz[i];
            let rhs = instance.u * cz[i] + witness.E[i];
            assert_eq!(lhs, rhs, "Constraint {i} not satisfied");
        }

        // Verify commitment openings
        assert!(gens.verify(&instance.W_bar, &witness.W, &witness.r_W));
        assert!(gens.verify(&instance.E_bar, &witness.E, &witness.r_E));
    }

    #[test]
    fn test_folding_preserves_satisfaction() {
        let mut rng = thread_rng();
        type F = Fr;

        // Create R1CS
        let configs = [StageConfig::new(1, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.num_vars + 10);

        // Create two satisfying pairs
        // Pair 1: non-relaxed (u=1, E=0)
        let round1 = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let blindfold_witness1 =
            BlindFoldWitness::new(F::from_u64(100), vec![StageWitness::new(vec![round1])]);
        let z1 = blindfold_witness1.assign(&r1cs);

        let witness_start = 1 + r1cs.num_public_inputs;
        let w1_vec: Vec<F> = z1[witness_start..].to_vec();
        let w1 = RelaxedR1CSWitness {
            E: vec![F::zero(); r1cs.num_constraints],
            r_E: F::zero(),
            W: w1_vec,
            r_W: F::random(&mut rng),
            round_coefficients: Vec::new(),
            round_blindings: Vec::new(),
        };
        let u1 = F::one();

        // Pair 2: random satisfying pair
        let (inst2, w2, z2) = sample_random_satisfying_pair(&gens, &r1cs, None, &mut rng);
        let u2 = inst2.u;

        // Compute cross-term
        let T = compute_cross_term(&r1cs, &z1, u1, &z2, u2);

        // Choose random folding challenge
        let r = F::rand(&mut rng);
        let r_sq = r * r;

        // Compute folded values
        let z_folded: Vec<F> = z1.iter().zip(&z2).map(|(a, b)| *a + r * *b).collect();
        let u_folded = u1 + r * u2;
        let E_folded: Vec<F> =
            w1.E.iter()
                .zip(T.iter())
                .zip(&w2.E)
                .map(|((e1, t), e2)| *e1 + r * *t + r_sq * *e2)
                .collect();

        // Verify (AZ') ∘ (BZ') = u'*(CZ') + E'
        let az = r1cs.a.mul_vector(&z_folded);
        let bz = r1cs.b.mul_vector(&z_folded);
        let cz = r1cs.c.mul_vector(&z_folded);

        for i in 0..r1cs.num_constraints {
            let lhs = az[i] * bz[i];
            let rhs = u_folded * cz[i] + E_folded[i];
            assert_eq!(lhs, rhs, "Folding broke constraint {i}");
        }
    }

    #[test]
    fn test_folded_commitments_homomorphism() {
        let mut rng = thread_rng();
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.num_vars + 10);

        // Create two random satisfying pairs
        let (inst1, wit1, _) = sample_random_satisfying_pair(&gens, &r1cs, None, &mut rng);
        let (inst2, wit2, _) = sample_random_satisfying_pair(&gens, &r1cs, None, &mut rng);

        // Compute cross-term (using placeholder Z vectors for this test)
        let T: Vec<F> = (0..r1cs.num_constraints)
            .map(|_| F::rand(&mut rng))
            .collect();
        let r_T = F::rand(&mut rng);
        let T_bar = gens.commit(&T, &r_T);

        // Choose folding challenge
        let r = F::rand(&mut rng);
        let r_sq = r * r;

        // Fold instances
        let inst_folded = inst1.fold(&inst2, &T_bar, r);

        // Fold witnesses
        let wit_folded = wit1.fold(&wit2, &T, r_T, r);

        // Verify W commitment homomorphism
        let W_bar_expected = inst1.W_bar + inst2.W_bar.scalar_mul(&r);
        assert_eq!(inst_folded.W_bar, W_bar_expected);

        // Verify folded commitment opens correctly
        assert!(gens.verify(&inst_folded.W_bar, &wit_folded.W, &wit_folded.r_W));

        // Verify E commitment homomorphism
        let E_bar_expected = inst1.E_bar + T_bar.scalar_mul(&r) + inst2.E_bar.scalar_mul(&r_sq);
        assert_eq!(inst_folded.E_bar, E_bar_expected);

        // Verify folded E commitment opens correctly
        assert!(gens.verify(&inst_folded.E_bar, &wit_folded.E, &wit_folded.r_E));
    }
}
