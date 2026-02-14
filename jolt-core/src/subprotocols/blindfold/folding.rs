//! Nova Folding Scheme for BlindFold
//!
//! Implements folding operations that combine two relaxed R1CS instance-witness
//! pairs into a single pair while preserving satisfiability.

use crate::curve::{JoltCurve, JoltGroupElement};
use crate::field::JoltField;
use crate::poly::commitment::pedersen::PedersenGenerators;
use rand_core::CryptoRngCore;

use super::r1cs::VerifierR1CS;
use super::relaxed_r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};

/// Compute the cross-term T for folding two R1CS instances.
///
///   T = (AZ₁) ∘ (BZ₂) + (AZ₂) ∘ (BZ₁) - u₁*(CZ₂) - u₂*(CZ₁)
pub fn compute_cross_term<F: JoltField>(
    r1cs: &VerifierR1CS<F>,
    z1: &[F],
    u1: F,
    z2: &[F],
    u2: F,
) -> Vec<F> {
    let az1 = r1cs.a.mul_vector(z1);
    let bz1 = r1cs.b.mul_vector(z1);
    let cz1 = r1cs.c.mul_vector(z1);

    let az2 = r1cs.a.mul_vector(z2);
    let bz2 = r1cs.b.mul_vector(z2);
    let cz2 = r1cs.c.mul_vector(z2);

    (0..r1cs.num_constraints)
        .map(|i| az1[i] * bz2[i] + az2[i] * bz1[i] - u1 * cz2[i] - u2 * cz1[i])
        .collect()
}

/// Commit rows of a flat vector interpreted as an R × C grid.
fn commit_rows<F: JoltField, C: JoltCurve>(
    gens: &PedersenGenerators<C>,
    flat: &[F],
    hyrax_C: usize,
    R: usize,
    row_blindings: &[F],
) -> Vec<C::G1> {
    debug_assert_eq!(row_blindings.len(), R);
    (0..R)
        .map(|i| {
            let start = i * hyrax_C;
            let end = (start + hyrax_C).min(flat.len());
            if start >= flat.len() {
                gens.blinding_generator.scalar_mul(&row_blindings[i])
            } else {
                gens.commit(&flat[start..end], &row_blindings[i])
            }
        })
        .collect()
}

/// Sample a random satisfying relaxed R1CS instance-witness pair.
///
/// W is arranged in grid layout: coefficient rows first (one per round, padded to C),
/// then non-coefficient values packed in subsequent rows.
pub fn sample_random_satisfying_pair<F: JoltField, C: JoltCurve, R: CryptoRngCore>(
    gens: &PedersenGenerators<C>,
    r1cs: &VerifierR1CS<F>,
    eval_commitment_gens: Option<(C::G1, C::G1)>,
    rng: &mut R,
) -> (RelaxedR1CSInstance<F, C>, RelaxedR1CSWitness<F>, Vec<F>) {
    let hyrax = &r1cs.hyrax;
    let hyrax_C = hyrax.C;
    let R_coeff = hyrax.R_coeff;
    let R_prime = hyrax.R_prime;
    let witness_len = R_prime * hyrax_C;

    let mut W = vec![F::zero(); witness_len];
    let mut w_row_blindings = vec![F::zero(); R_prime];
    let mut round_commitments: Vec<C::G1> = Vec::new();
    let mut eval_commitments: Vec<C::G1> = Vec::new();

    let mut allocated_openings: std::collections::HashSet<crate::poly::opening_proof::OpeningId> =
        std::collections::HashSet::new();

    // Track non-coeff write position (mirrors R1CS builder)
    let mut noncoeff_idx = R_coeff * hyrax_C;
    let mut round_idx = 0usize;

    for config in &r1cs.stage_configs {
        // Initial input variables (before rounds)
        if let Some(ref ii_config) = config.initial_input {
            if let Some(ref constraint) = ii_config.constraint {
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
                let num_aux = constraint.estimate_aux_var_count();
                for _ in 0..(num_new_openings + num_aux) {
                    W[noncoeff_idx] = F::random(rng);
                    noncoeff_idx += 1;
                }
            }
        }

        for _ in 0..config.num_rounds {
            let num_coeffs = config.poly_degree + 1;

            for k in 0..num_coeffs {
                W[round_idx * hyrax_C + k] = F::random(rng);
            }

            let blinding = F::random(rng);
            let row_start = round_idx * hyrax_C;
            let commitment = gens.commit(&W[row_start..row_start + hyrax_C], &blinding);
            w_row_blindings[round_idx] = blinding;
            round_commitments.push(commitment);

            // Only next_claim in non-coeff section (no Horner intermediates with baked challenges)
            W[noncoeff_idx] = F::random(rng);
            noncoeff_idx += 1;

            round_idx += 1;
        }

        // Final output variables
        if let Some(ref fout) = config.final_output {
            if let Some(ref constraint) = fout.constraint {
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
                let num_aux = constraint.estimate_aux_var_count();
                for _ in 0..(num_new_openings + num_aux) {
                    W[noncoeff_idx] = F::random(rng);
                    noncoeff_idx += 1;
                }
            } else {
                // Simple linear: only evaluation vars (batching coeffs baked, no accumulators)
                let num_evals = fout.num_evaluations;
                for _ in 0..num_evals {
                    W[noncoeff_idx] = F::random(rng);
                    noncoeff_idx += 1;
                }
            }
        }
    }

    // Extra constraints
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
        let num_aux = constraint.estimate_aux_var_count();

        for _ in 0..num_new_openings {
            W[noncoeff_idx] = F::random(rng);
            noncoeff_idx += 1;
        }

        let output_value = F::random(rng);
        W[noncoeff_idx] = output_value;
        noncoeff_idx += 1;

        for _ in 0..num_aux {
            W[noncoeff_idx] = F::random(rng);
            noncoeff_idx += 1;
        }

        let blinding = F::random(rng);
        W[noncoeff_idx] = blinding;
        noncoeff_idx += 1;

        let (g1_0, h1) = eval_commitment_gens.expect("Missing eval commitment generators");
        let commitment = g1_0.scalar_mul(&output_value) + h1.scalar_mul(&blinding);
        eval_commitments.push(commitment);
    }

    // Commit non-coeff rows
    let noncoeff_rows = hyrax.noncoeff_rows();
    let mut noncoeff_row_commitments = Vec::with_capacity(noncoeff_rows);
    for row in 0..noncoeff_rows {
        let start = R_coeff * hyrax_C + row * hyrax_C;
        let end = (start + hyrax_C).min(W.len());
        let row_blinding = F::random(rng);
        noncoeff_row_commitments.push(gens.commit(&W[start..end], &row_blinding));
        w_row_blindings[R_coeff + row] = row_blinding;
    }

    let u = loop {
        let candidate = F::random(rng);
        if !candidate.is_zero() {
            break candidate;
        }
    };

    // Z = [u, W...] (public inputs baked into R1CS coefficients)
    let mut z = Vec::with_capacity(r1cs.num_vars);
    z.push(u);
    z.extend_from_slice(&W);

    assert_eq!(
        z.len(),
        r1cs.num_vars,
        "z.len()={} != r1cs.num_vars={}. W.len()={}",
        z.len(),
        r1cs.num_vars,
        W.len()
    );

    // Compute E to satisfy: E = (AZ) ∘ (BZ) - u*(CZ)
    let az = r1cs.a.mul_vector(&z);
    let bz = r1cs.b.mul_vector(&z);
    let cz = r1cs.c.mul_vector(&z);

    let E: Vec<F> = (0..r1cs.num_constraints)
        .map(|i| az[i] * bz[i] - u * cz[i])
        .collect();

    // Commit E rows
    let (R_E, C_E) = hyrax.e_grid(r1cs.num_constraints);
    let e_row_blindings: Vec<F> = (0..R_E).map(|_| F::random(rng)).collect();
    let padded_E = {
        let mut e = E.clone();
        e.resize(R_E * C_E, F::zero());
        e
    };
    let e_row_commitments = commit_rows(gens, &padded_E, C_E, R_E, &e_row_blindings);

    let instance = RelaxedR1CSInstance {
        u,
        round_commitments,
        noncoeff_row_commitments,
        e_row_commitments,
        eval_commitments,
    };

    let witness = RelaxedR1CSWitness {
        E,
        W,
        w_row_blindings,
        e_row_blindings,
    };

    (instance, witness, z)
}

/// Compute T row commitments for Hyrax-style E opening.
///
/// T is the cross-term vector. We lay it out as rows × columns (same grid as E)
/// and commit each row.
pub fn commit_cross_term_rows<F: JoltField, C: JoltCurve>(
    gens: &PedersenGenerators<C>,
    T: &[F],
    R_E: usize,
    C_E: usize,
    rng: &mut impl CryptoRngCore,
) -> (Vec<C::G1>, Vec<F>) {
    let t_row_blindings: Vec<F> = (0..R_E).map(|_| F::random(rng)).collect();
    let padded_T = {
        let mut t = T.to_vec();
        t.resize(R_E * C_E, F::zero());
        t
    };
    let t_row_commitments = commit_rows(gens, &padded_T, C_E, R_E, &t_row_blindings);
    (t_row_commitments, t_row_blindings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::Bn254Curve;
    use crate::subprotocols::blindfold::r1cs::VerifierR1CSBuilder;
    use crate::subprotocols::blindfold::witness::{BlindFoldWitness, RoundWitness, StageWitness};
    use crate::subprotocols::blindfold::{BakedPublicInputs, StageConfig};
    use ark_bn254::Fr;
    use ark_std::{One, UniformRand, Zero};
    use rand::thread_rng;

    #[test]
    fn test_cross_term_computation() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
        let baked = BakedPublicInputs {
            challenges: vec![F::from_u64(3)],
            initial_claims: vec![F::from_u64(100)],
            ..Default::default()
        };
        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();

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

        assert!(r1cs.is_satisfied(&z1));

        let T = compute_cross_term(&r1cs, &z1, F::one(), &z2, F::one());
        assert_eq!(T.len(), r1cs.num_constraints);

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

        let configs = [StageConfig::new(1, 3)];
        let baked = BakedPublicInputs {
            challenges: vec![F::from_u64(3)],
            initial_claims: vec![F::from_u64(100)],
            ..Default::default()
        };
        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();

        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.hyrax.C + 1);

        let (instance, witness, z) = sample_random_satisfying_pair(&gens, &r1cs, None, &mut rng);

        assert!(!instance.u.is_zero());
        assert_eq!(witness.E.len(), r1cs.num_constraints);
        assert_eq!(witness.W.len(), r1cs.hyrax.R_prime * r1cs.hyrax.C);

        let az = r1cs.a.mul_vector(&z);
        let bz = r1cs.b.mul_vector(&z);
        let cz = r1cs.c.mul_vector(&z);

        for i in 0..r1cs.num_constraints {
            let lhs = az[i] * bz[i];
            let rhs = instance.u * cz[i] + witness.E[i];
            assert_eq!(lhs, rhs, "Constraint {i} not satisfied");
        }

        let hyrax_C = r1cs.hyrax.C;
        for (round, com) in instance.round_commitments.iter().enumerate() {
            let row_start = round * hyrax_C;
            let row_data = &witness.W[row_start..row_start + hyrax_C];
            assert!(gens.verify(com, row_data, &witness.w_row_blindings[round]));
        }

        let R_coeff = r1cs.hyrax.R_coeff;
        for (row, com) in instance.noncoeff_row_commitments.iter().enumerate() {
            let start = R_coeff * hyrax_C + row * hyrax_C;
            let end = (start + hyrax_C).min(witness.W.len());
            assert!(gens.verify(
                com,
                &witness.W[start..end],
                &witness.w_row_blindings[R_coeff + row]
            ));
        }
    }

    #[test]
    fn test_folding_preserves_satisfaction() {
        let mut rng = thread_rng();
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
        let baked = BakedPublicInputs {
            challenges: vec![F::from_u64(3)],
            initial_claims: vec![F::from_u64(100)],
            ..Default::default()
        };
        let builder = VerifierR1CSBuilder::<F>::new(&configs, &baked);
        let r1cs = builder.build();

        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.hyrax.C + 1);

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

        let w1_vec: Vec<F> = z1[1..].to_vec();
        let R_prime = r1cs.hyrax.R_prime;
        let w1 = RelaxedR1CSWitness {
            E: vec![F::zero(); r1cs.num_constraints],
            W: w1_vec,
            w_row_blindings: vec![F::zero(); R_prime],
            e_row_blindings: Vec::new(),
        };
        let u1 = F::one();

        let (_inst2, w2, z2) = sample_random_satisfying_pair(&gens, &r1cs, None, &mut rng);
        let u2 = _inst2.u;

        let T = compute_cross_term(&r1cs, &z1, u1, &z2, u2);

        let r = F::rand(&mut rng);
        let r_sq = r * r;

        let z_folded: Vec<F> = z1.iter().zip(&z2).map(|(a, b)| *a + r * *b).collect();
        let u_folded = u1 + r * u2;
        let E_folded: Vec<F> =
            w1.E.iter()
                .zip(T.iter())
                .zip(&w2.E)
                .map(|((e1, t), e2)| *e1 + r * *t + r_sq * *e2)
                .collect();

        let az = r1cs.a.mul_vector(&z_folded);
        let bz = r1cs.b.mul_vector(&z_folded);
        let cz = r1cs.c.mul_vector(&z_folded);

        for i in 0..r1cs.num_constraints {
            let lhs = az[i] * bz[i];
            let rhs = u_folded * cz[i] + E_folded[i];
            assert_eq!(lhs, rhs, "Folding broke constraint {i}");
        }
    }
}
