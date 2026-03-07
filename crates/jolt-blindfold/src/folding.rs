//! Nova folding for relaxed R1CS instances.
//!
//! Implements the IVC folding scheme from Nova (Kothapalli–Setty–Parno, 2021)
//! in a commitment-agnostic way. All witness-level operations are **pure field
//! arithmetic** — no commitment types, no group structure assumed.
//!
//! The public instance type [`RelaxedInstance`] is generic over a plain `C`
//! parameter with no trait bounds, and [`fold_instances`] takes a caller-provided
//! closure for commitment linear combination.  This lets the same code work with
//! Pedersen, lattice-based, or any other additively homomorphic commitment scheme.
//!
//! # Core operations
//!
//! | Function | Commitment-free? | Purpose |
//! |----------|:----------------:|---------|
//! | [`compute_cross_term`] | yes | $T = Az_1 \circ Bz_2 + Az_2 \circ Bz_1 - u_1 Cz_2 - u_2 Cz_1$ |
//! | [`fold_witnesses`] | yes | $w' = w_1 + r w_2$, $E' = E_1 + r T + r^2 E_2$ |
//! | [`fold_scalar`] | yes | $u' = u_1 + r u_2$ |
//! | [`sample_random_witness`] | yes | Random satisfying relaxed instance |
//! | [`check_relaxed_satisfaction`] | yes | Verifies $Az \circ Bz = u Cz + E$ |
//! | [`fold_instances`] | closure | Folds commitments via caller-provided `linear_combine` |

use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_spartan::R1CS;

/// Private witness for a relaxed R1CS instance.
///
/// Contains the full witness vector (used as input to
/// [`R1CS::multiply_witness`]) and the error vector.
/// This is pure field data — no commitment types.
pub struct RelaxedWitness<F: Field> {
    /// Full witness vector (including the leading constant `1`).
    pub w: Vec<F>,
    /// Error vector, length = number of constraints.
    pub e: Vec<F>,
}

/// Public instance for a relaxed R1CS.
///
/// Generic over `C` with **no trait bounds** — the commitment type is opaque
/// to this module. Additive homomorphism is assumed only at the call site of
/// [`fold_instances`], where the caller supplies the `linear_combine` closure.
pub struct RelaxedInstance<F, C> {
    /// Relaxation scalar ($u = 1$ for standard R1CS).
    pub u: F,
    /// Commitment to the witness vector.
    pub w_commitment: C,
    /// Commitment to the error vector.
    pub e_commitment: C,
}

/// Computes the cross-term $T$ from two relaxed instance witnesses.
///
/// $$T_i = (Az_1)_i (Bz_2)_i + (Az_2)_i (Bz_1)_i - u_1 (Cz_2)_i - u_2 (Cz_1)_i$$
///
/// The cross-term captures the "interaction" between two instances during folding.
/// When added to the error vectors via [`fold_witnesses`], it ensures the folded
/// instance satisfies the relaxed R1CS equation.
pub fn compute_cross_term<F: Field>(
    r1cs: &impl R1CS<F>,
    z1: &[F],
    u1: F,
    z2: &[F],
    u2: F,
) -> Vec<F> {
    let (az1, bz1, cz1) = r1cs.multiply_witness(z1);
    let (az2, bz2, cz2) = r1cs.multiply_witness(z2);
    let m = r1cs.num_constraints();

    let mut t = Vec::with_capacity(m);
    for i in 0..m {
        t.push(az1[i] * bz2[i] + az2[i] * bz1[i] - u1 * cz2[i] - u2 * cz1[i]);
    }
    t
}

/// Folds two relaxed witnesses into one.
///
/// $$w'_i = w_{1,i} + r \cdot w_{2,i}$$
/// $$E'_i = E_{1,i} + r \cdot T_i + r^2 \cdot E_{2,i}$$
pub fn fold_witnesses<F: Field>(
    wit1: &RelaxedWitness<F>,
    wit2: &RelaxedWitness<F>,
    cross_term: &[F],
    challenge: F,
) -> RelaxedWitness<F> {
    let r2 = challenge * challenge;

    let w: Vec<F> = wit1
        .w
        .iter()
        .zip(wit2.w.iter())
        .map(|(&a, &b)| a + challenge * b)
        .collect();

    let e: Vec<F> = wit1
        .e
        .iter()
        .zip(cross_term.iter())
        .zip(wit2.e.iter())
        .map(|((&e1, &t), &e2)| e1 + challenge * t + r2 * e2)
        .collect();

    RelaxedWitness { w, e }
}

/// Folds two relaxation scalars: $u' = u_1 + r \cdot u_2$.
#[inline]
pub fn fold_scalar<F: Field>(u1: F, u2: F, challenge: F) -> F {
    u1 + challenge * u2
}

/// Folds two public instances.
///
/// Commitment linear combination (`c1 + scalar * c2`) is provided by the
/// [`HomomorphicCommitment`] trait bound on `C`. This works with Pedersen
/// (via the blanket [`JoltGroup`](jolt_crypto::JoltGroup) impl), lattice-based,
/// or any other additively homomorphic commitment type.
pub fn fold_instances<F: Field, C: HomomorphicCommitment<F>>(
    inst1: &RelaxedInstance<F, C>,
    inst2: &RelaxedInstance<F, C>,
    t_commitment: &C,
    challenge: F,
) -> RelaxedInstance<F, C> {
    let u = fold_scalar(inst1.u, inst2.u, challenge);
    let w_commitment = C::linear_combine(&inst1.w_commitment, &inst2.w_commitment, &challenge);

    // E' = E1 + r*T + r²*E2
    let r2 = challenge * challenge;
    let intermediate = C::linear_combine(&inst1.e_commitment, t_commitment, &challenge);
    let e_commitment = C::linear_combine(&intermediate, &inst2.e_commitment, &r2);

    RelaxedInstance {
        u,
        w_commitment,
        e_commitment,
    }
}

/// Samples a random witness that satisfies the relaxed R1CS equation.
///
/// Picks a random non-zero $u$, random witness entries, then computes $E$
/// as $E_i = (Az)_i (Bz)_i - u (Cz)_i$ so that the relaxed equation holds
/// by construction.
///
/// Used to generate the "masking" instance in Nova folding — the random
/// instance acts as a one-time pad to hide the real witness.
pub fn sample_random_witness<F: Field>(
    r1cs: &impl R1CS<F>,
    rng: &mut impl rand_core::RngCore,
) -> (F, RelaxedWitness<F>) {
    // Random non-zero u
    let u = loop {
        let candidate = F::from_u64(rng.next_u64());
        if !candidate.is_zero() {
            break candidate;
        }
    };

    // Random witness vector
    let n = r1cs.num_variables();
    let mut w = Vec::with_capacity(n);
    // w[0] = 1 (constant term)
    w.push(F::one());
    for _ in 1..n {
        w.push(F::from_u64(rng.next_u64()));
    }

    // Compute E so that Az∘Bz = u·Cz + E holds
    let (az, bz, cz) = r1cs.multiply_witness(&w);
    let m = r1cs.num_constraints();
    let mut e = Vec::with_capacity(m);
    for i in 0..m {
        e.push(az[i] * bz[i] - u * cz[i]);
    }

    (u, RelaxedWitness { w, e })
}

/// Checks that a witness satisfies the relaxed R1CS equation.
///
/// $$\forall i: \; (Az)_i \cdot (Bz)_i = u \cdot (Cz)_i + E_i$$
///
/// Returns `Ok(())` on success, or `Err(i)` where `i` is the index of the
/// first violated constraint.
pub fn check_relaxed_satisfaction<F: Field>(
    r1cs: &impl R1CS<F>,
    u: F,
    witness: &RelaxedWitness<F>,
) -> Result<(), usize> {
    let (az, bz, cz) = r1cs.multiply_witness(&witness.w);
    let m = r1cs.num_constraints();
    for i in 0..m {
        if az[i] * bz[i] != u * cz[i] + witness.e[i] {
            return Err(i);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_spartan::SimpleR1CS;
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// x * x = y  with witness z = [1, x, y]
    fn x_squared_r1cs() -> SimpleR1CS<Fr> {
        SimpleR1CS::new(
            1,
            3,
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 2, Fr::from_u64(1))],
        )
    }

    /// Two-constraint system:
    ///   constraint 0: x * x = y
    ///   constraint 1: y * 1 = y
    fn two_constraint_r1cs() -> SimpleR1CS<Fr> {
        SimpleR1CS::new(
            2,
            3,
            vec![(0, 1, Fr::from_u64(1)), (1, 2, Fr::from_u64(1))],
            vec![(0, 1, Fr::from_u64(1)), (1, 0, Fr::from_u64(1))],
            vec![(0, 2, Fr::from_u64(1)), (1, 2, Fr::from_u64(1))],
        )
    }

    #[test]
    fn cross_term_correctness() {
        let r1cs = x_squared_r1cs();

        let z1 = vec![Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];
        let u1 = Fr::one();
        let z2 = vec![Fr::from_u64(1), Fr::from_u64(5), Fr::from_u64(25)];
        let u2 = Fr::one();

        let t = compute_cross_term(&r1cs, &z1, u1, &z2, u2);
        assert_eq!(t.len(), 1);

        // T[0] = az1*bz2 + az2*bz1 - u1*cz2 - u2*cz1 = 3*5 + 5*3 - 25 - 9 = -4
        let (az1, bz1, cz1) = r1cs.multiply_witness(&z1);
        let (az2, bz2, cz2) = r1cs.multiply_witness(&z2);
        let expected = az1[0] * bz2[0] + az2[0] * bz1[0] - u1 * cz2[0] - u2 * cz1[0];
        assert_eq!(t[0], expected);
    }

    #[test]
    fn witness_folding_formula() {
        let w1 = RelaxedWitness {
            w: vec![Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)],
            e: vec![Fr::from_u64(0)],
        };
        let w2 = RelaxedWitness {
            w: vec![Fr::from_u64(1), Fr::from_u64(5), Fr::from_u64(25)],
            e: vec![Fr::from_u64(7)],
        };
        let t = vec![Fr::from_u64(11)];
        let r = Fr::from_u64(2);

        let folded = fold_witnesses(&w1, &w2, &t, r);

        assert_eq!(folded.w[0], Fr::from_u64(3)); // 1 + 2*1
        assert_eq!(folded.w[1], Fr::from_u64(13)); // 3 + 2*5
        assert_eq!(folded.w[2], Fr::from_u64(59)); // 9 + 2*25

        // e' = 0 + 2*11 + 4*7 = 50
        assert_eq!(folded.e[0], Fr::from_u64(50));
    }

    #[test]
    fn fold_scalar_formula() {
        let u1 = Fr::from_u64(3);
        let u2 = Fr::from_u64(7);
        let r = Fr::from_u64(5);
        assert_eq!(fold_scalar(u1, u2, r), Fr::from_u64(38)); // 3 + 5*7
    }

    #[test]
    fn folded_witness_satisfies_relaxed_r1cs() {
        let r1cs = x_squared_r1cs();

        let z1 = vec![Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];
        let u1 = Fr::one();
        let w1 = RelaxedWitness {
            w: z1.clone(),
            e: vec![Fr::zero()],
        };

        let z2 = vec![Fr::from_u64(1), Fr::from_u64(5), Fr::from_u64(25)];
        let u2 = Fr::one();
        let w2 = RelaxedWitness {
            w: z2.clone(),
            e: vec![Fr::zero()],
        };

        let t = compute_cross_term(&r1cs, &z1, u1, &z2, u2);
        let r = Fr::from_u64(7);
        let folded = fold_witnesses(&w1, &w2, &t, r);
        let u_folded = fold_scalar(u1, u2, r);

        check_relaxed_satisfaction(&r1cs, u_folded, &folded).expect("folded must satisfy");
    }

    #[test]
    fn folded_two_constraint_system() {
        let r1cs = two_constraint_r1cs();

        let z1 = vec![Fr::from_u64(1), Fr::from_u64(4), Fr::from_u64(16)];
        let u1 = Fr::one();
        let w1 = RelaxedWitness {
            w: z1.clone(),
            e: vec![Fr::zero(); 2],
        };

        let z2 = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(4)];
        let u2 = Fr::one();
        let w2 = RelaxedWitness {
            w: z2.clone(),
            e: vec![Fr::zero(); 2],
        };

        let t = compute_cross_term(&r1cs, &z1, u1, &z2, u2);
        assert_eq!(t.len(), 2);

        let r = Fr::from_u64(13);
        let folded = fold_witnesses(&w1, &w2, &t, r);
        let u_folded = fold_scalar(u1, u2, r);

        check_relaxed_satisfaction(&r1cs, u_folded, &folded)
            .expect("two-constraint fold must satisfy");
    }

    #[test]
    fn random_witness_satisfies() {
        let r1cs = x_squared_r1cs();
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let (u, witness) = sample_random_witness(&r1cs, &mut rng);
        check_relaxed_satisfaction(&r1cs, u, &witness).expect("random witness must satisfy");
    }

    #[test]
    fn random_witness_two_constraints() {
        let r1cs = two_constraint_r1cs();
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let (u, witness) = sample_random_witness(&r1cs, &mut rng);
        check_relaxed_satisfaction(&r1cs, u, &witness).expect("random witness must satisfy");
    }

    #[test]
    fn identity_fold_r_equals_zero() {
        let r1cs = x_squared_r1cs();

        let z1 = vec![Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];
        let u1 = Fr::one();
        let w1 = RelaxedWitness {
            w: z1.clone(),
            e: vec![Fr::zero()],
        };

        let z2 = vec![Fr::from_u64(1), Fr::from_u64(5), Fr::from_u64(25)];
        let u2 = Fr::one();
        let w2 = RelaxedWitness {
            w: z2.clone(),
            e: vec![Fr::zero()],
        };

        let t = compute_cross_term(&r1cs, &z1, u1, &z2, u2);
        let r = Fr::zero();
        let folded = fold_witnesses(&w1, &w2, &t, r);
        let u_folded = fold_scalar(u1, u2, r);

        assert_eq!(u_folded, u1);
        assert_eq!(folded.w, w1.w);
        assert_eq!(folded.e, w1.e);
    }

    /// Mock commitment: a pair of field elements with element-wise linear combination.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct MockCom(Fr, Fr);

    impl HomomorphicCommitment<Fr> for MockCom {
        fn linear_combine(c1: &Self, c2: &Self, scalar: &Fr) -> Self {
            MockCom(c1.0 + *scalar * c2.0, c1.1 + *scalar * c2.1)
        }
    }

    #[test]
    fn instance_folding_with_mock_commitment() {
        let inst1 = RelaxedInstance {
            u: Fr::from_u64(1),
            w_commitment: MockCom(Fr::from_u64(10), Fr::from_u64(20)),
            e_commitment: MockCom(Fr::from_u64(0), Fr::from_u64(0)),
        };
        let inst2 = RelaxedInstance {
            u: Fr::from_u64(1),
            w_commitment: MockCom(Fr::from_u64(30), Fr::from_u64(40)),
            e_commitment: MockCom(Fr::from_u64(5), Fr::from_u64(6)),
        };
        let t_com = MockCom(Fr::from_u64(100), Fr::from_u64(200));
        let r = Fr::from_u64(3);

        let folded = fold_instances(&inst1, &inst2, &t_com, r);

        assert_eq!(folded.u, Fr::from_u64(4)); // 1 + 3*1

        // w' = (10+90, 20+120) = (100, 140)
        assert_eq!(
            folded.w_commitment,
            MockCom(Fr::from_u64(100), Fr::from_u64(140))
        );

        // e' = (0,0) + 3*(100,200) + 9*(5,6) = (345, 654)
        assert_eq!(
            folded.e_commitment,
            MockCom(Fr::from_u64(345), Fr::from_u64(654))
        );
    }

    #[test]
    fn end_to_end_fold_real_plus_random() {
        let r1cs = x_squared_r1cs();
        let mut rng = ChaCha20Rng::seed_from_u64(99);

        let z_real = vec![Fr::from_u64(1), Fr::from_u64(7), Fr::from_u64(49)];
        let u_real = Fr::one();
        let w_real = RelaxedWitness {
            w: z_real.clone(),
            e: vec![Fr::zero()],
        };

        let (u_rand, w_rand) = sample_random_witness(&r1cs, &mut rng);
        check_relaxed_satisfaction(&r1cs, u_rand, &w_rand).expect("random instance must satisfy");

        let t = compute_cross_term(&r1cs, &z_real, u_real, &w_rand.w, u_rand);
        let r = Fr::from_u64(42);
        let folded = fold_witnesses(&w_real, &w_rand, &t, r);
        let u_folded = fold_scalar(u_real, u_rand, r);

        check_relaxed_satisfaction(&r1cs, u_folded, &folded)
            .expect("real+random fold must satisfy");
    }

    #[test]
    fn end_to_end_with_verifier_r1cs() {
        use crate::verifier_r1cs::{
            assign_witness, build_verifier_r1cs, BakedPublicInputs, StageConfig,
        };

        // Build a verifier R1CS from a single degree-2 sumcheck stage with 3 rounds
        let claimed_sum = Fr::from_u64(44);
        let stages = vec![StageConfig {
            num_rounds: 3,
            degree: 2,
            claimed_sum,
        }];

        let challenges = vec![Fr::from_u64(11), Fr::from_u64(22), Fr::from_u64(33)];
        let baked = BakedPublicInputs {
            challenges: challenges.clone(),
        };

        let r1cs = build_verifier_r1cs(&stages, &baked);

        // Manually construct valid round coefficients.
        // For degree-2: round poly P(X) = c0 + c1*X + c2*X^2
        // Sum check: 2*c0 + c1 + c2 = running_sum
        // Eval check: c0 + r*c1 + r^2*c2 = next_running_sum
        let mut stage_coeffs = Vec::new();
        let mut running_sum = claimed_sum;

        for &r_i in &challenges {
            // Pick c2 freely, then derive c0, c1 from sum check
            let c2 = Fr::from_u64(1);
            // 2*c0 + c1 + c2 = running_sum
            // Choose c0 = running_sum / 3 (approximately — just pick consistent values)
            // Simpler: let c1 = 0, then 2*c0 + c2 = running_sum → c0 = (running_sum - c2) / 2
            // But we need inverse of 2...
            // Even simpler: c0 = 5, c1 = running_sum - 2*5 - 1 = running_sum - 11
            let c0 = Fr::from_u64(5);
            let c1 = running_sum - Fr::from_u64(2) * c0 - c2;

            stage_coeffs.push(vec![c0, c1, c2]);

            // Next running sum = P(r_i)
            running_sum = c0 + r_i * c1 + r_i * r_i * c2;
        }

        let z = assign_witness(&stages, &baked, &[stage_coeffs]);

        // Verify standard R1CS satisfaction
        let (az, bz, cz) = r1cs.multiply_witness(&z);
        for i in 0..r1cs.num_constraints() {
            assert_eq!(az[i] * bz[i], cz[i], "not satisfied at constraint {i}");
        }

        // Fold with random instance
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let w_real = RelaxedWitness {
            w: z.clone(),
            e: vec![Fr::zero(); r1cs.num_constraints()],
        };
        let u_real = Fr::one();

        let (u_rand, w_rand) = sample_random_witness(&r1cs, &mut rng);
        check_relaxed_satisfaction(&r1cs, u_rand, &w_rand).expect("random instance must satisfy");

        let t = compute_cross_term(&r1cs, &z, u_real, &w_rand.w, u_rand);
        let r = Fr::from_u64(17);
        let folded = fold_witnesses(&w_real, &w_rand, &t, r);
        let u_folded = fold_scalar(u_real, u_rand, r);

        check_relaxed_satisfaction(&r1cs, u_folded, &folded)
            .expect("folded verifier R1CS must satisfy");
    }
}
