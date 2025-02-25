use std::collections::HashMap;

use super::sumcheck::SumcheckInstanceProof;
use crate::{
    field::JoltField,
    jolt::instruction::JoltInstruction,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::{CompressedUniPoly, UniPoly},
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::unsafe_allocate_zero_vec,
        transcript::{AppendToTranscript, Transcript},
    },
};
use rayon::prelude::*;

pub fn prove_single_instruction<
    const TREE_WIDTH: usize,
    F: JoltField,
    I: JoltInstruction + Default,
    ProofTranscript: Transcript,
>(
    instructions: &[I],
    r_prime: Vec<F>,
    transcript: &mut ProofTranscript,
) {
    debug_assert!(TREE_WIDTH.is_power_of_two());
    let log_m = TREE_WIDTH.log_2();

    let T = instructions.len();
    let log_T = T.log_2();
    debug_assert_eq!(r_prime.len(), log_T);

    let eq_r_prime = EqPolynomial::evals(&r_prime);
    // log(K) + log(T)
    let num_rounds = 4 * log_m + log_T;
    let mut r: Vec<F> = Vec::with_capacity(num_rounds);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let mut u_evals: HashMap<u64, F> = HashMap::with_capacity(T);
    let mut t_evals: HashMap<u64, F> = HashMap::with_capacity(T);

    let mut previous_claim;

    let mut v = Vec::with_capacity(TREE_WIDTH);
    let mut w = Vec::with_capacity(TREE_WIDTH);
    let mut x = Vec::with_capacity(TREE_WIDTH);

    let two = F::from_u8(2);
    let chi_2 = [F::one() - two, two];

    #[cfg(test)]
    let mut val: MultilinearPolynomial<F> = MultilinearPolynomial::from(I::default().materialize());
    #[cfg(test)]
    let mut eq_ra: MultilinearPolynomial<F> = {
        let mut eq_ra: Vec<F> = unsafe_allocate_zero_vec(1 << 16);
        for (j, instruction) in instructions.iter().enumerate() {
            let k = instruction.to_lookup_index();
            eq_ra[k as usize] += eq_r_prime[j];
        }
        MultilinearPolynomial::from(eq_ra)
    };

    let mut j = 0;

    for phase in 0..3 {
        // Condensation
        if phase != 0 {
            u_evals.par_iter_mut().for_each(|(k, u)| {
                let k_bound = ((k >> ((4 - phase) * log_m)) % TREE_WIDTH as u64) as usize;
                *u *= v[k_bound as usize];
            });

            t_evals.par_iter_mut().for_each(|(k, t)| {
                let k_bound = ((k >> ((4 - phase) * log_m)) % TREE_WIDTH as u64) as usize;
                *t *= x[k_bound];
                *t += w[k_bound];
            });
        }

        // Build binary trees Q_\ell and Z for each \ell = 1, ..., \kappa
        let mut z_leaves = unsafe_allocate_zero_vec(TREE_WIDTH);
        // TODO(moodlezoup): parallelize
        for (j, instruction) in instructions.iter().enumerate() {
            let k = instruction.to_lookup_index();
            let u = match u_evals.get(&k) {
                Some(eval) => *eval,
                None => {
                    debug_assert_eq!(phase, 0);
                    let eval = eq_r_prime[j];
                    u_evals.insert(k, eval);
                    eval
                }
            };
            z_leaves[((k >> ((3 - phase) * log_m)) % TREE_WIDTH as u64) as usize] += u;
        }

        let mut Z_tree: Vec<Vec<F>> = vec![vec![]; log_m + 1];
        Z_tree[log_m] = z_leaves;
        for layer_index in (0..log_m).rev() {
            Z_tree[layer_index] = Z_tree[layer_index + 1]
                .par_chunks(2)
                .map(|pair| pair[0] + pair[1])
                .collect();
        }

        debug_assert_eq!(Z_tree[0].len(), 1);
        debug_assert_eq!(Z_tree[0][0], Z_tree[log_m].par_iter().sum());

        // TODO(moodlezoup): Handle \ell > 1
        let mut q_leaves = unsafe_allocate_zero_vec(TREE_WIDTH);
        // TODO(moodlezoup): parallelize
        for instruction in instructions.iter() {
            let k = instruction.to_lookup_index();
            let u = u_evals.get(&k).unwrap();
            let t = match t_evals.get(&k) {
                Some(eval) => *eval,
                None => {
                    debug_assert_eq!(phase, 0);
                    let eval = F::from_u64(instruction.lookup_entry());
                    t_evals.insert(k, eval);
                    eval
                }
            };
            q_leaves[((k >> ((3 - phase) * log_m)) % TREE_WIDTH as u64) as usize] += *u * t;
        }

        let mut Q_tree: Vec<Vec<F>> = vec![vec![]; log_m + 1];
        Q_tree[log_m] = q_leaves;
        for layer_index in (0..log_m).rev() {
            Q_tree[layer_index] = Q_tree[layer_index + 1]
                .par_chunks(2)
                .map(|pair| pair[0] + pair[1])
                .collect();
        }
        debug_assert_eq!(Q_tree[0].len(), 1);
        debug_assert_eq!(Q_tree[0][0], Q_tree[log_m].par_iter().sum());

        v = vec![F::one()];
        w = vec![F::zero()];
        x = vec![F::one()];

        previous_claim = Q_tree[0][0];

        for round in 0..log_m {
            #[cfg(test)]
            {
                let expected: F = (0..val.len())
                    .map(|k| eq_ra.get_bound_coeff(k) * val.get_bound_coeff(k))
                    .sum();
                assert_eq!(
                    expected, previous_claim,
                    "Sumcheck sanity check failed in round {j}"
                );
            }

            let q_layer = &Q_tree[round + 1];
            let z_layer = &Z_tree[round + 1];

            let q_inner_products: [F; 2] = (&v, &x)
                .into_par_iter()
                .enumerate()
                .map(|(b, (v, x))| {
                    // Element of v ◦ x
                    let v_x = *v * x;
                    // Contribution to <v ◦ x, q_even>
                    let q_v_x_even = v_x * q_layer[2 * b];
                    let q_v_x_odd = v_x * q_layer[2 * b + 1];

                    [q_v_x_even, q_v_x_odd]
                })
                .reduce(
                    || [F::zero(); 2],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                );

            let z_inner_products: [F; 4] = (&v, &w)
                .into_par_iter()
                .enumerate()
                .map(|(b, (v_b, w_b))| {
                    // Contribution to <v, z_even>
                    let z_v_even = z_layer[2 * b] * v_b;
                    // Contribution to <v, z_odd>
                    let z_v_odd = z_layer[2 * b + 1] * v_b;
                    // Contribution to <v ◦ w, z_even>
                    let z_v_w_even = z_v_even * w_b;
                    // Contribution to <v ◦ w, z_odd>
                    let z_v_w_odd = z_v_odd * w_b;

                    [z_v_even, z_v_odd, z_v_w_even, z_v_w_odd]
                })
                .reduce(
                    || [F::zero(); 4],
                    |running, new| {
                        [
                            running[0] + new[0],
                            running[1] + new[1],
                            running[2] + new[2],
                            running[3] + new[3],
                        ]
                    },
                );

            let mut univariate_poly_evals = [F::zero(), F::zero()];

            // Expression (52), c = 0
            univariate_poly_evals[0] +=
                I::default().multiplicative_update(F::zero(), j, false) * q_inner_products[0];
            // Expression (53), c = 0
            univariate_poly_evals[0] += z_inner_products[2]
                + I::default().additive_update(F::zero(), j, false) * z_inner_products[0];

            // Expression (52), c = 2
            univariate_poly_evals[1] +=
                chi_2[0] * I::default().multiplicative_update(two, j, false) * q_inner_products[0];
            univariate_poly_evals[1] +=
                chi_2[1] * I::default().multiplicative_update(two, j, true) * q_inner_products[1];
            // Expression (53), c = 2
            univariate_poly_evals[1] += chi_2[0]
                * (z_inner_products[2]
                    + I::default().additive_update(two, j, false) * z_inner_products[0]);
            univariate_poly_evals[1] += chi_2[1]
                * (z_inner_products[3]
                    + I::default().additive_update(two, j, true) * z_inner_products[1]);

            let univariate_poly = UniPoly::from_evals(&[
                univariate_poly_evals[0],
                previous_claim - univariate_poly_evals[0],
                univariate_poly_evals[1],
            ]);

            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            r.push(r_j);

            previous_claim = univariate_poly.evaluate(&r_j);

            v = v
                .into_par_iter()
                .flat_map(|v_i| {
                    let eval_1 = v_i * r_j;
                    [v_i - eval_1, eval_1]
                })
                .collect();

            let additive_updates = [
                I::default().additive_update(r_j, j, false),
                I::default().additive_update(r_j, j, true),
            ];
            w = w
                .into_par_iter()
                .flat_map(|w_i| [w_i + additive_updates[0], w_i + additive_updates[1]])
                .collect();

            let multiplicative_updates = [
                I::default().multiplicative_update(r_j, j, false),
                I::default().multiplicative_update(r_j, j, true),
            ];
            x = x
                .into_par_iter()
                .flat_map(|x_i| {
                    [
                        x_i * multiplicative_updates[0],
                        x_i * multiplicative_updates[1],
                    ]
                })
                .collect();

            #[cfg(test)]
            {
                eq_ra.bind_parallel(r_j, BindingOrder::HighToLow);
                val.bind_parallel(r_j, BindingOrder::HighToLow);
            }

            j += 1;
        }
    }

    todo!("You made it!")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{jolt::instruction::mulhu::MULHUInstruction, utils::transcript::KeccakTranscript};
    use ark_bn254::Fr;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_range_check() {
        const WORD_SIZE: usize = 8;
        const K: usize = 1 << 16;
        const T: usize = 1 << 4;

        let mut rng = StdRng::seed_from_u64(12345);

        let instructions: Vec<_> = (0..T)
            .map(|_| MULHUInstruction::<WORD_SIZE>::default().random(&mut rng))
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_prime: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        const TREE_WIDTH: usize = 1 << 4;
        prove_single_instruction::<TREE_WIDTH, _, _, _>(
            &instructions,
            r_prime,
            &mut prover_transcript,
        );
    }
}
