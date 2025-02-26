use super::sumcheck::SumcheckInstanceProof;
use crate::{
    field::JoltField,
    jolt::instruction::JoltInstruction,
    poly::{
        eq_poly::EqPolynomial,
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
use rayon::{prelude::*, slice::Iter};
use std::{collections::HashMap, ops::Index};

struct ExpandingTable<F: JoltField> {
    len: usize,
    values: Vec<F>,
    scratch_space: Vec<F>,
    update_fn: Box<dyn Fn(&mut [F], F, F, usize) + Sync>,
}

impl<F: JoltField> ExpandingTable<F> {
    fn new(capacity: usize, update_fn: Box<dyn Fn(&mut [F], F, F, usize) + Sync>) -> Self {
        Self {
            len: 0,
            values: unsafe_allocate_zero_vec(capacity),
            scratch_space: unsafe_allocate_zero_vec(capacity),
            update_fn,
        }
    }

    fn reset(&mut self, value: F) {
        self.values[0] = value;
        self.len = 1;
    }

    fn update(&mut self, r_j: F, j: usize) {
        self.values
            .par_iter()
            .zip(self.scratch_space.par_chunks_mut(2))
            .for_each(|(src, dest)| {
                (self.update_fn)(dest, *src, r_j, j);
            });
        std::mem::swap(&mut self.values, &mut self.scratch_space);
        self.len *= 2;
    }
}

impl<F: JoltField> Index<usize> for ExpandingTable<F> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        assert!(index < self.len);
        &self.values[index]
    }
}

impl<'data, F: JoltField> IntoParallelIterator for &'data ExpandingTable<F> {
    type Item = &'data F;
    type Iter = Iter<'data, F>;

    fn into_par_iter(self) -> Self::Iter {
        self.values[..self.len].into_par_iter()
    }
}

pub fn prove_single_instruction<
    const TREE_WIDTH: usize,
    F: JoltField,
    I: JoltInstruction + Default,
    ProofTranscript: Transcript,
>(
    instructions: &[I],
    r_cycle: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, F, [F; 4]) {
    debug_assert!(TREE_WIDTH.is_power_of_two());
    let log_m = TREE_WIDTH.log_2();

    let T = instructions.len();
    let log_T = T.log_2();
    debug_assert_eq!(r_cycle.len(), log_T);

    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    let span = tracing::span!(tracing::Level::INFO, "compute instruction_index_iters");
    let _guard = span.enter();
    let instruction_index_iters: Vec<_> = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            instructions
                .par_iter()
                .enumerate()
                .filter_map(move |(j, instruction)| {
                    let k = instruction.to_lookup_index();
                    if k as usize % num_chunks == i {
                        Some(j)
                    } else {
                        None
                    }
                })
        })
        .collect();
    drop(_guard);
    drop(span);

    let eq_r_prime = EqPolynomial::evals(&r_cycle);
    // log(K) + log(T)
    let num_rounds = 4 * log_m + log_T;
    let mut r: Vec<F> = Vec::with_capacity(num_rounds);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let span = tracing::span!(tracing::Level::INFO, "compute initial u evals");
    let _guard = span.enter();
    let mut u_evals: Vec<(u64, F)> = instructions
        .par_iter()
        .zip(eq_r_prime.par_iter())
        .map(|(instruction, eq_eval)| (instruction.to_lookup_index(), *eq_eval))
        .collect();
    drop(_guard);
    drop(span);

    let mut t_evals: HashMap<u64, F> = HashMap::with_capacity(T + TREE_WIDTH);

    let mut rv_claim = F::zero();
    let mut previous_claim = F::zero();

    let update_v = |dest: &mut [F], v_i, r_j, _| {
        debug_assert_eq!(dest.len(), 2);
        let eval_1 = v_i * r_j;
        dest[0] = v_i - eval_1;
        dest[1] = eval_1;
    };
    let mut v = ExpandingTable::new(TREE_WIDTH, Box::new(update_v));

    let update_w = |dest: &mut [F], w_i: F, r_j: F, j: usize| {
        debug_assert_eq!(dest.len(), 2);
        dest[0] = w_i + I::default().additive_update(r_j, j, false);
        dest[1] = w_i + I::default().additive_update(r_j, j, true);
    };
    let mut w = ExpandingTable::new(TREE_WIDTH, Box::new(update_w));

    let update_x = |dest: &mut [F], x_i: F, r_j: F, j: usize| {
        debug_assert_eq!(dest.len(), 2);
        dest[0] = x_i * I::default().multiplicative_update(r_j, j, false);
        dest[1] = x_i * I::default().multiplicative_update(r_j, j, true);
    };
    let mut x = ExpandingTable::new(TREE_WIDTH, Box::new(update_x));

    let two = F::from_u8(2);
    let chi_2 = [F::one() - two, two];

    #[cfg(test)]
    let mut val_test: MultilinearPolynomial<F> =
        MultilinearPolynomial::from(I::default().materialize());
    #[cfg(test)]
    let mut eq_ra_test: MultilinearPolynomial<F> = {
        let mut eq_ra: Vec<F> = unsafe_allocate_zero_vec(TREE_WIDTH.pow(4));
        for (j, instruction) in instructions.iter().enumerate() {
            let k = instruction.to_lookup_index();
            eq_ra[k as usize] += eq_r_prime[j];
        }
        MultilinearPolynomial::from(eq_ra)
    };

    let mut j = 0;
    let mut ra: Vec<MultilinearPolynomial<F>> = Vec::with_capacity(4);

    for phase in 0..3 {
        let span = tracing::span!(tracing::Level::INFO, "sparse-dense phase");
        let _guard = span.enter();

        // Condensation
        if phase == 0 {
            let span = tracing::span!(tracing::Level::INFO, "compute initial t evals");
            let _guard = span.enter();
            t_evals.par_extend(
                (0..(TREE_WIDTH as u64))
                    .into_par_iter()
                    .map(|k| (k, F::from_u64(I::default().materialize_entry(k)))),
            );
            t_evals.par_extend(
                u_evals
                    .par_iter()
                    .map(|(k, _)| (*k, F::from_u64(I::default().materialize_entry(*k)))),
            );
        } else {
            let span = tracing::span!(tracing::Level::INFO, "Update u_evals");
            let _guard = span.enter();
            u_evals.par_iter_mut().for_each(|(k, u)| {
                let k_bound = ((*k >> ((4 - phase) * log_m)) % TREE_WIDTH as u64) as usize;
                *u *= v[k_bound as usize];
            });
            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Update t_evals");
            let _guard = span.enter();
            t_evals.par_iter_mut().for_each(|(k, t)| {
                let k_bound = ((k >> ((4 - phase) * log_m)) % TREE_WIDTH as u64) as usize;
                *t *= x[k_bound];
                *t += w[k_bound];
            });
        }

        // Build binary trees Q_\ell and Z for each \ell = 1, ..., \kappa

        let span = tracing::span!(tracing::Level::INFO, "Compute Z tree leaves");
        let _guard = span.enter();
        let z_leaves = u_evals
            .par_chunks(chunk_size)
            .map(|u_evals_chunk| {
                let mut result: Vec<F> = unsafe_allocate_zero_vec(TREE_WIDTH);
                for (k, u) in u_evals_chunk.iter() {
                    result[((k >> ((3 - phase) * log_m)) % TREE_WIDTH as u64) as usize] += *u;
                }
                result
            })
            .reduce(
                || unsafe_allocate_zero_vec(TREE_WIDTH),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running
                },
            );

        drop(_guard);
        drop(span);
        let span = tracing::span!(tracing::Level::INFO, "Build Z tree");
        let _guard = span.enter();

        let mut Z_tree: Vec<Vec<F>> = vec![vec![]; log_m + 1];
        Z_tree[log_m] = z_leaves;
        for layer_index in (0..log_m).rev() {
            Z_tree[layer_index] = Z_tree[layer_index + 1]
                .par_chunks(2)
                .map(|pair| pair[0] + pair[1])
                .collect();
        }
        drop(_guard);
        drop(span);

        debug_assert_eq!(Z_tree[0].len(), 1);
        debug_assert_eq!(Z_tree[0][0], Z_tree[log_m].par_iter().sum());

        let span = tracing::span!(tracing::Level::INFO, "Compute Q tree leaves");
        let _guard = span.enter();

        // TODO(moodlezoup): Handle \ell > 1
        let q_leaves = u_evals
            .par_chunks(chunk_size)
            .map(|u_evals_chunk| {
                let mut result: Vec<F> = unsafe_allocate_zero_vec(TREE_WIDTH);
                for (k, u) in u_evals_chunk.iter() {
                    let t = t_evals.get(&k).unwrap();
                    result[((k >> ((3 - phase) * log_m)) % TREE_WIDTH as u64) as usize] += *u * t;
                }
                result
            })
            .reduce(
                || unsafe_allocate_zero_vec(TREE_WIDTH),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running
                },
            );

        drop(_guard);
        drop(span);
        let span = tracing::span!(tracing::Level::INFO, "Build Q tree");
        let _guard = span.enter();

        let mut Q_tree: Vec<Vec<F>> = vec![vec![]; log_m + 1];
        Q_tree[log_m] = q_leaves;
        for layer_index in (0..log_m).rev() {
            Q_tree[layer_index] = Q_tree[layer_index + 1]
                .par_chunks(2)
                .map(|pair| pair[0] + pair[1])
                .collect();
        }
        drop(_guard);
        drop(span);

        debug_assert_eq!(Q_tree[0].len(), 1);
        debug_assert_eq!(Q_tree[0][0], Q_tree[log_m].par_iter().sum());

        v.reset(F::one());
        w.reset(F::zero());
        x.reset(F::one());

        previous_claim = Q_tree[0][0];
        if phase == 0 {
            rv_claim = Q_tree[0][0];
        }

        for round in 0..log_m {
            let span = tracing::span!(tracing::Level::INFO, "sparse-dense sumcheck round");
            let _guard = span.enter();

            #[cfg(test)]
            {
                let expected: F = (0..val_test.len())
                    .map(|k| eq_ra_test.get_bound_coeff(k) * val_test.get_bound_coeff(k))
                    .sum();
                assert_eq!(
                    expected, previous_claim,
                    "Sumcheck sanity check failed in phase {phase} round {round}"
                );
            }

            let prover_message_span =
                tracing::span!(tracing::Level::INFO, "Compute univariate poly");
            let _prover_message_guard = prover_message_span.enter();

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

            drop(_prover_message_guard);
            drop(prover_message_span);

            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            r.push(r_j);

            previous_claim = univariate_poly.evaluate(&r_j);

            let binding_span = tracing::span!(tracing::Level::INFO, "Binding");
            let _binding_guard = binding_span.enter();

            v.update(r_j, j);
            w.update(r_j, j);
            x.update(r_j, j);

            #[cfg(test)]
            {
                eq_ra_test.bind_parallel(r_j, BindingOrder::HighToLow);
                val_test.bind_parallel(r_j, BindingOrder::HighToLow);
            }

            j += 1;
        }

        let span = tracing::span!(tracing::Level::INFO, "cache ra_i");
        let _guard = span.enter();

        let ra_i: Vec<F> = instructions
            .par_iter()
            .map(|instruction| {
                let k = instruction.to_lookup_index();
                let k_bound = ((k >> ((3 - phase) * log_m)) % TREE_WIDTH as u64) as usize;
                v[k_bound as usize]
            })
            .collect();
        ra.push(MultilinearPolynomial::from(ra_i));
    }

    // At this point we switch from sparse-dense sumcheck (see Section 7.1 of the Twist+Shout
    // paper) to "vanilla" Shout, i.e. Section 6.2 where d=4.
    // Note that we've already bound 3/4 of the address variables, so ra_1, ra_2, and ra_3
    // are fully bound when we start "vanilla" Shout.

    // Modified version of the C array described in Equation (47) of the Twist+Shout paper

    let span = tracing::span!(tracing::Level::INFO, "Materialize eq_ra");
    let _guard = span.enter();
    let eq_ra = instructions
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, instructions_chunk)| {
            let mut result: Vec<F> = unsafe_allocate_zero_vec(TREE_WIDTH);
            for (j, instruction) in instructions_chunk.iter().enumerate() {
                let j = chunk_index * chunk_size + j;
                let k = instruction.to_lookup_index();
                result[(k as usize) % TREE_WIDTH] +=
                    eq_r_prime[j] * ra[0].get_coeff(j) * ra[1].get_coeff(j) * ra[2].get_coeff(j);
            }
            result
        })
        .reduce(
            || unsafe_allocate_zero_vec(TREE_WIDTH),
            |mut running, new| {
                running
                    .par_iter_mut()
                    .zip(new.into_par_iter())
                    .for_each(|(x, y)| *x += y);
                running
            },
        );

    let mut eq_ra = MultilinearPolynomial::from(eq_ra);
    drop(_guard);
    drop(span);

    #[cfg(test)]
    {
        for i in 0..TREE_WIDTH {
            assert_eq!(eq_ra.get_bound_coeff(i), eq_ra_test.get_bound_coeff(i));
        }
    }

    let span = tracing::span!(tracing::Level::INFO, "Materialize val");
    let _guard = span.enter();
    let val: Vec<F> = (0..TREE_WIDTH)
        .into_par_iter()
        .map(|k| x[0] * t_evals.get(&(k as u64)).unwrap() + w[0])
        .collect();
    let mut val = MultilinearPolynomial::from(val);
    drop(_guard);
    drop(span);

    #[cfg(test)]
    {
        for i in 0..TREE_WIDTH {
            assert_eq!(val.get_bound_coeff(i), val_test.get_bound_coeff(i));
        }
    }

    v.reset(F::one());

    let span = tracing::span!(tracing::Level::INFO, "Next log(m) sumcheck rounds");
    let _guard = span.enter();

    for _round in 0..log_m {
        let univariate_poly_evals: [F; 2] = (0..eq_ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_ra_evals = eq_ra.sumcheck_evals(i, 2, BindingOrder::HighToLow);
                let val_evals = val.sumcheck_evals(i, 2, BindingOrder::HighToLow);

                [eq_ra_evals[0] * val_evals[0], eq_ra_evals[1] * val_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

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

        // Bind polynomials
        rayon::join(
            || eq_ra.bind_parallel(r_j, BindingOrder::HighToLow),
            || val.bind_parallel(r_j, BindingOrder::HighToLow),
        );

        v.update(r_j, j);
    }

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "cache ra_i");
    let _guard = span.enter();

    let ra_i: Vec<F> = instructions
        .par_iter()
        .map(|instruction| {
            let k = instruction.to_lookup_index();
            let k_bound = (k % TREE_WIDTH as u64) as usize;
            v[k_bound as usize]
        })
        .collect();
    ra.push(MultilinearPolynomial::from(ra_i));
    drop(_guard);
    drop(span);

    let mut eq_r_prime = MultilinearPolynomial::from(eq_r_prime);
    let val_eval = val.final_sumcheck_claim();

    let span = tracing::span!(tracing::Level::INFO, "last log(T) sumcheck rounds");
    let _guard = span.enter();

    for _round in 0..log_T {
        let mut univariate_poly_evals: [F; 5] = (0..eq_r_prime.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = eq_r_prime.sumcheck_evals(i, 5, BindingOrder::HighToLow);
                let ra_0_evals = ra[0].sumcheck_evals(i, 5, BindingOrder::HighToLow);
                let ra_1_evals = ra[1].sumcheck_evals(i, 5, BindingOrder::HighToLow);
                let ra_2_evals = ra[2].sumcheck_evals(i, 5, BindingOrder::HighToLow);
                let ra_3_evals = ra[3].sumcheck_evals(i, 5, BindingOrder::HighToLow);

                [
                    eq_evals[0] * ra_0_evals[0] * ra_1_evals[0] * ra_2_evals[0] * ra_3_evals[0],
                    eq_evals[1] * ra_0_evals[1] * ra_1_evals[1] * ra_2_evals[1] * ra_3_evals[1],
                    eq_evals[2] * ra_0_evals[2] * ra_1_evals[2] * ra_2_evals[2] * ra_3_evals[2],
                    eq_evals[3] * ra_0_evals[3] * ra_1_evals[3] * ra_2_evals[3] * ra_3_evals[3],
                    eq_evals[4] * ra_0_evals[4] * ra_1_evals[4] * ra_2_evals[4] * ra_3_evals[4],
                ]
            })
            .reduce(
                || [F::zero(); 5],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                        running[3] + new[3],
                        running[4] + new[4],
                    ]
                },
            );
        univariate_poly_evals
            .iter_mut()
            .for_each(|eval| *eval *= val_eval);

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
            univariate_poly_evals[3],
            univariate_poly_evals[4],
        ]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        ra.par_iter_mut()
            .chain([&mut eq_r_prime].into_par_iter())
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
    }

    (
        SumcheckInstanceProof::new(compressed_polys),
        rv_claim,
        [
            ra[0].final_sumcheck_claim(),
            ra[1].final_sumcheck_claim(),
            ra[2].final_sumcheck_claim(),
            ra[3].final_sumcheck_claim(),
        ],
    )
}

pub fn verify_single_instruction<
    F: JoltField,
    I: JoltInstruction + Default,
    ProofTranscript: Transcript,
>(
    proof: SumcheckInstanceProof<F, ProofTranscript>,
    K: usize,
    T: usize,
    r_cycle: Vec<F>,
    rv_claim: F,
    ra_claims: [F; 4],
    transcript: &mut ProofTranscript,
) -> Result<(), ProofVerifyError> {
    let (sumcheck_claim, r) = proof.verify(rv_claim, K.log_2() + T.log_2(), 5, transcript)?;
    let (r_address, r_cycle_prime) = r.split_at(K.log_2());

    let val_eval = I::default().evaluate_mle(r_address);
    let eq_eval_cycle = EqPolynomial::new(r_cycle).evaluate(r_cycle_prime);

    assert_eq!(
        eq_eval_cycle * ra_claims.iter().product::<F>() * val_eval,
        sumcheck_claim,
        "Read-checking sumcheck failed"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        jolt::instruction::{add::ADDInstruction, mulhu::MULHUInstruction},
        utils::transcript::KeccakTranscript,
    };
    use ark_bn254::Fr;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_mulhu() {
        const WORD_SIZE: usize = 8;
        const K: usize = 1 << 16;
        const T: usize = 1 << 8;

        let mut rng = StdRng::seed_from_u64(12345);

        let instructions: Vec<_> = (0..T)
            .map(|_| MULHUInstruction::<WORD_SIZE>::default().random(&mut rng))
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        const TREE_WIDTH: usize = 1 << 4;
        let (proof, rv_claim, ra_claims) = prove_single_instruction::<TREE_WIDTH, _, _, _>(
            &instructions,
            r_cycle,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());
        let verification_result = verify_single_instruction::<_, MULHUInstruction<WORD_SIZE>, _>(
            proof,
            K,
            T,
            r_cycle,
            rv_claim,
            ra_claims,
            &mut verifier_transcript,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn test_add() {
        const WORD_SIZE: usize = 8;
        const K: usize = 1 << 16;
        const T: usize = 1 << 8;

        let mut rng = StdRng::seed_from_u64(12345);

        let instructions: Vec<_> = (0..T)
            .map(|_| ADDInstruction::<WORD_SIZE>::default().random(&mut rng))
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        const TREE_WIDTH: usize = 1 << 4;
        let (proof, rv_claim, ra_claims) = prove_single_instruction::<TREE_WIDTH, _, _, _>(
            &instructions,
            r_cycle,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());
        let verification_result = verify_single_instruction::<_, ADDInstruction<WORD_SIZE>, _>(
            proof,
            K,
            T,
            r_cycle,
            rv_claim,
            ra_claims,
            &mut verifier_transcript,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }
}
