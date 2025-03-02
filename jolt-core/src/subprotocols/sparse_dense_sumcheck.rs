use super::sumcheck::SumcheckInstanceProof;
use crate::{
    field::JoltField,
    jolt::instruction::{mulhu::MULHUInstruction, JoltInstruction},
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
}

impl<F: JoltField> ExpandingTable<F> {
    #[tracing::instrument(skip_all, name = "ExpandingTable::new")]
    fn new(capacity: usize) -> Self {
        let (values, scratch_space) = rayon::join(
            || unsafe_allocate_zero_vec(capacity),
            || unsafe_allocate_zero_vec(capacity),
        );
        Self {
            len: 0,
            values,
            scratch_space,
        }
    }

    fn reset(&mut self, value: F) {
        self.values[0] = value;
        self.len = 1;
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

struct BinarySumTree<F: JoltField> {
    layers: Vec<Vec<F>>,
}

impl<F: JoltField> BinarySumTree<F> {
    #[tracing::instrument(skip_all, name = "BinarySumTree::new")]
    fn new(num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            layers.push(unsafe_allocate_zero_vec(1 << layer));
        }
        Self { layers }
    }

    fn leaves_mut(&mut self) -> &mut Vec<F> {
        let num_layers = self.layers.len();
        &mut self.layers[num_layers - 1]
    }

    #[tracing::instrument(skip_all, name = "BinarySumTree::build")]
    fn build(&mut self) {
        let num_layers = self.layers.len();
        for layer_index in (0..(num_layers - 1)).rev() {
            let (top, bottom) = self.layers.split_at_mut(layer_index + 1);
            let layer = top.last_mut().unwrap();
            let prev_layer = bottom.first().unwrap();
            layer
                .par_iter_mut()
                .zip(prev_layer.par_chunks(2))
                .for_each(|(dest, src)| *dest = src[0] + src[1]);
        }
    }
}

impl<F: JoltField> Index<usize> for BinarySumTree<F> {
    type Output = Vec<F>;

    fn index(&self, index: usize) -> &Vec<F> {
        assert!(index < self.layers.len());
        &self.layers[index]
    }
}

pub trait SparseDenseSumcheck<F: JoltField>: JoltInstruction + Default {
    const GAMMA: usize;
    // const ETA: usize

    #[tracing::instrument(skip_all, name = "SparseDenseSumcheck::compute_prover_message")]
    fn compute_prover_message(
        round: usize,
        Q_tree: &BinarySumTree<F>,
        Z_tree: &BinarySumTree<F>,
        r: &[F],
        j: usize,
        v: &ExpandingTable<F>,
        x: &ExpandingTable<F>,
        w: &ExpandingTable<F>,
    ) -> [F; 2] {
        let two = F::from_u8(2);
        let chi_2 = [F::one() - two, two];

        match Self::GAMMA {
            0 => {
                let q_layer = &Q_tree[round + 1];
                let z_layer = &Z_tree[round + 1];

                let q_inner_products: [F; 2] = (v, x)
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

                let z_inner_products: [F; 4] = (v, w)
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
                    Self::default().multiplicative_update(j, F::zero(), 0, None, None)
                        * q_inner_products[0];
                // Expression (53), c = 0
                univariate_poly_evals[0] += z_inner_products[2]
                    + Self::default().additive_update(j, F::zero(), 0, None, None)
                        * z_inner_products[0];

                // Expression (52), c = 2
                univariate_poly_evals[1] += chi_2[0]
                    * Self::default().multiplicative_update(j, two, 0, None, None)
                    * q_inner_products[0];
                univariate_poly_evals[1] += chi_2[1]
                    * Self::default().multiplicative_update(j, two, 1, None, None)
                    * q_inner_products[1];
                // Expression (53), c = 2
                univariate_poly_evals[1] += chi_2[0]
                    * (z_inner_products[2]
                        + Self::default().additive_update(j, two, 0, None, None)
                            * z_inner_products[0]);
                univariate_poly_evals[1] += chi_2[1]
                    * (z_inner_products[3]
                        + Self::default().additive_update(j, two, 1, None, None)
                            * z_inner_products[1]);

                univariate_poly_evals
            }
            1 => {
                let (q_layer, z_layer) = if j % 2 == 0 {
                    (&Q_tree[round + 1], &Z_tree[round + 1])
                } else {
                    (&Q_tree[round + 2], &Z_tree[round + 2])
                };

                debug_assert_eq!(q_layer.len(), z_layer.len());
                debug_assert_eq!(2 * v.len, q_layer.len());
                if j % 2 == 0 {
                    debug_assert_eq!(2 * x.len, q_layer.len());
                    debug_assert_eq!(2 * w.len, q_layer.len());
                } else {
                    debug_assert_eq!(4 * x.len, q_layer.len());
                    debug_assert_eq!(4 * w.len, q_layer.len());
                }

                todo!()
            }
            _ => unimplemented!("gamma > 1 not supported"),
        }
    }

    #[tracing::instrument(skip_all, name = "SparseDenseSumcheck::update_tables")]
    fn update_tables(
        v: &mut ExpandingTable<F>,
        x: &mut ExpandingTable<F>,
        w: &mut ExpandingTable<F>,
        r: &[F],
        j: usize,
    ) {
        Self::update_v_table(v, r, j);

        match Self::GAMMA {
            0 => {
                x.values
                    .par_iter()
                    .zip(x.scratch_space.par_chunks_mut(2))
                    .for_each(|(&x_i, dest)| {
                        dest[0] =
                            x_i * Self::default().multiplicative_update(j, r[j], 0, None, None);
                        dest[1] =
                            x_i * Self::default().multiplicative_update(j, r[j], 1, None, None);
                    });
                std::mem::swap(&mut x.values, &mut x.scratch_space);
                x.len *= 2;

                w.values
                    .par_iter()
                    .zip(w.scratch_space.par_chunks_mut(2))
                    .for_each(|(&w_i, dest)| {
                        dest[0] = w_i + Self::default().additive_update(j, r[j], 0, None, None);
                        dest[1] = w_i + Self::default().additive_update(j, r[j], 1, None, None);
                    });
                std::mem::swap(&mut w.values, &mut w.scratch_space);
                w.len *= 2;
            }
            1 => {
                if j % 2 == 0 {
                    return;
                }
                x.values
                    .par_iter()
                    .zip(x.scratch_space.par_chunks_mut(4))
                    .for_each(|(&x_i, dest)| {
                        for (b_j, b_next) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                            dest[2 * b_j + b_next] = x_i
                                * Self::default().multiplicative_update(
                                    j,
                                    r[j],
                                    b_j as u8,
                                    Some(r[j - 1]),
                                    Some(b_next as u8),
                                );
                        }
                    });
                std::mem::swap(&mut x.values, &mut x.scratch_space);
                x.len *= 4;

                w.values
                    .par_iter()
                    .zip(w.scratch_space.par_chunks_mut(4))
                    .for_each(|(&w_i, dest)| {
                        for (b_j, b_next) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                            dest[2 * b_j + b_next] = w_i
                                + Self::default().additive_update(
                                    j,
                                    r[j],
                                    b_j as u8,
                                    Some(r[j - 1]),
                                    Some(b_next as u8),
                                );
                        }
                    });
                std::mem::swap(&mut w.values, &mut w.scratch_space);
                w.len *= 4;
            }
            _ => unimplemented!("gamma > 1 not supported"),
        }
    }

    #[tracing::instrument(skip_all, name = "SparseDenseSumcheck::update_v_table")]
    fn update_v_table(v: &mut ExpandingTable<F>, r: &[F], j: usize) {
        v.values
            .par_iter()
            .zip(v.scratch_space.par_chunks_mut(2))
            .for_each(|(&v_i, dest)| {
                let eval_1 = r[j] * v_i;
                dest[0] = v_i - eval_1;
                dest[1] = eval_1;
            });
        std::mem::swap(&mut v.values, &mut v.scratch_space);
        v.len *= 2;
    }
}

impl<F: JoltField> SparseDenseSumcheck<F> for MULHUInstruction<32> {
    const GAMMA: usize = 0;
}

pub fn prove_single_instruction<
    const TREE_WIDTH: usize,
    F: JoltField,
    I: SparseDenseSumcheck<F>,
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
    let leaves_per_chunk = (TREE_WIDTH / num_chunks).max(1);

    let eq_r_prime = EqPolynomial::evals(&r_cycle);
    // log(K) + log(T)
    let num_rounds = 4 * log_m + log_T;
    let mut r: Vec<F> = Vec::with_capacity(num_rounds);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let span = tracing::span!(tracing::Level::INFO, "compute lookup indices");
    let _guard = span.enter();
    let lookup_indices: Vec<_> = instructions
        .par_iter()
        .map(|instruction| instruction.to_lookup_index())
        .collect();
    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "compute initial u evals");
    let _guard = span.enter();
    let mut u_evals = eq_r_prime.clone();
    drop(_guard);
    drop(span);

    let mut t_evals: HashMap<u64, F> = HashMap::with_capacity(T + TREE_WIDTH);
    let mut t_evals_vec: Vec<usize> = Vec::with_capacity(T);

    let mut rv_claim = F::zero();
    let mut previous_claim = F::zero();

    let mut v = ExpandingTable::new(TREE_WIDTH);
    let mut w = ExpandingTable::new(TREE_WIDTH);
    let mut x = ExpandingTable::new(TREE_WIDTH);

    #[cfg(test)]
    let mut val_test: MultilinearPolynomial<F> =
        MultilinearPolynomial::from(I::default().materialize());
    #[cfg(test)]
    let mut eq_ra_test: MultilinearPolynomial<F> = {
        let mut eq_ra: Vec<F> = unsafe_allocate_zero_vec(TREE_WIDTH.pow(4));
        for (j, k) in lookup_indices.iter().enumerate() {
            eq_ra[*k as usize] += eq_r_prime[j];
        }
        MultilinearPolynomial::from(eq_ra)
    };

    let mut j: usize = 0;
    let mut ra: Vec<MultilinearPolynomial<F>> = Vec::with_capacity(4);

    let (mut Z_tree, mut Q_tree) = rayon::join(
        || BinarySumTree::new(log_m + 1),
        || BinarySumTree::new(log_m + 1),
    );

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
                    .map(|k| (k, F::from_u64(I::default().materialize_entry(k))))
                    .chain(
                        lookup_indices
                            .par_iter()
                            .map(|k| (*k, F::from_u64(I::default().materialize_entry(*k)))),
                    ),
            );
            t_evals_vec = lookup_indices
                .par_iter()
                .map(|k| (t_evals.get(k).unwrap() as *const F) as usize)
                .collect();
        } else {
            let span = tracing::span!(tracing::Level::INFO, "Update u_evals");
            let _guard = span.enter();
            lookup_indices
                .par_iter()
                .zip(u_evals.par_iter_mut())
                .for_each(|(k, u)| {
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
        let span = tracing::span!(tracing::Level::INFO, "compute instruction_index_iters");
        let _guard = span.enter();
        let instruction_index_iters: Vec<_> = (0..num_chunks)
            .into_par_iter()
            .map(|i| {
                lookup_indices.iter().enumerate().filter_map(move |(j, k)| {
                    let group = ((k >> ((3 - phase) * log_m)) % TREE_WIDTH as u64)
                        / leaves_per_chunk as u64;
                    if group == i as u64 {
                        Some(j)
                    } else {
                        None
                    }
                })
            })
            .collect();
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Compute Q/Z tree leaves");
        let _guard = span.enter();
        let z_leaves = Z_tree.leaves_mut();
        let q_leaves = Q_tree.leaves_mut();
        if phase != 0 {
            rayon::join(
                || z_leaves.par_iter_mut().for_each(|leaf| *leaf = F::zero()),
                || q_leaves.par_iter_mut().for_each(|leaf| *leaf = F::zero()),
            );
        }

        instruction_index_iters
            .into_par_iter()
            .zip(z_leaves.par_chunks_mut(leaves_per_chunk))
            .zip(q_leaves.par_chunks_mut(leaves_per_chunk))
            .for_each(|((j_iter, z_leaves), q_leaves)| {
                j_iter.for_each(|j| {
                    let k = lookup_indices[j];
                    let u = u_evals[j];
                    // let t = t_evals.get(&k).unwrap();
                    let t = unsafe { *(t_evals_vec[j] as *const F) };
                    let leaf_index =
                        ((k >> ((3 - phase) * log_m)) % leaves_per_chunk as u64) as usize;
                    z_leaves[leaf_index] += u;
                    q_leaves[leaf_index] += u * t;
                });
            });
        drop(_guard);
        drop(span);

        rayon::join(|| Z_tree.build(), || Q_tree.build());

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

            let univariate_poly_evals =
                I::compute_prover_message(round, &Q_tree, &Z_tree, &r, j, &v, &x, &w);

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

            I::update_tables(&mut v, &mut x, &mut w, &r, j);

            #[cfg(test)]
            {
                eq_ra_test.bind_parallel(r_j, BindingOrder::HighToLow);
                val_test.bind_parallel(r_j, BindingOrder::HighToLow);
            }

            j += 1;
        }

        let span = tracing::span!(tracing::Level::INFO, "cache ra_i");
        let _guard = span.enter();

        let ra_i: Vec<F> = lookup_indices
            .par_iter()
            .map(|k| {
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
    let instruction_index_iters: Vec<_> = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            lookup_indices.iter().enumerate().filter_map(move |(j, k)| {
                let group = (k % TREE_WIDTH as u64) / leaves_per_chunk as u64;
                if group == i as u64 {
                    Some(j)
                } else {
                    None
                }
            })
        })
        .collect();

    let mut eq_ra: Vec<F> = unsafe_allocate_zero_vec(TREE_WIDTH);
    instruction_index_iters
        .into_par_iter()
        .zip(eq_ra.par_chunks_mut(leaves_per_chunk))
        .for_each(|(j_iter, leaves)| {
            j_iter.for_each(|j| {
                let k = lookup_indices[j];
                leaves[(k as usize) % leaves_per_chunk] +=
                    eq_r_prime[j] * ra[0].get_coeff(j) * ra[1].get_coeff(j) * ra[2].get_coeff(j);
            });
        });
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
        let span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _guard = span.enter();

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

        drop(_guard);
        drop(span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        let span = tracing::span!(tracing::Level::INFO, "Binding");
        let _guard = span.enter();

        // Bind polynomials
        rayon::join(
            || eq_ra.bind_parallel(r_j, BindingOrder::HighToLow),
            || val.bind_parallel(r_j, BindingOrder::HighToLow),
        );

        I::update_v_table(&mut v, &r, j);
        j += 1;
    }

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "cache ra_i");
    let _guard = span.enter();

    let ra_i: Vec<F> = lookup_indices
        .par_iter()
        .map(|k| {
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
        let span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _guard = span.enter();

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

        drop(_guard);
        drop(span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        let span = tracing::span!(tracing::Level::INFO, "Binding");
        let _guard = span.enter();

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

    impl<F: JoltField> SparseDenseSumcheck<F> for MULHUInstruction<8> {
        const GAMMA: usize = 0;
    }

    impl<F: JoltField> SparseDenseSumcheck<F> for ADDInstruction<8> {
        const GAMMA: usize = 0;
    }

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
