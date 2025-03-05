use super::sumcheck::SumcheckInstanceProof;
use crate::{
    field::JoltField,
    jolt::instruction::{and::ANDInstruction, mulhu::MULHUInstruction, JoltInstruction},
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

#[derive(Clone)]
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

impl<'data, F: JoltField> ParallelSlice<F> for &'data ExpandingTable<F> {
    fn as_parallel_slice(&self) -> &[F] {
        self.values[..self.len].as_parallel_slice()
    }
}

#[derive(Clone)]
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
    // const ETA: usize

    #[tracing::instrument(skip_all, name = "SparseDenseSumcheck::compute_prover_message")]
    fn compute_prover_message(
        Q_tree: &[BinarySumTree<F>],
        Z_tree: &BinarySumTree<F>,
        r: &[F],
        j: usize,
        v: &ExpandingTable<F>,
        x: &[ExpandingTable<F>],
        w: &[ExpandingTable<F>],
    ) -> [F; 2] {
        let eta = Self::default().eta();
        let two = F::from_u8(2);
        let chi_2 = [F::one() - two, two];

        let mut univariate_poly_evals = [F::zero(), F::zero()];

        let v_q_x: Vec<[F; 4]> = Self::compute_q_inner_products(Q_tree, j, v, x);
        let (v_z, v_z_w): ([F; 4], Vec<[F; 4]>) = Self::compute_z_inner_products(Z_tree, j, v, w);

        if j % 2 == 0 {
            for (b_x, b_y) in [(0, 0), (0, 1)] {
                let two_bit_index = 2 * b_x + b_y as usize;
                for l in 0..eta {
                    // Expression (52), c = 0
                    univariate_poly_evals[0] += Self::default().multiplicative_update(
                        l,
                        j,
                        F::zero(),
                        b_x as u8,
                        None,
                        Some(b_y),
                    ) * v_q_x[l][two_bit_index];
                    // Expression (53), c = 0
                    univariate_poly_evals[0] += v_z_w[l][two_bit_index]
                        + Self::default().additive_update(
                            l,
                            j,
                            F::zero(),
                            b_x as u8,
                            None,
                            Some(b_y),
                        ) * v_z[two_bit_index];
                }
            }

            for (b_x, b_y) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let two_bit_index = 2 * b_x + b_y as usize;
                for l in 0..eta {
                    // Expression (52), c = 2
                    univariate_poly_evals[1] += chi_2[b_x]
                        * Self::default().multiplicative_update(
                            l,
                            j,
                            two,
                            b_x as u8,
                            None,
                            Some(b_y),
                        )
                        * v_q_x[l][two_bit_index];
                    // Expression (53), c = 2
                    univariate_poly_evals[1] += chi_2[b_x]
                        * (v_z_w[l][two_bit_index]
                            + Self::default().additive_update(
                                l,
                                j,
                                two,
                                b_x as u8,
                                None,
                                Some(b_y),
                            ) * v_z[two_bit_index]);
                }
            }
        } else {
            let r_prev = Some(r[j - 1]);
            for (b_x, b_y) in [(0, 0), (1, 0)] {
                let two_bit_index = 2 * b_x + b_y as usize;
                for l in 0..eta {
                    // Expression (52), c = 0
                    univariate_poly_evals[0] += Self::default().multiplicative_update(
                        l,
                        j,
                        F::zero(),
                        b_y as u8,
                        r_prev,
                        None,
                    ) * v_q_x[l][two_bit_index];
                    // Expression (53), c = 0
                    univariate_poly_evals[0] += v_z_w[l][two_bit_index]
                        + Self::default().additive_update(l, j, F::zero(), b_y as u8, r_prev, None)
                            * v_z[two_bit_index];
                }
            }

            for (b_x, b_y) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let two_bit_index = 2 * b_x + b_y as usize;
                for l in 0..eta {
                    // Expression (52), c = 2
                    univariate_poly_evals[1] += chi_2[b_y]
                        * Self::default().multiplicative_update(l, j, two, b_y as u8, r_prev, None)
                        * v_q_x[l][two_bit_index];
                    // Expression (53), c = 2
                    univariate_poly_evals[1] += chi_2[b_y]
                        * (v_z_w[l][two_bit_index]
                            + Self::default().additive_update(l, j, two, b_y as u8, r_prev, None)
                                * v_z[two_bit_index]);
                }
            }
        }

        univariate_poly_evals
    }

    #[tracing::instrument(skip_all, name = "SparseDenseSumcheck::compute_q_inner_products")]
    fn compute_q_inner_products(
        Q_trees: &[BinarySumTree<F>],
        j: usize,
        v: &ExpandingTable<F>,
        x: &[ExpandingTable<F>],
    ) -> Vec<[F; 4]> {
        let log_m = Q_trees[0].layers.len() - 1;
        Q_trees
            .par_iter()
            .zip_eq(x.par_iter())
            .map(|(Q_l, x_l)| {
                if j % 2 == 0 {
                    let layer = &Q_l[(j % log_m) + 2];
                    v.par_iter()
                        .zip_eq(x_l.par_iter())
                        .zip_eq(layer.par_chunks(4))
                        .map(|((v_b, x_b), q_b)| {
                            let v_x = *v_b * x_b;
                            [v_x * q_b[0], v_x * q_b[1], v_x * q_b[2], v_x * q_b[3]]
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
                        )
                } else {
                    let layer = &Q_l[(j % log_m) + 1];
                    v.par_chunks(2)
                        .zip_eq(x_l.par_chunks(4))
                        .zip_eq(layer.par_chunks(4))
                        .map(|((v_b, x_b), q_b)| {
                            [
                                v_b[0] * x_b[0] * q_b[0],
                                v_b[0] * x_b[1] * q_b[1],
                                v_b[1] * x_b[2] * q_b[2],
                                v_b[1] * x_b[3] * q_b[3],
                            ]
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
                        )
                }
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "SparseDenseSumcheck::compute_z_inner_products")]
    fn compute_z_inner_products(
        Z_tree: &BinarySumTree<F>,
        j: usize,
        v: &ExpandingTable<F>,
        w: &[ExpandingTable<F>],
    ) -> ([F; 4], Vec<[F; 4]>) {
        let log_m = Z_tree.layers.len() - 1;
        if j % 2 == 0 {
            let layer = &Z_tree[(j % log_m) + 2];
            v.par_iter()
                .zip_eq(layer.par_chunks(4))
                .enumerate()
                .map(|(b, (v_b, z_b))| {
                    let v_z = [z_b[0] * v_b, z_b[1] * v_b, z_b[2] * v_b, z_b[3] * v_b];
                    let v_z_w = w
                        .iter()
                        .map(|w_l| {
                            [
                                v_z[0] * w_l[b],
                                v_z[1] * w_l[b],
                                v_z[2] * w_l[b],
                                v_z[3] * w_l[b],
                            ]
                        })
                        .collect();
                    (v_z, v_z_w)
                })
                .reduce(
                    || ([F::zero(); 4], vec![[F::zero(); 4]; w.len()]),
                    |running, new| {
                        (
                            [
                                running.0[0] + new.0[0],
                                running.0[1] + new.0[1],
                                running.0[2] + new.0[2],
                                running.0[3] + new.0[3],
                            ],
                            running
                                .1
                                .iter()
                                .zip(new.1.iter())
                                .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
                                .collect(),
                        )
                    },
                )
        } else {
            let layer = &Z_tree[(j % log_m) + 1];
            v.par_chunks(2)
                .zip_eq(layer.par_chunks(4))
                .enumerate()
                .map(|(b, (v_b, z_b))| {
                    let v_z = [
                        v_b[0] * z_b[0],
                        v_b[0] * z_b[1],
                        v_b[1] * z_b[2],
                        v_b[1] * z_b[3],
                    ];
                    let v_z_w = w
                        .iter()
                        .map(|w_l| {
                            [
                                v_z[0] * w_l[4 * b],
                                v_z[1] * w_l[4 * b + 1],
                                v_z[2] * w_l[4 * b + 2],
                                v_z[3] * w_l[4 * b + 3],
                            ]
                        })
                        .collect();
                    (v_z, v_z_w)
                })
                .reduce(
                    || ([F::zero(); 4], vec![[F::zero(); 4]; w.len()]),
                    |running, new| {
                        (
                            [
                                running.0[0] + new.0[0],
                                running.0[1] + new.0[1],
                                running.0[2] + new.0[2],
                                running.0[3] + new.0[3],
                            ],
                            running
                                .1
                                .iter()
                                .zip(new.1.iter())
                                .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
                                .collect(),
                        )
                    },
                )
        }
    }

    #[tracing::instrument(skip_all, name = "SparseDenseSumcheck::update_tables")]
    fn update_tables(
        v: &mut ExpandingTable<F>,
        x: &mut [ExpandingTable<F>],
        w: &mut [ExpandingTable<F>],
        r: &[F],
        j: usize,
    ) {
        Self::update_v_table(v, r, j);

        if j % 2 == 0 {
            x.par_iter_mut().enumerate().for_each(|(l, x_l)| {
                x_l.values[..x_l.len]
                    .par_iter()
                    .zip(x_l.scratch_space.par_chunks_mut(4))
                    .for_each(|(&x_val, dest)| {
                        for (b_j, b_next) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                            dest[2 * b_j + b_next] = x_val
                                * Self::default().multiplicative_update(
                                    l,
                                    j,
                                    r[j],
                                    b_j as u8,
                                    None,
                                    Some(b_next as u8),
                                );
                        }
                    });
                std::mem::swap(&mut x_l.values, &mut x_l.scratch_space);
                x_l.len *= 4;
            });

            w.par_iter_mut().enumerate().for_each(|(l, w_l)| {
                w_l.values[..w_l.len]
                    .par_iter()
                    .zip(w_l.scratch_space.par_chunks_mut(4))
                    .for_each(|(&w_val, dest)| {
                        for (b_j, b_next) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                            dest[2 * b_j + b_next] = w_val
                                + Self::default().additive_update(
                                    l,
                                    j,
                                    r[j],
                                    b_j as u8,
                                    None,
                                    Some(b_next as u8),
                                );
                        }
                    });
                std::mem::swap(&mut w_l.values, &mut w_l.scratch_space);
                w_l.len *= 4;
            });
        } else {
            x.par_iter_mut().enumerate().for_each(|(l, x_l)| {
                x_l.values[..x_l.len]
                    .par_iter()
                    .zip(x_l.scratch_space.par_iter_mut())
                    .enumerate()
                    .for_each(|(index, (&x_val, dest))| {
                        *dest = x_val
                            * Self::default().multiplicative_update(
                                l,
                                j,
                                r[j],
                                (index % 2) as u8,
                                Some(r[j - 1]),
                                None,
                            );
                    });
                std::mem::swap(&mut x_l.values, &mut x_l.scratch_space);
            });

            w.par_iter_mut().enumerate().for_each(|(l, w_l)| {
                w_l.values[..w_l.len]
                    .par_iter()
                    .zip(w_l.scratch_space.par_iter_mut())
                    .enumerate()
                    .for_each(|(index, (&w_val, dest))| {
                        *dest = w_val
                            + Self::default().additive_update(
                                l,
                                j,
                                r[j],
                                (index % 2) as u8,
                                Some(r[j - 1]),
                                None,
                            );
                    });
                std::mem::swap(&mut w_l.values, &mut w_l.scratch_space);
            });
        }
    }

    #[tracing::instrument(skip_all, name = "SparseDenseSumcheck::update_v_table")]
    fn update_v_table(v: &mut ExpandingTable<F>, r: &[F], j: usize) {
        v.values[..v.len]
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

impl<F: JoltField> SparseDenseSumcheck<F> for MULHUInstruction<32> {}
impl<F: JoltField> SparseDenseSumcheck<F> for ANDInstruction<32> {}

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

    let eta = I::default().eta();
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

    let mut t_evals: Vec<HashMap<u64, F>> = vec![HashMap::with_capacity(T + TREE_WIDTH); eta];
    let mut t_evals_vec: Vec<Vec<usize>> = vec![Vec::with_capacity(T); eta];

    let mut rv_claim = F::zero();
    let mut previous_claim = F::zero();

    let mut v = ExpandingTable::new(TREE_WIDTH);
    let mut w = vec![ExpandingTable::new(TREE_WIDTH); eta];
    let mut x = vec![ExpandingTable::new(TREE_WIDTH); eta];

    #[cfg(test)]
    let mut val_test: MultilinearPolynomial<F> =
        MultilinearPolynomial::from(I::default().materialize());
    #[cfg(test)]
    let mut eq_ra_test: MultilinearPolynomial<F> = {
        let mut eq_ra: Vec<F> = unsafe_allocate_zero_vec(val_test.len());
        for (j, k) in lookup_indices.iter().enumerate() {
            eq_ra[*k as usize] += eq_r_prime[j];
        }
        MultilinearPolynomial::from(eq_ra)
    };

    let mut j: usize = 0;
    let mut ra: Vec<MultilinearPolynomial<F>> = Vec::with_capacity(4);

    let (mut Z_tree, mut Q_tree) = rayon::join(
        || BinarySumTree::new(log_m + 1),
        || vec![BinarySumTree::new(log_m + 1); eta],
    );

    for phase in 0..3 {
        let span = tracing::span!(tracing::Level::INFO, "sparse-dense phase");
        let _guard = span.enter();

        // Condensation
        if phase == 0 {
            let span = tracing::span!(tracing::Level::INFO, "compute initial t evals");
            let _guard = span.enter();
            t_evals.par_iter_mut().enumerate().for_each(|(l, t_l)| {
                t_l.par_extend(
                    (0..(TREE_WIDTH as u64))
                        .into_par_iter()
                        .map(|k| (k, F::from_u64(I::default().subtable_entry(l, k))))
                        .chain(
                            lookup_indices
                                .par_iter()
                                .map(|k| (*k, F::from_u64(I::default().subtable_entry(l, *k)))),
                        ),
                )
            });
            t_evals_vec = t_evals
                .par_iter()
                .map(|t_evals| {
                    lookup_indices
                        .par_iter()
                        .map(|k| (t_evals.get(k).unwrap() as *const F) as usize)
                        .collect()
                })
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
            t_evals.par_iter_mut().enumerate().for_each(|(l, t_l)| {
                t_l.par_iter_mut().for_each(|(k, t)| {
                    let k_bound = ((k >> ((4 - phase) * log_m)) % TREE_WIDTH as u64) as usize;
                    *t *= x[l][k_bound];
                    *t += w[l][k_bound];
                });
            });
        }

        let k_shift = (3 - phase) * log_m;

        // Build binary trees Q_\ell and Z for each \ell = 1, ..., \kappa
        let span = tracing::span!(tracing::Level::INFO, "compute instruction_index_iters");
        let _guard = span.enter();
        let instruction_index_iters: Vec<_> = (0..num_chunks)
            .into_par_iter()
            .map(|i| {
                lookup_indices.iter().enumerate().filter_map(move |(j, k)| {
                    let group = ((k >> k_shift) % TREE_WIDTH as u64) / leaves_per_chunk as u64;
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
        if phase != 0 {
            z_leaves.par_iter_mut().for_each(|leaf| *leaf = F::zero());
            Q_tree.par_iter_mut().for_each(|tree| {
                tree.leaves_mut()
                    .par_iter_mut()
                    .for_each(|leaf| *leaf = F::zero())
            });
        }

        Q_tree.par_iter_mut().enumerate().for_each(|(l, tree)| {
            instruction_index_iters
                .par_iter()
                .zip(tree.leaves_mut().par_chunks_mut(leaves_per_chunk))
                .for_each(|(j_iter, q_leaves)| {
                    j_iter.clone().for_each(|j| {
                        let k = lookup_indices[j];
                        let u = u_evals[j];
                        // let t = t_evals.get(&k).unwrap();
                        let t = unsafe { *(t_evals_vec[l][j] as *const F) };
                        let leaf_index = ((k >> k_shift) % leaves_per_chunk as u64) as usize;
                        q_leaves[leaf_index] += u * t;
                    });
                });
        });

        instruction_index_iters
            .into_par_iter()
            .zip(z_leaves.par_chunks_mut(leaves_per_chunk))
            .for_each(|(j_iter, z_leaves)| {
                j_iter.for_each(|j| {
                    let k = lookup_indices[j];
                    let u = u_evals[j];
                    let leaf_index = ((k >> k_shift) % leaves_per_chunk as u64) as usize;
                    z_leaves[leaf_index] += u;
                });
            });

        drop(_guard);
        drop(span);

        Q_tree.par_iter_mut().for_each(|tree| tree.build());
        Z_tree.build();

        v.reset(F::one());
        w.iter_mut().for_each(|w_l| w_l.reset(F::zero()));
        x.iter_mut().for_each(|x_l| x_l.reset(F::one()));

        previous_claim = Q_tree.iter().map(|tree| tree[0][0]).sum();
        if phase == 0 {
            rv_claim = previous_claim;
        }

        for _round in 0..log_m {
            let span = tracing::span!(tracing::Level::INFO, "sparse-dense sumcheck round");
            let _guard = span.enter();

            let univariate_poly_evals =
                I::compute_prover_message(&Q_tree, &Z_tree, &r, j, &v, &x, &w);

            #[cfg(test)]
            {
                let expected: [F; 2] = (0..val_test.len() / 2)
                    .into_par_iter()
                    .map(|i| {
                        let eq_ra_evals = eq_ra_test.sumcheck_evals(i, 2, BindingOrder::HighToLow);
                        let val_evals = val_test.sumcheck_evals(i, 2, BindingOrder::HighToLow);

                        [eq_ra_evals[0] * val_evals[0], eq_ra_evals[1] * val_evals[1]]
                    })
                    .reduce(
                        || [F::zero(); 2],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    );
                assert_eq!(
                    expected, univariate_poly_evals,
                    "Sumcheck sanity check failed in phase {phase} round {_round}"
                );
            }

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
                let k_bound = ((k >> k_shift) % TREE_WIDTH as u64) as usize;
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
        .map(|k| {
            (0..eta)
                .map(|l| x[l][0] * t_evals[l].get(&(k as u64)).unwrap() + w[l][0])
                .sum()
        })
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
        jolt::instruction::{
            add::ADDInstruction, and::ANDInstruction, mulhu::MULHUInstruction, or::ORInstruction,
        },
        utils::transcript::KeccakTranscript,
    };
    use ark_bn254::Fr;
    use rand::{rngs::StdRng, SeedableRng};

    impl<F: JoltField> SparseDenseSumcheck<F> for MULHUInstruction<8> {}
    impl<F: JoltField> SparseDenseSumcheck<F> for ADDInstruction<8> {}
    impl<F: JoltField> SparseDenseSumcheck<F> for ANDInstruction<8> {}
    impl<F: JoltField> SparseDenseSumcheck<F> for ORInstruction<8> {}

    const WORD_SIZE: usize = 8;
    const K: usize = 1 << 16;
    const T: usize = 1 << 8;
    const TREE_WIDTH: usize = 1 << 4;

    #[test]
    fn test_mulhu() {
        let mut rng = StdRng::seed_from_u64(12345);

        let instructions: Vec<_> = (0..T)
            .map(|_| MULHUInstruction::<WORD_SIZE>::default().random(&mut rng))
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

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
        let mut rng = StdRng::seed_from_u64(12345);

        let instructions: Vec<_> = (0..T)
            .map(|_| ADDInstruction::<WORD_SIZE>::default().random(&mut rng))
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

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

    #[test]
    fn test_and() {
        let mut rng = StdRng::seed_from_u64(12345);

        let instructions: Vec<_> = (0..T)
            .map(|_| ANDInstruction::<WORD_SIZE>::default().random(&mut rng))
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let (proof, rv_claim, ra_claims) = prove_single_instruction::<TREE_WIDTH, _, _, _>(
            &instructions,
            r_cycle,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());
        let verification_result = verify_single_instruction::<_, ANDInstruction<WORD_SIZE>, _>(
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
    fn test_or() {
        let mut rng = StdRng::seed_from_u64(12345);

        let instructions: Vec<_> = (0..T)
            .map(|_| ORInstruction::<WORD_SIZE>::default().random(&mut rng))
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let (proof, rv_claim, ra_claims) = prove_single_instruction::<TREE_WIDTH, _, _, _>(
            &instructions,
            r_cycle,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());
        let verification_result = verify_single_instruction::<_, ORInstruction<WORD_SIZE>, _>(
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
