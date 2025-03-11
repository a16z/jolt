use super::sumcheck::SumcheckInstanceProof;
use crate::{
    field::JoltField,
    jolt::instruction::{and::ANDInstruction, mulhu::MULHUInstruction, JoltInstruction},
    poly::{
        dense_mlpoly::DensePolynomial,
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
use std::ops::Index;

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

    #[tracing::instrument(skip_all, name = "ExpandingTable::update")]
    fn update(&mut self, r_j: F) {
        self.values[..self.len]
            .par_iter()
            .zip(self.scratch_space.par_chunks_mut(2))
            .for_each(|(&v_i, dest)| {
                let eval_1 = r_j * v_i;
                dest[0] = v_i - eval_1;
                dest[1] = eval_1;
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

impl<'data, F: JoltField> ParallelSlice<F> for &'data ExpandingTable<F> {
    fn as_parallel_slice(&self) -> &[F] {
        self.values[..self.len].as_parallel_slice()
    }
}

pub trait SparseDenseSumcheckAlt<F: JoltField>: JoltInstruction + Default {
    const NUM_PREFIXES: usize;
    const NUM_SUFFIXES: usize;

    fn combine(prefixes: &[F], suffixes: &[F]) -> F;
    fn update_prefix_checkpoints(checkpoints: &mut [Option<F>], r_x: F, r_y: F, j: usize);
    fn prefix_mle(
        l: usize,
        checkpoints: &[Option<F>],
        r_x: Option<F>,
        c: u32,
        b: u32,
        b_len: usize,
        j: usize,
    ) -> F;
    fn suffix_mle(l: usize, b: u64, b_len: usize) -> u32;

    fn compute_prover_message(
        prefix_checkpoints: &[Option<F>],
        suffix_polys: &[DensePolynomial<F>],
        r: &[F],
        j: usize,
    ) -> [F; 2] {
        debug_assert_eq!(prefix_checkpoints.len(), Self::NUM_PREFIXES);
        debug_assert_eq!(suffix_polys.len(), Self::NUM_SUFFIXES);

        let len = suffix_polys[0].len();
        let log_len = len.log_2();

        let r_x = if j % 2 == 1 { r.last().copied() } else { None };

        let (eval_0, eval_2_left, eval_2_right): (F, F, F) = (0..len / 2)
            .into_par_iter()
            .map(|b| {
                // TODO(moodlezoup): Avoid allocations
                let prefixes_c0: Vec<_> = (0..Self::NUM_PREFIXES)
                    .map(|l| {
                        Self::prefix_mle(l, prefix_checkpoints, r_x, 0, b as u32, log_len - 1, j)
                    })
                    .collect();
                let prefixes_c2: Vec<_> = (0..Self::NUM_PREFIXES)
                    .map(|l| {
                        Self::prefix_mle(l, prefix_checkpoints, r_x, 2, b as u32, log_len - 1, j)
                    })
                    .collect();
                let suffixes_left: Vec<_> = (0..Self::NUM_SUFFIXES)
                    .map(|l| suffix_polys[l][b])
                    .collect();
                let suffixes_right: Vec<_> = (0..Self::NUM_SUFFIXES)
                    .map(|l| suffix_polys[l][b + len / 2])
                    .collect();
                (
                    Self::combine(&prefixes_c0, &suffixes_left),
                    Self::combine(&prefixes_c2, &suffixes_left),
                    Self::combine(&prefixes_c2, &suffixes_right),
                )
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |running, new| (running.0 + new.0, running.1 + new.1, running.2 + new.2),
            );

        [eval_0, eval_2_right + eval_2_right - eval_2_left]
    }
}

pub fn prove_single_instruction_alt<
    const LOG_K: usize,
    F: JoltField,
    I: SparseDenseSumcheckAlt<F>,
    ProofTranscript: Transcript,
>(
    instructions: &[I],
    r_cycle: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, F, [F; 4]) {
    debug_assert!(LOG_K.is_power_of_two());
    let log_m = LOG_K / 4;
    let m = log_m.pow2();

    let T = instructions.len();
    let log_T = T.log_2();
    debug_assert_eq!(r_cycle.len(), log_T);

    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let leaves_per_chunk = (m / num_chunks).max(1);

    let num_rounds = LOG_K + log_T;
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

    let (eq_r_prime, mut u_evals) = rayon::join(
        || EqPolynomial::evals(&r_cycle),
        || EqPolynomial::evals_with_r2(&r_cycle),
    );

    let mut prefix_checkpoints: Vec<Option<F>> = vec![None; I::NUM_PREFIXES];
    let mut v = ExpandingTable::new(m);

    let span = tracing::span!(tracing::Level::INFO, "compute rv_claim");
    let _guard = span.enter();
    let rv_claim = lookup_indices
        .par_iter()
        .zip(u_evals.par_iter())
        .map(|(k, u)| u.mul_u64_unchecked(I::default().materialize_entry(*k)))
        .sum();
    drop(_guard);
    drop(span);

    let mut previous_claim = rv_claim;

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

    let mut suffix_polys: Vec<DensePolynomial<F>> = (0..I::NUM_SUFFIXES)
        .into_par_iter()
        .map(|_| DensePolynomial::new(unsafe_allocate_zero_vec(m)))
        .collect();

    for phase in 0..3 {
        let span = tracing::span!(tracing::Level::INFO, "sparse-dense phase");
        let _guard = span.enter();

        // Condensation
        if phase != 0 {
            let span = tracing::span!(tracing::Level::INFO, "Update u_evals");
            let _guard = span.enter();
            lookup_indices
                .par_iter()
                .zip(u_evals.par_iter_mut())
                .for_each(|(k, u)| {
                    let k_bound = ((*k >> ((4 - phase) * log_m)) % m as u64) as usize;
                    *u *= v[k_bound as usize];
                });
            drop(_guard);
            drop(span);
        }

        let suffix_len = (3 - phase) * log_m;

        // Build binary trees Q_\ell for each \ell = 1, ..., \kappa
        let span = tracing::span!(tracing::Level::INFO, "compute instruction_index_iters");
        let _guard = span.enter();
        let instruction_index_iters: Vec<_> = (0..num_chunks)
            .into_par_iter()
            .map(|i| {
                lookup_indices.iter().enumerate().filter_map(move |(j, k)| {
                    let group = ((k >> suffix_len) % m as u64) / leaves_per_chunk as u64;
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

        let span = tracing::span!(tracing::Level::INFO, "Compute suffix polys");
        let _guard = span.enter();
        suffix_polys
            .par_iter_mut()
            .enumerate()
            .for_each(|(l, poly)| {
                if phase != 0 {
                    poly.len = m;
                    poly.num_vars = poly.len.log_2();
                    poly.Z.par_iter_mut().for_each(|eval| *eval = F::zero());
                }
                instruction_index_iters
                    .par_iter()
                    .zip(poly.Z.par_chunks_mut(leaves_per_chunk))
                    .for_each(|(j_iter, evals)| {
                        j_iter.clone().for_each(|j| {
                            let k = lookup_indices[j];
                            let u = u_evals[j];
                            let suffix_bits = k % (1 << suffix_len);
                            let t = I::suffix_mle(l, suffix_bits, suffix_len);
                            let index = ((k >> suffix_len) % leaves_per_chunk as u64) as usize;
                            evals[index] += u.mul_u64_unchecked(t as u64);
                        });
                    });
            });

        drop(_guard);
        drop(span);

        v.reset(F::one());

        for _round in 0..log_m {
            let span = tracing::span!(tracing::Level::INFO, "sparse-dense sumcheck round");
            let _guard = span.enter();

            let univariate_poly_evals =
                I::compute_prover_message(&prefix_checkpoints, &suffix_polys, &r, j);

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

            suffix_polys
                .par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
            v.update(r_j);

            {
                if r.len() % 2 == 0 {
                    let span = tracing::span!(tracing::Level::INFO, "Update prefix checkpoints");
                    let _guard = span.enter();
                    I::update_prefix_checkpoints(
                        &mut prefix_checkpoints,
                        r[r.len() - 2],
                        r[r.len() - 1],
                        j,
                    )
                }
            }

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
                let k_bound = ((k >> suffix_len) % m as u64) as usize;
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
                let group = (k % m as u64) / leaves_per_chunk as u64;
                if group == i as u64 {
                    Some(j)
                } else {
                    None
                }
            })
        })
        .collect();

    let mut eq_ra: Vec<F> = unsafe_allocate_zero_vec(m);
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
        for i in 0..m {
            assert_eq!(eq_ra.get_bound_coeff(i), eq_ra_test.get_bound_coeff(i));
        }
    }

    let span = tracing::span!(tracing::Level::INFO, "Materialize val");
    let _guard = span.enter();
    let prefixes: Vec<_> = prefix_checkpoints
        .iter()
        .map(|checkpoint| checkpoint.unwrap())
        .collect();
    let val: Vec<F> = (0..m)
        .into_par_iter()
        .map(|k| {
            let suffixes: Vec<_> = (0..I::NUM_SUFFIXES)
                .map(|l| F::from_u32(I::suffix_mle(l, k as u64, log_m)))
                .collect();
            I::combine(&prefixes, &suffixes)
        })
        .collect();
    let mut val = MultilinearPolynomial::from(val);
    drop(_guard);
    drop(span);

    #[cfg(test)]
    {
        for i in 0..m {
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

        v.update(r_j);
        j += 1;
    }

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "cache ra_i");
    let _guard = span.enter();

    let ra_i: Vec<F> = lookup_indices
        .par_iter()
        .map(|k| {
            let k_bound = (k % m as u64) as usize;
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
            and::ANDInstruction, mulhu::MULHUInstruction, or::ORInstruction, sltu::SLTUInstruction,
        },
        utils::{transcript::KeccakTranscript, uninterleave_bits},
    };
    use ark_bn254::Fr;
    use rand::{rngs::StdRng, SeedableRng};

    // impl<F: JoltField> SparseDenseSumcheck<F> for MULHUInstruction<8> {}
    // impl<F: JoltField> SparseDenseSumcheck<F> for ADDInstruction<8> {}
    // impl<F: JoltField> SparseDenseSumcheck<F> for ANDInstruction<8> {}

    const WORD_SIZE: usize = 8;
    const K: usize = 1 << 16;
    const LOG_K: usize = 16;
    const T: usize = 1 << 8;

    // #[test]
    // fn test_mulhu() {
    //     let mut rng = StdRng::seed_from_u64(12345);

    //     let instructions: Vec<_> = (0..T)
    //         .map(|_| MULHUInstruction::<WORD_SIZE>::default().random(&mut rng))
    //         .collect();

    //     let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
    //     let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

    //     let (proof, rv_claim, ra_claims) = prove_single_instruction::<TREE_WIDTH, _, _, _>(
    //         &instructions,
    //         r_cycle,
    //         &mut prover_transcript,
    //     );

    //     let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
    //     verifier_transcript.compare_to(prover_transcript);
    //     let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());
    //     let verification_result = verify_single_instruction::<_, MULHUInstruction<WORD_SIZE>, _>(
    //         proof,
    //         K,
    //         T,
    //         r_cycle,
    //         rv_claim,
    //         ra_claims,
    //         &mut verifier_transcript,
    //     );
    //     assert!(
    //         verification_result.is_ok(),
    //         "Verification failed with error: {:?}",
    //         verification_result.err()
    //     );
    // }

    // #[test]
    // fn test_add() {
    //     let mut rng = StdRng::seed_from_u64(12345);

    //     let instructions: Vec<_> = (0..T)
    //         .map(|_| ADDInstruction::<WORD_SIZE>::default().random(&mut rng))
    //         .collect();

    //     let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
    //     let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

    //     let (proof, rv_claim, ra_claims) = prove_single_instruction::<TREE_WIDTH, _, _, _>(
    //         &instructions,
    //         r_cycle,
    //         &mut prover_transcript,
    //     );

    //     let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
    //     verifier_transcript.compare_to(prover_transcript);
    //     let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());
    //     let verification_result = verify_single_instruction::<_, ADDInstruction<WORD_SIZE>, _>(
    //         proof,
    //         K,
    //         T,
    //         r_cycle,
    //         rv_claim,
    //         ra_claims,
    //         &mut verifier_transcript,
    //     );
    //     assert!(
    //         verification_result.is_ok(),
    //         "Verification failed with error: {:?}",
    //         verification_result.err()
    //     );
    // }

    // #[test]
    // fn test_and() {
    //     let mut rng = StdRng::seed_from_u64(12345);

    //     let instructions: Vec<_> = (0..T)
    //         .map(|_| ANDInstruction::<WORD_SIZE>::default().random(&mut rng))
    //         .collect();

    //     let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
    //     let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

    //     let (proof, rv_claim, ra_claims) = prove_single_instruction::<TREE_WIDTH, _, _, _>(
    //         &instructions,
    //         r_cycle,
    //         &mut prover_transcript,
    //     );

    //     let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
    //     verifier_transcript.compare_to(prover_transcript);
    //     let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());
    //     let verification_result = verify_single_instruction::<_, ANDInstruction<WORD_SIZE>, _>(
    //         proof,
    //         K,
    //         T,
    //         r_cycle,
    //         rv_claim,
    //         ra_claims,
    //         &mut verifier_transcript,
    //     );
    //     assert!(
    //         verification_result.is_ok(),
    //         "Verification failed with error: {:?}",
    //         verification_result.err()
    //     );
    // }

    #[test]
    fn test_or() {
        let mut rng = StdRng::seed_from_u64(12345);

        let instructions: Vec<_> = (0..T)
            .map(|_| ORInstruction::<WORD_SIZE>::default().random(&mut rng))
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let (proof, rv_claim, ra_claims) = prove_single_instruction_alt::<LOG_K, _, _, _>(
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

    #[test]
    fn test_sltu() {
        let mut rng = StdRng::seed_from_u64(12345);

        let instructions: Vec<_> = (0..T)
            .map(|_| SLTUInstruction::<WORD_SIZE>::default().random(&mut rng))
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let (proof, rv_claim, ra_claims) = prove_single_instruction_alt::<LOG_K, _, _, _>(
            &instructions,
            r_cycle,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());
        let verification_result = verify_single_instruction::<_, SLTUInstruction<WORD_SIZE>, _>(
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
