use std::iter;

use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    field::{JoltField, OptimizedMul},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{
        errors::ProofVerifyError,
        math::Math,
        transcript::{AppendToTranscript, Transcript},
    },
};

#[inline]
pub fn compute_initial_eval_claim<F: JoltField>(
    mle_vec: &Vec<&MultilinearPolynomial<F>>,
    r_cycle: &Vec<F>,
) -> F {
    let eq = MultilinearPolynomial::from(EqPolynomial::evals(&r_cycle));
    (0..r_cycle.len().pow2())
        .into_par_iter()
        .map(|j| {
            mle_vec
                .iter()
                .map(|poly| poly.get_bound_coeff(j))
                .product::<F>()
                * eq.get_bound_coeff(j)
        })
        .reduce(|| F::zero(), |running, new| running + new)
}

/// Contains proof for a generic sumcheck of the form
/// val = \sum_{j' \in \{0, 1\}^T} eq(j, j') \prod_{i=1}^d func(j),
/// which is the un-optimized form of the sumcheck in Appendix C of the Twist + Shout paper.
pub struct NaiveSumCheckProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    eq_claim: F,
    mle_claims: Vec<F>,
}

impl<F: JoltField, ProofTranscript: Transcript> NaiveSumCheckProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "NaiveSumCheckProof::prove")]
    pub fn prove(
        mle_vec: &mut Vec<&mut MultilinearPolynomial<F>>,
        r_cycle: &Vec<F>,
        previous_claim: &mut F,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>) {
        let span = tracing::span!(tracing::Level::INFO, "Initialize eq");
        let _guard = span.enter();
        let mut eq = MultilinearPolynomial::from(EqPolynomial::evals(&r_cycle));
        let log_T = r_cycle.len();
        let mut r: Vec<F> = Vec::with_capacity(r_cycle.len());
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(r_cycle.len());
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Loop over rounds");
        let _guard = span.enter();

        for round in 0..r_cycle.len() {
            let inner_span = tracing::span!(tracing::Level::INFO, "Compute evals");
            let _guard = inner_span.enter();

            let mut evals = (0..(log_T - round - 1).pow2())
                .into_par_iter()
                .map(|j| {
                    let res = eq.sumcheck_evals(j, mle_vec.len() + 1, BindingOrder::HighToLow);
                    let mle_evals = mle_vec
                        .iter()
                        .map(|poly| {
                            poly.sumcheck_evals(j, mle_vec.len() + 1, BindingOrder::HighToLow)
                        })
                        .collect::<Vec<_>>();

                    mle_evals
                        .into_iter()
                        .chain(iter::once(res))
                        .reduce(|running, new| {
                            running
                                .iter()
                                .zip(new.iter())
                                .map(|(a, b)| *a * b)
                                .collect::<Vec<_>>()
                        })
                        .unwrap()
                })
                .reduce(
                    || vec![F::zero(); mle_vec.len() + 1],
                    |running, new| {
                        running
                            .iter()
                            .zip(new.iter())
                            .map(|(a, b)| *a + b)
                            .collect::<Vec<_>>()
                    },
                );

            evals.insert(1, *previous_claim - evals[0]);
            assert_eq!(evals.len(), mle_vec.len() + 2);

            drop(_guard);
            drop(inner_span);

            let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
            let _guard = inner_span.enter();

            let univariate_poly = UniPoly::from_evals(&evals);
            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            *previous_claim = univariate_poly.evaluate(&r_j);
            r.push(r_j);

            drop(_guard);
            drop(inner_span);

            let inner_span = tracing::span!(tracing::Level::INFO, "Bind");
            let _guard = inner_span.enter();

            rayon::join(
                || eq.bind_parallel(r_j, BindingOrder::HighToLow),
                || {
                    mle_vec
                        .par_iter_mut()
                        .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow))
                },
            );

            drop(_guard);
            drop(inner_span);
        }

        drop(_guard);
        drop(span);

        (
            Self {
                sumcheck_proof: SumcheckInstanceProof::new(compressed_polys),
                eq_claim: eq.final_sumcheck_claim(),
                mle_claims: mle_vec
                    .iter()
                    .map(|func| func.final_sumcheck_claim())
                    .collect(),
            },
            r,
        )
    }

    pub fn verify(
        &self,
        r_prime: Vec<F>,
        claim: F,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let (_sumcheck_claim, _r_sumcheck) = self.sumcheck_proof.verify(
            claim,
            r_prime.len(),
            self.mle_claims.len() + 1,
            transcript,
        )?;

        assert_eq!(_r_sumcheck, r_prime);
        assert_eq!(
            _sumcheck_claim,
            self.eq_claim * self.mle_claims.iter().product::<F>()
        );

        Ok(())
    }
}

/// Contains the proof for a generic sumcheck of the form
/// val = \sum_{j_1, ..., j_d \in \{0, 1\}^T} eq(j, j_1, ..., j_d) \prod_{i=1}^d func(j_i),
/// where eq(j, j_1, ..., j_d) = 1 if j = j_1 = ... = j_d and 0 otherwise.
///
/// See Appendix C of the Twist + Shout paper for an example with ra virtualization.
pub struct LargeDSumCheckProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    eq_claim: F,
    mle_claims: Vec<F>,
}

impl<F: JoltField, ProofTranscript: Transcript> LargeDSumCheckProof<F, ProofTranscript> {
    // Compute the initial claim for the sumcheck
    // val = \sum_{j_1, ..., j_d \in \{0, 1\}^T} eq(j, j_1, ..., j_d) \prod_{i=1}^d func(j_i)
    //     = \sum_{j' \in \{0, 1\}^T} eq(j, j', ..., j') \prod_{i=1}^d func(j)
    #[tracing::instrument(skip_all, name = "LargeDSumCheckProof::prove")]
    pub fn prove<const D1: usize>(
        mle_vec: &mut Vec<&mut MultilinearPolynomial<F>>,
        r_cycle: &Vec<F>,
        previous_claim: &mut F,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>) {
        let mut C = F::one();
        let mut C_summands = [F::one(), F::one()];
        let T = r_cycle.len().pow2();
        let D = mle_vec.len();

        let span = tracing::span!(tracing::Level::INFO, "Initialize E_table");
        let _guard = span.enter();
        // Each table E_i stores the evaluations of eq(j_{>i}, r_cycle_{>i}) for each j_{>i}.
        // As we're binding from high to low, for each E_i we store eq(j_{<LogT - i}, r_cycle_{<+LogT - i}) instead.
        // TODO: not sure how much saving we get from batch computing this, maybe too small?.
        let E_table = (1..=T.log_2() - 1)
            .map(|i| {
                let evals =
                    EqPolynomial::evals(&r_cycle[i..].iter().map(|x| *x).collect::<Vec<_>>());
                evals
            })
            .collect::<Vec<_>>();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(D * T.log_2());
        let mut w: Vec<F> = Vec::with_capacity(D * T.log_2());
        drop(_guard);
        drop(span);

        assert_eq!(r_cycle.len(), T.log_2());

        let span = tracing::span!(tracing::Level::INFO, "Loop over j_idx");
        let _guard = span.enter();

        for j_idx in 0..T.log_2() {
            let inner_span = tracing::span!(
                tracing::Level::INFO,
                "Create before and after idx evals table"
            );
            let _guard = inner_span.enter();

            let size = (T.log_2() - j_idx - 1).pow2();
            let mut before_idx_evals = vec![F::one(); size];

            drop(_guard);
            drop(inner_span);

            let inner_span = tracing::span!(tracing::Level::INFO, "Compute after idx evals");
            let _guard = inner_span.enter();

            let after_idx_evals = (0..size)
                .into_par_iter()
                .map(|j| {
                    let mut cur = (
                        mle_vec[D - 1].get_bound_coeff(j),
                        mle_vec[D - 1].get_bound_coeff(j + mle_vec[D - 1].len() / 2),
                    );

                    let res: [(F, F); D1] = std::array::from_fn(|i| {
                        let entry = cur;
                        if i < D1 - 1 {
                            cur = (
                                cur.0 * mle_vec[D - 2 - i].get_bound_coeff(j),
                                cur.1
                                    * mle_vec[D - 2 - i]
                                        .get_bound_coeff(j + mle_vec[D - 2 - i].len() / 2),
                            );
                        }

                        entry
                    });

                    res
                })
                .collect::<Vec<_>>();

            drop(_guard);
            drop(inner_span);

            let inner_span = tracing::span!(tracing::Level::INFO, "Loop over d");
            let _guard = inner_span.enter();

            for d in 0..D {
                let _span = tracing::span!(tracing::Level::INFO, "Field operations");
                let _guard = _span.enter();

                let round = j_idx * D + d;

                if d == 0 {
                    if round > 0 {
                        C *= C_summands[0] + C_summands[1];
                    }

                    let r_cycle_val = r_cycle[j_idx];

                    C_summands[0] = r_cycle_val;
                    C_summands[1] = F::one() - r_cycle_val;
                }

                // Evaluate eq(r_round, w_1, ..., w_{idx - 1}, c, b) at c = 0, 2 and b = 0, 1
                let eq_evals_at_idx = [
                    (C_summands[1], F::zero()),
                    (-C_summands[1], C_summands[0] + C_summands[0]),
                ];

                let univariate_poly_evals = before_idx_evals
                    .par_iter_mut()
                    .take(size)
                    .zip(after_idx_evals.par_iter().take(size))
                    .enumerate()
                    .map(|(j, (before_idx_eval, after_idx_evals))| {
                        // let at_idx_evals =
                        // mle_vec[D - d - 1].sumcheck_evals(j, 2, BindingOrder::HighToLow);
                        let at_idx_evals = [
                            mle_vec[D - d - 1].get_bound_coeff(j),
                            mle_vec[D - d - 1].get_bound_coeff(j + mle_vec[D - d - 1].len() / 2)
                                + mle_vec[D - d - 1]
                                    .get_bound_coeff(j + mle_vec[D - d - 1].len() / 2)
                                - mle_vec[D - d - 1].get_bound_coeff(j),
                        ];

                        if d > 0 {
                            *before_idx_eval = before_idx_eval
                                .mul_1_optimized(mle_vec[D - d - 1].get_bound_coeff(j));
                        }

                        let eq_eval_after_idx = if j_idx < T.log_2() - 1 {
                            E_table[j_idx][j]
                        } else {
                            F::one()
                        };

                        let temp =
                            before_idx_eval.mul_1_optimized(eq_eval_after_idx.mul_1_optimized(C));

                        let tmp = if d < D - 1 {
                            eq_evals_at_idx[0].0 * temp * after_idx_evals[D - d - 2].0
                        } else {
                            eq_evals_at_idx[0].0 * temp
                        };

                        eq_evals_at_idx
                            .iter()
                            .zip(at_idx_evals.iter())
                            .map(|((c_eq_eval_0, c_eq_eval_1), at_idx_eval)| {
                                let factor =
                                    *at_idx_eval * *before_idx_eval * eq_eval_after_idx * C;
                                let eval_1 = if d < D - 1 {
                                    *c_eq_eval_1 * factor * after_idx_evals[D - d - 2].1
                                } else {
                                    *c_eq_eval_1 * factor
                                };

                                let eval_0 = if d < D - 1 {
                                    *c_eq_eval_0 * factor * after_idx_evals[D - d - 2].0
                                } else {
                                    *c_eq_eval_0 * factor
                                };

                                eval_0 + eval_1
                            })
                            .collect::<Vec<_>>()

                        // [
                        //     {
                        //         at_idx_evals[0] * tmp

                        //         // let eval_1 = if d < D - 1 {
                        //         //     eq_evals_at_idx[0].1 * factor * after_idx_evals[D - d - 2].1
                        //         // } else {
                        //         //     eq_evals_at_idx[0].1 * factor
                        //         // };
                        //     },
                        //     {
                        //         let factor: F = at_idx_evals[1] * temp;

                        //         let eval_0 = -tmp * eq_evals_at_idx[1].0;
                        //         let eval_1 = if d < D - 1 {
                        //             eq_evals_at_idx[1].1 * factor * after_idx_evals[D - d - 2].1
                        //         } else {
                        //             eq_evals_at_idx[1].1 * factor
                        //         };

                        //         eval_0 + eval_1
                        //     },
                        // ]
                    })
                    // .reduce(
                    //     || [F::zero(); 2],
                    //     |running, new| [running[0] + new[0], running[1] + new[1]],
                    // );
                .reduce(
                    || vec![F::zero(); 2],
                    |running, new| vec![running[0] + new[0], running[1] + new[1]],
                );

                drop(_guard);
                drop(_span);

                let _span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
                let _guard = _span.enter();

                let univariate_poly = UniPoly::from_evals(&[
                    univariate_poly_evals[0],
                    *previous_claim - univariate_poly_evals[0],
                    univariate_poly_evals[1],
                ]);
                let compressed_poly = univariate_poly.compress();
                compressed_poly.append_to_transcript(transcript);
                compressed_polys.push(compressed_poly);

                let w_j = transcript.challenge_scalar::<F>();
                *previous_claim = univariate_poly.evaluate(&w_j);
                w.push(w_j);

                drop(_guard);
                drop(_span);

                C_summands[0] *= w_j;
                C_summands[1] *= F::one() - w_j;
            }
            mle_vec.par_iter_mut().enumerate().for_each(|(d, mle)| {
                mle.bind_parallel(w[D - 1 - d], BindingOrder::HighToLow);
            });

            drop(_guard);
            drop(inner_span);
        }
        C *= C_summands[0] + C_summands[1];

        drop(_guard);
        drop(span);

        (
            Self {
                sumcheck_proof: SumcheckInstanceProof::new(compressed_polys),
                eq_claim: C,
                mle_claims: mle_vec
                    .iter()
                    .map(|func| func.final_sumcheck_claim())
                    .collect(),
            },
            w,
        )
    }

    pub fn verify(
        &self,
        claim: F,
        r_prime: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let (sumcheck_claim, _r_sumcheck) =
            self.sumcheck_proof
                .verify(claim, r_prime.len(), 2, transcript)?;

        assert_eq!(
            sumcheck_claim,
            self.eq_claim * self.mle_claims.iter().product::<F>()
        );

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::time::{Duration, Instant};

    use ark_bn254::Fr;
    use ark_std::test_rng;
    use criterion::Criterion;
    use rand_core::RngCore;
    use rayon::{
        iter::{
            IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
            ParallelIterator,
        },
        slice::ParallelSliceMut,
    };

    use crate::{
        field::JoltField,
        poly::{
            eq_poly::EqPolynomial,
            multilinear_polynomial::{
                BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
            },
            unipoly::UniPoly,
        },
        subprotocols::optimization::{
            compute_initial_eval_claim, LargeDSumCheckProof, NaiveSumCheckProof,
        },
        utils::{
            math::Math,
            thread::unsafe_allocate_zero_vec,
            transcript::{KeccakTranscript, Transcript},
        },
    };

    const MAX_NUM_BITS: u32 = 32;

    fn multi_eq<F: JoltField>(D: usize, T: usize) -> Vec<F> {
        // Compute the polynomial eq(j, j_1, ..., j_d) = 1 if j = j_1 = ... = j_d and 0 otherwise.

        let mut eq: Vec<F> = unsafe_allocate_zero_vec(T.pow(D as u32 + 1) as usize);
        let num_bits = (T as usize).log_2() * ((D + 1) as usize);
        let log_T = (T as usize).log_2();

        for i in 0..T.pow(D as u32 + 1) {
            // Lower index correspond to lower bits.
            let bits = (0..MAX_NUM_BITS)
                .take(num_bits)
                .map(|n| ((i >> n) & 1) as u32)
                .collect::<Vec<u32>>();
            assert_eq!(bits.len(), num_bits);

            // Highest log_T bits represent j.
            let last_j_bits = &bits[bits.len() - log_T..];
            let j = (0..D)
                .map(|d| {
                    bits[..bits.len() - log_T]
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| idx % (D as usize) == d as usize)
                        .map(|(_, bit)| *bit)
                        .collect::<Vec<u32>>()
                })
                .collect::<Vec<_>>();

            let mut bit_vec = bits[..bits.len() - log_T]
                .chunks(D as usize)
                .map(|x| x.to_owned())
                .collect::<Vec<_>>();
            bit_vec
                .iter_mut()
                .zip(last_j_bits.iter())
                .for_each(|(chunk, digit)| {
                    chunk.push(*digit);
                });
            assert_eq!(bit_vec[0].len(), D as usize + 1);

            let val = bit_vec
                .iter()
                .map(|val| {
                    let evals = val
                        .iter()
                        .map(|digit| (*digit, 1 - digit))
                        .reduce(|running, new| (running.0 * new.0, running.1 * new.1))
                        .unwrap();

                    evals.0 + evals.1
                })
                .product::<u32>();

            eq[i as usize] = if j.iter().all(|j| *j == *last_j_bits) {
                F::one()
            } else {
                F::zero()
            };

            assert_eq!(F::from_u32(val), eq[i as usize]);
        }

        eq
    }

    fn read_addresses_to_ra_vec<F: JoltField>(
        read_addresses: &Vec<usize>,
        K: usize,
        T: usize,
    ) -> Vec<F> {
        // TODO: can separate this into a single utils function to be used in the module.
        // Compute ra in cycle-major order, since we will be binding
        // from low-to-high starting with the cycle variables.
        // The higher bits represent the address variables.
        let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
        ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
            for j in 0..T {
                if read_addresses[j] == k {
                    ra_k[j] = F::one();
                }
            }
        });
        ra
    }

    struct TestPerf {
        naive_duration: Duration,
        optimized_duration: Duration,
    }

    #[test]
    fn test_large_d_optimization_sumcheck() {
        let test_inputs = [
            // (2, 1 << 10, 16),
            // (8, 1 << 10, 16),
            // (16, 1 << 10, 16),
            // (32, 1 << 10, 16),

            // (8, 1 << 20, 1 << 16),
            // (12, 1 << 20, 1 << 16),
            // (16, 1 << 20, 1 << 16),
            // (50, 1 << 20, 1 << 16),
            (16, 1 << 10),
        ];

        for (D, T) in test_inputs {
            large_d_optimization_ra_virtualization::<15>(D, T);
            // benchmark_large_d_optimization_ra_virtualization(&mut criterion, D, T);
        }
    }

    fn test_func_data(D: usize, T: usize) -> Vec<MultilinearPolynomial<Fr>> {
        let mut rng = test_rng();
        let mut val_vec: Vec<Vec<Fr>> = vec![unsafe_allocate_zero_vec(T); D];

        for j in 0..T {
            for i in 0..D {
                val_vec[i][j] = Fr::from_u32(rng.next_u32());
            }
        }

        let val_mle = val_vec
            .into_par_iter()
            .map(|val| MultilinearPolynomial::from(val))
            .collect::<Vec<_>>();

        val_mle
    }

    fn test_ra_data(D: usize, T: usize, K: usize) -> Vec<MultilinearPolynomial<Fr>> {
        let mut rng = test_rng();
        let mut read_addresses: Vec<Vec<usize>> = vec![Vec::with_capacity(T); D];

        for _ in 0..T {
            for i in 0..D {
                let read_address = rng.next_u32() as usize % K;
                read_addresses[i].push(read_address);
            }
        }
        panic!("Success");

        println!("Check point");
        let mut ra = read_addresses
            .iter()
            .map(|r| {
                let ra_vec = read_addresses_to_ra_vec::<Fr>(r, K, T);
                MultilinearPolynomial::from(ra_vec)
            })
            .collect::<Vec<_>>();

        let mut dummy_transcript = KeccakTranscript::new(b"dummy");
        let _r_address = (0..D)
            .map(|idx| {
                let r_address = dummy_transcript.challenge_vector::<Fr>(K.log_2());
                r_address.iter().for_each(|r| {
                    ra[idx].bind_parallel(*r, BindingOrder::HighToLow);
                });
                r_address
            })
            .collect::<Vec<_>>();
        ra
    }

    fn check_initial_eval_claim(
        D: usize,
        T: usize,
        r_cycle: &Vec<Fr>,
        ra: &Vec<MultilinearPolynomial<Fr>>,
    ) {
        assert!(T.is_power_of_two());
        let eq = multi_eq::<Fr>(D, T);
        let mut eq = MultilinearPolynomial::from(eq);

        r_cycle
            .iter()
            .for_each(|r| eq.bind_parallel(*r, BindingOrder::HighToLow));

        // Sanity check that eq is computed correctly.
        for j in 0..T.pow(D as u32) {
            let num_bits = D * T.log_2();
            // Lower index correspond to lower bits.
            let bits_f = (0..MAX_NUM_BITS)
                .take(num_bits)
                .map(|n| ((j >> n) & 1) as u32)
                .map(|bit| Fr::from_u32(bit))
                .map(|val| (val, Fr::from_u32(1) - val))
                .collect::<Vec<_>>();
            assert_eq!(bits_f.len(), num_bits);

            let mut j_bit_vec: Vec<Vec<_>> = bits_f
                .chunks(D)
                .map(|chunk| chunk.to_owned())
                .collect::<Vec<_>>();

            j_bit_vec
                .iter_mut()
                .zip(r_cycle.iter().rev())
                .for_each(|(chunk, digit)| {
                    chunk.push((*digit, Fr::from_u32(1) - *digit));
                });
            assert_eq!(j_bit_vec[0].len(), D + 1);

            let _eq_eval = j_bit_vec
                .into_iter()
                .map(|chunk| {
                    let res = chunk
                        .into_iter()
                        .reduce(|running, new| (running.0 * new.0, running.1 * new.1))
                        .unwrap();
                    res.0 + res.1
                })
                .product::<Fr>();
            // assert_eq!(eq_eval, eq.get_bound_coeff(j));
        }

        let previous_claim_bench = (0..T.pow(D as u32))
            .map(|j| {
                let num_bits = D * T.log_2();
                // Lower index correspond to lower bits.
                let bits = (0..MAX_NUM_BITS)
                    .take(num_bits)
                    .map(|n| ((j >> n) & 1) as u32)
                    .collect::<Vec<u32>>();
                assert_eq!(bits.len(), num_bits);

                let j_vec = (0..D)
                    .map(|d| {
                        let res: u32 = bits
                            .iter()
                            .enumerate()
                            .filter(|(idx, _)| idx % (D as usize) == d as usize)
                            .map(|(_, bit)| *bit)
                            .enumerate()
                            .map(|(idx, bit)| bit << idx)
                            .sum();
                        res
                    })
                    .collect::<Vec<_>>();

                j_vec
                    .iter()
                    .enumerate()
                    .map(|(idx, j_d)| ra[idx].get_bound_coeff(*j_d as usize))
                    .product::<Fr>()
                    * eq.get_bound_coeff(j)
            })
            .sum::<Fr>();

        let previous_claim = compute_initial_eval_claim(&ra.iter().collect::<Vec<_>>(), r_cycle);
        assert_eq!(previous_claim, previous_claim_bench);
    }

    fn benchmark_large_d_optimization_ra_virtualization<const D1: usize>(
        criterion: &mut Criterion,
        D: usize,
        T: usize,
    ) {
        let test_perf = large_d_optimization_ra_virtualization::<D1>(D, T);
        criterion.bench_function(
            &format!("large_d_optimization_ra_virtualization_{D}_{T}"),
            |b| b.iter(|| criterion::black_box(large_d_optimization_ra_virtualization::<D1>(D, T))),
        );
    }

    fn large_d_optimization_ra_virtualization<const D1: usize>(D: usize, T: usize) -> TestPerf {
        assert!(T.is_power_of_two(), "T: {T}");

        // Compute the sum-check
        // ra(k_1, ..., k_d, j) = \sum_{j_1, ..., j_d} eq(j, j_1, ..., j_d) \prod_{i=1}^d ra(k_i, j_i)
        // where eq(j, j_1, ..., j_d) = 1 if j = j_1 = ... = j_d and 0 otherwise.
        // let mut ra = test_ra_data(D, T, K);
        let mut ra = test_func_data(D, T);

        let mut ra_copy = ra.clone();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        if D < 6 && T < 1 << 6 {
            check_initial_eval_claim(D, T, &r_cycle, &ra);
        }
        let mut previous_claim =
            compute_initial_eval_claim(&ra.iter().map(|x| &*x).collect::<Vec<_>>(), &r_cycle);
        let mut previous_claim_copy = previous_claim;
        let claim_copy = previous_claim_copy.clone();
        let claim = previous_claim.clone();

        let start_time = Instant::now();
        let (proof, r_prime) = LargeDSumCheckProof::<Fr, KeccakTranscript>::prove::<D1>(
            &mut ra.iter_mut().collect::<Vec<_>>(),
            &r_cycle,
            &mut previous_claim,
            &mut prover_transcript,
        );
        let optimized_duration = start_time.elapsed();

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        let verification_result = proof.verify(claim, r_prime, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification (optimized sumcheck) failed: {verification_result:?}"
        );

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let start_time = Instant::now();
        let (proof, r_prime) = NaiveSumCheckProof::<Fr, KeccakTranscript>::prove(
            &mut ra_copy.iter_mut().collect::<Vec<_>>(),
            &r_cycle,
            &mut previous_claim_copy,
            &mut prover_transcript,
        );
        let naive_duration = start_time.elapsed();

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        let verification_result = proof.verify(r_prime, claim_copy, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification (naive sumcheck) failed: {verification_result:?}"
        );

        TestPerf {
            naive_duration,
            optimized_duration,
        }
    }
}
