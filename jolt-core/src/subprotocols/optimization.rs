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
    subprotocols::{
        karatsuba::{coeff_kara_16, coeff_naive, kara_17},
        sumcheck::SumcheckInstanceProof,
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        transcript::{AppendToTranscript, Transcript},
    },
};

fn verify_product_mle_claim<F: JoltField>(
    mle_vec: &Vec<&MultilinearPolynomial<F>>,
    r_cycle: &Vec<F>,
    claim: F,
) -> bool {
    todo!()
}

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

pub struct KaratsubaSumCheckProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    eq_claim: F,
    mle_claims: Vec<F>,
}

impl<F: JoltField, ProofTranscript: Transcript> KaratsubaSumCheckProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "KaratsubaSumCheckProof::prove")]
    pub fn prove(
        mle_vec: &mut Vec<&mut MultilinearPolynomial<F>>,
        r_cycle: &Vec<F>,
        previous_claim: &mut F,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>) {
        let d_plus_one = mle_vec.len() + 1;

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

            let evals = (0..(log_T - round - 1).pow2())
                .into_par_iter()
                .map(|j| {
                    let mle_evals: Vec<F> = match d_plus_one {
                        16 => {
                            // (constant, slope)
                            let flat: [F; 32] = mle_vec
                                .iter()
                                .flat_map(|x| {
                                    [
                                        x.get_bound_coeff(j),
                                        x.get_bound_coeff(j + x.len() / 2) - x.get_bound_coeff(j),
                                    ]
                                })
                                .chain(iter::once(eq.get_bound_coeff(j)))
                                .chain(iter::once(
                                    eq.get_bound_coeff(j + eq.len() / 2) - eq.get_bound_coeff(j),
                                ))
                                .collect::<Vec<_>>()
                                .try_into()
                                .unwrap();

                            let kera_res = coeff_kara_16(&flat);

                            Vec::from(kera_res)
                        }
                        _ => unimplemented!(),
                    };

                    // #[cfg(test)]
                    // {
                    //     let eq_evals =
                    //         eq.sumcheck_evals(j, mle_vec.len() + 1, BindingOrder::HighToLow);
                    //     let bench_mle_evals = mle_vec
                    //         .iter()
                    //         .map(|poly| {
                    //             poly.sumcheck_evals(j, mle_vec.len() + 1, BindingOrder::HighToLow)
                    //         })
                    //         .collect::<Vec<_>>();

                    //     let bench_evals = bench_mle_evals
                    //         .into_iter()
                    //         .chain(iter::once(eq_evals))
                    //         .reduce(|running, new| {
                    //             running
                    //                 .iter()
                    //                 .zip(new.iter())
                    //                 .map(|(a, b)| *a * b)
                    //                 .collect::<Vec<_>>()
                    //         })
                    //         .unwrap();

                    //     let poly = UniPoly {
                    //         coeffs: mle_evals.clone(),
                    //     };

                    //     for i in 0..bench_evals.len() + 1 {
                    //         if i == 1 {
                    //             continue;
                    //         }

                    //         let bench = if i == 0 {
                    //             bench_evals[0]
                    //         } else {
                    //             bench_evals[i - 1]
                    //         };

                    //         assert_eq!(poly.evaluate(&F::from_u32(i as u32)), bench, "i = {}", i);
                    //     }
                    // }

                    mle_evals
                })
                .reduce(
                    || vec![F::zero(); mle_vec.len() + 2],
                    |running, new| {
                        running
                            .iter()
                            .zip(new.iter())
                            .map(|(a, b)| *a + b)
                            .collect::<Vec<_>>()
                    },
                );

            assert_eq!(evals.len(), mle_vec.len() + 2);

            drop(_guard);
            drop(inner_span);

            let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
            let _guard = inner_span.enter();

            let univariate_poly = UniPoly { coeffs: evals };
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
                        mle_vec[0].get_bound_coeff(j),
                        mle_vec[0].get_bound_coeff(j + mle_vec[0].len() / 2),
                    );

                    let res: [(F, F); D1] = std::array::from_fn(|i| {
                        let entry = cur;
                        if i < D1 - 1 {
                            cur = (
                                cur.0 * mle_vec[i + 1].get_bound_coeff(j),
                                cur.1
                                    * mle_vec[i + 1].get_bound_coeff(j + mle_vec[i + 1].len() / 2),
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
                        let at_idx_evals = [
                            mle_vec[D - d - 1].get_bound_coeff(j),
                            mle_vec[D - d - 1].get_bound_coeff(j + mle_vec[D - d - 1].len() / 2)
                                + mle_vec[D - d - 1]
                                    .get_bound_coeff(j + mle_vec[D - d - 1].len() / 2)
                                - mle_vec[D - d - 1].get_bound_coeff(j),
                        ];

                        if d > 0 {
                            *before_idx_eval =
                                before_idx_eval.mul_1_optimized(mle_vec[D - d].get_bound_coeff(j));
                        }

                        let eq_eval_after_idx = if j_idx < T.log_2() - 1 {
                            E_table[j_idx][j]
                        } else {
                            F::one()
                        };

                        let temp =
                            before_idx_eval.mul_1_optimized(eq_eval_after_idx.mul_1_optimized(C));

                        let tmp = if d < D - 1 {
                            temp * after_idx_evals[D - d - 2].0
                        } else {
                            temp
                        };

                        let factor: F = at_idx_evals[1] * temp;

                        let eval_0 = -tmp * at_idx_evals[1];
                        let eval_1 = if d < D - 1 {
                            factor * after_idx_evals[D - d - 2].1
                        } else {
                            factor
                        };

                        [at_idx_evals[0] * tmp, eval_0, eval_1]
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );
                let univariate_poly_evals = [
                    univariate_poly_evals[0] * eq_evals_at_idx[0].0,
                    univariate_poly_evals[1] * eq_evals_at_idx[0].0
                        + univariate_poly_evals[2] * eq_evals_at_idx[1].1,
                ];

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

                mle_vec[D - d - 1].bind_parallel(w_j, BindingOrder::HighToLow);
            }
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
    use rand_core::RngCore;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    use crate::{
        field::JoltField,
        poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        subprotocols::optimization::{
            compute_initial_eval_claim, KaratsubaSumCheckProof, LargeDSumCheckProof,
            NaiveSumCheckProof,
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

    #[test]
    fn test_large_d_optimization_sumcheck() {
        large_d_optimization_ra_virtualization::<14>(15, 1 << 10);
        // large_d_optimization_ra_virtualization::<15>(16, 1 << 10);
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

    fn large_d_optimization_ra_virtualization<const D1: usize>(D: usize, T: usize) {
        assert!(T.is_power_of_two(), "T: {T}");

        // Compute the sum-check
        // ra(k_1, ..., k_d, j) = \sum_{j_1, ..., j_d} eq(j, j_1, ..., j_d) \prod_{i=1}^d ra(k_i, j_i)
        // where eq(j, j_1, ..., j_d) = 1 if j = j_1 = ... = j_d and 0 otherwise.
        // let mut ra = test_ra_data(D, T, K);
        let mut ra = test_func_data(D, T);
        let mut ra_copy = ra.clone();
        let mut ra_copy_2 = ra.clone();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        if D < 6 && T < 1 << 6 {
            check_initial_eval_claim(D, T, &r_cycle, &ra);
        }
        let mut previous_claim =
            compute_initial_eval_claim(&ra.iter().map(|x| &*x).collect::<Vec<_>>(), &r_cycle);
        let mut previous_claim_copy = previous_claim;
        let mut previous_claim_copy_2 = previous_claim;

        let claim = previous_claim.clone();
        let claim_copy = previous_claim_copy.clone();
        let claim_copy_2 = previous_claim_copy_2.clone();

        let start_time = Instant::now();
        let (proof, r_prime) = LargeDSumCheckProof::<Fr, KeccakTranscript>::prove::<D1>(
            &mut ra.iter_mut().collect::<Vec<_>>(),
            &r_cycle,
            &mut previous_claim,
            &mut prover_transcript,
        );
        let _optimized_duration = start_time.elapsed();

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
        let _naive_duration = start_time.elapsed();

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        let verification_result = proof.verify(r_prime, claim_copy, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification (naive sumcheck) failed: {verification_result:?}"
        );

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let start_time = Instant::now();
        let (proof, r_prime) = KaratsubaSumCheckProof::<Fr, KeccakTranscript>::prove(
            &mut ra_copy_2.iter_mut().collect::<Vec<_>>(),
            &r_cycle,
            &mut previous_claim_copy_2,
            &mut prover_transcript,
        );
        let _karatsuba_duration = start_time.elapsed();

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        let verification_result = proof.verify(r_prime, claim_copy_2, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification (naive sumcheck) failed: {verification_result:?}"
        );
    }
}
