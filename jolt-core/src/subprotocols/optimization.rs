use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    field::JoltField,
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

/// Contains the proof for a generic sumcheck of the form
/// val = \sum_{j_1, ..., j_d \in \{0, 1\}^T} eq(j, j_1, ..., j_d) \prod_{i=1}^d func(j_i),
/// where eq(j, j_1, ..., j_d) = 1 if j = j_1 = ... = j_d and 0 otherwise.
///
/// See Appendix C of the Twist + Shout paper for an example with ra virtualization.
pub struct LargeDSumCheckProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    eq_claim: F,
    func_claims: Vec<F>,
}

impl<F: JoltField, ProofTranscript: Transcript> LargeDSumCheckProof<F, ProofTranscript> {
    pub fn prove(
        eq: &mut MultilinearPolynomial<F>,
        funcs: &mut Vec<&mut MultilinearPolynomial<F>>,
        r_cycle: &Vec<F>,
        previous_claim: F,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>) {
        let mut C = F::one();
        let mut C_summands = [F::one(), F::one()];
        let D = funcs.len();
        let T = r_cycle.len().pow2();
        let mut previous_claim = previous_claim;

        let eval_points = [0, 2]
            .into_iter()
            .map(|x| F::from_u32(x))
            .collect::<Vec<_>>();

        // Each table E_i stores the evaluations of eq(j_{>i}, r_cycle_{>i}) for each j_{>i}.
        // As we're binding from high to low, for each E_i we store eq(j_{<LogT - i}, r_cycle_{<+LogT - i}) instead.
        let E_table = (1..=T.log_2() - 1)
            .map(|i| {
                let evals =
                    EqPolynomial::evals(&r_cycle[i..].iter().map(|x| *x).collect::<Vec<_>>());
                MultilinearPolynomial::from(evals)
            })
            .collect::<Vec<_>>();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(D * T.log_2());
        let mut w: Vec<F> = Vec::with_capacity(D * T.log_2());

        for round in 0..D * T.log_2() {
            let d = round % D;
            // j_idx is the index (counting backwards as we're binding from high to low) of the cycle variable vector
            // that we're binding for the corresponding lookup ra.
            let j_idx = round / D;

            if round % D == 0 {
                if round > 0 {
                    C *= C_summands[0] + C_summands[1];
                }

                let r_cycle_val = r_cycle[j_idx];

                C_summands[0] = r_cycle_val;
                C_summands[1] = F::one() - r_cycle_val;
            }

            // Compute eq(r_round, w_1, ..., w_{idx - 1}, c, b) for each c and b = 0, 1
            let eq_evals_at_idx = eval_points
                .iter()
                .map(|c| (((F::one() - c) * C_summands[1], *c * C_summands[0])))
                .collect::<Vec<(F, F)>>();

            let univariate_poly_evals = (0..(T.log_2() - j_idx - 1).pow2())
                .into_par_iter()
                .map(|j| {
                    let at_idx_evals =
                        funcs[D - d - 1].sumcheck_evals(j, 2, BindingOrder::HighToLow);

                    let eq_eval_after_idx = if j_idx < T.log_2() - 1 {
                        E_table[j_idx].get_coeff(j)
                    } else {
                        F::one()
                    };

                    let before_idx_evals = funcs
                        .iter()
                        .rev()
                        .take(d)
                        .map(|poly| poly.get_bound_coeff(j))
                        .product::<F>();

                    let after_idx_evals = funcs
                        .iter()
                        .take(D - d - 1)
                        .map(|poly| {
                            (
                                poly.get_bound_coeff(j),
                                poly.get_bound_coeff(j + poly.len() / 2),
                            )
                        })
                        .reduce(|running, new| (running.0 * new.0, running.1 * new.1));

                    eq_evals_at_idx
                        .iter()
                        .zip(at_idx_evals.iter())
                        .map(|((c_eq_eval_0, c_eq_eval_1), at_idx_eval)| {
                            let factor = *at_idx_eval * before_idx_evals * eq_eval_after_idx * C;
                            let eval_0 = if after_idx_evals.is_some() {
                                *c_eq_eval_0 * factor * after_idx_evals.unwrap().0
                            } else {
                                *c_eq_eval_0 * factor
                            };

                            let eval_1 = if after_idx_evals.is_some() {
                                *c_eq_eval_1 * factor * after_idx_evals.unwrap().1
                            } else {
                                *c_eq_eval_1 * factor
                            };

                            eval_0 + eval_1
                        })
                        .collect::<Vec<_>>()
                })
                .reduce(
                    || vec![F::zero(); eval_points.len()],
                    |running, new| {
                        running
                            .iter()
                            .zip(new.iter())
                            .map(|(a, b)| *a + b)
                            .collect::<Vec<_>>()
                    },
                );

            let univariate_poly = UniPoly::from_evals(&[
                univariate_poly_evals[0],
                previous_claim - univariate_poly_evals[0],
                univariate_poly_evals[1],
            ]);
            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let w_j = transcript.challenge_scalar::<F>();
            previous_claim = univariate_poly.evaluate(&w_j);
            w.push(w_j);

            rayon::join(
                || eq.bind_parallel(w_j, BindingOrder::HighToLow),
                || funcs[D - 1 - d].bind_parallel(w_j, BindingOrder::HighToLow),
            );

            C_summands[0] *= w_j;
            C_summands[1] *= F::one() - w_j;
        }
        C *= C_summands[0] + C_summands[1];

        (
            Self {
                sumcheck_proof: SumcheckInstanceProof::new(compressed_polys),
                eq_claim: C,
                func_claims: funcs
                    .iter()
                    .map(|func| func.final_sumcheck_claim())
                    .collect(),
            },
            w,
        )
    }

    pub fn verify(
        &self,
        r_prime: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let (_sumcheck_claim, _r_sumcheck) = self.sumcheck_proof.verify(
            self.eq_claim * self.func_claims.iter().product::<F>(),
            r_prime.len(),
            2,
            transcript,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_core::RngCore;
    use rayon::{
        iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
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
        subprotocols::optimization::LargeDSumCheckProof,
        utils::{
            math::Math,
            thread::unsafe_allocate_zero_vec,
            transcript::{KeccakTranscript, Transcript},
        },
    };

    fn multi_eq<F: JoltField>(D: u32, T: u32) -> Vec<F> {
        // Compute the polynomial eq(j, j_1, ..., j_d) = 1 if j = j_1 = ... = j_d and 0 otherwise.

        let mut eq: Vec<F> = unsafe_allocate_zero_vec(T.pow(D + 1) as usize);
        let num_bits = (T as usize).log_2() * ((D + 1) as usize);
        let log_T = (T as usize).log_2();

        for i in 0..T.pow(D + 1) {
            // Lower index correspond to lower bits.
            let bits = (0..32)
                .take(num_bits)
                .map(|n| (i >> n) & 1)
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

    #[test]
    fn test_large_d_optimization_sumcheck() {
        large_d_optimization_ra_virtualization(3, 1 << 3, 16);
        large_d_optimization_ra_virtualization(2, 1 << 4, 4);
        large_d_optimization_ra_virtualization(5, 1 << 3, 8);
        large_d_optimization_ra_virtualization(10, 1 << 2, 4);
    }

    fn large_d_optimization_ra_virtualization(D: usize, T: usize, K: usize) {
        assert!(T.is_power_of_two());
        assert!(K.is_power_of_two());

        // Compute the sum-check
        // ra(k_1, ..., k_d, j) = \sum_{j_1, ..., j_d} eq(j, j_1, ..., j_d) \prod_{i=1}^d ra(k_i, j_i)
        // where eq(j, j_1, ..., j_d) = 1 if j = j_1 = ... = j_d and 0 otherwise.
        let mut rng = test_rng();
        let mut read_addresses: Vec<Vec<usize>> = vec![Vec::with_capacity(T); D];

        for _ in 0..T {
            for i in 0..D {
                let read_address = rng.next_u32() as usize % K;
                read_addresses[i].push(read_address);
            }
        }
        let mut ra = read_addresses
            .iter()
            .map(|r| {
                let ra_vec = read_addresses_to_ra_vec::<Fr>(r, K, T);
                MultilinearPolynomial::from(ra_vec)
            })
            .collect::<Vec<_>>();

        let eq = multi_eq::<Fr>(D as u32, T as u32);
        let mut eq = MultilinearPolynomial::from(eq);

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        r_cycle
            .iter()
            .for_each(|r| eq.bind_parallel(*r, BindingOrder::HighToLow));

        // Sanity check that eq is computed correctly.
        for j in 0..T.pow(D as u32) {
            let num_bits = D * T.log_2();
            // Lower index correspond to lower bits.
            let bits_f = (0..32)
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
        let previous_claim = (0..T.pow(D as u32))
            .map(|j| {
                let num_bits = D * T.log_2();
                // Lower index correspond to lower bits.
                let bits = (0..32)
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

        let (proof, r_prime) = LargeDSumCheckProof::<Fr, KeccakTranscript>::prove(
            &mut eq,
            &mut ra.iter_mut().collect::<Vec<_>>(),
            &r_cycle,
            previous_claim,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        let verification_result = proof.verify(r_prime, &mut verifier_transcript);

        assert!(
            verification_result.is_ok(),
            "Verification failed: {verification_result:?}"
        );
    }
}
