use super::sumcheck::SumcheckInstanceProof;
use crate::{
    field::{JoltField, OptimizedMul},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::{CompressedUniPoly, UniPoly},
    },
    utils::{
        math::Math,
        thread::unsafe_allocate_zero_vec,
        transcript::{AppendToTranscript, Transcript},
    },
};
use rayon::prelude::*;

/// The Twist+Shout paper gives two different prover algorithms for the read-checking
/// and write-checking algorithms in Twist, called the "local algorithm" and
/// "alternative algorithm". The local algorithm has worse dependence on the pararmeter
/// d, but benefits from locality of memory accesses.
pub enum TwistAlgorithm {
    /// The "local algorithm" for Twist's read-checking and write-checking sumchecks,
    /// described in Sections 8.2.2, 8.2.3, 8.2.4. Worse dependence on d, but benefits
    /// from locality of memory accesses.
    Local,
    /// The "alternative algorithm" for Twist's read-checking and write-checking sumchecks,
    /// described in Section 8.2.5. Better dependence on d, but does not benefit
    /// from locality of memory accesses.
    Alternative,
}

pub struct TwistProof<F: JoltField, ProofTranscript: Transcript> {
    /// Joint sumcheck proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_sumcheck: SumcheckInstanceProof<F, ProofTranscript>,
    /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    ra_claim: F,
    /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
    rv_claim: F,
    /// The claimed evaluation wa(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    wa_claim: F,
    /// The claimed evaluation wv(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    wv_claim: F,
    /// The claimed evaluation val(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    val_claim: F,
    /// The claimed evaluation Inc(r, r') proven by the write-checking sumcheck.
    inc_claim: F,
}

pub fn prove_read_write_checking<F: JoltField, ProofTranscript: Transcript>(
    read_addresses: Vec<usize>,
    read_values: Vec<u32>,
    write_addresses: Vec<usize>,
    write_values: Vec<u32>,
    write_increments: Vec<i64>,
    r: Vec<F>,
    r_prime: Vec<F>,
    transcript: &mut ProofTranscript,
    algorithm: TwistAlgorithm,
) -> TwistProof<F, ProofTranscript> {
    match algorithm {
        TwistAlgorithm::Local => prove_read_write_checking_local(
            read_addresses,
            read_values,
            write_addresses,
            write_values,
            write_increments,
            &r,
            &r_prime,
            transcript,
        ),
        TwistAlgorithm::Alternative => unimplemented!(),
    }
}

fn prove_read_write_checking_local<F: JoltField, ProofTranscript: Transcript>(
    read_addresses: Vec<usize>,
    read_values: Vec<u32>,
    write_addresses: Vec<usize>,
    write_values: Vec<u32>,
    write_increments: Vec<i64>,
    r: &[F],
    r_prime: &[F],
    transcript: &mut ProofTranscript,
) -> TwistProof<F, ProofTranscript> {
    const DEGREE: usize = 3;
    let K = r.len().pow2();
    let T = r_prime.len().pow2();

    debug_assert_eq!(read_addresses.len(), T);
    debug_assert_eq!(read_values.len(), T);
    debug_assert_eq!(write_addresses.len(), T);
    debug_assert_eq!(write_values.len(), T);
    debug_assert_eq!(write_increments.len(), T);

    // Used to batch the read-checking and write-checking sumcheck
    // (see Section 4.2.1)
    let z: F = transcript.challenge_scalar();

    let num_rounds = K.log_2() + T.log_2();
    let mut r_sumcheck: Vec<F> = Vec::with_capacity(num_rounds);

    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = T / num_chunks;

    #[cfg(test)]
    let mut val_test = {
        // Compute Val in cycle-major order, since we will be binding
        // from low-to-high starting with the cycle variables
        let mut val: Vec<u32> = vec![0; K * T];
        val.par_chunks_mut(T).enumerate().for_each(|(k, val_k)| {
            let mut current_val = 0;
            for j in 0..T {
                val_k[j] = current_val;
                if write_addresses[j] == k {
                    current_val = write_values[j];
                }
            }
        });
        MultilinearPolynomial::from(val)
    };
    #[cfg(test)]
    let mut ra_test = {
        // Compute ra in cycle-major order, since we will be binding
        // from low-to-high starting with the cycle variables
        let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
        ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
            for j in 0..T {
                if read_addresses[j] == k {
                    ra_k[j] = F::one();
                }
            }
        });
        MultilinearPolynomial::from(ra)
    };
    #[cfg(test)]
    let mut wa_test = {
        // Compute wa in cycle-major order, since we will be binding
        // from low-to-high starting with the cycle variables
        let mut wa: Vec<F> = unsafe_allocate_zero_vec(K * T);
        wa.par_chunks_mut(T).enumerate().for_each(|(k, wa_k)| {
            for j in 0..T {
                if write_addresses[j] == k {
                    wa_k[j] = F::one();
                }
            }
        });
        MultilinearPolynomial::from(wa)
    };

    let deltas: Vec<Vec<i64>> = write_addresses[..T - chunk_size]
        .par_chunks_exact(chunk_size)
        .zip(write_increments[..T - chunk_size].par_chunks_exact(chunk_size))
        .map(|(address_chunk, increment_chunk)| {
            let mut delta = vec![0i64; K];
            for (k, increment) in address_chunk.iter().zip(increment_chunk.iter()) {
                delta[*k] += increment;
            }
            delta
        })
        .collect();

    // Value in register k before the jth cycle, for j \in {0, chunk_size, 2 * chunk_size, ...}
    let mut checkpoints: Vec<Vec<i64>> = Vec::with_capacity(num_chunks);
    // TODO(moodlezoup): Initial memory state may not be all zeros
    checkpoints.push(vec![0; K]);

    for (chunk_index, delta) in deltas.iter().enumerate() {
        let next_checkpoint = checkpoints[chunk_index]
            .par_iter()
            .zip(delta.par_iter())
            .map(|(val_k, delta_k)| val_k + delta_k)
            .collect();
        checkpoints.push(next_checkpoint);
    }
    // TODO(moodlezoup): could potentially generate these checkpoints in the tracer
    let checkpoints: Vec<Vec<F>> = checkpoints
        .into_par_iter()
        .map(|checkpoint| checkpoint.into_iter().map(|val| F::from_i64(val)).collect())
        .collect();

    #[cfg(test)]
    {
        // Check that checkpoints are correct
        for (chunk_index, V_chunk) in checkpoints.iter().enumerate() {
            let j = chunk_index * chunk_size;
            for (k, V_k) in V_chunk.iter().enumerate() {
                assert_eq!(*V_k, val_test.get_bound_coeff(k * T + j));
            }
        }
    }

    // A table that, in round i of sumcheck, stores all evaluations
    //     EQ(x, r_i, ..., r_1)
    // as x ranges over {0, 1}^i.
    // (As described in "Computing other necessary arrays and worst-case
    // accounting", Section 8.2.2)
    let mut A: Vec<F> = unsafe_allocate_zero_vec(chunk_size);
    A[0] = F::one();

    // Data structure described in Equation (72)
    let mut I: Vec<Vec<(usize, usize, F, F)>> = write_addresses
        .par_chunks(chunk_size)
        .zip(write_increments.par_chunks(chunk_size))
        .enumerate()
        .map(|(chunk_index, (address_chunk, increment_chunk))| {
            // Row index of the I matrix
            let mut j = chunk_index * chunk_size;
            let I_chunk = address_chunk
                .iter()
                .zip(increment_chunk.iter())
                .map(|(k, increment)| {
                    let inc = (j, *k, F::zero(), F::from_i64(*increment));
                    j += 1;
                    inc
                })
                .collect();
            I_chunk
        })
        .collect();

    let rv = MultilinearPolynomial::from(read_values);
    let mut wv = MultilinearPolynomial::from(write_values);

    // eq(r, k)
    let mut eq_r = MultilinearPolynomial::from(EqPolynomial::evals(r));
    // eq(r', j)
    let mut eq_r_prime = MultilinearPolynomial::from(EqPolynomial::evals(r_prime));

    // rv(r')
    let rv_eval = rv.evaluate(r_prime);
    // Inc(r, r')
    let inc_eval: F = write_addresses
        .par_iter()
        .zip(write_increments.par_iter())
        .enumerate()
        .map(|(cycle, (address, increment))| {
            eq_r.get_coeff(*address) * eq_r_prime.get_coeff(cycle) * F::from_i64(*increment)
        })
        .sum();
    // Linear combination of the read-checking claim (which is rv(r')) and the
    // write-checking claim (which is Inc(r, r'))
    let mut previous_claim = rv_eval + z * inc_eval;
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    // First log(T / num_chunks) rounds of sumcheck
    for round in 0..chunk_size.log_2() {
        #[cfg(test)]
        {
            let mut expected_claim = F::zero();
            for j in 0..(T >> round) {
                let mut inner_sum = F::zero();
                for k in 0..K {
                    let kj = k * (T >> round) + j;
                    // read-checking sumcheck
                    inner_sum += ra_test.get_bound_coeff(kj) * val_test.get_bound_coeff(kj);
                    // write-checking sumcheck
                    inner_sum += z
                        * eq_r.get_bound_coeff(k)
                        * wa_test.get_bound_coeff(kj)
                        * (wv.get_bound_coeff(j) - val_test.get_bound_coeff(kj))
                }
                expected_claim += eq_r_prime.get_bound_coeff(j) * inner_sum;
            }
            assert_eq!(
                expected_claim, previous_claim,
                "Sumcheck sanity check failed in round {round}"
            );
        }

        let univariate_poly_evals: [F; 3] = I
            .par_iter()
            .enumerate()
            .map(|(chunk_index, I_chunk)| {
                let mut evals = [F::zero(), F::zero(), F::zero()];

                // `val_j_0` will contain
                //     Val(k, j', 0, ..., 0)
                // as we iterate over rows j' \in {0, 1}^(log(T) - i)
                let mut val_j_0 = checkpoints[chunk_index].clone();
                // `val_j_r[0]` will contain
                //     Val(k, j'', 0, r_i, ..., r_1)
                // `val_j_r[1]` will contain
                //     Val(k, j'', 1, r_i, ..., r_1)
                // as we iterate over rows j' \in {0, 1}^(log(T) - i)
                let mut val_j_r: [Vec<F>; 2] =
                    [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)];
                // `ra[0]` will contain
                //     ra(k, j'', 0, r_i, ..., r_1)
                // `ra[1]` will contain
                //     ra(k, j'', 1, r_i, ..., r_1)
                // as we iterate over rows j' \in {0, 1}^(log(T) - i),
                // where j'' are the higher (log(T) - i - 1) bits of j'
                let mut ra: [Vec<F>; 2] =
                    [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)];
                // `wa[0]` will contain
                //     wa(k, j'', 0, r_i, ..., r_1)
                // `wa[1]` will contain
                //     wa(k, j'', 1, r_i, ..., r_1)
                // as we iterate over rows j' \in {0, 1}^(log(T) - i),
                // where j'' are the higher (log(T) - i - 1) bits of j'
                let mut wa: [Vec<F>; 2] =
                    [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)];

                // Iterate over I_chunk, two rows at a time.
                I_chunk
                    .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                    .for_each(|inc_chunk| {
                        let j_prime = inc_chunk[0].0; // row index

                        val_j_r[0] = val_j_0.clone();
                        for inc in inc_chunk {
                            let (row, col, inc_lt, inc) = *inc;
                            if row == j_prime {
                                // First of the two rows
                                val_j_r[0][col] += inc_lt;
                                val_j_0[col] += inc;
                            } else {
                                // TODO(moodlezoup): single pass over `inc_chunk`
                                break;
                            }
                        }
                        val_j_r[1] = val_j_0.clone();
                        for inc in inc_chunk {
                            let (row, col, inc_lt, inc) = *inc;
                            if row != j_prime {
                                // Second of the two rows
                                val_j_r[1][col] += inc_lt;
                                val_j_0[col] += inc;
                            }
                        }

                        ra[0].iter_mut().for_each(|ra_k| *ra_k = F::zero());
                        wa[0].iter_mut().for_each(|wa_k| *wa_k = F::zero());
                        for j in j_prime << round..(j_prime + 1) << round {
                            let j_bound = j % (1 << round);
                            let k = read_addresses[j];
                            ra[0][k] += A[j_bound];
                            let k = write_addresses[j];
                            wa[0][k] += A[j_bound];
                        }
                        ra[1].iter_mut().for_each(|ra_k| *ra_k = F::zero());
                        wa[1].iter_mut().for_each(|wa_k| *wa_k = F::zero());
                        for j in (j_prime + 1) << round..(j_prime + 2) << round {
                            let j_bound = j % (1 << round);
                            let k = read_addresses[j];
                            ra[1][k] += A[j_bound];
                            let k = write_addresses[j];
                            wa[1][k] += A[j_bound];
                        }

                        #[cfg(test)]
                        {
                            for k in 0..K {
                                // Check val
                                assert_eq!(
                                    val_test.get_bound_coeff(k * (T >> round) + j_prime),
                                    val_j_r[0][k],
                                );
                                assert_eq!(
                                    val_test.get_bound_coeff(k * (T >> round) + j_prime + 1),
                                    val_j_r[1][k],
                                );
                                // Check ra
                                assert_eq!(
                                    ra_test.get_bound_coeff(k * (T >> round) + j_prime),
                                    ra[0][k]
                                );
                                assert_eq!(
                                    ra_test.get_bound_coeff(k * (T >> round) + j_prime + 1),
                                    ra[1][k]
                                );
                                // Check wa
                                assert_eq!(
                                    wa_test.get_bound_coeff(k * (T >> round) + j_prime),
                                    wa[0][k]
                                );
                                assert_eq!(
                                    wa_test.get_bound_coeff(k * (T >> round) + j_prime + 1),
                                    wa[1][k]
                                );
                            }
                        }

                        let eq_r_prime_evals =
                            eq_r_prime.sumcheck_evals(j_prime / 2, DEGREE, BindingOrder::LowToHigh);
                        let wv_evals =
                            wv.sumcheck_evals(j_prime / 2, DEGREE, BindingOrder::LowToHigh);

                        let mut inner_sum_evals = [F::zero(); 3];
                        for k in 0..K {
                            let m_ra = ra[1][k] - ra[0][k];
                            let ra_eval_2 = ra[1][k] + m_ra;
                            let ra_eval_3 = ra_eval_2 + m_ra;

                            let m_wa = wa[1][k] - wa[0][k];
                            let wa_eval_2 = wa[1][k] + m_wa;
                            let wa_eval_3 = wa_eval_2 + m_wa;

                            let m_val = val_j_r[1][k] - val_j_r[0][k];
                            let val_eval_2 = val_j_r[1][k] + m_val;
                            let val_eval_3 = val_eval_2 + m_val;

                            // Read-checking sumcheck
                            inner_sum_evals[0] += ra[0][k].mul_0_optimized(val_j_r[0][k]);
                            inner_sum_evals[1] += ra_eval_2.mul_0_optimized(val_eval_2);
                            inner_sum_evals[2] += ra_eval_3.mul_0_optimized(val_eval_3);

                            let z_eq_r = z * eq_r.get_coeff(k);
                            // Write-checking sumcheck
                            inner_sum_evals[0] +=
                                z_eq_r * wa[0][k].mul_0_optimized(wv_evals[0] - val_j_r[0][k]);
                            inner_sum_evals[1] +=
                                z_eq_r * wa_eval_2.mul_0_optimized(wv_evals[1] - val_eval_2);
                            inner_sum_evals[2] +=
                                z_eq_r * wa_eval_3.mul_0_optimized(wv_evals[2] - val_eval_3);
                        }

                        evals[0] += eq_r_prime_evals[0] * inner_sum_evals[0];
                        evals[1] += eq_r_prime_evals[1] * inner_sum_evals[1];
                        evals[2] += eq_r_prime_evals[2] * inner_sum_evals[2];
                    });

                evals
            })
            .reduce(
                || [F::zero(); DEGREE],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
        ]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_sumcheck.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind I
        I.par_iter_mut().for_each(|I_chunk| {
            // Note: A given row in an I_chunk may not be ordered by k after binding
            let mut next_bound_index = 0;
            let mut bound_indices: Vec<Option<usize>> = vec![None; K];

            for i in 0..I_chunk.len() {
                let (j_prime, k, inc_lt, inc) = I_chunk[i];
                if let Some(bound_index) = bound_indices[k] {
                    if I_chunk[bound_index].0 == j_prime / 2 {
                        // Neighbor was already processed
                        debug_assert!(j_prime % 2 == 1);
                        I_chunk[bound_index].2 += r_j * inc_lt;
                        I_chunk[bound_index].3 += inc;
                        continue;
                    }
                }
                // First time this k has been encountered
                let bound_value = if j_prime % 2 == 0 {
                    // (1 - r_j) * inc_lt + r_j * inc
                    inc_lt + r_j * (inc - inc_lt)
                } else {
                    r_j * inc_lt
                };
                I_chunk[next_bound_index] = (j_prime / 2, k, bound_value, inc);
                bound_indices[k] = Some(next_bound_index);
                next_bound_index += 1;
            }
            I_chunk.truncate(next_bound_index);
        });

        rayon::join(
            || wv.bind_parallel(r_j, BindingOrder::LowToHigh),
            || eq_r_prime.bind_parallel(r_j, BindingOrder::LowToHigh),
        );

        #[cfg(test)]
        {
            val_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            ra_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            wa_test.bind_parallel(r_j, BindingOrder::LowToHigh);

            // Check that row indices of I are non-decreasing
            let mut current_row = 0;
            for I_chunk in I.iter() {
                for (row, _, _, _) in I_chunk {
                    if *row != current_row {
                        assert_eq!(*row, current_row + 1);
                        current_row = *row;
                    }
                }
            }
        }

        // Update A for this round (see Equation 55)
        let (A_left, A_right) = A.split_at_mut(1 << round);
        A_left
            .par_iter_mut()
            .zip(A_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r_j;
                *x -= *y;
            });
    }

    // At this point I has been bound to a point where each chunk contains a single row,
    // so we might as well materialize the full `ra`, `wa`, and `Val` polynomials and perform
    // standard sumcheck directly using those polynomials.

    // TODO(moodlezoup): Generate these polynomials in address-major order and bind variables
    // from high-to-low for remaining rounds?
    let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
    ra.par_chunks_mut(num_chunks)
        .enumerate()
        .for_each(|(k, ra_chunk)| {
            for (j, address) in read_addresses.iter().enumerate() {
                if *address == k {
                    let j_unbound = j / chunk_size;
                    let j_bound = j % chunk_size;
                    ra_chunk[j_unbound] += A[j_bound];
                }
            }
        });
    let mut ra = MultilinearPolynomial::from(ra);

    let mut wa: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
    wa.par_chunks_mut(num_chunks)
        .enumerate()
        .for_each(|(k, wa_chunk)| {
            for (j, address) in write_addresses.iter().enumerate() {
                if *address == k {
                    let j_unbound = j / chunk_size;
                    let j_bound = j % chunk_size;
                    wa_chunk[j_unbound] += A[j_bound];
                }
            }
        });
    let mut wa = MultilinearPolynomial::from(wa);

    let mut val: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
    val.par_chunks_mut(num_chunks)
        .enumerate()
        .for_each(|(k, val_chunk)| {
            let mut j = 0;
            I.iter().enumerate().for_each(|(chunk_index, I_chunk)| {
                let mut val_k_j_0 = checkpoints[chunk_index][k];
                for I_row in I_chunk.chunk_by(|a, b| a.0 == b.0) {
                    match I_row.iter().find(|inc| inc.1 == k) {
                        Some(inc) => {
                            let (_, _, inc_lt, inc) = *inc;
                            val_chunk[j] = val_k_j_0 + inc_lt;
                            val_k_j_0 += inc;
                        }
                        None => {
                            val_chunk[j] = val_k_j_0;
                        }
                    };
                    j += 1;
                }
            });
            debug_assert_eq!(j, val_chunk.len());
        });
    let mut val = MultilinearPolynomial::from(val);

    #[cfg(test)]
    {
        // `ra` should match `ra_test`
        assert_eq!(ra.len(), ra_test.len());
        for i in 0..ra.len() {
            assert_eq!(ra.get_bound_coeff(i), ra_test.get_bound_coeff(i));
        }
        // `wa` should match `wa_test`
        assert_eq!(wa.len(), wa_test.len());
        for i in 0..wa.len() {
            assert_eq!(wa.get_bound_coeff(i), wa_test.get_bound_coeff(i));
        }
        // `val` should match `val_test`
        assert_eq!(val.len(), val_test.len());
        for i in 0..val.len() {
            assert_eq!(val.get_bound_coeff(i), val_test.get_bound_coeff(i));
        }
    }

    // Remaining rounds of sumcheck
    for _round in 0..num_rounds - chunk_size.log_2() {
        let univariate_poly_evals: [F; 3] = if eq_r_prime.len() > 1 {
            // Not done binding cycle variables yet
            (0..eq_r_prime.len() / 2)
                .into_par_iter()
                .map(|j| {
                    let eq_r_prime_evals =
                        eq_r_prime.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                    let wv_evals = wv.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);

                    let inner_sum_evals: [F; 3] = (0..K)
                        .into_par_iter()
                        .map(|k| {
                            let index = k * eq_r_prime.len() / 2 + j;
                            let ra_evals =
                                ra.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh);
                            let wa_evals =
                                wa.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh);
                            let val_evals =
                                val.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh);

                            let z_eq_r = z * eq_r.get_coeff(k);

                            [
                                ra_evals[0] * val_evals[0]
                                    + z_eq_r * wa_evals[0] * (wv_evals[0] - val_evals[0]),
                                ra_evals[1] * val_evals[1]
                                    + z_eq_r * wa_evals[1] * (wv_evals[1] - val_evals[1]),
                                ra_evals[2] * val_evals[2]
                                    + z_eq_r * wa_evals[2] * (wv_evals[2] - val_evals[2]),
                            ]
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

                    [
                        eq_r_prime_evals[0] * inner_sum_evals[0],
                        eq_r_prime_evals[1] * inner_sum_evals[1],
                        eq_r_prime_evals[2] * inner_sum_evals[2],
                    ]
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
                )
        } else {
            // Cycle variables are fully bound, so:
            // eq(r', r_cycle) is a constant
            let eq_r_prime_eval = eq_r_prime.final_sumcheck_claim();
            // ...and wv(r_cycle) is a constant
            let wv_eval = wv.final_sumcheck_claim();

            let evals = (0..ra.len() / 2)
                .into_par_iter()
                .map(|k| {
                    let eq_r_evals = eq_r.sumcheck_evals(k, DEGREE, BindingOrder::LowToHigh);
                    let ra_evals = ra.sumcheck_evals(k, DEGREE, BindingOrder::LowToHigh);
                    let wa_evals = wa.sumcheck_evals(k, DEGREE, BindingOrder::LowToHigh);
                    let val_evals = val.sumcheck_evals(k, DEGREE, BindingOrder::LowToHigh);

                    [
                        ra_evals[0] * val_evals[0]
                            + z * eq_r_evals[0] * wa_evals[0] * (wv_eval - val_evals[0]),
                        ra_evals[1] * val_evals[1]
                            + z * eq_r_evals[1] * wa_evals[1] * (wv_eval - val_evals[1]),
                        ra_evals[2] * val_evals[2]
                            + z * eq_r_evals[2] * wa_evals[2] * (wv_eval - val_evals[2]),
                    ]
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
            [
                eq_r_prime_eval * evals[0],
                eq_r_prime_eval * evals[1],
                eq_r_prime_eval * evals[2],
            ]
        };

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
        ]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_sumcheck.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        if eq_r_prime.len() > 1 {
            // Bind a cycle variable j
            // Note that `eq_r` is a polynomial over only the address variables,
            // so it is not bound here
            [&mut ra, &mut wa, &mut wv, &mut val, &mut eq_r_prime]
                .into_par_iter()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        } else {
            // Bind an address variable k
            // Note that `wv` and `eq_r_prime` are polynomials over only the cycle
            // variables, so they are not bound here
            [&mut ra, &mut wa, &mut val, &mut eq_r]
                .into_par_iter()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    TwistProof {
        read_write_checking_sumcheck: SumcheckInstanceProof::new(compressed_polys),
        ra_claim: ra.final_sumcheck_claim(),
        rv_claim: rv_eval,
        wa_claim: wa.final_sumcheck_claim(),
        wv_claim: wv.final_sumcheck_claim(),
        val_claim: val.final_sumcheck_claim(),
        inc_claim: inc_eval,
    }
}

/// Implements the sumcheck prover for the Val-evaluation sumcheck described in
/// Section 8.1 and Appendix B of the Twist+Shout paper
/// TODO(moodlezoup): incorporate optimization from Appendix B.2
#[tracing::instrument(skip_all)]
pub fn prove_val_evaluation<F: JoltField, ProofTranscript: Transcript>(
    increments: Vec<(usize, i64)>,
    r_address: Vec<F>,
    r_cycle: Vec<F>,
    claimed_evaluation: F,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, F) {
    let T = r_cycle.len().pow2();

    // Compute the size-K table storing all eq(r_address, k) evaluations for
    // k \in {0, 1}^log(K)
    let eq_r_address = EqPolynomial::evals(&r_address);

    let span = tracing::span!(tracing::Level::INFO, "compute Inc");
    let _guard = span.enter();

    // Compute the Inc polynomial using the above table
    let inc: Vec<F> = increments
        .par_iter()
        .map(|(k, increment)| eq_r_address[*k] * F::from_i64(*increment))
        .collect();
    let mut inc = MultilinearPolynomial::from(inc);

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "compute E");
    let _guard = span.enter();

    let mut E: Vec<Vec<F>> = Vec::with_capacity(r_cycle.len() + 1);
    E.push(vec![F::one()]);
    for (i, r_i) in r_cycle.iter().enumerate() {
        let eq_table: Vec<F> = E[i]
            .par_iter()
            .flat_map(|eq_j_r| {
                let one_term = *eq_j_r * r_i;
                let zero_term = *eq_j_r - one_term;
                [zero_term, one_term]
            })
            .collect();
        E.push(eq_table);
    }

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "compute D");
    let _guard = span.enter();

    let mut D: Vec<Vec<F>> = Vec::with_capacity(r_cycle.len() + 1);
    D.push(vec![F::zero()]);
    for (i, r_i) in r_cycle.iter().enumerate() {
        let lt_table: Vec<F> = D[i]
            .par_iter()
            .zip(E[i].par_iter())
            .flat_map(|(D_i_x, E_i_x)| {
                let one_term = *D_i_x;
                let zero_term = *D_i_x + *r_i * E_i_x;
                [zero_term, one_term]
            })
            .collect();
        D.push(lt_table);
    }

    drop(_guard);
    drop(span);

    let mut lt = MultilinearPolynomial::from(D.pop().unwrap());

    let num_rounds = T.log_2();
    let mut previous_claim = claimed_evaluation;
    let mut r_cycle_prime: Vec<F> = Vec::with_capacity(num_rounds);

    const DEGREE: usize = 2;

    let span = tracing::span!(tracing::Level::INFO, "Val-evaluation sumcheck");
    let _guard = span.enter();

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _round in 0..num_rounds {
        #[cfg(test)]
        {
            let expected: F = (0..inc.len())
                .map(|j| inc.get_bound_coeff(j) * lt.get_bound_coeff(j))
                .sum::<F>();
            assert_eq!(
                expected, previous_claim,
                "Sumcheck sanity check failed in round {_round}"
            );
        }

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let univariate_poly_evals: [F; 2] = (0..inc.len() / 2)
            .into_par_iter()
            .map(|i| {
                let inc_evals = inc.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let lt_evals = lt.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [inc_evals[0] * lt_evals[0], inc_evals[1] * lt_evals[1]]
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

        drop(_inner_guard);
        drop(inner_span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_cycle_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || inc.bind_parallel(r_j, BindingOrder::LowToHigh),
            || lt.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let inc_claim = inc.final_sumcheck_claim();

    (
        SumcheckInstanceProof::new(compressed_polys),
        r_cycle_prime,
        inc_claim,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Fr;
    use ark_ff::Zero;
    use ark_std::test_rng;
    use rand_core::RngCore;

    #[test]
    fn val_evaluation_sumcheck() {
        const K: usize = 64;
        const T: usize = 1 << 8;

        let mut rng = test_rng();

        let increments: Vec<(usize, i64)> = (0..T)
            .map(|_| {
                let address = rng.next_u32() as usize % K;
                let increment = rng.next_u32() as i32 as i64;
                (address, increment)
            })
            .collect();

        // Compute the Val polynomial from increments
        let mut values = vec![Fr::zero(); K];
        let mut val: Vec<Fr> = Vec::with_capacity(K * T);
        for (k, increment) in increments.iter() {
            val.extend(values.iter());
            values[*k] += Fr::from_i64(*increment);
        }
        let val = MultilinearPolynomial::from(val);

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_address: Vec<Fr> = prover_transcript.challenge_vector(K.log_2());
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let val_evaluation = val.evaluate(&[r_cycle.clone(), r_address.clone()].concat());
        let (sumcheck_proof, _, _) = prove_val_evaluation(
            increments,
            r_address,
            r_cycle,
            val_evaluation,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _r_address: Vec<Fr> = verifier_transcript.challenge_vector(K.log_2());
        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        let verification_result =
            sumcheck_proof.verify(val_evaluation, T.log_2(), 2, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn read_write_checking_sumcheck_local() {
        const K: usize = 16;
        const T: usize = 1 << 8;

        let mut rng = test_rng();

        let mut registers = [0u32; K];
        let mut read_addresses: Vec<usize> = Vec::with_capacity(T);
        let mut read_values: Vec<u32> = Vec::with_capacity(T);
        let mut write_addresses: Vec<usize> = Vec::with_capacity(T);
        let mut write_values: Vec<u32> = Vec::with_capacity(T);
        let mut write_increments: Vec<i64> = Vec::with_capacity(T);
        for _ in 0..T {
            // Random read register
            let read_address = rng.next_u32() as usize % K;
            // Random write register
            let write_address = rng.next_u32() as usize % K;
            read_addresses.push(read_address);
            write_addresses.push(write_address);
            // Read the value currently in the read register
            read_values.push(registers[read_address]);
            // Random write value
            let write_value = rng.next_u32();
            write_values.push(write_value);
            // The increment is the difference between the new value and the old value
            let write_increment = (write_value as i64) - (registers[write_address] as i64);
            write_increments.push(write_increment);
            // Write the new value to the write register
            registers[write_address] = write_value;
        }

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r: Vec<Fr> = prover_transcript.challenge_vector(K.log_2());
        let r_prime: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let twist_proof = prove_read_write_checking_local(
            read_addresses,
            read_values,
            write_addresses,
            write_values,
            write_increments,
            &r,
            &r_prime,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _r: Vec<Fr> = verifier_transcript.challenge_vector(K.log_2());
        let _r_prime: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());
        let z: Fr = verifier_transcript.challenge_scalar();

        let initial_sumcheck_claim = twist_proof.rv_claim + z * twist_proof.inc_claim;

        let (sumcheck_claim, mut r_sumcheck) = twist_proof
            .read_write_checking_sumcheck
            .verify(
                initial_sumcheck_claim,
                T.log_2() + K.log_2(),
                3,
                &mut verifier_transcript,
            )
            .unwrap();
        r_sumcheck = r_sumcheck.into_iter().rev().collect();
        let (r_address, r_cycle) = r_sumcheck.split_at(K.log_2());

        // eq(r', r_cycle)
        let eq_eval_cycle = EqPolynomial::new(r_prime).evaluate(r_cycle);
        // eq(r, r_address)
        let eq_eval_address = EqPolynomial::new(r).evaluate(r_address);

        assert_eq!(
            eq_eval_cycle * twist_proof.ra_claim * twist_proof.val_claim
                + z * eq_eval_address
                    * eq_eval_cycle
                    * twist_proof.wa_claim
                    * (twist_proof.wv_claim - twist_proof.val_claim),
            sumcheck_claim
        );
    }
}
