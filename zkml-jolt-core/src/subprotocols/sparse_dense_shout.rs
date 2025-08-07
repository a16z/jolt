#![allow(clippy::too_many_arguments)]
use crate::jolt::execution_trace::WORD_SIZE;
use crate::jolt::execution_trace::{JoltONNXCycle, ONNXLookupQuery};
use jolt_core::subprotocols::sparse_dense_shout::{ExpandingTable, LookupBits};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::{
    field::JoltField,
    jolt::{
        instruction::InstructionLookup,
        lookup_table::{
            LookupTables,
            prefixes::{PrefixCheckpoint, PrefixEval, Prefixes},
        },
    },
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::{Endianness, IdentityPolynomial, OperandPolynomial, OperandSide},
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
        unipoly::{CompressedUniPoly, UniPoly},
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec, unsafe_zero_slice},
        transcript::{AppendToTranscript, Transcript},
    },
};
use onnx_tracer::constants::MAX_TENSOR_SIZE;
use onnx_tracer::trace_types::InterleavedBitsMarker;
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};

#[allow(clippy::type_complexity)]
pub fn prove_sparse_dense_shout<F: JoltField, ProofTranscript: Transcript>(
    trace: &[JoltONNXCycle],
    r_cycle: &[F],
    transcript: &mut ProofTranscript,
) -> (
    SumcheckInstanceProof<F, ProofTranscript>,
    F,
    [F; 4],
    F,
    Vec<F>,
    Vec<F>,
) {
    let log_K: usize = 2 * WORD_SIZE;
    let log_m = log_K / 4;
    let m = log_m.pow2();

    let T = trace.len() * MAX_TENSOR_SIZE;
    let log_T = T.log_2();
    debug_assert_eq!(r_cycle.len(), log_T);

    let num_rounds = log_K + log_T;
    let mut r: Vec<F> = Vec::with_capacity(num_rounds);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let span = tracing::span!(tracing::Level::INFO, "compute lookup indices");
    let _guard = span.enter();
    let lookup_indices: Vec<LookupBits> = trace
        .par_iter()
        .map(|cycle| {
            ONNXLookupQuery::<WORD_SIZE>::to_lookup_index(cycle)
                .iter()
                .map(|&index| LookupBits::new(index, log_K))
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect();

    drop(_guard);
    drop(span);

    let eq_r_prime_evals = EqPolynomial::evals(r_cycle);
    let mut u_evals = eq_r_prime_evals.clone();

    let mut prefix_checkpoints: Vec<PrefixCheckpoint<F>> = vec![None.into(); Prefixes::COUNT];
    let mut v = ExpandingTable::new(m);

    let gamma: F = transcript.challenge_scalar();
    let gamma_squared = gamma.square();

    let span = tracing::span!(tracing::Level::INFO, "compute input claim");
    let _guard = span.enter();
    let rv_input_claim: F = lookup_indices
        .par_iter()
        .zip(u_evals.par_iter())
        .enumerate()
        .map(|(i, (k, u))| {
            let cycle = trace.get(i / MAX_TENSOR_SIZE).unwrap();
            match cycle.lookup_table() {
                Some(table) => u.mul_u64(table.materialize_entry(k.into())),
                None => F::zero(),
            }
        })
        .sum();
    // TODO: these claims should be connected from spartan
    let (right_operand_evals, left_operand_evals): (Vec<u64>, Vec<u64>) = trace
        .par_iter()
        .flat_map(ONNXLookupQuery::<WORD_SIZE>::to_lookup_operands)
        .unzip();

    let right_operand_claim = MultilinearPolynomial::from(right_operand_evals).evaluate(r_cycle);
    let left_operand_claim = MultilinearPolynomial::from(left_operand_evals).evaluate(r_cycle);

    let input_claim =
        rv_input_claim + gamma * right_operand_claim + gamma_squared * left_operand_claim;
    drop(_guard);
    drop(span);

    let mut previous_claim = input_claim;

    let mut j: usize = 0;
    let mut ra: Vec<MultilinearPolynomial<F>> = Vec::with_capacity(4);

    let lookup_tables: Vec<_> = LookupTables::<WORD_SIZE>::iter().collect();
    let mut suffix_polys: Vec<Vec<DensePolynomial<F>>> = lookup_tables
        .par_iter()
        .map(|table| {
            table
                .suffixes()
                .par_iter()
                .map(|_| DensePolynomial::new(unsafe_allocate_zero_vec(m)))
                .collect()
        })
        .collect();

    let mut prefix_registry = PrefixRegistry::new();

    let right_operand_poly = OperandPolynomial::new(log_K, OperandSide::Right);
    let left_operand_poly = OperandPolynomial::new(log_K, OperandSide::Left);
    let identity_poly: IdentityPolynomial<F> =
        IdentityPolynomial::new_with_endianness(log_K, Endianness::Big);
    let mut right_operand_ps =
        PrefixSuffixDecomposition::new(Box::new(right_operand_poly), m.log_2(), log_K);
    let mut left_operand_ps =
        PrefixSuffixDecomposition::new(Box::new(left_operand_poly), m.log_2(), log_K);
    let mut identity_ps = PrefixSuffixDecomposition::new(Box::new(identity_poly), m.log_2(), log_K);

    let span = tracing::span!(tracing::Level::INFO, "Compute lookup_indices_by_table");
    let _guard = span.enter();

    let lookup_indices_by_table: Vec<_> = lookup_tables
        .par_iter()
        .map(|table| {
            let table_lookups: Vec<_> = lookup_indices
                .iter()
                .enumerate()
                .filter_map(|(j, k)| {
                    let cycle = trace.get(j / MAX_TENSOR_SIZE).unwrap();
                    match cycle.lookup_table() {
                        Some(lookup) => {
                            if LookupTables::<WORD_SIZE>::enum_index(&lookup)
                                == LookupTables::enum_index(table)
                            {
                                Some((j, k))
                            } else {
                                None
                            }
                        }
                        None => None,
                    }
                })
                .collect();
            table_lookups
        })
        .collect();
    let (lookup_indices_uninterleave, lookup_indices_identity): (Vec<_>, Vec<_>) = lookup_indices
        .par_iter()
        .enumerate()
        .partition_map(|(idx, item)| {
            let cycle = trace.get(idx / MAX_TENSOR_SIZE).unwrap();
            if cycle.instr().to_circuit_flags().is_interleaved_operands() {
                itertools::Either::Left((idx, item))
            } else {
                itertools::Either::Right((idx, item))
            }
        });

    drop(_guard);
    drop(span);

    for phase in 0..4 {
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
                    let (prefix, _) = k.split((4 - phase) * log_m);
                    let k_bound: usize = prefix % m;
                    *u *= v[k_bound];
                });
        }

        let suffix_len = (3 - phase) * log_m;

        // Initialize suffix poly for each suffix
        let suffix_poly_span = tracing::span!(tracing::Level::INFO, "Compute suffix polys");
        let _suffix_poly_guard = suffix_poly_span.enter();

        rayon::scope(|s| {
            s.spawn(|_| {
                lookup_tables
                    .par_iter()
                    .zip(suffix_polys.par_iter_mut())
                    .zip(lookup_indices_by_table.par_iter())
                    .for_each(|((table, polys), lookup_indices)| {
                        table
                            .suffixes()
                            .par_iter()
                            .zip(polys.par_iter_mut())
                            .for_each(|(suffix, poly)| {
                                if phase != 0 {
                                    // Reset polynomial
                                    poly.len = m;
                                    poly.num_vars = poly.len.log_2();
                                    unsafe_zero_slice(&mut poly.Z);
                                }

                                for (j, k) in lookup_indices.iter() {
                                    let (prefix_bits, suffix_bits) = k.split(suffix_len);
                                    let t = suffix.suffix_mle::<WORD_SIZE>(suffix_bits);
                                    if t != 0 {
                                        let u = u_evals[*j];
                                        poly.Z[prefix_bits % m] += u.mul_u64(t as u64);
                                    }
                                }
                            });
                    });
            });
            s.spawn(|_| right_operand_ps.init_Q(&u_evals, lookup_indices_uninterleave.iter()));
            s.spawn(|_| left_operand_ps.init_Q(&u_evals, lookup_indices_uninterleave.iter()));
            s.spawn(|_| identity_ps.init_Q(&u_evals, lookup_indices_identity.iter()));
        });
        identity_ps.init_P(&mut prefix_registry);
        right_operand_ps.init_P(&mut prefix_registry);
        left_operand_ps.init_P(&mut prefix_registry);

        drop(_suffix_poly_guard);
        drop(suffix_poly_span);

        v.reset(F::one());

        for _round in 0..log_m {
            let span = tracing::span!(tracing::Level::INFO, "sparse-dense sumcheck round");
            let _guard = span.enter();

            let univariate_poly_evals = compute_sumcheck_prover_message::<WORD_SIZE, F>(
                &prefix_checkpoints,
                &suffix_polys,
                &identity_ps,
                &right_operand_ps,
                &left_operand_ps,
                gamma,
                &r,
                j,
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

            let binding_span = tracing::span!(tracing::Level::INFO, "binding");
            let _binding_guard = binding_span.enter();

            rayon::scope(|s| {
                s.spawn(|_| {
                    suffix_polys.par_iter_mut().for_each(|polys| {
                        polys
                            .par_iter_mut()
                            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow))
                    });
                });
                s.spawn(|_| identity_ps.bind(r_j));
                s.spawn(|_| right_operand_ps.bind(r_j));
                s.spawn(|_| left_operand_ps.bind(r_j));
                s.spawn(|_| v.update(r_j));
            });

            {
                if r.len().is_multiple_of(2) {
                    Prefixes::update_checkpoints::<WORD_SIZE, F>(
                        &mut prefix_checkpoints,
                        r[r.len() - 2],
                        r[r.len() - 1],
                        j,
                    );
                }
            }

            j += 1;
        }

        let span = tracing::span!(tracing::Level::INFO, "cache ra_i");
        let _guard = span.enter();

        let ra_i: Vec<F> = lookup_indices
            .par_iter()
            .map(|k| {
                let (prefix, _) = k.split(suffix_len);
                let k_bound: usize = prefix % m;
                v[k_bound]
            })
            .collect();
        ra.push(MultilinearPolynomial::from(ra_i));

        prefix_registry.update_checkpoints();
    }

    drop_in_background_thread(suffix_polys);

    let mut eq_r_prime = MultilinearPolynomial::from(eq_r_prime_evals.clone());

    let span = tracing::span!(
        tracing::Level::INFO,
        "compute combined_instruction_val_poly"
    );
    let _guard = span.enter();

    let prefixes: Vec<PrefixEval<F>> = prefix_checkpoints
        .into_iter()
        .map(|checkpoint| checkpoint.unwrap())
        .collect();

    let mut combined_instruction_val_poly: Vec<F> = unsafe_allocate_zero_vec(T);
    combined_instruction_val_poly
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, val)| {
            let cycle = trace.get(i / MAX_TENSOR_SIZE).unwrap();
            let table: Option<LookupTables<WORD_SIZE>> = cycle.lookup_table();
            if let Some(table) = table {
                let suffixes: Vec<_> = table
                    .suffixes()
                    .iter()
                    .map(|suffix| {
                        F::from_u32(suffix.suffix_mle::<WORD_SIZE>(LookupBits::new(0, 0)))
                    })
                    .collect();
                *val += table.combine(&prefixes, &suffixes);
            }

            if cycle.instr().to_circuit_flags().is_interleaved_operands() {
                *val += gamma * prefix_registry.checkpoints[Prefix::RightOperand].unwrap()
                    + gamma_squared * prefix_registry.checkpoints[Prefix::LeftOperand].unwrap();
            } else {
                *val += gamma_squared * prefix_registry.checkpoints[Prefix::Identity].unwrap();
            }
        });
    let mut combined_instruction_val_poly =
        MultilinearPolynomial::from(combined_instruction_val_poly);

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "last log(T) sumcheck rounds");
    let _guard = span.enter();

    // TODO(moodlezoup): Implement optimization from Section 6.2.2 "An optimization leveraging small memory size"

    for _round in 0..log_T {
        let span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _guard = span.enter();

        let univariate_poly_evals: [F; 6] = (0..eq_r_prime.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = eq_r_prime.sumcheck_evals(i, 6, BindingOrder::HighToLow);
                let ra_0_evals = ra[0].sumcheck_evals(i, 6, BindingOrder::HighToLow);
                let ra_1_evals = ra[1].sumcheck_evals(i, 6, BindingOrder::HighToLow);
                let ra_2_evals = ra[2].sumcheck_evals(i, 6, BindingOrder::HighToLow);
                let ra_3_evals = ra[3].sumcheck_evals(i, 6, BindingOrder::HighToLow);
                let val_evals =
                    combined_instruction_val_poly.sumcheck_evals(i, 6, BindingOrder::HighToLow);

                std::array::from_fn(|i| {
                    eq_evals[i]
                        * ra_0_evals[i]
                        * ra_1_evals[i]
                        * ra_2_evals[i]
                        * ra_3_evals[i]
                        * val_evals[i]
                })
            })
            .reduce(
                || [F::zero(); 6],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                        running[3] + new[3],
                        running[4] + new[4],
                        running[5] + new[5],
                    ]
                },
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
            univariate_poly_evals[3],
            univariate_poly_evals[4],
            univariate_poly_evals[5],
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
            .chain([&mut combined_instruction_val_poly].into_par_iter())
            .chain([&mut eq_r_prime].into_par_iter())
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
    }

    let span = tracing::span!(tracing::Level::INFO, "compute flag claims");
    let _guard = span.enter();

    let r_cycle_prime = &r[r.len() - log_T..];
    let eq_r_cycle_prime = EqPolynomial::evals(r_cycle_prime);

    // Evaluate each flag polynomial on `r_cycle_prime` by computing its
    // dot product with EQ(r_cycle_prime, j)
    let flag_claims: Vec<_> = lookup_indices_by_table
        .into_par_iter()
        .map(|table_lookups| {
            table_lookups
                .into_iter()
                .map(|(j, _)| eq_r_cycle_prime[j])
                .sum::<F>()
        })
        .collect();
    let add_mul_sub_claims = lookup_indices_identity
        .into_par_iter()
        .map(|(j, _)| eq_r_cycle_prime[j])
        .sum::<F>();
    drop(_guard);
    drop(span);

    let ra_claims = [
        ra[0].final_sumcheck_claim(),
        ra[1].final_sumcheck_claim(),
        ra[2].final_sumcheck_claim(),
        ra[3].final_sumcheck_claim(),
    ];

    drop_in_background_thread((combined_instruction_val_poly, eq_r_prime, ra));

    (
        SumcheckInstanceProof::new(compressed_polys),
        input_claim,
        ra_claims,
        add_mul_sub_claims,
        flag_claims,
        eq_r_prime_evals,
    )
}

pub fn verify_sparse_dense_shout<
    const WORD_SIZE: usize,
    F: JoltField,
    ProofTranscript: Transcript,
>(
    proof: &SumcheckInstanceProof<F, ProofTranscript>,
    log_T: usize,
    r_cycle: Vec<F>,
    rv_claim: F,
    ra_claims: [F; 4],
    is_add_mul_sub_flag_claim: F,
    flag_claims: &[F],
    transcript: &mut ProofTranscript,
) -> Result<(), ProofVerifyError> {
    let log_K = 2 * WORD_SIZE;
    let first_log_K_rounds = SumcheckInstanceProof::new(proof.compressed_polys[..log_K].to_vec());
    let last_log_T_rounds = SumcheckInstanceProof::new(proof.compressed_polys[log_K..].to_vec());

    let gamma: F = transcript.challenge_scalar();
    let gamma_squared = gamma.square();

    // The first log(K) rounds' univariate polynomials are degree 2
    let (sumcheck_claim, r_address) = first_log_K_rounds.verify(rv_claim, log_K, 2, transcript)?;
    // The last log(T) rounds' univariate polynomials are degree 6
    let (sumcheck_claim, r_cycle_prime) =
        last_log_T_rounds.verify(sumcheck_claim, log_T, 6, transcript)?;

    let val_evals: Vec<_> = LookupTables::<WORD_SIZE>::iter()
        .map(|table| table.evaluate_mle(&r_address))
        .collect();
    let eq_eval_cycle = EqPolynomial::mle(&r_cycle, &r_cycle_prime);

    let rv_val_claim = flag_claims
        .iter()
        .zip(val_evals.iter())
        .map(|(flag, val)| *flag * val)
        .sum::<F>();

    let right_operand_eval = OperandPolynomial::new(log_K, OperandSide::Right).evaluate(&r_address);
    let left_operand_eval = OperandPolynomial::new(log_K, OperandSide::Left).evaluate(&r_address);
    let identity_poly_eval =
        IdentityPolynomial::new_with_endianness(log_K, Endianness::Big).evaluate(&r_address);

    let val_claim = rv_val_claim
        + (F::one() - is_add_mul_sub_flag_claim)
            * (gamma * right_operand_eval + gamma_squared * left_operand_eval)
        + gamma_squared * is_add_mul_sub_flag_claim * identity_poly_eval;

    assert_eq!(
        eq_eval_cycle * ra_claims.iter().product::<F>() * val_claim,
        sumcheck_claim,
        "Read-checking sumcheck failed"
    );

    Ok(())
}

/// Computes the bit-length of the suffix, for the current (`j`th) round
/// of sumcheck.
pub fn current_suffix_len(log_K: usize, j: usize) -> usize {
    // Number of sumcheck rounds per "phase" of sparse-dense sumcheck.
    let phase_length = log_K / 4;
    // The suffix length is 3/4 * log_K at the beginning and shrinks by
    // log_K / 4 after each phase.
    log_K - (j / phase_length + 1) * phase_length
}

/// Compute the sumcheck prover message in round `j` using the prefix-suffix
/// decomposition. In the first log(K) rounds of sumcheck, while we're
/// binding the "address" variables (and the "cycle" variables remain unbound),
/// the univariate polynomial computed each round is degree 2.
///
/// To see this, observe that:
///   eq(r', j) * ra_1(k_1, j) * ra_2(k_2, j) * ra_3(k_3, j) * ra_4(k_4, j)
/// is multilinear in k, since ra_1, ra_2, ra_3, and ra_4 are polynomials in
/// non-overlapping variables of k (and the eq term doesn't involve k at all).
/// Val(k) is clearly multilinear in k, so the whole summand
///   eq(r', j) (\prod_i ra_i(k_i, j)) * \sum_l flag_l * Val_l(k)
/// is degree 2 in each "address" variable k.
#[tracing::instrument(skip_all)]
fn compute_sumcheck_prover_message<const WORD_SIZE: usize, F: JoltField>(
    prefix_checkpoints: &[PrefixCheckpoint<F>],
    suffix_polys: &[Vec<DensePolynomial<F>>],
    identity_ps: &PrefixSuffixDecomposition<F, 2>,
    right_operand_ps: &PrefixSuffixDecomposition<F, 2>,
    left_operand_ps: &PrefixSuffixDecomposition<F, 2>,
    gamma: F,
    r: &[F],
    j: usize,
) -> [F; 2] {
    let mut read_checking = [F::zero(), F::zero()];
    let mut raf = [F::zero(), F::zero()];

    rayon::join(
        || {
            read_checking =
                prover_msg_read_checking::<WORD_SIZE, _>(prefix_checkpoints, suffix_polys, r, j);
        },
        || {
            raf = prover_msg_raf(identity_ps, right_operand_ps, left_operand_ps, gamma);
        },
    );

    [read_checking[0] + raf[0], read_checking[1] + raf[1]]
}

fn prover_msg_raf<F: JoltField>(
    identity_ps: &PrefixSuffixDecomposition<F, 2>,
    right_operand_ps: &PrefixSuffixDecomposition<F, 2>,
    left_operand_ps: &PrefixSuffixDecomposition<F, 2>,
    gamma: F,
) -> [F; 2] {
    let len = identity_ps.Q_len();
    let gamma_squared = gamma.square();
    let (left_0, left_2, right_0, right_2) = (0..len / 2)
        .into_par_iter()
        .map(|b| {
            let (i0, i2) = identity_ps.sumcheck_evals(b);
            let (r0, r2) = right_operand_ps.sumcheck_evals(b);
            let (l0, l2) = left_operand_ps.sumcheck_evals(b);
            (i0 + l0, i2 + l2, r0, r2)
        })
        .reduce(
            || (F::zero(), F::zero(), F::zero(), F::zero()),
            |running, new| {
                (
                    running.0 + new.0,
                    running.1 + new.1,
                    running.2 + new.2,
                    running.3 + new.3,
                )
            },
        );
    [
        gamma * right_0 + gamma_squared * left_0,
        gamma * right_2 + gamma_squared * left_2,
    ]
}

fn prover_msg_read_checking<const WORD_SIZE: usize, F: JoltField>(
    prefix_checkpoints: &[PrefixCheckpoint<F>],
    suffix_polys: &[Vec<DensePolynomial<F>>],
    r: &[F],
    j: usize,
) -> [F; 2] {
    let lookup_tables: Vec<_> = LookupTables::<WORD_SIZE>::iter().collect();

    let len = suffix_polys[0][0].len();
    let log_len = len.log_2();

    let r_x = if j % 2 == 1 { r.last().copied() } else { None };

    let (eval_0, eval_2_left, eval_2_right) = (0..len / 2)
        .into_par_iter()
        .flat_map_iter(|b| {
            let b = LookupBits::new(b as u64, log_len - 1);
            let prefixes_c0: Vec<_> = Prefixes::iter()
                .map(|prefix| prefix.prefix_mle::<WORD_SIZE, F>(prefix_checkpoints, r_x, 0, b, j))
                .collect();
            let prefixes_c2: Vec<_> = Prefixes::iter()
                .map(|prefix| prefix.prefix_mle::<WORD_SIZE, F>(prefix_checkpoints, r_x, 2, b, j))
                .collect();
            lookup_tables
                .iter()
                .zip(suffix_polys.iter())
                .map(move |(table, suffixes)| {
                    let suffixes_left: Vec<_> =
                        suffixes.iter().map(|suffix| suffix[b.into()]).collect();
                    let suffixes_right: Vec<_> = suffixes
                        .iter()
                        .map(|suffix| suffix[usize::from(b) + len / 2])
                        .collect();
                    (
                        table.combine(&prefixes_c0, &suffixes_left),
                        table.combine(&prefixes_c2, &suffixes_left),
                        table.combine(&prefixes_c2, &suffixes_right),
                    )
                })
        })
        .reduce(
            || (F::zero(), F::zero(), F::zero()),
            |running, new| (running.0 + new.0, running.1 + new.1, running.2 + new.2),
        );
    [eval_0, eval_2_right + eval_2_right - eval_2_left]
}

#[cfg(test)]
mod tests {
    use crate::jolt::execution_trace::jolt_execution_trace;

    use super::*;
    use ark_bn254::Fr;
    use jolt_core::utils::transcript::KeccakTranscript;
    use onnx_tracer::trace_types::{ONNXCycle, ONNXOpcode};
    use rand::{SeedableRng, rngs::StdRng};

    const WORD_SIZE: usize = 32;
    const TRACE_LEN: usize = 1 << 8;

    fn test_sparse_dense_shout(opcode: ONNXOpcode) {
        let LOG_T: usize = (TRACE_LEN * MAX_TENSOR_SIZE).log_2();
        let mut rng = StdRng::from_seed([0u8; 32]);

        let mut trace = Vec::with_capacity(TRACE_LEN);
        trace.resize(TRACE_LEN, ONNXCycle::random(opcode, &mut rng));
        let execution_trace = jolt_execution_trace(trace);
        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(LOG_T);

        let (proof, rv_claim, ra_claims, add_mul_sub_claim, flag_claims, _) =
            prove_sparse_dense_shout::<_, _>(&execution_trace, &r_cycle, &mut prover_transcript);

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(LOG_T);

        let verification_result = verify_sparse_dense_shout::<WORD_SIZE, _, _>(
            &proof,
            LOG_T,
            r_cycle,
            rv_claim,
            ra_claims,
            add_mul_sub_claim,
            &flag_claims,
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
        test_sparse_dense_shout(ONNXOpcode::Add);
    }

    #[test]
    fn test_sub() {
        test_sparse_dense_shout(ONNXOpcode::Sub);
    }

    #[test]
    fn test_mul() {
        test_sparse_dense_shout(ONNXOpcode::Mul);
    }

    #[test]
    fn test_asserteq() {
        test_sparse_dense_shout(ONNXOpcode::VirtualAssertEq);
    }

    #[test]
    fn test_advice() {
        test_sparse_dense_shout(ONNXOpcode::VirtualAdvice);
    }

    #[test]
    fn test_assertvaliddiv0() {
        test_sparse_dense_shout(ONNXOpcode::VirtualAssertValidDiv0);
    }

    #[test]
    fn test_assertvalidsignedremainder() {
        test_sparse_dense_shout(ONNXOpcode::VirtualAssertValidSignedRemainder);
    }

    #[test]
    fn test_move() {
        test_sparse_dense_shout(ONNXOpcode::VirtualMove);
    }
}
