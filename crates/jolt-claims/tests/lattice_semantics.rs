//! Semantic integration tests for the lattice module: build concrete one-hot
//! witness data and check the identities the relations claim — native OneHotTrace
//! member shape and fused-increment chunk semantics against concrete
//! multilinear evaluations.

use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::lattice::geometry::balanced_inc_value;
use jolt_claims::protocols::jolt::lattice::{
    one_hot_trace_columns, BalancedIncChunking, OneHotTraceShape,
};
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::{Fr, Ring};
use jolt_poly::{boolean_point_msb, eq_index_msb, EqPolynomial, Polynomial};
fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

/// MLE evaluation via the library's own (msb-first) convention — the same one
/// production code uses, so the tests pin the packing against it.
fn eval_mle(evals: &[Fr], point: &[Fr]) -> Fr {
    Polynomial::new(evals.to_vec()).evaluate(point)
}

/// A deterministic, non-boolean evaluation point (distinct small primes).
fn point(len: usize, seed: u64) -> Vec<Fr> {
    const PRIMES: [u64; 16] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59];
    (0..len)
        .map(|i| fr(PRIMES[(i + seed as usize) % PRIMES.len()] + seed))
        .collect()
}

/// One-hot evaluations over `(hot value ‖ instance)`: index
/// `(hot << log_rows) | row`.
fn one_hot_evals(value_bits: usize, log_rows: usize, hot: &[usize]) -> Vec<Fr> {
    assert_eq!(hot.len(), 1 << log_rows);
    let mut data = vec![fr(0); 1 << (value_bits + log_rows)];
    for (row, &value) in hot.iter().enumerate() {
        assert!(value < (1 << value_bits));
        data[(value << log_rows) | row] = fr(1);
    }
    data
}

fn digit_zero_evals(value_bits: usize, log_rows: usize, hot: &[usize]) -> Vec<Fr> {
    let mut data = one_hot_evals(value_bits, log_rows, hot);
    for (row, value) in hot.iter().copied().enumerate() {
        if value == 0 {
            data[row] = fr(0);
        }
    }
    data
}

/// Every semantic column uses the same `K x T` domain before the digit-zero row is
/// omitted from the commitment.
#[test]
#[expect(clippy::unwrap_used)]
fn one_hot_trace_columns_share_a_uniform_semantic_domain() {
    let log_t = 3;
    let log_k_chunk = 4;
    let shape = OneHotTraceShape {
        ra_layout: JoltRaPolynomialLayout::new(32, 1, 1).unwrap(),
        log_t,
        log_k_chunk,
    };
    let members = one_hot_trace_columns(&shape).unwrap();

    let chunk_count = BalancedIncChunking::new(log_k_chunk).unwrap().chunk_count();
    assert_eq!(chunk_count, 16);
    assert_eq!(members.len(), 51);
    assert_eq!(members[0], JoltCommittedPolynomial::InstructionRa(0));
    assert_eq!(members[48], JoltCommittedPolynomial::BalancedIncCarry);
    assert_eq!(members.last(), Some(&JoltCommittedPolynomial::RamRa(0)));

    for (index, polynomial) in members.iter().enumerate() {
        let hot = if *polynomial == JoltCommittedPolynomial::BalancedIncCarry {
            (0..1 << log_t).map(|t| t % 2).collect::<Vec<_>>()
        } else {
            (0..1 << log_t)
                .map(|t| (7 * t + 3 * index + 1) % (1 << log_k_chunk))
                .collect::<Vec<_>>()
        };
        let data = one_hot_evals(log_k_chunk, log_t, &hot);
        assert_eq!(data.len(), 1 << (log_k_chunk + log_t));
        for cycle in 0..1 << log_t {
            let hamming_weight = (0..1 << log_k_chunk)
                .map(|address| data[(address << log_t) | cycle])
                .sum::<Fr>();
            assert_eq!(hamming_weight, fr(1), "{polynomial:?}, cycle {cycle}");
        }
    }
}

#[test]
fn digit_zero_reconstruction_matches_semantic_one_hot_at_random_points() {
    let log_t = 3;
    let value_bits = 4;
    let rows = 1 << log_t;
    let r_address = point(value_bits, 2);
    let r_cycle = point(log_t, 5);
    let eq_cycle = EqPolynomial::<Fr>::evals(&r_cycle, None);
    let eq_zero = eq_index_msb::<Fr>(&r_address, 0);

    for hot in [
        vec![
            Some(0),
            Some(3),
            Some(0),
            Some(15),
            Some(1),
            Some(0),
            Some(8),
            Some(2),
        ],
        vec![
            None,
            Some(0),
            Some(7),
            None,
            Some(1),
            Some(0),
            None,
            Some(12),
        ],
    ] {
        let mut semantic = vec![fr(0); 1 << (value_bits + log_t)];
        let mut sparse = semantic.clone();
        let mut activation = vec![fr(0); rows];
        for (cycle, row) in hot.iter().copied().enumerate() {
            if let Some(row) = row {
                semantic[(row << log_t) | cycle] = fr(1);
                activation[cycle] = fr(1);
                if row != 0 {
                    sparse[(row << log_t) | cycle] = fr(1);
                }
            }
        }

        let full_point = [r_address.as_slice(), r_cycle.as_slice()].concat();
        let semantic_eval = eval_mle(&semantic, &full_point);
        let activation_eval = eval_mle(&activation, &r_cycle);
        let sparse_by_row = (0..1 << value_bits)
            .map(|row| {
                (0..rows)
                    .map(|cycle| eq_cycle[cycle] * sparse[(row << log_t) | cycle])
                    .sum::<Fr>()
            })
            .collect::<Vec<_>>();
        let recentered = eq_zero * activation_eval
            + sparse_by_row
                .iter()
                .enumerate()
                .map(|(row, value)| {
                    *value * (eq_index_msb::<Fr>(&r_address, row as u128) - eq_zero)
                })
                .sum::<Fr>();
        assert_eq!(recentered, semantic_eval);
    }
}

/// Centered radix digits and their signed carry reconstruct the fused
/// increment when digit-zero entries are absent from the commitment.
#[test]
#[expect(clippy::unwrap_used)]
fn balanced_chunk_decomposition_reconstructs_signed_increments() {
    let log_t = 3;
    let chunking = BalancedIncChunking::new(8).unwrap();
    let count = chunking.chunk_count();
    assert_eq!(count, 8);

    let values: [i128; 8] = [
        5,
        -7,
        0,
        (1 << 63) - 1,
        -(1 << 63),
        123_456_789,
        -987_654_321,
        0,
    ];

    let radix = 1i128 << chunking.chunk_width();
    let bias = (radix / 2) * (((1i128 << 64) - 1) / (radix - 1));
    let mask = radix - 1;
    let mut chunk_hot = vec![vec![0usize; values.len()]; count];
    let mut carry_hot = Vec::with_capacity(values.len());
    let mut fused_data = Vec::with_capacity(values.len());
    for (t, &value) in values.iter().enumerate() {
        let biased = value + bias;
        for (j, hot) in chunk_hot.iter_mut().enumerate() {
            let standard = (biased >> (chunking.chunk_width() * j)) & mask;
            hot[t] = ((standard + radix / 2) & mask) as usize;
        }
        carry_hot.push((biased >> 64).rem_euclid(radix) as usize);
        fused_data.push(Fr::from_i128(value));
    }
    let chunk_polynomials: Vec<Vec<Fr>> = chunk_hot
        .iter()
        .map(|hot| digit_zero_evals(8, log_t, hot))
        .collect();
    let carry_polynomial = digit_zero_evals(8, log_t, &carry_hot);

    let r_cycle = point(log_t, 1);
    let eq_cycle = EqPolynomial::<Fr>::evals(&r_cycle, None);

    let partial = |chunk: &[Fr]| -> Vec<Fr> {
        (0..256)
            .map(|a| {
                (0..values.len())
                    .map(|t| eq_cycle[t] * chunk[(a << log_t) | t])
                    .sum()
            })
            .collect::<Vec<Fr>>()
    };

    let mut reconstructed = fr(0);
    for (j, chunk) in chunk_polynomials.iter().enumerate() {
        let partials = partial(chunk);
        let decoded: Fr = partials
            .iter()
            .enumerate()
            .map(|(a, value)| balanced_inc_value(&boolean_point_msb::<Fr>(8, a)) * *value)
            .sum();
        reconstructed += chunking.place_value::<Fr>(j) * decoded;
    }
    let carry: Fr = partial(&carry_polynomial)
        .iter()
        .enumerate()
        .map(|(a, value)| balanced_inc_value(&boolean_point_msb::<Fr>(8, a)) * *value)
        .sum();

    let fused = eval_mle(&fused_data, &r_cycle);
    assert_eq!(reconstructed + Fr::pow2(64) * carry, fused);
}
