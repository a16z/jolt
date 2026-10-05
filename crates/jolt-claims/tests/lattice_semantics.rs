use jolt_claims::protocols::jolt::lattice::geometry::balanced_inc_value;
use jolt_claims::protocols::jolt::lattice::BalancedIncChunking;
use jolt_field::{Fr, Ring};
use jolt_poly::{boolean_point_msb, EqPolynomial, Polynomial};
fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

/// MLE evaluation via the library's own (msb-first) convention — the same one
/// production code uses, so the tests pin the packing against it.
fn eval_mle(evals: &[Fr], point: &[Fr]) -> Fr {
    Polynomial::new(evals.to_vec()).evaluate(point)
}

fn point(len: usize, seed: u64) -> Vec<Fr> {
    const PRIMES: [u64; 16] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59];
    (0..len)
        .map(|i| fr(PRIMES[(i + seed as usize) % PRIMES.len()] + seed))
        .collect()
}

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
