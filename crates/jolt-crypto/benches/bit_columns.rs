#![expect(
    clippy::print_stdout,
    reason = "benchmark binary reports measurements to stdout"
)]

use std::hint::black_box;
use std::time::Instant;

use jolt_crypto::ec::bn254::batch_addition::batch_g1_additions_multi;
use jolt_crypto::ec::bn254::bit_columns::g1_bit_columns_msm;
use jolt_crypto::{Bn254, Bn254G1, JoltGroup, PairingGroup};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;

fn main() {
    let max_rows = 1 << 18;
    let mut rng = ChaCha20Rng::seed_from_u64(0x00b1_7c01);
    let generator = Bn254::g1_generator();
    let scalars: Vec<Fr> = (0..max_rows).map(|_| Fr::random(&mut rng)).collect();
    let projective_bases: Vec<Bn254G1> = scalars
        .par_iter()
        .map(|scalar| generator.scalar_mul(scalar))
        .collect();
    let start = Instant::now();
    let affine_bases = Bn254::g1_to_affine(&projective_bases);
    println!(
        "projective-to-affine setup conversion: {:.3} ms",
        start.elapsed().as_secs_f64() * 1e3
    );

    let columns: Vec<Vec<u8>> = (0..163)
        .map(|_| (0..max_rows).map(|_| (rng.next_u32() & 1) as u8).collect())
        .collect();

    for (rows, count) in [
        (1 << 17, 1),
        (1 << 17, 16),
        (1 << 18, 22),
        (1 << 17, 163),
        (1 << 18, 163),
    ] {
        let columns: Vec<&[u8]> = columns[..count]
            .iter()
            .map(|column| &column[..rows])
            .collect();
        let indices: Vec<Vec<usize>> = columns
            .par_iter()
            .map(|column| {
                column
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &bit)| (bit != 0).then_some(index))
                    .collect()
            })
            .collect();
        let additions: usize = indices.iter().map(|set| set.len().saturating_sub(1)).sum();

        let start = Instant::now();
        let old = black_box(batch_g1_additions_multi(
            black_box(&projective_bases[..rows]),
            black_box(&indices),
        ));
        let old_time = start.elapsed();

        let start = Instant::now();
        let new = black_box(g1_bit_columns_msm(
            black_box(&affine_bases[..rows]),
            black_box(&columns),
        ));
        let new_time = start.elapsed();
        assert_eq!(new, old);

        println!(
            "rows=2^{} columns={count:3} additions={additions:9} old={:8.3} ms ({:6.2} ns/add) new={:8.3} ms ({:6.2} ns/add)",
            rows.ilog2(),
            old_time.as_secs_f64() * 1e3,
            old_time.as_nanos() as f64 / additions as f64,
            new_time.as_secs_f64() * 1e3,
            new_time.as_nanos() as f64 / additions as f64,
        );
    }
}
