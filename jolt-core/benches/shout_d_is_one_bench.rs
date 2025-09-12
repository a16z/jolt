use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::subprotocols::shout::core_shout_piop_d_is_one::prove_generic_core_shout_piop_d_is_one_w_gruen;
use jolt_core::transcripts::{Blake2bTranscript, Transcript};
use rand::RngCore;

/// Benchmark proving for the Gruen-optimized Shout sumcheck (d=1)
fn gruen_prover_benchmark(c: &mut Criterion) {
    const TABLE_SIZE: usize = 64; // 2^6
    const NUM_LOOKUPS: usize = 1 << 22; // 2^16

    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::rand(&mut rng)).collect();
    let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
        .map(|_| (rng.next_u32() as usize) % TABLE_SIZE)
        .collect();

    c.bench_function("gruen_prover_d1", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::new(b"benchmark");
            let _ = prove_generic_core_shout_piop_d_is_one_w_gruen(
                lookup_table.clone(),
                read_addresses.clone(),
                &mut transcript,
            );
        });
    });
}

criterion_group!(benches, gruen_prover_benchmark);
criterion_main!(benches);
