use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use criterion::{Bencher, Criterion};
use jolt_core::subprotocols::shout::core_shout_piop_d_is_one::prove_generic_core_shout_piop_d_is_one_w_gruen;
use jolt_core::transcripts::{Blake2bTranscript, Transcript};
use rand::RngCore;

/// Benchmark proving for the Gruen-optimized Shout sumcheck (d>1)
fn gruen_prover_benchmark(b: &mut Bencher) {
    const TABLE_SIZE: usize = 64; // 2^6
    const NUM_LOOKUPS: usize = 1 << 22; // 2^16

    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::rand(&mut rng)).collect();
    let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
        .map(|_| (rng.next_u32() as usize) % TABLE_SIZE)
        .collect();

    b.iter(|| {
        let mut transcript = Blake2bTranscript::new(b"benchmark");
        let _ = prove_generic_core_shout_piop_d_is_one_w_gruen(
            lookup_table.clone(),
            read_addresses.clone(),
            &mut transcript,
        );
    });
}

fn main() {
    // Create a Criterion object and configure it
    let mut criterion = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::new(30, 0))  // 30 seconds warm-up
        //.measurement_time(std::time::Duration::new(400, 0)) // 120 seconds measurement
        .sample_size(25); // Collect 50 samples

    // Manually run the benchmark function
    criterion.bench_function("gruen_prover_d_equals_1", gruen_prover_benchmark);
}
