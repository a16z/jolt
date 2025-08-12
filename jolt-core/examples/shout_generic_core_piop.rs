use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use jolt_core::subprotocols::shout::core_shout_piop_d_greater_one::{
    prove_generic_core_shout_pip_d_greater_than_one,
    prove_generic_core_shout_pip_d_greater_than_one_with_gruen,
};
use jolt_core::utils::transcript::KeccakTranscript;
use jolt_core::utils::transcript::Transcript;
use rand_core::RngCore;
use std::time::Instant;

fn main() {
    // ------- PROBLEM SETUP ----------------------
    const K: usize = 64;
    const T: usize = 1 << 22;
    const D: usize = 2;

    let n = (K as f64).powf(1.0 / D as f64).round() as usize;
    assert_eq!(n.pow(D as u32), K, "K must be a perfect power of N");

    let seed1: u64 = 42;
    let mut rng1 = StdRng::seed_from_u64(seed1);
    let lookup_table: Vec<Fr> = (0..K).map(|_| Fr::rand(&mut rng1)).collect();
    let read_addresses: Vec<usize> = (0..T).map(|_| (rng1.next_u32() as usize) % K).collect();

    let mut transcript = KeccakTranscript::new(b"bench");

    let start = Instant::now();
    let (
        _sumcheck_proof,
        _verifier_challenges,
        _sumcheck_claim,
        _ra_address_time_claim,
        _val_tau_claim,
        _eq_rcycle_rtime_claim,
        final_opening_lin,
    ) = prove_generic_core_shout_pip_d_greater_than_one(
        lookup_table.clone(),
        read_addresses.clone(),
        D,
        &mut transcript,
    );
    let duration = start.elapsed();
    println!(
        "{} \nNo optimisation: Execution time: {}",
        final_opening_lin,
        duration.as_millis()
    );

    let mut transcript = KeccakTranscript::new(b"bench");
    let start = Instant::now();
    let (
        _sumcheck_proof,
        _verifier_challenges,
        _sumcheck_claim,
        _ra_address_time_claim,
        _val_tau_claim,
        _eq_rcycle_rtime_claim,
        final_opening,
    ) = prove_generic_core_shout_pip_d_greater_than_one_with_gruen(
        lookup_table.clone(),
        read_addresses.clone(),
        D,
        &mut transcript,
    );
    let duration = start.elapsed();
    println!(
        "{} \nGruen Optimisaton- Execution time: {}",
        final_opening,
        duration.as_millis()
    );
}
