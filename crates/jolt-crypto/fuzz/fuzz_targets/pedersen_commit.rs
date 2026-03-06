#![no_main]
use jolt_crypto::{Bn254, Bn254G1, JoltCommitment, JoltGroup, Pedersen, PedersenSetup};
use jolt_field::{Field, Fr};
use libfuzzer_sys::fuzz_target;

/// Fixed small setup (4 generators) — deterministic so we don't waste fuzzer
/// entropy on setup.
fn fixed_setup() -> PedersenSetup<Bn254G1> {
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    let mut rng = ChaCha20Rng::seed_from_u64(0xf022);
    let gens: Vec<Bn254G1> = (0..4).map(|_| Bn254::random_g1(&mut rng)).collect();
    let blinding = Bn254::random_g1(&mut rng);
    PedersenSetup::new(gens, blinding)
}

fuzz_target!(|data: &[u8]| {
    // Need 5 * 32 = 160 bytes: 4 values + 1 blinding.
    if data.len() < 160 {
        return;
    }

    let setup = fixed_setup();

    let values: Vec<Fr> = (0..4)
        .map(|i| Fr::from_bytes(&data[i * 32..(i + 1) * 32]))
        .collect();
    let blinding = Fr::from_bytes(&data[128..160]);

    // Commit-verify round-trip
    let c = Pedersen::<Bn254G1>::commit(&setup, &values, &blinding);
    assert!(
        Pedersen::<Bn254G1>::verify(&setup, &c, &values, &blinding),
        "commit-verify round-trip failed"
    );

    // Homomorphism: C(v, r) + C(0, 0) == C(v, r)
    let zeros = vec![Fr::from_u64(0); 4];
    let zero_blind = Fr::from_u64(0);
    let c_zero = Pedersen::<Bn254G1>::commit(&setup, &zeros, &zero_blind);
    assert_eq!(c + c_zero, c, "adding zero commitment should be identity");

    // Commitment to zero values with blinding should equal blinding_gen * blinding
    let c_blind_only = Pedersen::<Bn254G1>::commit(&setup, &zeros, &blinding);
    let expected = setup.blinding_generator.scalar_mul(&blinding);
    assert_eq!(c_blind_only, expected, "zero-value commitment mismatch");
});
