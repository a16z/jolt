#![no_main]

//! Fuzz: random polynomial + random point must always commit-open-verify successfully.

use jolt_crypto::Bn254;
use jolt_field::{Field, Fr};
use jolt_hyperkzg::HyperKZGScheme;
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type TestScheme = HyperKZGScheme<Bn254>;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let seed = u64::from_le_bytes(data[..8].try_into().unwrap());
    let num_vars = (data.len() % 4) + 1; // 1..=4 variables
    let n = 1usize << num_vars;

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let pk = TestScheme::setup(&mut rng, n, g1, g2);
    let vk = TestScheme::verifier_setup(&pk);

    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let eval = poly.evaluate(&point);

    let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk);

    let mut pt = Blake2bTranscript::new(b"fuzz");
    let proof =
        <TestScheme as CommitmentScheme>::open(&poly, &point, eval, &pk, None, &mut pt);

    let mut vt = Blake2bTranscript::new(b"fuzz");
    <TestScheme as CommitmentScheme>::verify(&commitment, &point, eval, &proof, &vk, &mut vt)
        .expect("valid proof must verify");
});
