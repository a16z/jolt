use jolt_crypto::Bn254;
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::CommitmentScheme;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

pub type KzgPCS = HyperKZGScheme<Bn254>;

pub fn make_setup(max_degree: usize) -> (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xdead_beef);
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let pk = KzgPCS::setup(&mut rng, max_degree, g1, g2);
    let vk = KzgPCS::verifier_setup(&pk);
    (pk, vk)
}
