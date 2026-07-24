//! Regenerates the checked-in `seeds/deser_group/` corpus from genuine
//! serialized group elements, so mutation explores the neighborhood of valid
//! encodings instead of pure garbage. The decoders' accept paths (G2 subgroup
//! membership, GT r-torsion) are unreachable from random bytes.
//!
//! Run explicitly, then commit the outputs:
//! `cargo test --manifest-path crates/jolt-crypto/fuzz/Cargo.toml -- --ignored`

use std::fs;
use std::path::Path;

use jolt_crypto::{Bn254, JoltGroup, PairingGroup, PedersenSetup};
use jolt_field::{Fr, RandomSampling};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[test]
#[ignore = "writes the checked-in seed corpus; run explicitly and commit the outputs"]
fn generate_deser_group_seeds() {
    let seed_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("seeds/deser_group");
    fs::create_dir_all(&seed_dir).expect("seed directory");
    let config = bincode::config::standard();
    let mut rng = ChaCha20Rng::seed_from_u64(0x5EED);

    let g1 = Bn254::random_g1(&mut rng);
    let g2 = Bn254::g2_generator().scalar_mul(&Fr::random(&mut rng));
    let gt = Bn254::pairing(&g1, &g2);

    let g1_bytes = bincode::serde::encode_to_vec(&g1, config).expect("encode g1");
    fs::write(seed_dir.join("valid-g1-bincode"), g1_bytes).expect("write g1 seed");
    let g2_bytes = bincode::serde::encode_to_vec(&g2, config).expect("encode g2");
    fs::write(seed_dir.join("valid-g2-bincode"), g2_bytes).expect("write g2 seed");
    let gt_bytes = bincode::serde::encode_to_vec(&gt, config).expect("encode gt");
    fs::write(seed_dir.join("valid-gt-bincode"), gt_bytes).expect("write gt seed");

    let setup = PedersenSetup::new(
        (0..2).map(|_| Bn254::random_g1(&mut rng)).collect(),
        Bn254::random_g1(&mut rng),
    );
    let setup_bytes = bincode::serde::encode_to_vec(&setup, config).expect("encode setup");
    fs::write(seed_dir.join("valid-pedersen-setup-bincode"), setup_bytes)
        .expect("write setup seed");

    let g1_json = serde_json::to_vec(&g1).expect("encode g1 json");
    fs::write(seed_dir.join("valid-g1-json"), g1_json).expect("write g1 json seed");
}
