//! Re-encode the frozen norm measurement; never proves or imports an untrusted bincode proof.
use jolt_crypto::{Bn254, JoltGroup};
use jolt_field::{CanonicalBytes, Fr, One, Ring};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGSetupParams};
use jolt_openings::CommitmentScheme;
use jolt_spartan_verifier::preprocessed::{wire::verify_bytes, ComputationKey, PreprocessedProof};
use std::{error::Error, io::Write, path::PathBuf};

#[expect(
    clippy::print_stdout,
    reason = "intentional frozen fixture conversion diagnostic"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let source = PathBuf::from(
        args.next()
            .ok_or("expected frozen preprocessed-first directory")?,
    );
    let output = PathBuf::from(args.next().ok_or("expected fresh wire output path")?);
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let bytes = std::fs::read(source.join("proof.bin"))?;
    let key_bytes = std::fs::read(source.join("computation-key.bin"))?;
    let proof_digest = [
        165, 83, 8, 64, 45, 25, 33, 36, 105, 155, 162, 239, 26, 103, 16, 150, 101, 233, 0, 108, 18,
        93, 180, 136, 67, 121, 252, 151, 219, 82, 227, 58,
    ];
    let key_id = [
        100, 211, 106, 116, 241, 218, 174, 13, 61, 140, 255, 87, 76, 27, 164, 176, 178, 144, 79,
        42, 200, 41, 154, 111, 46, 108, 83, 235, 40, 78, 11, 56,
    ];
    if bytes.len() != 54716
        || ComputationKey::digest(&bytes) != proof_digest
        || key_bytes.len() != 476
        || ComputationKey::digest(&key_bytes) != key_id
    {
        return Err("not the exact accepted frozen norm fixture".into());
    }
    // Only this hash-authenticated historical fixture enters serde/bincode.
    let (proof, used): (PreprocessedProof, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
    if used != bytes.len() {
        return Err("trailing fixture bytes".into());
    }
    let beta = Fr::from_u64(7);
    let powers = std::iter::successors(Some(Bn254::g1_generator()), |point| {
        Some(point.scalar_mul(&beta))
    })
    .take(1 << 20)
    .collect();
    let (_, setup) = HyperKZGScheme::setup(HyperKZGSetupParams {
        g1_powers: powers,
        setup_id: [9; 32],
        max_public_degree: (1 << 20) - 1,
        g2: Bn254::g2_generator(),
        beta_g2: Bn254::g2_generator().scalar_mul(&beta),
    })?;
    let key = ComputationKey::decode_wire(&key_bytes, &key_id, &setup)?;
    let encoded = key.encode_proof(&proof)?;
    if encoded.len() != 54547 {
        return Err("unexpected norm wire length".into());
    }
    verify_bytes(
        &key_id,
        &setup,
        &key_bytes,
        &Fr::one().to_bytes_le_vec(),
        &encoded,
    )?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&encoded)?;
    println!(
        "accepted frozen norm wire bytes={} blake2b256={:?}",
        encoded.len(),
        ComputationKey::digest(&encoded)
    );
    Ok(())
}
