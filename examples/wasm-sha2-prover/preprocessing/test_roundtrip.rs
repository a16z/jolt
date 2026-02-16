use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::poly::commitment::dory::{
    ArkworksProverSetup, ArkworksVerifierSetup, DoryCommitmentScheme,
};
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing};
use jolt_core::zkvm::Serializable;
use std::io::Cursor;
use std::path::Path;

type ProverPrep = JoltProverPreprocessing<ark_bn254::Fr, DoryCommitmentScheme>;
type VerifierPrep = JoltVerifierPreprocessing<ark_bn254::Fr, DoryCommitmentScheme>;

fn test_prover_roundtrip(bytes: &[u8]) -> Result<(), String> {
    let total = bytes.len();
    println!("\nTesting Prover Preprocessing Roundtrip");
    println!("Total bytes: {total}");

    let mut cursor = Cursor::new(bytes);

    println!("Deserializing ArkworksProverSetup with Compress::No...");
    let generators = ArkworksProverSetup::deserialize_with_mode(
        &mut cursor,
        ark_serialize::Compress::No,
        ark_serialize::Validate::No,
    )
    .map_err(|e| format!("ProverSetup failed: {e}"))?;
    let pos_after_generators = cursor.position() as usize;
    println!("  OK - consumed {pos_after_generators} bytes");

    println!("Deserializing JoltSharedPreprocessing with Compress::No...");
    let shared = JoltSharedPreprocessing::deserialize_with_mode(
        &mut cursor,
        ark_serialize::Compress::No,
        ark_serialize::Validate::No,
    )
    .map_err(|e| {
        let pos = cursor.position();
        format!("SharedPreprocessing failed at pos {pos}: {e}")
    })?;
    let pos_after_shared = cursor.position() as usize;
    let shared_bytes = pos_after_shared - pos_after_generators;
    println!("  OK - consumed {shared_bytes} bytes (total: {pos_after_shared})");

    if pos_after_shared != bytes.len() {
        let total = bytes.len();
        let extra = total - pos_after_shared;
        return Err(format!(
            "Prover: consumed {pos_after_shared} bytes but file has {total} bytes ({extra} extra)"
        ));
    }

    println!("Testing full ProverPreprocessing::deserialize_from_bytes_uncompressed...");
    let _full = ProverPrep::deserialize_from_bytes_uncompressed(bytes)
        .map_err(|e| format!("Full deserialize failed: {e}"))?;
    println!("  OK");

    println!("Testing serialize -> deserialize roundtrip...");
    let prep = ProverPrep { generators, shared };
    let mut reserialized = Vec::new();
    prep.serialize_uncompressed(&mut reserialized)
        .map_err(|e| format!("Reserialize failed: {e}"))?;

    if reserialized.len() != bytes.len() {
        let orig = bytes.len();
        let reser = reserialized.len();
        return Err(format!(
            "Prover roundtrip size mismatch: original {orig} vs reserialized {reser}"
        ));
    }

    if reserialized != bytes {
        for (i, (a, b)) in bytes.iter().zip(reserialized.iter()).enumerate() {
            if a != b {
                return Err(format!(
                    "Prover roundtrip byte mismatch at position {i}: original {a:02x} vs reserialized {b:02x}"
                ));
            }
        }
    }
    println!("  OK - bytes match exactly");

    Ok(())
}

fn test_verifier_roundtrip(bytes: &[u8]) -> Result<(), String> {
    let total = bytes.len();
    println!("\nTesting Verifier Preprocessing Roundtrip");
    println!("Total bytes: {total}");

    let mut cursor = Cursor::new(bytes);

    println!("Deserializing ArkworksVerifierSetup with Compress::No...");
    let generators = ArkworksVerifierSetup::deserialize_with_mode(
        &mut cursor,
        ark_serialize::Compress::No,
        ark_serialize::Validate::No,
    )
    .map_err(|e| format!("VerifierSetup failed: {e}"))?;
    let pos_after_generators = cursor.position() as usize;
    println!("  OK - consumed {pos_after_generators} bytes");

    println!("Deserializing JoltSharedPreprocessing with Compress::No...");
    let shared = JoltSharedPreprocessing::deserialize_with_mode(
        &mut cursor,
        ark_serialize::Compress::No,
        ark_serialize::Validate::No,
    )
    .map_err(|e| {
        let pos = cursor.position();
        format!("SharedPreprocessing failed at pos {pos}: {e}")
    })?;
    let pos_after_shared = cursor.position() as usize;
    let shared_bytes = pos_after_shared - pos_after_generators;
    println!("  OK - consumed {shared_bytes} bytes (total: {pos_after_shared})");

    if pos_after_shared != bytes.len() {
        let total = bytes.len();
        let extra = total - pos_after_shared;
        return Err(format!(
            "Verifier: consumed {pos_after_shared} bytes but file has {total} bytes ({extra} extra)"
        ));
    }

    println!("Testing full VerifierPreprocessing::deserialize_from_bytes_uncompressed...");
    let _full = VerifierPrep::deserialize_from_bytes_uncompressed(bytes)
        .map_err(|e| format!("Full deserialize failed: {e}"))?;
    println!("  OK");

    println!("Testing serialize -> deserialize roundtrip...");
    let prep = VerifierPrep { generators, shared };
    let mut reserialized = Vec::new();
    prep.serialize_uncompressed(&mut reserialized)
        .map_err(|e| format!("Reserialize failed: {e}"))?;

    if reserialized.len() != bytes.len() {
        let orig = bytes.len();
        let reser = reserialized.len();
        return Err(format!(
            "Verifier roundtrip size mismatch: original {orig} vs reserialized {reser}"
        ));
    }

    if reserialized != bytes {
        for (i, (a, b)) in bytes.iter().zip(reserialized.iter()).enumerate() {
            if a != b {
                return Err(format!(
                    "Verifier roundtrip byte mismatch at position {i}: original {a:02x} vs reserialized {b:02x}"
                ));
            }
        }
    }
    println!("  OK - bytes match exactly");

    Ok(())
}

fn main() {
    let _ = jolt_inlines_sha2::init_inlines();
    let _ = jolt_inlines_secp256k1::init_inlines();
    let _ = jolt_inlines_keccak256::init_inlines();

    let www_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("www");

    let programs = [
        ("sha2", "sha2_prover.bin", "sha2_verifier.bin"),
        ("ecdsa", "ecdsa_prover.bin", "ecdsa_verifier.bin"),
        ("keccak", "keccak_prover.bin", "keccak_verifier.bin"),
    ];

    for (name, prover_file, verifier_file) in &programs {
        let prover_path = www_dir.join(prover_file);
        println!("\n[{name}] Reading prover preprocessing from {prover_path:?}");
        let prover_bytes = std::fs::read(&prover_path).expect("Failed to read prover file");

        match test_prover_roundtrip(&prover_bytes) {
            Ok(()) => println!("[{name}] Prover preprocessing: PASS"),
            Err(e) => {
                println!("[{name}] Prover preprocessing: FAIL - {e}");
                std::process::exit(1);
            }
        }

        let verifier_path = www_dir.join(verifier_file);
        println!("[{name}] Reading verifier preprocessing from {verifier_path:?}");
        let verifier_bytes = std::fs::read(&verifier_path).expect("Failed to read verifier file");

        match test_verifier_roundtrip(&verifier_bytes) {
            Ok(()) => println!("[{name}] Verifier preprocessing: PASS"),
            Err(e) => {
                println!("[{name}] Verifier preprocessing: FAIL - {e}");
                std::process::exit(1);
            }
        }
    }

    println!("\nAll tests passed!");
}
