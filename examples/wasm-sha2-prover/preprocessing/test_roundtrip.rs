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
    println!("\nTesting Prover Preprocessing Roundtrip");
    println!("Total bytes: {}", bytes.len());

    let mut cursor = Cursor::new(bytes);

    println!("Deserializing ArkworksProverSetup with Compress::No...");
    let generators = ArkworksProverSetup::deserialize_with_mode(
        &mut cursor,
        ark_serialize::Compress::No,
        ark_serialize::Validate::No,
    )
    .map_err(|e| format!("ProverSetup failed: {e}"))?;
    let pos_after_generators = cursor.position() as usize;
    println!("  OK - consumed {} bytes", pos_after_generators);

    println!("Deserializing JoltSharedPreprocessing with Compress::No...");
    let shared = JoltSharedPreprocessing::deserialize_with_mode(
        &mut cursor,
        ark_serialize::Compress::No,
        ark_serialize::Validate::No,
    )
    .map_err(|e| format!("SharedPreprocessing failed at pos {}: {e}", cursor.position()))?;
    let pos_after_shared = cursor.position() as usize;
    println!(
        "  OK - consumed {} bytes (total: {})",
        pos_after_shared - pos_after_generators,
        pos_after_shared
    );

    if pos_after_shared != bytes.len() {
        return Err(format!(
            "Prover: consumed {} bytes but file has {} bytes ({} extra)",
            pos_after_shared,
            bytes.len(),
            bytes.len() - pos_after_shared
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
        return Err(format!(
            "Prover roundtrip size mismatch: original {} vs reserialized {}",
            bytes.len(),
            reserialized.len()
        ));
    }

    if reserialized != bytes {
        for (i, (a, b)) in bytes.iter().zip(reserialized.iter()).enumerate() {
            if a != b {
                return Err(format!(
                    "Prover roundtrip byte mismatch at position {}: original {:02x} vs reserialized {:02x}",
                    i, a, b
                ));
            }
        }
    }
    println!("  OK - bytes match exactly");

    Ok(())
}

fn test_verifier_roundtrip(bytes: &[u8]) -> Result<(), String> {
    println!("\nTesting Verifier Preprocessing Roundtrip");
    println!("Total bytes: {}", bytes.len());

    let mut cursor = Cursor::new(bytes);

    println!("Deserializing ArkworksVerifierSetup with Compress::No...");
    let generators = ArkworksVerifierSetup::deserialize_with_mode(
        &mut cursor,
        ark_serialize::Compress::No,
        ark_serialize::Validate::No,
    )
    .map_err(|e| format!("VerifierSetup failed: {e}"))?;
    let pos_after_generators = cursor.position() as usize;
    println!("  OK - consumed {} bytes", pos_after_generators);

    println!("Deserializing JoltSharedPreprocessing with Compress::No...");
    let shared = JoltSharedPreprocessing::deserialize_with_mode(
        &mut cursor,
        ark_serialize::Compress::No,
        ark_serialize::Validate::No,
    )
    .map_err(|e| format!("SharedPreprocessing failed at pos {}: {e}", cursor.position()))?;
    let pos_after_shared = cursor.position() as usize;
    println!(
        "  OK - consumed {} bytes (total: {})",
        pos_after_shared - pos_after_generators,
        pos_after_shared
    );

    if pos_after_shared != bytes.len() {
        return Err(format!(
            "Verifier: consumed {} bytes but file has {} bytes ({} extra)",
            pos_after_shared,
            bytes.len(),
            bytes.len() - pos_after_shared
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
        return Err(format!(
            "Verifier roundtrip size mismatch: original {} vs reserialized {}",
            bytes.len(),
            reserialized.len()
        ));
    }

    if reserialized != bytes {
        for (i, (a, b)) in bytes.iter().zip(reserialized.iter()).enumerate() {
            if a != b {
                return Err(format!(
                    "Verifier roundtrip byte mismatch at position {}: original {:02x} vs reserialized {:02x}",
                    i, a, b
                ));
            }
        }
    }
    println!("  OK - bytes match exactly");

    Ok(())
}

fn main() {
    let _ = jolt_inlines_sha2::init_inlines();

    let www_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("www");

    let prover_path = www_dir.join("prover_preprocessing.bin");
    println!("Reading prover preprocessing from {:?}", prover_path);
    let prover_bytes = std::fs::read(&prover_path).expect("Failed to read prover file");

    match test_prover_roundtrip(&prover_bytes) {
        Ok(()) => println!("\nProver preprocessing: PASS"),
        Err(e) => {
            println!("\nProver preprocessing: FAIL - {}", e);
            std::process::exit(1);
        }
    }

    let verifier_path = www_dir.join("verifier_preprocessing.bin");
    println!("\nReading verifier preprocessing from {:?}", verifier_path);
    let verifier_bytes = std::fs::read(&verifier_path).expect("Failed to read verifier file");

    match test_verifier_roundtrip(&verifier_bytes) {
        Ok(()) => println!("\nVerifier preprocessing: PASS"),
        Err(e) => {
            println!("\nVerifier preprocessing: FAIL - {}", e);
            std::process::exit(1);
        }
    }

    println!("\nAll tests passed!");
}
