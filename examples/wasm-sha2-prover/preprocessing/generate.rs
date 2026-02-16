use jolt_core::poly::commitment::dory::{ArkworksVerifierSetup, DoryCommitmentScheme};
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing};
use jolt_core::zkvm::Serializable;
use std::path::{Path, PathBuf};

type ProverPrep = JoltProverPreprocessing<ark_bn254::Fr, DoryCommitmentScheme>;
type VerifierPrep = JoltVerifierPreprocessing<ark_bn254::Fr, DoryCommitmentScheme>;

struct ProgramSpec {
    name: &'static str,
    prover_file: &'static str,
    verifier_file: &'static str,
    elf_file: &'static str,
    compile: fn(&str) -> jolt_core::host::Program,
    preprocess_shared: fn(&mut jolt_core::host::Program) -> JoltSharedPreprocessing,
    preprocess_prover: fn(JoltSharedPreprocessing) -> ProverPrep,
    preprocess_verifier: fn(JoltSharedPreprocessing, ArkworksVerifierSetup) -> VerifierPrep,
}

fn generate_program(www_dir: &Path, spec: &ProgramSpec) {
    let name = spec.name;
    let target_dir = format!("/tmp/jolt-wasm-{name}-guest");
    std::fs::create_dir_all(&target_dir).expect("Failed to create target dir");

    println!("[{name}] Compiling guest program...");
    let mut program = (spec.compile)(&target_dir);

    println!("[{name}] Generating shared preprocessing...");
    let shared = (spec.preprocess_shared)(&mut program);

    let elf_contents = program
        .get_elf_contents()
        .expect("Failed to get ELF contents");

    println!("[{name}] Generating prover preprocessing...");
    let prover_preprocessing = (spec.preprocess_prover)(shared.clone());

    println!("[{name}] Generating verifier preprocessing...");
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing = (spec.preprocess_verifier)(shared, verifier_setup);

    write_file(
        www_dir,
        spec.prover_file,
        name,
        "Prover",
        &prover_preprocessing
            .serialize_to_bytes_uncompressed()
            .expect("Failed to serialize prover preprocessing"),
    );

    write_file(
        www_dir,
        spec.verifier_file,
        name,
        "Verifier",
        &verifier_preprocessing
            .serialize_to_bytes_uncompressed()
            .expect("Failed to serialize verifier preprocessing"),
    );

    let elf_path = www_dir.join(spec.elf_file);
    std::fs::write(&elf_path, &elf_contents).expect("Failed to write ELF");
    println!("[{name}] ELF: {} bytes -> {elf_path:?}", elf_contents.len());
}

fn write_file(www_dir: &Path, filename: &str, program: &str, kind: &str, bytes: &[u8]) {
    let path = www_dir.join(filename);
    std::fs::write(&path, bytes).expect("Failed to write file");
    println!(
        "[{program}] {kind} preprocessing: {} bytes -> {path:?}",
        bytes.len()
    );
}

fn main() {
    let _ = jolt_inlines_sha2::init_inlines();
    let _ = jolt_inlines_secp256k1::init_inlines();
    let _ = jolt_inlines_keccak256::init_inlines();

    let www_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("www");
    std::fs::create_dir_all(&www_dir).expect("Failed to create www dir");

    let specs = [
        ProgramSpec {
            name: "sha2",
            prover_file: "sha2_prover.bin",
            verifier_file: "sha2_verifier.bin",
            elf_file: "sha2.elf",
            compile: sha2_guest::compile_sha2,
            preprocess_shared: sha2_guest::preprocess_shared_sha2,
            preprocess_prover: sha2_guest::preprocess_prover_sha2,
            preprocess_verifier: sha2_guest::preprocess_verifier_sha2,
        },
        ProgramSpec {
            name: "ecdsa",
            prover_file: "ecdsa_prover.bin",
            verifier_file: "ecdsa_verifier.bin",
            elf_file: "ecdsa.elf",
            compile: secp256k1_ecdsa_verify_guest::compile_secp256k1_ecdsa_verify,
            preprocess_shared:
                secp256k1_ecdsa_verify_guest::preprocess_shared_secp256k1_ecdsa_verify,
            preprocess_prover:
                secp256k1_ecdsa_verify_guest::preprocess_prover_secp256k1_ecdsa_verify,
            preprocess_verifier:
                secp256k1_ecdsa_verify_guest::preprocess_verifier_secp256k1_ecdsa_verify,
        },
        ProgramSpec {
            name: "keccak",
            prover_file: "keccak_prover.bin",
            verifier_file: "keccak_verifier.bin",
            elf_file: "keccak.elf",
            compile: sha3_chain_guest::compile_sha3_chain,
            preprocess_shared: sha3_chain_guest::preprocess_shared_sha3_chain,
            preprocess_prover: sha3_chain_guest::preprocess_prover_sha3_chain,
            preprocess_verifier: sha3_chain_guest::preprocess_verifier_sha3_chain,
        },
    ];

    for spec in &specs {
        generate_program(&www_dir, spec);
    }

    println!("Done!");
}
