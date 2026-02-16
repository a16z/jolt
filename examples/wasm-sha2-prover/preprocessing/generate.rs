use jolt_core::zkvm::Serializable;
use std::path::Path;

fn main() {
    let _ = jolt_inlines_sha2::init_inlines();

    let target_dir = "/tmp/jolt-wasm-sha2-guest";
    std::fs::create_dir_all(target_dir).expect("Failed to create target dir");

    println!("Compiling guest program...");
    let mut program = sha2_guest::compile_sha2(target_dir);

    println!("Generating shared preprocessing...");
    let shared_preprocessing = sha2_guest::preprocess_shared_sha2(&mut program);

    let elf_contents = program
        .get_elf_contents()
        .expect("Failed to get ELF contents");

    println!("Generating prover preprocessing...");
    let prover_preprocessing = sha2_guest::preprocess_prover_sha2(shared_preprocessing.clone());

    println!("Generating verifier preprocessing...");
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        sha2_guest::preprocess_verifier_sha2(shared_preprocessing, verifier_setup);

    let www_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("www");
    std::fs::create_dir_all(&www_dir).expect("Failed to create www dir");

    let prover_bytes = prover_preprocessing
        .serialize_to_bytes_uncompressed()
        .expect("Failed to serialize prover preprocessing");
    let prover_path = www_dir.join("prover_preprocessing.bin");
    std::fs::write(&prover_path, &prover_bytes).expect("Failed to write prover preprocessing");
    println!(
        "Prover preprocessing: {} bytes -> {:?}",
        prover_bytes.len(),
        prover_path
    );

    let verifier_bytes = verifier_preprocessing
        .serialize_to_bytes_uncompressed()
        .expect("Failed to serialize verifier preprocessing");
    let verifier_path = www_dir.join("verifier_preprocessing.bin");
    std::fs::write(&verifier_path, &verifier_bytes).expect("Failed to write verifier preprocessing");
    println!(
        "Verifier preprocessing: {} bytes -> {:?}",
        verifier_bytes.len(),
        verifier_path
    );

    let elf_dst = www_dir.join("guest.elf");
    std::fs::write(&elf_dst, &elf_contents).expect("Failed to write guest ELF");
    println!("Guest ELF: {} bytes -> {elf_dst:?}", elf_contents.len());

    println!("Done!");
}
