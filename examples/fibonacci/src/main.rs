use jolt_sdk::serialize_and_print_size;
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let save_to_disk = std::env::args().any(|arg| arg == "--save");
    let run_groth16 = std::env::args().any(|arg| arg == "--groth16");

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fib(target_dir);

    let prover_preprocessing = guest::preprocess_prover_fib(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_fib(&prover_preprocessing);

    if save_to_disk {
        serialize_and_print_size(
            "Verifier Preprocessing",
            "/tmp/jolt_verifier_preprocessing.dat",
            &verifier_preprocessing,
        )
        .expect("Could not serialize preprocessing.");
    }

    let prove_fib = guest::build_prover_fib(program, prover_preprocessing);
    let verify_fib = guest::build_verifier_fib(verifier_preprocessing);

    let program_summary = guest::analyze_fib(10);
    program_summary
        .write_to_file("fib_10.txt".into())
        .expect("should write");

    let trace_file = "/tmp/fib_trace.bin";
    guest::trace_fib_to_file(trace_file, 50);
    info!("Trace file written to: {trace_file}.");

    let now = Instant::now();
    let (output, proof, io_device) = prove_fib(50);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    if save_to_disk {
        serialize_and_print_size("Proof", "/tmp/fib_proof.bin", &proof)
            .expect("Could not serialize proof.");
        serialize_and_print_size("io_device", "/tmp/fib_io_device.bin", &io_device)
            .expect("Could not serialize io_device.");
    }

    // Verify Stage 1 only (R1CS constraints) - for Groth16 transpilation experiment
    info!("Running Stage 1-only verification...");
    use jolt_core::zkvm::stage1_only_verifier::{
        Stage1OnlyPreprocessing, Stage1OnlyProof, Stage1OnlyVerifier,
    };
    use jolt_core::poly::commitment::dory::DoryCommitmentScheme;

    let (stage1_proof, opening_claims, commitments, ram_K) =
        Stage1OnlyProof::from_full_proof::<DoryCommitmentScheme>(&proof);

    let stage1_preprocessing = Stage1OnlyPreprocessing::new(proof.trace_length);

    let stage1_verifier = Stage1OnlyVerifier::new::<DoryCommitmentScheme>(
        stage1_preprocessing,
        stage1_proof.clone(),
        opening_claims.clone(),
        &io_device,
        &commitments,
        ram_K,
    )
    .expect("Failed to create Stage 1 verifier");

    stage1_verifier
        .verify()
        .expect("Stage 1 verification failed");
    info!("Stage 1 verification PASSED");

    // Run Groth16 circuit with real proof data (if enabled)
    #[cfg(feature = "groth16")]
    if run_groth16 {
        run_groth16_verification(
            &stage1_proof,
            &opening_claims,
            &io_device,
            &commitments,
            ram_K,
        );
    }

    let is_valid = verify_fib(50, output, io_device.panic, proof);
    info!("output: {output}");
    info!("valid: {is_valid}");
}

/// Transpile Stage 1 verification into a Groth16 proof for EVM-efficient verification.
///
/// This function demonstrates the Jolt → Groth16 transpilation pipeline:
///
/// ## Background
///
/// Jolt's native verifier uses Dory polynomial commitments and interactive sumcheck
/// protocols, which are efficient for native execution but expensive on-chain (~1.2B gas).
/// By transpiling Stage 1 (Spartan outer sumcheck) into a Groth16 circuit, we get:
/// - **Constant-size proof**: 2 G1 + 1 G2 elements (~256 bytes)
/// - **O(1) verification**: Single pairing check (~200k gas on EVM)
///
/// ## What the circuit verifies
///
/// 1. **Uni-skip first round**: Power sum check `Σ_j a_j * S_j == 0` over symmetric
///    domain D = {-4, -3, ..., 5}, where S_j are precomputed power sums.
///
/// 2. **Sumcheck round consistency**: For each round i, verifies `g_i(0) + g_i(1) == claim_{i-1}`
///    and computes the next claim as `g_i(r_i)`.
///
/// ## Current limitations
///
/// The final claim check (Lagrange kernel × eq polynomial × inner sum product) is not yet
/// implemented. This requires computing:
/// - `L(τ_high, r0)`: Lagrange kernel evaluation
/// - `eq(τ_low, r_tail)`: Multilinear equality polynomial
/// - `A(rx, r) * B(rx, r)`: Inner product from R1CS evaluations
///
/// For now, the polynomial opening verification is handled by Dory commitments in the
/// native Jolt verifier, and we trust this part.
///
/// ## Pipeline
///
/// ```text
/// Stage1OnlyProof ──► Stage1CircuitData ──► Stage1Circuit ──► Groth16 proof
///                     (witness extraction)   (R1CS constraints)
/// ```
///
/// # Arguments
///
/// * `stage1_proof` - The Stage 1 proof containing uni-skip polynomial and sumcheck rounds
/// * `opening_claims` - Polynomial evaluation claims from the Jolt proof
/// * `io_device` - Program I/O (inputs, outputs, panic status)
/// * `commitments` - Dory polynomial commitments from the Jolt proof
/// * `ram_K` - RAM size parameter (number of memory cells)
#[cfg(feature = "groth16")]
fn run_groth16_verification(
    stage1_proof: &jolt_core::zkvm::stage1_only_verifier::Stage1OnlyProof<ark_bn254::Fr, jolt_core::transcripts::Blake2bTranscript>,
    opening_claims: &jolt_core::poly::opening_proof::Openings<ark_bn254::Fr>,
    io_device: &common::jolt_device::JoltDevice,
    commitments: &[<jolt_core::poly::commitment::dory::DoryCommitmentScheme as jolt_core::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment],
    ram_K: usize,
) {
    use ark_bn254::Bn254;
    use ark_groth16::Groth16;
    use ark_snark::SNARK;
    use ark_std::rand::thread_rng;
    use jolt_core::groth16::{Stage1Circuit, Stage1CircuitData};
    use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
    use jolt_core::transcripts::Blake2bTranscript;
    use std::time::Instant;

    info!("=== Running Groth16 Verification ===");

    // Extract circuit data from the real Stage 1 proof
    info!("Extracting circuit data from Stage 1 proof...");
    let circuit_data = Stage1CircuitData::from_stage1_proof::<ark_bn254::Fr, DoryCommitmentScheme, Blake2bTranscript>(
        stage1_proof,
        opening_claims,
        io_device,
        commitments,
        ram_K,
    );

    info!("Circuit data extracted:");
    info!("  - tau challenges: {}", circuit_data.tau.len());
    info!("  - r0 challenge: present");
    info!("  - sumcheck challenges: {}", circuit_data.sumcheck_challenges.len());
    info!("  - uni-skip poly coeffs: {}", circuit_data.uni_skip_poly_coeffs.len());
    info!("  - sumcheck round polys: {}", circuit_data.sumcheck_round_polys.len());
    info!("  - r1cs input evals: {}", circuit_data.r1cs_input_evals.len());
    info!("  - trace length: {}", circuit_data.trace_length);
    info!("  - total public inputs: {}", circuit_data.public_input_count());

    // Create the Groth16 circuit
    let circuit = Stage1Circuit::from_data(circuit_data.clone());

    // Setup phase
    info!("Running Groth16 setup...");
    let mut rng = thread_rng();
    let setup_start = Instant::now();
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(circuit.clone(), &mut rng)
        .expect("Groth16 setup failed");
    let setup_time = setup_start.elapsed();
    info!("Groth16 setup completed in {:.2}s", setup_time.as_secs_f64());

    // Prove phase
    info!("Generating Groth16 proof...");
    let prove_start = Instant::now();
    let groth16_proof = Groth16::<Bn254>::prove(&pk, circuit.clone(), &mut rng)
        .expect("Groth16 proving failed");
    let prove_time = prove_start.elapsed();
    info!("Groth16 proof generated in {:.2}s", prove_time.as_secs_f64());

    // Verify phase
    info!("Verifying Groth16 proof...");
    let public_inputs = circuit.public_inputs();
    let verify_start = Instant::now();
    let is_valid = Groth16::<Bn254>::verify(&vk, &public_inputs, &groth16_proof)
        .expect("Groth16 verification failed");
    let verify_time = verify_start.elapsed();

    info!("=== Groth16 Results ===");
    info!("  Setup time:   {:.2}s", setup_time.as_secs_f64());
    info!("  Prove time:   {:.2}s", prove_time.as_secs_f64());
    info!("  Verify time:  {:.6}s", verify_time.as_secs_f64());
    info!("  Verification: {}", if is_valid { "PASSED ✓" } else { "FAILED ✗" });

    if is_valid {
        info!("Groth16 verification succeeded! The circuit verifies:");
        info!("  1. Uni-skip power sum check: Σ_j a_j * S_j == 0");
        info!("  2. Sumcheck round consistency: g_i(0) + g_i(1) == claim_{{i-1}}");
        info!("Note: Full Stage 1 verification requires additional final claim check.");
    } else {
        info!("ERROR: Groth16 verification failed unexpectedly!");
    }
}
