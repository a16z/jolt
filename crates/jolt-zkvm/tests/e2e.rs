//! End-to-end integration tests for the Jolt proving pipeline.
//!
//! Validates the full `prove()` pipeline using synthetic R1CS circuits
//! and claim reduction stages, with both MockPCS and Dory backends.

use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_spartan::{FirstRoundStrategy, SimpleR1CS, SpartanKey};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_zkvm::proof::{JoltProof, JoltProvingKey};
use jolt_zkvm::prover::prove;
use jolt_zkvm::stage::ProverStage;
use jolt_zkvm::stages::s1_spartan::{SpartanResult, SpartanStage};
use jolt_zkvm::stages::s3_claim_reductions::ClaimReductionStage;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type MockPCS = MockCommitmentScheme<Fr>;

fn challenge_fn(c: u128) -> Fr {
    Fr::from_u128(c)
}

/// Build a simple R1CS: x * x = y (1 constraint, 3 variables).
fn x_squared_circuit() -> SimpleR1CS<Fr> {
    SimpleR1CS::new(
        1,
        3,
        vec![(0, 1, Fr::from_u64(1))], // A: row 0, var 1 (x)
        vec![(0, 1, Fr::from_u64(1))], // B: row 0, var 1 (x)
        vec![(0, 2, Fr::from_u64(1))], // C: row 0, var 2 (y = x²)
    )
}

/// Pad a witness to the next power of two.
fn pad_witness(witness: &[Fr]) -> Vec<Fr> {
    let padded_len = witness.len().next_power_of_two();
    let mut padded = witness.to_vec();
    padded.resize(padded_len, Fr::from_u64(0));
    padded
}

/// Run S1 (Spartan) and return the result.
fn run_s1<PCS: CommitmentScheme<Field = Fr>>(
    r1cs: &SimpleR1CS<Fr>,
    key: &SpartanKey<Fr>,
    witness: &[Fr],
    pcs_setup: &PCS::ProverSetup,
    prover_transcript: &mut Blake2bTranscript,
) -> SpartanResult<Fr, PCS> {
    let padded = pad_witness(witness);
    SpartanStage::<PCS>::prove(
        r1cs,
        key,
        witness,
        &padded,
        pcs_setup,
        prover_transcript,
        FirstRoundStrategy::Standard,
    )
    .expect("S1 proving should succeed")
}

/// Build a synthetic claim reduction stage using Spartan's r_y as the eq point.
///
/// Creates two random polynomials with known coefficients, giving us a
/// deterministic sumcheck to verify against.
fn build_reduction_stage(r_y: &[Fr], rng: &mut ChaCha20Rng) -> ClaimReductionStage<Fr> {
    let num_vars = r_y.len();
    let n = 1usize << num_vars;

    let poly_a: Vec<Fr> = (0..n).map(|_| Fr::random(rng)).collect();
    let poly_b: Vec<Fr> = (0..n).map(|_| Fr::random(rng)).collect();
    let c0 = Fr::random(rng);
    let c1 = Fr::random(rng);

    ClaimReductionStage::increment(poly_a, poly_b, r_y.to_vec(), c0, c1)
}

/// Full E2E test parameterized over PCS.
fn run_e2e<PCS: AdditivelyHomomorphic<Field = Fr>>(
    prover_setup: &PCS::ProverSetup,
    verifier_setup: &PCS::VerifierSetup,
) {
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    // Circuit: x * x = y, witness: x=3, y=9
    let r1cs = x_squared_circuit();
    let spartan_key = SpartanKey::from_r1cs(&r1cs);
    let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

    let key = JoltProvingKey::<Fr, PCS> {
        spartan_key: spartan_key.clone(),
        pcs_prover_setup: prover_setup.clone(),
        pcs_verifier_setup: verifier_setup.clone(),
    };

    let mut pt = Blake2bTranscript::new(b"e2e-test");

    let spartan_result = run_s1::<PCS>(&r1cs, &spartan_key, &witness, prover_setup, &mut pt);
    let r_y = spartan_result.r_y.clone();

    let reduction_stage = build_reduction_stage(&r_y, &mut rng);
    let mut stages: Vec<Box<dyn ProverStage<Fr, Blake2bTranscript>>> =
        vec![Box::new(reduction_stage)];

    let proof: JoltProof<Fr, PCS> =
        prove::<PCS, Blake2bTranscript>(spartan_result, &mut stages, &key, &mut pt, challenge_fn);

    assert_eq!(
        proof.sumcheck_proofs.len(),
        1,
        "one stage → one sumcheck proof"
    );
    assert!(
        !proof.opening_proofs.proofs.is_empty(),
        "should have at least one opening proof"
    );

    let mut vt = Blake2bTranscript::new(b"e2e-test");
    SpartanStage::<PCS>::verify(
        &key.spartan_key,
        &proof.spartan_proof,
        verifier_setup,
        &mut vt,
    )
    .expect("S1 verification should succeed");
}

#[test]
fn e2e_mock_pcs() {
    run_e2e::<MockPCS>(&(), &());
}

mod dory {
    use super::*;
    use jolt_dory::DoryScheme;

    #[test]
    fn e2e_dory() {
        // SimpleR1CS with 3 vars → SpartanKey pads to next power of 2 → 4 vars
        // The claim reduction stage uses r_y which has num_witness_vars length.
        // We need the Dory setup to accommodate the largest polynomial.
        let r1cs = x_squared_circuit();
        let spartan_key = SpartanKey::from_r1cs(&r1cs);
        let num_vars = spartan_key.num_witness_vars();

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::setup_verifier(num_vars);

        run_e2e::<DoryScheme>(&prover_setup, &verifier_setup);
    }
}
