//! Contains the utility required to turn Dory arguments into a full-fledged PCS
//! Primarily makes use of the `eval_vmv_re` protocol
use ark_ff::{Field, PrimeField};
use ark_serialize::CanonicalSerialize;

use crate::{
    arithmetic::{Group, MultiScalarMul, Pairing},
    build_vmv_prover_state,
    builder::{DoryProofBuilder, DoryVerifyBuilder, VerificationBuilder},
    commit_to_rows,
    error::DoryError,
    inner_product::inner_product_verify,
    inner_product_prove,
    messages::VMVMessage,
    primitives::poly::compute_l_r_tensors,
    setup::{ProverSetup, VerifierSetup},
    state::{DoryProverState, DoryVerifierState},
    transcript::Transcript,
    vmv::{compute_nu, VMVVerifierState},
    vmv_state_to_dory_prover_state, ProofBuilder,
};

/// Implements the commit phase of the Eval-VMV-RE protocol from Dory Section 5
/// Proves the VMV relation: polynomial(point) = L^T × M × R
///
/// Note: Randomness terms (rC, rD2, rE1) are omitted since we don't need hiding
fn eval_vmv_re_prove<
    E: Pairing,
    T: Transcript<Scalar = <E::G1 as Group>::Scalar>,
    M1: MultiScalarMul<E::G1>,
>(
    mut proof_builder: DoryProofBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>,
    mut prover_state: DoryProverState<E>,
    prover_setup: &ProverSetup<E>,
) -> (
    DoryProofBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>,
    DoryProverState<E>,
)
where
    E::G1: Group + CanonicalSerialize,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar> + CanonicalSerialize,
    E::GT: Group<Scalar = <E::G1 as Group>::Scalar> + CanonicalSerialize,
    <E::G1 as Group>::Scalar: Field + PrimeField,
{
    // Validate inputs
    if prover_state.v1.is_empty() || prover_state.s1.is_empty() {
        println!("v1 or s1 is empty in eval_vmv_re_prove");
    }
    if prover_setup
        .g1_pows
        .get(prover_state.nu)
        .map_or(true, |v| v.is_empty())
    {
        println!("prover_setup.g1_pows[nu] is empty or nu is out of bounds");
    }

    // --- Protocol computations (Dory Section 5) ---

    // C = e(⟨T~₀, ~v⟩, Γ₂,fin)
    // Protocol: C = e(⟨~v, T~₀⟩, Γ₂,fin) + rC·HT (randomness omitted)
    let t_vec_v_inner_product = M1::msm(&prover_state.v1, &prover_state.s1);
    let c_val = E::pair(&t_vec_v_inner_product, &prover_setup.h2);

    // D₂ = e(⟨Γ₁[nu], ~v⟩, Γ₂,fin)
    // Protocol: D₂ = e(⟨Γ₁,~v⟩, Γ₂,fin) + rD₂·HT (randomness omitted)
    let g1_bases_at_nu = prover_setup
        .g1_pows
        .get(prover_state.nu)
        .map_or_else(|| &[][..], |bases_vec| bases_vec.as_slice());

    let gamma1_v_inner_product = if g1_bases_at_nu.is_empty() || prover_state.s1.is_empty() {
        E::G1::identity()
    } else {
        M1::msm(g1_bases_at_nu, &prover_state.s1)
    };
    let d2_val = E::pair(&gamma1_v_inner_product, &prover_setup.h2);

    // E₁ = ⟨T~₀, ~L⟩
    // Protocol: E₁ = ⟨~L, C₀⟩ + rE₁·H₁ (randomness omitted)
    if prover_state.s2.is_empty() && !prover_state.v1.is_empty() {
        println!("s2 is empty but v1 is not in E₁ calculation");
    }
    let e1_val = M1::msm(&prover_state.v1, &prover_state.s2);

    // Create VMV message for transcript
    let vmv_message = VMVMessage {
        c: c_val,
        d2: d2_val,
        e1: e1_val,
    };
    proof_builder = proof_builder.append_vmv_message(vmv_message);

    // Transform intermediate vector ~v into G2 elements for next phase
    // v₂ = ~v · Γ₂,fin (scalar multiplication in G2)
    let updated_v2 = prover_state
        .s1
        .iter()
        .map(|v_element| prover_setup.h2.scale(v_element))
        .collect::<Vec<E::G2>>();

    prover_state.v2 = updated_v2;

    (proof_builder, prover_state)
}

/// Create a new Dory evaluation proof
pub fn create_evaluation_proof<
    E: Pairing,
    T: Transcript<Scalar = <E::G1 as Group>::Scalar>,
    M1: MultiScalarMul<E::G1>,
    M2: MultiScalarMul<E::G2>,
>(
    initial_transcript: T, // DoryProofBuilder takes ownership of the transcript
    coeffs: &[<E::G1 as Group>::Scalar],
    point: &[<E::G1 as Group>::Scalar],
    sigma: usize,
    prover_setup: &ProverSetup<E>,
) -> DoryProofBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>
where
    E::G1: Group + CanonicalSerialize,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar> + CanonicalSerialize,
    E::GT: Group<Scalar = <E::G1 as Group>::Scalar> + CanonicalSerialize,
    <E::G1 as Group>::Scalar: Field + PrimeField,
{
    // 1. Compute parameters
    let nu = compute_nu(point.len(), sigma);
    // println!("nu length: {:?}", nu); -> useful for debug

    // 2. Compute row commits (T` in the paper?)
    let t_vec_prime = commit_to_rows::<E, M1>(coeffs, sigma, nu, prover_setup);

    // 3. Build VMV prover state
    let vmv_state = build_vmv_prover_state::<E>(coeffs, point, t_vec_prime, sigma, nu);

    // 4. Convert VMV state to DoryProverState
    let prover_state = vmv_state_to_dory_prover_state(vmv_state, prover_setup);

    // 5. Initialize the DoryProofBuilder
    let proof_builder = DoryProofBuilder::new(initial_transcript);

    // 6. Initial commitments
    let (final_proof_builder, proof_state) =
        eval_vmv_re_prove::<E, T, M1>(proof_builder, prover_state, prover_setup);

    // prove!
    inner_product_prove::<_, _, _, _, _, _, _, M1, M2>(
        final_proof_builder,
        proof_state,
        prover_setup,
        nu,
    )
}

// VERIFIER ANALOGUE:

/// Build the prover state for the VMV protocol
fn build_vmv_verifier_state<E: Pairing>(
    y: <E::G1 as Group>::Scalar,
    b_point: &[<E::G1 as Group>::Scalar],
    t: E::GT,
    sigma: usize,
    nu: usize,
) -> VMVVerifierState<E>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
    <E::G1 as Group>::Scalar: Copy,
{
    let (l_tensor, r_tensor) = compute_l_r_tensors(b_point, sigma, nu);

    VMVVerifierState {
        y,
        t,
        l_tensor,
        r_tensor,
        nu,
    }
}

/// Convert a VMVVerifierState to a DoryVerifierState
/// This handles the tensor structure and received messages into the normal verifier state
fn vmv_state_to_dory_verifier_state<E: Pairing>(
    vmv_state: VMVVerifierState<E>,
    vmv_message: &VMVMessage<E::G1, E::GT>,
    verifier_setup: &VerifierSetup<E>,
) -> DoryVerifierState<E>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    // Extract received messages
    let c = vmv_message.c.clone();
    let d_2 = vmv_message.d2.clone();
    let e_1 = vmv_message.e1.clone();

    // Extract values from VMV verifier state
    let d_1 = vmv_state.t;
    let s1_tensor = vmv_state.r_tensor;
    let s2_tensor = vmv_state.l_tensor;
    let nu = vmv_state.nu;

    let e_2 = verifier_setup.h2.scale(&vmv_state.y); //@TODO(markosg04) should be an MSM involved here?
                                                     // @TODO(markosg04) We don't compute this prover side as an optimization...

    let mut verifier_state = DoryVerifierState::new(c, d_1, d_2, e_1, e_2, nu);
    verifier_state.s1_tensor = Some(s1_tensor);
    verifier_state.s2_tensor = Some(s2_tensor);

    verifier_state
}

/// Verifier analogue of `eval_vmv_re` protocol in the paper
fn eval_vmv_re_verify<
    E: Pairing,
    T: Transcript<Scalar = <E::G1 as Group>::Scalar>,
    M1: MultiScalarMul<E::G1>,
>(
    mut verify_builder: DoryVerifyBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>,
    vmv_state: VMVVerifierState<E>,
    verifier_setup: &VerifierSetup<E>,
) -> (
    DoryVerifyBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>,
    DoryVerifierState<E>,
)
where
    E::G1: Group + CanonicalSerialize,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar> + CanonicalSerialize,
    E::GT: Group<Scalar = <E::G1 as Group>::Scalar> + CanonicalSerialize,
    <E::G1 as Group>::Scalar: Field + PrimeField,
{
    let vmv_message = verify_builder.process_vmv_message();
    let final_verifier_state =
        vmv_state_to_dory_verifier_state(vmv_state, &vmv_message, verifier_setup);
    // Return the updated verify builder and unchanged verifier state
    // The verifier state conversion should be handled by the caller
    (verify_builder, final_verifier_state)
}

/// Verify a dory evaluation proof
pub fn verify_evaluation_proof<
    E: Pairing,
    T: Transcript<Scalar = <E::G1 as Group>::Scalar>,
    M1: MultiScalarMul<E::G1>,
    M2: MultiScalarMul<E::G2>,
    MGT: MultiScalarMul<E::GT>,
>(
    proof: DoryProofBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>,
    commitment_batch: &[E::GT],
    batching_factors: &[<E::G1 as Group>::Scalar],
    evaluations: &[<E::G1 as Group>::Scalar],
    b_points: &[<E::G1 as Group>::Scalar],
    sigma: usize,
    verifier_setup: &VerifierSetup<E>,
    domain: &[u8],
) -> Result<(), DoryError>
where
    E::G1: Group + CanonicalSerialize,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar> + CanonicalSerialize,
    E::GT: Group<Scalar = <E::G1 as Group>::Scalar> + CanonicalSerialize,
    <E::G1 as Group>::Scalar: Field + PrimeField,
{
    // 1. Compute the MSM of commits and the factors
    let a_commit = MGT::msm(commitment_batch, batching_factors);

    // 2. Compute the product of evaluations and batching factors (batching factors should be 1)
    let product: <E::G1 as Group>::Scalar = evaluations
        .iter()
        .zip(batching_factors)
        .map(|(&e, &f)| e * f)
        .sum();

    // 3. Compute nu
    let nu = compute_nu(b_points.len(), sigma);

    // 4. Build VMV verifier state
    let vmv_state = build_vmv_verifier_state::<E>(product, b_points, a_commit, sigma, nu);

    // 5. Create verifier builder from proof
    let verify_builder = DoryVerifyBuilder::new_from_proof(domain, proof);

    // 6. Eval VMV re verifier side
    let (verify_builder, verifier_state) =
        eval_vmv_re_verify::<E, T, M1>(verify_builder, vmv_state, verifier_setup);

    // 7. Dory inner product verify
    inner_product_verify(verify_builder, verifier_state, verifier_setup, nu)
        .map_err(|_| DoryError::InvalidProof)
}
