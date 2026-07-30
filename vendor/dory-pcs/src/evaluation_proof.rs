//! Evaluation proof generation and verification using Eval-VMV-RE protocol
//!
//! Implements the full proof generation and verification by:
//! 1. Computing VMV message (C, D2, E1)
//! 2. Running max(nu, sigma) rounds of inner product protocol (reduce and fold)
//! 3. Producing the final scalar product message (transparent) or the
//!    scalar-product Σ-proof (ZK)
//!
//! ## Matrix Layout
//!
//! Supports flexible matrix layouts with constraint nu ≤ sigma:
//! - **Square matrices** (nu = sigma): Traditional layout, e.g., 16×16 for nu=4, sigma=4
//! - **Non-square matrices** (nu < sigma): Wider layouts, e.g., 8×16 for nu=3, sigma=4
//!
//! The protocol automatically pads shorter dimensions and uses max(nu, sigma) rounds
//! in the reduce-and-fold phase.
//!
//! ## Homomorphic Properties
//!
//! The evaluation proof protocol preserves the homomorphic properties of Dory commitments.
//! This enables proving evaluations of linear combinations:
//!
//! ```text
//! Com(r₁·P₁ + r₂·P₂) = r₁·Com(P₁) + r₂·Com(P₂)
//! ```
//!
//! See `examples/homomorphic.rs` for a complete demonstration.

use crate::error::DoryError;
use crate::messages::VMVMessage;
use crate::mode::Mode;
use crate::primitives::arithmetic::{DoryRoutines, Field, Group, PairingCurve};
use crate::primitives::poly::MultilinearLagrange;
use crate::primitives::transcript::Transcript;
use crate::proof::DoryProof;
use crate::proof::ProofMode;
use crate::reduce_and_fold::{DoryProverState, DoryVerifierState, FinalCheck};
use crate::setup::{ProverSetup, VerifierSetup};

/// Create evaluation proof for a polynomial at a point
///
/// Implements Eval-VMV-RE protocol from Dory Section 5.
/// The protocol proves that polynomial(point) = evaluation via the VMV relation:
/// evaluation = L^T × M × R
///
/// # Algorithm
/// 1. Compute or use provided row commitments (Tier 1 commitment)
/// 2. Split evaluation point into left and right vectors
/// 3. Compute v_vec (column evaluations)
/// 4. Create VMV message (C, D2, E1)
/// 5. Initialize prover state for inner product / reduce-and-fold protocol
/// 6. Run max(nu, sigma) rounds of reduce-and-fold (with automatic padding for non-square):
///    - First reduce: compute message and apply beta challenge (reduce)
///    - Second reduce: compute message and apply alpha challenge (fold)
/// 7. Apply Fold-Scalars to the witness (absorbs the point tensors s₁, s₂)
/// 8. Transparent: reveal the folded witness as the final scalar product message;
///    ZK: produce a scalar-product Σ-proof for the folded statement instead
///
/// # Parameters
/// - `polynomial`: Polynomial to prove evaluation for
/// - `point`: Evaluation point (length nu + sigma)
/// - `row_commitments`: Optional precomputed row commitments from polynomial.commit()
/// - `commit_blind`: GT-level blinding scalar from `commit()`. Ignored when
///   `row_commitments` is `None` (the blind is computed internally in that case).
/// - `nu`: Log₂ of number of rows (constraint: nu ≤ sigma)
/// - `sigma`: Log₂ of number of columns
/// - `setup`: Prover setup
/// - `transcript`: Fiat-Shamir transcript for challenge generation
///
/// # Returns
/// Complete Dory proof containing the VMV message, reduce messages, and the
/// final message (transparent) or Σ-proofs (ZK)
///
/// # Errors
/// Returns error if dimensions are invalid (nu > sigma) or protocol fails
///
/// # Matrix Layout
/// Supports both square (nu = sigma) and non-square (nu < sigma) matrices.
/// For non-square matrices, vectors are automatically padded to length 2^sigma.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "create_evaluation_proof")]
pub fn create_evaluation_proof<F, E, M1, M2, T, P, Mo>(
    polynomial: &P,
    point: &[F],
    row_commitments: Option<Vec<E::G1>>,
    commit_blind: F,
    nu: usize,
    sigma: usize,
    setup: &ProverSetup<E>,
    transcript: &mut T,
) -> Result<(DoryProof<E::G1, E::G2, E::GT>, Option<F>), DoryError>
where
    F: Field,
    E: PairingCurve,
    E::G1: Group<Scalar = F>,
    E::G2: Group<Scalar = F>,
    E::GT: Group<Scalar = F>,
    M1: DoryRoutines<E::G1>,
    M2: DoryRoutines<E::G2>,
    T: Transcript<Curve = E>,
    P: MultilinearLagrange<F>,
    Mo: Mode,
{
    if point.len() != nu + sigma {
        return Err(DoryError::InvalidPointDimension {
            expected: nu + sigma,
            actual: point.len(),
        });
    }

    // Validate matrix dimensions: nu must be ≤ sigma (rows ≤ columns)
    if nu > sigma {
        return Err(DoryError::InvalidSize {
            expected: sigma,
            actual: nu,
        });
    }

    let (row_commitments, commit_blind) = match row_commitments {
        Some(rc) => (rc, commit_blind),
        None => {
            let (_, rc, blind) = polynomial.commit::<E, Mo, M1>(nu, sigma, setup)?;
            (rc, blind)
        }
    };

    let (left_vec, right_vec) = polynomial.compute_evaluation_vectors(point, nu, sigma);
    let v_vec = polynomial.vector_matrix_product(&left_vec, nu, sigma);

    let mut padded_row_commitments = row_commitments.clone();
    if nu < sigma {
        padded_row_commitments.resize(1 << sigma, E::G1::identity());
    }

    // Sample VMV blinds (zero in Transparent, random in ZK)
    let (r_c, r_d2, r_e1, r_e2): (F, F, F, F) =
        (Mo::sample(), Mo::sample(), Mo::sample(), Mo::sample());

    let g2_fin = &setup.g2_vec[0];

    // C = e(⟨row_commitments, v_vec⟩, Γ2,fin) + r_c·HT
    let t_vec_v = M1::msm(&padded_row_commitments, &v_vec);
    let c = Mo::mask(E::pair(&t_vec_v, g2_fin), &setup.ht, &r_c);

    // D₂ = e(⟨Γ₁[sigma], v_vec⟩, Γ2,fin) + r_d2·HT
    let d2 = Mo::mask(
        E::pair(&M1::msm(&setup.g1_vec[..1 << sigma], &v_vec), g2_fin),
        &setup.ht,
        &r_d2,
    );

    // E₁ = ⟨row_commitments, left_vec⟩ + r_e1·H₁
    let e1 = Mo::mask(M1::msm(&row_commitments, &left_vec), &setup.h1, &r_e1);

    let vmv_message = VMVMessage { c, d2, e1 };

    transcript.append_serde(b"vmv_c", &vmv_message.c);
    transcript.append_serde(b"vmv_d2", &vmv_message.d2);
    transcript.append_serde(b"vmv_e1", &vmv_message.e1);

    #[cfg(feature = "zk")]
    let (zk_e2, zk_y_com, zk_sigma1, zk_sigma2, zk_r_y) = if Mo::BLINDING {
        use crate::reduce_and_fold::{generate_sigma1_proof, generate_sigma2_proof};
        let y = polynomial.evaluate(point);
        let r_y: F = Mo::sample();
        let e2 = Mo::mask(g2_fin.scale(&y), &setup.h2, &r_e2);
        let y_com = setup.g1_vec[0].scale(&y) + setup.h1.scale(&r_y);
        transcript.append_serde(b"vmv_e2", &e2);
        transcript.append_serde(b"vmv_y_com", &y_com);
        let s1 = generate_sigma1_proof::<E, T>(&y, &r_e2, &r_y, setup, transcript);
        let s2 = generate_sigma2_proof::<E, T>(&r_e1, &-r_d2, setup, transcript);
        (Some(e2), Some(y_com), Some(s1), Some(s2), Some(r_y))
    } else {
        (None, None, None, None, None)
    };

    // v₂ = v_vec · Γ₂,fin (each scalar scales g_fin)
    let v2 = M2::fixed_base_vector_scalar_mul(g2_fin, &v_vec);

    let mut padded_right_vec = right_vec.clone();
    let mut padded_left_vec = left_vec.clone();
    if nu < sigma {
        padded_right_vec.resize(1 << sigma, F::zero());
        padded_left_vec.resize(1 << sigma, F::zero());
    }

    let mut prover_state: DoryProverState<'_, E, Mo> = DoryProverState::new(
        padded_row_commitments, // v1 = T_vec_prime (row commitments, padded)
        v2,                     // v2 = v_vec · g_fin
        Some(v_vec),            // v2_scalars for first-round MSM+pair optimization
        padded_right_vec,       // s1 = right_vec (padded)
        padded_left_vec,        // s2 = left_vec (padded)
        setup,
    );
    prover_state.set_initial_blinds(commit_blind, r_c, r_d2, r_e1, r_e2);

    let num_rounds = nu.max(sigma);
    let mut first_messages = Vec::with_capacity(num_rounds);
    let mut second_messages = Vec::with_capacity(num_rounds);

    for _round in 0..num_rounds {
        let first_msg = prover_state.compute_first_message::<M1, M2>();

        transcript.append_serde(b"d1_left", &first_msg.d1_left);
        transcript.append_serde(b"d1_right", &first_msg.d1_right);
        transcript.append_serde(b"d2_left", &first_msg.d2_left);
        transcript.append_serde(b"d2_right", &first_msg.d2_right);
        transcript.append_serde(b"e1_beta", &first_msg.e1_beta);
        transcript.append_serde(b"e2_beta", &first_msg.e2_beta);

        let beta = transcript.challenge_scalar(b"beta");
        prover_state.apply_first_challenge::<M1, M2>(&beta);
        first_messages.push(first_msg);

        let second_msg = prover_state.compute_second_message::<M1, M2>();

        transcript.append_serde(b"c_plus", &second_msg.c_plus);
        transcript.append_serde(b"c_minus", &second_msg.c_minus);
        transcript.append_serde(b"e1_plus", &second_msg.e1_plus);
        transcript.append_serde(b"e1_minus", &second_msg.e1_minus);
        transcript.append_serde(b"e2_plus", &second_msg.e2_plus);
        transcript.append_serde(b"e2_minus", &second_msg.e2_minus);

        let alpha = transcript.challenge_scalar(b"alpha");
        prover_state.apply_second_challenge::<M1, M2>(&alpha);
        second_messages.push(second_msg);
    }

    let gamma = transcript.challenge_scalar(b"gamma");

    // Fold-Scalars (Dory paper §4.1): absorb the folded public scalars s₁, s₂
    // into the witness.
    prover_state.apply_fold_scalars(&gamma);

    #[cfg(feature = "zk")]
    let scalar_product_proof = if Mo::BLINDING {
        Some(prover_state.scalar_product_proof(transcript))
    } else {
        None
    };

    // Transparent mode reveals the folded witness as the final message; in ZK
    // mode the scalar-product Σ-proof above replaces it.
    let final_message = if Mo::BLINDING {
        None
    } else {
        let msg = prover_state.compute_final_message();
        transcript.append_serde(b"final_e1", &msg.e1);
        transcript.append_serde(b"final_e2", &msg.e2);
        Some(msg)
    };

    let _d = transcript.challenge_scalar(b"d");

    let proof = DoryProof {
        vmv_message,
        first_messages,
        second_messages,
        final_message,
        nu,
        sigma,
        #[cfg(feature = "zk")]
        e2: zk_e2,
        #[cfg(feature = "zk")]
        y_com: zk_y_com,
        #[cfg(feature = "zk")]
        sigma1_proof: zk_sigma1,
        #[cfg(feature = "zk")]
        sigma2_proof: zk_sigma2,
        #[cfg(feature = "zk")]
        scalar_product_proof,
    };
    assert!(
        proof.mode().is_ok(),
        "prover constructed a malformed proof shape"
    );
    #[cfg(feature = "zk")]
    return Ok((proof, zk_r_y));
    #[cfg(not(feature = "zk"))]
    Ok((proof, None))
}

/// Verify an evaluation proof
///
/// Verifies that a committed polynomial evaluates to the claimed value at the given point.
/// Works with both square and non-square matrix layouts (nu ≤ sigma).
///
/// # Algorithm
/// 1. Extract VMV message from proof
/// 2. Compute e2 = Γ2,fin * evaluation (or use proof.e2 in ZK mode)
/// 3. Initialize verifier state with commitment and VMV message
/// 4. Run max(nu, sigma) rounds of reduce-and-fold verification (with automatic padding)
/// 5. Derive gamma and d challenges
/// 6. Verify the final scalar product relation against the Fold-Scalars-updated
///    statement, with the VMV constraint batched in at the d² slot — a single
///    4-pairing check in both modes (transparent: revealed witness + direct VMV
///    pair; ZK: scalar-product Σ-proof + batched Σ₂ proof)
///
/// # Parameters
/// - `commitment`: Polynomial commitment (in GT) - can be a homomorphically combined commitment
/// - `evaluation`: Claimed evaluation result
/// - `point`: Evaluation point (length must equal proof.nu + proof.sigma)
/// - `proof`: Evaluation proof to verify (contains nu and sigma dimensions)
/// - `setup`: Verifier setup
/// - `transcript`: Fiat-Shamir transcript for challenge generation
///
/// # Returns
/// `Ok(())` if proof is valid, `Err(DoryError)` otherwise
///
/// # Homomorphic Verification
/// This function can verify proofs for homomorphically combined polynomials.
/// The commitment parameter should be the combined commitment, and the evaluation
/// should be the evaluation of the combined polynomial.
///
/// # Errors
/// Returns `DoryError::InvalidProof` if verification fails, or other variants
/// if the input parameters are incorrect (e.g., point dimension mismatch).
#[tracing::instrument(skip_all, name = "verify_evaluation_proof")]
pub fn verify_evaluation_proof<F, E, M1, M2, T>(
    commitment: E::GT,
    evaluation: F,
    point: &[F],
    proof: &DoryProof<E::G1, E::G2, E::GT>,
    setup: VerifierSetup<E>,
    transcript: &mut T,
) -> Result<(), DoryError>
where
    F: Field,
    E: PairingCurve,
    E::G1: Group<Scalar = F>,
    E::G2: Group<Scalar = F>,
    E::GT: Group<Scalar = F>,
    M1: DoryRoutines<E::G1>,
    M2: DoryRoutines<E::G2>,
    T: Transcript<Curve = E>,
{
    let nu = proof.nu;
    let sigma = proof.sigma;

    if point.len() != nu + sigma {
        return Err(DoryError::InvalidPointDimension {
            expected: nu + sigma,
            actual: point.len(),
        });
    }

    if nu > sigma {
        return Err(DoryError::InvalidSize {
            expected: sigma,
            actual: nu,
        });
    }

    // Single shape gate: a proof must be fully transparent or fully ZK;
    // mix-and-match shapes are rejected before anything is absorbed, and all
    // optional fields are only ever read through the returned mode.
    let mode = proof.mode()?;

    let vmv_message = &proof.vmv_message;
    transcript.append_serde(b"vmv_c", &vmv_message.c);
    transcript.append_serde(b"vmv_d2", &vmv_message.d2);
    transcript.append_serde(b"vmv_e1", &vmv_message.e1);

    // ZK mode: the Σ₁ proof is checked here; the Σ₂ proof (VMV constraint) is
    // only absorbed here — its check is batched into the final multi-pairing
    // (verify_final), so it is carried to the end together with its challenge.
    #[cfg(feature = "zk")]
    let (e2, zk_final) = match mode {
        ProofMode::Zk {
            e2,
            y_com,
            sigma1,
            sigma2,
            scalar_product,
        } => {
            use crate::reduce_and_fold::{absorb_sigma2_proof, verify_sigma1_proof};
            transcript.append_serde(b"vmv_e2", e2);
            transcript.append_serde(b"vmv_y_com", y_com);
            verify_sigma1_proof::<E, T>(e2, y_com, sigma1, &setup, transcript)?;
            let sigma2_c = absorb_sigma2_proof::<E, T>(sigma2, transcript);
            (*e2, Some((scalar_product, sigma2, sigma2_c)))
        }
        ProofMode::Transparent(..) => (setup.g2_0.scale(&evaluation), None),
    };
    #[cfg(not(feature = "zk"))]
    let e2 = setup.g2_0.scale(&evaluation);

    // Folded-scalar accumulation with per-round coordinates.
    // num_rounds = sigma (we fold column dimensions).
    let num_rounds = sigma;

    // Bounds check: reject proofs with mismatched message counts or that exceed setup capacity.
    let max_rounds = setup.max_log_n / 2;
    if num_rounds > max_rounds
        || proof.first_messages.len() != num_rounds
        || proof.second_messages.len() != num_rounds
    {
        return Err(DoryError::InvalidProof);
    }

    // s1 (right/prover): the σ column coordinates in natural order (LSB→MSB).
    // No padding here: the verifier folds across the σ column dimensions.
    // With MSB-first folding, these coordinates are only consumed after the first σ−ν rounds,
    // which correspond to the padded MSB dimensions on the left tensor, matching the prover.
    let s1_coords: Vec<F> = point[..sigma].to_vec();
    // s2 (left/prover): the ν row coordinates in natural order, followed by zeros for the extra
    // MSB dimensions. Conceptually this is s ⊗ [1,0]^(σ−ν): under MSB-first folds, the first
    // σ−ν rounds multiply s2 by α⁻¹ while contributing no right halves (since those entries are 0).
    let mut s2_coords: Vec<F> = vec![F::zero(); sigma];
    s2_coords[..nu].copy_from_slice(&point[sigma..sigma + nu]);

    let mut verifier_state = DoryVerifierState::new(
        vmv_message.c,  // c from VMV message
        commitment,     // d1 = commitment
        vmv_message.d2, // d2 from VMV message
        vmv_message.e1, // e1 from VMV message
        e2,             // e2 computed from evaluation
        s1_coords,      // s1: columns c0..c_{σ−1} (LSB→MSB), no padding; folded across σ dims
        s2_coords,      // s2: rows r0..r_{ν−1} then zeros in MSB dims (emulates s ⊗ [1,0]^(σ−ν))
        num_rounds,
        setup.clone(),
    );

    for round in 0..num_rounds {
        let first_msg = &proof.first_messages[round];
        let second_msg = &proof.second_messages[round];

        transcript.append_serde(b"d1_left", &first_msg.d1_left);
        transcript.append_serde(b"d1_right", &first_msg.d1_right);
        transcript.append_serde(b"d2_left", &first_msg.d2_left);
        transcript.append_serde(b"d2_right", &first_msg.d2_right);
        transcript.append_serde(b"e1_beta", &first_msg.e1_beta);
        transcript.append_serde(b"e2_beta", &first_msg.e2_beta);
        let beta = transcript.challenge_scalar(b"beta");

        transcript.append_serde(b"c_plus", &second_msg.c_plus);
        transcript.append_serde(b"c_minus", &second_msg.c_minus);
        transcript.append_serde(b"e1_plus", &second_msg.e1_plus);
        transcript.append_serde(b"e1_minus", &second_msg.e1_minus);
        transcript.append_serde(b"e2_plus", &second_msg.e2_plus);
        transcript.append_serde(b"e2_minus", &second_msg.e2_minus);
        let alpha = transcript.challenge_scalar(b"alpha");

        verifier_state.process_round(first_msg, second_msg, &alpha, &beta)?;
    }

    let gamma = transcript.challenge_scalar(b"gamma");

    // ZK mode: absorb the scalar product proof into the transcript before
    // deriving d
    #[cfg(feature = "zk")]
    if let Some((sp, sigma2, sigma2_c)) = zk_final {
        use crate::reduce_and_fold::absorb_scalar_product_proof;
        let sigma_c = absorb_scalar_product_proof::<E, T>(sp, transcript);
        let d = transcript.challenge_scalar(b"d");
        return verifier_state.verify_final(
            FinalCheck::Zk {
                scalar_product: sp,
                sigma_c,
                sigma2,
                sigma2_c,
            },
            &gamma,
            &d,
        );
    }

    // Transparent mode: the revealed folded witness is the final message.
    let msg = match mode {
        ProofMode::Transparent(msg, _) => msg,
        #[cfg(feature = "zk")]
        ProofMode::Zk { .. } => return Err(DoryError::InvalidProof),
    };
    transcript.append_serde(b"final_e1", &msg.e1);
    transcript.append_serde(b"final_e2", &msg.e2);
    let d = transcript.challenge_scalar(b"d");

    verifier_state.verify_final(FinalCheck::Transparent(msg), &gamma, &d)
}
