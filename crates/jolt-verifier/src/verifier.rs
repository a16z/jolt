//! Schedule-driven proof verification.
//!
//! The verifier walks a flat `Vec<VerifierOp>` — one match arm per variant,
//! mirroring the prover's flat `Vec<Op>` execution model. No per-stage
//! hand-written logic; the schedule encodes the full Fiat-Shamir replay.

use std::collections::HashMap;

use jolt_compiler::module::{
    ClaimFactor, ClaimFormula, PointNormalization, R1CSMatrix, VerifierOp,
};
use jolt_compiler::PolynomialId;
use jolt_field::Field;
use jolt_openings::{OpeningReduction, OpeningsError, VerifierClaim};
use jolt_poly::EqPolynomial;
use jolt_r1cs::R1csKey;
use jolt_sumcheck::{SumcheckClaim, SumcheckVerifier};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use jolt_verifier_backend::{
    helpers::{
        eq_eval as backend_eq_eval, identity_mle as backend_identity_mle,
        lagrange_basis_eval as backend_lagrange_basis_eval,
        lagrange_evals as backend_lagrange_evals,
        lagrange_kernel_eval as backend_lagrange_kernel_eval, lt_mle as backend_lt_mle,
        sparse_block_eval as backend_sparse_block_eval,
    },
    FieldBackend, Native,
};
use num_traits::Zero;

use crate::config::ProverConfig;
use crate::error::JoltError;
use crate::key::JoltVerifyingKey;
use crate::proof::{JoltProof, StageProof};
use crate::TRANSCRIPT_LABEL;

/// End-to-end proof verification using the [`Native`] backend.
///
/// Thin wrapper around [`verify_with_backend`]; identical behavior. Use
/// [`verify_with_backend`] directly with a [`Tracing`] backend to capture
/// the verifier's field-arithmetic AST for recursion / Lean export.
pub fn verify<F, PCS>(
    key: &JoltVerifyingKey<F, PCS>,
    proof: &JoltProof<F, PCS>,
    expected_io_hash: &[u8; 32],
) -> Result<(), JoltError>
where
    F: Field,
    PCS: OpeningReduction<Field = F>,
    PCS::Output: AppendToTranscript,
{
    let mut backend = Native::<F>::new();
    verify_with_backend(&mut backend, key, proof, expected_io_hash)
}

/// Backend-polymorphic proof verification.
///
/// Walks the flat [`VerifierOp`] sequence from the verifying key, replaying
/// the Fiat-Shamir transcript in lockstep with the prover. All field
/// arithmetic for the verifier's checks (sumcheck round combination, claim
/// formula evaluation, `CheckOutput` equality) routes through `B`; the
/// transcript-bytes / PCS surface stays native (it deals with serialized
/// commitments, not field arithmetic).
///
/// State buckets are kept as parallel `(F, B::Scalar)` pairs:
///
///   * `_f` slots feed the transcript and the PCS reducer (which require
///     a concrete field value to hash / commit).
///   * `_w` slots feed [`evaluate_formula_with_backend`] and
///     [`SumcheckVerifier::verify_with_backend`] so the AST sees every op.
///
/// For `Native<F>` this redundant pair degenerates to two F copies; for
/// `Tracing<F>` the F side keeps the verifier compatible with the existing
/// transcript / PCS while the wrapped side records the symbolic graph.
///
/// # Errors
///
/// Returns the same error variants as [`verify`]. `CheckOutput` mismatches
/// surface as [`JoltError::EvaluationMismatch`] (Native [`assert_eq`])
/// or are recorded as AST assertions to be discharged by the outer proof
/// (Tracing).
#[allow(clippy::too_many_lines)]
pub fn verify_with_backend<B, PCS>(
    backend: &mut B,
    key: &JoltVerifyingKey<B::F, PCS>,
    proof: &JoltProof<B::F, PCS>,
    expected_io_hash: &[u8; 32],
) -> Result<(), JoltError>
where
    B: FieldBackend,
    PCS: OpeningReduction<Field = B::F>,
    PCS::Output: AppendToTranscript,
{
    if proof.config.io_hash != *expected_io_hash {
        fn fmt_hash(h: &[u8; 32]) -> String {
            let mut s = String::with_capacity(64);
            for b in h {
                use std::fmt::Write;
                let _ = write!(s, "{b:02x}");
            }
            s
        }
        return Err(JoltError::IoHashMismatch {
            proof_hash: fmt_hash(&proof.config.io_hash),
            expected_hash: fmt_hash(expected_io_hash),
        });
    }
    proof.config.validate().map_err(JoltError::InvalidProof)?;

    let schedule = &key.schedule;
    let mut transcript = backend.new_transcript(TRANSCRIPT_LABEL);

    let mut challenges_f = vec![B::F::zero(); schedule.num_challenges];
    let mut challenges_w: Vec<B::Scalar> = (0..schedule.num_challenges)
        .map(|_| backend.const_zero())
        .collect();
    let mut evaluations_f: HashMap<PolynomialId, B::F> = HashMap::new();
    let mut evaluations_w: HashMap<PolynomialId, B::Scalar> = HashMap::new();
    let mut sumcheck_points_f: Vec<Vec<B::F>> = vec![Vec::new(); schedule.num_stages];
    let mut sumcheck_points_w: Vec<Vec<B::Scalar>> = vec![Vec::new(); schedule.num_stages];
    let mut final_evals_w: Vec<B::Scalar> = (0..schedule.num_stages)
        .map(|_| backend.const_zero())
        .collect();
    let mut commitment_map: HashMap<PolynomialId, PCS::Output> = HashMap::new();
    let mut commitments = proof.commitments.iter();
    let mut stage_proofs = proof.stage_proofs.iter();
    let mut current_stage: Option<&StageProof<B::F>> = None;
    let mut eval_cursor: usize = 0;
    let mut round_poly_cursor: usize = 0;
    let mut pcs_claims: Vec<VerifierClaim<B::F, PCS::Output>> = Vec::new();

    for op in &schedule.ops {
        match op {
            VerifierOp::Preamble => {
                transcript.append(&proof.config);
            }

            VerifierOp::BeginStage => {
                eval_cursor = 0;
                round_poly_cursor = 0;
                current_stage = Some(
                    stage_proofs
                        .next()
                        .ok_or_else(|| JoltError::InvalidProof("missing stage proof".into()))?,
                );
            }

            VerifierOp::AbsorbCommitment { poly, tag } => {
                let slot = commitments
                    .next()
                    .ok_or_else(|| JoltError::InvalidProof("missing commitment".into()))?;
                // `None` means the prover skipped this commit (all-zero
                // advice poly) to match jolt-core's transcript. Skip the
                // transcript append on the verifier side symmetrically;
                // downstream CollectOpeningClaim's `commitment_map.get()`
                // already handles a missing entry.
                if let Some(c) = slot {
                    transcript.append(&LabelWithCount(tag.as_bytes(), c.serialized_len()));
                    c.append_to_transcript(&mut transcript);
                    let _ = commitment_map.insert(*poly, c.clone());
                }
            }

            VerifierOp::Squeeze { challenge } => {
                let (val, val_w) = backend.squeeze(&mut transcript, "squeeze");
                challenges_f[challenge.0] = val;
                challenges_w[challenge.0] = val_w;
            }

            VerifierOp::AppendDomainSeparator { tag } => {
                // Match jolt-core's `append_bytes(label, &[])`: two update_state calls
                let label = tag.as_bytes();
                let mut packed = [0u8; 32];
                packed[..label.len()].copy_from_slice(label);
                transcript.append_bytes(&packed);
                transcript.append_bytes(&[]);
            }

            VerifierOp::AbsorbRoundPoly { num_coeffs: _, tag } => {
                // Used for *uniskip* round polynomials that sit before a
                // VerifySumcheck in the schedule (e.g. the outer
                // univariate-skip protocol). Their coefficients live in the
                // same `sp.round_polys` array as the sumcheck rounds, so we
                // bump the cursor and absorb them into the transcript here.
                // The corresponding eval is published separately via a
                // following RecordEvals + AbsorbEvals pair, so the round
                // poly itself does not need backend wrapping.
                let sp = current_stage.ok_or_else(|| {
                    JoltError::InvalidProof("no active stage proof for round poly".into())
                })?;
                let poly = sp
                    .round_polys
                    .round_polynomials
                    .get(round_poly_cursor)
                    .ok_or_else(|| {
                        JoltError::InvalidProof(format!(
                            "missing round poly at cursor {round_poly_cursor}"
                        ))
                    })?;
                let coeffs = poly.coefficients();
                transcript.append(&LabelWithCount(tag.as_bytes(), coeffs.len() as u64));
                for coeff in coeffs {
                    coeff.append_to_transcript(&mut transcript);
                }
                round_poly_cursor += 1;
            }

            VerifierOp::VerifySumcheck {
                instances,
                stage,
                batch_challenges,
                claim_tag,
                sumcheck_challenge_slots,
            } => {
                let sp = current_stage.ok_or_else(|| {
                    JoltError::InvalidProof("no active stage proof for sumcheck".into())
                })?;

                let max_rounds = instances.iter().map(|i| i.num_rounds).max().unwrap_or(0);
                let max_degree = instances.iter().map(|i| i.degree).max().unwrap_or(0);

                // Evaluate each instance's input claim through both worlds
                // in parallel. The native value drives the transcript when
                // the batch is more than one instance; the wrapped value
                // feeds the sumcheck verifier's running sum.
                let mut instance_claims_f: Vec<B::F> = Vec::with_capacity(instances.len());
                let mut instance_claims_w: Vec<B::Scalar> = Vec::with_capacity(instances.len());
                for inst in instances {
                    let val_f = evaluate_formula(
                        &inst.input_claim,
                        &evaluations_f,
                        &challenges_f,
                        &sumcheck_points_f,
                        None,
                        None,
                        &key.r1cs_key,
                        &proof.config,
                    )?;
                    let val_w = evaluate_formula_with_backend(
                        backend,
                        &inst.input_claim,
                        &evaluations_w,
                        &challenges_w,
                        &sumcheck_points_w,
                        None,
                        None,
                        &key.r1cs_key,
                        &proof.config,
                    )?;
                    instance_claims_f.push(val_f);
                    instance_claims_w.push(val_w);
                }

                let (combined_claim_f, combined_claim_w): (B::F, B::Scalar) =
                    if batch_challenges.is_empty() {
                        let mut acc_f = B::F::zero();
                        let mut acc_w = backend.const_zero();
                        for ((&c_f, c_w), inst) in instance_claims_f
                            .iter()
                            .zip(instance_claims_w.iter())
                            .zip(instances.iter())
                        {
                            let scale = 1u64 << (max_rounds - inst.num_rounds);
                            acc_f += c_f * B::F::from_u64(scale);
                            let scale_w = backend.const_i128(i128::from(scale));
                            let scaled = backend.mul(c_w, &scale_w);
                            acc_w = backend.add(&acc_w, &scaled);
                        }
                        (acc_f, acc_w)
                    } else {
                        let tag = claim_tag.as_ref().expect("claim_tag required for batched");
                        for &claim_val in &instance_claims_f {
                            transcript.append(&Label(tag.as_bytes()));
                            claim_val.append_to_transcript(&mut transcript);
                        }
                        for &ch_idx in batch_challenges {
                            let (val, val_w) = backend.squeeze(&mut transcript, "batch_squeeze");
                            challenges_f[ch_idx.0] = val;
                            challenges_w[ch_idx.0] = val_w;
                        }
                        let mut acc_f = B::F::zero();
                        let mut acc_w = backend.const_zero();
                        for (((&c_f, c_w), inst), &ch_idx) in instance_claims_f
                            .iter()
                            .zip(instance_claims_w.iter())
                            .zip(instances.iter())
                            .zip(batch_challenges.iter())
                        {
                            let coeff_f = challenges_f[ch_idx.0];
                            let scale = 1u64 << (max_rounds - inst.num_rounds);
                            acc_f += coeff_f * c_f * B::F::from_u64(scale);
                            let coeff_w = &challenges_w[ch_idx.0];
                            let cw = backend.mul(coeff_w, c_w);
                            let scale_w = backend.const_i128(i128::from(scale));
                            let scaled = backend.mul(&cw, &scale_w);
                            acc_w = backend.add(&acc_w, &scaled);
                        }
                        (acc_f, acc_w)
                    };

                let claim = SumcheckClaim {
                    num_vars: max_rounds,
                    degree: max_degree,
                    claimed_sum: combined_claim_f,
                };
                let round_polys = &sp.round_polys.round_polynomials
                    [round_poly_cursor..round_poly_cursor + max_rounds];
                let (fe_w, sc_w, sc_f) = SumcheckVerifier::verify_with_backend(
                    backend,
                    &claim,
                    round_polys,
                    combined_claim_w,
                    &mut transcript,
                    Some(b"sumcheck_poly"),
                    true,
                )
                .map_err(JoltError::Sumcheck)?;
                round_poly_cursor += max_rounds;

                for (i, slot) in sumcheck_challenge_slots.iter().enumerate() {
                    if i < sc_f.len() {
                        challenges_f[slot.0] = sc_f[i];
                        challenges_w[slot.0] = sc_w[i].clone();
                    }
                }

                final_evals_w[*stage] = fe_w;
                sumcheck_points_f[*stage] = sc_f;
                sumcheck_points_w[*stage] = sc_w;
            }

            VerifierOp::RecordEvals { evals } => {
                let sp = current_stage.ok_or_else(|| {
                    JoltError::InvalidProof("no active stage proof for evals".into())
                })?;
                for eval_desc in evals {
                    let value = *sp.evals.get(eval_cursor).ok_or_else(|| {
                        JoltError::InvalidProof(format!(
                            "missing eval {eval_cursor} in stage proof"
                        ))
                    })?;
                    let value_w = backend.wrap_proof(value, "record_eval");
                    let _ = evaluations_f.insert(eval_desc.poly, value);
                    let _ = evaluations_w.insert(eval_desc.poly, value_w);
                    eval_cursor += 1;
                }
            }

            VerifierOp::AbsorbEvals { polys, tag } => {
                for pi in polys {
                    if let Some(&val) = evaluations_f.get(pi) {
                        transcript.append(&Label(tag.as_bytes()));
                        val.append_to_transcript(&mut transcript);
                    }
                }
            }

            VerifierOp::CheckOutput {
                instances,
                stage,
                batch_challenges,
            } => {
                let sp = current_stage.ok_or_else(|| {
                    JoltError::InvalidProof("no active stage for CheckOutput".into())
                })?;
                let max_rounds = instances.iter().map(|i| i.num_rounds).max().unwrap_or(0);
                let raw_point_w = &sumcheck_points_w[*stage];
                let mut combined_output_w = backend.const_zero();
                let stage_evals_w: Vec<B::Scalar> = sp
                    .evals
                    .iter()
                    .map(|&v| backend.wrap_proof(v, "stage_eval"))
                    .collect();

                for (i, inst) in instances.iter().enumerate() {
                    let offset = max_rounds - inst.num_rounds;
                    let normalized_w =
                        apply_normalization(&raw_point_w[offset..], inst.normalize.as_ref());
                    let output_w = evaluate_formula_with_backend(
                        backend,
                        &inst.output_check,
                        &evaluations_w,
                        &challenges_w,
                        &sumcheck_points_w,
                        Some((*stage, &normalized_w)),
                        Some(&stage_evals_w),
                        &key.r1cs_key,
                        &proof.config,
                    )?;
                    let scaled_w = if batch_challenges.is_empty() {
                        output_w
                    } else {
                        let coeff_w = &challenges_w[batch_challenges[i].0];
                        backend.mul(coeff_w, &output_w)
                    };
                    combined_output_w = backend.add(&combined_output_w, &scaled_w);
                }

                backend
                    .assert_eq(&final_evals_w[*stage], &combined_output_w, "check_output")
                    .map_err(|_| JoltError::EvaluationMismatch {
                        stage: *stage,
                        reason: "sumcheck final eval does not match batched composition".into(),
                    })?;
            }

            VerifierOp::CollectOpeningClaim { poly, at_stage } => {
                if let Some(commitment) = commitment_map.get(poly) {
                    let eval = evaluations_f.get(poly).copied().ok_or_else(|| {
                        JoltError::InvalidProof(format!(
                            "evaluation for committed poly {poly:?} not set"
                        ))
                    })?;
                    pcs_claims.push(VerifierClaim {
                        commitment: commitment.clone(),
                        point: sumcheck_points_f[at_stage.0].clone(),
                        eval,
                    });
                }
            }

            VerifierOp::VerifyOpenings => {
                if pcs_claims.is_empty() {
                    continue;
                }
                let claims = std::mem::take(&mut pcs_claims);
                let reduced =
                    PCS::reduce_verifier(claims, &mut transcript).map_err(JoltError::Opening)?;

                if reduced.len() != proof.opening_proofs.len() {
                    return Err(JoltError::Opening(OpeningsError::VerificationFailed));
                }

                for (claim, opening_proof) in reduced.iter().zip(proof.opening_proofs.iter()) {
                    PCS::verify(
                        &claim.commitment,
                        &claim.point,
                        claim.eval,
                        opening_proof,
                        &key.pcs_setup,
                        &mut transcript,
                    )
                    .map_err(JoltError::Opening)?;
                }
            }
        }
    }

    Ok(())
}

/// Apply point normalization to raw sumcheck challenges.
fn apply_normalization<F: Clone>(raw: &[F], normalize: Option<&PointNormalization>) -> Vec<F> {
    match normalize {
        None => raw.to_vec(),
        Some(PointNormalization::Reverse) => raw.iter().rev().cloned().collect(),
        Some(PointNormalization::Segments {
            sizes,
            output_order,
        }) => {
            let mut segments: Vec<Vec<F>> = Vec::with_capacity(sizes.len());
            let mut cursor = 0;
            for &size in sizes {
                let seg: Vec<F> = raw[cursor..cursor + size].iter().rev().cloned().collect();
                segments.push(seg);
                cursor += size;
            }
            let mut result = Vec::with_capacity(raw.len());
            for &idx in output_order {
                result.extend_from_slice(&segments[idx]);
            }
            result
        }
    }
}

/// Backend-aware claim formula evaluation.
///
/// Mirrors [`evaluate_formula`] but routes every field operation through
/// the [`FieldBackend`] so a Tracing backend can capture the entire
/// formula (sum-of-products + the inner Lagrange / matrix-MLE / preprocessed
/// poly subcomputations) into an AST.
///
/// Native callers should pass `Native` as the backend and incur zero
/// overhead.
///
/// # Arguments
///
/// Same as [`evaluate_formula`], but every state bucket is `B::Scalar`-typed.
///
/// `r1cs_matrix_const(coeff)` is invoked for every nonzero entry the
/// formula touches; backends that want to dedupe constant matrix
/// coefficients (e.g. `F::one()`, `F::zero()`) should override
/// [`FieldBackend::wrap_public`] / `const_i128` accordingly.
///
/// # Errors
///
/// Returns [`JoltError::InvalidProof`] when the formula references an
/// evaluation that hasn't been recorded yet, when [`ClaimFactor::StagedEval`]
/// or [`ClaimFactor::LagrangeKernel`] (prover-only / unsupported variants)
/// appear, or when [`StageEval`] indexes out of bounds. Lagrange
/// inversions fail with [`BackendError`](jolt_verifier_backend::BackendError),
/// which is wrapped as [`JoltError::InvalidProof`].
// Wired into the VerifierOp interpreter in step 4c of the FieldBackend cutover.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn evaluate_formula_with_backend<B>(
    backend: &mut B,
    formula: &ClaimFormula,
    evaluations: &HashMap<PolynomialId, B::Scalar>,
    challenges: &[B::Scalar],
    sumcheck_points: &[Vec<B::Scalar>],
    point_override: Option<(usize, &[B::Scalar])>,
    stage_evals: Option<&[B::Scalar]>,
    r1cs_key: &R1csKey<B::F>,
    config: &ProverConfig,
) -> Result<B::Scalar, JoltError>
where
    B: FieldBackend,
{
    let mut sum = backend.const_zero();
    for term in &formula.terms {
        let coeff_w = backend.const_i128(term.coeff);
        let mut product = coeff_w;
        for factor in &term.factors {
            let factor_val: B::Scalar = match factor {
                ClaimFactor::Eval(poly) => evaluations.get(poly).cloned().ok_or_else(|| {
                    JoltError::InvalidProof(format!(
                        "evaluation {poly:?} referenced before available"
                    ))
                })?,
                ClaimFactor::Challenge(i) => challenges[i.0].clone(),
                ClaimFactor::EqChallengePair { a, b } => {
                    let one_w = backend.const_one();
                    let ra = challenges[a.0].clone();
                    let rb = challenges[b.0].clone();
                    let ab = backend.mul(&ra, &rb);
                    let one_minus_a = backend.sub(&one_w, &ra);
                    let one_minus_b = backend.sub(&one_w, &rb);
                    let cross = backend.mul(&one_minus_a, &one_minus_b);
                    backend.add(&ab, &cross)
                }
                ClaimFactor::EqEval {
                    challenges: chs,
                    at_stage,
                } => {
                    let r: Vec<B::Scalar> =
                        chs.iter().map(|&ci| challenges[ci.0].clone()).collect();
                    let s = resolve_point(sumcheck_points, point_override, at_stage.0);
                    backend_eq_eval(backend, &r, s)
                }
                ClaimFactor::EqEvalSlice {
                    challenges: chs,
                    at_stage,
                    offset,
                } => {
                    let r: Vec<B::Scalar> =
                        chs.iter().map(|&ci| challenges[ci.0].clone()).collect();
                    let s = resolve_point(sumcheck_points, point_override, at_stage.0);
                    backend_eq_eval(backend, &r, &s[*offset..*offset + r.len()])
                }
                ClaimFactor::LagrangeKernelDomain {
                    tau_challenge,
                    at_challenge,
                    domain_size,
                    domain_start,
                } => {
                    let tau = challenges[tau_challenge.0].clone();
                    let at = challenges[at_challenge.0].clone();
                    backend_lagrange_kernel_eval(backend, *domain_start, *domain_size, &tau, &at)
                        .map_err(|e| JoltError::InvalidProof(e.to_string()))?
                }
                ClaimFactor::LagrangeWeight {
                    challenge,
                    domain_size,
                    domain_start,
                    basis_index,
                } => {
                    let r = challenges[challenge.0].clone();
                    backend_lagrange_basis_eval(
                        backend,
                        *domain_start,
                        *domain_size,
                        *basis_index,
                        &r,
                    )
                    .map_err(|e| JoltError::InvalidProof(e.to_string()))?
                }
                ClaimFactor::UniformR1CSEval {
                    matrix,
                    eval_polys,
                    at_challenge,
                    num_constraints,
                    domain_start,
                } => {
                    let r0 = challenges[at_challenge.0].clone();
                    let basis =
                        backend_lagrange_evals(backend, *domain_start, *num_constraints, &r0)
                            .map_err(|e| JoltError::InvalidProof(e.to_string()))?;
                    let mut z: Vec<B::Scalar> = Vec::with_capacity(1 + eval_polys.len());
                    z.push(backend.const_one());
                    for p in eval_polys {
                        z.push(evaluations.get(p).cloned().ok_or_else(|| {
                            JoltError::InvalidProof(format!("R1CS eval {p:?} not available"))
                        })?);
                    }
                    let rows = match matrix {
                        R1CSMatrix::A => &r1cs_key.matrices.a,
                        R1CSMatrix::B => &r1cs_key.matrices.b,
                    };
                    let mut acc = backend.const_zero();
                    for (k, row) in rows[..*num_constraints].iter().enumerate() {
                        let mut dot = backend.const_zero();
                        for &(j, coeff) in row {
                            let coeff_w = backend.wrap_public(coeff, "r1cs_matrix_coeff");
                            let term = backend.mul(&coeff_w, &z[j]);
                            dot = backend.add(&dot, &term);
                        }
                        let weighted = backend.mul(&basis[k], &dot);
                        acc = backend.add(&acc, &weighted);
                    }
                    acc
                }
                ClaimFactor::GroupSplitR1CSEval {
                    matrix,
                    eval_polys,
                    at_r0,
                    at_r_group,
                    group0_indices,
                    group1_indices,
                    domain_size,
                    domain_start,
                } => {
                    let r0 = challenges[at_r0.0].clone();
                    let r_group = challenges[at_r_group.0].clone();
                    let basis = backend_lagrange_evals(backend, *domain_start, *domain_size, &r0)
                        .map_err(|e| JoltError::InvalidProof(e.to_string()))?;
                    let mut z: Vec<B::Scalar> = Vec::with_capacity(1 + eval_polys.len());
                    z.push(backend.const_one());
                    for p in eval_polys {
                        z.push(evaluations.get(p).cloned().ok_or_else(|| {
                            JoltError::InvalidProof(format!("R1CS eval {p:?} not available"))
                        })?);
                    }
                    let rows = match matrix {
                        R1CSMatrix::A => &r1cs_key.matrices.a,
                        R1CSMatrix::B => &r1cs_key.matrices.b,
                    };
                    // Closure over `&mut backend` requires explicit lifetime
                    // on the captured slice; inline-expand to keep borrow checker happy.
                    let mut g0 = backend.const_zero();
                    for (i, &idx) in group0_indices.iter().enumerate() {
                        let mut dot = backend.const_zero();
                        for &(j, coeff) in &rows[idx] {
                            let coeff_w = backend.wrap_public(coeff, "r1cs_matrix_coeff");
                            let term = backend.mul(&coeff_w, &z[j]);
                            dot = backend.add(&dot, &term);
                        }
                        let weighted = backend.mul(&basis[i], &dot);
                        g0 = backend.add(&g0, &weighted);
                    }
                    let mut g1 = backend.const_zero();
                    for (i, &idx) in group1_indices.iter().enumerate() {
                        let mut dot = backend.const_zero();
                        for &(j, coeff) in &rows[idx] {
                            let coeff_w = backend.wrap_public(coeff, "r1cs_matrix_coeff");
                            let term = backend.mul(&coeff_w, &z[j]);
                            dot = backend.add(&dot, &term);
                        }
                        let weighted = backend.mul(&basis[i], &dot);
                        g1 = backend.add(&g1, &weighted);
                    }
                    // g0 + r_group * (g1 - g0)
                    let diff = backend.sub(&g1, &g0);
                    let scaled = backend.mul(&r_group, &diff);
                    backend.add(&g0, &scaled)
                }
                ClaimFactor::StageEval(index) => {
                    let evals = stage_evals.ok_or_else(|| {
                        JoltError::InvalidProof("StageEval used but no stage_evals provided".into())
                    })?;
                    evals.get(*index).cloned().ok_or_else(|| {
                        JoltError::InvalidProof(format!(
                            "StageEval({index}) out of bounds (len={})",
                            evals.len()
                        ))
                    })?
                }
                ClaimFactor::PreprocessedPolyEval { poly, at_stage } => {
                    let point = resolve_point(sumcheck_points, point_override, at_stage.0);
                    evaluate_preprocessed_poly_with_backend(backend, *poly, point, config)?
                }
                ClaimFactor::StagedEval { .. } => {
                    return Err(JoltError::InvalidProof(
                        "StagedEval is prover-only; not supported in verifier formulas".into(),
                    ));
                }
                ClaimFactor::LagrangeKernel { .. } => {
                    return Err(JoltError::InvalidProof(format!(
                        "unsupported claim factor {factor:?} in compiled schedule"
                    )));
                }
            };
            product = backend.mul(&product, &factor_val);
        }
        sum = backend.add(&sum, &product);
    }
    Ok(sum)
}

/// Backend-aware preprocessed polynomial evaluation.
///
/// Mirrors [`evaluate_preprocessed_poly`] but routes through the
/// [`FieldBackend`]. Used inside [`evaluate_formula_with_backend`] for
/// [`ClaimFactor::PreprocessedPolyEval`].
// Wired into the VerifierOp interpreter in step 4c.
#[allow(dead_code)]
fn evaluate_preprocessed_poly_with_backend<B>(
    backend: &mut B,
    poly: PolynomialId,
    point: &[B::Scalar],
    config: &ProverConfig,
) -> Result<B::Scalar, JoltError>
where
    B: FieldBackend,
{
    match poly {
        PolynomialId::IoMask => {
            let io_start = config.input_word_offset as u128;
            let io_end = ((config.memory_start - config.ram_lowest_address) / 8) as u128;
            let lt_end = backend_lt_mle(backend, point, io_end);
            let lt_start = backend_lt_mle(backend, point, io_start);
            Ok(backend.sub(&lt_end, &lt_start))
        }
        PolynomialId::RamUnmap => {
            let identity = backend_identity_mle(backend, point);
            let eight = backend.const_i128(8);
            let scaled = backend.mul(&identity, &eight);
            let base = backend.const_i128(i128::from(config.ram_lowest_address));
            Ok(backend.add(&scaled, &base))
        }
        PolynomialId::ValIo => eval_io_mle_with_backend(backend, point, config),
        _ => Err(JoltError::InvalidProof(format!(
            "PreprocessedPolyEval({poly:?}) not a known preprocessed polynomial"
        ))),
    }
}

/// Backend-aware [`eval_io_mle`] mirror.
// Wired into the VerifierOp interpreter in step 4c.
#[allow(dead_code)]
fn eval_io_mle_with_backend<B>(
    backend: &mut B,
    r: &[B::Scalar],
    config: &ProverConfig,
) -> Result<B::Scalar, JoltError>
where
    B: FieldBackend,
{
    let io_end_words = ((config.memory_start - config.ram_lowest_address) / 8) as usize;
    let io_len = io_end_words.next_power_of_two().max(1);
    let num_io_vars = io_len.trailing_zeros() as usize;

    if num_io_vars > r.len() {
        return Err(JoltError::InvalidProof(format!(
            "ValIo: num_io_vars ({num_io_vars}) > point len ({})",
            r.len()
        )));
    }

    let (r_hi, r_lo) = r.split_at(r.len() - num_io_vars);

    let one_w = backend.const_one();
    let mut hi_scale = backend.const_one();
    for ri in r_hi {
        let one_minus_ri = backend.sub(&one_w, ri);
        hi_scale = backend.mul(&hi_scale, &one_minus_ri);
    }

    let mut acc = backend.const_zero();

    if !config.inputs.is_empty() {
        let words = bytes_to_words(&config.inputs);
        let block = backend_sparse_block_eval(backend, config.input_word_offset, &words, r_lo);
        acc = backend.add(&acc, &block);
    }

    if !config.outputs.is_empty() {
        let words = bytes_to_words(&config.outputs);
        let block = backend_sparse_block_eval(backend, config.output_word_offset, &words, r_lo);
        acc = backend.add(&acc, &block);
    }

    let panic_block = backend_sparse_block_eval(
        backend,
        config.panic_word_offset,
        &[config.panic as u64],
        r_lo,
    );
    acc = backend.add(&acc, &panic_block);

    if !config.panic {
        let term_block =
            backend_sparse_block_eval(backend, config.termination_word_offset, &[1u64], r_lo);
        acc = backend.add(&acc, &term_block);
    }

    Ok(backend.mul(&hi_scale, &acc))
}

/// Evaluate a symbolic claim formula using accumulated verifier state.
///
/// `point_override`: when `Some((stage, point))`, `EqEval`/`EqEvalSlice`
/// factors at that stage use the override point (for normalized CheckOutput).
///
/// `stage_evals`: when `Some(evals)`, `StageEval(i)` resolves to `evals[i]`.
/// Required for output-check formulas in batched stages.
#[allow(clippy::too_many_arguments)]
fn evaluate_formula<F: Field>(
    formula: &ClaimFormula,
    evaluations: &HashMap<PolynomialId, F>,
    challenges: &[F],
    sumcheck_points: &[Vec<F>],
    point_override: Option<(usize, &[F])>,
    stage_evals: Option<&[F]>,
    r1cs_key: &R1csKey<F>,
    config: &ProverConfig,
) -> Result<F, JoltError> {
    let mut sum = F::zero();
    for term in &formula.terms {
        let mut product = F::from_i128(term.coeff);
        for factor in &term.factors {
            let factor_val: F = match factor {
                ClaimFactor::Eval(poly) => evaluations.get(poly).copied().ok_or_else(|| {
                    JoltError::InvalidProof(format!(
                        "evaluation {poly:?} referenced before available"
                    ))
                })?,
                ClaimFactor::Challenge(i) => challenges[i.0],
                ClaimFactor::EqChallengePair { a, b } => {
                    let (ra, rb) = (challenges[a.0], challenges[b.0]);
                    ra * rb + (F::one() - ra) * (F::one() - rb)
                }
                ClaimFactor::EqEval {
                    challenges: chs,
                    at_stage,
                } => {
                    let r: Vec<F> = chs.iter().map(|&ci| challenges[ci.0]).collect();
                    let s = resolve_point(sumcheck_points, point_override, at_stage.0);
                    EqPolynomial::<F>::mle(&r, s)
                }
                ClaimFactor::EqEvalSlice {
                    challenges: chs,
                    at_stage,
                    offset,
                } => {
                    let r: Vec<F> = chs.iter().map(|&ci| challenges[ci.0]).collect();
                    let s = resolve_point(sumcheck_points, point_override, at_stage.0);
                    EqPolynomial::<F>::mle(&r, &s[*offset..*offset + r.len()])
                }
                ClaimFactor::LagrangeKernelDomain {
                    tau_challenge,
                    at_challenge,
                    domain_size,
                    domain_start,
                } => jolt_poly::lagrange::lagrange_kernel_eval(
                    *domain_start,
                    *domain_size,
                    challenges[tau_challenge.0],
                    challenges[at_challenge.0],
                ),
                ClaimFactor::LagrangeWeight {
                    challenge,
                    domain_size,
                    domain_start,
                    basis_index,
                } => jolt_poly::lagrange::lagrange_basis_eval(
                    *domain_start,
                    *domain_size,
                    *basis_index,
                    challenges[challenge.0],
                ),
                ClaimFactor::UniformR1CSEval {
                    matrix,
                    eval_polys,
                    at_challenge,
                    num_constraints,
                    domain_start,
                } => {
                    let r0 = challenges[at_challenge.0];
                    let basis =
                        jolt_poly::lagrange::lagrange_evals(*domain_start, *num_constraints, r0);
                    let mut z = Vec::with_capacity(1 + eval_polys.len());
                    z.push(F::one());
                    for p in eval_polys {
                        z.push(evaluations.get(p).copied().ok_or_else(|| {
                            JoltError::InvalidProof(format!("R1CS eval {p:?} not available"))
                        })?);
                    }
                    let rows = match matrix {
                        R1CSMatrix::A => &r1cs_key.matrices.a,
                        R1CSMatrix::B => &r1cs_key.matrices.b,
                    };
                    let mut acc = F::zero();
                    for (k, row) in rows[..*num_constraints].iter().enumerate() {
                        let mut dot = F::zero();
                        for &(j, coeff) in row {
                            dot += coeff * z[j];
                        }
                        acc += basis[k] * dot;
                    }
                    acc
                }
                ClaimFactor::GroupSplitR1CSEval {
                    matrix,
                    eval_polys,
                    at_r0,
                    at_r_group,
                    group0_indices,
                    group1_indices,
                    domain_size,
                    domain_start,
                } => {
                    let r0 = challenges[at_r0.0];
                    let r_group = challenges[at_r_group.0];
                    let basis =
                        jolt_poly::lagrange::lagrange_evals(*domain_start, *domain_size, r0);
                    let mut z = Vec::with_capacity(1 + eval_polys.len());
                    z.push(F::one());
                    for p in eval_polys {
                        z.push(evaluations.get(p).copied().ok_or_else(|| {
                            JoltError::InvalidProof(format!("R1CS eval {p:?} not available"))
                        })?);
                    }
                    let rows = match matrix {
                        R1CSMatrix::A => &r1cs_key.matrices.a,
                        R1CSMatrix::B => &r1cs_key.matrices.b,
                    };
                    let eval_group = |indices: &[usize]| -> F {
                        let mut acc = F::zero();
                        for (i, &idx) in indices.iter().enumerate() {
                            let mut dot = F::zero();
                            for &(j, coeff) in &rows[idx] {
                                dot += coeff * z[j];
                            }
                            acc += basis[i] * dot;
                        }
                        acc
                    };
                    let g0 = eval_group(group0_indices);
                    let g1 = eval_group(group1_indices);
                    g0 + r_group * (g1 - g0)
                }
                ClaimFactor::StageEval(index) => {
                    let evals = stage_evals.ok_or_else(|| {
                        JoltError::InvalidProof("StageEval used but no stage_evals provided".into())
                    })?;
                    *evals.get(*index).ok_or_else(|| {
                        JoltError::InvalidProof(format!(
                            "StageEval({index}) out of bounds (len={})",
                            evals.len()
                        ))
                    })?
                }
                ClaimFactor::PreprocessedPolyEval { poly, at_stage } => {
                    let point = resolve_point(sumcheck_points, point_override, at_stage.0);
                    evaluate_preprocessed_poly(*poly, point, config)?
                }
                ClaimFactor::StagedEval { .. } => {
                    return Err(JoltError::InvalidProof(
                        "StagedEval is prover-only; not supported in verifier formulas".into(),
                    ));
                }
                ClaimFactor::LagrangeKernel { .. } => {
                    return Err(JoltError::InvalidProof(format!(
                        "unsupported claim factor {factor:?} in compiled schedule"
                    )));
                }
            };
            product *= factor_val;
        }
        sum += product;
    }
    Ok(sum)
}

/// Evaluate a preprocessed polynomial at a given point.
///
/// These polynomials are derivable entirely from public data carried in
/// [`ProverConfig`]: memory layout, serialized I/O, and panic flag.
fn evaluate_preprocessed_poly<F: Field>(
    poly: PolynomialId,
    point: &[F],
    config: &ProverConfig,
) -> Result<F, JoltError> {
    match poly {
        PolynomialId::IoMask => {
            // 1 for addresses in the I/O range, 0 outside.
            // MLE = LT(r, io_end) - LT(r, io_start).
            // io_end = remap(memory_start) — includes advice regions and
            // power-of-2 padding (matches jolt-core's RangeMaskPolynomial
            // range [input_start, RAM_START_ADDRESS)).
            let io_start = config.input_word_offset as u128;
            let io_end = ((config.memory_start - config.ram_lowest_address) / 8) as u128;
            Ok(lt_mle(point, io_end) - lt_mle(point, io_start))
        }
        PolynomialId::RamUnmap => {
            // Maps remapped word index k back to physical byte address:
            //   unmap(k) = k * 8 + ram_lowest_address
            let identity = identity_mle(point);
            Ok(identity * F::from_u64(8) + F::from_u64(config.ram_lowest_address))
        }
        PolynomialId::ValIo => eval_io_mle(point, config),
        _ => Err(JoltError::InvalidProof(format!(
            "PreprocessedPolyEval({poly:?}) not a known preprocessed polynomial"
        ))),
    }
}

/// Evaluate LT(r, threshold): the MLE of the indicator `{x < threshold}`
/// over the Boolean hypercube.
///
/// Scans threshold bits MSB-first, accumulating the probability that a
/// random hypercube point is strictly less than the threshold.
fn lt_mle<F: Field>(r: &[F], threshold: u128) -> F {
    let n = r.len();
    debug_assert!(threshold < (1u128 << n), "threshold exceeds domain");
    let mut lt = F::zero();
    let mut eq = F::one();
    for (i, ri) in r.iter().enumerate() {
        let bit = (threshold >> (n - 1 - i)) & 1;
        if bit == 1 {
            lt += eq * (F::one() - *ri);
            eq *= *ri;
        } else {
            eq *= F::one() - *ri;
        }
    }
    lt
}

/// Evaluate the identity polynomial: the MLE of f(x) = x interpreted as
/// a binary integer. Returns `Σ_i r[i] * 2^(n-1-i)`.
fn identity_mle<F: Field>(r: &[F]) -> F {
    let n = r.len();
    let mut sum = F::zero();
    for (i, ri) in r.iter().enumerate() {
        sum += *ri * F::from_u128(1u128 << (n - 1 - i));
    }
    sum
}

/// Evaluate the I/O values MLE at point r.
///
/// The I/O polynomial is sparse: nonzero only at input word addresses,
/// output word addresses, the panic flag, and the termination flag.
/// Evaluated without materializing the full K-element vector.
fn eval_io_mle<F: Field>(r: &[F], config: &ProverConfig) -> Result<F, JoltError> {
    // The I/O region occupies the low portion of the address space.
    // Length = remap(RAM_START_ADDRESS) rounded up to a power of 2 —
    // matches jolt-core's `io_len_words` (MemoryLayout pads io_region_bytes
    // to the next power of 2, so this may exceed `termination_word_offset + 1`).
    let io_end_words = ((config.memory_start - config.ram_lowest_address) / 8) as usize;
    let io_len = io_end_words.next_power_of_two().max(1);
    let num_io_vars = io_len.trailing_zeros() as usize;

    if num_io_vars > r.len() {
        return Err(JoltError::InvalidProof(format!(
            "ValIo: num_io_vars ({num_io_vars}) > point len ({})",
            r.len()
        )));
    }

    let (r_hi, r_lo) = r.split_at(r.len() - num_io_vars);

    // High-order bits must be 0 for I/O addresses — scale by Π(1 - r_i).
    let mut hi_scale = F::one();
    for ri in r_hi {
        hi_scale *= F::one() - *ri;
    }

    let mut acc = F::zero();

    // Inputs region
    if !config.inputs.is_empty() {
        let words = bytes_to_words(&config.inputs);
        acc += sparse_block_eval(config.input_word_offset, &words, r_lo);
    }

    // Outputs region
    if !config.outputs.is_empty() {
        let words = bytes_to_words(&config.outputs);
        acc += sparse_block_eval(config.output_word_offset, &words, r_lo);
    }

    // Panic flag (one word)
    acc += sparse_block_eval(config.panic_word_offset, &[config.panic as u64], r_lo);

    // Termination flag (set when not panicking)
    if !config.panic {
        acc += sparse_block_eval(config.termination_word_offset, &[1u64], r_lo);
    }

    Ok(hi_scale * acc)
}

/// Pack a byte slice into 8-byte little-endian words.
fn bytes_to_words(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks(8)
        .map(|chunk| {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(word)
        })
        .collect()
}

/// Evaluate `Σ_j values[j] * eq(start + j, r)` where eq is the
/// multilinear equality polynomial at a Boolean hypercube index.
fn sparse_block_eval<F: Field>(start: usize, values: &[u64], r: &[F]) -> F {
    let mut acc = F::zero();
    for (j, &val) in values.iter().enumerate() {
        if val == 0 {
            continue;
        }
        acc += eq_at_index(start + j, r) * F::from_u64(val);
    }
    acc
}

/// Evaluate `eq(idx, r)` treating idx as a binary vector (MSB-first).
fn eq_at_index<F: Field>(idx: usize, r: &[F]) -> F {
    let n = r.len();
    let mut prod = F::one();
    for (i, ri) in r.iter().enumerate() {
        let bit = (idx >> (n - 1 - i)) & 1;
        if bit == 1 {
            prod *= *ri;
        } else {
            prod *= F::one() - *ri;
        }
    }
    prod
}

/// Resolve the sumcheck point for a stage, using `point_override` when
/// the stage matches.
#[inline]
fn resolve_point<'a, F>(
    sumcheck_points: &'a [Vec<F>],
    point_override: Option<(usize, &'a [F])>,
    stage: usize,
) -> &'a [F] {
    match point_override {
        Some((os, op)) if stage == os => op,
        _ => &sumcheck_points[stage],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{OneHotConfig, ReadWriteConfig};
    use jolt_compiler::module::{ClaimTerm, Evaluation, SumcheckInstance, VerifierStageIndex};
    use jolt_compiler::{ChallengeIdx, PolynomialId, VerifierSchedule};
    use jolt_field::Fr;
    use jolt_r1cs::ConstraintMatrices;
    use jolt_sumcheck::{proof::SumcheckProof, ClearRoundVerifier};
    use jolt_transcript::Blake2bTranscript;
    use num_traits::One;

    use crate::proof::StageProof;

    fn dummy_r1cs_key() -> R1csKey<Fr> {
        let m = ConstraintMatrices::new(1, 1, vec![vec![]], vec![vec![]], vec![vec![]]);
        R1csKey::new(m, 1)
    }

    fn dummy_config() -> ProverConfig {
        ProverConfig {
            trace_length: 1,
            ram_k: 1,
            bytecode_k: 1,
            one_hot_config: OneHotConfig::new(10),
            rw_config: ReadWriteConfig::new(10, 10),
            memory_start: 0,
            memory_end: 0,
            entry_address: 0,
            io_hash: [0u8; 32],
            max_input_size: 0,
            max_output_size: 0,
            heap_size: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            panic: false,
            ram_lowest_address: 0,
            input_word_offset: 0,
            output_word_offset: 0,
            panic_word_offset: 0,
            termination_word_offset: 0,
        }
    }

    /// Eval-only stage: BeginStage → RecordEvals → AbsorbEvals → Squeeze.
    #[test]
    fn verify_eval_only_schedule() {
        let poly_a = PolynomialId::RdInc;
        let schedule = VerifierSchedule {
            ops: vec![
                VerifierOp::BeginStage,
                VerifierOp::RecordEvals {
                    evals: vec![Evaluation {
                        poly: poly_a,
                        at_stage: VerifierStageIndex(0),
                    }],
                },
                VerifierOp::AbsorbEvals {
                    polys: vec![poly_a],
                    tag: jolt_compiler::DomainSeparator::OpeningClaim,
                },
                VerifierOp::Squeeze {
                    challenge: ChallengeIdx(0),
                },
            ],
            num_challenges: 1,
            num_polys: 1,
            num_stages: 1,
        };

        let stage_proofs = [StageProof {
            round_polys: SumcheckProof::default(),
            evals: vec![Fr::one()],
        }];

        let mut challenges = [Fr::zero(); 1];
        let mut evaluations: HashMap<PolynomialId, Fr> = HashMap::new();
        let mut transcript = Blake2bTranscript::new(b"test");
        let mut stage_iter = stage_proofs.iter();
        let mut current_stage: Option<&StageProof<Fr>> = None;

        for op in &schedule.ops {
            match op {
                VerifierOp::BeginStage => {
                    current_stage = stage_iter.next();
                }
                VerifierOp::RecordEvals { evals } => {
                    let sp = current_stage.unwrap();
                    for (ei, eval_desc) in evals.iter().enumerate() {
                        let _ = evaluations.insert(eval_desc.poly, sp.evals[ei]);
                    }
                }
                VerifierOp::AbsorbEvals { polys, tag } => {
                    for &pi in polys {
                        if let Some(&val) = evaluations.get(&pi) {
                            transcript.append(&Label(tag.as_bytes()));
                            val.append_to_transcript(&mut transcript);
                        }
                    }
                }
                VerifierOp::Squeeze { challenge } => {
                    challenges[challenge.0] = transcript.challenge();
                }
                _ => {}
            }
        }

        assert_eq!(evaluations.get(&poly_a), Some(&Fr::one()));
        assert_ne!(challenges[0], Fr::zero());
    }

    #[test]
    fn formula_evaluation() {
        let poly_a = PolynomialId::RdInc;
        let poly_b = PolynomialId::RamInc;
        let formula = ClaimFormula {
            terms: vec![
                ClaimTerm {
                    coeff: 2,
                    factors: vec![
                        ClaimFactor::Eval(poly_a),
                        ClaimFactor::Challenge(ChallengeIdx(0)),
                    ],
                },
                ClaimTerm {
                    coeff: 3,
                    factors: vec![ClaimFactor::Eval(poly_b)],
                },
            ],
        };

        let mut evaluations = HashMap::new();
        let _ = evaluations.insert(poly_a, Fr::from_u64(5));
        let _ = evaluations.insert(poly_b, Fr::from_u64(7));
        let challenges = vec![Fr::from_u64(3)];
        let sumcheck_points: Vec<Vec<Fr>> = vec![];

        let result = evaluate_formula(
            &formula,
            &evaluations,
            &challenges,
            &sumcheck_points,
            None,
            None,
            &dummy_r1cs_key(),
            &dummy_config(),
        )
        .unwrap();
        // 2 * 5 * 3 + 3 * 7 = 30 + 21 = 51
        assert_eq!(result, Fr::from_u64(51));
    }

    #[test]
    fn eq_eval_correctness() {
        let one = Fr::one();
        let zero = Fr::zero();

        assert_eq!(EqPolynomial::<Fr>::mle(&[one], &[one]), one);
        assert_eq!(EqPolynomial::<Fr>::mle(&[zero], &[zero]), one);
        assert_eq!(EqPolynomial::<Fr>::mle(&[one], &[zero]), zero);
        assert_eq!(EqPolynomial::<Fr>::mle(&[zero], &[one]), zero);
        assert_eq!(EqPolynomial::<Fr>::mle(&[one, zero], &[one, zero]), one);
        assert_eq!(EqPolynomial::<Fr>::mle(&[one, zero], &[zero, one]), zero);
    }

    /// Generate a sumcheck proof then verify it through the flat schedule.
    #[test]
    fn prove_then_verify_via_schedule() {
        use jolt_poly::UnivariatePoly;

        let three = Fr::from_u64(3);

        // Inline 2-round sumcheck proof for f(x₀, x₁) = 3·x₀·x₁.
        let mut evals = vec![Fr::zero(), Fr::zero(), Fr::zero(), three];
        let mut prover_transcript = Blake2bTranscript::new(b"schedule-test");
        let mut round_polys = Vec::new();

        // Round 0
        let half = evals.len() / 2;
        let s0: Fr = evals[..half].iter().copied().sum();
        let s1: Fr = evals[half..].iter().copied().sum();
        let poly0 = UnivariatePoly::new(vec![s0, s1 - s0]);
        for c in poly0.coefficients() {
            jolt_transcript::AppendToTranscript::append_to_transcript(c, &mut prover_transcript);
        }
        let r0: Fr = jolt_transcript::Transcript::challenge(&mut prover_transcript);
        let mut new_evals = Vec::with_capacity(half);
        for i in 0..half {
            new_evals.push(evals[i] * (Fr::one() - r0) + evals[i + half] * r0);
        }
        evals = new_evals;
        round_polys.push(poly0);

        // Round 1
        let s0: Fr = evals[..1].iter().copied().sum();
        let s1: Fr = evals[1..].iter().copied().sum();
        let poly1 = UnivariatePoly::new(vec![s0, s1 - s0]);
        for c in poly1.coefficients() {
            jolt_transcript::AppendToTranscript::append_to_transcript(c, &mut prover_transcript);
        }
        round_polys.push(poly1);

        let proof = SumcheckProof {
            round_polynomials: round_polys,
        };

        let stage_proofs = [StageProof {
            round_polys: proof,
            evals: vec![],
        }];

        let schedule = VerifierSchedule {
            ops: vec![
                VerifierOp::BeginStage,
                VerifierOp::VerifySumcheck {
                    instances: vec![SumcheckInstance {
                        input_claim: ClaimFormula {
                            terms: vec![ClaimTerm {
                                coeff: 3,
                                factors: vec![],
                            }],
                        },
                        output_check: ClaimFormula::zero(),
                        num_rounds: 2,
                        degree: 1,
                        normalize: None,
                    }],
                    stage: 0,
                    batch_challenges: Vec::new(),
                    claim_tag: None,
                    sumcheck_challenge_slots: Vec::new(),
                },
            ],
            num_challenges: 0,
            num_polys: 0,
            num_stages: 1,
        };

        let mut sumcheck_points: Vec<Vec<Fr>> = vec![Vec::new(); 1];
        let mut transcript = Blake2bTranscript::new(b"schedule-test");
        let evaluations: HashMap<PolynomialId, Fr> = HashMap::new();
        let challenges: Vec<Fr> = vec![];
        let mut stage_iter = stage_proofs.iter();
        let mut current_stage: Option<&StageProof<Fr>> = None;

        for op in &schedule.ops {
            match op {
                VerifierOp::BeginStage => {
                    current_stage = stage_iter.next();
                }
                VerifierOp::VerifySumcheck {
                    instances, stage, ..
                } => {
                    let sp = current_stage.unwrap();
                    let max_rounds = instances.iter().map(|i| i.num_rounds).max().unwrap_or(0);
                    let max_degree = instances.iter().map(|i| i.degree).max().unwrap_or(0);

                    let mut combined_claim = Fr::zero();
                    let key = dummy_r1cs_key();
                    for inst in instances {
                        let val = evaluate_formula(
                            &inst.input_claim,
                            &evaluations,
                            &challenges,
                            &sumcheck_points,
                            None,
                            None,
                            &key,
                            &dummy_config(),
                        )
                        .unwrap();
                        combined_claim +=
                            val * Fr::from_u64(1u64 << (max_rounds - inst.num_rounds));
                    }

                    let claim = SumcheckClaim {
                        num_vars: max_rounds,
                        degree: max_degree,
                        claimed_sum: combined_claim,
                    };
                    let clear = ClearRoundVerifier::new();
                    let (_fe, sc) = SumcheckVerifier::verify(
                        &claim,
                        &sp.round_polys.round_polynomials,
                        &mut transcript,
                        &clear,
                    )
                    .unwrap();
                    sumcheck_points[*stage] = sc;
                }
                _ => {}
            }
        }

        assert_eq!(sumcheck_points[0].len(), 2);
    }

    /// Test formula with EqEval factor.
    #[test]
    fn formula_with_eq_eval() {
        let poly_a = PolynomialId::RdInc;
        let formula = ClaimFormula {
            terms: vec![ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Eval(poly_a),
                    ClaimFactor::EqEval {
                        challenges: vec![ChallengeIdx(0), ChallengeIdx(1)],
                        at_stage: VerifierStageIndex(0),
                    },
                ],
            }],
        };

        let one = Fr::one();
        let zero = Fr::zero();
        let mut evaluations = HashMap::new();
        let _ = evaluations.insert(poly_a, Fr::from_u64(7));
        let challenges = vec![one, zero];
        let sumcheck_points = vec![vec![one, zero]];

        let cfg = dummy_config();
        let result = evaluate_formula(
            &formula,
            &evaluations,
            &challenges,
            &sumcheck_points,
            None,
            None,
            &dummy_r1cs_key(),
            &cfg,
        )
        .unwrap();
        assert_eq!(result, Fr::from_u64(7));

        let sumcheck_points2 = vec![vec![zero, one]];
        let result2 = evaluate_formula(
            &formula,
            &evaluations,
            &challenges,
            &sumcheck_points2,
            None,
            None,
            &dummy_r1cs_key(),
            &cfg,
        )
        .unwrap();
        assert_eq!(result2, Fr::zero());
    }

    /// Test that point_override works for CheckOutput pattern.
    #[test]
    fn formula_with_point_override() {
        let poly_a = PolynomialId::RdInc;
        let formula = ClaimFormula {
            terms: vec![ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Eval(poly_a),
                    ClaimFactor::EqEval {
                        challenges: vec![ChallengeIdx(0)],
                        at_stage: VerifierStageIndex(0),
                    },
                ],
            }],
        };

        let one = Fr::one();
        let zero = Fr::zero();
        let mut evaluations = HashMap::new();
        let _ = evaluations.insert(poly_a, Fr::from_u64(5));
        let challenges = vec![one];
        let sumcheck_points = vec![vec![zero]];
        let key = dummy_r1cs_key();
        let cfg = dummy_config();

        let result = evaluate_formula(
            &formula,
            &evaluations,
            &challenges,
            &sumcheck_points,
            None,
            None,
            &key,
            &cfg,
        )
        .unwrap();
        assert_eq!(result, Fr::zero());

        let override_point = vec![one];
        let result2 = evaluate_formula(
            &formula,
            &evaluations,
            &challenges,
            &sumcheck_points,
            Some((0, &override_point)),
            None,
            &key,
            &cfg,
        )
        .unwrap();
        assert_eq!(result2, Fr::from_u64(5));
    }

    #[test]
    fn lt_mle_basic() {
        let one = Fr::one();
        let zero = Fr::zero();

        // LT(0, 0) = 0 — nothing is less than 0
        assert_eq!(lt_mle::<Fr>(&[zero], 0), Fr::zero());
        // LT(0, 1) = 1 — index 0 < 1
        assert_eq!(lt_mle::<Fr>(&[zero], 1), Fr::one());
        // LT(1, 1) = 0 — index 1 is not < 1
        assert_eq!(lt_mle::<Fr>(&[one], 1), Fr::zero());

        // 2-bit: LT(_, 2) should be 1 for indices 0,1 and 0 for 2,3
        assert_eq!(lt_mle::<Fr>(&[zero, zero], 2), Fr::one()); // idx 0 < 2
        assert_eq!(lt_mle::<Fr>(&[zero, one], 2), Fr::one()); // idx 1 < 2
        assert_eq!(lt_mle::<Fr>(&[one, zero], 2), Fr::zero()); // idx 2 !< 2
        assert_eq!(lt_mle::<Fr>(&[one, one], 2), Fr::zero()); // idx 3 !< 2
    }

    #[test]
    fn identity_mle_basic() {
        let one = Fr::one();
        let zero = Fr::zero();

        assert_eq!(identity_mle::<Fr>(&[zero, zero]), Fr::zero());
        assert_eq!(identity_mle::<Fr>(&[zero, one]), Fr::one());
        assert_eq!(identity_mle::<Fr>(&[one, zero]), Fr::from_u64(2));
        assert_eq!(identity_mle::<Fr>(&[one, one]), Fr::from_u64(3));
    }

    #[test]
    fn io_mask_range_check() {
        let one = Fr::one();
        let zero = Fr::zero();

        // IoMask evaluates LT(r, io_end) - LT(r, io_start).
        // Range [1, 3) in a 4-element domain (2 bits):
        let mask = |r: &[Fr]| -> Fr { lt_mle(r, 3) - lt_mle(r, 1) };
        assert_eq!(mask(&[zero, zero]), Fr::zero()); // idx 0 not in [1,3)
        assert_eq!(mask(&[zero, one]), Fr::one()); // idx 1 in [1,3)
        assert_eq!(mask(&[one, zero]), Fr::one()); // idx 2 in [1,3)
        assert_eq!(mask(&[one, one]), Fr::zero()); // idx 3 not in [1,3)
    }

    // -----------------------------------------------------------------------
    // FieldBackend parity tests: every claim formula and preprocessed-poly
    // pattern produced by the prover must yield the exact same field value
    // through the new `evaluate_formula_with_backend` /
    // `evaluate_preprocessed_poly_with_backend` helpers when run with the
    // `Native<Fr>` backend, AND the `Tracing<Fr>` AST must replay to the
    // same value when fed the recorded wrap values.
    // -----------------------------------------------------------------------

    use jolt_openings::mock::MockCommitmentScheme;
    use jolt_verifier_backend::{replay_trace, Native, Tracing};

    /// Field-only parity tests pick a concrete (trivial) PCS so the
    /// `Tracing<PCS>` shape is type-honest. `MockCommitmentScheme<Fr>`
    /// has `Field = Fr` and `VerifierSetup = ()`, so field-side methods
    /// take `Fr` directly and `replay_trace` takes `&()`.
    type TraceMock = MockCommitmentScheme<Fr>;

    /// Adapter: lift a `(challenges, evaluations, sumcheck_points, override)`
    /// configuration through a `FieldBackend` and call
    /// `evaluate_formula_with_backend`. Returns the raw `B::Scalar` and the
    /// concrete `Fr` underneath (via the backend's wrap-value table for
    /// Tracing, or the value itself for Native).
    #[allow(clippy::too_many_arguments)]
    fn run_formula_backend<B>(
        backend: &mut B,
        formula: &ClaimFormula,
        evaluations: &HashMap<PolynomialId, Fr>,
        challenges: &[Fr],
        sumcheck_points: &[Vec<Fr>],
        point_override: Option<(usize, &[Fr])>,
        stage_evals: Option<&[Fr]>,
        r1cs_key: &R1csKey<Fr>,
        config: &ProverConfig,
    ) -> (B::Scalar, Vec<Vec<B::Scalar>>)
    where
        B: FieldBackend<F = Fr>,
    {
        let evals_w: HashMap<PolynomialId, B::Scalar> = evaluations
            .iter()
            .map(|(p, v)| (*p, backend.wrap_proof(*v, "eval")))
            .collect();
        let chs_w: Vec<B::Scalar> = challenges
            .iter()
            .map(|v| backend.wrap_challenge(*v, "ch"))
            .collect();
        let pts_w: Vec<Vec<B::Scalar>> = sumcheck_points
            .iter()
            .map(|p| {
                p.iter()
                    .map(|v| backend.wrap_challenge(*v, "sc_pt"))
                    .collect()
            })
            .collect();
        let override_owned: Option<Vec<B::Scalar>> = point_override.map(|(_, pts)| {
            pts.iter()
                .map(|v| backend.wrap_challenge(*v, "ov"))
                .collect()
        });
        let override_ref = point_override.map(|(s, _)| (s, override_owned.as_deref().unwrap()));
        let stage_w: Option<Vec<B::Scalar>> = stage_evals.map(|s| {
            s.iter()
                .map(|v| backend.wrap_proof(*v, "stage_eval"))
                .collect()
        });
        let stage_ref = stage_w.as_deref();

        let result = evaluate_formula_with_backend(
            backend,
            formula,
            &evals_w,
            &chs_w,
            &pts_w,
            override_ref,
            stage_ref,
            r1cs_key,
            config,
        )
        .unwrap();

        (result, pts_w)
    }

    #[allow(clippy::too_many_arguments)]
    fn check_formula_parity(
        formula: &ClaimFormula,
        evaluations: &HashMap<PolynomialId, Fr>,
        challenges: &[Fr],
        sumcheck_points: &[Vec<Fr>],
        point_override: Option<(usize, &[Fr])>,
        stage_evals: Option<&[Fr]>,
        r1cs_key: &R1csKey<Fr>,
        config: &ProverConfig,
    ) {
        let native_legacy = evaluate_formula(
            formula,
            evaluations,
            challenges,
            sumcheck_points,
            point_override,
            stage_evals,
            r1cs_key,
            config,
        )
        .unwrap();

        let mut nb = Native::<Fr>::new();
        let (native_backend, _) = run_formula_backend(
            &mut nb,
            formula,
            evaluations,
            challenges,
            sumcheck_points,
            point_override,
            stage_evals,
            r1cs_key,
            config,
        );
        assert_eq!(
            native_backend, native_legacy,
            "Native FieldBackend must match legacy evaluate_formula"
        );

        let mut tracer = Tracing::<TraceMock>::new();
        let (traced, _) = run_formula_backend(
            &mut tracer,
            formula,
            evaluations,
            challenges,
            sumcheck_points,
            point_override,
            stage_evals,
            r1cs_key,
            config,
        );
        let graph = tracer.snapshot();
        let wraps = tracer.wrap_values();
        let values = replay_trace::<TraceMock>(&graph, &wraps, &()).unwrap();
        let traced_replayed = values[traced.id.0 as usize];
        assert_eq!(
            traced_replayed, native_legacy,
            "Tracing replay must match legacy evaluate_formula"
        );
    }

    #[test]
    fn formula_backend_parity_eval_and_challenge() {
        let poly_a = PolynomialId::RdInc;
        let poly_b = PolynomialId::RamInc;
        let formula = ClaimFormula {
            terms: vec![
                ClaimTerm {
                    coeff: 2,
                    factors: vec![
                        ClaimFactor::Eval(poly_a),
                        ClaimFactor::Challenge(ChallengeIdx(0)),
                    ],
                },
                ClaimTerm {
                    coeff: -3,
                    factors: vec![ClaimFactor::Eval(poly_b)],
                },
            ],
        };

        let mut evaluations = HashMap::new();
        let _ = evaluations.insert(poly_a, Fr::from_u64(5));
        let _ = evaluations.insert(poly_b, Fr::from_u64(7));
        let challenges = vec![Fr::from_u64(11)];
        let sumcheck_points: Vec<Vec<Fr>> = vec![];

        check_formula_parity(
            &formula,
            &evaluations,
            &challenges,
            &sumcheck_points,
            None,
            None,
            &dummy_r1cs_key(),
            &dummy_config(),
        );
    }

    #[test]
    fn formula_backend_parity_eq_eval_and_pair() {
        let poly_a = PolynomialId::RdInc;
        let formula = ClaimFormula {
            terms: vec![ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Eval(poly_a),
                    ClaimFactor::EqEval {
                        challenges: vec![ChallengeIdx(0), ChallengeIdx(1)],
                        at_stage: VerifierStageIndex(0),
                    },
                    ClaimFactor::EqChallengePair {
                        a: ChallengeIdx(0),
                        b: ChallengeIdx(1),
                    },
                ],
            }],
        };
        let mut evaluations = HashMap::new();
        let _ = evaluations.insert(poly_a, Fr::from_u64(13));
        let challenges = vec![Fr::from_u64(7), Fr::from_u64(11)];
        let sumcheck_points = vec![vec![Fr::from_u64(3), Fr::from_u64(5)]];

        check_formula_parity(
            &formula,
            &evaluations,
            &challenges,
            &sumcheck_points,
            None,
            None,
            &dummy_r1cs_key(),
            &dummy_config(),
        );
    }

    #[test]
    fn formula_backend_parity_lagrange_kernel_and_weight() {
        let formula = ClaimFormula {
            terms: vec![ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::LagrangeKernelDomain {
                        tau_challenge: ChallengeIdx(0),
                        at_challenge: ChallengeIdx(1),
                        domain_size: 4,
                        domain_start: 0,
                    },
                    ClaimFactor::LagrangeWeight {
                        challenge: ChallengeIdx(1),
                        domain_size: 4,
                        domain_start: 0,
                        basis_index: 2,
                    },
                ],
            }],
        };
        let challenges = vec![Fr::from_u64(17), Fr::from_u64(23)];
        let sumcheck_points: Vec<Vec<Fr>> = vec![];

        check_formula_parity(
            &formula,
            &HashMap::new(),
            &challenges,
            &sumcheck_points,
            None,
            None,
            &dummy_r1cs_key(),
            &dummy_config(),
        );
    }

    #[test]
    fn formula_backend_parity_uniform_r1cs() {
        // Build a 2x3 A matrix (1 free var z[1] + the constant 1; pad to fit
        // the 3-variable shape via z[2] as a second eval poly):
        //   row0: 2*z[0] + 3*z[1]
        //   row1: 5*z[1] - 7*z[2]
        let two = Fr::from_u64(2);
        let three = Fr::from_u64(3);
        let five = Fr::from_u64(5);
        let neg_seven = -Fr::from_u64(7);
        let row0: Vec<(usize, Fr)> = vec![(0, two), (1, three)];
        let row1: Vec<(usize, Fr)> = vec![(1, five), (2, neg_seven)];
        let m = ConstraintMatrices::new(
            2,
            3,
            vec![row0, row1],
            vec![vec![], vec![]],
            vec![vec![], vec![]],
        );
        let key = R1csKey::new(m, 1);

        let poly_a = PolynomialId::RdInc;
        let poly_b = PolynomialId::RamInc;
        let formula = ClaimFormula {
            terms: vec![ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::UniformR1CSEval {
                    matrix: R1CSMatrix::A,
                    eval_polys: vec![poly_a, poly_b],
                    at_challenge: ChallengeIdx(0),
                    num_constraints: 2,
                    domain_start: 0,
                }],
            }],
        };
        let mut evaluations = HashMap::new();
        let _ = evaluations.insert(poly_a, Fr::from_u64(11));
        let _ = evaluations.insert(poly_b, Fr::from_u64(13));
        let challenges = vec![Fr::from_u64(2)];
        let sumcheck_points: Vec<Vec<Fr>> = vec![];

        check_formula_parity(
            &formula,
            &evaluations,
            &challenges,
            &sumcheck_points,
            None,
            None,
            &key,
            &dummy_config(),
        );
    }

    #[test]
    fn formula_backend_parity_stage_eval() {
        let formula = ClaimFormula {
            terms: vec![ClaimTerm {
                coeff: 4,
                factors: vec![ClaimFactor::StageEval(0), ClaimFactor::StageEval(1)],
            }],
        };
        let stage_evals = vec![Fr::from_u64(6), Fr::from_u64(9)];

        check_formula_parity(
            &formula,
            &HashMap::new(),
            &[],
            &[],
            None,
            Some(&stage_evals),
            &dummy_r1cs_key(),
            &dummy_config(),
        );
    }

    fn parity_config_with_io() -> ProverConfig {
        let mut cfg = dummy_config();
        cfg.ram_lowest_address = 64;
        cfg.memory_start = 64 + 8 * 4; // 4-word io region
        cfg.input_word_offset = 0;
        cfg.output_word_offset = 1;
        cfg.panic_word_offset = 2;
        cfg.termination_word_offset = 3;
        cfg.inputs = b"abcdefghABCD".to_vec(); // 12 bytes -> 2 words
        cfg.outputs = b"xyz".to_vec(); // 3 bytes -> 1 word
        cfg.panic = false;
        cfg
    }

    #[test]
    fn formula_backend_parity_preprocessed_io_mask() {
        // `lt_mle` requires `threshold < 2^point.len()`. With io_end = 4
        // (memory_start - ram_lowest_address = 32 bytes = 4 words) we need
        // a point of length >= 3.
        let cfg = parity_config_with_io();
        let formula = ClaimFormula {
            terms: vec![ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::PreprocessedPolyEval {
                    poly: PolynomialId::IoMask,
                    at_stage: VerifierStageIndex(0),
                }],
            }],
        };
        let sumcheck_points = vec![vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)]];
        check_formula_parity(
            &formula,
            &HashMap::new(),
            &[],
            &sumcheck_points,
            None,
            None,
            &dummy_r1cs_key(),
            &cfg,
        );
    }

    #[test]
    fn formula_backend_parity_preprocessed_ram_unmap() {
        let mut cfg = dummy_config();
        cfg.ram_lowest_address = 128;
        let formula = ClaimFormula {
            terms: vec![ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::PreprocessedPolyEval {
                    poly: PolynomialId::RamUnmap,
                    at_stage: VerifierStageIndex(0),
                }],
            }],
        };
        let sumcheck_points = vec![vec![Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(13)]];
        check_formula_parity(
            &formula,
            &HashMap::new(),
            &[],
            &sumcheck_points,
            None,
            None,
            &dummy_r1cs_key(),
            &cfg,
        );
    }

    #[test]
    fn formula_backend_parity_preprocessed_val_io() {
        let cfg = parity_config_with_io();
        let formula = ClaimFormula {
            terms: vec![ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::PreprocessedPolyEval {
                    poly: PolynomialId::ValIo,
                    at_stage: VerifierStageIndex(0),
                }],
            }],
        };
        // io_len_words = next_pow2(4) = 4, num_io_vars = 2, plus a
        // single hi-bit to exercise the hi_scale path.
        let sumcheck_points = vec![vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)]];
        check_formula_parity(
            &formula,
            &HashMap::new(),
            &[],
            &sumcheck_points,
            None,
            None,
            &dummy_r1cs_key(),
            &cfg,
        );
    }

    #[test]
    fn formula_backend_parity_point_override() {
        // Same shape as the existing `formula_with_point_override` test, but
        // exercised through the backend.
        let poly_a = PolynomialId::RdInc;
        let formula = ClaimFormula {
            terms: vec![ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Eval(poly_a),
                    ClaimFactor::EqEval {
                        challenges: vec![ChallengeIdx(0)],
                        at_stage: VerifierStageIndex(0),
                    },
                ],
            }],
        };
        let one = Fr::one();
        let zero = Fr::zero();
        let mut evaluations = HashMap::new();
        let _ = evaluations.insert(poly_a, Fr::from_u64(5));
        let challenges = vec![one];
        let sumcheck_points = vec![vec![zero]];
        let key = dummy_r1cs_key();
        let cfg = dummy_config();

        check_formula_parity(
            &formula,
            &evaluations,
            &challenges,
            &sumcheck_points,
            None,
            None,
            &key,
            &cfg,
        );
        let override_point = vec![one];
        check_formula_parity(
            &formula,
            &evaluations,
            &challenges,
            &sumcheck_points,
            Some((0, &override_point)),
            None,
            &key,
            &cfg,
        );
    }
}
