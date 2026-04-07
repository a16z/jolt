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
use jolt_crypto::HomomorphicCommitment;
use jolt_openings::{
    AdditivelyHomomorphic, OpeningReduction, OpeningsError, RlcReduction, VerifierClaim,
};
use jolt_poly::EqPolynomial;
use jolt_r1cs::R1csKey;
use jolt_sumcheck::{ClearRoundVerifier, SumcheckClaim, SumcheckVerifier};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Label, LabelWithCount, Transcript};

use crate::config::ProverConfig;
use crate::error::JoltError;
use crate::key::JoltVerifyingKey;
use crate::proof::{JoltProof, StageProof};
use crate::TRANSCRIPT_LABEL;

/// End-to-end proof verification.
///
/// Walks the flat [`VerifierOp`] sequence from the verifying key, replaying
/// the Fiat-Shamir transcript in lockstep with the prover. Returns `Ok(())`
/// if the proof is valid.
pub fn verify<F, PCS>(
    key: &JoltVerifyingKey<F, PCS>,
    proof: &JoltProof<F, PCS>,
    expected_io_hash: &[u8; 32],
) -> Result<(), JoltError>
where
    F: Field,
    PCS: AdditivelyHomomorphic<Field = F>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
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
    let mut transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_LABEL);

    let mut challenges = vec![F::zero(); schedule.num_challenges];
    let mut evaluations: HashMap<PolynomialId, F> = HashMap::new();
    let mut sumcheck_points: Vec<Vec<F>> = vec![Vec::new(); schedule.num_stages];
    let mut final_evals = vec![F::zero(); schedule.num_stages];
    let mut commitment_map: HashMap<PolynomialId, PCS::Output> = HashMap::new();
    let mut commitments = proof.commitments.iter();
    let mut stage_proofs = proof.stage_proofs.iter();
    let mut current_stage: Option<&StageProof<F>> = None;
    let mut eval_cursor: usize = 0;
    let mut round_poly_cursor: usize = 0;
    let mut pcs_claims: Vec<VerifierClaim<F, PCS::Output>> = Vec::new();

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
                let c = commitments
                    .next()
                    .ok_or_else(|| JoltError::InvalidProof("missing commitment".into()))?;
                transcript.append(&Label(tag.as_bytes()));
                c.append_to_transcript(&mut transcript);
                let _ = commitment_map.insert(*poly, c.clone());
            }

            VerifierOp::Squeeze { challenge } => {
                challenges[*challenge] = transcript.challenge();
            }

            VerifierOp::AbsorbRoundPoly { num_coeffs: _, tag } => {
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
            } => {
                let sp = current_stage.ok_or_else(|| {
                    JoltError::InvalidProof("no active stage proof for sumcheck".into())
                })?;

                let max_rounds = instances.iter().map(|i| i.num_rounds).max().unwrap_or(0);
                let max_degree = instances.iter().map(|i| i.degree).max().unwrap_or(0);

                // Evaluate each instance's input claim.
                let mut instance_claims: Vec<F> = Vec::with_capacity(instances.len());
                for inst in instances {
                    let val = evaluate_formula(
                        &inst.input_claim,
                        &evaluations,
                        &challenges,
                        &sumcheck_points,
                        None,
                        None,
                        &key.r1cs_key,
                        &proof.config,
                    )?;
                    instance_claims.push(val);
                }

                // Batched: absorb claims, squeeze independent coefficients.
                let combined_claim = if batch_challenges.is_empty() {
                    // Unbatched: scale by 2^offset only.
                    instance_claims
                        .iter()
                        .zip(instances.iter())
                        .map(|(&c, inst)| c * F::from_u64(1u64 << (max_rounds - inst.num_rounds)))
                        .sum()
                } else {
                    let tag = claim_tag.as_ref().expect("claim_tag required for batched");
                    for &claim_val in &instance_claims {
                        transcript.append(&Label(tag.as_bytes()));
                        claim_val.append_to_transcript(&mut transcript);
                    }
                    for &ch_idx in batch_challenges {
                        challenges[ch_idx] = transcript.challenge();
                    }
                    instance_claims
                        .iter()
                        .zip(instances.iter())
                        .zip(batch_challenges.iter())
                        .map(|((&c, inst), &ch_idx)| {
                            let coeff = challenges[ch_idx];
                            coeff * c * F::from_u64(1u64 << (max_rounds - inst.num_rounds))
                        })
                        .sum()
                };
                let claim = SumcheckClaim {
                    num_vars: max_rounds,
                    degree: max_degree,
                    claimed_sum: combined_claim,
                };
                let round_polys = &sp.round_polys.round_polynomials
                    [round_poly_cursor..round_poly_cursor + max_rounds];
                let round_verifier = ClearRoundVerifier::with_label(b"sumcheck_poly");
                let (fe, sc) =
                    SumcheckVerifier::verify(&claim, round_polys, &mut transcript, &round_verifier)
                        .map_err(JoltError::Sumcheck)?;
                round_poly_cursor += max_rounds;

                final_evals[*stage] = fe;
                sumcheck_points[*stage] = sc;
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
                    let _ = evaluations.insert(eval_desc.poly, value);
                    eval_cursor += 1;
                }
            }

            VerifierOp::AbsorbEvals { polys, tag } => {
                for pi in polys {
                    if let Some(&val) = evaluations.get(pi) {
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
                let raw_point = &sumcheck_points[*stage];
                let mut combined_output = F::zero();

                for (i, inst) in instances.iter().enumerate() {
                    let offset = max_rounds - inst.num_rounds;
                    let normalized =
                        apply_normalization(&raw_point[offset..], inst.normalize.as_ref());
                    let output = evaluate_formula(
                        &inst.output_check,
                        &evaluations,
                        &challenges,
                        &sumcheck_points,
                        Some((*stage, &normalized)),
                        Some(&sp.evals),
                        &key.r1cs_key,
                        &proof.config,
                    )?;
                    if batch_challenges.is_empty() {
                        combined_output += output;
                    } else {
                        combined_output += challenges[batch_challenges[i]] * output;
                    }
                }

                if final_evals[*stage] != combined_output {
                    return Err(JoltError::EvaluationMismatch {
                        stage: *stage,
                        reason: format!(
                            "sumcheck final eval ({:?}) does not match batched composition ({:?})",
                            final_evals[*stage], combined_output
                        ),
                    });
                }
            }

            VerifierOp::CollectOpeningClaim { poly, at_stage } => {
                if let Some(commitment) = commitment_map.get(poly) {
                    let eval = evaluations.get(poly).copied().ok_or_else(|| {
                        JoltError::InvalidProof(format!(
                            "evaluation for committed poly {poly:?} not set"
                        ))
                    })?;
                    pcs_claims.push(VerifierClaim {
                        commitment: commitment.clone(),
                        point: sumcheck_points[at_stage.0].clone(),
                        eval,
                    });
                }
            }

            VerifierOp::VerifyOpenings => {
                if pcs_claims.is_empty() {
                    continue;
                }
                let claims = std::mem::take(&mut pcs_claims);
                let reduced = <RlcReduction as OpeningReduction<PCS>>::reduce_verifier(
                    claims,
                    &(),
                    &mut transcript,
                )
                .map_err(JoltError::Opening)?;

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
            product *= match factor {
                ClaimFactor::Eval(poly) => evaluations.get(poly).copied().ok_or_else(|| {
                    JoltError::InvalidProof(format!(
                        "evaluation {poly:?} referenced before available"
                    ))
                })?,
                ClaimFactor::Challenge(i) => challenges[*i],
                ClaimFactor::EqChallengePair { a, b } => {
                    let (ra, rb) = (challenges[*a], challenges[*b]);
                    ra * rb + (F::one() - ra) * (F::one() - rb)
                }
                ClaimFactor::EqEval {
                    challenges: chs,
                    at_stage,
                } => {
                    let r: Vec<F> = chs.iter().map(|&ci| challenges[ci]).collect();
                    let s = resolve_point(sumcheck_points, point_override, at_stage.0);
                    EqPolynomial::<F>::mle(&r, s)
                }
                ClaimFactor::EqEvalSlice {
                    challenges: chs,
                    at_stage,
                    offset,
                } => {
                    let r: Vec<F> = chs.iter().map(|&ci| challenges[ci]).collect();
                    let s = resolve_point(sumcheck_points, point_override, at_stage.0);
                    EqPolynomial::<F>::mle(&r, &s[*offset..*offset + r.len()])
                }
                ClaimFactor::LagrangeKernelDomain {
                    tau_challenge,
                    at_challenge,
                    domain_size,
                } => jolt_poly::lagrange::lagrange_kernel_eval(
                    *domain_size,
                    challenges[*tau_challenge],
                    challenges[*at_challenge],
                ),
                ClaimFactor::LagrangeWeight {
                    challenge,
                    domain_size,
                    basis_index,
                } => jolt_poly::lagrange::lagrange_basis_eval(
                    *domain_size,
                    *basis_index,
                    challenges[*challenge],
                ),
                ClaimFactor::UniformR1CSEval {
                    matrix,
                    eval_polys,
                    at_challenge,
                    num_constraints,
                } => {
                    let r0 = challenges[*at_challenge];
                    let basis = jolt_poly::lagrange::lagrange_evals(0, *num_constraints, r0);
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
                ClaimFactor::LagrangeKernel { .. } => {
                    return Err(JoltError::InvalidProof(format!(
                        "unsupported claim factor {factor:?} in compiled schedule"
                    )));
                }
            };
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
            let io_start = config.input_word_offset as u128;
            let io_end = (config.termination_word_offset + 1) as u128;
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
    let io_end = config.termination_word_offset + 1;
    let io_len = io_end.next_power_of_two().max(1);
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
    use jolt_compiler::{PolynomialId, VerifierSchedule};
    use jolt_field::Fr;
    use jolt_r1cs::ConstraintMatrices;
    use jolt_sumcheck::proof::SumcheckProof;
    use jolt_transcript::Blake2bTranscript;
    use num_traits::{One, Zero};

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
                VerifierOp::Squeeze { challenge: 0 },
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
                    challenges[*challenge] = transcript.challenge();
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
                    factors: vec![ClaimFactor::Eval(poly_a), ClaimFactor::Challenge(0)],
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
                        challenges: vec![0, 1],
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
                        challenges: vec![0],
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
}
