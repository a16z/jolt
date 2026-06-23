use super::*;

pub(in crate::protocol::core) struct PreparedExtensionOpeningReduction<E: FieldCore> {
    pub(in crate::protocol::core) openings: Vec<E>,
    pub(in crate::protocol::core) proof_partials: Vec<E>,
    pub(in crate::protocol::core) row_coefficients: Vec<E>,
    pub(in crate::protocol::core) terms: Vec<ExtensionOpeningReductionTerm<E>>,
    pub(in crate::protocol::core) padded_point: Vec<E>,
    pub(in crate::protocol::core) split_bits: usize,
    pub(in crate::protocol::core) eta: Vec<E>,
    #[cfg(feature = "zk")]
    pub(in crate::protocol::core) input_claim: E,
    pub(in crate::protocol::core) true_input_claim: E,
    #[cfg(feature = "zk")]
    pub(in crate::protocol::core) sumcheck_pads: Vec<CompressedUniPoly<E>>,
}

pub(in crate::protocol::core) struct ProvedExtensionOpeningReduction<E: FieldCore> {
    pub(in crate::protocol::core) reduction: ExtensionOpeningReduction<E>,
    pub(in crate::protocol::core) row_coefficients: Vec<E>,
    pub(in crate::protocol::core) openings: Vec<E>,
    pub(in crate::protocol::core) protocol_point: Vec<E>,
}

pub(in crate::protocol::core) fn build_extension_opening_reduction_terms<F, E, P, const D: usize>(
    polys: &[&P],
    num_vars: usize,
    padded_len: usize,
    row_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
) -> Result<Vec<ExtensionOpeningReductionTerm<E>>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
    P: AkitaPolyOps<F, D>,
{
    let _span =
        tracing::info_span!("extension_opening_reduction_terms", num_terms = polys.len()).entered();
    if polys.len() != row_coefficients.len() {
        return Err(AkitaError::InvalidSize {
            expected: polys.len(),
            actual: row_coefficients.len(),
        });
    }

    if let Some(terms) = try_sparse_extension_opening_reduction_terms::<F, E, P, D>(
        polys,
        row_coefficients,
        tail_point,
        eta,
    )? {
        return Ok(terms);
    }

    build_dense_extension_opening_reduction_terms::<F, E, P, D>(
        polys,
        num_vars,
        padded_len,
        row_coefficients,
        tail_point,
        eta,
    )
}

fn try_sparse_extension_opening_reduction_terms<F, E, P, const D: usize>(
    polys: &[&P],
    row_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
) -> Result<Option<Vec<ExtensionOpeningReductionTerm<E>>>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
    P: AkitaPolyOps<F, D>,
{
    let _span =
        tracing::info_span!("extension_opening_sparse_terms", num_terms = polys.len()).entered();
    let Some(witness_evals) =
        P::tensor_packed_extension_sparse_linear_combination::<E>(polys, row_coefficients)?
    else {
        return Ok(None);
    };
    let lazy_rounds = tail_point.len().min(SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS);
    let term = if lazy_rounds == 0 {
        let factor_evals = {
            let _span = tracing::debug_span!(
                "extension_opening_factor_evals",
                tail_vars = tail_point.len()
            )
            .entered();
            tensor_equality_factor_evals::<F, E>(tail_point, eta)?
        };
        ExtensionOpeningReductionTerm::new_sparse(witness_evals, factor_evals, E::one())?
    } else {
        let _span = tracing::debug_span!(
            "extension_opening_lazy_tensor_factor",
            tail_vars = tail_point.len(),
            lazy_rounds
        )
        .entered();
        ExtensionOpeningReductionTerm::new_sparse_tensor_factor::<F>(
            witness_evals,
            tail_point.to_vec(),
            eta.to_vec(),
            E::one(),
            lazy_rounds,
        )?
    };
    Ok(Some(vec![term]))
}

fn build_dense_extension_opening_reduction_terms<F, E, P, const D: usize>(
    polys: &[&P],
    num_vars: usize,
    padded_len: usize,
    row_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
) -> Result<Vec<ExtensionOpeningReductionTerm<E>>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
    P: AkitaPolyOps<F, D>,
{
    let _span =
        tracing::info_span!("extension_opening_dense_witnesses", num_terms = polys.len()).entered();
    polys
        .iter()
        .zip(row_coefficients.iter().copied())
        .map(|(poly, coeff)| {
            let base_evals = {
                let _s = tracing::info_span!("eor_base_evals").entered();
                let mut base_evals = poly.base_evals()?;
                if base_evals.len() > padded_len {
                    return Err(AkitaError::InvalidSize {
                        expected: padded_len,
                        actual: base_evals.len(),
                    });
                }
                base_evals.resize(padded_len, F::zero());
                base_evals
            };
            let witness_evals = {
                let _s = tracing::info_span!("eor_packed_witness").entered();
                tensor_packed_witness_evals::<F, E>(num_vars, &base_evals)?
            };
            let factor_evals = tensor_equality_factor_evals::<F, E>(tail_point, eta)?;
            ExtensionOpeningReductionTerm::new(witness_evals, factor_evals, coeff)
        })
        .collect()
}

pub(in crate::protocol::core) fn prepare_extension_opening_reduction<F, E, T, P, const D: usize>(
    polys: &[&P],
    opening_batch: &OpeningBatch,
    shared_opening_point: &[E],
    #[cfg(feature = "zk")] public_openings: Option<&[E]>,
    pad_base_evals: bool,
    transcript: &mut T,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<PreparedExtensionOpeningReduction<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + MulBaseUnreduced<F>,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
{
    let num_claims = opening_batch.num_claims();
    let num_vars = opening_batch.num_vars();
    let _span =
        tracing::info_span!("prepare_extension_opening_reduction", num_claims, num_vars).entered();
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    if split_bits > num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: split_bits,
            actual: num_vars,
        });
    }
    if polys.len() != num_claims {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction input lengths do not match".to_string(),
        ));
    }
    let padded_len = 1usize.checked_shl(num_vars as u32).ok_or_else(|| {
        AkitaError::InvalidInput("extension-opening reduction table length overflow".to_string())
    })?;

    let mut padded_point = shared_opening_point.to_vec();
    padded_point.resize(num_vars, E::zero());

    let mut openings = Vec::with_capacity(num_claims);
    let mut partials = Vec::with_capacity(width.saturating_mul(num_claims));
    let mut row_partials_by_claim = Vec::with_capacity(num_claims);
    {
        let _span =
            tracing::info_span!("extension_opening_prepare_partials", width, split_bits).entered();
        if pad_base_evals {
            for poly in polys {
                let mut base_evals = poly.base_evals()?;
                if base_evals.len() > padded_len {
                    return Err(AkitaError::InvalidSize {
                        expected: padded_len,
                        actual: base_evals.len(),
                    });
                }
                base_evals.resize(padded_len, F::zero());
                let (opening, tensor) = derive_tensor_extension_opening_claim::<F, E>(
                    num_vars,
                    &base_evals,
                    &padded_point,
                )?;
                partials.extend(tensor.column_partials);
                openings.push(opening);
                row_partials_by_claim.push(tensor.row_partials);
            }
        } else {
            let point_partials =
                P::tensor_extension_column_partials_batch::<E>(polys, &padded_point)?;
            if point_partials.len() != num_claims {
                return Err(AkitaError::InvalidSize {
                    expected: num_claims,
                    actual: point_partials.len(),
                });
            }
            for column_partials in point_partials {
                let opening = derive_tensor_extension_opening_claim_from_partials::<F, E>(
                    &padded_point,
                    &column_partials,
                )?;
                let row_partials = tensor_row_partials_from_columns::<F, E>(&column_partials)?;
                partials.extend(column_partials);
                openings.push(opening);
                row_partials_by_claim.push(row_partials);
            }
        }
    }
    #[cfg(feature = "zk")]
    let (partial_masks, sumcheck_pads) = zk_hiding
        .take_extension_opening_reduction_pads::<E>(partials.len(), num_vars - split_bits)?;
    #[cfg(feature = "zk")]
    let proof_partials = partials
        .iter()
        .copied()
        .zip(partial_masks)
        .map(|(partial, mask)| partial + mask)
        .collect::<Vec<_>>();
    #[cfg(not(feature = "zk"))]
    let proof_partials = partials.clone();
    let row_coefficients = if pad_base_evals {
        if num_claims != 1 {
            return Err(AkitaError::InvalidInput(
                "recursive extension-opening reduction expects a single claim".to_string(),
            ));
        }
        vec![E::one()]
    } else {
        #[cfg(feature = "zk")]
        let transcript_openings = if let Some(public_openings) = public_openings {
            if public_openings.len() != openings.len() {
                return Err(AkitaError::InvalidSize {
                    expected: openings.len(),
                    actual: public_openings.len(),
                });
            }
            public_openings
        } else {
            openings.as_slice()
        };
        #[cfg(not(feature = "zk"))]
        let transcript_openings = openings.as_slice();
        append_claim_values_to_transcript::<F, E, T>(transcript_openings, transcript);
        sample_public_row_coefficients::<F, E, T>(opening_batch, transcript)?
    };
    if row_partials_by_claim.len() != row_coefficients.len() {
        return Err(AkitaError::InvalidSize {
            expected: row_partials_by_claim.len(),
            actual: row_coefficients.len(),
        });
    }
    let expected_partials = width
        .checked_mul(row_coefficients.len())
        .ok_or_else(|| AkitaError::InvalidInput("EOR partial count overflow".to_string()))?;
    if proof_partials.len() != expected_partials {
        return Err(AkitaError::InvalidSize {
            expected: expected_partials,
            actual: proof_partials.len(),
        });
    }
    let proof_row_partials_by_claim = proof_partials
        .chunks_exact(width)
        .map(tensor_row_partials_from_columns::<F, E>)
        .collect::<Result<Vec<_>, _>>()?;
    {
        let _span = tracing::debug_span!(
            "extension_opening_absorb_partials",
            partials_len = proof_partials.len()
        )
        .entered();
        for partial in &proof_partials {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
        }
    }
    let eta = (0..split_bits)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
        .collect::<Vec<_>>();
    let input_claim = {
        let _span = tracing::debug_span!("extension_opening_input_claim").entered();
        proof_row_partials_by_claim
            .iter()
            .zip(row_coefficients.iter().copied())
            .try_fold(E::zero(), |acc, (row_partials, coeff)| {
                tensor_reduction_claim_from_rows::<F, E>(row_partials, &eta)
                    .map(|claim| acc + coeff * claim)
            })?
    };
    let true_input_claim = row_partials_by_claim
        .iter()
        .zip(row_coefficients.iter().copied())
        .try_fold(E::zero(), |acc, (row_partials, coeff)| {
            tensor_reduction_claim_from_rows::<F, E>(row_partials, &eta)
                .map(|claim| acc + coeff * claim)
        })?;
    #[cfg(not(feature = "zk"))]
    debug_assert_eq!(input_claim, true_input_claim);

    let tail_point = &padded_point[split_bits..];
    let terms = build_extension_opening_reduction_terms::<F, E, P, D>(
        polys,
        num_vars,
        padded_len,
        &row_coefficients,
        tail_point,
        &eta,
    )?;

    Ok(PreparedExtensionOpeningReduction {
        openings,
        proof_partials,
        row_coefficients,
        terms,
        padded_point,
        split_bits,
        eta,
        #[cfg(feature = "zk")]
        input_claim,
        true_input_claim,
        #[cfg(feature = "zk")]
        sumcheck_pads,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prove_extension_opening_reduction<F, E, T, P, const D: usize>(
    polys: &[&P],
    opening_batch: &OpeningBatch,
    shared_opening_point: &[E],
    #[cfg(feature = "zk")] public_openings: Option<&[E]>,
    pad_base_evals: bool,
    transcript: &mut T,
    path: &'static str,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<ProvedExtensionOpeningReduction<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + HasUnreducedOps + HasOptimizedFold + MulBaseUnreduced<F> + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
{
    let _span = tracing::info_span!(
        "prove_extension_opening_reduction",
        path,
        num_claims = opening_batch.num_claims()
    )
    .entered();
    let prepared = prepare_extension_opening_reduction::<F, E, T, P, D>(
        polys,
        opening_batch,
        shared_opening_point,
        #[cfg(feature = "zk")]
        public_openings,
        pad_base_evals,
        transcript,
        #[cfg(feature = "zk")]
        zk_hiding,
    )?;
    let tail_point = &prepared.padded_point[prepared.split_bits..];
    let prover_claim =
        ExtensionOpeningReductionProver::input_claim_from_terms(prepared.terms.as_slice())?;
    if prover_claim != prepared.true_input_claim {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction input claim mismatch".to_string(),
        ));
    }
    let mut prover = {
        let _span = tracing::info_span!("extension_opening_reduction_prover_new", path).entered();
        ExtensionOpeningReductionProver::new(prepared.terms, prover_claim)?
    };
    let _eor_sumcheck_span = tracing::info_span!(
        "extension_opening_reduction_sumcheck",
        path = path,
        num_rounds = prover.num_rounds()
    )
    .entered();
    #[cfg(not(feature = "zk"))]
    let (sumcheck_proof, rho, final_claim) = prover.prove::<F, T, _>(transcript, |tr| {
        sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
    })?;
    #[cfg(feature = "zk")]
    let (sumcheck_proof, rho) = prover.prove_zk::<F, T, _>(
        prepared.input_claim,
        transcript,
        |tr| sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND),
        prepared.sumcheck_pads,
    )?;
    #[cfg(feature = "zk")]
    let final_claim_public =
        masked_sumcheck_final_claim(prepared.input_claim, &sumcheck_proof, &rho)?;
    let final_terms = prover.final_terms().ok_or_else(|| {
        AkitaError::InvalidInput(format!(
            "{path} extension-opening reduction has not reached a final point"
        ))
    })?;
    let final_factor =
        tensor_equality_factor_eval_at_point::<F, E>(tail_point, &prepared.eta, &rho)?;
    if final_terms
        .iter()
        .any(|(_, _, factor)| *factor != final_factor)
    {
        return Err(AkitaError::InvalidInput(format!(
            "{path} extension-opening reduction transparent factor mismatch"
        )));
    }
    let expected_final = final_terms
        .into_iter()
        .fold(E::zero(), |acc, (coeff, witness, factor)| {
            acc + coeff * witness * factor
        });
    #[cfg(feature = "zk")]
    let final_claim = expected_final;
    #[cfg(not(feature = "zk"))]
    if final_claim != expected_final {
        return Err(AkitaError::InvalidInput(format!(
            "{path} extension-opening reduction final oracle mismatch"
        )));
    }
    let protocol_point = {
        let _span = tracing::info_span!("extension_opening_protocol_point").entered();
        ring_subfield_packed_extension_opening_point::<F, E, D>(rho.len(), &rho)?
    };
    let reduction = ExtensionOpeningReduction {
        proof: ExtensionOpeningReductionProof {
            partials: prepared.proof_partials,
            #[cfg(not(feature = "zk"))]
            sumcheck: sumcheck_proof,
            #[cfg(feature = "zk")]
            sumcheck_proof_masked: sumcheck_proof,
        },
        final_claim,
        #[cfg(feature = "zk")]
        final_claim_public,
        final_factor,
    };

    Ok(ProvedExtensionOpeningReduction {
        reduction,
        row_coefficients: prepared.row_coefficients,
        openings: prepared.openings,
        protocol_point,
    })
}

pub(in crate::protocol::core) type MultiplierWeightSlices<'a, F, const D: usize> =
    (&'a [CyclotomicRing<F, D>], &'a [CyclotomicRing<F, D>]);
pub(in crate::protocol::core) type FoldedClaimEvals<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<Vec<CyclotomicRing<F, D>>>);
