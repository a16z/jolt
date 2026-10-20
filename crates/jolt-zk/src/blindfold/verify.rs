use jolt_crypto::{HomomorphicCommitment, VectorCommitment, VectorCommitmentOpening};
use jolt_field::{CanonicalDecode, JoltField};
use jolt_poly::EqPolynomial;
use jolt_r1cs::{ConstraintMatrices, MatrixColumnContributions};
use jolt_sumcheck::{SumcheckClaim, SumcheckVerifier};
use jolt_transcript::{Channel, Sponge, VerifierTranscript};

use super::prove::{INNER_SUMCHECK_DEGREE, OUTER_SUMCHECK_DEGREE};
use super::wire::{receive_opening, FoldedEvaluation, FoldingCommitments, OuterClaims};
use super::{
    BlindFoldProtocol, RelaxedError, RelaxedInstance, VerificationError, WitnessCoordinate,
};

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: JoltField,
    Com: Copy + HomomorphicCommitment<F> + CanonicalDecode,
{
    /// Verifies the BlindFold proof read from `transcript`.
    pub fn verify<VC, H>(
        &self,
        vc_setup: &VC::Setup,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
        H: Sponge,
    {
        let folded = self.folded_instance(transcript)?;
        self.verify_folded_eval_witness_bindings::<VC, H>(vc_setup, &folded, transcript)?;
        let outer = self.verify_outer_folded_r1cs::<VC, H>(vc_setup, &folded, transcript)?;
        self.verify_inner_folded_r1cs::<VC, H>(vc_setup, &folded, &outer, transcript)?;
        Ok(())
    }

    fn folded_instance<H: Sponge>(
        &self,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<RelaxedInstance<F, Com>, VerificationError<F>> {
        let sent = FoldingCommitments::receive(&self.dimensions, transcript)?;
        let committed = self.committed_relaxed_instance(&sent.auxiliary_rows)?;
        let random = self.random_relaxed_instance(
            &sent.random_pad_rows,
            &sent.random_auxiliary_rows,
            &sent.random_error_rows,
            &sent.random_eval,
            sent.random_u,
        )?;
        self.validate_cross_term_error_rows(&sent.cross_term_error_rows)?;

        let folding_challenge = transcript.challenge_small();
        Ok(committed.fold(&random, &sent.cross_term_error_rows, folding_challenge)?)
    }

    fn verify_outer_folded_r1cs<VC, H>(
        &self,
        vc_setup: &VC::Setup,
        folded: &RelaxedInstance<F, Com>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<OuterCheck<F>, VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
        H: Sponge,
    {
        let error_row_count = self.dimensions.error.row_count;
        if error_row_count == 0 || !error_row_count.is_power_of_two() {
            return Err(VerificationError::InvalidPowerOfTwo {
                name: "error row count",
                value: error_row_count,
            });
        }
        let row_vars = error_row_count.trailing_zeros() as usize;

        let error_row_len = self.dimensions.error.row_len;
        if error_row_len == 0 || !error_row_len.is_power_of_two() {
            return Err(VerificationError::InvalidPowerOfTwo {
                name: "error row length",
                value: error_row_len,
            });
        }
        let entry_vars = error_row_len.trailing_zeros() as usize;
        let num_vars =
            row_vars
                .checked_add(entry_vars)
                .ok_or(VerificationError::InvalidPowerOfTwo {
                    name: "outer sumcheck dimension",
                    value: usize::MAX,
                })?;
        if num_vars == 0 {
            return Err(VerificationError::DegenerateSumcheck {
                name: "outer folded R1CS sumcheck",
            });
        }

        let tau = transcript.challenges_small(num_vars);
        let claim = SumcheckClaim::new(num_vars, OUTER_SUMCHECK_DEGREE, F::zero());
        let outer = SumcheckVerifier::verify_compressed(&claim, transcript)
            .map_err(|source| VerificationError::OuterSumcheck { source })?;

        let OuterClaims {
            abc: [az_rx, bz_rx, cz_rx],
            error_opening,
        } = OuterClaims::receive(error_row_len, transcript)?;

        let (row_point, entry_point) = outer.point.split_at(row_vars);
        let e_rx = VC::verify_committed_rows(
            vc_setup,
            &folded.error_row_commitments,
            row_point,
            entry_point,
            &error_opening,
        )?;

        let eq_tau_rx = EqPolynomial::<F>::mle(&tau, &outer.point);
        let expected = eq_tau_rx * (az_rx * bz_rx - folded.u * cz_rx - e_rx);
        if outer.value != expected {
            return Err(VerificationError::OuterFinalClaimMismatch {
                expected,
                actual: outer.value,
            });
        }

        Ok(OuterCheck {
            point: outer.point.into_vec(),
            az_rx,
            bz_rx,
            cz_rx,
        })
    }

    fn verify_folded_eval_witness_bindings<VC, H>(
        &self,
        vc_setup: &VC::Setup,
        folded: &RelaxedInstance<F, Com>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
        H: Sponge,
    {
        let FoldedEvaluation { output, blinding } = FoldedEvaluation::receive(transcript)?;
        let [eval_commitment] = folded.eval_commitments.as_slice() else {
            return Err(RelaxedError::LengthMismatch {
                name: "folded eval commitments",
                expected: 1,
                actual: folded.eval_commitments.len(),
            }
            .into());
        };
        if !VC::verify(vc_setup, eval_commitment, &[output], &blinding) {
            return Err(VerificationError::EvalCommitmentMismatch { index: 0 });
        }

        let witness_row_len = self.dimensions.witness.row_len;
        let coordinates = self.final_opening;
        for (coordinate, expected, kind) in [
            (coordinates.evaluation, output, "output"),
            (coordinates.blinding, blinding, "blinding"),
        ] {
            let opening = receive_opening(witness_row_len, transcript)?;
            let opened = coordinate.verify_opening::<F, VC>(vc_setup, folded, &opening)?;
            if opened != expected {
                return Err(VerificationError::EvalWitnessMismatch { kind, index: 0 });
            }
            coordinate.require_dedicated_row(&opening, kind, 0)?;
        }

        Ok(())
    }

    fn verify_inner_folded_r1cs<VC, H>(
        &self,
        vc_setup: &VC::Setup,
        folded: &RelaxedInstance<F, Com>,
        outer: &OuterCheck<F>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
        H: Sponge,
    {
        let ra: F = transcript.challenge_small();
        let rb: F = transcript.challenge_small();
        let rc: F = transcript.challenge_small();
        let public = public_contributions(&self.r1cs, &outer.point, folded.u)?;
        let claim = ra * (outer.az_rx - public.a)
            + rb * (outer.bz_rx - public.b)
            + rc * (outer.cz_rx - public.c);

        let witness_row_count = self.dimensions.witness.row_count;
        if witness_row_count == 0 || !witness_row_count.is_power_of_two() {
            return Err(VerificationError::InvalidPowerOfTwo {
                name: "witness row count",
                value: witness_row_count,
            });
        }
        let row_vars = witness_row_count.trailing_zeros() as usize;

        let witness_row_len = self.dimensions.witness.row_len;
        if witness_row_len == 0 || !witness_row_len.is_power_of_two() {
            return Err(VerificationError::InvalidPowerOfTwo {
                name: "witness row length",
                value: witness_row_len,
            });
        }
        let entry_vars = witness_row_len.trailing_zeros() as usize;
        let num_vars =
            row_vars
                .checked_add(entry_vars)
                .ok_or(VerificationError::InvalidPowerOfTwo {
                    name: "inner sumcheck dimension",
                    value: usize::MAX,
                })?;
        if num_vars == 0 {
            return Err(VerificationError::DegenerateSumcheck {
                name: "inner folded R1CS sumcheck",
            });
        }
        let inner_claim = SumcheckClaim::new(num_vars, INNER_SUMCHECK_DEGREE, claim);
        let inner = SumcheckVerifier::verify_compressed(&inner_claim, transcript)
            .map_err(|source| VerificationError::InnerSumcheck { source })?;
        let witness_opening = receive_opening(witness_row_len, transcript)?;

        let (row_point, entry_point) = inner.point.split_at(row_vars);
        let w_ry = VC::verify_committed_rows(
            vc_setup,
            &folded.witness_row_commitments,
            row_point,
            entry_point,
            &witness_opening,
        )?;

        let l_w_at_ry = compute_l_w_at_ry(&self.r1cs, &outer.point, &inner.point, ra, rb, rc)?;
        let expected = l_w_at_ry * w_ry;
        if inner.value != expected {
            return Err(VerificationError::InnerFinalClaimMismatch {
                expected,
                actual: inner.value,
            });
        }

        Ok(())
    }
}

/// The outer sumcheck's reduced point and the matrix evaluations received at it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct OuterCheck<F> {
    point: Vec<F>,
    az_rx: F,
    bz_rx: F,
    cz_rx: F,
}

impl WitnessCoordinate {
    fn require_dedicated_row<F: JoltField>(
        self,
        opening: &VectorCommitmentOpening<F>,
        kind: &'static str,
        index: usize,
    ) -> Result<(), VerificationError<F>> {
        for (slot, value) in opening.combined_vector.iter().enumerate() {
            if slot != self.column && !value.is_zero() {
                return Err(VerificationError::EvalWitnessRowNotDedicated { kind, index });
            }
        }
        Ok(())
    }

    fn verify_opening<F, VC>(
        self,
        vc_setup: &VC::Setup,
        folded: &RelaxedInstance<F, VC::Output>,
        opening: &VectorCommitmentOpening<F>,
    ) -> Result<F, VerificationError<F>>
    where
        F: JoltField,
        VC: VectorCommitment<Field = F>,
        VC::Output: Copy + HomomorphicCommitment<F>,
    {
        let witness_row_count = folded.witness_row_commitments.len();
        if witness_row_count == 0 || !witness_row_count.is_power_of_two() {
            return Err(VerificationError::InvalidPowerOfTwo {
                name: "witness row count",
                value: witness_row_count,
            });
        }
        let row_vars = witness_row_count.trailing_zeros() as usize;

        let witness_row_len = opening.combined_vector.len();
        if witness_row_len == 0 || !witness_row_len.is_power_of_two() {
            return Err(VerificationError::InvalidPowerOfTwo {
                name: "witness row length",
                value: witness_row_len,
            });
        }
        let entry_vars = witness_row_len.trailing_zeros() as usize;
        let row_point = boolean_point::<F>(self.row, row_vars)?;
        let entry_point = boolean_point::<F>(self.column, entry_vars)?;
        Ok(VC::verify_committed_rows(
            vc_setup,
            &folded.witness_row_commitments,
            &row_point,
            &entry_point,
            opening,
        )?)
    }
}

fn public_contributions<F>(
    r1cs: &ConstraintMatrices<F>,
    rx: &[F],
    u: F,
) -> Result<MatrixColumnContributions<F>, VerificationError<F>>
where
    F: JoltField,
{
    let eq_rx = EqPolynomial::<F>::evals(rx, None);
    Ok(r1cs.public_column_contributions(&eq_rx, 0, u)?)
}

fn compute_l_w_at_ry<F>(
    r1cs: &ConstraintMatrices<F>,
    rx: &[F],
    ry: &[F],
    ra: F,
    rb: F,
    rc: F,
) -> Result<F, VerificationError<F>>
where
    F: JoltField,
{
    let eq_rx = EqPolynomial::<F>::evals(rx, None);
    let eq_ry = EqPolynomial::<F>::evals(ry, None);
    let w_len = power_of_two_len::<F>("inner point dimension", ry.len())?;
    Ok(r1cs.linear_form_bilinear_eval(&eq_rx, &eq_ry, 1, w_len, [ra, rb, rc])?)
}

fn power_of_two_len<F>(name: &'static str, num_vars: usize) -> Result<usize, VerificationError<F>>
where
    F: JoltField,
{
    if num_vars >= usize::BITS as usize {
        return Err(VerificationError::InvalidPowerOfTwo {
            name,
            value: num_vars,
        });
    }
    Ok(1usize << num_vars)
}

fn boolean_point<F>(index: usize, num_vars: usize) -> Result<Vec<F>, VerificationError<F>>
where
    F: JoltField,
{
    let len = power_of_two_len::<F>("boolean point dimension", num_vars)?;
    if index >= len {
        return Err(VerificationError::InvalidPowerOfTwo {
            name: "boolean point index",
            value: index,
        });
    }
    Ok((0..num_vars)
        .map(|bit| {
            let shift = num_vars - bit - 1;
            if ((index >> shift) & 1) == 1 {
                F::one()
            } else {
                F::zero()
            }
        })
        .collect())
}
