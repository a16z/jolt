use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_crypto::{HomomorphicCommitment, VectorCommitment, VectorCommitmentOpening};
use jolt_field::{Field, FieldCore, RingAccumulator, WithAccumulator};
use jolt_poly::EqPolynomial;
use jolt_r1cs::{ConstraintMatrices, MatrixColumnContributions};
use jolt_sumcheck::{BooleanHypercube, SumcheckClaim, SumcheckVerifier};
use jolt_transcript::{FsNargRead, FsTranscript};

use crate::{
    BlindFoldProtocol, RelaxedError, RelaxedInstance, VerificationError, WitnessCoordinate,
};

const OUTER_SUMCHECK_DEGREE: usize = 3;
const INNER_SUMCHECK_DEGREE: usize = 2;

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field,
    Com: Copy + HomomorphicCommitment<F> + CanonicalSerialize,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn verify_from_narg<VC, T>(
        &self,
        vc_setup: &VC::Setup,
        transcript: &mut T,
    ) -> Result<(), VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
        T: FsNargRead + FsTranscript<F>,
        Com: CanonicalDeserialize,
    {
        let folded = self.folded_instance_from_narg(transcript)?;
        let folded_eval_outputs = read_field_vec(transcript, "folded eval outputs")?;
        let folded_eval_blindings = read_field_vec(transcript, "folded eval blindings")?;
        ensure_len(
            "folded eval outputs",
            folded.eval_commitments.len(),
            folded_eval_outputs.len(),
        )?;
        ensure_len(
            "folded eval blindings",
            folded.eval_commitments.len(),
            folded_eval_blindings.len(),
        )?;
        verify_folded_eval_commitments::<F, VC>(
            vc_setup,
            &folded,
            &folded_eval_outputs,
            &folded_eval_blindings,
        )?;

        let coordinates = self.final_opening_witness_coordinates()?;
        let expected_outputs = coordinates
            .iter()
            .filter(|coordinates| coordinates.evaluation.is_some())
            .count();
        let expected_blindings = coordinates
            .iter()
            .filter(|coordinates| coordinates.blinding.is_some())
            .count();
        let folded_eval_output_openings = read_final_openings(
            transcript,
            expected_outputs,
            self.dimensions.witness.row_len,
        )?;
        let folded_eval_blinding_openings = read_final_openings(
            transcript,
            expected_blindings,
            self.dimensions.witness.row_len,
        )?;
        self.verify_folded_eval_witness_bindings_from_narg::<VC>(
            vc_setup,
            &folded,
            &folded_eval_outputs,
            &folded_eval_blindings,
            &folded_eval_output_openings,
            &folded_eval_blinding_openings,
        )?;

        let outer =
            self.verify_outer_folded_r1cs_from_narg::<VC, T>(vc_setup, &folded, transcript)?;
        self.verify_inner_folded_r1cs_from_narg::<VC, T>(vc_setup, &folded, &outer, transcript)?;
        Ok(())
    }
}

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field,
    Com: Clone + HomomorphicCommitment<F> + CanonicalSerialize,
{
    fn folded_instance_from_narg<T>(
        &self,
        transcript: &mut T,
    ) -> Result<RelaxedInstance<F, Com>, VerificationError<F>>
    where
        T: FsNargRead + FsTranscript<F>,
        Com: CanonicalDeserialize,
    {
        let committed = read_committed_instance_from_narg(self, transcript)?;

        let random_u = read_field_one(transcript, "random u")?;
        let random_round_commitments = read_narg_slice(transcript, "random round commitments")?;
        let random_output_claim_row_commitments =
            read_narg_slice(transcript, "random output claim row commitments")?;
        let random_auxiliary_row_commitments =
            read_narg_slice(transcript, "random auxiliary row commitments")?;
        let random_error_row_commitments =
            read_narg_slice(transcript, "random error row commitments")?;
        let random_eval_commitments = read_narg_slice(transcript, "random eval commitments")?;
        let random = self.random_relaxed_instance(
            &random_round_commitments,
            &random_output_claim_row_commitments,
            &random_auxiliary_row_commitments,
            &random_error_row_commitments,
            &random_eval_commitments,
            random_u,
        )?;

        let cross_term_error_row_commitments =
            read_narg_slice(transcript, "cross-term error row commitments")?;
        self.validate_cross_term_error_rows(&cross_term_error_row_commitments)?;

        let folding_challenge = transcript.challenge();
        Ok(committed.fold(
            &random,
            &cross_term_error_row_commitments,
            folding_challenge,
        )?)
    }
}

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field,
    Com: Copy + HomomorphicCommitment<F> + CanonicalSerialize,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    fn verify_outer_folded_r1cs_from_narg<VC, T>(
        &self,
        vc_setup: &VC::Setup,
        folded: &RelaxedInstance<F, Com>,
        transcript: &mut T,
    ) -> Result<OuterCheck<F>, VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
        T: FsNargRead + FsTranscript<F>,
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

        let tau = transcript.challenge_vector(num_vars);
        let claim = SumcheckClaim::new(num_vars, OUTER_SUMCHECK_DEGREE, F::zero());
        let outer =
            SumcheckVerifier::verify_compressed_from_narg(&claim, BooleanHypercube, transcript)
                .map_err(|source| VerificationError::OuterSumcheck { source })?;

        let azbzcz = read_field_vec(transcript, "outer final claims")?;
        let [az_rx, bz_rx, cz_rx] = <[F; 3]>::try_from(azbzcz.as_slice()).map_err(|_| {
            VerificationError::MalformedNarg {
                name: "outer final claims",
            }
        })?;
        let error_opening = read_vector_opening(transcript, error_row_len, "error opening")?;

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
}

fn verify_folded_eval_commitments<F, VC>(
    vc_setup: &VC::Setup,
    folded: &RelaxedInstance<F, VC::Output>,
    folded_eval_outputs: &[F],
    folded_eval_blindings: &[F],
) -> Result<(), VerificationError<F>>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + CanonicalSerialize,
{
    for (index, ((commitment, &output), &blinding)) in folded
        .eval_commitments
        .iter()
        .zip(folded_eval_outputs)
        .zip(folded_eval_blindings)
        .enumerate()
    {
        if !VC::verify(vc_setup, commitment, &[output], &blinding) {
            return Err(VerificationError::EvalCommitmentMismatch { index });
        }
    }

    Ok(())
}

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field,
    Com: Copy + HomomorphicCommitment<F> + CanonicalSerialize,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    fn verify_folded_eval_witness_bindings_from_narg<VC>(
        &self,
        vc_setup: &VC::Setup,
        folded: &RelaxedInstance<F, Com>,
        folded_eval_outputs: &[F],
        folded_eval_blindings: &[F],
        folded_eval_output_openings: &[VectorCommitmentOpening<F>],
        folded_eval_blinding_openings: &[VectorCommitmentOpening<F>],
    ) -> Result<(), VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
    {
        let coordinates = self.final_opening_witness_coordinates()?;
        ensure_len(
            "final opening bindings",
            coordinates.len(),
            folded.eval_commitments.len(),
        )?;

        let expected_outputs = coordinates
            .iter()
            .filter(|coordinates| coordinates.evaluation.is_some())
            .count();
        ensure_len(
            "folded eval output witness openings",
            expected_outputs,
            folded_eval_output_openings.len(),
        )?;
        let expected_blindings = coordinates
            .iter()
            .filter(|coordinates| coordinates.blinding.is_some())
            .count();
        ensure_len(
            "folded eval blinding witness openings",
            expected_blindings,
            folded_eval_blinding_openings.len(),
        )?;

        let mut output_openings = folded_eval_output_openings.iter();
        let mut blinding_openings = folded_eval_blinding_openings.iter();
        for (index, coordinates) in coordinates.iter().enumerate() {
            if let Some(coordinate) = coordinates.evaluation {
                let opening = output_openings.next().ok_or(RelaxedError::LengthMismatch {
                    name: "folded eval output witness openings",
                    expected: expected_outputs,
                    actual: folded_eval_output_openings.len(),
                })?;
                let opened = coordinate.verify_opening::<F, VC>(vc_setup, folded, opening)?;
                if opened != folded_eval_outputs[index] {
                    return Err(VerificationError::EvalWitnessMismatch {
                        kind: "output",
                        index,
                    });
                }
                coordinate.require_dedicated_row(opening, "output", index)?;
            }

            if let Some(coordinate) = coordinates.blinding {
                let opening = blinding_openings
                    .next()
                    .ok_or(RelaxedError::LengthMismatch {
                        name: "folded eval blinding witness openings",
                        expected: expected_blindings,
                        actual: folded_eval_blinding_openings.len(),
                    })?;
                let opened = coordinate.verify_opening::<F, VC>(vc_setup, folded, opening)?;
                if opened != folded_eval_blindings[index] {
                    return Err(VerificationError::EvalWitnessMismatch {
                        kind: "blinding",
                        index,
                    });
                }
                coordinate.require_dedicated_row(opening, "blinding", index)?;
            }
        }

        Ok(())
    }
}

impl WitnessCoordinate {
    fn require_dedicated_row<F: Field>(
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
        F: Field,
        VC: VectorCommitment<Field = F>,
        VC::Output: Copy + HomomorphicCommitment<F>,
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
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

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field,
    Com: Copy + HomomorphicCommitment<F> + CanonicalSerialize,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    fn verify_inner_folded_r1cs_from_narg<VC, T>(
        &self,
        vc_setup: &VC::Setup,
        folded: &RelaxedInstance<F, Com>,
        outer: &OuterCheck<F>,
        transcript: &mut T,
    ) -> Result<(), VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
        T: FsNargRead + FsTranscript<F>,
    {
        let ra = transcript.challenge();
        let rb = transcript.challenge();
        let rc = transcript.challenge();
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
        let inner = SumcheckVerifier::verify_compressed_from_narg(
            &inner_claim,
            BooleanHypercube,
            transcript,
        )
        .map_err(|source| VerificationError::InnerSumcheck { source })?;

        let witness_opening = read_vector_opening(transcript, witness_row_len, "witness opening")?;
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct OuterCheck<F> {
    point: Vec<F>,
    az_rx: F,
    bz_rx: F,
    cz_rx: F,
}

fn read_committed_instance_from_narg<F, Com, T>(
    protocol: &BlindFoldProtocol<F, Com>,
    transcript: &mut T,
) -> Result<RelaxedInstance<F, Com>, VerificationError<F>>
where
    F: Field,
    Com: Clone + HomomorphicCommitment<F> + CanonicalSerialize + CanonicalDeserialize,
    T: FsNargRead + FsTranscript<F>,
{
    transcript.absorb_field(&F::one());
    let round_commitments = protocol
        .sumcheck_consistency
        .iter()
        .flat_map(|consistency| consistency.rounds.iter())
        .map(|round| round.commitment.clone())
        .collect::<Vec<_>>();
    transcript.absorb(&round_commitments);
    let output_claim_row_commitments = protocol
        .committed_output_claims
        .iter()
        .flat_map(|claims| claims.commitments.iter().cloned())
        .collect::<Vec<_>>();
    transcript.absorb(&output_claim_row_commitments);
    let auxiliary_row_commitments = read_narg_slice(transcript, "auxiliary row commitments")?;
    let committed = protocol.committed_relaxed_instance(&auxiliary_row_commitments)?;
    transcript.absorb(&committed.error_row_commitments);
    transcript.absorb(&committed.eval_commitments);
    Ok(committed)
}

fn read_narg_slice<F, T, Value>(
    transcript: &mut T,
    name: &'static str,
) -> Result<Vec<Value>, VerificationError<F>>
where
    F: Field,
    T: FsNargRead + FsTranscript<F>,
    Value: CanonicalDeserialize,
{
    transcript
        .read_slice()
        .map_err(|_| VerificationError::MalformedNarg { name })
}

fn read_field_vec<F, T>(
    transcript: &mut T,
    name: &'static str,
) -> Result<Vec<F>, VerificationError<F>>
where
    F: Field,
    T: FsNargRead + FsTranscript<F>,
{
    transcript
        .read_field_slice()
        .map_err(|_| VerificationError::MalformedNarg { name })
}

fn read_field_one<F, T>(transcript: &mut T, name: &'static str) -> Result<F, VerificationError<F>>
where
    F: Field,
    T: FsNargRead + FsTranscript<F>,
{
    let values = read_field_vec(transcript, name)?;
    match <[F; 1]>::try_from(values.as_slice()) {
        Ok([value]) => Ok(value),
        Err(_) => Err(VerificationError::MalformedNarg { name }),
    }
}

fn read_final_openings<F, T>(
    transcript: &mut T,
    count: usize,
    row_len: usize,
) -> Result<Vec<VectorCommitmentOpening<F>>, VerificationError<F>>
where
    F: Field,
    T: FsNargRead + FsTranscript<F>,
{
    (0..count)
        .map(|_| read_vector_opening(transcript, row_len, "folded eval witness opening"))
        .collect()
}

fn read_vector_opening<F, T>(
    transcript: &mut T,
    expected_len: usize,
    name: &'static str,
) -> Result<VectorCommitmentOpening<F>, VerificationError<F>>
where
    F: Field,
    T: FsNargRead + FsTranscript<F>,
{
    let combined_vector = read_field_vec(transcript, name)?;
    if combined_vector.len() != expected_len {
        return Err(VerificationError::MalformedNarg { name });
    }
    let combined_blinding = read_field_one(transcript, name)?;
    Ok(VectorCommitmentOpening {
        combined_vector,
        combined_blinding,
    })
}

fn public_contributions<F>(
    r1cs: &ConstraintMatrices<F>,
    rx: &[F],
    u: F,
) -> Result<MatrixColumnContributions<F>, VerificationError<F>>
where
    F: Field,
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
    F: Field,
{
    let eq_rx = EqPolynomial::<F>::evals(rx, None);
    let eq_ry = EqPolynomial::<F>::evals(ry, None);
    let w_len = power_of_two_len::<F>("inner point dimension", ry.len())?;
    Ok(r1cs.linear_form_bilinear_eval(&eq_rx, &eq_ry, 1, w_len, [ra, rb, rc])?)
}

fn power_of_two_len<F>(name: &'static str, num_vars: usize) -> Result<usize, VerificationError<F>>
where
    F: FieldCore,
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
    F: Field,
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

fn ensure_len(name: &'static str, expected: usize, actual: usize) -> Result<(), RelaxedError> {
    if expected != actual {
        return Err(RelaxedError::LengthMismatch {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}
