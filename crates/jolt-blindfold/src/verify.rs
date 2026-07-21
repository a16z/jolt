use jolt_crypto::{HomomorphicCommitment, VectorCommitment, VectorCommitmentOpening};
use jolt_field::{Field, FieldCore};
use jolt_poly::EqPolynomial;
use jolt_r1cs::{ConstraintMatrices, MatrixColumnContributions};
use jolt_sumcheck::{BooleanHypercube, SumcheckClaim, SUMCHECK_ROUND_TRANSCRIPT_LABEL};
use jolt_transcript::{AppendToTranscript, Label, Transcript};

use crate::{
    BlindFoldProof, BlindFoldProtocol, RelaxedError, RelaxedInstance, VerificationError,
    WitnessCoordinate,
};

const OUTER_SUMCHECK_DEGREE: usize = 3;
const INNER_SUMCHECK_DEGREE: usize = 2;
const INNER_SUMCHECK_LABEL: &[u8] = b"inner_sumcheck_poly";

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field + AppendToTranscript,
    Com: Copy + HomomorphicCommitment<F> + AppendToTranscript,
{
    pub fn verify<VC, T>(
        &self,
        proof: &BlindFoldProof<F, Com>,
        vc_setup: &VC::Setup,
        transcript: &mut T,
    ) -> Result<(), VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
        T: Transcript<Challenge = F>,
    {
        let folded = self.folded_instance_from_proof(proof, transcript)?;
        ensure_len(
            "folded eval outputs",
            folded.eval_commitments.len(),
            proof.folded_eval_outputs.len(),
        )?;
        ensure_len(
            "folded eval blindings",
            folded.eval_commitments.len(),
            proof.folded_eval_blindings.len(),
        )?;
        proof.verify_folded_eval_commitments::<VC>(vc_setup, &folded)?;
        self.verify_folded_eval_witness_bindings::<VC, T>(proof, vc_setup, &folded, transcript)?;
        let outer = self.verify_outer_folded_r1cs::<VC, T>(proof, vc_setup, &folded, transcript)?;
        self.verify_inner_folded_r1cs::<VC, T>(proof, vc_setup, &folded, &outer, transcript)?;
        Ok(())
    }
}

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field + AppendToTranscript,
    Com: Clone + HomomorphicCommitment<F> + AppendToTranscript,
{
    fn folded_instance_from_proof<T>(
        &self,
        proof: &BlindFoldProof<F, Com>,
        transcript: &mut T,
    ) -> Result<RelaxedInstance<F, Com>, VerificationError<F>>
    where
        T: Transcript<Challenge = F>,
    {
        let committed = self.committed_relaxed_instance(&proof.auxiliary_row_commitments)?;
        committed.append_to_transcript(
            transcript,
            b"bf_committed_u",
            b"bf_committed_w",
            b"bf_committed_e",
            b"bf_committed_eval",
        );

        let random = self.random_relaxed_instance(
            &proof.random_round_commitments,
            &proof.random_output_claim_row_commitments,
            &proof.random_auxiliary_row_commitments,
            &proof.random_error_row_commitments,
            &proof.random_eval_commitments,
            proof.random_u,
        )?;
        random.append_to_transcript(
            transcript,
            b"bf_random_u",
            b"bf_random_w",
            b"bf_random_e",
            b"bf_random_eval",
        );

        self.validate_cross_term_error_rows(&proof.cross_term_error_row_commitments)?;
        transcript.append_values(b"bf_cross_e", &proof.cross_term_error_row_commitments);

        let folding_challenge = transcript.challenge();
        Ok(committed.fold(
            &random,
            &proof.cross_term_error_row_commitments,
            folding_challenge,
        )?)
    }
}

impl<F, Com> RelaxedInstance<F, Com>
where
    F: AppendToTranscript,
    Com: AppendToTranscript,
{
    fn append_to_transcript<T>(
        &self,
        transcript: &mut T,
        u_label: &'static [u8],
        witness_label: &'static [u8],
        error_label: &'static [u8],
        eval_label: &'static [u8],
    ) where
        T: Transcript,
    {
        transcript.append(&Label(u_label));
        self.u.append_to_transcript(transcript);
        transcript.append_values(witness_label, &self.witness_row_commitments);
        transcript.append_values(error_label, &self.error_row_commitments);
        transcript.append_values(eval_label, &self.eval_commitments);
    }
}

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field + AppendToTranscript,
    Com: Copy + HomomorphicCommitment<F> + AppendToTranscript,
{
    fn verify_outer_folded_r1cs<VC, T>(
        &self,
        proof: &BlindFoldProof<F, Com>,
        vc_setup: &VC::Setup,
        folded: &RelaxedInstance<F, Com>,
        transcript: &mut T,
    ) -> Result<OuterCheck<F>, VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
        T: Transcript<Challenge = F>,
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

        transcript.append(&Label(b"bf_spartan"));
        let tau = transcript.challenge_vector(num_vars);
        let claim = SumcheckClaim::new(num_vars, OUTER_SUMCHECK_DEGREE, F::zero());
        let outer = proof
            .outer_sumcheck
            .verify(
                &claim,
                BooleanHypercube,
                SUMCHECK_ROUND_TRANSCRIPT_LABEL,
                transcript,
            )
            .map_err(|source| VerificationError::OuterSumcheck { source })?;

        let (row_point, entry_point) = outer.point.split_at(row_vars);
        let e_rx = VC::verify_committed_rows(
            vc_setup,
            &folded.error_row_commitments,
            row_point,
            entry_point,
            &proof.error_opening,
        )?;

        let eq_tau_rx = EqPolynomial::<F>::mle(&tau, &outer.point);
        let expected = eq_tau_rx * (proof.az_rx * proof.bz_rx - folded.u * proof.cz_rx - e_rx);
        if outer.value != expected {
            return Err(VerificationError::OuterFinalClaimMismatch {
                expected,
                actual: outer.value,
            });
        }

        transcript.append_values(b"bf_az_bz_cz", &[proof.az_rx, proof.bz_rx, proof.cz_rx]);
        append_vector_opening(
            transcript,
            b"bf_error_opening",
            b"bf_error_blind",
            &proof.error_opening,
        );

        Ok(OuterCheck {
            point: outer.point.into_vec(),
        })
    }
}

impl<F, Com> BlindFoldProof<F, Com>
where
    F: Field,
    Com: Copy + AppendToTranscript,
{
    fn verify_folded_eval_commitments<VC>(
        &self,
        vc_setup: &VC::Setup,
        folded: &RelaxedInstance<F, Com>,
    ) -> Result<(), VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
    {
        for (index, ((commitment, &output), &blinding)) in folded
            .eval_commitments
            .iter()
            .zip(&self.folded_eval_outputs)
            .zip(&self.folded_eval_blindings)
            .enumerate()
        {
            if !VC::verify(vc_setup, commitment, &[output], &blinding) {
                return Err(VerificationError::EvalCommitmentMismatch { index });
            }
        }

        Ok(())
    }
}

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field + AppendToTranscript,
    Com: Copy + HomomorphicCommitment<F> + AppendToTranscript,
{
    fn verify_folded_eval_witness_bindings<VC, T>(
        &self,
        proof: &BlindFoldProof<F, Com>,
        vc_setup: &VC::Setup,
        folded: &RelaxedInstance<F, Com>,
        transcript: &mut T,
    ) -> Result<(), VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
        T: Transcript,
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
            proof.folded_eval_output_openings.len(),
        )?;
        let expected_blindings = coordinates
            .iter()
            .filter(|coordinates| coordinates.blinding.is_some())
            .count();
        ensure_len(
            "folded eval blinding witness openings",
            expected_blindings,
            proof.folded_eval_blinding_openings.len(),
        )?;

        let mut output_openings = proof.folded_eval_output_openings.iter();
        let mut blinding_openings = proof.folded_eval_blinding_openings.iter();
        for (index, coordinates) in coordinates.iter().enumerate() {
            if let Some(coordinate) = coordinates.evaluation {
                let opening = output_openings.next().ok_or(RelaxedError::LengthMismatch {
                    name: "folded eval output witness openings",
                    expected: expected_outputs,
                    actual: proof.folded_eval_output_openings.len(),
                })?;
                let opened = coordinate.verify_opening::<F, VC>(vc_setup, folded, opening)?;
                if opened != proof.folded_eval_outputs[index] {
                    return Err(VerificationError::EvalWitnessMismatch {
                        kind: "output",
                        index,
                    });
                }
                coordinate.require_dedicated_row(opening, "output", index)?;
                append_vector_opening(
                    transcript,
                    b"bf_eval_out_open",
                    b"bf_eval_out_blind",
                    opening,
                );
            }

            if let Some(coordinate) = coordinates.blinding {
                let opening = blinding_openings
                    .next()
                    .ok_or(RelaxedError::LengthMismatch {
                        name: "folded eval blinding witness openings",
                        expected: expected_blindings,
                        actual: proof.folded_eval_blinding_openings.len(),
                    })?;
                let opened = coordinate.verify_opening::<F, VC>(vc_setup, folded, opening)?;
                if opened != proof.folded_eval_blindings[index] {
                    return Err(VerificationError::EvalWitnessMismatch {
                        kind: "blinding",
                        index,
                    });
                }
                coordinate.require_dedicated_row(opening, "blinding", index)?;
                append_vector_opening(
                    transcript,
                    b"bf_eval_blind_open",
                    b"bf_eval_blind_bl",
                    opening,
                );
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
    F: Field + AppendToTranscript,
    Com: Copy + HomomorphicCommitment<F> + AppendToTranscript,
{
    fn verify_inner_folded_r1cs<VC, T>(
        &self,
        proof: &BlindFoldProof<F, Com>,
        vc_setup: &VC::Setup,
        folded: &RelaxedInstance<F, Com>,
        outer: &OuterCheck<F>,
        transcript: &mut T,
    ) -> Result<(), VerificationError<F>>
    where
        VC: VectorCommitment<Field = F, Output = Com>,
        T: Transcript<Challenge = F>,
    {
        let ra = transcript.challenge();
        let rb = transcript.challenge();
        let rc = transcript.challenge();
        let public = public_contributions(&self.r1cs, &outer.point, folded.u)?;
        let claim = ra * (proof.az_rx - public.a)
            + rb * (proof.bz_rx - public.b)
            + rc * (proof.cz_rx - public.c);

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
        let inner = proof
            .inner_sumcheck
            .verify(
                &inner_claim,
                BooleanHypercube,
                INNER_SUMCHECK_LABEL,
                transcript,
            )
            .map_err(|source| VerificationError::InnerSumcheck { source })?;

        let (row_point, entry_point) = inner.point.split_at(row_vars);
        let w_ry = VC::verify_committed_rows(
            vc_setup,
            &folded.witness_row_commitments,
            row_point,
            entry_point,
            &proof.witness_opening,
        )?;

        let l_w_at_ry = compute_l_w_at_ry(&self.r1cs, &outer.point, &inner.point, ra, rb, rc)?;
        let expected = l_w_at_ry * w_ry;
        if inner.value != expected {
            return Err(VerificationError::InnerFinalClaimMismatch {
                expected,
                actual: inner.value,
            });
        }

        append_vector_opening(
            transcript,
            b"bf_witness_opening",
            b"bf_witness_blind",
            &proof.witness_opening,
        );

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OuterCheck<F> {
    point: Vec<F>,
}

fn append_vector_opening<F, T>(
    transcript: &mut T,
    row_label: &'static [u8],
    blinding_label: &'static [u8],
    opening: &VectorCommitmentOpening<F>,
) where
    F: AppendToTranscript,
    T: Transcript,
{
    transcript.append_values(row_label, &opening.combined_vector);
    transcript.append(&Label(blinding_label));
    opening.combined_blinding.append_to_transcript(transcript);
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

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests should fail loudly")]
mod tests {
    use super::*;
    use crate::{
        r1cs::{FinalOpeningLayout, Layout},
        BlindFoldDimensions, RowDimensions, WitnessRowLayout,
    };
    use jolt_crypto::{
        Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment, VectorOpeningError,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::CompressedPoly;
    use jolt_r1cs::ConstraintMatrices;
    use jolt_sumcheck::CompressedSumcheckProof;
    use jolt_transcript::Blake2bTranscript;

    fn f(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn setup() -> PedersenSetup<Bn254G1> {
        let generator = Bn254::g1_generator();
        let message_generators = (1..=4).map(|i| generator.scalar_mul(&f(i))).collect();
        PedersenSetup::new(message_generators, generator.scalar_mul(&f(99)))
    }

    fn commitment(setup: &PedersenSetup<Bn254G1>, value: u64) -> Bn254G1 {
        Pedersen::<Bn254G1>::commit(setup, &[f(value)], &f(value + 1000))
    }

    fn commit_value(setup: &PedersenSetup<Bn254G1>, value: Fr, blinding: Fr) -> Bn254G1 {
        Pedersen::<Bn254G1>::commit(setup, &[value], &blinding)
    }

    fn identity() -> Bn254G1 {
        <Bn254G1 as JoltGroup>::identity()
    }

    fn protocol(setup: &PedersenSetup<Bn254G1>) -> BlindFoldProtocol<Fr, Bn254G1> {
        let _ = setup;
        empty_protocol(Vec::new())
    }

    fn protocol_with_eval(setup: &PedersenSetup<Bn254G1>) -> BlindFoldProtocol<Fr, Bn254G1> {
        empty_protocol(vec![commit_value(setup, f(7), f(70))])
    }

    fn empty_protocol(eval_commitments: Vec<Bn254G1>) -> BlindFoldProtocol<Fr, Bn254G1> {
        BlindFoldProtocol {
            sumcheck_consistency: Vec::new(),
            committed_output_claims: Vec::new(),
            r1cs: ConstraintMatrices::new(0, 1, Vec::new(), Vec::new(), Vec::new()),
            layout: Layout {
                witness_row_len: 1,
                stages: Vec::new(),
                final_openings: vec![
                    FinalOpeningLayout {
                        evaluation: None,
                        blinding: None,
                    };
                    eval_commitments.len()
                ],
            },
            dimensions: BlindFoldDimensions {
                witness: RowDimensions {
                    row_len: 1,
                    row_count: 1,
                },
                error: RowDimensions {
                    row_len: 1,
                    row_count: 1,
                },
                witness_rows: WitnessRowLayout {
                    coefficients: 0..0,
                    auxiliary: 0..0,
                    output_claims: 0..0,
                    padding: 0..1,
                },
                coefficient_rows: 0,
                output_claim_rows: 0,
                auxiliary_rows: 0,
                coefficient_values: 0,
                auxiliary_values: 0,
            },
            eval_commitments,
        }
    }

    fn witness_protocol() -> BlindFoldProtocol<Fr, Bn254G1> {
        BlindFoldProtocol {
            sumcheck_consistency: Vec::new(),
            committed_output_claims: Vec::new(),
            r1cs: ConstraintMatrices::new(
                1,
                2,
                vec![vec![(1, f(1))]],
                vec![Vec::new()],
                vec![Vec::new()],
            ),
            layout: Layout {
                witness_row_len: 1,
                stages: Vec::new(),
                final_openings: Vec::new(),
            },
            dimensions: BlindFoldDimensions {
                witness: RowDimensions {
                    row_len: 1,
                    row_count: 1,
                },
                error: RowDimensions {
                    row_len: 1,
                    row_count: 1,
                },
                witness_rows: WitnessRowLayout {
                    coefficients: 0..0,
                    output_claims: 0..0,
                    auxiliary: 0..1,
                    padding: 1..1,
                },
                coefficient_rows: 0,
                output_claim_rows: 0,
                auxiliary_rows: 1,
                coefficient_values: 0,
                auxiliary_values: 1,
            },
            eval_commitments: Vec::new(),
        }
    }

    fn inner_round_protocol() -> BlindFoldProtocol<Fr, Bn254G1> {
        let mut protocol = witness_protocol();
        protocol.dimensions.witness = RowDimensions {
            row_len: 1,
            row_count: 2,
        };
        protocol.dimensions.error = RowDimensions {
            row_len: 1,
            row_count: 2,
        };
        protocol.dimensions.witness_rows.auxiliary = 0..2;
        protocol.dimensions.witness_rows.padding = 2..2;
        protocol.dimensions.auxiliary_rows = 2;
        protocol.dimensions.auxiliary_values = 2;
        protocol
    }

    fn add_zero_inner_round(proof: &mut BlindFoldProof<Fr, Bn254G1>) {
        proof.inner_sumcheck.round_polynomials = vec![CompressedPoly::new(vec![f(0)])];
    }

    fn outer_round_protocol() -> BlindFoldProtocol<Fr, Bn254G1> {
        let mut protocol = empty_protocol(Vec::new());
        protocol.dimensions.error = RowDimensions {
            row_len: 1,
            row_count: 2,
        };
        protocol
    }

    fn coefficient_row_protocol() -> BlindFoldProtocol<Fr, Bn254G1> {
        let mut protocol = empty_protocol(Vec::new());
        protocol.dimensions.witness_rows = WitnessRowLayout {
            coefficients: 0..1,
            output_claims: 1..1,
            auxiliary: 1..1,
            padding: 1..1,
        };
        protocol.dimensions.coefficient_rows = 1;
        protocol.dimensions.coefficient_values = 1;
        protocol
    }

    fn opening(row_len: usize) -> VectorCommitmentOpening<Fr> {
        VectorCommitmentOpening {
            combined_vector: vec![f(0); row_len],
            combined_blinding: f(0),
        }
    }

    fn zero_outer_sumcheck(
        protocol: &BlindFoldProtocol<Fr, Bn254G1>,
    ) -> CompressedSumcheckProof<Fr> {
        let num_vars = protocol.dimensions.error.row_count.trailing_zeros() as usize
            + protocol.dimensions.error.row_len.trailing_zeros() as usize;
        CompressedSumcheckProof {
            round_polynomials: vec![CompressedPoly::new(vec![f(0)]); num_vars],
        }
    }

    fn proof(
        setup: &PedersenSetup<Bn254G1>,
        protocol: &BlindFoldProtocol<Fr, Bn254G1>,
    ) -> BlindFoldProof<Fr, Bn254G1> {
        BlindFoldProof {
            auxiliary_row_commitments: vec![
                commitment(setup, 41);
                protocol.dimensions.auxiliary_rows
            ],
            random_round_commitments: vec![identity(); protocol.dimensions.coefficient_rows],
            random_output_claim_row_commitments: vec![
                identity();
                protocol.dimensions.output_claim_rows
            ],
            random_auxiliary_row_commitments: vec![identity(); protocol.dimensions.auxiliary_rows],
            random_error_row_commitments: vec![identity(); protocol.dimensions.error.row_count],
            random_eval_commitments: vec![
                commit_value(setup, f(11), f(110));
                protocol.eval_commitments.len()
            ],
            random_u: f(3),
            cross_term_error_row_commitments: vec![identity(); protocol.dimensions.error.row_count],
            outer_sumcheck: zero_outer_sumcheck(protocol),
            az_rx: f(0),
            bz_rx: f(0),
            cz_rx: f(0),
            inner_sumcheck: CompressedSumcheckProof::default(),
            witness_opening: opening(protocol.dimensions.witness.row_len),
            error_opening: opening(protocol.dimensions.error.row_len),
            folded_eval_outputs: vec![f(0); protocol.eval_commitments.len()],
            folded_eval_blindings: vec![f(0); protocol.eval_commitments.len()],
            folded_eval_output_openings: Vec::new(),
            folded_eval_blinding_openings: Vec::new(),
        }
    }

    fn folding_challenge(
        protocol: &BlindFoldProtocol<Fr, Bn254G1>,
        proof: &BlindFoldProof<Fr, Bn254G1>,
    ) -> Fr {
        let committed = protocol
            .committed_relaxed_instance(&proof.auxiliary_row_commitments)
            .expect("committed instance builds");
        let random = protocol
            .random_relaxed_instance(
                &proof.random_round_commitments,
                &proof.random_output_claim_row_commitments,
                &proof.random_auxiliary_row_commitments,
                &proof.random_error_row_commitments,
                &proof.random_eval_commitments,
                proof.random_u,
            )
            .expect("random instance builds");
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");
        committed.append_to_transcript(
            &mut transcript,
            b"bf_committed_u",
            b"bf_committed_w",
            b"bf_committed_e",
            b"bf_committed_eval",
        );
        random.append_to_transcript(
            &mut transcript,
            b"bf_random_u",
            b"bf_random_w",
            b"bf_random_e",
            b"bf_random_eval",
        );
        transcript.append_values(b"bf_cross_e", &proof.cross_term_error_row_commitments);
        transcript.challenge()
    }

    fn proof_with_valid_eval_opening(
        setup: &PedersenSetup<Bn254G1>,
        protocol: &BlindFoldProtocol<Fr, Bn254G1>,
    ) -> BlindFoldProof<Fr, Bn254G1> {
        let mut proof = proof(setup, protocol);
        let folding_challenge = folding_challenge(protocol, &proof);
        proof.folded_eval_outputs = vec![f(7) + folding_challenge * f(11)];
        proof.folded_eval_blindings = vec![f(70) + folding_challenge * f(110)];
        proof
    }

    #[test]
    fn verify_rejects_degenerate_outer_sumcheck() {
        let setup = setup();
        let protocol = protocol(&setup);
        let proof = proof(&setup, &protocol);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");

        let error = protocol
            .verify::<Pedersen<Bn254G1>, _>(&proof, &setup, &mut transcript)
            .expect_err("degenerate outer sumcheck is rejected");

        assert!(matches!(
            error,
            VerificationError::DegenerateSumcheck {
                name: "outer folded R1CS sumcheck"
            }
        ));
    }

    #[test]
    fn folded_instance_uses_transcript_derived_challenge() {
        let setup = setup();
        let protocol = protocol(&setup);
        let proof = proof(&setup, &protocol);

        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");
        let folded = protocol
            .folded_instance_from_proof(&proof, &mut transcript)
            .expect("fold inputs are well-shaped");

        let committed = protocol
            .committed_relaxed_instance(&proof.auxiliary_row_commitments)
            .expect("committed instance builds");
        let random = protocol
            .random_relaxed_instance(
                &proof.random_round_commitments,
                &proof.random_output_claim_row_commitments,
                &proof.random_auxiliary_row_commitments,
                &proof.random_error_row_commitments,
                &proof.random_eval_commitments,
                proof.random_u,
            )
            .expect("random instance builds");
        let mut manual_transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");
        committed.append_to_transcript(
            &mut manual_transcript,
            b"bf_committed_u",
            b"bf_committed_w",
            b"bf_committed_e",
            b"bf_committed_eval",
        );
        random.append_to_transcript(
            &mut manual_transcript,
            b"bf_random_u",
            b"bf_random_w",
            b"bf_random_e",
            b"bf_random_eval",
        );
        manual_transcript.append_values(b"bf_cross_e", &proof.cross_term_error_row_commitments);
        let folding_challenge = manual_transcript.challenge();
        let expected = committed
            .fold(
                &random,
                &proof.cross_term_error_row_commitments,
                folding_challenge,
            )
            .expect("fold dimensions match");

        assert_eq!(folded, expected);
        assert_eq!(transcript.state(), manual_transcript.state());
    }

    #[test]
    fn verify_rejects_random_round_count_mismatch() {
        let setup = setup();
        let protocol = coefficient_row_protocol();
        let mut proof = proof(&setup, &protocol);
        let _ = proof.random_round_commitments.pop();
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");

        let error = protocol
            .verify::<Pedersen<Bn254G1>, _>(&proof, &setup, &mut transcript)
            .expect_err("random rows are missing");

        assert!(matches!(
            error,
            VerificationError::Relaxed(RelaxedError::LengthMismatch {
                name: "random round commitments",
                ..
            })
        ));
    }

    #[test]
    fn verify_rejects_folded_eval_output_count_mismatch() {
        let setup = setup();
        let protocol = protocol(&setup);
        let mut proof = proof(&setup, &protocol);
        proof.folded_eval_outputs.push(f(7));
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");

        let error = protocol
            .verify::<Pedersen<Bn254G1>, _>(&proof, &setup, &mut transcript)
            .expect_err("folded eval count differs");

        assert_eq!(
            error.to_string(),
            "folded eval outputs length mismatch: expected 0, got 1"
        );
    }

    #[test]
    fn verify_accepts_folded_eval_commitment_opening() {
        let setup = setup();
        let protocol = protocol_with_eval(&setup);
        let proof = proof_with_valid_eval_opening(&setup, &protocol);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");
        let folded = protocol
            .folded_instance_from_proof(&proof, &mut transcript)
            .expect("folded instance builds");

        proof
            .verify_folded_eval_commitments::<Pedersen<Bn254G1>>(&setup, &folded)
            .expect("folded eval commitment opens");
    }

    #[test]
    fn verify_rejects_bad_folded_eval_commitment_opening() {
        let setup = setup();
        let protocol = protocol_with_eval(&setup);
        let mut proof = proof_with_valid_eval_opening(&setup, &protocol);
        proof.folded_eval_outputs[0] += f(1);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");

        let error = protocol
            .verify::<Pedersen<Bn254G1>, _>(&proof, &setup, &mut transcript)
            .expect_err("folded eval commitment opening is wrong");

        assert!(matches!(
            error,
            VerificationError::EvalCommitmentMismatch { index: 0 }
        ));
    }

    #[test]
    fn verify_rejects_outer_sumcheck_round_count_mismatch() {
        let setup = setup();
        let protocol = outer_round_protocol();
        let mut proof = proof(&setup, &protocol);
        let _ = proof.outer_sumcheck.round_polynomials.pop();
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");

        let error = protocol
            .verify::<Pedersen<Bn254G1>, _>(&proof, &setup, &mut transcript)
            .expect_err("outer sumcheck has wrong length");

        assert!(matches!(
            error,
            VerificationError::OuterSumcheck {
                source: jolt_sumcheck::SumcheckError::WrongNumberOfRounds { .. },
            }
        ));
    }

    #[test]
    fn verify_rejects_outer_sumcheck_degree_bound() {
        let setup = setup();
        let protocol = outer_round_protocol();
        let mut proof = proof(&setup, &protocol);
        proof.outer_sumcheck.round_polynomials[0] =
            CompressedPoly::new(vec![f(0), f(0), f(0), f(0)]);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");

        let error = protocol
            .verify::<Pedersen<Bn254G1>, _>(&proof, &setup, &mut transcript)
            .expect_err("outer sumcheck degree is too high");

        assert!(matches!(
            error,
            VerificationError::OuterSumcheck {
                source: jolt_sumcheck::SumcheckError::DegreeBoundExceeded { got: 4, max: 3 },
            }
        ));
    }

    #[test]
    fn verify_rejects_bad_error_opening() {
        let setup = setup();
        let protocol = outer_round_protocol();
        let mut proof = proof(&setup, &protocol);
        proof.error_opening.combined_blinding = f(1);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");

        let error = protocol
            .verify::<Pedersen<Bn254G1>, _>(&proof, &setup, &mut transcript)
            .expect_err("error opening is not binding to folded rows");

        assert!(matches!(
            error,
            VerificationError::VectorOpening(VectorOpeningError::CommitmentMismatch)
        ));
    }

    #[test]
    fn verify_rejects_outer_final_claim_mismatch() {
        let setup = setup();
        let protocol = outer_round_protocol();
        let mut proof = proof(&setup, &protocol);
        proof.az_rx = f(1);
        proof.bz_rx = f(1);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");

        let error = protocol
            .verify::<Pedersen<Bn254G1>, _>(&proof, &setup, &mut transcript)
            .expect_err("outer final claim does not match opened error row");

        assert!(matches!(
            error,
            VerificationError::OuterFinalClaimMismatch { .. }
        ));
    }

    #[test]
    fn verify_rejects_inner_sumcheck_round_count_mismatch() {
        let setup = setup();
        let protocol = inner_round_protocol();
        let proof = proof(&setup, &protocol);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");

        let error = protocol
            .verify::<Pedersen<Bn254G1>, _>(&proof, &setup, &mut transcript)
            .expect_err("inner sumcheck has wrong length");

        assert!(matches!(
            error,
            VerificationError::InnerSumcheck {
                source: jolt_sumcheck::SumcheckError::WrongNumberOfRounds {
                    expected: 1,
                    got: 0,
                },
            }
        ));
    }

    #[test]
    fn verify_rejects_bad_witness_opening() {
        let setup = setup();
        let protocol = inner_round_protocol();
        let mut proof = proof(&setup, &protocol);
        add_zero_inner_round(&mut proof);
        proof.witness_opening.combined_blinding = f(1);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");

        let error = protocol
            .verify::<Pedersen<Bn254G1>, _>(&proof, &setup, &mut transcript)
            .expect_err("witness opening is not binding to folded rows");

        assert!(matches!(
            error,
            VerificationError::VectorOpening(VectorOpeningError::CommitmentMismatch)
        ));
    }

    #[test]
    fn verify_rejects_inner_final_claim_mismatch() {
        let setup = setup();
        let protocol = inner_round_protocol();
        let mut proof = proof(&setup, &protocol);
        add_zero_inner_round(&mut proof);
        proof.auxiliary_row_commitments = vec![
            commit_value(&setup, f(5), f(50)),
            commit_value(&setup, f(5), f(50)),
        ];
        proof.witness_opening = VectorCommitmentOpening {
            combined_vector: vec![f(5)],
            combined_blinding: f(50),
        };
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");

        let error = protocol
            .verify::<Pedersen<Bn254G1>, _>(&proof, &setup, &mut transcript)
            .expect_err("inner final claim does not match opened witness row");

        assert!(matches!(
            error,
            VerificationError::InnerFinalClaimMismatch { .. }
        ));
    }
}
