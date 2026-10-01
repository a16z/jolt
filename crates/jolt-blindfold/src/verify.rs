use jolt_crypto::{HomomorphicCommitment, VectorCommitment, VectorCommitmentOpening};
use jolt_field::{CanonicalDecode, JoltField};
use jolt_poly::EqPolynomial;
use jolt_r1cs::{ConstraintMatrices, MatrixColumnContributions};
use jolt_sumcheck::{SumcheckClaim, SumcheckVerifier};
use jolt_transcript::{Channel, Sponge, VerifierTranscript};

use crate::prove::{INNER_SUMCHECK_DEGREE, OUTER_SUMCHECK_DEGREE};
use crate::wire::{receive_opening, FoldingCommitments};
use crate::{
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
        let sent =
            FoldingCommitments::receive(&self.dimensions, self.eval_commitments.len(), transcript)?;
        let committed = self.committed_relaxed_instance(&sent.auxiliary_rows)?;
        let random = self.random_relaxed_instance(
            &sent.random_rounds,
            &sent.random_output_claim_rows,
            &sent.random_auxiliary_rows,
            &sent.random_error_rows,
            &sent.random_evals,
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

        let [az_rx, bz_rx, cz_rx]: [F; 3] = [
            transcript.receive()?,
            transcript.receive()?,
            transcript.receive()?,
        ];
        let error_opening = receive_opening(error_row_len, transcript)?;

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
        let eval_count = folded.eval_commitments.len();
        let folded_eval_outputs: Vec<F> = transcript.receive_n(eval_count)?;
        let folded_eval_blindings: Vec<F> = transcript.receive_n(eval_count)?;
        for (index, ((commitment, &output), &blinding)) in folded
            .eval_commitments
            .iter()
            .zip(&folded_eval_outputs)
            .zip(&folded_eval_blindings)
            .enumerate()
        {
            if !VC::verify(vc_setup, commitment, &[output], &blinding) {
                return Err(VerificationError::EvalCommitmentMismatch { index });
            }
        }

        let coordinates = self.final_opening_witness_coordinates()?;
        ensure_len("final opening bindings", coordinates.len(), eval_count)?;
        let witness_row_len = self.dimensions.witness.row_len;
        for (index, (coordinates, (&output, &blinding))) in coordinates
            .iter()
            .zip(folded_eval_outputs.iter().zip(&folded_eval_blindings))
            .enumerate()
        {
            for (coordinate, expected, kind) in [
                (coordinates.evaluation, output, "output"),
                (coordinates.blinding, blinding, "blinding"),
            ] {
                let Some(coordinate) = coordinate else {
                    continue;
                };
                let opening = receive_opening(witness_row_len, transcript)?;
                let opened = coordinate.verify_opening::<F, VC>(vc_setup, folded, &opening)?;
                if opened != expected {
                    return Err(VerificationError::EvalWitnessMismatch { kind, index });
                }
                coordinate.require_dedicated_row(&opening, kind, index)?;
            }
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
#[expect(clippy::indexing_slicing, reason = "tests index fixture data")]
mod tests {
    use super::*;
    use crate::{
        r1cs::{FinalOpeningLayout, Layout},
        BlindFoldDimensions, RowDimensions, WitnessRowLayout,
    };
    use jolt_crypto::{
        Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment, VectorOpeningError,
    };
    use jolt_field::{Fr, Ring};
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

    /// Regression: a proof with fewer `folded_eval_outputs` than the layout's
    /// eval coordinates previously reached `folded_eval_outputs[index]` and
    /// panicked; both the eager length gate and the per-coordinate lookup
    /// must surface the same typed length error instead.
    #[test]
    fn verify_rejects_truncated_folded_eval_outputs_without_panicking() {
        let setup = setup();
        let protocol = protocol_with_eval(&setup);
        let mut proof = proof_with_valid_eval_opening(&setup, &protocol);
        let _ = proof.folded_eval_outputs.pop();
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-verify");

        let error = protocol
            .verify::<Pedersen<Bn254G1>, _>(&proof, &setup, &mut transcript)
            .expect_err("truncated folded eval outputs are rejected");

        assert!(matches!(
            error,
            VerificationError::Relaxed(RelaxedError::LengthMismatch {
                name: "folded eval outputs",
                ..
            })
        ));
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
