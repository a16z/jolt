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
        wire::send_opening,
        BlindFoldDimensions, RowDimensions, WitnessRowLayout,
    };
    use jolt_crypto::{
        Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment, VectorOpeningError,
    };
    use jolt_field::{Fr, Ring};
    use jolt_r1cs::ConstraintMatrices;
    use jolt_transcript::{Blake2b512, ProtocolId, ProverTranscript, TranscriptError};

    type H = Blake2b512;

    const PROTOCOL: ProtocolId = ProtocolId::new::<H>("jolt-blindfold/verify-tests");
    const SESSION: &[u8] = b"blindfold-verify";

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

    fn outer_round_protocol() -> BlindFoldProtocol<Fr, Bn254G1> {
        let mut protocol = empty_protocol(Vec::new());
        protocol.dimensions.error = RowDimensions {
            row_len: 1,
            row_count: 2,
        };
        protocol
    }

    fn opening(row_len: usize) -> VectorCommitmentOpening<Fr> {
        VectorCommitmentOpening {
            combined_vector: vec![f(0); row_len],
            combined_blinding: f(0),
        }
    }

    fn num_vars(dimensions: RowDimensions) -> usize {
        (dimensions.row_count.trailing_zeros() + dimensions.row_len.trailing_zeros()) as usize
    }

    /// The prover messages of a BlindFold proof, written in the transcript
    /// order of `wire.rs` and `prove.rs`. Sumcheck rounds hold their
    /// compressed coefficients (every coefficient but the linear one).
    struct Messages {
        folding: FoldingCommitments<Fr, Bn254G1>,
        folded_eval_outputs: Vec<Fr>,
        folded_eval_blindings: Vec<Fr>,
        outer_rounds: Vec<Vec<Fr>>,
        abc: [Fr; 3],
        error_opening: VectorCommitmentOpening<Fr>,
        inner_rounds: Vec<Vec<Fr>>,
        witness_opening: VectorCommitmentOpening<Fr>,
    }

    impl Messages {
        fn zero(setup: &PedersenSetup<Bn254G1>, protocol: &BlindFoldProtocol<Fr, Bn254G1>) -> Self {
            let dimensions = &protocol.dimensions;
            let eval_count = protocol.eval_commitments.len();
            Self {
                folding: FoldingCommitments {
                    auxiliary_rows: vec![commitment(setup, 41); dimensions.auxiliary_rows],
                    random_u: f(3),
                    random_rounds: vec![identity(); dimensions.coefficient_rows],
                    random_output_claim_rows: vec![identity(); dimensions.output_claim_rows],
                    random_auxiliary_rows: vec![identity(); dimensions.auxiliary_rows],
                    random_error_rows: vec![identity(); dimensions.error.row_count],
                    random_evals: vec![commit_value(setup, f(11), f(110)); eval_count],
                    cross_term_error_rows: vec![identity(); dimensions.error.row_count],
                },
                folded_eval_outputs: vec![f(0); eval_count],
                folded_eval_blindings: vec![f(0); eval_count],
                outer_rounds: vec![vec![f(0); OUTER_SUMCHECK_DEGREE]; num_vars(dimensions.error)],
                abc: [f(0); 3],
                error_opening: opening(dimensions.error.row_len),
                inner_rounds: vec![vec![f(0); INNER_SUMCHECK_DEGREE]; num_vars(dimensions.witness)],
                witness_opening: opening(dimensions.witness.row_len),
            }
        }

        fn transcript() -> ProverTranscript<H> {
            ProverTranscript::new(&PROTOCOL, SESSION)
        }

        fn folding_challenge(&self) -> Fr {
            let mut transcript = Self::transcript();
            self.folding.send(&mut transcript);
            transcript.challenge_small()
        }

        fn with_valid_eval_opening(mut self) -> Self {
            let folding_challenge = self.folding_challenge();
            self.folded_eval_outputs = vec![f(7) + folding_challenge * f(11)];
            self.folded_eval_blindings = vec![f(70) + folding_challenge * f(110)];
            self
        }

        fn narg(&self) -> Vec<u8> {
            let mut transcript = Self::transcript();
            self.folding.send(&mut transcript);
            transcript.send_all(&self.folded_eval_outputs);
            transcript.send_all(&self.folded_eval_blindings);
            for round in &self.outer_rounds {
                transcript.send_all(round);
            }
            transcript.send_all(&self.abc);
            send_opening(&self.error_opening, &mut transcript);
            for round in &self.inner_rounds {
                transcript.send_all(round);
            }
            send_opening(&self.witness_opening, &mut transcript);
            transcript.finish()
        }
    }

    fn verify(
        protocol: &BlindFoldProtocol<Fr, Bn254G1>,
        setup: &PedersenSetup<Bn254G1>,
        narg: &[u8],
    ) -> Result<(), VerificationError<Fr>> {
        let mut transcript = VerifierTranscript::<H>::new(&PROTOCOL, SESSION, narg);
        protocol.verify::<Pedersen<Bn254G1>, H>(setup, &mut transcript)?;
        Ok(transcript.finish()?)
    }

    #[test]
    fn verify_rejects_degenerate_outer_sumcheck() {
        let setup = setup();
        let protocol = empty_protocol(Vec::new());
        let narg = Messages::zero(&setup, &protocol).narg();

        let error =
            verify(&protocol, &setup, &narg).expect_err("degenerate outer sumcheck is rejected");

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
        let protocol = inner_round_protocol();
        let messages = Messages::zero(&setup, &protocol);
        let narg = messages.narg();

        let mut transcript = VerifierTranscript::<H>::new(&PROTOCOL, SESSION, &narg);
        let folded = protocol
            .folded_instance(&mut transcript)
            .expect("fold inputs are well-shaped");

        let sent = &messages.folding;
        let committed = protocol
            .committed_relaxed_instance(&sent.auxiliary_rows)
            .expect("committed instance builds");
        let random = protocol
            .random_relaxed_instance(
                &sent.random_rounds,
                &sent.random_output_claim_rows,
                &sent.random_auxiliary_rows,
                &sent.random_error_rows,
                &sent.random_evals,
                sent.random_u,
            )
            .expect("random instance builds");
        let mut prover = Messages::transcript();
        sent.send(&mut prover);
        let folding_challenge: Fr = prover.challenge_small();
        let expected = committed
            .fold(&random, &sent.cross_term_error_rows, folding_challenge)
            .expect("fold dimensions match");

        assert_eq!(folded, expected);
        assert_eq!(
            transcript.challenge_bytes::<32>(),
            prover.challenge_bytes::<32>()
        );
    }

    #[test]
    fn verify_rejects_truncated_folded_eval_outputs() {
        let setup = setup();
        let protocol = protocol_with_eval(&setup);
        let messages = Messages::zero(&setup, &protocol).with_valid_eval_opening();
        let mut folding_only = Messages::transcript();
        messages.folding.send(&mut folding_only);
        let narg = folding_only.finish();

        let error = verify(&protocol, &setup, &narg).expect_err("truncated folded eval outputs");

        assert!(matches!(
            error,
            VerificationError::Transcript(TranscriptError::Truncated)
        ));
    }

    #[test]
    fn verify_accepts_folded_eval_commitment_opening() {
        let setup = setup();
        let protocol = protocol_with_eval(&setup);
        let narg = Messages::zero(&setup, &protocol)
            .with_valid_eval_opening()
            .narg();
        let mut transcript = VerifierTranscript::<H>::new(&PROTOCOL, SESSION, &narg);
        let folded = protocol
            .folded_instance(&mut transcript)
            .expect("folded instance builds");

        protocol
            .verify_folded_eval_witness_bindings::<Pedersen<Bn254G1>, H>(
                &setup,
                &folded,
                &mut transcript,
            )
            .expect("folded eval commitment opens");
    }

    #[test]
    fn verify_rejects_bad_folded_eval_commitment_opening() {
        let setup = setup();
        let protocol = protocol_with_eval(&setup);
        let mut messages = Messages::zero(&setup, &protocol).with_valid_eval_opening();
        messages.folded_eval_outputs[0] += f(1);

        let error = verify(&protocol, &setup, &messages.narg())
            .expect_err("folded eval commitment opening is wrong");

        assert!(matches!(
            error,
            VerificationError::EvalCommitmentMismatch { index: 0 }
        ));
    }

    #[test]
    fn verify_rejects_bad_error_opening() {
        let setup = setup();
        let protocol = outer_round_protocol();
        let mut messages = Messages::zero(&setup, &protocol);
        messages.error_opening.combined_blinding = f(1);

        let error = verify(&protocol, &setup, &messages.narg())
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
        let mut messages = Messages::zero(&setup, &protocol);
        messages.abc = [f(1), f(1), f(0)];

        let error = verify(&protocol, &setup, &messages.narg())
            .expect_err("outer final claim does not match opened error row");

        assert!(matches!(
            error,
            VerificationError::OuterFinalClaimMismatch { .. }
        ));
    }

    #[test]
    fn verify_rejects_bad_witness_opening() {
        let setup = setup();
        let protocol = inner_round_protocol();
        let mut messages = Messages::zero(&setup, &protocol);
        messages.witness_opening.combined_blinding = f(1);

        let error = verify(&protocol, &setup, &messages.narg())
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
        let mut messages = Messages::zero(&setup, &protocol);
        messages.folding.auxiliary_rows = vec![
            commit_value(&setup, f(5), f(50)),
            commit_value(&setup, f(5), f(50)),
        ];
        messages.witness_opening = VectorCommitmentOpening {
            combined_vector: vec![f(5)],
            combined_blinding: f(50),
        };

        let error = verify(&protocol, &setup, &messages.narg())
            .expect_err("inner final claim does not match opened witness row");

        assert!(matches!(
            error,
            VerificationError::InnerFinalClaimMismatch { .. }
        ));
    }
}
