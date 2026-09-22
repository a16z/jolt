//! Stage 5: the three-member batch (instruction read+RAF checking, RAM RA
//! claim reduction, registers value evaluation) — no uni-skip, every driver
//! generated.
//!
//! Pure orchestration mirroring `stage5::verify`: the one-hot formula
//! dimensions are built exactly as the verifier builds them, and the batch
//! inputs come from the verifier's own promoted `stage5_*_from_upstream`
//! wiring (stage 2's instruction claim-reduction triple and RAM openings,
//! stage 4's RAM val-check and registers-val openings — stage 3 does not
//! feed stage 5). The read+RAF member's typed relation data is the per-cycle
//! lookup rows, fetched by its kernel's `prepare` off the witness plane's
//! typed stage-5 rows accessor — never staged here.

use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckWitness;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage5::instruction_read_raf::InstructionReadRaf;
use jolt_verifier::stages::stage5::outputs::{
    Stage5ClearOutput, Stage5OutputClaims, Stage5Sumchecks,
};
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
use jolt_verifier::stages::stage5::{
    stage5_input_points_from_upstream, stage5_input_values_from_upstream,
};
use jolt_verifier::CheckedInputs;
use jolt_witness::JoltWitnessPlane;

use crate::recorder::ProofMode;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError, StageProver as _};

/// Stage 5's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier downstream stages consume.
pub struct Stage5ProverOutput<F: JoltField, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage5OutputClaims<F>,
    pub clear_output: Stage5ClearOutput<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 5 on `transcript` (positioned at the stage-4 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage5<F, PCS, VC, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut T,
) -> Result<Stage5ProverOutput<F, VC::Output>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let log_k = checked.ram_K.ilog2() as usize;
    let formula_dimensions = super::formula_dimensions(
        checked,
        config,
        preprocessing.verifier.program.bytecode_len(),
        JoltRelationId::InstructionReadRaf,
    )?;
    let trace_dimensions = formula_dimensions.trace;

    let sumchecks = Stage5Sumchecks {
        instruction_read_raf: InstructionReadRaf::new(formula_dimensions.instruction_read_raf),
        ram_ra_claim_reduction: RamRaClaimReduction::new(trace_dimensions, log_k),
        registers_val_evaluation: RegistersValEvaluation::new(trace_dimensions),
        #[cfg(feature = "field-inline")]
        field_registers_val_evaluation:
            jolt_verifier::stages::stage5::field_inline::val_evaluation_member(
                trace_dimensions.log_t(),
            ),
    };
    // Draws the instruction gamma, then the RAM gamma (registers draws
    // nothing, and so does the `field-inline` FR value-evaluation member) —
    // the generated declaration-order draw.
    let challenges = sumchecks.draw_challenges(transcript)?;

    let inputs = stage5_input_values_from_upstream(&stage2.output_values, &stage4.output_values);
    let input_points =
        stage5_input_points_from_upstream(&stage2.output_points, &stage4.output_points);

    let mut scheduler = backend.round_scheduler.build(session);
    let proved = sumchecks.prove(
        backend,
        session,
        &mut *scheduler,
        witness,
        &inputs,
        &input_points,
        &challenges,
        mode.recorder()?,
        transcript,
    )?;
    #[cfg(feature = "zk")]
    let (sumcheck_proof, committed_witness) = crate::recorder::split_recorded(proved.recorded)?;
    #[cfg(not(feature = "zk"))]
    let sumcheck_proof = proved.recorded.proof;

    let instruction_r_address = proved.output_points.instruction_r_address();
    Ok(Stage5ProverOutput {
        sumcheck_proof,
        claims: proved.output_claims.clone(),
        clear_output: Stage5ClearOutput {
            challenges,
            output_values: proved.output_claims,
            output_points: proved.output_points,
            instruction_r_address,
        },
        #[cfg(feature = "zk")]
        committed_witness,
    })
}

/// FR-on clear round-trips of the stage-5 recipe against the verifier's own
/// public constituents — `stage5::verify`'s clear body (the four-member batch
/// with the FR val-evaluation member, which draws no instance challenge) on a
/// twin transcript positioned by the stage-1..4 replays. The 32-byte
/// transcript-state equality pins the absorb order end to end. A second test
/// drives the FR val-evaluation kernel directly and ties both extracted
/// openings to direct MLE evaluations of the witness oracle's tables at the
/// bound point.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
    use jolt_claims::protocols::field_inline::relations::registers::FieldRegistersValEvaluationInputClaims;
    use jolt_claims::protocols::field_inline::{
        FieldInlineCommittedPolynomial, FieldInlinePolynomialId, FieldInlineVirtualPolynomial,
        FIELD_REGISTERS_LOG_K,
    };
    use jolt_claims::NoChallenges;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_kernels::ProverInputs;
    use jolt_poly::EqPolynomial;
    use jolt_transcript::{LegacyBlake2bTranscript as Blake2bTranscript, Transcript};
    use jolt_verifier::stages::relations::ConcreteSumcheck as _;
    use jolt_verifier::stages::stage5::field_inline as stage5_field_inline;
    use jolt_witness::JoltWitnessOracle as _;

    use super::*;
    use crate::recorder::ProofMode;
    use crate::stages::field_inline_fixtures::{
        fr_arithmetic_backend, fr_arithmetic_preprocessing, test_checked_inputs,
        test_prover_config, test_public_io, twins, LOG_T,
    };
    use crate::stages::stage1::prove_stage1;
    use crate::stages::stage2::prove_stage2;
    use crate::stages::stage3::prove_stage3;
    use crate::stages::stage4::prove_stage4;

    #[test]
    fn fr_arithmetic_stage5_round_trips_the_composed_verifier() {
        let witness = fr_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();
        let checked = test_checked_inputs();
        let preprocessing = fr_arithmetic_preprocessing();

        let mut prover_transcript = Blake2bTranscript::new(b"stage5-fr");
        let stage1 = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &backend,
            &mut session,
            &mode,
            LOG_T,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();
        let stage2 = prove_stage2::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &backend,
            &mut session,
            &mode,
            &config,
            &public_io,
            &stage1.clear_output,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();
        let stage3 = prove_stage3::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &backend,
            &mut session,
            &mode,
            &config,
            &stage1.clear_output,
            &stage2.clear_output,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();
        let stage4 = prove_stage4::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &backend,
            &mut session,
            &mode,
            &checked,
            &config,
            &preprocessing,
            &stage2.clear_output,
            &stage3.clear_output,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();
        let out = prove_stage5::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &backend,
            &mut session,
            &mode,
            &checked,
            &config,
            &preprocessing,
            &stage2.clear_output,
            &stage4.clear_output,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();

        // The verifier twin (stage5::verify's clear body), positioned by the
        // upstream replays.
        let mut transcript = Blake2bTranscript::new(b"stage5-fr");
        twins::replay_stage1(&mut transcript, &stage1);
        twins::replay_stage2(&mut transcript, &config, &public_io, &stage1, &stage2);
        twins::replay_stage3(&mut transcript, &stage1, &stage2, &stage3);
        twins::replay_stage4(
            &mut transcript,
            &config,
            &checked,
            &preprocessing,
            &stage2,
            &stage3,
            &stage4,
        );
        twins::replay_stage5(
            &mut transcript,
            &config,
            &checked,
            &preprocessing,
            &stage2,
            &stage4,
            &out,
        );

        assert_eq!(transcript.state(), prover_transcript.state());
    }

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// `Σ_i eq(point, i) · evals[i]` — the big-endian MLE the oracle grids and
    /// opening points share.
    fn mle(evals: &[Fr], point: &[Fr]) -> Fr {
        EqPolynomial::<Fr>::evals(point, None)
            .into_iter()
            .zip(evals)
            .map(|(eq, value)| eq * *value)
            .sum()
    }

    /// The FR val-evaluation kernel on the honest FR replay: every round
    /// message passes the engine's running-claim check starting from the
    /// relation's own input claim (the `FieldRegistersVal` MLE — honest FR
    /// register state makes the "value equals the sum of earlier increments"
    /// identity hold), and both extracted openings equal direct MLE
    /// evaluations of the witness oracle's tables at the derived
    /// `[address ‖ cycle]` opening point.
    #[test]
    fn fr_val_evaluation_kernel_outputs_match_direct_mle() {
        let witness = fr_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let oracle = witness.field_inline().unwrap();

        let relation = stage5_field_inline::val_evaluation_member::<Fr>(LOG_T);
        let upstream_point: Vec<Fr> = (0..(FIELD_REGISTERS_LOG_K + LOG_T) as u64)
            .map(|i| fr(100 + i))
            .collect();
        let table = |id: FieldInlinePolynomialId| oracle.oracle_table(id).unwrap();
        let registers_val_grid = table(FieldInlinePolynomialId::Virtual(
            FieldInlineVirtualPolynomial::FieldRegistersVal,
        ));
        let claims = FieldRegistersValEvaluationInputClaims::<Fr> {
            registers_val: mle(&registers_val_grid, &upstream_point),
        };
        let points = FieldRegistersValEvaluationInputClaims::<Vec<Fr>> {
            registers_val: upstream_point.clone(),
        };
        let challenges = NoChallenges::default();
        let mut kernel = backend
            .field_registers_val_evaluation
            .prepare(
                &mut session,
                &witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();

        let rounds = relation.rounds();
        let sumcheck_point: Vec<Fr> = (0..rounds as u64).map(|i| fr(200 + i)).collect();
        let mut previous_claim = relation.input_claim(&claims, &challenges).unwrap();
        for (round, challenge) in sumcheck_point.iter().enumerate() {
            let bind = (round > 0).then(|| sumcheck_point[round - 1]);
            let message = kernel.prove_round(bind, round, previous_claim).unwrap();
            previous_claim = message.evaluate(*challenge);
        }
        kernel
            .finish_rounds(*sumcheck_point.last().unwrap())
            .unwrap();

        let outputs = kernel.output_claims(&claims).unwrap();
        let output_points = relation
            .derive_opening_points(&sumcheck_point, &points)
            .unwrap();
        kernel
            .validate_derived_tables(&relation, &points, &output_points, &challenges)
            .unwrap();

        // Both openings share the `[upstream address ‖ own cycle]` point.
        let opening_point = output_points.rd_inc();
        let wa_grid = table(FieldInlinePolynomialId::Virtual(
            FieldInlineVirtualPolynomial::FieldRdWa,
        ));
        assert_eq!(outputs.rd_wa, mle(&wa_grid, opening_point));
        // `FieldRdInc` is cycle-only; its MLE at the joint point is its MLE at
        // the cycle sub-point (the address variables integrate out).
        let inc = table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ));
        let (_, cycle_sub_point) = opening_point.split_at(FIELD_REGISTERS_LOG_K);
        assert_eq!(outputs.rd_inc, mle(&inc, cycle_sub_point));

        let expected = relation
            .expected_output(&points, &outputs, &output_points, &challenges)
            .unwrap();
        assert_eq!(previous_claim, expected);
    }
}
