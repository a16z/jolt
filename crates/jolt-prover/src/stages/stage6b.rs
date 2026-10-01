//! Stage 6b: the cycle-phase batch — bytecode read+RAF and booleanity cycle
//! phases, RAM Hamming booleanity, both RA virtualizations, the increment
//! claim reduction, and the present precommitted claim-reduction cycle
//! phases (advice, committed bytecode, program image — head-aligned
//! members). A precommitted member whose schedule has active address-phase
//! rounds stages its intermediate claim here and — via the driver's uniform
//! post-extraction `park_residue` hook — parks its post-cycle bound state in
//! the proof session as plain data; stage 7's address-phase member reclaims
//! the carry.
//!
//! Pure orchestration mirroring `stage6b::verify`: the bytecode gamma is
//! carried from stage 6a's squeeze (no draw here), the post-6a draws and the
//! challenges aggregate come from the verifier's promoted `Stage6bDraws::draw`
//! and `cycle_challenges` helpers (the batch suppresses the generated draw),
//! the batch is built by the verifier's own promoted
//! `Stage6bSumchecks::build_from_parts` over the clear
//! carriers, and the driver's curation hook supplies
//! the verifier's promoted `stage6b_opening_values` — the curated order with
//! the runtime dedup of booleanity's `BytecodeRa` claims against the
//! bytecode read-RAF points (which fires when the bytecode address width is
//! a multiple of the committed chunk width).

#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckWitness;
use jolt_transcript::{ProverTranscript, Sponge};
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage3::outputs::Stage3ClearOutput;
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage5::outputs::Stage5ClearOutput;
use jolt_verifier::stages::stage6a::outputs::Stage6aClearOutput;
use jolt_verifier::stages::stage6b::batch::{Stage6bBuildParts, Stage6bDraws};
#[cfg(not(feature = "akita"))]
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::advice_reference_point_from_upstream;
use jolt_verifier::stages::stage6b::outputs::{
    Stage6bClearOutput, Stage6bOutputClaims, Stage6bSumchecks,
};
use jolt_verifier::stages::stage6b::{
    stage6b_input_points_from_upstream, stage6b_input_values_from_upstream,
};
use jolt_verifier::CheckedInputs;
use jolt_witness::JoltWitnessPlane;

use crate::recorder::ProofMode;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError, StageProver as _};

/// Stage 6b's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier stage 7 consumes. The precommitted reduction state
/// that spans into stage 7's address phase travels as `ProofSession` carries,
/// not output fields.
pub struct Stage6bProverOutput<F: JoltField> {
    pub claims: Stage6bOutputClaims<F>,
    pub clear_output: Stage6bClearOutput<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 6b on `transcript` (positioned at the stage-6a boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage6b<F, PCS, VC, H>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    stage6a: &Stage6aClearOutput<F>,
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut ProverTranscript<H>,
) -> Result<Stage6bProverOutput<F>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    let log_k = checked.ram_K.ilog2() as usize;
    let precommitted = &checked.precommitted;
    let formula_dimensions = super::formula_dimensions(
        checked,
        config,
        preprocessing.verifier.program.bytecode_len(),
        JoltRelationId::BytecodeReadRaf,
    )?;
    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let committed_program = precommitted.bytecode.is_some();

    // The bytecode gamma shares stage 6a's squeeze; the post-6a draws and the
    // challenges aggregate are the verifier's promoted two-front helpers.
    let carried = &stage6a.challenges;
    let draws = Stage6bDraws::draw(transcript, committed_program);

    // The batch, through the verifier's own promoted constructor over the
    // clear carriers. The full-program rows feed only the full-mode table
    // fold; they ride the witness plane (witness generation requires the
    // full program in every mode).
    let bytecode_table_rows = if committed_program {
        None
    } else {
        Some(witness.program_preprocessing().bytecode.bytecode.as_slice())
    };
    let entry_bytecode_index = preprocessing
        .verifier
        .program
        .entry_bytecode_index_checked(JoltRelationId::BytecodeReadRaf)?;
    let stage1_cycle_binding = stage1.cycle_binding_checked(JoltRelationId::BytecodeReadRaf)?;
    let sumchecks = Stage6bSumchecks::build_from_parts(Stage6bBuildParts {
        formula_dimensions: &formula_dimensions,
        ram_log_k: log_k,
        committed_chunk_bits: chunk_bits,
        precommitted,
        entry_bytecode_index,
        bytecode_table_rows,
        carried,
        eta: draws.eta,
        stage1_cycle_binding,
        stage2_points: &stage2.output_points,
        stage3_points: &stage3.output_points,
        stage4_points: &stage4.output_points,
        stage5_points: &stage5.output_points,
        stage6a_points: &stage6a.output_points,
        address_val_stages: stage6a.output_values.bytecode_read_raf.val_stages.clone(),
        #[cfg(not(feature = "akita"))]
        trusted_advice_reference_point: advice_reference_point_from_upstream(
            &stage4.ram_val_check_init,
            JoltAdviceKind::Trusted,
        ),
        #[cfg(not(feature = "akita"))]
        untrusted_advice_reference_point: advice_reference_point_from_upstream(
            &stage4.ram_val_check_init,
            JoltAdviceKind::Untrusted,
        ),
    })?;

    let cycle_challenges = sumchecks.cycle_challenges(carried, &draws);

    let inputs = stage6b_input_values_from_upstream(
        &sumchecks,
        &stage6a.output_values,
        &stage2.output_values,
        stage4,
        &stage5.output_values,
    )?;
    let input_points = stage6b_input_points_from_upstream(
        &sumchecks,
        &stage2.output_points,
        &stage4.output_points,
        &stage5.output_points,
    );

    // The committed-program weights: read back off the batch member (the
    // `build_from_parts` fold), for the clear carrier stage 7 consumes (the
    // bytecode reduction kernel reads them off its relation).
    let bytecode_weights = sumchecks
        .bytecode_reduction
        .as_ref()
        .map(|member| member.weights().clone());

    // The absorb order is the stage's curation override at its
    // `impl_stage_prover` invocation site (the promoted verifier helper's
    // canonical order, including the runtime booleanity-vs-bytecode point
    // dedup).
    let mut scheduler = backend.round_scheduler.build(session);
    let proved = sumchecks.prove(
        backend,
        session,
        &mut *scheduler,
        witness,
        &inputs,
        &input_points,
        &cycle_challenges,
        mode.recorder()?,
        transcript,
    )?;
    #[cfg(feature = "zk")]
    let committed_witness = proved.witness;

    Ok(Stage6bProverOutput {
        claims: proved.output_claims.clone(),
        clear_output: Stage6bClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
            bytecode_reduction_weights: bytecode_weights,
        },
        #[cfg(feature = "zk")]
        committed_witness,
    })
}

/// Clear round-trips with field-inline enabled of the stage-6b recipe through
/// the production stage-1..6b verifiers over the prover's argument string — on
/// both fixture profiles: the field-inactive ADDI trace (every field-inline
/// fold zero) and the field-active arithmetic trace (the composed bytecode
/// read-RAF kernels' field-inline stage-value legs carry real values). A
/// further test drives the field-inline increment-reduction kernel directly on
/// the field-inline-arithmetic replay and ties the extracted opening to a
/// direct MLE evaluation.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
    use crate::stages::field_inline_fixtures::proving::FixtureProver;
    use jolt_claims::protocols::field_inline::relations::claim_reductions::increments::{
        FieldRegistersIncClaimReductionChallenges, FieldRegistersIncClaimReductionInputClaims,
    };
    use jolt_claims::protocols::field_inline::{
        FieldInlineCommittedPolynomial, FieldInlinePolynomialId, FieldRegistersTraceDimensions,
    };
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_kernels::ProverInputs;
    use jolt_poly::EqPolynomial;
    use jolt_program::execution::OwnedTrace;
    use jolt_verifier::stages::relations::ConcreteSumcheck as _;
    use jolt_verifier::stages::stage6b::field_registers_inc_claim_reduction::FieldRegistersIncClaimReduction;
    use jolt_verifier::JoltSponge;
    use jolt_witness::{JoltWitnessOracle as _, TraceBackend};

    use super::*;
    use crate::stages::field_inline_fixtures::{
        addi_only_backend, addi_only_preprocessing, field_arithmetic_backend,
        field_arithmetic_preprocessing, fixture_transcript, test_checked_inputs,
        test_prover_config, test_public_io, verify_through, FixturePreprocessing, Through, LOG_T,
    };

    #[test]
    fn addi_only_stage6b_round_trips_the_composed_verifier() {
        stage6b_round_trips(addi_only_backend(), addi_only_preprocessing());
    }

    #[test]
    fn field_arithmetic_stage6b_round_trips_the_composed_verifier() {
        stage6b_round_trips(field_arithmetic_backend(), field_arithmetic_preprocessing());
    }

    fn stage6b_round_trips(
        trace_backend: TraceBackend<OwnedTrace>,
        preprocessing: FixturePreprocessing,
    ) {
        let witness = trace_backend.with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();
        let checked = test_checked_inputs();

        let mut prover_transcript = fixture_transcript();
        let ((((stage1, stage2, stage3), stage4), stage5), stage6a) = FixtureProver {
            backend: &backend,
            session: &mut session,
            mode: &mode,
            config: &config,
            public_io: &public_io,
            checked: &checked,
            preprocessing: &preprocessing,
            witness: &witness,
            transcript: &mut prover_transcript,
        }
        .through_stage6a();
        let _stage6b = prove_stage6b::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            &backend,
            &mut session,
            &mode,
            &checked,
            &config,
            &preprocessing,
            &stage1.clear_output,
            &stage2.clear_output,
            &stage3.clear_output,
            &stage4.clear_output,
            &stage5.clear_output,
            &stage6a.clear_output,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();

        verify_through(
            Through::Stage6b,
            &checked,
            &preprocessing,
            &prover_transcript,
        );
    }

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// `Σ_i eq(point, i) · evals[i]` — the big-endian MLE the oracle tables
    /// and opening points share.
    fn mle(evals: &[Fr], point: &[Fr]) -> Fr {
        EqPolynomial::<Fr>::evals(point, None)
            .into_iter()
            .zip(evals)
            .map(|(eq, value)| eq * *value)
            .sum()
    }

    /// The field-inline increment-reduction kernel on the honest field-inline replay: every round
    /// message passes the engine's running-claim check starting from the
    /// relation's own input claim (the two `FieldRdInc` MLEs folded by
    /// gamma), and the extracted reduced opening equals the direct MLE of the
    /// committed increment table at the reversed sumcheck point.
    #[test]
    fn field_register_inc_claim_reduction_kernel_output_matches_direct_mle() {
        let witness = field_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let oracle = witness.field_inline().unwrap();

        let read_write_cycle: Vec<Fr> = (0..LOG_T as u64).map(|i| fr(100 + i)).collect();
        let val_evaluation_cycle: Vec<Fr> = (0..LOG_T as u64).map(|i| fr(300 + i)).collect();
        let relation = FieldRegistersIncClaimReduction::<Fr>::new(
            FieldRegistersTraceDimensions::new(LOG_T),
            read_write_cycle.clone(),
            val_evaluation_cycle.clone(),
        );
        let inc = oracle
            .oracle_table(FieldInlinePolynomialId::Committed(
                FieldInlineCommittedPolynomial::FieldRdInc,
            ))
            .unwrap();
        let claims = FieldRegistersIncClaimReductionInputClaims::<Fr> {
            rd_inc_read_write: mle(&inc, &read_write_cycle),
            rd_inc_val_evaluation: mle(&inc, &val_evaluation_cycle),
        };
        let points = FieldRegistersIncClaimReductionInputClaims::<Vec<Fr>> {
            rd_inc_read_write: read_write_cycle,
            rd_inc_val_evaluation: val_evaluation_cycle,
        };
        let challenges = FieldRegistersIncClaimReductionChallenges { gamma: fr(7) };
        let mut kernel = backend
            .field_registers_inc_claim_reduction
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

        assert_eq!(outputs.rd_inc, mle(&inc, output_points.rd_inc()));

        let expected = relation
            .expected_output(&points, &outputs, &output_points, &challenges)
            .unwrap();
        assert_eq!(previous_claim, expected);
    }
}

/// ZK with field-inline enabled: the stage-6b committed witness carries the curated row count — the
/// alias-deduped cycle-point cell total, whose field-inline share is exactly the one
/// spliced reduced `FieldRdInc` row — and the production stage-1..6b zk
/// verifiers consume the prover's argument string.
#[cfg(all(test, feature = "field-inline", feature = "zk"))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_zk {
    use crate::stages::field_inline_fixtures::proving::FixtureProver;
    use jolt_claims::OutputClaims;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_verifier::JoltSponge;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        addi_only_backend, addi_only_preprocessing, fixture_transcript, test_checked_inputs,
        test_prover_config, test_public_io, test_vc_setup, verify_through, Through,
    };

    #[test]
    fn committed_stage6b_witness_carries_the_curated_rows_and_verifies() {
        let witness = addi_only_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let setup = test_vc_setup();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(Some(&setup)).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();
        let checked = test_checked_inputs();
        let preprocessing = addi_only_preprocessing();

        let mut transcript = fixture_transcript();
        let ((((stage1, stage2, stage3), stage4), stage5), stage6a) = FixtureProver {
            backend: &backend,
            session: &mut session,
            mode: &mode,
            config: &config,
            public_io: &public_io,
            checked: &checked,
            preprocessing: &preprocessing,
            witness: &witness,
            transcript: &mut transcript,
        }
        .through_stage6a();
        let out = prove_stage6b::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            &backend,
            &mut session,
            &mode,
            &checked,
            &config,
            &preprocessing,
            &stage1.clear_output,
            &stage2.clear_output,
            &stage3.clear_output,
            &stage4.clear_output,
            &stage5.clear_output,
            &stage6a.clear_output,
            &witness,
            &mut transcript,
        )
        .unwrap();

        // The committed row total is the verifier's expectation: the derived
        // output-point cell count minus the runtime booleanity-vs-bytecode
        // aliases. The field-inline reduction contributes exactly one of those cells —
        // its single reduced `FieldRdInc` opening.
        let values: Vec<Fr> = out
            .committed_witness
            .output_claim_rows
            .iter()
            .flatten()
            .copied()
            .collect();
        let cycle_points = &out.clear_output.output_points;
        let booleanity_point = cycle_points.booleanity_opening_point().unwrap().to_vec();
        let aliased = cycle_points
            .bytecode_read_raf
            .bytecode_ra
            .iter()
            .filter(|point| point.as_slice() == booleanity_point)
            .count();
        assert_eq!(
            values.len(),
            cycle_points.point_count().saturating_sub(aliased)
        );
        assert_eq!(
            OutputClaims::opening_values(&out.claims.field_registers_inc_claim_reduction).len(),
            1
        );

        verify_through(Through::Stage6b, &checked, &preprocessing, &transcript);
    }
}
