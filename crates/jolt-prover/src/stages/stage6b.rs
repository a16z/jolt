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
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
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
pub struct Stage6bProverOutput<F: JoltField, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage6bOutputClaims<F>,
    pub clear_output: Stage6bClearOutput<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 6b on `transcript` (positioned at the stage-6a boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage6b<F, PCS, VC, T>(
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
    transcript: &mut T,
) -> Result<Stage6bProverOutput<F, VC::Output>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
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
    // The FR bytecode side table, required fail-closed exactly like the
    // verifier's `Stage6bSumchecks::build` (committed-program preprocessing
    // cannot supply it, and neither can a full program preprocessed without FR
    // support).
    #[cfg(feature = "field-inline")]
    let field_inline_bytecode =
        jolt_verifier::stages::stage6b::field_inline::preprocessed_bytecode_table(
            &preprocessing.verifier.program,
        )?;
    let sumchecks = Stage6bSumchecks::build_from_parts(Stage6bBuildParts {
        formula_dimensions: &formula_dimensions,
        ram_log_k: log_k,
        committed_chunk_bits: chunk_bits,
        precommitted,
        entry_bytecode_index,
        bytecode_table_rows,
        #[cfg(feature = "field-inline")]
        field_inline_bytecode,
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
    let (sumcheck_proof, committed_witness) = crate::recorder::split_recorded(proved.recorded)?;
    #[cfg(not(feature = "zk"))]
    let sumcheck_proof = proved.recorded.proof;

    Ok(Stage6bProverOutput {
        sumcheck_proof,
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

/// FR-on clear round-trips of the stage-6b recipe against the verifier's own
/// public constituents — `stage6b::verify`'s clear body (the post-6a draws,
/// the batch with the FR increment-reduction member built by the promoted
/// `build_from_parts`, the curated `stage6b_opening_values` absorb with the
/// spliced reduced `FieldRdInc`) on a twin transcript positioned by the
/// stage-1..6a replays — on both fixture profiles: the FR-inactive ADDI
/// trace (every FR fold zero) and the FR-ACTIVE arithmetic trace (the
/// composed bytecode read-RAF kernels' FR stage-value legs carry real
/// values). A further test drives the FR increment-reduction kernel directly
/// on the FR-arithmetic replay and ties the extracted opening to a direct
/// MLE evaluation.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
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
    use jolt_transcript::{LegacyBlake2bTranscript as Blake2bTranscript, Transcript};
    use jolt_verifier::stages::relations::ConcreteSumcheck as _;
    use jolt_verifier::stages::stage6b::field_inline as stage6b_field_inline;
    use jolt_verifier::stages::stage6b::field_registers_inc_claim_reduction::FieldRegistersIncClaimReduction;
    use jolt_witness::{JoltWitnessOracle as _, TraceBackend};

    use super::*;
    use crate::stages::field_inline_fixtures::twins::FixturePreprocessing;
    use crate::stages::field_inline_fixtures::{
        addi_only_backend, addi_only_preprocessing, fr_arithmetic_backend,
        fr_arithmetic_preprocessing, test_checked_inputs, test_prover_config, test_public_io,
        twins, LOG_T, RAM_LOG_K,
    };
    use crate::stages::stage1::prove_stage1;
    use crate::stages::stage2::prove_stage2;
    use crate::stages::stage3::prove_stage3;
    use crate::stages::stage4::prove_stage4;
    use crate::stages::stage5::prove_stage5;
    use crate::stages::stage6a::prove_stage6a;

    #[test]
    fn addi_only_stage6b_round_trips_the_composed_verifier() {
        stage6b_round_trips(
            addi_only_backend(),
            addi_only_preprocessing(),
            b"stage6b-fr",
        );
    }

    #[test]
    fn fr_arithmetic_stage6b_round_trips_the_composed_verifier() {
        stage6b_round_trips(
            fr_arithmetic_backend(),
            fr_arithmetic_preprocessing(),
            b"stage6b-fr-active",
        );
    }

    fn stage6b_round_trips(
        trace_backend: TraceBackend<OwnedTrace>,
        preprocessing: FixturePreprocessing,
        label: &'static [u8],
    ) {
        let witness = trace_backend.with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();
        let checked = test_checked_inputs();

        let mut prover_transcript = Blake2bTranscript::new(label);
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
        let stage5 = prove_stage5::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
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
        let stage6a = prove_stage6a::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
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
            &witness,
            &mut prover_transcript,
        )
        .unwrap();
        let out = prove_stage6b::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
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

        // The verifier twin (stage6b::verify's clear body), positioned by
        // the upstream replays. The private wire-shape validator is
        // transcript-free and elided; `verify_clear`'s hard checks and the
        // final state equality pin the protocol content.
        let mut transcript = Blake2bTranscript::new(label);
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
            &stage5,
        );
        twins::replay_stage6a(
            &mut transcript,
            &config,
            &checked,
            &preprocessing,
            &stage1,
            &stage2,
            &stage3,
            &stage4,
            &stage5,
            &stage6a,
        );

        let formula_dimensions = crate::stages::formula_dimensions(
            &checked,
            &config,
            preprocessing.verifier.program.bytecode_len(),
            JoltRelationId::BytecodeReadRaf,
        )
        .unwrap();
        let carried = &stage6a.clear_output.challenges;
        let draws = Stage6bDraws::draw(&mut transcript, false);
        let program = preprocessing.program().unwrap();
        let field_inline_bytecode =
            stage6b_field_inline::preprocessed_bytecode_table(&preprocessing.verifier.program)
                .unwrap();
        let entry_bytecode_index = preprocessing
            .verifier
            .program
            .entry_bytecode_index_checked(JoltRelationId::BytecodeReadRaf)
            .unwrap();
        let stage1_cycle_binding = stage1
            .clear_output
            .cycle_binding_checked(JoltRelationId::BytecodeReadRaf)
            .unwrap();
        let sumchecks = Stage6bSumchecks::build_from_parts(Stage6bBuildParts {
            formula_dimensions: &formula_dimensions,
            ram_log_k: RAM_LOG_K,
            committed_chunk_bits: config.one_hot_config.committed_chunk_bits(),
            precommitted: &checked.precommitted,
            entry_bytecode_index,
            bytecode_table_rows: Some(program.bytecode.bytecode.as_slice()),
            field_inline_bytecode,
            carried,
            eta: draws.eta,
            stage1_cycle_binding,
            stage2_points: &stage2.clear_output.output_points,
            stage3_points: &stage3.clear_output.output_points,
            stage4_points: &stage4.clear_output.output_points,
            stage5_points: &stage5.clear_output.output_points,
            stage6a_points: &stage6a.clear_output.output_points,
            address_val_stages: stage6a
                .clear_output
                .output_values
                .bytecode_read_raf
                .val_stages
                .clone(),
            #[cfg(not(feature = "akita"))]
            trusted_advice_reference_point: advice_reference_point_from_upstream(
                &stage4.clear_output.ram_val_check_init,
                JoltAdviceKind::Trusted,
            ),
            #[cfg(not(feature = "akita"))]
            untrusted_advice_reference_point: advice_reference_point_from_upstream(
                &stage4.clear_output.ram_val_check_init,
                JoltAdviceKind::Untrusted,
            ),
        })
        .unwrap();
        let cycle_challenges = sumchecks.cycle_challenges(carried, &draws);
        let input_values = stage6b_input_values_from_upstream(
            &sumchecks,
            &stage6a.clear_output.output_values,
            &stage2.clear_output.output_values,
            &stage4.clear_output,
            &stage5.clear_output.output_values,
        )
        .unwrap();
        let input_points = stage6b_input_points_from_upstream(
            &sumchecks,
            &stage2.clear_output.output_points,
            &stage4.clear_output.output_points,
            &stage5.clear_output.output_points,
        );
        let cycle_points = sumchecks
            .verify_clear(
                &input_values,
                &input_points,
                &cycle_challenges,
                &out.claims,
                &out.sumcheck_proof,
                &mut transcript,
                6,
            )
            .unwrap();
        let booleanity_point = cycle_points.booleanity_opening_point().unwrap().to_vec();
        // The verifier's absorb: the curated order with the runtime
        // booleanity-vs-bytecode dedup (single-sourced with the prover's
        // curation through `stage6b_opening_values`).
        for value in jolt_verifier::stages::stage6b::stage6b_opening_values(
            &out.claims,
            &cycle_points.bytecode_read_raf.bytecode_ra,
            &booleanity_point,
        ) {
            transcript.append_labeled(b"opening_claim", &value);
        }

        assert_eq!(transcript.state(), prover_transcript.state());
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

    /// The FR increment-reduction kernel on the honest FR replay: every round
    /// message passes the engine's running-claim check starting from the
    /// relation's own input claim (the two `FieldRdInc` MLEs folded by
    /// gamma), and the extracted reduced opening equals the direct MLE of the
    /// committed increment table at the reversed sumcheck point.
    #[test]
    fn fr_inc_claim_reduction_kernel_output_matches_direct_mle() {
        let witness = fr_arithmetic_backend().with_field_inline().unwrap();
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

/// FR-on ZK: the stage-6b committed shell carries the curated row count — the
/// alias-deduped cycle-point cell total, whose FR share is exactly the one
/// spliced reduced `FieldRdInc` row.
#[cfg(all(test, feature = "field-inline", feature = "zk"))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_zk {
    use common::constants::MAX_BLINDFOLD_GENERATORS;
    use jolt_claims::OutputClaims;
    use jolt_crypto::{Bn254G1, Pedersen, PedersenSetup};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        addi_only_backend, addi_only_preprocessing, test_checked_inputs, test_prover_config,
        test_public_io, LOG_T,
    };
    use crate::stages::stage1::prove_stage1;
    use crate::stages::stage2::prove_stage2;
    use crate::stages::stage3::prove_stage3;
    use crate::stages::stage4::prove_stage4;
    use crate::stages::stage5::prove_stage5;
    use crate::stages::stage6a::prove_stage6a;

    const CAPACITY: usize = MAX_BLINDFOLD_GENERATORS;

    #[test]
    fn committed_stage6b_shell_carries_the_curated_rows() {
        let witness = addi_only_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let setup = PedersenSetup::new(vec![Bn254G1::default(); CAPACITY], Bn254G1::default());
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(Some(&setup)).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();
        let checked = test_checked_inputs();
        let preprocessing = addi_only_preprocessing();

        let mut transcript = Blake2bTranscript::new(b"stage6b-fr-zk");
        let stage1 = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &backend,
            &mut session,
            &mode,
            LOG_T,
            &witness,
            &mut transcript,
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
            &mut transcript,
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
            &mut transcript,
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
            &mut transcript,
        )
        .unwrap();
        let stage5 = prove_stage5::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &backend,
            &mut session,
            &mode,
            &checked,
            &config,
            &preprocessing,
            &stage2.clear_output,
            &stage4.clear_output,
            &witness,
            &mut transcript,
        )
        .unwrap();
        let stage6a = prove_stage6a::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
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
            &witness,
            &mut transcript,
        )
        .unwrap();
        let out = prove_stage6b::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
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
        // aliases. The FR reduction contributes exactly one of those cells —
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
    }
}
