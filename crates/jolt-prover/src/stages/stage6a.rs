//! Stage 6a: the two-member address-phase batch (bytecode read+RAF address
//! phase, booleanity address phase).
//!
//! Pure orchestration mirroring `stage6a::verify`: the generated aggregate
//! draw (the bytecode member's six gammas, then the booleanity member's
//! override — reference address/cycle derived from the stage-5 instruction
//! point the relation carries, plus the pad draw and gamma) — which the 6a
//! VERIFIER never evaluates, but this prover's booleanity kernel consumes
//! immediately (masses, eq tables, gamma weights) off the challenge aggregate,
//! carried downstream in `Stage6aCarriedChallenges` for stage 6b. Both members
//! are universal
//! `PrepareKernel` slots: the bytecode member's stage-value fold reads the
//! witness plane's program view, and its PC pushforward source (the
//! per-cycle bytecode indices) comes off the witness plane's typed stage-6
//! rows — both fetched inside `prepare`, never staged here.

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
use jolt_verifier::stages::stage6a::batch::Stage6aBuildParts;
use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhaseInputClaims;
use jolt_verifier::stages::stage6a::bytecode_read_raf::bytecode_read_raf_address_phase_input_values_from_upstream;
use jolt_verifier::stages::stage6a::outputs::{
    Stage6aCarriedChallenges, Stage6aClearOutput, Stage6aInputClaims, Stage6aOutputClaims,
    Stage6aSumchecks,
};
use jolt_verifier::CheckedInputs;
use jolt_witness::JoltWitnessPlane;

use crate::recorder::ProofMode;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError, StageProver as _};

/// Stage 6a's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier stage 6b consumes.
pub struct Stage6aProverOutput<F: JoltField, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage6aOutputClaims<F>,
    pub clear_output: Stage6aClearOutput<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 6a on `transcript` (positioned at the stage-5 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage6a<F, PCS, VC, T>(
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
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut T,
) -> Result<Stage6aProverOutput<F, VC::Output>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let formula_dimensions = super::formula_dimensions(
        checked,
        config,
        preprocessing.verifier.program.bytecode_len(),
        JoltRelationId::BytecodeReadRaf,
    )?;

    // The batch, through the verifier's own promoted constructor: the relation
    // carries the upstream cycle/register points and the entry index (full
    // geometry at construction) — the kernel's read path. Committed-program
    // mode stages the five raw bound `Val_s` values as extra wire claims; the
    // sumcheck itself is unchanged.
    let stage1_cycle_binding = stage1.cycle_binding_checked(JoltRelationId::BytecodeReadRaf)?;
    let entry_bytecode_index = preprocessing
        .verifier
        .program
        .entry_bytecode_index_checked(JoltRelationId::BytecodeReadRaf)?;
    let sumchecks = Stage6aSumchecks::build_from_parts(Stage6aBuildParts {
        formula_dimensions: &formula_dimensions,
        committed_chunk_bits: config.one_hot_config.committed_chunk_bits(),
        committed_program: checked.precommitted.bytecode.is_some(),
        entry_bytecode_index,
        stage1_cycle_binding: &stage1_cycle_binding,
        stage2_points: &stage2.output_points,
        stage3_points: &stage3.output_points,
        stage4_points: &stage4.output_points,
        stage5_points: &stage5.output_points,
    })?;
    // The FR kernel geometry, attached exactly like the verifier: the
    // preprocessed side table (required fail-closed, like the stage-6b build)
    // plus the stage-4/5 FR opening points the address kernel's FR
    // stage-value legs fold over.
    #[cfg(feature = "field-inline")]
    jolt_verifier::stages::stage6a::field_inline::attach_bytecode_geometry(
        &sumchecks.bytecode_read_raf,
        jolt_verifier::stages::stage6a::field_inline::preprocessed_bytecode_table(
            &preprocessing.verifier.program,
        )?,
        &stage4.output_points,
        &stage5.output_points,
    )?;
    // The generated per-member draw, mirroring the verifier: the bytecode
    // member's six squeezes (the fold gamma plus the five per-stage gammas),
    // then the booleanity member's override (the reference-address pad draw
    // and the gamma). The 6a verifier only carries the booleanity values; this
    // prover's booleanity kernel consumes them off the challenge aggregate.
    let address_challenges = sumchecks.draw_challenges(transcript)?;
    let carried = Stage6aCarriedChallenges::from(&address_challenges);

    let input_points = sumchecks.empty_input_points();
    let bytecode_input_values = bytecode_read_raf_address_phase_input_values_from_upstream(
        &stage1.output_values,
        &stage2.output_values,
        &stage3.output_values,
        &stage4.output_values,
        &stage5.output_values,
    );
    // The packed build folds the four reduced `Inc` claims into the bytecode
    // address-phase input at the fused-inc consumer stage slots — the same
    // wrapper the verifier's `stage6a::verify` applies.
    #[cfg(feature = "akita")]
    let bytecode_input_values =
        jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafAddressPhaseInputClaims {
            base: bytecode_input_values,
            inc: jolt_verifier::stages::stage6b::inc_claim_reduction::inc_claim_reduction_input_values_from_upstream(
                &stage2.output_values,
                &stage4.output_values,
                &stage5.output_values,
            ),
        };
    let inputs = Stage6aInputClaims {
        bytecode_read_raf: bytecode_input_values,
        booleanity: BooleanityAddressPhaseInputClaims::default(),
    };
    // The FR appendage of the composed input claim: the stage-1 carrier's
    // FieldOpFlag openings plus the stage-4/5 FR access openings, folded by
    // the extended gamma powers inside the relation's composed `input_claim`.
    #[cfg(feature = "field-inline")]
    jolt_verifier::stages::stage6a::field_inline::attach_bytecode_inputs(
        &sumchecks.bytecode_read_raf,
        stage1,
        &stage4.output_values,
        &stage5.output_values,
    )?;

    let mut scheduler = backend.round_scheduler.build(session);
    let proved = sumchecks.prove(
        backend,
        session,
        &mut *scheduler,
        witness,
        &inputs,
        &input_points,
        &address_challenges,
        mode.recorder()?,
        transcript,
    )?;
    #[cfg(feature = "zk")]
    let (sumcheck_proof, committed_witness) = crate::recorder::split_recorded(proved.recorded)?;
    #[cfg(not(feature = "zk"))]
    let sumcheck_proof = proved.recorded.proof;

    Ok(Stage6aProverOutput {
        sumcheck_proof,
        claims: proved.output_claims.clone(),
        clear_output: Stage6aClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
            challenges: carried,
        },
        #[cfg(feature = "zk")]
        committed_witness,
    })
}

/// FR-on clear round-trips of the stage-6a recipe against the verifier's own
/// public constituents — `stage6a::verify`'s clear body (the batch built by
/// the promoted `build_from_parts` with the FR side table on the bytecode
/// member, the FR appendage attach, the composed input claim with its
/// gamma-power extension) on a twin transcript positioned by the stage-1..5
/// replays, on the FR-ACTIVE arithmetic trace: the appendage openings are
/// nonzero, so the address kernel's FR stage-value legs are exercised for
/// real (round 0's engine check pins the composed input claim to the
/// summand).
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_transcript::{LegacyBlake2bTranscript as Blake2bTranscript, Transcript};
    use jolt_verifier::stages::stage6a::field_inline as stage6a_field_inline;

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
    use crate::stages::stage5::prove_stage5;

    #[test]
    fn fr_arithmetic_stage6a_round_trips_the_composed_verifier() {
        let witness = fr_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();
        let checked = test_checked_inputs();
        let preprocessing = fr_arithmetic_preprocessing();

        let mut prover_transcript = Blake2bTranscript::new(b"stage6a-fr");
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
        let out = prove_stage6a::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
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

        // The FR-active premise: the appendage the composed input claim folds
        // carries nonzero openings (the trace executes FR instructions), so
        // the round trip exercises the extension for real rather than the
        // zero-fold degenerate case.
        let appendage = stage6a_field_inline::bytecode_read_raf_inputs(
            &stage1.clear_output,
            &stage4.clear_output.output_values,
            &stage5.clear_output.output_values,
        )
        .unwrap();
        let zero = Fr::from_u64(0);
        assert!(appendage.field_op_flags.iter().any(|flag| *flag != zero));
        assert!(appendage.rd_wa_read_write != zero);

        // The verifier twin (stage6a::verify's clear body), positioned by the
        // upstream replays.
        let mut transcript = Blake2bTranscript::new(b"stage6a-fr");
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
            &out,
        );

        assert_eq!(transcript.state(), prover_transcript.state());
    }
}
