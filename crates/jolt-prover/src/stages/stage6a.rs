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

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::ComposedClaims;

use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckWitness;
use jolt_transcript::{Channel, ProverTranscript, Sponge};
use jolt_verifier::sites::STAGE6A;
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage3::outputs::Stage3ClearOutput;
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage5::outputs::Stage5ClearOutput;
use jolt_verifier::stages::stage6a::batch::Stage6aBuildParts;
use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhaseInputClaims;
use jolt_verifier::stages::stage6a::bytecode_read_raf::bytecode_read_raf_address_phase_input_values_from_upstream;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage6a::field_inline::field_inline_bytecode_read_raf_address_phase_input_values_from_upstream;
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
pub struct Stage6aProverOutput<F: JoltField> {
    pub claims: Stage6aOutputClaims<F>,
    pub clear_output: Stage6aClearOutput<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 6a on `transcript` (positioned at the stage-5 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage6a<F, PCS, VC, H>(
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
    transcript: &mut ProverTranscript<H>,
) -> Result<Stage6aProverOutput<F>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    transcript.site(STAGE6A);
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
    // The field-register access terms use their own upstream opening points.
    #[cfg(feature = "field-inline")]
    let sumchecks = jolt_verifier::stages::stage6a::field_inline::compose_bytecode_geometry(
        sumchecks,
        &stage4.output_points,
        &stage5.output_points,
    );
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
    #[cfg(feature = "field-inline")]
    let bytecode_input_values = ComposedClaims {
        base: bytecode_input_values,
        field_inline: field_inline_bytecode_read_raf_address_phase_input_values_from_upstream(
            &stage4.output_values,
            &stage5.output_values,
        ),
    };
    let inputs = Stage6aInputClaims {
        bytecode_read_raf: bytecode_input_values,
        booleanity: BooleanityAddressPhaseInputClaims::default(),
    };
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
    let committed_witness = proved.witness;

    Ok(Stage6aProverOutput {
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

/// Clear round-trips with field-inline enabled of the stage-6a recipe through
/// the production stage-1..6a verifiers over the prover's argument string, on
/// the field-active arithmetic trace: the appendage openings are nonzero, so
/// the address kernel's field-inline stage-value legs are exercised for real.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
    use crate::stages::field_inline_fixtures::proving::FixtureProver;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_verifier::JoltSponge;

    use super::*;
    use crate::recorder::ProofMode;
    use crate::stages::field_inline_fixtures::{
        field_arithmetic_backend, field_arithmetic_preprocessing, fixture_transcript,
        test_checked_inputs, test_prover_config, test_public_io, verify_through, Through,
    };

    #[test]
    fn field_arithmetic_stage6a_round_trips_the_composed_verifier() {
        let witness = field_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();
        let checked = test_checked_inputs();
        let preprocessing = field_arithmetic_preprocessing();

        let mut prover_transcript = fixture_transcript();
        let (((stage1, stage2, stage3), stage4), stage5) = FixtureProver {
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
        .through_stage5();
        let _stage6a = prove_stage6a::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
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

        // The field-active premise: the appendage the composed input claim folds
        // carries nonzero openings (the trace executes field-inline instructions), so
        // the round trip exercises the extension for real rather than the
        // zero-fold degenerate case.
        let appendage = field_inline_bytecode_read_raf_address_phase_input_values_from_upstream(
            &stage4.clear_output.output_values,
            &stage5.clear_output.output_values,
        );
        let zero = Fr::from_u64(0);
        assert!(appendage.rd_wa_read_write != zero);

        verify_through(
            Through::Stage6a,
            &checked,
            &preprocessing,
            &prover_transcript,
        );
    }
}
