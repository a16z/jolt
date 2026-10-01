//! Stage 4: the two-member batch (registers read/write checking, RAM value
//! check).
//!
//! Pure orchestration mirroring `stage4::verify`: the `Val_init`
//! decomposition (public initial-RAM evaluation + init structure) is built
//! with the verifier's own promoted helpers; the private opening VALUES
//! are evaluated through the backend as one batch (program image and advice,
//! staged transcript-silently before the RAM
//! value-check gamma draw). The stage's one curated behavior: a clear proof
//! sends the staged advice/program-image openings after the gamma draws and
//! the register and RAM openings after the rounds, while a committed proof
//! commits all of them in the claims struct's canonical order.

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::{JoltRelationId, TraceDimensions};
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_kernels::opening::RamInitialOpening;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckWitness;
use jolt_transcript::{Channel, ProverTranscript, Sponge};
#[cfg(feature = "field-inline")]
use jolt_verifier::config::JOLT_VERIFIER_CONFIG;
use jolt_verifier::sites::STAGE4;
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage3::outputs::Stage3ClearOutput;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage4::field_registers_read_write_checking::FieldRegistersReadWriteChecking;
use jolt_verifier::stages::stage4::outputs::{
    Stage4ClearOutput, Stage4OutputClaims, Stage4Sumchecks,
};
use jolt_verifier::stages::stage4::ram_val_check::RamValCheck;
use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
use jolt_verifier::stages::stage4::{
    public_initial_ram_evaluation, ram_val_check_init_structure, stage4_input_points_from_upstream,
    stage4_input_values_from_upstream, RamValCheckInitialEvaluation,
    VerifiedRamValCheckAdviceContribution,
};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::JoltWitnessPlane;

use crate::recorder::ProofMode;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError, StageProver as _};

/// Stage 4's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier downstream stages consume.
pub struct Stage4ProverOutput<F: JoltField> {
    pub claims: Stage4OutputClaims<F>,
    pub clear_output: Stage4ClearOutput<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 4 on `transcript` (positioned at the stage-3 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage4<F, PCS, VC, H>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut ProverTranscript<H>,
) -> Result<Stage4ProverOutput<F>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    transcript.site(STAGE4);
    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let register_dimensions = config
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);

    // The RAM points, validated exactly as the verifier does.
    let ram_read_write_opening_point = stage2.output_points.ram_read_write_point();
    let ram_output_check_opening_point = stage2.output_points.ram_output_check_point();
    if ram_read_write_opening_point.len() != log_k + log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: format!(
                "RAM read-write opening point length mismatch: expected {}, got {}",
                log_k + log_t,
                ram_read_write_opening_point.len()
            ),
        }
        .into());
    }
    let (r_address, _r_cycle_ram) = ram_read_write_opening_point.split_at(log_k);
    if ram_output_check_opening_point != r_address {
        return Err(ProverError::InvariantViolation {
            reason: "stage-2 RAM val and val_final opening points disagree",
        });
    }

    let public_eval = public_initial_ram_evaluation(checked, &preprocessing.verifier, r_address)?;
    // The prover-side untrusted-advice presence signal (the verifier reads the
    // proof's commitment slot).
    let untrusted_advice_present = !checked.public_io.untrusted_advice.is_empty();
    let init_structure =
        ram_val_check_init_structure(checked, untrusted_advice_present, r_address, public_eval)?;
    // Submit all private contributions together so device backends can share
    // one batch. Only scalar values cross this seam; geometry and transcript
    // ordering stay with this coordinator.
    let mut openings = Vec::new();
    if let Some(point) = init_structure.program_image_point.as_ref() {
        let layout =
            checked
                .precommitted
                .program_image
                .as_ref()
                .ok_or(ProverError::InvariantViolation {
                    reason: "program-image init contribution without a committed layout",
                })?;
        openings.push(RamInitialOpening::ProgramImage { layout, point });
    }
    openings.extend(init_structure.advice_blocks.iter().map(|(kind, block)| {
        RamInitialOpening::Advice {
            kind: *kind,
            point: &block.opening_point,
        }
    }));
    let values = if openings.is_empty() {
        Vec::new()
    } else {
        tracing::info_span!("RamInitialOpeningEvaluation::evaluate").in_scope(|| {
            backend
                .ram_initial_openings
                .evaluate(session, &openings, witness)
        })?
    };
    if values.len() != openings.len() {
        return Err(ProverError::InvariantViolation {
            reason: "initial RAM opening count does not match the requests",
        });
    }
    let mut values = values.into_iter();
    let program_image_contribution = init_structure
        .program_image_point
        .as_ref()
        .map(|point| {
            let value = values.next().ok_or(ProverError::InvariantViolation {
                reason: "missing program-image initial RAM opening",
            })?;
            Ok::<_, ProverError<F>>((point.clone(), value))
        })
        .transpose()?;
    let advice_contributions = init_structure
        .advice_blocks
        .iter()
        .zip(values)
        .map(
            |((kind, block), opening_value)| VerifiedRamValCheckAdviceContribution {
                kind: *kind,
                selector: block.selector,
                opening_point: block.opening_point.clone(),
                opening_value,
            },
        )
        .collect();
    let ram_val_check_init = RamValCheckInitialEvaluation {
        public_eval,
        program_image_contribution,
        advice_contributions,
    };

    let sumchecks = Stage4Sumchecks {
        registers_read_write: RegistersReadWriteChecking::new(register_dimensions),
        #[cfg(feature = "field-inline")]
        field_registers_read_write: FieldRegistersReadWriteChecking::new(
            JOLT_VERIFIER_CONFIG
                .field_inline
                .read_write_dimensions(log_t),
        ),
        ram_val_check: RamValCheck::new(trace_dimensions, log_k, init_structure.decomposition()),
    };
    // Draws the registers gamma, under `field-inline` the field-register read-write gamma,
    // then the RAM value-check gamma behind its `b"ram_val_check_gamma"` domain
    // separator (replayed by the relation's `draw_challenges` override).
    let challenges = sumchecks.draw_challenges(transcript)?;
    // The RAM value-check input claim consumes the staged openings, so a clear
    // proof sends them before the batch; a committed proof carries them in its
    // output-claim rows instead.
    #[cfg(not(feature = "zk"))]
    transcript.send_all(&ram_val_check_init.staged_openings().values());

    let inputs = stage4_input_values_from_upstream(
        &stage2.output_values,
        &stage3.output_values,
        &ram_val_check_init,
    );
    let input_points = stage4_input_points_from_upstream(
        &stage2.output_points,
        &stage3.output_points,
        &init_structure,
    );

    // The staged advice/program-image openings ride in from the RAM
    // value-check kernel (captured off its own consumed input claims at
    // prepare). The driver's curation sends the post-round openings in a clear
    // build and commits the full canonical order in a ZK build.
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
    let committed_witness = proved.witness;

    Ok(Stage4ProverOutput {
        claims: proved.output_claims.clone(),
        clear_output: Stage4ClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
            ram_val_check_init,
        },
        #[cfg(feature = "zk")]
        committed_witness,
    })
}

/// Clear round-trips with field-inline enabled of the stage-4 recipe through
/// the production stage-1..4 verifiers over the prover's argument string
/// (including the RAM value-check staged openings sent after the gamma draws).
/// A second test drives the field-register read-write kernel directly and ties
/// every extracted opening to a direct MLE evaluation of the witness oracle's
/// tables at the bound point.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
    use crate::stages::field_inline_fixtures::proving::FixtureProver;
    use jolt_claims::protocols::field_inline::relations::registers::{
        FieldRegistersReadWriteChallenges, FieldRegistersReadWriteInputClaims,
    };
    use jolt_claims::protocols::field_inline::{
        FieldInlineCommittedPolynomial, FieldInlinePolynomialId, FieldInlineVirtualPolynomial,
    };
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_kernels::ProverInputs;
    use jolt_poly::EqPolynomial;
    use jolt_verifier::stages::relations::ConcreteSumcheck as _;
    use jolt_verifier::JoltSponge;
    use jolt_witness::JoltWitnessOracle as _;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        field_arithmetic_backend, field_arithmetic_preprocessing, fixture_transcript,
        test_checked_inputs, test_prover_config, test_public_io, verify_through, Through, LOG_T,
    };

    #[test]
    fn field_arithmetic_stage4_round_trips_the_composed_verifier() {
        let witness = field_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();
        let checked = test_checked_inputs();
        let preprocessing = field_arithmetic_preprocessing();

        let mut prover_transcript = fixture_transcript();
        let (_stage1, stage2, stage3) = FixtureProver {
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
        .through_stage3();
        let _stage4 = prove_stage4::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
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

        verify_through(
            Through::Stage4,
            &checked,
            &preprocessing,
            &prover_transcript,
        );
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

    /// The field-register read-write kernel on the honest field-inline replay: every round message
    /// passes the engine's running-claim check starting from the relation's
    /// own input claim, and the extracted openings equal direct MLE
    /// evaluations of the witness oracle's tables at the derived
    /// `[address ‖ cycle]` opening point.
    #[test]
    fn field_register_read_write_kernel_outputs_match_direct_mle() {
        let witness = field_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let oracle = witness.field_inline().unwrap();

        let relation = FieldRegistersReadWriteChecking::<Fr>::new(
            JOLT_VERIFIER_CONFIG
                .field_inline
                .read_write_dimensions(LOG_T),
        );
        let r_cycle: Vec<Fr> = (0..LOG_T as u64).map(|i| fr(100 + i)).collect();
        let table = |id: FieldInlinePolynomialId| oracle.oracle_table(id).unwrap();
        let cycle_table = |polynomial: FieldInlineVirtualPolynomial| {
            table(FieldInlinePolynomialId::Virtual(polynomial))
        };
        let claims = FieldRegistersReadWriteInputClaims::<Fr> {
            rd_value: mle(
                &cycle_table(FieldInlineVirtualPolynomial::FieldRdValue),
                &r_cycle,
            ),
            rs1_value: mle(
                &cycle_table(FieldInlineVirtualPolynomial::FieldRs1Value),
                &r_cycle,
            ),
            rs2_value: mle(
                &cycle_table(FieldInlineVirtualPolynomial::FieldRs2Value),
                &r_cycle,
            ),
        };
        let points = FieldRegistersReadWriteInputClaims::<Vec<Fr>> {
            rd_value: r_cycle.clone(),
            rs1_value: r_cycle.clone(),
            rs2_value: r_cycle.clone(),
        };
        let challenges = FieldRegistersReadWriteChallenges { gamma: fr(7) };
        let mut kernel = backend
            .field_registers_read_write
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

        // The engine's round loop: bind the previous draw, check the running
        // claim, reduce through the returned round polynomial.
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

        // Every opening shares the `[address ‖ cycle]` point; each extracted
        // value must be the direct MLE of its oracle table there.
        let opening_point = output_points.registers_val();
        let grid = |polynomial| cycle_table(polynomial);
        assert_eq!(
            outputs.registers_val,
            mle(
                &grid(FieldInlineVirtualPolynomial::FieldRegistersVal),
                opening_point
            )
        );
        assert_eq!(
            outputs.rs1_ra,
            mle(
                &grid(FieldInlineVirtualPolynomial::FieldRs1Ra),
                opening_point
            )
        );
        assert_eq!(
            outputs.rs2_ra,
            mle(
                &grid(FieldInlineVirtualPolynomial::FieldRs2Ra),
                opening_point
            )
        );
        assert_eq!(
            outputs.rd_wa,
            mle(
                &grid(FieldInlineVirtualPolynomial::FieldRdWa),
                opening_point
            )
        );
        // `FieldRdInc` is cycle-only; its MLE at the joint point is its MLE at
        // the cycle sub-point (the address variables integrate out).
        let inc = table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ));
        let (_, cycle_sub_point) = opening_point.split_at(opening_point.len() - LOG_T);
        assert_eq!(outputs.rd_inc, mle(&inc, cycle_sub_point));

        // The relation's own output fold closes the loop: the final running
        // claim equals `expected_output` at the extracted claims.
        let expected = relation
            .expected_output(&points, &outputs, &output_points, &challenges)
            .unwrap();
        assert_eq!(previous_claim, expected);
    }
}

/// ZK with field-inline enabled: the stage-4 committed witness carries the curated row count — the
/// 5 ordinary register openings, the 5 spliced field-register read-write openings, and
/// the 2 RAM value-check openings (no advice / program-image rows at the
/// fixture's scale) — and the production stage-1..4 zk verifiers consume the
/// prover's argument string.
#[cfg(all(test, feature = "field-inline", feature = "zk"))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_zk {
    use crate::stages::field_inline_fixtures::proving::FixtureProver;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_verifier::JoltSponge;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        field_arithmetic_backend, field_arithmetic_preprocessing, fixture_transcript,
        test_checked_inputs, test_prover_config, test_public_io, test_vc_setup, verify_through,
        Through,
    };

    #[test]
    fn committed_stage4_witness_carries_the_curated_rows_and_verifies() {
        let witness = field_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let setup = test_vc_setup();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(Some(&setup)).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();
        let checked = test_checked_inputs();
        let preprocessing = field_arithmetic_preprocessing();

        let mut transcript = fixture_transcript();
        let (_stage1, stage2, stage3) = FixtureProver {
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
        .through_stage3();
        let out = prove_stage4::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
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

        let value_count: usize = out
            .committed_witness
            .output_claim_rows
            .iter()
            .map(Vec::len)
            .sum();
        assert_eq!(value_count, 12);

        verify_through(Through::Stage4, &checked, &preprocessing, &transcript);
    }
}
