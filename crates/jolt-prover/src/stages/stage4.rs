//! Stage 4: the two-member batch (registers read/write checking, RAM value
//! check).
//!
//! Pure orchestration mirroring `stage4::verify`: the `Val_init`
//! decomposition (public initial-RAM evaluation + init structure) is built
//! with the verifier's own promoted helpers; the advice blocks' opening
//! VALUES are the prover-only work (the advice polynomial evaluated at each
//! block's address sub-point, staged transcript-silently before the RAM
//! value-check gamma draw). The stage's one curated behavior: the batch
//! carries `no_opening_values`, so the final absorbs use the claims struct's
//! hand-ordered `opening_values()` (staged advice/program-image openings
//! first, then registers, then RAM).

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::{JoltRelationId, TraceDimensions};
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
use jolt_poly::sparse_segments_mle_msb;
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckWitness;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage3::outputs::Stage3ClearOutput;
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
pub struct Stage4ProverOutput<F: JoltField, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage4OutputClaims<F>,
    pub clear_output: Stage4ClearOutput<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 4 on `transcript` (positioned at the stage-3 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage4<F, PCS, VC, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut T,
) -> Result<Stage4ProverOutput<F, VC::Output>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
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
    // The committed program-image contribution: the image words' block MLE at
    // the RAM address point (the public initial-RAM evaluation switched to
    // inputs-only above, so this staged opening carries the image's share).
    let program_image_contribution = init_structure
        .program_image_point
        .as_ref()
        .map(|point| {
            let layout = checked.precommitted.program_image.as_ref().ok_or(
                ProverError::InvariantViolation {
                    reason: "program-image init contribution without a committed layout",
                },
            )?;
            // The full program rides the witness plane (witness generation
            // requires it in every mode, including committed-program runs
            // whose PREPROCESSING retains only commitments).
            let program = witness.program_preprocessing();
            let value = sparse_segments_mle_msb(
                std::iter::once((
                    layout.start_index() as u128,
                    program.ram.bytecode_words.as_slice(),
                )),
                point,
            );
            Ok::<_, ProverError<F>>((point.clone(), value))
        })
        .transpose()?;
    // The advice blocks' opening values: each advice polynomial evaluated at
    // its block's address sub-point. Staged before the RAM value-check gamma
    // draw, exactly as legacy's `prover_accumulate_advice` — transcript-silent
    // on this branch (the claims flush with the stage-4 batch openings).
    let advice_contributions = init_structure
        .advice_blocks
        .iter()
        .map(|(kind, block)| {
            // Backend-neutral kernel-seam span at the call boundary — see
            // the taxonomy's kernel-seam contract.
            let opening_value =
                tracing::info_span!("AdviceOpeningEvaluation::evaluate", kind = ?kind).in_scope(
                    || {
                        backend.advice_opening.evaluate(
                            session,
                            *kind,
                            &block.opening_point,
                            witness,
                        )
                    },
                )?;
            Ok(VerifiedRamValCheckAdviceContribution {
                kind: *kind,
                selector: block.selector,
                opening_point: block.opening_point.clone(),
                opening_value,
            })
        })
        .collect::<Result<Vec<_>, ProverError<F>>>()?;
    let ram_val_check_init = RamValCheckInitialEvaluation {
        public_eval,
        program_image_contribution,
        advice_contributions,
    };

    let sumchecks = Stage4Sumchecks {
        registers_read_write: RegistersReadWriteChecking::new(register_dimensions),
        #[cfg(feature = "field-inline")]
        field_registers_read_write: jolt_verifier::stages::stage4::field_inline::read_write_member(
            log_t,
        ),
        ram_val_check: RamValCheck::new(trace_dimensions, log_k, init_structure.decomposition()),
    };
    // Draws the registers gamma, under `field-inline` the FR read-write gamma,
    // then the RAM value-check gamma behind its `b"ram_val_check_gamma"` domain
    // separator (replayed by the relation's `draw_challenges` override).
    let challenges = sumchecks.draw_challenges(transcript)?;

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

    // No curation hook: the staged advice/program-image openings ride in from
    // the RAM value-check kernel (captured off its own consumed input claims
    // at prepare), and the stage's `no_opening_values` absorb order is the
    // batch's hand-written `opening_values` replacement (staged openings
    // first, then registers, then RAM) — the driver's default curation.
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

    Ok(Stage4ProverOutput {
        sumcheck_proof,
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

/// FR-on clear round-trips of the stage-4 recipe against the verifier's own
/// public constituents — `stage4::verify`'s clear body step for step (the
/// `Val_init` decomposition, the three-member batch with the FR read-write
/// member, the generated absorb splicing the five FR openings) on a twin
/// transcript positioned by the stage-1..3 replays. The 32-byte
/// transcript-state equality pins the absorb order end to end. A second test
/// drives the FR read-write kernel directly and ties every extracted opening
/// to a direct MLE evaluation of the witness oracle's tables at the bound
/// point.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
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
    use jolt_transcript::{LegacyBlake2bTranscript as Blake2bTranscript, Transcript};
    use jolt_verifier::stages::relations::ConcreteSumcheck as _;
    use jolt_verifier::stages::stage4::field_inline as stage4_field_inline;
    use jolt_witness::JoltWitnessOracle as _;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        fr_arithmetic_backend, fr_arithmetic_preprocessing, test_checked_inputs,
        test_prover_config, test_public_io, twins, LOG_T,
    };
    use crate::stages::stage1::prove_stage1;
    use crate::stages::stage2::prove_stage2;
    use crate::stages::stage3::prove_stage3;

    #[test]
    fn fr_arithmetic_stage4_round_trips_the_composed_verifier() {
        let witness = fr_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();
        let checked = test_checked_inputs();
        let preprocessing = fr_arithmetic_preprocessing();

        let mut prover_transcript = Blake2bTranscript::new(b"stage4-fr");
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
        let out = prove_stage4::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
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

        // The verifier twin (stage4::verify's clear body, shared as the
        // stage-5+ twins' replay), positioned by the upstream replays.
        let mut transcript = Blake2bTranscript::new(b"stage4-fr");
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

    /// The FR read-write kernel on the honest FR replay: every round message
    /// passes the engine's running-claim check starting from the relation's
    /// own input claim, and the extracted openings equal direct MLE
    /// evaluations of the witness oracle's tables at the derived
    /// `[address ‖ cycle]` opening point.
    #[test]
    fn fr_read_write_kernel_outputs_match_direct_mle() {
        let witness = fr_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let oracle = witness.field_inline().unwrap();

        let relation = stage4_field_inline::read_write_member::<Fr>(LOG_T);
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

/// FR-on ZK: the stage-4 committed shell carries the curated row count — the
/// 5 ordinary register openings, the 5 spliced FR read-write openings, and
/// the 2 RAM value-check openings (no advice / program-image rows at the
/// fixture's scale).
#[cfg(all(test, feature = "field-inline", feature = "zk"))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_zk {
    use common::constants::MAX_BLINDFOLD_GENERATORS;
    use jolt_crypto::{Bn254G1, Pedersen, PedersenSetup};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        fr_arithmetic_backend, fr_arithmetic_preprocessing, test_checked_inputs,
        test_prover_config, test_public_io, LOG_T,
    };
    use crate::stages::stage1::prove_stage1;
    use crate::stages::stage2::prove_stage2;
    use crate::stages::stage3::prove_stage3;

    const CAPACITY: usize = MAX_BLINDFOLD_GENERATORS;

    #[test]
    fn committed_stage4_shell_carries_the_curated_rows() {
        let witness = fr_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let setup = PedersenSetup::new(vec![Bn254G1::default(); CAPACITY], Bn254G1::default());
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(Some(&setup)).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();
        let checked = test_checked_inputs();
        let preprocessing = fr_arithmetic_preprocessing();

        let mut transcript = Blake2bTranscript::new(b"stage4-fr-zk");
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
        let out = prove_stage4::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
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

        let values: Vec<Fr> = out
            .committed_witness
            .output_claim_rows
            .iter()
            .flatten()
            .copied()
            .collect();
        assert_eq!(values.len(), 12);
    }
}
