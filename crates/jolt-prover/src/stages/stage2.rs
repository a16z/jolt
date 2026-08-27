//! Stage 2: the Spartan product uni-skip round and the batch (RAM read-write
//! checking, product remainder, instruction claim reduction, under
//! `field-inline` the FR claim reduction, RAM RAF evaluation, RAM output
//! check).
//!
//! Pure orchestration: the challenge draws, batch head, point derivation,
//! final-claim fold, and absorb order are `jolt-verifier`'s generated drivers
//! plus the same hand-coded choreography its `stage2::verify` performs (the
//! `τ_high` draw, the uni-skip); all compute is behind the backend's stage-2
//! slots.

use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
use jolt_claims::protocols::jolt::geometry::spartan::SpartanProductDimensions;
use jolt_claims::protocols::jolt::{JoltRelationId, TraceDimensions};
use jolt_claims::NoChallenges;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::PublicIoMemory;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckWitness;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_verifier::stages::stage2::outputs::{
    Stage2BatchSumchecks, Stage2ClearOutput, Stage2OutputClaims,
};
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_verifier::stages::stage2::product_uniskip::{
    product_uniskip_input_values_from_stage1, ProductUniskip,
};
use jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck;
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_verifier::stages::stage2::{product_tau_low, stage2_batch_input_values_from_upstream};
use jolt_verifier::stages::uniskip::draw_spartan_product_tau_high;
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;

use crate::recorder::ProofMode;
use crate::{ProverConfig, ProverError, StageProver as _};

/// Stage 2's outputs: the two wire proofs, the wire claims, and the
/// verifier-typed cross-stage carrier downstream stages consume.
pub struct Stage2ProverOutput<F: JoltField, C> {
    pub uniskip_proof: SumcheckProof<F, C>,
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage2OutputClaims<F>,
    pub clear_output: Stage2ClearOutput<F>,
    #[cfg(feature = "zk")]
    pub uniskip_witness: CommittedSumcheckWitness<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 2 on `transcript` (positioned at the stage-1 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage2<F, PCS, VC, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    config: &ProverConfig,
    public_io: &JoltDevice,
    stage1: &Stage1ClearOutput<F>,
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut T,
) -> Result<Stage2ProverOutput<F, VC::Output>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let log_t = config.trace_length.ilog2() as usize;
    let log_k = config.ram_K.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let read_write_dimensions = config.rw_config.ram_dimensions(log_t, log_k);
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let raf_dimensions =
        RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRafEvaluation,
                reason: error.to_string(),
            }
        })?;

    let tau_low = product_tau_low(&stage1.remainder_point(), log_t)?;

    // Backend-neutral kernel-seam spans at the call boundary, so every
    // `UniskipKernel` implementation inherits them — see the taxonomy's
    // kernel-seam contract.
    tracing::info_span!("SpartanProductUniskip::prepare").in_scope(|| {
        backend
            .spartan_product_uniskip
            .prepare(session, log_t, &tau_low, witness)
    })?;

    let tau_high: F = draw_spartan_product_tau_high(transcript);
    let uniskip_relation = ProductUniskip::new(product_dimensions, tau_high);
    // The FR lane inputs enter the composed input claim exactly as on the
    // verifier — through the shared seam attach, before `input_claim`.
    #[cfg(feature = "field-inline")]
    jolt_verifier::stages::stage2::field_inline::attach_uniskip_inputs(&uniskip_relation, stage1)?;
    let uniskip_inputs = product_uniskip_input_values_from_stage1(stage1);
    let uniskip_input_claim =
        uniskip_relation.input_claim(&uniskip_inputs, &NoChallenges::default())?;
    let uniskip_poly =
        tracing::info_span!("SpartanProductUniskip::first_round_poly").in_scope(|| {
            backend
                .spartan_product_uniskip
                .first_round_poly(session, &[tau_high])
        })?;
    // The COMPOSED jolt-r1cs uni-skip shape (feature-aware): identical to the
    // jolt-claims RV64-only constants FR-off, the FR-extended lane domain
    // under `field-inline` — the shape the verifier's stage-2 uni-skip checks.
    let proved_uniskip = mode.prove_uniskip(
        uniskip_poly,
        uniskip_input_claim,
        SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        transcript,
    )?;
    let uniskip_challenge = proved_uniskip.challenge;

    // The generated stage drivers, on the verifier's own batch type.
    let lowest_address = public_io.memory_layout.get_lowest_address();
    let public_memory = PublicIoMemory::new(public_io).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamOutputCheck,
            reason: error.to_string(),
        }
    })?;
    let sumchecks = Stage2BatchSumchecks {
        ram_read_write: RamReadWriteChecking::new(read_write_dimensions, log_k, tau_low.clone()),
        product_remainder: ProductRemainder::new(
            product_dimensions,
            uniskip_challenge,
            tau_high,
            tau_low.clone(),
        ),
        instruction_claim_reduction: InstructionClaimReduction::new(
            trace_dimensions,
            tau_low.clone(),
        ),
        #[cfg(feature = "field-inline")]
        field_registers_claim_reduction:
            jolt_verifier::stages::stage2::field_inline::claim_reduction_member(
                log_t,
                tau_low.clone(),
            ),
        ram_raf_evaluation: RamRafEvaluation::new(
            read_write_dimensions,
            raf_dimensions,
            log_k,
            lowest_address,
            tau_low.clone(),
        ),
        ram_output_check: RamOutputCheck::new(read_write_dimensions, public_memory),
    };
    // Both batch gammas, then the RAM output-check address reference point (the
    // last member's `draw_challenges` override) — the verifier's exact schedule.
    let challenges = sumchecks.draw_challenges(transcript)?;

    let input_points = sumchecks.empty_input_points();
    // Under `field-inline` the FR claim-reduction inputs wire from the
    // stage-1 FR carrier (fail-closed when absent) through the same shared
    // assembly the verifier runs.
    let inputs = stage2_batch_input_values_from_upstream(stage1, proved_uniskip.output_claim)?;

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

    #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
    let mut claims =
        Stage2OutputClaims::new(proved_uniskip.output_claim, proved.output_claims.clone());
    // Attach the FR product appendage the composed remainder kernel
    // published: `claims` is the wire carrier `stage2::verify` requires
    // fail-closed on FR-on proofs.
    #[cfg(feature = "field-inline")]
    {
        claims.field_inline_product = Some(
            sumchecks
                .product_remainder
                .field_inline_outputs()
                .ok_or(ProverError::Verifier(VerifierError::MissingProofPayload {
                    field: "stage2 FR product appendage (composed remainder kernel)",
                }))?
                .clone(),
        );
    }

    Ok(Stage2ProverOutput {
        uniskip_proof: proved_uniskip.proof,
        sumcheck_proof,
        claims,
        clear_output: Stage2ClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
            product_tau_low: tau_low,
        },
        #[cfg(feature = "zk")]
        uniskip_witness: proved_uniskip.witness,
        #[cfg(feature = "zk")]
        committed_witness,
    })
}

/// FR-on clear round-trips of the stage-2 recipe against the verifier's own
/// public constituents — `stage2::verify`'s clear body step for step (the
/// `τ_high` draw, the composed uni-skip via the seam attach and
/// `uniskip::verify_clear`, the six-member batch with the FR claim-reduction
/// member, the FR product appendage attach with its alias equality, and the
/// curated absorb) on a twin transcript. The full `stage2::verify` entrypoint
/// needs an assembled `JoltProof` (no test constructor for the joint-opening
/// slot), so this is the closest public seam; the 32-byte transcript-state
/// equality pins the absorb order end to end.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_verifier::stages::stage2::field_inline as stage2_field_inline;
    use jolt_verifier::stages::uniskip;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        fr_arithmetic_backend, test_prover_config, test_public_io, LOG_T,
    };
    use crate::stages::stage1::prove_stage1;

    #[test]
    fn fr_arithmetic_stage2_round_trips_the_composed_verifier() {
        let witness = fr_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();

        let mut prover_transcript = Blake2bTranscript::new(b"stage2-fr");
        let stage1 = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &backend,
            &mut session,
            &mode,
            LOG_T,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();
        let out = prove_stage2::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
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

        // The FR product appendage is carried, and the spec's alias table
        // holds on honest data: the FR claim-reduction member outputs equal
        // the appendage values polynomial-for-polynomial.
        let appendage = out.claims.field_inline_product.clone().unwrap();
        let reduction = &out.claims.batch_outputs.field_registers_claim_reduction;
        assert_eq!(reduction.rs1_value, appendage.rs1_value);
        assert_eq!(reduction.rs2_value, appendage.rs2_value);
        assert_eq!(reduction.rd_value, appendage.rd_value);

        // The verifier twin (stage2::verify's clear body).
        let mut transcript = Blake2bTranscript::new(b"stage2-fr");
        {
            // Stage 1's twin, to position the transcript at the stage-2
            // boundary (already round-tripped by stage 1's own tests).
            let tau =
                jolt_verifier::stages::uniskip::draw_spartan_outer_tau(&mut transcript, LOG_T);
            let uniskip_challenge = uniskip::verify_clear(
                &stage1.uniskip_proof,
                &uniskip::UniskipParams::spartan_outer(),
                Fr::from_u64(0),
                stage1.claims.uniskip_output_claim,
                &mut transcript,
            )
            .unwrap();
            let sumchecks =
                jolt_verifier::stages::stage1::outputs::Stage1BatchSumchecks {
                    outer_remainder:
                        jolt_verifier::stages::stage1::outer_remainder::OuterRemainder::new(
                            jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions::rv64(LOG_T),
                            tau,
                            uniskip_challenge,
                        ),
                };
            let batch_challenges = sumchecks.draw_challenges(&mut transcript).unwrap();
            let input_points = sumchecks.empty_input_points();
            let attached = jolt_verifier::stages::stage1::field_inline::attach_outer_outputs(
                &sumchecks,
                &stage1.claims,
            )
            .unwrap();
            let input_values = jolt_verifier::stages::stage1::outputs::Stage1BatchInputClaims {
                outer_remainder: jolt_verifier::stages::stage1::outer_remainder::outer_remainder_input_values_from_uniskip_output(
                    stage1.claims.uniskip_output_claim,
                ),
            };
            let _stage1_points = sumchecks
                .verify_clear(
                    &input_values,
                    &input_points,
                    &batch_challenges,
                    &stage1.claims.outer,
                    &stage1.sumcheck_proof,
                    &mut transcript,
                    1,
                )
                .unwrap();
            sumchecks.append_output_claims(&mut transcript, &stage1.claims.outer);
            jolt_verifier::stages::stage1::field_inline::append_outer_openings(
                &mut transcript,
                &attached,
            );
        }

        // Stage 2 proper.
        let log_t = LOG_T;
        let log_k = config.ram_K.ilog2() as usize;
        let trace_dimensions = TraceDimensions::new(log_t);
        let read_write_dimensions = config.rw_config.ram_dimensions(log_t, log_k);
        let product_dimensions = SpartanProductDimensions::new(log_t);
        let raf_dimensions = RamRafEvaluationDimensions::try_from(read_write_dimensions).unwrap();
        let tau_low = product_tau_low(&stage1.clear_output.remainder_point(), log_t).unwrap();

        let tau_high: Fr = draw_spartan_product_tau_high(&mut transcript);
        let uniskip_relation = ProductUniskip::new(product_dimensions, tau_high);
        stage2_field_inline::attach_uniskip_inputs(&uniskip_relation, &stage1.clear_output)
            .unwrap();
        let uniskip_inputs = product_uniskip_input_values_from_stage1(&stage1.clear_output);
        let uniskip_input_claim = uniskip_relation
            .input_claim(&uniskip_inputs, &NoChallenges::default())
            .unwrap();
        let uniskip_challenge = uniskip::verify_clear(
            &out.uniskip_proof,
            &uniskip::UniskipParams::spartan_product(),
            uniskip_input_claim,
            out.claims.product_uniskip_output_claim,
            &mut transcript,
        )
        .unwrap();

        let lowest_address = public_io.memory_layout.get_lowest_address();
        let public_memory = PublicIoMemory::new(&public_io).unwrap();
        let sumchecks = Stage2BatchSumchecks {
            ram_read_write: RamReadWriteChecking::new(
                read_write_dimensions,
                log_k,
                tau_low.clone(),
            ),
            product_remainder: ProductRemainder::new(
                product_dimensions,
                uniskip_challenge,
                tau_high,
                tau_low.clone(),
            ),
            instruction_claim_reduction: InstructionClaimReduction::new(
                trace_dimensions,
                tau_low.clone(),
            ),
            field_registers_claim_reduction: stage2_field_inline::claim_reduction_member(
                log_t,
                tau_low.clone(),
            ),
            ram_raf_evaluation: RamRafEvaluation::new(
                read_write_dimensions,
                raf_dimensions,
                log_k,
                lowest_address,
                tau_low.clone(),
            ),
            ram_output_check: RamOutputCheck::new(read_write_dimensions, public_memory),
        };
        let challenges = sumchecks.draw_challenges(&mut transcript).unwrap();
        let input_points = sumchecks.empty_input_points();
        sumchecks
            .validate_output_claims(&out.claims.batch_outputs)
            .unwrap();
        let attached_product =
            stage2_field_inline::attach_product_outputs(&sumchecks, &out.claims).unwrap();
        let input_values = stage2_batch_input_values_from_upstream(
            &stage1.clear_output,
            out.claims.product_uniskip_output_claim,
        )
        .unwrap();
        let _stage2_points = sumchecks
            .verify_clear(
                &input_values,
                &input_points,
                &challenges,
                &out.claims.batch_outputs,
                &out.sumcheck_proof,
                &mut transcript,
                2,
            )
            .unwrap();
        sumchecks.append_output_claims(
            &mut transcript,
            &out.claims.batch_outputs,
            &attached_product,
        );

        assert_eq!(transcript.state(), prover_transcript.state());
    }
}

/// FR-on ZK: the curated committed stage-2 shell (21 rows: 18 member
/// openings with the FR claim reduction, plus the 3 FR product-appendage
/// rows at the verifier's splice), the honest equality witness behind the
/// BlindFold `OpeningEquality` lowering, and the verifier replay landing on
/// the prover's forward transcript bytes.
#[cfg(all(test, feature = "field-inline", feature = "zk"))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_zk {
    use jolt_crypto::{Bn254G1, Pedersen, PedersenSetup};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_verifier::stages::stage2::field_inline as stage2_field_inline;
    use jolt_verifier::stages::uniskip;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        fr_arithmetic_backend, test_prover_config, test_public_io, LOG_T,
    };
    use crate::stages::stage1::prove_stage1;

    const CAPACITY: usize = common::constants::MAX_BLINDFOLD_GENERATORS;

    #[test]
    fn committed_stage2_shell_carries_the_curated_rows_and_replays() {
        let witness = fr_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let setup = PedersenSetup::new(vec![Bn254G1::default(); CAPACITY], Bn254G1::default());
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(Some(&setup)).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();

        let mut prover_transcript = Blake2bTranscript::new(b"stage2-fr-zk");
        let stage1 = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &backend,
            &mut session,
            &mode,
            LOG_T,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();
        let out = prove_stage2::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
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

        // 21 curated committed output-claim values: RAM read-write (3),
        // product remainder (8), the FR product appendage (3, the splice),
        // instruction claim reduction (2 non-aliased), the FR claim
        // reduction (3), RAM RAF (1), RAM output check (1).
        let values: Vec<Fr> = out
            .committed_witness
            .output_claim_rows
            .iter()
            .flatten()
            .copied()
            .collect();
        assert_eq!(values.len(), 21);

        // The BlindFold `OpeningEquality` witness is honestly satisfied: the
        // committed rows at the FR claim-reduction member positions equal the
        // spliced appendage rows for the same polynomial (rd/rs1/rs2 member
        // order vs rs1/rs2/rd appendage order).
        assert_eq!(values[16], values[13]); // rd_value
        assert_eq!(values[17], values[11]); // rs1_value
        assert_eq!(values[18], values[12]); // rs2_value

        // The replay (stage2::verify's zk body over its public constituents),
        // mirroring blindfold.rs's transcript hard check at stage scope.
        let checked = jolt_verifier::CheckedInputs {
            public_io: JoltDevice::default(),
            zk: true,
            trace_length: 1 << LOG_T,
            ram_K: 1 << 4,
            entry_address: crate::stages::field_inline_fixtures::ENTRY,
            preprocessing_digest: [0u8; 32],
            trusted_advice_commitment_present: false,
            vc_capacity: Some(CAPACITY),
            precommitted: jolt_verifier::stages::PrecommittedSchedule {
                trusted_advice: None,
                untrusted_advice: None,
                bytecode: None,
                program_image: None,
            },
        };
        let mut transcript = Blake2bTranscript::new(b"stage2-fr-zk");
        {
            // Stage 1's zk twin, to position the transcript.
            let tau = uniskip::draw_spartan_outer_tau(&mut transcript, LOG_T);
            let uniskip_step = uniskip::verify_zk(
                &checked,
                &stage1.uniskip_proof,
                &uniskip::UniskipParams::spartan_outer(),
                &mut transcript,
            )
            .unwrap();
            let sumchecks = jolt_verifier::stages::stage1::outputs::Stage1BatchSumchecks {
                outer_remainder:
                    jolt_verifier::stages::stage1::outer_remainder::OuterRemainder::new(
                        jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions::rv64(LOG_T),
                        tau,
                        uniskip_step.challenge,
                    ),
            };
            let _stage1_consistency = sumchecks
                .verify_zk(&stage1.sumcheck_proof, &mut transcript)
                .unwrap();
        }

        let log_t = LOG_T;
        let log_k = config.ram_K.ilog2() as usize;
        let read_write_dimensions = config.rw_config.ram_dimensions(log_t, log_k);
        let product_dimensions = SpartanProductDimensions::new(log_t);
        let raf_dimensions = RamRafEvaluationDimensions::try_from(read_write_dimensions).unwrap();
        let tau_low = product_tau_low(&stage1.clear_output.remainder_point(), log_t).unwrap();
        let tau_high: Fr = draw_spartan_product_tau_high(&mut transcript);
        let product_uniskip_step = uniskip::verify_zk(
            &checked,
            &out.uniskip_proof,
            &uniskip::UniskipParams::spartan_product(),
            &mut transcript,
        )
        .unwrap();

        let lowest_address = public_io.memory_layout.get_lowest_address();
        let public_memory = PublicIoMemory::new(&public_io).unwrap();
        let sumchecks = Stage2BatchSumchecks {
            ram_read_write: RamReadWriteChecking::new(
                read_write_dimensions,
                log_k,
                tau_low.clone(),
            ),
            product_remainder: ProductRemainder::new(
                product_dimensions,
                product_uniskip_step.challenge,
                tau_high,
                tau_low.clone(),
            ),
            instruction_claim_reduction: InstructionClaimReduction::new(
                TraceDimensions::new(log_t),
                tau_low.clone(),
            ),
            field_registers_claim_reduction: stage2_field_inline::claim_reduction_member(
                log_t,
                tau_low.clone(),
            ),
            ram_raf_evaluation: RamRafEvaluation::new(
                read_write_dimensions,
                raf_dimensions,
                log_k,
                lowest_address,
                tau_low.clone(),
            ),
            ram_output_check: RamOutputCheck::new(read_write_dimensions, public_memory),
        };
        let _challenges = sumchecks.draw_challenges(&mut transcript).unwrap();
        let _stage2_consistency = sumchecks
            .verify_zk(&out.sumcheck_proof, &mut transcript)
            .unwrap();

        assert_eq!(transcript.state(), prover_transcript.state());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FR-off, the composed jolt-r1cs product uni-skip constants equal the
    /// jolt-claims RV64-only constants this recipe previously passed — the
    /// swap is byte-neutral.
    #[cfg(not(feature = "field-inline"))]
    #[test]
    fn product_uniskip_constants_match_the_rv64_only_values() {
        use jolt_claims::protocols::jolt::geometry::dimensions::{
            PRODUCT_UNISKIP_DOMAIN_SIZE, PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
        };

        assert_eq!(
            SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
            PRODUCT_UNISKIP_DOMAIN_SIZE
        );
        assert_eq!(
            SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
            PRODUCT_UNISKIP_FIRST_ROUND_DEGREE
        );
    }

    /// FR-on, the composed product domain carries the two FR lanes — the
    /// spec's 5-point domain and its degree-12 first round.
    #[cfg(feature = "field-inline")]
    #[test]
    fn product_uniskip_constants_are_the_composed_fr_domains() {
        assert_eq!(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, 5);
        assert_eq!(SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE, 12);
    }
}
