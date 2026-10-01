//! Stage 2: the Spartan product uni-skip round and the batch (RAM read-write
//! checking, product remainder, instruction claim reduction, under
//! `field-inline` the field-inline claim reduction, RAM RAF evaluation, RAM output
//! check).
//!
//! Pure orchestration: the challenge draws, batch head, point derivation,
//! final-claim fold, and absorb order are `jolt-verifier`'s generated drivers
//! plus the same hand-coded choreography its `stage2::verify` performs (the
//! `τ_high` draw, the uni-skip); all compute is behind the backend's stage-2
//! slots.

use common::jolt_device::JoltDevice;
use jolt_claims::protocols::composed::geometry::{
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldRegistersTraceDimensions;
use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
use jolt_claims::protocols::jolt::geometry::spartan::SpartanProductDimensions;
use jolt_claims::protocols::jolt::{JoltRelationId, TraceDimensions};
use jolt_claims::NoChallenges;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::PublicIoMemory;
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckWitness;
use jolt_transcript::{Channel, ProverTranscript, Sponge};
use jolt_verifier::sites::STAGE2;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage1::Stage1ClearOutput;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage2::field_registers_claim_reduction::FieldRegistersClaimReduction;
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
pub struct Stage2ProverOutput<F: JoltField> {
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
pub fn prove_stage2<F, PCS, VC, H>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    config: &ProverConfig,
    public_io: &JoltDevice,
    stage1: &Stage1ClearOutput<F>,
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut ProverTranscript<H>,
) -> Result<Stage2ProverOutput<F>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    transcript.site(STAGE2);
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
    // The field-inline lane inputs enter the composed input claim exactly as on the
    // verifier — composed through the shared seam, before `input_claim`.
    let uniskip_inputs = product_uniskip_input_values_from_stage1(stage1);
    let uniskip_input_claim =
        uniskip_relation.input_claim(&uniskip_inputs, &NoChallenges::default())?;
    let uniskip_poly =
        tracing::info_span!("SpartanProductUniskip::first_round_poly").in_scope(|| {
            backend
                .spartan_product_uniskip
                .first_round_poly(session, &[tau_high], &uniskip_inputs)
        })?;
    // The canonical composed lane domain also determines the verifier's uni-skip check.
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
        field_registers_claim_reduction: FieldRegistersClaimReduction::new(
            FieldRegistersTraceDimensions::new(log_t),
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
    // Under `field-inline` the field-inline claim-reduction inputs wire from the
    // stage-1 field-inline carrier through the same shared
    // assembly the verifier runs.
    let inputs = stage2_batch_input_values_from_upstream(stage1, proved_uniskip.output_claim);

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

    let claims = Stage2OutputClaims::new(proved_uniskip.output_claim, proved.output_claims.clone());
    Ok(Stage2ProverOutput {
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

/// Clear round-trips with field-inline enabled of the stage-2 recipe through
/// the production stage-1/2 verifiers over the prover's argument string.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_verifier::JoltSponge;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        field_arithmetic_backend, field_arithmetic_preprocessing, fixture_transcript,
        test_checked_inputs, test_prover_config, test_public_io, verify_through, Through, LOG_T,
    };
    use crate::stages::stage1::prove_stage1;

    #[test]
    fn field_arithmetic_stage2_round_trips_the_composed_verifier() {
        let witness = field_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();

        let mut prover_transcript = fixture_transcript();
        let stage1 = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            &backend,
            &mut session,
            &mode,
            LOG_T,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();
        let out = prove_stage2::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
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

        // The field-inline product appendage is carried, and the spec's alias table
        // holds on honest data: the field-inline claim-reduction member outputs equal
        // the appendage values polynomial-for-polynomial.
        let appendage = &out.claims.batch_outputs.product_remainder.field_inline;
        let reduction = &out.claims.batch_outputs.field_registers_claim_reduction;
        assert_eq!(reduction.rs1_value, appendage.rs1_value);
        assert_eq!(reduction.rs2_value, appendage.rs2_value);
        assert_eq!(reduction.rd_value, appendage.rd_value);

        verify_through(
            Through::Stage2,
            &test_checked_inputs(),
            &field_arithmetic_preprocessing(),
            &prover_transcript,
        );
    }
}

/// Committed stage-2 output rows use the same canonical alias layout as
/// clear claims, and the production stage-1/2 zk verifiers consume the
/// prover's argument string.
#[cfg(all(test, feature = "field-inline", feature = "zk"))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_zk {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_verifier::JoltSponge;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        field_arithmetic_backend, field_arithmetic_preprocessing, fixture_transcript,
        test_checked_inputs, test_prover_config, test_public_io, test_vc_setup, verify_through,
        Through, LOG_T,
    };
    use crate::stages::stage1::prove_stage1;

    #[test]
    fn committed_stage2_witness_carries_the_curated_rows_and_verifies() {
        let witness = field_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let setup = test_vc_setup();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(Some(&setup)).unwrap();
        let config = test_prover_config();
        let public_io = test_public_io();

        let mut prover_transcript = fixture_transcript();
        let stage1 = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            &backend,
            &mut session,
            &mode,
            LOG_T,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();
        let out = prove_stage2::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
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

        // The three field-inline reduction openings alias the product member's rows.
        let value_count: usize = out
            .committed_witness
            .output_claim_rows
            .iter()
            .map(Vec::len)
            .sum();
        assert_eq!(value_count, 18);

        verify_through(
            Through::Stage2,
            &test_checked_inputs(),
            &field_arithmetic_preprocessing(),
            &prover_transcript,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without field-inline, the composed product geometry matches the RV64-only relation.
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

    /// With field-inline enabled, the composed product domain carries the two field-inline lanes —
    /// the spec's 5-point domain and its degree-12 first round.
    #[cfg(feature = "field-inline")]
    #[test]
    fn product_uniskip_constants_are_the_composed_field_domains() {
        assert_eq!(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, 5);
        assert_eq!(SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE, 12);
    }
}
