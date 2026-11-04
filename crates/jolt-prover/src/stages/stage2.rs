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
use jolt_algebra::JoltField;
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
use jolt_kernels::{JoltBackend, KernelContext};
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::PublicIoMemory;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{FieldTranscript, ProverTranscript};
use jolt_verifier::stages::pow::PowSite;

use crate::stages::grind_pow;
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

use crate::recorder::ProofMode;
use crate::{ProverConfig, ProverError, StageProver as _};

/// Stage 2's outputs: the two wire proofs, the wire claims, and the
/// verifier-typed cross-stage carrier downstream stages consume.
pub struct Stage2ProverOutput<F: JoltField, C> {
    /// Nonce ground before the `tau_high` draw.
    pub tau_high_pow_nonce: Option<u64>,
    /// Nonce ground before the stage's challenge burst.
    pub pow_nonce: Option<u64>,
    pub uniskip_proof: SumcheckProof<F, C>,
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage2OutputClaims<F>,
    pub clear_output: Stage2ClearOutput<F>,
}

/// Prove stage 2 on `transcript` (positioned at the stage-1 boundary).
#[tracing::instrument(skip_all)]
pub fn prove_stage2<F, PCS, VC, T>(
    backend: &mut KernelContext<'_, F, JoltBackend<F, PCS>>,
    mode: &mut ProofMode<VC>,
    config: &ProverConfig,
    public_io: &JoltDevice,
    stage1: &Stage1ClearOutput<F>,
    transcript: &mut T,
) -> Result<Stage2ProverOutput<F, VC::Output>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: FieldTranscript<F> + ProverTranscript,
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
    tracing::info_span!("SpartanProductUniskip::prepare")
        .in_scope(|| backend.prepare_product_uniskip(log_t, &tau_low))?;

    let tau_high_pow_nonce = grind_pow::<F, _>(PowSite::Stage2TauHigh, transcript)?;
    let tau_high: F = draw_spartan_product_tau_high(transcript)?;
    let uniskip_relation = ProductUniskip::new(product_dimensions, tau_high);
    // The field-inline lane inputs enter the composed input claim exactly as on the
    // verifier — composed through the shared seam, before `input_claim`.
    let uniskip_inputs = product_uniskip_input_values_from_stage1(stage1);
    let uniskip_input_claim =
        uniskip_relation.input_claim(&uniskip_inputs, &NoChallenges::default())?;
    let uniskip_poly = tracing::info_span!("SpartanProductUniskip::first_round_poly")
        .in_scope(|| backend.product_uniskip_poly(&[tau_high], &uniskip_inputs))?;
    // The canonical composed lane domain also determines the verifier's uni-skip check.
    let proved_uniskip = mode.prove_uniskip(
        uniskip_poly,
        uniskip_input_claim,
        SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        transcript,
    )?;
    let uniskip_challenge = proved_uniskip.challenge;

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
    let pow_nonce = grind_pow::<F, _>(PowSite::Stage2, transcript)?;
    let challenges = sumchecks.draw_challenges(transcript)?;

    let input_points = sumchecks.empty_input_points();
    // Under `field-inline` the field-inline claim-reduction inputs wire from the
    // stage-1 field-inline carrier through the same shared
    // assembly the verifier runs.
    let inputs = stage2_batch_input_values_from_upstream(stage1, proved_uniskip.output_claim);

    let mut scheduler = backend.build_round_scheduler();
    let proved = sumchecks.prove(
        backend,
        &mut *scheduler,
        &inputs,
        &input_points,
        &challenges,
        mode.recorder(),
        transcript,
    )?;
    #[cfg(not(feature = "zk"))]
    let sumcheck_proof = proved.recorded.proof.into();

    let claims = Stage2OutputClaims::new(proved_uniskip.output_claim, proved.output_claims.clone());
    Ok(Stage2ProverOutput {
        tau_high_pow_nonce,
        pow_nonce,
        uniskip_proof: proved_uniskip.proof,
        sumcheck_proof,
        claims,
        clear_output: Stage2ClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
            product_tau_low: tau_low,
        },
    })
}

/// Clear round-trips with field-inline enabled of the stage-2 recipe against the verifier's own
/// public constituents — `stage2::verify`'s clear body step for step (the
/// `τ_high` draw, the composed uni-skip via the seam attach and
/// `uniskip::verify_clear`, the six-member batch with the field-inline claim-reduction
/// member, the field-inline product appendage composition with its alias equality, and the
/// curated absorb) on a twin transcript. The full `stage2::verify` entrypoint
/// needs an assembled `JoltProof` (no test constructor for the joint-opening
/// slot), so this is the closest public seam; the 32-byte transcript-state
/// equality pins the absorb order end to end.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
    use crate::stages::field_inline_fixtures::twins;
    use jolt_algebra::Fr;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_transcript::{prover_transcript, Keccak, ProverState};

    use super::*;
    use crate::stages::field_inline_fixtures::{
        field_arithmetic_backend, test_mode, test_prover_config, test_public_io, LOG_T,
    };
    use crate::stages::stage1::prove_stage1;

    #[test]
    fn field_arithmetic_stage2_round_trips_the_composed_verifier() {
        let witness = field_arithmetic_backend().with_field_inline().unwrap();
        let backend = crate::stages::field_inline_fixtures::reference_backend();
        let mut context = backend.with_witness(&witness);
        let mut mode = test_mode();
        let config = test_prover_config();
        let public_io = test_public_io();

        let mut prover_state = prover_transcript::<Keccak>(
            &jolt_transcript::protocol_tag(b"stage2-field-inline"),
            b"",
        );
        let stage1 = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
            &mut context,
            &mut mode,
            LOG_T,
            &mut prover_state,
        )
        .unwrap();
        let out = prove_stage2::<Fr, DoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
            &mut context,
            &mut mode,
            &config,
            &public_io,
            &stage1.clear_output,
            &mut prover_state,
        )
        .unwrap();

        let appendage = out
            .claims
            .batch_outputs
            .product_remainder
            .field_inline
            .clone();
        let reduction = &out.claims.batch_outputs.field_registers_claim_reduction;
        assert_eq!(reduction.rs1_value, appendage.rs1_value);
        assert_eq!(reduction.rs2_value, appendage.rs2_value);
        assert_eq!(reduction.rd_value, appendage.rd_value);

        let mut transcript = prover_transcript::<Keccak>(
            &jolt_transcript::protocol_tag(b"stage2-field-inline"),
            b"",
        );
        twins::replay_stage1(&mut transcript, &stage1);
        twins::replay_stage2(&mut transcript, &config, &public_io, &stage1, &out);

        assert_eq!(
            transcript.verifier_message::<[u8; 32]>(),
            prover_state.verifier_message::<[u8; 32]>()
        );
    }
}

/// clear claims, and the production stage-1/2 zk verifiers consume the
/// prover's argument string.
#[cfg(all(test, feature = "field-inline", feature = "zk", not(feature = "akita")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_zk {
    use jolt_algebra::Fr;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_verifier::JoltSponge;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        field_arithmetic_backend, field_arithmetic_preprocessing, fixture_transcript,
        test_checked_inputs, test_mode, test_prover_config, test_public_io, verify_through,
        Through, LOG_T,
    };
    use crate::stages::stage1::prove_stage1;

    #[test]
    fn masked_stage2_round_trips_the_masked_verifier() {
        let witness = field_arithmetic_backend().with_field_inline().unwrap();
        let backend = crate::stages::field_inline_fixtures::reference_backend();
        let mut context = backend.with_witness(&witness);
        let mut mode = test_mode();
        let config = test_prover_config();
        let public_io = test_public_io();

        let mut prover_transcript = fixture_transcript();
        let stage1 = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            &mut context,
            &mut mode,
            LOG_T,
            &mut prover_transcript,
        )
        .unwrap();
        let _ = prove_stage2::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            &mut context,
            &mut mode,
            &config,
            &public_io,
            &stage1.clear_output,
            &mut prover_transcript,
        )
        .unwrap();

        verify_through(
            Through::Stage2,
            &test_checked_inputs(),
            &field_arithmetic_preprocessing(),
            &mut prover_transcript,
        );
    }
}
