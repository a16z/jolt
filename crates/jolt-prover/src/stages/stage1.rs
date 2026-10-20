//! Stage 1: the Spartan outer uni-skip round and the outer remainder
//! sumcheck.
//!
//! Pure orchestration: the challenge draws, batch head, point derivation,
//! final-claim fold, and absorb order are `jolt-verifier`'s generated
//! drivers; all compute (input-table materialization, the brute-forced
//! uni-skip polynomial, the remainder rounds) is behind the backend's
//! `spartan_outer_uniskip` and `spartan_outer_remainder` slots.

use jolt_algebra::JoltField;
use jolt_claims::protocols::composed::r1cs::{
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_crypto::VectorCommitment;
use jolt_kernels::{JoltBackend, KernelContext};
use jolt_openings::CommitmentScheme;
use jolt_transcript::{Channel, ProverTranscript, Sponge};
use jolt_verifier::sites::STAGE1;
use jolt_verifier::stages::stage1::outer_remainder::{
    outer_remainder_input_values_from_uniskip_output, OuterRemainder,
};
use jolt_verifier::stages::stage1::outputs::{
    Stage1BatchInputClaims, Stage1BatchSumchecks, Stage1ClearOutput,
};
use jolt_verifier::stages::uniskip::draw_spartan_outer_tau;

use crate::recorder::ProofMode;
use crate::{ProverError, StageProver as _};

/// Stage 1's outputs: the verifier-typed cross-stage carrier downstream stages consume.
pub struct Stage1ProverOutput<F: JoltField> {
    pub clear_output: Stage1ClearOutput<F>,
}

/// Prove stage 1 on `transcript` (positioned at the stage-0 boundary).
#[tracing::instrument(skip_all)]
pub fn prove_stage1<F, PCS, VC, H>(
    backend: &mut KernelContext<'_, F, JoltBackend<F, PCS>>,
    mode: &mut ProofMode<VC>,
    log_t: usize,
    transcript: &mut ProverTranscript<H>,
) -> Result<Stage1ProverOutput<F>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    transcript.site(STAGE1);
    let tau = draw_spartan_outer_tau(transcript, log_t);
    // Backend-neutral kernel-seam spans at the call boundary, so every
    // `UniskipKernel` implementation inherits them — see the taxonomy's
    // kernel-seam contract.
    tracing::info_span!("SpartanOuterUniskip::prepare")
        .in_scope(|| backend.prepare_outer_uniskip(log_t, &tau))?;

    let uniskip_poly = tracing::info_span!("SpartanOuterUniskip::first_round_poly")
        .in_scope(|| backend.outer_uniskip_poly())?;
    let proved_uniskip = mode.prove_uniskip(
        uniskip_poly,
        F::zero(),
        SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
        SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
        transcript,
    )?;
    let uniskip_challenge = proved_uniskip.challenge;

    let sumchecks = Stage1BatchSumchecks {
        outer_remainder: OuterRemainder::new(
            SpartanOuterDimensions::rv64(log_t),
            tau,
            uniskip_challenge,
        ),
    };
    let challenges = sumchecks.draw_challenges(transcript)?;
    let input_points = sumchecks.empty_input_points();
    let inputs = Stage1BatchInputClaims {
        outer_remainder: outer_remainder_input_values_from_uniskip_output(
            proved_uniskip.output_claim,
        ),
    };

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

    let clear_output = Stage1ClearOutput::new(proved.output_claims, proved.output_points);
    Ok(Stage1ProverOutput { clear_output })
}

/// Clear round-trips with field-inline enabled of the stage-1 recipe through
/// the production `stage1::verify` over the prover's argument string.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
    use jolt_algebra::Fr;
    use jolt_claims::protocols::field_inline::geometry::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS;
    use jolt_claims::protocols::field_inline::FieldInlinePolynomialId;
    use jolt_claims::OutputClaims;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::stage2::product_tau_low;
    use jolt_verifier::JoltSponge;
    use jolt_witness::{JoltWitnessOracle as _, TraceBackend};

    use super::*;
    use crate::stages::field_inline_fixtures::{
        field_arithmetic_backend, field_arithmetic_preprocessing, fixture_transcript,
        test_checked_inputs, test_mode, verify_through, FixturePreprocessing, Through, LOG_T,
    };

    fn round_trip(trace_backend: TraceBackend, preprocessing: FixturePreprocessing) {
        let witness = trace_backend.with_field_inline().unwrap();
        let backend = crate::stages::field_inline_fixtures::reference_backend();
        let mut context = backend.with_witness(&witness);
        let mut mode = test_mode();
        let mut prover_transcript = fixture_transcript();
        let out = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            &mut context,
            &mut mode,
            LOG_T,
            &mut prover_transcript,
        )
        .unwrap();

        let field_inline_outer = &out.clear_output.output_values.outer_remainder.field_inline;

        // The appendage values are honest evaluations: each field-inline cycle-domain
        // column's MLE at the stage-1 cycle binding (`tau_low`, the point
        // stage 2's field-inline wiring consumes).
        let tau_low = product_tau_low(&out.clear_output.remainder_point(), LOG_T).unwrap();
        let field_inline_oracle = witness.field_inline().unwrap();
        for (polynomial, value) in FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS
            .into_iter()
            .zip(OutputClaims::opening_values(field_inline_outer))
        {
            let table = field_inline_oracle
                .oracle_table(FieldInlinePolynomialId::Virtual(polynomial))
                .unwrap();
            assert_eq!(Polynomial::<Fr>::new(table).evaluate(&tau_low), value);
        }

        verify_through(
            Through::Stage1,
            &test_checked_inputs(),
            &preprocessing,
            &mut prover_transcript,
        );
    }

    #[test]
    fn field_arithmetic_stage1_round_trips_the_composed_verifier() {
        round_trip(field_arithmetic_backend(), field_arithmetic_preprocessing());
    }
}

/// ZK with field-inline enabled: the committed stage-1 witness carries the
/// composed rows, and the production `stage1::verify` zk branch consumes the
/// prover's argument string.
#[cfg(all(test, feature = "field-inline", feature = "zk"))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_zk {
    use jolt_algebra::Fr;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_verifier::JoltSponge;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        field_arithmetic_backend, field_arithmetic_preprocessing, fixture_transcript,
        test_checked_inputs, test_mode, verify_through, Through, LOG_T,
    };

    #[test]
    fn masked_stage1_round_trips_the_masked_verifier() {
        let witness = field_arithmetic_backend().with_field_inline().unwrap();
        let backend = crate::stages::field_inline_fixtures::reference_backend();
        let mut context = backend.with_witness(&witness);
        let mut mode = test_mode();
        let mut prover_transcript = fixture_transcript();
        let _ = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            &mut context,
            &mut mode,
            LOG_T,
            &mut prover_transcript,
        )
        .unwrap();

        verify_through(
            Through::Stage1,
            &test_checked_inputs(),
            &field_arithmetic_preprocessing(),
            &mut prover_transcript,
        );
    }
}
