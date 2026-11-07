//! Stage 1: the Spartan outer uni-skip round and the outer remainder
//! sumcheck.
//!
//! Pure orchestration: the challenge draws, batch head, point derivation,
//! final-claim fold, and absorb order are `jolt-verifier`'s generated
//! drivers; all compute (input-table materialization, the brute-forced
//! uni-skip polynomial, the remainder rounds) is behind the backend's
//! `spartan_outer_uniskip` and `spartan_outer_remainder` slots.

use jolt_claims::protocols::composed::r1cs::{
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_kernels::{JoltBackend, KernelContext};
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{FieldTranscript, ProverTranscript};
use jolt_verifier::stages::pow::PowSite;

use crate::stages::grind_pow;
use jolt_verifier::stages::stage1::outer_remainder::{
    outer_remainder_input_values_from_uniskip_output, OuterRemainder,
};
use jolt_verifier::stages::stage1::outputs::{
    Stage1BatchInputClaims, Stage1BatchSumchecks, Stage1ClearOutput, Stage1OutputClaims,
};
use jolt_verifier::stages::uniskip::draw_spartan_outer_tau;

use crate::recorder::ProofMode;
use crate::{ProverError, StageProver as _};

/// Stage 1's outputs: the two wire proofs, the wire claims, and the
/// verifier-typed cross-stage carrier downstream stages consume.
pub struct Stage1ProverOutput<F: JoltField, C> {
    /// Nonce ground before the `tau` draw.
    pub tau_pow_nonce: Option<u64>,
    /// Nonce ground before the stage's challenge burst.
    pub pow_nonce: Option<u64>,
    pub uniskip_proof: SumcheckProof<F, C>,
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage1OutputClaims<F>,
    pub clear_output: Stage1ClearOutput<F>,
}

/// Prove stage 1 on `transcript` (positioned at the stage-0 boundary).
#[tracing::instrument(skip_all)]
pub fn prove_stage1<F, PCS, VC, T>(
    backend: &mut KernelContext<'_, F, JoltBackend<F, PCS>>,
    mode: &mut ProofMode<VC>,
    log_t: usize,
    transcript: &mut T,
) -> Result<Stage1ProverOutput<F, VC::Output>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: FieldTranscript<F> + ProverTranscript,
{
    let tau_pow_nonce = grind_pow::<F, _>(PowSite::Stage1Tau, transcript)?;
    let tau = draw_spartan_outer_tau(transcript, log_t)?;
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
    let pow_nonce = grind_pow::<F, _>(PowSite::Stage1, transcript)?;
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
    let sumcheck_proof = proved.recorded.proof.into();

    let claims = Stage1OutputClaims::new(proved_uniskip.output_claim, proved.output_claims.clone());
    let clear_output = Stage1ClearOutput::new(proved.output_claims, proved.output_points);
    Ok(Stage1ProverOutput {
        tau_pow_nonce,
        pow_nonce,
        uniskip_proof: proved_uniskip.proof,
        sumcheck_proof,
        claims,
        clear_output,
    })
}

/// Clear round-trips with field-inline enabled of the stage-1 recipe against the verifier's own
/// public constituents — `stage1::verify`'s clear body step for step (the
/// tau draw, `uniskip::verify_clear`, the batch relations, the field-inline seam's
/// attach, `verify_clear`, and the two-part opening absorb), on a twin
/// transcript. The full `stage1::verify` entrypoint needs an assembled
/// `JoltProof`, whose joint-opening slot has no test constructor, so this is
/// the closest public seam; the 32-byte transcript-state equality pins the
/// absorb order end to end.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
    use crate::stages::field_inline_fixtures::twins;
    use jolt_claims::protocols::field_inline::geometry::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS;
    use jolt_claims::protocols::field_inline::FieldInlinePolynomialId;
    use jolt_claims::OutputClaims;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_poly::Polynomial;
    use jolt_transcript::{prover_transcript, Keccak, ProverState};
    use jolt_verifier::stages::stage2::product_tau_low;
    use jolt_witness::{JoltWitnessOracle as _, TraceBackend};

    use super::*;
    use crate::stages::field_inline_fixtures::{field_arithmetic_backend, test_mode, LOG_T};

    fn round_trip(trace_backend: TraceBackend) {
        let witness = trace_backend.with_field_inline().unwrap();
        let backend = crate::stages::field_inline_fixtures::reference_backend();
        let mut context = backend.with_witness(&witness);
        let mut mode = test_mode();
        let mut prover_state = prover_transcript::<Keccak>(
            &jolt_transcript::protocol_tag(b"stage1-field-inline"),
            b"",
        );
        let out = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
            &mut context,
            &mut mode,
            LOG_T,
            &mut prover_state,
        )
        .unwrap();

        let field_inline_outer = &out.claims.outer.outer_remainder.field_inline;

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

        let mut transcript = prover_transcript::<Keccak>(
            &jolt_transcript::protocol_tag(b"stage1-field-inline"),
            b"",
        );
        twins::replay_stage1(&mut transcript, &out);

        assert_eq!(
            transcript.verifier_message::<[u8; 32]>(),
            prover_state.verifier_message::<[u8; 32]>()
        );
    }

    #[test]
    fn field_arithmetic_stage1_round_trips_the_composed_verifier() {
        round_trip(field_arithmetic_backend());
    }
}

/// ZK with field-inline enabled: the committed stage-1 witness carries the
/// composed rows, and the production `stage1::verify` zk branch consumes the
/// prover's argument string.
#[cfg(all(test, feature = "field-inline", feature = "zk", not(feature = "akita")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_zk {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_transcript::{Keccak, ProverState};

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
        let _ = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
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
            mode,
            &mut prover_transcript,
        );
    }
}
