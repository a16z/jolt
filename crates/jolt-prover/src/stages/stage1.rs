//! Stage 1: the Spartan outer uni-skip round and the outer remainder
//! sumcheck.
//!
//! Pure orchestration: the challenge draws, batch head, point derivation,
//! final-claim fold, and absorb order are `jolt-verifier`'s generated
//! drivers; all compute (input-table materialization, the brute-forced
//! uni-skip polynomial, the remainder rounds) is behind the backend's
//! `spartan_outer_uniskip` and `spartan_outer_remainder` slots.

use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
};
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckWitness;
use jolt_transcript::{ProverTranscript, Sponge};
use jolt_verifier::stages::stage1::outer_remainder::{
    outer_remainder_input_values_from_uniskip_output, OuterRemainder,
};
use jolt_verifier::stages::stage1::outputs::{
    Stage1BatchInputClaims, Stage1BatchSumchecks, Stage1ClearOutput, Stage1OutputClaims,
};
use jolt_verifier::stages::uniskip::draw_spartan_outer_tau;
use jolt_witness::JoltWitnessPlane;

use crate::recorder::ProofMode;
use crate::{ProverError, StageProver as _};

/// Stage 1's outputs: the two wire proofs, the wire claims, and the
/// verifier-typed cross-stage carrier downstream stages consume.
pub struct Stage1ProverOutput<F: JoltField> {
    pub claims: Stage1OutputClaims<F>,
    pub clear_output: Stage1ClearOutput<F>,
    #[cfg(feature = "zk")]
    pub uniskip_witness: CommittedSumcheckWitness<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 1 on `transcript` (positioned at the stage-0 boundary).
#[tracing::instrument(skip_all)]
pub fn prove_stage1<F, PCS, VC, H>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    log_t: usize,
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut ProverTranscript<H>,
) -> Result<Stage1ProverOutput<F>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    let tau = draw_spartan_outer_tau(transcript, log_t);
    // Backend-neutral kernel-seam spans at the call boundary, so every
    // `UniskipKernel` implementation inherits them — see the taxonomy's
    // kernel-seam contract.
    tracing::info_span!("SpartanOuterUniskip::prepare").in_scope(|| {
        backend
            .spartan_outer_uniskip
            .prepare(session, log_t, &tau, witness)
    })?;

    let uniskip_poly =
        tracing::info_span!("SpartanOuterUniskip::first_round_poly").in_scope(|| {
            backend
                .spartan_outer_uniskip
                .first_round_poly(session, &[], &())
        })?;
    // The selected jolt-r1cs shape includes the field-inline rows when enabled.
    let proved_uniskip = mode.prove_uniskip(
        uniskip_poly,
        F::zero(),
        SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
        SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
        transcript,
    )?;
    let uniskip_challenge = proved_uniskip.challenge;

    // The generated stage drivers, on the verifier's own batch type.
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

    let claims = Stage1OutputClaims::new(proved_uniskip.output_claim, proved.output_claims.clone());
    let clear_output = Stage1ClearOutput::new(proved.output_claims, proved.output_points);
    Ok(Stage1ProverOutput {
        claims,
        clear_output,
        #[cfg(feature = "zk")]
        uniskip_witness: proved_uniskip.witness,
        #[cfg(feature = "zk")]
        committed_witness,
    })
}

/// Clear round-trips with field-inline enabled of the stage-1 recipe through
/// the production `stage1::verify` over the prover's argument string.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
    use jolt_claims::protocols::field_inline::geometry::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS;
    use jolt_claims::protocols::field_inline::FieldInlinePolynomialId;
    use jolt_claims::OutputClaims;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_poly::Polynomial;
    use jolt_program::execution::OwnedTrace;
    use jolt_verifier::stages::stage2::product_tau_low;
    use jolt_verifier::JoltSponge;
    use jolt_witness::{JoltWitnessOracle as _, TraceBackend};

    use super::*;
    use crate::stages::field_inline_fixtures::{
        addi_only_backend, addi_only_preprocessing, field_arithmetic_backend,
        field_arithmetic_preprocessing, fixture_transcript, test_checked_inputs, verify_through,
        FixturePreprocessing, Through, LOG_T,
    };

    fn round_trip(trace_backend: TraceBackend<OwnedTrace>, preprocessing: FixturePreprocessing) {
        let witness = trace_backend.with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let mut prover_transcript = fixture_transcript();
        let out = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            &backend,
            &mut session,
            &mode,
            LOG_T,
            &witness,
            &mut prover_transcript,
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

        verify_through(
            Through::Stage1,
            &test_checked_inputs(),
            &preprocessing,
            &prover_transcript,
        );
    }

    /// The ADDI-only field-inline trace: every field-inline column is zero, so this pins
    /// the composed protocol on a field-inline guest that executes no field-inline
    /// instruction.
    #[test]
    fn addi_only_stage1_round_trips_the_composed_verifier() {
        round_trip(addi_only_backend(), addi_only_preprocessing());
    }

    /// Actual field-inline rows via decoded field-inline instruction words (two field loads and a
    /// multiply).
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
    use common::constants::MAX_BLINDFOLD_GENERATORS;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_verifier::JoltSponge;

    use super::*;
    use crate::stages::field_inline_fixtures::{
        field_arithmetic_backend, field_arithmetic_preprocessing, fixture_transcript,
        test_checked_inputs, test_vc_setup, verify_through, Through, LOG_T,
    };

    const CAPACITY: usize = MAX_BLINDFOLD_GENERATORS;

    #[test]
    fn committed_stage1_witness_carries_the_composed_rows_and_verifies() {
        let witness = field_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let setup = test_vc_setup();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(Some(&setup)).unwrap();
        let mut prover_transcript = fixture_transcript();
        let out = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            &backend,
            &mut session,
            &mode,
            LOG_T,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();

        // The committed witness carries the composed 50 output-claim values
        // (45 common openings + five field value/product openings), row-committed in
        // capacity-sized chunks — the shape the verifier's
        // `composed_output_claim_count` check derives.
        let row_lens: Vec<usize> = out
            .committed_witness
            .output_claim_rows
            .iter()
            .map(Vec::len)
            .collect();
        let expected_row_lens: Vec<usize> = {
            let mut remaining = 50usize;
            let mut lens = Vec::new();
            while remaining > 0 {
                let take = remaining.min(CAPACITY);
                lens.push(take);
                remaining -= take;
            }
            lens
        };
        assert_eq!(row_lens, expected_row_lens);

        verify_through(
            Through::Stage1,
            &test_checked_inputs(),
            &field_arithmetic_preprocessing(),
            &prover_transcript,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without field-inline, the composed jolt-r1cs outer uni-skip constants equal the
    /// jolt-claims RV64-only constants this recipe previously passed — the
    /// swap is byte-neutral.
    #[cfg(not(feature = "field-inline"))]
    #[test]
    fn outer_uniskip_constants_match_the_rv64_only_values() {
        use jolt_claims::protocols::jolt::geometry::dimensions::{
            OUTER_UNISKIP_DOMAIN_SIZE, OUTER_UNISKIP_FIRST_ROUND_DEGREE,
        };

        assert_eq!(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, OUTER_UNISKIP_DOMAIN_SIZE);
        assert_eq!(
            SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
            OUTER_UNISKIP_FIRST_ROUND_DEGREE
        );
    }

    /// With field-inline enabled, the composed outer domain carries the appended field-inline rows
    /// — the spec's 15-point domain and its degree-42 first round.
    #[cfg(feature = "field-inline")]
    #[test]
    fn outer_uniskip_constants_are_the_composed_field_domains() {
        assert_eq!(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, 15);
        assert_eq!(SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE, 42);
    }
}
