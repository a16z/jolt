//! Stage 1: the Spartan outer uni-skip round and the outer remainder
//! sumcheck.
//!
//! Pure orchestration: the challenge draws, batch head, point derivation,
//! final-claim fold, and absorb order are `jolt-verifier`'s generated
//! drivers; all compute (input-table materialization, the brute-forced
//! uni-skip polynomial, the remainder rounds) is behind the backend's
//! `spartan_outer_uniskip` and `spartan_outer_remainder` slots.

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::relations::spartan::FieldRegistersSpartanOuterOutputClaims;
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
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage1::outer_remainder::{
    outer_remainder_input_values_from_uniskip_output, OuterRemainder,
};
use jolt_verifier::stages::stage1::outputs::{
    Stage1BatchInputClaims, Stage1BatchSumchecks, Stage1ClearOutput, Stage1OutputClaims,
};
use jolt_verifier::stages::uniskip::draw_spartan_outer_tau;
#[cfg(feature = "field-inline")]
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;

use crate::recorder::ProofMode;
use crate::{ProverError, StageProver as _};

/// Stage 1's outputs: the two wire proofs, the wire claims, and the
/// verifier-typed cross-stage carrier downstream stages consume.
pub struct Stage1ProverOutput<F: JoltField, C> {
    pub uniskip_proof: SumcheckProof<F, C>,
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage1OutputClaims<F>,
    pub clear_output: Stage1ClearOutput<F>,
    #[cfg(feature = "zk")]
    pub uniskip_witness: CommittedSumcheckWitness<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 1 on `transcript` (positioned at the stage-0 boundary).
#[tracing::instrument(skip_all)]
pub fn prove_stage1<F, PCS, VC, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    log_t: usize,
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut T,
) -> Result<Stage1ProverOutput<F, VC::Output>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
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

    let uniskip_poly = tracing::info_span!("SpartanOuterUniskip::first_round_poly")
        .in_scope(|| backend.spartan_outer_uniskip.first_round_poly(session, &[]))?;
    // The COMPOSED jolt-r1cs uni-skip shape (feature-aware): identical to the
    // jolt-claims RV64-only constants FR-off, the FR-extended row domain under
    // `field-inline` — the shape the verifier's stage-1 uni-skip checks.
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
    let (sumcheck_proof, committed_witness) = crate::recorder::split_recorded(proved.recorded)?;
    #[cfg(not(feature = "zk"))]
    let sumcheck_proof = proved.recorded.proof;

    #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
    let mut claims =
        Stage1OutputClaims::new(proved_uniskip.output_claim, proved.output_claims.clone());
    #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
    let mut clear_output = Stage1ClearOutput::new(proved.output_claims, proved.output_points);
    // Attach the FR Spartan-outer appendage the composed remainder kernel
    // published: `claims` is the wire carrier `stage1::verify` requires
    // fail-closed on FR-on proofs, `clear_output` the cross-stage carrier the
    // stage-2 recipe's FR wiring consumes.
    #[cfg(feature = "field-inline")]
    {
        let field_inline_outer = field_inline_outer_claims(&sumchecks)?;
        claims.field_inline_outer = Some(field_inline_outer.clone());
        clear_output.field_inline_output_values = Some(field_inline_outer);
    }
    Ok(Stage1ProverOutput {
        uniskip_proof: proved_uniskip.proof,
        sumcheck_proof,
        claims,
        clear_output,
        #[cfg(feature = "zk")]
        uniskip_witness: proved_uniskip.witness,
        #[cfg(feature = "zk")]
        committed_witness,
    })
}

/// Assemble the FR Spartan-outer appendage claims from the values the
/// composed remainder kernel published on the batch relation, through the
/// shared typed-claims constructor (ids resolved in appended-column order —
/// the same `outer_output_openings` order the verifier's seam absorbs).
#[cfg(feature = "field-inline")]
fn field_inline_outer_claims<F: JoltField>(
    sumchecks: &Stage1BatchSumchecks<F>,
) -> Result<FieldRegistersSpartanOuterOutputClaims<F>, ProverError<F>> {
    use jolt_claims::protocols::field_inline::geometry::spartan::outer_output_openings;
    use jolt_claims::OutputClaims as _;

    let values = sumchecks
        .outer_remainder
        .field_inline_outputs()
        .ok_or(ProverError::Verifier(VerifierError::MissingProofPayload {
            field: "stage1 FR Spartan-outer appendage (composed remainder kernel)",
        }))?;
    let openings = outer_output_openings();
    FieldRegistersSpartanOuterOutputClaims::from_opening_values(|id| {
        openings
            .iter()
            .position(|candidate| candidate == id)
            .and_then(|position| values.get(position).copied())
    })
    .map_err(|error| {
        ProverError::Verifier(VerifierError::StageClaimSumcheckFailed {
            stage: "Stage1Batch".to_string(),
            reason: format!("FR Spartan-outer appendage assembly failed: {error}"),
        })
    })
}

/// FR-on clear round-trips of the stage-1 recipe against the verifier's own
/// public constituents — `stage1::verify`'s clear body step for step (the
/// tau draw, `uniskip::verify_clear`, the batch relations, the FR seam's
/// attach, `verify_clear`, and the two-part opening absorb), on a twin
/// transcript. The full `stage1::verify` entrypoint needs an assembled
/// `JoltProof`, whose joint-opening slot has no test constructor, so this is
/// the closest public seam; the 32-byte transcript-state equality pins the
/// absorb order end to end.
#[cfg(all(test, feature = "field-inline", not(feature = "zk")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_round_trip {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_poly::Polynomial;
    use jolt_program::execution::OwnedTrace;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_verifier::stages::stage1::field_inline as stage1_field_inline;
    use jolt_verifier::stages::stage2::product_tau_low;
    use jolt_verifier::stages::uniskip::{self, UniskipParams};
    use jolt_witness::{JoltWitnessOracle as _, TraceBackend};

    use super::*;
    use crate::stages::field_inline_fixtures::{addi_only_backend, fr_arithmetic_backend, LOG_T};

    fn round_trip(trace_backend: TraceBackend<OwnedTrace>) {
        let witness = trace_backend.with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(None).unwrap();
        let mut prover_transcript = Blake2bTranscript::new(b"stage1-fr");
        let out = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &backend,
            &mut session,
            &mode,
            LOG_T,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();

        // The FR appendage rides both carriers, and they agree.
        let field_inline_outer = out.claims.field_inline_outer.clone().unwrap();
        assert_eq!(
            out.clear_output.field_inline_output_values.as_ref(),
            Some(&field_inline_outer)
        );

        // The appendage values are honest evaluations: each FR cycle-domain
        // column's MLE at the stage-1 cycle binding (`tau_low`, the point
        // stage 2's FR wiring consumes).
        let tau_low = product_tau_low(&out.clear_output.remainder_point(), LOG_T).unwrap();
        let field_inline_oracle = witness.field_inline().unwrap();
        for (polynomial, value) in
            jolt_claims::protocols::field_inline::geometry::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS
                .into_iter()
                .zip(
                    jolt_claims::OutputClaims::opening_values(&field_inline_outer),
                )
        {
            let table = field_inline_oracle
                .oracle_table(
                    jolt_claims::protocols::field_inline::FieldInlinePolynomialId::Virtual(
                        polynomial,
                    ),
                )
                .unwrap();
            assert_eq!(Polynomial::<Fr>::new(table).evaluate(&tau_low), value);
        }

        // The verifier twin.
        let mut transcript = Blake2bTranscript::new(b"stage1-fr");
        let tau = draw_spartan_outer_tau(&mut transcript, LOG_T);
        let uniskip_challenge = uniskip::verify_clear(
            &out.uniskip_proof,
            &UniskipParams::spartan_outer(),
            Fr::from_u64(0),
            out.claims.uniskip_output_claim,
            &mut transcript,
        )
        .unwrap();
        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(
                SpartanOuterDimensions::rv64(LOG_T),
                tau,
                uniskip_challenge,
            ),
        };
        let batch_challenges = sumchecks.draw_challenges(&mut transcript).unwrap();
        let input_points = sumchecks.empty_input_points();
        sumchecks.validate_output_claims(&out.claims.outer).unwrap();
        let attached = stage1_field_inline::attach_outer_outputs(&sumchecks, &out.claims).unwrap();
        let input_values = Stage1BatchInputClaims {
            outer_remainder: outer_remainder_input_values_from_uniskip_output(
                out.claims.uniskip_output_claim,
            ),
        };
        let _output_points = sumchecks
            .verify_clear(
                &input_values,
                &input_points,
                &batch_challenges,
                &out.claims.outer,
                &out.sumcheck_proof,
                &mut transcript,
                1,
            )
            .unwrap();
        sumchecks.append_output_claims(&mut transcript, &out.claims.outer);
        stage1_field_inline::append_outer_openings(&mut transcript, &attached);

        assert_eq!(transcript.state(), prover_transcript.state());
    }

    /// The ADDI-only FR-profile trace: every FR column is zero, so this pins
    /// the composed protocol on an FR-enabled guest that executes no FR
    /// instruction.
    #[test]
    fn addi_only_stage1_round_trips_the_composed_verifier() {
        round_trip(addi_only_backend());
    }

    /// Actual FR rows via decoded FR instruction words (two field loads and a
    /// multiply).
    #[test]
    fn fr_arithmetic_stage1_round_trips_the_composed_verifier() {
        round_trip(fr_arithmetic_backend());
    }
}

/// FR-on ZK: the committed stage-1 shell and the verifier replay. Mirrors
/// `blindfold.rs`'s hard transcript check at stage scope — the replay runs
/// `stage1::verify`'s zk body over its public constituents (the tau draw,
/// `uniskip::verify_zk`, the batch `verify_zk`) and must land on the
/// prover's forward transcript bytes.
#[cfg(all(test, feature = "field-inline", feature = "zk"))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod field_inline_zk {
    use common::constants::MAX_BLINDFOLD_GENERATORS;
    use common::jolt_device::JoltDevice;
    use jolt_crypto::{Bn254G1, Pedersen, PedersenSetup};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    #[cfg(feature = "akita")]
    use jolt_verifier::stages::stage8::field_inline_packed::FieldIncLimbsScheduled;
    use jolt_verifier::stages::uniskip::{self, UniskipParams};
    use jolt_verifier::stages::PrecommittedSchedule;
    use jolt_verifier::CheckedInputs;

    use super::*;
    use crate::stages::field_inline_fixtures::{fr_arithmetic_backend, ENTRY, LOG_T};

    const CAPACITY: usize = MAX_BLINDFOLD_GENERATORS;

    #[test]
    fn committed_stage1_shell_carries_the_composed_rows_and_replays() {
        let witness = fr_arithmetic_backend().with_field_inline().unwrap();
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let mut session = backend.begin_proof();
        let setup = PedersenSetup::new(vec![Bn254G1::default(); CAPACITY], Bn254G1::default());
        let mode = ProofMode::<Pedersen<Bn254G1>>::new(Some(&setup)).unwrap();
        let mut prover_transcript = Blake2bTranscript::new(b"stage1-fr-zk");
        let out = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &backend,
            &mut session,
            &mode,
            LOG_T,
            &witness,
            &mut prover_transcript,
        )
        .unwrap();

        // The committed shell carries the composed 48 output-claim values
        // (35 member openings + 13 FR appendage), row-committed in
        // capacity-sized chunks — the shape the verifier's
        // `composed_output_claim_count` check derives.
        let total: usize = out
            .committed_witness
            .output_claim_rows
            .iter()
            .map(Vec::len)
            .sum();
        assert_eq!(total, 48);
        let row_lens: Vec<usize> = out
            .committed_witness
            .output_claim_rows
            .iter()
            .map(Vec::len)
            .collect();
        let expected_row_lens: Vec<usize> = {
            let mut remaining = 48usize;
            let mut lens = Vec::new();
            while remaining > 0 {
                let take = remaining.min(CAPACITY);
                lens.push(take);
                remaining -= take;
            }
            lens
        };
        assert_eq!(row_lens, expected_row_lens);
        let committed = out.sumcheck_proof.as_committed().unwrap();
        assert_eq!(
            committed.output_claims.commitments.len(),
            48usize.div_ceil(CAPACITY)
        );

        // The replay.
        let checked = CheckedInputs {
            public_io: JoltDevice::default(),
            zk: true,
            trace_length: 1 << LOG_T,
            ram_K: 1 << 4,
            entry_address: ENTRY,
            preprocessing_digest: [0u8; 32],
            trusted_advice_commitment_present: false,
            vc_capacity: Some(CAPACITY),
            precommitted: PrecommittedSchedule {
                trusted_advice: None,
                untrusted_advice: None,
                bytecode: None,
                program_image: None,
                #[cfg(feature = "akita")]
                field_inc_limbs: Some(FieldIncLimbsScheduled),
            },
        };
        let mut transcript = Blake2bTranscript::new(b"stage1-fr-zk");
        let tau = draw_spartan_outer_tau(&mut transcript, LOG_T);
        let uniskip_step = uniskip::verify_zk(
            &checked,
            &out.uniskip_proof,
            &UniskipParams::spartan_outer(),
            &mut transcript,
        )
        .unwrap();
        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(
                SpartanOuterDimensions::rv64(LOG_T),
                tau,
                uniskip_step.challenge,
            ),
        };
        let _consistency = sumchecks
            .verify_zk(&out.sumcheck_proof, &mut transcript)
            .unwrap();

        assert_eq!(transcript.state(), prover_transcript.state());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FR-off, the composed jolt-r1cs outer uni-skip constants equal the
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

    /// FR-on, the composed outer domain carries the appended FR rows — the
    /// spec's 14-point domain and its degree-39 first round.
    #[cfg(feature = "field-inline")]
    #[test]
    fn outer_uniskip_constants_are_the_composed_fr_domains() {
        assert_eq!(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, 14);
        assert_eq!(SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE, 39);
    }
}
