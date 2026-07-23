use jolt_claims::protocols::jolt::{
    geometry::{booleanity::BooleanityDimensions, dimensions::JoltFormulaDimensions},
    JoltRelationId,
};
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

use super::{
    booleanity::{BooleanityAddressPhase, BooleanityAddressPhaseInputClaims},
    bytecode_read_raf::{
        bytecode_read_raf_address_phase_input_values_from_upstream, BytecodeReadRafAddressPhase,
    },
    outputs::{
        Stage6aCarriedChallenges, Stage6aClearOutput, Stage6aInputClaims, Stage6aOutput,
        Stage6aSumchecks, Stage6aZkOutput,
    },
};
use crate::{
    proof::JoltProof,
    stages::{
        stage1::Stage1Output, stage2::Stage2Output, stage3::Stage3Output, stage4::Stage4Output,
        stage5::Stage5Output, zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6a's address-phase input claim folds all five prior stage outputs directly; bundling them would reintroduce the removed `Deps` indirection."
)]
pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    formula_dimensions: &JoltFormulaDimensions,
    transcript: &mut T,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
    stage2: &Stage2Output<PCS::Field, VC::Output>,
    stage3: &Stage3Output<PCS::Field, VC::Output>,
    stage4: &Stage4Output<PCS::Field, VC::Output>,
    stage5: &Stage5Output<PCS::Field, VC::Output>,
) -> Result<Stage6aOutput<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let log_t = formula_dimensions.trace.log_t();

    let committed_program = checked.precommitted.bytecode.is_some();

    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        proof.one_hot_config.committed_chunk_bits(),
    );
    let address_sumchecks = Stage6aSumchecks {
        bytecode_read_raf: BytecodeReadRafAddressPhase::new(
            formula_dimensions.bytecode_read_raf,
            committed_program,
        ),
        booleanity: BooleanityAddressPhase::new(booleanity_dimensions),
    };

    // Six squeezes: the bytecode fold gamma plus the five per-stage folding
    // gammas, each formerly an inline `challenge_scalar_powers(..)` whose single
    // squeeze's degree-1 power equals the squeezed scalar. Byte- and value-equal
    // (test-locked in `bytecode_read_raf.rs`); the drawn scalars ride downstream
    // verbatim, and stage 6b's folds expand the power vectors they consume via
    // `stage_gamma_powers` (the same recurrence `challenge_scalar_powers` uses;
    // test-locked below).
    let address_challenges = address_sumchecks.draw_challenges(transcript)?;

    // WHY these draws live in stage 6a but feed only stage 6b: the prover's
    // booleanity subprotocol samples its gamma — and pads the reference address
    // with a fresh `challenge_vector` draw — before the 6a batch runs, so the
    // transcript schedule fixes them here. Stage 6b's booleanity member is their
    // only consumer, so they ride downstream as typed upstream values (the same
    // idiom as `Stage2ZkOutput`'s `product_tau_high`).
    let stage5_points = stage5.output_points();
    let stage5_instruction_address = stage5.instruction_r_address();
    let stage5_instruction_cycle = stage5_points.instruction_r_cycle();

    let mut booleanity_reference_address = stage5_instruction_address.to_vec();
    booleanity_reference_address.reverse();
    if booleanity_reference_address.len() < proof.one_hot_config.committed_chunk_bits() {
        let missing =
            proof.one_hot_config.committed_chunk_bits() - booleanity_reference_address.len();
        booleanity_reference_address.extend(transcript.challenge_vector(missing));
    } else {
        booleanity_reference_address = booleanity_reference_address
            [booleanity_reference_address.len() - proof.one_hot_config.committed_chunk_bits()..]
            .to_vec();
    }
    let mut booleanity_reference_cycle = stage5_instruction_cycle.to_vec();
    booleanity_reference_cycle.reverse();
    let booleanity_gamma = transcript.challenge();

    let carried = Stage6aCarriedChallenges {
        bytecode_read_raf: address_challenges.bytecode_read_raf,
        booleanity_reference_address,
        booleanity_reference_cycle,
        booleanity_gamma,
    };

    // Every member's input points are empty (the address phase reads only
    // opening values; produced points derive from its own sumcheck point).
    let address_input_points = address_sumchecks.empty_input_points();

    if checked.zk {
        let consistency =
            address_sumchecks.verify_zk(&proof.stages.stage6a_sumcheck_proof, transcript)?;
        let output_claims = committed::verify_output_claim_commitments(
            checked,
            &proof.stages.stage6a_sumcheck_proof,
            "stage6a_sumcheck_proof",
            address_sumchecks.output_claim_count(),
            JoltRelationId::BytecodeReadRaf,
        )?;
        let output_points = address_sumchecks
            .derive_opening_points(&consistency.challenges(), &address_input_points)?;
        return Ok(Stage6aOutput::Zk(Stage6aZkOutput {
            challenges: carried,
            consistency,
            output_claims,
            output_points,
        }));
    }

    let claims = &proof.clear_claims()?.stage6a;
    // Rejects val-stage claims whose presence or count disagrees with the
    // committed-program mode (the bytecode member's wire set carries the staged
    // `BytecodeValClaim` ids exactly when the program is committed).
    address_sumchecks.validate_output_claims(claims)?;

    // The bytecode address-phase input claim is the gamma-folded bind of every
    // prior clear stage opening (plus, under akita, the four reduced `Inc`
    // claims at the fused-inc consumer stage slots); the relation evaluates it
    // through its input `Expr` from these wired openings + the per-stage
    // folding gammas.
    let base_input_values = bytecode_read_raf_address_phase_input_values_from_upstream(
        &stage1.clear()?.output_values,
        &stage2.clear()?.output_values,
        &stage3.clear()?.output_values,
        &stage4.clear()?.output_values,
        &stage5.clear()?.output_values,
    );
    #[cfg(feature = "akita")]
    let base_input_values =
        jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafAddressPhaseInputClaims {
            base: base_input_values,
            inc: crate::stages::stage6b::inc_claim_reduction::inc_claim_reduction_input_values_from_upstream(
                &stage2.clear()?.output_values,
                &stage4.clear()?.output_values,
                &stage5.clear()?.output_values,
            ),
        };
    let address_input_values = Stage6aInputClaims {
        bytecode_read_raf: base_input_values,
        booleanity: BooleanityAddressPhaseInputClaims::default(),
    };

    let output_points = address_sumchecks.verify_clear(
        &address_input_values,
        &address_input_points,
        &address_challenges,
        claims,
        &proof.stages.stage6a_sumcheck_proof,
        transcript,
        6,
    )?;

    // The address-phase opening order (bytecode `intermediate`, each `val_stages`,
    // then booleanity `intermediate`) is single-sourced from the generated
    // `append_output_claims` (member declaration order = canonical Fiat-Shamir
    // order; no alias dedup in the address phase).
    address_sumchecks.append_output_claims(transcript, claims);

    Ok(Stage6aOutput::Clear(Stage6aClearOutput {
        output_values: claims.clone(),
        output_points,
        challenges: carried,
    }))
}

#[cfg(test)]
mod tests {
    use super::super::booleanity::BooleanityAddressPhaseOutputClaims;
    use super::super::bytecode_read_raf::BytecodeReadRafAddressPhaseOutputClaims;
    use super::super::outputs::Stage6aOutputClaims;
    use super::*;
    use crate::stages::relations::append_recording::RecordingTranscript;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn sample_claims() -> Stage6aOutputClaims<Fr> {
        Stage6aOutputClaims {
            bytecode_read_raf: BytecodeReadRafAddressPhaseOutputClaims {
                intermediate: fr(901),
                val_stages: Vec::new(),
            },
            booleanity: BooleanityAddressPhaseOutputClaims {
                intermediate: fr(902),
            },
        }
    }

    /// Locks the stage-6a address-phase Fiat-Shamir append order against silent
    /// drift: bytecode read-RAF `intermediate`, each `val_stages` entry, then
    /// booleanity `intermediate`. Single-sourced from the generated
    /// `append_output_claims`.
    #[test]
    #[expect(clippy::unwrap_used)]
    fn stage6a_output_claims_append_follows_canonical_order() {
        use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
        use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;

        let sumchecks = Stage6aSumchecks::<Fr> {
            bytecode_read_raf: BytecodeReadRafAddressPhase::new(
                BytecodeReadRafDimensions::new(3, 4, 2),
                true,
            ),
            booleanity: BooleanityAddressPhase::new(BooleanityDimensions::new(
                JoltRaPolynomialLayout::new(2, 1, 1).unwrap(),
                4,
                2,
            )),
        };
        let mut claims = sample_claims();
        claims.bytecode_read_raf.val_stages = (903..908).map(fr).collect();

        let mut got = RecordingTranscript::default();
        sumchecks.append_output_claims(&mut got, &claims);

        let mut want = RecordingTranscript::default();
        for value in [901, 903, 904, 905, 906, 907, 902].map(fr) {
            want.append_labeled(b"opening_claim", &value);
        }

        assert_eq!(got.chunks, want.chunks);
    }

    /// A transcript double whose every squeeze returns the same nontrivial
    /// scalar, so `challenge_scalar_powers`' output is a genuine power vector.
    #[derive(Clone, Default)]
    struct ConstantChallengeTranscript;

    impl Transcript for ConstantChallengeTranscript {
        type Challenge = Fr;
        fn new(_label: &'static [u8]) -> Self {
            Self
        }
        fn append_bytes(&mut self, _bytes: &[u8]) {}
        fn challenge(&mut self) -> Self::Challenge {
            Fr::from_u64(7)
        }
        fn state(&self) -> [u8; 32] {
            [0u8; 32]
        }
    }

    /// The `stage_gamma_powers` expansion of a drawn scalar must equal
    /// `challenge_scalar_powers`' output for the same squeezed scalar — the
    /// value-identity the stage-6a generated draw substitution relies on
    /// (the prover draws each stage's power vector inline; the verifier stores
    /// the scalar and expands at the stage-6b folds).
    #[test]
    fn stage_gamma_powers_matches_challenge_scalar_powers() {
        use jolt_claims::protocols::jolt::geometry::bytecode::BYTECODE_STAGE_GAMMA_COUNTS;
        use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;

        let gamma = Fr::from_u64(7);
        let challenges = BytecodeReadRafAddressPhaseChallenges {
            gamma,
            stage1_gamma: gamma,
            stage2_gamma: gamma,
            stage3_gamma: gamma,
            stage4_gamma: gamma,
            stage5_gamma: gamma,
        };
        let mut transcript = ConstantChallengeTranscript;
        for (powers, len) in challenges
            .stage_gamma_powers()
            .into_iter()
            .zip(BYTECODE_STAGE_GAMMA_COUNTS)
        {
            assert_eq!(powers, transcript.challenge_scalar_powers(len));
        }
    }
}
