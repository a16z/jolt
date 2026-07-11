//! Stage 7: the Hamming-weight claim-reduction batch plus the present advice
//! address phases (committed-program `Option` members remain rejected
//! upstream).
//!
//! Pure orchestration mirroring `stage7::verify`: the whole batch is the
//! verifier's own promoted `build_stage7_sumchecks` (an advice address phase
//! is present exactly when its layout is committed AND its schedule has
//! active address rounds), the challenges come from the generated
//! declaration-order draw, and the inputs from the promoted
//! `stage7_input_values_from_upstream`. The advice members are the SAME
//! kernel objects stage 6b bound through the cycle phase, transitioned here
//! to the address phase.

use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::advice_claim_reduction::AdviceReductionProver;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{
    prove_batch, ClearSumcheckRecorder, ProveRounds, SumcheckProof, SumcheckRecorder,
};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::stage4::Stage4ClearOutput;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::advice_address_phase::{
    TrustedAdviceAddressPhaseOutputClaims, UntrustedAdviceAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::stage7_hamming_virtualization_address_points;
use jolt_verifier::stages::stage7::outputs::{Stage7ClearOutput, Stage7OutputClaims};
use jolt_verifier::stages::stage7::{build_stage7_sumchecks, stage7_input_values_from_upstream};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

/// Stage 7's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier stage 8 consumes.
pub struct Stage7ProverOutput<F: Field, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage7OutputClaims<F>,
    pub clear_output: Stage7ClearOutput<F>,
}

/// Prove stage 7 on `transcript` (positioned at the stage-6b boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
pub fn prove_stage7<F, PCS, VC, C, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    stage4: &Stage4ClearOutput<F>,
    stage6b: &Stage6bClearOutput<F>,
    mut trusted_advice_member: Option<Box<dyn AdviceReductionProver<F>>>,
    mut untrusted_advice_member: Option<Box<dyn AdviceReductionProver<F>>>,
    witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    transcript: &mut T,
) -> Result<Stage7ProverOutput<F, C>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let precommitted = &checked.precommitted;
    if precommitted.bytecode.is_some() || precommitted.program_image.is_some() {
        return Err(ProverError::Unsupported {
            reason: "committed-program claim reductions are not yet supported",
        });
    }
    let formula_dimensions =
        jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions::try_from(
            config.one_hot_config.dimensions(
                log_t,
                2 * RISCV_XLEN,
                preprocessing.verifier.program.bytecode_len(),
                checked.ram_K,
            ),
        )
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: error.to_string(),
        })?;
    let hamming_dimensions = HammingWeightClaimReductionDimensions::new(
        formula_dimensions.ra_layout,
        config.one_hot_config.committed_chunk_bits(),
    );

    let booleanity_opening =
        stage6b
            .output_points
            .booleanity_opening_point()
            .ok_or(ProverError::Unsupported {
                reason: "stage-6b booleanity produced no opening point",
            })?;
    let (booleanity_r_address, booleanity_r_cycle) =
        booleanity_opening.split_at(hamming_dimensions.log_k_chunk);
    let virtualization_points =
        stage7_hamming_virtualization_address_points(hamming_dimensions, &stage6b.output_points)?;

    let sumchecks = build_stage7_sumchecks(
        hamming_dimensions,
        precommitted,
        &stage6b.output_points,
        Some((stage4, stage6b)),
    )?;
    let challenges = sumchecks.draw_challenges(transcript)?;

    let inputs = stage7_input_values_from_upstream(&sumchecks, stage6b)?;
    let input_points = sumchecks.empty_input_points();

    let mut recorder = ClearSumcheckRecorder::<F, C>::new();
    let (batch, coefficients) =
        sumchecks.begin_batch(&inputs, &challenges, &mut recorder, transcript)?;

    let mut hamming = backend.hamming_weight_claim_reduction.prepare(
        session,
        hamming_dimensions,
        booleanity_r_cycle,
        booleanity_r_address,
        &virtualization_points,
        &challenges.hamming_weight_claim_reduction,
        witness,
    )?;

    // The advice address phases: the stage-6b kernel objects, transitioned.
    // A member joins exactly when the batch declares it (layout committed AND
    // active address rounds) — the kernel exists whenever the layout does, so
    // absence here just leaves a cycle-completed kernel behind.
    let mut trusted_advice = match (&sumchecks.trusted_advice, trusted_advice_member.as_mut()) {
        (Some(_), Some(member)) => {
            member.transition_to_address_phase();
            Some(member)
        }
        (Some(_), None) => {
            return Err(ProverError::Unsupported {
                reason: "stage 6b carried no trusted-advice kernel for the scheduled address phase",
            });
        }
        (None, _) => None,
    };
    let mut untrusted_advice = match (
        &sumchecks.untrusted_advice,
        untrusted_advice_member.as_mut(),
    ) {
        (Some(_), Some(member)) => {
            member.transition_to_address_phase();
            Some(member)
        }
        (Some(_), None) => {
            return Err(ProverError::Unsupported {
                reason:
                    "stage 6b carried no untrusted-advice kernel for the scheduled address phase",
            });
        }
        (None, _) => None,
    };

    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![&mut *hamming];
    if let Some(member) = trusted_advice.as_mut() {
        members.push(&mut ***member);
    }
    if let Some(member) = untrusted_advice.as_mut() {
        members.push(&mut ***member);
    }
    let proved = prove_batch(&batch, &mut members, &mut recorder, transcript)?;

    let output_points = sumchecks.derive_opening_points(&proved.challenges, &input_points)?;
    let output_values = Stage7OutputClaims {
        hamming_weight_claim_reduction: hamming.output_claims()?,
        trusted_advice: trusted_advice
            .as_ref()
            .map(|member| {
                Ok::<_, ProverError<F>>(TrustedAdviceAddressPhaseOutputClaims {
                    trusted: member.final_claim()?,
                })
            })
            .transpose()?,
        untrusted_advice: untrusted_advice
            .as_ref()
            .map(|member| {
                Ok::<_, ProverError<F>>(UntrustedAdviceAddressPhaseOutputClaims {
                    untrusted: member.final_claim()?,
                })
            })
            .transpose()?,
        bytecode_address_phase: None,
        program_image_address_phase: None,
    };
    sumchecks.validate_output_claims(&output_values)?;
    let expected = sumchecks.expected_final_claim(
        &coefficients,
        &input_points,
        &output_values,
        &output_points,
        &challenges,
    )?;
    if expected != proved.final_claim {
        return Err(ProverError::FinalClaimMismatch {
            stage: "stage7",
            expected,
            got: proved.final_claim,
        });
    }

    let recorded = recorder.finish(&sumchecks.opening_values(&output_values), transcript)?;

    Ok(Stage7ProverOutput {
        sumcheck_proof: recorded.proof,
        claims: output_values.clone(),
        clear_output: Stage7ClearOutput {
            output_values,
            output_points,
        },
    })
}
