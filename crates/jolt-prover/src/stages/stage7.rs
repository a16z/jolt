//! Stage 7: the Hamming-weight claim-reduction batch plus the present
//! precommitted address phases (advice, committed bytecode, program image).
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
use jolt_kernels::precommitted_reduction::PrecommittedReductionProver;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{ClearSumcheckRecorder, SumcheckProof};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::stage4::Stage4ClearOutput;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::advice_address_phase::{
    TrustedAdviceAddressPhaseOutputClaims, UntrustedAdviceAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage7::committed_reduction_address_phase::{
    BytecodeReductionAddressPhaseOutputClaims, ProgramImageReductionAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage7::outputs::Stage7ExternalMembers;
use jolt_verifier::stages::stage7::outputs::{Stage7ClearOutput, Stage7OutputClaims};
use jolt_verifier::stages::stage7::{build_stage7_sumchecks, stage7_input_values_from_upstream};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use super::precommitted::{scalar_phase_adapter, PrecommittedKernelAdapter};
use crate::{BackendPreparer, JoltProverPreprocessing, ProverConfig, ProverError};

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
    mut trusted_advice_member: Option<Box<dyn PrecommittedReductionProver<F>>>,
    mut untrusted_advice_member: Option<Box<dyn PrecommittedReductionProver<F>>>,
    mut bytecode_reduction_member: Option<Box<dyn PrecommittedReductionProver<F>>>,
    mut program_image_member: Option<Box<dyn PrecommittedReductionProver<F>>>,
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

    let sumchecks = build_stage7_sumchecks(
        hamming_dimensions,
        precommitted,
        &stage6b.output_points,
        Some((stage4, stage6b)),
    )?;
    let challenges = sumchecks.draw_challenges(transcript)?;

    let inputs = stage7_input_values_from_upstream(&sumchecks, stage6b)?;
    let input_points = sumchecks.empty_input_points();

    // The precommitted address phases: the stage-6b kernel objects,
    // transitioned. A member joins exactly when the batch declares it (layout
    // committed AND active address rounds) -- the kernel exists whenever the
    // layout does, so absence here just leaves a cycle-completed kernel
    // behind. Each joins the generated driver as a typed external member
    // whose extraction is the phase's final opening.
    let trusted_advice = take_scheduled(
        sumchecks.trusted_advice.is_some(),
        trusted_advice_member.as_mut(),
        "stage 6b carried no trusted-advice kernel for the scheduled address phase",
    )?;
    let untrusted_advice = take_scheduled(
        sumchecks.untrusted_advice.is_some(),
        untrusted_advice_member.as_mut(),
        "stage 6b carried no untrusted-advice kernel for the scheduled address phase",
    )?;
    let bytecode_reduction = take_scheduled(
        sumchecks.bytecode_address_phase.is_some(),
        bytecode_reduction_member.as_mut(),
        "stage 6b carried no bytecode kernel for the scheduled address phase",
    )?;
    let program_image = take_scheduled(
        sumchecks.program_image_address_phase.is_some(),
        program_image_member.as_mut(),
        "stage 6b carried no program-image kernel for the scheduled address phase",
    )?;

    let mut trusted_advice = trusted_advice.map(|member| {
        scalar_phase_adapter(&mut **member, false, |trusted| {
            TrustedAdviceAddressPhaseOutputClaims { trusted }
        })
    });
    let mut untrusted_advice = untrusted_advice.map(|member| {
        scalar_phase_adapter(&mut **member, false, |untrusted| {
            UntrustedAdviceAddressPhaseOutputClaims { untrusted }
        })
    });
    let mut bytecode_reduction = bytecode_reduction.map(|member| {
        PrecommittedKernelAdapter::new(
            &mut **member,
            |member: &dyn PrecommittedReductionProver<F>| {
                Ok(BytecodeReductionAddressPhaseOutputClaims {
                    chunks: member.final_aux_claims()?,
                })
            },
        )
    });
    let mut program_image = program_image.map(|member| {
        scalar_phase_adapter(&mut **member, false, |program_image| {
            ProgramImageReductionAddressPhaseOutputClaims { program_image }
        })
    });

    let mut preparer = BackendPreparer {
        backend,
        session,
        witness,
        context: (),
    };
    let proved = sumchecks.prove_clear(
        &mut preparer,
        &inputs,
        &input_points,
        &challenges,
        Stage7ExternalMembers {
            trusted_advice: trusted_advice.as_mut().map(|adapter| adapter as _),
            untrusted_advice: untrusted_advice.as_mut().map(|adapter| adapter as _),
            bytecode_address_phase: bytecode_reduction.as_mut().map(|adapter| adapter as _),
            program_image_address_phase: program_image.as_mut().map(|adapter| adapter as _),
        },
        ClearSumcheckRecorder::<F, C>::new(),
        transcript,
    )?;

    Ok(Stage7ProverOutput {
        sumcheck_proof: proved.recorded.proof,
        claims: proved.output_claims.clone(),
        clear_output: Stage7ClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
        },
    })
}

/// Transition a stage-6b precommitted kernel into its scheduled stage-7
/// address phase: the batch declaring the phase without a carried kernel is a
/// cross-stage invariant violation; an undeclared phase leaves the
/// cycle-completed kernel behind.
fn take_scheduled<'a, F: Field>(
    scheduled: bool,
    member: Option<&'a mut Box<dyn PrecommittedReductionProver<F>>>,
    missing: &'static str,
) -> Result<Option<&'a mut Box<dyn PrecommittedReductionProver<F>>>, ProverError<F>> {
    match (scheduled, member) {
        (true, Some(member)) => {
            member.transition_to_address_phase();
            Ok(Some(member))
        }
        (true, None) => Err(ProverError::InvariantViolation { reason: missing }),
        (false, _) => Ok(None),
    }
}
