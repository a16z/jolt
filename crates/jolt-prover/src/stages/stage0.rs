//! Stage 0: input validation, the Fiat-Shamir preamble, and witness
//! commitment.
//!
//! The transcript work is the verifier's own exported code
//! ([`validate_inputs_from_parts`], [`absorb_transcript_preamble`],
//! [`absorb_transcript_commitments`]) — the two sides share the absorb
//! sequence structurally, so stage-0 Fiat-Shamir drift is impossible by
//! construction. The commitment compute is delegated to the `jolt-kernels`
//! witness-commitment kernel; only the absorbs happen here.

use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::{commit_witness, CommitmentGrid, WitnessCommitment};
use jolt_openings::{CommitmentScheme, StreamingCommitment};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::JoltCommitments;
use jolt_verifier::{
    absorb_transcript_commitments, absorb_transcript_preamble, validate_inputs_from_parts,
    CheckedInputs, ProofTranscriptConfig,
};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::{CommittedWitnessProvider, WitnessProvider};

use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

/// Stage 0's outputs: the validated inputs, the seeded transcript (positioned
/// exactly where the verifier's `verify_until_stage1` leaves its own), the
/// witness commitments in wire form, and the per-polynomial opening hints the
/// stage-8 joint opening will consume.
pub struct Stage0Output<PCS, T>
where
    PCS: CommitmentScheme,
{
    pub checked: CheckedInputs,
    pub transcript: T,
    pub commitments: JoltCommitments<PCS::Output>,
    pub hints: Vec<(JoltCommittedPolynomial, PCS::OpeningHint)>,
}

/// Validate inputs, seed the transcript, commit the witness, and absorb the
/// commitments. Advice and committed-program modes are not yet supported.
pub fn prove_stage0<F, PCS, VC, T, W>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    witness: &W,
    public_io: &JoltDevice,
) -> Result<Stage0Output<PCS, T>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment,
    PCS::Output: AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
    W: WitnessProvider<F, JoltVmNamespace> + CommittedWitnessProvider<F, JoltVmNamespace>,
{
    if !public_io.trusted_advice.is_empty() || !public_io.untrusted_advice.is_empty() {
        return Err(ProverError::Unsupported {
            reason: "advice commitment is not yet supported by the modular prover",
        });
    }
    if preprocessing.verifier.program.committed().is_some() {
        return Err(ProverError::Unsupported {
            reason: "committed-program mode is not yet supported by the modular prover",
        });
    }

    // The verifier's own input validation doubles as the prover's self-check
    // and produces the normalized `CheckedInputs` the preamble absorbs.
    let checked = validate_inputs_from_parts(
        &preprocessing.verifier,
        public_io,
        config.trace_length,
        config.ram_K,
        config.trace_polynomial_order,
        config.one_hot_config,
        false,
        false,
        false,
    )?;

    let mut transcript = T::new(b"Jolt");
    absorb_transcript_preamble(
        &checked,
        ProofTranscriptConfig {
            rw_config: config.rw_config,
            one_hot_config: config.one_hot_config,
            trace_polynomial_order: config.trace_polynomial_order,
        },
        &mut transcript,
    );

    let ids: Vec<JoltCommittedPolynomial> = witness
        .committed_oracle_order()?
        .into_iter()
        .filter(|id| {
            !matches!(
                id,
                JoltCommittedPolynomial::TrustedAdvice | JoltCommittedPolynomial::UntrustedAdvice
            )
        })
        .collect();
    let grid = CommitmentGrid {
        total_vars: config.commitment_total_vars(&public_io.memory_layout, false, false),
        log_t: config.trace_length.ilog2() as usize,
    };
    let committed = commit_witness::<F, PCS, W>(witness, &ids, grid, &preprocessing.pcs_setup)?;

    let (commitments, hints) = assemble_commitments::<PCS>(committed)?;
    absorb_transcript_commitments(&commitments, None, None, &mut transcript);

    Ok(Stage0Output {
        checked,
        transcript,
        commitments,
        hints,
    })
}

/// Split the kernel's flat id-ordered output into the proof's wire shape.
#[expect(
    clippy::type_complexity,
    reason = "the wire aggregate paired with its opening hints"
)]
fn assemble_commitments<PCS: CommitmentScheme>(
    committed: Vec<WitnessCommitment<PCS>>,
) -> Result<
    (
        JoltCommitments<PCS::Output>,
        Vec<(JoltCommittedPolynomial, PCS::OpeningHint)>,
    ),
    ProverError<PCS::Field>,
> {
    let mut rd_inc = None;
    let mut ram_inc = None;
    let mut instruction = Vec::new();
    let mut ram = Vec::new();
    let mut bytecode = Vec::new();
    let mut hints = Vec::with_capacity(committed.len());

    for entry in committed {
        let WitnessCommitment {
            id,
            commitment,
            hint,
        } = entry;
        match id {
            JoltCommittedPolynomial::RdInc => rd_inc = Some(commitment),
            JoltCommittedPolynomial::RamInc => ram_inc = Some(commitment),
            JoltCommittedPolynomial::InstructionRa(_) => instruction.push(commitment),
            JoltCommittedPolynomial::RamRa(_) => ram.push(commitment),
            JoltCommittedPolynomial::BytecodeRa(_) => bytecode.push(commitment),
            other => {
                return Err(ProverError::Unsupported {
                    reason: match other {
                        JoltCommittedPolynomial::TrustedAdvice
                        | JoltCommittedPolynomial::UntrustedAdvice => {
                            "advice polynomials are absorbed separately, not as main commitments"
                        }
                        _ => "precommitted polynomials are not main witness commitments",
                    },
                });
            }
        }
        hints.push((id, hint));
    }

    let (Some(rd_inc), Some(ram_inc)) = (rd_inc, ram_inc) else {
        return Err(ProverError::Unsupported {
            reason: "witness did not produce the RdInc/RamInc commitments",
        });
    };
    Ok((
        JoltCommitments::new(rd_inc, ram_inc, instruction, ram, bytecode),
        hints,
    ))
}
