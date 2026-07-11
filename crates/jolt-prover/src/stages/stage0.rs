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
use jolt_kernels::{CommitmentGrid, JoltBackend, ProofSession, WitnessCommitment};
use jolt_openings::CommitmentScheme;
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::JoltCommitments;
use jolt_verifier::{
    absorb_transcript_commitments, absorb_transcript_preamble, validate_inputs_from_parts,
    CheckedInputs, ProofTranscriptConfig,
};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::CommittedWitnessProvider;

use crate::config::advice_total_vars;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

/// The externally supplied trusted-advice commitment (produced at
/// preprocessing time, before any proving) and its opening hint. Mirrors
/// legacy's prover-constructor pair: the commitment is absorbed in stage 0 and
/// batched in stage 8, and the hint joins the stage-8 hint combination.
pub struct TrustedAdviceCommitment<PCS: CommitmentScheme> {
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// Stage 0's outputs: the validated inputs, the seeded transcript (positioned
/// exactly where the verifier's `verify_until_stage1` leaves its own), the
/// witness commitments in wire form, the untrusted-advice commitment (proved
/// at prove time, carried on the proof), and the per-polynomial opening hints
/// the stage-8 joint opening will consume (advice hints included).
pub struct Stage0Output<PCS, T>
where
    PCS: CommitmentScheme,
{
    pub checked: CheckedInputs,
    pub transcript: T,
    pub commitments: JoltCommitments<PCS::Output>,
    pub untrusted_advice_commitment: Option<PCS::Output>,
    pub hints: Vec<(JoltCommittedPolynomial, PCS::OpeningHint)>,
}

/// Validate inputs, seed the transcript, commit the witness (the untrusted
/// advice polynomial in its own balanced grid), and absorb the commitments
/// (main, then untrusted advice, then trusted advice — the verifier's own
/// absorb order). Committed-program mode is not yet supported.
pub fn prove_stage0<F, PCS, VC, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&TrustedAdviceCommitment<PCS>>,
    witness: &dyn CommittedWitnessProvider<F, JoltVmNamespace>,
    public_io: &JoltDevice,
) -> Result<Stage0Output<PCS, T>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    if preprocessing.verifier.program.committed().is_some() {
        return Err(ProverError::Unsupported {
            reason: "committed-program mode is not yet supported by the modular prover",
        });
    }
    let untrusted_advice_present = !public_io.untrusted_advice.is_empty();
    // Trusted-advice presence rides on the external commitment argument;
    // require it to agree with the advice bytes so a mismatch fails here
    // rather than as an opaque stage-4 sumcheck error (bytes without a
    // commitment) or as a nonstandard proof over the zero advice polynomial
    // (a commitment without bytes).
    if trusted_advice.is_some() == public_io.trusted_advice.is_empty() {
        return Err(ProverError::Unsupported {
            reason: "trusted-advice commitment presence disagrees with the trusted advice bytes",
        });
    }
    // The dominant-advice regime (an advice grid wider than the main
    // commitment grid) changes the main commit layout and the reduction
    // permutations; legacy supports it through a separate materialized commit
    // path with no e2e coverage, and the reference streaming commit does not
    // reproduce it. Guard it off like the other unverified modes.
    let main_total_vars =
        config.one_hot_config.committed_chunk_bits() + config.trace_length.ilog2() as usize;
    let advice_dominates = |max_size: u64| advice_total_vars(max_size) > main_total_vars;
    if (trusted_advice.is_some()
        && advice_dominates(public_io.memory_layout.max_trusted_advice_size))
        || (untrusted_advice_present
            && advice_dominates(public_io.memory_layout.max_untrusted_advice_size))
    {
        return Err(ProverError::Unsupported {
            reason: "dominant advice (advice grid wider than the main commitment grid) is not yet supported",
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
        trusted_advice.is_some(),
        untrusted_advice_present,
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
        total_vars: config.commitment_total_vars(
            &public_io.memory_layout,
            trusted_advice.is_some(),
            untrusted_advice_present,
        ),
        log_t: config.trace_length.ilog2() as usize,
    };
    let committed =
        backend
            .commit
            .commit_witness(session, witness, &ids, grid, &preprocessing.pcs_setup)?;
    let (commitments, mut hints) = assemble_commitments::<PCS>(committed)?;

    // The untrusted advice polynomial is committed at prove time in its OWN
    // balanced grid (its variable count comes from the memory layout's maximum
    // advice size, independent of the main grid); the trusted commitment
    // arrived from preprocessing.
    let untrusted_advice_commitment = if untrusted_advice_present {
        let advice_grid = CommitmentGrid {
            total_vars: advice_total_vars(public_io.memory_layout.max_untrusted_advice_size),
            log_t: 0,
        };
        let mut advice = backend.commit.commit_witness(
            session,
            witness,
            &[JoltCommittedPolynomial::UntrustedAdvice],
            advice_grid,
            &preprocessing.pcs_setup,
        )?;
        let advice = advice.pop().ok_or(ProverError::Unsupported {
            reason: "the commit slot produced no untrusted-advice commitment",
        })?;
        hints.push((advice.id, advice.hint));
        Some(advice.commitment)
    } else {
        None
    };
    if let Some(trusted) = trusted_advice {
        hints.push((JoltCommittedPolynomial::TrustedAdvice, trusted.hint.clone()));
    }

    absorb_transcript_commitments(
        &commitments,
        untrusted_advice_commitment.as_ref(),
        trusted_advice.map(|trusted| &trusted.commitment),
        &mut transcript,
    );

    Ok(Stage0Output {
        checked,
        transcript,
        commitments,
        untrusted_advice_commitment,
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
