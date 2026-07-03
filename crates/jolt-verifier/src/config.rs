//! Jolt verifier configuration: the protocol axes ([`JoltProtocolConfig`]:
//! zk × commitment), fixed at compile time. A proof self-describes its axes
//! and [`validate_proof_config`] rejects a mismatch.

use jolt_openings::CommitmentScheme;
use jolt_transcript::{AppendToTranscript, Label, Transcript, U64Word};
use serde::{Deserialize, Serialize};

use crate::{proof::JoltProof, verifier::CheckedInputs, VerifierError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkConfig {
    Transparent,
    BlindFold,
}

impl ZkConfig {
    /// Fiat-Shamir encoding; consensus-critical, must match the prover's
    /// preamble scalar.
    pub const fn transcript_scalar(self) -> u64 {
        match self {
            Self::Transparent => 0,
            Self::BlindFold => 1,
        }
    }
}

/// The commitment axis of the protocol: how committed polynomials are
/// discharged at the final opening.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentConfig {
    /// Per-polynomial commitments, RLC batch opening (requires additive
    /// homomorphism).
    Homomorphic,
    /// One packed witness per commitment lifecycle, reduction-sumcheck batch
    /// opening (no homomorphism; the Akita path).
    Packed,
}

impl CommitmentConfig {
    /// Fiat-Shamir encoding; consensus-critical, must match the prover's
    /// preamble scalar.
    pub const fn transcript_scalar(self) -> u64 {
        match self {
            Self::Homomorphic => 0,
            Self::Packed => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolConfig {
    pub zk: ZkConfig,
    pub commitment: CommitmentConfig,
}

impl AppendToTranscript for JoltProtocolConfig {
    /// WARNING: the byte layout is consensus-critical — it must stay
    /// identical to the tail of jolt-prover-legacy's `fiat_shamir_preamble`,
    /// which mirrors these scalars through its own transcript API.
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"zk_config"));
        transcript.append(&U64Word(self.zk.transcript_scalar()));
        transcript.append(&Label(b"commitment_config"));
        transcript.append(&U64Word(self.commitment.transcript_scalar()));
    }
}

/// The protocol config this build verifies, fixed at compile time by the
/// `zk` feature; proofs self-describe theirs and [`validate_proof_config`]
/// rejects a mismatch.
pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = JoltProtocolConfig {
    zk: if cfg!(feature = "zk") {
        ZkConfig::BlindFold
    } else {
        ZkConfig::Transparent
    },
    commitment: CommitmentConfig::Homomorphic,
};

pub fn validate_proof_config<PCS, VC, ZkProof, JointOpeningProof, Commitments, M>(
    config: &JoltProtocolConfig,
    proof: &JoltProof<PCS, VC, ZkProof, JointOpeningProof, Commitments, M>,
) -> Result<(), VerifierError>
where
    M: jolt_claims::protocols::jolt::JoltCommitmentMode,
    PCS: CommitmentScheme,
    VC: jolt_crypto::VectorCommitment<Field = PCS::Field>,
{
    if proof.protocol != *config {
        return Err(VerifierError::ProtocolConfigMismatch {
            expected: *config,
            got: proof.protocol,
        });
    }

    Ok(())
}

/// Fail-closed validation of the packed commitment axis. The packed
/// lifecycle is exactly one per-proof witness, so the committed-program and
/// advice commitment objects (and the zk path) are rejected until their
/// packed counterparts land.
pub fn validate_packed_inputs(checked: &CheckedInputs) -> Result<(), VerifierError> {
    if checked.zk {
        return Err(VerifierError::PackedRequiresTransparentProof);
    }
    if checked.precommitted.bytecode.is_some() || checked.precommitted.program_image.is_some() {
        return Err(VerifierError::PackedRequiresFullProgramMode);
    }
    if checked.precommitted.trusted_advice.is_some()
        || checked.precommitted.untrusted_advice.is_some()
    {
        return Err(VerifierError::PackedRequiresNoAdvice);
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::{CommittedProgramSchedule, PrecommittedSchedule};
    use common::jolt_device::JoltDevice;
    use jolt_claims::protocols::jolt::TracePolynomialOrder;

    fn checked_inputs(zk: bool, precommitted: PrecommittedSchedule) -> CheckedInputs {
        CheckedInputs {
            public_io: JoltDevice::default(),
            zk,
            trace_length: 1 << 8,
            ram_K: 1 << 10,
            entry_address: 0,
            preprocessing_digest: [0u8; 32],
            trusted_advice_commitment_present: false,
            vc_capacity: None,
            precommitted,
        }
    }

    fn schedule(
        trusted_advice: Option<usize>,
        untrusted_advice: Option<usize>,
        committed_program: Option<CommittedProgramSchedule>,
    ) -> PrecommittedSchedule {
        PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            8,
            8,
            trusted_advice,
            untrusted_advice,
            committed_program,
        )
        .unwrap()
    }

    fn committed_program() -> CommittedProgramSchedule {
        CommittedProgramSchedule {
            bytecode_len: 1 << 6,
            bytecode_chunk_count: 1,
            program_image_len_words: 1 << 6,
            program_image_start_index: 0,
        }
    }

    #[test]
    fn packed_accepts_full_mode_without_advice() {
        let checked = checked_inputs(false, schedule(None, None, None));
        assert!(validate_packed_inputs(&checked).is_ok());
    }

    #[test]
    fn packed_rejects_zk() {
        let checked = checked_inputs(true, schedule(None, None, None));
        assert!(matches!(
            validate_packed_inputs(&checked),
            Err(VerifierError::PackedRequiresTransparentProof)
        ));
    }

    #[test]
    fn packed_rejects_committed_program() {
        let checked = checked_inputs(false, schedule(None, None, Some(committed_program())));
        assert!(matches!(
            validate_packed_inputs(&checked),
            Err(VerifierError::PackedRequiresFullProgramMode)
        ));
    }

    #[test]
    fn packed_rejects_advice() {
        let checked = checked_inputs(false, schedule(Some(1 << 12), None, None));
        assert!(matches!(
            validate_packed_inputs(&checked),
            Err(VerifierError::PackedRequiresNoAdvice)
        ));

        let checked = checked_inputs(false, schedule(None, Some(1 << 12), None));
        assert!(matches!(
            validate_packed_inputs(&checked),
            Err(VerifierError::PackedRequiresNoAdvice)
        ));
    }
}
