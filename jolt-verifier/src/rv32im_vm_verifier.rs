use jolt_core::field::JoltField;
use jolt_core::jolt::vm::{JoltCommon, JoltVerifier};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::r1cs::constraints::JoltRV32IMConstraints;
use jolt_core::utils::transcript::Transcript;

const WORD_SIZE: usize = 32;

pub enum RV32IMJoltVMVerifier {}

impl<F, PCS, ProofTranscript> JoltCommon<WORD_SIZE, F, PCS, ProofTranscript>
    for RV32IMJoltVMVerifier
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
}

impl<F, PCS, ProofTranscript> JoltVerifier<WORD_SIZE, F, PCS, ProofTranscript>
    for RV32IMJoltVMVerifier
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Constraints = JoltRV32IMConstraints;
}
