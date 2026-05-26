use jolt_backends::CommitmentBackend;
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_verifier::JoltProof;
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, CommittedWitnessProvider, WitnessProvider,
};

use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

pub fn prove<PCS, VC, B, W>(
    _preprocessing: &JoltProverPreprocessing<PCS, VC>,
    _witness: &W,
    _config: ProverConfig,
    _backend: &mut B,
) -> Result<JoltProof<PCS, VC>, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    B: CommitmentBackend<PCS::Field, JoltVmNamespace, PCS>,
    W: CommittedWitnessProvider<PCS::Field, JoltVmNamespace>
        + WitnessProvider<PCS::Field, JoltVmNamespace>,
{
    Err(ProverError::FrontierNotImplemented {
        frontier: "full Jolt proof",
    })
}
