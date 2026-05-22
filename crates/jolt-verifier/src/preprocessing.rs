//! Verifier preprocessing inputs.

use jolt_crypto::{DeriveSetup, VectorCommitment};
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::JoltProgramPreprocessing;

#[derive(Clone)]
pub struct JoltVerifierPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub program: JoltProgramPreprocessing,
    pub preprocessing_digest: [u8; 32],
    pub pcs_setup: PCS::VerifierSetup,
    pub vc_setup: Option<VC::Setup>,
}

impl<PCS, VC> JoltVerifierPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub fn new(
        program: JoltProgramPreprocessing,
        preprocessing_digest: [u8; 32],
        pcs_setup: PCS::VerifierSetup,
        vc_setup: Option<VC::Setup>,
    ) -> Self {
        Self {
            program,
            preprocessing_digest,
            pcs_setup,
            vc_setup,
        }
    }

    /// Reuses the PCS setup source to derive the vector-commitment setup.
    pub fn from_pcs_prover_setup(
        program: JoltProgramPreprocessing,
        preprocessing_digest: [u8; 32],
        pcs_prover_setup: &PCS::ProverSetup,
        vc_capacity: usize,
    ) -> Self
    where
        VC::Setup: DeriveSetup<PCS::ProverSetup>,
    {
        Self {
            program,
            preprocessing_digest,
            pcs_setup: PCS::verifier_setup(pcs_prover_setup),
            vc_setup: Some(VC::Setup::derive(pcs_prover_setup, vc_capacity)),
        }
    }
}
