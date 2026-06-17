//! Verifier preprocessing inputs.

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::formulas::bytecode::FieldInlineBytecodeRow;
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
    #[cfg(feature = "field-inline")]
    pub field_inline_bytecode: Option<Vec<FieldInlineBytecodeRow>>,
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
            #[cfg(feature = "field-inline")]
            field_inline_bytecode: None,
            pcs_setup,
            vc_setup,
        }
    }

    #[cfg(feature = "field-inline")]
    pub fn with_field_inline_bytecode(mut self, bytecode: Vec<FieldInlineBytecodeRow>) -> Self {
        self.field_inline_bytecode = Some(bytecode);
        self
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
            #[cfg(feature = "field-inline")]
            field_inline_bytecode: None,
            pcs_setup: PCS::verifier_setup(pcs_prover_setup),
            vc_setup: Some(VC::Setup::derive(pcs_prover_setup, vc_capacity)),
        }
    }
}
