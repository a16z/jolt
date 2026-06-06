//! Concrete Dory implementation of the generic PCS-assist verifier boundary.

pub mod artifacts;
pub mod config;
pub mod error;
pub mod native_final;
pub mod proof;
pub mod setup;
pub mod stages;
pub mod verifier;

pub use config::DoryAssistConfig;
pub use error::{DoryAssistStage, DoryAssistVerifierError};
pub use proof::{
    DoryAssistDoryReduceClaims, DoryAssistDoryReducePublicClaims,
    DoryAssistDoryReduceScalarFoldClaims, DoryAssistG1AdditionClaims, DoryAssistG1Claims,
    DoryAssistG1CoordinateClaims, DoryAssistG1PointClaims, DoryAssistG1PublicClaims,
    DoryAssistG1ScalarMultiplicationBoundaryClaims,
    DoryAssistG1ScalarMultiplicationBoundaryPublicClaims, DoryAssistG1ScalarMultiplicationClaims,
    DoryAssistG1ScalarMultiplicationShiftClaims, DoryAssistG2AdditionClaims, DoryAssistG2Claims,
    DoryAssistG2CoordinateClaims, DoryAssistG2PointClaims, DoryAssistG2PublicClaims,
    DoryAssistG2ScalarMultiplicationBoundaryClaims,
    DoryAssistG2ScalarMultiplicationBoundaryPublicClaims, DoryAssistG2ScalarMultiplicationClaims,
    DoryAssistG2ScalarMultiplicationShiftClaims, DoryAssistGtExponentiationBasePowerClaims,
    DoryAssistGtExponentiationBoundaryClaims, DoryAssistGtExponentiationBoundaryPublicClaims,
    DoryAssistGtExponentiationClaims, DoryAssistGtExponentiationDigitBitnessClaims,
    DoryAssistGtExponentiationDigitSelectorClaims, DoryAssistGtExponentiationShiftClaims,
    DoryAssistGtMultiplicationClaims, DoryAssistGtMultiplicationOpeningClaims,
    DoryAssistGtMultiplicationRowClaims, DoryAssistInputPublicClaims, DoryAssistOpeningClaim,
    DoryAssistOpeningClaims, DoryAssistProof, DoryAssistProofClaims, DoryAssistPublicOutputs,
    DoryAssistStage1Claims, DoryAssistStage1PublicClaims,
};
pub use setup::{
    derive_hyrax_prover_setup, derive_hyrax_verifier_setup, DoryAssistHyrax,
    DoryAssistHyraxProverSetup, DoryAssistHyraxVerifierSetup, DORY_ASSIST_HYRAX_GRUMPKIN_DOMAIN,
    DORY_ASSIST_HYRAX_GRUMPKIN_SEED, DORY_ASSIST_HYRAX_GRUMPKIN_SETUP_SEED,
};
pub use stages::{
    stage1::Stage1Proof as DoryAssistStage1Proof, stage2::Stage2Proof as DoryAssistStage2Proof,
    stage3::Stage3Proof as DoryAssistStage3Proof, DoryAssistStageProofs,
};
pub use verifier::{
    checked_clear_inputs, checked_zk_inputs, verify_clear, verify_zk, CheckedInputs, ClearInputs,
    ClearOpeningStatement, DoryAssist, ZkInputs, ZkOpeningStatement,
};
