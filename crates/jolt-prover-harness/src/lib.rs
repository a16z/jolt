//! Temporary migration harness for the modular Jolt prover.
//!
//! This crate is dev/test infrastructure. It may depend on `jolt-core` and
//! concrete backends while the modular prover is being ported. Production
//! crates must not depend on it.

pub mod checkpoint;
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
pub mod core_fixture;
#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
pub mod core_zk_fixture;
pub mod fixtures;
pub mod ingest;
pub mod manifest;
pub mod matrix;
pub mod metrics;
pub mod optimization;
pub mod parity;
pub mod perf;
pub mod sdk_fixture;

#[cfg(feature = "field-inline")]
pub mod field_inline;

pub use checkpoint::{
    BlindFoldCheckpoint, CommitmentCheckpoint, FrontierCheckpoint, NamedValue, OpeningCheckpoint,
    StageCheckpoint,
};
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
pub use core_fixture::{
    load_core_verifier_fixture, load_stage0_advice_commitment_kernel_benchmark_fixture,
    load_stage0_commitment_kernel_benchmark_fixture,
    load_stage0_commitment_verifier_replay_fixture, load_stage1_spartan_outer_checkpoint_fixture,
    load_stage1_spartan_outer_kernel_benchmark_fixture,
    load_stage1_spartan_outer_verifier_replay_fixture,
    load_stage2_instruction_claim_opening_checkpoint_fixture,
    load_stage2_product_remainder_opening_checkpoint_fixture,
    load_stage2_product_uniskip_kernel_benchmark_fixture,
    load_stage2_product_uniskip_verifier_replay_fixture,
    load_stage2_ram_read_write_opening_checkpoint_fixture,
    load_stage2_ram_terminal_opening_checkpoint_fixture,
    load_stage2_regular_batch_input_checkpoint_fixture,
    load_stage2_regular_batch_input_kernel_benchmark_fixture,
    load_stage2_regular_batch_sumcheck_kernel_benchmark_fixture,
    load_stage2_regular_batch_verifier_replay_fixture,
    load_stage3_output_opening_checkpoint_fixture,
    load_stage3_regular_batch_input_checkpoint_fixture,
    load_stage3_regular_batch_input_kernel_benchmark_fixture,
    load_stage3_regular_batch_sumcheck_kernel_benchmark_fixture,
    load_stage3_regular_batch_verifier_replay_fixture,
    load_stage4_output_opening_checkpoint_fixture,
    load_stage4_regular_batch_input_checkpoint_fixture,
    load_stage4_regular_batch_input_kernel_benchmark_fixture,
    load_stage4_regular_batch_sumcheck_kernel_benchmark_fixture,
    load_stage4_regular_batch_verifier_replay_fixture,
    load_stage5_output_opening_checkpoint_fixture,
    load_stage5_regular_batch_input_checkpoint_fixture,
    load_stage5_regular_batch_input_kernel_benchmark_fixture,
    load_stage5_regular_batch_sumcheck_kernel_benchmark_fixture,
    load_stage5_regular_batch_verifier_replay_fixture,
    load_stage6_output_opening_checkpoint_fixture,
    load_stage6_regular_batch_input_checkpoint_fixture,
    load_stage6_regular_batch_input_kernel_benchmark_fixture,
    load_stage6_regular_batch_sumcheck_kernel_benchmark_fixture,
    load_stage6_regular_batch_verifier_replay_fixture,
    load_stage7_regular_batch_input_checkpoint_fixture,
    load_stage7_regular_batch_input_kernel_benchmark_fixture,
    load_stage7_regular_batch_sumcheck_kernel_benchmark_fixture,
    load_stage7_regular_batch_verifier_replay_fixture, load_stage8_joint_opening_replay_fixture,
    load_stage8_opening_structure_checkpoint_fixture,
    load_stage8_ra_constituent_checkpoint_fixture, load_top_level_clear_prover_fixture,
    CoreVerifierFixture, Stage0AdviceCommitmentKernelBenchmarkFixture,
    Stage0AdviceCommitmentKernelShape, Stage0CommitmentKernelBenchmarkFixture,
    Stage0CommitmentKernelShape, Stage1SpartanOuterCheckpoint,
    Stage1SpartanOuterKernelBenchmarkFixture, Stage2InstructionClaimOpeningCheckpoint,
    Stage2ProductRemainderOpeningCheckpoint, Stage2ProductUniskipKernelBenchmarkFixture,
    Stage2RamReadWriteOpeningCheckpoint, Stage2RamTerminalOpeningCheckpoint,
    Stage2RegularBatchInputCheckpoint, Stage2RegularBatchInputKernelBenchmarkFixture,
    Stage2RegularBatchSumcheckExpected, Stage2RegularBatchSumcheckKernelBenchmarkFixture,
    Stage2RegularBatchVerifierReplayCheckpoint, Stage3OutputOpeningCheckpoint,
    Stage3RegularBatchInputCheckpoint, Stage3RegularBatchInputKernelBenchmarkFixture,
    Stage3RegularBatchSumcheckExpected, Stage3RegularBatchSumcheckKernelBenchmarkFixture,
    Stage3RegularBatchVerifierReplayCheckpoint, Stage4OutputOpeningCheckpoint,
    Stage4RegularBatchInputCheckpoint, Stage4RegularBatchInputKernelBenchmarkFixture,
    Stage4RegularBatchSumcheckExpected, Stage4RegularBatchSumcheckKernelBenchmarkFixture,
    Stage4RegularBatchVerifierReplayCheckpoint, Stage5OutputOpeningCheckpoint,
    Stage5RegularBatchInputCheckpoint, Stage5RegularBatchInputKernelBenchmarkFixture,
    Stage5RegularBatchSumcheckExpected, Stage5RegularBatchSumcheckKernelBenchmarkFixture,
    Stage5RegularBatchVerifierReplayCheckpoint, Stage6OutputOpeningCheckpoint,
    Stage6RegularBatchInputCheckpoint, Stage6RegularBatchInputKernelBenchmarkFixture,
    Stage6RegularBatchSumcheckExpected, Stage6RegularBatchSumcheckKernelBenchmarkFixture,
    Stage6RegularBatchVerifierReplayCheckpoint, Stage7RegularBatchInputCheckpoint,
    Stage7RegularBatchInputKernelBenchmarkFixture, Stage7RegularBatchSumcheckExpected,
    Stage7RegularBatchSumcheckKernelBenchmarkFixture, Stage7RegularBatchVerifierReplayCheckpoint,
    Stage8JointOpeningReplayCheckpoint, Stage8OpeningStructureCheckpoint,
    Stage8RaConstituentCheckpoint, TopLevelClearProverFixture,
};
#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
pub use core_zk_fixture::{load_zk_core_verifier_fixture, CoreZkVerifierFixture};
pub use fixtures::{
    CoreFixtureProvider, FixtureArtifacts, FixtureKind, FixtureProvider, FixtureRequest,
    FixtureSource, StaticFixtureProvider,
};
pub use ingest::{IngestionSurface, ProgramArtifactKind, ProverInputDescriptor};
pub use manifest::{
    registered_frontiers, FeatureMode, FrontierGate, FrontierManifest, FrontierSpec,
};
pub use metrics::{FrontierMetrics, RunMetrics};
pub use optimization::{
    registered_backend_kernel_ports, validate_frontier_kernel_accounting,
    validate_frontier_optimization_ids, validate_global_cpu_backend_inventory_coverage,
    BackendKernelFamily, BackendKernelPortLedger, BackendKernelPortSpec, KernelPortStatus,
    KnownOptimizationIds,
};
pub use parity::{compare_named_values, ComparisonTarget, ParityMismatch, ParityReport};
pub use perf::{
    evaluate_perf, kernel_benchmark_evidence_path, validate_frontier_replacement_ready,
    validate_kernel_benchmark_evidence, validate_parity_certified_kernel_evidence,
    validate_parity_certified_kernel_evidence_files, GateStatus, KernelBenchmarkEvidence,
    KernelMemoryBudget, PerfEvaluation, PerfGate,
};
pub use sdk_fixture::{trace_sdk_guest, SdkGuestTraceFixture, SdkGuestTraceRequest};

use thiserror::Error;

pub type HarnessResult<T> = Result<T, HarnessError>;

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error("invalid frontier manifest entry `{frontier}`: {reason}")]
    InvalidManifest {
        frontier: &'static str,
        reason: String,
    },
    #[error("fixture `{fixture:?}` is not available for `{context}`")]
    FixtureUnavailable {
        fixture: FixtureKind,
        context: &'static str,
    },
    #[error("core fixture `{fixture:?}` failed during `{context}`: {reason}")]
    CoreFixture {
        fixture: FixtureKind,
        context: &'static str,
        reason: String,
    },
    #[error("invalid prover input from `{surface}`: {reason}")]
    InvalidIngestion { surface: String, reason: String },
    #[error("feature `{feature}` is required for `{context}`")]
    MissingFeature {
        feature: &'static str,
        context: &'static str,
    },
    #[error("invalid optimization inventory: {reason}")]
    InvalidOptimizationInventory { reason: String },
    #[error("invalid backend kernel inventory entry `{kernel}`: {reason}")]
    InvalidKernelInventory {
        kernel: &'static str,
        reason: String,
    },
    #[error("invalid benchmark evidence for `{kernel}` / `{benchmark}`: {reason}")]
    InvalidBenchmarkEvidence {
        kernel: String,
        benchmark: String,
        reason: String,
    },
    #[error("parity target `{target}` failed: {reason}")]
    Parity { target: String, reason: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
