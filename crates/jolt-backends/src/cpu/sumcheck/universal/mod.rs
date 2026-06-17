//! Adapters from legacy stage-specific sumcheck traits to
//! [`jolt_sumcheck_prover::SumcheckBackend`].

mod stage4;
mod regular_batch;

pub use regular_batch::PreMaterializedRegularBatchBackend;
pub use stage4::PreMaterializedStage4Backend;
