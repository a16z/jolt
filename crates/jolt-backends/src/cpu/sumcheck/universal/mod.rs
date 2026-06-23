//! Adapters from legacy stage-specific sumcheck traits to
//! [`jolt_sumcheck_prover::SumcheckBackend`].

mod regular_batch;
mod stage4;

pub use regular_batch::PreMaterializedRegularBatchBackend;
pub use stage4::PreMaterializedStage4Backend;
