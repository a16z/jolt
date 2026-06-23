//! Prover-side protocol orchestration helpers.

pub mod core;
pub mod extension_opening_reduction;
pub mod fold_grind;
pub mod fold_grind_observer;
#[cfg(feature = "zk")]
pub(crate) mod masking;
pub mod prg;
pub mod ring_relation;
pub mod ring_relation_witness;
pub mod ring_switch;
pub mod sumcheck;
#[cfg(feature = "zk")]
pub(crate) mod zk_hiding_commit;

pub use akita_types::RingRelationInstance;
pub use core::{
    batched_prove, prepare_batched_prove_inputs, prove, prove_root, prove_root_direct,
    prove_suffix, prove_terminal_root_fold_with_params, PreparedBatchedProveInputs,
    ProveLevelOutput, RecursiveSuffixOutcome, SuffixProverState,
};
pub use fold_grind::ProverTranscriptGrind;
pub use fold_grind_observer::{FoldGrindObservation, FoldGrindObserverGuard};
pub use ring_relation::{compute_relation_quotient, generate_y, RingRelationProver};
pub use ring_relation_witness::RingRelationWitness;
pub use ring_switch::{commit_next_w, RingSwitchOutput};
