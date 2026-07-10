//! The runtime seam: [`JoltBackend`] is the value `jolt-prover` proves
//! against — one boxed object-safe slot per kernel entry — and
//! [`ProofSession`] is the backend-owned state with proof lifetime. Swapping
//! a kernel implementation, mixing implementations per slot, running two
//! backends side by side, and choosing a configuration from the hardware are
//! all value construction, never compilation. See
//! `specs/clean-slate-prover.md`, "The backend seam".

use std::any::{Any, TypeId};
use std::collections::HashMap;

use jolt_field::Field;
use jolt_openings::{CommitmentScheme, StreamingCommitment};

use crate::commitment::CommitWitness;
use crate::instruction_claim_reduction::InstructionClaimReductionProver;
use crate::ram_output_check::RamOutputCheckProver;
use crate::ram_raf_evaluation::RamRafEvaluationProver;
use crate::ram_read_write::RamReadWriteProver;
use crate::spartan_outer::SpartanOuterProver;
use crate::spartan_product::SpartanProductProver;

/// The kernel registry: one independently swappable slot per kernel entry.
///
/// `F` and `PCS` are deployment constants, not swap targets — the PCS traits
/// are structurally non-object-safe and their associated types are wire
/// types, so they stay type parameters.
pub struct JoltBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    pub commit: Box<dyn CommitWitness<F, PCS>>,
    pub spartan_outer: Box<dyn SpartanOuterProver<F>>,
    pub spartan_product: Box<dyn SpartanProductProver<F>>,
    pub ram_read_write: Box<dyn RamReadWriteProver<F>>,
    pub instruction_claim_reduction: Box<dyn InstructionClaimReductionProver<F>>,
    pub ram_raf_evaluation: Box<dyn RamRafEvaluationProver<F>>,
    pub ram_output_check: Box<dyn RamOutputCheckProver<F>>,
}

impl<F, PCS> JoltBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    /// The always-present reference backend: every slot served by the naive
    /// tier. It is the equivalence anchor optimized backends are tested
    /// against, and the fallback partial backends compose over. Its commit
    /// slot streams, hence the [`StreamingCommitment`] bound — a
    /// reference-implementation requirement, not a seam one.
    pub fn reference() -> Self
    where
        PCS: StreamingCommitment,
    {
        Self {
            commit: Box::new(ReferenceBackend),
            spartan_outer: Box::new(ReferenceBackend),
            spartan_product: Box::new(ReferenceBackend),
            ram_read_write: Box::new(ReferenceBackend),
            instruction_claim_reduction: Box::new(ReferenceBackend),
            ram_raf_evaluation: Box::new(ReferenceBackend),
            ram_output_check: Box::new(ReferenceBackend),
        }
    }

    /// Open the proof-scoped session that slot state lives in. One session
    /// per proof; drop it when the proof is assembled.
    pub fn begin_proof(&self) -> ProofSession {
        ProofSession::default()
    }
}

/// The reference implementations' marker type: implements every slot trait
/// (each per-relation module hosts its impl next to the kernel it wraps).
pub struct ReferenceBackend;

/// Backend-owned state with proof lifetime, opaque to orchestration.
///
/// Slots stash and share private state keyed by a backend-private type, so
/// per-slot mixing of backend families cannot collide: witness-upload
/// residency, cross-member shared tables, and cross-stage carries all live
/// here, invisible to the stage recipes that thread `&mut ProofSession`
/// through every slot call.
#[derive(Default)]
pub struct ProofSession {
    state: HashMap<TypeId, Box<dyn Any>>,
}

impl ProofSession {
    /// The calling backend's private state, created by `init` on first
    /// access. `T` is the backend-private key: choose one type per backend
    /// family.
    #[expect(
        clippy::expect_used,
        reason = "the map entry is keyed by T's TypeId, so the downcast is infallible"
    )]
    pub fn state_or_insert_with<T: Any>(&mut self, init: impl FnOnce() -> T) -> &mut T {
        self.state
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(init()))
            .downcast_mut::<T>()
            .expect("ProofSession state entry keyed by its own TypeId")
    }

    /// The calling backend's private state, if any slot created it yet.
    pub fn state<T: Any>(&self) -> Option<&T> {
        self.state
            .get(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast_ref::<T>())
    }
}
