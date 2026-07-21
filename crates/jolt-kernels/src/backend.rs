//! The runtime seam: [`JoltBackend`] is the value `jolt-prover` proves
//! against — one boxed object-safe slot per kernel entry — and
//! [`ProofSession`] is the backend-owned state with proof lifetime. Swapping
//! a kernel implementation, mixing implementations per slot, running two
//! backends side by side, and choosing a configuration from the hardware are
//! all value construction, never compilation. See
//! `specs/clean-slate-prover.md`, "The backend seam".

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, ProverInputs, SumcheckInputClaims,
    SumcheckKernel, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck;
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_verifier::stages::stage3::outputs::{
    InstructionInput, RegistersClaimReduction, SpartanShift,
};
use jolt_verifier::stages::stage4::ram_val_check::RamValCheck;
use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhase;
use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase;
use jolt_verifier::stages::stage6b::booleanity::Booleanity;
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity;
use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use crate::commitment::CommitWitness;
use crate::opening::JointOpeningPolynomials;
use crate::precommitted_reduction::{
    AdviceClaimReduction, BytecodeClaimReduction, ProgramImageClaimReduction,
};
use crate::spartan_outer::SpartanOuterProver;
use crate::spartan_product::SpartanProductProver;
use crate::KernelError;

/// The universal backend trait behind [`JoltBackend`]'s naive-served slots:
/// mint the [`SumcheckKernel`] that proves `R`, from the proof session, the
/// witness plane, and the member's protocol inputs. The relation instance
/// inside [`ProverInputs`] IS the typed request — kernels read
/// dimensions/points off its accessors instead of receiving them as restated
/// constructor arguments, so batch/kernel geometry divergence is
/// unrepresentable.
///
/// Named after std's `BuildHasher` shape: the stored verb-phrase trait mints
/// the worker that does the compute — platform ([`JoltBackend`]) → operation
/// (`PrepareKernel`) → execution ([`SumcheckKernel`]). Bespoke slots (uni-skip
/// handoffs, typed-row witnesses, precommitted phase spans, commit, joint
/// opening) keep hand-shaped traits in their own modules.
pub trait PrepareKernel<F, R>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, R>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = R>>, KernelError<F>>;
}

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
    pub ram_read_write: Box<dyn PrepareKernel<F, RamReadWriteChecking<F>>>,
    pub instruction_claim_reduction: Box<dyn PrepareKernel<F, InstructionClaimReduction<F>>>,
    pub ram_raf_evaluation: Box<dyn PrepareKernel<F, RamRafEvaluation<F>>>,
    pub ram_output_check: Box<dyn PrepareKernel<F, RamOutputCheck<F>>>,
    pub spartan_shift: Box<dyn PrepareKernel<F, SpartanShift<F>>>,
    pub instruction_input: Box<dyn PrepareKernel<F, InstructionInput<F>>>,
    pub registers_claim_reduction: Box<dyn PrepareKernel<F, RegistersClaimReduction<F>>>,
    pub registers_read_write: Box<dyn PrepareKernel<F, RegistersReadWriteChecking<F>>>,
    pub ram_val_check: Box<dyn PrepareKernel<F, RamValCheck<F>>>,
    pub instruction_read_raf: Box<dyn PrepareKernel<F, InstructionReadRaf<F>>>,
    pub ram_ra_claim_reduction: Box<dyn PrepareKernel<F, RamRaClaimReduction<F>>>,
    pub registers_val_evaluation: Box<dyn PrepareKernel<F, RegistersValEvaluation<F>>>,
    pub bytecode_read_raf_address: Box<dyn PrepareKernel<F, BytecodeReadRafAddressPhase<F>>>,
    pub booleanity_address: Box<dyn PrepareKernel<F, BooleanityAddressPhase<F>>>,
    pub bytecode_read_raf_cycle: Box<dyn PrepareKernel<F, BytecodeReadRafCycle<F>>>,
    pub booleanity_cycle: Box<dyn PrepareKernel<F, Booleanity<F>>>,
    pub ram_hamming_booleanity: Box<dyn PrepareKernel<F, RamHammingBooleanity<F>>>,
    pub ram_ra_virtualization: Box<dyn PrepareKernel<F, RamRaVirtualization<F>>>,
    pub instruction_ra_virtualization: Box<dyn PrepareKernel<F, InstructionRaVirtualization<F>>>,
    pub inc_claim_reduction: Box<dyn PrepareKernel<F, IncClaimReduction<F>>>,
    pub advice_claim_reduction: Box<dyn AdviceClaimReduction<F>>,
    pub bytecode_claim_reduction: Box<dyn BytecodeClaimReduction<F>>,
    pub program_image_claim_reduction: Box<dyn ProgramImageClaimReduction<F>>,
    pub hamming_weight_claim_reduction: Box<dyn PrepareKernel<F, HammingWeightClaimReduction<F>>>,
    pub joint_opening: Box<dyn JointOpeningPolynomials<F>>,
}

impl<F, PCS> JoltBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    /// Open the proof-scoped session that slot state lives in. One session
    /// per proof; drop it when the proof is assembled.
    pub fn begin_proof(&self) -> ProofSession {
        ProofSession::default()
    }
}

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

    /// Park `value` as a cross-stage carry, replacing any previous carry of
    /// the same type. The producing side (a stage front or an earlier stage's
    /// kernel) parks; the consuming kernel's `prepare` reclaims with
    /// [`take`](Self::take) — a missing or stale carry is a proof-time
    /// [`KernelError`](crate::KernelError), the accepted cost of keeping every
    /// batch member uniform.
    pub fn park<T: Any>(&mut self, value: T) {
        let _ = self.state.insert(TypeId::of::<T>(), Box::new(value));
    }

    /// Reclaim (remove and return) a parked carry, if present.
    pub fn take<T: Any>(&mut self) -> Option<T> {
        self.state
            .remove(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast::<T>().ok())
            .map(|boxed| *boxed)
    }
}

/// The prover-retained program data, parked in the [`ProofSession`] at proof
/// start (a `park`, read non-destructively via [`ProofSession::state`]): the
/// stage-6 table folds — the bytecode stage-value fold, the reduction chunk
/// grids, the program-image words — read the full program through this carry
/// instead of threading preprocessing borrows through every kernel call.
pub struct RetainedProgram {
    pub program: Arc<JoltProgramPreprocessing>,
}
