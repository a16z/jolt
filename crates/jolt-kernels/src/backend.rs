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
use jolt_field::{Field, FieldCore};
use jolt_kernels_derive::KernelSlots;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
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
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
    BytecodeReductionCyclePhase, ProgramImageReductionCyclePhase, TrustedAdviceCyclePhase,
    UntrustedAdviceCyclePhase,
};
use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity;
use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;
use jolt_verifier::stages::stage7::advice_address_phase::{
    TrustedAdviceAddressPhase, UntrustedAdviceAddressPhase,
};
use jolt_verifier::stages::stage7::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, ProgramImageReductionAddressPhase,
};
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use crate::commitment::CommitWitness;
use crate::kernel::{ProverInputs, SumcheckKernel};
use crate::opening::{AdviceOpeningEvaluation, JointOpeningPolynomials};
use crate::uniskip::UniskipKernel;
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
/// fronts, typed-row witnesses, precommitted phase spans, commit, joint
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

/// Type-indexed slot resolution on a kernel registry: the one place a
/// registry's field names are reached from generic driver code. `jolt-prover`'s
/// generated stage drivers bound their kernel source `B` by one `HasKernel<F,
/// R>` per batch member, then fetch each member's [`PrepareKernel`] through
/// [`kernel`](Self::kernel). Never implemented by hand for [`JoltBackend`]:
/// `#[derive(KernelSlots)]` emits one impl per `Box<dyn PrepareKernel<F, R>>`
/// field, so the field's own type is the relation→slot mapping and registry
/// and resolution cannot diverge. Any other registry — a partial backend, a
/// test double — implements it the same way, derived or by hand.
///
/// A relation with no slot is a missing-`HasKernel` bound error at the
/// consuming stage impl, and so is a slot mis-declared past the derive's
/// syntactic match (a non-`Box<dyn PrepareKernel<..>>` field yields no impl).
/// That match is single-bound: a `Box<dyn PrepareKernel<F, R> + Send>` (any
/// extra bound) is silently skipped and surfaces the same distant way.
pub trait HasKernel<F, R>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    fn kernel(&self) -> &dyn PrepareKernel<F, R>;
}

/// The kernel registry: one independently swappable slot per kernel entry.
///
/// `F` and `PCS` are deployment constants, not swap targets — the PCS traits
/// are structurally non-object-safe and their associated types are wire
/// types, so they stay type parameters. Every batch member's slot is a
/// `Box<dyn PrepareKernel<F, R>>`, resolved by type through the
/// `#[derive(KernelSlots)]`-emitted [`HasKernel`] impls; the remaining slots
/// are the bespoke non-sumcheck duties (commit streaming, the uni-skip
/// fronts, the advice opening evaluation, the joint opening).
#[derive(KernelSlots)]
#[kernel_slots(crate = "crate")]
pub struct JoltBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    pub commit: Box<dyn CommitWitness<F, PCS>>,
    pub spartan_outer_uniskip: Box<dyn UniskipKernel<F, OuterRemainder<F>>>,
    pub spartan_outer_remainder: Box<dyn PrepareKernel<F, OuterRemainder<F>>>,
    pub spartan_product_uniskip: Box<dyn UniskipKernel<F, ProductRemainder<F>>>,
    pub spartan_product_remainder: Box<dyn PrepareKernel<F, ProductRemainder<F>>>,
    pub ram_read_write: Box<dyn PrepareKernel<F, RamReadWriteChecking<F>>>,
    pub instruction_claim_reduction: Box<dyn PrepareKernel<F, InstructionClaimReduction<F>>>,
    pub ram_raf_evaluation: Box<dyn PrepareKernel<F, RamRafEvaluation<F>>>,
    pub ram_output_check: Box<dyn PrepareKernel<F, RamOutputCheck<F>>>,
    pub spartan_shift: Box<dyn PrepareKernel<F, SpartanShift<F>>>,
    pub instruction_input: Box<dyn PrepareKernel<F, InstructionInput<F>>>,
    pub registers_claim_reduction: Box<dyn PrepareKernel<F, RegistersClaimReduction<F>>>,
    pub registers_read_write: Box<dyn PrepareKernel<F, RegistersReadWriteChecking<F>>>,
    pub ram_val_check: Box<dyn PrepareKernel<F, RamValCheck<F>>>,
    pub advice_opening: Box<dyn AdviceOpeningEvaluation<F>>,
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
    pub trusted_advice_cycle: Box<dyn PrepareKernel<F, TrustedAdviceCyclePhase<F>>>,
    pub untrusted_advice_cycle: Box<dyn PrepareKernel<F, UntrustedAdviceCyclePhase<F>>>,
    pub bytecode_reduction_cycle: Box<dyn PrepareKernel<F, BytecodeReductionCyclePhase<F>>>,
    pub program_image_reduction_cycle:
        Box<dyn PrepareKernel<F, ProgramImageReductionCyclePhase<F>>>,
    pub hamming_weight_claim_reduction: Box<dyn PrepareKernel<F, HammingWeightClaimReduction<F>>>,
    pub trusted_advice_address: Box<dyn PrepareKernel<F, TrustedAdviceAddressPhase<F>>>,
    pub untrusted_advice_address: Box<dyn PrepareKernel<F, UntrustedAdviceAddressPhase<F>>>,
    pub bytecode_reduction_address: Box<dyn PrepareKernel<F, BytecodeReductionAddressPhase<F>>>,
    pub program_image_reduction_address:
        Box<dyn PrepareKernel<F, ProgramImageReductionAddressPhase<F>>>,
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
    /// the same type. The producing side (a stage front, a backend slot's
    /// `prepare`, or a kernel's post-extraction
    /// [`park_residue`](crate::SumcheckKernel::park_residue)) parks; the
    /// consuming kernel's `prepare` reclaims with [`take`](Self::take) — a
    /// missing or stale carry is a proof-time
    /// [`KernelError`](crate::KernelError), the accepted cost of keeping
    /// every batch member uniform.
    pub fn park<T: Any>(&mut self, value: T) {
        let _ = self.state.insert(TypeId::of::<T>(), Box::new(value));
    }

    /// Reclaim (remove and return) a parked carry, if present.
    #[expect(
        clippy::expect_used,
        reason = "the map entry is keyed by T's TypeId, so the downcast is infallible"
    )]
    pub fn take<T: Any>(&mut self) -> Option<T> {
        self.state.remove(&TypeId::of::<T>()).map(|boxed| {
            *boxed
                .downcast::<T>()
                .expect("ProofSession state entry keyed by its own TypeId")
        })
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

impl RetainedProgram {
    /// The session-resident program, read non-destructively — the shared
    /// fetch of the stage-6 table-fold kernels' `prepare`.
    pub fn from_session<F: FieldCore>(
        session: &ProofSession,
    ) -> Result<Arc<JoltProgramPreprocessing>, KernelError<F>> {
        Ok(session
            .state::<Self>()
            .ok_or(KernelError::InvariantViolation {
                reason: "prover-retained program data was not parked in the proof session",
            })?
            .program
            .clone())
    }
}

#[cfg(test)]
mod kernel_slots_derive_tests {
    use jolt_field::Fr;

    use super::*;

    struct StubPrepare;

    impl PrepareKernel<Fr, SpartanShift<Fr>> for StubPrepare {
        fn prepare(
            &self,
            _session: &mut ProofSession,
            _witness: &dyn JoltVmWitnessPlane<Fr>,
            _inputs: ProverInputs<'_, Fr, SpartanShift<Fr>>,
        ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = SpartanShift<Fr>>>, KernelError<Fr>>
        {
            Err(KernelError::Unsupported {
                reason: "stub slot for the KernelSlots derive test",
            })
        }
    }

    // Compiling proves the derive skipped the non-kernel fields: an impl
    // emitted for them would not type-check.
    #[derive(KernelSlots)]
    #[kernel_slots(crate = "crate")]
    struct ToyRegistry<F: Field> {
        label: String,
        shift: Box<dyn PrepareKernel<F, SpartanShift<F>>>,
        slot_count: usize,
    }

    fn slot<R, B>(registry: &B) -> &dyn PrepareKernel<Fr, R>
    where
        R: ConcreteSumcheck<Fr>,
        B: HasKernel<Fr, R>,
        SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
        SumcheckOutputClaims<Fr, R>: OutputClaims<Fr>,
        ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
    {
        registry.kernel()
    }

    #[test]
    fn derived_has_kernel_resolves_the_declared_slot() {
        let registry = ToyRegistry::<Fr> {
            label: "toy".to_string(),
            shift: Box::new(StubPrepare),
            slot_count: 1,
        };
        assert_eq!(registry.label, "toy");
        assert_eq!(registry.slot_count, 1);

        let resolved: *const () = std::ptr::from_ref(slot::<SpartanShift<Fr>, _>(&registry)).cast();
        let field: *const () = std::ptr::from_ref(&*registry.shift).cast();
        assert_eq!(resolved, field);
    }
}
