//! The runtime seam: [`JoltBackend`] is the value `jolt-prover` proves
//! against — one boxed object-safe slot per kernel entry — and
//! [`ProofSession`] is the backend-owned state with proof lifetime. Swapping
//! a kernel implementation, mixing implementations per slot, running two
//! backends side by side, and choosing a configuration from the hardware are
//! all value construction, never compilation. See
//! `specs/clean-slate-prover.md`, "The backend seam".

use std::any::{Any, TypeId};
use std::collections::HashMap;

use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::Field;
use jolt_kernels_derive::KernelSlots;
use jolt_openings::CommitmentScheme;
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
use jolt_witness::JoltWitnessPlane;

use jolt_sumcheck::RoundScheduler;

use crate::commitment::CommitWitness;
use crate::kernel::{ProverInputs, SumcheckKernel};
use crate::opening::{AdviceOpeningEvaluation, JointOpeningPolynomials};
use crate::uniskip::UniskipKernel;
use crate::KernelError;

/// Factory behind [`JoltBackend::round_scheduler`]: stage fronts mint one
/// scheduler per stage via `build`. Takes [`ProofSession`] so a device
/// traversal shares the carry its kernels park in `prepare`, and so
/// per-proof state cannot leak onto the long-lived backend.
pub trait BuildRoundScheduler<F: Field> {
    fn build(&self, session: &mut ProofSession) -> Box<dyn RoundScheduler<F>>;
}

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
///
/// Also the registry seam: `jolt-prover`'s generated stage drivers bound
/// their kernel source `B` by one `PrepareKernel<F, R>` per batch member, so
/// a registry is any type implementing it per slot. Never implemented by
/// hand for [`JoltBackend`]: `#[derive(KernelSlots)]` emits one impl per
/// `Box<dyn PrepareKernel<F, R>>` field, delegating to that field, so the
/// field's own type is the relation→slot mapping and registry and resolution
/// cannot diverge. A relation with no slot is a missing-`PrepareKernel`
/// bound error at the consuming stage impl, and so is a slot mis-declared
/// past the derive's syntactic match (a non-`Box<dyn PrepareKernel<..>>`
/// field yields no impl). That match is single-bound: a `Box<dyn
/// PrepareKernel<F, R> + Send>` (any extra bound) is silently skipped and
/// surfaces the same distant way.
pub trait PrepareKernel<F, R>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    /// Starts optional work that can overlap the host's setup before `prepare`.
    fn prefetch(&self, _session: &mut ProofSession) -> Result<(), KernelError<F>> {
        Ok(())
    }

    /// Starts relation-dependent work before the stage driver prepares its members.
    fn prefetch_relation(
        &self,
        session: &mut ProofSession,
        _relation: &R,
    ) -> Result<(), KernelError<F>> {
        self.prefetch(session)
    }

    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, R>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = R>>, KernelError<F>>;
}

/// The kernel registry: one independently swappable slot per kernel entry.
///
/// `F` and `PCS` are deployment constants, not swap targets — the PCS traits
/// are structurally non-object-safe and their associated types are wire
/// types, so they stay type parameters. Every batch member's slot is a
/// `Box<dyn PrepareKernel<F, R>>`, reached by type through the
/// `#[derive(KernelSlots)]`-emitted delegating [`PrepareKernel`] impls; the
/// remaining slots are the bespoke non-sumcheck duties (commit streaming, the
/// uni-skip fronts, the advice opening evaluation, the joint opening, and the
/// round-traversal factory).
#[derive(KernelSlots)]
#[kernel_slots(crate = "crate")]
pub struct JoltBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    pub commit: Box<dyn CommitWitness<F, PCS>>,
    pub round_scheduler: Box<dyn BuildRoundScheduler<F>>,
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

/// [`Allocative`](allocative::Allocative) when the `allocative` feature is
/// on, vacuous otherwise. Everything stored in a [`ProofSession`] must be
/// heap-measurable so the profile harness's per-stage flamegraphs can
/// attribute the cross-stage carries — the dominant retained memory — rather
/// than an opaque `Box<dyn Any>`.
#[cfg(feature = "allocative")]
pub trait MaybeAllocative: allocative::Allocative + Send {}
#[cfg(feature = "allocative")]
impl<T: allocative::Allocative + Send + ?Sized> MaybeAllocative for T {}
/// [`Allocative`](https://docs.rs/allocative) when the `allocative` feature
/// is on, vacuous otherwise.
#[cfg(not(feature = "allocative"))]
pub trait MaybeAllocative: Send {}
#[cfg(not(feature = "allocative"))]
impl<T: Send + ?Sized> MaybeAllocative for T {}

/// One session entry: the erased value plus, under the `allocative` feature,
/// a monomorphized visitor captured at insertion — where the concrete type
/// is still known — so heap flamegraphs can see through the `dyn Any`.
struct Carry {
    value: Box<dyn Any + Send>,
    #[cfg(feature = "allocative")]
    visit: fn(&(dyn Any + Send), &mut allocative::Visitor<'_>),
}

impl Carry {
    fn new<T: Any + MaybeAllocative>(value: T) -> Self {
        Self {
            value: Box::new(value),
            #[cfg(feature = "allocative")]
            visit: visit_carry::<T>,
        }
    }
}

/// Visits one carry's concrete value, keyed by its type name (the frame
/// label in the rendered flamegraph).
#[cfg(feature = "allocative")]
fn visit_carry<T: Any + allocative::Allocative>(
    value: &(dyn Any + Send),
    visitor: &mut allocative::Visitor<'_>,
) {
    if let Some(value) = value.downcast_ref::<T>() {
        visitor.visit_field(allocative::Key::new(std::any::type_name::<T>()), value);
    }
}

/// Allocator-reserved bytes behind a `Vec` of flat elements. Field elements
/// carry no per-element heap (true of every production field), so parked
/// kernels can size their tables arithmetically — no `F: Allocative` bound
/// leaking into the generic reference impls that park them.
#[cfg(feature = "allocative")]
pub(crate) fn vec_heap_bytes<T>(v: &Vec<T>) -> usize {
    v.capacity() * size_of::<T>()
}

/// [`vec_heap_bytes`] for a table-of-tables: the outer spine plus every
/// inner reservation.
#[cfg(feature = "allocative")]
pub(crate) fn nested_vec_heap_bytes<T>(v: &Vec<Vec<T>>) -> usize {
    v.capacity() * size_of::<Vec<T>>()
        + v.iter()
            .map(|inner| inner.capacity() * size_of::<T>())
            .sum::<usize>()
}

/// Heap bytes behind a dense polynomial's evaluation table, by `len()` —
/// [`Polynomial`](jolt_poly::Polynomial) exposes no capacity. Exact at the
/// mid-stage snapshot (taken before any binding, when freshly built tables
/// have `len == capacity`); undercounts the truncated slack of bound state.
#[cfg(feature = "allocative")]
pub(crate) fn poly_heap_bytes<T>(poly: &jolt_poly::Polynomial<T>) -> usize {
    poly.len() * size_of::<T>()
}

/// [`poly_heap_bytes`] summed over a table list, plus the outer spine.
#[cfg(feature = "allocative")]
pub(crate) fn polys_heap_bytes<T>(polys: &Vec<jolt_poly::Polynomial<T>>) -> usize {
    polys.capacity() * size_of::<jolt_poly::Polynomial<T>>()
        + polys.iter().map(poly_heap_bytes).sum::<usize>()
}

/// Heap retained by an unbound Gruen split-equality table. The cache stores
/// every power-of-two prefix below each current table, so each side has
/// `2 * current_len - 1` field elements.
#[cfg(feature = "allocative")]
pub(crate) fn gruen_heap_bytes<F: jolt_field::Field>(
    split: &jolt_poly::GruenSplitEqPolynomial<F>,
) -> usize {
    let in_len = split.e_in_current_len();
    let out_len = split.e_out_current_len();
    let in_levels = in_len.ilog2() as usize + 1;
    let out_levels = out_len.ilog2() as usize + 1;
    let point_len = in_levels + out_levels - 1;
    point_len * size_of::<F>()
        + (in_levels + out_levels) * size_of::<Vec<F>>()
        + (2 * in_len + 2 * out_len - 2) * size_of::<F>()
}

/// Visit a shared flat vector without walking its elements. This preserves
/// `Arc` deduplication while keeping heap snapshots O(1) in the trace size.
#[cfg(feature = "allocative")]
pub(crate) fn visit_arc_vec<T>(
    visitor: &mut allocative::Visitor<'_>,
    key: allocative::Key,
    value: &std::sync::Arc<Vec<T>>,
) {
    let Some(mut visitor) = visitor.enter_shared(
        key,
        size_of::<*const Vec<T>>(),
        std::sync::Arc::as_ptr(value).cast(),
    ) else {
        return;
    };
    visitor.visit_simple(
        allocative::Key::new("ArcInner"),
        2 * size_of::<usize>() + size_of::<Vec<T>>(),
    );
    visitor.visit_simple(
        allocative::Key::new("elements"),
        value.capacity() * size_of::<T>(),
    );
    visitor.exit();
}

/// Backend-owned state with proof lifetime, opaque to orchestration.
///
/// Slots stash and share private state keyed by a backend-private type, so
/// per-slot mixing of backend families cannot collide: witness-upload
/// residency, cross-member shared tables, and cross-stage carries all live
/// here, invisible to the stage recipes that thread `&mut ProofSession`
/// through every slot call.
///
/// Inserted state must be [`MaybeAllocative`]: under the `allocative`
/// feature the session captures a per-entry visitor, so per-stage heap
/// flamegraphs attribute the carries' real contents.
#[derive(Default)]
pub struct ProofSession {
    state: HashMap<TypeId, Carry>,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2ProductInstructionPrefetch<F> {
    pub instruction_gamma: F,
}

#[cfg(feature = "allocative")]
impl<F> allocative::Allocative for Stage2ProductInstructionPrefetch<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.enter_self_sized::<Self>().exit();
    }
}

#[derive(Clone, Debug)]
pub struct Stage5InstructionReadRafPrefetch<F> {
    pub lookup_output_point: Vec<F>,
}

#[cfg(feature = "allocative")]
impl<F> allocative::Allocative for Stage5InstructionReadRafPrefetch<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple(
            allocative::Key::new("lookup_output_point"),
            vec_heap_bytes(&self.lookup_output_point),
        );
    }
}

impl ProofSession {
    /// The calling backend's private state, created by `init` on first
    /// access. `T` is the backend-private key: choose one type per backend
    /// family.
    #[expect(
        clippy::expect_used,
        reason = "the map entry is keyed by T's TypeId, so the downcast is infallible"
    )]
    pub fn state_or_insert_with<T: Any + MaybeAllocative>(
        &mut self,
        init: impl FnOnce() -> T,
    ) -> &mut T {
        self.state
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Carry::new(init()))
            .value
            .downcast_mut::<T>()
            .expect("ProofSession state entry keyed by its own TypeId")
    }

    /// The calling backend's private state, if any slot created it yet.
    pub fn state<T: Any>(&self) -> Option<&T> {
        self.state
            .get(&TypeId::of::<T>())
            .and_then(|carry| carry.value.downcast_ref::<T>())
    }

    /// Park `value` as a cross-stage carry, replacing any previous carry of
    /// the same type. The producing side (a stage front, a backend slot's
    /// `prepare`, or a kernel's post-extraction
    /// [`park_residue`](crate::SumcheckKernel::park_residue)) parks; the
    /// consuming kernel's `prepare` reclaims with [`take`](Self::take) — a
    /// missing or stale carry is a proof-time
    /// [`KernelError`](crate::KernelError), the accepted cost of keeping
    /// every batch member uniform.
    pub fn park<T: Any + MaybeAllocative>(&mut self, value: T) {
        let _ = self.state.insert(TypeId::of::<T>(), Carry::new(value));
    }

    /// Reclaim (remove and return) a parked carry, if present.
    #[expect(
        clippy::expect_used,
        reason = "the map entry is keyed by T's TypeId, so the downcast is infallible"
    )]
    pub fn take<T: Any>(&mut self) -> Option<T> {
        self.state.remove(&TypeId::of::<T>()).map(|carry| {
            *carry
                .value
                .downcast::<T>()
                .expect("ProofSession state entry keyed by its own TypeId")
        })
    }
}

/// Deep visitation: each entry's monomorphized visitor (captured at
/// insertion) sees through the `Box<dyn Any>`, so per-stage flamegraphs
/// attribute the parked kernel tables — the dominant retained memory —
/// keyed by their type names.
#[cfg(feature = "allocative")]
impl allocative::Allocative for ProofSession {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        for carry in self.state.values() {
            (carry.visit)(carry.value.as_ref(), &mut visitor);
        }
        visitor.exit();
    }
}

#[cfg(test)]
mod kernel_slots_derive_tests {
    use jolt_field::Fr;

    use super::*;

    struct StubPrepare;
    #[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
    struct Prefetched;

    impl PrepareKernel<Fr, SpartanShift<Fr>> for StubPrepare {
        fn prefetch(&self, session: &mut ProofSession) -> Result<(), KernelError<Fr>> {
            session.park(Prefetched);
            Ok(())
        }

        fn prepare(
            &self,
            _session: &mut ProofSession,
            _witness: &dyn JoltWitnessPlane<Fr>,
            _inputs: ProverInputs<'_, Fr, SpartanShift<Fr>>,
        ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = SpartanShift<Fr>>>, KernelError<Fr>>
        {
            Err(KernelError::Unsupported {
                reason: "stub slot for the KernelSlots derive test",
            })
        }
    }

    // Compiling proves the derive's wiring end to end: the generic bound
    // resolves (a delegating impl exists for the kernel field) and the
    // non-kernel fields were skipped (an impl emitted for them would not
    // type-check). A second slot for the same relation would be a
    // conflicting-impl error.
    #[derive(KernelSlots)]
    #[kernel_slots(crate = "crate")]
    struct ToyRegistry<F: Field> {
        label: String,
        shift: Box<dyn PrepareKernel<F, SpartanShift<F>>>,
        slot_count: usize,
    }

    #[test]
    fn derived_slots_resolve_by_relation_type() {
        let registry = ToyRegistry::<Fr> {
            label: "toy".to_string(),
            shift: Box::new(StubPrepare),
            slot_count: 1,
        };
        assert_eq!(registry.label, "toy");
        assert_eq!(registry.slot_count, 1);
        let mut session = ProofSession::default();
        assert!(
            <ToyRegistry<Fr> as PrepareKernel<Fr, SpartanShift<Fr>>>::prefetch(
                &registry,
                &mut session
            )
            .is_ok()
        );
        assert!(session.state::<Prefetched>().is_some());
    }
}
