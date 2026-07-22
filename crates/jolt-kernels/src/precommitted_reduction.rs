//! The precommitted claim-reduction family: the two-phase (stage-6b cycle
//! phase → stage-7 address phase) prover object, the session carries that
//! span it across the batch boundary, and the per-phase batch-member views
//! both phases mount.
//!
//! One [`PrecommittedReductionProver`] object spans two batches under two
//! different relations, so it cannot itself be a typed `SumcheckKernel`. The
//! stage-6b cycle builders wrap the object in a
//! [`SharedPrecommittedReduction`] handle, park a clone in the
//! [`ProofSession`] under the kind's carry key, and mount a
//! [`CycleReductionKernel`] whose extraction resolves the
//! intermediate-vs-final wire claim from the schedule (`has_address_phase`,
//! read off the relation's layout at prepare time). Stage 7's address-phase
//! members reclaim the carry through [`SessionCarriedKernels`], flip the
//! phase, and mount an [`AddressReductionKernel`] with the final-opening
//! extraction — so the per-phase wire-claim assembly is single-sourced here
//! for every backend family.
//!
//! Four kinds share the machinery — trusted advice, untrusted advice,
//! committed bytecode, program image — each as a plain set of impls below:
//! the two phase views are generic over the relation (one `ProveRounds`
//! delegation each), and each kind contributes one `SumcheckKernel` impl per
//! phase (the typed wire-claim assembly), one cycle builder, and one
//! stage-7 `PrepareKernel` impl reclaiming its carry. The scalar kinds
//! resolve intermediate-vs-final through [`CycleReductionKernel::scalar_claim`];
//! the bytecode kind's chunked shape spells its resolution out in its own
//! impl.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::{ProverInputs, SumcheckKernelError};
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
    BytecodeReductionCyclePhase, BytecodeReductionCyclePhaseOutputClaims,
    ProgramImageReductionCyclePhase, ProgramImageReductionCyclePhaseOutputClaims,
    TrustedAdviceCyclePhase, TrustedAdviceCyclePhaseOutputClaims, UntrustedAdviceCyclePhase,
    UntrustedAdviceCyclePhaseOutputClaims,
};
use jolt_verifier::stages::stage7::advice_address_phase::{
    TrustedAdviceAddressPhase, TrustedAdviceAddressPhaseOutputClaims, UntrustedAdviceAddressPhase,
    UntrustedAdviceAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage7::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, BytecodeReductionAddressPhaseOutputClaims,
    ProgramImageReductionAddressPhase, ProgramImageReductionAddressPhaseOutputClaims,
};
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use crate::{KernelError, PrepareKernel, ProofSession, SessionCarriedKernels, SumcheckKernel};

/// One precommitted reduction member, spanning stages 6b and 7: the cycle
/// phase binds first, stages the handoff claim, and — when the schedule has
/// active address rounds — the SAME object transitions
/// ([`Self::transition_to_address_phase`]) and binds the address phase.
pub trait PrecommittedReductionProver<F: Field>: ProveRounds<F> {
    /// Flip the member from the cycle phase to the address phase. Call between
    /// the stage-6b and stage-7 batches.
    fn transition_to_address_phase(&mut self);

    /// The intermediate claim staged at the cycle→address handoff:
    /// `Σ_i value(i) · eq(i) · scale` over the bound tables. Meaningful only
    /// when the schedule has an address phase.
    fn cycle_intermediate_claim(&self) -> F;

    /// The fully bound value coefficient — the reduction's final opening
    /// value (the advice/program-image polynomial's own opening; for the
    /// bytecode reduction, the chunk-weighted fold the per-chunk claims sum
    /// to). Errors while any variable remains unbound.
    fn final_claim(&self) -> Result<F, SumcheckKernelError<F>>;

    /// The fully bound `aux` coefficients — the per-chunk `BytecodeChunk(i)`
    /// opening values. Empty for reductions without aux tables. Errors while
    /// any variable remains unbound.
    fn final_aux_claims(&self) -> Result<Vec<F>, SumcheckKernelError<F>>;
}

/// The shared handle over one two-phase reduction: the mounted batch-member
/// view and the session carry each hold one. `Rc<RefCell<..>>` because the
/// object is exclusively borrowed only inside round calls — the parked clone
/// is dormant until stage 7 reclaims it.
pub type SharedPrecommittedReduction<F> = Rc<RefCell<Box<dyn PrecommittedReductionProver<F>>>>;

/// The trusted-advice reduction's 6b→7 session carry.
pub struct ParkedTrustedAdviceReduction<F: Field>(pub SharedPrecommittedReduction<F>);
/// The untrusted-advice reduction's 6b→7 session carry.
pub struct ParkedUntrustedAdviceReduction<F: Field>(pub SharedPrecommittedReduction<F>);
/// The committed-bytecode reduction's 6b→7 session carry.
pub struct ParkedBytecodeReduction<F: Field>(pub SharedPrecommittedReduction<F>);
/// The program-image reduction's 6b→7 session carry.
pub struct ParkedProgramImageReduction<F: Field>(pub SharedPrecommittedReduction<F>);

/// The stage-6b cycle-phase batch-member view over a shared two-phase
/// reduction: `ProveRounds` delegates to the shared object, and the per-kind
/// `SumcheckKernel` impls assemble the typed wire claims, resolving
/// intermediate-vs-final from the schedule (`has_address_phase`) captured at
/// mount time.
struct CycleReductionKernel<F: Field, R> {
    shared: SharedPrecommittedReduction<F>,
    has_address_phase: bool,
    _relation: PhantomData<fn() -> R>,
}

impl<F: Field, R> CycleReductionKernel<F, R> {
    fn new(shared: SharedPrecommittedReduction<F>, has_address_phase: bool) -> Self {
        Self {
            shared,
            has_address_phase,
            _relation: PhantomData,
        }
    }

    /// The schedule-resolved scalar wire claim: the intermediate handoff
    /// claim when the address phase continues, else the final opening. The
    /// single source of the intermediate-vs-final resolution for the
    /// scalar-shaped kinds (advice, program image); the bytecode kind's
    /// chunked wire shape spells the same resolution out in its own
    /// `output_claims`.
    fn scalar_claim(&self) -> Result<F, SumcheckKernelError<F>> {
        let prover = self.shared.borrow();
        if self.has_address_phase {
            Ok(prover.cycle_intermediate_claim())
        } else {
            prover.final_claim()
        }
    }
}

impl<F: Field, R> ProveRounds<F> for CycleReductionKernel<F, R> {
    fn num_rounds(&self) -> usize {
        self.shared.borrow().num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        self.shared
            .borrow_mut()
            .prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.shared.borrow_mut().finish_rounds(bind)
    }
}

/// The stage-7 address-phase batch-member view over a reclaimed two-phase
/// reduction: `ProveRounds` delegates to the shared object, and the per-kind
/// `SumcheckKernel` impls extract the final openings from the fully bound
/// tables.
struct AddressReductionKernel<F: Field, R> {
    shared: SharedPrecommittedReduction<F>,
    _relation: PhantomData<fn() -> R>,
}

impl<F: Field, R> AddressReductionKernel<F, R> {
    fn new(shared: SharedPrecommittedReduction<F>) -> Self {
        Self {
            shared,
            _relation: PhantomData,
        }
    }
}

impl<F: Field, R> ProveRounds<F> for AddressReductionKernel<F, R> {
    fn num_rounds(&self) -> usize {
        self.shared.borrow().num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        self.shared
            .borrow_mut()
            .prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.shared.borrow_mut().finish_rounds(bind)
    }
}

/// Mount `prover` as its stage-6b cycle-phase batch member: park the shared
/// handle for stage 7 under the kind's carry key and mount the cycle view.
pub fn trusted_advice_cycle_kernel<F: Field>(
    session: &mut ProofSession,
    prover: Box<dyn PrecommittedReductionProver<F>>,
    has_address_phase: bool,
) -> Box<dyn SumcheckKernel<F, Relation = TrustedAdviceCyclePhase<F>>> {
    let shared: SharedPrecommittedReduction<F> = Rc::new(RefCell::new(prover));
    session.park(ParkedTrustedAdviceReduction(Rc::clone(&shared)));
    Box::new(CycleReductionKernel::<F, TrustedAdviceCyclePhase<F>>::new(
        shared,
        has_address_phase,
    ))
}

/// Mount `prover` as its stage-6b cycle-phase batch member: park the shared
/// handle for stage 7 under the kind's carry key and mount the cycle view.
pub fn untrusted_advice_cycle_kernel<F: Field>(
    session: &mut ProofSession,
    prover: Box<dyn PrecommittedReductionProver<F>>,
    has_address_phase: bool,
) -> Box<dyn SumcheckKernel<F, Relation = UntrustedAdviceCyclePhase<F>>> {
    let shared: SharedPrecommittedReduction<F> = Rc::new(RefCell::new(prover));
    session.park(ParkedUntrustedAdviceReduction(Rc::clone(&shared)));
    Box::new(
        CycleReductionKernel::<F, UntrustedAdviceCyclePhase<F>>::new(shared, has_address_phase),
    )
}

/// Mount `prover` as its stage-6b cycle-phase batch member: park the shared
/// handle for stage 7 under the kind's carry key and mount the cycle view.
pub fn bytecode_reduction_cycle_kernel<F: Field>(
    session: &mut ProofSession,
    prover: Box<dyn PrecommittedReductionProver<F>>,
    has_address_phase: bool,
) -> Box<dyn SumcheckKernel<F, Relation = BytecodeReductionCyclePhase<F>>> {
    let shared: SharedPrecommittedReduction<F> = Rc::new(RefCell::new(prover));
    session.park(ParkedBytecodeReduction(Rc::clone(&shared)));
    Box::new(
        CycleReductionKernel::<F, BytecodeReductionCyclePhase<F>>::new(shared, has_address_phase),
    )
}

/// Mount `prover` as its stage-6b cycle-phase batch member: park the shared
/// handle for stage 7 under the kind's carry key and mount the cycle view.
pub fn program_image_reduction_cycle_kernel<F: Field>(
    session: &mut ProofSession,
    prover: Box<dyn PrecommittedReductionProver<F>>,
    has_address_phase: bool,
) -> Box<dyn SumcheckKernel<F, Relation = ProgramImageReductionCyclePhase<F>>> {
    let shared: SharedPrecommittedReduction<F> = Rc::new(RefCell::new(prover));
    session.park(ParkedProgramImageReduction(Rc::clone(&shared)));
    Box::new(
        CycleReductionKernel::<F, ProgramImageReductionCyclePhase<F>>::new(
            shared,
            has_address_phase,
        ),
    )
}

impl<F: Field> SumcheckKernel<F> for CycleReductionKernel<F, TrustedAdviceCyclePhase<F>> {
    type Relation = TrustedAdviceCyclePhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<TrustedAdviceCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(TrustedAdviceCyclePhaseOutputClaims {
            trusted: self.scalar_claim()?,
        })
    }
}

impl<F: Field> SumcheckKernel<F> for CycleReductionKernel<F, UntrustedAdviceCyclePhase<F>> {
    type Relation = UntrustedAdviceCyclePhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<UntrustedAdviceCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(UntrustedAdviceCyclePhaseOutputClaims {
            untrusted: self.scalar_claim()?,
        })
    }
}

impl<F: Field> SumcheckKernel<F> for CycleReductionKernel<F, BytecodeReductionCyclePhase<F>> {
    type Relation = BytecodeReductionCyclePhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<BytecodeReductionCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        // The chunked counterpart of `scalar_claim`: an address phase stages
        // the intermediate handoff claim (chunks come later, at stage 7); a
        // cycle-only schedule ends here with the per-chunk openings.
        let prover = self.shared.borrow();
        Ok(if self.has_address_phase {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: Some(prover.cycle_intermediate_claim()),
                chunks: Vec::new(),
            }
        } else {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: None,
                chunks: prover.final_aux_claims()?,
            }
        })
    }
}

impl<F: Field> SumcheckKernel<F> for CycleReductionKernel<F, ProgramImageReductionCyclePhase<F>> {
    type Relation = ProgramImageReductionCyclePhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<ProgramImageReductionCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(ProgramImageReductionCyclePhaseOutputClaims {
            program_image: self.scalar_claim()?,
        })
    }
}

impl<F: Field> SumcheckKernel<F> for AddressReductionKernel<F, TrustedAdviceAddressPhase<F>> {
    type Relation = TrustedAdviceAddressPhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<TrustedAdviceAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(TrustedAdviceAddressPhaseOutputClaims {
            trusted: self.shared.borrow().final_claim()?,
        })
    }
}

impl<F: Field> SumcheckKernel<F> for AddressReductionKernel<F, UntrustedAdviceAddressPhase<F>> {
    type Relation = UntrustedAdviceAddressPhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<UntrustedAdviceAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(UntrustedAdviceAddressPhaseOutputClaims {
            untrusted: self.shared.borrow().final_claim()?,
        })
    }
}

impl<F: Field> SumcheckKernel<F> for AddressReductionKernel<F, BytecodeReductionAddressPhase<F>> {
    type Relation = BytecodeReductionAddressPhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<BytecodeReductionAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(BytecodeReductionAddressPhaseOutputClaims {
            chunks: self.shared.borrow().final_aux_claims()?,
        })
    }
}

impl<F: Field> SumcheckKernel<F>
    for AddressReductionKernel<F, ProgramImageReductionAddressPhase<F>>
{
    type Relation = ProgramImageReductionAddressPhase<F>;

    fn output_claims(
        &mut self,
    ) -> Result<ProgramImageReductionAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(ProgramImageReductionAddressPhaseOutputClaims {
            program_image: self.shared.borrow().final_claim()?,
        })
    }
}

impl<F: Field> PrepareKernel<F, TrustedAdviceAddressPhase<F>> for SessionCarriedKernels {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltVmWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, TrustedAdviceAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = TrustedAdviceAddressPhase<F>>>, KernelError<F>>
    {
        let ParkedTrustedAdviceReduction(shared) = session
            .take::<ParkedTrustedAdviceReduction<F>>()
            .ok_or(KernelError::InvariantViolation {
                reason:
                    "stage 6b parked no trusted-advice reduction for the scheduled address phase",
            })?;
        shared.borrow_mut().transition_to_address_phase();
        Ok(Box::new(AddressReductionKernel::<
            F,
            TrustedAdviceAddressPhase<F>,
        >::new(shared)))
    }
}

impl<F: Field> PrepareKernel<F, UntrustedAdviceAddressPhase<F>> for SessionCarriedKernels {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltVmWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, UntrustedAdviceAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = UntrustedAdviceAddressPhase<F>>>, KernelError<F>>
    {
        let ParkedUntrustedAdviceReduction(shared) = session
            .take::<ParkedUntrustedAdviceReduction<F>>()
            .ok_or(KernelError::InvariantViolation {
                reason:
                    "stage 6b parked no untrusted-advice reduction for the scheduled address phase",
            })?;
        shared.borrow_mut().transition_to_address_phase();
        Ok(Box::new(AddressReductionKernel::<
            F,
            UntrustedAdviceAddressPhase<F>,
        >::new(shared)))
    }
}

impl<F: Field> PrepareKernel<F, BytecodeReductionAddressPhase<F>> for SessionCarriedKernels {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltVmWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, BytecodeReductionAddressPhase<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = BytecodeReductionAddressPhase<F>>>,
        KernelError<F>,
    > {
        let ParkedBytecodeReduction(shared) = session.take::<ParkedBytecodeReduction<F>>().ok_or(
            KernelError::InvariantViolation {
                reason: "stage 6b parked no bytecode reduction for the scheduled address phase",
            },
        )?;
        shared.borrow_mut().transition_to_address_phase();
        Ok(Box::new(AddressReductionKernel::<
            F,
            BytecodeReductionAddressPhase<F>,
        >::new(shared)))
    }
}

impl<F: Field> PrepareKernel<F, ProgramImageReductionAddressPhase<F>> for SessionCarriedKernels {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltVmWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, ProgramImageReductionAddressPhase<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = ProgramImageReductionAddressPhase<F>>>,
        KernelError<F>,
    > {
        let ParkedProgramImageReduction(shared) = session
            .take::<ParkedProgramImageReduction<F>>()
            .ok_or(KernelError::InvariantViolation {
                reason:
                    "stage 6b parked no program-image reduction for the scheduled address phase",
            })?;
        shared.borrow_mut().transition_to_address_phase();
        Ok(Box::new(AddressReductionKernel::<
            F,
            ProgramImageReductionAddressPhase<F>,
        >::new(shared)))
    }
}
