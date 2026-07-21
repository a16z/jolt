//! The precommitted claim-reduction family: the two-phase (stage-6b cycle
//! phase → stage-7 address phase) prover object, the session carries that
//! span it across the batch boundary, and the shared batch-member view both
//! phases mount.
//!
//! One [`PrecommittedReductionProver`] object spans two batches under two
//! different relations, so it cannot itself be a typed `SumcheckKernel`. The
//! stage-6b cycle kernel wraps the object in a [`SharedPrecommittedReduction`]
//! handle, parks a clone in the [`ProofSession`] under the member's carry key,
//! and mounts a [`SharedReductionKernel`] whose extraction resolves the
//! intermediate-vs-final wire claim from the schedule
//! (`has_address_phase`, read off the relation's layout at prepare time).
//! Stage 7's address-phase members reclaim the carry through
//! [`SessionCarriedKernels`], flip the phase, and mount the same view with
//! the final-opening extraction — so the per-phase wire-claim assembly is
//! single-sourced here for every backend family.

use std::cell::RefCell;
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

/// The wire-claim assembly a [`SharedReductionKernel`] runs at extraction.
type ReductionExtract<F, C> =
    Box<dyn Fn(&dyn PrecommittedReductionProver<F>) -> Result<C, SumcheckKernelError<F>>>;

/// One phase's batch-member view over a shared two-phase reduction:
/// `ProveRounds` delegates to the shared object; `extract` assembles the
/// phase's typed wire claims. Constructed by the per-relation kernel
/// builders below, which single-source the intermediate-vs-final resolution.
pub struct SharedReductionKernel<F: Field, C> {
    shared: SharedPrecommittedReduction<F>,
    extract: ReductionExtract<F, C>,
}

impl<F: Field, C> ProveRounds<F> for SharedReductionKernel<F, C> {
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

/// Emit the per-relation `SumcheckKernel` impl plus the two phase-kernel
/// builders for one precommitted member: the cycle builder parks the carry
/// and resolves intermediate-vs-final from `has_address_phase`; the address
/// builder mounts the final-opening extraction. `$wire` builds the CYCLE
/// claims struct from `(prover, has_address_phase)`; `$final_wire` the
/// ADDRESS claims struct from the fully bound prover.
macro_rules! precommitted_member {
    (
        $cycle_relation:ty, $cycle_claims:ty, $cycle_builder:ident,
        $address_relation:ty, $address_claims:ty, $address_builder:ident,
        $carry:ident, $missing:literal,
        cycle = $wire:expr, address = $final_wire:expr $(,)?
    ) => {
        impl<F: Field> SumcheckKernel<F> for SharedReductionKernel<F, $cycle_claims> {
            type Relation = $cycle_relation;

            fn output_claims(&mut self) -> Result<$cycle_claims, SumcheckKernelError<F>> {
                (self.extract)(&**self.shared.borrow())
            }
        }

        impl<F: Field> SumcheckKernel<F> for SharedReductionKernel<F, $address_claims> {
            type Relation = $address_relation;

            fn output_claims(&mut self) -> Result<$address_claims, SumcheckKernelError<F>> {
                (self.extract)(&**self.shared.borrow())
            }
        }

        /// Mount `prover` as its stage-6b cycle-phase batch member: park the
        /// shared handle for stage 7 and resolve the wire claim from the
        /// schedule (`has_address_phase` — the intermediate handoff claim
        /// when the address phase continues, else the final opening).
        pub fn $cycle_builder<F: Field>(
            session: &mut ProofSession,
            prover: Box<dyn PrecommittedReductionProver<F>>,
            has_address_phase: bool,
        ) -> Box<dyn SumcheckKernel<F, Relation = $cycle_relation>> {
            let shared: SharedPrecommittedReduction<F> = Rc::new(RefCell::new(prover));
            session.park($carry(Rc::clone(&shared)));
            let wire = $wire;
            Box::new(SharedReductionKernel {
                shared,
                extract: Box::new(move |prover| wire(prover, has_address_phase)),
            })
        }

        fn $address_builder<F: Field>(
            shared: SharedPrecommittedReduction<F>,
        ) -> Box<dyn SumcheckKernel<F, Relation = $address_relation>> {
            Box::new(SharedReductionKernel {
                shared,
                extract: Box::new($final_wire),
            })
        }

        impl<F: Field> PrepareKernel<F, $address_relation> for SessionCarriedKernels {
            fn prepare(
                &self,
                session: &mut ProofSession,
                _witness: &dyn JoltVmWitnessPlane<F>,
                _inputs: ProverInputs<'_, F, $address_relation>,
            ) -> Result<Box<dyn SumcheckKernel<F, Relation = $address_relation>>, KernelError<F>>
            {
                let $carry(shared) = session
                    .take::<$carry<F>>()
                    .ok_or(KernelError::InvariantViolation { reason: $missing })?;
                shared.borrow_mut().transition_to_address_phase();
                Ok($address_builder(shared))
            }
        }
    };
}

precommitted_member!(
    TrustedAdviceCyclePhase<F>,
    TrustedAdviceCyclePhaseOutputClaims<F>,
    trusted_advice_cycle_kernel,
    TrustedAdviceAddressPhase<F>,
    TrustedAdviceAddressPhaseOutputClaims<F>,
    trusted_advice_address_kernel,
    ParkedTrustedAdviceReduction,
    "stage 6b parked no trusted-advice reduction for the scheduled address phase",
    cycle = |prover: &dyn PrecommittedReductionProver<F>, has_address_phase| {
        Ok(TrustedAdviceCyclePhaseOutputClaims {
            trusted: if has_address_phase {
                prover.cycle_intermediate_claim()
            } else {
                prover.final_claim()?
            },
        })
    },
    address = |prover: &dyn PrecommittedReductionProver<F>| {
        Ok(TrustedAdviceAddressPhaseOutputClaims {
            trusted: prover.final_claim()?,
        })
    },
);

precommitted_member!(
    UntrustedAdviceCyclePhase<F>,
    UntrustedAdviceCyclePhaseOutputClaims<F>,
    untrusted_advice_cycle_kernel,
    UntrustedAdviceAddressPhase<F>,
    UntrustedAdviceAddressPhaseOutputClaims<F>,
    untrusted_advice_address_kernel,
    ParkedUntrustedAdviceReduction,
    "stage 6b parked no untrusted-advice reduction for the scheduled address phase",
    cycle = |prover: &dyn PrecommittedReductionProver<F>, has_address_phase| {
        Ok(UntrustedAdviceCyclePhaseOutputClaims {
            untrusted: if has_address_phase {
                prover.cycle_intermediate_claim()
            } else {
                prover.final_claim()?
            },
        })
    },
    address = |prover: &dyn PrecommittedReductionProver<F>| {
        Ok(UntrustedAdviceAddressPhaseOutputClaims {
            untrusted: prover.final_claim()?,
        })
    },
);

precommitted_member!(
    BytecodeReductionCyclePhase<F>,
    BytecodeReductionCyclePhaseOutputClaims<F>,
    bytecode_reduction_cycle_kernel,
    BytecodeReductionAddressPhase<F>,
    BytecodeReductionAddressPhaseOutputClaims<F>,
    bytecode_reduction_address_kernel,
    ParkedBytecodeReduction,
    "stage 6b parked no bytecode reduction for the scheduled address phase",
    cycle = |prover: &dyn PrecommittedReductionProver<F>, has_address_phase| {
        Ok(if has_address_phase {
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
    },
    address = |prover: &dyn PrecommittedReductionProver<F>| {
        Ok(BytecodeReductionAddressPhaseOutputClaims {
            chunks: prover.final_aux_claims()?,
        })
    },
);

precommitted_member!(
    ProgramImageReductionCyclePhase<F>,
    ProgramImageReductionCyclePhaseOutputClaims<F>,
    program_image_reduction_cycle_kernel,
    ProgramImageReductionAddressPhase<F>,
    ProgramImageReductionAddressPhaseOutputClaims<F>,
    program_image_reduction_address_kernel,
    ParkedProgramImageReduction,
    "stage 6b parked no program-image reduction for the scheduled address phase",
    cycle = |prover: &dyn PrecommittedReductionProver<F>, has_address_phase| {
        Ok(ProgramImageReductionCyclePhaseOutputClaims {
            program_image: if has_address_phase {
                prover.cycle_intermediate_claim()
            } else {
                prover.final_claim()?
            },
        })
    },
    address = |prover: &dyn PrecommittedReductionProver<F>| {
        Ok(ProgramImageReductionAddressPhaseOutputClaims {
            program_image: prover.final_claim()?,
        })
    },
);
