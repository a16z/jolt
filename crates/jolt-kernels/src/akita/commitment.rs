//! Backend-owned trace preparation and commitment for the Akita prover.

use jolt_claims::protocols::jolt::lattice::{
    OneHotTraceLayoutPlan, OneHotTraceShape, ONE_HOT_TRACE_LAYOUT,
};
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, GroupSetupMetadata, OpeningsError};
use jolt_witness::JoltWitnessPlane;

use crate::{KernelError, ProofSession};

#[cfg(feature = "field-inline")]
pub struct FieldIncObject<PCS: CommitmentScheme> {
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// Canonical trace geometry checked against the setup before backend dispatch.
/// Hints are in protocol order: untrusted advice, trusted advice, then
/// direct-program objects.
pub struct WitnessCommitRequest<'a, PCS: CommitmentScheme> {
    validated: WitnessCommitPlan<'a, PCS>,
    precommitted_hints: &'a [&'a PCS::OpeningHint],
}

/// Admitted trace geometry and batch capacity awaiting auxiliary opening hints.
pub struct WitnessCommitPlan<'a, PCS: CommitmentScheme> {
    setup: &'a PCS::ProverSetup,
    shape: OneHotTraceShape,
    plan: OneHotTraceLayoutPlan,
    num_precommitted: usize,
}

impl<'a, PCS: CommitmentScheme> WitnessCommitPlan<'a, PCS> {
    pub fn with_precommitted_hints(
        self,
        precommitted_hints: &'a [&'a PCS::OpeningHint],
    ) -> Result<WitnessCommitRequest<'a, PCS>, OpeningsError> {
        if precommitted_hints.len() != self.num_precommitted {
            return Err(OpeningsError::InvalidBatch(
                "Akita auxiliary hint count differs from the admitted batch".into(),
            ));
        }
        Ok(WitnessCommitRequest {
            validated: self,
            precommitted_hints,
        })
    }
}

impl<'a, PCS: CommitmentScheme> WitnessCommitRequest<'a, PCS> {
    /// Admit geometry and batch capacity before auxiliary commitments exist.
    pub fn validate_setup(
        setup: &'a PCS::ProverSetup,
        shape: OneHotTraceShape,
        num_precommitted: usize,
    ) -> Result<WitnessCommitPlan<'a, PCS>, OpeningsError>
    where
        PCS::ProverSetup: GroupSetupMetadata,
    {
        let plan = ONE_HOT_TRACE_LAYOUT.plan(&shape)?;
        if setup.default_layout_digest() != plan.layout_digest() {
            return Err(OpeningsError::InvalidSetup(
                "the Akita setup's layout digest is not the canonical OneHotTrace digest".into(),
            ));
        }
        let required_batch_polys = num_precommitted
            .checked_add(plan.ids().len())
            .and_then(|count| count.checked_add(usize::from(cfg!(feature = "field-inline"))))
            .ok_or_else(|| {
                OpeningsError::InvalidBatch("Akita batch polynomial count overflow".into())
            })?;
        if setup.max_num_vars() != plan.num_vars()
            || setup.max_num_polys_per_commitment_group() != plan.ids().len()
            || setup.max_total_batch_polys() < required_batch_polys
            || setup.one_hot_k() != 1usize << shape.log_k_chunk
        {
            return Err(OpeningsError::InvalidSetup(
                "the Akita setup's dimensions disagree with the canonical OneHotTrace shape".into(),
            ));
        }
        Ok(WitnessCommitPlan {
            setup,
            shape,
            plan,
            num_precommitted,
        })
    }

    pub fn setup(&self) -> &'a PCS::ProverSetup {
        self.validated.setup
    }

    pub fn shape(&self) -> &OneHotTraceShape {
        &self.validated.shape
    }

    pub fn plan(&self) -> &OneHotTraceLayoutPlan {
        &self.validated.plan
    }

    pub fn precommitted_hints(&self) -> &'a [&'a PCS::OpeningHint] {
        self.precommitted_hints
    }
}

/// Public commitments and private opening state retained by the coordinator.
pub struct WitnessCommitment<PCS: CommitmentScheme> {
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
    #[cfg(feature = "field-inline")]
    pub field_inc: FieldIncObject<PCS>,
}

/// Commit the canonical native trace witness and return its commitment and opening hints.
///
/// Implementations own witness preparation and resource release, and may use
/// caller-initialized state in the proof session. The coordinator owns protocol
/// validation and transcript absorption.
pub trait CommitWitness<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    fn commit_witness(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        request: WitnessCommitRequest<'_, PCS>,
    ) -> Result<WitnessCommitment<PCS>, KernelError<F>>;
}
