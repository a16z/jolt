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
    setup: &'a PCS::ProverSetup,
    shape: OneHotTraceShape,
    plan: OneHotTraceLayoutPlan,
    precommitted_hints: &'a [&'a PCS::OpeningHint],
}

impl<'a, PCS: CommitmentScheme> WitnessCommitRequest<'a, PCS> {
    pub fn new(
        setup: &'a PCS::ProverSetup,
        shape: OneHotTraceShape,
        precommitted_hints: &'a [&'a PCS::OpeningHint],
    ) -> Result<Self, OpeningsError>
    where
        PCS::ProverSetup: GroupSetupMetadata,
    {
        let plan = ONE_HOT_TRACE_LAYOUT.plan(&shape)?;
        if setup.default_layout_digest() != plan.layout_digest() {
            return Err(OpeningsError::InvalidSetup(
                "the Akita setup's layout digest is not the canonical OneHotTrace digest".into(),
            ));
        }
        let required_batch_polys = precommitted_hints
            .len()
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
        Ok(Self {
            setup,
            shape,
            plan,
            precommitted_hints,
        })
    }

    pub fn setup(&self) -> &'a PCS::ProverSetup {
        self.setup
    }

    pub fn shape(&self) -> &OneHotTraceShape {
        &self.shape
    }

    pub fn plan(&self) -> &OneHotTraceLayoutPlan {
        &self.plan
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
