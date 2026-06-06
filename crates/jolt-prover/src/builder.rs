//! Shared helpers for assembling verifier-owned proof components from
//! semantically keyed backend outputs while retaining prover-only state.

use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    fmt::Debug,
};

use jolt_backends::CommitmentResult;
use jolt_openings::CommitmentScheme;
use jolt_witness::{OracleKind, WitnessNamespace};

use crate::ProverError;

type BuilderResult<T> = Result<T, VerifierComponentBuildError>;
type BuiltCommitmentComponent<Id, PCS, VerifierComponent> =
    BuiltVerifierComponent<VerifierComponent, BTreeMap<Id, <PCS as CommitmentScheme>::OpeningHint>>;

pub(crate) trait VerifierComponentId: Copy + Ord + Debug {}

impl<T> VerifierComponentId for T where T: Copy + Ord + Debug {}

#[derive(Debug)]
pub(crate) struct VerifierComponentBuildError {
    reason: String,
}

impl VerifierComponentBuildError {
    fn new(reason: String) -> Self {
        Self { reason }
    }

    fn into_commitment_error(self) -> ProverError {
        ProverError::InvalidCommitmentOutput {
            reason: self.reason,
        }
    }
}

pub(crate) struct BuiltVerifierComponent<VerifierComponent, ProverState> {
    pub(crate) verifier: VerifierComponent,
    pub(crate) prover_state: ProverState,
}

pub(crate) trait VerifierComponentSpec<Builder> {
    type VerifierComponent;

    fn assemble(self, builder: &mut Builder) -> Result<Self::VerifierComponent, ProverError>;
}

pub(crate) struct VerifierComponentBuilder<Id, VerifierValue, ProverValue>
where
    Id: VerifierComponentId,
{
    label: &'static str,
    component_name: &'static str,
    pending: BTreeMap<Id, (VerifierValue, ProverValue)>,
    prover_values: BTreeMap<Id, ProverValue>,
}

impl<Id, VerifierValue, ProverValue> VerifierComponentBuilder<Id, VerifierValue, ProverValue>
where
    Id: VerifierComponentId,
{
    pub(crate) fn new(label: &'static str, component_name: &'static str) -> Self {
        Self {
            label,
            component_name,
            pending: BTreeMap::new(),
            prover_values: BTreeMap::new(),
        }
    }

    pub(crate) fn insert(
        &mut self,
        id: Id,
        verifier_value: VerifierValue,
        prover_value: ProverValue,
    ) -> BuilderResult<()> {
        match self.pending.entry(id) {
            Entry::Occupied(_) => Err(VerifierComponentBuildError::new(format!(
                "duplicate {} {} for {id:?}",
                self.label, self.component_name
            ))),
            Entry::Vacant(entry) => {
                let _ = entry.insert((verifier_value, prover_value));
                Ok(())
            }
        }
    }

    pub(crate) fn take(&mut self, id: Id) -> BuilderResult<VerifierValue> {
        let (verifier_value, prover_value) = self.pending.remove(&id).ok_or_else(|| {
            VerifierComponentBuildError::new(format!(
                "missing {} {} for {id:?}",
                self.label, self.component_name
            ))
        })?;
        let _ = self.prover_values.insert(id, prover_value);
        Ok(verifier_value)
    }

    pub(crate) fn take_vec(
        &mut self,
        id_at: impl Fn(usize) -> Id,
        count: usize,
    ) -> BuilderResult<Vec<VerifierValue>> {
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            values.push(self.take(id_at(index))?);
        }
        Ok(values)
    }

    pub(crate) fn take_optional(
        &mut self,
        id: Id,
        expected: bool,
    ) -> BuilderResult<Option<VerifierValue>> {
        expected.then(|| self.take(id)).transpose()
    }

    pub(crate) fn finish(self) -> BuilderResult<BTreeMap<Id, ProverValue>> {
        if let Some(unexpected) = self.pending.keys().next().copied() {
            return Err(VerifierComponentBuildError::new(format!(
                "unexpected {} {} for {unexpected:?}",
                self.label, self.component_name
            )));
        }
        Ok(self.prover_values)
    }
}

pub(crate) struct VerifierCommitmentBuilder<Id, PCS>
where
    Id: VerifierComponentId,
    PCS: CommitmentScheme,
{
    components: VerifierComponentBuilder<Id, PCS::Output, PCS::OpeningHint>,
}

impl<Id, PCS> VerifierCommitmentBuilder<Id, PCS>
where
    Id: VerifierComponentId,
    PCS: CommitmentScheme,
{
    pub(crate) fn from_backend_result<N>(
        label: &'static str,
        result: CommitmentResult<N, PCS>,
    ) -> Result<Self, ProverError>
    where
        N: WitnessNamespace<CommittedId = Id>,
    {
        let mut components = VerifierComponentBuilder::new(label, "commitment output");
        let mut slots = BTreeSet::new();

        for output in result.commitments {
            if !slots.insert(output.slot.0) {
                return Err(ProverError::InvalidCommitmentOutput {
                    reason: format!("duplicate {label} commitment output slot {:?}", output.slot),
                });
            }
            let OracleKind::Committed(polynomial) = output.oracle.kind else {
                return Err(ProverError::InvalidCommitmentOutput {
                    reason: format!("{label} commitment backend emitted a virtual oracle output"),
                });
            };
            if output.rows == 0 {
                return Err(ProverError::InvalidCommitmentOutput {
                    reason: format!("{label} commitment output for {polynomial:?} has zero rows"),
                });
            }
            components
                .insert(polynomial, output.commitment, output.opening_hint)
                .map_err(VerifierComponentBuildError::into_commitment_error)?;
        }

        Ok(Self { components })
    }

    pub(crate) fn build<Spec>(
        mut self,
        spec: Spec,
    ) -> Result<BuiltCommitmentComponent<Id, PCS, Spec::VerifierComponent>, ProverError>
    where
        Spec: VerifierComponentSpec<Self>,
    {
        let verifier = spec.assemble(&mut self)?;
        let prover_state = self.finish()?;
        Ok(BuiltVerifierComponent {
            verifier,
            prover_state,
        })
    }

    pub(crate) fn take(&mut self, id: Id) -> Result<PCS::Output, ProverError> {
        self.components
            .take(id)
            .map_err(VerifierComponentBuildError::into_commitment_error)
    }

    pub(crate) fn take_vec(
        &mut self,
        id_at: impl Fn(usize) -> Id,
        count: usize,
    ) -> Result<Vec<PCS::Output>, ProverError> {
        self.components
            .take_vec(id_at, count)
            .map_err(VerifierComponentBuildError::into_commitment_error)
    }

    pub(crate) fn take_optional(
        &mut self,
        id: Id,
        expected: bool,
    ) -> Result<Option<PCS::Output>, ProverError> {
        self.components
            .take_optional(id, expected)
            .map_err(VerifierComponentBuildError::into_commitment_error)
    }

    pub(crate) fn finish(self) -> Result<BTreeMap<Id, PCS::OpeningHint>, ProverError> {
        self.components
            .finish()
            .map_err(VerifierComponentBuildError::into_commitment_error)
    }
}
