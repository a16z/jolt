use super::util::{power_of_two_log_rows, require_unique_ids};
use crate::{
    MaterializationPolicy, NamespaceId, OracleDescriptor, OracleKind, OracleRef,
    PolynomialEncoding, PolynomialView, PublicValue, RetentionHint, ViewRequirement,
    WitnessDimensions, WitnessError, WitnessNamespace, WitnessProvider,
};

pub const WRAPPER_NAMESPACE: NamespaceId = NamespaceId::new("wrapper");

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WrapperNamespace {}

impl WitnessNamespace for WrapperNamespace {
    type CommittedId = WrapperCommittedPolynomial;
    type VirtualId = WrapperVirtualPolynomial;
    type OpeningId = WrapperOpeningId;
    type PublicId = WrapperPublicId;
    type ChallengeId = WrapperChallengeId;

    const ID: NamespaceId = WRAPPER_NAMESPACE;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WrapperCommittedPolynomial {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WrapperVirtualPolynomial {
    TransparentProofFields,
    ClearOpeningClaims,
    StageIntermediates,
    TranscriptReplayState,
    SumcheckVerifierEquations,
    OpeningSnapshot,
    DoryAssistVariables,
    R1csAssignment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WrapperOpeningId {
    AssignmentSlice(WrapperVirtualPolynomial),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WrapperPublicId {
    VerifierConfigDigest,
    PublicIoDigest,
    ProofTranscriptDigest,
    OpeningSnapshotDigest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WrapperChallengeId {
    TranscriptRound,
    SumcheckRound,
    OpeningBatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrapperAssignmentSlice<F> {
    pub id: WrapperVirtualPolynomial,
    pub values: Vec<F>,
}

impl<F> WrapperAssignmentSlice<F> {
    pub fn new(id: WrapperVirtualPolynomial, values: Vec<F>) -> Self {
        Self { id, values }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrapperPublicInput<F> {
    pub id: WrapperPublicId,
    pub values: Vec<PublicValue<F>>,
}

impl<F> WrapperPublicInput<F> {
    pub fn new(id: WrapperPublicId, values: Vec<PublicValue<F>>) -> Self {
        Self { id, values }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrapperAssignmentWitness<F> {
    slices: Vec<WrapperAssignmentSlice<F>>,
    public_inputs: Vec<WrapperPublicInput<F>>,
}

impl<F> WrapperAssignmentWitness<F> {
    pub fn new(
        slices: Vec<WrapperAssignmentSlice<F>>,
        public_inputs: Vec<WrapperPublicInput<F>>,
    ) -> Result<Self, WitnessError> {
        require_unique_ids(
            WRAPPER_NAMESPACE,
            slices.iter().map(|slice| slice.id),
            "assignment slice",
        )?;
        require_unique_ids(
            WRAPPER_NAMESPACE,
            public_inputs.iter().map(|public_input| public_input.id),
            "public input",
        )?;

        for slice in &slices {
            let _ = power_of_two_log_rows(WRAPPER_NAMESPACE, slice.values.len())?;
        }

        Ok(Self {
            slices,
            public_inputs,
        })
    }

    pub fn assignment_slice(&self, id: WrapperVirtualPolynomial) -> Result<&[F], WitnessError> {
        self.slices
            .iter()
            .find(|slice| slice.id == id)
            .map(|slice| slice.values.as_slice())
            .ok_or(WitnessError::UnknownOracle {
                namespace: WRAPPER_NAMESPACE.name,
            })
    }

    pub fn public_input(&self, id: WrapperPublicId) -> Result<&[PublicValue<F>], WitnessError> {
        self.public_inputs
            .iter()
            .find(|public_input| public_input.id == id)
            .map(|public_input| public_input.values.as_slice())
            .ok_or(WitnessError::UnknownOracle {
                namespace: WRAPPER_NAMESPACE.name,
            })
    }

    fn dimensions(&self, id: WrapperVirtualPolynomial) -> Result<WitnessDimensions, WitnessError> {
        let rows = self.assignment_slice(id)?.len();
        Ok(WitnessDimensions::new(power_of_two_log_rows(
            WRAPPER_NAMESPACE,
            rows,
        )?))
    }
}

impl<F> WitnessProvider<F, WrapperNamespace> for WrapperAssignmentWitness<F> {
    fn describe_oracle(
        &self,
        oracle: OracleRef<WrapperNamespace>,
    ) -> Result<OracleDescriptor<WrapperNamespace>, WitnessError> {
        let OracleKind::Virtual(id) = oracle.kind;
        Ok(OracleDescriptor::new(
            oracle,
            self.dimensions(id)?,
            PolynomialEncoding::Dense,
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<WrapperNamespace>,
    ) -> Result<Vec<ViewRequirement<WrapperNamespace>>, WitnessError> {
        let descriptor =
            <Self as WitnessProvider<F, WrapperNamespace>>::describe_oracle(self, oracle)?;
        Ok(vec![ViewRequirement::new(
            descriptor.reference,
            descriptor.encoding,
            MaterializationPolicy::BackendChoice,
            RetentionHint::Permanent,
        )])
    }

    fn oracle_view(
        &self,
        requirement: ViewRequirement<WrapperNamespace>,
    ) -> Result<PolynomialView<'_, F, WrapperNamespace>, WitnessError> {
        let descriptor = <Self as WitnessProvider<F, WrapperNamespace>>::describe_oracle(
            self,
            requirement.oracle,
        )?;
        let OracleKind::Virtual(id) = requirement.oracle.kind;
        Ok(PolynomialView::borrowed(
            descriptor,
            self.assignment_slice(id)?,
        ))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::WitnessProvider;

    fn slice(id: WrapperVirtualPolynomial, values: &[u64]) -> WrapperAssignmentSlice<u64> {
        WrapperAssignmentSlice::new(id, values.to_vec())
    }

    fn witness() -> WrapperAssignmentWitness<u64> {
        WrapperAssignmentWitness::new(
            vec![
                slice(WrapperVirtualPolynomial::TransparentProofFields, &[1, 2]),
                slice(
                    WrapperVirtualPolynomial::SumcheckVerifierEquations,
                    &[10, 11, 12, 13],
                ),
            ],
            vec![WrapperPublicInput::new(
                WrapperPublicId::VerifierConfigDigest,
                vec![PublicValue::new("config_digest", 42)],
            )],
        )
        .unwrap()
    }

    #[test]
    fn describes_wrapper_assignment_slice_without_vm_namespace() {
        let witness = witness();
        let oracle =
            OracleRef::virtual_polynomial(WrapperVirtualPolynomial::TransparentProofFields);

        let descriptor = witness.describe_oracle(oracle).unwrap();

        assert_eq!(descriptor.reference, oracle);
        assert_eq!(descriptor.dimensions, WitnessDimensions::new(1));
        assert_eq!(descriptor.encoding, PolynomialEncoding::Dense);
        assert_ne!(
            WRAPPER_NAMESPACE,
            crate::protocols::jolt_vm::JoltVmNamespace::ID
        );
    }

    #[test]
    fn returns_borrowed_wrapper_assignment_view() {
        let witness = witness();
        let oracle =
            OracleRef::virtual_polynomial(WrapperVirtualPolynomial::SumcheckVerifierEquations);
        let requirement = witness.view_requirements(oracle).unwrap().remove(0);

        let view = witness.oracle_view(requirement).unwrap();

        assert_eq!(view.as_slice(), Some([10, 11, 12, 13].as_slice()));
        assert_eq!(view.descriptor().dimensions, WitnessDimensions::new(2));
    }

    #[test]
    fn exposes_public_inputs_by_typed_id() {
        let witness = witness();

        assert_eq!(
            witness
                .public_input(WrapperPublicId::VerifierConfigDigest)
                .unwrap(),
            &[PublicValue::new("config_digest", 42)]
        );
        assert!(matches!(
            witness.public_input(WrapperPublicId::OpeningSnapshotDigest),
            Err(WitnessError::UnknownOracle { .. })
        ));
    }

    #[test]
    fn rejects_invalid_assignment_shapes() {
        let duplicate = WrapperAssignmentWitness::new(
            vec![
                slice(WrapperVirtualPolynomial::TranscriptReplayState, &[1, 2]),
                slice(WrapperVirtualPolynomial::TranscriptReplayState, &[3, 4]),
            ],
            Vec::new(),
        );
        assert!(matches!(
            duplicate,
            Err(WitnessError::InvalidWitnessData { .. })
        ));

        let non_power_of_two = WrapperAssignmentWitness::new(
            vec![slice(WrapperVirtualPolynomial::OpeningSnapshot, &[1, 2, 3])],
            Vec::new(),
        );
        assert!(matches!(
            non_power_of_two,
            Err(WitnessError::InvalidDimensions { .. })
        ));
    }
}
