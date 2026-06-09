use crate::{
    CommittedWitnessProvider, MaterializationPolicy, NamespaceId, OracleDescriptor, OracleKind,
    OracleRef, OracleViewRequest, PolynomialChunk, PolynomialEncoding, PolynomialStream,
    PolynomialView, PublicValue, RetentionHint, ViewRequirement, WitnessDimensions, WitnessError,
    WitnessNamespace, WitnessProvider,
};

pub const DORY_ASSIST_NAMESPACE: NamespaceId = NamespaceId::new("dory_assist");

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DoryAssistNamespace {}

impl WitnessNamespace for DoryAssistNamespace {
    type CommittedId = DoryAssistCommittedPolynomial;
    type VirtualId = DoryAssistVirtualPolynomial;
    type OpeningId = DoryAssistOpeningId;
    type PublicId = DoryAssistPublicId;
    type ChallengeId = DoryAssistChallengeId;

    const ID: NamespaceId = DORY_ASSIST_NAMESPACE;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DoryAssistOperationFamily {
    G1,
    G2,
    Gt,
    Pairing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DoryAssistPackingId {
    DenseTrace,
    PrefixCodes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DoryAssistCommittedPolynomial {
    OperationTrace(DoryAssistOperationFamily),
    PrefixPacking(DoryAssistPackingId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DoryAssistVirtualPolynomial {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DoryAssistOpeningId {
    OperationTrace(DoryAssistOperationFamily),
    DenseTrace,
    PrefixPacking,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DoryAssistPublicId {
    DoryProofArtifact,
    VerifierSetupDigest,
    JoltEvaluationClaim,
    PairingFinalCheckInput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DoryAssistChallengeId {
    G1,
    G2,
    Gt,
    Pairing,
    Packing,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoryAssistCommittedColumn<F> {
    pub id: DoryAssistCommittedPolynomial,
    pub values: Vec<F>,
}

impl<F> DoryAssistCommittedColumn<F> {
    pub fn new(id: DoryAssistCommittedPolynomial, values: Vec<F>) -> Self {
        Self { id, values }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoryAssistPublicInput<F> {
    pub id: DoryAssistPublicId,
    pub values: Vec<PublicValue<F>>,
}

impl<F> DoryAssistPublicInput<F> {
    pub fn new(id: DoryAssistPublicId, values: Vec<PublicValue<F>>) -> Self {
        Self { id, values }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoryAssistWitness<F> {
    committed: Vec<DoryAssistCommittedColumn<F>>,
    public_inputs: Vec<DoryAssistPublicInput<F>>,
}

impl<F> DoryAssistWitness<F> {
    pub fn new(
        committed: Vec<DoryAssistCommittedColumn<F>>,
        public_inputs: Vec<DoryAssistPublicInput<F>>,
    ) -> Result<Self, WitnessError> {
        require_unique_ids(committed.iter().map(|column| column.id), "committed column")?;
        require_unique_ids(
            public_inputs.iter().map(|public_input| public_input.id),
            "public input",
        )?;

        for column in &committed {
            let _ = power_of_two_log_rows(column.values.len())?;
        }

        Ok(Self {
            committed,
            public_inputs,
        })
    }

    pub fn committed_values(
        &self,
        id: DoryAssistCommittedPolynomial,
    ) -> Result<&[F], WitnessError> {
        self.committed
            .iter()
            .find(|column| column.id == id)
            .map(|column| column.values.as_slice())
            .ok_or(WitnessError::UnknownOracle {
                namespace: DORY_ASSIST_NAMESPACE.name,
            })
    }

    pub fn public_input(&self, id: DoryAssistPublicId) -> Result<&[PublicValue<F>], WitnessError> {
        self.public_inputs
            .iter()
            .find(|public_input| public_input.id == id)
            .map(|public_input| public_input.values.as_slice())
            .ok_or(WitnessError::UnknownOracle {
                namespace: DORY_ASSIST_NAMESPACE.name,
            })
    }

    fn dimensions(
        &self,
        id: DoryAssistCommittedPolynomial,
    ) -> Result<WitnessDimensions, WitnessError> {
        let rows = self.committed_values(id)?.len();
        Ok(WitnessDimensions::new(rows, power_of_two_log_rows(rows)?))
    }
}

impl<F: Clone> WitnessProvider<F, DoryAssistNamespace> for DoryAssistWitness<F> {
    fn describe_oracle(
        &self,
        oracle: OracleRef<DoryAssistNamespace>,
    ) -> Result<OracleDescriptor<DoryAssistNamespace>, WitnessError> {
        match oracle.kind {
            OracleKind::Committed(id) => Ok(OracleDescriptor::new(
                oracle,
                self.dimensions(id)?,
                PolynomialEncoding::Dense,
            )),
            OracleKind::Virtual(_) => Err(WitnessError::UnknownOracle {
                namespace: DORY_ASSIST_NAMESPACE.name,
            }),
        }
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<DoryAssistNamespace>,
    ) -> Result<Vec<ViewRequirement<DoryAssistNamespace>>, WitnessError> {
        let descriptor =
            <Self as WitnessProvider<F, DoryAssistNamespace>>::describe_oracle(self, oracle)?;
        Ok(vec![ViewRequirement::new(
            descriptor.reference,
            descriptor.encoding,
            MaterializationPolicy::BackendChoice,
            RetentionHint::Permanent,
        )])
    }

    fn oracle_view(
        &self,
        request: OracleViewRequest<DoryAssistNamespace>,
    ) -> Result<PolynomialView<'_, F, DoryAssistNamespace>, WitnessError> {
        let descriptor = <Self as WitnessProvider<F, DoryAssistNamespace>>::describe_oracle(
            self,
            request.oracle(),
        )?;
        let OracleKind::Committed(id) = request.oracle().kind;
        Ok(PolynomialView::borrowed(
            descriptor,
            self.committed_values(id)?,
        ))
    }

    fn committed_stream<'a>(
        &'a self,
        id: DoryAssistCommittedPolynomial,
        chunk_size: usize,
    ) -> Result<Box<dyn PolynomialStream<F> + 'a>, WitnessError>
    where
        F: 'a,
        DoryAssistNamespace: 'a,
    {
        if chunk_size == 0 {
            return Err(WitnessError::InvalidDimensions {
                namespace: DORY_ASSIST_NAMESPACE.name,
                reason: "stream chunk size must be nonzero".to_owned(),
            });
        }

        Ok(Box::new(DoryAssistCommittedStream {
            values: self.committed_values(id)?,
            emitted: 0,
            chunk_size,
        }))
    }
}

impl<F: Clone> CommittedWitnessProvider<F, DoryAssistNamespace> for DoryAssistWitness<F> {
    fn committed_oracle_order(&self) -> Result<Vec<DoryAssistCommittedPolynomial>, WitnessError> {
        Ok(self.committed.iter().map(|column| column.id).collect())
    }
}

pub struct DoryAssistCommittedStream<'a, F> {
    values: &'a [F],
    emitted: usize,
    chunk_size: usize,
}

impl<F: Clone> PolynomialStream<F> for DoryAssistCommittedStream<'_, F> {
    fn next_chunk(&mut self) -> Result<Option<PolynomialChunk<F>>, WitnessError> {
        if self.emitted >= self.values.len() {
            return Ok(None);
        }

        let end = self
            .emitted
            .saturating_add(self.chunk_size)
            .min(self.values.len());
        let chunk = self.values[self.emitted..end].to_vec();
        self.emitted = end;

        Ok(Some(PolynomialChunk::Dense(chunk)))
    }
}

fn require_unique_ids<Id>(
    ids: impl IntoIterator<Item = Id>,
    label: &'static str,
) -> Result<(), WitnessError>
where
    Id: Copy + Eq + core::fmt::Debug,
{
    let mut seen = Vec::new();
    for id in ids {
        if seen.contains(&id) {
            return Err(WitnessError::InvalidWitnessData {
                namespace: DORY_ASSIST_NAMESPACE.name,
                reason: format!("duplicate {label} id: {id:?}"),
            });
        }
        seen.push(id);
    }
    Ok(())
}

fn power_of_two_log_rows(rows: usize) -> Result<usize, WitnessError> {
    if rows == 0 || !rows.is_power_of_two() {
        return Err(WitnessError::InvalidDimensions {
            namespace: DORY_ASSIST_NAMESPACE.name,
            reason: format!("row count must be a nonzero power of two, got {rows}"),
        });
    }
    Ok(rows.trailing_zeros() as usize)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::WitnessProvider;

    fn column(id: DoryAssistCommittedPolynomial, values: &[u64]) -> DoryAssistCommittedColumn<u64> {
        DoryAssistCommittedColumn::new(id, values.to_vec())
    }

    fn witness() -> DoryAssistWitness<u64> {
        DoryAssistWitness::new(
            vec![
                column(
                    DoryAssistCommittedPolynomial::OperationTrace(DoryAssistOperationFamily::G1),
                    &[1, 2, 3, 4],
                ),
                column(
                    DoryAssistCommittedPolynomial::PrefixPacking(DoryAssistPackingId::DenseTrace),
                    &[10, 11, 12, 13],
                ),
            ],
            vec![DoryAssistPublicInput::new(
                DoryAssistPublicId::JoltEvaluationClaim,
                vec![PublicValue::new("claim", 99)],
            )],
        )
        .unwrap()
    }

    #[test]
    fn describes_operation_trace_without_vm_namespace() {
        let witness = witness();
        let oracle = OracleRef::committed(DoryAssistCommittedPolynomial::OperationTrace(
            DoryAssistOperationFamily::G1,
        ));

        let descriptor = witness.describe_oracle(oracle).unwrap();

        assert_eq!(descriptor.reference, oracle);
        assert_eq!(descriptor.dimensions, WitnessDimensions::new(4, 2));
        assert_eq!(descriptor.encoding, PolynomialEncoding::Dense);
        assert_ne!(
            DORY_ASSIST_NAMESPACE,
            crate::protocols::jolt_vm::JoltVmNamespace::ID
        );
    }

    #[test]
    fn returns_borrowed_operation_trace_view() {
        let witness = witness();
        let oracle = OracleRef::committed(DoryAssistCommittedPolynomial::PrefixPacking(
            DoryAssistPackingId::DenseTrace,
        ));
        let requirement = witness.view_requirements(oracle).unwrap().remove(0);

        let view = witness
            .oracle_view(OracleViewRequest::new(requirement))
            .unwrap();

        assert_eq!(view.as_slice(), Some([10, 11, 12, 13].as_slice()));
        assert_eq!(view.descriptor().dimensions, WitnessDimensions::new(4, 2));
    }

    #[test]
    fn streams_committed_columns_through_core_trait() {
        let witness = witness();
        let mut stream = witness
            .committed_stream(
                DoryAssistCommittedPolynomial::OperationTrace(DoryAssistOperationFamily::G1),
                3,
            )
            .unwrap();

        assert_eq!(
            stream.next_chunk().unwrap(),
            Some(PolynomialChunk::Dense(vec![1, 2, 3]))
        );
        assert_eq!(
            stream.next_chunk().unwrap(),
            Some(PolynomialChunk::Dense(vec![4]))
        );
        assert_eq!(stream.next_chunk().unwrap(), None);
    }

    #[test]
    fn batch_stream_uses_committed_ids() {
        let witness = witness();
        let ids = [
            DoryAssistCommittedPolynomial::OperationTrace(DoryAssistOperationFamily::G1),
            DoryAssistCommittedPolynomial::PrefixPacking(DoryAssistPackingId::DenseTrace),
        ];
        let mut stream = witness.committed_batch_stream(&ids, 2).unwrap();

        let batch = stream.next_batch().unwrap().unwrap();

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.chunks[0].0, ids[0]);
        assert_eq!(batch.chunks[0].1, PolynomialChunk::Dense(vec![1, 2]));
        assert_eq!(batch.chunks[1].0, ids[1]);
        assert_eq!(batch.chunks[1].1, PolynomialChunk::Dense(vec![10, 11]));
    }

    #[test]
    fn exposes_public_inputs_by_typed_id() {
        let witness = witness();

        assert_eq!(
            witness
                .public_input(DoryAssistPublicId::JoltEvaluationClaim)
                .unwrap(),
            &[PublicValue::new("claim", 99)]
        );
        assert!(matches!(
            witness.public_input(DoryAssistPublicId::VerifierSetupDigest),
            Err(WitnessError::UnknownOracle { .. })
        ));
    }

    #[test]
    fn rejects_invalid_synthetic_shapes() {
        let duplicate = DoryAssistWitness::new(
            vec![
                column(
                    DoryAssistCommittedPolynomial::OperationTrace(DoryAssistOperationFamily::G1),
                    &[1, 2],
                ),
                column(
                    DoryAssistCommittedPolynomial::OperationTrace(DoryAssistOperationFamily::G1),
                    &[3, 4],
                ),
            ],
            Vec::new(),
        );
        assert!(matches!(
            duplicate,
            Err(WitnessError::InvalidWitnessData { .. })
        ));

        let non_power_of_two = DoryAssistWitness::new(
            vec![column(
                DoryAssistCommittedPolynomial::OperationTrace(DoryAssistOperationFamily::G2),
                &[1, 2, 3],
            )],
            Vec::new(),
        );
        assert!(matches!(
            non_power_of_two,
            Err(WitnessError::InvalidDimensions { .. })
        ));
    }
}
