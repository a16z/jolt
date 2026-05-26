use crate::{
    BackendError, CommitmentBackend, CommitmentRequest, CommitmentRequestItem, CommitmentSlot,
};

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{mock::MockCommitmentScheme, CommitmentScheme};
use jolt_witness::{
    MaterializationPolicy, NamespaceId, OracleDescriptor, OracleRef, OracleViewRequest,
    PolynomialChunk, PolynomialChunkKind, PolynomialEncoding, PolynomialStream, PolynomialView,
    RetentionHint, ViewRequirement, WitnessDimensions, WitnessError, WitnessNamespace,
    WitnessProvider,
};

use super::{CpuBackend, CpuBackendConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TestNamespace {}

impl WitnessNamespace for TestNamespace {
    type ChallengeId = u8;
    type CommittedId = u8;
    type OpeningId = u8;
    type PublicId = u8;
    type VirtualId = u8;

    const ID: NamespaceId = NamespaceId::new("cpu_test");
}

#[derive(Clone, Debug)]
struct TestFieldWitness {
    encoding: PolynomialEncoding,
    dimensions: WitnessDimensions,
    chunks: Vec<PolynomialChunk<Fr>>,
}

struct TestFieldStream {
    chunks: std::vec::IntoIter<PolynomialChunk<Fr>>,
}

impl PolynomialStream<Fr> for TestFieldStream {
    fn next_chunk(&mut self) -> Result<Option<PolynomialChunk<Fr>>, WitnessError> {
        Ok(self.chunks.next())
    }
}

impl WitnessProvider<Fr, TestNamespace> for TestFieldWitness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<TestNamespace>,
    ) -> Result<OracleDescriptor<TestNamespace>, WitnessError> {
        Ok(OracleDescriptor::new(
            oracle,
            self.dimensions,
            self.encoding,
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<TestNamespace>,
    ) -> Result<Vec<ViewRequirement<TestNamespace>>, WitnessError> {
        Ok(vec![ViewRequirement::new(
            oracle,
            self.encoding,
            MaterializationPolicy::BackendChoice,
            RetentionHint::ThroughStage8,
        )])
    }

    fn oracle_view(
        &self,
        _request: OracleViewRequest<TestNamespace>,
    ) -> Result<PolynomialView<'_, Fr, TestNamespace>, WitnessError> {
        Err(WitnessError::UnsupportedFrontier {
            frontier: "cpu test field oracle views",
        })
    }

    fn committed_stream<'a>(
        &'a self,
        _id: u8,
        _chunk_size: usize,
    ) -> Result<Box<dyn PolynomialStream<Fr> + 'a>, WitnessError>
    where
        TestNamespace: 'a,
    {
        Ok(Box::new(TestFieldStream {
            chunks: self.chunks.clone().into_iter(),
        }))
    }
}

fn requirement(
    encoding: PolynomialEncoding,
    materialization: MaterializationPolicy,
) -> ViewRequirement<TestNamespace> {
    ViewRequirement::new(
        OracleRef::committed(7),
        encoding,
        materialization,
        RetentionHint::ThroughStage8,
    )
}

#[test]
fn cpu_commitment_backend_commits_compact_stream_by_slot() -> Result<(), String> {
    let values = vec![
        Fr::from_i64(1),
        Fr::from_i64(-2),
        Fr::from_i64(3),
        Fr::from_i64(4),
    ];
    let witness = TestFieldWitness {
        encoding: PolynomialEncoding::Compact,
        dimensions: WitnessDimensions::new(4, 2),
        chunks: vec![
            PolynomialChunk::I128(vec![1]),
            PolynomialChunk::I128(vec![-2, 3]),
            PolynomialChunk::I128(vec![4]),
        ],
    };
    let requirement = requirement(
        PolynomialEncoding::Compact,
        MaterializationPolicy::Streaming,
    );
    let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
        CommitmentSlot(5),
        requirement,
    )]);
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 2,
    });

    let result =
        <CpuBackend as CommitmentBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::commit(
            &mut backend,
            &request,
            &witness,
            &(),
        )
        .map_err(|error| error.to_string())?;
    let expected_poly = jolt_poly::Polynomial::new(values);
    let (expected, ()) = MockCommitmentScheme::commit(&expected_poly, &());

    assert_eq!(result.resolved_witness.len(), 1);
    assert_eq!(result.resolved_witness[0].slot, CommitmentSlot(5));
    assert_eq!(result.resolved_witness[0].requirement, requirement);
    assert_eq!(result.streamed_witness[0].slot, CommitmentSlot(5));
    assert_eq!(
        result.streamed_witness[0]
            .chunks
            .iter()
            .map(|chunk| chunk.rows)
            .collect::<Vec<_>>(),
        vec![1, 2, 1]
    );
    assert_eq!(result.commitments.len(), 1);
    assert_eq!(result.commitments[0].slot, CommitmentSlot(5));
    assert_eq!(result.commitments[0].oracle, OracleRef::committed(7));
    assert_eq!(result.commitments[0].rows, 4);
    assert_eq!(result.commitments[0].commitment, expected);
    Ok(())
}

#[test]
fn cpu_commitment_backend_commits_one_hot_stream_without_dense_witness() -> Result<(), String> {
    let indices = [Some(1), None, Some(0), Some(3)];
    let witness = TestFieldWitness {
        encoding: PolynomialEncoding::OneHot,
        dimensions: WitnessDimensions::new(16, 4),
        chunks: vec![
            PolynomialChunk::OneHot(indices[..2].to_vec()),
            PolynomialChunk::OneHot(indices[2..].to_vec()),
        ],
    };
    let requirement = ViewRequirement::new(
        OracleRef::committed(9),
        PolynomialEncoding::OneHot,
        MaterializationPolicy::Streaming,
        RetentionHint::ThroughStage8,
    );
    let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
        CommitmentSlot(6),
        requirement,
    )]);
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 2,
    });

    let result =
        <CpuBackend as CommitmentBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::commit(
            &mut backend,
            &request,
            &witness,
            &(),
        )
        .map_err(|error| error.to_string())?;
    let expected_indices = vec![Some(1), None, Some(0), Some(3)];
    let expected_poly = jolt_poly::OneHotPolynomial::new(4, expected_indices);
    let (expected, ()) = MockCommitmentScheme::commit(&expected_poly, &());

    assert_eq!(result.streamed_witness[0].rows, 4);
    assert_eq!(
        result.streamed_witness[0]
            .chunks
            .iter()
            .map(|chunk| (chunk.kind, chunk.rows))
            .collect::<Vec<_>>(),
        vec![
            (PolynomialChunkKind::OneHot, 2),
            (PolynomialChunkKind::OneHot, 2),
        ]
    );
    assert_eq!(result.commitments.len(), 1);
    assert_eq!(result.commitments[0].slot, CommitmentSlot(6));
    assert_eq!(result.commitments[0].oracle, OracleRef::committed(9));
    assert_eq!(result.commitments[0].rows, 16);
    assert_eq!(result.commitments[0].commitment, expected);
    Ok(())
}

#[test]
fn cpu_commitment_backend_rejects_requirement_encoding_mismatch() {
    let witness = TestFieldWitness {
        encoding: PolynomialEncoding::Dense,
        dimensions: WitnessDimensions::new(4, 2),
        chunks: Vec::new(),
    };
    let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
        CommitmentSlot(1),
        requirement(
            PolynomialEncoding::Compact,
            MaterializationPolicy::Streaming,
        ),
    )]);
    let mut backend = CpuBackend::default();

    let result =
        <CpuBackend as CommitmentBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::commit(
            &mut backend,
            &request,
            &witness,
            &(),
        );

    assert!(matches!(
        result,
        Err(BackendError::Witness(WitnessError::InvalidWitnessData {
            namespace: "cpu_test",
            ..
        }))
    ));
}

#[test]
fn cpu_commitment_backend_requires_streaming_materialization() {
    let witness = TestFieldWitness {
        encoding: PolynomialEncoding::Compact,
        dimensions: WitnessDimensions::new(4, 2),
        chunks: Vec::new(),
    };
    let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
        CommitmentSlot(1),
        requirement(
            PolynomialEncoding::Compact,
            MaterializationPolicy::BackendChoice,
        ),
    )]);
    let mut backend = CpuBackend::default();

    let result =
        <CpuBackend as CommitmentBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::commit(
            &mut backend,
            &request,
            &witness,
            &(),
        );

    assert!(matches!(
        result,
        Err(BackendError::Witness(WitnessError::InvalidWitnessData {
            namespace: "cpu_test",
            ..
        }))
    ));
}
