#![expect(
    clippy::expect_used,
    reason = "commitment stage tests should fail loudly on impossible fixtures"
)]

use jolt_backends::{
    Backend, BackendError, CommitmentBackend, CommitmentRequest, CommitmentResult, CommitmentSlot,
    CommittedPolynomialOutput, StreamedWitnessOutput,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial;
use jolt_claims::protocols::jolt::{formulas::ra::JoltRaPolynomialLayout, JoltCommittedPolynomial};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{mock::MockCommitmentScheme, CommitmentScheme};
use jolt_poly::Polynomial;
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, CommittedWitnessProvider, MaterializationPolicy,
    NamespaceId, OracleDescriptor, OracleKind, OracleRef, OracleViewRequest, PolynomialEncoding,
    PolynomialView, RetentionHint, ViewRequirement, WitnessDimensions, WitnessError,
    WitnessNamespace, WitnessProvider,
};

use super::*;
use crate::ProverError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TestNamespace {}

impl WitnessNamespace for TestNamespace {
    type ChallengeId = u8;
    type CommittedId = u8;
    type OpeningId = u8;
    type PublicId = u8;
    type VirtualId = u8;

    const ID: NamespaceId = NamespaceId::new("commitment_stage_test");
}

#[derive(Clone, Debug)]
struct TestWitness {
    committed: Vec<u8>,
}

impl<F> WitnessProvider<F, TestNamespace> for TestWitness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<TestNamespace>,
    ) -> Result<OracleDescriptor<TestNamespace>, WitnessError> {
        Ok(OracleDescriptor::new(
            oracle,
            WitnessDimensions::new(8, 3),
            match oracle.kind {
                OracleKind::Committed(1) => PolynomialEncoding::OneHot,
                _ => PolynomialEncoding::Compact,
            },
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<TestNamespace>,
    ) -> Result<Vec<ViewRequirement<TestNamespace>>, WitnessError> {
        let retention = match oracle.kind {
            OracleKind::Committed(2) => RetentionHint::ThroughBlindFold,
            _ => RetentionHint::ThroughStage8,
        };
        Ok(vec![ViewRequirement::new(
            oracle,
            PolynomialEncoding::Dense,
            MaterializationPolicy::BackendChoice,
            retention,
        )])
    }

    fn oracle_view(
        &self,
        _request: OracleViewRequest<TestNamespace>,
    ) -> Result<PolynomialView<'_, F, TestNamespace>, WitnessError> {
        Err(WitnessError::UnsupportedFrontier {
            frontier: "test oracle views",
        })
    }
}

impl<F> CommittedWitnessProvider<F, TestNamespace> for TestWitness {
    fn committed_oracle_order(&self) -> Result<Vec<u8>, WitnessError> {
        Ok(self.committed.clone())
    }
}

type MockPcs = MockCommitmentScheme<Fr>;

fn mock_commitment(seed: u64) -> <MockPcs as jolt_crypto::Commitment>::Output {
    let poly = Polynomial::new(vec![Fr::from_u64(seed), Fr::from_u64(seed + 1)]);
    <MockPcs as CommitmentScheme>::commit(&poly, &()).0
}

fn jolt_output(
    slot: u32,
    polynomial: JoltCommittedPolynomial,
    seed: u64,
) -> CommittedPolynomialOutput<JoltVmNamespace, MockPcs> {
    CommittedPolynomialOutput::new(
        CommitmentSlot(slot),
        OracleRef::committed(polynomial),
        2,
        mock_commitment(seed),
        (),
    )
}

fn stage_config(
    include_trusted_advice: bool,
    include_untrusted_advice: bool,
) -> CommitmentStageConfig {
    let ra_layout = JoltRaPolynomialLayout::new(2, 1, 2)
        .expect("test Jolt RA polynomial layout should be valid");
    CommitmentStageConfig::new(ra_layout, include_trusted_advice, include_untrusted_advice)
}

fn jolt_commitment_result(
    commitments: Vec<CommittedPolynomialOutput<JoltVmNamespace, MockPcs>>,
) -> CommitmentResult<JoltVmNamespace, MockPcs> {
    CommitmentResult::new(Vec::new(), Vec::new(), commitments)
}

#[cfg(not(feature = "field-inline"))]
fn assemble_test_jolt_commitment_stage(
    result: CommitmentResult<JoltVmNamespace, MockPcs>,
    config: CommitmentStageConfig,
) -> Result<CommitmentStageOutput<MockPcs>, ProverError> {
    CommitmentStageOutput::from_backend_result(result, config)
}

#[cfg(feature = "field-inline")]
fn assemble_test_jolt_commitment_stage(
    result: CommitmentResult<JoltVmNamespace, MockPcs>,
    config: CommitmentStageConfig,
) -> Result<CommitmentStageOutput<MockPcs>, ProverError> {
    CommitmentStageOutput::from_backend_result(
        result,
        config,
        vec![FieldInlineCommittedPolynomialOutput::new(
            FieldInlineCommittedPolynomial::FieldRdInc,
            2,
            mock_commitment(99),
            (),
        )],
    )
}

#[derive(Debug, Default)]
struct RecordingBackend {
    last_request: Option<CommitmentRequest<TestNamespace>>,
}

impl Backend for RecordingBackend {
    fn name(&self) -> &'static str {
        "recording"
    }
}

impl CommitmentBackend<Fr, TestNamespace, MockPcs> for RecordingBackend {
    fn commit<W>(
        &mut self,
        request: &CommitmentRequest<TestNamespace>,
        _witness: &W,
        _setup: &(),
    ) -> Result<CommitmentResult<TestNamespace, MockPcs>, BackendError>
    where
        W: WitnessProvider<Fr, TestNamespace>,
    {
        self.last_request = Some(request.clone());
        Ok(CommitmentResult::new(
            Vec::new(),
            vec![StreamedWitnessOutput::new(CommitmentSlot(0), Vec::new())],
            Vec::new(),
        ))
    }
}

#[test]
fn commitment_request_uses_committed_order_and_streaming_requirements() -> Result<(), String> {
    let witness = TestWitness {
        committed: vec![1, 2],
    };

    let request =
        build_commitment_request::<Fr, TestNamespace, _>(&witness).map_err(|e| e.to_string())?;

    assert_eq!(request.items.len(), 2);
    assert_eq!(request.items[0].slot, CommitmentSlot(0));
    assert_eq!(request.items[0].requirement.oracle, OracleRef::committed(1));
    assert_eq!(
        request.items[0].requirement.encoding,
        PolynomialEncoding::OneHot
    );
    assert_eq!(
        request.items[0].requirement.materialization,
        MaterializationPolicy::Streaming
    );
    assert_eq!(request.items[1].slot, CommitmentSlot(1));
    assert_eq!(request.items[1].requirement.oracle, OracleRef::committed(2));
    assert_eq!(
        request.items[1].requirement.retention,
        RetentionHint::ThroughBlindFold
    );
    Ok(())
}

#[test]
fn commit_witness_schedules_the_backend_owned_request() -> Result<(), String> {
    let witness = TestWitness { committed: vec![3] };
    let mut backend = RecordingBackend::default();

    let result = commit_witness::<Fr, TestNamespace, _, _, MockPcs>(&witness, &mut backend, &())
        .map_err(|e| e.to_string())?;

    let request = backend
        .last_request
        .expect("recording backend should capture commitment request");
    assert_eq!(request.items.len(), 1);
    assert_eq!(request.items[0].requirement.oracle, OracleRef::committed(3));
    assert_eq!(result.streamed_witness.len(), 1);
    Ok(())
}

#[test]
fn jolt_commitment_output_maps_by_polynomial_not_backend_order() -> Result<(), String> {
    let result = jolt_commitment_result(vec![
        jolt_output(8, JoltCommittedPolynomial::RamRa(1), 7),
        jolt_output(1, JoltCommittedPolynomial::RdInc, 1),
        jolt_output(3, JoltCommittedPolynomial::InstructionRa(1), 4),
        jolt_output(7, JoltCommittedPolynomial::TrustedAdvice, 9),
        jolt_output(4, JoltCommittedPolynomial::BytecodeRa(0), 5),
        jolt_output(0, JoltCommittedPolynomial::RamInc, 2),
        jolt_output(2, JoltCommittedPolynomial::InstructionRa(0), 3),
        jolt_output(9, JoltCommittedPolynomial::UntrustedAdvice, 10),
        jolt_output(6, JoltCommittedPolynomial::RamRa(0), 6),
    ]);

    let output = assemble_test_jolt_commitment_stage(result, stage_config(true, true))
        .map_err(|e| e.to_string())?;

    assert_eq!(output.commitments.rd_inc, mock_commitment(1));
    assert_eq!(output.commitments.ram_inc, mock_commitment(2));
    assert_eq!(
        output.commitments.ra.instruction,
        vec![mock_commitment(3), mock_commitment(4)]
    );
    assert_eq!(
        output.commitments.ra.ram,
        vec![mock_commitment(6), mock_commitment(7),]
    );
    assert_eq!(output.commitments.ra.bytecode, vec![mock_commitment(5)]);
    assert_eq!(
        output.trusted_advice_commitment(),
        Some(&mock_commitment(9))
    );
    assert_eq!(
        output.untrusted_advice_commitment(),
        Some(&mock_commitment(10))
    );
    assert_eq!(output.opening_hints.len(), 9);
    assert!(output
        .opening_hints
        .contains_key(&JoltCommittedPolynomial::RamInc));
    Ok(())
}

#[test]
fn jolt_commitment_output_requires_all_layout_polynomials() {
    let result = jolt_commitment_result(vec![
        jolt_output(0, JoltCommittedPolynomial::RamInc, 2),
        jolt_output(1, JoltCommittedPolynomial::RdInc, 1),
        jolt_output(2, JoltCommittedPolynomial::InstructionRa(0), 3),
        jolt_output(4, JoltCommittedPolynomial::BytecodeRa(0), 5),
        jolt_output(6, JoltCommittedPolynomial::RamRa(0), 6),
        jolt_output(8, JoltCommittedPolynomial::RamRa(1), 7),
    ]);

    let error = assemble_test_jolt_commitment_stage(result, stage_config(false, false))
        .err()
        .expect("missing InstructionRa(1) should fail output construction");

    assert!(matches!(
        error,
        ProverError::InvalidCommitmentOutput { reason }
            if reason.contains("missing Jolt commitment output for InstructionRa(1)")
    ));
}

#[test]
fn jolt_commitment_output_rejects_unplanned_advice_commitments() {
    let result = jolt_commitment_result(vec![
        jolt_output(0, JoltCommittedPolynomial::RamInc, 2),
        jolt_output(1, JoltCommittedPolynomial::RdInc, 1),
        jolt_output(2, JoltCommittedPolynomial::InstructionRa(0), 3),
        jolt_output(3, JoltCommittedPolynomial::InstructionRa(1), 4),
        jolt_output(4, JoltCommittedPolynomial::BytecodeRa(0), 5),
        jolt_output(6, JoltCommittedPolynomial::RamRa(0), 6),
        jolt_output(8, JoltCommittedPolynomial::RamRa(1), 7),
        jolt_output(9, JoltCommittedPolynomial::TrustedAdvice, 9),
    ]);

    let error = assemble_test_jolt_commitment_stage(result, stage_config(false, false))
        .err()
        .expect("disabled trusted advice commitment should fail output construction");

    assert!(matches!(
        error,
        ProverError::InvalidCommitmentOutput { reason }
            if reason.contains("unexpected Jolt committed polynomial TrustedAdvice")
    ));
}

#[test]
fn jolt_commitment_output_rejects_duplicate_slots() {
    let result = jolt_commitment_result(vec![
        jolt_output(0, JoltCommittedPolynomial::RamInc, 2),
        jolt_output(0, JoltCommittedPolynomial::RdInc, 1),
    ]);

    let error = assemble_test_jolt_commitment_stage(result, stage_config(false, false))
        .err()
        .expect("duplicate backend slots should fail output construction");

    assert!(matches!(
        error,
        ProverError::InvalidCommitmentOutput { reason }
            if reason.contains("duplicate commitment output slot CommitmentSlot(0)")
    ));
}

#[cfg(feature = "field-inline")]
#[test]
fn jolt_commitment_output_includes_field_inline_commitment() -> Result<(), String> {
    let result = jolt_commitment_result(vec![
        jolt_output(0, JoltCommittedPolynomial::RamInc, 2),
        jolt_output(1, JoltCommittedPolynomial::RdInc, 1),
        jolt_output(2, JoltCommittedPolynomial::InstructionRa(0), 3),
        jolt_output(3, JoltCommittedPolynomial::InstructionRa(1), 4),
        jolt_output(4, JoltCommittedPolynomial::BytecodeRa(0), 5),
        jolt_output(6, JoltCommittedPolynomial::RamRa(0), 6),
        jolt_output(8, JoltCommittedPolynomial::RamRa(1), 7),
    ]);
    let field_inline_outputs = vec![FieldInlineCommittedPolynomialOutput::new(
        FieldInlineCommittedPolynomial::FieldRdInc,
        2,
        mock_commitment(11),
        (),
    )];

    let output = CommitmentStageOutput::from_backend_result(
        result,
        stage_config(false, false),
        field_inline_outputs,
    )
    .map_err(|e| e.to_string())?;

    assert_eq!(
        output.commitments.field_inline.field_registers.rd_inc,
        mock_commitment(11)
    );
    assert!(output
        .field_inline_opening_hints
        .contains_key(&FieldInlineCommittedPolynomial::FieldRdInc));
    Ok(())
}
