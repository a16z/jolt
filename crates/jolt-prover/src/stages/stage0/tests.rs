#![expect(
    clippy::expect_used,
    reason = "commitment stage tests should fail loudly on impossible fixtures"
)]

use jolt_backends::{
    Backend, BackendError, CommitmentBackend, CommitmentMode, CommitmentRequest, CommitmentResult,
    CommitmentSlot, CommittedPolynomialOutput, StreamedWitnessOutput,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial;
use jolt_claims::protocols::jolt::{
    formulas::{dimensions::TracePolynomialOrder, ra::JoltRaPolynomialLayout},
    JoltCommittedPolynomial,
};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{mock::MockCommitmentScheme, CommitmentScheme};
use jolt_poly::Polynomial;
use jolt_verifier::config::{JoltProtocolConfig, ZkConfig};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::FieldInlineNamespace;
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

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug)]
struct TestFieldInlineWitness;

#[cfg(feature = "field-inline")]
impl<F> WitnessProvider<F, FieldInlineNamespace> for TestFieldInlineWitness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<FieldInlineNamespace>,
    ) -> Result<OracleDescriptor<FieldInlineNamespace>, WitnessError> {
        Ok(OracleDescriptor::new(
            oracle,
            WitnessDimensions::new(4, 2),
            PolynomialEncoding::Dense,
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<FieldInlineNamespace>,
    ) -> Result<Vec<ViewRequirement<FieldInlineNamespace>>, WitnessError> {
        Ok(vec![ViewRequirement::new(
            oracle,
            PolynomialEncoding::Dense,
            MaterializationPolicy::Streaming,
            RetentionHint::ThroughBlindFold,
        )])
    }

    fn oracle_view(
        &self,
        _request: OracleViewRequest<FieldInlineNamespace>,
    ) -> Result<PolynomialView<'_, F, FieldInlineNamespace>, WitnessError> {
        Err(WitnessError::UnsupportedFrontier {
            frontier: "test field-inline oracle views",
        })
    }
}

#[cfg(feature = "field-inline")]
impl<F> CommittedWitnessProvider<F, FieldInlineNamespace> for TestFieldInlineWitness {
    fn committed_oracle_order(&self) -> Result<Vec<FieldInlineCommittedPolynomial>, WitnessError> {
        Ok(vec![FieldInlineCommittedPolynomial::FieldRdInc])
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
    CommitmentStageOutput::from_backend_result(result, field_inline_commitment_result(99), config)
}

#[cfg(feature = "field-inline")]
fn field_inline_commitment_result(
    seed: u64,
) -> CommitmentResult<FieldInlineNamespace, MockCommitmentScheme<Fr>> {
    CommitmentResult::new(
        Vec::new(),
        Vec::new(),
        vec![CommittedPolynomialOutput::new(
            CommitmentSlot(0),
            OracleRef::committed(FieldInlineCommittedPolynomial::FieldRdInc),
            2,
            mock_commitment(seed),
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
        W: WitnessProvider<Fr, TestNamespace> + Sync + ?Sized,
    {
        self.last_request = Some(request.clone());
        Ok(CommitmentResult::new(
            Vec::new(),
            vec![StreamedWitnessOutput::new(CommitmentSlot(0), Vec::new())],
            Vec::new(),
        ))
    }
}

#[derive(Clone, Debug)]
struct JoltVmTestWitness {
    committed: Vec<JoltCommittedPolynomial>,
}

impl<F> WitnessProvider<F, JoltVmNamespace> for JoltVmTestWitness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<OracleDescriptor<JoltVmNamespace>, WitnessError> {
        Ok(OracleDescriptor::new(
            oracle,
            WitnessDimensions::new(8, 3),
            PolynomialEncoding::Compact,
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<Vec<ViewRequirement<JoltVmNamespace>>, WitnessError> {
        Ok(vec![ViewRequirement::new(
            oracle,
            PolynomialEncoding::Compact,
            MaterializationPolicy::Streaming,
            RetentionHint::ThroughStage8,
        )])
    }

    fn oracle_view(
        &self,
        _request: OracleViewRequest<JoltVmNamespace>,
    ) -> Result<PolynomialView<'_, F, JoltVmNamespace>, WitnessError> {
        Err(WitnessError::UnsupportedFrontier {
            frontier: "test Jolt VM oracle views",
        })
    }
}

impl<F> CommittedWitnessProvider<F, JoltVmNamespace> for JoltVmTestWitness {
    fn committed_oracle_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        Ok(self.committed.clone())
    }
}

#[derive(Debug, Default)]
struct JoltVmRecordingBackend {
    last_request: Option<CommitmentRequest<JoltVmNamespace>>,
    #[cfg(feature = "field-inline")]
    last_field_inline_request: Option<CommitmentRequest<FieldInlineNamespace>>,
}

impl Backend for JoltVmRecordingBackend {
    fn name(&self) -> &'static str {
        "jolt-vm-recording"
    }
}

impl CommitmentBackend<Fr, JoltVmNamespace, MockPcs> for JoltVmRecordingBackend {
    fn commit<W>(
        &mut self,
        request: &CommitmentRequest<JoltVmNamespace>,
        _witness: &W,
        _setup: &(),
    ) -> Result<CommitmentResult<JoltVmNamespace, MockPcs>, BackendError>
    where
        W: WitnessProvider<Fr, JoltVmNamespace> + Sync + ?Sized,
    {
        self.last_request = Some(request.clone());
        let commitments = request
            .items
            .iter()
            .map(|item| {
                let OracleKind::Committed(polynomial) = item.requirement.oracle.kind else {
                    unreachable!("commitment request should contain committed oracles only")
                };
                jolt_output(item.slot.0, polynomial, u64::from(item.slot.0) + 1)
            })
            .collect();
        Ok(CommitmentResult::new(Vec::new(), Vec::new(), commitments))
    }
}

#[cfg(feature = "field-inline")]
impl CommitmentBackend<Fr, FieldInlineNamespace, MockPcs> for JoltVmRecordingBackend {
    fn commit<W>(
        &mut self,
        request: &CommitmentRequest<FieldInlineNamespace>,
        _witness: &W,
        _setup: &(),
    ) -> Result<CommitmentResult<FieldInlineNamespace, MockPcs>, BackendError>
    where
        W: WitnessProvider<Fr, FieldInlineNamespace> + Sync + ?Sized,
    {
        self.last_field_inline_request = Some(request.clone());
        Ok(CommitmentResult::new(
            Vec::new(),
            vec![StreamedWitnessOutput::new(CommitmentSlot(0), Vec::new())],
            vec![CommittedPolynomialOutput::new(
                CommitmentSlot(0),
                OracleRef::committed(FieldInlineCommittedPolynomial::FieldRdInc),
                4,
                mock_commitment(42),
                (),
            )],
        ))
    }
}

#[test]
fn canonical_stage0_contract_requests_and_assembles_verifier_fields() -> Result<(), String> {
    let witness = JoltVmTestWitness {
        committed: vec![
            JoltCommittedPolynomial::RdInc,
            JoltCommittedPolynomial::RamInc,
            JoltCommittedPolynomial::InstructionRa(0),
            JoltCommittedPolynomial::InstructionRa(1),
            JoltCommittedPolynomial::RamRa(0),
            JoltCommittedPolynomial::RamRa(1),
            JoltCommittedPolynomial::BytecodeRa(0),
            JoltCommittedPolynomial::TrustedAdvice,
            JoltCommittedPolynomial::UntrustedAdvice,
        ],
    };
    let config = stage_config(true, true);
    #[cfg(feature = "field-inline")]
    let field_inline_witness = TestFieldInlineWitness;
    let input = CommitmentStageInput::new(
        &witness,
        &(),
        config,
        JoltProtocolConfig::for_zk(true),
        #[cfg(feature = "field-inline")]
        &field_inline_witness,
    );
    assert_eq!(input.protocol.zk, ZkConfig::BlindFold);
    let mut backend = JoltVmRecordingBackend::default();

    let output = prove::<Fr, _, _, MockPcs>(input, &mut backend).map_err(|e| e.to_string())?;

    let request = backend
        .last_request
        .expect("recording backend should capture canonical Stage 0 request");
    assert!(request
        .items
        .iter()
        .all(|item| item.mode == CommitmentMode::Zk));
    let requested = request
        .items
        .iter()
        .map(|item| item.requirement.oracle.kind)
        .collect::<Vec<_>>();
    assert_eq!(
        requested,
        vec![
            OracleKind::Committed(JoltCommittedPolynomial::RdInc),
            OracleKind::Committed(JoltCommittedPolynomial::RamInc),
            OracleKind::Committed(JoltCommittedPolynomial::InstructionRa(0)),
            OracleKind::Committed(JoltCommittedPolynomial::InstructionRa(1)),
            OracleKind::Committed(JoltCommittedPolynomial::RamRa(0)),
            OracleKind::Committed(JoltCommittedPolynomial::RamRa(1)),
            OracleKind::Committed(JoltCommittedPolynomial::BytecodeRa(0)),
            OracleKind::Committed(JoltCommittedPolynomial::TrustedAdvice),
            OracleKind::Committed(JoltCommittedPolynomial::UntrustedAdvice),
        ]
    );

    assert_eq!(output.commitments.rd_inc, mock_commitment(1));
    assert_eq!(output.commitments.ram_inc, mock_commitment(2));
    assert_eq!(
        output.commitments.ra.instruction,
        vec![mock_commitment(3), mock_commitment(4)]
    );
    assert_eq!(
        output.commitments.ra.ram,
        vec![mock_commitment(5), mock_commitment(6)]
    );
    assert_eq!(output.commitments.ra.bytecode, vec![mock_commitment(7)]);
    assert_eq!(
        output.trusted_advice_commitment.as_ref(),
        Some(&mock_commitment(8))
    );
    assert_eq!(
        output.untrusted_advice_commitment.as_ref(),
        Some(&mock_commitment(9))
    );
    #[cfg(feature = "field-inline")]
    {
        let field_inline_request = backend
            .last_field_inline_request
            .expect("recording backend should capture field-inline Stage 0 request");
        assert_eq!(field_inline_request.items.len(), 1);
        assert_eq!(field_inline_request.items[0].mode, CommitmentMode::Zk);
        assert_eq!(
            field_inline_request.items[0].requirement.oracle,
            OracleRef::committed(FieldInlineCommittedPolynomial::FieldRdInc)
        );
    }
    #[cfg(feature = "field-inline")]
    assert_eq!(
        output.commitments.field_inline.field_registers.rd_inc,
        mock_commitment(42)
    );
    Ok(())
}

#[test]
fn commitment_request_uses_committed_order_and_streaming_requirements() -> Result<(), String> {
    let witness = TestWitness {
        committed: vec![1, 2],
    };

    let request = build_commitment_request::<Fr, TestNamespace, _>(
        &witness,
        CommitmentMode::Transparent,
        TracePolynomialOrder::CycleMajor,
        None,
    )
    .map_err(|e| e.to_string())?;

    assert_eq!(request.items.len(), 2);
    assert_eq!(request.items[0].slot, CommitmentSlot(0));
    assert_eq!(request.items[0].mode, CommitmentMode::Transparent);
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
    assert_eq!(request.items[1].mode, CommitmentMode::Transparent);
    assert_eq!(request.items[1].requirement.oracle, OracleRef::committed(2));
    assert_eq!(
        request.items[1].requirement.retention,
        RetentionHint::ThroughBlindFold
    );
    Ok(())
}

#[test]
fn commit_schedules_the_backend_owned_request() -> Result<(), String> {
    let witness = TestWitness { committed: vec![3] };
    let mut backend = RecordingBackend::default();

    let result = super::prove::commit::<Fr, TestNamespace, _, _, MockPcs>(
        &witness,
        &mut backend,
        &(),
        CommitmentMode::Zk,
        TracePolynomialOrder::CycleMajor,
        None,
    )
    .map_err(|e| e.to_string())?;

    let request = backend
        .last_request
        .expect("recording backend should capture commitment request");
    assert_eq!(request.items.len(), 1);
    assert_eq!(request.items[0].requirement.oracle, OracleRef::committed(3));
    assert_eq!(request.items[0].mode, CommitmentMode::Zk);
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
        output.trusted_advice_commitment.as_ref(),
        Some(&mock_commitment(9))
    );
    assert_eq!(
        output.untrusted_advice_commitment.as_ref(),
        Some(&mock_commitment(10))
    );
    assert_eq!(output.prover_state.opening_hints.len(), 9);
    assert!(output
        .prover_state
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
            if reason.contains("unexpected Jolt commitment output for TrustedAdvice")
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
            if reason.contains("duplicate Jolt commitment output slot CommitmentSlot(0)")
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
    let output = CommitmentStageOutput::from_backend_result(
        result,
        field_inline_commitment_result(11),
        stage_config(false, false),
    )
    .map_err(|e| e.to_string())?;

    assert_eq!(
        output.commitments.field_inline.field_registers.rd_inc,
        mock_commitment(11)
    );
    assert!(output
        .prover_state
        .field_inline_opening_hints
        .contains_key(&FieldInlineCommittedPolynomial::FieldRdInc));
    Ok(())
}
