use jolt_backends::{cpu::CpuBackend, SumcheckBackend};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::formulas::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::jolt::formulas::spartan::SpartanOuterDimensions;
use jolt_claims::protocols::jolt::formulas::spartan::SPARTAN_OUTER_R1CS_INPUTS;
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_prover::stages::stage1::{
    input::Stage1ProverConfig,
    output::{r1cs_input_claims_from_evaluations, spartan_outer_claims_from_r1cs_inputs},
    request::{
        build_stage1_r1cs_evaluation_request, build_stage1_r1cs_materialization_request,
        build_stage1_request, STAGE1_REMAINDER_SLOT, STAGE1_UNISKIP_SLOT,
    },
};
#[cfg(feature = "field-inline")]
use jolt_prover::stages::stage1::{
    output::{
        field_inline_r1cs_input_claims_from_evaluations,
        field_inline_stage1_claims_from_r1cs_inputs, stage1_claims_from_r1cs_inputs,
        Stage1R1csInputClaim,
    },
    request::{build_stage1_field_inline_r1cs_evaluation_request, r1cs_input_slot},
};
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
use jolt_prover_harness::{FeatureMode, FixtureKind};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::FieldInlineNamespace;
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, MaterializationPolicy, OracleDescriptor, OracleKind,
    OracleRef, OracleViewRequest, PolynomialEncoding, PolynomialView, RetentionHint,
    ViewRequirement, WitnessDimensions, WitnessError, WitnessNamespace, WitnessProvider,
};

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage1_spartan_outer_frontier_requires_correctness_and_performance_gates() -> Result<(), String>
{
    let manifest =
        jolt_prover_harness::registered_frontiers().map_err(|error| error.to_string())?;
    let frontier = manifest
        .find("stage1_spartan_outer_requests")
        .ok_or_else(|| "stage1 Spartan outer frontier is missing".to_owned())?;

    assert!(frontier.requires_verifier_correctness());
    assert!(frontier.requires_core_performance());
    assert!(frontier.perf.is_some());
    Ok(())
}

#[test]
fn stage1_spartan_outer_frontier_is_replacement_ready_with_certified_kernel_evidence(
) -> Result<(), String> {
    let known = jolt_prover_harness::KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .map_err(|error| error.to_string())?;
    let ledger =
        jolt_prover_harness::registered_backend_kernel_ports(&known).map_err(|e| e.to_string())?;
    let manifest =
        jolt_prover_harness::registered_frontiers().map_err(|error| error.to_string())?;
    let frontier = manifest
        .find("stage1_spartan_outer_requests")
        .ok_or_else(|| "stage1 Spartan outer frontier is missing".to_owned())?;
    let port = ledger
        .find("cpu_spartan_outer_prefix_product_sum")
        .ok_or_else(|| "cpu_spartan_outer_prefix_product_sum ledger entry is missing".to_owned())?;
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or_else(|| "failed to locate workspace root".to_owned())?;
    let evidence = port
        .certification_evidence_files
        .iter()
        .map(|path| {
            jolt_prover_harness::KernelBenchmarkEvidence::read_json(&workspace_root.join(path))
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, String>>()?;

    jolt_prover_harness::validate_frontier_replacement_ready(*frontier, &known, &ledger, &evidence)
        .map_err(|error| error.to_string())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage1_cpu_r1cs_checkpoint_matches_core_muldiv_fixture() -> Result<(), String> {
    let request = jolt_prover_harness::FixtureRequest::new(
        FixtureKind::MuldivSmall,
        FeatureMode::Transparent,
    );
    let checkpoint = jolt_prover_harness::load_stage1_spartan_outer_checkpoint_fixture(&request)
        .map_err(|error| error.to_string())?;

    assert_eq!(checkpoint.claim_count, SPARTAN_OUTER_R1CS_INPUTS.len());
    assert_eq!(
        checkpoint.opening_point.len(),
        checkpoint.fixture.proof.trace_length.ilog2() as usize
    );
    checkpoint
        .fixture
        .verify()
        .map_err(|error| error.to_string())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage1_cpu_r1cs_checkpoint_matches_core_advice_fixture() -> Result<(), String> {
    let request = jolt_prover_harness::FixtureRequest::new(
        FixtureKind::AdviceConsumer,
        FeatureMode::Transparent,
    );
    let checkpoint = jolt_prover_harness::load_stage1_spartan_outer_checkpoint_fixture(&request)
        .map_err(|error| error.to_string())?;

    assert!(checkpoint.fixture.trusted_advice_commitment.is_some());
    assert!(checkpoint
        .fixture
        .proof
        .untrusted_advice_commitment
        .is_some());
    assert_eq!(checkpoint.claim_count, SPARTAN_OUTER_R1CS_INPUTS.len());
    assert_eq!(
        checkpoint.opening_point.len(),
        checkpoint.fixture.proof.trace_length.ilog2() as usize
    );
    checkpoint
        .fixture
        .verify()
        .map_err(|error| error.to_string())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage1_cpu_spartan_outer_verifier_replay_verifies_against_core_muldiv_fixture(
) -> Result<(), String> {
    let request = jolt_prover_harness::FixtureRequest::new(
        FixtureKind::MuldivSmall,
        FeatureMode::Transparent,
    );
    let fixture = jolt_prover_harness::load_stage1_spartan_outer_verifier_replay_fixture(&request)
        .map_err(|error| error.to_string())?;

    fixture.verify().map_err(|error| error.to_string())
}

#[test]
#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage1_cpu_spartan_outer_verifier_replay_verifies_against_core_advice_fixture(
) -> Result<(), String> {
    let request = jolt_prover_harness::FixtureRequest::new(
        FixtureKind::AdviceConsumer,
        FeatureMode::Transparent,
    );
    let fixture = jolt_prover_harness::load_stage1_spartan_outer_verifier_replay_fixture(&request)
        .map_err(|error| error.to_string())?;

    assert!(fixture.trusted_advice_commitment.is_some());
    assert!(fixture.proof.untrusted_advice_commitment.is_some());
    fixture.verify().map_err(|error| error.to_string())
}

#[test]
fn stage1_spartan_outer_request_resolves_on_cpu_backend() -> Result<(), String> {
    let witness = Stage1Witness;
    let request = build_stage1_request::<Fr, _>(Stage1ProverConfig::new(4), &witness)
        .map_err(|error| error.to_string())?;
    let mut backend = CpuBackend::default();

    let resolution = <CpuBackend as SumcheckBackend<Fr, JoltVmNamespace>>::resolve_sumcheck_views(
        &mut backend,
        &request.sumchecks,
        &witness,
    )
    .map_err(|error| error.to_string())?;

    assert_eq!(request.sumchecks.instances.len(), 2);
    assert_eq!(
        resolution.resolved_witness.len(),
        request.r1cs_inputs.len() * request.sumchecks.instances.len()
    );

    for (index, input) in request.r1cs_inputs.iter().enumerate() {
        let uniskip = &resolution.resolved_witness[index];
        let remainder = &resolution.resolved_witness[index + request.r1cs_inputs.len()];
        assert_eq!(uniskip.slot, STAGE1_UNISKIP_SLOT);
        assert_eq!(remainder.slot, STAGE1_REMAINDER_SLOT);
        assert_eq!(uniskip.view_index, index);
        assert_eq!(remainder.view_index, index);
        assert_eq!(uniskip.requirement, input.view);
        assert_eq!(remainder.requirement, input.view);
        assert_eq!(
            uniskip.descriptor.reference,
            OracleRef::virtual_polynomial(input.variable)
        );
        assert_eq!(
            remainder.descriptor.reference,
            OracleRef::virtual_polynomial(input.variable)
        );
        assert_eq!(uniskip.descriptor.encoding, PolynomialEncoding::Dense);
        assert_eq!(uniskip.descriptor.dimensions, WitnessDimensions::new(16, 4));
    }

    Ok(())
}

#[test]
fn stage1_r1cs_input_evaluations_run_on_cpu_backend() -> Result<(), String> {
    let witness = Stage1Witness;
    let config = Stage1ProverConfig::new(4);
    let request = build_stage1_r1cs_evaluation_request(
        config,
        &witness,
        vec![
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ],
    )
    .map_err(|error| error.to_string())?;
    let mut backend = CpuBackend::default();

    let evaluations =
        <CpuBackend as SumcheckBackend<Fr, JoltVmNamespace>>::evaluate_sumcheck_views(
            &mut backend,
            &request.evaluations,
            &witness,
        )
        .map_err(|error| error.to_string())?;
    let claims = r1cs_input_claims_from_evaluations(&request, evaluations)
        .map_err(|error| error.to_string())?;
    let outer_claims =
        spartan_outer_claims_from_r1cs_inputs(&claims).map_err(|error| error.to_string())?;

    assert_eq!(claims.len(), SPARTAN_OUTER_R1CS_INPUTS.len());
    for (index, claim) in claims.iter().enumerate() {
        assert_eq!(claim.variable, SPARTAN_OUTER_R1CS_INPUTS[index]);
        assert_eq!(claim.value, Fr::from_u64(index as u64 + 1));
    }
    assert_eq!(outer_claims.left_instruction_input, Fr::from_u64(1));
    assert_eq!(outer_claims.right_instruction_input, Fr::from_u64(2));
    assert_eq!(outer_claims.flags.is_last_in_sequence, Fr::from_u64(35));

    Ok(())
}

#[test]
fn stage1_r1cs_input_materializations_run_on_cpu_backend() -> Result<(), String> {
    let witness = Stage1Witness;
    let config = Stage1ProverConfig::new(4);
    let request = build_stage1_r1cs_materialization_request::<Fr, _>(config, &witness)
        .map_err(|error| error.to_string())?;
    let mut backend = CpuBackend::default();

    let materialized =
        <CpuBackend as SumcheckBackend<Fr, JoltVmNamespace>>::materialize_sumcheck_views(
            &mut backend,
            &request.materializations,
            &witness,
        )
        .map_err(|error| error.to_string())?;

    assert_eq!(materialized.len(), SPARTAN_OUTER_R1CS_INPUTS.len());
    for (index, output) in materialized.iter().enumerate() {
        assert_eq!(output.slot, request.r1cs_inputs[index].slot);
        assert_eq!(output.values, vec![Fr::from_u64(index as u64 + 1); 16]);
    }

    Ok(())
}

#[test]
#[cfg(feature = "field-inline")]
fn stage1_field_inline_r1cs_input_evaluations_run_on_cpu_backend() -> Result<(), String> {
    let witness = Stage1FieldInlineWitness;
    let config = Stage1ProverConfig::new(4);
    let request = build_stage1_field_inline_r1cs_evaluation_request(
        config,
        &witness,
        vec![
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ],
    )
    .map_err(|error| error.to_string())?;
    let mut backend = CpuBackend::default();

    let evaluations =
        <CpuBackend as SumcheckBackend<Fr, FieldInlineNamespace>>::evaluate_sumcheck_views(
            &mut backend,
            &request.evaluations,
            &witness,
        )
        .map_err(|error| error.to_string())?;
    let claims = field_inline_r1cs_input_claims_from_evaluations(&request, evaluations)
        .map_err(|error| error.to_string())?;
    let field_inline =
        field_inline_stage1_claims_from_r1cs_inputs(&claims).map_err(|error| error.to_string())?;
    let stage_claims =
        stage1_claims_from_r1cs_inputs(Fr::from_u64(77), &base_r1cs_claims(), &claims)
            .map_err(|error| error.to_string())?;

    assert_eq!(claims.len(), FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS.len());
    for (index, claim) in claims.iter().enumerate() {
        assert_eq!(
            claim.variable,
            FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS[index]
        );
        assert_eq!(claim.value, Fr::from_u64(index as u64 + 41));
    }
    assert_eq!(field_inline.field_rs1_value, Fr::from_u64(41));
    assert_eq!(field_inline.field_inv_product, Fr::from_u64(45));
    assert_eq!(field_inline.flags.load_imm, Fr::from_u64(53));
    assert_eq!(stage_claims.uniskip_output_claim, Fr::from_u64(77));
    assert_eq!(
        stage_claims
            .spartan_outer_claims(&SpartanOuterDimensions::rv64(config.log_t))
            .map_err(|error| error.to_string())?
            .len(),
        SPARTAN_OUTER_R1CS_INPUTS.len() + FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS.len()
    );

    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Stage1Witness;

impl WitnessProvider<Fr, JoltVmNamespace> for Stage1Witness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<OracleDescriptor<JoltVmNamespace>, WitnessError> {
        let OracleKind::Virtual(_) = oracle.kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: JoltVmNamespace::ID.name,
            });
        };
        Ok(OracleDescriptor::new(
            oracle,
            WitnessDimensions::new(16, 4),
            PolynomialEncoding::Dense,
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
    ) -> Result<Vec<ViewRequirement<JoltVmNamespace>>, WitnessError> {
        let descriptor = self.describe_oracle(oracle)?;
        Ok(vec![ViewRequirement::new(
            descriptor.reference,
            descriptor.encoding,
            MaterializationPolicy::BackendChoice,
            RetentionHint::ThroughStage8,
        )])
    }

    fn oracle_view(
        &self,
        request: OracleViewRequest<JoltVmNamespace>,
    ) -> Result<PolynomialView<'_, Fr, JoltVmNamespace>, WitnessError> {
        let descriptor = self.describe_oracle(request.oracle())?;
        let OracleKind::Virtual(variable) = request.oracle().kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: JoltVmNamespace::ID.name,
            });
        };
        let index = SPARTAN_OUTER_R1CS_INPUTS
            .iter()
            .position(|candidate| *candidate == variable)
            .ok_or(WitnessError::UnknownOracle {
                namespace: JoltVmNamespace::ID.name,
            })?;
        Ok(PolynomialView::owned(
            descriptor,
            vec![Fr::from_u64(index as u64 + 1); 16],
        ))
    }
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Copy, Debug)]
struct Stage1FieldInlineWitness;

#[cfg(feature = "field-inline")]
impl WitnessProvider<Fr, FieldInlineNamespace> for Stage1FieldInlineWitness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<FieldInlineNamespace>,
    ) -> Result<OracleDescriptor<FieldInlineNamespace>, WitnessError> {
        let OracleKind::Virtual(_) = oracle.kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: FieldInlineNamespace::ID.name,
            });
        };
        Ok(OracleDescriptor::new(
            oracle,
            WitnessDimensions::new(16, 4),
            PolynomialEncoding::Dense,
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<FieldInlineNamespace>,
    ) -> Result<Vec<ViewRequirement<FieldInlineNamespace>>, WitnessError> {
        let descriptor = self.describe_oracle(oracle)?;
        Ok(vec![ViewRequirement::new(
            descriptor.reference,
            descriptor.encoding,
            MaterializationPolicy::BackendChoice,
            RetentionHint::ThroughBlindFold,
        )])
    }

    fn oracle_view(
        &self,
        request: OracleViewRequest<FieldInlineNamespace>,
    ) -> Result<PolynomialView<'_, Fr, FieldInlineNamespace>, WitnessError> {
        let descriptor = self.describe_oracle(request.oracle())?;
        let OracleKind::Virtual(variable) = request.oracle().kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: FieldInlineNamespace::ID.name,
            });
        };
        let index = FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS
            .iter()
            .position(|candidate| *candidate == variable)
            .ok_or(WitnessError::UnknownOracle {
                namespace: FieldInlineNamespace::ID.name,
            })?;
        Ok(PolynomialView::owned(
            descriptor,
            vec![Fr::from_u64(index as u64 + 41); 16],
        ))
    }
}

#[cfg(feature = "field-inline")]
fn base_r1cs_claims() -> Vec<Stage1R1csInputClaim<Fr>> {
    SPARTAN_OUTER_R1CS_INPUTS
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| Stage1R1csInputClaim {
            variable,
            slot: r1cs_input_slot(index),
            value: Fr::from_u64(index as u64 + 1),
        })
        .collect()
}
