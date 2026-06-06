use serde::{Deserialize, Serialize};

use crate::{perf::PerfGate, FixtureKind, HarnessError, HarnessResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureMode {
    Transparent,
    Zk,
    FieldInline,
    ZkFieldInline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrontierGate {
    VerifierCorrectness,
    CorePerformanceParity,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FrontierSpec {
    pub name: &'static str,
    pub fixtures: &'static [crate::FixtureKind],
    pub features: &'static [FeatureMode],
    pub gates: &'static [FrontierGate],
    pub perf: Option<PerfGate>,
    pub optimization_ids: &'static [&'static str],
    pub backend_kernel_ports: &'static [&'static str],
}

impl FrontierSpec {
    pub fn validate(self) -> HarnessResult<()> {
        if self.name.trim().is_empty() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier name must not be empty".to_owned(),
            });
        }
        if self.fixtures.is_empty() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier must name at least one fixture".to_owned(),
            });
        }
        if has_duplicate(self.fixtures) {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier fixture list contains duplicates".to_owned(),
            });
        }
        if self.features.is_empty() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier must name at least one feature mode".to_owned(),
            });
        }
        if has_duplicate(self.features) {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier feature-mode list contains duplicates".to_owned(),
            });
        }
        if self.gates.is_empty() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier must name at least one harness gate".to_owned(),
            });
        }
        if has_duplicate(self.gates) {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier gate list contains duplicates".to_owned(),
            });
        }
        if !self.requires_verifier_correctness() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier must include jolt-verifier correctness as a gate".to_owned(),
            });
        }
        if !self.requires_core_performance() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier must include core performance parity as a gate".to_owned(),
            });
        }
        if self.optimization_ids.is_empty() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier must name optimization-inventory IDs".to_owned(),
            });
        }
        if has_duplicate(self.optimization_ids) {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier optimization ID list contains duplicates".to_owned(),
            });
        }
        if self.optimization_ids.contains(&"NON-PERF") {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier must name concrete core optimization IDs".to_owned(),
            });
        }
        if self.backend_kernel_ports.is_empty() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier must name backend kernel ledger entries".to_owned(),
            });
        }
        if has_duplicate(self.backend_kernel_ports) {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier backend kernel ledger list contains duplicates".to_owned(),
            });
        }
        if self.perf.is_none() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "core performance parity requires a perf gate".to_owned(),
            });
        }
        if let Some(perf) = self.perf {
            if let Err(reason) = perf.validate() {
                return Err(HarnessError::InvalidManifest {
                    frontier: self.name,
                    reason: reason.to_owned(),
                });
            }
        }
        Ok(())
    }

    pub fn requires_verifier_correctness(self) -> bool {
        self.gates.contains(&FrontierGate::VerifierCorrectness)
    }

    pub fn requires_core_performance(self) -> bool {
        self.gates.contains(&FrontierGate::CorePerformanceParity)
    }
}

fn has_duplicate<T: PartialEq>(values: &[T]) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(index, value)| values[..index].contains(value))
}

#[derive(Clone, Debug, Default)]
pub struct FrontierManifest {
    frontiers: Vec<FrontierSpec>,
}

impl FrontierManifest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, frontier: FrontierSpec) -> HarnessResult<()> {
        frontier.validate()?;
        if self
            .frontiers
            .iter()
            .any(|existing| existing.name == frontier.name)
        {
            return Err(HarnessError::InvalidManifest {
                frontier: frontier.name,
                reason: "duplicate frontier name".to_owned(),
            });
        }
        self.frontiers.push(frontier);
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = &FrontierSpec> {
        self.frontiers.iter()
    }

    pub fn find(&self, name: &str) -> Option<&FrontierSpec> {
        self.frontiers.iter().find(|frontier| frontier.name == name)
    }

    pub fn validate_all(&self) -> HarnessResult<()> {
        for frontier in &self.frontiers {
            frontier.validate()?;
        }
        Ok(())
    }
}

const STAGE0_COMMITMENT_FIXTURES: &[FixtureKind] = &[FixtureKind::MuldivSmall];
const STAGE0_ZK_COMMITMENT_FIXTURES: &[FixtureKind] = &[FixtureKind::ZkMuldivSmall];
const STAGE0_ADVICE_FIXTURES: &[FixtureKind] = &[FixtureKind::AdviceConsumer];
const STAGE0_FIELD_INLINE_FIXTURES: &[FixtureKind] = &[FixtureKind::FieldInlineSmall];
const ZK_BLINDFOLD_FIXTURES: &[FixtureKind] =
    &[FixtureKind::ZkMuldivSmall, FixtureKind::ZkAdviceConsumer];
const STAGE1_SPARTAN_OUTER_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE2_PRODUCT_UNISKIP_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE2_REGULAR_BATCH_INPUT_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE2_REGULAR_BATCH_SUMCHECK_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE2_PRODUCT_REMAINDER_OPENING_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE2_INSTRUCTION_CLAIM_OPENING_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE2_RAM_READ_WRITE_OPENING_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE2_RAM_TERMINAL_OPENING_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE3_REGULAR_BATCH_INPUT_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE3_OUTPUT_OPENING_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE3_REGULAR_BATCH_SUMCHECK_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE4_REGULAR_BATCH_INPUT_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE4_OUTPUT_OPENING_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE4_REGULAR_BATCH_SUMCHECK_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE5_REGULAR_BATCH_INPUT_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE5_OUTPUT_OPENING_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE5_REGULAR_BATCH_SUMCHECK_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE6_REGULAR_BATCH_INPUT_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE6_OUTPUT_OPENING_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE6_REGULAR_BATCH_SUMCHECK_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const TRANSPARENT_FEATURES: &[FeatureMode] = &[FeatureMode::Transparent];
const ZK_FEATURES: &[FeatureMode] = &[FeatureMode::Zk];
const FIELD_INLINE_FEATURES: &[FeatureMode] = &[FeatureMode::FieldInline];
const CORRECTNESS_AND_PERF_GATES: &[FrontierGate] = &[
    FrontierGate::VerifierCorrectness,
    FrontierGate::CorePerformanceParity,
];
const STAGE0_COMMITMENT_OPTIMIZATIONS: &[&str] = &["OPT-COM-001", "OPT-COM-006"];
const STAGE0_ADVICE_OPTIMIZATIONS: &[&str] = &["OPT-COM-005", "OPT-COM-006"];
const STAGE0_FIELD_INLINE_OPTIMIZATIONS: &[&str] = &["OPT-COM-001", "OPT-COM-006"];
const ZK_BLINDFOLD_OPTIMIZATIONS: &[&str] =
    &["OPT-ZK-001", "OPT-ZK-002", "OPT-ZK-003", "OPT-ZK-006"];
const STAGE1_SPARTAN_OUTER_OPTIMIZATIONS: &[&str] = &["OPT-SC-007", "OPT-SC-008", "OPT-EQ-004"];
const STAGE2_PRODUCT_UNISKIP_OPTIMIZATIONS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];
const STAGE2_REGULAR_BATCH_INPUT_OPTIMIZATIONS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];
const STAGE2_REGULAR_BATCH_SUMCHECK_OPTIMIZATIONS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];
const STAGE2_PRODUCT_REMAINDER_OPENING_OPTIMIZATIONS: &[&str] = &["OPT-OPEN-008"];
const STAGE2_INSTRUCTION_CLAIM_OPENING_OPTIMIZATIONS: &[&str] = &["OPT-OPEN-008"];
const STAGE2_RAM_READ_WRITE_OPENING_OPTIMIZATIONS: &[&str] = &["OPT-OPEN-008"];
const STAGE2_RAM_TERMINAL_OPENING_OPTIMIZATIONS: &[&str] = &["OPT-OPEN-008"];
const STAGE3_REGULAR_BATCH_INPUT_OPTIMIZATIONS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];
const STAGE3_OUTPUT_OPENING_OPTIMIZATIONS: &[&str] = &["OPT-OPEN-008"];
const STAGE3_REGULAR_BATCH_SUMCHECK_OPTIMIZATIONS: &[&str] =
    &["OPT-SC-007", "OPT-EQ-004", "OPT-SP-006"];
const STAGE4_REGULAR_BATCH_INPUT_OPTIMIZATIONS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];
const STAGE4_OUTPUT_OPENING_OPTIMIZATIONS: &[&str] = &["OPT-OPEN-008"];
const STAGE4_REGULAR_BATCH_SUMCHECK_OPTIMIZATIONS: &[&str] = &[
    "OPT-SC-007",
    "OPT-EQ-004",
    "OPT-RW-001",
    "OPT-RW-002",
    "OPT-RW-003",
    "OPT-RW-004",
    "OPT-RW-005",
    "OPT-RW-006",
    "OPT-RW-007",
    "OPT-RW-008",
    "OPT-RW-009",
    "OPT-RW-010",
    "OPT-REL-006",
    "OPT-REL-007",
];
const STAGE5_REGULAR_BATCH_INPUT_OPTIMIZATIONS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];
const STAGE5_OUTPUT_OPENING_OPTIMIZATIONS: &[&str] = &["OPT-OPEN-008"];
const STAGE5_REGULAR_BATCH_SUMCHECK_OPTIMIZATIONS: &[&str] = &[
    "OPT-SC-007",
    "OPT-EQ-004",
    "OPT-REL-001",
    "OPT-REL-002",
    "OPT-REL-003",
    "OPT-REL-010",
];
const STAGE6_REGULAR_BATCH_INPUT_OPTIMIZATIONS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];
const STAGE6_OUTPUT_OPENING_OPTIMIZATIONS: &[&str] = &["OPT-OPEN-008"];
const STAGE6_REGULAR_BATCH_SUMCHECK_OPTIMIZATIONS: &[&str] = &[
    "OPT-SC-007",
    "OPT-EQ-004",
    "OPT-RA-003",
    "OPT-RA-007",
    "OPT-RA-008",
    "OPT-REL-004",
    "OPT-REL-005",
    "OPT-REL-011",
    "OPT-REL-014",
];
const STAGE0_COMMITMENT_KERNELS: &[&str] = &["cpu_streaming_commitments"];
const STAGE0_ZK_COMMITMENT_KERNELS: &[&str] = &["cpu_zk_streaming_commitments"];
const STAGE0_ADVICE_KERNELS: &[&str] = &["cpu_advice_commitment_contexts"];
const STAGE0_FIELD_INLINE_KERNELS: &[&str] = &["cpu_field_inline_commitments"];
const ZK_BLINDFOLD_KERNELS: &[&str] = &[
    "cpu_blindfold_round_commitments",
    "cpu_blindfold_backend_kernels",
];
const STAGE1_SPARTAN_OUTER_KERNELS: &[&str] = &["cpu_spartan_outer_prefix_product_sum"];
const STAGE2_PRODUCT_UNISKIP_KERNELS: &[&str] = &["cpu_spartan_product_uniskip"];
const STAGE2_REGULAR_BATCH_INPUT_KERNELS: &[&str] = &["cpu_stage2_regular_batch_input_claims"];
const STAGE2_REGULAR_BATCH_SUMCHECK_KERNELS: &[&str] = &["cpu_stage2_regular_batch_sumcheck"];
const MATERIALIZED_OPENING_KERNELS: &[&str] = &["cpu_materialized_opening_evaluations"];
const STAGE3_REGULAR_BATCH_INPUT_KERNELS: &[&str] = &["cpu_stage3_regular_batch_input_claims"];
const STAGE3_REGULAR_BATCH_SUMCHECK_KERNELS: &[&str] = &["cpu_stage3_regular_batch_sumcheck"];
const STAGE4_REGULAR_BATCH_INPUT_KERNELS: &[&str] = &["cpu_stage4_regular_batch_input_claims"];
const STAGE4_REGULAR_BATCH_SUMCHECK_KERNELS: &[&str] = &["cpu_stage4_regular_batch_sumcheck"];
const STAGE5_REGULAR_BATCH_INPUT_KERNELS: &[&str] = &["cpu_stage5_regular_batch_input_claims"];
const STAGE5_REGULAR_BATCH_SUMCHECK_KERNELS: &[&str] = &["cpu_stage5_regular_batch_sumcheck"];
const STAGE6_REGULAR_BATCH_INPUT_KERNELS: &[&str] = &["cpu_stage6_regular_batch_input_claims"];
const STAGE6_REGULAR_BATCH_SUMCHECK_KERNELS: &[&str] = &["cpu_stage6_regular_batch_sumcheck"];
const STAGE7_REGULAR_BATCH_INPUT_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE7_REGULAR_BATCH_INPUT_OPTIMIZATIONS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];
const STAGE7_REGULAR_BATCH_INPUT_KERNELS: &[&str] = &["cpu_stage7_regular_batch_input_claims"];
const STAGE7_REGULAR_BATCH_SUMCHECK_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE7_REGULAR_BATCH_SUMCHECK_OPTIMIZATIONS: &[&str] = &[
    "OPT-SC-007",
    "OPT-EQ-004",
    "OPT-RA-003",
    "OPT-RA-007",
    "OPT-RA-008",
    "OPT-REL-013",
];
const STAGE7_REGULAR_BATCH_SUMCHECK_KERNELS: &[&str] = &["cpu_stage7_regular_batch_sumcheck"];
const STAGE8_FINAL_OPENING_FIXTURES: &[FixtureKind] =
    &[FixtureKind::MuldivSmall, FixtureKind::AdviceConsumer];
const STAGE8_ZK_FINAL_OPENING_FIXTURES: &[FixtureKind] =
    &[FixtureKind::ZkMuldivSmall, FixtureKind::ZkAdviceConsumer];
const STAGE8_FIELD_INLINE_FINAL_OPENING_FIXTURES: &[FixtureKind] = &[FixtureKind::FieldInlineSmall];
const STAGE8_FINAL_OPENING_OPTIMIZATIONS: &[&str] = &[
    "OPT-COM-006",
    "OPT-POLY-013",
    "OPT-OPEN-001",
    "OPT-OPEN-003",
    "OPT-OPEN-004",
    "OPT-OPEN-005",
    "OPT-OPEN-006",
    "OPT-OPEN-007",
    "OPT-OPEN-009",
];
const STAGE8_ZK_FINAL_OPENING_OPTIMIZATIONS: &[&str] = &[
    "OPT-COM-006",
    "OPT-POLY-013",
    "OPT-OPEN-001",
    "OPT-OPEN-002",
    "OPT-OPEN-003",
    "OPT-OPEN-004",
    "OPT-OPEN-005",
    "OPT-OPEN-006",
    "OPT-OPEN-007",
    "OPT-OPEN-009",
    "OPT-OPEN-010",
];
const STAGE8_FINAL_OPENING_KERNELS: &[&str] = &[
    "cpu_streaming_commitments",
    "cpu_opening_stage8_kernels",
    "cpu_rlc_polynomial_vector_matrix_product",
];

pub fn registered_frontiers() -> HarnessResult<FrontierManifest> {
    let mut manifest = FrontierManifest::new();
    manifest.register(perf_frontier(
        "stage0_commitments",
        STAGE0_COMMITMENT_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE0_COMMITMENT_OPTIMIZATIONS,
        STAGE0_COMMITMENT_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage0_zk_commitments",
        STAGE0_ZK_COMMITMENT_FIXTURES,
        ZK_FEATURES,
        STAGE0_COMMITMENT_OPTIMIZATIONS,
        STAGE0_ZK_COMMITMENT_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage0_advice_commitments",
        STAGE0_ADVICE_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE0_ADVICE_OPTIMIZATIONS,
        STAGE0_ADVICE_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage0_field_inline_commitments",
        STAGE0_FIELD_INLINE_FIXTURES,
        FIELD_INLINE_FEATURES,
        STAGE0_FIELD_INLINE_OPTIMIZATIONS,
        STAGE0_FIELD_INLINE_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "zk_blindfold_core_fixture",
        ZK_BLINDFOLD_FIXTURES,
        ZK_FEATURES,
        ZK_BLINDFOLD_OPTIMIZATIONS,
        ZK_BLINDFOLD_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage1_spartan_outer_requests",
        STAGE1_SPARTAN_OUTER_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE1_SPARTAN_OUTER_OPTIMIZATIONS,
        STAGE1_SPARTAN_OUTER_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage2_product_uniskip",
        STAGE2_PRODUCT_UNISKIP_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE2_PRODUCT_UNISKIP_OPTIMIZATIONS,
        STAGE2_PRODUCT_UNISKIP_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage2_regular_batch_inputs",
        STAGE2_REGULAR_BATCH_INPUT_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE2_REGULAR_BATCH_INPUT_OPTIMIZATIONS,
        STAGE2_REGULAR_BATCH_INPUT_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage2_regular_batch_sumcheck",
        STAGE2_REGULAR_BATCH_SUMCHECK_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE2_REGULAR_BATCH_SUMCHECK_OPTIMIZATIONS,
        STAGE2_REGULAR_BATCH_SUMCHECK_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage2_ram_read_write_openings",
        STAGE2_RAM_READ_WRITE_OPENING_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE2_RAM_READ_WRITE_OPENING_OPTIMIZATIONS,
        MATERIALIZED_OPENING_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage2_ram_terminal_openings",
        STAGE2_RAM_TERMINAL_OPENING_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE2_RAM_TERMINAL_OPENING_OPTIMIZATIONS,
        MATERIALIZED_OPENING_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage2_product_remainder_openings",
        STAGE2_PRODUCT_REMAINDER_OPENING_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE2_PRODUCT_REMAINDER_OPENING_OPTIMIZATIONS,
        MATERIALIZED_OPENING_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage2_instruction_claim_openings",
        STAGE2_INSTRUCTION_CLAIM_OPENING_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE2_INSTRUCTION_CLAIM_OPENING_OPTIMIZATIONS,
        MATERIALIZED_OPENING_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage3_regular_batch_inputs",
        STAGE3_REGULAR_BATCH_INPUT_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE3_REGULAR_BATCH_INPUT_OPTIMIZATIONS,
        STAGE3_REGULAR_BATCH_INPUT_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage3_output_openings",
        STAGE3_OUTPUT_OPENING_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE3_OUTPUT_OPENING_OPTIMIZATIONS,
        MATERIALIZED_OPENING_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage3_regular_batch_sumcheck",
        STAGE3_REGULAR_BATCH_SUMCHECK_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE3_REGULAR_BATCH_SUMCHECK_OPTIMIZATIONS,
        STAGE3_REGULAR_BATCH_SUMCHECK_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage4_regular_batch_inputs",
        STAGE4_REGULAR_BATCH_INPUT_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE4_REGULAR_BATCH_INPUT_OPTIMIZATIONS,
        STAGE4_REGULAR_BATCH_INPUT_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage4_output_openings",
        STAGE4_OUTPUT_OPENING_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE4_OUTPUT_OPENING_OPTIMIZATIONS,
        MATERIALIZED_OPENING_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage4_regular_batch_sumcheck",
        STAGE4_REGULAR_BATCH_SUMCHECK_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE4_REGULAR_BATCH_SUMCHECK_OPTIMIZATIONS,
        STAGE4_REGULAR_BATCH_SUMCHECK_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage5_regular_batch_inputs",
        STAGE5_REGULAR_BATCH_INPUT_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE5_REGULAR_BATCH_INPUT_OPTIMIZATIONS,
        STAGE5_REGULAR_BATCH_INPUT_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage5_output_openings",
        STAGE5_OUTPUT_OPENING_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE5_OUTPUT_OPENING_OPTIMIZATIONS,
        MATERIALIZED_OPENING_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage5_regular_batch_sumcheck",
        STAGE5_REGULAR_BATCH_SUMCHECK_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE5_REGULAR_BATCH_SUMCHECK_OPTIMIZATIONS,
        STAGE5_REGULAR_BATCH_SUMCHECK_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage6_regular_batch_inputs",
        STAGE6_REGULAR_BATCH_INPUT_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE6_REGULAR_BATCH_INPUT_OPTIMIZATIONS,
        STAGE6_REGULAR_BATCH_INPUT_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage6_output_openings",
        STAGE6_OUTPUT_OPENING_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE6_OUTPUT_OPENING_OPTIMIZATIONS,
        MATERIALIZED_OPENING_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage6_regular_batch_sumcheck",
        STAGE6_REGULAR_BATCH_SUMCHECK_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE6_REGULAR_BATCH_SUMCHECK_OPTIMIZATIONS,
        STAGE6_REGULAR_BATCH_SUMCHECK_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage7_regular_batch_inputs",
        STAGE7_REGULAR_BATCH_INPUT_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE7_REGULAR_BATCH_INPUT_OPTIMIZATIONS,
        STAGE7_REGULAR_BATCH_INPUT_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage7_regular_batch_sumcheck",
        STAGE7_REGULAR_BATCH_SUMCHECK_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE7_REGULAR_BATCH_SUMCHECK_OPTIMIZATIONS,
        STAGE7_REGULAR_BATCH_SUMCHECK_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage8_final_opening",
        STAGE8_FINAL_OPENING_FIXTURES,
        TRANSPARENT_FEATURES,
        STAGE8_FINAL_OPENING_OPTIMIZATIONS,
        STAGE8_FINAL_OPENING_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage8_zk_final_opening",
        STAGE8_ZK_FINAL_OPENING_FIXTURES,
        ZK_FEATURES,
        STAGE8_ZK_FINAL_OPENING_OPTIMIZATIONS,
        STAGE8_FINAL_OPENING_KERNELS,
    ))?;
    manifest.register(perf_frontier(
        "stage8_field_inline_final_opening",
        STAGE8_FIELD_INLINE_FINAL_OPENING_FIXTURES,
        FIELD_INLINE_FEATURES,
        STAGE8_FINAL_OPENING_OPTIMIZATIONS,
        STAGE8_FINAL_OPENING_KERNELS,
    ))?;
    Ok(manifest)
}

fn perf_frontier(
    name: &'static str,
    fixtures: &'static [FixtureKind],
    features: &'static [FeatureMode],
    optimization_ids: &'static [&'static str],
    backend_kernel_ports: &'static [&'static str],
) -> FrontierSpec {
    FrontierSpec {
        name,
        fixtures,
        features,
        gates: CORRECTNESS_AND_PERF_GATES,
        perf: Some(PerfGate::canonical_frontier()),
        optimization_ids,
        backend_kernel_ports,
    }
}
