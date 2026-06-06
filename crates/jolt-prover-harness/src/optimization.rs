use std::collections::{BTreeMap, BTreeSet};

use crate::{FrontierSpec, HarnessError, HarnessResult};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KnownOptimizationIds {
    ids: BTreeSet<String>,
    port_targets_by_id: BTreeMap<String, BTreeSet<String>>,
}

impl KnownOptimizationIds {
    pub fn parse_inventory(markdown: &str) -> HarnessResult<Self> {
        let mut ids = BTreeSet::new();
        let mut port_targets_by_id = BTreeMap::new();
        for line in markdown.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("| OPT-") {
                continue;
            }
            let columns = trimmed.split('|').map(str::trim).collect::<Vec<_>>();
            let Some(id) = columns.get(1).copied() else {
                continue;
            };
            if !ids.insert(id.to_owned()) {
                return Err(HarnessError::InvalidOptimizationInventory {
                    reason: format!("duplicate optimization ID `{id}`"),
                });
            }
            let Some(port_target) = columns.get(5).copied() else {
                return Err(HarnessError::InvalidOptimizationInventory {
                    reason: format!("optimization ID `{id}` has no port target column"),
                });
            };
            let targets = port_target
                .split('+')
                .map(str::trim)
                .filter(|target| !target.is_empty())
                .map(str::to_owned)
                .collect::<BTreeSet<_>>();
            if targets.is_empty() {
                return Err(HarnessError::InvalidOptimizationInventory {
                    reason: format!("optimization ID `{id}` has no parsed port targets"),
                });
            }
            let _ = port_targets_by_id.insert(id.to_owned(), targets);
        }
        if ids.is_empty() {
            return Err(HarnessError::InvalidOptimizationInventory {
                reason: "optimization inventory contains no OPT-* IDs".to_owned(),
            });
        }
        Ok(Self {
            ids,
            port_targets_by_id,
        })
    }

    pub fn contains(&self, id: &str) -> bool {
        self.ids.contains(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.ids.iter().map(String::as_str)
    }

    pub fn port_targets(&self, id: &str) -> Option<impl Iterator<Item = &str>> {
        self.port_targets_by_id
            .get(id)
            .map(|targets| targets.iter().map(String::as_str))
    }

    pub fn requires_cpu_backend(&self, id: &str) -> bool {
        self.port_targets_by_id
            .get(id)
            .is_some_and(|targets| targets.contains("cpu-backend"))
    }
}

pub fn validate_frontier_optimization_ids(
    frontier: FrontierSpec,
    known: &KnownOptimizationIds,
) -> HarnessResult<()> {
    frontier.validate()?;
    for id in frontier.optimization_ids {
        if known.contains(id) {
            continue;
        }
        return Err(HarnessError::InvalidManifest {
            frontier: frontier.name,
            reason: format!("unknown optimization-inventory ID `{id}`"),
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum KernelPortStatus {
    Required,
    Ported,
    ParityCertified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKernelFamily {
    Commitments,
    Sumcheck,
    Openings,
    BlindFold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendKernelPortSpec {
    pub name: &'static str,
    pub family: BackendKernelFamily,
    pub optimization_ids: &'static [&'static str],
    pub source_locations: &'static [&'static str],
    pub cpu_entrypoints: &'static [&'static str],
    pub microbenchmarks: &'static [&'static str],
    pub certification_evidence_files: &'static [&'static str],
    pub status: KernelPortStatus,
}

impl BackendKernelPortSpec {
    pub fn validate(self, known: &KnownOptimizationIds) -> HarnessResult<()> {
        if self.name.trim().is_empty() {
            return Err(HarnessError::InvalidKernelInventory {
                kernel: self.name,
                reason: "kernel port name must not be empty".to_owned(),
            });
        }
        if self.optimization_ids.is_empty() {
            return Err(HarnessError::InvalidKernelInventory {
                kernel: self.name,
                reason: "kernel port must name optimization IDs".to_owned(),
            });
        }
        if self.source_locations.is_empty() {
            return Err(HarnessError::InvalidKernelInventory {
                kernel: self.name,
                reason: "kernel port must name core source locations".to_owned(),
            });
        }
        if self.cpu_entrypoints.is_empty() {
            return Err(HarnessError::InvalidKernelInventory {
                kernel: self.name,
                reason: "kernel port must name CPU backend entrypoints".to_owned(),
            });
        }
        if self.microbenchmarks.is_empty() {
            return Err(HarnessError::InvalidKernelInventory {
                kernel: self.name,
                reason: "kernel port must name microbenchmarks".to_owned(),
            });
        }
        if self.status >= KernelPortStatus::ParityCertified
            && self.certification_evidence_files.is_empty()
        {
            return Err(HarnessError::InvalidKernelInventory {
                kernel: self.name,
                reason: "parity-certified kernel must name benchmark evidence files".to_owned(),
            });
        }
        if has_duplicate(self.optimization_ids) {
            return Err(HarnessError::InvalidKernelInventory {
                kernel: self.name,
                reason: "kernel port optimization ID list contains duplicates".to_owned(),
            });
        }
        if has_duplicate(self.certification_evidence_files) {
            return Err(HarnessError::InvalidKernelInventory {
                kernel: self.name,
                reason: "kernel benchmark evidence file list contains duplicates".to_owned(),
            });
        }
        for id in self.optimization_ids {
            if !known.contains(id) {
                return Err(HarnessError::InvalidKernelInventory {
                    kernel: self.name,
                    reason: format!("unknown optimization-inventory ID `{id}`"),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct BackendKernelPortLedger {
    ports: Vec<BackendKernelPortSpec>,
}

impl BackendKernelPortLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        port: BackendKernelPortSpec,
        known: &KnownOptimizationIds,
    ) -> HarnessResult<()> {
        port.validate(known)?;
        if self.ports.iter().any(|existing| existing.name == port.name) {
            return Err(HarnessError::InvalidKernelInventory {
                kernel: port.name,
                reason: "duplicate backend kernel port name".to_owned(),
            });
        }
        self.ports.push(port);
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = &BackendKernelPortSpec> {
        self.ports.iter()
    }

    pub fn find(&self, name: &str) -> Option<&BackendKernelPortSpec> {
        self.ports.iter().find(|port| port.name == name)
    }

    pub fn status_for_optimization(&self, id: &str) -> Option<KernelPortStatus> {
        self.ports
            .iter()
            .filter(|port| port.optimization_ids.contains(&id))
            .map(|port| port.status)
            .max()
    }

    pub fn covers_optimization(&self, id: &str, minimum: KernelPortStatus) -> bool {
        self.status_for_optimization(id)
            .is_some_and(|status| status >= minimum)
    }
}

pub fn registered_backend_kernel_ports(
    known: &KnownOptimizationIds,
) -> HarnessResult<BackendKernelPortLedger> {
    let mut ledger = BackendKernelPortLedger::new();
    for port in BACKEND_KERNEL_PORTS {
        ledger.register(*port, known)?;
    }
    Ok(ledger)
}

pub fn validate_frontier_kernel_accounting(
    frontier: FrontierSpec,
    known: &KnownOptimizationIds,
    ledger: &BackendKernelPortLedger,
    minimum: KernelPortStatus,
) -> HarnessResult<()> {
    validate_frontier_optimization_ids(frontier, known)?;
    for port_name in frontier.backend_kernel_ports {
        if ledger.find(port_name).is_some() {
            continue;
        }
        return Err(HarnessError::InvalidManifest {
            frontier: frontier.name,
            reason: format!("unknown backend kernel ledger entry `{port_name}`"),
        });
    }
    for id in frontier.optimization_ids {
        if !known.requires_cpu_backend(id) {
            continue;
        }
        if frontier.backend_kernel_ports.iter().any(|port_name| {
            ledger
                .find(port_name)
                .is_some_and(|port| port.optimization_ids.contains(id) && port.status >= minimum)
        }) {
            continue;
        }
        return Err(HarnessError::InvalidManifest {
            frontier: frontier.name,
            reason: format!(
                "optimization `{id}` requires a `{minimum:?}` CPU backend kernel ledger entry"
            ),
        });
    }
    Ok(())
}

pub fn validate_global_cpu_backend_inventory_coverage(
    known: &KnownOptimizationIds,
    ledger: &BackendKernelPortLedger,
) -> HarnessResult<()> {
    let missing = known
        .iter()
        .filter(|id| known.requires_cpu_backend(id))
        .filter(|id| !ledger.covers_optimization(id, KernelPortStatus::Required))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }
    Err(HarnessError::InvalidOptimizationInventory {
        reason: format!(
            "cpu-backend optimization IDs missing backend kernel ledger entries: {}",
            missing.join(", ")
        ),
    })
}

const BACKEND_KERNEL_PORTS: &[BackendKernelPortSpec] = &[
    BackendKernelPortSpec {
        name: "cpu_streaming_commitments",
        family: BackendKernelFamily::Commitments,
        optimization_ids: &["OPT-COM-001", "OPT-COM-006"],
        source_locations: &[
            "jolt-core/src/zkvm/prover.rs::generate_and_commit_witness_polynomials",
            "jolt-core/src/poly/opening_proof.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::commitments::commit",
            "jolt_backends::cpu::commitments::stream::commit_streamed_witness",
        ],
        microbenchmarks: &["frontier_perf/stage0_commitments"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_streaming_commitments/frontier_perf_stage0_commitments.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_advice_commitment_contexts",
        family: BackendKernelFamily::Commitments,
        optimization_ids: &["OPT-COM-005", "OPT-COM-006"],
        source_locations: &[
            "jolt-core/src/zkvm/prover.rs::generate_and_commit_trusted_advice",
            "jolt-core/src/zkvm/prover.rs::generate_and_commit_untrusted_advice",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::commitments::commit"],
        microbenchmarks: &["frontier_perf/stage0_advice_commitments"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_advice_commitment_contexts/frontier_perf_stage0_advice_commitments.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_zk_streaming_commitments",
        family: BackendKernelFamily::Commitments,
        optimization_ids: &["OPT-COM-001", "OPT-COM-006"],
        source_locations: &[
            "jolt-core/src/zkvm/prover.rs::generate_and_commit_witness_polynomials",
            "jolt-core/src/poly/commitment/dory/commitment_scheme.rs::maybe_blind_commitment",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::commitments::commit",
            "jolt_backends::cpu::commitments::stream",
            "jolt_dory::DoryScheme::finish_zk",
        ],
        microbenchmarks: &["frontier_perf/stage0_zk_commitments"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_zk_streaming_commitments/frontier_perf_stage0_zk_commitments.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_field_inline_commitments",
        family: BackendKernelFamily::Commitments,
        optimization_ids: &["OPT-COM-001", "OPT-COM-006"],
        source_locations: &[
            "jolt-core/src/zkvm/prover.rs::generate_and_commit_witness_polynomials",
            "jolt-core/src/zkvm/field_inline",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::commitments::commit",
            "jolt_backends::cpu::field_inline",
        ],
        microbenchmarks: &["frontier_perf/stage0_field_inline_commitments"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_field_inline_commitments/frontier_perf_stage0_field_inline_commitments.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_spartan_outer_prefix_product_sum",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-SC-007", "OPT-SC-008", "OPT-EQ-004"],
        source_locations: &[
            "jolt-core/src/zkvm/spartan/outer.rs",
            "jolt-core/src/zkvm/r1cs/evaluation.rs",
            "jolt-core/src/poly/split_eq_poly.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::sumcheck::kernels::spartan_outer::evaluate_prefix_product_sums",
            "jolt_backends::cpu::sumcheck::kernels::spartan_outer::evaluate_spartan_outer_uniskip_rows",
        ],
        microbenchmarks: &[
            "cpu_sumcheck/outer_uniskip_prefix_sum",
            "cpu_sumcheck/outer_remainder_bound_prefix_sum",
        ],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_spartan_outer_prefix_product_sum/cpu_sumcheck_outer_uniskip_prefix_sum.json",
            "target/frontier-metrics/kernel-evidence/cpu_spartan_outer_prefix_product_sum/cpu_sumcheck_outer_remainder_bound_prefix_sum.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_spartan_product_uniskip",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-SC-007", "OPT-EQ-004"],
        source_locations: &[
            "jolt-core/src/zkvm/spartan/product.rs",
            "jolt-core/src/zkvm/r1cs/evaluation.rs::ProductVirtualEval",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::sumcheck::kernels::spartan_product::evaluate_row_products",
            "jolt_backends::cpu::sumcheck::kernels::spartan_product::evaluate_product_uniskip_rows",
        ],
        microbenchmarks: &[
            "cpu_sumcheck/spartan_product_uniskip",
            "cpu_sumcheck/spartan_product_uniskip_raw",
        ],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_spartan_product_uniskip/cpu_sumcheck_spartan_product_uniskip_raw.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_stage2_regular_batch_input_claims",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-SC-007", "OPT-EQ-004"],
        source_locations: &[
            "jolt-core/src/zkvm/ram/read_write_checking.rs",
            "jolt-core/src/zkvm/instruction_lookups/ra_virtual.rs",
            "jolt-core/src/zkvm/spartan/product.rs",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::sumcheck"],
        microbenchmarks: &["frontier_perf/stage2_regular_batch_inputs"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_stage2_regular_batch_input_claims/frontier_perf_stage2_regular_batch_inputs.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_stage2_regular_batch_sumcheck",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-SC-007", "OPT-EQ-004"],
        source_locations: &[
            "jolt-core/src/zkvm/ram/read_write_checking.rs",
            "jolt-core/src/zkvm/claim_reductions/instruction_lookups.rs",
            "jolt-core/src/zkvm/spartan/product.rs",
            "jolt-core/src/zkvm/ram/raf_evaluation.rs",
            "jolt-core/src/zkvm/ram/output_check.rs",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::sumcheck"],
        microbenchmarks: &["frontier_perf/stage2_regular_batch_sumcheck"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_stage2_regular_batch_sumcheck/frontier_perf_stage2_regular_batch_sumcheck.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_stage3_regular_batch_input_claims",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-SC-007", "OPT-EQ-004"],
        source_locations: &[
            "jolt-core/src/zkvm/spartan/shift.rs",
            "jolt-core/src/zkvm/spartan/instruction_input.rs",
            "jolt-core/src/zkvm/registers/read_write_checking.rs",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::sumcheck"],
        microbenchmarks: &["frontier_perf/stage3_regular_batch_inputs"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_stage3_regular_batch_input_claims/frontier_perf_stage3_regular_batch_inputs.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_stage3_regular_batch_sumcheck",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-SC-007", "OPT-EQ-004", "OPT-SP-006"],
        source_locations: &[
            "jolt-core/src/zkvm/spartan/shift.rs",
            "jolt-core/src/zkvm/spartan/instruction_input.rs",
            "jolt-core/src/zkvm/registers/read_write_checking.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::sumcheck",
            "jolt_backends::cpu::sumcheck::kernels::stage3_shift",
        ],
        microbenchmarks: &["frontier_perf/stage3_regular_batch_sumcheck"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_stage3_regular_batch_sumcheck/frontier_perf_stage3_regular_batch_sumcheck.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_stage4_regular_batch_input_claims",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-SC-007", "OPT-EQ-004"],
        source_locations: &[
            "jolt-core/src/zkvm/ram/val_check.rs",
            "jolt-core/src/zkvm/bytecode/read_raf_checking.rs",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::sumcheck"],
        microbenchmarks: &["frontier_perf/stage4_regular_batch_inputs"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_stage4_regular_batch_input_claims/frontier_perf_stage4_regular_batch_inputs.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_stage4_regular_batch_sumcheck",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
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
        ],
        source_locations: &[
            "jolt-core/src/subprotocols/read_write_matrix",
            "jolt-core/src/zkvm/registers/read_write_checking.rs",
            "jolt-core/src/zkvm/ram/val_check.rs",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::sumcheck"],
        microbenchmarks: &["frontier_perf/stage4_regular_batch_sumcheck"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_stage4_regular_batch_sumcheck/frontier_perf_stage4_regular_batch_sumcheck.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_field_inline_stage4_registers_read_write",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
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
            "OPT-FLD-002",
            "OPT-FLD-003",
        ],
        source_locations: &[
            "jolt-core/src/subprotocols/read_write_matrix",
            "jolt-core/src/zkvm/field_inline",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::read_write_matrix::FieldRegistersReadWriteState",
            "jolt_backends::cpu::sumcheck::Stage4ReadWriteSumcheckBackend",
        ],
        microbenchmarks: &["frontier_perf/stage4_field_inline_registers_read_write"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_field_inline_stage4_registers_read_write/frontier_perf_stage4_field_inline_registers_read_write.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_stage5_regular_batch_input_claims",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-SC-007", "OPT-EQ-004"],
        source_locations: &[
            "jolt-core/src/zkvm/registers/val_evaluation.rs",
            "jolt-core/src/zkvm/bytecode/read_raf_checking.rs",
            "jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::sumcheck"],
        microbenchmarks: &["frontier_perf/stage5_regular_batch_inputs"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_stage5_regular_batch_input_claims/frontier_perf_stage5_regular_batch_inputs.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_stage5_regular_batch_sumcheck",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
            "OPT-SC-007",
            "OPT-EQ-004",
            "OPT-REL-001",
            "OPT-REL-002",
            "OPT-REL-003",
            "OPT-REL-010",
        ],
        source_locations: &[
            "jolt-core/src/zkvm/registers/val_evaluation.rs",
            "jolt-core/src/zkvm/bytecode/read_raf_checking.rs",
            "jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::sumcheck"],
        microbenchmarks: &["frontier_perf/stage5_regular_batch_sumcheck"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_stage5_regular_batch_sumcheck/frontier_perf_stage5_regular_batch_sumcheck.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_field_inline_stage5_registers_val_evaluation",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-FLD-003", "OPT-REL-010"],
        source_locations: &[
            "jolt-core/src/zkvm/field_inline",
            "jolt-core/src/zkvm/registers/val_evaluation.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::read_write_matrix::FieldRegistersValEvaluationState",
            "jolt_backends::cpu::sumcheck::Stage5ValueEvaluationSumcheckBackend",
        ],
        microbenchmarks: &["frontier_perf/stage5_field_inline_registers_val_evaluation"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_field_inline_stage5_registers_val_evaluation/frontier_perf_stage5_field_inline_registers_val_evaluation.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_stage6_regular_batch_input_claims",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-SC-007", "OPT-EQ-004"],
        source_locations: &[
            "jolt-core/src/zkvm/claim_reductions/increments.rs",
            "jolt-core/src/zkvm/ram/hamming_booleanity.rs",
            "jolt-core/src/zkvm/claim_reductions/advice.rs",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::sumcheck"],
        microbenchmarks: &["frontier_perf/stage6_regular_batch_inputs"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_stage6_regular_batch_input_claims/frontier_perf_stage6_regular_batch_inputs.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_stage6_regular_batch_sumcheck",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
            "OPT-SC-007",
            "OPT-EQ-004",
            "OPT-RA-003",
            "OPT-RA-007",
            "OPT-RA-008",
            "OPT-REL-004",
            "OPT-REL-005",
            "OPT-REL-011",
            "OPT-REL-014",
        ],
        source_locations: &[
            "jolt-core/src/zkvm/bytecode/read_raf_checking.rs",
            "jolt-core/src/subprotocols/booleanity.rs",
            "jolt-core/src/zkvm/claim_reductions/increments.rs",
            "jolt-core/src/zkvm/ram/hamming_booleanity.rs",
            "jolt-core/src/zkvm/ram/ra_virtual.rs",
            "jolt-core/src/zkvm/instruction_lookups/ra_virtual.rs",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::sumcheck"],
        microbenchmarks: &["frontier_perf/stage6_regular_batch_sumcheck"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_stage6_regular_batch_sumcheck/frontier_perf_stage6_regular_batch_sumcheck.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_field_inline_stage6_registers_inc_claim_reduction",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-FLD-003", "OPT-REL-011"],
        source_locations: &[
            "jolt-core/src/zkvm/field_inline",
            "jolt-core/src/zkvm/claim_reductions/increments.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::read_write_matrix::FieldRegistersIncClaimReductionState",
            "jolt_backends::cpu::sumcheck::Stage6RegularBatchSumcheckBackend",
        ],
        microbenchmarks: &["frontier_perf/stage6_field_inline_registers_inc_claim_reduction"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_field_inline_stage6_registers_inc_claim_reduction/frontier_perf_stage6_field_inline_registers_inc_claim_reduction.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_stage7_regular_batch_input_claims",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-SC-007", "OPT-EQ-004"],
        source_locations: &["jolt-core/src/zkvm/claim_reductions/hamming_weight.rs"],
        cpu_entrypoints: &["jolt_backends::cpu::sumcheck"],
        microbenchmarks: &["frontier_perf/stage7_regular_batch_inputs"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_stage7_regular_batch_input_claims/frontier_perf_stage7_regular_batch_inputs.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_stage7_regular_batch_sumcheck",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
            "OPT-SC-007",
            "OPT-EQ-004",
            "OPT-RA-003",
            "OPT-RA-007",
            "OPT-RA-008",
            "OPT-REL-013",
        ],
        source_locations: &[
            "jolt-core/src/zkvm/claim_reductions/hamming_weight.rs",
            "jolt-core/src/zkvm/claim_reductions/advice.rs",
            "jolt-core/src/poly/shared_ra_polys.rs",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::sumcheck"],
        microbenchmarks: &["frontier_perf/stage7_regular_batch_sumcheck"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_stage7_regular_batch_sumcheck/frontier_perf_stage7_regular_batch_sumcheck.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_materialized_opening_evaluations",
        family: BackendKernelFamily::Openings,
        optimization_ids: &["OPT-OPEN-008"],
        source_locations: &[
            "jolt-core/src/poly/rlc_polynomial.rs::materialize_from_context",
            "jolt-core/src/poly/rlc_polynomial.rs::address_major_vmp",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::openings::materialize_opening_rlc"],
        microbenchmarks: &["cpu_openings/rlc_materialized_fallback"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_materialized_opening_evaluations/cpu_openings_rlc_materialized_fallback.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_blindfold_round_commitments",
        family: BackendKernelFamily::BlindFold,
        optimization_ids: &["OPT-ZK-001"],
        source_locations: &[
            "jolt-core/src/subprotocols/sumcheck.rs::prove_zk",
            "jolt-core/src/subprotocols/univariate_skip.rs",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::blindfold"],
        microbenchmarks: &["frontier_perf/zk_blindfold_core_fixture"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_blindfold_round_commitments/frontier_perf_zk_blindfold_core_fixture.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_trace_shape_and_instrumentation",
        family: BackendKernelFamily::Commitments,
        optimization_ids: &["OPT-GEN-002", "OPT-GEN-003", "OPT-GEN-006", "OPT-GEN-007"],
        source_locations: &[
            "jolt-core/src/zkvm/prover.rs::adjust_trace_length_for_advice",
            "jolt-core/src/zkvm/prover.rs::gen_from_trace",
            "jolt-core/src/poly/commitment/dory/dory_globals.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::commitments",
            "jolt_backends::cpu::instrumentation",
        ],
        microbenchmarks: &["frontier_perf/trace_shape_and_instrumentation"],
        certification_evidence_files: &[],
        status: KernelPortStatus::Required,
    },
    BackendKernelPortSpec {
        name: "cpu_dory_commitment_backlog",
        family: BackendKernelFamily::Commitments,
        optimization_ids: &["OPT-COM-002", "OPT-COM-003", "OPT-COM-004", "OPT-COM-010"],
        source_locations: &[
            "jolt-core/src/zkvm/prover.rs::generate_and_commit_witness_polynomials",
            "jolt-core/src/poly/commitment/dory/commitment_scheme.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::commitments::commit",
            "jolt_backends::cpu::commitments::stream",
        ],
        microbenchmarks: &["frontier_perf/dory_commitment_backlog"],
        certification_evidence_files: &[],
        status: KernelPortStatus::Required,
    },
    BackendKernelPortSpec {
        name: "cpu_one_hot_commitments",
        family: BackendKernelFamily::Commitments,
        optimization_ids: &["OPT-COM-007", "OPT-COM-008"],
        source_locations: &[
            "jolt-core/src/poly/one_hot_polynomial.rs::commit_rows",
            "jolt-core/src/poly/commitment/dory/commitment_scheme.rs",
        ],
        cpu_entrypoints: &[
            "jolt_dory::DoryScheme::commit",
            "jolt_backends::cpu::commitments::stream::commit_streamed_witness",
        ],
        microbenchmarks: &["frontier_perf/one_hot_commitments"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_one_hot_commitments/frontier_perf_one_hot_commitments.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_polynomial_representation_kernels",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
            "OPT-POLY-001",
            "OPT-POLY-002",
            "OPT-POLY-003",
            "OPT-POLY-004",
            "OPT-POLY-005",
            "OPT-POLY-006",
            "OPT-POLY-007",
            "OPT-POLY-008",
            "OPT-POLY-009",
            "OPT-POLY-010",
            "OPT-POLY-011",
            "OPT-POLY-012",
            "OPT-POLY-013",
            "OPT-POLY-014",
        ],
        source_locations: &[
            "jolt-core/src/poly/multilinear_polynomial.rs",
            "jolt-core/src/poly/compact_polynomial.rs",
            "jolt-core/src/poly/dense_mlpoly.rs",
            "jolt-core/src/poly/one_hot_polynomial.rs",
            "jolt-core/src/poly/rlc_polynomial.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::poly",
            "jolt_backends::cpu::sumcheck::kernels",
        ],
        microbenchmarks: &[
            "cpu_poly/compact_bind",
            "cpu_poly/split_eq_evaluate",
            "cpu_poly/inside_out_evaluate",
            "cpu_poly/dense_batch_evaluate",
            "cpu_poly/dense_dot_product_low_optimized",
            "cpu_poly/linear_combination",
            "cpu_poly/one_hot_evaluate",
            "cpu_poly/one_hot_vmp",
            "cpu_poly/rlc_vmp",
            "frontier_perf/stage8_rlc",
        ],
        certification_evidence_files: &[],
        status: KernelPortStatus::Required,
    },
    BackendKernelPortSpec {
        name: "cpu_compact_polynomial_bind",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
            "OPT-POLY-002",
            "OPT-POLY-003",
            "OPT-POLY-004",
            "OPT-POLY-005",
        ],
        source_locations: &[
            "jolt-core/src/poly/compact_polynomial.rs::bind",
            "jolt-core/src/poly/compact_polynomial.rs::bind_parallel",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::poly::bind_compact_first_high_to_low",
            "jolt_backends::cpu::poly::bind_compact_first_low_to_high",
            "jolt_backends::cpu::poly::bind_field_high_to_low",
            "jolt_backends::cpu::poly::bind_field_low_to_high",
        ],
        microbenchmarks: &["cpu_poly/compact_bind"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_compact_polynomial_bind/cpu_poly_compact_bind.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_split_eq_polynomial_evaluation",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-POLY-006", "OPT-POLY-007"],
        source_locations: &[
            "jolt-core/src/poly/dense_mlpoly.rs::split_eq_evaluate",
            "jolt-core/src/poly/compact_polynomial.rs::split_eq_evaluate",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::poly::dense_split_eq_evaluate",
            "jolt_backends::cpu::poly::compact_split_eq_evaluate",
        ],
        microbenchmarks: &["cpu_poly/split_eq_evaluate"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_split_eq_polynomial_evaluation/cpu_poly_split_eq_evaluate.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_inside_out_polynomial_evaluation",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-POLY-008"],
        source_locations: &[
            "jolt-core/src/poly/dense_mlpoly.rs::inside_out_evaluate",
            "jolt-core/src/poly/compact_polynomial.rs::inside_out_evaluate",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::poly::dense_inside_out_evaluate",
            "jolt_backends::cpu::poly::compact_inside_out_evaluate",
        ],
        microbenchmarks: &["cpu_poly/inside_out_evaluate"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_inside_out_polynomial_evaluation/cpu_poly_inside_out_evaluate.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_dense_batch_polynomial_evaluation",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-POLY-009"],
        source_locations: &["jolt-core/src/poly/dense_mlpoly.rs::batch_evaluate"],
        cpu_entrypoints: &[
            "jolt_backends::cpu::poly::dense_batch_evaluate",
            "jolt_backends::cpu::poly::dense_batch_split_eq_evaluate",
        ],
        microbenchmarks: &["cpu_poly/dense_batch_evaluate"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_dense_batch_polynomial_evaluation/cpu_poly_dense_batch_evaluate.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_dense_dot_product_low_optimized",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-POLY-014"],
        source_locations: &[
            "jolt-core/src/poly/dense_mlpoly.rs::evaluate_at_chi_low_optimized",
            "jolt-core/src/utils/mod.rs::compute_dotproduct_low_optimized",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::poly::dense_dot_product_low_optimized"],
        microbenchmarks: &["cpu_poly/dense_dot_product_low_optimized"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_dense_dot_product_low_optimized/cpu_poly_dense_dot_product_low_optimized.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_mixed_polynomial_linear_combination",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-POLY-010"],
        source_locations: &[
            "jolt-core/src/poly/dense_mlpoly.rs::linear_combination",
            "jolt-core/src/poly/multilinear_polynomial.rs::get_scaled_coeff",
        ],
        cpu_entrypoints: &["jolt_backends::cpu::poly::linear_combination"],
        microbenchmarks: &["cpu_poly/linear_combination"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_mixed_polynomial_linear_combination/cpu_poly_linear_combination.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_one_hot_polynomial_evaluation",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-POLY-011"],
        source_locations: &["jolt-core/src/poly/one_hot_polynomial.rs::evaluate"],
        cpu_entrypoints: &["jolt_backends::cpu::poly::one_hot_evaluate"],
        microbenchmarks: &["cpu_poly/one_hot_evaluate"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_one_hot_polynomial_evaluation/cpu_poly_one_hot_evaluate.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_one_hot_vector_matrix_product",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-POLY-012"],
        source_locations: &["jolt-core/src/poly/one_hot_polynomial.rs::vector_matrix_product"],
        cpu_entrypoints: &["jolt_backends::cpu::poly::one_hot_vector_matrix_product"],
        microbenchmarks: &["cpu_poly/one_hot_vmp"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_one_hot_vector_matrix_product/cpu_poly_one_hot_vmp.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_rlc_polynomial_vector_matrix_product",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-POLY-013"],
        source_locations: &["jolt-core/src/poly/rlc_polynomial.rs::vector_matrix_product"],
        cpu_entrypoints: &["jolt_backends::cpu::poly::materialized_rlc_vector_matrix_product"],
        microbenchmarks: &["cpu_poly/rlc_vmp"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_rlc_polynomial_vector_matrix_product/cpu_poly_rlc_vmp.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_eq_table_generation",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-EQ-001", "OPT-EQ-002"],
        source_locations: &[
            "jolt-core/src/poly/eq_poly.rs::evals",
            "jolt-core/src/poly/eq_poly.rs::evals_cached",
            "jolt-core/src/poly/eq_poly.rs::evals_cached_rev",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::eq::evals",
            "jolt_backends::cpu::eq::evals_cached",
            "jolt_backends::cpu::eq::evals_cached_rev",
        ],
        microbenchmarks: &["cpu_sumcheck/eq_tables"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_eq_table_generation/cpu_sumcheck_eq_tables.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_eq_aligned_block_generation",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-EQ-003"],
        source_locations: &[
            "jolt-core/src/poly/eq_poly.rs::evals_for_aligned_block",
            "jolt-core/src/poly/eq_poly.rs::evals_for_max_aligned_block",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::eq::evals_for_aligned_block",
            "jolt_backends::cpu::eq::evals_for_max_aligned_block",
        ],
        microbenchmarks: &["cpu_sumcheck/eq_aligned_blocks"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_eq_aligned_block_generation/cpu_sumcheck_eq_aligned_blocks.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_split_eq_streaming_windows",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-EQ-005"],
        source_locations: &[
            "jolt-core/src/poly/split_eq_poly.rs::E_out_in_for_window",
            "jolt-core/src/poly/split_eq_poly.rs::E_active_for_window",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::split_eq::e_out_in_for_window",
            "jolt_backends::cpu::split_eq::e_active_for_window",
        ],
        microbenchmarks: &["cpu_sumcheck/split_eq_windows"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_split_eq_streaming_windows/cpu_sumcheck_split_eq_windows.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_unipoly_interpolation",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-EQ-006", "OPT-EQ-007"],
        source_locations: &[
            "jolt-core/src/poly/unipoly.rs::from_evals",
            "jolt-core/src/poly/unipoly.rs::from_evals_and_hint",
            "jolt-core/src/poly/unipoly.rs::from_evals_toom",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::univariate::from_evals",
            "jolt_backends::cpu::univariate::from_evals_and_hint",
            "jolt_backends::cpu::univariate::from_evals_toom",
        ],
        microbenchmarks: &["cpu_sumcheck/unipoly_interpolation"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_unipoly_interpolation/cpu_sumcheck_unipoly_interpolation.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_compressed_unipoly",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-EQ-008"],
        source_locations: &[
            "jolt-core/src/poly/unipoly.rs::compress",
            "jolt-core/src/poly/unipoly.rs::CompressedUniPoly::decompress",
            "jolt-core/src/poly/unipoly.rs::CompressedUniPoly::eval_from_hint",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::univariate::compress",
            "jolt_backends::cpu::univariate::decompress",
            "jolt_backends::cpu::univariate::eval_from_hint",
        ],
        microbenchmarks: &["cpu_sumcheck/compressed_unipoly"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_compressed_unipoly/cpu_sumcheck_compressed_unipoly.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_lagrange_many",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-EQ-009"],
        source_locations: &[
            "jolt-core/src/poly/lagrange_poly.rs::LagrangePolynomial::evals",
            "jolt-core/src/poly/lagrange_poly.rs::LagrangePolynomial::lagrange_kernel",
            "jolt-core/src/poly/lagrange_poly.rs::LagrangePolynomial::evaluate_many",
            "jolt-core/src/poly/lagrange_poly.rs::LagrangePolynomial::interpolate_coeffs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::lagrange::centered_evals",
            "jolt_backends::cpu::lagrange::centered_kernel",
            "jolt_backends::cpu::lagrange::centered_evaluate_many",
            "jolt_backends::cpu::lagrange::centered_interpolate_coeffs",
        ],
        microbenchmarks: &["cpu_sumcheck/lagrange_many"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_lagrange_many/cpu_sumcheck_lagrange_many.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_ra_one_hot_pushforward_kernels",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
            "OPT-RA-001",
            "OPT-RA-002",
            "OPT-RA-003",
            "OPT-RA-004",
            "OPT-RA-005",
            "OPT-RA-006",
            "OPT-RA-007",
            "OPT-RA-008",
            "OPT-RA-009",
            "OPT-RA-010",
        ],
        source_locations: &[
            "jolt-core/src/poly/shared_ra_polys.rs",
            "jolt-core/src/poly/ra_poly.rs",
            "jolt-core/src/zkvm/witness.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::ra",
            "jolt_backends::cpu::sumcheck::kernels",
        ],
        microbenchmarks: &[
            "cpu_sumcheck/ra_pushforward",
            "cpu_sumcheck/ra_delayed_materialization",
        ],
        certification_evidence_files: &[],
        status: KernelPortStatus::Required,
    },
    BackendKernelPortSpec {
        name: "cpu_ra_delayed_materialization",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-RA-007"],
        source_locations: &["jolt-core/src/poly/ra_poly.rs"],
        cpu_entrypoints: &["jolt_backends::cpu::ra::RaPolynomial"],
        microbenchmarks: &["cpu_sumcheck/ra_delayed_materialization"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_ra_delayed_materialization/cpu_sumcheck_ra_delayed_materialization.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_shared_ra_delayed_materialization",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-RA-001", "OPT-RA-008", "OPT-RA-009", "OPT-RA-010"],
        source_locations: &[
            "jolt-core/src/poly/shared_ra_polys.rs::RaIndices",
            "jolt-core/src/poly/shared_ra_polys.rs::SharedRaPolynomials",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::ra::RaCycleIndices",
            "jolt_backends::cpu::ra::SharedRaPolynomials",
        ],
        microbenchmarks: &["cpu_sumcheck/shared_ra_delayed_materialization"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_shared_ra_delayed_materialization/cpu_sumcheck_shared_ra_delayed_materialization.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_ra_pushforward",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-RA-003", "OPT-RA-004", "OPT-RA-005", "OPT-RA-006"],
        source_locations: &["jolt-core/src/poly/shared_ra_polys.rs::compute_all_G_impl"],
        cpu_entrypoints: &["jolt_backends::cpu::ra::pushforward_indices"],
        microbenchmarks: &["cpu_sumcheck/ra_pushforward"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_ra_pushforward/cpu_sumcheck_ra_pushforward.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_sparse_read_write_matrix_kernels",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
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
        ],
        source_locations: &[
            "jolt-core/src/subprotocols/read_write_matrix",
            "jolt-core/src/zkvm/ram",
            "jolt-core/src/zkvm/registers",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::read_write_matrix",
            "jolt_backends::cpu::sumcheck::kernels",
        ],
        microbenchmarks: &[
            "cpu_sumcheck/read_write_bind",
            "cpu_sumcheck/read_write_relation_messages",
        ],
        certification_evidence_files: &[],
        status: KernelPortStatus::Required,
    },
    BackendKernelPortSpec {
        name: "cpu_read_write_one_hot_coeff_lookup",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-RW-007"],
        source_locations: &["jolt-core/src/subprotocols/read_write_matrix/one_hot_coeffs.rs"],
        cpu_entrypoints: &[
            "jolt_backends::cpu::read_write_matrix::OneHotCoeffTable",
            "jolt_backends::cpu::read_write_matrix::OneHotCoeffIndex",
        ],
        microbenchmarks: &["cpu_sumcheck/read_write_one_hot_coeff_lookup"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_read_write_one_hot_coeff_lookup/cpu_sumcheck_read_write_one_hot_coeff_lookup.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_read_write_cycle_major_bind",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
            "OPT-RW-001",
            "OPT-RW-002",
            "OPT-RW-004",
            "OPT-RW-005",
            "OPT-RW-006",
        ],
        source_locations: &[
            "jolt-core/src/subprotocols/read_write_matrix/cycle_major.rs",
            "jolt-core/src/subprotocols/read_write_matrix/one_hot_coeffs.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::read_write_matrix::ReadWriteMatrixCycleMajor",
            "jolt_backends::cpu::read_write_matrix::CycleMajorMatrixEntry",
        ],
        microbenchmarks: &["cpu_sumcheck/read_write_cycle_major_bind"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_read_write_cycle_major_bind/cpu_sumcheck_read_write_cycle_major_bind.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_read_write_cycle_major_message",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-RW-008"],
        source_locations: &["jolt-core/src/subprotocols/read_write_matrix/cycle_major.rs"],
        cpu_entrypoints: &[
            "jolt_backends::cpu::read_write_matrix::ReadWriteMatrixCycleMajor::prover_message_contribution",
            "jolt_backends::cpu::read_write_matrix::CycleMajorMessageEntry",
        ],
        microbenchmarks: &["cpu_sumcheck/read_write_cycle_major_message"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_read_write_cycle_major_message/cpu_sumcheck_read_write_cycle_major_message.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_read_write_cycle_to_address_major",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-RW-003", "OPT-RW-009"],
        source_locations: &["jolt-core/src/subprotocols/read_write_matrix/address_major.rs"],
        cpu_entrypoints: &[
            "jolt_backends::cpu::read_write_matrix::ReadWriteMatrixAddressMajor",
            "jolt_backends::cpu::read_write_matrix::CycleMajorToAddressMajor",
        ],
        microbenchmarks: &["cpu_sumcheck/read_write_cycle_to_address_major"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_read_write_cycle_to_address_major/cpu_sumcheck_read_write_cycle_to_address_major.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_batched_sumcheck_protocol_kernels",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
            "OPT-SC-001",
            "OPT-SC-003",
            "OPT-SC-004",
            "OPT-SC-005",
            "OPT-SC-006",
            "OPT-SC-009",
            "OPT-SC-010",
            "OPT-SC-011",
            "OPT-SC-012",
        ],
        source_locations: &[
            "jolt-core/src/subprotocols/sumcheck.rs",
            "jolt-core/src/subprotocols/univariate_skip.rs",
            "jolt-core/src/subprotocols/streaming_sumcheck.rs",
            "jolt-core/src/subprotocols/streaming_schedule.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::sumcheck",
            "jolt_backends::cpu::sumcheck::kernels",
        ],
        microbenchmarks: &[
            "cpu_sumcheck/batched_rounds",
            "cpu_sumcheck/streaming_schedule",
        ],
        certification_evidence_files: &[],
        status: KernelPortStatus::Required,
    },
    BackendKernelPortSpec {
        name: "cpu_streaming_schedule",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-SC-009", "OPT-SC-010"],
        source_locations: &["jolt-core/src/subprotocols/streaming_schedule.rs"],
        cpu_entrypoints: &[
            "jolt_backends::cpu::schedule::HalfSplitSchedule",
            "jolt_backends::cpu::schedule::LinearOnlySchedule",
        ],
        microbenchmarks: &["cpu_sumcheck/streaming_schedule"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_streaming_schedule/cpu_sumcheck_streaming_schedule.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_spartan_stage_kernels",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
            "OPT-SP-001",
            "OPT-SP-002",
            "OPT-SP-003",
            "OPT-SP-004",
            "OPT-SP-005",
            "OPT-SP-006",
        ],
        source_locations: &[
            "jolt-core/src/zkvm/spartan",
            "jolt-core/src/zkvm/r1cs",
            "jolt-core/src/poly/split_eq_poly.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::sumcheck::kernels::spartan_outer",
            "jolt_backends::cpu::sumcheck::kernels::spartan_product",
        ],
        microbenchmarks: &[
            "cpu_sumcheck/spartan_outer",
            "cpu_sumcheck/spartan_product",
            "frontier_perf/stage3_spartan_specialized_polys",
        ],
        certification_evidence_files: &[],
        status: KernelPortStatus::Required,
    },
    BackendKernelPortSpec {
        name: "cpu_relation_stage_kernels",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
            "OPT-REL-001",
            "OPT-REL-002",
            "OPT-REL-003",
            "OPT-REL-004",
            "OPT-REL-005",
            "OPT-REL-006",
            "OPT-REL-007",
            "OPT-REL-008",
            "OPT-REL-009",
            "OPT-REL-010",
            "OPT-REL-011",
            "OPT-REL-012",
            "OPT-REL-013",
            "OPT-REL-014",
            "OPT-REL-015",
        ],
        source_locations: &[
            "jolt-core/src/poly/prefix_suffix.rs",
            "jolt-core/src/zkvm/instruction_lookups",
            "jolt-core/src/zkvm/bytecode",
            "jolt-core/src/zkvm/ram",
            "jolt-core/src/zkvm/registers",
            "jolt-core/src/zkvm/claim_reductions",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::relations",
            "jolt_backends::cpu::sumcheck::kernels",
        ],
        microbenchmarks: &[
            "cpu_sumcheck/prefix_suffix_lookup",
            "frontier_perf/relation_stage_claims",
        ],
        certification_evidence_files: &[],
        status: KernelPortStatus::Required,
    },
    BackendKernelPortSpec {
        name: "cpu_opening_stage8_kernels",
        family: BackendKernelFamily::Openings,
        optimization_ids: &[
            "OPT-OPEN-001",
            "OPT-OPEN-002",
            "OPT-OPEN-003",
            "OPT-OPEN-005",
            "OPT-OPEN-006",
            "OPT-OPEN-007",
        ],
        source_locations: &[
            "jolt-core/src/poly/opening_proof.rs",
            "jolt-core/src/poly/rlc_polynomial.rs",
            "jolt-core/src/zkvm/prover.rs::prove_stage8",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::openings",
            "jolt_backends::cpu::sumcheck::evaluate_sumcheck_views",
            "jolt_backends::cpu::poly::stage8_streaming_rlc_vector_matrix_product",
        ],
        microbenchmarks: &["frontier_perf/stage8_streaming_rlc", "cpu_poly/rlc_vmp"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_opening_stage8_kernels/frontier_perf_stage8_streaming_rlc.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_blindfold_backend_kernels",
        family: BackendKernelFamily::BlindFold,
        optimization_ids: &["OPT-ZK-002", "OPT-ZK-003", "OPT-ZK-006"],
        source_locations: &["jolt-core/src/zkvm/prover.rs::prove_blindfold"],
        cpu_entrypoints: &[
            "jolt_backends::cpu::blindfold",
            "jolt_backends::cpu::commitments",
        ],
        microbenchmarks: &["frontier_perf/blindfold_witness_rows"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_blindfold_backend_kernels/frontier_perf_blindfold_witness_rows.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_memory_parallelism_kernels",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &[
            "OPT-MEM-001",
            "OPT-MEM-002",
            "OPT-MEM-003",
            "OPT-MEM-004",
            "OPT-MEM-005",
            "OPT-MEM-006",
            "OPT-MEM-007",
            "OPT-MEM-008",
            "OPT-MEM-009",
            "OPT-MEM-010",
        ],
        source_locations: &[
            "jolt-core/src/utils/thread.rs",
            "jolt-core/src/poly",
            "jolt-core/src/zkvm/prover.rs",
            "jolt-core/src/subprotocols/read_write_matrix",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::memory",
            "jolt_backends::cpu::sumcheck::kernels",
        ],
        microbenchmarks: &[
            "cpu_sumcheck/parallel_scaling",
            "cpu_sumcheck/allocation_profile",
        ],
        certification_evidence_files: &[],
        status: KernelPortStatus::Required,
    },
    BackendKernelPortSpec {
        name: "cpu_field_arithmetic_kernels",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-FLD-001", "OPT-FLD-002", "OPT-FLD-003", "OPT-FLD-004"],
        source_locations: &[
            "jolt-core/src/field",
            "jolt-core/src/utils/small_scalar.rs",
            "jolt-core/src/subprotocols/mles_product_sum.rs",
            "jolt-core/src/poly/one_hot_polynomial.rs",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::field",
            "jolt_backends::cpu::sumcheck::kernels",
        ],
        microbenchmarks: &[
            "cpu_sumcheck/field_products",
            "frontier_perf/one_hot_group_additions",
        ],
        certification_evidence_files: &[],
        status: KernelPortStatus::Required,
    },
    BackendKernelPortSpec {
        name: "cpu_linear_product_d4",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-FLD-003"],
        source_locations: &["jolt-core/src/subprotocols/mles_product_sum.rs::eval_prod_4_assign"],
        cpu_entrypoints: &["jolt_backends::cpu::field::eval_linear_product_d4_assign"],
        microbenchmarks: &["cpu_field/linear_product_d4"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_linear_product_d4/cpu_field_linear_product_d4.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_linear_product_small_degrees",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-FLD-003"],
        source_locations: &[
            "jolt-core/src/subprotocols/mles_product_sum.rs::eval_prod_2_assign",
            "jolt-core/src/subprotocols/mles_product_sum.rs::eval_prod_3_assign",
            "jolt-core/src/subprotocols/mles_product_sum.rs::eval_prod_5_assign",
            "jolt-core/src/subprotocols/mles_product_sum.rs::eval_prod_6_assign",
            "jolt-core/src/subprotocols/mles_product_sum.rs::eval_prod_7_assign",
        ],
        cpu_entrypoints: &[
            "jolt_backends::cpu::field::eval_linear_product_d2_assign",
            "jolt_backends::cpu::field::eval_linear_product_d3_assign",
            "jolt_backends::cpu::field::eval_linear_product_d5_assign",
            "jolt_backends::cpu::field::eval_linear_product_d6_assign",
            "jolt_backends::cpu::field::eval_linear_product_d7_assign",
        ],
        microbenchmarks: &["cpu_field/linear_product_small_degrees"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_linear_product_small_degrees/cpu_field_linear_product_small_degrees.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_linear_product_d8",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-FLD-003"],
        source_locations: &["jolt-core/src/subprotocols/mles_product_sum.rs::eval_prod_8_assign"],
        cpu_entrypoints: &["jolt_backends::cpu::field::eval_linear_product_d8_assign"],
        microbenchmarks: &["cpu_field/linear_product_d8"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_linear_product_d8/cpu_field_linear_product_d8.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_linear_product_d16",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-FLD-003"],
        source_locations: &["jolt-core/src/subprotocols/mles_product_sum.rs::eval_prod_16_assign"],
        cpu_entrypoints: &["jolt_backends::cpu::field::eval_linear_product_d16_assign"],
        microbenchmarks: &["cpu_field/linear_product_d16"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_linear_product_d16/cpu_field_linear_product_d16.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
    BackendKernelPortSpec {
        name: "cpu_linear_product_d32",
        family: BackendKernelFamily::Sumcheck,
        optimization_ids: &["OPT-FLD-003"],
        source_locations: &["jolt-core/src/subprotocols/mles_product_sum.rs::eval_prod_32_assign"],
        cpu_entrypoints: &["jolt_backends::cpu::field::eval_linear_product_d32_assign"],
        microbenchmarks: &["cpu_field/linear_product_d32"],
        certification_evidence_files: &[
            "target/frontier-metrics/kernel-evidence/cpu_linear_product_d32/cpu_field_linear_product_d32.json",
        ],
        status: KernelPortStatus::ParityCertified,
    },
];

fn has_duplicate<T: PartialEq>(values: &[T]) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(index, value)| values[..index].contains(value))
}
