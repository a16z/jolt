use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::metrics::RunMetrics;
use crate::optimization::{
    validate_frontier_kernel_accounting, BackendKernelPortLedger, BackendKernelPortSpec,
    KernelPortStatus, KnownOptimizationIds,
};
use crate::{FrontierSpec, HarnessError, HarnessResult};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerfGate {
    pub warn_ratio: f64,
    pub fail_ratio: f64,
    pub min_samples: u32,
    pub confirmation_size: Option<u64>,
    pub require_time: bool,
    pub require_peak_rss: bool,
}

impl PerfGate {
    pub const fn canonical_frontier() -> Self {
        Self {
            warn_ratio: 1.05,
            fail_ratio: 1.15,
            min_samples: 3,
            confirmation_size: Some(1_048_576),
            require_time: true,
            require_peak_rss: true,
        }
    }

    pub fn validate(self) -> Result<(), &'static str> {
        if !self.warn_ratio.is_finite() || !self.fail_ratio.is_finite() {
            return Err("perf gate ratios must be finite");
        }
        if self.warn_ratio < 1.0 {
            return Err("perf gate warn ratio must be at least 1.0");
        }
        if self.fail_ratio <= self.warn_ratio {
            return Err("perf gate fail ratio must be greater than warn ratio");
        }
        if self.min_samples == 0 {
            return Err("perf gate must require at least one sample");
        }
        if matches!(self.confirmation_size, Some(0)) {
            return Err("perf gate confirmation size must be nonzero when present");
        }
        if !self.require_time && !self.require_peak_rss {
            return Err("perf gate must require at least one measured axis");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateStatus {
    Pass,
    Warn,
    Fail,
    NotMeasured,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerfEvaluation {
    pub status: GateStatus,
    pub time_ratio: Option<f64>,
    pub peak_rss_ratio: Option<f64>,
}

impl PerfEvaluation {
    pub fn passes_gate(&self) -> bool {
        matches!(self.status, GateStatus::Pass | GateStatus::Warn)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelMemoryBudget {
    pub input_bytes: u64,
    pub expected_peak_working_bytes: u64,
    pub budget_bytes: u64,
}

impl KernelMemoryBudget {
    pub const fn new(
        input_bytes: u64,
        expected_peak_working_bytes: u64,
        budget_bytes: u64,
    ) -> Self {
        Self {
            input_bytes,
            expected_peak_working_bytes,
            budget_bytes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KernelBenchmarkEvidence {
    pub kernel: String,
    pub benchmark: String,
    pub samples: u32,
    pub optimization_ids: Vec<String>,
    pub core: RunMetrics,
    pub modular: RunMetrics,
    pub memory: KernelMemoryBudget,
}

impl KernelBenchmarkEvidence {
    pub fn canonical_artifact_path(&self, workspace_root: &Path) -> PathBuf {
        kernel_benchmark_evidence_path(workspace_root, &self.kernel, &self.benchmark)
    }

    pub fn read_json(path: &Path) -> HarnessResult<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    pub fn to_json_pretty(&self) -> HarnessResult<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn write_json(&self, path: &Path) -> HarnessResult<()> {
        std::fs::write(path, self.to_json_pretty()?)?;
        Ok(())
    }

    pub fn write_canonical_json(&self, workspace_root: &Path) -> HarnessResult<PathBuf> {
        let path = self.canonical_artifact_path(workspace_root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        self.write_json(&path)?;
        Ok(path)
    }
}

pub fn kernel_benchmark_evidence_path(
    workspace_root: &Path,
    kernel: &str,
    benchmark: &str,
) -> PathBuf {
    workspace_root
        .join("target/frontier-metrics/kernel-evidence")
        .join(sanitize_artifact_component(kernel))
        .join(format!("{}.json", sanitize_artifact_component(benchmark)))
}

pub fn evaluate_perf(gate: PerfGate, core: &RunMetrics, modular: &RunMetrics) -> PerfEvaluation {
    let time_ratio = ratio(modular.time_ms, core.time_ms);
    let peak_rss_ratio = ratio_u64(modular.peak_rss_bytes, core.peak_rss_bytes);
    let mut status = GateStatus::NotMeasured;
    if gate.require_time {
        status = combine_status(status, ratio_status(gate, time_ratio, true));
    } else {
        status = combine_status(status, ratio_status(gate, time_ratio, false));
    }
    if gate.require_peak_rss {
        status = combine_status(status, ratio_status(gate, peak_rss_ratio, true));
    } else {
        status = combine_status(status, ratio_status(gate, peak_rss_ratio, false));
    }

    PerfEvaluation {
        status,
        time_ratio,
        peak_rss_ratio,
    }
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator)) if denominator > 0.0 => Some(numerator / denominator),
        _ => None,
    }
}

fn ratio_u64(numerator: Option<u64>, denominator: Option<u64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator)) if denominator > 0 => {
            Some(numerator as f64 / denominator as f64)
        }
        _ => None,
    }
}

fn ratio_status(gate: PerfGate, ratio: Option<f64>, required: bool) -> GateStatus {
    match ratio {
        Some(ratio) if ratio > gate.fail_ratio => GateStatus::Fail,
        Some(ratio) if ratio > gate.warn_ratio => GateStatus::Warn,
        Some(_) => GateStatus::Pass,
        None if required => GateStatus::Fail,
        None => GateStatus::NotMeasured,
    }
}

fn combine_status(left: GateStatus, right: GateStatus) -> GateStatus {
    match (left, right) {
        (GateStatus::Fail, _) | (_, GateStatus::Fail) => GateStatus::Fail,
        (GateStatus::Warn, _) | (_, GateStatus::Warn) => GateStatus::Warn,
        (GateStatus::Pass, _) | (_, GateStatus::Pass) => GateStatus::Pass,
        (GateStatus::NotMeasured, GateStatus::NotMeasured) => GateStatus::NotMeasured,
    }
}

pub fn validate_kernel_benchmark_evidence(
    gate: PerfGate,
    port: BackendKernelPortSpec,
    evidence: &KernelBenchmarkEvidence,
) -> HarnessResult<PerfEvaluation> {
    gate.validate()
        .map_err(|reason| invalid_evidence(port.name, &evidence.benchmark, reason.to_owned()))?;
    if evidence.kernel != port.name {
        return Err(invalid_evidence(
            port.name,
            &evidence.benchmark,
            format!(
                "evidence was recorded for kernel `{}` instead of `{}`",
                evidence.kernel, port.name
            ),
        ));
    }
    if !port.microbenchmarks.contains(&evidence.benchmark.as_str()) {
        return Err(invalid_evidence(
            port.name,
            &evidence.benchmark,
            "benchmark is not registered for this backend kernel".to_owned(),
        ));
    }
    if evidence.samples < gate.min_samples {
        return Err(invalid_evidence(
            port.name,
            &evidence.benchmark,
            format!(
                "benchmark evidence has {} sample(s), but the gate requires {}",
                evidence.samples, gate.min_samples
            ),
        ));
    }
    validate_evidence_optimization_ids(port, evidence)?;
    validate_memory_budget(port, evidence)?;

    let evaluation = evaluate_perf(gate, &evidence.core, &evidence.modular);
    if !evaluation.passes_gate() {
        return Err(invalid_evidence(
            port.name,
            &evidence.benchmark,
            format!(
                "kernel performance parity failed with status {:?}, time ratio {:?}, peak RSS ratio {:?}",
                evaluation.status, evaluation.time_ratio, evaluation.peak_rss_ratio
            ),
        ));
    }
    Ok(evaluation)
}

pub fn validate_parity_certified_kernel_evidence(
    ledger: &BackendKernelPortLedger,
    evidences: &[KernelBenchmarkEvidence],
    gate: PerfGate,
) -> HarnessResult<()> {
    for port in ledger
        .iter()
        .filter(|port| port.status >= KernelPortStatus::ParityCertified)
    {
        let _evaluation = validate_evidence_for_port(gate, *port, evidences)?;
    }
    Ok(())
}

pub fn validate_parity_certified_kernel_evidence_files(
    workspace_root: &Path,
    ledger: &BackendKernelPortLedger,
    gate: PerfGate,
) -> HarnessResult<()> {
    for port in ledger
        .iter()
        .filter(|port| port.status >= KernelPortStatus::ParityCertified)
    {
        let mut evidences = Vec::with_capacity(port.certification_evidence_files.len());
        for evidence_file in port.certification_evidence_files {
            let path = workspace_root.join(evidence_file);
            evidences.push(KernelBenchmarkEvidence::read_json(&path)?);
        }
        let _evaluation = validate_evidence_for_port(gate, *port, &evidences)?;
    }
    Ok(())
}

pub fn validate_frontier_replacement_ready(
    frontier: FrontierSpec,
    known: &KnownOptimizationIds,
    ledger: &BackendKernelPortLedger,
    evidences: &[KernelBenchmarkEvidence],
) -> HarnessResult<()> {
    validate_frontier_kernel_accounting(
        frontier,
        known,
        ledger,
        KernelPortStatus::ParityCertified,
    )?;
    let Some(gate) = frontier.perf else {
        return Err(HarnessError::InvalidManifest {
            frontier: frontier.name,
            reason: "frontier replacement requires a perf gate".to_owned(),
        });
    };

    let required_ports = frontier
        .backend_kernel_ports
        .iter()
        .filter_map(|port_name| ledger.find(port_name))
        .filter(|port| port.status >= KernelPortStatus::ParityCertified)
        .filter(|port| {
            port.optimization_ids
                .iter()
                .any(|id| frontier.optimization_ids.contains(id) && known.requires_cpu_backend(id))
        })
        .map(|port| port.name)
        .collect::<BTreeSet<_>>();

    for port_name in required_ports {
        let Some(port) = ledger.find(port_name) else {
            return Err(HarnessError::InvalidManifest {
                frontier: frontier.name,
                reason: format!("unknown backend kernel ledger entry `{port_name}`"),
            });
        };
        let _evaluation = validate_evidence_for_port(gate, *port, evidences)?;
    }
    Ok(())
}

fn validate_evidence_for_port(
    gate: PerfGate,
    port: BackendKernelPortSpec,
    evidences: &[KernelBenchmarkEvidence],
) -> HarnessResult<PerfEvaluation> {
    let mut failures = Vec::new();
    for evidence in evidences
        .iter()
        .filter(|evidence| evidence.kernel == port.name)
    {
        match validate_kernel_benchmark_evidence(gate, port, evidence) {
            Ok(evaluation) => return Ok(evaluation),
            Err(error) => failures.push(error.to_string()),
        }
    }

    let reason = if failures.is_empty() {
        "missing benchmark evidence for parity-certified kernel".to_owned()
    } else {
        failures.join("; ")
    };
    Err(invalid_evidence(port.name, "<missing>", reason))
}

fn validate_evidence_optimization_ids(
    port: BackendKernelPortSpec,
    evidence: &KernelBenchmarkEvidence,
) -> HarnessResult<()> {
    let expected = port
        .optimization_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let actual = evidence
        .optimization_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if actual == expected {
        return Ok(());
    }
    Err(invalid_evidence(
        port.name,
        &evidence.benchmark,
        format!(
            "benchmark evidence optimization IDs {:?} do not match registered IDs {:?}",
            actual, expected
        ),
    ))
}

fn validate_memory_budget(
    port: BackendKernelPortSpec,
    evidence: &KernelBenchmarkEvidence,
) -> HarnessResult<()> {
    if evidence.memory.budget_bytes == 0 {
        return Err(invalid_evidence(
            port.name,
            &evidence.benchmark,
            "analytical memory budget must be nonzero".to_owned(),
        ));
    }
    if evidence.memory.expected_peak_working_bytes > evidence.memory.budget_bytes {
        return Err(invalid_evidence(
            port.name,
            &evidence.benchmark,
            "expected peak working memory exceeds analytical budget".to_owned(),
        ));
    }
    if let Some(measured_peak) = evidence.modular.peak_rss_bytes {
        if measured_peak > evidence.memory.budget_bytes {
            return Err(invalid_evidence(
                port.name,
                &evidence.benchmark,
                format!(
                    "measured modular peak RSS {measured_peak} exceeds analytical budget {}",
                    evidence.memory.budget_bytes
                ),
            ));
        }
    }
    Ok(())
}

fn invalid_evidence(kernel: &'static str, benchmark: &str, reason: String) -> HarnessError {
    HarnessError::InvalidBenchmarkEvidence {
        kernel: kernel.to_owned(),
        benchmark: benchmark.to_owned(),
        reason,
    }
}

fn sanitize_artifact_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
