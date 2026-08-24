#![expect(
    clippy::print_stdout,
    clippy::unwrap_used,
    reason = "the fixed benchmark fails loudly on malformed fixtures or proofs"
)]

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use akita_config::CommitmentConfig;
use akita_field::CanonicalField;
use akita_types::{AkitaScheduleLookupKey, OpeningClaimsLayout};
use jolt_akita::configs::JoltOneHotK256;
use jolt_akita::{
    AkitaBatchProof, AkitaCommitment, AkitaField, AkitaNativeBatchPolynomials, AkitaNativeBatching,
    AkitaProverHint, AkitaProverSetup, AkitaScheme, AkitaSetupParams, AkitaVerifierSetup,
    OwnedTraceOneHotRows, TraceCommitmentBackend, TraceOneHotRows, AKITA_ONE_HOT_K256,
};
use jolt_openings::{BatchOpeningScheme, CommitmentScheme, EvaluationClaim, VerifierOpeningClaim};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::{Blake2bTranscript, Transcript};
use serde::{Deserialize, Serialize};
use tracing::span::{Attributes, Id};
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

const LAYOUT_DIGEST: [u8; 32] = [0xE7; 32];
const COLUMN_CAPACITY: usize = 32;
const TRANSCRIPT_LABEL: &[u8] = b"jolt-akita/eval-proof-v1";
const FIXTURE_PATTERN: &str = "1+((row_mod_128+17*column)_mod_128)";

#[derive(Clone, Copy)]
struct CaseSpec {
    id: &'static str,
    log_t: usize,
    columns: usize,
    populated_rows: usize,
}

const CASES: [CaseSpec; 3] = [
    CaseSpec {
        id: "t25-c29",
        log_t: 25,
        columns: 29,
        populated_rows: 1 << 25,
    },
    CaseSpec {
        id: "t25-c30",
        log_t: 25,
        columns: 30,
        populated_rows: 1 << 25,
    },
    CaseSpec {
        id: "t28-c30-partial",
        log_t: 28,
        columns: 30,
        populated_rows: 253_779_321,
    },
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Cpu,
    Metal,
}

impl BackendChoice {
    const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

struct Args {
    case: CaseSpec,
    order: Vec<BackendChoice>,
    anchor: Option<PathBuf>,
    output: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ShapeRecord {
    log_t: usize,
    physical_rows: usize,
    populated_rows: usize,
    columns: usize,
    column_capacity: usize,
    one_hot_k: usize,
    num_vars: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RevisionRecord {
    jolt: String,
    akita: String,
    rustc: String,
    profile: String,
    metal_feature: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PreparationRecord {
    fixture_ns: u64,
    setup_ns: u64,
    backend_prepare_ns: u64,
    #[serde(default)]
    commit_ns: Option<u64>,
    #[serde(default)]
    metal_commit_ns: Option<u64>,
    #[serde(default)]
    metal_commitment_parity: Option<bool>,
    #[serde(default)]
    metal_opening_index_ns: Option<u64>,
    #[serde(default)]
    metal_opening_index_gpu_ns: Option<u64>,
    #[serde(default)]
    metal_opening_index_bytes: Option<usize>,
    commitment_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct MeasurementRecord {
    backend: String,
    selected_route: String,
    complete_opening_ns: u64,
    subphases_ns: BTreeMap<String, u64>,
    subphase_sum_ns: u64,
    other_ns: u64,
    subphases_disjoint: bool,
    gpu_active_ns: Option<u64>,
    command_wall_ns: Option<u64>,
    #[serde(default)]
    opening_index_ns: Option<u64>,
    #[serde(default)]
    opening_index_gpu_ns: Option<u64>,
    #[serde(default)]
    opening_index_bytes: Option<usize>,
    #[serde(default)]
    packed_decompose_wall_ns: Option<u64>,
    #[serde(default)]
    packed_decompose_gpu_ns: Option<u64>,
    #[serde(default)]
    packed_decompose_consumer_ns: Option<u64>,
    #[serde(default)]
    packed_decompose_prepare_ns: Option<u64>,
    #[serde(default)]
    packed_decompose_postprocess_ns: Option<u64>,
    #[serde(default)]
    packed_decompose_total_ns: Option<u64>,
    #[serde(default)]
    packed_decompose_indexed_calls: Option<usize>,
    #[serde(default)]
    packed_decompose_direct_digit_bytes: Option<usize>,
    #[serde(default)]
    recursive_commit_matrix_cache_hits: Option<usize>,
    #[serde(default)]
    recursive_commit_matrix_cache_misses: Option<usize>,
    #[serde(default)]
    recursive_commit_matrix_ntt_ns: Option<u64>,
    #[serde(default)]
    recursive_commit_matrix_ntt_gpu_ns: Option<u64>,
    #[serde(default)]
    recursive_commit_matrix_ntt_bytes: Option<usize>,
    #[serde(default)]
    linear_source_command_wall_ns: Option<u64>,
    #[serde(default)]
    linear_source_gpu_ns: Option<u64>,
    #[serde(default)]
    direct_range_command_wall_ns: Option<u64>,
    #[serde(default)]
    direct_range_gpu_ns: Option<u64>,
    #[serde(default)]
    direct_range_buffer_setup_ns: Option<u64>,
    #[serde(default)]
    direct_relation_command_wall_ns: Option<u64>,
    #[serde(default)]
    direct_relation_gpu_ns: Option<u64>,
    #[serde(default)]
    direct_relation_buffer_setup_ns: Option<u64>,
    upload_ns: Option<u64>,
    readback_ns: Option<u64>,
    allocation_bytes: Option<usize>,
    cpu_fallback_calls: Option<usize>,
    planned_cpu_calls: Option<usize>,
    planned_cpu_work_units: Option<usize>,
    cpu_tail_work_units: Option<usize>,
    metrics_complete: bool,
    fallback: bool,
    rss_before_bytes: Option<u64>,
    rss_after_bytes: Option<u64>,
    peak_rss_bytes: Option<u64>,
    claimed_evaluation: String,
    proof_digest: String,
    proof_size_bytes: usize,
    verifier_ns: u64,
    verifier_ok: bool,
    transcript_match: bool,
    cpu_parity: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct HarnessRecord {
    schema_version: u32,
    evaluator: String,
    case_id: String,
    run_order: Vec<String>,
    revisions: RevisionRecord,
    fixture_digest: String,
    schedule_digest: String,
    shape: ShapeRecord,
    preparation: PreparationRecord,
    measurements: Vec<MeasurementRecord>,
    proof_byte_parity: Option<bool>,
    verifier_all_ok: bool,
    peak_rss_within_90_gib: bool,
}

#[derive(Clone, Default)]
struct PhaseCollector {
    elapsed: Arc<Mutex<BTreeMap<String, Duration>>>,
}

#[derive(Default)]
struct PhaseTiming {
    active_entries: usize,
    active_start: Option<Instant>,
    elapsed: Duration,
}

impl PhaseCollector {
    fn clear(&self) {
        self.elapsed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    fn snapshot(&self) -> BTreeMap<String, u64> {
        self.elapsed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|(name, duration)| (name.clone(), duration_ns(*duration)))
            .collect()
    }
}

impl<S> Layer<S> for PhaseCollector
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attributes: &Attributes<'_>, id: &Id, context: Context<'_, S>) {
        let Some(span) = context.span(id) else {
            return;
        };
        if phase_bucket(span.name()).is_some() {
            span.extensions_mut().insert(PhaseTiming::default());
        }
    }

    fn on_enter(&self, id: &Id, context: Context<'_, S>) {
        let Some(span) = context.span(id) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        let Some(timing) = extensions.get_mut::<PhaseTiming>() else {
            return;
        };
        if timing.active_entries == 0 {
            timing.active_start = Some(Instant::now());
        }
        timing.active_entries += 1;
    }

    fn on_exit(&self, id: &Id, context: Context<'_, S>) {
        let Some(span) = context.span(id) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        let Some(timing) = extensions.get_mut::<PhaseTiming>() else {
            return;
        };
        if timing.active_entries == 0 {
            return;
        }
        timing.active_entries -= 1;
        if timing.active_entries == 0 {
            if let Some(started) = timing.active_start.take() {
                timing.elapsed += started.elapsed();
            }
        }
    }

    fn on_close(&self, id: Id, context: Context<'_, S>) {
        let Some(span) = context.span(&id) else {
            return;
        };
        let Some(bucket) = phase_bucket(span.name()) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        let Some(mut timing) = extensions.remove::<PhaseTiming>() else {
            return;
        };
        if let Some(started) = timing.active_start.take() {
            timing.elapsed += started.elapsed();
        }
        *self
            .elapsed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(bucket.to_string())
            .or_default() += timing.elapsed;
    }
}

fn phase_bucket(name: &str) -> Option<&'static str> {
    match name {
        "prepare_ntt_cache" => Some("ntt_prepare"),
        "TracePackedOneHot::evaluate_and_fold" => Some("root_evaluate_fold"),
        "TracePackedOneHot::decompose_fold_cpu" => Some("root_decompose_fold_cpu"),
        "TracePackedOneHot::decompose_fold_metal_single" => {
            Some("root_decompose_fold_metal_single")
        }
        "TracePackedOneHot::decompose_fold_metal_batch" => Some("root_decompose_fold_metal_batch"),
        "TracePackedOneHot::decompose_fold_metal_streaming" => {
            Some("root_decompose_fold_metal_streaming")
        }
        "coefficient_packing_trace_onehot_partials" => Some("root_coefficient_packing"),
        "coefficient_packing_partials" | "MetalCommitBackend::suffix_coefficient_packing" => {
            Some("recursive_coefficient_packing")
        }
        "RingRelationProver::new" => Some("ring_relation_prepare_total"),
        "ring_relation_prepare_inputs" => Some("diag_relation_prepare_inputs"),
        "ring_relation_opening_rows" => Some("diag_relation_opening_rows"),
        "coefficient_packing_d_input" => Some("diag_relation_d_input"),
        "coefficient_packing_d_concat" => Some("diag_relation_d_concat"),
        "compute_relation_v" => Some("diag_relation_compute_v"),
        "relation_compression_witness" => Some("diag_relation_compression"),
        "ring_relation_fold_grind" => Some("diag_relation_fold_grind"),
        "fold_grind_sample" => Some("diag_fold_grind_sample"),
        "fold_grind_acceptance_check" => Some("diag_fold_grind_acceptance"),
        "fold_grind_live_replay" => Some("diag_fold_grind_live_replay"),
        "root_commit_prefix_merge" => Some("diag_root_commit_prefix_merge"),
        "root_z_digit_decompose" => Some("diag_root_z_digit_decompose"),
        "root_z_digit_append" => Some("diag_root_z_digit_append"),
        "root_z_commit_prefix" => Some("diag_root_z_commit_prefix"),
        "root_static_et_commit_prefix" => Some("root_static_et_commit_prefix"),
        "recursive_commit_output_decode" => Some("diag_recursive_commit_output_decode"),
        "ring_relation_build_instance" => Some("diag_relation_build_instance"),
        "ring_relation_build_witness" => Some("diag_relation_build_witness"),
        "ring_switch_build_w" => Some("ring_switch_build_w"),
        "ring_switch_allocate_output" => Some("diag_ring_allocate"),
        "ring_switch_emit_group_segments" => Some("diag_ring_emit_groups"),
        "ring_switch_emit_native_a_segments" => Some("diag_ring_emit_native_a"),
        "ring_switch_decompose_z_planes" => Some("diag_ring_decompose_z"),
        "ring_switch_emit_z_planes" => Some("diag_ring_emit_z"),
        "ring_switch_emit_t_segments" => Some("diag_ring_emit_t"),
        "ring_switch_emit_e_segments" => Some("diag_ring_emit_e"),
        "compute_multi_group_relation_quotient" => Some("diag_ring_relation_quotient"),
        "MetalCommitBackend::recursive_witness_commit_inner" => Some("diag_recursive_commit_inner"),
        "ring_switch_emit_r_rows" => Some("diag_ring_emit_r"),
        "commit_w_level" => Some("next_witness_commit"),
        "ring_switch_finalize" => Some("ring_switch_finalize"),
        "stage1_sumcheck" => Some("stage1_sumcheck"),
        "stage2_opening_preparation" => Some("stage2_preparation"),
        "stage2_additional_terms_prepare" => Some("diag_stage2_additional_terms"),
        "stage2_sumcheck" => Some("stage2_sumcheck"),
        "direct_relation_state_prepare" => Some("diag_stage2_state_prepare"),
        "direct_relation_host_prepare" => Some("diag_stage2_host_prepare"),
        "direct_relation_two_round_prefix_host" => Some("diag_stage2_prefix_host"),
        "direct_relation_session_setup" => Some("diag_stage2_session_setup"),
        "direct_relation_host_round" => Some("diag_stage2_host_round"),
        "direct_relation_host_bind" => Some("diag_stage2_host_bind"),
        "fold_evaluate_claims" => Some("recursive_fold_evaluation"),
        _ => None,
    }
}

struct CommittedShape {
    num_vars: usize,
}

impl MultilinearPoly<AkitaField> for CommittedShape {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, _point: &[AkitaField]) -> AkitaField {
        unreachable!("the retained Akita hint owns the opening source")
    }

    fn for_each_row(&self, _sigma: usize, _f: &mut dyn FnMut(usize, &[AkitaField])) {
        unreachable!("the retained Akita hint owns the opening source")
    }

    fn is_one_hot(&self) -> bool {
        true
    }
}

struct Fixture {
    prover_setup: AkitaProverSetup,
    verifier_setup: AkitaVerifierSetup,
    commitment: Option<AkitaCommitment>,
    cpu_hint: Option<AkitaProverHint>,
    metal_hint: Option<AkitaProverHint>,
    rows: Arc<dyn TraceOneHotRows>,
    point: Vec<AkitaField>,
    evaluation: AkitaField,
    digest: String,
    schedule_digest: String,
    shape: ShapeRecord,
    preparation: PreparationRecord,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut raw = std::env::args().skip(1);
    let mut case = None;
    let mut backend = None;
    let mut anchor = None;
    let mut output = None;
    while let Some(flag) = raw.next() {
        if flag == "--bench" {
            continue;
        }
        let value = raw
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--case" => {
                case = CASES
                    .iter()
                    .copied()
                    .find(|candidate| candidate.id == value);
            }
            "--backend" => backend = Some(value),
            "--anchor" => anchor = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    let case = case.ok_or("--case must be t25-c29, t25-c30, or t28-c30-partial")?;
    let order = match backend.as_deref() {
        Some("cpu") => vec![BackendChoice::Cpu],
        Some("metal") => vec![BackendChoice::Metal],
        Some("pair") => vec![BackendChoice::Cpu, BackendChoice::Metal],
        Some("pair-reversed") => vec![BackendChoice::Metal, BackendChoice::Cpu],
        _ => return Err("--backend must be cpu, metal, pair, or pair-reversed".into()),
    };
    Ok(Args {
        case,
        order,
        anchor,
        output,
    })
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn deterministic_point(num_vars: usize) -> Vec<AkitaField> {
    let mut state = 0xA17A_5EED_D15C_A11Eu64;
    (0..num_vars)
        .map(|_| {
            let low = u128::from(splitmix64(&mut state));
            let high = u128::from(splitmix64(&mut state));
            AkitaField::from_canonical_u128_reduced(low | (high << 64))
        })
        .collect()
}

fn eq_index(point: &[AkitaField], index: usize) -> AkitaField {
    point
        .iter()
        .enumerate()
        .fold(AkitaField::one(), |weight, (position, challenge)| {
            let shift = point.len() - 1 - position;
            if index & (1usize << shift) == 0 {
                weight * (AkitaField::one() - *challenge)
            } else {
                weight * *challenge
            }
        })
}

fn prefix_eq_weight(point: &[AkitaField], exclusive: usize) -> AkitaField {
    if exclusive == 0 {
        return AkitaField::zero();
    }
    let domain = 1usize << point.len();
    if exclusive >= domain {
        return AkitaField::one();
    }
    let mut below = AkitaField::zero();
    let mut equal_prefix = AkitaField::one();
    for (position, challenge) in point.iter().enumerate() {
        let shift = point.len() - 1 - position;
        let zero_weight = AkitaField::one() - *challenge;
        if exclusive & (1usize << shift) == 0 {
            equal_prefix *= zero_weight;
        } else {
            below += equal_prefix * zero_weight;
            equal_prefix *= *challenge;
        }
    }
    below
}

fn selected_row(row: usize, column: usize) -> usize {
    1 + ((row + 17 * column) & 127)
}

fn fixture_evaluation(
    point: &[AkitaField],
    log_t: usize,
    columns: usize,
    populated_rows: usize,
) -> AkitaField {
    let column_point = &point[..5];
    let row_point = &point[5..5 + log_t];
    let hot_point = &point[5 + log_t..];
    let (row_high_point, row_low_point) = row_point.split_at(log_t - 7);
    let row_weights = (0..128)
        .map(|low| {
            let high_count = if populated_rows > low {
                (populated_rows - 1 - low) / 128 + 1
            } else {
                0
            };
            eq_index(row_low_point, low) * prefix_eq_weight(row_high_point, high_count)
        })
        .collect::<Vec<_>>();

    let mut evaluation = AkitaField::zero();
    for column in 0..columns {
        let mut column_evaluation = AkitaField::zero();
        for (low, row_weight) in row_weights.iter().enumerate() {
            column_evaluation += *row_weight * eq_index(hot_point, selected_row(low, column));
        }
        evaluation += eq_index(column_point, column) * column_evaluation;
    }
    evaluation
}

fn validate_oracle() {
    let log_t = 7;
    let rows = 1usize << log_t;
    let columns = 3;
    let populated_rows = 123;
    let num_vars = log_t + 8 + 5;
    let point = deterministic_point(num_vars);
    let mut dense = vec![AkitaField::zero(); COLUMN_CAPACITY * rows * AKITA_ONE_HOT_K256];
    for column in 0..columns {
        for row in 0..populated_rows {
            let index = column * rows * AKITA_ONE_HOT_K256
                + row * AKITA_ONE_HOT_K256
                + selected_row(row, column);
            dense[index] = AkitaField::one();
        }
    }
    let dense_evaluation = Polynomial::new(dense).evaluate(&point);
    assert_eq!(
        dense_evaluation,
        fixture_evaluation(&point, log_t, columns, populated_rows)
    );
}

fn field_hex(value: AkitaField) -> String {
    format!("0x{:032x}", value.to_canonical_u128())
}

fn digest_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn fixture_digest(shape: &ShapeRecord, point: &[AkitaField]) -> String {
    let mut bytes = serde_json::to_vec(shape).unwrap();
    bytes.extend_from_slice(FIXTURE_PATTERN.as_bytes());
    bytes.extend_from_slice(&LAYOUT_DIGEST);
    for value in point {
        bytes.extend_from_slice(&value.to_canonical_u128().to_le_bytes());
    }
    digest_bytes(&bytes)
}

fn schedule_digest(num_vars: usize) -> Result<String, Box<dyn Error>> {
    let layout = OpeningClaimsLayout::new(num_vars, 1)?.root_final_group_layout()?;
    let row = JoltOneHotK256::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(layout))?;
    Ok(digest_bytes(format!("{:?}", row.schedule()).as_bytes()))
}

fn revision(path: &Path) -> String {
    let head = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(path)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map_or_else(
            || "unknown".to_string(),
            |output| String::from_utf8_lossy(&output.stdout).trim().to_string(),
        );
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(path)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .is_some_and(|output| !output.stdout.is_empty());
    if dirty {
        format!("{head}-dirty")
    } else {
        head
    }
}

fn rustc_revision() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map_or_else(
            || "unknown".to_string(),
            |output| String::from_utf8_lossy(&output.stdout).trim().to_string(),
        )
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn rss_bytes() -> Option<u64> {
    memory_stats::memory_stats().map(|stats| stats.physical_mem as u64)
}

fn build_fixture(case: CaseSpec) -> Result<Fixture, Box<dyn Error>> {
    let physical_rows = 1usize << case.log_t;
    let num_vars = case.log_t + 8 + 5;
    let shape = ShapeRecord {
        log_t: case.log_t,
        physical_rows,
        populated_rows: case.populated_rows,
        columns: case.columns,
        column_capacity: COLUMN_CAPACITY,
        one_hot_k: AKITA_ONE_HOT_K256,
        num_vars,
    };

    let fixture_start = Instant::now();
    let point = deterministic_point(num_vars);
    let evaluation = fixture_evaluation(&point, case.log_t, case.columns, case.populated_rows);
    let rows: Arc<dyn TraceOneHotRows> = Arc::new(OwnedTraceOneHotRows::from_row_fn(
        AKITA_ONE_HOT_K256,
        COLUMN_CAPACITY,
        case.columns,
        physical_rows,
        |row, selected| {
            for (column, value) in selected.iter_mut().enumerate() {
                *value = if row < case.populated_rows {
                    selected_row(row, column) as u8
                } else {
                    0
                };
            }
        },
    )?);
    let fixture_ns = duration_ns(fixture_start.elapsed());

    let setup_start = Instant::now();
    let (prover_setup, verifier_setup) = AkitaScheme::setup(AkitaSetupParams::one_hot_only(
        num_vars,
        1,
        LAYOUT_DIGEST,
        AKITA_ONE_HOT_K256,
    ))?;
    let setup_ns = duration_ns(setup_start.elapsed());

    let fixture_digest = fixture_digest(&shape, &point);
    let schedule_digest = schedule_digest(num_vars)?;

    Ok(Fixture {
        prover_setup,
        verifier_setup,
        commitment: None,
        cpu_hint: None,
        metal_hint: None,
        rows,
        point,
        evaluation,
        digest: fixture_digest,
        schedule_digest,
        shape,
        preparation: PreparationRecord {
            fixture_ns,
            setup_ns,
            backend_prepare_ns: 0,
            commit_ns: None,
            metal_commit_ns: None,
            metal_commitment_parity: None,
            metal_opening_index_ns: None,
            metal_opening_index_gpu_ns: None,
            metal_opening_index_bytes: None,
            commitment_digest: String::new(),
        },
    })
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_backend(
    setup: &AkitaProverSetup,
) -> Result<(TraceCommitmentBackend, u64), Box<dyn Error>> {
    let start = Instant::now();
    let backend = TraceCommitmentBackend::metal_required()?;
    backend.prepare_opening_backend(setup)?;
    Ok((backend, duration_ns(start.elapsed())))
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_backend(
    _setup: &AkitaProverSetup,
) -> Result<(TraceCommitmentBackend, u64), Box<dyn Error>> {
    Err("the Metal measurement requires --features metal on macOS".into())
}

fn statement(
    fixture: &Fixture,
) -> Result<Vec<VerifierOpeningClaim<AkitaField, AkitaCommitment>>, Box<dyn Error>> {
    Ok(vec![VerifierOpeningClaim {
        commitment: fixture
            .commitment
            .clone()
            .ok_or("fixture has no commitment")?,
        evaluation: EvaluationClaim::new(fixture.point.clone(), fixture.evaluation),
    }])
}

fn measure(
    fixture: &Fixture,
    backend_choice: BackendChoice,
    backend: &TraceCommitmentBackend,
    phases: &PhaseCollector,
) -> Result<(AkitaBatchProof, MeasurementRecord), Box<dyn Error>> {
    let hint = match backend_choice {
        BackendChoice::Cpu => fixture
            .cpu_hint
            .as_ref()
            .ok_or("CPU measurement has no CPU-produced commitment hint")?,
        BackendChoice::Metal => fixture
            .metal_hint
            .as_ref()
            .ok_or("Metal measurement has no Metal-produced commitment hint")?,
    }
    .clone()
    .with_trace_backend(backend.clone())?;
    let statement = statement(fixture)?;
    let shape = CommittedShape {
        num_vars: fixture.shape.num_vars,
    };
    let polynomials: AkitaNativeBatchPolynomials<'_> =
        vec![&shape as &dyn MultilinearPoly<AkitaField>];
    let mut prover_transcript = Blake2bTranscript::new(TRANSCRIPT_LABEL);

    phases.clear();
    let rss_before_bytes = rss_bytes();
    let start = Instant::now();
    let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &fixture.prover_setup,
        statement.clone(),
        polynomials,
        hint,
        &mut prover_transcript,
    )?;
    let complete_opening_ns = duration_ns(start.elapsed());
    let rss_after_bytes = rss_bytes();
    let subphases_ns = phases.snapshot();
    let subphase_sum_ns = subphases_ns.values().copied().sum::<u64>();
    let subphases_disjoint = subphase_sum_ns <= complete_opening_ns;
    let other_ns = complete_opening_ns.saturating_sub(subphase_sum_ns);

    let verify_start = Instant::now();
    let mut verifier_transcript = Blake2bTranscript::new(TRANSCRIPT_LABEL);
    let verifier_ok = <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        &fixture.verifier_setup,
        &statement,
        &proof,
        &mut verifier_transcript,
    )
    .is_ok();
    let verifier_ns = duration_ns(verify_start.elapsed());
    let transcript_match = prover_transcript.state() == verifier_transcript.state();
    let proof_bytes = serde_json::to_vec(&proof)?;
    let metal_metrics = backend.last_metal_opening_metrics()?;
    let is_metal = backend_choice == BackendChoice::Metal;

    Ok((
        proof,
        MeasurementRecord {
            backend: backend_choice.label().to_string(),
            selected_route: backend.mode_name().to_string(),
            complete_opening_ns,
            subphases_ns,
            subphase_sum_ns,
            other_ns,
            subphases_disjoint,
            gpu_active_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.gpu_active_time)),
            command_wall_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.command_wall_time)),
            opening_index_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.opening_index_time)),
            opening_index_gpu_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.opening_index_gpu_time)),
            opening_index_bytes: metal_metrics
                .as_ref()
                .map(|metrics| metrics.opening_index_bytes),
            packed_decompose_wall_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.packed_decompose_wall_time)),
            packed_decompose_gpu_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.packed_decompose_gpu_time)),
            packed_decompose_consumer_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.packed_decompose_consumer_time)),
            packed_decompose_prepare_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.packed_decompose_prepare_time)),
            packed_decompose_postprocess_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.packed_decompose_postprocess_time)),
            packed_decompose_total_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.packed_decompose_total_time)),
            packed_decompose_indexed_calls: metal_metrics
                .as_ref()
                .map(|metrics| metrics.packed_decompose_indexed_calls),
            packed_decompose_direct_digit_bytes: metal_metrics
                .as_ref()
                .map(|metrics| metrics.packed_decompose_direct_digit_bytes),
            recursive_commit_matrix_cache_hits: metal_metrics
                .as_ref()
                .map(|metrics| metrics.recursive_commit_matrix_cache_hits),
            recursive_commit_matrix_cache_misses: metal_metrics
                .as_ref()
                .map(|metrics| metrics.recursive_commit_matrix_cache_misses),
            recursive_commit_matrix_ntt_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.recursive_commit_matrix_ntt_time)),
            recursive_commit_matrix_ntt_gpu_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.recursive_commit_matrix_ntt_gpu_time)),
            recursive_commit_matrix_ntt_bytes: metal_metrics
                .as_ref()
                .map(|metrics| metrics.recursive_commit_matrix_ntt_bytes),
            linear_source_command_wall_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.linear_source_command_wall_time)),
            linear_source_gpu_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.linear_source_gpu_time)),
            direct_range_command_wall_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.direct_range_command_wall_time)),
            direct_range_gpu_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.direct_range_gpu_time)),
            direct_range_buffer_setup_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.direct_range_buffer_setup_time)),
            direct_relation_command_wall_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.direct_relation_command_wall_time)),
            direct_relation_gpu_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.direct_relation_gpu_time)),
            direct_relation_buffer_setup_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.direct_relation_buffer_setup_time)),
            upload_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.upload_time)),
            readback_ns: metal_metrics
                .as_ref()
                .map(|metrics| duration_ns(metrics.readback_time)),
            allocation_bytes: metal_metrics
                .as_ref()
                .map(|metrics| metrics.allocation_bytes),
            cpu_fallback_calls: metal_metrics
                .as_ref()
                .map(|metrics| metrics.cpu_fallback_calls),
            planned_cpu_calls: metal_metrics
                .as_ref()
                .map(|metrics| metrics.planned_cpu_calls),
            planned_cpu_work_units: metal_metrics
                .as_ref()
                .map(|metrics| metrics.planned_cpu_work_units),
            cpu_tail_work_units: metal_metrics
                .as_ref()
                .map(|metrics| metrics.cpu_tail_work_units),
            metrics_complete: !is_metal || metal_metrics.is_some(),
            fallback: is_metal
                && metal_metrics
                    .as_ref()
                    .is_none_or(|metrics| metrics.cpu_fallback_calls != 0),
            rss_before_bytes,
            rss_after_bytes,
            peak_rss_bytes: jolt_profiling::peak_rss_bytes(),
            claimed_evaluation: field_hex(fixture.evaluation),
            proof_digest: digest_bytes(&proof_bytes),
            proof_size_bytes: proof_bytes.len(),
            verifier_ns,
            verifier_ok,
            transcript_match,
            cpu_parity: None,
        },
    ))
}

fn apply_anchor(record: &mut HarnessRecord, anchor_path: &Path) -> Result<(), Box<dyn Error>> {
    let anchor: HarnessRecord = serde_json::from_slice(&std::fs::read(anchor_path)?)?;
    if anchor.fixture_digest != record.fixture_digest
        || anchor.schedule_digest != record.schedule_digest
        || anchor.shape != record.shape
        || anchor.preparation.commitment_digest != record.preparation.commitment_digest
    {
        return Err("CPU anchor fixture, schedule, or commitment does not match this run".into());
    }
    let cpu = anchor
        .measurements
        .iter()
        .find(|measurement| measurement.backend == "cpu")
        .ok_or("CPU anchor contains no CPU measurement")?;
    for measurement in &mut record.measurements {
        if measurement.backend == "metal" {
            measurement.cpu_parity = Some(measurement.proof_digest == cpu.proof_digest);
        }
    }
    Ok(())
}

fn validate_anchor_fixture(fixture: &Fixture, anchor_path: &Path) -> Result<bool, Box<dyn Error>> {
    let anchor: HarnessRecord = serde_json::from_slice(&std::fs::read(anchor_path)?)?;
    if anchor.fixture_digest != fixture.digest
        || anchor.schedule_digest != fixture.schedule_digest
        || anchor.shape != fixture.shape
        || anchor.preparation.commitment_digest != fixture.preparation.commitment_digest
    {
        return Err("CPU anchor fixture, schedule, or commitment does not match this run".into());
    }
    Ok(anchor
        .measurements
        .iter()
        .any(|measurement| measurement.backend == "cpu"))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    validate_oracle();
    let phases = PhaseCollector::default();
    tracing_subscriber::registry().with(phases.clone()).init();

    let mut fixture = build_fixture(args.case)?;
    let wants_cpu = args.order.contains(&BackendChoice::Cpu);
    let wants_metal = args.order.contains(&BackendChoice::Metal);
    let cpu_backend = TraceCommitmentBackend::cpu();
    let (metal_backend, metal_prepare_ns) = if args.order.contains(&BackendChoice::Metal) {
        let (backend, elapsed) = metal_backend(&fixture.prover_setup)?;
        (Some(backend), elapsed)
    } else {
        (None, 0)
    };
    fixture.preparation.backend_prepare_ns = metal_prepare_ns;

    if wants_cpu {
        let start = Instant::now();
        let (commitment, hint) = AkitaScheme::commit_trace_one_hot(
            &cpu_backend,
            &fixture.prover_setup,
            LAYOUT_DIGEST,
            COLUMN_CAPACITY,
            fixture.rows.clone(),
        )?;
        fixture.preparation.commit_ns = Some(duration_ns(start.elapsed()));
        fixture.commitment = Some(commitment);
        fixture.cpu_hint = Some(hint);
    }
    if let Some(backend) = metal_backend.as_ref() {
        let start = Instant::now();
        let (metal_commitment, metal_hint) = AkitaScheme::commit_trace_one_hot(
            backend,
            &fixture.prover_setup,
            LAYOUT_DIGEST,
            COLUMN_CAPACITY,
            fixture.rows.clone(),
        )?;
        fixture.preparation.metal_commit_ns = Some(duration_ns(start.elapsed()));
        if let Some(metrics) = backend.last_metal_commit_metrics()? {
            fixture.preparation.metal_opening_index_ns =
                Some(duration_ns(metrics.opening_index_time));
            fixture.preparation.metal_opening_index_gpu_ns =
                Some(duration_ns(metrics.opening_index_gpu_time));
            fixture.preparation.metal_opening_index_bytes = Some(metrics.opening_index_bytes);
        }
        if let Some(cpu_commitment) = fixture.commitment.as_ref() {
            let parity = metal_commitment == *cpu_commitment;
            fixture.preparation.metal_commitment_parity = Some(parity);
            if !parity {
                return Err("CPU and Metal commitments differ".into());
            }
        } else {
            fixture.commitment = Some(metal_commitment);
        }
        fixture.metal_hint = Some(metal_hint);
    }
    let commitment = fixture
        .commitment
        .as_ref()
        .ok_or("selected backends produced no commitment")?;
    fixture.preparation.commitment_digest = digest_bytes(&serde_json::to_vec(commitment)?);

    if let Some(anchor_path) = args.anchor.as_deref() {
        if !validate_anchor_fixture(&fixture, anchor_path)? {
            return Err("CPU anchor contains no CPU measurement".into());
        }
        if wants_metal && !wants_cpu {
            fixture.preparation.metal_commitment_parity = Some(true);
        }
    }

    let mut measurements = Vec::with_capacity(args.order.len());
    let mut proofs = Vec::with_capacity(args.order.len());
    for choice in &args.order {
        let backend = match choice {
            BackendChoice::Cpu => &cpu_backend,
            BackendChoice::Metal => metal_backend.as_ref().unwrap(),
        };
        let (proof, measurement) = measure(&fixture, *choice, backend, &phases)?;
        proofs.push(proof);
        measurements.push(measurement);
    }

    let proof_byte_parity = if proofs.len() == 2 {
        let parity = proofs[0] == proofs[1];
        for measurement in &mut measurements {
            if measurement.backend == "metal" {
                measurement.cpu_parity = Some(parity);
            }
        }
        Some(parity)
    } else {
        None
    };
    let akita_dir = std::env::var_os("JOLT_AKITA_EVAL_AKITA_DIR").map_or_else(
        || PathBuf::from("/Users/mgeorghiades/worktrees/akita-metal-eval-proof"),
        PathBuf::from,
    );
    let mut record = HarnessRecord {
        schema_version: 3,
        evaluator: "akita_eval_proof".to_string(),
        case_id: args.case.id.to_string(),
        run_order: args
            .order
            .iter()
            .map(|choice| choice.label().to_string())
            .collect(),
        revisions: RevisionRecord {
            jolt: revision(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .as_path(),
            ),
            akita: revision(&akita_dir),
            rustc: rustc_revision(),
            profile: "bench".to_string(),
            metal_feature: cfg!(all(feature = "metal", target_os = "macos")),
        },
        fixture_digest: fixture.digest,
        schedule_digest: fixture.schedule_digest,
        shape: fixture.shape,
        preparation: fixture.preparation,
        verifier_all_ok: measurements
            .iter()
            .all(|measurement| measurement.verifier_ok && measurement.transcript_match),
        peak_rss_within_90_gib: measurements.iter().all(|measurement| {
            measurement
                .peak_rss_bytes
                .is_some_and(|bytes| bytes <= 90 * 1024 * 1024 * 1024)
        }),
        measurements,
        proof_byte_parity,
    };
    if let Some(anchor) = args.anchor.as_deref() {
        apply_anchor(&mut record, anchor)?;
    }
    let json = serde_json::to_string(&record)?;
    if let Some(output) = args.output {
        std::fs::write(output, format!("{json}\n"))?;
    }
    println!("{json}");
    Ok(())
}
