use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(all(
    feature = "core-fixtures",
    any(
        not(feature = "zk"),
        all(feature = "zk", not(feature = "field-inline"))
    )
))]
use std::{
    hint::black_box,
    mem::size_of,
    path::{Path, PathBuf},
    time::Instant,
};

#[cfg(all(
    feature = "core-fixtures",
    any(
        not(feature = "zk"),
        all(feature = "zk", not(feature = "field-inline"))
    )
))]
use jolt_prover_harness::{KernelBenchmarkEvidence, KernelMemoryBudget};

use jolt_prover_harness::{evaluate_perf, PerfGate, RunMetrics};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

static CURRENT_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

// SAFETY: every method delegates to `System` with the original layout and only
// updates atomic counters after successful allocation-size changes.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: this forwards the exact allocation request to the system allocator.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: this forwards the exact allocation request to the system allocator.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record_dealloc(layout.size());
        // SAFETY: `ptr` and `layout` come from the caller's matching allocation.
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: this forwards the exact reallocation request to the system allocator.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            let old_size = layout.size();
            if new_size >= old_size {
                record_alloc(new_size - old_size);
            } else {
                record_dealloc(old_size - new_size);
            }
        }
        new_ptr
    }
}

fn record_alloc(size: usize) {
    let current = CURRENT_ALLOCATED.fetch_add(size, Ordering::SeqCst) + size;
    loop {
        let peak = PEAK_ALLOCATED.load(Ordering::SeqCst);
        if current <= peak {
            break;
        }
        if PEAK_ALLOCATED
            .compare_exchange(peak, current, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            break;
        }
    }
}

fn record_dealloc(size: usize) {
    let _ = CURRENT_ALLOCATED.fetch_sub(size, Ordering::SeqCst);
}

const FRONTIER_PERF_BENCHMARKS: &[&str] = &[
    "frontier_perf/stage0_commitments",
    "frontier_perf/stage0_zk_commitments",
    "frontier_perf/stage0_advice_commitments",
    "frontier_perf/stage0_field_inline_commitments",
    "frontier_perf/one_hot_commitments",
    "cpu_poly/compact_bind",
    "cpu_poly/split_eq_evaluate",
    "cpu_poly/inside_out_evaluate",
    "cpu_poly/dense_batch_evaluate",
    "cpu_poly/dense_dot_product_low_optimized",
    "cpu_poly/linear_combination",
    "cpu_poly/one_hot_evaluate",
    "cpu_poly/one_hot_vmp",
    "cpu_poly/rlc_vmp",
    "cpu_field/linear_product_small_degrees",
    "cpu_field/linear_product_d4",
    "cpu_field/linear_product_d8",
    "cpu_field/linear_product_d16",
    "cpu_field/linear_product_d32",
    "cpu_sumcheck/streaming_schedule",
    "cpu_sumcheck/ra_delayed_materialization",
    "cpu_sumcheck/shared_ra_delayed_materialization",
    "cpu_sumcheck/ra_pushforward",
    "cpu_sumcheck/read_write_one_hot_coeff_lookup",
    "cpu_sumcheck/read_write_cycle_major_bind",
    "cpu_sumcheck/read_write_cycle_major_message",
    "cpu_sumcheck/read_write_cycle_to_address_major",
    "frontier_perf/stage2_regular_batch_inputs",
    "frontier_perf/stage2_regular_batch_sumcheck",
    "frontier_perf/stage3_regular_batch_inputs",
    "frontier_perf/stage3_regular_batch_sumcheck",
    "frontier_perf/stage4_regular_batch_inputs",
    "frontier_perf/stage4_regular_batch_sumcheck",
    "frontier_perf/stage4_field_inline_registers_read_write",
    "frontier_perf/stage5_regular_batch_inputs",
    "frontier_perf/stage5_regular_batch_sumcheck",
    "frontier_perf/stage5_field_inline_registers_val_evaluation",
    "frontier_perf/stage6_regular_batch_inputs",
    "frontier_perf/stage6_regular_batch_sumcheck",
    "frontier_perf/stage6_field_inline_registers_inc_claim_reduction",
    "frontier_perf/stage7_regular_batch_sumcheck",
    "frontier_perf/stage8_streaming_rlc",
    "frontier_perf/zk_blindfold_core_fixture",
    "frontier_perf/blindfold_witness_rows",
];

fn main() {
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_streaming_commitments") {
        write_stage0_streaming_commitment_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_zk_streaming_commitments")
    {
        write_stage0_zk_streaming_commitment_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_blindfold_round_commitments")
    {
        write_blindfold_round_commitment_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_blindfold_backend_kernels")
    {
        write_blindfold_backend_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_advice_commitment_contexts")
    {
        write_stage0_advice_commitment_context_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_field_inline_commitments")
    {
        write_stage0_field_inline_commitment_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_one_hot_commitments") {
        write_one_hot_commitment_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_spartan_outer_prefix_product_sum")
    {
        write_stage1_spartan_outer_prefix_product_sum_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_spartan_product_uniskip") {
        write_stage2_product_uniskip_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_stage2_regular_batch_input_claims")
    {
        write_stage2_regular_batch_input_claim_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_stage2_regular_batch_sumcheck")
    {
        write_stage2_regular_batch_sumcheck_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_stage3_regular_batch_input_claims")
    {
        write_stage3_regular_batch_input_claim_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_stage3_regular_batch_sumcheck")
    {
        write_stage3_regular_batch_sumcheck_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_stage4_regular_batch_input_claims")
    {
        write_stage4_regular_batch_input_claim_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_stage4_regular_batch_sumcheck")
    {
        write_stage4_regular_batch_sumcheck_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_field_inline_stage4_registers_read_write")
    {
        write_stage4_field_inline_registers_read_write_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_stage5_regular_batch_input_claims")
    {
        write_stage5_regular_batch_input_claim_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_stage5_regular_batch_sumcheck")
    {
        write_stage5_regular_batch_sumcheck_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_field_inline_stage5_registers_val_evaluation")
    {
        write_stage5_field_inline_registers_val_evaluation_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_stage6_regular_batch_input_claims")
    {
        write_stage6_regular_batch_input_claim_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_stage6_regular_batch_sumcheck")
    {
        write_stage6_regular_batch_sumcheck_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_field_inline_stage6_registers_inc_claim_reduction")
    {
        write_stage6_field_inline_registers_inc_claim_reduction_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_stage7_regular_batch_input_claims")
    {
        write_stage7_regular_batch_input_claim_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_stage7_regular_batch_sumcheck")
    {
        write_stage7_regular_batch_sumcheck_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_materialized_opening_evaluations")
    {
        write_materialized_opening_rlc_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_opening_stage8_kernels") {
        write_stage8_streaming_rlc_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_eq_table_generation") {
        write_eq_table_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_eq_aligned_block_generation")
    {
        write_eq_aligned_block_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_split_eq_streaming_windows")
    {
        write_split_eq_streaming_window_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_unipoly_interpolation") {
        write_unipoly_interpolation_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_compressed_unipoly") {
        write_compressed_unipoly_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_lagrange_many") {
        write_lagrange_many_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_compact_polynomial_bind") {
        write_compact_polynomial_bind_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_split_eq_polynomial_evaluation")
    {
        write_split_eq_polynomial_evaluation_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_inside_out_polynomial_evaluation")
    {
        write_inside_out_polynomial_evaluation_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_dense_batch_polynomial_evaluation")
    {
        write_dense_batch_polynomial_evaluation_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_dense_dot_product_low_optimized")
    {
        write_dense_dot_product_low_optimized_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_mixed_polynomial_linear_combination")
    {
        write_mixed_polynomial_linear_combination_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_one_hot_polynomial_evaluation")
    {
        write_one_hot_polynomial_evaluation_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_one_hot_vector_matrix_product")
    {
        write_one_hot_vector_matrix_product_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_rlc_polynomial_vector_matrix_product")
    {
        write_rlc_polynomial_vector_matrix_product_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_linear_product_small_degrees")
    {
        write_linear_product_small_degrees_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_linear_product_d4") {
        write_linear_product_d4_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_linear_product_d8") {
        write_linear_product_d8_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_linear_product_d16") {
        write_linear_product_d16_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_linear_product_d32") {
        write_linear_product_d32_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_streaming_schedule") {
        write_streaming_schedule_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_ra_delayed_materialization")
    {
        write_ra_delayed_materialization_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_shared_ra_delayed_materialization")
    {
        write_shared_ra_delayed_materialization_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref() == Ok("cpu_ra_pushforward") {
        write_ra_pushforward_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_read_write_one_hot_coeff_lookup")
    {
        write_read_write_one_hot_coeff_lookup_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_read_write_cycle_major_bind")
    {
        write_read_write_cycle_major_bind_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_read_write_cycle_major_message")
    {
        write_read_write_cycle_major_message_kernel_evidence();
        return;
    }
    if std::env::var("JOLT_WRITE_KERNEL_EVIDENCE").as_deref()
        == Ok("cpu_read_write_cycle_to_address_major")
    {
        write_read_write_cycle_to_address_major_kernel_evidence();
        return;
    }

    let gate = PerfGate::canonical_frontier();
    let core = RunMetrics::new(Some(100.0), Some(1_000), None);
    let modular = RunMetrics::new(Some(104.0), Some(1_020), None);
    let _evaluation = evaluate_perf(gate, &core, &modular);
    let _benchmarks = std::hint::black_box(FRONTIER_PERF_BENCHMARKS);
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage0_streaming_commitment_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage0_commitment_kernel_benchmark_fixture, validate_kernel_benchmark_evidence,
        FeatureMode, FixtureKind, FixtureRequest, KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_streaming_commitments";
    const BENCHMARK: &str = "frontier_perf/stage0_commitments";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-COM-001", "OPT-COM-006"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage0_commitment_kernel_benchmark_fixture(&request)
        .expect("load stage0 commitment kernel fixture");
    let shape = fixture.shape().expect("derive stage0 commitment shape");

    let core = measure_samples(samples, || {
        let count = fixture
            .run_core_streaming_commitments()
            .expect("run core stage0 streaming commitments");
        let _ = black_box(count);
    });

    let modular = measure_samples(samples, || {
        let count = fixture
            .run_modular_streaming_commitments()
            .expect("run modular stage0 streaming commitments");
        let _ = black_box(count);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage0_streaming_commitment_memory(shape),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered stage0 streaming commitment kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("stage0 streaming commitment kernel evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical stage0 streaming commitment evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
#[derive(Clone, Copy)]
struct Stage0ZkCommitmentKernelShape {
    rows: usize,
    row_width: usize,
    pcs_rows: usize,
    chunks: usize,
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
impl Stage0ZkCommitmentKernelShape {
    const fn new(rows: usize, chunk_size: usize) -> Self {
        let log_rows = rows.trailing_zeros() as usize;
        let row_width = 1_usize << log_rows.div_ceil(2);
        Self {
            rows,
            row_width,
            pcs_rows: rows / row_width,
            chunks: rows.div_ceil(chunk_size),
        }
    }
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage0_zk_streaming_commitment_kernel_evidence() {
    use jolt_backends::{
        cpu::{CpuBackend, CpuBackendConfig},
        CommitmentBackend, CommitmentMode, CommitmentRequest, CommitmentRequestItem,
        CommitmentSlot,
    };
    use jolt_core::poly::commitment::{
        commitment_scheme::{
            CommitmentScheme as CoreCommitmentScheme,
            StreamingCommitmentScheme as CoreStreamingCommitmentScheme,
        },
        dory::{DoryCommitmentScheme, DoryContext, DoryGlobals, DoryLayout},
    };
    use jolt_dory::{DoryProverSetup, DoryScheme};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};
    use jolt_witness::{
        CommittedWitnessProvider, MaterializationPolicy, NamespaceId, OracleDescriptor, OracleKind,
        OracleRef, OracleViewRequest, PolynomialChunk, PolynomialEncoding, PolynomialStream,
        PolynomialView, RetentionHint, ViewRequirement, WitnessDimensions, WitnessError,
        WitnessNamespace, WitnessProvider,
    };

    const KERNEL: &str = "cpu_zk_streaming_commitments";
    const BENCHMARK: &str = "frontier_perf/stage0_zk_commitments";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-COM-001", "OPT-COM-006"];
    const ROWS: usize = 1 << 16;
    const CHUNK_SIZE: usize = 1024;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum ZkCommitmentNamespace {}

    impl WitnessNamespace for ZkCommitmentNamespace {
        type ChallengeId = u8;
        type CommittedId = u8;
        type OpeningId = u8;
        type PublicId = u8;
        type VirtualId = u8;

        const ID: NamespaceId = NamespaceId::new("stage0_zk_commitment_bench");
    }

    struct ZkCommitmentFixture {
        values: Vec<u64>,
        core_setup: <DoryCommitmentScheme as CoreCommitmentScheme>::ProverSetup,
        modular_setup: DoryProverSetup,
    }

    impl ZkCommitmentFixture {
        fn new(rows: usize) -> Self {
            let values = (0..rows)
                .map(|index| (index as u64).wrapping_mul(29).wrapping_add(17))
                .collect::<Vec<_>>();
            let num_vars = rows.trailing_zeros() as usize;
            let core_setup = DoryCommitmentScheme::setup_prover(num_vars);
            let modular_setup = DoryScheme::setup_prover(num_vars);
            Self {
                values,
                core_setup,
                modular_setup,
            }
        }

        fn shape(&self) -> Stage0ZkCommitmentKernelShape {
            Stage0ZkCommitmentKernelShape::new(self.values.len(), CHUNK_SIZE)
        }

        fn run_core_zk_streaming_commitment(&self) -> usize {
            let shape = self.shape();
            DoryGlobals::initialize_context(
                1,
                self.values.len(),
                DoryContext::Main,
                Some(DoryLayout::CycleMajor),
            )
            .expect("initialize core Dory ZK context");
            let tier1 = self
                .values
                .chunks(shape.row_width)
                .map(|chunk| {
                    <DoryCommitmentScheme as CoreStreamingCommitmentScheme>::process_chunk::<u64>(
                        &self.core_setup,
                        chunk,
                    )
                })
                .collect::<Vec<_>>();
            let (_commitment, _hint) =
                <DoryCommitmentScheme as CoreStreamingCommitmentScheme>::aggregate_chunks(
                    &self.core_setup,
                    None,
                    &tier1,
                );
            1
        }

        fn run_modular_zk_streaming_commitment(&self) -> Result<usize, String> {
            let witness = ZkCommitmentWitness {
                values: &self.values,
            };
            let request = CommitmentRequest::new(vec![CommitmentRequestItem::with_mode(
                CommitmentSlot(0),
                ViewRequirement::new(
                    OracleRef::committed(0),
                    PolynomialEncoding::Compact,
                    MaterializationPolicy::Streaming,
                    RetentionHint::ThroughBlindFold,
                ),
                CommitmentMode::Zk,
            )]);
            let mut backend = CpuBackend::new(CpuBackendConfig {
                preserve_core_fast_path: true,
                commitment_chunk_size: CHUNK_SIZE,
            });
            let result =
                <CpuBackend as CommitmentBackend<Fr, ZkCommitmentNamespace, DoryScheme>>::commit(
                    &mut backend,
                    &request,
                    &witness,
                    &self.modular_setup,
                )
                .map_err(|error| error.to_string())?;
            if result.commitments.len() != 1 {
                return Err(format!(
                    "modular ZK commitment emitted {} commitments, expected 1",
                    result.commitments.len()
                ));
            }
            Ok(1)
        }
    }

    struct ZkCommitmentWitness<'a> {
        values: &'a [u64],
    }

    impl WitnessProvider<Fr, ZkCommitmentNamespace> for ZkCommitmentWitness<'_> {
        fn describe_oracle(
            &self,
            oracle: OracleRef<ZkCommitmentNamespace>,
        ) -> Result<OracleDescriptor<ZkCommitmentNamespace>, WitnessError> {
            let OracleKind::Committed(0) = oracle.kind else {
                return Err(WitnessError::UnsupportedFrontier {
                    frontier: "stage0 ZK commitment benchmark oracle",
                });
            };
            Ok(OracleDescriptor::new(
                oracle,
                WitnessDimensions::new(
                    self.values.len(),
                    self.values.len().trailing_zeros() as usize,
                ),
                PolynomialEncoding::Compact,
            ))
        }

        fn view_requirements(
            &self,
            oracle: OracleRef<ZkCommitmentNamespace>,
        ) -> Result<Vec<ViewRequirement<ZkCommitmentNamespace>>, WitnessError> {
            Ok(vec![ViewRequirement::new(
                oracle,
                PolynomialEncoding::Compact,
                MaterializationPolicy::Streaming,
                RetentionHint::ThroughBlindFold,
            )])
        }

        fn oracle_view(
            &self,
            request: OracleViewRequest<ZkCommitmentNamespace>,
        ) -> Result<PolynomialView<'_, Fr, ZkCommitmentNamespace>, WitnessError> {
            let descriptor = self.describe_oracle(request.oracle())?;
            let values = self
                .values
                .iter()
                .copied()
                .map(Fr::from_u64)
                .collect::<Vec<_>>();
            Ok(PolynomialView::owned(descriptor, values))
        }

        fn committed_stream<'b>(
            &'b self,
            id: u8,
            chunk_size: usize,
        ) -> Result<Box<dyn PolynomialStream<Fr> + 'b>, WitnessError>
        where
            Fr: 'b,
            ZkCommitmentNamespace: 'b,
        {
            match id {
                0 => Ok(Box::new(ZkCommitmentStream {
                    values: self.values,
                    emitted: 0,
                    chunk_size,
                })),
                _ => Err(WitnessError::UnsupportedFrontier {
                    frontier: "stage0 ZK commitment benchmark stream",
                }),
            }
        }
    }

    impl CommittedWitnessProvider<Fr, ZkCommitmentNamespace> for ZkCommitmentWitness<'_> {
        fn committed_oracle_order(&self) -> Result<Vec<u8>, WitnessError> {
            Ok(vec![0])
        }
    }

    struct ZkCommitmentStream<'a> {
        values: &'a [u64],
        emitted: usize,
        chunk_size: usize,
    }

    impl PolynomialStream<Fr> for ZkCommitmentStream<'_> {
        fn next_chunk(&mut self) -> Result<Option<PolynomialChunk<Fr>>, WitnessError> {
            if self.emitted >= self.values.len() {
                return Ok(None);
            }
            let end = self
                .emitted
                .saturating_add(self.chunk_size)
                .min(self.values.len());
            let values = self.values[self.emitted..end].to_vec();
            self.emitted = end;
            Ok(Some(PolynomialChunk::U64(values)))
        }
    }

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let fixture = ZkCommitmentFixture::new(ROWS);
    let shape = fixture.shape();

    let core = measure_samples(samples, || {
        let count = fixture.run_core_zk_streaming_commitment();
        let _ = black_box(count);
    });

    let modular = measure_samples(samples, || {
        let count = fixture
            .run_modular_zk_streaming_commitment()
            .expect("run modular stage0 ZK streaming commitment");
        let _ = black_box(count);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage0_zk_streaming_commitment_memory(shape),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered stage0 ZK streaming commitment kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect(
                "stage0 ZK streaming commitment kernel evidence should pass the canonical gate",
            );
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical stage0 ZK streaming commitment evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_blindfold_round_commitment_kernel_evidence() {
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_blindfold_round_commitments";
    const BENCHMARK: &str = "frontier_perf/zk_blindfold_core_fixture";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-ZK-001"];

    let samples = blindfold_kernel_evidence_samples();
    let workspace = workspace_root();
    let fixture = BlindFoldKernelFixture::new();
    let direct = fixture
        .run_direct_round_commitments()
        .expect("run direct BlindFold round commitments");
    let backend = fixture
        .run_backend_round_commitments()
        .expect("run backend BlindFold round commitments");
    assert_eq!(direct, backend);

    let core = measure_samples(samples, || {
        let commitments = fixture
            .run_direct_round_commitments()
            .expect("run direct BlindFold round commitments");
        let _commitments = black_box(commitments);
    });

    let modular = measure_samples(samples, || {
        let commitments = fixture
            .run_backend_round_commitments()
            .expect("run backend BlindFold round commitments");
        let _commitments = black_box(commitments);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: blindfold_round_commitment_memory(
            fixture.round_rows.len(),
            fixture.round_row_len(),
        ),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered BlindFold round commitment kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("BlindFold round commitment evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical BlindFold round commitment evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_blindfold_backend_kernel_evidence() {
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_blindfold_backend_kernels";
    const BENCHMARK: &str = "frontier_perf/blindfold_witness_rows";
    const OPTIMIZATION_IDS: [&str; 3] = ["OPT-ZK-002", "OPT-ZK-003", "OPT-ZK-006"];

    let samples = blindfold_kernel_evidence_samples();
    let workspace = workspace_root();
    let fixture = BlindFoldKernelFixture::new();
    let direct = fixture
        .run_direct_backend_kernel_suite()
        .expect("run direct BlindFold backend kernel suite");
    let backend = fixture
        .run_backend_kernel_suite()
        .expect("run backend BlindFold kernel suite");
    assert_eq!(direct, backend);

    let core = measure_samples(samples, || {
        let digest = fixture
            .run_direct_backend_kernel_suite()
            .expect("run direct BlindFold backend kernel suite");
        let _digest = black_box(digest);
    });

    let modular = measure_samples(samples, || {
        let digest = fixture
            .run_backend_kernel_suite()
            .expect("run backend BlindFold kernel suite");
        let _digest = black_box(digest);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: blindfold_backend_kernel_memory(&fixture),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered BlindFold backend kernel row");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("BlindFold backend kernel evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical BlindFold backend kernel evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
struct BlindFoldKernelFixture {
    setup: jolt_crypto::PedersenSetup<jolt_crypto::Bn254G1>,
    round_rows: Vec<Vec<jolt_field::Fr>>,
    round_blindings: Vec<jolt_field::Fr>,
    witness_rows: Vec<Vec<jolt_field::Fr>>,
    witness_blindings: Vec<jolt_field::Fr>,
    output_rows: Vec<Vec<jolt_field::Fr>>,
    output_blindings: Vec<jolt_field::Fr>,
    real_witness: Vec<jolt_field::Fr>,
    random_witness: Vec<jolt_field::Fr>,
    real_u: jolt_field::Fr,
    random_u: jolt_field::Fr,
    r1cs: jolt_r1cs::ConstraintMatrices<jolt_field::Fr>,
    error_row_count: usize,
    error_row_len: usize,
    row_point: Vec<jolt_field::Fr>,
    entry_point: Vec<jolt_field::Fr>,
    challenge: jolt_field::Fr,
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct BlindFoldBackendKernelDigest {
    witness_commitments: Vec<jolt_crypto::Bn254G1>,
    output_commitments: Vec<jolt_crypto::Bn254G1>,
    real_error_rows: Vec<Vec<jolt_field::Fr>>,
    random_error_rows: Vec<Vec<jolt_field::Fr>>,
    cross_term_error_rows: Vec<Vec<jolt_field::Fr>>,
    folded_witness_rows: Vec<Vec<jolt_field::Fr>>,
    folded_witness_blindings: Vec<jolt_field::Fr>,
    folded_error_rows: Vec<Vec<jolt_field::Fr>>,
    folded_error_blindings: Vec<jolt_field::Fr>,
    folded_eval_outputs: Vec<jolt_field::Fr>,
    witness_opening: jolt_crypto::VectorCommitmentOpening<jolt_field::Fr>,
    witness_evaluation: jolt_field::Fr,
    error_opening: jolt_crypto::VectorCommitmentOpening<jolt_field::Fr>,
    error_evaluation: jolt_field::Fr,
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
impl BlindFoldKernelFixture {
    fn new() -> Self {
        const VC_CAPACITY: usize = 16;
        const ROUND_ROWS: usize = 512;
        const ROUND_ROW_LEN: usize = 4;
        const WITNESS_ROWS: usize = 64;
        const WITNESS_ROW_LEN: usize = 16;
        const ERROR_ROW_COUNT: usize = 64;
        const ERROR_ROW_LEN: usize = 16;
        const WITNESS_VALUES: usize = 31;

        Self {
            setup: blindfold_pedersen_setup(VC_CAPACITY),
            round_rows: blindfold_rows(ROUND_ROWS, ROUND_ROW_LEN, 10_000),
            round_blindings: blindfold_scalars(ROUND_ROWS, 20_000),
            witness_rows: blindfold_rows(WITNESS_ROWS, WITNESS_ROW_LEN, 30_000),
            witness_blindings: blindfold_scalars(WITNESS_ROWS, 40_000),
            output_rows: blindfold_rows(WITNESS_ROWS, WITNESS_ROW_LEN, 50_000),
            output_blindings: blindfold_scalars(WITNESS_ROWS, 60_000),
            real_witness: blindfold_scalars(WITNESS_VALUES, 70_000),
            random_witness: blindfold_scalars(WITNESS_VALUES, 80_000),
            real_u: blindfold_scalar(90_001),
            random_u: blindfold_scalar(90_003),
            r1cs: blindfold_r1cs(
                ERROR_ROW_COUNT * ERROR_ROW_LEN,
                WITNESS_VALUES + 1,
                WITNESS_VALUES,
            ),
            error_row_count: ERROR_ROW_COUNT,
            error_row_len: ERROR_ROW_LEN,
            row_point: blindfold_scalars(6, 100_000),
            entry_point: blindfold_scalars(4, 110_000),
            challenge: blindfold_scalar(120_001),
        }
    }

    fn round_row_len(&self) -> usize {
        self.round_rows.first().map_or(0, Vec::len)
    }

    fn run_direct_round_commitments(&self) -> Result<Vec<jolt_crypto::Bn254G1>, String> {
        use jolt_blindfold::{BlindFoldRowCommitter, DirectBlindFoldRowCommitter};
        use jolt_crypto::{Bn254G1, Pedersen};
        use jolt_field::Fr;

        let mut committer = DirectBlindFoldRowCommitter;
        <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<Fr, Pedersen<Bn254G1>>>::commit_rows(
            &mut committer,
            &self.setup,
            &self.round_rows,
            &self.round_blindings,
            "round polynomial rows",
        )
        .map_err(|error| error.to_string())
    }

    fn run_backend_round_commitments(&self) -> Result<Vec<jolt_crypto::Bn254G1>, String> {
        use jolt_backends::cpu::CpuBackend;
        use jolt_backends::{BlindFoldBackend, BlindFoldRowCommitmentRequest};
        use jolt_crypto::{Bn254G1, Pedersen};

        let mut backend = CpuBackend::default();
        Ok(backend
            .commit_blindfold_rows::<Pedersen<Bn254G1>>(
                BlindFoldRowCommitmentRequest::new(
                    "round polynomial rows",
                    &self.round_rows,
                    &self.round_blindings,
                ),
                &self.setup,
            )
            .map_err(|error| error.to_string())?
            .commitments)
    }

    fn run_direct_backend_kernel_suite(&self) -> Result<BlindFoldBackendKernelDigest, String> {
        use jolt_blindfold::{BlindFoldRowCommitter, DirectBlindFoldRowCommitter};
        use jolt_crypto::{Bn254G1, Pedersen};
        use jolt_field::Fr;

        let mut committer = DirectBlindFoldRowCommitter;
        let witness_commitments = <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<
            Fr,
            Pedersen<Bn254G1>,
        >>::commit_rows(
            &mut committer,
            &self.setup,
            &self.witness_rows,
            &self.witness_blindings,
            "witness rows",
        )
        .map_err(|error| error.to_string())?;
        let output_commitments = <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<
            Fr,
            Pedersen<Bn254G1>,
        >>::commit_rows(
            &mut committer,
            &self.setup,
            &self.output_rows,
            &self.output_blindings,
            "output claim rows",
        )
        .map_err(|error| error.to_string())?;
        let real_error_rows = <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<
            Fr,
            Pedersen<Bn254G1>,
        >>::compute_error_rows(
            &mut committer,
            &self.r1cs,
            self.real_u,
            &self.real_witness,
            self.error_row_count,
            self.error_row_len,
            "real error rows",
        )
        .map_err(|error| error.to_string())?;
        let random_error_rows = <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<
            Fr,
            Pedersen<Bn254G1>,
        >>::compute_error_rows(
            &mut committer,
            &self.r1cs,
            self.random_u,
            &self.random_witness,
            self.error_row_count,
            self.error_row_len,
            "random error rows",
        )
        .map_err(|error| error.to_string())?;
        let cross_term_error_rows = <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<
            Fr,
            Pedersen<Bn254G1>,
        >>::compute_cross_term_error_rows(
            &mut committer,
            &self.r1cs,
            self.real_u,
            &self.real_witness,
            self.random_u,
            &self.random_witness,
            self.error_row_count,
            self.error_row_len,
            "cross-term error rows",
        )
        .map_err(|error| error.to_string())?;
        let folded_witness_rows = <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<
            Fr,
            Pedersen<Bn254G1>,
        >>::fold_rows(
            &mut committer,
            &self.witness_rows,
            &self.output_rows,
            self.challenge,
            "folded witness rows",
        )
        .map_err(|error| error.to_string())?;
        let folded_witness_blindings = <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<
            Fr,
            Pedersen<Bn254G1>,
        >>::fold_scalars(
            &mut committer,
            &self.witness_blindings,
            &self.output_blindings,
            self.challenge,
            "folded witness blindings",
        )
        .map_err(|error| error.to_string())?;
        let folded_error_rows = <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<
            Fr,
            Pedersen<Bn254G1>,
        >>::fold_error_rows(
            &mut committer,
            &real_error_rows,
            &cross_term_error_rows,
            &random_error_rows,
            self.challenge,
            "folded error rows",
        )
        .map_err(|error| error.to_string())?;
        let folded_error_blindings = <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<
            Fr,
            Pedersen<Bn254G1>,
        >>::fold_error_scalars(
            &mut committer,
            &self.witness_blindings,
            &self.output_blindings,
            &self.round_blindings[..self.witness_blindings.len()],
            self.challenge,
            "folded error blindings",
        )
        .map_err(|error| error.to_string())?;
        let folded_eval_outputs = <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<
            Fr,
            Pedersen<Bn254G1>,
        >>::fold_scalars(
            &mut committer,
            &self.real_witness,
            &self.random_witness,
            self.challenge,
            "folded eval outputs",
        )
        .map_err(|error| error.to_string())?;
        let (witness_opening, witness_evaluation) =
            <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<Fr, Pedersen<Bn254G1>>>::open_rows(
                &mut committer,
                &self.setup,
                &folded_witness_rows,
                &folded_witness_blindings,
                &self.row_point,
                &self.entry_point,
                "folded witness row opening",
            )
            .map_err(|error| error.to_string())?;
        let (error_opening, error_evaluation) =
            <DirectBlindFoldRowCommitter as BlindFoldRowCommitter<Fr, Pedersen<Bn254G1>>>::open_rows(
                &mut committer,
                &self.setup,
                &folded_error_rows,
                &folded_error_blindings,
                &self.row_point,
                &self.entry_point,
                "folded error row opening",
            )
            .map_err(|error| error.to_string())?;

        Ok(BlindFoldBackendKernelDigest {
            witness_commitments,
            output_commitments,
            real_error_rows,
            random_error_rows,
            cross_term_error_rows,
            folded_witness_rows,
            folded_witness_blindings,
            folded_error_rows,
            folded_error_blindings,
            folded_eval_outputs,
            witness_opening,
            witness_evaluation,
            error_opening,
            error_evaluation,
        })
    }

    fn run_backend_kernel_suite(&self) -> Result<BlindFoldBackendKernelDigest, String> {
        use jolt_backends::cpu::CpuBackend;
        use jolt_backends::{
            BlindFoldBackend, BlindFoldCrossTermErrorRowsRequest, BlindFoldErrorRowsRequest,
            BlindFoldFoldErrorRowsRequest, BlindFoldFoldErrorScalarsRequest,
            BlindFoldFoldRowsRequest, BlindFoldFoldScalarsRequest, BlindFoldRowCommitmentRequest,
            BlindFoldRowOpeningRequest,
        };
        use jolt_crypto::{Bn254G1, Pedersen};

        let mut backend = CpuBackend::default();
        let witness_commitments = backend
            .commit_blindfold_rows::<Pedersen<Bn254G1>>(
                BlindFoldRowCommitmentRequest::new(
                    "witness rows",
                    &self.witness_rows,
                    &self.witness_blindings,
                ),
                &self.setup,
            )
            .map_err(|error| error.to_string())?
            .commitments;
        let output_commitments = backend
            .commit_blindfold_rows::<Pedersen<Bn254G1>>(
                BlindFoldRowCommitmentRequest::new(
                    "output claim rows",
                    &self.output_rows,
                    &self.output_blindings,
                ),
                &self.setup,
            )
            .map_err(|error| error.to_string())?
            .commitments;
        let real_error_rows = backend
            .compute_blindfold_error_rows(BlindFoldErrorRowsRequest::new(
                "real error rows",
                &self.r1cs,
                self.real_u,
                &self.real_witness,
                self.error_row_count,
                self.error_row_len,
            ))
            .map_err(|error| error.to_string())?
            .rows;
        let random_error_rows = backend
            .compute_blindfold_error_rows(BlindFoldErrorRowsRequest::new(
                "random error rows",
                &self.r1cs,
                self.random_u,
                &self.random_witness,
                self.error_row_count,
                self.error_row_len,
            ))
            .map_err(|error| error.to_string())?
            .rows;
        let cross_term_error_rows = backend
            .compute_blindfold_cross_term_error_rows(BlindFoldCrossTermErrorRowsRequest::new(
                "cross-term error rows",
                &self.r1cs,
                self.real_u,
                &self.real_witness,
                self.random_u,
                &self.random_witness,
                self.error_row_count,
                self.error_row_len,
            ))
            .map_err(|error| error.to_string())?
            .rows;
        let folded_witness_rows = backend
            .fold_blindfold_rows(BlindFoldFoldRowsRequest::new(
                "folded witness rows",
                &self.witness_rows,
                &self.output_rows,
                self.challenge,
            ))
            .map_err(|error| error.to_string())?
            .rows;
        let folded_witness_blindings = backend
            .fold_blindfold_scalars(BlindFoldFoldScalarsRequest::new(
                "folded witness blindings",
                &self.witness_blindings,
                &self.output_blindings,
                self.challenge,
            ))
            .map_err(|error| error.to_string())?
            .scalars;
        let folded_error_rows = backend
            .fold_blindfold_error_rows(BlindFoldFoldErrorRowsRequest::new(
                "folded error rows",
                &real_error_rows,
                &cross_term_error_rows,
                &random_error_rows,
                self.challenge,
            ))
            .map_err(|error| error.to_string())?
            .rows;
        let folded_error_blindings = backend
            .fold_blindfold_error_scalars(BlindFoldFoldErrorScalarsRequest::new(
                "folded error blindings",
                &self.witness_blindings,
                &self.output_blindings,
                &self.round_blindings[..self.witness_blindings.len()],
                self.challenge,
            ))
            .map_err(|error| error.to_string())?
            .scalars;
        let folded_eval_outputs = backend
            .fold_blindfold_scalars(BlindFoldFoldScalarsRequest::new(
                "folded eval outputs",
                &self.real_witness,
                &self.random_witness,
                self.challenge,
            ))
            .map_err(|error| error.to_string())?
            .scalars;
        let witness_opening = backend
            .open_blindfold_rows::<Pedersen<Bn254G1>>(
                BlindFoldRowOpeningRequest::new(
                    "folded witness row opening",
                    &folded_witness_rows,
                    &folded_witness_blindings,
                    &self.row_point,
                    &self.entry_point,
                ),
                &self.setup,
            )
            .map_err(|error| error.to_string())?;
        let error_opening = backend
            .open_blindfold_rows::<Pedersen<Bn254G1>>(
                BlindFoldRowOpeningRequest::new(
                    "folded error row opening",
                    &folded_error_rows,
                    &folded_error_blindings,
                    &self.row_point,
                    &self.entry_point,
                ),
                &self.setup,
            )
            .map_err(|error| error.to_string())?;

        Ok(BlindFoldBackendKernelDigest {
            witness_commitments,
            output_commitments,
            real_error_rows,
            random_error_rows,
            cross_term_error_rows,
            folded_witness_rows,
            folded_witness_blindings,
            folded_error_rows,
            folded_error_blindings,
            folded_eval_outputs,
            witness_opening: witness_opening.opening,
            witness_evaluation: witness_opening.evaluation,
            error_opening: error_opening.opening,
            error_evaluation: error_opening.evaluation,
        })
    }
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn blindfold_pedersen_setup(capacity: usize) -> jolt_crypto::PedersenSetup<jolt_crypto::Bn254G1> {
    use jolt_crypto::{Bn254, JoltGroup, PedersenSetup};
    use jolt_field::{Fr, FromPrimitiveInt};

    let generator = Bn254::g1_generator();
    let message_generators = (0..capacity)
        .map(|index| generator.scalar_mul(&Fr::from_u64(130_000 + index as u64)))
        .collect::<Vec<_>>();
    PedersenSetup::new(
        message_generators,
        generator.scalar_mul(&Fr::from_u64(130_999)),
    )
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn blindfold_r1cs(
    num_constraints: usize,
    num_vars: usize,
    witness_values: usize,
) -> jolt_r1cs::ConstraintMatrices<jolt_field::Fr> {
    let a = (0..num_constraints)
        .map(|row| {
            vec![
                (
                    1 + row % witness_values,
                    blindfold_scalar(140_000 + row as u64 * 3),
                ),
                (0, blindfold_scalar(150_000 + row as u64)),
            ]
        })
        .collect::<Vec<_>>();
    let b = (0..num_constraints)
        .map(|row| {
            vec![(
                1 + (row * 7 + 3) % witness_values,
                blindfold_scalar(160_000 + row as u64 * 5),
            )]
        })
        .collect::<Vec<_>>();
    let c = (0..num_constraints)
        .map(|row| {
            vec![(
                1 + (row * 11 + 5) % witness_values,
                blindfold_scalar(170_000 + row as u64 * 7),
            )]
        })
        .collect::<Vec<_>>();
    jolt_r1cs::ConstraintMatrices::new(num_constraints, num_vars, a, b, c)
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn blindfold_rows(rows: usize, row_len: usize, seed: u64) -> Vec<Vec<jolt_field::Fr>> {
    (0..rows)
        .map(|row| {
            (0..row_len)
                .map(|column| blindfold_scalar(seed + row as u64 * 257 + column as u64 * 17))
                .collect()
        })
        .collect()
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn blindfold_scalars(len: usize, seed: u64) -> Vec<jolt_field::Fr> {
    (0..len)
        .map(|index| blindfold_scalar(seed + index as u64 * 19))
        .collect()
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn blindfold_scalar(value: u64) -> jolt_field::Fr {
    use jolt_field::{Fr, FromPrimitiveInt};

    Fr::from_u64(value)
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn blindfold_kernel_evidence_samples() -> u32 {
    std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples)
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage0_advice_commitment_context_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage0_advice_commitment_kernel_benchmark_fixture, validate_kernel_benchmark_evidence,
        FeatureMode, FixtureKind, FixtureRequest, KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_advice_commitment_contexts";
    const BENCHMARK: &str = "frontier_perf/stage0_advice_commitments";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-COM-005", "OPT-COM-006"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::AdviceConsumer, FeatureMode::Transparent);
    let fixture = load_stage0_advice_commitment_kernel_benchmark_fixture(&request)
        .expect("load stage0 advice commitment kernel fixture");
    fixture
        .verify_commitment_parity()
        .expect("stage0 advice commitment parity");
    let shape = fixture.shape();

    let core = measure_samples(samples, || {
        let count = fixture
            .run_core_advice_context_commitments()
            .expect("run core stage0 advice commitments");
        let _ = black_box(count);
    });

    let modular = measure_samples(samples, || {
        let count = fixture
            .run_modular_advice_context_commitments()
            .expect("run modular stage0 advice commitments");
        let _ = black_box(count);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage0_advice_commitment_context_memory(shape),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered stage0 advice commitment kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("stage0 advice commitment kernel evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical stage0 advice commitment evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stage0FieldInlineCommitmentKernelShape {
    rows: usize,
    row_width: usize,
    pcs_rows: usize,
    chunk_size: usize,
    chunks: usize,
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
impl Stage0FieldInlineCommitmentKernelShape {
    const fn new(rows: usize, chunk_size: usize) -> Self {
        let log_rows = rows.trailing_zeros() as usize;
        let row_width = 1_usize << log_rows.div_ceil(2);
        Self {
            rows,
            row_width,
            pcs_rows: rows / row_width,
            chunk_size,
            chunks: rows.div_ceil(chunk_size),
        }
    }
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage0_field_inline_commitment_kernel_evidence() {
    use jolt_backends::{
        cpu::{CpuBackend, CpuBackendConfig},
        CommitmentBackend, CommitmentRequest, CommitmentRequestItem, CommitmentSlot,
    };
    use jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial;
    use jolt_dory::{DoryProverSetup, DoryScheme};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_openings::{CommitmentScheme, StreamingCommitment};
    use jolt_poly::Polynomial;
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};
    use jolt_witness::protocols::jolt_vm::field_inline::FieldInlineNamespace;
    use jolt_witness::{
        CommittedWitnessProvider, MaterializationPolicy, OracleDescriptor, OracleKind, OracleRef,
        OracleViewRequest, PolynomialChunk, PolynomialEncoding, PolynomialStream, PolynomialView,
        RetentionHint, ViewRequirement, WitnessDimensions, WitnessError, WitnessProvider,
    };

    const KERNEL: &str = "cpu_field_inline_commitments";
    const BENCHMARK: &str = "frontier_perf/stage0_field_inline_commitments";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-COM-001", "OPT-COM-006"];
    const ROWS: usize = 1 << 16;
    const CHUNK_SIZE: usize = 1024;

    struct FieldInlineCommitmentFixture {
        values: Vec<Fr>,
        setup: DoryProverSetup,
    }

    impl FieldInlineCommitmentFixture {
        fn new(rows: usize) -> Self {
            let values = (0..rows)
                .map(|index| Fr::from_u64((index as u64).wrapping_mul(17).wrapping_add(11)))
                .collect::<Vec<_>>();
            let setup = DoryScheme::setup_prover(rows.trailing_zeros() as usize);
            Self { values, setup }
        }

        fn shape(&self) -> Stage0FieldInlineCommitmentKernelShape {
            Stage0FieldInlineCommitmentKernelShape::new(self.values.len(), CHUNK_SIZE)
        }

        fn reference_direct_commitment(
            &self,
        ) -> (
            <DoryScheme as jolt_crypto::Commitment>::Output,
            <DoryScheme as CommitmentScheme>::OpeningHint,
        ) {
            let polynomial = Polynomial::new(self.values.clone());
            DoryScheme::commit(&polynomial, &self.setup)
        }

        fn reference_streaming_commitment(
            &self,
        ) -> (
            <DoryScheme as jolt_crypto::Commitment>::Output,
            <DoryScheme as CommitmentScheme>::OpeningHint,
        ) {
            let shape = self.shape();
            let mut partial = DoryScheme::begin(&self.setup);
            for row in self.values.chunks(shape.row_width) {
                DoryScheme::feed(&mut partial, row, &self.setup);
            }
            DoryScheme::finish_with_hint(partial, &self.setup)
        }

        fn modular_streaming_commitment(
            &self,
        ) -> Result<
            (
                <DoryScheme as jolt_crypto::Commitment>::Output,
                <DoryScheme as CommitmentScheme>::OpeningHint,
            ),
            String,
        > {
            let witness = FieldInlineCommitmentWitness {
                values: &self.values,
            };
            let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
                CommitmentSlot(0),
                ViewRequirement::new(
                    OracleRef::committed(FieldInlineCommittedPolynomial::FieldRdInc),
                    PolynomialEncoding::Dense,
                    MaterializationPolicy::Streaming,
                    RetentionHint::ThroughBlindFold,
                ),
            )]);
            let mut backend = CpuBackend::new(CpuBackendConfig {
                preserve_core_fast_path: true,
                commitment_chunk_size: CHUNK_SIZE,
            });
            let result =
                <CpuBackend as CommitmentBackend<Fr, FieldInlineNamespace, DoryScheme>>::commit(
                    &mut backend,
                    &request,
                    &witness,
                    &self.setup,
                )
                .map_err(|error| error.to_string())?;
            let [output] = result.commitments.as_slice() else {
                return Err(format!(
                    "field-inline commitment backend emitted {} outputs, expected 1",
                    result.commitments.len()
                ));
            };
            Ok((output.commitment.clone(), output.opening_hint.clone()))
        }

        fn verify_commitment_parity(&self) -> Result<(), String> {
            let direct = self.reference_direct_commitment().0;
            let reference = self.reference_streaming_commitment().0;
            let modular = self.modular_streaming_commitment()?.0;
            if direct == reference && reference == modular {
                Ok(())
            } else {
                Err(
                    "field-inline direct, reference-streaming, and backend-streaming commitments differ"
                        .to_owned(),
                )
            }
        }

        fn run_reference_streaming_commitment(&self) -> usize {
            let _commitment = self.reference_streaming_commitment();
            1
        }

        fn run_modular_streaming_commitment(&self) -> Result<usize, String> {
            let _commitment = self.modular_streaming_commitment()?;
            Ok(1)
        }
    }

    struct FieldInlineCommitmentWitness<'a> {
        values: &'a [Fr],
    }

    impl WitnessProvider<Fr, FieldInlineNamespace> for FieldInlineCommitmentWitness<'_> {
        fn describe_oracle(
            &self,
            oracle: OracleRef<FieldInlineNamespace>,
        ) -> Result<OracleDescriptor<FieldInlineNamespace>, WitnessError> {
            let OracleKind::Committed(FieldInlineCommittedPolynomial::FieldRdInc) = oracle.kind
            else {
                return Err(WitnessError::UnsupportedFrontier {
                    frontier: "field-inline commitment benchmark oracle",
                });
            };
            Ok(OracleDescriptor::new(
                oracle,
                WitnessDimensions::new(
                    self.values.len(),
                    self.values.len().trailing_zeros() as usize,
                ),
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
            request: OracleViewRequest<FieldInlineNamespace>,
        ) -> Result<PolynomialView<'_, Fr, FieldInlineNamespace>, WitnessError> {
            let descriptor = self.describe_oracle(request.oracle())?;
            Ok(PolynomialView::owned(descriptor, self.values.to_vec()))
        }

        fn committed_stream<'b>(
            &'b self,
            id: FieldInlineCommittedPolynomial,
            chunk_size: usize,
        ) -> Result<Box<dyn PolynomialStream<Fr> + 'b>, WitnessError>
        where
            Fr: 'b,
            FieldInlineNamespace: 'b,
        {
            match id {
                FieldInlineCommittedPolynomial::FieldRdInc => {
                    Ok(Box::new(FieldInlineCommitmentStream {
                        values: self.values,
                        emitted: 0,
                        chunk_size,
                    }))
                }
            }
        }
    }

    impl CommittedWitnessProvider<Fr, FieldInlineNamespace> for FieldInlineCommitmentWitness<'_> {
        fn committed_oracle_order(
            &self,
        ) -> Result<Vec<FieldInlineCommittedPolynomial>, WitnessError> {
            Ok(vec![FieldInlineCommittedPolynomial::FieldRdInc])
        }
    }

    struct FieldInlineCommitmentStream<'a> {
        values: &'a [Fr],
        emitted: usize,
        chunk_size: usize,
    }

    impl PolynomialStream<Fr> for FieldInlineCommitmentStream<'_> {
        fn next_chunk(&mut self) -> Result<Option<PolynomialChunk<Fr>>, WitnessError> {
            if self.emitted >= self.values.len() {
                return Ok(None);
            }
            let end = self
                .emitted
                .saturating_add(self.chunk_size)
                .min(self.values.len());
            let values = self.values[self.emitted..end].to_vec();
            self.emitted = end;
            Ok(Some(PolynomialChunk::Dense(values)))
        }
    }

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let fixture = FieldInlineCommitmentFixture::new(ROWS);
    fixture
        .verify_commitment_parity()
        .expect("stage0 field-inline commitment parity");
    let shape = fixture.shape();

    let core = measure_samples(samples, || {
        let count = fixture.run_reference_streaming_commitment();
        let _ = black_box(count);
    });

    let modular = measure_samples(samples, || {
        let count = fixture
            .run_modular_streaming_commitment()
            .expect("run modular stage0 field-inline commitment");
        let _ = black_box(count);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage0_field_inline_commitment_memory(shape),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered stage0 field-inline commitment kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect(
                "stage0 field-inline commitment kernel evidence should pass the canonical gate",
            );
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical stage0 field-inline commitment evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_one_hot_commitment_kernel_evidence() {
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        poly::{
            commitment::{
                commitment_scheme::CommitmentScheme as CoreCommitmentScheme,
                dory::{DoryCommitmentScheme, DoryContext, DoryGlobals, DoryLayout},
            },
            multilinear_polynomial::MultilinearPolynomial,
            one_hot_polynomial::OneHotPolynomial as CoreOneHotPolynomial,
        },
    };
    use jolt_dory::DoryScheme;
    use jolt_openings::CommitmentScheme as ModularCommitmentScheme;
    use jolt_poly::{OneHotIndexOrder, OneHotPolynomial};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_one_hot_commitments";
    const BENCHMARK: &str = "frontier_perf/one_hot_commitments";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-COM-007", "OPT-COM-008"];
    const K: usize = 256;
    const T: usize = 1 << 16;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let indices = deterministic_one_hot_indices(K, T);
    let num_vars = (K * T).trailing_zeros() as usize;

    let _ = DoryGlobals::initialize_context(K, T, DoryContext::Main, Some(DoryLayout::CycleMajor));
    let core_setup = DoryCommitmentScheme::setup_prover(num_vars);
    let core_poly = MultilinearPolynomial::OneHot(CoreOneHotPolynomial::<CoreFr>::from_indices(
        indices.clone(),
        K,
    ));

    let modular_setup = DoryScheme::setup_prover(num_vars);
    let modular_poly =
        OneHotPolynomial::new_with_index_order(K, indices, OneHotIndexOrder::ColumnMajor);

    let core = measure_samples(samples, || {
        let (commitment, hint) = DoryCommitmentScheme::commit(&core_poly, &core_setup);
        let _ = black_box((commitment, hint));
    });

    let modular = measure_samples(samples, || {
        let (commitment, hint) = DoryScheme::commit(&modular_poly, &modular_setup);
        let _ = black_box((commitment, hint));
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: one_hot_commitment_memory(K, T),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered one-hot commitment kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("one-hot commitment kernel evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical one-hot commitment evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage1_spartan_outer_prefix_product_sum_kernel_evidence() {
    use jolt_backends::{
        cpu::CpuBackend, BackendKernelMetadata, BackendRelationId, BackendValueSlot,
        SumcheckBackend, SumcheckSpartanOuterRemainderStateRequest, SumcheckSpartanOuterRow,
        SumcheckSpartanOuterUniskipQuery, SumcheckSpartanOuterUniskipRequest,
    };
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::opening_proof::{
            OpeningPoint as CoreOpeningPoint, ProverOpeningAccumulator as CoreOpeningAccumulator,
            SumcheckId as CoreSumcheckId,
        },
        subprotocols::streaming_schedule::LinearOnlySchedule,
        subprotocols::sumcheck_prover::SumcheckInstanceProver,
        transcripts::Blake2bTranscript as CoreBlake2bTranscript,
        zkvm::{
            instruction::CircuitFlags,
            r1cs::inputs::R1CSCycleInputs,
            spartan::outer::{
                OuterRemainingStreamingSumcheck, OuterSharedState, OuterUniSkipParams,
                OuterUniSkipProver,
            },
            witness::VirtualPolynomial as CoreVirtualPolynomial,
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::lagrange::{centered_domain_start, centered_lagrange_kernel};
    use jolt_prover_harness::{
        load_stage1_spartan_outer_kernel_benchmark_fixture, validate_kernel_benchmark_evidence,
        FeatureMode, FixtureKind, FixtureRequest, KnownOptimizationIds,
    };
    use jolt_r1cs::{
        constraints::{
            jolt::{
                spartan_outer_constraints, spartan_outer_opening_columns,
                spartan_outer_row_weights, SPARTAN_OUTER_REMAINDER_DEGREE,
                SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
            },
            rv64,
        },
        ConstraintMatrices,
    };
    use std::sync::Arc;

    const KERNEL: &str = "cpu_spartan_outer_prefix_product_sum";
    const UNISKIP_BENCHMARK: &str = "cpu_sumcheck/outer_uniskip_prefix_sum";
    const REMAINDER_BENCHMARK: &str = "cpu_sumcheck/outer_remainder_bound_prefix_sum";
    const OPTIMIZATION_IDS: [&str; 3] = ["OPT-SC-007", "OPT-SC-008", "OPT-EQ-004"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage1_spartan_outer_kernel_benchmark_fixture(&request)
        .expect("load Stage 1 Spartan outer kernel fixture");
    let core_tau = core_stage1_tau(fixture.log_t);
    let modular_tau = modular_stage1_tau(fixture.log_t);
    let core_uniskip_challenge = <CoreFr as JoltField>::Challenge::from(0x5300_0001u128);
    let modular_uniskip_challenge = Fr::from_u64(0x5300_0001);
    let core_remainder_round0 = <CoreFr as JoltField>::Challenge::from(0x5300_0101u128);
    let modular_remainder_round0 = Fr::from_u64(0x5300_0101);
    let rows = stage1_spartan_outer_rows_from_core(&fixture.core_trace, &fixture.core_bytecode);
    let modular_request = stage1_uniskip_raw_row_request(&rows, &modular_tau);

    let core = measure_samples(samples, || {
        let params = OuterUniSkipParams::<CoreFr> {
            tau: core_tau.clone(),
        };
        let mut prover =
            OuterUniSkipProver::initialize(params, &fixture.core_trace, &fixture.core_bytecode);
        let poly = <OuterUniSkipProver<CoreFr> as SumcheckInstanceProver<
            CoreFr,
            CoreBlake2bTranscript,
        >>::compute_message(&mut prover, 0, <CoreFr as JoltField>::from_u64(0));
        let _ = black_box(poly.coeffs.len());
    });

    let modular = measure_samples(samples, || {
        let mut backend = CpuBackend::default();
        let outputs = <CpuBackend as SumcheckBackend<
            Fr,
            jolt_witness::protocols::jolt_vm::JoltVmNamespace,
        >>::evaluate_sumcheck_spartan_outer_uniskip_rows(
            &mut backend, &modular_request
        )
        .expect("evaluate modular Stage 1 raw uniskip rows");
        let _ = black_box(outputs);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: UNISKIP_BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage1_spartan_outer_prefix_product_memory(
            fixture.log_t,
            rows.len(),
            SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE - 1,
        ),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 1 Spartan outer prefix-product kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 1 Spartan outer prefix-product evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 1 Spartan outer prefix-product evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    let core_uniskip_output_claim = core_stage1_uniskip_output_claim(
        &fixture.core_trace,
        &fixture.core_bytecode,
        &core_tau,
        core_uniskip_challenge,
    );
    let matrices = spartan_outer_constraints::<Fr>();
    let input_columns = spartan_outer_opening_columns();
    let modular_remainder_request = stage1_remainder_state_request(
        &fixture.r1cs_inputs,
        &input_columns,
        &matrices,
        &modular_tau,
        modular_uniskip_challenge,
        modular_remainder_round0,
    );

    let core = measure_samples_with_setup(
        samples,
        || {
            core_stage1_remainder_after_round0(
                &fixture.core_trace,
                &fixture.core_bytecode,
                fixture.log_t,
                &core_tau,
                core_uniskip_challenge,
                core_uniskip_output_claim,
                core_remainder_round0,
            )
        },
        |(prover, previous_claim)| {
            let poly = <OuterRemainingStreamingSumcheck<CoreFr, LinearOnlySchedule> as SumcheckInstanceProver<
                CoreFr,
                CoreBlake2bTranscript,
            >>::compute_message(prover, 1, *previous_claim);
            let _ = black_box(poly.coeffs.len());
        },
    );

    let modular = measure_samples_with_setup(
        samples,
        || {
            let mut backend = CpuBackend::default();
            <CpuBackend as SumcheckBackend<
                Fr,
                jolt_witness::protocols::jolt_vm::JoltVmNamespace,
            >>::materialize_sumcheck_spartan_outer_remainder_state(
                &mut backend,
                &modular_remainder_request,
            )
            .expect("materialize modular Stage 1 remainder state")
        },
        |state| {
            let mut backend = CpuBackend::default();
            let round = <CpuBackend as SumcheckBackend<
                Fr,
                jolt_witness::protocols::jolt_vm::JoltVmNamespace,
            >>::evaluate_sumcheck_spartan_outer_remainder_round(
                &mut backend, state
            )
            .expect("evaluate modular Stage 1 remainder round");
            let _ = black_box(round);
        },
    );

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: REMAINDER_BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage1_spartan_outer_remainder_prefix_product_memory(
            fixture.log_t,
            fixture.r1cs_inputs.len(),
            1usize << fixture.log_t,
            matrices.a.len(),
            sparse_matrix_terms(&matrices),
            SPARTAN_OUTER_REMAINDER_DEGREE + 1,
            2,
        ),
    };

    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 1 Spartan outer remainder evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 1 Spartan outer remainder evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    fn core_stage1_tau(log_t: usize) -> Vec<<CoreFr as JoltField>::Challenge> {
        (0..(log_t + 2))
            .map(|index| <CoreFr as JoltField>::Challenge::from(0x5100_0000 + index as u128 * 31))
            .collect()
    }

    fn modular_stage1_tau(log_t: usize) -> Vec<Fr> {
        (0..(log_t + 2))
            .map(|index| Fr::from_u64(0x5100_0000 + index as u64 * 31))
            .collect()
    }

    fn core_stage1_uniskip_output_claim(
        trace: &Arc<Vec<tracer::instruction::Cycle>>,
        bytecode: &jolt_core::zkvm::bytecode::BytecodePreprocessing,
        tau: &[<CoreFr as JoltField>::Challenge],
        challenge: <CoreFr as JoltField>::Challenge,
    ) -> CoreFr {
        let params = OuterUniSkipParams::<CoreFr> { tau: tau.to_vec() };
        let mut prover = OuterUniSkipProver::initialize(params, trace, bytecode);
        let poly = <OuterUniSkipProver<CoreFr> as SumcheckInstanceProver<
            CoreFr,
            CoreBlake2bTranscript,
        >>::compute_message(&mut prover, 0, <CoreFr as JoltField>::from_u64(0));
        poly.evaluate(&challenge)
    }

    fn core_stage1_remainder_after_round0(
        trace: &Arc<Vec<tracer::instruction::Cycle>>,
        bytecode: &jolt_core::zkvm::bytecode::BytecodePreprocessing,
        log_t: usize,
        tau: &[<CoreFr as JoltField>::Challenge],
        uniskip_challenge: <CoreFr as JoltField>::Challenge,
        uniskip_output_claim: CoreFr,
        round0_challenge: <CoreFr as JoltField>::Challenge,
    ) -> (
        OuterRemainingStreamingSumcheck<CoreFr, LinearOnlySchedule>,
        CoreFr,
    ) {
        let uni_skip_params = OuterUniSkipParams::<CoreFr> { tau: tau.to_vec() };
        let mut opening_accumulator = CoreOpeningAccumulator::new(log_t);
        opening_accumulator.append_virtual(
            CoreVirtualPolynomial::UnivariateSkip,
            CoreSumcheckId::SpartanOuter,
            CoreOpeningPoint::new(vec![uniskip_challenge]),
            uniskip_output_claim,
        );
        let shared = OuterSharedState::new(
            Arc::clone(trace),
            bytecode,
            &uni_skip_params,
            &opening_accumulator,
        );
        let schedule = LinearOnlySchedule::new(uni_skip_params.tau.len() - 1);
        let mut prover = OuterRemainingStreamingSumcheck::new(shared, schedule);
        let round0_poly = <OuterRemainingStreamingSumcheck<
            CoreFr,
            LinearOnlySchedule,
        > as SumcheckInstanceProver<CoreFr, CoreBlake2bTranscript>>::compute_message(
            &mut prover,
            0,
            uniskip_output_claim,
        );
        let previous_claim = round0_poly.evaluate(&round0_challenge);
        <OuterRemainingStreamingSumcheck<CoreFr, LinearOnlySchedule> as SumcheckInstanceProver<
            CoreFr,
            CoreBlake2bTranscript,
        >>::ingest_challenge(&mut prover, round0_challenge, 0);
        (prover, previous_claim)
    }

    fn stage1_uniskip_raw_row_request<'a>(
        rows: &'a [SumcheckSpartanOuterRow],
        tau: &[Fr],
    ) -> SumcheckSpartanOuterUniskipRequest<'a, Fr> {
        let degree = SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE - 1;
        let tau_low = &tau[..tau.len() - 1];
        let queries = uniskip_targets(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE)
            .into_iter()
            .enumerate()
            .map(|(index, target)| {
                SumcheckSpartanOuterUniskipQuery::new(
                    BackendValueSlot(index as u32),
                    tau_low.to_vec(),
                    centered_lagrange_integer_coeffs(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, target),
                    Fr::from_u64(1),
                )
            })
            .take(degree)
            .collect();
        SumcheckSpartanOuterUniskipRequest::new(
            "evidence.stage1.spartan_outer.uniskip_raw_rows",
            rows,
            queries,
        )
        .with_kernel_metadata(BackendKernelMetadata::new(
            Some(BackendRelationId::new(
                "jolt_vm",
                "spartan_outer.uniskip_first_round",
            )),
            &OPTIMIZATION_IDS,
        ))
    }

    fn stage1_remainder_state_request<'a>(
        r1cs_inputs: &'a [Vec<Fr>],
        input_columns: &'a [usize],
        matrices: &'a ConstraintMatrices<Fr>,
        tau: &[Fr],
        uniskip_challenge: Fr,
        stream_challenge: Fr,
    ) -> SumcheckSpartanOuterRemainderStateRequest<'a, Fr> {
        let tau_low = &tau[..tau.len() - 1];
        let tau_scale = centered_lagrange_kernel(
            SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
            tau[tau.len() - 1],
            uniskip_challenge,
        )
        .expect("valid Stage 1 uniskip domain");
        let row_weights_at_zero = spartan_outer_row_weights(uniskip_challenge, Fr::from_u64(0))
            .expect("valid Stage 1 row weights at zero");
        let row_weights_at_one = spartan_outer_row_weights(uniskip_challenge, Fr::from_u64(1))
            .expect("valid Stage 1 row weights at one");
        SumcheckSpartanOuterRemainderStateRequest::new(
            "evidence.stage1.spartan_outer.remainder_state",
            r1cs_inputs,
            input_columns,
            rv64::const_column(),
            &matrices.a,
            &matrices.b,
            tau_low.to_vec(),
            row_weights_at_zero,
            row_weights_at_one,
            stream_challenge,
            tau_scale,
        )
        .with_kernel_metadata(BackendKernelMetadata::new(
            Some(BackendRelationId::new("jolt_vm", "spartan_outer.remainder")),
            &OPTIMIZATION_IDS,
        ))
    }

    fn sparse_matrix_terms(matrices: &ConstraintMatrices<Fr>) -> usize {
        matrices.a.iter().chain(&matrices.b).map(Vec::len).sum()
    }

    fn stage1_spartan_outer_rows_from_core(
        trace: &[tracer::instruction::Cycle],
        bytecode: &jolt_core::zkvm::bytecode::BytecodePreprocessing,
    ) -> Vec<SumcheckSpartanOuterRow> {
        (0..trace.len())
            .map(|index| {
                let row = R1CSCycleInputs::from_trace::<CoreFr>(bytecode, trace, index);
                SumcheckSpartanOuterRow {
                    left_instruction_input: row.left_input,
                    right_instruction_input: row.right_input.to_i128(),
                    product_magnitude: row.product.magnitude_as_u128(),
                    product_is_positive: row.product.is_positive,
                    should_branch: row.should_branch,
                    pc: row.pc,
                    unexpanded_pc: row.unexpanded_pc,
                    imm: row.imm.to_i128(),
                    ram_address: row.ram_addr,
                    rs1_value: row.rs1_read_value,
                    rs2_value: row.rs2_read_value,
                    rd_write_value: row.rd_write_value,
                    ram_read_value: row.ram_read_value,
                    ram_write_value: row.ram_write_value,
                    left_lookup_operand: row.left_lookup,
                    right_lookup_operand: row.right_lookup,
                    next_unexpanded_pc: row.next_unexpanded_pc,
                    next_pc: row.next_pc,
                    next_is_virtual: row.next_is_virtual,
                    next_is_first_in_sequence: row.next_is_first_in_sequence,
                    lookup_output: row.lookup_output,
                    should_jump: row.should_jump,
                    flag_add_operands: row.flags[CircuitFlags::AddOperands],
                    flag_subtract_operands: row.flags[CircuitFlags::SubtractOperands],
                    flag_multiply_operands: row.flags[CircuitFlags::MultiplyOperands],
                    flag_load: row.flags[CircuitFlags::Load],
                    flag_store: row.flags[CircuitFlags::Store],
                    flag_jump: row.flags[CircuitFlags::Jump],
                    flag_write_lookup_output_to_rd: row.flags[CircuitFlags::WriteLookupOutputToRD],
                    flag_virtual_instruction: row.flags[CircuitFlags::VirtualInstruction],
                    flag_assert: row.flags[CircuitFlags::Assert],
                    flag_do_not_update_unexpanded_pc: row.flags
                        [CircuitFlags::DoNotUpdateUnexpandedPC],
                    flag_advice: row.flags[CircuitFlags::Advice],
                    flag_is_compressed: row.flags[CircuitFlags::IsCompressed],
                    flag_is_first_in_sequence: row.flags[CircuitFlags::IsFirstInSequence],
                    flag_is_last_in_sequence: row.flags[CircuitFlags::IsLastInSequence],
                }
            })
            .collect()
    }

    fn centered_lagrange_integer_coeffs(domain_size: usize, target: i64) -> Vec<i32> {
        let start = -(((domain_size - 1) / 2) as i64);
        (0..domain_size)
            .map(|index| {
                let x_i = start + index as i64;
                let mut numerator = 1i128;
                let mut denominator = 1i128;
                for other in 0..domain_size {
                    if other == index {
                        continue;
                    }
                    let x_j = start + other as i64;
                    numerator *= i128::from(target - x_j);
                    denominator *= i128::from(x_i - x_j);
                }
                assert_eq!(numerator % denominator, 0);
                i32::try_from(numerator / denominator).expect("small centered Lagrange coefficient")
            })
            .collect()
    }

    fn uniskip_targets(domain_size: usize) -> Vec<i64> {
        let degree = domain_size - 1;
        let base_left = centered_domain_start(domain_size).expect("valid uniskip domain");
        let base_right = base_left + domain_size as i64 - 1;
        let ext_left = -(degree as i64);
        let ext_right = degree as i64;
        let mut targets = Vec::with_capacity(degree);
        let mut left = base_left - 1;
        let mut right = base_right + 1;

        while targets.len() < degree && left >= ext_left && right <= ext_right {
            targets.push(left);
            if targets.len() < degree {
                targets.push(right);
            }
            left -= 1;
            right += 1;
        }
        while targets.len() < degree && left >= ext_left {
            targets.push(left);
            left -= 1;
        }
        while targets.len() < degree && right <= ext_right {
            targets.push(right);
            right += 1;
        }
        targets
    }
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage2_product_uniskip_kernel_evidence() {
    use jolt_backends::{
        cpu::CpuBackend, BackendKernelMetadata, BackendRelationId, BackendValueSlot,
        SumcheckBackend, SumcheckProductUniskipRequest, SumcheckProductUniskipRow,
        SumcheckRowProductQuery,
    };
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        zkvm::spartan::product::{ProductVirtualUniSkipParams, ProductVirtualUniSkipProver},
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{
        load_stage2_product_uniskip_kernel_benchmark_fixture, validate_kernel_benchmark_evidence,
        FeatureMode, FixtureKind, FixtureRequest, KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_spartan_product_uniskip";
    const BENCHMARK: &str = "cpu_sumcheck/spartan_product_uniskip_raw";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-SC-007", "OPT-EQ-004"];
    const PRODUCT_WEIGHTS: [[i64; 3]; 2] = [[3, -3, 1], [1, -3, 3]];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage2_product_uniskip_kernel_benchmark_fixture(&request)
        .expect("load Stage 2 product uni-skip kernel fixture");
    let tau_core = core_tau(fixture.log_t);
    let modular_request = raw_product_uniskip_request(&fixture.rows, fixture.log_t);

    let core = measure_samples(samples, || {
        let params = ProductVirtualUniSkipParams::<CoreFr> {
            tau: tau_core.clone(),
            base_evals: [<CoreFr as JoltField>::from_u64(0); 3],
        };
        let prover = ProductVirtualUniSkipProver::initialize(params, &fixture.core_trace);
        let _ = black_box(prover.params.tau.len());
    });

    let modular = measure_samples(samples, || {
        let mut backend = CpuBackend::default();
        let outputs =
            <CpuBackend as SumcheckBackend<
                Fr,
                jolt_witness::protocols::jolt_vm::JoltVmNamespace,
            >>::evaluate_sumcheck_product_uniskip_rows(&mut backend, &modular_request)
            .expect("evaluate modular product uni-skip rows");
        let _ = black_box(outputs);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: raw_product_uniskip_memory(fixture.rows.len(), fixture.log_t),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered product uni-skip kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("product uni-skip kernel evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical product uni-skip evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    fn core_tau(log_t: usize) -> Vec<<CoreFr as JoltField>::Challenge> {
        (0..=log_t)
            .map(|index| <CoreFr as JoltField>::Challenge::from(0x00C0_FFEE + index as u128 * 17))
            .collect()
    }

    fn raw_product_uniskip_request(
        rows: &[SumcheckProductUniskipRow],
        log_t: usize,
    ) -> SumcheckProductUniskipRequest<'_, Fr> {
        let point = (0..log_t)
            .map(|index| Fr::from_u64(0x00C0_FFEE + index as u64 * 17))
            .collect::<Vec<_>>();
        let queries = PRODUCT_WEIGHTS
            .iter()
            .enumerate()
            .map(|(index, weights)| {
                SumcheckRowProductQuery::new(
                    BackendValueSlot(index as u32),
                    point.clone(),
                    weights.iter().copied().map(Fr::from_i64).collect(),
                    Fr::from_u64(1),
                )
            })
            .collect();
        SumcheckProductUniskipRequest::new("evidence.stage2.product_uniskip.raw", rows, queries)
            .with_kernel_metadata(BackendKernelMetadata::new(
                Some(BackendRelationId::new(
                    "jolt_vm",
                    "spartan_product.uniskip_first_round",
                )),
                &OPTIMIZATION_IDS,
            ))
    }
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage2_regular_batch_input_claim_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage2_regular_batch_input_kernel_benchmark_fixture,
        validate_kernel_benchmark_evidence, FeatureMode, FixtureKind, FixtureRequest,
        KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_stage2_regular_batch_input_claims";
    const BENCHMARK: &str = "frontier_perf/stage2_regular_batch_inputs";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-SC-007", "OPT-EQ-004"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage2_regular_batch_input_kernel_benchmark_fixture(&request)
        .expect("load Stage 2 regular-batch input-claim kernel fixture");
    assert_eq!(
        fixture
            .run_reference_prefix()
            .expect("run reference Stage 2 regular-batch input claims"),
        fixture.expected
    );
    assert_eq!(
        fixture
            .run_modular_prefix()
            .expect("run modular Stage 2 regular-batch input claims"),
        fixture.expected
    );

    let core = measure_samples(samples, || {
        let prefix = fixture
            .run_reference_prefix()
            .expect("run reference Stage 2 regular-batch input claims");
        let _ = black_box(prefix);
    });

    let modular = measure_samples(samples, || {
        let prefix = fixture
            .run_modular_prefix()
            .expect("run modular Stage 2 regular-batch input claims");
        let _ = black_box(prefix);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage2_regular_batch_input_claim_memory(fixture.config.log_t, fixture.config.log_k),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 2 regular-batch input-claim kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 2 regular-batch input-claim evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 2 regular-batch input-claim evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage3_regular_batch_input_claim_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage3_regular_batch_input_kernel_benchmark_fixture,
        validate_kernel_benchmark_evidence, FeatureMode, FixtureKind, FixtureRequest,
        KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_stage3_regular_batch_input_claims";
    const BENCHMARK: &str = "frontier_perf/stage3_regular_batch_inputs";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-SC-007", "OPT-EQ-004"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage3_regular_batch_input_kernel_benchmark_fixture(&request)
        .expect("load Stage 3 regular-batch input-claim kernel fixture");
    assert_eq!(fixture.run_reference_prefix(), fixture.expected);
    assert_eq!(
        fixture
            .run_modular_prefix()
            .expect("run modular Stage 3 regular-batch input claims"),
        fixture.expected
    );

    let core = measure_samples(samples, || {
        let prefix = fixture.run_reference_prefix();
        let _ = black_box(prefix);
    });

    let modular = measure_samples(samples, || {
        let prefix = fixture
            .run_modular_prefix()
            .expect("run modular Stage 3 regular-batch input claims");
        let _ = black_box(prefix);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage3_regular_batch_input_claim_memory(fixture.config.log_t),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 3 regular-batch input-claim kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 3 regular-batch input-claim evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 3 regular-batch input-claim evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage4_regular_batch_input_claim_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage4_regular_batch_input_kernel_benchmark_fixture,
        validate_kernel_benchmark_evidence, FeatureMode, FixtureKind, FixtureRequest,
        KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_stage4_regular_batch_input_claims";
    const BENCHMARK: &str = "frontier_perf/stage4_regular_batch_inputs";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-SC-007", "OPT-EQ-004"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage4_regular_batch_input_kernel_benchmark_fixture(&request)
        .expect("load Stage 4 regular-batch input-claim kernel fixture");
    assert_eq!(fixture.run_reference_prefix(), fixture.expected);
    assert_eq!(
        fixture
            .run_modular_prefix()
            .expect("run modular Stage 4 regular-batch input claims"),
        fixture.expected
    );

    let core = measure_samples(samples, || {
        let prefix = fixture.run_reference_prefix();
        let _ = black_box(prefix);
    });

    let modular = measure_samples(samples, || {
        let prefix = fixture
            .run_modular_prefix()
            .expect("run modular Stage 4 regular-batch input claims");
        let _ = black_box(prefix);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage4_regular_batch_input_claim_memory(fixture.config.log_t, fixture.config.log_k),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 4 regular-batch input-claim kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 4 regular-batch input-claim evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 4 regular-batch input-claim evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage4_regular_batch_sumcheck_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage4_regular_batch_sumcheck_kernel_benchmark_fixture,
        validate_kernel_benchmark_evidence, FeatureMode, FixtureKind, FixtureRequest,
        KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_stage4_regular_batch_sumcheck";
    const BENCHMARK: &str = "frontier_perf/stage4_regular_batch_sumcheck";
    const OPTIMIZATION_IDS: [&str; 14] = [
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

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage4_regular_batch_sumcheck_kernel_benchmark_fixture(&request)
        .expect("load Stage 4 regular-batch sumcheck kernel fixture");
    assert_eq!(
        fixture
            .run_reference_sumcheck()
            .expect("run reference Stage 4 regular-batch sumcheck"),
        fixture.expected.challenges.len() * 2
    );
    let modular_proof = fixture
        .run_modular_sumcheck()
        .expect("run modular Stage 4 regular-batch sumcheck");
    assert_eq!(modular_proof.proof, fixture.expected.proof);
    assert_eq!(modular_proof.challenges, fixture.expected.challenges);
    assert_eq!(
        modular_proof.batching_coefficients,
        fixture.expected.batching_coefficients
    );
    assert_eq!(modular_proof.output_claim, fixture.expected.output_claim);

    let core = measure_samples(samples, || {
        let rounds = fixture
            .run_reference_sumcheck()
            .expect("run reference Stage 4 regular-batch sumcheck");
        let _rounds = black_box(rounds);
    });

    let modular = measure_samples(samples, || {
        let proof = fixture
            .run_modular_sumcheck()
            .expect("run modular Stage 4 regular-batch sumcheck");
        let _proof = black_box(proof.output_claim);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage4_regular_batch_sumcheck_memory(fixture.config.log_t, fixture.config.log_k),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 4 regular-batch sumcheck kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 4 regular-batch sumcheck evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 4 regular-batch sumcheck evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage6_regular_batch_input_claim_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage6_regular_batch_input_kernel_benchmark_fixture,
        validate_kernel_benchmark_evidence, FeatureMode, FixtureKind, FixtureRequest,
        KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_stage6_regular_batch_input_claims";
    const BENCHMARK: &str = "frontier_perf/stage6_regular_batch_inputs";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-SC-007", "OPT-EQ-004"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage6_regular_batch_input_kernel_benchmark_fixture(&request)
        .expect("load Stage 6 regular-batch input-claim kernel fixture");
    assert_eq!(
        fixture
            .run_reference_prefix()
            .expect("run reference Stage 6 regular-batch input claims"),
        fixture.expected
    );
    assert_eq!(
        fixture
            .run_modular_prefix()
            .expect("run modular Stage 6 regular-batch input claims"),
        fixture.expected
    );

    let core = measure_samples(samples, || {
        let prefix = fixture
            .run_reference_prefix()
            .expect("run reference Stage 6 regular-batch input claims");
        let _ = black_box(prefix);
    });

    let modular = measure_samples(samples, || {
        let prefix = fixture
            .run_modular_prefix()
            .expect("run modular Stage 6 regular-batch input claims");
        let _ = black_box(prefix);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage6_regular_batch_input_claim_memory(fixture.config.log_t, fixture.config.log_k),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 6 regular-batch input-claim kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 6 regular-batch input-claim evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 6 regular-batch input-claim evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage6_regular_batch_sumcheck_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage6_regular_batch_sumcheck_kernel_benchmark_fixture,
        validate_kernel_benchmark_evidence, FeatureMode, FixtureKind, FixtureRequest,
        KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_stage6_regular_batch_sumcheck";
    const BENCHMARK: &str = "frontier_perf/stage6_regular_batch_sumcheck";
    const OPTIMIZATION_IDS: [&str; 9] = [
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

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage6_regular_batch_sumcheck_kernel_benchmark_fixture(&request)
        .expect("load Stage 6 regular-batch sumcheck kernel fixture");
    assert_eq!(
        fixture
            .run_reference_sumcheck()
            .expect("run reference Stage 6 regular-batch sumcheck"),
        fixture.expected.challenges.len() * 2
    );
    let modular_proof = fixture
        .run_modular_sumcheck()
        .expect("run modular Stage 6 regular-batch sumcheck");
    assert_eq!(modular_proof.proof, fixture.expected.proof);
    assert_eq!(modular_proof.sumcheck_point, fixture.expected.challenges);
    assert_eq!(
        modular_proof.batching_coefficients,
        fixture.expected.batching_coefficients
    );
    assert_eq!(
        modular_proof.sumcheck_final_claim,
        fixture.expected.output_claim
    );

    let core = measure_samples(samples, || {
        let rounds = fixture
            .run_reference_sumcheck()
            .expect("run reference Stage 6 regular-batch sumcheck");
        let _rounds = black_box(rounds);
    });

    let modular = measure_samples(samples, || {
        let proof = fixture
            .run_modular_sumcheck()
            .expect("run modular Stage 6 regular-batch sumcheck");
        let _proof = black_box(proof.sumcheck_final_claim);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage6_regular_batch_sumcheck_memory(fixture.config.log_t, fixture.config.log_k),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 6 regular-batch sumcheck kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 6 regular-batch sumcheck evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 6 regular-batch sumcheck evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage4_field_inline_registers_read_write_kernel_evidence() {
    use jolt_backends::{
        cpu::read_write_matrix::FieldRegistersReadWriteState, SumcheckFieldRegisterRead,
        SumcheckFieldRegisterWrite, SumcheckFieldRegistersReadWriteRow,
        SumcheckFieldRegistersReadWriteStateRequest, SumcheckRegistersReadWriteOutput,
    };
    use jolt_claims::protocols::{field_inline::FieldInlineConfig, jolt::JoltOneHotConfig};
    use jolt_field::{Fr, FromPrimitiveInt, Invertible};
    use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
    use jolt_prover_harness::{
        trace_sdk_guest, validate_kernel_benchmark_evidence, KnownOptimizationIds,
        SdkGuestTraceRequest,
    };
    use jolt_witness::protocols::jolt_vm::{
        field_inline::{
            FieldInlineRegisterReadWriteRow, FieldInlineRegisterReadWriteRows,
            TraceBackedFieldInlineWitness,
        },
        JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackedJoltVmWitness,
    };
    use rayon::prelude::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct FieldReadWriteRun {
        final_claim: Fr,
        output: SumcheckRegistersReadWriteOutput<Fr>,
    }

    fn deterministic_point(len: usize, seed: u64) -> Vec<Fr> {
        (0..len)
            .map(|index| Fr::from_u64(seed + (index as u64) * 17))
            .collect()
    }

    fn backend_field_row(
        row: FieldInlineRegisterReadWriteRow<Fr>,
    ) -> SumcheckFieldRegistersReadWriteRow<Fr> {
        SumcheckFieldRegistersReadWriteRow {
            rs1: row.rs1.map(|read| SumcheckFieldRegisterRead {
                register: read.register,
                value: read.value,
            }),
            rs2: row.rs2.map(|read| SumcheckFieldRegisterRead {
                register: read.register,
                value: read.value,
            }),
            rd: row.rd.map(|write| SumcheckFieldRegisterWrite {
                register: write.register,
                pre_value: write.pre_value,
                post_value: write.post_value,
            }),
            rd_increment: row.rd_increment,
        }
    }

    fn real_field_inline_rows() -> (Vec<SumcheckFieldRegistersReadWriteRow<Fr>>, usize) {
        let inputs = postcard::to_stdvec(&7u32).expect("serialize field-inline benchmark input");
        let fixture = trace_sdk_guest(
            SdkGuestTraceRequest::new("field-inline-eq-poly-guest", inputs)
                .with_field_inline(true)
                .with_max_padded_trace_length(1 << 9),
        )
        .expect("trace real field-inline eq-poly guest");
        let log_t = fixture.padded_trace_length.trailing_zeros() as usize;
        let witness_config = JoltVmWitnessConfig::new(
            log_t,
            1,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        );
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&fixture.program, &fixture.preprocessing, fixture.trace),
        );
        let field_witness = witness
            .field_inline_witness()
            .expect("build field-inline witness");
        let rows = <TraceBackedFieldInlineWitness<'_, '_, _> as FieldInlineRegisterReadWriteRows<
            Fr,
        >>::field_inline_register_read_write_rows(&field_witness)
        .expect("materialize field-inline register rows")
        .into_iter()
        .map(backend_field_row)
        .collect::<Vec<_>>();
        (rows, log_t)
    }

    fn field_registers_read_write_input_claim(
        rows: &[SumcheckFieldRegistersReadWriteRow<Fr>],
        r_cycle: &[Fr],
        gamma: Fr,
    ) -> Fr {
        let eq_cycle = jolt_poly::EqPolynomial::<Fr>::evals(r_cycle, None);
        rows.iter()
            .zip(eq_cycle)
            .map(|(row, eq)| {
                let rd = row
                    .rd
                    .map_or_else(|| Fr::from_u64(0), |write| write.post_value);
                let rs1 = row.rs1.map_or_else(|| Fr::from_u64(0), |read| read.value);
                let rs2 = row.rs2.map_or_else(|| Fr::from_u64(0), |read| read.value);
                eq * (rd + gamma * rs1 + gamma * gamma * rs2)
            })
            .sum()
    }

    fn poly_evals_at_0_2_3(polynomial: &Polynomial<Fr>, index: usize) -> [Fr; 3] {
        let (lo, hi) = polynomial.sumcheck_eval_pair(index, BindingOrder::LowToHigh);
        let step = hi - lo;
        [lo, hi + step, hi + step + step]
    }

    fn dense_factor_polynomials(
        rows: &[SumcheckFieldRegistersReadWriteRow<Fr>],
        r_cycle: &[Fr],
        gamma: Fr,
        log_k: usize,
    ) -> (
        Polynomial<Fr>,
        Polynomial<Fr>,
        Polynomial<Fr>,
        Polynomial<Fr>,
        Polynomial<Fr>,
    ) {
        let cycles = rows.len();
        let register_count = 1usize << log_k;
        let eq_cycle = jolt_poly::EqPolynomial::<Fr>::evals(r_cycle, None);
        let len = cycles * register_count;
        let mut eq = jolt_poly::thread::unsafe_allocate_zero_vec(len);
        let mut inc = jolt_poly::thread::unsafe_allocate_zero_vec(len);
        let mut ra = jolt_poly::thread::unsafe_allocate_zero_vec(len);
        let mut wa = jolt_poly::thread::unsafe_allocate_zero_vec(len);
        let mut val = jolt_poly::thread::unsafe_allocate_zero_vec(len);
        let mut registers = vec![Fr::from_u64(0); register_count];
        let gamma2 = gamma * gamma;

        for (cycle, row) in rows.iter().enumerate() {
            for register in 0..register_count {
                let index = register * cycles + cycle;
                eq[index] = eq_cycle[cycle];
                inc[index] = row.rd_increment;
                val[index] = registers[register];
            }
            if let Some(read) = row.rs1 {
                ra[usize::from(read.register) * cycles + cycle] += gamma;
            }
            if let Some(read) = row.rs2 {
                ra[usize::from(read.register) * cycles + cycle] += gamma2;
            }
            if let Some(write) = row.rd {
                wa[usize::from(write.register) * cycles + cycle] = Fr::from_u64(1);
                registers[usize::from(write.register)] = write.post_value;
            }
        }

        (
            Polynomial::new(eq),
            Polynomial::new(inc),
            Polynomial::new(ra),
            Polynomial::new(wa),
            Polynomial::new(val),
        )
    }

    fn compute_rs2_ra_claim(
        rows: &[SumcheckFieldRegistersReadWriteRow<Fr>],
        r_address: &[Fr],
        r_cycle: &[Fr],
    ) -> Fr {
        let eq_address = jolt_poly::EqPolynomial::<Fr>::evals(r_address, None);
        let eq_cycle = jolt_poly::EqPolynomial::<Fr>::evals(r_cycle, None);
        rows.par_iter()
            .zip(eq_cycle.par_iter())
            .filter_map(|(row, &eq)| {
                row.rs2
                    .map(|read| eq * eq_address[usize::from(read.register)])
            })
            .sum()
    }

    fn run_dense_field_registers_read_write_sumcheck(
        request: &SumcheckFieldRegistersReadWriteStateRequest<Fr>,
        challenges: &[Fr],
        r_address: &[Fr],
        r_cycle: &[Fr],
    ) -> FieldReadWriteRun {
        let (mut eq, mut inc, mut ra, mut wa, mut val) = dense_factor_polynomials(
            &request.rows,
            &request.r_cycle,
            request.gamma,
            request.log_k,
        );
        let mut claim = request.input_claim;
        for &challenge in challenges {
            let evals = (0..eq.len() / 2)
                .into_par_iter()
                .map(|index| {
                    let eq = poly_evals_at_0_2_3(&eq, index);
                    let inc = poly_evals_at_0_2_3(&inc, index);
                    let ra = poly_evals_at_0_2_3(&ra, index);
                    let wa = poly_evals_at_0_2_3(&wa, index);
                    let val = poly_evals_at_0_2_3(&val, index);
                    std::array::from_fn(|point| {
                        eq[point] * (ra[point] * val[point] + wa[point] * (val[point] + inc[point]))
                    })
                })
                .reduce(
                    || [Fr::from_u64(0); 3],
                    |left, right| std::array::from_fn(|index| left[index] + right[index]),
                );
            let round = UnivariatePoly::from_evals_and_hint(claim, &evals);
            claim = round.evaluate(challenge);
            eq.bind_with_order(challenge, BindingOrder::LowToHigh);
            inc.bind_with_order(challenge, BindingOrder::LowToHigh);
            ra.bind_with_order(challenge, BindingOrder::LowToHigh);
            wa.bind_with_order(challenge, BindingOrder::LowToHigh);
            val.bind_with_order(challenge, BindingOrder::LowToHigh);
        }

        let combined_ra = ra.evaluations()[0];
        let rs2_ra = compute_rs2_ra_claim(&request.rows, r_address, r_cycle);
        let rs1_ra = (combined_ra - request.gamma * request.gamma * rs2_ra)
            * request
                .gamma
                .inverse()
                .expect("field-register read-write gamma should be invertible");
        FieldReadWriteRun {
            final_claim: claim,
            output: SumcheckRegistersReadWriteOutput {
                registers_val: val.evaluations()[0],
                rs1_ra,
                rs2_ra,
                rd_wa: wa.evaluations()[0],
                rd_inc: inc.evaluations()[0],
            },
        }
    }

    fn run_modular_field_registers_read_write_sumcheck(
        request: &SumcheckFieldRegistersReadWriteStateRequest<Fr>,
        challenges: &[Fr],
        opening_point: &[Fr],
    ) -> FieldReadWriteRun {
        let mut state =
            FieldRegistersReadWriteState::new("cpu", "bench.field_inline_read_write", request)
                .expect("materialize field-inline read-write state");
        let mut claim = request.input_claim;
        for &challenge in challenges {
            let round = state
                .evaluate_round("cpu", "bench.field_inline_read_write.round", claim)
                .expect("evaluate field-inline read-write round");
            claim = round.evaluate(challenge);
            state
                .bind("cpu", "bench.field_inline_read_write.bind", challenge)
                .expect("bind field-inline read-write state");
        }
        FieldReadWriteRun {
            final_claim: claim,
            output: state
                .output_claims(opening_point)
                .expect("field-inline read-write output claims"),
        }
    }

    const KERNEL: &str = "cpu_field_inline_stage4_registers_read_write";
    const BENCHMARK: &str = "frontier_perf/stage4_field_inline_registers_read_write";
    const OPTIMIZATION_IDS: [&str; 12] = [
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
    ];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let (rows, log_t) = real_field_inline_rows();
    let config = FieldInlineConfig::native_v1();
    let dimensions = config.read_write_dimensions(log_t);
    let log_k = dimensions.log_k();
    let fixed_cycle = deterministic_point(log_t, 29);
    let gamma = Fr::from_u64(47);
    let input_claim = field_registers_read_write_input_claim(&rows, &fixed_cycle, gamma);
    let challenges = deterministic_point(log_t + log_k, 101);
    let opening = dimensions
        .read_write_opening_point(&challenges)
        .expect("derive field-register read-write opening point");
    let request = SumcheckFieldRegistersReadWriteStateRequest::new(
        "bench.field_inline_read_write",
        rows,
        fixed_cycle,
        gamma,
        input_claim,
        log_t,
        log_k,
        dimensions.phase1_num_rounds(),
        dimensions.phase2_num_rounds(),
    );

    let expected = run_dense_field_registers_read_write_sumcheck(
        &request,
        &challenges,
        &opening.r_address,
        &opening.r_cycle,
    );
    let actual = run_modular_field_registers_read_write_sumcheck(
        &request,
        &challenges,
        &opening.opening_point,
    );
    assert_eq!(actual, expected);

    let core = measure_samples(samples, || {
        let output = run_dense_field_registers_read_write_sumcheck(
            &request,
            &challenges,
            &opening.r_address,
            &opening.r_cycle,
        );
        let _ = black_box(output);
    });

    let modular = measure_samples(samples, || {
        let output = run_modular_field_registers_read_write_sumcheck(
            &request,
            &challenges,
            &opening.opening_point,
        );
        let _ = black_box(output);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage4_field_inline_read_write_memory(
            request.rows.len(),
            request.log_t,
            request.log_k,
        ),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered field-inline Stage 4 read-write kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("field-inline Stage 4 read-write evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical field-inline Stage 4 read-write evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage5_field_inline_registers_val_evaluation_kernel_evidence() {
    use jolt_backends::{
        cpu::read_write_matrix::FieldRegistersValEvaluationState, SumcheckFieldRegisterRead,
        SumcheckFieldRegisterWrite, SumcheckFieldRegistersReadWriteRow,
        SumcheckFieldRegistersValEvaluationStateRequest,
    };
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
    use jolt_prover_harness::{
        trace_sdk_guest, validate_kernel_benchmark_evidence, KnownOptimizationIds,
        SdkGuestTraceRequest,
    };
    use jolt_witness::protocols::jolt_vm::{
        field_inline::{
            FieldInlineRegisterReadWriteRow, FieldInlineRegisterReadWriteRows,
            TraceBackedFieldInlineWitness,
        },
        JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackedJoltVmWitness,
    };
    use rayon::prelude::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct FieldValRun {
        final_claim: Fr,
        field_rd_inc: Fr,
        field_rd_wa: Fr,
    }

    fn deterministic_point(len: usize, seed: u64) -> Vec<Fr> {
        (0..len)
            .map(|index| Fr::from_u64(seed + (index as u64) * 17))
            .collect()
    }

    fn backend_field_row(
        row: FieldInlineRegisterReadWriteRow<Fr>,
    ) -> SumcheckFieldRegistersReadWriteRow<Fr> {
        SumcheckFieldRegistersReadWriteRow {
            rs1: row.rs1.map(|read| SumcheckFieldRegisterRead {
                register: read.register,
                value: read.value,
            }),
            rs2: row.rs2.map(|read| SumcheckFieldRegisterRead {
                register: read.register,
                value: read.value,
            }),
            rd: row.rd.map(|write| SumcheckFieldRegisterWrite {
                register: write.register,
                pre_value: write.pre_value,
                post_value: write.post_value,
            }),
            rd_increment: row.rd_increment,
        }
    }

    fn real_field_inline_rows() -> (Vec<SumcheckFieldRegistersReadWriteRow<Fr>>, usize) {
        let inputs = postcard::to_stdvec(&7u32).expect("serialize field-inline benchmark input");
        let fixture = trace_sdk_guest(
            SdkGuestTraceRequest::new("field-inline-eq-poly-guest", inputs)
                .with_field_inline(true)
                .with_max_padded_trace_length(1 << 9),
        )
        .expect("trace real field-inline eq-poly guest");
        let log_t = fixture.padded_trace_length.trailing_zeros() as usize;
        let witness_config = JoltVmWitnessConfig::new(
            log_t,
            1,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        );
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&fixture.program, &fixture.preprocessing, fixture.trace),
        );
        let field_witness = witness
            .field_inline_witness()
            .expect("build field-inline witness");
        let rows = <TraceBackedFieldInlineWitness<'_, '_, _> as FieldInlineRegisterReadWriteRows<
            Fr,
        >>::field_inline_register_read_write_rows(&field_witness)
        .expect("materialize field-inline register rows")
        .into_iter()
        .map(backend_field_row)
        .collect::<Vec<_>>();
        (rows, log_t)
    }

    fn field_registers_val_input_claim(
        rows: &[SumcheckFieldRegistersReadWriteRow<Fr>],
        r_address: &[Fr],
        r_cycle: &[Fr],
    ) -> Fr {
        let eq_address = jolt_poly::EqPolynomial::<Fr>::evals(r_address, None);
        let eq_cycle = jolt_poly::EqPolynomial::<Fr>::evals(r_cycle, None);
        let mut state = vec![Fr::from_u64(0); eq_address.len()];
        let mut claim = Fr::from_u64(0);
        for (cycle, row) in rows.iter().enumerate() {
            for (register, value) in state.iter().enumerate() {
                claim += eq_address[register] * eq_cycle[cycle] * *value;
            }
            if let Some(write) = row.rd {
                state[usize::from(write.register)] = write.post_value;
            }
        }
        claim
    }

    fn poly_evals_at_0_2_3(polynomial: &Polynomial<Fr>, index: usize) -> [Fr; 3] {
        let (lo, hi) = polynomial.sumcheck_eval_pair(index, BindingOrder::LowToHigh);
        let step = hi - lo;
        [lo, hi + step, hi + step + step]
    }

    fn run_dense_field_inline_val_sumcheck(
        request: &SumcheckFieldRegistersValEvaluationStateRequest<Fr>,
        challenges: &[Fr],
    ) -> FieldValRun {
        let register_count = 1usize << request.log_k;
        let eq_register = jolt_poly::EqPolynomial::<Fr>::evals(&request.r_address, None);
        let mut inc = Polynomial::new(
            request
                .rows
                .iter()
                .map(|row| row.rd_increment)
                .collect::<Vec<_>>(),
        );
        let mut wa = Polynomial::new(
            request
                .rows
                .iter()
                .map(|row| {
                    row.rd.map_or_else(
                        || Fr::from_u64(0),
                        |write| {
                            let register = usize::from(write.register);
                            assert!(register < register_count);
                            eq_register[register]
                        },
                    )
                })
                .collect::<Vec<_>>(),
        );
        let mut lt = Polynomial::new(jolt_poly::LtPolynomial::<Fr>::evaluations(&request.r_cycle));
        let mut claim = request.input_claim;
        for &challenge in challenges {
            let evals = (0..inc.len() / 2)
                .into_par_iter()
                .map(|index| {
                    let inc = poly_evals_at_0_2_3(&inc, index);
                    let wa = poly_evals_at_0_2_3(&wa, index);
                    let lt = poly_evals_at_0_2_3(&lt, index);
                    std::array::from_fn(|point| inc[point] * wa[point] * lt[point])
                })
                .reduce(
                    || [Fr::from_u64(0); 3],
                    |left, right| std::array::from_fn(|index| left[index] + right[index]),
                );
            let round = UnivariatePoly::from_evals_and_hint(claim, &evals);
            claim = round.evaluate(challenge);
            inc.bind_with_order(challenge, BindingOrder::LowToHigh);
            wa.bind_with_order(challenge, BindingOrder::LowToHigh);
            lt.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        FieldValRun {
            final_claim: claim,
            field_rd_inc: inc.evaluations()[0],
            field_rd_wa: wa.evaluations()[0],
        }
    }

    fn run_modular_field_inline_val_sumcheck(
        request: &SumcheckFieldRegistersValEvaluationStateRequest<Fr>,
        challenges: &[Fr],
    ) -> FieldValRun {
        let mut state =
            FieldRegistersValEvaluationState::new("cpu", "bench.field_inline_val", request)
                .expect("materialize field-inline value-evaluation state");
        let mut claim = request.input_claim;
        for &challenge in challenges {
            let round = state
                .evaluate_round("cpu", "bench.field_inline_val.round", claim)
                .expect("evaluate field-inline value-evaluation round");
            claim = round.evaluate(challenge);
            state
                .bind("cpu", "bench.field_inline_val.bind", challenge)
                .expect("bind field-inline value-evaluation state");
        }
        let output = state
            .output_claims()
            .expect("field-inline value-evaluation output claims");
        FieldValRun {
            final_claim: claim,
            field_rd_inc: output.field_rd_inc,
            field_rd_wa: output.field_rd_wa,
        }
    }

    const KERNEL: &str = "cpu_field_inline_stage5_registers_val_evaluation";
    const BENCHMARK: &str = "frontier_perf/stage5_field_inline_registers_val_evaluation";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-FLD-003", "OPT-REL-010"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let (rows, log_t) = real_field_inline_rows();
    let log_k = 4;
    let r_address = deterministic_point(log_k, 11);
    let r_cycle = deterministic_point(log_t, 29);
    let input_claim = field_registers_val_input_claim(&rows, &r_address, &r_cycle);
    let challenges = deterministic_point(log_t, 101);
    let request = SumcheckFieldRegistersValEvaluationStateRequest::new(
        "bench.field_inline_val",
        rows,
        r_address,
        r_cycle,
        input_claim,
        log_t,
        log_k,
    );

    let expected = run_dense_field_inline_val_sumcheck(&request, &challenges);
    let actual = run_modular_field_inline_val_sumcheck(&request, &challenges);
    assert_eq!(actual, expected);

    let core = measure_samples(samples, || {
        let output = run_dense_field_inline_val_sumcheck(&request, &challenges);
        let _ = black_box(output);
    });

    let modular = measure_samples(samples, || {
        let output = run_modular_field_inline_val_sumcheck(&request, &challenges);
        let _ = black_box(output);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage5_field_inline_val_evaluation_memory(request.rows.len(), log_t, log_k),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered field-inline Stage 5 value-evaluation kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect(
                "field-inline Stage 5 value-evaluation evidence should pass the canonical gate",
            );
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical field-inline Stage 5 value-evaluation evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage6_field_inline_registers_inc_claim_reduction_kernel_evidence() {
    use jolt_backends::{
        cpu::read_write_matrix::FieldRegistersIncClaimReductionState, SumcheckFieldRegisterRead,
        SumcheckFieldRegisterWrite, SumcheckFieldRegistersIncClaimReductionStateRequest,
        SumcheckFieldRegistersReadWriteRow,
    };
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
    use jolt_prover_harness::{
        trace_sdk_guest, validate_kernel_benchmark_evidence, KnownOptimizationIds,
        SdkGuestTraceRequest,
    };
    use jolt_witness::protocols::jolt_vm::{
        field_inline::{
            FieldInlineRegisterReadWriteRow, FieldInlineRegisterReadWriteRows,
            TraceBackedFieldInlineWitness,
        },
        JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackedJoltVmWitness,
    };
    use rayon::prelude::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct FieldIncRun {
        final_claim: Fr,
        field_rd_inc: Fr,
    }

    fn deterministic_point(log_t: usize, seed: u64) -> Vec<Fr> {
        (0..log_t)
            .map(|index| Fr::from_u64(seed + (index as u64) * 17))
            .collect()
    }

    fn backend_field_row(
        row: FieldInlineRegisterReadWriteRow<Fr>,
    ) -> SumcheckFieldRegistersReadWriteRow<Fr> {
        SumcheckFieldRegistersReadWriteRow {
            rs1: row.rs1.map(|read| SumcheckFieldRegisterRead {
                register: read.register,
                value: read.value,
            }),
            rs2: row.rs2.map(|read| SumcheckFieldRegisterRead {
                register: read.register,
                value: read.value,
            }),
            rd: row.rd.map(|write| SumcheckFieldRegisterWrite {
                register: write.register,
                pre_value: write.pre_value,
                post_value: write.post_value,
            }),
            rd_increment: row.rd_increment,
        }
    }

    fn real_field_inline_rows() -> (Vec<SumcheckFieldRegistersReadWriteRow<Fr>>, usize) {
        let inputs = postcard::to_stdvec(&7u32).expect("serialize field-inline benchmark input");
        let fixture = trace_sdk_guest(
            SdkGuestTraceRequest::new("field-inline-eq-poly-guest", inputs)
                .with_field_inline(true)
                .with_max_padded_trace_length(1 << 9),
        )
        .expect("trace real field-inline eq-poly guest");
        let log_t = fixture.padded_trace_length.trailing_zeros() as usize;
        let witness_config = JoltVmWitnessConfig::new(
            log_t,
            1,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        );
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&fixture.program, &fixture.preprocessing, fixture.trace),
        );
        let field_witness = witness
            .field_inline_witness()
            .expect("build field-inline witness");
        let rows = <TraceBackedFieldInlineWitness<'_, '_, _> as FieldInlineRegisterReadWriteRows<
            Fr,
        >>::field_inline_register_read_write_rows(&field_witness)
        .expect("materialize field-inline register rows")
        .into_iter()
        .map(backend_field_row)
        .collect::<Vec<_>>();
        (rows, log_t)
    }

    fn reverse_cycle_table(values: Vec<Fr>, log_t: usize) -> Vec<Fr> {
        let mut reversed = jolt_poly::thread::unsafe_allocate_zero_vec(values.len());
        for (cycle, value) in values.into_iter().enumerate() {
            reversed[cycle.reverse_bits() >> (usize::BITS as usize - log_t)] = value;
        }
        reversed
    }

    fn field_inline_inc_input_claim(
        rows: &[SumcheckFieldRegistersReadWriteRow<Fr>],
        r_cycle_read_write: &[Fr],
        r_cycle_val_evaluation: &[Fr],
        gamma: Fr,
        log_t: usize,
    ) -> Fr {
        let field_rd_inc = reverse_cycle_table(
            rows.iter().map(|row| row.rd_increment).collect::<Vec<_>>(),
            log_t,
        );
        let eq_read_write = jolt_poly::EqPolynomial::<Fr>::evals(r_cycle_read_write, None);
        let eq_val_evaluation = jolt_poly::EqPolynomial::<Fr>::evals(r_cycle_val_evaluation, None);
        field_rd_inc
            .into_iter()
            .zip(eq_read_write.into_iter().zip(eq_val_evaluation))
            .map(|(inc, (read_write, val_evaluation))| inc * (read_write + gamma * val_evaluation))
            .sum()
    }

    fn poly_evals_at_0_2_3(polynomial: &Polynomial<Fr>, index: usize) -> [Fr; 3] {
        let (lo, hi) = polynomial.sumcheck_eval_pair(index, BindingOrder::HighToLow);
        let step = hi - lo;
        [lo, hi + step, hi + step + step]
    }

    fn run_dense_field_inline_inc_sumcheck(
        request: &SumcheckFieldRegistersIncClaimReductionStateRequest<Fr>,
        challenges: &[Fr],
    ) -> FieldIncRun {
        let mut field_rd_inc = Polynomial::new(reverse_cycle_table(
            request
                .rows
                .iter()
                .map(|row| row.rd_increment)
                .collect::<Vec<_>>(),
            request.log_t,
        ));
        let eq_read_write = jolt_poly::EqPolynomial::<Fr>::evals(&request.r_cycle_read_write, None);
        let eq_val_evaluation =
            jolt_poly::EqPolynomial::<Fr>::evals(&request.r_cycle_val_evaluation, None);
        let mut coeff = Polynomial::new(
            eq_read_write
                .into_iter()
                .zip(eq_val_evaluation)
                .map(|(read_write, val_evaluation)| read_write + request.gamma * val_evaluation)
                .collect::<Vec<_>>(),
        );
        let mut claim = request.input_claim;
        for &challenge in challenges {
            let evals = (0..field_rd_inc.len() / 2)
                .into_par_iter()
                .map(|index| {
                    let inc = poly_evals_at_0_2_3(&field_rd_inc, index);
                    let coeff = poly_evals_at_0_2_3(&coeff, index);
                    std::array::from_fn(|point| inc[point] * coeff[point])
                })
                .reduce(
                    || [Fr::from_u64(0); 3],
                    |left, right| std::array::from_fn(|index| left[index] + right[index]),
                );
            let round = UnivariatePoly::from_evals_and_hint(claim, &evals);
            claim = round.evaluate(challenge);
            field_rd_inc.bind_with_order(challenge, BindingOrder::HighToLow);
            coeff.bind_with_order(challenge, BindingOrder::HighToLow);
        }
        FieldIncRun {
            final_claim: claim,
            field_rd_inc: field_rd_inc.evaluations()[0],
        }
    }

    fn run_modular_field_inline_inc_sumcheck(
        request: &SumcheckFieldRegistersIncClaimReductionStateRequest<Fr>,
        challenges: &[Fr],
    ) -> FieldIncRun {
        let mut state =
            FieldRegistersIncClaimReductionState::new("cpu", "bench.field_inline_inc", request)
                .expect("materialize field-inline increment claim-reduction state");
        let mut claim = request.input_claim;
        for &challenge in challenges {
            let round = state
                .evaluate_round("cpu", "bench.field_inline_inc.round", claim)
                .expect("evaluate field-inline increment claim-reduction round");
            claim = round.evaluate(challenge);
            state
                .bind("cpu", "bench.field_inline_inc.bind", challenge)
                .expect("bind field-inline increment claim-reduction state");
        }
        FieldIncRun {
            final_claim: claim,
            field_rd_inc: state
                .output_claims()
                .expect("field-inline increment output claims")
                .field_rd_inc,
        }
    }

    const KERNEL: &str = "cpu_field_inline_stage6_registers_inc_claim_reduction";
    const BENCHMARK: &str = "frontier_perf/stage6_field_inline_registers_inc_claim_reduction";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-FLD-003", "OPT-REL-011"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let (rows, log_t) = real_field_inline_rows();
    let r_cycle_read_write = deterministic_point(log_t, 11);
    let r_cycle_val_evaluation = deterministic_point(log_t, 29);
    let gamma = Fr::from_u64(47);
    let input_claim = field_inline_inc_input_claim(
        &rows,
        &r_cycle_read_write,
        &r_cycle_val_evaluation,
        gamma,
        log_t,
    );
    let challenges = deterministic_point(log_t, 101);
    let request = SumcheckFieldRegistersIncClaimReductionStateRequest::new(
        "bench.field_inline_inc",
        rows,
        r_cycle_read_write,
        r_cycle_val_evaluation,
        gamma,
        input_claim,
        log_t,
    );

    let expected = run_dense_field_inline_inc_sumcheck(&request, &challenges);
    let actual = run_modular_field_inline_inc_sumcheck(&request, &challenges);
    assert_eq!(actual, expected);

    let core = measure_samples(samples, || {
        let output = run_dense_field_inline_inc_sumcheck(&request, &challenges);
        let _ = black_box(output);
    });

    let modular = measure_samples(samples, || {
        let output = run_modular_field_inline_inc_sumcheck(&request, &challenges);
        let _ = black_box(output);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage6_field_inline_inc_claim_reduction_memory(request.rows.len(), request.log_t),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered field-inline Stage 6 increment kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("field-inline Stage 6 increment evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical field-inline Stage 6 increment evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage7_regular_batch_input_claim_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage7_regular_batch_input_kernel_benchmark_fixture,
        validate_kernel_benchmark_evidence, FeatureMode, FixtureKind, FixtureRequest,
        KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_stage7_regular_batch_input_claims";
    const BENCHMARK: &str = "frontier_perf/stage7_regular_batch_inputs";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-SC-007", "OPT-EQ-004"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage7_regular_batch_input_kernel_benchmark_fixture(&request)
        .expect("load Stage 7 regular-batch input-claim kernel fixture");
    assert_eq!(fixture.run_reference_prefix(), fixture.expected);
    assert_eq!(
        fixture
            .run_modular_prefix()
            .expect("run modular Stage 7 regular-batch input claims"),
        fixture.expected
    );

    let core = measure_samples(samples, || {
        let prefix = fixture.run_reference_prefix();
        let _ = black_box(prefix);
    });

    let modular = measure_samples(samples, || {
        let prefix = fixture
            .run_modular_prefix()
            .expect("run modular Stage 7 regular-batch input claims");
        let _ = black_box(prefix);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage7_regular_batch_input_claim_memory(
            fixture.config.log_t,
            fixture.config.hamming_dimensions.log_k_chunk,
        ),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 7 regular-batch input-claim kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 7 regular-batch input-claim evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 7 regular-batch input-claim evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage7_regular_batch_sumcheck_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage7_regular_batch_sumcheck_kernel_benchmark_fixture,
        validate_kernel_benchmark_evidence, FeatureMode, FixtureKind, FixtureRequest,
        KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_stage7_regular_batch_sumcheck";
    const BENCHMARK: &str = "frontier_perf/stage7_regular_batch_sumcheck";
    const OPTIMIZATION_IDS: [&str; 6] = [
        "OPT-SC-007",
        "OPT-EQ-004",
        "OPT-RA-003",
        "OPT-RA-007",
        "OPT-RA-008",
        "OPT-REL-013",
    ];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::AdviceConsumer, FeatureMode::Transparent);
    let fixture = load_stage7_regular_batch_sumcheck_kernel_benchmark_fixture(&request)
        .expect("load Stage 7 regular-batch sumcheck kernel fixture");
    assert_eq!(
        fixture
            .run_reference_sumcheck()
            .expect("run reference Stage 7 regular-batch sumcheck"),
        fixture.expected.challenges.len() * 2
    );
    let modular_proof = fixture
        .run_modular_sumcheck()
        .expect("run modular Stage 7 regular-batch sumcheck");
    assert_eq!(modular_proof.stage7_sumcheck_proof, fixture.expected.proof);
    assert_eq!(
        modular_proof
            .verifier_output
            .batch
            .hamming_weight_claim_reduction
            .sumcheck_point,
        fixture.expected.challenges
    );
    assert_eq!(
        modular_proof.verifier_output.public.batching_coefficients,
        fixture.expected.batching_coefficients
    );
    assert_eq!(
        modular_proof.verifier_output.batch.sumcheck_final_claim,
        fixture.expected.output_claim
    );

    let core = measure_samples(samples, || {
        let rounds = fixture
            .run_reference_sumcheck()
            .expect("run reference Stage 7 regular-batch sumcheck");
        let _rounds = black_box(rounds);
    });

    let modular = measure_samples(samples, || {
        let proof = fixture
            .run_modular_sumcheck()
            .expect("run modular Stage 7 regular-batch sumcheck");
        let _proof = black_box(proof.verifier_output.batch.sumcheck_final_claim);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage7_regular_batch_sumcheck_memory(
            fixture.config.log_t,
            fixture.config.hamming_dimensions.log_k_chunk,
            fixture.config.hamming_dimensions.layout.total(),
        ),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 7 regular-batch sumcheck kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 7 regular-batch sumcheck evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 7 regular-batch sumcheck evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage5_regular_batch_input_claim_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage5_regular_batch_input_kernel_benchmark_fixture,
        validate_kernel_benchmark_evidence, FeatureMode, FixtureKind, FixtureRequest,
        KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_stage5_regular_batch_input_claims";
    const BENCHMARK: &str = "frontier_perf/stage5_regular_batch_inputs";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-SC-007", "OPT-EQ-004"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage5_regular_batch_input_kernel_benchmark_fixture(&request)
        .expect("load Stage 5 regular-batch input-claim kernel fixture");
    assert_eq!(fixture.run_reference_prefix(), fixture.expected);
    assert_eq!(
        fixture
            .run_modular_prefix()
            .expect("run modular Stage 5 regular-batch input claims"),
        fixture.expected
    );

    let core = measure_samples(samples, || {
        let prefix = fixture.run_reference_prefix();
        let _ = black_box(prefix);
    });

    let modular = measure_samples(samples, || {
        let prefix = fixture
            .run_modular_prefix()
            .expect("run modular Stage 5 regular-batch input claims");
        let _ = black_box(prefix);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage5_regular_batch_input_claim_memory(fixture.config.log_t, fixture.config.log_k),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 5 regular-batch input-claim kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 5 regular-batch input-claim evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 5 regular-batch input-claim evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage5_regular_batch_sumcheck_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage5_regular_batch_sumcheck_kernel_benchmark_fixture,
        validate_kernel_benchmark_evidence, FeatureMode, FixtureKind, FixtureRequest,
        KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_stage5_regular_batch_sumcheck";
    const BENCHMARK: &str = "frontier_perf/stage5_regular_batch_sumcheck";
    const OPTIMIZATION_IDS: [&str; 6] = [
        "OPT-SC-007",
        "OPT-EQ-004",
        "OPT-REL-001",
        "OPT-REL-002",
        "OPT-REL-003",
        "OPT-REL-010",
    ];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage5_regular_batch_sumcheck_kernel_benchmark_fixture(&request)
        .expect("load Stage 5 regular-batch sumcheck kernel fixture");
    assert_eq!(
        fixture
            .run_reference_sumcheck()
            .expect("run reference Stage 5 regular-batch sumcheck"),
        fixture.expected.challenges.len() * 2
    );
    let modular_proof = fixture
        .run_modular_sumcheck()
        .expect("run modular Stage 5 regular-batch sumcheck");
    assert_eq!(modular_proof.proof, fixture.expected.proof);
    assert_eq!(modular_proof.challenges, fixture.expected.challenges);
    assert_eq!(
        modular_proof.batching_coefficients,
        fixture.expected.batching_coefficients
    );
    assert_eq!(modular_proof.output_claim, fixture.expected.output_claim);

    let core = measure_samples(samples, || {
        let rounds = fixture
            .run_reference_sumcheck()
            .expect("run reference Stage 5 regular-batch sumcheck");
        let _rounds = black_box(rounds);
    });

    let modular = measure_samples(samples, || {
        let proof = fixture
            .run_modular_sumcheck()
            .expect("run modular Stage 5 regular-batch sumcheck");
        let _proof = black_box(proof.output_claim);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage5_regular_batch_sumcheck_memory(fixture.config.log_t, fixture.config.log_k),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 5 regular-batch sumcheck kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 5 regular-batch sumcheck evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 5 regular-batch sumcheck evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_materialized_opening_rlc_kernel_evidence() {
    use std::sync::Arc;

    use jolt_backends::{
        cpu::CpuBackend, OpeningBackend, OpeningRlcComponent, OpeningRlcMaterializationRequest,
    };
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::{
            dense_mlpoly::DensePolynomial, multilinear_polynomial::MultilinearPolynomial,
            rlc_polynomial::RLCPolynomial,
        },
        zkvm::witness::CommittedPolynomial,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_openings::mock::MockCommitmentScheme;
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};
    use jolt_witness::{
        MaterializationPolicy, NamespaceId, OracleDescriptor, OracleKind, OracleRef,
        OracleViewRequest, PolynomialEncoding, PolynomialView, RetentionHint, ViewRequirement,
        WitnessDimensions, WitnessError, WitnessNamespace, WitnessProvider,
    };

    const KERNEL: &str = "cpu_materialized_opening_evaluations";
    const BENCHMARK: &str = "cpu_openings/rlc_materialized_fallback";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-OPEN-008"];
    const LOG_ROWS: usize = 18;
    const COMPONENTS: usize = 16;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum EvidenceNamespace {}

    impl WitnessNamespace for EvidenceNamespace {
        type ChallengeId = u8;
        type CommittedId = u8;
        type OpeningId = u8;
        type PublicId = u8;
        type VirtualId = u8;

        const ID: NamespaceId = NamespaceId::new("opening_rlc_evidence");
    }

    struct EvidenceWitness {
        values: Vec<Vec<Fr>>,
        dimensions: WitnessDimensions,
    }

    impl WitnessProvider<Fr, EvidenceNamespace> for EvidenceWitness {
        fn describe_oracle(
            &self,
            oracle: OracleRef<EvidenceNamespace>,
        ) -> Result<OracleDescriptor<EvidenceNamespace>, WitnessError> {
            let OracleKind::Committed(id) = oracle.kind else {
                return Err(WitnessError::UnknownOracle {
                    namespace: EvidenceNamespace::ID.name,
                });
            };
            if usize::from(id) >= self.values.len() {
                return Err(WitnessError::UnknownOracle {
                    namespace: EvidenceNamespace::ID.name,
                });
            }
            Ok(OracleDescriptor::new(
                oracle,
                self.dimensions,
                PolynomialEncoding::Dense,
            ))
        }

        fn view_requirements(
            &self,
            oracle: OracleRef<EvidenceNamespace>,
        ) -> Result<Vec<ViewRequirement<EvidenceNamespace>>, WitnessError> {
            let _descriptor = self.describe_oracle(oracle)?;
            Ok(vec![ViewRequirement::new(
                oracle,
                PolynomialEncoding::Dense,
                MaterializationPolicy::BackendChoice,
                RetentionHint::ThroughStage8,
            )])
        }

        fn oracle_view(
            &self,
            request: OracleViewRequest<EvidenceNamespace>,
        ) -> Result<PolynomialView<'_, Fr, EvidenceNamespace>, WitnessError> {
            let OracleKind::Committed(id) = request.oracle().kind else {
                return Err(WitnessError::UnknownOracle {
                    namespace: EvidenceNamespace::ID.name,
                });
            };
            let values = self
                .values
                .get(usize::from(id))
                .ok_or(WitnessError::UnknownOracle {
                    namespace: EvidenceNamespace::ID.name,
                })?;
            let descriptor = self.describe_oracle(request.oracle())?;
            Ok(PolynomialView::borrowed(descriptor, values))
        }
    }

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let rows = 1usize << LOG_ROWS;

    let core_polys = (0..COMPONENTS)
        .map(|component| {
            let values = (0..rows)
                .map(|row| <CoreFr as JoltField>::from_u64((component * rows + row + 1) as u64))
                .collect();
            Arc::new(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                values,
            )))
        })
        .collect::<Vec<_>>();
    let core_poly_ids = (0..COMPONENTS)
        .map(|component| match component {
            0 => CommittedPolynomial::RdInc,
            1 => CommittedPolynomial::RamInc,
            _ => CommittedPolynomial::TrustedAdvice,
        })
        .collect::<Vec<_>>();
    let core_scalars = (0..COMPONENTS)
        .map(|component| <CoreFr as JoltField>::from_u64(50_000 + component as u64))
        .collect::<Vec<_>>();

    let witness = EvidenceWitness {
        values: (0..COMPONENTS)
            .map(|component| {
                (0..rows)
                    .map(|row| Fr::from_u64((component * rows + row + 1) as u64))
                    .collect()
            })
            .collect(),
        dimensions: WitnessDimensions::new(rows, LOG_ROWS),
    };
    let components = (0..COMPONENTS)
        .map(|component| {
            OpeningRlcComponent::new(
                ViewRequirement::new(
                    OracleRef::committed(component as u8),
                    PolynomialEncoding::Dense,
                    MaterializationPolicy::BackendChoice,
                    RetentionHint::ThroughStage8,
                ),
                Fr::from_u64(50_000 + component as u64),
            )
        })
        .collect();
    let request =
        OpeningRlcMaterializationRequest::new("evidence.openings.materialized_rlc", components);

    let core = measure_samples(samples, || {
        let rlc = RLCPolynomial::linear_combination(
            core_poly_ids.clone(),
            core_polys.clone(),
            &core_scalars,
            None,
        );
        let _len = black_box(rlc.dense_rlc.len());
    });

    let modular = measure_samples(samples, || {
        let mut backend = CpuBackend::default();
        let output = <CpuBackend as OpeningBackend<
            Fr,
            EvidenceNamespace,
            MockCommitmentScheme<Fr>,
        >>::materialize_opening_rlc(&mut backend, &request, &witness)
        .expect("materialize modular opening RLC");
        let _len = black_box(output.values.len());
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: materialized_opening_rlc_memory::<EvidenceNamespace>(rows, COMPONENTS),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered materialized opening RLC kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("materialized opening RLC kernel evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical materialized opening RLC evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage8_streaming_rlc_kernel_evidence() {
    use std::sync::Arc;

    use jolt_backends::cpu::poly::{
        stage8_streaming_rlc_vector_matrix_product, Stage8StreamingRlcVectorMatrixProductInput,
    };
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::{
            commitment::dory::{DoryContext, DoryGlobals, DoryLayout},
            eq_poly::EqPolynomial as CoreEqPolynomial,
            rlc_polynomial::{RLCPolynomial, RLCStreamingData, StreamingRLCContext, TraceSource},
        },
        zkvm::{
            bytecode::get_pc_for_cycle, instruction::LookupQuery as CoreLookupQuery,
            ram::remap_address, witness::CommittedPolynomial,
        },
    };
    use jolt_field::Fr;
    use jolt_prover_harness::{
        load_stage0_commitment_kernel_benchmark_fixture, validate_kernel_benchmark_evidence,
        FeatureMode, FixtureKind, FixtureRequest, KnownOptimizationIds,
    };
    use jolt_witness::protocols::jolt_vm::JoltVmStage6Row;
    use tracer::instruction::RAMAccess;

    const KERNEL: &str = "cpu_opening_stage8_kernels";
    const BENCHMARK: &str = "frontier_perf/stage8_streaming_rlc";
    const OPTIMIZATION_IDS: [&str; 6] = [
        "OPT-OPEN-001",
        "OPT-OPEN-002",
        "OPT-OPEN-003",
        "OPT-OPEN-005",
        "OPT-OPEN-006",
        "OPT-OPEN-007",
    ];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage0_commitment_kernel_benchmark_fixture(&request)
        .expect("load Stage 8 streaming RLC fixture");
    let shape = fixture.shape().expect("Stage 8 streaming RLC shape");
    let trace_rows = shape.trace_length;
    let one_hot_params = fixture.core_one_hot_params();
    DoryGlobals::initialize_context(
        one_hot_params.k_chunk,
        trace_rows,
        DoryContext::Main,
        Some(DoryLayout::CycleMajor),
    )
    .expect("initialize core Dory globals for Stage 8 streaming RLC");
    let num_columns = DoryGlobals::get_num_columns();
    let num_rows = DoryGlobals::get_max_num_rows();
    let rows_per_address = trace_rows / num_columns;

    let core_address_factors = CoreEqPolynomial::<CoreFr>::evals(
        &(0..one_hot_params.log_k_chunk)
            .map(|index| <CoreFr as JoltField>::from_u64(710_000 + index as u64 * 17))
            .collect::<Vec<_>>(),
    );
    let core_row_factors = CoreEqPolynomial::<CoreFr>::evals(
        &(0..rows_per_address.ilog2() as usize)
            .map(|index| <CoreFr as JoltField>::from_u64(720_000 + index as u64 * 19))
            .collect::<Vec<_>>(),
    );
    let core_left = core_address_factors
        .iter()
        .flat_map(|&address| core_row_factors.iter().map(move |&row| address * row))
        .collect::<Vec<_>>();
    assert_eq!(core_left.len(), num_rows);
    let modular_left = core_left.iter().copied().map(Fr::from).collect::<Vec<_>>();

    let core_ram_inc_coefficient = <CoreFr as JoltField>::from_u64(730_001);
    let core_rd_inc_coefficient = <CoreFr as JoltField>::from_u64(730_003);
    let core_instruction_coefficients = (0..one_hot_params.instruction_d)
        .map(|index| <CoreFr as JoltField>::from_u64(731_000 + index as u64 * 5))
        .collect::<Vec<_>>();
    let core_bytecode_coefficients = (0..one_hot_params.bytecode_d)
        .map(|index| <CoreFr as JoltField>::from_u64(732_000 + index as u64 * 7))
        .collect::<Vec<_>>();
    let core_ram_coefficients = (0..one_hot_params.ram_d)
        .map(|index| <CoreFr as JoltField>::from_u64(733_000 + index as u64 * 11))
        .collect::<Vec<_>>();
    let modular_ram_inc_coefficient = Fr::from(core_ram_inc_coefficient);
    let modular_rd_inc_coefficient = Fr::from(core_rd_inc_coefficient);
    let modular_instruction_coefficients = core_instruction_coefficients
        .iter()
        .copied()
        .map(Fr::from)
        .collect::<Vec<_>>();
    let modular_bytecode_coefficients = core_bytecode_coefficients
        .iter()
        .copied()
        .map(Fr::from)
        .collect::<Vec<_>>();
    let modular_ram_coefficients = core_ram_coefficients
        .iter()
        .copied()
        .map(Fr::from)
        .collect::<Vec<_>>();

    let core_dense_polys = vec![
        (CommittedPolynomial::RamInc, core_ram_inc_coefficient),
        (CommittedPolynomial::RdInc, core_rd_inc_coefficient),
    ];
    let mut core_onehot_polys = Vec::new();
    core_onehot_polys.extend(
        core_instruction_coefficients
            .iter()
            .copied()
            .enumerate()
            .map(|(index, coefficient)| (CommittedPolynomial::InstructionRa(index), coefficient)),
    );
    core_onehot_polys.extend(
        core_bytecode_coefficients
            .iter()
            .copied()
            .enumerate()
            .map(|(index, coefficient)| (CommittedPolynomial::BytecodeRa(index), coefficient)),
    );
    core_onehot_polys.extend(
        core_ram_coefficients
            .iter()
            .copied()
            .enumerate()
            .map(|(index, coefficient)| (CommittedPolynomial::RamRa(index), coefficient)),
    );
    let core_trace = Arc::new(fixture.core_trace().to_vec());
    let core_rlc = RLCPolynomial {
        dense_rlc: Vec::new(),
        one_hot_rlc: Vec::new(),
        streaming_context: Some(Arc::new(StreamingRLCContext {
            dense_polys: core_dense_polys,
            onehot_polys: core_onehot_polys,
            advice_polys: Vec::new(),
            trace_source: TraceSource::Materialized(Arc::clone(&core_trace)),
            preprocessing: Arc::new(RLCStreamingData {
                bytecode: Arc::new(fixture.core_bytecode().clone()),
                memory_layout: fixture.core_memory_layout().clone(),
            }),
            one_hot_params: one_hot_params.clone(),
        })),
    };

    let modular_rows = fixture
        .core_trace()
        .iter()
        .map(|cycle| {
            let ram_access = cycle.ram_access();
            let ram_address = match ram_access {
                RAMAccess::Read(read) => Some(read.address),
                RAMAccess::Write(write) => Some(write.address),
                RAMAccess::NoOp => None,
            };
            let remapped_ram_address = ram_address
                .and_then(|address| remap_address(address, fixture.core_memory_layout()))
                .map(|address| address as usize);
            let ram_increment = match ram_access {
                RAMAccess::Write(write) => write.post_value as i128 - write.pre_value as i128,
                RAMAccess::Read(_) | RAMAccess::NoOp => 0,
            };
            let rd_increment = cycle
                .rd_write()
                .map_or(0, |(_, pre, post)| post as i128 - pre as i128);
            JoltVmStage6Row {
                instruction_lookup_index: CoreLookupQuery::<64>::to_lookup_index(cycle),
                bytecode_index: get_pc_for_cycle(fixture.core_bytecode(), cycle),
                remapped_ram_address,
                ram_access_nonzero: ram_address.is_some_and(|address| address != 0),
                ram_increment,
                rd_increment,
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(modular_rows.len(), trace_rows);

    let core_reference = core_rlc.vector_matrix_product(&core_left);
    let modular_reference = stage8_streaming_rlc_vector_matrix_product(
        Stage8StreamingRlcVectorMatrixProductInput {
            rows: &modular_rows,
            field_rd_inc: None,
            log_t: shape.log_t,
            committed_chunk_bits: one_hot_params.log_k_chunk,
            trace_polynomial_order:
                jolt_claims::protocols::jolt::formulas::dimensions::TracePolynomialOrder::CycleMajor,
            ram_inc_coefficient: modular_ram_inc_coefficient,
            rd_inc_coefficient: modular_rd_inc_coefficient,
            field_rd_inc_coefficient: None,
            instruction_coefficients: &modular_instruction_coefficients,
            bytecode_coefficients: &modular_bytecode_coefficients,
            ram_coefficients: &modular_ram_coefficients,
            left_vec: &modular_left,
            num_columns,
        },
    );
    assert_eq!(
        core_reference
            .iter()
            .copied()
            .map(Fr::from)
            .collect::<Vec<_>>(),
        modular_reference
    );

    let core = measure_samples(samples, || {
        let result = core_rlc.vector_matrix_product(&core_left);
        let _result = black_box(result);
    });

    let modular = measure_samples(samples, || {
        let result = stage8_streaming_rlc_vector_matrix_product(
            Stage8StreamingRlcVectorMatrixProductInput {
                rows: &modular_rows,
                field_rd_inc: None,
                log_t: shape.log_t,
                committed_chunk_bits: one_hot_params.log_k_chunk,
                trace_polynomial_order:
                    jolt_claims::protocols::jolt::formulas::dimensions::TracePolynomialOrder::CycleMajor,
                ram_inc_coefficient: modular_ram_inc_coefficient,
                rd_inc_coefficient: modular_rd_inc_coefficient,
                field_rd_inc_coefficient: None,
                instruction_coefficients: &modular_instruction_coefficients,
                bytecode_coefficients: &modular_bytecode_coefficients,
                ram_coefficients: &modular_ram_coefficients,
                left_vec: &modular_left,
                num_columns,
            },
        );
        let _result = black_box(result);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage8_streaming_rlc_memory(
            trace_rows,
            num_rows,
            num_columns,
            one_hot_params.k_chunk,
            one_hot_params.instruction_d + one_hot_params.bytecode_d + one_hot_params.ram_d,
        ),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 8 opening kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 8 streaming RLC evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 8 streaming RLC evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_eq_table_kernel_evidence() {
    use jolt_backends::cpu::eq;
    use jolt_core::{ark_bn254::Fr as CoreFr, field::JoltField, poly::eq_poly::EqPolynomial};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_eq_table_generation";
    const BENCHMARK: &str = "cpu_sumcheck/eq_tables";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-EQ-001", "OPT-EQ-002"];
    const LOG_VARS: usize = 18;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let core_point = (0..LOG_VARS)
        .map(|index| <CoreFr as JoltField>::Challenge::from(90_000 + index as u128 * 19))
        .collect::<Vec<_>>();
    let core_scale = <CoreFr as JoltField>::from_u64(31);
    let modular_point = (0..LOG_VARS)
        .map(|index| Fr::from_u64(90_000 + index as u64 * 19))
        .collect::<Vec<_>>();
    let modular_scale = Fr::from_u64(31);

    let core = measure_samples(samples, || {
        let evals = EqPolynomial::<CoreFr>::evals_with_scaling(&core_point, Some(core_scale));
        let cached = EqPolynomial::<CoreFr>::evals_serial_cached(&core_point, Some(core_scale));
        let cached_rev =
            EqPolynomial::<CoreFr>::evals_serial_cached_rev(&core_point, Some(core_scale));
        let _len = black_box(
            evals.len()
                + cached.last().expect("cached eq table").len()
                + cached_rev.last().expect("reverse cached eq table").len(),
        );
    });

    let modular = measure_samples(samples, || {
        let evals = eq::evals(&modular_point, Some(modular_scale));
        let cached = eq::evals_cached(&modular_point, Some(modular_scale));
        let cached_rev = eq::evals_cached_rev(&modular_point, Some(modular_scale));
        let _len = black_box(
            evals.len()
                + cached.last().expect("cached eq table").len()
                + cached_rev.last().expect("reverse cached eq table").len(),
        );
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: eq_table_memory(LOG_VARS),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger.find(KERNEL).expect("registered eq table kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("eq table kernel evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical eq table evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_eq_aligned_block_kernel_evidence() {
    use jolt_backends::cpu::eq;
    use jolt_core::{ark_bn254::Fr as CoreFr, field::JoltField, poly::eq_poly::EqPolynomial};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_eq_aligned_block_generation";
    const BENCHMARK: &str = "cpu_sumcheck/eq_aligned_blocks";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-EQ-003"];
    const LOG_VARS: usize = 24;
    const SCAN_START: usize = 12_288;
    const SCAN_LEN: usize = 1 << 18;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let core_point = (0..LOG_VARS)
        .map(|index| <CoreFr as JoltField>::Challenge::from(120_000 + index as u128 * 23))
        .collect::<Vec<_>>();
    let modular_point = (0..LOG_VARS)
        .map(|index| Fr::from_u64(120_000 + index as u64 * 23))
        .collect::<Vec<_>>();

    let core = measure_samples(samples, || {
        let mut total = 0usize;
        for block_vars in [12usize, 16] {
            let block_size = 1usize << block_vars;
            let start = 3usize << block_vars;
            let block =
                EqPolynomial::<CoreFr>::evals_for_aligned_block(&core_point, start, block_size);
            total += block.len();
        }
        let mut cursor = SCAN_START;
        let end = SCAN_START + SCAN_LEN;
        while cursor < end {
            let (block_size, block) = EqPolynomial::<CoreFr>::evals_for_max_aligned_block(
                &core_point,
                cursor,
                end - cursor,
            );
            total += block.len();
            cursor += block_size;
        }
        let _total = black_box(total);
    });

    let modular = measure_samples(samples, || {
        let mut total = 0usize;
        for block_vars in [12usize, 16] {
            let block_size = 1usize << block_vars;
            let start = 3usize << block_vars;
            let block = eq::evals_for_aligned_block(&modular_point, start, block_size);
            total += block.len();
        }
        let mut cursor = SCAN_START;
        let end = SCAN_START + SCAN_LEN;
        while cursor < end {
            let (block_size, block) =
                eq::evals_for_max_aligned_block(&modular_point, cursor, end - cursor);
            total += block.len();
            cursor += block_size;
        }
        let _total = black_box(total);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: eq_aligned_block_memory(LOG_VARS, SCAN_LEN),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered eq aligned-block kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("eq aligned-block kernel evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical eq aligned-block evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_split_eq_streaming_window_kernel_evidence() {
    use jolt_backends::cpu::split_eq;
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::{
            multilinear_polynomial::BindingOrder as CoreBindingOrder,
            split_eq_poly::GruenSplitEqPolynomial,
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::BindingOrder;
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_split_eq_streaming_windows";
    const BENCHMARK: &str = "cpu_sumcheck/split_eq_windows";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-EQ-005"];
    const LOG_VARS: usize = 24;
    const WINDOW_SIZE: usize = 10;
    const ROUNDS: usize = 8;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let core_point = (0..LOG_VARS)
        .map(|index| <CoreFr as JoltField>::Challenge::from(130_000 + index as u128 * 29))
        .collect::<Vec<_>>();
    let core_challenges = (0..ROUNDS)
        .map(|index| <CoreFr as JoltField>::Challenge::from(131_000 + index as u128 * 31))
        .collect::<Vec<_>>();
    let modular_point = (0..LOG_VARS)
        .map(|index| Fr::from_u64(130_000 + index as u64 * 29))
        .collect::<Vec<_>>();
    let modular_challenges = (0..ROUNDS)
        .map(|index| Fr::from_u64(131_000 + index as u64 * 31))
        .collect::<Vec<_>>();

    let core = measure_samples(samples, || {
        let mut split =
            GruenSplitEqPolynomial::<CoreFr>::new(&core_point, CoreBindingOrder::LowToHigh);
        let mut total = 0usize;
        for &challenge in &core_challenges {
            let (e_out, e_in) = split.E_out_in_for_window(WINDOW_SIZE);
            total += e_out.len() + e_in.len();
            total += split.E_active_for_window(WINDOW_SIZE).len();
            split.bind(challenge);
        }
        let _total = black_box(total);
    });

    let modular = measure_samples(samples, || {
        let mut split = split_eq::gruen(&modular_point, BindingOrder::LowToHigh);
        let mut total = 0usize;
        for &challenge in &modular_challenges {
            let (e_out, e_in) = split_eq::e_out_in_for_window(&split, WINDOW_SIZE);
            total += e_out.len() + e_in.len();
            total += split_eq::e_active_for_window(&split, WINDOW_SIZE).len();
            split.bind(challenge);
        }
        let _total = black_box(total);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: split_eq_window_memory(LOG_VARS, WINDOW_SIZE, ROUNDS),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered split-eq streaming-window kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("split-eq streaming-window evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical split-eq streaming-window evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_unipoly_interpolation_kernel_evidence() {
    use jolt_backends::cpu::univariate;
    use jolt_core::{ark_bn254::Fr as CoreFr, field::JoltField, poly::unipoly::UniPoly};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_unipoly_interpolation";
    const BENCHMARK: &str = "cpu_sumcheck/unipoly_interpolation";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-EQ-006", "OPT-EQ-007"];
    const ITERS: usize = 16_384;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();

    let core_quadratic = [
        <CoreFr as JoltField>::from_u64(100_001),
        <CoreFr as JoltField>::from_u64(100_019),
        <CoreFr as JoltField>::from_u64(100_043),
    ];
    let core_cubic = [
        <CoreFr as JoltField>::from_u64(101_001),
        <CoreFr as JoltField>::from_u64(101_019),
        <CoreFr as JoltField>::from_u64(101_043),
        <CoreFr as JoltField>::from_u64(101_071),
    ];
    let core_hinted = [core_cubic[0], core_cubic[2], core_cubic[3]];
    let core_hint = core_cubic[0] + core_cubic[1];
    let core_toom = [
        <CoreFr as JoltField>::from_u64(102_001),
        <CoreFr as JoltField>::from_u64(102_019),
        <CoreFr as JoltField>::from_u64(102_043),
        <CoreFr as JoltField>::from_u64(102_071),
        <CoreFr as JoltField>::from_u64(102_101),
    ];

    let modular_quadratic = [
        Fr::from_u64(100_001),
        Fr::from_u64(100_019),
        Fr::from_u64(100_043),
    ];
    let modular_cubic = [
        Fr::from_u64(101_001),
        Fr::from_u64(101_019),
        Fr::from_u64(101_043),
        Fr::from_u64(101_071),
    ];
    let modular_hinted = [modular_cubic[0], modular_cubic[2], modular_cubic[3]];
    let modular_hint = modular_cubic[0] + modular_cubic[1];
    let modular_toom = [
        Fr::from_u64(102_001),
        Fr::from_u64(102_019),
        Fr::from_u64(102_043),
        Fr::from_u64(102_071),
        Fr::from_u64(102_101),
    ];

    let core = measure_samples(samples, || {
        let mut total = 0usize;
        for _ in 0..ITERS {
            total += UniPoly::<CoreFr>::from_evals(&core_quadratic).coeffs.len();
            total += UniPoly::<CoreFr>::from_evals(&core_cubic).coeffs.len();
            total += UniPoly::<CoreFr>::from_evals_and_hint(core_hint, &core_hinted)
                .coeffs
                .len();
            total += UniPoly::<CoreFr>::from_evals_toom(&core_toom).coeffs.len();
        }
        let _total = black_box(total);
    });

    let modular = measure_samples(samples, || {
        let mut total = 0usize;
        for _ in 0..ITERS {
            total += univariate::from_evals(&modular_quadratic)
                .coefficients()
                .len();
            total += univariate::from_evals(&modular_cubic).coefficients().len();
            total += univariate::from_evals_and_hint(modular_hint, &modular_hinted)
                .coefficients()
                .len();
            total += univariate::from_evals_toom(&modular_toom)
                .coefficients()
                .len();
        }
        let _total = black_box(total);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: unipoly_interpolation_memory(),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered unipoly interpolation kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("unipoly interpolation evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical unipoly interpolation evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_compressed_unipoly_kernel_evidence() {
    use jolt_backends::cpu::univariate;
    use jolt_core::{ark_bn254::Fr as CoreFr, field::JoltField, poly::unipoly::UniPoly};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::UnivariatePoly;
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_compressed_unipoly";
    const BENCHMARK: &str = "cpu_sumcheck/compressed_unipoly";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-EQ-008"];
    const ITERS: usize = 16_384;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();

    let core_poly = UniPoly::<CoreFr>::from_coeff(
        (0..9)
            .map(|index| <CoreFr as JoltField>::from_u64(140_000 + index as u64 * 17))
            .collect(),
    );
    let core_hint = core_poly.eval_at_zero() + core_poly.eval_at_one();
    let core_compressed = core_poly.compress();
    let core_points = (0..8)
        .map(|index| <CoreFr as JoltField>::Challenge::from(141_000 + index as u128 * 19))
        .collect::<Vec<_>>();

    let modular_poly = UnivariatePoly::new(
        (0..9)
            .map(|index| Fr::from_u64(140_000 + index as u64 * 17))
            .collect(),
    );
    let modular_hint =
        modular_poly.evaluate(Fr::from_u64(0)) + modular_poly.evaluate(Fr::from_u64(1));
    let modular_compressed = univariate::compress(&modular_poly);
    let modular_points = (0..8)
        .map(|index| Fr::from_u64(141_000 + index as u64 * 19))
        .collect::<Vec<_>>();

    let core = measure_samples(samples, || {
        let mut total = 0usize;
        let mut eval_sum = <CoreFr as JoltField>::from_u64(0);
        for _ in 0..ITERS {
            total += core_poly.compress().coeffs_except_linear_term.len();
            total += core_compressed.decompress(&core_hint).coeffs.len();
            for point in &core_points {
                eval_sum += core_compressed.eval_from_hint(&core_hint, point);
            }
        }
        let _total = black_box((total, eval_sum));
    });

    let modular = measure_samples(samples, || {
        let mut total = 0usize;
        let mut eval_sum = Fr::from_u64(0);
        for _ in 0..ITERS {
            total += univariate::compress(&modular_poly)
                .coeffs_except_linear_term()
                .len();
            total += univariate::decompress(&modular_compressed, modular_hint)
                .coefficients()
                .len();
            for &point in &modular_points {
                eval_sum += univariate::eval_from_hint(&modular_compressed, modular_hint, point);
            }
        }
        let _total = black_box((total, eval_sum));
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: compressed_unipoly_memory(),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered compressed unipoly kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("compressed unipoly evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical compressed unipoly evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_lagrange_many_kernel_evidence() {
    use jolt_backends::cpu::lagrange;
    use jolt_core::{
        ark_bn254::Fr as CoreFr, field::JoltField, poly::lagrange_poly::LagrangePolynomial,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_lagrange_many";
    const BENCHMARK: &str = "cpu_sumcheck/lagrange_many";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-EQ-009"];
    const N: usize = 10;
    const ITERS: usize = 4_096;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();

    let core_values =
        core::array::from_fn(|index| <CoreFr as JoltField>::from_u64(150_000 + index as u64 * 17));
    let core_point = <CoreFr as JoltField>::from_u64(151_000);
    let core_other = <CoreFr as JoltField>::from_u64(151_019);
    let core_points = (0..16)
        .map(|index| <CoreFr as JoltField>::from_u64(152_000 + index as u64 * 19))
        .collect::<Vec<_>>();

    let modular_values = core::array::from_fn(|index| Fr::from_u64(150_000 + index as u64 * 17));
    let modular_point = Fr::from_u64(151_000);
    let modular_other = Fr::from_u64(151_019);
    let modular_points = (0..16)
        .map(|index| Fr::from_u64(152_000 + index as u64 * 19))
        .collect::<Vec<_>>();

    let core = measure_samples(samples, || {
        let mut total = 0usize;
        let mut eval_sum = <CoreFr as JoltField>::from_u64(0);
        for _ in 0..ITERS {
            total += LagrangePolynomial::<CoreFr>::evals::<CoreFr, N>(&core_point).len();
            eval_sum += LagrangePolynomial::<CoreFr>::lagrange_kernel::<CoreFr, N>(
                &core_point,
                &core_other,
            );
            total += LagrangePolynomial::<CoreFr>::evaluate_many::<CoreFr, N>(
                &core_values,
                &core_points,
            )
            .len();
            eval_sum += LagrangePolynomial::<CoreFr>::interpolate_coeffs::<N>(&core_values)[0];
        }
        let _total = black_box((total, eval_sum));
    });

    let modular = measure_samples(samples, || {
        let mut total = 0usize;
        let mut eval_sum = Fr::from_u64(0);
        for _ in 0..ITERS {
            total += lagrange::centered_evals::<Fr, N>(modular_point)
                .expect("centered evals")
                .len();
            eval_sum += lagrange::centered_kernel(N, modular_point, modular_other)
                .expect("centered kernel");
            total += lagrange::centered_evaluate_many::<Fr, N>(&modular_values, &modular_points)
                .expect("centered evaluate_many")
                .len();
            eval_sum += lagrange::centered_interpolate_coeffs::<Fr, N>(&modular_values)
                .expect("centered interpolate coeffs")[0];
        }
        let _total = black_box((total, eval_sum));
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: lagrange_many_memory(N, core_points.len()),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Lagrange batch kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Lagrange batch evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Lagrange batch evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_compact_polynomial_bind_kernel_evidence() {
    use jolt_backends::cpu::poly;
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::{
            compact_polynomial::CompactPolynomial,
            multilinear_polynomial::{BindingOrder as CoreBindingOrder, PolynomialBinding},
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_compact_polynomial_bind";
    const BENCHMARK: &str = "cpu_poly/compact_bind";
    const OPTIMIZATION_IDS: [&str; 4] = [
        "OPT-POLY-002",
        "OPT-POLY-003",
        "OPT-POLY-004",
        "OPT-POLY-005",
    ];
    const LOG_ROWS: usize = 20;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let rows = 1usize << LOG_ROWS;
    let core_coeffs = compact_bind_coeffs(rows);
    let modular_coeffs = core_coeffs.clone();
    let core_r0 = <CoreFr as JoltField>::Challenge::from(17_u128);
    let core_r1 = <CoreFr as JoltField>::Challenge::from(29_u128);
    let modular_r0 = Fr::from_u64(17);
    let modular_r1 = Fr::from_u64(29);

    let core = measure_samples(samples, || {
        let mut high = CompactPolynomial::<u8, CoreFr>::from_coeffs(core_coeffs.clone());
        high.bind_parallel(core_r0, CoreBindingOrder::HighToLow);
        high.bind_parallel(core_r1, CoreBindingOrder::HighToLow);
        let mut low = CompactPolynomial::<u8, CoreFr>::from_coeffs(core_coeffs.clone());
        low.bind_parallel(core_r0, CoreBindingOrder::LowToHigh);
        low.bind_parallel(core_r1, CoreBindingOrder::LowToHigh);
        let _len = black_box(high.len() + low.len());
    });

    let modular = measure_samples(samples, || {
        let mut high = poly::bind_compact_first_high_to_low::<_, Fr>(&modular_coeffs, modular_r0);
        poly::bind_field_high_to_low(&mut high, modular_r1);
        let mut low = poly::bind_compact_first_low_to_high::<_, Fr>(&modular_coeffs, modular_r0);
        low = poly::bind_field_low_to_high(&low, modular_r1);
        let _len = black_box(high.len() + low.len());
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: compact_polynomial_bind_memory(rows),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered compact polynomial bind kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("compact polynomial bind evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical compact polynomial bind evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    fn compact_bind_coeffs(rows: usize) -> Vec<u8> {
        (0..rows)
            .map(|index| ((index * 13 + (index >> 5) * 7 + 3) % 251) as u8)
            .collect()
    }
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_split_eq_polynomial_evaluation_kernel_evidence() {
    use jolt_backends::cpu::{eq, poly};
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::{
            compact_polynomial::CompactPolynomial, dense_mlpoly::DensePolynomial,
            eq_poly::EqPolynomial,
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_split_eq_polynomial_evaluation";
    const BENCHMARK: &str = "cpu_poly/split_eq_evaluate";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-POLY-006", "OPT-POLY-007"];
    const LOG_ROWS: usize = 20;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let rows = 1usize << LOG_ROWS;
    let split = LOG_ROWS / 2;

    let core_point = (0..LOG_ROWS)
        .map(|index| <CoreFr as JoltField>::Challenge::from(1_001_u128 + index as u128 * 17))
        .collect::<Vec<_>>();
    let core_eq_one = EqPolynomial::<CoreFr>::evals(&core_point[..split]);
    let core_eq_two = EqPolynomial::<CoreFr>::evals(&core_point[split..]);
    let core_dense = DensePolynomial::<CoreFr>::new(
        (0..rows)
            .map(|index| match index % 17 {
                0 => <CoreFr as JoltField>::from_u64(0),
                1 => <CoreFr as JoltField>::from_u64(1),
                _ => <CoreFr as JoltField>::from_u64(50_000 + index as u64 * 3),
            })
            .collect(),
    );
    let core_compact = CompactPolynomial::<u8, CoreFr>::from_coeffs(split_eq_coeffs(rows));

    let modular_point = (0..LOG_ROWS)
        .map(|index| Fr::from_u64(1_001 + index as u64 * 17))
        .collect::<Vec<_>>();
    let modular_eq_one = eq::evals(&modular_point[..split], None);
    let modular_eq_two = eq::evals(&modular_point[split..], None);
    let modular_dense = (0..rows)
        .map(|index| match index % 17 {
            0 => Fr::from_u64(0),
            1 => Fr::from_u64(1),
            _ => Fr::from_u64(50_000 + index as u64 * 3),
        })
        .collect::<Vec<_>>();
    let modular_compact = split_eq_coeffs(rows);

    let core = measure_samples(samples, || {
        let dense_eval = core_dense.split_eq_evaluate(LOG_ROWS, &core_eq_one, &core_eq_two);
        let compact_eval = core_compact.split_eq_evaluate(LOG_ROWS, &core_eq_one, &core_eq_two);
        let _evals = black_box((dense_eval, compact_eval));
    });

    let modular = measure_samples(samples, || {
        let dense_eval = poly::dense_split_eq_evaluate(
            &modular_dense,
            LOG_ROWS,
            &modular_eq_one,
            &modular_eq_two,
        );
        let compact_eval = poly::compact_split_eq_evaluate::<_, Fr>(
            &modular_compact,
            LOG_ROWS,
            &modular_eq_one,
            &modular_eq_two,
        );
        let _evals = black_box((dense_eval, compact_eval));
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: split_eq_polynomial_evaluate_memory(rows, core_eq_one.len(), core_eq_two.len()),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered split-eq polynomial evaluation kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("split-eq polynomial evaluation evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical split-eq polynomial evaluation evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    fn split_eq_coeffs(rows: usize) -> Vec<u8> {
        (0..rows)
            .map(|index| match index % 17 {
                0 => 0,
                1 => 1,
                _ => ((index * 19 + (index >> 4) * 5 + 11) % 251) as u8,
            })
            .collect()
    }
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_inside_out_polynomial_evaluation_kernel_evidence() {
    use jolt_backends::cpu::poly;
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::{compact_polynomial::CompactPolynomial, dense_mlpoly::DensePolynomial},
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_inside_out_polynomial_evaluation";
    const BENCHMARK: &str = "cpu_poly/inside_out_evaluate";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-POLY-008"];
    const LOG_ROWS: usize = 20;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let rows = 1usize << LOG_ROWS;

    let core_point = (0..LOG_ROWS)
        .map(|index| <CoreFr as JoltField>::from_u64(2_001 + index as u64 * 19))
        .collect::<Vec<_>>();
    let core_dense = DensePolynomial::<CoreFr>::new(
        (0..rows)
            .map(|index| match index % 19 {
                0 => <CoreFr as JoltField>::from_u64(0),
                1 => <CoreFr as JoltField>::from_u64(1),
                _ => <CoreFr as JoltField>::from_u64(80_000 + index as u64 * 5),
            })
            .collect(),
    );
    let core_compact = CompactPolynomial::<u8, CoreFr>::from_coeffs(inside_out_coeffs(rows));

    let modular_point = (0..LOG_ROWS)
        .map(|index| Fr::from_u64(2_001 + index as u64 * 19))
        .collect::<Vec<_>>();
    let modular_dense = (0..rows)
        .map(|index| match index % 19 {
            0 => Fr::from_u64(0),
            1 => Fr::from_u64(1),
            _ => Fr::from_u64(80_000 + index as u64 * 5),
        })
        .collect::<Vec<_>>();
    let modular_compact = inside_out_coeffs(rows);

    let core = measure_samples(samples, || {
        let dense_eval = core_dense.inside_out_evaluate(&core_point);
        let compact_eval = core_compact.inside_out_evaluate(&core_point);
        let _evals = black_box((dense_eval, compact_eval));
    });

    let modular = measure_samples(samples, || {
        let dense_eval = poly::dense_inside_out_evaluate(&modular_dense, &modular_point);
        let compact_eval =
            poly::compact_inside_out_evaluate::<_, Fr>(&modular_compact, &modular_point);
        let _evals = black_box((dense_eval, compact_eval));
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: inside_out_polynomial_evaluate_memory(rows, core_point.len()),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered inside-out polynomial evaluation kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("inside-out polynomial evaluation evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical inside-out polynomial evaluation evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    fn inside_out_coeffs(rows: usize) -> Vec<u8> {
        (0..rows)
            .map(|index| match index % 19 {
                0 => 0,
                1 => 1,
                _ => ((index * 23 + (index >> 5) * 7 + 13) % 251) as u8,
            })
            .collect()
    }
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_dense_batch_polynomial_evaluation_kernel_evidence() {
    use jolt_backends::cpu::poly;
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::{
            dense_mlpoly::DensePolynomial,
            multilinear_polynomial::PolynomialEvaluation as CorePolynomialEvaluation,
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_dense_batch_polynomial_evaluation";
    const BENCHMARK: &str = "cpu_poly/dense_batch_evaluate";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-POLY-009"];
    const LOG_ROWS: usize = 18;
    const NUM_POLYS: usize = 8;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let rows = 1usize << LOG_ROWS;

    let core_point = (0..LOG_ROWS)
        .map(|index| <CoreFr as JoltField>::from_u64(3_001 + index as u64 * 23))
        .collect::<Vec<_>>();
    let core_polys = (0..NUM_POLYS)
        .map(|poly_index| {
            DensePolynomial::<CoreFr>::new(
                (0..rows)
                    .map(|row| match (row + poly_index) % 23 {
                        0 => <CoreFr as JoltField>::from_u64(0),
                        1 => <CoreFr as JoltField>::from_u64(1),
                        _ => <CoreFr as JoltField>::from_u64(
                            90_000 + row as u64 * 7 + poly_index as u64 * 1_001,
                        ),
                    })
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let core_refs = core_polys.iter().collect::<Vec<_>>();

    let modular_point = (0..LOG_ROWS)
        .map(|index| Fr::from_u64(3_001 + index as u64 * 23))
        .collect::<Vec<_>>();
    let modular_polys = (0..NUM_POLYS)
        .map(|poly_index| {
            (0..rows)
                .map(|row| match (row + poly_index) % 23 {
                    0 => Fr::from_u64(0),
                    1 => Fr::from_u64(1),
                    _ => Fr::from_u64(90_000 + row as u64 * 7 + poly_index as u64 * 1_001),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let modular_refs = modular_polys.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let core_check = <DensePolynomial<CoreFr> as CorePolynomialEvaluation<CoreFr>>::batch_evaluate(
        &core_refs,
        &core_point,
    );
    let modular_check = poly::dense_batch_evaluate(&modular_refs, &modular_point);
    assert_eq!(core_check.len(), modular_check.len());

    let core = measure_samples(samples, || {
        let evals = <DensePolynomial<CoreFr> as CorePolynomialEvaluation<CoreFr>>::batch_evaluate(
            &core_refs,
            &core_point,
        );
        let _evals = black_box(evals);
    });

    let modular = measure_samples(samples, || {
        let evals = poly::dense_batch_evaluate(&modular_refs, &modular_point);
        let _evals = black_box(evals);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: dense_batch_polynomial_evaluate_memory(rows, NUM_POLYS, LOG_ROWS),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered dense batch polynomial evaluation kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("dense batch polynomial evaluation evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical dense batch polynomial evaluation evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_dense_dot_product_low_optimized_kernel_evidence() {
    use jolt_backends::cpu::poly;
    use jolt_core::{
        ark_bn254::Fr as CoreFr, field::JoltField, poly::dense_mlpoly::DensePolynomial,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_dense_dot_product_low_optimized";
    const BENCHMARK: &str = "cpu_poly/dense_dot_product_low_optimized";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-POLY-014"];
    const LOG_ROWS: usize = 20;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let rows = 1usize << LOG_ROWS;

    let core_values = (0..rows)
        .map(|index| match index % 29 {
            0 => <CoreFr as JoltField>::from_u64(0),
            1 => <CoreFr as JoltField>::from_u64(1),
            _ => <CoreFr as JoltField>::from_u64(110_000 + index as u64 * 11),
        })
        .collect::<Vec<_>>();
    let core_chis = (0..rows)
        .map(|index| match index % 31 {
            0 => <CoreFr as JoltField>::from_u64(0),
            1 => <CoreFr as JoltField>::from_u64(1),
            _ => <CoreFr as JoltField>::from_u64(120_000 + index as u64 * 13),
        })
        .collect::<Vec<_>>();
    let core_poly = DensePolynomial::<CoreFr>::new(core_values);

    let modular_values = (0..rows)
        .map(|index| match index % 29 {
            0 => Fr::from_u64(0),
            1 => Fr::from_u64(1),
            _ => Fr::from_u64(110_000 + index as u64 * 11),
        })
        .collect::<Vec<_>>();
    let modular_chis = (0..rows)
        .map(|index| match index % 31 {
            0 => Fr::from_u64(0),
            1 => Fr::from_u64(1),
            _ => Fr::from_u64(120_000 + index as u64 * 13),
        })
        .collect::<Vec<_>>();

    let core = measure_samples(samples, || {
        let eval = core_poly.evaluate_at_chi_low_optimized(&core_chis);
        let _eval = black_box(eval);
    });

    let modular = measure_samples(samples, || {
        let eval = poly::dense_dot_product_low_optimized(&modular_values, &modular_chis);
        let _eval = black_box(eval);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: dense_dot_product_low_optimized_memory(rows),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered dense low-optimized dot product kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("dense low-optimized dot product evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical dense low-optimized dot product evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_mixed_polynomial_linear_combination_kernel_evidence() {
    use jolt_backends::cpu::poly::{self, LinearCombinationInput};
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::{dense_mlpoly::DensePolynomial, multilinear_polynomial::MultilinearPolynomial},
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_mixed_polynomial_linear_combination";
    const BENCHMARK: &str = "cpu_poly/linear_combination";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-POLY-010"];
    const LOG_ROWS: usize = 20;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let rows = 1usize << LOG_ROWS;

    let core_dense = (0..rows)
        .map(|index| match index % 23 {
            0 => <CoreFr as JoltField>::from_u64(0),
            1 => <CoreFr as JoltField>::from_u64(1),
            _ => <CoreFr as JoltField>::from_u64(130_000 + index as u64 * 11),
        })
        .collect::<Vec<_>>();
    let core_u8 = linear_combination_u8(rows);
    let core_i64 = linear_combination_i64(rows);
    let core_u128 = linear_combination_u128(rows);
    let core_coefficients = [
        <CoreFr as JoltField>::from_u64(3),
        <CoreFr as JoltField>::from_u64(5),
        <CoreFr as JoltField>::from_u64(7),
        <CoreFr as JoltField>::from_u64(11),
    ];
    let core_polys = [
        MultilinearPolynomial::<CoreFr>::LargeScalars(DensePolynomial::new(core_dense)),
        MultilinearPolynomial::<CoreFr>::from(core_u8.clone()),
        MultilinearPolynomial::<CoreFr>::from(core_i64.clone()),
        MultilinearPolynomial::<CoreFr>::from(core_u128.clone()),
    ];
    let core_refs = core_polys.iter().collect::<Vec<_>>();

    let modular_dense = (0..rows)
        .map(|index| match index % 23 {
            0 => Fr::from_u64(0),
            1 => Fr::from_u64(1),
            _ => Fr::from_u64(130_000 + index as u64 * 11),
        })
        .collect::<Vec<_>>();
    let modular_u8 = core_u8;
    let modular_i64 = core_i64;
    let modular_u128 = core_u128;
    let modular_coefficients = [
        Fr::from_u64(3),
        Fr::from_u64(5),
        Fr::from_u64(7),
        Fr::from_u64(11),
    ];
    let modular_inputs = [
        LinearCombinationInput::Dense(modular_dense.as_slice()),
        LinearCombinationInput::U8(modular_u8.as_slice()),
        LinearCombinationInput::I64(modular_i64.as_slice()),
        LinearCombinationInput::U128(modular_u128.as_slice()),
    ];

    let core = measure_samples(samples, || {
        let result = DensePolynomial::<CoreFr>::linear_combination(&core_refs, &core_coefficients);
        let _len = black_box(result.len());
    });

    let modular = measure_samples(samples, || {
        let result = poly::linear_combination(&modular_inputs, &modular_coefficients);
        let _len = black_box(result.len());
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: mixed_polynomial_linear_combination_memory(rows, core_refs.len()),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered mixed polynomial linear combination kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("mixed polynomial linear combination evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical mixed polynomial linear combination evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    fn linear_combination_u8(rows: usize) -> Vec<u8> {
        (0..rows)
            .map(|index| ((index * 13 + (index >> 4) * 7 + 5) % 251) as u8)
            .collect()
    }

    fn linear_combination_i64(rows: usize) -> Vec<i64> {
        (0..rows).map(|index| index as i64 * 19 - 50_000).collect()
    }

    fn linear_combination_u128(rows: usize) -> Vec<u128> {
        (0..rows)
            .map(|index| 1_000_000u128 + index as u128 * 17)
            .collect()
    }
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_one_hot_polynomial_evaluation_kernel_evidence() {
    use jolt_backends::cpu::poly;
    use jolt_core::{
        ark_bn254::Fr as CoreFr, field::JoltField,
        poly::one_hot_polynomial::OneHotPolynomial as CoreOneHotPolynomial,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_one_hot_polynomial_evaluation";
    const BENCHMARK: &str = "cpu_poly/one_hot_evaluate";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-POLY-011"];
    const K: usize = 256;
    const T: usize = 1 << 20;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let indices = deterministic_one_hot_indices(K, T);

    let core_point = (0..((K * T).trailing_zeros() as usize))
        .map(|index| <CoreFr as JoltField>::from_u64(140_000 + index as u64 * 17))
        .collect::<Vec<_>>();
    let core_poly = CoreOneHotPolynomial::<CoreFr>::from_indices(indices.clone(), K);

    let modular_point = (0..((K * T).trailing_zeros() as usize))
        .map(|index| Fr::from_u64(140_000 + index as u64 * 17))
        .collect::<Vec<_>>();
    let modular_indices = indices;

    let core = measure_samples(samples, || {
        let eval = core_poly.evaluate(&core_point);
        let _eval = black_box(eval);
    });

    let modular = measure_samples(samples, || {
        let eval = poly::one_hot_evaluate(K, &modular_indices, &modular_point);
        let _eval = black_box(eval);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: one_hot_polynomial_evaluate_memory(K, T),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered one-hot polynomial evaluation kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("one-hot polynomial evaluation evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical one-hot polynomial evaluation evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_one_hot_vector_matrix_product_kernel_evidence() {
    use jolt_backends::cpu::poly;
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::{
            commitment::dory::{DoryContext, DoryGlobals, DoryLayout},
            one_hot_polynomial::OneHotPolynomial as CoreOneHotPolynomial,
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::OneHotIndexOrder;
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_one_hot_vector_matrix_product";
    const BENCHMARK: &str = "cpu_poly/one_hot_vmp";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-POLY-012"];
    const K: usize = 256;
    const T: usize = 1 << 20;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let indices = deterministic_one_hot_indices(K, T);

    let _ = DoryGlobals::initialize_context(K, T, DoryContext::Main, Some(DoryLayout::CycleMajor));
    let num_columns = DoryGlobals::get_num_columns();
    let num_rows = DoryGlobals::get_max_num_rows();
    let core_poly = CoreOneHotPolynomial::<CoreFr>::from_indices(indices.clone(), K);
    let core_left = (0..num_rows)
        .map(|index| <CoreFr as JoltField>::from_u64(150_000 + index as u64 * 19))
        .collect::<Vec<_>>();
    let core_coeff = <CoreFr as JoltField>::from_u64(23);

    let modular_indices = indices;
    let modular_left = (0..num_rows)
        .map(|index| Fr::from_u64(150_000 + index as u64 * 19))
        .collect::<Vec<_>>();
    let modular_coeff = Fr::from_u64(23);

    let core = measure_samples(samples, || {
        let mut result = vec![<CoreFr as JoltField>::from_u64(0); num_columns];
        core_poly.vector_matrix_product(&core_left, core_coeff, &mut result);
        let _result = black_box(result);
    });

    let modular = measure_samples(samples, || {
        let result = poly::one_hot_vector_matrix_product(
            K,
            &modular_indices,
            &modular_left,
            modular_coeff,
            num_columns,
            OneHotIndexOrder::ColumnMajor,
        );
        let _result = black_box(result);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: one_hot_vector_matrix_product_memory(K, T, num_rows, num_columns),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered one-hot vector-matrix product kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("one-hot vector-matrix product evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical one-hot vector-matrix product evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_rlc_polynomial_vector_matrix_product_kernel_evidence() {
    use std::sync::Arc;

    use jolt_backends::cpu::poly::{self, OneHotVectorMatrixProductInput};
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::{
            commitment::dory::{DoryContext, DoryGlobals, DoryLayout},
            multilinear_polynomial::MultilinearPolynomial,
            one_hot_polynomial::OneHotPolynomial as CoreOneHotPolynomial,
            rlc_polynomial::RLCPolynomial,
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::OneHotIndexOrder;
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_rlc_polynomial_vector_matrix_product";
    const BENCHMARK: &str = "cpu_poly/rlc_vmp";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-POLY-013"];
    const K: usize = 256;
    const T: usize = 1 << 20;
    const ONE_HOT_COMPONENTS: usize = 4;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let _ = DoryGlobals::initialize_context(K, T, DoryContext::Main, Some(DoryLayout::CycleMajor));
    let num_columns = DoryGlobals::get_num_columns();
    let num_rows = DoryGlobals::get_max_num_rows();

    let core_dense_rlc = rlc_dense_values::<CoreFr, _>(T, <CoreFr as JoltField>::from_u64);
    let core_left = (0..num_rows)
        .map(|index| <CoreFr as JoltField>::from_u64(170_000 + index as u64 * 19))
        .collect::<Vec<_>>();
    let core_indices = rlc_one_hot_indices(K, T, ONE_HOT_COMPONENTS);
    let core_one_hot_rlc = core_indices
        .iter()
        .enumerate()
        .map(|(component, indices)| {
            let coeff = <CoreFr as JoltField>::from_u64(31 + component as u64 * 7);
            (
                coeff,
                Arc::new(MultilinearPolynomial::OneHot(
                    CoreOneHotPolynomial::<CoreFr>::from_indices(indices.clone(), K),
                )),
            )
        })
        .collect::<Vec<_>>();
    let core_rlc = RLCPolynomial {
        dense_rlc: core_dense_rlc,
        one_hot_rlc: core_one_hot_rlc,
        streaming_context: None,
    };

    let modular_dense_rlc = rlc_dense_values(T, Fr::from_u64);
    let modular_left = (0..num_rows)
        .map(|index| Fr::from_u64(170_000 + index as u64 * 19))
        .collect::<Vec<_>>();
    let modular_indices = core_indices;
    let modular_one_hot_rlc = modular_indices
        .iter()
        .enumerate()
        .map(|(component, indices)| OneHotVectorMatrixProductInput {
            k: K,
            indices,
            coefficient: Fr::from_u64(31 + component as u64 * 7),
            index_order: OneHotIndexOrder::ColumnMajor,
        })
        .collect::<Vec<_>>();

    let core = measure_samples(samples, || {
        let result = core_rlc.vector_matrix_product(&core_left);
        let _result = black_box(result);
    });

    let modular = measure_samples(samples, || {
        let result = poly::materialized_rlc_vector_matrix_product(
            &modular_dense_rlc,
            &modular_one_hot_rlc,
            &modular_left,
            num_columns,
        );
        let _result = black_box(result);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: rlc_vector_matrix_product_memory(T, ONE_HOT_COMPONENTS, num_rows, num_columns),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered RLC polynomial vector-matrix product kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("RLC polynomial vector-matrix product evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical RLC polynomial vector-matrix product evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    fn rlc_dense_values<F, FromU64>(rows: usize, from_u64: FromU64) -> Vec<F>
    where
        FromU64: Fn(u64) -> F,
    {
        (0..rows)
            .map(|index| match index % 29 {
                0 => from_u64(0),
                1 => from_u64(1),
                _ => from_u64(160_000 + index as u64 * 17),
            })
            .collect()
    }

    fn rlc_one_hot_indices(k: usize, rows: usize, components: usize) -> Vec<Vec<Option<u8>>> {
        (0..components)
            .map(|component| {
                (0..rows)
                    .map(|row| {
                        if (row + component) % 17 == 0 {
                            None
                        } else {
                            Some(((row * 13 + row / 5 + component * 29 + 7) % k) as u8)
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_linear_product_small_degrees_kernel_evidence() {
    use jolt_backends::cpu::field;
    use jolt_core::{
        ark_bn254::Fr as CoreFr, field::JoltField,
        subprotocols::mles_product_sum::eval_linear_prod_assign,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_linear_product_small_degrees";
    const BENCHMARK: &str = "cpu_field/linear_product_small_degrees";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-FLD-003"];
    const PRODUCTS: usize = 1 << 18;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();

    let core_d2 = linear_product_products::<2, CoreFr, _>(PRODUCTS, |value| {
        <CoreFr as JoltField>::from_u64(value)
    });
    let core_d3 = linear_product_products::<3, CoreFr, _>(PRODUCTS, |value| {
        <CoreFr as JoltField>::from_u64(value)
    });
    let core_d5 = linear_product_products::<5, CoreFr, _>(PRODUCTS, |value| {
        <CoreFr as JoltField>::from_u64(value)
    });
    let core_d6 = linear_product_products::<6, CoreFr, _>(PRODUCTS, |value| {
        <CoreFr as JoltField>::from_u64(value)
    });
    let core_d7 = linear_product_products::<7, CoreFr, _>(PRODUCTS, |value| {
        <CoreFr as JoltField>::from_u64(value)
    });
    let modular_d2 = linear_product_products::<2, Fr, _>(PRODUCTS, Fr::from_u64);
    let modular_d3 = linear_product_products::<3, Fr, _>(PRODUCTS, Fr::from_u64);
    let modular_d5 = linear_product_products::<5, Fr, _>(PRODUCTS, Fr::from_u64);
    let modular_d6 = linear_product_products::<6, Fr, _>(PRODUCTS, Fr::from_u64);
    let modular_d7 = linear_product_products::<7, Fr, _>(PRODUCTS, Fr::from_u64);

    let core = measure_samples(samples, || {
        let mut sink = [<CoreFr as JoltField>::from_u64(0); 23];
        let mut output = vec![<CoreFr as JoltField>::from_u64(0); 7];
        accumulate_core_linear_products::<2>(&core_d2, &mut output, &mut sink[0..2]);
        accumulate_core_linear_products::<3>(&core_d3, &mut output, &mut sink[2..5]);
        accumulate_core_linear_products::<5>(&core_d5, &mut output, &mut sink[5..10]);
        accumulate_core_linear_products::<6>(&core_d6, &mut output, &mut sink[10..16]);
        accumulate_core_linear_products::<7>(&core_d7, &mut output, &mut sink[16..23]);
        let _sink = black_box(sink);
    });

    let modular = measure_samples(samples, || {
        let mut sink = [Fr::from_u64(0); 23];
        let mut output = vec![Fr::from_u64(0); 7];
        accumulate_modular_linear_products::<2>(
            &modular_d2,
            &mut output,
            &mut sink[0..2],
            field::eval_linear_product_d2_assign,
        );
        accumulate_modular_linear_products::<3>(
            &modular_d3,
            &mut output,
            &mut sink[2..5],
            field::eval_linear_product_d3_assign,
        );
        accumulate_modular_linear_products::<5>(
            &modular_d5,
            &mut output,
            &mut sink[5..10],
            field::eval_linear_product_d5_assign,
        );
        accumulate_modular_linear_products::<6>(
            &modular_d6,
            &mut output,
            &mut sink[10..16],
            field::eval_linear_product_d6_assign,
        );
        accumulate_modular_linear_products::<7>(
            &modular_d7,
            &mut output,
            &mut sink[16..23],
            field::eval_linear_product_d7_assign,
        );
        let _sink = black_box(sink);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: linear_product_small_degrees_memory(PRODUCTS),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered small-degree linear product kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("small-degree linear product evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical small-degree linear product evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    fn accumulate_core_linear_products<const D: usize>(
        products: &[[(CoreFr, CoreFr); D]],
        output: &mut [CoreFr],
        sink: &mut [CoreFr],
    ) {
        for pairs in products {
            eval_linear_prod_assign(pairs, &mut output[..D]);
            for (sink, value) in sink.iter_mut().zip(output[..D].iter()) {
                *sink += *value;
            }
        }
    }

    fn accumulate_modular_linear_products<const D: usize>(
        products: &[[(Fr, Fr); D]],
        output: &mut [Fr],
        sink: &mut [Fr],
        eval: fn(&[(Fr, Fr); D], &mut [Fr]),
    ) {
        for pairs in products {
            eval(pairs, &mut output[..D]);
            for (sink, value) in sink.iter_mut().zip(output[..D].iter()) {
                *sink += *value;
            }
        }
    }

    fn linear_product_products<const D: usize, F, FromU64>(
        count: usize,
        from_u64: FromU64,
    ) -> Vec<[(F, F); D]>
    where
        FromU64: Fn(u64) -> F,
    {
        (0..count)
            .map(|product| {
                core::array::from_fn(|factor| {
                    (
                        from_u64(
                            170_000 + product as u64 * 17 + D as u64 * 101 + factor as u64 * 3,
                        ),
                        from_u64(
                            190_000 + product as u64 * 19 + D as u64 * 103 + factor as u64 * 5,
                        ),
                    )
                })
            })
            .collect()
    }
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_linear_product_d4_kernel_evidence() {
    use jolt_backends::cpu::field;
    use jolt_core::{
        ark_bn254::Fr as CoreFr, field::JoltField,
        subprotocols::mles_product_sum::eval_linear_prod_assign,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_linear_product_d4";
    const BENCHMARK: &str = "cpu_field/linear_product_d4";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-FLD-003"];
    const PRODUCTS: usize = 1 << 20;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();

    let core_products = linear_product_d4_products::<CoreFr, _>(PRODUCTS, |value| {
        <CoreFr as JoltField>::from_u64(value)
    });
    let modular_products = linear_product_d4_products(PRODUCTS, Fr::from_u64);

    let core = measure_samples(samples, || {
        let mut sink = [<CoreFr as JoltField>::from_u64(0); 4];
        let mut output = vec![<CoreFr as JoltField>::from_u64(0); 4];
        for pairs in &core_products {
            eval_linear_prod_assign(pairs, &mut output);
            for (sink, value) in sink.iter_mut().zip(output.iter()) {
                *sink += *value;
            }
        }
        let _sink = black_box(sink);
    });

    let modular = measure_samples(samples, || {
        let mut sink = [Fr::from_u64(0); 4];
        let mut output = vec![Fr::from_u64(0); 4];
        for pairs in &modular_products {
            field::eval_linear_product_d4_assign(pairs, &mut output);
            for (sink, value) in sink.iter_mut().zip(output.iter()) {
                *sink += *value;
            }
        }
        let _sink = black_box(sink);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: linear_product_d4_memory(PRODUCTS),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered degree-4 linear product kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("degree-4 linear product evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical degree-4 linear product evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    fn linear_product_d4_products<F, FromU64>(count: usize, from_u64: FromU64) -> Vec<[(F, F); 4]>
    where
        FromU64: Fn(u64) -> F,
    {
        (0..count)
            .map(|product| {
                core::array::from_fn(|factor| {
                    (
                        from_u64(190_000 + product as u64 * 17 + factor as u64 * 3),
                        from_u64(210_000 + product as u64 * 19 + factor as u64 * 5),
                    )
                })
            })
            .collect()
    }
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_linear_product_d8_kernel_evidence() {
    use jolt_backends::cpu::field;
    use jolt_core::{
        ark_bn254::Fr as CoreFr, field::JoltField,
        subprotocols::mles_product_sum::eval_linear_prod_assign,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_linear_product_d8";
    const BENCHMARK: &str = "cpu_field/linear_product_d8";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-FLD-003"];
    const PRODUCTS: usize = 1 << 19;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();

    let core_products = linear_product_d8_products::<CoreFr, _>(PRODUCTS, |value| {
        <CoreFr as JoltField>::from_u64(value)
    });
    let modular_products = linear_product_d8_products(PRODUCTS, Fr::from_u64);

    let core = measure_samples(samples, || {
        let mut sink = [<CoreFr as JoltField>::from_u64(0); 8];
        let mut output = vec![<CoreFr as JoltField>::from_u64(0); 8];
        for pairs in &core_products {
            eval_linear_prod_assign(pairs, &mut output);
            for (sink, value) in sink.iter_mut().zip(output.iter()) {
                *sink += *value;
            }
        }
        let _sink = black_box(sink);
    });

    let modular = measure_samples(samples, || {
        let mut sink = [Fr::from_u64(0); 8];
        let mut output = vec![Fr::from_u64(0); 8];
        for pairs in &modular_products {
            field::eval_linear_product_d8_assign(pairs, &mut output);
            for (sink, value) in sink.iter_mut().zip(output.iter()) {
                *sink += *value;
            }
        }
        let _sink = black_box(sink);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: linear_product_d8_memory(PRODUCTS),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered degree-8 linear product kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("degree-8 linear product evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical degree-8 linear product evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    fn linear_product_d8_products<F, FromU64>(count: usize, from_u64: FromU64) -> Vec<[(F, F); 8]>
    where
        FromU64: Fn(u64) -> F,
    {
        (0..count)
            .map(|product| {
                core::array::from_fn(|factor| {
                    (
                        from_u64(230_000 + product as u64 * 17 + factor as u64 * 3),
                        from_u64(250_000 + product as u64 * 19 + factor as u64 * 5),
                    )
                })
            })
            .collect()
    }
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_linear_product_d16_kernel_evidence() {
    use jolt_backends::cpu::field;
    use jolt_core::{
        ark_bn254::Fr as CoreFr, field::JoltField,
        subprotocols::mles_product_sum::eval_linear_prod_assign,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_linear_product_d16";
    const BENCHMARK: &str = "cpu_field/linear_product_d16";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-FLD-003"];
    const PRODUCTS: usize = 1 << 18;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();

    let core_products = linear_product_d16_products::<CoreFr, _>(PRODUCTS, |value| {
        <CoreFr as JoltField>::from_u64(value)
    });
    let modular_products = linear_product_d16_products(PRODUCTS, Fr::from_u64);

    let core = measure_samples(samples, || {
        let mut sink = [<CoreFr as JoltField>::from_u64(0); 16];
        let mut output = vec![<CoreFr as JoltField>::from_u64(0); 16];
        for pairs in &core_products {
            eval_linear_prod_assign(pairs, &mut output);
            for (sink, value) in sink.iter_mut().zip(output.iter()) {
                *sink += *value;
            }
        }
        let _sink = black_box(sink);
    });

    let modular = measure_samples(samples, || {
        let mut sink = [Fr::from_u64(0); 16];
        let mut output = vec![Fr::from_u64(0); 16];
        for pairs in &modular_products {
            field::eval_linear_product_d16_assign(pairs, &mut output);
            for (sink, value) in sink.iter_mut().zip(output.iter()) {
                *sink += *value;
            }
        }
        let _sink = black_box(sink);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: linear_product_d16_memory(PRODUCTS),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered degree-16 linear product kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("degree-16 linear product evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical degree-16 linear product evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    fn linear_product_d16_products<F, FromU64>(count: usize, from_u64: FromU64) -> Vec<[(F, F); 16]>
    where
        FromU64: Fn(u64) -> F,
    {
        (0..count)
            .map(|product| {
                core::array::from_fn(|factor| {
                    (
                        from_u64(270_000 + product as u64 * 17 + factor as u64 * 3),
                        from_u64(290_000 + product as u64 * 19 + factor as u64 * 5),
                    )
                })
            })
            .collect()
    }
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_linear_product_d32_kernel_evidence() {
    use jolt_backends::cpu::field;
    use jolt_core::{
        ark_bn254::Fr as CoreFr, field::JoltField,
        subprotocols::mles_product_sum::eval_linear_prod_assign,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_linear_product_d32";
    const BENCHMARK: &str = "cpu_field/linear_product_d32";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-FLD-003"];
    const PRODUCTS: usize = 1 << 17;

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();

    let core_products = linear_product_d32_products::<CoreFr, _>(PRODUCTS, |value| {
        <CoreFr as JoltField>::from_u64(value)
    });
    let modular_products = linear_product_d32_products(PRODUCTS, Fr::from_u64);

    let core = measure_samples(samples, || {
        let mut sink = [<CoreFr as JoltField>::from_u64(0); 32];
        let mut output = vec![<CoreFr as JoltField>::from_u64(0); 32];
        for pairs in &core_products {
            eval_linear_prod_assign(pairs, &mut output);
            for (sink, value) in sink.iter_mut().zip(output.iter()) {
                *sink += *value;
            }
        }
        let _sink = black_box(sink);
    });

    let modular = measure_samples(samples, || {
        let mut sink = [Fr::from_u64(0); 32];
        let mut output = vec![Fr::from_u64(0); 32];
        for pairs in &modular_products {
            field::eval_linear_product_d32_assign(pairs, &mut output);
            for (sink, value) in sink.iter_mut().zip(output.iter()) {
                *sink += *value;
            }
        }
        let _sink = black_box(sink);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: linear_product_d32_memory(PRODUCTS),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered degree-32 linear product kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("degree-32 linear product evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical degree-32 linear product evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );

    fn linear_product_d32_products<F, FromU64>(count: usize, from_u64: FromU64) -> Vec<[(F, F); 32]>
    where
        FromU64: Fn(u64) -> F,
    {
        (0..count)
            .map(|product| {
                core::array::from_fn(|factor| {
                    (
                        from_u64(310_000 + product as u64 * 23 + factor as u64 * 7),
                        from_u64(330_000 + product as u64 * 29 + factor as u64 * 11),
                    )
                })
            })
            .collect()
    }
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_streaming_schedule_kernel_evidence() {
    use jolt_backends::cpu::schedule as modular_schedule;
    use jolt_core::subprotocols::streaming_schedule as core_schedule;
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_streaming_schedule";
    const BENCHMARK: &str = "cpu_sumcheck/streaming_schedule";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-SC-009", "OPT-SC-010"];
    const ITERS: usize = 16_384;
    const SHAPES: [(usize, usize); 5] = [(20, 2), (40, 2), (64, 2), (64, 3), (96, 4)];

    fn core_fingerprint<S: core_schedule::StreamingSchedule>(schedule: &S) -> usize {
        let mut total = schedule.switch_over_point() ^ schedule.num_rounds();
        for round in 0..schedule.num_rounds() {
            total = total.wrapping_mul(131).wrapping_add(
                usize::from(schedule.is_window_start(round)) + schedule.num_unbound_vars(round),
            );
        }
        total
    }

    fn modular_fingerprint<S: modular_schedule::StreamingSchedule>(schedule: &S) -> usize {
        let mut total = schedule.switch_over_point() ^ schedule.num_rounds();
        for round in 0..schedule.num_rounds() {
            total = total.wrapping_mul(131).wrapping_add(
                usize::from(schedule.is_window_start(round)) + schedule.num_unbound_vars(round),
            );
        }
        total
    }

    for &(rounds, degree) in &SHAPES {
        let core = core_schedule::HalfSplitSchedule::new(rounds, degree);
        let modular = modular_schedule::HalfSplitSchedule::new(rounds, degree);
        assert_eq!(core_fingerprint(&core), modular_fingerprint(&modular));
        for round in 0..rounds {
            assert_eq!(
                core_schedule::HalfSplitSchedule::optimal_window_size(round, degree),
                modular_schedule::HalfSplitSchedule::optimal_window_size(round, degree)
            );
        }
    }

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();

    let core = measure_samples(samples, || {
        let mut total = 0usize;
        for _ in 0..ITERS {
            for &(rounds, degree) in &SHAPES {
                let schedule =
                    core_schedule::HalfSplitSchedule::new(black_box(rounds), black_box(degree));
                total ^= core_fingerprint(&schedule);
            }
            let linear = core_schedule::LinearOnlySchedule::new(black_box(64));
            total ^= core_fingerprint(&linear);
        }
        let _total = black_box(total);
    });

    let modular = measure_samples(samples, || {
        let mut total = 0usize;
        for _ in 0..ITERS {
            for &(rounds, degree) in &SHAPES {
                let schedule =
                    modular_schedule::HalfSplitSchedule::new(black_box(rounds), black_box(degree));
                total ^= modular_fingerprint(&schedule);
            }
            let linear = modular_schedule::LinearOnlySchedule::new(black_box(64));
            total ^= modular_fingerprint(&linear);
        }
        let _total = black_box(total);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: streaming_schedule_memory(),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered streaming schedule kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("streaming schedule kernel evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical streaming schedule evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_ra_delayed_materialization_kernel_evidence() {
    use std::sync::Arc;

    use jolt_backends::cpu::ra as modular_ra;
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::{
            multilinear_polynomial::{BindingOrder as CoreBindingOrder, PolynomialBinding as _},
            ra_poly::RaPolynomial as CoreRaPolynomial,
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::BindingOrder;
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_ra_delayed_materialization";
    const BENCHMARK: &str = "cpu_sumcheck/ra_delayed_materialization";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-RA-007"];
    const LOG_LEN: usize = 19;
    const K: usize = 256;
    const ITERS: usize = 8;

    fn core_field_from_index(index: usize) -> CoreFr {
        <CoreFr as JoltField>::from_u64((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    fn lookup_indices(log_len: usize, k: usize) -> Vec<Option<u8>> {
        (0..(1usize << log_len))
            .map(|index| {
                if index % 17 == 0 {
                    None
                } else {
                    Some(((index * 37 + (index >> 3) * 13 + 11) % k) as u8)
                }
            })
            .collect()
    }

    fn core_eq_evals(k: usize) -> Vec<CoreFr> {
        (0..k)
            .map(|index| core_field_from_index(910_000 + index))
            .collect()
    }

    fn modular_eq_evals(core: &[CoreFr]) -> Vec<Fr> {
        core.iter().copied().map(Fr::from).collect()
    }

    fn core_fingerprint(poly: &CoreRaPolynomial<u8, CoreFr>) -> CoreFr {
        (0..poly.len()).step_by((poly.len() / 64).max(1)).fold(
            <CoreFr as JoltField>::from_u64(poly.len() as u64),
            |acc, index| acc + poly.get_bound_coeff(index),
        )
    }

    fn modular_fingerprint(poly: &modular_ra::RaPolynomial<u8, Fr>) -> Fr {
        (0..poly.len())
            .step_by((poly.len() / 64).max(1))
            .fold(Fr::from_u64(poly.len() as u64), |acc, index| {
                acc + poly.get_bound_coeff(index)
            })
    }

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let core_indices = Arc::new(lookup_indices(LOG_LEN, K));
    let modular_indices = Arc::new(core_indices.as_ref().clone());
    let core_eq = core_eq_evals(K);
    let modular_eq = modular_eq_evals(&core_eq);
    let core_challenges = [
        <CoreFr as JoltField>::Challenge::from(920_001u128),
        <CoreFr as JoltField>::Challenge::from(920_003u128),
        <CoreFr as JoltField>::Challenge::from(920_009u128),
        <CoreFr as JoltField>::Challenge::from(920_027u128),
    ];
    let modular_challenges = core_challenges.map(|challenge| Fr::from(CoreFr::from(challenge)));

    let mut core_reference =
        CoreRaPolynomial::<u8, CoreFr>::new(core_indices.clone(), core_eq.clone());
    let mut modular_reference =
        modular_ra::RaPolynomial::<u8, Fr>::new(modular_indices.clone(), modular_eq.clone());
    for (&core_challenge, &modular_challenge) in
        core_challenges.iter().zip(modular_challenges.iter())
    {
        core_reference.bind_parallel(core_challenge, CoreBindingOrder::LowToHigh);
        modular_reference.bind_parallel(modular_challenge, BindingOrder::LowToHigh);
    }
    assert_eq!(
        core_fingerprint(&core_reference),
        CoreFr::from(modular_fingerprint(&modular_reference))
    );

    let core = measure_samples(samples, || {
        let mut total = <CoreFr as JoltField>::from_u64(0);
        for _ in 0..ITERS {
            let mut poly =
                CoreRaPolynomial::<u8, CoreFr>::new(core_indices.clone(), core_eq.clone());
            for &challenge in &core_challenges {
                poly.bind_parallel(black_box(challenge), CoreBindingOrder::LowToHigh);
            }
            total += core_fingerprint(&poly);
        }
        let _total = black_box(total);
    });

    let modular = measure_samples(samples, || {
        let mut total = Fr::from_u64(0);
        for _ in 0..ITERS {
            let mut poly = modular_ra::RaPolynomial::<u8, Fr>::new(
                modular_indices.clone(),
                modular_eq.clone(),
            );
            for &challenge in &modular_challenges {
                poly.bind_parallel(black_box(challenge), BindingOrder::LowToHigh);
            }
            total += modular_fingerprint(&poly);
        }
        let _total = black_box(total);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: ra_delayed_materialization_memory(LOG_LEN, K),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered RA delayed materialization kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("RA delayed materialization evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical RA delayed materialization evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_shared_ra_delayed_materialization_kernel_evidence() {
    use jolt_backends::cpu::ra as modular_ra;
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::{
            multilinear_polynomial::BindingOrder as CoreBindingOrder,
            shared_ra_polys::{
                RaIndices as CoreRaCycleIndices, SharedRaPolynomials as CoreSharedRaPolynomials,
                MAX_BYTECODE_D as CORE_MAX_BYTECODE_D, MAX_INSTRUCTION_D as CORE_MAX_INSTRUCTION_D,
                MAX_RAM_D as CORE_MAX_RAM_D,
            },
        },
        zkvm::config::{OneHotConfig as CoreOneHotConfig, OneHotParams as CoreOneHotParams},
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::BindingOrder;
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_shared_ra_delayed_materialization";
    const BENCHMARK: &str = "cpu_sumcheck/shared_ra_delayed_materialization";
    const OPTIMIZATION_IDS: [&str; 4] = ["OPT-RA-001", "OPT-RA-008", "OPT-RA-009", "OPT-RA-010"];
    const LOG_LEN: usize = 16;
    const ITERS: usize = 4;

    fn core_field_from_index(index: usize) -> CoreFr {
        <CoreFr as JoltField>::from_u64((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    fn core_params() -> CoreOneHotParams {
        CoreOneHotParams::from_config(
            &CoreOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
            1usize << 24,
            1usize << 32,
        )
    }

    fn modular_layout(core: &CoreOneHotParams) -> modular_ra::RaFamilyLayout {
        modular_ra::RaFamilyLayout::new(
            core.k_chunk,
            core.instruction_d,
            core.bytecode_d,
            core.ram_d,
        )
    }

    fn core_indices(log_len: usize, params: &CoreOneHotParams) -> Vec<CoreRaCycleIndices> {
        (0..(1usize << log_len))
            .map(|row| {
                let mut instruction = [0u8; CORE_MAX_INSTRUCTION_D];
                let mut bytecode = [0u8; CORE_MAX_BYTECODE_D];
                let mut ram = [None; CORE_MAX_RAM_D];
                for (chunk, value) in instruction
                    .iter_mut()
                    .enumerate()
                    .take(params.instruction_d)
                {
                    *value = ((row * 13 + chunk * 5 + (row >> 2)) % params.k_chunk) as u8;
                }
                for (chunk, value) in bytecode.iter_mut().enumerate().take(params.bytecode_d) {
                    *value = ((row * 7 + chunk * 11 + (row >> 3)) % params.k_chunk) as u8;
                }
                for (chunk, value) in ram.iter_mut().enumerate().take(params.ram_d) {
                    if (row + chunk) % 5 != 0 {
                        *value = Some(((row * 3 + chunk * 17 + (row >> 1)) % params.k_chunk) as u8);
                    }
                }
                CoreRaCycleIndices {
                    instruction,
                    bytecode,
                    ram,
                }
            })
            .collect()
    }

    fn modular_indices(core: &[CoreRaCycleIndices]) -> Vec<modular_ra::RaCycleIndices> {
        core.iter()
            .map(|entry| modular_ra::RaCycleIndices {
                instruction: entry.instruction,
                bytecode: entry.bytecode,
                ram: entry.ram,
            })
            .collect()
    }

    fn core_tables(params: &CoreOneHotParams) -> Vec<Vec<CoreFr>> {
        let num_polys = params.instruction_d + params.bytecode_d + params.ram_d;
        (0..num_polys)
            .map(|poly_idx| {
                (0..params.k_chunk)
                    .map(|entry| core_field_from_index(930_000 + poly_idx * 101 + entry * 19))
                    .collect()
            })
            .collect()
    }

    fn modular_tables(core: &[Vec<CoreFr>]) -> Vec<Vec<Fr>> {
        core.iter()
            .map(|table| table.iter().copied().map(Fr::from).collect())
            .collect()
    }

    fn core_fingerprint(polys: &CoreSharedRaPolynomials<CoreFr>) -> CoreFr {
        let mut total = <CoreFr as JoltField>::from_u64((polys.num_polys() ^ polys.len()) as u64);
        for poly_idx in 0..polys.num_polys() {
            for row in (0..polys.len()).step_by((polys.len() / 32).max(1)) {
                total += polys.get_bound_coeff(poly_idx, row);
            }
        }
        total
    }

    fn modular_fingerprint(polys: &modular_ra::SharedRaPolynomials<Fr>) -> Fr {
        let mut total = Fr::from_u64((polys.num_polys() ^ polys.len()) as u64);
        for poly_idx in 0..polys.num_polys() {
            for row in (0..polys.len()).step_by((polys.len() / 32).max(1)) {
                total += polys.get_bound_coeff(poly_idx, row);
            }
        }
        total
    }

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let core_params = core_params();
    let modular_layout = modular_layout(&core_params);
    let core_indices = core_indices(LOG_LEN, &core_params);
    let modular_indices = modular_indices(&core_indices);
    let core_tables = core_tables(&core_params);
    let modular_tables = modular_tables(&core_tables);
    let core_challenges = [
        <CoreFr as JoltField>::Challenge::from(940_001u128),
        <CoreFr as JoltField>::Challenge::from(940_003u128),
        <CoreFr as JoltField>::Challenge::from(940_009u128),
        <CoreFr as JoltField>::Challenge::from(940_027u128),
    ];
    let modular_challenges = core_challenges.map(|challenge| Fr::from(CoreFr::from(challenge)));

    let mut core_reference = CoreSharedRaPolynomials::<CoreFr>::new(
        core_tables.clone(),
        core_indices.clone(),
        core_params.clone(),
    );
    let mut modular_reference = modular_ra::SharedRaPolynomials::<Fr>::new(
        modular_tables.clone(),
        modular_indices.clone(),
        modular_layout,
    );
    for (&core_challenge, &modular_challenge) in
        core_challenges.iter().zip(modular_challenges.iter())
    {
        core_reference.bind_in_place(core_challenge, CoreBindingOrder::LowToHigh);
        modular_reference.bind_in_place(modular_challenge, BindingOrder::LowToHigh);
    }
    assert_eq!(
        core_fingerprint(&core_reference),
        CoreFr::from(modular_fingerprint(&modular_reference))
    );

    let core = measure_samples(samples, || {
        let mut total = <CoreFr as JoltField>::from_u64(0);
        for _ in 0..ITERS {
            let mut polys = CoreSharedRaPolynomials::<CoreFr>::new(
                core_tables.clone(),
                core_indices.clone(),
                core_params.clone(),
            );
            for &challenge in &core_challenges {
                polys.bind_in_place(black_box(challenge), CoreBindingOrder::LowToHigh);
            }
            total += core_fingerprint(&polys);
        }
        let _total = black_box(total);
    });

    let modular = measure_samples(samples, || {
        let mut total = Fr::from_u64(0);
        for _ in 0..ITERS {
            let mut polys = modular_ra::SharedRaPolynomials::<Fr>::new(
                modular_tables.clone(),
                modular_indices.clone(),
                modular_layout,
            );
            for &challenge in &modular_challenges {
                polys.bind_in_place(black_box(challenge), BindingOrder::LowToHigh);
            }
            total += modular_fingerprint(&polys);
        }
        let _total = black_box(total);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: shared_ra_delayed_materialization_memory(LOG_LEN, modular_layout),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered shared RA delayed materialization kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("shared RA delayed materialization evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical shared RA delayed materialization evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_ra_pushforward_kernel_evidence() {
    use jolt_backends::cpu::ra as modular_ra;
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        poly::shared_ra_polys::{
            compute_all_G, compute_all_G_and_ra_indices, RaIndices as CoreRaCycleIndices,
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{
        load_stage0_commitment_kernel_benchmark_fixture, validate_kernel_benchmark_evidence,
        FeatureMode, FixtureKind, FixtureRequest, KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_ra_pushforward";
    const BENCHMARK: &str = "cpu_sumcheck/ra_pushforward";
    const OPTIMIZATION_IDS: [&str; 4] = ["OPT-RA-003", "OPT-RA-004", "OPT-RA-005", "OPT-RA-006"];
    const ITERS: usize = 4;

    fn modular_layout(
        params: &jolt_core::zkvm::config::OneHotParams,
    ) -> modular_ra::RaFamilyLayout {
        modular_ra::RaFamilyLayout::new(
            params.k_chunk,
            params.instruction_d,
            params.bytecode_d,
            params.ram_d,
        )
    }

    fn modular_indices(core: &[CoreRaCycleIndices]) -> Vec<modular_ra::RaCycleIndices> {
        core.iter()
            .map(|entry| modular_ra::RaCycleIndices {
                instruction: entry.instruction,
                bytecode: entry.bytecode,
                ram: entry.ram,
            })
            .collect()
    }

    fn core_fingerprint(tables: &[Vec<CoreFr>]) -> CoreFr {
        tables.iter().enumerate().fold(
            <CoreFr as JoltField>::from_u64(tables.len() as u64),
            |mut acc, (poly_idx, table)| {
                for (entry, &value) in table.iter().enumerate().step_by((table.len() / 8).max(1)) {
                    acc += value + <CoreFr as JoltField>::from_u64((poly_idx ^ entry) as u64);
                }
                acc
            },
        )
    }

    fn modular_fingerprint(tables: &[Vec<Fr>]) -> Fr {
        tables.iter().enumerate().fold(
            Fr::from_u64(tables.len() as u64),
            |mut acc, (poly_idx, table)| {
                for (entry, &value) in table.iter().enumerate().step_by((table.len() / 8).max(1)) {
                    acc += value + Fr::from_u64((poly_idx ^ entry) as u64);
                }
                acc
            },
        )
    }

    fn assert_tables_match(core: &[Vec<CoreFr>], modular: &[Vec<Fr>]) {
        assert_eq!(core.len(), modular.len());
        for (core_table, modular_table) in core.iter().zip(modular) {
            assert_eq!(core_table.len(), modular_table.len());
            for (&core_value, &modular_value) in core_table.iter().zip(modular_table) {
                let modular_value: CoreFr = modular_value.into();
                assert_eq!(core_value, modular_value);
            }
        }
    }

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let fixture_request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage0_commitment_kernel_benchmark_fixture(&fixture_request)
        .expect("load stage0 commitment kernel benchmark fixture");
    let shape = fixture.shape().expect("stage0 commitment fixture shape");
    let one_hot_params = fixture.core_one_hot_params();
    let layout = modular_layout(&one_hot_params);
    let core_r_cycle = (0..shape.log_t)
        .map(|index| <CoreFr as JoltField>::Challenge::from(960_000u128 + index as u128 * 17))
        .collect::<Vec<_>>();
    let modular_r_cycle = core_r_cycle
        .iter()
        .copied()
        .map(|challenge| Fr::from(CoreFr::from(challenge)))
        .collect::<Vec<_>>();

    let (core_reference, core_indices) = compute_all_G_and_ra_indices::<CoreFr>(
        fixture.core_trace(),
        fixture.core_bytecode(),
        fixture.core_memory_layout(),
        &one_hot_params,
        &core_r_cycle,
    );
    let modular_indices = modular_indices(&core_indices);
    let modular_reference =
        modular_ra::pushforward_indices(&modular_indices, layout, &modular_r_cycle);
    assert_tables_match(&core_reference, &modular_reference);
    assert_eq!(
        core_fingerprint(&core_reference),
        CoreFr::from(modular_fingerprint(&modular_reference))
    );

    let core = measure_samples(samples, || {
        let mut total = <CoreFr as JoltField>::from_u64(0);
        for _ in 0..ITERS {
            let tables = compute_all_G::<CoreFr>(
                black_box(fixture.core_trace()),
                black_box(fixture.core_bytecode()),
                black_box(fixture.core_memory_layout()),
                black_box(&one_hot_params),
                black_box(&core_r_cycle),
            );
            total += core_fingerprint(&tables);
        }
        let _total = black_box(total);
    });

    let modular = measure_samples(samples, || {
        let mut total = Fr::from_u64(0);
        for _ in 0..ITERS {
            let tables = modular_ra::pushforward_indices(
                black_box(&modular_indices),
                layout,
                black_box(&modular_r_cycle),
            );
            total += modular_fingerprint(&tables);
        }
        let _total = black_box(total);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: ra_pushforward_memory(shape.log_t, layout),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered RA pushforward kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("RA pushforward evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical RA pushforward evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_read_write_one_hot_coeff_lookup_kernel_evidence() {
    use jolt_backends::cpu::read_write_matrix as modular_read_write;
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        subprotocols::read_write_matrix::{
            LookupTableIndex as CoreLookupTableIndex, OneHotCoeffLookupTable as CoreLookupTable,
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_read_write_one_hot_coeff_lookup";
    const BENCHMARK: &str = "cpu_sumcheck/read_write_one_hot_coeff_lookup";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-RW-007"];
    const ITERS: usize = 256;
    const SAMPLE_INDICES: [u16; 10] = [0, 1, 2, 3, 17, 251, 4095, 16383, 32767, 65535];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();

    let core_initial = [
        <CoreFr as JoltField>::from_u64(0),
        <CoreFr as JoltField>::from_u64(1),
        <CoreFr as JoltField>::from_u64(210_001),
        <CoreFr as JoltField>::from_u64(210_019),
    ];
    let core_challenges = [
        <CoreFr as JoltField>::Challenge::from(211_001u128),
        <CoreFr as JoltField>::Challenge::from(211_019u128),
        <CoreFr as JoltField>::Challenge::from(211_043u128),
    ];
    let modular_initial = [
        Fr::from_u64(0),
        Fr::from_u64(1),
        Fr::from_u64(210_001),
        Fr::from_u64(210_019),
    ];
    let modular_challenges = core_challenges.map(|challenge| Fr::from(CoreFr::from(challenge)));

    let mut core_reference = CoreLookupTable::new(core_initial.to_vec());
    let mut modular_reference = modular_read_write::OneHotCoeffTable::new(modular_initial.to_vec());
    for (&core_challenge, &modular_challenge) in
        core_challenges.iter().zip(modular_challenges.iter())
    {
        core_reference.bind(core_challenge);
        modular_reference.bind(modular_challenge);
    }
    assert!(core_reference.is_saturated());
    assert!(modular_reference.is_saturated());
    for &index in &SAMPLE_INDICES {
        let core_value = core_reference[CoreLookupTableIndex(index)];
        let modular_value: CoreFr =
            modular_reference[modular_read_write::OneHotCoeffIndex(index)].into();
        assert_eq!(core_value, modular_value);
    }

    let core = measure_samples(samples, || {
        let mut total = <CoreFr as JoltField>::from_u64(0);
        for _ in 0..ITERS {
            let mut table = CoreLookupTable::new(core_initial.to_vec());
            for &challenge in &core_challenges {
                table.bind(black_box(challenge));
            }
            for &index in &SAMPLE_INDICES {
                total += table[CoreLookupTableIndex(index)];
            }
        }
        let _total = black_box(total);
    });

    let modular = measure_samples(samples, || {
        let mut total = Fr::from_u64(0);
        for _ in 0..ITERS {
            let mut table = modular_read_write::OneHotCoeffTable::new(modular_initial.to_vec());
            for &challenge in &modular_challenges {
                table.bind(black_box(challenge));
            }
            for &index in &SAMPLE_INDICES {
                total += table[modular_read_write::OneHotCoeffIndex(index)];
            }
        }
        let _total = black_box(total);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: read_write_one_hot_coeff_lookup_memory(),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered read-write one-hot coefficient lookup kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect(
                "read-write one-hot coefficient lookup evidence should pass the canonical gate",
            );
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical read-write one-hot coefficient lookup evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_read_write_cycle_major_bind_kernel_evidence() {
    use jolt_backends::cpu::read_write_matrix as modular_read_write;
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        subprotocols::read_write_matrix::{
            AddressMajorMatrixEntry as CoreAddressMajorEntryTrait,
            CycleMajorMatrixEntry as CoreCycleMajorEntryTrait,
            OneHotCoeffLookupTable as CoreLookupTable,
            ReadWriteMatrixCycleMajor as CoreCycleMajorMatrix,
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_read_write_cycle_major_bind";
    const BENCHMARK: &str = "cpu_sumcheck/read_write_cycle_major_bind";
    const OPTIMIZATION_IDS: [&str; 5] = [
        "OPT-RW-001",
        "OPT-RW-002",
        "OPT-RW-004",
        "OPT-RW-005",
        "OPT-RW-006",
    ];
    const ROW_PAIRS: usize = 2;
    const TERMS_PER_ROW: usize = 40_000;
    const ITERS: usize = 8;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct CoreCycleEntry {
        row: usize,
        column: usize,
        value: CoreFr,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct CoreAddressEntry {
        row: usize,
        column: usize,
        value: CoreFr,
    }

    impl CoreCycleMajorEntryTrait<CoreFr> for CoreCycleEntry {
        type AddressMajor = CoreAddressEntry;

        fn row(&self) -> usize {
            self.row
        }

        fn column(&self) -> usize {
            self.column
        }

        fn bind_entries(
            even: Option<&Self>,
            odd: Option<&Self>,
            r: <CoreFr as JoltField>::Challenge,
            _: Option<&CoreLookupTable<CoreFr>>,
            _: Option<&CoreLookupTable<CoreFr>>,
        ) -> Self {
            let r: CoreFr = r.into();
            match (even, odd) {
                (Some(even), Some(odd)) => Self {
                    row: even.row / 2,
                    column: even.column,
                    value: even.value + r * (odd.value - even.value),
                },
                (Some(even), None) => Self {
                    row: even.row / 2,
                    column: even.column,
                    value: (<CoreFr as JoltField>::from_u64(1) - r) * even.value,
                },
                (None, Some(odd)) => Self {
                    row: odd.row / 2,
                    column: odd.column,
                    value: r * odd.value,
                },
                (None, None) => unreachable!("cycle-major bind requires at least one entry"),
            }
        }

        fn compute_evals(
            _: Option<&Self>,
            _: Option<&Self>,
            _: [CoreFr; 2],
            _: CoreFr,
            _: Option<&CoreLookupTable<CoreFr>>,
            _: Option<&CoreLookupTable<CoreFr>>,
        ) -> [<CoreFr as JoltField>::UnreducedProduct; 2] {
            let zero = <CoreFr as JoltField>::from_u64(0);
            [zero.mul_to_product(zero), zero.mul_to_product(zero)]
        }

        fn to_address_major(
            self,
            _: Option<&CoreLookupTable<CoreFr>>,
            _: Option<&CoreLookupTable<CoreFr>>,
        ) -> Self::AddressMajor {
            CoreAddressEntry {
                row: self.row,
                column: self.column,
                value: self.value,
            }
        }
    }

    impl CoreAddressMajorEntryTrait<CoreFr> for CoreAddressEntry {
        fn row(&self) -> usize {
            self.row
        }

        fn column(&self) -> usize {
            self.column
        }

        fn prev_val(&self) -> CoreFr {
            self.value
        }

        fn next_val(&self) -> CoreFr {
            self.value
        }

        fn bind_entries(
            even: Option<&Self>,
            odd: Option<&Self>,
            _: CoreFr,
            _: CoreFr,
            r: <CoreFr as JoltField>::Challenge,
        ) -> Self {
            let r: CoreFr = r.into();
            match (even, odd) {
                (Some(even), Some(odd)) => Self {
                    row: even.row,
                    column: even.column / 2,
                    value: even.value + r * (odd.value - even.value),
                },
                (Some(even), None) => Self {
                    row: even.row,
                    column: even.column / 2,
                    value: (<CoreFr as JoltField>::from_u64(1) - r) * even.value,
                },
                (None, Some(odd)) => Self {
                    row: odd.row,
                    column: odd.column / 2,
                    value: r * odd.value,
                },
                (None, None) => unreachable!("address-major bind requires at least one entry"),
            }
        }

        fn compute_evals(
            _: Option<&Self>,
            _: Option<&Self>,
            _: CoreFr,
            _: CoreFr,
            _: CoreFr,
            _: CoreFr,
            _: CoreFr,
        ) -> [<CoreFr as JoltField>::UnreducedProduct; 2] {
            let zero = <CoreFr as JoltField>::from_u64(0);
            [zero.mul_to_product(zero), zero.mul_to_product(zero)]
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct ModularCycleEntry {
        row: usize,
        column: usize,
        value: Fr,
    }

    impl modular_read_write::CycleMajorMatrixEntry<Fr> for ModularCycleEntry {
        fn row(&self) -> usize {
            self.row
        }

        fn column(&self) -> usize {
            self.column
        }

        fn bind_entries(
            even: Option<&Self>,
            odd: Option<&Self>,
            r: Fr,
            _: Option<&modular_read_write::OneHotCoeffTable<Fr>>,
            _: Option<&modular_read_write::OneHotCoeffTable<Fr>>,
        ) -> Self {
            match (even, odd) {
                (Some(even), Some(odd)) => Self {
                    row: even.row / 2,
                    column: even.column,
                    value: even.value + r * (odd.value - even.value),
                },
                (Some(even), None) => Self {
                    row: even.row / 2,
                    column: even.column,
                    value: (Fr::from_u64(1) - r) * even.value,
                },
                (None, Some(odd)) => Self {
                    row: odd.row / 2,
                    column: odd.column,
                    value: r * odd.value,
                },
                (None, None) => unreachable!("cycle-major bind requires at least one entry"),
            }
        }
    }

    fn core_field_from_index(index: usize) -> CoreFr {
        <CoreFr as JoltField>::from_u64((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    fn core_entries(row_pairs: usize, terms_per_row: usize) -> Vec<CoreCycleEntry> {
        let mut entries = Vec::with_capacity(row_pairs * terms_per_row * 2);
        for pair in 0..row_pairs {
            let even_row = 2 * pair;
            let odd_row = even_row + 1;
            for index in 0..terms_per_row {
                entries.push(CoreCycleEntry {
                    row: even_row,
                    column: index * 2,
                    value: core_field_from_index(230_000 + pair * terms_per_row + index),
                });
            }
            for index in 0..terms_per_row {
                entries.push(CoreCycleEntry {
                    row: odd_row,
                    column: if index % 3 == 0 {
                        index * 2
                    } else {
                        index * 2 + 1
                    },
                    value: core_field_from_index(330_000 + pair * terms_per_row + index),
                });
            }
        }
        entries
    }

    fn modular_entries(core_entries: &[CoreCycleEntry]) -> Vec<ModularCycleEntry> {
        core_entries
            .iter()
            .map(|entry| ModularCycleEntry {
                row: entry.row,
                column: entry.column,
                value: Fr::from(entry.value),
            })
            .collect()
    }

    fn bound_len(row_pairs: usize, terms_per_row: usize) -> usize {
        row_pairs * (terms_per_row + terms_per_row - terms_per_row.div_ceil(3))
    }

    fn core_fingerprint(entries: &[CoreCycleEntry]) -> CoreFr {
        entries.iter().step_by((entries.len() / 64).max(1)).fold(
            <CoreFr as JoltField>::from_u64(entries.len() as u64),
            |acc, entry| {
                acc + entry.value
                    + <CoreFr as JoltField>::from_u64((entry.row ^ entry.column) as u64)
            },
        )
    }

    fn modular_fingerprint(entries: &[ModularCycleEntry]) -> Fr {
        entries
            .iter()
            .step_by((entries.len() / 64).max(1))
            .fold(Fr::from_u64(entries.len() as u64), |acc, entry| {
                acc + entry.value + Fr::from_u64((entry.row ^ entry.column) as u64)
            })
    }

    fn assert_entries_match(
        core_entries: &[CoreCycleEntry],
        modular_entries: &[ModularCycleEntry],
    ) {
        assert_eq!(core_entries.len(), modular_entries.len());
        for (core, modular) in core_entries.iter().zip(modular_entries) {
            assert_eq!(core.row, modular.row);
            assert_eq!(core.column, modular.column);
            let modular_value: CoreFr = modular.value.into();
            assert_eq!(core.value, modular_value);
        }
    }

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let core_input = core_entries(ROW_PAIRS, TERMS_PER_ROW);
    let modular_input = modular_entries(&core_input);
    let core_challenge = <CoreFr as JoltField>::Challenge::from(430_001u128);
    let modular_challenge = Fr::from(CoreFr::from(core_challenge));

    let mut core_reference = CoreCycleMajorMatrix::<CoreFr, CoreCycleEntry>::default();
    core_reference.entries.clone_from(&core_input);
    core_reference.bind(core_challenge);
    let mut modular_reference =
        modular_read_write::ReadWriteMatrixCycleMajor::new(modular_input.clone());
    modular_reference.bind(modular_challenge);
    assert_entries_match(&core_reference.entries, &modular_reference.entries);
    assert_eq!(
        core_reference.entries.len(),
        bound_len(ROW_PAIRS, TERMS_PER_ROW)
    );

    let core = measure_samples(samples, || {
        let mut total = <CoreFr as JoltField>::from_u64(0);
        for _ in 0..ITERS {
            let mut matrix = CoreCycleMajorMatrix::<CoreFr, CoreCycleEntry>::default();
            matrix.entries.clone_from(&core_input);
            matrix.bind(black_box(core_challenge));
            total += core_fingerprint(&matrix.entries);
        }
        let _total = black_box(total);
    });

    let modular = measure_samples(samples, || {
        let mut total = Fr::from_u64(0);
        for _ in 0..ITERS {
            let mut matrix =
                modular_read_write::ReadWriteMatrixCycleMajor::new(modular_input.clone());
            matrix.bind(black_box(modular_challenge));
            total += modular_fingerprint(&matrix.entries);
        }
        let _total = black_box(total);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: read_write_cycle_major_bind_memory(
            core_input.len(),
            bound_len(ROW_PAIRS, TERMS_PER_ROW),
            ROW_PAIRS,
        ),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered read-write cycle-major bind kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("read-write cycle-major bind evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical read-write cycle-major bind evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_read_write_cycle_major_message_kernel_evidence() {
    use jolt_backends::cpu::read_write_matrix as modular_read_write;
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        subprotocols::read_write_matrix::{
            AddressMajorMatrixEntry as CoreAddressMajorEntryTrait,
            CycleMajorMatrixEntry as CoreCycleMajorEntryTrait,
            OneHotCoeffLookupTable as CoreLookupTable,
            ReadWriteMatrixCycleMajor as CoreCycleMajorMatrix,
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt, RingAccumulator, WithAccumulator};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_read_write_cycle_major_message";
    const BENCHMARK: &str = "cpu_sumcheck/read_write_cycle_major_message";
    const OPTIMIZATION_IDS: [&str; 1] = ["OPT-RW-008"];
    const TERMS_PER_ROW: usize = 40_000;
    const ITERS: usize = 16;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct CoreCycleEntry {
        row: usize,
        column: usize,
        value: CoreFr,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct CoreAddressEntry {
        row: usize,
        column: usize,
        value: CoreFr,
    }

    impl CoreCycleMajorEntryTrait<CoreFr> for CoreCycleEntry {
        type AddressMajor = CoreAddressEntry;

        fn row(&self) -> usize {
            self.row
        }

        fn column(&self) -> usize {
            self.column
        }

        fn bind_entries(
            even: Option<&Self>,
            odd: Option<&Self>,
            r: <CoreFr as JoltField>::Challenge,
            _: Option<&CoreLookupTable<CoreFr>>,
            _: Option<&CoreLookupTable<CoreFr>>,
        ) -> Self {
            let r: CoreFr = r.into();
            match (even, odd) {
                (Some(even), Some(odd)) => Self {
                    row: even.row / 2,
                    column: even.column,
                    value: even.value + r * (odd.value - even.value),
                },
                (Some(even), None) => Self {
                    row: even.row / 2,
                    column: even.column,
                    value: (<CoreFr as JoltField>::from_u64(1) - r) * even.value,
                },
                (None, Some(odd)) => Self {
                    row: odd.row / 2,
                    column: odd.column,
                    value: r * odd.value,
                },
                (None, None) => unreachable!("cycle-major bind requires at least one entry"),
            }
        }

        fn compute_evals(
            even: Option<&Self>,
            odd: Option<&Self>,
            inc_evals: [CoreFr; 2],
            gamma: CoreFr,
            _: Option<&CoreLookupTable<CoreFr>>,
            _: Option<&CoreLookupTable<CoreFr>>,
        ) -> [<CoreFr as JoltField>::UnreducedProduct; 2] {
            let [eval_at_zero, eval_slope] = core_entry_evals(even, odd);
            [
                eval_at_zero.mul_to_product(inc_evals[0] + gamma),
                eval_slope.mul_to_product(inc_evals[1] + gamma),
            ]
        }

        fn to_address_major(
            self,
            _: Option<&CoreLookupTable<CoreFr>>,
            _: Option<&CoreLookupTable<CoreFr>>,
        ) -> Self::AddressMajor {
            CoreAddressEntry {
                row: self.row,
                column: self.column,
                value: self.value,
            }
        }
    }

    impl CoreAddressMajorEntryTrait<CoreFr> for CoreAddressEntry {
        fn row(&self) -> usize {
            self.row
        }

        fn column(&self) -> usize {
            self.column
        }

        fn prev_val(&self) -> CoreFr {
            self.value
        }

        fn next_val(&self) -> CoreFr {
            self.value
        }

        fn bind_entries(
            even: Option<&Self>,
            odd: Option<&Self>,
            _: CoreFr,
            _: CoreFr,
            r: <CoreFr as JoltField>::Challenge,
        ) -> Self {
            let r: CoreFr = r.into();
            match (even, odd) {
                (Some(even), Some(odd)) => Self {
                    row: even.row,
                    column: even.column / 2,
                    value: even.value + r * (odd.value - even.value),
                },
                (Some(even), None) => Self {
                    row: even.row,
                    column: even.column / 2,
                    value: (<CoreFr as JoltField>::from_u64(1) - r) * even.value,
                },
                (None, Some(odd)) => Self {
                    row: odd.row,
                    column: odd.column / 2,
                    value: r * odd.value,
                },
                (None, None) => unreachable!("address-major bind requires at least one entry"),
            }
        }

        fn compute_evals(
            _: Option<&Self>,
            _: Option<&Self>,
            _: CoreFr,
            _: CoreFr,
            _: CoreFr,
            _: CoreFr,
            _: CoreFr,
        ) -> [<CoreFr as JoltField>::UnreducedProduct; 2] {
            let zero = <CoreFr as JoltField>::from_u64(0);
            [zero.mul_to_product(zero), zero.mul_to_product(zero)]
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct ModularCycleEntry {
        row: usize,
        column: usize,
        value: Fr,
    }

    impl modular_read_write::CycleMajorMatrixEntry<Fr> for ModularCycleEntry {
        fn row(&self) -> usize {
            self.row
        }

        fn column(&self) -> usize {
            self.column
        }

        fn bind_entries(
            even: Option<&Self>,
            odd: Option<&Self>,
            r: Fr,
            _: Option<&modular_read_write::OneHotCoeffTable<Fr>>,
            _: Option<&modular_read_write::OneHotCoeffTable<Fr>>,
        ) -> Self {
            match (even, odd) {
                (Some(even), Some(odd)) => Self {
                    row: even.row / 2,
                    column: even.column,
                    value: even.value + r * (odd.value - even.value),
                },
                (Some(even), None) => Self {
                    row: even.row / 2,
                    column: even.column,
                    value: (Fr::from_u64(1) - r) * even.value,
                },
                (None, Some(odd)) => Self {
                    row: odd.row / 2,
                    column: odd.column,
                    value: r * odd.value,
                },
                (None, None) => unreachable!("cycle-major bind requires at least one entry"),
            }
        }
    }

    impl modular_read_write::CycleMajorMessageEntry<Fr> for ModularCycleEntry {
        fn accumulate_evals(
            even: Option<&Self>,
            odd: Option<&Self>,
            inc_evals: [Fr; 2],
            gamma: Fr,
            accumulators: &mut [<Fr as WithAccumulator>::Accumulator; 2],
            _: Option<&modular_read_write::OneHotCoeffTable<Fr>>,
            _: Option<&modular_read_write::OneHotCoeffTable<Fr>>,
        ) {
            let [eval_at_zero, eval_slope] = modular_entry_evals(even, odd);
            accumulators[0].fmadd(eval_at_zero, inc_evals[0] + gamma);
            accumulators[1].fmadd(eval_slope, inc_evals[1] + gamma);
        }
    }

    fn core_field_from_index(index: usize) -> CoreFr {
        <CoreFr as JoltField>::from_u64((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    fn core_entries(terms_per_row: usize) -> Vec<CoreCycleEntry> {
        let mut entries = Vec::with_capacity(terms_per_row * 2);
        for index in 0..terms_per_row {
            entries.push(CoreCycleEntry {
                row: 0,
                column: index * 2,
                value: core_field_from_index(730_000 + index),
            });
        }
        for index in 0..terms_per_row {
            entries.push(CoreCycleEntry {
                row: 1,
                column: if index % 3 == 0 {
                    index * 2
                } else {
                    index * 2 + 1
                },
                value: core_field_from_index(830_000 + index),
            });
        }
        entries
    }

    fn modular_entries(core_entries: &[CoreCycleEntry]) -> Vec<ModularCycleEntry> {
        core_entries
            .iter()
            .map(|entry| ModularCycleEntry {
                row: entry.row,
                column: entry.column,
                value: Fr::from(entry.value),
            })
            .collect()
    }

    fn core_entry_evals(
        even: Option<&CoreCycleEntry>,
        odd: Option<&CoreCycleEntry>,
    ) -> [CoreFr; 2] {
        match (even, odd) {
            (Some(even), Some(odd)) => [even.value, odd.value - even.value],
            (Some(even), None) => [even.value, -even.value],
            (None, Some(odd)) => [<CoreFr as JoltField>::from_u64(0), odd.value],
            (None, None) => unreachable!("message contribution requires at least one entry"),
        }
    }

    fn modular_entry_evals(
        even: Option<&ModularCycleEntry>,
        odd: Option<&ModularCycleEntry>,
    ) -> [Fr; 2] {
        match (even, odd) {
            (Some(even), Some(odd)) => [even.value, odd.value - even.value],
            (Some(even), None) => [even.value, -even.value],
            (None, Some(odd)) => [Fr::from_u64(0), odd.value],
            (None, None) => unreachable!("message contribution requires at least one entry"),
        }
    }

    fn core_fingerprint(evals: [CoreFr; 2]) -> CoreFr {
        evals[0] + evals[1]
    }

    fn modular_fingerprint(evals: [Fr; 2]) -> Fr {
        evals[0] + evals[1]
    }

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let core_input = core_entries(TERMS_PER_ROW);
    let modular_input = modular_entries(&core_input);
    let core_inc = [
        core_field_from_index(840_001),
        core_field_from_index(840_003),
    ];
    let core_gamma = core_field_from_index(840_009);
    let modular_inc = core_inc.map(Fr::from);
    let modular_gamma = Fr::from(core_gamma);

    let mut core_reference = CoreCycleMajorMatrix::<CoreFr, CoreCycleEntry>::default();
    core_reference.entries.clone_from(&core_input);
    let (core_even, core_odd) = core_reference.entries.split_at(TERMS_PER_ROW);
    let core_reference_evals =
        core_reference.prover_message_contribution(core_even, core_odd, core_inc, core_gamma);
    let modular_reference =
        modular_read_write::ReadWriteMatrixCycleMajor::new(modular_input.clone());
    let (modular_even, modular_odd) = modular_reference.entries.split_at(TERMS_PER_ROW);
    let modular_reference_evals = modular_reference.prover_message_contribution(
        modular_even,
        modular_odd,
        modular_inc,
        modular_gamma,
    );
    assert_eq!(
        core_reference_evals[0],
        CoreFr::from(modular_reference_evals[0])
    );
    assert_eq!(
        core_reference_evals[1],
        CoreFr::from(modular_reference_evals[1])
    );

    let core = measure_samples(samples, || {
        let mut total = <CoreFr as JoltField>::from_u64(0);
        for _ in 0..ITERS {
            let mut matrix = CoreCycleMajorMatrix::<CoreFr, CoreCycleEntry>::default();
            matrix.entries.clone_from(&core_input);
            let (even, odd) = matrix.entries.split_at(TERMS_PER_ROW);
            total += core_fingerprint(
                matrix.prover_message_contribution(even, odd, core_inc, core_gamma),
            );
        }
        let _total = black_box(total);
    });

    let modular = measure_samples(samples, || {
        let mut total = Fr::from_u64(0);
        for _ in 0..ITERS {
            let matrix = modular_read_write::ReadWriteMatrixCycleMajor::new(modular_input.clone());
            let (even, odd) = matrix.entries.split_at(TERMS_PER_ROW);
            total += modular_fingerprint(matrix.prover_message_contribution(
                even,
                odd,
                modular_inc,
                modular_gamma,
            ));
        }
        let _total = black_box(total);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: read_write_cycle_major_message_memory(core_input.len()),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered read-write cycle-major message kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("read-write cycle-major message evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical read-write cycle-major message evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_read_write_cycle_to_address_major_kernel_evidence() {
    use jolt_backends::cpu::read_write_matrix as modular_read_write;
    use jolt_core::{
        ark_bn254::Fr as CoreFr,
        field::JoltField,
        subprotocols::read_write_matrix::{
            AddressMajorMatrixEntry as CoreAddressMajorEntryTrait,
            CycleMajorMatrixEntry as CoreCycleMajorEntryTrait,
            OneHotCoeffLookupTable as CoreLookupTable,
            ReadWriteMatrixAddressMajor as CoreAddressMajorMatrix,
            ReadWriteMatrixCycleMajor as CoreCycleMajorMatrix,
        },
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_harness::{validate_kernel_benchmark_evidence, KnownOptimizationIds};

    const KERNEL: &str = "cpu_read_write_cycle_to_address_major";
    const BENCHMARK: &str = "cpu_sumcheck/read_write_cycle_to_address_major";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-RW-003", "OPT-RW-009"];
    const ROW_PAIRS: usize = 128;
    const TERMS_PER_ROW: usize = 512;
    const ITERS: usize = 16;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct CoreCycleEntry {
        row: usize,
        column: usize,
        value: CoreFr,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct CoreAddressEntry {
        row: usize,
        column: usize,
        value: CoreFr,
    }

    impl CoreCycleMajorEntryTrait<CoreFr> for CoreCycleEntry {
        type AddressMajor = CoreAddressEntry;

        fn row(&self) -> usize {
            self.row
        }

        fn column(&self) -> usize {
            self.column
        }

        fn bind_entries(
            even: Option<&Self>,
            odd: Option<&Self>,
            r: <CoreFr as JoltField>::Challenge,
            _: Option<&CoreLookupTable<CoreFr>>,
            _: Option<&CoreLookupTable<CoreFr>>,
        ) -> Self {
            let r: CoreFr = r.into();
            match (even, odd) {
                (Some(even), Some(odd)) => Self {
                    row: even.row / 2,
                    column: even.column,
                    value: even.value + r * (odd.value - even.value),
                },
                (Some(even), None) => Self {
                    row: even.row / 2,
                    column: even.column,
                    value: (<CoreFr as JoltField>::from_u64(1) - r) * even.value,
                },
                (None, Some(odd)) => Self {
                    row: odd.row / 2,
                    column: odd.column,
                    value: r * odd.value,
                },
                (None, None) => unreachable!("cycle-major bind requires at least one entry"),
            }
        }

        fn compute_evals(
            _: Option<&Self>,
            _: Option<&Self>,
            _: [CoreFr; 2],
            _: CoreFr,
            _: Option<&CoreLookupTable<CoreFr>>,
            _: Option<&CoreLookupTable<CoreFr>>,
        ) -> [<CoreFr as JoltField>::UnreducedProduct; 2] {
            let zero = <CoreFr as JoltField>::from_u64(0);
            [zero.mul_to_product(zero), zero.mul_to_product(zero)]
        }

        fn to_address_major(
            self,
            _: Option<&CoreLookupTable<CoreFr>>,
            _: Option<&CoreLookupTable<CoreFr>>,
        ) -> Self::AddressMajor {
            CoreAddressEntry {
                row: self.row,
                column: self.column,
                value: self.value,
            }
        }
    }

    impl CoreAddressMajorEntryTrait<CoreFr> for CoreAddressEntry {
        fn row(&self) -> usize {
            self.row
        }

        fn column(&self) -> usize {
            self.column
        }

        fn prev_val(&self) -> CoreFr {
            self.value
        }

        fn next_val(&self) -> CoreFr {
            self.value
        }

        fn bind_entries(
            even: Option<&Self>,
            odd: Option<&Self>,
            _: CoreFr,
            _: CoreFr,
            r: <CoreFr as JoltField>::Challenge,
        ) -> Self {
            let r: CoreFr = r.into();
            match (even, odd) {
                (Some(even), Some(odd)) => Self {
                    row: even.row,
                    column: even.column / 2,
                    value: even.value + r * (odd.value - even.value),
                },
                (Some(even), None) => Self {
                    row: even.row,
                    column: even.column / 2,
                    value: (<CoreFr as JoltField>::from_u64(1) - r) * even.value,
                },
                (None, Some(odd)) => Self {
                    row: odd.row,
                    column: odd.column / 2,
                    value: r * odd.value,
                },
                (None, None) => unreachable!("address-major bind requires at least one entry"),
            }
        }

        fn compute_evals(
            _: Option<&Self>,
            _: Option<&Self>,
            _: CoreFr,
            _: CoreFr,
            _: CoreFr,
            _: CoreFr,
            _: CoreFr,
        ) -> [<CoreFr as JoltField>::UnreducedProduct; 2] {
            let zero = <CoreFr as JoltField>::from_u64(0);
            [zero.mul_to_product(zero), zero.mul_to_product(zero)]
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct ModularCycleEntry {
        row: usize,
        column: usize,
        value: Fr,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct ModularAddressEntry {
        row: usize,
        column: usize,
        value: Fr,
    }

    impl modular_read_write::CycleMajorMatrixEntry<Fr> for ModularCycleEntry {
        fn row(&self) -> usize {
            self.row
        }

        fn column(&self) -> usize {
            self.column
        }

        fn bind_entries(
            even: Option<&Self>,
            odd: Option<&Self>,
            r: Fr,
            _: Option<&modular_read_write::OneHotCoeffTable<Fr>>,
            _: Option<&modular_read_write::OneHotCoeffTable<Fr>>,
        ) -> Self {
            match (even, odd) {
                (Some(even), Some(odd)) => Self {
                    row: even.row / 2,
                    column: even.column,
                    value: even.value + r * (odd.value - even.value),
                },
                (Some(even), None) => Self {
                    row: even.row / 2,
                    column: even.column,
                    value: (Fr::from_u64(1) - r) * even.value,
                },
                (None, Some(odd)) => Self {
                    row: odd.row / 2,
                    column: odd.column,
                    value: r * odd.value,
                },
                (None, None) => unreachable!("cycle-major bind requires at least one entry"),
            }
        }
    }

    impl modular_read_write::AddressMajorMatrixEntry<Fr> for ModularAddressEntry {
        fn row(&self) -> usize {
            self.row
        }

        fn column(&self) -> usize {
            self.column
        }
    }

    impl modular_read_write::CycleMajorToAddressMajor<Fr> for ModularCycleEntry {
        type AddressMajor = ModularAddressEntry;

        fn to_address_major(
            self,
            _: Option<&modular_read_write::OneHotCoeffTable<Fr>>,
            _: Option<&modular_read_write::OneHotCoeffTable<Fr>>,
        ) -> Self::AddressMajor {
            ModularAddressEntry {
                row: self.row,
                column: self.column,
                value: self.value,
            }
        }
    }

    fn core_field_from_index(index: usize) -> CoreFr {
        <CoreFr as JoltField>::from_u64((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    fn core_entries(row_pairs: usize, terms_per_row: usize) -> Vec<CoreCycleEntry> {
        let mut entries = Vec::with_capacity(row_pairs * terms_per_row * 2);
        for pair in 0..row_pairs {
            let even_row = 2 * pair;
            let odd_row = even_row + 1;
            for index in 0..terms_per_row {
                entries.push(CoreCycleEntry {
                    row: even_row,
                    column: index * 2,
                    value: core_field_from_index(530_000 + pair * terms_per_row + index),
                });
            }
            for index in 0..terms_per_row {
                entries.push(CoreCycleEntry {
                    row: odd_row,
                    column: if index % 3 == 0 {
                        index * 2
                    } else {
                        index * 2 + 1
                    },
                    value: core_field_from_index(630_000 + pair * terms_per_row + index),
                });
            }
        }
        entries
    }

    fn modular_entries(core_entries: &[CoreCycleEntry]) -> Vec<ModularCycleEntry> {
        core_entries
            .iter()
            .map(|entry| ModularCycleEntry {
                row: entry.row,
                column: entry.column,
                value: Fr::from(entry.value),
            })
            .collect()
    }

    fn core_fingerprint(entries: &[CoreAddressEntry]) -> CoreFr {
        entries.iter().step_by((entries.len() / 64).max(1)).fold(
            <CoreFr as JoltField>::from_u64(entries.len() as u64),
            |acc, entry| {
                acc + entry.value
                    + <CoreFr as JoltField>::from_u64((entry.row ^ entry.column) as u64)
            },
        )
    }

    fn modular_fingerprint(entries: &[ModularAddressEntry]) -> Fr {
        entries
            .iter()
            .step_by((entries.len() / 64).max(1))
            .fold(Fr::from_u64(entries.len() as u64), |acc, entry| {
                acc + entry.value + Fr::from_u64((entry.row ^ entry.column) as u64)
            })
    }

    fn assert_entries_match(
        core_entries: &[CoreAddressEntry],
        modular_entries: &[ModularAddressEntry],
    ) {
        assert_eq!(core_entries.len(), modular_entries.len());
        for (core, modular) in core_entries.iter().zip(modular_entries) {
            assert_eq!(core.row, modular.row);
            assert_eq!(core.column, modular.column);
            let modular_value: CoreFr = modular.value.into();
            assert_eq!(core.value, modular_value);
        }
    }

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let core_input = core_entries(ROW_PAIRS, TERMS_PER_ROW);
    let modular_input = modular_entries(&core_input);

    let mut core_cycle = CoreCycleMajorMatrix::<CoreFr, CoreCycleEntry>::default();
    core_cycle.entries.clone_from(&core_input);
    let core_reference: CoreAddressMajorMatrix<CoreFr, CoreAddressEntry> = core_cycle.into();
    let modular_cycle = modular_read_write::ReadWriteMatrixCycleMajor::new(modular_input.clone());
    let modular_reference: modular_read_write::ReadWriteMatrixAddressMajor<
        Fr,
        ModularAddressEntry,
    > = modular_cycle.into();
    assert_entries_match(&core_reference.entries, &modular_reference.entries);

    let core = measure_samples(samples, || {
        let mut total = <CoreFr as JoltField>::from_u64(0);
        for _ in 0..ITERS {
            let mut cycle = CoreCycleMajorMatrix::<CoreFr, CoreCycleEntry>::default();
            cycle.entries.clone_from(&core_input);
            let address: CoreAddressMajorMatrix<CoreFr, CoreAddressEntry> = cycle.into();
            total += core_fingerprint(&address.entries);
        }
        let _total = black_box(total);
    });

    let modular = measure_samples(samples, || {
        let mut total = Fr::from_u64(0);
        for _ in 0..ITERS {
            let cycle = modular_read_write::ReadWriteMatrixCycleMajor::new(modular_input.clone());
            let address: modular_read_write::ReadWriteMatrixAddressMajor<Fr, ModularAddressEntry> =
                cycle.into();
            total += modular_fingerprint(&address.entries);
        }
        let _total = black_box(total);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: read_write_cycle_to_address_major_memory(core_input.len()),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered read-write cycle-to-address kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("read-write cycle-to-address evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical read-write cycle-to-address evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage0_streaming_commitment_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_streaming_commitments requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
)))]
#[expect(clippy::panic)]
fn write_stage0_zk_streaming_commitment_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_zk_streaming_commitments requires --features core-fixtures,zk"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
)))]
#[expect(clippy::panic)]
fn write_blindfold_round_commitment_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_blindfold_round_commitments requires --features core-fixtures,zk"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
)))]
#[expect(clippy::panic)]
fn write_blindfold_backend_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_blindfold_backend_kernels requires --features core-fixtures,zk"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage0_advice_commitment_context_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_advice_commitment_contexts requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage0_field_inline_commitment_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_field_inline_commitments requires --features core-fixtures,field-inline"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_one_hot_commitment_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_one_hot_commitments requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage1_spartan_outer_prefix_product_sum_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_spartan_outer_prefix_product_sum requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage2_product_uniskip_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_spartan_product_uniskip requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage2_regular_batch_input_claim_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage2_regular_batch_input_claims requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage3_regular_batch_input_claim_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage3_regular_batch_input_claims requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage4_regular_batch_input_claim_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage4_regular_batch_input_claims requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage5_regular_batch_input_claim_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage5_regular_batch_input_claims requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage5_regular_batch_sumcheck_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage5_regular_batch_sumcheck requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage6_regular_batch_input_claim_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage6_regular_batch_input_claims requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage6_regular_batch_sumcheck_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage6_regular_batch_sumcheck requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage4_field_inline_registers_read_write_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_field_inline_stage4_registers_read_write requires --features core-fixtures,field-inline"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage5_field_inline_registers_val_evaluation_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_field_inline_stage5_registers_val_evaluation requires --features core-fixtures,field-inline"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage6_field_inline_registers_inc_claim_reduction_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_field_inline_stage6_registers_inc_claim_reduction requires --features core-fixtures,field-inline"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage7_regular_batch_input_claim_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage7_regular_batch_input_claims requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage7_regular_batch_sumcheck_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage7_regular_batch_sumcheck requires --features core-fixtures"
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage2_regular_batch_sumcheck_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage2_regular_batch_sumcheck_kernel_benchmark_fixture,
        validate_kernel_benchmark_evidence, FeatureMode, FixtureKind, FixtureRequest,
        KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_stage2_regular_batch_sumcheck";
    const BENCHMARK: &str = "frontier_perf/stage2_regular_batch_sumcheck";
    const OPTIMIZATION_IDS: [&str; 2] = ["OPT-SC-007", "OPT-EQ-004"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage2_regular_batch_sumcheck_kernel_benchmark_fixture(&request)
        .expect("load Stage 2 regular-batch sumcheck kernel fixture");
    assert_eq!(
        fixture
            .run_reference_sumcheck()
            .expect("run reference Stage 2 regular-batch sumcheck"),
        fixture.expected.challenges.len() * 2
    );
    let modular_proof = fixture
        .run_modular_sumcheck()
        .expect("run modular Stage 2 regular-batch sumcheck");
    assert_eq!(modular_proof.proof, fixture.expected.proof);
    assert_eq!(modular_proof.challenges, fixture.expected.challenges);
    assert_eq!(
        modular_proof.batching_coefficients,
        fixture.expected.batching_coefficients
    );
    assert_eq!(modular_proof.output_claim, fixture.expected.output_claim);

    let core = measure_samples(samples, || {
        let rounds = fixture
            .run_reference_sumcheck()
            .expect("run reference Stage 2 regular-batch sumcheck");
        let _rounds = black_box(rounds);
    });

    let modular = measure_samples(samples, || {
        let proof = fixture
            .run_modular_sumcheck()
            .expect("run modular Stage 2 regular-batch sumcheck");
        let _proof = black_box(proof.output_claim);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage2_regular_batch_sumcheck_memory(fixture.config.log_t, fixture.config.log_k),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 2 regular-batch sumcheck kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 2 regular-batch sumcheck evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 2 regular-batch sumcheck evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::print_stdout)]
fn write_stage3_regular_batch_sumcheck_kernel_evidence() {
    use jolt_prover_harness::{
        load_stage3_regular_batch_sumcheck_kernel_benchmark_fixture,
        validate_kernel_benchmark_evidence, FeatureMode, FixtureKind, FixtureRequest,
        KnownOptimizationIds,
    };

    const KERNEL: &str = "cpu_stage3_regular_batch_sumcheck";
    const BENCHMARK: &str = "frontier_perf/stage3_regular_batch_sumcheck";
    const OPTIMIZATION_IDS: [&str; 3] = ["OPT-SC-007", "OPT-EQ-004", "OPT-SP-006"];

    let samples = std::env::var("JOLT_KERNEL_EVIDENCE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PerfGate::canonical_frontier().min_samples);
    let workspace = workspace_root();
    let request = FixtureRequest::new(FixtureKind::MuldivSmall, FeatureMode::Transparent);
    let fixture = load_stage3_regular_batch_sumcheck_kernel_benchmark_fixture(&request)
        .expect("load Stage 3 regular-batch sumcheck kernel fixture");
    assert_eq!(
        fixture
            .run_reference_sumcheck()
            .expect("run reference Stage 3 regular-batch sumcheck"),
        fixture.expected.challenges.len() * 2
    );
    let modular_proof = fixture
        .run_modular_sumcheck()
        .expect("run modular Stage 3 regular-batch sumcheck");
    assert_eq!(modular_proof.proof, fixture.expected.proof);
    assert_eq!(modular_proof.challenges, fixture.expected.challenges);
    assert_eq!(
        modular_proof.batching_coefficients,
        fixture.expected.batching_coefficients
    );
    assert_eq!(modular_proof.output_claim, fixture.expected.output_claim);

    let core = measure_samples(samples, || {
        let rounds = fixture
            .run_reference_sumcheck()
            .expect("run reference Stage 3 regular-batch sumcheck");
        let _rounds = black_box(rounds);
    });

    let modular = measure_samples(samples, || {
        let proof = fixture
            .run_modular_sumcheck()
            .expect("run modular Stage 3 regular-batch sumcheck");
        let _proof = black_box(proof.output_claim);
    });

    let evidence = KernelBenchmarkEvidence {
        kernel: KERNEL.to_owned(),
        benchmark: BENCHMARK.to_owned(),
        samples,
        optimization_ids: OPTIMIZATION_IDS.iter().map(|id| (*id).to_owned()).collect(),
        core,
        modular,
        memory: stage3_regular_batch_sumcheck_memory(fixture.config.log_t),
    };

    let known = KnownOptimizationIds::parse_inventory(include_str!(
        "../../../specs/jolt-core-prover-optimization-inventory.md"
    ))
    .expect("parse optimization inventory");
    let ledger = jolt_prover_harness::registered_backend_kernel_ports(&known)
        .expect("registered backend kernel ledger");
    let port = ledger
        .find(KERNEL)
        .expect("registered Stage 3 regular-batch sumcheck kernel");
    let evaluation =
        validate_kernel_benchmark_evidence(PerfGate::canonical_frontier(), *port, &evidence)
            .expect("Stage 3 regular-batch sumcheck evidence should pass the canonical gate");
    let path = evidence
        .write_canonical_json(&workspace)
        .expect("write canonical Stage 3 regular-batch sumcheck evidence");
    println!(
        "wrote {} with status {:?}, time ratio {:?}, memory ratio {:?}",
        path.display(),
        evaluation.status,
        evaluation.time_ratio,
        evaluation.peak_rss_ratio
    );
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage2_regular_batch_sumcheck_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage2_regular_batch_sumcheck requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage3_regular_batch_sumcheck_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage3_regular_batch_sumcheck requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage4_regular_batch_sumcheck_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_stage4_regular_batch_sumcheck requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_materialized_opening_rlc_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_materialized_opening_evaluations requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_eq_table_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_eq_table_generation requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_eq_aligned_block_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_eq_aligned_block_generation requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_split_eq_streaming_window_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_split_eq_streaming_windows requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_unipoly_interpolation_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_unipoly_interpolation requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_compressed_unipoly_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_compressed_unipoly requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_lagrange_many_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_lagrange_many requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_compact_polynomial_bind_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_compact_polynomial_bind requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_split_eq_polynomial_evaluation_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_split_eq_polynomial_evaluation requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_inside_out_polynomial_evaluation_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_inside_out_polynomial_evaluation requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_dense_batch_polynomial_evaluation_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_dense_batch_polynomial_evaluation requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_dense_dot_product_low_optimized_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_dense_dot_product_low_optimized requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_mixed_polynomial_linear_combination_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_mixed_polynomial_linear_combination requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_one_hot_polynomial_evaluation_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_one_hot_polynomial_evaluation requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_one_hot_vector_matrix_product_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_one_hot_vector_matrix_product requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_rlc_polynomial_vector_matrix_product_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_rlc_polynomial_vector_matrix_product requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_stage8_streaming_rlc_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_opening_stage8_kernels requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_linear_product_small_degrees_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_linear_product_small_degrees requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_linear_product_d4_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_linear_product_d4 requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_linear_product_d8_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_linear_product_d8 requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_linear_product_d16_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_linear_product_d16 requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_linear_product_d32_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_linear_product_d32 requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_streaming_schedule_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_streaming_schedule requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_ra_delayed_materialization_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_ra_delayed_materialization requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_shared_ra_delayed_materialization_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_shared_ra_delayed_materialization requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_ra_pushforward_kernel_evidence() {
    panic!("JOLT_WRITE_KERNEL_EVIDENCE=cpu_ra_pushforward requires --features core-fixtures")
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_read_write_one_hot_coeff_lookup_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_read_write_one_hot_coeff_lookup requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_read_write_cycle_major_bind_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_read_write_cycle_major_bind requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_read_write_cycle_major_message_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_read_write_cycle_major_message requires --features core-fixtures"
    )
}

#[cfg(not(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
)))]
#[expect(clippy::panic)]
fn write_read_write_cycle_to_address_major_kernel_evidence() {
    panic!(
        "JOLT_WRITE_KERNEL_EVIDENCE=cpu_read_write_cycle_to_address_major requires --features core-fixtures"
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage0_streaming_commitment_memory(
    shape: jolt_prover_harness::Stage0CommitmentKernelShape,
) -> KernelMemoryBudget {
    use jolt_crypto::{Bn254G1, Bn254GT};
    use jolt_program::execution::TraceRow;

    let g1 = size_of::<Bn254G1>();
    let gt = size_of::<Bn254GT>();
    let vec_header = size_of::<Vec<Bn254G1>>();
    let option_index = size_of::<Option<u8>>();
    let input_bytes = shape.trace_length * size_of::<TraceRow>()
        + shape.committed_polynomials * size_of::<usize>();
    let retained_hint_rows = (shape.dense_polynomials * shape.core_cycle_chunks
        + shape.one_hot_polynomials * shape.dory_rows)
        * g1;
    let output_commitments = shape.committed_polynomials * gt;
    let streamed_chunk_records =
        shape.committed_polynomials * shape.core_cycle_chunks * 3 * size_of::<usize>();
    let one_hot_commit_scratch =
        shape.dory_rows * vec_header + shape.trace_length * size_of::<usize>();
    let streaming_index_buffer = shape.trace_length * option_index;
    let peak_working = retained_hint_rows
        + output_commitments
        + streamed_chunk_records
        + one_hot_commit_scratch
        + streaming_index_buffer;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 128 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn stage0_zk_streaming_commitment_memory(
    shape: Stage0ZkCommitmentKernelShape,
) -> KernelMemoryBudget {
    use jolt_core::ark_bn254::G1Affine;
    use jolt_crypto::{Bn254G1, Bn254GT};
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let g1 = size_of::<Bn254G1>();
    let gt = size_of::<Bn254GT>();
    let input_bytes = shape.rows * size_of::<u64>();
    let retained_hint_rows = shape.pcs_rows * g1 + field;
    let output_commitment = gt;
    let row_buffer = shape.row_width * size_of::<u64>();
    let row_msm_bases = shape.row_width * size_of::<G1Affine>();
    let streamed_chunk_records = shape.chunks * 3 * size_of::<usize>();
    let peak_working = retained_hint_rows
        + output_commitment
        + row_buffer
        + row_msm_bases
        + streamed_chunk_records;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 128 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn blindfold_round_commitment_memory(row_count: usize, row_len: usize) -> KernelMemoryBudget {
    use jolt_crypto::Bn254G1;
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let g1 = size_of::<Bn254G1>();
    let input_bytes = row_count * row_len * field + row_count * field;
    let setup_bytes = row_len * g1 + g1;
    let output_bytes = row_count * g1;
    let peak_working = input_bytes + setup_bytes + output_bytes + row_count * size_of::<Vec<Fr>>();
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 128 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn blindfold_backend_kernel_memory(fixture: &BlindFoldKernelFixture) -> KernelMemoryBudget {
    use jolt_crypto::Bn254G1;
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let g1 = size_of::<Bn254G1>();
    let row_bytes = |rows: &[Vec<Fr>]| {
        rows.iter().map(Vec::len).sum::<usize>() * field + std::mem::size_of_val(rows)
    };
    let scalar_bytes = |values: &[Fr]| std::mem::size_of_val(values);
    let sparse_entry = size_of::<usize>() + field;
    let sparse_entries = fixture
        .r1cs
        .a
        .iter()
        .chain(&fixture.r1cs.b)
        .chain(&fixture.r1cs.c)
        .map(Vec::len)
        .sum::<usize>();

    let input_bytes = row_bytes(&fixture.witness_rows)
        + row_bytes(&fixture.output_rows)
        + scalar_bytes(&fixture.witness_blindings)
        + scalar_bytes(&fixture.output_blindings)
        + scalar_bytes(&fixture.real_witness)
        + scalar_bytes(&fixture.random_witness)
        + sparse_entries * sparse_entry
        + scalar_bytes(&fixture.row_point)
        + scalar_bytes(&fixture.entry_point);
    let commitments = (fixture.witness_rows.len() + fixture.output_rows.len()) * g1;
    let error_rows = 3 * fixture.error_row_count * fixture.error_row_len * field;
    let folded_rows = 2 * fixture.error_row_count * fixture.error_row_len * field
        + fixture.witness_rows.len() * fixture.witness_rows[0].len() * field;
    let openings = 2 * (fixture.witness_rows[0].len() * field + field + size_of::<Vec<Fr>>());
    let peak_working = input_bytes + commitments + error_rows + folded_rows + openings;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 128 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage0_advice_commitment_context_memory(
    shape: jolt_prover_harness::Stage0AdviceCommitmentKernelShape,
) -> KernelMemoryBudget {
    use jolt_crypto::{Bn254G1, Bn254GT};
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let g1 = size_of::<Bn254G1>();
    let gt = size_of::<Bn254GT>();
    let rows = shape.trusted_rows + shape.untrusted_rows;
    let input_bytes = rows * size_of::<u64>();
    let retained_hint_rows = (shape.trusted_pcs_rows + shape.untrusted_pcs_rows) * g1;
    let output_commitments = shape.commitment_count * gt;
    let max_row_width = shape.trusted_row_width.max(shape.untrusted_row_width);
    let max_chunk_rows = shape.trusted_rows.max(shape.untrusted_rows).min(1024);
    let dense_stream_buffers = max_row_width * field + max_chunk_rows * (size_of::<u64>() + field);
    let streamed_chunk_records = shape.commitment_count * 3 * size_of::<usize>();
    let peak_working =
        retained_hint_rows + output_commitments + dense_stream_buffers + streamed_chunk_records;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 128 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
fn stage0_field_inline_commitment_memory(
    shape: Stage0FieldInlineCommitmentKernelShape,
) -> KernelMemoryBudget {
    use jolt_crypto::{Bn254G1, Bn254GT};
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let g1 = size_of::<Bn254G1>();
    let gt = size_of::<Bn254GT>();
    let input_bytes = shape.rows * field;
    let retained_hint_rows = shape.pcs_rows * g1;
    let output_commitment = gt;
    let dense_stream_buffers = shape.row_width * field + shape.chunk_size.min(shape.rows) * field;
    let streamed_chunk_records = shape.chunks * 3 * size_of::<usize>();
    let peak_working =
        retained_hint_rows + output_commitment + dense_stream_buffers + streamed_chunk_records;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 128 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn one_hot_commitment_memory(k: usize, trace_rows: usize) -> KernelMemoryBudget {
    use jolt_core::ark_bn254::G1Affine;
    use jolt_crypto::{Bn254G1, Bn254GT};

    let total_vars = (k * trace_rows).trailing_zeros() as usize;
    let dory_cols = 1usize << total_vars.div_ceil(2);
    let dory_rows = 1usize << (total_vars - total_vars.div_ceil(2));
    let input_bytes = trace_rows * size_of::<Option<u8>>();
    let retained_hint_rows = dory_rows * size_of::<Bn254G1>();
    let output_commitment = size_of::<Bn254GT>();
    let base_affines = dory_cols * size_of::<G1Affine>();
    let row_index_vectors = dory_rows * size_of::<Vec<usize>>();
    let row_indices = trace_rows * size_of::<usize>();
    let batch_addition_points = trace_rows * size_of::<G1Affine>();
    let peak_working = retained_hint_rows
        + output_commitment
        + base_affines
        + row_index_vectors
        + row_indices
        + batch_addition_points;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 128 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage1_spartan_outer_prefix_product_memory(
    log_t: usize,
    row_count: usize,
    query_count: usize,
) -> KernelMemoryBudget {
    use jolt_backends::{
        BackendValueSlot, SumcheckLinearProductOutput, SumcheckSpartanOuterRow,
        SumcheckSpartanOuterUniskipQuery,
    };
    use jolt_field::Fr;
    use jolt_r1cs::constraints::jolt::SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE;

    let field = size_of::<Fr>();
    let input_bytes = row_count * size_of::<SumcheckSpartanOuterRow>()
        + query_count
            * (size_of::<SumcheckSpartanOuterUniskipQuery<Fr>>()
                + (log_t + 1) * field
                + SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE * size_of::<i32>());
    let output_bytes = query_count * size_of::<SumcheckLinearProductOutput<Fr>>();
    let slot_set_bytes = query_count * size_of::<BackendValueSlot>() * 4;
    let eq_tables =
        tensor_eq_entries(log_t) * field + tensor_eq_outer_entries(log_t) * query_count * field;
    let accumulators = rayon::current_num_threads() * query_count * field;
    let peak_working = output_bytes + slot_set_bytes + eq_tables + accumulators;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (input_bytes + peak_working * 8 + 256 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage1_spartan_outer_remainder_prefix_product_memory(
    log_t: usize,
    input_count: usize,
    row_count: usize,
    sparse_row_count: usize,
    sparse_term_count: usize,
    query_count: usize,
    fixed_prefix_len: usize,
) -> KernelMemoryBudget {
    use jolt_backends::{
        BackendValueSlot, SumcheckLinearProductOutput, SumcheckPrefixProductSumQuery,
    };
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let witness_bytes = row_count * input_count * field;
    let sparse_rows = 2 * sparse_row_count * size_of::<Vec<(usize, Fr)>>()
        + sparse_term_count * size_of::<(usize, Fr)>();
    let query_bytes = query_count
        * (size_of::<SumcheckPrefixProductSumQuery<Fr>>()
            + (log_t + 1) * field
            + fixed_prefix_len * field
            + 2 * sparse_row_count * field);
    let input_bytes = witness_bytes + sparse_rows + query_bytes;
    let output_bytes = query_count * size_of::<SumcheckLinearProductOutput<Fr>>();
    let slot_set_bytes = query_count * size_of::<BackendValueSlot>() * 4;
    let layout_bytes = input_count * size_of::<(usize, usize)>() * 4;
    let eq_tables = tensor_eq_entries(log_t + 1 - fixed_prefix_len) * field;
    let bound_prefix_scratch = 4 * row_count * field;
    let peak_working =
        output_bytes + slot_set_bytes + layout_bytes + eq_tables + bound_prefix_scratch;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (input_bytes + peak_working * 8 + 256 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn deterministic_one_hot_indices(k: usize, trace_rows: usize) -> Vec<Option<u8>> {
    (0..trace_rows)
        .map(|row| {
            if row % 17 == 0 {
                None
            } else {
                Some(((row.wrapping_mul(37) ^ (row >> 3)) % k) as u8)
            }
        })
        .collect()
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn raw_product_uniskip_memory(rows: usize, log_t: usize) -> KernelMemoryBudget {
    use jolt_backends::{
        BackendValueSlot, SumcheckLinearProductOutput, SumcheckProductUniskipRow,
        SumcheckRowProductQuery,
    };
    use jolt_field::Fr;

    const QUERY_COUNT: usize = 2;
    const PRODUCT_ROWS: usize = 3;
    let field = size_of::<Fr>();
    let input_bytes = rows * size_of::<SumcheckProductUniskipRow>()
        + QUERY_COUNT
            * (size_of::<SumcheckRowProductQuery<Fr>>() + log_t * field + PRODUCT_ROWS * field);
    let output_bytes = QUERY_COUNT * size_of::<SumcheckLinearProductOutput<Fr>>();
    let slot_set_bytes = QUERY_COUNT * size_of::<BackendValueSlot>() * 4;
    let eq_tables =
        tensor_eq_entries(log_t) * field + tensor_eq_outer_entries(log_t) * QUERY_COUNT * field;
    let peak_working = output_bytes + slot_set_bytes + eq_tables;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_regular_batch_input_claim_memory(log_t: usize, log_k: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_prover::stages::stage2::output::{
        Stage2RegularBatchInputClaims, Stage2RegularBatchPrefixOutput,
    };
    use jolt_transcript::Blake2bTranscript;

    let field = size_of::<Fr>();
    let input_bytes = size_of::<Stage2RegularBatchInputClaims<Fr>>()
        + size_of::<Blake2bTranscript<Fr>>()
        + (log_t + log_k + 1) * field;
    let output_bytes = size_of::<Stage2RegularBatchPrefixOutput<Fr>>() + log_k * field;
    let peak_working = input_bytes + output_bytes;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage3_regular_batch_input_claim_memory(log_t: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_prover::stages::stage3::output::{
        Stage3RegularBatchInputClaims, Stage3RegularBatchPrefixOutput,
    };
    use jolt_transcript::Blake2bTranscript;

    let field = size_of::<Fr>();
    let input_bytes = size_of::<Stage3RegularBatchInputClaims<Fr>>()
        + size_of::<Blake2bTranscript<Fr>>()
        + (log_t + 1) * field;
    let output_bytes = size_of::<Stage3RegularBatchPrefixOutput<Fr>>();
    let peak_working = input_bytes + output_bytes;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage4_regular_batch_input_claim_memory(log_t: usize, log_k: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_prover::stages::stage4::output::{
        Stage4RegularBatchInputClaims, Stage4RegularBatchPrefixOutput,
    };
    use jolt_transcript::Blake2bTranscript;

    let field = size_of::<Fr>();
    let input_bytes = size_of::<Stage4RegularBatchInputClaims<Fr>>()
        + size_of::<Blake2bTranscript<Fr>>()
        + (log_t + log_k + 1) * field;
    let output_bytes = size_of::<Stage4RegularBatchPrefixOutput<Fr>>();
    let peak_working = input_bytes + output_bytes;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage4_regular_batch_sumcheck_memory(log_t: usize, log_k: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_poly::Polynomial;

    let field = size_of::<Fr>();
    let trace_rows = 1usize << log_t;
    let register_rows = 1usize << (log_t + 5);
    let ram_rows = 1usize << (log_t + log_k);
    let materialized_entries = 4 * register_rows + ram_rows + 2 * trace_rows;
    let instance_entries = 6 * register_rows + 3 * trace_rows;
    let polynomial_headers = 9 * size_of::<Polynomial<Fr>>();
    let input_bytes = (materialized_entries + instance_entries) * field + polynomial_headers;
    let round_working = input_bytes + (log_t + 5) * field * 8;
    KernelMemoryBudget::new(
        input_bytes as u64,
        round_working as u64,
        (round_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage5_regular_batch_input_claim_memory(log_t: usize, log_k: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_prover::stages::stage5::output::{
        Stage5RegularBatchInputClaims, Stage5RegularBatchPrefixOutput,
    };
    use jolt_transcript::Blake2bTranscript;

    let field = size_of::<Fr>();
    let input_bytes = size_of::<Stage5RegularBatchInputClaims<Fr>>()
        + size_of::<Blake2bTranscript<Fr>>()
        + (log_t + log_k + 1) * field;
    let output_bytes = size_of::<Stage5RegularBatchPrefixOutput<Fr>>();
    let peak_working = input_bytes + output_bytes;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage5_regular_batch_sumcheck_memory(log_t: usize, log_k: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_poly::Polynomial;

    let field = size_of::<Fr>();
    let trace_rows = 1usize << log_t;
    let ram_rows = 1usize << (log_t + log_k);
    let register_rows = 1usize << (log_t + 5);
    let materialized_entries = ram_rows + register_rows + trace_rows;
    let reduced_entries = 10 * trace_rows;
    let instruction_entries = 24 * trace_rows;
    let polynomial_headers = 12 * size_of::<Polynomial<Fr>>();
    let input_bytes =
        (materialized_entries + reduced_entries + instruction_entries) * field + polynomial_headers;
    let round_working = input_bytes + (log_t + 128) * field * 8;
    KernelMemoryBudget::new(
        input_bytes as u64,
        round_working as u64,
        (round_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage6_regular_batch_input_claim_memory(log_t: usize, log_k: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_prover::stages::stage6::output::{
        Stage6RegularBatchInputClaims, Stage6RegularBatchPrefixOutput,
    };
    use jolt_transcript::Blake2bTranscript;

    let field = size_of::<Fr>();
    let input_bytes = size_of::<Stage6RegularBatchInputClaims<Fr>>()
        + size_of::<Blake2bTranscript<Fr>>()
        + (log_t + log_k + 1) * field;
    let output_bytes = size_of::<Stage6RegularBatchPrefixOutput<Fr>>();
    let peak_working = input_bytes + output_bytes;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage6_regular_batch_sumcheck_memory(log_t: usize, log_k: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_poly::Polynomial;
    use jolt_prover::stages::stage6::output::Stage6RegularBatchProofOutput;

    let field = size_of::<Fr>();
    let trace_rows = 1usize << log_t;
    let ram_rows = 1usize << log_k;
    let delayed_ra_rows = 8 * trace_rows;
    let split_eq_tables = 12 * trace_rows;
    let dense_cycle_rows = 10 * trace_rows;
    let increment_rows = 4 * trace_rows;
    let bytecode_rows = 16 * ram_rows.min(trace_rows.max(1));
    let polynomial_headers = 64 * size_of::<Polynomial<Fr>>();
    let input_bytes =
        (delayed_ra_rows + split_eq_tables + dense_cycle_rows + increment_rows + bytecode_rows)
            * field
            + polynomial_headers;
    let output_bytes = size_of::<Stage6RegularBatchProofOutput<Fr, jolt_crypto::Bn254G1>>();
    let round_working = input_bytes + output_bytes + (log_t + 256) * field * 64;
    KernelMemoryBudget::new(
        input_bytes as u64,
        round_working as u64,
        (round_working * 8).max(512 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
fn stage4_field_inline_read_write_memory(
    rows: usize,
    log_t: usize,
    log_k: usize,
) -> KernelMemoryBudget {
    use jolt_backends::SumcheckFieldRegistersReadWriteRow;
    use jolt_field::Fr;
    use jolt_poly::Polynomial;

    let field = size_of::<Fr>();
    let row_bytes = size_of::<SumcheckFieldRegistersReadWriteRow<Fr>>();
    let trace_rows = 1usize << log_t;
    let register_rows = 1usize << log_k;
    let dense_reference_rows = trace_rows * register_rows;
    let input_bytes = rows * row_bytes;
    let inc = trace_rows * field + size_of::<Polynomial<Fr>>();
    let gruen_split = 3 * (1usize << log_t.div_ceil(2)) * field;
    let sparse_entries = rows * 3 * (8 * field + 32);
    let late_materialized = 3 * register_rows * field + 3 * size_of::<Polynomial<Fr>>();
    let dense_reference = 5 * dense_reference_rows * field + 5 * size_of::<Polynomial<Fr>>();
    let peak_working = inc + gruen_split + sparse_entries + late_materialized;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (input_bytes + peak_working.max(dense_reference) * 8 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
fn stage5_field_inline_val_evaluation_memory(
    rows: usize,
    log_t: usize,
    log_k: usize,
) -> KernelMemoryBudget {
    use jolt_backends::SumcheckFieldRegistersReadWriteRow;
    use jolt_field::Fr;
    use jolt_poly::Polynomial;

    let field = size_of::<Fr>();
    let row_bytes = size_of::<SumcheckFieldRegistersReadWriteRow<Fr>>();
    let register_rows = 1usize << log_k;
    let lt_lo = 1usize << (log_t / 2);
    let lt_hi = 1usize << (log_t - log_t / 2);
    let input_bytes = rows * row_bytes;
    let inc = rows * field + size_of::<Polynomial<Fr>>();
    let wa = rows * field + size_of::<Polynomial<Fr>>();
    let wa_eq = register_rows * field;
    let lt_split = (lt_lo + 2 * lt_hi) * field;
    let peak_working = inc + wa + wa_eq + lt_split;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (input_bytes + peak_working * 8 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
fn stage6_field_inline_inc_claim_reduction_memory(rows: usize, log_t: usize) -> KernelMemoryBudget {
    use jolt_backends::SumcheckFieldRegistersReadWriteRow;
    use jolt_field::Fr;
    use jolt_poly::Polynomial;

    let field = size_of::<Fr>();
    let row_bytes = size_of::<SumcheckFieldRegistersReadWriteRow<Fr>>();
    let prefix_vars = log_t / 2;
    let suffix_vars = log_t - prefix_vars;
    let prefix_rows = 1usize << prefix_vars;
    let suffix_rows = 1usize << suffix_vars;
    let input_bytes = rows * row_bytes;
    let prefix_tables = 4 * prefix_rows * field + 4 * size_of::<Polynomial<Fr>>();
    let suffix_tables = 2 * suffix_rows * field + 2 * size_of::<Polynomial<Fr>>();
    let prefix_eq = prefix_rows * field;
    let cloned_rows = rows * row_bytes;
    let peak_working = cloned_rows + prefix_tables.max(suffix_tables + prefix_eq);
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (input_bytes + peak_working * 8 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage7_regular_batch_input_claim_memory(log_t: usize, log_k_chunk: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_prover::stages::stage7::output::{
        Stage7RegularBatchInputClaims, Stage7RegularBatchPrefixOutput,
    };
    use jolt_transcript::Blake2bTranscript;

    let field = size_of::<Fr>();
    let input_bytes = size_of::<Stage7RegularBatchInputClaims<Fr>>()
        + size_of::<Blake2bTranscript<Fr>>()
        + (log_t + log_k_chunk + 1) * field;
    let output_bytes = size_of::<Stage7RegularBatchPrefixOutput<Fr>>();
    let peak_working = input_bytes + output_bytes;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage7_regular_batch_sumcheck_memory(
    log_t: usize,
    log_k_chunk: usize,
    total_ra_polys: usize,
) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_poly::Polynomial;
    use jolt_prover::stages::stage7::output::Stage7ProverOutput;

    let field = size_of::<Fr>();
    let trace_rows = 1usize << log_t;
    let k_rows = 1usize << log_k_chunk;
    let ra_index_rows = trace_rows * total_ra_polys;
    let pushforward_tables = total_ra_polys * k_rows;
    let eq_tables = (total_ra_polys + 1) * k_rows;
    let generic_product_tables = (2 * total_ra_polys + 1) * k_rows;
    let polynomial_headers = (3 * total_ra_polys + 2) * size_of::<Polynomial<Fr>>();
    let input_bytes = (ra_index_rows + pushforward_tables + eq_tables + generic_product_tables)
        * field
        + polynomial_headers;
    let output_bytes =
        size_of::<Stage7ProverOutput<Fr, jolt_sumcheck::SumcheckProof<Fr, jolt_crypto::Bn254G1>>>();
    let round_working = input_bytes + output_bytes + (log_t + log_k_chunk + 256) * field * 64;
    KernelMemoryBudget::new(
        input_bytes as u64,
        round_working as u64,
        (round_working * 8).max(512 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage2_regular_batch_sumcheck_memory(log_t: usize, log_k: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_poly::Polynomial;

    let field = size_of::<Fr>();
    let trace_rows = 1usize << log_t;
    let ram_rows = 1usize << (log_t + log_k);
    let ram_address_rows = 1usize << log_k;
    let dense_factor_entries = 4 * ram_rows + 13 * trace_rows + 4 * ram_address_rows;
    let polynomial_headers = 24 * size_of::<Polynomial<Fr>>();
    let input_bytes = dense_factor_entries * field + polynomial_headers;
    let round_working = dense_factor_entries * field + (log_t + log_k) * field * 8;
    KernelMemoryBudget::new(
        input_bytes as u64,
        round_working as u64,
        (round_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage3_regular_batch_sumcheck_memory(log_t: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_poly::Polynomial;

    let field = size_of::<Fr>();
    let trace_rows = 1usize << log_t;
    let dense_factor_entries = 24 * trace_rows;
    let polynomial_headers = 24 * size_of::<Polynomial<Fr>>();
    let input_bytes = dense_factor_entries * field + polynomial_headers;
    let round_working = input_bytes + log_t * field * 8;
    KernelMemoryBudget::new(
        input_bytes as u64,
        round_working as u64,
        (round_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn materialized_opening_rlc_memory<N: jolt_witness::WitnessNamespace>(
    rows: usize,
    components: usize,
) -> KernelMemoryBudget {
    use jolt_backends::{OpeningRlcComponent, OpeningRlcMaterializationResult};
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes =
        rows * components * field + components * size_of::<OpeningRlcComponent<Fr, N>>();
    let peak_working = rows * field
        + components * size_of::<(Fr, &[Fr])>()
        + size_of::<OpeningRlcMaterializationResult<Fr>>();
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn eq_table_memory(log_vars: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let rows = 1usize << log_vars;
    let cached_entries = (1usize << (log_vars + 1)) - 1;
    let vec_headers = 2 * (log_vars + 1) * size_of::<Vec<Fr>>();
    let input_bytes = log_vars * field;
    let peak_working = rows * field + 2 * cached_entries * field + vec_headers;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn eq_aligned_block_memory(log_vars: usize, max_block_entries: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = log_vars * field;
    let peak_working = max_block_entries * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn split_eq_window_memory(
    log_vars: usize,
    window_size: usize,
    rounds: usize,
) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let out_bits = log_vars / 2;
    let in_bits = log_vars.saturating_sub(1 + out_bits);
    let cached_entries = ((1usize << (out_bits + 1)) - 1) + ((1usize << (in_bits + 1)) - 1);
    let vec_headers = (out_bits + in_bits + 2) * size_of::<Vec<Fr>>();
    let active_entries = 1usize << window_size.saturating_sub(1);
    let input_bytes = (log_vars + rounds) * field;
    let peak_working =
        log_vars * field + cached_entries * field + vec_headers + active_entries * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn unipoly_interpolation_memory() -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = (3 + 4 + 3 + 5) * field;
    let peak_working = 8 * size_of::<Vec<Fr>>() + 64 * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn compressed_unipoly_memory() -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = (9 + 8 + 1) * field;
    let peak_working = 2 * 9 * field + 4 * size_of::<Vec<Fr>>();
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn lagrange_many_memory(domain_size: usize, point_count: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = (domain_size + point_count + 2) * field;
    let peak_working = (8 * domain_size + point_count) * field + 4 * size_of::<Vec<Fr>>();
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn compact_polynomial_bind_memory(rows: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = rows * size_of::<u8>();
    let first_bind_outputs = rows * field;
    let retained_compact_inputs = rows * size_of::<u8>() * 2;
    let peak_working = retained_compact_inputs + first_bind_outputs;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn split_eq_polynomial_evaluate_memory(
    rows: usize,
    eq_one_len: usize,
    eq_two_len: usize,
) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = rows * (field + size_of::<u8>()) + (eq_one_len + eq_two_len) * field;
    let peak_working = input_bytes + 2 * size_of::<Vec<Fr>>() + size_of::<Vec<u8>>();
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn inside_out_polynomial_evaluate_memory(rows: usize, point_len: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = rows * (field + size_of::<u8>()) + point_len * field;
    let peak_working = input_bytes + rows * field + 2 * size_of::<Vec<Fr>>() + size_of::<Vec<u8>>();
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn dense_batch_polynomial_evaluate_memory(
    rows: usize,
    num_polys: usize,
    point_len: usize,
) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let split_entries = (1usize << (point_len / 2)) + (1usize << (point_len - point_len / 2));
    let input_bytes = rows * num_polys * field + point_len * field;
    let peak_working = input_bytes + split_entries * field + rows * field + num_polys * field * 128;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 128 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn dense_dot_product_low_optimized_memory(rows: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = rows * 2 * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        input_bytes as u64,
        (input_bytes * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn mixed_polynomial_linear_combination_memory(rows: usize, num_polys: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let compact_inputs = rows * (size_of::<u8>() + size_of::<i64>() + size_of::<u128>());
    let input_bytes = rows * field + compact_inputs + num_polys * field;
    let peak_working = input_bytes + rows * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn one_hot_polynomial_evaluate_memory(k: usize, trace_rows: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let point_len = (k * trace_rows).ilog2() as usize;
    let input_bytes = trace_rows * size_of::<Option<u8>>() + point_len * field;
    let peak_working = input_bytes + (k + trace_rows) * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn one_hot_vector_matrix_product_memory(
    _k: usize,
    trace_rows: usize,
    matrix_rows: usize,
    matrix_columns: usize,
) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = trace_rows * size_of::<Option<u8>>() + matrix_rows * field + field;
    let peak_working = input_bytes + matrix_columns * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn rlc_vector_matrix_product_memory(
    dense_rows: usize,
    one_hot_components: usize,
    matrix_rows: usize,
    matrix_columns: usize,
) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = dense_rows * field
        + one_hot_components * dense_rows * size_of::<Option<u8>>()
        + matrix_rows * field
        + one_hot_components * field;
    let peak_working = input_bytes + matrix_columns * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn stage8_streaming_rlc_memory(
    trace_rows: usize,
    matrix_rows: usize,
    matrix_columns: usize,
    address_columns: usize,
    one_hot_components: usize,
) -> KernelMemoryBudget {
    use jolt_field::Fr;
    use jolt_witness::protocols::jolt_vm::JoltVmStage6Row;

    let field = size_of::<Fr>();
    let input_bytes = trace_rows * size_of::<JoltVmStage6Row>()
        + matrix_rows * field
        + (2 + one_hot_components) * field;
    let folded_tables = one_hot_components * address_columns * field;
    let row_factors = (matrix_rows / address_columns).max(1) * field;
    let accumulators = 2 * matrix_columns * field;
    let peak_working = input_bytes + folded_tables + row_factors + accumulators;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn linear_product_small_degrees_memory(products: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let total_degrees = 2 + 3 + 5 + 6 + 7;
    let input_bytes = products * total_degrees * 2 * field;
    let peak_working = input_bytes + 14 * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn linear_product_d4_memory(products: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = products * 8 * field;
    let peak_working = input_bytes + 8 * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn linear_product_d8_memory(products: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = products * 16 * field;
    let peak_working = input_bytes + 16 * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn linear_product_d16_memory(products: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = products * 32 * field;
    let peak_working = input_bytes + 32 * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn linear_product_d32_memory(products: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = products * 64 * field;
    let peak_working = input_bytes + 64 * field;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn streaming_schedule_memory() -> KernelMemoryBudget {
    let input_bytes = 5 * size_of::<(usize, usize)>();
    let max_rounds = 96usize;
    let peak_working = max_rounds * size_of::<usize>()
        + size_of::<jolt_backends::cpu::schedule::HalfSplitSchedule>()
        + size_of::<jolt_backends::cpu::schedule::LinearOnlySchedule>();
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 64 + 16 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn ra_delayed_materialization_memory(log_len: usize, k: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let len = 1usize << log_len;
    let input_bytes = len * size_of::<Option<u8>>() + k * size_of::<Fr>();
    let split_tables = 8 * k * size_of::<Fr>();
    let materialized = (len / 8) * size_of::<Fr>();
    let peak_working = input_bytes + split_tables + materialized;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn shared_ra_delayed_materialization_memory(
    log_len: usize,
    layout: jolt_backends::cpu::ra::RaFamilyLayout,
) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let len = 1usize << log_len;
    let input_bytes = len * size_of::<jolt_backends::cpu::ra::RaCycleIndices>()
        + layout.num_polys() * layout.k_chunk * size_of::<Fr>();
    let split_tables = 8 * layout.num_polys() * layout.k_chunk * size_of::<Fr>();
    let materialized = layout.num_polys() * (len / 8) * size_of::<Fr>();
    let peak_working = input_bytes + split_tables + materialized;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn ra_pushforward_memory(
    log_len: usize,
    layout: jolt_backends::cpu::ra::RaFamilyLayout,
) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let len = 1usize << log_len;
    let hi_entries = 1usize << (log_len - log_len / 2);
    let lo_entries = 1usize << (log_len / 2);
    let num_polys = layout.num_polys();
    let threads = rayon::current_num_threads().max(1);
    let input_bytes =
        len * size_of::<jolt_backends::cpu::ra::RaCycleIndices>() + log_len * size_of::<Fr>();
    let eq_tables = (hi_entries + lo_entries) * size_of::<Fr>();
    let partials = threads * num_polys * layout.k_chunk * size_of::<Fr>();
    let locals = threads * num_polys * layout.k_chunk * size_of::<Fr>();
    let touched = threads * num_polys * layout.k_chunk * (size_of::<usize>() + size_of::<bool>());
    let output = num_polys * layout.k_chunk * size_of::<Fr>();
    let peak_working = input_bytes + eq_tables + partials + locals + touched + output;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn read_write_one_hot_coeff_lookup_memory() -> KernelMemoryBudget {
    use jolt_field::Fr;

    let field = size_of::<Fr>();
    let input_bytes = (4 + 3) * field;
    let peak_working = (1usize << 16) * field + size_of::<Vec<Fr>>();
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn read_write_cycle_major_bind_memory(
    input_entries: usize,
    bound_entries: usize,
    row_pairs: usize,
) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let entry = 2 * size_of::<usize>() + size_of::<Fr>();
    let input_bytes = input_entries * entry;
    let peak_working = input_entries * entry
        + bound_entries * entry
        + row_pairs * (size_of::<(usize, usize)>() + 4 * size_of::<usize>());
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 4 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn read_write_cycle_major_message_memory(input_entries: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let entry = 2 * size_of::<usize>() + size_of::<Fr>();
    let input_bytes = input_entries * entry;
    let accumulator_bytes = rayon::current_num_threads().max(1)
        * size_of::<<Fr as jolt_field::WithAccumulator>::Accumulator>()
        * 2;
    let peak_working = input_bytes + accumulator_bytes;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn read_write_cycle_to_address_major_memory(input_entries: usize) -> KernelMemoryBudget {
    use jolt_field::Fr;

    let entry = 2 * size_of::<usize>() + size_of::<Fr>();
    let input_bytes = input_entries * entry;
    let peak_working = input_entries * entry * 3;
    KernelMemoryBudget::new(
        input_bytes as u64,
        peak_working as u64,
        (peak_working * 8 + 64 * 1024 * 1024) as u64,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn tensor_eq_entries(log_rows: usize) -> usize {
    let out_bits = log_rows / 2;
    let in_bits = log_rows - out_bits;
    (1usize << out_bits) + (1usize << in_bits)
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "field-inline"),
    not(feature = "zk")
))]
fn tensor_eq_outer_entries(log_rows: usize) -> usize {
    1usize << (log_rows / 2)
}

#[cfg(all(
    feature = "core-fixtures",
    any(
        not(feature = "zk"),
        all(feature = "zk", not(feature = "field-inline"))
    )
))]
fn measure_samples(samples: u32, mut run: impl FnMut()) -> RunMetrics {
    let mut total_ms = 0.0;
    let mut max_peak = 0usize;
    for _ in 0..samples {
        let base = CURRENT_ALLOCATED.load(Ordering::SeqCst);
        PEAK_ALLOCATED.store(base, Ordering::SeqCst);
        let start = Instant::now();
        run();
        total_ms += start.elapsed().as_secs_f64() * 1_000.0;
        max_peak = max_peak.max(PEAK_ALLOCATED.load(Ordering::SeqCst).saturating_sub(base));
    }
    RunMetrics::new(
        Some(total_ms / f64::from(samples)),
        Some(max_peak as u64),
        None,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    any(
        not(feature = "zk"),
        all(feature = "zk", not(feature = "field-inline"))
    )
))]
#[cfg_attr(
    any(feature = "field-inline", feature = "zk"),
    expect(
        dead_code,
        reason = "only some feature-gated evidence writers need setup-aware timing"
    )
)]
fn measure_samples_with_setup<S>(
    samples: u32,
    mut setup: impl FnMut() -> S,
    mut run: impl FnMut(&mut S),
) -> RunMetrics {
    let mut total_ms = 0.0;
    let mut max_peak = 0usize;
    for _ in 0..samples {
        let mut state = setup();
        let base = CURRENT_ALLOCATED.load(Ordering::SeqCst);
        PEAK_ALLOCATED.store(base, Ordering::SeqCst);
        let start = Instant::now();
        run(&mut state);
        total_ms += start.elapsed().as_secs_f64() * 1_000.0;
        max_peak = max_peak.max(PEAK_ALLOCATED.load(Ordering::SeqCst).saturating_sub(base));
        let _ = black_box(&mut state);
    }
    RunMetrics::new(
        Some(total_ms / f64::from(samples)),
        Some(max_peak as u64),
        None,
    )
}

#[cfg(all(
    feature = "core-fixtures",
    any(
        not(feature = "zk"),
        all(feature = "zk", not(feature = "field-inline"))
    )
))]
#[expect(clippy::expect_used)]
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}
