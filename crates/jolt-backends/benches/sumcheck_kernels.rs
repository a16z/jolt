#![expect(unused_results)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    mem::size_of,
    sync::atomic::{AtomicUsize, Ordering},
    sync::Arc,
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_backends::{
    cpu::{eq, lagrange, ra, read_write_matrix, schedule, split_eq, univariate, CpuBackend},
    BackendKernelMetadata, BackendRelationId, BackendValueSlot, OpeningBackend,
    OpeningRlcComponent, OpeningRlcMaterializationRequest, OpeningRlcMaterializationResult,
    SumcheckBackend, SumcheckEvaluationOutput, SumcheckEvaluationRequest,
    SumcheckLinearProductOutput, SumcheckLinearProductQuery, SumcheckLinearProductRequest,
    SumcheckPrefixProductSumQuery, SumcheckPrefixProductSumRequest, SumcheckProductUniskipRequest,
    SumcheckProductUniskipRow, SumcheckRowProductQuery, SumcheckRowProductRequest,
    SumcheckViewEvaluationRequest,
};
use jolt_field::{Fr, FromPrimitiveInt, RingAccumulator, WithAccumulator};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, MaterializationPolicy, NamespaceId, OracleDescriptor,
    OracleKind, OracleRef, OracleViewRequest, PolynomialEncoding, PolynomialView, RetentionHint,
    ViewRequirement, WitnessDimensions, WitnessError, WitnessNamespace, WitnessProvider,
};
use rayon::prelude::*;

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

fn measured_peak_bytes(run: impl FnOnce()) -> usize {
    let base = CURRENT_ALLOCATED.load(Ordering::SeqCst);
    PEAK_ALLOCATED.store(base, Ordering::SeqCst);
    run();
    PEAK_ALLOCATED.load(Ordering::SeqCst).saturating_sub(base)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BenchNamespace {}

impl WitnessNamespace for BenchNamespace {
    type ChallengeId = u8;
    type CommittedId = u8;
    type OpeningId = u8;
    type PublicId = u8;
    type VirtualId = u8;

    const ID: NamespaceId = NamespaceId::new("sumcheck_kernel_bench");
}

struct DenseViewBenchWitness {
    values: Vec<Vec<Fr>>,
    dimensions: WitnessDimensions,
}

impl WitnessProvider<Fr, BenchNamespace> for DenseViewBenchWitness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<BenchNamespace>,
    ) -> Result<OracleDescriptor<BenchNamespace>, WitnessError> {
        let OracleKind::Committed(id) = oracle.kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: BenchNamespace::ID.name,
            });
        };
        if usize::from(id) >= self.values.len() {
            return Err(WitnessError::UnknownOracle {
                namespace: BenchNamespace::ID.name,
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
        oracle: OracleRef<BenchNamespace>,
    ) -> Result<Vec<ViewRequirement<BenchNamespace>>, WitnessError> {
        let _ = self.describe_oracle(oracle)?;
        Ok(vec![ViewRequirement::new(
            oracle,
            PolynomialEncoding::Dense,
            MaterializationPolicy::BackendChoice,
            RetentionHint::ThroughStage8,
        )])
    }

    fn oracle_view(
        &self,
        request: OracleViewRequest<BenchNamespace>,
    ) -> Result<PolynomialView<'_, Fr, BenchNamespace>, WitnessError> {
        let OracleKind::Committed(id) = request.oracle().kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: BenchNamespace::ID.name,
            });
        };
        let values = self
            .values
            .get(usize::from(id))
            .ok_or(WitnessError::UnknownOracle {
                namespace: BenchNamespace::ID.name,
            })?;
        let descriptor = self.describe_oracle(request.oracle())?;
        Ok(PolynomialView::borrowed(descriptor, values))
    }
}

#[derive(Clone, Copy)]
struct KernelShape {
    name: &'static str,
    log_rows: usize,
    columns: usize,
    sparse_rows: usize,
    terms_per_side: usize,
    queries: usize,
}

impl KernelShape {
    const fn rows(self) -> usize {
        1usize << self.log_rows
    }
}

struct KernelFixture {
    shape: KernelShape,
    witness_polynomials: Vec<Vec<Fr>>,
    product_rows: Vec<SumcheckProductUniskipRow>,
    input_columns: Vec<usize>,
    constant_column: usize,
    left_rows: Vec<Vec<(usize, Fr)>>,
    right_rows: Vec<Vec<(usize, Fr)>>,
    linear_queries: Vec<SumcheckLinearProductQuery<Fr>>,
    linear_boolean_queries: Vec<SumcheckLinearProductQuery<Fr>>,
    row_queries: Vec<SumcheckRowProductQuery<Fr>>,
}

#[derive(Clone, Copy)]
struct ViewEvalShape {
    name: &'static str,
    log_rows: usize,
    views: usize,
    unique_points: usize,
}

impl ViewEvalShape {
    const fn rows(self) -> usize {
        1usize << self.log_rows
    }
}

struct ViewEvalFixture {
    shape: ViewEvalShape,
    witness: DenseViewBenchWitness,
    request: SumcheckEvaluationRequest<Fr, BenchNamespace>,
}

#[derive(Clone, Copy)]
struct RlcMaterializationShape {
    name: &'static str,
    log_rows: usize,
    components: usize,
}

impl RlcMaterializationShape {
    const fn rows(self) -> usize {
        1usize << self.log_rows
    }
}

struct RlcMaterializationFixture {
    shape: RlcMaterializationShape,
    witness: DenseViewBenchWitness,
    request: OpeningRlcMaterializationRequest<Fr, BenchNamespace>,
}

impl RlcMaterializationFixture {
    fn new(shape: RlcMaterializationShape) -> Self {
        let rows = shape.rows();
        let values = (0..shape.components)
            .map(|component| {
                (0..rows)
                    .map(|row| field_from_index(40_000 + component * rows + row))
                    .collect()
            })
            .collect();
        let witness = DenseViewBenchWitness {
            values,
            dimensions: WitnessDimensions::new(rows, shape.log_rows),
        };
        let components = (0..shape.components)
            .map(|index| {
                let requirement = ViewRequirement::new(
                    OracleRef::committed(index as u8),
                    PolynomialEncoding::Dense,
                    MaterializationPolicy::BackendChoice,
                    RetentionHint::ThroughStage8,
                );
                OpeningRlcComponent::new(requirement, field_from_index(50_000 + index))
            })
            .collect();
        let request =
            OpeningRlcMaterializationRequest::new("bench.opening_rlc_materialization", components);
        Self {
            shape,
            witness,
            request,
        }
    }

    fn analytical_memory(&self) -> AnalyticalMemory {
        let rows = self.shape.rows();
        let field = size_of::<Fr>();
        let input = rows * self.shape.components * field;
        let peak_working = rows * field
            + self.shape.components * size_of::<OpeningRlcComponent<Fr, BenchNamespace>>();
        AnalyticalMemory {
            input,
            peak_working,
            budget: peak_working * 4 + 16 * 1024 * 1024,
        }
    }
}

impl ViewEvalFixture {
    fn new(shape: ViewEvalShape) -> Self {
        let rows = shape.rows();
        let values = (0..shape.views)
            .map(|view| {
                (0..rows)
                    .map(|row| field_from_index(10_000 + view * rows + row))
                    .collect()
            })
            .collect();
        let witness = DenseViewBenchWitness {
            values,
            dimensions: WitnessDimensions::new(rows, shape.log_rows),
        };
        let points = (0..shape.unique_points)
            .map(|index| point(shape.log_rows, 900 + index))
            .collect::<Vec<_>>();
        let views = (0..shape.views)
            .map(|index| {
                let requirement = ViewRequirement::new(
                    OracleRef::committed(index as u8),
                    PolynomialEncoding::Dense,
                    MaterializationPolicy::BackendChoice,
                    RetentionHint::ThroughStage8,
                );
                SumcheckViewEvaluationRequest::new(
                    BackendValueSlot(index as u32),
                    requirement,
                    points[index % points.len()].clone(),
                )
            })
            .collect();
        let request = SumcheckEvaluationRequest::new("bench.view_evaluations", views);
        Self {
            shape,
            witness,
            request,
        }
    }

    fn analytical_memory(&self) -> AnalyticalMemory {
        let rows = self.shape.rows();
        let field = size_of::<Fr>();
        let input = rows * self.shape.views * field;
        let max_views_per_point = self.shape.views.div_ceil(self.shape.unique_points.max(1));
        let peak_working = tensor_eq_entries(self.shape.log_rows) * field
            + tensor_eq_outer_entries(self.shape.log_rows) * max_views_per_point * field
            + self.shape.views * self.shape.log_rows * field
            + self.shape.views * size_of::<SumcheckEvaluationOutput<Fr>>();
        AnalyticalMemory {
            input,
            peak_working,
            budget: peak_working * 4 + 16 * 1024 * 1024,
        }
    }
}

impl KernelFixture {
    fn new(shape: KernelShape) -> Self {
        let rows = shape.rows();
        let witness_polynomials = (0..shape.columns)
            .map(|column| {
                (0..rows)
                    .map(|row| field_from_index(1 + column * rows + row))
                    .collect()
            })
            .collect();
        let input_columns = (0..shape.columns).collect::<Vec<_>>();
        let constant_column = shape.columns;
        let left_rows = sparse_rows(shape, 11, constant_column);
        let right_rows = sparse_rows(shape, 23, constant_column);
        let linear_queries = (0..shape.queries)
            .map(|query| {
                SumcheckLinearProductQuery::new(
                    BackendValueSlot(query as u32),
                    point(shape.log_rows, 101 + query),
                    row_weights(shape.sparse_rows, 201 + query),
                    field_from_index(301 + query),
                )
            })
            .collect();
        let linear_boolean_queries = (0..shape.queries)
            .map(|query| {
                SumcheckLinearProductQuery::new(
                    BackendValueSlot(query as u32),
                    boolean_point(shape.log_rows, query),
                    row_weights(shape.sparse_rows, 701 + query),
                    field_from_index(801 + query),
                )
            })
            .collect();
        let row_queries = (0..shape.queries)
            .map(|query| {
                SumcheckRowProductQuery::new(
                    BackendValueSlot(query as u32),
                    point(shape.log_rows, 401 + query),
                    row_weights(shape.sparse_rows, 501 + query),
                    field_from_index(601 + query),
                )
            })
            .collect();

        Self {
            shape,
            witness_polynomials,
            product_rows: Vec::new(),
            input_columns,
            constant_column,
            left_rows,
            right_rows,
            linear_queries,
            linear_boolean_queries,
            row_queries,
        }
    }

    fn new_product_uniskip(shape: KernelShape) -> Self {
        let rows = shape.rows();
        let product_rows = product_uniskip_rows(rows);
        let witness_polynomials = product_uniskip_witness_polynomials(&product_rows);
        let input_columns = (0..shape.columns).collect::<Vec<_>>();
        let constant_column = shape.columns;
        let row_queries = (0..shape.queries)
            .map(|query| {
                SumcheckRowProductQuery::new(
                    BackendValueSlot(query as u32),
                    point(shape.log_rows, 1_401 + query),
                    row_weights(PRODUCT_UNISKIP_ROWS, 1_501 + query),
                    field_from_index(1_601 + query),
                )
            })
            .collect();

        Self {
            shape,
            witness_polynomials,
            product_rows,
            input_columns,
            constant_column,
            left_rows: product_uniskip_left_rows(),
            right_rows: product_uniskip_right_rows(constant_column),
            linear_queries: Vec::new(),
            linear_boolean_queries: Vec::new(),
            row_queries,
        }
    }

    fn linear_request(&self) -> SumcheckLinearProductRequest<'_, Fr> {
        SumcheckLinearProductRequest::new(
            "bench.spartan_outer.linear_products",
            &self.witness_polynomials,
            &self.input_columns,
            self.constant_column,
            &self.left_rows,
            &self.right_rows,
            self.linear_queries.clone(),
        )
        .with_kernel_metadata(BackendKernelMetadata::new(
            Some(BackendRelationId::new("jolt_vm", "spartan_outer.remainder")),
            &["OPT-SC-007", "OPT-SC-008", "OPT-EQ-004"],
        ))
    }

    fn linear_boolean_request(&self) -> SumcheckLinearProductRequest<'_, Fr> {
        SumcheckLinearProductRequest::new(
            "bench.spartan_outer.boolean_linear_products",
            &self.witness_polynomials,
            &self.input_columns,
            self.constant_column,
            &self.left_rows,
            &self.right_rows,
            self.linear_boolean_queries.clone(),
        )
        .with_kernel_metadata(BackendKernelMetadata::new(
            Some(BackendRelationId::new(
                "jolt_vm",
                "spartan_outer.uniskip_first_round",
            )),
            &["OPT-SC-007", "OPT-SC-008", "OPT-EQ-004"],
        ))
    }

    fn outer_uniskip_sum_request(&self) -> SumcheckPrefixProductSumRequest<'_, Fr> {
        let shared_point = point(self.shape.log_rows + 1, 1_901);
        let queries = (0..self.shape.queries)
            .map(|query| {
                SumcheckPrefixProductSumQuery::new(
                    BackendValueSlot(query as u32),
                    shared_point.clone(),
                    Vec::new(),
                    self.shape.log_rows + 1,
                    spartan_outer_stream_weights(self.shape.sparse_rows, 2_001 + query, false),
                    spartan_outer_stream_weights(self.shape.sparse_rows, 2_101 + query, true),
                    field_from_index(2_201 + query),
                )
            })
            .collect();
        SumcheckPrefixProductSumRequest::new(
            "bench.spartan_outer.uniskip_prefix_sum",
            &self.witness_polynomials,
            &self.input_columns,
            self.constant_column,
            &self.left_rows,
            &self.right_rows,
            queries,
        )
        .with_kernel_metadata(BackendKernelMetadata::new(
            Some(BackendRelationId::new(
                "jolt_vm",
                "spartan_outer.uniskip_first_round",
            )),
            &["OPT-SC-007", "OPT-SC-008", "OPT-EQ-004"],
        ))
    }

    fn outer_remainder_bound_prefix_request(&self) -> SumcheckPrefixProductSumRequest<'_, Fr> {
        let shared_point = point(self.shape.log_rows + 1, 2_301);
        let stream = field_from_index(2_401);
        let row_weights_at_zero =
            spartan_outer_stream_weights(self.shape.sparse_rows, 2_501, false);
        let row_weights_at_one = spartan_outer_stream_weights(self.shape.sparse_rows, 2_601, true);
        let queries = (0..self.shape.queries)
            .map(|query| {
                SumcheckPrefixProductSumQuery::new(
                    BackendValueSlot(query as u32),
                    shared_point.clone(),
                    vec![stream, field_from_index(2_701 + query)],
                    self.shape.log_rows - 1,
                    row_weights_at_zero.clone(),
                    row_weights_at_one.clone(),
                    field_from_index(2_801 + query),
                )
            })
            .collect();
        SumcheckPrefixProductSumRequest::new(
            "bench.spartan_outer.remainder_bound_prefix_sum",
            &self.witness_polynomials,
            &self.input_columns,
            self.constant_column,
            &self.left_rows,
            &self.right_rows,
            queries,
        )
        .with_kernel_metadata(BackendKernelMetadata::new(
            Some(BackendRelationId::new("jolt_vm", "spartan_outer.remainder")),
            &["OPT-SC-007", "OPT-SC-008", "OPT-EQ-004"],
        ))
    }

    fn row_request(&self) -> SumcheckRowProductRequest<'_, Fr> {
        SumcheckRowProductRequest::new(
            "bench.spartan_product.row_products",
            &self.witness_polynomials,
            &self.input_columns,
            self.constant_column,
            &self.left_rows,
            &self.right_rows,
            self.row_queries.clone(),
        )
        .with_kernel_metadata(BackendKernelMetadata::new(
            Some(BackendRelationId::new(
                "jolt_vm",
                "spartan_product.uniskip_first_round",
            )),
            &["OPT-SC-007", "OPT-EQ-004"],
        ))
    }

    fn grouped_row_request(&self) -> SumcheckRowProductRequest<'_, Fr> {
        let shared_point = point(self.shape.log_rows, 1_337);
        let queries = self
            .row_queries
            .iter()
            .cloned()
            .map(|mut query| {
                query.eq_point.clone_from(&shared_point);
                query
            })
            .collect();
        SumcheckRowProductRequest::new(
            "bench.spartan_product.grouped_row_products",
            &self.witness_polynomials,
            &self.input_columns,
            self.constant_column,
            &self.left_rows,
            &self.right_rows,
            queries,
        )
        .with_kernel_metadata(BackendKernelMetadata::new(
            Some(BackendRelationId::new(
                "jolt_vm",
                "spartan_product.uniskip_first_round",
            )),
            &["OPT-SC-007", "OPT-EQ-004"],
        ))
    }

    fn product_uniskip_request(&self) -> SumcheckRowProductRequest<'_, Fr> {
        let shared_point = point(self.shape.log_rows, 1_707);
        let queries = self
            .row_queries
            .iter()
            .cloned()
            .map(|mut query| {
                query.eq_point.clone_from(&shared_point);
                query
            })
            .collect();
        SumcheckRowProductRequest::new(
            "bench.spartan_product.product_uniskip",
            &self.witness_polynomials,
            &self.input_columns,
            self.constant_column,
            &self.left_rows,
            &self.right_rows,
            queries,
        )
        .with_kernel_metadata(BackendKernelMetadata::new(
            Some(BackendRelationId::new(
                "jolt_vm",
                "spartan_product.uniskip_first_round",
            )),
            &["OPT-SC-007", "OPT-EQ-004"],
        ))
    }

    fn raw_product_uniskip_request(&self) -> SumcheckProductUniskipRequest<'_, Fr> {
        SumcheckProductUniskipRequest::new(
            "bench.spartan_product.product_uniskip_raw",
            &self.product_rows,
            product_uniskip_extended_queries(self.shape.log_rows),
        )
        .with_kernel_metadata(BackendKernelMetadata::new(
            Some(BackendRelationId::new(
                "jolt_vm",
                "spartan_product.uniskip_first_round",
            )),
            &["OPT-SC-007", "OPT-EQ-004"],
        ))
    }

    fn analytical_memory(&self, kind: KernelKind) -> AnalyticalMemory {
        let shape = self.shape;
        let field = size_of::<Fr>();
        let rows = shape.rows();
        let sparse_terms = 2 * shape.sparse_rows * (shape.terms_per_side + 1);
        let query_count = shape.queries;
        let active_queries = query_count.min(rayon::current_num_threads()).max(1);

        let witness_bytes = rows * shape.columns * field;
        let raw_row_bytes = rows * size_of::<SumcheckProductUniskipRow>();
        let sparse_row_bytes = 2 * shape.sparse_rows * size_of::<Vec<(usize, Fr)>>()
            + sparse_terms * size_of::<(usize, Fr)>();
        let point_bytes = query_count * shape.log_rows * field;
        let row_weight_bytes = query_count * shape.sparse_rows * field;
        let query_bytes = match kind {
            KernelKind::Linear | KernelKind::LinearBoolean => {
                query_count * size_of::<SumcheckLinearProductQuery<Fr>>()
                    + point_bytes
                    + row_weight_bytes
            }
            KernelKind::PrefixProductSum => {
                query_count * size_of::<SumcheckPrefixProductSumQuery<Fr>>()
                    + point_bytes
                    + 2 * row_weight_bytes
            }
            KernelKind::PrefixProductBound => {
                query_count * size_of::<SumcheckPrefixProductSumQuery<Fr>>()
                    + point_bytes
                    + 2 * row_weight_bytes
            }
            KernelKind::Row | KernelKind::RowGrouped | KernelKind::ProductUniskipRaw => {
                query_count * size_of::<SumcheckRowProductQuery<Fr>>()
                    + point_bytes
                    + row_weight_bytes
            }
        };

        let layout_bytes = shape.columns * size_of::<(usize, usize)>() * 4;
        let slot_set_bytes = query_count * size_of::<BackendValueSlot>() * 4;
        let output_bytes = query_count * size_of::<SumcheckLinearProductOutput<Fr>>();
        let eq_tables = match kind {
            KernelKind::LinearBoolean | KernelKind::PrefixProductSum => 0,
            KernelKind::Linear => active_queries * rows * field,
            KernelKind::PrefixProductBound => tensor_eq_entries(shape.log_rows) * field,
            KernelKind::Row => active_queries * tensor_eq_entries(shape.log_rows) * field,
            KernelKind::RowGrouped | KernelKind::ProductUniskipRaw => {
                tensor_eq_entries(shape.log_rows) * field
                    + tensor_eq_outer_entries(shape.log_rows) * shape.queries * field
            }
        };
        let per_query_scratch = match kind {
            KernelKind::LinearBoolean | KernelKind::PrefixProductSum => 0,
            KernelKind::PrefixProductBound => 4 * rows * field,
            KernelKind::Linear | KernelKind::Row => active_queries * shape.columns * field,
            KernelKind::RowGrouped | KernelKind::ProductUniskipRaw => 0,
        };
        let peak_working_bytes =
            layout_bytes + slot_set_bytes + output_bytes + eq_tables + per_query_scratch;
        let input_bytes = match kind {
            KernelKind::ProductUniskipRaw => raw_row_bytes + query_bytes,
            _ => witness_bytes + sparse_row_bytes + query_bytes,
        };

        AnalyticalMemory {
            input: input_bytes,
            peak_working: peak_working_bytes,
            budget: peak_working_bytes * 4 + 16 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy)]
enum KernelKind {
    Linear,
    LinearBoolean,
    PrefixProductSum,
    PrefixProductBound,
    Row,
    RowGrouped,
    ProductUniskipRaw,
}

const PRODUCT_UNISKIP_ROWS: usize = 3;
const PRODUCT_UNISKIP_EXTENDED_EVALS: usize = 2;
const PRODUCT_UNISKIP_EXTENDED_COEFFS: [[i64; PRODUCT_UNISKIP_ROWS];
    PRODUCT_UNISKIP_EXTENDED_EVALS] = [[3, -3, 1], [1, -3, 3]];

#[derive(Clone, Copy)]
struct AnalyticalMemory {
    input: usize,
    peak_working: usize,
    budget: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct BenchCycleMajorEntry {
    row: usize,
    column: usize,
    value: Fr,
}

#[derive(Clone, Copy, Debug, Default)]
struct BenchAddressMajorEntry {
    row: usize,
    column: usize,
    value: Fr,
}

#[derive(Clone, Copy, Debug, Default)]
struct BenchAddressMajorValueEntry {
    row: usize,
    column: usize,
    prev_val: Fr,
    next_val: Fr,
    val_coeff: Fr,
    ra_coeff: Fr,
}

impl read_write_matrix::CycleMajorMatrixEntry<Fr> for BenchCycleMajorEntry {
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
        _: Option<&read_write_matrix::OneHotCoeffTable<Fr>>,
        _: Option<&read_write_matrix::OneHotCoeffTable<Fr>>,
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

impl read_write_matrix::AddressMajorMatrixEntry<Fr> for BenchAddressMajorEntry {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        self.column
    }
}

impl read_write_matrix::AddressMajorMatrixEntry<Fr> for BenchAddressMajorValueEntry {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        self.column
    }
}

impl read_write_matrix::AddressMajorBindableEntry<Fr> for BenchAddressMajorValueEntry {
    fn prev_val(&self) -> Fr {
        self.prev_val
    }

    fn next_val(&self) -> Fr {
        self.next_val
    }

    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        even_checkpoint: Fr,
        odd_checkpoint: Fr,
        r: Fr,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => Self {
                row: even.row,
                column: even.column / 2,
                prev_val: even.prev_val + r * (odd.prev_val - even.prev_val),
                next_val: even.next_val + r * (odd.next_val - even.next_val),
                val_coeff: even.val_coeff + r * (odd.val_coeff - even.val_coeff),
                ra_coeff: even.ra_coeff + r * (odd.ra_coeff - even.ra_coeff),
            },
            (Some(even), None) => Self {
                row: even.row,
                column: even.column / 2,
                prev_val: even.prev_val + r * (odd_checkpoint - even.prev_val),
                next_val: even.next_val + r * (odd_checkpoint - even.next_val),
                val_coeff: even.val_coeff + r * (odd_checkpoint - even.val_coeff),
                ra_coeff: (Fr::from_u64(1) - r) * even.ra_coeff,
            },
            (None, Some(odd)) => Self {
                row: odd.row,
                column: odd.column / 2,
                prev_val: even_checkpoint + r * (odd.prev_val - even_checkpoint),
                next_val: even_checkpoint + r * (odd.next_val - even_checkpoint),
                val_coeff: even_checkpoint + r * (odd.val_coeff - even_checkpoint),
                ra_coeff: r * odd.ra_coeff,
            },
            (None, None) => unreachable!("address-major bind requires at least one entry"),
        }
    }
}

impl read_write_matrix::AddressMajorMessageEntry<Fr> for BenchAddressMajorValueEntry {
    fn accumulate_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inputs: read_write_matrix::AddressMajorMessageInputs<Fr>,
        accumulators: &mut [<Fr as WithAccumulator>::Accumulator; 2],
    ) {
        let read_write_matrix::AddressMajorMessageInputs {
            even_checkpoint,
            odd_checkpoint,
            inc_eval,
            eq_eval,
            gamma,
        } = inputs;
        let (ra_evals, val_evals) = match (even, odd) {
            (Some(even), Some(odd)) => (
                [even.ra_coeff, odd.ra_coeff + odd.ra_coeff - even.ra_coeff],
                [
                    even.val_coeff,
                    odd.val_coeff + odd.val_coeff - even.val_coeff,
                ],
            ),
            (Some(even), None) => (
                [even.ra_coeff, -even.ra_coeff],
                [
                    even.val_coeff,
                    odd_checkpoint + odd_checkpoint - even.val_coeff,
                ],
            ),
            (None, Some(odd)) => (
                [Fr::from_u64(0), odd.ra_coeff + odd.ra_coeff],
                [
                    even_checkpoint,
                    odd.val_coeff + odd.val_coeff - even_checkpoint,
                ],
            ),
            (None, None) => unreachable!("address-major message requires at least one entry"),
        };
        accumulators[0].fmadd(
            eq_eval,
            ra_evals[0] * (val_evals[0] + gamma * (inc_eval + val_evals[0])),
        );
        accumulators[1].fmadd(
            eq_eval,
            ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1])),
        );
    }
}

impl read_write_matrix::CycleMajorToAddressMajor<Fr> for BenchCycleMajorEntry {
    type AddressMajor = BenchAddressMajorEntry;

    fn to_address_major(
        self,
        _: Option<&read_write_matrix::OneHotCoeffTable<Fr>>,
        _: Option<&read_write_matrix::OneHotCoeffTable<Fr>>,
    ) -> Self::AddressMajor {
        BenchAddressMajorEntry {
            row: self.row,
            column: self.column,
            value: self.value,
        }
    }
}

impl read_write_matrix::CycleMajorMessageEntry<Fr> for BenchCycleMajorEntry {
    fn accumulate_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [Fr; 2],
        gamma: Fr,
        accumulators: &mut [<Fr as WithAccumulator>::Accumulator; 2],
        _: Option<&read_write_matrix::OneHotCoeffTable<Fr>>,
        _: Option<&read_write_matrix::OneHotCoeffTable<Fr>>,
    ) {
        let [eval_at_zero, eval_slope] = cycle_major_entry_evals(even, odd);
        accumulators[0].fmadd(eval_at_zero, inc_evals[0] + gamma);
        accumulators[1].fmadd(eval_slope, inc_evals[1] + gamma);
    }
}

fn sparse_rows(shape: KernelShape, salt: usize, constant_column: usize) -> Vec<Vec<(usize, Fr)>> {
    (0..shape.sparse_rows)
        .map(|row| {
            let mut terms = (0..shape.terms_per_side)
                .map(|term| {
                    (
                        (row + term + salt) % shape.columns,
                        field_from_index(salt * 1_000 + row * 17 + term),
                    )
                })
                .collect::<Vec<_>>();
            terms.push((constant_column, field_from_index(salt * 10_000 + row)));
            terms
        })
        .collect()
}

fn product_uniskip_left_rows() -> Vec<Vec<(usize, Fr)>> {
    vec![
        vec![(0, Fr::from_u64(1))],
        vec![(1, Fr::from_u64(1))],
        vec![(2, Fr::from_u64(1))],
    ]
}

fn product_uniskip_right_rows(constant_column: usize) -> Vec<Vec<(usize, Fr)>> {
    vec![
        vec![(3, Fr::from_u64(1))],
        vec![(4, Fr::from_u64(1))],
        vec![(constant_column, Fr::from_u64(1)), (5, Fr::from_i64(-1))],
    ]
}

fn product_uniskip_rows(rows: usize) -> Vec<SumcheckProductUniskipRow> {
    (0..rows)
        .map(|row| {
            SumcheckProductUniskipRow::new(
                (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                ((row as u64).wrapping_mul(17) ^ 1) & 1,
                row % 5 == 0,
                (row as i128 * 1_000_003) - (rows as i128 / 3),
                row % 7 == 0,
                row % 11 == 0,
            )
        })
        .collect()
}

fn product_uniskip_witness_polynomials(rows: &[SumcheckProductUniskipRow]) -> Vec<Vec<Fr>> {
    vec![
        rows.iter()
            .map(|row| Fr::from_u64(row.left_instruction))
            .collect(),
        rows.iter()
            .map(|row| Fr::from_u64(row.lookup_output))
            .collect(),
        rows.iter()
            .map(|row| Fr::from_bool(row.jump_flag))
            .collect(),
        rows.iter()
            .map(|row| Fr::from_i128(row.right_instruction))
            .collect(),
        rows.iter()
            .map(|row| Fr::from_bool(row.branch_flag))
            .collect(),
        rows.iter()
            .map(|row| Fr::from_bool(row.next_is_noop))
            .collect(),
    ]
}

fn product_uniskip_extended_queries(log_rows: usize) -> Vec<SumcheckRowProductQuery<Fr>> {
    let shared_point = point(log_rows, 1_707);
    PRODUCT_UNISKIP_EXTENDED_COEFFS
        .iter()
        .enumerate()
        .map(|(query, coeffs)| {
            SumcheckRowProductQuery::new(
                BackendValueSlot(query as u32),
                shared_point.clone(),
                coeffs.iter().copied().map(Fr::from_i64).collect(),
                Fr::from_u64(1),
            )
        })
        .collect()
}

fn row_weights(count: usize, salt: usize) -> Vec<Fr> {
    (0..count)
        .map(|index| field_from_index(salt * 1_000 + index))
        .collect()
}

fn spartan_outer_stream_weights(count: usize, salt: usize, stream_one: bool) -> Vec<Fr> {
    const FIRST_GROUP: [usize; 10] = [1, 2, 3, 4, 5, 6, 11, 14, 17, 18];
    const SECOND_GROUP: [usize; 9] = [0, 7, 8, 9, 10, 12, 13, 15, 16];
    let active = if stream_one {
        SECOND_GROUP.as_slice()
    } else {
        FIRST_GROUP.as_slice()
    };
    let mut weights = vec![Fr::from_u64(0); count];
    for (index, &row) in active.iter().enumerate() {
        if row < count {
            weights[row] = field_from_index(salt * 1_000 + index);
        }
    }
    weights
}

fn point(len: usize, salt: usize) -> Vec<Fr> {
    (0..len)
        .map(|index| field_from_index(salt * 1_000 + index))
        .collect()
}

fn boolean_point(len: usize, assignment: usize) -> Vec<Fr> {
    (0..len)
        .rev()
        .map(|bit| {
            if ((assignment >> bit) & 1) == 0 {
                Fr::from_u64(0)
            } else {
                Fr::from_u64(1)
            }
        })
        .collect()
}

fn field_from_index(index: usize) -> Fr {
    Fr::from_u64((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn ra_lookup_indices(log_len: usize, k: usize) -> Vec<Option<u8>> {
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

fn ra_eq_evals(k: usize) -> Vec<Fr> {
    (0..k)
        .map(|index| field_from_index(910_000 + index))
        .collect()
}

fn ra_delayed_materialization_memory(log_len: usize, k: usize) -> AnalyticalMemory {
    let len = 1usize << log_len;
    let input = len * size_of::<Option<u8>>() + k * size_of::<Fr>();
    let split_tables = 8 * k * size_of::<Fr>();
    let materialized = (len / 8) * size_of::<Fr>();
    let peak_working = input + split_tables + materialized;
    AnalyticalMemory {
        input,
        peak_working,
        budget: peak_working * 8 + 16 * 1024 * 1024,
    }
}

fn ra_fingerprint(poly: &ra::RaPolynomial<u8, Fr>) -> Fr {
    (0..poly.len())
        .step_by((poly.len() / 64).max(1))
        .fold(Fr::from_u64(poly.len() as u64), |acc, index| {
            acc + poly.get_bound_coeff(index)
        })
}

fn shared_ra_layout() -> ra::RaFamilyLayout {
    ra::RaFamilyLayout::new(16, 32, 6, 8)
}

fn shared_ra_indices(log_len: usize, layout: ra::RaFamilyLayout) -> Vec<ra::RaCycleIndices> {
    (0..(1usize << log_len))
        .map(|row| {
            let mut instruction = [0u8; ra::MAX_INSTRUCTION_CHUNKS];
            let mut bytecode = [0u8; ra::MAX_BYTECODE_CHUNKS];
            let mut ram = [None; ra::MAX_RAM_CHUNKS];
            for (chunk, value) in instruction
                .iter_mut()
                .enumerate()
                .take(layout.instruction_chunks)
            {
                *value = ((row * 13 + chunk * 5 + (row >> 2)) % layout.k_chunk) as u8;
            }
            for (chunk, value) in bytecode.iter_mut().enumerate().take(layout.bytecode_chunks) {
                *value = ((row * 7 + chunk * 11 + (row >> 3)) % layout.k_chunk) as u8;
            }
            for (chunk, value) in ram.iter_mut().enumerate().take(layout.ram_chunks) {
                if (row + chunk) % 5 != 0 {
                    *value = Some(((row * 3 + chunk * 17 + (row >> 1)) % layout.k_chunk) as u8);
                }
            }
            ra::RaCycleIndices {
                instruction,
                bytecode,
                ram,
            }
        })
        .collect()
}

fn shared_ra_tables(layout: ra::RaFamilyLayout) -> Vec<Vec<Fr>> {
    (0..layout.num_polys())
        .map(|poly_idx| {
            (0..layout.k_chunk)
                .map(|entry| field_from_index(930_000 + poly_idx * 101 + entry * 19))
                .collect()
        })
        .collect()
}

fn shared_ra_delayed_materialization_memory(
    log_len: usize,
    layout: ra::RaFamilyLayout,
) -> AnalyticalMemory {
    let len = 1usize << log_len;
    let input = len * size_of::<ra::RaCycleIndices>()
        + layout.num_polys() * layout.k_chunk * size_of::<Fr>();
    let split_tables = 8 * layout.num_polys() * layout.k_chunk * size_of::<Fr>();
    let materialized = layout.num_polys() * (len / 8) * size_of::<Fr>();
    let peak_working = input + split_tables + materialized;
    AnalyticalMemory {
        input,
        peak_working,
        budget: peak_working * 8 + 16 * 1024 * 1024,
    }
}

fn ra_pushforward_memory(log_len: usize, layout: ra::RaFamilyLayout) -> AnalyticalMemory {
    let len = 1usize << log_len;
    let hi_entries = 1usize << (log_len - log_len / 2);
    let lo_entries = 1usize << (log_len / 2);
    let num_polys = layout.num_polys();
    let threads = rayon::current_num_threads().max(1);
    let input = len * size_of::<ra::RaCycleIndices>() + log_len * size_of::<Fr>();
    let eq_tables = (hi_entries + lo_entries) * size_of::<Fr>();
    let partials = threads * num_polys * layout.k_chunk * size_of::<Fr>();
    let locals = threads * num_polys * layout.k_chunk * size_of::<Fr>();
    let touched = threads * num_polys * layout.k_chunk * (size_of::<usize>() + size_of::<bool>());
    let output = num_polys * layout.k_chunk * size_of::<Fr>();
    let peak_working = input + eq_tables + partials + locals + touched + output;
    AnalyticalMemory {
        input,
        peak_working,
        budget: peak_working * 8 + 16 * 1024 * 1024,
    }
}

fn shared_ra_fingerprint(polys: &ra::SharedRaPolynomials<Fr>) -> Fr {
    let mut total = Fr::from_u64((polys.num_polys() ^ polys.len()) as u64);
    for poly_idx in 0..polys.num_polys() {
        for row in (0..polys.len()).step_by((polys.len() / 32).max(1)) {
            total += polys.get_bound_coeff(poly_idx, row);
        }
    }
    total
}

fn ra_pushforward_fingerprint(tables: &[Vec<Fr>]) -> Fr {
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

fn cycle_major_bind_entries(row_pairs: usize, terms_per_row: usize) -> Vec<BenchCycleMajorEntry> {
    let mut entries = Vec::with_capacity(row_pairs * terms_per_row * 2);
    for pair in 0..row_pairs {
        let even_row = 2 * pair;
        let odd_row = even_row + 1;
        for index in 0..terms_per_row {
            entries.push(BenchCycleMajorEntry {
                row: even_row,
                column: index * 2,
                value: field_from_index(230_000 + pair * terms_per_row + index),
            });
        }
        for index in 0..terms_per_row {
            entries.push(BenchCycleMajorEntry {
                row: odd_row,
                column: if index % 3 == 0 {
                    index * 2
                } else {
                    index * 2 + 1
                },
                value: field_from_index(330_000 + pair * terms_per_row + index),
            });
        }
    }
    entries
}

fn address_major_bind_entries(
    column_pairs: usize,
    terms_per_col: usize,
) -> Vec<BenchAddressMajorValueEntry> {
    let mut entries = Vec::with_capacity(column_pairs * terms_per_col * 2);
    for pair in 0..column_pairs {
        let even_col = 2 * pair;
        let odd_col = even_col + 1;
        for index in 0..terms_per_col {
            entries.push(BenchAddressMajorValueEntry {
                row: index * 2,
                column: even_col,
                prev_val: field_from_index(510_000 + pair * terms_per_col + index),
                next_val: field_from_index(520_000 + pair * terms_per_col + index),
                val_coeff: field_from_index(530_000 + pair * terms_per_col + index),
                ra_coeff: field_from_index(540_000 + pair * terms_per_col + index),
            });
        }
        for index in 0..terms_per_col {
            entries.push(BenchAddressMajorValueEntry {
                row: if index % 3 == 0 {
                    index * 2
                } else {
                    index * 2 + 1
                },
                column: odd_col,
                prev_val: field_from_index(610_000 + pair * terms_per_col + index),
                next_val: field_from_index(620_000 + pair * terms_per_col + index),
                val_coeff: field_from_index(630_000 + pair * terms_per_col + index),
                ra_coeff: field_from_index(640_000 + pair * terms_per_col + index),
            });
        }
    }
    entries
}

fn cycle_major_bind_bound_len(row_pairs: usize, terms_per_row: usize) -> usize {
    row_pairs * (terms_per_row + terms_per_row - terms_per_row.div_ceil(3))
}

fn address_major_bind_bound_len(column_pairs: usize, terms_per_col: usize) -> usize {
    column_pairs * (terms_per_col + terms_per_col - terms_per_col.div_ceil(3))
}

fn cycle_major_bind_memory(
    input_entries: usize,
    bound_entries: usize,
    row_pairs: usize,
) -> AnalyticalMemory {
    let input = input_entries * size_of::<BenchCycleMajorEntry>();
    let output = bound_entries * size_of::<BenchCycleMajorEntry>();
    let row_metadata = row_pairs
        * (size_of::<(usize, usize)>()
            + size_of::<&[BenchCycleMajorEntry]>()
            + size_of::<&mut [std::mem::MaybeUninit<BenchCycleMajorEntry>]>());
    AnalyticalMemory {
        input,
        peak_working: input + output + row_metadata,
        budget: (input + output + row_metadata) * 4 + 16 * 1024 * 1024,
    }
}

fn address_major_bind_memory(
    input_entries: usize,
    bound_entries: usize,
    column_pairs: usize,
) -> AnalyticalMemory {
    let input = input_entries * size_of::<BenchAddressMajorValueEntry>()
        + column_pairs * 2 * size_of::<Fr>();
    let output =
        bound_entries * size_of::<BenchAddressMajorValueEntry>() + column_pairs * size_of::<Fr>();
    let col_metadata = column_pairs
        * (size_of::<(usize, usize)>()
            + size_of::<&[BenchAddressMajorValueEntry]>()
            + size_of::<&mut [std::mem::MaybeUninit<BenchAddressMajorValueEntry>]>());
    AnalyticalMemory {
        input,
        peak_working: input + output + col_metadata,
        budget: (input + output + col_metadata) * 4 + 16 * 1024 * 1024,
    }
}

fn cycle_major_bind_fingerprint(entries: &[BenchCycleMajorEntry]) -> Fr {
    entries
        .iter()
        .step_by((entries.len() / 64).max(1))
        .fold(Fr::from_u64(entries.len() as u64), |acc, entry| {
            acc + entry.value + Fr::from_u64((entry.row ^ entry.column) as u64)
        })
}

fn address_major_bind_fingerprint(entries: &[BenchAddressMajorValueEntry]) -> Fr {
    entries.iter().step_by((entries.len() / 64).max(1)).fold(
        Fr::from_u64(entries.len() as u64),
        |acc, entry| {
            acc + entry.val_coeff
                + entry.ra_coeff
                + entry.prev_val
                + entry.next_val
                + Fr::from_u64((entry.row ^ entry.column) as u64)
        },
    )
}

fn cycle_major_message_rows(
    entries: &[BenchCycleMajorEntry],
) -> (&[BenchCycleMajorEntry], &[BenchCycleMajorEntry]) {
    let odd_start = entries.partition_point(|entry| entry.row.is_multiple_of(2));
    entries.split_at(odd_start)
}

fn address_major_message_cols(
    entries: &[BenchAddressMajorValueEntry],
) -> (
    &[BenchAddressMajorValueEntry],
    &[BenchAddressMajorValueEntry],
) {
    let odd_start = entries.partition_point(|entry| entry.column.is_multiple_of(2));
    entries.split_at(odd_start)
}

fn cycle_major_message_fingerprint(evals: [Fr; 2]) -> Fr {
    evals[0] + evals[1]
}

fn address_major_message_fingerprint(evals: [Fr; 2]) -> Fr {
    evals[0] + evals[1]
}

fn cycle_major_entry_evals(
    even: Option<&BenchCycleMajorEntry>,
    odd: Option<&BenchCycleMajorEntry>,
) -> [Fr; 2] {
    match (even, odd) {
        (Some(even), Some(odd)) => [even.value, odd.value - even.value],
        (Some(even), None) => [even.value, -even.value],
        (None, Some(odd)) => [Fr::from_u64(0), odd.value],
        (None, None) => unreachable!("message contribution requires at least one entry"),
    }
}

fn address_major_val_init(columns: usize) -> Vec<Fr> {
    (0..columns)
        .map(|column| field_from_index(710_000 + column))
        .collect()
}

fn cycle_major_message_memory(entries: usize) -> AnalyticalMemory {
    let input = entries * size_of::<BenchCycleMajorEntry>();
    let peak_working = size_of::<[Fr; 2]>() * rayon::current_num_threads().max(1);
    AnalyticalMemory {
        input,
        peak_working,
        budget: input * 4 + peak_working * 64 + 16 * 1024 * 1024,
    }
}

fn address_major_message_memory(entries: usize, rows: usize) -> AnalyticalMemory {
    let input = entries * size_of::<BenchAddressMajorValueEntry>() + 2 * rows * size_of::<Fr>();
    let peak_working = size_of::<[Fr; 2]>() * rayon::current_num_threads().max(1);
    AnalyticalMemory {
        input,
        peak_working,
        budget: input * 4 + peak_working * 64 + 16 * 1024 * 1024,
    }
}

fn cycle_to_address_major_memory(entries: usize) -> AnalyticalMemory {
    let input = entries * size_of::<BenchCycleMajorEntry>();
    let output = entries * size_of::<BenchAddressMajorEntry>();
    let sort_scratch = entries * size_of::<BenchCycleMajorEntry>();
    AnalyticalMemory {
        input,
        peak_working: input + output + sort_scratch,
        budget: (input + output + sort_scratch) * 4 + 16 * 1024 * 1024,
    }
}

fn address_major_fingerprint(entries: &[BenchAddressMajorEntry]) -> Fr {
    entries
        .iter()
        .step_by((entries.len() / 64).max(1))
        .fold(Fr::from_u64(entries.len() as u64), |acc, entry| {
            acc + entry.value + Fr::from_u64((entry.row ^ entry.column) as u64)
        })
}

fn tensor_eq_entries(log_rows: usize) -> usize {
    let out_bits = log_rows / 2;
    let in_bits = log_rows - out_bits;
    (1usize << out_bits) + (1usize << in_bits)
}

fn tensor_eq_outer_entries(log_rows: usize) -> usize {
    1usize << (log_rows / 2)
}

fn run_linear(
    request: &SumcheckLinearProductRequest<'_, Fr>,
) -> Vec<SumcheckLinearProductOutput<Fr>> {
    let mut backend = CpuBackend::default();
    match <CpuBackend as SumcheckBackend<Fr, JoltVmNamespace>>::evaluate_sumcheck_linear_products(
        &mut backend,
        request,
    ) {
        Ok(outputs) => outputs,
        Err(error) => {
            black_box(error.to_string());
            std::process::abort();
        }
    }
}

fn run_prefix_product_sum(
    request: &SumcheckPrefixProductSumRequest<'_, Fr>,
) -> Vec<SumcheckLinearProductOutput<Fr>> {
    let mut backend = CpuBackend::default();
    match <CpuBackend as SumcheckBackend<Fr, JoltVmNamespace>>::evaluate_sumcheck_prefix_product_sums(
        &mut backend,
        request,
    ) {
        Ok(outputs) => outputs,
        Err(error) => {
            black_box(error.to_string());
            std::process::abort();
        }
    }
}

fn run_row(request: &SumcheckRowProductRequest<'_, Fr>) -> Vec<SumcheckLinearProductOutput<Fr>> {
    let mut backend = CpuBackend::default();
    match <CpuBackend as SumcheckBackend<Fr, JoltVmNamespace>>::evaluate_sumcheck_row_products(
        &mut backend,
        request,
    ) {
        Ok(outputs) => outputs,
        Err(error) => {
            black_box(error.to_string());
            std::process::abort();
        }
    }
}

fn run_product_uniskip_rows(
    request: &SumcheckProductUniskipRequest<'_, Fr>,
) -> Vec<SumcheckLinearProductOutput<Fr>> {
    let mut backend = CpuBackend::default();
    match <CpuBackend as SumcheckBackend<Fr, JoltVmNamespace>>::evaluate_sumcheck_product_uniskip_rows(
        &mut backend,
        request,
    ) {
        Ok(outputs) => outputs,
        Err(error) => {
            black_box(error.to_string());
            std::process::abort();
        }
    }
}

fn run_view_evaluations(
    witness: &DenseViewBenchWitness,
    request: &SumcheckEvaluationRequest<Fr, BenchNamespace>,
) -> Vec<SumcheckEvaluationOutput<Fr>> {
    let mut backend = CpuBackend::default();
    match <CpuBackend as SumcheckBackend<Fr, BenchNamespace>>::evaluate_sumcheck_views(
        &mut backend,
        request,
        witness,
    ) {
        Ok(outputs) => outputs,
        Err(error) => {
            black_box(error.to_string());
            std::process::abort();
        }
    }
}

fn run_opening_rlc_materialization(
    witness: &DenseViewBenchWitness,
    request: &OpeningRlcMaterializationRequest<Fr, BenchNamespace>,
) -> OpeningRlcMaterializationResult<Fr> {
    let mut backend = CpuBackend::default();
    match <CpuBackend as OpeningBackend<Fr, BenchNamespace, MockCommitmentScheme<Fr>>>::materialize_opening_rlc(
        &mut backend,
        request,
        witness,
    ) {
        Ok(result) => result,
        Err(error) => {
            black_box(error.to_string());
            std::process::abort();
        }
    }
}

fn warm_rayon() {
    let _: usize = (0..rayon::current_num_threads()).into_par_iter().sum();
}

fn assert_memory_budget(memory: AnalyticalMemory, measured_peak: usize) {
    assert!(
        measured_peak <= memory.budget,
        "measured peak working memory exceeded analytical budget: measured={} expected_peak={} budget={} input={}",
        memory_label(measured_peak),
        memory_label(memory.peak_working),
        memory_label(memory.budget),
        memory_label(memory.input),
    );
}

fn mib(bytes: usize) -> usize {
    bytes / (1024 * 1024)
}

fn memory_label(bytes: usize) -> String {
    if bytes < 1024 * 1024 {
        format!("{}KiB", bytes.div_ceil(1024))
    } else {
        format!("{}MiB", mib(bytes))
    }
}

fn bench_linear_products(c: &mut Criterion) {
    warm_rayon();
    let mut group = c.benchmark_group("cpu_sumcheck/linear_product");
    group.sample_size(10);

    for shape in [
        KernelShape {
            name: "mid",
            log_rows: 16,
            columns: 12,
            sparse_rows: 48,
            terms_per_side: 3,
            queries: 8,
        },
        KernelShape {
            name: "wide",
            log_rows: 18,
            columns: 12,
            sparse_rows: 64,
            terms_per_side: 3,
            queries: 8,
        },
    ] {
        let fixture = KernelFixture::new(shape);
        let request = fixture.linear_request();
        let memory = fixture.analytical_memory(KernelKind::Linear);
        let measured_peak = measured_peak_bytes(|| {
            black_box(run_linear(&request));
        });
        assert_memory_budget(memory, measured_peak);

        group.throughput(Throughput::Elements((shape.rows() * shape.queries) as u64));
        group.bench_with_input(
            BenchmarkId::new(
                shape.name,
                format!(
                    "rows=2^{} cols={} q={} work={} measured={}",
                    shape.log_rows,
                    shape.columns,
                    shape.queries,
                    memory_label(memory.peak_working),
                    memory_label(measured_peak)
                ),
            ),
            &request,
            |bench, request| {
                bench.iter(|| {
                    black_box(run_linear(black_box(request)));
                });
            },
        );
    }

    group.finish();
}

fn bench_linear_boolean_products(c: &mut Criterion) {
    warm_rayon();
    let mut group = c.benchmark_group("cpu_sumcheck/linear_product_boolean");
    group.sample_size(10);

    for shape in [
        KernelShape {
            name: "stage1_mid",
            log_rows: 16,
            columns: 35,
            sparse_rows: 20,
            terms_per_side: 3,
            queries: 4096,
        },
        KernelShape {
            name: "stage1_wide",
            log_rows: 18,
            columns: 35,
            sparse_rows: 20,
            terms_per_side: 3,
            queries: 8192,
        },
    ] {
        let fixture = KernelFixture::new(shape);
        let request = fixture.linear_boolean_request();
        let memory = fixture.analytical_memory(KernelKind::LinearBoolean);
        let measured_peak = measured_peak_bytes(|| {
            black_box(run_linear(&request));
        });
        assert_memory_budget(memory, measured_peak);

        group.throughput(Throughput::Elements(shape.queries as u64));
        group.bench_with_input(
            BenchmarkId::new(
                shape.name,
                format!(
                    "rows=2^{} cols={} q={} work={} measured={}",
                    shape.log_rows,
                    shape.columns,
                    shape.queries,
                    memory_label(memory.peak_working),
                    memory_label(measured_peak)
                ),
            ),
            &request,
            |bench, request| {
                bench.iter(|| {
                    black_box(run_linear(black_box(request)));
                });
            },
        );
    }

    group.finish();
}

fn bench_outer_uniskip_prefix_sums(c: &mut Criterion) {
    warm_rayon();
    let mut group = c.benchmark_group("cpu_sumcheck/outer_uniskip_prefix_sum");
    group.sample_size(10);

    for shape in [
        KernelShape {
            name: "stage1_mid",
            log_rows: 16,
            columns: 35,
            sparse_rows: 19,
            terms_per_side: 3,
            queries: 9,
        },
        KernelShape {
            name: "stage1_wide",
            log_rows: 18,
            columns: 35,
            sparse_rows: 19,
            terms_per_side: 3,
            queries: 9,
        },
    ] {
        let fixture = KernelFixture::new(shape);
        let request = fixture.outer_uniskip_sum_request();
        let memory = fixture.analytical_memory(KernelKind::PrefixProductSum);
        let measured_peak = measured_peak_bytes(|| {
            black_box(run_prefix_product_sum(&request));
        });
        assert_memory_budget(memory, measured_peak);

        group.throughput(Throughput::Elements(
            (shape.rows() * 2 * shape.queries) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new(
                shape.name,
                format!(
                    "rows=2^{} cols={} q={} work={} measured={}",
                    shape.log_rows,
                    shape.columns,
                    shape.queries,
                    memory_label(memory.peak_working),
                    memory_label(measured_peak)
                ),
            ),
            &request,
            |bench, request| {
                bench.iter(|| {
                    black_box(run_prefix_product_sum(black_box(request)));
                });
            },
        );
    }

    group.finish();
}

fn bench_outer_remainder_bound_prefix_sums(c: &mut Criterion) {
    warm_rayon();
    let mut group = c.benchmark_group("cpu_sumcheck/outer_remainder_bound_prefix_sum");
    group.sample_size(10);

    for shape in [
        KernelShape {
            name: "stage1_mid_round1",
            log_rows: 16,
            columns: 35,
            sparse_rows: 19,
            terms_per_side: 3,
            queries: 4,
        },
        KernelShape {
            name: "stage1_wide_round1",
            log_rows: 18,
            columns: 35,
            sparse_rows: 19,
            terms_per_side: 3,
            queries: 4,
        },
    ] {
        let fixture = KernelFixture::new(shape);
        let request = fixture.outer_remainder_bound_prefix_request();
        let memory = fixture.analytical_memory(KernelKind::PrefixProductBound);
        let measured_peak = measured_peak_bytes(|| {
            black_box(run_prefix_product_sum(&request));
        });
        assert_memory_budget(memory, measured_peak);

        let effective_rows = shape.rows() + (shape.rows() / 2) * shape.queries;
        group.throughput(Throughput::Elements(effective_rows as u64));
        group.bench_with_input(
            BenchmarkId::new(
                shape.name,
                format!(
                    "rows=2^{} cols={} q={} work={} measured={}",
                    shape.log_rows,
                    shape.columns,
                    shape.queries,
                    memory_label(memory.peak_working),
                    memory_label(measured_peak)
                ),
            ),
            &request,
            |bench, request| {
                bench.iter(|| {
                    black_box(run_prefix_product_sum(black_box(request)));
                });
            },
        );
    }

    group.finish();
}

fn bench_row_products(c: &mut Criterion) {
    warm_rayon();
    let mut group = c.benchmark_group("cpu_sumcheck/row_product");
    group.sample_size(10);

    for shape in [
        KernelShape {
            name: "mid",
            log_rows: 16,
            columns: 6,
            sparse_rows: 32,
            terms_per_side: 2,
            queries: 4,
        },
        KernelShape {
            name: "wide",
            log_rows: 18,
            columns: 6,
            sparse_rows: 48,
            terms_per_side: 2,
            queries: 4,
        },
    ] {
        let fixture = KernelFixture::new(shape);
        let request = fixture.row_request();
        let memory = fixture.analytical_memory(KernelKind::Row);
        let measured_peak = measured_peak_bytes(|| {
            black_box(run_row(&request));
        });
        assert_memory_budget(memory, measured_peak);

        group.throughput(Throughput::Elements((shape.rows() * shape.queries) as u64));
        group.bench_with_input(
            BenchmarkId::new(
                shape.name,
                format!(
                    "rows=2^{} cols={} q={} work={} measured={}",
                    shape.log_rows,
                    shape.columns,
                    shape.queries,
                    memory_label(memory.peak_working),
                    memory_label(measured_peak)
                ),
            ),
            &request,
            |bench, request| {
                bench.iter(|| {
                    black_box(run_row(black_box(request)));
                });
            },
        );
    }

    group.finish();
}

fn bench_grouped_row_products(c: &mut Criterion) {
    warm_rayon();
    let mut group = c.benchmark_group("cpu_sumcheck/row_product_grouped");
    group.sample_size(10);

    for shape in [
        KernelShape {
            name: "stage2_mid",
            log_rows: 16,
            columns: 6,
            sparse_rows: 32,
            terms_per_side: 2,
            queries: 8,
        },
        KernelShape {
            name: "stage2_wide",
            log_rows: 18,
            columns: 6,
            sparse_rows: 48,
            terms_per_side: 2,
            queries: 8,
        },
    ] {
        let fixture = KernelFixture::new(shape);
        let request = fixture.grouped_row_request();
        let memory = fixture.analytical_memory(KernelKind::RowGrouped);
        let measured_peak = measured_peak_bytes(|| {
            black_box(run_row(&request));
        });
        assert_memory_budget(memory, measured_peak);

        group.throughput(Throughput::Elements((shape.rows() * shape.queries) as u64));
        group.bench_with_input(
            BenchmarkId::new(
                shape.name,
                format!(
                    "rows=2^{} cols={} q={} work={} measured={}",
                    shape.log_rows,
                    shape.columns,
                    shape.queries,
                    memory_label(memory.peak_working),
                    memory_label(measured_peak)
                ),
            ),
            &request,
            |bench, request| {
                bench.iter(|| {
                    black_box(run_row(black_box(request)));
                });
            },
        );
    }

    group.finish();
}

fn bench_spartan_product_uniskip(c: &mut Criterion) {
    warm_rayon();
    let mut group = c.benchmark_group("cpu_sumcheck/spartan_product_uniskip");
    group.sample_size(10);

    for shape in [
        KernelShape {
            name: "stage2_mid",
            log_rows: 16,
            columns: 6,
            sparse_rows: PRODUCT_UNISKIP_ROWS,
            terms_per_side: 1,
            queries: PRODUCT_UNISKIP_EXTENDED_EVALS,
        },
        KernelShape {
            name: "stage2_wide",
            log_rows: 18,
            columns: 6,
            sparse_rows: PRODUCT_UNISKIP_ROWS,
            terms_per_side: 1,
            queries: PRODUCT_UNISKIP_EXTENDED_EVALS,
        },
    ] {
        let fixture = KernelFixture::new_product_uniskip(shape);
        let request = fixture.product_uniskip_request();
        let memory = fixture.analytical_memory(KernelKind::RowGrouped);
        let measured_peak = measured_peak_bytes(|| {
            black_box(run_row(&request));
        });
        assert_memory_budget(memory, measured_peak);

        group.throughput(Throughput::Elements((shape.rows() * shape.queries) as u64));
        group.bench_with_input(
            BenchmarkId::new(
                shape.name,
                format!(
                    "rows=2^{} cols={} q={} work={} measured={}",
                    shape.log_rows,
                    shape.columns,
                    shape.queries,
                    memory_label(memory.peak_working),
                    memory_label(measured_peak)
                ),
            ),
            &request,
            |bench, request| {
                bench.iter(|| {
                    black_box(run_row(black_box(request)));
                });
            },
        );
    }

    group.finish();
}

fn bench_spartan_product_uniskip_raw(c: &mut Criterion) {
    warm_rayon();
    let mut group = c.benchmark_group("cpu_sumcheck/spartan_product_uniskip_raw");
    group.sample_size(10);

    for shape in [
        KernelShape {
            name: "stage2_mid",
            log_rows: 16,
            columns: 6,
            sparse_rows: PRODUCT_UNISKIP_ROWS,
            terms_per_side: 1,
            queries: PRODUCT_UNISKIP_EXTENDED_EVALS,
        },
        KernelShape {
            name: "stage2_wide",
            log_rows: 18,
            columns: 6,
            sparse_rows: PRODUCT_UNISKIP_ROWS,
            terms_per_side: 1,
            queries: PRODUCT_UNISKIP_EXTENDED_EVALS,
        },
    ] {
        let fixture = KernelFixture::new_product_uniskip(shape);
        let request = fixture.raw_product_uniskip_request();
        let memory = fixture.analytical_memory(KernelKind::ProductUniskipRaw);
        let measured_peak = measured_peak_bytes(|| {
            black_box(run_product_uniskip_rows(&request));
        });
        assert_memory_budget(memory, measured_peak);

        group.throughput(Throughput::Elements((shape.rows() * shape.queries) as u64));
        group.bench_with_input(
            BenchmarkId::new(
                shape.name,
                format!(
                    "rows=2^{} cols={} q={} work={} measured={}",
                    shape.log_rows,
                    shape.columns,
                    shape.queries,
                    memory_label(memory.peak_working),
                    memory_label(measured_peak)
                ),
            ),
            &request,
            |bench, request| {
                bench.iter(|| {
                    black_box(run_product_uniskip_rows(black_box(request)));
                });
            },
        );
    }

    group.finish();
}

fn bench_view_evaluations(c: &mut Criterion) {
    warm_rayon();
    let mut group = c.benchmark_group("cpu_sumcheck/view_evaluation");
    group.sample_size(10);

    for shape in [
        ViewEvalShape {
            name: "grouped_mid",
            log_rows: 16,
            views: 16,
            unique_points: 4,
        },
        ViewEvalShape {
            name: "grouped_wide",
            log_rows: 18,
            views: 16,
            unique_points: 4,
        },
    ] {
        let fixture = ViewEvalFixture::new(shape);
        let memory = fixture.analytical_memory();
        let measured_peak = measured_peak_bytes(|| {
            black_box(run_view_evaluations(&fixture.witness, &fixture.request));
        });
        assert_memory_budget(memory, measured_peak);

        group.throughput(Throughput::Elements((shape.rows() * shape.views) as u64));
        group.bench_with_input(
            BenchmarkId::new(
                shape.name,
                format!(
                    "rows=2^{} views={} points={} work={} measured={}",
                    shape.log_rows,
                    shape.views,
                    shape.unique_points,
                    memory_label(memory.peak_working),
                    memory_label(measured_peak)
                ),
            ),
            &fixture,
            |bench, fixture| {
                bench.iter(|| {
                    black_box(run_view_evaluations(
                        black_box(&fixture.witness),
                        black_box(&fixture.request),
                    ));
                });
            },
        );
    }

    group.finish();
}

fn bench_opening_rlc_materialization(c: &mut Criterion) {
    warm_rayon();
    let mut group = c.benchmark_group("cpu_openings/rlc_materialized_fallback");
    group.sample_size(10);

    for shape in [
        RlcMaterializationShape {
            name: "mid",
            log_rows: 16,
            components: 16,
        },
        RlcMaterializationShape {
            name: "wide",
            log_rows: 18,
            components: 16,
        },
    ] {
        let fixture = RlcMaterializationFixture::new(shape);
        let memory = fixture.analytical_memory();
        let measured_peak = measured_peak_bytes(|| {
            black_box(run_opening_rlc_materialization(
                &fixture.witness,
                &fixture.request,
            ));
        });
        assert_memory_budget(memory, measured_peak);

        group.throughput(Throughput::Elements(
            (shape.rows() * shape.components) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new(
                shape.name,
                format!(
                    "rows=2^{} components={} work={} measured={}",
                    shape.log_rows,
                    shape.components,
                    memory_label(memory.peak_working),
                    memory_label(measured_peak)
                ),
            ),
            &fixture,
            |bench, fixture| {
                bench.iter(|| {
                    black_box(run_opening_rlc_materialization(
                        black_box(&fixture.witness),
                        black_box(&fixture.request),
                    ));
                });
            },
        );
    }

    group.finish();
}

fn bench_eq_tables(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_sumcheck/eq_tables");
    for log_vars in [18usize, 20] {
        let point = (0..log_vars)
            .map(|index| field_from_index(80_000 + index))
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(1u64 << log_vars));
        group.bench_with_input(
            BenchmarkId::new("evals", format!("vars={log_vars}")),
            &point,
            |b, point| {
                b.iter(|| black_box(eq::evals(black_box(point), None)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cached", format!("vars={log_vars}")),
            &point,
            |b, point| {
                b.iter(|| black_box(eq::evals_cached(black_box(point), None)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cached_rev", format!("vars={log_vars}")),
            &point,
            |b, point| {
                b.iter(|| black_box(eq::evals_cached_rev(black_box(point), None)));
            },
        );
    }
    group.finish();
}

fn bench_eq_aligned_blocks(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_sumcheck/eq_aligned_blocks");
    let log_vars = 24usize;
    let point = (0..log_vars)
        .map(|index| field_from_index(90_000 + index))
        .collect::<Vec<_>>();

    for block_vars in [12usize, 16] {
        let block_size = 1usize << block_vars;
        let start = 3usize << block_vars;
        group.throughput(Throughput::Elements(block_size as u64));
        group.bench_with_input(
            BenchmarkId::new("aligned", format!("vars={log_vars} block=2^{block_vars}")),
            &(start, block_size),
            |b, &(start, block_size)| {
                b.iter(|| {
                    black_box(eq::evals_for_aligned_block(
                        black_box(&point),
                        black_box(start),
                        black_box(block_size),
                    ))
                });
            },
        );
    }

    let scan_start = 12_288usize;
    let scan_len = 1usize << 18;
    group.throughput(Throughput::Elements(scan_len as u64));
    group.bench_function("max_aligned_scan/range=2^18", |b| {
        b.iter(|| {
            let mut cursor = scan_start;
            let end = scan_start + scan_len;
            let mut total = 0usize;
            while cursor < end {
                let (block_size, values) = eq::evals_for_max_aligned_block(
                    black_box(&point),
                    black_box(cursor),
                    black_box(end - cursor),
                );
                total += values.len();
                cursor += block_size;
            }
            black_box(total)
        });
    });

    group.finish();
}

fn bench_split_eq_windows(c: &mut Criterion) {
    const LOG_VARS: usize = 24;
    const WINDOW_SIZE: usize = 10;
    const ROUNDS: usize = 8;

    let mut group = c.benchmark_group("cpu_sumcheck/split_eq_windows");
    group.throughput(Throughput::Elements(
        (ROUNDS * (1usize << (WINDOW_SIZE - 1))) as u64,
    ));

    let point = (0..LOG_VARS)
        .map(|index| field_from_index(110_000 + index))
        .collect::<Vec<_>>();
    let challenges = (0..ROUNDS)
        .map(|index| field_from_index(111_000 + index))
        .collect::<Vec<_>>();

    group.bench_function(
        format!("active_low_to_high/vars={LOG_VARS} window={WINDOW_SIZE}"),
        |b| {
            b.iter(|| {
                let mut split = split_eq::gruen(black_box(&point), BindingOrder::LowToHigh);
                let mut total = 0usize;
                for &challenge in &challenges {
                    let (e_out, e_in) =
                        split_eq::e_out_in_for_window(black_box(&split), WINDOW_SIZE);
                    total += e_out.len() + e_in.len();
                    total += split_eq::e_active_for_window(black_box(&split), WINDOW_SIZE).len();
                    split.bind(black_box(challenge));
                }
                black_box(total)
            });
        },
    );

    group.finish();
}

fn bench_unipoly_interpolation(c: &mut Criterion) {
    const ITERS: usize = 16_384;
    const TOOM_ITERS: usize = 1_024;
    let mut group = c.benchmark_group("cpu_sumcheck/unipoly_interpolation");
    group.throughput(Throughput::Elements(ITERS as u64));

    let quadratic = [
        field_from_index(100_001),
        field_from_index(100_019),
        field_from_index(100_043),
    ];
    group.bench_function("from_evals_degree2", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..ITERS {
                total += univariate::from_evals(black_box(&quadratic))
                    .coefficients()
                    .len();
            }
            black_box(total)
        });
    });

    let cubic = [
        field_from_index(101_001),
        field_from_index(101_019),
        field_from_index(101_043),
        field_from_index(101_071),
    ];
    group.bench_function("from_evals_degree3", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..ITERS {
                total += univariate::from_evals(black_box(&cubic))
                    .coefficients()
                    .len();
            }
            black_box(total)
        });
    });

    let hinted = [cubic[0], cubic[2], cubic[3]];
    let hint = cubic[0] + cubic[1];
    group.bench_function("from_evals_and_hint_degree3", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..ITERS {
                total += univariate::from_evals_and_hint(black_box(hint), black_box(&hinted))
                    .coefficients()
                    .len();
            }
            black_box(total)
        });
    });

    let toom = [
        field_from_index(102_001),
        field_from_index(102_019),
        field_from_index(102_043),
        field_from_index(102_071),
        field_from_index(102_101),
    ];
    group.bench_function("from_evals_toom_degree4", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..TOOM_ITERS {
                total += univariate::from_evals_toom(black_box(&toom))
                    .coefficients()
                    .len();
            }
            black_box(total)
        });
    });

    group.finish();
}

fn bench_compressed_unipoly(c: &mut Criterion) {
    const ITERS: usize = 16_384;

    let mut group = c.benchmark_group("cpu_sumcheck/compressed_unipoly");
    group.throughput(Throughput::Elements(ITERS as u64));

    let poly = UnivariatePoly::new(
        (0..9)
            .map(|index| field_from_index(120_000 + index * 17))
            .collect::<Vec<_>>(),
    );
    let hint = poly.evaluate(Fr::from_u64(0)) + poly.evaluate(Fr::from_u64(1));
    let compressed = univariate::compress(&poly);
    let points = (0..8)
        .map(|index| field_from_index(121_000 + index * 19))
        .collect::<Vec<_>>();

    group.bench_function("compress_degree8", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..ITERS {
                total += univariate::compress(black_box(&poly))
                    .coeffs_except_linear_term()
                    .len();
            }
            black_box(total)
        });
    });

    group.bench_function("decompress_degree8", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..ITERS {
                total += univariate::decompress(black_box(&compressed), black_box(hint))
                    .coefficients()
                    .len();
            }
            black_box(total)
        });
    });

    group.bench_function("eval_from_hint_degree8", |b| {
        b.iter(|| {
            let mut total = Fr::from_u64(0);
            for _ in 0..ITERS {
                for &point in &points {
                    total += univariate::eval_from_hint(
                        black_box(&compressed),
                        black_box(hint),
                        black_box(point),
                    );
                }
            }
            black_box(total)
        });
    });

    group.finish();
}

fn require_centered_domain<T>(
    result: Result<T, jolt_poly::lagrange::CenteredIntegerDomainError>,
) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            black_box(error.to_string());
            std::process::abort();
        }
    }
}

fn bench_lagrange_many(c: &mut Criterion) {
    const N: usize = 10;
    const ITERS: usize = 4_096;

    let mut group = c.benchmark_group("cpu_sumcheck/lagrange_many");
    let values = core::array::from_fn(|index| field_from_index(130_000 + index * 17));
    let point = field_from_index(131_000);
    let other = field_from_index(131_019);
    let points = (0..16)
        .map(|index| field_from_index(132_000 + index * 19))
        .collect::<Vec<_>>();

    group.throughput(Throughput::Elements(ITERS as u64));
    group.bench_function("centered_evals_n10", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..ITERS {
                total +=
                    require_centered_domain(lagrange::centered_evals::<Fr, N>(black_box(point)))
                        .len();
            }
            black_box(total)
        });
    });

    group.bench_function("centered_kernel_n10", |b| {
        b.iter(|| {
            let mut total = Fr::from_u64(0);
            for _ in 0..ITERS {
                total += require_centered_domain(lagrange::centered_kernel(
                    N,
                    black_box(point),
                    black_box(other),
                ));
            }
            black_box(total)
        });
    });

    group.throughput(Throughput::Elements((ITERS * points.len()) as u64));
    group.bench_function("evaluate_many_n10_points16", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..ITERS {
                total += require_centered_domain(lagrange::centered_evaluate_many::<Fr, N>(
                    black_box(&values),
                    black_box(&points),
                ))
                .len();
            }
            black_box(total)
        });
    });

    group.throughput(Throughput::Elements(ITERS as u64));
    group.bench_function("interpolate_coeffs_n10", |b| {
        b.iter(|| {
            let mut total = Fr::from_u64(0);
            for _ in 0..ITERS {
                total += require_centered_domain(lagrange::centered_interpolate_coeffs::<Fr, N>(
                    black_box(&values),
                ))[0];
            }
            black_box(total)
        });
    });

    group.finish();
}

fn bench_streaming_schedule(c: &mut Criterion) {
    const ITERS: usize = 16_384;
    const SHAPES: [(usize, usize); 5] = [(20, 2), (40, 2), (64, 2), (64, 3), (96, 4)];

    let mut group = c.benchmark_group("cpu_sumcheck/streaming_schedule");
    group.throughput(Throughput::Elements((ITERS * SHAPES.len()) as u64));
    group.bench_function("half_split_and_linear_only", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..ITERS {
                for &(rounds, degree) in &SHAPES {
                    let schedule =
                        schedule::HalfSplitSchedule::new(black_box(rounds), black_box(degree));
                    total ^= schedule_fingerprint(&schedule);
                }
                let linear = schedule::LinearOnlySchedule::new(black_box(64));
                total ^= schedule_fingerprint(&linear);
            }
            black_box(total)
        });
    });
    group.finish();
}

fn bench_ra_delayed_materialization(c: &mut Criterion) {
    const LOG_LEN: usize = 19;
    const K: usize = 256;

    warm_rayon();
    let lookup_indices = Arc::new(ra_lookup_indices(LOG_LEN, K));
    let eq_evals = ra_eq_evals(K);
    let challenges = [
        field_from_index(920_001),
        field_from_index(920_003),
        field_from_index(920_009),
        field_from_index(920_027),
    ];
    let memory = ra_delayed_materialization_memory(LOG_LEN, K);
    let measured_peak = measured_peak_bytes(|| {
        let mut poly = ra::RaPolynomial::<u8, Fr>::new(lookup_indices.clone(), eq_evals.clone());
        for &challenge in &challenges {
            poly.bind_parallel(challenge, BindingOrder::LowToHigh);
        }
        black_box(ra_fingerprint(&poly));
    });
    assert_memory_budget(memory, measured_peak);

    let mut group = c.benchmark_group("cpu_sumcheck/ra_delayed_materialization");
    group.sample_size(10);
    group.throughput(Throughput::Elements(lookup_indices.len() as u64));
    group.bench_function(
        format!(
            "three_specialized_rounds_plus_dense_tail/t={} k={} work={} measured={}",
            lookup_indices.len(),
            K,
            memory_label(memory.peak_working),
            memory_label(measured_peak)
        ),
        |b| {
            b.iter(|| {
                let mut poly =
                    ra::RaPolynomial::<u8, Fr>::new(lookup_indices.clone(), eq_evals.clone());
                for &challenge in &challenges {
                    poly.bind_parallel(black_box(challenge), BindingOrder::LowToHigh);
                }
                black_box(ra_fingerprint(&poly));
            });
        },
    );
    group.finish();
}

fn bench_ra_pushforward(c: &mut Criterion) {
    const LOG_LEN: usize = 16;

    warm_rayon();
    let layout = shared_ra_layout();
    let indices = shared_ra_indices(LOG_LEN, layout);
    let r_cycle = (0..LOG_LEN)
        .map(|index| field_from_index(950_000 + index))
        .collect::<Vec<_>>();
    let memory = ra_pushforward_memory(LOG_LEN, layout);
    let measured_peak = measured_peak_bytes(|| {
        black_box(ra_pushforward_fingerprint(&ra::pushforward_indices(
            &indices, layout, &r_cycle,
        )));
    });
    assert_memory_budget(memory, measured_peak);

    let mut group = c.benchmark_group("cpu_sumcheck/ra_pushforward");
    group.sample_size(10);
    group.throughput(Throughput::Elements(
        (indices.len() * layout.num_polys()) as u64,
    ));
    group.bench_function(
        format!(
            "split_eq_touched_accumulators/t={} polys={} k={} work={} measured={}",
            indices.len(),
            layout.num_polys(),
            layout.k_chunk,
            memory_label(memory.peak_working),
            memory_label(measured_peak)
        ),
        |b| {
            b.iter(|| {
                black_box(ra_pushforward_fingerprint(&ra::pushforward_indices(
                    black_box(&indices),
                    layout,
                    black_box(&r_cycle),
                )));
            });
        },
    );
    group.finish();
}

fn bench_shared_ra_delayed_materialization(c: &mut Criterion) {
    const LOG_LEN: usize = 16;

    warm_rayon();
    let layout = shared_ra_layout();
    let indices = shared_ra_indices(LOG_LEN, layout);
    let tables = shared_ra_tables(layout);
    let challenges = [
        field_from_index(940_001),
        field_from_index(940_003),
        field_from_index(940_009),
        field_from_index(940_027),
    ];
    let memory = shared_ra_delayed_materialization_memory(LOG_LEN, layout);
    let measured_peak = measured_peak_bytes(|| {
        let mut polys = ra::SharedRaPolynomials::<Fr>::new(tables.clone(), indices.clone(), layout);
        for &challenge in &challenges {
            polys.bind_in_place(challenge, BindingOrder::LowToHigh);
        }
        black_box(shared_ra_fingerprint(&polys));
    });
    assert_memory_budget(memory, measured_peak);

    let mut group = c.benchmark_group("cpu_sumcheck/shared_ra_delayed_materialization");
    group.sample_size(10);
    group.throughput(Throughput::Elements(
        (indices.len() * layout.num_polys()) as u64,
    ));
    group.bench_function(
        format!(
            "all_families_three_specialized_rounds/t={} polys={} k={} work={} measured={}",
            indices.len(),
            layout.num_polys(),
            layout.k_chunk,
            memory_label(memory.peak_working),
            memory_label(measured_peak)
        ),
        |b| {
            b.iter(|| {
                let mut polys =
                    ra::SharedRaPolynomials::<Fr>::new(tables.clone(), indices.clone(), layout);
                for &challenge in &challenges {
                    polys.bind_in_place(black_box(challenge), BindingOrder::LowToHigh);
                }
                black_box(shared_ra_fingerprint(&polys));
            });
        },
    );
    group.finish();
}

fn bench_read_write_one_hot_coeff_lookup(c: &mut Criterion) {
    const ITERS: usize = 512;
    const ROUNDS: usize = 3;

    let initial = [
        Fr::from_u64(0),
        Fr::from_u64(1),
        field_from_index(210_001),
        field_from_index(210_019),
    ];
    let challenges = [
        field_from_index(211_001),
        field_from_index(211_019),
        field_from_index(211_043),
    ];
    let sample_indices = [0u16, 1, 2, 3, 17, 251, 4095, 16383, 32767, 65535];

    let mut group = c.benchmark_group("cpu_sumcheck/read_write_one_hot_coeff_lookup");
    group.throughput(Throughput::Elements(
        (ITERS * (1usize << 16) * ROUNDS) as u64,
    ));
    group.bench_function("bind_to_saturation", |b| {
        b.iter(|| {
            let mut total = Fr::from_u64(0);
            for _ in 0..ITERS {
                let mut table = read_write_matrix::OneHotCoeffTable::new(initial.to_vec());
                for &challenge in &challenges {
                    table.bind(black_box(challenge));
                }
                for &index in &sample_indices {
                    total += table[read_write_matrix::OneHotCoeffIndex(index)];
                }
            }
            black_box(total)
        });
    });
    group.finish();
}

fn bench_read_write_cycle_major_bind(c: &mut Criterion) {
    const ROW_PAIRS: usize = 2;
    const TERMS_PER_ROW: usize = 40_000;

    warm_rayon();
    let challenge = field_from_index(430_001);
    let entries = cycle_major_bind_entries(ROW_PAIRS, TERMS_PER_ROW);
    let bound_len = cycle_major_bind_bound_len(ROW_PAIRS, TERMS_PER_ROW);
    let memory = cycle_major_bind_memory(entries.len(), bound_len, ROW_PAIRS);
    let measured_peak = measured_peak_bytes(|| {
        let mut matrix = read_write_matrix::ReadWriteMatrixCycleMajor::new(entries.clone());
        matrix.bind(challenge);
        black_box(cycle_major_bind_fingerprint(&matrix.entries));
    });
    assert_memory_budget(memory, measured_peak);

    let mut group = c.benchmark_group("cpu_sumcheck/read_write_cycle_major_bind");
    group.sample_size(10);
    group.throughput(Throughput::Elements(entries.len() as u64));
    group.bench_function(
        format!(
            "sparse_row_merge_parallel/input={} output={} work={} measured={}",
            entries.len(),
            bound_len,
            memory_label(memory.peak_working),
            memory_label(measured_peak)
        ),
        |b| {
            b.iter(|| {
                let mut matrix =
                    read_write_matrix::ReadWriteMatrixCycleMajor::new(black_box(entries.clone()));
                matrix.bind(black_box(challenge));
                black_box(cycle_major_bind_fingerprint(&matrix.entries));
            });
        },
    );
    group.finish();
}

fn bench_read_write_cycle_major_message(c: &mut Criterion) {
    const ROW_PAIRS: usize = 1;
    const TERMS_PER_ROW: usize = 40_000;

    warm_rayon();
    let entries = cycle_major_bind_entries(ROW_PAIRS, TERMS_PER_ROW);
    let matrix = read_write_matrix::ReadWriteMatrixCycleMajor::new(entries);
    let (even_row, odd_row) = cycle_major_message_rows(&matrix.entries);
    let inc_evals = [field_from_index(440_001), field_from_index(440_003)];
    let gamma = field_from_index(440_009);
    let memory = cycle_major_message_memory(matrix.entries.len());
    let measured_peak = measured_peak_bytes(|| {
        black_box(cycle_major_message_fingerprint(
            matrix.prover_message_contribution(even_row, odd_row, inc_evals, gamma),
        ));
    });
    assert_memory_budget(memory, measured_peak);

    let mut group = c.benchmark_group("cpu_sumcheck/read_write_cycle_major_message");
    group.sample_size(10);
    group.throughput(Throughput::Elements(matrix.entries.len() as u64));
    group.bench_function(
        format!(
            "sparse_row_message/input={} work={} measured={}",
            matrix.entries.len(),
            memory_label(memory.peak_working),
            memory_label(measured_peak)
        ),
        |b| {
            b.iter(|| {
                black_box(cycle_major_message_fingerprint(
                    matrix.prover_message_contribution(
                        black_box(even_row),
                        black_box(odd_row),
                        black_box(inc_evals),
                        black_box(gamma),
                    ),
                ));
            });
        },
    );
    group.finish();
}

fn bench_read_write_address_major_bind(c: &mut Criterion) {
    const COLUMN_PAIRS: usize = 2;
    const TERMS_PER_COL: usize = 40_000;

    warm_rayon();
    let challenge = field_from_index(650_001);
    let entries = address_major_bind_entries(COLUMN_PAIRS, TERMS_PER_COL);
    let val_init = address_major_val_init(COLUMN_PAIRS * 2);
    let bound_len = address_major_bind_bound_len(COLUMN_PAIRS, TERMS_PER_COL);
    let memory = address_major_bind_memory(entries.len(), bound_len, COLUMN_PAIRS);
    let measured_peak = measured_peak_bytes(|| {
        let mut matrix = read_write_matrix::ReadWriteMatrixAddressMajor::new_with_val_init(
            entries.clone(),
            val_init.clone(),
        );
        matrix.bind(challenge);
        black_box(address_major_bind_fingerprint(&matrix.entries));
    });
    assert_memory_budget(memory, measured_peak);

    let mut group = c.benchmark_group("cpu_sumcheck/read_write_address_major_bind");
    group.sample_size(10);
    group.throughput(Throughput::Elements(entries.len() as u64));
    group.bench_function(
        format!(
            "sparse_col_merge_parallel/input={} output={} work={} measured={}",
            entries.len(),
            bound_len,
            memory_label(memory.peak_working),
            memory_label(measured_peak)
        ),
        |b| {
            b.iter(|| {
                let mut matrix = read_write_matrix::ReadWriteMatrixAddressMajor::new_with_val_init(
                    black_box(entries.clone()),
                    black_box(val_init.clone()),
                );
                matrix.bind(black_box(challenge));
                black_box(address_major_bind_fingerprint(&matrix.entries));
            });
        },
    );
    group.finish();
}

fn bench_read_write_address_major_message(c: &mut Criterion) {
    const COLUMN_PAIRS: usize = 1;
    const TERMS_PER_COL: usize = 40_000;

    warm_rayon();
    let entries = address_major_bind_entries(COLUMN_PAIRS, TERMS_PER_COL);
    let (even_col, odd_col) = address_major_message_cols(&entries);
    let rows = entries
        .iter()
        .map(|entry| entry.row)
        .max()
        .map_or(0, |row| row + 1);
    let inc = (0..rows)
        .map(|row| field_from_index(660_000 + row))
        .collect::<Vec<_>>();
    let eq = (0..rows)
        .map(|row| field_from_index(670_000 + row))
        .collect::<Vec<_>>();
    let even_checkpoint = field_from_index(680_001);
    let odd_checkpoint = field_from_index(680_003);
    let gamma = field_from_index(680_009);
    let memory = address_major_message_memory(entries.len(), rows);
    let measured_peak = measured_peak_bytes(|| {
        black_box(
            address_major_message_fingerprint(read_write_matrix::ReadWriteMatrixAddressMajor::<
                Fr,
                BenchAddressMajorValueEntry,
            >::prover_message_contribution(
                even_col,
                odd_col,
                even_checkpoint,
                odd_checkpoint,
                &inc,
                &eq,
                gamma,
            )),
        );
    });
    assert_memory_budget(memory, measured_peak);

    let mut group = c.benchmark_group("cpu_sumcheck/read_write_address_major_message");
    group.sample_size(10);
    group.throughput(Throughput::Elements(entries.len() as u64));
    group.bench_function(
        format!(
            "sparse_col_message/input={} rows={} work={} measured={}",
            entries.len(),
            rows,
            memory_label(memory.peak_working),
            memory_label(measured_peak)
        ),
        |b| {
            b.iter(|| {
                black_box(
                    address_major_message_fingerprint(
                        read_write_matrix::ReadWriteMatrixAddressMajor::<
                            Fr,
                            BenchAddressMajorValueEntry,
                        >::prover_message_contribution(
                            black_box(even_col),
                            black_box(odd_col),
                            black_box(even_checkpoint),
                            black_box(odd_checkpoint),
                            black_box(&inc),
                            black_box(&eq),
                            black_box(gamma),
                        ),
                    ),
                );
            });
        },
    );
    group.finish();
}

fn bench_read_write_cycle_to_address_major(c: &mut Criterion) {
    const ROW_PAIRS: usize = 128;
    const TERMS_PER_ROW: usize = 512;

    warm_rayon();
    let entries = cycle_major_bind_entries(ROW_PAIRS, TERMS_PER_ROW);
    let memory = cycle_to_address_major_memory(entries.len());
    let measured_peak = measured_peak_bytes(|| {
        let cycle_major = read_write_matrix::ReadWriteMatrixCycleMajor::new(entries.clone());
        let address_major: read_write_matrix::ReadWriteMatrixAddressMajor<
            Fr,
            BenchAddressMajorEntry,
        > = cycle_major.into();
        black_box(address_major_fingerprint(&address_major.entries));
    });
    assert_memory_budget(memory, measured_peak);

    let mut group = c.benchmark_group("cpu_sumcheck/read_write_cycle_to_address_major");
    group.sample_size(10);
    group.throughput(Throughput::Elements(entries.len() as u64));
    group.bench_function(
        format!(
            "parallel_sort_map/input={} work={} measured={}",
            entries.len(),
            memory_label(memory.peak_working),
            memory_label(measured_peak)
        ),
        |b| {
            b.iter(|| {
                let cycle_major =
                    read_write_matrix::ReadWriteMatrixCycleMajor::new(black_box(entries.clone()));
                let address_major: read_write_matrix::ReadWriteMatrixAddressMajor<
                    Fr,
                    BenchAddressMajorEntry,
                > = cycle_major.into();
                black_box(address_major_fingerprint(&address_major.entries));
            });
        },
    );
    group.finish();
}

fn schedule_fingerprint<S: schedule::StreamingSchedule>(schedule: &S) -> usize {
    let mut total = schedule.switch_over_point() ^ schedule.num_rounds();
    for round in 0..schedule.num_rounds() {
        total = total.wrapping_mul(131).wrapping_add(
            usize::from(schedule.is_window_start(round)) + schedule.num_unbound_vars(round),
        );
    }
    total
}

criterion_group!(
    benches,
    bench_linear_products,
    bench_linear_boolean_products,
    bench_outer_uniskip_prefix_sums,
    bench_outer_remainder_bound_prefix_sums,
    bench_row_products,
    bench_grouped_row_products,
    bench_spartan_product_uniskip,
    bench_spartan_product_uniskip_raw,
    bench_view_evaluations,
    bench_opening_rlc_materialization,
    bench_eq_tables,
    bench_eq_aligned_blocks,
    bench_split_eq_windows,
    bench_unipoly_interpolation,
    bench_compressed_unipoly,
    bench_lagrange_many,
    bench_streaming_schedule,
    bench_ra_delayed_materialization,
    bench_ra_pushforward,
    bench_shared_ra_delayed_materialization,
    bench_read_write_one_hot_coeff_lookup,
    bench_read_write_cycle_major_bind,
    bench_read_write_cycle_major_message,
    bench_read_write_address_major_bind,
    bench_read_write_address_major_message,
    bench_read_write_cycle_to_address_major,
);
criterion_main!(benches);
