#![expect(
    unused_results,
    clippy::expect_used,
    reason = "benchmarks should fail loudly if a setup or proof path is malformed"
)]

//! Microbenchmarks for the Jolt Akita adapter paths.
//!
//! These cases separate commitment time from opening-proof time. The sparse
//! logical case preserves Jolt's row-major `k=256` one-hot representation, so
//! `AkitaScheme` should route to Akita's native `D64OneHot` backend. The sparse materialized case
//! uses the same one-hot data expanded into a dense evaluation table, forcing
//! the slower dense path and making the expected sparse speedup visible. The
//! `akita_prover` groups use the same Akita-order data without Jolt commitment
//! wrappers or transcript bridging; those numbers bound the overhead introduced
//! by the adapter. The Dory groups run the same logical dense,
//! sparse one-hot, and sparse materialized inputs over BN254 Fr as a familiar
//! PCS baseline, not as a field-for-field security comparison.
//!
//! The default dimensions mirror Akita's own `akita_e2e` Criterion bench:
//! `nv=15,20,25` for dense and one-hot paths, with long measurement windows for
//! `nv >= 20`. For quick local smoke runs, set
//! `JOLT_AKITA_BENCH_NUM_VARS=15` or another comma-separated list.
//! Batch-path benches keep the same physical data volume by using four logical
//! polynomials with two fewer variables: e.g. the `nv=20` batch case opens four
//! `nv=18` polynomials either through Akita's native grouped opening or through
//! one prefix-packed `nv=20` polynomial.

#![expect(
    clippy::unwrap_used,
    reason = "benchmarks and tests unwrap successful PCS operations"
)]

use std::hint::black_box;
use std::time::Duration;

use akita_config::{
    proof_optimized::fp128::{D64Dense as AkitaConfig, D64OneHot as AkitaOneHotConfig},
    CommitmentConfig,
};
use akita_pcs::{AkitaCommitmentScheme, ComputeBackendSetup, CpuBackend};
use akita_prover::{
    AkitaProverSetup as BackendProverSetup, CpuPreparedSetup, DensePoly, OneHotPoly,
    ProverOpeningData,
};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaCommitmentHint, BasisMode, Commitment, OpeningClaims, PointVariableSelection,
    PolynomialGroupClaims,
};
use criterion::{criterion_group, BatchSize, BenchmarkGroup, BenchmarkId, Criterion};
use jolt_akita::{
    jolt_to_akita_evals, reverse_point, AkitaField, AkitaNativeBatching, AkitaProverHint,
    AkitaScheme, AkitaSetupParams, AKITA_ONE_HOT_K256,
};
use jolt_dory::{DoryCommitment, DoryHint, DoryScheme};
use jolt_field::{Field, Fr, FromPrimitiveInt};
use jolt_openings::{
    prove_packed_openings, BatchOpeningScheme, CommitmentScheme, EvaluationClaim,
    PackedOpeningProof, PackedProverGroup, PackedProverObject, PrefixPackedStatement,
    PrefixPacking, VerifierOpeningClaim,
};
use jolt_poly::{MultilinearPoly, OneHotPolynomial, Polynomial};
use jolt_transcript::{Blake2bTranscript, Transcript};

const LAYOUT_DIGEST: [u8; 32] = [0xA5; 32];
const NUM_POLYS: usize = 1;
const BATCH_POLYS: usize = 4;
const BATCH_PREFIX_BITS: usize = 2;
const DEFAULT_NUM_VARS_CASES: [usize; 3] = [15, 20, 25];
const DEFAULT_TRACE_NUM_VARS: usize = 20;

type BackendScheme = AkitaCommitmentScheme<AkitaConfig>;
type OneHotBackendScheme = AkitaCommitmentScheme<AkitaOneHotConfig>;
type BackendCommitment = Commitment<AkitaField>;
type BackendDensePoly = DensePoly<AkitaField>;
type BackendHint = AkitaCommitmentHint<AkitaField>;
type BackendSetup = BackendProverSetup<AkitaField>;
type BackendPreparedSetup = CpuPreparedSetup<AkitaField>;
type BackendOneHotPoly = OneHotPoly<AkitaField, u8>;

const AKITA_D: usize = <AkitaConfig as CommitmentConfig>::D;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BatchId {
    Poly0,
    Poly1,
    Poly2,
    Poly3,
}

impl BatchId {
    const ALL: [Self; BATCH_POLYS] = [Self::Poly0, Self::Poly1, Self::Poly2, Self::Poly3];
}

type AkitaPackedStatement = PrefixPackedStatement<AkitaField, BatchId, jolt_akita::AkitaCommitment>;
type AkitaPackedProof = PackedOpeningProof<AkitaField, <AkitaScheme as CommitmentScheme>::Proof>;

#[derive(Clone, Copy)]
enum DataPath {
    DenseData,
    SparseDataSparsePath,
    SparseDataDensePath,
}

impl DataPath {
    const ALL: [Self; 3] = [
        Self::DenseData,
        Self::SparseDataSparsePath,
        Self::SparseDataDensePath,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::DenseData => "dense_data",
            Self::SparseDataSparsePath => "sparse_data_sparse_path",
            Self::SparseDataDensePath => "sparse_data_dense_path",
        }
    }
}

struct AkitaProverBenchSetup {
    dense_prover: BackendSetup,
    dense_prepared: BackendPreparedSetup,
    one_hot_prover: BackendSetup,
    one_hot_prepared: BackendPreparedSetup,
}

struct AkitaCase {
    point: Vec<AkitaField>,
    dense_poly: Polynomial<AkitaField>,
    dense_eval: AkitaField,
    sparse_one_hot: OneHotPolynomial,
    sparse_dense_poly: Polynomial<AkitaField>,
    sparse_eval: AkitaField,
    setup: <AkitaScheme as CommitmentScheme>::ProverSetup,
    akita_prover_setup: AkitaProverBenchSetup,
    backend_dense_poly: BackendDensePoly,
    backend_sparse_one_hot_poly: BackendOneHotPoly,
    backend_sparse_dense_poly: BackendDensePoly,
}

struct DoryCase {
    point: Vec<Fr>,
    dense_poly: Polynomial<Fr>,
    dense_eval: Fr,
    sparse_one_hot: OneHotPolynomial,
    sparse_dense_poly: Polynomial<Fr>,
    sparse_eval: Fr,
    setup: <DoryScheme as CommitmentScheme>::ProverSetup,
}

struct AkitaBatchCase {
    logical_point: Vec<AkitaField>,
    polynomials: Vec<Polynomial<AkitaField>>,
    evaluations: Vec<AkitaField>,
    native_setup: <AkitaScheme as CommitmentScheme>::ProverSetup,
    packed_pcs_setup: <AkitaScheme as CommitmentScheme>::ProverSetup,
    packing: PrefixPacking<BatchId>,
    packed_polynomial: Polynomial<AkitaField>,
    packed_claims: Vec<(BatchId, EvaluationClaim<AkitaField>)>,
}

fn configure_group(
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    num_vars: usize,
) {
    group.sample_size(10);
    if num_vars >= 20 {
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(30));
    } else {
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(5));
    }
}

fn num_vars_cases() -> Vec<usize> {
    let Some(raw) = std::env::var_os("JOLT_AKITA_BENCH_NUM_VARS") else {
        return DEFAULT_NUM_VARS_CASES.to_vec();
    };
    let raw = raw
        .into_string()
        .expect("JOLT_AKITA_BENCH_NUM_VARS must be valid UTF-8");
    let cases = raw
        .split(',')
        .map(str::trim)
        .filter(|case| !case.is_empty())
        .map(|case| {
            case.parse::<usize>()
                .expect("JOLT_AKITA_BENCH_NUM_VARS entries must be integers")
        })
        .collect::<Vec<_>>();
    assert!(
        !cases.is_empty(),
        "JOLT_AKITA_BENCH_NUM_VARS must contain at least one dimension"
    );
    cases
}

fn batch_logical_num_vars_cases() -> Vec<usize> {
    debug_assert_eq!(BATCH_POLYS, 1usize << BATCH_PREFIX_BITS);
    num_vars_cases()
        .into_iter()
        .map(|num_vars| {
            num_vars
                .checked_sub(BATCH_PREFIX_BITS)
                .expect("batch physical dimension must include packing prefix bits")
        })
        .collect()
}

fn trace_num_vars() -> usize {
    std::env::var("JOLT_AKITA_TRACE_NUM_VARS")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(DEFAULT_TRACE_NUM_VARS)
}

fn criterion_filter_matches(group_name: &str) -> bool {
    let filters = std::env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with("--"))
        .collect::<Vec<_>>();
    if filters.is_empty() {
        return true;
    }

    filters.iter().any(|filter| {
        filter
            .split('|')
            .any(|part| part.is_empty() || group_name.contains(part) || part.contains(group_name))
    })
}

fn field<F: FromPrimitiveInt>(value: u64) -> F {
    F::from_u64(value)
}

fn deterministic_dense_poly<F: Field>(num_vars: usize) -> Polynomial<F> {
    deterministic_dense_poly_with_offset(num_vars, 0)
}

fn deterministic_dense_poly_with_offset<F: Field>(num_vars: usize, offset: u64) -> Polynomial<F> {
    let len = 1usize << num_vars;
    let evals = (0..len)
        .map(|i| field::<F>(((i as u64 * 17 + offset * 19 + 5) % 31) + 1))
        .collect();
    Polynomial::new(evals)
}

fn deterministic_point<F: Field>(num_vars: usize) -> Vec<F> {
    (0..num_vars)
        .map(|i| field::<F>(((i as u64 * 7 + 11) % 97) + 2))
        .collect()
}

fn sparse_one_hot(num_vars: usize) -> OneHotPolynomial {
    let rows = (1usize << num_vars) / AKITA_ONE_HOT_K256;
    let indices = (0..rows)
        .map(|row| Some(((row * 17 + 3) % AKITA_ONE_HOT_K256) as u8))
        .collect();
    OneHotPolynomial::new(AKITA_ONE_HOT_K256, indices)
}

fn materialize_sparse<F: Field>(poly: &OneHotPolynomial) -> Polynomial<F> {
    let mut evals = vec![F::zero(); 1usize << poly.num_vars()];
    <OneHotPolynomial as MultilinearPoly<F>>::for_each_one(poly, &mut |index| {
        evals[index] = F::one();
    });
    Polynomial::new(evals)
}

fn materialize_packed(
    polynomials: &[(BatchId, Polynomial<AkitaField>)],
) -> (PrefixPacking<BatchId>, Polynomial<AkitaField>) {
    let packing = PrefixPacking::new(
        polynomials
            .iter()
            .map(|(id, polynomial)| (*id, polynomial.num_vars())),
    )
    .expect("valid prefix packing");
    let packed_len = 1usize << packing.packed_num_vars;
    let mut packed_evaluations = vec![AkitaField::zero(); packed_len];

    for (id, polynomial) in polynomials {
        let slot = &packing[id];
        let offset = slot.prefix_index() << slot.num_vars;
        for (local_index, evaluation) in polynomial.evals().iter().copied().enumerate() {
            packed_evaluations[offset + local_index] = evaluation;
        }
    }

    (packing, Polynomial::new(packed_evaluations))
}

fn make_backend_one_hot_poly(poly: &OneHotPolynomial) -> BackendOneHotPoly {
    BackendOneHotPoly::new(AKITA_ONE_HOT_K256, AKITA_D, poly.indices().to_vec())
        .expect("valid one-hot backend polynomial")
}

fn make_backend_dense_poly(poly: &Polynomial<AkitaField>) -> BackendDensePoly {
    let evals = jolt_to_akita_evals(poly.num_vars(), poly.evals()).expect("valid dimensions");
    BackendDensePoly::from_field_evals(poly.num_vars(), AKITA_D, &evals)
        .expect("valid dense backend polynomial")
}

fn akita_case(num_vars: usize) -> AkitaCase {
    let dense_poly = deterministic_dense_poly(num_vars);
    let sparse_one_hot = sparse_one_hot(num_vars);
    let sparse_dense_poly = materialize_sparse(&sparse_one_hot);
    let point = deterministic_point(num_vars);
    let dense_eval = dense_poly.evaluate(&point);
    let sparse_eval =
        <OneHotPolynomial as MultilinearPoly<AkitaField>>::evaluate(&sparse_one_hot, &point);
    let (setup, _) =
        AkitaScheme::setup(AkitaSetupParams::new(num_vars, NUM_POLYS, LAYOUT_DIGEST)).unwrap();
    let backend_prover = BackendScheme::setup_prover(num_vars, NUM_POLYS)
        .expect("Akita backend setup should succeed");
    let backend_prepared = CpuBackend
        .prepare_setup(&backend_prover)
        .expect("Akita backend setup preparation should succeed");
    let one_hot_backend_prover = OneHotBackendScheme::setup_prover(num_vars, NUM_POLYS)
        .expect("Akita one-hot backend setup should succeed");
    let one_hot_backend_prepared = CpuBackend
        .prepare_setup(&one_hot_backend_prover)
        .expect("Akita one-hot backend setup preparation should succeed");
    let backend_dense_poly = make_backend_dense_poly(&dense_poly);
    let backend_sparse_one_hot_poly = make_backend_one_hot_poly(&sparse_one_hot);
    let backend_sparse_dense_poly = make_backend_dense_poly(&sparse_dense_poly);

    AkitaCase {
        point,
        dense_poly,
        dense_eval,
        sparse_one_hot,
        sparse_dense_poly,
        sparse_eval,
        setup,
        akita_prover_setup: AkitaProverBenchSetup {
            dense_prover: backend_prover,
            dense_prepared: backend_prepared,
            one_hot_prover: one_hot_backend_prover,
            one_hot_prepared: one_hot_backend_prepared,
        },
        backend_dense_poly,
        backend_sparse_one_hot_poly,
        backend_sparse_dense_poly,
    }
}

fn dory_case(num_vars: usize) -> DoryCase {
    let dense_poly = deterministic_dense_poly(num_vars);
    let sparse_one_hot = sparse_one_hot(num_vars);
    let sparse_dense_poly = materialize_sparse(&sparse_one_hot);
    let point = deterministic_point(num_vars);
    let dense_eval = dense_poly.evaluate(&point);
    let sparse_eval = <OneHotPolynomial as MultilinearPoly<Fr>>::evaluate(&sparse_one_hot, &point);

    DoryCase {
        point,
        dense_poly,
        dense_eval,
        sparse_one_hot,
        sparse_dense_poly,
        sparse_eval,
        setup: DoryScheme::setup_prover(num_vars),
    }
}

fn akita_batch_case(logical_num_vars: usize) -> AkitaBatchCase {
    let physical_num_vars = logical_num_vars + BATCH_PREFIX_BITS;
    let logical_point = deterministic_point(logical_num_vars);
    let id_polynomials = BatchId::ALL
        .into_iter()
        .enumerate()
        .map(|(index, id)| {
            (
                id,
                deterministic_dense_poly_with_offset(logical_num_vars, 100 + index as u64),
            )
        })
        .collect::<Vec<_>>();
    let polynomials = id_polynomials
        .iter()
        .map(|(_, polynomial)| polynomial.clone())
        .collect::<Vec<_>>();
    let evaluations = polynomials
        .iter()
        .map(|polynomial| polynomial.evaluate(&logical_point))
        .collect::<Vec<_>>();
    let (packing, packed_polynomial) = materialize_packed(&id_polynomials);
    assert_eq!(packing.packed_num_vars, physical_num_vars);
    let packed_claims = BatchId::ALL
        .iter()
        .zip(&evaluations)
        .map(|(&id, &evaluation)| (id, EvaluationClaim::new(logical_point.clone(), evaluation)))
        .collect::<Vec<_>>();
    let (native_setup, _) = AkitaScheme::setup(AkitaSetupParams::new(
        logical_num_vars,
        BATCH_POLYS,
        LAYOUT_DIGEST,
    ))
    .unwrap();
    let (packed_pcs, _) = AkitaScheme::setup(AkitaSetupParams::new(
        physical_num_vars,
        NUM_POLYS,
        LAYOUT_DIGEST,
    ))
    .unwrap();

    AkitaBatchCase {
        logical_point,
        polynomials,
        evaluations,
        native_setup,
        packed_pcs_setup: packed_pcs,
        packing,
        packed_polynomial,
        packed_claims,
    }
}

fn native_batch_statement(
    case: &AkitaBatchCase,
    commitment: jolt_akita::AkitaCommitment,
) -> jolt_akita::AkitaNativeBatchStatement {
    case.evaluations
        .iter()
        .copied()
        .map(|evaluation| VerifierOpeningClaim {
            commitment: commitment.clone(),
            evaluation: EvaluationClaim::new(case.logical_point.clone(), evaluation),
        })
        .collect()
}

fn native_batch_polynomials(case: &AkitaBatchCase) -> jolt_akita::AkitaNativeBatchPolynomials<'_> {
    case.polynomials
        .iter()
        .map(|polynomial| polynomial as &(dyn MultilinearPoly<AkitaField> + '_))
        .collect()
}

fn native_batch_commit(case: &AkitaBatchCase) -> (jolt_akita::AkitaCommitment, AkitaProverHint) {
    AkitaScheme::commit_group(
        &case.native_setup,
        LAYOUT_DIGEST,
        black_box(case.polynomials.as_slice()),
    )
    .expect("black-box batch commit should succeed")
}

fn native_batch_open(
    case: &AkitaBatchCase,
    commitment: jolt_akita::AkitaCommitment,
    hint: AkitaProverHint,
) -> jolt_akita::AkitaBatchProof {
    let mut transcript = Blake2bTranscript::new(b"jolt-akita/black-box-batch-bench");
    <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &case.native_setup,
        native_batch_statement(case, commitment),
        native_batch_polynomials(case),
        hint,
        &mut transcript,
    )
    .expect("black-box batch proof should succeed")
}

fn packed_batch_commit(case: &AkitaBatchCase) -> (jolt_akita::AkitaCommitment, AkitaProverHint) {
    AkitaScheme::commit(black_box(&case.packed_polynomial), &case.packed_pcs_setup).unwrap()
}

fn packed_batch_statement(
    case: &AkitaBatchCase,
    commitment: jolt_akita::AkitaCommitment,
) -> AkitaPackedStatement {
    PrefixPackedStatement::new(commitment, case.packed_claims.clone())
}

fn packed_batch_open(
    case: &AkitaBatchCase,
    commitment: jolt_akita::AkitaCommitment,
    hint: AkitaProverHint,
) -> AkitaPackedProof {
    let mut transcript = Blake2bTranscript::new(b"jolt-akita/packed-batch-bench");
    let statement = packed_batch_statement(case, commitment);
    prove_packed_openings::<AkitaScheme, BatchId, _>(
        vec![PackedProverObject {
            packing: &case.packing,
            statement: &statement,
            polynomial: &case.packed_polynomial,
            setup: &case.packed_pcs_setup,
        }],
        vec![PackedProverGroup::singleton(0, Some(hint))],
        &mut transcript,
    )
    .expect("packed batch proof should succeed")
}

fn wrapper_commit(
    case: &AkitaCase,
    path: DataPath,
) -> (jolt_akita::AkitaCommitment, AkitaProverHint) {
    match path {
        DataPath::DenseData => {
            AkitaScheme::commit(black_box(&case.dense_poly), &case.setup).unwrap()
        }
        DataPath::SparseDataSparsePath => {
            AkitaScheme::commit(black_box(&case.sparse_one_hot), &case.setup).unwrap()
        }
        DataPath::SparseDataDensePath => AkitaScheme::commit_group(
            &case.setup,
            LAYOUT_DIGEST,
            black_box(std::slice::from_ref(&case.sparse_dense_poly)),
        )
        .expect("dense-path sparse commit should succeed"),
    }
}

fn wrapper_open(
    case: &AkitaCase,
    path: DataPath,
    hint: AkitaProverHint,
) -> jolt_akita::AkitaBatchProof {
    let mut transcript = Blake2bTranscript::new(b"jolt-akita/bench");
    match path {
        DataPath::DenseData => AkitaScheme::open(
            &case.dense_poly,
            &case.point,
            case.dense_eval,
            &case.setup,
            Some(hint),
            &mut transcript,
        )
        .unwrap(),
        DataPath::SparseDataSparsePath => AkitaScheme::open(
            &case.sparse_one_hot,
            &case.point,
            case.sparse_eval,
            &case.setup,
            Some(hint),
            &mut transcript,
        )
        .unwrap(),
        DataPath::SparseDataDensePath => AkitaScheme::open(
            &case.sparse_dense_poly,
            &case.point,
            case.sparse_eval,
            &case.setup,
            Some(hint),
            &mut transcript,
        )
        .unwrap(),
    }
}

fn akita_prover_commit_dense(
    setup: &AkitaProverBenchSetup,
    poly: &BackendDensePoly,
) -> (BackendCommitment, BackendHint) {
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend,
        &setup.dense_prepared,
        setup.dense_prover.expanded.as_ref(),
    )
    .expect("uniform backend stack");
    BackendScheme::commit(
        &setup.dense_prover,
        black_box(std::slice::from_ref(poly)),
        &stack,
    )
    .expect("Akita backend dense commit should succeed")
}

fn akita_prover_commit_one_hot(
    setup: &AkitaProverBenchSetup,
    poly: &BackendOneHotPoly,
) -> (BackendCommitment, BackendHint) {
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend,
        &setup.one_hot_prepared,
        setup.one_hot_prover.expanded.as_ref(),
    )
    .expect("uniform backend stack");
    OneHotBackendScheme::commit(
        &setup.one_hot_prover,
        black_box(std::slice::from_ref(poly)),
        &stack,
    )
    .expect("Akita backend one-hot commit should succeed")
}

fn akita_prover_claims<'a, P>(
    point: &[AkitaField],
    evaluations: Vec<AkitaField>,
    polynomials: &'a [&'a P],
    commitment: &BackendCommitment,
    hint: BackendHint,
) -> ProverOpeningData<'a, AkitaField, P, AkitaField> {
    let group = PolynomialGroupClaims::new(
        PointVariableSelection::prefix(point.len(), point.len()).expect("full-point prover group"),
        evaluations,
        commitment.clone(),
    )
    .expect("prover group claims");
    let claims = OpeningClaims::from_groups(point.to_vec(), vec![group]).expect("prover claims");
    ProverOpeningData::new(claims, vec![hint], vec![polynomials]).expect("prover opening data")
}

fn akita_prover_open_dense(
    case: &AkitaCase,
    poly: &BackendDensePoly,
    evaluation: AkitaField,
    commitment: BackendCommitment,
    hint: BackendHint,
) -> akita_types::AkitaBatchedProof<AkitaField, AkitaField> {
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend,
        &case.akita_prover_setup.dense_prepared,
        case.akita_prover_setup.dense_prover.expanded.as_ref(),
    )
    .expect("uniform backend stack");
    let poly_refs = [poly];
    let mut transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/native-bench");
    BackendScheme::batched_prove(
        &case.akita_prover_setup.dense_prover,
        akita_prover_claims(&case.point, vec![evaluation], &poly_refs, &commitment, hint),
        &stack,
        &mut transcript,
        BasisMode::Lagrange,
    )
    .expect("Akita backend dense proof should succeed")
}

fn akita_prover_open_one_hot(
    case: &AkitaCase,
    poly: &BackendOneHotPoly,
    evaluation: AkitaField,
    commitment: BackendCommitment,
    hint: BackendHint,
) -> akita_types::AkitaBatchedProof<AkitaField, AkitaField> {
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend,
        &case.akita_prover_setup.one_hot_prepared,
        case.akita_prover_setup.one_hot_prover.expanded.as_ref(),
    )
    .expect("uniform backend stack");
    let poly_refs = [poly];
    let backend_point = reverse_point(&case.point);
    let mut transcript = AkitaTranscript::<AkitaField>::new(b"jolt-akita/native-bench");
    OneHotBackendScheme::batched_prove(
        &case.akita_prover_setup.one_hot_prover,
        akita_prover_claims(
            &backend_point,
            vec![evaluation],
            &poly_refs,
            &commitment,
            hint,
        ),
        &stack,
        &mut transcript,
        BasisMode::Lagrange,
    )
    .expect("Akita backend one-hot proof should succeed")
}

fn dory_commit(case: &DoryCase, path: DataPath) -> (DoryCommitment, DoryHint) {
    match path {
        DataPath::DenseData => {
            DoryScheme::commit(black_box(&case.dense_poly), &case.setup).unwrap()
        }
        DataPath::SparseDataSparsePath => {
            DoryScheme::commit(black_box(&case.sparse_one_hot), &case.setup).unwrap()
        }
        DataPath::SparseDataDensePath => {
            DoryScheme::commit(black_box(&case.sparse_dense_poly), &case.setup).unwrap()
        }
    }
}

fn dory_open(case: &DoryCase, path: DataPath, hint: DoryHint) -> jolt_dory::DoryProof {
    let mut transcript = Blake2bTranscript::new(b"jolt-akita/dory-baseline-bench");
    match path {
        DataPath::DenseData => DoryScheme::open(
            &case.dense_poly,
            &case.point,
            case.dense_eval,
            &case.setup,
            Some(hint),
            &mut transcript,
        )
        .unwrap(),
        DataPath::SparseDataSparsePath => DoryScheme::open(
            &case.sparse_one_hot,
            &case.point,
            case.sparse_eval,
            &case.setup,
            Some(hint),
            &mut transcript,
        )
        .unwrap(),
        DataPath::SparseDataDensePath => DoryScheme::open(
            &case.sparse_dense_poly,
            &case.point,
            case.sparse_eval,
            &case.setup,
            Some(hint),
            &mut transcript,
        )
        .unwrap(),
    }
}

fn bench_jolt_akita_commit(c: &mut Criterion) {
    if !criterion_filter_matches("jolt_akita/commit") {
        return;
    }
    for num_vars in num_vars_cases() {
        let group_name = format!("jolt_akita/commit/nv{num_vars}");
        if !criterion_filter_matches(&group_name) {
            continue;
        }
        let mut group = c.benchmark_group(group_name);
        configure_group(&mut group, num_vars);
        let case = akita_case(num_vars);
        for path in DataPath::ALL {
            group.bench_with_input(
                BenchmarkId::from_parameter(path.label()),
                &(path, &case),
                |b, &(path, case)| {
                    b.iter(|| black_box(wrapper_commit(case, path)));
                },
            );
        }
        group.finish();
    }
}

fn bench_jolt_akita_open(c: &mut Criterion) {
    if !criterion_filter_matches("jolt_akita/open") {
        return;
    }
    for num_vars in num_vars_cases() {
        let group_name = format!("jolt_akita/open/nv{num_vars}");
        if !criterion_filter_matches(&group_name) {
            continue;
        }
        let mut group = c.benchmark_group(group_name);
        configure_group(&mut group, num_vars);
        let case = akita_case(num_vars);
        for path in DataPath::ALL {
            let (_, hint) = wrapper_commit(&case, path);
            group.bench_with_input(
                BenchmarkId::from_parameter(path.label()),
                &(path, &case, hint),
                |b, &(path, case, ref hint)| {
                    b.iter_batched(
                        || hint.clone(),
                        |hint| black_box(wrapper_open(case, path, hint)),
                        BatchSize::SmallInput,
                    );
                },
            );
        }
        group.finish();
    }
}

fn bench_akita_prover_commit(c: &mut Criterion) {
    if !criterion_filter_matches("akita_prover/commit") {
        return;
    }
    for num_vars in num_vars_cases() {
        let group_name = format!("akita_prover/commit/nv{num_vars}");
        if !criterion_filter_matches(&group_name) {
            continue;
        }
        let mut group = c.benchmark_group(group_name);
        configure_group(&mut group, num_vars);
        let case = akita_case(num_vars);
        group.bench_function(
            BenchmarkId::from_parameter(DataPath::DenseData.label()),
            |b| {
                b.iter(|| {
                    black_box(akita_prover_commit_dense(
                        &case.akita_prover_setup,
                        &case.backend_dense_poly,
                    ));
                });
            },
        );
        group.bench_function(
            BenchmarkId::from_parameter(DataPath::SparseDataSparsePath.label()),
            |b| {
                b.iter(|| {
                    black_box(akita_prover_commit_one_hot(
                        &case.akita_prover_setup,
                        &case.backend_sparse_one_hot_poly,
                    ));
                });
            },
        );
        group.bench_function(
            BenchmarkId::from_parameter(DataPath::SparseDataDensePath.label()),
            |b| {
                b.iter(|| {
                    black_box(akita_prover_commit_dense(
                        &case.akita_prover_setup,
                        &case.backend_sparse_dense_poly,
                    ));
                });
            },
        );
        group.finish();
    }
}

fn bench_akita_prover_open(c: &mut Criterion) {
    if !criterion_filter_matches("akita_prover/open") {
        return;
    }
    for num_vars in num_vars_cases() {
        let group_name = format!("akita_prover/open/nv{num_vars}");
        if !criterion_filter_matches(&group_name) {
            continue;
        }
        let mut group = c.benchmark_group(group_name);
        configure_group(&mut group, num_vars);
        let case = akita_case(num_vars);
        let (dense_commitment, dense_hint) =
            akita_prover_commit_dense(&case.akita_prover_setup, &case.backend_dense_poly);
        let (sparse_commitment, sparse_hint) = akita_prover_commit_one_hot(
            &case.akita_prover_setup,
            &case.backend_sparse_one_hot_poly,
        );
        let (sparse_dense_commitment, sparse_dense_hint) =
            akita_prover_commit_dense(&case.akita_prover_setup, &case.backend_sparse_dense_poly);

        group.bench_function(
            BenchmarkId::from_parameter(DataPath::DenseData.label()),
            |b| {
                b.iter_batched(
                    || (dense_commitment.clone(), dense_hint.clone()),
                    |(commitment, hint)| {
                        black_box(akita_prover_open_dense(
                            &case,
                            &case.backend_dense_poly,
                            case.dense_eval,
                            commitment,
                            hint,
                        ));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_function(
            BenchmarkId::from_parameter(DataPath::SparseDataSparsePath.label()),
            |b| {
                b.iter_batched(
                    || (sparse_commitment.clone(), sparse_hint.clone()),
                    |(commitment, hint)| {
                        black_box(akita_prover_open_one_hot(
                            &case,
                            &case.backend_sparse_one_hot_poly,
                            case.sparse_eval,
                            commitment,
                            hint,
                        ));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_function(
            BenchmarkId::from_parameter(DataPath::SparseDataDensePath.label()),
            |b| {
                b.iter_batched(
                    || (sparse_dense_commitment.clone(), sparse_dense_hint.clone()),
                    |(commitment, hint)| {
                        black_box(akita_prover_open_dense(
                            &case,
                            &case.backend_sparse_dense_poly,
                            case.sparse_eval,
                            commitment,
                            hint,
                        ));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.finish();
    }
}

fn bench_dory_commit(c: &mut Criterion) {
    if !criterion_filter_matches("dory_baseline/commit") {
        return;
    }
    for num_vars in num_vars_cases() {
        let group_name = format!("dory_baseline/commit/nv{num_vars}");
        if !criterion_filter_matches(&group_name) {
            continue;
        }
        let mut group = c.benchmark_group(group_name);
        configure_group(&mut group, num_vars);
        let case = dory_case(num_vars);
        for path in DataPath::ALL {
            group.bench_with_input(
                BenchmarkId::from_parameter(path.label()),
                &(path, &case),
                |b, &(path, case)| {
                    b.iter(|| black_box(dory_commit(case, path)));
                },
            );
        }
        group.finish();
    }
}

fn bench_dory_open(c: &mut Criterion) {
    if !criterion_filter_matches("dory_baseline/open") {
        return;
    }
    for num_vars in num_vars_cases() {
        let group_name = format!("dory_baseline/open/nv{num_vars}");
        if !criterion_filter_matches(&group_name) {
            continue;
        }
        let mut group = c.benchmark_group(group_name);
        configure_group(&mut group, num_vars);
        let case = dory_case(num_vars);
        for path in DataPath::ALL {
            let (_, hint) = dory_commit(&case, path);
            group.bench_with_input(
                BenchmarkId::from_parameter(path.label()),
                &(path, &case, hint),
                |b, &(path, case, ref hint)| {
                    b.iter_batched(
                        || hint.clone(),
                        |hint| black_box(dory_open(case, path, hint)),
                        BatchSize::SmallInput,
                    );
                },
            );
        }
        group.finish();
    }
}

fn bench_akita_native_commit(c: &mut Criterion) {
    if !criterion_filter_matches("akita_native/commit") {
        return;
    }
    for logical_num_vars in batch_logical_num_vars_cases() {
        let physical_num_vars = logical_num_vars + BATCH_PREFIX_BITS;
        let group_name = format!(
            "akita_native/commit/logical_nv{logical_num_vars}_total_nv{physical_num_vars}_np{BATCH_POLYS}"
        );
        if !criterion_filter_matches(&group_name) {
            continue;
        }
        let mut group = c.benchmark_group(group_name);
        configure_group(&mut group, physical_num_vars);
        let case = akita_batch_case(logical_num_vars);
        group.bench_function(BenchmarkId::from_parameter("group_commit"), |b| {
            b.iter(|| black_box(native_batch_commit(&case)));
        });
        group.finish();
    }
}

fn bench_akita_native_open(c: &mut Criterion) {
    if !criterion_filter_matches("akita_native/open") {
        return;
    }
    for logical_num_vars in batch_logical_num_vars_cases() {
        let physical_num_vars = logical_num_vars + BATCH_PREFIX_BITS;
        let group_name = format!(
            "akita_native/open/logical_nv{logical_num_vars}_total_nv{physical_num_vars}_np{BATCH_POLYS}"
        );
        if !criterion_filter_matches(&group_name) {
            continue;
        }
        let mut group = c.benchmark_group(group_name);
        configure_group(&mut group, physical_num_vars);
        let case = akita_batch_case(logical_num_vars);
        let (commitment, hint) = native_batch_commit(&case);
        group.bench_function(BenchmarkId::from_parameter("batch_prove"), |b| {
            b.iter_batched(
                || (commitment.clone(), hint.clone()),
                |(commitment, hint)| black_box(native_batch_open(&case, commitment, hint)),
                BatchSize::SmallInput,
            );
        });
        group.finish();
    }
}

fn bench_akita_packed_materialize(c: &mut Criterion) {
    if !criterion_filter_matches("akita_packed/materialize") {
        return;
    }
    for logical_num_vars in batch_logical_num_vars_cases() {
        let physical_num_vars = logical_num_vars + BATCH_PREFIX_BITS;
        let group_name = format!(
            "akita_packed/materialize/logical_nv{logical_num_vars}_packed_nv{physical_num_vars}_np{BATCH_POLYS}"
        );
        if !criterion_filter_matches(&group_name) {
            continue;
        }
        let mut group = c.benchmark_group(group_name);
        configure_group(&mut group, physical_num_vars);
        let id_polynomials = BatchId::ALL
            .into_iter()
            .enumerate()
            .map(|(index, id)| {
                (
                    id,
                    deterministic_dense_poly_with_offset(logical_num_vars, 100 + index as u64),
                )
            })
            .collect::<Vec<_>>();
        group.bench_function(BenchmarkId::from_parameter("pack_polynomial"), |b| {
            b.iter(|| black_box(materialize_packed(black_box(&id_polynomials))));
        });
        group.finish();
    }
}

fn bench_akita_packed_commit(c: &mut Criterion) {
    if !criterion_filter_matches("akita_packed/commit") {
        return;
    }
    for logical_num_vars in batch_logical_num_vars_cases() {
        let physical_num_vars = logical_num_vars + BATCH_PREFIX_BITS;
        let group_name = format!(
            "akita_packed/commit/logical_nv{logical_num_vars}_packed_nv{physical_num_vars}_np{BATCH_POLYS}"
        );
        if !criterion_filter_matches(&group_name) {
            continue;
        }
        let mut group = c.benchmark_group(group_name);
        configure_group(&mut group, physical_num_vars);
        let case = akita_batch_case(logical_num_vars);
        group.bench_function(BenchmarkId::from_parameter("packed_commit"), |b| {
            b.iter(|| black_box(packed_batch_commit(&case)));
        });
        group.finish();
    }
}

fn bench_akita_packed_open(c: &mut Criterion) {
    if !criterion_filter_matches("akita_packed/open") {
        return;
    }
    for logical_num_vars in batch_logical_num_vars_cases() {
        let physical_num_vars = logical_num_vars + BATCH_PREFIX_BITS;
        let group_name = format!(
            "akita_packed/open/logical_nv{logical_num_vars}_packed_nv{physical_num_vars}_np{BATCH_POLYS}"
        );
        if !criterion_filter_matches(&group_name) {
            continue;
        }
        let mut group = c.benchmark_group(group_name);
        configure_group(&mut group, physical_num_vars);
        let case = akita_batch_case(logical_num_vars);
        let (commitment, hint) = packed_batch_commit(&case);
        group.bench_function(BenchmarkId::from_parameter("packed_prove"), |b| {
            b.iter_batched(
                || (commitment.clone(), hint.clone()),
                |(commitment, hint)| black_box(packed_batch_open(&case, commitment, hint)),
                BatchSize::SmallInput,
            );
        });
        group.finish();
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        bench_jolt_akita_commit,
        bench_jolt_akita_open,
        bench_akita_prover_commit,
        bench_akita_prover_open,
        bench_dory_commit,
        bench_dory_open,
        bench_akita_native_commit,
        bench_akita_native_open,
        bench_akita_packed_materialize,
        bench_akita_packed_commit,
        bench_akita_packed_open
);

fn run_trace_profile() {
    use jolt_profiling::{setup_tracing, TracingFormat};

    let num_vars = trace_num_vars();
    let trace_name = format!("jolt_akita_paths_n{num_vars}");
    let _guards = setup_tracing(&[TracingFormat::Chrome], &trace_name);
    let case = akita_case(num_vars);

    for path in DataPath::ALL {
        let _span = tracing::info_span!("jolt_akita_trace_profile", path = path.label()).entered();
        let (_, wrapper_hint) = wrapper_commit(&case, path);
        for _ in 0..3 {
            black_box(wrapper_open(&case, path, wrapper_hint.clone()));
        }

        match path {
            DataPath::DenseData => {
                let (commitment, hint) =
                    akita_prover_commit_dense(&case.akita_prover_setup, &case.backend_dense_poly);
                for _ in 0..3 {
                    black_box(akita_prover_open_dense(
                        &case,
                        &case.backend_dense_poly,
                        case.dense_eval,
                        commitment.clone(),
                        hint.clone(),
                    ));
                }
            }
            DataPath::SparseDataSparsePath => {
                let (commitment, hint) = akita_prover_commit_one_hot(
                    &case.akita_prover_setup,
                    &case.backend_sparse_one_hot_poly,
                );
                for _ in 0..3 {
                    black_box(akita_prover_open_one_hot(
                        &case,
                        &case.backend_sparse_one_hot_poly,
                        case.sparse_eval,
                        commitment.clone(),
                        hint.clone(),
                    ));
                }
            }
            DataPath::SparseDataDensePath => {
                let (commitment, hint) = akita_prover_commit_dense(
                    &case.akita_prover_setup,
                    &case.backend_sparse_dense_poly,
                );
                for _ in 0..3 {
                    black_box(akita_prover_open_dense(
                        &case,
                        &case.backend_sparse_dense_poly,
                        case.sparse_eval,
                        commitment.clone(),
                        hint.clone(),
                    ));
                }
            }
        }
    }
}

fn main() {
    if std::env::var_os("JOLT_AKITA_TRACE_PROFILE").is_some() {
        run_trace_profile();
    } else {
        benches();
    }
}
