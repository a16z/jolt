//! Linker-level tests for the `fuse_ops` trait surface and
//! `JOLT_FUSE_DEBUG=1` dual-path validation harness installed by ticket 0.
//!
//! These verify the wiring in [`jolt_compute::link`]: the backend's
//! `fuse_ops` return value combined with [`FuseDebugMode::from_env`]
//! should yield the right combination of `ops` and `shadow_ops` on the
//! produced [`Executable`]. The runtime `BatchRoundEvaluate` shadow
//! assertion is exercised end-to-end when the standard correctness gate
//! (muldiv / `modular_self_verify`) is re-run with `JOLT_FUSE_DEBUG=1`.
//!
//! Env-var reads happen inside `link()`, so every test that toggles
//! `JOLT_FUSE_DEBUG` takes the shared `ENV_LOCK` to serialize access —
//! `#[cargo nextest]` runs integration tests in parallel by default.

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

use jolt_compiler::module::{BatchIdx, ClaimFormula, Module, Op, Schedule, VerifierSchedule};
use jolt_compiler::{BindingOrder, KernelSpec, PolynomialId};
use jolt_compute::{
    link, BatchInstanceSpec, Buf, ComputeBackend, FuseDebugMode, HandleId, HandleShape,
    LookupTraceData, Scalar,
};
use jolt_field::Field;

type Fr = jolt_field::Fr;

// ─────────────────────────────────────────────────────────────────────────────
// Test backend
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal `ComputeBackend` used only for linker-wiring tests.
///
/// `link()` calls `backend.compile(&spec)` once per kernel and
/// `backend.fuse_ops(&ops)` once. With an empty `module.prover.kernels`,
/// only `fuse_ops` fires; every other method panics because the linker
/// never dispatches there. Tests configure `fuse_result` to produce the
/// desired `Some(..) | None` response.
type FuseFn = dyn Fn(&[Op]) -> Option<Vec<Op>> + Send + Sync;

struct TestBackend {
    fuse_result: Box<FuseFn>,
}

impl TestBackend {
    fn new<R>(fuse_result: R) -> Self
    where
        R: Fn(&[Op]) -> Option<Vec<Op>> + Send + Sync + 'static,
    {
        Self {
            fuse_result: Box::new(fuse_result),
        }
    }
}

macro_rules! never {
    ($name:literal) => {
        unreachable!(concat!(
            "TestBackend::",
            $name,
            " must not be called during linker-only tests"
        ))
    };
}

impl ComputeBackend for TestBackend {
    type Buffer<T: Scalar> = Vec<T>;
    type CompiledKernel<F: Field> = ();

    fn fuse_ops(&self, ops: &[Op]) -> Option<Vec<Op>> {
        (self.fuse_result)(ops)
    }

    fn compile<F: Field>(&self, _spec: &KernelSpec) -> Self::CompiledKernel<F> {
        never!("compile")
    }

    fn reduce<F: Field>(
        &self,
        _kernel: &Self::CompiledKernel<F>,
        _inputs: &[&Buf<Self, F>],
        _challenges: &[F],
    ) -> Vec<F> {
        never!("reduce")
    }

    fn bind<F: Field>(
        &self,
        _kernel: &Self::CompiledKernel<F>,
        _inputs: &mut [Buf<Self, F>],
        _scalar: F,
    ) {
        never!("bind")
    }

    fn interpolate_inplace<F: Field>(
        &self,
        _buf: &mut Self::Buffer<F>,
        _scalar: F,
        _order: BindingOrder,
    ) {
        never!("interpolate_inplace")
    }

    fn upload<T: Scalar>(&self, _data: &[T]) -> Self::Buffer<T> {
        never!("upload")
    }

    fn download<T: Scalar>(&self, _buf: &Self::Buffer<T>) -> Vec<T> {
        never!("download")
    }

    fn alloc<T: Scalar>(&self, _len: usize) -> Self::Buffer<T> {
        never!("alloc")
    }

    fn len<T: Scalar>(&self, _buf: &Self::Buffer<T>) -> usize {
        never!("len")
    }

    fn eq_table<F: Field>(&self, _point: &[F]) -> Self::Buffer<F> {
        never!("eq_table")
    }

    fn lt_table<F: Field>(&self, _point: &[F]) -> Self::Buffer<F> {
        never!("lt_table")
    }

    fn eq_plus_one_table<F: Field>(&self, _point: &[F]) -> (Self::Buffer<F>, Self::Buffer<F>) {
        never!("eq_plus_one_table")
    }

    fn duplicate_interleave<F: Field>(&self, _buf: &Self::Buffer<F>) -> Self::Buffer<F> {
        never!("duplicate_interleave")
    }

    fn regroup_constraints<F: Field>(
        &self,
        _buf: &Self::Buffer<F>,
        _group_indices: &[Vec<usize>],
        _old_stride: usize,
        _new_stride: usize,
        _num_cycles: usize,
    ) -> Self::Buffer<F> {
        never!("regroup_constraints")
    }

    fn evaluate_claim<F: Field>(
        &self,
        _formula: &ClaimFormula,
        _evaluations: &HashMap<PolynomialId, F>,
        _staged_evals: &HashMap<(PolynomialId, usize), F>,
        _challenges: &[F],
    ) -> F {
        never!("evaluate_claim")
    }

    fn evaluate_mle<F: Field>(&self, _evals: &[F], _point: &[F]) -> F {
        never!("evaluate_mle")
    }

    fn uniskip_encode<F: Field>(
        &self,
        _raw_evals: &mut [F],
        _domain_size: usize,
        _domain_start: i64,
        _tau: F,
        _zero_base: bool,
        _num_coeffs: usize,
    ) -> Vec<F> {
        never!("uniskip_encode")
    }

    fn compressed_encode<F: Field>(&self, _evals: &[F]) -> Vec<F> {
        never!("compressed_encode")
    }

    fn interpolate_evaluate<F: Field>(&self, _evals: &[F], _point: F) -> F {
        never!("interpolate_evaluate")
    }

    fn extend_evals<F: Field>(&self, _evals: &[F], _target_len: usize) -> Vec<F> {
        never!("extend_evals")
    }

    fn scale_from_host<F: Field>(&self, _data: &[F], _scale: F) -> Self::Buffer<F> {
        never!("scale_from_host")
    }

    fn transpose_from_host<F: Field>(
        &self,
        _data: &[F],
        _rows: usize,
        _cols: usize,
    ) -> Self::Buffer<F> {
        never!("transpose_from_host")
    }

    fn eq_gather<F: Field>(&self, _eq_point: &[F], _index_data: &[F]) -> Self::Buffer<F> {
        never!("eq_gather")
    }

    fn eq_pushforward<F: Field>(
        &self,
        _eq_point: &[F],
        _index_data: &[F],
        _output_size: usize,
    ) -> Self::Buffer<F> {
        never!("eq_pushforward")
    }

    fn eq_project<F: Field>(
        &self,
        _source_data: &[F],
        _eq_point: &[F],
        _inner_size: usize,
        _outer_size: usize,
    ) -> Self::Buffer<F> {
        never!("eq_project")
    }

    fn lagrange_project<F: Field>(
        &self,
        _buf: &Self::Buffer<F>,
        _challenge: F,
        _domain_start: i64,
        _domain_size: usize,
        _stride: usize,
        _group_offsets: &[usize],
        _scale: F,
    ) -> Self::Buffer<F> {
        never!("lagrange_project")
    }

    fn segmented_reduce<F: Field>(
        &self,
        _kernel: &Self::CompiledKernel<F>,
        _inputs: &[&Self::Buffer<F>],
        _outer_eq: &[F],
        _inner_only: &[bool],
        _inner_size: usize,
        _challenges: &[F],
    ) -> Vec<F> {
        never!("segmented_reduce")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────────

/// Serializes `JOLT_FUSE_DEBUG` reads across parallel tests.
fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn empty_module() -> Module {
    Module {
        polys: vec![],
        challenges: vec![],
        prover: Schedule {
            ops: vec![],
            kernels: vec![],
            batched_sumchecks: vec![],
        },
        verifier: VerifierSchedule {
            ops: vec![],
            num_challenges: 0,
            num_polys: 0,
            num_stages: 0,
        },
    }
}

/// Scoped env-var toggle — restores prior value on drop.
struct EnvScope {
    key: &'static str,
    prior: Option<String>,
}

impl EnvScope {
    fn set(key: &'static str, value: &str) -> Self {
        let prior = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key, prior }
    }

    fn unset(key: &'static str) -> Self {
        let prior = std::env::var(key).ok();
        std::env::remove_var(key);
        Self { key, prior }
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        match &self.prior {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fuse_debug_mode_from_env_unset_is_off() {
    let _lock = env_lock();
    let _scope = EnvScope::unset("JOLT_FUSE_DEBUG");
    assert_eq!(FuseDebugMode::from_env(), FuseDebugMode::Off);
    assert!(!FuseDebugMode::from_env().is_on());
}

#[test]
fn fuse_debug_mode_from_env_one_is_on() {
    let _lock = env_lock();
    let _scope = EnvScope::set("JOLT_FUSE_DEBUG", "1");
    assert_eq!(FuseDebugMode::from_env(), FuseDebugMode::On);
    assert!(FuseDebugMode::from_env().is_on());
}

#[test]
fn fuse_debug_mode_from_env_other_value_is_off() {
    let _lock = env_lock();
    let _scope = EnvScope::set("JOLT_FUSE_DEBUG", "true");
    assert_eq!(FuseDebugMode::from_env(), FuseDebugMode::Off);
}

/// Identity path: backend opts out of fusion (`fuse_ops` returns `None`).
/// Linker must preserve the compiler's raw ops and leave `shadow_ops`
/// empty regardless of env var.
#[test]
fn link_identity_fuse_no_shadow_when_debug_off() {
    let _lock = env_lock();
    let _scope = EnvScope::unset("JOLT_FUSE_DEBUG");
    let backend = TestBackend::new(|_| None);
    let exe = link::<_, Fr>(empty_module(), &backend);
    assert!(exe.shadow_ops.is_none());
    assert!(!exe.has_fuse_debug());
    assert!(exe.ops.is_empty());
}

#[test]
fn link_identity_fuse_no_shadow_when_debug_on() {
    let _lock = env_lock();
    let _scope = EnvScope::set("JOLT_FUSE_DEBUG", "1");
    let backend = TestBackend::new(|_| None);
    let exe = link::<_, Fr>(empty_module(), &backend);
    assert!(
        exe.shadow_ops.is_none(),
        "identity fuse must not produce shadow — harness would compare fused vs itself"
    );
    assert!(!exe.has_fuse_debug());
}

/// Non-identity path: backend returns `Some(..)`. Off → no shadow (release
/// mode); On → shadow is the compiler's original op stream, available to
/// the runtime's `BatchRoundEvaluate` dual-path assertion.
#[test]
fn link_non_identity_fuse_no_shadow_when_debug_off() {
    let _lock = env_lock();
    let _scope = EnvScope::unset("JOLT_FUSE_DEBUG");
    let fused = vec![Op::BatchRoundEvaluate {
        batch: BatchIdx(0),
        round: 0,
        instances: vec![],
    }];
    let fused_for_backend = fused.clone();
    let backend = TestBackend::new(move |_| Some(fused_for_backend.clone()));
    let exe = link::<_, Fr>(empty_module(), &backend);
    assert!(
        exe.shadow_ops.is_none(),
        "debug off: shadow must stay empty to keep release hot path free"
    );
    assert_eq!(exe.ops.len(), 1);
    assert!(matches!(exe.ops[0], Op::BatchRoundEvaluate { .. }));
}

#[test]
fn link_non_identity_fuse_populates_shadow_when_debug_on() {
    let _lock = env_lock();
    let _scope = EnvScope::set("JOLT_FUSE_DEBUG", "1");
    // Give the module a non-empty raw op stream so the backend sees it
    // AND the linker has something to preserve as the shadow. `Preamble`
    // is payload-free and trivial to construct.
    let raw = vec![Op::Preamble];
    let mut module = empty_module();
    module.prover.ops = raw;
    let fused = vec![Op::BatchRoundEvaluate {
        batch: BatchIdx(0),
        round: 0,
        instances: vec![],
    }];
    let fused_for_backend = fused.clone();
    let backend = TestBackend::new(move |_| Some(fused_for_backend.clone()));
    let exe = link::<_, Fr>(module, &backend);
    assert!(exe.has_fuse_debug());
    assert_eq!(exe.ops.len(), 1);
    assert!(
        matches!(exe.ops[0], Op::BatchRoundEvaluate { .. }),
        "prover runtime must execute the fused stream"
    );
    let shadow = exe.shadow_ops.as_ref().expect("shadow must be populated");
    assert_eq!(shadow.len(), 1);
    assert!(
        matches!(shadow[0], Op::Preamble),
        "shadow must be the pre-fusion raw stream for dual-path replay"
    );
}

/// The backend sees the raw pre-fusion op stream (what the compiler
/// emitted), not the already-fused rewrite.
#[test]
fn fuse_ops_receives_raw_compiler_stream() {
    let _lock = env_lock();
    let _scope = EnvScope::unset("JOLT_FUSE_DEBUG");
    let raw = vec![
        Op::Preamble,
        Op::BatchRoundBegin {
            batch: BatchIdx(0),
            round: 0,
            max_evals: 2,
            bind_challenge: None,
        },
    ];
    let mut module = empty_module();
    module.prover.ops = raw;

    let backend = TestBackend::new(move |ops| {
        assert_eq!(ops.len(), 2, "fuse_ops must see the compiler's raw stream");
        assert!(matches!(ops[0], Op::Preamble));
        assert!(matches!(ops[1], Op::BatchRoundBegin { .. }));
        None
    });
    let _ = link::<_, Fr>(module, &backend);
}

// Drag imports into scope so `cargo clippy` does not flag them when the
// corresponding branches are not exercised. `BatchInstanceSpec`, the
// handle API, and `LookupTraceData` are part of the public surface we
// assert compiles alongside the new trait method.
#[allow(dead_code)]
fn _compile_public_surface_smoke<B: ComputeBackend>(
    _: &B,
    _: &[BatchInstanceSpec<'_, B, Fr>],
    _: HandleShape<'_, Fr>,
    _: HandleId,
    _: &LookupTraceData,
) {
}
