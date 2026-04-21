//! Opaque, stateful handles managed by [`CpuBackend`].
//!
//! A handle is an ML-runtime–style resource (cf. cuBLAS/cuDNN) — the caller
//! opens one at a shape, binds it round-by-round, queries it, then closes it.
//! The backend owns all state; the caller only holds a [`HandleId`].
//!
//! The storage is type-erased via `Box<dyn Any + Send + Sync>` so the trait
//! surface carries no `dyn` anywhere. Downcasting to `CpuHandleState<F>` is
//! the only type recovery; it fails loudly if misused (wrong F).
//!
//! Shared state lives in a module-private `OnceLock<HandleStore>` — matching
//! the fact that `CpuBackend` is a unit struct. A CUDA or Metal backend would
//! instead hold its own per-instance store.
//!
//! Currently supports the `Scratch` variant; `Eq` (GruenSplitEqPolynomial)
//! follows in a subsequent iteration.
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use jolt_compute::{HandleId, HandleShape};
use jolt_field::Field;

/// Concrete per-`F` state held behind a [`HandleId`].
pub(crate) enum CpuHandleState<F: Field> {
    /// A simple scratchpad; [`CpuBackend::bind_handle`] writes round `r`
    /// values into slot `round`, [`CpuBackend::query_handle`] reads them out.
    Scratch(Vec<F>),
    /// A pre-built equality polynomial evaluation table over a fixed point.
    ///
    /// Built once at [`HandleStore::open`] time via
    /// `jolt_poly::EqPolynomial::evals(point, None)`. Shared across any
    /// number of `query_handle` calls — amortizes the build cost whenever
    /// the same eq point is reused by multiple kernel invocations.
    ///
    /// The point is fixed at open time; `bind_handle` on this variant
    /// panics (a future iter may introduce an incremental-bind variant
    /// backed by a Gruen-style prefix structure).
    ///
    /// Wrapped in `Arc` so long-running parallel consumers
    /// (e.g. `eq_project_from_handle`) can clone the handle and drop the
    /// outer `HandleStore` mutex guard before running the work — otherwise
    /// concurrent handle access across rayon threads serializes on the
    /// process-global store mutex.
    Eq(Arc<Vec<F>>),
}

pub(crate) struct HandleStore {
    entries: Mutex<HashMap<HandleId, Box<dyn Any + Send + Sync>>>,
    next_id: AtomicU32,
}

impl HandleStore {
    fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            next_id: AtomicU32::new(0),
        }
    }

    /// Process-wide store. Matches `CpuBackend` being a unit struct.
    pub(crate) fn global() -> &'static Self {
        static STORE: OnceLock<HandleStore> = OnceLock::new();
        STORE.get_or_init(HandleStore::new)
    }

    pub(crate) fn open<F: Field>(&self, shape: HandleShape<'_, F>) -> HandleId {
        let state: CpuHandleState<F> = match shape {
            HandleShape::Scratch { size } => CpuHandleState::Scratch(vec![F::zero(); size]),
            HandleShape::Eq {
                challenges,
                order: _,
            } => CpuHandleState::Eq(Arc::new(jolt_poly::EqPolynomial::<F>::evals(
                challenges, None,
            ))),
            _ => panic!("HandleShape variant not supported by CpuBackend"),
        };
        let id = HandleId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let _ = self.entries.lock().unwrap().insert(id, Box::new(state));
        id
    }

    pub(crate) fn close(&self, id: HandleId) {
        let _ = self.entries.lock().unwrap().remove(&id);
    }

    pub(crate) fn with_state_mut<F: Field, R>(
        &self,
        id: HandleId,
        f: impl FnOnce(&mut CpuHandleState<F>) -> R,
    ) -> R {
        let mut entries = self.entries.lock().unwrap();
        let boxed = entries
            .get_mut(&id)
            .unwrap_or_else(|| panic!("HandleId({}) not open", id.0));
        let state = boxed
            .downcast_mut::<CpuHandleState<F>>()
            .unwrap_or_else(|| panic!("HandleId({}) opened with different F type", id.0));
        f(state)
    }

    pub(crate) fn with_state<F: Field, R>(
        &self,
        id: HandleId,
        f: impl FnOnce(&CpuHandleState<F>) -> R,
    ) -> R {
        let entries = self.entries.lock().unwrap();
        let boxed = entries
            .get(&id)
            .unwrap_or_else(|| panic!("HandleId({}) not open", id.0));
        let state = boxed
            .downcast_ref::<CpuHandleState<F>>()
            .unwrap_or_else(|| panic!("HandleId({}) opened with different F type", id.0));
        f(state)
    }
}
