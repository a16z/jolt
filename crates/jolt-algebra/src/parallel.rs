//! Conditional parallelism helpers.
//!
//! The `cfg_*!` macros expand to rayon parallel iterators when a `parallel`
//! feature is enabled and to the standard sequential equivalents otherwise.
//!
//! WARNING: the `#[cfg(feature = "parallel")]` inside each macro body is
//! resolved at the *expansion* site, so the macros dispatch on the consuming
//! crate's own `parallel` feature (which must activate rayon there). This is
//! the baseline's design, ported unchanged; within this crate the expansion
//! site is this crate, so its `parallel` feature governs.
//!
//! These helpers are backend-independent and available in every feature
//! configuration. Consumers: the ring and NTT code here, jolt-crypto's
//! lattice primitives, and Akita's kernels.

#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

#[doc(hidden)]
#[cfg(feature = "parallel")]
pub use rayon::join as __rayon_join;

/// Returns `.par_iter()` when `parallel` is enabled, `.iter()` otherwise.
#[macro_export]
macro_rules! cfg_iter {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        let it = $e.par_iter();
        #[cfg(not(feature = "parallel"))]
        let it = $e.iter();
        it
    }};
}

/// Returns `.par_iter_mut()` when `parallel` is enabled, `.iter_mut()` otherwise.
#[macro_export]
macro_rules! cfg_iter_mut {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        let it = $e.par_iter_mut();
        #[cfg(not(feature = "parallel"))]
        let it = $e.iter_mut();
        it
    }};
}

/// Returns `.into_par_iter()` when `parallel` is enabled, `.into_iter()` otherwise.
#[macro_export]
macro_rules! cfg_into_iter {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        let it = $e.into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let it = $e.into_iter();
        it
    }};
}

/// Returns `.par_chunks(n)` when `parallel` is enabled, `.chunks(n)` otherwise.
#[macro_export]
macro_rules! cfg_chunks {
    ($e:expr, $n:expr) => {{
        #[cfg(feature = "parallel")]
        let it = $e.par_chunks($n);
        #[cfg(not(feature = "parallel"))]
        let it = $e.chunks($n);
        it
    }};
}

/// Returns `.par_chunks_mut(n)` when `parallel` is enabled, `.chunks_mut(n)` otherwise.
#[macro_export]
macro_rules! cfg_chunks_mut {
    ($e:expr, $n:expr) => {{
        #[cfg(feature = "parallel")]
        let it = $e.par_chunks_mut($n);
        #[cfg(not(feature = "parallel"))]
        let it = $e.chunks_mut($n);
        it
    }};
}

/// Runs two closures potentially in parallel via `rayon::join`.
///
/// Without `parallel`: runs them sequentially and returns the pair.
#[macro_export]
macro_rules! cfg_join {
    ($f_a:expr, $f_b:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $crate::solinas::parallel::__rayon_join($f_a, $f_b);
        #[cfg(not(feature = "parallel"))]
        let result = ($f_a(), $f_b());
        result
    }};
}

/// Parallel fold-reduce over a range.
///
/// With `parallel`: `range.into_par_iter().fold(identity, fold_op).reduce(identity, reduce_op)`.
/// Without: `range.into_iter().fold(identity(), fold_op)` — `reduce_op` is
/// unused, so serial and parallel results agree only when `reduce_op` is
/// consistent with `fold_op` (associative combination of partials).
#[macro_export]
macro_rules! cfg_fold_reduce {
    ($range:expr, $identity:expr, $fold_op:expr, $reduce_op:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $range
            .into_par_iter()
            .fold($identity, $fold_op)
            .reduce($identity, $reduce_op);
        #[cfg(not(feature = "parallel"))]
        let result = $range.into_iter().fold(($identity)(), $fold_op);
        result
    }};
}

/// Fallible parallel fold-reduce over a range.
///
/// The identity argument is a zero-argument factory, evaluated once and
/// shared by reference, matching Rayon's `try_fold` and `try_reduce`
/// contracts. Without `parallel`, this uses the equivalent sequential
/// `try_fold`, calling the factory once for its initial accumulator.
#[macro_export]
macro_rules! cfg_try_fold_reduce {
    ($range:expr, $identity_factory:expr, $fold_op:expr, $reduce_op:expr) => {{
        let identity_factory = $identity_factory;
        #[cfg(feature = "parallel")]
        let result = $range
            .into_par_iter()
            .try_fold(&identity_factory, $fold_op)
            .try_reduce(&identity_factory, $reduce_op);
        #[cfg(not(feature = "parallel"))]
        let result = $range.into_iter().try_fold(identity_factory(), $fold_op);
        result
    }};
}

pub use crate::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
    cfg_try_fold_reduce,
};
