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
//! Consumer audit (checkpoint 9): no crate in the workspace, and nothing in
//! this rebuild, currently expands any of these macros — see the
//! dropped-specialization notes in `specs/jolt-field-rebuild.md`. The component is ported whole
//! because the approved parity scope names the rayon helpers and all seven
//! macros are equally (un)consumed, leaving no evidence basis for a partial
//! subset.

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
/// Without `parallel`, this uses the equivalent sequential `try_fold`.
#[macro_export]
macro_rules! cfg_try_fold_reduce {
    ($range:expr, $identity:expr, $fold_op:expr, $reduce_op:expr) => {{
        #[cfg(feature = "parallel")]
        let result = $range
            .into_par_iter()
            .try_fold($identity, $fold_op)
            .try_reduce($identity, $reduce_op);
        #[cfg(not(feature = "parallel"))]
        let result = $range.into_iter().try_fold(($identity)(), $fold_op);
        result
    }};
}

pub use crate::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
    cfg_try_fold_reduce,
};
