//! Conditional parallelism utilities.
//!
//! When the `parallel` feature is enabled, the `cfg_iter!` family of macros
//! expand to rayon's parallel iterators. Otherwise they fall back to standard
//! sequential iterators.

#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

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
        let result = rayon::join($f_a, $f_b);
        #[cfg(not(feature = "parallel"))]
        let result = ($f_a(), $f_b());
        result
    }};
}

/// Parallel fold-reduce over a range.
///
/// With `parallel`: `range.into_par_iter().fold(identity, fold_op).reduce(identity, reduce_op)`.
/// Without: `range.into_iter().fold(identity(), fold_op)`.
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

pub use crate::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
};
