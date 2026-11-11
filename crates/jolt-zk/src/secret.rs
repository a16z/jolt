//! Heap buffers of secret values that are zeroed when freed.

use std::ops::{Deref, DerefMut};
use std::slice::Iter;

use jolt_algebra::cfg_chunks_mut;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zeroize::Zeroize;

/// Elements per chunk of [`wipe`].
const WIPE_CHUNK: usize = 1 << 16;

/// A fixed-length heap buffer of secret plain values, overwritten with zero
/// bytes over its whole capacity by [`wipe`] when dropped.
///
/// `zeroize::Zeroizing<Vec<T>>` needs `T: Zeroize`, which the generic field
/// types do not provide, and wipes a `Vec` twice on one thread, the second
/// time byte by byte. It derefs to a slice only, so it never reallocates and
/// leaves an unwiped copy behind; the `Vec` it is built from should not have
/// reallocated either.
pub(crate) struct SecretVec<T> {
    values: Vec<T>,
    /// [`wipe`] for `T`, taken where `T: Copy + Send` is known, so that
    /// neither this struct nor the generic structs that hold one need those
    /// bounds for `Drop`.
    wipe: fn(&mut Vec<T>),
}

impl<T: Copy + Send> From<Vec<T>> for SecretVec<T> {
    fn from(values: Vec<T>) -> Self {
        Self {
            values,
            wipe: wipe::<T>,
        }
    }
}

impl<T> Deref for SecretVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.values
    }
}

impl<T> DerefMut for SecretVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}

impl<'a, T> IntoIterator for &'a SecretVec<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<T> Drop for SecretVec<T> {
    fn drop(&mut self) {
        (self.wipe)(&mut self.values);
    }
}

/// Empties `values` and zeroes its whole capacity with one volatile write
/// per element. With `parallel`, chunks are wiped on Rayon workers because
/// buffers reach tens of megabytes. `T` must own no heap, which `clear` would free unwiped.
pub(crate) fn wipe<T: Send>(values: &mut Vec<T>) {
    values.clear();
    cfg_chunks_mut!(values.spare_capacity_mut(), WIPE_CHUNK)
        .for_each(|chunk| chunk.iter_mut().zeroize());
}
