//! Heap buffers of secret values that are zeroed when freed.

use std::ops::{Deref, DerefMut};

use zeroize::Zeroize;

/// A fixed-length heap buffer of secret plain values, overwritten with zero
/// bytes over its whole capacity (volatile writes, through `zeroize`) when
/// dropped.
///
/// `zeroize::Zeroizing<Vec<T>>` needs `T: Zeroize`, which the generic field
/// types do not provide; this writes zero bytes into the spare capacity
/// instead, so it needs no bound on the elements beyond owning no heap
/// (`T: Copy` at construction). It derefs to a slice only, so it never
/// reallocates and leaves an unwiped copy behind; the `Vec` it is built from
/// should not have reallocated either.
pub(crate) struct SecretVec<T>(Vec<T>);

impl<T: Copy> From<Vec<T>> for SecretVec<T> {
    fn from(values: Vec<T>) -> Self {
        Self(values)
    }
}

impl<T> Deref for SecretVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T> DerefMut for SecretVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T> Drop for SecretVec<T> {
    fn drop(&mut self) {
        // `clear` drops no heap (`T: Copy`) and writes nothing; the whole
        // allocation is then spare capacity.
        self.0.clear();
        self.0.spare_capacity_mut().zeroize();
    }
}
