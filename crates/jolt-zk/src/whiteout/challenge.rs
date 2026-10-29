//! Multiplication by the Σ-protocol's challenge `c ∈ R = F[X]/(X^64 + 1)`.
//!
//! On a native element of `R_{64m}`, `c` acts as `c(Y^m)`: component by
//! component in the stride view, so `c·s` never repacks. The integer form is
//! the exact centred-integer product the rejection rule needs; it reads `c`
//! only as a dense coefficient vector, so neither its time nor its addresses
//! depend on `c` (a rejected attempt's `c` is secret).

use jolt_algebra::{lift_ct, CyclotomicRing, Field};
use jolt_crypto::lattice::CHALLENGE_RING_DIM;

#[cfg(feature = "whiteout-prover")]
use super::WhiteoutError;
#[cfg(feature = "whiteout-prover")]
use rayon::prelude::*;

/// The dense 64-coefficient form of a challenge, from
/// `SparseChallenge::to_dense_ct`.
pub type DenseChallenge = [i8; CHALLENGE_RING_DIM];

/// `c·a` in `Z[X]/(X^64 + 1)`, exactly, for one challenge-ring element:
/// `out[j..] += c_j·a[..64−j]` and `out[..j] −= c_j·a[64−j..]`, at public
/// offsets, in loops the compiler vectorizes.
#[cfg(feature = "whiteout-prover")]
fn negacyclic_i64(c: &DenseChallenge, a: &[i64], out: &mut [i64]) {
    out.fill(0);
    for (j, &cj) in c.iter().enumerate() {
        let cj = i64::from(cj);
        let (high_out, low_out) = out.split_at_mut(j);
        let (low_in, high_in) = a.split_at(CHALLENGE_RING_DIM - j);
        for (dst, &value) in low_out.iter_mut().zip(low_in) {
            *dst += cj * value;
        }
        for (dst, &value) in high_out.iter_mut().zip(high_in) {
            *dst -= cj * value;
        }
    }
}

/// `c(Y^m)·x` for native coefficients `x` of elements of dimension
/// `native_dim = 64m`, exactly over the integers, in parallel over elements,
/// into a caller-owned buffer.
#[cfg(feature = "whiteout-prover")]
pub(crate) fn challenge_times_i64_into(
    c: &DenseChallenge,
    native_dim: usize,
    x: &[i64],
    out: &mut [i64],
) -> Result<(), WhiteoutError> {
    let stride = native_dim / CHALLENGE_RING_DIM;
    if stride == 0 || !x.len().is_multiple_of(native_dim) || out.len() != x.len() {
        return Err(WhiteoutError::InvalidRelation(
            "challenge product needs whole native elements",
        ));
    }
    x.par_chunks_exact(native_dim)
        .zip(out.par_chunks_exact_mut(native_dim))
        .for_each(|(element, target)| {
            let mut component = [0i64; CHALLENGE_RING_DIM];
            let mut product = [0i64; CHALLENGE_RING_DIM];
            for k in 0..stride {
                for (dst, &value) in component
                    .iter_mut()
                    .zip(element.iter().skip(k).step_by(stride))
                {
                    *dst = value;
                }
                negacyclic_i64(c, &component, &mut product);
                for (dst, &value) in target.iter_mut().skip(k).step_by(stride).zip(&product) {
                    *dst = value;
                }
            }
        });
    Ok(())
}

/// `c(Y^m)·a` for one native ring element over `F`.
pub(crate) fn challenge_times_ring<F: Field, const D: usize>(
    c: &DenseChallenge,
    a: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let c = challenge_ring::<F>(c);
    componentwise(a, |component| c * component)
}

/// The transpose of [`challenge_times_ring`]: the weights on `a` of
/// `⟨w, c(Y^m)·a⟩`. Per stride component, multiplication by `c` transposes
/// to `σ(σ(·)·c)`, since `⟨w, c·x⟩ = ct(σ(w)·c·x)`.
pub(crate) fn challenge_times_ring_transpose<F: Field, const D: usize>(
    c: &DenseChallenge,
    w: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let c = challenge_ring::<F>(c);
    componentwise(w, |component| (component.sigma_m1() * c).sigma_m1())
}

/// `map` applied to each stride component of a native ring element.
fn componentwise<F: Field, const D: usize>(
    a: &CyclotomicRing<F, D>,
    map: impl Fn(CyclotomicRing<F, CHALLENGE_RING_DIM>) -> CyclotomicRing<F, CHALLENGE_RING_DIM>,
) -> CyclotomicRing<F, D> {
    let stride = D / CHALLENGE_RING_DIM;
    let mut out = CyclotomicRing::zero();
    for k in 0..stride {
        let component: Vec<F> = a
            .coefficients()
            .iter()
            .skip(k)
            .step_by(stride)
            .copied()
            .collect();
        let product = map(CyclotomicRing::from_slice(&component));
        for (dst, value) in out
            .coefficients_mut()
            .iter_mut()
            .skip(k)
            .step_by(stride)
            .zip(product.coefficients())
        {
            *dst = *value;
        }
    }
    out
}

/// `c` as an element of the challenge ring over `F`.
pub(crate) fn challenge_ring<F: Field>(
    c: &DenseChallenge,
) -> CyclotomicRing<F, CHALLENGE_RING_DIM> {
    CyclotomicRing::from_coefficients(c.map(|coefficient| lift_ct(i64::from(coefficient))))
}
