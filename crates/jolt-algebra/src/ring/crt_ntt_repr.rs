//! CRT+NTT-domain representation of cyclotomic ring elements.

use std::array::from_fn;

use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::crt::GarnerData;
use crate::ntt::field_limbs::FieldLimbScales;
use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth, I32_LAZY_DOT_BATCH};
use crate::{CanonicalEncoding, CrtCapacity, Field, NttKernelPlan};

/// CRT+NTT-domain representation of a cyclotomic ring element.
///
/// Stores `K` arrays of `D` [`MontCoeff<W>`] values, one per CRT prime.
/// Multiplication is pointwise per prime.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CyclotomicCrtNtt<W: PrimeWidth, const K: usize, const D: usize> {
    /// Per-prime NTT-domain Montgomery limbs.
    pub limbs: [[MontCoeff<W>; D]; K],
}

/// Field types that can convert to/from the CRT+NTT representation.
///
/// Blanket-implemented for all `Field + CanonicalEncoding` types.
pub trait CrtNttConvertibleField: Field + CanonicalEncoding {}

impl<F: Field + CanonicalEncoding> CrtNttConvertibleField for F {}

/// Bundled CRT+NTT parameters for a fixed width/prime-count/degree tuple.
///
/// Keeps primes/twiddles/Garner constants consistent and avoids passing them
/// independently at every call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrtNttParamSet<W: PrimeWidth, const K: usize, const D: usize> {
    /// CRT primes with Montgomery constants.
    pub primes: [NttPrime<W>; K],
    /// Per-prime twiddle tables for forward/inverse NTT.
    ///
    /// Boxed because the tables dominate the parameter set (about 200 KB at
    /// `K = 6`, `D = 1024`); inline, every by-value holder of the parameters
    /// would carry them.
    pub twiddles: Box<[NttTwiddles<W, D>; K]>,
    /// Garner reconstruction constants for CRT lift-back.
    pub garner: GarnerData<K>,
    /// Per-prime constants for reducing field coefficients into residues.
    field_scales: [FieldLimbScales; K],
    /// Host arithmetic kernels selected when this parameter set was prepared.
    kernel_plan: NttKernelPlan,
}

mod convert;
mod lut;
mod mixed;
mod ops;
#[cfg(test)]
mod tests;

pub use lut::{CenteredMontLut, DigitMontLut};
pub use mixed::{
    cyclic_ntt_with_i16_tail_to_ring, mat_vec_i16_with_tail, ntt_with_i16_tail_to_ring,
    I16TailParams,
};

impl<W: PrimeWidth, const K: usize, const D: usize> CrtNttParamSet<W, K, D> {
    /// Host kernel plan selected when these parameters were prepared.
    #[must_use]
    pub const fn kernel_plan(&self) -> NttKernelPlan {
        self.kernel_plan
    }

    /// Number of contiguous products the prepared backend can reduce as one
    /// pointwise dot batch. A value of one preserves column-at-a-time traversal.
    #[must_use]
    pub const fn pointwise_dot_batch_size(&self) -> usize {
        if self.uses_lazy_i32_dot() {
            I32_LAZY_DOT_BATCH
        } else {
            1
        }
    }

    pub(crate) const fn uses_lazy_i32_dot(&self) -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            core::mem::size_of::<W>() == core::mem::size_of::<i32>()
                && self.kernel_plan.uses_avx2_i32_dot()
        }
        #[cfg(target_arch = "aarch64")]
        {
            core::mem::size_of::<W>() == core::mem::size_of::<i32>() && self.kernel_plan.uses_neon()
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Build a full parameter set from CRT primes.
    ///
    /// Computes per-prime twiddles and Garner reconstruction constants.
    pub fn new(primes: [NttPrime<W>; K]) -> Self {
        // Collect on the heap: a `[NttTwiddles; K]` value would take hundreds
        // of kilobytes of stack before boxing.
        let twiddles = primes
            .iter()
            .map(|&prime| NttTwiddles::compute(prime))
            .collect::<Box<[_]>>()
            .try_into()
            .expect("one twiddle table per prime");
        let garner = GarnerData::compute(&primes);
        Self {
            primes,
            twiddles,
            garner,
            field_scales: primes.map(FieldLimbScales::new),
            kernel_plan: NttKernelPlan::detect::<W>(),
        }
    }

    /// Exact CRT product capacity of this parameter set.
    pub fn crt_capacity(&self) -> CrtCapacity {
        CrtCapacity::from_prime_moduli(self.primes.iter().map(|prime| prime.p.to_i64() as u128))
    }

    fn reconstruct<F: CrtNttConvertibleField>(&self, canonical: &[[W; D]; K]) -> [F; D] {
        let (weights, _) = self.garner.field_weights::<F>();
        let mut digits = canonical.map(|limb| limb.map(|residue| residue.to_i64()));
        self.garner.centered_mixed_radix(&mut digits);
        from_fn(|index| {
            self.garner
                .digits_to_field(&from_fn(|limb| digits[limb][index]), &weights)
        })
    }
}
