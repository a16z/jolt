//! Linear functionals as constant terms of ring-linear maps.
//!
//! Write `ct(a)` for the constant coefficient of `a` in `Z_q[X]/(X^D + 1)`
//! and `σ` for the automorphism `X -> X^{-1}`. Then
//! `ct(σ(a) · b) = Σ_i a_i b_i` ([`CyclotomicRing::coefficient_inner_product`]),
//! so a field-linear functional on the coefficients of a vector over the ring
//! is the constant term of a ring-linear map. This is the standard
//! constant-term technique of lattice-based proofs (Esgin–Nguyen–Seiler 2020,
//! Lyubashevsky–Nguyen–Plançon 2022).

use jolt_field::{CyclotomicRing, Field};

use super::WhiteoutError;

/// A field-linear functional `ℓ(s) = Σ_k Σ_i ℓ_{k,i} s_{k,i}` on `R_q^n`.
///
/// It stores `ℓ̂_k`, the ring element with coefficients `(ℓ_{k,i})_i`.
/// [`Self::apply`] is the unique ring-linear map `L` with `ct ∘ L = ℓ`,
/// namely `L(s) = Σ_k σ(ℓ̂_k) s_k`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftedFunctional<F: Field, const D: usize> {
    lifted: Vec<CyclotomicRing<F, D>>,
}

impl<F: Field, const D: usize> LiftedFunctional<F, D> {
    /// The functional with coefficients `coefficients`, listed ring element
    /// by ring element. The length must be a multiple of `D`, and `D` must be
    /// nonzero.
    pub fn from_field_coefficients(coefficients: &[F]) -> Result<Self, WhiteoutError> {
        if D == 0 || !coefficients.len().is_multiple_of(D) {
            return Err(WhiteoutError::FunctionalLength {
                degree: D,
                actual: coefficients.len(),
            });
        }
        Ok(Self {
            lifted: coefficients
                .chunks_exact(D)
                .map(CyclotomicRing::from_slice)
                .collect(),
        })
    }

    /// The number `n` of ring coordinates the functional acts on.
    pub fn len(&self) -> usize {
        self.lifted.len()
    }

    /// Whether the functional acts on no coordinates.
    pub fn is_empty(&self) -> bool {
        self.lifted.is_empty()
    }

    /// The ring-linear map `L(s) = Σ_k σ(ℓ̂_k) s_k`.
    pub fn apply(&self, s: &[CyclotomicRing<F, D>]) -> Result<CyclotomicRing<F, D>, WhiteoutError> {
        self.check_len(s)?;
        let mut out = CyclotomicRing::zero();
        for (l, x) in self.lifted.iter().zip(s) {
            l.sigma_m1().mul_accumulate_into(x, &mut out);
        }
        Ok(out)
    }

    /// The value `ℓ(s)`, which equals `ct(self.apply(s))`, computed with
    /// `n · D` multiplications instead of `n` ring products.
    pub fn evaluate(&self, s: &[CyclotomicRing<F, D>]) -> Result<F, WhiteoutError> {
        self.check_len(s)?;
        Ok(self.lifted.iter().zip(s).fold(F::zero(), |acc, (l, x)| {
            acc + l.coefficient_inner_product(x)
        }))
    }

    fn check_len(&self, s: &[CyclotomicRing<F, D>]) -> Result<(), WhiteoutError> {
        if s.len() == self.lifted.len() {
            Ok(())
        } else {
            Err(WhiteoutError::RingVectorLength {
                expected: self.lifted.len(),
                actual: s.len(),
            })
        }
    }
}
