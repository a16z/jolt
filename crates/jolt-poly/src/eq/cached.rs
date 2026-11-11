use super::alloc::{check_layout, domain_len, EqAlloc, Global, TableInit};
use super::EqPolynomial;
use crate::point::PointLike;
use jolt_field::Field;
use std::ops::{Mul, SubAssign};
use std::panic::Location;
impl<F: Field> EqPolynomial<F> {
    /// `result[j]` is the eq table over the `j` most-significant variables.
    ///
    /// Returns `n+1` tables where `result[0] = [scaling_factor]` (eq over 0 vars).
    #[track_caller]
    pub fn evals_cached_msb<P: PointLike + ?Sized>(r: &P, scaling_factor: Option<F>) -> Vec<Vec<F>>
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F> + SubAssign<F>,
    {
        match Self::evals_cached_msb_in(r, scaling_factor, Global) {
            Ok(v) => v,
            Err(never) => match never {},
        }
    }

    #[track_caller]
    #[tracing::instrument(skip_all, name = "EqPolynomial::evals_cached_msb")]
    pub fn evals_cached_msb_in<P: PointLike + ?Sized, A: EqAlloc>(
        r: &P,
        scaling_factor: Option<F>,
        policy: A,
    ) -> Result<Vec<Vec<F>>, A::Error>
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F> + SubAssign<F>,
    {
        let at = Location::caller();
        Self::build_cached::<_, _, true>(r, scaling_factor, policy, at)
    }

    /// `result[j]` is the eq table over the `j` least-significant variables.
    ///
    /// `result[0] = [scaling_factor]`. Each new variable is the new MSB.
    #[track_caller]
    pub fn evals_cached_lsb<P: PointLike + ?Sized>(r: &P, scaling_factor: Option<F>) -> Vec<Vec<F>>
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        match Self::evals_cached_lsb_in(r, scaling_factor, Global) {
            Ok(v) => v,
            Err(never) => match never {},
        }
    }

    #[track_caller]
    pub fn evals_cached_lsb_in<P: PointLike + ?Sized, A: EqAlloc>(
        r: &P,
        scaling_factor: Option<F>,
        policy: A,
    ) -> Result<Vec<Vec<F>>, A::Error>
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        let at = Location::caller();
        Self::build_cached::<_, _, false>(r, scaling_factor, policy, at)
    }

    fn build_cached<P: PointLike + ?Sized, A: EqAlloc, const MSB: bool>(
        r: &P,
        scaling_factor: Option<F>,
        policy: A,
        at: &'static Location<'static>,
    ) -> Result<Vec<Vec<F>>, A::Error>
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        let len = domain_len(r.num_vars(), policy)?;
        // `len` is a power of two that fits in usize, so the level total
        // `2 * len - 1` fits too.
        let elements = len + (len - 1);
        check_layout::<F, _>(len, policy)?;
        check_layout::<Vec<F>, _>(r.num_vars() + 1, policy)?;
        policy.admit::<F>(elements, at)?;
        let mut evals = policy.storage(r.num_vars() + 1)?;
        for i in 0..=r.num_vars() {
            evals.push(policy.table(
                1 << i,
                TableInit::Repeat(scaling_factor.unwrap_or(F::one())),
            )?);
        }
        Self::fill_cached::<_, MSB>(r, &mut evals);
        Ok(evals)
    }

    #[inline]
    pub(crate) fn fill_cached<P, const MSB: bool>(r: &P, evals: &mut [Vec<F>])
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        if MSB {
            let mut size = 1;
            for j in 0..r.num_vars() {
                let r_j = r.msb(j);
                size *= 2;
                for i in (0..size).rev().step_by(2) {
                    let scalar = evals[j][i / 2];
                    evals[j + 1][i] = scalar * r_j;
                    evals[j + 1][i - 1] = scalar - evals[j + 1][i];
                }
            }
        } else {
            let mut size = 1;
            for j in 0..r.num_vars() {
                let r_j = r.msb(r.num_vars() - 1 - j);
                for i in 0..size {
                    let scalar = evals[j][i];
                    let multiple = 1 << j;
                    evals[j + 1][i + multiple] = scalar * r_j;
                    evals[j + 1][i] = scalar - evals[j + 1][i + multiple];
                }
                size *= 2;
            }
        }
    }
}
