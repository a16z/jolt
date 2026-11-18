use crate::eq::alloc::{check_layout, domain_len, EqAlloc, Global};
use crate::{BindingOrder, EqPolynomial, PointLike, Polynomial, HIGH_TO_LOW};
use jolt_field::Field;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::panic::Location;
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GruenSplitEqPolynomial<F: Field> {
    current_index: usize,
    current_scalar: F,
    point: Vec<F>,
    e_in_vec: Vec<Vec<F>>,
    e_out_vec: Vec<Vec<F>>,
    binding_order: BindingOrder,
}

#[path = "round.rs"]
mod round;

/// A read view of one active equality round.
pub struct GruenRound<'state, F: Field> {
    state: &'state GruenSplitEqPolynomial<F>,
    coordinate: F,
}

/// An exclusive round view, consumed when its challenge is bound.
///
/// A live handle prevents borrowing the whole owner mutably:
/// ```compile_fail,E0499
/// use jolt_field::Field;
/// use jolt_poly::GruenSplitEqPolynomial;
/// fn touch<F: Field>(_: &mut GruenSplitEqPolynomial<F>) {}
/// fn conflict<F: Field>(state: &mut GruenSplitEqPolynomial<F>, challenge: F) {
///     if let Some(handle) = state.round_mut() {
///         touch(state);
///         handle.bind(challenge);
///     }
/// }
/// ```
pub struct GruenRoundMut<'state, F: Field> {
    state: &'state mut GruenSplitEqPolynomial<F>,
    coordinate: F,
}

impl<F: Field> GruenSplitEqPolynomial<F> {
    #[track_caller]
    pub fn new<P: PointLike<Coord = F> + ?Sized>(point: &P, binding_order: BindingOrder) -> Self {
        Self::new_with_scaling(point, binding_order, None)
    }

    /// Builds tables in significance order and stores one MSB-first point copy.
    #[track_caller]
    pub fn new_with_scaling<P: PointLike<Coord = F> + ?Sized>(
        point: &P,
        binding_order: BindingOrder,
        scaling_factor: Option<F>,
    ) -> Self {
        match Self::from_point_in(point, binding_order, scaling_factor, Global) {
            Ok(v) => v,
            Err(never) => match never {},
        }
    }

    /// Admits each cached factor separately. Outer storage and the point copy
    /// are allocated through the policy but excluded from its table cap.
    /// An empty point produces a finished state with unit tables.
    #[track_caller]
    pub fn from_point_in<P: PointLike<Coord = F> + ?Sized, A: EqAlloc>(
        point: &P,
        binding_order: BindingOrder,
        scaling_factor: Option<F>,
        policy: A,
    ) -> Result<Self, A::Error> {
        let at = Location::caller();
        let n = point.num_vars();
        let _ = domain_len(n, policy)?;
        let (out_point, in_point) = Self::table_points(point.coords(), binding_order);
        check_layout::<F, _>(n, policy)?;
        let out_elements = EqPolynomial::<F>::cached_elements(out_point.len(), policy)?;
        let in_elements = EqPolynomial::<F>::cached_elements(in_point.len(), policy)?;
        policy.admit::<F>(out_elements, at)?;
        policy.admit::<F>(in_elements, at)?;
        let mut canonical_point = policy.storage(n)?;
        if P::ENDIANNESS == HIGH_TO_LOW {
            canonical_point.extend_from_slice(point.coords());
        } else {
            canonical_point.extend((0..n).map(|k| point.msb(k)));
        }
        let (out_point, in_point) = Self::table_points(&canonical_point, binding_order);
        let allocate = |part: &[F]| match binding_order {
            BindingOrder::LowToHigh => {
                let _span = tracing::info_span!("EqPolynomial::evals_cached_msb").entered();
                EqPolynomial::<F>::allocate_cached::<_, _, true>(part, None, policy)
            }
            BindingOrder::HighToLow => {
                EqPolynomial::<F>::allocate_cached::<_, _, false>(part, None, policy)
            }
        };
        #[cfg(feature = "parallel")]
        let (e_out_vec, e_in_vec) = rayon::join(|| allocate(out_point), || allocate(in_point));
        #[cfg(not(feature = "parallel"))]
        let (e_out_vec, e_in_vec) = (allocate(out_point), allocate(in_point));
        Ok(Self {
            current_index: match binding_order {
                BindingOrder::LowToHigh => n,
                BindingOrder::HighToLow => 0,
            },
            current_scalar: scaling_factor.unwrap_or(F::one()),
            point: canonical_point,
            e_in_vec: e_in_vec?,
            e_out_vec: e_out_vec?,
            binding_order,
        })
    }

    #[inline(always)]
    fn table_points(point: &[F], binding_order: BindingOrder) -> (&[F], &[F]) {
        if point.is_empty() {
            return (point, point);
        }
        let split = point.len() / 2;
        match binding_order {
            BindingOrder::LowToHigh => {
                let head = &point[..point.len() - 1];
                head.split_at(split.min(head.len()))
            }
            BindingOrder::HighToLow => {
                let tail = &point[1..];
                let (inner, outer) = tail.split_at(split.min(tail.len()));
                (outer, inner)
            }
        }
    }

    /// Bytes [`Self::new`] requests for a nonempty `vars`-variable point bound
    /// [`BindingOrder::LowToHigh`], or `None` past `usize::MAX`: the copied
    /// point and the [`EqPolynomial::evals_cached_msb`] tables over the point
    /// without its last variable, split at `vars / 2` into `2P - 1` out entries
    /// for `P = 2^⌊vars/2⌋` and `H - 1` in entries for `H = 2^⌈vars/2⌉`, with
    /// their `vars + 1` `Vec` headers. The empty point's layout differs.
    /// Excludes allocator slack.
    pub fn low_to_high_bytes(vars: usize) -> Option<usize> {
        let pow2 = |bits: usize| 1usize.checked_shl(u32::try_from(bits).ok()?);
        let out = pow2(vars / 2)?;
        let inner = pow2(vars.div_ceil(2))?;
        vars.checked_add(out.checked_mul(2)?)?
            .checked_add(inner)?
            .checked_sub(2)?
            .checked_mul(size_of::<F>())?
            .checked_add(vars.checked_add(1)?.checked_mul(size_of::<Vec<F>>())?)
    }

    pub fn current_scalar(&self) -> F {
        self.current_scalar
    }

    /// Number of variables still awaiting a challenge; zero denotes a finished state.
    pub fn remaining_rounds(&self) -> usize {
        match self.binding_order {
            BindingOrder::LowToHigh => self.current_index,
            BindingOrder::HighToLow => self.point.len() - self.current_index,
        }
    }

    /// Borrows the active round; returns `None` exactly when no variables remain.
    pub fn round(&self) -> Option<GruenRound<'_, F>> {
        let coordinate = *self.point.get(self.coordinate_index())?;
        Some(GruenRound {
            state: self,
            coordinate,
        })
    }

    /// Exclusively borrows the active round until its consuming bind.
    pub fn round_mut(&mut self) -> Option<GruenRoundMut<'_, F>> {
        let coordinate = *self.point.get(self.coordinate_index())?;
        Some(GruenRoundMut {
            state: self,
            coordinate,
        })
    }

    fn coordinate_index(&self) -> usize {
        match self.binding_order {
            BindingOrder::LowToHigh => self.current_index.wrapping_sub(1),
            BindingOrder::HighToLow => self.current_index,
        }
    }

    /// Tables `(inner, outer)` for the round after `binds` more challenges.
    /// Returns `None` when `binds >= remaining_rounds()`, including a finished state.
    pub fn remaining_tables_after(&self, binds: usize) -> Option<(&[F], &[F])> {
        if binds >= self.remaining_rounds() {
            return None;
        }
        let inner = self.e_in_vec.len() - 1;
        let outer = self.e_out_vec.len() - 1;
        let inner_binds = binds.min(inner);
        Some((
            &self.e_in_vec[inner - inner_binds],
            &self.e_out_vec[outer - (binds - inner_binds)],
        ))
    }

    pub fn e_in_current(&self) -> &[F] {
        &self.e_in_vec[self.e_in_vec.len() - 1]
    }

    pub fn e_out_current(&self) -> &[F] {
        &self.e_out_vec[self.e_out_vec.len() - 1]
    }

    pub fn e_in_current_len(&self) -> usize {
        self.e_in_current().len()
    }

    pub fn e_out_current_len(&self) -> usize {
        self.e_out_current().len()
    }

    pub fn e_out_in_for_window(&self, window_size: usize) -> (&[F], &[F]) {
        assert!(
            matches!(self.binding_order, BindingOrder::LowToHigh),
            "streaming split-eq windows are only defined for low-to-high"
        );

        let window_size = core::cmp::min(window_size, self.current_index);
        let head_len = self.current_index.saturating_sub(window_size);
        let split = self.point.len() / 2;

        let head_out_bits = core::cmp::min(head_len, split);
        let head_in_bits = head_len.saturating_sub(head_out_bits);

        debug_assert_eq!(head_out_bits + head_in_bits, head_len);
        debug_assert!(head_out_bits < self.e_out_vec.len());
        debug_assert!(head_in_bits < self.e_in_vec.len());

        (&self.e_out_vec[head_out_bits], &self.e_in_vec[head_in_bits])
    }

    pub fn e_active_for_window(&self, window_size: usize) -> Vec<F> {
        assert!(
            matches!(self.binding_order, BindingOrder::LowToHigh),
            "streaming split-eq windows are only defined for low-to-high"
        );

        if window_size <= 1 {
            return vec![F::one()];
        }

        let num_unbound = self.current_index;
        if window_size > num_unbound {
            return vec![F::one()];
        }

        let remaining_point = &self.point[..num_unbound];
        let window_start = remaining_point.len() - window_size;
        let (_, window_point) = remaining_point.split_at(window_start);
        let (active_point, _) = window_point.split_at(window_size - 1);
        EqPolynomial::<F>::evals(active_point, None)
    }

    pub fn merge(&self) -> Polynomial<F> {
        let evals = match self.binding_order {
            BindingOrder::LowToHigh => EqPolynomial::<F>::evals(
                &self.point[..self.current_index],
                Some(self.current_scalar),
            ),
            BindingOrder::HighToLow => EqPolynomial::<F>::evals(
                &self.point[self.current_index..],
                Some(self.current_scalar),
            ),
        };
        Polynomial::new(evals)
    }

    #[inline(always)]
    pub fn group_index(&self, x_out: usize, x_in: usize) -> usize {
        let in_bits = self.e_in_current_len().trailing_zeros() as usize;
        (x_out << in_bits) | x_in
    }

    /// Sum of the remaining-variable equality weights on `start..domain_len`.
    /// The current variable's linear factor and bound scalar are applied by
    /// the Gruen reconstruction, not included in this weight.
    /// Only defined for low-to-high binding, where group and polynomial order agree.
    pub fn suffix_weight(&self, start: usize) -> F {
        assert!(
            matches!(self.binding_order, BindingOrder::LowToHigh),
            "split-eq suffix weights are only defined for low-to-high"
        );
        EqPolynomial::suffix_sum(&self.point[..self.current_index.saturating_sub(1)], start)
    }

    #[inline]
    pub fn par_fold_out_in<
        OuterAcc: Send,
        InnerAcc: Send,
        MakeInner: Fn() -> InnerAcc + Sync + Send,
        InnerStep: Fn(&mut InnerAcc, usize, usize, F) + Sync + Send,
        OuterStep: Fn(usize, F, InnerAcc) -> OuterAcc + Sync + Send,
        Merge: Fn(OuterAcc, OuterAcc) -> OuterAcc + Sync + Send,
    >(
        &self,
        make_inner: MakeInner,
        inner_step: InnerStep,
        outer_step: OuterStep,
        merge: Merge,
    ) -> OuterAcc {
        self.par_fold_out_in_prefix(
            self.e_out_current_len() * self.e_in_current_len(),
            make_inner,
            inner_step,
            outer_step,
            merge,
        )
    }

    /// Fold the first `len` remaining-variable rows in group order. `len` counts
    /// polynomial rows only for [`BindingOrder::LowToHigh`]; in that order a caller
    /// can add a constant suffix's contribution using [`Self::suffix_weight`].
    pub fn par_fold_out_in_prefix<
        OuterAcc: Send,
        InnerAcc: Send,
        MakeInner: Fn() -> InnerAcc + Sync + Send,
        InnerStep: Fn(&mut InnerAcc, usize, usize, F) + Sync + Send,
        OuterStep: Fn(usize, F, InnerAcc) -> OuterAcc + Sync + Send,
        Merge: Fn(OuterAcc, OuterAcc) -> OuterAcc + Sync + Send,
    >(
        &self,
        len: usize,
        make_inner: MakeInner,
        inner_step: InnerStep,
        outer_step: OuterStep,
        merge: Merge,
    ) -> OuterAcc {
        let e_out = self.e_out_current();
        let e_in = self.e_in_current();
        assert!(len <= e_out.len() * e_in.len());
        let outer_len = len.div_ceil(e_in.len());
        #[cfg(feature = "parallel")]
        {
            (0..outer_len)
                .into_par_iter()
                .map(|x_out| {
                    let mut inner_acc = make_inner();
                    let inner_len = (len - x_out * e_in.len()).min(e_in.len());
                    for (x_in, &e_in) in e_in[..inner_len].iter().enumerate() {
                        let row = self.group_index(x_out, x_in);
                        inner_step(&mut inner_acc, row, x_in, e_in);
                    }
                    outer_step(x_out, e_out[x_out], inner_acc)
                })
                .reduce_with(merge)
                .unwrap_or_else(|| {
                    let inner_acc = make_inner();
                    outer_step(0, F::zero(), inner_acc)
                })
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut acc = None;
            for (x_out, &e_out_val) in e_out[..outer_len].iter().enumerate() {
                let mut inner_acc = make_inner();
                let inner_len = (len - x_out * e_in.len()).min(e_in.len());
                for (x_in, &e_in_val) in e_in[..inner_len].iter().enumerate() {
                    let row = self.group_index(x_out, x_in);
                    inner_step(&mut inner_acc, row, x_in, e_in_val);
                }
                let value = outer_step(x_out, e_out_val, inner_acc);
                acc = Some(match acc {
                    Some(acc) => merge(acc, value),
                    None => value,
                });
            }
            acc.unwrap_or_else(|| {
                let inner_acc = make_inner();
                outer_step(0, F::zero(), inner_acc)
            })
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test module asserts successful reconstruction and forbidden lazy endpoint evaluation"
)]
#[path = "tests.rs"]
mod tests;
