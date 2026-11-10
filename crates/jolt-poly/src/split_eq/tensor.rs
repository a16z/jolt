use jolt_field::{Accumulator, Field, WithAccumulator};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{Endianness, EqPolynomial, PointLike, PointRef, HIGH_TO_LOW};

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEqTable<F: Field> {
    e_out: Vec<F>,
    e_in: Vec<F>,
    in_bits: usize,
}

impl<F: Field> TensorEqTable<F> {
    pub fn new(point: &[F]) -> Self {
        Self::new_with_scaling(point, F::one())
    }

    pub fn new_with_scaling(point: &[F], scale: F) -> Self {
        Self::new_with_scaling_and_split(point, scale, point.len() / 2)
    }

    /// Factors `scale * eq(point, x)` with the first `split` coordinates in the
    /// outer table. `split` must be at most `point.len()`.
    pub fn new_with_scaling_and_split(point: &[F], scale: F, split: usize) -> Self {
        Self::from_point(point, scale, split)
    }

    /// Factors `scale * eq(point, x)` with `outer_vars` most-significant
    /// variables outside. Index layout is `(x_out << in_bits) | x_in`.
    pub fn from_point<P: PointLike<Coord = F> + ?Sized>(
        point: &P,
        scale: F,
        outer_vars: usize,
    ) -> Self {
        let out_point = point.msb_range(0..outer_vars);
        let in_point = point.msb_range(outer_vars..point.num_vars());
        if P::ENDIANNESS == HIGH_TO_LOW {
            Self::from_parts(
                PointRef::high_to_low(out_point.coords()),
                PointRef::high_to_low(in_point.coords()),
                scale,
            )
        } else {
            Self::from_parts(
                PointRef::low_to_high(out_point.coords()),
                PointRef::low_to_high(in_point.coords()),
                scale,
            )
        }
    }

    fn from_parts<const E: Endianness>(
        out_point: PointRef<'_, E, F>,
        in_point: PointRef<'_, E, F>,
        scale: F,
    ) -> Self {
        #[cfg(feature = "parallel")]
        let (e_out, e_in) = rayon::join(
            || EqPolynomial::<F>::evals(&out_point, Some(scale)),
            || EqPolynomial::<F>::evals(&in_point, None),
        );
        #[cfg(not(feature = "parallel"))]
        let (e_out, e_in) = (
            EqPolynomial::<F>::evals(&out_point, Some(scale)),
            EqPolynomial::<F>::evals(&in_point, None),
        );
        Self {
            e_out,
            e_in,
            in_bits: in_point.num_vars(),
        }
    }

    pub fn len(&self) -> usize {
        self.e_out.len() * self.e_in.len()
    }

    pub fn is_empty(&self) -> bool {
        self.e_out.is_empty() || self.e_in.is_empty()
    }

    pub fn e_out(&self) -> &[F] {
        &self.e_out
    }

    pub fn e_in(&self) -> &[F] {
        &self.e_in
    }

    fn factors_at_index(&self, index: usize) -> (F, F) {
        let x_out = index >> self.in_bits;
        let x_in = index & ((1usize << self.in_bits) - 1);
        (self.e_out[x_out], self.e_in[x_in])
    }

    pub fn evaluate_index(&self, index: usize) -> F {
        let (outer, inner) = self.factors_at_index(index);
        outer * inner
    }

    /// Sum the tables at one Boolean index with a single accumulator reduction.
    pub fn sum_at_index(tables: &[Self], index: usize) -> F
    where
        F: WithAccumulator,
    {
        let mut sum = F::Accumulator::default();
        for table in tables {
            let (outer, inner) = table.factors_at_index(index);
            sum.fmadd(outer, inner);
        }
        sum.reduce()
    }

    pub fn evaluate_slices(&self, values: &[&[F]]) -> Vec<F> {
        if values.is_empty() {
            return Vec::new();
        }
        debug_assert!(
            values.iter().all(|values| values.len() == self.len()),
            "TensorEqTable::evaluate_slices length mismatch"
        );

        self.par_fold_out_in(
            || vec![F::zero(); values.len()],
            |inner, row, _x_in, e_in| {
                if e_in.is_zero() {
                    return;
                }
                for (accumulator, values) in inner.iter_mut().zip(values) {
                    *accumulator += e_in * values[row];
                }
            },
            |_x_out, e_out, mut inner| {
                if e_out.is_zero() {
                    inner.fill(F::zero());
                } else {
                    for value in &mut inner {
                        *value *= e_out;
                    }
                }
                inner
            },
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            },
        )
    }

    #[inline(always)]
    pub fn group_index(&self, x_out: usize, x_in: usize) -> usize {
        (x_out << self.in_bits) | x_in
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
        self.par_fold_out_in_prefix(self.len(), make_inner, inner_step, outer_step, merge)
    }

    /// Fold the first `len` rows; callers account for any omitted suffix separately.
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
        assert!(len <= self.len());
        let outer_len = len.div_ceil(self.e_in.len());
        #[cfg(feature = "parallel")]
        {
            (0..outer_len)
                .into_par_iter()
                .map(|x_out| {
                    let mut inner_acc = make_inner();
                    for (x_in, &e_in) in self
                        .e_in
                        .iter()
                        .take(len - x_out * self.e_in.len())
                        .enumerate()
                    {
                        let row = self.group_index(x_out, x_in);
                        inner_step(&mut inner_acc, row, x_in, e_in);
                    }
                    outer_step(x_out, self.e_out[x_out], inner_acc)
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
            for (x_out, &e_out) in self.e_out.iter().take(outer_len).enumerate() {
                let mut inner_acc = make_inner();
                for (x_in, &e_in) in self
                    .e_in
                    .iter()
                    .take(len - x_out * self.e_in.len())
                    .enumerate()
                {
                    let row = self.group_index(x_out, x_in);
                    inner_step(&mut inner_acc, row, x_in, e_in);
                }
                let value = outer_step(x_out, e_out, inner_acc);
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
