use std::{iter::zip, mem, sync::Arc};

use jolt_field::{Field, SignedScalarAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};
use rayon::prelude::*;

pub const MAX_INSTRUCTION_CHUNKS: usize = 32;
pub const MAX_BYTECODE_CHUNKS: usize = 6;
pub const MAX_RAM_CHUNKS: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RaFamilyLayout {
    pub k_chunk: usize,
    pub instruction_chunks: usize,
    pub bytecode_chunks: usize,
    pub ram_chunks: usize,
}

impl RaFamilyLayout {
    pub fn new(
        k_chunk: usize,
        instruction_chunks: usize,
        bytecode_chunks: usize,
        ram_chunks: usize,
    ) -> Self {
        assert!(
            k_chunk <= u8::MAX as usize + 1,
            "RA chunk size exceeds u8 chunk index range"
        );
        assert!(
            instruction_chunks <= MAX_INSTRUCTION_CHUNKS,
            "instruction RA chunks exceed fixed CPU bound"
        );
        assert!(
            bytecode_chunks <= MAX_BYTECODE_CHUNKS,
            "bytecode RA chunks exceed fixed CPU bound"
        );
        assert!(
            ram_chunks <= MAX_RAM_CHUNKS,
            "RAM RA chunks exceed fixed CPU bound"
        );
        Self {
            k_chunk,
            instruction_chunks,
            bytecode_chunks,
            ram_chunks,
        }
    }

    #[inline]
    pub const fn num_polys(self) -> usize {
        self.instruction_chunks + self.bytecode_chunks + self.ram_chunks
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RaCycleIndices {
    pub instruction: [u8; MAX_INSTRUCTION_CHUNKS],
    pub bytecode: [u8; MAX_BYTECODE_CHUNKS],
    pub ram: [Option<u8>; MAX_RAM_CHUNKS],
}

impl RaCycleIndices {
    #[inline]
    pub fn get_index(&self, poly_idx: usize, layout: RaFamilyLayout) -> Option<u8> {
        if poly_idx < layout.instruction_chunks {
            Some(self.instruction[poly_idx])
        } else if poly_idx < layout.instruction_chunks + layout.bytecode_chunks {
            Some(self.bytecode[poly_idx - layout.instruction_chunks])
        } else {
            self.ram[poly_idx - layout.instruction_chunks - layout.bytecode_chunks]
        }
    }
}

pub fn pushforward_indices<F: Field>(
    indices: &[RaCycleIndices],
    layout: RaFamilyLayout,
    r_cycle: &[F],
) -> Vec<Vec<F>> {
    assert_eq!(
        indices.len(),
        1usize << r_cycle.len(),
        "RA pushforward needs one cycle-index row per cycle hypercube point"
    );
    let num_polys = layout.num_polys();
    if num_polys == 0 {
        return Vec::new();
    }

    let lo_bits = r_cycle.len() / 2;
    let hi_bits = r_cycle.len() - lo_bits;
    let (r_hi, r_lo) = r_cycle.split_at(hi_bits);
    let (eq_hi, eq_lo) = rayon::join(
        || EqPolynomial::<F>::evals(r_hi, None),
        || EqPolynomial::<F>::evals(r_lo, None),
    );
    let in_len = eq_lo.len();
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = eq_hi.len().div_ceil(num_threads);

    eq_hi
        .par_chunks(chunk_size.max(1))
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut partial = (0..num_polys)
                .map(|_| jolt_poly::thread::unsafe_allocate_zero_vec::<F>(layout.k_chunk))
                .collect::<Vec<_>>();
            let mut local = (0..num_polys)
                .map(|_| {
                    vec![<F as jolt_field::WithSmallScalarAccumulator>::SmallScalarAccumulator::default(); layout.k_chunk]
                })
                .collect::<Vec<_>>();
            let mut touched = (0..num_polys)
                .map(|_| Vec::<usize>::with_capacity(layout.k_chunk))
                .collect::<Vec<_>>();
            let mut touched_flags = (0..num_polys)
                .map(|_| vec![false; layout.k_chunk])
                .collect::<Vec<_>>();

            let chunk_start = chunk_idx * chunk_size;
            for (local_idx, &eq_hi_eval) in chunk.iter().enumerate() {
                for poly_idx in 0..num_polys {
                    for index in touched[poly_idx].drain(..) {
                        local[poly_idx][index] = Default::default();
                        touched_flags[poly_idx][index] = false;
                    }
                }

                let cycle_base = (chunk_start + local_idx) * in_len;
                for (cycle_offset, &eq_lo_eval) in eq_lo.iter().enumerate() {
                    let row = indices[cycle_base + cycle_offset];
                    for poly_idx in 0..layout.instruction_chunks {
                    accumulate_pushforward_slot(
                        poly_idx,
                        usize::from(row.instruction[poly_idx]),
                        eq_lo_eval,
                            &mut local,
                            &mut touched,
                            &mut touched_flags,
                        );
                    }
                    for chunk in 0..layout.bytecode_chunks {
                        accumulate_pushforward_slot(
                            layout.instruction_chunks + chunk,
                            usize::from(row.bytecode[chunk]),
                            eq_lo_eval,
                            &mut local,
                            &mut touched,
                            &mut touched_flags,
                        );
                    }
                    for chunk in 0..layout.ram_chunks {
                        if let Some(index) = row.ram[chunk] {
                            accumulate_pushforward_slot(
                                layout.instruction_chunks + layout.bytecode_chunks + chunk,
                                usize::from(index),
                                eq_lo_eval,
                                &mut local,
                                &mut touched,
                                &mut touched_flags,
                            );
                        }
                    }
                }

                for poly_idx in 0..num_polys {
                    for &index in &touched[poly_idx] {
                        partial[poly_idx][index] += eq_hi_eval * local[poly_idx][index].reduce();
                    }
                }
            }
            partial
        })
        .reduce(
            || {
                (0..num_polys)
                    .map(|_| jolt_poly::thread::unsafe_allocate_zero_vec::<F>(layout.k_chunk))
                    .collect::<Vec<_>>()
            },
            |mut left, right| {
                for (left_poly, right_poly) in left.iter_mut().zip(right.iter()) {
                    left_poly
                        .par_iter_mut()
                        .zip(right_poly.par_iter())
                        .for_each(|(left_value, right_value)| *left_value += *right_value);
                }
                left
            },
        )
}

fn accumulate_pushforward_slot<F: Field>(
    poly_idx: usize,
    index: usize,
    value: F,
    local: &mut [Vec<<F as jolt_field::WithSmallScalarAccumulator>::SmallScalarAccumulator>],
    touched: &mut [Vec<usize>],
    touched_flags: &mut [Vec<bool>],
) {
    if !touched_flags[poly_idx][index] {
        touched_flags[poly_idx][index] = true;
        touched[poly_idx].push(index);
    }
    local[poly_idx][index].add(value);
}

#[derive(Clone, Debug, PartialEq)]
pub enum RaPolynomial<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> {
    Round1(RaPolynomialRound1<I, F>),
    Round2(RaPolynomialRound2<I, F>),
    Round3(RaPolynomialRound3<I, F>),
    RoundN(Polynomial<F>),
}

impl<I, F> RaPolynomial<I, F>
where
    I: Into<usize> + Copy + Default + Send + Sync + 'static,
    F: Field,
{
    pub fn new(indices: Arc<Vec<Option<I>>>, eq_evals: Vec<F>) -> Self {
        Self::Round1(RaPolynomialRound1 { eq_evals, indices })
    }

    #[inline]
    pub fn get_bound_coeff(&self, j: usize) -> F {
        match self {
            Self::Round1(poly) => poly.get_bound_coeff(j),
            Self::Round2(poly) => poly.get_bound_coeff(j),
            Self::Round3(poly) => poly.get_bound_coeff(j),
            Self::RoundN(poly) => poly.evaluations()[j],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Round1(poly) => poly.len(),
            Self::Round2(poly) => poly.len(),
            Self::Round3(poly) => poly.len(),
            Self::RoundN(poly) => poly.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        match self {
            Self::Round1(poly) => *self = Self::Round2(mem::take(poly).bind(r, order)),
            Self::Round2(poly) => *self = Self::Round3(mem::take(poly).bind(r, order)),
            Self::Round3(poly) => *self = Self::RoundN(mem::take(poly).bind(r, order)),
            Self::RoundN(poly) => poly.bind_with_order(r, order),
        }
    }

    #[inline]
    pub fn final_sumcheck_claim(&self) -> Option<F> {
        match self {
            Self::RoundN(poly) if poly.len() == 1 => Some(poly.evaluations()[0]),
            _ => None,
        }
    }

    #[inline]
    pub fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = vec![F::zero(); degree];
        match order {
            BindingOrder::HighToLow => {
                evals[0] = self.get_bound_coeff(index);
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.get_bound_coeff(index + self.len() / 2);
                let slope = eval - evals[0];
                for output in evals.iter_mut().skip(1) {
                    eval += slope;
                    *output = eval;
                }
            }
            BindingOrder::LowToHigh => {
                evals[0] = self.get_bound_coeff(2 * index);
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.get_bound_coeff(2 * index + 1);
                let slope = eval - evals[0];
                for output in evals.iter_mut().skip(1) {
                    eval += slope;
                    *output = eval;
                }
            }
        }
        evals
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SharedRaPolynomials<F: Field> {
    Round1(SharedRaRound1<F>),
    Round2(SharedRaRound2<F>),
    Round3(SharedRaRound3<F>),
    RoundN(Vec<Polynomial<F>>),
}

impl<F: Field> SharedRaPolynomials<F> {
    pub fn new(tables: Vec<Vec<F>>, indices: Vec<RaCycleIndices>, layout: RaFamilyLayout) -> Self {
        assert_eq!(
            tables.len(),
            layout.num_polys(),
            "shared RA table count must match family layout"
        );
        debug_assert!(tables.iter().all(|table| table.len() == layout.k_chunk));
        Self::Round1(SharedRaRound1 {
            tables,
            indices,
            num_polys: layout.num_polys(),
            layout,
        })
    }

    #[inline]
    pub fn num_polys(&self) -> usize {
        match self {
            Self::Round1(round) => round.num_polys,
            Self::Round2(round) => round.num_polys,
            Self::Round3(round) => round.num_polys,
            Self::RoundN(polys) => polys.len(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Round1(round) => round.indices.len(),
            Self::Round2(round) => round.indices.len() / 2,
            Self::Round3(round) => round.indices.len() / 4,
            Self::RoundN(polys) => polys.first().map_or(0, Polynomial::len),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        match self {
            Self::Round1(round) => round.get_bound_coeff(poly_idx, j),
            Self::Round2(round) => round.get_bound_coeff(poly_idx, j),
            Self::Round3(round) => round.get_bound_coeff(poly_idx, j),
            Self::RoundN(polys) => polys[poly_idx].evaluations()[j],
        }
    }

    pub fn bind(self, r: F, order: BindingOrder) -> Self {
        match self {
            Self::Round1(round) => Self::Round2(round.bind(r, order)),
            Self::Round2(round) => Self::Round3(round.bind(r, order)),
            Self::Round3(round) => Self::RoundN(round.bind(r, order)),
            Self::RoundN(mut polys) => {
                polys
                    .par_iter_mut()
                    .for_each(|poly| poly.bind_with_order(r, order));
                Self::RoundN(polys)
            }
        }
    }

    pub fn bind_in_place(&mut self, r: F, order: BindingOrder) {
        match self {
            Self::Round1(round) => *self = Self::Round2(mem::take(round).bind(r, order)),
            Self::Round2(round) => *self = Self::Round3(mem::take(round).bind(r, order)),
            Self::Round3(round) => *self = Self::RoundN(mem::take(round).bind(r, order)),
            Self::RoundN(polys) => {
                polys
                    .par_iter_mut()
                    .for_each(|poly| poly.bind_with_order(r, order));
            }
        }
    }

    pub fn final_sumcheck_claim(&self, poly_idx: usize) -> Option<F> {
        match self {
            Self::RoundN(polys) if polys[poly_idx].len() == 1 => {
                Some(polys[poly_idx].evaluations()[0])
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SharedRaRound1<F: Field> {
    tables: Vec<Vec<F>>,
    indices: Vec<RaCycleIndices>,
    num_polys: usize,
    layout: RaFamilyLayout,
}

impl<F: Field> SharedRaRound1<F> {
    #[inline]
    fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        self.indices[j]
            .get_index(poly_idx, self.layout)
            .map_or(F::zero(), |k| self.tables[poly_idx][usize::from(k)])
    }

    fn bind(self, r0: F, order: BindingOrder) -> SharedRaRound2<F> {
        let eq_0_r0 = EqPolynomial::<F>::mle(&[F::zero()], &[r0]);
        let eq_1_r0 = EqPolynomial::<F>::mle(&[F::one()], &[r0]);
        let (tables_0, tables_1) = rayon::join(
            || {
                self.tables
                    .par_iter()
                    .map(|table| {
                        table
                            .iter()
                            .map(|value| eq_0_r0 * *value)
                            .collect::<Vec<F>>()
                    })
                    .collect::<Vec<Vec<F>>>()
            },
            || {
                self.tables
                    .par_iter()
                    .map(|table| {
                        table
                            .iter()
                            .map(|value| eq_1_r0 * *value)
                            .collect::<Vec<F>>()
                    })
                    .collect::<Vec<Vec<F>>>()
            },
        );
        SharedRaRound2 {
            tables_0,
            tables_1,
            indices: self.indices,
            num_polys: self.num_polys,
            layout: self.layout,
            binding_order: order,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SharedRaRound2<F: Field> {
    tables_0: Vec<Vec<F>>,
    tables_1: Vec<Vec<F>>,
    indices: Vec<RaCycleIndices>,
    num_polys: usize,
    layout: RaFamilyLayout,
    binding_order: BindingOrder,
}

impl<F: Field> SharedRaRound2<F> {
    #[inline]
    fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let mid = self.indices.len() / 2;
                let h_0 = self.indices[j]
                    .get_index(poly_idx, self.layout)
                    .map_or(F::zero(), |k| self.tables_0[poly_idx][usize::from(k)]);
                let h_1 = self.indices[mid + j]
                    .get_index(poly_idx, self.layout)
                    .map_or(F::zero(), |k| self.tables_1[poly_idx][usize::from(k)]);
                h_0 + h_1
            }
            BindingOrder::LowToHigh => {
                let h_0 = self.indices[2 * j]
                    .get_index(poly_idx, self.layout)
                    .map_or(F::zero(), |k| self.tables_0[poly_idx][usize::from(k)]);
                let h_1 = self.indices[2 * j + 1]
                    .get_index(poly_idx, self.layout)
                    .map_or(F::zero(), |k| self.tables_1[poly_idx][usize::from(k)]);
                h_0 + h_1
            }
        }
    }

    fn bind(self, r1: F, order: BindingOrder) -> SharedRaRound3<F> {
        assert_eq!(order, self.binding_order);
        let eq_0_r1 = EqPolynomial::<F>::mle(&[F::zero()], &[r1]);
        let eq_1_r1 = EqPolynomial::<F>::mle(&[F::one()], &[r1]);

        let mut tables_00 = self.tables_0.clone();
        let mut tables_01 = self.tables_0;
        let mut tables_10 = self.tables_1.clone();
        let mut tables_11 = self.tables_1;

        rayon::join(
            || {
                rayon::join(
                    || {
                        tables_00
                            .par_iter_mut()
                            .for_each(|table| table.par_iter_mut().for_each(|f| *f *= eq_0_r1));
                    },
                    || {
                        tables_01
                            .par_iter_mut()
                            .for_each(|table| table.par_iter_mut().for_each(|f| *f *= eq_1_r1));
                    },
                );
            },
            || {
                rayon::join(
                    || {
                        tables_10
                            .par_iter_mut()
                            .for_each(|table| table.par_iter_mut().for_each(|f| *f *= eq_0_r1));
                    },
                    || {
                        tables_11
                            .par_iter_mut()
                            .for_each(|table| table.par_iter_mut().for_each(|f| *f *= eq_1_r1));
                    },
                );
            },
        );

        SharedRaRound3 {
            tables_00,
            tables_01,
            tables_10,
            tables_11,
            indices: self.indices,
            num_polys: self.num_polys,
            layout: self.layout,
            binding_order: order,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SharedRaRound3<F: Field> {
    tables_00: Vec<Vec<F>>,
    tables_01: Vec<Vec<F>>,
    tables_10: Vec<Vec<F>>,
    tables_11: Vec<Vec<F>>,
    indices: Vec<RaCycleIndices>,
    num_polys: usize,
    layout: RaFamilyLayout,
    binding_order: BindingOrder,
}

impl<F: Field> SharedRaRound3<F> {
    #[inline]
    fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let quarter = self.indices.len() / 4;
                let h_00 = self.indices[j]
                    .get_index(poly_idx, self.layout)
                    .map_or(F::zero(), |k| self.tables_00[poly_idx][usize::from(k)]);
                let h_01 = self.indices[quarter + j]
                    .get_index(poly_idx, self.layout)
                    .map_or(F::zero(), |k| self.tables_01[poly_idx][usize::from(k)]);
                let h_10 = self.indices[2 * quarter + j]
                    .get_index(poly_idx, self.layout)
                    .map_or(F::zero(), |k| self.tables_10[poly_idx][usize::from(k)]);
                let h_11 = self.indices[3 * quarter + j]
                    .get_index(poly_idx, self.layout)
                    .map_or(F::zero(), |k| self.tables_11[poly_idx][usize::from(k)]);
                h_00 + h_01 + h_10 + h_11
            }
            BindingOrder::LowToHigh => {
                let h_00 = self.indices[4 * j]
                    .get_index(poly_idx, self.layout)
                    .map_or(F::zero(), |k| self.tables_00[poly_idx][usize::from(k)]);
                let h_10 = self.indices[4 * j + 1]
                    .get_index(poly_idx, self.layout)
                    .map_or(F::zero(), |k| self.tables_10[poly_idx][usize::from(k)]);
                let h_01 = self.indices[4 * j + 2]
                    .get_index(poly_idx, self.layout)
                    .map_or(F::zero(), |k| self.tables_01[poly_idx][usize::from(k)]);
                let h_11 = self.indices[4 * j + 3]
                    .get_index(poly_idx, self.layout)
                    .map_or(F::zero(), |k| self.tables_11[poly_idx][usize::from(k)]);
                h_00 + h_10 + h_01 + h_11
            }
        }
    }

    fn bind(self, r2: F, order: BindingOrder) -> Vec<Polynomial<F>> {
        assert_eq!(order, self.binding_order);
        let eq_0_r2 = EqPolynomial::<F>::mle(&[F::zero()], &[r2]);
        let eq_1_r2 = EqPolynomial::<F>::mle(&[F::one()], &[r2]);

        let mut tables_000 = self.tables_00.clone();
        let mut tables_001 = self.tables_00;
        let mut tables_010 = self.tables_01.clone();
        let mut tables_011 = self.tables_01;
        let mut tables_100 = self.tables_10.clone();
        let mut tables_101 = self.tables_10;
        let mut tables_110 = self.tables_11.clone();
        let mut tables_111 = self.tables_11;

        rayon::join(
            || {
                [
                    &mut tables_000,
                    &mut tables_010,
                    &mut tables_100,
                    &mut tables_110,
                ]
                .into_par_iter()
                .for_each(|tables| {
                    tables
                        .par_iter_mut()
                        .for_each(|table| table.par_iter_mut().for_each(|f| *f *= eq_0_r2));
                });
            },
            || {
                [
                    &mut tables_001,
                    &mut tables_011,
                    &mut tables_101,
                    &mut tables_111,
                ]
                .into_par_iter()
                .for_each(|tables| {
                    tables
                        .par_iter_mut()
                        .for_each(|table| table.par_iter_mut().for_each(|f| *f *= eq_1_r2));
                });
            },
        );

        let low_to_high_table_groups = [
            &tables_000,
            &tables_100,
            &tables_010,
            &tables_110,
            &tables_001,
            &tables_101,
            &tables_011,
            &tables_111,
        ];
        let high_to_low_table_groups = [
            &tables_000,
            &tables_001,
            &tables_010,
            &tables_011,
            &tables_100,
            &tables_101,
            &tables_110,
            &tables_111,
        ];
        let indices = &self.indices;
        let layout = self.layout;
        let new_len = indices.len() / 8;

        (0..self.num_polys)
            .into_par_iter()
            .map(|poly_idx| {
                let coeffs = match order {
                    BindingOrder::LowToHigh => (0..new_len)
                        .into_par_iter()
                        .map(|j| {
                            (0..8)
                                .map(|offset| {
                                    indices[8 * j + offset].get_index(poly_idx, layout).map_or(
                                        F::zero(),
                                        |k| {
                                            low_to_high_table_groups[offset][poly_idx]
                                                [usize::from(k)]
                                        },
                                    )
                                })
                                .sum()
                        })
                        .collect::<Vec<F>>(),
                    BindingOrder::HighToLow => {
                        let eighth = indices.len() / 8;
                        (0..new_len)
                            .into_par_iter()
                            .map(|j| {
                                (0..8)
                                    .map(|segment| {
                                        indices[segment * eighth + j]
                                            .get_index(poly_idx, layout)
                                            .map_or(F::zero(), |k| {
                                                high_to_low_table_groups[segment][poly_idx]
                                                    [usize::from(k)]
                                            })
                                    })
                                    .sum()
                            })
                            .collect::<Vec<F>>()
                    }
                };
                Polynomial::from(coeffs)
            })
            .collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RaPolynomialRound1<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> {
    eq_evals: Vec<F>,
    indices: Arc<Vec<Option<I>>>,
}

impl<I, F> RaPolynomialRound1<I, F>
where
    I: Into<usize> + Copy + Default + Send + Sync + 'static,
    F: Field,
{
    #[inline]
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn bind(self, r0: F, binding_order: BindingOrder) -> RaPolynomialRound2<I, F> {
        let eq_0_r0 = EqPolynomial::<F>::mle(&[F::zero()], &[r0]);
        let eq_1_r0 = EqPolynomial::<F>::mle(&[F::one()], &[r0]);
        let eq_evals_0 = self
            .eq_evals
            .par_iter()
            .map(|value| eq_0_r0 * *value)
            .collect();
        let eq_evals_1 = self
            .eq_evals
            .par_iter()
            .map(|value| eq_1_r0 * *value)
            .collect();
        RaPolynomialRound2 {
            eq_evals_0,
            eq_evals_1,
            indices: self.indices,
            binding_order,
        }
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        self.indices[j].map_or(F::zero(), |i| self.eq_evals[i.into()])
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RaPolynomialRound2<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> {
    eq_evals_0: Vec<F>,
    eq_evals_1: Vec<F>,
    indices: Arc<Vec<Option<I>>>,
    binding_order: BindingOrder,
}

impl<I, F> RaPolynomialRound2<I, F>
where
    I: Into<usize> + Copy + Default + Send + Sync + 'static,
    F: Field,
{
    #[inline]
    fn len(&self) -> usize {
        self.indices.len() / 2
    }

    fn bind(self, r1: F, binding_order: BindingOrder) -> RaPolynomialRound3<I, F> {
        assert_eq!(binding_order, self.binding_order);
        let eq_0_r1 = EqPolynomial::<F>::mle(&[F::zero()], &[r1]);
        let eq_1_r1 = EqPolynomial::<F>::mle(&[F::one()], &[r1]);
        let mut eq_evals_00 = self.eq_evals_0.clone();
        let mut eq_evals_01 = self.eq_evals_0;
        let mut eq_evals_10 = self.eq_evals_1.clone();
        let mut eq_evals_11 = self.eq_evals_1;

        eq_evals_00
            .par_iter_mut()
            .for_each(|value| *value *= eq_0_r1);
        eq_evals_01
            .par_iter_mut()
            .for_each(|value| *value *= eq_1_r1);
        eq_evals_10
            .par_iter_mut()
            .for_each(|value| *value *= eq_0_r1);
        eq_evals_11
            .par_iter_mut()
            .for_each(|value| *value *= eq_1_r1);

        RaPolynomialRound3 {
            eq_evals_00,
            eq_evals_01,
            eq_evals_10,
            eq_evals_11,
            indices: self.indices,
            binding_order,
        }
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let mid = self.indices.len() / 2;
                let h_0 = self.indices[j].map_or(F::zero(), |i| self.eq_evals_0[i.into()]);
                let h_1 = self.indices[mid + j].map_or(F::zero(), |i| self.eq_evals_1[i.into()]);
                h_0 + h_1
            }
            BindingOrder::LowToHigh => {
                let h_0 = self.indices[2 * j].map_or(F::zero(), |i| self.eq_evals_0[i.into()]);
                let h_1 = self.indices[2 * j + 1].map_or(F::zero(), |i| self.eq_evals_1[i.into()]);
                h_0 + h_1
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RaPolynomialRound3<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> {
    eq_evals_00: Vec<F>,
    eq_evals_01: Vec<F>,
    eq_evals_10: Vec<F>,
    eq_evals_11: Vec<F>,
    indices: Arc<Vec<Option<I>>>,
    binding_order: BindingOrder,
}

impl<I, F> RaPolynomialRound3<I, F>
where
    I: Into<usize> + Copy + Default + Send + Sync + 'static,
    F: Field,
{
    #[inline]
    fn len(&self) -> usize {
        self.indices.len() / 4
    }

    fn bind(self, r2: F, binding_order: BindingOrder) -> Polynomial<F> {
        assert_eq!(binding_order, self.binding_order);
        let eq_0_r2 = EqPolynomial::<F>::mle(&[F::zero()], &[r2]);
        let eq_1_r2 = EqPolynomial::<F>::mle(&[F::one()], &[r2]);
        let mut eq_evals_000 = self.eq_evals_00.clone();
        let mut eq_evals_001 = self.eq_evals_00;
        let mut eq_evals_010 = self.eq_evals_01.clone();
        let mut eq_evals_011 = self.eq_evals_01;
        let mut eq_evals_100 = self.eq_evals_10.clone();
        let mut eq_evals_101 = self.eq_evals_10;
        let mut eq_evals_110 = self.eq_evals_11.clone();
        let mut eq_evals_111 = self.eq_evals_11;

        eq_evals_000
            .par_iter_mut()
            .for_each(|value| *value *= eq_0_r2);
        eq_evals_010
            .par_iter_mut()
            .for_each(|value| *value *= eq_0_r2);
        eq_evals_100
            .par_iter_mut()
            .for_each(|value| *value *= eq_0_r2);
        eq_evals_110
            .par_iter_mut()
            .for_each(|value| *value *= eq_0_r2);
        eq_evals_001
            .par_iter_mut()
            .for_each(|value| *value *= eq_1_r2);
        eq_evals_011
            .par_iter_mut()
            .for_each(|value| *value *= eq_1_r2);
        eq_evals_101
            .par_iter_mut()
            .for_each(|value| *value *= eq_1_r2);
        eq_evals_111
            .par_iter_mut()
            .for_each(|value| *value *= eq_1_r2);

        let indices = &self.indices;
        let len = indices.len() / 8;
        let mut coeffs = jolt_poly::thread::unsafe_allocate_zero_vec(len);
        let chunk_size = 1 << 16;

        match self.binding_order {
            BindingOrder::HighToLow => {
                coeffs.par_chunks_mut(chunk_size).enumerate().for_each(
                    |(chunk_index, evals_chunk)| {
                        for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                            let h_000 = indices[j].map_or(F::zero(), |i| eq_evals_000[i.into()]);
                            let h_001 =
                                indices[j + len].map_or(F::zero(), |i| eq_evals_001[i.into()]);
                            let h_010 =
                                indices[j + len * 2].map_or(F::zero(), |i| eq_evals_010[i.into()]);
                            let h_011 =
                                indices[j + len * 3].map_or(F::zero(), |i| eq_evals_011[i.into()]);
                            let h_100 =
                                indices[j + len * 4].map_or(F::zero(), |i| eq_evals_100[i.into()]);
                            let h_101 =
                                indices[j + len * 5].map_or(F::zero(), |i| eq_evals_101[i.into()]);
                            let h_110 =
                                indices[j + len * 6].map_or(F::zero(), |i| eq_evals_110[i.into()]);
                            let h_111 =
                                indices[j + len * 7].map_or(F::zero(), |i| eq_evals_111[i.into()]);
                            *eval = h_000 + h_010 + h_100 + h_110 + h_001 + h_011 + h_101 + h_111;
                        }
                    },
                );
            }
            BindingOrder::LowToHigh => {
                coeffs.par_chunks_mut(chunk_size).enumerate().for_each(
                    |(chunk_index, evals_chunk)| {
                        for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                            let h_000 =
                                indices[8 * j].map_or(F::zero(), |i| eq_evals_000[i.into()]);
                            let h_100 =
                                indices[8 * j + 1].map_or(F::zero(), |i| eq_evals_100[i.into()]);
                            let h_010 =
                                indices[8 * j + 2].map_or(F::zero(), |i| eq_evals_010[i.into()]);
                            let h_110 =
                                indices[8 * j + 3].map_or(F::zero(), |i| eq_evals_110[i.into()]);
                            let h_001 =
                                indices[8 * j + 4].map_or(F::zero(), |i| eq_evals_001[i.into()]);
                            let h_101 =
                                indices[8 * j + 5].map_or(F::zero(), |i| eq_evals_101[i.into()]);
                            let h_011 =
                                indices[8 * j + 6].map_or(F::zero(), |i| eq_evals_011[i.into()]);
                            let h_111 =
                                indices[8 * j + 7].map_or(F::zero(), |i| eq_evals_111[i.into()]);
                            *eval = h_000 + h_010 + h_100 + h_110 + h_001 + h_011 + h_101 + h_111;
                        }
                    },
                );
            }
        }

        Polynomial::from(coeffs)
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let quarter = self.indices.len() / 4;
                let h_00 = self.indices[j].map_or(F::zero(), |i| self.eq_evals_00[i.into()]);
                let h_01 =
                    self.indices[quarter + j].map_or(F::zero(), |i| self.eq_evals_01[i.into()]);
                let h_10 =
                    self.indices[quarter * 2 + j].map_or(F::zero(), |i| self.eq_evals_10[i.into()]);
                let h_11 =
                    self.indices[quarter * 3 + j].map_or(F::zero(), |i| self.eq_evals_11[i.into()]);
                h_00 + h_10 + h_01 + h_11
            }
            BindingOrder::LowToHigh => {
                let h_00 = self.indices[4 * j].map_or(F::zero(), |i| self.eq_evals_00[i.into()]);
                let h_10 =
                    self.indices[4 * j + 1].map_or(F::zero(), |i| self.eq_evals_10[i.into()]);
                let h_01 =
                    self.indices[4 * j + 2].map_or(F::zero(), |i| self.eq_evals_01[i.into()]);
                let h_11 =
                    self.indices[4 * j + 3].map_or(F::zero(), |i| self.eq_evals_11[i.into()]);
                h_00 + h_10 + h_01 + h_11
            }
        }
    }
}
