use jolt_claims::protocols::jolt::geometry::ra::{JoltRaPolynomial, JoltRaPolynomialLayout};
use jolt_field::{Accumulator, JoltField};
use jolt_witness::witnesses::RaChunkSelector;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::instruction_read_raf::InstructionCycleRow;
use super::support::eq_table;
use crate::KernelError;

pub(crate) enum ColumnSelector {
    Instruction(RaChunkSelector),
    Bytecode(RaChunkSelector),
    Ram(RaChunkSelector),
}

impl ColumnSelector {
    pub(crate) fn for_layout<F: JoltField>(
        layout: JoltRaPolynomialLayout,
        chunk_bits: usize,
    ) -> Result<Vec<Self>, KernelError<F>> {
        layout
            .polynomials()
            .map(|polynomial| {
                Ok(match polynomial {
                    JoltRaPolynomial::Instruction(index) => Self::Instruction(
                        RaChunkSelector::new(index, layout.instruction(), chunk_bits)?,
                    ),
                    JoltRaPolynomial::Bytecode(index) => {
                        Self::Bytecode(RaChunkSelector::new(index, layout.bytecode(), chunk_bits)?)
                    }
                    JoltRaPolynomial::Ram(index) => {
                        Self::Ram(RaChunkSelector::new(index, layout.ram(), chunk_bits)?)
                    }
                })
            })
            .collect()
    }

    #[inline]
    pub(crate) fn index(&self, row: &InstructionCycleRow) -> Option<usize> {
        match self {
            Self::Instruction(selector) => Some(selector.chunk_u128(row.lookup_index)),
            Self::Bytecode(selector) => {
                Some(selector.chunk_usize(row.mapped_pc().unwrap_or_default()))
            }
            Self::Ram(selector) => row
                .remapped_ram_address()
                .map(|address| selector.chunk_usize(address as usize)),
        }
    }
}

pub(crate) fn parallel_outer_bits(log_t: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        // Four blocks per worker preserve work stealing while maximizing the
        // number of inner additions amortized by each outer multiplication.
        rayon::current_num_threads()
            .saturating_mul(4)
            .next_power_of_two()
            .ilog2()
            .min(log_t as u32) as usize
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = log_t;
        0
    }
}

/// Split-eq one-hot pushforward shared by booleanity and Hamming reduction.
/// Each inner block accumulates only additions; the outer weight is applied
/// after bucket reduction, preserving the exact field sum.
pub(crate) fn split_eq_pushforwards<F: JoltField>(
    rows: &[InstructionCycleRow],
    selectors: &[ColumnSelector],
    k_chunk: usize,
    point: &[F],
    outer_bits: usize,
) -> Vec<Vec<F>> {
    debug_assert_eq!(rows.len(), 1usize << point.len());
    debug_assert!(outer_bits <= point.len());
    let (outer_point, inner_point) = point.split_at(outer_bits);
    #[cfg(feature = "parallel")]
    let (e_out, e_in) = rayon::join(|| eq_table(outer_point), || eq_table(inner_point));
    #[cfg(not(feature = "parallel"))]
    let (e_out, e_in) = (eq_table(outer_point), eq_table(inner_point));
    let inner_len = e_in.len();

    struct State<F: JoltField> {
        partial: Vec<Vec<F::Accumulator>>,
        block: Vec<Vec<F::Accumulator>>,
    }
    let zero = || State::<F> {
        partial: vec![vec![F::Accumulator::default(); k_chunk]; selectors.len()],
        block: vec![vec![F::Accumulator::default(); k_chunk]; selectors.len()],
    };
    let scatter = |mut state: State<F>, outer_index: usize| {
        let base = outer_index * inner_len;
        for (inner_index, weight) in e_in.iter().enumerate() {
            let row = &rows[base + inner_index];
            for (selector, block) in selectors.iter().zip(state.block.iter_mut()) {
                if let Some(k) = selector.index(row) {
                    block[k].add(*weight);
                }
            }
        }
        let outer_weight = e_out[outer_index];
        for (partial, block) in state.partial.iter_mut().zip(state.block.iter_mut()) {
            for (partial, block) in partial.iter_mut().zip(block.iter_mut()) {
                let value = std::mem::take(block).reduce();
                if value != F::zero() {
                    partial.fmadd(outer_weight, value);
                }
            }
        }
        state
    };
    let finish = |state: State<F>| -> Vec<Vec<F>> {
        state
            .partial
            .into_iter()
            .map(|buckets| buckets.into_iter().map(|bucket| bucket.reduce()).collect())
            .collect()
    };
    let merge = |mut left: State<F>, right: State<F>| {
        for (left, right) in left.partial.iter_mut().zip(right.partial) {
            for (left, right) in left.iter_mut().zip(right) {
                left.merge(right);
            }
        }
        left
    };

    #[cfg(feature = "parallel")]
    {
        finish(
            (0..e_out.len())
                .into_par_iter()
                .fold(zero, scatter)
                .reduce(zero, merge),
        )
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = merge;
        finish((0..e_out.len()).fold(zero(), scatter))
    }
}
