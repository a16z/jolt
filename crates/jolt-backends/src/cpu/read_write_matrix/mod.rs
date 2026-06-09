use jolt_field::{AdditiveAccumulator, Field, OptimizedMul, RingAccumulator, WithAccumulator};
use rayon::prelude::*;
use std::{cmp::Ordering, marker::PhantomData, mem::MaybeUninit};

#[cfg(feature = "field-inline")]
mod field_registers;
mod instruction;
mod ram;
mod registers;
mod stage6;
#[cfg(feature = "field-inline")]
pub use field_registers::{
    FieldRegistersIncClaimReductionState, FieldRegistersReadWriteState,
    FieldRegistersValEvaluationState,
};
pub use instruction::InstructionReadRafState;
pub use ram::{
    RamAddressMajorEntry, RamCycleMajorEntry, RamOutputCheckState, RamRaClaimReductionState,
    RamRafState, RamReadWriteState, RamValCheckState,
};
pub use registers::{RegistersReadWriteState, RegistersValEvaluationState};
pub use stage6::{
    BooleanityState, BytecodeReadRafState, IncClaimReductionState,
    InstructionRaVirtualizationState, RamHammingBooleanityState, RamRaVirtualizationState,
};

const MAX_LOOKUP_TABLE_SIZE: usize = 1 << 16;

#[derive(Debug, Clone, Default)]
pub struct OneHotCoeffTable<F: Field> {
    lookup_table: Vec<F>,
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct OneHotCoeffIndex(pub u16);

impl<F: Field> std::ops::Index<OneHotCoeffIndex> for OneHotCoeffTable<F> {
    type Output = F;

    fn index(&self, index: OneHotCoeffIndex) -> &Self::Output {
        &self.lookup_table[index.0 as usize]
    }
}

impl<F: Field> OneHotCoeffTable<F> {
    pub fn new(init_coeffs: Vec<F>) -> Self {
        debug_assert!(init_coeffs.len().is_power_of_two());
        Self {
            lookup_table: init_coeffs,
        }
    }

    pub fn bind(&mut self, r: F) {
        assert!(self.lookup_table.len() < MAX_LOOKUP_TABLE_SIZE);
        self.lookup_table = self
            .lookup_table
            .par_iter()
            .flat_map(|a| self.lookup_table.par_iter().map(|b| *b + r * (*a - *b)))
            .collect();
    }

    pub fn is_saturated(&self) -> bool {
        self.lookup_table.len() >= MAX_LOOKUP_TABLE_SIZE
    }

    pub fn len(&self) -> usize {
        self.lookup_table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lookup_table.is_empty()
    }
}

pub trait OneHotCoeff<F: Field>: Send + Sync {
    fn bind(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F,
        lookup_table: Option<&OneHotCoeffTable<F>>,
    ) -> Self;

    fn evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        lookup_table: Option<&OneHotCoeffTable<F>>,
    ) -> [F; 2];

    fn to_field(&self, lookup_table: Option<&OneHotCoeffTable<F>>) -> F;
}

impl<F: Field> OneHotCoeff<F> for F {
    fn bind(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F,
        _: Option<&OneHotCoeffTable<F>>,
    ) -> Self {
        match (even, odd) {
            (Some(&even), Some(&odd)) => even + r.mul_0_optimized(odd - even),
            (Some(&even), None) => (F::one() - r).mul_1_optimized(even),
            (None, Some(&odd)) => r.mul_1_optimized(odd),
            (None, None) => unreachable!("one-hot coefficient bind needs at least one entry"),
        }
    }

    fn evals(even: Option<&Self>, odd: Option<&Self>, _: Option<&OneHotCoeffTable<F>>) -> [F; 2] {
        match (even, odd) {
            (Some(&even), Some(&odd)) => [even, odd - even],
            (Some(&even), None) => [even, -even],
            (None, Some(&odd)) => [F::zero(), odd],
            (None, None) => unreachable!("one-hot coefficient eval needs at least one entry"),
        }
    }

    fn to_field(&self, _: Option<&OneHotCoeffTable<F>>) -> F {
        *self
    }
}

impl<F: Field> OneHotCoeff<F> for OneHotCoeffIndex {
    fn bind(
        even: Option<&Self>,
        odd: Option<&Self>,
        _r: F,
        lookup_table: Option<&OneHotCoeffTable<F>>,
    ) -> Self {
        let Some(table) = lookup_table else {
            unreachable!("lookup-table coefficients require a lookup table");
        };
        let coeff_index_bitwidth = table.len().trailing_zeros();
        debug_assert!(coeff_index_bitwidth <= 8);

        match (even, odd) {
            (Some(&even), Some(&odd)) => OneHotCoeffIndex((odd.0 << coeff_index_bitwidth) | even.0),
            (Some(&even), None) => even,
            (None, Some(&odd)) => OneHotCoeffIndex(odd.0 << coeff_index_bitwidth),
            (None, None) => unreachable!("one-hot coefficient bind needs at least one entry"),
        }
    }

    fn evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        lookup_table: Option<&OneHotCoeffTable<F>>,
    ) -> [F; 2] {
        let Some(table) = lookup_table else {
            unreachable!("lookup-table coefficients require a lookup table");
        };
        match (even, odd) {
            (Some(&even), Some(&odd)) => [table[even], table[odd] - table[even]],
            (Some(&even), None) => [table[even], -table[even]],
            (None, Some(&odd)) => [F::zero(), table[odd]],
            (None, None) => unreachable!("one-hot coefficient eval needs at least one entry"),
        }
    }

    fn to_field(&self, lookup_table: Option<&OneHotCoeffTable<F>>) -> F {
        let Some(table) = lookup_table else {
            unreachable!("lookup-table coefficients require a lookup table");
        };
        table[*self]
    }
}

pub trait CycleMajorMatrixEntry<F: Field>: Send + Sync + Sized {
    fn row(&self) -> usize;

    fn column(&self) -> usize;

    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F,
        ra_lookup_table: Option<&OneHotCoeffTable<F>>,
        wa_lookup_table: Option<&OneHotCoeffTable<F>>,
    ) -> Self;
}

pub trait AddressMajorMatrixEntry<F: Field>: Send + Sync + Sized {
    fn row(&self) -> usize;

    fn column(&self) -> usize;
}

pub trait AddressMajorBindableEntry<F: Field>: AddressMajorMatrixEntry<F> {
    fn prev_val(&self) -> F;

    fn next_val(&self) -> F;

    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        even_checkpoint: F,
        odd_checkpoint: F,
        r: F,
    ) -> Self;
}

#[derive(Clone, Copy, Debug)]
pub struct AddressMajorMessageInputs<F: Field> {
    pub even_checkpoint: F,
    pub odd_checkpoint: F,
    pub inc_eval: F,
    pub eq_eval: F,
    pub gamma: F,
}

pub trait AddressMajorMessageEntry<F: Field>: AddressMajorBindableEntry<F> {
    fn accumulate_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inputs: AddressMajorMessageInputs<F>,
        accumulators: &mut [<F as WithAccumulator>::Accumulator; 2],
    );
}

pub trait CycleMajorToAddressMajor<F: Field>: CycleMajorMatrixEntry<F> {
    type AddressMajor: AddressMajorMatrixEntry<F>;

    fn to_address_major(
        self,
        ra_lookup_table: Option<&OneHotCoeffTable<F>>,
        wa_lookup_table: Option<&OneHotCoeffTable<F>>,
    ) -> Self::AddressMajor;
}

pub trait CycleMajorMessageEntry<F: Field>: CycleMajorMatrixEntry<F> {
    fn accumulate_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [F; 2],
        gamma: F,
        accumulators: &mut [<F as WithAccumulator>::Accumulator; 2],
        ra_lookup_table: Option<&OneHotCoeffTable<F>>,
        wa_lookup_table: Option<&OneHotCoeffTable<F>>,
    );
}

#[derive(Debug, Default, Clone)]
pub struct ReadWriteMatrixCycleMajor<F: Field, E: CycleMajorMatrixEntry<F>> {
    pub entries: Vec<E>,
    pub ra_lookup_table: Option<OneHotCoeffTable<F>>,
    pub wa_lookup_table: Option<OneHotCoeffTable<F>>,
}

impl<F: Field, E: CycleMajorMatrixEntry<F>> ReadWriteMatrixCycleMajor<F, E> {
    pub fn new(entries: Vec<E>) -> Self {
        Self {
            entries,
            ra_lookup_table: None,
            wa_lookup_table: None,
        }
    }

    pub fn bind(&mut self, r: F) {
        let ra_lookup_table = self.ra_lookup_table.as_ref();
        let wa_lookup_table = self.wa_lookup_table.as_ref();

        let row_lengths = self
            .entries
            .par_chunk_by(|x, y| x.row() / 2 == y.row() / 2)
            .map(|entries| {
                let odd_row_start_index =
                    entries.partition_point(|entry| entry.row().is_multiple_of(2));
                let (even_row, odd_row) = entries.split_at(odd_row_start_index);
                let bound_len = Self::bind_rows(
                    even_row,
                    odd_row,
                    r,
                    &mut [],
                    true,
                    ra_lookup_table,
                    wa_lookup_table,
                );
                (entries.len(), bound_len)
            })
            .collect::<Vec<_>>();

        let bound_length = row_lengths.iter().map(|(_, bound_len)| bound_len).sum();
        let mut bound_entries = Vec::with_capacity(bound_length);
        let mut bound_entries_slice = bound_entries.spare_capacity_mut();
        let mut unbound_entries_slice = self.entries.as_slice();

        let mut output_slices = Vec::with_capacity(row_lengths.len());
        let mut input_slices = Vec::with_capacity(row_lengths.len());
        for (unbound_len, bound_len) in &row_lengths {
            let output_slice;
            (output_slice, bound_entries_slice) = bound_entries_slice.split_at_mut(*bound_len);
            output_slices.push(output_slice);
            let input_slice;
            (input_slice, unbound_entries_slice) = unbound_entries_slice.split_at(*unbound_len);
            input_slices.push(input_slice);
        }

        input_slices
            .par_iter()
            .zip(output_slices.into_par_iter())
            .for_each(|(input_slice, output_slice)| {
                let odd_row_start_index =
                    input_slice.partition_point(|entry| entry.row().is_multiple_of(2));
                let (even_row, odd_row) = input_slice.split_at(odd_row_start_index);
                let _ = Self::bind_rows(
                    even_row,
                    odd_row,
                    r,
                    output_slice,
                    false,
                    ra_lookup_table,
                    wa_lookup_table,
                );
            });

        // SAFETY: each disjoint spare-capacity slice is fully initialized by
        // `bind_rows`, and the dry-run lengths sum to `bound_length`.
        unsafe {
            bound_entries.set_len(bound_length);
        }
        self.entries = bound_entries;

        if let Some(ra_lookup_table) = self.ra_lookup_table.as_mut() {
            if !ra_lookup_table.is_saturated() {
                ra_lookup_table.bind(r);
            }
        }
        if let Some(wa_lookup_table) = self.wa_lookup_table.as_mut() {
            if !wa_lookup_table.is_saturated() {
                wa_lookup_table.bind(r);
            }
        }
    }

    fn bind_rows(
        even_row: &[E],
        odd_row: &[E],
        r: F,
        out: &mut [MaybeUninit<E>],
        dry_run: bool,
        ra_lookup_table: Option<&OneHotCoeffTable<F>>,
        wa_lookup_table: Option<&OneHotCoeffTable<F>>,
    ) -> usize {
        const PAR_THRESHOLD: usize = 32_768;

        if even_row.len() + odd_row.len() <= PAR_THRESHOLD {
            return Self::seq_bind_rows(
                even_row,
                odd_row,
                r,
                out,
                dry_run,
                ra_lookup_table,
                wa_lookup_table,
            );
        }

        let (even_pivot_idx, odd_pivot_idx) = if even_row.len() > odd_row.len() {
            let even_pivot_idx = even_row.len() / 2;
            let pivot = even_row[even_pivot_idx].column();
            let odd_pivot_idx = odd_row.partition_point(|x| x.column() < pivot);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = odd_row.len() / 2;
            let pivot = odd_row[odd_pivot_idx].column();
            let even_pivot_idx = even_row.partition_point(|x| x.column() < pivot);
            (even_pivot_idx, odd_pivot_idx)
        };

        let out_len = out.len();
        let (left_out, right_out) = if dry_run {
            out.split_at_mut(0)
        } else {
            out.split_at_mut(even_pivot_idx + odd_pivot_idx)
        };

        let (left_merged_len, right_merged_len) = rayon::join(
            || {
                Self::bind_rows(
                    &even_row[..even_pivot_idx],
                    &odd_row[..odd_pivot_idx],
                    r,
                    left_out,
                    true,
                    ra_lookup_table,
                    wa_lookup_table,
                )
            },
            || {
                Self::bind_rows(
                    &even_row[even_pivot_idx..],
                    &odd_row[odd_pivot_idx..],
                    r,
                    right_out,
                    true,
                    ra_lookup_table,
                    wa_lookup_table,
                )
            },
        );

        if !dry_run {
            assert_eq!(out_len, left_merged_len + right_merged_len);
            let (left_out, right_out) = out.split_at_mut(left_merged_len);
            let _ = rayon::join(
                || {
                    Self::bind_rows(
                        &even_row[..even_pivot_idx],
                        &odd_row[..odd_pivot_idx],
                        r,
                        left_out,
                        false,
                        ra_lookup_table,
                        wa_lookup_table,
                    )
                },
                || {
                    Self::bind_rows(
                        &even_row[even_pivot_idx..],
                        &odd_row[odd_pivot_idx..],
                        r,
                        right_out,
                        false,
                        ra_lookup_table,
                        wa_lookup_table,
                    )
                },
            );
        }

        left_merged_len + right_merged_len
    }

    fn seq_bind_rows(
        even: &[E],
        odd: &[E],
        r: F,
        out: &mut [MaybeUninit<E>],
        dry_run: bool,
        ra_lookup_table: Option<&OneHotCoeffTable<F>>,
        wa_lookup_table: Option<&OneHotCoeffTable<F>>,
    ) -> usize {
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;

        while i < even.len() && j < odd.len() {
            match even[i].column().cmp(&odd[j].column()) {
                Ordering::Equal => {
                    if !dry_run {
                        out[k] = MaybeUninit::new(E::bind_entries(
                            Some(&even[i]),
                            Some(&odd[j]),
                            r,
                            ra_lookup_table,
                            wa_lookup_table,
                        ));
                    }
                    i += 1;
                    j += 1;
                    k += 1;
                }
                Ordering::Less => {
                    if !dry_run {
                        out[k] = MaybeUninit::new(E::bind_entries(
                            Some(&even[i]),
                            None,
                            r,
                            ra_lookup_table,
                            wa_lookup_table,
                        ));
                    }
                    i += 1;
                    k += 1;
                }
                Ordering::Greater => {
                    if !dry_run {
                        out[k] = MaybeUninit::new(E::bind_entries(
                            None,
                            Some(&odd[j]),
                            r,
                            ra_lookup_table,
                            wa_lookup_table,
                        ));
                    }
                    j += 1;
                    k += 1;
                }
            }
        }

        if dry_run {
            return k + even[i..].len() + odd[j..].len();
        }

        for remaining_even_entry in &even[i..] {
            out[k] = MaybeUninit::new(E::bind_entries(
                Some(remaining_even_entry),
                None,
                r,
                ra_lookup_table,
                wa_lookup_table,
            ));
            k += 1;
        }
        for remaining_odd_entry in &odd[j..] {
            out[k] = MaybeUninit::new(E::bind_entries(
                None,
                Some(remaining_odd_entry),
                r,
                ra_lookup_table,
                wa_lookup_table,
            ));
            k += 1;
        }

        assert_eq!(out.len(), k);
        k
    }
}

impl<F, E> ReadWriteMatrixCycleMajor<F, E>
where
    F: Field,
    E: CycleMajorMessageEntry<F>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn prover_message_contribution(
        &self,
        even_row: &[E],
        odd_row: &[E],
        inc_evals: [F; 2],
        gamma: F,
    ) -> [F; 2] {
        const PAR_THRESHOLD: usize = 32_768;

        if even_row.len() + odd_row.len() <= PAR_THRESHOLD {
            return self.seq_prover_message_contribution(even_row, odd_row, inc_evals, gamma);
        }

        let (even_pivot_idx, odd_pivot_idx) = if even_row.len() > odd_row.len() {
            let even_pivot_idx = even_row.len() / 2;
            let pivot = even_row[even_pivot_idx].column();
            let odd_pivot_idx = odd_row.partition_point(|x| x.column() < pivot);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = odd_row.len() / 2;
            let pivot = odd_row[odd_pivot_idx].column();
            let even_pivot_idx = even_row.partition_point(|x| x.column() < pivot);
            (even_pivot_idx, odd_pivot_idx)
        };

        let (left_evals, right_evals) = rayon::join(
            || {
                self.prover_message_contribution(
                    &even_row[..even_pivot_idx],
                    &odd_row[..odd_pivot_idx],
                    inc_evals,
                    gamma,
                )
            },
            || {
                self.prover_message_contribution(
                    &even_row[even_pivot_idx..],
                    &odd_row[odd_pivot_idx..],
                    inc_evals,
                    gamma,
                )
            },
        );

        std::array::from_fn(|i| left_evals[i] + right_evals[i])
    }

    fn seq_prover_message_contribution(
        &self,
        even: &[E],
        odd: &[E],
        inc_evals: [F; 2],
        gamma: F,
    ) -> [F; 2] {
        let mut i = 0;
        let mut j = 0;
        let mut accumulators = [<F as WithAccumulator>::Accumulator::default(); 2];

        while i < even.len() && j < odd.len() {
            match even[i].column().cmp(&odd[j].column()) {
                Ordering::Equal => {
                    E::accumulate_evals(
                        Some(&even[i]),
                        Some(&odd[j]),
                        inc_evals,
                        gamma,
                        &mut accumulators,
                        self.ra_lookup_table.as_ref(),
                        self.wa_lookup_table.as_ref(),
                    );
                    i += 1;
                    j += 1;
                }
                Ordering::Less => {
                    E::accumulate_evals(
                        Some(&even[i]),
                        None,
                        inc_evals,
                        gamma,
                        &mut accumulators,
                        self.ra_lookup_table.as_ref(),
                        self.wa_lookup_table.as_ref(),
                    );
                    i += 1;
                }
                Ordering::Greater => {
                    E::accumulate_evals(
                        None,
                        Some(&odd[j]),
                        inc_evals,
                        gamma,
                        &mut accumulators,
                        self.ra_lookup_table.as_ref(),
                        self.wa_lookup_table.as_ref(),
                    );
                    j += 1;
                }
            }
        }

        for remaining_even_entry in &even[i..] {
            E::accumulate_evals(
                Some(remaining_even_entry),
                None,
                inc_evals,
                gamma,
                &mut accumulators,
                self.ra_lookup_table.as_ref(),
                self.wa_lookup_table.as_ref(),
            );
        }
        for remaining_odd_entry in &odd[j..] {
            E::accumulate_evals(
                None,
                Some(remaining_odd_entry),
                inc_evals,
                gamma,
                &mut accumulators,
                self.ra_lookup_table.as_ref(),
                self.wa_lookup_table.as_ref(),
            );
        }

        accumulators.map(AdditiveAccumulator::reduce)
    }
}

#[derive(Debug, Default, Clone)]
pub struct ReadWriteMatrixAddressMajor<F: Field, E: AddressMajorMatrixEntry<F>> {
    pub entries: Vec<E>,
    pub val_init: Vec<F>,
    _marker: PhantomData<F>,
}

impl<F: Field, E: AddressMajorMatrixEntry<F>> ReadWriteMatrixAddressMajor<F, E> {
    pub fn new(entries: Vec<E>) -> Self {
        Self::new_with_val_init(entries, Vec::new())
    }

    pub fn new_with_val_init(entries: Vec<E>, val_init: Vec<F>) -> Self {
        Self {
            entries,
            val_init,
            _marker: PhantomData,
        }
    }
}

impl<F: Field, E: CycleMajorToAddressMajor<F>> From<ReadWriteMatrixCycleMajor<F, E>>
    for ReadWriteMatrixAddressMajor<F, E::AddressMajor>
{
    fn from(mut cycle_major: ReadWriteMatrixCycleMajor<F, E>) -> Self {
        let mut entries = std::mem::take(&mut cycle_major.entries);
        entries.par_sort_by(|a, b| match a.column().cmp(&b.column()) {
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
            Ordering::Equal => a.row().cmp(&b.row()),
        });
        let entries = entries
            .into_par_iter()
            .map(|entry| {
                entry.to_address_major(
                    cycle_major.ra_lookup_table.as_ref(),
                    cycle_major.wa_lookup_table.as_ref(),
                )
            })
            .collect();
        Self::new(entries)
    }
}

impl<F, E> ReadWriteMatrixAddressMajor<F, E>
where
    F: Field,
    E: AddressMajorBindableEntry<F>,
{
    pub fn bind(&mut self, r: F) {
        assert!(
            !self.val_init.is_empty(),
            "address-major bind needs initial column checkpoints"
        );
        assert!(
            self.val_init.len().is_power_of_two(),
            "address-major initial checkpoint vector length must be a power of two"
        );

        let col_lengths = self
            .entries
            .par_chunk_by(|x, y| x.column() / 2 == y.column() / 2)
            .map(|entries| {
                let odd_col_start_index =
                    entries.partition_point(|entry| entry.column().is_multiple_of(2));
                let (even_col, odd_col) = entries.split_at(odd_col_start_index);
                let bound_len =
                    Self::bind_cols(even_col, odd_col, F::zero(), F::zero(), r, &mut [], true);
                (entries.len(), bound_len)
            })
            .collect::<Vec<_>>();

        let bound_length = col_lengths.iter().map(|(_, bound_len)| bound_len).sum();
        let mut bound_entries = Vec::with_capacity(bound_length);
        let mut bound_entries_slice = bound_entries.spare_capacity_mut();
        let mut unbound_entries_slice = self.entries.as_slice();

        let mut output_slices = Vec::with_capacity(col_lengths.len());
        let mut input_slices = Vec::with_capacity(col_lengths.len());
        for (unbound_len, bound_len) in &col_lengths {
            let output_slice;
            (output_slice, bound_entries_slice) = bound_entries_slice.split_at_mut(*bound_len);
            output_slices.push(output_slice);
            let input_slice;
            (input_slice, unbound_entries_slice) = unbound_entries_slice.split_at(*unbound_len);
            input_slices.push(input_slice);
        }

        input_slices
            .par_iter()
            .zip(output_slices.into_par_iter())
            .for_each(|(input_slice, output_slice)| {
                let odd_col_start_index =
                    input_slice.partition_point(|entry| entry.column().is_multiple_of(2));
                let (even_col, odd_col) = input_slice.split_at(odd_col_start_index);
                let even_col_idx = 2 * (input_slice[0].column() / 2);
                let odd_col_idx = even_col_idx + 1;
                let _ = Self::bind_cols(
                    even_col,
                    odd_col,
                    self.val_init[even_col_idx],
                    self.val_init[odd_col_idx],
                    r,
                    output_slice,
                    false,
                );
            });

        // SAFETY: each disjoint spare-capacity slice is fully initialized by
        // `bind_cols`, and the dry-run lengths sum to `bound_length`.
        unsafe {
            bound_entries.set_len(bound_length);
        }
        self.entries = bound_entries;
        self.bind_val_init(r);
    }

    fn bind_val_init(&mut self, r: F) {
        self.val_init = self
            .val_init
            .par_chunks_exact(2)
            .map(|pair| pair[0] + r * (pair[1] - pair[0]))
            .collect();
    }

    fn bind_cols(
        even_col: &[E],
        odd_col: &[E],
        even_checkpoint: F,
        odd_checkpoint: F,
        r: F,
        out: &mut [MaybeUninit<E>],
        dry_run: bool,
    ) -> usize {
        const PAR_THRESHOLD: usize = 32_768;

        if even_col.len() + odd_col.len() <= PAR_THRESHOLD {
            return Self::seq_bind_cols(
                even_col,
                odd_col,
                even_checkpoint,
                odd_checkpoint,
                r,
                out,
                dry_run,
            );
        }

        let (even_pivot_idx, odd_pivot_idx) = if even_col.len() > odd_col.len() {
            let even_pivot_idx = even_col.len() / 2;
            let pivot = even_col[even_pivot_idx].row();
            let odd_pivot_idx = odd_col.partition_point(|entry| entry.row() < pivot);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = odd_col.len() / 2;
            let pivot = odd_col[odd_pivot_idx].row();
            let even_pivot_idx = even_col.partition_point(|entry| entry.row() < pivot);
            (even_pivot_idx, odd_pivot_idx)
        };

        let out_len = out.len();
        let (left_out, right_out) = if dry_run {
            out.split_at_mut(0)
        } else {
            out.split_at_mut(even_pivot_idx + odd_pivot_idx)
        };

        let (left_merged_len, right_merged_len) = rayon::join(
            || {
                Self::bind_cols(
                    &even_col[..even_pivot_idx],
                    &odd_col[..odd_pivot_idx],
                    F::zero(),
                    F::zero(),
                    r,
                    left_out,
                    true,
                )
            },
            || {
                Self::bind_cols(
                    &even_col[even_pivot_idx..],
                    &odd_col[odd_pivot_idx..],
                    F::zero(),
                    F::zero(),
                    r,
                    right_out,
                    true,
                )
            },
        );

        if !dry_run {
            assert_eq!(out_len, left_merged_len + right_merged_len);
            let (left_out, right_out) = out.split_at_mut(left_merged_len);
            let _ = rayon::join(
                || {
                    Self::bind_cols(
                        &even_col[..even_pivot_idx],
                        &odd_col[..odd_pivot_idx],
                        even_checkpoint,
                        odd_checkpoint,
                        r,
                        left_out,
                        false,
                    )
                },
                || {
                    let even_checkpoint = if even_col.is_empty() {
                        even_checkpoint
                    } else if even_pivot_idx != 0 {
                        even_col[even_pivot_idx - 1].next_val()
                    } else {
                        even_col[even_pivot_idx].prev_val()
                    };
                    let odd_checkpoint = if odd_col.is_empty() {
                        odd_checkpoint
                    } else if odd_pivot_idx != 0 {
                        odd_col[odd_pivot_idx - 1].next_val()
                    } else {
                        odd_col[odd_pivot_idx].prev_val()
                    };
                    Self::bind_cols(
                        &even_col[even_pivot_idx..],
                        &odd_col[odd_pivot_idx..],
                        even_checkpoint,
                        odd_checkpoint,
                        r,
                        right_out,
                        false,
                    )
                },
            );
        }

        left_merged_len + right_merged_len
    }

    fn seq_bind_cols(
        even: &[E],
        odd: &[E],
        mut even_checkpoint: F,
        mut odd_checkpoint: F,
        r: F,
        out: &mut [MaybeUninit<E>],
        dry_run: bool,
    ) -> usize {
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;

        while i < even.len() && j < odd.len() {
            match even[i].row().cmp(&odd[j].row()) {
                Ordering::Equal => {
                    if !dry_run {
                        out[k] = MaybeUninit::new(E::bind_entries(
                            Some(&even[i]),
                            Some(&odd[j]),
                            even_checkpoint,
                            odd_checkpoint,
                            r,
                        ));
                    }
                    even_checkpoint = even[i].next_val();
                    odd_checkpoint = odd[j].next_val();
                    i += 1;
                    j += 1;
                    k += 1;
                }
                Ordering::Less => {
                    if !dry_run {
                        out[k] = MaybeUninit::new(E::bind_entries(
                            Some(&even[i]),
                            None,
                            even_checkpoint,
                            odd_checkpoint,
                            r,
                        ));
                    }
                    even_checkpoint = even[i].next_val();
                    i += 1;
                    k += 1;
                }
                Ordering::Greater => {
                    if !dry_run {
                        out[k] = MaybeUninit::new(E::bind_entries(
                            None,
                            Some(&odd[j]),
                            even_checkpoint,
                            odd_checkpoint,
                            r,
                        ));
                    }
                    odd_checkpoint = odd[j].next_val();
                    j += 1;
                    k += 1;
                }
            }
        }

        for remaining_even_entry in &even[i..] {
            if !dry_run {
                out[k] = MaybeUninit::new(E::bind_entries(
                    Some(remaining_even_entry),
                    None,
                    even_checkpoint,
                    odd_checkpoint,
                    r,
                ));
            }
            even_checkpoint = remaining_even_entry.next_val();
            k += 1;
        }
        for remaining_odd_entry in &odd[j..] {
            if !dry_run {
                out[k] = MaybeUninit::new(E::bind_entries(
                    None,
                    Some(remaining_odd_entry),
                    even_checkpoint,
                    odd_checkpoint,
                    r,
                ));
            }
            odd_checkpoint = remaining_odd_entry.next_val();
            k += 1;
        }

        if !dry_run {
            assert_eq!(out.len(), k);
        }
        k
    }
}

impl<F, E> ReadWriteMatrixAddressMajor<F, E>
where
    F: Field,
    E: AddressMajorMessageEntry<F>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn prover_message_contribution(
        even_col: &[E],
        odd_col: &[E],
        even_checkpoint: F,
        odd_checkpoint: F,
        inc: &[F],
        eq: &[F],
        gamma: F,
    ) -> [F; 2] {
        const PAR_THRESHOLD: usize = 32_768;

        if even_col.len() + odd_col.len() <= PAR_THRESHOLD {
            return Self::seq_prover_message_contribution(
                even_col,
                odd_col,
                even_checkpoint,
                odd_checkpoint,
                inc,
                eq,
                gamma,
            );
        }

        let (even_pivot_idx, odd_pivot_idx) = if even_col.len() > odd_col.len() {
            let even_pivot_idx = even_col.len() / 2;
            let pivot = even_col[even_pivot_idx].row();
            let odd_pivot_idx = odd_col.partition_point(|entry| entry.row() < pivot);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = odd_col.len() / 2;
            let pivot = odd_col[odd_pivot_idx].row();
            let even_pivot_idx = even_col.partition_point(|entry| entry.row() < pivot);
            (even_pivot_idx, odd_pivot_idx)
        };

        let (top_evals, bottom_evals) = rayon::join(
            || {
                Self::prover_message_contribution(
                    &even_col[..even_pivot_idx],
                    &odd_col[..odd_pivot_idx],
                    even_checkpoint,
                    odd_checkpoint,
                    inc,
                    eq,
                    gamma,
                )
            },
            || {
                let even_checkpoint = if even_col.is_empty() {
                    even_checkpoint
                } else if even_pivot_idx != 0 {
                    even_col[even_pivot_idx - 1].next_val()
                } else {
                    even_col[even_pivot_idx].prev_val()
                };
                let odd_checkpoint = if odd_col.is_empty() {
                    odd_checkpoint
                } else if odd_pivot_idx != 0 {
                    odd_col[odd_pivot_idx - 1].next_val()
                } else {
                    odd_col[odd_pivot_idx].prev_val()
                };
                Self::prover_message_contribution(
                    &even_col[even_pivot_idx..],
                    &odd_col[odd_pivot_idx..],
                    even_checkpoint,
                    odd_checkpoint,
                    inc,
                    eq,
                    gamma,
                )
            },
        );

        [
            top_evals[0] + bottom_evals[0],
            top_evals[1] + bottom_evals[1],
        ]
    }

    fn seq_prover_message_contribution(
        even: &[E],
        odd: &[E],
        mut even_checkpoint: F,
        mut odd_checkpoint: F,
        inc: &[F],
        eq: &[F],
        gamma: F,
    ) -> [F; 2] {
        let mut i = 0;
        let mut j = 0;
        let mut accumulators = [<F as WithAccumulator>::Accumulator::default(); 2];

        while i < even.len() && j < odd.len() {
            match even[i].row().cmp(&odd[j].row()) {
                Ordering::Equal => {
                    E::accumulate_evals(
                        Some(&even[i]),
                        Some(&odd[j]),
                        AddressMajorMessageInputs {
                            even_checkpoint,
                            odd_checkpoint,
                            inc_eval: inc[even[i].row()],
                            eq_eval: eq[even[i].row()],
                            gamma,
                        },
                        &mut accumulators,
                    );
                    even_checkpoint = even[i].next_val();
                    odd_checkpoint = odd[j].next_val();
                    i += 1;
                    j += 1;
                }
                Ordering::Less => {
                    E::accumulate_evals(
                        Some(&even[i]),
                        None,
                        AddressMajorMessageInputs {
                            even_checkpoint,
                            odd_checkpoint,
                            inc_eval: inc[even[i].row()],
                            eq_eval: eq[even[i].row()],
                            gamma,
                        },
                        &mut accumulators,
                    );
                    even_checkpoint = even[i].next_val();
                    i += 1;
                }
                Ordering::Greater => {
                    E::accumulate_evals(
                        None,
                        Some(&odd[j]),
                        AddressMajorMessageInputs {
                            even_checkpoint,
                            odd_checkpoint,
                            inc_eval: inc[odd[j].row()],
                            eq_eval: eq[odd[j].row()],
                            gamma,
                        },
                        &mut accumulators,
                    );
                    odd_checkpoint = odd[j].next_val();
                    j += 1;
                }
            }
        }

        for remaining_even_entry in &even[i..] {
            E::accumulate_evals(
                Some(remaining_even_entry),
                None,
                AddressMajorMessageInputs {
                    even_checkpoint,
                    odd_checkpoint,
                    inc_eval: inc[remaining_even_entry.row()],
                    eq_eval: eq[remaining_even_entry.row()],
                    gamma,
                },
                &mut accumulators,
            );
            even_checkpoint = remaining_even_entry.next_val();
        }
        for remaining_odd_entry in &odd[j..] {
            E::accumulate_evals(
                None,
                Some(remaining_odd_entry),
                AddressMajorMessageInputs {
                    even_checkpoint,
                    odd_checkpoint,
                    inc_eval: inc[remaining_odd_entry.row()],
                    eq_eval: eq[remaining_odd_entry.row()],
                    gamma,
                },
                &mut accumulators,
            );
            odd_checkpoint = remaining_odd_entry.next_val();
        }

        accumulators.map(AdditiveAccumulator::reduce)
    }
}
