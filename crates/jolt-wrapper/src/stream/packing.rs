use jolt_crypto::ec::bn254::bit_columns::g1_bit_columns_msm;
use jolt_crypto::Bn254;
use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, NoopVerifierObserver, VerifierObserver};
use jolt_openings::CommitmentScheme;
use jolt_poly::EqPolynomial;
use rayon::prelude::*;

use super::{ColumnId, Commitment, StreamError};

const RLC_BLOCK_ROWS: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Column {
    /// Values checked as 0/1 before commitment; the proved relation must enforce booleanity.
    Bits(Vec<u8>),
    /// Small-scalar commitment input; the proved relation must enforce its range requirement.
    U16(Vec<u16>),
    U32(Vec<u32>),
    Fr(Vec<Fr>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Self::Bits(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::Fr(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn value(&self, row: usize) -> Fr {
        match self {
            Self::Bits(values) => Fr::from_u64(u64::from(values[row])),
            Self::U16(values) => Fr::from_u64(u64::from(values[row])),
            Self::U32(values) => Fr::from_u64(u64::from(values[row])),
            Self::Fr(values) => values[row],
        }
    }
}

/// `ceil(columns / k)` polynomials with `packed[row * k + slot] = column[g*k+slot][row]`.
/// Thus the row variables precede the `log2(k)` low column-slot variables in an opening point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackingLayout {
    pub rows: usize,
    pub column_count: usize,
    pub k: usize,
    pub group_count: usize,
    pub padded_group_count: usize,
    pub padded_column_count: usize,
}

impl PackingLayout {
    pub fn new(rows: usize, column_count: usize, k: usize) -> Result<Self, StreamError> {
        if column_count == 0 {
            return Err(StreamError::NoColumns);
        }
        if !k.is_power_of_two() {
            return Err(StreamError::InvalidPacking(k));
        }
        if rows == 0 || !rows.is_power_of_two() {
            return Err(StreamError::RowCount {
                column: 0,
                expected: rows.next_power_of_two(),
                actual: rows,
            });
        }
        let group_count = column_count.div_ceil(k);
        let padded_group_count = group_count.next_power_of_two();
        let padded_column_count = padded_group_count
            .checked_mul(k)
            .ok_or(StreamError::PackedLengthOverflow)?;
        Ok(Self {
            rows,
            column_count,
            k,
            group_count,
            padded_group_count,
            padded_column_count,
        })
    }

    pub fn row_vars(self) -> usize {
        self.rows.trailing_zeros() as usize
    }

    pub fn group_vars(self) -> usize {
        self.padded_group_count.trailing_zeros() as usize
    }

    pub fn slot_vars(self) -> usize {
        self.k.trailing_zeros() as usize
    }

    pub fn column_vars(self) -> usize {
        self.group_vars() + self.slot_vars()
    }

    pub fn packed_vars(self) -> usize {
        self.row_vars() + self.slot_vars()
    }

    pub fn split_column_point(self, point: &[Fr]) -> Result<(&[Fr], &[Fr]), StreamError> {
        if point.len() != self.column_vars() {
            return Err(StreamError::PointDimension {
                expected: self.column_vars(),
                actual: point.len(),
            });
        }
        Ok(point.split_at(self.group_vars()))
    }

    pub fn group_weights(self, column_point: &[Fr]) -> Result<Vec<Fr>, StreamError> {
        self.group_weights_observed(column_point, &mut NoopVerifierObserver)
    }

    pub(crate) fn group_weights_observed<O: VerifierObserver>(
        self,
        column_point: &[Fr],
        observer: &mut O,
    ) -> Result<Vec<Fr>, StreamError> {
        let (group_point, _) = self.split_column_point(column_point)?;
        let mut evaluations = vec![Fr::one(); self.padded_group_count];
        let mut size = 1;
        for &challenge in group_point {
            size *= 2;
            for index in (0..size).rev().step_by(2) {
                let scalar = evaluations[index / 2];
                evaluations[index] = observer.fr_mul(scalar, challenge);
                evaluations[index - 1] = scalar - evaluations[index];
            }
        }
        evaluations.truncate(self.group_count);
        Ok(evaluations)
    }

    pub fn packed_point(
        self,
        row_point: &[Fr],
        column_point: &[Fr],
    ) -> Result<Vec<Fr>, StreamError> {
        if row_point.len() != self.row_vars() {
            return Err(StreamError::PointDimension {
                expected: self.row_vars(),
                actual: row_point.len(),
            });
        }
        let (_, slot_point) = self.split_column_point(column_point)?;
        let mut point = Vec::with_capacity(self.packed_vars());
        point.extend_from_slice(row_point);
        point.extend_from_slice(slot_point);
        Ok(point)
    }
}

#[derive(Clone, Debug)]
pub enum PackedPolynomial {
    Bits(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    Fr(Vec<Fr>),
}

impl PackedPolynomial {
    fn len(&self) -> usize {
        match self {
            Self::Bits(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::Fr(values) => values.len(),
        }
    }

    fn accumulate_rlc(
        &self,
        target: &mut [Fr],
        start: usize,
        slots: &[usize],
        packing: usize,
        weight: Fr,
    ) {
        match self {
            Self::Bits(values) => accumulate_rows(
                target,
                &values[start..start + target.len()],
                slots,
                packing,
                |entry| (*entry == 1).then_some(weight),
            ),
            Self::U16(values) => accumulate_rows(
                target,
                &values[start..start + target.len()],
                slots,
                packing,
                |&entry| (entry != 0).then(|| weight.mul_u64(u64::from(entry))),
            ),
            Self::U32(values) => accumulate_rows(
                target,
                &values[start..start + target.len()],
                slots,
                packing,
                |&entry| (entry != 0).then(|| weight.mul_u64(u64::from(entry))),
            ),
            Self::Fr(values) => accumulate_rows(
                target,
                &values[start..start + target.len()],
                slots,
                packing,
                |&entry| (!entry.is_zero()).then(|| entry * weight),
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PackedColumns {
    pub layout: PackingLayout,
    pub polynomials: Vec<PackedPolynomial>,
    pub commitments: Vec<Commitment>,
}

impl PackedColumns {
    pub fn column_evaluations(&self, row_point: &[Fr]) -> Result<Vec<Fr>, StreamError> {
        let expected = self.layout.row_vars();
        if row_point.len() != expected {
            return Err(StreamError::PointDimension {
                expected,
                actual: row_point.len(),
            });
        }
        let row_weights = EqPolynomial::<Fr>::evals(row_point, None);
        let bound_groups: Vec<Vec<Fr>> = self
            .polynomials
            .par_iter()
            .map(|polynomial| {
                let mut values = vec![Fr::zero(); self.layout.k];
                match polynomial {
                    PackedPolynomial::Bits(evaluations) => {
                        for (&row_weight, row) in row_weights
                            .iter()
                            .zip(evaluations.chunks_exact(self.layout.k))
                        {
                            for (value, &entry) in values.iter_mut().zip(row) {
                                if entry == 1 {
                                    *value += row_weight;
                                }
                            }
                        }
                    }
                    PackedPolynomial::U16(evaluations) => {
                        for (&row_weight, row) in row_weights
                            .iter()
                            .zip(evaluations.chunks_exact(self.layout.k))
                        {
                            for (value, &entry) in values.iter_mut().zip(row) {
                                *value += row_weight.mul_u64(u64::from(entry));
                            }
                        }
                    }
                    PackedPolynomial::U32(evaluations) => {
                        for (&row_weight, row) in row_weights
                            .iter()
                            .zip(evaluations.chunks_exact(self.layout.k))
                        {
                            for (value, &entry) in values.iter_mut().zip(row) {
                                *value += row_weight.mul_u64(u64::from(entry));
                            }
                        }
                    }
                    PackedPolynomial::Fr(evaluations) => {
                        for (&row_weight, row) in row_weights
                            .iter()
                            .zip(evaluations.chunks_exact(self.layout.k))
                        {
                            for (value, &entry) in values.iter_mut().zip(row) {
                                *value += row_weight * entry;
                            }
                        }
                    }
                }
                values
            })
            .collect();
        let mut values: Vec<Fr> = (0..self.layout.column_count)
            .map(|column| bound_groups[column / self.layout.k][column % self.layout.k])
            .collect();
        values.resize(self.layout.padded_column_count, Fr::zero());
        Ok(values)
    }

    pub(crate) fn column_evaluations_from_bound(
        &self,
        bound: &[(ColumnId, Fr)],
    ) -> Result<Vec<Fr>, StreamError> {
        let mut values = vec![None; self.layout.column_count];
        for &(column, value) in bound {
            let index = column.index(self.layout.k)?;
            let target = values.get_mut(index).ok_or(StreamError::ColumnOutOfRange {
                column: index,
                columns: self.layout.column_count,
            })?;
            if target.is_some_and(|existing| existing != value) {
                return Err(StreamError::OpeningClaim);
            }
            *target = Some(value);
        }
        let mut values = values
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(StreamError::OpeningClaim)?;
        values.resize(self.layout.padded_column_count, Fr::zero());
        Ok(values)
    }

    pub fn rlc_evaluations(&self, weights: &[Fr]) -> Result<Vec<Fr>, StreamError> {
        self.rlc_evaluations_skipping(weights, &[])
    }

    pub(crate) fn rlc_evaluations_skipping(
        &self,
        weights: &[Fr],
        zero_columns: &[ColumnId],
    ) -> Result<Vec<Fr>, StreamError> {
        if weights.len() != self.polynomials.len() {
            return Err(StreamError::OpeningClaim);
        }
        let length = self
            .polynomials
            .first()
            .ok_or(StreamError::NoColumns)?
            .len();
        if self
            .polynomials
            .iter()
            .any(|polynomial| polynomial.len() != length)
        {
            return Err(StreamError::StageCount);
        }
        let mut active_slots = vec![vec![true; self.layout.k]; self.polynomials.len()];
        for &column in zero_columns {
            let index = column.index(self.layout.k)?;
            if index >= self.layout.column_count {
                return Err(StreamError::ColumnOutOfRange {
                    column: index,
                    columns: self.layout.column_count,
                });
            }
            active_slots[index / self.layout.k][index % self.layout.k] = false;
        }
        let active_slots = active_slots
            .into_iter()
            .map(|slots| {
                slots
                    .into_iter()
                    .enumerate()
                    .filter_map(|(slot, active)| active.then_some(slot))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let block_len = self
            .layout
            .k
            .checked_mul(RLC_BLOCK_ROWS)
            .ok_or(StreamError::PackedLengthOverflow)?;
        let mut combined = vec![Fr::zero(); length];
        combined
            .par_chunks_mut(block_len)
            .enumerate()
            .for_each(|(block, target)| {
                let start = block * block_len;
                for ((polynomial, &weight), slots) in
                    self.polynomials.iter().zip(weights).zip(&active_slots)
                {
                    if weight.is_zero() || slots.is_empty() {
                        continue;
                    }
                    polynomial.accumulate_rlc(target, start, slots, self.layout.k, weight);
                }
            });
        Ok(combined)
    }
}

fn accumulate_rows<T>(
    target: &mut [Fr],
    source: &[T],
    slots: &[usize],
    packing: usize,
    mut contribution: impl FnMut(&T) -> Option<Fr>,
) {
    for (target, source) in target
        .chunks_exact_mut(packing)
        .zip(source.chunks_exact(packing))
    {
        for &slot in slots {
            if let Some(value) = contribution(&source[slot]) {
                target[slot] += value;
            }
        }
    }
}

pub fn commit_packed(
    columns: &[Column],
    k: usize,
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<PackedColumns, StreamError> {
    if columns.is_empty() {
        return Err(StreamError::NoColumns);
    }
    let rows = columns[0].len();
    let layout = PackingLayout::new(rows, columns.len(), k)?;
    for (column, values) in columns.iter().enumerate() {
        if values.len() != rows {
            return Err(StreamError::RowCount {
                column,
                expected: rows,
                actual: values.len(),
            });
        }
        if let Column::Bits(bits) = values {
            if let Some((row, &value)) = bits.iter().enumerate().find(|(_, bit)| **bit > 1) {
                return Err(StreamError::InvalidBit { column, row, value });
            }
        }
    }
    let packed_len = layout
        .rows
        .checked_mul(k)
        .ok_or(StreamError::PackedLengthOverflow)?;
    if setup.g1_powers().len() < packed_len {
        return Err(StreamError::SetupTooSmall {
            required: packed_len,
            actual: setup.g1_powers().len(),
        });
    }
    let groups = layout.group_count;
    let polynomials: Vec<PackedPolynomial> = (0..groups)
        .into_par_iter()
        .map(|group| {
            let mut group_columns = columns.iter().skip(group * k).take(k);
            if group_columns
                .clone()
                .all(|column| matches!(column, Column::Bits(_)))
            {
                let mut packed = vec![0; packed_len];
                for row in 0..rows {
                    for slot in 0..k {
                        if let Some(Column::Bits(column)) = columns.get(group * k + slot) {
                            packed[row * k + slot] = column[row];
                        }
                    }
                }
                PackedPolynomial::Bits(packed)
            } else if group_columns
                .clone()
                .all(|column| matches!(column, Column::U16(_)))
            {
                let mut packed = vec![0; packed_len];
                for row in 0..rows {
                    for slot in 0..k {
                        if let Some(Column::U16(column)) = columns.get(group * k + slot) {
                            packed[row * k + slot] = column[row];
                        }
                    }
                }
                PackedPolynomial::U16(packed)
            } else if group_columns.all(|column| matches!(column, Column::U32(_))) {
                let mut packed = vec![0; packed_len];
                for row in 0..rows {
                    for slot in 0..k {
                        if let Some(Column::U32(column)) = columns.get(group * k + slot) {
                            packed[row * k + slot] = column[row];
                        }
                    }
                }
                PackedPolynomial::U32(packed)
            } else {
                let mut packed = vec![Fr::zero(); packed_len];
                for row in 0..rows {
                    for slot in 0..k {
                        if let Some(column) = columns.get(group * k + slot) {
                            packed[row * k + slot] = column.value(row);
                        }
                    }
                }
                PackedPolynomial::Fr(packed)
            }
        })
        .collect();

    let bit_groups: Vec<usize> = polynomials
        .iter()
        .enumerate()
        .filter_map(|(group, polynomial)| {
            matches!(polynomial, PackedPolynomial::Bits(_)).then_some(group)
        })
        .collect();
    let bit_refs: Vec<&[u8]> = polynomials
        .iter()
        .filter_map(|polynomial| match polynomial {
            PackedPolynomial::Bits(values) => Some(values.as_slice()),
            PackedPolynomial::U16(_) | PackedPolynomial::U32(_) | PackedPolynomial::Fr(_) => None,
        })
        .collect();
    let mut indexed_commitments: Vec<(usize, Commitment)> = bit_groups
        .into_iter()
        .zip(
            g1_bit_columns_msm(&setup.g1_powers()[..packed_len], &bit_refs)
                .into_iter()
                .map(Commitment::new),
        )
        .collect();
    let mut other_commitments = (0..groups)
        .filter(|&group| !matches!(polynomials[group], PackedPolynomial::Bits(_)))
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|group| {
            let commitment = match &polynomials[group] {
                PackedPolynomial::U16(values) => Commitment::new(Bn254::g1_affine_msm_small(
                    &setup.g1_powers()[..packed_len],
                    values,
                )),
                PackedPolynomial::U32(values) => Commitment::new(Bn254::g1_affine_msm_small(
                    &setup.g1_powers()[..packed_len],
                    values,
                )),
                PackedPolynomial::Fr(values) => HyperKZGScheme::<Bn254>::commit(values, setup)
                    .map(|(commitment, ())| commitment)
                    .map_err(StreamError::Commitment)?,
                PackedPolynomial::Bits(_) => return Err(StreamError::StageCount),
            };
            Ok((group, commitment))
        })
        .collect::<Result<Vec<_>, StreamError>>()?;
    indexed_commitments.append(&mut other_commitments);
    indexed_commitments.sort_unstable_by_key(|(group, _)| *group);
    let commitments = indexed_commitments
        .into_iter()
        .map(|(_, commitment)| commitment)
        .collect();

    Ok(PackedColumns {
        layout,
        polynomials,
        commitments,
    })
}

pub fn combine_packed_phases(phases: Vec<PackedColumns>) -> Result<PackedColumns, StreamError> {
    let first = phases.first().ok_or(StreamError::NoColumns)?;
    let rows = first.layout.rows;
    let k = first.layout.k;
    let last = phases.len().saturating_sub(1);
    for (index, phase) in phases.iter().enumerate() {
        if phase.layout.rows != rows
            || phase.layout.k != k
            || (index != last && phase.layout.column_count % k != 0)
        {
            return Err(StreamError::StageCount);
        }
    }
    let column_count = phases.iter().map(|phase| phase.layout.column_count).sum();
    let layout = PackingLayout::new(rows, column_count, k)?;
    let mut polynomials = Vec::with_capacity(layout.group_count);
    let mut commitments = Vec::with_capacity(layout.group_count);
    for phase in phases {
        polynomials.extend(phase.polynomials);
        commitments.extend(phase.commitments);
    }
    if polynomials.len() != layout.group_count || commitments.len() != layout.group_count {
        return Err(StreamError::StageCount);
    }
    Ok(PackedColumns {
        layout,
        polynomials,
        commitments,
    })
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests require valid packed layouts")]

    use super::*;

    #[test]
    fn bound_columns_require_complete_consistent_coverage() {
        let packed = PackedColumns {
            layout: PackingLayout::new(2, 3, 2).unwrap(),
            polynomials: vec![
                PackedPolynomial::U16(vec![1, 2, 3, 4]),
                PackedPolynomial::Fr(vec![
                    Fr::from_u64(5),
                    Fr::zero(),
                    Fr::from_u64(7),
                    Fr::zero(),
                ]),
            ],
            commitments: Vec::new(),
        };
        let expected = packed.column_evaluations(&[Fr::from_u64(2)]).unwrap();
        let mut bound = vec![
            (ColumnId { group: 0, slot: 0 }, Fr::from_u64(5)),
            (ColumnId { group: 0, slot: 1 }, Fr::from_u64(6)),
            (ColumnId { group: 1, slot: 0 }, Fr::from_u64(9)),
        ];
        assert_eq!(
            packed.column_evaluations_from_bound(&bound).unwrap(),
            expected
        );
        assert!(matches!(
            packed.column_evaluations_from_bound(&bound[..2]),
            Err(StreamError::OpeningClaim)
        ));
        bound.push((ColumnId { group: 1, slot: 0 }, Fr::from_u64(10)));
        assert!(matches!(
            packed.column_evaluations_from_bound(&bound),
            Err(StreamError::OpeningClaim)
        ));
    }

    #[test]
    fn typed_rlc_skips_only_padding_across_row_blocks() {
        for rows in [2, 128] {
            let bits: Vec<_> = (0..rows).flat_map(|row| [(row % 2) as u8, 0]).collect();
            let small: Vec<_> = (0..rows).flat_map(|row| [row as u16, 0]).collect();
            let wide: Vec<_> = (0..rows).flat_map(|row| [65_536 + row as u32, 0]).collect();
            let full: Vec<_> = (0..rows)
                .flat_map(|row| [-Fr::from_u64(row as u64 + 1), Fr::zero()])
                .collect();
            let packed = PackedColumns {
                layout: PackingLayout::new(rows, 8, 2).unwrap(),
                polynomials: vec![
                    PackedPolynomial::Bits(bits),
                    PackedPolynomial::U16(small),
                    PackedPolynomial::U32(wide),
                    PackedPolynomial::Fr(full),
                ],
                commitments: Vec::new(),
            };
            let weights = [2, 3, 5, 7].map(Fr::from_u64);
            let padding: Vec<_> = (0..4).map(|group| ColumnId { group, slot: 1 }).collect();
            let expected: Vec<_> = (0..rows)
                .flat_map(|row| {
                    let value = 2 * (row % 2) + 3 * row + 5 * (65_536 + row) - 7 * (row + 1);
                    [Fr::from_u64(value as u64), Fr::zero()]
                })
                .collect();
            assert_eq!(
                packed.rlc_evaluations_skipping(&weights, &padding).unwrap(),
                expected
            );
            assert_eq!(packed.rlc_evaluations(&weights).unwrap(), expected);
        }
    }
}
