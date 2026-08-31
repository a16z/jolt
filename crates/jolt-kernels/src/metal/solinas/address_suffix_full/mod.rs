use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};

use super::Fp128;

pub const ADDRESS_SUFFIX_BINS: usize = 256;
pub const ADDRESS_SUFFIX_TABLES: usize = LookupTableKind::<RISCV_XLEN>::COUNT;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressSuffixFullSums {
    values: Vec<Fp128>,
    table_offsets: Vec<usize>,
}

impl AddressSuffixFullSums {
    pub(super) fn from_values(values: Vec<Fp128>, table_offsets: Vec<usize>) -> Self {
        Self {
            values,
            table_offsets,
        }
    }

    pub fn as_flat_slice(&self) -> &[Fp128] {
        &self.values
    }

    pub fn table(&self, table: usize) -> Option<&[Fp128]> {
        let start = *self.table_offsets.get(table)? * ADDRESS_SUFFIX_BINS;
        let end = *self.table_offsets.get(table + 1)? * ADDRESS_SUFFIX_BINS;
        Some(&self.values[start..end])
    }

    pub fn suffix(&self, table: usize, suffix: usize) -> Option<&[Fp128]> {
        let table_start = *self.table_offsets.get(table)?;
        let table_end = *self.table_offsets.get(table + 1)?;
        (table_start + suffix < table_end).then(|| {
            let start = (table_start + suffix) * ADDRESS_SUFFIX_BINS;
            &self.values[start..start + ADDRESS_SUFFIX_BINS]
        })
    }
}
