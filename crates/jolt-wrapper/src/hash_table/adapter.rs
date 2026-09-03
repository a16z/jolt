use jolt_field::Fr;

use crate::stream::{
    AffineForm as StreamAffineForm, Column, ColumnId as StreamColumnId, Term as StreamTerm,
    TermContext as StreamTermContext, TermExporter, TermObserver,
};

use super::terms::{self, FinalContext};
use super::{HashTable, Relation, WiringStatement, WIRED_BITS, WIRED_WORDS};

pub struct StreamColumns {
    pub columns: Vec<Column>,
    pub ids: Vec<StreamColumnId>,
    pub group_count: usize,
}

impl StreamColumns {
    pub fn new(table: &HashTable, packing: usize, group_offset: usize) -> Self {
        assert!(packing.is_power_of_two());
        let rows = 1usize << table.log_rows;
        let mut columns = Vec::new();
        let mut ids = vec![StreamColumnId { group: 0, slot: 0 }; terms::COLUMNS];

        for (local, values) in table.bits.iter().chain(&table.wired_bits).enumerate() {
            ids[local] = id(group_offset, columns.len(), packing);
            columns.push(Column::Bits(values.clone()));
        }
        pad_bits(&mut columns, rows, packing);

        for (word, values) in table.wired_words.iter().enumerate() {
            ids[terms::WIRED_WORD_BASE + word] = id(group_offset, columns.len(), packing);
            columns.push(Column::U32(values.clone()));
        }
        pad_u32(&mut columns, rows, packing);

        for (local, values) in [&table.vk.lo_is_const, &table.vk.hi_is_const]
            .into_iter()
            .enumerate()
        {
            let column = terms::VK_BASE + 2 * local;
            ids[column] = id(group_offset, columns.len(), packing);
            columns.push(Column::Bits(values.clone()));
        }
        pad_bits(&mut columns, rows, packing);

        for (local, values) in [&table.vk.lo_const, &table.vk.hi_const]
            .into_iter()
            .enumerate()
        {
            let column = terms::VK_BASE + 2 * local + 1;
            ids[column] = id(group_offset, columns.len(), packing);
            columns.push(Column::U16(values.clone()));
        }
        pad_u16(&mut columns, rows, packing);

        let group_count = columns.len() / packing;
        debug_assert_eq!(table.bits.len() + table.wired_bits.len(), WIRED_BITS + 163);
        debug_assert_eq!(table.wired_words.len(), WIRED_WORDS);
        Self {
            columns,
            ids,
            group_count,
        }
    }
}

pub struct StreamTermExporter<'a> {
    pub relation: &'a Relation,
    pub wiring: &'a WiringStatement<'a>,
    pub tau_rows: &'a [Fr],
    pub tau_wiring: &'a [Fr],
    pub public: &'a super::PublicInputs,
    pub columns: &'a [StreamColumnId],
    pub row_member: usize,
    pub wiring_member: usize,
}

impl StreamTermExporter<'_> {
    fn export(&self, context: &StreamTermContext<'_>) -> Vec<StreamTerm> {
        let local = terms::terms(&FinalContext {
            relation: self.relation,
            wiring: self.wiring,
            tau_rows: self.tau_rows,
            tau_wiring: self.tau_wiring,
            challenges: context.row_point,
            rho_rows: context.batching_coefficients[self.row_member],
            rho_wiring: context.batching_coefficients[self.wiring_member],
            public: self.public,
        });
        local
            .into_iter()
            .map(|term| StreamTerm {
                coefficient: term.coefficient,
                factors: term
                    .factors
                    .into_iter()
                    .map(|form| StreamAffineForm {
                        constant: form.constant,
                        weights: form
                            .weights
                            .into_iter()
                            .map(|(column, weight)| (self.columns[column], weight))
                            .collect(),
                    })
                    .collect(),
            })
            .collect()
    }
}

impl TermExporter for StreamTermExporter<'_> {
    fn terms(&self, context: &StreamTermContext<'_>) -> Vec<StreamTerm> {
        self.export(context)
    }

    fn terms_observed(
        &self,
        context: &StreamTermContext<'_>,
        _observer: &mut dyn TermObserver,
    ) -> Vec<StreamTerm> {
        self.export(context)
    }
}

fn id(group_offset: usize, column: usize, packing: usize) -> StreamColumnId {
    StreamColumnId {
        group: group_offset + column / packing,
        slot: column % packing,
    }
}

fn pad_bits(columns: &mut Vec<Column>, rows: usize, packing: usize) {
    while !columns.len().is_multiple_of(packing) {
        columns.push(Column::Bits(vec![0; rows]));
    }
}

fn pad_u16(columns: &mut Vec<Column>, rows: usize, packing: usize) {
    while !columns.len().is_multiple_of(packing) {
        columns.push(Column::U16(vec![0; rows]));
    }
}

fn pad_u32(columns: &mut Vec<Column>, rows: usize, packing: usize) {
    while !columns.len().is_multiple_of(packing) {
        columns.push(Column::U32(vec![0; rows]));
    }
}
