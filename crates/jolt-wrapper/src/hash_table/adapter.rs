//! T1 on the wrapper stream: the committed column groups in packing order,
//! the members built from the stream's post-commitment challenges, and the
//! `TermExporter` mapping T1's local terms to the stream's physical ids.
//!
//! Protocol order (prover and verifier alike): commit T1's groups → the
//! stream draws `T1Challenges::count(log_rows)` challenges for the phase →
//! `T1Challenges::from_challenges` → row / wiring members (prover) and the
//! exporter's terms after stage A (both).

use jolt_field::Fr;

use crate::stream::{
    AffineForm as StreamAffineForm, Column, ColumnId as StreamColumnId, Term as StreamTerm,
    TermContext as StreamTermContext, TermExporter, TermObserver,
};

use super::terms::{self, FinalContext, T1Challenges, COLUMNS, VK_BASE, WIRED_WORD_BASE};
use super::{HashTable, HashTableProver, PublicInputs, Relation, WiringProver, WIRED_BITS};

/// T1's columns as stream groups: committed bits (state, carries, message,
/// canonicality witness, wired bits), the wired u32 words, then the
/// verifier-key columns (four bit selectors, two u16 constants).
pub struct StreamColumns {
    pub columns: Vec<Column>,
    /// Physical id of every local column id (`COLUMNS`).
    pub ids: Vec<StreamColumnId>,
    pub group_count: usize,
}

impl StreamColumns {
    /// # Panics
    ///
    /// Panics unless `packing` is a power of two.
    pub fn new(table: &HashTable, packing: usize, group_offset: usize) -> Self {
        assert!(packing.is_power_of_two());
        let rows = table.rows();
        let mut columns = Vec::new();
        let mut ids = vec![StreamColumnId { group: 0, slot: 0 }; COLUMNS];
        let mut push = |columns: &mut Vec<Column>, local: usize, column: Column| {
            ids[local] = id(group_offset, columns.len(), packing);
            columns.push(column);
        };
        for (local, values) in table.bits.iter().chain(&table.wired_bits).enumerate() {
            push(&mut columns, local, Column::Bits(values.clone()));
        }
        pad(&mut columns, packing, || Column::Bits(vec![0; rows]));
        for (word, values) in table.wired_words.iter().enumerate() {
            push(
                &mut columns,
                WIRED_WORD_BASE + word,
                Column::U32(values.clone()),
            );
        }
        pad(&mut columns, packing, || Column::U32(vec![0; rows]));
        let vk = &table.vk;
        for (local, values) in [
            (0, &vk.lo_is_const),
            (2, &vk.hi_is_const),
            (4, &vk.wire_aligned),
            (5, &vk.wire_shifted),
        ] {
            push(&mut columns, VK_BASE + local, Column::Bits(values.clone()));
        }
        pad(&mut columns, packing, || Column::Bits(vec![0; rows]));
        for (local, values) in [(1, &vk.lo_const), (3, &vk.hi_const)] {
            push(&mut columns, VK_BASE + local, Column::U16(values.clone()));
        }
        pad(&mut columns, packing, || Column::U16(vec![0; rows]));
        debug_assert_eq!(table.bits.len() + table.wired_bits.len(), WIRED_WORD_BASE);
        debug_assert_eq!(table.wired_bits.len(), WIRED_BITS);
        let group_count = columns.len() / packing;
        Self {
            columns,
            ids,
            group_count,
        }
    }
}

/// The prover's two stage-A members for one table and one draw of the
/// stream's challenges. `relation` outlives the row member it is lent to.
pub struct Members<'a> {
    pub rows: HashTableProver<'a>,
    pub wiring: WiringProver,
    pub input_claims: [Fr; 2],
}

impl<'a> Members<'a> {
    pub fn new(table: &'a HashTable, relation: &'a Relation, challenges: &T1Challenges) -> Self {
        let wiring_statement = challenges.wiring();
        let rows = HashTableProver::new(relation, table, challenges.tau_rows.clone());
        let wiring = WiringProver::new(
            &wiring_statement,
            &table.bits,
            &table.wired_bits,
            &table.wired_words,
            &table.vk,
            &table.public,
            challenges.tau_wiring.clone(),
        );
        Self {
            rows,
            wiring,
            input_claims: challenges.input_claims(&table.public),
        }
    }
}

/// T1's `TermExporter`: derives the members' randomizers from the phase
/// challenges the stream drew after T1's commitments and maps the local
/// terms to physical ids.
pub struct StreamTermExporter<'a> {
    pub log_rows: usize,
    /// Offset of T1's `T1Challenges::count(log_rows)` challenges in
    /// `TermContext::challenges`.
    pub challenge_offset: usize,
    pub public: &'a PublicInputs,
    pub columns: &'a [StreamColumnId],
    pub row_member: usize,
    pub wiring_member: usize,
}

impl StreamTermExporter<'_> {
    fn export(
        &self,
        context: &StreamTermContext<'_>,
        mul: &mut dyn FnMut(Fr, Fr) -> Fr,
    ) -> Vec<StreamTerm> {
        let count = T1Challenges::count(self.log_rows);
        let challenges = T1Challenges::from_challenges(
            &context.challenges[self.challenge_offset..self.challenge_offset + count],
            self.log_rows,
        );
        let local = terms::terms(
            &FinalContext {
                challenges: &challenges,
                row_point: context.row_point,
                rho_rows: context.batching_coefficients[self.row_member],
                rho_wiring: context.batching_coefficients[self.wiring_member],
                public: self.public,
            },
            mul,
        );
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
        self.export(context, &mut |a, b| a * b)
    }

    fn terms_observed(
        &self,
        context: &StreamTermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<StreamTerm> {
        self.export(context, &mut |a, b| observer.fr_mul(a, b))
    }
}

fn id(group_offset: usize, column: usize, packing: usize) -> StreamColumnId {
    StreamColumnId {
        group: group_offset + column / packing,
        slot: column % packing,
    }
}

fn pad(columns: &mut Vec<Column>, packing: usize, zero: impl Fn() -> Column) {
    while !columns.len().is_multiple_of(packing) {
        columns.push(zero());
    }
}
