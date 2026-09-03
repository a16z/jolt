//! T1 on the wrapper stream: the verifier key (schedule + the verifier-key
//! column groups committed once), the prover's column groups in packing
//! order, the members built from the stream's post-commitment challenges,
//! and the `TermExporter` mapping T1's local terms to physical ids.
//!
//! Protocol order (prover and verifier alike): commit T1's prover groups →
//! the stream draws `T1Challenges::count(log_rows)` challenges for the phase
//! → `T1Challenges::from_challenges` → row / wiring members (prover) and the
//! exporter's terms after stage A (both). The six verifier-key columns
//! (`VkColumn::ALL`) are the last groups of T1's block and verifier-key data:
//! `HashTableKey::pinned_commitments` go into
//! `AssemblyStatement::pinned_commitments`, proofs omit them and the verifier
//! opens against the key's commitments.

use std::ops::Range;

use jolt_crypto::Bn254;
use jolt_field::Fr;
use jolt_hyperkzg::HyperKZGProverSetup;

use crate::stream::{
    commit_packed, AffineForm as StreamAffineForm, Column, ColumnId as StreamColumnId, Commitment,
    StreamError, Term as StreamTerm, TermContext as StreamTermContext, TermExporter, TermObserver,
};

use super::eq::plain;
use super::schedule::SymbolicSchedule;
use super::terms::{self, FinalContext, T1Challenges, COLUMNS, VK_BASE, WIRED_WORD_BASE};
use super::wiring::VkColumns;
use super::{
    HashTable, HashTableProver, PublicInputs, Relation, WiringProver, WIRED_BITS, WIRED_WORDS,
};

/// The verifier-key columns as stream groups (bit selectors, then the u16
/// constants), each column tagged with its local id (`None` = zero padding).
fn vk_group_columns(vk: &VkColumns, packing: usize) -> Vec<(Option<usize>, Column)> {
    let rows = vk.lo_is_const.len();
    let mut columns = Vec::new();
    for (local, values) in [
        (0, &vk.lo_is_const),
        (2, &vk.hi_is_const),
        (4, &vk.wire_aligned),
        (5, &vk.wire_shifted),
    ] {
        columns.push((Some(VK_BASE + local), Column::Bits(values.clone())));
    }
    pad(&mut columns, packing, || Column::Bits(vec![0; rows]));
    for (local, values) in [(1, &vk.lo_const), (3, &vk.hi_const)] {
        columns.push((Some(VK_BASE + local), Column::U16(values.clone())));
    }
    pad(&mut columns, packing, || Column::U16(vec![0; rows]));
    columns
}

/// Groups of T1's prover-committed columns: the bit columns (committed and
/// wired), then the wired u32 words.
pub fn prover_group_count(packing: usize) -> usize {
    WIRED_WORD_BASE.div_ceil(packing) + WIRED_WORDS.div_ceil(packing)
}

/// Absolute group indices of T1's verifier-key groups when T1's block starts
/// at `group_offset`.
pub fn vk_group_range(packing: usize, group_offset: usize) -> Range<usize> {
    let start = group_offset + prover_group_count(packing);
    start..start + 4usize.div_ceil(packing) + 2usize.div_ceil(packing)
}

/// T1's verifier key: the symbolic schedule and the verifier-key column
/// groups committed once from it. Immutable after construction: the
/// commitments are exactly the `vk_group_range(packing, 0)` groups of
/// `schedule.vk_columns()` (checked in `new`), so every pin extraction covers
/// every verifier-key group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashTableKey {
    schedule: SymbolicSchedule,
    vk: VkColumns,
    packing: usize,
    commitments: Vec<Commitment>,
}

impl HashTableKey {
    pub fn new(
        schedule: SymbolicSchedule,
        packing: usize,
        setup: &HyperKZGProverSetup<Bn254>,
    ) -> Result<Self, StreamError> {
        let vk = schedule.vk_columns();
        let columns: Vec<Column> = vk_group_columns(&vk, packing)
            .into_iter()
            .map(|(_, column)| column)
            .collect();
        let packed = commit_packed(&columns, packing, setup)?;
        if packed.commitments.len() != vk_group_range(packing, 0).len() {
            return Err(StreamError::StageCount);
        }
        Ok(Self {
            schedule,
            vk,
            packing,
            commitments: packed.commitments,
        })
    }

    pub fn schedule(&self) -> &SymbolicSchedule {
        &self.schedule
    }

    pub fn vk(&self) -> &VkColumns {
        &self.vk
    }

    pub fn packing(&self) -> usize {
        self.packing
    }

    /// Commitments of the verifier-key groups (`vk_group_range` order).
    pub fn commitments(&self) -> &[Commitment] {
        &self.commitments
    }

    /// `(group index, commitment)` of every verifier-key group when T1's block
    /// starts at `group_offset` — the wrapper key pins the proof's commitments
    /// at these indices to these values.
    pub fn pinned_commitments(&self, group_offset: usize) -> Vec<(usize, Commitment)> {
        vk_group_range(self.packing, group_offset)
            .zip(self.commitments.iter().copied())
            .collect()
    }

    pub(crate) fn column_ids(&self, group_offset: usize) -> Vec<StreamColumnId> {
        let mut ids = vec![StreamColumnId { group: 0, slot: 0 }; COLUMNS];
        for (local, target) in ids.iter_mut().take(WIRED_WORD_BASE).enumerate() {
            *target = id(group_offset, local, self.packing);
        }
        let word_base = WIRED_WORD_BASE.div_ceil(self.packing) * self.packing;
        for word in 0..WIRED_WORDS {
            ids[WIRED_WORD_BASE + word] = id(group_offset, word_base + word, self.packing);
        }
        let vk_base = prover_group_count(self.packing) * self.packing;
        for (offset, (local, _)) in vk_group_columns(&self.vk, self.packing)
            .into_iter()
            .enumerate()
        {
            if let Some(local) = local {
                ids[local] = id(group_offset, vk_base + offset, self.packing);
            }
        }
        ids
    }
}

/// T1's columns as stream groups: committed bits (state, carries, message,
/// canonicality witness, wired bits), the wired u32 words, then the
/// verifier-key groups (from the table's copy of the key columns; the key
/// pins their commitments).
pub struct StreamColumns {
    pub columns: Vec<Column>,
    /// Physical id of every local column id (`terms::COLUMNS`).
    pub ids: Vec<StreamColumnId>,
    pub group_count: usize,
    /// Absolute indices of the verifier-key groups.
    pub vk_groups: Range<usize>,
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
        pad_plain(&mut columns, packing, || Column::Bits(vec![0; rows]));
        for (word, values) in table.wired_words.iter().enumerate() {
            push(
                &mut columns,
                WIRED_WORD_BASE + word,
                Column::U32(values.clone()),
            );
        }
        pad_plain(&mut columns, packing, || Column::U32(vec![0; rows]));
        debug_assert_eq!(columns.len() / packing, prover_group_count(packing));
        debug_assert_eq!(table.wired_bits.len(), WIRED_BITS);
        for (local, column) in vk_group_columns(&table.vk, packing) {
            match local {
                Some(local) => push(&mut columns, local, column),
                None => columns.push(column),
            }
        }
        let group_count = columns.len() / packing;
        let vk_groups = vk_group_range(packing, group_offset);
        debug_assert_eq!(vk_groups.end, group_offset + group_count);
        Self {
            columns,
            ids,
            group_count,
            vk_groups,
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

/// T1's `TermExporter` and the verifier's T1 statement: derives the members'
/// randomizers from the phase challenges the stream drew after T1's
/// commitments, their input claims (`input_claims`) and the local terms mapped
/// to physical ids. The observed variants route every field multiplication
/// of the derivation through the observer.
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
    /// The verifier's two member input claims (row relation `0`, wiring public
    /// constant) from the phase challenges, every multiplication observed —
    /// the verifier's statement derivation, to be counted with the stream's
    /// `VerifierCost`.
    pub fn input_claims(&self, challenges: &[Fr], observer: &mut dyn TermObserver) -> [Fr; 2] {
        let count = T1Challenges::count(self.log_rows);
        let mut mul = |a, b| observer.fr_mul(a, b);
        T1Challenges::from_challenges_with(
            &challenges[self.challenge_offset..self.challenge_offset + count],
            self.log_rows,
            &mut mul,
        )
        .input_claims_with(self.public, &mut mul)
    }

    fn export(
        &self,
        context: &StreamTermContext<'_>,
        mul: &mut dyn FnMut(Fr, Fr) -> Fr,
    ) -> Vec<StreamTerm> {
        let count = T1Challenges::count(self.log_rows);
        let challenges = T1Challenges::from_challenges_with(
            &context.challenges[self.challenge_offset..self.challenge_offset + count],
            self.log_rows,
            mul,
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
    fn max_factors(&self) -> usize {
        2
    }

    fn terms(&self, context: &StreamTermContext<'_>) -> Vec<StreamTerm> {
        self.export(context, &mut plain)
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

fn pad(columns: &mut Vec<(Option<usize>, Column)>, packing: usize, zero: impl Fn() -> Column) {
    while !columns.len().is_multiple_of(packing) {
        columns.push((None, zero()));
    }
}

fn pad_plain(columns: &mut Vec<Column>, packing: usize, zero: impl Fn() -> Column) {
    while !columns.len().is_multiple_of(packing) {
        columns.push(zero());
    }
}
