//! Witness generation: the traced compressions of a schedule laid out as
//! 128-row cells, the wired columns materialized through the wiring table,
//! and the verifier-key columns.
//!
//! Cell positions: 112 half-G steps (`(round · 8 + g) · 2 + half`), 8 chaining
//! rows (`D' = v[j] ^ v[j + 8]` with `A' = v[j + 8]`), 2 challenge rows
//! (`D' = out[8 + 2i]`, `B' = out[9 + 2i]`, `A' = cv[2i]`, `C' = cv[2i + 1]`),
//! the next cell's block length and flags in `m` at positions 122 / 123, and
//! zero rows. `m` holds the block words on round-0 rows and a free balancing
//! word on the challenge rows; every other message use is the wired `m_in`.

use rayon::prelude::*;

use super::blake3::{Block, G_PER_ROUND, ROTATIONS, ROUNDS};
use super::layout::{
    WiredWord, WordColumn, A_OUT, B_XOR, CARRY_A_HI, CARRY_A_LO, CARRY_C, COMMITTED, C_OUT, D_XOR,
    MESSAGE, WORD_BITS,
};
use super::schedule::{CellIndex, CellPlan, JoltSchedule};
use super::wiring::{
    materialize, word, PublicInputs, VkColumns, CHAINING_POS, CHALLENGE_POS, CHALLENGE_ROWS,
    LOG_CELL, NEXT_FLAGS_POS, NEXT_LEN_POS,
};

/// The table: committed bit columns, wired columns, verifier-key columns and
/// public inputs, all over `2^log_rows` rows.
#[derive(Clone, Debug)]
pub struct HashTable {
    /// 163 committed columns.
    pub bits: Vec<Vec<u8>>,
    /// 64 wired bit columns (`din`, then `bin`).
    pub wired_bits: Vec<Vec<u8>>,
    /// The wired words, `WiredWord::ALL` order.
    pub wired_words: Vec<Vec<u32>>,
    pub vk: VkColumns,
    pub public: PublicInputs,
    pub log_rows: usize,
}

fn set_word(bits: &mut [Vec<u8>], base: usize, row: usize, word: u32) {
    for k in 0..WORD_BITS {
        bits[base + k][row] = ((word >> k) & 1) as u8;
    }
}

/// Fill the committed words of one cell from its traced compression; `next`
/// is the following cell's `(block_len, flags)`.
fn fill_cell(bits: &mut [Vec<u8>], base: usize, block: &Block, next: (u32, u32)) {
    let compression = &block.compression;
    for round in 0..ROUNDS {
        for g in 0..G_PER_ROUND {
            for half in 0..ROTATIONS.len() {
                let row = base + (round * G_PER_ROUND + g) * 2 + half;
                let step = &compression.steps[(round * G_PER_ROUND + g) * 2 + half];
                set_word(bits, A_OUT, row, step.a_out);
                set_word(bits, D_XOR, row, step.d_xor);
                set_word(bits, C_OUT, row, step.c_out);
                set_word(bits, B_XOR, row, step.b_xor);
            }
        }
    }
    for (w, word) in compression.block.iter().enumerate() {
        set_word(bits, MESSAGE, base + w, *word);
    }
    for j in 0..8 {
        let row = base + CHAINING_POS + j;
        set_word(bits, A_OUT, row, compression.v[j + 8]);
        set_word(bits, D_XOR, row, compression.out[j]);
    }
    for i in 0..CHALLENGE_ROWS {
        let row = base + CHALLENGE_POS + i;
        set_word(bits, A_OUT, row, compression.cv[2 * i]);
        set_word(bits, D_XOR, row, compression.out[8 + 2 * i]);
        set_word(bits, C_OUT, row, compression.cv[2 * i + 1]);
        set_word(bits, B_XOR, row, compression.out[9 + 2 * i]);
        // The free message word balances `A' = a_in + bin + m` (a_in = out[10]
        // on the first challenge row, 0 on the second; bin = v[9 + 2i]).
        let a_in = if i == 0 {
            compression.v[10] ^ compression.cv[2]
        } else {
            0
        };
        let m = compression.cv[2 * i]
            .wrapping_sub(a_in)
            .wrapping_sub(compression.v[9 + 2 * i]);
        set_word(bits, MESSAGE, row, m);
    }
    set_word(bits, MESSAGE, base + NEXT_LEN_POS, next.0);
    set_word(bits, MESSAGE, base + NEXT_FLAGS_POS, next.1);
}

impl HashTable {
    /// Lay out the schedule's cells (padding included) and materialize the
    /// wired columns through the wiring table.
    pub fn build(schedule: &JoltSchedule) -> Self {
        let blocks = schedule.table_blocks();
        let log_rows = schedule.symbolic.log_rows;
        let rows = 1usize << log_rows;
        let mut bits = vec![vec![0u8; rows]; COMMITTED];
        for (cell, block) in blocks.iter().enumerate() {
            let next = blocks.get(cell + 1).map_or(
                (CellPlan::PADDING.block_len, CellPlan::PADDING.flags),
                |next| (next.compression.block_len, next.compression.flags),
            );
            fill_cell(&mut bits, cell << LOG_CELL, block, next);
        }
        let public = schedule.public_inputs();
        let (wired_bits, wired_words) = materialize(&bits, &public);
        // Carries from the materialized operands (one owner of the add rule).
        let a_in = &wired_words[WiredWord::AIn.index()];
        let c_in = &wired_words[WiredWord::CIn.index()];
        let rot_d = &wired_words[WiredWord::RotD.index()];
        let m_in = &wired_words[WiredWord::MIn.index()];
        let carries: Vec<(u8, u8, u8)> = (0..rows)
            .into_par_iter()
            .map(|row| {
                let bin = (0..WORD_BITS).fold(0u64, |acc, k| {
                    acc | u64::from(wired_bits[WORD_BITS + k][row]) << k
                });
                let sum_a = u64::from(a_in[row]) + bin + u64::from(m_in[row]);
                let sum_c = u64::from(c_in[row]) + u64::from(rot_d[row]);
                let a_out = u64::from(word(&bits, WordColumn::AOut, row));
                let c_out = u64::from(word(&bits, WordColumn::COut, row));
                let carry_a = sum_a.wrapping_sub(a_out) >> 32;
                let carry_c = sum_c.wrapping_sub(c_out) >> 32;
                ((carry_a & 1) as u8, (carry_a >> 1) as u8, carry_c as u8)
            })
            .collect();
        for (row, (lo, hi, c)) in carries.into_iter().enumerate() {
            bits[CARRY_A_LO][row] = lo;
            bits[CARRY_A_HI][row] = hi;
            bits[CARRY_C][row] = c;
        }
        Self {
            bits,
            wired_bits,
            wired_words,
            vk: schedule.symbolic.vk_columns(),
            public,
            log_rows,
        }
    }

    pub fn rows(&self) -> usize {
        1 << self.log_rows
    }

    /// The rows holding the eight output words (`D'`) of a cell.
    pub fn chaining_rows(&self, cell: CellIndex) -> [usize; 8] {
        let base = cell.first_row() + CHAINING_POS;
        std::array::from_fn(|i| base + i)
    }

    /// The 32-bit word held in a committed column group of `row`.
    pub fn word(&self, column: WordColumn, row: usize) -> u32 {
        word(&self.bits, column, row)
    }
}
