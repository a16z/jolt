//! Witness generation: the traced compressions of a transcript segment laid
//! out as rows, the wiring every wired column is copied through, and the link
//! table from message bytes to their external sources.
//!
//! Rows per compression: 112 half-G steps (index `(round·8 + g)·2 + half`),
//! 8 chaining rows materializing `out[i] = v[i] ^ v[i + 8]` in `D'`, and — for
//! a squeeze — 4 challenge rows materializing `out[8 + i] = v[8 + i] ^ cv[i]`.
//! A chaining/challenge row is a half-G row with `bin = m = c_in = rot_d = 0`:
//! the ternary add copies `a_in` into `A'` and the first XOR produces the
//! output word.

use std::ops::Range;

use super::blake3::{
    last_writer, schedule, Block, ByteOrigin, G_INDICES, G_PER_ROUND, HALF_STEPS, IV, ROTATIONS,
    ROUNDS,
};
use super::layout::{
    WordColumn, A_OUT, B_XOR, CARRY_A_HI, CARRY_A_LO, CARRY_C, COMMITTED, C_OUT, D_XOR, MESSAGE,
    WIRED_BITS, WIRED_WORDS, WORD_BITS,
};

pub const CHAINING_ROWS: usize = 8;
pub const CHALLENGE_ROWS: usize = 4;
/// Rows of a compression without / with a squeeze.
pub const ROWS_PER_BLOCK: usize = HALF_STEPS + CHAINING_ROWS;
pub const ROWS_PER_SQUEEZE_BLOCK: usize = ROWS_PER_BLOCK + CHALLENGE_ROWS;

/// Where a wired input of a row comes from. A `Word` feed reads the 32-bit
/// word held in a committed column group of another (or the same) row,
/// rotated right by `rot`: bit `k` of the input is bit `(k + rot) mod 32` of
/// the source word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feed {
    Zero,
    Word {
        column: WordColumn,
        row: u32,
        rot: u8,
    },
    /// A profile constant (IV word, block length, flags, counter).
    Const(u32),
    /// Word `i` of the public-input state the first compression chains from.
    StateIn(u8),
}

/// The five wired inputs of a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowFeeds {
    pub din: Feed,
    pub bin: Feed,
    pub a_in: Feed,
    pub c_in: Feed,
    pub rot_d: Feed,
}

const ZERO_FEEDS: RowFeeds = RowFeeds {
    din: Feed::Zero,
    bin: Feed::Zero,
    a_in: Feed::Zero,
    c_in: Feed::Zero,
    rot_d: Feed::Zero,
};

/// Where a row's committed message word comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageSource {
    /// Zero (chaining, challenge and padding rows).
    None,
    /// The block word's first use (round 0); its bytes are in the link table.
    First,
    /// A later use of the block word first used in `row`.
    Copy { row: u32 },
}

/// Byte `byte` of the message word of first-use row `row` is absorbed byte
/// `origin` (`None`: zero padding of a partial final block).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MessageLink {
    pub row: u32,
    pub byte: u8,
    pub origin: Option<ByteOrigin>,
}

/// The 16 challenge bytes of a squeeze are the `D'` words of `rows` (word
/// `i` = bytes `4i..4i + 4`, little-endian).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChallengeLink {
    pub item: u32,
    pub rows: [u32; CHALLENGE_ROWS],
}

/// The table: committed bit columns, wired columns, wiring and links.
#[derive(Clone, Debug)]
pub struct HashTable {
    /// 163 committed columns of `2^log_rows` bits.
    pub bits: Vec<Vec<u8>>,
    /// 64 wired bit columns (`din`, then `bin`).
    pub wired_bits: Vec<Vec<u8>>,
    /// 3 wired word columns (`a_in`, `c_in`, `rot_d`).
    pub wired_words: Vec<Vec<u32>>,
    pub feeds: Vec<RowFeeds>,
    pub message_sources: Vec<MessageSource>,
    pub links: Vec<MessageLink>,
    pub challenges: Vec<ChallengeLink>,
    /// The chaining value the first compression starts from (public input).
    pub state_in: [u32; 8],
    /// First row of each compression.
    pub block_rows: Vec<usize>,
    pub rows: usize,
    pub log_rows: usize,
}

struct Builder {
    bits: Vec<Vec<u8>>,
    feeds: Vec<RowFeeds>,
    message_sources: Vec<MessageSource>,
    links: Vec<MessageLink>,
    challenges: Vec<ChallengeLink>,
}

fn set_word(bits: &mut [Vec<u8>], base: usize, row: usize, word: u32) {
    for k in 0..WORD_BITS {
        bits[base + k][row] = ((word >> k) & 1) as u8;
    }
}

impl Builder {
    /// Row of half `half` of step `g` of `round` within a compression at `base`.
    fn step_row(base: usize, round: usize, g: usize, half: usize) -> usize {
        base + (round * G_PER_ROUND + g) * 2 + half
    }

    /// Feed reading state index `index` as the first half of step `g` of
    /// `round` sees it: a diagonal step reads the column step of the same
    /// round that wrote the index (second half); a column step reads the
    /// diagonal step of the previous round. `a`/`c` words are add outputs,
    /// `b`/`d` words are rotated XOR outputs. Round 0 column steps read the
    /// initial state (`None`).
    fn state_word(base: usize, round: usize, g: usize, index: usize) -> Option<Feed> {
        let row = if g >= G_PER_ROUND / 2 {
            Self::step_row(base, round, index % 4, 1)
        } else if round == 0 {
            return None;
        } else {
            Self::step_row(base, round - 1, last_writer(index), 1)
        };
        Some(Self::word_feed(row, index))
    }

    /// Feed reading state index `index` as left by the second half of step
    /// row `row`.
    fn word_feed(row: usize, index: usize) -> Feed {
        let row = row as u32;
        let (rot_d, rot_b) = ROTATIONS[1];
        match index / 4 {
            0 => Feed::Word {
                column: WordColumn::AOut,
                row,
                rot: 0,
            },
            1 => Feed::Word {
                column: WordColumn::BXor,
                row,
                rot: rot_b as u8,
            },
            2 => Feed::Word {
                column: WordColumn::COut,
                row,
                rot: 0,
            },
            _ => Feed::Word {
                column: WordColumn::DXor,
                row,
                rot: rot_d as u8,
            },
        }
    }

    /// Feed reading state index `index` of the final state (after round 7).
    fn final_state(base: usize, index: usize) -> Feed {
        Self::word_feed(
            Self::step_row(base, ROUNDS - 1, last_writer(index), 1),
            index,
        )
    }

    /// Feed reading word `i` of this compression's chaining value: the
    /// previous compression's chaining row, or the public input for the first.
    fn chaining_value(previous_base: Option<usize>, i: usize) -> Feed {
        match previous_base {
            Some(base) => Feed::Word {
                column: WordColumn::DXor,
                row: (base + HALF_STEPS + i) as u32,
                rot: 0,
            },
            None => Feed::StateIn(i as u8),
        }
    }

    fn push_block(&mut self, block: &Block, base: usize, previous_base: Option<usize>) {
        let compression = &block.compression;
        for round in 0..ROUNDS {
            for (g, [a, b, c, d]) in G_INDICES.iter().copied().enumerate() {
                for (half, &(rot_d, _)) in ROTATIONS.iter().enumerate() {
                    let row = Self::step_row(base, round, g, half);
                    let step = &compression.steps[(round * G_PER_ROUND + g) * 2 + half];
                    set_word(&mut self.bits, A_OUT, row, step.a_out);
                    set_word(&mut self.bits, D_XOR, row, step.d_xor);
                    set_word(&mut self.bits, C_OUT, row, step.c_out);
                    set_word(&mut self.bits, B_XOR, row, step.b_xor);
                    self.bits[CARRY_A_LO][row] = (step.a_carry & 1) as u8;
                    self.bits[CARRY_A_HI][row] = (step.a_carry >> 1) as u8;
                    self.bits[CARRY_C][row] = step.c_carry as u8;
                    set_word(&mut self.bits, MESSAGE, row, step.m);
                    let rot_d_feed = Feed::Word {
                        column: WordColumn::DXor,
                        row: row as u32,
                        rot: rot_d as u8,
                    };
                    let (first_rot_d, first_rot_b) = ROTATIONS[0];
                    self.feeds[row] = if half == 1 {
                        let previous = (row - 1) as u32;
                        RowFeeds {
                            din: Feed::Word {
                                column: WordColumn::DXor,
                                row: previous,
                                rot: first_rot_d as u8,
                            },
                            bin: Feed::Word {
                                column: WordColumn::BXor,
                                row: previous,
                                rot: first_rot_b as u8,
                            },
                            a_in: Feed::Word {
                                column: WordColumn::AOut,
                                row: previous,
                                rot: 0,
                            },
                            c_in: Feed::Word {
                                column: WordColumn::COut,
                                row: previous,
                                rot: 0,
                            },
                            rot_d: rot_d_feed,
                        }
                    } else {
                        // Initial state: cv, IV[0..4], counter (0, 0), block length, flags.
                        let input = |index: usize| {
                            Self::state_word(base, round, g, index).unwrap_or(match index {
                                0..=7 => Self::chaining_value(previous_base, index),
                                8..=11 => Feed::Const(IV[index - 8]),
                                12 | 13 => Feed::Const(0),
                                14 => Feed::Const(compression.block_len),
                                _ => Feed::Const(compression.flags),
                            })
                        };
                        RowFeeds {
                            din: input(d),
                            bin: input(b),
                            a_in: input(a),
                            c_in: input(c),
                            rot_d: rot_d_feed,
                        }
                    };
                    let word = schedule(round, 2 * g + half);
                    self.message_sources[row] = if round == 0 {
                        for byte in 0..4 {
                            self.links.push(MessageLink {
                                row: row as u32,
                                byte: byte as u8,
                                origin: block.origins[4 * word + byte],
                            });
                        }
                        MessageSource::First
                    } else {
                        MessageSource::Copy {
                            row: (base + word) as u32,
                        }
                    };
                }
            }
        }
        // Chaining rows: D' = v[j] ^ v[j + 8] with A' = v[j + 8] copied through a_in.
        for j in 0..CHAINING_ROWS {
            let row = base + HALF_STEPS + j;
            let high = compression.v[j + 8];
            set_word(&mut self.bits, A_OUT, row, high);
            set_word(&mut self.bits, D_XOR, row, compression.v[j] ^ high);
            self.feeds[row] = RowFeeds {
                din: Self::final_state(base, j),
                a_in: Self::final_state(base, j + 8),
                ..ZERO_FEEDS
            };
        }
        if let Some(item) = block.squeeze {
            let mut rows = [0u32; CHALLENGE_ROWS];
            for (i, slot) in rows.iter_mut().enumerate() {
                let row = base + ROWS_PER_BLOCK + i;
                *slot = row as u32;
                set_word(&mut self.bits, A_OUT, row, compression.cv[i]);
                set_word(
                    &mut self.bits,
                    D_XOR,
                    row,
                    compression.v[8 + i] ^ compression.cv[i],
                );
                self.feeds[row] = RowFeeds {
                    din: Self::final_state(base, 8 + i),
                    a_in: Self::chaining_value(previous_base, i),
                    ..ZERO_FEEDS
                };
            }
            self.challenges.push(ChallengeLink { item, rows });
        }
    }
}

/// Rows the compressions in `blocks` occupy.
pub fn row_count(blocks: &[Block]) -> usize {
    blocks
        .iter()
        .map(|block| {
            if block.squeeze.is_some() {
                ROWS_PER_SQUEEZE_BLOCK
            } else {
                ROWS_PER_BLOCK
            }
        })
        .sum()
}

impl HashTable {
    /// Lay out `chain[blocks]` as a table of `2^log_rows` rows
    /// (`log_rows = None`: the smallest power of two that fits).
    ///
    /// # Panics
    ///
    /// Panics if the rows do not fit `2^log_rows`.
    pub fn build(chain: &[Block], blocks: Range<usize>, log_rows: Option<usize>) -> Self {
        let selected = &chain[blocks];
        let rows = row_count(selected);
        let log_rows = log_rows.unwrap_or_else(|| rows.next_power_of_two().ilog2() as usize);
        let padded = 1usize << log_rows;
        assert!(rows <= padded, "{rows} rows do not fit 2^{log_rows}");
        let mut builder = Builder {
            bits: vec![vec![0u8; padded]; COMMITTED],
            feeds: vec![ZERO_FEEDS; padded],
            message_sources: vec![MessageSource::None; padded],
            links: Vec::with_capacity(selected.len() * 64),
            challenges: Vec::new(),
        };
        let mut block_rows = Vec::with_capacity(selected.len());
        let mut base = 0;
        let mut previous_base = None;
        for block in selected {
            block_rows.push(base);
            builder.push_block(block, base, previous_base);
            previous_base = Some(base);
            base += if block.squeeze.is_some() {
                ROWS_PER_SQUEEZE_BLOCK
            } else {
                ROWS_PER_BLOCK
            };
        }
        let state_in = selected
            .first()
            .map_or([0; 8], |block| block.compression.cv);
        let mut table = Self {
            bits: builder.bits,
            wired_bits: vec![vec![0u8; padded]; WIRED_BITS],
            wired_words: vec![vec![0u32; padded]; WIRED_WORDS],
            feeds: builder.feeds,
            message_sources: builder.message_sources,
            links: builder.links,
            challenges: builder.challenges,
            state_in,
            block_rows,
            rows,
            log_rows,
        };
        table.materialize_wired();
        table
    }

    /// Bit `k` of a feed's 32-bit value.
    pub fn feed_bit(&self, feed: Feed, k: usize) -> u8 {
        match feed {
            Feed::Zero => 0,
            Feed::Word { column, row, rot } => {
                self.bits[column.base() + (k + usize::from(rot)) % WORD_BITS][row as usize]
            }
            Feed::Const(word) => ((word >> k) & 1) as u8,
            Feed::StateIn(i) => ((self.state_in[usize::from(i)] >> k) & 1) as u8,
        }
    }

    pub fn feed_word(&self, feed: Feed) -> u32 {
        (0..WORD_BITS).fold(0, |acc, k| acc | u32::from(self.feed_bit(feed, k)) << k)
    }

    /// Fill the wired columns from the committed columns through the feeds.
    fn materialize_wired(&mut self) {
        let padded = 1usize << self.log_rows;
        let mut wired_bits = vec![vec![0u8; padded]; WIRED_BITS];
        let mut wired_words = vec![vec![0u32; padded]; WIRED_WORDS];
        for row in 0..padded {
            let feeds = self.feeds[row];
            for k in 0..WORD_BITS {
                wired_bits[k][row] = self.feed_bit(feeds.din, k);
                wired_bits[WORD_BITS + k][row] = self.feed_bit(feeds.bin, k);
            }
            wired_words[0][row] = self.feed_word(feeds.a_in);
            wired_words[1][row] = self.feed_word(feeds.c_in);
            wired_words[2][row] = self.feed_word(feeds.rot_d);
        }
        self.wired_bits = wired_bits;
        self.wired_words = wired_words;
    }

    /// The rows holding the eight output words (`D'`) of compression `block`.
    pub fn chaining_rows(&self, block: usize) -> [usize; CHAINING_ROWS] {
        let base = self.block_rows[block] + HALF_STEPS;
        std::array::from_fn(|i| base + i)
    }

    /// The 32-bit word held in a committed column group of `row`.
    pub fn word(&self, column: WordColumn, row: usize) -> u32 {
        (0..WORD_BITS).fold(0, |acc, k| {
            acc | u32::from(self.bits[column.base() + k][row]) << k
        })
    }
}
