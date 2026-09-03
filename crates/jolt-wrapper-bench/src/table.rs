//! Random witness tables in a layout's shape (the relation is not satisfied;
//! the sumcheck proves whatever the sum is, which is what the cost model needs).

use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;

use crate::relation::Layout;

pub struct Table {
    /// Committed bit columns, column-space indices `0..committed`.
    pub bits: Vec<Vec<u8>>,
    /// Wired bit columns in `Layout::wired_bits` order.
    pub wired_bits: Vec<Vec<u8>>,
    /// Wired 32-bit word columns in `Layout::wired_ints` order.
    pub wired_ints: Vec<Vec<u64>>,
}

impl Table {
    pub fn random(layout: Layout, log_rows: usize, seed: u64) -> Self {
        let rows = 1usize << log_rows;
        let bit_columns = |count: usize, salt: u64| -> Vec<Vec<u8>> {
            (0..count)
                .into_par_iter()
                .map(|c| {
                    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ salt ^ (c as u64) << 8);
                    (0..rows).map(|_| (rng.next_u32() & 1) as u8).collect()
                })
                .collect()
        };
        let wired_ints = (0..layout.wired_ints().len())
            .into_par_iter()
            .map(|c| {
                let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xdead ^ (c as u64) << 8);
                (0..rows).map(|_| u64::from(rng.next_u32())).collect()
            })
            .collect();
        Self {
            bits: bit_columns(layout.committed(), 0x1111),
            wired_bits: bit_columns(layout.wired_bits().len(), 0x2222),
            wired_ints,
        }
    }
}
