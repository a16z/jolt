/// Types usable as one-hot position indices.
///
/// Implemented for `u8`, `u16`, `u32`, and `usize`.
pub trait OneHotIndex: Copy + Send + Sync + std::fmt::Debug + 'static {
    /// Convert to `usize` for indexing.
    fn as_usize(self) -> usize;
}

impl OneHotIndex for u8 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for u16 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for u32 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for usize {
    #[inline]
    fn as_usize(self) -> usize {
        self
    }
}

/// Entry semantics shared by the one-hot kernel modules.
pub(crate) trait OneHotEntry: Sync {
    fn pos_in_block(&self) -> usize;

    fn coeffs(&self) -> &[u16];

    #[inline(always)]
    fn commit_col(&self, num_digits: usize) -> usize {
        self.pos_in_block() * num_digits
    }
}

/// Compact record for a single nonzero ring element in the
/// single-chunk layout.
///
/// In the single-chunk layout each ring element overlaps at most one
/// one-hot chunk, so the ring has exactly one hot coefficient (value 1)
/// and `D - 1` zero coefficients. We store nothing about the zero
/// rings and nothing about the zero coefficients of the nonzero ring;
/// the entry just pins down *which* ring element we are talking about
/// (`pos_in_block`, inside the flat per-block layout) and *which* of
/// its `D` coefficients is the hot one (`coeff_idx`).
///
/// This layout applies when `K >= D && D | K`: one one-hot chunk spans
/// `K/D` consecutive ring elements, so every ring element falls
/// entirely inside one chunk and hence contains at most one hot
/// coefficient.
///
/// # Example
///
/// Take `K = 64`, `D = 32`, and look at the first chunk. Its flat
/// field-position range is `[0, 64)`; it contributes to ring elements
/// `0` (coefficients at positions `[0, 32)`) and `1` (positions
/// `[32, 64)`). Say the hot position inside this chunk is 60, so
/// field position 60 is 1 and all other positions in `[0, 64)` are 0.
/// Then:
///
/// - `ring_idx = 60 / 32 = 1` (ring element 0 has no hot coefficient
///   and is skipped entirely; ring element 1 carries the hot one);
/// - `coeff_idx = 60 % 32 = 28`.
///
/// If that ring lives in the first block of the flat layout,
/// `pos_in_block = 1` (the second ring element of block 0). The stored
/// entry is `SingleChunkEntry { pos_in_block: 1, coeff_idx: 28 }`, and
/// no entry is emitted for ring 0.
///
/// # Invariants
///
/// Fields are private and accessed via public accessors or the
/// internal `OneHotEntry` trait. The caller-owned invariants
/// `pos_in_block < block_len <= u32::MAX` and `coeff_idx < D <= 65536` are pre-validated in
/// `FlatBlocks::<SingleChunkEntry>::from_indices`; the
/// constructor just stores the already-narrowed fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SingleChunkEntry {
    pos_in_block: u32,
    coeff_idx: u16,
}

impl SingleChunkEntry {
    /// Construct a single-chunk entry from already-validated native-width fields.
    #[inline]
    pub(crate) fn new(pos_in_block: u32, coeff_idx: u16) -> Self {
        Self {
            pos_in_block,
            coeff_idx,
        }
    }

    /// Position within the block (0..block_len).
    #[inline]
    pub fn pos_in_block(self) -> usize {
        self.pos_in_block as usize
    }

    /// Index of the single hot coefficient inside the ring element (0..D).
    #[inline]
    pub fn coeff_idx(self) -> usize {
        self.coeff_idx as usize
    }
}

impl OneHotEntry for SingleChunkEntry {
    #[inline(always)]
    fn pos_in_block(&self) -> usize {
        self.pos_in_block as usize
    }

    #[inline(always)]
    fn coeffs(&self) -> &[u16] {
        std::slice::from_ref(&self.coeff_idx)
    }
}

/// Compact record for a single nonzero ring element in the
/// multi-chunk layout.
///
/// In the multi-chunk layout one ring element spans exactly `D/K`
/// whole consecutive one-hot chunks, so the ring can carry anywhere
/// from zero to `D/K` hot coefficients. We only emit an entry for
/// rings that have at least one, and within that entry we store
/// exactly which coefficients are hot (`nonzero_coeffs`) and where
/// the ring lives in the flat per-block layout (`pos_in_block`).
/// Everything else about the ring (its zero coefficients, its
/// neighbouring zero rings) is left implicit.
///
/// This layout applies when `K < D` with `K | D`: each ring element
/// contains exactly `D/K` whole consecutive chunks, each contributing
/// at most one hot coefficient to that ring.
///
/// # Worked example
///
/// Take `K = 8`, `D = 32`, so each ring element covers `D/K = 4`
/// consecutive chunks. Look at ring element 0, whose flat
/// field-position range is `[0, 32)` — chunks 0, 1, 2, 3 live inside
/// it:
///
/// - chunk 0 (field positions `[0, 8)`): hot at chunk-local index 3,
///   i.e. field position 3 → contributes `coeff_idx = 3`;
/// - chunk 1 (positions `[8, 16)`): all zero, contributes nothing;
/// - chunk 2 (positions `[16, 24)`): hot at chunk-local index 5, i.e.
///   field position 21 → contributes `coeff_idx = 21`;
/// - chunk 3 (positions `[24, 32)`): all zero, contributes nothing.
///
/// `coeff_idx` for a ring is just `field_pos % D` — the chunk boundary
/// doesn't enter the computation once we've landed inside the ring. If
/// this ring sits at position 0 in its block, the stored entry is
/// `MultiChunkEntry { pos_in_block: 0, nonzero_coeffs: [3, 21] }`. No
/// entry is emitted for rings whose four covering chunks are all zero.
///
/// # Why this representation
///
/// As with [`SingleChunkEntry`], we pay nothing for the zero rings and
/// nothing for the zero coefficients of the nonzero rings, so memory
/// stays proportional to the number of distinct nonzero rings and the
/// kernels skip the zeros on the hot path.
///
/// # Invariants
///
/// Fields are private and accessed via public accessors or the
/// internal `OneHotEntry` trait. The caller-owned invariants
/// `pos_in_block < block_len <= u32::MAX` and every `coeff < D <= 65536` are pre-validated in
/// `FlatBlocks::<MultiChunkEntry>::from_indices`; the
/// constructor just stores the already-narrowed fields.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiChunkEntry {
    pos_in_block: u32,
    nonzero_coeffs: Vec<u16>,
}

impl MultiChunkEntry {
    /// Construct a multi-chunk entry from already-validated native-width
    /// fields.
    #[inline]
    pub(crate) fn new(pos_in_block: u32, nonzero_coeffs: Vec<u16>) -> Self {
        Self {
            pos_in_block,
            nonzero_coeffs,
        }
    }

    /// Position within the block (0..block_len).
    #[inline]
    pub fn pos_in_block(&self) -> usize {
        self.pos_in_block as usize
    }

    /// Hot coefficient indices inside the ring element, each `< D`.
    #[inline]
    pub fn nonzero_coeffs(&self) -> &[u16] {
        &self.nonzero_coeffs
    }
}

impl OneHotEntry for MultiChunkEntry {
    #[inline(always)]
    fn pos_in_block(&self) -> usize {
        self.pos_in_block as usize
    }

    #[inline(always)]
    fn coeffs(&self) -> &[u16] {
        &self.nonzero_coeffs
    }
}

#[inline]
pub(crate) fn shift_accumulation_count<E: OneHotEntry>(entries: &[E]) -> usize {
    entries.iter().map(|entry| entry.coeffs().len()).sum()
}
