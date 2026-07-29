use crate::instruction::Instruction;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

const EMPTY: u32 = u32::MAX;

/// Pre-decoded instruction cache over the executable address range.
///
/// Decoding is a pure function of `(word, pc, is_compressed)`, so each PC in
/// the text range is decoded once and reused on every revisit. Stores into the
/// executable range invalidate the overlapping slots, which keeps the cache
/// coherent even under self-modifying code (re-decoded entries append to the
/// arena; stale entries are unreachable and their memory is bounded by the
/// number of rewrites).
#[derive(Clone, Debug)]
pub struct DecodeCache {
    text_base: u64,
    text_end: u64,
    /// One slot per halfword in `[text_base, text_end)`: `EMPTY` or an index
    /// into `entries`. Allocated lazily on first insert so that emulator
    /// snapshots (checkpoints) stay cheap until they are actually replayed.
    slots: Vec<u32>,
    entries: Vec<CachedDecode>,
}

#[derive(Clone, Debug)]
pub struct CachedDecode {
    pub instr: Instruction,
    /// Instruction length in bytes (2 or 4), for advancing the PC.
    pub len: u8,
}

impl DecodeCache {
    /// A cache with an empty executable range; every lookup misses and every
    /// insert/invalidate is a no-op.
    pub fn empty() -> Self {
        Self {
            text_base: 0,
            text_end: 0,
            slots: Vec::new(),
            entries: Vec::new(),
        }
    }

    /// Set the executable address range this cache covers, clearing any
    /// previously cached entries.
    pub fn init(&mut self, text_base: u64, text_end: u64) {
        debug_assert!(text_base <= text_end);
        self.text_base = text_base;
        self.text_end = text_end;
        self.slots = Vec::new();
        self.entries = Vec::new();
    }

    /// Permanently disable the cache: every lookup misses and every insert is
    /// a no-op, restoring fetch-per-tick semantics.
    ///
    /// Required while checkpoint saving is active: replaying a chunk needs its
    /// text bytes in the chunk's first-touch memory image, which only happens
    /// if every executed instruction actually fetches from memory.
    pub fn disable(&mut self) {
        self.init(0, 0);
    }

    /// A copy that keeps the executable range but drops all cached entries.
    /// Used for emulator snapshots: replay re-populates the cache lazily.
    pub fn snapshot_with_empty_entries(&self) -> Self {
        Self {
            text_base: self.text_base,
            text_end: self.text_end,
            slots: Vec::new(),
            entries: Vec::new(),
        }
    }

    #[inline]
    pub fn lookup(&self, pc: u64) -> Option<&CachedDecode> {
        if pc < self.text_base || pc >= self.text_end {
            return None;
        }
        let slot = ((pc - self.text_base) >> 1) as usize;
        match self.slots.get(slot) {
            Some(&index) if index != EMPTY => Some(&self.entries[index as usize]),
            _ => None,
        }
    }

    pub fn insert(&mut self, pc: u64, instr: Instruction, len: u8) {
        if pc < self.text_base || pc >= self.text_end {
            return;
        }
        if self.slots.is_empty() {
            let num_slots = ((self.text_end - self.text_base) >> 1) as usize;
            self.slots = vec![EMPTY; num_slots];
        }
        let slot = ((pc - self.text_base) >> 1) as usize;
        let index = u32::try_from(self.entries.len()).unwrap_or(EMPTY);
        if index == EMPTY {
            return;
        }
        self.entries.push(CachedDecode { instr, len });
        self.slots[slot] = index;
    }

    /// Invalidate any decoded slots overlapping a store to
    /// `[address, address + width)`. The happy path (store outside the
    /// executable range) is a compare-and-branch.
    #[inline]
    pub fn invalidate_store(&mut self, address: u64, width: u64) {
        if address < self.text_end && address.wrapping_add(width) > self.text_base {
            self.invalidate_range(address, width);
        }
    }

    #[cold]
    #[inline(never)]
    fn invalidate_range(&mut self, address: u64, width: u64) {
        // A 4-byte instruction starting 2 bytes before `address` still
        // overlaps the store, so widen the cleared range accordingly.
        let start = address.saturating_sub(2).max(self.text_base);
        let end = (address + width).min(self.text_end);
        if start >= end {
            return;
        }
        let lo = ((start - self.text_base) >> 1) as usize;
        let hi = ((end - 1 - self.text_base) >> 1) as usize;
        for slot in lo..=hi {
            if let Some(entry) = self.slots.get_mut(slot) {
                *entry = EMPTY;
            }
        }
    }
}
