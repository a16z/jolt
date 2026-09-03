use std::mem::size_of;

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::Zero as _;

#[cfg(test)]
use super::Fp128;

#[cfg(test)]
mod prefix;
mod runtime;

pub(super) const SOURCE: &str = include_str!("shader.metal");

#[cfg(feature = "test-utils")]
pub(crate) use runtime::RamReadWritePreparationTiming;
pub(crate) use runtime::{
    RamRafSegmentedAddressPlane, RamReadWriteDispatchTiming, RamReadWriteFinish,
    RamReadWriteSequence,
};

pub const RAM_READ_WRITE_ADDRESS_PIPELINE: &str = "solinas_ram_read_write_address";
pub const RAM_READ_WRITE_ADDRESS_BOUNDED_PIPELINE: &str = "solinas_ram_read_write_address_bounded";
pub const RAM_READ_WRITE_ADDRESS_HOT_COUNT_PIPELINE: &str =
    "solinas_ram_read_write_address_hot_count";
pub const RAM_READ_WRITE_ADDRESS_HOT_PREFIX_PIPELINE: &str =
    "solinas_ram_read_write_address_hot_prefix";
pub const RAM_READ_WRITE_ADDRESS_HOT_SCATTER_PIPELINE: &str =
    "solinas_ram_read_write_address_hot_scatter";
pub const RAM_READ_WRITE_ADDRESS_HOT_MESSAGE_PIPELINE: &str =
    "solinas_ram_read_write_address_hot_message";
pub const RAM_READ_WRITE_CYCLE_PIPELINE: &str = "solinas_ram_read_write_cycle";
pub const RAM_READ_WRITE_REDUCTION_PIPELINE: &str = "solinas_ram_read_write_reduce";
pub const RAM_READ_WRITE_INITIAL_SCATTER_PIPELINE: &str = "solinas_ram_read_write_initial_scatter";
pub const RAM_READ_WRITE_PREFIX_ADDRESS_PIPELINE: &str = "solinas_ram_read_write_prefix_address";
pub const RAM_READ_WRITE_PREFIX_ADDRESS_TRANSITION_PIPELINE: &str =
    "solinas_ram_read_write_prefix_address_transition";
pub const RAM_READ_WRITE_PREFIX_HOT_TRANSITION_PIPELINE: &str =
    "solinas_ram_read_write_prefix_hot_transition";
pub const RAM_READ_WRITE_PREFIX_CYCLE_PIPELINE: &str = "solinas_ram_read_write_prefix_cycle";
pub const RAM_READ_WRITE_PREFIX_CYCLE_TRANSITION_PIPELINE: &str =
    "solinas_ram_read_write_prefix_cycle_transition";
pub const RAM_READ_WRITE_THREADS: usize = 256;
pub const RAM_READ_WRITE_SIMD_WIDTH: usize = 32;
pub const RAM_READ_WRITE_REDUCTION_WIDTH: usize = 32;
pub const RAM_READ_WRITE_CYCLE_TILE_LOG2: usize = 12;
pub const RAM_READ_WRITE_CYCLE_THREADGROUP_BYTES_MAX: u64 = 1024;
pub const RAM_READ_WRITE_HOT_SEGMENT_THRESHOLD: usize = 1 << 6;
pub const RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE: usize = 1 << 12;
pub const RAM_READ_WRITE_BOUNDED_SEGMENT_MAX: usize = RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE;
pub const RAM_READ_WRITE_BOUNDED_THREADGROUP_BYTES_MAX: u64 = 1024;
pub const RAM_READ_WRITE_HOT_COMPACTION_THREADS: usize = 256;
pub const RAM_READ_WRITE_HOT_THREADGROUP_BYTES_MAX: u64 = 1024;
pub const RAM_READ_WRITE_RECORD_PREFIX_LOG_T_MIN: usize = 29;
pub const RAM_READ_WRITE_RECORD_PREFIX_ROUNDS: usize = 6;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Segment {
    offset: u32,
    length: u32,
    capacity: u32,
    aux_offset: u32,
}

const _: [(); 16] = [(); size_of::<Segment>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct HotChunk {
    hot_index: u32,
    local_offset: u32,
}

const _: [(); 8] = [(); size_of::<HotChunk>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct HotSegment {
    segment_index: u32,
    first_chunk: u32,
    chunk_count: u32,
    aux_offset: u32,
}

const _: [(); 16] = [(); size_of::<HotSegment>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PhaseParams {
    work_items: u32,
    output_stride: u32,
    e_in_length: u32,
    bind: u32,
    emit_message: u32,
    hot_source_aux: u32,
    hot_threshold: u32,
    source_initial: u32,
}

const _: [(); 32] = [(); size_of::<PhaseParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PrefixPhaseParams {
    records: u32,
    output_stride: u32,
    e_in_length: u32,
    rounds_bound: u32,
    bind: u32,
    reserved: [u32; 3],
}

const _: [(); 32] = [(); size_of::<PrefixPhaseParams>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CycleProductRoot {
    pub block: usize,
    pub hamming: AkitaField,
    pub increment: AkitaField,
}

pub(crate) struct SparseCycleProduct {
    entries: Vec<CycleProductRoot>,
    rounds_bound: usize,
    rounds: usize,
}

impl SparseCycleProduct {
    pub(crate) fn from_roots(
        entries: Vec<CycleProductRoot>,
        rounds_bound: usize,
        rounds: usize,
    ) -> Self {
        debug_assert!(entries.windows(2).all(|pair| pair[0].block < pair[1].block));
        Self {
            entries,
            rounds_bound,
            rounds,
        }
    }

    fn pair(&self, parent: usize) -> ([AkitaField; 2], [AkitaField; 2]) {
        let low_block = 2 * parent;
        let high_block = low_block + 1;
        let low = self.value(low_block);
        let high = self.value(high_block);
        ([low.0, high.0 - low.0], [low.1, high.1 - low.1])
    }

    fn value(&self, block: usize) -> (AkitaField, AkitaField) {
        self.entries
            .binary_search_by_key(&block, |entry| entry.block)
            .ok()
            .map_or((AkitaField::zero(), AkitaField::zero()), |index| {
                let entry = self.entries[index];
                (entry.hamming, entry.increment)
            })
    }

    pub(crate) fn quadratic_coefficients(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> [AkitaField; 2] {
        let in_bits = e_in.len().trailing_zeros() as usize;
        let in_mask = e_in.len() - 1;
        let parents = self.entries.last().map_or(0, |entry| entry.block / 2 + 1);
        (0..parents).fold([AkitaField::zero(); 2], |mut sum, parent| {
            let (hamming, increment) = self.pair(parent);
            if hamming != [AkitaField::zero(); 2] || increment != [AkitaField::zero(); 2] {
                let head = e_out[parent >> in_bits] * e_in[parent & in_mask];
                sum[0] += head * hamming[0] * increment[0];
                sum[1] += head * hamming[1] * increment[1];
            }
            sum
        })
    }

    pub(crate) fn bind(&mut self, challenge: AkitaField) {
        let mut bound = Vec::with_capacity(self.entries.len());
        let mut index = 0;
        while index < self.entries.len() {
            let parent = self.entries[index].block / 2;
            let mut low = (AkitaField::zero(), AkitaField::zero());
            let mut high = (AkitaField::zero(), AkitaField::zero());
            while self
                .entries
                .get(index)
                .is_some_and(|entry| entry.block / 2 == parent)
            {
                let entry = self.entries[index];
                if entry.block.is_multiple_of(2) {
                    low = (entry.hamming, entry.increment);
                } else {
                    high = (entry.hamming, entry.increment);
                }
                index += 1;
            }
            let hamming = low.0 + challenge * (high.0 - low.0);
            let increment = low.1 + challenge * (high.1 - low.1);
            if hamming != AkitaField::zero() || increment != AkitaField::zero() {
                bound.push(CycleProductRoot {
                    block: parent,
                    hamming,
                    increment,
                });
            }
        }
        self.entries = bound;
        self.rounds_bound += 1;
    }

    pub(crate) fn final_increment(&self) -> Option<AkitaField> {
        if self.rounds_bound != self.rounds {
            return None;
        }
        match self.entries.as_slice() {
            [] => Some(AkitaField::zero()),
            [entry] if entry.block == 0 => Some(entry.increment),
            _ => None,
        }
    }
}

#[cfg(test)]
fn fp128_lerp(low: Fp128, high: Fp128, challenge: AkitaField) -> AkitaField {
    let low = low.into_jolt_field::<AkitaField>();
    let high = high.into_jolt_field::<AkitaField>();
    low + challenge * (high - low)
}

#[cfg(test)]
mod tests {
    use jolt_field::{FromPrimitiveInt, One as _};

    use super::*;

    #[test]
    fn sparse_cycle_product_binds_missing_sides() {
        let mut product = SparseCycleProduct::from_roots(
            vec![
                CycleProductRoot {
                    block: 0,
                    hamming: AkitaField::one(),
                    increment: AkitaField::from_i128(-3),
                },
                CycleProductRoot {
                    block: 3,
                    hamming: AkitaField::one(),
                    increment: AkitaField::from_u64(5),
                },
            ],
            0,
            2,
        );
        let challenge = AkitaField::from_u64(7);
        product.bind(challenge);
        assert_eq!(product.entries.len(), 2);
        assert_eq!(product.entries[0].block, 0);
        assert_eq!(product.entries[0].hamming, AkitaField::one() - challenge);
        assert_eq!(product.entries[1].block, 1);
        assert_eq!(product.entries[1].hamming, challenge);
        product.bind(AkitaField::from_u64(11));
        assert!(product.final_increment().is_some());
    }

    #[test]
    fn fp128_lerp_matches_field_arithmetic() {
        let low = AkitaField::from_u64(9);
        let high = AkitaField::from_u64(27);
        let challenge = AkitaField::from_u64(5);
        assert_eq!(
            fp128_lerp(
                Fp128::from_jolt_field(&low),
                Fp128::from_jolt_field(&high),
                challenge,
            ),
            low + challenge * (high - low)
        );
    }
}
