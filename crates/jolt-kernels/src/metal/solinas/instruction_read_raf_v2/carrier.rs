//! Producer receipts and the stable grouped-address topology.

use std::ops::Range;

use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use thiserror::Error;

pub const ADDRESS_BITS: usize = 2 * RISCV_XLEN;
pub const PHASE_BITS: usize = 8;
pub const ADDRESS_PHASES: usize = ADDRESS_BITS / PHASE_BITS;
pub const PHASE_BINS: usize = 1 << PHASE_BITS;
pub const LOOKUP_TABLES: usize = LookupTableKind::<RISCV_XLEN>::COUNT;
pub const TABLE_SELECTOR_VALUES: usize = LOOKUP_TABLES + 1;
pub const RAF_SELECTOR_VALUES: usize = 2;
pub const GROUPED_SEGMENTS: usize = TABLE_SELECTOR_VALUES * RAF_SELECTOR_VALUES;
pub const GROUPED_SEGMENT_OFFSETS: usize = GROUPED_SEGMENTS + 1;
pub const VIRTUAL_RA_FACTORS: usize = 4;
pub const PHASES_PER_RA_FACTOR: usize = ADDRESS_PHASES / VIRTUAL_RA_FACTORS;

const _: () = assert!(ADDRESS_BITS == 128);
const _: () = assert!(ADDRESS_PHASES == 16);
const _: () = assert!(LOOKUP_TABLES == 40);
const _: () = assert!(GROUPED_SEGMENTS == 82);
const _: () = assert!(PHASES_PER_RA_FACTOR == 4);

/// Identity shared by every plane projected from one producer allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerIdentity {
    device_registry_id: u64,
    source_allocation_identity: usize,
    source_generation: u64,
    rows: usize,
}

impl ProducerIdentity {
    pub fn new(
        device_registry_id: u64,
        source_allocation_identity: usize,
        source_generation: u64,
        rows: usize,
    ) -> Result<Self, CarrierError> {
        validate_rows(rows)?;
        if device_registry_id == 0 {
            return Err(CarrierError::MissingDeviceIdentity);
        }
        if source_allocation_identity == 0 {
            return Err(CarrierError::MissingAllocationIdentity {
                plane: "source rows",
            });
        }
        if source_generation == 0 {
            return Err(CarrierError::MissingSourceGeneration);
        }
        Ok(Self {
            device_registry_id,
            source_allocation_identity,
            source_generation,
            rows,
        })
    }

    pub const fn device_registry_id(self) -> u64 {
        self.device_registry_id
    }

    pub const fn source_allocation_identity(self) -> usize {
        self.source_allocation_identity
    }

    pub const fn source_generation(self) -> u64 {
        self.source_generation
    }

    pub const fn rows(self) -> usize {
        self.rows
    }
}

/// Receipt for one immutable plane derived from [`ProducerIdentity`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlaneReceipt {
    producer: ProducerIdentity,
    allocation_identity: usize,
}

impl PlaneReceipt {
    pub fn new(
        producer: ProducerIdentity,
        allocation_identity: usize,
        plane: &'static str,
    ) -> Result<Self, CarrierError> {
        if allocation_identity == 0 {
            return Err(CarrierError::MissingAllocationIdentity { plane });
        }
        Ok(Self {
            producer,
            allocation_identity,
        })
    }

    pub const fn producer(self) -> ProducerIdentity {
        self.producer
    }

    pub const fn allocation_identity(self) -> usize {
        self.allocation_identity
    }
}

/// Immutable cycle-order projection owned by the stage-5 facts producer.
#[derive(Clone, Copy, Debug)]
pub struct CycleOrderPlane<'a, T> {
    values: &'a [T],
    receipt: PlaneReceipt,
}

impl<'a, T> CycleOrderPlane<'a, T> {
    pub fn new(
        values: &'a [T],
        receipt: PlaneReceipt,
        plane: &'static str,
    ) -> Result<Self, CarrierError> {
        let expected = receipt.producer.rows;
        if values.len() != expected {
            return Err(CarrierError::PlaneLength {
                plane,
                expected,
                got: values.len(),
            });
        }
        Ok(Self { values, receipt })
    }

    pub const fn values(self) -> &'a [T] {
        self.values
    }

    pub const fn receipt(self) -> PlaneReceipt {
        self.receipt
    }
}

/// A stable counting layout in `2 * table_plus_one + raf_flag` order.
///
/// Every segment preserves cycle order. The private fields ensure that a
/// topology can be published only after checking its permutation and claims.
#[derive(Debug, Eq, PartialEq)]
pub struct GroupedAddressTopology {
    claims_receipt: PlaneReceipt,
    allocation_identity: usize,
    segment_offsets: [u32; GROUPED_SEGMENT_OFFSETS],
    grouped_to_cycle: Vec<u32>,
}

impl GroupedAddressTopology {
    /// Builds the required O(T) stable counting layout at the producer.
    pub fn stable_from_claims(
        claims: CycleOrderPlane<'_, u8>,
        allocation_identity: usize,
    ) -> Result<Self, CarrierError> {
        if allocation_identity == 0 {
            return Err(CarrierError::MissingAllocationIdentity {
                plane: "grouped topology",
            });
        }
        let rows = claims.values;
        let mut counts = [0usize; GROUPED_SEGMENTS];
        for &claim in rows {
            counts[decode_claim(claim)?.segment()] += 1;
        }

        let mut segment_offsets = [0u32; GROUPED_SEGMENT_OFFSETS];
        let mut cursor = 0usize;
        for (segment, count) in counts.into_iter().enumerate() {
            cursor = cursor
                .checked_add(count)
                .ok_or(CarrierError::IndexOverflow {
                    name: "segment rows",
                })?;
            segment_offsets[segment + 1] = shader_index("segment offset", cursor)?;
        }

        let mut next: [usize; GROUPED_SEGMENTS] =
            std::array::from_fn(|segment| segment_offsets[segment] as usize);
        let mut grouped_to_cycle = vec![0u32; rows.len()];
        for (cycle, &claim) in rows.iter().enumerate() {
            let segment = decode_claim(claim)?.segment();
            let grouped = next[segment];
            next[segment] += 1;
            grouped_to_cycle[grouped] = shader_index("cycle index", cycle)?;
        }

        Self::from_checked_parts(
            claims,
            allocation_identity,
            segment_offsets,
            grouped_to_cycle,
        )
    }

    /// Attaches producer-built storage after checking every ordering invariant.
    pub fn from_checked_parts(
        claims: CycleOrderPlane<'_, u8>,
        allocation_identity: usize,
        segment_offsets: [u32; GROUPED_SEGMENT_OFFSETS],
        grouped_to_cycle: Vec<u32>,
    ) -> Result<Self, CarrierError> {
        if allocation_identity == 0 {
            return Err(CarrierError::MissingAllocationIdentity {
                plane: "grouped topology",
            });
        }
        validate_topology(claims.values, &segment_offsets, &grouped_to_cycle)?;
        Ok(Self {
            claims_receipt: claims.receipt,
            allocation_identity,
            segment_offsets,
            grouped_to_cycle,
        })
    }

    pub const fn producer(&self) -> ProducerIdentity {
        self.claims_receipt.producer
    }

    pub const fn claims_receipt(&self) -> PlaneReceipt {
        self.claims_receipt
    }

    pub const fn allocation_identity(&self) -> usize {
        self.allocation_identity
    }

    pub const fn rows(&self) -> usize {
        self.claims_receipt.producer.rows
    }

    pub const fn segment_offsets(&self) -> &[u32; GROUPED_SEGMENT_OFFSETS] {
        &self.segment_offsets
    }

    pub fn grouped_to_cycle(&self) -> &[u32] {
        &self.grouped_to_cycle
    }

    pub fn segment_range(&self, segment: usize) -> Result<Range<usize>, CarrierError> {
        if segment >= GROUPED_SEGMENTS {
            return Err(CarrierError::InvalidSegment(segment));
        }
        Ok(self.segment_offsets[segment] as usize..self.segment_offsets[segment + 1] as usize)
    }

    pub fn segment_len(&self, segment: usize) -> Result<usize, CarrierError> {
        let range = self.segment_range(segment)?;
        Ok(range.end - range.start)
    }
}

/// Borrowed facts accepted by address, cycle, flag-opening, and RA consumers.
#[derive(Clone, Copy, Debug)]
pub struct InstructionFactsCarrier<'a> {
    lookups: CycleOrderPlane<'a, u128>,
    claims: CycleOrderPlane<'a, u8>,
    topology: &'a GroupedAddressTopology,
}

impl<'a> InstructionFactsCarrier<'a> {
    pub fn attach(
        expected_device_registry_id: u64,
        lookups: CycleOrderPlane<'a, u128>,
        claims: CycleOrderPlane<'a, u8>,
        topology: &'a GroupedAddressTopology,
    ) -> Result<Self, CarrierError> {
        if expected_device_registry_id == 0 {
            return Err(CarrierError::MissingDeviceIdentity);
        }
        let producer = lookups.receipt.producer;
        if producer.device_registry_id != expected_device_registry_id {
            return Err(CarrierError::DeviceMismatch {
                expected: expected_device_registry_id,
                got: producer.device_registry_id,
            });
        }
        for (plane, actual) in [
            ("claim plane", claims.receipt.producer),
            ("grouped topology", topology.producer()),
        ] {
            if actual != producer {
                return Err(CarrierError::ProducerMismatch { plane });
            }
        }
        if claims.receipt != topology.claims_receipt {
            return Err(CarrierError::ClaimPlaneReceiptMismatch {
                expected: topology.claims_receipt.allocation_identity,
                got: claims.receipt.allocation_identity,
            });
        }
        for &claim in claims.values {
            let _ = decode_claim(claim)?;
        }
        Ok(Self {
            lookups,
            claims,
            topology,
        })
    }

    pub const fn producer(self) -> ProducerIdentity {
        self.lookups.receipt.producer
    }

    pub const fn rows(self) -> usize {
        self.lookups.receipt.producer.rows
    }

    pub const fn lookups_cycle_order(self) -> &'a [u128] {
        self.lookups.values
    }

    pub const fn claims_cycle_order(self) -> &'a [u8] {
        self.claims.values
    }

    pub const fn topology(self) -> &'a GroupedAddressTopology {
        self.topology
    }

    pub const fn claims_receipt(self) -> PlaneReceipt {
        self.claims.receipt
    }

    pub fn cycle_fact(self, cycle: usize) -> Result<InstructionFact, CarrierError> {
        let lookup =
            self.lookups
                .values
                .get(cycle)
                .copied()
                .ok_or(CarrierError::CycleOutOfRange {
                    cycle,
                    rows: self.rows(),
                })?;
        let claim = self.claims.values[cycle];
        let decoded = decode_claim(claim)?;
        Ok(InstructionFact {
            lookup,
            table_index: decoded.table_index,
            raf_flag: decoded.raf_flag,
        })
    }

    pub fn grouped_fact(&self, grouped: usize) -> Result<InstructionFact, CarrierError> {
        let cycle = self.topology.grouped_to_cycle.get(grouped).copied().ok_or(
            CarrierError::GroupedRowOutOfRange {
                grouped,
                rows: self.rows(),
            },
        )? as usize;
        (*self).cycle_fact(cycle)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionFact {
    lookup: u128,
    table_index: Option<usize>,
    raf_flag: bool,
}

impl InstructionFact {
    pub fn new(
        lookup: u128,
        table_index: Option<usize>,
        raf_flag: bool,
    ) -> Result<Self, CarrierError> {
        if let Some(table) = table_index {
            validate_table(table)?;
        }
        Ok(Self {
            lookup,
            table_index,
            raf_flag,
        })
    }

    pub const fn lookup(self) -> u128 {
        self.lookup
    }

    pub const fn table_index(self) -> Option<usize> {
        self.table_index
    }

    pub const fn raf_flag(self) -> bool {
        self.raf_flag
    }

    pub fn claim_byte(self) -> Result<u8, CarrierError> {
        pack_claim(self.table_index, self.raf_flag)
    }

    pub fn segment(self) -> Result<usize, CarrierError> {
        segment_index(self.table_index, self.raf_flag)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodedClaim {
    table_index: Option<usize>,
    raf_flag: bool,
}

impl DecodedClaim {
    pub const fn table_index(self) -> Option<usize> {
        self.table_index
    }

    pub const fn raf_flag(self) -> bool {
        self.raf_flag
    }

    pub const fn table_plus_one(self) -> usize {
        match self.table_index {
            Some(table) => table + 1,
            None => 0,
        }
    }

    pub const fn segment(self) -> usize {
        2 * self.table_plus_one() + self.raf_flag as usize
    }
}

pub fn pack_claim(table_index: Option<usize>, raf_flag: bool) -> Result<u8, CarrierError> {
    let table_plus_one = match table_index {
        Some(table) => {
            validate_table(table)?;
            table + 1
        }
        None => 0,
    };
    Ok(table_plus_one as u8 | (u8::from(raf_flag) << 7))
}

pub fn decode_claim(packed: u8) -> Result<DecodedClaim, CarrierError> {
    let table_plus_one = usize::from(packed & 0x7f);
    if table_plus_one > LOOKUP_TABLES {
        return Err(CarrierError::InvalidClaimByte(packed));
    }
    Ok(DecodedClaim {
        table_index: table_plus_one.checked_sub(1),
        raf_flag: packed & 0x80 != 0,
    })
}

pub fn segment_index(table_index: Option<usize>, raf_flag: bool) -> Result<usize, CarrierError> {
    let table_plus_one = match table_index {
        Some(table) => {
            validate_table(table)?;
            table + 1
        }
        None => 0,
    };
    Ok(2 * table_plus_one + usize::from(raf_flag))
}

pub const fn segment_selectors(segment: usize) -> Option<(Option<usize>, bool)> {
    if segment >= GROUPED_SEGMENTS {
        return None;
    }
    let table_plus_one = segment / 2;
    let table = if table_plus_one == 0 {
        None
    } else {
        Some(table_plus_one - 1)
    };
    Some((table, segment & 1 != 0))
}

fn validate_topology(
    claims: &[u8],
    segment_offsets: &[u32; GROUPED_SEGMENT_OFFSETS],
    grouped_to_cycle: &[u32],
) -> Result<(), CarrierError> {
    let rows = claims.len();
    if grouped_to_cycle.len() != rows {
        return Err(CarrierError::TopologyLength {
            expected: rows,
            got: grouped_to_cycle.len(),
        });
    }
    let rows_u32 = shader_index("topology rows", rows)?;
    if segment_offsets[0] != 0 || segment_offsets[GROUPED_SEGMENTS] != rows_u32 {
        return Err(CarrierError::InvalidSegmentCoverage {
            rows,
            first: segment_offsets[0],
            last: segment_offsets[GROUPED_SEGMENTS],
        });
    }
    for (segment, pair) in segment_offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(CarrierError::NonMonotoneSegmentOffset {
                segment,
                start: pair[0],
                end: pair[1],
            });
        }
    }

    let mut seen = vec![0u64; rows.div_ceil(64)];
    for segment in 0..GROUPED_SEGMENTS {
        let start = segment_offsets[segment] as usize;
        let end = segment_offsets[segment + 1] as usize;
        let mut previous_cycle = None;
        for (position, &cycle_u32) in grouped_to_cycle[start..end].iter().enumerate() {
            let cycle = cycle_u32 as usize;
            if cycle >= rows {
                return Err(CarrierError::TopologyCycleOutOfRange {
                    grouped: start + position,
                    cycle,
                    rows,
                });
            }
            let word = &mut seen[cycle / 64];
            let bit = 1u64 << (cycle % 64);
            if *word & bit != 0 {
                return Err(CarrierError::DuplicateTopologyCycle { cycle });
            }
            *word |= bit;

            let actual = decode_claim(claims[cycle])?.segment();
            if actual != segment {
                return Err(CarrierError::TopologySegmentMismatch {
                    grouped: start + position,
                    expected: segment,
                    got: actual,
                });
            }
            if previous_cycle.is_some_and(|previous| previous >= cycle) {
                return Err(CarrierError::UnstableSegmentOrder {
                    segment,
                    previous: previous_cycle.unwrap_or(cycle),
                    cycle,
                });
            }
            previous_cycle = Some(cycle);
        }
    }
    Ok(())
}

fn validate_rows(rows: usize) -> Result<(), CarrierError> {
    if rows < 2 || !rows.is_power_of_two() || rows > u32::MAX as usize {
        Err(CarrierError::InvalidRows(rows))
    } else {
        Ok(())
    }
}

fn validate_table(table: usize) -> Result<(), CarrierError> {
    if table >= LOOKUP_TABLES {
        Err(CarrierError::InvalidTable(table))
    } else {
        Ok(())
    }
}

fn shader_index(name: &'static str, value: usize) -> Result<u32, CarrierError> {
    u32::try_from(value).map_err(|_| CarrierError::IndexOverflow { name })
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum CarrierError {
    #[error("InstructionReadRaf facts need a power-of-two row count in 2..2^32, got {0}")]
    InvalidRows(usize),
    #[error("InstructionReadRaf producer device identity is zero")]
    MissingDeviceIdentity,
    #[error("InstructionReadRaf producer source generation is zero")]
    MissingSourceGeneration,
    #[error("InstructionReadRaf {plane} allocation identity is zero")]
    MissingAllocationIdentity { plane: &'static str },
    #[error("InstructionReadRaf carrier expected device {expected}, got {got}")]
    DeviceMismatch { expected: u64, got: u64 },
    #[error("InstructionReadRaf {plane} does not originate from the lookup producer")]
    ProducerMismatch { plane: &'static str },
    #[error(
        "InstructionReadRaf topology belongs to claim allocation {expected}, not attached allocation {got}"
    )]
    ClaimPlaneReceiptMismatch { expected: usize, got: usize },
    #[error("InstructionReadRaf {plane} has {got} rows, expected {expected}")]
    PlaneLength {
        plane: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("InstructionReadRaf table index {0} is outside the 40-table specialization")]
    InvalidTable(usize),
    #[error("InstructionReadRaf packed claim byte 0x{0:02x} selects an invalid table")]
    InvalidClaimByte(u8),
    #[error("InstructionReadRaf grouped segment {0} is outside the fixed layout")]
    InvalidSegment(usize),
    #[error("InstructionReadRaf grouped topology has {got} rows, expected {expected}")]
    TopologyLength { expected: usize, got: usize },
    #[error(
        "InstructionReadRaf grouped segment coverage has first={first}, last={last}, expected 0..{rows}"
    )]
    InvalidSegmentCoverage { rows: usize, first: u32, last: u32 },
    #[error("InstructionReadRaf grouped segment {segment} decreases from {start} to {end}")]
    NonMonotoneSegmentOffset {
        segment: usize,
        start: u32,
        end: u32,
    },
    #[error(
        "InstructionReadRaf grouped row {grouped} contains cycle {cycle}, outside {rows} rows"
    )]
    TopologyCycleOutOfRange {
        grouped: usize,
        cycle: usize,
        rows: usize,
    },
    #[error("InstructionReadRaf grouped topology repeats cycle {cycle}")]
    DuplicateTopologyCycle { cycle: usize },
    #[error(
        "InstructionReadRaf grouped row {grouped} belongs to segment {got}, expected {expected}"
    )]
    TopologySegmentMismatch {
        grouped: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "InstructionReadRaf segment {segment} is not stable: cycle {cycle} follows {previous}"
    )]
    UnstableSegmentOrder {
        segment: usize,
        previous: usize,
        cycle: usize,
    },
    #[error("InstructionReadRaf cycle {cycle} is outside {rows} rows")]
    CycleOutOfRange { cycle: usize, rows: usize },
    #[error("InstructionReadRaf grouped row {grouped} is outside {rows} rows")]
    GroupedRowOutOfRange { grouped: usize, rows: usize },
    #[error("InstructionReadRaf {name} exceeds the 32-bit backend index")]
    IndexOverflow { name: &'static str },
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use fixed valid receipts")]
mod tests {
    use super::*;

    fn producer(rows: usize, generation: u64) -> ProducerIdentity {
        ProducerIdentity::new(7, 0x1000, generation, rows).unwrap()
    }

    fn receipt(
        producer: ProducerIdentity,
        allocation_identity: usize,
        plane: &'static str,
    ) -> PlaneReceipt {
        PlaneReceipt::new(producer, allocation_identity, plane).unwrap()
    }

    #[test]
    fn carrier_binds_topology_to_exact_claim_receipt_and_generation() {
        let lookups = [0u128, 1, 2, 3];
        let claims = [
            pack_claim(None, false).unwrap(),
            pack_claim(Some(0), false).unwrap(),
            pack_claim(Some(0), true).unwrap(),
            pack_claim(None, true).unwrap(),
        ];
        let producer = producer(lookups.len(), 9);
        let lookup_plane = CycleOrderPlane::new(
            &lookups,
            receipt(producer, 0x2000, "lookup plane"),
            "lookup plane",
        )
        .unwrap();
        let claim_receipt = receipt(producer, 0x3000, "claim plane");
        let claim_plane = CycleOrderPlane::new(&claims, claim_receipt, "claim plane").unwrap();
        let topology = GroupedAddressTopology::stable_from_claims(claim_plane, 0x4000).unwrap();

        let carrier =
            InstructionFactsCarrier::attach(7, lookup_plane, claim_plane, &topology).unwrap();
        assert_eq!(carrier.claims_receipt(), claim_receipt);
        assert_eq!(topology.claims_receipt(), claim_receipt);
        assert_eq!(carrier.producer().source_generation(), 9);
        assert!(matches!(
            InstructionFactsCarrier::attach(8, lookup_plane, claim_plane, &topology),
            Err(CarrierError::DeviceMismatch {
                expected: 8,
                got: 7,
            })
        ));

        let replacement_receipt = receipt(producer, 0x3001, "replacement claim plane");
        let replacement =
            CycleOrderPlane::new(&claims, replacement_receipt, "replacement claim plane").unwrap();
        assert!(matches!(
            InstructionFactsCarrier::attach(7, lookup_plane, replacement, &topology),
            Err(CarrierError::ClaimPlaneReceiptMismatch {
                expected: 0x3000,
                got: 0x3001,
            })
        ));

        let next_producer = ProducerIdentity::new(7, 0x1000, 10, lookups.len()).unwrap();
        let next_claims = CycleOrderPlane::new(
            &claims,
            receipt(next_producer, 0x3000, "next claim plane"),
            "next claim plane",
        )
        .unwrap();
        assert!(matches!(
            InstructionFactsCarrier::attach(7, lookup_plane, next_claims, &topology),
            Err(CarrierError::ProducerMismatch {
                plane: "claim plane",
            })
        ));
    }

    #[test]
    fn topology_validation_rejects_every_permutation_drift() {
        let claims = [0u8; 4];
        let producer = producer(claims.len(), 1);
        let claim_plane = CycleOrderPlane::new(
            &claims,
            receipt(producer, 0x3000, "claim plane"),
            "claim plane",
        )
        .unwrap();
        let mut offsets = [4u32; GROUPED_SEGMENT_OFFSETS];
        offsets[0] = 0;

        assert!(GroupedAddressTopology::from_checked_parts(
            claim_plane,
            0x4000,
            offsets,
            vec![0, 1, 2, 3],
        )
        .is_ok());
        assert!(matches!(
            GroupedAddressTopology::from_checked_parts(claim_plane, 0x4000, offsets, vec![0, 1, 2],),
            Err(CarrierError::TopologyLength {
                expected: 4,
                got: 3,
            })
        ));
        let mut incomplete = offsets;
        incomplete[GROUPED_SEGMENTS] = 3;
        assert!(matches!(
            GroupedAddressTopology::from_checked_parts(
                claim_plane,
                0x4000,
                incomplete,
                vec![0, 1, 2, 3],
            ),
            Err(CarrierError::InvalidSegmentCoverage { last: 3, .. })
        ));
        assert!(matches!(
            GroupedAddressTopology::from_checked_parts(
                claim_plane,
                0x4000,
                offsets,
                vec![0, 1, 1, 3],
            ),
            Err(CarrierError::DuplicateTopologyCycle { cycle: 1 })
        ));
        assert!(matches!(
            GroupedAddressTopology::from_checked_parts(
                claim_plane,
                0x4000,
                offsets,
                vec![0, 1, 2, 4],
            ),
            Err(CarrierError::TopologyCycleOutOfRange { cycle: 4, .. })
        ));
        assert!(matches!(
            GroupedAddressTopology::from_checked_parts(
                claim_plane,
                0x4000,
                offsets,
                vec![1, 0, 2, 3],
            ),
            Err(CarrierError::UnstableSegmentOrder { .. })
        ));

        let mut nonmonotone = offsets;
        nonmonotone[1] = 3;
        nonmonotone[2] = 2;
        assert!(matches!(
            GroupedAddressTopology::from_checked_parts(
                claim_plane,
                0x4000,
                nonmonotone,
                vec![0, 1, 2, 3],
            ),
            Err(CarrierError::NonMonotoneSegmentOffset { segment: 1, .. })
        ));

        let wrong_claims = [0x80, 0, 0, 0];
        let wrong_plane = CycleOrderPlane::new(
            &wrong_claims,
            receipt(producer, 0x3001, "wrong claim plane"),
            "wrong claim plane",
        )
        .unwrap();
        assert!(matches!(
            GroupedAddressTopology::from_checked_parts(
                wrong_plane,
                0x4000,
                offsets,
                vec![0, 1, 2, 3],
            ),
            Err(CarrierError::TopologySegmentMismatch { grouped: 0, .. })
        ));
    }

    #[test]
    fn identities_and_claim_encoding_fail_closed() {
        assert_eq!(
            ProducerIdentity::new(0, 0x1000, 1, 4),
            Err(CarrierError::MissingDeviceIdentity)
        );
        assert_eq!(
            ProducerIdentity::new(7, 0x1000, 0, 4),
            Err(CarrierError::MissingSourceGeneration)
        );
        assert_eq!(
            PlaneReceipt::new(producer(4, 1), 0, "claim plane"),
            Err(CarrierError::MissingAllocationIdentity {
                plane: "claim plane",
            })
        );
        let short = [0u8; 2];
        assert!(matches!(
            CycleOrderPlane::new(
                &short,
                receipt(producer(4, 1), 0x3000, "claim plane"),
                "claim plane",
            ),
            Err(CarrierError::PlaneLength {
                expected: 4,
                got: 2,
                ..
            })
        ));
        assert_eq!(decode_claim(41), Err(CarrierError::InvalidClaimByte(41)));
        for table in 0..LOOKUP_TABLES {
            for raf in [false, true] {
                let packed = pack_claim(Some(table), raf).unwrap();
                let decoded = decode_claim(packed).unwrap();
                assert_eq!(decoded.table_index(), Some(table));
                assert_eq!(decoded.raf_flag(), raf);
                assert_eq!(decoded.segment(), segment_index(Some(table), raf).unwrap());
            }
        }
    }
}
