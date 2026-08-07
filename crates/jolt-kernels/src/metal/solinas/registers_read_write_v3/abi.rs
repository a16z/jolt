use core::{marker::PhantomData, mem::size_of, num::NonZeroU64, num::NonZeroUsize};

use super::RegistersRwV3Error;

pub(crate) const REGISTER_CSR_BLOCK_CYCLES: usize = 256;
pub(crate) const REGISTER_CSR_COLUMNS: usize = 128;
pub(crate) const REGISTER_ADDRESS_BITS: usize = 7;
pub(crate) const REGISTER_FP128_BYTES: usize = 16;

const _: () = assert!(REGISTER_CSR_BLOCK_CYCLES == 1 << 8);
const _: () = assert!(REGISTER_CSR_COLUMNS == 1 << REGISTER_ADDRESS_BITS);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RegisterEventCounts {
    rs1: usize,
    rs2: usize,
    rd: usize,
}

impl RegisterEventCounts {
    pub(crate) const fn new(rs1: usize, rs2: usize, rd: usize) -> Self {
        Self { rs1, rs2, rd }
    }

    pub(crate) const fn rs1(self) -> usize {
        self.rs1
    }

    pub(crate) const fn rs2(self) -> usize {
        self.rs2
    }

    pub(crate) const fn rd(self) -> usize {
        self.rd
    }

    pub(crate) fn checked_total(self) -> Result<usize, RegistersRwV3Error> {
        self.rs1
            .checked_add(self.rs2)
            .and_then(|sum| sum.checked_add(self.rd))
            .ok_or(RegistersRwV3Error::SizeOverflow("event total"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterGeometry {
    cycles: usize,
    log_t: usize,
    blocks: usize,
    block_columns: usize,
}

impl RegisterGeometry {
    pub(crate) fn new(cycles: usize) -> Result<Self, RegistersRwV3Error> {
        if cycles == 0 || !cycles.is_power_of_two() || cycles > u32::MAX as usize {
            return Err(RegistersRwV3Error::InvalidCycleCount(cycles));
        }
        let blocks = cycles.div_ceil(REGISTER_CSR_BLOCK_CYCLES);
        let block_columns = blocks
            .checked_mul(REGISTER_CSR_COLUMNS)
            .ok_or(RegistersRwV3Error::SizeOverflow("CSR block columns"))?;
        Ok(Self {
            cycles,
            log_t: cycles.trailing_zeros() as usize,
            blocks,
            block_columns,
        })
    }

    pub(crate) const fn cycles(self) -> usize {
        self.cycles
    }

    pub(crate) const fn log_t(self) -> usize {
        self.log_t
    }

    pub(crate) const fn blocks(self) -> usize {
        self.blocks
    }

    pub(crate) const fn block_columns(self) -> usize {
        self.block_columns
    }

    pub(crate) fn offset_entries(self) -> Result<usize, RegistersRwV3Error> {
        self.block_columns
            .checked_add(1)
            .ok_or(RegistersRwV3Error::SizeOverflow("CSR offset entries"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterCsrCensus {
    geometry: RegisterGeometry,
    events: RegisterEventCounts,
}

impl RegisterCsrCensus {
    pub(crate) fn new(
        geometry: RegisterGeometry,
        events: RegisterEventCounts,
    ) -> Result<Self, RegistersRwV3Error> {
        for (plane, count) in [
            ("rs1 positions", events.rs1),
            ("rs2 positions", events.rs2),
            ("rd positions", events.rd),
        ] {
            if count > geometry.cycles {
                return Err(RegistersRwV3Error::InvalidEventCount {
                    plane,
                    cycles: geometry.cycles,
                    count,
                });
            }
            if u32::try_from(count).is_err() {
                return Err(RegistersRwV3Error::EventCountOverflow { plane });
            }
        }
        Ok(Self { geometry, events })
    }

    pub(crate) const fn geometry(self) -> RegisterGeometry {
        self.geometry
    }

    pub(crate) const fn events(self) -> RegisterEventCounts {
        self.events
    }

    pub(crate) fn storage_bytes(self) -> Result<u128, RegistersRwV3Error> {
        let block_columns = self.geometry.block_columns as u128;
        let offset_entries = self.geometry.offset_entries()? as u128;
        let event_positions = self.events.checked_total()? as u128;
        checked_sum_u128(&[
            checked_mul_u128("start-value bytes", block_columns, size_of::<u64>() as u128)?,
            checked_mul_u128(
                "offset bytes",
                offset_entries,
                (3 * size_of::<u32>()) as u128,
            )?,
            event_positions,
            checked_mul_u128(
                "rd post-value bytes",
                self.events.rd as u128,
                size_of::<u64>() as u128,
            )?,
        ])
    }
}

pub(crate) const REGISTER_LOG26_CENSUS: RegisterCsrCensus = RegisterCsrCensus {
    geometry: RegisterGeometry {
        cycles: 1 << 26,
        log_t: 26,
        blocks: 1 << 18,
        block_columns: 1 << 25,
    },
    events: RegisterEventCounts {
        rs1: 59_652_323,
        rs2: 55_924_053,
        rd: 50_331_648,
    },
};

pub(crate) const REGISTER_LOG26_CSR_BYTES: u128 = 1_239_649_860;
pub(crate) const REGISTER_LOG26_PRODUCER_BYTES: u128 = 2_313_391_684;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlaneShape {
    elements: usize,
    bytes: usize,
}

impl PlaneShape {
    fn new(
        name: &'static str,
        elements: usize,
        element_bytes: usize,
    ) -> Result<Self, RegistersRwV3Error> {
        let bytes = elements
            .checked_mul(element_bytes)
            .ok_or(RegistersRwV3Error::SizeOverflow(name))?;
        Ok(Self { elements, bytes })
    }

    pub(crate) const fn elements(self) -> usize {
        self.elements
    }

    pub(crate) const fn bytes(self) -> usize {
        self.bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterPlaneLayout {
    start_values: PlaneShape,
    offsets: PlaneShape,
    rs1_positions: PlaneShape,
    rs2_positions: PlaneShape,
    rd_positions: PlaneShape,
    rd_post_values: PlaneShape,
    rd_inc: PlaneShape,
}

impl RegisterPlaneLayout {
    pub(crate) fn new(census: RegisterCsrCensus) -> Result<Self, RegistersRwV3Error> {
        let geometry = census.geometry;
        let events = census.events;
        Ok(Self {
            start_values: PlaneShape::new(
                "start-value plane",
                geometry.block_columns,
                size_of::<u64>(),
            )?,
            offsets: PlaneShape::new("offset plane", geometry.offset_entries()?, size_of::<u32>())?,
            rs1_positions: PlaneShape::new("rs1 position plane", events.rs1, size_of::<u8>())?,
            rs2_positions: PlaneShape::new("rs2 position plane", events.rs2, size_of::<u8>())?,
            rd_positions: PlaneShape::new("rd position plane", events.rd, size_of::<u8>())?,
            rd_post_values: PlaneShape::new("rd post-value plane", events.rd, size_of::<u64>())?,
            rd_inc: PlaneShape::new("rd increment plane", geometry.cycles, REGISTER_FP128_BYTES)?,
        })
    }

    pub(crate) const fn start_values(self) -> PlaneShape {
        self.start_values
    }

    pub(crate) const fn offsets(self) -> PlaneShape {
        self.offsets
    }

    pub(crate) const fn rs1_positions(self) -> PlaneShape {
        self.rs1_positions
    }

    pub(crate) const fn rs2_positions(self) -> PlaneShape {
        self.rs2_positions
    }

    pub(crate) const fn rd_positions(self) -> PlaneShape {
        self.rd_positions
    }

    pub(crate) const fn rd_post_values(self) -> PlaneShape {
        self.rd_post_values
    }

    pub(crate) const fn rd_inc(self) -> PlaneShape {
        self.rd_inc
    }

    pub(crate) fn csr_bytes(self) -> Result<usize, RegistersRwV3Error> {
        checked_sum_usize(&[
            self.start_values.bytes,
            self.offsets
                .bytes
                .checked_mul(3)
                .ok_or(RegistersRwV3Error::SizeOverflow("three offset planes"))?,
            self.rs1_positions.bytes,
            self.rs2_positions.bytes,
            self.rd_positions.bytes,
            self.rd_post_values.bytes,
        ])
    }

    pub(crate) fn producer_bytes(self) -> Result<usize, RegistersRwV3Error> {
        self.csr_bytes()?
            .checked_add(self.rd_inc.bytes)
            .ok_or(RegistersRwV3Error::SizeOverflow("producer bytes"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OrderedPrefixDigest([u64; 4]);

impl OrderedPrefixDigest {
    pub(crate) fn new(words: [u64; 4]) -> Result<Self, RegistersRwV3Error> {
        if words == [0; 4] {
            return Err(RegistersRwV3Error::ZeroOrderedPrefixDigest);
        }
        Ok(Self(words))
    }

    pub(crate) const fn words(self) -> [u64; 4] {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterProducerIdentity {
    device_registry_id: NonZeroU64,
    source_allocation_identity: NonZeroUsize,
    source_allocation_bytes: usize,
    generation: NonZeroU64,
    cycles: usize,
    ordered_prefix_digest: OrderedPrefixDigest,
}

impl RegisterProducerIdentity {
    pub(crate) fn new(
        device_registry_id: u64,
        source_allocation_identity: usize,
        source_allocation_bytes: usize,
        generation: u64,
        cycles: usize,
        ordered_prefix_digest: OrderedPrefixDigest,
    ) -> Result<Self, RegistersRwV3Error> {
        let _geometry = RegisterGeometry::new(cycles)?;
        let device_registry_id = NonZeroU64::new(device_registry_id)
            .ok_or(RegistersRwV3Error::MissingIdentity("device registry"))?;
        let source_allocation_identity = NonZeroUsize::new(source_allocation_identity)
            .ok_or(RegistersRwV3Error::MissingIdentity("source allocation"))?;
        if source_allocation_bytes == 0 {
            return Err(RegistersRwV3Error::MissingIdentity(
                "source allocation bytes",
            ));
        }
        let generation = NonZeroU64::new(generation)
            .ok_or(RegistersRwV3Error::MissingIdentity("source generation"))?;
        Ok(Self {
            device_registry_id,
            source_allocation_identity,
            source_allocation_bytes,
            generation,
            cycles,
            ordered_prefix_digest,
        })
    }

    pub(crate) const fn device_registry_id(self) -> u64 {
        self.device_registry_id.get()
    }

    pub(crate) const fn source_allocation_identity(self) -> usize {
        self.source_allocation_identity.get()
    }

    pub(crate) const fn source_allocation_bytes(self) -> usize {
        self.source_allocation_bytes
    }

    pub(crate) const fn generation(self) -> u64 {
        self.generation.get()
    }

    pub(crate) const fn cycles(self) -> usize {
        self.cycles
    }

    pub(crate) const fn ordered_prefix_digest(self) -> OrderedPrefixDigest {
        self.ordered_prefix_digest
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlaneAllocation {
    device_registry_id: NonZeroU64,
    allocation_identity: NonZeroUsize,
    initialized_generation: NonZeroU64,
    elements: usize,
    bytes: usize,
    initialization_completed: bool,
}

impl PlaneAllocation {
    pub(crate) fn new(
        device_registry_id: u64,
        allocation_identity: usize,
        initialized_generation: u64,
        elements: usize,
        bytes: usize,
        initialization_completed: bool,
    ) -> Result<Self, RegistersRwV3Error> {
        let device_registry_id = NonZeroU64::new(device_registry_id)
            .ok_or(RegistersRwV3Error::MissingIdentity("plane device"))?;
        let allocation_identity = NonZeroUsize::new(allocation_identity)
            .ok_or(RegistersRwV3Error::MissingIdentity("plane allocation"))?;
        let initialized_generation = NonZeroU64::new(initialized_generation)
            .ok_or(RegistersRwV3Error::MissingIdentity("plane generation"))?;
        Ok(Self {
            device_registry_id,
            allocation_identity,
            initialized_generation,
            elements,
            bytes,
            initialization_completed,
        })
    }

    pub(crate) const fn allocation_identity(self) -> usize {
        self.allocation_identity.get()
    }
}

pub(super) trait RegisterPlane {
    const NAME: &'static str;
}

macro_rules! register_planes {
    ($(($type:ident, $name:literal)),+ $(,)?) => {
        $(
            #[derive(Debug, Eq, PartialEq)]
            pub(super) struct $type;

            impl RegisterPlane for $type {
                const NAME: &'static str = $name;
            }
        )+
    };
}

register_planes!(
    (StartValues, "start values"),
    (Rs1Offsets, "rs1 offsets"),
    (Rs2Offsets, "rs2 offsets"),
    (RdOffsets, "rd offsets"),
    (Rs1Positions, "rs1 positions"),
    (Rs2Positions, "rs2 positions"),
    (RdPositions, "rd positions"),
    (RdPostValues, "rd post values"),
    (RdInc, "rd increment"),
);

#[derive(Debug, Eq, PartialEq)]
pub(super) struct AllocationReceipt<P: RegisterPlane> {
    device_registry_id: NonZeroU64,
    allocation_identity: NonZeroUsize,
    initialized_generation: NonZeroU64,
    elements: usize,
    bytes: usize,
    _plane: PhantomData<fn() -> P>,
}

impl<P: RegisterPlane> AllocationReceipt<P> {
    fn admit(
        producer: RegisterProducerIdentity,
        allocation: PlaneAllocation,
        expected: PlaneShape,
    ) -> Result<Self, RegistersRwV3Error> {
        if allocation.device_registry_id != producer.device_registry_id {
            return Err(RegistersRwV3Error::PlaneDeviceMismatch {
                plane: P::NAME,
                expected: producer.device_registry_id(),
                got: allocation.device_registry_id.get(),
            });
        }
        if allocation.initialized_generation != producer.generation {
            return Err(RegistersRwV3Error::PlaneGenerationMismatch {
                plane: P::NAME,
                expected: producer.generation(),
                got: allocation.initialized_generation.get(),
            });
        }
        if !allocation.initialization_completed {
            return Err(RegistersRwV3Error::PlaneInitializationIncomplete { plane: P::NAME });
        }
        if allocation.elements != expected.elements || allocation.bytes != expected.bytes {
            return Err(RegistersRwV3Error::PlaneShape {
                plane: P::NAME,
                expected_elements: expected.elements,
                got_elements: allocation.elements,
                expected_bytes: expected.bytes,
                got_bytes: allocation.bytes,
            });
        }
        Ok(Self {
            device_registry_id: allocation.device_registry_id,
            allocation_identity: allocation.allocation_identity,
            initialized_generation: allocation.initialized_generation,
            elements: allocation.elements,
            bytes: allocation.bytes,
            _plane: PhantomData,
        })
    }

    pub(crate) const fn device_registry_id(&self) -> u64 {
        self.device_registry_id.get()
    }

    pub(crate) const fn allocation_identity(&self) -> usize {
        self.allocation_identity.get()
    }

    pub(crate) const fn initialized_generation(&self) -> u64 {
        self.initialized_generation.get()
    }

    pub(crate) const fn elements(&self) -> usize {
        self.elements
    }

    pub(crate) const fn bytes(&self) -> usize {
        self.bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterPlaneAllocations {
    start_values: PlaneAllocation,
    rs1_offsets: PlaneAllocation,
    rs2_offsets: PlaneAllocation,
    rd_offsets: PlaneAllocation,
    rs1_positions: PlaneAllocation,
    rs2_positions: PlaneAllocation,
    rd_positions: PlaneAllocation,
    rd_post_values: PlaneAllocation,
    rd_inc: PlaneAllocation,
}

impl RegisterPlaneAllocations {
    #[expect(
        clippy::too_many_arguments,
        reason = "one descriptor per immutable ABI plane"
    )]
    pub(crate) const fn new(
        start_values: PlaneAllocation,
        rs1_offsets: PlaneAllocation,
        rs2_offsets: PlaneAllocation,
        rd_offsets: PlaneAllocation,
        rs1_positions: PlaneAllocation,
        rs2_positions: PlaneAllocation,
        rd_positions: PlaneAllocation,
        rd_post_values: PlaneAllocation,
        rd_inc: PlaneAllocation,
    ) -> Self {
        Self {
            start_values,
            rs1_offsets,
            rs2_offsets,
            rd_offsets,
            rs1_positions,
            rs2_positions,
            rd_positions,
            rd_post_values,
            rd_inc,
        }
    }

    fn identities(&self) -> [usize; 9] {
        [
            self.start_values.allocation_identity(),
            self.rs1_offsets.allocation_identity(),
            self.rs2_offsets.allocation_identity(),
            self.rd_offsets.allocation_identity(),
            self.rs1_positions.allocation_identity(),
            self.rs2_positions.allocation_identity(),
            self.rd_positions.allocation_identity(),
            self.rd_post_values.allocation_identity(),
            self.rd_inc.allocation_identity(),
        ]
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct RegisterPlaneReceipts {
    pub(super) start_values: AllocationReceipt<StartValues>,
    pub(super) rs1_offsets: AllocationReceipt<Rs1Offsets>,
    pub(super) rs2_offsets: AllocationReceipt<Rs2Offsets>,
    pub(super) rd_offsets: AllocationReceipt<RdOffsets>,
    pub(super) rs1_positions: AllocationReceipt<Rs1Positions>,
    pub(super) rs2_positions: AllocationReceipt<Rs2Positions>,
    pub(super) rd_positions: AllocationReceipt<RdPositions>,
    pub(super) rd_post_values: AllocationReceipt<RdPostValues>,
    pub(super) rd_inc: AllocationReceipt<RdInc>,
}

impl RegisterPlaneReceipts {
    pub(super) fn admit(
        producer: RegisterProducerIdentity,
        layout: RegisterPlaneLayout,
        allocations: &RegisterPlaneAllocations,
    ) -> Result<Self, RegistersRwV3Error> {
        let identities = allocations.identities();
        for (index, identity) in identities.iter().copied().enumerate() {
            if identity == producer.source_allocation_identity()
                || identities[..index].contains(&identity)
            {
                return Err(RegistersRwV3Error::DuplicateAllocationIdentity { identity });
            }
        }
        Ok(Self {
            start_values: AllocationReceipt::admit(
                producer,
                allocations.start_values,
                layout.start_values,
            )?,
            rs1_offsets: AllocationReceipt::admit(
                producer,
                allocations.rs1_offsets,
                layout.offsets,
            )?,
            rs2_offsets: AllocationReceipt::admit(
                producer,
                allocations.rs2_offsets,
                layout.offsets,
            )?,
            rd_offsets: AllocationReceipt::admit(producer, allocations.rd_offsets, layout.offsets)?,
            rs1_positions: AllocationReceipt::admit(
                producer,
                allocations.rs1_positions,
                layout.rs1_positions,
            )?,
            rs2_positions: AllocationReceipt::admit(
                producer,
                allocations.rs2_positions,
                layout.rs2_positions,
            )?,
            rd_positions: AllocationReceipt::admit(
                producer,
                allocations.rd_positions,
                layout.rd_positions,
            )?,
            rd_post_values: AllocationReceipt::admit(
                producer,
                allocations.rd_post_values,
                layout.rd_post_values,
            )?,
            rd_inc: AllocationReceipt::admit(producer, allocations.rd_inc, layout.rd_inc)?,
        })
    }
}

fn checked_mul_u128(
    name: &'static str,
    left: u128,
    right: u128,
) -> Result<u128, RegistersRwV3Error> {
    left.checked_mul(right)
        .ok_or(RegistersRwV3Error::SizeOverflow(name))
}

fn checked_sum_u128(values: &[u128]) -> Result<u128, RegistersRwV3Error> {
    values.iter().try_fold(0u128, |sum, value| {
        sum.checked_add(*value)
            .ok_or(RegistersRwV3Error::SizeOverflow("CSR byte total"))
    })
}

fn checked_sum_usize(values: &[usize]) -> Result<usize, RegistersRwV3Error> {
    values.iter().try_fold(0usize, |sum, value| {
        sum.checked_add(*value)
            .ok_or(RegistersRwV3Error::SizeOverflow("plane byte total"))
    })
}
