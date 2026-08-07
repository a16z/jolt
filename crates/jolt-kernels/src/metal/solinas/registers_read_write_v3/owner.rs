use core::ops::Range;

use super::abi::{
    AllocationReceipt, OrderedPrefixDigest, RdInc, RdOffsets, RdPositions, RdPostValues,
    RegisterCsrCensus, RegisterEventCounts, RegisterGeometry, RegisterPlaneAllocations,
    RegisterPlaneLayout, RegisterPlaneReceipts, RegisterProducerIdentity, Rs1Offsets, Rs1Positions,
    Rs2Offsets, Rs2Positions, StartValues, REGISTER_CSR_BLOCK_CYCLES, REGISTER_CSR_COLUMNS,
};
use super::RegistersRwV3Error;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RegisterRead {
    register: u8,
    value: u64,
}

impl RegisterRead {
    pub(crate) const fn new(register: u8, value: u64) -> Self {
        Self { register, value }
    }

    pub(crate) const fn register(self) -> u8 {
        self.register
    }

    pub(crate) const fn value(self) -> u64 {
        self.value
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RegisterWrite {
    register: u8,
    pre_value: u64,
    post_value: u64,
}

impl RegisterWrite {
    pub(crate) const fn new(register: u8, pre_value: u64, post_value: u64) -> Self {
        Self {
            register,
            pre_value,
            post_value,
        }
    }

    pub(crate) const fn register(self) -> u8 {
        self.register
    }

    pub(crate) const fn pre_value(self) -> u64 {
        self.pre_value
    }

    pub(crate) const fn post_value(self) -> u64 {
        self.post_value
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RegisterRow {
    rs1: Option<RegisterRead>,
    rs2: Option<RegisterRead>,
    rd: Option<RegisterWrite>,
}

impl RegisterRow {
    pub(crate) const fn new(
        rs1: Option<RegisterRead>,
        rs2: Option<RegisterRead>,
        rd: Option<RegisterWrite>,
    ) -> Self {
        Self { rs1, rs2, rd }
    }

    pub(crate) const fn rs1(self) -> Option<RegisterRead> {
        self.rs1
    }

    pub(crate) const fn rs2(self) -> Option<RegisterRead> {
        self.rs2
    }

    pub(crate) const fn rd(self) -> Option<RegisterWrite> {
        self.rd
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RegisterCsr256Parts {
    pub(crate) cycles: usize,
    pub(crate) start_values: Vec<u64>,
    pub(crate) rs1_offsets: Vec<u32>,
    pub(crate) rs2_offsets: Vec<u32>,
    pub(crate) rd_offsets: Vec<u32>,
    pub(crate) rs1_positions: Vec<u8>,
    pub(crate) rs2_positions: Vec<u8>,
    pub(crate) rd_positions: Vec<u8>,
    pub(crate) rd_post_values: Vec<u64>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RegisterCsrColumn<'a> {
    start_value: u64,
    rs1_positions: &'a [u8],
    rs2_positions: &'a [u8],
    rd_positions: &'a [u8],
    rd_post_values: &'a [u64],
}

impl<'a> RegisterCsrColumn<'a> {
    pub(super) const fn start_value(self) -> u64 {
        self.start_value
    }

    pub(super) const fn rs1_positions(self) -> &'a [u8] {
        self.rs1_positions
    }

    pub(super) const fn rs2_positions(self) -> &'a [u8] {
        self.rs2_positions
    }

    pub(super) const fn rd_positions(self) -> &'a [u8] {
        self.rd_positions
    }

    pub(super) const fn rd_post_values(self) -> &'a [u64] {
        self.rd_post_values
    }
}

/// Structurally checked CSR-256 storage.
///
/// Offsets cover every `(block, register)` column. Positions are strictly
/// increasing within a column, and rd post-values carry into the next block's
/// start value. Raw read and write pre-values are certified only by
/// [`CertifiedRegisterOwner::build`], because they are absent from this ABI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RegisterCsr256 {
    geometry: RegisterGeometry,
    parts: RegisterCsr256Parts,
}

impl RegisterCsr256 {
    pub(crate) fn from_parts(parts: RegisterCsr256Parts) -> Result<Self, RegistersRwV3Error> {
        let geometry = RegisterGeometry::new(parts.cycles)?;
        let csr = Self { geometry, parts };
        csr.validate()?;
        Ok(csr)
    }

    pub(crate) fn validate(&self) -> Result<(), RegistersRwV3Error> {
        require_length(
            "start values",
            self.geometry.block_columns(),
            self.parts.start_values.len(),
        )?;
        validate_plane(
            "rs1",
            self.geometry,
            &self.parts.rs1_offsets,
            &self.parts.rs1_positions,
        )?;
        validate_plane(
            "rs2",
            self.geometry,
            &self.parts.rs2_offsets,
            &self.parts.rs2_positions,
        )?;
        validate_plane(
            "rd",
            self.geometry,
            &self.parts.rd_offsets,
            &self.parts.rd_positions,
        )?;
        require_length(
            "rd post values",
            self.parts.rd_positions.len(),
            self.parts.rd_post_values.len(),
        )?;
        validate_block_state_flow(&self.parts, self.geometry)
    }

    pub(crate) const fn geometry(&self) -> RegisterGeometry {
        self.geometry
    }

    pub(crate) fn event_counts(&self) -> RegisterEventCounts {
        RegisterEventCounts::new(
            self.parts.rs1_positions.len(),
            self.parts.rs2_positions.len(),
            self.parts.rd_positions.len(),
        )
    }

    pub(crate) const fn parts(&self) -> &RegisterCsr256Parts {
        &self.parts
    }

    pub(crate) fn into_parts(self) -> RegisterCsr256Parts {
        self.parts
    }

    pub(super) fn block_len(&self, block: usize) -> Result<usize, RegistersRwV3Error> {
        if block >= self.geometry.blocks() {
            return Err(RegistersRwV3Error::IndexOutOfRange {
                name: "CSR block",
                index: block,
                length: self.geometry.blocks(),
            });
        }
        let start = block
            .checked_mul(REGISTER_CSR_BLOCK_CYCLES)
            .ok_or(RegistersRwV3Error::SizeOverflow("CSR block start"))?;
        Ok((self.geometry.cycles() - start).min(REGISTER_CSR_BLOCK_CYCLES))
    }

    pub(super) fn column(
        &self,
        block: usize,
        register: usize,
    ) -> Result<RegisterCsrColumn<'_>, RegistersRwV3Error> {
        if register >= REGISTER_CSR_COLUMNS {
            return Err(RegistersRwV3Error::IndexOutOfRange {
                name: "CSR register",
                index: register,
                length: REGISTER_CSR_COLUMNS,
            });
        }
        if block >= self.geometry.blocks() {
            return Err(RegistersRwV3Error::IndexOutOfRange {
                name: "CSR block",
                index: block,
                length: self.geometry.blocks(),
            });
        }
        let header = block
            .checked_mul(REGISTER_CSR_COLUMNS)
            .and_then(|base| base.checked_add(register))
            .ok_or(RegistersRwV3Error::SizeOverflow("CSR column header"))?;
        let rs1 = checked_offset_range("rs1", &self.parts.rs1_offsets, header)?;
        let rs2 = checked_offset_range("rs2", &self.parts.rs2_offsets, header)?;
        let rd = checked_offset_range("rd", &self.parts.rd_offsets, header)?;
        let start_value = self.parts.start_values.get(header).copied().ok_or(
            RegistersRwV3Error::IndexOutOfRange {
                name: "start values",
                index: header,
                length: self.parts.start_values.len(),
            },
        )?;
        let rs1_positions =
            self.parts
                .rs1_positions
                .get(rs1)
                .ok_or(RegistersRwV3Error::IndexOutOfRange {
                    name: "rs1 positions",
                    index: header,
                    length: self.parts.rs1_positions.len(),
                })?;
        let rs2_positions =
            self.parts
                .rs2_positions
                .get(rs2)
                .ok_or(RegistersRwV3Error::IndexOutOfRange {
                    name: "rs2 positions",
                    index: header,
                    length: self.parts.rs2_positions.len(),
                })?;
        let rd_positions =
            self.parts
                .rd_positions
                .get(rd.clone())
                .ok_or(RegistersRwV3Error::IndexOutOfRange {
                    name: "rd positions",
                    index: header,
                    length: self.parts.rd_positions.len(),
                })?;
        let rd_post_values =
            self.parts
                .rd_post_values
                .get(rd)
                .ok_or(RegistersRwV3Error::IndexOutOfRange {
                    name: "rd post values",
                    index: header,
                    length: self.parts.rd_post_values.len(),
                })?;
        Ok(RegisterCsrColumn {
            start_value,
            rs1_positions,
            rs2_positions,
            rd_positions,
            rd_post_values,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RegisterStateFlowCertificate {
    cycles: usize,
    events: RegisterEventCounts,
    nonzero_rd_increments: usize,
    initial_values: [u64; REGISTER_CSR_COLUMNS],
    final_values: [u64; REGISTER_CSR_COLUMNS],
}

impl RegisterStateFlowCertificate {
    pub(crate) const fn cycles(&self) -> usize {
        self.cycles
    }

    pub(crate) const fn events(&self) -> RegisterEventCounts {
        self.events
    }

    pub(crate) const fn nonzero_rd_increments(&self) -> usize {
        self.nonzero_rd_increments
    }

    pub(crate) const fn initial_values(&self) -> &[u64; REGISTER_CSR_COLUMNS] {
        &self.initial_values
    }

    pub(crate) const fn final_values(&self) -> &[u64; REGISTER_CSR_COLUMNS] {
        &self.final_values
    }
}

/// Device-bound proof that all nine immutable planes match one CSR census.
///
/// Fields and issuers are private to this package. The type is not `Clone`, so
/// later sequence code can consume it exactly once when taking stage-1 state.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct CertifiedRegisterOwnerReceipt {
    producer: RegisterProducerIdentity,
    census: RegisterCsrCensus,
    layout: RegisterPlaneLayout,
    planes: RegisterPlaneReceipts,
}

impl CertifiedRegisterOwnerReceipt {
    fn issue(
        producer: RegisterProducerIdentity,
        census: RegisterCsrCensus,
        layout: RegisterPlaneLayout,
        planes: RegisterPlaneReceipts,
    ) -> Self {
        Self {
            producer,
            census,
            layout,
            planes,
        }
    }

    pub(crate) const fn producer(&self) -> RegisterProducerIdentity {
        self.producer
    }

    pub(crate) const fn census(&self) -> RegisterCsrCensus {
        self.census
    }

    pub(crate) const fn layout(&self) -> RegisterPlaneLayout {
        self.layout
    }

    pub(crate) fn verify_binding(
        &self,
        device_registry_id: u64,
        generation: u64,
        ordered_prefix_digest: OrderedPrefixDigest,
    ) -> Result<(), RegistersRwV3Error> {
        if self.producer.device_registry_id() != device_registry_id {
            return Err(RegistersRwV3Error::ReceiptDeviceMismatch {
                expected: self.producer.device_registry_id(),
                got: device_registry_id,
            });
        }
        if self.producer.generation() != generation {
            return Err(RegistersRwV3Error::ReceiptGenerationMismatch {
                expected: self.producer.generation(),
                got: generation,
            });
        }
        if self.producer.ordered_prefix_digest() != ordered_prefix_digest {
            return Err(RegistersRwV3Error::ReceiptDigestMismatch);
        }
        Ok(())
    }

    pub(crate) fn allocation_identities(&self) -> [usize; 9] {
        [
            self.planes.start_values.allocation_identity(),
            self.planes.rs1_offsets.allocation_identity(),
            self.planes.rs2_offsets.allocation_identity(),
            self.planes.rd_offsets.allocation_identity(),
            self.planes.rs1_positions.allocation_identity(),
            self.planes.rs2_positions.allocation_identity(),
            self.planes.rd_positions.allocation_identity(),
            self.planes.rd_post_values.allocation_identity(),
            self.planes.rd_inc.allocation_identity(),
        ]
    }

    pub(super) const fn start_values(&self) -> &AllocationReceipt<StartValues> {
        &self.planes.start_values
    }

    pub(super) const fn rs1_offsets(&self) -> &AllocationReceipt<Rs1Offsets> {
        &self.planes.rs1_offsets
    }

    pub(super) const fn rs2_offsets(&self) -> &AllocationReceipt<Rs2Offsets> {
        &self.planes.rs2_offsets
    }

    pub(super) const fn rd_offsets(&self) -> &AllocationReceipt<RdOffsets> {
        &self.planes.rd_offsets
    }

    pub(super) const fn rs1_positions(&self) -> &AllocationReceipt<Rs1Positions> {
        &self.planes.rs1_positions
    }

    pub(super) const fn rs2_positions(&self) -> &AllocationReceipt<Rs2Positions> {
        &self.planes.rs2_positions
    }

    pub(super) const fn rd_positions(&self) -> &AllocationReceipt<RdPositions> {
        &self.planes.rd_positions
    }

    pub(super) const fn rd_post_values(&self) -> &AllocationReceipt<RdPostValues> {
        &self.planes.rd_post_values
    }

    pub(super) const fn rd_inc(&self) -> &AllocationReceipt<RdInc> {
        &self.planes.rd_inc
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct CertifiedRegisterOwner {
    csr: RegisterCsr256,
    state_flow: RegisterStateFlowCertificate,
    receipt: CertifiedRegisterOwnerReceipt,
}

impl CertifiedRegisterOwner {
    /// Builds CSR storage and validates every raw value before issuing device receipts.
    pub(crate) fn build(
        producer: RegisterProducerIdentity,
        allocations: &RegisterPlaneAllocations,
        rows: &[RegisterRow],
        initial_values: &[u64; REGISTER_CSR_COLUMNS],
    ) -> Result<Self, RegistersRwV3Error> {
        let geometry = RegisterGeometry::new(rows.len())?;
        if producer.cycles() != geometry.cycles() {
            return Err(RegistersRwV3Error::ProducerCycleMismatch {
                expected: geometry.cycles(),
                got: producer.cycles(),
            });
        }
        let (csr, state_flow) = build_csr(rows, initial_values, geometry)?;
        let census = RegisterCsrCensus::new(geometry, csr.event_counts())?;
        let layout = RegisterPlaneLayout::new(census)?;
        let planes = RegisterPlaneReceipts::admit(producer, layout, allocations)?;
        let receipt = CertifiedRegisterOwnerReceipt::issue(producer, census, layout, planes);
        Ok(Self {
            csr,
            state_flow,
            receipt,
        })
    }

    pub(crate) const fn csr(&self) -> &RegisterCsr256 {
        &self.csr
    }

    pub(crate) const fn state_flow(&self) -> &RegisterStateFlowCertificate {
        &self.state_flow
    }

    pub(crate) const fn receipt(&self) -> &CertifiedRegisterOwnerReceipt {
        &self.receipt
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        RegisterCsr256,
        RegisterStateFlowCertificate,
        CertifiedRegisterOwnerReceipt,
    ) {
        (self.csr, self.state_flow, self.receipt)
    }
}

fn build_csr(
    rows: &[RegisterRow],
    initial_values: &[u64; REGISTER_CSR_COLUMNS],
    geometry: RegisterGeometry,
) -> Result<(RegisterCsr256, RegisterStateFlowCertificate), RegistersRwV3Error> {
    let offset_capacity = geometry.offset_entries()?;
    let mut start_values = Vec::with_capacity(geometry.block_columns());
    let mut rs1_offsets = Vec::with_capacity(offset_capacity);
    let mut rs2_offsets = Vec::with_capacity(offset_capacity);
    let mut rd_offsets = Vec::with_capacity(offset_capacity);
    rs1_offsets.push(0);
    rs2_offsets.push(0);
    rd_offsets.push(0);

    let mut rs1_positions = Vec::new();
    let mut rs2_positions = Vec::new();
    let mut rd_positions = Vec::new();
    let mut rd_post_values = Vec::new();
    let mut rs1_by_register: [Vec<u8>; REGISTER_CSR_COLUMNS] = core::array::from_fn(|_| Vec::new());
    let mut rs2_by_register: [Vec<u8>; REGISTER_CSR_COLUMNS] = core::array::from_fn(|_| Vec::new());
    let mut rd_by_register: [Vec<u8>; REGISTER_CSR_COLUMNS] = core::array::from_fn(|_| Vec::new());
    let mut rd_posts_by_register: [Vec<u64>; REGISTER_CSR_COLUMNS] =
        core::array::from_fn(|_| Vec::new());
    let mut state = *initial_values;
    let mut nonzero_rd_increments = 0usize;

    for (block, block_rows) in rows.chunks(REGISTER_CSR_BLOCK_CYCLES).enumerate() {
        start_values.extend_from_slice(&state);
        for register in 0..REGISTER_CSR_COLUMNS {
            rs1_by_register[register].clear();
            rs2_by_register[register].clear();
            rd_by_register[register].clear();
            rd_posts_by_register[register].clear();
        }

        for (position, row) in block_rows.iter().copied().enumerate() {
            let cycle = block
                .checked_mul(REGISTER_CSR_BLOCK_CYCLES)
                .and_then(|base| base.checked_add(position))
                .ok_or(RegistersRwV3Error::SizeOverflow("cycle index"))?;
            let local = u8::try_from(position)
                .map_err(|_| RegistersRwV3Error::SizeOverflow("CSR local position"))?;
            if let Some(read) = row.rs1() {
                let register = checked_register(cycle, "rs1", read.register())?;
                check_read(cycle, "rs1", read, state[register])?;
                rs1_by_register[register].push(local);
            }
            if let Some(read) = row.rs2() {
                let register = checked_register(cycle, "rs2", read.register())?;
                check_read(cycle, "rs2", read, state[register])?;
                rs2_by_register[register].push(local);
            }
            if let Some(write) = row.rd() {
                let register = checked_register(cycle, "rd", write.register())?;
                let expected = state[register];
                if write.pre_value() != expected {
                    return Err(RegistersRwV3Error::WritePreValueMismatch {
                        cycle,
                        register: write.register(),
                        expected,
                        got: write.pre_value(),
                    });
                }
                if write.post_value() != write.pre_value() {
                    nonzero_rd_increments = nonzero_rd_increments.checked_add(1).ok_or(
                        RegistersRwV3Error::SizeOverflow("nonzero rd increment count"),
                    )?;
                }
                rd_by_register[register].push(local);
                rd_posts_by_register[register].push(write.post_value());
                state[register] = write.post_value();
            }
        }

        for register in 0..REGISTER_CSR_COLUMNS {
            rs1_positions.extend_from_slice(&rs1_by_register[register]);
            rs1_offsets.push(event_offset("rs1", rs1_positions.len())?);
            rs2_positions.extend_from_slice(&rs2_by_register[register]);
            rs2_offsets.push(event_offset("rs2", rs2_positions.len())?);
            rd_positions.extend_from_slice(&rd_by_register[register]);
            rd_post_values.extend_from_slice(&rd_posts_by_register[register]);
            rd_offsets.push(event_offset("rd", rd_positions.len())?);
        }
    }

    let csr = RegisterCsr256::from_parts(RegisterCsr256Parts {
        cycles: geometry.cycles(),
        start_values,
        rs1_offsets,
        rs2_offsets,
        rd_offsets,
        rs1_positions,
        rs2_positions,
        rd_positions,
        rd_post_values,
    })?;
    let state_flow = RegisterStateFlowCertificate {
        cycles: geometry.cycles(),
        events: csr.event_counts(),
        nonzero_rd_increments,
        initial_values: *initial_values,
        final_values: state,
    };
    Ok((csr, state_flow))
}

fn validate_plane(
    plane: &'static str,
    geometry: RegisterGeometry,
    offsets: &[u32],
    positions: &[u8],
) -> Result<(), RegistersRwV3Error> {
    let expected_offsets = geometry.offset_entries()?;
    require_length(plane, expected_offsets, offsets.len())?;
    let first = offsets
        .first()
        .copied()
        .ok_or(RegistersRwV3Error::PlaneLength {
            plane,
            expected: expected_offsets,
            got: 0,
        })?;
    if first != 0 {
        return Err(RegistersRwV3Error::OffsetStart { plane, got: first });
    }
    for (header, pair) in offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(RegistersRwV3Error::OffsetOrder {
                plane,
                header,
                start: pair[0],
                end: pair[1],
            });
        }
    }
    let terminal = offsets.last().copied().unwrap_or_default();
    if terminal as usize != positions.len() {
        return Err(RegistersRwV3Error::OffsetTerminal {
            plane,
            expected: positions.len(),
            got: terminal,
        });
    }

    for block in 0..geometry.blocks() {
        let block_len = if block + 1 == geometry.blocks() {
            let block_start = block * REGISTER_CSR_BLOCK_CYCLES;
            geometry.cycles() - block_start
        } else {
            REGISTER_CSR_BLOCK_CYCLES
        };
        let mut seen = [false; REGISTER_CSR_BLOCK_CYCLES];
        for register in 0..REGISTER_CSR_COLUMNS {
            let header = block * REGISTER_CSR_COLUMNS + register;
            let range = checked_offset_range(plane, offsets, header)?;
            let column = positions
                .get(range)
                .ok_or(RegistersRwV3Error::OffsetTerminal {
                    plane,
                    expected: positions.len(),
                    got: terminal,
                })?;
            if column.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(RegistersRwV3Error::PositionOrder { plane, header });
            }
            for &position in column {
                let local = usize::from(position);
                if local >= block_len {
                    return Err(RegistersRwV3Error::PositionOutOfBlock {
                        plane,
                        block,
                        block_len,
                        position,
                    });
                }
                if core::mem::replace(&mut seen[local], true) {
                    return Err(RegistersRwV3Error::DuplicateCycleEvent {
                        plane,
                        cycle: block * REGISTER_CSR_BLOCK_CYCLES + local,
                    });
                }
            }
        }
    }
    Ok(())
}

fn validate_block_state_flow(
    parts: &RegisterCsr256Parts,
    geometry: RegisterGeometry,
) -> Result<(), RegistersRwV3Error> {
    for register in 0..REGISTER_CSR_COLUMNS {
        let mut expected =
            parts
                .start_values
                .get(register)
                .copied()
                .ok_or(RegistersRwV3Error::PlaneLength {
                    plane: "start values",
                    expected: geometry.block_columns(),
                    got: parts.start_values.len(),
                })?;
        for block in 0..geometry.blocks() {
            let header = block * REGISTER_CSR_COLUMNS + register;
            let got =
                parts
                    .start_values
                    .get(header)
                    .copied()
                    .ok_or(RegistersRwV3Error::PlaneLength {
                        plane: "start values",
                        expected: geometry.block_columns(),
                        got: parts.start_values.len(),
                    })?;
            if got != expected {
                return Err(RegistersRwV3Error::BlockStateMismatch {
                    block,
                    register,
                    expected,
                    got,
                });
            }
            let range = checked_offset_range("rd", &parts.rd_offsets, header)?;
            let posts = parts
                .rd_post_values
                .get(range)
                .ok_or(RegistersRwV3Error::PlaneLength {
                    plane: "rd post values",
                    expected: parts.rd_positions.len(),
                    got: parts.rd_post_values.len(),
                })?;
            if let Some(last) = posts.last() {
                expected = *last;
            }
        }
    }
    Ok(())
}

fn checked_register(
    cycle: usize,
    access: &'static str,
    register: u8,
) -> Result<usize, RegistersRwV3Error> {
    let index = usize::from(register);
    if index >= REGISTER_CSR_COLUMNS {
        Err(RegistersRwV3Error::InvalidRegister {
            cycle,
            access,
            register,
        })
    } else {
        Ok(index)
    }
}

fn check_read(
    cycle: usize,
    access: &'static str,
    read: RegisterRead,
    expected: u64,
) -> Result<(), RegistersRwV3Error> {
    if read.value() != expected {
        Err(RegistersRwV3Error::ReadValueMismatch {
            cycle,
            access,
            register: read.register(),
            expected,
            got: read.value(),
        })
    } else {
        Ok(())
    }
}

fn event_offset(plane: &'static str, events: usize) -> Result<u32, RegistersRwV3Error> {
    u32::try_from(events).map_err(|_| RegistersRwV3Error::EventCountOverflow { plane })
}

fn checked_offset_range(
    plane: &'static str,
    offsets: &[u32],
    header: usize,
) -> Result<Range<usize>, RegistersRwV3Error> {
    let start = offsets
        .get(header)
        .copied()
        .ok_or(RegistersRwV3Error::IndexOutOfRange {
            name: plane,
            index: header,
            length: offsets.len(),
        })? as usize;
    let next = header
        .checked_add(1)
        .ok_or(RegistersRwV3Error::SizeOverflow("offset header"))?;
    let end = offsets
        .get(next)
        .copied()
        .ok_or(RegistersRwV3Error::IndexOutOfRange {
            name: plane,
            index: next,
            length: offsets.len(),
        })? as usize;
    Ok(start..end)
}

fn require_length(
    plane: &'static str,
    expected: usize,
    got: usize,
) -> Result<(), RegistersRwV3Error> {
    if got == expected {
        Ok(())
    } else {
        Err(RegistersRwV3Error::PlaneLength {
            plane,
            expected,
            got,
        })
    }
}
