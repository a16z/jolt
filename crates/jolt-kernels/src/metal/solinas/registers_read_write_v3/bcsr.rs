use core::{
    mem::size_of,
    num::{NonZeroU64, NonZeroUsize},
};

use jolt_field::{AkitaField, FromPrimitiveInt};

use super::super::registers_val_evaluation_backend::RegistersValResidentInputAbi;
use super::super::{Fp128, AKITA_OFFSET_FFFFA7F7};
use super::abi::{
    OrderedPrefixDigest, RegisterEventCounts, REGISTER_CSR_BLOCK_CYCLES, REGISTER_CSR_COLUMNS,
    REGISTER_FP128_BYTES,
};
use super::owner::{RegisterRead, RegisterRow, RegisterWrite};
use super::RegistersRwV3Error;

pub(crate) const REGISTER_BCSR_OFFSET_ENTRIES: usize = REGISTER_CSR_COLUMNS + 1;
pub(crate) const REGISTER_BCSR_POSITION_SLOTS: usize = REGISTER_CSR_BLOCK_CYCLES;
pub(crate) const REGISTER_ABSENT_INDEX: u8 = u8::MAX;

const REGISTER_BCSR_PLANE_COUNT: usize = 10;

const _: () = assert!(REGISTER_BCSR_OFFSET_ENTRIES == 129);
const _: () = assert!(REGISTER_BCSR_POSITION_SLOTS == 256);
const _: () = assert!(REGISTER_CSR_COLUMNS < REGISTER_ABSENT_INDEX as usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterBcsrGeometry {
    cycles: usize,
    blocks: usize,
}

impl RegisterBcsrGeometry {
    pub(crate) fn new(cycles: usize) -> Result<Self, RegistersRwV3Error> {
        if cycles == 0 || u32::try_from(cycles).is_err() {
            return Err(RegistersRwV3Error::InvalidBcsrCycleCount(cycles));
        }
        Ok(Self {
            cycles,
            blocks: cycles.div_ceil(REGISTER_BCSR_POSITION_SLOTS),
        })
    }

    pub(crate) const fn cycles(self) -> usize {
        self.cycles
    }

    pub(crate) const fn blocks(self) -> usize {
        self.blocks
    }

    pub(crate) fn block_len(self, block: usize) -> Result<usize, RegistersRwV3Error> {
        if block >= self.blocks {
            return Err(RegistersRwV3Error::IndexOutOfRange {
                name: "BCSR block",
                index: block,
                length: self.blocks,
            });
        }
        let start = block
            .checked_mul(REGISTER_BCSR_POSITION_SLOTS)
            .ok_or(RegistersRwV3Error::SizeOverflow("BCSR block start"))?;
        Ok((self.cycles - start).min(REGISTER_BCSR_POSITION_SLOTS))
    }

    fn checked_elements(
        self,
        name: &'static str,
        elements_per_block: usize,
    ) -> Result<usize, RegistersRwV3Error> {
        self.blocks
            .checked_mul(elements_per_block)
            .ok_or(RegistersRwV3Error::SizeOverflow(name))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterBcsrPlaneShape {
    elements: usize,
    bytes: usize,
}

impl RegisterBcsrPlaneShape {
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

/// Primitive-element layout of the padded block planes.
///
/// The three offset planes contain 129 `u16` values for each block. Event
/// planes and rd post-values reserve 256 slots for each block; the terminal
/// offset determines which prefix is initialized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterBcsrLayout {
    geometry: RegisterBcsrGeometry,
    start_values: RegisterBcsrPlaneShape,
    offsets: RegisterBcsrPlaneShape,
    positions: RegisterBcsrPlaneShape,
    rd_post_values: RegisterBcsrPlaneShape,
    rd_index: RegisterBcsrPlaneShape,
    rd_inc: RegisterBcsrPlaneShape,
}

impl RegisterBcsrLayout {
    pub(crate) fn new(geometry: RegisterBcsrGeometry) -> Result<Self, RegistersRwV3Error> {
        Ok(Self {
            geometry,
            start_values: RegisterBcsrPlaneShape::new(
                "BCSR start-value plane",
                geometry.checked_elements("BCSR start-value elements", REGISTER_CSR_COLUMNS)?,
                size_of::<u64>(),
            )?,
            offsets: RegisterBcsrPlaneShape::new(
                "BCSR offset plane",
                geometry.checked_elements("BCSR offset elements", REGISTER_BCSR_OFFSET_ENTRIES)?,
                size_of::<u16>(),
            )?,
            positions: RegisterBcsrPlaneShape::new(
                "BCSR position plane",
                geometry
                    .checked_elements("BCSR position elements", REGISTER_BCSR_POSITION_SLOTS)?,
                size_of::<u8>(),
            )?,
            rd_post_values: RegisterBcsrPlaneShape::new(
                "BCSR rd post-value plane",
                geometry.checked_elements(
                    "BCSR rd post-value elements",
                    REGISTER_BCSR_POSITION_SLOTS,
                )?,
                size_of::<u64>(),
            )?,
            rd_index: RegisterBcsrPlaneShape::new(
                "BCSR rd-index plane",
                geometry.cycles(),
                size_of::<u8>(),
            )?,
            rd_inc: RegisterBcsrPlaneShape::new(
                "BCSR rd-increment plane",
                geometry.cycles(),
                REGISTER_FP128_BYTES,
            )?,
        })
    }

    pub(crate) const fn geometry(self) -> RegisterBcsrGeometry {
        self.geometry
    }

    pub(crate) const fn start_values(self) -> RegisterBcsrPlaneShape {
        self.start_values
    }

    pub(crate) const fn offsets(self) -> RegisterBcsrPlaneShape {
        self.offsets
    }

    pub(crate) const fn positions(self) -> RegisterBcsrPlaneShape {
        self.positions
    }

    pub(crate) const fn rd_post_values(self) -> RegisterBcsrPlaneShape {
        self.rd_post_values
    }

    pub(crate) const fn rd_index(self) -> RegisterBcsrPlaneShape {
        self.rd_index
    }

    pub(crate) const fn rd_inc(self) -> RegisterBcsrPlaneShape {
        self.rd_inc
    }

    pub(crate) fn topology_bytes(self) -> Result<usize, RegistersRwV3Error> {
        checked_sum(&[
            self.start_values.bytes(),
            checked_product("three BCSR offset planes", self.offsets.bytes(), 3)?,
            checked_product("three BCSR position planes", self.positions.bytes(), 3)?,
            self.rd_post_values.bytes(),
        ])
    }

    pub(crate) fn registers_val_bytes(self) -> Result<usize, RegistersRwV3Error> {
        checked_sum(&[self.rd_index.bytes(), self.rd_inc.bytes()])
    }

    pub(crate) fn producer_bytes(self) -> Result<usize, RegistersRwV3Error> {
        checked_sum(&[self.topology_bytes()?, self.registers_val_bytes()?])
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RegisterBcsr256Parts {
    pub(crate) cycles: usize,
    pub(crate) start_values: Vec<[u64; REGISTER_CSR_COLUMNS]>,
    pub(crate) rs1_offsets: Vec<[u16; REGISTER_BCSR_OFFSET_ENTRIES]>,
    pub(crate) rs2_offsets: Vec<[u16; REGISTER_BCSR_OFFSET_ENTRIES]>,
    pub(crate) rd_offsets: Vec<[u16; REGISTER_BCSR_OFFSET_ENTRIES]>,
    pub(crate) rs1_positions: Vec<[u8; REGISTER_BCSR_POSITION_SLOTS]>,
    pub(crate) rs2_positions: Vec<[u8; REGISTER_BCSR_POSITION_SLOTS]>,
    pub(crate) rd_positions: Vec<[u8; REGISTER_BCSR_POSITION_SLOTS]>,
    pub(crate) rd_post_values: Vec<[u64; REGISTER_BCSR_POSITION_SLOTS]>,
    pub(crate) rd_index: Vec<u8>,
    pub(crate) rd_inc: Vec<Fp128>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterBcsrReadEvent {
    cycle: usize,
    register: u8,
}

impl RegisterBcsrReadEvent {
    pub(crate) const fn cycle(self) -> usize {
        self.cycle
    }

    pub(crate) const fn register(self) -> u8 {
        self.register
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterBcsrWriteEvent {
    cycle: usize,
    register: u8,
    post_value: u64,
}

struct RegisterBcsrBlockEventMaps {
    rs1: [u8; REGISTER_BCSR_POSITION_SLOTS],
    rs2: [u8; REGISTER_BCSR_POSITION_SLOTS],
    rd: [u8; REGISTER_BCSR_POSITION_SLOTS],
    rd_post_values: [u64; REGISTER_BCSR_POSITION_SLOTS],
}

impl RegisterBcsrWriteEvent {
    pub(crate) const fn cycle(self) -> usize {
        self.cycle
    }

    pub(crate) const fn register(self) -> u8 {
        self.register
    }

    pub(crate) const fn post_value(self) -> u64 {
        self.post_value
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RegisterBcsrStateFlowCertificate {
    geometry: RegisterBcsrGeometry,
    events: RegisterEventCounts,
    zero_delta_writes: usize,
    initial_values: [u64; REGISTER_CSR_COLUMNS],
    final_values: [u64; REGISTER_CSR_COLUMNS],
}

impl RegisterBcsrStateFlowCertificate {
    pub(crate) const fn geometry(&self) -> RegisterBcsrGeometry {
        self.geometry
    }

    pub(crate) const fn events(&self) -> RegisterEventCounts {
        self.events
    }

    pub(crate) const fn zero_delta_writes(&self) -> usize {
        self.zero_delta_writes
    }

    pub(crate) const fn initial_values(&self) -> &[u64; REGISTER_CSR_COLUMNS] {
        &self.initial_values
    }

    pub(crate) const fn final_values(&self) -> &[u64; REGISTER_CSR_COLUMNS] {
        &self.final_values
    }
}

/// Checked scalar form of the BCSR-256 device planes.
///
/// Construction from rows checks read values, write pre-values, register
/// bounds, and the carried register state at every block boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RegisterBcsr256 {
    geometry: RegisterBcsrGeometry,
    parts: RegisterBcsr256Parts,
}

impl RegisterBcsr256 {
    pub(crate) fn from_parts(parts: RegisterBcsr256Parts) -> Result<Self, RegistersRwV3Error> {
        let geometry = RegisterBcsrGeometry::new(parts.cycles)?;
        let value = Self { geometry, parts };
        value.validate()?;
        Ok(value)
    }

    pub(crate) fn from_rows(
        rows: &[RegisterRow],
        initial_values: &[u64; REGISTER_CSR_COLUMNS],
    ) -> Result<(Self, RegisterBcsrStateFlowCertificate), RegistersRwV3Error> {
        let geometry = RegisterBcsrGeometry::new(rows.len())?;
        let layout = RegisterBcsrLayout::new(geometry)?;
        let mut parts = empty_parts(geometry, layout);
        let mut state = *initial_values;
        let mut zero_delta_writes = 0usize;

        for (block, block_rows) in rows.chunks(REGISTER_BCSR_POSITION_SLOTS).enumerate() {
            parts.start_values[block] = state;
            let mut rs1_by_register: [Vec<u8>; REGISTER_CSR_COLUMNS] =
                core::array::from_fn(|_| Vec::new());
            let mut rs2_by_register: [Vec<u8>; REGISTER_CSR_COLUMNS] =
                core::array::from_fn(|_| Vec::new());
            let mut rd_by_register: [Vec<u8>; REGISTER_CSR_COLUMNS] =
                core::array::from_fn(|_| Vec::new());
            let mut rd_posts_by_register: [Vec<u64>; REGISTER_CSR_COLUMNS] =
                core::array::from_fn(|_| Vec::new());

            for (position, row) in block_rows.iter().copied().enumerate() {
                let cycle = block
                    .checked_mul(REGISTER_BCSR_POSITION_SLOTS)
                    .and_then(|start| start.checked_add(position))
                    .ok_or(RegistersRwV3Error::SizeOverflow("BCSR cycle index"))?;
                let local = u8::try_from(position)
                    .map_err(|_| RegistersRwV3Error::SizeOverflow("BCSR local position"))?;

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
                    if write.pre_value() == write.post_value() {
                        zero_delta_writes = zero_delta_writes.checked_add(1).ok_or(
                            RegistersRwV3Error::SizeOverflow("BCSR zero-delta write count"),
                        )?;
                    }
                    rd_by_register[register].push(local);
                    rd_posts_by_register[register].push(write.post_value());
                    parts.rd_index[cycle] = write.register();
                    parts.rd_inc[cycle] = fp128_increment(write.pre_value(), write.post_value());
                    state[register] = write.post_value();
                }
            }

            flatten_positions(
                "rs1",
                &rs1_by_register,
                &mut parts.rs1_offsets[block],
                &mut parts.rs1_positions[block],
            )?;
            flatten_positions(
                "rs2",
                &rs2_by_register,
                &mut parts.rs2_offsets[block],
                &mut parts.rs2_positions[block],
            )?;
            flatten_writes(
                &rd_by_register,
                &rd_posts_by_register,
                &mut parts.rd_offsets[block],
                &mut parts.rd_positions[block],
                &mut parts.rd_post_values[block],
            )?;
        }

        let value = Self::from_parts(parts)?;
        let certificate = RegisterBcsrStateFlowCertificate {
            geometry,
            events: value.event_counts(),
            zero_delta_writes,
            initial_values: *initial_values,
            final_values: state,
        };
        Ok((value, certificate))
    }

    pub(crate) fn validate(&self) -> Result<(), RegistersRwV3Error> {
        let blocks = self.geometry.blocks();
        for (plane, got) in [
            ("BCSR start values", self.parts.start_values.len()),
            ("BCSR rs1 offsets", self.parts.rs1_offsets.len()),
            ("BCSR rs2 offsets", self.parts.rs2_offsets.len()),
            ("BCSR rd offsets", self.parts.rd_offsets.len()),
            ("BCSR rs1 positions", self.parts.rs1_positions.len()),
            ("BCSR rs2 positions", self.parts.rs2_positions.len()),
            ("BCSR rd positions", self.parts.rd_positions.len()),
            ("BCSR rd post values", self.parts.rd_post_values.len()),
        ] {
            require_length(plane, blocks, got)?;
        }
        require_length(
            "BCSR rd index",
            self.geometry.cycles(),
            self.parts.rd_index.len(),
        )?;
        require_length(
            "BCSR rd increment",
            self.geometry.cycles(),
            self.parts.rd_inc.len(),
        )?;

        validate_position_plane(
            "rs1",
            self.geometry,
            &self.parts.rs1_offsets,
            &self.parts.rs1_positions,
        )?;
        validate_position_plane(
            "rs2",
            self.geometry,
            &self.parts.rs2_offsets,
            &self.parts.rs2_positions,
        )?;
        validate_position_plane(
            "rd",
            self.geometry,
            &self.parts.rd_offsets,
            &self.parts.rd_positions,
        )?;
        validate_post_padding(
            self.geometry,
            &self.parts.rd_offsets,
            &self.parts.rd_post_values,
        )?;
        validate_rd_index(&self.parts, self.geometry)?;
        validate_block_state_flow(&self.parts, self.geometry)?;
        validate_rd_increment(&self.parts, self.geometry)
    }

    pub(crate) const fn geometry(&self) -> RegisterBcsrGeometry {
        self.geometry
    }

    pub(crate) const fn parts(&self) -> &RegisterBcsr256Parts {
        &self.parts
    }

    pub(crate) fn into_parts(self) -> RegisterBcsr256Parts {
        self.parts
    }

    pub(crate) fn event_counts(&self) -> RegisterEventCounts {
        RegisterEventCounts::new(
            terminal_sum(&self.parts.rs1_offsets),
            terminal_sum(&self.parts.rs2_offsets),
            terminal_sum(&self.parts.rd_offsets),
        )
    }

    pub(crate) fn block_start_values(
        &self,
        block: usize,
    ) -> Result<&[u64; REGISTER_CSR_COLUMNS], RegistersRwV3Error> {
        self.parts
            .start_values
            .get(block)
            .ok_or(RegistersRwV3Error::IndexOutOfRange {
                name: "BCSR block start values",
                index: block,
                length: self.geometry.blocks(),
            })
    }

    pub(crate) fn rd_index(&self) -> &[u8] {
        &self.parts.rd_index
    }

    pub(crate) fn rd_inc(&self) -> &[Fp128] {
        &self.parts.rd_inc
    }

    pub(crate) fn rs1_events(&self) -> Result<Vec<RegisterBcsrReadEvent>, RegistersRwV3Error> {
        self.read_events(&self.parts.rs1_offsets, &self.parts.rs1_positions)
    }

    pub(crate) fn rs2_events(&self) -> Result<Vec<RegisterBcsrReadEvent>, RegistersRwV3Error> {
        self.read_events(&self.parts.rs2_offsets, &self.parts.rs2_positions)
    }

    pub(crate) fn rd_events(&self) -> Result<Vec<RegisterBcsrWriteEvent>, RegistersRwV3Error> {
        let mut events = Vec::with_capacity(self.event_counts().rd());
        for block in 0..self.geometry.blocks() {
            let maps = self.block_event_maps(block)?;
            let block_len = self.geometry.block_len(block)?;
            let cycle_base = block * REGISTER_BCSR_POSITION_SLOTS;
            for (local, (&register, &post_value)) in maps
                .rd
                .iter()
                .zip(&maps.rd_post_values)
                .take(block_len)
                .enumerate()
            {
                if register != REGISTER_ABSENT_INDEX {
                    events.push(RegisterBcsrWriteEvent {
                        cycle: cycle_base + local,
                        register,
                        post_value,
                    });
                }
            }
        }
        Ok(events)
    }

    pub(crate) fn reconstruct_rows(&self) -> Result<Vec<RegisterRow>, RegistersRwV3Error> {
        let mut rows = Vec::with_capacity(self.geometry.cycles());
        let mut state = *self.block_start_values(0)?;
        for block in 0..self.geometry.blocks() {
            let block_start = self.block_start_values(block)?;
            for (register, (&expected, &got)) in state.iter().zip(block_start).enumerate() {
                if expected != got {
                    return Err(RegistersRwV3Error::BlockStateMismatch {
                        block,
                        register,
                        expected,
                        got,
                    });
                }
            }
            let maps = self.block_event_maps(block)?;
            let block_len = self.geometry.block_len(block)?;
            let row_maps = maps
                .rs1
                .iter()
                .zip(&maps.rs2)
                .zip(&maps.rd)
                .zip(&maps.rd_post_values);
            for (((&rs1_index, &rs2_index), &rd_index), &post_value) in row_maps.take(block_len) {
                let rs1 = read_from_map(rs1_index, &state);
                let rs2 = read_from_map(rs2_index, &state);
                let rd = if rd_index == REGISTER_ABSENT_INDEX {
                    None
                } else {
                    let register = usize::from(rd_index);
                    let write = RegisterWrite::new(rd_index, state[register], post_value);
                    state[register] = post_value;
                    Some(write)
                };
                rows.push(RegisterRow::new(rs1, rs2, rd));
            }
        }
        Ok(rows)
    }

    fn read_events(
        &self,
        offsets: &[[u16; REGISTER_BCSR_OFFSET_ENTRIES]],
        positions: &[[u8; REGISTER_BCSR_POSITION_SLOTS]],
    ) -> Result<Vec<RegisterBcsrReadEvent>, RegistersRwV3Error> {
        let mut events = Vec::with_capacity(terminal_sum(offsets));
        for block in 0..self.geometry.blocks() {
            let map = event_register_map(&offsets[block], &positions[block]);
            let block_len = self.geometry.block_len(block)?;
            let cycle_base = block * REGISTER_BCSR_POSITION_SLOTS;
            for (local, &register) in map.iter().take(block_len).enumerate() {
                if register != REGISTER_ABSENT_INDEX {
                    events.push(RegisterBcsrReadEvent {
                        cycle: cycle_base + local,
                        register,
                    });
                }
            }
        }
        Ok(events)
    }

    fn block_event_maps(
        &self,
        block: usize,
    ) -> Result<RegisterBcsrBlockEventMaps, RegistersRwV3Error> {
        if block >= self.geometry.blocks() {
            return Err(RegistersRwV3Error::IndexOutOfRange {
                name: "BCSR block",
                index: block,
                length: self.geometry.blocks(),
            });
        }
        let rs1 = event_register_map(
            &self.parts.rs1_offsets[block],
            &self.parts.rs1_positions[block],
        );
        let rs2 = event_register_map(
            &self.parts.rs2_offsets[block],
            &self.parts.rs2_positions[block],
        );
        let rd = event_register_map(
            &self.parts.rd_offsets[block],
            &self.parts.rd_positions[block],
        );
        let mut posts = [0; REGISTER_BCSR_POSITION_SLOTS];
        for register in 0..REGISTER_CSR_COLUMNS {
            let range = offset_range(&self.parts.rd_offsets[block], register);
            for index in range {
                posts[usize::from(self.parts.rd_positions[block][index])] =
                    self.parts.rd_post_values[block][index];
            }
        }
        Ok(RegisterBcsrBlockEventMaps {
            rs1,
            rs2,
            rd,
            rd_post_values: posts,
        })
    }
}

fn empty_parts(geometry: RegisterBcsrGeometry, layout: RegisterBcsrLayout) -> RegisterBcsr256Parts {
    debug_assert_eq!(geometry, layout.geometry());
    RegisterBcsr256Parts {
        cycles: geometry.cycles(),
        start_values: vec![[0; REGISTER_CSR_COLUMNS]; geometry.blocks()],
        rs1_offsets: vec![[0; REGISTER_BCSR_OFFSET_ENTRIES]; geometry.blocks()],
        rs2_offsets: vec![[0; REGISTER_BCSR_OFFSET_ENTRIES]; geometry.blocks()],
        rd_offsets: vec![[0; REGISTER_BCSR_OFFSET_ENTRIES]; geometry.blocks()],
        rs1_positions: vec![[0; REGISTER_BCSR_POSITION_SLOTS]; geometry.blocks()],
        rs2_positions: vec![[0; REGISTER_BCSR_POSITION_SLOTS]; geometry.blocks()],
        rd_positions: vec![[0; REGISTER_BCSR_POSITION_SLOTS]; geometry.blocks()],
        rd_post_values: vec![[0; REGISTER_BCSR_POSITION_SLOTS]; geometry.blocks()],
        rd_index: vec![REGISTER_ABSENT_INDEX; layout.rd_index().elements()],
        rd_inc: vec![Fp128::ZERO; layout.rd_inc().elements()],
    }
}

fn flatten_positions(
    plane: &'static str,
    by_register: &[Vec<u8>; REGISTER_CSR_COLUMNS],
    offsets: &mut [u16; REGISTER_BCSR_OFFSET_ENTRIES],
    positions: &mut [u8; REGISTER_BCSR_POSITION_SLOTS],
) -> Result<(), RegistersRwV3Error> {
    let mut cursor = 0usize;
    offsets[0] = 0;
    for (register, register_positions) in by_register.iter().enumerate() {
        let end = cursor
            .checked_add(register_positions.len())
            .ok_or(RegistersRwV3Error::SizeOverflow("BCSR position cursor"))?;
        if end > REGISTER_BCSR_POSITION_SLOTS {
            return Err(RegistersRwV3Error::BcsrOffsetTerminal {
                plane,
                block: 0,
                maximum: REGISTER_BCSR_POSITION_SLOTS,
                got: u16::try_from(end).unwrap_or(u16::MAX),
            });
        }
        positions[cursor..end].copy_from_slice(register_positions);
        cursor = end;
        offsets[register + 1] = u16::try_from(cursor)
            .map_err(|_| RegistersRwV3Error::SizeOverflow("BCSR offset sentinel"))?;
    }
    Ok(())
}

fn flatten_writes(
    by_register: &[Vec<u8>; REGISTER_CSR_COLUMNS],
    posts_by_register: &[Vec<u64>; REGISTER_CSR_COLUMNS],
    offsets: &mut [u16; REGISTER_BCSR_OFFSET_ENTRIES],
    positions: &mut [u8; REGISTER_BCSR_POSITION_SLOTS],
    post_values: &mut [u64; REGISTER_BCSR_POSITION_SLOTS],
) -> Result<(), RegistersRwV3Error> {
    let mut cursor = 0usize;
    offsets[0] = 0;
    for register in 0..REGISTER_CSR_COLUMNS {
        if by_register[register].len() != posts_by_register[register].len() {
            return Err(RegistersRwV3Error::PlaneLength {
                plane: "BCSR rd post values",
                expected: by_register[register].len(),
                got: posts_by_register[register].len(),
            });
        }
        let end = cursor.checked_add(by_register[register].len()).ok_or(
            RegistersRwV3Error::SizeOverflow("BCSR write position cursor"),
        )?;
        if end > REGISTER_BCSR_POSITION_SLOTS {
            return Err(RegistersRwV3Error::BcsrOffsetTerminal {
                plane: "rd",
                block: 0,
                maximum: REGISTER_BCSR_POSITION_SLOTS,
                got: u16::try_from(end).unwrap_or(u16::MAX),
            });
        }
        positions[cursor..end].copy_from_slice(&by_register[register]);
        post_values[cursor..end].copy_from_slice(&posts_by_register[register]);
        cursor = end;
        offsets[register + 1] = u16::try_from(cursor)
            .map_err(|_| RegistersRwV3Error::SizeOverflow("BCSR rd offset sentinel"))?;
    }
    Ok(())
}

fn validate_position_plane(
    plane: &'static str,
    geometry: RegisterBcsrGeometry,
    offsets: &[[u16; REGISTER_BCSR_OFFSET_ENTRIES]],
    positions: &[[u8; REGISTER_BCSR_POSITION_SLOTS]],
) -> Result<(), RegistersRwV3Error> {
    for block in 0..geometry.blocks() {
        let block_offsets = &offsets[block];
        if block_offsets[0] != 0 {
            return Err(RegistersRwV3Error::OffsetStart {
                plane,
                got: u32::from(block_offsets[0]),
            });
        }
        for (register, pair) in block_offsets.windows(2).enumerate() {
            if pair[0] > pair[1] {
                return Err(RegistersRwV3Error::OffsetOrder {
                    plane,
                    header: block * REGISTER_CSR_COLUMNS + register,
                    start: u32::from(pair[0]),
                    end: u32::from(pair[1]),
                });
            }
        }
        let block_len = geometry.block_len(block)?;
        let terminal = block_offsets[REGISTER_CSR_COLUMNS];
        if usize::from(terminal) > block_len {
            return Err(RegistersRwV3Error::BcsrOffsetTerminal {
                plane,
                block,
                maximum: block_len,
                got: terminal,
            });
        }
        let mut seen = [false; REGISTER_BCSR_POSITION_SLOTS];
        for register in 0..REGISTER_CSR_COLUMNS {
            let range = offset_range(block_offsets, register);
            let column = &positions[block][range];
            if column.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(RegistersRwV3Error::PositionOrder {
                    plane,
                    header: block * REGISTER_CSR_COLUMNS + register,
                });
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
                        cycle: block * REGISTER_BCSR_POSITION_SLOTS + local,
                    });
                }
            }
        }
        for (slot, &padding) in positions[block][usize::from(terminal)..].iter().enumerate() {
            if padding != 0 {
                return Err(RegistersRwV3Error::BcsrNonzeroPadding {
                    plane,
                    block,
                    slot: usize::from(terminal) + slot,
                });
            }
        }
    }
    Ok(())
}

fn validate_post_padding(
    geometry: RegisterBcsrGeometry,
    rd_offsets: &[[u16; REGISTER_BCSR_OFFSET_ENTRIES]],
    rd_post_values: &[[u64; REGISTER_BCSR_POSITION_SLOTS]],
) -> Result<(), RegistersRwV3Error> {
    for block in 0..geometry.blocks() {
        let terminal = usize::from(rd_offsets[block][REGISTER_CSR_COLUMNS]);
        for (slot, &padding) in rd_post_values[block][terminal..].iter().enumerate() {
            if padding != 0 {
                return Err(RegistersRwV3Error::BcsrNonzeroPadding {
                    plane: "rd post values",
                    block,
                    slot: terminal + slot,
                });
            }
        }
    }
    Ok(())
}

fn validate_rd_index(
    parts: &RegisterBcsr256Parts,
    geometry: RegisterBcsrGeometry,
) -> Result<(), RegistersRwV3Error> {
    for block in 0..geometry.blocks() {
        let expected = event_register_map(&parts.rd_offsets[block], &parts.rd_positions[block]);
        let block_len = geometry.block_len(block)?;
        let cycle_base = block * REGISTER_BCSR_POSITION_SLOTS;
        for (local, &expected_register) in expected.iter().take(block_len).enumerate() {
            let cycle = cycle_base + local;
            let got = parts.rd_index[cycle];
            if got != REGISTER_ABSENT_INDEX && usize::from(got) >= REGISTER_CSR_COLUMNS {
                return Err(RegistersRwV3Error::InvalidRegister {
                    cycle,
                    access: "rd index",
                    register: got,
                });
            }
            if got != expected_register {
                return Err(RegistersRwV3Error::RdIndexMismatch {
                    cycle,
                    expected: expected_register,
                    got,
                });
            }
        }
    }
    Ok(())
}

fn validate_rd_increment(
    parts: &RegisterBcsr256Parts,
    geometry: RegisterBcsrGeometry,
) -> Result<(), RegistersRwV3Error> {
    let mut state = parts.start_values[0];
    for block in 0..geometry.blocks() {
        let rd = event_register_map(&parts.rd_offsets[block], &parts.rd_positions[block]);
        let mut posts = [0u64; REGISTER_BCSR_POSITION_SLOTS];
        for register in 0..REGISTER_CSR_COLUMNS {
            for index in offset_range(&parts.rd_offsets[block], register) {
                posts[usize::from(parts.rd_positions[block][index])] =
                    parts.rd_post_values[block][index];
            }
        }
        let block_len = geometry.block_len(block)?;
        let cycle_base = block * REGISTER_BCSR_POSITION_SLOTS;
        for (local, &rd_index) in rd.iter().take(block_len).enumerate() {
            let cycle = cycle_base + local;
            let expected = if rd_index == REGISTER_ABSENT_INDEX {
                Fp128::ZERO
            } else {
                let register = usize::from(rd_index);
                let value = fp128_increment(state[register], posts[local]);
                state[register] = posts[local];
                value
            };
            if !parts.rd_inc[cycle].is_canonical(AKITA_OFFSET_FFFFA7F7)
                || parts.rd_inc[cycle] != expected
            {
                return Err(RegistersRwV3Error::IncrementMismatch { cycle });
            }
        }
    }
    Ok(())
}

fn validate_block_state_flow(
    parts: &RegisterBcsr256Parts,
    geometry: RegisterBcsrGeometry,
) -> Result<(), RegistersRwV3Error> {
    let mut expected = parts.start_values[0];
    for block in 0..geometry.blocks() {
        for (register, expected_value) in expected.iter_mut().enumerate() {
            let got = parts.start_values[block][register];
            if got != *expected_value {
                return Err(RegistersRwV3Error::BlockStateMismatch {
                    block,
                    register,
                    expected: *expected_value,
                    got,
                });
            }
            let range = offset_range(&parts.rd_offsets[block], register);
            if let Some(&last) = parts.rd_post_values[block][range].last() {
                *expected_value = last;
            }
        }
    }
    Ok(())
}

fn event_register_map(
    offsets: &[u16; REGISTER_BCSR_OFFSET_ENTRIES],
    positions: &[u8; REGISTER_BCSR_POSITION_SLOTS],
) -> [u8; REGISTER_BCSR_POSITION_SLOTS] {
    let mut result = [REGISTER_ABSENT_INDEX; REGISTER_BCSR_POSITION_SLOTS];
    for register in 0..REGISTER_CSR_COLUMNS {
        for index in offset_range(offsets, register) {
            result[usize::from(positions[index])] = register as u8;
        }
    }
    result
}

fn offset_range(
    offsets: &[u16; REGISTER_BCSR_OFFSET_ENTRIES],
    register: usize,
) -> core::ops::Range<usize> {
    usize::from(offsets[register])..usize::from(offsets[register + 1])
}

fn terminal_sum(offsets: &[[u16; REGISTER_BCSR_OFFSET_ENTRIES]]) -> usize {
    offsets
        .iter()
        .map(|block| usize::from(block[REGISTER_CSR_COLUMNS]))
        .sum()
}

fn read_from_map(register: u8, state: &[u64; REGISTER_CSR_COLUMNS]) -> Option<RegisterRead> {
    (register != REGISTER_ABSENT_INDEX)
        .then(|| RegisterRead::new(register, state[usize::from(register)]))
}

fn fp128_increment(pre_value: u64, post_value: u64) -> Fp128 {
    let increment = AkitaField::from_i128(i128::from(post_value) - i128::from(pre_value));
    Fp128::from_jolt_field(&increment)
}

fn checked_register(
    cycle: usize,
    access: &'static str,
    register: u8,
) -> Result<usize, RegistersRwV3Error> {
    let register_index = usize::from(register);
    if register_index >= REGISTER_CSR_COLUMNS {
        Err(RegistersRwV3Error::InvalidRegister {
            cycle,
            access,
            register,
        })
    } else {
        Ok(register_index)
    }
}

fn check_read(
    cycle: usize,
    access: &'static str,
    read: RegisterRead,
    expected: u64,
) -> Result<(), RegistersRwV3Error> {
    if read.value() == expected {
        Ok(())
    } else {
        Err(RegistersRwV3Error::ReadValueMismatch {
            cycle,
            access,
            register: read.register(),
            expected,
            got: read.value(),
        })
    }
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

fn checked_product(
    name: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, RegistersRwV3Error> {
    left.checked_mul(right)
        .ok_or(RegistersRwV3Error::SizeOverflow(name))
}

fn checked_sum(values: &[usize]) -> Result<usize, RegistersRwV3Error> {
    values.iter().try_fold(0usize, |sum, value| {
        sum.checked_add(*value)
            .ok_or(RegistersRwV3Error::SizeOverflow("BCSR byte total"))
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterBcsrSourceProvenance {
    device_registry_id: NonZeroU64,
    source_allocation_identity: NonZeroUsize,
    source_allocation_bytes: NonZeroUsize,
    generation: NonZeroU64,
    cycles: usize,
    ordered_prefix_digest: OrderedPrefixDigest,
}

impl RegisterBcsrSourceProvenance {
    pub(crate) fn new(
        device_registry_id: u64,
        source_allocation_identity: usize,
        source_allocation_bytes: usize,
        generation: u64,
        cycles: usize,
        ordered_prefix_digest: OrderedPrefixDigest,
    ) -> Result<Self, RegistersRwV3Error> {
        let _geometry = RegisterBcsrGeometry::new(cycles)?;
        Ok(Self {
            device_registry_id: NonZeroU64::new(device_registry_id)
                .ok_or(RegistersRwV3Error::MissingIdentity("BCSR source device"))?,
            source_allocation_identity: NonZeroUsize::new(source_allocation_identity).ok_or(
                RegistersRwV3Error::MissingIdentity("BCSR source allocation"),
            )?,
            source_allocation_bytes: NonZeroUsize::new(source_allocation_bytes).ok_or(
                RegistersRwV3Error::MissingIdentity("BCSR source allocation bytes"),
            )?,
            generation: NonZeroU64::new(generation).ok_or(RegistersRwV3Error::MissingIdentity(
                "BCSR source generation",
            ))?,
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
        self.source_allocation_bytes.get()
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

/// Registry metadata for an initialized device buffer.
///
/// This type carries no Metal handle and never owns or retains an allocation.
/// Consumers resolve the identity against the device registry after checking
/// the generation and completion serial.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterBcsrPlaneProvenance {
    device_registry_id: NonZeroU64,
    allocation_identity: NonZeroUsize,
    generation: NonZeroU64,
    initialization_serial: NonZeroU64,
    elements: usize,
    bytes: usize,
}

impl RegisterBcsrPlaneProvenance {
    pub(crate) fn new(
        device_registry_id: u64,
        allocation_identity: usize,
        generation: u64,
        initialization_serial: u64,
        elements: usize,
        bytes: usize,
    ) -> Result<Self, RegistersRwV3Error> {
        Ok(Self {
            device_registry_id: NonZeroU64::new(device_registry_id)
                .ok_or(RegistersRwV3Error::MissingIdentity("BCSR plane device"))?,
            allocation_identity: NonZeroUsize::new(allocation_identity)
                .ok_or(RegistersRwV3Error::MissingIdentity("BCSR plane allocation"))?,
            generation: NonZeroU64::new(generation)
                .ok_or(RegistersRwV3Error::MissingIdentity("BCSR plane generation"))?,
            initialization_serial: NonZeroU64::new(initialization_serial).ok_or(
                RegistersRwV3Error::MissingIdentity("BCSR plane initialization serial"),
            )?,
            elements,
            bytes,
        })
    }

    pub(crate) const fn device_registry_id(self) -> u64 {
        self.device_registry_id.get()
    }

    pub(crate) const fn allocation_identity(self) -> usize {
        self.allocation_identity.get()
    }

    pub(crate) const fn generation(self) -> u64 {
        self.generation.get()
    }

    pub(crate) const fn initialization_serial(self) -> u64 {
        self.initialization_serial.get()
    }

    pub(crate) const fn elements(self) -> usize {
        self.elements
    }

    pub(crate) const fn bytes(self) -> usize {
        self.bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterBcsrPlaneProvenances {
    pub(crate) start_values: RegisterBcsrPlaneProvenance,
    pub(crate) rs1_offsets: RegisterBcsrPlaneProvenance,
    pub(crate) rs2_offsets: RegisterBcsrPlaneProvenance,
    pub(crate) rd_offsets: RegisterBcsrPlaneProvenance,
    pub(crate) rs1_positions: RegisterBcsrPlaneProvenance,
    pub(crate) rs2_positions: RegisterBcsrPlaneProvenance,
    pub(crate) rd_positions: RegisterBcsrPlaneProvenance,
    pub(crate) rd_post_values: RegisterBcsrPlaneProvenance,
    pub(crate) rd_index: RegisterBcsrPlaneProvenance,
    pub(crate) rd_inc: RegisterBcsrPlaneProvenance,
}

impl RegisterBcsrPlaneProvenances {
    #[expect(
        clippy::too_many_arguments,
        reason = "one provenance record per BCSR device plane"
    )]
    pub(crate) const fn new(
        start_values: RegisterBcsrPlaneProvenance,
        rs1_offsets: RegisterBcsrPlaneProvenance,
        rs2_offsets: RegisterBcsrPlaneProvenance,
        rd_offsets: RegisterBcsrPlaneProvenance,
        rs1_positions: RegisterBcsrPlaneProvenance,
        rs2_positions: RegisterBcsrPlaneProvenance,
        rd_positions: RegisterBcsrPlaneProvenance,
        rd_post_values: RegisterBcsrPlaneProvenance,
        rd_index: RegisterBcsrPlaneProvenance,
        rd_inc: RegisterBcsrPlaneProvenance,
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
            rd_index,
            rd_inc,
        }
    }

    fn identities(self) -> [usize; REGISTER_BCSR_PLANE_COUNT] {
        [
            self.start_values.allocation_identity(),
            self.rs1_offsets.allocation_identity(),
            self.rs2_offsets.allocation_identity(),
            self.rd_offsets.allocation_identity(),
            self.rs1_positions.allocation_identity(),
            self.rs2_positions.allocation_identity(),
            self.rd_positions.allocation_identity(),
            self.rd_post_values.allocation_identity(),
            self.rd_index.allocation_identity(),
            self.rd_inc.allocation_identity(),
        ]
    }
}

/// Checked metadata for the BCSR planes produced by one source generation.
///
/// Copying this receipt copies only registry evidence. Device allocation
/// ownership remains with the runtime registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegisterBcsrReceipt {
    source: RegisterBcsrSourceProvenance,
    layout: RegisterBcsrLayout,
    planes: RegisterBcsrPlaneProvenances,
}

impl RegisterBcsrReceipt {
    pub(crate) fn admit(
        source: RegisterBcsrSourceProvenance,
        layout: RegisterBcsrLayout,
        planes: &RegisterBcsrPlaneProvenances,
    ) -> Result<Self, RegistersRwV3Error> {
        if source.cycles() != layout.geometry().cycles() {
            return Err(RegistersRwV3Error::ProducerCycleMismatch {
                expected: layout.geometry().cycles(),
                got: source.cycles(),
            });
        }

        let descriptors = [
            (
                "BCSR start values",
                planes.start_values,
                layout.start_values(),
            ),
            ("BCSR rs1 offsets", planes.rs1_offsets, layout.offsets()),
            ("BCSR rs2 offsets", planes.rs2_offsets, layout.offsets()),
            ("BCSR rd offsets", planes.rd_offsets, layout.offsets()),
            (
                "BCSR rs1 positions",
                planes.rs1_positions,
                layout.positions(),
            ),
            (
                "BCSR rs2 positions",
                planes.rs2_positions,
                layout.positions(),
            ),
            ("BCSR rd positions", planes.rd_positions, layout.positions()),
            (
                "BCSR rd post values",
                planes.rd_post_values,
                layout.rd_post_values(),
            ),
            ("BCSR rd index", planes.rd_index, layout.rd_index()),
            ("BCSR rd increment", planes.rd_inc, layout.rd_inc()),
        ];
        for (plane, provenance, expected) in descriptors {
            if provenance.device_registry_id() != source.device_registry_id() {
                return Err(RegistersRwV3Error::PlaneDeviceMismatch {
                    plane,
                    expected: source.device_registry_id(),
                    got: provenance.device_registry_id(),
                });
            }
            if provenance.generation() != source.generation() {
                return Err(RegistersRwV3Error::PlaneGenerationMismatch {
                    plane,
                    expected: source.generation(),
                    got: provenance.generation(),
                });
            }
            if provenance.elements() != expected.elements()
                || provenance.bytes() != expected.bytes()
            {
                return Err(RegistersRwV3Error::PlaneShape {
                    plane,
                    expected_elements: expected.elements(),
                    got_elements: provenance.elements(),
                    expected_bytes: expected.bytes(),
                    got_bytes: provenance.bytes(),
                });
            }
        }

        let identities = planes.identities();
        for (index, identity) in identities.iter().copied().enumerate() {
            if identity == source.source_allocation_identity()
                || identities[..index].contains(&identity)
            {
                return Err(RegistersRwV3Error::DuplicateAllocationIdentity { identity });
            }
        }
        Ok(Self {
            source,
            layout,
            planes: *planes,
        })
    }

    pub(crate) const fn source(self) -> RegisterBcsrSourceProvenance {
        self.source
    }

    pub(crate) const fn layout(self) -> RegisterBcsrLayout {
        self.layout
    }

    pub(crate) fn allocation_identities(self) -> [usize; REGISTER_BCSR_PLANE_COUNT] {
        self.planes.identities()
    }

    pub(crate) fn verify_binding(
        self,
        device_registry_id: u64,
        generation: u64,
        ordered_prefix_digest: OrderedPrefixDigest,
    ) -> Result<(), RegistersRwV3Error> {
        if device_registry_id != self.source.device_registry_id() {
            return Err(RegistersRwV3Error::ReceiptDeviceMismatch {
                expected: self.source.device_registry_id(),
                got: device_registry_id,
            });
        }
        if generation != self.source.generation() {
            return Err(RegistersRwV3Error::ReceiptGenerationMismatch {
                expected: self.source.generation(),
                got: generation,
            });
        }
        if ordered_prefix_digest != self.source.ordered_prefix_digest() {
            return Err(RegistersRwV3Error::ReceiptDigestMismatch);
        }
        Ok(())
    }

    pub(crate) fn registers_val_input(
        self,
    ) -> Result<RegistersValInputReceipt, RegistersRwV3Error> {
        let cycles = self.layout.geometry().cycles();
        if cycles < 4 || !cycles.is_power_of_two() {
            return Err(RegistersRwV3Error::InvalidRegistersValHandoff(cycles));
        }
        Ok(RegistersValInputReceipt {
            source: self.source,
            rd_index: self.planes.rd_index,
            rd_inc: self.planes.rd_inc,
        })
    }
}

/// Zero-copy stage-4 handoff for the two planes consumed by RegistersVal.
///
/// The receipt contains allocation identities, not buffers. Resolving both
/// identities on the recorded device and generation is required before use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegistersValInputReceipt {
    source: RegisterBcsrSourceProvenance,
    rd_index: RegisterBcsrPlaneProvenance,
    rd_inc: RegisterBcsrPlaneProvenance,
}

impl RegistersValInputReceipt {
    pub(crate) const fn cycles(self) -> usize {
        self.source.cycles()
    }

    pub(crate) const fn device_registry_id(self) -> u64 {
        self.source.device_registry_id()
    }

    pub(crate) const fn generation(self) -> u64 {
        self.source.generation()
    }

    pub(crate) const fn ordered_prefix_digest(self) -> OrderedPrefixDigest {
        self.source.ordered_prefix_digest()
    }

    pub(crate) const fn rd_index(self) -> RegisterBcsrPlaneProvenance {
        self.rd_index
    }

    pub(crate) const fn rd_inc(self) -> RegisterBcsrPlaneProvenance {
        self.rd_inc
    }

    pub(crate) fn resident_abi(self) -> Result<RegistersValResidentInputAbi, RegistersRwV3Error> {
        RegistersValResidentInputAbi::new(
            self.cycles(),
            self.device_registry_id(),
            self.rd_inc.allocation_identity(),
            self.rd_index.allocation_identity(),
            self.generation(),
        )
        .map_err(|_| RegistersRwV3Error::InvalidRegistersValHandoff(self.cycles()))
    }
}
