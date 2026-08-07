use core::mem::size_of;

use thiserror::Error;

pub const REGISTER_CSR_BLOCK_CYCLES: usize = 256;
pub const REGISTER_CSR_COLUMNS: usize = 128;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegisterEventCounts {
    pub rs1: u64,
    pub rs2: u64,
    pub rd: u64,
}

/// Logical CSR storage model. A census describes counts, not measured workload
/// evidence; callers must not use the analytical constant for runtime admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisterCsrCensus {
    cycles: u64,
    events: RegisterEventCounts,
}

impl RegisterCsrCensus {
    pub fn new(cycles: u64, events: RegisterEventCounts) -> Result<Self, RegisterOwnerError> {
        Self { cycles, events }.validate()
    }

    pub fn validate(self) -> Result<Self, RegisterOwnerError> {
        if self.cycles > u64::from(u32::MAX) {
            return Err(RegisterOwnerError::CycleCountTooLarge(self.cycles));
        }
        for (plane, count) in [
            ("rs1", self.events.rs1),
            ("rs2", self.events.rs2),
            ("rd", self.events.rd),
        ] {
            if count > self.cycles {
                return Err(RegisterOwnerError::InvalidCensusEventCount {
                    plane,
                    cycles: self.cycles,
                    count,
                });
            }
        }
        Ok(self)
    }

    pub const fn cycles(self) -> u64 {
        self.cycles
    }

    pub const fn events(self) -> RegisterEventCounts {
        self.events
    }

    pub fn block_count(self) -> Result<u64, RegisterOwnerError> {
        let _census = self.validate()?;
        Ok(self.cycles.div_ceil(REGISTER_CSR_BLOCK_CYCLES as u64))
    }

    pub fn block_columns(self) -> Result<u64, RegisterOwnerError> {
        self.block_count()?
            .checked_mul(REGISTER_CSR_COLUMNS as u64)
            .ok_or(RegisterOwnerError::SizeOverflow)
    }

    pub fn storage_bytes(self) -> Result<u128, RegisterOwnerError> {
        let _census = self.validate()?;
        let block_columns = u128::from(self.block_columns()?);
        let offset_entries = block_columns
            .checked_add(1)
            .ok_or(RegisterOwnerError::SizeOverflow)?;
        let position_entries = u128::from(self.events.rs1)
            .checked_add(u128::from(self.events.rs2))
            .and_then(|value| value.checked_add(u128::from(self.events.rd)))
            .ok_or(RegisterOwnerError::SizeOverflow)?;

        checked_sum(&[
            checked_mul(size_of::<u64>() as u128, block_columns)?,
            checked_mul(3 * size_of::<u32>() as u128, offset_entries)?,
            checked_mul(size_of::<u8>() as u128, position_entries)?,
            checked_mul(size_of::<u64>() as u128, u128::from(self.events.rd))?,
        ])
    }
}

/// Analytical fixture from the pre-implementation log-26 topology census.
/// Production code must replace it with counts from the admitted witness.
pub const REGISTER_CSR_NON_AUTHORITATIVE_LOG_T_26_CENSUS: RegisterCsrCensus = RegisterCsrCensus {
    cycles: 1 << 26,
    events: RegisterEventCounts {
        rs1: 59_652_323,
        rs2: 55_924_053,
        rd: 50_331_648,
    },
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegisterOwnerRead {
    pub register: u8,
    pub value: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegisterOwnerWrite {
    pub register: u8,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegisterOwnerRow {
    pub rs1: Option<RegisterOwnerRead>,
    pub rs2: Option<RegisterOwnerRead>,
    pub rd: Option<RegisterOwnerWrite>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegisterCsr256Parts {
    pub cycles: usize,
    pub start_values: Vec<u64>,
    pub rs1_offsets: Vec<u32>,
    pub rs2_offsets: Vec<u32>,
    pub rd_offsets: Vec<u32>,
    pub rs1_positions: Vec<u8>,
    pub rs2_positions: Vec<u8>,
    pub rd_positions: Vec<u8>,
    pub rd_post_values: Vec<u64>,
}

/// Structurally checked CSR-256 register event storage.
///
/// Construction checks plane lengths, offsets, event order, one event per
/// plane and cycle, last-block bounds, and carried state between blocks. Read
/// values and rd pre-values are checked by [`CertifiedRegisterOwner::build`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegisterCsr256 {
    parts: RegisterCsr256Parts,
    block_count: usize,
}

impl RegisterCsr256 {
    pub fn from_parts(parts: RegisterCsr256Parts) -> Result<Self, RegisterOwnerError> {
        let block_count = checked_block_count(parts.cycles)?;
        let owner = Self { parts, block_count };
        owner.validate()?;
        Ok(owner)
    }

    pub fn validate(&self) -> Result<(), RegisterOwnerError> {
        let block_columns = self
            .block_count
            .checked_mul(REGISTER_CSR_COLUMNS)
            .ok_or(RegisterOwnerError::SizeOverflow)?;
        if self.parts.start_values.len() != block_columns {
            return Err(RegisterOwnerError::PlaneLength {
                plane: "start values",
                expected: block_columns,
                got: self.parts.start_values.len(),
            });
        }
        validate_plane(
            "rs1",
            self.parts.cycles,
            self.block_count,
            &self.parts.rs1_offsets,
            &self.parts.rs1_positions,
        )?;
        validate_plane(
            "rs2",
            self.parts.cycles,
            self.block_count,
            &self.parts.rs2_offsets,
            &self.parts.rs2_positions,
        )?;
        validate_plane(
            "rd",
            self.parts.cycles,
            self.block_count,
            &self.parts.rd_offsets,
            &self.parts.rd_positions,
        )?;
        if self.parts.rd_post_values.len() != self.parts.rd_positions.len() {
            return Err(RegisterOwnerError::PlaneLength {
                plane: "rd post values",
                expected: self.parts.rd_positions.len(),
                got: self.parts.rd_post_values.len(),
            });
        }
        validate_block_state_flow(&self.parts, self.block_count)
    }

    pub const fn cycles(&self) -> usize {
        self.parts.cycles
    }

    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    pub fn event_counts(&self) -> RegisterEventCounts {
        RegisterEventCounts {
            rs1: self.parts.rs1_positions.len() as u64,
            rs2: self.parts.rs2_positions.len() as u64,
            rd: self.parts.rd_positions.len() as u64,
        }
    }

    pub fn storage_bytes(&self) -> u128 {
        checked_slice_bytes::<u64>(&self.parts.start_values)
            + checked_slice_bytes::<u32>(&self.parts.rs1_offsets)
            + checked_slice_bytes::<u32>(&self.parts.rs2_offsets)
            + checked_slice_bytes::<u32>(&self.parts.rd_offsets)
            + checked_slice_bytes::<u8>(&self.parts.rs1_positions)
            + checked_slice_bytes::<u8>(&self.parts.rs2_positions)
            + checked_slice_bytes::<u8>(&self.parts.rd_positions)
            + checked_slice_bytes::<u64>(&self.parts.rd_post_values)
    }

    pub const fn parts(&self) -> &RegisterCsr256Parts {
        &self.parts
    }

    pub fn into_parts(self) -> RegisterCsr256Parts {
        self.parts
    }

    pub fn derive_rd_increment_activity(&self, cap: usize) -> RdIncrementActivity {
        let mut entries = Some(Vec::with_capacity(cap.min(self.parts.rd_positions.len())));
        let mut nonzero_count = 0usize;
        for block in 0..self.block_count {
            for register in 0..REGISTER_CSR_COLUMNS {
                let header = block * REGISTER_CSR_COLUMNS + register;
                let range = offset_range(&self.parts.rd_offsets, header);
                let mut previous = self.parts.start_values[header];
                for event in range {
                    let post = self.parts.rd_post_values[event];
                    let increment = i128::from(post) - i128::from(previous);
                    if increment != 0 {
                        nonzero_count += 1;
                        if nonzero_count <= cap {
                            let cycle = (block * REGISTER_CSR_BLOCK_CYCLES
                                + usize::from(self.parts.rd_positions[event]))
                                as u32;
                            if let Some(entries) = entries.as_mut() {
                                entries.push(RdIncrement { cycle, increment });
                            }
                        } else {
                            entries = None;
                        }
                    }
                    previous = post;
                }
            }
        }

        match entries {
            Some(mut entries) => {
                entries.sort_unstable_by_key(|entry| entry.cycle);
                RdIncrementActivity::Complete { cap, entries }
            }
            None => RdIncrementActivity::Overflow { cap, nonzero_count },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RdIncrement {
    pub cycle: u32,
    pub increment: i128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RdIncrementActivity {
    Complete {
        cap: usize,
        entries: Vec<RdIncrement>,
    },
    Overflow {
        cap: usize,
        nonzero_count: usize,
    },
}

impl RdIncrementActivity {
    pub fn cap(&self) -> usize {
        match self {
            Self::Complete { cap, .. } | Self::Overflow { cap, .. } => *cap,
        }
    }

    pub fn entries(&self) -> Option<&[RdIncrement]> {
        match self {
            Self::Complete { entries, .. } => Some(entries),
            Self::Overflow { .. } => None,
        }
    }

    pub fn nonzero_count(&self) -> usize {
        match self {
            Self::Complete { entries, .. } => entries.len(),
            Self::Overflow { nonzero_count, .. } => *nonzero_count,
        }
    }

    pub const fn overflowed(&self) -> bool {
        matches!(self, Self::Overflow { .. })
    }
}

/// Evidence that raw reads and write pre-values matched a single carried
/// register state while the CSR was produced.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegisterStateFlowCertificate {
    cycles: usize,
    events: RegisterEventCounts,
    nonzero_rd_increments: usize,
    initial_values: [u64; REGISTER_CSR_COLUMNS],
    final_values: [u64; REGISTER_CSR_COLUMNS],
}

impl RegisterStateFlowCertificate {
    pub const fn cycles(&self) -> usize {
        self.cycles
    }

    pub const fn events(&self) -> RegisterEventCounts {
        self.events
    }

    pub const fn nonzero_rd_increments(&self) -> usize {
        self.nonzero_rd_increments
    }

    pub const fn initial_values(&self) -> &[u64; REGISTER_CSR_COLUMNS] {
        &self.initial_values
    }

    pub const fn final_values(&self) -> &[u64; REGISTER_CSR_COLUMNS] {
        &self.final_values
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedRegisterOwner {
    csr: RegisterCsr256,
    state_flow: RegisterStateFlowCertificate,
    rd_increment_activity: RdIncrementActivity,
}

impl CertifiedRegisterOwner {
    /// Builds the CSR and state-flow certificate in one row pass.
    ///
    /// The caller still has to bind `rows` and `initial_values` to the proof
    /// session's witness generation; this backend-neutral type has no runtime
    /// allocation or device identity.
    pub fn build(
        rows: &[RegisterOwnerRow],
        initial_values: &[u64; REGISTER_CSR_COLUMNS],
        rd_increment_activity_cap: usize,
    ) -> Result<Self, RegisterOwnerError> {
        let block_count = checked_block_count(rows.len())?;
        let block_columns = block_count
            .checked_mul(REGISTER_CSR_COLUMNS)
            .ok_or(RegisterOwnerError::SizeOverflow)?;
        let offset_capacity = block_columns
            .checked_add(1)
            .ok_or(RegisterOwnerError::SizeOverflow)?;

        let mut start_values = Vec::with_capacity(block_columns);
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
        let mut rs1_by_register: [Vec<u8>; REGISTER_CSR_COLUMNS] =
            core::array::from_fn(|_| Vec::new());
        let mut rs2_by_register: [Vec<u8>; REGISTER_CSR_COLUMNS] =
            core::array::from_fn(|_| Vec::new());
        let mut rd_by_register: [Vec<u8>; REGISTER_CSR_COLUMNS] =
            core::array::from_fn(|_| Vec::new());
        let mut rd_posts_by_register: [Vec<u64>; REGISTER_CSR_COLUMNS] =
            core::array::from_fn(|_| Vec::new());
        let mut state = *initial_values;

        for (block, block_rows) in rows.chunks(REGISTER_CSR_BLOCK_CYCLES).enumerate() {
            start_values.extend_from_slice(&state);
            for column in 0..REGISTER_CSR_COLUMNS {
                rs1_by_register[column].clear();
                rs2_by_register[column].clear();
                rd_by_register[column].clear();
                rd_posts_by_register[column].clear();
            }

            for (position, row) in block_rows.iter().enumerate() {
                let cycle = block * REGISTER_CSR_BLOCK_CYCLES + position;
                let position =
                    u8::try_from(position).map_err(|_| RegisterOwnerError::SizeOverflow)?;
                if let Some(read) = row.rs1 {
                    let register = checked_register(cycle, "rs1", read.register)?;
                    check_read(cycle, "rs1", read, state[register])?;
                    rs1_by_register[register].push(position);
                }
                if let Some(read) = row.rs2 {
                    let register = checked_register(cycle, "rs2", read.register)?;
                    check_read(cycle, "rs2", read, state[register])?;
                    rs2_by_register[register].push(position);
                }
                if let Some(write) = row.rd {
                    let register = checked_register(cycle, "rd", write.register)?;
                    if state[register] != write.pre_value {
                        return Err(RegisterOwnerError::WritePreValueMismatch {
                            cycle,
                            register: write.register,
                            expected: state[register],
                            got: write.pre_value,
                        });
                    }
                    rd_by_register[register].push(position);
                    rd_posts_by_register[register].push(write.post_value);
                    state[register] = write.post_value;
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
            cycles: rows.len(),
            start_values,
            rs1_offsets,
            rs2_offsets,
            rd_offsets,
            rs1_positions,
            rs2_positions,
            rd_positions,
            rd_post_values,
        })?;
        let rd_increment_activity = csr.derive_rd_increment_activity(rd_increment_activity_cap);
        let state_flow = RegisterStateFlowCertificate {
            cycles: rows.len(),
            events: csr.event_counts(),
            nonzero_rd_increments: rd_increment_activity.nonzero_count(),
            initial_values: *initial_values,
            final_values: state,
        };

        Ok(Self {
            csr,
            state_flow,
            rd_increment_activity,
        })
    }

    pub const fn csr(&self) -> &RegisterCsr256 {
        &self.csr
    }

    pub const fn state_flow(&self) -> &RegisterStateFlowCertificate {
        &self.state_flow
    }

    pub const fn rd_increment_activity(&self) -> &RdIncrementActivity {
        &self.rd_increment_activity
    }
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RegisterOwnerError {
    #[error("register owner cycle count {0} exceeds the u32 event index space")]
    CycleCountTooLarge(u64),
    #[error("register owner size arithmetic overflowed")]
    SizeOverflow,
    #[error("register owner {plane} census has {count} events for {cycles} cycles")]
    InvalidCensusEventCount {
        plane: &'static str,
        cycles: u64,
        count: u64,
    },
    #[error("register owner {plane} length is {got}, expected {expected}")]
    PlaneLength {
        plane: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("register owner {plane} offsets must start at zero, got {got}")]
    OffsetStart { plane: &'static str, got: u32 },
    #[error("register owner {plane} offsets decrease at header {header}: {start} to {end}")]
    OffsetOrder {
        plane: &'static str,
        header: usize,
        start: u32,
        end: u32,
    },
    #[error("register owner {plane} terminal offset is {got}, expected {expected}")]
    OffsetTerminal {
        plane: &'static str,
        expected: usize,
        got: u32,
    },
    #[error("register owner {plane} positions are not increasing at header {header}")]
    PositionOrder { plane: &'static str, header: usize },
    #[error("register owner {plane} position {position} exceeds block {block} length {block_len}")]
    PositionOutOfBlock {
        plane: &'static str,
        block: usize,
        block_len: usize,
        position: u8,
    },
    #[error("register owner {plane} has more than one event at cycle {cycle}")]
    DuplicateCycleEvent { plane: &'static str, cycle: usize },
    #[error(
        "register owner block {block} register {register} starts at {got}, expected {expected}"
    )]
    BlockStateMismatch {
        block: usize,
        register: usize,
        expected: u64,
        got: u64,
    },
    #[error("register owner {access} index {register} at cycle {cycle} is out of range")]
    InvalidRegister {
        cycle: usize,
        access: &'static str,
        register: u8,
    },
    #[error(
        "register owner {access} read at cycle {cycle}, register {register}, is {got}, expected {expected}"
    )]
    ReadValueMismatch {
        cycle: usize,
        access: &'static str,
        register: u8,
        expected: u64,
        got: u64,
    },
    #[error(
        "register owner rd pre-value at cycle {cycle}, register {register}, is {got}, expected {expected}"
    )]
    WritePreValueMismatch {
        cycle: usize,
        register: u8,
        expected: u64,
        got: u64,
    },
    #[error("register owner {plane} event count exceeds u32")]
    EventCountOverflow { plane: &'static str },
}

fn checked_block_count(cycles: usize) -> Result<usize, RegisterOwnerError> {
    if cycles > u32::MAX as usize {
        return Err(RegisterOwnerError::CycleCountTooLarge(cycles as u64));
    }
    Ok(cycles.div_ceil(REGISTER_CSR_BLOCK_CYCLES))
}

fn validate_plane(
    plane: &'static str,
    cycles: usize,
    block_count: usize,
    offsets: &[u32],
    positions: &[u8],
) -> Result<(), RegisterOwnerError> {
    let block_columns = block_count
        .checked_mul(REGISTER_CSR_COLUMNS)
        .ok_or(RegisterOwnerError::SizeOverflow)?;
    let expected_offsets = block_columns
        .checked_add(1)
        .ok_or(RegisterOwnerError::SizeOverflow)?;
    if offsets.len() != expected_offsets {
        return Err(RegisterOwnerError::PlaneLength {
            plane,
            expected: expected_offsets,
            got: offsets.len(),
        });
    }
    let Some(&first) = offsets.first() else {
        return Err(RegisterOwnerError::PlaneLength {
            plane,
            expected: expected_offsets,
            got: 0,
        });
    };
    if first != 0 {
        return Err(RegisterOwnerError::OffsetStart { plane, got: first });
    }
    for (header, pair) in offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(RegisterOwnerError::OffsetOrder {
                plane,
                header,
                start: pair[0],
                end: pair[1],
            });
        }
    }
    let terminal = offsets.last().copied().unwrap_or_default();
    if terminal as usize != positions.len() {
        return Err(RegisterOwnerError::OffsetTerminal {
            plane,
            expected: positions.len(),
            got: terminal,
        });
    }

    for block in 0..block_count {
        let block_start = block * REGISTER_CSR_BLOCK_CYCLES;
        let block_len = (cycles - block_start).min(REGISTER_CSR_BLOCK_CYCLES);
        let mut seen = [false; REGISTER_CSR_BLOCK_CYCLES];
        for register in 0..REGISTER_CSR_COLUMNS {
            let header = block * REGISTER_CSR_COLUMNS + register;
            let range = offset_range(offsets, header);
            let column = &positions[range];
            if column.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(RegisterOwnerError::PositionOrder { plane, header });
            }
            for &position in column {
                let local = usize::from(position);
                if local >= block_len {
                    return Err(RegisterOwnerError::PositionOutOfBlock {
                        plane,
                        block,
                        block_len,
                        position,
                    });
                }
                if core::mem::replace(&mut seen[local], true) {
                    return Err(RegisterOwnerError::DuplicateCycleEvent {
                        plane,
                        cycle: block_start + local,
                    });
                }
            }
        }
    }
    Ok(())
}

fn validate_block_state_flow(
    parts: &RegisterCsr256Parts,
    block_count: usize,
) -> Result<(), RegisterOwnerError> {
    if block_count < 2 {
        return Ok(());
    }
    for register in 0..REGISTER_CSR_COLUMNS {
        let mut value = parts.start_values[register];
        for block in 0..block_count {
            let header = block * REGISTER_CSR_COLUMNS + register;
            let start = parts.start_values[header];
            if start != value {
                return Err(RegisterOwnerError::BlockStateMismatch {
                    block,
                    register,
                    expected: value,
                    got: start,
                });
            }
            for post in &parts.rd_post_values[offset_range(&parts.rd_offsets, header)] {
                value = *post;
            }
        }
    }
    Ok(())
}

fn checked_register(
    cycle: usize,
    access: &'static str,
    register: u8,
) -> Result<usize, RegisterOwnerError> {
    let register_index = usize::from(register);
    if register_index >= REGISTER_CSR_COLUMNS {
        return Err(RegisterOwnerError::InvalidRegister {
            cycle,
            access,
            register,
        });
    }
    Ok(register_index)
}

fn check_read(
    cycle: usize,
    access: &'static str,
    read: RegisterOwnerRead,
    expected: u64,
) -> Result<(), RegisterOwnerError> {
    if read.value != expected {
        return Err(RegisterOwnerError::ReadValueMismatch {
            cycle,
            access,
            register: read.register,
            expected,
            got: read.value,
        });
    }
    Ok(())
}

fn event_offset(plane: &'static str, events: usize) -> Result<u32, RegisterOwnerError> {
    u32::try_from(events).map_err(|_| RegisterOwnerError::EventCountOverflow { plane })
}

fn offset_range(offsets: &[u32], header: usize) -> core::ops::Range<usize> {
    offsets[header] as usize..offsets[header + 1] as usize
}

fn checked_slice_bytes<T>(slice: &[T]) -> u128 {
    slice.len() as u128 * size_of::<T>() as u128
}

fn checked_mul(left: u128, right: u128) -> Result<u128, RegisterOwnerError> {
    left.checked_mul(right)
        .ok_or(RegisterOwnerError::SizeOverflow)
}

fn checked_sum(values: &[u128]) -> Result<u128, RegisterOwnerError> {
    values.iter().try_fold(0u128, |sum, value| {
        sum.checked_add(*value)
            .ok_or(RegisterOwnerError::SizeOverflow)
    })
}
