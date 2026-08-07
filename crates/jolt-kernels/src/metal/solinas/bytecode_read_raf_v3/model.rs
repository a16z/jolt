use thiserror::Error;

use super::owner::{
    BytecodeReadRafReceipt, CELL_BYTES, INNER_SIGN_BYTES, MAGNITUDE_BYTES, SHARED_ROW_BYTES,
};
use super::relation::{FUSED_STAGES, RA_FACTORS, STAGES};

const FIELD_BYTES: u128 = 16;
const NANOS_PER_SECOND: u128 = 1_000_000_000;
const Q10_PRODUCTS_PER_PAIR: u128 = 10;
const DENSE_PLANES: u128 = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FamilyShape {
    rows: u128,
    addresses: u128,
    inner_length: u128,
    outer_length: u128,
    nonempty_cells: u128,
    log_t: usize,
}

impl FamilyShape {
    pub fn from_receipt(receipt: BytecodeReadRafReceipt) -> Result<Self, AccountingError> {
        let shape = Self {
            rows: receipt.cycles() as u128,
            addresses: receipt.addresses() as u128,
            inner_length: receipt.inner_length() as u128,
            outer_length: receipt.outer_length() as u128,
            nonempty_cells: receipt.nonempty_cells() as u128,
            log_t: receipt.log_t(),
        };
        shape.validate()?;
        Ok(shape)
    }

    pub fn new(
        log_t: usize,
        addresses: u128,
        inner_length: u128,
        nonempty_cells: u128,
    ) -> Result<Self, AccountingError> {
        let rows = domain_size(log_t)?;
        if inner_length == 0 || rows % inner_length != 0 {
            return Err(AccountingError::InvalidShape);
        }
        let shape = Self {
            rows,
            addresses,
            inner_length,
            outer_length: rows / inner_length,
            nonempty_cells,
            log_t,
        };
        shape.validate()?;
        Ok(shape)
    }

    pub const fn rows(self) -> u128 {
        self.rows
    }

    pub const fn addresses(self) -> u128 {
        self.addresses
    }

    pub const fn inner_length(self) -> u128 {
        self.inner_length
    }

    pub const fn outer_length(self) -> u128 {
        self.outer_length
    }

    pub const fn nonempty_cells(self) -> u128 {
        self.nonempty_cells
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    fn validate(self) -> Result<(), AccountingError> {
        let cells = checked_mul(self.addresses, self.outer_length)?;
        if self.rows < 4
            || !self.rows.is_power_of_two()
            || self.addresses == 0
            || !self.addresses.is_power_of_two()
            || self.inner_length == 0
            || !self.inner_length.is_power_of_two()
            || self.rows != checked_mul(self.inner_length, self.outer_length)?
            || self.nonempty_cells < self.outer_length
            || self.nonempty_cells > cells
        {
            return Err(AccountingError::InvalidShape);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressAccounting {
    useful_signed_products: u128,
    useful_outer_products: u128,
    equality_generation_products: u128,
    stage_updates: u128,
    compact_occurrence_bytes: u128,
    producer_incremental_write_bytes: u128,
    requested_bytes: u128,
    unavoidable_bytes: u128,
}

impl AddressAccounting {
    pub const fn useful_signed_products(self) -> u128 {
        self.useful_signed_products
    }

    pub const fn useful_outer_products(self) -> u128 {
        self.useful_outer_products
    }

    pub const fn equality_generation_products(self) -> u128 {
        self.equality_generation_products
    }

    pub fn useful_products(self) -> Result<u128, AccountingError> {
        checked_sum([
            self.useful_signed_products,
            self.useful_outer_products,
            self.equality_generation_products,
        ])
    }

    pub const fn stage_updates(self) -> u128 {
        self.stage_updates
    }

    pub const fn compact_occurrence_bytes(self) -> u128 {
        self.compact_occurrence_bytes
    }

    pub const fn producer_incremental_write_bytes(self) -> u128 {
        self.producer_incremental_write_bytes
    }

    pub const fn requested_bytes(self) -> u128 {
        self.requested_bytes
    }

    pub const fn unavoidable_bytes(self) -> u128 {
        self.unavoidable_bytes
    }

    /// Exactly `9 / 12 = 0.75` for the one-pass compact occurrence stream.
    pub fn updates_per_compact_byte(self) -> (u128, u128) {
        (self.stage_updates, self.compact_occurrence_bytes)
    }

    pub fn roof(self, rates: RoofRates) -> Result<Roof, AccountingError> {
        rates.roof(self.useful_products()?, self.unavoidable_bytes)
    }
}

pub fn address_accounting(shape: FamilyShape) -> Result<AddressAccounting, AccountingError> {
    shape.validate()?;
    let cells = checked_mul(shape.addresses, shape.outer_length)?;
    let compact_bytes = checked_mul(INNER_SIGN_BYTES + MAGNITUDE_BYTES, shape.rows)?;
    let cell_bytes = checked_mul(CELL_BYTES, cells)?;
    let equality_lo_bytes = checked_mul(
        checked_mul(STAGES as u128, FIELD_BYTES)?,
        shape.inner_length,
    )?;
    let equality_hi_bytes = checked_mul(
        checked_mul(STAGES as u128, FIELD_BYTES)?,
        shape.outer_length,
    )?;
    let output_bytes = checked_mul(checked_mul(STAGES as u128, FIELD_BYTES)?, shape.addresses)?;
    let logical_lo_requests = checked_mul(checked_mul(STAGES as u128, FIELD_BYTES)?, shape.rows)?;
    let logical_hi_requests = checked_mul(
        checked_mul(STAGES as u128, FIELD_BYTES)?,
        shape.nonempty_cells,
    )?;
    let equality_nodes_per_stage = checked_add(
        checked_mul(
            2,
            shape
                .inner_length
                .checked_sub(1)
                .ok_or(AccountingError::Overflow)?,
        )?,
        checked_mul(
            2,
            shape
                .outer_length
                .checked_sub(1)
                .ok_or(AccountingError::Overflow)?,
        )?,
    )?;
    Ok(AddressAccounting {
        useful_signed_products: checked_mul(FUSED_STAGES as u128, shape.rows)?,
        useful_outer_products: checked_mul(STAGES as u128, shape.nonempty_cells)?,
        equality_generation_products: checked_mul(STAGES as u128, equality_nodes_per_stage)?,
        stage_updates: checked_mul(STAGES as u128, shape.rows)?,
        compact_occurrence_bytes: compact_bytes,
        producer_incremental_write_bytes: checked_add(compact_bytes, cell_bytes)?,
        requested_bytes: checked_sum([
            cell_bytes,
            compact_bytes,
            logical_lo_requests,
            logical_hi_requests,
            output_bytes,
        ])?,
        unavoidable_bytes: checked_sum([
            cell_bytes,
            compact_bytes,
            equality_lo_bytes,
            equality_hi_bytes,
            output_bytes,
        ])?,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CycleRoundKind {
    RowMessage,
    RowBindMessage,
    DenseBindMessage,
    TerminalBind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CycleRoundAccounting {
    kind: CycleRoundKind,
    source_elements: u128,
    useful_products: u128,
    requested_bytes: u128,
    unavoidable_bytes: u128,
    work_items: u128,
}

impl CycleRoundAccounting {
    pub const fn kind(self) -> CycleRoundKind {
        self.kind
    }

    pub const fn source_elements(self) -> u128 {
        self.source_elements
    }

    pub const fn useful_products(self) -> u128 {
        self.useful_products
    }

    pub const fn requested_bytes(self) -> u128 {
        self.requested_bytes
    }

    pub const fn unavoidable_bytes(self) -> u128 {
        self.unavoidable_bytes
    }

    pub const fn work_items(self) -> u128 {
        self.work_items
    }

    pub fn roof(self, rates: RoofRates) -> Result<Roof, AccountingError> {
        rates.roof(self.useful_products, self.unavoidable_bytes)
    }
}

pub fn cycle_round_accounting(
    shape: FamilyShape,
    kind: CycleRoundKind,
    source_elements: u128,
) -> Result<CycleRoundAccounting, AccountingError> {
    shape.validate()?;
    if source_elements < 2 || !source_elements.is_power_of_two() || source_elements > shape.rows {
        return Err(AccountingError::InvalidRoundElements(source_elements));
    }
    let root_fields = checked_mul(STAGES as u128, FIELD_BYTES)?;
    let ra_table_bytes = checked_mul(
        checked_mul(
            RA_FACTORS as u128,
            1 << super::relation::COMMITTED_CHUNK_BITS,
        )?,
        FIELD_BYTES,
    )?;
    let dense_source_bytes = checked_mul(checked_mul(DENSE_PLANES, FIELD_BYTES)?, source_elements)?;
    let dense_destination_bytes = dense_source_bytes / 2;

    let (useful_products, requested_bytes, unavoidable_bytes, work_items) = match kind {
        CycleRoundKind::RowMessage => {
            if source_elements != shape.rows {
                return Err(AccountingError::InvalidRoundElements(source_elements));
            }
            let coefficient_products = checked_mul(STAGES as u128, source_elements)?;
            let q10_products = checked_mul(Q10_PRODUCTS_PER_PAIR, source_elements / 2)?;
            let row_bytes = checked_mul(SHARED_ROW_BYTES, source_elements)?;
            let logical_root_bytes = checked_mul(root_fields, source_elements)?;
            let logical_ra_bytes = checked_mul(
                checked_mul(RA_FACTORS as u128, FIELD_BYTES)?,
                source_elements,
            )?;
            let root_unique = checked_add(
                checked_mul(root_fields, shape.inner_length)?,
                checked_mul(root_fields, shape.outer_length)?,
            )?;
            (
                checked_add(coefficient_products, q10_products)?,
                checked_sum([row_bytes, logical_root_bytes, logical_ra_bytes])?,
                checked_sum([row_bytes, root_unique, ra_table_bytes])?,
                source_elements / 2,
            )
        }
        CycleRoundKind::RowBindMessage => {
            if source_elements != shape.rows {
                return Err(AccountingError::InvalidRoundElements(source_elements));
            }
            let coefficient_products = checked_mul(STAGES as u128, source_elements / 2)?;
            let tail_bind_products = checked_mul((RA_FACTORS + 1) as u128, source_elements / 2)?;
            let q10_products = checked_mul(Q10_PRODUCTS_PER_PAIR, source_elements / 4)?;
            let root_bind_products = checked_mul(STAGES as u128, shape.inner_length / 2)?;
            let row_bytes = checked_mul(SHARED_ROW_BYTES, source_elements)?;
            let logical_root_bytes = checked_mul(root_fields, source_elements / 2)?;
            let logical_ra_bytes = checked_mul(
                checked_mul(RA_FACTORS as u128, FIELD_BYTES)?,
                source_elements,
            )?;
            let root_unique = checked_sum([
                checked_mul(root_fields, shape.inner_length)?,
                checked_mul(root_fields, shape.inner_length / 2)?,
                checked_mul(root_fields, shape.outer_length)?,
            ])?;
            (
                checked_sum([
                    coefficient_products,
                    tail_bind_products,
                    q10_products,
                    root_bind_products,
                ])?,
                checked_sum([
                    row_bytes,
                    logical_root_bytes,
                    logical_ra_bytes,
                    dense_destination_bytes,
                ])?,
                checked_sum([
                    row_bytes,
                    dense_destination_bytes,
                    root_unique,
                    ra_table_bytes,
                ])?,
                source_elements / 4,
            )
        }
        CycleRoundKind::DenseBindMessage => {
            if source_elements < 4 {
                return Err(AccountingError::InvalidRoundElements(source_elements));
            }
            let bind_products = checked_mul(DENSE_PLANES, source_elements / 2)?;
            let q10_products = checked_mul(Q10_PRODUCTS_PER_PAIR, source_elements / 4)?;
            (
                checked_add(bind_products, q10_products)?,
                checked_add(dense_source_bytes, dense_destination_bytes)?,
                checked_add(dense_source_bytes, dense_destination_bytes)?,
                source_elements / 4,
            )
        }
        CycleRoundKind::TerminalBind => {
            if source_elements != 2 {
                return Err(AccountingError::InvalidRoundElements(source_elements));
            }
            (
                DENSE_PLANES,
                checked_add(dense_source_bytes, dense_destination_bytes)?,
                checked_add(dense_source_bytes, dense_destination_bytes)?,
                1,
            )
        }
    };
    Ok(CycleRoundAccounting {
        kind,
        source_elements,
        useful_products,
        requested_bytes,
        unavoidable_bytes,
        work_items,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoofRates {
    bytes_per_second: u128,
    products_per_second: u128,
}

impl RoofRates {
    pub fn new(bytes_per_second: u128, products_per_second: u128) -> Result<Self, AccountingError> {
        if bytes_per_second == 0 {
            return Err(AccountingError::ZeroRate("bytes"));
        }
        if products_per_second == 0 {
            return Err(AccountingError::ZeroRate("products"));
        }
        Ok(Self {
            bytes_per_second,
            products_per_second,
        })
    }

    pub const fn bytes_per_second(self) -> u128 {
        self.bytes_per_second
    }

    pub const fn products_per_second(self) -> u128 {
        self.products_per_second
    }

    fn roof(self, products: u128, bytes: u128) -> Result<Roof, AccountingError> {
        let compute_ns = rate_ns(products, self.products_per_second)?;
        let traffic_ns = rate_ns(bytes, self.bytes_per_second)?;
        Ok(Roof {
            useful_products: products,
            unavoidable_bytes: bytes,
            compute_ns,
            traffic_ns,
            lower_bound_ns: compute_ns.max(traffic_ns),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Roof {
    useful_products: u128,
    unavoidable_bytes: u128,
    compute_ns: u128,
    traffic_ns: u128,
    lower_bound_ns: u128,
}

impl Roof {
    pub const fn useful_products(self) -> u128 {
        self.useful_products
    }

    pub const fn unavoidable_bytes(self) -> u128 {
        self.unavoidable_bytes
    }

    pub const fn compute_ns(self) -> u128 {
        self.compute_ns
    }

    pub const fn traffic_ns(self) -> u128 {
        self.traffic_ns
    }

    pub const fn lower_bound_ns(self) -> u128 {
        self.lower_bound_ns
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionProfile {
    metal_rates: RoofRates,
    cpu_rates: RoofRates,
    metal_round_ns: u128,
    cpu_round_ns: u128,
    handoff_ns: u128,
    threads_per_threadgroup: u128,
    minimum_threadgroups: u128,
}

impl ExecutionProfile {
    pub fn new(
        metal_rates: RoofRates,
        cpu_rates: RoofRates,
        fixed_costs: FixedCosts,
        occupancy: OccupancyFloor,
    ) -> Result<Self, SelectionError> {
        if occupancy.threads_per_threadgroup == 0 || occupancy.minimum_threadgroups == 0 {
            return Err(SelectionError::ZeroOccupancy);
        }
        Ok(Self {
            metal_rates,
            cpu_rates,
            metal_round_ns: fixed_costs.metal_round_ns,
            cpu_round_ns: fixed_costs.cpu_round_ns,
            handoff_ns: fixed_costs.handoff_ns,
            threads_per_threadgroup: occupancy.threads_per_threadgroup,
            minimum_threadgroups: occupancy.minimum_threadgroups,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedCosts {
    pub metal_round_ns: u128,
    pub cpu_round_ns: u128,
    pub handoff_ns: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OccupancyFloor {
    pub threads_per_threadgroup: u128,
    pub minimum_threadgroups: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpeedupTarget {
    cpu_member_ns: u128,
    minimum_speedup: u128,
}

impl SpeedupTarget {
    pub fn new(cpu_member_ns: u128, minimum_speedup: u128) -> Result<Self, SelectionError> {
        if cpu_member_ns == 0 || minimum_speedup == 0 {
            return Err(SelectionError::ZeroTarget);
        }
        Ok(Self {
            cpu_member_ns,
            minimum_speedup,
        })
    }

    pub const fn cap_ns(self) -> u128 {
        self.cpu_member_ns / self.minimum_speedup
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CycleCutoffPlan {
    metal_message_rounds: usize,
    dense_handoff_elements: u128,
    projected_ns: u128,
    host_only_ns: u128,
    target_cap_ns: u128,
}

impl CycleCutoffPlan {
    pub const fn metal_message_rounds(self) -> usize {
        self.metal_message_rounds
    }

    pub const fn dense_handoff_elements(self) -> u128 {
        self.dense_handoff_elements
    }

    pub const fn projected_ns(self) -> u128 {
        self.projected_ns
    }

    pub const fn host_only_ns(self) -> u128 {
        self.host_only_ns
    }

    pub const fn target_cap_ns(self) -> u128 {
        self.target_cap_ns
    }

    pub const fn clears_target(self) -> bool {
        self.projected_ns <= self.target_cap_ns
    }
}

/// Chooses zero Metal rounds or a prefix containing both row-derived rounds.
pub fn select_cycle_cutoff(
    shape: FamilyShape,
    profile: ExecutionProfile,
    target: SpeedupTarget,
) -> Result<CycleCutoffPlan, SelectionError> {
    shape.validate()?;
    let schedule = cycle_schedule(shape)?;
    let host_only_ns = project_slice(&schedule, profile.cpu_rates, profile.cpu_round_ns)?;
    let mut best_rounds = 0usize;
    let mut best_handoff = 0u128;
    let mut best_ns = host_only_ns;

    for prefix in 2..=schedule.len().saturating_sub(1) {
        let metal_rounds = &schedule[..prefix];
        if metal_rounds.iter().any(|round| {
            round.work_items.div_ceil(profile.threads_per_threadgroup)
                < profile.minimum_threadgroups
        }) {
            break;
        }
        let metal_ns = project_slice(metal_rounds, profile.metal_rates, profile.metal_round_ns)?;
        let cpu_ns = project_slice(&schedule[prefix..], profile.cpu_rates, profile.cpu_round_ns)?;
        let handoff_elements = shape.rows >> (prefix - 1);
        let handoff_bytes = checked_mul(checked_mul(DENSE_PLANES, FIELD_BYTES)?, handoff_elements)?;
        let handoff_ns = checked_add(
            rate_ns(handoff_bytes, profile.metal_rates.bytes_per_second)?,
            profile.handoff_ns,
        )?;
        let projected = checked_sum([metal_ns, handoff_ns, cpu_ns])?;
        if projected < best_ns {
            best_ns = projected;
            best_rounds = prefix;
            best_handoff = handoff_elements;
        }
    }
    Ok(CycleCutoffPlan {
        metal_message_rounds: best_rounds,
        dense_handoff_elements: best_handoff,
        projected_ns: best_ns,
        host_only_ns,
        target_cap_ns: target.cap_ns(),
    })
}

fn cycle_schedule(shape: FamilyShape) -> Result<Vec<CycleRoundAccounting>, AccountingError> {
    let mut schedule = Vec::with_capacity(shape.log_t + 1);
    schedule.push(cycle_round_accounting(
        shape,
        CycleRoundKind::RowMessage,
        shape.rows,
    )?);
    schedule.push(cycle_round_accounting(
        shape,
        CycleRoundKind::RowBindMessage,
        shape.rows,
    )?);
    let mut source = shape.rows / 2;
    while source >= 4 {
        schedule.push(cycle_round_accounting(
            shape,
            CycleRoundKind::DenseBindMessage,
            source,
        )?);
        source /= 2;
    }
    schedule.push(cycle_round_accounting(
        shape,
        CycleRoundKind::TerminalBind,
        2,
    )?);
    Ok(schedule)
}

fn project_slice(
    rounds: &[CycleRoundAccounting],
    rates: RoofRates,
    fixed_round_ns: u128,
) -> Result<u128, AccountingError> {
    rounds.iter().try_fold(0u128, |total, round| {
        checked_add(
            total,
            checked_add(round.roof(rates)?.lower_bound_ns, fixed_round_ns)?,
        )
    })
}

fn rate_ns(units: u128, rate: u128) -> Result<u128, AccountingError> {
    if rate == 0 {
        return Err(AccountingError::ZeroRate("roof"));
    }
    checked_mul(units, NANOS_PER_SECOND).map(|value| value.div_ceil(rate))
}

fn domain_size(log_size: usize) -> Result<u128, AccountingError> {
    let shift = u32::try_from(log_size).map_err(|_| AccountingError::Overflow)?;
    1u128.checked_shl(shift).ok_or(AccountingError::Overflow)
}

fn checked_sum(values: impl IntoIterator<Item = u128>) -> Result<u128, AccountingError> {
    values.into_iter().try_fold(0u128, checked_add)
}

fn checked_add(left: u128, right: u128) -> Result<u128, AccountingError> {
    left.checked_add(right).ok_or(AccountingError::Overflow)
}

fn checked_mul(left: u128, right: u128) -> Result<u128, AccountingError> {
    left.checked_mul(right).ok_or(AccountingError::Overflow)
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum AccountingError {
    #[error("bytecode read/RAF analytical shape is invalid")]
    InvalidShape,
    #[error("bytecode cycle round source length {0} is invalid")]
    InvalidRoundElements(u128),
    #[error("bytecode read/RAF {0} roof rate is zero")]
    ZeroRate(&'static str),
    #[error("bytecode read/RAF accounting overflowed")]
    Overflow,
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum SelectionError {
    #[error(transparent)]
    Accounting(#[from] AccountingError),
    #[error("bytecode read/RAF occupancy floor is zero")]
    ZeroOccupancy,
    #[error("bytecode read/RAF speedup target is zero")]
    ZeroTarget,
}
