use thiserror::Error;

use super::model::{AccountingError, RoofRates};
use super::owner::{OwnerError, RamCycleFamilyOwner};
use super::topology::LevelCensus;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionLane {
    HostSparse,
    MetalPrefix,
    DenseFallback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RwLevelSchedule {
    Flat,
    Grouped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[expect(
    clippy::struct_field_names,
    reason = "the nanosecond suffix distinguishes latency from work counts"
)]
pub struct ExecutionOverheads {
    metal_round_ns: u128,
    cpu_round_ns: u128,
    handoff_ns: u128,
    metal_threadgroup_ns: u128,
}

impl ExecutionOverheads {
    pub const fn new(
        metal_round_ns: u128,
        cpu_round_ns: u128,
        handoff_ns: u128,
        metal_threadgroup_ns: u128,
    ) -> Self {
        Self {
            metal_round_ns,
            cpu_round_ns,
            handoff_ns,
            metal_threadgroup_ns,
        }
    }

    pub const fn metal_round_ns(self) -> u128 {
        self.metal_round_ns
    }

    pub const fn cpu_round_ns(self) -> u128 {
        self.cpu_round_ns
    }

    pub const fn handoff_ns(self) -> u128 {
        self.handoff_ns
    }

    pub const fn metal_threadgroup_ns(self) -> u128 {
        self.metal_threadgroup_ns
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionProfile {
    metal_rates: RoofRates,
    cpu_rates: RoofRates,
    overheads: ExecutionOverheads,
    threadgroup_width: u64,
    minimum_metal_threadgroups: u64,
}

impl ExecutionProfile {
    pub fn new(
        metal_rates: RoofRates,
        cpu_rates: RoofRates,
        overheads: ExecutionOverheads,
        threadgroup_width: u64,
        minimum_metal_threadgroups: u64,
    ) -> Result<Self, SelectionError> {
        if threadgroup_width == 0 {
            return Err(SelectionError::ZeroThreadgroupWidth);
        }
        if minimum_metal_threadgroups == 0 {
            return Err(SelectionError::ZeroMinimumThreadgroups);
        }
        Ok(Self {
            metal_rates,
            cpu_rates,
            overheads,
            threadgroup_width,
            minimum_metal_threadgroups,
        })
    }

    pub const fn metal_rates(self) -> RoofRates {
        self.metal_rates
    }

    pub const fn cpu_rates(self) -> RoofRates {
        self.cpu_rates
    }

    pub const fn overheads(self) -> ExecutionOverheads {
        self.overheads
    }

    pub const fn threadgroup_width(self) -> u64 {
        self.threadgroup_width
    }

    pub const fn minimum_metal_threadgroups(self) -> u64 {
        self.minimum_metal_threadgroups
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CycleCutoffPlan {
    lane: ExecutionLane,
    cycle_cutoff: usize,
    projected_cycle_ns: u128,
    projected_host_only_ns: u128,
    schedules: Box<[RwLevelSchedule]>,
    metal_threadgroups: u128,
}

impl CycleCutoffPlan {
    pub const fn lane(&self) -> ExecutionLane {
        self.lane
    }

    pub const fn cycle_cutoff(&self) -> usize {
        self.cycle_cutoff
    }

    pub const fn projected_cycle_ns(&self) -> u128 {
        self.projected_cycle_ns
    }

    pub const fn projected_host_only_ns(&self) -> u128 {
        self.projected_host_only_ns
    }

    pub fn read_write_schedules(&self) -> &[RwLevelSchedule] {
        &self.schedules
    }

    pub const fn metal_threadgroups(&self) -> u128 {
        self.metal_threadgroups
    }
}

pub fn select_value_check(
    owner: &RamCycleFamilyOwner,
    profile: ExecutionProfile,
) -> Result<CycleCutoffPlan, SelectionError> {
    owner.verify_integrity()?;
    validate_profile_owner(owner, profile)?;
    let census = owner.block_topology().census();
    validate_census(census, owner.receipt().log_t())?;
    let mut rounds = Vec::with_capacity(owner.receipt().log_t());
    for level in census.iter().skip(1) {
        let entries = u128::from(level.entries());
        let products = checked_mul(12, entries)?;
        let logical_bytes = checked_mul(144, entries)?;
        let threadgroups = threadgroups(level.entries(), profile.threadgroup_width)?;
        rounds.push(RoundProjection {
            metal_ns: metal_round_cost(profile, products, logical_bytes, threadgroups)?,
            cpu_ns: cpu_round_cost(profile, products, logical_bytes)?,
            threadgroups,
        });
    }
    choose_cutoff(&rounds, profile, Box::new([]))
}

pub fn select_read_write(
    owner: &RamCycleFamilyOwner,
    profile: ExecutionProfile,
) -> Result<CycleCutoffPlan, SelectionError> {
    owner.verify_integrity()?;
    validate_profile_owner(owner, profile)?;
    let census = owner.read_write_topology().census();
    validate_census(census, owner.receipt().log_t())?;
    let mut rounds = Vec::with_capacity(owner.receipt().log_t());
    let mut schedules = Vec::with_capacity(owner.receipt().log_t());

    for level in census.iter().skip(1) {
        if level.groups() > level.entries() || level.tiles() < level.groups() {
            return Err(SelectionError::InvalidReadWriteCensus);
        }
        let entries = u128::from(level.entries());
        let groups = u128::from(level.groups());
        let flat_products = checked_add(checked_mul(8, entries)?, checked_mul(2, groups)?)?;
        let grouped_products = checked_add(checked_mul(6, entries)?, checked_mul(4, groups)?)?;
        let flat_threadgroups = threadgroups(level.entries(), profile.threadgroup_width)?;
        let grouped_threadgroups = u128::from(level.tiles());
        let shared_bytes = checked_add(checked_mul(64, entries)?, checked_mul(128, groups)?)?;
        let flat_bytes = checked_add(shared_bytes, checked_mul(32, flat_threadgroups)?)?;
        let grouped_bytes = checked_add(shared_bytes, checked_mul(32, grouped_threadgroups)?)?;
        let flat_metal_ns =
            metal_round_cost(profile, flat_products, flat_bytes, flat_threadgroups)?;
        let grouped_metal_ns = metal_round_cost(
            profile,
            grouped_products,
            grouped_bytes,
            grouped_threadgroups,
        )?;
        let (schedule, products, bytes, threadgroups, metal_ns) =
            if grouped_metal_ns < flat_metal_ns {
                (
                    RwLevelSchedule::Grouped,
                    grouped_products,
                    grouped_bytes,
                    grouped_threadgroups,
                    grouped_metal_ns,
                )
            } else {
                (
                    RwLevelSchedule::Flat,
                    flat_products,
                    flat_bytes,
                    flat_threadgroups,
                    flat_metal_ns,
                )
            };
        schedules.push(schedule);
        rounds.push(RoundProjection {
            metal_ns,
            cpu_ns: cpu_round_cost(profile, products, bytes)?,
            threadgroups,
        });
    }
    choose_cutoff(&rounds, profile, schedules.into_boxed_slice())
}

#[derive(Clone, Copy)]
struct RoundProjection {
    metal_ns: u128,
    cpu_ns: u128,
    threadgroups: u128,
}

fn choose_cutoff(
    rounds: &[RoundProjection],
    profile: ExecutionProfile,
    schedules: Box<[RwLevelSchedule]>,
) -> Result<CycleCutoffPlan, SelectionError> {
    let host_only = checked_sum(rounds.iter().map(|round| round.cpu_ns))?;
    let mut maximum_prefix = 0usize;
    let minimum = u128::from(profile.minimum_metal_threadgroups);
    for round in rounds {
        if round.threadgroups < minimum {
            break;
        }
        maximum_prefix = maximum_prefix
            .checked_add(1)
            .ok_or(SelectionError::Overflow)?;
    }

    let mut best_cutoff = 0usize;
    let mut best_ns = host_only;
    for cutoff in 1..=maximum_prefix {
        let metal_ns = checked_sum(
            rounds
                .get(..cutoff)
                .ok_or(SelectionError::InvalidCutoff)?
                .iter()
                .map(|round| round.metal_ns),
        )?;
        let cpu_ns = checked_sum(
            rounds
                .get(cutoff..)
                .ok_or(SelectionError::InvalidCutoff)?
                .iter()
                .map(|round| round.cpu_ns),
        )?;
        let handoff_ns = if cutoff < rounds.len() {
            profile.overheads.handoff_ns
        } else {
            0
        };
        let projected = checked_sum([metal_ns, cpu_ns, handoff_ns])?;
        if projected < best_ns {
            best_cutoff = cutoff;
            best_ns = projected;
        }
    }
    let metal_threadgroups = checked_sum(
        rounds
            .get(..best_cutoff)
            .ok_or(SelectionError::InvalidCutoff)?
            .iter()
            .map(|round| round.threadgroups),
    )?;
    Ok(CycleCutoffPlan {
        lane: if best_cutoff == 0 {
            ExecutionLane::HostSparse
        } else {
            ExecutionLane::MetalPrefix
        },
        cycle_cutoff: best_cutoff,
        projected_cycle_ns: best_ns,
        projected_host_only_ns: host_only,
        schedules,
        metal_threadgroups,
    })
}

fn metal_round_cost(
    profile: ExecutionProfile,
    products: u128,
    logical_bytes: u128,
    threadgroups: u128,
) -> Result<u128, SelectionError> {
    let roof = profile.metal_rates.account(products, logical_bytes)?;
    checked_sum([
        profile.overheads.metal_round_ns,
        checked_mul(profile.overheads.metal_threadgroup_ns, threadgroups)?,
        roof.lower_bound_ns(),
    ])
}

fn cpu_round_cost(
    profile: ExecutionProfile,
    products: u128,
    logical_bytes: u128,
) -> Result<u128, SelectionError> {
    let roof = profile.cpu_rates.account(products, logical_bytes)?;
    checked_add(profile.overheads.cpu_round_ns, roof.lower_bound_ns())
}

fn threadgroups(entries: u64, width: u64) -> Result<u128, SelectionError> {
    if width == 0 {
        return Err(SelectionError::ZeroThreadgroupWidth);
    }
    Ok(u128::from(entries.div_ceil(width)))
}

fn validate_profile_owner(
    owner: &RamCycleFamilyOwner,
    profile: ExecutionProfile,
) -> Result<(), SelectionError> {
    let owner_width =
        u64::try_from(owner.receipt().threadgroup_width()).map_err(|_| SelectionError::Overflow)?;
    if owner_width != profile.threadgroup_width {
        return Err(SelectionError::ThreadgroupWidthMismatch {
            owner: owner_width,
            profile: profile.threadgroup_width,
        });
    }
    Ok(())
}

fn validate_census(census: &[LevelCensus], log_t: usize) -> Result<(), SelectionError> {
    let expected = log_t.checked_add(1).ok_or(SelectionError::Overflow)?;
    if census.len() != expected {
        return Err(SelectionError::CensusLength {
            expected,
            got: census.len(),
        });
    }
    Ok(())
}

fn checked_add(left: u128, right: u128) -> Result<u128, SelectionError> {
    left.checked_add(right).ok_or(SelectionError::Overflow)
}

fn checked_mul(left: u128, right: u128) -> Result<u128, SelectionError> {
    left.checked_mul(right).ok_or(SelectionError::Overflow)
}

fn checked_sum<I>(values: I) -> Result<u128, SelectionError>
where
    I: IntoIterator<Item = u128>,
{
    values.into_iter().try_fold(0u128, checked_add)
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum SelectionError {
    #[error(transparent)]
    Owner(#[from] OwnerError),
    #[error(transparent)]
    Accounting(#[from] AccountingError),
    #[error("RAM selector threadgroup width must be nonzero")]
    ZeroThreadgroupWidth,
    #[error("RAM selector minimum Metal threadgroups must be nonzero")]
    ZeroMinimumThreadgroups,
    #[error("RAM selector owner width {owner} differs from profile width {profile}")]
    ThreadgroupWidthMismatch { owner: u64, profile: u64 },
    #[error("RAM selector read/write census is invalid")]
    InvalidReadWriteCensus,
    #[error("RAM selector topology census has {got} levels, expected {expected}")]
    CensusLength { expected: usize, got: usize },
    #[error("RAM selector produced an invalid cutoff")]
    InvalidCutoff,
    #[error("RAM selector arithmetic overflowed")]
    Overflow,
}
