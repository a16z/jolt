//! Analytical selector for the RAM read/write successor design.
//!
//! This file is not registered by `solinas/mod.rs`. It fixes the cost model
//! that must be evaluated against a real access-topology census before shader
//! implementation starts.

use core::fmt;

pub const CPU_MEMBER_NS: u64 = 154_710_378;
pub const CPU_PREPARE_NS: u64 = 110_985_042;
pub const CPU_CYCLE_NS: u64 = 41_641_794;
pub const CPU_ADDRESS_TAIL_NS: u64 = 1_793_751;
pub const FIVE_X_NS: u64 = 30_942_075;
pub const EIGHT_X_NS: u64 = 19_338_797;

pub const COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const FIELD_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;
pub const ROUND_WAIT_NS: u64 = 141_000;
pub const HOST_FS_NS: u64 = 2_000;
pub const CYCLE_ROUNDS: u64 = 26;
pub const PROTOCOL_ROUNDS: u64 = 39;

const FIELD_BYTES: u64 = 16;
const STATE_BYTES: u64 = 2 * FIELD_BYTES;
const MESSAGE_PAIR_BYTES: u64 = 2 * FIELD_BYTES;
const FLAT_EVENT_BYTES: u64 = 32;
const GROUPED_EVENT_BYTES: u64 = 24;
const FLAT_GROUP_BYTES: u64 = 16;
const GROUPED_GROUP_BYTES: u64 = 24;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Schedule {
    Flat,
    Grouped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CycleLevel {
    /// Number of `(cycle block, address)` entries after this bind.
    pub entries: u64,
    /// Number of nonempty cycle blocks after this bind.
    pub groups: u64,
    /// Sum over groups of `ceil(group_entries / threads)`.
    pub grouped_tiles: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LevelCost {
    pub schedule: Schedule,
    pub products: u64,
    pub bytes: u64,
    pub local_floor_ns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Projection {
    pub levels: Vec<LevelCost>,
    pub products: u64,
    pub bytes: u64,
    pub compute_floor_ns: u64,
    pub traffic_floor_ns: u64,
    pub active_floor_ns: u64,
    pub fixed_primary_ns: u64,
    pub optimistic_primary_ns: u64,
    pub conservative_primary_ns: u64,
    pub resident_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelError {
    WrongLevelCount,
    InvalidLevel,
    Overflow,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongLevelCount => {
                f.write_str("RAM successor needs one count for each cycle round")
            }
            Self::InvalidLevel => f.write_str("RAM successor level counts are inconsistent"),
            Self::Overflow => f.write_str("RAM successor cost arithmetic overflowed"),
        }
    }
}

/// Selects the cheaper exact schedule at each level.
///
/// `accesses` is `E_0`; `levels[b - 1]` contains `(E_b, G_b, D_b)`.
/// Empty streams use all-zero counts. The projection keeps producer wall out
/// of the PIOP boundary, but includes the owner join in `owner_join_ns`.
pub fn project(
    accesses: u64,
    levels: &[CycleLevel],
    threads: u64,
    owner_join_ns: u64,
) -> Result<Projection, ModelError> {
    if levels.len() != CYCLE_ROUNDS as usize || threads == 0 {
        return Err(ModelError::WrongLevelCount);
    }
    validate(accesses, levels, threads)?;

    let mut selected = Vec::with_capacity(levels.len());
    let mut total_products = 0_u64;
    let mut total_bytes = add(initialization_bytes(accesses)?, mul(STATE_BYTES, accesses)?)?;
    let mut previous_entries = accesses;
    let mut previous_groups = accesses;
    let mut maximum_entries = accesses;
    let mut maximum_groups = accesses;
    let mut topology_bytes = 0_u64;
    let mut maximum_scratch = 0_u64;

    for level in levels {
        let flat = flat_level(*level, previous_groups, threads)?;
        let grouped = grouped_level(*level, previous_groups)?;
        let cost = if grouped.local_floor_ns < flat.local_floor_ns {
            grouped
        } else {
            flat
        };
        total_products = add(total_products, cost.products)?;
        total_bytes = add(total_bytes, cost.bytes)?;
        total_bytes = add(
            total_bytes,
            mul(STATE_BYTES, add(previous_entries, level.entries)?)?,
        )?;
        topology_bytes = add(
            topology_bytes,
            match cost.schedule {
                Schedule::Flat => add(
                    mul(FLAT_EVENT_BYTES, level.entries)?,
                    mul(FLAT_GROUP_BYTES, level.groups)?,
                )?,
                Schedule::Grouped => add(
                    mul(GROUPED_EVENT_BYTES, level.entries)?,
                    mul(GROUPED_GROUP_BYTES, level.groups)?,
                )?,
            },
        )?;
        maximum_scratch = maximum_scratch.max(match cost.schedule {
            Schedule::Flat => level.entries.div_ceil(threads),
            Schedule::Grouped => level.grouped_tiles,
        });
        maximum_entries = maximum_entries.max(level.entries);
        maximum_groups = maximum_groups.max(level.groups);
        previous_entries = level.entries;
        previous_groups = level.groups;
        selected.push(cost);
    }

    let products_ns = rate_ns(total_products, FIELD_PRODUCTS_PER_SECOND)?;
    let traffic_ns = rate_ns(total_bytes, COPY_BYTES_PER_SECOND)?;
    let active_ns = products_ns.max(traffic_ns);
    let fixed_ns = fixed_primary_ns(owner_join_ns)?;
    let optimistic_ns = add(fixed_ns, active_ns)?;
    let conservative_ns = add(fixed_ns, add(products_ns, traffic_ns)?)?;
    let resident_bytes = [
        mul(24, accesses)?,
        topology_bytes,
        mul(2 * STATE_BYTES, maximum_entries)?,
        mul(2 * FIELD_BYTES, maximum_groups)?,
        mul(2 * MESSAGE_PAIR_BYTES, maximum_scratch)?,
        393_184,
    ]
    .into_iter()
    .try_fold(0_u64, add)?;

    Ok(Projection {
        levels: selected,
        products: total_products,
        bytes: total_bytes,
        compute_floor_ns: products_ns,
        traffic_floor_ns: traffic_ns,
        active_floor_ns: active_ns,
        fixed_primary_ns: fixed_ns,
        optimistic_primary_ns: optimistic_ns,
        conservative_primary_ns: conservative_ns,
        resident_bytes,
    })
}

fn flat_level(
    level: CycleLevel,
    present_increment_sources: u64,
    threads: u64,
) -> Result<LevelCost, ModelError> {
    // Six message products and two state-bind products for each entry, plus
    // one eq-weight and one increment bind for each group.
    let products = add(mul(8, level.entries)?, mul(2, level.groups)?)?;
    let partials = level.entries.div_ceil(threads);
    let bytes = [
        mul(2 * FLAT_EVENT_BYTES, level.entries)?,
        mul(176, level.groups)?,
        mul(2 * FIELD_BYTES, present_increment_sources)?,
        reduction_bytes(partials)?,
    ]
    .into_iter()
    .try_fold(0_u64, add)?;
    Ok(level_cost(Schedule::Flat, products, bytes)?)
}

fn grouped_level(
    level: CycleLevel,
    present_increment_sources: u64,
) -> Result<LevelCost, ModelError> {
    // Factor the eq weight outside each group's address sum: four message
    // products and two state binds per entry, then three weight products and
    // one increment bind per group.
    let products = add(mul(6, level.entries)?, mul(4, level.groups)?)?;
    let segmented_reduction = add(
        mul(2 * MESSAGE_PAIR_BYTES, level.grouped_tiles)?,
        add(
            mul(MESSAGE_PAIR_BYTES, level.groups)?,
            reduction_bytes(level.groups)?,
        )?,
    )?;
    let bytes = [
        mul(2 * GROUPED_EVENT_BYTES, level.entries)?,
        mul(104, level.groups)?,
        mul(2 * FIELD_BYTES, present_increment_sources)?,
        segmented_reduction,
    ]
    .into_iter()
    .try_fold(0_u64, add)?;
    Ok(level_cost(Schedule::Grouped, products, bytes)?)
}

fn level_cost(schedule: Schedule, products: u64, bytes: u64) -> Result<LevelCost, ModelError> {
    let compute_ns = rate_ns(products, FIELD_PRODUCTS_PER_SECOND)?;
    let traffic_ns = rate_ns(bytes, COPY_BYTES_PER_SECOND)?;
    Ok(LevelCost {
        schedule,
        products,
        bytes,
        local_floor_ns: compute_ns.max(traffic_ns),
    })
}

fn initialization_bytes(accesses: u64) -> Result<u64, ModelError> {
    // 24-byte record read, 32-byte state write, 16-byte increment write.
    mul(72, accesses)
}

fn fixed_primary_ns(owner_join_ns: u64) -> Result<u64, ModelError> {
    [
        owner_join_ns,
        ROUND_WAIT_NS,
        mul(CYCLE_ROUNDS, ROUND_WAIT_NS)?,
        ROUND_WAIT_NS,
        CPU_ADDRESS_TAIL_NS,
        mul(PROTOCOL_ROUNDS, HOST_FS_NS)?,
    ]
    .into_iter()
    .try_fold(0_u64, add)
}

fn reduction_bytes(mut partials: u64) -> Result<u64, ModelError> {
    if partials == 0 {
        return Ok(0);
    }
    let mut bytes = mul(MESSAGE_PAIR_BYTES, partials)?;
    while partials > 1 {
        let next = partials.div_ceil(32);
        bytes = add(
            bytes,
            add(
                mul(MESSAGE_PAIR_BYTES, partials)?,
                mul(MESSAGE_PAIR_BYTES, next)?,
            )?,
        )?;
        partials = next;
    }
    Ok(bytes)
}

fn validate(accesses: u64, levels: &[CycleLevel], threads: u64) -> Result<(), ModelError> {
    let mut previous_entries = accesses;
    let mut previous_groups = accesses;
    for level in levels {
        let empty = level.entries == 0 && level.groups == 0 && level.grouped_tiles == 0;
        let nonempty = level.entries > 0
            && level.groups > 0
            && level.groups <= level.entries
            && level.grouped_tiles >= level.groups
            && level.grouped_tiles <= level.entries
            && level.entries <= previous_entries
            && level.groups <= previous_groups
            && level.entries >= previous_entries.div_ceil(2)
            && level.groups >= previous_groups.div_ceil(2)
            && level.grouped_tiles >= level.entries.div_ceil(threads);
        if (accesses == 0 && !empty) || (accesses > 0 && !nonempty) {
            return Err(ModelError::InvalidLevel);
        }
        previous_entries = level.entries;
        previous_groups = level.groups;
    }
    if accesses > 0
        && levels
            .last()
            .is_none_or(|level| level.entries == 0 || level.groups != 1)
    {
        return Err(ModelError::InvalidLevel);
    }
    Ok(())
}

fn rate_ns(work: u64, rate: u64) -> Result<u64, ModelError> {
    let numerator = u128::from(work) * 1_000_000_000;
    u64::try_from(numerator.div_ceil(u128::from(rate))).map_err(|_| ModelError::Overflow)
}

fn add(left: u64, right: u64) -> Result<u64, ModelError> {
    left.checked_add(right).ok_or(ModelError::Overflow)
}

fn mul(left: u64, right: u64) -> Result<u64, ModelError> {
    left.checked_mul(right).ok_or(ModelError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hot_levels(accesses: u64) -> Vec<CycleLevel> {
        (1..=CYCLE_ROUNDS)
            .map(|round| {
                let count = (accesses >> round).max(1);
                CycleLevel {
                    entries: count,
                    groups: count,
                    grouped_tiles: count,
                }
            })
            .collect()
    }

    #[test]
    fn hot_address_keeps_flat_schedule() {
        let projection = project(1 << 22, &hot_levels(1 << 22), 256, 0).unwrap();
        assert!(projection
            .levels
            .iter()
            .all(|level| level.schedule == Schedule::Flat));
        assert!(projection.optimistic_primary_ns < FIVE_X_NS);
    }

    #[test]
    fn grouped_schedule_wins_for_wide_groups() {
        let mut levels = Vec::new();
        let mut entries = 1_u64 << 20;
        let mut groups = 1_u64 << 20;
        for round in 0..CYCLE_ROUNDS {
            if round >= 6 {
                entries = entries.div_ceil(2);
            }
            groups = groups.div_ceil(2);
            levels.push(CycleLevel {
                entries,
                groups,
                grouped_tiles: entries.div_ceil(256).max(groups),
            });
        }
        let projection = project(1 << 20, &levels, 256, 0).unwrap();
        assert!(projection
            .levels
            .iter()
            .any(|level| level.schedule == Schedule::Grouped));
    }

    #[test]
    fn malformed_group_tile_census_is_rejected() {
        let mut levels = hot_levels(1 << 22);
        levels[0].grouped_tiles = 0;
        assert_eq!(
            project(1 << 22, &levels, 256, 0),
            Err(ModelError::InvalidLevel)
        );
    }
}
