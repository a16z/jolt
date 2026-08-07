use super::execution_abi::{
    COLUMNS, HISTOGRAM_HIGH_LENGTH, HISTOGRAM_LOW_LENGTH, MAX_DENSE_OUTER_LENGTH, MAX_TRACE_LOG_T,
    RAW_OUTER_LENGTH, RAW_ROUND_ZERO_INNER_LENGTH, SIMD_WIDTH, TARGET_SHARD_CYCLES,
    TARGET_SHARD_LOG_T,
};
use super::RegistersRwV3Error;

pub(crate) const CPU_BASELINE_NS: u64 = 934_665_875;
pub(crate) const LATEST_DIAGNOSTIC_CPU_NS: u64 = 971_178_000;
pub(crate) const CURRENT_CPU_FALLBACK_NS: u64 = 948_053_000;
pub(crate) const PRODUCER_PURSUIT_CAP_NS: u64 = 45_000_000;
pub(crate) const ANALYTICAL_EXECUTION_LOW_NS: u64 = 89_100_000;
pub(crate) const ANALYTICAL_EXECUTION_HIGH_NS: u64 = 118_500_000;

pub(crate) const LOG26_TOPOLOGY_ALLOCATION_BYTES: u64 = 1_209_532_416;
pub(crate) const LOG26_TOPOLOGY_INITIALIZED_BYTES: u64 = 1_039_896_120;
pub(crate) const LOG26_REGISTERS_VAL_BYTES: u64 = 1_140_850_688;
pub(crate) const LOG26_PRODUCER_ALLOCATION_BYTES: u64 = 2_350_383_104;
pub(crate) const LOG26_PRODUCER_INITIALIZED_WRITE_BYTES: u64 = 2_180_746_808;

pub(crate) const LOG26_RAW_CACHE_BYTES: u64 = 14_194_005_688;
pub(crate) const LOG26_RAW_REQUEST_EXCESS_BYTES: u64 = 18_452_848_704;
pub(crate) const LOG26_RAW_REQUESTED_BYTES: u64 = 32_646_854_392;
pub(crate) const LOG26_DENSE_CACHE_BYTES: u64 = 4_847_746_576;
pub(crate) const LOG26_DENSE_REQUESTED_BYTES: u64 = 4_849_843_264;
pub(crate) const LOG26_HISTOGRAM_CACHE_BYTES: u64 = 320_384_568;
pub(crate) const LOG26_HISTOGRAM_REQUESTED_BYTES: u64 = 2_169_475_512;

pub(crate) const LOG26_EXECUTION_CACHE_BYTES: u64 = 19_362_136_832;
pub(crate) const LOG26_EXECUTION_REQUESTED_BYTES: u64 = 39_666_173_168;
pub(crate) const LOG26_LIFECYCLE_CACHE_BYTES: u64 = 21_542_883_640;
pub(crate) const LOG26_LIFECYCLE_REQUESTED_BYTES: u64 = 41_846_919_976;

pub(crate) const LOG26_MAJOR_ARENAS_BYTES: u64 = 5_571_608_576;
pub(crate) const FIXED_COEFFICIENT_ARENA_BYTES: u64 = 3_145_728;
pub(crate) const FIXED_LOCAL_WEIGHT_ARENA_BYTES: u64 = 16_384;
pub(crate) const FIXED_EQUALITY_ARENAS_BYTES: u64 = 196_608;
pub(crate) const FIXED_PARTIAL_A_BYTES: u64 = 786_432;
pub(crate) const FIXED_PARTIAL_B_BYTES: u64 = 24_576;
pub(crate) const FIXED_SCRATCH_EXCLUDING_READBACK_BYTES: u64 = 4_169_728;
pub(crate) const LOG26_TAIL_READBACK_BYTES: u64 = 12_320;
pub(crate) const LOG26_PEAK_LOGICAL_BYTES: u64 = 5_575_790_624;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TraceExecutionPlan {
    pub(crate) log_t: u32,
    pub(crate) metal_shards: u32,
    pub(crate) metal_cycle_rounds: u32,
    pub(crate) cpu_high_cycle_rounds: u32,
    pub(crate) cpu_address_rounds: u32,
}

impl TraceExecutionPlan {
    pub(crate) fn for_log_t(log_t: u32) -> Result<Self, RegistersRwV3Error> {
        if !(TARGET_SHARD_LOG_T..=MAX_TRACE_LOG_T).contains(&log_t) {
            return Err(RegistersRwV3Error::InvalidExecutionLogT(log_t));
        }
        Ok(Self {
            log_t,
            metal_shards: 1 << (log_t - TARGET_SHARD_LOG_T),
            metal_cycle_rounds: TARGET_SHARD_LOG_T,
            cpu_high_cycle_rounds: log_t - TARGET_SHARD_LOG_T,
            cpu_address_rounds: 7,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProductCensus {
    pub(crate) full: u64,
    pub(crate) half: u64,
}

impl ProductCensus {
    fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            full: self.full.checked_add(other.full)?,
            half: self.half.checked_add(other.half)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ExecutionWork {
    pub(crate) products: ProductCensus,
    pub(crate) cache_unique_bytes: u64,
    pub(crate) requested_bytes: u64,
}

impl ExecutionWork {
    fn checked_add(self, other: Self) -> Result<Self, RegistersRwV3Error> {
        Ok(Self {
            products: self.products.checked_add(other.products).ok_or(
                RegistersRwV3Error::SizeOverflow("registers RW execution products"),
            )?,
            cache_unique_bytes: self
                .cache_unique_bytes
                .checked_add(other.cache_unique_bytes)
                .ok_or(RegistersRwV3Error::SizeOverflow(
                    "registers RW cache traffic",
                ))?,
            requested_bytes: self
                .requested_bytes
                .checked_add(other.requested_bytes)
                .ok_or(RegistersRwV3Error::SizeOverflow(
                    "registers RW requested traffic",
                ))?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawRoundProductCensus {
    pub(crate) round: u8,
    pub(crate) products: ProductCensus,
}

pub(crate) const LOG26_RAW_ROUND_PRODUCTS: [RawRoundProductCensus; 9] = [
    raw_products(0, 99_331_770),
    raw_products(1, 100_317_099),
    raw_products(2, 132_669_829),
    raw_products(3, 139_114_412),
    raw_products(4, 142_176_919),
    raw_products(5, 138_645_053),
    raw_products(6, 125_185_001),
    raw_products(7, 100_879_655),
    raw_products(8, 141_656_012),
];

const fn raw_products(round: u8, half: u64) -> RawRoundProductCensus {
    let full = if round == 0 {
        49_152
    } else {
        let width = 1u64 << round;
        let remaining = (TARGET_SHARD_CYCLES as u64) >> round;
        let groups = (TARGET_SHARD_CYCLES as u64) >> (round + 1);
        let increment_endpoints = if round == 1 { 2 * groups } else { groups };
        3 * width * width
            + remaining
            + increment_endpoints
            + 2 * groups
            + 2 * RAW_OUTER_LENGTH as u64
    };
    RawRoundProductCensus {
        round,
        products: ProductCensus { full, half },
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LaunchAccounting {
    pub(crate) dispatches: u32,
    pub(crate) barriers: u32,
    pub(crate) command_buffers: u32,
    pub(crate) host_waits: u32,
}

impl LaunchAccounting {
    fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            dispatches: self.dispatches.checked_add(other.dispatches)?,
            barriers: self.barriers.checked_add(other.barriers)?,
            command_buffers: self.command_buffers.checked_add(other.command_buffers)?,
            host_waits: self.host_waits.checked_add(other.host_waits)?,
        })
    }
}

pub(crate) const LOG26_RAW_LAUNCHES: LaunchAccounting = LaunchAccounting {
    dispatches: 44,
    barriers: 35,
    command_buffers: 9,
    host_waits: 9,
};

pub(crate) const LOG26_DENSE_LAUNCHES: LaunchAccounting = LaunchAccounting {
    dispatches: 50,
    barriers: 33,
    command_buffers: 17,
    host_waits: 17,
};

pub(crate) const LOG26_HISTOGRAM_LAUNCHES: LaunchAccounting = LaunchAccounting {
    dispatches: 4,
    barriers: 3,
    command_buffers: 1,
    host_waits: 1,
};

pub(crate) const LOG26_TOTAL_LAUNCHES: LaunchAccounting = LaunchAccounting {
    dispatches: 98,
    barriers: 71,
    command_buffers: 27,
    host_waits: 27,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TimeBudget {
    pub(crate) name: &'static str,
    pub(crate) complete_cap_ns: u64,
    pub(crate) execution_cap_at_producer_pursuit_ns: u64,
}

pub(crate) const TIME_BUDGETS: [TimeBudget; 4] = [
    budget("hard 5x", CPU_BASELINE_NS / 5),
    budget("target 6x", CPU_BASELINE_NS / 6),
    budget("stretch 7x", CPU_BASELINE_NS / 7),
    budget("stretch 8x", CPU_BASELINE_NS / 8),
];

const fn budget(name: &'static str, complete_cap_ns: u64) -> TimeBudget {
    TimeBudget {
        name,
        complete_cap_ns,
        execution_cap_at_producer_pursuit_ns: complete_cap_ns
            .saturating_sub(PRODUCER_PURSUIT_CAP_NS),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Log26ExecutionModel {
    pub(crate) raw: ExecutionWork,
    pub(crate) dense: ExecutionWork,
    pub(crate) histogram: ExecutionWork,
    pub(crate) execution: ExecutionWork,
    pub(crate) launches: LaunchAccounting,
    pub(crate) peak_logical_bytes: u64,
}

impl Log26ExecutionModel {
    pub(crate) fn checked() -> Result<Self, RegistersRwV3Error> {
        let raw_products =
            LOG26_RAW_ROUND_PRODUCTS
                .iter()
                .try_fold(ProductCensus::default(), |sum, round| {
                    sum.checked_add(round.products)
                        .ok_or(RegistersRwV3Error::SizeOverflow("raw product census"))
                })?;
        let raw_cache = raw_cache_bytes()?;
        let raw_requested = raw_cache
            .checked_add(LOG26_RAW_REQUEST_EXCESS_BYTES)
            .ok_or(RegistersRwV3Error::SizeOverflow(
                "registers RW raw requested traffic",
            ))?;
        let raw = ExecutionWork {
            products: raw_products,
            cache_unique_bytes: raw_cache,
            requested_bytes: raw_requested,
        };
        let dense = dense_work()?;
        let histogram = histogram_work()?;
        let execution = raw.checked_add(dense)?.checked_add(histogram)?;
        let launches = LOG26_RAW_LAUNCHES
            .checked_add(LOG26_DENSE_LAUNCHES)
            .and_then(|sum| sum.checked_add(LOG26_HISTOGRAM_LAUNCHES))
            .ok_or(RegistersRwV3Error::SizeOverflow(
                "registers RW launch census",
            ))?;

        if raw.products
            != (ProductCensus {
                full: 184_336_380,
                half: 1_119_975_750,
            })
            || raw.cache_unique_bytes != LOG26_RAW_CACHE_BYTES
            || raw.requested_bytes != LOG26_RAW_REQUESTED_BYTES
            || dense.cache_unique_bytes != LOG26_DENSE_CACHE_BYTES
            || dense.requested_bytes != LOG26_DENSE_REQUESTED_BYTES
            || histogram.cache_unique_bytes != LOG26_HISTOGRAM_CACHE_BYTES
            || histogram.requested_bytes != LOG26_HISTOGRAM_REQUESTED_BYTES
            || execution.products
                != (ProductCensus {
                    full: 321_518_580,
                    half: 1_119_975_750,
                })
            || execution.cache_unique_bytes != LOG26_EXECUTION_CACHE_BYTES
            || execution.requested_bytes != LOG26_EXECUTION_REQUESTED_BYTES
            || launches != LOG26_TOTAL_LAUNCHES
            || trace_peak_logical_bytes(TARGET_SHARD_LOG_T)? != LOG26_PEAK_LOGICAL_BYTES
        {
            return Err(RegistersRwV3Error::AnalyticalCensusMismatch);
        }

        Ok(Self {
            raw,
            dense,
            histogram,
            execution,
            launches,
            peak_logical_bytes: LOG26_PEAK_LOGICAL_BYTES,
        })
    }

    pub(crate) const fn projected_complete_ns() -> (u64, u64) {
        (
            PRODUCER_PURSUIT_CAP_NS + ANALYTICAL_EXECUTION_LOW_NS,
            PRODUCER_PURSUIT_CAP_NS + ANALYTICAL_EXECUTION_HIGH_NS,
        )
    }

    pub(crate) fn projected_speedup() -> (f64, f64) {
        let (low_ns, high_ns) = Self::projected_complete_ns();
        (
            CPU_BASELINE_NS as f64 / high_ns as f64,
            CPU_BASELINE_NS as f64 / low_ns as f64,
        )
    }
}

pub(crate) fn trace_peak_logical_bytes(log_t: u32) -> Result<u64, RegistersRwV3Error> {
    if !(TARGET_SHARD_LOG_T..=MAX_TRACE_LOG_T).contains(&log_t) {
        return Err(RegistersRwV3Error::InvalidExecutionLogT(log_t));
    }
    let cycles = 1u64 << log_t;
    let major_arenas = cycles.checked_mul(10_627).map(|bytes| bytes / 128).ok_or(
        RegistersRwV3Error::SizeOverflow("registers RW major arenas"),
    )?;
    let shards = 1u64 << (log_t - TARGET_SHARD_LOG_T);
    let readback =
        LOG26_TAIL_READBACK_BYTES
            .checked_mul(shards)
            .ok_or(RegistersRwV3Error::SizeOverflow(
                "registers RW sharded tail readback",
            ))?;
    major_arenas
        .checked_add(FIXED_SCRATCH_EXCLUDING_READBACK_BYTES)
        .and_then(|bytes| bytes.checked_add(readback))
        .ok_or(RegistersRwV3Error::SizeOverflow(
            "registers RW peak logical bytes",
        ))
}

fn raw_cache_bytes() -> Result<u64, RegistersRwV3Error> {
    let round_zero = LOG26_TOPOLOGY_INITIALIZED_BYTES
        + 16 * u64::from(RAW_ROUND_ZERO_INNER_LENGTH + RAW_OUTER_LENGTH)
        + reduction_traffic_bytes(6, u64::from(RAW_OUTER_LENGTH))?;
    let mut total = round_zero;
    for round in 1..=8u32 {
        let width = 1u64 << round;
        let remaining = u64::from(TARGET_SHARD_CYCLES) >> round;
        let inner = 1u64 << (12 - round);
        let mut bytes = LOG26_TOPOLOGY_INITIALIZED_BYTES
            + 48 * remaining
            + 96 * width * width
            + 64 * width
            + 16 * (inner + u64::from(RAW_OUTER_LENGTH))
            + reduction_traffic_bytes(2, u64::from(RAW_OUTER_LENGTH))?;
        if round == 8 {
            bytes += 24 * u64::from(TARGET_SHARD_CYCLES);
        }
        total = total
            .checked_add(bytes)
            .ok_or(RegistersRwV3Error::SizeOverflow("raw cache traffic"))?;
    }
    Ok(total)
}

fn dense_work() -> Result<ExecutionWork, RegistersRwV3Error> {
    let mut work = ExecutionWork::default();
    for round in 9..=25u32 {
        let destination_rows = u64::from(TARGET_SHARD_CYCLES) >> round;
        let pairs = destination_rows / 2;
        let outer = pairs.min(u64::from(MAX_DENSE_OUTER_LENGTH));
        let inner = pairs / outer;
        let products = (4 * u64::from(COLUMNS) + 3) * destination_rows + 2 * outer;
        let cache = 144 * u64::from(COLUMNS) * destination_rows
            + 48 * destination_rows
            + 16 * (inner + outer)
            + reduction_traffic_bytes(2, outer)?;
        let requested = 144 * u64::from(COLUMNS) * destination_rows
            + 48 * destination_rows
            + 16 * (pairs + outer)
            + reduction_traffic_bytes(2, outer)?;
        work = work.checked_add(ExecutionWork {
            products: ProductCensus {
                full: products,
                half: 0,
            },
            cache_unique_bytes: cache,
            requested_bytes: requested,
        })?;
    }
    Ok(work)
}

fn histogram_work() -> Result<ExecutionWork, RegistersRwV3Error> {
    let topology_stream =
        4 * (u64::from(TARGET_SHARD_CYCLES / 256) * 129) + 59_652_323 + 55_924_053;
    let cache = topology_stream
        + 16 * u64::from(HISTOGRAM_HIGH_LENGTH + HISTOGRAM_LOW_LENGTH)
        + reduction_traffic_bytes(2 * u64::from(COLUMNS), u64::from(HISTOGRAM_HIGH_LENGTH))?;
    Ok(ExecutionWork {
        products: ProductCensus {
            full: 2 * u64::from(COLUMNS) * u64::from(HISTOGRAM_HIGH_LENGTH),
            half: 0,
        },
        cache_unique_bytes: cache,
        requested_bytes: LOG26_HISTOGRAM_REQUESTED_BYTES,
    })
}

fn reduction_traffic_bytes(columns: u64, mut input_count: u64) -> Result<u64, RegistersRwV3Error> {
    let mut bytes = 16u64
        .checked_mul(columns)
        .and_then(|value| value.checked_mul(input_count))
        .ok_or(RegistersRwV3Error::SizeOverflow(
            "reduction initial output traffic",
        ))?;
    while input_count > 1 {
        let output_count = input_count.div_ceil(u64::from(SIMD_WIDTH));
        let level = input_count
            .checked_add(output_count)
            .and_then(|elements| elements.checked_mul(columns))
            .and_then(|elements| elements.checked_mul(16))
            .ok_or(RegistersRwV3Error::SizeOverflow("reduction level traffic"))?;
        bytes = bytes
            .checked_add(level)
            .ok_or(RegistersRwV3Error::SizeOverflow("reduction traffic"))?;
        input_count = output_count;
    }
    bytes
        .checked_add(16 * columns)
        .ok_or(RegistersRwV3Error::SizeOverflow(
            "reduction readback traffic",
        ))
}
