use super::abi::{
    RegisterCsrCensus, RegisterEventCounts, RegisterGeometry, RegisterPlaneLayout,
    REGISTER_LOG26_CENSUS, REGISTER_LOG26_CSR_BYTES, REGISTER_LOG26_PRODUCER_BYTES,
};
use super::RegistersRwV3Error;

pub(crate) const FROZEN_CPU_MEDIAN_SECONDS: f64 = 0.934_665_875;
pub(crate) const HOST_RESERVE_SECONDS: f64 = 0.008_756_582;
pub(crate) const ROOF_EFFICIENCY: f64 = 0.80;
pub(crate) const ROUND8_JUNCTION_CAP_SECONDS: f64 = 0.007_941_690;
pub(crate) const LOG26_PEAK_RESIDENT_BYTES: u64 = 5_276_143_172;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PhaseWork {
    full_products: u64,
    half_products: u64,
    cache_unique_bytes: u64,
    requested_bytes: u64,
}

impl PhaseWork {
    pub(crate) const fn new(
        full_products: u64,
        half_products: u64,
        cache_unique_bytes: u64,
        requested_bytes: u64,
    ) -> Self {
        Self {
            full_products,
            half_products,
            cache_unique_bytes,
            requested_bytes,
        }
    }

    pub(crate) const fn full_products(self) -> u64 {
        self.full_products
    }

    pub(crate) const fn half_products(self) -> u64 {
        self.half_products
    }

    pub(crate) const fn cache_unique_bytes(self) -> u64 {
        self.cache_unique_bytes
    }

    pub(crate) const fn requested_bytes(self) -> u64 {
        self.requested_bytes
    }

    pub(crate) fn checked_add(self, other: Self) -> Result<Self, RegistersRwV3Error> {
        Ok(Self {
            full_products: checked_add("full products", self.full_products, other.full_products)?,
            half_products: checked_add("half products", self.half_products, other.half_products)?,
            cache_unique_bytes: checked_add(
                "cache-unique bytes",
                self.cache_unique_bytes,
                other.cache_unique_bytes,
            )?,
            requested_bytes: checked_add(
                "requested bytes",
                self.requested_bytes,
                other.requested_bytes,
            )?,
        })
    }

    pub(crate) fn intensity(self) -> f64 {
        (self.full_products + self.half_products) as f64 / self.cache_unique_bytes as f64
    }
}

pub(crate) const LOG26_RAW_ROUND_0: PhaseWork =
    PhaseWork::new(167_788_547, 80_211_070, 4_461_613_324, 6_284_941_504);
pub(crate) const LOG26_RAW_ROUND_1: PhaseWork =
    PhaseWork::new(100_679_693, 100_317_099, 2_850_968_148, 4_727_590_040);
pub(crate) const LOG26_RAW_ROUNDS_2_TO_4: PhaseWork =
    PhaseWork::new(73_450_494, 413_961_160, 5_130_314_732, 11_997_976_968);
pub(crate) const LOG26_RAW_ROUNDS_5_TO_7: PhaseWork =
    PhaseWork::new(9_288_816, 364_709_709, 3_899_204_236, 9_773_005_624);
pub(crate) const LOG26_RAW_ROUND_8_JUNCTION: PhaseWork =
    PhaseWork::new(868_480, 141_656_012, 2_869_819_780, 5_140_505_912);
pub(crate) const LOG26_ROUND8_MATERIALIZATION_DELTA_PRODUCTS: u64 = 50_069_504;
pub(crate) const LOG26_RAW_SLICES: [PhaseWork; 5] = [
    LOG26_RAW_ROUND_0,
    LOG26_RAW_ROUND_1,
    LOG26_RAW_ROUNDS_2_TO_4,
    LOG26_RAW_ROUNDS_5_TO_7,
    LOG26_RAW_ROUND_8_JUNCTION,
];

pub(crate) const LOG26_RAW_TOTAL: PhaseWork =
    PhaseWork::new(352_076_030, 1_100_855_050, 19_211_920_220, 37_924_020_048);
pub(crate) const LOG26_DENSE: PhaseWork =
    PhaseWork::new(135_085_048, 0, 4_847_746_576, 4_849_843_264);
pub(crate) const LOG26_OUTPUT: PhaseWork = PhaseWork::new(262_144, 0, 395_816_832, 2_246_087_416);
pub(crate) const LOG26_EXECUTION: PhaseWork =
    PhaseWork::new(487_423_222, 1_100_855_050, 24_455_483_628, 45_019_950_728);
pub(crate) const LOG26_LIFECYCLE_CACHE_BYTES: u64 = 26_768_875_312;
pub(crate) const LOG26_LIFECYCLE_REQUESTED_BYTES: u64 = 47_333_342_412;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Log26Accounting {
    census: RegisterCsrCensus,
    layout: RegisterPlaneLayout,
    raw: PhaseWork,
    dense: PhaseWork,
    output: PhaseWork,
    execution: PhaseWork,
    lifecycle_cache_bytes: u64,
    lifecycle_requested_bytes: u64,
    peak_resident_bytes: u64,
}

impl Log26Accounting {
    pub(crate) fn checked() -> Result<Self, RegistersRwV3Error> {
        let geometry = RegisterGeometry::new(1 << 26)?;
        let census = RegisterCsrCensus::new(
            geometry,
            RegisterEventCounts::new(59_652_323, 55_924_053, 50_331_648),
        )?;
        let layout = RegisterPlaneLayout::new(census)?;
        let raw = LOG26_RAW_SLICES
            .iter()
            .copied()
            .try_fold(PhaseWork::default(), PhaseWork::checked_add)?;
        let execution = raw.checked_add(LOG26_DENSE)?.checked_add(LOG26_OUTPUT)?;
        let producer_bytes = u64::try_from(layout.producer_bytes()?)
            .map_err(|_| RegistersRwV3Error::SizeOverflow("producer byte conversion"))?;
        let lifecycle_cache_bytes = producer_bytes
            .checked_add(execution.cache_unique_bytes)
            .ok_or(RegistersRwV3Error::SizeOverflow("lifecycle cache bytes"))?;
        let lifecycle_requested_bytes = producer_bytes
            .checked_add(execution.requested_bytes)
            .ok_or(RegistersRwV3Error::SizeOverflow(
                "lifecycle requested bytes",
            ))?;
        if census != REGISTER_LOG26_CENSUS
            || census.storage_bytes()? != REGISTER_LOG26_CSR_BYTES
            || layout.producer_bytes()? as u128 != REGISTER_LOG26_PRODUCER_BYTES
            || raw != LOG26_RAW_TOTAL
            || execution != LOG26_EXECUTION
            || lifecycle_cache_bytes != LOG26_LIFECYCLE_CACHE_BYTES
            || lifecycle_requested_bytes != LOG26_LIFECYCLE_REQUESTED_BYTES
        {
            return Err(RegistersRwV3Error::AnalyticalCensusMismatch);
        }
        Ok(Self {
            census,
            layout,
            raw,
            dense: LOG26_DENSE,
            output: LOG26_OUTPUT,
            execution,
            lifecycle_cache_bytes,
            lifecycle_requested_bytes,
            peak_resident_bytes: LOG26_PEAK_RESIDENT_BYTES,
        })
    }

    pub(crate) const fn census(self) -> RegisterCsrCensus {
        self.census
    }

    pub(crate) const fn layout(self) -> RegisterPlaneLayout {
        self.layout
    }

    pub(crate) const fn raw(self) -> PhaseWork {
        self.raw
    }

    pub(crate) const fn dense(self) -> PhaseWork {
        self.dense
    }

    pub(crate) const fn output(self) -> PhaseWork {
        self.output
    }

    pub(crate) const fn execution(self) -> PhaseWork {
        self.execution
    }

    pub(crate) const fn lifecycle_cache_bytes(self) -> u64 {
        self.lifecycle_cache_bytes
    }

    pub(crate) const fn lifecycle_requested_bytes(self) -> u64 {
        self.lifecycle_requested_bytes
    }

    pub(crate) const fn peak_resident_bytes(self) -> u64 {
        self.peak_resident_bytes
    }

    pub(crate) const fn raw_slices() -> [PhaseWork; 5] {
        LOG26_RAW_SLICES
    }

    pub(crate) const fn round8_materialization_delta_products() -> u64 {
        LOG26_ROUND8_MATERIALIZATION_DELTA_PRODUCTS
    }

    pub(crate) fn project(
        self,
        rates: RoofRates,
        cpu_seconds: f64,
        efficiency: f64,
        host_reserve_seconds: f64,
    ) -> Result<LifecycleProjection, RegistersRwV3Error> {
        rates.validate()?;
        require_positive("CPU baseline", cpu_seconds)?;
        require_positive("roof efficiency", efficiency)?;
        if efficiency > 1.0 {
            return Err(RegistersRwV3Error::InvalidRoofParameter("roof efficiency"));
        }
        require_positive("host reserve", host_reserve_seconds)?;

        let producer_seconds = self.layout.producer_bytes()? as f64 / rates.copy_bytes / efficiency;
        let raw_seconds = phase_cap(self.raw, rates, efficiency);
        let dense_seconds = phase_cap(self.dense, rates, efficiency);
        let output_seconds = phase_cap(self.output, rates, efficiency);
        let cache_aware_seconds =
            producer_seconds + raw_seconds + dense_seconds + output_seconds + host_reserve_seconds;
        let requested_seconds =
            self.lifecycle_requested_bytes as f64 / rates.copy_bytes / efficiency
                + host_reserve_seconds;
        Ok(LifecycleProjection {
            cpu_seconds,
            producer_seconds,
            raw_seconds,
            dense_seconds,
            output_seconds,
            host_reserve_seconds,
            cache_aware_seconds,
            requested_seconds,
            cache_aware_gates: GateReport::new(cpu_seconds, cache_aware_seconds),
            requested_gates: GateReport::new(cpu_seconds, requested_seconds),
        })
    }

    pub(crate) fn m4_projection(self) -> Result<LifecycleProjection, RegistersRwV3Error> {
        self.project(
            M4_MAX_ROOF_RATES,
            FROZEN_CPU_MEDIAN_SECONDS,
            ROOF_EFFICIENCY,
            HOST_RESERVE_SECONDS,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RoofRates {
    copy_bytes: f64,
    full_products: f64,
    half_products: f64,
}

impl RoofRates {
    pub(crate) fn new(
        copy_bytes_per_second: f64,
        full_products_per_second: f64,
        half_products_per_second: f64,
    ) -> Result<Self, RegistersRwV3Error> {
        let rates = Self {
            copy_bytes: copy_bytes_per_second,
            full_products: full_products_per_second,
            half_products: half_products_per_second,
        };
        rates.validate()?;
        Ok(rates)
    }

    pub(crate) const fn copy_bytes_per_second(self) -> f64 {
        self.copy_bytes
    }

    pub(crate) const fn full_products_per_second(self) -> f64 {
        self.full_products
    }

    pub(crate) const fn half_products_per_second(self) -> f64 {
        self.half_products
    }

    fn validate(self) -> Result<(), RegistersRwV3Error> {
        require_positive("copy rate", self.copy_bytes)?;
        require_positive("full-product rate", self.full_products)?;
        require_positive("half-product rate", self.half_products)
    }
}

pub(crate) const M4_MAX_ROOF_RATES: RoofRates = RoofRates {
    copy_bytes: 451_701_710_520.0,
    full_products: 18_100_000_000.0,
    half_products: 26_272_000_000.0,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpeedupGate {
    Five,
    Six,
    Eight,
}

impl SpeedupGate {
    pub(crate) const fn multiplier(self) -> u8 {
        match self {
            Self::Five => 5,
            Self::Six => 6,
            Self::Eight => 8,
        }
    }

    pub(crate) fn budget_seconds(self, cpu_seconds: f64) -> f64 {
        cpu_seconds / f64::from(self.multiplier())
    }

    pub(crate) fn clears(self, cpu_seconds: f64, candidate_seconds: f64) -> bool {
        candidate_seconds <= self.budget_seconds(cpu_seconds)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GateReport {
    hard_five_x: bool,
    target_six_x: bool,
    pursue_eight_x: bool,
}

impl GateReport {
    fn new(cpu_seconds: f64, candidate_seconds: f64) -> Self {
        Self {
            hard_five_x: SpeedupGate::Five.clears(cpu_seconds, candidate_seconds),
            target_six_x: SpeedupGate::Six.clears(cpu_seconds, candidate_seconds),
            pursue_eight_x: SpeedupGate::Eight.clears(cpu_seconds, candidate_seconds),
        }
    }

    pub(crate) const fn hard_five_x(self) -> bool {
        self.hard_five_x
    }

    pub(crate) const fn target_six_x(self) -> bool {
        self.target_six_x
    }

    pub(crate) const fn pursue_eight_x(self) -> bool {
        self.pursue_eight_x
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LifecycleProjection {
    cpu_seconds: f64,
    producer_seconds: f64,
    raw_seconds: f64,
    dense_seconds: f64,
    output_seconds: f64,
    host_reserve_seconds: f64,
    cache_aware_seconds: f64,
    requested_seconds: f64,
    cache_aware_gates: GateReport,
    requested_gates: GateReport,
}

impl LifecycleProjection {
    pub(crate) const fn producer_seconds(self) -> f64 {
        self.producer_seconds
    }

    pub(crate) const fn raw_seconds(self) -> f64 {
        self.raw_seconds
    }

    pub(crate) const fn dense_seconds(self) -> f64 {
        self.dense_seconds
    }

    pub(crate) const fn output_seconds(self) -> f64 {
        self.output_seconds
    }

    pub(crate) const fn host_reserve_seconds(self) -> f64 {
        self.host_reserve_seconds
    }

    pub(crate) const fn cache_aware_seconds(self) -> f64 {
        self.cache_aware_seconds
    }

    pub(crate) const fn requested_seconds(self) -> f64 {
        self.requested_seconds
    }

    pub(crate) fn cache_aware_speedup(self) -> f64 {
        self.cpu_seconds / self.cache_aware_seconds
    }

    pub(crate) fn requested_speedup(self) -> f64 {
        self.cpu_seconds / self.requested_seconds
    }

    pub(crate) const fn cache_aware_gates(self) -> GateReport {
        self.cache_aware_gates
    }

    pub(crate) const fn requested_gates(self) -> GateReport {
        self.requested_gates
    }

    pub(crate) fn eight_x_headroom_seconds(self) -> f64 {
        SpeedupGate::Eight.budget_seconds(self.cpu_seconds) - self.cache_aware_seconds
    }

    pub(crate) fn fallback_wall_removed_fraction(self, cache_aware: bool) -> f64 {
        let candidate = if cache_aware {
            self.cache_aware_seconds
        } else {
            self.requested_seconds
        };
        1.0 - candidate / self.cpu_seconds
    }
}

fn phase_cap(work: PhaseWork, rates: RoofRates, efficiency: f64) -> f64 {
    let compute = work.full_products as f64 / rates.full_products
        + work.half_products as f64 / rates.half_products;
    let traffic = work.cache_unique_bytes as f64 / rates.copy_bytes;
    compute.max(traffic) / efficiency
}

fn checked_add(name: &'static str, left: u64, right: u64) -> Result<u64, RegistersRwV3Error> {
    left.checked_add(right)
        .ok_or(RegistersRwV3Error::SizeOverflow(name))
}

fn require_positive(name: &'static str, value: f64) -> Result<(), RegistersRwV3Error> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(RegistersRwV3Error::InvalidRoofParameter(name))
    }
}
