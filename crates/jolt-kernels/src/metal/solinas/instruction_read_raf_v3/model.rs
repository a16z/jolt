use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};

use super::abi::InstructionReadRafGeometry;
use super::{
    InstructionReadRafV3Error, ADDRESS_BINS, ADDRESS_PHASES, FP128_BYTES, INSTRUCTION_ROW_BYTES,
};

const CLAIM_BYTES: u128 = 1;
const LOOKUP_BYTES: u128 = 16;
const CYCLE_INDEX_BYTES: u128 = 4;
const OFFSET_BYTES: u128 = 4;
const JOB_BYTES: u128 = 16;
const ADDRESS_JOB_LANES: u128 = 6;
const ADDRESS_EXPANDED_LANES: u128 = 6 + 88;
const ADDRESS_ROUND_EVALS: u128 = 3;

const _: () = assert!(LookupTableKind::<RISCV_XLEN>::COUNT == 40);
const _: () = assert!(ADDRESS_EXPANDED_LANES == 94);

/// Retained controls from the M4 Max kernel campaign.  They are calibration
/// inputs, not claims about nominal Apple specifications.
pub(crate) const M4_MAX_RETAINED_RATES: RoofRates = RoofRates {
    bandwidth_bytes_per_second: 420.68 * 1024.0 * 1024.0 * 1024.0,
    useful_products_per_second: 16.42e9,
    dispatch_seconds: 25.0e-6,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SequenceWork {
    pub(crate) useful_products: u128,
    pub(crate) compulsory_bytes: u128,
    pub(crate) cache_unique_bytes: u128,
    pub(crate) requested_bytes: u128,
    pub(crate) peak_owned_bytes: u128,
    pub(crate) dispatches: u64,
}

impl SequenceWork {
    pub(crate) fn checked_add(self, other: Self) -> Result<Self, InstructionReadRafV3Error> {
        Ok(Self {
            useful_products: checked_add(
                "useful products",
                self.useful_products,
                other.useful_products,
            )?,
            compulsory_bytes: checked_add(
                "compulsory bytes",
                self.compulsory_bytes,
                other.compulsory_bytes,
            )?,
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
            peak_owned_bytes: self.peak_owned_bytes.max(other.peak_owned_bytes),
            dispatches: self
                .dispatches
                .checked_add(other.dispatches)
                .ok_or(InstructionReadRafV3Error::SizeOverflow("dispatch count"))?,
        })
    }

    pub(crate) fn arithmetic_intensity(self) -> f64 {
        self.useful_products as f64 / self.requested_bytes as f64
    }

    pub(crate) fn projected_seconds(
        self,
        rates: RoofRates,
        efficiency: f64,
    ) -> Result<f64, InstructionReadRafV3Error> {
        rates.validate()?;
        positive("roof efficiency", efficiency)?;
        if efficiency > 1.0 {
            return Err(InstructionReadRafV3Error::InvalidModelParameter(
                "roof efficiency",
            ));
        }
        let traffic = self.requested_bytes as f64 / rates.bandwidth_bytes_per_second;
        let compute = self.useful_products as f64 / rates.useful_products_per_second;
        Ok(traffic.max(compute) / efficiency + self.dispatches as f64 * rates.dispatch_seconds)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RoofRates {
    pub(crate) bandwidth_bytes_per_second: f64,
    pub(crate) useful_products_per_second: f64,
    pub(crate) dispatch_seconds: f64,
}

impl RoofRates {
    fn validate(self) -> Result<(), InstructionReadRafV3Error> {
        positive("bandwidth", self.bandwidth_bytes_per_second)?;
        positive("product rate", self.useful_products_per_second)?;
        if !self.dispatch_seconds.is_finite() || self.dispatch_seconds < 0.0 {
            return Err(InstructionReadRafV3Error::InvalidModelParameter(
                "dispatch latency",
            ));
        }
        Ok(())
    }
}

/// Measured topology census.  Scalar-product counts come from an exact walk of
/// the production rows; substituting a synthetic table mix cannot promote the
/// atom path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AddressCensus {
    pub(crate) rows: u64,
    pub(crate) atoms: u64,
    pub(crate) split_mass_partials: u64,
    pub(crate) phase_jobs: [u64; ADDRESS_PHASES],
    pub(crate) raf_scalar_products: u64,
    pub(crate) suffix_scalar_products: u64,
    pub(crate) accumulated_terms: u64,
    /// Bytes charged if the topology is built inside the measured PIOP.
    pub(crate) topology_build_bytes: u64,
    /// True only when an upstream instruction owner already emits the exact
    /// topology and its cost is charged to that owner.
    pub(crate) producer_coowned: bool,
}

impl AddressCensus {
    fn validate(
        self,
        geometry: InstructionReadRafGeometry,
        topology_required: bool,
    ) -> Result<(), InstructionReadRafV3Error> {
        if self.rows != geometry.cycles() as u64 {
            return Err(InstructionReadRafV3Error::InvalidCensus(
                "row count differs from relation geometry",
            ));
        }
        if self.atoms == 0 || self.atoms > self.rows {
            return Err(InstructionReadRafV3Error::InvalidCensus(
                "atom count is outside 1..=rows",
            ));
        }
        if self.phase_jobs.contains(&0) {
            return Err(InstructionReadRafV3Error::InvalidCensus(
                "every address phase needs at least one scan job",
            ));
        }
        if self.split_mass_partials > self.rows {
            return Err(InstructionReadRafV3Error::InvalidCensus(
                "split mass partials exceed rows",
            ));
        }
        if topology_required && !self.producer_coowned && self.topology_build_bytes == 0 {
            return Err(InstructionReadRafV3Error::InvalidCensus(
                "member-local topology construction must be charged",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AddressPath {
    DenseRows,
    CompressedAtoms,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExecutionModel {
    pub(crate) address_path: AddressPath,
    pub(crate) address: SequenceWork,
    pub(crate) cycle: SequenceWork,
    pub(crate) topology_build: SequenceWork,
    pub(crate) total: SequenceWork,
}

impl ExecutionModel {
    pub(crate) fn compressed(
        geometry: InstructionReadRafGeometry,
        census: AddressCensus,
        cutoff_elements: usize,
    ) -> Result<Self, InstructionReadRafV3Error> {
        census.validate(geometry, true)?;
        let address = compressed_address_work(geometry, census)?;
        let cycle = cycle_work(geometry, cutoff_elements)?;
        let topology_build = if census.producer_coowned {
            SequenceWork::default()
        } else {
            SequenceWork {
                compulsory_bytes: census.topology_build_bytes as u128,
                cache_unique_bytes: census.topology_build_bytes as u128,
                requested_bytes: census.topology_build_bytes as u128,
                peak_owned_bytes: census.topology_build_bytes as u128,
                dispatches: 1,
                ..SequenceWork::default()
            }
        };
        let total = address.checked_add(cycle)?.checked_add(topology_build)?;
        Ok(Self {
            address_path: AddressPath::CompressedAtoms,
            address,
            cycle,
            topology_build,
            total,
        })
    }

    pub(crate) fn dense(
        geometry: InstructionReadRafGeometry,
        census: AddressCensus,
        cutoff_elements: usize,
    ) -> Result<Self, InstructionReadRafV3Error> {
        census.validate(geometry, false)?;
        let address = dense_address_work(geometry, census)?;
        let cycle = cycle_work(geometry, cutoff_elements)?;
        let total = address.checked_add(cycle)?;
        Ok(Self {
            address_path: AddressPath::DenseRows,
            address,
            cycle,
            topology_build: SequenceWork::default(),
            total,
        })
    }

    pub(crate) fn gate(
        self,
        cpu_seconds: f64,
        host_reserve_seconds: f64,
        rates: RoofRates,
        efficiency: f64,
        minimum_speedup: f64,
    ) -> Result<GateReport, InstructionReadRafV3Error> {
        positive("CPU baseline", cpu_seconds)?;
        positive("minimum speedup", minimum_speedup)?;
        if !host_reserve_seconds.is_finite() || host_reserve_seconds < 0.0 {
            return Err(InstructionReadRafV3Error::InvalidModelParameter(
                "host reserve",
            ));
        }
        let device_seconds = self.total.projected_seconds(rates, efficiency)?;
        let projected_seconds = device_seconds + host_reserve_seconds;
        let speedup = cpu_seconds / projected_seconds;
        let target_seconds = cpu_seconds / minimum_speedup;
        Ok(GateReport {
            device_seconds,
            host_reserve_seconds,
            projected_seconds,
            target_seconds,
            speedup,
            passes: projected_seconds <= target_seconds,
            pursue_headroom: speedup >= minimum_speedup * 1.15,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GateReport {
    pub(crate) device_seconds: f64,
    pub(crate) host_reserve_seconds: f64,
    pub(crate) projected_seconds: f64,
    pub(crate) target_seconds: f64,
    pub(crate) speedup: f64,
    pub(crate) passes: bool,
    pub(crate) pursue_headroom: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CutoffDecision {
    pub(crate) cutoff_elements: usize,
    pub(crate) first_cpu_round_seconds: f64,
    pub(crate) last_gpu_round_seconds: f64,
}

/// Chooses the first width at which a resident Product5 transition loses to
/// the optimized CPU tail.  The comparison includes one dispatch at every GPU
/// width and uses the exact `80n read + 40n write` factor traffic.
pub(crate) fn choose_cycle_cutoff(
    initial_width: usize,
    minimum_width: usize,
    rates: RoofRates,
    efficiency: f64,
    cpu_seconds_per_element: f64,
) -> Result<CutoffDecision, InstructionReadRafV3Error> {
    rates.validate()?;
    positive("roof efficiency", efficiency)?;
    positive("CPU seconds per element", cpu_seconds_per_element)?;
    if initial_width == 0
        || minimum_width == 0
        || !initial_width.is_power_of_two()
        || !minimum_width.is_power_of_two()
        || minimum_width > initial_width
    {
        return Err(InstructionReadRafV3Error::InvalidModelParameter(
            "cycle cutoff geometry",
        ));
    }
    let mut width = initial_width;
    let mut last_gpu_round_seconds = 0.0;
    loop {
        let gpu_work = product5_transition_work(width as u128, 5)?;
        let gpu_seconds = gpu_work.projected_seconds(rates, efficiency)?;
        let cpu_seconds = width as f64 * cpu_seconds_per_element;
        if gpu_seconds >= cpu_seconds || width == minimum_width {
            return Ok(CutoffDecision {
                cutoff_elements: width,
                first_cpu_round_seconds: cpu_seconds,
                last_gpu_round_seconds,
            });
        }
        last_gpu_round_seconds = gpu_seconds;
        width /= 2;
    }
}

fn compressed_address_work(
    geometry: InstructionReadRafGeometry,
    census: AddressCensus,
) -> Result<SequenceWork, InstructionReadRafV3Error> {
    let rows = census.rows as u128;
    let atoms = census.atoms as u128;
    let split = census.split_mass_partials as u128;
    let jobs: u128 = census.phase_jobs.iter().map(|value| *value as u128).sum();
    let useful_products = checked_sum(&[
        rows,
        15 * atoms,
        census.raf_scalar_products as u128,
        census.suffix_scalar_products as u128,
    ])?;

    // Phase zero reads the cycle permutation and two split-eq factors, then
    // writes one mass per atom.  Later phases request a raw key and update one
    // mass in place.  Split mass partials are written and read once.
    let phase_zero = checked_sum(&[
        CYCLE_INDEX_BYTES * rows,
        2 * FP128_BYTES as u128 * rows,
        (JOB_BYTES + LOOKUP_BYTES) * census.phase_jobs[0] as u128,
        FP128_BYTES as u128 * atoms,
        2 * FP128_BYTES as u128 * split,
    ])?;
    let later_phases = 15 * (LOOKUP_BYTES + 2 * FP128_BYTES as u128) * atoms;
    let common = address_partial_and_host_bytes(jobs)?;
    let requested_bytes = checked_sum(&[phase_zero, later_phases, common])?;
    let topology = checked_sum(&[
        CYCLE_INDEX_BYTES * rows,
        OFFSET_BYTES * (atoms + 1),
        (LOOKUP_BYTES + CLAIM_BYTES) * atoms,
        OFFSET_BYTES * super::abi::ADDRESS_SEGMENT_OFFSETS as u128,
    ])?;
    let partial_peak = largest_partial_bytes(&census.phase_jobs)?;
    let output_bytes = ADDRESS_EXPANDED_LANES * ADDRESS_BINS as u128 * FP128_BYTES as u128;
    let compulsory_bytes = checked_sum(&[
        topology,
        FP128_BYTES as u128 * atoms,
        FP128_BYTES as u128 * split,
        partial_peak,
        output_bytes,
    ])?;
    let equality_cache = split_eq_cache_bytes(geometry)?;
    let phase_tables = ADDRESS_PHASES as u128 * ADDRESS_BINS as u128 * FP128_BYTES as u128;
    Ok(SequenceWork {
        useful_products,
        compulsory_bytes,
        cache_unique_bytes: checked_sum(&[compulsory_bytes, equality_cache, phase_tables])?,
        requested_bytes,
        peak_owned_bytes: checked_sum(&[
            topology,
            FP128_BYTES as u128 * atoms,
            FP128_BYTES as u128 * split,
            partial_peak,
            output_bytes,
            phase_tables,
        ])?,
        dispatches: u64::try_from(2 * ADDRESS_PHASES)
            .map_err(|_| InstructionReadRafV3Error::SizeOverflow("address dispatch count"))?,
    })
}

fn dense_address_work(
    geometry: InstructionReadRafGeometry,
    census: AddressCensus,
) -> Result<SequenceWork, InstructionReadRafV3Error> {
    let rows = census.rows as u128;
    let jobs: u128 = census.phase_jobs.iter().map(|value| *value as u128).sum();
    let useful_products = checked_sum(&[
        16 * rows,
        census.raf_scalar_products as u128,
        census.suffix_scalar_products as u128,
    ])?;
    // Only lookup limbs and the packed selector word are logically read from
    // each 40-byte producer row.  Phase zero also requests two split-eq
    // factors and writes the mutable weight.  Fifteen phases read 24 row bytes
    // and read/write the 16-byte weight.
    let phase_zero = (24 + 2 * FP128_BYTES as u128 + FP128_BYTES as u128) * rows;
    let later_phases = 15 * (24 + 2 * FP128_BYTES as u128) * rows;
    let common = address_partial_and_host_bytes(jobs)?;
    let requested_bytes = checked_sum(&[phase_zero, later_phases, common])?;
    let partial_peak = largest_partial_bytes(&census.phase_jobs)?;
    let output_bytes = ADDRESS_EXPANDED_LANES * ADDRESS_BINS as u128 * FP128_BYTES as u128;
    let compulsory_bytes = checked_sum(&[
        INSTRUCTION_ROW_BYTES as u128 * rows,
        FP128_BYTES as u128 * rows,
        partial_peak,
        output_bytes,
    ])?;
    let equality_cache = split_eq_cache_bytes(geometry)?;
    let phase_tables = ADDRESS_PHASES as u128 * ADDRESS_BINS as u128 * FP128_BYTES as u128;
    Ok(SequenceWork {
        useful_products,
        compulsory_bytes,
        cache_unique_bytes: checked_sum(&[compulsory_bytes, equality_cache, phase_tables])?,
        requested_bytes,
        peak_owned_bytes: checked_sum(&[
            FP128_BYTES as u128 * rows,
            partial_peak,
            output_bytes,
            phase_tables,
        ])?,
        dispatches: u64::try_from(2 * ADDRESS_PHASES)
            .map_err(|_| InstructionReadRafV3Error::SizeOverflow("address dispatch count"))?,
    })
}

fn address_partial_and_host_bytes(jobs: u128) -> Result<u128, InstructionReadRafV3Error> {
    let partial_fields = ADDRESS_JOB_LANES * ADDRESS_BINS as u128 * jobs;
    let partial_write_read = 2 * partial_fields * FP128_BYTES as u128;
    let expanded_output = ADDRESS_EXPANDED_LANES * ADDRESS_BINS as u128 * FP128_BYTES as u128;
    let output_write_and_host_read = 2 * ADDRESS_PHASES as u128 * expanded_output;
    let phase_table_uploads = ADDRESS_PHASES as u128 * ADDRESS_BINS as u128 * FP128_BYTES as u128;
    let round_message_readbacks = ADDRESS_BITS_U128 * ADDRESS_ROUND_EVALS * FP128_BYTES as u128;
    let challenge_uploads = ADDRESS_BITS_U128 * FP128_BYTES as u128;
    checked_sum(&[
        partial_write_read,
        output_write_and_host_read,
        phase_table_uploads,
        round_message_readbacks,
        challenge_uploads,
    ])
}

const ADDRESS_BITS_U128: u128 = super::ADDRESS_BITS as u128;

fn cycle_work(
    geometry: InstructionReadRafGeometry,
    cutoff_elements: usize,
) -> Result<SequenceWork, InstructionReadRafV3Error> {
    let cycles = geometry.cycles();
    if cutoff_elements == 0 || !cutoff_elements.is_power_of_two() || cutoff_elements >= cycles {
        return Err(InstructionReadRafV3Error::InvalidModelParameter(
            "cycle cutoff",
        ));
    }
    let factors = geometry.cycle_factors();
    let cycles_u128 = cycles as u128;
    let factor_bytes = factors as u128 * FP128_BYTES as u128;

    // First message reads the producer row and split equality factors without
    // writing full-width factor tables.  After the host challenge, the fused
    // handoff rereads the source, writes the half-width factors, and computes
    // the following message before those values leave registers.
    let first_message_bytes = (24 + 2 * FP128_BYTES as u128) * cycles_u128;
    let handoff_bytes = 24 * cycles_u128 + factor_bytes * (cycles_u128 / 2);
    let mut requested_bytes = checked_sum(&[first_message_bytes, handoff_bytes])?;
    let mut useful_products = cycle_message_products(cycles_u128 / 2, factors)?;
    useful_products = checked_add(
        "cycle useful products",
        useful_products,
        factors as u128 * cycles_u128 / 2,
    )?;
    useful_products = checked_add(
        "cycle useful products",
        useful_products,
        cycle_message_products(cycles_u128 / 4, factors)?,
    )?;

    let mut width = cycles / 2;
    let mut dispatches = 2u64;
    while width > cutoff_elements {
        let transition = product5_transition_work(width as u128, factors)?;
        requested_bytes = checked_add(
            "cycle requested bytes",
            requested_bytes,
            transition.requested_bytes,
        )?;
        useful_products = checked_add(
            "cycle useful products",
            useful_products,
            transition.useful_products,
        )?;
        dispatches = dispatches
            .checked_add(1)
            .ok_or(InstructionReadRafV3Error::SizeOverflow(
                "cycle dispatch count",
            ))?;
        width /= 2;
    }
    let readback_bytes = factor_bytes * width as u128;

    // Terminal flags read the co-produced byte plane.  Equality-factor
    // requests are logical traffic; the split tables are cache-unique.
    let flag_bytes = checked_sum(&[
        CLAIM_BYTES * cycles_u128,
        2 * FP128_BYTES as u128 * cycles_u128,
        2 * (LookupTableKind::<RISCV_XLEN>::COUNT as u128 + 1)
            * FP128_BYTES as u128
            * (1u128 << (geometry.log_t() / 2)),
    ])?;
    requested_bytes = checked_sum(&[requested_bytes, readback_bytes, flag_bytes])?;
    useful_products = checked_add("cycle useful products", useful_products, cycles_u128)?;
    dispatches = dispatches
        .checked_add(4)
        .ok_or(InstructionReadRafV3Error::SizeOverflow(
            "flag dispatch count",
        ))?;

    let half_factor_arena = factor_bytes * (cycles_u128 / 2);
    let flag_partials = (LookupTableKind::<RISCV_XLEN>::COUNT as u128 + 1)
        * FP128_BYTES as u128
        * (1u128 << (geometry.log_t() / 2));
    let compulsory_bytes = checked_sum(&[
        INSTRUCTION_ROW_BYTES as u128 * cycles_u128,
        CLAIM_BYTES * cycles_u128,
        half_factor_arena,
        flag_partials,
        split_eq_cache_bytes(geometry)?,
    ])?;
    Ok(SequenceWork {
        useful_products,
        compulsory_bytes,
        cache_unique_bytes: compulsory_bytes,
        requested_bytes,
        peak_owned_bytes: checked_sum(&[half_factor_arena, flag_partials])?,
        dispatches,
    })
}

fn product5_transition_work(
    width: u128,
    factors: usize,
) -> Result<SequenceWork, InstructionReadRafV3Error> {
    if width < 2 || !width.is_power_of_two() {
        return Err(InstructionReadRafV3Error::InvalidModelParameter(
            "Product5 transition width",
        ));
    }
    let factors = factors as u128;
    let pairs = width / 2;
    let requested_bytes = factors * FP128_BYTES as u128 * (width + pairs);
    let useful_products = checked_add(
        "Product5 transition products",
        factors * pairs,
        cycle_message_products(pairs / 2, factors as usize)?,
    )?;
    Ok(SequenceWork {
        useful_products,
        compulsory_bytes: requested_bytes,
        cache_unique_bytes: requested_bytes,
        requested_bytes,
        peak_owned_bytes: factors * FP128_BYTES as u128 * pairs,
        dispatches: 1,
    })
}

/// Product of `factors` linear factors at `factors` Gruen q-grid points, plus
/// two products per pair that fold `E_in` into the combined-value endpoints.
fn cycle_message_products(pairs: u128, factors: usize) -> Result<u128, InstructionReadRafV3Error> {
    if factors < 2 {
        return Err(InstructionReadRafV3Error::InvalidVirtualRa(
            factors.saturating_sub(1),
        ));
    }
    checked_add(
        "cycle message products",
        2 * pairs,
        factors as u128 * (factors as u128 - 1) * pairs,
    )
}

fn split_eq_cache_bytes(
    geometry: InstructionReadRafGeometry,
) -> Result<u128, InstructionReadRafV3Error> {
    let in_len = 1u128 << (geometry.log_t() / 2);
    let out_len = 1u128 << (geometry.log_t() - geometry.log_t() / 2);
    checked_add(
        "split equality bytes",
        in_len * FP128_BYTES as u128,
        out_len * FP128_BYTES as u128,
    )
}

fn largest_partial_bytes(
    phase_jobs: &[u64; ADDRESS_PHASES],
) -> Result<u128, InstructionReadRafV3Error> {
    let jobs = phase_jobs.iter().copied().max().unwrap_or(0) as u128;
    ADDRESS_JOB_LANES
        .checked_mul(ADDRESS_BINS as u128)
        .and_then(|value| value.checked_mul(FP128_BYTES as u128))
        .and_then(|value| value.checked_mul(jobs))
        .ok_or(InstructionReadRafV3Error::SizeOverflow(
            "address partial workspace",
        ))
}

fn positive(name: &'static str, value: f64) -> Result<(), InstructionReadRafV3Error> {
    if !value.is_finite() || value <= 0.0 {
        return Err(InstructionReadRafV3Error::InvalidModelParameter(name));
    }
    Ok(())
}

fn checked_add(
    name: &'static str,
    left: u128,
    right: u128,
) -> Result<u128, InstructionReadRafV3Error> {
    left.checked_add(right)
        .ok_or(InstructionReadRafV3Error::SizeOverflow(name))
}

fn checked_sum(values: &[u128]) -> Result<u128, InstructionReadRafV3Error> {
    values.iter().try_fold(0u128, |sum, value| {
        checked_add("analytical sum", sum, *value)
    })
}
