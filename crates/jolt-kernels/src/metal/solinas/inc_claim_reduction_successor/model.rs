//! Exact log-26 accounting for the increment claim-reduction successor.

pub const FIELD_BYTES: u128 = 16;
pub const ROW_BYTES: u128 = 16;
pub const SIMD_WIDTH: usize = 32;
pub const M4_MAX_GPU_CORES: usize = 40;
pub const M4_MAX_COPY_BYTES_PER_SECOND: f64 = 451_701_710_520.0;
pub const SIGNED_HALF_PRODUCTS_PER_SECOND: f64 = 26_272_000_000.0;
pub const PROMOTION_FRACTION: f64 = 0.8;

pub const FUSED_OWNER_CPU_SAMPLES_NS: [u64; 5] = [
    1_156_804_791,
    1_254_933_123,
    1_168_671_791,
    1_206_041_790,
    1_203_638_208,
];

pub const FUSED_OWNER_METAL_SAMPLES_NS: [u64; 5] = [
    350_824_336,
    363_178_339,
    349_465_502,
    348_842_671,
    341_891_626,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitGeometry {
    pub log_t: u32,
    pub prefix_bits: u32,
    pub suffix_bits: u32,
    pub rows: usize,
    pub prefix_elements: usize,
    pub suffix_elements: usize,
    pub q_partitions: usize,
}

impl SplitGeometry {
    pub fn balanced(log_t: u32, q_partitions: usize) -> Self {
        assert!(log_t >= 2 && log_t < usize::BITS);
        assert!(q_partitions.is_power_of_two());
        let prefix_bits = log_t / 2;
        let suffix_bits = log_t - prefix_bits;
        let prefix_elements = 1usize << prefix_bits;
        let suffix_elements = 1usize << suffix_bits;
        assert!(suffix_elements.is_multiple_of(q_partitions));
        assert!(prefix_elements.is_multiple_of(SIMD_WIDTH));
        Self {
            log_t,
            prefix_bits,
            suffix_bits,
            rows: 1usize << log_t,
            prefix_elements,
            suffix_elements,
            q_partitions,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Topology {
    pub nonzero_rows: usize,
    pub ram_active_high_indices: usize,
    pub rd_active_high_indices: usize,
    pub active_low_indices: usize,
    pub ram_simd_iterations: usize,
    pub rd_simd_iterations: usize,
    pub union_simd_iterations: usize,
}

impl Topology {
    pub fn dense_homogeneous(geometry: SplitGeometry) -> Self {
        let simd_iterations = geometry.rows / SIMD_WIDTH;
        Self {
            nonzero_rows: geometry.rows,
            ram_active_high_indices: geometry.suffix_elements,
            rd_active_high_indices: 0,
            active_low_indices: geometry.prefix_elements,
            ram_simd_iterations: simd_iterations,
            rd_simd_iterations: 0,
            union_simd_iterations: simd_iterations,
        }
    }

    pub fn dense_maximally_mixed(geometry: SplitGeometry) -> Self {
        let simd_iterations = geometry.rows / SIMD_WIDTH;
        Self {
            nonzero_rows: geometry.rows,
            ram_active_high_indices: geometry.suffix_elements,
            rd_active_high_indices: geometry.suffix_elements,
            active_low_indices: geometry.prefix_elements,
            ram_simd_iterations: simd_iterations,
            rd_simd_iterations: simd_iterations,
            union_simd_iterations: simd_iterations,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitWork {
    pub useful_q_products: u128,
    pub useful_fold_products: u128,
    pub issued_q_products: u128,
    pub issued_fold_products: u128,
    pub q_streaming_bytes: u128,
    pub q_cache_unique_bytes: u128,
    pub q_requested_bytes: u128,
    pub fold_streaming_bytes: u128,
    pub fold_cache_unique_bytes: u128,
    pub fold_requested_bytes: u128,
    pub readback_bytes: u128,
}

impl SplitWork {
    pub fn new(geometry: SplitGeometry, topology: Topology) -> Self {
        let rows = geometry.rows as u128;
        let prefix = geometry.prefix_elements as u128;
        let suffix = geometry.suffix_elements as u128;
        let partitions = geometry.q_partitions as u128;
        let q_streaming_bytes = ROW_BYTES * rows + 128 * partitions * prefix + 64 * prefix;
        let q_cache_lookup_bytes =
            32 * (topology.ram_active_high_indices + topology.rd_active_high_indices) as u128;
        let q_requested_lookup_bytes =
            32 * (topology.ram_simd_iterations + topology.rd_simd_iterations) as u128;
        let fold_streaming_bytes = ROW_BYTES * rows + 32 * suffix;
        let fold_cache_lookup_bytes = 16 * topology.active_low_indices as u128;
        let fold_requested_lookup_bytes = 16 * topology.nonzero_rows as u128;
        Self {
            useful_q_products: 2 * topology.nonzero_rows as u128,
            useful_fold_products: topology.nonzero_rows as u128,
            issued_q_products: 64
                * (topology.ram_simd_iterations + topology.rd_simd_iterations) as u128,
            issued_fold_products: 32 * topology.union_simd_iterations as u128,
            q_streaming_bytes,
            q_cache_unique_bytes: q_streaming_bytes + q_cache_lookup_bytes,
            q_requested_bytes: q_streaming_bytes + q_requested_lookup_bytes,
            fold_streaming_bytes,
            fold_cache_unique_bytes: fold_streaming_bytes + fold_cache_lookup_bytes,
            fold_requested_bytes: fold_streaming_bytes + fold_requested_lookup_bytes,
            readback_bytes: 64 * prefix + 32 * suffix,
        }
    }

    pub fn perfect_bytes(self) -> u128 {
        self.q_streaming_bytes + self.fold_streaming_bytes
    }

    pub fn cache_unique_bytes(self) -> u128 {
        self.q_cache_unique_bytes + self.fold_cache_unique_bytes
    }

    pub fn requested_bytes(self) -> u128 {
        self.q_requested_bytes + self.fold_requested_bytes
    }

    pub fn q_incremental_bytes_when_row_read_is_shared(self, geometry: SplitGeometry) -> u128 {
        self.q_streaming_bytes - ROW_BYTES * geometry.rows as u128
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseBound {
    pub compute_ms: f64,
    pub traffic_ms: f64,
    pub floor_ms: f64,
    pub promotion_gate_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitBounds {
    pub q_perfect: PhaseBound,
    pub fold_perfect: PhaseBound,
    pub q_cache_unique: PhaseBound,
    pub fold_cache_unique: PhaseBound,
    pub q_requested: PhaseBound,
    pub fold_requested: PhaseBound,
}

impl SplitBounds {
    pub fn new(work: SplitWork) -> Self {
        Self {
            q_perfect: phase_bound(work.issued_q_products, work.q_streaming_bytes),
            fold_perfect: phase_bound(work.issued_fold_products, work.fold_streaming_bytes),
            q_cache_unique: phase_bound(work.issued_q_products, work.q_cache_unique_bytes),
            fold_cache_unique: phase_bound(work.issued_fold_products, work.fold_cache_unique_bytes),
            q_requested: phase_bound(work.issued_q_products, work.q_requested_bytes),
            fold_requested: phase_bound(work.issued_fold_products, work.fold_requested_bytes),
        }
    }

    pub fn perfect_gate_ms(self) -> f64 {
        self.q_perfect.promotion_gate_ms + self.fold_perfect.promotion_gate_ms
    }

    pub fn cache_unique_gate_ms(self) -> f64 {
        self.q_cache_unique.promotion_gate_ms + self.fold_cache_unique.promotion_gate_ms
    }

    pub fn requested_gate_ms(self) -> f64 {
        self.q_requested.promotion_gate_ms + self.fold_requested.promotion_gate_ms
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoragePlan {
    pub borrowed_rows: u128,
    pub suffix_equalities: u128,
    pub q_partials: u128,
    pub q_outputs: u128,
    pub prefix_equality: u128,
    pub dense_outputs: u128,
    pub counters: u128,
}

impl StoragePlan {
    pub fn new(geometry: SplitGeometry) -> Self {
        let prefix = geometry.prefix_elements as u128;
        let suffix = geometry.suffix_elements as u128;
        Self {
            borrowed_rows: ROW_BYTES * geometry.rows as u128,
            suffix_equalities: 4 * FIELD_BYTES * suffix,
            q_partials: 4 * geometry.q_partitions as u128 * FIELD_BYTES * prefix,
            q_outputs: 4 * FIELD_BYTES * prefix,
            prefix_equality: FIELD_BYTES * prefix,
            dense_outputs: 2 * FIELD_BYTES * suffix,
            counters: 32,
        }
    }

    pub fn sequence_owned(self) -> u128 {
        self.suffix_equalities
            + self.q_partials
            + self.q_outputs
            + self.prefix_equality
            + self.dense_outputs
            + self.counters
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaunchSupply {
    pub q_groups: usize,
    pub q_reduce_groups: usize,
    pub fold_groups: usize,
    pub q_groups_per_core: f64,
    pub q_reduce_groups_per_core: f64,
    pub fold_groups_per_core: f64,
    pub q_accumulator_words_per_lane: usize,
    pub fold_accumulator_words_per_lane: usize,
}

impl LaunchSupply {
    pub fn new(geometry: SplitGeometry) -> Self {
        let q_groups = geometry.q_partitions * geometry.prefix_elements / SIMD_WIDTH;
        let q_reduce_groups = geometry.prefix_elements / SIMD_WIDTH;
        let fold_groups = geometry.suffix_elements;
        Self {
            q_groups,
            q_reduce_groups,
            fold_groups,
            q_groups_per_core: q_groups as f64 / M4_MAX_GPU_CORES as f64,
            q_reduce_groups_per_core: q_reduce_groups as f64 / M4_MAX_GPU_CORES as f64,
            fold_groups_per_core: fold_groups as f64 / M4_MAX_GPU_CORES as f64,
            q_accumulator_words_per_lane: 16,
            fold_accumulator_words_per_lane: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FactoredRoundService {
    pub compulsory_bytes_through_midpoint: u128,
    pub command_completions_through_first_suffix_message: usize,
}

impl FactoredRoundService {
    pub fn alias_aware(geometry: SplitGeometry) -> Self {
        let rows = geometry.rows as u128;
        let suffix = geometry.suffix_elements as u128;
        Self {
            compulsory_bytes_through_midpoint: 96 * rows - 96 * suffix,
            command_completions_through_first_suffix_message: geometry.prefix_bits as usize + 1,
        }
    }

    pub fn traffic_floor_ms(self) -> f64 {
        milliseconds(
            self.compulsory_bytes_through_midpoint,
            M4_MAX_COPY_BYTES_PER_SECOND,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OwnerBars {
    pub cpu_median_ns: u64,
    pub five_x_cap_ns: u64,
    pub eight_x_cap_ns: u64,
    pub current_metal_median_ns: u64,
}

impl OwnerBars {
    pub fn frozen() -> Self {
        let cpu_median_ns = median_five(FUSED_OWNER_CPU_SAMPLES_NS);
        Self {
            cpu_median_ns,
            five_x_cap_ns: cpu_median_ns / 5,
            eight_x_cap_ns: cpu_median_ns / 8,
            current_metal_median_ns: median_five(FUSED_OWNER_METAL_SAMPLES_NS),
        }
    }
}

fn phase_bound(products: u128, bytes: u128) -> PhaseBound {
    let compute_ms = milliseconds(products, SIGNED_HALF_PRODUCTS_PER_SECOND);
    let traffic_ms = milliseconds(bytes, M4_MAX_COPY_BYTES_PER_SECOND);
    let floor_ms = compute_ms.max(traffic_ms);
    PhaseBound {
        compute_ms,
        traffic_ms,
        floor_ms,
        promotion_gate_ms: floor_ms / PROMOTION_FRACTION,
    }
}

fn milliseconds(amount: u128, rate_per_second: f64) -> f64 {
    amount as f64 * 1_000.0 / rate_per_second
}

fn median_five(mut samples: [u64; 5]) -> u64 {
    samples.sort_unstable();
    samples[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) {
        assert!((left - right).abs() < 0.001, "{left} != {right}");
    }

    #[test]
    fn log26_split_counts_are_exact() {
        let geometry = SplitGeometry::balanced(26, 8);
        let work = SplitWork::new(geometry, Topology::dense_homogeneous(geometry));
        assert_eq!(geometry.prefix_elements, 8192);
        assert_eq!(geometry.suffix_elements, 8192);
        assert_eq!(work.useful_q_products, 134_217_728);
        assert_eq!(work.useful_fold_products, 67_108_864);
        assert_eq!(work.q_streaming_bytes, 1_082_654_720);
        assert_eq!(work.fold_streaming_bytes, 1_074_003_968);
        assert_eq!(work.perfect_bytes(), 2_156_658_688);
        assert_eq!(work.cache_unique_bytes(), 2_157_051_904);
        assert_eq!(work.requested_bytes(), 3_297_509_376);
        assert_eq!(work.readback_bytes, 786_432);
        assert_eq!(
            work.q_incremental_bytes_when_row_read_is_shared(geometry),
            8_912_896
        );
    }

    #[test]
    fn dense_selector_mix_changes_issue_not_useful_work() {
        let geometry = SplitGeometry::balanced(26, 8);
        let homogeneous = SplitWork::new(geometry, Topology::dense_homogeneous(geometry));
        let mixed = SplitWork::new(geometry, Topology::dense_maximally_mixed(geometry));
        assert_eq!(homogeneous.useful_q_products, mixed.useful_q_products);
        assert_eq!(homogeneous.useful_fold_products, mixed.useful_fold_products);
        assert_eq!(mixed.issued_q_products, 2 * homogeneous.issued_q_products);
        assert_eq!(mixed.issued_fold_products, homogeneous.issued_fold_products);
        assert_eq!(mixed.requested_bytes(), 3_364_618_240);
    }

    #[test]
    fn log26_roofs_pin_the_pursuit_bars() {
        let geometry = SplitGeometry::balanced(26, 8);
        let homogeneous = SplitBounds::new(SplitWork::new(
            geometry,
            Topology::dense_homogeneous(geometry),
        ));
        let mixed = SplitBounds::new(SplitWork::new(
            geometry,
            Topology::dense_maximally_mixed(geometry),
        ));
        close(homogeneous.perfect_gate_ms(), 9.579);
        close(homogeneous.cache_unique_gate_ms(), 9.579);
        close(homogeneous.requested_gate_ms(), 12.329);
        close(mixed.cache_unique_gate_ms(), 15.965);
        close(mixed.requested_gate_ms(), 18.715);
    }

    #[test]
    fn precommitted_style_round_service_loses_to_two_scans_on_traffic_alone() {
        let geometry = SplitGeometry::balanced(26, 8);
        let split = SplitWork::new(geometry, Topology::dense_homogeneous(geometry));
        let factored = FactoredRoundService::alias_aware(geometry);
        assert_eq!(factored.compulsory_bytes_through_midpoint, 6_441_664_512);
        assert_eq!(
            factored.command_completions_through_first_suffix_message,
            14
        );
        assert!(factored.compulsory_bytes_through_midpoint > 2 * split.perfect_bytes());
        close(factored.traffic_floor_ms(), 14.261);
    }

    #[test]
    fn launch_supply_covers_every_gpu_core() {
        let launch = LaunchSupply::new(SplitGeometry::balanced(26, 8));
        assert_eq!(launch.q_groups, 2048);
        assert_eq!(launch.q_reduce_groups, 256);
        assert_eq!(launch.fold_groups, 8192);
        close(launch.q_groups_per_core, 51.2);
        close(launch.q_reduce_groups_per_core, 6.4);
        close(launch.fold_groups_per_core, 204.8);
    }

    #[test]
    fn storage_and_fused_owner_bars_are_frozen() {
        let storage = StoragePlan::new(SplitGeometry::balanced(26, 8));
        assert_eq!(storage.borrowed_rows, 1_073_741_824);
        assert_eq!(storage.sequence_owned(), 5_636_128);
        let bars = OwnerBars::frozen();
        assert_eq!(bars.cpu_median_ns, 1_203_638_208);
        assert_eq!(bars.five_x_cap_ns, 240_727_641);
        assert_eq!(bars.eight_x_cap_ns, 150_454_776);
        assert_eq!(bars.current_metal_median_ns, 349_465_502);
    }
}
