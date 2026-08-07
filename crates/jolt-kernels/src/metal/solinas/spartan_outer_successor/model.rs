pub const LOG_T: usize = 26;
pub const CYCLES: u64 = 1 << LOG_T;

pub const COPY_BYTES_PER_SECOND: f64 = 420.68 * (1u64 << 30) as f64;
pub const FIELD_PRODUCTS_PER_SECOND: f64 = 24.08e9;
pub const UNISKIP_FIELD_PRODUCTS_PER_SECOND: f64 = 16.42e9;
pub const U32_MADDS_PER_SECOND: f64 = 958.0e9;

pub const CPU_UNISKIP_NS: u64 = 2_722_459_375;
pub const CPU_REMAINDER_NS: u64 = 1_093_453_625;
pub const CPU_STAGE_NS: u64 = 3_816_080_708;
pub const METAL_UNISKIP_NS: u64 = 307_663_833;
pub const METAL_REMAINDER_NS: u64 = 222_638_250;
pub const METAL_STAGE_NS: u64 = 530_542_250;
pub const HOST_FIRST_POLY_NS: u64 = 213_958;

pub const MATERIALIZE_PRODUCTS: u64 = 23 * CYCLES;
pub const OPENING_PRODUCTS: u64 = 17 * CYCLES;
pub const SHIFT_SUCCESSOR_EXTRA_PRODUCTS: u64 = 2 * CYCLES;
pub const UNISKIP_FIELD_PRODUCTS: u64 = 18 * CYCLES;
pub const UNISKIP_U32_MADDS: u64 = 522 * CYCLES;

pub const UNISKIP_TWO_PASS_BYTES: u64 = 24 * (1u64 << 30);
pub const MATERIALIZE_BYTES: u64 = 12 * (1u64 << 30);
pub const PREFIX_BYTES: u64 = 11_267_997_696;
pub const OPENING_BYTES: u64 = 10_746_593_280;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projection {
    pub materialize_ms: f64,
    pub prefix_ms: f64,
    pub opening_ms: f64,
    pub service_ms: f64,
}

impl Projection {
    pub fn remainder_ms(self) -> f64 {
        self.materialize_ms + self.prefix_ms + self.opening_ms + self.service_ms
    }

    pub fn remainder_speedup(self) -> f64 {
        ns_to_ms(CPU_REMAINDER_NS) / self.remainder_ms()
    }

    pub fn stage_ms(self) -> f64 {
        ns_to_ms(METAL_UNISKIP_NS) + ns_to_ms(HOST_FIRST_POLY_NS) + self.remainder_ms()
    }

    pub fn stage_speedup(self) -> f64 {
        ns_to_ms(CPU_STAGE_NS) / self.stage_ms()
    }
}

pub fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

pub fn speedup(cpu_ns: u64, metal_ns: u64) -> f64 {
    cpu_ns as f64 / metal_ns as f64
}

pub fn latency_cap_ms(cpu_ns: u64, ratio: f64) -> f64 {
    ns_to_ms(cpu_ns) / ratio
}

pub fn traffic_floor_ms(bytes: u64) -> f64 {
    bytes as f64 / COPY_BYTES_PER_SECOND * 1e3
}

pub fn product_floor_ms(products: u64, products_per_second: f64) -> f64 {
    products as f64 / products_per_second * 1e3
}

pub fn unchanged_protocol_floor_ms() -> f64 {
    product_floor_ms(MATERIALIZE_PRODUCTS, FIELD_PRODUCTS_PER_SECOND)
        + traffic_floor_ms(PREFIX_BYTES)
        + product_floor_ms(OPENING_PRODUCTS, FIELD_PRODUCTS_PER_SECOND)
}

pub fn candidate_envelope() -> Projection {
    Projection {
        materialize_ms: 75.0,
        prefix_ms: 32.0,
        opening_ms: 55.0,
        service_ms: 7.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn measured_ratios_match_the_profile() {
        close(speedup(CPU_UNISKIP_NS, METAL_UNISKIP_NS), 8.849, 0.002);
        close(speedup(CPU_REMAINDER_NS, METAL_REMAINDER_NS), 4.911, 0.002);
        close(speedup(CPU_STAGE_NS, METAL_STAGE_NS), 7.193, 0.002);
    }

    #[test]
    fn candidate_envelope_clears_remainder_five_x_and_stage_eight_x() {
        let candidate = candidate_envelope();
        assert!(candidate.remainder_speedup() > 6.4);
        assert!(candidate.stage_speedup() >= 8.0);
        assert!(candidate.remainder_ms() <= latency_cap_ms(CPU_REMAINDER_NS, 5.0));
    }

    #[test]
    fn standalone_remainder_eight_x_has_no_service_budget() {
        let arithmetic_and_traffic = unchanged_protocol_floor_ms();
        let cap = latency_cap_ms(CPU_REMAINDER_NS, 8.0);
        assert!(cap - arithmetic_and_traffic < 1.0);
    }

    #[test]
    fn carrier_successor_products_are_priced() {
        close(
            product_floor_ms(SHIFT_SUCCESSOR_EXTRA_PRODUCTS, FIELD_PRODUCTS_PER_SECOND),
            5.574,
            0.01,
        );
    }

    #[test]
    fn uniskip_floors_do_not_select_another_shader_tweak() {
        close(traffic_floor_ms(UNISKIP_TWO_PASS_BYTES), 57.06, 0.05);
        close(
            product_floor_ms(UNISKIP_FIELD_PRODUCTS, UNISKIP_FIELD_PRODUCTS_PER_SECOND),
            73.57,
            0.05,
        );
        close(
            product_floor_ms(UNISKIP_U32_MADDS, U32_MADDS_PER_SECOND),
            36.57,
            0.05,
        );
    }
}
