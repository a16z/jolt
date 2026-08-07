pub const LOG_T: usize = 26;
pub const COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const MATCHED_HALF_TERMS_PER_SECOND: u64 = 33_168_000_000;
pub const MATCHED_FULL_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;
pub const FROZEN_CPU_MEMBER_NS: u64 = 99_905_582;
pub const FROZEN_CPU_OUTER_PAIR_NS: u64 = 1_015_295_537;
pub const FROZEN_CPU_INSTRUCTION_PAIR_NS: u64 = 827_118_001;

const FIELD_BYTES: u64 = 16;
const NATIVE_BYTES: u64 = 8;

fn ceil_div(numerator: u128, denominator: u128) -> u64 {
    let value = numerator.div_ceil(denominator);
    assert!(value <= u128::from(u64::MAX));
    value as u64
}

pub fn rate_ns(units: u64, units_per_second: u64) -> u64 {
    ceil_div(
        u128::from(units) * 1_000_000_000,
        u128::from(units_per_second),
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    pub log_t: usize,
    pub partial_blocks: u64,
}

impl Geometry {
    pub fn new(log_t: usize, partial_blocks: u64) -> Result<Self, &'static str> {
        if log_t < 2 || partial_blocks == 0 {
            return Err("invalid successor geometry");
        }
        let geometry = Self {
            log_t,
            partial_blocks,
        };
        if !geometry.suffix_elements().is_multiple_of(partial_blocks) {
            return Err("partial blocks must divide the suffix domain");
        }
        Ok(geometry)
    }

    pub const fn rows(self) -> u64 {
        1u64 << self.log_t
    }

    pub const fn prefix_vars(self) -> usize {
        self.log_t - self.log_t / 2
    }

    pub const fn suffix_vars(self) -> usize {
        self.log_t / 2
    }

    pub const fn prefix_elements(self) -> u64 {
        1u64 << self.prefix_vars()
    }

    pub const fn suffix_elements(self) -> u64 {
        1u64 << self.suffix_vars()
    }

    pub const fn q_half_terms(self) -> u64 {
        3 * self.rows()
    }

    pub const fn midpoint_half_terms(self) -> u64 {
        self.rows()
    }

    pub const fn displaced_opening_full_products(self) -> u64 {
        3 * self.rows() + 3 * self.suffix_elements()
    }

    pub const fn q_partial_roundtrip_bytes(self) -> u64 {
        2 * 3 * self.partial_blocks * self.prefix_elements() * FIELD_BYTES
    }

    pub const fn component_bytes(self) -> u64 {
        3 * self.prefix_elements() * FIELD_BYTES
    }

    pub const fn rd_plane_bytes(self) -> u64 {
        self.rows() * NATIVE_BYTES
    }

    pub const fn fused_stage1_new_bytes(self) -> u64 {
        self.q_partial_roundtrip_bytes() + self.component_bytes() + self.rd_plane_bytes()
    }

    pub const fn midpoint_bytes(self) -> u64 {
        self.rd_plane_bytes() + FIELD_BYTES * (self.prefix_elements() + self.suffix_elements())
    }

    pub fn stage1_floor_ns(self) -> u64 {
        rate_ns(self.q_half_terms(), MATCHED_HALF_TERMS_PER_SECOND).max(rate_ns(
            self.fused_stage1_new_bytes(),
            COPY_BYTES_PER_SECOND,
        ))
    }

    pub fn midpoint_floor_ns(self) -> u64 {
        rate_ns(self.midpoint_half_terms(), MATCHED_HALF_TERMS_PER_SECOND)
            .max(rate_ns(self.midpoint_bytes(), COPY_BYTES_PER_SECOND))
    }

    pub fn gross_gpu_floor_ns(self) -> u64 {
        self.stage1_floor_ns() + self.midpoint_floor_ns()
    }

    pub fn utilization_cap_ns(self, percent: u64) -> Result<u64, &'static str> {
        if !(1..=100).contains(&percent) {
            return Err("utilization must be in 1..=100");
        }
        Ok(ceil_div(
            u128::from(self.gross_gpu_floor_ns()) * 100,
            u128::from(percent),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log26_screen_is_exact() {
        let geometry = Geometry::new(LOG_T, 256).unwrap();
        assert_eq!(geometry.rows(), 67_108_864);
        assert_eq!(geometry.prefix_elements(), 8_192);
        assert_eq!(geometry.suffix_elements(), 8_192);
        assert_eq!(geometry.q_half_terms(), 201_326_592);
        assert_eq!(geometry.midpoint_half_terms(), 67_108_864);
        assert_eq!(geometry.q_partial_roundtrip_bytes(), 201_326_592);
        assert_eq!(geometry.component_bytes(), 393_216);
        assert_eq!(geometry.rd_plane_bytes(), 536_870_912);
        assert_eq!(geometry.stage1_floor_ns(), 6_069_905);
        assert_eq!(geometry.midpoint_floor_ns(), 2_023_302);
        assert_eq!(geometry.gross_gpu_floor_ns(), 8_093_207);
        assert_eq!(geometry.utilization_cap_ns(80).unwrap(), 10_116_509);
        assert!(geometry.utilization_cap_ns(80).unwrap() < FROZEN_CPU_MEMBER_NS / 8);
    }

    #[test]
    fn frozen_bars_do_not_drift() {
        assert_eq!(FROZEN_CPU_MEMBER_NS / 5, 19_981_116);
        assert_eq!(FROZEN_CPU_MEMBER_NS / 8, 12_488_197);
        assert_eq!(FROZEN_CPU_OUTER_PAIR_NS / 5, 203_059_107);
        assert_eq!(FROZEN_CPU_INSTRUCTION_PAIR_NS / 5, 165_423_600);
        assert_eq!(
            rate_ns(
                Geometry::new(LOG_T, 256)
                    .unwrap()
                    .displaced_opening_full_products(),
                MATCHED_FULL_PRODUCTS_PER_SECOND,
            ),
            11_124_374
        );
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        assert!(Geometry::new(1, 256).is_err());
        assert!(Geometry::new(10, 0).is_err());
        assert!(Geometry::new(10, 48).is_err());
        assert!(Geometry::new(10, 32)
            .unwrap()
            .utilization_cap_ns(0)
            .is_err());
    }
}
