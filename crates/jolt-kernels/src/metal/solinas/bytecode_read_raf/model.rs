use super::{
    BytecodeReadRafDiagnostics, BytecodeReadRafError, BytecodeReadRafShape, BytecodeReadRafStatus,
    BYTECODE_ADDRESS_AKITA_OFFSET, BYTECODE_ADDRESS_SIMD_WIDTH,
};

pub const BYTECODE_ADDRESS_LOG26_CPU_SAMPLES_NS: [u64; 5] = [
    172_796_544,
    198_165_708,
    181_211_502,
    190_915_958,
    198_945_292,
];
pub const BYTECODE_ADDRESS_LOG26_CPU_MEDIAN_NS: u64 = 190_915_958;
pub const BYTECODE_ADDRESS_LOG26_CPU_PREPARE_MEDIAN_NS: u64 = 182_930_333;
pub const BYTECODE_ADDRESS_LOG26_PROVE_ROUND_TOTAL_NS: u64 = 7_918_251;
pub const BYTECODE_ADDRESS_FIVE_X_CAP_NS: u64 = BYTECODE_ADDRESS_LOG26_CPU_MEDIAN_NS / 5;
pub const BYTECODE_ADDRESS_EIGHT_X_CAP_NS: u64 = BYTECODE_ADDRESS_LOG26_CPU_MEDIAN_NS / 8;
pub const BYTECODE_ADDRESS_COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const BYTECODE_ADDRESS_FULL_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;
pub const BYTECODE_ADDRESS_U64_ACCEPTANCE_FLOOR_PER_SECOND: u64 = 26_272_000_000;
pub const BYTECODE_ADDRESS_PLUS_CYCLE_LOG26_CPU_MEDIAN_NS: u64 = 1_203_638_208;
pub const BYTECODE_CYCLE_LOG26_METAL_MEDIAN_NS: u64 = 160_876_418;
pub const BYTECODE_ADDRESS_PLUS_CYCLE_EIGHT_X_CAP_NS: u64 =
    BYTECODE_ADDRESS_PLUS_CYCLE_LOG26_CPU_MEDIAN_NS / 8;

const _: () =
    assert!(BYTECODE_CYCLE_LOG26_METAL_MEDIAN_NS > BYTECODE_ADDRESS_PLUS_CYCLE_EIGHT_X_CAP_NS);

pub const fn bytecode_address_akita_modulus() -> u128 {
    u128::MAX - (BYTECODE_ADDRESS_AKITA_OFFSET as u128 - 1)
}

/// Independent canonical model for the shader's signed 128-by-64 product.
pub fn exact_signed_u64_product_oracle(
    coefficient: u128,
    magnitude: u64,
    negative: bool,
) -> Result<u128, BytecodeReadRafError> {
    let modulus = bytecode_address_akita_modulus();
    if coefficient >= modulus {
        return Err(BytecodeReadRafError::NonCanonicalCoefficient(coefficient));
    }

    let magnitude = u128::from(magnitude);
    let low_product = u128::from(coefficient as u64) * magnitude;
    let high_product = (coefficient >> 64) * magnitude;
    let shifted_high = u128::from(high_product as u64) << 64;
    let (low, low_carry) = low_product.overflowing_add(shifted_high);
    let high = (high_product >> 64) + u128::from(low_carry);
    let correction = high * u128::from(BYTECODE_ADDRESS_AKITA_OFFSET);
    let (mut reduced, correction_carry) = low.overflowing_add(correction);
    if correction_carry {
        let (folded, second_carry) =
            reduced.overflowing_add(u128::from(BYTECODE_ADDRESS_AKITA_OFFSET));
        if second_carry {
            return Err(BytecodeReadRafError::ArithmeticOverflow(
                "signed u64 reduction",
            ));
        }
        reduced = folded;
    }
    if reduced >= modulus {
        reduced -= modulus;
    }
    if negative && reduced != 0 {
        Ok(modulus - reduced)
    } else {
        Ok(reduced)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BytecodeReadRafCsrCharge {
    LogicalTwoPass,
    CachedSecondPass,
    ReusedProducer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BytecodeReadRafFusedProductPath {
    FullWidth,
    ExactU64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafRoofRates {
    pub copy_bytes_per_second: u64,
    pub full_products_per_second: Option<u64>,
    pub u64_products_per_second: Option<u64>,
    pub field_additions_per_second: Option<u64>,
    pub reduction_lane_additions_per_second: Option<u64>,
    pub csr_atomic_operations_per_second: Option<u64>,
    pub nine_accumulator_updates_per_second: Option<u64>,
}

impl BytecodeReadRafRoofRates {
    pub const M4_MAX_UNMATCHED: Self = Self {
        copy_bytes_per_second: BYTECODE_ADDRESS_COPY_BYTES_PER_SECOND,
        full_products_per_second: None,
        u64_products_per_second: None,
        field_additions_per_second: None,
        reduction_lane_additions_per_second: None,
        csr_atomic_operations_per_second: None,
        nine_accumulator_updates_per_second: None,
    };

    fn validate(
        self,
        product_path: BytecodeReadRafFusedProductPath,
    ) -> Result<ValidatedRoofRates, BytecodeReadRafError> {
        let copy_bytes_per_second = nonzero_rate("copy bytes", self.copy_bytes_per_second)?;
        let full_products_per_second =
            matched_rate("full products", self.full_products_per_second)?;
        let u64_products_per_second = match product_path {
            BytecodeReadRafFusedProductPath::FullWidth => None,
            BytecodeReadRafFusedProductPath::ExactU64 => Some(matched_rate(
                "signed-u64 products",
                self.u64_products_per_second,
            )?),
        };
        Ok(ValidatedRoofRates {
            copy_bytes_per_second,
            full_products_per_second,
            u64_products_per_second,
            field_additions_per_second: matched_rate(
                "field additions",
                self.field_additions_per_second,
            )?,
            reduction_lane_additions_per_second: matched_rate(
                "SIMD reduction-lane additions",
                self.reduction_lane_additions_per_second,
            )?,
            csr_atomic_operations_per_second: matched_rate(
                "CSR atomic operations",
                self.csr_atomic_operations_per_second,
            )?,
            nine_accumulator_updates_per_second: matched_rate(
                "nine-accumulator output updates",
                self.nine_accumulator_updates_per_second,
            )?,
        })
    }
}

#[derive(Clone, Copy)]
#[expect(
    clippy::struct_field_names,
    reason = "the unit suffix keeps analytical rates unambiguous"
)]
struct ValidatedRoofRates {
    copy_bytes_per_second: u64,
    full_products_per_second: u64,
    u64_products_per_second: Option<u64>,
    field_additions_per_second: u64,
    reduction_lane_additions_per_second: u64,
    csr_atomic_operations_per_second: u64,
    nine_accumulator_updates_per_second: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafWorkload {
    rows: u64,
    outer_blocks: u64,
    runs: u64,
    long_runs: u64,
}

impl BytecodeReadRafWorkload {
    /// Constructs a workload whose aggregate run counts can cover every outer block.
    pub fn new(
        shape: BytecodeReadRafShape,
        runs: usize,
        long_runs: usize,
        short_threshold: usize,
    ) -> Result<Self, BytecodeReadRafError> {
        if runs < shape.outer_length || runs > shape.run_capacity {
            return Err(BytecodeReadRafError::InvalidRunCount {
                minimum: shape.outer_length,
                maximum: shape.run_capacity,
                got: runs,
            });
        }
        if short_threshold == 0 || short_threshold > shape.inner_length {
            return Err(BytecodeReadRafError::InvalidShortThreshold(short_threshold));
        }
        if long_runs > runs {
            return Err(BytecodeReadRafError::InvalidLongRunCount {
                maximum: runs as u64,
                got: long_runs,
            });
        }
        let minimum_long_length =
            short_threshold
                .checked_add(1)
                .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                    "minimum long-run length",
                ))?;
        let maximum_long_runs = shape
            .outer_length
            .checked_mul(shape.inner_length / minimum_long_length)
            .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                "maximum long-run count",
            ))?;
        if long_runs > maximum_long_runs {
            return Err(BytecodeReadRafError::InfeasibleRunPartition {
                rows: shape.rows,
                runs,
                long_runs,
                short_threshold,
            });
        }
        let blocks_with_long_runs = long_runs.min(shape.outer_length);
        let all_short_blocks = shape.outer_length - blocks_with_long_runs;
        let all_short_runs_per_block = shape.inner_length.div_ceil(short_threshold);
        let minimum_outer_partition_runs = long_runs
            .checked_add(
                all_short_blocks
                    .checked_mul(all_short_runs_per_block)
                    .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                        "all-short outer runs",
                    ))?,
            )
            .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                "minimum outer-partition runs",
            ))?;
        // Evenly spread long runs maximize sum(min(K, I - threshold * long)).
        let base_long_runs = long_runs / shape.outer_length;
        let extra_long_blocks = long_runs % shape.outer_length;
        let base_long_occurrences = base_long_runs.checked_mul(short_threshold).ok_or(
            BytecodeReadRafError::ArithmeticOverflow("base long-run occurrences"),
        )?;
        let base_run_capacity = shape
            .inner_length
            .checked_sub(base_long_occurrences)
            .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                "base outer run capacity",
            ))?;
        let base_maximum_runs = shape.addresses.min(base_run_capacity);
        let extra_maximum_runs = if extra_long_blocks == 0 {
            0
        } else {
            let extra_long_occurrences = base_long_runs
                .checked_add(1)
                .and_then(|count| count.checked_mul(short_threshold))
                .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                    "extra long-run occurrences",
                ))?;
            let extra_run_capacity = shape
                .inner_length
                .checked_sub(extra_long_occurrences)
                .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                    "extra outer run capacity",
                ))?;
            shape.addresses.min(extra_run_capacity)
        };
        let maximum_outer_partition_runs = (shape.outer_length - extra_long_blocks)
            .checked_mul(base_maximum_runs)
            .and_then(|base| {
                extra_long_blocks
                    .checked_mul(extra_maximum_runs)
                    .and_then(|extra| base.checked_add(extra))
            })
            .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                "maximum outer-partition runs",
            ))?;
        let minimum_occurrences = runs
            .checked_add(long_runs.checked_mul(short_threshold).ok_or(
                BytecodeReadRafError::ArithmeticOverflow("long-run occurrence floor"),
            )?)
            .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                "run occurrence floor",
            ))?;
        let short_runs = runs - long_runs;
        let maximum_occurrences = short_runs
            .checked_mul(short_threshold)
            .and_then(|short| {
                long_runs
                    .checked_mul(shape.inner_length)
                    .and_then(|long| short.checked_add(long))
            })
            .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                "run occurrence ceiling",
            ))?;
        if (all_short_blocks != 0 && all_short_runs_per_block > shape.addresses)
            || runs < minimum_outer_partition_runs
            || runs > maximum_outer_partition_runs
            || minimum_occurrences > shape.rows
            || maximum_occurrences < shape.rows
        {
            return Err(BytecodeReadRafError::InfeasibleRunPartition {
                rows: shape.rows,
                runs,
                long_runs,
                short_threshold,
            });
        }
        Ok(Self {
            rows: shape.rows as u64,
            outer_blocks: shape.outer_length as u64,
            runs: runs as u64,
            long_runs: long_runs as u64,
        })
    }

    pub fn from_telemetry(
        shape: BytecodeReadRafShape,
        status: BytecodeReadRafStatus,
        diagnostics: BytecodeReadRafDiagnostics,
        short_threshold: usize,
    ) -> Result<Self, BytecodeReadRafError> {
        diagnostics.validate(shape, status, short_threshold)?;
        let counts = status.validate(shape)?;
        Self::new(
            shape,
            counts.total()? as usize,
            counts.long_runs as usize,
            short_threshold,
        )
    }

    pub const fn rows(self) -> u64 {
        self.rows
    }

    pub const fn runs(self) -> u64 {
        self.runs
    }

    pub const fn long_runs(self) -> u64 {
        self.long_runs
    }

    pub fn fused_products(self) -> Result<u64, BytecodeReadRafError> {
        checked_linear_u64("fused products", 4, self.rows, 0, self.runs)
    }

    pub fn outer_products(self) -> Result<u64, BytecodeReadRafError> {
        checked_linear_u64("outer products", 0, self.rows, 9, self.runs)
    }

    pub fn fused_issued_lane_products_upper(self) -> Result<u64, BytecodeReadRafError> {
        checked_add_u64(
            "issued fused lane products",
            self.fused_products()?,
            checked_product_u64("masked fused lane products", 124, self.long_runs)?,
        )
    }

    pub fn outer_issued_lane_products(self) -> Result<u64, BytecodeReadRafError> {
        checked_add_u64(
            "issued outer lane products",
            self.outer_products()?,
            checked_product_u64("masked outer lane products", 279, self.long_runs)?,
        )
    }

    pub fn field_accumulation_additions(self) -> Result<u64, BytecodeReadRafError> {
        checked_linear_u64("field accumulation additions", 9, self.rows, 0, self.runs)
    }

    pub fn field_accumulation_issued_lane_additions_upper(
        self,
    ) -> Result<u64, BytecodeReadRafError> {
        checked_add_u64(
            "issued field accumulation additions",
            self.field_accumulation_additions()?,
            checked_product_u64("masked field additions", 279, self.long_runs)?,
        )
    }

    pub fn long_simd_useful_reduction_additions(self) -> Result<u64, BytecodeReadRafError> {
        checked_product_u64("long-run useful SIMD additions", 279, self.long_runs)
    }

    pub fn long_simd_issued_lane_additions(self) -> Result<u64, BytecodeReadRafError> {
        checked_product_u64("long-run issued SIMD additions", 1_440, self.long_runs)
    }

    pub fn csr_atomic_operations(self) -> Result<u64, BytecodeReadRafError> {
        let row_atomics = checked_product_u64("CSR row atomics", 2, self.rows)?;
        let group_atomics = checked_product_u64("CSR group atomics", 4, self.outer_blocks)?;
        let diagnostic_atomics = checked_product_u64("CSR diagnostic atomics", 3, self.runs)?;
        checked_add_u64(
            "CSR atomic operations",
            checked_add_u64("CSR row and group atomics", row_atomics, group_atomics)?,
            diagnostic_atomics,
        )
    }

    pub const fn nine_accumulator_updates(self) -> u64 {
        self.runs
    }

    pub fn csr_bytes(self, charge: BytecodeReadRafCsrCharge) -> Result<u64, BytecodeReadRafError> {
        let row_and_run_bytes = match charge {
            BytecodeReadRafCsrCharge::LogicalTwoPass => {
                checked_linear_u64("logical CSR bytes", 84, self.rows, 40, self.runs)
            }
            BytecodeReadRafCsrCharge::CachedSecondPass => {
                checked_linear_u64("cached CSR bytes", 44, self.rows, 40, self.runs)
            }
            BytecodeReadRafCsrCharge::ReusedProducer => return Ok(0),
        }?;
        checked_add_u64(
            "CSR bytes with status",
            row_and_run_bytes,
            checked_product_u64("CSR status bytes", 32, self.outer_blocks)?,
        )
    }

    pub fn run_bytes(self) -> Result<u64, BytecodeReadRafError> {
        checked_linear_u64("run bytes", 44, self.rows, 376, self.runs)
    }

    pub fn shader_logical_e_lo_bytes(self) -> Result<u64, BytecodeReadRafError> {
        checked_linear_u64("shader-logical E_lo bytes", 144, self.rows, 0, self.runs)
    }

    pub fn projection(
        self,
        rates: BytecodeReadRafRoofRates,
        csr_charge: BytecodeReadRafCsrCharge,
        product_path: BytecodeReadRafFusedProductPath,
        utilization_percent: u64,
    ) -> Result<BytecodeReadRafRoofProjection, BytecodeReadRafError> {
        let rates = rates.validate(product_path)?;
        if !(1..=100).contains(&utilization_percent) {
            return Err(BytecodeReadRafError::InvalidUtilization(
                utilization_percent,
            ));
        }

        let csr_traffic_roof_ns =
            rate_ns(self.csr_bytes(csr_charge)?, rates.copy_bytes_per_second)?;
        let csr_atomic_roof_ns = rate_ns(
            self.csr_atomic_operations()?,
            rates.csr_atomic_operations_per_second,
        )?;
        let csr_roof_ns = csr_traffic_roof_ns.max(csr_atomic_roof_ns);
        let run_traffic_roof_ns = rate_ns(self.run_bytes()?, rates.copy_bytes_per_second)?;
        let fused_product_roof_ns =
            match product_path {
                BytecodeReadRafFusedProductPath::FullWidth => rate_ns(
                    self.fused_issued_lane_products_upper()?,
                    rates.full_products_per_second,
                )?,
                BytecodeReadRafFusedProductPath::ExactU64 => rate_ns(
                    self.fused_issued_lane_products_upper()?,
                    rates.u64_products_per_second.ok_or(
                        BytecodeReadRafError::MissingMatchedRate("signed-u64 products"),
                    )?,
                )?,
            };
        let outer_product_roof_ns = rate_ns(
            self.outer_issued_lane_products()?,
            rates.full_products_per_second,
        )?;
        let product_compute_roof_ns = checked_add_u64(
            "issued product compute time",
            fused_product_roof_ns,
            outer_product_roof_ns,
        )?;
        let field_add_roof_ns = rate_ns(
            self.field_accumulation_issued_lane_additions_upper()?,
            rates.field_additions_per_second,
        )?;
        let reduction_add_roof_ns = rate_ns(
            self.long_simd_issued_lane_additions()?,
            rates.reduction_lane_additions_per_second,
        )?;
        let nine_accumulator_roof_ns = rate_ns(
            self.nine_accumulator_updates(),
            rates.nine_accumulator_updates_per_second,
        )?;
        let run_compute_roof_ns = [
            product_compute_roof_ns,
            field_add_roof_ns,
            reduction_add_roof_ns,
            nine_accumulator_roof_ns,
        ]
        .into_iter()
        .try_fold(0u64, |sum, value| {
            checked_add_u64("run compute time", sum, value)
        })?;
        let run_roof_ns = run_traffic_roof_ns.max(run_compute_roof_ns);
        let csr_cap_ns = utilization_cap_ns(csr_roof_ns, utilization_percent)?;
        let run_cap_ns = utilization_cap_ns(run_roof_ns, utilization_percent)?;
        let gpu_cap_ns = checked_add_u64("GPU cap", csr_cap_ns, run_cap_ns)?;
        Ok(BytecodeReadRafRoofProjection {
            csr_traffic_roof_ns,
            csr_atomic_roof_ns,
            csr_roof_ns,
            run_traffic_roof_ns,
            fused_product_roof_ns,
            outer_product_roof_ns,
            product_compute_roof_ns,
            field_add_roof_ns,
            reduction_add_roof_ns,
            nine_accumulator_roof_ns,
            run_compute_roof_ns,
            csr_cap_ns,
            run_cap_ns,
            gpu_cap_ns,
        })
    }

    pub const fn clears_five_x(complete_member_ns: u64) -> bool {
        complete_member_ns <= BYTECODE_ADDRESS_FIVE_X_CAP_NS
    }

    pub const fn clears_eight_x(complete_member_ns: u64) -> bool {
        complete_member_ns <= BYTECODE_ADDRESS_EIGHT_X_CAP_NS
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafRoofProjection {
    pub csr_traffic_roof_ns: u64,
    pub csr_atomic_roof_ns: u64,
    pub csr_roof_ns: u64,
    pub run_traffic_roof_ns: u64,
    pub fused_product_roof_ns: u64,
    pub outer_product_roof_ns: u64,
    pub product_compute_roof_ns: u64,
    pub field_add_roof_ns: u64,
    pub reduction_add_roof_ns: u64,
    pub nine_accumulator_roof_ns: u64,
    pub run_compute_roof_ns: u64,
    pub csr_cap_ns: u64,
    pub run_cap_ns: u64,
    pub gpu_cap_ns: u64,
}

/// Maximum attainable long-run count after applying the limit in every outer block.
pub fn maximum_long_runs(
    shape: BytecodeReadRafShape,
    short_threshold: usize,
) -> Result<u64, BytecodeReadRafError> {
    if short_threshold == 0 || short_threshold > shape.inner_length {
        return Err(BytecodeReadRafError::InvalidShortThreshold(short_threshold));
    }
    let minimum_long_length =
        short_threshold
            .checked_add(1)
            .ok_or(BytecodeReadRafError::ArithmeticOverflow(
                "minimum long-run length",
            ))?;
    let per_outer = shape.inner_length / minimum_long_length;
    let total = shape.outer_length.checked_mul(per_outer).ok_or(
        BytecodeReadRafError::ArithmeticOverflow("maximum long-run count"),
    )?;
    u64::try_from(total)
        .map_err(|_| BytecodeReadRafError::ArithmeticOverflow("maximum long-run count"))
}

pub const fn maximum_masked_lanes_per_long_run() -> u64 {
    BYTECODE_ADDRESS_SIMD_WIDTH as u64 - 1
}

fn nonzero_rate(name: &'static str, value: u64) -> Result<u64, BytecodeReadRafError> {
    if value == 0 {
        Err(BytecodeReadRafError::ZeroRate(name))
    } else {
        Ok(value)
    }
}

fn matched_rate(name: &'static str, value: Option<u64>) -> Result<u64, BytecodeReadRafError> {
    nonzero_rate(
        name,
        value.ok_or(BytecodeReadRafError::MissingMatchedRate(name))?,
    )
}

fn checked_linear_u64(
    name: &'static str,
    row_coefficient: u64,
    rows: u64,
    run_coefficient: u64,
    runs: u64,
) -> Result<u64, BytecodeReadRafError> {
    let value = u128::from(row_coefficient) * u128::from(rows)
        + u128::from(run_coefficient) * u128::from(runs);
    u64::try_from(value).map_err(|_| BytecodeReadRafError::ArithmeticOverflow(name))
}

fn checked_product_u64(
    name: &'static str,
    left: u64,
    right: u64,
) -> Result<u64, BytecodeReadRafError> {
    left.checked_mul(right)
        .ok_or(BytecodeReadRafError::ArithmeticOverflow(name))
}

fn checked_add_u64(name: &'static str, left: u64, right: u64) -> Result<u64, BytecodeReadRafError> {
    left.checked_add(right)
        .ok_or(BytecodeReadRafError::ArithmeticOverflow(name))
}

fn rate_ns(work: u64, rate_per_second: u64) -> Result<u64, BytecodeReadRafError> {
    if rate_per_second == 0 {
        return Err(BytecodeReadRafError::ZeroRate("rate"));
    }
    let numerator = u128::from(work) * 1_000_000_000u128;
    let denominator = u128::from(rate_per_second);
    let result = numerator.div_ceil(denominator);
    u64::try_from(result).map_err(|_| BytecodeReadRafError::ArithmeticOverflow("rate time"))
}

fn utilization_cap_ns(roof_ns: u64, utilization_percent: u64) -> Result<u64, BytecodeReadRafError> {
    let cap = (u128::from(roof_ns) * 100).div_ceil(u128::from(utilization_percent));
    u64::try_from(cap).map_err(|_| BytecodeReadRafError::ArithmeticOverflow("utilization cap"))
}
