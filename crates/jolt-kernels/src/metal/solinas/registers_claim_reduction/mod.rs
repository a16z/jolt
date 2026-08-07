//! Checked geometry, roof model, midpoint-alias contract, and scalar oracle for
//! registers claim reduction.

mod bcsr_midpoint_runtime;
mod bcsr_runtime;
mod resident_bcsr;
mod runtime;

#[cfg(feature = "test-utils")]
pub use bcsr_midpoint_runtime::{
    RegistersClaimBcsrMidpointBenchmarkInvocation, RegistersClaimBcsrMidpointBenchmarkObservation,
};
#[cfg(feature = "test-utils")]
pub use bcsr_runtime::{
    RegistersClaimBcsrBenchmarkError, RegistersClaimBcsrBenchmarkInvocation,
    RegistersClaimBcsrBenchmarkObservation,
};
pub use resident_bcsr::*;
pub use runtime::*;

#[cfg(test)]
#[expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "tests use fixed valid fixtures"
)]
mod tests;

use std::mem::{align_of, size_of};

use jolt_field::Field;
use jolt_poly::{EqPolynomial, UnivariatePoly};
use thiserror::Error;

pub(super) const SOURCE: &str = include_str!("shader.metal");
pub(super) const BCSR_SOURCE: &str = include_str!("bcsr_shader.metal");

pub const REGISTERS_CLAIM_SIMD_WIDTH: usize = 32;
pub const REGISTERS_CLAIM_OUTPUT_COLUMNS: usize = 3;
pub const REGISTERS_CLAIM_GAMMA_POWERS: usize = 2;
pub const REGISTERS_CLAIM_WIDE_LIMBS: usize = 7;
pub const REGISTERS_CLAIM_INITIAL_METAL_LOG_T: usize = 25;
pub const REGISTERS_CLAIM_TARGET_LOG_T: usize = 26;
pub const REGISTERS_CLAIM_AKITA_OFFSET: u32 = 0xffff_a7f7;

pub const REGISTERS_CLAIM_FROZEN_CPU_NS: u64 = 99_905_582;
pub const REGISTERS_CLAIM_FIVE_X_GATE_NS: u64 = 19_981_116;
pub const REGISTERS_CLAIM_SEVEN_X_GATE_NS: u64 = 14_272_226;
pub const REGISTERS_CLAIM_EIGHT_X_GATE_NS: u64 = 12_488_197;
pub const REGISTERS_CLAIM_HALF_WIDTH_FLOOR_PER_SECOND: u64 = 26_272_000_000;
pub const REGISTERS_CLAIM_COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const REGISTERS_CLAIM_CONSERVATIVE_FULL_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;

pub const INSTRUCTION_INPUT_RS1_TABLE: usize = 1;
pub const INSTRUCTION_INPUT_RS2_TABLE: usize = 5;

pub const LINEAR_Q_RD_WRITE_VALUE_SLOT: u64 = 0;
pub const LINEAR_Q_RS1_VALUE_SLOT: u64 = 1;
pub const LINEAR_Q_RS2_VALUE_SLOT: u64 = 2;
pub const LINEAR_Q_GAMMA_POWERS_SLOT: u64 = 3;
pub const LINEAR_Q_EQ_SUFFIX_SLOT: u64 = 4;
pub const LINEAR_Q_OUTPUT_SLOT: u64 = 5;
pub const LINEAR_Q_PARAMS_SLOT: u64 = 6;

pub const DIRECT_FOLD_RD_WRITE_VALUE_SLOT: u64 = 0;
pub const DIRECT_FOLD_RS1_VALUE_SLOT: u64 = 1;
pub const DIRECT_FOLD_RS2_VALUE_SLOT: u64 = 2;
pub const DIRECT_FOLD_EQ_PREFIX_SLOT: u64 = 3;
pub const DIRECT_FOLD_OUTPUT_SLOT: u64 = 4;
pub const DIRECT_FOLD_PARAMS_SLOT: u64 = 5;
pub const DIRECT_FOLD_THREADGROUP_SLOT: u64 = 0;

pub(crate) const BUILD_LINEAR_PIPELINE: &str = "solinas_registers_claim_build_linear_q";
pub(crate) const BUILD_LINEAR_CANONICAL_PIPELINE: &str =
    "solinas_registers_claim_build_linear_q_canonical";
pub(crate) const DIRECT_FOLD_PIPELINE: &str = "solinas_registers_claim_fold_direct";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RegistersClaimAccumulator {
    Deferred224,
    #[default]
    Canonical128,
}

impl RegistersClaimAccumulator {
    pub(crate) const fn pipeline(self) -> &'static str {
        match self {
            Self::Deferred224 => BUILD_LINEAR_PIPELINE,
            Self::Canonical128 => BUILD_LINEAR_CANONICAL_PIPELINE,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Deferred224 => "deferred224",
            Self::Canonical128 => "canonical128",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegistersClaimParams {
    pub rows: u32,
    pub prefix_elements: u32,
    pub suffix_elements: u32,
    pub reserved: u32,
}

const _: [(); 16] = [(); size_of::<RegistersClaimParams>()];
const _: [(); 4] = [(); align_of::<RegistersClaimParams>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimKernelConfig {
    pub build_threads_per_threadgroup: usize,
    pub fold_threads_per_threadgroup: usize,
    pub accumulator: RegistersClaimAccumulator,
}

impl Default for RegistersClaimKernelConfig {
    fn default() -> Self {
        Self {
            build_threads_per_threadgroup: 128,
            fold_threads_per_threadgroup: 128,
            accumulator: RegistersClaimAccumulator::Canonical128,
        }
    }
}

impl RegistersClaimKernelConfig {
    pub fn validate(self) -> Result<Self, RegistersClaimPlanError> {
        for (phase, width) in [
            ("build", self.build_threads_per_threadgroup),
            ("fold", self.fold_threads_per_threadgroup),
        ] {
            if width == 0 || !width.is_multiple_of(REGISTERS_CLAIM_SIMD_WIDTH) {
                return Err(RegistersClaimPlanError::InvalidThreadgroupWidth { phase, width });
            }
        }
        Ok(self)
    }

    pub fn alias_fold_threadgroup_bytes(self) -> Result<usize, RegistersClaimPlanError> {
        let config = self.validate()?;
        threadgroup_bytes(1, config.fold_threads_per_threadgroup, "alias fold")
    }

    pub fn direct_fold_threadgroup_bytes(self) -> Result<usize, RegistersClaimPlanError> {
        let config = self.validate()?;
        threadgroup_bytes(
            REGISTERS_CLAIM_OUTPUT_COLUMNS,
            config.fold_threads_per_threadgroup,
            "direct fold",
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimGeometry {
    rows: usize,
    log_t: usize,
    prefix_vars: usize,
    suffix_vars: usize,
    prefix_elements: usize,
    suffix_elements: usize,
}

impl RegistersClaimGeometry {
    pub fn new(rows: usize) -> Result<Self, RegistersClaimPlanError> {
        if rows < 2 || !rows.is_power_of_two() {
            return Err(RegistersClaimPlanError::InvalidRows(rows));
        }
        let _ = abi_count("rows", rows)?;

        let log_t = rows.trailing_zeros() as usize;
        let suffix_vars = log_t / 2;
        let prefix_vars = log_t - suffix_vars;
        let prefix_elements = checked_power_of_two("prefix elements", prefix_vars)?;
        let suffix_elements = checked_power_of_two("suffix elements", suffix_vars)?;
        debug_assert_eq!(prefix_elements * suffix_elements, rows);

        Ok(Self {
            rows,
            log_t,
            prefix_vars,
            suffix_vars,
            prefix_elements,
            suffix_elements,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn prefix_vars(self) -> usize {
        self.prefix_vars
    }

    pub const fn suffix_vars(self) -> usize {
        self.suffix_vars
    }

    pub const fn prefix_elements(self) -> usize {
        self.prefix_elements
    }

    pub const fn suffix_elements(self) -> usize {
        self.suffix_elements
    }

    pub fn row_index(self, x_hi: usize, x_lo: usize) -> Result<usize, RegistersClaimPlanError> {
        if x_hi >= self.suffix_elements || x_lo >= self.prefix_elements {
            return Err(RegistersClaimPlanError::CoordinateOutOfRange {
                x_hi,
                x_lo,
                suffix_elements: self.suffix_elements,
                prefix_elements: self.prefix_elements,
            });
        }
        Ok(x_hi * self.prefix_elements + x_lo)
    }

    pub fn params(self) -> Result<RegistersClaimParams, RegistersClaimPlanError> {
        Ok(RegistersClaimParams {
            rows: abi_count("rows", self.rows)?,
            prefix_elements: abi_count("prefix elements", self.prefix_elements)?,
            suffix_elements: abi_count("suffix elements", self.suffix_elements)?,
            reserved: 0,
        })
    }

    pub fn linear_q_storage(self) -> Result<RegistersClaimLinearQStorage, RegistersClaimPlanError> {
        let native_plane_bytes = checked_bytes("linear-q native plane", self.rows, 8)?;
        let resident_input_bytes = checked_product(
            "linear-q resident inputs",
            native_plane_bytes,
            REGISTERS_CLAIM_OUTPUT_COLUMNS,
        )?;
        let gamma_powers_bytes =
            checked_bytes("linear-q gamma powers", REGISTERS_CLAIM_GAMMA_POWERS, 16)?;
        let eq_suffix_bytes = checked_bytes("linear-q equality suffix", self.suffix_elements, 16)?;
        let output_bytes = checked_bytes("linear-q output", self.prefix_elements, 16)?;
        let private_bytes = checked_sum(
            "linear-q private allocations",
            &[gamma_powers_bytes, eq_suffix_bytes, output_bytes],
        )?;
        let total_resident_bytes = checked_sum(
            "linear-q total resident bytes",
            &[resident_input_bytes, private_bytes],
        )?;
        let roof_compulsory_bytes = checked_sum(
            "linear-q compulsory bytes",
            &[resident_input_bytes, eq_suffix_bytes, output_bytes],
        )?;
        let shader_issued_bytes = checked_sum(
            "linear-q shader-issued bytes",
            &[
                resident_input_bytes,
                checked_bytes("linear-q issued equality bytes", self.rows, 16)?,
                output_bytes,
            ],
        )?;

        Ok(RegistersClaimLinearQStorage {
            native_plane_bytes,
            resident_input_bytes,
            gamma_powers_bytes,
            eq_suffix_bytes,
            output_bytes,
            private_bytes,
            total_resident_bytes,
            roof_compulsory_bytes,
            shader_issued_bytes,
        })
    }

    pub fn storage(self) -> Result<RegistersClaimStorage, RegistersClaimPlanError> {
        let native_plane_bytes = checked_bytes("native plane", self.rows, size_of::<u64>())?;
        let native_planes_bytes = checked_product(
            "native planes",
            native_plane_bytes,
            REGISTERS_CLAIM_OUTPUT_COLUMNS,
        )?;
        let combined_control_bytes = checked_bytes("combined control", self.rows, 16)?;
        let prefix_field_bytes = checked_bytes("prefix field table", self.prefix_elements, 16)?;
        let suffix_field_bytes = checked_bytes("suffix field table", self.suffix_elements, 16)?;
        let partial_q_handoff_bytes = checked_product(
            "partial-q handoff",
            prefix_field_bytes,
            REGISTERS_CLAIM_OUTPUT_COLUMNS,
        )?;
        let alias_snapshot_bytes = checked_product("alias snapshot", suffix_field_bytes, 2)?;
        let direct_dense_bytes = checked_product(
            "direct dense outputs",
            suffix_field_bytes,
            REGISTERS_CLAIM_OUTPUT_COLUMNS,
        )?;

        let alias_peak_bytes = checked_sum(
            "alias peak",
            &[
                native_planes_bytes,
                prefix_field_bytes,
                suffix_field_bytes,
                alias_snapshot_bytes,
                REGISTERS_CLAIM_GAMMA_POWERS * 16,
                size_of::<RegistersClaimParams>(),
            ],
        )?;
        let direct_peak_bytes = checked_sum(
            "direct peak",
            &[
                native_planes_bytes,
                prefix_field_bytes,
                suffix_field_bytes,
                direct_dense_bytes,
                REGISTERS_CLAIM_GAMMA_POWERS * 16,
                size_of::<RegistersClaimParams>(),
            ],
        )?;
        let cached_control_peak_bytes = checked_sum(
            "cached control peak",
            &[
                native_planes_bytes,
                combined_control_bytes,
                prefix_field_bytes,
                suffix_field_bytes,
                REGISTERS_CLAIM_GAMMA_POWERS * 16,
                size_of::<RegistersClaimParams>(),
            ],
        )?;

        Ok(RegistersClaimStorage {
            native_plane_bytes,
            native_planes_bytes,
            combined_control_bytes,
            prefix_field_bytes,
            suffix_field_bytes,
            partial_q_handoff_bytes,
            alias_snapshot_bytes,
            direct_dense_bytes,
            alias_peak_bytes,
            direct_peak_bytes,
            cached_control_peak_bytes,
        })
    }

    pub fn work(
        self,
        strategy: RegistersClaimStrategy,
    ) -> Result<RegistersClaimWork, RegistersClaimPlanError> {
        let rows = self.rows as u64;
        let prefix = self.prefix_elements as u64;
        let suffix = self.suffix_elements as u64;
        let table_bytes = checked_u64_sum(
            "projection table bytes",
            &[
                checked_u64_product("projection table bytes", 16, prefix)?,
                checked_u64_product("projection table bytes", 16, suffix)?,
            ],
        )?;

        let build = match strategy {
            RegistersClaimStrategy::AliasLinear | RegistersClaimStrategy::DirectLinear => {
                RegistersClaimPhaseWork {
                    half_width_terms: checked_u64_product("linear build terms", 3, rows)?,
                    full_products: checked_u64_product("linear q combination", 2, prefix)?,
                    compulsory_bytes: checked_u64_sum(
                        "linear build bytes",
                        &[
                            checked_u64_product("linear native bytes", 24, rows)?,
                            table_bytes,
                        ],
                    )?,
                }
            }
            RegistersClaimStrategy::CachedCombinedControl => RegistersClaimPhaseWork {
                half_width_terms: checked_u64_product("cached build half terms", 2, rows)?,
                full_products: rows,
                compulsory_bytes: checked_u64_sum(
                    "cached build bytes",
                    &[
                        checked_u64_product("cached row bytes", 40, rows)?,
                        table_bytes,
                    ],
                )?,
            },
        };

        let (fold, host_full_products) = match strategy {
            RegistersClaimStrategy::AliasLinear => (
                RegistersClaimPhaseWork {
                    half_width_terms: rows,
                    full_products: 0,
                    compulsory_bytes: checked_u64_sum(
                        "alias fold bytes",
                        &[
                            checked_u64_product("alias native bytes", 8, rows)?,
                            table_bytes,
                        ],
                    )?,
                },
                checked_u64_sum(
                    "alias host products",
                    &[
                        checked_u64_product("prefix host products", 4, prefix)?,
                        checked_u64_product("suffix host products", 8, suffix)?,
                    ],
                )?
                .checked_sub(12)
                .ok_or(RegistersClaimPlanError::SizeOverflow {
                    name: "alias host products",
                })?,
            ),
            RegistersClaimStrategy::DirectLinear => (
                RegistersClaimPhaseWork {
                    half_width_terms: checked_u64_product("direct fold terms", 3, rows)?,
                    full_products: 0,
                    compulsory_bytes: checked_u64_sum(
                        "direct fold bytes",
                        &[
                            checked_u64_product("direct native bytes", 24, rows)?,
                            checked_u64_product("direct prefix bytes", 16, prefix)?,
                            checked_u64_product("direct output bytes", 48, suffix)?,
                        ],
                    )?,
                },
                checked_u64_sum(
                    "direct host products",
                    &[
                        checked_u64_product("prefix host products", 4, prefix)?,
                        checked_u64_product("suffix host products", 9, suffix)?,
                    ],
                )?
                .checked_sub(13)
                .ok_or(RegistersClaimPlanError::SizeOverflow {
                    name: "direct host products",
                })?,
            ),
            RegistersClaimStrategy::CachedCombinedControl => (
                RegistersClaimPhaseWork {
                    half_width_terms: 0,
                    full_products: rows,
                    compulsory_bytes: checked_u64_sum(
                        "cached fold bytes",
                        &[
                            checked_u64_product("cached combined bytes", 16, rows)?,
                            table_bytes,
                        ],
                    )?,
                },
                checked_u64_sum(
                    "cached host products",
                    &[
                        checked_u64_product("prefix host products", 4, prefix)?,
                        checked_u64_product("suffix host products", 4, suffix)?,
                    ],
                )?
                .checked_sub(8)
                .ok_or(RegistersClaimPlanError::SizeOverflow {
                    name: "cached host products",
                })?,
            ),
        };

        Ok(RegistersClaimWork {
            build,
            fold,
            host_full_products,
        })
    }

    pub fn resident_projection_work(
        self,
    ) -> Result<RegistersClaimResidentProjectionWork, RegistersClaimPlanError> {
        let rows = self.rows as u64;
        let prefix = self.prefix_elements as u64;
        let suffix = self.suffix_elements as u64;
        let shared_projection = RegistersClaimPhaseWork {
            half_width_terms: checked_u64_product("shared projection terms", 3, rows)?,
            full_products: 0,
            compulsory_bytes: checked_u64_sum(
                "shared projection bytes",
                &[
                    checked_u64_product("shared projection native bytes", 24, rows)?,
                    checked_u64_product("shared projection weight bytes", 16, suffix)?,
                    checked_u64_product("shared projection output bytes", 48, prefix)?,
                ],
            )?,
        };
        let midpoint_fold = RegistersClaimPhaseWork {
            half_width_terms: rows,
            full_products: 0,
            compulsory_bytes: checked_u64_sum(
                "resident midpoint fold bytes",
                &[
                    checked_u64_product("resident midpoint native bytes", 8, rows)?,
                    checked_u64_product("resident midpoint prefix bytes", 16, prefix)?,
                    checked_u64_product("resident midpoint output bytes", 16, suffix)?,
                ],
            )?,
        };
        Ok(RegistersClaimResidentProjectionWork {
            shared_projection,
            midpoint_fold,
            stage1_opening_dot_full_products: checked_u64_product(
                "stage-1 register opening products",
                3,
                prefix,
            )?,
            stage1_opening_dot_bytes: checked_u64_sum(
                "stage-1 register opening bytes",
                &[
                    checked_u64_product("stage-1 component read bytes", 48, prefix)?,
                    checked_u64_product("stage-1 equality read bytes", 16, prefix)?,
                    48,
                ],
            )?,
            displaced_stage1_full_products: checked_u64_sum(
                "displaced stage-1 register products",
                &[
                    checked_u64_product("displaced stage-1 inner products", 3, rows)?,
                    checked_u64_product("displaced stage-1 outer products", 3, suffix)?,
                ],
            )?,
            stage3_q_combine_full_products: checked_u64_product(
                "stage-3 q combination products",
                2,
                prefix,
            )?,
            stage3_q_combine_bytes: checked_u64_product("stage-3 q combination bytes", 64, prefix)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimLinearQStorage {
    pub native_plane_bytes: usize,
    pub resident_input_bytes: usize,
    pub gamma_powers_bytes: usize,
    pub eq_suffix_bytes: usize,
    pub output_bytes: usize,
    pub private_bytes: usize,
    pub total_resident_bytes: usize,
    pub roof_compulsory_bytes: usize,
    pub shader_issued_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimLinearQPlan {
    pub geometry: RegistersClaimGeometry,
    pub config: RegistersClaimKernelConfig,
    pub params: RegistersClaimParams,
    pub storage: RegistersClaimLinearQStorage,
}

impl RegistersClaimLinearQPlan {
    pub fn new(
        rows: usize,
        max_buffer_length: usize,
        config: RegistersClaimKernelConfig,
    ) -> Result<Self, RegistersClaimPlanError> {
        let geometry = RegistersClaimGeometry::new(rows)?;
        let config = config.validate()?;
        let params = geometry.params()?;
        let storage = geometry.linear_q_storage()?;
        for (name, bytes) in [
            ("linear-q native plane", storage.native_plane_bytes),
            ("linear-q gamma powers", storage.gamma_powers_bytes),
            ("linear-q equality suffix", storage.eq_suffix_bytes),
            ("linear-q output", storage.output_bytes),
        ] {
            validate_buffer(name, bytes, max_buffer_length)?;
        }

        Ok(Self {
            geometry,
            config,
            params,
            storage,
        })
    }

    pub const fn useful_threads(self) -> usize {
        self.geometry.prefix_elements()
    }

    pub const fn threads_per_threadgroup(self) -> usize {
        self.config.build_threads_per_threadgroup
    }

    pub fn threadgroups(self) -> usize {
        self.useful_threads()
            .div_ceil(self.threads_per_threadgroup())
    }

    pub fn dispatched_threads(self) -> Result<usize, RegistersClaimPlanError> {
        checked_product(
            "linear-q dispatched threads",
            self.threadgroups(),
            self.threads_per_threadgroup(),
        )
    }

    pub fn work(self) -> Result<RegistersClaimPhaseWork, RegistersClaimPlanError> {
        Ok(self
            .geometry
            .work(RegistersClaimStrategy::AliasLinear)?
            .build)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimStorage {
    pub native_plane_bytes: usize,
    pub native_planes_bytes: usize,
    pub combined_control_bytes: usize,
    pub prefix_field_bytes: usize,
    pub suffix_field_bytes: usize,
    pub partial_q_handoff_bytes: usize,
    pub alias_snapshot_bytes: usize,
    pub direct_dense_bytes: usize,
    pub alias_peak_bytes: usize,
    pub direct_peak_bytes: usize,
    pub cached_control_peak_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistersClaimStrategy {
    AliasLinear,
    DirectLinear,
    CachedCombinedControl,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimPlan {
    pub geometry: RegistersClaimGeometry,
    pub config: RegistersClaimKernelConfig,
    pub params: RegistersClaimParams,
    pub storage: RegistersClaimStorage,
    pub strategy: RegistersClaimStrategy,
}

impl RegistersClaimPlan {
    pub fn new(
        rows: usize,
        max_buffer_length: usize,
        config: RegistersClaimKernelConfig,
        strategy: RegistersClaimStrategy,
    ) -> Result<Self, RegistersClaimPlanError> {
        let geometry = RegistersClaimGeometry::new(rows)?;
        let config = config.validate()?;
        let params = geometry.params()?;
        let storage = geometry.storage()?;

        validate_buffer(
            "native value plane",
            storage.native_plane_bytes,
            max_buffer_length,
        )?;
        match strategy {
            RegistersClaimStrategy::AliasLinear => validate_buffer(
                "alias dense snapshot",
                storage.alias_snapshot_bytes,
                max_buffer_length,
            )?,
            RegistersClaimStrategy::DirectLinear => validate_buffer(
                "direct dense outputs",
                storage.direct_dense_bytes,
                max_buffer_length,
            )?,
            RegistersClaimStrategy::CachedCombinedControl => validate_buffer(
                "cached combined control",
                storage.combined_control_bytes,
                max_buffer_length,
            )?,
        }

        Ok(Self {
            geometry,
            config,
            params,
            storage,
            strategy,
        })
    }

    pub const fn build_threads(self) -> usize {
        self.geometry.prefix_elements()
    }

    pub const fn fold_threadgroups(self) -> usize {
        self.geometry.suffix_elements()
    }

    pub fn fold_threadgroup_bytes(self) -> Result<usize, RegistersClaimPlanError> {
        match self.strategy {
            RegistersClaimStrategy::AliasLinear => self.config.alias_fold_threadgroup_bytes(),
            RegistersClaimStrategy::DirectLinear => self.config.direct_fold_threadgroup_bytes(),
            RegistersClaimStrategy::CachedCombinedControl => {
                self.config.alias_fold_threadgroup_bytes()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegistersClaimAdmission {
    pub resident_register_planes: bool,
    pub half_width_promoted: bool,
    pub stage1_partial_q_handoff: bool,
    pub midpoint_alias_handoff: bool,
    pub fair_producer_accounting: bool,
}

impl RegistersClaimAdmission {
    pub const fn admits(self, strategy: RegistersClaimStrategy) -> bool {
        let common = self.resident_register_planes
            && self.half_width_promoted
            && self.fair_producer_accounting;
        match strategy {
            RegistersClaimStrategy::AliasLinear => common && self.midpoint_alias_handoff,
            RegistersClaimStrategy::DirectLinear => common,
            RegistersClaimStrategy::CachedCombinedControl => false,
        }
    }

    pub const fn admits_resident_alias(self) -> bool {
        self.admits(RegistersClaimStrategy::AliasLinear) && self.stage1_partial_q_handoff
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistersClaimBackendChoice {
    Cpu,
    Metal(RegistersClaimStrategy),
    MetalResidentAlias,
}

pub fn choose_hybrid_backend(
    plan: RegistersClaimPlan,
    admission: RegistersClaimAdmission,
    optimized_cpu_ns: u64,
    projected_complete_metal_ns: u64,
) -> RegistersClaimBackendChoice {
    if plan.geometry.log_t() < REGISTERS_CLAIM_INITIAL_METAL_LOG_T
        || !admission.admits(plan.strategy)
        || !meets_speedup(optimized_cpu_ns, projected_complete_metal_ns, 5)
    {
        RegistersClaimBackendChoice::Cpu
    } else {
        RegistersClaimBackendChoice::Metal(plan.strategy)
    }
}

pub fn choose_resident_route_before_stage1(
    geometry: RegistersClaimGeometry,
    admission: RegistersClaimAdmission,
    optimized_cpu_ns: u64,
    projected_complete_metal_ns: u64,
) -> RegistersClaimBackendChoice {
    if geometry.log_t() < REGISTERS_CLAIM_INITIAL_METAL_LOG_T
        || !admission.admits_resident_alias()
        || !meets_speedup(optimized_cpu_ns, projected_complete_metal_ns, 5)
    {
        RegistersClaimBackendChoice::Cpu
    } else {
        RegistersClaimBackendChoice::MetalResidentAlias
    }
}

pub const fn meets_speedup(cpu_ns: u64, metal_ns: u64, multiplier: u64) -> bool {
    match metal_ns.checked_mul(multiplier) {
        Some(scaled) => scaled <= cpu_ns,
        None => false,
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegistersClaimPhaseWork {
    pub half_width_terms: u64,
    pub full_products: u64,
    pub compulsory_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegistersClaimPhaseCeiling {
    pub half_width_floor_ns: u64,
    pub full_product_floor_ns: u64,
    pub arithmetic_floor_ns: u64,
    pub traffic_floor_ns: u64,
    pub roof_floor_ns: u64,
    pub utilization_cap_ns: u64,
}

impl RegistersClaimPhaseWork {
    pub fn calibrated_ceiling(
        self,
        rates: RegistersClaimRoofRates,
        utilization_percent: u64,
    ) -> Result<RegistersClaimPhaseCeiling, RegistersClaimPlanError> {
        let _ = rates.validate()?;
        if !(1..=100).contains(&utilization_percent) {
            return Err(RegistersClaimPlanError::InvalidUtilization(
                utilization_percent,
            ));
        }
        let half_width_floor_ns =
            rate_ns(self.half_width_terms, rates.half_width_terms_per_second)?;
        let full_product_floor_ns = rate_ns(self.full_products, rates.full_products_per_second)?;
        let arithmetic_floor_ns = half_width_floor_ns
            .checked_add(full_product_floor_ns)
            .ok_or(RegistersClaimPlanError::SizeOverflow {
                name: "phase arithmetic time",
            })?;
        let traffic_floor_ns = rate_ns(self.compulsory_bytes, rates.copy_bytes_per_second)?;
        let roof_floor_ns = arithmetic_floor_ns.max(traffic_floor_ns);
        let utilization_cap_ns = u64::try_from(div_ceil_u128(
            u128::from(roof_floor_ns) * 100,
            u128::from(utilization_percent),
        ))
        .map_err(|_| RegistersClaimPlanError::SizeOverflow {
            name: "phase utilization cap",
        })?;
        Ok(RegistersClaimPhaseCeiling {
            half_width_floor_ns,
            full_product_floor_ns,
            arithmetic_floor_ns,
            traffic_floor_ns,
            roof_floor_ns,
            utilization_cap_ns,
        })
    }

    pub fn roof_floor_ns(
        self,
        rates: RegistersClaimRoofRates,
    ) -> Result<u64, RegistersClaimPlanError> {
        Ok(self.calibrated_ceiling(rates, 100)?.roof_floor_ns)
    }

    pub fn utilization_cap_ns(
        self,
        rates: RegistersClaimRoofRates,
        utilization_percent: u64,
    ) -> Result<u64, RegistersClaimPlanError> {
        Ok(self
            .calibrated_ceiling(rates, utilization_percent)?
            .utilization_cap_ns)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegistersClaimWork {
    pub build: RegistersClaimPhaseWork,
    pub fold: RegistersClaimPhaseWork,
    pub host_full_products: u64,
}

impl RegistersClaimWork {
    pub fn gpu_active_floor_ns(
        self,
        rates: RegistersClaimRoofRates,
    ) -> Result<u64, RegistersClaimPlanError> {
        self.build
            .roof_floor_ns(rates)?
            .checked_add(self.fold.roof_floor_ns(rates)?)
            .ok_or(RegistersClaimPlanError::SizeOverflow {
                name: "GPU-active floor",
            })
    }

    pub fn gpu_active_utilization_cap_ns(
        self,
        rates: RegistersClaimRoofRates,
        utilization_percent: u64,
    ) -> Result<u64, RegistersClaimPlanError> {
        self.build
            .utilization_cap_ns(rates, utilization_percent)?
            .checked_add(self.fold.utilization_cap_ns(rates, utilization_percent)?)
            .ok_or(RegistersClaimPlanError::SizeOverflow {
                name: "GPU-active utilization cap",
            })
    }

    pub fn projected_complete_ns(
        self,
        rates: RegistersClaimRoofRates,
        fixed_producer_host_wait_ns: u64,
    ) -> Result<u64, RegistersClaimPlanError> {
        self.gpu_active_floor_ns(rates)?
            .checked_add(fixed_producer_host_wait_ns)
            .ok_or(RegistersClaimPlanError::SizeOverflow {
                name: "complete projected time",
            })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegistersClaimResidentProjectionWork {
    pub shared_projection: RegistersClaimPhaseWork,
    pub midpoint_fold: RegistersClaimPhaseWork,
    pub stage1_opening_dot_full_products: u64,
    pub stage1_opening_dot_bytes: u64,
    pub displaced_stage1_full_products: u64,
    pub stage3_q_combine_full_products: u64,
    pub stage3_q_combine_bytes: u64,
}

impl RegistersClaimResidentProjectionWork {
    pub fn charged_gpu_floor_ns(
        self,
        rates: RegistersClaimRoofRates,
    ) -> Result<u64, RegistersClaimPlanError> {
        self.shared_projection
            .roof_floor_ns(rates)?
            .checked_add(self.midpoint_fold.roof_floor_ns(rates)?)
            .ok_or(RegistersClaimPlanError::SizeOverflow {
                name: "resident charged GPU time",
            })
    }

    pub fn host_full_products(self) -> Result<u64, RegistersClaimPlanError> {
        self.stage1_opening_dot_full_products
            .checked_add(self.stage3_q_combine_full_products)
            .ok_or(RegistersClaimPlanError::SizeOverflow {
                name: "resident host products",
            })
    }

    pub fn stage3_incremental_gpu_floor_ns(
        self,
        rates: RegistersClaimRoofRates,
    ) -> Result<u64, RegistersClaimPlanError> {
        self.midpoint_fold.roof_floor_ns(rates)
    }

    pub fn displaced_stage1_floor_ns(
        self,
        rates: RegistersClaimRoofRates,
    ) -> Result<u64, RegistersClaimPlanError> {
        let _ = rates.validate()?;
        rate_ns(
            self.displaced_stage1_full_products,
            rates.full_products_per_second,
        )
    }

    pub fn projection_path_logical_bytes(self) -> Result<u64, RegistersClaimPlanError> {
        checked_u64_sum(
            "resident projection-path bytes",
            &[
                self.shared_projection.compulsory_bytes,
                self.midpoint_fold.compulsory_bytes,
                self.stage1_opening_dot_bytes,
                self.stage3_q_combine_bytes,
            ],
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimRoofRates {
    pub copy_bytes_per_second: u64,
    pub half_width_terms_per_second: u64,
    pub full_products_per_second: u64,
}

impl RegistersClaimRoofRates {
    pub const CONSERVATIVE: Self = Self {
        copy_bytes_per_second: REGISTERS_CLAIM_COPY_BYTES_PER_SECOND,
        half_width_terms_per_second: REGISTERS_CLAIM_HALF_WIDTH_FLOOR_PER_SECOND,
        full_products_per_second: REGISTERS_CLAIM_CONSERVATIVE_FULL_PRODUCTS_PER_SECOND,
    };

    pub fn validate(self) -> Result<Self, RegistersClaimPlanError> {
        for (name, value) in [
            ("copy bytes", self.copy_bytes_per_second),
            ("half-width terms", self.half_width_terms_per_second),
            ("full products", self.full_products_per_second),
        ] {
            if value == 0 {
                return Err(RegistersClaimPlanError::ZeroRate { name });
            }
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RegisterValuePlanes<'a> {
    rd_write: &'a [u64],
    rs1: &'a [u64],
    rs2: &'a [u64],
}

impl<'a> RegisterValuePlanes<'a> {
    pub fn new(
        geometry: RegistersClaimGeometry,
        rd_write_value: &'a [u64],
        rs1_value: &'a [u64],
        rs2_value: &'a [u64],
    ) -> Result<Self, RegistersClaimPlanError> {
        for (name, values) in [
            ("rd_write_value", rd_write_value),
            ("rs1_value", rs1_value),
            ("rs2_value", rs2_value),
        ] {
            if values.len() != geometry.rows() {
                return Err(RegistersClaimPlanError::WrongPlaneLength {
                    name,
                    expected: geometry.rows(),
                    actual: values.len(),
                });
            }
        }
        Ok(Self {
            rd_write: rd_write_value,
            rs1: rs1_value,
            rs2: rs2_value,
        })
    }

    pub fn row(self, index: usize) -> [u64; REGISTERS_CLAIM_OUTPUT_COLUMNS] {
        [self.rd_write[index], self.rs1[index], self.rs2[index]]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistersClaimPrefixTables<F> {
    pub p: Vec<F>,
    pub q: Vec<F>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistersClaimLinearComponents<F> {
    pub rd_write_value: Vec<F>,
    pub rs1_value: Vec<F>,
    pub rs2_value: Vec<F>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistersClaimPartialQHandoff<F> {
    generation: u64,
    product_tau_low: Vec<F>,
    components: RegistersClaimLinearComponents<F>,
}

impl<F: Field> RegistersClaimPartialQHandoff<F> {
    pub fn new(
        geometry: RegistersClaimGeometry,
        generation: u64,
        product_tau_low: Vec<F>,
        components: RegistersClaimLinearComponents<F>,
    ) -> Result<Self, RegistersClaimOracleError> {
        let _ = split_tau(geometry, &product_tau_low)?;
        let actual = validate_three_tables(
            &components.rd_write_value,
            &components.rs1_value,
            &components.rs2_value,
            "stage-1 partial-q component",
        )?;
        if actual != geometry.prefix_elements() {
            return Err(RegistersClaimOracleError::WrongTableLength {
                name: "stage-1 partial-q component",
                expected: geometry.prefix_elements(),
                actual,
            });
        }
        Ok(Self {
            generation,
            product_tau_low,
            components,
        })
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn product_tau_low(&self) -> &[F] {
        &self.product_tau_low
    }

    pub fn components(&self) -> &RegistersClaimLinearComponents<F> {
        &self.components
    }

    pub fn validate_identity(
        &self,
        expected_generation: u64,
        expected_product_tau_low: &[F],
    ) -> Result<(), RegistersClaimOracleError> {
        if self.generation != expected_generation {
            return Err(RegistersClaimOracleError::PartialQGenerationMismatch {
                expected: expected_generation,
                actual: self.generation,
            });
        }
        if self.product_tau_low != expected_product_tau_low {
            return Err(RegistersClaimOracleError::PartialQPointMismatch);
        }
        Ok(())
    }

    pub fn stage1_register_openings(
        &self,
        geometry: RegistersClaimGeometry,
    ) -> Result<RegistersClaimOutputs<F>, RegistersClaimOracleError> {
        let (_, tau_lo) = split_tau(geometry, &self.product_tau_low)?;
        let e_in = EqPolynomial::<F>::evals(tau_lo, None);
        Ok(RegistersClaimOutputs {
            rd_write_value: dot_table(&e_in, &self.components.rd_write_value)?,
            rs1_value: dot_table(&e_in, &self.components.rs1_value)?,
            rs2_value: dot_table(&e_in, &self.components.rs2_value)?,
        })
    }

    pub fn stage3_prefix_tables(
        &self,
        geometry: RegistersClaimGeometry,
        expected_generation: u64,
        expected_product_tau_low: &[F],
        gamma: F,
    ) -> Result<RegistersClaimPrefixTables<F>, RegistersClaimOracleError> {
        self.validate_identity(expected_generation, expected_product_tau_low)?;
        let (_, tau_lo) = split_tau(geometry, &self.product_tau_low)?;
        Ok(RegistersClaimPrefixTables {
            p: EqPolynomial::<F>::evals(tau_lo, None),
            q: combine_linear_components(&self.components, gamma)?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistersClaimCachedControl<F> {
    pub combined: Vec<F>,
    pub prefix: RegistersClaimPrefixTables<F>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistersClaimDenseOutputs<F> {
    pub rd_write_value: Vec<F>,
    pub rs1_value: Vec<F>,
    pub rs2_value: Vec<F>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistersClaimAliasDense<F> {
    pub combined: Vec<F>,
    pub rs1_value: Vec<F>,
    pub rs2_value: Vec<F>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimOutputs<F> {
    pub rd_write_value: F,
    pub rs1_value: F,
    pub rs2_value: F,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistersClaimAliasSnapshot<F> {
    pub prefix_challenges: Vec<F>,
    pub rs1_value: Vec<F>,
    pub rs2_value: Vec<F>,
}

impl<F: Field> RegistersClaimAliasSnapshot<F> {
    pub fn new(
        geometry: RegistersClaimGeometry,
        prefix_challenges: Vec<F>,
        rs1_value: Vec<F>,
        rs2_value: Vec<F>,
    ) -> Result<Self, RegistersClaimOracleError> {
        if prefix_challenges.len() != geometry.prefix_vars() {
            return Err(RegistersClaimOracleError::WrongChallengeCount {
                phase: "midpoint alias",
                expected: geometry.prefix_vars(),
                actual: prefix_challenges.len(),
            });
        }
        for (name, actual) in [
            ("midpoint rs1 alias", rs1_value.len()),
            ("midpoint rs2 alias", rs2_value.len()),
        ] {
            if actual != geometry.suffix_elements() {
                return Err(RegistersClaimOracleError::WrongTableLength {
                    name,
                    expected: geometry.suffix_elements(),
                    actual,
                });
            }
        }
        Ok(Self {
            prefix_challenges,
            rs1_value,
            rs2_value,
        })
    }

    pub fn validate_identity(
        &self,
        expected_prefix_challenges: &[F],
    ) -> Result<(), RegistersClaimOracleError> {
        if self.prefix_challenges != expected_prefix_challenges {
            return Err(RegistersClaimOracleError::AliasPrefixMismatch);
        }
        Ok(())
    }
}

pub fn combined_value<F: Field>(values: [u64; 3], gamma: F) -> F {
    let gamma_sq = gamma * gamma;
    F::from_u64(values[0]) + gamma * F::from_u64(values[1]) + gamma_sq * F::from_u64(values[2])
}

pub fn dense_register_openings<F: Field>(
    geometry: RegistersClaimGeometry,
    planes: RegisterValuePlanes<'_>,
    product_tau_low: &[F],
) -> Result<RegistersClaimOutputs<F>, RegistersClaimOracleError> {
    let _ = split_tau(geometry, product_tau_low)?;
    let weights = EqPolynomial::<F>::evals(product_tau_low, None);
    let mut outputs = RegistersClaimOutputs {
        rd_write_value: F::zero(),
        rs1_value: F::zero(),
        rs2_value: F::zero(),
    };
    for (row, weight) in weights.into_iter().enumerate() {
        let values = planes.row(row);
        outputs.rd_write_value += weight * F::from_u64(values[0]);
        outputs.rs1_value += weight * F::from_u64(values[1]);
        outputs.rs2_value += weight * F::from_u64(values[2]);
    }
    Ok(outputs)
}

pub fn build_linear_components<F: Field>(
    geometry: RegistersClaimGeometry,
    planes: RegisterValuePlanes<'_>,
    tau: &[F],
) -> Result<RegistersClaimLinearComponents<F>, RegistersClaimOracleError> {
    let (tau_hi, _) = split_tau(geometry, tau)?;
    let eq_suffix = EqPolynomial::<F>::evals(tau_hi, None);
    let mut components = RegistersClaimLinearComponents {
        rd_write_value: vec![F::zero(); geometry.prefix_elements()],
        rs1_value: vec![F::zero(); geometry.prefix_elements()],
        rs2_value: vec![F::zero(); geometry.prefix_elements()],
    };

    for (x_hi, weight) in eq_suffix.into_iter().enumerate() {
        for x_lo in 0..geometry.prefix_elements() {
            let row = geometry.row_index(x_hi, x_lo)?;
            let values = planes.row(row);
            components.rd_write_value[x_lo] += weight * F::from_u64(values[0]);
            components.rs1_value[x_lo] += weight * F::from_u64(values[1]);
            components.rs2_value[x_lo] += weight * F::from_u64(values[2]);
        }
    }
    Ok(components)
}

pub fn build_partial_q_handoff<F: Field>(
    geometry: RegistersClaimGeometry,
    planes: RegisterValuePlanes<'_>,
    product_tau_low: &[F],
    generation: u64,
) -> Result<RegistersClaimPartialQHandoff<F>, RegistersClaimOracleError> {
    let components = build_linear_components(geometry, planes, product_tau_low)?;
    RegistersClaimPartialQHandoff::new(geometry, generation, product_tau_low.to_vec(), components)
}

pub fn combine_linear_components<F: Field>(
    components: &RegistersClaimLinearComponents<F>,
    gamma: F,
) -> Result<Vec<F>, RegistersClaimOracleError> {
    let length = validate_three_tables(
        &components.rd_write_value,
        &components.rs1_value,
        &components.rs2_value,
        "linear component",
    )?;
    let gamma_sq = gamma * gamma;
    Ok((0..length)
        .map(|index| {
            components.rd_write_value[index]
                + gamma * components.rs1_value[index]
                + gamma_sq * components.rs2_value[index]
        })
        .collect())
}

pub fn build_linear_q<F: Field>(
    geometry: RegistersClaimGeometry,
    planes: RegisterValuePlanes<'_>,
    tau: &[F],
    gamma: F,
) -> Result<RegistersClaimPrefixTables<F>, RegistersClaimOracleError> {
    let (_, tau_lo) = split_tau(geometry, tau)?;
    let components = build_linear_components(geometry, planes, tau)?;
    Ok(RegistersClaimPrefixTables {
        p: EqPolynomial::<F>::evals(tau_lo, None),
        q: combine_linear_components(&components, gamma)?,
    })
}

pub fn build_dense_reference_q<F: Field>(
    geometry: RegistersClaimGeometry,
    planes: RegisterValuePlanes<'_>,
    tau: &[F],
    gamma: F,
) -> Result<RegistersClaimPrefixTables<F>, RegistersClaimOracleError> {
    let (tau_hi, tau_lo) = split_tau(geometry, tau)?;
    let p = EqPolynomial::<F>::evals(tau_lo, None);
    let eq_suffix = EqPolynomial::<F>::evals(tau_hi, None);
    let gamma_sq = gamma * gamma;
    let mut q = vec![F::zero(); geometry.prefix_elements()];
    for (x_hi, weight) in eq_suffix.into_iter().enumerate() {
        for (x_lo, q_value) in q.iter_mut().enumerate() {
            let row = geometry.row_index(x_hi, x_lo)?;
            let [rd_write_value, rs1_value, rs2_value] = planes.row(row);
            let combined = F::from_u64(rd_write_value)
                + gamma * F::from_u64(rs1_value)
                + gamma_sq * F::from_u64(rs2_value);
            *q_value += weight * combined;
        }
    }
    Ok(RegistersClaimPrefixTables { p, q })
}

pub fn build_cached_control<F: Field>(
    geometry: RegistersClaimGeometry,
    planes: RegisterValuePlanes<'_>,
    tau: &[F],
    gamma: F,
) -> Result<RegistersClaimCachedControl<F>, RegistersClaimOracleError> {
    let (tau_hi, tau_lo) = split_tau(geometry, tau)?;
    let p = EqPolynomial::<F>::evals(tau_lo, None);
    let eq_suffix = EqPolynomial::<F>::evals(tau_hi, None);
    let mut combined = vec![F::zero(); geometry.rows()];
    let mut q = vec![F::zero(); geometry.prefix_elements()];
    for (x_hi, weight) in eq_suffix.into_iter().enumerate() {
        for (x_lo, q_value) in q.iter_mut().enumerate() {
            let row = geometry.row_index(x_hi, x_lo)?;
            let value = combined_value(planes.row(row), gamma);
            combined[row] = value;
            *q_value += weight * value;
        }
    }
    Ok(RegistersClaimCachedControl {
        combined,
        prefix: RegistersClaimPrefixTables { p, q },
    })
}

pub fn fold_alias_rd<F: Field>(
    geometry: RegistersClaimGeometry,
    rd_write_value: &[u64],
    prefix_challenges: &[F],
) -> Result<Vec<F>, RegistersClaimOracleError> {
    if rd_write_value.len() != geometry.rows() {
        return Err(RegistersClaimOracleError::WrongNativeLength {
            name: "rd_write_value",
            expected: geometry.rows(),
            actual: rd_write_value.len(),
        });
    }
    let eq_prefix = prefix_equality(geometry, prefix_challenges)?;
    let mut dense = vec![F::zero(); geometry.suffix_elements()];
    for (x_hi, output) in dense.iter_mut().enumerate() {
        let row_start = x_hi * geometry.prefix_elements();
        for (x_lo, weight) in eq_prefix.iter().copied().enumerate() {
            *output += weight * F::from_u64(rd_write_value[row_start + x_lo]);
        }
    }
    Ok(dense)
}

pub fn assemble_alias_dense<F: Field>(
    geometry: RegistersClaimGeometry,
    rd_write_value: Vec<F>,
    aliases: RegistersClaimAliasSnapshot<F>,
    expected_prefix_challenges: &[F],
    gamma: F,
) -> Result<RegistersClaimAliasDense<F>, RegistersClaimOracleError> {
    aliases.validate_identity(expected_prefix_challenges)?;
    if rd_write_value.len() != geometry.suffix_elements() {
        return Err(RegistersClaimOracleError::WrongTableLength {
            name: "midpoint rd dense",
            expected: geometry.suffix_elements(),
            actual: rd_write_value.len(),
        });
    }
    let gamma_sq = gamma * gamma;
    let combined = rd_write_value
        .into_iter()
        .zip(&aliases.rs1_value)
        .zip(&aliases.rs2_value)
        .map(|((rd, rs1), rs2)| rd + gamma * *rs1 + gamma_sq * *rs2)
        .collect();
    Ok(RegistersClaimAliasDense {
        combined,
        rs1_value: aliases.rs1_value,
        rs2_value: aliases.rs2_value,
    })
}

pub fn fold_cached_control<F: Field>(
    geometry: RegistersClaimGeometry,
    combined: &[F],
    prefix_challenges: &[F],
) -> Result<Vec<F>, RegistersClaimOracleError> {
    if combined.len() != geometry.rows() {
        return Err(RegistersClaimOracleError::WrongTableLength {
            name: "combined control",
            expected: geometry.rows(),
            actual: combined.len(),
        });
    }
    let eq_prefix = prefix_equality(geometry, prefix_challenges)?;
    let mut dense = vec![F::zero(); geometry.suffix_elements()];
    for (x_hi, output) in dense.iter_mut().enumerate() {
        let row_start = x_hi * geometry.prefix_elements();
        for (x_lo, weight) in eq_prefix.iter().copied().enumerate() {
            *output += weight * combined[row_start + x_lo];
        }
    }
    Ok(dense)
}

pub fn fold_direct<F: Field>(
    geometry: RegistersClaimGeometry,
    planes: RegisterValuePlanes<'_>,
    prefix_challenges: &[F],
) -> Result<RegistersClaimDenseOutputs<F>, RegistersClaimOracleError> {
    let eq_prefix = prefix_equality(geometry, prefix_challenges)?;
    let mut outputs = RegistersClaimDenseOutputs {
        rd_write_value: vec![F::zero(); geometry.suffix_elements()],
        rs1_value: vec![F::zero(); geometry.suffix_elements()],
        rs2_value: vec![F::zero(); geometry.suffix_elements()],
    };

    for x_hi in 0..geometry.suffix_elements() {
        let row_start = x_hi * geometry.prefix_elements();
        for (x_lo, weight) in eq_prefix.iter().copied().enumerate() {
            let values = planes.row(row_start + x_lo);
            outputs.rd_write_value[x_hi] += weight * F::from_u64(values[0]);
            outputs.rs1_value[x_hi] += weight * F::from_u64(values[1]);
            outputs.rs2_value[x_hi] += weight * F::from_u64(values[2]);
        }
    }
    Ok(outputs)
}

pub fn combine_dense_outputs<F: Field>(
    outputs: &RegistersClaimDenseOutputs<F>,
    gamma: F,
) -> Result<Vec<F>, RegistersClaimOracleError> {
    let length = validate_three_tables(
        &outputs.rd_write_value,
        &outputs.rs1_value,
        &outputs.rs2_value,
        "dense output",
    )?;
    let gamma_sq = gamma * gamma;
    Ok((0..length)
        .map(|index| {
            outputs.rd_write_value[index]
                + gamma * outputs.rs1_value[index]
                + gamma_sq * outputs.rs2_value[index]
        })
        .collect())
}

pub fn suffix_equality<F: Field>(
    geometry: RegistersClaimGeometry,
    tau: &[F],
    prefix_challenges: &[F],
) -> Result<Vec<F>, RegistersClaimOracleError> {
    let (tau_hi, tau_lo) = split_tau(geometry, tau)?;
    let prefix_point = reversed_prefix_point(geometry, prefix_challenges)?;
    let scale = EqPolynomial::<F>::mle(&prefix_point, tau_lo);
    Ok(EqPolynomial::<F>::evals(tau_hi, Some(scale)))
}

pub fn round_endpoints<F: Field>(
    left: &[F],
    right: &[F],
) -> Result<[F; 2], RegistersClaimOracleError> {
    validate_pair_tables(left, right)?;
    let mut at_zero = F::zero();
    let mut at_two = F::zero();
    for y in 0..left.len() / 2 {
        let left_zero = left[2 * y];
        let left_one = left[2 * y + 1];
        let right_zero = right[2 * y];
        let right_one = right[2 * y + 1];
        at_zero += left_zero * right_zero;
        at_two += (left_one + left_one - left_zero) * (right_one + right_one - right_zero);
    }
    Ok([at_zero, at_two])
}

pub fn round_polynomial<F: Field>(
    previous_claim: F,
    left: &[F],
    right: &[F],
) -> Result<UnivariatePoly<F>, RegistersClaimOracleError> {
    let endpoints = round_endpoints(left, right)?;
    Ok(UnivariatePoly::from_evals_and_hint(
        previous_claim,
        &endpoints,
    ))
}

pub fn bind_table<F: Field>(
    table: &mut Vec<F>,
    challenge: F,
) -> Result<(), RegistersClaimOracleError> {
    if table.len() < 2 || !table.len().is_power_of_two() {
        return Err(RegistersClaimOracleError::InvalidRoundTableLength(
            table.len(),
        ));
    }
    let half = table.len() / 2;
    for y in 0..half {
        let low = table[2 * y];
        table[y] = low + challenge * (table[2 * y + 1] - low);
    }
    table.truncate(half);
    Ok(())
}

pub fn bind_dense_outputs<F: Field>(
    outputs: &mut RegistersClaimDenseOutputs<F>,
    challenge: F,
) -> Result<(), RegistersClaimOracleError> {
    validate_pair_tables(&outputs.rd_write_value, &outputs.rs1_value)?;
    validate_pair_tables(&outputs.rd_write_value, &outputs.rs2_value)?;
    bind_table(&mut outputs.rd_write_value, challenge)?;
    bind_table(&mut outputs.rs1_value, challenge)?;
    bind_table(&mut outputs.rs2_value, challenge)
}

pub fn bind_alias_dense<F: Field>(
    state: &mut RegistersClaimAliasDense<F>,
    challenge: F,
) -> Result<(), RegistersClaimOracleError> {
    validate_pair_tables(&state.combined, &state.rs1_value)?;
    validate_pair_tables(&state.combined, &state.rs2_value)?;
    bind_table(&mut state.combined, challenge)?;
    bind_table(&mut state.rs1_value, challenge)?;
    bind_table(&mut state.rs2_value, challenge)
}

pub fn finalize_alias_dense<F: Field>(
    state: &RegistersClaimAliasDense<F>,
    gamma: F,
) -> Result<RegistersClaimOutputs<F>, RegistersClaimOracleError> {
    for (name, actual) in [
        ("combined final opening", state.combined.len()),
        ("rs1 final alias", state.rs1_value.len()),
        ("rs2 final alias", state.rs2_value.len()),
    ] {
        if actual != 1 {
            return Err(RegistersClaimOracleError::WrongTableLength {
                name,
                expected: 1,
                actual,
            });
        }
    }
    let gamma_sq = gamma * gamma;
    Ok(RegistersClaimOutputs {
        rd_write_value: state.combined[0]
            - gamma * state.rs1_value[0]
            - gamma_sq * state.rs2_value[0],
        rs1_value: state.rs1_value[0],
        rs2_value: state.rs2_value[0],
    })
}

pub fn finalize_outputs<F: Field>(
    combined_opening: F,
    outputs: &RegistersClaimDenseOutputs<F>,
    gamma: F,
) -> Result<RegistersClaimOutputs<F>, RegistersClaimOracleError> {
    for (name, actual) in [
        ("rd final output", outputs.rd_write_value.len()),
        ("rs1 final output", outputs.rs1_value.len()),
        ("rs2 final output", outputs.rs2_value.len()),
    ] {
        if actual != 1 {
            return Err(RegistersClaimOracleError::WrongTableLength {
                name,
                expected: 1,
                actual,
            });
        }
    }
    let result = RegistersClaimOutputs {
        rd_write_value: outputs.rd_write_value[0],
        rs1_value: outputs.rs1_value[0],
        rs2_value: outputs.rs2_value[0],
    };
    if output_combination(result, gamma) != combined_opening {
        return Err(RegistersClaimOracleError::FinalCombinationMismatch);
    }
    Ok(result)
}

pub fn output_combination<F: Field>(outputs: RegistersClaimOutputs<F>, gamma: F) -> F {
    outputs.rd_write_value + gamma * outputs.rs1_value + gamma * gamma * outputs.rs2_value
}

pub fn verifier_output_term<F: Field>(
    tau: &[F],
    bound_challenges: &[F],
    outputs: RegistersClaimOutputs<F>,
    gamma: F,
) -> Result<F, RegistersClaimOracleError> {
    if bound_challenges.len() != tau.len() {
        return Err(RegistersClaimOracleError::WrongChallengeCount {
            phase: "complete sumcheck",
            expected: tau.len(),
            actual: bound_challenges.len(),
        });
    }
    let output_point: Vec<F> = bound_challenges.iter().rev().copied().collect();
    let eq_spartan = EqPolynomial::<F>::mle(&output_point, tau);
    Ok(eq_spartan * output_combination(outputs, gamma))
}

fn prefix_equality<F: Field>(
    geometry: RegistersClaimGeometry,
    prefix_challenges: &[F],
) -> Result<Vec<F>, RegistersClaimOracleError> {
    let prefix_point = reversed_prefix_point(geometry, prefix_challenges)?;
    Ok(EqPolynomial::<F>::evals(&prefix_point, None))
}

fn reversed_prefix_point<F: Field>(
    geometry: RegistersClaimGeometry,
    prefix_challenges: &[F],
) -> Result<Vec<F>, RegistersClaimOracleError> {
    if prefix_challenges.len() != geometry.prefix_vars() {
        return Err(RegistersClaimOracleError::WrongChallengeCount {
            phase: "prefix",
            expected: geometry.prefix_vars(),
            actual: prefix_challenges.len(),
        });
    }
    Ok(prefix_challenges.iter().rev().copied().collect())
}

fn split_tau<F>(
    geometry: RegistersClaimGeometry,
    tau: &[F],
) -> Result<(&[F], &[F]), RegistersClaimOracleError> {
    if tau.len() != geometry.log_t() {
        return Err(RegistersClaimOracleError::WrongChallengeCount {
            phase: "tau",
            expected: geometry.log_t(),
            actual: tau.len(),
        });
    }
    Ok(tau.split_at(geometry.suffix_vars()))
}

fn validate_three_tables<F>(
    first: &[F],
    second: &[F],
    third: &[F],
    name: &'static str,
) -> Result<usize, RegistersClaimOracleError> {
    if second.len() != first.len() {
        return Err(RegistersClaimOracleError::WrongTableLength {
            name,
            expected: first.len(),
            actual: second.len(),
        });
    }
    if third.len() != first.len() {
        return Err(RegistersClaimOracleError::WrongTableLength {
            name,
            expected: first.len(),
            actual: third.len(),
        });
    }
    Ok(first.len())
}

fn validate_pair_tables<F>(left: &[F], right: &[F]) -> Result<(), RegistersClaimOracleError> {
    if left.len() != right.len() {
        return Err(RegistersClaimOracleError::MismatchedRoundTables {
            left: left.len(),
            right: right.len(),
        });
    }
    if left.len() < 2 || !left.len().is_power_of_two() {
        return Err(RegistersClaimOracleError::InvalidRoundTableLength(
            left.len(),
        ));
    }
    Ok(())
}

fn dot_table<F: Field>(left: &[F], right: &[F]) -> Result<F, RegistersClaimOracleError> {
    if left.len() != right.len() {
        return Err(RegistersClaimOracleError::MismatchedRoundTables {
            left: left.len(),
            right: right.len(),
        });
    }
    Ok(left
        .iter()
        .zip(right)
        .fold(F::zero(), |sum, (left, right)| sum + *left * *right))
}

fn threadgroup_bytes(
    columns: usize,
    threads: usize,
    phase: &'static str,
) -> Result<usize, RegistersClaimPlanError> {
    let simdgroups = threads / REGISTERS_CLAIM_SIMD_WIDTH;
    let fields = checked_product(phase, columns, simdgroups)?;
    checked_product(phase, fields, 16)
}

fn checked_power_of_two(
    name: &'static str,
    exponent: usize,
) -> Result<usize, RegistersClaimPlanError> {
    let exponent =
        u32::try_from(exponent).map_err(|_| RegistersClaimPlanError::SizeOverflow { name })?;
    1usize
        .checked_shl(exponent)
        .ok_or(RegistersClaimPlanError::SizeOverflow { name })
}

fn checked_bytes(
    name: &'static str,
    elements: usize,
    element_bytes: usize,
) -> Result<usize, RegistersClaimPlanError> {
    checked_product(name, elements, element_bytes)
}

fn checked_product(
    name: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, RegistersClaimPlanError> {
    left.checked_mul(right)
        .ok_or(RegistersClaimPlanError::SizeOverflow { name })
}

fn checked_sum(name: &'static str, terms: &[usize]) -> Result<usize, RegistersClaimPlanError> {
    terms.iter().try_fold(0usize, |sum, term| {
        sum.checked_add(*term)
            .ok_or(RegistersClaimPlanError::SizeOverflow { name })
    })
}

fn checked_u64_product(
    name: &'static str,
    left: u64,
    right: u64,
) -> Result<u64, RegistersClaimPlanError> {
    left.checked_mul(right)
        .ok_or(RegistersClaimPlanError::SizeOverflow { name })
}

fn checked_u64_sum(name: &'static str, terms: &[u64]) -> Result<u64, RegistersClaimPlanError> {
    terms.iter().try_fold(0u64, |sum, term| {
        sum.checked_add(*term)
            .ok_or(RegistersClaimPlanError::SizeOverflow { name })
    })
}

fn abi_count(name: &'static str, value: usize) -> Result<u32, RegistersClaimPlanError> {
    u32::try_from(value).map_err(|_| RegistersClaimPlanError::AbiCountOverflow { name, value })
}

fn validate_buffer(
    name: &'static str,
    bytes: usize,
    max_buffer_length: usize,
) -> Result<(), RegistersClaimPlanError> {
    if bytes > max_buffer_length {
        return Err(RegistersClaimPlanError::BufferTooLarge {
            name,
            bytes,
            max_buffer_length,
        });
    }
    Ok(())
}

fn rate_ns(units: u64, units_per_second: u64) -> Result<u64, RegistersClaimPlanError> {
    if units_per_second == 0 {
        return Err(RegistersClaimPlanError::ZeroRate { name: "roof" });
    }
    let numerator = u128::from(units) * 1_000_000_000;
    let value = div_ceil_u128(numerator, u128::from(units_per_second));
    u64::try_from(value).map_err(|_| RegistersClaimPlanError::SizeOverflow { name: "roof time" })
}

fn div_ceil_u128(numerator: u128, denominator: u128) -> u128 {
    numerator / denominator + (!numerator.is_multiple_of(denominator)) as u128
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum RegistersClaimPlanError {
    #[error(
        "registers claim reduction requires a power-of-two row count of at least two, got {0}"
    )]
    InvalidRows(usize),
    #[error("{name} count {value} does not fit the u32 shader ABI")]
    AbiCountOverflow { name: &'static str, value: usize },
    #[error("{name} size overflow")]
    SizeOverflow { name: &'static str },
    #[error("{name} buffer needs {bytes} bytes but maxBufferLength is {max_buffer_length}")]
    BufferTooLarge {
        name: &'static str,
        bytes: usize,
        max_buffer_length: usize,
    },
    #[error("{phase} threadgroup width {width} is not a nonzero multiple of 32")]
    InvalidThreadgroupWidth { phase: &'static str, width: usize },
    #[error("roof rate for {name} is zero")]
    ZeroRate { name: &'static str },
    #[error("roof utilization must be in 1..=100, got {0}")]
    InvalidUtilization(u64),
    #[error(
        "row coordinate ({x_hi}, {x_lo}) exceeds geometry ({suffix_elements}, {prefix_elements})"
    )]
    CoordinateOutOfRange {
        x_hi: usize,
        x_lo: usize,
        suffix_elements: usize,
        prefix_elements: usize,
    },
    #[error("{name} plane has length {actual}, expected {expected}")]
    WrongPlaneLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum RegistersClaimOracleError {
    #[error(transparent)]
    Plan(#[from] RegistersClaimPlanError),
    #[error("{phase} has {actual} challenges, expected {expected}")]
    WrongChallengeCount {
        phase: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{name} table has length {actual}, expected {expected}")]
    WrongTableLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{name} native plane has length {actual}, expected {expected}")]
    WrongNativeLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("round tables have different lengths: {left} and {right}")]
    MismatchedRoundTables { left: usize, right: usize },
    #[error("round table length must be a power of two of at least two, got {0}")]
    InvalidRoundTableLength(usize),
    #[error("midpoint alias prefix does not match this member's bound challenges")]
    AliasPrefixMismatch,
    #[error("stage-1 partial-q generation is {actual}, expected {expected}")]
    PartialQGenerationMismatch { expected: u64, actual: u64 },
    #[error("stage-1 partial-q point does not match product_tau_low")]
    PartialQPointMismatch,
    #[error("final register openings do not recombine to the bound combined table")]
    FinalCombinationMismatch,
}
