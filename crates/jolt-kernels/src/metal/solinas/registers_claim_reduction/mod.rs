//! Checked geometry, roof model, midpoint-alias contract, and scalar oracle for
//! registers claim reduction.

mod runtime;

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
use jolt_poly::EqPolynomial;
use thiserror::Error;

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub const REGISTERS_CLAIM_SIMD_WIDTH: usize = 32;
pub const REGISTERS_CLAIM_OUTPUT_COLUMNS: usize = 3;
pub const REGISTERS_CLAIM_GAMMA_POWERS: usize = 2;
pub const REGISTERS_CLAIM_TARGET_LOG_T: usize = 26;
pub const REGISTERS_CLAIM_AKITA_OFFSET: u32 = 0xffff_a7f7;

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

pub const ALIAS_FOLD_RD_WRITE_VALUE_SLOT: u64 = 0;
pub const ALIAS_FOLD_EQ_PREFIX_SLOT: u64 = 1;
pub const ALIAS_FOLD_OUTPUT_SLOT: u64 = 2;
pub const ALIAS_FOLD_PARAMS_SLOT: u64 = 3;
pub const ALIAS_FOLD_THREADGROUP_SLOT: u64 = 0;

pub(crate) const BUILD_LINEAR_CANONICAL_PIPELINE: &str =
    "solinas_registers_claim_build_linear_q_canonical";
pub(crate) const DIRECT_FOLD_PIPELINE: &str = "solinas_registers_claim_fold_direct";
pub(crate) const ALIAS_FOLD_PIPELINE: &str = "solinas_registers_claim_fold_alias_rd";

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
}

impl Default for RegistersClaimKernelConfig {
    fn default() -> Self {
        Self {
            build_threads_per_threadgroup: 128,
            fold_threads_per_threadgroup: 128,
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
        Ok(RegistersClaimStorage {
            native_plane_bytes,
            native_planes_bytes,
            prefix_field_bytes,
            suffix_field_bytes,
            partial_q_handoff_bytes,
            alias_snapshot_bytes,
            direct_dense_bytes,
            alias_peak_bytes,
            direct_peak_bytes,
        })
    }

    fn linear_q_work(self) -> Result<RegistersClaimPhaseWork, RegistersClaimPlanError> {
        let rows = self.rows as u64;
        let prefix = self.prefix_elements as u64;

        Ok(RegistersClaimPhaseWork {
            half_width_terms: checked_u64_product("linear build terms", 3, rows)?,
            full_products: checked_u64_product("linear q combination", 2, prefix)?,
        })
    }

    fn fold_work(
        self,
        strategy: RegistersClaimStrategy,
    ) -> Result<RegistersClaimPhaseWork, RegistersClaimPlanError> {
        let rows = self.rows as u64;
        let half_width_terms = match strategy {
            RegistersClaimStrategy::AliasLinear => rows,
            RegistersClaimStrategy::DirectLinear => {
                checked_u64_product("direct fold terms", 3, rows)?
            }
        };

        Ok(RegistersClaimPhaseWork {
            half_width_terms,
            full_products: 0,
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
        self.geometry.linear_q_work()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimStorage {
    pub native_plane_bytes: usize,
    pub native_planes_bytes: usize,
    pub prefix_field_bytes: usize,
    pub suffix_field_bytes: usize,
    pub partial_q_handoff_bytes: usize,
    pub alias_snapshot_bytes: usize,
    pub direct_dense_bytes: usize,
    pub alias_peak_bytes: usize,
    pub direct_peak_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistersClaimStrategy {
    AliasLinear,
    DirectLinear,
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
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegistersClaimPhaseWork {
    pub half_width_terms: u64,
    pub full_products: u64,
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
        if generation == 0 {
            return Err(RegistersClaimOracleError::InvalidPartialQGeneration);
        }
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
pub struct RegistersClaimDenseOutputs<F> {
    pub rd_write_value: Vec<F>,
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
    #[error("stage-1 partial-q generation must be nonzero")]
    InvalidPartialQGeneration,
    #[error("stage-1 partial-q point does not match product_tau_low")]
    PartialQPointMismatch,
    #[error("final register openings do not recombine to the bound combined table")]
    FinalCombinationMismatch,
}
