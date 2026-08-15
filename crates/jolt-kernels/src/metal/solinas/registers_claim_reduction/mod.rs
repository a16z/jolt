//! Checked geometry, roof model, midpoint-alias contract, and scalar oracle for
//! registers claim reduction.

mod runtime;

pub use runtime::*;

use std::mem::{align_of, size_of};

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use thiserror::Error;

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub const REGISTERS_CLAIM_SIMD_WIDTH: usize = 32;
pub const REGISTERS_CLAIM_AKITA_OFFSET: u32 = 0xffff_a7f7;

pub const INSTRUCTION_INPUT_RS1_TABLE: usize = 1;
pub const INSTRUCTION_INPUT_RS2_TABLE: usize = 5;

pub const ALIAS_FOLD_RD_WRITE_VALUE_SLOT: u64 = 0;
pub const ALIAS_FOLD_EQ_PREFIX_SLOT: u64 = 1;
pub const ALIAS_FOLD_OUTPUT_SLOT: u64 = 2;
pub const ALIAS_FOLD_PARAMS_SLOT: u64 = 3;
pub const ALIAS_FOLD_THREADGROUP_SLOT: u64 = 0;

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
    pub fold_threads_per_threadgroup: usize,
}

impl Default for RegistersClaimKernelConfig {
    fn default() -> Self {
        Self {
            fold_threads_per_threadgroup: 128,
        }
    }
}

impl RegistersClaimKernelConfig {
    pub fn validate(self) -> Result<Self, RegistersClaimPlanError> {
        let width = self.fold_threads_per_threadgroup;
        if width == 0 || !width.is_multiple_of(REGISTERS_CLAIM_SIMD_WIDTH) {
            return Err(RegistersClaimPlanError::InvalidThreadgroupWidth {
                phase: "fold",
                width,
            });
        }
        Ok(self)
    }

    pub fn alias_fold_threadgroup_bytes(self) -> Result<usize, RegistersClaimPlanError> {
        let config = self.validate()?;
        threadgroup_bytes(1, config.fold_threads_per_threadgroup, "alias fold")
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

    copy_field_getters! { pub, {
        rows: usize,
        log_t: usize,
        prefix_vars: usize,
        suffix_vars: usize,
        prefix_elements: usize,
        suffix_elements: usize,
    }}

    pub fn params(self) -> Result<RegistersClaimParams, RegistersClaimPlanError> {
        Ok(RegistersClaimParams {
            rows: abi_count("rows", self.rows)?,
            prefix_elements: abi_count("prefix elements", self.prefix_elements)?,
            suffix_elements: abi_count("suffix elements", self.suffix_elements)?,
            reserved: 0,
        })
    }
}

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

    copy_field_getters! { pub, { generation: u64 }}
    ref_field_getters! { pub, {
        product_tau_low: [F],
        components: RegistersClaimLinearComponents<F>,
    }}

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

fn checked_product(
    name: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, RegistersClaimPlanError> {
    left.checked_mul(right)
        .ok_or(RegistersClaimPlanError::SizeOverflow { name })
}

fn abi_count(name: &'static str, value: usize) -> Result<u32, RegistersClaimPlanError> {
    u32::try_from(value).map_err(|_| RegistersClaimPlanError::AbiCountOverflow { name, value })
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
    #[error("{phase} threadgroup width {width} is not a nonzero multiple of 32")]
    InvalidThreadgroupWidth { phase: &'static str, width: usize },
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
