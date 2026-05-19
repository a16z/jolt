use std::{cmp::min, error::Error, fmt, ops::Range};

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use serde::{Deserialize, Serialize};

use super::{
    bytecode::BytecodeReadRafDimensions,
    instruction::{InstructionRaVirtualizationDimensions, InstructionReadRafDimensions},
    ra::JoltRaPolynomialLayout,
    ram::RamRaVirtualizationDimensions,
};

pub const REGISTER_ADDRESS_BITS: usize = 7;
pub const OUTER_UNISKIP_DOMAIN_SIZE: usize = 10;
pub const OUTER_UNISKIP_FIRST_ROUND_DEGREE: usize = 27;
pub const PRODUCT_UNISKIP_DOMAIN_SIZE: usize = 3;
pub const PRODUCT_UNISKIP_FIRST_ROUND_DEGREE: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoltSumcheckDomain {
    BooleanHypercube,
    CenteredInteger { domain_size: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltSumcheckSpec {
    pub domain: JoltSumcheckDomain,
    pub rounds: usize,
    pub degree: usize,
}

impl JoltSumcheckSpec {
    pub const fn boolean(rounds: usize, degree: usize) -> Self {
        Self {
            domain: JoltSumcheckDomain::BooleanHypercube,
            rounds,
            degree,
        }
    }

    pub const fn centered_integer(domain_size: usize, rounds: usize, degree: usize) -> Self {
        Self {
            domain: JoltSumcheckDomain::CenteredInteger { domain_size },
            rounds,
            degree,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TracePolynomialOrder {
    #[default]
    CycleMajor,
    AddressMajor,
}

impl TracePolynomialOrder {
    pub const fn transcript_scalar(self) -> u64 {
        match self {
            Self::CycleMajor => 0,
            Self::AddressMajor => 1,
        }
    }

    pub const fn address_cycle_to_index(
        self,
        address: usize,
        cycle: usize,
        num_addresses: usize,
        num_cycles: usize,
    ) -> usize {
        match self {
            Self::CycleMajor => address * num_cycles + cycle,
            Self::AddressMajor => cycle * num_addresses + address,
        }
    }

    pub const fn index_to_address_cycle(
        self,
        index: usize,
        num_addresses: usize,
        num_cycles: usize,
    ) -> (usize, usize) {
        match self {
            Self::CycleMajor => (index / num_cycles, index % num_cycles),
            Self::AddressMajor => (index % num_addresses, index / num_addresses),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TraceDimensions {
    log_t: usize,
}

impl TraceDimensions {
    pub const fn new(log_t: usize) -> Self {
        Self { log_t }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn sumcheck(self, degree: usize) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t, degree)
    }

    pub fn cycle_opening_point<F: Field>(
        self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        if challenges.len() != self.log_t {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: self.log_t,
                got: challenges.len(),
            });
        }

        Ok(challenges.iter().rev().copied().collect())
    }
}

impl From<usize> for TraceDimensions {
    fn from(log_t: usize) -> Self {
        Self::new(log_t)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ReadWriteDimensions {
    log_t: usize,
    log_k: usize,
    phase1_num_rounds: usize,
    phase2_num_rounds: usize,
}

impl ReadWriteDimensions {
    pub const fn new(
        log_t: usize,
        log_k: usize,
        phase1_num_rounds: usize,
        phase2_num_rounds: usize,
    ) -> Self {
        Self {
            log_t,
            log_k,
            phase1_num_rounds,
            phase2_num_rounds,
        }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn log_k(self) -> usize {
        self.log_k
    }

    pub const fn phase1_num_rounds(self) -> usize {
        self.phase1_num_rounds
    }

    pub const fn phase2_num_rounds(self) -> usize {
        self.phase2_num_rounds
    }

    pub const fn phase3_cycle_rounds(self) -> usize {
        self.log_t - self.phase1_num_rounds
    }

    pub const fn read_write_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t + self.log_k, 3)
    }

    pub const fn raf_evaluation_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t + self.log_k - self.phase1_num_rounds, 2)
    }

    pub const fn output_check_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t + self.log_k - self.phase1_num_rounds, 3)
    }

    pub fn read_write_opening_point<F: Field>(
        self,
        challenges: &[F],
    ) -> Result<ReadWriteOpeningPoint<F>, JoltFormulaPointError> {
        self.validate_phase_split()?;
        let expected = self.log_t + self.log_k;
        if challenges.len() != expected {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected,
                got: challenges.len(),
            });
        }

        let (phase1, rest) = challenges.split_at(self.phase1_num_rounds);
        let (phase2, rest) = rest.split_at(self.phase2_num_rounds);
        let (phase3_cycle, phase3_address) = rest.split_at(self.log_t - self.phase1_num_rounds);

        let r_cycle = phase3_cycle
            .iter()
            .rev()
            .copied()
            .chain(phase1.iter().rev().copied())
            .collect::<Vec<_>>();
        let r_address = phase3_address
            .iter()
            .rev()
            .copied()
            .chain(phase2.iter().rev().copied())
            .collect::<Vec<_>>();
        let opening_point = [r_address.as_slice(), r_cycle.as_slice()].concat();

        Ok(ReadWriteOpeningPoint {
            r_address,
            r_cycle,
            opening_point,
        })
    }

    pub fn address_opening_point<F: Field>(
        self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        self.validate_phase_split()?;
        let cycle_gap_rounds = self.phase3_cycle_rounds();
        let expected = self.log_k + cycle_gap_rounds;
        if challenges.len() != expected {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected,
                got: challenges.len(),
            });
        }

        let phase3_address_start = self.phase2_num_rounds + cycle_gap_rounds;
        let mut address = Vec::with_capacity(self.log_k);
        address.extend_from_slice(&challenges[..self.phase2_num_rounds]);
        address.extend_from_slice(&challenges[phase3_address_start..]);
        address.reverse();
        Ok(address)
    }

    const fn validate_phase_split(self) -> Result<(), JoltFormulaPointError> {
        if self.phase1_num_rounds > self.log_t || self.phase2_num_rounds > self.log_k {
            return Err(JoltFormulaPointError::InvalidReadWritePhaseSplit {
                phase1_num_rounds: self.phase1_num_rounds,
                log_t: self.log_t,
                phase2_num_rounds: self.phase2_num_rounds,
                log_k: self.log_k,
            });
        }
        Ok(())
    }
}

impl From<(usize, usize, usize, usize)> for ReadWriteDimensions {
    fn from(
        (log_t, log_k, phase1_num_rounds, phase2_num_rounds): (usize, usize, usize, usize),
    ) -> Self {
        Self::new(log_t, log_k, phase1_num_rounds, phase2_num_rounds)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadWriteOpeningPoint<F: Field> {
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub opening_point: Vec<F>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum JoltFormulaPointError {
    InvalidReadWritePhaseSplit {
        phase1_num_rounds: usize,
        log_t: usize,
        phase2_num_rounds: usize,
        log_k: usize,
    },
    ChallengeLengthMismatch {
        expected: usize,
        got: usize,
    },
    OpeningPointLengthMismatch {
        expected: usize,
        got: usize,
    },
    EvaluationDomainLengthMismatch {
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for JoltFormulaPointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidReadWritePhaseSplit {
                phase1_num_rounds,
                log_t,
                phase2_num_rounds,
                log_k,
            } => write!(
                f,
                "invalid read-write phase split: phase1 {phase1_num_rounds}/{log_t}, phase2 {phase2_num_rounds}/{log_k}"
            ),
            Self::ChallengeLengthMismatch { expected, got } => {
                write!(
                    f,
                    "challenge length mismatch: expected {expected}, got {got}"
                )
            }
            Self::OpeningPointLengthMismatch { expected, got } => {
                write!(
                    f,
                    "opening point length mismatch: expected {expected}, got {got}"
                )
            }
            Self::EvaluationDomainLengthMismatch { expected, got } => {
                write!(
                    f,
                    "evaluation domain length mismatch: expected {expected}, got {got}"
                )
            }
        }
    }
}

impl Error for JoltFormulaPointError {}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltReadWriteConfig {
    pub ram_rw_phase1_num_rounds: u8,
    pub ram_rw_phase2_num_rounds: u8,
    pub registers_rw_phase1_num_rounds: u8,
    pub registers_rw_phase2_num_rounds: u8,
}

impl JoltReadWriteConfig {
    pub const fn ram_dimensions(self, log_t: usize, ram_log_k: usize) -> ReadWriteDimensions {
        ReadWriteDimensions::new(
            log_t,
            ram_log_k,
            self.ram_rw_phase1_num_rounds as usize,
            self.ram_rw_phase2_num_rounds as usize,
        )
    }

    pub const fn register_dimensions(
        self,
        log_t: usize,
        register_log_k: usize,
    ) -> ReadWriteDimensions {
        ReadWriteDimensions::new(
            log_t,
            register_log_k,
            self.registers_rw_phase1_num_rounds as usize,
            self.registers_rw_phase2_num_rounds as usize,
        )
    }

    pub const fn needs_single_advice_opening(self, log_t: usize) -> bool {
        self.ram_rw_phase1_num_rounds as usize == log_t
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CommitmentMatrixShape {
    column_vars: usize,
    row_vars: usize,
}

impl CommitmentMatrixShape {
    pub const fn new(column_vars: usize, row_vars: usize) -> Self {
        Self {
            column_vars,
            row_vars,
        }
    }

    pub const fn column_vars(self) -> usize {
        self.column_vars
    }

    pub const fn row_vars(self) -> usize {
        self.row_vars
    }

    pub const fn total_vars(self) -> usize {
        self.column_vars + self.row_vars
    }

    pub const fn balanced(total_vars: usize) -> Self {
        let column_vars = total_vars.div_ceil(2);
        Self {
            column_vars,
            row_vars: total_vars - column_vars,
        }
    }

    pub fn advice_from_max_bytes(max_advice_size_bytes: usize) -> Self {
        let words = max_advice_size_bytes / 8;
        let len = words.next_power_of_two().max(1);
        Self::balanced(log2_power_of_two(len))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdviceClaimReductionLayout {
    trace_order: TracePolynomialOrder,
    log_t: usize,
    log_k_chunk: usize,
    main_shape: CommitmentMatrixShape,
    advice_shape: CommitmentMatrixShape,
    cycle_phase_col_rounds: Range<usize>,
    cycle_phase_row_rounds: Range<usize>,
}

impl AdviceClaimReductionLayout {
    pub fn balanced(
        trace_order: TracePolynomialOrder,
        log_t: usize,
        log_k_chunk: usize,
        max_advice_size_bytes: usize,
    ) -> Self {
        Self::new(
            trace_order,
            log_t,
            log_k_chunk,
            CommitmentMatrixShape::balanced(log_k_chunk + log_t),
            CommitmentMatrixShape::advice_from_max_bytes(max_advice_size_bytes),
        )
    }

    pub fn new(
        trace_order: TracePolynomialOrder,
        log_t: usize,
        log_k_chunk: usize,
        main_shape: CommitmentMatrixShape,
        advice_shape: CommitmentMatrixShape,
    ) -> Self {
        let (cycle_phase_col_rounds, cycle_phase_row_rounds) =
            cycle_phase_round_schedule(trace_order, log_t, log_k_chunk, main_shape, advice_shape);
        Self {
            trace_order,
            log_t,
            log_k_chunk,
            main_shape,
            advice_shape,
            cycle_phase_col_rounds,
            cycle_phase_row_rounds,
        }
    }

    pub const fn trace_order(&self) -> TracePolynomialOrder {
        self.trace_order
    }

    pub const fn log_t(&self) -> usize {
        self.log_t
    }

    pub const fn log_k_chunk(&self) -> usize {
        self.log_k_chunk
    }

    pub const fn main_shape(&self) -> CommitmentMatrixShape {
        self.main_shape
    }

    pub const fn advice_shape(&self) -> CommitmentMatrixShape {
        self.advice_shape
    }

    pub fn cycle_phase_col_rounds(&self) -> Range<usize> {
        self.cycle_phase_col_rounds.clone()
    }

    pub fn cycle_phase_row_rounds(&self) -> Range<usize> {
        self.cycle_phase_row_rounds.clone()
    }

    pub fn active_cycle_phase_rounds(&self) -> usize {
        self.cycle_phase_col_rounds.len() + self.cycle_phase_row_rounds.len()
    }

    pub fn cycle_phase_rounds(&self) -> usize {
        if !self.cycle_phase_row_rounds.is_empty() {
            self.cycle_phase_row_rounds.end - self.cycle_phase_col_rounds.start
        } else {
            self.cycle_phase_col_rounds.len()
        }
    }

    pub fn address_phase_rounds(&self) -> usize {
        self.advice_shape
            .total_vars()
            .saturating_sub(self.active_cycle_phase_rounds())
    }

    pub fn dimensions(&self) -> AdviceClaimReductionDimensions {
        AdviceClaimReductionDimensions::new(self.cycle_phase_rounds(), self.address_phase_rounds())
    }

    pub fn cycle_phase_opening_point<F: Field>(
        &self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        let mut advice_var_challenges = self.cycle_phase_variable_challenges(challenges)?;
        advice_var_challenges.reverse();
        Ok(advice_var_challenges)
    }

    pub fn cycle_phase_variable_challenges<F: Field>(
        &self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        let expected = self.cycle_phase_rounds();
        if challenges.len() != expected {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected,
                got: challenges.len(),
            });
        }

        let mut advice_var_challenges = Vec::with_capacity(self.active_cycle_phase_rounds());
        advice_var_challenges.extend_from_slice(&challenges[self.cycle_phase_col_rounds.clone()]);
        advice_var_challenges.extend_from_slice(&challenges[self.cycle_phase_row_rounds.clone()]);
        Ok(advice_var_challenges)
    }

    pub fn address_phase_opening_point<F: Field>(
        &self,
        cycle_var_challenges: &[F],
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        let expected_cycle = self.active_cycle_phase_rounds();
        if cycle_var_challenges.len() != expected_cycle {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: expected_cycle,
                got: cycle_var_challenges.len(),
            });
        }
        let expected_address = self.address_phase_rounds();
        if challenges.len() != expected_address {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: expected_address,
                got: challenges.len(),
            });
        }

        let mut point = match self.trace_order {
            TracePolynomialOrder::CycleMajor => [cycle_var_challenges, challenges].concat(),
            TracePolynomialOrder::AddressMajor => [challenges, cycle_var_challenges].concat(),
        };
        point.reverse();
        Ok(point)
    }

    pub fn cycle_phase_final_output_scale<F: Field>(
        &self,
        reference_opening_point: &[F],
        challenges: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        let opening_point = self.cycle_phase_opening_point(challenges)?;
        self.final_output_scale_at(reference_opening_point, &opening_point)
    }

    pub fn address_phase_final_output_scale<F: Field>(
        &self,
        reference_opening_point: &[F],
        cycle_var_challenges: &[F],
        challenges: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        let opening_point = self.address_phase_opening_point(cycle_var_challenges, challenges)?;
        self.final_output_scale_at(reference_opening_point, &opening_point)
    }

    fn final_output_scale_at<F: Field>(
        &self,
        reference_opening_point: &[F],
        opening_point: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        if reference_opening_point.len() != opening_point.len() {
            return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
                expected: reference_opening_point.len(),
                got: opening_point.len(),
            });
        }

        Ok(
            EqPolynomial::<F>::mle(opening_point, reference_opening_point)
                * self.dummy_cycle_phase_scale::<F>(),
        )
    }

    fn dummy_cycle_phase_scale<F: Field>(&self) -> F {
        let two_inv = F::from_u64(2).inv_or_zero();
        (0..self.dummy_cycle_phase_rounds()).fold(F::one(), |scale, _| scale * two_inv)
    }

    pub fn dummy_cycle_phase_rounds(&self) -> usize {
        self.cycle_phase_rounds()
            .saturating_sub(self.active_cycle_phase_rounds())
    }
}

fn cycle_phase_round_schedule(
    trace_order: TracePolynomialOrder,
    log_t: usize,
    log_k_chunk: usize,
    main_shape: CommitmentMatrixShape,
    advice_shape: CommitmentMatrixShape,
) -> (Range<usize>, Range<usize>) {
    match trace_order {
        TracePolynomialOrder::CycleMajor => {
            let col_binding_rounds = 0..min(log_t, advice_shape.column_vars());
            let row_binding_rounds = min(log_t, main_shape.column_vars())
                ..min(log_t, main_shape.column_vars() + advice_shape.row_vars());
            (col_binding_rounds, row_binding_rounds)
        }
        TracePolynomialOrder::AddressMajor => {
            let col_binding_rounds = 0..advice_shape.column_vars().saturating_sub(log_k_chunk);
            let row_binding_rounds = main_shape.column_vars().saturating_sub(log_k_chunk)
                ..min(
                    log_t,
                    main_shape.column_vars().saturating_sub(log_k_chunk) + advice_shape.row_vars(),
                );
            (col_binding_rounds, row_binding_rounds)
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AdviceClaimReductionDimensions {
    cycle_phase_rounds: usize,
    address_phase_rounds: usize,
}

impl AdviceClaimReductionDimensions {
    pub const fn new(cycle_phase_rounds: usize, address_phase_rounds: usize) -> Self {
        Self {
            cycle_phase_rounds,
            address_phase_rounds,
        }
    }

    pub const fn cycle_phase_rounds(self) -> usize {
        self.cycle_phase_rounds
    }

    pub const fn address_phase_rounds(self) -> usize {
        self.address_phase_rounds
    }

    pub const fn has_address_phase(self) -> bool {
        self.address_phase_rounds > 0
    }

    pub const fn cycle_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.cycle_phase_rounds, 2)
    }

    pub const fn address_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.address_phase_rounds, 2)
    }
}

impl From<(usize, usize)> for AdviceClaimReductionDimensions {
    fn from((cycle_phase_rounds, address_phase_rounds): (usize, usize)) -> Self {
        Self::new(cycle_phase_rounds, address_phase_rounds)
    }
}

fn log2_power_of_two(value: usize) -> usize {
    debug_assert!(value.is_power_of_two());
    value.trailing_zeros() as usize
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum JoltFormulaDimensionsError {
    Zero {
        name: &'static str,
    },
    Overflow {
        name: &'static str,
    },
    InvalidChunkOrder {
        committed_chunk_bits: usize,
        lookup_virtual_chunk_bits: usize,
    },
    NotDivisible {
        value_name: &'static str,
        value: usize,
        divisor_name: &'static str,
        divisor: usize,
    },
    InvalidPhaseRounds {
        phase1_num_rounds: usize,
        log_t: usize,
    },
}

impl JoltFormulaDimensionsError {
    pub(crate) const fn zero(name: &'static str) -> Self {
        Self::Zero { name }
    }

    pub(crate) const fn overflow(name: &'static str) -> Self {
        Self::Overflow { name }
    }

    pub(crate) const fn not_divisible(
        value_name: &'static str,
        value: usize,
        divisor_name: &'static str,
        divisor: usize,
    ) -> Self {
        Self::NotDivisible {
            value_name,
            value,
            divisor_name,
            divisor,
        }
    }
}

impl fmt::Display for JoltFormulaDimensionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero { name } => write!(f, "{name} must be nonzero"),
            Self::Overflow { name } => write!(f, "{name} overflowed"),
            Self::InvalidChunkOrder {
                committed_chunk_bits,
                lookup_virtual_chunk_bits,
            } => write!(
                f,
                "lookup_virtual_chunk_bits ({lookup_virtual_chunk_bits}) must be >= committed_chunk_bits ({committed_chunk_bits})"
            ),
            Self::NotDivisible {
                value_name,
                value,
                divisor_name,
                divisor,
            } => write!(
                f,
                "{value_name} ({value}) must be divisible by {divisor_name} ({divisor})"
            ),
            Self::InvalidPhaseRounds {
                phase1_num_rounds,
                log_t,
            } => write!(
                f,
                "phase1_num_rounds ({phase1_num_rounds}) must be <= log_t ({log_t})"
            ),
        }
    }
}

impl Error for JoltFormulaDimensionsError {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JoltOneHotDimensions {
    pub log_t: usize,
    pub instruction_address_bits: usize,
    pub bytecode_k: usize,
    pub ram_k: usize,
    pub committed_chunk_bits: usize,
    pub lookup_virtual_chunk_bits: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltOneHotConfig {
    pub log_k_chunk: u8,
    pub lookups_ra_virtual_log_k_chunk: u8,
}

impl JoltOneHotConfig {
    pub const fn committed_chunk_bits(self) -> usize {
        self.log_k_chunk as usize
    }

    pub const fn lookup_virtual_chunk_bits(self) -> usize {
        self.lookups_ra_virtual_log_k_chunk as usize
    }

    pub fn committed_address_chunks<F: Field>(self, r_address: &[F]) -> Vec<Vec<F>> {
        let chunk_bits = self.committed_chunk_bits();
        if chunk_bits == 0 {
            return Vec::new();
        }

        let padding = r_address
            .len()
            .next_multiple_of(chunk_bits)
            .saturating_sub(r_address.len());
        let mut padded = Vec::with_capacity(r_address.len() + padding);
        padded.extend((0..padding).map(|_| F::zero()));
        padded.extend_from_slice(r_address);
        padded
            .chunks(chunk_bits)
            .map(<[F]>::to_vec)
            .collect::<Vec<_>>()
    }

    pub const fn dimensions(
        self,
        log_t: usize,
        instruction_address_bits: usize,
        bytecode_k: usize,
        ram_k: usize,
    ) -> JoltOneHotDimensions {
        JoltOneHotDimensions {
            log_t,
            instruction_address_bits,
            bytecode_k,
            ram_k,
            committed_chunk_bits: self.committed_chunk_bits(),
            lookup_virtual_chunk_bits: self.lookup_virtual_chunk_bits(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JoltFormulaDimensions {
    pub trace: TraceDimensions,
    pub ra_layout: JoltRaPolynomialLayout,
    pub instruction_read_raf: InstructionReadRafDimensions,
    pub instruction_ra_virtualization: InstructionRaVirtualizationDimensions,
    pub bytecode_read_raf: BytecodeReadRafDimensions,
    pub ram_ra_virtualization: RamRaVirtualizationDimensions,
}

impl TryFrom<JoltOneHotDimensions> for JoltFormulaDimensions {
    type Error = JoltFormulaDimensionsError;

    fn try_from(dimensions: JoltOneHotDimensions) -> Result<Self, Self::Error> {
        require_nonzero(
            dimensions.instruction_address_bits,
            "instruction_address_bits",
        )?;
        require_nonzero(dimensions.bytecode_k, "bytecode_k")?;
        require_nonzero(dimensions.ram_k, "ram_k")?;
        require_nonzero(dimensions.committed_chunk_bits, "committed_chunk_bits")?;
        require_nonzero(
            dimensions.lookup_virtual_chunk_bits,
            "lookup_virtual_chunk_bits",
        )?;

        if dimensions.lookup_virtual_chunk_bits < dimensions.committed_chunk_bits {
            return Err(JoltFormulaDimensionsError::InvalidChunkOrder {
                committed_chunk_bits: dimensions.committed_chunk_bits,
                lookup_virtual_chunk_bits: dimensions.lookup_virtual_chunk_bits,
            });
        }

        require_divisible(
            "lookup_virtual_chunk_bits",
            dimensions.lookup_virtual_chunk_bits,
            "committed_chunk_bits",
            dimensions.committed_chunk_bits,
        )?;
        require_divisible(
            "instruction_address_bits",
            dimensions.instruction_address_bits,
            "lookup_virtual_chunk_bits",
            dimensions.lookup_virtual_chunk_bits,
        )?;

        let instruction_address_bits = dimensions.instruction_address_bits;
        let bytecode_log_k = ceil_log_2(dimensions.bytecode_k);
        let ram_log_k = ceil_log_2(dimensions.ram_k);
        let instruction_d = instruction_address_bits.div_ceil(dimensions.committed_chunk_bits);
        let bytecode_d = bytecode_log_k.div_ceil(dimensions.committed_chunk_bits);
        let ram_d = ram_log_k.div_ceil(dimensions.committed_chunk_bits);
        let virtual_instruction_ra_polys =
            instruction_address_bits / dimensions.lookup_virtual_chunk_bits;
        let committed_per_virtual =
            dimensions.lookup_virtual_chunk_bits / dimensions.committed_chunk_bits;

        Ok(Self {
            trace: dimensions.log_t.into(),
            ra_layout: JoltRaPolynomialLayout::try_from((instruction_d, bytecode_d, ram_d))?,
            instruction_read_raf: InstructionReadRafDimensions::try_from((
                dimensions.log_t,
                instruction_address_bits,
                virtual_instruction_ra_polys,
            ))?,
            instruction_ra_virtualization: InstructionRaVirtualizationDimensions::try_from((
                dimensions.log_t,
                virtual_instruction_ra_polys,
                committed_per_virtual,
            ))?,
            bytecode_read_raf: (dimensions.log_t, bytecode_log_k, bytecode_d).into(),
            ram_ra_virtualization: (dimensions.log_t, ram_d).into(),
        })
    }
}

fn require_nonzero(value: usize, name: &'static str) -> Result<(), JoltFormulaDimensionsError> {
    if value == 0 {
        Err(JoltFormulaDimensionsError::zero(name))
    } else {
        Ok(())
    }
}

fn require_divisible(
    value_name: &'static str,
    value: usize,
    divisor_name: &'static str,
    divisor: usize,
) -> Result<(), JoltFormulaDimensionsError> {
    if value.is_multiple_of(divisor) {
        Ok(())
    } else {
        Err(JoltFormulaDimensionsError::not_divisible(
            value_name,
            value,
            divisor_name,
            divisor,
        ))
    }
}

fn ceil_log_2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on unexpected errors")]

    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt, Invertible};

    fn dimensions() -> JoltOneHotDimensions {
        JoltOneHotDimensions {
            log_t: 20,
            instruction_address_bits: 128,
            bytecode_k: 1024,
            ram_k: 4096,
            committed_chunk_bits: 8,
            lookup_virtual_chunk_bits: 32,
        }
    }

    #[test]
    fn derives_all_runtime_formula_dimensions() -> Result<(), JoltFormulaDimensionsError> {
        let dimensions = JoltFormulaDimensions::try_from(dimensions())?;

        assert_eq!(dimensions.ra_layout.instruction(), 16);
        assert_eq!(dimensions.trace.log_t(), 20);
        assert_eq!(dimensions.ra_layout.bytecode(), 2);
        assert_eq!(dimensions.ra_layout.ram(), 2);
        assert_eq!(
            dimensions.instruction_read_raf.sumcheck(),
            JoltSumcheckSpec::boolean(148, 6)
        );
        assert_eq!(dimensions.instruction_read_raf.num_virtual_ra_polys(), 4);
        assert_eq!(
            dimensions
                .instruction_ra_virtualization
                .num_committed_per_virtual(),
            4
        );
        assert_eq!(
            dimensions
                .instruction_ra_virtualization
                .num_committed_ra_polys(),
            16
        );
        assert_eq!(dimensions.bytecode_read_raf.num_committed_ra_polys(), 2);
        assert_eq!(dimensions.ram_ra_virtualization.num_committed_ra_polys(), 2);
        Ok(())
    }

    #[test]
    fn supports_zero_bytecode_and_ram_d() -> Result<(), JoltFormulaDimensionsError> {
        let dimensions = JoltFormulaDimensions::try_from(JoltOneHotDimensions {
            bytecode_k: 1,
            ram_k: 1,
            ..dimensions()
        })?;

        assert_eq!(dimensions.ra_layout.instruction(), 16);
        assert_eq!(dimensions.ra_layout.bytecode(), 0);
        assert_eq!(dimensions.ra_layout.ram(), 0);
        assert_eq!(dimensions.ra_layout.total(), 16);
        assert_eq!(dimensions.bytecode_read_raf.num_committed_ra_polys(), 0);
        assert_eq!(dimensions.ram_ra_virtualization.num_committed_ra_polys(), 0);
        Ok(())
    }

    #[test]
    fn trace_dimensions_normalize_cycle_opening_point() {
        let challenges = [Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];

        assert_eq!(
            TraceDimensions::new(3)
                .cycle_opening_point(&challenges)
                .unwrap_or_else(|err| panic!("cycle point should normalize: {err}")),
            vec![Fr::from_u64(7), Fr::from_u64(5), Fr::from_u64(3)]
        );
    }

    #[test]
    fn trace_polynomial_order_indexes_match_protocol_order() {
        assert_eq!(
            TracePolynomialOrder::CycleMajor.address_cycle_to_index(3, 4, 10, 20),
            64
        );
        assert_eq!(
            TracePolynomialOrder::AddressMajor.address_cycle_to_index(3, 4, 10, 20),
            43
        );
        assert_eq!(
            TracePolynomialOrder::CycleMajor.index_to_address_cycle(64, 10, 20),
            (3, 4)
        );
        assert_eq!(
            TracePolynomialOrder::AddressMajor.index_to_address_cycle(43, 10, 20),
            (3, 4)
        );
        assert_eq!(TracePolynomialOrder::CycleMajor.transcript_scalar(), 0);
        assert_eq!(TracePolynomialOrder::AddressMajor.transcript_scalar(), 1);
    }

    #[test]
    fn commitment_matrix_shapes_follow_balanced_policy() {
        let shape = CommitmentMatrixShape::balanced(13);
        assert_eq!(shape.column_vars(), 7);
        assert_eq!(shape.row_vars(), 6);
        assert_eq!(shape.total_vars(), 13);

        assert_eq!(
            CommitmentMatrixShape::advice_from_max_bytes(64),
            CommitmentMatrixShape::new(2, 1)
        );
        assert_eq!(
            CommitmentMatrixShape::advice_from_max_bytes(0),
            CommitmentMatrixShape::new(0, 0)
        );
    }

    #[test]
    fn advice_layout_extracts_cycle_phase_variables_without_dory_globals() {
        let layout =
            AdviceClaimReductionLayout::balanced(TracePolynomialOrder::CycleMajor, 8, 4, 64);
        let challenges = (1..=7).map(Fr::from_u64).collect::<Vec<_>>();

        assert_eq!(layout.main_shape(), CommitmentMatrixShape::new(6, 6));
        assert_eq!(layout.advice_shape(), CommitmentMatrixShape::new(2, 1));
        assert_eq!(layout.cycle_phase_col_rounds(), 0..2);
        assert_eq!(layout.cycle_phase_row_rounds(), 6..7);
        assert_eq!(layout.cycle_phase_rounds(), 7);
        assert_eq!(layout.active_cycle_phase_rounds(), 3);
        assert_eq!(layout.address_phase_rounds(), 0);
        assert_eq!(layout.dummy_cycle_phase_rounds(), 4);
        assert_eq!(
            layout
                .cycle_phase_variable_challenges(&challenges)
                .unwrap_or_else(|error| panic!("cycle phase variables should extract: {error}")),
            vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(7)]
        );
        assert_eq!(
            layout
                .cycle_phase_opening_point(&challenges)
                .unwrap_or_else(|error| panic!("cycle phase point should normalize: {error}")),
            vec![Fr::from_u64(7), Fr::from_u64(2), Fr::from_u64(1)]
        );
    }

    #[test]
    fn advice_layout_extracts_address_phase_point_without_dory_globals() {
        let layout = AdviceClaimReductionLayout::new(
            TracePolynomialOrder::CycleMajor,
            8,
            4,
            CommitmentMatrixShape::balanced(12),
            CommitmentMatrixShape::balanced(8),
        );
        let cycle_challenges = (1..=8).map(Fr::from_u64).collect::<Vec<_>>();
        let cycle_vars = layout
            .cycle_phase_variable_challenges(&cycle_challenges)
            .unwrap_or_else(|error| panic!("cycle variables should extract: {error}"));
        let address_challenges = [Fr::from_u64(101), Fr::from_u64(102)];

        assert_eq!(layout.cycle_phase_col_rounds(), 0..4);
        assert_eq!(layout.cycle_phase_row_rounds(), 6..8);
        assert_eq!(layout.active_cycle_phase_rounds(), 6);
        assert_eq!(layout.address_phase_rounds(), 2);
        assert_eq!(
            layout
                .address_phase_opening_point(&cycle_vars, &address_challenges)
                .unwrap_or_else(|error| panic!("address phase point should normalize: {error}")),
            vec![
                Fr::from_u64(102),
                Fr::from_u64(101),
                Fr::from_u64(8),
                Fr::from_u64(7),
                Fr::from_u64(4),
                Fr::from_u64(3),
                Fr::from_u64(2),
                Fr::from_u64(1),
            ]
        );
    }

    #[test]
    fn advice_layout_tracks_address_major_cycle_gap() {
        let layout =
            AdviceClaimReductionLayout::balanced(TracePolynomialOrder::AddressMajor, 8, 4, 64);
        let challenges = (1..=3).map(Fr::from_u64).collect::<Vec<_>>();
        let cycle_vars = layout
            .cycle_phase_variable_challenges(&challenges)
            .unwrap_or_else(|error| panic!("cycle variables should extract: {error}"));

        assert_eq!(layout.cycle_phase_col_rounds(), 0..0);
        assert_eq!(layout.cycle_phase_row_rounds(), 2..3);
        assert_eq!(layout.cycle_phase_rounds(), 3);
        assert_eq!(layout.active_cycle_phase_rounds(), 1);
        assert_eq!(layout.address_phase_rounds(), 2);
        assert_eq!(layout.dummy_cycle_phase_rounds(), 2);
        assert_eq!(cycle_vars, vec![Fr::from_u64(3)]);
        assert_eq!(
            layout
                .address_phase_opening_point(&cycle_vars, &[Fr::from_u64(101), Fr::from_u64(102)])
                .unwrap_or_else(|error| panic!("address phase point should normalize: {error}")),
            vec![Fr::from_u64(3), Fr::from_u64(102), Fr::from_u64(101)]
        );
    }

    #[test]
    fn advice_final_output_scale_includes_cycle_phase_dummy_rounds() {
        let layout =
            AdviceClaimReductionLayout::balanced(TracePolynomialOrder::AddressMajor, 8, 4, 64);
        let challenges = [Fr::from_u64(11), Fr::from_u64(12), Fr::from_u64(1)];
        let opening_point = layout
            .cycle_phase_opening_point(&challenges)
            .unwrap_or_else(|error| panic!("cycle phase point should normalize: {error}"));
        let two_inv = Fr::from_u64(2).inv_or_zero();

        assert_eq!(
            layout
                .cycle_phase_final_output_scale(&opening_point, &challenges)
                .unwrap_or_else(|error| panic!("final output scale should compute: {error}")),
            two_inv * two_inv
        );
    }

    #[test]
    fn rejects_zero_dimensions() {
        assert_eq!(
            JoltFormulaDimensions::try_from(JoltOneHotDimensions {
                instruction_address_bits: 0,
                ..dimensions()
            }),
            Err(JoltFormulaDimensionsError::zero("instruction_address_bits"))
        );
        assert_eq!(
            JoltFormulaDimensions::try_from(JoltOneHotDimensions {
                bytecode_k: 0,
                ..dimensions()
            }),
            Err(JoltFormulaDimensionsError::zero("bytecode_k"))
        );
    }

    #[test]
    fn rejects_incompatible_chunks() {
        assert_eq!(
            JoltFormulaDimensions::try_from(JoltOneHotDimensions {
                committed_chunk_bits: 16,
                lookup_virtual_chunk_bits: 8,
                ..dimensions()
            }),
            Err(JoltFormulaDimensionsError::InvalidChunkOrder {
                committed_chunk_bits: 16,
                lookup_virtual_chunk_bits: 8,
            })
        );
        assert_eq!(
            JoltFormulaDimensions::try_from(JoltOneHotDimensions {
                lookup_virtual_chunk_bits: 20,
                ..dimensions()
            }),
            Err(JoltFormulaDimensionsError::not_divisible(
                "lookup_virtual_chunk_bits",
                20,
                "committed_chunk_bits",
                8,
            ))
        );
        assert_eq!(
            JoltFormulaDimensions::try_from(JoltOneHotDimensions {
                lookup_virtual_chunk_bits: 48,
                ..dimensions()
            }),
            Err(JoltFormulaDimensionsError::not_divisible(
                "instruction_address_bits",
                128,
                "lookup_virtual_chunk_bits",
                48,
            ))
        );
    }

    #[test]
    fn read_write_dimensions_normalize_full_opening_point() {
        let dimensions = ReadWriteDimensions::new(4, 3, 1, 2);
        let challenges = (1..=7).map(Fr::from_u64).collect::<Vec<_>>();

        let point = dimensions
            .read_write_opening_point(&challenges)
            .unwrap_or_else(|error| panic!("read-write opening point should evaluate: {error}"));

        assert_eq!(
            point.r_cycle,
            vec![
                Fr::from_u64(6),
                Fr::from_u64(5),
                Fr::from_u64(4),
                Fr::from_u64(1)
            ]
        );
        assert_eq!(
            point.r_address,
            vec![Fr::from_u64(7), Fr::from_u64(3), Fr::from_u64(2)]
        );
        assert_eq!(
            point.opening_point,
            vec![
                Fr::from_u64(7),
                Fr::from_u64(3),
                Fr::from_u64(2),
                Fr::from_u64(6),
                Fr::from_u64(5),
                Fr::from_u64(4),
                Fr::from_u64(1),
            ]
        );
    }

    #[test]
    fn read_write_dimensions_extract_address_opening_point() {
        let dimensions = ReadWriteDimensions::new(4, 3, 1, 2);
        let challenges = (10..=15).map(Fr::from_u64).collect::<Vec<_>>();

        assert_eq!(
            dimensions
                .address_opening_point(&challenges)
                .unwrap_or_else(|error| panic!("address opening point should evaluate: {error}")),
            vec![Fr::from_u64(15), Fr::from_u64(11), Fr::from_u64(10)]
        );
    }

    #[test]
    fn read_write_point_helpers_reject_bad_shapes() {
        let dimensions = ReadWriteDimensions::new(4, 3, 5, 2);
        assert_eq!(
            dimensions.read_write_opening_point::<Fr>(&[]),
            Err(JoltFormulaPointError::InvalidReadWritePhaseSplit {
                phase1_num_rounds: 5,
                log_t: 4,
                phase2_num_rounds: 2,
                log_k: 3,
            })
        );

        let dimensions = ReadWriteDimensions::new(4, 3, 1, 2);
        assert_eq!(
            dimensions.address_opening_point::<Fr>(&[Fr::from_u64(0)]),
            Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: 6,
                got: 1,
            })
        );
    }
}
