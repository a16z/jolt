use std::{error::Error, fmt};

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
}

impl From<(usize, usize, usize, usize)> for ReadWriteDimensions {
    fn from(
        (log_t, log_k, phase1_num_rounds, phase2_num_rounds): (usize, usize, usize, usize),
    ) -> Self {
        Self::new(log_t, log_k, phase1_num_rounds, phase2_num_rounds)
    }
}

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
    use super::*;

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
}
