//! Configuration types compatible with current `jolt-core` proof artifacts.

use common::constants::{
    INSTRUCTION_PHASES_THRESHOLD_LOG_T, ONEHOT_CHUNK_THRESHOLD_LOG_T, REGISTER_COUNT, XLEN,
};

pub const LOG_K: usize = XLEN * 2;

pub fn get_instruction_sumcheck_phases(log_t: usize) -> usize {
    if log_t < INSTRUCTION_PHASES_THRESHOLD_LOG_T {
        16
    } else {
        8
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReadWriteConfig {
    pub ram_rw_phase1_num_rounds: u8,
    pub ram_rw_phase2_num_rounds: u8,
    pub registers_rw_phase1_num_rounds: u8,
    pub registers_rw_phase2_num_rounds: u8,
}

impl ReadWriteConfig {
    pub fn validate(&self, log_t: usize, ram_log_k: usize) -> Result<(), String> {
        let log_register_count = REGISTER_COUNT.ilog2() as usize;
        if (self.ram_rw_phase1_num_rounds as usize) > log_t {
            return Err(format!(
                "ram_rw_phase1_num_rounds ({}) exceeds log_T ({log_t})",
                self.ram_rw_phase1_num_rounds
            ));
        }
        if (self.ram_rw_phase2_num_rounds as usize) > ram_log_k {
            return Err(format!(
                "ram_rw_phase2_num_rounds ({}) exceeds log_ram_K ({ram_log_k})",
                self.ram_rw_phase2_num_rounds
            ));
        }
        if (self.registers_rw_phase1_num_rounds as usize) > log_t {
            return Err(format!(
                "registers_rw_phase1_num_rounds ({}) exceeds log_T ({log_t})",
                self.registers_rw_phase1_num_rounds
            ));
        }
        if (self.registers_rw_phase2_num_rounds as usize) > log_register_count {
            return Err(format!(
                "registers_rw_phase2_num_rounds ({}) exceeds log_register_count ({log_register_count})",
                self.registers_rw_phase2_num_rounds
            ));
        }
        Ok(())
    }

    pub fn needs_single_advice_opening(&self, log_t: usize) -> bool {
        self.ram_rw_phase1_num_rounds as usize == log_t
    }
}

impl TryFrom<(usize, usize)> for ReadWriteConfig {
    type Error = String;

    fn try_from((log_t, ram_log_k): (usize, usize)) -> Result<Self, Self::Error> {
        let config = Self {
            ram_rw_phase1_num_rounds: checked_u8(log_t, "log_T")?,
            ram_rw_phase2_num_rounds: checked_u8(ram_log_k, "ram_log_K")?,
            registers_rw_phase1_num_rounds: checked_u8(log_t, "log_T")?,
            registers_rw_phase2_num_rounds: REGISTER_COUNT.ilog2() as u8,
        };
        config.validate(log_t, ram_log_k)?;
        Ok(config)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OneHotConfig {
    pub log_k_chunk: u8,
    pub lookups_ra_virtual_log_k_chunk: u8,
}

impl OneHotConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.log_k_chunk != 4 && self.log_k_chunk != 8 {
            return Err(format!(
                "log_k_chunk ({}) must be either 4 or 8",
                self.log_k_chunk
            ));
        }

        let log_k_chunk = self.log_k_chunk as usize;
        let lookups_chunk = self.lookups_ra_virtual_log_k_chunk as usize;

        if lookups_chunk < log_k_chunk {
            return Err(format!(
                "lookups_ra_virtual_log_k_chunk ({lookups_chunk}) must be >= log_k_chunk ({log_k_chunk})"
            ));
        }

        if lookups_chunk > LOG_K {
            return Err(format!(
                "lookups_ra_virtual_log_k_chunk ({lookups_chunk}) must be <= LOG_K ({LOG_K})"
            ));
        }

        if !lookups_chunk.is_multiple_of(log_k_chunk) {
            return Err(format!(
                "lookups_ra_virtual_log_k_chunk ({lookups_chunk}) must be a multiple of log_k_chunk ({log_k_chunk})"
            ));
        }

        if !LOG_K.is_multiple_of(lookups_chunk) {
            return Err(format!(
                "LOG_K ({LOG_K}) must be divisible by lookups_ra_virtual_log_k_chunk ({lookups_chunk})"
            ));
        }

        Ok(())
    }
}

impl From<usize> for OneHotConfig {
    fn from(log_t: usize) -> Self {
        let log_k_chunk = if log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };
        let lookups_ra_virtual_log_k_chunk = if log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            LOG_K / 8
        } else {
            LOG_K / 4
        };

        Self {
            log_k_chunk: log_k_chunk as u8,
            lookups_ra_virtual_log_k_chunk: lookups_ra_virtual_log_k_chunk as u8,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OneHotParams {
    pub log_k_chunk: usize,
    pub lookups_ra_virtual_log_k_chunk: usize,
    pub k_chunk: usize,

    pub bytecode_k: usize,
    pub ram_k: usize,

    pub instruction_d: usize,
    pub bytecode_d: usize,
    pub ram_d: usize,

    instruction_shifts: Vec<usize>,
    ram_shifts: Vec<usize>,
    bytecode_shifts: Vec<usize>,
}

impl OneHotParams {
    pub fn ram_address_chunk(&self, address: u64, idx: usize) -> Option<u8> {
        let shift = *self.ram_shifts.get(idx)?;
        Some(((address >> shift) & (self.k_chunk - 1) as u64) as u8)
    }

    pub fn bytecode_pc_chunk(&self, pc: usize, idx: usize) -> Option<u8> {
        let shift = *self.bytecode_shifts.get(idx)?;
        Some(((pc >> shift) & (self.k_chunk - 1)) as u8)
    }

    pub fn lookup_index_chunk(&self, index: u128, idx: usize) -> Option<u8> {
        let shift = *self.instruction_shifts.get(idx)?;
        Some(((index >> shift) & (self.k_chunk - 1) as u128) as u8)
    }
}

impl TryFrom<(&OneHotConfig, usize, usize)> for OneHotParams {
    type Error = String;

    fn try_from(
        (config, bytecode_k, ram_k): (&OneHotConfig, usize, usize),
    ) -> Result<Self, Self::Error> {
        config.validate()?;
        if bytecode_k == 0 {
            return Err("bytecode_k must be nonzero".to_owned());
        }
        if ram_k == 0 {
            return Err("ram_k must be nonzero".to_owned());
        }

        let log_k_chunk = config.log_k_chunk as usize;
        let lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk as usize;
        let k_chunk = 1usize
            .checked_shl(log_k_chunk as u32)
            .ok_or_else(|| format!("log_k_chunk ({log_k_chunk}) is too large"))?;

        let instruction_d = LOG_K.div_ceil(log_k_chunk);
        let bytecode_d = ceil_log_2_nonzero(bytecode_k)?.div_ceil(log_k_chunk);
        let ram_d = ceil_log_2_nonzero(ram_k)?.div_ceil(log_k_chunk);

        let instruction_shifts = (0..instruction_d)
            .map(|i| log_k_chunk * (instruction_d - 1 - i))
            .collect();
        let ram_shifts = (0..ram_d).map(|i| log_k_chunk * (ram_d - 1 - i)).collect();
        let bytecode_shifts = (0..bytecode_d)
            .map(|i| log_k_chunk * (bytecode_d - 1 - i))
            .collect();

        Ok(Self {
            log_k_chunk,
            lookups_ra_virtual_log_k_chunk,
            k_chunk,
            bytecode_k,
            ram_k,
            instruction_d,
            bytecode_d,
            ram_d,
            instruction_shifts,
            ram_shifts,
            bytecode_shifts,
        })
    }
}

impl TryFrom<(OneHotConfig, usize, usize)> for OneHotParams {
    type Error = String;

    fn try_from(
        (config, bytecode_k, ram_k): (OneHotConfig, usize, usize),
    ) -> Result<Self, Self::Error> {
        Self::try_from((&config, bytecode_k, ram_k))
    }
}

impl TryFrom<&OneHotParams> for OneHotConfig {
    type Error = String;

    fn try_from(params: &OneHotParams) -> Result<Self, Self::Error> {
        Ok(OneHotConfig {
            log_k_chunk: checked_u8(params.log_k_chunk, "log_k_chunk")?,
            lookups_ra_virtual_log_k_chunk: checked_u8(
                params.lookups_ra_virtual_log_k_chunk,
                "lookups_ra_virtual_log_k_chunk",
            )?,
        })
    }
}

fn checked_u8(value: usize, name: &str) -> Result<u8, String> {
    u8::try_from(value).map_err(|_| format!("{name} ({value}) exceeds u8::MAX ({})", u8::MAX))
}

fn ceil_log_2_nonzero(value: usize) -> Result<usize, String> {
    if value == 0 {
        return Err("log_2 input must be nonzero".to_owned());
    }
    if value.is_power_of_two() {
        Ok(value.ilog2() as usize)
    } else {
        Ok((usize::BITS - value.leading_zeros()) as usize)
    }
}

#[cfg(test)]
mod tests {
    use common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T;

    use super::*;

    #[test]
    fn read_write_config_accepts_default_values() -> Result<(), String> {
        let config = ReadWriteConfig::try_from((20, 16))?;

        assert_eq!(config.ram_rw_phase1_num_rounds, 20);
        assert_eq!(config.ram_rw_phase2_num_rounds, 16);
        assert_eq!(config.registers_rw_phase1_num_rounds, 20);
        assert_eq!(
            config.registers_rw_phase2_num_rounds,
            REGISTER_COUNT.ilog2() as u8
        );
        assert_eq!(config.validate(20, 16), Ok(()));
        assert!(config.needs_single_advice_opening(20));
        assert!(!config.needs_single_advice_opening(21));
        Ok(())
    }

    #[test]
    fn read_write_config_rejects_out_of_range_phase_lengths() {
        assert_eq!(
            ReadWriteConfig {
                ram_rw_phase1_num_rounds: 11,
                ram_rw_phase2_num_rounds: 5,
                registers_rw_phase1_num_rounds: 10,
                registers_rw_phase2_num_rounds: REGISTER_COUNT.ilog2() as u8,
            }
            .validate(10, 5),
            Err("ram_rw_phase1_num_rounds (11) exceeds log_T (10)".to_owned())
        );

        assert_eq!(
            ReadWriteConfig {
                ram_rw_phase1_num_rounds: 10,
                ram_rw_phase2_num_rounds: 6,
                registers_rw_phase1_num_rounds: 10,
                registers_rw_phase2_num_rounds: REGISTER_COUNT.ilog2() as u8,
            }
            .validate(10, 5),
            Err("ram_rw_phase2_num_rounds (6) exceeds log_ram_K (5)".to_owned())
        );

        assert_eq!(
            ReadWriteConfig {
                ram_rw_phase1_num_rounds: 10,
                ram_rw_phase2_num_rounds: 5,
                registers_rw_phase1_num_rounds: 11,
                registers_rw_phase2_num_rounds: REGISTER_COUNT.ilog2() as u8,
            }
            .validate(10, 5),
            Err("registers_rw_phase1_num_rounds (11) exceeds log_T (10)".to_owned())
        );

        assert_eq!(
            ReadWriteConfig {
                ram_rw_phase1_num_rounds: 10,
                ram_rw_phase2_num_rounds: 5,
                registers_rw_phase1_num_rounds: 10,
                registers_rw_phase2_num_rounds: REGISTER_COUNT.ilog2() as u8 + 1,
            }
            .validate(10, 5),
            Err(format!(
                "registers_rw_phase2_num_rounds ({}) exceeds log_register_count ({})",
                REGISTER_COUNT.ilog2() as u8 + 1,
                REGISTER_COUNT.ilog2()
            ))
        );
    }

    #[test]
    fn read_write_config_rejects_constructor_overflow() {
        assert_eq!(
            ReadWriteConfig::try_from((u8::MAX as usize + 1, 1)),
            Err("log_T (256) exceeds u8::MAX (255)".to_owned())
        );
        assert_eq!(
            ReadWriteConfig::try_from((1, u8::MAX as usize + 1)),
            Err("ram_log_K (256) exceeds u8::MAX (255)".to_owned())
        );
    }

    #[test]
    fn read_write_config_accepts_zero_round_edge_case() {
        let config = ReadWriteConfig {
            ram_rw_phase1_num_rounds: 0,
            ram_rw_phase2_num_rounds: 0,
            registers_rw_phase1_num_rounds: 0,
            registers_rw_phase2_num_rounds: 0,
        };

        assert_eq!(config.validate(0, 0), Ok(()));
        assert!(config.needs_single_advice_opening(0));
    }

    #[test]
    fn one_hot_config_accepts_default_values() {
        let small = OneHotConfig::from(ONEHOT_CHUNK_THRESHOLD_LOG_T - 1);
        assert_eq!(small.log_k_chunk, 4);
        assert_eq!(small.lookups_ra_virtual_log_k_chunk, 16);
        assert_eq!(small.validate(), Ok(()));

        let large = OneHotConfig::from(ONEHOT_CHUNK_THRESHOLD_LOG_T);
        assert_eq!(large.log_k_chunk, 8);
        assert_eq!(large.lookups_ra_virtual_log_k_chunk, 32);
        assert_eq!(large.validate(), Ok(()));
    }

    #[test]
    fn instruction_sumcheck_phase_boundary_matches_core_policy() {
        assert_eq!(get_instruction_sumcheck_phases(0), 16);
        assert_eq!(
            get_instruction_sumcheck_phases(INSTRUCTION_PHASES_THRESHOLD_LOG_T - 1),
            16
        );
        assert_eq!(
            get_instruction_sumcheck_phases(INSTRUCTION_PHASES_THRESHOLD_LOG_T),
            8
        );
        assert_eq!(
            get_instruction_sumcheck_phases(INSTRUCTION_PHASES_THRESHOLD_LOG_T + 1),
            8
        );
    }

    #[test]
    fn one_hot_config_accepts_numeric_boundaries() {
        for config in [
            OneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 4,
            },
            OneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 8,
            },
            OneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: LOG_K as u8,
            },
            OneHotConfig {
                log_k_chunk: 8,
                lookups_ra_virtual_log_k_chunk: 8,
            },
            OneHotConfig {
                log_k_chunk: 8,
                lookups_ra_virtual_log_k_chunk: LOG_K as u8,
            },
        ] {
            assert_eq!(config.validate(), Ok(()));
        }
    }

    #[test]
    fn one_hot_config_rejects_invalid_values() {
        assert_eq!(
            OneHotConfig {
                log_k_chunk: 5,
                lookups_ra_virtual_log_k_chunk: 16,
            }
            .validate(),
            Err("log_k_chunk (5) must be either 4 or 8".to_owned())
        );

        assert_eq!(
            OneHotConfig {
                log_k_chunk: 0,
                lookups_ra_virtual_log_k_chunk: 0,
            }
            .validate(),
            Err("log_k_chunk (0) must be either 4 or 8".to_owned())
        );

        assert_eq!(
            OneHotConfig {
                log_k_chunk: 8,
                lookups_ra_virtual_log_k_chunk: 4,
            }
            .validate(),
            Err("lookups_ra_virtual_log_k_chunk (4) must be >= log_k_chunk (8)".to_owned())
        );

        assert_eq!(
            OneHotConfig {
                log_k_chunk: 8,
                lookups_ra_virtual_log_k_chunk: u8::MAX,
            }
            .validate(),
            Err("lookups_ra_virtual_log_k_chunk (255) must be <= LOG_K (128)".to_owned())
        );

        assert_eq!(
            OneHotConfig {
                log_k_chunk: 8,
                lookups_ra_virtual_log_k_chunk: 129,
            }
            .validate(),
            Err("lookups_ra_virtual_log_k_chunk (129) must be <= LOG_K (128)".to_owned())
        );

        assert_eq!(
            OneHotConfig {
                log_k_chunk: 8,
                lookups_ra_virtual_log_k_chunk: 12,
            }
            .validate(),
            Err(
                "lookups_ra_virtual_log_k_chunk (12) must be a multiple of log_k_chunk (8)"
                    .to_owned()
            )
        );

        assert_eq!(
            OneHotConfig {
                log_k_chunk: 8,
                lookups_ra_virtual_log_k_chunk: 24,
            }
            .validate(),
            Err("LOG_K (128) must be divisible by lookups_ra_virtual_log_k_chunk (24)".to_owned())
        );
    }

    #[test]
    fn one_hot_params_reconstructs_derived_values() -> Result<(), String> {
        let config = OneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        };
        let params = OneHotParams::try_from((config, 1024, 4096))?;

        assert_eq!(params.log_k_chunk, 8);
        assert_eq!(params.lookups_ra_virtual_log_k_chunk, 32);
        assert_eq!(params.k_chunk, 256);
        assert_eq!(params.instruction_d, 16);
        assert_eq!(params.bytecode_d, 2);
        assert_eq!(params.ram_d, 2);
        assert_eq!(OneHotConfig::try_from(&params), Ok(config));
        assert_eq!(params.lookup_index_chunk(0x1234, 14), Some(0x12));
        assert_eq!(params.lookup_index_chunk(0x1234, 15), Some(0x34));
        assert_eq!(params.bytecode_pc_chunk(0x1234, 0), Some(0x12));
        assert_eq!(params.bytecode_pc_chunk(0x1234, 1), Some(0x34));
        assert_eq!(params.ram_address_chunk(0x1234, 0), Some(0x12));
        assert_eq!(params.ram_address_chunk(0x1234, 1), Some(0x34));
        Ok(())
    }

    #[test]
    fn one_hot_params_rejects_invalid_numeric_inputs() {
        assert_eq!(
            OneHotParams::try_from((
                OneHotConfig {
                    log_k_chunk: 0,
                    lookups_ra_virtual_log_k_chunk: 0,
                },
                1,
                1,
            )),
            Err("log_k_chunk (0) must be either 4 or 8".to_owned())
        );

        assert_eq!(
            OneHotParams::try_from((
                OneHotConfig {
                    log_k_chunk: 4,
                    lookups_ra_virtual_log_k_chunk: 16,
                },
                0,
                1,
            )),
            Err("bytecode_k must be nonzero".to_owned())
        );

        assert_eq!(
            OneHotParams::try_from((
                OneHotConfig {
                    log_k_chunk: 4,
                    lookups_ra_virtual_log_k_chunk: 16,
                },
                1,
                0,
            )),
            Err("ram_k must be nonzero".to_owned())
        );
    }

    #[test]
    fn one_hot_params_handles_non_power_of_two_sizes() -> Result<(), String> {
        let params = OneHotParams::try_from((
            OneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
            17,
            257,
        ))?;

        assert_eq!(params.bytecode_d, 2);
        assert_eq!(params.ram_d, 3);
        assert_eq!(params.bytecode_pc_chunk(0x123, 0), Some(2));
        assert_eq!(params.bytecode_pc_chunk(0x123, 1), Some(3));
        assert_eq!(params.ram_address_chunk(0x123, 0), Some(1));
        assert_eq!(params.ram_address_chunk(0x123, 1), Some(2));
        assert_eq!(params.ram_address_chunk(0x123, 2), Some(3));
        Ok(())
    }

    #[test]
    fn one_hot_params_chunk_methods_reject_out_of_bounds_indexes() -> Result<(), String> {
        let params = OneHotParams::try_from((
            OneHotConfig {
                log_k_chunk: 8,
                lookups_ra_virtual_log_k_chunk: 32,
            },
            1024,
            4096,
        ))?;

        assert_eq!(params.lookup_index_chunk(0, params.instruction_d), None);
        assert_eq!(params.bytecode_pc_chunk(0, params.bytecode_d), None);
        assert_eq!(params.ram_address_chunk(0, params.ram_d), None);
        Ok(())
    }

    #[test]
    fn one_hot_params_to_config_rejects_truncating_values() {
        let params = OneHotParams {
            log_k_chunk: u8::MAX as usize + 1,
            lookups_ra_virtual_log_k_chunk: 16,
            ..OneHotParams::default()
        };

        assert_eq!(
            OneHotConfig::try_from(&params),
            Err("log_k_chunk (256) exceeds u8::MAX (255)".to_owned())
        );
    }

    #[test]
    fn ceil_log_2_nonzero_rejects_zero_and_handles_boundaries() {
        assert_eq!(
            ceil_log_2_nonzero(0),
            Err("log_2 input must be nonzero".to_owned())
        );
        assert_eq!(ceil_log_2_nonzero(1), Ok(0));
        assert_eq!(ceil_log_2_nonzero(2), Ok(1));
        assert_eq!(ceil_log_2_nonzero(3), Ok(2));
        assert_eq!(ceil_log_2_nonzero(usize::MAX), Ok(usize::BITS as usize));
    }
}
