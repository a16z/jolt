//! Checked host/device ABI for the isolated registers-value successor.

use core::mem::size_of;

use super::super::PipelineLimits;

pub const FIELD_BYTES: u64 = 16;
pub const REGISTER_ADDRESS_DOMAIN: u32 = 128;
pub const RD_INDEX_ABSENT: u8 = u8::MAX;
pub const PRODUCER_STAGE: u32 = 4;
pub const DIRECT_FIRST_SAMPLES: u64 = 3;
pub const DIRECT_FIRST_SIMD_WIDTH: usize = 32;
const DIRECT_FIRST_MAX_SIMDGROUPS: usize = DIRECT_FIRST_SIMD_WIDTH;

const INPUT_CANONICAL_FIELDS: u32 = 1 << 0;
const INPUT_CYCLE_ORDERED: u32 = 1 << 1;
const INPUT_COMPLETE: u32 = 1 << 2;
const REQUIRED_INPUT_FLAGS: u32 = INPUT_CANONICAL_FIELDS | INPUT_CYCLE_ORDERED | INPUT_COMPLETE;

/// Metadata owned with the two resident input buffers. Buffer handles remain
/// host-side; allocation identities make an accidental copy observable.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValResidentInputAbi {
    pub rows: u64,
    pub rd_inc_bytes: u64,
    pub rd_index_bytes: u64,
    pub device_registry_id: u64,
    pub rd_inc_allocation_id: u64,
    pub rd_index_allocation_id: u64,
    pub generation: u64,
    pub producer_stage: u32,
    pub flags: u32,
}

impl RegistersValResidentInputAbi {
    pub fn new(
        rows: usize,
        device_registry_id: u64,
        rd_inc_allocation_id: usize,
        rd_index_allocation_id: usize,
        generation: u64,
    ) -> Result<Self, RegistersValAbiError> {
        if rows < 4 || !rows.is_power_of_two() {
            return Err(RegistersValAbiError::InvalidRows { got: rows });
        }
        let rows = u64::try_from(rows).map_err(|_| RegistersValAbiError::SizeOverflow)?;
        let rd_inc_bytes = rows
            .checked_mul(FIELD_BYTES)
            .ok_or(RegistersValAbiError::SizeOverflow)?;
        let rd_inc_allocation_id =
            u64::try_from(rd_inc_allocation_id).map_err(|_| RegistersValAbiError::SizeOverflow)?;
        let rd_index_allocation_id = u64::try_from(rd_index_allocation_id)
            .map_err(|_| RegistersValAbiError::SizeOverflow)?;
        let value = Self {
            rows,
            rd_inc_bytes,
            rd_index_bytes: rows,
            device_registry_id,
            rd_inc_allocation_id,
            rd_index_allocation_id,
            generation,
            producer_stage: PRODUCER_STAGE,
            flags: REQUIRED_INPUT_FLAGS,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(self) -> Result<(), RegistersValAbiError> {
        if self.rows < 4 || !self.rows.is_power_of_two() {
            return Err(RegistersValAbiError::InvalidRows {
                got: usize::try_from(self.rows).unwrap_or(usize::MAX),
            });
        }
        let expected_inc = self
            .rows
            .checked_mul(FIELD_BYTES)
            .ok_or(RegistersValAbiError::SizeOverflow)?;
        if self.rd_inc_bytes != expected_inc {
            return Err(RegistersValAbiError::LengthMismatch {
                plane: "rd_inc",
                expected: expected_inc,
                got: self.rd_inc_bytes,
            });
        }
        if self.rd_index_bytes != self.rows {
            return Err(RegistersValAbiError::LengthMismatch {
                plane: "rd_index",
                expected: self.rows,
                got: self.rd_index_bytes,
            });
        }
        if self.device_registry_id == 0
            || self.rd_inc_allocation_id == 0
            || self.rd_index_allocation_id == 0
            || self.rd_inc_allocation_id == self.rd_index_allocation_id
            || self.generation == 0
        {
            return Err(RegistersValAbiError::InvalidIdentity);
        }
        if self.producer_stage != PRODUCER_STAGE || self.flags != REQUIRED_INPUT_FLAGS {
            return Err(RegistersValAbiError::InvalidProducerContract);
        }
        Ok(())
    }
}

/// Device parameters for the flattened three-accumulator first-message
/// experiment. All lengths are elements, not bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValDirectFirstParams {
    cycles: u32,
    pairs: u32,
    lt_lo_length: u32,
    high_blocks: u32,
    threadgroups: u32,
    threads_per_threadgroup: u32,
    address_domain: u32,
    absent_register: u32,
}

impl RegistersValDirectFirstParams {
    pub fn validate(self) -> Result<(), RegistersValAbiError> {
        if self.cycles < 4
            || !self.cycles.is_power_of_two()
            || self.pairs != self.cycles / 2
            || self.lt_lo_length < 2
            || !self.lt_lo_length.is_power_of_two()
            || self.high_blocks == 0
            || !self.high_blocks.is_power_of_two()
            || self.lt_lo_length.checked_mul(self.high_blocks) != Some(self.cycles)
            || self.threadgroups == 0
            || self.threads_per_threadgroup < DIRECT_FIRST_SIMD_WIDTH as u32
            || !self
                .threads_per_threadgroup
                .is_multiple_of(DIRECT_FIRST_SIMD_WIDTH as u32)
            || self.threads_per_threadgroup
                > (DIRECT_FIRST_SIMD_WIDTH * DIRECT_FIRST_MAX_SIMDGROUPS) as u32
            || u64::from(self.threadgroups) * u64::from(self.threads_per_threadgroup)
                > u64::from(u32::MAX)
            || self.address_domain != REGISTER_ADDRESS_DOMAIN
            || self.absent_register != u32::from(RD_INDEX_ABSENT)
        {
            return Err(RegistersValAbiError::InvalidDirectParams);
        }
        Ok(())
    }

    pub const fn cycles(self) -> u32 {
        self.cycles
    }

    pub const fn pairs(self) -> u32 {
        self.pairs
    }

    pub const fn lt_lo_length(self) -> u32 {
        self.lt_lo_length
    }

    pub const fn high_blocks(self) -> u32 {
        self.high_blocks
    }

    pub const fn threadgroups(self) -> u32 {
        self.threadgroups
    }

    pub const fn threads_per_threadgroup(self) -> u32 {
        self.threads_per_threadgroup
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValDirectFirstDispatch {
    pub threadgroups: usize,
    pub threads_per_threadgroup: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValDirectFirstDeviceLimits {
    pub pipeline: PipelineLimits,
    pub max_threadgroup_memory_bytes: u64,
    pub max_buffer_bytes: u64,
}

/// A checked dispatch and its exact allocation requirements. Callers dispatch
/// with the two dimensions returned here; the shader checks the actual values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValDirectFirstLaunch {
    params: RegistersValDirectFirstParams,
    grid_threads: u32,
    partial_fields: u64,
    partial_bytes: u64,
    dynamic_threadgroup_bytes: u64,
    total_threadgroup_bytes: u64,
}

impl RegistersValDirectFirstLaunch {
    pub fn new(
        cycles: usize,
        lt_lo_length: usize,
        high_blocks: usize,
        dispatch: RegistersValDirectFirstDispatch,
        limits: RegistersValDirectFirstDeviceLimits,
    ) -> Result<Self, RegistersValAbiError> {
        if cycles < 4 || !cycles.is_power_of_two() {
            return Err(RegistersValAbiError::InvalidRows { got: cycles });
        }
        if lt_lo_length < 2
            || !lt_lo_length.is_power_of_two()
            || high_blocks == 0
            || !high_blocks.is_power_of_two()
            || lt_lo_length.checked_mul(high_blocks) != Some(cycles)
        {
            return Err(RegistersValAbiError::InvalidSplit {
                cycles,
                lt_lo_length,
                high_blocks,
            });
        }
        if dispatch.threadgroups == 0 {
            return Err(RegistersValAbiError::InvalidThreadgroups);
        }
        if limits.pipeline.thread_execution_width != DIRECT_FIRST_SIMD_WIDTH {
            return Err(RegistersValAbiError::UnsupportedExecutionWidth {
                expected: DIRECT_FIRST_SIMD_WIDTH,
                got: limits.pipeline.thread_execution_width,
            });
        }
        if dispatch.threads_per_threadgroup < DIRECT_FIRST_SIMD_WIDTH
            || !dispatch
                .threads_per_threadgroup
                .is_multiple_of(DIRECT_FIRST_SIMD_WIDTH)
            || dispatch.threads_per_threadgroup
                > DIRECT_FIRST_SIMD_WIDTH * DIRECT_FIRST_MAX_SIMDGROUPS
        {
            return Err(RegistersValAbiError::InvalidThreadgroupWidth {
                got: dispatch.threads_per_threadgroup,
            });
        }
        if dispatch.threads_per_threadgroup > limits.pipeline.max_total_threads_per_threadgroup {
            return Err(RegistersValAbiError::ThreadgroupWidthExceedsPipeline {
                got: dispatch.threads_per_threadgroup,
                maximum: limits.pipeline.max_total_threads_per_threadgroup,
            });
        }

        let threadgroups =
            u32::try_from(dispatch.threadgroups).map_err(|_| RegistersValAbiError::SizeOverflow)?;
        let threads_per_threadgroup = u32::try_from(dispatch.threads_per_threadgroup)
            .map_err(|_| RegistersValAbiError::SizeOverflow)?;
        let grid_threads = dispatch
            .threadgroups
            .checked_mul(dispatch.threads_per_threadgroup)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(RegistersValAbiError::GridThreadsOverflow)?;
        let partial_fields = DIRECT_FIRST_SAMPLES
            .checked_mul(u64::from(threadgroups))
            .ok_or(RegistersValAbiError::SizeOverflow)?;
        let partial_bytes = partial_fields
            .checked_mul(FIELD_BYTES)
            .ok_or(RegistersValAbiError::SizeOverflow)?;
        if partial_bytes > limits.max_buffer_bytes {
            return Err(RegistersValAbiError::PartialBufferTooLong {
                required: partial_bytes,
                maximum: limits.max_buffer_bytes,
            });
        }
        let simdgroups = dispatch.threads_per_threadgroup / DIRECT_FIRST_SIMD_WIDTH;
        let dynamic_threadgroup_bytes = DIRECT_FIRST_SAMPLES
            .checked_mul(u64::try_from(simdgroups).map_err(|_| RegistersValAbiError::SizeOverflow)?)
            .and_then(|value| value.checked_mul(FIELD_BYTES))
            .ok_or(RegistersValAbiError::SizeOverflow)?;
        let total_threadgroup_bytes = limits
            .pipeline
            .static_threadgroup_memory_length
            .checked_add(dynamic_threadgroup_bytes)
            .ok_or(RegistersValAbiError::SizeOverflow)?;
        if total_threadgroup_bytes > limits.max_threadgroup_memory_bytes {
            return Err(RegistersValAbiError::ThreadgroupMemoryTooLong {
                required: total_threadgroup_bytes,
                maximum: limits.max_threadgroup_memory_bytes,
            });
        }

        let params = RegistersValDirectFirstParams {
            cycles: u32::try_from(cycles).map_err(|_| RegistersValAbiError::SizeOverflow)?,
            pairs: u32::try_from(cycles / 2).map_err(|_| RegistersValAbiError::SizeOverflow)?,
            lt_lo_length: u32::try_from(lt_lo_length)
                .map_err(|_| RegistersValAbiError::SizeOverflow)?,
            high_blocks: u32::try_from(high_blocks)
                .map_err(|_| RegistersValAbiError::SizeOverflow)?,
            threadgroups,
            threads_per_threadgroup,
            address_domain: REGISTER_ADDRESS_DOMAIN,
            absent_register: u32::from(RD_INDEX_ABSENT),
        };
        params.validate()?;
        Ok(Self {
            params,
            grid_threads,
            partial_fields,
            partial_bytes,
            dynamic_threadgroup_bytes,
            total_threadgroup_bytes,
        })
    }

    pub const fn params(self) -> RegistersValDirectFirstParams {
        self.params
    }

    pub const fn threadgroups(self) -> usize {
        self.params.threadgroups as usize
    }

    pub const fn threads_per_threadgroup(self) -> usize {
        self.params.threads_per_threadgroup as usize
    }

    pub const fn grid_threads(self) -> u32 {
        self.grid_threads
    }

    pub const fn partial_fields(self) -> u64 {
        self.partial_fields
    }

    pub const fn partial_bytes(self) -> u64 {
        self.partial_bytes
    }

    pub const fn dynamic_threadgroup_bytes(self) -> u64 {
        self.dynamic_threadgroup_bytes
    }

    pub const fn total_threadgroup_bytes(self) -> u64 {
        self.total_threadgroup_bytes
    }

    pub fn validate_dispatch(
        self,
        threadgroups: usize,
        threads_per_threadgroup: usize,
    ) -> Result<(), RegistersValAbiError> {
        if threadgroups != self.threadgroups()
            || threads_per_threadgroup != self.threads_per_threadgroup()
        {
            return Err(RegistersValAbiError::DispatchMismatch {
                expected_threadgroups: self.threadgroups(),
                got_threadgroups: threadgroups,
                expected_threads: self.threads_per_threadgroup(),
                got_threads: threads_per_threadgroup,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistersValAbiError {
    InvalidRows {
        got: usize,
    },
    LengthMismatch {
        plane: &'static str,
        expected: u64,
        got: u64,
    },
    InvalidIdentity,
    InvalidProducerContract,
    InvalidSplit {
        cycles: usize,
        lt_lo_length: usize,
        high_blocks: usize,
    },
    InvalidThreadgroups,
    InvalidThreadgroupWidth {
        got: usize,
    },
    UnsupportedExecutionWidth {
        expected: usize,
        got: usize,
    },
    ThreadgroupWidthExceedsPipeline {
        got: usize,
        maximum: usize,
    },
    GridThreadsOverflow,
    PartialBufferTooLong {
        required: u64,
        maximum: u64,
    },
    ThreadgroupMemoryTooLong {
        required: u64,
        maximum: u64,
    },
    DispatchMismatch {
        expected_threadgroups: usize,
        got_threadgroups: usize,
        expected_threads: usize,
        got_threads: usize,
    },
    InvalidDirectParams,
    SizeOverflow,
}

const _: () = assert!(size_of::<RegistersValResidentInputAbi>() == 64);
const _: () = assert!(size_of::<RegistersValDirectFirstParams>() == 32);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test setup")]
mod tests {
    use super::*;

    fn direct_limits() -> RegistersValDirectFirstDeviceLimits {
        RegistersValDirectFirstDeviceLimits {
            pipeline: PipelineLimits {
                thread_execution_width: 32,
                max_total_threads_per_threadgroup: 1024,
                static_threadgroup_memory_length: 0,
            },
            max_threadgroup_memory_bytes: 32 * 1024,
            max_buffer_bytes: 1 << 30,
        }
    }

    fn target_launch() -> RegistersValDirectFirstLaunch {
        RegistersValDirectFirstLaunch::new(
            1 << 26,
            1 << 13,
            1 << 13,
            RegistersValDirectFirstDispatch {
                threadgroups: 8192,
                threads_per_threadgroup: 128,
            },
            direct_limits(),
        )
        .unwrap()
    }

    #[test]
    fn log_26_resident_abi_is_exact() {
        let abi = RegistersValResidentInputAbi::new(1 << 26, 7, 11, 13, 17).unwrap();
        assert_eq!(abi.rd_inc_bytes, 1_073_741_824);
        assert_eq!(abi.rd_index_bytes, 67_108_864);
        assert_eq!(abi.rd_inc_bytes + abi.rd_index_bytes, 1_140_850_688);
        abi.validate().unwrap();
    }

    #[test]
    fn direct_first_shape_and_scratch_are_exact() {
        let launch = target_launch();
        let params = launch.params();
        assert_eq!(params.cycles(), 1 << 26);
        assert_eq!(params.pairs(), 1 << 25);
        assert_eq!(params.lt_lo_length(), 1 << 13);
        assert_eq!(params.high_blocks(), 1 << 13);
        assert_eq!(params.threadgroups(), 8192);
        assert_eq!(params.threads_per_threadgroup(), 128);
        assert_eq!(launch.grid_threads(), 1_048_576);
        assert_eq!(launch.partial_fields(), 24_576);
        assert_eq!(launch.partial_bytes(), 393_216);
        assert_eq!(launch.dynamic_threadgroup_bytes(), 192);
        assert_eq!(launch.total_threadgroup_bytes(), 192);
        launch.validate_dispatch(8192, 128).unwrap();
    }

    #[test]
    fn identities_must_be_distinct_and_nonzero() {
        assert_eq!(
            RegistersValResidentInputAbi::new(1 << 20, 7, 11, 11, 17),
            Err(RegistersValAbiError::InvalidIdentity)
        );
    }

    #[test]
    fn resident_metadata_validation_fails_closed() {
        let valid = RegistersValResidentInputAbi::new(1 << 20, 7, 11, 13, 17).unwrap();

        let mut candidate = valid;
        candidate.rd_inc_bytes -= FIELD_BYTES;
        assert!(candidate.validate().is_err());
        candidate = valid;
        candidate.rd_index_bytes -= 1;
        assert!(candidate.validate().is_err());
        candidate = valid;
        candidate.device_registry_id = 0;
        assert!(candidate.validate().is_err());
        candidate = valid;
        candidate.generation = 0;
        assert!(candidate.validate().is_err());
        candidate = valid;
        candidate.producer_stage += 1;
        assert!(candidate.validate().is_err());
        candidate = valid;
        candidate.flags ^= INPUT_COMPLETE;
        assert!(candidate.validate().is_err());
    }

    #[test]
    fn direct_params_and_dispatch_validation_fail_closed() {
        let launch = target_launch();
        assert!(launch.validate_dispatch(8191, 128).is_err());
        assert!(launch.validate_dispatch(8192, 64).is_err());

        let valid = launch.params();
        let mut candidate = valid;
        candidate.pairs -= 1;
        assert_eq!(
            candidate.validate(),
            Err(RegistersValAbiError::InvalidDirectParams)
        );
        candidate = valid;
        candidate.threadgroups = 0;
        assert_eq!(
            candidate.validate(),
            Err(RegistersValAbiError::InvalidDirectParams)
        );
        candidate = valid;
        candidate.threads_per_threadgroup = 48;
        assert_eq!(
            candidate.validate(),
            Err(RegistersValAbiError::InvalidDirectParams)
        );
        candidate = valid;
        candidate.address_domain = 127;
        assert_eq!(
            candidate.validate(),
            Err(RegistersValAbiError::InvalidDirectParams)
        );
    }

    #[test]
    fn direct_launch_rejects_pipeline_and_allocation_mismatches() {
        let dispatch = RegistersValDirectFirstDispatch {
            threadgroups: 8192,
            threads_per_threadgroup: 128,
        };
        let mut limits = direct_limits();
        limits.pipeline.thread_execution_width = 64;
        assert!(
            RegistersValDirectFirstLaunch::new(1 << 26, 1 << 13, 1 << 13, dispatch, limits,)
                .is_err()
        );

        limits = direct_limits();
        limits.pipeline.max_total_threads_per_threadgroup = 64;
        assert!(
            RegistersValDirectFirstLaunch::new(1 << 26, 1 << 13, 1 << 13, dispatch, limits,)
                .is_err()
        );

        limits = direct_limits();
        limits.max_buffer_bytes = 393_215;
        assert!(
            RegistersValDirectFirstLaunch::new(1 << 26, 1 << 13, 1 << 13, dispatch, limits,)
                .is_err()
        );

        limits = direct_limits();
        limits.max_threadgroup_memory_bytes = 191;
        assert!(
            RegistersValDirectFirstLaunch::new(1 << 26, 1 << 13, 1 << 13, dispatch, limits,)
                .is_err()
        );
    }

    #[test]
    fn direct_launch_rejects_a_grid_that_overflows_shader_indices() {
        let dispatch = RegistersValDirectFirstDispatch {
            threadgroups: 1 << 24,
            threads_per_threadgroup: 256,
        };
        assert_eq!(
            RegistersValDirectFirstLaunch::new(
                1 << 26,
                1 << 13,
                1 << 13,
                dispatch,
                direct_limits(),
            ),
            Err(RegistersValAbiError::GridThreadsOverflow)
        );
    }
}
