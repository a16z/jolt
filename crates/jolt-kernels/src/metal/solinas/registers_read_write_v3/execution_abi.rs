use core::mem::{align_of, size_of};

use super::super::Fp128;
use super::RegistersRwV3Error;

pub(crate) const SOURCE: &str = include_str!("execution_shader.metal");

pub(crate) const COLUMNS: u32 = 128;
pub(crate) const BLOCK_CYCLES: u32 = 256;
pub(crate) const OFFSET_STRIDE: u32 = 129;
pub(crate) const RAW_OUTER_LENGTH: u32 = 8_192;
pub(crate) const RAW_ROUND_ZERO_INNER_LENGTH: u32 = 4_096;
pub(crate) const MAX_DENSE_OUTER_LENGTH: u32 = 8_192;
pub(crate) const HISTOGRAM_HIGH_LENGTH: u32 = 8_192;
pub(crate) const HISTOGRAM_LOW_LENGTH: u32 = 8_192;
pub(crate) const THREADS_PER_THREADGROUP: u32 = 128;
pub(crate) const REDUCTION_THREADS: u32 = 256;
pub(crate) const SIMD_WIDTH: u32 = 32;
pub(crate) const TARGET_SHARD_LOG_T: u32 = 26;
pub(crate) const TARGET_SHARD_CYCLES: u32 = 1 << TARGET_SHARD_LOG_T;
pub(crate) const MAX_TRACE_LOG_T: u32 = 28;
pub(crate) const RAW_BASIS_LANES: u32 = 3;
pub(crate) const RAW_LOCAL_WEIGHT_LANES: u32 = RAW_BASIS_LANES + 1;
pub(crate) const RAW_BASIS_ONE: u32 = 0;
pub(crate) const RAW_BASIS_GAMMA: u32 = 1;
pub(crate) const RAW_BASIS_GAMMA_SQUARED: u32 = 2;

pub(crate) const RAW_ROUND_ZERO_PIPELINE: &str = "solinas_registers_rw_v3_raw_round_zero";
pub(crate) const RAW_COEFFICIENT_PIPELINE: &str = "solinas_registers_rw_v3_raw_coefficients";
pub(crate) const RAW_REPLAY_PIPELINE: &str = "solinas_registers_rw_v3_raw_replay";
pub(crate) const DENSE_BIND_MESSAGE_PIPELINE: &str = "solinas_registers_rw_v3_dense_bind_message";
pub(crate) const REDUCE_COLUMNS_PIPELINE: &str = "solinas_registers_rw_v3_reduce_columns";
pub(crate) const HISTOGRAM_PIPELINE: &str = "solinas_registers_rw_v3_histogram";

pub(crate) mod round_zero_slot {
    pub(crate) const START_VALUES: u64 = 0;
    pub(crate) const RS1_OFFSETS: u64 = 1;
    pub(crate) const RS1_POSITIONS: u64 = 2;
    pub(crate) const RS2_OFFSETS: u64 = 3;
    pub(crate) const RS2_POSITIONS: u64 = 4;
    pub(crate) const RD_OFFSETS: u64 = 5;
    pub(crate) const RD_POSITIONS: u64 = 6;
    pub(crate) const RD_POST_VALUES: u64 = 7;
    pub(crate) const E_IN: u64 = 8;
    pub(crate) const E_OUT: u64 = 9;
    pub(crate) const PARTIALS: u64 = 10;
    pub(crate) const PARAMS: u64 = 11;
}

pub(crate) mod raw_coefficient_slot {
    pub(crate) const LOCAL_WEIGHTS: u64 = 0;
    pub(crate) const COEFFICIENTS: u64 = 1;
    pub(crate) const PARAMS: u64 = 2;
}

pub(crate) mod raw_replay_slot {
    pub(crate) const START_VALUES: u64 = 0;
    pub(crate) const RS1_OFFSETS: u64 = 1;
    pub(crate) const RS1_POSITIONS: u64 = 2;
    pub(crate) const RS2_OFFSETS: u64 = 3;
    pub(crate) const RS2_POSITIONS: u64 = 4;
    pub(crate) const RD_OFFSETS: u64 = 5;
    pub(crate) const RD_POSITIONS: u64 = 6;
    pub(crate) const RD_POST_VALUES: u64 = 7;
    pub(crate) const COEFFICIENTS: u64 = 8;
    pub(crate) const REPLAY_E_IN: u64 = 9;
    pub(crate) const E_OUT: u64 = 10;
    pub(crate) const CHALLENGE: u64 = 11;
    pub(crate) const INC_SOURCE: u64 = 12;
    pub(crate) const INC_DESTINATION: u64 = 13;
    pub(crate) const PARTIALS: u64 = 14;
    pub(crate) const DENSE_STATE_DESTINATION: u64 = 15;
    pub(crate) const PARAMS: u64 = 16;
}

pub(crate) mod dense_slot {
    pub(crate) const STATE_SOURCE: u64 = 0;
    pub(crate) const INC_SOURCE: u64 = 1;
    pub(crate) const E_IN: u64 = 2;
    pub(crate) const E_OUT: u64 = 3;
    pub(crate) const CHALLENGE: u64 = 4;
    pub(crate) const STATE_DESTINATION: u64 = 5;
    pub(crate) const INC_DESTINATION: u64 = 6;
    pub(crate) const PARTIALS: u64 = 7;
    pub(crate) const PARAMS: u64 = 8;
}

pub(crate) mod reduction_slot {
    pub(crate) const INPUT: u64 = 0;
    pub(crate) const OUTPUT: u64 = 1;
    pub(crate) const PARAMS: u64 = 2;
}

pub(crate) mod histogram_slot {
    pub(crate) const RS1_OFFSETS: u64 = 0;
    pub(crate) const RS1_POSITIONS: u64 = 1;
    pub(crate) const RS2_OFFSETS: u64 = 2;
    pub(crate) const RS2_POSITIONS: u64 = 3;
    pub(crate) const E_HI: u64 = 4;
    pub(crate) const E_LO: u64 = 5;
    pub(crate) const PARTIALS: u64 = 6;
    pub(crate) const PARAMS: u64 = 7;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BufferAccess {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BufferBinding {
    pub(crate) slot: u64,
    pub(crate) name: &'static str,
    pub(crate) access: BufferAccess,
}

const fn read(slot: u64, name: &'static str) -> BufferBinding {
    BufferBinding {
        slot,
        name,
        access: BufferAccess::Read,
    }
}

const fn write(slot: u64, name: &'static str) -> BufferBinding {
    BufferBinding {
        slot,
        name,
        access: BufferAccess::Write,
    }
}

pub(crate) const ROUND_ZERO_BINDINGS: [BufferBinding; 12] = [
    read(round_zero_slot::START_VALUES, "start_values"),
    read(round_zero_slot::RS1_OFFSETS, "rs1_offsets"),
    read(round_zero_slot::RS1_POSITIONS, "rs1_positions"),
    read(round_zero_slot::RS2_OFFSETS, "rs2_offsets"),
    read(round_zero_slot::RS2_POSITIONS, "rs2_positions"),
    read(round_zero_slot::RD_OFFSETS, "rd_offsets"),
    read(round_zero_slot::RD_POSITIONS, "rd_positions"),
    read(round_zero_slot::RD_POST_VALUES, "rd_post_values"),
    read(round_zero_slot::E_IN, "e_in"),
    read(round_zero_slot::E_OUT, "e_out"),
    write(round_zero_slot::PARTIALS, "partials"),
    read(round_zero_slot::PARAMS, "params"),
];

pub(crate) const RAW_COEFFICIENT_BINDINGS: [BufferBinding; 3] = [
    read(raw_coefficient_slot::LOCAL_WEIGHTS, "local_weights"),
    write(raw_coefficient_slot::COEFFICIENTS, "coefficients"),
    read(raw_coefficient_slot::PARAMS, "params"),
];

pub(crate) const RAW_REPLAY_BINDINGS: [BufferBinding; 17] = [
    read(raw_replay_slot::START_VALUES, "start_values"),
    read(raw_replay_slot::RS1_OFFSETS, "rs1_offsets"),
    read(raw_replay_slot::RS1_POSITIONS, "rs1_positions"),
    read(raw_replay_slot::RS2_OFFSETS, "rs2_offsets"),
    read(raw_replay_slot::RS2_POSITIONS, "rs2_positions"),
    read(raw_replay_slot::RD_OFFSETS, "rd_offsets"),
    read(raw_replay_slot::RD_POSITIONS, "rd_positions"),
    read(raw_replay_slot::RD_POST_VALUES, "rd_post_values"),
    read(raw_replay_slot::COEFFICIENTS, "coefficients"),
    read(raw_replay_slot::REPLAY_E_IN, "replay_e_in"),
    read(raw_replay_slot::E_OUT, "e_out"),
    read(raw_replay_slot::CHALLENGE, "challenge"),
    read(raw_replay_slot::INC_SOURCE, "inc_source"),
    write(raw_replay_slot::INC_DESTINATION, "inc_destination"),
    write(raw_replay_slot::PARTIALS, "partials"),
    write(
        raw_replay_slot::DENSE_STATE_DESTINATION,
        "dense_state_destination",
    ),
    read(raw_replay_slot::PARAMS, "params"),
];

pub(crate) const DENSE_BINDINGS: [BufferBinding; 9] = [
    read(dense_slot::STATE_SOURCE, "state_source"),
    read(dense_slot::INC_SOURCE, "inc_source"),
    read(dense_slot::E_IN, "e_in"),
    read(dense_slot::E_OUT, "e_out"),
    read(dense_slot::CHALLENGE, "challenge"),
    write(dense_slot::STATE_DESTINATION, "state_destination"),
    write(dense_slot::INC_DESTINATION, "inc_destination"),
    write(dense_slot::PARTIALS, "partials"),
    read(dense_slot::PARAMS, "params"),
];

pub(crate) const REDUCTION_BINDINGS: [BufferBinding; 3] = [
    read(reduction_slot::INPUT, "input"),
    write(reduction_slot::OUTPUT, "output"),
    read(reduction_slot::PARAMS, "params"),
];

pub(crate) const HISTOGRAM_BINDINGS: [BufferBinding; 8] = [
    read(histogram_slot::RS1_OFFSETS, "rs1_offsets"),
    read(histogram_slot::RS1_POSITIONS, "rs1_positions"),
    read(histogram_slot::RS2_OFFSETS, "rs2_offsets"),
    read(histogram_slot::RS2_POSITIONS, "rs2_positions"),
    read(histogram_slot::E_HI, "e_hi"),
    read(histogram_slot::E_LO, "e_lo"),
    write(histogram_slot::PARTIALS, "partials"),
    read(histogram_slot::PARAMS, "params"),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PipelineReadiness {
    AbiOnly,
    ExactShader,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PipelineDescriptor {
    pub(crate) name: &'static str,
    pub(crate) readiness: PipelineReadiness,
    pub(crate) threads_per_threadgroup: u32,
    pub(crate) threadgroup_memory_bytes: Option<u32>,
    pub(crate) bindings: &'static [BufferBinding],
}

pub(crate) const RAW_ROUND_ZERO_DESCRIPTOR: PipelineDescriptor = PipelineDescriptor {
    name: RAW_ROUND_ZERO_PIPELINE,
    readiness: PipelineReadiness::AbiOnly,
    threads_per_threadgroup: THREADS_PER_THREADGROUP,
    threadgroup_memory_bytes: None,
    bindings: &ROUND_ZERO_BINDINGS,
};

pub(crate) const RAW_COEFFICIENT_DESCRIPTOR: PipelineDescriptor = PipelineDescriptor {
    name: RAW_COEFFICIENT_PIPELINE,
    readiness: PipelineReadiness::AbiOnly,
    threads_per_threadgroup: REDUCTION_THREADS,
    threadgroup_memory_bytes: None,
    bindings: &RAW_COEFFICIENT_BINDINGS,
};

pub(crate) const RAW_REPLAY_DESCRIPTOR: PipelineDescriptor = PipelineDescriptor {
    name: RAW_REPLAY_PIPELINE,
    readiness: PipelineReadiness::AbiOnly,
    threads_per_threadgroup: THREADS_PER_THREADGROUP,
    threadgroup_memory_bytes: None,
    bindings: &RAW_REPLAY_BINDINGS,
};

pub(crate) const DENSE_DESCRIPTOR: PipelineDescriptor = PipelineDescriptor {
    name: DENSE_BIND_MESSAGE_PIPELINE,
    readiness: PipelineReadiness::ExactShader,
    threads_per_threadgroup: THREADS_PER_THREADGROUP,
    threadgroup_memory_bytes: Some(8 * size_of::<Fp128>() as u32),
    bindings: &DENSE_BINDINGS,
};

pub(crate) const REDUCTION_DESCRIPTOR: PipelineDescriptor = PipelineDescriptor {
    name: REDUCE_COLUMNS_PIPELINE,
    readiness: PipelineReadiness::ExactShader,
    threads_per_threadgroup: REDUCTION_THREADS,
    threadgroup_memory_bytes: Some(0),
    bindings: &REDUCTION_BINDINGS,
};

pub(crate) const HISTOGRAM_DESCRIPTOR: PipelineDescriptor = PipelineDescriptor {
    name: HISTOGRAM_PIPELINE,
    readiness: PipelineReadiness::ExactShader,
    threads_per_threadgroup: THREADS_PER_THREADGROUP,
    threadgroup_memory_bytes: Some(0),
    bindings: &HISTOGRAM_BINDINGS,
};

pub(crate) const PIPELINES: [PipelineDescriptor; 6] = [
    RAW_ROUND_ZERO_DESCRIPTOR,
    RAW_COEFFICIENT_DESCRIPTOR,
    RAW_REPLAY_DESCRIPTOR,
    DENSE_DESCRIPTOR,
    REDUCTION_DESCRIPTOR,
    HISTOGRAM_DESCRIPTOR,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HostSchedule {
    FiatShamirAfterCommand,
    OverlapAddressTailThenJoin,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PhaseDescriptor {
    pub(crate) name: &'static str,
    pub(crate) first_round: Option<u8>,
    pub(crate) last_round: Option<u8>,
    pub(crate) dispatches: u8,
    pub(crate) barriers: u8,
    pub(crate) command_buffers: u8,
    pub(crate) host_schedule: HostSchedule,
}

pub(crate) const PHASES: [PhaseDescriptor; 5] = [
    PhaseDescriptor {
        name: "raw round zero",
        first_round: Some(0),
        last_round: Some(0),
        dispatches: 4,
        barriers: 3,
        command_buffers: 1,
        host_schedule: HostSchedule::FiatShamirAfterCommand,
    },
    PhaseDescriptor {
        name: "raw replay rounds 1-7",
        first_round: Some(1),
        last_round: Some(7),
        dispatches: 35,
        barriers: 28,
        command_buffers: 7,
        host_schedule: HostSchedule::FiatShamirAfterCommand,
    },
    PhaseDescriptor {
        name: "round-8 dense junction",
        first_round: Some(8),
        last_round: Some(8),
        dispatches: 5,
        barriers: 4,
        command_buffers: 1,
        host_schedule: HostSchedule::FiatShamirAfterCommand,
    },
    PhaseDescriptor {
        name: "dense rounds 9-25",
        first_round: Some(9),
        last_round: Some(25),
        dispatches: 50,
        barriers: 33,
        command_buffers: 17,
        host_schedule: HostSchedule::FiatShamirAfterCommand,
    },
    PhaseDescriptor {
        name: "read histograms",
        first_round: None,
        last_round: None,
        dispatches: 4,
        barriers: 3,
        command_buffers: 1,
        host_schedule: HostSchedule::OverlapAddressTailThenJoin,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DispatchGeometry {
    pub(crate) threadgroups: [u64; 3],
    pub(crate) threads_per_threadgroup: [u64; 3],
    pub(crate) dynamic_threadgroup_bytes: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DenseState {
    pub(crate) val: Fp128,
    pub(crate) ra: Fp128,
    pub(crate) wa: Fp128,
}

const _: [(); 48] = [(); size_of::<DenseState>()];
const _: [(); 16] = [(); align_of::<DenseState>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawRoundZeroParams {
    pub(crate) cycles: u32,
    pub(crate) blocks: u32,
    pub(crate) blocks_per_outer: u32,
    pub(crate) e_in_length: u32,
    pub(crate) e_out_length: u32,
    pub(crate) columns: u32,
    pub(crate) offset_stride: u32,
    pub(crate) position_stride: u32,
}

const _: [(); 32] = [(); size_of::<RawRoundZeroParams>()];

impl RawRoundZeroParams {
    pub(crate) const fn target_shard() -> Self {
        Self {
            cycles: TARGET_SHARD_CYCLES,
            blocks: TARGET_SHARD_CYCLES / BLOCK_CYCLES,
            blocks_per_outer: 32,
            e_in_length: RAW_ROUND_ZERO_INNER_LENGTH,
            e_out_length: RAW_OUTER_LENGTH,
            columns: COLUMNS,
            offset_stride: OFFSET_STRIDE,
            position_stride: BLOCK_CYCLES,
        }
    }

    pub(crate) const fn dispatch(self) -> DispatchGeometry {
        DispatchGeometry {
            threadgroups: [self.e_out_length as u64, 1, 1],
            threads_per_threadgroup: [THREADS_PER_THREADGROUP as u64, 1, 1],
            dynamic_threadgroup_bytes: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawCoefficientParams {
    pub(crate) round: u32,
    pub(crate) width: u32,
    pub(crate) basis_weight_fields: u32,
    pub(crate) strict_suffix_fields: u32,
    pub(crate) local_weight_fields: u32,
    pub(crate) coefficient_fields: u32,
    pub(crate) logical_products: u32,
    pub(crate) reserved: u32,
}

const _: [(); 32] = [(); size_of::<RawCoefficientParams>()];

impl RawCoefficientParams {
    /// Local weights are `[B_1, B_gamma, B_gamma_squared, strict_suffix]`;
    /// coefficients are basis-major `Q_b[p, d] = B_b[p] * strict_suffix[d]`.
    pub(crate) fn for_round(round: u32) -> Result<Self, RegistersRwV3Error> {
        validate_raw_round(round)?;
        let width = 1u32 << round;
        let width_squared = width
            .checked_mul(width)
            .ok_or(RegistersRwV3Error::SizeOverflow("raw coefficient width"))?;
        let basis_weight_fields = RAW_BASIS_LANES * width;
        let strict_suffix_fields = width;
        let coefficient_fields = RAW_BASIS_LANES * width_squared;
        Ok(Self {
            round,
            width,
            basis_weight_fields,
            strict_suffix_fields,
            local_weight_fields: RAW_LOCAL_WEIGHT_LANES * width,
            coefficient_fields,
            logical_products: coefficient_fields,
            reserved: 0,
        })
    }

    pub(crate) const fn basis_weight_offset(self, basis: u32) -> Option<u32> {
        if basis < RAW_BASIS_LANES {
            Some(basis * self.width)
        } else {
            None
        }
    }

    pub(crate) const fn strict_suffix_offset(self) -> u32 {
        self.basis_weight_fields
    }

    pub(crate) const fn coefficient_index(self, basis: u32, p: u32, d: u32) -> Option<u32> {
        if basis >= RAW_BASIS_LANES || p >= self.width || d >= self.width {
            return None;
        }
        Some(basis * self.width * self.width + p * self.width + d)
    }

    pub(crate) const fn dispatch(self) -> DispatchGeometry {
        DispatchGeometry {
            threadgroups: [
                self.logical_products.div_ceil(REDUCTION_THREADS) as u64,
                1,
                1,
            ],
            threads_per_threadgroup: [REDUCTION_THREADS as u64, 1, 1],
            dynamic_threadgroup_bytes: 0,
        }
    }
}

pub(crate) const RAW_FLAG_MATERIALIZE_DENSE: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawReplayParams {
    pub(crate) round: u32,
    pub(crate) cycles: u32,
    pub(crate) blocks: u32,
    pub(crate) width: u32,
    pub(crate) remaining_cycles: u32,
    pub(crate) nonempty_pairs: u32,
    pub(crate) replay_e_in_length: u32,
    pub(crate) e_out_length: u32,
    pub(crate) columns: u32,
    pub(crate) offset_stride: u32,
    pub(crate) position_stride: u32,
    pub(crate) flags: u32,
}

const _: [(); 48] = [(); size_of::<RawReplayParams>()];

impl RawReplayParams {
    pub(crate) fn target_shard(
        round: u32,
        nonempty_pairs: u32,
    ) -> Result<Self, RegistersRwV3Error> {
        validate_raw_round(round)?;
        Ok(Self {
            round,
            cycles: TARGET_SHARD_CYCLES,
            blocks: TARGET_SHARD_CYCLES / BLOCK_CYCLES,
            width: 1 << round,
            remaining_cycles: TARGET_SHARD_CYCLES >> round,
            nonempty_pairs,
            replay_e_in_length: 1 << (12 - round),
            e_out_length: RAW_OUTER_LENGTH,
            columns: COLUMNS,
            offset_stride: OFFSET_STRIDE,
            position_stride: BLOCK_CYCLES,
            flags: if round == 8 {
                RAW_FLAG_MATERIALIZE_DENSE
            } else {
                0
            },
        })
    }

    pub(crate) const fn dispatch(self) -> DispatchGeometry {
        DispatchGeometry {
            threadgroups: [self.e_out_length as u64, 1, 1],
            threads_per_threadgroup: [THREADS_PER_THREADGROUP as u64, 1, 1],
            dynamic_threadgroup_bytes: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DenseRoundParams {
    pub(crate) source_rows: u32,
    pub(crate) destination_rows: u32,
    pub(crate) pair_count: u32,
    pub(crate) e_in_length: u32,
    pub(crate) e_out_length: u32,
    pub(crate) columns: u32,
    pub(crate) round: u32,
    pub(crate) reserved: u32,
}

const _: [(); 32] = [(); size_of::<DenseRoundParams>()];

impl DenseRoundParams {
    pub(crate) fn target_shard(round: u32) -> Result<Self, RegistersRwV3Error> {
        if !(9..=25).contains(&round) {
            return Err(RegistersRwV3Error::InvalidExecutionRound(round));
        }
        let destination_rows = TARGET_SHARD_CYCLES >> round;
        let source_rows = 2 * destination_rows;
        let pair_count = destination_rows / 2;
        let e_out_length = pair_count.min(MAX_DENSE_OUTER_LENGTH);
        let e_in_length = pair_count / e_out_length;
        Ok(Self {
            source_rows,
            destination_rows,
            pair_count,
            e_in_length,
            e_out_length,
            columns: COLUMNS,
            round,
            reserved: 0,
        })
    }

    pub(crate) const fn dynamic_threadgroup_bytes() -> u32 {
        8 * size_of::<Fp128>() as u32
    }

    pub(crate) const fn dispatch(self) -> DispatchGeometry {
        DispatchGeometry {
            threadgroups: [self.e_out_length as u64, 1, 1],
            threads_per_threadgroup: [THREADS_PER_THREADGROUP as u64, 1, 1],
            dynamic_threadgroup_bytes: Self::dynamic_threadgroup_bytes(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReductionParams {
    pub(crate) input_count: u32,
    pub(crate) output_count: u32,
    pub(crate) columns: u32,
    pub(crate) reserved: u32,
}

const _: [(); 16] = [(); size_of::<ReductionParams>()];

impl ReductionParams {
    pub(crate) fn new(input_count: u32, columns: u32) -> Result<Self, RegistersRwV3Error> {
        if input_count == 0 || columns == 0 {
            return Err(RegistersRwV3Error::InvalidExecutionParameter(
                "reduction dimensions",
            ));
        }
        Ok(Self {
            input_count,
            output_count: input_count.div_ceil(SIMD_WIDTH),
            columns,
            reserved: 0,
        })
    }

    pub(crate) const fn dispatch(self) -> DispatchGeometry {
        DispatchGeometry {
            threadgroups: [
                self.input_count.div_ceil(REDUCTION_THREADS) as u64,
                self.columns as u64,
                1,
            ],
            threads_per_threadgroup: [REDUCTION_THREADS as u64, 1, 1],
            dynamic_threadgroup_bytes: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HistogramParams {
    pub(crate) cycles: u32,
    pub(crate) blocks: u32,
    pub(crate) blocks_per_hi: u32,
    pub(crate) e_hi_length: u32,
    pub(crate) e_lo_length: u32,
    pub(crate) columns: u32,
    pub(crate) offset_stride: u32,
    pub(crate) position_stride: u32,
}

const _: [(); 32] = [(); size_of::<HistogramParams>()];

impl HistogramParams {
    pub(crate) const fn target_shard() -> Self {
        Self {
            cycles: TARGET_SHARD_CYCLES,
            blocks: TARGET_SHARD_CYCLES / BLOCK_CYCLES,
            blocks_per_hi: 32,
            e_hi_length: HISTOGRAM_HIGH_LENGTH,
            e_lo_length: HISTOGRAM_LOW_LENGTH,
            columns: COLUMNS,
            offset_stride: OFFSET_STRIDE,
            position_stride: BLOCK_CYCLES,
        }
    }

    pub(crate) const fn dispatch(self) -> DispatchGeometry {
        DispatchGeometry {
            threadgroups: [self.e_hi_length as u64, 1, 1],
            threads_per_threadgroup: [THREADS_PER_THREADGROUP as u64, 1, 1],
            dynamic_threadgroup_bytes: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Arena {
    StartValues,
    Rs1Offsets,
    Rs1Positions,
    Rs2Offsets,
    Rs2Positions,
    RdOffsets,
    RdPositions,
    RdPostValues,
    RdIndex,
    RdInc,
    IncScratchA,
    IncScratchB,
    DenseStateA,
    DenseStateB,
    RawLocalWeights,
    Coefficients,
    EqualityA,
    EqualityB,
    PartialA,
    PartialB,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SequencePoint {
    ProducerReceipt,
    RawRound(u8),
    AfterRawRound1,
    Round8Junction,
    DenseRound(u8),
    AfterDenseRound25,
    HistogramComplete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LifetimeDisposition {
    Release,
    RegistersVal,
    HistogramScratch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ArenaLifetime {
    pub(crate) arena: Arena,
    pub(crate) first_use: SequencePoint,
    pub(crate) last_use: SequencePoint,
    pub(crate) disposition: LifetimeDisposition,
}

pub(crate) const ARENA_LIFETIMES: [ArenaLifetime; 20] = [
    lifetime(
        Arena::StartValues,
        SequencePoint::RawRound(0),
        SequencePoint::Round8Junction,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::Rs1Offsets,
        SequencePoint::RawRound(0),
        SequencePoint::HistogramComplete,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::Rs1Positions,
        SequencePoint::RawRound(0),
        SequencePoint::HistogramComplete,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::Rs2Offsets,
        SequencePoint::RawRound(0),
        SequencePoint::HistogramComplete,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::Rs2Positions,
        SequencePoint::RawRound(0),
        SequencePoint::HistogramComplete,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::RdOffsets,
        SequencePoint::RawRound(0),
        SequencePoint::Round8Junction,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::RdPositions,
        SequencePoint::RawRound(0),
        SequencePoint::Round8Junction,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::RdPostValues,
        SequencePoint::RawRound(0),
        SequencePoint::Round8Junction,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::RdIndex,
        SequencePoint::ProducerReceipt,
        SequencePoint::AfterRawRound1,
        LifetimeDisposition::RegistersVal,
    ),
    lifetime(
        Arena::RdInc,
        SequencePoint::RawRound(1),
        SequencePoint::AfterRawRound1,
        LifetimeDisposition::RegistersVal,
    ),
    lifetime(
        Arena::IncScratchA,
        SequencePoint::RawRound(1),
        SequencePoint::AfterDenseRound25,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::IncScratchB,
        SequencePoint::RawRound(2),
        SequencePoint::AfterDenseRound25,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::DenseStateA,
        SequencePoint::Round8Junction,
        SequencePoint::AfterDenseRound25,
        LifetimeDisposition::HistogramScratch,
    ),
    lifetime(
        Arena::DenseStateB,
        SequencePoint::DenseRound(9),
        SequencePoint::AfterDenseRound25,
        LifetimeDisposition::HistogramScratch,
    ),
    lifetime(
        Arena::RawLocalWeights,
        SequencePoint::RawRound(1),
        SequencePoint::Round8Junction,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::Coefficients,
        SequencePoint::RawRound(1),
        SequencePoint::Round8Junction,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::EqualityA,
        SequencePoint::RawRound(0),
        SequencePoint::HistogramComplete,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::EqualityB,
        SequencePoint::RawRound(0),
        SequencePoint::HistogramComplete,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::PartialA,
        SequencePoint::RawRound(0),
        SequencePoint::HistogramComplete,
        LifetimeDisposition::Release,
    ),
    lifetime(
        Arena::PartialB,
        SequencePoint::RawRound(0),
        SequencePoint::HistogramComplete,
        LifetimeDisposition::Release,
    ),
];

const fn lifetime(
    arena: Arena,
    first_use: SequencePoint,
    last_use: SequencePoint,
    disposition: LifetimeDisposition,
) -> ArenaLifetime {
    ArenaLifetime {
        arena,
        first_use,
        last_use,
        disposition,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegistersValHandoff {
    pub(crate) completion_point: SequencePoint,
    pub(crate) planes: [Arena; 2],
}

pub(crate) const REGISTERS_VAL_HANDOFF: RegistersValHandoff = RegistersValHandoff {
    completion_point: SequencePoint::AfterRawRound1,
    planes: [Arena::RdIndex, Arena::RdInc],
};

fn validate_raw_round(round: u32) -> Result<(), RegistersRwV3Error> {
    if !(1..=8).contains(&round) {
        return Err(RegistersRwV3Error::InvalidExecutionRound(round));
    }
    Ok(())
}
