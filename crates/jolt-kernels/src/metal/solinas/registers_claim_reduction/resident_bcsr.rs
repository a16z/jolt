use core::mem::size_of;

use super::RegistersClaimPlanError;

pub const REGISTERS_CLAIM_BCSR_TARGET_LOG_T: u32 = 26;
pub const REGISTERS_CLAIM_BCSR_CYCLES: u64 = 1 << REGISTERS_CLAIM_BCSR_TARGET_LOG_T;
pub const REGISTERS_CLAIM_BCSR_BLOCKS: u64 = REGISTERS_CLAIM_BCSR_CYCLES / 256;
pub const REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS: u64 = 8_192;
pub const REGISTERS_CLAIM_BCSR_SUFFIX_ELEMENTS: u64 = 8_192;
pub const REGISTERS_CLAIM_BCSR_PARTIAL_BLOCKS: u64 = 256;

pub const REGISTERS_CLAIM_BCSR_RS1_EVENTS: u64 = 59_652_323;
pub const REGISTERS_CLAIM_BCSR_RS2_EVENTS: u64 = 55_924_053;
pub const REGISTERS_CLAIM_BCSR_RD_EVENTS: u64 = 50_331_648;
pub const REGISTERS_CLAIM_BCSR_ALL_EVENTS: u64 = 165_908_024;

pub const REGISTERS_CLAIM_BCSR_TOPOLOGY_BYTES: u64 = 1_209_532_416;
pub const REGISTERS_CLAIM_BCSR_INITIALIZED_TOPOLOGY_BYTES: u64 = 1_039_896_120;
pub const REGISTERS_CLAIM_BCSR_SHARED_PRODUCER_BYTES: u64 = 2_350_383_104;
pub const REGISTERS_CLAIM_BCSR_SHARED_PRODUCER_WRITES: u64 = 2_180_746_808;
pub const REGISTERS_CLAIM_BCSR_HALF_WIDTH_TERMS_PER_SECOND: u64 = 33_168_000_000;
pub const REGISTERS_CLAIM_BCSR_TRAFFIC_BYTES_PER_SECOND: u64 = 451_701_710_520;

pub const BCSR_COMPONENT_PIPELINE: &str = "solinas_registers_claim_bcsr_components";
pub const BCSR_INDEXED_COMPONENT_PIPELINE: &str = "solinas_registers_claim_bcsr_indexed_components";
pub const BCSR_COMPONENT_REDUCE_PIPELINE: &str = "solinas_registers_claim_bcsr_reduce_components";
pub const BCSR_MIDPOINT_PIPELINE: &str = "solinas_registers_claim_bcsr_fold_rd_midpoint";

pub const BCSR_COMPONENT_THREADGROUPS: u64 = 8_192;
pub const BCSR_COMPONENT_THREADS_PER_THREADGROUP: u64 = 256;
pub const BCSR_COMPONENT_REPLAY_BYTES: u64 = 3 * 256 * size_of::<u64>() as u64;
pub const BCSR_COMPONENT_WEIGHT_THREADGROUP_BYTES: u64 = 16;
pub const BCSR_COMPONENT_THREADGROUP_BYTES: u64 =
    BCSR_COMPONENT_REPLAY_BYTES + BCSR_COMPONENT_WEIGHT_THREADGROUP_BYTES;
pub const BCSR_INDEXED_EVENT_BYTES: u64 = 256 * size_of::<u16>() as u64;
pub const BCSR_INDEXED_THREADGROUP_BYTES: u64 =
    BCSR_INDEXED_EVENT_BYTES + BCSR_COMPONENT_WEIGHT_THREADGROUP_BYTES;
pub const BCSR_COMPONENT_REDUCE_THREADGROUPS: u64 = 96;
pub const BCSR_COMPONENT_REDUCE_THREADS_PER_THREADGROUP: u64 = 256;
pub const BCSR_MIDPOINT_THREADGROUPS: u64 = 8_192;
pub const BCSR_MIDPOINT_THREADS_PER_THREADGROUP: u64 = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistersClaimBcsrReplayStrategy {
    ColumnReplay,
    IndexedPredecessor,
}

impl RegistersClaimBcsrReplayStrategy {
    pub const fn name(self) -> &'static str {
        match self {
            Self::ColumnReplay => "column-replay",
            Self::IndexedPredecessor => "indexed-predecessor",
        }
    }

    pub(crate) const fn component_pipeline(self) -> &'static str {
        match self {
            Self::ColumnReplay => BCSR_COMPONENT_PIPELINE,
            Self::IndexedPredecessor => BCSR_INDEXED_COMPONENT_PIPELINE,
        }
    }

    pub(crate) const fn threadgroup_bytes(self) -> u64 {
        match self {
            Self::ColumnReplay => BCSR_COMPONENT_THREADGROUP_BYTES,
            Self::IndexedPredecessor => BCSR_INDEXED_THREADGROUP_BYTES,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimBcsrKernelConfig {
    pub partial_blocks: usize,
    pub replay: RegistersClaimBcsrReplayStrategy,
}

impl Default for RegistersClaimBcsrKernelConfig {
    fn default() -> Self {
        Self {
            partial_blocks: REGISTERS_CLAIM_BCSR_PARTIAL_BLOCKS as usize,
            replay: RegistersClaimBcsrReplayStrategy::ColumnReplay,
        }
    }
}

pub const BCSR_COMPONENT_START_VALUES_SLOT: u64 = 0;
pub const BCSR_COMPONENT_RS1_OFFSETS_SLOT: u64 = 1;
pub const BCSR_COMPONENT_RS1_POSITIONS_SLOT: u64 = 2;
pub const BCSR_COMPONENT_RS2_OFFSETS_SLOT: u64 = 3;
pub const BCSR_COMPONENT_RS2_POSITIONS_SLOT: u64 = 4;
pub const BCSR_COMPONENT_RD_OFFSETS_SLOT: u64 = 5;
pub const BCSR_COMPONENT_RD_POSITIONS_SLOT: u64 = 6;
pub const BCSR_COMPONENT_RD_POST_VALUES_SLOT: u64 = 7;
pub const BCSR_COMPONENT_EQ_SUFFIX_SLOT: u64 = 8;
pub const BCSR_COMPONENT_PARTIALS_SLOT: u64 = 9;
pub const BCSR_COMPONENT_PARAMS_SLOT: u64 = 10;
pub const BCSR_COMPONENT_THREADGROUP_SLOT: u64 = 0;

pub const BCSR_INDEXED_START_VALUES_SLOT: u64 = 0;
pub const BCSR_INDEXED_RD_OFFSETS_SLOT: u64 = 1;
pub const BCSR_INDEXED_RD_POSITIONS_SLOT: u64 = 2;
pub const BCSR_INDEXED_RD_POST_VALUES_SLOT: u64 = 3;
pub const BCSR_INDEXED_RS1_INDEX_SLOT: u64 = 4;
pub const BCSR_INDEXED_RS2_INDEX_SLOT: u64 = 5;
pub const BCSR_INDEXED_EQ_SUFFIX_SLOT: u64 = 6;
pub const BCSR_INDEXED_PARTIALS_SLOT: u64 = 7;
pub const BCSR_INDEXED_PARAMS_SLOT: u64 = 8;

pub const BCSR_COMPONENT_REDUCE_INPUT_SLOT: u64 = 0;
pub const BCSR_COMPONENT_REDUCE_OUTPUT_SLOT: u64 = 1;
pub const BCSR_COMPONENT_REDUCE_PARAMS_SLOT: u64 = 2;

pub const BCSR_MIDPOINT_RD_OFFSETS_SLOT: u64 = 0;
pub const BCSR_MIDPOINT_RD_POSITIONS_SLOT: u64 = 1;
pub const BCSR_MIDPOINT_RD_POST_VALUES_SLOT: u64 = 2;
pub const BCSR_MIDPOINT_EQ_PREFIX_SLOT: u64 = 3;
pub const BCSR_MIDPOINT_OUTPUT_SLOT: u64 = 4;
pub const BCSR_MIDPOINT_PARAMS_SLOT: u64 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimBcsrComponentParams {
    pub cycles: u32,
    pub blocks: u32,
    pub prefix_elements: u32,
    pub suffix_elements: u32,
    pub partial_blocks: u32,
    pub low_blocks: u32,
    pub suffixes_per_partial: u32,
    pub columns: u32,
}

const _: [(); 32] = [(); size_of::<RegistersClaimBcsrComponentParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimBcsrReduceParams {
    pub partial_blocks: u32,
    pub prefix_elements: u32,
    pub columns: u32,
    pub reserved: u32,
}

const _: [(); 16] = [(); size_of::<RegistersClaimBcsrReduceParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimBcsrMidpointParams {
    pub blocks: u32,
    pub prefix_elements: u32,
    pub suffix_elements: u32,
    pub low_blocks: u32,
    pub columns: u32,
    pub offset_stride: u32,
    pub position_stride: u32,
    pub reserved: u32,
}

const _: [(); 32] = [(); size_of::<RegistersClaimBcsrMidpointParams>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistersClaimResidentSource {
    DenseValuePlanes,
    SparseEventValues,
    BcsrStateFlow,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegistersClaimTraffic {
    pub cache_unique_bytes: u64,
    pub shader_requested_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegistersClaimResidentPhase {
    pub half_width_terms: u64,
    pub traffic: RegistersClaimTraffic,
    pub dispatches: u32,
    pub added_command_buffers: u32,
    pub added_waits: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimSourceCost {
    pub source: RegistersClaimResidentSource,
    pub additional_persistent_bytes: u64,
    pub additional_producer_writes: u64,
    pub component_source_reads: u64,
    pub midpoint_source_reads: u64,
}

impl RegistersClaimSourceCost {
    pub const fn charged_source_bytes(self) -> u64 {
        self.additional_producer_writes + self.component_source_reads + self.midpoint_source_reads
    }
}

pub const REGISTERS_CLAIM_LOG26_SOURCE_COSTS: [RegistersClaimSourceCost; 3] = [
    RegistersClaimSourceCost {
        source: RegistersClaimResidentSource::DenseValuePlanes,
        additional_persistent_bytes: 1_610_612_736,
        additional_producer_writes: 1_610_612_736,
        component_source_reads: 1_610_612_736,
        midpoint_source_reads: 536_870_912,
    },
    RegistersClaimSourceCost {
        source: RegistersClaimResidentSource::SparseEventValues,
        additional_persistent_bytes: 924_611_008,
        additional_producer_writes: 924_611_008,
        component_source_reads: 1_494_745_080,
        midpoint_source_reads: 453_509_120,
    },
    RegistersClaimSourceCost {
        source: RegistersClaimResidentSource::BcsrStateFlow,
        additional_persistent_bytes: 0,
        additional_producer_writes: 0,
        component_source_reads: 1_039_896_120,
        midpoint_source_reads: 520_617_984,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimResidentRoof {
    pub optimistic_floor_ns: u64,
    pub no_cache_floor_ns: u64,
    pub optimistic_eighty_percent_ns: u64,
    pub no_cache_eighty_percent_ns: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersClaimBcsrPlan {
    pub components: RegistersClaimResidentPhase,
    pub midpoint: RegistersClaimResidentPhase,
    pub component_partial_bytes: u64,
    pub component_carrier_bytes: u64,
    pub midpoint_output_bytes: u64,
    pub host_full_products: u64,
    pub host_logical_bytes: u64,
    pub roof: RegistersClaimResidentRoof,
}

impl RegistersClaimBcsrPlan {
    pub fn log26() -> Result<Self, RegistersClaimPlanError> {
        let component_partial_bytes =
            3 * REGISTERS_CLAIM_BCSR_PARTIAL_BLOCKS * REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS * 16;
        let component_carrier_bytes = 3 * REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS * 16;
        let midpoint_output_bytes = REGISTERS_CLAIM_BCSR_SUFFIX_ELEMENTS * 16;

        let components = RegistersClaimResidentPhase {
            half_width_terms: REGISTERS_CLAIM_BCSR_ALL_EVENTS,
            traffic: RegistersClaimTraffic {
                cache_unique_bytes: REGISTERS_CLAIM_BCSR_INITIALIZED_TOPOLOGY_BYTES
                    + 16 * REGISTERS_CLAIM_BCSR_SUFFIX_ELEMENTS
                    + 2 * component_partial_bytes
                    + component_carrier_bytes,
                shader_requested_bytes: REGISTERS_CLAIM_BCSR_INITIALIZED_TOPOLOGY_BYTES
                    + 16 * REGISTERS_CLAIM_BCSR_BLOCKS
                    + 2 * component_partial_bytes
                    + component_carrier_bytes,
            },
            dispatches: 2,
            added_command_buffers: 0,
            added_waits: 0,
        };
        let rd_topology_bytes = REGISTERS_CLAIM_BCSR_BLOCKS * 129 * 2
            + REGISTERS_CLAIM_BCSR_RD_EVENTS
            + 8 * REGISTERS_CLAIM_BCSR_RD_EVENTS;
        let midpoint = RegistersClaimResidentPhase {
            half_width_terms: REGISTERS_CLAIM_BCSR_RD_EVENTS,
            traffic: RegistersClaimTraffic {
                cache_unique_bytes: rd_topology_bytes
                    + 16 * REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS
                    + midpoint_output_bytes,
                shader_requested_bytes: rd_topology_bytes
                    + 16 * REGISTERS_CLAIM_BCSR_RD_EVENTS
                    + midpoint_output_bytes,
            },
            dispatches: 1,
            added_command_buffers: 1,
            added_waits: 1,
        };

        let optimistic_floor_ns = phase_floor_ns(components, false)?
            .checked_add(phase_floor_ns(midpoint, false)?)
            .ok_or(RegistersClaimPlanError::SizeOverflow {
                name: "resident BCSR optimistic floor",
            })?;
        let no_cache_floor_ns = phase_floor_ns(components, true)?
            .checked_add(phase_floor_ns(midpoint, true)?)
            .ok_or(RegistersClaimPlanError::SizeOverflow {
                name: "resident BCSR no-cache floor",
            })?;
        let roof = RegistersClaimResidentRoof {
            optimistic_floor_ns,
            no_cache_floor_ns,
            optimistic_eighty_percent_ns: utilization_ns(optimistic_floor_ns, 80)?,
            no_cache_eighty_percent_ns: utilization_ns(no_cache_floor_ns, 80)?,
        };

        let plan = Self {
            components,
            midpoint,
            component_partial_bytes,
            component_carrier_bytes,
            midpoint_output_bytes,
            host_full_products: 9 * REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS
                + 8 * REGISTERS_CLAIM_BCSR_SUFFIX_ELEMENTS
                - 12,
            host_logical_bytes: 64 * REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS
                + 48
                + 64 * REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS
                + 160 * REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS
                - 256
                + 320 * REGISTERS_CLAIM_BCSR_SUFFIX_ELEMENTS
                - 256,
            roof,
        };
        if plan.components.traffic.cache_unique_bytes != 1_241_747_000
            || plan.components.traffic.shader_requested_bytes != 1_245_810_232
            || plan.midpoint.traffic.cache_unique_bytes != 520_880_128
            || plan.midpoint.traffic.shader_requested_bytes != 1_326_055_424
            || plan.host_full_products != 139_252
            || plan.host_logical_bytes != 4_980_272
            || plan.roof.optimistic_eighty_percent_ns != 8_149_410
            || plan.roof.no_cache_eighty_percent_ns != 9_922_175
        {
            return Err(RegistersClaimPlanError::SizeOverflow {
                name: "resident BCSR log-26 census",
            });
        }
        Ok(plan)
    }

    pub const fn component_params(self) -> RegistersClaimBcsrComponentParams {
        RegistersClaimBcsrComponentParams {
            cycles: REGISTERS_CLAIM_BCSR_CYCLES as u32,
            blocks: REGISTERS_CLAIM_BCSR_BLOCKS as u32,
            prefix_elements: REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS as u32,
            suffix_elements: REGISTERS_CLAIM_BCSR_SUFFIX_ELEMENTS as u32,
            partial_blocks: REGISTERS_CLAIM_BCSR_PARTIAL_BLOCKS as u32,
            low_blocks: (REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS / 256) as u32,
            suffixes_per_partial: (REGISTERS_CLAIM_BCSR_SUFFIX_ELEMENTS
                / REGISTERS_CLAIM_BCSR_PARTIAL_BLOCKS) as u32,
            columns: 128,
        }
    }

    pub const fn reduce_params(self) -> RegistersClaimBcsrReduceParams {
        RegistersClaimBcsrReduceParams {
            partial_blocks: REGISTERS_CLAIM_BCSR_PARTIAL_BLOCKS as u32,
            prefix_elements: REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS as u32,
            columns: 3,
            reserved: 0,
        }
    }

    pub const fn midpoint_params(self) -> RegistersClaimBcsrMidpointParams {
        RegistersClaimBcsrMidpointParams {
            blocks: REGISTERS_CLAIM_BCSR_BLOCKS as u32,
            prefix_elements: REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS as u32,
            suffix_elements: REGISTERS_CLAIM_BCSR_SUFFIX_ELEMENTS as u32,
            low_blocks: (REGISTERS_CLAIM_BCSR_PREFIX_ELEMENTS / 256) as u32,
            columns: 128,
            offset_stride: 129,
            position_stride: 256,
            reserved: 0,
        }
    }
}

fn phase_floor_ns(
    phase: RegistersClaimResidentPhase,
    use_requested_traffic: bool,
) -> Result<u64, RegistersClaimPlanError> {
    let compute = rate_ns(
        phase.half_width_terms,
        REGISTERS_CLAIM_BCSR_HALF_WIDTH_TERMS_PER_SECOND,
    )?;
    let bytes = if use_requested_traffic {
        phase.traffic.shader_requested_bytes
    } else {
        phase.traffic.cache_unique_bytes
    };
    Ok(compute.max(rate_ns(
        bytes,
        REGISTERS_CLAIM_BCSR_TRAFFIC_BYTES_PER_SECOND,
    )?))
}

fn utilization_ns(floor_ns: u64, percent: u64) -> Result<u64, RegistersClaimPlanError> {
    let numerator = u128::from(floor_ns) * 100;
    let value = numerator.div_ceil(u128::from(percent));
    u64::try_from(value).map_err(|_| RegistersClaimPlanError::SizeOverflow {
        name: "resident BCSR utilization ceiling",
    })
}

fn rate_ns(units: u64, rate: u64) -> Result<u64, RegistersClaimPlanError> {
    let value = (u128::from(units) * 1_000_000_000).div_ceil(u128::from(rate));
    u64::try_from(value).map_err(|_| RegistersClaimPlanError::SizeOverflow {
        name: "resident BCSR roof time",
    })
}
