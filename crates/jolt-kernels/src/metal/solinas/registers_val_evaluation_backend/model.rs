//! Checked geometry, schedule, storage, and roofline accounting.

use core::mem::size_of;

pub const FROZEN_EVALUATOR: &str = "benchmark-runs/metal-piop-eval/20260806-133709-697013";
pub const FROZEN_REVISION: &str = "5f520c21e338632aa0bf5936ceb02be6c22fa40f";
pub const FROZEN_COMPLETE_CPU_NS: u64 = 337_038_126;
pub const TARGET_FIVE_X_NS: u64 = FROZEN_COMPLETE_CPU_NS / 5;
pub const TARGET_EIGHT_X_NS: u64 = FROZEN_COMPLETE_CPU_NS / 8;
pub const TARGET_LOG_T: usize = 26;
pub const FROZEN_CPU_TAIL_2_16_NS: u64 = 3_808_875;

pub const FIELD_BYTES: u128 = 16;
pub const INDEX_BYTES: u128 = 1;
pub const HOST_OPTION_INDEX_BYTES: u128 = 2;
pub const RESIDENT_INPUT_BYTES_PER_ROW: u128 = FIELD_BYTES + INDEX_BYTES;
pub const STAGE4_SOURCE_BYTES_PER_ROW: u128 = FIELD_BYTES + HOST_OPTION_INDEX_BYTES;
pub const STAGE4_PUBLISH_BYTES_PER_ROW: u128 =
    STAGE4_SOURCE_BYTES_PER_ROW + RESIDENT_INPUT_BYTES_PER_ROW;
pub const REGISTER_ADDRESS_DOMAIN: usize = 128;
pub const DEFAULT_TRACE_CUTOFF_ELEMENTS: usize = 1 << 20;
pub const DEFAULT_CPU_TAIL_ELEMENTS: usize = 1 << 16;

pub const M4_MAX_COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const M4_MAX_SIX_ACCUMULATOR_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;
pub const M4_MAX_DIRECT_PRODUCTS_PER_SECOND: u64 = 32_330_000_000;

const _: () = assert!(size_of::<Option<u8>>() == HOST_OPTION_INDEX_BYTES as usize);

pub const fn five_x_accepts(metal_ns: u64) -> bool {
    (metal_ns as u128) * 5 <= FROZEN_COMPLETE_CPU_NS as u128
}

pub const fn eight_x_accepts(metal_ns: u64) -> bool {
    (metal_ns as u128) * 8 <= FROZEN_COMPLETE_CPU_NS as u128
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValShape {
    elements: usize,
    log_t: usize,
    low_bits: usize,
    high_bits: usize,
    low_elements: usize,
    high_elements: usize,
}

impl RegistersValShape {
    pub fn new(elements: usize) -> Result<Self, RegistersValPlanError> {
        if elements < 4 || !elements.is_power_of_two() {
            return Err(RegistersValPlanError::InvalidElements { got: elements });
        }
        let log_t = elements.ilog2() as usize;
        let low_bits = log_t / 2;
        let high_bits = log_t - low_bits;
        let low_elements = checked_pow2(low_bits)?;
        let high_elements = checked_pow2(high_bits)?;
        Ok(Self {
            elements,
            log_t,
            low_bits,
            high_bits,
            low_elements,
            high_elements,
        })
    }

    pub const fn elements(self) -> usize {
        self.elements
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn low_bits(self) -> usize {
        self.low_bits
    }

    pub const fn high_bits(self) -> usize {
        self.high_bits
    }

    pub const fn low_elements(self) -> usize {
        self.low_elements
    }

    pub const fn high_elements(self) -> usize {
        self.high_elements
    }

    /// The smallest resident dense state the current split-LT primitive can
    /// export. One low variable remains for the CPU to bind.
    pub fn split_handoff_elements(self) -> Result<usize, RegistersValPlanError> {
        self.high_elements
            .checked_mul(2)
            .ok_or(RegistersValPlanError::SizeOverflow)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValConfig {
    pub trace_cutoff_elements: usize,
    pub cpu_tail_elements: usize,
}

impl Default for RegistersValConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: DEFAULT_TRACE_CUTOFF_ELEMENTS,
            cpu_tail_elements: DEFAULT_CPU_TAIL_ELEMENTS,
        }
    }
}

impl RegistersValConfig {
    fn validate(self) -> Result<(), RegistersValPlanError> {
        for (name, value) in [
            ("trace cutoff", self.trace_cutoff_elements),
            ("CPU-tail cutoff", self.cpu_tail_elements),
        ] {
            if value == 0 || !value.is_power_of_two() {
                return Err(RegistersValPlanError::InvalidCutoff { name, got: value });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistersValExecution {
    OptimizedCpu(RegistersValFallback),
    MetalHybrid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistersValFallback {
    TraceBelowCutoff,
    NoUsefulDevicePrefix,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoundOwner {
    MetalFirstMessage,
    MetalNativeTransition,
    MetalDenseTransition,
    CpuTail,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValRound {
    pub round: usize,
    pub owner: RoundOwner,
    /// State length before this round's optional bind.
    pub source_elements: usize,
    /// State length used to form this round's message.
    pub message_elements: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistersValPlan {
    shape: RegistersValShape,
    execution: RegistersValExecution,
    effective_cpu_tail_elements: usize,
    rounds: Vec<RegistersValRound>,
}

impl RegistersValPlan {
    pub fn new(
        shape: RegistersValShape,
        config: RegistersValConfig,
    ) -> Result<Self, RegistersValPlanError> {
        config.validate()?;
        if shape.elements < config.trace_cutoff_elements {
            return Ok(Self::cpu_only(
                shape,
                RegistersValFallback::TraceBelowCutoff,
            ));
        }

        let effective_cpu_tail_elements = config
            .cpu_tail_elements
            .max(shape.split_handoff_elements()?);
        if effective_cpu_tail_elements >= shape.elements {
            return Ok(Self::cpu_only(
                shape,
                RegistersValFallback::NoUsefulDevicePrefix,
            ));
        }
        if !effective_cpu_tail_elements.is_power_of_two()
            || !shape.elements.is_multiple_of(effective_cpu_tail_elements)
        {
            return Err(RegistersValPlanError::InvalidEffectiveTail {
                got: effective_cpu_tail_elements,
            });
        }

        let device_binds = (shape.elements / effective_cpu_tail_elements).ilog2() as usize;
        let device_messages = device_binds + 1;
        if device_messages > shape.log_t {
            return Err(RegistersValPlanError::DeviceWindowOutsideRelation {
                messages: device_messages,
                rounds: shape.log_t,
            });
        }

        let mut rounds = Vec::with_capacity(shape.log_t);
        for round in 0..shape.log_t {
            let source_elements = if round == 0 {
                shape.elements
            } else {
                shape.elements >> (round - 1)
            };
            let message_elements = shape.elements >> round;
            let owner = if round >= device_messages {
                RoundOwner::CpuTail
            } else {
                match round {
                    0 => RoundOwner::MetalFirstMessage,
                    1 => RoundOwner::MetalNativeTransition,
                    _ => RoundOwner::MetalDenseTransition,
                }
            };
            rounds.push(RegistersValRound {
                round,
                owner,
                source_elements,
                message_elements,
            });
        }

        Ok(Self {
            shape,
            execution: RegistersValExecution::MetalHybrid,
            effective_cpu_tail_elements,
            rounds,
        })
    }

    fn cpu_only(shape: RegistersValShape, fallback: RegistersValFallback) -> Self {
        let rounds = (0..shape.log_t)
            .map(|round| RegistersValRound {
                round,
                owner: RoundOwner::CpuTail,
                source_elements: if round == 0 {
                    shape.elements
                } else {
                    shape.elements >> (round - 1)
                },
                message_elements: shape.elements >> round,
            })
            .collect();
        Self {
            shape,
            execution: RegistersValExecution::OptimizedCpu(fallback),
            effective_cpu_tail_elements: shape.elements,
            rounds,
        }
    }

    pub const fn shape(&self) -> RegistersValShape {
        self.shape
    }

    pub const fn execution(&self) -> RegistersValExecution {
        self.execution
    }

    pub const fn effective_cpu_tail_elements(&self) -> usize {
        self.effective_cpu_tail_elements
    }

    pub fn rounds(&self) -> &[RegistersValRound] {
        &self.rounds
    }

    pub fn device_message_count(&self) -> usize {
        self.rounds
            .iter()
            .take_while(|round| round.owner != RoundOwner::CpuTail)
            .count()
    }

    pub fn dense_transition_count(&self) -> usize {
        self.rounds
            .iter()
            .filter(|round| round.owner == RoundOwner::MetalDenseTransition)
            .count()
    }

    pub fn work(&self, variant: KernelVariant) -> Result<Vec<PhaseWork>, RegistersValPlanError> {
        if self.execution != RegistersValExecution::MetalHybrid {
            return Ok(Vec::new());
        }
        let n = self.shape.elements as u128;
        let high = self.shape.high_elements as u128;
        let tail = self.effective_cpu_tail_elements as u128;
        let dense_transitions = self.dense_transition_count() as u128;
        let dense_sources = n
            .checked_sub(2 * tail)
            .ok_or(RegistersValPlanError::SizeOverflow)?;

        let (first_products, native_products, dense_products) = match variant {
            KernelVariant::FactorizedSixAccumulator => (
                3 * n + 6 * high,
                5 * n / 2 + 6 * high,
                5 * dense_sources / 2 + 6 * high * dense_transitions,
            ),
            KernelVariant::DirectLtThreeAccumulator => {
                (9 * n / 2, 13 * n / 4, 13 * dense_sources / 4)
            }
        };

        Ok(vec![
            PhaseWork {
                phase: MetalPhase::FirstMessage,
                useful_products: first_products,
                compulsory_bytes: 17 * n,
            },
            PhaseWork {
                phase: MetalPhase::NativeTransition,
                useful_products: native_products,
                compulsory_bytes: 33 * n,
            },
            PhaseWork {
                phase: MetalPhase::DenseLadder,
                useful_products: dense_products,
                compulsory_bytes: 48 * dense_sources,
            },
        ])
    }

    pub fn resident_bytes(&self) -> Result<ResidentBytes, RegistersValPlanError> {
        if self.execution != RegistersValExecution::MetalHybrid {
            return Ok(ResidentBytes {
                borrowed_inputs: 0,
                sequence_owned: 0,
                peak: 0,
                cpu_readback: 0,
                largest_buffer: 0,
            });
        }
        let n = self.shape.elements as u128;
        let high = self.shape.high_elements as u128;
        let low = self.shape.low_elements as u128;
        let borrowed_inputs = 17 * n;
        let dense_arenas = 24 * n;
        let reduction_scratch = 2 * 3 * high * FIELD_BYTES;
        let split_lt = (low + 2 * high) * FIELD_BYTES;
        let eq_address = REGISTER_ADDRESS_DOMAIN as u128 * FIELD_BYTES;
        let sequence_owned = dense_arenas
            .checked_add(reduction_scratch)
            .and_then(|value| value.checked_add(split_lt))
            .and_then(|value| value.checked_add(eq_address))
            .ok_or(RegistersValPlanError::SizeOverflow)?;
        let peak = borrowed_inputs
            .checked_add(sequence_owned)
            .ok_or(RegistersValPlanError::SizeOverflow)?;
        let cpu_readback = 2 * self.effective_cpu_tail_elements as u128 * FIELD_BYTES;
        Ok(ResidentBytes {
            borrowed_inputs,
            sequence_owned,
            peak,
            cpu_readback,
            largest_buffer: n * FIELD_BYTES,
        })
    }

    pub fn project(
        &self,
        variant: KernelVariant,
        controls: RegistersValRoofControls,
    ) -> Result<Vec<PhaseProjection>, RegistersValPlanError> {
        controls.validate()?;
        let product_rate = match variant {
            KernelVariant::FactorizedSixAccumulator => controls.six_accumulator_products_per_second,
            KernelVariant::DirectLtThreeAccumulator => controls.direct_products_per_second,
        };
        self.work(variant)?
            .into_iter()
            .map(|work| {
                let arithmetic_floor_ns = rate_time_ns(work.useful_products, product_rate)?;
                let traffic_floor_ns =
                    rate_time_ns(work.compulsory_bytes, controls.copy_bytes_per_second)?;
                let roof_floor_ns = arithmetic_floor_ns.max(traffic_floor_ns);
                let admitted_ns = ceil_div(
                    roof_floor_ns
                        .checked_mul(100)
                        .ok_or(RegistersValPlanError::SizeOverflow)?,
                    controls.admitted_percent as u128,
                );
                Ok(PhaseProjection {
                    work,
                    arithmetic_floor_ns,
                    traffic_floor_ns,
                    roof_floor_ns,
                    admitted_ns,
                })
            })
            .collect()
    }

    /// Baseline producer: stage 4 copies the already-materialized canonical
    /// increment and `rd` index tables into two proof-session-owned buffers.
    /// A future native Metal producer may replace this only with allocation-
    /// identity evidence.
    pub fn producer_projection(
        &self,
        controls: RegistersValRoofControls,
    ) -> Result<ProducerProjection, RegistersValPlanError> {
        controls.validate()?;
        if self.execution != RegistersValExecution::MetalHybrid {
            return Ok(ProducerProjection {
                source_read_bytes: 0,
                published_bytes: 0,
                logical_bytes: 0,
                traffic_floor_ns: 0,
                admitted_ns: 0,
            });
        }
        let rows = self.shape.elements as u128;
        let source_read_bytes = STAGE4_SOURCE_BYTES_PER_ROW
            .checked_mul(rows)
            .ok_or(RegistersValPlanError::SizeOverflow)?;
        let published_bytes = RESIDENT_INPUT_BYTES_PER_ROW
            .checked_mul(rows)
            .ok_or(RegistersValPlanError::SizeOverflow)?;
        let logical_bytes = STAGE4_PUBLISH_BYTES_PER_ROW
            .checked_mul(rows)
            .ok_or(RegistersValPlanError::SizeOverflow)?;
        let traffic_floor_ns = rate_time_ns(logical_bytes, controls.copy_bytes_per_second)?;
        let admitted_ns = ceil_div(
            traffic_floor_ns
                .checked_mul(100)
                .ok_or(RegistersValPlanError::SizeOverflow)?,
            controls.admitted_percent as u128,
        );
        Ok(ProducerProjection {
            source_read_bytes,
            published_bytes,
            logical_bytes,
            traffic_floor_ns,
            admitted_ns,
        })
    }

    pub fn fixed_boundary_projection(
        &self,
        variant: KernelVariant,
        controls: RegistersValRoofControls,
    ) -> Result<FixedBoundaryProjection, RegistersValPlanError> {
        if self.execution != RegistersValExecution::MetalHybrid {
            return Err(RegistersValPlanError::NoMetalProjection);
        }
        if self.effective_cpu_tail_elements != DEFAULT_CPU_TAIL_ELEMENTS {
            return Err(RegistersValPlanError::UnmodeledCpuTail {
                got: self.effective_cpu_tail_elements,
            });
        }
        let metal_prefix_admitted_ns =
            self.project(variant, controls)?
                .iter()
                .try_fold(0u128, |sum, phase| {
                    sum.checked_add(phase.admitted_ns)
                        .ok_or(RegistersValPlanError::SizeOverflow)
                })?;
        let producer_admitted_ns = self.producer_projection(controls)?.admitted_ns;
        let readback_bytes = 2u128
            .checked_mul(self.effective_cpu_tail_elements as u128)
            .and_then(|value| value.checked_mul(FIELD_BYTES))
            .ok_or(RegistersValPlanError::SizeOverflow)?;
        let readback_floor_ns = rate_time_ns(readback_bytes, controls.copy_bytes_per_second)?;
        let readback_admitted_ns = ceil_div(
            readback_floor_ns
                .checked_mul(100)
                .ok_or(RegistersValPlanError::SizeOverflow)?,
            controls.admitted_percent as u128,
        );
        let accounted_ns = metal_prefix_admitted_ns
            .checked_add(producer_admitted_ns)
            .and_then(|value| value.checked_add(readback_admitted_ns))
            .and_then(|value| value.checked_add(FROZEN_CPU_TAIL_2_16_NS as u128))
            .ok_or(RegistersValPlanError::SizeOverflow)?;
        Ok(FixedBoundaryProjection {
            metal_prefix_admitted_ns,
            producer_admitted_ns,
            readback_admitted_ns,
            cpu_tail_ns: FROZEN_CPU_TAIL_2_16_NS as u128,
            accounted_ns,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelVariant {
    /// The existing low-level implementation: three A and three B accumulators.
    FactorizedSixAccumulator,
    /// A proposed flattened reduction which forms LT directly per sample.
    DirectLtThreeAccumulator,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalPhase {
    FirstMessage,
    NativeTransition,
    DenseLadder,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseWork {
    pub phase: MetalPhase,
    pub useful_products: u128,
    pub compulsory_bytes: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValRoofControls {
    pub copy_bytes_per_second: u64,
    pub six_accumulator_products_per_second: u64,
    pub direct_products_per_second: u64,
    pub admitted_percent: u8,
}

impl Default for RegistersValRoofControls {
    fn default() -> Self {
        Self {
            copy_bytes_per_second: M4_MAX_COPY_BYTES_PER_SECOND,
            six_accumulator_products_per_second: M4_MAX_SIX_ACCUMULATOR_PRODUCTS_PER_SECOND,
            direct_products_per_second: M4_MAX_DIRECT_PRODUCTS_PER_SECOND,
            admitted_percent: 80,
        }
    }
}

impl RegistersValRoofControls {
    fn validate(self) -> Result<(), RegistersValPlanError> {
        if self.copy_bytes_per_second == 0
            || self.six_accumulator_products_per_second == 0
            || self.direct_products_per_second == 0
        {
            return Err(RegistersValPlanError::ZeroRoofControl);
        }
        if !(1..=100).contains(&self.admitted_percent) {
            return Err(RegistersValPlanError::InvalidRoofPercent {
                got: self.admitted_percent,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseProjection {
    pub work: PhaseWork,
    pub arithmetic_floor_ns: u128,
    pub traffic_floor_ns: u128,
    pub roof_floor_ns: u128,
    pub admitted_ns: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentBytes {
    pub borrowed_inputs: u128,
    pub sequence_owned: u128,
    pub peak: u128,
    pub cpu_readback: u128,
    pub largest_buffer: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerProjection {
    pub source_read_bytes: u128,
    pub published_bytes: u128,
    pub logical_bytes: u128,
    pub traffic_floor_ns: u128,
    pub admitted_ns: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedBoundaryProjection {
    pub metal_prefix_admitted_ns: u128,
    pub producer_admitted_ns: u128,
    pub readback_admitted_ns: u128,
    pub cpu_tail_ns: u128,
    pub accounted_ns: u128,
}

impl FixedBoundaryProjection {
    pub const fn headroom_ns(self, target_ns: u64) -> Option<u128> {
        (target_ns as u128).checked_sub(self.accounted_ns)
    }
}

fn checked_pow2(bits: usize) -> Result<usize, RegistersValPlanError> {
    1usize
        .checked_shl(u32::try_from(bits).map_err(|_| RegistersValPlanError::SizeOverflow)?)
        .ok_or(RegistersValPlanError::SizeOverflow)
}

fn rate_time_ns(work: u128, rate: u64) -> Result<u128, RegistersValPlanError> {
    let scaled = work
        .checked_mul(1_000_000_000)
        .ok_or(RegistersValPlanError::SizeOverflow)?;
    Ok(ceil_div(scaled, rate as u128))
}

const fn ceil_div(numerator: u128, denominator: u128) -> u128 {
    numerator / denominator + (!numerator.is_multiple_of(denominator)) as u128
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistersValPlanError {
    InvalidElements { got: usize },
    InvalidCutoff { name: &'static str, got: usize },
    InvalidEffectiveTail { got: usize },
    DeviceWindowOutsideRelation { messages: usize, rounds: usize },
    ZeroRoofControl,
    InvalidRoofPercent { got: u8 },
    NoMetalProjection,
    UnmodeledCpuTail { got: usize },
    SizeOverflow,
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test setup")]
mod tests {
    use super::*;

    #[test]
    fn target_schedule_and_accounting_are_frozen() {
        let shape = RegistersValShape::new(1 << TARGET_LOG_T).unwrap();
        let plan = RegistersValPlan::new(shape, RegistersValConfig::default()).unwrap();

        assert_eq!(plan.execution(), RegistersValExecution::MetalHybrid);
        assert_eq!(plan.effective_cpu_tail_elements(), 1 << 16);
        assert_eq!(plan.device_message_count(), 11);
        assert_eq!(plan.dense_transition_count(), 9);
        assert_eq!(plan.rounds()[10].owner, RoundOwner::MetalDenseTransition);
        assert_eq!(plan.rounds()[11].owner, RoundOwner::CpuTail);

        let work = plan.work(KernelVariant::FactorizedSixAccumulator).unwrap();
        assert_eq!(work[0].useful_products, 201_375_744);
        assert_eq!(work[0].compulsory_bytes, 1_140_850_688);
        assert_eq!(work[1].useful_products, 167_821_312);
        assert_eq!(work[1].compulsory_bytes, 2_214_592_512);
        assert_eq!(work[2].useful_products, 167_886_848);
        assert_eq!(work[2].compulsory_bytes, 3_214_934_016);

        let resident = plan.resident_bytes().unwrap();
        assert_eq!(resident.borrowed_inputs, 1_140_850_688);
        assert_eq!(resident.sequence_owned, 1_611_794_432);
        assert_eq!(resident.peak, 2_752_645_120);
        assert_eq!(resident.cpu_readback, 2_097_152);
        assert_eq!(resident.largest_buffer, 1_073_741_824);

        let controls = RegistersValRoofControls::default();
        let producer = plan.producer_projection(controls).unwrap();
        assert_eq!(producer.source_read_bytes, 1_207_959_552);
        assert_eq!(producer.published_bytes, 1_140_850_688);
        assert_eq!(producer.logical_bytes, 2_348_810_240);
        assert_eq!(producer.traffic_floor_ns, 5_199_915);
        assert_eq!(producer.admitted_ns, 6_499_894);

        let factorized_phases = plan
            .project(KernelVariant::FactorizedSixAccumulator, controls)
            .unwrap();
        assert_eq!(factorized_phases[2].arithmetic_floor_ns, 9_275_517);
        assert_eq!(factorized_phases[2].admitted_ns, 11_594_397);

        let factorized = plan
            .fixed_boundary_projection(KernelVariant::FactorizedSixAccumulator, controls)
            .unwrap();
        assert_eq!(factorized.metal_prefix_admitted_ns, 37_091_432);
        assert_eq!(factorized.readback_admitted_ns, 5_804);
        assert_eq!(factorized.accounted_ns, 47_406_005);
        assert_eq!(factorized.headroom_ns(TARGET_FIVE_X_NS), Some(20_001_620));

        let direct = plan
            .fixed_boundary_projection(KernelVariant::DirectLtThreeAccumulator, controls)
            .unwrap();
        assert_eq!(direct.metal_prefix_admitted_ns, 29_005_520);
        assert_eq!(direct.accounted_ns, 39_320_093);
        assert_eq!(direct.headroom_ns(TARGET_EIGHT_X_NS), Some(2_809_672));
    }

    #[test]
    fn split_boundary_limits_the_device_prefix() {
        let shape = RegistersValShape::new(1 << TARGET_LOG_T).unwrap();
        let plan = RegistersValPlan::new(
            shape,
            RegistersValConfig {
                trace_cutoff_elements: 1,
                cpu_tail_elements: 1 << 13,
            },
        )
        .unwrap();

        assert_eq!(shape.split_handoff_elements().unwrap(), 1 << 14);
        assert_eq!(plan.effective_cpu_tail_elements(), 1 << 14);
        assert_eq!(plan.device_message_count(), 13);
        assert_eq!(plan.rounds()[13].owner, RoundOwner::CpuTail);
    }

    #[test]
    fn cpu_fallback_has_no_metal_storage_or_work() {
        let shape = RegistersValShape::new(1 << 18).unwrap();
        let plan = RegistersValPlan::new(shape, RegistersValConfig::default()).unwrap();

        assert_eq!(
            plan.execution(),
            RegistersValExecution::OptimizedCpu(RegistersValFallback::TraceBelowCutoff)
        );
        assert!(plan
            .work(KernelVariant::FactorizedSixAccumulator)
            .unwrap()
            .is_empty());
        assert_eq!(
            plan.resident_bytes().unwrap(),
            ResidentBytes {
                borrowed_inputs: 0,
                sequence_owned: 0,
                peak: 0,
                cpu_readback: 0,
                largest_buffer: 0,
            }
        );
    }

    #[test]
    fn acceptance_uses_cross_multiplication() {
        assert!(five_x_accepts(67_407_625));
        assert!(!five_x_accepts(67_407_626));
        assert!(eight_x_accepts(42_129_765));
        assert!(!eight_x_accepts(42_129_766));
    }
}
