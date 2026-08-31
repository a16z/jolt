use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLResourceOptions, MTLSize,
};

use super::booleanity::{balanced_bias, selector_abi, write_fields};
use super::{
    completed_command_gpu_time, set_inline_bytes, BooleanityRows, BooleanitySelector, Fp128,
    MetalError, PipelineLimits, SolinasMetal, AKITA_OFFSET_FFFFA7F7,
};

const SIMD_WIDTH: usize = 32;
const BINS: usize = 256;
const ACCUMULATOR_WORDS: usize = 5;
const MAX_SELECTORS_PER_TILE: usize = 6;
const MAX_INNER_LOG2: usize = 16;
const TILE_PIPELINE: &str = "solinas_booleanity_address_tile";
const PRODUCTION_TILE_PIPELINES: [&str; 5] = [
    "solinas_booleanity_address_tile_0",
    "solinas_booleanity_address_tile_1",
    "solinas_booleanity_address_tile_2",
    "solinas_booleanity_address_tile_3",
    "solinas_booleanity_address_tile_4",
];
const PRODUCTION_THREE_RAM_TILE_PIPELINES: [&str; 5] = [
    "solinas_booleanity_address_tile_0",
    "solinas_booleanity_address_tile_1",
    "solinas_booleanity_address_tile_2",
    "solinas_booleanity_address_tile_ram3_3",
    "solinas_booleanity_address_tile_ram3_4",
];
const PRODUCTION_TILE_PIPELINES_3: [&str; 10] = [
    "solinas_booleanity_address_tile_3_0",
    "solinas_booleanity_address_tile_3_1",
    "solinas_booleanity_address_tile_3_2",
    "solinas_booleanity_address_tile_3_3",
    "solinas_booleanity_address_tile_3_4",
    "solinas_booleanity_address_tile_3_5",
    "solinas_booleanity_address_tile_3_6",
    "solinas_booleanity_address_tile_3_7",
    "solinas_booleanity_address_tile_3_8",
    "solinas_booleanity_address_tile_3_9",
];
const FINALIZE_PIPELINE: &str = "solinas_booleanity_address_finalize";

#[derive(Clone, Copy)]
enum ProductionSelectorSchedule {
    TwoRam,
    ThreeRam,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BooleanityAddressPushforwardConfig {
    pub inner_log2: usize,
    pub selectors_per_tile: usize,
    pub tile_threads_per_threadgroup: Option<usize>,
    pub finalize_threads_per_threadgroup: Option<usize>,
}

impl Default for BooleanityAddressPushforwardConfig {
    fn default() -> Self {
        Self {
            inner_log2: 15,
            selectors_per_tile: 6,
            tile_threads_per_threadgroup: Some(512),
            finalize_threads_per_threadgroup: Some(1024),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    polys: u32,
    k: u32,
    e_in_length: u32,
    e_out_length: u32,
    selector_offset: u32,
    selectors_in_tile: u32,
    chunk_bits: u32,
    inc_bias: u64,
}

struct Buffers {
    rows: BooleanityRows,
    selectors: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    partials: Buffer,
    output: Buffer,
}

pub struct BooleanityAddressPushforward {
    context: SolinasMetal,
    tile_pipelines: Vec<ComputePipelineState>,
    finalize_pipeline: ComputePipelineState,
    buffers: Buffers,
    rows: usize,
    polys: usize,
    e_in_length: usize,
    e_out_length: usize,
    selectors_per_tile: usize,
    production_specialized: bool,
    tile_threads_per_threadgroup: usize,
    finalize_threads_per_threadgroup: usize,
    completed: Cell<bool>,
}

impl SolinasMetal {
    pub fn prepare_booleanity_address_pushforward(
        &self,
        rows: BooleanityRows,
        selectors: &[BooleanitySelector],
        reference_cycle: &[AkitaField],
        config: BooleanityAddressPushforwardConfig,
    ) -> Result<BooleanityAddressPushforward, MetalError> {
        if config.inner_log2 > reference_cycle.len() {
            return Err(MetalError::InvalidBooleanityAddressInnerLog2(
                config.inner_log2,
            ));
        }
        let split = reference_cycle.len() - config.inner_log2;
        let (out_point, in_point) = reference_cycle.split_at(split);
        let e_out = EqPolynomial::evals(out_point, None);
        let e_in = EqPolynomial::evals(in_point, None);
        self.prepare_booleanity_address_pushforward_with_weights(
            rows, selectors, &e_in, &e_out, config,
        )
    }

    pub(crate) fn prepare_booleanity_address_pushforward_with_weights(
        &self,
        rows: BooleanityRows,
        selectors: &[BooleanitySelector],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: BooleanityAddressPushforwardConfig,
    ) -> Result<BooleanityAddressPushforward, MetalError> {
        if self.offset != AKITA_OFFSET_FFFFA7F7 {
            return Err(MetalError::UnexpectedSolinasOffset {
                expected: AKITA_OFFSET_FFFFA7F7,
                got: self.offset,
            });
        }
        self.validate_booleanity_rows(&rows)?;
        let rows_len = rows.len();
        if rows_len == 0 || !rows_len.is_power_of_two() {
            return Err(MetalError::InvalidBooleanityRows(rows_len));
        }
        if selectors.is_empty() {
            return Err(MetalError::EmptyInput);
        }
        if config.selectors_per_tile == 0 || config.selectors_per_tile > MAX_SELECTORS_PER_TILE {
            return Err(MetalError::InvalidBooleanityAddressSelectorTile(
                config.selectors_per_tile,
            ));
        }
        if config.inner_log2 > MAX_INNER_LOG2 {
            return Err(MetalError::InvalidBooleanityAddressInnerLog2(
                config.inner_log2,
            ));
        }
        let e_in_length =
            1usize
                .checked_shl(u32::try_from(config.inner_log2).map_err(|_| {
                    MetalError::InvalidBooleanityAddressInnerLog2(config.inner_log2)
                })?)
                .ok_or(MetalError::InvalidBooleanityAddressInnerLog2(
                    config.inner_log2,
                ))?;
        if e_in.len() != e_in_length
            || e_out.is_empty()
            || e_in_length
                .checked_mul(e_out.len())
                .ok_or(MetalError::InputTooLong(rows_len))?
                != rows_len
        {
            return Err(MetalError::BooleanityWeightShape {
                expected: rows_len,
                covered: e_in.len().saturating_mul(e_out.len()),
            });
        }
        let chunk_bits = BINS.ilog2() as usize;
        let selector_abi = selectors
            .iter()
            .copied()
            .map(|selector| selector_abi(selector, chunk_bits))
            .collect::<Result<Vec<_>, _>>()?;
        let production_schedule = production_selector_schedule(selectors);
        let tile_pipeline_names = match (config.selectors_per_tile, production_schedule) {
            (MAX_SELECTORS_PER_TILE, Some(ProductionSelectorSchedule::TwoRam)) => {
                PRODUCTION_TILE_PIPELINES.to_vec()
            }
            (MAX_SELECTORS_PER_TILE, Some(ProductionSelectorSchedule::ThreeRam)) => {
                PRODUCTION_THREE_RAM_TILE_PIPELINES.to_vec()
            }
            (3, Some(ProductionSelectorSchedule::TwoRam)) => PRODUCTION_TILE_PIPELINES_3.to_vec(),
            _ => vec![TILE_PIPELINE],
        };
        let production_specialized = tile_pipeline_names.len() > 1;
        let tile_pipelines = tile_pipeline_names
            .iter()
            .map(|name| self.compile_named_pipeline(name))
            .collect::<Result<Vec<_>, _>>()?;
        let finalize_pipeline = self.compile_named_pipeline(FINALIZE_PIPELINE)?;
        let tile_pipeline_limits = tile_pipelines.iter().map(Self::limits).collect::<Vec<_>>();
        let tile_limits = PipelineLimits {
            thread_execution_width: tile_pipeline_limits[0].thread_execution_width,
            max_total_threads_per_threadgroup: tile_pipeline_limits
                .iter()
                .map(|limits| limits.max_total_threads_per_threadgroup)
                .min()
                .ok_or(MetalError::EmptyInput)?,
            static_threadgroup_memory_length: tile_pipeline_limits
                .iter()
                .map(|limits| limits.static_threadgroup_memory_length)
                .max()
                .ok_or(MetalError::EmptyInput)?,
        };
        let finalize_limits = Self::limits(&finalize_pipeline);
        for (pipeline, limits) in tile_pipeline_names
            .iter()
            .copied()
            .zip(tile_pipeline_limits.iter().copied())
            .chain([(FINALIZE_PIPELINE, finalize_limits)])
        {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedBooleanityExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let tile_threads_per_threadgroup =
            Self::resolve_threadgroup_width(config.tile_threads_per_threadgroup, tile_limits)?;
        let finalize_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.finalize_threads_per_threadgroup,
            finalize_limits,
        )?;
        if finalize_threads_per_threadgroup < BINS
            || !finalize_threads_per_threadgroup.is_multiple_of(BINS)
        {
            return Err(MetalError::InvalidBooleanityAddressFinalizeWidth(
                finalize_threads_per_threadgroup,
            ));
        }

        let accumulator_bytes = config
            .selectors_per_tile
            .checked_mul(BINS)
            .and_then(|value| value.checked_mul(ACCUMULATOR_WORDS))
            .and_then(|value| value.checked_mul(size_of::<u32>()))
            .ok_or(MetalError::InputTooLong(config.selectors_per_tile))?;
        let maximum_threadgroup_bytes = self.device.max_threadgroup_memory_length();
        let tile_threadgroup_bytes = tile_limits
            .static_threadgroup_memory_length
            .checked_add(accumulator_bytes as u64)
            .ok_or(MetalError::InputTooLong(accumulator_bytes))?;
        if tile_threadgroup_bytes > maximum_threadgroup_bytes {
            return Err(MetalError::BooleanityAddressThreadgroupMemory {
                requested: tile_threadgroup_bytes,
                maximum: maximum_threadgroup_bytes,
            });
        }
        let finalize_bytes = finalize_threads_per_threadgroup
            .checked_mul(size_of::<Fp128>())
            .ok_or(MetalError::InputTooLong(finalize_threads_per_threadgroup))?;
        let finalize_threadgroup_bytes = finalize_limits
            .static_threadgroup_memory_length
            .checked_add(finalize_bytes as u64)
            .ok_or(MetalError::InputTooLong(finalize_bytes))?;
        if finalize_threadgroup_bytes > maximum_threadgroup_bytes {
            return Err(MetalError::BooleanityAddressThreadgroupMemory {
                requested: finalize_threadgroup_bytes,
                maximum: maximum_threadgroup_bytes,
            });
        }

        let partial_elements = e_out
            .len()
            .checked_mul(config.selectors_per_tile)
            .and_then(|value| value.checked_mul(BINS))
            .ok_or(MetalError::InputTooLong(e_out.len()))?;
        let output_elements = selectors
            .len()
            .checked_mul(BINS)
            .ok_or(MetalError::InputTooLong(selectors.len()))?;
        let selector_bytes = byte_length::<super::booleanity::SelectorAbi>(selector_abi.len())?;
        let e_in_bytes = byte_length::<Fp128>(e_in.len())?;
        let e_out_bytes = byte_length::<Fp128>(e_out.len())?;
        let partial_bytes = byte_length::<Fp128>(partial_elements)?;
        let output_bytes = byte_length::<Fp128>(output_elements)?;
        for bytes in [
            selector_bytes,
            e_in_bytes,
            e_out_bytes,
            partial_bytes,
            output_bytes,
        ] {
            self.validate_buffer_length(bytes)?;
        }
        let additional = [
            selector_bytes,
            e_in_bytes,
            e_out_bytes,
            partial_bytes,
            output_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(MetalError::InputTooLong(output_elements))
        })?;
        self.validate_additional_working_set(additional)?;

        let selectors_buffer = self.device.new_buffer_with_data(
            selector_abi.as_ptr().cast(),
            selector_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let e_in_buffer = self
            .device
            .new_buffer(e_in_bytes, MTLResourceOptions::StorageModeShared);
        let e_out_buffer = self
            .device
            .new_buffer(e_out_bytes, MTLResourceOptions::StorageModeShared);
        write_fields(&e_in_buffer, e_in.len(), e_in)?;
        write_fields(&e_out_buffer, e_out.len(), e_out)?;

        Ok(BooleanityAddressPushforward {
            context: self.clone(),
            tile_pipelines,
            finalize_pipeline,
            buffers: Buffers {
                rows,
                selectors: selectors_buffer,
                e_in: e_in_buffer,
                e_out: e_out_buffer,
                partials: self
                    .device
                    .new_buffer(partial_bytes, MTLResourceOptions::StorageModeShared),
                output: self
                    .device
                    .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared),
            },
            rows: rows_len,
            polys: selectors.len(),
            e_in_length,
            e_out_length: e_out.len(),
            selectors_per_tile: config.selectors_per_tile,
            production_specialized,
            tile_threads_per_threadgroup,
            finalize_threads_per_threadgroup,
            completed: Cell::new(false),
        })
    }
}

impl BooleanityAddressPushforward {
    copy_field_getters! { pub, {
        tile_threads_per_threadgroup: usize,
        finalize_threads_per_threadgroup: usize,
        uses_production_specialization => production_specialized: bool,
        row_count => rows: usize,
        polys: usize,
        selectors_per_tile: usize,
        e_in_length: usize,
        e_out_length: usize,
    }}

    pub const fn selector_tiles(&self) -> usize {
        self.polys.div_ceil(self.selectors_per_tile)
    }

    pub fn resident_row_identity(&self) -> usize {
        self.buffers.rows.allocation_identity()
    }

    pub const fn output_elements(&self) -> usize {
        self.polys * BINS
    }

    pub const fn partial_bytes(&self) -> u64 {
        (self.e_out_length * self.selectors_per_tile * BINS * size_of::<Fp128>()) as u64
    }

    pub fn static_buffer_identities(&self) -> [usize; 5] {
        [
            self.buffers.selectors.as_ptr() as usize,
            self.buffers.e_in.as_ptr() as usize,
            self.buffers.e_out.as_ptr() as usize,
            self.buffers.partials.as_ptr() as usize,
            self.buffers.output.as_ptr() as usize,
        ]
    }

    pub fn execute(&self) -> Result<(), MetalError> {
        self.execute_timed().map(|_| ())
    }

    pub fn execute_timed(&self) -> Result<Duration, MetalError> {
        self.completed.set(false);
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            for (tile_index, selector_offset) in
                (0..self.polys).step_by(self.selectors_per_tile).enumerate()
            {
                let selectors_in_tile = self.selectors_per_tile.min(self.polys - selector_offset);
                let params = Params {
                    rows: u32::try_from(self.rows)
                        .map_err(|_| MetalError::InputTooLong(self.rows))?,
                    polys: u32::try_from(self.polys)
                        .map_err(|_| MetalError::InputTooLong(self.polys))?,
                    k: BINS as u32,
                    e_in_length: u32::try_from(self.e_in_length)
                        .map_err(|_| MetalError::InputTooLong(self.e_in_length))?,
                    e_out_length: u32::try_from(self.e_out_length)
                        .map_err(|_| MetalError::InputTooLong(self.e_out_length))?,
                    selector_offset: u32::try_from(selector_offset)
                        .map_err(|_| MetalError::InputTooLong(selector_offset))?,
                    selectors_in_tile: u32::try_from(selectors_in_tile)
                        .map_err(|_| MetalError::InputTooLong(selectors_in_tile))?,
                    chunk_bits: BINS.ilog2(),
                    inc_bias: balanced_bias(BINS.ilog2() as usize),
                };

                let tile = command_buffer.new_compute_command_encoder();
                let pipeline_index = if self.production_specialized {
                    tile_index
                } else {
                    0
                };
                tile.set_compute_pipeline_state(&self.tile_pipelines[pipeline_index]);
                tile.set_buffer(0, Some(self.buffers.rows.buffer()), 0);
                tile.set_buffer(1, Some(&self.buffers.selectors), 0);
                tile.set_buffer(2, Some(&self.buffers.e_in), 0);
                tile.set_buffer(3, Some(&self.buffers.e_out), 0);
                tile.set_buffer(4, Some(&self.buffers.partials), 0);
                set_inline_bytes(tile, 5, &params);
                tile.set_threadgroup_memory_length(
                    0,
                    (selectors_in_tile * BINS * ACCUMULATOR_WORDS * size_of::<u32>()) as u64,
                );
                tile.dispatch_thread_groups(
                    MTLSize {
                        width: self.e_out_length as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.tile_threads_per_threadgroup as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                tile.end_encoding();

                let finalize = command_buffer.new_compute_command_encoder();
                finalize.set_compute_pipeline_state(&self.finalize_pipeline);
                finalize.set_buffer(0, Some(&self.buffers.partials), 0);
                finalize.set_buffer(1, Some(&self.buffers.output), 0);
                set_inline_bytes(finalize, 2, &params);
                finalize.set_threadgroup_memory_length(
                    0,
                    (self.finalize_threads_per_threadgroup * size_of::<Fp128>()) as u64,
                );
                finalize.dispatch_thread_groups(
                    MTLSize {
                        width: selectors_in_tile as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.finalize_threads_per_threadgroup as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                finalize.end_encoding();
            }

            command_buffer.commit();
            command_buffer.wait_until_completed();
            let gpu_active = completed_command_gpu_time(command_buffer)?;
            self.completed.set(true);
            Ok(gpu_active)
        })
    }

    pub fn read_masses(&self) -> Result<Vec<AkitaField>, MetalError> {
        let mut output = vec![AkitaField::zero(); self.output_elements()];
        self.read_masses_into(&mut output)?;
        Ok(output)
    }

    pub fn read_masses_into(&self, output: &mut [AkitaField]) -> Result<(), MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        if output.len() != self.output_elements() {
            return Err(MetalError::LengthMismatch {
                lhs: output.len(),
                rhs: self.output_elements(),
            });
        }
        // SAFETY: the shared output buffer contains exactly `output_elements`
        // fields and the command buffer has completed.
        let values = unsafe {
            slice::from_raw_parts(
                self.buffers.output.contents().cast::<Fp128>(),
                self.output_elements(),
            )
        };
        self.context
            .validate_inputs("Booleanity address output", values)?;
        for (output, value) in output.iter_mut().zip(values) {
            *output = value.into_jolt_field();
        }
        Ok(())
    }
}

fn production_selector_schedule(
    selectors: &[BooleanitySelector],
) -> Option<ProductionSelectorSchedule> {
    let (ram_chunks, schedule) = match selectors.len() {
        29 => (2, ProductionSelectorSchedule::TwoRam),
        30 => (3, ProductionSelectorSchedule::ThreeRam),
        _ => return None,
    };
    let inc_start = 18 + ram_chunks;
    selectors
        .iter()
        .copied()
        .enumerate()
        .all(|(index, selector)| {
            let expected = if index < 16 {
                BooleanitySelector::Lookup {
                    shift: (8 * (15 - index)) as u32,
                }
            } else if index < 18 {
                BooleanitySelector::Bytecode {
                    shift: (8 * (17 - index)) as u32,
                }
            } else if index < inc_start {
                BooleanitySelector::Ram {
                    shift: (8 * (inc_start - 1 - index)) as u32,
                }
            } else if index < inc_start + 8 {
                BooleanitySelector::FusedInc {
                    shift: (8 * (index - inc_start)) as u32,
                }
            } else if index == inc_start + 8 {
                BooleanitySelector::FusedIncMsb
            } else {
                return false;
            };
            selector == expected
        })
        .then_some(schedule)
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

const _: () = assert!(size_of::<Params>() == 40);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::{AkitaField, FromPrimitiveInt};

    use super::{
        balanced_bias, BooleanityAddressPushforwardConfig, BooleanitySelector, SolinasMetal, BINS,
    };
    use crate::metal::solinas::BooleanityRow;

    #[test]
    fn pushforward_matches_exact_cpu_oracle_across_selector_tiles() {
        let rows = rows(1 << 12);
        let e_in = fields(1 << 10, 0x1234_5678_9abc_def0);
        let e_out = fields(4, 0x0ddc_0ffe_e15e_beef);
        let context = SolinasMetal::for_akita().unwrap();
        let resident = context.prepare_booleanity_rows(&rows).unwrap();
        let resident_identity = resident.allocation_identity();

        for ram_chunks in [2, 3] {
            let selectors = selectors(ram_chunks);
            let expected = oracle(&rows, &selectors, &e_in, &e_out);
            for selectors_per_tile in [1, 3, 4, 6] {
                let invocation = context
                    .prepare_booleanity_address_pushforward_with_weights(
                        resident.clone(),
                        &selectors,
                        &e_in,
                        &e_out,
                        BooleanityAddressPushforwardConfig {
                            inner_log2: 10,
                            selectors_per_tile,
                            tile_threads_per_threadgroup: Some(256),
                            finalize_threads_per_threadgroup: Some(256),
                        },
                    )
                    .unwrap();
                assert_eq!(invocation.resident_row_identity(), resident_identity);
                assert_eq!(
                    invocation.selector_tiles(),
                    selectors.len().div_ceil(selectors_per_tile)
                );
                assert_eq!(
                    invocation.uses_production_specialization(),
                    selectors_per_tile == 6 || (ram_chunks == 2 && selectors_per_tile == 3)
                );
                let identities = invocation.static_buffer_identities();
                invocation.execute().unwrap();
                assert_eq!(invocation.static_buffer_identities(), identities);
                assert_eq!(invocation.read_masses().unwrap(), expected);
            }
        }
    }

    #[test]
    fn pushforward_reduces_adversarial_carry_bucket() {
        let rows = vec![BooleanityRow::new(0, Some(0), Some(0), 0).unwrap(); 1 << 12];
        let selectors = vec![BooleanitySelector::Lookup { shift: 0 }; 6];
        let near_modulus = AkitaField::from_u128(u128::MAX - 0xffff_a7f7);
        let e_in = vec![near_modulus; rows.len()];
        let e_out = vec![AkitaField::from_u64(17)];
        let expected = oracle(&rows, &selectors, &e_in, &e_out);
        let context = SolinasMetal::for_akita().unwrap();
        let resident = context.prepare_booleanity_rows(&rows).unwrap();
        let invocation = context
            .prepare_booleanity_address_pushforward_with_weights(
                resident,
                &selectors,
                &e_in,
                &e_out,
                BooleanityAddressPushforwardConfig {
                    inner_log2: 12,
                    selectors_per_tile: 6,
                    tile_threads_per_threadgroup: Some(1024),
                    finalize_threads_per_threadgroup: Some(1024),
                },
            )
            .unwrap();
        invocation.execute().unwrap();
        assert_eq!(invocation.read_masses().unwrap(), expected);
    }

    fn selectors(ram_chunks: usize) -> Vec<BooleanitySelector> {
        let mut selectors = (0..16)
            .map(|index| BooleanitySelector::Lookup {
                shift: (8 * (15 - index)) as u32,
            })
            .collect::<Vec<_>>();
        selectors.extend([8, 0].map(|shift| BooleanitySelector::Bytecode { shift }));
        selectors.extend((0..ram_chunks).map(|index| BooleanitySelector::Ram {
            shift: (8 * (ram_chunks - 1 - index)) as u32,
        }));
        selectors.extend((0..8).map(|index| BooleanitySelector::FusedInc {
            shift: (8 * index) as u32,
        }));
        selectors.push(BooleanitySelector::FusedIncMsb);
        selectors
    }

    fn rows(count: usize) -> Vec<BooleanityRow> {
        let mut state = 0xa11a_5eed_0123_4567u128;
        (0..count)
            .map(|row| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 43;
                let pc = (!row.is_multiple_of(7)).then_some(((state >> 61) as u64) & 0x1ffe);
                let ram =
                    (!row.is_multiple_of(11)).then_some((state as u64) & (u32::MAX as u64 - 1));
                let inc = match row % 4 {
                    0 => row as i128,
                    1 => -(row as i128),
                    2 => u64::MAX as i128 - row as i128,
                    _ => -(u64::MAX as i128) + row as i128,
                };
                BooleanityRow::new(state, pc, ram, inc).unwrap()
            })
            .collect()
    }

    fn fields(count: usize, mut state: u64) -> Vec<AkitaField> {
        (0..count)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut value = state;
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                value ^= value >> 31;
                AkitaField::from_u128(
                    u128::from(value) | (u128::from(!value) << 64 & (u128::MAX >> 1)),
                )
            })
            .collect()
    }

    fn oracle(
        rows: &[BooleanityRow],
        selectors: &[BooleanitySelector],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Vec<AkitaField> {
        let mut output = vec![AkitaField::zero(); selectors.len() * BINS];
        for (row_index, row) in rows.iter().copied().enumerate() {
            let x_out = row_index / e_in.len();
            let x_in = row_index % e_in.len();
            let weight = e_out[x_out] * e_in[x_in];
            for (selector_index, selector) in selectors.iter().copied().enumerate() {
                if let Some(hot) = hot_index(row, selector) {
                    output[selector_index * BINS + hot] += weight;
                }
            }
        }
        output
    }

    fn hot_index(row: BooleanityRow, selector: BooleanitySelector) -> Option<usize> {
        let words = row.words();
        let mask = (BINS - 1) as u64;
        match selector {
            BooleanitySelector::Lookup { shift } => {
                let word = if shift < 64 { words[0] } else { words[1] };
                let shift = if shift < 64 { shift } else { shift - 64 };
                Some(((word >> shift) & mask) as usize)
            }
            BooleanitySelector::Bytecode { shift } => {
                let plus_one = words[4] & 0x00ff_ffff_ffff_ffff;
                (plus_one != 0).then(|| (((plus_one - 1) >> shift) & mask) as usize)
            }
            BooleanitySelector::Ram { shift } => {
                (words[2] != 0).then(|| (((words[2] - 1) >> shift) & mask) as usize)
            }
            BooleanitySelector::FusedInc { shift } => {
                let (biased, _) = biased_inc(words);
                let standard = (biased >> shift) & mask;
                Some(((standard + (BINS / 2) as u64) & mask) as usize)
            }
            BooleanitySelector::FusedIncMsb => {
                let (_, carry) = biased_inc(words);
                Some((carry as usize) & (BINS - 1))
            }
        }
    }

    fn biased_inc(words: [u64; 5]) -> (u64, i32) {
        let bias = balanced_bias(BINS.ilog2() as usize);
        let magnitude = words[3];
        if words[4] >> 63 != 0 {
            (
                bias.wrapping_sub(magnitude),
                if magnitude > bias { -1 } else { 0 },
            )
        } else {
            let biased = bias.wrapping_add(magnitude);
            (biased, i32::from(biased < bias))
        }
    }
}
