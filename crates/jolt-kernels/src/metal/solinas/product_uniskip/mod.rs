//! Two-node Metal front for Spartan product uni-skip.
//!
//! Stage 1 already supplies the values at the three base-domain nodes. This
//! module computes only the two remaining evaluations at `-2` and `2`.

use std::{
    mem::{align_of, size_of},
    slice,
    time::Duration,
};

use jolt_field::signed::{S128, S192, S256};
use jolt_field::{AkitaField, Field, SignedProductAccumulator as _, WithSignedProductAccumulator};
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use thiserror::Error;

pub use super::product_remainder::{ProductRemainderRow, ProductRemainderRows};
use super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub const PRODUCT_UNISKIP_EXTENDED_NODES: usize = 2;
pub const PRODUCT_UNISKIP_SIMD_WIDTH: usize = 32;
pub const PRODUCT_UNISKIP_NODE_ORDER: [i64; PRODUCT_UNISKIP_EXTENDED_NODES] = [-2, 2];
pub const PRODUCT_UNISKIP_EXTENSION_COEFFICIENTS: [[i64; 3]; 2] = [[3, -3, 1], [1, -3, 3]];

pub(crate) const BLOCKS_PIPELINE: &str = "solinas_product_uniskip_extended_blocks2";
pub(crate) const STAGE1_BLOCKS_PIPELINE: &str = "solinas_product_uniskip_stage1_extended_blocks2";
pub(crate) const REDUCTION_PIPELINE: &str = "solinas_product_uniskip_reduce2";

const _: [(); 40] = [(); size_of::<ProductRemainderRow>()];
const _: [(); 8] = [(); align_of::<ProductRemainderRow>()];

fn cpu_extended_product(row: ProductRemainderRow, coefficients: &[i64; 3]) -> S256 {
    let left = i128::from(coefficients[0]) * i128::from(row.left_instruction_input())
        + i128::from(coefficients[1]) * i128::from(row.lookup_output())
        + i128::from(coefficients[2]) * i128::from(u8::from(row.jump()));
    let right_wide = S192::from_i128(row.right_instruction_input());
    let mut right = S192::from_i64(coefficients[0]).mul_trunc::<3, 3>(&right_wide);
    right += S192::from_i64(
        coefficients[1] * i64::from(u8::from(row.branch()))
            + coefficients[2] * i64::from(u8::from(!row.next_is_noop())),
    );
    S128::from_i128(left).mul_trunc::<3, 4>(&right)
}

pub fn evaluate_product_uniskip_extensions_cpu(
    rows: &[ProductRemainderRow],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> Result<ProductUniskipExtendedNodes<AkitaField>, ProductUniskipShapeError> {
    let _ = ProductUniskipBlockParams::new(rows.len(), e_in.len(), e_out.len())?;
    let block = |x_out: usize| {
        let mut accumulators: [
            <AkitaField as WithSignedProductAccumulator>::SignedProductAccumulator;
            PRODUCT_UNISKIP_EXTENDED_NODES
        ] = Default::default();
        for (x_in, &weight) in e_in.iter().enumerate() {
            let row = rows[x_out * e_in.len() + x_in];
            for (accumulator, coefficients) in accumulators
                .iter_mut()
                .zip(&PRODUCT_UNISKIP_EXTENSION_COEFFICIENTS)
            {
                accumulator.fmadd_s256(weight, &cpu_extended_product(row, coefficients));
            }
        }
        std::array::from_fn(|node| e_out[x_out] * accumulators[node].reduce())
    };
    let merge = |mut left: [AkitaField; PRODUCT_UNISKIP_EXTENDED_NODES], right| {
        for (left, right) in left.iter_mut().zip(right) {
            *left += right;
        }
        left
    };
    #[cfg(feature = "parallel")]
    let values = (0..e_out.len()).into_par_iter().map(block).reduce(
        || [AkitaField::zero(); PRODUCT_UNISKIP_EXTENDED_NODES],
        merge,
    );
    #[cfg(not(feature = "parallel"))]
    let values = (0..e_out.len())
        .map(block)
        .fold([AkitaField::zero(); PRODUCT_UNISKIP_EXTENDED_NODES], merge);
    Ok(ProductUniskipExtendedNodes {
        minus_two: values[0],
        plus_two: values[1],
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductUniskipConfig {
    pub threads_per_threadgroup: Option<usize>,
}

impl Default for ProductUniskipConfig {
    fn default() -> Self {
        Self {
            threads_per_threadgroup: Some(64),
        }
    }
}

/// The three base-domain values supplied by stage 1.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductUniskipKnownNodes<F> {
    pub product: F,
    pub should_branch: F,
    pub should_jump: F,
}

impl<F: Copy> ProductUniskipKnownNodes<F> {
    /// Returns values in centered-domain order `[-1, 0, 1]`.
    pub const fn as_array(self) -> [F; 3] {
        [self.product, self.should_branch, self.should_jump]
    }
}

/// The two evaluations returned by Metal, ordered as `[-2, 2]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductUniskipExtendedNodes<F> {
    pub minus_two: F,
    pub plus_two: F,
}

impl<F: Copy> ProductUniskipExtendedNodes<F> {
    pub const fn as_array(self) -> [F; PRODUCT_UNISKIP_EXTENDED_NODES] {
        [self.minus_two, self.plus_two]
    }

    /// Builds the five evaluations needed to interpolate `t1`.
    pub const fn assemble(self, known: ProductUniskipKnownNodes<F>) -> [F; 5] {
        [
            self.minus_two,
            known.product,
            known.should_branch,
            known.should_jump,
            self.plus_two,
        ]
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ProductUniskipShapeError {
    #[error("product uni-skip needs a nonzero power-of-two row count, got {0}")]
    InvalidRows(usize),
    #[error(
        "product uni-skip {phase} weights have e_in={e_in}, e_out={e_out}; expected product {expected}"
    )]
    WeightShape {
        phase: &'static str,
        expected: usize,
        e_in: usize,
        e_out: usize,
    },
    #[error("product uni-skip {name} storage has length {got}, expected {expected}")]
    StorageLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("product uni-skip {name} element count exceeds its 32-bit shader index")]
    ShaderIndexOverflow { name: &'static str },
    #[error("product uni-skip {name} byte length overflows host indexing")]
    ByteLengthOverflow { name: &'static str },
    #[error("product uni-skip reduction only supports two columns, got {0}")]
    InvalidReductionColumns(usize),
    #[error("product uni-skip reduction needs at least one input")]
    EmptyReduction,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProductUniskipBlockParams {
    pub(crate) rows: u32,
    pub(crate) e_in_length: u32,
    pub(crate) e_out_length: u32,
    pub(crate) _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProductUniskipReductionParams {
    pub(crate) input_count: u32,
    pub(crate) output_count: u32,
    pub(crate) columns: u32,
    pub(crate) _reserved: u32,
}

const _: [(); 16] = [(); size_of::<ProductUniskipBlockParams>()];
const _: [(); 16] = [(); size_of::<ProductUniskipReductionParams>()];

impl ProductUniskipBlockParams {
    pub(crate) fn new(
        rows: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, ProductUniskipShapeError> {
        validate_rows(rows)?;
        validate_weight_shape("blocks", rows, e_in_length, e_out_length)?;
        validate_partial_index(PRODUCT_UNISKIP_EXTENDED_NODES, e_out_length)?;
        Ok(Self {
            rows: shader_count("rows", rows)?,
            e_in_length: shader_count("e_in", e_in_length)?,
            e_out_length: shader_count("e_out", e_out_length)?,
            _reserved: 0,
        })
    }
}

impl ProductUniskipReductionParams {
    pub(crate) fn new(
        input_count: usize,
        columns: usize,
    ) -> Result<Self, ProductUniskipShapeError> {
        if columns != PRODUCT_UNISKIP_EXTENDED_NODES {
            return Err(ProductUniskipShapeError::InvalidReductionColumns(columns));
        }
        if input_count == 0 {
            return Err(ProductUniskipShapeError::EmptyReduction);
        }
        let output_count = input_count.div_ceil(PRODUCT_UNISKIP_SIMD_WIDTH);
        validate_partial_index(columns, input_count)?;
        validate_partial_index(columns, output_count)?;
        Ok(Self {
            input_count: shader_count("reduction input", input_count)?,
            output_count: shader_count("reduction output", output_count)?,
            columns: shader_count("reduction columns", columns)?,
            _reserved: 0,
        })
    }
}

/// Storage needed by the two-node sequence.
///
/// `shared_row_bytes` belongs to the row allocation also consumed by product
/// remainder. `scratch_bytes` covers both equality tables and two partial
/// buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductUniskipStorageLayout {
    rows: usize,
    shared_row_bytes: usize,
    e_in_fields: usize,
    e_out_fields: usize,
    partial_fields: usize,
    scratch_bytes: usize,
    resident_bytes: usize,
}

impl ProductUniskipStorageLayout {
    pub fn new(
        rows: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
    ) -> Result<Self, ProductUniskipShapeError> {
        validate_rows(rows)?;
        let covered = e_in_capacity.checked_mul(e_out_capacity).ok_or(
            ProductUniskipShapeError::ByteLengthOverflow {
                name: "weight capacity",
            },
        )?;
        if e_in_capacity == 0 || e_out_capacity == 0 || covered < rows {
            return Err(ProductUniskipShapeError::WeightShape {
                phase: "storage capacity",
                expected: rows,
                e_in: e_in_capacity,
                e_out: e_out_capacity,
            });
        }

        let partial_fields = checked_product(
            "partial buffer",
            PRODUCT_UNISKIP_EXTENDED_NODES,
            e_out_capacity,
        )?;
        for (name, fields) in [
            ("e_in", e_in_capacity),
            ("e_out", e_out_capacity),
            ("partial buffer", partial_fields),
        ] {
            let _ = shader_count(name, fields)?;
        }

        let scratch_fields = [
            e_in_capacity,
            e_out_capacity,
            partial_fields,
            partial_fields,
        ]
        .into_iter()
        .try_fold(0usize, |sum, fields| sum.checked_add(fields))
        .ok_or(ProductUniskipShapeError::ByteLengthOverflow { name: "scratch" })?;
        let scratch_bytes = checked_product("scratch", scratch_fields, size_of::<super::Fp128>())?;
        let shared_row_bytes =
            checked_product("shared rows", rows, size_of::<ProductRemainderRow>())?;
        let resident_bytes = shared_row_bytes.checked_add(scratch_bytes).ok_or(
            ProductUniskipShapeError::ByteLengthOverflow {
                name: "resident set",
            },
        )?;

        Ok(Self {
            rows,
            shared_row_bytes,
            e_in_fields: e_in_capacity,
            e_out_fields: e_out_capacity,
            partial_fields,
            scratch_bytes,
            resident_bytes,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn shared_row_bytes(self) -> usize {
        self.shared_row_bytes
    }

    pub const fn e_in_fields(self) -> usize {
        self.e_in_fields
    }

    pub const fn e_out_fields(self) -> usize {
        self.e_out_fields
    }

    pub const fn partial_fields(self) -> usize {
        self.partial_fields
    }

    pub const fn scratch_bytes(self) -> usize {
        self.scratch_bytes
    }

    pub const fn resident_bytes(self) -> usize {
        self.resident_bytes
    }
}

struct ProductUniskipBuffers {
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct ProductUniskipInvocation {
    context: SolinasMetal,
    blocks_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    blocks_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    rows: ProductRemainderRows,
    buffers: ProductUniskipBuffers,
    layout: ProductUniskipStorageLayout,
    e_in_length: usize,
    e_out_length: usize,
    threads_per_threadgroup: usize,
}

impl SolinasMetal {
    pub fn prepare_product_uniskip(
        &self,
        rows: &ProductRemainderRows,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: ProductUniskipConfig,
    ) -> Result<ProductUniskipInvocation, MetalError> {
        if rows.device_registry_id() != self.device_registry_id() {
            return Err(MetalError::ProductUniskipRowsDevice {
                expected: self.device_registry_id(),
                got: rows.device_registry_id(),
            });
        }
        let params = ProductUniskipBlockParams::new(rows.len(), e_in.len(), e_out.len())?;
        let layout = ProductUniskipStorageLayout::new(rows.len(), e_in.len(), e_out.len())?;
        let scratch_bytes = u64::try_from(layout.scratch_bytes())
            .map_err(|_| MetalError::InputTooLong(layout.scratch_bytes()))?;
        self.validate_additional_working_set(scratch_bytes)?;

        let blocks_pipeline_name = if rows.stage1_buffers().is_some() {
            STAGE1_BLOCKS_PIPELINE
        } else {
            BLOCKS_PIPELINE
        };
        let blocks_pipeline = self.compile_named_pipeline(blocks_pipeline_name)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCTION_PIPELINE)?;
        let blocks_limits = Self::limits(&blocks_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (blocks_pipeline_name, blocks_limits),
            (REDUCTION_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != PRODUCT_UNISKIP_SIMD_WIDTH {
                return Err(MetalError::UnsupportedProductUniskipExecutionWidth {
                    pipeline,
                    expected: PRODUCT_UNISKIP_SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, blocks_limits)?;

        let e_in = e_in.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();
        let e_out = e_out.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();
        self.validate_inputs("product uni-skip e_in", &e_in)?;
        self.validate_inputs("product uni-skip e_out", &e_out)?;
        let partial_bytes = field_bytes(layout.partial_fields())?;
        self.validate_buffer_length(partial_bytes)?;

        Ok(ProductUniskipInvocation {
            context: self.clone(),
            blocks_pipeline,
            reduction_pipeline,
            blocks_limits,
            reduction_limits,
            rows: rows.clone(),
            buffers: ProductUniskipBuffers {
                e_in: buffer_from_slice(&self.device, &e_in),
                e_out: buffer_from_slice(&self.device, &e_out),
                partial_a: self
                    .device
                    .new_buffer(partial_bytes, MTLResourceOptions::StorageModeShared),
                partial_b: self
                    .device
                    .new_buffer(partial_bytes, MTLResourceOptions::StorageModeShared),
            },
            layout,
            e_in_length: params.e_in_length as usize,
            e_out_length: params.e_out_length as usize,
            threads_per_threadgroup,
        })
    }
}

impl ProductUniskipInvocation {
    pub fn execute(&self) -> Result<ProductUniskipExtendedNodes<AkitaField>, MetalError> {
        self.execute_timed().map(|(values, _)| values)
    }

    pub fn execute_timed(
        &self,
    ) -> Result<(ProductUniskipExtendedNodes<AkitaField>, Duration), MetalError> {
        let params = ProductUniskipBlockParams::new(
            self.layout.rows(),
            self.e_in_length,
            self.e_out_length,
        )?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.blocks_pipeline);
            if let Some((compact, residual)) = self.rows.stage1_buffers() {
                encoder.set_buffer(0, Some(compact), 0);
                encoder.set_buffer(1, Some(residual), 0);
                encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 5, &params);
            } else {
                let rows =
                    self.rows
                        .packed_buffer()
                        .ok_or(MetalError::InvalidProductRemainderState(
                            "packed product uni-skip lost its row buffer",
                        ))?;
                encoder.set_buffer(0, Some(rows), 0);
                encoder.set_buffer(1, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(2, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(3, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 4, &params);
            }
            encoder.set_threadgroup_memory_length(
                0,
                product_uniskip_threadgroup_bytes(self.threads_per_threadgroup) as u64,
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.e_out_length as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );

            let final_in_a = encode_product_uniskip_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                self.e_out_length,
            )?;
            encoder.end_encoding();
            finish_product_uniskip_command(
                &self.context,
                command_buffer,
                if final_in_a {
                    &self.buffers.partial_a
                } else {
                    &self.buffers.partial_b
                },
            )
        })
    }

    pub const fn storage_layout(&self) -> ProductUniskipStorageLayout {
        self.layout
    }

    pub fn resident_buffer_count(&self) -> usize {
        self.rows.allocation_identities().len() + 4
    }

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn blocks_pipeline_limits(&self) -> PipelineLimits {
        self.blocks_limits
    }

    pub const fn reduction_pipeline_limits(&self) -> PipelineLimits {
        self.reduction_limits
    }

    pub const fn useful_multiplications(&self) -> usize {
        4 * self.layout.rows() + PRODUCT_UNISKIP_EXTENDED_NODES * self.e_out_length
    }

    pub fn row_allocation_identity(&self) -> usize {
        self.rows.allocation_identity()
    }
}

fn product_uniskip_threadgroup_bytes(threads_per_threadgroup: usize) -> usize {
    PRODUCT_UNISKIP_EXTENDED_NODES
        * (threads_per_threadgroup / PRODUCT_UNISKIP_SIMD_WIDTH)
        * size_of::<Fp128>()
}

fn encode_product_uniskip_reductions(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    partial_a: &Buffer,
    partial_b: &Buffer,
    mut input_count: usize,
) -> Result<bool, MetalError> {
    let mut input_a = true;
    while input_count > 1 {
        let params =
            ProductUniskipReductionParams::new(input_count, PRODUCT_UNISKIP_EXTENDED_NODES)?;
        let output_count = params.output_count as usize;
        encoder.set_compute_pipeline_state(pipeline);
        let (input, output) = if input_a {
            (partial_a, partial_b)
        } else {
            (partial_b, partial_a)
        };
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        set_inline_bytes(encoder, 2, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: output_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: PRODUCT_UNISKIP_SIMD_WIDTH as u64,
                height: 1,
                depth: 1,
            },
        );
        input_count = output_count;
        input_a = !input_a;
    }
    Ok(input_a)
}

fn finish_product_uniskip_command(
    context: &SolinasMetal,
    command_buffer: &metal::CommandBufferRef,
    output: &Buffer,
) -> Result<(ProductUniskipExtendedNodes<AkitaField>, Duration), MetalError> {
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let status = command_buffer.status();
    if status != MTLCommandBufferStatus::Completed {
        return Err(MetalError::CommandFailed(status));
    }
    let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
    let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
        return Err(MetalError::InvalidGpuTimestamps { start, end });
    }
    let values = unsafe {
        // SAFETY: the completed reduction leaves two fields at the front of
        // the selected shared buffer.
        slice::from_raw_parts(
            output.contents().cast::<Fp128>(),
            PRODUCT_UNISKIP_EXTENDED_NODES,
        )
    };
    context.validate_inputs("product uni-skip extended nodes", values)?;
    Ok((
        ProductUniskipExtendedNodes {
            minus_two: values[0].into_jolt_field(),
            plus_two: values[1].into_jolt_field(),
        },
        Duration::from_secs_f64(end - start),
    ))
}

fn field_bytes(fields: usize) -> Result<u64, MetalError> {
    fields
        .checked_mul(size_of::<Fp128>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(fields))
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

fn validate_rows(rows: usize) -> Result<(), ProductUniskipShapeError> {
    if rows == 0 || !rows.is_power_of_two() {
        return Err(ProductUniskipShapeError::InvalidRows(rows));
    }
    let _ = shader_count("rows", rows)?;
    Ok(())
}

fn validate_weight_shape(
    phase: &'static str,
    expected: usize,
    e_in: usize,
    e_out: usize,
) -> Result<(), ProductUniskipShapeError> {
    let covered = e_in.checked_mul(e_out);
    if e_in == 0 || e_out == 0 || covered != Some(expected) {
        return Err(ProductUniskipShapeError::WeightShape {
            phase,
            expected,
            e_in,
            e_out,
        });
    }
    Ok(())
}

fn validate_partial_index(
    columns: usize,
    fields_per_column: usize,
) -> Result<(), ProductUniskipShapeError> {
    let fields = checked_product("partial buffer", columns, fields_per_column)?;
    let _ = shader_count("partial buffer", fields)?;
    Ok(())
}

fn shader_count(name: &'static str, value: usize) -> Result<u32, ProductUniskipShapeError> {
    u32::try_from(value).map_err(|_| ProductUniskipShapeError::ShaderIndexOverflow { name })
}

fn checked_product(
    name: &'static str,
    lhs: usize,
    rhs: usize,
) -> Result<usize, ProductUniskipShapeError> {
    lhs.checked_mul(rhs)
        .ok_or(ProductUniskipShapeError::ByteLengthOverflow { name })
}

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod reference {
    use super::*;

    pub fn extended_node_values<F: Field>(
        rows: &[ProductRemainderRow],
        e_in: &[F],
        e_out: &[F],
    ) -> Result<ProductUniskipExtendedNodes<F>, ProductUniskipShapeError> {
        let _ = ProductUniskipBlockParams::new(rows.len(), e_in.len(), e_out.len())?;
        let coefficients = PRODUCT_UNISKIP_EXTENSION_COEFFICIENTS.map(|row| {
            row.map(|coefficient| {
                if coefficient < 0 {
                    -F::from_u64(coefficient.unsigned_abs())
                } else {
                    F::from_u64(coefficient as u64)
                }
            })
        });
        let mut endpoints = [F::zero(); PRODUCT_UNISKIP_EXTENDED_NODES];
        for (x_out, &outer_weight) in e_out.iter().enumerate() {
            let mut inner = [F::zero(); PRODUCT_UNISKIP_EXTENDED_NODES];
            for (x_in, &inner_weight) in e_in.iter().enumerate() {
                let row = rows[x_out * e_in.len() + x_in];
                for (sum, weights) in inner.iter_mut().zip(&coefficients) {
                    let (left, right) = row.relation_values(weights);
                    *sum += inner_weight * left * right;
                }
            }
            for (endpoint, inner) in endpoints.iter_mut().zip(inner) {
                *endpoint += outer_weight * inner;
            }
        }
        Ok(ProductUniskipExtendedNodes {
            minus_two: endpoints[0],
            plus_two: endpoints[1],
        })
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests use fixed valid shapes")]
mod tests {
    use jolt_field::AkitaField;

    use super::*;

    fn edge_rows() -> Vec<ProductRemainderRow> {
        vec![
            ProductRemainderRow::new(
                u64::MAX,
                i128::MIN,
                true,
                true,
                u64::MAX - 1,
                true,
                false,
                true,
            ),
            ProductRemainderRow::new(0, -1, false, false, 1, false, true, false),
            ProductRemainderRow::new(17, 0, true, false, 23, false, false, true),
            ProductRemainderRow::new(
                u64::MAX - 2,
                i128::MAX,
                false,
                true,
                u64::MAX,
                true,
                true,
                false,
            ),
        ]
    }

    fn direct_node(
        rows: &[ProductRemainderRow],
        weights: [AkitaField; 3],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> AkitaField {
        let mut result = AkitaField::zero();
        for (x_out, &outer_weight) in e_out.iter().enumerate() {
            let mut inner = AkitaField::zero();
            for (x_in, &inner_weight) in e_in.iter().enumerate() {
                let row = rows[x_out * e_in.len() + x_in];
                let (left, right) = row.relation_values(&weights);
                inner += inner_weight * left * right;
            }
            result += outer_weight * inner;
        }
        result
    }

    #[test]
    fn product_row_abi_is_shared_without_translation() {
        assert_eq!(size_of::<ProductRemainderRow>(), 40);
        assert_eq!(align_of::<ProductRemainderRow>(), 8);
    }

    #[test]
    fn extended_nodes_match_direct_edge_evaluations() {
        let rows = edge_rows();
        let e_in = [AkitaField::from_u64(2), AkitaField::from_u64(3)];
        let e_out = [AkitaField::from_u64(5), AkitaField::from_u64(7)];
        let got = reference::extended_node_values(&rows, &e_in, &e_out)
            .expect("the fixed shape is valid");
        let three = AkitaField::from_u64(3);
        let expected = ProductUniskipExtendedNodes {
            minus_two: direct_node(&rows, [three, -three, AkitaField::one()], &e_in, &e_out),
            plus_two: direct_node(&rows, [AkitaField::one(), -three, three], &e_in, &e_out),
        };
        assert_eq!(got, expected);
    }

    #[test]
    fn known_nodes_fill_centered_extended_order() {
        let rows = edge_rows();
        let e_in = [AkitaField::from_u64(2), AkitaField::from_u64(3)];
        let e_out = [AkitaField::from_u64(5), AkitaField::from_u64(7)];
        let extended = reference::extended_node_values(&rows, &e_in, &e_out)
            .expect("the fixed shape is valid");
        let zero = AkitaField::zero();
        let one = AkitaField::one();
        let known = ProductUniskipKnownNodes {
            product: direct_node(&rows, [one, zero, zero], &e_in, &e_out),
            should_branch: direct_node(&rows, [zero, one, zero], &e_in, &e_out),
            should_jump: direct_node(&rows, [zero, zero, one], &e_in, &e_out),
        };
        let assembled = extended.assemble(known);
        let three = AkitaField::from_u64(3);
        let expected = [
            direct_node(&rows, [three, -three, one], &e_in, &e_out),
            known.product,
            known.should_branch,
            known.should_jump,
            direct_node(&rows, [one, -three, three], &e_in, &e_out),
        ];
        assert_eq!(assembled, expected);
    }

    #[test]
    fn block_shape_requires_exact_eq_factorization() {
        assert_eq!(
            ProductUniskipBlockParams::new(0, 0, 0),
            Err(ProductUniskipShapeError::InvalidRows(0))
        );
        assert_eq!(
            ProductUniskipBlockParams::new(6, 2, 3),
            Err(ProductUniskipShapeError::InvalidRows(6))
        );
        assert_eq!(
            ProductUniskipBlockParams::new(8, 2, 2),
            Err(ProductUniskipShapeError::WeightShape {
                phase: "blocks",
                expected: 8,
                e_in: 2,
                e_out: 2,
            })
        );
        assert!(ProductUniskipBlockParams::new(8, 2, 4).is_ok());
    }

    #[test]
    fn reduction_shape_is_fixed_to_two_columns() {
        let params =
            ProductUniskipReductionParams::new(33, 2).expect("two nonempty columns are valid");
        assert_eq!(params.input_count, 33);
        assert_eq!(params.output_count, 2);
        assert_eq!(
            ProductUniskipReductionParams::new(33, 5),
            Err(ProductUniskipShapeError::InvalidReductionColumns(5))
        );
        assert_eq!(
            ProductUniskipReductionParams::new(0, 2),
            Err(ProductUniskipShapeError::EmptyReduction)
        );
    }

    #[test]
    fn target_scale_storage_layout_is_pinned() {
        let rows = 1usize << 26;
        let split = 1usize << 13;
        let layout = ProductUniskipStorageLayout::new(rows, split, split)
            .expect("the target-scale layout is valid");
        let expected_partial_fields = PRODUCT_UNISKIP_EXTENDED_NODES * split;
        let expected_scratch_fields = 2 * split + 2 * expected_partial_fields;

        assert_eq!(layout.rows(), rows);
        assert_eq!(layout.shared_row_bytes(), rows * 40);
        assert_eq!(layout.e_in_fields(), split);
        assert_eq!(layout.e_out_fields(), split);
        assert_eq!(layout.partial_fields(), expected_partial_fields);
        assert_eq!(layout.scratch_bytes(), expected_scratch_fields * 16);
        assert_eq!(
            layout.resident_bytes(),
            rows * 40 + expected_scratch_fields * 16
        );
    }
}
