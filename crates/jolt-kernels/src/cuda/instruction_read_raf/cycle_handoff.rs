use cudarc::driver::CudaSlice;
use cudarc::driver::PushKernelArg;
use jolt_field::Fr;

use super::address_phase::{DeviceRows, CHUNK_LEN, CHUNK_SIZE};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::DeviceFrVec;
use crate::cuda::common::error::CudaError;

pub struct HandoffInputs<'a> {
    pub rows: &'a DeviceRows,
    pub cycles: usize,
    pub v_tables: &'a [DeviceFrVec],
    pub table_values: &'a [Fr],
    pub raf_interleaved: Fr,
    pub raf_identity: Fr,
    pub ra_count: usize,
    pub address_bits: usize,
}

pub struct HandoffTables {
    pub combined_val: DeviceFrVec,
    pub ra: Vec<DeviceFrVec>,
}

pub fn build_cycle_tables(
    context: &CudaKernelContext,
    inputs: &HandoffInputs<'_>,
) -> Result<HandoffTables, CudaError> {
    let cycles = inputs.cycles;
    if cycles > inputs.rows.cycles() {
        return Err(CudaError::LengthMismatch {
            expected: cycles,
            got: inputs.rows.cycles(),
        });
    }
    let phases = inputs.address_bits / CHUNK_LEN;
    if inputs.v_tables.len() != phases {
        return Err(CudaError::LengthMismatch {
            expected: phases,
            got: inputs.v_tables.len(),
        });
    }
    for table in inputs.v_tables {
        if table.len() != CHUNK_SIZE {
            return Err(CudaError::LengthMismatch {
                expected: CHUNK_SIZE,
                got: table.len(),
            });
        }
    }
    if inputs.ra_count == 0 || !phases.is_multiple_of(inputs.ra_count) {
        return Err(CudaError::InvariantViolation {
            reason: "the virtual RA count must divide the address phase count",
        });
    }

    let combined_val = build_combined_val(context, inputs, cycles)?;
    let mut ra = Vec::with_capacity(inputs.ra_count);
    let phases_per_ra = phases / inputs.ra_count;
    let pointers = context.device_pointers(&inputs.v_tables.iter().collect::<Vec<_>>())?;
    for index in 0..inputs.ra_count {
        ra.push(build_ra(
            context,
            inputs,
            &pointers,
            index * phases_per_ra,
            phases_per_ra,
            phases,
            cycles,
        )?);
    }
    Ok(HandoffTables { combined_val, ra })
}

fn build_combined_val(
    context: &CudaKernelContext,
    inputs: &HandoffInputs<'_>,
    cycles: usize,
) -> Result<DeviceFrVec, CudaError> {
    let table_values = context.upload(inputs.table_values)?;
    let raf_interleaved = context.upload(&[inputs.raf_interleaved])?;
    let raf_identity = context.upload(&[inputs.raf_identity])?;
    let table_count = CudaKernelContext::count_of(inputs.table_values.len())?;
    let count = CudaKernelContext::count_of(cycles)?;

    let mut out = context.alloc(cycles)?;
    let mut builder = context.stream().launch_builder(context.ap_combined_val());
    let _ = builder.arg(inputs.rows.table_index());
    let _ = builder.arg(inputs.rows.raf_flag());
    let _ = builder.arg(table_values.limbs());
    let _ = builder.arg(raf_interleaved.limbs());
    let _ = builder.arg(raf_identity.limbs());
    let _ = builder.arg(&table_count);
    let _ = builder.arg(out.limbs_mut());
    let _ = builder.arg(&count);
    // SAFETY: thread `j < cycles` reads `table_index[j]` and `raf_flag[j]` of
    // `cycles`, `table_values[table]` only after bounds-checking `table` against
    // `table_count` (the uploaded length), and one of the two single-element RAF
    // buffers. Writes only `out[j]` of `cycles`, a fresh allocation.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    Ok(out)
}

fn build_ra(
    context: &CudaKernelContext,
    inputs: &HandoffInputs<'_>,
    pointers: &CudaSlice<u64>,
    phase_offset: usize,
    phases_per_ra: usize,
    phases: usize,
    cycles: usize,
) -> Result<DeviceFrVec, CudaError> {
    let mut out = context.alloc(cycles)?;
    let count = CudaKernelContext::count_of(cycles)?;
    let phase_offset = CudaKernelContext::count_of(phase_offset)?;
    let phases_per_ra_arg = CudaKernelContext::count_of(phases_per_ra)?;
    let phases_arg = CudaKernelContext::count_of(phases)?;

    let mut builder = context.stream().launch_builder(context.ap_ra());
    let _ = builder.arg(inputs.rows.lookup_index());
    let _ = builder.arg(pointers);
    let _ = builder.arg(&phase_offset);
    let _ = builder.arg(&phases_per_ra_arg);
    let _ = builder.arg(&phases_arg);
    let _ = builder.arg(out.limbs_mut());
    let _ = builder.arg(&count);
    // SAFETY: thread `j < cycles` reads `lookup_index[2j]`/`[2j+1]` of
    // `2 * cycles` and, for `q < phases_per_ra`, `v_tables[phase_offset + q]`
    // indexed at a chunk masked to `CHUNK_SIZE - 1`. Callers pass
    // `phase_offset + phases_per_ra <= phases` and every `v` table is
    // length-checked as `CHUNK_SIZE` above, so both the pointer array and each
    // table read are in range. Writes only `out[j]` of `cycles`.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    Ok(out)
}
