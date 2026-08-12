#![expect(
    dead_code,
    reason = "implementation target: the instruction read-RAF kernel wires this once it lands"
)]

use cudarc::driver::PushKernelArg;
use jolt_field::Fr;

use super::address_phase::{DeviceRows, CHUNK_LEN, CHUNK_SIZE};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::DeviceFrVec;
use crate::cuda::common::error::CudaError;

pub struct HandoffInputs<'a> {
    pub rows: &'a DeviceRows,
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
    let cycles = inputs.rows.cycles();
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
    pointers: &cudarc::driver::CudaSlice<u64>,
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::instruction::{
        InstructionReadRafDimensions, CANONICAL_INSTRUCTION_ADDRESS,
    };
    use jolt_field::Fr;
    use jolt_lookup_tables::lookup_bits::LookupBits;
    use jolt_lookup_tables::tables::suffixes::SuffixEval;
    use jolt_lookup_tables::tables::LookupTableKind;
    use jolt_lookup_tables::XLEN as RISCV_XLEN;
    use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};
    use proptest::prelude::*;

    use super::super::address_phase::{DeviceRows, CHUNK_LEN};
    use super::{build_cycle_tables, HandoffInputs};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::fr;
    use crate::reference::instruction_read_raf::{
        InstructionReadRafKernel, InstructionReadRafWitness,
    };
    use crate::reference::views::eq_table;

    const ADDRESS_BITS: usize = 128;
    const RA_COUNT: usize = 8;

    fn rows(log_t: usize, seed: u64) -> Vec<InstructionReadRafWitness> {
        let tables: Vec<LookupTableKind<RISCV_XLEN>> =
            LookupTableKind::<RISCV_XLEN>::iter().collect();
        (0..1usize << log_t)
            .map(|j| {
                let mixed = (j as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(seed);
                let index = (u128::from(mixed) << 61) | u128::from(mixed.rotate_left(17));
                InstructionReadRafWitness {
                    lookup_index: LookupIndex(index),
                    table_index: TableIndex(if mixed.is_multiple_of(11) {
                        None
                    } else {
                        Some(tables[(mixed % tables.len() as u64) as usize].index())
                    }),
                    raf_flag: InstructionRafFlag(mixed.is_multiple_of(3)),
                }
            })
            .collect()
    }

    fn reference_at_handoff(
        log_t: usize,
        seed: u64,
    ) -> (InstructionReadRafKernel<Fr>, Vec<Vec<Fr>>) {
        let dimensions = InstructionReadRafDimensions::try_from((log_t, ADDRESS_BITS, RA_COUNT))
            .expect("dimensions");
        let r_reduction: Vec<Fr> = (0..log_t).map(|i| fr(seed + i as u64 + 3)).collect();
        let mut kernel = InstructionReadRafKernel::new(
            dimensions,
            &r_reduction,
            rows(log_t, seed),
            fr(seed + 1),
        )
        .expect("reference kernel");
        for round in 0..ADDRESS_BITS {
            kernel
                .bind(fr(seed + round as u64 + 71))
                .expect("reference bind");
        }
        let v_tables = (0..ADDRESS_BITS / CHUNK_LEN)
            .map(|phase| {
                let challenges: Vec<Fr> = (0..CHUNK_LEN)
                    .map(|i| fr(seed + (phase * CHUNK_LEN + i) as u64 + 71))
                    .collect();
                eq_table(&challenges)
            })
            .collect();
        (kernel, v_tables)
    }

    proptest! {
        #[test]
        fn cycle_tables_match_the_reference_handoff(
            log_t in 4usize..=9,
            seed in any::<u64>(),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let (host, host_v_tables) = reference_at_handoff(log_t, seed);

            let gamma = host.gamma;
            let gamma_sqr = gamma * gamma;
            let empty = LookupBits::new(0, 0);
            let table_values: Vec<Fr> = LookupTableKind::<RISCV_XLEN>::iter()
                .map(|table| {
                    let suffixes: Vec<SuffixEval<Fr>> = table
                        .suffixes()
                        .iter()
                        .map(|suffix| SuffixEval::from(Fr::from(suffix.suffix_mle(empty))))
                        .collect();
                    table.combine(host.prefix_checkpoints(), &suffixes)
                })
                .collect();
            let raf_interleaved = gamma * host.raf_left.checkpoint
                + gamma_sqr * host.raf_right.checkpoint;
            let mut raf_identity = gamma_sqr * host.raf_identity.checkpoint;
            if CANONICAL_INSTRUCTION_ADDRESS {
                raf_identity += gamma_sqr * gamma * host.raf_upper_all_ones.checkpoint;
            }

            let jolt_rows = rows(log_t, seed);
            let indices: Vec<u128> = jolt_rows.iter().map(|r| r.lookup_index.0).collect();
            let tables: Vec<Option<usize>> = jolt_rows.iter().map(|r| r.table_index.0).collect();
            let flags: Vec<bool> = jolt_rows.iter().map(|r| r.raf_flag.0).collect();
            let device_rows = DeviceRows::new(context, &indices, &tables, &flags)
                .expect("device rows");
            let v_tables: Vec<_> = host_v_tables
                .iter()
                .map(|table| context.upload(table).expect("upload v table"))
                .collect();

            let got = build_cycle_tables(
                context,
                &HandoffInputs {
                    rows: &device_rows,
                    v_tables: &v_tables,
                    table_values: &table_values,
                    raf_interleaved,
                    raf_identity,
                    ra_count: RA_COUNT,
                    address_bits: ADDRESS_BITS,
                },
            )
            .expect("device build_cycle_tables");

            let expected = host.cycle_tables.as_ref().expect("cycle tables");
            prop_assert_eq!(
                &got.combined_val.to_host().expect("download"),
                &expected.combined_val.evals().to_vec(),
                "combined_val diverged"
            );
            prop_assert_eq!(got.ra.len(), expected.ra.len());
            for (index, (device, want)) in got.ra.iter().zip(&expected.ra).enumerate() {
                prop_assert_eq!(
                    &device.to_host().expect("download"),
                    &want.evals().to_vec(),
                    "ra[{}] diverged",
                    index
                );
            }
        }
    }
}
