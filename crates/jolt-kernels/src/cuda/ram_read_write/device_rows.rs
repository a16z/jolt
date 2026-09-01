use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::Fr;

use jolt_witness::backend::cuda::{DeviceTrace, EXTRA_RAM_READ, EXTRA_RAM_WRITE};

#[cfg(test)]
use super::witness::RamReadWriteWitness;
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::DeviceFrVec;
use crate::cuda::common::error::CudaError;
#[cfg(test)]
use crate::cuda::common::ra_poly::COLD;
use crate::cuda::common::read_write_matrix::{DeviceCoeffs, DeviceReadWriteMatrix};

pub struct DeviceRamRows {
    address: CudaSlice<u32>,
    read_value: CudaSlice<u64>,
    write_value: CudaSlice<u64>,
    cycles: usize,
}

impl DeviceRamRows {
    pub fn from_device(
        trace: &DeviceTrace,
        addresses: usize,
        cycles: usize,
    ) -> Result<Self, CudaError> {
        if cycles == 0 || trace.cycles() < cycles {
            return Err(CudaError::InvariantViolation {
                reason: "the device RAM sources do not cover the requested cycles",
            });
        }
        let (address, _) =
            trace
                .remapped_ram_words(addresses)
                .map_err(|_| CudaError::InvariantViolation {
                    reason: "the device residency could not remap the RAM addresses",
                })?;
        let read_value =
            trace
                .extra_word_column(EXTRA_RAM_READ)
                .map_err(|_| CudaError::InvariantViolation {
                    reason: "the device residency could not serve the RAM read values",
                })?;
        let write_value = trace.extra_word_column(EXTRA_RAM_WRITE).map_err(|_| {
            CudaError::InvariantViolation {
                reason: "the device residency could not serve the RAM write values",
            }
        })?;
        Ok(Self {
            address,
            read_value,
            write_value,
            cycles,
        })
    }

    #[cfg(test)]
    pub fn upload(
        context: &CudaKernelContext,
        rows: &[RamReadWriteWitness],
    ) -> Result<Self, CudaError> {
        let mut address = Vec::with_capacity(rows.len());
        let mut read_value = Vec::with_capacity(rows.len());
        let mut write_value = Vec::with_capacity(rows.len());
        for row in rows {
            let encoded = match row.address.0 {
                Some(word) => u32::try_from(word).map_err(|_| CudaError::InvariantViolation {
                    reason: "a remapped RAM word address exceeded the u32 column domain",
                })?,
                None => COLD,
            };
            if encoded == COLD && row.address.0.is_some() {
                return Err(CudaError::InvariantViolation {
                    reason: "a remapped RAM word address collided with the absent-address sentinel",
                });
            }
            address.push(encoded);
            read_value.push(row.read_value.0);
            write_value.push(row.write_value.0);
        }
        Ok(Self {
            address: context.upload_u32_slice(&address)?,
            read_value: context.upload_u64_slice(&read_value)?,
            write_value: context.upload_u64_slice(&write_value)?,
            cycles: rows.len(),
        })
    }

    #[cfg(test)]
    pub fn address(&self) -> &CudaSlice<u32> {
        &self.address
    }

    #[cfg(test)]
    pub fn read_value(&self) -> &CudaSlice<u64> {
        &self.read_value
    }

    #[cfg(test)]
    pub fn write_value(&self) -> &CudaSlice<u64> {
        &self.write_value
    }

    pub fn matrix(
        &self,
        context: &CudaKernelContext,
        wa_scale: Fr,
    ) -> Result<DeviceReadWriteMatrix, CudaError> {
        let cycles = CudaKernelContext::count_of(self.cycles)?;
        let mut flags = context.alloc_u32(self.cycles)?;
        let mut builder = context.stream().launch_builder(context.rrw_flags());
        let _ = builder.arg(&self.address);
        let _ = builder.arg(&cycles);
        let _ = builder.arg(&mut flags);
        // SAFETY: thread `j < cycles` reads `address[j]` and writes only
        // `flags[j]`; both buffers hold `cycles` u32s and are distinct.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(cycles)) }?;
        context.stream().synchronize()?;

        let (offsets, entries) = context.exclusive_scan_with_total_u32(&flags, self.cycles)?;

        let mut out_rows = context.alloc_u32(entries)?;
        let mut out_cols = context.alloc_u32(entries)?;
        let mut out_val = context.alloc(entries)?;
        let mut out_prev = context.alloc_u64(entries)?;
        let mut out_next = context.alloc_u64(entries)?;
        let mut out_coeffs = context.alloc(entries)?;

        let mut builder = context.stream().launch_builder(context.rrw_scatter());
        let _ = builder.arg(&self.address);
        let _ = builder.arg(&self.read_value);
        let _ = builder.arg(&self.write_value);
        let _ = builder.arg(&offsets);
        let _ = builder.arg(&cycles);
        let _ = builder.arg(&mut out_rows);
        let _ = builder.arg(&mut out_cols);
        let _ = builder.arg(out_val.limbs_mut());
        let _ = builder.arg(&mut out_prev);
        let _ = builder.arg(&mut out_next);
        let _ = builder.arg(out_coeffs.limbs_mut());
        // SAFETY: thread `j < cycles` returns unless `address[j]` is a present
        // column, and otherwise writes each `out_*` buffer exactly once at
        // `offsets[j]`. `offsets` is the exclusive scan of the present-address
        // flags, so those indices are distinct across threads and all `< entries`
        // (the scan's total). Reads are `address`/`read_value`/`write_value`/
        // `offsets` at `j`, all `cycles` long. Inputs and outputs are distinct.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(cycles)) }?;
        context.stream().synchronize()?;

        DeviceReadWriteMatrix::from_device_parts(
            context,
            out_rows,
            out_cols,
            out_val,
            out_prev,
            out_next,
            DeviceCoeffs::Direct(out_coeffs),
            context.upload(&[wa_scale])?,
            1,
            entries,
        )
    }

    pub fn inc(&self, context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
        let cycles = CudaKernelContext::count_of(self.cycles)?;
        let mut out = context.alloc(self.cycles)?;
        let mut builder = context.stream().launch_builder(context.fr_delta_u64());
        let _ = builder.arg(&self.read_value);
        let _ = builder.arg(&self.write_value);
        let _ = builder.arg(&cycles);
        let _ = builder.arg(out.limbs_mut());
        // SAFETY: thread `j < cycles` reads `read_value[j]` and `write_value[j]`
        // (each `cycles` u64s) and writes the `LIMBS` field limbs at `out[j]` of
        // `cycles` field elements. Inputs and output are distinct allocations.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(cycles)) }?;
        context.stream().synchronize()?;
        Ok(out)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use common::jolt_device::MemoryLayout;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_program::execution::{RamAccess, RamRead, RamWrite, TraceRow};
    use jolt_witness::witnesses::{
        Extract, RamInc, RamReadValue, RamWriteValue, RemappedRamAddress, ToField, WitnessEnv,
    };
    use jolt_witness::{FixedBackend, ProgramSource};

    use super::DeviceRamRows;
    use crate::cuda::common::context::{shared_context, CudaKernelContext};
    use crate::cuda::common::read_write_matrix::{DeviceReadWriteMatrix, MatrixEntry};
    use crate::cuda::common::testing::RowPlane;
    use crate::cuda::ram_read_write::witness::{matrix_entries, RamReadWriteWitness};

    const LOG_T: usize = 8;
    const RAM_K: usize = 64;

    fn trace_rows() -> Vec<TraceRow> {
        (0..1usize << LOG_T)
            .map(|cycle| {
                let word = 1 + (cycle as u64 * 7) % (RAM_K as u64 - 1);
                let address = 8 * word;
                let access = match cycle % 5 {
                    0 => RamAccess::NoOp,
                    1 => RamAccess::Read(RamRead {
                        address,
                        value: 900 + cycle as u64,
                    }),
                    2 => RamAccess::Write(RamWrite {
                        address,
                        pre_value: 100 + cycle as u64,
                        post_value: 700 + cycle as u64,
                    }),
                    3 => RamAccess::Write(RamWrite {
                        address,
                        pre_value: 400 + cycle as u64,
                        post_value: 400 + cycle as u64,
                    }),
                    _ => RamAccess::Write(RamWrite {
                        address,
                        pre_value: 5_000 + cycle as u64,
                        post_value: 12,
                    }),
                };
                TraceRow {
                    ram_access: access,
                    ..TraceRow::default()
                }
            })
            .collect()
    }

    fn bundles(rows: &[TraceRow], layout: &MemoryLayout) -> Vec<RamReadWriteWitness> {
        rows.iter()
            .map(|row| {
                let address = match row.ram_access {
                    RamAccess::Read(read) => Some(read.address),
                    RamAccess::Write(write) => Some(write.address),
                    RamAccess::NoOp => None,
                };
                let (read_value, write_value) = match row.ram_access {
                    RamAccess::Read(read) => (read.value, read.value),
                    RamAccess::Write(write) => (write.pre_value, write.post_value),
                    RamAccess::NoOp => (0, 0),
                };
                RamReadWriteWitness {
                    address: RemappedRamAddress(
                        address.and_then(|a| layout.remap_word_address(a).ok().flatten()),
                    ),
                    read_value: RamReadValue(read_value),
                    write_value: RamWriteValue(write_value),
                }
            })
            .collect()
    }

    #[test]
    fn device_matrix_matches_host_entries() {
        let Some(context) = shared_context() else {
            return;
        };
        let layout = MemoryLayout::default();
        let rows = trace_rows();
        let bundles = bundles(&rows, &layout);
        let gamma = Fr::from_u64(103);

        let expected = host_entries(context, &bundles, gamma);
        let device = DeviceRamRows::upload(context, &bundles)
            .expect("upload rows")
            .matrix(context, gamma)
            .expect("device matrix");

        assert_eq!(
            device.to_host(context).expect("download device"),
            expected,
            "device-built entries disagree with the host construction",
        );
    }

    fn host_entries(
        context: &CudaKernelContext,
        bundles: &[RamReadWriteWitness],
        gamma: Fr,
    ) -> Vec<MatrixEntry> {
        DeviceReadWriteMatrix::new(context, &matrix_entries(bundles), 1, Some(gamma))
            .expect("host-built device matrix")
            .to_host(context)
            .expect("download host-built")
    }

    #[test]
    fn device_inc_matches_witness_atom() {
        let Some(context) = shared_context() else {
            return;
        };
        let layout = MemoryLayout::default();
        let rows = trace_rows();
        let bundles = bundles(&rows, &layout);

        let plane = RowPlane::new(FixedBackend::new(), "ram inc oracle", LOG_T, Vec::new());
        let env = WitnessEnv::new(ProgramSource::program_preprocessing(&plane));
        let expected: Vec<Fr> = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                RamInc::extract(row, rows.get(index + 1), &env)
                    .expect("ram increment")
                    .to_field()
            })
            .collect();

        let got = DeviceRamRows::upload(context, &bundles)
            .expect("upload rows")
            .inc(context)
            .expect("device inc")
            .to_host()
            .expect("download inc");

        assert_eq!(
            got, expected,
            "device increments disagree with RamInc::extract"
        );
    }
}
