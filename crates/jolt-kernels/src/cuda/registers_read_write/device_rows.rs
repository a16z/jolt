use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::{Fr, FromPrimitiveInt};

use jolt_witness::backend::cuda::{
    DeviceAtomColumns, DeviceTrace, EXTRA_RD_POST, EXTRA_RS1, EXTRA_RS2,
};

#[cfg(test)]
use super::witness::RegistersReadWriteWitness;
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::DeviceFrVec;
use crate::cuda::common::error::CudaError;
#[cfg(test)]
use crate::cuda::common::ra_poly::COLD;
use crate::cuda::common::read_write_matrix::{CoeffTables, DeviceCoeffs, DeviceReadWriteMatrix};

const COEFF_WIDTH: usize = 2;

pub(crate) fn register_coeff_tables(gamma: Fr) -> CoeffTables {
    let gamma_sq = gamma * gamma;
    CoeffTables {
        values: [
            vec![Fr::from_u64(0), gamma, gamma_sq, gamma + gamma_sq],
            vec![Fr::from_u64(0), Fr::from_u64(1)],
        ],
    }
}

fn witness_error(_error: jolt_witness::WitnessError) -> CudaError {
    CudaError::InvariantViolation {
        reason: "the device residency could not serve a packed register column",
    }
}

pub struct DeviceRegisterRows {
    rs1_address: CudaSlice<u32>,
    rs1_value: CudaSlice<u64>,
    rs2_address: CudaSlice<u32>,
    rs2_value: CudaSlice<u64>,
    rd_address: CudaSlice<u32>,
    rd_pre_value: CudaSlice<u64>,
    rd_post_value: CudaSlice<u64>,
    cycles: usize,
}

impl DeviceRegisterRows {
    pub fn from_device(
        context: &CudaKernelContext,
        trace: &DeviceTrace,
        atoms: &DeviceAtomColumns,
        cycles: usize,
    ) -> Result<Self, CudaError> {
        if cycles == 0 || trace.cycles() < cycles {
            return Err(CudaError::InvariantViolation {
                reason: "the device register sources do not cover the requested cycles",
            });
        }
        Ok(Self {
            rs1_address: context.clone_u32(&atoms.rs1_address)?,
            rs1_value: trace.extra_word_column(EXTRA_RS1).map_err(witness_error)?,
            rs2_address: context.clone_u32(&atoms.rs2_address)?,
            rs2_value: trace.extra_word_column(EXTRA_RS2).map_err(witness_error)?,
            rd_address: context.clone_u32(&atoms.rd_address)?,
            rd_pre_value: context.clone_u64(&atoms.rd_pre_value)?,
            rd_post_value: trace
                .extra_word_column(EXTRA_RD_POST)
                .map_err(witness_error)?,
            cycles,
        })
    }

    #[cfg(test)]
    pub fn upload(
        context: &CudaKernelContext,
        rows: &[RegistersReadWriteWitness],
    ) -> Result<Self, CudaError> {
        let encode = |register: Option<u8>| register.map_or(COLD, u32::from);
        let mut rs1_address = Vec::with_capacity(rows.len());
        let mut rs1_value = Vec::with_capacity(rows.len());
        let mut rs2_address = Vec::with_capacity(rows.len());
        let mut rs2_value = Vec::with_capacity(rows.len());
        let mut rd_address = Vec::with_capacity(rows.len());
        let mut rd_pre_value = Vec::with_capacity(rows.len());
        let mut rd_post_value = Vec::with_capacity(rows.len());
        for row in rows {
            rs1_address.push(encode(row.rs1_address.0));
            rs1_value.push(row.rs1_value.0);
            rs2_address.push(encode(row.rs2_address.0));
            rs2_value.push(row.rs2_value.0);
            rd_address.push(encode(row.rd_address.0));
            rd_pre_value.push(row.rd_pre_value.0);
            rd_post_value.push(row.rd_post_value.0);
        }
        Ok(Self {
            rs1_address: context.upload_u32_slice(&rs1_address)?,
            rs1_value: context.upload_u64_slice(&rs1_value)?,
            rs2_address: context.upload_u32_slice(&rs2_address)?,
            rs2_value: context.upload_u64_slice(&rs2_value)?,
            rd_address: context.upload_u32_slice(&rd_address)?,
            rd_pre_value: context.upload_u64_slice(&rd_pre_value)?,
            rd_post_value: context.upload_u64_slice(&rd_post_value)?,
            cycles: rows.len(),
        })
    }

    #[cfg(test)]
    pub fn rs1_address(&self) -> &CudaSlice<u32> {
        &self.rs1_address
    }

    #[cfg(test)]
    pub fn rs2_address(&self) -> &CudaSlice<u32> {
        &self.rs2_address
    }

    #[cfg(test)]
    pub fn rd_address(&self) -> &CudaSlice<u32> {
        &self.rd_address
    }

    #[cfg(test)]
    pub fn rs1_value(&self) -> &CudaSlice<u64> {
        &self.rs1_value
    }

    #[cfg(test)]
    pub fn rs2_value(&self) -> &CudaSlice<u64> {
        &self.rs2_value
    }

    #[cfg(test)]
    pub fn rd_pre_value(&self) -> &CudaSlice<u64> {
        &self.rd_pre_value
    }

    #[cfg(test)]
    pub fn rd_post_value(&self) -> &CudaSlice<u64> {
        &self.rd_post_value
    }

    pub fn into_rs2_address(self) -> CudaSlice<u32> {
        self.rs2_address
    }

    pub fn matrix(
        &self,
        context: &CudaKernelContext,
        gamma: Fr,
    ) -> Result<DeviceReadWriteMatrix, CudaError> {
        let cycles = CudaKernelContext::count_of(self.cycles)?;
        let mut counts = context.alloc_u32(self.cycles)?;
        let mut builder = context.stream().launch_builder(context.reg_count());
        let _ = builder.arg(&self.rs1_address);
        let _ = builder.arg(&self.rs2_address);
        let _ = builder.arg(&self.rd_address);
        let _ = builder.arg(&cycles);
        let _ = builder.arg(&mut counts);
        // SAFETY: thread `j < cycles` reads the three address arrays at `j` (each
        // `cycles` u32s) and writes only `counts[j]`. The slot arrays it fills are
        // thread-local `REG_SLOTS`-element stacks and `reg_build_slots` never
        // returns more than `REG_SLOTS`, since it appends at most once per operand.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(cycles)) }?;
        context.stream().synchronize()?;

        let (offsets, entries) = context.exclusive_scan_with_total_u32(&counts, self.cycles)?;

        let mut out_rows = context.alloc_u32(entries)?;
        let mut out_cols = context.alloc_u32(entries)?;
        let mut out_val = context.alloc(entries)?;
        let mut out_prev = context.alloc_u64(entries)?;
        let mut out_next = context.alloc_u64(entries)?;
        let mut out_coeff_index = context.alloc_u16_unset(entries * COEFF_WIDTH)?;

        let mut builder = context.stream().launch_builder(context.reg_scatter());
        let _ = builder.arg(&self.rs1_address);
        let _ = builder.arg(&self.rs1_value);
        let _ = builder.arg(&self.rs2_address);
        let _ = builder.arg(&self.rs2_value);
        let _ = builder.arg(&self.rd_address);
        let _ = builder.arg(&self.rd_pre_value);
        let _ = builder.arg(&self.rd_post_value);
        let _ = builder.arg(&offsets);
        let _ = builder.arg(&cycles);
        let _ = builder.arg(&mut out_rows);
        let _ = builder.arg(&mut out_cols);
        let _ = builder.arg(out_val.limbs_mut());
        let _ = builder.arg(&mut out_prev);
        let _ = builder.arg(&mut out_next);
        let _ = builder.arg(&mut out_coeff_index);
        // SAFETY: thread `j < cycles` reads the seven per-cycle arrays at `j` and
        // `offsets[j]`. It writes
        // `counts[j] = reg_build_slots(..)` slots starting at `offsets[j]` in each
        // `out_*` buffer — the SAME count the scan was built from, because both
        // kernels call `reg_build_slots`. `offsets` is that scan and `entries` its
        // total, so the per-cycle ranges are disjoint and inside every buffer.
        // `out_coeff_index` is indexed `(k + i) * 2 + lane` over `entries * 2`
        // elements, and each value is a `ra_kind`/`wa_flag` that `reg_build_slots`
        // confines to the length of the lane's table.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(cycles)) }?;
        context.stream().synchronize()?;

        DeviceReadWriteMatrix::from_device_parts(
            context,
            out_rows,
            out_cols,
            out_val,
            out_prev,
            out_next,
            DeviceCoeffs::Indexed {
                index: out_coeff_index,
                luts: DeviceReadWriteMatrix::upload_luts(context, &register_coeff_tables(gamma))?,
            },
            context.upload(&[Fr::from_u64(0)])?,
            COEFF_WIDTH,
            entries,
        )
    }

    pub fn inc(&self, context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
        let cycles = CudaKernelContext::count_of(self.cycles)?;
        let mut out = context.alloc(self.cycles)?;
        let mut builder = context.stream().launch_builder(context.fr_delta_u64());
        let _ = builder.arg(&self.rd_pre_value);
        let _ = builder.arg(&self.rd_post_value);
        let _ = builder.arg(&cycles);
        let _ = builder.arg(out.limbs_mut());
        // SAFETY: thread `j < cycles` reads `rd_pre_value[j]` and
        // `rd_post_value[j]` (each `cycles` u64s) and writes the `LIMBS` field
        // limbs at `out[j]` of `cycles` field elements. Distinct allocations.
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
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_program::execution::{RegisterRead, RegisterState, RegisterWrite, TraceRow};
    use jolt_witness::witnesses::{
        Extract, RdAddress, RdInc, RdWriteValue, Rs1Value, Rs2Value, ToField, WitnessEnv,
    };
    use jolt_witness::{FixedBackend, ProgramSource};

    use super::DeviceRegisterRows;
    use crate::cuda::common::context::{shared_context, CudaKernelContext};
    use crate::cuda::common::read_write_matrix::{DeviceReadWriteMatrix, MatrixEntry};
    use crate::cuda::common::testing::RowPlane;
    use crate::cuda::registers_read_write::witness::{
        matrix_entries, RdPreValue, RegistersReadWriteWitness, Rs1Address, Rs2Address,
    };

    const LOG_T: usize = 8;
    const COEFF_WIDTH: usize = 2;

    type Activity = (Option<u8>, Option<u8>, Option<u8>);

    const ACTIVITY: [Activity; 17] = [
        (None, None, None),
        (Some(3), None, None),
        (None, Some(5), None),
        (None, None, Some(7)),
        (Some(9), Some(9), None),
        (Some(11), None, Some(11)),
        (None, Some(13), Some(13)),
        (Some(2), Some(4), Some(6)),
        (Some(6), Some(6), Some(6)),
        (Some(1), Some(2), Some(1)),
        (Some(1), Some(2), Some(2)),
        (Some(120), Some(121), Some(122)),
        (Some(6), Some(4), Some(2)),
        (Some(9), Some(3), None),
        (Some(4), Some(8), Some(1)),
        (Some(30), Some(30), Some(12)),
        (Some(40), Some(20), Some(40)),
    ];

    fn trace_rows(seed: u64) -> Vec<TraceRow> {
        let mut state = vec![0u64; 128];
        (0..1usize << LOG_T)
            .map(|cycle| {
                let (rs1, rs2, rd) = ACTIVITY[(cycle + seed as usize) % ACTIVITY.len()];
                let mut registers = RegisterState::default();
                if let Some(register) = rs1 {
                    registers.rs1 = Some(RegisterRead {
                        register,
                        value: state[usize::from(register)],
                    });
                }
                if let Some(register) = rs2 {
                    registers.rs2 = Some(RegisterRead {
                        register,
                        value: state[usize::from(register)],
                    });
                }
                if let Some(register) = rd {
                    let pre_value = state[usize::from(register)];
                    let post_value = if cycle % 3 == 0 {
                        pre_value
                    } else {
                        pre_value.wrapping_sub(1 + cycle as u64)
                    };
                    registers.rd = Some(RegisterWrite {
                        register,
                        pre_value,
                        post_value,
                    });
                    state[usize::from(register)] = post_value;
                }
                TraceRow {
                    registers,
                    ..TraceRow::default()
                }
            })
            .collect()
    }

    fn bundles(rows: &[TraceRow]) -> Vec<RegistersReadWriteWitness> {
        rows.iter()
            .map(|row| RegistersReadWriteWitness {
                rs1_address: Rs1Address(row.registers.rs1.map(|read| read.register)),
                rs1_value: Rs1Value(row.registers.rs1.map_or(0, |read| read.value)),
                rs2_address: Rs2Address(row.registers.rs2.map(|read| read.register)),
                rs2_value: Rs2Value(row.registers.rs2.map_or(0, |read| read.value)),
                rd_address: RdAddress(row.registers.rd.map(|write| write.register)),
                rd_pre_value: RdPreValue(row.registers.rd.map_or(0, |write| write.pre_value)),
                rd_post_value: RdWriteValue(row.registers.rd.map_or(0, |write| write.post_value)),
            })
            .collect()
    }

    fn host_entries(
        context: &CudaKernelContext,
        bundles: &[RegistersReadWriteWitness],
        gamma: Fr,
    ) -> Vec<MatrixEntry> {
        DeviceReadWriteMatrix::new(context, &matrix_entries(bundles, gamma), COEFF_WIDTH, None)
            .expect("host-built device matrix")
            .to_host(context)
            .expect("download host-built")
    }

    #[test]
    fn device_matrix_matches_host_entries() {
        let Some(context) = shared_context() else {
            return;
        };
        for seed in 0..4u64 {
            let bundles = bundles(&trace_rows(seed));
            let gamma = Fr::from_u64(101 + seed);

            let expected = host_entries(context, &bundles, gamma);
            let got = DeviceRegisterRows::upload(context, &bundles)
                .expect("upload rows")
                .matrix(context, gamma)
                .expect("device matrix")
                .to_host(context)
                .expect("download device");

            assert_eq!(
                got, expected,
                "seed {seed}: device-built entries disagree with the host construction",
            );
        }
    }

    #[test]
    fn device_inc_matches_witness_atom() {
        let Some(context) = shared_context() else {
            return;
        };
        let rows = trace_rows(1);
        let bundles = bundles(&rows);

        let plane = RowPlane::new(FixedBackend::new(), "rd inc oracle", LOG_T, Vec::new());
        let env = WitnessEnv::new(ProgramSource::program_preprocessing(&plane));
        let expected: Vec<Fr> = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                RdInc::extract(row, rows.get(index + 1), &env)
                    .expect("rd increment")
                    .to_field()
            })
            .collect();

        let got = DeviceRegisterRows::upload(context, &bundles)
            .expect("upload rows")
            .inc(context)
            .expect("device inc")
            .to_host()
            .expect("download inc");

        assert_eq!(
            got, expected,
            "device increments disagree with RdInc::extract"
        );
    }
}
