use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_riscv::{CircuitFlags, InstructionFlags as InstructionFlagKind};
use jolt_witness::backend::cuda::{
    circuit_flag_bit, instruction_flag_bit, DeviceAtomColumns, FLAG_BIT_NEXT_IS_NOOP,
};

#[cfg(test)]
use super::witness::Packed;
use super::witness::{self, NARROW, WIDE};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::error::CudaError;

pub struct DeviceProductColumns {
    narrow: CudaSlice<u64>,
    wide: CudaSlice<u64>,
    flags: CudaSlice<u32>,
    layout: CudaSlice<u32>,
    cycles: usize,
}

impl DeviceProductColumns {
    pub fn from_device(
        context: &CudaKernelContext,
        atoms: &DeviceAtomColumns,
        cycles: usize,
    ) -> Result<Self, CudaError> {
        if cycles == 0 {
            return Err(CudaError::InvariantViolation {
                reason: "the device product columns need at least one cycle",
            });
        }
        let mut narrow = context.alloc_u64(cycles * NARROW)?;
        let mut wide = context.alloc_u64(cycles * WIDE * 2)?;
        let mut flags = context.alloc_u32(cycles)?;
        let sources = context.upload_u32_slice(&Self::gather_bit_sources()?)?;
        let sign_base = witness::SIGN_BIT_BASE;

        let count = CudaKernelContext::count_of(cycles)?;
        let mut builder = context.stream().launch_builder(context.sp_gather());
        let _ = builder.arg(&atoms.flags);
        let _ = builder.arg(&sources);
        let _ = builder.arg(&sign_base);
        let _ = builder.arg(&atoms.left_instruction_input);
        let _ = builder.arg(&atoms.right_instruction_input);
        let _ = builder.arg(&atoms.lookup_output);
        let _ = builder.arg(&mut narrow);
        let _ = builder.arg(&mut wide);
        let _ = builder.arg(&mut flags);
        let _ = builder.arg(&count);
        // SAFETY: thread `t < cycles` writes the `NARROW` u64s at `t * NARROW`,
        // the `WIDE * 2` u64s at `t * WIDE * 2` and `flags[t]`, all inside
        // allocations sized for `cycles` rows, and reads index `t` of each atom
        // column plus the five in-range entries of `sources`. Every buffer is a
        // distinct allocation.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

        Ok(Self {
            narrow,
            wide,
            flags,
            layout: context.upload_u32_slice(&witness::CLAIM_LAYOUT)?,
            cycles,
        })
    }

    fn gather_bit_sources() -> Result<Vec<u32>, CudaError> {
        let missing = || CudaError::InvariantViolation {
            reason: "a Spartan product flag has no canonical device bit",
        };
        Ok(vec![
            circuit_flag_bit(CircuitFlags::Jump).ok_or_else(missing)?,
            circuit_flag_bit(CircuitFlags::WriteLookupOutputToRD).ok_or_else(missing)?,
            circuit_flag_bit(CircuitFlags::VirtualInstruction).ok_or_else(missing)?,
            instruction_flag_bit(InstructionFlagKind::Branch).ok_or_else(missing)?,
            FLAG_BIT_NEXT_IS_NOOP,
        ])
    }

    #[cfg(test)]
    pub fn new(context: &CudaKernelContext, packed: &Packed) -> Result<Self, CudaError> {
        let cycles = packed.flags.len();
        if cycles == 0
            || packed.narrow.len() != cycles * NARROW
            || packed.wide.len() != cycles * WIDE * 2
        {
            return Err(CudaError::InvariantViolation {
                reason: "the packed product columns must hold one entry per cycle per column",
            });
        }
        Ok(Self {
            narrow: context.upload_u64_slice(&packed.narrow)?,
            wide: context.upload_u64_slice(&packed.wide)?,
            flags: context.upload_u32_slice(&packed.flags)?,
            layout: context.upload_u32_slice(&witness::CLAIM_LAYOUT)?,
            cycles,
        })
    }

    pub const fn cycles(&self) -> usize {
        self.cycles
    }

    pub const fn narrow(&self) -> &CudaSlice<u64> {
        &self.narrow
    }

    pub const fn wide(&self) -> &CudaSlice<u64> {
        &self.wide
    }

    pub const fn flags(&self) -> &CudaSlice<u32> {
        &self.flags
    }

    pub const fn layout(&self) -> &CudaSlice<u32> {
        &self.layout
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_field::Fr;
    use jolt_witness::collect_bundles;

    use super::super::witness::{self, SpartanProductWitness};
    use super::DeviceProductColumns;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::with_r1cs_witness;
    use crate::cuda::witness::session_atom_columns;
    use crate::ProofSession;

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 10;

    #[test]
    fn device_product_columns_match_the_host_encoder() {
        let Some(context) = shared_context() else {
            return;
        };
        with_r1cs_witness(
            LOG_T,
            RAM_K,
            JoltOneHotConfig {
                log_k_chunk: 8,
                lookups_ra_virtual_log_k_chunk: 32,
            },
            7,
            |witness| {
                let cycles = 1usize << LOG_T;
                let rows: Vec<SpartanProductWitness> =
                    collect_bundles(witness, cycles).expect("reference product rows");
                let expected = DeviceProductColumns::new(context, &witness::pack(&rows))
                    .expect("host-encoded columns");

                let mut session = ProofSession::default();
                let atoms = session_atom_columns::<Fr>(context, &mut session, witness, cycles)
                    .expect("atom columns");
                let got = DeviceProductColumns::from_device(context, &atoms, cycles)
                    .expect("device-gathered columns");

                let expected_flags = context.download_u32(expected.flags()).expect("flags");
                assert!(
                    expected_flags.iter().any(|&mask| mask != expected_flags[0]),
                    "every cycle has the same mask, so a kernel ignoring the row would pass",
                );
                assert_eq!(
                    context.download_u64(got.narrow()).expect("narrow"),
                    context.download_u64(expected.narrow()).expect("narrow"),
                    "the narrow slots diverge",
                );
                assert_eq!(
                    context.download_u64(got.wide()).expect("wide"),
                    context.download_u64(expected.wide()).expect("wide"),
                    "the wide limbs diverge",
                );
                assert_eq!(
                    context.download_u32(got.flags()).expect("flags"),
                    expected_flags,
                    "the flag masks diverge",
                );
            },
        );
    }
}
