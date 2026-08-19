use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::Field;
use jolt_riscv::CircuitFlags;
use jolt_witness::backend::cuda::{
    circuit_flag_bit, DeviceAtomColumns, DeviceTrace, FLAG_BIT_PRODUCT_NEGATIVE,
    FLAG_BIT_SHOULD_BRANCH, FLAG_BIT_SHOULD_JUMP,
};

#[cfg(test)]
use super::witness::Packed;
use super::witness::{
    self, FIRST_IN_SEQUENCE_BIT, NARROW, NEXT_IS_FIRST_IN_SEQUENCE_BIT, NEXT_IS_VIRTUAL_BIT,
    NEXT_PC_SLOT, NEXT_UNEXPANDED_PC_SLOT, PC_SLOT, SIGN_BIT_BASE, UNEXPANDED_PC_SLOT,
    VIRTUAL_INSTRUCTION_BIT, WIDE,
};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::error::CudaError;

const GATHER_BITS: usize = 16;

const GATHER_CIRCUIT_ORDER: [CircuitFlags; 14] = [
    CircuitFlags::AddOperands,
    CircuitFlags::SubtractOperands,
    CircuitFlags::MultiplyOperands,
    CircuitFlags::Load,
    CircuitFlags::Store,
    CircuitFlags::Jump,
    CircuitFlags::WriteLookupOutputToRD,
    CircuitFlags::VirtualInstruction,
    CircuitFlags::Assert,
    CircuitFlags::DoNotUpdateUnexpandedPC,
    CircuitFlags::Advice,
    CircuitFlags::IsCompressed,
    CircuitFlags::IsFirstInSequence,
    CircuitFlags::IsLastInSequence,
];

pub struct DeviceR1csInputs {
    narrow: CudaSlice<u64>,
    wide: CudaSlice<u64>,
    flags: CudaSlice<u32>,
    layout: CudaSlice<u32>,
    cycles: usize,
}

impl DeviceR1csInputs {
    #[cfg(feature = "allocative")]
    pub fn device_bytes(&self) -> usize {
        (self.narrow.len() + self.wide.len()) * size_of::<u64>()
            + (self.flags.len() + self.layout.len()) * size_of::<u32>()
    }

    #[cfg(test)]
    pub fn new(context: &CudaKernelContext, packed: &Packed) -> Result<Self, CudaError> {
        let cycles = packed.flags.len();
        if cycles == 0
            || packed.narrow.len() != cycles * NARROW
            || packed.wide.len() != cycles * WIDE * 2
        {
            return Err(CudaError::InvariantViolation {
                reason: "the packed R1CS inputs must hold one entry per cycle per column",
            });
        }

        let narrow = context.upload_u64_slice(&packed.narrow)?;
        let wide = context.upload_u64_slice(&packed.wide)?;
        let raw = context.upload_u32_slice(&packed.flags)?;
        Self::finish(context, narrow, wide, raw, cycles)
    }

    pub fn from_device(
        context: &CudaKernelContext,
        trace: &DeviceTrace,
        atoms: &DeviceAtomColumns,
        pc_words: &CudaSlice<u32>,
        cycles: usize,
    ) -> Result<Self, CudaError> {
        if cycles == 0 || trace.cycles() < cycles || pc_words.len() < cycles {
            return Err(CudaError::InvariantViolation {
                reason: "the device R1CS sources do not cover the requested cycles",
            });
        }

        let mut narrow = context.alloc_u64(cycles * NARROW)?;
        let mut wide = context.alloc_u64(cycles * WIDE * 2)?;
        let mut raw = context.alloc_u32(cycles)?;
        let sources = context.upload_u32_slice(&Self::gather_bit_sources()?)?;
        let mut unmapped = context.upload_u64_slice(&[u64::MAX])?;
        let product_sign = FLAG_BIT_PRODUCT_NEGATIVE;

        let count = CudaKernelContext::count_of(cycles)?;
        let mut builder = context.stream().launch_builder(context.so_gather());
        let _ = builder.arg(trace.extras());
        let _ = builder.arg(trace.unexpanded_pc());
        let _ = builder.arg(trace.ram_address());
        let _ = builder.arg(pc_words);
        let _ = builder.arg(&atoms.flags);
        let _ = builder.arg(&sources);
        let _ = builder.arg(&product_sign);
        let _ = builder.arg(&atoms.left_instruction_input);
        let _ = builder.arg(&atoms.right_instruction_input);
        let _ = builder.arg(&atoms.left_lookup_operand);
        let _ = builder.arg(&atoms.right_lookup_operand);
        let _ = builder.arg(&atoms.lookup_output);
        let _ = builder.arg(&atoms.product_magnitude);
        let _ = builder.arg(&SIGN_BIT_BASE);
        let _ = builder.arg(&mut narrow);
        let _ = builder.arg(&mut wide);
        let _ = builder.arg(&mut raw);
        let _ = builder.arg(&mut unmapped);
        let _ = builder.arg(&count);
        // SAFETY: thread `t < cycles` writes the `NARROW` u64s at
        // `t * NARROW`, the `WIDE * 2` u64s at `t * WIDE * 2`, and `raw[t]` —
        // all inside allocations sized for `cycles` rows. It reads index `t` of
        // each source column and the `EXTRA_WORDS` consecutive words at
        // `t * EXTRA_WORDS`, all of which cover at least `cycles` rows by the
        // check above, plus the 16 in-range entries of `sources`. `unmapped` is
        // written only by `atomicMin`. Every buffer is a distinct allocation.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        context.stream().synchronize()?;

        let unmapped = context.download_u64(&unmapped)?;
        if unmapped.first().is_some_and(|&cycle| cycle != u64::MAX) {
            return Err(CudaError::InvariantViolation {
                reason: "a Spartan outer cycle has no bytecode PC mapping, so its Pc column is \
                         undefined",
            });
        }
        Self::finish(context, narrow, wide, raw, cycles)
    }

    fn gather_bit_sources() -> Result<Vec<u32>, CudaError> {
        let missing = || CudaError::InvariantViolation {
            reason: "a Spartan outer flag has no canonical device bit",
        };
        let mut sources = Vec::with_capacity(GATHER_BITS);
        sources.push(FLAG_BIT_SHOULD_BRANCH);
        sources.push(FLAG_BIT_SHOULD_JUMP);
        for flag in GATHER_CIRCUIT_ORDER {
            sources.push(circuit_flag_bit(flag).ok_or_else(missing)?);
        }
        if sources.len() != GATHER_BITS {
            return Err(CudaError::LengthMismatch {
                expected: GATHER_BITS,
                got: sources.len(),
            });
        }
        Ok(sources)
    }

    fn finish(
        context: &CudaKernelContext,
        mut narrow: CudaSlice<u64>,
        wide: CudaSlice<u64>,
        raw: CudaSlice<u32>,
        cycles: usize,
    ) -> Result<Self, CudaError> {
        let layout = context.upload_u32_slice(&witness::LAYOUT)?;
        let mut flags = context.alloc_u32(cycles)?;

        let count = CudaKernelContext::count_of(cycles)?;
        let pc = CudaKernelContext::count_of(PC_SLOT)?;
        let unexpanded_pc = CudaKernelContext::count_of(UNEXPANDED_PC_SLOT)?;
        let next_pc = CudaKernelContext::count_of(NEXT_PC_SLOT)?;
        let next_unexpanded_pc = CudaKernelContext::count_of(NEXT_UNEXPANDED_PC_SLOT)?;
        let mut builder = context.stream().launch_builder(context.so_shift());
        let _ = builder.arg(&raw);
        let _ = builder.arg(&mut narrow);
        let _ = builder.arg(&mut flags);
        let _ = builder.arg(&pc);
        let _ = builder.arg(&unexpanded_pc);
        let _ = builder.arg(&next_pc);
        let _ = builder.arg(&next_unexpanded_pc);
        let _ = builder.arg(&VIRTUAL_INSTRUCTION_BIT);
        let _ = builder.arg(&FIRST_IN_SEQUENCE_BIT);
        let _ = builder.arg(&NEXT_IS_VIRTUAL_BIT);
        let _ = builder.arg(&NEXT_IS_FIRST_IN_SEQUENCE_BIT);
        let _ = builder.arg(&count);
        // SAFETY: thread `t < cycles` reads `raw[t]` and, when `t + 1 < cycles`,
        // `raw[t + 1]` — both inside `raw`'s `cycles` entries — plus the two source
        // slots of row `t + 1`, inside `narrow`'s `cycles * NARROW` u64s. It writes
        // `flags[t]` (a fresh allocation, one slot per thread) and slots
        // `NEXT_PC_SLOT` / `NEXT_UNEXPANDED_PC_SLOT` of row `t`. Those two write
        // slots are disjoint from the two read slots (`PC_SLOT`,
        // `UNEXPANDED_PC_SLOT`), which no thread writes, so the in-place shift
        // cannot read a value another thread has already overwritten.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

        Ok(Self {
            narrow,
            wide,
            flags,
            layout,
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

pub struct LinearForms {
    offsets: Vec<u32>,
    counts: Vec<u32>,
    terms: Vec<u32>,
    coefficients: Vec<jolt_field::Fr>,
    constants: Vec<jolt_field::Fr>,
}

impl LinearForms {
    pub fn new() -> Self {
        Self {
            offsets: Vec::new(),
            counts: Vec::new(),
            terms: Vec::new(),
            coefficients: Vec::new(),
            constants: Vec::new(),
        }
    }

    pub fn push<F: Field>(&mut self, weights: &[F], constant: F) -> Result<(), CudaError> {
        if weights.len() != witness::VARIABLES {
            return Err(CudaError::LengthMismatch {
                expected: witness::VARIABLES,
                got: weights.len(),
            });
        }
        let start = u32::try_from(self.terms.len()).map_err(|_| CudaError::InvariantViolation {
            reason: "the Spartan outer linear forms exceed a u32 term offset",
        })?;
        let mut count = 0u32;
        for (variable, &weight) in weights.iter().enumerate() {
            if weight.is_zero() {
                continue;
            }
            self.terms.push(witness::LAYOUT[variable]);
            self.coefficients
                .push(crate::cuda::common::device::require_fr(weight)?);
            count += 1;
        }
        self.offsets.push(start);
        self.counts.push(count);
        self.constants
            .push(crate::cuda::common::device::require_fr(constant)?);
        Ok(())
    }

    pub fn upload(&self, context: &CudaKernelContext) -> Result<DeviceLinearForms, CudaError> {
        Ok(DeviceLinearForms {
            offsets: context.upload_u32_slice(&self.offsets)?,
            counts: context.upload_u32_slice(&self.counts)?,
            terms: context.upload_u32_slice(&self.terms)?,
            coefficients: context.upload(&self.coefficients)?,
            constants: context.upload(&self.constants)?,
        })
    }
}

pub struct DeviceLinearForms {
    pub offsets: CudaSlice<u32>,
    pub counts: CudaSlice<u32>,
    pub terms: CudaSlice<u32>,
    pub coefficients: crate::cuda::common::device::DeviceFrVec,
    pub constants: crate::cuda::common::device::DeviceFrVec,
}

impl DeviceLinearForms {
    pub fn bind_args<'a>(&'a self, builder: &mut cudarc::driver::LaunchArgs<'a>) {
        let _ = builder.arg(&self.offsets);
        let _ = builder.arg(&self.counts);
        let _ = builder.arg(self.constants.limbs());
        let _ = builder.arg(&self.terms);
        let _ = builder.arg(self.coefficients.limbs());
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

    use super::super::witness::{self, SpartanOuterWitness};
    use super::DeviceR1csInputs;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::device_columns::device_pc_words;
    use crate::cuda::common::testing::with_r1cs_witness;
    use crate::cuda::witness::{session_atom_columns, session_device_trace};
    use crate::ProofSession;

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 10;

    #[test]
    fn device_r1cs_inputs_match_the_host_encoder() {
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
                let rows: Vec<SpartanOuterWitness> =
                    collect_bundles(witness, cycles).expect("reference R1CS inputs");
                let expected = DeviceR1csInputs::new(context, &witness::pack(&rows))
                    .expect("host-encoded inputs");

                let mut session = ProofSession::default();
                let trace = session_device_trace::<Fr>(context, &mut session, witness, cycles)
                    .expect("device residency");
                let atoms = session_atom_columns::<Fr>(context, &mut session, witness, cycles)
                    .expect("atom columns");
                let pc_words = device_pc_words::<Fr>(context, &mut session, witness, cycles)
                    .expect("mapped pc words");
                let got = DeviceR1csInputs::from_device(context, &trace, &atoms, &pc_words, cycles)
                    .expect("device-gathered inputs");

                assert_eq!(got.cycles(), expected.cycles());
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
                let got_flags = context.download_u32(got.flags()).expect("flags");
                let expected_flags = context.download_u32(expected.flags()).expect("flags");
                assert!(
                    expected_flags.iter().any(|&mask| mask != expected_flags[0]),
                    "every cycle has the same mask, so a kernel ignoring the row would pass",
                );
                assert_eq!(got_flags, expected_flags, "the flag masks diverge");
            },
        );
    }

    #[test]
    fn the_kernel_source_agrees_on_the_gather_layout() {
        let source = include_str!("../kernels/spartan_outer.cu");
        for (name, value) in [
            ("SO_NARROW", witness::NARROW),
            ("SO_WIDE", witness::WIDE),
            ("SO_EXTRA_WORDS", jolt_witness::backend::cuda::EXTRA_WORDS),
            ("SO_EXTRA_RS1", jolt_witness::backend::cuda::EXTRA_RS1),
            ("SO_EXTRA_RS2", jolt_witness::backend::cuda::EXTRA_RS2),
            (
                "SO_EXTRA_RD_POST",
                jolt_witness::backend::cuda::EXTRA_RD_POST,
            ),
            (
                "SO_EXTRA_RAM_READ",
                jolt_witness::backend::cuda::EXTRA_RAM_READ,
            ),
            (
                "SO_EXTRA_RAM_WRITE",
                jolt_witness::backend::cuda::EXTRA_RAM_WRITE,
            ),
            ("SO_EXTRA_IMM_LO", jolt_witness::backend::cuda::EXTRA_IMM_LO),
            ("SO_EXTRA_IMM_HI", jolt_witness::backend::cuda::EXTRA_IMM_HI),
            ("SO_GATHER_BITS", super::GATHER_BITS),
        ] {
            let expected = format!("#define {name} {value}");
            assert!(
                source.contains(&expected),
                "the CUDA source must declare `{expected}`",
            );
        }
        assert_eq!(
            super::GATHER_CIRCUIT_ORDER.len() + 2,
            super::GATHER_BITS,
            "the gather bit map must cover should-branch, should-jump and every circuit flag",
        );
    }
}
