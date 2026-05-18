use std::collections::VecDeque;

pub use num_bigint::BigUint as NBigUint;
use tracer::{
    instruction::inline::INLINE,
    utils::{
        inline_sequence_writer::{write_inline_trace, InlineDescriptor, SequenceInputs},
        virtual_registers::VirtualRegisterAllocator,
    },
};

pub use inventory;
pub use jolt_program::expand::{
    ExpandedInstructionSequence, ExpansionError, InlineAdmissibility, InlineExpansionBuilder,
    InlineOperands, InlineRegister, SafetyRequirement, Value,
};
pub use tracer::emulator::cpu::Cpu;
pub use tracer::instruction::format::format_inline::FormatInline;
pub use tracer::instruction::inline::InlineRegistration;
pub use tracer::utils::inline_sequence_writer::AppendMode;
pub use tracer::InlineExtension;

pub mod instruction {
    macro_rules! alias_instruction {
        ($module:ident, $alias:ident, $target:ident) => {
            pub mod $module {
                pub use jolt_riscv::instructions::$target as $alias;
            }
        };
    }

    alias_instruction!(add, ADD, Add);
    alias_instruction!(addi, ADDI, Addi);
    alias_instruction!(and, AND, And);
    alias_instruction!(andi, ANDI, AndI);
    alias_instruction!(andn, ANDN, Andn);
    alias_instruction!(ld, LD, Ld);
    alias_instruction!(lui, LUI, Lui);
    alias_instruction!(lw, LW, Lw);
    alias_instruction!(mul, MUL, Mul);
    alias_instruction!(mulhu, MULHU, MulHU);
    alias_instruction!(or, OR, Or);
    alias_instruction!(sd, SD, Sd);
    alias_instruction!(slli, SLLI, SllI);
    alias_instruction!(sltu, SLTU, SltU);
    alias_instruction!(srli, SRLI, SrlI);
    alias_instruction!(srliw, SRLIW, SrlIW);
    alias_instruction!(sub, SUB, Sub);
    alias_instruction!(virtual_advice, VirtualAdvice, VirtualAdvice);
    alias_instruction!(virtual_assert_eq, VirtualAssertEQ, AssertEq);
    alias_instruction!(virtual_assert_lte, VirtualAssertLTE, AssertLte);
    pub mod virtual_xor_rot {
        pub use jolt_riscv::instructions::VirtualXorRot16 as VirtualXORROT16;
        pub use jolt_riscv::instructions::VirtualXorRot24 as VirtualXORROT24;
        pub use jolt_riscv::instructions::VirtualXorRot32 as VirtualXORROT32;
        pub use jolt_riscv::instructions::VirtualXorRot63 as VirtualXORROT63;
    }
    pub mod virtual_xor_rotw {
        pub use jolt_riscv::instructions::VirtualXorRotW12 as VirtualXORROTW12;
        pub use jolt_riscv::instructions::VirtualXorRotW16 as VirtualXORROTW16;
        pub use jolt_riscv::instructions::VirtualXorRotW7 as VirtualXORROTW7;
        pub use jolt_riscv::instructions::VirtualXorRotW8 as VirtualXORROTW8;
    }
    alias_instruction!(
        virtual_zero_extend_word,
        VirtualZeroExtendWord,
        VirtualZeroExtendWord
    );
}

/// Convert a slice of `u64` limbs (little-endian) to `NBigUint`.
pub fn limbs_to_nbiguint(limbs: &[u64]) -> NBigUint {
    let mut bytes = Vec::with_capacity(limbs.len() * 8);
    for &limb in limbs {
        for i in 0..8 {
            bytes.push(((limb >> (i * 8)) & 0xFF) as u8);
        }
    }
    NBigUint::from_bytes_le(&bytes)
}

/// Convert an `NBigUint` to a `Vec<u64>` of little-endian limbs.
pub fn nbiguint_to_limbs(n: &NBigUint) -> Vec<u64> {
    let bytes = n.to_bytes_le();
    let mut limbs = vec![0u64; bytes.len().div_ceil(8)];
    for (i, byte) in bytes.iter().enumerate() {
        limbs[i / 8] |= (*byte as u64) << ((i % 8) * 8);
    }
    limbs
}

/// Type of multiplication-style modular operation (multiply, square, or divide).
pub enum MulqType {
    Mul,
    Square,
    Div,
}

/// Shared advice computation for modular multiply/square/divide inlines.
///
/// Reads `a` from `rs1` and `b` from `rs2` (or `rs1` for square), computes the
/// quotient advice, and returns limbs as a `VecDeque<u64>`.
///
/// - `modulus`: given `is_scalar_field`, returns the field modulus as `NBigUint`
/// - `field_inv_mul`: given `(b_limbs, a_limbs)`, returns `b^{-1} * a mod q` as `NBigUint`
pub fn mulq_advice(
    operands: &FormatInline,
    cpu: &mut Cpu,
    is_scalar_field: bool,
    op_type: &MulqType,
    modulus: impl Fn(bool) -> NBigUint,
    field_inv_mul: impl Fn(&[u64; 4], &[u64; 4]) -> NBigUint,
) -> VecDeque<u64> {
    let a_addr = cpu.x[operands.rs1 as usize] as u64;
    let a = [
        cpu.mmu.load_doubleword(a_addr).unwrap().0,
        cpu.mmu.load_doubleword(a_addr + 8).unwrap().0,
        cpu.mmu.load_doubleword(a_addr + 16).unwrap().0,
        cpu.mmu.load_doubleword(a_addr + 24).unwrap().0,
    ];
    let b_addr = match op_type {
        MulqType::Square => a_addr,
        _ => cpu.x[operands.rs2 as usize] as u64,
    };
    let b = [
        cpu.mmu.load_doubleword(b_addr).unwrap().0,
        cpu.mmu.load_doubleword(b_addr + 8).unwrap().0,
        cpu.mmu.load_doubleword(b_addr + 16).unwrap().0,
        cpu.mmu.load_doubleword(b_addr + 24).unwrap().0,
    ];
    let a_big: NBigUint = limbs_to_nbiguint(&a);
    let b_big: NBigUint = limbs_to_nbiguint(&b);
    let q_big: NBigUint = modulus(is_scalar_field);
    match op_type {
        MulqType::Div => {
            let c_big = field_inv_mul(&b, &a);
            let c_limbs = nbiguint_to_limbs(&c_big);
            let quotient = (&b_big * &c_big) / &q_big;
            let quotient_limbs = nbiguint_to_limbs(&quotient);
            assert!(quotient_limbs.len() <= 4, "Result does not fit in 4 limbs");
            let mut padded_limbs = vec![0u64; 8];
            for i in 0..c_limbs.len() {
                padded_limbs[2 * i] = c_limbs[i];
            }
            for i in 0..quotient_limbs.len() {
                padded_limbs[2 * i + 1] = quotient_limbs[i];
            }
            VecDeque::from(padded_limbs)
        }
        _ => {
            let quotient = (a_big * b_big) / &q_big;
            let limbs = nbiguint_to_limbs(&quotient);
            assert!(limbs.len() <= 4, "Result does not fit in 4 limbs");
            let mut padded_limbs = vec![0u64; 4];
            padded_limbs[..limbs.len()].copy_from_slice(&limbs[..]);
            VecDeque::from(padded_limbs)
        }
    }
}

/// Trait for declaring an inline operation's metadata and sequence builder.
///
/// Implement this for each sub-inline (e.g. `Sha256Compression`, `Secp256k1MulQ`),
/// then pass the types to [`register_inlines!`] to generate registration boilerplate.
pub trait InlineOp: Send + Sync {
    const OPCODE: u32;
    const FUNCT3: u32;
    const FUNCT7: u32;
    const NAME: &'static str;
    const ADMISSIBILITY: InlineAdmissibility;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError>;

    fn build_advice(_operands: FormatInline, _cpu: &mut Cpu) -> Option<VecDeque<u64>> {
        None
    }
}

/// Write the default inline trace for a single `InlineOp` to `file` with the given `mode`.
pub fn store_trace<T: InlineOp>(file: &str, mode: AppendMode) -> Result<(), String> {
    let inline_info = InlineDescriptor::new(T::NAME.to_string(), T::OPCODE, T::FUNCT3, T::FUNCT7);
    let inputs = SequenceInputs::default();
    let inline = INLINE {
        opcode: T::OPCODE,
        funct3: T::FUNCT3,
        funct7: T::FUNCT7,
        address: inputs.address,
        operands: (&inputs).into(),
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: inputs.is_compressed,
    };
    let instructions = inline.inline_sequence(&VirtualRegisterAllocator::default());
    write_inline_trace(file, &inline_info, &inputs, &instructions, mode).map_err(|e| e.to_string())
}

/// Extension trait adding paired u32 load/store helpers to [`InlineExpansionBuilder`].
///
/// These combine two adjacent u32 values into a single 64-bit memory access,
/// halving the number of load/store instructions for u32 arrays on RV64.
pub trait InlineBuilderExt {
    fn load_paired_u32(&mut self, temp: u8, base: u8, offset: i64, vr_lo: u8, vr_hi: u8);
    fn store_paired_u32(&mut self, base: u8, offset: i64, vr_lo: u8, vr_hi: u8);
    fn load_paired_u32_dirty(&mut self, base: u8, offset: i64, vr_lo: u8, vr_hi: u8);
    /// Emit `count` pairs of (VirtualAdvice, SD) instructions.
    /// Each iteration loads one advice value into `vr`, then stores it to
    /// `base_reg + i*8`.
    fn emit_advice_stores(&mut self, vr: u8, base_reg: u8, count: usize);
}

impl InlineBuilderExt for InlineExpansionBuilder {
    /// Load two packed u32 from 8-byte aligned `base+offset` into `vr_lo` and `vr_hi`.
    /// Clean extraction: `vr_lo` gets zero-extended low 32 bits; `vr_hi` gets high 32 bits.
    /// Clobbers `temp` for the intermediate 64-bit load.
    fn load_paired_u32(&mut self, temp: u8, base: u8, offset: i64, vr_lo: u8, vr_hi: u8) {
        use instruction::ld::LD;
        use instruction::srli::SRLI;
        use instruction::virtual_zero_extend_word::VirtualZeroExtendWord;
        self.emit_ld::<LD>(temp, base, offset);
        self.emit_i::<VirtualZeroExtendWord>(vr_lo, temp, 0);
        self.emit_i::<SRLI>(vr_hi, temp, 32);
    }

    /// Store two u32 values to 8-byte aligned `base+offset` as a single SD.
    /// WARNING: clobbers both `vr_lo` and `vr_hi`.
    fn store_paired_u32(&mut self, base: u8, offset: i64, vr_lo: u8, vr_hi: u8) {
        use instruction::or::OR;
        use instruction::sd::SD;
        use instruction::slli::SLLI;
        use instruction::virtual_zero_extend_word::VirtualZeroExtendWord;
        self.emit_i::<VirtualZeroExtendWord>(vr_lo, vr_lo, 0);
        self.emit_i::<SLLI>(vr_hi, vr_hi, 32);
        self.emit_r::<OR>(vr_lo, vr_hi, vr_lo);
        self.emit_s::<SD>(base, vr_lo, offset);
    }

    /// Load two packed u32 from 8-byte aligned `base+offset` into `vr_lo` and `vr_hi`.
    /// WARNING: leaves junk in upper 32 bits of `vr_lo`. Safe only when downstream ops
    /// preserve correctness independent of upper bits (e.g. SHA-256 32-bit arithmetic).
    fn load_paired_u32_dirty(&mut self, base: u8, offset: i64, vr_lo: u8, vr_hi: u8) {
        use instruction::ld::LD;
        use instruction::srli::SRLI;
        self.emit_ld::<LD>(vr_lo, base, offset);
        self.emit_i::<SRLI>(vr_hi, vr_lo, 32);
    }

    fn emit_advice_stores(&mut self, vr: u8, base_reg: u8, count: usize) {
        use instruction::sd::SD;
        use instruction::virtual_advice::VirtualAdvice;
        for i in 0..count {
            self.emit_j::<VirtualAdvice>(vr, 0);
            self.emit_s::<SD>(base_reg, vr, i as i64 * 8);
        }
    }
}

/// Extension trait adding multiply-accumulate helpers to [`InlineExpansionBuilder`].
///
/// These emit RISC-V instruction patterns for 64-bit multiply-accumulate with
/// carry propagation, used by modular arithmetic inlines (secp256k1, P-256, etc.).
/// All methods take raw virtual register IDs (`u8`) and emit instructions via
/// `self.emit_r`.
pub trait MulAccExt {
    // (c2, c1) = lower(a * b) + c1; clobbers aux
    fn mac_low(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8);
    // (c2, c1) = upper(a * b) + c1; clobbers aux
    fn mac_high(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8);
    // (c2, c1) += lower(a * b); clobbers aux
    fn mac_low_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8);
    // (c2, c1) += upper(a * b); clobbers aux
    fn mac_high_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8);
    // if carry_exists: mac_low_w_carry, else: mac_low
    fn mac_low_conditional(&mut self, carry_exists: bool, c2: u8, c1: u8, a: u8, b: u8, aux: u8);
    // if carry_exists: mac_high_w_carry, else: mac_high
    fn mac_high_conditional(&mut self, carry_exists: bool, c2: u8, c1: u8, a: u8, b: u8, aux: u8);
    // (c2, c1) = 2*lower(a * b) + c1; clobbers aux
    fn m2ac_low(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8);
    // (c2, c1) = 2*upper(a * b) + c1; clobbers aux
    fn m2ac_high(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8);
    // (c2, c1) += 2*lower(a * b); clobbers aux, aux2
    fn m2ac_low_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8, aux2: u8);
    // (c2, c1) += 2*upper(a * b); clobbers aux, aux2
    fn m2ac_high_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8, aux2: u8);
    // (c2, c1) = c1 + val; sets c2 = carry (no multiply, for p[0]=1 case)
    fn adc(&mut self, c2: u8, c1: u8, val: u8);
    // (c2, c1) += val with existing carry; clobbers aux
    fn adc_w_carry(&mut self, c2: u8, c1: u8, val: u8, aux: u8);
    // if carry_exists: adc_w_carry, else: adc
    fn add_conditional(&mut self, carry_exists: bool, c2: u8, c1: u8, val: u8, aux: u8);
}

impl MulAccExt for InlineExpansionBuilder {
    fn mac_low(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.emit_r::<instruction::mul::MUL>(aux, a, b);
        self.emit_r::<instruction::add::ADD>(c1, c1, aux);
        self.emit_r::<instruction::sltu::SLTU>(c2, c1, aux);
    }

    fn mac_high(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.emit_r::<instruction::mulhu::MULHU>(aux, a, b);
        self.emit_r::<instruction::add::ADD>(c1, c1, aux);
        self.emit_r::<instruction::sltu::SLTU>(c2, c1, aux);
    }

    fn mac_low_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.emit_r::<instruction::mul::MUL>(aux, a, b);
        self.emit_r::<instruction::add::ADD>(c1, c1, aux);
        self.emit_r::<instruction::sltu::SLTU>(aux, c1, aux);
        self.emit_r::<instruction::add::ADD>(c2, c2, aux);
    }

    fn mac_high_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.emit_r::<instruction::mulhu::MULHU>(aux, a, b);
        self.emit_r::<instruction::add::ADD>(c1, c1, aux);
        self.emit_r::<instruction::sltu::SLTU>(aux, c1, aux);
        self.emit_r::<instruction::add::ADD>(c2, c2, aux);
    }

    fn mac_low_conditional(&mut self, carry_exists: bool, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        if carry_exists {
            self.mac_low_w_carry(c2, c1, a, b, aux);
        } else {
            self.mac_low(c2, c1, a, b, aux);
        }
    }

    fn mac_high_conditional(&mut self, carry_exists: bool, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        if carry_exists {
            self.mac_high_w_carry(c2, c1, a, b, aux);
        } else {
            self.mac_high(c2, c1, a, b, aux);
        }
    }

    fn m2ac_low(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.emit_r::<instruction::mul::MUL>(aux, a, b);
        self.emit_r::<instruction::add::ADD>(c1, c1, aux);
        self.emit_r::<instruction::sltu::SLTU>(c2, c1, aux);
        self.emit_r::<instruction::add::ADD>(c1, c1, aux);
        self.emit_r::<instruction::sltu::SLTU>(aux, c1, aux);
        self.emit_r::<instruction::add::ADD>(c2, c2, aux);
    }

    fn m2ac_high(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.emit_r::<instruction::mulhu::MULHU>(aux, a, b);
        self.emit_r::<instruction::add::ADD>(c1, c1, aux);
        self.emit_r::<instruction::sltu::SLTU>(c2, c1, aux);
        self.emit_r::<instruction::add::ADD>(c1, c1, aux);
        self.emit_r::<instruction::sltu::SLTU>(aux, c1, aux);
        self.emit_r::<instruction::add::ADD>(c2, c2, aux);
    }

    fn m2ac_low_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8, aux2: u8) {
        self.emit_r::<instruction::mul::MUL>(aux, a, b);
        self.emit_r::<instruction::add::ADD>(c1, c1, aux);
        self.emit_r::<instruction::sltu::SLTU>(aux2, c1, aux);
        self.emit_r::<instruction::add::ADD>(c2, c2, aux2);
        self.emit_r::<instruction::add::ADD>(c1, c1, aux);
        self.emit_r::<instruction::sltu::SLTU>(aux2, c1, aux);
        self.emit_r::<instruction::add::ADD>(c2, c2, aux2);
    }

    fn m2ac_high_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8, aux2: u8) {
        self.emit_r::<instruction::mulhu::MULHU>(aux, a, b);
        self.emit_r::<instruction::add::ADD>(c1, c1, aux);
        self.emit_r::<instruction::sltu::SLTU>(aux2, c1, aux);
        self.emit_r::<instruction::add::ADD>(c2, c2, aux2);
        self.emit_r::<instruction::add::ADD>(c1, c1, aux);
        self.emit_r::<instruction::sltu::SLTU>(aux2, c1, aux);
        self.emit_r::<instruction::add::ADD>(c2, c2, aux2);
    }

    fn adc(&mut self, c2: u8, c1: u8, val: u8) {
        self.emit_r::<instruction::add::ADD>(c1, c1, val);
        self.emit_r::<instruction::sltu::SLTU>(c2, c1, val);
    }

    fn adc_w_carry(&mut self, c2: u8, c1: u8, val: u8, aux: u8) {
        self.emit_r::<instruction::add::ADD>(c1, c1, val);
        self.emit_r::<instruction::sltu::SLTU>(aux, c1, val);
        self.emit_r::<instruction::add::ADD>(c2, c2, aux);
    }

    fn add_conditional(&mut self, carry_exists: bool, c2: u8, c1: u8, val: u8, aux: u8) {
        if carry_exists {
            self.adc_w_carry(c2, c1, val, aux);
        } else {
            self.adc(c2, c1, val);
        }
    }
}

/// Generate `store_inlines()` and submit `InlineRegistration` entries to `inventory`.
///
/// Each `InlineOp` type gets an `inventory::submit!` that registers it at link time.
/// The tracer's INLINE instruction discovers these registrations automatically.
///
/// ```ignore
/// register_inlines! {
///     trace_file: "sha256_trace.joltinline",
///     extension: jolt_inlines_sdk::host::InlineExtension::Sha2,
///     ops: [Sha256Compression, Sha256CompressionInitial],
/// }
/// ```
#[macro_export]
macro_rules! register_inlines {
    (
        trace_file: $trace_file:expr,
        extension: $extension:expr,
        ops: [$first:ty $(, $rest:ty)*$(,)?]$(,)?
    ) => {
        pub fn store_inlines() -> Result<(), String> {
            $crate::host::store_trace::<$first>(
                $trace_file,
                $crate::host::AppendMode::Overwrite,
            )?;
            $($crate::host::store_trace::<$rest>(
                $trace_file,
                $crate::host::AppendMode::Append,
            )?;)*
            Ok(())
        }

        $crate::__submit_inline_op!($first, $extension);
        $($crate::__submit_inline_op!($rest, $extension);)*
    };
}

/// Helper macro to submit a single `InlineOp` to inventory.
#[macro_export]
macro_rules! __submit_inline_op {
    ($op:ty, $extension:expr) => {
        const _: () = {
            assert!(
                <$op as $crate::host::InlineOp>::OPCODE == 0x0B
                    || <$op as $crate::host::InlineOp>::OPCODE == 0x2B,
                "OPCODE must be 0x0B (custom-0) or 0x2B (custom-1)"
            );
            assert!(<$op as $crate::host::InlineOp>::FUNCT3 <= 7);
            assert!(<$op as $crate::host::InlineOp>::FUNCT7 <= 127);
        };

        $crate::host::inventory::submit! {
            $crate::host::InlineRegistration {
                opcode: <$op as $crate::host::InlineOp>::OPCODE,
                funct3: <$op as $crate::host::InlineOp>::FUNCT3,
            funct7: <$op as $crate::host::InlineOp>::FUNCT7,
            extension: $extension,
            name: <$op as $crate::host::InlineOp>::NAME,
            admissibility: <$op as $crate::host::InlineOp>::ADMISSIBILITY,
            build_sequence: <$op as $crate::host::InlineOp>::build_sequence,
            build_advice: <$op as $crate::host::InlineOp>::build_advice,
        }
        }
    };
}
