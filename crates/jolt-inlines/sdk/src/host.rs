use std::collections::VecDeque;

use jolt_tracer::{
    instruction::inline::INLINE,
    utils::{
        inline_sequence_writer::{write_inline_trace, InlineDescriptor, SequenceInputs},
        virtual_registers::VirtualRegisterAllocator,
    },
};
pub use num_bigint::BigUint as NBigUint;

pub use inventory;
pub use jolt_program::expand::{
    ExpandedInstructionSequence, ExpansionError, InlineExpansionBuilder, InlineOperands,
    InlineRegister, Value,
};
pub use jolt_tracer::emulator::cpu::Cpu;
pub use jolt_tracer::instruction::format::format_inline::FormatInline;
pub use jolt_tracer::instruction::inline::InlineRegistration;
pub use jolt_tracer::utils::inline_sequence_writer::AppendMode;
pub use jolt_tracer::InlineExtension;

pub type FieldElementLimbs = [u64; 4];

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

fn load_field_element_limbs(cpu: &mut Cpu, address: u64) -> FieldElementLimbs {
    [
        cpu.mmu.load_doubleword(address).unwrap().0,
        cpu.mmu.load_doubleword(address + 8).unwrap().0,
        cpu.mmu.load_doubleword(address + 16).unwrap().0,
        cpu.mmu.load_doubleword(address + 24).unwrap().0,
    ]
}

fn nbiguint_to_field_limbs(n: &NBigUint) -> FieldElementLimbs {
    let limbs = nbiguint_to_limbs(n);
    assert!(limbs.len() <= 4, "Result does not fit in 4 limbs");
    let mut padded = [0u64; 4];
    padded[..limbs.len()].copy_from_slice(&limbs);
    padded
}

fn mulq_operands(
    operands: &FormatInline,
    cpu: &mut Cpu,
    op_type: &MulqType,
) -> (FieldElementLimbs, FieldElementLimbs) {
    let a_addr = cpu.x[operands.rs1 as usize] as u64;
    let a = load_field_element_limbs(cpu, a_addr);
    let b_addr = match op_type {
        MulqType::Square => a_addr,
        _ => cpu.x[operands.rs2 as usize] as u64,
    };
    let b = load_field_element_limbs(cpu, b_addr);
    (a, b)
}

/// Shared quotient advice computation for modular multiply/square inlines.
///
/// Reads `a` from `rs1` and `b` from `rs2` (or `rs1` for square), then computes
/// `w` such that `a * b = w * q + c`.
pub fn mulq_quotient_advice(
    operands: &FormatInline,
    cpu: &mut Cpu,
    is_scalar_field: bool,
    op_type: &MulqType,
    modulus: impl Fn(bool) -> NBigUint,
) -> QuotientAdvice {
    assert!(
        !matches!(op_type, MulqType::Div),
        "division advice must use mulq_division_advice"
    );
    let (a, b) = mulq_operands(operands, cpu, op_type);
    let quotient = (limbs_to_nbiguint(&a) * limbs_to_nbiguint(&b)) / modulus(is_scalar_field);
    QuotientAdvice {
        quotient: nbiguint_to_field_limbs(&quotient),
    }
}

/// Shared advice computation for modular division inlines.
///
/// The returned `result` is `a / b`; `quotient` is `w` such that
/// `b * result = w * q + a`. Runtime advice rows consume these values as
/// `result[0], quotient[0], result[1], quotient[1], ...`.
pub fn mulq_division_advice(
    operands: &FormatInline,
    cpu: &mut Cpu,
    is_scalar_field: bool,
    modulus: impl Fn(bool) -> NBigUint,
    field_inv_mul: impl Fn(&FieldElementLimbs, &FieldElementLimbs) -> NBigUint,
) -> ModularDivisionAdvice {
    let (a, b) = mulq_operands(operands, cpu, &MulqType::Div);
    let result = field_inv_mul(&b, &a);
    let quotient = (limbs_to_nbiguint(&b) * &result) / modulus(is_scalar_field);
    ModularDivisionAdvice {
        result: nbiguint_to_field_limbs(&result),
        quotient: nbiguint_to_field_limbs(&quotient),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoAdvice;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuotientAdvice {
    pub quotient: FieldElementLimbs,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModularDivisionAdvice {
    pub result: FieldElementLimbs,
    pub quotient: FieldElementLimbs,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldElementAdvice {
    pub limbs: FieldElementLimbs,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SignedU128Advice {
    pub magnitude: [u64; 2],
    pub is_negative: bool,
}

impl SignedU128Advice {
    pub fn from_u128(value: u128, is_negative: bool) -> Self {
        Self {
            magnitude: [value as u64, (value >> 64) as u64],
            is_negative,
        }
    }

    fn sign_word(self) -> u64 {
        u64::from(self.is_negative)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlvDecompositionAdvice {
    pub k1: SignedU128Advice,
    pub k2: SignedU128Advice,
}

impl GlvDecompositionAdvice {
    pub fn from_sign_abs(decomposition: [(bool, u128); 2]) -> Self {
        let [(k1_negative, k1), (k2_negative, k2)] = decomposition;
        Self {
            k1: SignedU128Advice::from_u128(k1, k1_negative),
            k2: SignedU128Advice::from_u128(k2, k2_negative),
        }
    }
}

pub trait InlineAdvice: Default {
    fn into_runtime_advice(self) -> Option<VecDeque<u64>>;
}

impl InlineAdvice for NoAdvice {
    fn into_runtime_advice(self) -> Option<VecDeque<u64>> {
        None
    }
}

impl InlineAdvice for QuotientAdvice {
    fn into_runtime_advice(self) -> Option<VecDeque<u64>> {
        Some(VecDeque::from(self.quotient))
    }
}

impl InlineAdvice for ModularDivisionAdvice {
    fn into_runtime_advice(self) -> Option<VecDeque<u64>> {
        let mut advice = VecDeque::with_capacity(8);
        for i in 0..4 {
            advice.push_back(self.result[i]);
            advice.push_back(self.quotient[i]);
        }
        Some(advice)
    }
}

impl InlineAdvice for FieldElementAdvice {
    fn into_runtime_advice(self) -> Option<VecDeque<u64>> {
        Some(VecDeque::from(self.limbs))
    }
}

impl InlineAdvice for GlvDecompositionAdvice {
    fn into_runtime_advice(self) -> Option<VecDeque<u64>> {
        Some(VecDeque::from([
            self.k1.sign_word(),
            self.k1.magnitude[0],
            self.k1.magnitude[1],
            self.k2.sign_word(),
            self.k2.magnitude[0],
            self.k2.magnitude[1],
        ]))
    }
}

pub trait InlineOp: Send + Sync {
    /// Typed runtime advice values produced by this inline.
    type Advice: InlineAdvice;

    /// RISC-V custom opcode used to identify this inline.
    const OPCODE: u32;
    /// RISC-V funct3 selector used with `OPCODE`.
    const FUNCT3: u32;
    /// RISC-V funct7 selector used with `OPCODE` and `FUNCT3`.
    const FUNCT7: u32;
    /// Human-readable registration name used in diagnostics and fixtures.
    const NAME: &'static str;

    /// Build the static expansion recipe for this inline.
    ///
    /// This method must be deterministic and tracer-free: it receives decoded
    /// inline operands and records symbolic rows through `InlineExpansionBuilder`.
    /// Runtime CPU state and concrete advice values belong in `build_advice`,
    /// not in this static recipe.
    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError>;

    /// Optionally compute runtime advice values for this inline.
    ///
    /// The returned values are consumed in order by `VirtualAdvice` rows emitted
    /// by `build_sequence`. Returning `NoAdvice` means the static recipe contains
    /// no runtime-advice rows.
    fn build_advice(_operands: FormatInline, _cpu: &mut Cpu) -> Self::Advice {
        Self::Advice::default()
    }

    fn build_runtime_advice(operands: FormatInline, cpu: &mut Cpu) -> Option<VecDeque<u64>> {
        Self::build_advice(operands, cpu).into_runtime_advice()
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

/// Extension trait adding common load/store helpers to [`InlineExpansionBuilder`].
///
/// Paired u32 helpers combine two adjacent u32 values into a single 64-bit
/// memory access, halving the number of load/store instructions for u32 arrays
/// on RV64.
pub trait InlineBuilderExt {
    fn load_u64_range(&mut self, base: u8, offset_start: i64, registers: &[InlineRegister]);
    fn store_u64_range(&mut self, base: u8, offset_start: i64, registers: &[InlineRegister]);
    fn load_u32_range(&mut self, base: u8, offset_start: i64, registers: &[InlineRegister]);
    fn load_paired_u32_range_dirty(
        &mut self,
        base: u8,
        offset_start: i64,
        registers: &[InlineRegister],
    );
    fn load_paired_u32(&mut self, temp: u8, base: u8, offset: i64, vr_lo: u8, vr_hi: u8);
    fn store_paired_u32(&mut self, base: u8, offset: i64, vr_lo: u8, vr_hi: u8);
    fn load_paired_u32_dirty(&mut self, base: u8, offset: i64, vr_lo: u8, vr_hi: u8);
    /// Emit `count` pairs of (VirtualAdvice, SD) instructions.
    /// Each iteration loads one advice value into `vr`, then stores it to
    /// `base_reg + i*8`.
    fn emit_advice_stores(&mut self, vr: u8, base_reg: u8, count: usize);
}

impl InlineBuilderExt for InlineExpansionBuilder {
    fn load_u64_range(&mut self, base: u8, offset_start: i64, registers: &[InlineRegister]) {
        use instruction::ld::LD;
        for (i, register) in registers.iter().enumerate() {
            self.emit_ld::<LD>(**register, base, offset_start + i as i64 * 8);
        }
    }

    fn store_u64_range(&mut self, base: u8, offset_start: i64, registers: &[InlineRegister]) {
        use instruction::sd::SD;
        for (i, register) in registers.iter().enumerate() {
            self.emit_s::<SD>(base, **register, offset_start + i as i64 * 8);
        }
    }

    fn load_u32_range(&mut self, base: u8, offset_start: i64, registers: &[InlineRegister]) {
        use instruction::lw::LW;
        for (i, register) in registers.iter().enumerate() {
            self.emit_ld::<LW>(**register, base, offset_start + i as i64 * 4);
        }
    }

    fn load_paired_u32_range_dirty(
        &mut self,
        base: u8,
        offset_start: i64,
        registers: &[InlineRegister],
    ) {
        assert_eq!(
            registers.len() % 2,
            0,
            "paired u32 range requires an even register count"
        );
        for (i, pair) in registers.chunks_exact(2).enumerate() {
            self.load_paired_u32_dirty(base, offset_start + i as i64 * 8, *pair[0], *pair[1]);
        }
    }

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
            build_sequence: <$op as $crate::host::InlineOp>::build_sequence,
            build_advice: <$op as $crate::host::InlineOp>::build_runtime_advice,
        }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::InlineAdvice;
    use super::{
        GlvDecompositionAdvice, InlineBuilderExt, InlineExpansionBuilder, InlineRegister,
        ModularDivisionAdvice, NoAdvice, QuotientAdvice, SignedU128Advice,
    };
    use std::collections::VecDeque;

    #[test]
    fn no_advice_converts_to_no_runtime_queue() {
        assert!(NoAdvice.into_runtime_advice().is_none());
    }

    #[test]
    fn quotient_advice_converts_to_runtime_queue() {
        let advice = QuotientAdvice {
            quotient: [1, 2, 3, 4],
        };

        assert_eq!(
            advice.into_runtime_advice(),
            Some(VecDeque::from([1, 2, 3, 4]))
        );
    }

    #[test]
    fn division_advice_interleaves_result_and_quotient() {
        let advice = ModularDivisionAdvice {
            result: [10, 11, 12, 13],
            quotient: [20, 21, 22, 23],
        };

        assert_eq!(
            advice.into_runtime_advice(),
            Some(VecDeque::from([10, 20, 11, 21, 12, 22, 13, 23]))
        );
    }

    #[test]
    fn glv_advice_serializes_signs_before_magnitudes() {
        let advice = GlvDecompositionAdvice {
            k1: SignedU128Advice::from_u128((3u128 << 64) | 2, true),
            k2: SignedU128Advice::from_u128((5u128 << 64) | 4, false),
        };

        assert_eq!(
            advice.into_runtime_advice(),
            Some(VecDeque::from([1, 2, 3, 0, 4, 5]))
        );
    }

    #[test]
    fn range_helpers_accept_register_slices() {
        fn assert_methods(
            asm: &mut InlineExpansionBuilder,
            base: u8,
            registers: &[InlineRegister],
        ) {
            asm.load_u64_range(base, 0, registers);
            asm.store_u64_range(base, 0, registers);
            asm.load_u32_range(base, 0, registers);
            asm.load_paired_u32_range_dirty(base, 0, registers);
        }

        let _ = assert_methods;
    }
}
