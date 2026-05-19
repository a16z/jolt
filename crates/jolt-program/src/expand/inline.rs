use std::ops::Deref;

use jolt_riscv::{
    JoltInstructionKind, NormalizedOperands, SourceInstructionKind, SourceInstructionRow,
};

use crate::expand::{
    allocator::{FIRST_INLINE_REGISTER, NUM_INLINE_REGISTERS},
    grammar::{ExpandedInstructionSequence, ExpansionBuilder, InlineTempId, RegisterOperand},
    ExpansionError,
};

/// Register operands decoded from an inline-format source instruction.
///
/// Inline encoding names the destination register `rs3` in the tracer format;
/// in normalized rows it is stored as `rd`. Static builders use this type so
/// shipped inline crates do not need to know that storage detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InlineOperands {
    pub rs1: u8,
    pub rs2: u8,
    pub rs3: u8,
}

impl InlineOperands {
    pub fn from_source_row(row: SourceInstructionRow) -> Result<Self, ExpansionError> {
        Ok(Self {
            rs1: row
                .operands
                .rs1
                .ok_or(ExpansionError::MalformedInstruction(
                    "inline instruction missing rs1",
                ))?,
            rs2: row
                .operands
                .rs2
                .ok_or(ExpansionError::MalformedInstruction(
                    "inline instruction missing rs2",
                ))?,
            rs3: row.operands.rd.ok_or(ExpansionError::MalformedInstruction(
                "inline instruction missing rs3",
            ))?,
        })
    }
}

impl From<InlineOperands> for NormalizedOperands {
    fn from(operands: InlineOperands) -> Self {
        Self {
            rd: Some(operands.rs3),
            rs1: Some(operands.rs1),
            rs2: Some(operands.rs2),
            imm: 0,
        }
    }
}

/// Operand value used by recipe helpers that can constant-fold simple logic.
///
/// Helper methods such as `add`, `xor`, and `rotri32` return `Imm` when both
/// inputs are known constants and emit rows only when a register result is
/// required.
#[derive(Clone, Copy, Debug)]
pub enum Value {
    Imm(u64),
    Reg(u8),
}

/// Builder-owned handle for a virtual register allocated to a registered inline.
///
/// The encoded `u8` dereferences to the concrete register number expected by
/// existing inline sequence builders. Internally, the handle carries a symbolic
/// temp id so the materializer can bind it to the central `ExpansionAllocator`
/// and later append reset rows for every inline register that was touched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InlineRegister {
    encoded: u8,
    temp: InlineTempId,
}

impl InlineRegister {
    const fn new(temp: InlineTempId) -> Self {
        Self {
            encoded: FIRST_INLINE_REGISTER + temp.0,
            temp,
        }
    }
}

impl Deref for InlineRegister {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        &self.encoded
    }
}

/// Marker trait that lets inline crates keep `emit_r::<ADD>(...)` ergonomics.
///
/// `SOURCE_KIND` is always the decoded source instruction. `JOLT_KIND` is
/// `Some` only for target-legal final instructions; when it is `None`, the
/// builder records a source-only helper row and the materializer recursively
/// expands it through the central `jolt-program` pipeline.
pub trait InlineInstruction {
    const SOURCE_KIND: SourceInstructionKind;
    const JOLT_KIND: Option<JoltInstructionKind>;
}

macro_rules! impl_inline_instruction {
    (
        instructions: [$($instr:ident => $marker:ident => $canonical_name:expr),* $(,)?]
    ) => {
        $(
            impl<T> InlineInstruction for jolt_riscv::instructions::$marker<T> {
                const SOURCE_KIND: SourceInstructionKind =
                    jolt_riscv::SourceInstruction::$marker(
                        jolt_riscv::instructions::$marker(())
                    );
                const JOLT_KIND: Option<JoltInstructionKind> =
                    Self::SOURCE_KIND.jolt_kind();
            }
        )*
    };
}

jolt_riscv::for_each_instruction_kind!(impl_inline_instruction);

/// Public recipe builder for registered inline static expansion.
///
/// The builder records symbolic operations, not concrete tracer instructions.
/// Inline crates may allocate/release inline registers and emit ordinary
/// instruction-shaped rows. Final register binding, recursive helper expansion,
/// metadata stamping, reset rows, and target-profile validation are deferred to
/// `ExpansionState`.
pub struct InlineExpansionBuilder {
    inner: ExpansionBuilder,
    inline_live: [bool; NUM_INLINE_REGISTERS],
}

impl InlineExpansionBuilder {
    /// Start a recipe for one inline source row.
    ///
    /// The `source` row supplies the final address, compression bit, and
    /// top-level operands used for metadata stamping after materialization.
    pub fn new(source: SourceInstructionRow) -> Self {
        Self {
            inner: ExpansionBuilder::new(source),
            inline_live: [false; NUM_INLINE_REGISTERS],
        }
    }

    /// Allocate an inline virtual-register handle.
    ///
    /// The handle can be used anywhere a register operand is expected. It must
    /// be released before `finalize`; the materializer rejects leaked handles
    /// and appends reset rows for all handles that were allocated and released.
    pub fn allocate_for_inline(&mut self) -> Result<InlineRegister, ExpansionError> {
        let Some(index) = self.inline_live.iter().position(|allocated| !allocated) else {
            return Err(ExpansionError::VirtualRegisterExhausted { pool: "inline" });
        };
        self.inline_live[index] = true;
        let temp = InlineTempId(index as u8);
        self.inner.allocate_inline(temp);
        Ok(InlineRegister::new(temp))
    }

    /// Release a previously allocated inline register handle.
    ///
    /// Release is explicit so static recipes do not depend on Rust drop order or
    /// tracer-only RAII behavior. Double release is reported during
    /// materialization when the symbolic release cannot be matched to a live
    /// allocation.
    pub fn release(&mut self, register: InlineRegister) {
        let index = register.temp.index();
        if index < self.inline_live.len() {
            self.inline_live[index] = false;
        }
        self.inner.release_inline(register.temp);
    }

    /// Allocate a fixed-size array of inline registers.
    ///
    /// This keeps shipped crypto builders concise while preserving explicit
    /// release through `release_many` or `release_iter`.
    pub fn allocate_inline_array<const N: usize>(
        &mut self,
    ) -> Result<[InlineRegister; N], ExpansionError> {
        let mut registers = Vec::with_capacity(N);
        for _ in 0..N {
            registers.push(self.allocate_for_inline()?);
        }
        match registers.try_into() {
            Ok(registers) => Ok(registers),
            Err(_) => unreachable!("vector length is fixed by the loop"),
        }
    }

    /// Release every handle in a fixed-size inline-register array.
    pub fn release_many<const N: usize>(&mut self, registers: [InlineRegister; N]) {
        for register in registers {
            self.release(register);
        }
    }

    /// Release every handle produced by an iterator.
    pub fn release_iter(&mut self, registers: impl IntoIterator<Item = InlineRegister>) {
        for register in registers {
            self.release(register);
        }
    }

    /// Emit an R-format row or recursively expand a source-only R-format helper.
    pub fn emit_r<Op: InlineInstruction>(&mut self, rd: u8, rs1: u8, rs2: u8) {
        match Op::JOLT_KIND {
            Some(kind) => self.inner.emit_r(
                kind,
                Self::register_operand(rd),
                Self::register_operand(rs1),
                Self::register_operand(rs2),
            ),
            None => self.inner.expand_r(
                Op::SOURCE_KIND,
                Self::register_operand(rd),
                Self::register_operand(rs1),
                Self::register_operand(rs2),
            ),
        }
    }

    /// Emit an I-format row with an unsigned immediate.
    pub fn emit_i<Op: InlineInstruction>(&mut self, rd: u8, rs1: u8, imm: u64) {
        self.emit_i_signed::<Op>(rd, rs1, imm as i128);
    }

    /// Emit a load-shaped I-format row with a signed byte offset.
    pub fn emit_ld<Op: InlineInstruction>(&mut self, rd: u8, rs1: u8, imm: i64) {
        self.emit_i_signed::<Op>(rd, rs1, imm as i128);
    }

    /// Emit a J-format virtual row or recursively expand a source-only helper.
    pub fn emit_j<Op: InlineInstruction>(&mut self, rd: u8, imm: u64) {
        match Op::JOLT_KIND {
            Some(kind) => self
                .inner
                .emit_j(kind, Self::register_operand(rd), imm as i128),
            None => self
                .inner
                .expand_j(Op::SOURCE_KIND, Self::register_operand(rd), imm as i128),
        }
    }

    /// Emit a U-format row or recursively expand a source-only helper.
    pub fn emit_u<Op: InlineInstruction>(&mut self, rd: u8, imm: u64) {
        match Op::JOLT_KIND {
            Some(kind) => self
                .inner
                .emit_u(kind, Self::register_operand(rd), imm as i128),
            None => self
                .inner
                .expand_u(Op::SOURCE_KIND, Self::register_operand(rd), imm as i128),
        }
    }

    /// Emit an S-format row with a signed byte offset.
    pub fn emit_s<Op: InlineInstruction>(&mut self, rs1: u8, rs2: u8, imm: i64) {
        match Op::JOLT_KIND {
            Some(kind) => self.inner.emit_s(
                kind,
                Self::register_operand(rs1),
                Self::register_operand(rs2),
                imm as i128,
            ),
            None => self.inner.expand_s(
                Op::SOURCE_KIND,
                Self::register_operand(rs1),
                Self::register_operand(rs2),
                imm as i128,
            ),
        }
    }

    /// Emit a B-format row with a signed branch offset.
    pub fn emit_b<Op: InlineInstruction>(&mut self, rs1: u8, rs2: u8, imm: i64) {
        match Op::JOLT_KIND {
            Some(kind) => self.inner.emit_b(
                kind,
                Self::register_operand(rs1),
                Self::register_operand(rs2),
                imm as i128,
            ),
            None => self.inner.expand_b(
                Op::SOURCE_KIND,
                Self::register_operand(rs1),
                Self::register_operand(rs2),
                imm as i128,
            ),
        }
    }

    /// Emit a virtual shift row whose immediate encodes the shift mask.
    pub fn emit_vshift_i<Op: InlineInstruction>(&mut self, rd: u8, rs1: u8, imm: u64) {
        self.emit_i::<Op>(rd, rs1, imm);
    }

    /// Emit a virtual shift row whose shift amount comes from `rs2`.
    pub fn emit_vshift_r<Op: InlineInstruction>(&mut self, rd: u8, rs1: u8, rs2: u8) {
        self.emit_r::<Op>(rd, rs1, rs2);
    }

    /// Emit an address/alignment assertion helper.
    ///
    /// These helpers are source-only pseudo-I rows: they read `rs1` and an
    /// immediate offset but do not write `rd`, so they must always be routed
    /// through recursive source expansion.
    pub fn emit_align<Op: InlineInstruction>(&mut self, rs1: u8, imm: i64) {
        self.inner
            .expand_address(Op::SOURCE_KIND, Self::register_operand(rs1), imm as i128);
    }

    /// Emit or constant-fold a binary operation.
    ///
    /// Register/register operands use `OR`, register/immediate operands use
    /// `OI`, and immediate/immediate operands are folded with `fold`.
    pub fn bin<OR: InlineInstruction, OI: InlineInstruction>(
        &mut self,
        rs1: Value,
        rs2: Value,
        rd: u8,
        fold: fn(u64, u64) -> u64,
    ) -> Value {
        match (rs1, rs2) {
            (Value::Reg(r1), Value::Reg(r2)) => {
                self.emit_r::<OR>(rd, r1, r2);
                Value::Reg(rd)
            }
            (Value::Reg(r1), Value::Imm(imm)) => {
                self.emit_i::<OI>(rd, r1, imm);
                Value::Reg(rd)
            }
            (Value::Imm(_), Value::Reg(_)) => self.bin::<OR, OI>(rs2, rs1, rd, fold),
            (Value::Imm(i1), Value::Imm(i2)) => Value::Imm(fold(i1, i2)),
        }
    }

    /// Emit or constant-fold wrapping addition.
    pub fn add(&mut self, rs1: Value, rs2: Value, rd: u8) -> Value {
        self.bin::<jolt_riscv::instructions::Add, jolt_riscv::instructions::Addi>(
            rs1,
            rs2,
            rd,
            |x, y| x.wrapping_add(y),
        )
    }

    /// Emit or constant-fold a 32-bit logical right shift.
    pub fn srli(&mut self, rs1: Value, shamt: u32, rd: u8) -> Value {
        if shamt == 0 {
            return self.xor(rs1, Value::Imm(0), rd);
        }
        match rs1 {
            Value::Reg(rs1) => {
                self.emit_i::<jolt_riscv::instructions::SrlIW>(rd, rs1, (shamt & 0x1f) as u64);
                Value::Reg(rd)
            }
            Value::Imm(val) => Value::Imm(((val as u32) >> shamt) as u64),
        }
    }

    /// Emit or constant-fold a 32-bit rotate-right operation.
    pub fn rotri32(&mut self, rs1: Value, shamt: u32, rd: u8) -> Value {
        if shamt == 0 || shamt == 32 {
            return self.xor(rs1, Value::Imm(0), rd);
        }
        let ones = (1u64 << (32 - shamt)) - 1;
        let mask = ones << shamt;
        match rs1 {
            Value::Reg(rs1_reg) => {
                self.emit_vshift_i::<jolt_riscv::instructions::VirtualRotriw>(rd, rs1_reg, mask);
                Value::Reg(rd)
            }
            Value::Imm(val) => Value::Imm(((val as u32).rotate_right(shamt)) as u64),
        }
    }

    /// Emit or constant-fold the XOR of two 32-bit rotate-right operations.
    pub fn rotri_xor_rotri32(
        &mut self,
        rs1: Value,
        imm1: u32,
        imm2: u32,
        rd: u8,
        scratch: u8,
    ) -> Value {
        let r1 = self.rotri32(rs1, imm1, scratch);
        let r2 = self.rotri32(rs1, imm2, rd);
        self.xor(r1, r2, rd)
    }

    /// Emit or constant-fold bitwise XOR.
    pub fn xor(&mut self, rs1: Value, rs2: Value, rd: u8) -> Value {
        self.bin::<jolt_riscv::instructions::Xor, jolt_riscv::instructions::XorI>(
            rs1,
            rs2,
            rd,
            |x, y| x ^ y,
        )
    }

    /// Emit or constant-fold bitwise AND.
    pub fn and(&mut self, rs1: Value, rs2: Value, rd: u8) -> Value {
        self.bin::<jolt_riscv::instructions::And, jolt_riscv::instructions::AndI>(
            rs1,
            rs2,
            rd,
            |x, y| x & y,
        )
    }

    /// Emit or constant-fold a 64-bit rotate-right operation.
    pub fn rotri(&mut self, rs1: Value, imm: u64, rd: u8) -> Value {
        match rs1 {
            Value::Reg(rs1) => {
                self.emit_vshift_i::<jolt_riscv::instructions::VirtualRotri>(rd, rs1, imm);
                Value::Reg(rd)
            }
            Value::Imm(val) => {
                let shift = imm.trailing_zeros();
                Value::Imm(val.rotate_right(shift))
            }
        }
    }

    /// Emit or constant-fold a 64-bit rotate-right by `amount` bits.
    pub fn rotr64(&mut self, rs1: Value, amount: u32, rd: u8) -> Value {
        if amount == 0 || amount == 64 {
            return self.xor(rs1, Value::Imm(0), rd);
        }

        match rs1 {
            Value::Reg(rs1_reg) => {
                let ones = (1u64 << (64 - amount as u64)) - 1;
                let imm = ones << amount as u64;
                self.rotri(Value::Reg(rs1_reg), imm, rd)
            }
            Value::Imm(val) => Value::Imm(val.rotate_right(amount)),
        }
    }

    /// Emit or constant-fold a 64-bit rotate-left by `amount` bits.
    pub fn rotl64(&mut self, rs1: Value, amount: u32, rd: u8) -> Value {
        self.rotr64(rs1, 64 - amount, rd)
    }

    /// Finish recipe construction.
    ///
    /// This does not validate leaked registers or profile support. Those checks
    /// run during materialization so provider output and recursive helper rows
    /// are validated in one central place.
    pub fn finalize(self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        self.inner.finalize()
    }

    fn emit_i_signed<Op: InlineInstruction>(&mut self, rd: u8, rs1: u8, imm: i128) {
        match Op::JOLT_KIND {
            Some(kind) => self.inner.emit_i(
                kind,
                Self::register_operand(rd),
                Self::register_operand(rs1),
                imm,
            ),
            None => self.inner.expand_i(
                Op::SOURCE_KIND,
                Self::register_operand(rd),
                Self::register_operand(rs1),
                imm,
            ),
        }
    }

    fn register_operand(register: u8) -> RegisterOperand {
        if (FIRST_INLINE_REGISTER..FIRST_INLINE_REGISTER + NUM_INLINE_REGISTERS as u8)
            .contains(&register)
        {
            RegisterOperand::InlineTemp(InlineTempId(register - FIRST_INLINE_REGISTER))
        } else {
            RegisterOperand::Register(register)
        }
    }
}
