use common::constants::RAM_START_ADDRESS;
use jolt_riscv::instructions::{VirtualAdviceLoad, VirtualSignExtendWord};
use jolt_riscv::SourceInstructionKind as Kind;

use super::*;

/// Computes SC success from the selected width's reservation register.
///
/// The RAM guard prevents failure-path stores from touching I/O and excludes
/// the cleared reservation value zero. LR.W sets only the word reservation;
/// LR.D sets both. Equality therefore enforces the tracer's deterministic
/// policy, including the width check, without permitting spurious failure.
pub(in crate::expand) fn expand_sc_success(
    asm: &mut ExpansionBuilder,
    address: RegisterOperand,
    reservation: RegisterOperand,
) -> Result<TempId, ExpansionError> {
    let success = asm.allocate()?;
    asm.emit_u(Kind::LUI, success.operand(), RAM_START_ADDRESS as i128);
    asm.emit_b(Kind::VirtualAssertLTE, success.operand(), address, 0);
    asm.emit_r(Kind::XOR, success.operand(), reservation, address);
    asm.emit_i(Kind::SLTIU, success.operand(), success.operand(), 1);
    Ok(success)
}

/// Width class of a sub-doubleword access.
///
/// One value selects every width-specific row of a sequence. `VirtualWindowMask`
/// and `VirtualShiftData` for `H`/`W` ignore the offset bits that natural
/// alignment clears, so they are sound only next to floor-addressed rows of the
/// same class: those rows' `AddressOffset` output leaves the bits in
/// `RamAddress`, which RAM RAF evaluation rejects (see R1CS constraint 0 in
/// jolt-claims).
#[derive(Clone, Copy)]
pub(in crate::expand) enum AccessWidth {
    Byte,
    Halfword,
    Word,
}

impl AccessWidth {
    fn load(self) -> Kind {
        match self {
            Self::Byte => Kind::VirtualLoadB,
            Self::Halfword => Kind::VirtualLoadH,
            Self::Word => Kind::VirtualLoadW,
        }
    }

    fn store(self) -> Kind {
        match self {
            Self::Byte => Kind::VirtualStoreB,
            Self::Halfword => Kind::VirtualStoreH,
            Self::Word => Kind::VirtualStoreW,
        }
    }

    fn window_mask(self) -> Kind {
        match self {
            Self::Byte => Kind::VirtualWindowMaskB,
            Self::Halfword => Kind::VirtualWindowMaskH,
            Self::Word => Kind::VirtualWindowMaskW,
        }
    }

    fn shift_data(self) -> Kind {
        match self {
            Self::Byte => Kind::VirtualShiftDataB,
            Self::Halfword => Kind::VirtualShiftDataH,
            Self::Word => Kind::VirtualShiftDataW,
        }
    }
}

/// The addressed lane of a floor-addressed access to `ea = base + offset`:
/// `doubleword` holds the doubleword containing `ea` and `lane` the byte mask of
/// the `width`-byte lane at `ea`.
#[derive(Clone, Copy)]
pub(in crate::expand) struct LaneWindow {
    width: AccessWidth,
    doubleword: RegisterOperand,
    lane: RegisterOperand,
    base: RegisterOperand,
    offset: i128,
}

impl LaneWindow {
    /// Emits `[VirtualLoad][VirtualWindowMask]`.
    ///
    /// The floor-addressed rows take the signed immediate: R1CS constraint 0
    /// forms `RamAddress` from the same `Imm` column as their lookup index, so
    /// the `u64`-wrapped form `format_i_imm` gives the window mask would put the
    /// address `2^64` away.
    pub(in crate::expand) fn open(
        asm: &mut ExpansionBuilder,
        width: AccessWidth,
        doubleword: RegisterOperand,
        lane: RegisterOperand,
        base: RegisterOperand,
        offset: i128,
    ) -> Self {
        asm.emit_ld(width.load(), doubleword, base, offset);
        asm.emit_i(width.window_mask(), lane, base, format_i_imm(offset));
        Self {
            width,
            doubleword,
            lane,
            base,
            offset,
        }
    }

    /// Emits the parallel extract of the lane into `destination`, sign- or
    /// zero-extended.
    pub(in crate::expand) fn extract(
        self,
        asm: &mut ExpansionBuilder,
        destination: RegisterOperand,
        signed: bool,
    ) {
        let extract = if signed {
            Kind::VirtualPextSigned
        } else {
            Kind::VirtualPext
        };
        asm.emit_r(extract, destination, self.doubleword, self.lane);
    }

    /// Emits `[ANDN][VirtualShiftData][ADD][VirtualStore]`: replaces the lane
    /// with the low `width` bytes of `value` and stores the doubleword back.
    /// Overwrites `lane`.
    ///
    /// The shifter reads the effective address as `rs2 + imm` with the base in
    /// its `rs2` slot, and its interleaved right operand must stay below
    /// `2^64`. The shifted bytes lie inside the lane `ANDN` cleared, so a plain
    /// `ADD` merges them without carries.
    pub(in crate::expand) fn merge(self, asm: &mut ExpansionBuilder, value: RegisterOperand) {
        let Self {
            width,
            doubleword,
            lane,
            base,
            offset,
        } = self;
        asm.emit_r(Kind::ANDN, doubleword, doubleword, lane);
        asm.emit_shift_data(width.shift_data(), lane, value, base, offset);
        asm.emit_r(Kind::ADD, doubleword, doubleword, lane);
        asm.emit_s(width.store(), base, doubleword, offset);
    }
}

/// Lowers `LB`/`LBU`/`LH`/`LHU`/`LW`/`LWU`: the floor-addressed `VirtualLoad`
/// row loads the doubleword containing `ea = rs1 + imm`, `VirtualWindowMask`
/// turns `ea` into the byte mask of the addressed lane, and a parallel-extract
/// lookup pulls the lane out of the doubleword.
pub(in crate::expand) fn expand_narrow_load(
    instruction: &SourceInstructionRow,
    width: AccessWidth,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let doubleword = asm.allocate()?;
    let mask = asm.allocate()?;

    LaneWindow::open(
        &mut asm,
        width,
        doubleword.operand(),
        mask.operand(),
        reg(rs1(instruction)?),
        instruction.operands.imm,
    )
    .extract(&mut asm, reg(rd(instruction)?), signed);
    asm.release_many([doubleword, mask]);

    asm.finalize()
}

/// Lowers an advice load with byte length 1, 2, 4, or 8.
///
/// `VirtualAdviceLoad` reads from the advice tape rather than RAM. Narrow
/// advice loads are signed loads, so the helper left-shifts the value into the
/// high bits and arithmetic-shifts it back to XLEN.
pub(in crate::expand) fn expand_advice_load(
    instruction: &SourceInstructionRow,
    byte_len: i128,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_j(
        Kind::VirtualAdviceLoad(VirtualAdviceLoad(())),
        reg(rd(instruction)?),
        byte_len,
    );
    if byte_len < 8 {
        let shift = 64 - byte_len * 8;
        asm.emit_i(
            Kind::SLLI,
            reg(rd(instruction)?),
            reg(rd(instruction)?),
            shift,
        );
        asm.emit_i(
            Kind::SRAI,
            reg(rd(instruction)?),
            reg(rd(instruction)?),
            shift,
        );
    }

    asm.finalize()
}

/// Lowers arithmetic/bitwise doubleword AMOs using the shared read-modify-write shape.
///
/// The old memory value is loaded, `op(old, rs2)` is stored back, and the old
/// value is copied to `rd`. Jolt traces are sequential, so the atomicity
/// contract reduces to preserving this single-instruction read/modify/write
/// order in the expanded bytecode.
pub(in crate::expand) fn expand_amo_d(
    instruction: &SourceInstructionRow,
    op: SourceInstructionKind,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rs2 = asm.allocate()?;
    let v_rd = asm.allocate()?;

    asm.emit_i(Kind::LD, v_rd.operand(), reg(rs1(instruction)?), 0);
    asm.emit_r(op, v_rs2.operand(), v_rd.operand(), reg(rs2(instruction)?));
    asm.emit_s(Kind::SD, reg(rs1(instruction)?), v_rs2.operand(), 0);
    asm.emit_i(Kind::ADDI, reg(rd(instruction)?), v_rd.operand(), 0);
    asm.release_many([v_rs2, v_rd]);

    asm.finalize()
}

/// Lowers signed or unsigned doubleword AMO min/max.
///
/// `compare_op` decides whether `rs2` should replace the old memory value.
/// The update is computed as `old + take_rs2 * (rs2 - old)`, so the same
/// arithmetic shape handles min and max once the comparison operands are
/// ordered appropriately.
pub(in crate::expand) fn expand_amo_minmax_d(
    instruction: &SourceInstructionRow,
    compare_op: SourceInstructionKind,
    min: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let (cmp_rs1, cmp_rs2): (RegisterOperand, RegisterOperand) = if min {
        (reg(rs2(instruction)?), v0.operand())
    } else {
        (v0.operand(), reg(rs2(instruction)?))
    };

    asm.emit_i(Kind::LD, v0.operand(), reg(rs1(instruction)?), 0);
    asm.emit_r(compare_op, v1.operand(), cmp_rs1, cmp_rs2);
    asm.emit_r(
        Kind::SUB,
        v2.operand(),
        reg(rs2(instruction)?),
        v0.operand(),
    );
    asm.emit_r(Kind::MUL, v2.operand(), v2.operand(), v1.operand());
    asm.emit_r(Kind::ADD, v1.operand(), v0.operand(), v2.operand());
    asm.emit_s(Kind::SD, reg(rs1(instruction)?), v1.operand(), 0);
    asm.emit_i(Kind::ADDI, reg(rd(instruction)?), v0.operand(), 0);
    asm.release_many([v0, v1, v2]);

    asm.finalize()
}

/// Lowers arithmetic/bitwise word AMOs as a read-modify-write of the word lane.
///
/// The old word is extracted sign-extended, `op(old, rs2)` supplies the new
/// word, whose low 32 bits are merged back and stored, and `rd` receives the old
/// word last because it may alias `rs1` or `rs2`. The floor-addressed rows
/// reject an `rs1` that is not word-aligned.
pub(in crate::expand) fn expand_amo_w(
    instruction: &SourceInstructionRow,
    op: SourceInstructionKind,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let doubleword = asm.allocate()?;
    let lane = asm.allocate()?;
    let old = asm.allocate()?;
    let new = asm.allocate()?;

    let window = LaneWindow::open(
        &mut asm,
        AccessWidth::Word,
        doubleword.operand(),
        lane.operand(),
        reg(rs1(instruction)?),
        0,
    );
    window.extract(&mut asm, old.operand(), true);
    asm.emit_r(op, new.operand(), old.operand(), reg(rs2(instruction)?));
    window.merge(&mut asm, new.operand());
    asm.emit_i(Kind::ADDI, reg(rd(instruction)?), old.operand(), 0);
    asm.release_many([doubleword, lane, old, new]);

    asm.finalize()
}

/// Lowers signed or unsigned word AMO min/max.
///
/// `compare_op` (`SLT` or `SLTU`) compares the sign-extended old word with the
/// sign-extended low word of `rs2`; sign extension preserves both the signed and
/// the unsigned order of 32-bit words. The new value is computed as
/// `old + take_rs2 * (rs2 - old)`, whose low word is the selected one, so the
/// same arithmetic handles min and max once the comparison operands are ordered
/// appropriately.
pub(in crate::expand) fn expand_amo_minmax_w(
    instruction: &SourceInstructionRow,
    compare_op: SourceInstructionKind,
    min: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let doubleword = asm.allocate()?;
    let lane = asm.allocate()?;
    let old = asm.allocate()?;
    let v_rs2 = asm.allocate()?;
    let take_rs2 = asm.allocate()?;

    let window = LaneWindow::open(
        &mut asm,
        AccessWidth::Word,
        doubleword.operand(),
        lane.operand(),
        reg(rs1(instruction)?),
        0,
    );
    window.extract(&mut asm, old.operand(), true);
    asm.emit_i(
        Kind::VirtualSignExtendWord(VirtualSignExtendWord(())),
        v_rs2.operand(),
        reg(rs2(instruction)?),
        0,
    );
    let (cmp_rs1, cmp_rs2) = if min {
        (v_rs2.operand(), old.operand())
    } else {
        (old.operand(), v_rs2.operand())
    };
    asm.emit_r(compare_op, take_rs2.operand(), cmp_rs1, cmp_rs2);
    asm.emit_r(
        Kind::SUB,
        v_rs2.operand(),
        reg(rs2(instruction)?),
        old.operand(),
    );
    asm.emit_r(
        Kind::MUL,
        v_rs2.operand(),
        v_rs2.operand(),
        take_rs2.operand(),
    );
    asm.emit_r(Kind::ADD, v_rs2.operand(), v_rs2.operand(), old.operand());
    window.merge(&mut asm, v_rs2.operand());
    asm.emit_i(Kind::ADDI, reg(rd(instruction)?), old.operand(), 0);
    asm.release_many([doubleword, lane, old, v_rs2, take_rs2]);

    asm.finalize()
}

/// Lowers a narrow store (`SB`/`SH`/`SW`) via a read-modify-write of the
/// containing doubleword: the addressed lane is cleared and replaced by the
/// low bytes of `rs2`.
pub(in crate::expand) fn expand_narrow_store(
    instruction: &SourceInstructionRow,
    width: AccessWidth,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let doubleword = asm.allocate()?;
    let lane = asm.allocate()?;

    LaneWindow::open(
        &mut asm,
        width,
        doubleword.operand(),
        lane.operand(),
        reg(rs1(instruction)?),
        instruction.operands.imm,
    )
    .merge(&mut asm, reg(rs2(instruction)?));
    asm.release_many([doubleword, lane]);

    asm.finalize()
}
