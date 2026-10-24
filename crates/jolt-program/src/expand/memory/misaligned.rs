use jolt_riscv::SourceInstructionKind as Kind;

use super::*;
use crate::expand::variant::{DualAccess, DualWidth};

/// What a dual-expanded source instruction does with the bytes at its
/// effective address.
#[derive(Clone, Copy)]
enum DualOperation {
    /// Writes them to `rd`, sign- or zero-extended from the access width
    /// (moot for doublewords).
    Load { signed: bool },
    /// Replaces them with the low bytes of `rs2`.
    Store,
}

/// The misaligned sequence of a top-level dual load or store, given the head
/// row of its aligned expansion; `None` for an instruction expanded once.
#[expect(
    clippy::wildcard_enum_match_arm,
    reason = "every other source kind has a single expansion"
)]
pub(in crate::expand) fn expand_misaligned(
    instruction: &SourceInstruction,
    aligned_head: Option<&JoltInstructionRow>,
) -> Result<Option<(DualAccess, ExpandedInstructionSequence)>, ExpansionError> {
    let operation = match instruction.kind() {
        Kind::LH | Kind::LW | Kind::LD => DualOperation::Load { signed: true },
        Kind::LHU | Kind::LWU => DualOperation::Load { signed: false },
        Kind::SH | Kind::SW | Kind::SD => DualOperation::Store,
        _ => return Ok(None),
    };
    let access = aligned_head.and_then(DualAccess::from_aligned_head).ok_or(
        ExpansionError::MalformedInstruction(
            "aligned expansion does not open with a natural-width access",
        ),
    )?;
    let row = instruction.row();
    let sequence = match operation {
        DualOperation::Load { signed } => expand_misaligned_load(row, access, signed),
        DualOperation::Store => expand_misaligned_store(row, access),
    }?;
    Ok(Some((access, sequence)))
}

/// Signed immediate of the access's last byte, which addresses `hi`.
fn last_byte_offset(access: DualAccess) -> i128 {
    access.offset + i128::from(access.width.bytes() - 1)
}

/// Byte-granular floor loads of the doublewords holding the first and the
/// last accessed byte (one doubleword unless the access crosses a boundary),
/// and the mask `!0 << 8·(ea mod 8)` that splits them at `ea`.
///
/// The floor rows take signed immediates, the mask the `u64`-wrapped one (see
/// `LaneWindow::open`).
fn open_doublewords(
    asm: &mut ExpansionBuilder,
    access: DualAccess,
    lo: TempId,
    hi: TempId,
    mask: TempId,
) {
    let base = reg(access.base);
    asm.emit_ld(Kind::VirtualLoadB, lo.operand(), base, access.offset);
    asm.emit_ld(
        Kind::VirtualLoadB,
        hi.operand(),
        base,
        last_byte_offset(access),
    );
    asm.emit_i(
        Kind::VirtualWindowMaskD,
        mask.operand(),
        base,
        format_i_imm(access.offset),
    );
}

/// Lowers a misaligned `LH`/`LHU`/`LW`/`LWU`/`LD`.
///
/// `Pext(lo, mask)` moves `lo`'s bytes from `ea` on to the bottom and
/// `FunnelHigh(hi, mask)` moves `hi`'s bytes below `ea mod 8` to the top, so
/// their (carry-free) sum holds the eight bytes from `ea` on. When the access
/// does not cross, its bytes all come from the `Pext` half; at offset 0 the
/// funnel half is 0.
fn expand_misaligned_load(
    instruction: &SourceInstructionRow,
    access: DualAccess,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let lo = asm.allocate()?;
    let hi = asm.allocate()?;
    let mask = asm.allocate()?;
    let rd = reg(rd(instruction)?);

    open_doublewords(&mut asm, access, lo, hi, mask);
    asm.emit_r(
        Kind::VirtualPext,
        lo.operand(),
        lo.operand(),
        mask.operand(),
    );
    asm.emit_r(
        Kind::VirtualFunnelHigh,
        hi.operand(),
        hi.operand(),
        mask.operand(),
    );
    match access.width {
        DualWidth::Doubleword => asm.emit_r(Kind::ADD, rd, lo.operand(), hi.operand()),
        DualWidth::Word => {
            asm.emit_r(Kind::ADD, lo.operand(), lo.operand(), hi.operand());
            let extend = if signed {
                Kind::VIRTUAL_SIGN_EXTEND_WORD
            } else {
                Kind::VIRTUAL_ZERO_EXTEND_WORD
            };
            asm.emit_i(extend, rd, lo.operand(), 0);
        }
        DualWidth::Halfword => {
            asm.emit_r(Kind::ADD, lo.operand(), lo.operand(), hi.operand());
            asm.emit_i(Kind::VirtualWindowMaskH, mask.operand(), reg(0), 0);
            let extract = if signed {
                Kind::VirtualPextSigned
            } else {
                Kind::VirtualPext
            };
            asm.emit_r(extract, rd, lo.operand(), mask.operand());
        }
    }
    asm.release_many([lo, hi, mask]);

    asm.finalize()
}

/// Lowers a misaligned `SH`/`SW`/`SD` as read-modify-writes of `lo` and `hi`.
///
/// With `scale = 2^(8·(ea mod 8))` (the negated mask), `MUL` by `scale` places
/// a value's low bytes from `ea mod 8` up in `lo`'s frame and `MULHU` by
/// `scale` places the bytes that spill past the doubleword in `hi`'s. `hi` is
/// written first: when the access does not cross, `hi` is `lo`'s doubleword
/// and nothing spills, so it writes back the loaded value before `lo`'s merge
/// lands.
fn expand_misaligned_store(
    instruction: &SourceInstructionRow,
    access: DualAccess,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let lo = asm.allocate()?;
    let hi = asm.allocate()?;
    let mask = asm.allocate()?;
    let scale = asm.allocate()?;
    let term = asm.allocate()?;
    let value = reg(rs2(instruction)?);
    let base = reg(access.base);

    let last = last_byte_offset(access);

    open_doublewords(&mut asm, access, lo, hi, mask);
    asm.emit_r(Kind::SUB, scale.operand(), reg(0), mask.operand());
    match access.width {
        DualWidth::Doubleword => {
            asm.emit_r(Kind::AND, hi.operand(), hi.operand(), mask.operand());
            asm.emit_r(Kind::MULHU, term.operand(), value, scale.operand());
            asm.emit_r(Kind::ADD, hi.operand(), hi.operand(), term.operand());
            asm.emit_s(Kind::VirtualStoreB, base, hi.operand(), last);
            asm.emit_r(Kind::ANDN, lo.operand(), lo.operand(), mask.operand());
            asm.emit_r(Kind::MUL, term.operand(), value, scale.operand());
        }
        DualWidth::Halfword | DualWidth::Word => {
            // `mask` becomes the lane mask at offset 0 and `data` the value's
            // lane bytes, both shifted into each doubleword's frame by `scale`.
            let lane = mask;
            let data = asm.allocate()?;
            let lane_mask = if matches!(access.width, DualWidth::Halfword) {
                Kind::VirtualWindowMaskH
            } else {
                Kind::VirtualWindowMaskW
            };
            asm.emit_i(lane_mask, lane.operand(), reg(0), 0);
            asm.emit_r(Kind::AND, data.operand(), value, lane.operand());
            asm.emit_r(Kind::MULHU, term.operand(), lane.operand(), scale.operand());
            asm.emit_r(Kind::ANDN, hi.operand(), hi.operand(), term.operand());
            asm.emit_r(Kind::MULHU, term.operand(), data.operand(), scale.operand());
            asm.emit_r(Kind::ADD, hi.operand(), hi.operand(), term.operand());
            asm.emit_s(Kind::VirtualStoreB, base, hi.operand(), last);
            asm.emit_r(Kind::MUL, term.operand(), lane.operand(), scale.operand());
            asm.emit_r(Kind::ANDN, lo.operand(), lo.operand(), term.operand());
            asm.emit_r(Kind::MUL, term.operand(), data.operand(), scale.operand());
            asm.release(data);
        }
    }
    asm.emit_r(Kind::ADD, lo.operand(), lo.operand(), term.operand());
    asm.emit_s(Kind::VirtualStoreB, base, lo.operand(), access.offset);
    asm.release_many([lo, hi, mask, scale, term]);

    asm.finalize()
}
