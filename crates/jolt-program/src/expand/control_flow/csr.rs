//! Reads and writes of the machine-mode CSRs Jolt models.
//!
//! Jolt's hart is M-mode only (no S/U modes, no F/D/V, no interrupt sources),
//! so most `mstatus` fields are read-only. The CSRs live in reserved virtual
//! registers. Writes follow the privileged specification for that configuration.
//! Matching Sail also requires its `mcause` write policy to store raw values,
//! as described in `specs/m-mode-trap-csr-semantics.md`.
//!
//! | CSR      | Virtual register holds        | Write of `v`                                   |
//! |----------|-------------------------------|------------------------------------------------|
//! | mstatus  | `mstatus & (MIE \| MPIE)`     | `v & (MIE \| MPIE)`                            |
//! | mtvec    | `mtvec`                       | `v`, but a reserved MODE (`v[1] = 1`) keeps the old MODE |
//! | mepc     | `mepc`                        | `v & !1` (IALIGN = 16)                         |
//! | mcause, mtval, mscratch | the CSR        | `v`                                            |
//!
//! `mstatus.MPP` is read-only `M` (`0b11`) on an M-only hart: an `xPP` field
//! can hold only privilege modes up to `x` that are implemented. The virtual
//! register therefore stores only the writable bits, and every read ORs in
//! [`MSTATUS_READ_ONLY_ONES`]. The reset value 0 of the virtual register reads
//! as `0x1800`: MPP = M, MIE = 0.

use super::*;
use jolt_riscv::instructions::VirtualShiftRightBitmask;
use jolt_riscv::SourceInstructionKind as Kind;

/// `mstatus.MIE`, bit 3.
pub(super) const MSTATUS_MIE: i128 = 1 << 3;
/// `mstatus.MPIE`, bit 7.
pub(super) const MSTATUS_MPIE: i128 = 1 << 7;
/// The writable `mstatus` bits of an M-only hart: MIE and MPIE.
pub(super) const MSTATUS_WRITABLE: i128 = MSTATUS_MIE | MSTATUS_MPIE;
/// Read-only one bits of `mstatus`: MPP = M (`0b11` at bits 12:11).
pub(super) const MSTATUS_READ_ONLY_ONES: i128 = 0b11 << 11;

/// Emits `rd := csr`. `mstatus` reads add the read-only MPP bits; the other
/// CSRs are copied.
pub(super) fn emit_csr_read(
    asm: &mut ExpansionBuilder,
    csr: u16,
    virtual_reg: u8,
    rd: RegisterOperand,
) {
    if csr == CSR_MSTATUS {
        asm.emit_i(Kind::ORI, rd, reg(virtual_reg), MSTATUS_READ_ONLY_ONES);
    } else {
        asm.emit_i(Kind::ADDI, rd, reg(virtual_reg), 0);
    }
}

/// Emits the `CSRRW` body: `rd := csr` unless `rd = x0` (then the
/// CSR is not read), and `csr := legalize(csr, source)`.
///
/// `mtvec` consumes the source into a masked difference before reading, so an
/// alias needs no copy. Other aliased swaps preserve the source explicitly.
pub(super) fn emit_csr_swap(
    asm: &mut ExpansionBuilder,
    csr: u16,
    virtual_reg: u8,
    rd: u8,
    source: RegisterOperand,
) -> Result<(), ExpansionError> {
    if csr == CSR_MTVEC && source != reg(0) {
        let diff = asm.allocate()?;
        let mask = asm.allocate()?;
        // Xtvec_Ignore keeps the old MODE for reserved writes. The mask is
        // !0 for a legal MODE and !3 otherwise, excluding reserved MODE changes.
        asm.emit_r(Kind::XOR, diff.operand(), source, reg(virtual_reg));
        asm.emit_i(Kind::ANDI, mask.operand(), source, 2);
        asm.emit_i(
            Kind::VirtualShiftRightBitmask(VirtualShiftRightBitmask(())),
            mask.operand(),
            mask.operand(),
            0,
        );
        asm.emit_r(Kind::AND, diff.operand(), diff.operand(), mask.operand());
        asm.release(mask);
        if rd != 0 {
            emit_csr_read(asm, csr, virtual_reg, reg(rd));
        }
        asm.emit_r(
            Kind::XOR,
            reg(virtual_reg),
            reg(virtual_reg),
            diff.operand(),
        );
        asm.release(diff);
        return Ok(());
    }

    let preserved = if rd != 0 && source == reg(rd) {
        let temp = asm.allocate()?;
        asm.emit_i(Kind::ADDI, temp.operand(), source, 0);
        Some(temp)
    } else {
        None
    };
    let source = preserved.map_or(source, TempId::operand);
    if rd != 0 {
        emit_csr_read(asm, csr, virtual_reg, reg(rd));
    }
    match csr {
        CSR_MSTATUS => asm.emit_i(Kind::ANDI, reg(virtual_reg), source, MSTATUS_WRITABLE),
        CSR_MEPC => asm.emit_i(Kind::ANDI, reg(virtual_reg), source, format_i_imm(-2)),
        _ => asm.emit_i(Kind::ADDI, reg(virtual_reg), source, 0),
    }
    if let Some(temp) = preserved {
        asm.release(temp);
    }
    Ok(())
}

/// The operation shared by register and immediate bit updates.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CsrBitUpdate {
    /// `CSRRS` / `CSRRSI`: `csr := legalize(csr, csr | bits)`.
    Set,
    /// `CSRRC` / `CSRRCI`: `csr := legalize(csr, csr & !bits)`.
    Clear,
}

/// Emits a register bit update, suppressing writes for `rs1 = x0`.
///
/// A set first removes operand bits that cannot change the architectural CSR.
/// This also consumes an aliased `rs1` before the read overwrites it. For
/// `mtvec`, setting a reserved MODE preserves the old MODE, so only BASE bits
/// of that operand contribute. Clearing preserves every CSR's legal set.
pub(super) fn emit_csr_bit_update(
    asm: &mut ExpansionBuilder,
    csr: u16,
    virtual_reg: u8,
    rd: u8,
    bits: RegisterOperand,
    update: CsrBitUpdate,
) -> Result<(), ExpansionError> {
    if bits == reg(0) {
        if rd == 0 {
            asm.emit_i(Kind::ADDI, reg(0), reg(0), 0);
        } else {
            emit_csr_read(asm, csr, virtual_reg, reg(rd));
        }
        return Ok(());
    }

    let prepared = match (update, csr) {
        (CsrBitUpdate::Set, CSR_MSTATUS | CSR_MEPC | CSR_MTVEC) => {
            let temp = asm.allocate()?;
            match csr {
                CSR_MSTATUS => {
                    asm.emit_i(Kind::ANDI, temp.operand(), bits, MSTATUS_WRITABLE);
                }
                CSR_MEPC => asm.emit_i(Kind::ANDI, temp.operand(), bits, format_i_imm(-2)),
                _ => {
                    asm.emit_i(Kind::ANDI, temp.operand(), bits, 2);
                    asm.emit_i(
                        Kind::VirtualShiftRightBitmask(VirtualShiftRightBitmask(())),
                        temp.operand(),
                        temp.operand(),
                        0,
                    );
                    asm.emit_r(Kind::AND, temp.operand(), bits, temp.operand());
                }
            }
            Some(temp)
        }
        _ if rd != 0 && bits == reg(rd) => {
            let temp = asm.allocate()?;
            asm.emit_i(Kind::ADDI, temp.operand(), bits, 0);
            Some(temp)
        }
        _ => None,
    };
    let bits = prepared.map_or(bits, TempId::operand);
    if rd != 0 {
        emit_csr_read(asm, csr, virtual_reg, reg(rd));
    }
    let kind = match update {
        CsrBitUpdate::Set => Kind::OR,
        CsrBitUpdate::Clear => Kind::ANDN,
    };
    asm.emit_r(kind, reg(virtual_reg), reg(virtual_reg), bits);
    if let Some(temp) = prepared {
        asm.release(temp);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "test setup must fail on expansion errors"
    )]

    use super::*;
    use crate::expand::{expand_instruction, ExpansionAllocator};
    use jolt_riscv::{JoltInstructionRow, RV64IMAC_JOLT};

    fn assert_count(kind: Kind, csr: u16, rd: u8, rs1: u8, expected: usize) {
        let source = SourceInstruction::new(
            kind,
            SourceInstructionRow {
                address: 0x8000_0000,
                operands: NormalizedOperands {
                    rd: Some(rd),
                    rs1: Some(rs1),
                    rs2: None,
                    imm: i128::from(csr),
                },
                inline: None,
                is_compressed: false,
            },
        );
        let expanded = expand_instruction(&source, &mut ExpansionAllocator::new(), RV64IMAC_JOLT)
            .expect("system expansion");
        assert_eq!(
            expanded.len(),
            expected,
            "{kind:?} {csr:#x} rd={rd} rs1/uimm={rs1}"
        );
        for row in expanded.into_iter().map(JoltInstructionRow::from) {
            for register in [row.operands.rd, row.operands.rs1, row.operands.rs2]
                .into_iter()
                .flatten()
            {
                assert!(register < 48, "temporary outside v40–v47: {row:?}");
            }
        }
    }

    #[test]
    fn system_expansion_counts_match_spec_paths() {
        // Literal costs from specs/m-mode-trap-csr-semantics.md, in column
        // order mstatus, mepc, mtvec, raw. All three raw CSRs share each cost.
        let register_paths = [
            (Kind::CSRRW, 0, 0, [1, 1, 1, 1]),
            (Kind::CSRRW, 0, 5, [1, 1, 5, 1]),
            (Kind::CSRRW, 6, 0, [2, 2, 2, 2]),
            (Kind::CSRRW, 6, 5, [2, 2, 6, 2]),
            (Kind::CSRRW, 5, 5, [3, 3, 6, 3]),
            (Kind::CSRRS, 0, 0, [1, 1, 1, 1]),
            (Kind::CSRRS, 6, 0, [1, 1, 1, 1]),
            (Kind::CSRRS, 0, 5, [2, 2, 4, 1]),
            (Kind::CSRRS, 6, 5, [3, 3, 5, 2]),
            (Kind::CSRRS, 5, 5, [3, 3, 5, 3]),
            (Kind::CSRRC, 0, 0, [1, 1, 1, 1]),
            (Kind::CSRRC, 6, 0, [1, 1, 1, 1]),
            (Kind::CSRRC, 0, 5, [1, 1, 1, 1]),
            (Kind::CSRRC, 6, 5, [2, 2, 2, 2]),
            (Kind::CSRRC, 5, 5, [3, 3, 3, 3]),
        ];
        for (kind, rd, rs1, costs) in register_paths {
            for (csr, cost) in [CSR_MSTATUS, CSR_MEPC, CSR_MTVEC, 0x340, 0x342, 0x343]
                .into_iter()
                .zip([costs[0], costs[1], costs[2], costs[3], costs[3], costs[3]])
            {
                assert_count(kind, csr, rd, rs1, cost);
            }
        }

        // Each value class is listed explicitly, including effective no-ops.
        let immediate_paths: &[(Kind, u16, &[u8], [usize; 2])] = &[
            (
                Kind::CSRRWI,
                CSR_MSTATUS,
                &[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                ],
                [1, 2],
            ),
            (
                Kind::CSRRWI,
                CSR_MEPC,
                &[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                ],
                [1, 2],
            ),
            (
                Kind::CSRRWI,
                CSR_MTVEC,
                &[
                    0, 1, 2, 3, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29,
                ],
                [1, 2],
            ),
            (
                Kind::CSRRWI,
                CSR_MTVEC,
                &[6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31],
                [2, 3],
            ),
            (
                Kind::CSRRWI,
                0x340,
                &[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                ],
                [1, 2],
            ),
            (
                Kind::CSRRSI,
                CSR_MSTATUS,
                &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23],
                [1, 1],
            ),
            (
                Kind::CSRRSI,
                CSR_MSTATUS,
                &[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31],
                [1, 2],
            ),
            (Kind::CSRRSI, CSR_MEPC, &[0, 1], [1, 1]),
            (
                Kind::CSRRSI,
                CSR_MEPC,
                &[
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31,
                ],
                [1, 2],
            ),
            (Kind::CSRRSI, CSR_MTVEC, &[0, 2, 3], [1, 1]),
            (
                Kind::CSRRSI,
                CSR_MTVEC,
                &[
                    1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31,
                ],
                [1, 2],
            ),
            (Kind::CSRRSI, 0x340, &[0], [1, 1]),
            (
                Kind::CSRRSI,
                0x340,
                &[
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31,
                ],
                [1, 2],
            ),
            (
                Kind::CSRRCI,
                CSR_MSTATUS,
                &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23],
                [1, 1],
            ),
            (
                Kind::CSRRCI,
                CSR_MSTATUS,
                &[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31],
                [1, 2],
            ),
            (Kind::CSRRCI, CSR_MEPC, &[0, 1], [1, 1]),
            (
                Kind::CSRRCI,
                CSR_MEPC,
                &[
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31,
                ],
                [1, 2],
            ),
            (Kind::CSRRCI, CSR_MTVEC, &[0, 2], [1, 1]),
            (
                Kind::CSRRCI,
                CSR_MTVEC,
                &[
                    1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31,
                ],
                [1, 2],
            ),
            (Kind::CSRRCI, 0x340, &[0], [1, 1]),
            (
                Kind::CSRRCI,
                0x340,
                &[
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31,
                ],
                [1, 2],
            ),
        ];
        for &(kind, csr, values, costs) in immediate_paths {
            let csrs: &[u16] = if csr == 0x340 {
                &[0x340, 0x342, 0x343]
            } else {
                std::slice::from_ref(&csr)
            };
            for &csr in csrs {
                for &uimm in values {
                    for (rd, cost) in [0, 5].into_iter().zip(costs) {
                        assert_count(kind, csr, rd, uimm, cost);
                    }
                }
            }
        }
        assert_count(Kind::ECALL, 0, 0, 0, 6);
        assert_count(Kind::MRET, 0, 0, 0, 3);
    }
}
