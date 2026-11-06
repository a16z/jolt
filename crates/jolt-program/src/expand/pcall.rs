use super::{ExpandedInstructionSequence, ExpansionError, InlineExpansionBuilder, InlineOperands};
use common::constants::RAM_START_ADDRESS;
use jolt_riscv::instructions::{PcallCall, PcallRead, PcallReturn, PcallWrite};
use jolt_riscv::JoltInstructionKind as Kind;

pub(super) fn build_sequence(
    mut builder: InlineExpansionBuilder,
    operands: InlineOperands,
    absorb: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    debug_assert!(operands.rs3 == 0 && (absorb || operands.rs2 == 0));
    builder.emit_pcall(
        Kind::PcallCall(PcallCall(())),
        operands.rs1,
        RAM_START_ADDRESS,
    );
    if absorb {
        for i in 0..17 {
            builder.emit_pcall(Kind::PcallRead(PcallRead(())), operands.rs2, 8 * i);
        }
    }
    for i in 0..25 {
        builder.emit_pcall(Kind::PcallWrite(PcallWrite(())), operands.rs1, 8 * i);
    }
    builder.emit_pcall(
        Kind::PcallReturn(PcallReturn(())),
        if absorb { operands.rs2 } else { operands.rs1 },
        RAM_START_ADDRESS,
    );
    builder.finalize()
}
