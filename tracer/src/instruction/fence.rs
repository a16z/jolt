use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr, emulator::cpu::Cpu, instruction::format::format_fence::FormatFence,
};

use super::{RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = FENCE,
    mask   = 0x0000707f,
    match  = 0x0000000f,
    format = FormatFence,
    ram    = ()
);

impl FENCE {
    fn exec(&self, _: &mut Cpu, _: &mut <FENCE as RISCVInstruction>::RAMAccess) {
        // no-op
    }
}

impl RISCVTrace for FENCE {}

#[cfg(test)]
mod tests {
    use crate::instruction::Instruction;

    /// MISC-MEM carries `fence` (funct3 = 000) only. Zifencei's `fence.i`
    /// (funct3 = 001) and the reserved funct3 values are outside RV64IMAC, so
    /// `decode` must reject them with an error instead of handing the word to
    /// `FENCE::new`, whose `MASK` requires funct3 = 000.
    #[test]
    fn decode_rejects_misc_mem_funct3_other_than_fence() {
        for funct3 in 1..8 {
            let word = (funct3 << 12) | 0x0000_000f;
            let error = Instruction::decode(word, 0x1000, false).expect_err(
                "decode must reject a non-FENCE MISC-MEM encoding with an Err, not panic",
            );
            assert!(error.contains("MISC-MEM"), "funct3={funct3:03b}: {error}");
        }
    }

    /// `fence rw, rw` still decodes to [`FENCE`].
    #[test]
    fn decode_accepts_fence() {
        let decoded = Instruction::decode(0x0ff0_000f, 0x1000, false).expect("fence should decode");
        assert!(matches!(decoded, Instruction::FENCE(_)), "{decoded:?}");
    }
}
