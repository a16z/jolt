//! The complete Jolt instruction set registry.
//!
//! [`JoltInstructionSet`] collects all instruction implementations into an
//! array-indexed registry for O(1) opcode dispatch.

use crate::opcodes;
use crate::traits::Instruction;

/// Registry of all Jolt instructions, indexed by opcode for fast dispatch.
///
/// Instructions are stored in a flat array where the index equals the opcode,
/// enabling O(1) lookup without hashing.
#[derive(Default)]
pub struct JoltInstructionSet {
    instructions: Vec<Box<dyn Instruction>>,
}

impl JoltInstructionSet {
    /// Creates a new instruction set with all RV64IMAC and virtual instructions registered.
    pub fn new() -> Self {
        use crate::rv::arithmetic::*;
        use crate::rv::arithmetic_w::*;
        use crate::rv::branch::*;
        use crate::rv::compare::*;
        use crate::rv::jump::*;
        use crate::rv::load::*;
        use crate::rv::logic::*;
        use crate::rv::shift::*;
        use crate::rv::shift_w::*;
        use crate::rv::store::*;
        use crate::rv::system::*;
        use crate::virtual_::advice::*;
        use crate::virtual_::arithmetic::*;
        use crate::virtual_::assert::*;
        use crate::virtual_::bitwise::*;
        use crate::virtual_::byte::*;
        use crate::virtual_::division::*;
        use crate::virtual_::extension::*;
        use crate::virtual_::shift::*;
        use crate::virtual_::xor_rotate::*;

        let all: Vec<Box<dyn Instruction>> = vec![
            // RV64I arithmetic (0-3)
            Box::new(Add),
            Box::new(Sub),
            Box::new(Lui),
            Box::new(Auipc),
            // RV64M multiply/divide (4-11)
            Box::new(Mul),
            Box::new(MulH),
            Box::new(MulHSU),
            Box::new(MulHU),
            Box::new(Div),
            Box::new(DivU),
            Box::new(Rem),
            Box::new(RemU),
            // RV64I arithmetic W-suffix (12-13)
            Box::new(AddW),
            Box::new(SubW),
            // RV64M W-suffix (14-18)
            Box::new(MulW),
            Box::new(DivW),
            Box::new(DivUW),
            Box::new(RemW),
            Box::new(RemUW),
            // RV64I logic (19-24)
            Box::new(And),
            Box::new(Or),
            Box::new(Xor),
            Box::new(AndI),
            Box::new(OrI),
            Box::new(XorI),
            // RV64I shifts (25-30)
            Box::new(Sll),
            Box::new(Srl),
            Box::new(Sra),
            Box::new(SllI),
            Box::new(SrlI),
            Box::new(SraI),
            // RV64I shifts W-suffix (31-36)
            Box::new(SllW),
            Box::new(SrlW),
            Box::new(SraW),
            Box::new(SllIW),
            Box::new(SrlIW),
            Box::new(SraIW),
            // RV64I compare (37-40)
            Box::new(Slt),
            Box::new(SltU),
            Box::new(SltI),
            Box::new(SltIU),
            // RV64I branch (41-46)
            Box::new(Beq),
            Box::new(Bne),
            Box::new(Blt),
            Box::new(Bge),
            Box::new(BltU),
            Box::new(BgeU),
            // RV64I load (47-53)
            Box::new(Lb),
            Box::new(Lbu),
            Box::new(Lh),
            Box::new(Lhu),
            Box::new(Lw),
            Box::new(Lwu),
            Box::new(Ld),
            // RV64I store (54-57)
            Box::new(Sb),
            Box::new(Sh),
            Box::new(Sw),
            Box::new(Sd),
            // RV64I system (58-61)
            Box::new(Ecall),
            Box::new(Ebreak),
            Box::new(Fence),
            Box::new(Noop),
            // RV64I immediate aliases (62-63)
            Box::new(Addi),
            Box::new(AddiW),
            // RV64I jump (64-65)
            Box::new(Jal),
            Box::new(Jalr),
            // Zbb extension (66)
            Box::new(Andn),
            // Virtual arithmetic (67-74)
            Box::new(AssertEq),
            Box::new(AssertLte),
            Box::new(Pow2),
            Box::new(MovSign),
            Box::new(Pow2I),
            Box::new(Pow2W),
            Box::new(Pow2IW),
            Box::new(MulI),
            // Virtual assert (75-79)
            Box::new(AssertValidDiv0),
            Box::new(AssertValidUnsignedRemainder),
            Box::new(AssertMulUNoOverflow),
            Box::new(AssertWordAlignment),
            Box::new(AssertHalfwordAlignment),
            // Virtual shift (80-87)
            Box::new(VirtualSrl),
            Box::new(VirtualSrli),
            Box::new(VirtualSra),
            Box::new(VirtualSrai),
            Box::new(VirtualShiftRightBitmask),
            Box::new(VirtualShiftRightBitmaski),
            Box::new(VirtualRotri),
            Box::new(VirtualRotriw),
            // Virtual division (88-89)
            Box::new(VirtualChangeDivisor),
            Box::new(VirtualChangeDivisorW),
            // Virtual extension (90-91)
            Box::new(VirtualSignExtendWord),
            Box::new(VirtualZeroExtendWord),
            // Virtual XOR-rotate (92-99)
            Box::new(VirtualXorRot32),
            Box::new(VirtualXorRot24),
            Box::new(VirtualXorRot16),
            Box::new(VirtualXorRot63),
            Box::new(VirtualXorRotW16),
            Box::new(VirtualXorRotW12),
            Box::new(VirtualXorRotW8),
            Box::new(VirtualXorRotW7),
            // Virtual byte (100)
            Box::new(VirtualRev8W),
            // Virtual advice/IO (101-104)
            Box::new(VirtualAdvice),
            Box::new(VirtualAdviceLen),
            Box::new(VirtualAdviceLoad),
            Box::new(VirtualHostIO),
        ];

        debug_assert_eq!(all.len(), opcodes::COUNT as usize);
        let mut sorted: Vec<(u32, Box<dyn Instruction>)> =
            all.into_iter().map(|i| (i.opcode(), i)).collect();
        sorted.sort_by_key(|(op, _)| *op);
        for (i, (op, _)) in sorted.iter().enumerate() {
            debug_assert_eq!(*op as usize, i, "expected opcode {i} but got {op}");
        }
        let instructions = sorted.into_iter().map(|(_, i)| i).collect();

        Self { instructions }
    }

    /// Look up an instruction by opcode. Returns `None` if the opcode is out of range.
    #[inline]
    pub fn instruction(&self, opcode: u32) -> Option<&dyn Instruction> {
        self.instructions.get(opcode as usize).map(|b| b.as_ref())
    }

    /// Iterate over all registered instructions in opcode order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &dyn Instruction> {
        self.instructions.iter().map(|b| b.as_ref())
    }

    /// Total number of registered instructions.
    #[inline]
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Returns `true` if no instructions are registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_opcodes_covered() {
        let set = JoltInstructionSet::new();
        assert_eq!(set.len(), opcodes::COUNT as usize);

        for (i, instr) in set.iter().enumerate() {
            assert_eq!(
                instr.opcode() as usize,
                i,
                "instruction {} has opcode {} but is at index {}",
                instr.name(),
                instr.opcode(),
                i
            );
        }
    }

    #[test]
    fn lookup_by_opcode() {
        let set = JoltInstructionSet::new();
        let add = set.instruction(opcodes::ADD).unwrap();
        assert_eq!(add.name(), "ADD");
        assert_eq!(add.execute(3, 5), 8);
    }

    #[test]
    fn out_of_range_returns_none() {
        let set = JoltInstructionSet::new();
        assert!(set.instruction(opcodes::COUNT).is_none());
        assert!(set.instruction(u32::MAX).is_none());
    }

    #[test]
    fn unique_names() {
        let set = JoltInstructionSet::new();
        let mut names: Vec<&str> = set.iter().map(|i| i.name()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate instruction names found");
    }
}
