use jolt_core::zkvm::instruction::{
    InstructionFlags as _, InstructionLookup as _, InterleavedBitsMarker as _,
};
use strum::IntoEnumIterator as _;
use tracer::instruction::RV32IMInstruction;

use crate::{
    constants::JoltParameterSet,
    modules::{AsModule, Module},
    util::{indent, ZkLeanReprField},
    MleAst,
};

/// Represents how an instructions operands should be combined into a single vector.
#[derive(Debug, Clone, Copy)]
pub enum OperandInterleaving {
    /// Indicates that the operands should be concatenated:
    ///     rs1 || rs2
    Concatenated,
    /// Indicates that the operands should be interleaved:
    ///     rs1[0] || rs2[0] || rs1[1] || rs2[1] || ...
    Interleaved,
}

impl OperandInterleaving {
    /// Extract the operand interleaving for an instruction
    fn instruction_interleaving(instr: &RV32IMInstruction) -> Self {
        if instr.circuit_flags().is_interleaved_operands() {
            Self::Interleaved
        } else {
            Self::Concatenated
        }
    }
}

impl std::fmt::Display for OperandInterleaving {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Interleaved => write!(f, "Interleaved"),
            Self::Concatenated => write!(f, "Concatenated"),
        }
    }
}

/// Wrapper around a JoltInstruction
// TODO: Make this generic over the instruction set
#[derive(Debug, Clone)]
pub struct ZkLeanInstruction<J> {
    instruction: tracer::instruction::RV32IMInstruction,
    interleaving: OperandInterleaving,
    phantom: std::marker::PhantomData<J>,
}

impl<J> From<RV32IMInstruction> for ZkLeanInstruction<J> {
    fn from(value: RV32IMInstruction) -> Self {
        let interleaving = OperandInterleaving::instruction_interleaving(&value);

        Self {
            instruction: value,
            interleaving,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<J: JoltParameterSet> ZkLeanInstruction<J> {
    pub fn name(&self) -> String {
        let name = <&'static str>::from(&self.instruction);
        let word_size = J::WORD_SIZE;

        format!("{name}_{word_size}")
    }

    pub fn evaluate_mle<F: ZkLeanReprField>(&self, reg_name: char) -> F {
        let num_variables = 2 * J::WORD_SIZE;
        let reg = F::register(reg_name, num_variables);

        self.instruction
            .lookup_table()
            .expect(format!("{} is not an instruction with an MLE", self.name()).as_str())
            .evaluate_mle::<F>(&reg)
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        RV32IMInstruction::iter().filter_map(|instr| match instr {
            RV32IMInstruction::NoOp | RV32IMInstruction::UNIMPL
                // Virtual instruction sequences
                | RV32IMInstruction::DIV(_)
                | RV32IMInstruction::DIVU(_)
                | RV32IMInstruction::LB(_)
                | RV32IMInstruction::LBU(_)
                | RV32IMInstruction::LH(_)
                | RV32IMInstruction::LHU(_)
                | RV32IMInstruction::MULH(_)
                | RV32IMInstruction::MULHSU(_)
                | RV32IMInstruction::REM(_)
                | RV32IMInstruction::REMU(_)
                | RV32IMInstruction::SB(_)
                | RV32IMInstruction::SH(_)
                | RV32IMInstruction::SLL(_)
                | RV32IMInstruction::SLLI(_)
                | RV32IMInstruction::SRA(_)
                | RV32IMInstruction::SRAI(_)
                | RV32IMInstruction::SRL(_)
                | RV32IMInstruction::SRLI(_)
                | RV32IMInstruction::INLINE(_)

                // RV64I
                | RV32IMInstruction::ADDIW(_)
                | RV32IMInstruction::SLLIW(_)
                | RV32IMInstruction::SRLIW(_)
                | RV32IMInstruction::SRAIW(_)
                | RV32IMInstruction::ADDW(_)
                | RV32IMInstruction::SUBW(_)
                | RV32IMInstruction::SLLW(_)
                | RV32IMInstruction::SRLW(_)
                | RV32IMInstruction::SRAW(_)
                | RV32IMInstruction::LWU(_)

                // RV64M
                | RV32IMInstruction::DIVUW(_)
                | RV32IMInstruction::DIVW(_)
                | RV32IMInstruction::MULW(_)
                | RV32IMInstruction::REMUW(_)
                | RV32IMInstruction::REMW(_)

                // RV32A
                | RV32IMInstruction::LRW(_)
                | RV32IMInstruction::SCW(_)
                | RV32IMInstruction::AMOSWAPW(_)
                | RV32IMInstruction::AMOADDW(_)
                | RV32IMInstruction::AMOANDW(_)
                | RV32IMInstruction::AMOORW(_)
                | RV32IMInstruction::AMOXORW(_)
                | RV32IMInstruction::AMOMINW(_)
                | RV32IMInstruction::AMOMAXW(_)
                | RV32IMInstruction::AMOMINUW(_)
                | RV32IMInstruction::AMOMAXUW(_)

                // RV64A
                | RV32IMInstruction::LRD(_)
                | RV32IMInstruction::SCD(_)
                | RV32IMInstruction::AMOSWAPD(_)
                | RV32IMInstruction::AMOADDD(_)
                | RV32IMInstruction::AMOANDD(_)
                | RV32IMInstruction::AMOORD(_)
                | RV32IMInstruction::AMOXORD(_)
                | RV32IMInstruction::AMOMIND(_)
                | RV32IMInstruction::AMOMAXD(_)
                | RV32IMInstruction::AMOMINUD(_)
                | RV32IMInstruction::AMOMAXUD(_)

                // XXX Instructions with lookup tables but no MLE (???)
                // TODO: Find a better way to filter these out.
                | RV32IMInstruction::FENCE(_)
                | RV32IMInstruction::LW(_)
                | RV32IMInstruction::ECALL(_)
                | RV32IMInstruction::SW(_)

                // Instructions with no lookup table
                // TODO: Find a better way to filter these out.
                | RV32IMInstruction::LD(_)
                | RV32IMInstruction::SD(_)

                // XXX Temporarily disabled. Too many nodes.
                // See https://gitlab-ext.galois.com/jb4/jolt-fork/-/issues/14
                | RV32IMInstruction::BEQ(_)
                | RV32IMInstruction::BNE(_)
                | RV32IMInstruction::BGE(_)
                | RV32IMInstruction::BGEU(_)
                | RV32IMInstruction::BLT(_)
                | RV32IMInstruction::BLTU(_)
                | RV32IMInstruction::SLT(_)
                | RV32IMInstruction::SLTI(_)
                | RV32IMInstruction::SLTIU(_)
                | RV32IMInstruction::SLTU(_)
                | RV32IMInstruction::VirtualAssertLTE(_)
                | RV32IMInstruction::VirtualAssertEQ(_)
                | RV32IMInstruction::VirtualAssertValidSignedRemainder(_)
                | RV32IMInstruction::VirtualAssertValidUnsignedRemainder(_)
                => None,
            _ => Some(Self::from(instr)),
        })
    }

    pub fn to_instruction_set(&self) -> RV32IMInstruction {
        self.instruction.clone()
    }

    /// Pretty print an instruction as a ZkLean `ComposedLookupTable`.
    pub fn zklean_pretty_print<F: ZkLeanReprField>(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let name = self.name();
        let interleaving = self.interleaving;
        let num_variables = 2 * J::WORD_SIZE;
        let mle = self.evaluate_mle::<F>('x').as_computation();

        f.write_fmt(format_args!(
            "{}def {name} [Field f] : Instruction f {num_variables} :=\n",
            indent(indent_level),
        ))?;
        indent_level += 1;
        f.write_fmt(format_args!(
            "{}instructionFromMLE {interleaving} (fun x => {mle})\n",
            indent(indent_level),
        ))?;

        Ok(())
    }
}

pub struct ZkLeanInstructions<J> {
    instructions: Vec<ZkLeanInstruction<J>>,
}

impl<J: JoltParameterSet> ZkLeanInstructions<J> {
    pub fn extract() -> Self {
        Self {
            instructions: ZkLeanInstruction::<J>::iter().collect(),
        }
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        indent_level: usize,
    ) -> std::io::Result<()> {
        for instruction in &self.instructions {
            instruction.zklean_pretty_print::<MleAst<5400>>(f, indent_level)?;
        }
        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![String::from("ZkLean"), String::from("Jolt.Subtables")]
    }
}

impl<J: JoltParameterSet> AsModule for ZkLeanInstructions<J> {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents: Vec<u8> = vec![];
        self.zklean_pretty_print(&mut contents, 0)?;

        Ok(Module {
            name: String::from("Instructions"),
            imports: self.zklean_imports(),
            contents,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::arb_field_elem;

    use jolt_core::field::JoltField;

    use proptest::{collection::vec, prelude::*};

    type RefField = ark_bn254::Fr;
    type TestField = crate::mle_ast::MleAst<5400>;
    type ParamSet = crate::constants::RV32IParameterSet;

    #[derive(Clone)]
    struct TestableInstruction<J: JoltParameterSet> {
        reference: RV32IMInstruction,
        test: ZkLeanInstruction<J>,
    }

    impl<J: JoltParameterSet> std::fmt::Debug for TestableInstruction<J> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("{}", self.test.name()))
        }
    }

    impl<J: JoltParameterSet> TestableInstruction<J> {
        fn iter() -> impl Iterator<Item = Self> {
            ZkLeanInstruction::iter().map(|instr| Self {
                reference: instr.instruction.clone(),
                test: instr,
            })
        }

        fn evaluate_reference_mle<R: JoltField>(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), 2 * J::WORD_SIZE);

            self.reference.lookup_table().unwrap().evaluate_mle(inputs)
        }

        fn evaluate_test_mle<R: JoltField, T: ZkLeanReprField>(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), 2 * J::WORD_SIZE);

            let ast: T = self.test.evaluate_mle('x');
            ast.evaluate(inputs)
        }
    }

    fn arb_instruction<J: JoltParameterSet>() -> impl Strategy<Value = TestableInstruction<J>> {
        let num_instrs = TestableInstruction::<J>::iter().count();

        (0..num_instrs).prop_map(|n| TestableInstruction::iter().nth(n).unwrap())
    }

    fn arb_instruction_and_input<J: JoltParameterSet + Clone, R: JoltField>(
    ) -> impl Strategy<Value = (TestableInstruction<J>, Vec<R>)> {
        arb_instruction().prop_flat_map(|instr| {
            let input_len = 2 * J::WORD_SIZE;
            let inputs = vec(arb_field_elem::<R>(), input_len);

            (Just(instr), inputs)
        })
    }

    proptest! {
        #[test]
        fn evaluate_mle(
            (instr, inputs) in arb_instruction_and_input::<ParamSet, RefField>(),
        ) {
            prop_assert_eq!(
                instr.evaluate_test_mle::<_, TestField>(&inputs),
                instr.evaluate_reference_mle(&inputs),
            );
        }
    }
}
