use jolt_core::{field::JoltField, jolt::{instruction::JoltInstruction, vm::rv32i_vm::RV32I}};
use strum::IntoEnumIterator as _;

use crate::{constants::JoltParameterSet, modules::{AsModule, Module}, subtable::ZkLeanSubtable, util::{indent, ZkLeanReprField}, MleAst};

/// Wrapper around a JoltInstruction
// TODO: Make this generic over the instruction set
#[derive(Debug, Clone)]
pub struct ZkLeanInstruction<J> {
    instruction: RV32I,
    phantom: std::marker::PhantomData<J>,
}

impl<J> From<RV32I> for ZkLeanInstruction<J> {
    fn from(value: RV32I) -> Self {
        Self {
            instruction: value,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<J: JoltParameterSet> ZkLeanInstruction<J> {
    pub fn name(&self) -> String {
        let name = <&'static str>::from(&self.instruction);
        let word_size = J::WORD_SIZE;
        let c = J::C;
        let log_m = J::LOG_M;

        format!("{name}_{word_size}_{c}_{log_m}")
    }

    /// The number of field elements in the input to `combine_lookups`. See the doc comment for
    /// [`JoltInstruction::combine_lookups`] for more info.
    fn num_lookups<F: JoltField>(&self) -> usize {
        // We need one wire for each subtable evaluation, i.e., one wire per subtable, per chunk
        self.instruction
            .subtables::<F>(J::C, 1 << J::LOG_M)
            .iter()
            .flat_map(|(_, ixs)| ixs.iter())
            .count()
    }

    fn combine_lookups<F: ZkLeanReprField>(&self, reg_name: char) -> F {
        let reg_size = self.num_lookups::<F>();
        let reg = F::register(reg_name, reg_size);
        self.instruction.combine_lookups(&reg, J::C, 1 << J::LOG_M)
    }

    fn subtables<F: ZkLeanReprField>(&self) -> impl Iterator<Item = (ZkLeanSubtable<F, J>, usize)> {
        self.instruction
            .subtables(J::C, 1 << J::LOG_M)
            .into_iter()
            .flat_map(|(subtable, ixs)|
                ixs.iter()
                    .map(|ix| (ZkLeanSubtable::<F, J>::from(&subtable), ix))
                    .collect::<Vec<_>>()
            )
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        RV32I::iter().map(Self::from)
    }

    pub fn to_instruction_set(&self) -> RV32I {
        self.instruction
    }

    /// Pretty print an instruction as a ZkLean `ComposedLookupTable`.
    pub fn zklean_pretty_print<F: ZkLeanReprField>(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let name = self.name();
        let log_m = J::LOG_M;
        let c = J::C;
        let mle = self.combine_lookups::<F>('x').as_computation();
        let subtables = std::iter::once("#[ ".to_string())
            .chain(self.subtables::<F>().map(|(subtable, ix)|
                format!("({}, {ix})", subtable.name()))
                    .intersperse(", ".to_string())
            )
            .chain(std::iter::once(" ].toVector".to_string()))
            .fold(String::new(), |acc, s| format!("{acc}{s}"));

        f.write_fmt(format_args!(
                "{}def {name} [Field f] : ComposedLookupTable f {log_m} {c}\n",
                indent(indent_level),
        ))?;
        indent_level += 1;
        f.write_fmt(format_args!(
                "{}:= mkComposedLookupTable {subtables} (fun x => {mle})\n",
                indent(indent_level),
        ))?;

        Ok(())
    }
}

pub struct ZkLeanInstructions<J> {
    instructions: Vec<ZkLeanInstruction<J>>,
    phantom: std::marker::PhantomData<J>,
}

impl<J: JoltParameterSet> ZkLeanInstructions<J> {
    pub fn extract() -> Self {
        Self {
            instructions: ZkLeanInstruction::<J>::iter().collect(),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        indent_level: usize,
    ) -> std::io::Result<()> {
        for instruction in &self.instructions {
            instruction.zklean_pretty_print::<MleAst<2048>>(f, indent_level)?;
        }
        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![
            String::from("ZkLean"),
            String::from("Jolt.Subtables"),
        ]
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

    use proptest::{prelude::*, collection::vec};
    use strum::EnumCount as _;

    type RefField = ark_bn254::Fr;
    type TestField = crate::mle_ast::MleAst<2048>;
    type ParamSet = crate::constants::RV32IParameterSet;

    #[derive(Clone)]
    struct TestableInstruction<J: JoltParameterSet> {
        reference: RV32I,
        test: ZkLeanInstruction<J>,
    }

    impl<J: JoltParameterSet> std::fmt::Debug for TestableInstruction<J> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("{}", self.test.name()))
        }
    }

    impl<J: JoltParameterSet> TestableInstruction<J> {
        fn iter() -> impl Iterator<Item = Self> {
            RV32I::iter()
                .zip(ZkLeanInstruction::iter())
                .map(|(reference, test)| Self { reference, test })
        }

        fn reference_combine_lookups<R: JoltField>(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), self.test.num_lookups::<R>());

            self.reference.combine_lookups(inputs, J::C, J::M)
        }

        fn test_combine_lookups<R: JoltField, T: ZkLeanReprField>(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), self.test.num_lookups::<R>());

            let ast: T = self.test.combine_lookups('x');
            ast.evaluate(inputs)
        }
    }

    fn arb_instruction<J: JoltParameterSet>()
        -> impl Strategy<Value = TestableInstruction<J>>
    {
        (0..RV32I::COUNT)
            .prop_map(|n| TestableInstruction::iter().nth(n).unwrap())
    }

    fn arb_instruction_and_input<J: JoltParameterSet + Clone, R: JoltField>()
        -> impl Strategy<Value = (TestableInstruction<J>, Vec<R>)>
    {
        arb_instruction().prop_flat_map(|instr| {
            let input_len = instr.test.num_lookups::<R>();
            let inputs = vec(arb_field_elem::<R>(), input_len);

            (Just(instr), inputs)
        })
    }

    proptest! {
        #[test]
        fn combine_lookups(
            (instr, inputs) in arb_instruction_and_input::<ParamSet, RefField>(),
        ) {
            // NOTE: Omitting this causes index OOB errors when converting from `uXX`
            crate::util::initialize_fields();

            prop_assert_eq!(
                instr.test_combine_lookups::<_, TestField>(&inputs),
                instr.reference_combine_lookups(&inputs),
            );
        }
    }
}
