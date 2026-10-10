use jolt_program::field_inline::FieldEncodedValue;
use jolt_riscv::FieldInlineOp;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
    instruction::{
        format::format_field_inline::FormatFieldInline,
        registers::field_inline::RegisterStateFieldInline, RISCVInstruction, RISCVTrace,
    },
};

use super::{decode_field, encode_field, ProofField};

#[cfg(any(feature = "test-utils", test))]
use crate::instruction::{Cycle, RISCVCycle};
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;

declare_riscv_instr!(
    name   = FIELD_ADVICE_LIMB,
    mask   = FieldInlineOp::AdviceLimb.instruction_mask(),
    match  = FieldInlineOp::AdviceLimb.instruction_match(),
    format = FormatFieldInline,
    registers = RegisterStateFieldInline,
    ram    = (),
    {
        #[cfg(any(feature = "test-utils", test))]
        fn random_cycle(rng: &mut StdRng) -> RISCVCycle<Self> {
            use crate::emulator::terminal::DummyTerminal;
            use common::constants::RISCV_REGISTER_COUNT;
            use jolt_field::Field;
            use jolt_riscv::FIELD_REGISTER_COUNT;
            use rand::{Rng, RngCore};

            let x_register = rng.gen_range(1..RISCV_REGISTER_COUNT);
            let field_register = rng.gen_range(0..FIELD_REGISTER_COUNT);
            let quotient_register = rng.gen_range(0..FIELD_REGISTER_COUNT);
            let word = Self::MATCH
                | (u32::from(x_register) << 7)
                | (u32::from(field_register) << 15)
                | (u32::from(quotient_register) << 20);
            let instruction = Self::new(word, rng.next_u64() & !3, false, false);
            let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
            cpu.write_register(usize::from(x_register), rng.next_u64() as i64);
            cpu.field_registers.write(field_register, encode_field(ProofField::random(rng)));
            if quotient_register != field_register {
                cpu.field_registers.write(quotient_register, encode_field(ProofField::random(rng)));
            }
            let mut trace = Vec::with_capacity(1);
            instruction.trace(&mut cpu, Some(&mut trace));
            let Some(Cycle::FIELD_ADVICE_LIMB(cycle)) = trace.pop() else {
                panic!("limb advice emits one cycle of its own kind");
            };
            cycle
        }

        #[cfg(any(feature = "test-utils", test))]
        fn initialize_test_cpu(cycle: &RISCVCycle<Self>, cpu: &mut Cpu) {
            let trace = cycle.register_state.to_field_inline_trace(FieldInlineOp::AdviceLimb);
            if let Some(write) = trace.rd {
                cpu.field_registers.write(write.register, write.pre_value);
            }
            for read in [trace.rs1, trace.rs2].into_iter().flatten() {
                cpu.field_registers.write(read.register, read.value);
            }
        }
    }
);

impl FIELD_ADVICE_LIMB {
    /// Honest advice generation chooses the canonical low limb and quotient.
    /// Constraints permit other choices; the guest validates the full readout.
    fn exec(&self, cpu: &mut Cpu, _: &mut <Self as RISCVInstruction>::RAMAccess) {
        let field_register = self.operands.rs1.unwrap_or(0);
        let quotient_register = self.operands.rs2.unwrap_or(0);
        let x_register = self.operands.rd.unwrap_or(0);
        // x0 discards the write constrained to equal the advised limb.
        assert!(
            x_register != 0,
            "FIELD_ADVICE_LIMB to x0 at pc 0x{:x}: x0 discards the write, store to a real register",
            cpu.read_pc(),
        );
        let field_value = cpu.field_registers.read(field_register);
        let canonical = encode_field(decode_field::<ProofField>(field_value));
        let mut low = [0u8; 8];
        low.copy_from_slice(&canonical.bytes_le[..8]);
        let x_value = u64::from_le_bytes(low);
        let mut quotient = FieldEncodedValue::zero();
        quotient.bytes_le[..FieldEncodedValue::BYTE_LEN as usize - 8]
            .copy_from_slice(&canonical.bytes_le[8..]);
        cpu.write_register(x_register as usize, x_value as i64);
        cpu.field_registers.write(quotient_register, quotient);
    }
}

impl RISCVTrace for FIELD_ADVICE_LIMB {}

#[cfg(test)]
mod tests {
    use common::constants::RISCV_REGISTER_COUNT;
    use jolt_field::{CanonicalBytes, CanonicalEncoding};
    use jolt_riscv::FIELD_REGISTER_COUNT;
    use rand::SeedableRng;

    use super::*;
    use crate::emulator::terminal::DummyTerminal;
    use crate::instruction::registers::InstructionRegisterState;

    #[test]
    fn randomized_advice_cycles_have_canonical_replayable_field_state() {
        let mut rng = StdRng::seed_from_u64(12345);
        let mut saw_alias = false;
        let mut saw_wide_source = false;
        for _ in 0..512 {
            let cycle = FIELD_ADVICE_LIMB::random_cycle(&mut rng);
            let operands = cycle.instruction.operands();
            assert!((1..RISCV_REGISTER_COUNT).contains(&operands.rd.unwrap()));
            let trace = cycle
                .register_state
                .to_field_inline_trace(FieldInlineOp::AdviceLimb);
            assert_eq!(trace.op, Some(FieldInlineOp::AdviceLimb));
            let source = trace.rs1.unwrap();
            assert_eq!(operands.rs1, Some(source.register));
            assert!(source.register < FIELD_REGISTER_COUNT);
            let source_value =
                ProofField::from_bytes_le_checked(&source.value.bytes_le[..ProofField::NUM_BYTES])
                    .unwrap();
            assert!(source.value.bytes_le[ProofField::NUM_BYTES..]
                .iter()
                .all(|byte| *byte == 0));
            saw_wide_source |= source_value.to_u64_checked().is_none();

            let limb = cycle.register_state.rd_values().unwrap().1;
            let quotient = trace.rd.unwrap().post_value;
            let mut recomposed = FieldEncodedValue::zero();
            recomposed.bytes_le[..8].copy_from_slice(&limb.to_le_bytes());
            recomposed.bytes_le[8..]
                .copy_from_slice(&quotient.bytes_le[..FieldEncodedValue::BYTE_LEN as usize - 8]);
            assert_eq!(recomposed, source.value);
            assert!(
                quotient.bytes_le[FieldEncodedValue::BYTE_LEN as usize - 8..]
                    .iter()
                    .all(|byte| *byte == 0)
            );

            if let Some(write) = trace.rd {
                assert_eq!(operands.rs2, Some(write.register));
                assert!(write.register < FIELD_REGISTER_COUNT);
                if write.register == source.register {
                    saw_alias = true;
                    assert_eq!(write.pre_value, source.value);
                }
            }

            let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
            cpu.write_register(
                usize::from(operands.rd.unwrap()),
                cycle.register_state.rd_values().unwrap().0 as i64,
            );
            FIELD_ADVICE_LIMB::initialize_test_cpu(&cycle, &mut cpu);
            assert_eq!(cpu.field_registers.read(source.register), source.value);
            if let Some(write) = trace.rd {
                assert_eq!(cpu.field_registers.read(write.register), write.pre_value);
            }
            let mut replay = Vec::new();
            cycle.instruction.trace(&mut cpu, Some(&mut replay));
            assert_eq!(replay.len(), 1);
            let expected: Cycle = cycle.into();
            assert_eq!(replay[0], expected);
        }
        assert!(
            saw_alias,
            "advice fixtures must exercise source/quotient aliasing"
        );
        assert!(
            saw_wide_source,
            "advice fixtures must exercise full-width sources"
        );
    }
}
