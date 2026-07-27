use serde::{Deserialize, Serialize};

use crate::emulator::cpu::Cpu;

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct ADDC {
    pub address: u64,
    pub operands: FormatR,
    pub prev_aux: u64,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}

impl RISCVInstruction for ADDC {
    const MASK: u32 = 0xfe00_707f;
    const MATCH: u32 = 0x0c00_005b;

    type Format = FormatR;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn source_kind(&self) -> jolt_riscv::SourceInstructionKind {
        jolt_riscv::SourceInstructionKind::Addc(jolt_riscv::instructions::Addc(()))
    }

    fn new(word: u32, address: u64, validate: bool, compressed: bool) -> Self {
        if validate {
            debug_assert_eq!(
                word & Self::MASK,
                Self::MATCH,
                "word: {:x}, mask: {:x}, word & mask: {:x}, match: {:x}",
                word,
                Self::MASK,
                word & Self::MASK,
                Self::MATCH
            );
        }
        Self {
            address,
            operands: FormatR::parse(word),
            prev_aux: 0,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: compressed,
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        Self {
            address: rand::RngCore::next_u64(rng),
            operands: FormatR::random(rng),
            prev_aux: rand::RngCore::next_u64(rng),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        let sum = (cpu.x[self.operands.rs1 as usize] as u64 as u128)
            + (cpu.x[self.operands.rs2 as usize] as u64 as u128)
            + self.prev_aux as u128;
        cpu.write_register(
            self.operands.rd as usize,
            cpu.sign_extend(sum as u64 as i64),
        );
    }

    fn aux_input_word(&self) -> u64 {
        self.prev_aux
    }
}

impl RISCVTrace for ADDC {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<super::Cycle>>) {
        let mut cycle = super::RISCVCycle {
            instruction: Self {
                prev_aux: cpu.last_lookup_high_word(),
                ..*self
            },
            register_state: Default::default(),
            ram_access: Default::default(),
        };
        cycle
            .instruction
            .operands()
            .capture_pre_execution_state(&mut cycle.register_state, cpu);
        cycle.instruction.execute(cpu, &mut cycle.ram_access);
        cycle
            .instruction
            .operands()
            .capture_post_execution_state(&mut cycle.register_state, cpu);
        cpu.set_last_lookup_high_word(super::trace_lookup_high_word(&cycle));
        if let Some(trace_vec) = trace {
            trace_vec.push(cycle.into());
        }
    }
}

impl From<super::SourceInstructionRow> for ADDC {
    fn from(row: super::SourceInstructionRow) -> Self {
        Self {
            address: row.address as u64,
            operands: row.operands.into(),
            prev_aux: 0,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: row.is_compressed,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};

    use super::{FormatR, RISCVTrace, ADDC};

    #[test]
    fn addc_treats_register_operands_as_unsigned_limbs() {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        cpu.x[1] = u64::MAX as i64;
        cpu.x[2] = 1;
        cpu.set_last_lookup_high_word(1);

        ADDC {
            operands: FormatR {
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            ..Default::default()
        }
        .trace(&mut cpu, None);

        assert_eq!(cpu.x[3] as u64, 1);
        assert_eq!(cpu.last_lookup_high_word(), 1);
    }
}
