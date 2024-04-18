use crate::emulator::cpu::{Instruction, INSTRUCTIONS};

pub fn decode_raw(word: u32) -> Result<Instruction, ()> {
    match decode_and_get_instruction_index(word) {
        Ok(index) => Ok(INSTRUCTIONS[index].clone()),
        Err(()) => Err(()),
    }
}

fn decode_and_get_instruction_index(word: u32) -> Result<usize, ()> {
    for (i, inst) in INSTRUCTIONS.iter().enumerate() {
        if (word & inst.mask) == inst.data {
            return Ok(i);
        }
    }
    Err(())
}
