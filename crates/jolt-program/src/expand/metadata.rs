use jolt_riscv::NormalizedInstruction;

pub fn set_sequence_metadata(
    instruction: &mut NormalizedInstruction,
    is_first_in_sequence: bool,
    virtual_sequence_remaining: Option<u16>,
) {
    instruction.is_first_in_sequence = is_first_in_sequence;
    instruction.virtual_sequence_remaining = virtual_sequence_remaining;
}
