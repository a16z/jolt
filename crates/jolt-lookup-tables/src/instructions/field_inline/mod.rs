//! Field-inline bridges whose x-register writes use range-check lookups.

pub mod advice_limb;
pub mod store_to_register;

#[cfg(test)]
mod test {
    use jolt_riscv::FieldInlineOp;
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    use tracer::emulator::cpu::Cpu;
    use tracer::instruction::field_inline::FIELD_LOAD_ACCUMULATE_FROM_REGISTER;
    use tracer::instruction::{RISCVInstruction, RISCVTrace};

    pub(super) fn values() -> impl Iterator<Item = u64> {
        let mut rng = StdRng::seed_from_u64(12345);
        [0, 1, 42, 1 << 63, u64::MAX]
            .into_iter()
            .chain((0..10_000).map(move |_| rng.next_u64()))
    }

    pub(super) fn initialize_cpu(cpu: &mut Cpu, value: u128) {
        let load = FIELD_LOAD_ACCUMULATE_FROM_REGISTER::new(
            FieldInlineOp::LoadAccumulateFromRegister.instruction_match() | (1 << 7) | (5 << 15),
            0x8000_0000,
            false,
            false,
        );
        // Initialize field register 1 through the tracer in either proof field.
        for word in [(value >> 64) as u64, value as u64] {
            cpu.write_register(5, word as i64);
            load.trace(cpu, None);
        }
        cpu.write_register(3, !(value as u64) as i64);
    }
}
