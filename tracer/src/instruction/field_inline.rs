#![expect(
    non_camel_case_types,
    reason = "Tracer concrete instruction names mirror generated Jolt instruction constants"
)]

#[cfg(not(feature = "fp128-field-inline"))]
use jolt_field::Fr;
#[cfg(feature = "fp128-field-inline")]
use jolt_field::Prime128OffsetA7F7;
use jolt_field::{CanonicalEncoding, Field};
use jolt_program::field_inline::{
    FieldEncodedValue, FieldInlineBridge, FieldInlineTraceData, FieldRegisterRead,
    FieldRegisterWrite,
};
use jolt_riscv::{FieldInlineOp, SourceInstructionKind};
use serde::{Deserialize, Serialize};

#[cfg(any(feature = "test-utils", test))]
use super::RISCVCycle;
use super::{
    format::{format_field_inline::FormatFieldInline, InstructionFormat},
    RAMAccess, RAMRead, RISCVInstruction, RISCVTrace,
};
use crate::emulator::cpu::Cpu;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FieldInlineCycleData {
    pub trace: Option<FieldInlineTraceData>,
    /// The word read by a memory-sourced load; `None` for every other op.
    pub ram_read: Option<RAMRead>,
}

impl From<FieldInlineCycleData> for RAMAccess {
    fn from(value: FieldInlineCycleData) -> Self {
        value.ram_read.map_or(Self::NoOp, Self::Read)
    }
}

macro_rules! field_instruction {
    ($name:ident, $op:expr, $source_kind:expr $(, { $($extra:item)* })?) => {
        #[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
        pub struct $name {
            pub address: u64,
            pub operands: FormatFieldInline,
            pub virtual_sequence_remaining: Option<u16>,
            pub is_first_in_sequence: bool,
            pub is_compressed: bool,
        }

        impl RISCVInstruction for $name {
            const MASK: u32 = $op.instruction_mask();
            const MATCH: u32 = $op.instruction_match();

            type Format = FormatFieldInline;
            type RAMAccess = FieldInlineCycleData;

            fn operands(&self) -> &Self::Format {
                &self.operands
            }

            fn source_kind(&self) -> SourceInstructionKind {
                $source_kind
            }

            fn new(word: u32, address: u64, _validate: bool, is_compressed: bool) -> Self {
                Self {
                    address,
                    operands: FormatFieldInline::parse(word),
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed,
                }
            }

            fn execute(&self, cpu: &mut Cpu, ram_access: &mut Self::RAMAccess) {
                *ram_access = execute_field_inline($op, self.operands, cpu);
            }

            $($($extra)*)?
        }

        impl RISCVTrace for $name {}

        impl From<super::SourceInstructionRow> for $name {
            fn from(row: super::SourceInstructionRow) -> Self {
                let mut operands = FormatFieldInline::from(row.operands);
                operands.op = Some($op);
                Self {
                    address: row.address as u64,
                    operands,
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed: row.is_compressed,
                }
            }
        }
    };
}

field_instruction!(
    FIELD_ADD,
    FieldInlineOp::Add,
    SourceInstructionKind::FIELD_ADD
);
field_instruction!(
    FIELD_SUB,
    FieldInlineOp::Sub,
    SourceInstructionKind::FIELD_SUB
);
field_instruction!(
    FIELD_MUL,
    FieldInlineOp::Mul,
    SourceInstructionKind::FIELD_MUL
);
field_instruction!(
    FIELD_INV,
    FieldInlineOp::Inv,
    SourceInstructionKind::FIELD_INV
);
field_instruction!(
    FIELD_ASSERT_EQ,
    FieldInlineOp::AssertEq,
    SourceInstructionKind::FIELD_ASSERT_EQ
);
field_instruction!(
    FIELD_LOAD_ACCUMULATE_FROM_REGISTER,
    FieldInlineOp::LoadAccumulateFromRegister,
    SourceInstructionKind::FIELD_LOAD_ACCUMULATE_FROM_REGISTER
);
field_instruction!(
    FIELD_ASSERT_ZERO,
    FieldInlineOp::AssertZero,
    SourceInstructionKind::FIELD_ASSERT_ZERO
);
field_instruction!(
    FIELD_LOAD_IMM,
    FieldInlineOp::LoadImm,
    SourceInstructionKind::FIELD_LOAD_IMM
);
field_instruction!(
    FIELD_LOAD_ACCUMULATE_FROM_MEMORY,
    FieldInlineOp::LoadAccumulateFromMemory,
    SourceInstructionKind::FIELD_LOAD_ACCUMULATE_FROM_MEMORY
);
field_instruction!(
    FIELD_ADVICE_LIMB,
    FieldInlineOp::AdviceLimb,
    SourceInstructionKind::FIELD_ADVICE_LIMB,
    {
        #[cfg(any(feature = "test-utils", test))]
        fn random_cycle(rng: &mut StdRng) -> RISCVCycle<Self> {
            use super::format::format_field_inline::RegisterStateFormatFieldInline;
            use common::constants::RISCV_REGISTER_COUNT;
            use jolt_riscv::FIELD_REGISTER_COUNT;
            use rand::{Rng, RngCore};

            let x_register = rng.gen_range(1..RISCV_REGISTER_COUNT);
            let field_register = rng.gen_range(0..FIELD_REGISTER_COUNT);
            let field_value = encode_field(ProofField::random(rng));
            let mut low = [0u8; 8];
            low.copy_from_slice(&field_value.bytes_le[..8]);
            let x_value = u64::from_le_bytes(low);
            let quotient_register = rng.gen_range(0..FIELD_REGISTER_COUNT);
            let pre_value = if quotient_register == field_register {
                field_value
            } else {
                encode_field(ProofField::random(rng))
            };
            let mut quotient = FieldEncodedValue::zero();
            quotient.bytes_le[..FieldEncodedValue::BYTE_LEN as usize - 8]
                .copy_from_slice(&field_value.bytes_le[8..]);
            let word = Self::MATCH
                | (u32::from(x_register) << 7)
                | (u32::from(field_register) << 15)
                | (u32::from(quotient_register) << 20);
            RISCVCycle {
                instruction: Self::new(word, rng.next_u64() & !3, false, false),
                register_state: RegisterStateFormatFieldInline {
                    rs1: None,
                    rd_pre: Some(!x_value),
                    rd_post: Some(x_value),
                },
                ram_access: FieldInlineCycleData {
                    trace: Some(FieldInlineTraceData {
                        op: Some(FieldInlineOp::AdviceLimb),
                        rs1: Some(FieldRegisterRead {
                            register: field_register,
                            value: field_value,
                        }),
                        rd: Some(FieldRegisterWrite {
                            register: quotient_register,
                            pre_value,
                            post_value: quotient,
                        }),
                        bridge: Some(FieldInlineBridge::AdviceLimb {
                            field_register,
                            field_value,
                            x_register,
                            x_value,
                        }),
                        ..Default::default()
                    }),
                    ram_read: None,
                },
            }
        }

        #[cfg(any(feature = "test-utils", test))]
        fn initialize_test_cpu(cycle: &RISCVCycle<Self>, cpu: &mut Cpu) {
            let trace = cycle
                .ram_access
                .trace
                .expect("field advice fixture payload");
            if let Some(write) = trace.rd {
                cpu.field_registers.write(write.register, write.pre_value);
            }
            for read in [trace.rs1, trace.rs2].into_iter().flatten() {
                cpu.field_registers.write(read.register, read.value);
            }
        }
    }
);

// The proof field the tracer executes over — the single selection point:
// `fp128-field-inline` (the akita chain) selects the fp128 akita field, the
// homomorphic (Dory) builds keep BN254 Fr. FieldValueEncoding::ACTIVE in
// jolt-program flips with the same feature, so a trace and its metadata
// always agree on the encoding. Everything below is generic over F; no other
// line names a concrete field.
#[cfg(not(feature = "fp128-field-inline"))]
type ProofField = Fr;
#[cfg(feature = "fp128-field-inline")]
type ProofField = Prime128OffsetA7F7;

fn execute_field_inline(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineCycleData {
    execute_over::<ProofField>(op, operands, cpu)
}

fn execute_over<F: Field + CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineCycleData {
    let pure = |trace| FieldInlineCycleData {
        trace: Some(trace),
        ram_read: None,
    };
    match op {
        FieldInlineOp::Add => pure(execute_binary::<F>(op, operands, cpu, |left, right| {
            left + right
        })),
        FieldInlineOp::Sub => pure(execute_binary::<F>(op, operands, cpu, |left, right| {
            left - right
        })),
        FieldInlineOp::Mul => {
            let mut trace = execute_binary::<F>(op, operands, cpu, |left, right| left * right);
            trace.product = trace.rd.map(|write| write.post_value);
            pure(trace)
        }
        FieldInlineOp::Inv => pure(execute_inverse::<F>(op, operands, cpu)),
        FieldInlineOp::AssertEq => pure(execute_assert_eq::<F>(op, operands, cpu)),
        FieldInlineOp::LoadAccumulateFromRegister => pure(
            execute_load_accumulate_from_register::<F>(op, operands, cpu),
        ),
        FieldInlineOp::AssertZero => pure(execute_assert_zero::<F>(op, operands, cpu)),
        FieldInlineOp::LoadImm => pure(execute_load_imm(op, operands, cpu)),
        FieldInlineOp::LoadAccumulateFromMemory => {
            execute_load_accumulate_from_memory::<F>(op, operands, cpu)
        }
        FieldInlineOp::AdviceLimb => pure(execute_advice_limb::<F>(op, operands, cpu)),
    }
}

/// Honest advice generation chooses the canonical low limb and quotient.
/// Constraints permit other choices; the guest validates the full readout.
fn execute_advice_limb<F: Field + CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let field_register = operands.rs1.unwrap_or(0);
    let quotient_register = operands.rs2.unwrap_or(0);
    let x_register = operands.rd.unwrap_or(0);
    // x0 discards the write constrained to equal the advised limb.
    assert!(
        x_register != 0,
        "FIELD_ADVICE_LIMB to x0 at pc 0x{:x}: x0 discards the write, store to a real register",
        cpu.read_pc(),
    );
    let field_value = cpu.field_registers.read(field_register);
    let canonical = encode_field(decode_field::<F>(field_value));
    let mut low = [0u8; 8];
    low.copy_from_slice(&canonical.bytes_le[..8]);
    let x_value = u64::from_le_bytes(low);
    let mut quotient = FieldEncodedValue::zero();
    quotient.bytes_le[..FieldEncodedValue::BYTE_LEN as usize - 8]
        .copy_from_slice(&canonical.bytes_le[8..]);
    cpu.write_register(x_register as usize, x_value as i64);
    let pre_value = cpu.field_registers.read(quotient_register);
    cpu.field_registers.write(quotient_register, quotient);
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead {
            register: field_register,
            value: field_value,
        }),
        rd: Some(FieldRegisterWrite {
            register: quotient_register,
            pre_value,
            post_value: quotient,
        }),
        bridge: Some(FieldInlineBridge::AdviceLimb {
            field_register,
            field_value,
            x_register,
            x_value,
        }),
        ..Default::default()
    }
}

/// `field_rd = field_rd · 2^64 + mem[x_rs1 + offset]`, the word also written to the
/// scratch x-register `rd` so the cycle is an ordinary `LD` to the RV64 rows.
fn execute_load_accumulate_from_memory<F: Field + CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineCycleData {
    let x_base = operands.rs1.unwrap_or(0);
    let x_register = operands.rd.unwrap_or(0);
    let field_register = operands.rs2.unwrap_or(0);
    // The scratch register must take the write: x0 would drop it and the
    // load rows equate the rd write with the loaded word.
    assert!(
        x_register != 0,
        "field-inline LoadAccumulateFromMemory to x0 at pc 0x{:x}: the scratch register must be a real register",
        cpu.read_pc(),
    );
    let address = (cpu.read_register(x_base) as u64).wrapping_add(operands.imm as u64);
    let (word, ram_read) = cpu
        .get_mut_mmu()
        .load_doubleword(address)
        .unwrap_or_else(|_| panic!("MMU load error at pc 0x{:x}", cpu.read_pc()));
    cpu.write_register(x_register as usize, word as i64);
    let pre_value = cpu.field_registers.read(field_register);
    let value = accumulate_word::<F>(pre_value, word);
    cpu.field_registers.write(field_register, value);
    FieldInlineCycleData {
        trace: Some(FieldInlineTraceData {
            op: Some(op),
            rs1: Some(FieldRegisterRead {
                register: field_register,
                value: pre_value,
            }),
            rd: Some(FieldRegisterWrite {
                register: field_register,
                pre_value,
                post_value: value,
            }),
            bridge: Some(FieldInlineBridge::LoadAccumulateFromMemory {
                x_base,
                x_register,
                word,
                field_value: value,
            }),
            ..Default::default()
        }),
        ram_read: Some(ram_read),
    }
}

fn execute_binary<F: CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
    f: impl FnOnce(F, F) -> F,
) -> FieldInlineTraceData {
    let rs1_register = operands.rs1.unwrap_or(0);
    let rs2_register = operands.rs2.unwrap_or(0);
    let rd_register = operands.rd.unwrap_or(0);
    let rs1_value = cpu.field_registers.read(rs1_register);
    let rs2_value = cpu.field_registers.read(rs2_register);
    let pre_value = cpu.field_registers.read(rd_register);
    let post_value = encode_field(f(decode_field(rs1_value), decode_field(rs2_value)));
    cpu.field_registers.write(rd_register, post_value);
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead {
            register: rs1_register,
            value: rs1_value,
        }),
        rs2: Some(FieldRegisterRead {
            register: rs2_register,
            value: rs2_value,
        }),
        rd: Some(FieldRegisterWrite {
            register: rd_register,
            pre_value,
            post_value,
        }),
        ..Default::default()
    }
}

fn execute_inverse<F: Field + CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let rs1_register = operands.rs1.unwrap_or(0);
    let rd_register = operands.rd.unwrap_or(0);
    let rs1_value = cpu.field_registers.read(rs1_register);
    let pre_value = cpu.field_registers.read(rd_register);
    // inv(0) fails closed: the R1CS row `IsFieldInv * (FieldInvProduct - 1) = 0`
    // demands `rs1 * rd == 1`, which is unsatisfiable for rs1 = 0, so a trace
    // containing FIELD_INV(0) can never be proven. Trapping here surfaces the
    // guest bug at trace time instead of as a downstream sumcheck failure; the
    // guest-side API is responsible for guarding zero before emitting FIELD_INV.
    let inverse = decode_field::<F>(rs1_value).inverse().unwrap_or_else(|| {
        panic!(
            "FIELD_INV of zero at pc 0x{:x} (field register {}): the inverse constraint is \
             unsatisfiable for a zero operand; guard the guest-side inverse",
            cpu.read_pc(),
            rs1_register,
        )
    });
    let post_value = encode_field(inverse);
    cpu.field_registers.write(rd_register, post_value);
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead {
            register: rs1_register,
            value: rs1_value,
        }),
        rd: Some(FieldRegisterWrite {
            register: rd_register,
            pre_value,
            post_value,
        }),
        inv_product: Some(encode_field(decode_field::<F>(rs1_value) * inverse)),
        ..Default::default()
    }
}

fn execute_assert_eq<F: CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let rs1_register = operands.rs1.unwrap_or(0);
    let rs2_register = operands.rs2.unwrap_or(0);
    let rs1_value = cpu.field_registers.read(rs1_register);
    let rs2_value = cpu.field_registers.read(rs2_register);
    assert_eq!(
        decode_field::<F>(rs1_value),
        decode_field::<F>(rs2_value),
        "field-inline assert_eq failed"
    );
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead {
            register: rs1_register,
            value: rs1_value,
        }),
        rs2: Some(FieldRegisterRead {
            register: rs2_register,
            value: rs2_value,
        }),
        ..Default::default()
    }
}

fn execute_load_accumulate_from_register<F: Field + CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let x_register = operands.rs1.unwrap_or(0);
    let rd_register = operands.rd.unwrap_or(0);
    let x_value = cpu.read_register(x_register) as u64;
    let pre_value = cpu.field_registers.read(rd_register);
    let field_value = accumulate_word::<F>(pre_value, x_value);
    cpu.field_registers.write(rd_register, field_value);
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead {
            register: rd_register,
            value: pre_value,
        }),
        rd: Some(FieldRegisterWrite {
            register: rd_register,
            pre_value,
            post_value: field_value,
        }),
        bridge: Some(FieldInlineBridge::LoadAccumulateFromRegister {
            x_register,
            x_value,
            field_value,
        }),
        ..Default::default()
    }
}

fn execute_assert_zero<F: Field + CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let register = operands.rs1.unwrap_or(0);
    let value = cpu.field_registers.read(register);
    assert_eq!(
        decode_field::<F>(value),
        F::from_u64(0),
        "FIELD_ASSERT_ZERO of nonzero field register {register} at pc 0x{:x}",
        cpu.read_pc(),
    );
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead { register, value }),
        ..Default::default()
    }
}

fn execute_load_imm(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let rd_register = operands.rd.unwrap_or(0);
    // Decoded FIELD_LOAD_IMM immediates are zero-extended 12-bit values (0..=4095);
    // anything else can only arrive through synthetic instruction construction and
    // indicates a caller bug, so fail loudly instead of loading zero.
    let value = u64::try_from(operands.imm).map_or_else(
        |_| {
            panic!(
                "FIELD_LOAD_IMM with out-of-range immediate {} at pc 0x{:x}: decoded \
                 field-inline immediates are zero-extended 12-bit values",
                operands.imm,
                cpu.read_pc(),
            )
        },
        FieldEncodedValue::from_u64,
    );
    let pre_value = cpu.field_registers.read(rd_register);
    cpu.field_registers.write(rd_register, value);
    FieldInlineTraceData {
        op: Some(op),
        rd: Some(FieldRegisterWrite {
            register: rd_register,
            pre_value,
            post_value: value,
        }),
        ..Default::default()
    }
}

fn accumulate_word<F: Field + CanonicalEncoding>(
    previous: FieldEncodedValue,
    word: u64,
) -> FieldEncodedValue {
    encode_field(decode_field::<F>(previous) * F::from_u128(1u128 << 64) + F::from_u64(word))
}

fn decode_field<F: CanonicalEncoding>(value: FieldEncodedValue) -> F {
    F::from_bytes_le_reduced(&value.bytes_le)
}

fn encode_field<F: CanonicalEncoding>(value: F) -> FieldEncodedValue {
    // A field wider than the fixed 32-byte register buffer cannot ride
    // FieldEncodedValue; reject it at monomorphization, not per encode.
    const { assert!(F::NUM_BYTES <= FieldEncodedValue::BYTE_LEN as usize) }
    let mut encoded = FieldEncodedValue::zero();
    // Narrower fields occupy the low NUM_BYTES; the rest of the buffer stays
    // zero (FieldValueEncoding::byte_len tags the valid width).
    value.to_bytes_le(&mut encoded.bytes_le[..F::NUM_BYTES]);
    encoded
}

#[cfg(test)]
mod tests {
    use common::constants::RISCV_REGISTER_COUNT;
    use jolt_field::{CanonicalBytes, Ring};
    use jolt_riscv::FIELD_REGISTER_COUNT;
    use rand::SeedableRng;

    use super::*;
    use crate::emulator::terminal::DummyTerminal;
    use crate::instruction::Cycle;

    #[test]
    fn randomized_advice_cycles_have_canonical_replayable_field_state() {
        let mut rng = StdRng::seed_from_u64(12345);
        let mut saw_alias = false;
        let mut saw_wide_source = false;
        for _ in 0..512 {
            let cycle = FIELD_ADVICE_LIMB::random_cycle(&mut rng);
            let operands = cycle.instruction.operands();
            assert_eq!(operands.op, Some(FieldInlineOp::AdviceLimb));
            assert!((1..RISCV_REGISTER_COUNT).contains(&operands.rd.unwrap()));
            let trace = cycle.ram_access.trace.unwrap();
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
                cycle.register_state.rd_pre.unwrap() as i64,
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

    /// Encode/decode roundtrip over the build's ProofField, including values
    /// above 2^64 (multi-limb) and the buffer-width contract a narrower field
    /// relies on: bytes past NUM_BYTES stay zero, so downstream full-buffer
    /// reduced decodes (jolt-witness) see the same element.
    #[test]
    fn proof_field_roundtrips_through_the_register_encoding() {
        let wide = ProofField::from_u128((1u128 << 64) + 3);
        let values = [
            ProofField::from_u64(0),
            ProofField::from_u64(1),
            ProofField::from_u64(u64::MAX),
            wide,
            ProofField::from_u64(0) - ProofField::from_u64(1),
            wide.inverse().unwrap(),
        ];
        for value in values {
            let encoded = encode_field(value);
            assert!(encoded.bytes_le[ProofField::NUM_BYTES..]
                .iter()
                .all(|byte| *byte == 0));
            assert_eq!(decode_field::<ProofField>(encoded), value);
        }
    }
}
