//! End-to-end: SDK instruction-word encoders produce bytes that the
//! tracer decodes, executes, and emits `FieldRegEvent` for. These events
//! then feed `jolt_witness::replay_field_regs`, which produces the
//! per-cycle snapshots consumed by the FR Twist materializers.
//!
//! Validates the SDK ↔ tracer ↔ FR Twist interface end-to-end.

use jolt_inlines_bn254_fr::{encode_fadd, encode_field_mov};
use tracer::emulator::cpu::Cpu;
use tracer::emulator::terminal::DummyTerminal;
use tracer::instruction::{Cycle, Instruction};

#[test]
fn encoded_fieldmov_fieldadd_sequence_executes_cleanly() {
    // Guest-level ISA: load two integer register values into FR slots,
    // then add them.
    //
    //   FieldMov f1, x5     (x5 = 10 → f1 = 10)
    //   FieldMov f2, x6     (x6 = 20 → f2 = 20)
    //   FieldAdd f3, f1, f2 (f3 = 30)
    let words = [
        encode_field_mov(/* frd = */ 1, /* rs1 = */ 5),
        encode_field_mov(2, 6),
        encode_fadd(3, 1, 2),
    ];

    let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
    cpu.x[5] = 10;
    cpu.x[6] = 20;

    let mut trace: Vec<Cycle> = Vec::new();
    for (i, &word) in words.iter().enumerate() {
        let address = 0x1000 + (i as u64 * 4);
        let instr =
            Instruction::decode(word, address, false).expect("tracer should decode SDK encoding");
        instr.trace(&mut cpu, Some(&mut trace));
    }

    // Three cycles emitted, three FieldRegEvents.
    assert_eq!(trace.len(), 3, "one Cycle per instruction");
    assert_eq!(
        cpu.field_reg_events.len(),
        3,
        "one FieldRegEvent per FR instruction"
    );

    // FieldMov writes into FR slot 1 (first event) and FR slot 2 (second).
    assert_eq!(cpu.field_reg_events[0].slot, 1);
    assert_eq!(cpu.field_reg_events[0].new, [10, 0, 0, 0]);
    assert_eq!(cpu.field_reg_events[1].slot, 2);
    assert_eq!(cpu.field_reg_events[1].new, [20, 0, 0, 0]);

    // FieldAdd writes into slot 3 with value 30.
    assert_eq!(cpu.field_reg_events[2].slot, 3);
    assert_eq!(cpu.field_reg_events[2].new, [30, 0, 0, 0]);

    // CPU state reflects the writes.
    assert_eq!(cpu.field_regs[1], [10, 0, 0, 0]);
    assert_eq!(cpu.field_regs[2], [20, 0, 0, 0]);
    assert_eq!(cpu.field_regs[3], [30, 0, 0, 0]);
}

#[test]
fn decoded_instruction_round_trips_through_sdk() {
    // Encoding any instruction and decoding it must produce a matching
    // Instruction variant with the correct operands.
    let word = encode_fadd(5, 3, 1);
    let instr = Instruction::decode(word, 0, false).unwrap();
    match instr {
        Instruction::FieldOp(op) => {
            assert_eq!(op.operands.rd, 5);
            assert_eq!(op.operands.rs1, 3);
            assert_eq!(op.operands.rs2, 1);
            assert_eq!(op.funct3, jolt_inlines_bn254_fr::FUNCT3_FADD as u8);
        }
        other => panic!("expected Instruction::FieldOp, got {other:?}"),
    }
}

#[test]
fn field_mov_and_sll_decoders_match_sdk_encoders() {
    use jolt_inlines_bn254_fr::{encode_field_sll128, encode_field_sll192, encode_field_sll64};

    let mov = Instruction::decode(encode_field_mov(2, 7), 0, false).unwrap();
    assert!(matches!(mov, Instruction::FieldMov(_)), "got {mov:?}");

    let s64 = Instruction::decode(encode_field_sll64(3, 8), 0, false).unwrap();
    assert!(matches!(s64, Instruction::FieldSLL64(_)), "got {s64:?}");

    let s128 = Instruction::decode(encode_field_sll128(4, 9), 0, false).unwrap();
    assert!(matches!(s128, Instruction::FieldSLL128(_)), "got {s128:?}");

    let s192 = Instruction::decode(encode_field_sll192(5, 10), 0, false).unwrap();
    assert!(matches!(s192, Instruction::FieldSLL192(_)), "got {s192:?}");
}
