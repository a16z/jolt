//! The field-register read-RAF fold against the per-row stage values.
#![cfg(feature = "field-inline")]

use jolt_claims::protocols::field_inline::geometry::bytecode::{
    read_raf_folded_stage_values, read_raf_stage_values, FieldInlineBytecodeReadRafStageValueInputs,
};
use jolt_field::{Fr, Ring};
use jolt_riscv::JoltInstructionRow;
use jolt_riscv::{JoltInstructionKind, NormalizedOperands};

#[test]
fn folded_stage_values_match_weighted_row_values() {
    let row = |instruction_kind, rs1, rs2, rd| JoltInstructionRow {
        instruction_kind,
        operands: NormalizedOperands {
            rs1,
            rs2,
            rd,
            imm: 0,
        },
        ..JoltInstructionRow::default()
    };
    let bytecode = vec![
        row(JoltInstructionKind::FIELD_MUL, Some(1), Some(2), Some(3)),
        row(JoltInstructionKind::FIELD_MUL, Some(3), Some(3), Some(15)),
        row(JoltInstructionKind::FIELD_ADD, Some(0), Some(7), Some(1)),
        row(JoltInstructionKind::ADD, Some(1), Some(2), Some(3)),
        JoltInstructionRow::default(),
    ];
    let field = |count: usize, offset: u64| {
        (0..count)
            .map(|value| Fr::from_u64(value as u64 + offset))
            .collect::<Vec<_>>()
    };
    let read_write_point = field(4, 3);
    let val_evaluation_point = field(4, 7);
    let stage4_gammas = std::array::from_fn(|i| Fr::from_u64(i as u64 + 11));
    let stage5_gammas = std::array::from_fn(|i| Fr::from_u64(i as u64 + 19));
    let inputs = || FieldInlineBytecodeReadRafStageValueInputs {
        bytecode: &bytecode,
        field_register_read_write_point: &read_write_point,
        field_register_val_evaluation_point: &val_evaluation_point,
        stage4_gammas: &stage4_gammas,
        stage5_gammas: &stage5_gammas,
    };
    let address_eq = field(bytecode.len(), 29);

    let mut expected = [Fr::from_u64(0); 5];
    for (values, eq) in read_raf_stage_values(inputs()).iter().zip(&address_eq) {
        for (sum, value) in expected.iter_mut().zip(values) {
            *sum += *eq * *value;
        }
    }
    assert_ne!(
        expected[3],
        Fr::from_u64(0),
        "rows must carry field operands"
    );
    assert_eq!(
        read_raf_folded_stage_values(inputs(), &address_eq),
        expected
    );
}

/// The committed-field wire contract assigns every staged value a distinct η power.
#[cfg(feature = "akita")]
#[test]
fn committed_reduction_authenticates_all_eight_bytecode_wires() {
    use jolt_claims::protocols::jolt::{
        geometry::claim_reductions::bytecode::bytecode_val_stage_opening,
        relations::claim_reductions::bytecode::CyclePhase, BytecodeClaimReductionChallenge,
        PrecommittedReductionDimensions,
    };
    use jolt_claims::{challenge, opening, SymbolicSumcheck};

    let phase = CyclePhase::new(PrecommittedReductionDimensions::new(1, 1, true));
    let input = phase.input_expression::<Fr>();
    let eta = challenge(BytecodeClaimReductionChallenge::Eta);
    assert_eq!(input.terms.len(), 8);
    for wire in 0..8 {
        let term = eta.clone().pow(wire) * opening(bytecode_val_stage_opening(wire));
        assert!(
            input.terms.contains(&term.terms[0]),
            "missing η^{wire} coefficient for wire {wire}"
        );
    }
}
