//! Per-cycle R1CS witness extraction from [`CycleRow`] data.
//!
//! Produces the witness vector for one cycle, matching the variable
//! layout in [`jolt_r1cs::constraints::rv64`].

use jolt_field::Field;
use jolt_instructions::flags::{CircuitFlags, InstructionFlags, NUM_CIRCUIT_FLAGS};
use jolt_r1cs::constraints::rv64::*;
use jolt_witness::derived::{
    limbs_to_field, FieldRegEvent, FIELD_OP_FUNCT3_FADD, FIELD_OP_FUNCT3_FINV,
    FIELD_OP_FUNCT3_FMUL, FIELD_OP_FUNCT3_FSUB,
};

use crate::bytecode::BytecodePreprocessing;
use crate::CycleRow;

/// Per-cycle R1CS witness matching `jolt_r1cs::constraints::rv64` layout.
pub fn r1cs_cycle_witness<C: CycleRow, F: Field>(
    trace: &[C],
    t: usize,
    bytecode: &BytecodePreprocessing,
) -> [F; NUM_VARS_PER_CYCLE] {
    let cycle = &trace[t];
    let next = trace.get(t + 1);

    let mut w = [F::from_u64(0); NUM_VARS_PER_CYCLE];
    w[V_CONST] = F::from_u64(1);

    if cycle.is_noop() {
        w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = F::from_u64(1);
        w[V_NEXT_IS_NOOP] = F::from_u64(next.is_none_or(|c| c.is_noop()) as u64);
        fill_next_fields(&mut w, next, bytecode);
        return w;
    }

    let cflags = cycle.circuit_flags();
    let iflags = cycle.instruction_flags();

    // Instruction inputs
    let left_input = if iflags[InstructionFlags::LeftOperandIsPC] {
        cycle.unexpanded_pc()
    } else if iflags[InstructionFlags::LeftOperandIsRs1Value] {
        cycle.rs1_read().map_or(0, |(_, v)| v)
    } else {
        0
    };

    let right_input: i128 = if iflags[InstructionFlags::RightOperandIsImm] {
        cycle.imm()
    } else if iflags[InstructionFlags::RightOperandIsRs2Value] {
        cycle.rs2_read().map_or(0, |(_, v)| v as i128)
    } else {
        0
    };

    w[V_LEFT_INSTRUCTION_INPUT] = F::from_u64(left_input);
    w[V_RIGHT_INSTRUCTION_INPUT] = F::from_i128(right_input);
    w[V_PRODUCT] = w[V_LEFT_INSTRUCTION_INPUT] * w[V_RIGHT_INSTRUCTION_INPUT];

    // Lookup output
    let lookup_output = cycle.lookup_output();
    w[V_LOOKUP_OUTPUT] = F::from_u64(lookup_output);

    // Lookup operands — must match R1CS constraints
    let (left_lookup, right_lookup) = lookup_operands(
        left_input,
        right_input,
        w[V_PRODUCT],
        &cflags,
        lookup_output,
    );
    w[V_LEFT_LOOKUP_OPERAND] = left_lookup;
    w[V_RIGHT_LOOKUP_OPERAND] = right_lookup;

    // Registers
    w[V_RS1_VALUE] = F::from_u64(cycle.rs1_read().map_or(0, |(_, v)| v));
    w[V_RS2_VALUE] = F::from_u64(cycle.rs2_read().map_or(0, |(_, v)| v));
    w[V_RD_WRITE_VALUE] = F::from_u64(cycle.rd_write().map_or(0, |(_, _, post)| post));

    // RAM
    w[V_RAM_ADDRESS] = F::from_u64(cycle.ram_access_address().unwrap_or(0));
    w[V_RAM_READ_VALUE] = F::from_u64(cycle.ram_read_value().unwrap_or(0));
    w[V_RAM_WRITE_VALUE] = F::from_u64(cycle.ram_write_value().unwrap_or(0));

    // PCs
    w[V_PC] = F::from_u64(bytecode.get_pc(cycle) as u64);
    w[V_UNEXPANDED_PC] = F::from_u64(cycle.unexpanded_pc());
    w[V_IMM] = F::from_i128(cycle.imm());

    // Circuit flags
    w[V_FLAG_ADD_OPERANDS] = F::from_u64(cflags[CircuitFlags::AddOperands] as u64);
    w[V_FLAG_SUBTRACT_OPERANDS] = F::from_u64(cflags[CircuitFlags::SubtractOperands] as u64);
    w[V_FLAG_MULTIPLY_OPERANDS] = F::from_u64(cflags[CircuitFlags::MultiplyOperands] as u64);
    w[V_FLAG_LOAD] = F::from_u64(cflags[CircuitFlags::Load] as u64);
    w[V_FLAG_STORE] = F::from_u64(cflags[CircuitFlags::Store] as u64);
    w[V_FLAG_JUMP] = F::from_u64(cflags[CircuitFlags::Jump] as u64);
    w[V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] =
        F::from_u64(cflags[CircuitFlags::WriteLookupOutputToRD] as u64);
    w[V_FLAG_VIRTUAL_INSTRUCTION] = F::from_u64(cflags[CircuitFlags::VirtualInstruction] as u64);
    w[V_FLAG_ASSERT] = F::from_u64(cflags[CircuitFlags::Assert] as u64);
    w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] =
        F::from_u64(cflags[CircuitFlags::DoNotUpdateUnexpandedPC] as u64);
    w[V_FLAG_ADVICE] = F::from_u64(cflags[CircuitFlags::Advice] as u64);
    w[V_FLAG_IS_COMPRESSED] = F::from_u64(cflags[CircuitFlags::IsCompressed] as u64);
    w[V_FLAG_IS_FIRST_IN_SEQUENCE] = F::from_u64(cflags[CircuitFlags::IsFirstInSequence] as u64);
    w[V_FLAG_IS_LAST_IN_SEQUENCE] = F::from_u64(cflags[CircuitFlags::IsLastInSequence] as u64);

    // Product factors
    w[V_BRANCH] = F::from_u64(iflags[InstructionFlags::Branch] as u64);
    w[V_SHOULD_BRANCH] = w[V_LOOKUP_OUTPUT] * w[V_BRANCH];

    // Next-cycle fields
    fill_next_fields(&mut w, next, bytecode);
    let next_is_noop = next.is_none_or(|c| c.is_noop());
    w[V_NEXT_IS_NOOP] = F::from_u64(next_is_noop as u64);
    w[V_SHOULD_JUMP] = w[V_FLAG_JUMP] * (F::from_u64(1) - w[V_NEXT_IS_NOOP]);

    w
}

fn fill_next_fields<C: CycleRow, F: Field>(
    w: &mut [F; NUM_VARS_PER_CYCLE],
    next: Option<&C>,
    bytecode: &BytecodePreprocessing,
) {
    if let Some(nc) = next {
        w[V_NEXT_PC] = F::from_u64(bytecode.get_pc(nc) as u64);
        w[V_NEXT_UNEXPANDED_PC] = F::from_u64(nc.unexpanded_pc());
        let nc_flags = nc.circuit_flags();
        w[V_NEXT_IS_VIRTUAL] = F::from_u64(nc_flags[CircuitFlags::VirtualInstruction] as u64);
        w[V_NEXT_IS_FIRST_IN_SEQUENCE] =
            F::from_u64(nc_flags[CircuitFlags::IsFirstInSequence] as u64);
    }
}

/// Lookup operands matching the R1CS constraints:
/// - Add: `(0, left + right)`
/// - Sub: `(0, left - right + 2^64)`
/// - Mul: `(0, product)`
/// - Advice: `(0, lookup_output)`
/// - Default: `(left, right)`
fn lookup_operands<F: Field>(
    left: u64,
    right: i128,
    product: F,
    cflags: &[bool; NUM_CIRCUIT_FLAGS],
    lookup_output: u64,
) -> (F, F) {
    if cflags[CircuitFlags::AddOperands] {
        (F::from_u64(0), F::from_i128(left as i128 + right))
    } else if cflags[CircuitFlags::SubtractOperands] {
        (
            F::from_u64(0),
            F::from_i128(left as i128 - right + (1i128 << 64)),
        )
    } else if cflags[CircuitFlags::MultiplyOperands] {
        (F::from_u64(0), product)
    } else if cflags[CircuitFlags::Advice] {
        (F::from_u64(0), F::from_u64(lookup_output))
    } else {
        (F::from_u64(left), F::from_i128(right))
    }
}

/// Flat R1CS witness for the entire trace.
///
/// Returns `Vec<F>` of length `trace.len() * num_vars_padded`.
pub fn build_r1cs_witness<C: CycleRow, F: Field>(
    trace: &[C],
    bytecode: &BytecodePreprocessing,
    num_vars_padded: usize,
) -> Vec<F> {
    let n = trace.len();
    let mut witness = vec![F::from_u64(0); n * num_vars_padded];
    for t in 0..n {
        let row = r1cs_cycle_witness::<C, F>(trace, t, bytecode);
        witness[t * num_vars_padded..t * num_vars_padded + NUM_VARS_PER_CYCLE]
            .copy_from_slice(&row);
    }
    witness
}

/// Write the BN254 Fr coprocessor R1CS columns (`V_FLAG_IS_FIELD_*` and
/// `V_FIELD_OP_{A,B,RESULT}`) from a stream of [`FieldRegEvent`]s carrying
/// `FieldOpPayload`s. Events without a payload (e.g. `FMov{I2F,F2I}`) are
/// skipped — they do not activate any FieldOp gate.
///
/// Callers must ensure:
/// - `witness` was produced by [`build_r1cs_witness`] (or equivalent) over a
///   trace of length `num_cycles`, padded to `num_vars_padded` per cycle.
/// - Each event's `cycle` is within `0..num_cycles` and its `op.funct3` is one
///   of the FieldOp selectors (0x02..=0x05). Violations panic in debug.
///
/// This is the canonical bridge between the FR Twist raw-event stream and the
/// R1CS FADD/FSUB gates (`rv64_constraints`). Tests and provers call this
/// after `build_r1cs_witness` / `extract_trace` to activate the FieldOp rows.
pub fn apply_field_op_events_to_r1cs<F: Field>(
    witness: &mut [F],
    num_cycles: usize,
    num_vars_padded: usize,
    events: &[FieldRegEvent],
) {
    for e in events {
        let Some(op) = e.op else {
            continue;
        };
        debug_assert!(e.cycle < num_cycles, "FieldRegEvent cycle out of range");

        let (flag_idx, valid_funct3) = match op.funct3 {
            FIELD_OP_FUNCT3_FMUL => (V_FLAG_IS_FIELD_MUL, true),
            FIELD_OP_FUNCT3_FADD => (V_FLAG_IS_FIELD_ADD, true),
            FIELD_OP_FUNCT3_FSUB => (V_FLAG_IS_FIELD_SUB, true),
            FIELD_OP_FUNCT3_FINV => (V_FLAG_IS_FIELD_INV, true),
            _ => (0, false),
        };
        debug_assert!(valid_funct3, "invalid FieldOp funct3: {:#x}", op.funct3);
        if !valid_funct3 {
            continue;
        }

        let base = e.cycle * num_vars_padded;
        witness[base + flag_idx] = F::from_u64(1);
        witness[base + V_FIELD_OP_A] = limbs_to_field::<F>(&op.a);
        witness[base + V_FIELD_OP_B] = limbs_to_field::<F>(&op.b);
        witness[base + V_FIELD_OP_RESULT] = limbs_to_field::<F>(&e.new);
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_r1cs::constraints::rv64::rv64_constraints;
    use jolt_witness::derived::FieldOpPayload;
    use num_traits::identities::Zero;

    /// Build a no-op-style witness vector with the right padding, activate a
    /// single FADD event via the helper, and confirm the R1CS accepts it.
    #[test]
    fn fadd_event_activates_gate() {
        let num_cycles = 4;
        let num_vars_padded = NUM_VARS_PER_CYCLE.next_power_of_two();
        let mut witness = vec![Fr::zero(); num_cycles * num_vars_padded];

        // Mark every cycle as no-op-padded (const=1, DoNotUpdatePC=1, NextIsNoop=1)
        // so all rows are vacuously satisfied before we activate FADD.
        for t in 0..num_cycles {
            let base = t * num_vars_padded;
            witness[base + V_CONST] = Fr::from_u64(1);
            witness[base + V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::from_u64(1);
            witness[base + V_NEXT_IS_NOOP] = Fr::from_u64(1);
        }

        let events = vec![FieldRegEvent {
            cycle: 2,
            slot: 3,
            old: [0, 0, 0, 0],
            new: [579, 0, 0, 0], // 123 + 456
            op: Some(FieldOpPayload {
                funct3: FIELD_OP_FUNCT3_FADD,
                a: [123, 0, 0, 0],
                b: [456, 0, 0, 0],
            }),
        }];

        apply_field_op_events_to_r1cs::<Fr>(&mut witness, num_cycles, num_vars_padded, &events);

        // Confirm the right slots got populated at cycle 2.
        let base = 2 * num_vars_padded;
        assert_eq!(witness[base + V_FLAG_IS_FIELD_ADD], Fr::from_u64(1));
        assert_eq!(witness[base + V_FIELD_OP_A], Fr::from_u64(123));
        assert_eq!(witness[base + V_FIELD_OP_B], Fr::from_u64(456));
        assert_eq!(witness[base + V_FIELD_OP_RESULT], Fr::from_u64(579));

        // R1CS accepts the activated FADD cycle.
        let matrices = rv64_constraints::<Fr>();
        let cycle2 = &witness[base..base + NUM_VARS_PER_CYCLE];
        matrices
            .check_witness(cycle2)
            .expect("FADD event should produce a satisfying witness");
    }

    #[test]
    fn fadd_event_with_wrong_result_rejects() {
        let num_cycles = 1;
        let num_vars_padded = NUM_VARS_PER_CYCLE.next_power_of_two();
        let mut witness = vec![Fr::zero(); num_cycles * num_vars_padded];
        witness[V_CONST] = Fr::from_u64(1);
        witness[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::from_u64(1);
        witness[V_NEXT_IS_NOOP] = Fr::from_u64(1);

        // Tampered: claim 100 + 200 = 999.
        let events = vec![FieldRegEvent {
            cycle: 0,
            slot: 0,
            old: [0, 0, 0, 0],
            new: [999, 0, 0, 0],
            op: Some(FieldOpPayload {
                funct3: FIELD_OP_FUNCT3_FADD,
                a: [100, 0, 0, 0],
                b: [200, 0, 0, 0],
            }),
        }];

        apply_field_op_events_to_r1cs::<Fr>(&mut witness, num_cycles, num_vars_padded, &events);

        let matrices = rv64_constraints::<Fr>();
        let cycle0 = &witness[..NUM_VARS_PER_CYCLE];
        assert!(
            matrices.check_witness(cycle0).is_err(),
            "tampered FADD result must violate the gate"
        );
    }

    #[test]
    fn fsub_event_activates_gate() {
        let num_cycles = 1;
        let num_vars_padded = NUM_VARS_PER_CYCLE.next_power_of_two();
        let mut witness = vec![Fr::zero(); num_cycles * num_vars_padded];
        witness[V_CONST] = Fr::from_u64(1);
        witness[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::from_u64(1);
        witness[V_NEXT_IS_NOOP] = Fr::from_u64(1);

        // 1000 - 250 = 750
        let events = vec![FieldRegEvent {
            cycle: 0,
            slot: 5,
            old: [0, 0, 0, 0],
            new: [750, 0, 0, 0],
            op: Some(FieldOpPayload {
                funct3: FIELD_OP_FUNCT3_FSUB,
                a: [1000, 0, 0, 0],
                b: [250, 0, 0, 0],
            }),
        }];

        apply_field_op_events_to_r1cs::<Fr>(&mut witness, num_cycles, num_vars_padded, &events);

        let matrices = rv64_constraints::<Fr>();
        matrices
            .check_witness(&witness[..NUM_VARS_PER_CYCLE])
            .expect("FSUB with a-b=result should satisfy gate 20");
    }

    #[test]
    fn non_field_op_events_are_skipped() {
        let num_cycles = 2;
        let num_vars_padded = NUM_VARS_PER_CYCLE.next_power_of_two();
        let mut witness = vec![Fr::zero(); num_cycles * num_vars_padded];

        // FMov-style event — no payload. Must not touch FieldOp columns.
        let events = vec![FieldRegEvent {
            cycle: 1,
            slot: 0,
            old: [0, 0, 0, 0],
            new: [42, 0, 0, 0],
            op: None,
        }];

        apply_field_op_events_to_r1cs::<Fr>(&mut witness, num_cycles, num_vars_padded, &events);

        let base = num_vars_padded; // cycle 1
        assert_eq!(witness[base + V_FLAG_IS_FIELD_ADD], Fr::zero());
        assert_eq!(witness[base + V_FLAG_IS_FIELD_SUB], Fr::zero());
        assert_eq!(witness[base + V_FIELD_OP_A], Fr::zero());
        assert_eq!(witness[base + V_FIELD_OP_B], Fr::zero());
        assert_eq!(witness[base + V_FIELD_OP_RESULT], Fr::zero());
    }
}
