use ark_std::test_rng;
use onnx_tracer::{
    constants::{MAX_TENSOR_SIZE, TENSOR_REGISTER_COUNT},
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};
use rand::RngCore;

use crate::jolt::execution_trace::WORD_SIZE;
use crate::{
    jolt::{
        execution_trace::{JoltONNXCycle, ONNXLookupQuery},
        instruction::VirtualInstructionSequence,
    },
    utils::u64_vec_to_i128_iter,
};

/// Tests the consistency and correctness of a virtual instruction sequence.
/// In detail:
/// 1. Sets the tensor_registers to given values for `x` and `y`.
/// 2. Constructs an `RVTraceRow` with the provided opcode and register values.
/// 3. Generates the virtual instruction sequence using the specified instruction type.
/// 4. Iterates over each row in the virtual sequence and validates the state changes.
/// 5. Verifies that the tensor_registers `t_x` and `t_y` have not been modified (not clobbered).
/// 6. Ensures that the result of the instruction sequence is correctly written to the `td` register.
/// 7. Checks that no unintended modifications have been made to other tensor_registers.
pub fn jolt_virtual_sequence_test<I: VirtualInstructionSequence>(opcode: ONNXOpcode) {
    let mut rng = test_rng();

    for _ in 0..1000 {
        // Randomly select tensor register's indices for t_x, t_y, and td (destination tensor register).
        // t_x and t_y are source tensor_registers, td is the destination tensor register.
        let t_x = rng.next_u64() % 32;
        let t_y = rng.next_u64() % 32;

        // Ensure td is not zero
        let mut td = rng.next_u64() % 32;
        while td == 0 {
            td = rng.next_u64() % 32;
        }

        // Assign a random value to x, but if t_x is zero, force x to be zero.
        // This simulates the behavior of register zero.
        let x = if t_x == 0 {
            vec![0u64; MAX_TENSOR_SIZE]
        } else {
            (0..MAX_TENSOR_SIZE)
                .map(|_| rng.next_u32() as u64)
                .collect::<Vec<u64>>()
        };

        // Assign a value to y:
        // - If t_y == t_x, y is set to x (ensures both source (tensor) tensor_registers have the same value).
        // - If t_y is zero, y is forced to zero (simulating zero (tensor) register).
        // - Otherwise, y is assigned a random value.
        let y = if t_y == t_x {
            x.clone()
        } else if t_y == 0 {
            vec![0u64; MAX_TENSOR_SIZE]
        } else {
            (0..MAX_TENSOR_SIZE)
                .map(|_| rng.next_u32() as u64)
                .collect::<Vec<u64>>()
        };
        let result = I::sequence_output(x.clone(), y.clone());

        let mut tensor_registers =
            vec![vec![0u64; MAX_TENSOR_SIZE]; TENSOR_REGISTER_COUNT as usize];
        tensor_registers[t_x as usize] = x.clone();
        tensor_registers[t_y as usize] = y.clone();

        let cycle = ONNXCycle {
            instr: ONNXInstr {
                address: rng.next_u64() as usize,
                opcode: opcode.clone(),
                ts1: Some(t_x as usize),
                ts2: Some(t_y as usize),
                td: Some(td as usize),
                imm: None,
                virtual_sequence_remaining: None,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i128_iter(&x))),
                ts2_val: Some(Tensor::from(u64_vec_to_i128_iter(&y))),
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i128_iter(&result))),
            },
            advice_value: None,
        };

        let virtual_sequence = I::virtual_trace(cycle);
        assert_eq!(virtual_sequence.len(), I::SEQUENCE_LENGTH);

        for cycle in virtual_sequence {
            if let Some(ts1_addr) = cycle.instr.ts1 {
                assert_eq!(tensor_registers[ts1_addr], cycle.ts1_vals(), "{cycle:?}");
            }

            if let Some(ts2_addr) = cycle.instr.ts2 {
                assert_eq!(tensor_registers[ts2_addr], cycle.ts2_vals(), "{cycle:?}");
            }

            let output =
                ONNXLookupQuery::<WORD_SIZE>::to_lookup_output(&JoltONNXCycle::from(cycle.clone()));
            if let Some(td_addr) = cycle.instr.td {
                tensor_registers[td_addr] = output;
                assert_eq!(tensor_registers[td_addr], cycle.td_post_vals(), "{cycle:?}");
            } else {
                assert!(output == vec![1; MAX_TENSOR_SIZE], "{cycle:?}");
            }
        }

        for (index, val) in tensor_registers.iter().enumerate() {
            if index as u64 == t_x {
                if t_x != td {
                    // Check that t_x hasn't been clobbered
                    assert_eq!(*val, x);
                }
            } else if index as u64 == t_y {
                if t_y != td {
                    // Check that t_y hasn't been clobbered
                    assert_eq!(*val, y);
                }
            } else if index as u64 == td {
                // Check that result was written to td
                assert_eq!(
                    *val, result,
                    "Lookup mismatch for x {x:?} y {y:?} td {td:?}"
                );
            } else if index < 32 {
                // None of the other "real" registers were touched
                assert_eq!(
                    *val,
                    vec![0u64; MAX_TENSOR_SIZE],
                    "Other 'real' registers should not be touched"
                );
            }
        }
    }
}
