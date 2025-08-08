// /// Tests the consistency and correctness of a virtual instruction sequence.
// /// In detail:
// /// 1. Sets the registers to given values for rs1 and rs2.
// /// 2. Constructs an RVTraceRow with the provided instruction and register values.
// /// 3. Generates the virtual instruction sequence using the RISCVTrace trait.
// /// 4. Iterates over each row in the virtual sequence and validates the state changes.
// /// 5. Verifies that rs1 and rs2 have not been clobbered.
// /// 6. Ensures that the result is correctly written to the rd register.
// /// 7. Checks that no unintended modifications have been made to other registers.
// pub fn virtual_sequence_trace_test<I: RISCVInstruction + VirtualInstructionSequence + Copy>() {
//     let mut rng = StdRng::seed_from_u64(12345);

//     for _ in 0..1000 {
//         let rs1 = rng.next_u64() % 32;
//         let rs2 = rng.next_u64() % 32;
//         let mut rd = rng.next_u64() % 32;
//         while rd == 0 {
//             rd = rng.next_u64() % 32;
//         }
//         let rs1_val = if rs1 == 0 { 0 } else { rng.next_u32() as u64 };
//         let rs2_val = if rs2 == rs1 {
//             rs1_val
//         } else if rs2 == 0 {
//             0
//         } else {
//             rng.next_u32() as u64
//         };

//         let mut registers = vec![0u64; REGISTER_COUNT as usize];
//         registers[rs1 as usize] = rs1_val;
//         registers[rs2 as usize] = rs2_val;

//         let instruction = I::new(rng.next_u32(), rng.next_u64());

//         let mut cpu = Cpu::new(Xlen::Bit32);
//         cpu.x[rs1 as usize] = rs1_val;
//         cpu.x[rs2 as usize] = rs2_val;

//         instruction.trace(&mut cpu);

//         // for cycle in cpu.trace {
//         //     let ram_access = cycle.ram_access();
//         //     match ram_access {
//         //         RAMAccess::Read(read) => {
//         //             assert_eq!(
//         //                 registers[read.address as usize], read.value,
//         //                 "RAM read value mismatch"
//         //             );
//         //         }
//         //         RAMAccess::Write(write) => {
//         //             assert_eq!(
//         //                 registers[write.address as usize], write.pre_value,
//         //                 "RAM write pre-value mismatch"
//         //             );
//         //             registers[write.address as usize] = write.post_value;
//         //         }
//         //         RAMAccess::NoOp => {}
//         //     }
//         // }

//         // Verify state
//         let rd_post_val = cpu.x[rd as usize];

//         // Check registers not clobbered
//         if rs1 != rd {
//             assert_eq!(registers[rs1 as usize], rs1_val, "rs1 was clobbered");
//         }
//         if rs2 != rd {
//             assert_eq!(registers[rs2 as usize], rs2_val, "rs2 was clobbered");
//         }

//         // Verify other registers untouched
//         for i in 0..32 {
//             if i != rs1 as usize && i != rs2 as usize && i != rd as usize {
//                 assert_eq!(
//                     registers[i], 0,
//                     "Register {} was modified when it shouldn't have been",
//                     i
//                 );
//             }
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use crate::{
        emulator::cpu::Xlen,
        instruction::{
            RISCVCycle, RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, ADDIW, ADDW,
            AMOADDD, AMOADDW, AMOANDD, AMOANDW, AMOMAXD, AMOMAXUD, AMOMAXUW, AMOMAXW, AMOMIND,
            AMOMINUD, AMOMINUW, AMOMINW, AMOORD, AMOORW, AMOSWAPD, AMOSWAPW, AMOXORD, AMOXORW, DIV,
            DIVU, DIVUW, DIVW, LB, LBU, LH, LHU, LW, LWU, MULH, MULHSU, MULW, REM, REMU, REMUW,
            REMW, SB, SH, SHA256, SHA256INIT, SLL, SLLI, SLLIW, SLLW, SRA, SRAI, SRAIW, SRAW, SRL,
            SRLI, SRLIW, SRLW, SUBW, SW,
        },
    };
    use std::any::type_name;

    fn validate_sequence_ordering(sequence: &[RV32IMInstruction]) -> Result<(), String> {
        let expected_remaining = sequence.len();

        for (index, instr) in sequence.iter().enumerate() {
            let normalized = instr.normalize();
            let current_remaining = normalized.virtual_sequence_remaining;

            if current_remaining.is_none() {
                return Err(format!("Instruction {index} has no remaining"));
            }

            let current = current_remaining.unwrap();

            if (current as usize + index + 1) != expected_remaining {
                return Err(format!(
                    "Invalid remaining at index {}: expected {}, got {}",
                    index,
                    expected_remaining - index - 1,
                    current
                ));
            }
        }

        Ok(())
    }

    // Get the type name at runtime
    fn get_instruction_name<T: RISCVInstruction>(_instance: &T) -> &'static str {
        type_name::<T>().split("::").last().unwrap_or("Unknown")
    }

    trait VirtualInstructionSequenceWrapper {
        fn virtual_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction>;
        fn get_type_name(&self) -> &'static str;
    }

    impl<T> VirtualInstructionSequenceWrapper for T
    where
        T: RISCVTrace + RISCVInstruction,
        RISCVCycle<T>: Into<RV32IMCycle>,
    {
        fn virtual_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
            self.virtual_sequence(xlen)
        }

        fn get_type_name(&self) -> &'static str {
            get_instruction_name(self)
        }
    }

    #[test]
    fn test_remaining() {
        println!("{}", get_instruction_name(&DIV::new(0, 0, false, false)));

        let xlen_32 = Xlen::Bit32;
        let instruction_pairs_32: Vec<Box<dyn VirtualInstructionSequenceWrapper>> = vec![
            Box::new(DIV::new(0, 0, false, false)),
            Box::new(DIVU::new(0, 0, false, false)),
            Box::new(LB::new(0, 0, false, false)),
            Box::new(LBU::new(0, 0, false, false)),
            Box::new(LH::new(0, 0, false, false)),
            Box::new(LHU::new(0, 0, false, false)),
            Box::new(MULH::new(0, 0, false, false)),
            Box::new(MULHSU::new(0, 0, false, false)),
            Box::new(REM::new(0, 0, false, false)),
            Box::new(REMU::new(0, 0, false, false)),
            Box::new(SB::new(0, 0, false, false)),
            Box::new(SH::new(0, 0, false, false)),
            Box::new(SLL::new(0, 0, false, false)),
            Box::new(SLLI::new(0, 0, false, false)),
            Box::new(SRA::new(0, 0, false, false)),
            Box::new(SRAI::new(0, 0, false, false)),
            Box::new(SRL::new(0, 0, false, false)),
            Box::new(SRLI::new(0, 0, false, false)),
            Box::new(SHA256::new(0, 0, false, false)),
            Box::new(SHA256INIT::new(0, 0, false, false)),
        ];

        let xlen_64 = Xlen::Bit64;
        let instruction_sequence_pairs_64: Vec<Box<dyn VirtualInstructionSequenceWrapper>> = vec![
            Box::new(ADDIW::new(0, 0, false, false)),
            Box::new(ADDW::new(0, 0, false, false)),
            Box::new(AMOADDD::new(0, 0, false, false)),
            Box::new(AMOADDW::new(0, 0, false, false)),
            Box::new(AMOANDD::new(0, 0, false, false)),
            Box::new(AMOANDW::new(0, 0, false, false)),
            Box::new(AMOMAXD::new(0, 0, false, false)),
            Box::new(AMOMAXUD::new(0, 0, false, false)),
            Box::new(AMOMAXUW::new(0, 0, false, false)),
            Box::new(AMOMAXW::new(0, 0, false, false)),
            Box::new(AMOMIND::new(0, 0, false, false)),
            Box::new(AMOMINUD::new(0, 0, false, false)),
            Box::new(AMOMINUW::new(0, 0, false, false)),
            Box::new(AMOMINW::new(0, 0, false, false)),
            Box::new(AMOORD::new(0, 0, false, false)),
            Box::new(AMOORW::new(0, 0, false, false)),
            Box::new(AMOSWAPD::new(0, 0, false, false)),
            Box::new(AMOSWAPW::new(0, 0, false, false)),
            Box::new(AMOXORD::new(0, 0, false, false)),
            Box::new(AMOXORW::new(0, 0, false, false)),
            Box::new(DIV::new(0, 0, false, false)),
            Box::new(DIVU::new(0, 0, false, false)),
            Box::new(DIVUW::new(0, 0, false, false)),
            Box::new(DIVW::new(0, 0, false, false)),
            Box::new(LB::new(0, 0, false, false)),
            Box::new(LBU::new(0, 0, false, false)),
            Box::new(LH::new(0, 0, false, false)),
            Box::new(LHU::new(0, 0, false, false)),
            Box::new(LW::new(0, 0, false, false)),
            Box::new(LWU::new(0, 0, false, false)),
            Box::new(MULH::new(0, 0, false, false)),
            Box::new(MULHSU::new(0, 0, false, false)),
            Box::new(MULW::new(0, 0, false, false)),
            Box::new(REM::new(0, 0, false, false)),
            Box::new(REMU::new(0, 0, false, false)),
            Box::new(REMUW::new(0, 0, false, false)),
            Box::new(REMW::new(0, 0, false, false)),
            Box::new(SB::new(0, 0, false, false)),
            Box::new(SH::new(0, 0, false, false)),
            Box::new(SLLI::new(0, 0, false, false)),
            Box::new(SLLIW::new(0, 0, false, false)),
            Box::new(SLL::new(0, 0, false, false)),
            Box::new(SLLW::new(0, 0, false, false)),
            Box::new(SRAI::new(0, 0, false, false)),
            Box::new(SRAIW::new(0, 0, false, false)),
            Box::new(SRA::new(0, 0, false, false)),
            Box::new(SRAW::new(0, 0, false, false)),
            Box::new(SRLI::new(0, 0, false, false)),
            Box::new(SRLIW::new(0, 0, false, false)),
            Box::new(SRL::new(0, 0, false, false)),
            Box::new(SRLW::new(0, 0, false, false)),
            Box::new(SUBW::new(0, 0, false, false)),
            Box::new(SW::new(0, 0, false, false)),
        ];

        let mut failures_32 = Vec::new();
        for instruction in instruction_pairs_32 {
            let sequence = instruction.virtual_sequence(xlen_32);
            if sequence.is_empty() {
                continue;
            }

            if let Err(e) = validate_sequence_ordering(&sequence) {
                failures_32.push(format!(
                    "32-bit instruction {}: {}",
                    instruction.get_type_name(),
                    e
                ));
            }
        }

        let mut failures_64 = Vec::new();
        for instruction in instruction_sequence_pairs_64 {
            let sequence = instruction.virtual_sequence(xlen_64);
            if sequence.is_empty() {
                continue;
            }

            if let Err(e) = validate_sequence_ordering(&sequence) {
                failures_64.push(format!(
                    "64-bit instruction {}: {}",
                    instruction.get_type_name(),
                    e
                ));
            }
        }

        // Report all failures
        let total_failures = failures_32.len() + failures_64.len();
        assert!(
            total_failures == 0,
            "Found {} virtual sequence validation failures:\n{}{}",
            total_failures,
            if !failures_32.is_empty() {
                format!(
                    "\n32-bit instruction failures:\n{}",
                    failures_32
                        .iter()
                        .map(|f| format!("  - {f}\n"))
                        .collect::<String>()
                )
            } else {
                String::new()
            },
            if !failures_64.is_empty() {
                format!(
                    "\n64-bit instruction failures:\n{}",
                    failures_64
                        .iter()
                        .map(|f| format!("  - {f}\n"))
                        .collect::<String>()
                )
            } else {
                String::new()
            }
        );
    }
}
