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
