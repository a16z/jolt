#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, reason = "integration tests should fail loudly")]
mod tests {
    use std::sync::Arc;

    use common::{
        constants::RAM_START_ADDRESS,
        jolt_device::{JoltDevice, MemoryLayout},
    };
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_openings::CommitmentScheme;
    use jolt_program::{
        execution::{JoltProgram, MemoryImage, OwnedTrace, TraceOutput},
        preprocess::JoltProgramPreprocessing,
    };
    use jolt_prover::{JoltBackend, JoltProverPreprocessing, ProverConfig};
    use jolt_riscv::{
        CapturedState, JoltInstructionKind as Kind, JoltInstructionRow, JoltTraceRow, LoadState,
        NonMemoryState, NormalizedOperands, StoreState, RV64IMAC_JOLT,
    };
    use jolt_transcript::LegacyBlake2bTranscript;
    use jolt_verifier::{preprocessing::ProgramPreprocessing, JoltVerifierPreprocessing};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    const MAX_TRACE_LENGTH: usize = 1 << 16;

    fn program_image() -> Vec<(u64, u8)> {
        [
            0x7fff_f0b7_u32,
            0x0040_8093,
            0x0000_b103,
            0x0010_0193,
            0x0030_b423,
            0x0000_00e7,
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .enumerate()
        .map(|(offset, byte)| (RAM_START_ADDRESS + offset as u64, byte))
        .collect()
    }

    fn instruction(
        instruction_kind: Kind,
        address: usize,
        operands: NormalizedOperands,
    ) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind,
            address,
            operands,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    // Verifier preprocessing accepts public/deserialized layouts without
    // requiring the lowest remapped address to be eight-byte aligned.
    fn misaligned_remap_layout() -> MemoryLayout {
        let lowest = RAM_START_ADDRESS - 0xffc;
        MemoryLayout {
            program_size: 24,
            max_trusted_advice_size: 0,
            trusted_advice_start: lowest,
            trusted_advice_end: lowest,
            max_untrusted_advice_size: 0,
            untrusted_advice_start: lowest,
            untrusted_advice_end: lowest,
            max_input_size: 0,
            max_output_size: 0,
            input_start: lowest,
            input_end: lowest,
            output_start: lowest,
            output_end: lowest,
            stack_size: 0,
            stack_end: RAM_START_ADDRESS + 24,
            heap_size: 40,
            heap_end: RAM_START_ADDRESS + 64,
            panic: lowest,
            termination: lowest + 8,
            io_end: lowest + 16,
        }
    }

    #[test]
    fn misaligned_doubleword_access_is_rejected() {
        let lowest = RAM_START_ADDRESS - 0xffc;
        assert_eq!(lowest % 8, 4);
        let addresses = (0..6)
            .map(|index| RAM_START_ADDRESS as usize + 4 * index)
            .collect::<Vec<_>>();
        let instructions = vec![
            instruction(
                Kind::LUI,
                addresses[0],
                NormalizedOperands {
                    rd: Some(1),
                    imm: 0x7fff_f000,
                    ..Default::default()
                },
            ),
            instruction(
                Kind::ADDI,
                addresses[1],
                NormalizedOperands {
                    rs1: Some(1),
                    rd: Some(1),
                    imm: 4,
                    ..Default::default()
                },
            ),
            instruction(
                Kind::LD,
                addresses[2],
                NormalizedOperands {
                    rs1: Some(1),
                    rd: Some(2),
                    ..Default::default()
                },
            ),
            instruction(
                Kind::ADDI,
                addresses[3],
                NormalizedOperands {
                    rs1: Some(0),
                    rd: Some(3),
                    imm: 1,
                    ..Default::default()
                },
            ),
            instruction(
                Kind::SD,
                addresses[4],
                NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(3),
                    imm: 8,
                    ..Default::default()
                },
            ),
            instruction(
                Kind::JALR,
                addresses[5],
                NormalizedOperands {
                    rs1: Some(0),
                    rd: Some(1),
                    ..Default::default()
                },
            ),
        ];
        let layout = misaligned_remap_layout();
        assert_eq!(layout.remapped_word_address(lowest), Ok(0));
        assert_eq!(layout.remapped_word_address(lowest + 8), Ok(1));
        let mut program_preprocessing = JoltProgramPreprocessing::new(
            instructions.clone(),
            program_image(),
            layout.clone(),
            addresses[0] as u64,
            MAX_TRACE_LENGTH,
            RV64IMAC_JOLT,
        )
        .expect("program preprocessing");
        // Pack the program image by the custom remap so the RAM value-check
        // cannot reject this witness for an unrelated initial-state mismatch.
        let image_start = layout
            .remapped_word_address(RAM_START_ADDRESS)
            .expect("program image start");
        let mut remapped_image = vec![0_u64; 4];
        for (address, byte) in program_image() {
            let index = layout.remapped_word_address(address).expect("image byte") - image_start;
            remapped_image[index as usize] |= u64::from(byte) << (8 * (address & 7));
        }
        program_preprocessing.ram.bytecode_words = remapped_image;
        let program_preprocessing = Arc::new(program_preprocessing);
        let pc = |index: usize| {
            program_preprocessing
                .bytecode
                .get_pc(&instructions[index])
                .expect("instruction must have a bytecode slot") as u32
        };
        let rows = vec![
            JoltTraceRow::from_components(
                CapturedState::NonMemory(NonMemoryState {
                    rd_write_value: 0x7fff_f000,
                    ..Default::default()
                }),
                &instructions[0],
                pc(0),
            )
            .expect("LUI row"),
            JoltTraceRow::from_components(
                CapturedState::NonMemory(NonMemoryState {
                    rs1_value: 0x7fff_f000,
                    rd_pre_value: 0x7fff_f000,
                    rd_write_value: lowest,
                    ..Default::default()
                }),
                &instructions[1],
                pc(1),
            )
            .expect("ADDI address row"),
            JoltTraceRow::from_components(
                CapturedState::Load(LoadState {
                    rs1_value: lowest,
                    ram_address: lowest,
                    rd_pre_value: 0,
                    rd_write_value: 0,
                }),
                &instructions[2],
                pc(2),
            )
            .expect("misaligned LD row"),
            JoltTraceRow::from_components(
                CapturedState::NonMemory(NonMemoryState {
                    rd_write_value: 1,
                    ..Default::default()
                }),
                &instructions[3],
                pc(3),
            )
            .expect("ADDI value row"),
            JoltTraceRow::from_components(
                CapturedState::Store(StoreState {
                    rs1_value: lowest,
                    rs2_value: 1,
                    ram_read_value: 0,
                    ram_address: lowest + 8,
                }),
                &instructions[4],
                pc(4),
            )
            .expect("misaligned SD row"),
            JoltTraceRow::from_components(
                CapturedState::NonMemory(NonMemoryState {
                    rd_pre_value: lowest,
                    rd_write_value: addresses[5] as u64 + 4,
                    ..Default::default()
                }),
                &instructions[5],
                pc(5),
            )
            .expect("JALR row"),
        ];

        let config = ProverConfig::derive_compact::<Fr>(
            &rows,
            &layout,
            program_preprocessing.ram.min_bytecode_address,
            program_preprocessing.ram.bytecode_words.len(),
            MAX_TRACE_LENGTH,
        )
        .expect("proof config");
        let public_io = JoltDevice {
            memory_layout: layout.clone(),
            ..Default::default()
        };
        let trace = TraceOutput::new(
            Arc::new(rows),
            public_io.clone(),
            Some(MemoryImage {
                bytes: program_image(),
            }),
            None,
        );
        let program = Arc::new(JoltProgram::from_parts(
            Vec::new(),
            instructions,
            program_image(),
            RAM_START_ADDRESS + 24,
            RAM_START_ADDRESS,
        ));
        let witness = TraceBackend::<OwnedTrace>::from_compact(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(&program, &program_preprocessing, trace),
        );

        let total_vars = config.commitment_total_vars(&layout, false, false, None);
        let pcs_setup = DoryScheme::setup_prover(total_vars);
        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            ProgramPreprocessing::Full(program_preprocessing),
            [0; 32],
            DoryScheme::verifier_setup(&pcs_setup),
            None,
        );
        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup,
            committed_program: None,
        };
        let proof = jolt_prover::dory::prove::<
            Fr,
            DoryScheme,
            Pedersen<Bn254G1>,
            LegacyBlake2bTranscript<Fr>,
            _,
        >(
            &JoltBackend::reference(),
            &prover_preprocessing,
            &config,
            None,
            &witness,
            &public_io,
        );
        let rejected = match proof {
            Ok(proof) => jolt_verifier::verify::<
                Fr,
                DoryScheme,
                Pedersen<Bn254G1>,
                LegacyBlake2bTranscript<Fr>,
            >(&prover_preprocessing.verifier, &public_io, &proof, None)
            .is_err(),
            Err(_) => true,
        };
        assert!(rejected, "misaligned LD/SD proof was accepted");
    }
}
