//! Soundness: fresh proofs of malformed LBU witnesses do not verify.
//!
//! Each case changes one `VirtualLoadB` / `VirtualExtractByteUnsigned` pair of
//! the memory-ops guest trace and proves the result with fresh commitments
//! under the same trusted preprocessing. The ordinary prover refuses such
//! witnesses through its own round and final-claim checks, so the ignored
//! `produce_lbu_proofs` is meant for a build with those checks disabled, and
//! `verify_lbu_proofs` for an unmodified build. `prepare_trusted_lbu_fixture`
//! pins the verifier preprocessing both sides use.

#[cfg(all(
    feature = "prover-fixtures",
    not(any(feature = "akita", feature = "field-inline"))
))]
mod support;

#[cfg(all(
    feature = "prover-fixtures",
    not(any(feature = "akita", feature = "field-inline"))
))]
#[expect(clippy::expect_used, reason = "soundness fixtures should fail loudly")]
mod adversarial {
    use std::collections::BTreeMap;
    use std::ops::Range;
    use std::path::PathBuf;
    use std::sync::Arc;
    #[cfg(not(feature = "zk"))]
    use std::thread;
    use std::{env, fs};

    use common::jolt_device::JoltDevice;
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltPolynomialId, JoltVirtualPolynomial,
    };
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_program::execution::{JoltProgram, TraceData, TraceOutput};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_prover::dory::DoryProverPreprocessing;
    use jolt_prover::{dory, JoltBackend, JoltSharedPreprocessing, ProverConfig, ProverError};
    use jolt_riscv::{JoltInstructionKind as Kind, JoltTraceRow};
    use jolt_transcript::{Keccak, ProverState, VerifierState};
    use jolt_verifier::proof::JoltProof;
    use jolt_verifier::{JoltVerifierPreprocessing, VerifierError};
    use jolt_witness::{
        ChunkVisitor, JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle, ProgramSource,
        RandomAccessRows, RowSource, Shape, TraceBackend, WitnessError,
    };

    #[cfg(feature = "zk")]
    use crate::support::with_zk_stack as wide_stack;
    use crate::support::{self, GuestCase};

    type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;
    type Verifier = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;

    /// Malformed witnesses, each a single change to one LBU pair.
    const ATTACKS: [&str; 9] = [
        // RAM word and scratch register `2^64 + w`; the selected byte is unchanged.
        "wide-word",
        // The extraction row claims another instruction's lookup table.
        "table",
        // The base read by the extraction is off by one; the result follows it.
        "offset",
        // The extracted byte is wrong.
        "result",
        // The extraction reads its base from another register.
        "base",
        // The extraction uses another immediate; the result follows it.
        "immediate",
        // The scratch read differs from the loaded word outside the selected byte.
        "scratch",
        // The extraction row is skipped.
        "skip",
        // Execution enters the extraction row without the load.
        "enter",
    ];
    const HONEST: &str = "honest";

    struct Fixture {
        program: Arc<JoltProgram>,
        preprocessing: DoryProverPreprocessing,
        trace: TraceOutput,
        /// Index of the `VirtualLoadB` row; the extraction follows it.
        load: usize,
    }

    fn fixture() -> Fixture {
        let prepared = support::prepare(&GuestCase {
            func: Some("memory_ops"),
            ..GuestCase::new("memory-ops-guest")
        });
        let load = prepared
            .trace
            .trace
            .proof_rows()
            .windows(2)
            .position(|pair| {
                pair.first().and_then(JoltTraceRow::instruction_kind) == Some(Kind::VirtualLoadB)
                    && pair.get(1).and_then(JoltTraceRow::instruction_kind)
                        == Some(Kind::VirtualExtractByteUnsigned)
            })
            .expect("the memory-ops trace contains an LBU");
        let preprocessing = dory::from_shared(
            JoltSharedPreprocessing::new(prepared.preprocessing).expect("shared preprocessing"),
        )
        .expect("Dory preprocessing");
        Fixture {
            program: prepared.program,
            preprocessing,
            trace: prepared.trace,
            load,
        }
    }

    fn byte(word: u64, offset: u64) -> u64 {
        (word >> (8 * (offset & 7))) & 0xff
    }

    /// Rebuilds row `index` after `edit` changes its instruction and register
    /// observations; the row keeps its bytecode PC and RAM access.
    fn edit_row(
        rows: &mut [JoltTraceRow],
        index: usize,
        edit: impl FnOnce(&mut jolt_riscv::JoltInstructionRow, &mut jolt_riscv::RegisterState),
    ) {
        let row = rows.get_mut(index).expect("row in trace");
        let mut instruction = row.instruction();
        let mut registers = row.registers();
        edit(&mut instruction, &mut registers);
        let edited = JoltTraceRow::new(
            instruction,
            registers,
            row.ram_access(),
            u32::try_from(row.pc()).expect("bytecode PC fits"),
        )
        .expect("edited row satisfies the row carrier");
        assert_ne!(edited, *row, "the attack must change the row");
        *row = edited;
    }

    fn malformed_rows(f: &Fixture, attack: &str) -> Vec<JoltTraceRow> {
        let mut rows = f.trace.trace.rows().to_vec();
        let extract = f.load + 1;
        let row = *rows.get(extract).expect("extraction row");
        let (word, base) = (row.rs1_value(), row.rs2_value());
        let imm = i64::try_from(row.imm()).expect("LBU immediate fits");
        let offset = base.wrapping_add_signed(imm);
        let write = |registers: &mut jolt_riscv::RegisterState, value: u64| {
            registers
                .rd
                .as_mut()
                .expect("the extraction writes its destination")
                .post_value = value;
        };
        match attack {
            HONEST | "wide-word" => {}
            "table" => edit_row(&mut rows, extract, |instruction, _| {
                instruction.instruction_kind = Kind::AND;
            }),
            "offset" => edit_row(&mut rows, extract, |_, registers| {
                registers.rs2.as_mut().expect("base read").value = base.wrapping_add(1);
                write(registers, byte(word, offset.wrapping_add(1)));
            }),
            "result" => edit_row(&mut rows, extract, |_, registers| {
                write(registers, byte(word, offset) ^ 1);
            }),
            "base" => edit_row(&mut rows, extract, |instruction, registers| {
                instruction.operands.rs2 = Some(0);
                registers.rs2.as_mut().expect("base read").register = 0;
            }),
            "immediate" => edit_row(&mut rows, extract, |instruction, registers| {
                instruction.operands.imm += 1;
                write(registers, byte(word, offset.wrapping_add(1)));
            }),
            "scratch" => edit_row(&mut rows, extract, |_, registers| {
                registers.rs1.as_mut().expect("scratch read").value =
                    word ^ (1 << (8 * (offset.wrapping_add(1) & 7)));
            }),
            "skip" => {
                let _skipped = rows.remove(extract);
            }
            "enter" => {
                let _skipped = rows.remove(f.load);
            }
            _ => unreachable!("unknown attack {attack}"),
        }
        rows
    }

    /// `2^64 + w` in every oracle table that carries the loaded word, at the
    /// load (RAM read, scratch write) and the extraction (scratch read, left
    /// operand). The compact trace rows are 64-bit, so lookup kernels that
    /// read rows still see `w`.
    struct WideWord<'a> {
        witness: &'a TraceBackend,
        load: usize,
    }

    impl JoltWitnessOracle<Fr> for WideWord<'_> {
        fn shape(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError> {
            <TraceBackend as JoltWitnessOracle<Fr>>::shape(self.witness, id)
        }

        fn oracle_table(&self, id: JoltPolynomialId) -> Result<Vec<Fr>, WitnessError> {
            use JoltVirtualPolynomial as V;
            let mut table =
                <TraceBackend as JoltWitnessOracle<Fr>>::oracle_table(self.witness, id)?;
            let cycle = match id {
                JoltPolynomialId::Virtual(V::RamReadValue | V::RamWriteValue | V::RdWriteValue)
                | JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc) => Some(self.load),
                JoltPolynomialId::Virtual(
                    V::Rs1Value | V::LeftInstructionInput | V::LeftLookupOperand,
                ) => Some(self.load + 1),
                _ => None,
            };
            if let Some(cycle) = cycle {
                let two_32 = Fr::from_u64(1 << 32);
                *table.get_mut(cycle).expect("cycle in domain") += two_32 * two_32;
            }
            Ok(table)
        }

        fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
            <TraceBackend as JoltWitnessOracle<Fr>>::committed_order(self.witness)
        }
    }

    impl RowSource for WideWord<'_> {
        fn visit_chunks(
            &self,
            range: Range<usize>,
            chunk_size: usize,
            visitor: &mut ChunkVisitor<'_>,
        ) -> Result<(), WitnessError> {
            self.witness.visit_chunks(range, chunk_size, visitor)
        }

        fn random_access(&self) -> Option<RandomAccessRows> {
            self.witness.random_access()
        }
    }

    impl ProgramSource for WideWord<'_> {
        fn program_preprocessing(&self) -> &JoltProgramPreprocessing {
            self.witness.program_preprocessing()
        }
    }

    fn prove(f: &Fixture, attack: &str) -> Result<Proof, ProverError<Fr>> {
        let trace = TraceOutput::new(
            TraceData::new(malformed_rows(f, attack)),
            f.trace.device.clone(),
            f.trace.final_memory.clone(),
            f.trace.advice_tape.clone(),
        );
        let program = f.preprocessing.program_arc().expect("full program");
        let config = ProverConfig::derive_from_dimensions::<Fr>(
            trace.dimensions,
            &program.memory_layout,
            program.ram.min_bytecode_address,
            program.ram.bytecode_words.len(),
            program.max_padded_trace_length,
        )
        .expect("derive config");
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(&f.program, &program, trace),
        );
        let backend = JoltBackend::reference();
        let context = if attack == "wide-word" {
            backend.with_witness(WideWord {
                witness: &witness,
                load: f.load,
            })
        } else {
            backend.with_witness(&witness)
        };
        dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
            context,
            &f.preprocessing,
            &config,
            None,
            &f.trace.device,
        )
    }

    fn verify(verifier: &Verifier, io: &JoltDevice, proof: &Proof) -> Result<(), VerifierError> {
        jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, VerifierState<'_, Keccak>>(
            verifier, io, proof, None,
        )
    }

    fn encode<T: serde::Serialize>(value: &T) -> Vec<u8> {
        bincode::serde::encode_to_vec(value, bincode::config::standard()).expect("encode")
    }

    fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> T {
        let (value, consumed) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard()).expect("decode");
        assert_eq!(consumed, bytes.len(), "trailing bytes");
        value
    }

    fn directory() -> PathBuf {
        env::var_os("LBU_PROOF_DIR")
            .expect("LBU_PROOF_DIR names the proof directory")
            .into()
    }

    fn trusted_bytes(f: &Fixture) -> Vec<u8> {
        encode(&(&f.preprocessing.verifier, &f.trace.device))
    }

    /// Proving recurses deeply; run on a wide stack as the ZK suites do.
    #[cfg(not(feature = "zk"))]
    fn wide_stack(body: impl FnOnce() + Send + 'static) {
        thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(body)
            .expect("spawn test thread")
            .join()
            .expect("test thread panicked");
    }

    #[test]
    fn lbu_fixture_honest_proof_is_accepted() {
        wide_stack(|| {
            let f = fixture();
            let proof = prove(&f, HONEST).expect("honest proof");
            verify(&f.preprocessing.verifier, &f.trace.device, &proof)
                .expect("honest proof is accepted");
        });
    }

    #[test]
    #[ignore = "writes the trusted preprocessing for an external adversarial prover"]
    fn prepare_trusted_lbu_fixture() {
        wide_stack(|| {
            let f = fixture();
            fs::write(directory().join("trusted.bin"), trusted_bytes(&f))
                .expect("write trusted fixture");
        });
    }

    /// Writes `<case>.proof`, or `<case>.refused` with the prover's error.
    #[test]
    #[ignore = "needs a prover build without its self-checks"]
    fn produce_lbu_proofs() {
        wide_stack(|| {
            let f = fixture();
            let trusted = fs::read(directory().join("trusted.bin")).expect("trusted fixture");
            assert!(
                trusted == trusted_bytes(&f),
                "this build's preprocessing differs from the trusted fixture"
            );
            for attack in std::iter::once(HONEST).chain(ATTACKS) {
                match prove(&f, attack) {
                    Ok(proof) => {
                        fs::write(directory().join(format!("{attack}.proof")), encode(&proof))
                    }
                    Err(error) => fs::write(
                        directory().join(format!("{attack}.refused")),
                        format!("{error:?}\n"),
                    ),
                }
                .expect("write outcome");
            }
            assert!(
                directory().join(format!("{HONEST}.proof")).exists(),
                "the honest witness produced no proof"
            );
        });
    }

    /// Verifies every produced proof against the trusted preprocessing and
    /// writes `verification.json`. Must run in an unmodified build.
    #[test]
    #[ignore = "consumes proofs from an external adversarial prover"]
    fn verify_lbu_proofs() {
        wide_stack(|| {
            let (verifier, io): (Verifier, JoltDevice) =
                decode(&fs::read(directory().join("trusted.bin")).expect("trusted fixture"));
            let mut outcomes = BTreeMap::new();
            let mut accepted = Vec::new();
            for attack in std::iter::once(HONEST).chain(ATTACKS) {
                let path = directory().join(format!("{attack}.proof"));
                let outcome = if path.exists() {
                    let proof: Proof = decode(&fs::read(&path).expect("read proof"));
                    match verify(&verifier, &io, &proof) {
                        Ok(()) => {
                            if attack != HONEST {
                                accepted.push(attack);
                            }
                            "ACCEPT".to_owned()
                        }
                        Err(error) => format!("REJECT {error:?}"),
                    }
                } else {
                    "NO_PROOF".to_owned()
                };
                let _previous = outcomes.insert(attack, outcome);
            }
            fs::write(
                directory().join("verification.json"),
                serde_json::to_vec_pretty(&outcomes).expect("encode outcomes"),
            )
            .expect("write outcomes");
            assert!(
                outcomes.get(HONEST).map(String::as_str) == Some("ACCEPT"),
                "honest control not accepted: {outcomes:?}"
            );
            assert!(
                accepted.is_empty(),
                "LBU_ACCEPTED_MALICIOUS_WITNESS {accepted:?}"
            );
        });
    }
}
