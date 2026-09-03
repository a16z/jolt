//! The transcript table on a real fibonacci 2^18 proof with
//! `Blake3Transcript`: the chain replay is checked byte-exactly against the
//! verifier's transcript (every state, every challenge, Dory segment
//! included) while the schedule is built, then the table's shape is pinned
//! and the row sumcheck is run at full size.
//!
//! `cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet -E 'binary(hash_table_fixture)' --no-capture`

#![cfg(feature = "prover-fixtures")]
#![expect(clippy::expect_used, clippy::print_stdout)]

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use blake3::Hasher;
use common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T;
use common::jolt_device::{JoltDevice, MemoryLayout};
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::{Field, Fr, One, Zero};
use jolt_program::execution::{JoltProgram, OwnedTrace, TraceOutput, TraceRow};
use jolt_prover::{JoltBackend, JoltProverPreprocessing, ProverConfig};
use jolt_prover_legacy::ark_bn254::Fr as LegacyFr;
use jolt_prover_legacy::curve::Bn254Curve;
use jolt_prover_legacy::host::Program;
use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
use jolt_sumcheck::prover::{prove_batch, SequentialRounds};
use jolt_sumcheck::recorder::ClearSumcheckRecorder;
use jolt_sumcheck::{BatchMember, BatchPrelude};
use jolt_transcript::{Blake3Transcript, Transcript};
use jolt_verifier::proof::JoltProof;
use jolt_verifier::JoltVerifierPreprocessing;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use jolt_wrapper::hash_table::layout::{WordColumn, D_XOR, MESSAGE};
use jolt_wrapper::hash_table::table::{ROWS_PER_BLOCK, ROWS_PER_SQUEEZE_BLOCK};
use jolt_wrapper::hash_table::{
    Event, HashTable, HashTableProver, ItemClass, JoltSchedule, Recorded, RecordingTranscript,
    Relation, COMMITTED, CONSTRAINTS, DEGREE, WIRED_BITS, WIRED_WORDS,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tracer::execution_backend::TracerBackend;

type Pcs = DoryScheme;
type Vc = Pedersen<Bn254G1>;
type Proof = JoltProof<Pcs, Vc>;
type VerifierPreprocessing = JoltVerifierPreprocessing<Pcs, Vc>;
type Recording = RecordingTranscript<Blake3Transcript>;
type LegacyPreprocessing = LegacyProverPreprocessing<LegacyFr, Bn254Curve, DoryCommitmentScheme>;

const TRACE_LENGTH: usize = 1 << 18;
const FIBONACCI_UNITS: u32 = 19_660;
const CACHE: &str = "/Volumes/Dev/scratch/wrapper-fixtures/fibonacci_2_18_blake3.bin";

/// Pinned shape of the fibonacci 2^18 table (L = 18, K = 13, σ = 11): 267
/// compressions absorb the 41 commitments (with the 22-byte preamble tail
/// sharing their first block), 1,017 the hidden segment (stages 1–7 and the
/// stage-8 RLC), 535 the Dory proof elements.
const EXPECTED_BLOCKS: usize = 267 + 1_017 + 535;
const EXPECTED_SQUEEZES: usize = 376;
const EXPECTED_ROWS: usize = ROWS_PER_BLOCK * EXPECTED_BLOCKS
    + (ROWS_PER_SQUEEZE_BLOCK - ROWS_PER_BLOCK) * EXPECTED_SQUEEZES;

fn setup_total_vars(memory_layout: &MemoryLayout, max_padded_trace_length: usize) -> usize {
    let advice_vars = |bytes: u64| -> usize {
        ((bytes / 8) as usize).next_power_of_two().max(1).ilog2() as usize
    };
    let max_log_t = max_padded_trace_length.ilog2() as usize;
    let max_log_k_chunk = if max_log_t >= ONEHOT_CHUNK_THRESHOLD_LOG_T {
        8
    } else {
        4
    };
    (max_log_k_chunk + max_log_t)
        .max(advice_vars(memory_layout.max_trusted_advice_size))
        .max(advice_vars(memory_layout.max_untrusted_advice_size))
}

/// The `dory_byte_diff` fixture recipe: legacy guest build and preprocessing,
/// modular trace, derived config, padded witness, Dory setup, one proof.
fn generate() -> (VerifierPreprocessing, JoltDevice, Proof) {
    let mut program = Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&FIBONACCI_UNITS).expect("serialize inputs");
    let (bytecode, init_memory_state, _, entry_address) = program.decode();
    let trace_output = program
        .trace_with_backend(&mut TracerBackend::new(), &inputs, &[], &[])
        .expect("trace fibonacci");
    let elf_contents = program.get_elf_contents().expect("elf contents");
    let legacy_program =
        LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
            .expect("legacy preprocess");
    let memory_layout = trace_output.device.memory_layout.clone();
    let shared = JoltSharedPreprocessing::new(legacy_program, memory_layout.clone(), TRACE_LENGTH);
    let legacy_preprocessing = LegacyPreprocessing::new(shared);
    let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);
    let program_preprocessing = verifier_preprocessing
        .program
        .as_full_arc()
        .expect("full program preprocessing");
    let jolt_program = Arc::new(JoltProgram::from_elf_bytes(elf_contents));
    println!("trace rows: {}", trace_output.trace.rows().len());
    let config = ProverConfig::derive::<Fr>(
        trace_output.trace.rows(),
        &memory_layout,
        verifier_preprocessing.program.min_bytecode_address(),
        verifier_preprocessing.program.program_image_len_words(),
        TRACE_LENGTH,
    )
    .expect("derive config");
    assert_eq!(config.trace_length, TRACE_LENGTH);
    let public_io = trace_output.device.clone();
    let mut rows = trace_output.trace.rows().to_vec();
    rows.resize(config.trace_length, TraceRow::default());
    let padded = TraceOutput::new(
        OwnedTrace::new(rows),
        trace_output.device,
        trace_output.final_memory,
        trace_output.advice_tape,
    );
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        ),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );
    let preprocessing = JoltProverPreprocessing::<Pcs, Vc> {
        verifier: verifier_preprocessing,
        pcs_setup: Pcs::setup_prover(setup_total_vars(&memory_layout, TRACE_LENGTH)),
        committed_program: None,
    };
    let start = Instant::now();
    let proof = jolt_prover::dory::prove::<Fr, Pcs, Vc, Blake3Transcript, _>(
        &JoltBackend::<Fr, Pcs>::optimized(),
        &preprocessing,
        &config,
        None,
        &witness,
        &public_io,
    )
    .expect("prove fibonacci");
    println!("prove: {:.2} s", start.elapsed().as_secs_f64());
    (preprocessing.verifier, public_io, proof)
}

fn fixture() -> (VerifierPreprocessing, JoltDevice, Proof) {
    let path = PathBuf::from(CACHE);
    if let Ok(bytes) = std::fs::read(&path) {
        let (fixture, _): ((VerifierPreprocessing, JoltDevice, Proof), usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .expect("decode cached fixture");
        println!("fixture: loaded {} ({} B)", path.display(), bytes.len());
        return fixture;
    }
    let fixture = generate();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).expect("fixture cache dir");
    }
    let bytes = bincode::serde::encode_to_vec(&fixture, bincode::config::standard())
        .expect("encode fixture");
    std::fs::write(&path, &bytes).expect("write fixture cache");
    println!(
        "fixture: generated and cached {} ({} B)",
        path.display(),
        bytes.len()
    );
    fixture
}

/// The verifier run through the recording transcript; the proof must verify.
fn record(
    preprocessing: &VerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &Proof,
) -> Vec<Recorded> {
    let _ = Recording::take_log();
    let start = Instant::now();
    jolt_verifier::verify::<Fr, Pcs, Vc, Recording>(preprocessing, public_io, proof, None)
        .expect("proof verifies");
    println!("verify (recorded): {:.3} s", start.elapsed().as_secs_f64());
    Recording::take_log()
}

#[test]
fn fibonacci_2_18_table() {
    let (preprocessing, public_io, proof) = fixture();
    let log = record(&preprocessing, &public_io, &proof);

    // (i) byte-exact chain: `JoltSchedule::new` fails on the first state or
    // challenge that differs from the recorded transcript.
    let start = Instant::now();
    let schedule = JoltSchedule::new(&log).expect("chain replays the recorded transcript");
    let replay_secs = start.elapsed().as_secs_f64();
    let blocks = schedule.table_blocks();
    let squeezes = blocks.iter().filter(|b| b.squeeze.is_some()).count();
    let first_squeeze = blocks
        .iter()
        .position(|b| b.squeeze.is_some())
        .expect("a squeeze");
    let rlc = schedule.rlc_block - schedule.blocks.start;
    println!(
        "compressions: {} = commitments {} + hidden (stages 1–8 RLC) {} + Dory {}; squeezes {}; wires {}; log items {}",
        blocks.len(),
        first_squeeze + 1,
        rlc - first_squeeze,
        blocks.len() - 1 - rlc,
        squeezes,
        schedule.wires,
        log.len()
    );
    let mut elements: BTreeMap<String, usize> = BTreeMap::new();
    let mut public_bytes = 0usize;
    for (item, class) in schedule.classes.iter().enumerate() {
        if let ItemClass::Element { kind, .. } = class {
            *elements.entry(format!("{kind:?}")).or_default() += 1;
        }
        if let (ItemClass::Public, Event::Append { bytes, .. }) = (class, &log[item].event) {
            public_bytes += bytes.len();
        }
    }
    println!("elements: {elements:?}; preamble bytes (native): {public_bytes}");
    assert_eq!(elements["CommitmentGt"], 41);
    assert_eq!(elements["DoryGt"], 68);
    assert_eq!(elements["DoryG1"], 35);
    assert_eq!(elements["DoryG2"], 34);
    assert_eq!(schedule.wires, 1_199);
    assert_eq!(schedule.squeezes, squeezes);

    // (iv) shape.
    let start = Instant::now();
    let table = HashTable::build(&schedule.chain.blocks, schedule.blocks.clone(), None);
    let build_secs = start.elapsed().as_secs_f64();
    println!(
        "table: {} rows (2^{}), {} committed + {} wired bit columns + {} wired words, {} constraints, degree {}; replay {:.3} s, build {:.3} s",
        table.rows, table.log_rows, COMMITTED, WIRED_BITS, WIRED_WORDS, CONSTRAINTS, DEGREE, replay_secs, build_secs
    );
    assert_eq!(blocks.len(), EXPECTED_BLOCKS);
    assert_eq!(squeezes, EXPECTED_SQUEEZES);
    assert_eq!(table.rows, EXPECTED_ROWS);
    assert_eq!(table.log_rows, 18);

    // (iii) links: every absorbed byte of the segment exactly once.
    let mut covered = HashSet::new();
    let mut padding = 0usize;
    let mut by_class: BTreeMap<String, usize> = BTreeMap::new();
    for link in &table.links {
        match link.origin {
            Some(origin) => {
                assert!(covered.insert(origin), "byte linked twice: {origin:?}");
                let class = match schedule.classes[origin.item as usize] {
                    ItemClass::Element { kind, .. } => format!("{kind:?}"),
                    other => format!("{other:?}")
                        .split(' ')
                        .next()
                        .unwrap_or("")
                        .trim_end_matches(" {")
                        .to_string(),
                };
                *by_class.entry(class).or_default() += 1;
            }
            None => padding += 1,
        }
    }
    let mut absorbed = 0usize;
    for (item, recorded) in log.iter().enumerate() {
        if let Event::Append { bytes, .. } = &recorded.event {
            match schedule.classes[item] {
                ItemClass::Outside | ItemClass::Public => {}
                _ => absorbed += bytes.len(),
            }
        }
    }
    let public_in_table = covered
        .iter()
        .filter(|origin| matches!(schedule.classes[origin.item as usize], ItemClass::Public))
        .count();
    println!(
        "links: {} = absorbed {} + preamble tail {} + padding {}; by class {:?}",
        table.links.len(),
        absorbed,
        public_in_table,
        padding,
        by_class
    );
    assert_eq!(covered.len(), absorbed + public_in_table);
    assert_eq!(table.links.len(), 64 * blocks.len());
    assert_eq!(covered.len() + padding, table.links.len());

    // Public outputs: the chaining rows of the RLC-γ and of the final
    // compression hold the keys the transcript continued from (the recorded
    // `state()` after a squeeze is the empty keyed digest under that key).
    let key_after = |block: usize| -> [u8; 32] {
        let rows = table.chaining_rows(block);
        let mut key = [0u8; 32];
        for (i, row) in rows.into_iter().enumerate() {
            key[4 * i..4 * i + 4].copy_from_slice(&table.word(WordColumn::DXor, row).to_le_bytes());
        }
        key
    };
    for (block, name) in [(rlc, "state_rlc"), (blocks.len() - 1, "state_out")] {
        let item = blocks[block].squeeze.expect("squeeze block") as usize;
        let expected = log[item].state;
        let digest = Hasher::new_keyed(&key_after(block)).finalize();
        assert_eq!(
            *digest.as_bytes(),
            expected,
            "{name} keys the recorded transcript"
        );
    }

    // (ii) the row sumcheck at full size, through the batch engine.
    let mut rng = StdRng::seed_from_u64(0xf1b0);
    let gamma = Fr::random(&mut rng);
    let gammas: Vec<Fr> = std::iter::successors(Some(Fr::one()), |g| Some(*g * gamma))
        .take(CONSTRAINTS)
        .collect();
    let relation = Relation::new(&gammas);
    let tau: Vec<Fr> = (0..table.log_rows).map(|_| Fr::random(&mut rng)).collect();
    let log_rows = table.log_rows;
    let start = Instant::now();
    let mut prover = HashTableProver::new(&relation, table, tau.clone());
    let input_claim = prover.input_claim();
    let round0_secs = start.elapsed().as_secs_f64();
    assert_eq!(
        input_claim,
        Fr::zero(),
        "the real table satisfies the relation"
    );
    let prelude = BatchPrelude::new(
        vec![BatchMember {
            input_claim,
            coefficient: Fr::one(),
            rounds: log_rows,
            offset: 0,
        }],
        log_rows,
        DEGREE,
    );
    let mut recorder = ClearSumcheckRecorder::<Fr, ()>::new();
    let mut transcript = Blake3Transcript::<Fr>::new(b"hash-table-fixture");
    let proved = prove_batch(
        &prelude,
        &mut [&mut prover],
        &mut SequentialRounds,
        &mut recorder,
        &mut transcript,
    )
    .expect("row sumcheck");
    let sumcheck_secs = start.elapsed().as_secs_f64();
    let evals = prover.column_evals();
    assert_eq!(
        relation.final_check(&tau, &proved.challenges, &evals),
        proved.final_claim
    );
    println!(
        "row sumcheck: {:.3} s ({} rounds, degree {}; round 0 on bits {:.3} s)",
        sumcheck_secs, log_rows, DEGREE, round0_secs
    );

    // Any single committed bit flipped breaks the relation.
    let table = HashTable::build(&schedule.chain.blocks, schedule.blocks.clone(), None);
    for (column, row) in [
        (0, 0),
        (D_XOR + 5, table.rows / 2),
        (MESSAGE + 31, table.rows - 1),
    ] {
        let mut flipped = table.clone();
        flipped.bits[column][row] ^= 1;
        let prover = HashTableProver::new(&relation, flipped, tau.clone());
        assert_ne!(
            prover.input_claim(),
            Fr::zero(),
            "flip at column {column} row {row}"
        );
    }
}
