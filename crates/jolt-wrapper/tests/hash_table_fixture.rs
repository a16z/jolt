//! The transcript table on a real fibonacci 2^18 proof with
//! `Blake3Transcript`: the chain replay is checked byte-exactly against the
//! verifier's transcript (every state, every challenge, Dory segment
//! included) while the schedule is built, then the table's shape is pinned,
//! both stage-A members run at full size and the exported terms are checked
//! against the native final checks.
//!
//! `cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet -E 'binary(hash_table_fixture)' --no-capture`

#![cfg(feature = "prover-fixtures")]
#![expect(clippy::expect_used, clippy::print_stdout)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use blake3::Hasher;
use common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T;
use common::jolt_device::{JoltDevice, MemoryLayout};
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_poly::EqPolynomial;
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
use jolt_wrapper::hash_table::layout::{D_XOR, MESSAGE};
use jolt_wrapper::hash_table::terms::{
    challenge125, challenge_scalar128, evaluate_terms, fr_word, kernel_counts, terms, vk_id,
    LinkMap, WIRED_BIT_BASE,
};
use jolt_wrapper::hash_table::wiring::CELL_ROWS;
use jolt_wrapper::hash_table::{AffineForm, CellIndex};
use jolt_wrapper::hash_table::{
    ByteSource, ColumnEvals, Decoder, Event, FinalContext, HashTable, HashTableProver, ItemClass,
    JoltSchedule, Recorded, RecordingTranscript, Relation, VkColumn, VkEvals, WiringProver,
    WiringStatement, WordColumn, COMMITTED, CONSTRAINTS, DEGREE, WIRED_BITS, WIRED_WORDS,
    WIRING_TERMS,
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
/// stage-8 RLC), 535 the Dory proof elements; 128 rows per compression.
const EXPECTED_CELLS: usize = 267 + 1_017 + 535;
const EXPECTED_SQUEEZES: u32 = 376;

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

fn powers(rng: &mut StdRng, count: usize) -> Vec<Fr> {
    let gamma = Fr::random(rng);
    std::iter::successors(Some(Fr::one()), |g| Some(*g * gamma))
        .take(count)
        .collect()
}

/// Verifier-key column evaluations at the bound point (MLE of the columns).
fn vk_evals(table: &HashTable, challenges: &[Fr]) -> VkEvals {
    let r: Vec<Fr> = challenges.iter().rev().copied().collect();
    let eq = EqPolynomial::<Fr>::evals(&r, None);
    let mle = |values: &dyn Fn(usize) -> u64| {
        eq.iter().enumerate().fold(Fr::zero(), |acc, (row, w)| {
            acc + *w * Fr::from_u64(values(row))
        })
    };
    VkEvals {
        lo_is_const: mle(&|row| u64::from(table.vk.lo_is_const[row])),
        lo_const: mle(&|row| u64::from(table.vk.lo_const[row])),
        hi_is_const: mle(&|row| u64::from(table.vk.hi_is_const[row])),
        hi_const: mle(&|row| u64::from(table.vk.hi_const[row])),
    }
}

fn column_eval<'a>(evals: &'a ColumnEvals, vk: &VkEvals) -> impl Fn(usize) -> Fr + 'a {
    let vk = [vk.lo_is_const, vk.lo_const, vk.hi_is_const, vk.hi_const];
    move |id: usize| {
        if id < COMMITTED {
            evals.committed[id]
        } else if id < WIRED_BIT_BASE + WIRED_BITS {
            evals.wired_bits[id - WIRED_BIT_BASE]
        } else if id < WIRED_BIT_BASE + WIRED_BITS + WIRED_WORDS {
            evals.wired_words[id - WIRED_BIT_BASE - WIRED_BITS]
        } else {
            vk[id - vk_id(VkColumn::LoIsConst)]
        }
    }
}

#[test]
fn fibonacci_2_18_table() {
    let (preprocessing, public_io, proof) = fixture();
    let log = record(&preprocessing, &public_io, &proof);

    // (i) byte-exact chain: `JoltSchedule::new` fails on the first state or
    // challenge that differs from the recorded transcript.
    let start = Instant::now();
    let schedule = JoltSchedule::new(&log, None).expect("chain replays the recorded transcript");
    let replay_secs = start.elapsed().as_secs_f64();
    let symbolic = &schedule.symbolic;
    let blocks = schedule.table_blocks();
    let active = symbolic.active_cells();
    let first_squeeze = symbolic
        .cells
        .iter()
        .position(|c| c.squeeze.is_some())
        .expect("a squeeze");
    let rlc = symbolic.rlc_cell.0;
    println!(
        "compressions: {} = commitments {} + hidden (stages 1–8 RLC) {} + Dory {}; squeezes {}; wires {}; log items {}; cells 2^{}",
        active,
        first_squeeze + 1,
        rlc - first_squeeze,
        active - 1 - rlc,
        symbolic.squeezes,
        symbolic.wires,
        log.len(),
        symbolic.log_rows - 7
    );
    let mut elements: BTreeMap<String, usize> = BTreeMap::new();
    for class in &schedule.classes {
        if let ItemClass::Element { kind, .. } = class {
            *elements.entry(format!("{kind:?}")).or_default() += 1;
        }
    }
    println!(
        "elements: {elements:?}; preamble tail bytes: {}",
        symbolic.tail.len()
    );
    assert_eq!(elements["CommitmentGt"], 41);
    assert_eq!(elements["DoryGt"], 68);
    assert_eq!(elements["DoryG1"], 35);
    assert_eq!(elements["DoryG2"], 34);
    assert_eq!(symbolic.wires, 1_199);
    assert_eq!(symbolic.squeezes, EXPECTED_SQUEEZES);
    assert_eq!(active, EXPECTED_CELLS);
    assert_eq!(symbolic.log_rows, 18);
    assert_eq!(symbolic.tail.len(), 22);

    // (iv) shape and identities.
    let start = Instant::now();
    let table = HashTable::build(&schedule);
    let build_secs = start.elapsed().as_secs_f64();
    let links = LinkMap::new(symbolic);
    let mut by_kind: BTreeMap<String, usize> = BTreeMap::new();
    for (source, _, _) in &links.bytes {
        let name = match source {
            ByteSource::Element { kind, .. } => format!("{kind:?}"),
            other => format!("{other:?}")
                .split(' ')
                .next()
                .unwrap_or("")
                .to_string(),
        };
        *by_kind.entry(name).or_default() += 1;
    }
    let constants = symbolic
        .cells
        .iter()
        .flat_map(|c| c.bytes)
        .filter(|b| matches!(b, ByteSource::Constant(_)))
        .count();
    let pinned_halves = table
        .vk
        .lo_is_const
        .iter()
        .chain(&table.vk.hi_is_const)
        .filter(|b| **b == 1)
        .count();
    println!(
        "table: 2^{} rows ({} active cells × {}), {} committed + {} wired bits + {} wired words, {} constraints, degree {}; replay {:.3} s, build {:.3} s",
        table.log_rows, active, CELL_ROWS, COMMITTED, WIRED_BITS, WIRED_WORDS, CONSTRAINTS, DEGREE, replay_secs, build_secs
    );
    println!(
        "links: {} wires, {} challenges, {} element/public bytes {:?}; {} constant bytes → {} pinned half-words",
        links.wires.len(), links.challenges.len(), links.bytes.len(), by_kind, constants, pinned_halves
    );
    assert_eq!(links.wires.len(), 1_199);
    assert_eq!(links.challenges.len(), 376);
    assert_eq!(by_kind["CommitmentGt"], 41 * 384);
    assert_eq!(by_kind["DoryGt"], 68 * 384);
    assert_eq!(by_kind["DoryG1"], 35 * 32);
    assert_eq!(by_kind["DoryG2"], 34 * 64);
    assert_eq!(by_kind["Public"], 22);

    // Virtual value columns: every recorded challenge and wire value.
    let row_eval = |form: &AffineForm, row: usize| {
        form.evaluate(&|id| {
            if id < COMMITTED {
                Fr::from_u64(u64::from(table.bits[id][row]))
            } else if id < WIRED_BIT_BASE + WIRED_BITS {
                Fr::from_u64(u64::from(table.wired_bits[id - WIRED_BIT_BASE][row]))
            } else {
                Fr::from_u32(table.wired_words[id - WIRED_BIT_BASE - WIRED_BITS][row])
            }
        })
    };
    let (c125, c128, fr) = (challenge125(), challenge_scalar128(), fr_word());
    let squeeze_items: Vec<usize> = log
        .iter()
        .enumerate()
        .filter(|(_, r)| matches!(r.event, Event::Squeeze { .. }))
        .map(|(i, _)| i)
        .collect();
    for (squeeze, row) in &links.challenges {
        let Event::Squeeze { decoder, value } = &log[squeeze_items[squeeze.index as usize]].event
        else {
            unreachable!()
        };
        let form = match decoder {
            Decoder::Challenge125 => &c125,
            Decoder::Scalar128 => &c128,
        };
        assert_eq!(row_eval(form, *row), *value, "squeeze {}", squeeze.index);
    }
    let mut wire_items = log.iter().enumerate().filter(|(i, r)| {
        matches!(schedule.classes[*i], ItemClass::Wire { .. })
            && matches!(r.event, Event::Append { .. })
    });
    for (index, row) in &links.wires {
        let (_, recorded) = wire_items.next().expect("wire item");
        let Event::Append { bytes, .. } = &recorded.event else {
            unreachable!()
        };
        let mut le = bytes.clone();
        le.reverse();
        assert_eq!(
            row_eval(&fr, *row),
            <Fr as jolt_field::CanonicalEncoding>::from_bytes_le_reduced(&le),
            "wire {index}"
        );
    }

    // Public outputs: the chaining rows of the RLC-γ and of the final
    // compression hold the keys the transcript continued from.
    let key_after = |cell: CellIndex| -> [u8; 32] {
        let mut key = [0u8; 32];
        for (i, row) in table.chaining_rows(cell).into_iter().enumerate() {
            key[4 * i..4 * i + 4].copy_from_slice(&table.word(WordColumn::DXor, row).to_le_bytes());
        }
        key
    };
    for (cell, name) in [
        (symbolic.rlc_cell, "state_rlc"),
        (symbolic.last_squeeze_cell, "state_out"),
    ] {
        let item = blocks[cell.0].squeeze.expect("squeeze block") as usize;
        let digest = Hasher::new_keyed(&key_after(cell)).finalize();
        assert_eq!(
            *digest.as_bytes(),
            log[item].state,
            "{name} keys the recorded transcript"
        );
    }

    // (ii) both members at full size through the batch engine; exported
    // terms against the native final checks.
    let mut rng = StdRng::seed_from_u64(0xf1b0);
    let relation = Relation::new(&powers(&mut rng, CONSTRAINTS));
    let wiring_gammas = powers(&mut rng, WIRING_TERMS);
    let wiring = WiringStatement {
        gammas: &wiring_gammas,
        log_rows: table.log_rows,
    };
    let tau_rows: Vec<Fr> = (0..table.log_rows).map(|_| Fr::random(&mut rng)).collect();
    let tau_wiring: Vec<Fr> = (0..table.log_rows).map(|_| Fr::random(&mut rng)).collect();
    let rho = [Fr::random(&mut rng), Fr::random(&mut rng)];
    let log_rows = table.log_rows;
    let start = Instant::now();
    let mut rows = HashTableProver::new(&relation, &table, tau_rows.clone());
    let rows_round0_secs = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut wires = WiringProver::new(
        &wiring,
        &table.bits,
        &table.wired_bits,
        &table.wired_words,
        &table.vk,
        &table.public,
        tau_wiring.clone(),
    );
    let wiring_setup_secs = start.elapsed().as_secs_f64();
    assert_eq!(
        rows.input_claim(),
        Fr::zero(),
        "the real table satisfies the relation"
    );
    assert_eq!(
        wires.input_claim(),
        wiring.input_claim(&tau_wiring, &table.public),
        "the real table satisfies the wiring"
    );
    let prelude = BatchPrelude::new(
        [rows.input_claim(), wires.input_claim()]
            .into_iter()
            .zip(rho)
            .map(|(input_claim, coefficient)| BatchMember {
                input_claim,
                coefficient,
                rounds: log_rows,
                offset: 0,
            })
            .collect(),
        log_rows,
        DEGREE,
    );
    let mut recorder = ClearSumcheckRecorder::<Fr, ()>::new();
    let mut transcript = Blake3Transcript::<Fr>::new(b"hash-table-fixture");
    let start = Instant::now();
    let proved = prove_batch(
        &prelude,
        &mut [&mut rows, &mut wires],
        &mut SequentialRounds,
        &mut recorder,
        &mut transcript,
    )
    .expect("stage A");
    let stage_secs = start.elapsed().as_secs_f64();
    let evals = rows.column_evals();
    let vk = vk_evals(&table, &proved.challenges);
    let native = rho[0] * relation.final_check(&tau_rows, &proved.challenges, &evals)
        + rho[1] * wiring.final_check(&tau_wiring, &proved.challenges, &evals, &vk, &table.public);
    assert_eq!(native, proved.final_claim);
    let ctx = FinalContext {
        relation: &relation,
        wiring: &wiring,
        tau_rows: &tau_rows,
        tau_wiring: &tau_wiring,
        challenges: &proved.challenges,
        rho_rows: rho[0],
        rho_wiring: rho[1],
        public: &table.public,
    };
    let start = Instant::now();
    let exported = terms(&ctx);
    let terms_secs = start.elapsed().as_secs_f64();
    assert_eq!(
        evaluate_terms(&exported, &column_eval(&evals, &vk)),
        proved.final_claim
    );
    let (kernels, entries, forms) = kernel_counts();
    println!(
        "stage A (2 members, {} rounds, degree {}): {:.3} s (rows round 0 {:.3} s, wiring setup {:.3} s); terms: {} (max degree {}), built in {:.3} s; kernels: {} distinct, {} entries, {} value forms",
        log_rows,
        DEGREE,
        stage_secs,
        rows_round0_secs,
        wiring_setup_secs,
        exported.len(),
        exported.iter().map(|t| t.factors.len()).max().unwrap_or(0),
        terms_secs,
        kernels,
        entries,
        forms
    );

    // Any single committed bit flipped breaks a member.
    for (column, row) in [
        (0, 0),
        (D_XOR + 5, 77 * CELL_ROWS + 9),
        (MESSAGE + 31, 1_000 * CELL_ROWS + 3),
    ] {
        let mut flipped = table.clone();
        flipped.bits[column][row] ^= 1;
        let rows = HashTableProver::new(&relation, &flipped, tau_rows.clone());
        let wires = WiringProver::new(
            &wiring,
            &flipped.bits,
            &flipped.wired_bits,
            &flipped.wired_words,
            &flipped.vk,
            &flipped.public,
            tau_wiring.clone(),
        );
        assert!(
            rows.input_claim() != Fr::zero()
                || wires.input_claim() != wiring.input_claim(&tau_wiring, &table.public),
            "flip at column {column} row {row}"
        );
    }
}
