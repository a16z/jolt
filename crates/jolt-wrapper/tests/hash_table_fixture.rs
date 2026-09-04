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
use jolt_field::{CanonicalEncoding, Field, Fr, Ring, Zero};
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
use jolt_wrapper::hash_table::schedule::preamble;
use jolt_wrapper::hash_table::terms::{
    challenge125, challenge_scalar128, evaluate_terms, fr_word, fr_word_shifted, terms, vk_id,
    LinkMap, WIRED_BIT_BASE,
};
use jolt_wrapper::hash_table::wiring::{source, Source, WordSlot, CELL_ROWS};
use jolt_wrapper::hash_table::{AffineForm, CellIndex};
use jolt_wrapper::hash_table::{
    ByteSource, ColumnEvals, Decoder, Event, FinalContext, HashTable, HashTableProver, ItemClass,
    JoltSchedule, Members, PublicInputs, Recorded, RecordingTranscript, StreamColumns,
    StreamTermExporter, SymbolicSchedule, T1Challenges, VkColumn, VkEvals, WiringProver,
    WordColumn, COMMITTED, CONSTRAINTS, DEGREE, MODULUS_HI, WIRED_BITS, WIRED_WORDS,
};
use jolt_wrapper::stream::{TermContext, TermExporter, TermObserver};
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

fn random_challenges(rng: &mut StdRng, log_rows: usize) -> T1Challenges {
    let raw: Vec<Fr> = (0..T1Challenges::count(log_rows))
        .map(|_| Fr::random(rng))
        .collect();
    T1Challenges::from_challenges(&raw, log_rows)
}

struct MulCounter(usize);

impl TermObserver for MulCounter {
    fn fr_mul(&mut self, left: Fr, right: Fr) -> Fr {
        self.0 += 1;
        left * right
    }
}

/// Distinct wiring kernels `(slot, group, weights)`, `(position, slot)`
/// entries and distinct value forms `(group, weights)` of the position table.
fn kernel_counts() -> (usize, usize, usize) {
    let mut kernels = Vec::new();
    let mut forms = Vec::new();
    let mut entries = 0;
    for p in 0..CELL_ROWS {
        for slot in WordSlot::all() {
            let (group, weights) = match source(p, slot) {
                Source::Cell { group, weights, .. }
                | Source::Previous { group, weights, .. }
                | Source::Next { group, weights, .. } => (group, weights),
                Source::Zero | Source::Const(_) => continue,
            };
            entries += 1;
            if !kernels.contains(&(slot, group, weights)) {
                kernels.push((slot, group, weights));
            }
            if !forms.contains(&(group, weights)) {
                forms.push((group, weights));
            }
        }
    }
    (kernels.len(), entries, forms.len())
}

/// Verifier-key column evaluations at the bound point (MLE of the columns).
fn vk_evals(table: &HashTable, challenges: &[Fr]) -> VkEvals {
    let eq = EqPolynomial::<Fr>::evals(challenges, None);
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
        wire_aligned: mle(&|row| u64::from(table.vk.wire_aligned[row])),
        wire_shifted: mle(&|row| u64::from(table.vk.wire_shifted[row])),
    }
}

fn column_eval<'a>(evals: &'a ColumnEvals, vk: &VkEvals) -> impl Fn(usize) -> Fr + 'a {
    let vk = [
        vk.lo_is_const,
        vk.lo_const,
        vk.hi_is_const,
        vk.hi_const,
        vk.wire_aligned,
        vk.wire_shifted,
    ];
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

    // (i) byte-exact chain: the replay fails on the first state or challenge
    // that differs from the recorded transcript. The key is derived once from
    // the reference run; the proof's run is only checked against it.
    let start = Instant::now();
    let key = SymbolicSchedule::from_reference(&log, None).expect("reference run");
    let public = PublicInputs::from_preamble(&preamble(&log), &key).expect("public inputs");
    let schedule =
        JoltSchedule::witness(&log, &key).expect("chain replays the recorded transcript");
    let replay_secs = start.elapsed().as_secs_f64();
    let symbolic = &key;
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
        symbolic.tail_len
    );
    assert_eq!(elements["CommitmentGt"], 41);
    assert_eq!(elements["DoryGt"], 68);
    assert_eq!(elements["DoryG1"], 35);
    assert_eq!(elements["DoryG2"], 34);
    assert_eq!(symbolic.wires, 1_199);
    assert_eq!(symbolic.squeezes, EXPECTED_SQUEEZES);
    assert_eq!(active, EXPECTED_CELLS);
    assert_eq!(symbolic.log_rows, 18);
    assert_eq!(symbolic.tail_len, 22);
    assert_eq!(public.tail.len(), 22);

    // (iv) shape and identities.
    let start = Instant::now();
    let table = HashTable::build(&schedule, &public);
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
    assert_eq!(links.wires.len() + links.wires_shifted.len(), 1_199);
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
    let (c125, c128) = (challenge125(), challenge_scalar128());
    let (fr, fr_shifted) = (fr_word(), fr_word_shifted());
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
    let mut all_wires: Vec<(u32, usize, bool)> = links
        .wires
        .iter()
        .map(|&(i, r)| (i, r, false))
        .chain(links.wires_shifted.iter().map(|&(i, r)| (i, r, true)))
        .collect();
    all_wires.sort_unstable();
    for (index, row, shifted) in all_wires {
        let (_, recorded) = wire_items.next().expect("wire item");
        let Event::Append { bytes, .. } = &recorded.event else {
            unreachable!()
        };
        let mut le = bytes.clone();
        le.reverse();
        let form = if shifted { &fr_shifted } else { &fr };
        assert_eq!(
            row_eval(form, row),
            Fr::from_bytes_le_reduced(&le),
            "wire {index}"
        );
        // The canonicality constraint (`top 64 bits < r_hi`) is complete for
        // every real wire.
        let top = u64::from_be_bytes(bytes[..8].try_into().expect("eight bytes"));
        assert!(top < MODULUS_HI, "wire {index} is canonical with slack");
    }
    println!(
        "wires: {} aligned + {} shifted",
        links.wires.len(),
        links.wires_shifted.len()
    );

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
    let challenges = random_challenges(&mut rng, table.log_rows);
    let relation = challenges.relation();
    let wiring = challenges.wiring();
    let rho = [Fr::random(&mut rng), Fr::random(&mut rng)];
    let log_rows = table.log_rows;
    let start = Instant::now();
    let mut members = Members::new(&table, &relation, &challenges);
    let members_secs = start.elapsed().as_secs_f64();
    assert_eq!(
        members.rows.input_claim(),
        Fr::zero(),
        "the real table satisfies the relation"
    );
    assert_eq!(
        members.wiring.input_claim(),
        challenges.input_claims(&table.public)[1],
        "the real table satisfies the wiring, canonicality included"
    );
    let prelude = BatchPrelude::new(
        members
            .input_claims
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
        &mut [&mut members.rows, &mut members.wiring],
        &mut SequentialRounds,
        &mut recorder,
        &mut transcript,
    )
    .expect("stage A");
    let stage_secs = start.elapsed().as_secs_f64();
    let evals = members.rows.column_evals();
    let vk = vk_evals(&table, &proved.challenges);
    let native = rho[0] * relation.final_check(&challenges.tau_rows, &proved.challenges, &evals)
        + rho[1]
            * wiring.final_check(
                &challenges.tau_wiring,
                &proved.challenges,
                &evals,
                &vk,
                &table.public,
            );
    assert_eq!(native, proved.final_claim);
    let ctx = FinalContext {
        challenges: &challenges,
        row_point: &proved.challenges,
        rho_rows: rho[0],
        rho_wiring: rho[1],
        public: &table.public,
    };
    let start = Instant::now();
    let exported = terms(&ctx, &mut |a, b| a * b);
    let terms_secs = start.elapsed().as_secs_f64();
    assert_eq!(
        evaluate_terms(&exported, &column_eval(&evals, &vk)),
        proved.final_claim
    );
    // The stream exporter's verifier work: field multiplications for the
    // terms from the phase challenges.
    let raw: Vec<Fr> = challenges
        .tau_rows
        .iter()
        .chain(&challenges.tau_wiring)
        .copied()
        .chain([challenges.relation_gammas[1], challenges.wiring_gammas[1]])
        .collect();
    let columns = StreamColumns::new(&table, 16, 0);
    let exporter = StreamTermExporter {
        log_rows,
        challenge_offset: 0,
        public: &table.public,
        columns: &columns.ids,
        row_member: 0,
        wiring_member: 1,
    };
    let mut counter = MulCounter(0);
    let observed = exporter.terms_observed(
        &TermContext {
            row_point: &proved.challenges,
            batching_coefficients: &rho,
            challenges: &raw,
        },
        &mut counter,
    );
    assert_eq!(observed.len(), exported.len());
    let term_mults = counter.0;
    let mut statement_counter = MulCounter(0);
    let statement_claims = exporter.input_claims(&raw, &mut statement_counter);
    assert_eq!(statement_claims, challenges.input_claims(&table.public));
    let statement_mults = statement_counter.0;
    let (kernels, entries, forms) = kernel_counts();
    println!(
        "stage A (2 members, {} rounds, degree {}): {:.3} s (members setup {:.3} s); terms: T = {} (max degree d = {}), built in {:.3} s; verifier Fr multiplications: terms {} + statement (challenge powers, wiring constant) {} = {}; kernels: {} distinct, {} entries, {} value forms; stream columns {} = {} prover groups + {} verifier-key groups of 16",
        log_rows,
        DEGREE,
        stage_secs,
        members_secs,
        exported.len(),
        exported.iter().map(|t| t.factors.len()).max().unwrap_or(0),
        terms_secs,
        term_mults,
        statement_mults,
        term_mults + statement_mults,
        kernels,
        entries,
        forms,
        columns.columns.len(),
        columns.vk_groups.start,
        columns.vk_groups.len()
    );
    assert_eq!(exported.len(), COMMITTED + 4 + 1);
    assert_eq!(columns.group_count, 22);
    assert_eq!(columns.vk_groups, 20..22);
    assert_eq!((term_mults, statement_mults), (4_206, 705));

    // Any single committed bit flipped breaks a member.
    for (column, row) in [
        (0, 0),
        (D_XOR + 5, 77 * CELL_ROWS + 9),
        (MESSAGE + 31, 1_000 * CELL_ROWS + 3),
    ] {
        let mut flipped = table.clone();
        flipped.bits[column][row] ^= 1;
        let rows = HashTableProver::new(&relation, &flipped, challenges.tau_rows.clone());
        let wires = WiringProver::new(
            &wiring,
            &flipped.bits,
            &flipped.wired_bits,
            &flipped.wired_words,
            &flipped.vk,
            &flipped.public,
            challenges.tau_wiring.clone(),
        );
        assert!(
            rows.input_claim() != Fr::zero()
                || wires.input_claim() != challenges.input_claims(&table.public)[1],
            "flip at column {column} row {row}"
        );
    }
}
