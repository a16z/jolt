//! The transcript table on a synthetic Jolt-shaped transcript: chain
//! replay against `Blake3Transcript`, the row relation through
//! `jolt_sumcheck::prove_batch`, single-bit-flip rejection, link
//! completeness and row counts.

#![expect(clippy::expect_used)]

use std::collections::{HashMap, HashSet};

use jolt_field::{CanonicalEncoding, Field, Fr, One, Zero};
use jolt_sumcheck::prover::{prove_batch, SequentialRounds};
use jolt_sumcheck::recorder::ClearSumcheckRecorder;
use jolt_sumcheck::{BatchMember, BatchPrelude};
use jolt_transcript::{
    AppendToTranscript, Blake3Transcript, Label, LabelWithCount, Transcript, U64Word,
};
use jolt_wrapper::hash_table::blake3::ByteOrigin;
use jolt_wrapper::hash_table::layout::{WordColumn, MESSAGE};
use jolt_wrapper::hash_table::table::{ROWS_PER_BLOCK, ROWS_PER_SQUEEZE_BLOCK};
use jolt_wrapper::hash_table::{
    ColumnEvals, Decoder, Event, HashTable, HashTableProver, ItemClass, JoltSchedule,
    MessageSource, Recorded, RecordingTranscript, Relation, COMMITTED, CONSTRAINTS, DEGREE,
    WIRED_BITS, WIRED_WORDS,
};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

type Recording = RecordingTranscript<Blake3Transcript>;

fn random_bytes(rng: &mut StdRng, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    rng.fill_bytes(&mut bytes);
    bytes
}

/// A transcript run shaped like `jolt_verifier::verify`: preamble, 384-byte
/// commitments, labeled sumcheck rounds of raw field elements with
/// squeezes, a Dory segment of group elements, and an opening claim after
/// the last squeeze.
fn synthetic_log(
    seed: u64,
    commitments: usize,
    rounds: usize,
    dory_rounds: usize,
) -> Vec<Recorded> {
    let mut rng = StdRng::seed_from_u64(seed);
    let _ = Recording::take_log();
    let mut t = Recording::new(b"Jolt");
    t.append(&LabelWithCount(b"preprocessing_digest", 32));
    t.append_bytes(&random_bytes(&mut rng, 32));
    for label in [&b"max_input_size"[..], b"heap_size", b"trace_length"] {
        t.append(&Label(label));
        t.append(&U64Word(rng.next_u64()));
    }
    t.append(&LabelWithCount(b"inputs", 3));
    t.append_bytes(&random_bytes(&mut rng, 3));
    for _ in 0..commitments {
        t.append(&LabelWithCount(b"commitment", 384));
        t.append_bytes(&random_bytes(&mut rng, 384));
    }
    for round in 0..rounds {
        let degree = 2 + round % 3;
        t.append(&LabelWithCount(b"sumcheck_poly", degree as u64));
        for _ in 0..degree {
            Fr::random(&mut rng).append_to_transcript(&mut t);
        }
        if round % 2 == 0 {
            let _: Fr = t.challenge();
        } else {
            let _: Fr = t.challenge_scalar();
        }
        if round % 7 == 3 {
            t.append(&LabelWithCount(b"ram_val_check_gamma", 0));
            t.append_bytes(&[]);
        }
    }
    t.append(&LabelWithCount(b"rlc_claims", 4));
    for _ in 0..4 {
        Fr::random(&mut rng).append_to_transcript(&mut t);
    }
    let _: Fr = t.challenge_scalar();
    for len in [384, 384, 32] {
        t.append(&LabelWithCount(b"dory_serde", len));
        t.append_bytes(&random_bytes(&mut rng, len as usize));
    }
    for _ in 0..dory_rounds {
        for len in [384, 384, 384, 384, 32, 64] {
            t.append(&LabelWithCount(b"dory_serde", len));
            t.append_bytes(&random_bytes(&mut rng, len as usize));
        }
        let _: Fr = t.challenge_scalar();
        for len in [384, 384, 32, 32, 64, 64] {
            t.append(&LabelWithCount(b"dory_serde", len));
            t.append_bytes(&random_bytes(&mut rng, len as usize));
        }
        let _: Fr = t.challenge_scalar();
    }
    let _: Fr = t.challenge_scalar();
    for len in [32, 64] {
        t.append(&LabelWithCount(b"dory_serde", len));
        t.append_bytes(&random_bytes(&mut rng, len as usize));
    }
    let _: Fr = t.challenge_scalar();
    t.append(&LabelWithCount(b"opening_point", 2));
    Fr::random(&mut rng).append_to_transcript(&mut t);
    Fr::random(&mut rng).append_to_transcript(&mut t);
    t.append(&Label(b"opening_eval"));
    Fr::random(&mut rng).append_to_transcript(&mut t);
    Recording::take_log()
}

fn gammas(rng: &mut StdRng) -> Vec<Fr> {
    let gamma = Fr::random(rng);
    std::iter::successors(Some(Fr::one()), |g| Some(*g * gamma))
        .take(CONSTRAINTS)
        .collect()
}

/// Run the row sumcheck through `prove_batch` and return
/// `(input claim, final claim, challenges, column evaluations)`.
fn run_sumcheck(
    relation: &Relation,
    table: HashTable,
    tau: &[Fr],
) -> (Fr, Fr, Vec<Fr>, ColumnEvals) {
    let log_rows = table.log_rows;
    let mut prover = HashTableProver::new(relation, table, tau.to_vec());
    let input_claim = prover.input_claim();
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
    let mut transcript = Blake3Transcript::<Fr>::new(b"hash-table-test");
    let proved = prove_batch(
        &prelude,
        &mut [&mut prover],
        &mut SequentialRounds,
        &mut recorder,
        &mut transcript,
    )
    .expect("row sumcheck");
    (
        input_claim,
        proved.final_claim,
        proved.challenges,
        prover.column_evals(),
    )
}

#[test]
fn chain_replays_the_recorded_transcript_and_lays_out_the_table() {
    let log = synthetic_log(1, 3, 12, 2);
    let schedule = JoltSchedule::new(&log).expect("schedule");
    let blocks = schedule.table_blocks();
    let squeezes = blocks.iter().filter(|b| b.squeeze.is_some()).count();
    assert_eq!(schedule.squeezes, squeezes);
    assert_eq!(schedule.squeezes, 12 + 1 + 2 * 2 + 1 + 1);
    // Round degrees cycle 2, 3, 4 over 12 rounds; then the four RLC claims.
    assert_eq!(schedule.wires, 4 * (2 + 3 + 4) + 4);
    assert!(
        schedule
            .classes
            .iter()
            .all(|c| !matches!(c, ItemClass::Public))
            || schedule.first_commitment_item > 0
    );

    let table = HashTable::build(&schedule.chain.blocks, schedule.blocks.clone(), None);
    assert_eq!(
        table.rows,
        ROWS_PER_BLOCK * blocks.len() + (ROWS_PER_SQUEEZE_BLOCK - ROWS_PER_BLOCK) * squeezes
    );
    assert_eq!(table.bits.len(), COMMITTED);
    assert_eq!(table.wired_bits.len(), WIRED_BITS);
    assert_eq!(table.wired_words.len(), WIRED_WORDS);
    assert_eq!(table.challenges.len(), squeezes);
    assert_eq!(table.state_in, blocks[0].compression.cv);

    // Every compression's chaining rows hold its output words, and every
    // challenge row holds the recorded challenge bytes.
    for (b, block) in blocks.iter().enumerate() {
        for (i, row) in table.chaining_rows(b).into_iter().enumerate() {
            assert_eq!(table.word(WordColumn::DXor, row), block.compression.out[i]);
        }
    }
    for link in &table.challenges {
        let Recorded { event, .. } = &log[link.item as usize];
        let Event::Squeeze { decoder, value } = event else {
            unreachable!("challenge link points at {event:?}")
        };
        let bytes: Vec<u8> = link
            .rows
            .iter()
            .flat_map(|&row| table.word(WordColumn::DXor, row as usize).to_le_bytes())
            .collect();
        let decoded = match decoder {
            Decoder::Challenge125 => Fr::from_challenge_bytes(&bytes),
            Decoder::Scalar128 => Fr::from_scalar_challenge_bytes(&bytes),
        };
        assert_eq!(decoded, *value);
    }
}

#[test]
fn links_cover_every_absorbed_byte_once() {
    let log = synthetic_log(2, 4, 20, 3);
    let schedule = JoltSchedule::new(&log).expect("schedule");
    let table = HashTable::build(&schedule.chain.blocks, schedule.blocks.clone(), None);
    assert_eq!(table.links.len(), 64 * schedule.table_blocks().len());

    let mut covered = HashSet::new();
    let mut padding = 0usize;
    for link in &table.links {
        assert_eq!(
            table.message_sources[link.row as usize],
            MessageSource::First
        );
        match link.origin {
            Some(origin) => assert!(covered.insert(origin), "byte linked twice: {origin:?}"),
            None => padding += 1,
        }
    }
    // Absorbed bytes of the segment: every append from the first commitment
    // label to the last squeeze, minus the preamble bytes that share the
    // first block (public input, covered as well).
    let mut absorbed = 0usize;
    let mut first_block_public = 0usize;
    for (item, recorded) in log.iter().enumerate() {
        if let Event::Append { bytes, .. } = &recorded.event {
            match schedule.classes[item] {
                ItemClass::Outside => {}
                ItemClass::Public => {
                    first_block_public += bytes
                        .iter()
                        .enumerate()
                        .filter(|(offset, _)| {
                            covered.contains(&ByteOrigin {
                                item: item as u32,
                                offset: *offset as u32,
                            })
                        })
                        .count();
                }
                _ => absorbed += bytes.len(),
            }
        }
    }
    assert_eq!(covered.len(), absorbed + first_block_public);
    assert!(first_block_public < 64);
    let blocks = schedule.table_blocks();
    let expected_padding: usize = blocks
        .iter()
        .map(|b| 64 - b.compression.block_len as usize)
        .sum();
    assert_eq!(padding, expected_padding);
    // Each first-use row has exactly four links; every later use copies it.
    let mut per_row = HashMap::new();
    for link in &table.links {
        *per_row.entry(link.row).or_insert(0usize) += 1;
    }
    assert!(per_row.values().all(|&n| n == 4));
    for (row, source) in table.message_sources.iter().enumerate() {
        if let MessageSource::Copy { row: first } = source {
            assert_eq!(table.message_sources[*first as usize], MessageSource::First);
            let word = |r: usize| {
                (0..32).fold(0u32, |acc, k| {
                    acc | u32::from(table.bits[MESSAGE + k][r]) << k
                })
            };
            assert_eq!(word(row), word(*first as usize));
        }
    }
    // Item classes: one wire per raw 32-byte append before Dory, elements by length.
    let elements = schedule
        .classes
        .iter()
        .filter(|c| matches!(c, ItemClass::Element { .. }))
        .count();
    assert_eq!(elements, 4 + 3 + 3 * 12 + 2);
}

#[test]
fn row_relation_holds_and_rejects_single_bit_flips() {
    let log = synthetic_log(3, 2, 10, 1);
    let schedule = JoltSchedule::new(&log).expect("schedule");
    let table = HashTable::build(&schedule.chain.blocks, schedule.blocks.clone(), None);
    let mut rng = StdRng::seed_from_u64(0x7a11);
    let relation = Relation::new(&gammas(&mut rng));
    let tau: Vec<Fr> = (0..table.log_rows).map(|_| Fr::random(&mut rng)).collect();

    let rows = table.rows;
    let (input_claim, final_claim, challenges, evals) =
        run_sumcheck(&relation, table.clone(), &tau);
    assert_eq!(input_claim, Fr::zero(), "the table satisfies the relation");
    assert_eq!(relation.final_check(&tau, &challenges, &evals), final_claim);

    for _ in 0..8 {
        let mut flipped = table.clone();
        let column = rng.gen_range(0..COMMITTED);
        let row = rng.gen_range(0..rows);
        flipped.bits[column][row] ^= 1;
        let prover = HashTableProver::new(&relation, flipped, tau.clone());
        assert_ne!(
            prover.input_claim(),
            Fr::zero(),
            "flipping column {column} row {row} must break the relation"
        );
    }
}
