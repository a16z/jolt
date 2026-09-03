//! The transcript table on a synthetic Jolt-shaped transcript: chain replay
//! against `Blake3Transcript`, the row relation and the wiring zero-check
//! through `jolt_sumcheck::prove_batch`, the exported terms against the
//! native final checks, rejection of flipped bits, forged constants,
//! mis-routed copies and wrong kernel strides, and the determinism of the
//! verifier-key columns in the profile.

#![expect(clippy::expect_used, clippy::print_stdout)]

use std::collections::HashSet;

use jolt_field::{CanonicalEncoding, Field, Fr, One, Ring, Zero};
use jolt_poly::EqPolynomial;
use jolt_sumcheck::prover::{prove_batch, SequentialRounds};
use jolt_sumcheck::recorder::ClearSumcheckRecorder;
use jolt_sumcheck::{BatchMember, BatchPrelude};
use jolt_transcript::{
    AppendToTranscript, Blake3Transcript, Label, LabelWithCount, Transcript, U64Word,
};
use jolt_wrapper::hash_table::layout::{D_XOR, MESSAGE};
use jolt_wrapper::hash_table::terms::{
    challenge125, challenge_scalar128, evaluate_terms, fr_word, fr_word_shifted, kernel_counts,
    terms, vk_id, LinkMap, COLUMNS, WIRED_BIT_BASE,
};
use jolt_wrapper::hash_table::wiring::{CELL_ROWS, CHALLENGE_POS};
use jolt_wrapper::hash_table::{AffineForm, CellIndex, ItemClass, SymbolicSchedule};
use jolt_wrapper::hash_table::{
    ByteSource, ColumnEvals, Decoder, Event, FinalContext, HashTable, HashTableProver,
    JoltSchedule, Recorded, RecordingTranscript, Relation, VkColumn, VkEvals, WiredWord,
    WiringProver, WiringStatement, WordColumn, COMMITTED, CONSTRAINTS, DEGREE, WIRED_BITS,
    WIRED_WORDS, WIRING_TERMS,
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
/// the last squeeze. `preamble_extra` bytes shift the segment like the real
/// 1,046-byte preamble does (a 22-byte tail in the first block).
fn synthetic_log(
    seed: u64,
    commitments: usize,
    rounds: usize,
    dory_rounds: usize,
    preamble_extra: usize,
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
    t.append(&LabelWithCount(b"inputs", preamble_extra as u64));
    t.append_bytes(&random_bytes(&mut rng, preamble_extra));
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

fn powers(rng: &mut StdRng, count: usize) -> Vec<Fr> {
    let gamma = Fr::random(rng);
    std::iter::successors(Some(Fr::one()), |g| Some(*g * gamma))
        .take(count)
        .collect()
}

fn random_point(rng: &mut StdRng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Fr::random(rng)).collect()
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

struct Run {
    input_claims: [Fr; 2],
    final_claim: Fr,
    challenges: Vec<Fr>,
    evals: ColumnEvals,
}

/// Both members through `prove_batch` with batch coefficients `rho`.
fn run_stage(
    relation: &Relation,
    wiring: &WiringStatement<'_>,
    table: &HashTable,
    tau_rows: &[Fr],
    tau_wiring: &[Fr],
    rho: [Fr; 2],
) -> Run {
    let log_rows = table.log_rows;
    let mut rows = HashTableProver::new(relation, table, tau_rows.to_vec());
    let mut wires = WiringProver::new(
        wiring,
        &table.bits,
        &table.wired_bits,
        &table.wired_words,
        &table.vk,
        &table.public,
        tau_wiring.to_vec(),
    );
    let input_claims = [rows.input_claim(), wires.input_claim()];
    let prelude = BatchPrelude::new(
        input_claims
            .iter()
            .zip(rho)
            .map(|(claim, coefficient)| BatchMember {
                input_claim: *claim,
                coefficient,
                rounds: log_rows,
                offset: 0,
            })
            .collect(),
        log_rows,
        DEGREE,
    );
    let mut recorder = ClearSumcheckRecorder::<Fr, ()>::new();
    let mut transcript = Blake3Transcript::<Fr>::new(b"hash-table-test");
    let proved = prove_batch(
        &prelude,
        &mut [&mut rows, &mut wires],
        &mut SequentialRounds,
        &mut recorder,
        &mut transcript,
    )
    .expect("stage A");
    Run {
        input_claims,
        final_claim: proved.final_claim,
        challenges: proved.challenges,
        evals: rows.column_evals(),
    }
}

/// Column evaluation by exported column id: committed, wired, then the
/// verifier-key columns.
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
fn chain_replays_and_cells_hold_the_recorded_words() {
    let log = synthetic_log(1, 3, 12, 2, 54);
    let schedule = JoltSchedule::new(&log, None).expect("schedule");
    let symbolic = &schedule.symbolic;
    assert_eq!(symbolic.squeezes, 12 + 1 + 2 * 2 + 1 + 1);
    // Round degrees cycle 2, 3, 4 over 12 rounds; then the four RLC claims.
    assert_eq!(symbolic.wires, 4 * (2 + 3 + 4) + 4);
    let active = symbolic.active_cells();
    assert_eq!(schedule.table_blocks().len(), 1 << (symbolic.log_rows - 7));
    assert!(active <= schedule.table_blocks().len());

    let table = HashTable::build(&schedule);
    assert_eq!(table.bits.len(), COMMITTED);
    assert_eq!(table.wired_bits.len(), WIRED_BITS);
    assert_eq!(table.wired_words.len(), WIRED_WORDS);
    assert_eq!(
        table.public.state_in,
        schedule.table_blocks()[0].compression.cv
    );

    // Chaining rows hold every compression's output words; the first
    // challenge row of a squeeze cell decodes to the recorded challenge.
    for (cell, block) in schedule.table_blocks().iter().enumerate() {
        for (i, row) in table.chaining_rows(CellIndex(cell)).into_iter().enumerate() {
            assert_eq!(table.word(WordColumn::DXor, row), block.compression.out[i]);
        }
    }
    let point_eval = |form: &AffineForm, row: usize| {
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
    let mut checked = 0;
    for (squeeze, row) in symbolic.challenge_rows() {
        let item = log
            .iter()
            .enumerate()
            .filter(|(_, r)| matches!(r.event, Event::Squeeze { .. }))
            .nth(squeeze.index as usize)
            .map(|(i, _)| i)
            .expect("squeeze item");
        let Event::Squeeze { decoder, value } = &log[item].event else {
            unreachable!()
        };
        assert_eq!(*decoder, squeeze.decoder);
        let form = match decoder {
            Decoder::Challenge125 => &c125,
            Decoder::Scalar128 => &c128,
        };
        assert_eq!(point_eval(form, row), *value, "squeeze {}", squeeze.index);
        assert_eq!(row % CELL_ROWS, CHALLENGE_POS);
        checked += 1;
    }
    assert_eq!(checked, symbolic.squeezes as usize);
    // Every wire's row evaluates `fr_word` (or `fr_word_shifted` for the
    // wires absorbed two bytes into their word, before the first squeeze) to
    // the absorbed field element.
    let links = LinkMap::new(symbolic);
    assert_eq!(
        links.wires.len() + links.wires_shifted.len(),
        symbolic.wires as usize
    );
    assert!(
        !links.wires_shifted.is_empty(),
        "the synthetic first round is misaligned"
    );
    let (fr, fr_shifted) = (fr_word(), fr_word_shifted());
    let mut all: Vec<(u32, usize, bool)> = links
        .wires
        .iter()
        .map(|&(i, r)| (i, r, false))
        .chain(links.wires_shifted.iter().map(|&(i, r)| (i, r, true)))
        .collect();
    all.sort_unstable();
    let mut wire_items = log.iter().enumerate().filter(|(i, r)| {
        matches!(schedule.classes[*i], ItemClass::Wire { .. })
            && matches!(r.event, Event::Append { .. })
    });
    for (index, row, shifted) in all {
        let (_, recorded) = wire_items.next().expect("wire item");
        let Event::Append { bytes, .. } = &recorded.event else {
            unreachable!()
        };
        let mut le = bytes.clone();
        le.reverse();
        let form = if shifted { &fr_shifted } else { &fr };
        assert_eq!(
            point_eval(form, row),
            Fr::from_bytes_le_reduced(&le),
            "wire {index}"
        );
    }
}

#[test]
fn byte_identities_cover_every_absorbed_byte_once() {
    let log = synthetic_log(2, 4, 20, 3, 54);
    let schedule = JoltSchedule::new(&log, None).expect("schedule");
    let mut covered = HashSet::new();
    let mut public = 0usize;
    for cell in &schedule.symbolic.cells {
        for source in cell.bytes {
            match source {
                ByteSource::Padding | ByteSource::Constant(_) => {}
                ByteSource::Public { .. } => public += 1,
                other => assert!(covered.insert(other), "byte linked twice: {other:?}"),
            }
        }
    }
    let mut absorbed = 0usize;
    for (item, recorded) in log.iter().enumerate() {
        if let Event::Append { bytes, .. } = &recorded.event {
            match schedule.classes[item] {
                ItemClass::Wire { .. } | ItemClass::Element { .. } => absorbed += bytes.len(),
                _ => {}
            }
        }
    }
    assert_eq!(covered.len(), absorbed);
    assert_eq!(public, schedule.symbolic.tail.len());
    assert!(public < 64 && public.is_multiple_of(2));
    // A second run of the same shape with other values has the same symbolic
    // schedule and verifier-key columns.
    let other = JoltSchedule::new(&synthetic_log(99, 4, 20, 3, 54), None).expect("schedule");
    let strip = |s: &SymbolicSchedule| {
        let mut s = s.clone();
        s.tail.clear();
        s
    };
    assert_eq!(strip(&schedule.symbolic), strip(&other.symbolic));
    assert_eq!(schedule.symbolic.vk_columns(), other.symbolic.vk_columns());
}

struct Setup {
    relation: Relation,
    wiring_gammas: Vec<Fr>,
    tau_rows: Vec<Fr>,
    tau_wiring: Vec<Fr>,
    rho: [Fr; 2],
}

impl Setup {
    fn new(rng: &mut StdRng, log_rows: usize) -> Self {
        Self {
            relation: Relation::new(&powers(rng, CONSTRAINTS)),
            wiring_gammas: powers(rng, WIRING_TERMS),
            tau_rows: random_point(rng, log_rows),
            tau_wiring: random_point(rng, log_rows),
            rho: [Fr::random(rng), Fr::random(rng)],
        }
    }

    fn wiring(&self, log_rows: usize) -> WiringStatement<'_> {
        WiringStatement {
            gammas: &self.wiring_gammas,
            log_rows,
        }
    }
}

#[test]
fn members_hold_and_terms_match_the_native_final_checks() {
    let log = synthetic_log(3, 2, 10, 1, 54);
    let schedule = JoltSchedule::new(&log, None).expect("schedule");
    let table = HashTable::build(&schedule);
    let mut rng = StdRng::seed_from_u64(0x7a11);
    let setup = Setup::new(&mut rng, table.log_rows);
    let wiring = setup.wiring(table.log_rows);
    let run = run_stage(
        &setup.relation,
        &wiring,
        &table,
        &setup.tau_rows,
        &setup.tau_wiring,
        setup.rho,
    );
    assert_eq!(
        run.input_claims[0],
        Fr::zero(),
        "the table satisfies the relation"
    );
    assert_eq!(
        run.input_claims[1],
        wiring.input_claim(&setup.tau_wiring, &table.public),
        "the wiring sum is the public constant"
    );
    let vk = vk_evals(&table, &run.challenges);
    let native_rows = setup
        .relation
        .final_check(&setup.tau_rows, &run.challenges, &run.evals);
    let native_wiring = wiring.final_check(
        &setup.tau_wiring,
        &run.challenges,
        &run.evals,
        &vk,
        &table.public,
    );
    assert_eq!(
        setup.rho[0] * native_rows + setup.rho[1] * native_wiring,
        run.final_claim
    );
    let ctx = FinalContext {
        relation: &setup.relation,
        wiring: &wiring,
        tau_rows: &setup.tau_rows,
        tau_wiring: &setup.tau_wiring,
        challenges: &run.challenges,
        rho_rows: setup.rho[0],
        rho_wiring: setup.rho[1],
        public: &table.public,
    };
    let exported = terms(&ctx);
    let eval = column_eval(&run.evals, &vk);
    let only = |rows: Fr, wiring: Fr| {
        let ctx = FinalContext {
            rho_rows: rows,
            rho_wiring: wiring,
            ..ctx
        };
        evaluate_terms(&terms(&ctx), &eval)
    };
    assert_eq!(
        only(Fr::one(), Fr::zero()),
        native_rows,
        "row-relation terms"
    );
    assert_eq!(only(Fr::zero(), Fr::one()), native_wiring, "wiring terms");
    assert_eq!(
        evaluate_terms(&exported, &eval),
        run.final_claim,
        "exported terms reproduce the batched final claim"
    );
    let max_degree = exported.iter().map(|t| t.factors.len()).max().unwrap_or(0);
    assert_eq!(max_degree, 2);
    assert_eq!(exported.len(), COMMITTED + WIRED_BITS + 2 + 1);
    assert!(exported
        .iter()
        .flat_map(|t| &t.factors)
        .flat_map(|f| &f.weights)
        .all(|(id, _)| *id < COLUMNS));
    let (kernels, entries, forms) = kernel_counts();
    println!(
        "kernels: {kernels} distinct, {entries} (position, slot) entries, {forms} value forms"
    );
    assert!(entries <= 1_024 && forms <= 64);
}

/// The wiring zero-check on a tampered table: `Σ_row H(row)` differs from
/// the public input claim.
fn wiring_rejects(setup: &Setup, table: &HashTable, what: &str) {
    let wiring = setup.wiring(table.log_rows);
    let prover = WiringProver::new(
        &wiring,
        &table.bits,
        &table.wired_bits,
        &table.wired_words,
        &table.vk,
        &table.public,
        setup.tau_wiring.clone(),
    );
    assert_ne!(
        prover.input_claim(),
        wiring.input_claim(&setup.tau_wiring, &table.public),
        "{what} must break the wiring check"
    );
}

#[test]
fn tampered_tables_are_rejected() {
    let log = synthetic_log(4, 2, 8, 1, 54);
    let schedule = JoltSchedule::new(&log, None).expect("schedule");
    let table = HashTable::build(&schedule);
    let mut rng = StdRng::seed_from_u64(0xbad);
    let setup = Setup::new(&mut rng, table.log_rows);
    let active_rows = schedule.symbolic.active_cells() * CELL_ROWS;

    // Any state / carry bit flipped breaks the row relation; a round-0
    // message bit flipped breaks the wiring (its wired copies and pins).
    for _ in 0..8 {
        let mut flipped = table.clone();
        let column = rng.gen_range(0..MESSAGE);
        let row = rng.gen_range(0..active_rows);
        flipped.bits[column][row] ^= 1;
        let prover = HashTableProver::new(&setup.relation, &flipped, setup.tau_rows.clone());
        assert_ne!(
            prover.input_claim(),
            Fr::zero(),
            "flip column {column} row {row}"
        );
    }
    let mut flipped = table.clone();
    flipped.bits[MESSAGE + 9][3 * CELL_ROWS + 7] ^= 1;
    wiring_rejects(&setup, &flipped, "a flipped round-0 message bit");
    // A wired copy routed from the wrong row (the step before its source).
    let mut routed = table.clone();
    let row = 5 * CELL_ROWS + 17;
    routed.wired_bits[3][row] ^= 1;
    wiring_rejects(&setup, &routed, "a mis-routed din bit");
    let mut routed = table.clone();
    routed.wired_words[WiredWord::MIn.index()][row] ^= 0x8000_0000;
    wiring_rejects(&setup, &routed, "a mis-routed message copy");
    // A forged protocol constant: a label byte of a round-0 message word.
    let (label_row, bit) = (0..active_rows)
        .find_map(|r| {
            (r % CELL_ROWS < 16 && table.vk.lo_is_const[r] == 1 && table.vk.lo_const[r] != 0)
                .then_some((r, 0))
        })
        .expect("a constant half-word");
    let mut forged = table.clone();
    forged.bits[MESSAGE + bit][label_row] ^= 1;
    wiring_rejects(&setup, &forged, "a forged label byte");
    // A forged public preamble byte in the first block.
    let mut forged = table.clone();
    forged.bits[MESSAGE + 3][0] ^= 1;
    wiring_rejects(&setup, &forged, "a forged preamble byte");
    // A forged block length for the next compression (position 122).
    let mut forged = table.clone();
    forged.bits[MESSAGE][122] ^= 1;
    wiring_rejects(&setup, &forged, "a forged block length");
    // A wrong kernel stride: the verifier's kernels at a shifted position.
    let wiring = setup.wiring(table.log_rows);
    let mut shifted_tau = setup.tau_wiring.clone();
    shifted_tau.swap(table.log_rows - 1, table.log_rows - 2);
    let run = run_stage(
        &setup.relation,
        &wiring,
        &table,
        &setup.tau_rows,
        &setup.tau_wiring,
        setup.rho,
    );
    let vk = vk_evals(&table, &run.challenges);
    let native = wiring.final_check(
        &shifted_tau,
        &run.challenges,
        &run.evals,
        &vk,
        &table.public,
    );
    let honest = wiring.final_check(
        &setup.tau_wiring,
        &run.challenges,
        &run.evals,
        &vk,
        &table.public,
    );
    assert_ne!(native, honest, "kernels evaluated at a shifted stride");
    let _ = D_XOR;
}
