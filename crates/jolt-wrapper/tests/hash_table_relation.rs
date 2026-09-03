//! The transcript table on a synthetic Jolt-shaped transcript: chain replay
//! against `Blake3Transcript`, the row relation and the wiring zero-check
//! through `jolt_sumcheck::prove_batch`, the exported terms against the
//! native final checks, the verifier-key schedule across runs of the
//! profile, and verifier-path rejection of non-canonical field encodings,
//! foreign randomizers, flipped bits, forged constants, mis-routed copies
//! and wrong kernel strides.

#![expect(clippy::expect_used, clippy::print_stdout)]

use std::collections::HashSet;

use jolt_crypto::Bn254;
use jolt_field::{CanonicalEncoding, Field, Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_poly::EqPolynomial;
use jolt_sumcheck::prover::{prove_batch, SequentialRounds};
use jolt_sumcheck::recorder::ClearSumcheckRecorder;
use jolt_sumcheck::{BatchMember, BatchPrelude};
use jolt_transcript::{
    AppendToTranscript, Blake3Transcript, Label, LabelWithCount, Transcript, U64Word,
};
use jolt_wrapper::hash_table::layout::{D_XOR, MESSAGE};
use jolt_wrapper::hash_table::schedule::preamble;
use jolt_wrapper::hash_table::terms::{
    challenge125, challenge_scalar128, evaluate_terms, fr_word, fr_word_shifted, terms, vk_id,
    wired_word_id, LinkMap, COLUMNS, WIRED_BIT_BASE,
};
use jolt_wrapper::hash_table::wiring::{source, Source, WordSlot, CELL_ROWS, CHALLENGE_POS};
use jolt_wrapper::hash_table::{
    AffineForm, ByteSource, CellIndex, ColumnEvals, Decoder, Event, FinalContext, HashTable,
    HashTableProver, ItemClass, JoltSchedule, Members, PublicInputs, Recorded, RecordingTranscript,
    ScheduleError, StreamColumns, StreamTermExporter, SymbolicSchedule, T1Challenges, VkColumn,
    VkEvals, WiredWord, WiringProver, WordColumn, COMMITTED, DEGREE, MODULUS_HI, WIRED_BITS,
    WIRED_WORDS,
};
use jolt_wrapper::stream::{
    commit_packed, commitment_prefix_challenges, prove_assembly, verify_assembly_with_cost,
    AssemblyMemberStatement, AssemblyStatement, Commitment, CommitmentPhase, StageMember,
    StageMemberSpec, StreamError, TermContext, TermExporter, TermObserver, VerifierCost,
    WrapperProof,
};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

type Recording = RecordingTranscript<Blake3Transcript>;

/// BN254 scalar field modulus `r`, big-endian.
const MODULUS_BE: [u8; 32] = [
    0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29, 0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d,
    0x28, 0x33, 0xe8, 0x48, 0x79, 0xb9, 0x70, 0x91, 0x43, 0xe1, 0xf5, 0x93, 0xf0, 0x00, 0x00, 0x01,
];

/// `bytes + times · r` as a 32-byte big-endian encoding (no overflow for the
/// small values used here).
fn plus_modulus(mut bytes: [u8; 32], times: u8) -> [u8; 32] {
    for _ in 0..times {
        let mut carry = 0u16;
        for i in (0..32).rev() {
            let sum = u16::from(bytes[i]) + u16::from(MODULUS_BE[i]) + carry;
            bytes[i] = sum as u8;
            carry = sum >> 8;
        }
        assert_eq!(carry, 0, "encoding overflow");
    }
    bytes
}

fn canonical_one() -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[31] = 1;
    bytes
}

fn random_bytes(rng: &mut StdRng, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    rng.fill_bytes(&mut bytes);
    bytes
}

/// Replace one sumcheck-round element's 32 absorbed bytes by raw bytes.
#[derive(Clone, Copy)]
struct Tamper {
    round: usize,
    element: usize,
    bytes: [u8; 32],
}

/// A transcript run shaped like `jolt_verifier::verify`: preamble, 384-byte
/// commitments, labeled sumcheck rounds of raw field elements with
/// squeezes, a Dory segment of group elements, and an opening claim after
/// the last squeeze. `preamble_extra` bytes shift the segment like the real
/// 1,046-byte preamble does (a 22-byte tail in the first block).
fn synthetic_log_with(
    seed: u64,
    commitments: usize,
    rounds: usize,
    dory_rounds: usize,
    preamble_extra: usize,
    tamper: Option<Tamper>,
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
        for element in 0..degree {
            match tamper {
                Some(tamper) if tamper.round == round && tamper.element == element => {
                    t.append_bytes(&tamper.bytes);
                }
                _ => Fr::random(&mut rng).append_to_transcript(&mut t),
            }
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

fn synthetic_log(
    seed: u64,
    commitments: usize,
    rounds: usize,
    dory_rounds: usize,
    preamble_extra: usize,
) -> Vec<Recorded> {
    synthetic_log_with(seed, commitments, rounds, dory_rounds, preamble_extra, None)
}

/// A run's verifier key, public inputs, witness schedule and table.
struct Instance {
    key: SymbolicSchedule,
    public: PublicInputs,
    schedule: JoltSchedule,
    table: HashTable,
}

fn instance(log: &[Recorded]) -> Instance {
    let key = SymbolicSchedule::from_reference(log, None).expect("key");
    instance_under(log, key)
}

fn instance_under(log: &[Recorded], key: SymbolicSchedule) -> Instance {
    let public = PublicInputs::from_preamble(&preamble(log), &key).expect("public inputs");
    let schedule = JoltSchedule::witness(log, &key).expect("witness");
    let table = HashTable::build(&schedule, &public);
    Instance {
        key,
        public,
        schedule,
        table,
    }
}

fn random_point(rng: &mut StdRng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Fr::random(rng)).collect()
}

fn random_challenges(rng: &mut StdRng, log_rows: usize) -> T1Challenges {
    T1Challenges::from_challenges(&random_point(rng, T1Challenges::count(log_rows)), log_rows)
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

/// Column evaluation by exported column id: committed, wired, then the
/// verifier-key columns in `VkColumn::ALL` order.
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

struct Run {
    input_claims: [Fr; 2],
    final_claim: Fr,
    challenges: Vec<Fr>,
    evals: ColumnEvals,
}

/// Both members through `prove_batch` with batch coefficients `rho`.
fn run_stage(table: &HashTable, challenges: &T1Challenges, rho: [Fr; 2]) -> Run {
    let log_rows = table.log_rows;
    let relation = challenges.relation();
    let mut members = Members::new(table, &relation, challenges);
    let input_claims = [members.rows.input_claim(), members.wiring.input_claim()];
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
        &mut [&mut members.rows, &mut members.wiring],
        &mut SequentialRounds,
        &mut recorder,
        &mut transcript,
    )
    .expect("stage A");
    Run {
        input_claims,
        final_claim: proved.final_claim,
        challenges: proved.challenges,
        evals: members.rows.column_evals(),
    }
}

struct MulCounter(usize);

impl TermObserver for MulCounter {
    fn fr_mul(&mut self, left: Fr, right: Fr) -> Fr {
        self.0 += 1;
        left * right
    }
}

#[test]
fn chain_replays_and_cells_hold_the_recorded_words() {
    let log = synthetic_log(1, 3, 12, 2, 54);
    let Instance {
        key: symbolic,
        schedule,
        table,
        ..
    } = instance(&log);
    let symbolic = &symbolic;
    assert_eq!(symbolic.squeezes, 12 + 1 + 2 * 2 + 1 + 1);
    // Round degrees cycle 2, 3, 4 over 12 rounds; then the four RLC claims.
    assert_eq!(symbolic.wires, 4 * (2 + 3 + 4) + 4);
    let active = symbolic.active_cells();
    assert_eq!(schedule.table_blocks().len(), 1 << (symbolic.log_rows - 7));
    assert!(active <= schedule.table_blocks().len());

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

// ------------------------------------------------------------ stream harness

const K: usize = 16;
const DIGEST: [u8; 32] = [7; 32];
/// The stream's stage-A encoding is a degree-5 tensor row; T1's members
/// (degree 3) ride under it.
const STAGE_DEGREE: usize = 5;

fn prover_setup(rows: usize) -> HyperKZGProverSetup<Bn254> {
    HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(97),
        rows * K,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    )
}

fn statement(
    key: &SymbolicSchedule,
    public: &PublicInputs,
    columns: &StreamColumns,
    input_claims: [Fr; 2],
) -> AssemblyStatement {
    AssemblyStatement {
        key_digest: DIGEST,
        public_inputs: public.field_elements(),
        rows: key.rows(),
        column_count: columns.columns.len(),
        k: K,
        members: input_claims
            .iter()
            .map(|&input_claim| AssemblyMemberStatement {
                input_claim,
                spec: StageMemberSpec {
                    rounds: key.log_rows,
                    degree: STAGE_DEGREE,
                    offset: 0,
                },
            })
            .collect(),
        commitment_phases: vec![CommitmentPhase {
            group_count: columns.group_count,
            challenge_count: T1Challenges::count(key.log_rows),
        }],
        pinned_commitments: Vec::new(),
    }
}

fn exporter<'a>(
    key: &SymbolicSchedule,
    public: &'a PublicInputs,
    columns: &'a StreamColumns,
) -> StreamTermExporter<'a> {
    StreamTermExporter {
        log_rows: key.log_rows,
        challenge_offset: 0,
        public,
        columns: &columns.ids,
        row_member: 0,
        wiring_member: 1,
    }
}

/// T1's phase challenges as the stream draws them after the commitments.
fn drawn_challenges(
    key: &SymbolicSchedule,
    public: &PublicInputs,
    commitments: &[Commitment],
) -> T1Challenges {
    let count = T1Challenges::count(key.log_rows);
    let drawn =
        commitment_prefix_challenges(&DIGEST, &public.field_elements(), &[(commitments, count)]);
    T1Challenges::from_challenges(&drawn, key.log_rows)
}

/// The stream proof of `table` by the honest prover code with the members'
/// own input claims in the statement — for a tampered table, an adversary's
/// self-consistent proof.
fn prove(
    table: &HashTable,
    key: &SymbolicSchedule,
    public: &PublicInputs,
    setup: &HyperKZGProverSetup<Bn254>,
) -> (WrapperProof, StreamColumns) {
    let columns = StreamColumns::new(table, K, 0);
    let packed = commit_packed(&columns.columns, K, setup).expect("commit");
    let challenges = drawn_challenges(key, public, &packed.commitments);
    let relation = challenges.relation();
    let mut members = Members::new(table, &relation, &challenges);
    let claims = [members.rows.input_claim(), members.wiring.input_claim()];
    let statement = statement(key, public, &columns, claims);
    let exporter = exporter(key, public, &columns);
    let mut stage = [
        StageMember {
            prover: &mut members.rows,
            input_claim: claims[0],
            degree: STAGE_DEGREE,
            offset: 0,
        },
        StageMember {
            prover: &mut members.wiring,
            input_claim: claims[1],
            degree: STAGE_DEGREE,
            offset: 0,
        },
    ];
    let proof = prove_assembly(&packed, &statement, &mut stage, &[&exporter], setup)
        .expect("assembly proof");
    (proof, columns)
}

/// The honest verifier: statement and exporter from the key and public
/// inputs, member claims from the key's wiring constant, challenges from the
/// proof's commitments.
fn verify(
    proof: &WrapperProof,
    key: &SymbolicSchedule,
    public: &PublicInputs,
    columns: &StreamColumns,
    setup: &HyperKZGVerifierSetup<Bn254>,
) -> Result<VerifierCost, StreamError> {
    let challenges = drawn_challenges(key, public, &proof.commitments);
    let statement = statement(key, public, columns, challenges.input_claims(public));
    let exporter = exporter(key, public, columns);
    verify_assembly_with_cost(proof, &statement, &[&exporter], setup).map(|(_, cost)| cost)
}

// ------------------------------------------------------------------- tests

#[test]
fn byte_identities_cover_every_absorbed_byte_once() {
    let log = synthetic_log(2, 4, 20, 3, 54);
    let Instance { key, schedule, .. } = instance(&log);
    let mut covered = HashSet::new();
    let mut public = 0usize;
    for cell in &key.cells {
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
    assert_eq!(public, key.tail_len);
    assert!(public < 64 && public.is_multiple_of(2));
    // The key is a function of the profile: a second run of the same shape
    // with other values has the same key and verifier-key columns.
    let other =
        SymbolicSchedule::from_reference(&synthetic_log(99, 4, 20, 3, 54), None).expect("schedule");
    assert_eq!(key, other);
    assert_eq!(key.vk_columns(), other.vk_columns());
}

#[test]
fn members_hold_and_terms_match_the_native_final_checks() {
    let log = synthetic_log(3, 2, 10, 1, 54);
    let Instance { table, .. } = instance(&log);
    let mut rng = StdRng::seed_from_u64(0x7a11);
    let challenges = random_challenges(&mut rng, table.log_rows);
    let rho = [Fr::random(&mut rng), Fr::random(&mut rng)];
    let run = run_stage(&table, &challenges, rho);
    assert_eq!(
        run.input_claims[0],
        Fr::zero(),
        "the table satisfies the relation"
    );
    assert_eq!(
        run.input_claims,
        challenges.input_claims(&table.public),
        "the wiring sum is the public constant"
    );
    let vk = vk_evals(&table, &run.challenges);
    let relation = challenges.relation();
    let wiring = challenges.wiring();
    let native_rows = relation.final_check(&challenges.tau_rows, &run.challenges, &run.evals);
    let native_wiring = wiring.final_check(
        &challenges.tau_wiring,
        &run.challenges,
        &run.evals,
        &vk,
        &table.public,
    );
    assert_eq!(
        rho[0] * native_rows + rho[1] * native_wiring,
        run.final_claim
    );
    let eval = column_eval(&run.evals, &vk);
    let exported_with = |rows: Fr, wiring: Fr| {
        let ctx = FinalContext {
            challenges: &challenges,
            row_point: &run.challenges,
            rho_rows: rows,
            rho_wiring: wiring,
            public: &table.public,
        };
        terms(&ctx, &mut |a, b| a * b)
    };
    assert_eq!(
        evaluate_terms(&exported_with(Fr::one(), Fr::zero()), &eval),
        native_rows,
        "row-relation terms"
    );
    assert_eq!(
        evaluate_terms(&exported_with(Fr::zero(), Fr::one()), &eval),
        native_wiring,
        "wiring terms"
    );
    let exported = exported_with(rho[0], rho[1]);
    assert_eq!(
        evaluate_terms(&exported, &eval),
        run.final_claim,
        "exported terms reproduce the batched final claim"
    );
    let max_degree = exported.iter().map(|t| t.factors.len()).max().unwrap_or(0);
    assert_eq!(max_degree, 2);
    assert_eq!(exported.len(), COMMITTED + WIRED_BITS + 4 + 1);
    assert!(exported
        .iter()
        .flat_map(|t| &t.factors)
        .flat_map(|f| &f.weights)
        .all(|(id, _)| *id < COLUMNS));
    // The stream exporter derives the same terms from the phase challenges
    // and reports the verifier's field multiplications.
    let raw: Vec<Fr> = challenges
        .tau_rows
        .iter()
        .chain(&challenges.tau_wiring)
        .copied()
        .chain([challenges.relation_gammas[1], challenges.wiring_gammas[1]])
        .collect();
    let columns = StreamColumns::new(&table, K, 0);
    let exporter = exporter(
        &SymbolicSchedule::from_reference(&log, None).expect("key"),
        &table.public,
        &columns,
    );
    let context = TermContext {
        row_point: &run.challenges,
        batching_coefficients: &rho,
        challenges: &raw,
    };
    let mut counter = MulCounter(0);
    let observed = exporter.terms_observed(&context, &mut counter);
    assert_eq!(observed.len(), exported.len());
    assert_eq!(
        observed[0].coefficient, exported[0].coefficient,
        "stream terms are the local terms with physical ids"
    );
    let (kernels, entries, forms) = kernel_counts();
    println!(
        "terms: {} (max degree {max_degree}); verifier Fr multiplications: {}; kernels: {kernels} distinct, {entries} (position, slot) entries, {forms} value forms",
        exported.len(),
        counter.0
    );
    assert!(counter.0 <= 5_000, "Fr multiplication budget");
    assert!(entries <= 1_024 && forms <= 64);
}

/// The wiring zero-check on a tampered table: `Σ_row H(row)` differs from
/// the public input claim.
fn wiring_rejects(challenges: &T1Challenges, table: &HashTable, what: &str) {
    let wiring = challenges.wiring();
    let prover = WiringProver::new(
        &wiring,
        &table.bits,
        &table.wired_bits,
        &table.wired_words,
        &table.vk,
        &table.public,
        challenges.tau_wiring.clone(),
    );
    assert_ne!(
        prover.input_claim(),
        challenges.input_claims(&table.public)[1],
        "{what} must break the wiring check"
    );
}

#[test]
fn tampered_tables_are_rejected() {
    let log = synthetic_log(4, 2, 8, 1, 54);
    let Instance { key, table, .. } = instance(&log);
    let mut rng = StdRng::seed_from_u64(0xbad);
    let challenges = random_challenges(&mut rng, table.log_rows);
    let relation = challenges.relation();
    let active_rows = key.active_cells() * CELL_ROWS;

    // Any state / carry bit flipped breaks the row relation; a round-0
    // message bit flipped breaks the wiring (its wired copies and pins).
    for _ in 0..8 {
        let mut flipped = table.clone();
        let column = rng.gen_range(0..MESSAGE);
        let row = rng.gen_range(0..active_rows);
        flipped.bits[column][row] ^= 1;
        let prover = HashTableProver::new(&relation, &flipped, challenges.tau_rows.clone());
        assert_ne!(
            prover.input_claim(),
            Fr::zero(),
            "flip column {column} row {row}"
        );
    }
    let mut flipped = table.clone();
    flipped.bits[MESSAGE + 9][3 * CELL_ROWS + 7] ^= 1;
    wiring_rejects(&challenges, &flipped, "a flipped round-0 message bit");
    // A wired copy routed from the wrong row (the step before its source).
    let mut routed = table.clone();
    let row = 5 * CELL_ROWS + 17;
    routed.wired_bits[3][row] ^= 1;
    wiring_rejects(&challenges, &routed, "a mis-routed din bit");
    let mut routed = table.clone();
    routed.wired_words[WiredWord::MIn.index()][row] ^= 0x8000_0000;
    wiring_rejects(&challenges, &routed, "a mis-routed message copy");
    // A forged protocol constant: a label byte of a round-0 message word.
    let (label_row, bit) = (0..active_rows)
        .find_map(|r| {
            (r % CELL_ROWS < 16 && table.vk.lo_is_const[r] == 1 && table.vk.lo_const[r] != 0)
                .then_some((r, 0))
        })
        .expect("a constant half-word");
    let mut forged = table.clone();
    forged.bits[MESSAGE + bit][label_row] ^= 1;
    wiring_rejects(&challenges, &forged, "a forged label byte");
    // A forged public preamble byte in the first block.
    let mut forged = table.clone();
    forged.bits[MESSAGE + 3][0] ^= 1;
    wiring_rejects(&challenges, &forged, "a forged preamble byte");
    // A forged block length for the next compression (position 122).
    let mut forged = table.clone();
    forged.bits[MESSAGE][122] ^= 1;
    wiring_rejects(&challenges, &forged, "a forged block length");
    // A forged canonicality witness bit.
    let (wire_row, _) = key
        .wire_rows()
        .iter()
        .map(|(_, row, shifted)| (*row, *shifted))
        .next()
        .expect("a wire");
    let mut forged = table.clone();
    forged.bits[COMMITTED - 1][wire_row] ^= 1;
    wiring_rejects(&challenges, &forged, "a forged canonicality bit");
    // A wrong kernel stride: the verifier's kernels at a shifted position.
    let wiring = challenges.wiring();
    let mut shifted_tau = challenges.tau_wiring.clone();
    shifted_tau.swap(table.log_rows - 1, table.log_rows - 2);
    let run = run_stage(&table, &challenges, [Fr::one(), Fr::one()]);
    let vk = vk_evals(&table, &run.challenges);
    let native = wiring.final_check(
        &shifted_tau,
        &run.challenges,
        &run.evals,
        &vk,
        &table.public,
    );
    let honest = wiring.final_check(
        &challenges.tau_wiring,
        &run.challenges,
        &run.evals,
        &vk,
        &table.public,
    );
    assert_ne!(native, honest, "kernels evaluated at a shifted stride");
    let _ = D_XOR;
}

/// The virtual wire value of 32 absorbed bytes at a wire row of the given
/// alignment, from the bytes alone (what a `CopyLink` would read).
fn wire_value(bytes: &[u8; 32], shifted: bool) -> Fr {
    let mut values = vec![Fr::zero(); COLUMNS];
    let first = if shifted {
        [0, 0, bytes[0], bytes[1]]
    } else {
        bytes[..4].try_into().expect("four bytes")
    };
    let word = u32::from_le_bytes(first);
    for k in 0..32 {
        values[MESSAGE + k] = Fr::from_u64(u64::from((word >> k) & 1));
    }
    for i in 1..8u8 {
        let start = if shifted {
            4 * usize::from(i) - 2
        } else {
            4 * usize::from(i)
        };
        let word = u32::from_be_bytes(bytes[start..start + 4].try_into().expect("four bytes"));
        values[wired_word_id(WiredWord::FrNext(i))] = Fr::from_u32(word);
    }
    if shifted {
        values[wired_word_id(WiredWord::FrTail)] =
            Fr::from_u64(u64::from(u16::from_be_bytes([bytes[30], bytes[31]])));
    }
    let form = if shifted {
        fr_word_shifted()
    } else {
        fr_word()
    };
    form.evaluate(&|id| values[id])
}

#[test]
fn modulus_hi_is_the_top_word_of_the_field_modulus() {
    let mut le = MODULUS_BE;
    le.reverse();
    assert_eq!(Fr::from_bytes_le_reduced(&le), Fr::zero());
    assert_eq!(
        MODULUS_HI,
        u64::from_be_bytes(MODULUS_BE[..8].try_into().expect("eight bytes"))
    );
}

/// B1: the virtual wire value cannot tell `x` from `x + r`; the wiring
/// member's canonicality constraint rejects every non-canonical encoding
/// through the verify path, at both wire alignments.
#[test]
fn noncanonical_wire_bytes_are_rejected() {
    let shape = (2, 10, 1, 54);
    let honest = synthetic_log(3, shape.0, shape.1, shape.2, shape.3);
    let Instance { key, .. } = instance(&honest);
    let setup = prover_setup(key.rows());
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    // Round 0's elements precede the first squeeze (shifted wires); round 1's
    // follow it (aligned).
    for (round, shifted) in [(0, true), (1, false)] {
        for times in [1u8, 2] {
            let bytes = plus_modulus(canonical_one(), times);
            let mut le = bytes;
            le.reverse();
            assert!(Fr::from_bytes_le_checked(&le).is_none());
            assert_eq!(wire_value(&canonical_one(), shifted), Fr::one());
            assert_eq!(
                wire_value(&bytes, shifted),
                Fr::one(),
                "the value column aliases x and x + {times}r"
            );
            let tampered = synthetic_log_with(
                3,
                shape.0,
                shape.1,
                shape.2,
                shape.3,
                Some(Tamper {
                    round,
                    element: 0,
                    bytes,
                }),
            );
            let Instance { public, table, .. } = instance_under(&tampered, key.clone());
            let (proof, columns) = prove(&table, &key, &public, &setup);
            assert!(
                verify(&proof, &key, &public, &columns, &verifier_setup).is_err(),
                "round {round} (shifted {shifted}) encoded as x + {times}r must be rejected"
            );
        }
    }
}

/// B2: the verifier's schedule is the key's, never the proof's — a proof of
/// another run of the profile verifies under the key; a run of another shape
/// is not a witness of the key.
#[test]
fn key_schedule_is_proof_independent() {
    let log_a = synthetic_log(3, 2, 10, 1, 54);
    let log_b = synthetic_log(5, 2, 10, 1, 54);
    let a = instance(&log_a);
    let b = instance_under(&log_b, a.key.clone());
    assert_ne!(a.public, b.public);
    assert_ne!(a.table.bits, b.table.bits);
    let setup = prover_setup(a.key.rows());
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let (proof_a, columns) = prove(&a.table, &a.key, &a.public, &setup);
    let cost = verify(&proof_a, &a.key, &a.public, &columns, &verifier_setup).expect("proof A");
    println!("verifier cost: {cost:?}");
    let (proof_b, columns_b) = prove(&b.table, &a.key, &b.public, &setup);
    assert_eq!(
        columns.ids, columns_b.ids,
        "the column map is a key constant"
    );
    let _ = verify(&proof_b, &a.key, &b.public, &columns, &verifier_setup)
        .expect("proof B under key A");
    // Public inputs of the wrong run reject the proof.
    assert!(verify(&proof_b, &a.key, &a.public, &columns, &verifier_setup).is_err());
    // Other shapes are not witnesses of the key.
    assert!(matches!(
        JoltSchedule::witness(&synthetic_log(6, 3, 10, 1, 54), &a.key),
        Err(ScheduleError::ShapeMismatch)
    ));
    assert!(matches!(
        JoltSchedule::witness(&synthetic_log(6, 2, 11, 1, 54), &a.key),
        Err(ScheduleError::ShapeMismatch)
    ));
    let other_tail = synthetic_log(6, 2, 10, 1, 56);
    assert!(matches!(
        PublicInputs::from_preamble(&preamble(&other_tail), &a.key),
        Err(ScheduleError::PreambleTail { .. })
    ));
}

/// B3: `τ` and the batching challenges are the stream's post-commitment
/// challenges. A prover using its own randomizers cannot link its stage to
/// the exporter's terms; a proof's challenges change with its commitments.
#[test]
fn randomizers_are_bound_to_the_commitments() {
    let log = synthetic_log(3, 2, 10, 1, 54);
    let Instance {
        key, public, table, ..
    } = instance(&log);
    assert_eq!(T1Challenges::count(key.log_rows), 2 * key.log_rows + 2);
    let setup = prover_setup(key.rows());
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let columns = StreamColumns::new(&table, K, 0);
    let packed = commit_packed(&columns.columns, K, &setup).expect("commit");
    let mut rng = StdRng::seed_from_u64(0xc0de);
    let own = random_challenges(&mut rng, key.log_rows);
    let relation = own.relation();
    let mut members = Members::new(&table, &relation, &own);
    let claims = members.input_claims;
    let statement = statement(&key, &public, &columns, claims);
    let exporter = exporter(&key, &public, &columns);
    let mut stage = [
        StageMember {
            prover: &mut members.rows,
            input_claim: claims[0],
            degree: STAGE_DEGREE,
            offset: 0,
        },
        StageMember {
            prover: &mut members.wiring,
            input_claim: claims[1],
            degree: STAGE_DEGREE,
            offset: 0,
        },
    ];
    assert!(
        matches!(
            prove_assembly(&packed, &statement, &mut stage, &[&exporter], &setup),
            Err(StreamError::StageLink)
        ),
        "own randomizers do not link to the transcript's terms"
    );
    // A proof whose commitments are re-ordered re-derives other challenges.
    let (proof, columns) = prove(&table, &key, &public, &setup);
    let _ = verify(&proof, &key, &public, &columns, &verifier_setup).expect("honest proof");
    let mut swapped = proof.clone();
    swapped.commitments.swap(0, 1);
    assert!(verify(&swapped, &key, &public, &columns, &verifier_setup).is_err());
}
