//! PERF-1: full-statement wrapper-prover profile at `2^18` rows (ignored
//! release gate). Columns: 163 T1 bits + 17 one-hot bits, 54 T2 `u16` chunks,
//! one multiplicity, 18 LogUp helpers, one Spartan witness column; packing
//! `k = 8` (`ell = 21`) and `k = 16` (`ell = 22`). Every phase reports wall
//! seconds, busy threads (`getrusage` CPU / wall), peak RSS and the 1-minute
//! load average; each shape runs `REPEATS` times.
//!
//! ```text
//! CARGO_TARGET_DIR=/Volumes/Dev/cargo-target/perf1 cargo nextest run -p jolt-wrapper --release \
//!   perf1_full_statement_profile --run-ignored ignored-only --cargo-quiet --no-capture
//! CARGO_TARGET_DIR=/Volumes/Dev/cargo-target/perf1 cargo nextest run -p jolt-wrapper --release \
//!   perf2_msm_profile --run-ignored ignored-only --cargo-quiet --no-capture
//! CARGO_TARGET_DIR=/Volumes/Dev/cargo-target/perf1 cargo nextest run -p jolt-wrapper --release \
//!   perf2_commit_open_profile --run-ignored ignored-only --cargo-quiet --no-capture
//! ```

#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    reason = "ignored profiling gate with dimensions fixed by the statement"
)]

use std::fmt::Write as _;
use std::time::Instant;

use ark_bn254::Fq;
use jolt_crypto::ec::bn254::bit_columns::g1_bit_columns_msm;
use jolt_crypto::{Bn254, PairingGroup};
use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::CommitmentScheme;
use jolt_poly::{BindingOrder, MultilinearPoly, Polynomial, UnivariatePoly};
use jolt_r1cs::ConstraintMatrices;
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{
    AppendToTranscript, Blake3Transcript, Keccak256Transcript, Label, LabelWithCount, Transcript,
    U64Word,
};
use jolt_wrapper::hash_table::wiring::CELL_ROWS;
use jolt_wrapper::hash_table::{
    HashTable, HashTableProver, JoltSchedule, Recorded, RecordingTranscript, Relation, CONSTRAINTS,
};
use jolt_wrapper::limb_table::columns::Columns;
use jolt_wrapper::limb_table::program::{Program, Slot};
use jolt_wrapper::limb_table::relation::{RowRelation, RowSumcheck};
use jolt_wrapper::limb_table::wiring::Wiring;
use jolt_wrapper::spartan::{prove_spartan, SpartanPublicInputs};
use jolt_wrapper::stream::{
    commit_packed, prove_kzg_stage, prove_stage, prove_stream, verify_stream_with_cost, Column,
    ColumnReduction, PackedColumns, StageAEncoding, StageMember, TensorStreamStatement, TensorTerm,
};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;

const ROWS_LOG: usize = 18;
const ROWS: usize = 1 << ROWS_LOG;
const SRS_LOG: usize = 22;
const T1_BITS: usize = 163;
const ONE_HOT_BITS: usize = 17;
const T2_CHUNKS: usize = 54;
const T2_HELPERS: usize = 18;
const T2_SLOTS: usize = 12;
const SPARTAN_LOG: usize = 14;
const REPEATS: usize = 3;

// ---------------------------------------------------------------- metering

fn cpu_seconds() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage fills the provided rusage struct for the calling process.
    let usage = unsafe {
        assert_eq!(libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()), 0);
        usage.assume_init()
    };
    let seconds = |time: libc::timeval| time.tv_sec as f64 + time.tv_usec as f64 * 1e-6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

fn max_rss_mib() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: as above.
    let usage = unsafe {
        assert_eq!(libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()), 0);
        usage.assume_init()
    };
    usage.ru_maxrss as f64 / (1u64 << 20) as f64
}

fn load_average() -> f64 {
    let mut loads = [0f64; 3];
    // SAFETY: getloadavg writes at most three doubles into the buffer.
    let filled = unsafe { libc::getloadavg(loads.as_mut_ptr(), 3) };
    assert!(filled >= 1);
    loads[0]
}

struct Sample {
    name: String,
    wall: f64,
    busy: f64,
    rss: f64,
    load: f64,
}

struct Report {
    samples: Vec<Sample>,
    log: String,
}

impl Report {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            log: String::new(),
        }
    }

    fn measure<T>(&mut self, name: &str, f: impl FnOnce() -> T) -> T {
        let load = load_average();
        let cpu = cpu_seconds();
        let start = Instant::now();
        let value = f();
        let wall = start.elapsed().as_secs_f64();
        let busy = (cpu_seconds() - cpu) / wall.max(1e-9);
        let rss = max_rss_mib();
        let line = format!(
            "{name:<44} {:>9.1} ms  busy {busy:>5.2}  rss {rss:>7.0} MiB  load {load:.2}",
            wall * 1e3
        );
        println!("{line}");
        writeln!(self.log, "{line}").expect("write log");
        self.samples.push(Sample {
            name: name.to_owned(),
            wall,
            busy,
            rss,
            load,
        });
        value
    }

    fn note(&mut self, line: &str) {
        println!("{line}");
        writeln!(self.log, "{line}").expect("write log");
    }

    /// One row per distinct phase: min / median wall, busy threads at the median run.
    fn summary(&self) -> String {
        let mut names: Vec<&str> = Vec::new();
        for sample in &self.samples {
            if !names.contains(&sample.name.as_str()) {
                names.push(&sample.name);
            }
        }
        let mut out = String::new();
        writeln!(
            out,
            "{:<44} {:>9} {:>9} {:>6} {:>8} {:>5} {:>3}",
            "phase", "min ms", "med ms", "busy", "rss MiB", "load", "n"
        )
        .expect("write summary");
        for name in names {
            let mut runs: Vec<&Sample> = self
                .samples
                .iter()
                .filter(|sample| sample.name == name)
                .collect();
            runs.sort_by(|a, b| a.wall.total_cmp(&b.wall));
            let median = runs[runs.len() / 2];
            writeln!(
                out,
                "{name:<44} {:>9.1} {:>9.1} {:>6.2} {:>8.0} {:>5.2} {:>3}",
                runs[0].wall * 1e3,
                median.wall * 1e3,
                median.busy,
                runs.iter().map(|run| run.rss).fold(0.0, f64::max),
                median.load,
                runs.len()
            )
            .expect("write summary");
        }
        out
    }
}

// ---------------------------------------------------------------- witness shapes

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// The full-statement column set in packing order: bits, `u16` chunks, then
/// the field columns (multiplicity, LogUp helpers, Spartan witness).
fn full_statement_columns() -> Vec<Column> {
    let mut columns: Vec<Column> = (0..T1_BITS)
        .into_par_iter()
        .map(|column| {
            Column::Bits(
                (0..ROWS)
                    .map(|row| (mix(row as u64 ^ ((column as u64) << 32)) & 1) as u8)
                    .collect(),
            )
        })
        .collect();
    columns.par_extend((0..ONE_HOT_BITS).into_par_iter().map(|column| {
        Column::Bits(
            (0..ROWS)
                .map(|row| u8::from(mix(row as u64) % ONE_HOT_BITS as u64 == column as u64))
                .collect(),
        )
    }));
    columns.par_extend((0..T2_CHUNKS).into_par_iter().map(|column| {
        Column::U16(
            (0..ROWS)
                .map(|row| mix(row as u64 ^ ((column as u64 + 1000) << 32)) as u16)
                .collect(),
        )
    }));
    columns.push(Column::Fr(
        (0..ROWS)
            .map(|row| Fr::from_u64(if row < 1 << 16 { 54 * 4 } else { 0 }))
            .collect(),
    ));
    columns.par_extend((0..T2_HELPERS).into_par_iter().map(|column| {
        let mut rng = StdRng::seed_from_u64(0x5e1f + column as u64);
        Column::Fr((0..ROWS).map(|_| Fr::random(&mut rng)).collect())
    }));
    let mut rng = StdRng::seed_from_u64(0x57a7);
    columns.push(Column::Fr(
        (0..ROWS)
            .map(|row| {
                if row < 1 << SPARTAN_LOG {
                    Fr::random(&mut rng)
                } else {
                    Fr::zero()
                }
            })
            .collect(),
    ));
    assert_eq!(columns.len(), 254);
    columns
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GroupKind {
    Bits,
    U16,
    Mixed,
}

fn group_kinds(columns: &[Column], k: usize) -> Vec<GroupKind> {
    columns
        .chunks(k)
        .map(|group| {
            if group.iter().all(|column| matches!(column, Column::Bits(_))) {
                GroupKind::Bits
            } else if group.iter().all(|column| matches!(column, Column::U16(_))) {
                GroupKind::U16
            } else {
                GroupKind::Mixed
            }
        })
        .collect()
}

fn packed_bits(columns: &[Column], group: usize, k: usize) -> Vec<u8> {
    let mut packed = vec![0u8; ROWS * k];
    for slot in 0..k {
        if let Some(Column::Bits(column)) = columns.get(group * k + slot) {
            for (row, &bit) in column.iter().enumerate() {
                packed[row * k + slot] = bit;
            }
        }
    }
    packed
}

fn packed_u16(columns: &[Column], group: usize, k: usize) -> Vec<u16> {
    let mut packed = vec![0u16; ROWS * k];
    for slot in 0..k {
        if let Some(Column::U16(column)) = columns.get(group * k + slot) {
            for (row, &value) in column.iter().enumerate() {
                packed[row * k + slot] = value;
            }
        }
    }
    packed
}

/// Commit kernels by group kind, measured separately from `commit_packed`.
fn profile_commit_kernels(
    report: &mut Report,
    columns: &[Column],
    packed: &PackedColumns,
    k: usize,
    setup: &HyperKZGProverSetup<Bn254>,
) {
    let kinds = group_kinds(columns, k);
    let n = ROWS * k;
    let bases = &setup.g1_powers()[..n];
    let bit_groups: Vec<usize> = (0..kinds.len())
        .filter(|&g| kinds[g] == GroupKind::Bits)
        .collect();
    let u16_groups: Vec<usize> = (0..kinds.len())
        .filter(|&g| kinds[g] == GroupKind::U16)
        .collect();
    let mixed_groups: Vec<usize> = (0..kinds.len())
        .filter(|&g| kinds[g] == GroupKind::Mixed)
        .collect();
    report.note(&format!(
        "k={k}: {} bit groups, {} u16 groups, {} mixed/Fr groups (N = 2^{})",
        bit_groups.len(),
        u16_groups.len(),
        mixed_groups.len(),
        n.trailing_zeros()
    ));
    let bits: Vec<Vec<u8>> = bit_groups
        .par_iter()
        .map(|&g| packed_bits(columns, g, k))
        .collect();
    let refs: Vec<&[u8]> = bits.iter().map(Vec::as_slice).collect();
    let bit_commitments = report.measure(
        &format!("commit  bit groups x{} (bit kernel)", bits.len()),
        || g1_bit_columns_msm(bases, &refs),
    );
    for (commitment, &g) in bit_commitments.iter().zip(&bit_groups) {
        assert_eq!(*commitment, packed.commitments[g].point());
    }
    let chunks: Vec<Vec<u16>> = u16_groups
        .par_iter()
        .map(|&g| packed_u16(columns, g, k))
        .collect();
    let u16_commitments = report.measure(
        &format!("commit  u16 groups x{} (small-scalar)", chunks.len()),
        || {
            chunks
                .par_iter()
                .map(|packed| Bn254::g1_affine_msm_small(bases, packed))
                .collect::<Vec<_>>()
        },
    );
    for (commitment, &g) in u16_commitments.iter().zip(&u16_groups) {
        assert_eq!(*commitment, packed.commitments[g].point());
    }
    let mixed_commitments = report.measure(
        &format!(
            "commit  mixed/Fr groups x{} (Pippenger)",
            mixed_groups.len()
        ),
        || {
            mixed_groups
                .par_iter()
                .map(|&g| {
                    HyperKZGScheme::<Bn254>::commit(packed.evaluations[g].as_slice(), setup)
                        .expect("commit group")
                        .0
                })
                .collect::<Vec<_>>()
        },
    );
    for (commitment, &g) in mixed_commitments.iter().zip(&mixed_groups) {
        assert_eq!(*commitment, packed.commitments[g]);
    }
    let mut rng = StdRng::seed_from_u64(0xf00d);
    let full: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let _ = report.measure("msm     one full-width N-point Pippenger", || {
        Bn254::g1_affine_msm(bases, &full)
    });
    let half: Vec<Fr> = full[..n / 2].to_vec();
    let _ = report.measure("msm     one full-width N/2-point Pippenger", || {
        Bn254::g1_affine_msm(&bases[..n / 2], &half)
    });
}

// ---------------------------------------------------------------- T1 (Blake3 transcript table)

type Recording = RecordingTranscript<Blake3Transcript>;

fn random_bytes(rng: &mut StdRng, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    rng.fill_bytes(&mut bytes);
    bytes
}

/// A transcript run shaped like `jolt_verifier::verify` (the
/// `hash_table_relation` synthetic log): preamble, 384-byte commitments,
/// labeled sumcheck rounds with squeezes, a Dory segment, an opening claim.
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

fn t1_rows(rounds: usize) -> usize {
    let schedule = JoltSchedule::new(&synthetic_log(1, 41, rounds, 11), None).expect("schedule");
    schedule.symbolic.active_cells() * CELL_ROWS
}

/// The largest synthetic schedule that fits `2^18` rows (the real fibonacci
/// 2^18 table is 219,784 rows).
fn t1_table() -> HashTable {
    let (low, high) = (100, 300);
    let per_round = (t1_rows(high) - t1_rows(low)) as f64 / (high - low) as f64;
    let mut rounds = low + ((219_784 - t1_rows(low)) as f64 / per_round) as usize;
    while t1_rows(rounds) > ROWS {
        rounds -= 4;
    }
    let schedule =
        JoltSchedule::new(&synthetic_log(1, 41, rounds, 11), Some(ROWS_LOG)).expect("schedule");
    HashTable::build(&schedule)
}

/// Pure sumcheck rounds through a Blake3 transcript: no round commitments.
fn raw_rounds(prover: &mut dyn ProveRounds<Fr>, input_claim: Fr) {
    let mut transcript = Blake3Transcript::<Fr>::new(b"perf1-raw");
    let mut bind = None;
    let mut claim = input_claim;
    for round in 0..prover.num_rounds() {
        let polynomial = prover.prove_round(bind, round, claim).expect("round");
        let challenge: Fr = transcript.challenge();
        claim = polynomial.evaluate(challenge);
        bind = Some(challenge);
    }
    prover
        .finish_rounds(bind.expect("at least one round"))
        .expect("finish");
}

fn t1_profile(report: &mut Report, table: &HashTable, setup: &HyperKZGProverSetup<Bn254>) {
    let mut rng = StdRng::seed_from_u64(0x71);
    let gamma = Fr::random(&mut rng);
    let gammas: Vec<Fr> = std::iter::successors(Some(Fr::one()), |g| Some(*g * gamma))
        .take(CONSTRAINTS)
        .collect();
    let relation = Relation::new(&gammas);
    let tau: Vec<Fr> = (0..ROWS_LOG).map(|_| Fr::random(&mut rng)).collect();
    let mut prover = report.measure("T1      construct (round 0 on bit columns)", || {
        HashTableProver::new(&relation, table, tau.clone())
    });
    let claim = prover.input_claim();
    report.measure("T1      rounds 1..18, clear (no KZG)", || {
        raw_rounds(&mut prover, claim);
    });
    drop(prover);
    let mut prover = HashTableProver::new(&relation, table, tau);
    let claim = prover.input_claim();
    let mut transcript = Keccak256Transcript::<Fr>::new(b"perf1-t1");
    let _ = report.measure("T1      rounds 1..18 + KZG round commits + BDFG", || {
        prove_kzg_stage(&mut prover, claim, 5, setup, &mut transcript).expect("T1 stage")
    });
}

// ---------------------------------------------------------------- T2 (limb table)

struct T2Witness {
    program: Program,
    columns: Columns,
    helpers: Vec<Vec<Fr>>,
    multiplicities: Vec<u32>,
    alpha: Fr,
}

/// `2^18` compute rows of twelve products over random earlier rows.
fn t2_witness() -> T2Witness {
    let mut rng = StdRng::seed_from_u64(0x72);
    let constants_start = (ROWS - 2) as u32;
    let mut program = Program::new(constants_start..ROWS as u32);
    let input_count = 16;
    for index in 0..input_count {
        let _ = program.input(index);
    }
    let kappas = [1, -1, 2, -2, 3, -3];
    while program.cursor() < constants_start {
        let id = program.cursor();
        let slots = (0..T2_SLOTS)
            .map(|_| Slot {
                x: rng.gen_range(0..id),
                y: rng.gen_range(0..id),
                kappa: kappas[rng.gen_range(0..kappas.len())],
            })
            .collect();
        let _ = program.compute(slots);
    }
    let inputs: Vec<Fq> = (0..input_count).map(|_| Fq::from(rng.next_u64())).collect();
    let values = program.evaluate(&inputs).expect("evaluate program");
    let columns = Columns::generate(&program, &values, ROWS_LOG);
    let alpha = Fr::random(&mut rng);
    let (helpers, multiplicities) = columns.logup_columns(alpha);
    assert_eq!(helpers.len(), T2_HELPERS);
    T2Witness {
        program,
        columns,
        helpers,
        multiplicities,
        alpha,
    }
}

fn t2_profile(report: &mut Report, witness: &T2Witness, setup: &HyperKZGProverSetup<Bn254>) {
    let mut rng = StdRng::seed_from_u64(0x73);
    let tau: Vec<Fr> = (0..ROWS_LOG).map(|_| Fr::random(&mut rng)).collect();
    let relation = RowRelation::new(
        ROWS_LOG,
        T2_SLOTS,
        witness.alpha,
        tau,
        Fr::random(&mut rng),
        Fr::random(&mut rng),
    );
    let wiring = Wiring {
        program: &witness.program,
        num_slots: T2_SLOTS,
    };
    let mut prover = report.measure("T2      construct (row matrix, 150 Fr/row)", || {
        RowSumcheck::new(
            &relation,
            &witness.program,
            &witness.columns,
            &witness.helpers,
            &witness.multiplicities,
            &wiring,
        )
    });
    let claim = report.measure("T2      input claim", || prover.input_claim());
    let mut transcript = Keccak256Transcript::<Fr>::new(b"perf1-t2");
    let _ = report.measure("T2      rounds 0..18 + KZG round commits + BDFG", || {
        prove_kzg_stage(&mut prover, claim, 5, setup, &mut transcript).expect("T2 stage")
    });
}

// ---------------------------------------------------------------- Spartan (2^14)

fn spartan_instance(log_size: usize) -> (ConstraintMatrices<Fr>, Vec<Fr>) {
    let size = 1usize << log_size;
    let quarter = size / 4;
    let mut witness = vec![Fr::zero(); size];
    for index in 0..quarter {
        let a_value = Fr::from_u64(index as u64 + 2);
        let b_value = Fr::from_u64(index as u64 * 3 + 5);
        witness[index] = a_value;
        witness[quarter + index] = b_value;
        witness[2 * quarter + index] = a_value * b_value;
        witness[3 * quarter + index] = Fr::from_u64(index as u64 * 11 + 1);
    }
    let witness_start = 1;
    let mut a = Vec::with_capacity(size);
    let mut b = Vec::with_capacity(size);
    let mut c = Vec::with_capacity(size);
    for row in 0..size {
        let index = mix(row as u64) as usize & (quarter - 1);
        a.push(vec![(witness_start + index, Fr::one())]);
        b.push(vec![(witness_start + quarter + index, Fr::one())]);
        c.push(vec![(witness_start + 2 * quarter + index, Fr::one())]);
    }
    (
        ConstraintMatrices::new(size, witness_start + size, a, b, c),
        witness,
    )
}

fn spartan_profile(report: &mut Report, setup: &HyperKZGProverSetup<Bn254>) {
    let (r1cs, witness) = spartan_instance(SPARTAN_LOG);
    let _ = report.measure("Spartan outer+inner (+own 2^14 W commit/open)", || {
        prove_spartan(
            &[7; 32],
            &r1cs,
            SpartanPublicInputs {
                known: &[],
                challenges: &[],
            },
            &witness,
            setup,
        )
        .expect("spartan")
    });
}

// ---------------------------------------------------------------- stream: stage B, RLC, opening

/// The gate's synthetic degree-five row prover over five bit columns.
struct TimingRow {
    columns: Vec<Polynomial<Fr>>,
    rounds: usize,
    claim: Fr,
}

impl TimingRow {
    fn new(columns: Vec<Vec<Fr>>) -> Self {
        let rows = columns[0].len();
        let claim = (0..rows)
            .into_par_iter()
            .map(|row| columns.iter().map(|column| column[row]).product::<Fr>())
            .sum();
        Self {
            columns: columns.into_iter().map(Polynomial::new).collect(),
            rounds: rows.trailing_zeros() as usize,
            claim,
        }
    }

    fn bind(&mut self, challenge: Fr) {
        for column in &mut self.columns {
            column.bind_with_order(challenge, BindingOrder::HighToLow);
        }
    }
}

impl ProveRounds<Fr> for TimingRow {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.columns[0].len() / 2;
        let evaluations: Vec<Fr> = (0..=5)
            .map(|x| {
                let x = Fr::from_u64(x);
                (0..half)
                    .into_par_iter()
                    .map(|row| {
                        self.columns
                            .iter()
                            .map(|column| column.sumcheck_round_eval(row, x))
                            .product::<Fr>()
                    })
                    .sum()
            })
            .collect();
        if evaluations[0] + evaluations[1] != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: evaluations[0] + evaluations[1],
            });
        }
        Ok(UnivariatePoly::from_evals(&evaluations))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}

fn rlc(polynomials: &[Vec<Fr>], weights: &[Fr]) -> Vec<Fr> {
    (0..polynomials[0].len())
        .into_par_iter()
        .map(|index| {
            polynomials
                .iter()
                .zip(weights)
                .map(|(polynomial, &weight)| polynomial[index] * weight)
                .sum()
        })
        .collect()
}

fn horner(f: &[Fr], u: Fr) -> Fr {
    f.iter().rev().fold(Fr::zero(), |acc, &c| acc * u + c)
}

fn vanishing(u: &[Fr; 3]) -> [Fr; 4] {
    [
        -(u[0] * u[1] * u[2]),
        u[0] * u[1] + u[0] * u[2] + u[1] * u[2],
        -(u[0] + u[1] + u[2]),
        Fr::one(),
    ]
}

fn divide_cubic(f: &[Fr], divisor: &[Fr; 4]) -> Vec<Fr> {
    let mut quotient = vec![Fr::zero(); f.len() - 3];
    for i in (0..quotient.len()).rev() {
        let mut coefficient = f[i + 3];
        for offset in 1..=3 {
            if let Some(next) = quotient.get(i + offset) {
                coefficient -= divisor[3 - offset] * *next;
            }
        }
        quotient[i] = coefficient;
    }
    quotient
}

/// The HyperKZG opening split into its passes (same arithmetic as
/// `HyperKZGScheme::open`, timed piecewise).
fn open_replica(
    report: &mut Report,
    setup: &HyperKZGProverSetup<Bn254>,
    evals: &[Fr],
    point: &[Fr],
) {
    let ell = point.len();
    let polys = report.measure("open    1. fold (ell-1 halvings)", || {
        let mut polys = vec![evals.to_vec()];
        for &xi in point.iter().skip(1).rev() {
            let prev = polys.last().expect("seeded");
            let next: Vec<Fr> = prev
                .par_chunks_exact(2)
                .map(|pair| pair[0] + xi * (pair[1] - pair[0]))
                .collect();
            polys.push(next);
        }
        polys
    });
    assert_eq!(polys.len(), ell);
    let _ = report.measure("open    2. fold commitments (ell-1 MSMs, N-2 pts)", || {
        polys
            .par_iter()
            .skip(1)
            .map(|p| Bn254::g1_affine_msm(&setup.g1_powers()[..p.len()], p))
            .collect::<Vec<_>>()
    });
    let mut rng = StdRng::seed_from_u64(0x0c);
    let r = Fr::random(&mut rng);
    let u = [r, -r, r * r];
    let _ = report.measure("open    3. evaluations r,-r,r^2 (serial reference)", || {
        polys
            .par_iter()
            .map(|f| u.map(|ui| horner(f, ui)))
            .collect::<Vec<_>>()
    });
    let q = Fr::random(&mut rng);
    let b = report.measure("open    4a. B = sum q^j P_j", || {
        let mut b = polys[0].clone();
        let mut qj = q;
        for f in polys.iter().skip(1) {
            b.par_iter_mut()
                .zip(f.par_iter())
                .for_each(|(bi, &fi)| *bi += qj * fi);
            qj *= q;
        }
        b
    });
    let h = report.measure("open    4b. B / cubic divisor (serial reference)", || {
        divide_cubic(&b, &vanishing(&u))
    });
    let _ = report.measure("open    5. quotient MSM (N-3 pts)", || {
        Bn254::g1_affine_msm(&setup.g1_powers()[..h.len()], &h)
    });
}

fn stream_profile(
    report: &mut Report,
    columns: &[Column],
    packed: &PackedColumns,
    k: usize,
    setup: &HyperKZGProverSetup<Bn254>,
    verifier_setup: &HyperKZGVerifierSetup<Bn254>,
) {
    let row_columns: Vec<Vec<Fr>> = columns[..5]
        .iter()
        .map(|column| match column {
            Column::Bits(values) => values
                .iter()
                .map(|&value| Fr::from_u64(u64::from(value)))
                .collect(),
            Column::U16(_) | Column::Fr(_) => unreachable!("the first columns are bits"),
        })
        .collect();
    let mut row = TimingRow::new(row_columns);
    let statement = TensorStreamStatement {
        key_digest: [29; 32],
        rows: ROWS,
        column_count: columns.len(),
        k,
        row_input_claim: row.claim,
        row_degree: 5,
        stage_a_encoding: StageAEncoding::KzgCommitted,
        terms: vec![TensorTerm {
            coefficient: Fr::one(),
            columns: vec![0, 1, 2, 3, 4],
        }],
    };
    let proof = report.measure(
        "stream  prove_stream e2e (synthetic deg-5 row prover)",
        || prove_stream(packed, &statement, &mut row, setup).expect("prove stream"),
    );
    let (_, cost) = report.measure("verify  verify_stream_with_cost", || {
        verify_stream_with_cost(&proof, &statement, verifier_setup).expect("verify stream")
    });
    report.note(&format!(
        "k={k}: payload {} B, bincode {} B, verifier {cost:?}",
        proof.payload_bytes(),
        proof.bincode_bytes()
    ));

    let mut rng = StdRng::seed_from_u64(0x0b);
    let r_a: Vec<Fr> = (0..ROWS_LOG).map(|_| Fr::random(&mut rng)).collect();
    let s: Vec<Fr> = (0..packed.layout.column_vars())
        .map(|_| Fr::random(&mut rng))
        .collect();
    let column_values = report.measure("stream  column evaluations at r_A (all groups)", || {
        packed.column_evaluations(&r_a).expect("column values")
    });
    let mut reductions: Vec<ColumnReduction> = (0..5)
        .map(|column| ColumnReduction::new(column_values.clone(), column).expect("reduction"))
        .collect();
    let claims: Vec<Fr> = reductions
        .iter()
        .map(ColumnReduction::input_claim)
        .collect();
    let mut members: Vec<StageMember<'_>> = reductions
        .iter_mut()
        .zip(&claims)
        .map(|(reduction, &input_claim)| StageMember {
            prover: reduction,
            input_claim,
            degree: 2,
            offset: 0,
        })
        .collect();
    let mut transcript = Keccak256Transcript::<Fr>::new(b"perf1-b");
    let _ = report.measure("stream  stage B (5 column reductions)", || {
        prove_stage(&mut members, &mut transcript).expect("stage B")
    });
    drop(members);
    let weights = packed.layout.group_weights(&s).expect("group weights");
    let point = packed.layout.packed_point(&r_a, &s).expect("packed point");
    let combined = report.measure("open    RLC of packed polys (groups x N)", || {
        rlc(&packed.evaluations, &weights)
    });
    let _ = report.measure("open    evaluate combined at the point (check)", || {
        combined.as_slice().evaluate(&point)
    });
    let mut transcript = Keccak256Transcript::<Fr>::new(b"perf1-open");
    let _ = report.measure("open    HyperKZG::open total", || {
        HyperKZGScheme::<Bn254>::open(setup, &combined, &point, &mut transcript).expect("open")
    });
    open_replica(report, setup, &combined, &point);
}

// ---------------------------------------------------------------- gate

#[test]
#[ignore = "full-statement wrapper-prover profile at 2^18 rows"]
fn perf1_full_statement_profile() {
    let mut report = Report::new();
    report.note(&format!(
        "PERF-1 full-statement profile: rows 2^{ROWS_LOG}, {} rayon threads, load {:.2}",
        rayon::current_num_threads(),
        load_average()
    ));
    let setup = report.measure("setup   SRS 2^22 setup_from_secret (fixed-base)", || {
        HyperKZGScheme::<Bn254>::setup_from_secret(
            Fr::from_u64(23),
            1 << SRS_LOG,
            Bn254::g1_generator(),
            Bn254::g2_generator(),
        )
    });
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let columns = report.measure(
        "witness synthetic column data (254 columns)",
        full_statement_columns,
    );
    let t1 = report.measure("witness T1 synthetic transcript table", t1_table);
    report.note(&format!(
        "T1 table: {} rows of {ROWS} ({:.1}% full)",
        t1.rows(),
        100.0 * t1.rows() as f64 / ROWS as f64
    ));
    let t2 = report.measure("witness T2 synthetic program, chunks, LogUp", t2_witness);
    for k in [8usize, 16] {
        for repeat in 0..REPEATS {
            report.note(&format!("--- k={k} repeat {repeat}"));
            let packed = report
                .measure(&format!("commit  commit_packed k={k} (all groups)"), || {
                    commit_packed(&columns, k, &setup).expect("commit packed")
                });
            if repeat == 0 {
                profile_commit_kernels(&mut report, &columns, &packed, k, &setup);
            }
            t1_profile(&mut report, &t1, &setup);
            t2_profile(&mut report, &t2, &setup);
            spartan_profile(&mut report, &setup);
            stream_profile(&mut report, &columns, &packed, k, &setup, &verifier_setup);
        }
    }
    let summary = report.summary();
    println!("{summary}");
    std::fs::write(
        "/tmp/perf1-profile.txt",
        format!("{}\n{summary}", report.log),
    )
    .expect("write profile");
}

#[test]
#[ignore = "PERF-2 single-MSM benchmark"]
fn perf2_msm_profile() {
    let max_n = 1 << 21;
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(23),
        max_n,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let mut rng = StdRng::seed_from_u64(0xf002);
    let scalars: Vec<Fr> = (0..max_n).map(|_| Fr::random(&mut rng)).collect();
    for log_n in [20, 21] {
        let n = 1 << log_n;
        for repeat in 0..3 {
            let load = load_average();
            let start = Instant::now();
            let result = Bn254::g1_affine_msm(&setup.g1_powers()[..n], &scalars[..n]);
            let elapsed = start.elapsed();
            let _ = std::hint::black_box(result);
            println!(
                "PERF2 log_n={log_n} repeat={repeat} seconds={:.6} us_per_point={:.6} load={load:.2}",
                elapsed.as_secs_f64(),
                elapsed.as_secs_f64() * 1e6 / n as f64,
            );
        }
    }
}

#[test]
#[ignore = "PERF-2 commit and opening benchmark"]
fn perf2_commit_open_profile() {
    let mut report = Report::new();
    let n = 1 << 21;
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(23),
        n,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let columns = full_statement_columns();
    let packed = commit_packed(&columns, 8, &setup).expect("commit packed columns");
    profile_commit_kernels(&mut report, &columns, &packed, 8, &setup);

    let mut rng = StdRng::seed_from_u64(0xf003);
    let polynomial = (0..n).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
    let point = (0..21).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
    let mut transcript = Keccak256Transcript::<Fr>::new(b"perf2-open");
    let _ = report.measure("open    HyperKZG::open total", || {
        HyperKZGScheme::<Bn254>::open(&setup, &polynomial, &point, &mut transcript).expect("open")
    });
    println!("{}", report.summary());
}
