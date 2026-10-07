//! Byte-link parity harness (spec §4). One fixture is proved and verified over
//! a challenge tape keyed by draw role (`relations::DrawRole`) rather than
//! transcript position, so every draw that survives in both builds takes the
//! same value in both. Each S1–S7 boundary is then digested three ways —
//! retained statements (member, opening ids, values, points), retained draw
//! keys, and retained batch members (order, rounds, degree, point offset,
//! input and output expressions) — for comparison against one frozen record.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the parity harness must fail loudly on a malformed session or fixture"
)]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::ops::Range;

use jolt_akita::{AkitaField, AkitaScheme};
use jolt_claims::protocols::jolt::lattice::{ByteTraceLayoutPlan, OneHotTraceShape};
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltRelationId};
use jolt_host::Program;
use jolt_prover::akita::preprocessing::{self, AkitaTranscript, AkitaVc};
use jolt_transcript::Transcript;
use jolt_verifier::fs_audit::{self, BatchMember};
use jolt_verifier::proof::JoltProofClaims;
use jolt_verifier::stages::relations::{DrawRole, OutputClaims};
use jolt_verifier::stages::{
    build_formula_dimensions, stage1, stage2, stage3, stage4, stage5, stage6a, stage6b, stage7,
};
use jolt_verifier::{validate_and_seed_transcript, verify, VerifierError};
use serde::Serialize;

use super::akita_fixtures::{derive_config, prove_prepared, AkitaFixtureCase};
use super::guest_fixtures::prepare_guest;

type Tape = ParityTranscript<AkitaTranscript>;

/// The batch members the byte link removes (spec §3). No retained digest
/// covers them, nor the rounds that only they bind.
const REMOVED_MEMBERS: [(&str, &str); 4] = [
    ("Stage6a", "booleanity"),
    ("Stage6b", "booleanity"),
    ("Stage6b", "ram_hamming_booleanity"),
    ("Stage7", "hamming_weight_claim_reduction"),
];

fn is_removed(batch: &str, member: &str) -> bool {
    REMOVED_MEMBERS.contains(&(batch, member))
}

/// One tape draw: its role (`None` for draws ordered by transcript position
/// alone) and its ordinal among the draws of that role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TapeDraw {
    role: Option<DrawRole>,
    ordinal: u64,
}

impl TapeDraw {
    /// Whether a build without the removed members makes this draw.
    fn is_retained(&self, rounds: &BTreeMap<&str, BatchRounds>) -> bool {
        match self.role {
            None => true,
            Some(
                DrawRole::MemberChallenges { batch, member }
                | DrawRole::BatchingCoefficient { batch, member },
            ) => !is_removed(batch, member),
            Some(DrawRole::Rounds { batch }) => self.ordinal < rounds[batch].retained,
        }
    }
}

/// Round challenges of one batch: every member's schedule, and the prefix
/// some retained member binds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BatchRounds {
    scheduled: u64,
    retained: u64,
}

#[derive(Default)]
struct TapeSession {
    ordinals: BTreeMap<Option<DrawRole>, u64>,
    draws: Vec<TapeDraw>,
}

thread_local! {
    static TAPE: RefCell<Option<TapeSession>> = const { RefCell::new(None) };
}

/// Runs `f` with every [`ParityTranscript`] challenge read off a fresh tape;
/// returns the draws in consumption order.
fn with_tape<R>(f: impl FnOnce() -> R) -> (R, Vec<TapeDraw>) {
    TAPE.with_borrow_mut(|tape| {
        assert!(tape.is_none(), "nested parity tapes are unsupported");
        *tape = Some(TapeSession::default());
    });
    let output = f();
    let tape = TAPE
        .with_borrow_mut(Option::take)
        .expect("parity tape disappeared");
    (output, tape.draws)
}

fn tape_len() -> usize {
    TAPE.with_borrow(|tape| tape.as_ref().map_or(0, |tape| tape.draws.len()))
}

fn next_draw() -> TapeDraw {
    TAPE.with_borrow_mut(|tape| {
        let tape = tape
            .as_mut()
            .expect("parity transcript drawn outside a tape session");
        let role = fs_audit::current_role();
        let ordinal = tape.ordinals.entry(role).or_default();
        let draw = TapeDraw {
            role,
            ordinal: *ordinal,
        };
        *ordinal += 1;
        tape.draws.push(draw);
        draw
    })
}

/// A transcript whose challenges are a pseudorandom function of the
/// [`TapeDraw`] key, independent of the absorbed bytes.
#[derive(Default)]
struct ParityTranscript<T> {
    inner: T,
}

impl<T: Transcript> Transcript for ParityTranscript<T> {
    type Challenge = T::Challenge;

    fn new(label: &'static [u8]) -> Self {
        Self {
            inner: T::new(label),
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.inner.append_bytes(bytes);
    }

    fn challenge(&mut self) -> Self::Challenge {
        let mut key = T::new(b"jolt-byte-link-parity-tape");
        key.append_bytes(format!("{:?}", next_draw()).as_bytes());
        key.challenge()
    }

    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut sponge = AkitaTranscript::new(b"jolt-byte-link-parity-digest");
    sponge.append_bytes(bytes);
    sponge.state()
}

fn encode<S: Serialize + ?Sized>(value: &S) -> Vec<u8> {
    postcard::to_stdvec(value).expect("serialize a parity record entry")
}

/// Retained statements of one stage, in member declaration order.
#[derive(Default)]
struct Statements(Vec<u8>);

impl Statements {
    fn push<S: Serialize + ?Sized>(&mut self, value: &S) {
        self.0.extend(encode(value));
    }

    fn member<V, P>(&mut self, name: &str, values: &V, points: &P)
    where
        V: OutputClaims<AkitaField> + Serialize,
        P: Serialize,
    {
        self.push(&(name, values.canonical_order(), values, points));
    }

    fn optional_member<V, P>(&mut self, name: &str, values: Option<&V>, points: Option<&P>)
    where
        V: OutputClaims<AkitaField> + Serialize,
        P: Serialize,
    {
        let ids = values.map(|values| values.canonical_order());
        self.push(&(name, ids, values, points));
    }
}

/// One S1–S7 boundary's retained digests.
struct StageDigest {
    statements: [u8; 32],
    draws: [u8; 32],
    catalog: [u8; 32],
}

struct StageRun {
    name: &'static str,
    statements: Statements,
    members: Vec<BatchMember<AkitaField>>,
    draws: Range<usize>,
}

impl StageRun {
    fn digest(&self, draws: &[TapeDraw]) -> StageDigest {
        let draws = &draws[self.draws.clone()];
        let mut rounds = BTreeMap::<&str, BatchRounds>::new();
        for member in &self.members {
            let end = (member.point_offset + member.rounds) as u64;
            let batch = rounds.entry(member.batch).or_default();
            batch.scheduled = batch.scheduled.max(end);
            if !is_removed(member.batch, member.member) {
                batch.retained = batch.retained.max(end);
            }
        }
        let mut drawn = BTreeMap::<&str, u64>::new();
        for draw in draws {
            if let Some(DrawRole::Rounds { batch }) = draw.role {
                *drawn.entry(batch).or_default() += 1;
            }
        }
        let scheduled = rounds
            .iter()
            .map(|(batch, rounds)| (*batch, rounds.scheduled))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            drawn, scheduled,
            "{}: round draws per batch must match the instantiated batch schedule",
            self.name
        );
        let draw_keys = draws
            .iter()
            .filter(|draw| draw.is_retained(&rounds))
            .map(|draw| format!("{draw:?}"))
            .collect::<Vec<_>>();
        let catalog = self
            .members
            .iter()
            .filter(|member| !is_removed(member.batch, member.member))
            .flat_map(|member| {
                encode(&(
                    member.batch,
                    member.member,
                    member.relation,
                    member.rounds,
                    member.degree,
                    member.point_offset,
                    &member.input,
                    &member.output,
                ))
            })
            .collect::<Vec<_>>();
        StageDigest {
            statements: digest(&self.statements.0),
            draws: digest(&encode(&draw_keys)),
            catalog: digest(&catalog),
        }
    }
}

/// The frozen-record view of one parity proof.
pub struct ParityRecord {
    /// Digest of the program, its executed trace rows, and its public I/O:
    /// equal records are about the same execution.
    fixture: [u8; 32],
    stages: Vec<(&'static str, StageDigest)>,
}

impl ParityRecord {
    /// One line per digest, the frozen-record format.
    pub fn lines(&self) -> Vec<String> {
        let hex = |bytes: &[u8; 32]| {
            bytes.iter().fold(String::new(), |mut hex, byte| {
                write!(hex, "{byte:02x}").expect("write to a String");
                hex
            })
        };
        let mut lines = vec![format!("fixture {}", hex(&self.fixture))];
        for (stage, digest) in &self.stages {
            lines.push(format!("{stage} statements {}", hex(&digest.statements)));
            lines.push(format!("{stage} draws {}", hex(&digest.draws)));
            lines.push(format!("{stage} catalog {}", hex(&digest.catalog)));
        }
        lines
    }
}

/// Proves the parity fixture — the committed-program muldiv guest at forced
/// K=2^8, the byte trace's 16/2/2 geometry — over the tape, checks that the
/// verifier accepts it drawing the prover's exact tape, and digests its
/// retained S1–S7 statements.
pub fn parity_record() -> ParityRecord {
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    let run = prepare_guest(Program::new("muldiv-guest"), &inputs, &[], &[]);
    let mut config = derive_config(&run);
    config.one_hot_config = JoltOneHotConfig {
        log_k_chunk: 8,
        lookups_ra_virtual_log_k_chunk: 32,
    };
    let fixture = digest(&encode(&(
        &run.program_preprocessing,
        &run.trace.device,
        // The derived `Debug` of `JoltTraceRow` spells every row field.
        digest(format!("{:?}", run.trace.trace).as_bytes()),
        config.trace_length as u64,
        config.ram_K as u64,
    )));
    let preprocessing =
        preprocessing::preprocess_committed(run.program_preprocessing.clone(), &config, 2)
            .expect("committed Akita preprocessing");
    let (case, prover_draws) =
        with_tape(|| prove_prepared::<Tape>(run, config, preprocessing, &[]));

    let (accepted, verifier_draws) = with_tape(|| {
        verify::<AkitaField, AkitaScheme, AkitaVc, Tape>(
            &case.preprocessing,
            &case.public_io,
            &case.proof,
            case.trusted_advice_commitment.as_ref(),
        )
    });
    accepted.expect("the verifier must accept the parity proof over the tape");
    assert_eq!(
        prover_draws, verifier_draws,
        "prover and verifier must draw the same tape keys in the same order"
    );

    let (runs, draws) = with_tape(|| stage_runs(&case));
    let runs = runs.expect("retained-stage replay");
    ParityRecord {
        fixture,
        stages: runs
            .iter()
            .map(|run| (run.name, run.digest(&draws)))
            .collect(),
    }
}

fn stage<O>(
    runs: &mut Vec<StageRun>,
    name: &'static str,
    verify: impl FnOnce() -> Result<O, VerifierError>,
    statements: impl FnOnce(&O, &mut Statements) -> Result<(), VerifierError>,
) -> Result<O, VerifierError> {
    let start = tape_len();
    let (output, members) = fs_audit::record_batch_members(verify);
    let output = output?;
    let mut collected = Statements::default();
    statements(&output, &mut collected)?;
    runs.push(StageRun {
        name,
        statements: collected,
        members,
        draws: start..tape_len(),
    });
    Ok(output)
}

macro_rules! members {
    ($statements:ident, $output:ident; $($member:ident),+) => {
        $($statements.member(
            stringify!($member),
            &$output.output_values.$member,
            &$output.output_points.$member,
        );)+
    };
}

macro_rules! optional_members {
    ($statements:ident, $output:ident; $($member:ident),+) => {
        $($statements.optional_member(
            stringify!($member),
            $output.output_values.$member.as_ref(),
            $output.output_points.$member.as_ref(),
        );)+
    };
}

/// Replays the verifier's S1–S7 over the active tape, collecting each stage's
/// retained statements, batch members, and draws. Mirrors `verify`'s stage
/// order through stage 7; lists every member except [`REMOVED_MEMBERS`].
fn stage_runs(case: &AkitaFixtureCase) -> Result<Vec<StageRun>, VerifierError> {
    let proof = &case.proof;
    let preprocessing = &case.preprocessing;
    let JoltProofClaims::Clear(claims) = &proof.claims else {
        panic!("Akita proofs carry clear claims");
    };
    let (checked, mut transcript) = validate_and_seed_transcript::<AkitaScheme, AkitaVc, Tape, _>(
        preprocessing,
        &case.public_io,
        proof,
        case.trusted_advice_commitment.as_ref(),
    )?;
    let log_t = checked.trace_length.ilog2() as usize;
    let dimensions = build_formula_dimensions(
        proof,
        preprocessing,
        &checked,
        log_t,
        JoltRelationId::InstructionReadRaf,
    )?;
    let shape = OneHotTraceShape {
        ra_layout: dimensions.ra_layout,
        log_t,
        log_k_chunk: proof.one_hot_config.committed_chunk_bits(),
    };
    if let Err(error) = ByteTraceLayoutPlan::new(&shape) {
        panic!("the parity fixture lacks the byte-trace geometry: {error}");
    }

    let mut runs = Vec::new();
    let s1 = stage(
        &mut runs,
        "stage1",
        || stage1::verify(&checked, proof, &mut transcript),
        |output, statements| {
            let output = output.clear()?;
            statements.push(&claims.stage1.uniskip_output_claim);
            members!(statements, output; outer_remainder);
            Ok(())
        },
    )?;
    let s2 = stage(
        &mut runs,
        "stage2",
        || stage2::verify(&checked, proof, &mut transcript, &s1),
        |output, statements| {
            let output = output.clear()?;
            statements.push(&claims.stage2.product_uniskip_output_claim);
            members!(statements, output; ram_read_write, product_remainder,
                instruction_claim_reduction, ram_raf_evaluation, ram_output_check);
            Ok(())
        },
    )?;
    let s3 = stage(
        &mut runs,
        "stage3",
        || stage3::verify(&checked, proof, &mut transcript, &s1, &s2),
        |output, statements| {
            let output = output.clear()?;
            members!(statements, output; shift, instruction_input, registers_claim_reduction);
            Ok(())
        },
    )?;
    let s4 = stage(
        &mut runs,
        "stage4",
        || stage4::verify(&checked, preprocessing, proof, &mut transcript, &s2, &s3),
        |output, statements| {
            let output = output.clear()?;
            members!(statements, output; registers_read_write, ram_val_check);
            let init = &output.ram_val_check_init;
            let advice = init
                .advice_contributions
                .iter()
                .map(|advice| {
                    let point = &advice.opening_point;
                    (advice.kind, advice.selector, point, advice.opening_value)
                })
                .collect::<Vec<_>>();
            statements.push(&(init.public_eval, &init.program_image_contribution, advice));
            Ok(())
        },
    )?;
    let s5 = stage(
        &mut runs,
        "stage5",
        || stage5::verify(&checked, proof, &dimensions, &mut transcript, &s2, &s4),
        |output, statements| {
            let output = output.clear()?;
            members!(statements, output; instruction_read_raf, ram_ra_claim_reduction,
                registers_val_evaluation);
            Ok(())
        },
    )?;
    let s6a = stage(
        &mut runs,
        "stage6a",
        || {
            stage6a::verify(
                &checked,
                preprocessing,
                proof,
                &dimensions,
                &mut transcript,
                &s1,
                &s2,
                &s3,
                &s4,
                &s5,
            )
        },
        |output, statements| {
            let output = output.clear()?;
            members!(statements, output; bytecode_read_raf);
            Ok(())
        },
    )?;
    let s6b = stage(
        &mut runs,
        "stage6b",
        || {
            stage6b::verify(
                &checked,
                preprocessing,
                proof,
                &dimensions,
                &mut transcript,
                &s1,
                &s2,
                &s3,
                &s4,
                &s5,
                &s6a,
            )
        },
        |output, statements| {
            let output = output.clear()?;
            members!(statements, output; bytecode_read_raf, ram_ra_virtualization,
                instruction_ra_virtualization);
            optional_members!(statements, output; bytecode_reduction, program_image_reduction);
            Ok(())
        },
    )?;
    let _stage7 = stage(
        &mut runs,
        "stage7",
        || stage7::verify(&checked, proof, &dimensions, &mut transcript, &s4, &s6b),
        |output, statements| {
            let output = output.clear()?;
            optional_members!(statements, output; bytecode_address_phase,
                program_image_address_phase);
            Ok(())
        },
    )?;
    Ok(runs)
}
