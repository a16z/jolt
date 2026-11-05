//! Whiteout (zero knowledge over Akita): the masked-site census (HT1,
//! `specs/whiteout-jolt-integration.md` §7), the tamper suite (V5), HT3, the
//! pad layout's counting equality (V4) and the compensating-shift probe.
//!
//! A Whiteout proof has two parts. Jolt's stage messages are the proof's
//! masked tape: every value is m̂ = m + p, absorbed where the clear protocol
//! absorbs m. The pad commitment, Akita's masked levels, the Σ-protocol and
//! the response proof are the argument string. The census classifies both:
//! Jolt's absorbs from an audit transcript over the verifier, and the
//! argument string from Akita's prover-side record of its message ranges
//! (`akita_transcript::thread_proof_ranges`, kept by the fixture).

#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "census and tamper checks fail loudly"
)]

use std::collections::BTreeMap;
use std::ops::Range;

use akita_params::zk::ZK_CUT_LEVEL;
use akita_transcript::{
    ProofMessageRange, ProtocolMessageKind, ProtocolSiteId, SITE_FAMILY_NEXT_WITNESS,
    SITE_FAMILY_OPENING_PAYLOAD, SITE_FAMILY_STAGE1, SITE_FAMILY_STAGE2, SITE_FAMILY_SUMCHECK,
};
use jolt_akita::{AkitaField, AkitaScheme};
use jolt_algebra::{CanonicalBytes, Ring};
use jolt_poly::CompressedPoly;
use jolt_prover::akita::preprocessing::AkitaVc;
use jolt_sumcheck::{ClearProof, CompressedSumcheckProof, SumcheckProof};
use jolt_transcript::{verifier_transcript, Keccak, Script, ScriptedTranscript, VerifierState};
use jolt_verifier::proof::{JoltProofClaims, JoltStageProofs};
use jolt_verifier::stages::pow::PowSite;
use jolt_verifier::stages::stage8::akita::WhiteoutObject;
use jolt_verifier::whiteout::{pad_layout, verify_transcript, whiteout_frame};
use jolt_verifier::{
    validate_and_seed_transcript, JoltProtocolConfig, VerifierError, ZkConfig, JOLT_SESSION,
};
use jolt_zk::masking::MaskedMessages;

use crate::support::akita_fixtures::{
    whiteout_advice_case, whiteout_committed_case, whiteout_muldiv_case, whiteout_rerun_case,
    AkitaFixtureCase, AkitaJoltProof,
};

type F = AkitaField;

const FIELD_BYTES: usize = <F as CanonicalBytes>::NUM_BYTES;

/// The census classes (§7). `clear-only` absorbs never run here, so the
/// runtime census has no such class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    Statement,
    Commitment,
    Pads,
    MaskedSlot,
    ChallengeDerived,
    Sigma,
    Response,
}

/// How an operation reaches the sponge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    /// A private value sent masked over its pads.
    Masked,
    /// A prover message absorbed as sent.
    Clear,
    /// A verifier-known value absorbed.
    Public,
}

/// Where an operation is: a Jolt absorb (its index among the audit's
/// absorbs after seeding, or during seeding), or a range of the argument
/// string.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Origin {
    Seed(usize),
    Absorb(usize),
    Narg(Range<usize>),
}

/// One classified operation.
#[derive(Clone, Debug)]
struct Item {
    origin: Origin,
    kind: Kind,
    bytes: usize,
    /// For masked tape absorbs, the tape values they carry.
    tape: Option<Range<usize>>,
    /// For argument-string ranges, Akita's site, if Akita sent it.
    site: Option<ProtocolSiteId>,
    class: Option<Class>,
}

/// The verifier's own challenges, priced as the proof field's challenge set,
/// so the audit transcript checks every proof of work.
#[derive(Default)]
struct Passthrough;

impl Script<F> for Passthrough {
    const UNPRICED: bool = false;

    fn draw(&mut self, _challenges: &mut [F]) {}
}

type Audit<'a> = ScriptedTranscript<VerifierState<'a, Keccak>, F, Passthrough>;

/// Jolt's absorbs over one verification of a Whiteout proof: those made
/// while binding the commitments, then every later one. Akita's own absorbs
/// go to its state directly and are classified from the argument string.
#[derive(Clone, Debug)]
struct JoltTrace {
    seed: Vec<Vec<u8>>,
    after: Vec<Vec<u8>>,
    result: Result<(), String>,
}

/// Verifies `proof` against `case`'s statement on an audit transcript.
fn jolt_trace(case: &AkitaFixtureCase, proof: &AkitaJoltProof) -> JoltTrace {
    let seeded = validate_and_seed_transcript(
        &case.preprocessing,
        &case.public_io,
        proof,
        case.trusted_advice_commitment.as_ref(),
        |statement| {
            Audit::scripted_by(
                verifier_transcript::<Keccak>(&JOLT_SESSION, statement, &proof.narg),
                Passthrough,
            )
        },
    );
    let (checked, mut transcript) = match seeded {
        Ok(seeded) => seeded,
        Err(error) => {
            return JoltTrace {
                seed: Vec::new(),
                after: Vec::new(),
                result: Err(format!("{error:?}")),
            }
        }
    };
    let seed: Vec<Vec<u8>> = transcript.absorbed().iter().map(|b| b.to_vec()).collect();
    let verified = verify_transcript::<F, AkitaScheme, AkitaVc, _>(
        &case.preprocessing,
        proof,
        &checked,
        case.trusted_advice_commitment.as_ref(),
        &mut transcript,
        None,
    );
    let all: Vec<Vec<u8>> = transcript.absorbed().iter().map(|b| b.to_vec()).collect();
    let after = all[seed.len()..].to_vec();
    let result = verified
        .map_err(|error| format!("{error:?}"))
        .and_then(|()| {
            transcript
                .into_inner()
                .check_eof()
                .map_err(|error| format!("{error:?}"))
        });
    JoltTrace {
        seed,
        after,
        result,
    }
}

fn tape_of(proof: &AkitaJoltProof) -> &MaskedMessages<F> {
    match &proof.claims {
        JoltProofClaims::Zk { masked, .. } => masked,
        JoltProofClaims::Clear(_) => panic!("a Whiteout proof carries a masked tape"),
    }
}

fn tape_mut(proof: &mut AkitaJoltProof) -> &mut MaskedMessages<F> {
    match &mut proof.claims {
        JoltProofClaims::Zk { masked, .. } => masked,
        JoltProofClaims::Clear(_) => panic!("a Whiteout proof carries a masked tape"),
    }
}

fn encode(values: &[F]) -> Vec<u8> {
    values
        .iter()
        .flat_map(CanonicalBytes::to_bytes_le_vec)
        .collect()
}

/// The absorbs a proof's statement fixes outside the message stream.
struct Expected {
    /// The commitments' absorbs, from an honest seeding.
    seed: Vec<Vec<u8>>,
    /// Stage 8's own absorbs: each packed precommitted object's reduction
    /// binds its two layout words, its digest and its point, and never an
    /// evaluation (the clear reduction's `clear-only` absorb).
    stage8: Vec<usize>,
    /// The stage proof-of-work nonces the proof carries.
    stage_nonces: Vec<u64>,
}

/// [`Expected`] of `case`'s honest proof, from the statement's opening plan.
fn expected_absorbs(case: &AkitaFixtureCase) -> Expected {
    let (checked, _) = validate_and_seed_transcript(
        &case.preprocessing,
        &case.public_io,
        &case.proof,
        case.trusted_advice_commitment.as_ref(),
        |statement| verifier_transcript::<Keccak>(&JOLT_SESSION, statement, &case.proof.narg),
    )
    .expect("the honest proof seeds");
    let frame = whiteout_frame(
        &case.preprocessing,
        &case.proof,
        &checked,
        case.trusted_advice_commitment.as_ref(),
    )
    .expect("the Whiteout frame");
    let mut stage8 = Vec::new();
    for object in &frame.plan.objects {
        match object {
            WhiteoutObject::Packed { plan, .. } => {
                stage8.extend([8, 8, 32, plan.packing().logical_num_vars() * FIELD_BYTES]);
            }
            #[cfg(feature = "field-inline")]
            WhiteoutObject::FieldInc { .. } => {}
            WhiteoutObject::FusedInc { .. } => {}
        }
    }
    let stage_nonces = PowSite::ALL
        .iter()
        .filter_map(|&site| case.proof.stages.pow_nonces.get(site))
        .collect();
    Expected {
        seed: jolt_trace(case, &case.proof).seed,
        stage8,
        stage_nonces,
    }
}

/// Classifies Jolt's absorbs. During seeding they must be exactly the honest
/// commitments' absorbs. Afterwards each is the next run of masked tape
/// values (a masked slot), a proof-of-work nonce of the tape or of a stage
/// (challenge-derived), or one of Stage 8's own statement-derived absorbs;
/// the tape must be absorbed whole and in order, and Stage 8's absorbs must
/// be exactly the statement's.
fn jolt_items(
    trace: &JoltTrace,
    tape: &MaskedMessages<F>,
    expected: &Expected,
) -> (Vec<Item>, Vec<String>) {
    let mut items = Vec::new();
    let mut failures = Vec::new();
    if trace.seed != expected.seed {
        failures.push(format!(
            "the commitments' absorbs ({} of them) are not the statement's ({})",
            trace.seed.len(),
            expected.seed.len()
        ));
    }
    for (index, bytes) in trace.seed.iter().enumerate() {
        items.push(Item {
            origin: Origin::Seed(index),
            kind: Kind::Public,
            bytes: bytes.len(),
            tape: None,
            site: None,
            class: Some(Class::Commitment),
        });
    }
    let mut value = 0;
    let mut nonce = 0;
    let mut stage_nonces = expected.stage_nonces.clone();
    let mut stage8 = Vec::new();
    for (index, bytes) in trace.after.iter().enumerate() {
        let width = bytes.len() / FIELD_BYTES;
        let tape_run = (bytes.len().is_multiple_of(FIELD_BYTES)
            && width > 0
            && value + width <= tape.values.len()
            && *bytes == encode(&tape.values[value..value + width]))
        .then(|| value..value + width);
        let item = |kind, class, tape| Item {
            origin: Origin::Absorb(index),
            kind,
            bytes: bytes.len(),
            tape,
            site: None,
            class,
        };
        if let Some(run) = tape_run {
            value = run.end;
            items.push(item(Kind::Masked, Some(Class::MaskedSlot), Some(run)));
            continue;
        }
        if bytes.len() == 8 {
            let word = u64::from_le_bytes(bytes[..].try_into().expect("8 bytes"));
            if tape.nonces.get(nonce) == Some(&word) {
                nonce += 1;
                items.push(item(Kind::Public, Some(Class::ChallengeDerived), None));
                continue;
            }
            if let Some(position) = stage_nonces.iter().position(|&n| n == word) {
                let _ = stage_nonces.remove(position);
                items.push(item(Kind::Public, Some(Class::ChallengeDerived), None));
                continue;
            }
        }
        stage8.push(bytes.len());
        items.push(item(Kind::Public, Some(Class::Statement), None));
    }
    if stage8 != expected.stage8 {
        failures.push(format!(
            "the absorbs outside the tape and nonces are {stage8:?}, not Stage 8's {:?}",
            expected.stage8
        ));
    }
    if value != tape.values.len() || nonce != tape.nonces.len() {
        failures.push(format!(
            "the masked tape was absorbed to value {value} of {} and nonce {nonce} of {}",
            tape.values.len(),
            tape.nonces.len()
        ));
    }
    if !stage_nonces.is_empty() {
        failures.push(format!("stage nonces {stage_nonces:?} were never absorbed"));
    }
    (items, failures)
}

/// A site identity from its eight little-endian words.
fn site_of(bytes: [u8; 32]) -> ProtocolSiteId {
    let word = |index: usize| {
        u32::from_le_bytes(bytes[4 * index..4 * index + 4].try_into().expect("4 bytes"))
    };
    ProtocolSiteId {
        family: word(0),
        invocation: word(1),
        level: word(2),
        stage: word(3),
        round: word(4),
        group: word(5),
        limb: word(6),
        detail: word(7),
    }
}

/// The class of one of Akita's argument-string ranges at its masked levels
/// (`akita:specs/whiteout-akita.md` §4.2): rounds and their claims masked
/// (sumcheck, stage 2, stage 1), the D image and u_a sent clear but hidden
/// by their own randomness, grinding and fold-response nonces public.
fn prefix_class(range: &ProofMessageRange, len: usize) -> (Kind, Option<Class>) {
    let site = site_of(range.context.site_id);
    let kind = range.context.kind;
    let nonce = kind == ProtocolMessageKind::GrindingNonce as u32
        || kind == ProtocolMessageKind::FoldResponseNonce as u32;
    match range.pads {
        Some((_, count)) => {
            let masked_family = matches!(
                site.family,
                SITE_FAMILY_SUMCHECK | SITE_FAMILY_STAGE2 | SITE_FAMILY_STAGE1
            );
            let whole = count * FIELD_BYTES == len;
            (
                Kind::Masked,
                (masked_family && whole).then_some(Class::MaskedSlot),
            )
        }
        None if nonce => (Kind::Clear, Some(Class::ChallengeDerived)),
        None => (
            Kind::Clear,
            matches!(
                site.family,
                SITE_FAMILY_OPENING_PAYLOAD | SITE_FAMILY_NEXT_WITNESS
            )
            .then_some(Class::Commitment),
        ),
    }
}

/// Classifies the argument string. The prover's region marks split it into
/// the pad commitment (Jolt's, after stage 0), Akita's masked levels, Σ
/// (Jolt's) and the response proof (Akita's, from its recorded start to the
/// end); stages 1–7 write none of it. Akita's record of its messages
/// classifies the masked levels, which it must tile.
fn narg_items(
    ranges: &[ProofMessageRange],
    regions: &[(&str, usize)],
    narg_len: usize,
) -> (Vec<Item>, Vec<String>) {
    let mut items = Vec::new();
    let mut failures = Vec::new();
    let mark = |name: &str| {
        regions
            .iter()
            .find(|(region, _)| *region == name)
            .map(|(_, at)| *at)
    };
    let (Some(stage0), Some(pads), Some(stage7), Some(prefix), Some(response), Some(end)) = (
        mark("stage0"),
        mark("pads"),
        mark("stage7"),
        mark("akita_masked_prefix"),
        mark("response"),
        mark("sigma_response"),
    ) else {
        return (items, vec![format!("missing region marks: {regions:?}")]);
    };
    if stage7 != pads {
        failures.push(format!(
            "stages 1-7 wrote {} argument-string bytes",
            stage7 - pads
        ));
    }
    if end != narg_len {
        failures.push(format!(
            "the argument string is {narg_len} bytes, the prover's {end}"
        ));
    }
    let gap = |range: Range<usize>, class: Option<Class>, items: &mut Vec<Item>| {
        if !range.is_empty() {
            items.push(Item {
                bytes: range.len(),
                origin: Origin::Narg(range),
                kind: Kind::Clear,
                tape: None,
                site: None,
                class,
            });
        }
    };
    gap(0..stage0, Some(Class::Commitment), &mut items);
    for (part, range, class) in [
        ("pad commitment", stage0..pads, Class::Pads),
        ("Σ-protocol", prefix..response, Class::Sigma),
        ("response proof", response..narg_len, Class::Response),
    ] {
        if range.is_empty() {
            failures.push(format!("no {part} bytes"));
        }
        gap(range, Some(class), &mut items);
    }

    let mut ranges = ranges.to_vec();
    ranges.sort_by_key(|range| range.start);
    let mut cursor = pads;
    for range in &ranges {
        let stop = range.start + range.len;
        let (kind, class) = if range.start == cursor && stop <= prefix {
            prefix_class(range, range.len)
        } else {
            failures.push(format!(
                "Akita's message at {}..{stop} is outside its masked levels or overlaps",
                range.start
            ));
            (Kind::Clear, None)
        };
        items.push(Item {
            origin: Origin::Narg(range.start..stop),
            kind,
            bytes: range.len,
            tape: None,
            site: Some(site_of(range.context.site_id)),
            class,
        });
        cursor = cursor.max(stop);
    }
    if cursor != prefix {
        failures.push(format!(
            "Akita's record covers its masked levels to {cursor}, not {prefix}"
        ));
        gap(cursor.min(prefix)..prefix, None, &mut items);
    }
    (items, failures)
}

/// The census: bytes per class, or every unclassified operation.
fn census(
    trace: &JoltTrace,
    tape: &MaskedMessages<F>,
    ranges: &[ProofMessageRange],
    regions: &[(&str, usize)],
    narg_len: usize,
    expected: &Expected,
) -> Result<BTreeMap<Class, usize>, Vec<String>> {
    let (mut items, mut failures) = jolt_items(trace, tape, expected);
    let (narg, narg_failures) = narg_items(ranges, regions, narg_len);
    items.extend(narg);
    failures.extend(narg_failures);
    let mut totals = BTreeMap::new();
    for item in items {
        match item.class {
            Some(class) => *totals.entry(class).or_default() += item.bytes,
            None => failures.push(format!("{item:?} has no class")),
        }
    }
    if failures.is_empty() {
        Ok(totals)
    } else {
        Err(failures)
    }
}

fn classify(case: &AkitaFixtureCase, trace: &JoltTrace) -> Vec<Item> {
    let expected = expected_absorbs(case);
    let (mut items, _) = jolt_items(trace, tape_of(&case.proof), &expected);
    items.extend(narg_items(&case.narg_ranges, &case.narg_regions, case.proof.narg.len()).0);
    items
}

fn honest_of(case: &'static AkitaFixtureCase) -> (&'static AkitaFixtureCase, JoltTrace) {
    let trace = jolt_trace(case, &case.proof);
    trace.result.clone().expect("the honest proof verifies");
    assert!(
        !case.narg_ranges.is_empty(),
        "the fixture kept Akita's record of the argument string"
    );
    (case, trace)
}

fn honest() -> (&'static AkitaFixtureCase, JoltTrace) {
    honest_of(whiteout_muldiv_case())
}

/// The Whiteout cases with precommitted groups: both advice kinds, and a
/// committed program (with `field-inline`, every case also opens `FieldInc`,
/// and committed programs are excluded by the field-inline protocol).
fn precommitted_cases() -> Vec<&'static AkitaFixtureCase> {
    let mut cases = vec![whiteout_advice_case()];
    if cfg!(not(feature = "field-inline")) {
        cases.push(whiteout_committed_case());
    }
    cases
}

/// HT1 over one case: every Jolt absorb and every argument-string byte
/// belongs to exactly one class, the commitments and Stage 8 absorb exactly
/// their statement's, the masked tape is absorbed whole and in order, and the
/// argument string's parts tile it, so no private value reaches the sponge
/// clear.
fn assert_census(case: &'static AkitaFixtureCase) {
    let (case, trace) = honest_of(case);
    let totals = census(
        &trace,
        tape_of(&case.proof),
        &case.narg_ranges,
        &case.narg_regions,
        case.proof.narg.len(),
        &expected_absorbs(case),
    )
    .unwrap_or_else(|failures| panic!("unclassified operations:\n{}", failures.join("\n")));
    for class in [
        Class::Commitment,
        Class::Pads,
        Class::MaskedSlot,
        Class::ChallengeDerived,
        Class::Sigma,
        Class::Response,
    ] {
        assert!(
            totals.contains_key(&class),
            "no {class:?} bytes: {totals:?}"
        );
    }
    // Every Jolt stage message is on the tape: the clear proof fields stay
    // empty.
    assert_eq!(
        case.proof.stages,
        JoltStageProofs::masked(case.proof.stages.pow_nonces)
    );
}

/// HT1 on muldiv: the trace group alone.
#[test]
fn whiteout_masked_site_census() {
    assert_census(whiteout_muldiv_case());
}

/// HT1 with precommitted groups (W6f): advice and committed programs open
/// masked, their packing reductions absorb no evaluation.
#[test]
fn whiteout_masked_site_census_with_precommitted_groups() {
    for case in precommitted_cases() {
        assert_census(case);
    }
}

/// `trace` with the masked tape absorb `item` replaced by its clear values
/// (m̂ − p): a Jolt private value routed clear.
fn jolt_routed_clear(case: &AkitaFixtureCase, trace: &JoltTrace, item: &Item) -> JoltTrace {
    let (Origin::Absorb(index), Some(run)) = (&item.origin, &item.tape) else {
        panic!("a masked tape absorb");
    };
    let tape = tape_of(&case.proof);
    let clear: Vec<F> = run
        .clone()
        .map(|slot| tape.values[slot] - case.pads[slot])
        .collect();
    let mut routed = trace.clone();
    routed.after[*index] = encode(&clear);
    routed
}

/// `ranges` with the masked marker of `range` removed: an Akita private
/// value routed clear.
fn akita_routed_clear(
    ranges: &[ProofMessageRange],
    range: &Range<usize>,
) -> Vec<ProofMessageRange> {
    let mut ranges = ranges.to_vec();
    let target = ranges
        .iter_mut()
        .find(|candidate| candidate.start == range.start)
        .expect("the range");
    assert!(target.pads.is_some(), "a masked range");
    target.pads = None;
    ranges
}

/// Positive control (§7): one Jolt claim and one Akita stage-1 value routed
/// clear are each flagged.
#[test]
fn whiteout_census_flags_a_jolt_claim_and_an_akita_value_routed_clear() {
    let (case, trace) = honest();
    let expected = expected_absorbs(case);
    let items = classify(case, &trace);
    let tape = tape_of(&case.proof);
    let narg_len = case.proof.narg.len();
    // The last stage message before the auxiliary block is a stage-7 claim
    // run; the auxiliary block is the tape's last absorb.
    let masked: Vec<&Item> = items
        .iter()
        .filter(|item| item.kind == Kind::Masked && item.tape.is_some())
        .collect();
    let jolt_claim = masked[masked.len() - 2];
    assert!(
        census(
            &jolt_routed_clear(case, &trace, jolt_claim),
            tape,
            &case.narg_ranges,
            &case.narg_regions,
            narg_len,
            &expected
        )
        .is_err(),
        "a Jolt claim routed clear is not flagged"
    );
    let akita_stage1 = items
        .iter()
        .find(|item| {
            item.kind == Kind::Masked
                && item
                    .site
                    .is_some_and(|site| site.family == SITE_FAMILY_STAGE1)
        })
        .expect("Akita's stage 1 sends masked values");
    let Origin::Narg(range) = &akita_stage1.origin else {
        unreachable!("Akita's values are in the argument string");
    };
    assert!(
        census(
            &trace,
            tape,
            &akita_routed_clear(&case.narg_ranges, range),
            &case.narg_regions,
            narg_len,
            &expected
        )
        .is_err(),
        "an Akita stage-1 value routed clear is not flagged"
    );
}

/// Mutation check of the census: routing any masked message clear, masking
/// an Akita message but for its last element, an extra absorb inside the
/// stages or at Stage 8, an extra commitment absorb, and an unrecorded byte
/// of the argument string are each flagged.
#[test]
fn whiteout_census_flags_every_unmasked_message() {
    let (case, trace) = honest();
    let expected = expected_absorbs(case);
    let tape = tape_of(&case.proof);
    let narg_len = case.proof.narg.len();
    let items = classify(case, &trace);
    let flagged = |trace: &JoltTrace, ranges: &[ProofMessageRange], narg_len: usize| {
        census(trace, tape, ranges, &case.narg_regions, narg_len, &expected).is_err()
    };
    assert!(!flagged(&trace, &case.narg_ranges, narg_len));

    let masked: Vec<&Item> = items
        .iter()
        .filter(|item| item.kind == Kind::Masked)
        .collect();
    assert!(masked.len() > 100, "{} masked messages", masked.len());
    let missed: Vec<&Origin> = masked
        .iter()
        .filter(|item| match &item.origin {
            Origin::Absorb(_) => !flagged(
                &jolt_routed_clear(case, &trace, item),
                &case.narg_ranges,
                narg_len,
            ),
            Origin::Narg(range) => !flagged(
                &trace,
                &akita_routed_clear(&case.narg_ranges, range),
                narg_len,
            ),
            Origin::Seed(_) => unreachable!("seeding absorbs are public"),
        })
        .map(|item| &item.origin)
        .collect();
    assert!(missed.is_empty(), "unflagged clear routings: {missed:?}");

    let wide = case
        .narg_ranges
        .iter()
        .find(|range| range.pads.is_some_and(|(_, count)| count > 1))
        .expect("an Akita message of several masked elements");
    let mut partial = case.narg_ranges.clone();
    for range in &mut partial {
        if range.start == wide.start {
            range.pads = range.pads.map(|(first, count)| (first, count - 1));
        }
    }
    assert!(
        flagged(&trace, &partial, narg_len),
        "an Akita message masked but for its last element is not flagged"
    );

    let first_tape = trace
        .after
        .iter()
        .position(|bytes| *bytes == encode(&tape.values[..bytes.len() / FIELD_BYTES]))
        .expect("the first tape absorb");
    for (label, at, bytes) in [
        (
            "a field element absorbed clear inside the stages",
            first_tape + 1,
            vec![0u8; FIELD_BYTES],
        ),
        (
            "an evaluation absorbed at Stage 8",
            trace.after.len(),
            vec![0u8; FIELD_BYTES],
        ),
    ] {
        let mut extra = trace.clone();
        extra.after.insert(at, bytes);
        assert!(
            flagged(&extra, &case.narg_ranges, narg_len),
            "{label} is not flagged"
        );
    }
    let mut extra = trace.clone();
    extra.seed.push(vec![0u8; 32]);
    assert!(
        flagged(&extra, &case.narg_ranges, narg_len),
        "an extra absorb among the commitments is not flagged"
    );
    assert!(
        flagged(&trace, &case.narg_ranges, narg_len + 1),
        "a byte past Akita's last range is not flagged"
    );
}

/// Every proof `tamper` makes from `case`'s proof must reject.
fn assert_rejects(case: &AkitaFixtureCase, label: &str, tamper: impl FnOnce(&mut AkitaJoltProof)) {
    let mut proof = case.proof.clone();
    tamper(&mut proof);
    assert!(case.verify_proof(&proof).is_err(), "{label} is accepted");
}

/// The tamper of one item: a flipped byte at `offset` of an argument-string
/// range, or `+1` on a tape value of a masked absorb, or `+1` on a Jolt
/// nonce, else `None` (a statement-derived absorb has no wire form).
fn tamper_item(
    case: &AkitaFixtureCase,
    item: &Item,
    pick: usize,
) -> Option<(String, AkitaJoltProof)> {
    let mut proof = case.proof.clone();
    match (&item.origin, &item.tape) {
        (Origin::Narg(range), _) if !range.is_empty() => {
            let at = range.start + pick.min(range.len() - 1);
            proof.narg[at] ^= 1;
            Some((format!("narg byte {at} ({:?})", item.class), proof))
        }
        (Origin::Absorb(_), Some(run)) => {
            let at = run.start + pick.min(run.len() - 1);
            tape_mut(&mut proof).values[at] += F::from_u64(1);
            Some((format!("tape value {at} ({:?})", item.class), proof))
        }
        _ => None,
    }
}

/// V5: one tamper at the first, middle and last element of the first,
/// middle and last item of every (class, kind) the census finds, so each
/// class (masked stage values, t_p, Akita's masked and clear messages, the
/// auxiliary block, Σ and the response proof) is exercised; then every tape
/// nonce.
fn assert_tamper_per_census_class(case: &'static AkitaFixtureCase) {
    let (case, trace) = honest_of(case);
    let mut groups: BTreeMap<(Class, Kind), Vec<Item>> = BTreeMap::new();
    for item in classify(case, &trace) {
        if let Some(class) = item.class {
            groups.entry((class, item.kind)).or_default().push(item);
        }
    }
    let mut failures = Vec::new();
    let mut tried = 0;
    for (key, items) in &groups {
        for item in [0, items.len() / 2, items.len() - 1].map(|pick| &items[pick]) {
            for pick in [0, item.bytes / 2, usize::MAX] {
                if let Some((label, proof)) = tamper_item(case, item, pick) {
                    tried += 1;
                    if case.verify_proof(&proof).is_ok() {
                        failures.push(format!("{key:?}: {label}"));
                    }
                }
            }
        }
    }
    for at in 0..tape_of(&case.proof).nonces.len() {
        let mut proof = case.proof.clone();
        tape_mut(&mut proof).nonces[at] ^= 1;
        tried += 1;
        if case.verify_proof(&proof).is_ok() {
            failures.push(format!("tape nonce {at}"));
        }
    }
    assert!(tried > 20, "{tried} tampers");
    assert!(failures.is_empty(), "accepted tampers: {failures:#?}");
}

#[test]
fn whiteout_tamper_per_census_class() {
    assert_tamper_per_census_class(whiteout_muldiv_case());
}

/// [`whiteout_tamper_per_census_class`] with precommitted groups (W6f).
#[test]
fn whiteout_tamper_per_census_class_with_precommitted_groups() {
    for case in precommitted_cases() {
        assert_tamper_per_census_class(case);
    }
}

/// V5: the message sweep, budgeted: flipped bytes spread over every
/// argument-string message and `+1` on values spread over the whole tape,
/// plus the structural tampers (truncations, a trailing byte, an empty
/// string, a dropped or extra tape value or nonce).
#[test]
fn whiteout_muldiv_message_sweep() {
    let (case, trace) = honest();
    let items = classify(case, &trace);
    let mut failures = Vec::new();
    let mut tried = 0;
    for item in items
        .iter()
        .filter(|item| matches!(item.origin, Origin::Narg(_)))
    {
        for pick in [0, item.bytes / 3, 2 * item.bytes / 3, usize::MAX] {
            if let Some((label, proof)) = tamper_item(case, item, pick) {
                tried += 1;
                if case.verify_proof(&proof).is_ok() {
                    failures.push(label);
                }
            }
        }
    }
    let values = tape_of(&case.proof).values.len();
    let step = (values / 256).max(1);
    for at in (0..values).step_by(step).chain([values - 1]) {
        let mut proof = case.proof.clone();
        tape_mut(&mut proof).values[at] += F::from_u64(1);
        tried += 1;
        if case.verify_proof(&proof).is_ok() {
            failures.push(format!("tape value {at}"));
        }
    }
    assert!(tried > 300, "{tried} tampers");
    assert!(failures.is_empty(), "accepted tampers: {failures:#?}");

    let narg = case.proof.narg.len();
    for len in [0, 1, narg / 4, narg / 2, narg - 1] {
        assert_rejects(case, &format!("the string truncated to {len}"), |proof| {
            proof.narg.truncate(len);
        });
    }
    assert_rejects(case, "a trailing byte", |proof| proof.narg.push(0));
    assert_rejects(case, "a dropped tape value", |proof| {
        let _ = tape_mut(proof).values.pop();
    });
    assert_rejects(case, "an extra tape value", |proof| {
        tape_mut(proof).values.push(F::from_u64(0));
    });
    assert_rejects(case, "an extra tape nonce", |proof| {
        tape_mut(proof).nonces.push(0);
    });
    if !tape_of(&case.proof).nonces.is_empty() {
        assert_rejects(case, "a dropped tape nonce", |proof| {
            let _ = tape_mut(proof).nonces.pop();
        });
    }
}

/// A47: the cut sends u_a compressed, one 16-coefficient payload per chunk
/// of Whiteout's split chain, whose chunks hold up to 15,115 coefficients
/// at the 169-bit SIS tables (O5, D34): muldiv's u_a is one chunk payload.
#[test]
fn whiteout_compressed_residual_commitment_is_one_chunk_payload() {
    let (case, _) = honest();
    let cut = case
        .narg_ranges
        .iter()
        .find(|range| {
            let site = site_of(range.context.site_id);
            site.family == SITE_FAMILY_NEXT_WITNESS && site.level == ZK_CUT_LEVEL
        })
        .expect("the cut level's successor payload");
    assert_eq!(
        cut.context.atom_count, 16,
        "the cut sends one chunk payload"
    );
}

/// N4: one byte of an Akita masked-level message, and a value in each part
/// of the auxiliary block (Akita's first, Jolt's last), each reject. Akita's
/// rows are pinned by `shifts`.
#[test]
fn whiteout_masked_level_and_auxiliary_block_tampers() {
    let (case, trace) = honest();
    let items = classify(case, &trace);
    let akita_level = items
        .iter()
        .find(|item| item.kind == Kind::Masked && matches!(item.origin, Origin::Narg(_)))
        .expect("Akita's masked levels send masked values");
    let Origin::Narg(level) = &akita_level.origin else {
        unreachable!("an argument-string range");
    };
    assert_rejects(
        case,
        "a flipped byte of an Akita masked-level value",
        |proof| {
            proof.narg[level.start] ^= 1;
        },
    );
    let block = items
        .iter()
        .rev()
        .find_map(|item| item.tape.clone())
        .expect("the auxiliary block is the tape's last message");
    for (label, at) in [
        ("the auxiliary block's Akita part", block.start),
        ("the auxiliary block's Jolt part", block.end - 1),
    ] {
        assert_rejects(case, label, |proof| {
            tape_mut(proof).values[at] += F::from_u64(1);
        });
    }
}

/// HT3: a rerun after an aborted attempt shares no commitment, pad
/// commitment or masked message with it, and its proof verifies.
#[test]
fn whiteout_rerun_reuses_nothing_from_the_aborted_attempt() {
    let (case, aborted) = whiteout_rerun_case();
    jolt_trace(&case, &case.proof)
        .result
        .expect("the rerun's proof verifies");
    use jolt_transcript::Encoding as _;
    assert_ne!(
        case.proof.commitments.encode().as_ref(),
        aborted.commitment.as_slice(),
        "the trace commitment repeats the aborted attempt"
    );
    let tape = encode(&tape_of(&case.proof).values);
    let shared = tape.len().min(aborted.tape.len()) / FIELD_BYTES;
    assert!(
        shared > 100,
        "the attempts share a long tape prefix to compare"
    );
    for value in 0..shared {
        let range = value * FIELD_BYTES..(value + 1) * FIELD_BYTES;
        assert_ne!(
            tape[range.clone()],
            aborted.tape[range],
            "masked tape value {value} repeats the aborted attempt"
        );
    }
    let mut compared = 0;
    for item in narg_items(&case.narg_ranges, &case.narg_regions, case.proof.narg.len()).0 {
        let Origin::Narg(range) = item.origin else {
            continue;
        };
        let fresh = matches!(
            item.class,
            Some(Class::Commitment | Class::Pads | Class::MaskedSlot)
        );
        if fresh && range.end <= aborted.narg.len() && range.len() >= FIELD_BYTES {
            assert_ne!(
                case.proof.narg[range.clone()],
                aborted.narg[range.clone()],
                "{:?} at {range:?} repeats the aborted attempt",
                item.class
            );
            compared += 1;
        }
    }
    assert!(
        compared > 10,
        "{compared} argument-string messages compared"
    );
}

/// V4 counting equality: the pad layout depends on the proof's shape only:
/// the honest proof, its tape zeroed, and its tape and argument string
/// randomized give the same layout, which fills exactly the prover's pads.
#[test]
fn whiteout_pad_layout_counts_the_same_on_any_bytes() {
    let (case, _) = honest();
    let layout = |proof: &AkitaJoltProof| {
        let (checked, _) = validate_and_seed_transcript(
            &case.preprocessing,
            &case.public_io,
            proof,
            case.trusted_advice_commitment.as_ref(),
            |statement| verifier_transcript::<Keccak>(&JOLT_SESSION, statement, &proof.narg),
        )
        .expect("the statement seeds");
        let frame = whiteout_frame(
            &case.preprocessing,
            proof,
            &checked,
            case.trusted_advice_commitment.as_ref(),
        )
        .expect("the frame");
        pad_layout(&case.preprocessing, proof, &checked, &frame).expect("pad layout")
    };
    let honest = layout(&case.proof);
    assert_eq!(honest.layout().num_pads, case.pads.len());
    let mut zeroed = case.proof.clone();
    for value in &mut tape_mut(&mut zeroed).values {
        *value = F::from_u64(0);
    }
    assert_eq!(layout(&zeroed), honest);
    let mut random = case.proof.clone();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for value in &mut tape_mut(&mut random).values {
        *value = F::from_u64(next());
    }
    for byte in &mut random.narg {
        *byte = next().to_le_bytes()[0];
    }
    assert_eq!(layout(&random), honest);
}

/// V5: the statement and proof-shape tampers, and the Whiteout proof
/// declared as a clear or BlindFold proof.
#[test]
fn whiteout_statement_and_mode_tampers() {
    let case = whiteout_muldiv_case();
    for zk in [ZkConfig::Transparent, ZkConfig::BlindFold] {
        let mut proof = case.proof.clone();
        proof.protocol = JoltProtocolConfig {
            zk,
            ..proof.protocol
        };
        assert!(matches!(
            case.verify_proof(&proof),
            Err(VerifierError::ProtocolConfigMismatch { .. })
        ));
    }
    assert_rejects(case, "a doubled trace length", |proof| {
        proof.trace_length *= 2;
    });
    assert_rejects(case, "a halved trace length", |proof| {
        proof.trace_length /= 2;
    });
    assert_rejects(case, "a doubled RAM size", |proof| proof.ram_K *= 2);
    assert_rejects(case, "a missing RV64 increment commitment", |proof| {
        proof.fused_inc_commitment = None;
    });
    assert_rejects(
        case,
        "the trace commitment in the increments' place",
        |proof| {
            proof.fused_inc_commitment = Some(proof.commitments.clone());
        },
    );
    assert_rejects(
        case,
        "an untrusted advice commitment the statement lacks",
        |proof| {
            proof.untrusted_advice_commitment = Some(proof.commitments.clone());
        },
    );
    assert_rejects(case, "a clear round beside the masked tape", |proof| {
        proof.stages.stage3_sumcheck_proof =
            SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
                round_polynomials: vec![CompressedPoly::new(vec![F::from_u64(1)])],
                pow_nonces: Vec::new(),
            }));
    });
    let mut io = case.public_io.clone();
    if let Some(byte) = io.outputs.first_mut() {
        *byte ^= 1;
    } else {
        io.outputs.push(1);
    }
    assert!(
        jolt_verifier::whiteout::verify::<F, AkitaScheme, AkitaVc, VerifierState<'_, Keccak>>(
            &case.preprocessing,
            &io,
            &case.proof,
            case.trusted_advice_commitment.as_ref(),
        )
        .is_err(),
        "a proof verifies against other outputs"
    );
}

/// Shifts of the final-opening claims, with every auxiliary value
/// recomputed as a consistent cheater would: the honest pads, the relation,
/// and the residual witness (bound to the trace commitment).
mod shifts {
    use jolt_akita::{AkitaField, AkitaScheme};
    use jolt_algebra::{Field, One, Zero};
    use jolt_prover::akita::preprocessing::AkitaVc;
    use jolt_transcript::{verifier_transcript, ClaimDomain, Keccak};
    use jolt_verifier::whiteout::{
        pad_layout, whiteout_frame, whiteout_prefix, whiteout_relation, whiteout_stages,
    };
    use jolt_verifier::{validate_and_seed_transcript, JOLT_SESSION};
    use jolt_zk::masking::{ResidualRelation, ResidualVar, ZkError};
    use jolt_zk::whiteout::MaskedGroupOpening;

    use crate::support::akita_fixtures::{
        whiteout_advice_case, whiteout_muldiv_case, whiteout_rerun_case, AkitaFixtureCase,
    };

    type F = AkitaField;

    /// Akita's auxiliary values without its checks.
    #[derive(Default)]
    struct CheaterValues {
        auxiliary: Vec<F>,
    }

    impl ClaimDomain<F> for CheaterValues {
        type Value = F;
        type Error = ZkError;

        fn constant(&self, value: F) -> F {
            value
        }

        fn add(&self, lhs: &F, rhs: &F) -> F {
            *lhs + *rhs
        }

        fn scale(&self, value: &F, scale: F) -> F {
            *value * scale
        }

        fn define_auxiliary(&mut self, value: F) -> Result<F, ZkError> {
            self.auxiliary.push(value);
            Ok(value)
        }

        fn product(&mut self, x: F, y: F) -> Result<F, ZkError> {
            self.auxiliary.push(x * y);
            Ok(x * y)
        }

        fn products_into(&mut self, _: (F, F), _: Vec<(F, F)>, _: F) -> Result<(), ZkError> {
            Ok(())
        }

        fn check_zero(&mut self, _: F) -> Result<(), ZkError> {
            Ok(())
        }
    }

    /// A basis of `{x : Σ_k x_k · columns[k] = 0}`, and the rank.
    fn null_space(columns: &[Vec<F>]) -> (Vec<Vec<F>>, usize) {
        let width = columns.len();
        let height = columns.first().map_or(0, Vec::len);
        let mut rows: Vec<Vec<F>> = (0..height)
            .map(|row| columns.iter().map(|column| column[row]).collect())
            .collect();
        let mut pivots = Vec::new();
        for column in 0..width {
            let rank = pivots.len();
            let Some(pivot) = (rank..height).find(|&row| !rows[row][column].is_zero()) else {
                continue;
            };
            rows.swap(rank, pivot);
            let inverse = rows[rank][column].inverse().expect("nonzero pivot");
            for entry in &mut rows[rank] {
                *entry *= inverse;
            }
            let pivot_row = rows[rank].clone();
            for (row, values) in rows.iter_mut().enumerate() {
                if row != rank && !values[column].is_zero() {
                    let factor = values[column];
                    for (entry, pivot_entry) in values.iter_mut().zip(&pivot_row) {
                        *entry -= factor * *pivot_entry;
                    }
                }
            }
            pivots.push(column);
        }
        let basis = (0..width)
            .filter(|column| !pivots.contains(column))
            .map(|free| {
                let mut vector = vec![F::zero(); width];
                vector[free] = F::one();
                for (row, &pivot) in pivots.iter().enumerate() {
                    vector[pivot] = -rows[row][free];
                }
                vector
            })
            .collect();
        (basis, pivots.len())
    }

    /// Pins a known protocol property (D25): Akita's rows apply exactly one
    /// functional to the final-opening claims. If Akita adds binding (more
    /// rows, or a second functional), this test fails; update D25 and the
    /// spec's prototype deltas with the new argument, then this test.
    ///
    /// N1: a shift of the final-opening claims that Jolt's rows cannot see
    /// meets Akita's rows only through one functional, drawn by Akita's
    /// EvaluationBatch after t_p and every masked value. So a shift in that
    /// functional's kernel passes every row: the binding is Schwartz–Zippel
    /// over the batch challenges, not a row the shift could not satisfy.
    /// The functional weights every claim and is fresh per proof: a rerun of
    /// the same statement draws another one.
    #[test]
    fn whiteout_final_openings_meet_akita_through_the_batch_challenges() {
        let first = assert_shifts_meet_akita_through_one_functional(whiteout_muldiv_case(), 2);
        let (rerun, _) = whiteout_rerun_case();
        let second = assert_shifts_meet_akita_through_one_functional(&rerun, 2);
        assert_eq!(first.len(), second.len());
        assert_ne!(
            first, second,
            "the functional is drawn afresh for each proof"
        );
    }

    /// The same with precommitted groups (W6f): untrusted and trusted
    /// advice's claims join the trace group's under the one functional.
    /// Their stage-4 openings also feed Jolt's RAM value check, so Jolt's
    /// rows see one more combination of the claims.
    #[test]
    fn whiteout_advice_final_openings_meet_akita_through_the_batch_challenges() {
        let _ = assert_shifts_meet_akita_through_one_functional(whiteout_advice_case(), 3);
    }

    /// `jolt_seen` is the rank of Jolt's rows on the final-opening claims.
    /// Returns Akita's functional on the claims, normalized to its first
    /// coefficient.
    fn assert_shifts_meet_akita_through_one_functional(
        case: &AkitaFixtureCase,
        jolt_seen: usize,
    ) -> Vec<F> {
        let pads = &case.pads;
        let proof = &case.proof;
        let (checked, mut transcript) = validate_and_seed_transcript(
            &case.preprocessing,
            &case.public_io,
            proof,
            case.trusted_advice_commitment.as_ref(),
            |statement| verifier_transcript::<Keccak>(&JOLT_SESSION, statement, &proof.narg),
        )
        .expect("the statement seeds");
        let frame = whiteout_frame(
            &case.preprocessing,
            proof,
            &checked,
            case.trusted_advice_commitment.as_ref(),
        )
        .expect("the frame");
        let derived = pad_layout(&case.preprocessing, proof, &checked, &frame).expect("pad layout");
        let layout = derived.layout().clone();
        let mut stages = whiteout_stages::<F, AkitaScheme, AkitaVc, _, _>(
            &case.preprocessing,
            proof,
            &checked,
            case.trusted_advice_commitment.as_ref(),
            &mut transcript,
            Some(&derived),
        )
        .expect("masked stages");
        let prefix =
            whiteout_prefix::<F, AkitaScheme, _>(&mut stages, &mut transcript).expect("prefix");
        let relation = whiteout_relation(&mut stages, &prefix, &mut transcript).expect("relation");

        let claims: Vec<_> = stages
            .claims
            .iter()
            .flat_map(|group| &group.evaluations)
            .collect();
        let witness = [prefix
            .residual_claim()
            .witness_eval()
            .evaluate(pads, &[])
            .expect("cut evaluation")];
        assert_eq!(
            relation.first_violation(pads, &witness).expect("rows"),
            None
        );

        let akita_aux = prefix
            .auxiliary_values(pads)
            .expect("Akita auxiliary")
            .len();
        let auxiliary = |pads: &[F]| -> Vec<F> {
            let mut akita = CheaterValues::default();
            prefix
                .lower(&mut akita, |form| form.evaluate(pads, &[]))
                .expect("Akita auxiliary values");
            let mut values = akita.auxiliary;
            values.extend(
                stages
                    .residual
                    .auxiliary_values(|operand| Ok(operand.masked - pads[operand.pad.0]))
                    .expect("Jolt auxiliary values"),
            );
            values
        };
        let masked_auxiliary: Vec<F> = auxiliary(pads)
            .iter()
            .zip(&pads[layout.auxiliary.clone()])
            .map(|(value, pad)| *value + *pad)
            .collect();
        let mut akita_rows = ResidualRelation::new(layout.clone(), 1);
        AkitaScheme::append_prefix_rows(&prefix, &masked_auxiliary[..akita_aux], &mut akita_rows)
            .expect("Akita rows");
        let (akita_affine, akita_products) = (akita_rows.affine.len(), akita_rows.products.len());

        // Each claim's form reads one base pad, with a coefficient: −1 for a
        // claim that is its own masked slot, a packing weight times −1 for a
        // packed precommitted claim. Lowering the pad by one moves the claim
        // by `gains[claim]`.
        let (slots, gains): (Vec<usize>, Vec<F>) = claims
            .iter()
            .map(|form| {
                let pads: Vec<(usize, F)> = form
                    .terms
                    .iter()
                    .filter_map(|(variable, coefficient)| match variable {
                        ResidualVar::Pad(pad) => Some((pad.0, *coefficient)),
                        ResidualVar::Witness(_) => None,
                    })
                    .collect();
                assert_eq!(pads.len(), 1, "a final-opening claim reads one masked slot");
                (pads[0].0, -pads[0].1)
            })
            .unzip();
        let cheat = |delta: &[F]| -> Vec<F> {
            let mut shifted = pads.clone();
            for (&slot, &by) in slots.iter().zip(delta) {
                shifted[slot] -= by;
            }
            let values = auxiliary(&shifted);
            for (index, slot) in layout.auxiliary.clone().enumerate() {
                shifted[slot] = masked_auxiliary[index] - values[index];
            }
            shifted
        };
        let rows = |assignment: &[F], akita: bool| -> Vec<F> {
            let (affine, products) = if akita {
                (
                    &relation.affine[..akita_affine],
                    &relation.products[..akita_products],
                )
            } else {
                (
                    &relation.affine[akita_affine..],
                    &relation.products[akita_products..],
                )
            };
            affine
                .iter()
                .map(|row| row.evaluate(assignment, &witness).expect("row"))
                .chain(
                    products
                        .iter()
                        .map(|row| row.evaluate(assignment, &witness).expect("row")),
                )
                .collect()
        };
        let unit = |claim: usize| -> Vec<F> {
            let mut delta = vec![F::zero(); slots.len()];
            delta[claim] = F::one();
            delta
        };
        let jolt: Vec<Vec<F>> = (0..slots.len())
            .map(|claim| rows(&cheat(&unit(claim)), false))
            .collect();
        let akita: Vec<Vec<F>> = (0..slots.len())
            .map(|claim| rows(&cheat(&unit(claim)), true))
            .collect();

        let (jolt_null, _) = null_space(&jolt);
        assert_eq!(
            jolt_null.len(),
            slots.len() - jolt_seen,
            "the rank of Jolt's rows on the claims"
        );
        let (_, akita_rank) = null_space(&akita);
        assert_eq!(
            akita_rank, 1,
            "Akita's rows apply one functional to the claims"
        );

        // The functional φ on the claims, read off one row Akita's rows move.
        let row = (0..akita[0].len())
            .find(|&row| akita.iter().any(|column| !column[row].is_zero()))
            .expect("a claim shift moves an Akita row");
        let phi_of: Vec<F> = akita.iter().map(|column| column[row]).collect();
        assert!(
            phi_of.iter().all(|coefficient| !coefficient.is_zero()),
            "Akita's functional weights every claim"
        );
        let phi = |delta: &[F]| -> F {
            delta
                .iter()
                .zip(&phi_of)
                .fold(F::zero(), |sum, (by, coefficient)| sum + *by * *coefficient)
        };

        // Two Jolt-invisible shifts combined into φ's kernel pass every row.
        let (first, second) = {
            let mut moving = jolt_null.iter().filter(|vector| !phi(vector).is_zero());
            (
                moving.next().expect("a shift Akita sees"),
                moving.next().expect("a second shift Akita sees"),
            )
        };
        let ratio = phi(first) * phi(second).inverse().expect("nonzero");
        let combined: Vec<F> = first
            .iter()
            .zip(second)
            .map(|(a, b)| *a - ratio * *b)
            .collect();
        assert!(combined.iter().any(|by| !by.is_zero()));
        assert!(
            rows(&cheat(first), true)
                .iter()
                .any(|value| !value.is_zero()),
            "a single Jolt-invisible shift fails Akita's rows"
        );
        assert_eq!(
            relation
                .first_violation(&cheat(&combined), &witness)
                .expect("rows"),
            None,
            "a shift in the batch functional's kernel passes under fixed challenges"
        );
        let normalizer = (phi_of[0] * gains[0]).inverse().expect("nonzero");
        phi_of
            .iter()
            .zip(&gains)
            .map(|(coefficient, gain)| *coefficient * normalizer * *gain)
            .collect()
    }
}
