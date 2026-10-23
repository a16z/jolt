//! Whiteout (zero knowledge over Akita): the masked-site census (HT1,
//! `specs/whiteout-jolt-integration.md` §7) and the tamper suite (V5).

#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "census and tamper checks fail loudly"
)]

use std::collections::BTreeMap;

use akita_params::transcript_site::{
    SITE_FAMILY_FOLD_BINDING, SITE_FAMILY_NEXT_WITNESS, SITE_FAMILY_OPENING_PAYLOAD,
    SITE_FAMILY_ROOT_STATEMENT, SITE_FAMILY_STAGE1, SITE_FAMILY_STAGE2, SITE_FAMILY_SUMCHECK,
};
use akita_params::{GrindingSite, ProtocolSiteId};
use jolt_akita::AkitaField;
use jolt_akita::AkitaScheme;
use jolt_algebra::CanonicalBytes;
use jolt_prover::akita::preprocessing::AkitaVc;
use jolt_transcript::{SiteId, TranscriptEvent, TranscriptOp};
use jolt_verifier::sites::{STAGE8, WHITEOUT, WHITEOUT_SIGMA};
use jolt_verifier::whiteout::pad_layout;
use jolt_verifier::{JoltProtocolConfig, JoltSponge, VerifierError, ZkConfig};

use super::checks;
use crate::support::akita_fixtures::{whiteout_muldiv_case, whiteout_rerun_case, AkitaFixtureCase};
use crate::support::narg::{assert_coverage_closure, Region, Trace, TracedCase};

const FIELD_BYTES: usize = <AkitaField as CanonicalBytes>::NUM_BYTES;

/// Akita's response prover's site (`akita_pcs::zk::RESPONSE_SITE`).
const RESPONSE_SITE: SiteId = SiteId::label("whiteout/response");

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

/// How a transcript operation reaches the sponge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A prover message sent masked over its pads.
    Masked,
    /// A prover message absorbed as sent.
    Clear,
    /// A verifier-known value absorbed.
    Public,
}

/// Who set an event's site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Site {
    Jolt(SiteId),
    Response,
    Akita(ProtocolSiteId),
}

impl Site {
    fn of(site: SiteId) -> Self {
        if site.0.starts_with(b"jolt/") {
            return Self::Jolt(site);
        }
        if site == RESPONSE_SITE {
            return Self::Response;
        }
        Self::Akita(ProtocolSiteId::from_bytes(site.0))
    }

    /// The Akita family, if Akita set the site.
    fn akita_family(self) -> Option<u32> {
        match self {
            Self::Akita(site) => Some(site.family),
            Self::Jolt(_) | Self::Response => None,
        }
    }
}

/// An Akita grinding nonce: a grinding site (`GrindingSite`, whose families
/// start at the evaluation batch's) with a nonce width in its detail.
fn is_grinding_nonce(site: Site) -> bool {
    let first = GrindingSite::EvaluationBatch { level: 0 }.site_id(0).family;
    matches!(site, Site::Akita(site) if site.family >= first && site.detail != 0)
}

/// Where an operation sits: its region, and whether Akita's response proof
/// has begun (its first event carries the response site).
#[derive(Clone, Copy, Debug)]
struct Position {
    region: Region,
    in_response: bool,
}

/// The class of a `kind` operation at `site` and `at`, if the protocol has
/// one there. Akita's masked-level rows follow
/// `akita:specs/whiteout-akita.md` §4.2: rounds and their claims masked
/// (sumcheck, stage 2, stage 1), the D image and u_a public but hidden by
/// their randomness, the root statement and fold bindings public, grinding
/// nonces public.
fn class(at: Position, site: Site, kind: Kind) -> Option<Class> {
    use Class as C;
    use Kind as K;
    use Region as R;
    let family = site.akita_family();
    match (at.region, site, kind) {
        (R::Preamble, Site::Jolt(_), K::Clear | K::Public) => Some(C::Statement),
        (R::Commitments, Site::Jolt(_), K::Clear) => Some(C::Commitment),
        (R::Pads, Site::Jolt(_), K::Clear) => Some(C::Pads),
        (
            R::Stage1
            | R::Stage2
            | R::Stage3
            | R::Stage4
            | R::Stage5
            | R::Stage6a
            | R::Stage6b
            | R::Stage7,
            Site::Jolt(_),
            K::Masked,
        ) => Some(C::MaskedSlot),
        // The catalog digest and the selected row's digest, absorbed only
        // (`stage8_digests` pins their count and size).
        (R::Stage8, Site::Jolt(site), K::Public) if site == STAGE8 => Some(C::Statement),
        (R::Stage8, Site::Akita(_), K::Masked)
            if matches!(
                family,
                Some(SITE_FAMILY_SUMCHECK | SITE_FAMILY_STAGE2 | SITE_FAMILY_STAGE1)
            ) =>
        {
            Some(C::MaskedSlot)
        }
        (R::Stage8, Site::Akita(_), K::Clear)
            if matches!(
                family,
                Some(SITE_FAMILY_OPENING_PAYLOAD | SITE_FAMILY_NEXT_WITNESS)
            ) =>
        {
            Some(C::Commitment)
        }
        (R::Stage8, Site::Akita(_), K::Public) if family == Some(SITE_FAMILY_ROOT_STATEMENT) => {
            Some(C::Statement)
        }
        (R::Stage8, Site::Akita(_), K::Public) if family == Some(SITE_FAMILY_FOLD_BINDING) => {
            Some(C::ChallengeDerived)
        }
        (R::Stage8, _, K::Clear) if is_grinding_nonce(site) => Some(C::ChallengeDerived),
        // The auxiliary block, then the Σ-protocol's messages and targets.
        (R::Whiteout, Site::Jolt(site), K::Masked) if site == WHITEOUT && !at.in_response => {
            Some(C::MaskedSlot)
        }
        (R::Whiteout, Site::Jolt(site), K::Clear | K::Public)
            if site == WHITEOUT_SIGMA && !at.in_response =>
        {
            Some(C::Sigma)
        }
        // The response proof: a clear Akita proof of the Σ response, after
        // its first site.
        (R::Whiteout, Site::Response | Site::Akita(_), K::Clear | K::Public) if at.in_response => {
            Some(C::Response)
        }
        _ => None,
    }
}

/// One classified operation.
#[derive(Debug)]
struct Classified {
    index: usize,
    region: Region,
    kind: Kind,
    bytes: usize,
    class: Option<Class>,
}

/// Classifies every message and public absorb of `trace`. A message is
/// masked when the next event marks it masked over exactly its own width.
fn classify(trace: &Trace) -> Vec<Classified> {
    let regions = trace.regions();
    let events: &[TranscriptEvent] = &trace.events;
    let mut out = Vec::new();
    let mut in_response = false;
    for (index, event) in events.iter().enumerate() {
        in_response |= event.site == RESPONSE_SITE;
        let kind = match event.op {
            TranscriptOp::Message => match events.get(index + 1).map(|next| next.op) {
                Some(TranscriptOp::Masked { .. })
                    if events[index + 1].len * FIELD_BYTES == event.len =>
                {
                    Kind::Masked
                }
                _ => Kind::Clear,
            },
            TranscriptOp::Public => Kind::Public,
            TranscriptOp::Challenge
            | TranscriptOp::ForkAbsorb
            | TranscriptOp::ForkSqueeze
            | TranscriptOp::Masked { .. } => continue,
        };
        out.push(Classified {
            index,
            region: regions[index],
            kind,
            bytes: event.len,
            class: class(
                Position {
                    region: regions[index],
                    in_response,
                },
                Site::of(event.site),
                kind,
            ),
        });
    }
    out
}

/// Stage 8's own absorbs: the catalog digest and the selected row's
/// digest, 32 bytes each, and nothing else at its site.
const STAGE8_DIGESTS: [usize; 2] = [32, 32];

/// The census: bytes per class, or every unclassified operation.
fn census(trace: &Trace) -> Result<BTreeMap<Class, usize>, Vec<String>> {
    let mut totals = BTreeMap::new();
    let mut failures = Vec::new();
    let stage8: Vec<usize> = trace
        .events
        .iter()
        .filter(|event| event.site == STAGE8 && event.op != TranscriptOp::Challenge)
        .map(|event| event.len)
        .collect();
    if stage8 != STAGE8_DIGESTS {
        failures.push(format!(
            "Stage 8's own absorbs are {stage8:?}, not its two digests"
        ));
    }
    for item in classify(trace) {
        match item.class {
            Some(class) => *totals.entry(class).or_default() += item.bytes,
            None => failures.push(format!(
                "event {} ({:?} {:?}, {} bytes, site {:?}) has no class",
                item.index,
                item.region,
                item.kind,
                item.bytes,
                Site::of(trace.events[item.index].site),
            )),
        }
    }
    if failures.is_empty() {
        Ok(totals)
    } else {
        Err(failures)
    }
}

fn honest() -> (&'static AkitaFixtureCase, Trace) {
    let case = whiteout_muldiv_case();
    let trace = case.honest_trace();
    assert_coverage_closure(&trace, case.proof().narg.len());
    (case, trace)
}

/// HT1: every received byte and public absorb of an honest Whiteout proof
/// belongs to exactly one class, and the message bytes tile the argument
/// string (coverage closure), so no private value reaches the sponge clear.
#[test]
fn whiteout_masked_site_census() {
    let (case, trace) = honest();
    let totals = census(&trace)
        .unwrap_or_else(|failures| panic!("unclassified operations:\n{}", failures.join("\n")));
    let message_bytes: usize = classify(&trace)
        .iter()
        .filter(|item| item.kind != Kind::Public)
        .map(|item| item.bytes)
        .sum();
    assert_eq!(message_bytes, case.proof().narg.len());
    for class in [
        Class::Statement,
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
}

/// `trace` with the masked marker of event `message` removed: its value
/// routed clear.
fn routed_clear(trace: &Trace, message: usize) -> Trace {
    let mut events = trace.events.clone();
    assert!(matches!(
        events[message + 1].op,
        TranscriptOp::Masked { .. }
    ));
    let _ = events.remove(message + 1);
    Trace {
        events,
        result: Ok(()),
    }
}

fn masked_messages(trace: &Trace) -> Vec<Classified> {
    classify(trace)
        .into_iter()
        .filter(|item| item.kind == Kind::Masked)
        .collect()
}

/// Positive control (§7): one Jolt claim and one Akita stage-1 value
/// routed clear are each flagged.
#[test]
fn whiteout_census_flags_a_jolt_claim_and_an_akita_value_routed_clear() {
    let (_, trace) = honest();
    let masked = masked_messages(&trace);
    let jolt_claim = masked
        .iter()
        .rfind(|item| item.region == Region::Stage3)
        .expect("Stage 3 sends masked claims");
    let akita_stage1 = masked
        .iter()
        .find(|item| {
            matches!(
                Site::of(trace.events[item.index].site),
                Site::Akita(ProtocolSiteId {
                    family: SITE_FAMILY_STAGE1,
                    ..
                })
            )
        })
        .expect("Akita's stage 1 sends masked values");
    for item in [jolt_claim, akita_stage1] {
        assert!(
            census(&routed_clear(&trace, item.index)).is_err(),
            "{item:?} routed clear is not flagged"
        );
    }
}

/// Mutation check of the census: removing any masked marker, masking only
/// part of a message, or adding a public absorb inside a masked stage is
/// flagged.
#[test]
fn whiteout_census_flags_every_unmasked_message() {
    let (_, trace) = honest();
    let masked = masked_messages(&trace);
    assert!(masked.len() > 100);
    let missed: Vec<usize> = masked
        .iter()
        .filter(|item| census(&routed_clear(&trace, item.index)).is_ok())
        .map(|item| item.index)
        .collect();
    assert!(missed.is_empty(), "unflagged clear routings: {missed:?}");

    let wide = masked
        .iter()
        .find(|item| item.bytes > FIELD_BYTES)
        .expect("a masked message of several elements");
    let mut events = trace.events.clone();
    events[wide.index + 1].len -= 1;
    assert!(
        census(&Trace {
            events,
            result: Ok(())
        })
        .is_err(),
        "a message masked but for its last element is not flagged"
    );

    // A clear-only absorb at Stage 8's site (an evaluation, as the clear
    // build's reduction absorbs): a public absorb, then a clear message.
    let digest = trace
        .events
        .iter()
        .position(|event| event.site == STAGE8)
        .expect("Stage 8's digests");
    for op in [TranscriptOp::Public, TranscriptOp::Message] {
        let mut events = trace.events.clone();
        let mut extra = events[digest].clone();
        extra.op = op;
        extra.len = FIELD_BYTES;
        events.insert(digest + 1, extra);
        assert!(
            census(&Trace {
                events,
                result: Ok(())
            })
            .is_err(),
            "a clear {op:?} at Stage 8's site is not flagged"
        );
    }

    let stage5 = masked
        .iter()
        .find(|item| item.region == Region::Stage5)
        .expect("Stage 5 sends masked messages");
    let mut events = trace.events.clone();
    let public = TranscriptEvent {
        op: TranscriptOp::Public,
        narg: None,
        ..events[stage5.index].clone()
    };
    events.insert(stage5.index, public);
    assert!(
        census(&Trace {
            events,
            result: Ok(())
        })
        .is_err(),
        "a public absorb inside Stage 5 is not flagged"
    );
}

/// V5: the single-byte sweep (each region's first and last tamper kept)
/// rejects by Whiteout's deadline, plus truncation, a trailing byte, an
/// empty string and wrong protocol words.
#[test]
fn whiteout_muldiv_message_sweep() {
    let case = whiteout_muldiv_case();
    checks::message_sweep(case, Some(512));
    checks::structural_tampers(case, Some(64));
}

/// V5: one tamper at the first, middle and last message of every (region,
/// class) the census finds, so each class (masked stage values, t_p, the
/// hiding commitment, Akita's masked and public messages, both parts of the
/// auxiliary block, Σ and the response proof) is exercised.
#[test]
fn whiteout_tamper_per_census_class() {
    let (case, trace) = honest();
    let mut groups: BTreeMap<(Region, Class, bool), Vec<std::ops::Range<usize>>> = BTreeMap::new();
    for item in classify(&trace) {
        if let (Some(class), Some(range)) = (item.class, trace.events[item.index].narg.clone()) {
            groups
                .entry((item.region, class, item.kind == Kind::Masked))
                .or_default()
                .push(range);
        }
    }
    let mut failures = Vec::new();
    for (key, ranges) in &groups {
        let picks = [0, ranges.len() / 2, ranges.len() - 1];
        for range in picks.iter().map(|&pick| &ranges[pick]) {
            if range.is_empty() {
                continue;
            }
            let mut tampered = case.proof().clone();
            tampered.narg[range.start + range.len() / 2] ^= 1;
            if case.verify_proof(&tampered).is_ok() {
                failures.push(format!("{key:?} at {range:?}"));
            }
        }
    }
    // The auxiliary block is one message: Akita's part leads, Jolt's follows.
    let aux = groups
        .get(&(Region::Whiteout, Class::MaskedSlot, true))
        .and_then(|ranges| ranges.first())
        .expect("the auxiliary block");
    for offset in [0, aux.len() - 1] {
        let mut tampered = case.proof().clone();
        tampered.narg[aux.start + offset] ^= 1;
        if case.verify_proof(&tampered).is_ok() {
            failures.push(format!("auxiliary block byte {offset}"));
        }
    }
    assert!(failures.is_empty(), "accepted tampers: {failures:#?}");
}

/// N4: one byte of an Akita masked-level message, and one in each part of
/// the auxiliary block (Akita's first, Jolt's last), each reject. Byte
/// tampers re-randomize every later challenge, so they are load-bearing for
/// the Σ check, not for the rows; Akita's rows are pinned by
/// `shifts::whiteout_final_openings_meet_akita_through_the_batch_challenges`.
#[test]
fn whiteout_masked_level_and_auxiliary_block_tampers() {
    let (case, trace) = honest();
    let masked = masked_messages(&trace);
    let akita_level = masked
        .iter()
        .find(|item| {
            item.region == Region::Stage8
                && matches!(Site::of(trace.events[item.index].site), Site::Akita(_))
        })
        .expect("Akita's masked levels send masked values");
    let aux = masked
        .iter()
        .find(|item| item.region == Region::Whiteout)
        .expect("the auxiliary block");
    let range = |item: &Classified| {
        trace.events[item.index]
            .narg
            .clone()
            .expect("a message has a range")
    };
    let (level, block) = (range(akita_level), range(aux));
    for (label, position) in [
        ("an Akita masked-level value", level.start),
        ("the auxiliary block's Akita part", block.start),
        ("the auxiliary block's Jolt part", block.end - 1),
    ] {
        let mut tampered = case.proof().clone();
        tampered.narg[position] ^= 1;
        assert!(
            case.verify_proof(&tampered).is_err(),
            "a flipped byte in {label} is accepted"
        );
    }
}

/// HT3: a rerun after an aborted attempt shares no commitment, pad
/// commitment or masked message with it, and its proof verifies.
#[test]
fn whiteout_rerun_reuses_nothing_from_the_aborted_attempt() {
    let (case, aborted) = whiteout_rerun_case();
    let trace = case.honest_trace();
    // The trace commitment's header (layout digest and sizes) is public
    // metadata; its image follows.
    let header = trace
        .messages()
        .into_iter()
        .find(|message| message.region == Region::Commitments)
        .expect("the trace commitment")
        .range;
    let mut compared = 0;
    for item in classify(&trace) {
        let fresh = matches!(
            item.class,
            Some(Class::Commitment | Class::Pads | Class::MaskedSlot)
        );
        let Some(range) = trace.events[item.index].narg.clone() else {
            continue;
        };
        if fresh && range != header && range.end <= aborted.len() && range.len() >= FIELD_BYTES {
            assert_ne!(
                case.proof().narg[range.clone()],
                aborted[range.clone()],
                "{:?} at {range:?} repeats the aborted attempt",
                item.region
            );
            compared += 1;
        }
    }
    assert!(
        compared > 100,
        "the attempts share a long prefix to compare"
    );
}

/// V4 counting equality: the pad layout the verifier derives over zero
/// bytes equals the one over the proof's own bytes and over random bytes,
/// and fills exactly the prover's pads.
#[test]
fn whiteout_pad_layout_counts_the_same_on_any_bytes() {
    let (case, trace) = honest();
    let narg = &case.proof().narg;
    let layout = |prefix: &[u8]| {
        pad_layout::<AkitaField, AkitaScheme, AkitaVc, JoltSponge>(
            &case.preprocessing,
            &case.public_io,
            None,
            prefix,
        )
        .expect("pad layout")
    };
    let seeded = trace
        .messages()
        .into_iter()
        .find(|message| message.region == Region::Pads)
        .expect("the pad commitment")
        .range
        .start;
    let zeros = layout(&narg[..seeded]);
    assert_eq!(zeros.layout().num_pads, case.pads.len());
    assert_eq!(layout(narg), zeros);
    let mut random = narg[..seeded].to_vec();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    random.extend((seeded..narg.len()).map(|_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.to_le_bytes()[0]
    }));
    assert_eq!(layout(&random), zeros);
}

/// V5: the statement and header tampers, and the Whiteout proof declared
/// as a clear or BlindFold proof.
#[test]
fn whiteout_statement_and_mode_tampers() {
    let case = whiteout_muldiv_case();
    checks::statement_tampers(case);
    checks::header_equivocations(case);
    for zk in [ZkConfig::Transparent, ZkConfig::BlindFold] {
        let mut proof = case.proof().clone();
        proof.protocol = JoltProtocolConfig {
            zk,
            ..proof.protocol
        };
        assert!(matches!(
            case.verify_proof(&proof),
            Err(VerifierError::ProtocolConfigMismatch { .. })
        ));
    }
}

#[path = "../support/fs_transcript.rs"]
#[expect(dead_code, reason = "the audit sponge's replay half is unused here")]
mod fs_transcript;

/// Shifts of the final-opening claims, with every auxiliary value
/// recomputed as a consistent cheater would: the honest pads, the relation,
/// the residual witness (bound to the trace commitment), and the
/// EvaluationBatch challenges the run drew.
mod shifts {
    use jolt_akita::{AkitaField, AkitaScheme};
    use jolt_algebra::{Field, One, Zero};
    use jolt_prover::akita::preprocessing::AkitaVc;
    use jolt_transcript::{ClaimDomain, TranscriptOp, VerifierTranscript};
    use jolt_verifier::whiteout::{
        pad_layout, whiteout_prefix, whiteout_relation, whiteout_stages,
    };
    use jolt_verifier::{jolt_protocol_id, JoltSponge, JOLT_SESSION};
    use jolt_zk::masking::{ResidualRelation, ResidualVar, ZkError};
    use jolt_zk::whiteout::MaskedGroupOpening;

    use super::fs_transcript::{decode_small_challenge, record, AuditSponge};
    use akita_params::GrindingSite;

    use super::Site;
    use crate::support::akita_fixtures::whiteout_muldiv_case;

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
    /// meets Akita's rows only through one functional, whose coefficients
    /// are the EvaluationBatch challenges (drawn after t_p and every masked
    /// value). So a shift in that functional's kernel passes every row under
    /// these challenges: the binding is Schwartz–Zippel over the challenges,
    /// not a row the shift could not satisfy.
    #[test]
    fn whiteout_final_openings_meet_akita_through_the_batch_challenges() {
        let case = whiteout_muldiv_case();
        let pads = &case.pads;
        let derived = pad_layout::<F, AkitaScheme, AkitaVc, JoltSponge>(
            &case.preprocessing,
            &case.public_io,
            None,
            &case.proof.narg,
        )
        .expect("pad layout");
        let layout = derived.layout().clone();
        let ((stages, prefix, relation, events), tape) = record(|| {
            let mut transcript = VerifierTranscript::<AuditSponge>::new(
                &jolt_protocol_id::<AuditSponge>(),
                JOLT_SESSION,
                &case.proof.narg,
            );
            let mut stages = whiteout_stages::<F, AkitaScheme, AkitaVc, AuditSponge>(
                &case.preprocessing,
                &case.public_io,
                None,
                &mut transcript,
                Some(&derived),
            )
            .expect("masked stages");
            let (prefix, _) = whiteout_prefix(&mut stages, &mut transcript).expect("prefix");
            let relation =
                whiteout_relation(&mut stages, &prefix, &mut transcript).expect("relation");
            let events = transcript.events().to_vec();
            (stages, prefix, relation, events)
        });

        // The EvaluationBatch challenges, one per claim, from the tape: the
        // batch's draws (detail 0, not its grinding nonce's).
        let evaluation_batch = GrindingSite::EvaluationBatch { level: 0 }.site_id(0);
        let mut offset = 0;
        let mut batch = Vec::new();
        for event in &events {
            if event.op != TranscriptOp::Challenge {
                continue;
            }
            if matches!(
                Site::of(event.site),
                Site::Akita(site) if site.family == evaluation_batch.family && site.detail == 0
            ) {
                batch.push(decode_small_challenge::<F>(
                    &tape.bytes[offset..offset + event.len],
                ));
            }
            offset += event.len;
        }
        assert_eq!(
            offset,
            tape.bytes.len(),
            "every challenge byte is on the tape"
        );
        let claims = &stages.claim.evaluations;
        assert_eq!(
            batch.len(),
            claims.len(),
            "one EvaluationBatch challenge per claim"
        );

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

        // Each claim's form reads one base pad.
        let slots: Vec<usize> = claims
            .iter()
            .map(|form| {
                let pads: Vec<usize> = form
                    .terms
                    .iter()
                    .filter_map(|(variable, _)| match variable {
                        ResidualVar::Pad(pad) => Some(pad.0),
                        ResidualVar::Witness(_) => None,
                    })
                    .collect();
                assert_eq!(pads.len(), 1, "a final-opening claim is one masked slot");
                pads[0]
            })
            .collect();
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
            slots.len() - 1,
            "Jolt's rows see one combination"
        );
        let (_, akita_rank) = null_space(&akita);
        assert_eq!(
            akita_rank, 1,
            "Akita's rows apply one functional to the claims"
        );

        // That functional is the batch challenges, up to one scale.
        let row = (0..akita[0].len())
            .find(|&row| !akita[0][row].is_zero())
            .expect("a claim shift moves an Akita row");
        let scale = akita[0][row] * batch[0].inverse().expect("nonzero challenge");
        for (claim, challenge) in batch.iter().enumerate() {
            assert_eq!(akita[claim][row], scale * *challenge, "claim {claim}");
        }

        // Two Jolt-invisible shifts combined into the functional's kernel
        // pass every row under these challenges.
        let phi = |delta: &[F]| -> F {
            delta
                .iter()
                .zip(&batch)
                .fold(F::zero(), |sum, (by, challenge)| sum + *by * *challenge)
        };
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
    }
}
