//! Argument-string tamper harness: traced verification, protocol regions, and
//! the per-message rejection deadlines the soundness sweeps assert.
//!
//! A proof is its NARG, and every prover message is a byte range of it. Under
//! the `logging` feature the verifier transcript records each operation with
//! its protocol site, so a run over a tampered NARG reports how far
//! verification got before it rejected.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "tamper harness helpers fail loudly when a fixture or trace breaks an assumption"
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::ops::Range;

use common::jolt_device::JoltDevice;
use jolt_crypto::{Commitment, VectorCommitment};
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_transcript::{
    Channel, ProverTranscript, SiteId, TranscriptError, TranscriptEvent, TranscriptOp,
    VerifierTranscript,
};
use jolt_verifier::sites::{
    BLINDFOLD, COMMITMENTS, PREAMBLE, STAGE1, STAGE2, STAGE3, STAGE4, STAGE5, STAGE6A, STAGE6B,
    STAGE7, STAGE8,
};
use jolt_verifier::{
    jolt_protocol_id, JoltProof, JoltSponge, JoltVerifierPreprocessing, ProofHeader, VerifierError,
    JOLT_SESSION,
};

/// A Jolt protocol region, in transcript order. Each is entered by the
/// verifier setting the matching [`jolt_verifier::sites`] site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Region {
    Preamble,
    Commitments,
    Stage1,
    Stage2,
    Stage3,
    Stage4,
    Stage5,
    Stage6a,
    Stage6b,
    Stage7,
    Stage8,
    BlindFold,
}

impl Region {
    pub const ALL: [Self; 12] = [
        Self::Preamble,
        Self::Commitments,
        Self::Stage1,
        Self::Stage2,
        Self::Stage3,
        Self::Stage4,
        Self::Stage5,
        Self::Stage6a,
        Self::Stage6b,
        Self::Stage7,
        Self::Stage8,
        Self::BlindFold,
    ];

    /// The regions a proof of this build passes through.
    pub fn expected() -> Vec<Self> {
        Self::ALL
            .into_iter()
            .filter(|region| cfg!(feature = "zk") || *region != Self::BlindFold)
            .collect()
    }

    fn from_site(site: SiteId) -> Option<Self> {
        [
            (PREAMBLE, Self::Preamble),
            (COMMITMENTS, Self::Commitments),
            (STAGE1, Self::Stage1),
            (STAGE2, Self::Stage2),
            (STAGE3, Self::Stage3),
            (STAGE4, Self::Stage4),
            (STAGE5, Self::Stage5),
            (STAGE6A, Self::Stage6a),
            (STAGE6B, Self::Stage6b),
            (STAGE7, Self::Stage7),
            (STAGE8, Self::Stage8),
            (BLINDFOLD, Self::BlindFold),
        ]
        .into_iter()
        .find_map(|(candidate, region)| (candidate == site).then_some(region))
    }

    /// Whether the region's prover messages are all clear field elements
    /// (sumcheck coefficients and opening claims).
    pub fn is_clear_sumcheck_stage(self) -> bool {
        !cfg!(feature = "zk") && self >= Self::Stage1 && self <= Self::Stage7
    }

    /// The latest region by which the verifier must reject an altered public
    /// statement (preamble values or public commitments), which only moves
    /// the challenges: Stage 1 in the clear, Stage 8 under ZK, as for the
    /// header and commitments in [`Trace::messages`].
    pub fn statement_deadline() -> Self {
        if cfg!(feature = "zk") {
            Self::Stage8
        } else {
            Self::Stage1
        }
    }
}

/// One prover message of a traced run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub region: Region,
    pub range: Range<usize>,
    /// The latest region by which an alteration of this message must be
    /// rejected.
    pub deadline: Region,
}

/// The outcome of one verification over an argument string, with every
/// transcript operation it completed.
#[derive(Debug)]
pub struct Trace {
    pub events: Vec<TranscriptEvent>,
    pub result: Result<(), VerifierError>,
}

impl Trace {
    /// The region of each event. A nested protocol (the PCS) may tag its own
    /// sites; those operations belong to the Jolt region that called it.
    pub fn regions(&self) -> Vec<Region> {
        let mut current = Region::Preamble;
        self.events
            .iter()
            .map(|event| {
                if let Some(region) = Region::from_site(event.site) {
                    current = region;
                }
                current
            })
            .collect()
    }

    /// The run's prover messages with their rejection deadlines.
    ///
    /// Every message is absorbed, so altering one changes its value and
    /// re-randomizes every later challenge. The deadline is the first region
    /// that checks either:
    ///
    /// - Clear builds: the region of the first challenge drawn after the
    ///   message (its own region if none follows). Every region that draws a
    ///   challenge ends in a check that depends on it: each stage's batched
    ///   sumcheck ends in its output check (compressed rounds recover each
    ///   round's omitted coefficient from the running claim, so the rounds are
    ///   checked through that final evaluation), uni-skip checks its first
    ///   round's sum explicitly, and Stage 8's opening checks the joint claim.
    ///   So round polynomials are caught in their own stage, the header and
    ///   commitments by Stage 1 (or in place, by `validate_inputs` or the
    ///   commitment decoder), and a stage's final output claims, sent after
    ///   its last challenge, by the next region. That last case is real:
    ///   some Stage 1 outer openings are consumed only as Stage 2's
    ///   product-virtualization inputs, some Stage 2 claims only by Stage 3's
    ///   output check, and Stage 6a's address-phase claims only by Stage 6b's.
    /// - ZK builds: Stage 8 for the header, commitments, and Stages 1–7, whose
    ///   messages are hiding commitments with no clear check; the Stage 8
    ///   Dory evaluation proof is bound to the transcript and to the opening
    ///   point every stage challenge feeds. Stage 8 and BlindFold messages by
    ///   their own region, which verifies them before returning.
    pub fn messages(&self) -> Vec<Message> {
        let regions = self.regions();
        let mut next_challenge = None;
        let mut deadlines = vec![None; self.events.len()];
        for (index, event) in self.events.iter().enumerate().rev() {
            deadlines[index] = next_challenge;
            if event.op == TranscriptOp::Challenge {
                next_challenge = Some(regions[index]);
            }
        }
        self.events
            .iter()
            .zip(regions)
            .zip(deadlines)
            .filter(|((event, _), _)| event.op == TranscriptOp::Message)
            .map(|((event, region), next_challenge)| Message {
                region,
                range: event
                    .narg
                    .clone()
                    .expect("every message event records its argument-string range"),
                deadline: if cfg!(feature = "zk") {
                    region.max(Region::Stage8)
                } else {
                    next_challenge.unwrap_or(region).max(region)
                },
            })
            .collect()
    }

    /// Where verification stopped: the region of the last completed
    /// operation (the preamble if none completed). A rejection between two
    /// regions, before the later one performs any operation, is attributed
    /// to the earlier one.
    pub fn stop_region(&self) -> Region {
        self.regions().last().copied().unwrap_or(Region::Preamble)
    }
}

/// The polynomial commitment type of a fixture's scheme.
pub type CommitmentOf<C> = <<C as TracedCase>::Pcs as Commitment>::Output;

/// A fixture the harness can verify over an arbitrary statement and argument
/// string.
pub trait TracedCase {
    type Pcs: CommitmentScheme;
    type Vc: VectorCommitment<Field = <Self::Pcs as CommitmentScheme>::Field>;

    /// Width of one field element message.
    const FIELD_BYTES: usize;

    fn preprocessing(&self) -> &JoltVerifierPreprocessing<Self::Pcs, Self::Vc>;

    fn public_io(&self) -> &JoltDevice;

    fn proof(&self) -> &JoltProof;

    fn trusted_advice_commitment(&self) -> Option<&<Self::Pcs as Commitment>::Output>;

    /// The public `jolt_verifier::verify` entry point.
    fn verify_statement(
        preprocessing: &JoltVerifierPreprocessing<Self::Pcs, Self::Vc>,
        public_io: &JoltDevice,
        proof: &JoltProof,
        trusted_advice_commitment: Option<&<Self::Pcs as Commitment>::Output>,
    ) -> Result<(), VerifierError>;

    /// Verifies `narg` exactly as `jolt_verifier::verify` does after its
    /// protocol-config check, keeping the transcript's event log.
    fn trace_statement(
        preprocessing: &JoltVerifierPreprocessing<Self::Pcs, Self::Vc>,
        public_io: &JoltDevice,
        trusted_advice_commitment: Option<&<Self::Pcs as Commitment>::Output>,
        narg: &[u8],
    ) -> Trace;

    fn trace(&self, narg: &[u8]) -> Trace {
        Self::trace_statement(
            self.preprocessing(),
            self.public_io(),
            self.trusted_advice_commitment(),
            narg,
        )
    }

    fn verify_proof(&self, proof: &JoltProof) -> Result<(), VerifierError> {
        Self::verify_statement(
            self.preprocessing(),
            self.public_io(),
            proof,
            self.trusted_advice_commitment(),
        )
    }

    /// The honest proof against `preprocessing` in place of the fixture's.
    fn verify_statement_with(
        &self,
        preprocessing: &JoltVerifierPreprocessing<Self::Pcs, Self::Vc>,
    ) -> Result<(), VerifierError> {
        Self::verify_statement(
            preprocessing,
            self.public_io(),
            self.proof(),
            self.trusted_advice_commitment(),
        )
    }

    /// The honest proof's trace, checked to accept through both the traced
    /// replay and the public entry point.
    fn honest_trace(&self) -> Trace {
        self.verify_proof(self.proof())
            .expect("honest fixture verifies through jolt_verifier::verify");
        let trace = self.trace(&self.proof().narg);
        if let Err(error) = &trace.result {
            panic!("honest fixture rejected by the traced replay: {error:?}");
        }
        trace
    }

    fn with_narg(&self, narg: Vec<u8>) -> JoltProof {
        JoltProof {
            protocol: self.proof().protocol,
            narg,
        }
    }
}

/// Ends a traced run: records the events, then applies the exact-consumption
/// check `verify` applies.
fn finish_trace(
    transcript: VerifierTranscript<'_, JoltSponge>,
    result: Result<(), VerifierError>,
) -> Trace {
    let events = transcript.events().to_vec();
    let result = result.and_then(|()| transcript.finish().map_err(VerifierError::from));
    Trace { events, result }
}

fn new_transcript(narg: &[u8]) -> VerifierTranscript<'_, JoltSponge> {
    VerifierTranscript::new(&jolt_protocol_id::<JoltSponge>(), JOLT_SESSION, narg)
}

#[cfg(not(feature = "akita"))]
mod dory {
    use common::jolt_device::JoltDevice;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::{DoryCommitment, DoryScheme};
    use jolt_field::{CanonicalBytes, Fr};
    use jolt_transcript::Channel;
    use jolt_verifier::sites::BLINDFOLD;
    use jolt_verifier::{
        verify, verify_stages, JoltProof, JoltSponge, JoltVerifierPreprocessing, VerifiedStages,
        VerifierError,
    };

    use super::{finish_trace, new_transcript, Trace, TracedCase};

    type Preprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;

    fn trace(
        preprocessing: &Preprocessing,
        public_io: &JoltDevice,
        trusted_advice_commitment: Option<&DoryCommitment>,
        narg: &[u8],
    ) -> Trace {
        let mut transcript = new_transcript(narg);
        let result = verify_stages::<Fr, DoryScheme, Pedersen<Bn254G1>, _>(
            preprocessing,
            public_io,
            trusted_advice_commitment,
            &mut transcript,
        )
        .and_then(|stages| match stages {
            VerifiedStages::Clear => Ok(()),
            VerifiedStages::Zk(protocol) => {
                transcript.site(BLINDFOLD);
                let vc_setup = preprocessing
                    .vc_setup
                    .as_ref()
                    .ok_or(VerifierError::MissingVectorCommitmentSetup)?;
                protocol
                    .verify::<Pedersen<Bn254G1>, _>(vc_setup, &mut transcript)
                    .map_err(|error| VerifierError::BlindFoldVerificationFailed {
                        reason: error.to_string(),
                    })
            }
        });
        finish_trace(transcript, result)
    }

    macro_rules! impl_traced_case {
        ($case:ty) => {
            impl TracedCase for $case {
                type Pcs = DoryScheme;
                type Vc = Pedersen<Bn254G1>;
                const FIELD_BYTES: usize = <Fr as CanonicalBytes>::NUM_BYTES;

                fn preprocessing(&self) -> &Preprocessing {
                    &self.preprocessing
                }

                fn public_io(&self) -> &JoltDevice {
                    &self.public_io
                }

                fn proof(&self) -> &JoltProof {
                    &self.proof
                }

                fn trusted_advice_commitment(&self) -> Option<&DoryCommitment> {
                    self.trusted_advice_commitment.as_ref()
                }

                fn verify_statement(
                    preprocessing: &Preprocessing,
                    public_io: &JoltDevice,
                    proof: &JoltProof,
                    trusted_advice_commitment: Option<&DoryCommitment>,
                ) -> Result<(), VerifierError> {
                    verify::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
                        preprocessing,
                        public_io,
                        proof,
                        trusted_advice_commitment,
                    )
                }

                fn trace_statement(
                    preprocessing: &Preprocessing,
                    public_io: &JoltDevice,
                    trusted_advice_commitment: Option<&DoryCommitment>,
                    narg: &[u8],
                ) -> Trace {
                    trace(preprocessing, public_io, trusted_advice_commitment, narg)
                }
            }
        };
    }

    #[cfg(not(feature = "zk"))]
    impl_traced_case!(crate::support::verifier_fixtures::VerifierFixtureCase);
    #[cfg(feature = "zk")]
    impl_traced_case!(crate::support::verifier_fixtures::ZkVerifierFixtureCase);
}

#[cfg(feature = "akita")]
mod akita {
    use common::jolt_device::JoltDevice;
    use jolt_akita::{AkitaCommitment, AkitaField, AkitaScheme};
    use jolt_crypto::Commitment;
    use jolt_field::CanonicalBytes;
    use jolt_prover::akita::preprocessing::AkitaVc;
    use jolt_verifier::stages::{
        stage1, stage2, stage3, stage4, stage5, stage6a, stage6b, stage7, stage8,
    };
    use jolt_verifier::verifier::SeededTranscript;
    use jolt_verifier::{
        seed_transcript, verify, JoltProof, JoltSponge, JoltVerifierPreprocessing, VerifierError,
    };

    use super::{finish_trace, new_transcript, Trace, TracedCase};
    use crate::support::akita_fixtures::AkitaFixtureCase;

    type Preprocessing = JoltVerifierPreprocessing<AkitaScheme, AkitaVc>;

    /// `jolt_verifier::verify`'s Akita body after its protocol-config check.
    /// The Akita build exports no stage-spine entry point, so the sequence is
    /// restated here; `TracedCase::honest_trace` checks it against `verify`.
    fn stages(
        preprocessing: &Preprocessing,
        public_io: &JoltDevice,
        trusted_advice_commitment: Option<&AkitaCommitment>,
        transcript: &mut jolt_transcript::VerifierTranscript<'_, JoltSponge>,
    ) -> Result<(), VerifierError> {
        let SeededTranscript {
            checked,
            commitments,
            formula_dimensions,
        } = seed_transcript(
            preprocessing,
            public_io,
            trusted_advice_commitment,
            transcript,
        )?;
        let dims = &formula_dimensions;
        let s1 =
            stage1::verify::<AkitaField, <AkitaVc as Commitment>::Output, _>(&checked, transcript)?;
        let s2 = stage2::verify(&checked, transcript, &s1)?;
        let s3 = stage3::verify(&checked, transcript, &s1, &s2)?;
        let s4 = stage4::verify(&checked, preprocessing, transcript, &s2, &s3)?;
        let s5 = stage5::verify(&checked, dims, transcript, &s2, &s4)?;
        let s6a = stage6a::verify(
            &checked,
            preprocessing,
            dims,
            transcript,
            &s1,
            &s2,
            &s3,
            &s4,
            &s5,
        )?;
        let s6b = stage6b::verify(
            &checked,
            preprocessing,
            dims,
            transcript,
            &s1,
            &s2,
            &s3,
            &s4,
            &s5,
            &s6a,
        )?;
        let s7 = stage7::verify(&checked, dims, transcript, &s4, &s6b)?;
        let s8 = stage8::verify::<AkitaField, AkitaScheme, AkitaVc, _>(
            &checked,
            preprocessing,
            &commitments,
            dims,
            trusted_advice_commitment,
            transcript,
            &s4,
            &s6b,
            &s7,
        )?;
        match s8 {
            stage8::Stage8Output::Clear => Ok(()),
            stage8::Stage8Output::Zk(_) => {
                Err(VerifierError::ExpectedClearProof { field: "stage8" })
            }
        }
    }

    impl TracedCase for AkitaFixtureCase {
        type Pcs = AkitaScheme;
        type Vc = AkitaVc;
        const FIELD_BYTES: usize = <AkitaField as CanonicalBytes>::NUM_BYTES;

        fn preprocessing(&self) -> &Preprocessing {
            &self.preprocessing
        }

        fn public_io(&self) -> &JoltDevice {
            &self.public_io
        }

        fn proof(&self) -> &JoltProof {
            &self.proof
        }

        fn trusted_advice_commitment(&self) -> Option<&AkitaCommitment> {
            self.trusted_advice_commitment.as_ref()
        }

        fn verify_statement(
            preprocessing: &Preprocessing,
            public_io: &JoltDevice,
            proof: &JoltProof,
            trusted_advice_commitment: Option<&AkitaCommitment>,
        ) -> Result<(), VerifierError> {
            verify::<AkitaField, AkitaScheme, AkitaVc, JoltSponge>(
                preprocessing,
                public_io,
                proof,
                trusted_advice_commitment,
            )
        }

        fn trace_statement(
            preprocessing: &Preprocessing,
            public_io: &JoltDevice,
            trusted_advice_commitment: Option<&AkitaCommitment>,
            narg: &[u8],
        ) -> Trace {
            let mut transcript = new_transcript(narg);
            let result = stages(
                preprocessing,
                public_io,
                trusted_advice_commitment,
                &mut transcript,
            );
            finish_trace(transcript, result)
        }
    }
}

/// Asserts the honest message events tile the argument string: every byte
/// belongs to exactly one message, in order, and only message events carry a
/// range. Also asserts every region of this build carries at least one
/// message and no foreign Jolt region appears.
pub fn assert_coverage_closure(trace: &Trace, narg_len: usize) {
    let mut cursor = 0;
    for event in &trace.events {
        match (event.op, &event.narg) {
            (TranscriptOp::Message, Some(range)) => {
                assert_eq!(
                    range.start, cursor,
                    "message at {range:?} leaves bytes {cursor}..{} unaccounted",
                    range.start
                );
                assert_eq!(
                    range.len(),
                    event.len,
                    "message length disagrees with its range"
                );
                cursor = range.end;
            }
            (TranscriptOp::Message, None) => panic!("message event without a range: {event:?}"),
            (TranscriptOp::Public | TranscriptOp::Challenge, None) => {}
            (TranscriptOp::Public | TranscriptOp::Challenge, Some(_)) => {
                panic!("non-message event carries an argument-string range: {event:?}")
            }
        }
    }
    assert_eq!(
        cursor, narg_len,
        "argument-string bytes after the last message"
    );

    let present: Vec<Region> = {
        let mut regions: Vec<_> = trace.messages().into_iter().map(|m| m.region).collect();
        regions.dedup();
        regions
    };
    assert_eq!(
        present,
        Region::expected(),
        "message regions, in transcript order, differ from this build's protocol regions"
    );
}

/// One single-byte alteration of the argument string.
#[derive(Clone, Debug)]
pub struct Tamper {
    pub region: Region,
    pub deadline: Region,
    pub message: Range<usize>,
    pub offset: usize,
}

/// The tampers of `message`: the low byte of every field element in a clear
/// sumcheck stage (each flip is a well-formed claim or coefficient off by
/// one), else the first, middle, and last byte. An empty message (a
/// zero-count vector) has none.
pub fn tampers_of<C: TracedCase>(message: &Message) -> Vec<Tamper> {
    let len = message.range.len();
    let offsets: Vec<usize> = if len == 0 {
        Vec::new()
    } else if message.region.is_clear_sumcheck_stage() {
        assert_eq!(
            len % C::FIELD_BYTES,
            0,
            "clear stage message {message:?} is not a whole number of field elements"
        );
        (0..len).step_by(C::FIELD_BYTES).collect()
    } else {
        let mut offsets = vec![0, len / 2, len - 1];
        offsets.dedup();
        offsets
    };
    offsets
        .into_iter()
        .map(|offset| Tamper {
            region: message.region,
            deadline: message.deadline,
            message: message.range.clone(),
            offset: message.range.start + offset,
        })
        .collect()
}

/// Every tamper of every message, or a deterministic stride sample of at
/// most `budget` of them that always keeps each region's first and last
/// tamper.
pub fn select_tampers<C: TracedCase>(messages: &[Message], budget: Option<usize>) -> Vec<Tamper> {
    let all: Vec<Tamper> = messages.iter().flat_map(tampers_of::<C>).collect();
    let Some(budget) = budget.filter(|&budget| budget < all.len()) else {
        return all;
    };
    let stride = all.len().div_ceil(budget);
    all.iter()
        .enumerate()
        .filter(|(index, tamper)| {
            index % stride == 0
                || index
                    .checked_sub(1)
                    .and_then(|previous| all.get(previous))
                    .map(|t| t.region)
                    != Some(tamper.region)
                || all.get(index + 1).map(|t| t.region) != Some(tamper.region)
        })
        .map(|(_, tamper)| tamper.clone())
        .collect()
}

/// Where each tamper of a sweep was rejected, keyed by (message region,
/// rejection region).
#[derive(Debug, Default)]
pub struct SweepReport {
    pub tampers: usize,
    pub messages: usize,
    pub rejections: BTreeMap<(Region, Region), usize>,
}

impl SweepReport {
    pub fn summary(&self) -> String {
        let mut out = format!("{} tampers over {} messages\n", self.tampers, self.messages);
        for ((message, rejected), count) in &self.rejections {
            let _ = writeln!(out, "  {message:?} -> {rejected:?}: {count}");
        }
        out
    }
}

/// Flips the low bit of each selected byte, one tamper at a time, and
/// requires every tampered proof to reject no later than its message's
/// deadline ([`Trace::messages`]).
pub fn sweep<C: TracedCase>(case: &C, budget: Option<usize>) -> SweepReport {
    let honest = case.honest_trace();
    let messages = honest.messages();
    let tampers = select_tampers::<C>(&messages, budget);
    let mut report = SweepReport {
        tampers: tampers.len(),
        messages: messages.len(),
        ..SweepReport::default()
    };
    let mut failures = Vec::new();
    let mut narg = case.proof().narg.clone();
    for tamper in &tampers {
        narg[tamper.offset] ^= 1;
        let trace = case.trace(&narg);
        narg[tamper.offset] ^= 1;
        let stopped = trace.stop_region();
        *report
            .rejections
            .entry((tamper.region, stopped))
            .or_default() += 1;
        if trace.result.is_ok() {
            failures.push(format!("accepted: {tamper:?}"));
        } else if stopped > tamper.deadline {
            failures.push(format!(
                "rejected in {stopped:?} after its deadline: {tamper:?}: {:?}",
                trace.result
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} tampers broke their deadline:\n{}\n{}",
        failures.len(),
        tampers.len(),
        report.summary(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    report
}

/// The argument string cut at each selected message boundary must fail
/// with a truncation error: typed, or carried in the reason of the wrapping
/// error (the sumcheck verifier, the commitment decoder, the opening
/// verifier, and BlindFold report transcript failures as strings).
///
/// WARNING: `jolt-akita`'s native opening maps every Akita verifier error,
/// truncation included, to the bare `OpeningsError::VerificationFailed`, so
/// on the Akita build a cut inside Stage 8 can only be required to fail that
/// opening.
pub fn assert_truncations_reject<C: TracedCase>(case: &C, budget: Option<usize>) {
    let honest = case.honest_trace();
    let narg = &case.proof().narg;
    let boundaries: Vec<(usize, Region)> = honest
        .messages()
        .iter()
        .map(|message| (message.range.start, message.region))
        .collect();
    let stride = budget.map_or(1, |budget| boundaries.len().div_ceil(budget).max(1));
    let mut failures = Vec::new();
    for (index, &(boundary, region)) in boundaries.iter().enumerate() {
        if index % stride != 0 && index + 1 != boundaries.len() {
            continue;
        }
        let result = case.trace(&narg[..boundary]).result;
        let opaque_akita_opening = cfg!(feature = "akita")
            && region == Region::Stage8
            && matches!(
                &result,
                Err(VerifierError::FinalOpeningVerificationFailed { reason })
                    if *reason == OpeningsError::VerificationFailed.to_string()
            );
        if !is_truncation(&result) && !opaque_akita_opening {
            failures.push(format!("cut at {boundary}: {result:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "truncations not rejected as truncated:\n{}",
        failures.join("\n")
    );
}

fn is_truncation(result: &Result<(), VerifierError>) -> bool {
    let truncated = TranscriptError::Truncated.to_string();
    match result {
        Err(VerifierError::Transcript(TranscriptError::Truncated)) => true,
        Err(
            VerifierError::MalformedCommitment { reason }
            | VerifierError::FinalOpeningVerificationFailed { reason }
            | VerifierError::FinalOpeningBatchFailed { reason }
            | VerifierError::BlindFoldVerificationFailed { reason }
            | VerifierError::StageClaimSumcheckFailed { reason, .. },
        ) => reason.contains(&truncated),
        _ => false,
    }
}

/// The proof header's encoding, produced by the production encoder.
pub fn encode_header(header: &ProofHeader) -> Vec<u8> {
    let mut transcript =
        ProverTranscript::<JoltSponge>::new(&jolt_protocol_id::<JoltSponge>(), JOLT_SESSION);
    transcript.site(PREAMBLE);
    header.send(&mut transcript);
    transcript.finish()
}

/// The proof's header, decoded by the production decoder.
pub fn decode_header(narg: &[u8]) -> ProofHeader {
    ProofHeader::receive(&mut new_transcript(narg)).expect("honest proof header decodes")
}

/// `narg` with its header replaced by `header`'s encoding.
pub fn with_header(narg: &[u8], header: &ProofHeader) -> Vec<u8> {
    let honest_len = encode_header(&decode_header(narg)).len();
    let mut out = encode_header(header);
    out.extend_from_slice(&narg[honest_len..]);
    out
}

/// The `index`-th message of `region` in the honest trace.
pub fn message_in(trace: &Trace, region: Region, index: usize) -> Range<usize> {
    trace
        .messages()
        .into_iter()
        .filter(|message| message.region == region)
        .nth(index)
        .unwrap_or_else(|| panic!("no message {index} in {region:?}"))
        .range
}
