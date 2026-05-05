//! Canonical comparison artifacts for Jolt-on-Bolt equivalence gates.
//!
//! These types are intentionally representation-only. They give core and
//! generated Bolt adapters a common shape for public artifacts, but they do not
//! encode Jolt stage semantics, point normalization rules, witness
//! materialization, or claim-reduction formulas.

use crate::TranscriptEvent;
use jolt_profiling::PerfMetrics;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactSource {
    Core,
    Bolt,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EquivalenceRun<F> {
    pub source: ArtifactSource,
    pub commitments: CommitmentTrace,
    pub transcript: TranscriptTrace,
    pub stages: Vec<StageArtifacts<F>>,
    pub opening_claims: OpeningClaims<F>,
    pub verifier_result: VerifierResult,
    pub perf: Option<PerfMetrics>,
}

impl<F> EquivalenceRun<F> {
    pub fn new(source: ArtifactSource) -> Self {
        Self {
            source,
            commitments: CommitmentTrace::default(),
            transcript: TranscriptTrace::default(),
            stages: Vec::new(),
            opening_claims: OpeningClaims::default(),
            verifier_result: VerifierResult::not_run(),
            perf: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommitmentTrace {
    pub commitments: Vec<CommitmentArtifact>,
}

/// One public commitment slot.
///
/// `bytes = None` represents an intentionally skipped optional commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentArtifact {
    pub label: String,
    pub artifact: String,
    pub bytes: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TranscriptTrace {
    pub events: Vec<TranscriptEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageArtifacts<F> {
    pub stage: String,
    pub sumchecks: Vec<SumcheckArtifacts<F>>,
    pub opening_batches: Vec<OpeningBatchArtifacts>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckArtifacts<F> {
    pub driver: String,
    pub point: Vec<F>,
    pub round_polynomials: Vec<Vec<F>>,
    pub evals: Vec<NamedScalar<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedScalar<F> {
    pub name: String,
    pub value: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningClaims<F> {
    pub claims: Vec<OpeningClaim<F>>,
}

impl<F> Default for OpeningClaims<F> {
    fn default() -> Self {
        Self { claims: Vec::new() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningClaim<F> {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub kind: OpeningClaimKind,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpeningClaimKind {
    Committed,
    Virtual,
    Advice,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningBatchArtifacts {
    pub symbol: String,
    pub ordered_claims: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierResult {
    pub accepted: bool,
    pub error: Option<String>,
}

impl VerifierResult {
    pub fn accepted() -> Self {
        Self {
            accepted: true,
            error: None,
        }
    }

    pub fn rejected(error: impl Into<String>) -> Self {
        Self {
            accepted: false,
            error: Some(error.into()),
        }
    }

    pub fn not_run() -> Self {
        Self {
            accepted: false,
            error: Some("verifier not run".to_owned()),
        }
    }
}
