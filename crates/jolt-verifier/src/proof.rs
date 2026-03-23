//! Proof types for the Jolt proving pipeline.

use crate::config::ProverConfig;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_spartan::UniformSpartanProof;
use jolt_sumcheck::SumcheckProof;
use serde::{Deserialize, Serialize};

/// Per-stage proof: sumcheck round polynomials + polynomial evaluations.
///
/// `round_polys` are verified by the sumcheck verifier. `evals` are the
/// polynomial evaluations at the sumcheck challenge point, used to check
/// the output formula and to chain input claims to downstream stages.
/// Committed polynomial evaluations are ultimately verified by PCS.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct StageProof<F: Field> {
    pub round_polys: SumcheckProof<F>,
    pub evals: Vec<F>,
}

/// Complete Jolt proof for one program execution.
///
/// Self-contained: [`config`](Self::config) carries all parameters the
/// verifier needs. The verifier re-derives all Fiat-Shamir challenges
/// from the transcript — no challenges are stored in the proof.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JoltProof<F: Field, PCS: CommitmentScheme<Field = F>> {
    pub config: ProverConfig,
    pub spartan_proof: UniformSpartanProof<F>,
    pub stage_proofs: Vec<StageProof<F>>,
    pub opening_proofs: Vec<PCS::Proof>,
    pub witness_commitment: PCS::Output,
    pub commitments: Vec<PCS::Output>,
}
