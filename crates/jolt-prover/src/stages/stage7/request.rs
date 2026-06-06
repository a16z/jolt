use jolt_backends::SumcheckRaPushforwardRequest;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::Field;
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, OracleRef, OracleViewRequest, WitnessProvider,
};

use super::input::Stage7ProverConfig;
use super::output::{Stage7RegularBatchInputClaims, Stage7RegularBatchPrefixOutput};
use crate::ProverError;

/// Backend sumcheck request for the Stage 7 hamming-weight RA-family claim
/// reduction, assembled from the Fiat-Shamir prefix.
///
/// Carries the per-instance input claim and the batching challenge
/// `hamming_gamma`, plus the committed RA-family polynomial identifiers
/// (instruction, then bytecode, then RAM) and the pushforward parameters
/// (`log_k_chunk`, `log_t`, `r_cycle`) that the prover hands to
/// `SumcheckBackend::materialize_sumcheck_ra_pushforward` to build the
/// per-polynomial `G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)` tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7RegularBatchRequest<F: Field> {
    pub input_claims: Stage7RegularBatchInputClaims<F>,
    pub hamming_gamma: F,
    pub instruction_ids: Vec<JoltCommittedPolynomial>,
    pub bytecode_ids: Vec<JoltCommittedPolynomial>,
    pub ram_ids: Vec<JoltCommittedPolynomial>,
    pub log_k_chunk: usize,
    pub log_t: usize,
    pub r_cycle: Vec<F>,
}

impl<F: Field> Stage7RegularBatchRequest<F> {
    pub fn from_prefix(
        config: &Stage7ProverConfig,
        prefix: &Stage7RegularBatchPrefixOutput<F>,
        r_cycle: Vec<F>,
    ) -> Self {
        let layout = config.hamming_dimensions.layout;
        let instruction_ids = (0..layout.instruction())
            .map(JoltCommittedPolynomial::InstructionRa)
            .collect();
        let bytecode_ids = (0..layout.bytecode())
            .map(JoltCommittedPolynomial::BytecodeRa)
            .collect();
        let ram_ids = (0..layout.ram())
            .map(JoltCommittedPolynomial::RamRa)
            .collect();
        Self {
            input_claims: prefix.input_claims.clone(),
            hamming_gamma: prefix.hamming_gamma,
            instruction_ids,
            bytecode_ids,
            ram_ids,
            log_k_chunk: config.hamming_dimensions.log_k_chunk,
            log_t: config.log_t,
            r_cycle,
        }
    }

    pub fn num_polys(&self) -> usize {
        self.instruction_ids.len() + self.bytecode_ids.len() + self.ram_ids.len()
    }

    pub fn committed_ids(&self) -> Vec<JoltCommittedPolynomial> {
        let mut ids = Vec::with_capacity(self.num_polys());
        ids.extend_from_slice(&self.instruction_ids);
        ids.extend_from_slice(&self.bytecode_ids);
        ids.extend_from_slice(&self.ram_ids);
        ids
    }

    pub fn pushforward_request(&self) -> SumcheckRaPushforwardRequest<F, JoltVmNamespace> {
        SumcheckRaPushforwardRequest::new(
            "stage7.hamming.pushforward",
            self.instruction_ids.clone(),
            self.bytecode_ids.clone(),
            self.ram_ids.clone(),
            self.log_k_chunk,
            self.r_cycle.clone(),
            1usize << self.log_t,
        )
    }
}

/// Evaluate the reduced RA-family output openings at the verifier-derived hamming
/// opening point via the generic witness oracle views (no materialization).
///
/// This is the Stage 7 output-opening request: for each committed RA polynomial
/// it builds the primary `OracleViewRequest` and evaluates it, yielding the
/// reduced opening claim that Stage 8 batches into the final PCS opening.
pub fn evaluate_committed_openings<F, W>(
    witness: &W,
    polynomials: &[JoltCommittedPolynomial],
    point: &[F],
) -> Result<Vec<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    polynomials
        .iter()
        .map(|polynomial| evaluate_committed_opening(witness, *polynomial, point))
        .collect()
}

fn evaluate_committed_opening<F, W>(
    witness: &W,
    polynomial: JoltCommittedPolynomial,
    point: &[F],
) -> Result<F, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let oracle = OracleRef::committed(polynomial);
    let requirement = witness
        .view_requirements(oracle)?
        .into_iter()
        .next()
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("Stage 7 RA oracle {polynomial:?} exposes no view requirement"),
        })?;
    let request = OracleViewRequest::new(requirement);
    witness
        .try_evaluate_oracle_view(request, point)?
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 7 RA opening for {polynomial:?} could not be evaluated without materialization"
            ),
        })
}
