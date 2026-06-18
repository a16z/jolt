#[cfg(test)]
use jolt_backends::SumcheckMaterializationOutput;
use jolt_backends::{
    spartan_outer_row, stage1_r1cs_input_slot, BackendValueSlot, SumcheckBackend,
    SumcheckLinearProductOutput, SumcheckPrefixProductSumQuery, SumcheckPrefixProductSumRequest,
    SumcheckSpartanOuterRemainderQuery, SumcheckSpartanOuterRemainderRequest,
    SumcheckSpartanOuterRemainderRound, SumcheckSpartanOuterRemainderRowStateRequest,
    SumcheckSpartanOuterRemainderState, SumcheckSpartanOuterRemainderStateRequest,
    SumcheckSpartanOuterRow, SumcheckSpartanOuterUniskipQuery, SumcheckSpartanOuterUniskipRequest,
    SPARTAN_OUTER_REMAINDER_RELATION, SPARTAN_OUTER_UNISKIP_RELATION,
    STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS,
};
#[cfg(test)]
use jolt_backends::{SumcheckMaterializationRequest, SumcheckViewMaterializationRequest};
use jolt_claims::protocols::jolt::{
    formulas::spartan::{SpartanOuterDimensions, SPARTAN_OUTER_R1CS_INPUTS},
    JoltVirtualPolynomial,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::{
    lagrange::{
        centered_domain_start, centered_lagrange_evals, centered_lagrange_kernel,
        interpolate_to_coeffs, poly_mul,
    },
    TensorEqTable, UnivariatePoly,
};
use jolt_r1cs::{
    constraints::{
        jolt::{
            spartan_outer_constraints, spartan_outer_opening_columns, spartan_outer_row_weights,
            JoltSpartanOuterRemainder, JoltSpartanOuterRemainderChallenges,
            SPARTAN_OUTER_REMAINDER_DEGREE, SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
        },
        rv64,
    },
    ConstraintMatrices,
};
use jolt_riscv::CircuitFlags;
use jolt_sumcheck::{
    append_sumcheck_claim, ClearProof, ClearSumcheckProof, CompressedLabeledRoundPoly,
    CompressedSumcheckProof, LabeledRoundPoly, RoundMessage, SumcheckProof,
};
use jolt_transcript::Transcript;
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage1::Stage1PublicOutput;
use jolt_verifier::stages::stage1::{stage1_clear_output, Stage1ClearOutput};
use jolt_witness::protocols::jolt_vm::JoltVmSpartanOuterRows;
use jolt_witness::protocols::jolt_vm::{JoltVmNamespace, JoltVmSpartanOuterRow};
use jolt_witness::WitnessProvider;
#[cfg(test)]
use jolt_witness::{OracleRef, ViewRequirement};

#[cfg(feature = "zk")]
use crate::committed::{CommittedSumcheckBuilder, CommittedSumcheckWitness};
use crate::stages::invalid_sumcheck_output;
#[cfg(test)]
use crate::stages::primary_view_requirement;
use crate::ProverError;

const STAGE1_RV64_R1CS_INPUT_COUNT: usize = SPARTAN_OUTER_R1CS_INPUTS.len();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1ProverConfig {
    pub log_t: usize,
    #[cfg(feature = "zk")]
    pub committed_rounds: bool,
}

impl Stage1ProverConfig {
    pub const fn new(log_t: usize) -> Self {
        Self {
            log_t,
            #[cfg(feature = "zk")]
            committed_rounds: false,
        }
    }

    pub fn dimensions(self) -> SpartanOuterDimensions {
        SpartanOuterDimensions::rv64(self.log_t)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage1ProverInput<'a, W> {
    pub config: Stage1ProverConfig,
    pub witness: &'a W,
}

impl<'a, W> Stage1ProverInput<'a, W> {
    pub const fn new(config: Stage1ProverConfig, witness: &'a W) -> Self {
        Self { config, witness }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1ProofComponent<F: Field, Proof> {
    pub uniskip_proof: Proof,
    pub remainder_proof: Proof,
    pub uniskip_output_claim: F,
    pub remainder_output_claim: F,
    pub r1cs_input_claims: Vec<Stage1R1csInputClaim<F>>,
    /// Verifier-mirroring Stage 1 output that downstream stages consume as a
    /// dependency, produced by the clear prover (`prove`). The pure-backend test
    /// path (`from_backend_result`) leaves it `None`; the full-proof
    /// orchestrator requires it.
    pub verifier_output: Option<Stage1ClearOutput<F>>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1CommittedProofComponent<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub uniskip_proof: SumcheckProof<F, VC::Output>,
    pub remainder_proof: SumcheckProof<F, VC::Output>,
    pub public: Stage1PublicOutput<F>,
    pub verifier_output: Stage1ClearOutput<F>,
    pub uniskip_output_claim_values: Vec<F>,
    pub remainder_output_claim_values: Vec<F>,
    pub(crate) uniskip_committed_witness: CommittedSumcheckWitness<F>,
    pub(crate) remainder_committed_witness: CommittedSumcheckWitness<F>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1R1csInputClaim<F: Field> {
    pub variable: JoltVirtualPolynomial,
    pub slot: BackendValueSlot,
    pub value: F,
}

impl<F: Field> Stage1R1csInputClaim<F> {
    pub(crate) fn verifier_input(&self) -> (JoltVirtualPolynomial, F) {
        (self.variable, self.value)
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage1R1csMaterializationRequest {
    materializations: SumcheckMaterializationRequest<JoltVmNamespace>,
    r1cs_inputs: Vec<Stage1R1csInputRequest>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stage1R1csInputRequest {
    variable: JoltVirtualPolynomial,
    slot: BackendValueSlot,
    view: ViewRequirement<JoltVmNamespace>,
}

#[cfg(test)]
fn build_stage1_r1cs_materialization_request<F, W>(
    config: Stage1ProverConfig,
    witness: &W,
) -> Result<Stage1R1csMaterializationRequest, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let r1cs_inputs = config
        .dimensions()
        .variables()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| {
            let oracle = OracleRef::virtual_polynomial(variable);
            let view = primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle)?;
            Ok(Stage1R1csInputRequest {
                variable,
                slot: stage1_r1cs_input_slot(index),
                view,
            })
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let views = r1cs_inputs
        .iter()
        .map(|input| SumcheckViewMaterializationRequest::new(input.slot, input.view))
        .collect();

    Ok(Stage1R1csMaterializationRequest {
        materializations: SumcheckMaterializationRequest::new(
            "stage1.spartan_outer.r1cs_inputs.materialize",
            views,
        ),
        r1cs_inputs,
    })
}

/// Proves Stage 1 with the trace-row Spartan-outer CPU kernel, in verifier
/// order: prelude challenges, uni-skip, remainder, opening claims.
pub fn prove<F, W, B, T, C>(
    input: Stage1ProverInput<'_, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage1ProofComponent<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: JoltVmSpartanOuterRows + WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
{
    let config = input.config;
    let witness = input.witness;
    let spartan_rows = materialize_stage1_spartan_outer_rows(config, witness)?;
    let backend_rows = spartan_rows
        .iter()
        .copied()
        .map(spartan_outer_row)
        .collect::<Vec<_>>();
    let context = SpartanOuterContext::new(config, Vec::new());

    prove_stage1_transparent_sumchecks_with_context(
        config,
        backend,
        transcript,
        context,
        Some(&backend_rows),
        |_, backend, tau| {
            prove_uniskip_round_poly_from_spartan_outer_backend_rows(
                config,
                &backend_rows,
                backend,
                tau,
            )
        },
        |context, _backend, opening_point| {
            let r1cs_input_claims = evaluate_stage1_r1cs_inputs_from_spartan_outer_rows_at_point(
                context.config,
                &spartan_rows,
                &opening_point,
            )?;
            Ok(Stage1OpeningClaims { r1cs_input_claims })
        },
    )
}

#[cfg(feature = "zk")]
pub fn prove_committed_proof_component<F, W, B, T, VC>(
    input: Stage1ProverInput<'_, W>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage1CommittedProofComponent<F, VC>, ProverError>
where
    F: Field,
    W: JoltVmSpartanOuterRows + WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: jolt_transcript::AppendToTranscript,
{
    let config = input.config;
    let witness = input.witness;
    let spartan_rows = materialize_stage1_spartan_outer_rows(config, witness)?;
    let backend_rows = spartan_rows
        .iter()
        .copied()
        .map(spartan_outer_row)
        .collect::<Vec<_>>();
    let context = SpartanOuterContext::new(config, Vec::new());

    prove_stage1_committed_sumchecks_with_context::<F, B, T, VC, _, _>(
        config,
        backend,
        transcript,
        vc_setup,
        context,
        Some(&backend_rows),
        |_, backend, tau| {
            prove_uniskip_round_poly_from_spartan_outer_backend_rows(
                config,
                &backend_rows,
                backend,
                tau,
            )
        },
        |context, _backend, opening_point| {
            let r1cs_input_claims = evaluate_stage1_r1cs_inputs_from_spartan_outer_rows_at_point(
                context.config,
                &spartan_rows,
                &opening_point,
            )?;
            Ok(Stage1OpeningClaims { r1cs_input_claims })
        },
    )
}

#[cfg(test)]
pub(super) fn prove_stage1_transparent_sumchecks<F, W, B, T, C>(
    config: Stage1ProverConfig,
    witness: &W,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage1ProofComponent<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
{
    let request = build_stage1_r1cs_materialization_request::<F, W>(config, witness)?;
    let materializations =
        backend.materialize_sumcheck_views(&request.materializations, witness)?;
    let r1cs_inputs = materialized_stage1_r1cs_inputs(config, &request, materializations)?;
    let context = SpartanOuterContext::new(config, r1cs_inputs);

    prove_stage1_transparent_sumchecks_with_context(
        config,
        backend,
        transcript,
        context,
        None,
        prove_uniskip_round_poly,
        |context, backend, opening_point| {
            let r1cs_input_claims =
                evaluate_stage1_r1cs_inputs_from_context(context, &opening_point)?;
            let _ = backend;
            Ok(Stage1OpeningClaims { r1cs_input_claims })
        },
    )
}

fn prove_stage1_transparent_sumchecks_with_context<F, B, T, C, U, E>(
    config: Stage1ProverConfig,
    backend: &mut B,
    transcript: &mut T,
    context: SpartanOuterContext<F>,
    backend_rows: Option<&[SumcheckSpartanOuterRow]>,
    prove_uniskip: U,
    evaluate_openings: E,
) -> Result<Stage1ProofComponent<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
    U: FnOnce(&SpartanOuterContext<F>, &mut B, &[F]) -> Result<UnivariatePoly<F>, ProverError>,
    E: FnOnce(
        &SpartanOuterContext<F>,
        &mut B,
        Vec<F>,
    ) -> Result<Stage1OpeningClaims<F>, ProverError>,
{
    let tau = transcript.challenge_vector(config.log_t + 2);
    let uniskip_poly = prove_uniskip(&context, backend, &tau)?;
    LabeledRoundPoly::uniskip(&uniskip_poly).append_to_transcript(transcript);
    let uniskip_challenge = transcript.challenge();
    let uniskip_output_claim = uniskip_poly.evaluate(uniskip_challenge);
    let uniskip_proof = SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof {
        round_polynomials: vec![uniskip_poly],
    }));

    transcript.append_labeled(b"opening_claim", &uniskip_output_claim);
    append_sumcheck_claim(transcript, &uniskip_output_claim);
    let batching_coefficient = transcript.challenge_scalar();

    let remainder = if let Some(rows) = backend_rows {
        prove_remainder_sumcheck_from_spartan_outer_backend_rows(
            config,
            rows,
            backend,
            &tau,
            uniskip_challenge,
            batching_coefficient,
            uniskip_output_claim,
            transcript,
        )
    } else {
        prove_remainder_sumcheck(
            &context,
            backend,
            &tau,
            uniskip_challenge,
            batching_coefficient,
            uniskip_output_claim,
            transcript,
        )
    }?;

    let opening_point = normalized_remainder_opening_point(&remainder.challenges);
    let opening_claims = evaluate_openings(&context, backend, opening_point)?;
    let opening_values = opening_claims.opening_values();
    let expected_remainder_output =
        JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
            tau: &tau,
            uniskip: uniskip_challenge,
            remainder: &remainder.challenges,
        })
        .map_err(invalid_sumcheck_output)?
        .expected_output_claim(&opening_values)
        .map_err(invalid_sumcheck_output)?
            * batching_coefficient;
    if remainder.output_claim != expected_remainder_output {
        return Err(invalid_sumcheck_output(
            "Stage 1 transparent remainder proof final claim did not match R1CS openings",
        ));
    }
    for opening_claim in &opening_values {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }

    let verifier_output = Some(stage1_clear_output(
        tau,
        uniskip_challenge,
        uniskip_output_claim,
        batching_coefficient,
        remainder.challenges.clone(),
        remainder.output_claim,
        expected_remainder_output,
        opening_claims
            .r1cs_input_claims
            .iter()
            .map(Stage1R1csInputClaim::verifier_input),
    )?);

    Ok(Stage1ProofComponent {
        uniskip_proof,
        remainder_proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials: remainder.round_polynomials,
        })),
        uniskip_output_claim,
        remainder_output_claim: remainder.output_claim,
        r1cs_input_claims: opening_claims.r1cs_input_claims,
        verifier_output,
    })
}

#[cfg(feature = "zk")]
#[expect(
    clippy::too_many_arguments,
    reason = "Temporary shared clear/ZK Stage 1 helper; Phase 6 will collapse this into a mode-aware stage input."
)]
fn prove_stage1_committed_sumchecks_with_context<F, B, T, VC, U, E>(
    config: Stage1ProverConfig,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
    context: SpartanOuterContext<F>,
    backend_rows: Option<&[SumcheckSpartanOuterRow]>,
    prove_uniskip: U,
    evaluate_openings: E,
) -> Result<Stage1CommittedProofComponent<F, VC>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: jolt_transcript::AppendToTranscript,
    U: FnOnce(&SpartanOuterContext<F>, &mut B, &[F]) -> Result<UnivariatePoly<F>, ProverError>,
    E: FnOnce(
        &SpartanOuterContext<F>,
        &mut B,
        Vec<F>,
    ) -> Result<Stage1OpeningClaims<F>, ProverError>,
{
    let tau = transcript.challenge_vector(config.log_t + 2);
    let uniskip_poly = prove_uniskip(&context, backend, &tau)?;
    let mut uniskip_builder = CommittedSumcheckBuilder::<F, VC>::new(vc_setup, 1)?;
    let uniskip_challenge = uniskip_builder.commit_round(&uniskip_poly, transcript)?;
    let uniskip_output_claim = uniskip_poly.evaluate(uniskip_challenge);
    let uniskip_output_claim_values = vec![uniskip_output_claim];
    let uniskip_built = uniskip_builder.finish(&uniskip_output_claim_values, transcript)?;

    let batching_coefficient = transcript.challenge_scalar();
    let remainder = if let Some(rows) = backend_rows {
        prove_remainder_committed_sumcheck_from_spartan_outer_backend_rows::<F, B, T, VC>(
            config,
            rows,
            backend,
            &tau,
            uniskip_challenge,
            batching_coefficient,
            uniskip_output_claim,
            transcript,
            vc_setup,
        )
    } else {
        prove_remainder_committed_sumcheck::<F, B, T, VC>(
            &context,
            backend,
            &tau,
            uniskip_challenge,
            batching_coefficient,
            uniskip_output_claim,
            transcript,
            vc_setup,
        )
    }?;

    let opening_point = normalized_remainder_opening_point(&remainder.challenges);
    let opening_claims = evaluate_openings(&context, backend, opening_point)?;
    let remainder_output_claim_values = opening_claims.opening_values();
    let expected_remainder_output =
        JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
            tau: &tau,
            uniskip: uniskip_challenge,
            remainder: &remainder.challenges,
        })
        .map_err(invalid_sumcheck_output)?
        .expected_output_claim(&remainder_output_claim_values)
        .map_err(invalid_sumcheck_output)?
            * batching_coefficient;
    if remainder.output_claim != expected_remainder_output {
        return Err(invalid_sumcheck_output(
            "Stage 1 committed remainder proof final claim did not match R1CS openings",
        ));
    }

    let remainder_built = remainder
        .builder
        .finish(&remainder_output_claim_values, transcript)?;
    let verifier_output = stage1_clear_output(
        tau,
        uniskip_challenge,
        uniskip_output_claim,
        batching_coefficient,
        remainder.challenges,
        remainder.output_claim,
        expected_remainder_output,
        opening_claims
            .r1cs_input_claims
            .iter()
            .map(Stage1R1csInputClaim::verifier_input),
    )?;

    Ok(Stage1CommittedProofComponent {
        uniskip_proof: uniskip_built.proof,
        remainder_proof: remainder_built.proof,
        public: verifier_output.public.clone(),
        verifier_output,
        uniskip_output_claim_values,
        remainder_output_claim_values,
        uniskip_committed_witness: uniskip_built.witness,
        remainder_committed_witness: remainder_built.witness,
    })
}

fn materialize_stage1_spartan_outer_rows<W>(
    config: Stage1ProverConfig,
    witness: &W,
) -> Result<Vec<JoltVmSpartanOuterRow>, ProverError>
where
    W: JoltVmSpartanOuterRows,
{
    let rows = witness.spartan_outer_rows()?;
    let expected_rows = checked_trace_rows(config.log_t, "Stage 1")?;
    if rows.len() != expected_rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 1 Spartan outer witness returned {} rows, expected {expected_rows}",
                rows.len()
            ),
        });
    }
    Ok(rows)
}

#[cfg(test)]
fn evaluate_stage1_r1cs_inputs_from_context<F>(
    context: &SpartanOuterContext<F>,
    point: &[F],
) -> Result<Vec<Stage1R1csInputClaim<F>>, ProverError>
where
    F: Field,
{
    let dimensions = context.config.dimensions();
    let variables = dimensions.variables();
    let values = evaluate_context_r1cs_columns(context, 0, variables.len(), point)?;
    variables
        .iter()
        .copied()
        .enumerate()
        .zip(values)
        .map(|((index, variable), value)| {
            Ok(Stage1R1csInputClaim {
                variable,
                slot: stage1_r1cs_input_slot(index),
                value,
            })
        })
        .collect()
}

fn evaluate_stage1_r1cs_inputs_from_spartan_outer_rows_at_point<F>(
    config: Stage1ProverConfig,
    rows: &[JoltVmSpartanOuterRow],
    point: &[F],
) -> Result<Vec<Stage1R1csInputClaim<F>>, ProverError>
where
    F: Field,
{
    if point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 1 raw-row R1CS evaluation point has {} variables, expected {}",
                point.len(),
                config.log_t
            ),
        });
    }
    let expected_rows = checked_trace_rows(config.log_t, "Stage 1")?;
    if rows.len() != expected_rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 1 raw-row R1CS evaluation received {} rows, expected {expected_rows}",
                rows.len()
            ),
        });
    }

    let dimensions = config.dimensions();
    let variables = dimensions.variables();
    if variables == SPARTAN_OUTER_R1CS_INPUTS.as_slice() {
        return Ok(evaluate_stage1_rv64_r1cs_inputs_from_spartan_outer_rows_at_point(rows, point));
    }
    for &variable in variables {
        if !spartan_outer_row_variable_supported(variable) {
            return Err(invalid_sumcheck_output(format!(
                "unsupported Stage 1 Spartan outer R1CS input {variable:?}"
            )));
        }
    }

    let eq = TensorEqTable::new(point);
    let values = eq.par_fold_out_in(
        || vec![F::zero(); variables.len()],
        |inner, row_index, _x_in, e_in| {
            if e_in.is_zero() {
                return;
            }
            let row = &rows[row_index];
            for (accumulator, &variable) in inner.iter_mut().zip(variables) {
                let value = spartan_outer_row_value_known(row, variable).unwrap_or(F::zero());
                *accumulator += e_in * value;
            }
        },
        |_x_out, e_out, mut inner| {
            if e_out.is_zero() {
                inner.fill(F::zero());
            } else {
                for value in &mut inner {
                    *value *= e_out;
                }
            }
            inner
        },
        |mut left, right| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        },
    );

    variables
        .iter()
        .copied()
        .enumerate()
        .zip(values)
        .map(|((index, variable), value)| {
            Ok(Stage1R1csInputClaim {
                variable,
                slot: stage1_r1cs_input_slot(index),
                value,
            })
        })
        .collect()
}

fn evaluate_stage1_rv64_r1cs_inputs_from_spartan_outer_rows_at_point<F>(
    rows: &[JoltVmSpartanOuterRow],
    point: &[F],
) -> Vec<Stage1R1csInputClaim<F>>
where
    F: Field,
{
    let eq = TensorEqTable::new(point);
    let values = eq.par_fold_out_in(
        || [F::zero(); STAGE1_RV64_R1CS_INPUT_COUNT],
        |inner, row_index, _x_in, e_in| {
            if e_in.is_zero() {
                return;
            }
            let row = &rows[row_index];
            inner[0] += e_in * F::from_u64(row.left_instruction_input);
            inner[1] += e_in * F::from_i128(row.right_instruction_input);
            let product = e_in * F::from_u128(row.product_magnitude);
            if row.product_is_positive {
                inner[2] += product;
            } else {
                inner[2] -= product;
            }
            accumulate_bool(&mut inner[3], e_in, row.should_branch);
            inner[4] += e_in * F::from_u64(row.pc);
            inner[5] += e_in * F::from_u64(row.unexpanded_pc);
            inner[6] += e_in * F::from_i128(row.imm);
            inner[7] += e_in * F::from_u64(row.ram_address);
            inner[8] += e_in * F::from_u64(row.rs1_value);
            inner[9] += e_in * F::from_u64(row.rs2_value);
            inner[10] += e_in * F::from_u64(row.rd_write_value);
            inner[11] += e_in * F::from_u64(row.ram_read_value);
            inner[12] += e_in * F::from_u64(row.ram_write_value);
            inner[13] += e_in * F::from_u64(row.left_lookup_operand);
            inner[14] += e_in * F::from_u128(row.right_lookup_operand);
            inner[15] += e_in * F::from_u64(row.next_unexpanded_pc);
            inner[16] += e_in * F::from_u64(row.next_pc);
            accumulate_bool(&mut inner[17], e_in, row.next_is_virtual);
            accumulate_bool(&mut inner[18], e_in, row.next_is_first_in_sequence);
            inner[19] += e_in * F::from_u64(row.lookup_output);
            accumulate_bool(&mut inner[20], e_in, row.should_jump);
            accumulate_bool(&mut inner[21], e_in, row.flag_add_operands);
            accumulate_bool(&mut inner[22], e_in, row.flag_subtract_operands);
            accumulate_bool(&mut inner[23], e_in, row.flag_multiply_operands);
            accumulate_bool(&mut inner[24], e_in, row.flag_load);
            accumulate_bool(&mut inner[25], e_in, row.flag_store);
            accumulate_bool(&mut inner[26], e_in, row.flag_jump);
            accumulate_bool(&mut inner[27], e_in, row.flag_write_lookup_output_to_rd);
            accumulate_bool(&mut inner[28], e_in, row.flag_virtual_instruction);
            accumulate_bool(&mut inner[29], e_in, row.flag_assert);
            accumulate_bool(&mut inner[30], e_in, row.flag_do_not_update_unexpanded_pc);
            accumulate_bool(&mut inner[31], e_in, row.flag_advice);
            accumulate_bool(&mut inner[32], e_in, row.flag_is_compressed);
            accumulate_bool(&mut inner[33], e_in, row.flag_is_first_in_sequence);
            accumulate_bool(&mut inner[34], e_in, row.flag_is_last_in_sequence);
        },
        |_x_out, e_out, mut inner| {
            if e_out.is_zero() {
                inner.fill(F::zero());
            } else {
                for value in &mut inner {
                    *value *= e_out;
                }
            }
            inner
        },
        |mut left, right| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        },
    );

    SPARTAN_OUTER_R1CS_INPUTS
        .iter()
        .copied()
        .enumerate()
        .zip(values)
        .map(|((index, variable), value)| Stage1R1csInputClaim {
            variable,
            slot: stage1_r1cs_input_slot(index),
            value,
        })
        .collect()
}

#[inline]
fn accumulate_bool<F: Field>(accumulator: &mut F, scale: F, flag: bool) {
    if flag {
        *accumulator += scale;
    }
}

#[cfg(test)]
fn evaluate_context_r1cs_columns<F>(
    context: &SpartanOuterContext<F>,
    start: usize,
    count: usize,
    point: &[F],
) -> Result<Vec<F>, ProverError>
where
    F: Field,
{
    if point.len() != context.config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 1 R1CS context evaluation point has {} variables, expected {}",
                point.len(),
                context.config.log_t
            ),
        });
    }
    let end = start
        .checked_add(count)
        .ok_or_else(|| invalid_sumcheck_output("Stage 1 R1CS context column range overflowed"))?;
    if end > context.r1cs_inputs.len() {
        return Err(invalid_sumcheck_output(format!(
            "Stage 1 R1CS context has {} columns, requested range {start}..{end}",
            context.r1cs_inputs.len()
        )));
    }
    let expected_rows = checked_trace_rows(context.config.log_t, "Stage 1")?;
    for (index, column) in context.r1cs_inputs[start..end].iter().enumerate() {
        if column.len() != expected_rows {
            return Err(invalid_sumcheck_output(format!(
                "Stage 1 R1CS context column {} has {} rows, expected {expected_rows}",
                start + index,
                column.len()
            )));
        }
    }
    let eq = TensorEqTable::new(point);
    Ok(eq.evaluate_slices(
        &context.r1cs_inputs[start..end]
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>(),
    ))
}

fn spartan_outer_row_variable_supported(variable: JoltVirtualPolynomial) -> bool {
    matches!(
        variable,
        JoltVirtualPolynomial::LeftInstructionInput
            | JoltVirtualPolynomial::RightInstructionInput
            | JoltVirtualPolynomial::Product
            | JoltVirtualPolynomial::ShouldBranch
            | JoltVirtualPolynomial::PC
            | JoltVirtualPolynomial::UnexpandedPC
            | JoltVirtualPolynomial::Imm
            | JoltVirtualPolynomial::RamAddress
            | JoltVirtualPolynomial::Rs1Value
            | JoltVirtualPolynomial::Rs2Value
            | JoltVirtualPolynomial::RdWriteValue
            | JoltVirtualPolynomial::RamReadValue
            | JoltVirtualPolynomial::RamWriteValue
            | JoltVirtualPolynomial::LeftLookupOperand
            | JoltVirtualPolynomial::RightLookupOperand
            | JoltVirtualPolynomial::NextUnexpandedPC
            | JoltVirtualPolynomial::NextPC
            | JoltVirtualPolynomial::NextIsVirtual
            | JoltVirtualPolynomial::NextIsFirstInSequence
            | JoltVirtualPolynomial::LookupOutput
            | JoltVirtualPolynomial::ShouldJump
            | JoltVirtualPolynomial::OpFlags(_)
    )
}

fn spartan_outer_row_value_known<F>(
    row: &JoltVmSpartanOuterRow,
    variable: JoltVirtualPolynomial,
) -> Option<F>
where
    F: Field,
{
    let value = match variable {
        JoltVirtualPolynomial::LeftInstructionInput => F::from_u64(row.left_instruction_input),
        JoltVirtualPolynomial::RightInstructionInput => F::from_i128(row.right_instruction_input),
        JoltVirtualPolynomial::Product => signed_product_from_spartan_outer_row(row),
        JoltVirtualPolynomial::ShouldBranch => F::from_bool(row.should_branch),
        JoltVirtualPolynomial::PC => F::from_u64(row.pc),
        JoltVirtualPolynomial::UnexpandedPC => F::from_u64(row.unexpanded_pc),
        JoltVirtualPolynomial::Imm => F::from_i128(row.imm),
        JoltVirtualPolynomial::RamAddress => F::from_u64(row.ram_address),
        JoltVirtualPolynomial::Rs1Value => F::from_u64(row.rs1_value),
        JoltVirtualPolynomial::Rs2Value => F::from_u64(row.rs2_value),
        JoltVirtualPolynomial::RdWriteValue => F::from_u64(row.rd_write_value),
        JoltVirtualPolynomial::RamReadValue => F::from_u64(row.ram_read_value),
        JoltVirtualPolynomial::RamWriteValue => F::from_u64(row.ram_write_value),
        JoltVirtualPolynomial::LeftLookupOperand => F::from_u64(row.left_lookup_operand),
        JoltVirtualPolynomial::RightLookupOperand => F::from_u128(row.right_lookup_operand),
        JoltVirtualPolynomial::NextUnexpandedPC => F::from_u64(row.next_unexpanded_pc),
        JoltVirtualPolynomial::NextPC => F::from_u64(row.next_pc),
        JoltVirtualPolynomial::NextIsVirtual => F::from_bool(row.next_is_virtual),
        JoltVirtualPolynomial::NextIsFirstInSequence => F::from_bool(row.next_is_first_in_sequence),
        JoltVirtualPolynomial::LookupOutput => F::from_u64(row.lookup_output),
        JoltVirtualPolynomial::ShouldJump => F::from_bool(row.should_jump),
        JoltVirtualPolynomial::OpFlags(flag) => F::from_bool(spartan_outer_flag(row, flag)),
        _ => return None,
    };
    Some(value)
}

fn signed_product_from_spartan_outer_row<F: Field>(row: &JoltVmSpartanOuterRow) -> F {
    let magnitude = F::from_u128(row.product_magnitude);
    if row.product_is_positive {
        magnitude
    } else {
        -magnitude
    }
}

fn spartan_outer_flag(row: &JoltVmSpartanOuterRow, flag: CircuitFlags) -> bool {
    match flag {
        CircuitFlags::AddOperands => row.flag_add_operands,
        CircuitFlags::SubtractOperands => row.flag_subtract_operands,
        CircuitFlags::MultiplyOperands => row.flag_multiply_operands,
        CircuitFlags::Load => row.flag_load,
        CircuitFlags::Store => row.flag_store,
        CircuitFlags::Jump => row.flag_jump,
        CircuitFlags::WriteLookupOutputToRD => row.flag_write_lookup_output_to_rd,
        CircuitFlags::VirtualInstruction => row.flag_virtual_instruction,
        CircuitFlags::Assert => row.flag_assert,
        CircuitFlags::DoNotUpdateUnexpandedPC => row.flag_do_not_update_unexpanded_pc,
        CircuitFlags::Advice => row.flag_advice,
        CircuitFlags::IsCompressed => row.flag_is_compressed,
        CircuitFlags::IsFirstInSequence => row.flag_is_first_in_sequence,
        CircuitFlags::IsLastInSequence => row.flag_is_last_in_sequence,
    }
}

struct SpartanOuterContext<F: Field> {
    config: Stage1ProverConfig,
    r1cs_inputs: Vec<Vec<F>>,
    opening_columns: Vec<usize>,
    matrices: ConstraintMatrices<F>,
}

impl<F: Field> SpartanOuterContext<F> {
    fn new(config: Stage1ProverConfig, r1cs_inputs: Vec<Vec<F>>) -> Self {
        Self {
            config,
            r1cs_inputs,
            opening_columns: spartan_outer_opening_columns(),
            matrices: spartan_outer_constraints(),
        }
    }

    fn prefix_product_sum_request(
        &self,
        label: &'static str,
        queries: Vec<SumcheckPrefixProductSumQuery<F>>,
    ) -> SumcheckPrefixProductSumRequest<'_, F> {
        SumcheckPrefixProductSumRequest::new(
            label,
            &self.r1cs_inputs,
            &self.opening_columns,
            rv64::const_column(),
            &self.matrices.a,
            &self.matrices.b,
            queries,
        )
    }

    fn remainder_state_request(
        &self,
        params: &RemainderEvalParams<'_, F>,
        stream_challenge: F,
    ) -> Result<SumcheckSpartanOuterRemainderStateRequest<'_, F>, ProverError> {
        let tau_kernel_scale =
            spartan_outer_tau_kernel_scale(params.tau, params.uniskip_challenge)?
                * params.batching_coefficient;
        let tau_low = &params.tau[..params.tau.len() - 1];
        let (row_weights_at_zero, row_weights_at_one) =
            spartan_outer_stream_row_weights(params.uniskip_challenge)?;
        Ok(SumcheckSpartanOuterRemainderStateRequest::new(
            "stage1.spartan_outer.remainder_state",
            &self.r1cs_inputs,
            &self.opening_columns,
            rv64::const_column(),
            &self.matrices.a,
            &self.matrices.b,
            tau_low.to_vec(),
            row_weights_at_zero,
            row_weights_at_one,
            stream_challenge,
            tau_kernel_scale,
        )
        .with_relation(SPARTAN_OUTER_REMAINDER_RELATION)
        .with_optimization_ids(STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS))
    }
}

struct RemainderProof<F: Field> {
    round_polynomials: Vec<jolt_poly::CompressedPoly<F>>,
    challenges: Vec<F>,
    output_claim: F,
}

#[cfg(feature = "zk")]
struct PendingCommittedRemainder<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    builder: CommittedSumcheckBuilder<'a, F, VC>,
    challenges: Vec<F>,
    output_claim: F,
}

struct Stage1OpeningClaims<F: Field> {
    r1cs_input_claims: Vec<Stage1R1csInputClaim<F>>,
}

impl<F: Field> Stage1OpeningClaims<F> {
    fn opening_values(&self) -> Vec<F> {
        let values = self
            .r1cs_input_claims
            .iter()
            .map(|claim| claim.value)
            .collect::<Vec<_>>();
        {
            values
        }
    }
}

struct RemainderEvalParams<'a, F: Field> {
    tau: &'a [F],
    uniskip_challenge: F,
    batching_coefficient: F,
}

#[cfg(test)]
fn materialized_stage1_r1cs_inputs<F>(
    config: Stage1ProverConfig,
    request: &Stage1R1csMaterializationRequest,
    materializations: Vec<SumcheckMaterializationOutput<F>>,
) -> Result<Vec<Vec<F>>, ProverError>
where
    F: Field,
{
    let mut by_slot = std::collections::BTreeMap::new();
    for output in materializations {
        if by_slot.insert(output.slot, output.values).is_some() {
            return Err(invalid_sumcheck_output(format!(
                "duplicate Stage 1 R1CS materialization slot {:?}",
                output.slot
            )));
        }
    }
    let expected_len = 1usize << config.log_t;
    let inputs = request
        .r1cs_inputs
        .iter()
        .map(|input| {
            let values = by_slot.remove(&input.slot).ok_or_else(|| {
                invalid_sumcheck_output(format!(
                    "missing Stage 1 R1CS materialization for {:?}",
                    input.variable
                ))
            })?;
            if values.len() != expected_len {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 1 R1CS input {:?} materialized {} rows, expected {expected_len}",
                    input.variable,
                    values.len()
                )));
            }
            Ok(values)
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    if let Some(slot) = by_slot.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 1 R1CS materialization slot {slot:?}"
        )));
    }
    Ok(inputs)
}

#[cfg(test)]
fn prove_uniskip_round_poly<F, B>(
    context: &SpartanOuterContext<F>,
    backend: &mut B,
    tau: &[F],
) -> Result<UnivariatePoly<F>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    if tau.len() != context.config.log_t + 2 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 1 tau has {} variables, expected {}",
            tau.len(),
            context.config.log_t + 2
        )));
    }
    let degree = SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE - 1;
    let targets = uniskip_targets(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, degree)?;
    let tau_low = &tau[..tau.len() - 1];
    let queries = targets
        .into_iter()
        .enumerate()
        .map(|(index, target)| {
            let (row_weights_at_zero, row_weights_at_one) =
                spartan_outer_stream_row_weights(F::from_i64(target))?;
            Ok(SumcheckPrefixProductSumQuery::new(
                value_slot(index)?,
                tau_low.to_vec(),
                Vec::new(),
                context.config.log_t + 1,
                row_weights_at_zero,
                row_weights_at_one,
                F::one(),
            ))
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let product_request = context
        .prefix_product_sum_request("stage1.spartan_outer.uniskip_t1", queries)
        .with_relation(SPARTAN_OUTER_UNISKIP_RELATION)
        .with_optimization_ids(STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS);
    let outputs = backend.evaluate_sumcheck_prefix_product_sums(&product_request)?;
    let extended_evals = ordered_linear_product_outputs(outputs, degree)?;
    build_uniskip_first_round_poly(
        SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
        &extended_evals,
        tau[tau.len() - 1],
    )
}

fn prove_uniskip_round_poly_from_spartan_outer_backend_rows<F, B>(
    config: Stage1ProverConfig,
    rows: &[SumcheckSpartanOuterRow],
    backend: &mut B,
    tau: &[F],
) -> Result<UnivariatePoly<F>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    if tau.len() != config.log_t + 2 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 1 tau has {} variables, expected {}",
            tau.len(),
            config.log_t + 2
        )));
    }
    let expected_rows = checked_trace_rows(config.log_t, "Stage 1")?;
    if rows.len() != expected_rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 1 Spartan outer witness returned {} rows, expected {expected_rows}",
                rows.len()
            ),
        });
    }

    let degree = SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE - 1;
    let tau_low = &tau[..tau.len() - 1];
    let queries = uniskip_targets(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, degree)?
        .into_iter()
        .enumerate()
        .map(|(index, target)| {
            Ok(SumcheckSpartanOuterUniskipQuery::new(
                value_slot(index)?,
                tau_low.to_vec(),
                centered_lagrange_integer_coeffs(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, target)?,
                F::one(),
            ))
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let request = SumcheckSpartanOuterUniskipRequest::new(
        "stage1.spartan_outer.uniskip_raw_rows",
        rows,
        queries,
    )
    .with_relation(SPARTAN_OUTER_UNISKIP_RELATION)
    .with_optimization_ids(STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS);
    let outputs = backend.evaluate_sumcheck_spartan_outer_uniskip_rows(&request)?;
    let extended_evals = ordered_linear_product_outputs(outputs, degree)?;
    build_uniskip_first_round_poly(
        SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
        &extended_evals,
        tau[tau.len() - 1],
    )
}

fn prove_remainder_sumcheck<F, B, T>(
    context: &SpartanOuterContext<F>,
    backend: &mut B,
    tau: &[F],
    uniskip_challenge: F,
    batching_coefficient: F,
    uniskip_output_claim: F,
    transcript: &mut T,
) -> Result<RemainderProof<F>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
{
    let num_vars = context.config.log_t + 1;
    let mut prefix = Vec::with_capacity(num_vars);
    let mut round_polynomials = Vec::with_capacity(num_vars);
    let mut running_claim = uniskip_output_claim * batching_coefficient;
    let params = RemainderEvalParams {
        tau,
        uniskip_challenge,
        batching_coefficient,
    };
    let mut state = None;

    for round in 0..num_vars {
        let suffix_vars = num_vars - round - 1;
        let evaluations = if round == 0 {
            sum_remainder_suffix_evals(context, backend, &params, &prefix, suffix_vars)?
        } else {
            let state_ref = state.as_ref().ok_or_else(|| {
                invalid_sumcheck_output("Stage 1 remainder state was not initialized")
            })?;
            let round_inputs =
                backend.evaluate_sumcheck_spartan_outer_remainder_round(state_ref)?;
            let round_poly =
                remainder_round_poly_from_state(state_ref, round_inputs, running_claim)?;
            finish_remainder_round(
                round,
                round_poly,
                transcript,
                &mut running_claim,
                &mut prefix,
                &mut round_polynomials,
            )?;
            if round + 1 < num_vars {
                let state_mut = state.as_mut().ok_or_else(|| {
                    invalid_sumcheck_output("Stage 1 remainder state was not initialized")
                })?;
                backend.bind_sumcheck_spartan_outer_remainder_state(state_mut, prefix[round])?;
            }
            continue;
        };
        let round_poly = UnivariatePoly::interpolate_over_integers(&evaluations);
        finish_remainder_round(
            round,
            round_poly,
            transcript,
            &mut running_claim,
            &mut prefix,
            &mut round_polynomials,
        )?;
        if round == 0 {
            state = Some(materialize_remainder_state(
                context,
                backend,
                &params,
                prefix[round],
            )?);
        }
    }

    Ok(RemainderProof {
        round_polynomials,
        challenges: prefix,
        output_claim: running_claim,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Temporary backend-row helper retaining the existing Stage 1 transparent flow until stage inputs are consolidated."
)]
fn prove_remainder_sumcheck_from_spartan_outer_backend_rows<F, B, T>(
    config: Stage1ProverConfig,
    rows: &[SumcheckSpartanOuterRow],
    backend: &mut B,
    tau: &[F],
    uniskip_challenge: F,
    batching_coefficient: F,
    uniskip_output_claim: F,
    transcript: &mut T,
) -> Result<RemainderProof<F>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
{
    let num_vars = config.log_t + 1;
    let mut prefix = Vec::with_capacity(num_vars);
    let mut round_polynomials = Vec::with_capacity(num_vars);
    let mut running_claim = uniskip_output_claim * batching_coefficient;
    let params = RemainderEvalParams {
        tau,
        uniskip_challenge,
        batching_coefficient,
    };
    let mut state = None;

    for round in 0..num_vars {
        let suffix_vars = num_vars - round - 1;
        if round == 0 {
            let evaluations = sum_remainder_suffix_evals_from_spartan_outer_backend_rows(
                config,
                rows,
                backend,
                &params,
                &prefix,
                suffix_vars,
            )?;
            let round_poly = UnivariatePoly::interpolate_over_integers(&evaluations);
            finish_remainder_round(
                round,
                round_poly,
                transcript,
                &mut running_claim,
                &mut prefix,
                &mut round_polynomials,
            )?;
            state = Some(materialize_remainder_row_state(
                config,
                rows,
                backend,
                &params,
                prefix[round],
            )?);
            continue;
        }

        let state_ref = state.as_ref().ok_or_else(|| {
            invalid_sumcheck_output("Stage 1 remainder state was not initialized")
        })?;
        let round_inputs = backend.evaluate_sumcheck_spartan_outer_remainder_round(state_ref)?;
        let round_poly = remainder_round_poly_from_state(state_ref, round_inputs, running_claim)?;
        finish_remainder_round(
            round,
            round_poly,
            transcript,
            &mut running_claim,
            &mut prefix,
            &mut round_polynomials,
        )?;
        if round + 1 < num_vars {
            let state_mut = state.as_mut().ok_or_else(|| {
                invalid_sumcheck_output("Stage 1 remainder state was not initialized")
            })?;
            backend.bind_sumcheck_spartan_outer_remainder_state(state_mut, prefix[round])?;
        }
    }

    Ok(RemainderProof {
        round_polynomials,
        challenges: prefix,
        output_claim: running_claim,
    })
}

#[cfg(feature = "zk")]
#[expect(
    clippy::too_many_arguments,
    reason = "Mirrors the transparent Stage 1 remainder loop while swapping only the proof recorder."
)]
fn prove_remainder_committed_sumcheck<'a, F, B, T, VC>(
    context: &SpartanOuterContext<F>,
    backend: &mut B,
    tau: &[F],
    uniskip_challenge: F,
    batching_coefficient: F,
    uniskip_output_claim: F,
    transcript: &mut T,
    vc_setup: &'a VC::Setup,
) -> Result<PendingCommittedRemainder<'a, F, VC>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: jolt_transcript::AppendToTranscript,
{
    let num_vars = context.config.log_t + 1;
    let mut prefix = Vec::with_capacity(num_vars);
    let mut running_claim = uniskip_output_claim * batching_coefficient;
    let params = RemainderEvalParams {
        tau,
        uniskip_challenge,
        batching_coefficient,
    };
    let mut state = None;
    let mut builder = CommittedSumcheckBuilder::<F, VC>::new(vc_setup, num_vars)?;

    for round in 0..num_vars {
        let suffix_vars = num_vars - round - 1;
        let round_poly = if round == 0 {
            let evaluations =
                sum_remainder_suffix_evals(context, backend, &params, &prefix, suffix_vars)?;
            UnivariatePoly::interpolate_over_integers(&evaluations)
        } else {
            let state_ref = state.as_ref().ok_or_else(|| {
                invalid_sumcheck_output("Stage 1 committed remainder state was not initialized")
            })?;
            let round_inputs =
                backend.evaluate_sumcheck_spartan_outer_remainder_round(state_ref)?;
            remainder_round_poly_from_state(state_ref, round_inputs, running_claim)?
        };

        let challenge = commit_remainder_round(
            round,
            &round_poly,
            transcript,
            &mut running_claim,
            &mut builder,
        )?;
        prefix.push(challenge);

        if round == 0 {
            state = Some(materialize_remainder_state(
                context, backend, &params, challenge,
            )?);
        } else if round + 1 < num_vars {
            let state_mut = state.as_mut().ok_or_else(|| {
                invalid_sumcheck_output("Stage 1 committed remainder state was not initialized")
            })?;
            backend.bind_sumcheck_spartan_outer_remainder_state(state_mut, challenge)?;
        }
    }

    Ok(PendingCommittedRemainder {
        builder,
        challenges: prefix,
        output_claim: running_claim,
    })
}

#[cfg(feature = "zk")]
#[expect(
    clippy::too_many_arguments,
    reason = "Mirrors the transparent raw-row Stage 1 remainder loop while swapping only the proof recorder."
)]
fn prove_remainder_committed_sumcheck_from_spartan_outer_backend_rows<'a, F, B, T, VC>(
    config: Stage1ProverConfig,
    rows: &[SumcheckSpartanOuterRow],
    backend: &mut B,
    tau: &[F],
    uniskip_challenge: F,
    batching_coefficient: F,
    uniskip_output_claim: F,
    transcript: &mut T,
    vc_setup: &'a VC::Setup,
) -> Result<PendingCommittedRemainder<'a, F, VC>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: jolt_transcript::AppendToTranscript,
{
    let num_vars = config.log_t + 1;
    let mut prefix = Vec::with_capacity(num_vars);
    let mut running_claim = uniskip_output_claim * batching_coefficient;
    let params = RemainderEvalParams {
        tau,
        uniskip_challenge,
        batching_coefficient,
    };
    let mut state = None;
    let mut builder = CommittedSumcheckBuilder::<F, VC>::new(vc_setup, num_vars)?;

    for round in 0..num_vars {
        let suffix_vars = num_vars - round - 1;
        let round_poly = if round == 0 {
            let evaluations = sum_remainder_suffix_evals_from_spartan_outer_backend_rows(
                config,
                rows,
                backend,
                &params,
                &prefix,
                suffix_vars,
            )?;
            UnivariatePoly::interpolate_over_integers(&evaluations)
        } else {
            let state_ref = state.as_ref().ok_or_else(|| {
                invalid_sumcheck_output("Stage 1 committed remainder state was not initialized")
            })?;
            let round_inputs =
                backend.evaluate_sumcheck_spartan_outer_remainder_round(state_ref)?;
            remainder_round_poly_from_state(state_ref, round_inputs, running_claim)?
        };

        let challenge = commit_remainder_round(
            round,
            &round_poly,
            transcript,
            &mut running_claim,
            &mut builder,
        )?;
        prefix.push(challenge);

        if round == 0 {
            state = Some(materialize_remainder_row_state(
                config, rows, backend, &params, challenge,
            )?);
        } else if round + 1 < num_vars {
            let state_mut = state.as_mut().ok_or_else(|| {
                invalid_sumcheck_output("Stage 1 committed remainder state was not initialized")
            })?;
            backend.bind_sumcheck_spartan_outer_remainder_state(state_mut, challenge)?;
        }
    }

    Ok(PendingCommittedRemainder {
        builder,
        challenges: prefix,
        output_claim: running_claim,
    })
}

fn finish_remainder_round<F, T>(
    round: usize,
    round_poly: UnivariatePoly<F>,
    transcript: &mut T,
    running_claim: &mut F,
    prefix: &mut Vec<F>,
    round_polynomials: &mut Vec<jolt_poly::CompressedPoly<F>>,
) -> Result<(), ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
    if round_sum != *running_claim {
        return Err(invalid_sumcheck_output(format!(
            "Stage 1 remainder round {round} sumcheck invariant failed"
        )));
    }

    CompressedLabeledRoundPoly::sumcheck(&round_poly).append_to_transcript(transcript);
    let challenge = transcript.challenge();
    *running_claim = round_poly.evaluate(challenge);
    prefix.push(challenge);
    round_polynomials.push(round_poly.compress());
    Ok(())
}

#[cfg(feature = "zk")]
fn commit_remainder_round<F, T, VC>(
    round: usize,
    round_poly: &UnivariatePoly<F>,
    transcript: &mut T,
    running_claim: &mut F,
    builder: &mut CommittedSumcheckBuilder<'_, F, VC>,
) -> Result<F, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: jolt_transcript::AppendToTranscript,
{
    let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
    if round_sum != *running_claim {
        return Err(invalid_sumcheck_output(format!(
            "Stage 1 committed remainder round {round} sumcheck invariant failed"
        )));
    }

    let challenge = builder.commit_round(round_poly, transcript)?;
    *running_claim = round_poly.evaluate(challenge);
    Ok(challenge)
}

fn remainder_round_poly_from_state<F: Field>(
    state: &SumcheckSpartanOuterRemainderState<F>,
    round: SumcheckSpartanOuterRemainderRound<F>,
    previous_claim: F,
) -> Result<UnivariatePoly<F>, ProverError> {
    if state.active_len == 0 || !state.active_len.is_power_of_two() {
        return Err(invalid_sumcheck_output(format!(
            "Stage 1 remainder state active length {} is invalid",
            state.active_len
        )));
    }
    let active_log = state.active_len.trailing_zeros() as usize;
    if active_log == 0 {
        return Err(invalid_sumcheck_output(
            "Stage 1 remainder state has no unbound variables",
        ));
    }
    let challenge_index = active_log - 1;
    let Some(&tau_current) = state.eq_point.get(challenge_index) else {
        return Err(invalid_sumcheck_output(format!(
            "Stage 1 remainder state missing equality challenge {challenge_index}"
        )));
    };

    let eq_eval_1 = state.scale * tau_current;
    let eq_eval_0 = state.scale - eq_eval_1;
    let eq_slope = eq_eval_1 - eq_eval_0;
    let eq_eval_2 = eq_eval_1 + eq_slope;
    let eq_eval_3 = eq_eval_2 + eq_slope;

    let cubic_eval_0 = eq_eval_0 * round.q_at_zero;
    let cubic_eval_1 = previous_claim - cubic_eval_0;
    let Some(eq_eval_1_inv) = eq_eval_1.inverse() else {
        return Err(invalid_sumcheck_output(
            "Stage 1 remainder current equality evaluation was not invertible",
        ));
    };
    let quadratic_eval_1 = cubic_eval_1 * eq_eval_1_inv;
    let q_inf_times_2 = round.q_at_infinity + round.q_at_infinity;
    let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - round.q_at_zero + q_inf_times_2;
    let quadratic_eval_3 =
        quadratic_eval_2 + quadratic_eval_1 - round.q_at_zero + q_inf_times_2 + q_inf_times_2;

    Ok(UnivariatePoly::from_evals(&[
        cubic_eval_0,
        cubic_eval_1,
        eq_eval_2 * quadratic_eval_2,
        eq_eval_3 * quadratic_eval_3,
    ]))
}

fn sum_remainder_suffix_evals<F: Field>(
    context: &SpartanOuterContext<F>,
    backend: &mut impl SumcheckBackend<F, JoltVmNamespace>,
    params: &RemainderEvalParams<'_, F>,
    prefix: &[F],
    suffix_vars: usize,
) -> Result<Vec<F>, ProverError> {
    let expected_vars = context.config.log_t + 1;
    if prefix.len() + 1 + suffix_vars != expected_vars {
        return Err(invalid_sumcheck_output(format!(
            "Stage 1 remainder prefix fixes {} variables and leaves {suffix_vars} suffix variables, expected {expected_vars} total variables",
            prefix.len() + 1,
        )));
    }
    let tau_kernel_scale = spartan_outer_tau_kernel_scale(params.tau, params.uniskip_challenge)?
        * params.batching_coefficient;
    let tau_low = &params.tau[..params.tau.len() - 1];
    let (row_weights_at_zero, row_weights_at_one) =
        spartan_outer_stream_row_weights(params.uniskip_challenge)?;
    let queries = (0..=SPARTAN_OUTER_REMAINDER_DEGREE)
        .map(|point| {
            let mut fixed_prefix = Vec::with_capacity(prefix.len() + 1);
            fixed_prefix.extend_from_slice(prefix);
            fixed_prefix.push(F::from_u64(point as u64));
            Ok(SumcheckPrefixProductSumQuery::new(
                value_slot(point)?,
                tau_low.to_vec(),
                fixed_prefix,
                suffix_vars,
                row_weights_at_zero.clone(),
                row_weights_at_one.clone(),
                tau_kernel_scale,
            ))
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let product_request = context
        .prefix_product_sum_request("stage1.spartan_outer.remainder_suffix", queries)
        .with_relation(SPARTAN_OUTER_REMAINDER_RELATION)
        .with_optimization_ids(STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS);
    let outputs = backend.evaluate_sumcheck_prefix_product_sums(&product_request)?;
    ordered_linear_product_outputs(outputs, SPARTAN_OUTER_REMAINDER_DEGREE + 1)
}

fn sum_remainder_suffix_evals_from_spartan_outer_backend_rows<F: Field>(
    config: Stage1ProverConfig,
    rows: &[SumcheckSpartanOuterRow],
    backend: &mut impl SumcheckBackend<F, JoltVmNamespace>,
    params: &RemainderEvalParams<'_, F>,
    prefix: &[F],
    suffix_vars: usize,
) -> Result<Vec<F>, ProverError> {
    let expected_vars = config.log_t + 1;
    if prefix.len() + 1 + suffix_vars != expected_vars {
        return Err(invalid_sumcheck_output(format!(
            "Stage 1 raw-row remainder prefix fixes {} variables and leaves {suffix_vars} suffix variables, expected {expected_vars} total variables",
            prefix.len() + 1,
        )));
    }
    let expected_rows = checked_trace_rows(config.log_t, "Stage 1")?;
    if rows.len() != expected_rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 1 raw-row remainder received {} rows, expected {expected_rows}",
                rows.len()
            ),
        });
    }
    let tau_kernel_scale = spartan_outer_tau_kernel_scale(params.tau, params.uniskip_challenge)?
        * params.batching_coefficient;
    let tau_low = &params.tau[..params.tau.len() - 1];
    let queries = (0..=SPARTAN_OUTER_REMAINDER_DEGREE)
        .map(|point| {
            let mut fixed_prefix = Vec::with_capacity(prefix.len() + 1);
            fixed_prefix.extend_from_slice(prefix);
            fixed_prefix.push(F::from_u64(point as u64));
            Ok(SumcheckSpartanOuterRemainderQuery::new(
                value_slot(point)?,
                tau_low.to_vec(),
                fixed_prefix,
                suffix_vars,
                params.uniskip_challenge,
                tau_kernel_scale,
            )
            .with_uniskip_domain_size(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE))
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let request = SumcheckSpartanOuterRemainderRequest::new(
        "stage1.spartan_outer.remainder_raw_rows",
        rows,
        queries,
    )
    .with_relation(SPARTAN_OUTER_REMAINDER_RELATION)
    .with_optimization_ids(STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS);
    let outputs = backend.evaluate_sumcheck_spartan_outer_remainder_rows(&request)?;
    ordered_linear_product_outputs(outputs, SPARTAN_OUTER_REMAINDER_DEGREE + 1)
}

fn materialize_remainder_state<F: Field>(
    context: &SpartanOuterContext<F>,
    backend: &mut impl SumcheckBackend<F, JoltVmNamespace>,
    params: &RemainderEvalParams<'_, F>,
    stream_challenge: F,
) -> Result<SumcheckSpartanOuterRemainderState<F>, ProverError> {
    let request = context.remainder_state_request(params, stream_challenge)?;
    Ok(backend.materialize_sumcheck_spartan_outer_remainder_state(&request)?)
}

fn materialize_remainder_row_state<F: Field>(
    config: Stage1ProverConfig,
    rows: &[SumcheckSpartanOuterRow],
    backend: &mut impl SumcheckBackend<F, JoltVmNamespace>,
    params: &RemainderEvalParams<'_, F>,
    stream_challenge: F,
) -> Result<SumcheckSpartanOuterRemainderState<F>, ProverError> {
    let expected_rows = checked_trace_rows(config.log_t, "Stage 1")?;
    if rows.len() != expected_rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 1 raw-row remainder state received {} rows, expected {expected_rows}",
                rows.len()
            ),
        });
    }
    let tau_kernel_scale = spartan_outer_tau_kernel_scale(params.tau, params.uniskip_challenge)?
        * params.batching_coefficient;
    let tau_low = &params.tau[..params.tau.len() - 1];
    let request = SumcheckSpartanOuterRemainderRowStateRequest::new(
        "stage1.spartan_outer.remainder_raw_row_state",
        rows,
        tau_low.to_vec(),
        params.uniskip_challenge,
        stream_challenge,
        tau_kernel_scale,
    )
    .with_uniskip_domain_size(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE)
    .with_relation(SPARTAN_OUTER_REMAINDER_RELATION)
    .with_optimization_ids(STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS);
    Ok(backend.materialize_sumcheck_spartan_outer_remainder_row_state(&request)?)
}

fn build_uniskip_first_round_poly<F: Field>(
    domain_size: usize,
    extended_evals: &[F],
    tau_high: F,
) -> Result<UnivariatePoly<F>, ProverError> {
    let degree = domain_size - 1;
    if extended_evals.len() != degree {
        return Err(invalid_sumcheck_output(format!(
            "Stage 1 uniskip extended eval count mismatch: got {}, expected {degree}",
            extended_evals.len()
        )));
    }

    let extended_size = 2 * degree + 1;
    let extended_start = -(degree as i64);
    let mut t1_values = vec![F::zero(); extended_size];
    let targets = uniskip_targets(domain_size, degree)?;
    for (target, &value) in targets.iter().zip(extended_evals) {
        let index = usize::try_from(target - extended_start).map_err(|_| {
            invalid_sumcheck_output(format!("Stage 1 uniskip target {target} is out of range"))
        })?;
        t1_values[index] = value;
    }

    let t1_coeffs = interpolate_to_coeffs(extended_start, &t1_values);
    let base_start = centered_domain_start(domain_size).map_err(invalid_sumcheck_output)?;
    let lagrange_values =
        centered_lagrange_evals(domain_size, tau_high).map_err(invalid_sumcheck_output)?;
    let lagrange_coeffs = interpolate_to_coeffs(base_start, &lagrange_values);
    Ok(UnivariatePoly::new(poly_mul(&lagrange_coeffs, &t1_coeffs)))
}

fn uniskip_targets(domain_size: usize, degree: usize) -> Result<Vec<i64>, ProverError> {
    let base_left = centered_domain_start(domain_size).map_err(invalid_sumcheck_output)?;
    let base_right = base_left
        + i64::try_from(domain_size).map_err(|_| {
            invalid_sumcheck_output(format!(
                "Stage 1 uniskip domain size {domain_size} is too large"
            ))
        })?
        - 1;
    let ext_left = -(degree as i64);
    let ext_right = degree as i64;
    let mut targets = Vec::with_capacity(degree);
    let mut left = base_left - 1;
    let mut right = base_right + 1;

    while targets.len() < degree && left >= ext_left && right <= ext_right {
        targets.push(left);
        if targets.len() < degree {
            targets.push(right);
        }
        left -= 1;
        right += 1;
    }
    while targets.len() < degree && left >= ext_left {
        targets.push(left);
        left -= 1;
    }
    while targets.len() < degree && right <= ext_right {
        targets.push(right);
        right += 1;
    }

    Ok(targets)
}

fn spartan_outer_tau_kernel_scale<F: Field>(tau: &[F], uniskip: F) -> Result<F, ProverError> {
    if tau.len() < 2 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 1 tau has {} variables, expected at least 2",
            tau.len()
        )));
    }
    let tau_high = tau[tau.len() - 1];
    centered_lagrange_kernel(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, tau_high, uniskip)
        .map_err(invalid_sumcheck_output)
}

fn spartan_outer_stream_row_weights<F: Field>(uniskip: F) -> Result<(Vec<F>, Vec<F>), ProverError> {
    let at_zero = spartan_outer_row_weights(uniskip, F::zero()).map_err(invalid_sumcheck_output)?;
    let at_one = spartan_outer_row_weights(uniskip, F::one()).map_err(invalid_sumcheck_output)?;
    Ok((at_zero, at_one))
}

fn normalized_remainder_opening_point<F: Field>(remainder_challenges: &[F]) -> Vec<F> {
    let mut point = remainder_challenges[1..].to_vec();
    point.reverse();
    point
}

fn centered_lagrange_integer_coeffs(
    domain_size: usize,
    target: i64,
) -> Result<Vec<i32>, ProverError> {
    let start = centered_domain_start(domain_size).map_err(invalid_sumcheck_output)?;
    (0..domain_size)
        .map(|index| {
            let x_i = start
                + i64::try_from(index).map_err(|_| {
                    invalid_sumcheck_output(format!(
                        "Stage 1 Lagrange index {index} is out of range"
                    ))
                })?;
            let mut numerator = 1i128;
            let mut denominator = 1i128;
            for other in 0..domain_size {
                if other == index {
                    continue;
                }
                let x_j = start
                    + i64::try_from(other).map_err(|_| {
                        invalid_sumcheck_output(format!(
                            "Stage 1 Lagrange index {other} is out of range"
                        ))
                    })?;
                numerator *= i128::from(target - x_j);
                denominator *= i128::from(x_i - x_j);
            }
            if denominator == 0 || numerator % denominator != 0 {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 1 centered Lagrange coefficient for target {target} is non-integral"
                )));
            }
            i32::try_from(numerator / denominator).map_err(|_| {
                invalid_sumcheck_output(format!(
                    "Stage 1 centered Lagrange coefficient for target {target} exceeds i32"
                ))
            })
        })
        .collect()
}

fn checked_trace_rows(log_t: usize, label: &str) -> Result<usize, ProverError> {
    1usize
        .checked_shl(log_t as u32)
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("{label} trace length overflows for log_t={log_t}"),
        })
}

fn value_slot(index: usize) -> Result<BackendValueSlot, ProverError> {
    Ok(BackendValueSlot(u32::try_from(index).map_err(|_| {
        invalid_sumcheck_output(format!(
            "Stage 1 query index {index} exceeds value slot range"
        ))
    })?))
}

fn ordered_linear_product_outputs<F: Field>(
    outputs: Vec<SumcheckLinearProductOutput<F>>,
    expected_count: usize,
) -> Result<Vec<F>, ProverError> {
    if outputs.len() != expected_count {
        return Err(invalid_sumcheck_output(format!(
            "Stage 1 linear product backend returned {} outputs, expected {expected_count}",
            outputs.len()
        )));
    }
    let mut seen = vec![false; expected_count];
    let mut ordered = vec![F::zero(); expected_count];
    for output in outputs {
        let index = usize::try_from(output.slot.0).map_err(|_| {
            invalid_sumcheck_output(format!(
                "Stage 1 linear product output slot {:?} is out of range",
                output.slot
            ))
        })?;
        if index >= expected_count {
            return Err(invalid_sumcheck_output(format!(
                "Stage 1 linear product output slot {:?} exceeds expected count {expected_count}",
                output.slot
            )));
        }
        if seen[index] {
            return Err(invalid_sumcheck_output(format!(
                "duplicate Stage 1 linear product output slot {:?}",
                output.slot
            )));
        }
        seen[index] = true;
        ordered[index] = output.value;
    }
    Ok(ordered)
}
