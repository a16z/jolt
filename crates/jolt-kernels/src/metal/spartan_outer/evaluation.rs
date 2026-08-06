use std::time::{Duration, Instant};

use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_field::AkitaField;
use jolt_poly::lagrange::centered_lagrange_kernel;
use jolt_poly::{boolean_point_msb, UnivariatePoly};
use jolt_r1cs::constraints::jolt::{spartan_outer_constraints, spartan_outer_row_weights};
use jolt_sumcheck::{
    prove_batch, ClearSumcheckRecorder, ProveRounds, SumcheckProof, SumcheckRecorder as _,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage1::outer_remainder::{
    outer_remainder_input_values_from_uniskip_output, OuterRemainder,
};
use jolt_verifier::stages::stage1::outputs::{
    Stage1BatchInputClaims, Stage1BatchOutputClaims, Stage1BatchSumchecks,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{MetalOuterRemainderKernel, MetalOuterResidentMetadata, OUTER_DOMAIN, OUTER_VARIABLES};
use crate::metal::solinas::{
    MetalError, OuterRemainderDispatchCounts, OuterRemainderSequence, OuterRemainderSequenceConfig,
    PipelineLimits, SolinasMetal, SpartanOuterUniskipRow, SpartanOuterUniskipRows,
};
use crate::{KernelError, SumcheckKernel as _, SumcheckKernelError};

const TRANSCRIPT_LABEL: &[u8] = b"metal-outer-successor-v2";

#[derive(Debug, thiserror::Error)]
pub enum OuterRemainderEvalError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("OuterRemainder protocol evaluation failed: {0}")]
    Protocol(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OuterRemainderEvalResult {
    pub input_claim: AkitaField,
    pub coefficient: AkitaField,
    pub round_polynomials: Vec<Vec<AkitaField>>,
    pub challenges: Vec<AkitaField>,
    pub final_claim: AkitaField,
    pub member_claims: Vec<AkitaField>,
    pub proof: SumcheckProof<AkitaField, AkitaField>,
    pub output_claims: Vec<AkitaField>,
    pub opening_point: Vec<AkitaField>,
    pub expected_final_claim: AkitaField,
    pub transcript_state: [u8; 32],
}

pub struct OuterRemainderEvalSample {
    pub result: OuterRemainderEvalResult,
    pub member_wall: Duration,
    pub member_gpu_active: Duration,
    pub setup_wall: Duration,
    pub setup_gpu_active: Duration,
    pub dispatch_counts: OuterRemainderDispatchCounts,
    pub initialized_bytes: u64,
    pub storage_owned_bytes: u64,
    pub tail_elements: usize,
    pub round_device_buffer_allocations: usize,
    pub pipeline_limits: [PipelineLimits; 5],
}

pub struct OuterRemainderEvalFixture {
    log_t: usize,
    rows: SpartanOuterUniskipRows,
    tau: Vec<AkitaField>,
    uniskip_challenge: AkitaField,
    input_claim: AkitaField,
}

impl OuterRemainderEvalFixture {
    pub fn new(
        context: &SolinasMetal,
        log_t: usize,
        seed: u64,
    ) -> Result<Self, OuterRemainderEvalError> {
        if !(4..=28).contains(&log_t) {
            return Err(MetalError::InvalidOuterRemainderRows(1usize << log_t.min(63)).into());
        }
        let cycles = 1usize << log_t;
        let selected_cycle = cycles - 1;
        let selected_stream = 1;
        let selected_row = synthetic_row(selected_cycle, seed);
        let rows = context.prepare_spartan_outer_uniskip_rows_with_fill(
            cycles,
            |instruction_input, residual| {
                #[cfg(feature = "parallel")]
                instruction_input
                    .par_iter_mut()
                    .zip(residual.par_iter_mut())
                    .enumerate()
                    .for_each(|(index, (instruction_input, residual))| {
                        (*instruction_input, *residual) = synthetic_row(index, seed).split();
                    });
                #[cfg(not(feature = "parallel"))]
                for (index, (instruction_input, residual)) in instruction_input
                    .iter_mut()
                    .zip(residual.iter_mut())
                    .enumerate()
                {
                    (*instruction_input, *residual) = synthetic_row(index, seed).split();
                }
                Ok(())
            },
        )?;
        let selected_index = (selected_cycle << 1) | selected_stream;
        let mut tau = boolean_point_msb::<AkitaField>(log_t + 1, selected_index);
        tau.push(nonzero_field(seed ^ 0x3c6e_f372_fe94_f82b));
        let uniskip_challenge = nonzero_field(seed ^ 0xa54f_f53a_5f1d_36f1);
        let input_claim = selected_input_claim(
            selected_row,
            selected_stream,
            tau[log_t + 1],
            uniskip_challenge,
        )?;
        if input_claim == AkitaField::zero() {
            return Err(protocol_error(
                "deterministic fixture produced a zero input claim",
            ));
        }
        Ok(Self {
            log_t,
            rows,
            tau,
            uniskip_challenge,
            input_claim,
        })
    }

    pub const fn log_t(&self) -> usize {
        self.log_t
    }

    pub const fn cycles(&self) -> usize {
        1usize << self.log_t
    }

    pub fn run(
        &self,
        context: &SolinasMetal,
        config: OuterRemainderSequenceConfig,
    ) -> Result<OuterRemainderEvalSample, OuterRemainderEvalError> {
        if self.cycles() < config.cpu_tail_elements {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "evaluation trace must reach the configured CPU tail",
            )
            .into());
        }

        let setup_started = Instant::now();
        let storage = context.prepare_outer_remainder_sequence_storage(self.cycles(), config)?;
        let initialization = storage.initialization();
        let storage_owned_bytes = storage.owned_bytes();
        let metadata = MetalOuterResidentMetadata {
            compact_rows_storage_id: self.rows.instruction_input_allocation_identity(),
            residual_rows_storage_id: self.rows.allocation_identity(),
            device_registry_id: self.rows.device_registry_id(),
            resident_rows: self.cycles(),
            compact_retained: false,
        };
        let sequence = storage.attach(self.rows.clone())?;
        let pipeline_limits = pipeline_limits(&sequence);
        let setup_wall = setup_started.elapsed();

        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(
                SpartanOuterDimensions::rv64(self.log_t),
                self.tau.clone(),
                self.uniskip_challenge,
            ),
        };
        let mut transcript = Blake2bTranscript::<AkitaField>::new(TRANSCRIPT_LABEL);
        let challenges = sumchecks
            .draw_challenges(&mut transcript)
            .map_err(protocol_error)?;
        let input_points = sumchecks.empty_input_points();
        let inputs = Stage1BatchInputClaims {
            outer_remainder: outer_remainder_input_values_from_uniskip_output(self.input_claim),
        };
        let resolved_input = sumchecks
            .outer_remainder
            .input_claim(&inputs.outer_remainder, &challenges.outer_remainder)
            .map_err(protocol_error)?;
        if resolved_input != self.input_claim {
            return Err(protocol_error(
                "relation input claim disagrees with the fixture",
            ));
        }

        let member_started = Instant::now();
        let mut recorder = ClearSumcheckRecorder::<AkitaField, AkitaField>::new();
        let (prelude, coefficients) = sumchecks
            .begin_batch(&inputs, &challenges, &mut recorder, &mut transcript)
            .map_err(protocol_error)?;
        let kernel = MetalOuterRemainderKernel::from_attached_sequence(
            self.log_t,
            &self.tau,
            self.uniskip_challenge,
            sequence,
            metadata,
            config.cpu_tail_elements,
        )
        .map_err(protocol_error)?;
        let mut member = CapturingRounds::new(kernel);
        let mut rounds: Vec<&mut dyn ProveRounds<AkitaField>> = vec![&mut member];
        let proved = prove_batch(&prelude, &mut rounds, &mut recorder, &mut transcript)
            .map_err(protocol_error)?;
        let output_points = sumchecks
            .derive_opening_points(&proved.challenges, &input_points)
            .map_err(protocol_error)?;
        member
            .inner
            .validate_derived_tables(
                &sumchecks.outer_remainder,
                &input_points.outer_remainder,
                &output_points.outer_remainder,
                &challenges.outer_remainder,
            )
            .map_err(protocol_error)?;
        let member_output_claims = member
            .inner
            .output_claims(&inputs.outer_remainder)
            .map_err(protocol_error)?;
        let output_claims = Stage1BatchOutputClaims {
            outer_remainder: member_output_claims,
        };
        sumchecks
            .validate_output_claims(&output_claims)
            .map_err(protocol_error)?;
        let output_claim_values = sumchecks.opening_values(&output_claims);
        let expected_final_claim = sumchecks
            .expected_final_claim(
                &coefficients,
                &input_points,
                &output_claims,
                &output_points,
                &challenges,
            )
            .map_err(protocol_error)?;
        if expected_final_claim != proved.final_claim {
            return Err(protocol_error(
                "prover final claim disagrees with the relation output fold",
            ));
        }
        let recorded = recorder
            .finish(&output_claim_values, &mut transcript)
            .map_err(protocol_error)?;
        if recorded.committed_witness.is_some() {
            return Err(protocol_error(
                "clear evaluator retained a committed witness",
            ));
        }
        let transcript_state = transcript.state();
        let member_wall = member_started.elapsed();
        let member_gpu_active = member
            .inner
            .completed_gpu_active
            .ok_or_else(|| protocol_error("kernel did not retain completed GPU time"))?;
        let dispatch_counts = member
            .inner
            .completed_dispatch_counts
            .ok_or_else(|| protocol_error("kernel did not retain completed dispatch counts"))?;
        let tail_elements = member
            .inner
            .completed_tail_elements
            .ok_or_else(|| protocol_error("kernel did not retain its CPU-tail boundary"))?;
        let round_device_buffer_allocations = member
            .inner
            .completed_round_device_buffer_allocations
            .ok_or_else(|| protocol_error("kernel did not retain its allocation count"))?;

        verify_clear_twin(
            self.input_claim,
            coefficients.outer_remainder,
            sumchecks.outer_remainder.rounds(),
            sumchecks.outer_remainder.degree(),
            &recorded.proof,
            &proved.challenges,
            proved.final_claim,
            &output_claim_values,
            transcript_state,
        )?;

        Ok(OuterRemainderEvalSample {
            result: OuterRemainderEvalResult {
                input_claim: self.input_claim,
                coefficient: coefficients.outer_remainder,
                round_polynomials: member.round_polynomials,
                challenges: proved.challenges.clone(),
                final_claim: proved.final_claim,
                member_claims: proved.member_claims,
                proof: recorded.proof,
                output_claims: output_claim_values,
                opening_point: proved.challenges.iter().rev().copied().collect(),
                expected_final_claim,
                transcript_state,
            },
            member_wall,
            member_gpu_active,
            setup_wall,
            setup_gpu_active: initialization.gpu_active,
            dispatch_counts,
            initialized_bytes: initialization.bytes,
            storage_owned_bytes,
            tail_elements,
            round_device_buffer_allocations,
            pipeline_limits,
        })
    }
}

struct CapturingRounds {
    inner: MetalOuterRemainderKernel,
    round_polynomials: Vec<Vec<AkitaField>>,
}

impl CapturingRounds {
    fn new(inner: MetalOuterRemainderKernel) -> Self {
        Self {
            inner,
            round_polynomials: Vec::new(),
        }
    }
}

impl ProveRounds<AkitaField> for CapturingRounds {
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, jolt_sumcheck::SumcheckError<AkitaField>> {
        let polynomial = self.inner.prove_round(bind, round, previous_claim)?;
        self.round_polynomials
            .push(polynomial.coefficients().to_vec());
        Ok(polynomial)
    }

    fn finish_rounds(
        &mut self,
        bind: AkitaField,
    ) -> Result<(), jolt_sumcheck::SumcheckError<AkitaField>> {
        self.inner.finish_rounds(bind)
    }
}

fn selected_input_claim(
    row: SpartanOuterUniskipRow,
    stream: usize,
    tau_high: AkitaField,
    uniskip_challenge: AkitaField,
) -> Result<AkitaField, OuterRemainderEvalError> {
    let matrices = spartan_outer_constraints::<AkitaField>();
    let columns = (1..=OUTER_VARIABLES).collect::<Vec<_>>();
    let weights = spartan_outer_row_weights(uniskip_challenge, AkitaField::from_u64(stream as u64))
        .map_err(protocol_error)?;
    let weighted = matrices
        .weighted_columns(&weights, &columns)
        .map_err(protocol_error)?;
    let constants = matrices
        .public_column_contributions(&weights, 0, AkitaField::one())
        .map_err(protocol_error)?;
    let fields = row.spartan_outer_fields::<AkitaField>();
    let mut az = constants.a;
    let mut bz = constants.b;
    for ((a, b), value) in weighted.a.iter().zip(&weighted.b).zip(fields) {
        az += *a * value;
        bz += *b * value;
    }
    let kernel = centered_lagrange_kernel(OUTER_DOMAIN, tau_high, uniskip_challenge)
        .map_err(protocol_error)?;
    Ok(kernel * az * bz)
}

#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the clear verifier boundary"
)]
fn verify_clear_twin(
    input_claim: AkitaField,
    coefficient: AkitaField,
    rounds: usize,
    degree: usize,
    proof: &SumcheckProof<AkitaField, AkitaField>,
    expected_point: &[AkitaField],
    expected_value: AkitaField,
    output_claims: &[AkitaField],
    expected_state: [u8; 32],
) -> Result<(), OuterRemainderEvalError> {
    let mut transcript = Blake2bTranscript::<AkitaField>::new(TRANSCRIPT_LABEL);
    let mut recorder = ClearSumcheckRecorder::<AkitaField, AkitaField>::new();
    recorder.absorb_input_claims(&[input_claim], &mut transcript);
    if transcript.challenge_scalar() != coefficient {
        return Err(protocol_error(
            "clear twin drew a different batch coefficient",
        ));
    }
    let reduction = proof
        .verify_compressed_boolean(rounds, degree, coefficient * input_claim, &mut transcript)
        .map_err(protocol_error)?;
    for claim in output_claims {
        transcript.append_labeled(jolt_sumcheck::OPENING_CLAIM_TRANSCRIPT_LABEL, claim);
    }
    if reduction.point.as_slice() != expected_point
        || reduction.value != expected_value
        || transcript.state() != expected_state
    {
        return Err(protocol_error(
            "clear verifier twin disagrees with the prover",
        ));
    }
    Ok(())
}

fn pipeline_limits(sequence: &OuterRemainderSequence) -> [PipelineLimits; 5] {
    [
        sequence.materialize_pipeline_limits(),
        sequence.stream_bind_pipeline_limits(),
        sequence.transition_pipeline_limits(),
        sequence.opening_pipeline_limits(),
        sequence.reduction_pipeline_limits(),
    ]
}

fn synthetic_row(index: usize, seed: u64) -> SpartanOuterUniskipRow {
    let mut words = [0u64; 20];
    for (word, value) in words[..19].iter_mut().enumerate() {
        *value = splitmix(seed ^ index as u64 ^ (word as u64).wrapping_mul(0x1000_0001));
    }
    words[2] &= (1 << 24) - 1;
    words[4] &= (1 << 24) - 1;
    words[8] = 0;
    words[15] &= (1 << 24) - 1;
    let selector = splitmix(seed ^ index as u64 ^ 0xa5a5_5a5a);
    let mut flags = 0u64;
    match selector % 3 {
        1 => flags |= 1 << 0,
        2 => flags |= 1 << 1,
        _ => {}
    }
    match (selector >> 2) % 4 {
        1 => flags |= 1 << 2,
        2 => flags |= 1 << 3,
        3 => flags |= 1 << 4,
        _ => {}
    }
    for bit in 5..=24 {
        flags |= ((selector >> (bit + 7)) & 1) << bit;
    }
    words[19] = flags;
    SpartanOuterUniskipRow::from_words(words)
}

fn nonzero_field(seed: u64) -> AkitaField {
    AkitaField::from_u64((splitmix(seed) & ((1u64 << 56) - 1)) | 1)
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn protocol_error(error: impl ToString) -> OuterRemainderEvalError {
    OuterRemainderEvalError::Protocol(error.to_string())
}

impl From<KernelError<AkitaField>> for OuterRemainderEvalError {
    fn from(error: KernelError<AkitaField>) -> Self {
        protocol_error(error)
    }
}

impl From<SumcheckKernelError<AkitaField>> for OuterRemainderEvalError {
    fn from(error: SumcheckKernelError<AkitaField>) -> Self {
        protocol_error(error)
    }
}
