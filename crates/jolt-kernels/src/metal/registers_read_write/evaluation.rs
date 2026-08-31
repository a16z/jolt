use std::{
    mem::size_of,
    sync::Arc,
    time::{Duration, Instant},
};

use jolt_claims::protocols::jolt::geometry::dimensions::{
    ReadWriteDimensions, REGISTER_ADDRESS_BITS,
};
use jolt_claims::OutputClaims as _;
use jolt_field::{AkitaField, FixedBytes, TranscriptChallenge};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial};
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChallenges, RegistersReadWriteChecking, RegistersReadWriteInputClaims,
};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::metal::solinas::registers_claim_reduction::RegistersClaimResidentRdPlane;
use crate::metal::solinas::{
    DeviceInfo, MetalError, RegistersReadWriteCycleObservation, RegistersReadWriteStage1Source,
    SolinasMetal,
};
use crate::optimized::registers_read_write::{
    AlignedPackedRegisterRows, OptimizedRegistersReadWrite, PackedRegisterCycleRow,
};
use crate::optimized::spartan_outer::prepare_metal_spartan_outer_stage1_owner_witness_rows;
use crate::ProofSession;

const METAL_OPERAND_CLAIMS_LOG_T_MIN: usize = 25;

/// Failure from the fixed CPU/Metal registers read/write evaluator.
#[derive(Debug, thiserror::Error)]
pub enum RegistersReadWriteEvalError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("registers read/write evaluator failed: {0}")]
    Kernel(String),
}

/// Canonical outputs compared between the optimized CPU and Metal arms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistersReadWriteEvalResult {
    round_polynomials: Vec<Vec<AkitaField>>,
    final_claim: AkitaField,
    output_claims: Vec<AkitaField>,
}

impl RegistersReadWriteEvalResult {
    /// FNV-1a over length-delimited canonical field bytes.
    pub fn checksum(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut write = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        write(&(self.round_polynomials.len() as u64).to_le_bytes());
        for polynomial in &self.round_polynomials {
            write(&(polynomial.len() as u64).to_le_bytes());
            for value in polynomial {
                write(&value.to_bytes_array());
            }
        }
        write(&self.final_claim.to_bytes_array());
        write(&(self.output_claims.len() as u64).to_le_bytes());
        for value in &self.output_claims {
            write(&value.to_bytes_array());
        }
        hash
    }

    pub fn rounds(&self) -> usize {
        self.round_polynomials.len()
    }

    pub fn output_claims(&self) -> usize {
        self.output_claims.len()
    }
}

/// One timed sumcheck call. A round after round zero includes the preceding
/// challenge bind, matching the prover engine's call boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersReadWriteRoundTiming {
    pub round: usize,
    pub wall: Duration,
}

/// One device-owned cycle round, including any shrinking-state allocation.
#[derive(Clone, Copy, Debug)]
pub struct RegistersReadWriteMetalCycleTiming {
    pub round: usize,
    pub allocation: Duration,
    pub wall: Duration,
    pub gpu_active: Duration,
    pub prefill_gpu_active: Duration,
    pub live_entries: usize,
    pub resident_bytes: usize,
    pub peak_transition_bytes: usize,
}

/// Exact structural counts of the fixed source distribution. Entry level zero
/// is the unbound cycle table; each following level unions adjacent blocks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistersReadWriteShapeSnapshot {
    pub rs1_reads: usize,
    pub rs2_reads: usize,
    pub writes: usize,
    pub rs1_rs2_same_register: usize,
    pub rd_same_as_read_register: usize,
    pub rd_distinct_from_reads: usize,
    pub rd_distinct_signed_39_overflow: usize,
    pub active_registers: usize,
    pub entries_by_cycle_level: Vec<usize>,
    pub read_entries_by_cycle_level: Vec<usize>,
    pub write_entries_by_cycle_level: Vec<usize>,
    pub value_change_entries_by_cycle_level: Vec<usize>,
    pub packed_source_row_bytes: usize,
    pub indexed_entry_bytes: usize,
    pub direct_entry_bytes: usize,
}

/// Optimized CPU service timings for one registers read/write member.
#[derive(Clone, Debug)]
pub struct RegistersReadWriteCpuEvalSample {
    pub result: RegistersReadWriteEvalResult,
    pub member_wall: Duration,
    pub source_to_state_wall: Duration,
    pub kernel_setup_wall: Duration,
    pub prepare_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
    pub round_timings: Vec<RegistersReadWriteRoundTiming>,
    pub metal_first_message_prepare_wall: Option<Duration>,
    pub metal_first_message_wall: Option<Duration>,
    pub metal_first_message_gpu_active: Option<Duration>,
    pub metal_first_message_threads: Option<usize>,
    pub metal_first_message_execution_width: Option<usize>,
    pub metal_first_message_static_threadgroup_bytes: Option<u64>,
    pub metal_first_message_resident_bytes: Option<usize>,
    pub metal_first_message_source_zero_copy: Option<bool>,
    pub metal_cycle_sequence_prepare_wall: Option<Duration>,
    pub metal_cycle_timings: Vec<RegistersReadWriteMetalCycleTiming>,
    pub metal_cycle_finish_allocation: Option<Duration>,
    pub metal_cycle_finish_wall: Option<Duration>,
    pub metal_cycle_finish_gpu_active: Option<Duration>,
    pub metal_cycle_finish_resident_bytes: Option<usize>,
    pub metal_cycle_peak_transition_bytes: Option<usize>,
    pub metal_operand_claims_prepare_wall: Option<Duration>,
    pub metal_operand_claims_wall: Option<Duration>,
    pub metal_operand_claims_gpu_active: Option<Duration>,
}

/// Metal service timings. Until the relation has a device implementation,
/// this records the production optimized-CPU host route under the same fixed
/// boundary.
pub type RegistersReadWriteMetalEvalSample = RegistersReadWriteCpuEvalSample;

/// Real-witness fixture shared by the isolated optimized CPU and Metal arms.
pub struct RegistersReadWriteCpuMetalEvalFixture {
    context: SolinasMetal,
    rows: Arc<AlignedPackedRegisterRows>,
    stage1_source: Option<RegistersReadWriteStage1Source>,
    stage1_rd_post: Option<RegistersClaimResidentRdPlane>,
    physical_rows: usize,
    log_t: usize,
    r_cycle: Vec<AkitaField>,
    gamma: AkitaField,
    input_values: RegistersReadWriteInputClaims<AkitaField>,
    input_claim: AkitaField,
    challenges: Vec<AkitaField>,
    shape: RegistersReadWriteShapeSnapshot,
    fixture_wall: Duration,
}

impl RegistersReadWriteCpuMetalEvalFixture {
    /// Collects the production register source rows once, outside both arms.
    pub fn new(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        seed: u64,
    ) -> Result<Self, RegistersReadWriteEvalError> {
        Self::new_with_source(witness, log_t, seed, false)
    }

    /// Uses the co-produced Stage-1 buffers as the Metal cycle source while
    /// retaining the packed source as an independent correctness oracle.
    pub fn new_stage1(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        seed: u64,
    ) -> Result<Self, RegistersReadWriteEvalError> {
        Self::new_with_source(witness, log_t, seed, true)
    }

    fn new_with_source(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        seed: u64,
        stage1_source: bool,
    ) -> Result<Self, RegistersReadWriteEvalError> {
        if !(4..=28).contains(&log_t) {
            return Err(RegistersReadWriteEvalError::Kernel(
                "registers read/write evaluator geometry is outside the supported range".to_owned(),
            ));
        }
        let fixture_started = Instant::now();
        let context = SolinasMetal::for_akita_production()?;
        let cycles = 1usize << log_t;
        let access = witness.random_access().ok_or_else(|| {
            RegistersReadWriteEvalError::Kernel(
                "registers read/write evaluator requires a random-access witness".to_owned(),
            )
        })?;
        if access.cycles() < cycles {
            return Err(RegistersReadWriteEvalError::Kernel(
                "registers read/write witness is shorter than the cycle domain".to_owned(),
            ));
        }
        let physical_rows = access.physical_rows().min(cycles);
        let rows = Arc::new(
            AlignedPackedRegisterRows::collect(&access, physical_rows, log_t >= 28)
                .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?,
        );
        let (stage1_source, stage1_rd_post) = if stage1_source {
            let (outer_rows, mut ready) = prepare_metal_spartan_outer_stage1_owner_witness_rows(
                &context, witness, cycles, false, true, false, false, false,
            )
            .map_err(|error| RegistersReadWriteEvalError::Kernel(format!("{error:?}")))?;
            let source = ready.registers_read_write.take().ok_or_else(|| {
                RegistersReadWriteEvalError::Kernel(
                    "Stage-1 preparation returned no register source".to_owned(),
                )
            })?;
            drop(outer_rows);
            if log_t <= 20 {
                validate_stage1_source(&source, &rows)?;
            }
            let rd_post = context
                .prepare_test_registers_claim_resident_rd_plane(cycles, physical_rows, |row| {
                    rows.as_slice()[row].rd_post_value
                })
                .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
            (Some(source), Some(rd_post))
        } else {
            (None, None)
        };
        let shape = source_shape(rows.as_slice());
        let r_cycle = (0..log_t)
            .map(|index| challenge_field(seed ^ 0x3c6e_f372_fe94_f82b ^ index as u64))
            .collect::<Vec<_>>();
        let gamma = challenge_field(seed ^ 0xa54f_f53a_5f1d_36f1);
        let input_values = evaluate_input_values(rows.as_slice(), &r_cycle);
        let relation = RegistersReadWriteChecking::new(ReadWriteDimensions::new(
            log_t,
            REGISTER_ADDRESS_BITS,
            log_t,
            0,
        ));
        let relation_challenges = RegistersReadWriteChallenges { gamma };
        let input_claim = relation
            .input_claim(&input_values, &relation_challenges)
            .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
        let challenges = (0..log_t + REGISTER_ADDRESS_BITS)
            .map(|round| challenge_field(seed ^ 0x9e37_79b9_7f4a_7c15 ^ round as u64))
            .collect();
        Ok(Self {
            context,
            rows,
            stage1_source,
            stage1_rd_post,
            physical_rows,
            log_t,
            r_cycle,
            gamma,
            input_values,
            input_claim,
            challenges,
            shape,
            fixture_wall: fixture_started.elapsed(),
        })
    }

    pub fn log_t(&self) -> usize {
        self.log_t
    }

    pub fn log_k(&self) -> usize {
        REGISTER_ADDRESS_BITS
    }

    pub fn cycles(&self) -> usize {
        1usize << self.log_t
    }

    pub fn physical_rows(&self) -> usize {
        self.physical_rows
    }

    pub fn fixture_wall(&self) -> Duration {
        self.fixture_wall
    }

    pub fn source_bytes(&self) -> usize {
        self.rows.logical_bytes()
    }

    pub fn shape(&self) -> &RegistersReadWriteShapeSnapshot {
        &self.shape
    }

    pub fn device_info(&self) -> DeviceInfo {
        self.context.device_info()
    }

    pub fn run_cpu(&self) -> Result<RegistersReadWriteCpuEvalSample, RegistersReadWriteEvalError> {
        self.run_optimized_host(false)
    }

    pub fn run_metal(
        &self,
    ) -> Result<RegistersReadWriteMetalEvalSample, RegistersReadWriteEvalError> {
        if self.log_t >= 16 {
            self.run_metal_cycle_sequence()
        } else {
            self.run_optimized_host(false)
        }
    }

    /// Primes the co-produced Stage-1 source before executing the unchanged
    /// Metal sequence. This is an evaluator-only residency ablation.
    pub fn run_metal_primed(
        &self,
    ) -> Result<RegistersReadWriteMetalEvalSample, RegistersReadWriteEvalError> {
        let (Some(source), Some(rd_post)) = (&self.stage1_source, &self.stage1_rd_post) else {
            return Err(RegistersReadWriteEvalError::Kernel(
                "resident-source primer requires the Stage-1 evaluator source".to_owned(),
            ));
        };
        let observation = self
            .context
            .submit_registers_read_write_source_primer(source, rd_post)?
            .join()?;
        if observation.pages == 0
            || observation.read_bytes != observation.pages * size_of::<u64>()
            || observation.output_bytes == 0
            || observation.join_wall > observation.total_wall
            || observation.gpu_active.is_zero()
        {
            return Err(RegistersReadWriteEvalError::Kernel(
                "resident-source primer returned invalid telemetry".to_owned(),
            ));
        }
        self.run_metal()
    }

    fn run_metal_cycle_sequence(
        &self,
    ) -> Result<RegistersReadWriteMetalEvalSample, RegistersReadWriteEvalError> {
        let dimensions = ReadWriteDimensions::new(self.log_t, REGISTER_ADDRESS_BITS, self.log_t, 0);
        let relation = RegistersReadWriteChecking::new(dimensions);
        let relation_challenges = RegistersReadWriteChallenges { gamma: self.gamma };
        let input_points = RegistersReadWriteInputClaims {
            rd_write_value: self.r_cycle.clone(),
            rs1_value: self.r_cycle.clone(),
            rs2_value: self.r_cycle.clone(),
        };

        let member_started = Instant::now();
        let use_metal_operand_claims =
            self.stage1_source.is_some() || self.log_t >= METAL_OPERAND_CLAIMS_LOG_T_MIN;
        let sequence_prepare_started = Instant::now();
        let mut sequence = match (&self.stage1_source, &self.stage1_rd_post) {
            (Some(source), Some(rd_post)) => self
                .context
                .prepare_registers_read_write_cycle_sequence_from_stage1(
                    source.clone(),
                    rd_post.clone(),
                    self.log_t,
                    self.gamma,
                )?,
            (None, None) => self.context.prepare_registers_read_write_cycle_sequence(
                Arc::clone(&self.rows),
                self.log_t,
                self.gamma,
            )?,
            _ => {
                return Err(RegistersReadWriteEvalError::Kernel(
                    "Stage-1 source and resident rd-post plane disagree".to_owned(),
                ));
            }
        };
        let sequence_prepare_wall = sequence_prepare_started.elapsed();
        let prepare_wall = sequence_prepare_wall;
        let rounds_started = Instant::now();
        let mut gruen = GruenSplitEqPolynomial::new(&self.r_cycle, BindingOrder::LowToHigh);
        let mut previous_claim = self.input_claim;
        let mut round_polynomials = Vec::with_capacity(self.challenges.len());
        let mut round_timings = Vec::with_capacity(self.challenges.len());
        let mut metal_cycle_timings = Vec::with_capacity(self.log_t);
        let first_round_started = Instant::now();
        let first = sequence.message(gruen.e_in_current(), gruen.e_out_current(), self.gamma)?;
        let polynomial =
            gruen.gruen_poly_deg_3(first.quadratic[0], first.quadratic[1], previous_claim);
        round_timings.push(RegistersReadWriteRoundTiming {
            round: 0,
            wall: first_round_started.elapsed(),
        });
        metal_cycle_timings.push(cycle_timing(0, first));
        previous_claim = polynomial.evaluate(self.challenges[0]);
        round_polynomials.push(polynomial.coefficients().to_vec());

        for round in 1..self.log_t {
            let round_started = Instant::now();
            let bind = self.challenges[round - 1];
            gruen.bind(bind);
            let observation =
                sequence.bind_and_message(bind, gruen.e_in_current(), gruen.e_out_current())?;
            let polynomial = gruen.gruen_poly_deg_3(
                observation.quadratic[0],
                observation.quadratic[1],
                previous_claim,
            );
            round_timings.push(RegistersReadWriteRoundTiming {
                round,
                wall: round_started.elapsed(),
            });
            metal_cycle_timings.push(cycle_timing(round, observation));
            previous_claim = polynomial.evaluate(self.challenges[round]);
            round_polynomials.push(polynomial.coefficients().to_vec());
        }

        let finish = sequence.finish(self.challenges[self.log_t - 1])?;
        let kernel_setup_started = Instant::now();
        let mut kernel = OptimizedRegistersReadWrite::prepare_after_cycle_phase(
            self.log_t,
            REGISTER_ADDRESS_BITS,
            &self.r_cycle,
            if use_metal_operand_claims {
                None
            } else {
                Some(self.rows.as_slice())
            },
            &self.challenges[..self.log_t],
            finish.roots,
            finish.increment,
        )
        .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
        let kernel_setup_wall = kernel_setup_started.elapsed();

        let mut bind = None;
        for round in self.log_t..self.challenges.len() {
            let round_started = Instant::now();
            let polynomial = kernel
                .prove_round(bind, round, previous_claim)
                .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
            round_timings.push(RegistersReadWriteRoundTiming {
                round,
                wall: round_started.elapsed(),
            });
            let challenge = self.challenges[round];
            previous_claim = polynomial.evaluate(challenge);
            round_polynomials.push(polynomial.coefficients().to_vec());
            bind = Some(challenge);
        }
        let rounds_wall = rounds_started.elapsed();

        let final_challenge = self.challenges.last().copied().ok_or_else(|| {
            RegistersReadWriteEvalError::Kernel(
                "registers read/write evaluator has no terminal challenge".to_owned(),
            )
        })?;
        let finish_started = Instant::now();
        kernel
            .finish_rounds(final_challenge)
            .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
        let finish_wall = finish_started.elapsed();

        let output_started = Instant::now();
        let output_points = relation
            .derive_opening_points(&self.challenges, &input_points)
            .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
        let mut output_claims = kernel
            .output_claims(&self.input_values)
            .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
        kernel
            .validate_derived_tables(
                &relation,
                &input_points,
                &output_points,
                &relation_challenges,
            )
            .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
        let metal_operand_claims = if use_metal_operand_claims {
            let cycle_point = self.challenges[..self.log_t]
                .iter()
                .rev()
                .copied()
                .collect::<Vec<_>>();
            let address_point = self.challenges[self.log_t..]
                .iter()
                .rev()
                .copied()
                .collect::<Vec<_>>();
            let observation = sequence.operand_claims(&cycle_point, &address_point)?;
            output_claims.rs1_ra = observation.claims[0];
            output_claims.rs2_ra = observation.claims[1];
            Some(observation)
        } else {
            None
        };
        let output_claims = output_claims.opening_values();
        let output_wall = output_started.elapsed();
        let peak_transition_bytes = metal_cycle_timings
            .iter()
            .map(|timing| timing.peak_transition_bytes)
            .chain(core::iter::once(finish.peak_transition_bytes))
            .max();

        Ok(RegistersReadWriteCpuEvalSample {
            result: RegistersReadWriteEvalResult {
                round_polynomials,
                final_claim: previous_claim,
                output_claims,
            },
            member_wall: member_started.elapsed(),
            source_to_state_wall: Duration::ZERO,
            kernel_setup_wall,
            prepare_wall,
            rounds_wall,
            finish_wall,
            output_wall,
            round_timings,
            metal_first_message_prepare_wall: Some(sequence_prepare_wall),
            metal_first_message_wall: Some(first.wall),
            metal_first_message_gpu_active: Some(first.gpu_active),
            metal_first_message_threads: Some(sequence.threads()),
            metal_first_message_execution_width: Some(sequence.limits().thread_execution_width),
            metal_first_message_static_threadgroup_bytes: Some(
                sequence.limits().static_threadgroup_memory_length,
            ),
            metal_first_message_resident_bytes: Some(first.resident_bytes),
            metal_first_message_source_zero_copy: Some(true),
            metal_cycle_sequence_prepare_wall: Some(sequence_prepare_wall),
            metal_cycle_timings,
            metal_cycle_finish_allocation: Some(finish.allocation),
            metal_cycle_finish_wall: Some(finish.wall),
            metal_cycle_finish_gpu_active: Some(finish.gpu_active),
            metal_cycle_finish_resident_bytes: Some(finish.resident_bytes),
            metal_cycle_peak_transition_bytes: peak_transition_bytes,
            metal_operand_claims_prepare_wall: metal_operand_claims.map(|value| value.prepare),
            metal_operand_claims_wall: metal_operand_claims.map(|value| value.wall),
            metal_operand_claims_gpu_active: metal_operand_claims.map(|value| value.gpu_active),
        })
    }

    fn run_optimized_host(
        &self,
        metal_first_message: bool,
    ) -> Result<RegistersReadWriteCpuEvalSample, RegistersReadWriteEvalError> {
        let dimensions = ReadWriteDimensions::new(self.log_t, REGISTER_ADDRESS_BITS, self.log_t, 0);
        let relation = RegistersReadWriteChecking::new(dimensions);
        let relation_challenges = RegistersReadWriteChallenges { gamma: self.gamma };
        let input_points = RegistersReadWriteInputClaims {
            rd_write_value: self.r_cycle.clone(),
            rs1_value: self.r_cycle.clone(),
            rs2_value: self.r_cycle.clone(),
        };

        let member_started = Instant::now();
        let source_to_state_started = Instant::now();
        let prepared = OptimizedRegistersReadWrite::precompute_packed::<AkitaField>(
            self.rows.as_slice(),
            self.cycles(),
        )
        .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
        let source_to_state_wall = source_to_state_started.elapsed();

        let kernel_setup_started = Instant::now();
        let mut session = ProofSession::default();
        let mut kernel = OptimizedRegistersReadWrite::prepare_precomputed(
            &mut session,
            crate::ProverInputs {
                relation: &relation,
                claims: &self.input_values,
                points: &input_points,
                challenges: &relation_challenges,
            },
            prepared,
        )
        .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
        let kernel_setup_wall = kernel_setup_started.elapsed();
        let metal_gruen = metal_first_message
            .then(|| GruenSplitEqPolynomial::new(&self.r_cycle, BindingOrder::LowToHigh));
        let metal_first_message_prepare_started = Instant::now();
        let metal_first_message_invocation = metal_gruen
            .as_ref()
            .map(|gruen| {
                self.context.prepare_registers_read_write_first_message(
                    self.rows.device_view(),
                    gruen.e_in_current(),
                    gruen.e_out_current(),
                    self.gamma,
                )
            })
            .transpose()?;
        let metal_first_message_prepare_wall = metal_first_message_invocation
            .as_ref()
            .map(|_| metal_first_message_prepare_started.elapsed());
        let prepare_wall = source_to_state_wall
            + kernel_setup_wall
            + metal_first_message_prepare_wall.unwrap_or_default();

        let rounds_started = Instant::now();
        let mut bind = None;
        let mut previous_claim = self.input_claim;
        let mut round_polynomials = Vec::with_capacity(self.challenges.len());
        let mut round_timings = Vec::with_capacity(self.challenges.len());
        let mut metal_first_message_observation = None;
        for (round, &challenge) in self.challenges.iter().enumerate() {
            let round_started = Instant::now();
            let polynomial = if round == 0 {
                if let Some(invocation) = &metal_first_message_invocation {
                    let observation = invocation.execute()?;
                    let gruen = metal_gruen.as_ref().ok_or_else(|| {
                        RegistersReadWriteEvalError::Kernel(
                            "Metal first message lost its Gruen state".to_owned(),
                        )
                    })?;
                    let polynomial = gruen.gruen_poly_deg_3(
                        observation.quadratic[0],
                        observation.quadratic[1],
                        previous_claim,
                    );
                    metal_first_message_observation = Some(observation);
                    polynomial
                } else {
                    kernel
                        .prove_round(bind, round, previous_claim)
                        .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?
                }
            } else {
                kernel
                    .prove_round(bind, round, previous_claim)
                    .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?
            };
            round_timings.push(RegistersReadWriteRoundTiming {
                round,
                wall: round_started.elapsed(),
            });
            previous_claim = polynomial.evaluate(challenge);
            round_polynomials.push(polynomial.coefficients().to_vec());
            bind = Some(challenge);
        }
        let rounds_wall = rounds_started.elapsed();

        let final_challenge = self.challenges.last().copied().ok_or_else(|| {
            RegistersReadWriteEvalError::Kernel(
                "registers read/write evaluator has no terminal challenge".to_owned(),
            )
        })?;
        let finish_started = Instant::now();
        kernel
            .finish_rounds(final_challenge)
            .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
        let finish_wall = finish_started.elapsed();

        let output_started = Instant::now();
        let output_points = relation
            .derive_opening_points(&self.challenges, &input_points)
            .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
        let output_claims = kernel
            .output_claims(&self.input_values)
            .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
        kernel
            .validate_derived_tables(
                &relation,
                &input_points,
                &output_points,
                &relation_challenges,
            )
            .map_err(|error| RegistersReadWriteEvalError::Kernel(error.to_string()))?;
        let output_claims = output_claims.opening_values();
        let output_wall = output_started.elapsed();

        Ok(RegistersReadWriteCpuEvalSample {
            result: RegistersReadWriteEvalResult {
                round_polynomials,
                final_claim: previous_claim,
                output_claims,
            },
            member_wall: member_started.elapsed(),
            source_to_state_wall,
            kernel_setup_wall,
            prepare_wall,
            rounds_wall,
            finish_wall,
            output_wall,
            round_timings,
            metal_first_message_prepare_wall,
            metal_first_message_wall: metal_first_message_observation.map(|value| value.wall),
            metal_first_message_gpu_active: metal_first_message_observation
                .map(|value| value.gpu_active),
            metal_first_message_threads: metal_first_message_observation.map(|value| value.threads),
            metal_first_message_execution_width: metal_first_message_observation
                .map(|value| value.limits.thread_execution_width),
            metal_first_message_static_threadgroup_bytes: metal_first_message_observation
                .map(|value| value.limits.static_threadgroup_memory_length),
            metal_first_message_resident_bytes: metal_first_message_observation
                .map(|value| value.resident_bytes),
            metal_first_message_source_zero_copy: metal_first_message_observation
                .map(|value| value.source_zero_copy),
            metal_cycle_sequence_prepare_wall: None,
            metal_cycle_timings: Vec::new(),
            metal_cycle_finish_allocation: None,
            metal_cycle_finish_wall: None,
            metal_cycle_finish_gpu_active: None,
            metal_cycle_finish_resident_bytes: None,
            metal_cycle_peak_transition_bytes: None,
            metal_operand_claims_prepare_wall: None,
            metal_operand_claims_wall: None,
            metal_operand_claims_gpu_active: None,
        })
    }
}

fn validate_stage1_source(
    source: &RegistersReadWriteStage1Source,
    packed: &AlignedPackedRegisterRows,
) -> Result<(), RegistersReadWriteEvalError> {
    let source_view = source.device_view();
    if source.device_sidecar_bytes() != source_view.cycles {
        return Err(RegistersReadWriteEvalError::Kernel(
            "Stage-1 destination-index sidecar has the wrong length".to_owned(),
        ));
    }
    let packed_view = packed.device_view();
    if source_view.physical_rows != packed_view.rows()
        || source_view.active_registers != packed_view.active_registers()
        || source_view.remap_registers != packed_view.remaps_registers()
        || source_view.register_unmap != packed_view.register_unmap()
    {
        return Err(RegistersReadWriteEvalError::Kernel(
            "Stage-1 and packed register-source metadata disagree".to_owned(),
        ));
    }
    let mut previous_write = [0u64; 128];
    for (row_index, expected) in packed.as_slice().iter().copied().enumerate() {
        let actual = source
            .decode_row(row_index, expected.rd_post_value)
            .ok_or_else(|| {
                RegistersReadWriteEvalError::Kernel(format!(
                    "Stage-1 register source is missing row {row_index}"
                ))
            })?;
        if actual.unpack() != expected.unpack() {
            return Err(RegistersReadWriteEvalError::Kernel(format!(
                "Stage-1 and packed register sources disagree at row {row_index}"
            )));
        }
        if expected.rd_index != u8::MAX {
            let previous = previous_write[usize::from(expected.rd_index)];
            if expected.rd_pre_value != previous {
                return Err(RegistersReadWriteEvalError::Kernel(format!(
                    "register predecessor invariant failed at row {row_index}"
                )));
            }
            previous_write[usize::from(expected.rd_index)] = expected.rd_post_value;
        }
    }
    Ok(())
}

fn cycle_timing(
    round: usize,
    observation: RegistersReadWriteCycleObservation,
) -> RegistersReadWriteMetalCycleTiming {
    RegistersReadWriteMetalCycleTiming {
        round,
        allocation: observation.allocation,
        wall: observation.wall,
        gpu_active: observation.gpu_active,
        prefill_gpu_active: observation.prefill_gpu_active,
        live_entries: observation.live_entries,
        resident_bytes: observation.resident_bytes,
        peak_transition_bytes: observation.peak_transition_bytes,
    }
}

fn evaluate_input_values(
    rows: &[PackedRegisterCycleRow],
    r_cycle: &[AkitaField],
) -> RegistersReadWriteInputClaims<AkitaField> {
    let hi_bits = r_cycle.len().div_ceil(2);
    let low_bits = r_cycle.len() - hi_bits;
    let e_hi = EqPolynomial::<AkitaField>::evals(&r_cycle[..hi_bits], None);
    let e_lo = EqPolynomial::<AkitaField>::evals(&r_cycle[hi_bits..], None);
    let rows_per_block = 1usize << low_bits;
    let fold_block = |high: usize| {
        let start = high * rows_per_block;
        if start >= rows.len() {
            return [AkitaField::zero(); 3];
        }
        let end = (start + rows_per_block).min(rows.len());
        let mut values = [AkitaField::zero(); 3];
        for (low, row) in rows[start..end].iter().enumerate() {
            let weight = e_hi[high] * e_lo[low];
            if row.rd_index != u8::MAX {
                values[0] += weight * AkitaField::from_u64(row.rd_post_value);
            }
            if row.rs1_index != u8::MAX {
                values[1] += weight * AkitaField::from_u64(row.rs1_value);
            }
            if row.rs2_index != u8::MAX {
                values[2] += weight * AkitaField::from_u64(row.rs2_value);
            }
        }
        values
    };
    #[cfg(feature = "parallel")]
    let values = (0..e_hi.len()).into_par_iter().map(fold_block).reduce(
        || [AkitaField::zero(); 3],
        |left, right| [left[0] + right[0], left[1] + right[1], left[2] + right[2]],
    );
    #[cfg(not(feature = "parallel"))]
    let values = (0..e_hi.len())
        .map(fold_block)
        .fold([AkitaField::zero(); 3], |left, right| {
            [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
        });
    RegistersReadWriteInputClaims {
        rd_write_value: values[0],
        rs1_value: values[1],
        rs2_value: values[2],
    }
}

fn source_shape(rows: &[PackedRegisterCycleRow]) -> RegistersReadWriteShapeSnapshot {
    let row_mask = |row: &PackedRegisterCycleRow| {
        let mut mask = 0u128;
        for index in [row.rs1_index, row.rs2_index, row.rd_index] {
            if index != u8::MAX {
                mask |= 1u128 << index;
            }
        }
        mask
    };
    let write_mask = |row: &PackedRegisterCycleRow| {
        if row.rd_index != u8::MAX {
            1u128 << row.rd_index
        } else {
            0
        }
    };
    let read_mask = |row: &PackedRegisterCycleRow| {
        let mut mask = 0u128;
        for index in [row.rs1_index, row.rs2_index] {
            if index != u8::MAX {
                mask |= 1u128 << index;
            }
        }
        mask
    };
    let value_change_mask = |row: &PackedRegisterCycleRow| {
        if row.rd_index != u8::MAX && row.rd_pre_value != row.rd_post_value {
            1u128 << row.rd_index
        } else {
            0
        }
    };
    let row_relationships = |row: &PackedRegisterCycleRow| {
        let rs1_rs2_same = usize::from(
            row.rs1_index != u8::MAX && row.rs2_index != u8::MAX && row.rs1_index == row.rs2_index,
        );
        let rd_matches_read = row.rd_index != u8::MAX
            && (row.rd_index == row.rs1_index || row.rd_index == row.rs2_index);
        let rd_distinct = row.rd_index != u8::MAX && !rd_matches_read;
        let delta = i128::from(row.rd_post_value) - i128::from(row.rd_pre_value);
        let signed_39_min = -(1i128 << 38);
        let signed_39_max = (1i128 << 38) - 1;
        (
            rs1_rs2_same,
            usize::from(rd_matches_read),
            usize::from(rd_distinct),
            usize::from(rd_distinct && !(signed_39_min..=signed_39_max).contains(&delta)),
        )
    };
    #[cfg(feature = "parallel")]
    let (
        rs1_reads,
        rs2_reads,
        writes,
        active_mask,
        rs1_rs2_same_register,
        rd_same_as_read_register,
        rd_distinct_from_reads,
        rd_distinct_signed_39_overflow,
    ) = rows
        .par_iter()
        .map(|row| {
            let relationships = row_relationships(row);
            (
                usize::from(row.rs1_index != u8::MAX),
                usize::from(row.rs2_index != u8::MAX),
                usize::from(row.rd_index != u8::MAX),
                row_mask(row),
                relationships.0,
                relationships.1,
                relationships.2,
                relationships.3,
            )
        })
        .reduce(
            || (0, 0, 0, 0, 0, 0, 0, 0),
            |left, right| {
                (
                    left.0 + right.0,
                    left.1 + right.1,
                    left.2 + right.2,
                    left.3 | right.3,
                    left.4 + right.4,
                    left.5 + right.5,
                    left.6 + right.6,
                    left.7 + right.7,
                )
            },
        );
    #[cfg(not(feature = "parallel"))]
    let (
        rs1_reads,
        rs2_reads,
        writes,
        active_mask,
        rs1_rs2_same_register,
        rd_same_as_read_register,
        rd_distinct_from_reads,
        rd_distinct_signed_39_overflow,
    ) = rows.iter().fold(
        (0, 0, 0, 0, 0, 0, 0, 0),
        |(rs1, rs2, rd, mask, same_reads, rd_matches, rd_distinct, overflows), row| {
            let relationships = row_relationships(row);
            (
                rs1 + usize::from(row.rs1_index != u8::MAX),
                rs2 + usize::from(row.rs2_index != u8::MAX),
                rd + usize::from(row.rd_index != u8::MAX),
                mask | row_mask(row),
                same_reads + relationships.0,
                rd_matches + relationships.1,
                rd_distinct + relationships.2,
                overflows + relationships.3,
            )
        },
    );

    #[cfg(feature = "parallel")]
    let mut masks: Vec<(u128, u128, u128, u128)> = rows
        .par_iter()
        .map(|row| {
            (
                row_mask(row),
                read_mask(row),
                write_mask(row),
                value_change_mask(row),
            )
        })
        .collect();
    #[cfg(not(feature = "parallel"))]
    let mut masks: Vec<(u128, u128, u128, u128)> = rows
        .iter()
        .map(|row| {
            (
                row_mask(row),
                read_mask(row),
                write_mask(row),
                value_change_mask(row),
            )
        })
        .collect();
    let mut entries_by_cycle_level = Vec::new();
    let mut read_entries_by_cycle_level = Vec::new();
    let mut write_entries_by_cycle_level = Vec::new();
    let mut value_change_entries_by_cycle_level = Vec::new();
    loop {
        #[cfg(feature = "parallel")]
        let (entries, read_entries, write_entries, value_change_entries) = masks
            .par_iter()
            .map(|(touches, reads, writes, changes)| {
                (
                    touches.count_ones() as usize,
                    reads.count_ones() as usize,
                    writes.count_ones() as usize,
                    changes.count_ones() as usize,
                )
            })
            .reduce(
                || (0, 0, 0, 0),
                |left, right| {
                    (
                        left.0 + right.0,
                        left.1 + right.1,
                        left.2 + right.2,
                        left.3 + right.3,
                    )
                },
            );
        #[cfg(not(feature = "parallel"))]
        let (entries, read_entries, write_entries, value_change_entries) = masks
            .iter()
            .map(|(touches, reads, writes, changes)| {
                (
                    touches.count_ones() as usize,
                    reads.count_ones() as usize,
                    writes.count_ones() as usize,
                    changes.count_ones() as usize,
                )
            })
            .fold((0, 0, 0, 0), |left, right| {
                (
                    left.0 + right.0,
                    left.1 + right.1,
                    left.2 + right.2,
                    left.3 + right.3,
                )
            });
        entries_by_cycle_level.push(entries);
        read_entries_by_cycle_level.push(read_entries);
        write_entries_by_cycle_level.push(write_entries);
        value_change_entries_by_cycle_level.push(value_change_entries);
        if masks.len() <= 1 {
            break;
        }
        #[cfg(feature = "parallel")]
        let next = masks
            .par_chunks(2)
            .map(|pair| {
                let low = pair[0];
                let high = pair.get(1).copied().unwrap_or((0, 0, 0, 0));
                (
                    low.0 | high.0,
                    low.1 | high.1,
                    low.2 | high.2,
                    low.3 | high.3,
                )
            })
            .collect();
        #[cfg(not(feature = "parallel"))]
        let next = masks
            .chunks(2)
            .map(|pair| {
                let low = pair[0];
                let high = pair.get(1).copied().unwrap_or((0, 0, 0, 0));
                (
                    low.0 | high.0,
                    low.1 | high.1,
                    low.2 | high.2,
                    low.3 | high.3,
                )
            })
            .collect();
        masks = next;
    }
    let (indexed_entry_bytes, direct_entry_bytes) =
        OptimizedRegistersReadWrite::evaluator_entry_sizes::<AkitaField>();
    RegistersReadWriteShapeSnapshot {
        rs1_reads,
        rs2_reads,
        writes,
        rs1_rs2_same_register,
        rd_same_as_read_register,
        rd_distinct_from_reads,
        rd_distinct_signed_39_overflow,
        active_registers: active_mask.count_ones() as usize,
        entries_by_cycle_level,
        read_entries_by_cycle_level,
        write_entries_by_cycle_level,
        value_change_entries_by_cycle_level,
        packed_source_row_bytes: std::mem::size_of::<PackedRegisterCycleRow>(),
        indexed_entry_bytes,
        direct_entry_bytes,
    }
}

fn challenge_field(seed: u64) -> AkitaField {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&splitmix(seed).to_le_bytes());
    bytes[8..].copy_from_slice(&splitmix(seed ^ 0xd1b5_4a32_d192_ed03).to_le_bytes());
    AkitaField::from_challenge_bytes(&bytes)
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimized::registers_read_write::RegisterCycleRow;

    #[test]
    fn packed_source_row_has_device_layout() {
        assert_eq!(std::mem::size_of::<PackedRegisterCycleRow>(), 40);
        let row = RegisterCycleRow {
            rs1: Some((7, 11)),
            rs2: None,
            rd: Some((9, 13, 17)),
        };
        assert_eq!(PackedRegisterCycleRow::pack(row).unpack(), row);
    }
}
