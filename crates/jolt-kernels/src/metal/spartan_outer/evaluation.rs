use std::time::{Duration, Instant};

use jolt_claims::{NoChallenges, OutputClaims as _};
use jolt_field::{AkitaField, FixedBytes, TranscriptChallenge};
use jolt_sumcheck::ProveRounds;
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage1::outer_remainder::{
    outer_remainder_input_values_from_uniskip_output, OuterRemainder, OuterRemainderInputClaims,
};
use jolt_witness::JoltWitnessPlane;

use super::{
    MetalOuterRemainderKernel, MetalOuterResidentMetadata, OuterRemainderGpuActiveBreakdown,
};
use crate::metal::solinas::{
    MetalError, OuterRemainderSequenceConfig, OuterRemainderStorageEvalStats,
    OuterRemainderStorageInitialization, PipelineLimits, SolinasMetal, SpartanOuterUniskipRows,
};
use crate::optimized::spartan_outer::{
    compute_optimized_outer_eval_input_claim, prepare_metal_spartan_outer_witness_rows,
    run_optimized_outer_eval, OptimizedOuterEvalResult, OptimizedOuterEvalSample,
};
use crate::SumcheckKernel as _;

/// Failure from the fixed CPU/Metal Outer evaluator.
#[derive(Debug, thiserror::Error)]
pub enum OuterRemainderEvalError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("Outer evaluator failed: {0}")]
    Kernel(String),
}

/// Canonical outputs compared between the optimized CPU and Metal arms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OuterRemainderEvalResult {
    round_polynomials: Vec<Vec<AkitaField>>,
    final_claim: AkitaField,
    output_claims: Vec<AkitaField>,
}

impl OuterRemainderEvalResult {
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

impl From<OptimizedOuterEvalResult> for OuterRemainderEvalResult {
    fn from(value: OptimizedOuterEvalResult) -> Self {
        Self {
            round_polynomials: value.round_polynomials,
            final_claim: value.final_claim,
            output_claims: value.output_claims,
        }
    }
}

/// Optimized CPU service timings for one Outer member.
#[derive(Clone, Debug)]
pub struct OuterRemainderCpuEvalSample {
    pub result: OuterRemainderEvalResult,
    pub member_wall: Duration,
    pub prepare_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
}

impl From<OptimizedOuterEvalSample> for OuterRemainderCpuEvalSample {
    fn from(value: OptimizedOuterEvalSample) -> Self {
        Self {
            result: value.result.into(),
            member_wall: value.member_wall,
            prepare_wall: value.prepare_wall,
            rounds_wall: value.rounds_wall,
            finish_wall: value.finish_wall,
            output_wall: value.output_wall,
        }
    }
}

/// Resolved threadgroup widths used by an Outer Metal arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OuterRemainderThreadSnapshot {
    pub materialize: usize,
    pub stream_bind: usize,
    pub transition: usize,
    pub opening: usize,
    pub reduction: usize,
    pub registers_claim_build: usize,
    pub registers_claim_reduce: usize,
    pub registers_claim_dot: usize,
}

/// Pipeline limits available from the runtime Metal API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OuterRemainderPipelineSnapshot {
    pub materialize: PipelineLimits,
    pub stream_bind: PipelineLimits,
    pub transition: PipelineLimits,
    pub opening: PipelineLimits,
    pub reduction: PipelineLimits,
    pub registers_claim_build: Option<PipelineLimits>,
    pub registers_claim_reduce: Option<PipelineLimits>,
    pub registers_claim_dot: Option<PipelineLimits>,
    pub threads: OuterRemainderThreadSnapshot,
    pub opening_dynamic_threadgroup_bytes: u64,
}

impl From<OuterRemainderStorageEvalStats> for OuterRemainderPipelineSnapshot {
    fn from(value: OuterRemainderStorageEvalStats) -> Self {
        Self {
            materialize: value.materialize_limits,
            stream_bind: value.stream_bind_limits,
            transition: value.transition_limits,
            opening: value.opening_limits,
            reduction: value.reduction_limits,
            registers_claim_build: value.registers_claim_build_limits,
            registers_claim_reduce: value.registers_claim_reduce_limits,
            registers_claim_dot: value.registers_claim_dot_limits,
            threads: OuterRemainderThreadSnapshot {
                materialize: value.materialize_threads,
                stream_bind: value.stream_bind_threads,
                transition: value.transition_threads,
                opening: value.opening_threads,
                reduction: value.reduction_threads,
                registers_claim_build: value.registers_claim_build_threads,
                registers_claim_reduce: value.registers_claim_reduce_threads,
                registers_claim_dot: value.registers_claim_dot_threads,
            },
            opening_dynamic_threadgroup_bytes: value.opening_dynamic_threadgroup_bytes,
        }
    }
}

/// Metal service timings. `charged_wall` includes setup and the member.
#[derive(Clone, Debug)]
pub struct OuterRemainderMetalEvalSample {
    pub result: OuterRemainderEvalResult,
    pub registers_claim_carrier: bool,
    pub borrowed_state_b: bool,
    pub storage_initialization: OuterRemainderStorageInitialization,
    pub setup_wall: Duration,
    pub member_wall: Duration,
    pub materialize_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
    pub member_gpu_active: Duration,
    pub phase_gpu_active: OuterRemainderGpuActiveBreakdown,
    pub initialization_gpu_active: Duration,
    pub initialized_bytes: u64,
    pub initialization_device_buffers: usize,
    pub storage_owned_bytes: u64,
    pub pipelines: OuterRemainderPipelineSnapshot,
}

impl OuterRemainderMetalEvalSample {
    pub fn charged_wall(&self) -> Duration {
        self.setup_wall
            .checked_add(self.member_wall)
            .unwrap_or(Duration::MAX)
    }
}

/// Real-witness fixture shared by all timed arms.
pub struct OuterRemainderCpuMetalEvalFixture {
    context: SolinasMetal,
    rows: SpartanOuterUniskipRows,
    producer_state_b: Option<metal::Buffer>,
    log_t: usize,
    tau: Vec<AkitaField>,
    uniskip_challenge: AkitaField,
    input_claim: AkitaField,
    challenges: Vec<AkitaField>,
    fixture_wall: Duration,
}

impl OuterRemainderCpuMetalEvalFixture {
    /// Builds the resident rows outside all timed member boundaries.
    pub fn new(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        seed: u64,
        borrow_product_state_b: bool,
    ) -> Result<Self, OuterRemainderEvalError> {
        if !(4..=28).contains(&log_t) {
            return Err(OuterRemainderEvalError::Kernel(
                "log_t must be between 4 and 28".to_owned(),
            ));
        }
        let cycles = 1usize << log_t;
        let fixture_started = Instant::now();
        let context = SolinasMetal::for_akita()?;
        let rows = prepare_metal_spartan_outer_witness_rows(&context, witness, cycles)
            .map_err(|error| OuterRemainderEvalError::Kernel(error.to_string()))?;
        let producer_state_b = borrow_product_state_b
            .then(|| context.prepare_eval_outer_state_b(cycles))
            .transpose()?;
        let uniskip_challenge = challenge_field(seed ^ 0xa54f_f53a_5f1d_36f1);
        let tau = (0..log_t + 2)
            .map(|index| challenge_field(seed ^ 0x3c6e_f372_fe94_f82b ^ index as u64))
            .collect::<Vec<_>>();
        let input_claim =
            compute_optimized_outer_eval_input_claim(witness, log_t, &tau, uniskip_challenge)
                .map_err(OuterRemainderEvalError::Kernel)?;
        if input_claim == AkitaField::zero() {
            return Err(OuterRemainderEvalError::Kernel(
                "production-like evaluator challenges produced a zero input claim".to_owned(),
            ));
        }
        let challenges = (0..=log_t)
            .map(|round| challenge_field(seed ^ 0x9e37_79b9_7f4a_7c15 ^ round as u64))
            .collect();
        Ok(Self {
            context,
            rows,
            producer_state_b,
            log_t,
            tau,
            uniskip_challenge,
            input_claim,
            challenges,
            fixture_wall: fixture_started.elapsed(),
        })
    }

    pub fn log_t(&self) -> usize {
        self.log_t
    }

    pub fn cycles(&self) -> usize {
        1usize << self.log_t
    }

    pub fn fixture_wall(&self) -> Duration {
        self.fixture_wall
    }

    pub fn resident_row_bytes(&self) -> u64 {
        160u64 * self.cycles() as u64
    }

    pub fn producer_state_b_bytes(&self) -> u64 {
        self.producer_state_b
            .as_ref()
            .map_or(0, |buffer| buffer.length())
    }

    pub fn device_info(&self) -> crate::metal::solinas::DeviceInfo {
        self.context.device_info()
    }

    pub fn run_cpu(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<OuterRemainderCpuEvalSample, OuterRemainderEvalError> {
        run_optimized_outer_eval(
            witness,
            self.log_t,
            &self.tau,
            self.uniskip_challenge,
            self.input_claim,
            &self.challenges,
        )
        .map(Into::into)
        .map_err(OuterRemainderEvalError::Kernel)
    }

    pub fn run_metal(
        &self,
        registers_claim_carrier: bool,
        storage_initialization: OuterRemainderStorageInitialization,
    ) -> Result<OuterRemainderMetalEvalSample, OuterRemainderEvalError> {
        let config = OuterRemainderSequenceConfig {
            storage_initialization,
            registers_claim_carrier,
            ..OuterRemainderSequenceConfig::default()
        };
        let setup_started = Instant::now();
        let storage = if let Some(state_b) = &self.producer_state_b {
            self.context
                .prepare_outer_remainder_sequence_storage_borrowing_state_b(
                    self.cycles(),
                    config,
                    state_b.clone(),
                )?
        } else {
            self.context
                .prepare_outer_remainder_sequence_storage(self.cycles(), config)?
        };
        let storage_owned_bytes = storage.owned_bytes();
        let storage_stats = storage.eval_stats()?;
        let setup_wall = setup_started.elapsed();
        let sequence = storage.attach(self.rows.clone())?;
        let metadata = MetalOuterResidentMetadata {
            compact_rows_storage_id: self.rows.instruction_input_allocation_identity(),
            residual_rows_storage_id: self.rows.allocation_identity(),
            device_registry_id: self.rows.device_registry_id(),
            resident_rows: self.cycles(),
        };

        let relation = OuterRemainder::new(
            jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions::rv64(
                self.log_t,
            ),
            self.tau.clone(),
            self.uniskip_challenge,
        );
        let claims = outer_remainder_input_values_from_uniskip_output(self.input_claim);
        let points = OuterRemainderInputClaims::<Vec<AkitaField>>::default();
        let no_challenges = NoChallenges::<AkitaField>::default();

        let member_started = Instant::now();
        let materialize_started = Instant::now();
        let mut kernel = MetalOuterRemainderKernel::from_attached_sequence(
            self.log_t,
            &self.tau,
            self.uniskip_challenge,
            sequence,
            metadata,
            config.cpu_tail_elements,
        )
        .map_err(|error| OuterRemainderEvalError::Kernel(error.to_string()))?;
        let materialize_wall = materialize_started.elapsed();

        let rounds_started = Instant::now();
        let mut bind = None;
        let mut previous_claim = self.input_claim;
        let mut round_polynomials = Vec::with_capacity(self.challenges.len());
        for (round, &challenge) in self.challenges.iter().enumerate() {
            let polynomial = kernel
                .prove_round(bind, round, previous_claim)
                .map_err(|error| OuterRemainderEvalError::Kernel(error.to_string()))?;
            previous_claim = polynomial.evaluate(challenge);
            round_polynomials.push(polynomial.coefficients().to_vec());
            bind = Some(challenge);
        }
        let rounds_wall = rounds_started.elapsed();

        let finish_started = Instant::now();
        let final_challenge = self.challenges.last().copied().ok_or_else(|| {
            OuterRemainderEvalError::Kernel("Outer evaluator has no terminal challenge".to_owned())
        })?;
        kernel
            .finish_rounds(final_challenge)
            .map_err(|error| OuterRemainderEvalError::Kernel(error.to_string()))?;
        let finish_wall = finish_started.elapsed();

        let output_started = Instant::now();
        let output_points = relation
            .derive_opening_points(&self.challenges, &points)
            .map_err(|error| OuterRemainderEvalError::Kernel(error.to_string()))?;
        let output_claims = kernel
            .output_claims(&claims)
            .map_err(|error| OuterRemainderEvalError::Kernel(error.to_string()))?;
        kernel
            .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
            .map_err(|error| OuterRemainderEvalError::Kernel(error.to_string()))?;
        let output_claims = output_claims.opening_values();
        let output_wall = output_started.elapsed();
        let member_wall = member_started.elapsed();
        let phase_gpu_active = kernel.gpu_active_breakdown;
        let member_gpu_active = kernel
            .sequence
            .as_ref()
            .map_or(Duration::ZERO, |sequence| sequence.gpu_active_time());
        if phase_gpu_active.total() != Some(member_gpu_active) {
            return Err(OuterRemainderEvalError::Kernel(
                "Metal phase GPU times do not sum to the member total".to_owned(),
            ));
        }

        Ok(OuterRemainderMetalEvalSample {
            result: OuterRemainderEvalResult {
                round_polynomials,
                final_claim: previous_claim,
                output_claims,
            },
            registers_claim_carrier,
            borrowed_state_b: storage_stats.borrowed_state_b,
            storage_initialization,
            setup_wall,
            member_wall,
            materialize_wall,
            rounds_wall,
            finish_wall,
            output_wall,
            member_gpu_active,
            phase_gpu_active,
            initialization_gpu_active: storage_stats.initialization_gpu_active,
            initialized_bytes: storage_stats.initialized_bytes,
            initialization_device_buffers: storage_stats.initialization_device_buffers,
            storage_owned_bytes,
            pipelines: storage_stats.into(),
        })
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
