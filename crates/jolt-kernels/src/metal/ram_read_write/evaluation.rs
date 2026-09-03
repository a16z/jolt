use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use jolt_claims::protocols::jolt::geometry::ram::ram_val_final;
use jolt_claims::OutputClaims as _;
use jolt_field::Zero as _;
use jolt_field::{
    CanonicalBytes as _, CanonicalEncoding as _, Prime128OffsetA7F7 as AkitaField, Ring as _,
};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial};
use jolt_sumcheck::ProveRounds;
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage2::ram_read_write_checking::{
    RamReadWriteChallenges, RamReadWriteChecking, RamReadWriteInputClaims,
};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{MetalPhase, MetalRamReadWriteKernel};
use crate::metal::ram_records::{RamAccessColumns, RamAccessValues, NO_ACCESS};
use crate::metal::solinas::{
    MetalError, RamReadWriteDispatchTiming, RamReadWritePreparationTiming, SolinasMetal,
};
use crate::optimized::ram_read_write::{
    run_optimized_ram_read_write_eval, OptimizedRamReadWriteEvalInputs,
    OptimizedRamReadWriteEvalResult, OptimizedRamReadWriteEvalSample,
};
use crate::ram_access::RamAccessTape;
use crate::reference::views::dense_view;
use crate::{ProofSession, SumcheckKernel as _};

/// Failure from the fixed CPU/Metal RAM read/write evaluator.
#[derive(Debug, thiserror::Error)]
pub enum RamReadWriteEvalError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("RAM read/write evaluator failed: {0}")]
    Kernel(String),
}

/// Canonical outputs compared between the optimized CPU and Metal arms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamReadWriteEvalResult {
    round_polynomials: Vec<Vec<AkitaField>>,
    final_claim: AkitaField,
    output_claims: Vec<AkitaField>,
}

impl RamReadWriteEvalResult {
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
                write(&value.to_bytes_le_vec());
            }
        }
        write(&self.final_claim.to_bytes_le_vec());
        write(&(self.output_claims.len() as u64).to_le_bytes());
        for value in &self.output_claims {
            write(&value.to_bytes_le_vec());
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

impl From<OptimizedRamReadWriteEvalResult> for RamReadWriteEvalResult {
    fn from(value: OptimizedRamReadWriteEvalResult) -> Self {
        Self {
            round_polynomials: value.round_polynomials,
            final_claim: value.final_claim,
            output_claims: value.output_claims,
        }
    }
}

/// Optimized CPU service timings for one RAM read/write member.
#[derive(Clone, Debug)]
pub struct RamReadWriteCpuEvalSample {
    pub result: RamReadWriteEvalResult,
    pub member_wall: Duration,
    pub prepare_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
}

impl From<OptimizedRamReadWriteEvalSample> for RamReadWriteCpuEvalSample {
    fn from(value: OptimizedRamReadWriteEvalSample) -> Self {
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

/// Input distribution and resident storage for one Metal sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamReadWriteBucketSnapshot {
    pub accesses: usize,
    pub active_addresses: usize,
    pub maximum_segment: usize,
    pub p50_segment: usize,
    pub p95_segment: usize,
    pub p99_segment: usize,
    pub hot_addresses: usize,
    pub hot_message_chunks: usize,
    pub hot_state_entries: usize,
    pub hot_compaction_threads: usize,
    pub hot_compaction_threadgroup_bytes: u64,
    pub hot_auxiliary_bytes: usize,
    pub address_bytes: usize,
    pub cycle_bytes: usize,
    pub resident_bytes: usize,
}

/// Wall-time decomposition inside sequence construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamReadWritePreparationSnapshot {
    pub bucket_plan: Duration,
    pub allocation: Duration,
    pub initialization_and_scatter: Duration,
    pub pipeline_setup: Duration,
    pub sequence_total: Duration,
}

/// GPU timestamp attribution between the sequence's dispatch boundaries.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamReadWriteDispatchSnapshot {
    pub address: Duration,
    pub hot_count: Duration,
    pub hot_prefix: Duration,
    pub hot_scatter: Duration,
    pub hot_message: Duration,
    pub cycle: Duration,
    pub reductions: Duration,
}

impl From<RamReadWriteDispatchTiming> for RamReadWriteDispatchSnapshot {
    fn from(value: RamReadWriteDispatchTiming) -> Self {
        Self {
            address: value.address,
            hot_count: value.hot_count,
            hot_prefix: value.hot_prefix,
            hot_scatter: value.hot_scatter,
            hot_message: value.hot_message,
            cycle: value.cycle,
            reductions: value.reductions,
        }
    }
}

impl From<RamReadWritePreparationTiming> for RamReadWritePreparationSnapshot {
    fn from(value: RamReadWritePreparationTiming) -> Self {
        Self {
            bucket_plan: value.bucket_plan,
            allocation: value.allocation,
            initialization_and_scatter: value.initialization_and_scatter,
            pipeline_setup: value.pipeline_setup,
            sequence_total: value.total,
        }
    }
}

/// One timed sumcheck call and the Metal work it completed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamReadWriteRoundTiming {
    pub round: usize,
    pub wall: Duration,
    pub sequence_wall: Duration,
    pub gpu_active: Duration,
    pub dispatch: Option<RamReadWriteDispatchSnapshot>,
}

/// Metal service timings. Source-column collection is excluded symmetrically.
#[derive(Clone, Debug)]
pub struct RamReadWriteMetalEvalSample {
    pub result: RamReadWriteEvalResult,
    pub member_wall: Duration,
    pub prepare_wall: Duration,
    pub final_memory_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
    pub cycle_sequence_wall: Duration,
    pub cycle_sequence_gpu_active: Duration,
    pub dispatch: Option<RamReadWriteDispatchSnapshot>,
    pub preparation: RamReadWritePreparationSnapshot,
    pub buckets: RamReadWriteBucketSnapshot,
    pub round_timings: Vec<RamReadWriteRoundTiming>,
}

/// Real-witness fixture shared by the isolated optimized CPU and Metal arms.
pub struct RamReadWriteCpuMetalEvalFixture {
    context: SolinasMetal,
    columns: Arc<RamAccessColumns>,
    values: RamAccessValues,
    tape: RamAccessTape,
    log_t: usize,
    log_k: usize,
    tau_low: Vec<AkitaField>,
    gamma: AkitaField,
    input_values: RamReadWriteInputClaims<AkitaField>,
    input_claim: AkitaField,
    challenges: Vec<AkitaField>,
    fixture_wall: Duration,
}

impl RamReadWriteCpuMetalEvalFixture {
    /// Collects the production witness source columns once, outside both arms.
    pub fn new(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        log_k: usize,
        seed: u64,
    ) -> Result<Self, RamReadWriteEvalError> {
        if !(4..=28).contains(&log_t) || log_k > 32 {
            return Err(RamReadWriteEvalError::Kernel(
                "RAM read/write evaluator geometry is outside the supported range".to_owned(),
            ));
        }
        let fixture_started = Instant::now();
        let context = SolinasMetal::for_akita_production()?;
        let address_count = 1usize << log_k;
        let mut session = ProofSession::default();
        let columns = RamAccessColumns::shared(&mut session, witness, log_t)
            .map_err(|error| RamReadWriteEvalError::Kernel(error.to_string()))?;
        columns
            .validate_addresses::<AkitaField>(address_count)
            .map_err(|error| RamReadWriteEvalError::Kernel(error.to_string()))?;
        let values = session.take::<RamAccessValues>().ok_or_else(|| {
            RamReadWriteEvalError::Kernel(
                "RAM access collection did not publish its value columns".to_owned(),
            )
        })?;
        let tape = session.take::<RamAccessTape>().ok_or_else(|| {
            RamReadWriteEvalError::Kernel(
                "RAM access collection did not publish its certificate".to_owned(),
            )
        })?;
        tape.validate(log_t, address_count)
            .map_err(|error| RamReadWriteEvalError::Kernel(error.to_string()))?;
        if !tape.increment_compatible() || !tape.ram_ra_compatible() || !tape.hamming_exact() {
            return Err(RamReadWriteEvalError::Kernel(
                "RAM witness is not eligible for the production Metal route".to_owned(),
            ));
        }
        let tau_low = (0..log_t)
            .map(|index| challenge_field(seed ^ 0x3c6e_f372_fe94_f82b ^ index as u64))
            .collect::<Vec<_>>();
        let gamma = challenge_field(seed ^ 0xa54f_f53a_5f1d_36f1);
        let (ram_read_value, ram_write_value) =
            evaluate_access_values(&columns.addresses, &values, &tau_low)?;
        let input_values = RamReadWriteInputClaims {
            ram_read_value,
            ram_write_value,
        };
        let dimensions =
            jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions::new(
                log_t, log_k, log_t, log_k,
            );
        let relation = RamReadWriteChecking::new(dimensions, log_k, tau_low.clone());
        let relation_challenges = RamReadWriteChallenges { gamma };
        let input_claim = relation
            .input_claim(&input_values, &relation_challenges)
            .map_err(|error| RamReadWriteEvalError::Kernel(error.to_string()))?;
        let challenges = (0..log_t + log_k)
            .map(|round| challenge_field(seed ^ 0x9e37_79b9_7f4a_7c15 ^ round as u64))
            .collect();
        Ok(Self {
            context,
            columns,
            values,
            tape,
            log_t,
            log_k,
            tau_low,
            gamma,
            input_values,
            input_claim,
            challenges,
            fixture_wall: fixture_started.elapsed(),
        })
    }

    pub fn log_t(&self) -> usize {
        self.log_t
    }

    pub fn log_k(&self) -> usize {
        self.log_k
    }

    pub fn cycles(&self) -> usize {
        1usize << self.log_t
    }

    pub fn addresses(&self) -> usize {
        1usize << self.log_k
    }

    pub fn access_count(&self) -> usize {
        self.tape.access_count()
    }

    pub fn fixture_wall(&self) -> Duration {
        self.fixture_wall
    }

    pub fn source_bytes(&self) -> usize {
        self.columns.addresses.capacity() * std::mem::size_of::<u32>()
            + self.values.pre_values.capacity() * std::mem::size_of::<u64>()
            + self.values.post_values.capacity() * std::mem::size_of::<u64>()
    }

    pub fn device_info(&self) -> crate::metal::solinas::DeviceInfo {
        self.context.device_info()
    }

    pub fn run_cpu(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<RamReadWriteCpuEvalSample, RamReadWriteEvalError> {
        run_optimized_ram_read_write_eval(OptimizedRamReadWriteEvalInputs {
            witness,
            log_t: self.log_t,
            log_k: self.log_k,
            tau_low: &self.tau_low,
            gamma: self.gamma,
            input_values: &self.input_values,
            input_claim: self.input_claim,
            challenges: &self.challenges,
        })
        .map(Into::into)
        .map_err(RamReadWriteEvalError::Kernel)
    }

    pub fn run_metal(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<RamReadWriteMetalEvalSample, RamReadWriteEvalError> {
        self.run_metal_inner(witness, false, None)
    }

    pub fn run_metal_profiled(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<RamReadWriteMetalEvalSample, RamReadWriteEvalError> {
        self.run_metal_inner(witness, true, None)
    }

    pub fn run_metal_with_hot_threshold(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        hot_segment_threshold: usize,
        profile_dispatches: bool,
    ) -> Result<RamReadWriteMetalEvalSample, RamReadWriteEvalError> {
        self.run_metal_inner(witness, profile_dispatches, Some(hot_segment_threshold))
    }

    #[expect(
        clippy::unchecked_time_subtraction,
        reason = "cumulative sequence timers only increase during each measured round"
    )]
    fn run_metal_inner(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        profile_dispatches: bool,
        hot_segment_threshold: Option<usize>,
    ) -> Result<RamReadWriteMetalEvalSample, RamReadWriteEvalError> {
        let address_count = self.addresses();
        let values = &self.values;
        let member_started = Instant::now();
        let prepare_started = Instant::now();
        let mut sequence = match hot_segment_threshold {
            Some(threshold) => self
                .context
                .prepare_ram_read_write_sequence_with_hot_threshold(
                    &self.columns.addresses,
                    &values.pre_values,
                    &values.post_values,
                    self.log_t,
                    address_count,
                    threshold,
                )?,
            None => self.context.prepare_ram_read_write_sequence(
                &self.columns.addresses,
                &values.pre_values,
                &values.post_values,
                self.log_t,
                address_count,
            )?,
        };
        if profile_dispatches {
            sequence.enable_dispatch_timing()?;
        }
        let stats = sequence.bucket_stats();
        let preparation = sequence.preparation_timing().into();
        let resident_bytes = sequence.resident_bytes();

        let final_memory_started = Instant::now();
        let mut initial_memory = dense_view::<AkitaField>(witness, ram_val_final())
            .map_err(|error| RamReadWriteEvalError::Kernel(error.to_string()))?
            .into_iter()
            .map(|value| {
                value
                    .to_u128_checked()
                    .and_then(|value| u64::try_from(value).ok())
                    .ok_or_else(|| {
                        RamReadWriteEvalError::Kernel(
                            "RAM final memory is not canonically representable as u64".to_owned(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if initial_memory.len() != address_count {
            return Err(RamReadWriteEvalError::Kernel(format!(
                "RAM final memory length mismatch: expected {address_count}, got {}",
                initial_memory.len()
            )));
        }
        sequence.apply_initial_memory(&mut initial_memory)?;
        let val_init = Polynomial::new(
            initial_memory
                .into_iter()
                .map(AkitaField::from_u64)
                .collect(),
        );
        let final_memory_wall = final_memory_started.elapsed();
        let prepare_wall = prepare_started.elapsed();

        let mut kernel = MetalRamReadWriteKernel {
            phase: Some(MetalPhase::Cycle {
                sequence: Box::new(sequence),
                gruen: GruenSplitEqPolynomial::new(&self.tau_low, BindingOrder::LowToHigh),
            }),
            cycle_tail: None,
            val_init,
            gamma: self.gamma,
            log_t: self.log_t,
            log_k: self.log_k,
            access_count: self.access_count(),
            cycle_sequence_wall: Duration::ZERO,
            cycle_sequence_gpu_active: Duration::ZERO,
            cycle_dispatch_timing: RamReadWriteDispatchTiming::default(),
        };
        let dimensions =
            jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions::new(
                self.log_t, self.log_k, self.log_t, self.log_k,
            );
        let relation = RamReadWriteChecking::new(dimensions, self.log_k, self.tau_low.clone());
        let relation_challenges = RamReadWriteChallenges { gamma: self.gamma };
        let input_points = RamReadWriteInputClaims::<Vec<AkitaField>>::default();

        let rounds_started = Instant::now();
        let mut bind = None;
        let mut previous_claim = self.input_claim;
        let mut round_polynomials = Vec::with_capacity(self.challenges.len());
        let mut round_timings = Vec::with_capacity(self.challenges.len());
        for (round, &challenge) in self.challenges.iter().enumerate() {
            let sequence_wall_before = kernel.cycle_sequence_wall;
            let gpu_before = kernel.cycle_sequence_gpu_active;
            let dispatch_before = kernel.cycle_dispatch_timing;
            let round_started = Instant::now();
            let polynomial = kernel
                .prove_round(bind, round, previous_claim)
                .map_err(|error| RamReadWriteEvalError::Kernel(error.to_string()))?;
            let round_wall = round_started.elapsed();
            previous_claim = polynomial.evaluate(challenge);
            round_polynomials.push(polynomial.coefficients().to_vec());
            round_timings.push(RamReadWriteRoundTiming {
                round,
                wall: round_wall,
                sequence_wall: kernel.cycle_sequence_wall - sequence_wall_before,
                gpu_active: kernel.cycle_sequence_gpu_active - gpu_before,
                dispatch: profile_dispatches
                    .then(|| dispatch_delta(kernel.cycle_dispatch_timing, dispatch_before).into()),
            });
            bind = Some(challenge);
        }
        let rounds_wall = rounds_started.elapsed();

        let final_challenge = self.challenges.last().copied().ok_or_else(|| {
            RamReadWriteEvalError::Kernel(
                "RAM read/write evaluator has no terminal challenge".to_owned(),
            )
        })?;
        let finish_started = Instant::now();
        kernel
            .finish_rounds(final_challenge)
            .map_err(|error| RamReadWriteEvalError::Kernel(error.to_string()))?;
        let finish_wall = finish_started.elapsed();

        let output_started = Instant::now();
        let output_points = relation
            .derive_opening_points(&self.challenges, &input_points)
            .map_err(|error| RamReadWriteEvalError::Kernel(error.to_string()))?;
        let output_claims = kernel
            .output_claims(&self.input_values)
            .map_err(|error| RamReadWriteEvalError::Kernel(error.to_string()))?;
        kernel
            .validate_derived_tables(
                &relation,
                &input_points,
                &output_points,
                &relation_challenges,
            )
            .map_err(|error| RamReadWriteEvalError::Kernel(error.to_string()))?;
        let output_claims = output_claims.opening_values();
        let output_wall = output_started.elapsed();
        let member_wall = member_started.elapsed();

        Ok(RamReadWriteMetalEvalSample {
            result: RamReadWriteEvalResult {
                round_polynomials,
                final_claim: previous_claim,
                output_claims,
            },
            member_wall,
            prepare_wall,
            final_memory_wall,
            rounds_wall,
            finish_wall,
            output_wall,
            cycle_sequence_wall: kernel.cycle_sequence_wall,
            cycle_sequence_gpu_active: kernel.cycle_sequence_gpu_active,
            dispatch: profile_dispatches.then(|| kernel.cycle_dispatch_timing.into()),
            preparation,
            buckets: RamReadWriteBucketSnapshot {
                accesses: stats.accesses,
                active_addresses: stats.active_addresses,
                maximum_segment: stats.maximum_segment,
                p50_segment: stats.p50_segment,
                p95_segment: stats.p95_segment,
                p99_segment: stats.p99_segment,
                hot_addresses: stats.hot_addresses,
                hot_message_chunks: stats.hot_message_chunks,
                hot_state_entries: stats.hot_state_entries,
                hot_compaction_threads: stats.hot_compaction_threads,
                hot_compaction_threadgroup_bytes: stats.hot_compaction_threadgroup_bytes,
                hot_auxiliary_bytes: stats.hot_auxiliary_bytes,
                address_bytes: stats.address_bytes,
                cycle_bytes: stats.cycle_bytes,
                resident_bytes,
            },
            round_timings,
        })
    }
}

fn evaluate_access_values(
    addresses: &[u32],
    values: &RamAccessValues,
    point: &[AkitaField],
) -> Result<(AkitaField, AkitaField), RamReadWriteEvalError> {
    let expected = 1usize << point.len();
    if addresses.len() != expected
        || values.pre_values.len() != expected
        || values.post_values.len() != expected
    {
        return Err(RamReadWriteEvalError::Kernel(
            "RAM access columns do not match the evaluation point".to_owned(),
        ));
    }
    let split = point.len() / 2;
    let e_out = EqPolynomial::<AkitaField>::evals(&point[..split], None);
    let e_in = EqPolynomial::<AkitaField>::evals(&point[split..], None);
    let in_bits = e_in.len().trailing_zeros() as usize;
    let fold_outer = |out: usize| {
        let base = out << in_bits;
        let mut read = AkitaField::zero();
        let mut write = AkitaField::zero();
        for (inner, &inner_weight) in e_in.iter().enumerate() {
            let row = base | inner;
            if addresses[row] == NO_ACCESS {
                continue;
            }
            let weight = e_out[out] * inner_weight;
            read += weight * AkitaField::from_u64(values.pre_values[row]);
            write += weight * AkitaField::from_u64(values.post_values[row]);
        }
        (read, write)
    };
    #[cfg(feature = "parallel")]
    {
        Ok((0..e_out.len()).into_par_iter().map(fold_outer).reduce(
            || (AkitaField::zero(), AkitaField::zero()),
            |left, right| (left.0 + right.0, left.1 + right.1),
        ))
    }
    #[cfg(not(feature = "parallel"))]
    {
        Ok((0..e_out.len())
            .map(fold_outer)
            .fold((AkitaField::zero(), AkitaField::zero()), |left, right| {
                (left.0 + right.0, left.1 + right.1)
            }))
    }
}

fn challenge_field(seed: u64) -> AkitaField {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&splitmix(seed).to_le_bytes());
    bytes[8..].copy_from_slice(&splitmix(seed ^ 0xd1b5_4a32_d192_ed03).to_le_bytes());
    AkitaField::from_bytes_le_reduced(&bytes)
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn dispatch_delta(
    after: RamReadWriteDispatchTiming,
    before: RamReadWriteDispatchTiming,
) -> RamReadWriteDispatchTiming {
    let delta = |after: Duration, before: Duration| after.saturating_sub(before);
    RamReadWriteDispatchTiming {
        address: delta(after.address, before.address),
        hot_count: delta(after.hot_count, before.hot_count),
        hot_prefix: delta(after.hot_prefix, before.hot_prefix),
        hot_scatter: delta(after.hot_scatter, before.hot_scatter),
        hot_message: delta(after.hot_message, before.hot_message),
        cycle: delta(after.cycle, before.cycle),
        reductions: delta(after.reductions, before.reductions),
    }
}
