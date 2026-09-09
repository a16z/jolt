use std::{slice, time::Duration, time::Instant};

use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::OutputClaims as _;
use jolt_field::{
    Accumulator as _, CanonicalBytes as _, CanonicalEncoding as _,
    Prime128OffsetA7F7 as AkitaField, WithAccumulator,
};
use jolt_poly::EqPlusOnePrefixSuffix;
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage3::spartan_shift::{
    SpartanShift, SpartanShiftChallenges, SpartanShiftInputClaims,
};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::super::backend::{MetalBackend, MetalConfig};
use super::super::solinas::spartan_shift::{SpartanShiftPlan, SpartanShiftResidentRows};
use super::super::spartan_dense::SpartanDenseResidentOwner;
use crate::optimized::spartan_outer::prepare_metal_spartan_outer_shift_witness_rows;
use crate::optimized::spartan_shift::OptimizedSpartanShift;
use crate::{PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

#[derive(Debug, thiserror::Error)]
pub enum SpartanShiftEvalError {
    #[error("Spartan shift evaluator failed: {0}")]
    Kernel(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpartanShiftEvalResult {
    round_polynomials: Vec<Vec<AkitaField>>,
    final_claim: AkitaField,
    output_claims: Vec<AkitaField>,
}

impl SpartanShiftEvalResult {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanShiftRoundTiming {
    pub round: usize,
    pub wall: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpartanShiftShapeSnapshot {
    pub cycles: usize,
    pub prefix_elements: usize,
    pub suffix_elements: usize,
    pub resident_source_bytes: usize,
    pub native_value_bytes: usize,
    pub native_flag_bytes: usize,
    pub partial_bytes: usize,
    pub q_bytes: usize,
    pub dense_output_bytes: usize,
    pub total_resident_bytes: usize,
    pub build_unique_bytes: usize,
    pub build_coalesced_bytes_with_halo: usize,
    pub fold_unique_bytes: usize,
    pub readback_bytes: usize,
    pub mixed_full_products: usize,
    pub mixed_half_products: usize,
    pub fold_half_products: usize,
    pub command_buffers: usize,
    pub dispatches: usize,
    pub build_threads_per_threadgroup: usize,
    pub high_tile_elements: usize,
    pub fold_threads_per_threadgroup: usize,
    pub production_trace_cutoff_elements: usize,
    pub forced_device_below_production_cutoff: bool,
    pub unexpanded_pc_max: u64,
    pub pc_max: u64,
    pub unexpanded_pc_above_u32: usize,
    pub pc_above_u32: usize,
    pub native_width_census_wall: Duration,
    pub calibration_prefix_gpu_active: Duration,
    pub calibration_fold_gpu_active: Duration,
}

#[derive(Clone, Debug)]
pub struct SpartanShiftEvalSample {
    pub result: SpartanShiftEvalResult,
    pub member_wall: Duration,
    pub prepare_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
    pub round_timings: Vec<SpartanShiftRoundTiming>,
}

pub struct SpartanShiftCpuMetalEvalFixture {
    backend: MetalBackend,
    resident_rows: SpartanShiftResidentRows,
    relation: SpartanShift<AkitaField>,
    claims: SpartanShiftInputClaims<AkitaField>,
    points: SpartanShiftInputClaims<Vec<AkitaField>>,
    relation_challenges: SpartanShiftChallenges<AkitaField>,
    sumcheck_challenges: Vec<AkitaField>,
    initial_claim: AkitaField,
    shape: SpartanShiftShapeSnapshot,
    fixture_wall: Duration,
}

impl SpartanShiftCpuMetalEvalFixture {
    pub fn new(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        seed: u64,
        build_threads_per_threadgroup: Option<usize>,
        high_tile_elements: Option<usize>,
        fold_threads_per_threadgroup: Option<usize>,
    ) -> Result<Self, SpartanShiftEvalError> {
        if !(4..=28).contains(&log_t) {
            return Err(eval_error(
                "Spartan shift evaluator geometry is outside the supported range",
            ));
        }
        let fixture_started = Instant::now();
        let cycles = 1usize << log_t;
        let r_outer = challenge_sequence(seed ^ 0x07e5_12a4_5f6b_c891, log_t);
        let r_product = challenge_sequence(seed ^ 0xface_3141_5926_5358, log_t);
        let gamma = challenge_field(seed ^ 0x5eed_0f0f_1234_5678);
        let relation = SpartanShift::new(
            TraceDimensions::new(log_t),
            r_outer.clone(),
            r_product.clone(),
        );
        let relation_challenges = SpartanShiftChallenges { gamma };
        let sumcheck_challenges = challenge_sequence(seed ^ 0x9e37_79b9_7f4a_7c15, log_t);

        let mut metal_config = MetalConfig::production();
        let production_trace_cutoff_elements = metal_config.spartan_shift.trace_cutoff_elements;
        let forced_device_below_production_cutoff = cycles < production_trace_cutoff_elements;
        if forced_device_below_production_cutoff {
            metal_config.spartan_shift.trace_cutoff_elements = 2;
        }
        if let Some(value) = build_threads_per_threadgroup {
            metal_config
                .spartan_shift
                .dispatch
                .build_threads_per_threadgroup = value;
        }
        if let Some(value) = high_tile_elements {
            metal_config.spartan_shift.dispatch.high_tile_elements = value;
        }
        if let Some(value) = fold_threads_per_threadgroup {
            metal_config
                .spartan_shift
                .dispatch
                .fold_threads_per_threadgroup = value;
        }
        let backend = MetalBackend::new(metal_config).map_err(kernel_error)?;
        let (_, resident_rows) =
            prepare_metal_spartan_outer_shift_witness_rows(&backend.context, witness, cycles)
                .map_err(|error| kernel_error(format!("{error:?}")))?;
        let census_started = Instant::now();
        let [unexpanded_pc_buffer, pc_buffer, _] = resident_rows.source_buffers();
        // SAFETY: each immutable shared buffer owns exactly `cycles` u64 values.
        let unexpanded_pc =
            unsafe { slice::from_raw_parts(unexpanded_pc_buffer.contents().cast::<u64>(), cycles) };
        // SAFETY: same ownership and immutability argument as `unexpanded_pc`.
        let pc = unsafe { slice::from_raw_parts(pc_buffer.contents().cast::<u64>(), cycles) };
        let (unexpanded_pc_max, unexpanded_pc_above_u32) = native_width_stats(unexpanded_pc);
        let (pc_max, pc_above_u32) = native_width_stats(pc);
        let native_width_census_wall = census_started.elapsed();
        let plan = SpartanShiftPlan::new(cycles, metal_config.spartan_shift.dispatch)
            .map_err(kernel_error)?;

        let prefix_observation = backend
            .context
            .prepare_spartan_shift_prefix(
                &resident_rows,
                &r_outer,
                &r_product,
                gamma,
                metal_config.spartan_shift.dispatch,
            )
            .map_err(kernel_error)?
            .execute()
            .map_err(kernel_error)?;
        let outer = EqPlusOnePrefixSuffix::new(&r_outer);
        let product = EqPlusOnePrefixSuffix::new(&r_product);
        let p = [
            outer.prefix_0,
            outer.prefix_1,
            product.prefix_0,
            product.prefix_1,
        ];
        let mut initial_claim = <AkitaField as WithAccumulator>::Accumulator::default();
        for (p, q) in p.iter().zip(prefix_observation.q.iter()) {
            for (&p, &q) in p.iter().zip(q) {
                initial_claim.fmadd(p, q);
            }
        }
        let initial_claim = initial_claim.reduce();

        let fold_observation = backend
            .context
            .prepare_spartan_shift_fold(
                &resident_rows,
                &sumcheck_challenges[..plan.geometry.prefix_vars()],
                metal_config.spartan_shift.dispatch,
            )
            .map_err(kernel_error)?
            .execute()
            .map_err(kernel_error)?;
        let shape = SpartanShiftShapeSnapshot {
            cycles,
            prefix_elements: plan.geometry.prefix_elements(),
            suffix_elements: plan.geometry.suffix_elements(),
            resident_source_bytes: resident_rows.resident_bytes(),
            native_value_bytes: plan.storage.native_value_bytes,
            native_flag_bytes: plan.storage.native_flag_bytes,
            partial_bytes: plan.storage.partial_bytes,
            q_bytes: plan.storage.q_bytes,
            dense_output_bytes: plan.storage.dense_output_bytes,
            total_resident_bytes: plan.storage.total_resident_bytes,
            build_unique_bytes: plan.cost.build_unique_bytes,
            build_coalesced_bytes_with_halo: plan.cost.build_coalesced_bytes_with_halo,
            fold_unique_bytes: plan.cost.fold_unique_bytes,
            readback_bytes: plan.cost.readback_bytes,
            mixed_full_products: plan.cost.mixed_full_products,
            mixed_half_products: plan.cost.mixed_half_products,
            fold_half_products: plan.cost.fold_half_products,
            command_buffers: plan.cost.command_buffers,
            dispatches: plan.cost.dispatches,
            build_threads_per_threadgroup: plan.config.build_threads_per_threadgroup,
            high_tile_elements: plan.config.high_tile_elements,
            fold_threads_per_threadgroup: plan.config.fold_threads_per_threadgroup,
            production_trace_cutoff_elements,
            forced_device_below_production_cutoff,
            unexpanded_pc_max,
            pc_max,
            unexpanded_pc_above_u32,
            pc_above_u32,
            native_width_census_wall,
            calibration_prefix_gpu_active: prefix_observation.gpu_active,
            calibration_fold_gpu_active: fold_observation.gpu_active,
        };
        Ok(Self {
            backend,
            resident_rows,
            relation,
            claims: SpartanShiftInputClaims::default(),
            points: SpartanShiftInputClaims::default(),
            relation_challenges,
            sumcheck_challenges,
            initial_claim,
            shape,
            fixture_wall: fixture_started.elapsed(),
        })
    }

    pub fn log_t(&self) -> usize {
        self.sumcheck_challenges.len()
    }

    pub fn cycles(&self) -> usize {
        self.shape.cycles
    }

    pub fn fixture_wall(&self) -> Duration {
        self.fixture_wall
    }

    pub fn shape(&self) -> &SpartanShiftShapeSnapshot {
        &self.shape
    }

    pub const fn metal_route(&self) -> &'static str {
        "resident_prefix_fold_v1"
    }

    pub fn run_cpu(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<SpartanShiftEvalSample, SpartanShiftEvalError> {
        let member_started = Instant::now();
        let prepare_started = Instant::now();
        let mut kernel = OptimizedSpartanShift
            .prepare(
                &mut ProofSession::default(),
                witness,
                ProverInputs {
                    relation: &self.relation,
                    claims: &self.claims,
                    points: &self.points,
                    challenges: &self.relation_challenges,
                },
            )
            .map_err(kernel_error)?;
        let prepare_wall = prepare_started.elapsed();
        self.run_prepared(&mut kernel, member_started, prepare_wall)
    }

    pub fn run_metal(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<SpartanShiftEvalSample, SpartanShiftEvalError> {
        let member_started = Instant::now();
        let prepare_started = Instant::now();
        let owner = SpartanDenseResidentOwner::from_co_produced_shift(self.resident_rows.clone())
            .map_err(kernel_error)?;
        let mut session = ProofSession::default();
        session.park(owner);
        let route_before = self.backend.spartan_shift_sequences();
        let mut kernel = self
            .backend
            .prepare(
                &mut session,
                witness,
                ProverInputs {
                    relation: &self.relation,
                    claims: &self.claims,
                    points: &self.points,
                    challenges: &self.relation_challenges,
                },
            )
            .map_err(kernel_error)?;
        let prepare_wall = prepare_started.elapsed();
        if self.backend.spartan_shift_sequences() != route_before + 1 {
            return Err(eval_error(
                "Spartan shift scored Metal arm did not select the resident device route",
            ));
        }
        self.run_prepared(&mut kernel, member_started, prepare_wall)
    }

    fn run_prepared(
        &self,
        kernel: &mut Box<dyn SumcheckKernel<AkitaField, Relation = SpartanShift<AkitaField>>>,
        member_started: Instant,
        prepare_wall: Duration,
    ) -> Result<SpartanShiftEvalSample, SpartanShiftEvalError> {
        let rounds_started = Instant::now();
        let mut previous_claim = self.initial_claim;
        let mut round_polynomials = Vec::with_capacity(self.sumcheck_challenges.len());
        let mut round_timings = Vec::with_capacity(self.sumcheck_challenges.len());
        for round in 0..self.sumcheck_challenges.len() {
            let round_started = Instant::now();
            let bind = round
                .checked_sub(1)
                .map(|previous| self.sumcheck_challenges[previous]);
            let polynomial = kernel
                .prove_round(bind, round, previous_claim)
                .map_err(kernel_error)?;
            round_timings.push(SpartanShiftRoundTiming {
                round,
                wall: round_started.elapsed(),
            });
            previous_claim = polynomial.evaluate(self.sumcheck_challenges[round]);
            round_polynomials.push(polynomial.coefficients().to_vec());
        }
        let rounds_wall = rounds_started.elapsed();

        let final_challenge = self
            .sumcheck_challenges
            .last()
            .copied()
            .ok_or_else(|| eval_error("Spartan shift evaluator has no terminal challenge"))?;
        let finish_started = Instant::now();
        kernel
            .finish_rounds(final_challenge)
            .map_err(kernel_error)?;
        let finish_wall = finish_started.elapsed();

        let output_started = Instant::now();
        let output_points = self
            .relation
            .derive_opening_points(&self.sumcheck_challenges, &self.points)
            .map_err(kernel_error)?;
        let output_claims = kernel.output_claims(&self.claims).map_err(kernel_error)?;
        kernel
            .validate_derived_tables(
                &self.relation,
                &self.points,
                &output_points,
                &self.relation_challenges,
            )
            .map_err(kernel_error)?;
        let expected_final = self
            .relation
            .expected_output(
                &self.points,
                &output_claims,
                &output_points,
                &self.relation_challenges,
            )
            .map_err(kernel_error)?;
        if previous_claim != expected_final {
            return Err(eval_error(
                "Spartan shift terminal relation does not match the bound sumcheck claim",
            ));
        }
        let output_claims = output_claims.opening_values();
        let output_wall = output_started.elapsed();

        Ok(SpartanShiftEvalSample {
            result: SpartanShiftEvalResult {
                round_polynomials,
                final_claim: previous_claim,
                output_claims,
            },
            member_wall: member_started.elapsed(),
            prepare_wall,
            rounds_wall,
            finish_wall,
            output_wall,
            round_timings,
        })
    }
}

fn native_width_stats(values: &[u64]) -> (u64, usize) {
    let combine = |left: (u64, usize), right: (u64, usize)| (left.0.max(right.0), left.1 + right.1);
    #[cfg(feature = "parallel")]
    {
        values
            .par_iter()
            .map(|&value| (value, usize::from(value > u64::from(u32::MAX))))
            .reduce(|| (0, 0), combine)
    }
    #[cfg(not(feature = "parallel"))]
    {
        values.iter().copied().fold((0, 0), |acc, value| {
            combine(acc, (value, usize::from(value > u64::from(u32::MAX))))
        })
    }
}

fn challenge_sequence(seed: u64, length: usize) -> Vec<AkitaField> {
    (0..length)
        .map(|index| challenge_field(seed ^ index as u64))
        .collect()
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

fn kernel_error(error: impl ToString) -> SpartanShiftEvalError {
    eval_error(error.to_string())
}

fn eval_error(message: impl Into<String>) -> SpartanShiftEvalError {
    SpartanShiftEvalError::Kernel(message.into())
}
