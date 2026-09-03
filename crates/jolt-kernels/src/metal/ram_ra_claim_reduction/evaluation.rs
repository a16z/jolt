use std::{mem::size_of, sync::Arc, time::Duration, time::Instant};

use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::relations::ram::{
    RamRaClaimReductionChallenges, RamRaClaimReductionInputClaims,
};
use jolt_claims::OutputClaims as _;
use jolt_field::Zero as _;
use jolt_field::{CanonicalBytes as _, CanonicalEncoding as _, Prime128OffsetA7F7 as AkitaField};
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
use jolt_witness::JoltWitnessPlane;

use crate::metal::backend::{MetalBackend, MetalConfig};
use crate::metal::ram_cycle_family::shared_ram_cycle_family_owner;
use crate::metal::ram_records::{RamAccessColumns, RamIncrementActivity, NO_ACCESS};
use crate::metal::solinas::ram_cycle_family::RamCycleFamilyOwner;
use crate::metal::solinas::{DeviceInfo, MetalError};
use crate::optimized::OptimizedBackend;
use crate::ram_access::RamAccessTape;
use crate::{PrepareKernel, ProofSession, ProverInputs};

#[derive(Debug, thiserror::Error)]
pub enum RamRaClaimReductionEvalError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("RAM RA claim-reduction evaluator failed: {0}")]
    Kernel(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamRaClaimReductionEvalResult {
    round_polynomials: Vec<Vec<AkitaField>>,
    final_claim: AkitaField,
    output_claims: Vec<AkitaField>,
}

impl RamRaClaimReductionEvalResult {
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

    pub fn first_difference(&self, other: &Self) -> Option<String> {
        if self.round_polynomials.len() != other.round_polynomials.len() {
            return Some("round count".to_string());
        }
        for (round, (expected, got)) in self
            .round_polynomials
            .iter()
            .zip(&other.round_polynomials)
            .enumerate()
        {
            if expected != got {
                let coefficient = expected
                    .iter()
                    .zip(got)
                    .position(|(expected, got)| expected != got);
                return Some(format!("round {round}, coefficient {coefficient:?}"));
            }
        }
        if self.final_claim != other.final_claim {
            return Some("final claim".to_string());
        }
        if self.output_claims != other.output_claims {
            return Some("output claims".to_string());
        }
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimReductionRoundTiming {
    pub round: usize,
    pub wall: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamRaClaimReductionShapeSnapshot {
    pub cycles: usize,
    pub active_cycle_bound: usize,
    pub active_high_elements: usize,
    pub active_q_slices: usize,
    pub compact_access_records: usize,
    pub addresses: usize,
    pub accesses: usize,
    pub no_access_cycles: usize,
    pub nonzero_increments: usize,
    pub maximum_address: Option<u32>,
    pub prefix_bits: usize,
    pub suffix_bits: usize,
    pub address_source_bytes: usize,
    pub address_eq_bytes: usize,
    pub q_table_bytes: usize,
    pub h_prime_bytes: usize,
    pub q_full_field_products: usize,
    pub h_prime_full_field_products: usize,
}

#[derive(Clone, Debug)]
pub struct RamRaClaimReductionEvalSample {
    pub result: RamRaClaimReductionEvalResult,
    pub member_wall: Duration,
    pub prepare_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
    pub q_wall: Option<Duration>,
    pub q_gpu_active: Option<Duration>,
    pub q_wait_wall: Option<Duration>,
    pub q_readback_wall: Option<Duration>,
    pub address_alias_reused: Option<bool>,
    pub h_wall: Option<Duration>,
    pub h_gpu_active: Option<Duration>,
    pub round_timings: Vec<RamRaClaimReductionRoundTiming>,
}

pub type RamRaClaimReductionMetalEvalSample = RamRaClaimReductionEvalSample;

pub struct RamRaClaimReductionCpuMetalEvalFixture {
    backend: MetalBackend,
    columns: Arc<RamAccessColumns>,
    increments: Arc<RamIncrementActivity>,
    relation: RamRaClaimReduction<AkitaField>,
    claims: RamRaClaimReductionInputClaims<AkitaField>,
    points: RamRaClaimReductionInputClaims<Vec<AkitaField>>,
    relation_challenges: RamRaClaimReductionChallenges<AkitaField>,
    sumcheck_challenges: Vec<AkitaField>,
    input_claim: AkitaField,
    shape: RamRaClaimReductionShapeSnapshot,
    fixture_wall: Duration,
    sparse_owner: Option<Arc<RamCycleFamilyOwner>>,
}

impl RamRaClaimReductionCpuMetalEvalFixture {
    pub fn new(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        log_k: usize,
        seed: u64,
    ) -> Result<Self, RamRaClaimReductionEvalError> {
        Self::new_with_q_slices(witness, log_t, log_k, seed, 1)
    }

    pub fn new_with_q_slices(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        log_k: usize,
        seed: u64,
        q_slices: usize,
    ) -> Result<Self, RamRaClaimReductionEvalError> {
        Self::new_with_routing(witness, log_t, log_k, seed, q_slices, false)
    }

    pub fn new_with_production_routing(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        log_k: usize,
        seed: u64,
        q_slices: usize,
    ) -> Result<Self, RamRaClaimReductionEvalError> {
        Self::new_with_routing(witness, log_t, log_k, seed, q_slices, true)
    }

    fn new_with_routing(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        log_k: usize,
        seed: u64,
        q_slices: usize,
        production_routing: bool,
    ) -> Result<Self, RamRaClaimReductionEvalError> {
        if !(4..=28).contains(&log_t) || !(1..u32::BITS as usize).contains(&log_k) {
            return Err(eval_error(
                "RAM RA claim-reduction evaluator geometry is outside the supported range",
            ));
        }
        let fixture_started = Instant::now();
        let cycles = 1usize << log_t;
        let addresses = 1usize << log_k;
        let mut config = MetalConfig::production();
        config.ram_ra_claim_reduction.q_slices = q_slices;
        if !production_routing {
            config.ram_ra_claim_reduction.trace_cutoff_elements = 1usize << log_t;
        }
        let backend = MetalBackend::new(config)?;
        let mut source_session = ProofSession::default();
        let columns =
            RamAccessColumns::shared(&mut source_session, witness, log_t).map_err(kernel_error)?;
        columns
            .validate_addresses::<AkitaField>(addresses)
            .map_err(kernel_error)?;
        let increments = source_session
            .state::<Arc<RamIncrementActivity>>()
            .cloned()
            .ok_or_else(|| eval_error("RAM source did not publish increment activity"))?;
        let sparse_owner = production_routing
            .then(|| shared_ram_cycle_family_owner(&mut source_session, witness, log_t, log_k))
            .transpose()
            .map_err(kernel_error)?
            .flatten();

        let r_address = (0..log_k)
            .map(|index| challenge_field(seed ^ 0xa54f_f53a_5f1d_36f1 ^ index as u64))
            .collect::<Vec<_>>();
        let cycle_point = |domain: u64| {
            (0..log_t)
                .map(|index| challenge_field(seed ^ domain ^ index as u64))
                .collect::<Vec<_>>()
        };
        let points = RamRaClaimReductionInputClaims {
            raf: [r_address.clone(), cycle_point(0x3c6e_f372_fe94_f82b)].concat(),
            read_write: [r_address.clone(), cycle_point(0xbb67_ae85_84ca_a73b)].concat(),
            val_check: [r_address, cycle_point(0x510e_527f_ade6_82d1)].concat(),
        };
        let relation = RamRaClaimReduction::new(TraceDimensions::new(log_t), log_k);
        let relation_challenges = RamRaClaimReductionChallenges {
            gamma: challenge_field(seed ^ 0x1f83_d9ab_fb41_bd6b),
        };
        let input_claim =
            probe_input_claim(witness, &columns, &relation, &points, &relation_challenges)?;
        let claims = RamRaClaimReductionInputClaims {
            raf: input_claim,
            read_write: AkitaField::zero(),
            val_check: AkitaField::zero(),
        };
        let sumcheck_challenges = (0..log_t)
            .map(|round| challenge_field(seed ^ 0x9e37_79b9_7f4a_7c15 ^ round as u64))
            .collect::<Vec<_>>();
        let accesses = columns
            .addresses
            .iter()
            .filter(|&&address| address != NO_ACCESS)
            .count();
        let maximum_address = columns
            .addresses
            .iter()
            .copied()
            .filter(|&address| address != NO_ACCESS)
            .max();
        let prefix_bits = log_t / 2;
        let suffix_bits = log_t - prefix_bits;
        let active_cycle_bound = columns.active_cycle_bound();
        let active_high_elements = active_cycle_bound.div_ceil(1usize << prefix_bits);
        let high_per_slice = (1usize << suffix_bits) / q_slices;
        let active_q_slices = active_high_elements
            .div_ceil(high_per_slice)
            .clamp(1, q_slices);
        let compact_access_records = columns
            .ram_ra_sparse_layout()
            .map_or(0, |layout| layout.h_records().len());
        let shape = RamRaClaimReductionShapeSnapshot {
            cycles,
            active_cycle_bound,
            active_high_elements,
            active_q_slices,
            compact_access_records,
            addresses,
            accesses,
            no_access_cycles: cycles - accesses,
            nonzero_increments: increments.len(),
            maximum_address,
            prefix_bits,
            suffix_bits,
            address_source_bytes: cycles * size_of::<u32>(),
            address_eq_bytes: addresses * size_of::<AkitaField>(),
            q_table_bytes: 3 * (1usize << prefix_bits) * size_of::<AkitaField>(),
            h_prime_bytes: (1usize << suffix_bits) * size_of::<AkitaField>(),
            q_full_field_products: 3 * accesses,
            h_prime_full_field_products: accesses,
        };
        Ok(Self {
            backend,
            columns,
            increments,
            relation,
            claims,
            points,
            relation_challenges,
            sumcheck_challenges,
            input_claim,
            shape,
            fixture_wall: fixture_started.elapsed(),
            sparse_owner,
        })
    }

    pub fn log_t(&self) -> usize {
        self.relation.trace_dimensions().log_t()
    }

    pub fn log_k(&self) -> usize {
        self.relation.ram_log_k()
    }

    pub fn cycles(&self) -> usize {
        self.shape.cycles
    }

    pub fn fixture_wall(&self) -> Duration {
        self.fixture_wall
    }

    pub fn shape(&self) -> &RamRaClaimReductionShapeSnapshot {
        &self.shape
    }

    pub fn device_info(&self) -> DeviceInfo {
        self.backend.context.device_info()
    }

    pub fn metal_route(&self) -> &'static str {
        if self.cycles()
            >= self
                .backend
                .config
                .ram_ra_claim_reduction
                .trace_cutoff_elements
        {
            if self.backend.ram_ra_claim_sparse_sequences() != 0 {
                "host_sparse_v1"
            } else if self.shape.compact_access_records != 0 {
                "no_copy_sparse_records_hybrid_v1"
            } else if self.backend.config.ram_ra_claim_reduction.q_slices == 1 {
                "no_copy_q_hybrid_v1"
            } else {
                "no_copy_q_sliced_hybrid_v1"
            }
        } else {
            "optimized_cpu_host"
        }
    }

    pub const fn q_slices(&self) -> usize {
        self.backend.config.ram_ra_claim_reduction.q_slices
    }

    pub fn run_cpu(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<RamRaClaimReductionEvalSample, RamRaClaimReductionEvalError> {
        self.run(witness, false)
    }

    pub fn run_metal(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<RamRaClaimReductionMetalEvalSample, RamRaClaimReductionEvalError> {
        self.run(witness, true)
    }

    fn run(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        metal: bool,
    ) -> Result<RamRaClaimReductionEvalSample, RamRaClaimReductionEvalError> {
        let inputs = || ProverInputs {
            relation: &self.relation,
            claims: &self.claims,
            points: &self.points,
            challenges: &self.relation_challenges,
        };

        let mut session = ProofSession::default();
        session.park(Arc::clone(&self.columns));
        if metal {
            session.park(Arc::clone(&self.increments));
            if let Some(owner) = &self.sparse_owner {
                session.park(Arc::clone(owner));
            } else {
                session.park(RamAccessTape::new(
                    self.log_t(),
                    self.shape.accesses,
                    None,
                    true,
                    true,
                    true,
                ));
            }
        }
        let metal_sequences_before = self.backend.ram_ra_claim_metal_sequences();
        let member_started = Instant::now();
        let prepare_started = Instant::now();
        let mut kernel = if metal {
            self.backend.prepare(&mut session, witness, inputs())
        } else {
            OptimizedBackend.prepare(&mut session, witness, inputs())
        }
        .map_err(kernel_error)?;
        let prepare_wall = prepare_started.elapsed();
        let used_metal_sequence =
            metal && self.backend.ram_ra_claim_metal_sequences() != metal_sequences_before;
        let q_wall = used_metal_sequence
            .then(|| Duration::from_nanos(self.backend.ram_ra_claim_q_wall_ns() as u64));
        let q_gpu_active = used_metal_sequence
            .then(|| Duration::from_nanos(self.backend.ram_ra_claim_q_gpu_ns() as u64));
        let q_wait_wall = used_metal_sequence
            .then(|| Duration::from_nanos(self.backend.ram_ra_claim_q_wait_wall_ns() as u64));
        let q_readback_wall = used_metal_sequence
            .then(|| Duration::from_nanos(self.backend.ram_ra_claim_q_readback_wall_ns() as u64));
        let address_alias_reused =
            used_metal_sequence.then(|| self.backend.ram_ra_claim_address_alias_reuses() != 0);

        let rounds_started = Instant::now();
        let mut previous_claim = self.input_claim;
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
            round_timings.push(RamRaClaimReductionRoundTiming {
                round,
                wall: round_started.elapsed(),
            });
            previous_claim = polynomial.evaluate(self.sumcheck_challenges[round]);
            round_polynomials.push(polynomial.coefficients().to_vec());
        }
        let rounds_wall = rounds_started.elapsed();
        let h_wall = used_metal_sequence
            .then(|| Duration::from_nanos(self.backend.ram_ra_claim_h_wall_ns() as u64));
        let h_gpu_active = used_metal_sequence
            .then(|| Duration::from_nanos(self.backend.ram_ra_claim_h_gpu_ns() as u64));

        let final_challenge = self
            .sumcheck_challenges
            .last()
            .copied()
            .ok_or_else(|| eval_error("RAM RA claim-reduction evaluator has no challenge"))?;
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
        let output_claims = output_claims.opening_values();
        let output_wall = output_started.elapsed();

        Ok(RamRaClaimReductionEvalSample {
            result: RamRaClaimReductionEvalResult {
                round_polynomials,
                final_claim: previous_claim,
                output_claims,
            },
            member_wall: member_started.elapsed(),
            prepare_wall,
            rounds_wall,
            finish_wall,
            output_wall,
            q_wall,
            q_gpu_active,
            q_wait_wall,
            q_readback_wall,
            address_alias_reused,
            h_wall,
            h_gpu_active,
            round_timings,
        })
    }
}

fn probe_input_claim(
    witness: &dyn JoltWitnessPlane<AkitaField>,
    columns: &Arc<RamAccessColumns>,
    relation: &RamRaClaimReduction<AkitaField>,
    points: &RamRaClaimReductionInputClaims<Vec<AkitaField>>,
    challenges: &RamRaClaimReductionChallenges<AkitaField>,
) -> Result<AkitaField, RamRaClaimReductionEvalError> {
    let claims = RamRaClaimReductionInputClaims::<AkitaField>::default();
    let mut session = ProofSession::default();
    session.park(Arc::clone(columns));
    let mut kernel = OptimizedBackend
        .prepare(
            &mut session,
            witness,
            ProverInputs {
                relation,
                claims: &claims,
                points,
                challenges,
            },
        )
        .map_err(kernel_error)?;
    match kernel.prove_round(None, 0, AkitaField::zero()) {
        Ok(_) => Ok(AkitaField::zero()),
        Err(SumcheckError::RoundCheckFailed { actual, .. }) => Ok(actual),
        Err(error) => Err(kernel_error(error)),
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

fn eval_error(message: impl Into<String>) -> RamRaClaimReductionEvalError {
    RamRaClaimReductionEvalError::Kernel(message.into())
}

fn kernel_error(error: impl ToString) -> RamRaClaimReductionEvalError {
    eval_error(error.to_string())
}
