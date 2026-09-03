use std::{cell::OnceCell, mem::size_of, sync::Arc, time::Duration, time::Instant};

use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::geometry::ram::RamValCheckInit;
use jolt_claims::protocols::jolt::relations::ram::{RamValCheckChallenges, RamValCheckInputClaims};
use jolt_claims::protocols::jolt::{JoltDerivedId, RamValCheckPublic};
use jolt_claims::OutputClaims as _;
use jolt_field::{Field as _, One as _, Zero as _};
use jolt_field::{
    FixedBytes, FromPrimitiveInt, Prime128OffsetA7F7 as AkitaField, TranscriptChallenge,
};
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage4::ram_val_check::{RamValCheck, RamValCheckOutputClaims};
use jolt_witness::JoltWitnessPlane;

use crate::metal::ram_records::{RamAccessColumns, RamIncrementActivity, NO_ACCESS};
use crate::metal::solinas::ram_cycle_family::RamBlockTopology;
use crate::metal::solinas::SolinasMetal;
use crate::optimized::support::SplitLt;
use crate::optimized::OptimizedBackend;
use crate::reference::views::eq_table;
use crate::{PrepareKernel, ProofSession, ProverInputs};

#[derive(Debug, thiserror::Error)]
pub enum RamValCheckEvalError {
    #[error("RAM value-check evaluator failed: {0}")]
    Kernel(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamValCheckEvalResult {
    round_polynomials: Vec<Vec<AkitaField>>,
    final_claim: AkitaField,
    output_claims: Vec<AkitaField>,
}

impl RamValCheckEvalResult {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamValCheckRoundTiming {
    pub round: usize,
    pub wall: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamValCheckShapeSnapshot {
    pub cycles: usize,
    pub addresses: usize,
    pub accesses: usize,
    pub no_access_cycles: usize,
    pub nonzero_increments: usize,
    pub maximum_address: Option<u32>,
    pub active_increment_pairs_by_width: Vec<usize>,
    pub increment_topology_entries_by_level: Vec<u64>,
    pub increment_topology_bytes: usize,
    pub increment_topology_wall: Duration,
    pub address_source_bytes: usize,
    pub increment_source_bytes: usize,
    pub dense_inc_field_bytes: usize,
    pub address_eq_bytes: usize,
    pub split_lt_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct RamValCheckCpuEvalSample {
    pub result: RamValCheckEvalResult,
    pub member_wall: Duration,
    pub prepare_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
    pub round_timings: Vec<RamValCheckRoundTiming>,
}

pub struct RamValCheckCpuMetalEvalFixture {
    columns: Arc<RamAccessColumns>,
    increments: Arc<RamIncrementActivity>,
    metal: OnceCell<SolinasMetal>,
    relation: RamValCheck<AkitaField>,
    claims: RamValCheckInputClaims<AkitaField>,
    points: RamValCheckInputClaims<Vec<AkitaField>>,
    relation_challenges: RamValCheckChallenges<AkitaField>,
    sumcheck_challenges: Vec<AkitaField>,
    shape: RamValCheckShapeSnapshot,
    fixture_wall: Duration,
}

impl RamValCheckCpuMetalEvalFixture {
    /// Collects the production RAM address and sparse-increment sources once.
    pub fn new(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        log_k: usize,
        seed: u64,
    ) -> Result<Self, RamValCheckEvalError> {
        if !(4..=28).contains(&log_t) || !(1..u32::BITS as usize).contains(&log_k) {
            return Err(eval_error(
                "RAM value-check evaluator geometry is outside the supported range",
            ));
        }
        let fixture_started = Instant::now();
        let cycles = 1usize << log_t;
        let addresses = 1usize << log_k;
        let mut source_session = ProofSession::default();
        let columns =
            RamAccessColumns::shared(&mut source_session, witness, log_t).map_err(kernel_error)?;
        columns
            .validate_addresses::<AkitaField>(addresses)
            .map_err(kernel_error)?;
        let increments = source_session
            .take::<Arc<RamIncrementActivity>>()
            .ok_or_else(|| eval_error("RAM source did not publish increment activity"))?;
        let topology_started = Instant::now();
        let increment_topology =
            RamBlockTopology::build(log_t, &[], increments.cycle_slice()).map_err(kernel_error)?;
        let increment_topology_wall = topology_started.elapsed();
        let increment_topology_entries_by_level = increment_topology
            .census()
            .iter()
            .map(|level| level.entries())
            .collect();
        let increment_topology_bytes = increment_topology.owned_heap_bytes();
        drop(increment_topology);

        let r_address = (0..log_k)
            .map(|index| challenge_field(seed ^ 0xa54f_f53a_5f1d_36f1 ^ index as u64))
            .collect::<Vec<_>>();
        let r_cycle = (0..log_t)
            .map(|index| challenge_field(seed ^ 0x3c6e_f372_fe94_f82b ^ index as u64))
            .collect::<Vec<_>>();
        let gamma = challenge_field(seed ^ 0xbb67_ae85_84ca_a73b);
        let relation = RamValCheck::new(
            TraceDimensions::new(log_t),
            log_k,
            RamValCheckInit::full(AkitaField::zero()),
        );
        let points = RamValCheckInputClaims {
            ram_val: [r_address.clone(), r_cycle.clone()].concat(),
            ram_val_final: r_address.clone(),
            untrusted_advice: None,
            trusted_advice: None,
            program_image: None,
        };
        let relation_challenges = RamValCheckChallenges { gamma };
        let input_claim = sparse_input_claim(
            &columns,
            &increments,
            &r_address,
            &r_cycle,
            relation_challenges.gamma,
        )?;
        let claims = RamValCheckInputClaims {
            ram_val: input_claim,
            ram_val_final: AkitaField::zero(),
            untrusted_advice: None,
            trusted_advice: None,
            program_image: None,
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
        let shape = RamValCheckShapeSnapshot {
            cycles,
            addresses,
            accesses,
            no_access_cycles: cycles - accesses,
            nonzero_increments: increments.len(),
            maximum_address,
            active_increment_pairs_by_width: active_increment_pairs(&increments),
            increment_topology_entries_by_level,
            increment_topology_bytes,
            increment_topology_wall,
            address_source_bytes: cycles * size_of::<u32>(),
            increment_source_bytes: increments.len() * (size_of::<u64>() + size_of::<i128>()),
            dense_inc_field_bytes: cycles * size_of::<AkitaField>(),
            address_eq_bytes: addresses * size_of::<AkitaField>(),
            split_lt_bytes: 3 * (1usize << log_t.div_ceil(2)) * size_of::<AkitaField>(),
        };
        Ok(Self {
            columns,
            increments,
            metal: OnceCell::new(),
            relation,
            claims,
            points,
            relation_challenges,
            sumcheck_challenges,
            shape,
            fixture_wall: fixture_started.elapsed(),
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

    pub fn shape(&self) -> &RamValCheckShapeSnapshot {
        &self.shape
    }

    pub const fn metal_route(&self) -> &'static str {
        "sparse_increment_width32_v1"
    }

    pub fn run_cpu(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<RamValCheckCpuEvalSample, RamValCheckEvalError> {
        let mut session = ProofSession::default();
        session.park(Arc::clone(&self.columns));
        let inputs = || ProverInputs {
            relation: &self.relation,
            claims: &self.claims,
            points: &self.points,
            challenges: &self.relation_challenges,
        };
        let member_started = Instant::now();
        let prepare_started = Instant::now();
        let mut kernel = OptimizedBackend
            .prepare(&mut session, witness, inputs())
            .map_err(kernel_error)?;
        let prepare_wall = prepare_started.elapsed();

        let rounds_started = Instant::now();
        let mut previous_claim = self.claims.ram_val;
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
            round_timings.push(RamValCheckRoundTiming {
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
            .ok_or_else(|| eval_error("RAM value-check evaluator has no terminal challenge"))?;
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

        Ok(RamValCheckCpuEvalSample {
            result: RamValCheckEvalResult {
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

    pub fn run_metal(
        &self,
        _witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<RamValCheckCpuEvalSample, RamValCheckEvalError> {
        if self.metal.get().is_none() {
            let context = SolinasMetal::for_akita().map_err(kernel_error)?;
            self.metal
                .set(context)
                .map_err(|_| eval_error("RAM value-check Metal context initialized twice"))?;
        }
        let context = self
            .metal
            .get()
            .ok_or_else(|| eval_error("RAM value-check Metal context is unavailable"))?;
        let (r_address, r_cycle) = self.points.ram_val.split_at(self.log_k());

        let member_started = Instant::now();
        let prepare_started = Instant::now();
        let mut sequence = context
            .prepare_ram_val_sequence(
                Arc::clone(&self.columns),
                Arc::clone(&self.increments),
                r_address,
                r_cycle,
                self.relation_challenges.gamma,
            )
            .map_err(kernel_error)?;
        let prepare_wall = prepare_started.elapsed();

        let rounds_started = Instant::now();
        let mut previous_claim = self.claims.ram_val;
        let mut round_polynomials = Vec::with_capacity(self.sumcheck_challenges.len());
        let mut round_timings = Vec::with_capacity(self.sumcheck_challenges.len());
        for round in 0..self.sumcheck_challenges.len() {
            let round_started = Instant::now();
            let evaluations = match round.checked_sub(1) {
                Some(previous) => sequence
                    .bind_and_message(self.sumcheck_challenges[previous])
                    .map_err(kernel_error)?,
                None => sequence.message().map_err(kernel_error)?,
            };
            let polynomial =
                jolt_poly::UnivariatePoly::from_evals_and_hint(previous_claim, &evaluations);
            round_timings.push(RamValCheckRoundTiming {
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
            .ok_or_else(|| eval_error("RAM value-check evaluator has no terminal challenge"))?;
        let finish_started = Instant::now();
        let terminal = sequence
            .finish_bind(final_challenge)
            .map_err(kernel_error)?;
        let finish_wall = finish_started.elapsed();

        let output_started = Instant::now();
        let output_points = self
            .relation
            .derive_opening_points(&self.sumcheck_challenges, &self.points)
            .map_err(kernel_error)?;
        let id = JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma);
        let expected_lt = self
            .relation
            .derive_output_term(&id, &self.points, &output_points, &self.relation_challenges)
            .map_err(kernel_error)?;
        if terminal[2] != expected_lt {
            return Err(eval_error(
                "RAM value-check Metal sequence produced a derived LT mismatch",
            ));
        }
        let output_claims = RamValCheckOutputClaims {
            untrusted_advice: self.claims.untrusted_advice,
            trusted_advice: self.claims.trusted_advice,
            program_image: self.claims.program_image,
            ram_ra: terminal[1],
            ram_inc: terminal[0],
        }
        .opening_values();
        let output_wall = output_started.elapsed();

        Ok(RamValCheckCpuEvalSample {
            result: RamValCheckEvalResult {
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

fn sparse_input_claim(
    columns: &RamAccessColumns,
    increments: &RamIncrementActivity,
    r_address: &[AkitaField],
    r_cycle: &[AkitaField],
    gamma: AkitaField,
) -> Result<AkitaField, RamValCheckEvalError> {
    let eq_address = eq_table(r_address);
    let lt = SplitLt::new_plus_constant(r_cycle, gamma);
    let mut claim = AkitaField::zero();
    for (cycle, increment) in increments.records() {
        let address = columns
            .addresses
            .get(cycle)
            .copied()
            .ok_or_else(|| eval_error("RAM increment cycle is outside the address source"))?;
        if address == NO_ACCESS {
            continue;
        }
        let ra = eq_address
            .get(address as usize)
            .copied()
            .ok_or_else(|| eval_error("RAM increment address is outside the equality table"))?;
        let (lt_lo, lt_hi) = lt.pair(cycle / 2);
        let lt_value = if cycle.is_multiple_of(2) {
            lt_lo
        } else {
            lt_hi
        };
        claim += AkitaField::from_i128(increment) * ra * lt_value;
    }
    Ok(claim)
}

fn active_increment_pairs(increments: &RamIncrementActivity) -> Vec<usize> {
    [1usize, 2, 4, 8, 16]
        .into_iter()
        .map(|width| {
            let mut previous = None;
            let mut active = 0usize;
            for (cycle, _) in increments.records() {
                let pair = cycle / (2 * width);
                if previous != Some(pair) {
                    active += 1;
                    previous = Some(pair);
                }
            }
            active
        })
        .collect()
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

fn kernel_error(error: impl ToString) -> RamValCheckEvalError {
    eval_error(error.to_string())
}

fn eval_error(message: impl Into<String>) -> RamValCheckEvalError {
    RamValCheckEvalError::Kernel(message.into())
}
