use std::{cell::OnceCell, mem::size_of, sync::Arc, time::Duration, time::Instant};

use jolt_claims::protocols::jolt::{JoltDerivedId, RamHammingBooleanityPublic};
use jolt_claims::OutputClaims as _;
use jolt_field::{Field as _, One as _, Zero as _};
use jolt_field::{FixedBytes, Prime128OffsetA7F7 as AkitaField, TranscriptChallenge};
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::{
    RamHammingBooleanity, RamHammingBooleanityInputClaims, RamHammingBooleanityOutputClaims,
};
use jolt_witness::JoltWitnessPlane;

use crate::metal::solinas::SolinasMetal;
use crate::optimized::ram_hamming_booleanity::OptimizedRamHammingBooleanity;
use crate::optimized::ram_trace::{RamAccessColumns, NO_ACCESS};
use crate::{PrepareKernel, ProofSession, ProverInputs};

#[derive(Debug, thiserror::Error)]
pub enum RamHammingBooleanityEvalError {
    #[error("RAM Hamming booleanity evaluator failed: {0}")]
    Kernel(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamHammingBooleanityEvalResult {
    round_polynomials: Vec<Vec<AkitaField>>,
    final_claim: AkitaField,
    output_claims: Vec<AkitaField>,
}

impl RamHammingBooleanityEvalResult {
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
pub struct RamHammingBooleanityRoundTiming {
    pub round: usize,
    pub wall: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamHammingBooleanityShapeSnapshot {
    pub cycles: usize,
    pub accesses: usize,
    pub no_access_cycles: usize,
    pub access_runs: usize,
    pub mixed_parents_by_child_width: Vec<usize>,
    pub census_wall: Duration,
    pub address_source_bytes: usize,
    pub packed_access_bytes: usize,
    pub dense_h_field_bytes: usize,
    pub width_32_dense_a_bytes: usize,
    pub width_32_dense_b_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct RamHammingBooleanityEvalSample {
    pub result: RamHammingBooleanityEvalResult,
    pub member_wall: Duration,
    pub prepare_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
    pub round_timings: Vec<RamHammingBooleanityRoundTiming>,
}

pub struct RamHammingBooleanityCpuEvalFixture {
    columns: Arc<RamAccessColumns>,
    metal: OnceCell<SolinasMetal>,
    relation: RamHammingBooleanity<AkitaField>,
    claims: RamHammingBooleanityInputClaims<AkitaField>,
    points: RamHammingBooleanityInputClaims<Vec<AkitaField>>,
    sumcheck_challenges: Vec<AkitaField>,
    shape: RamHammingBooleanityShapeSnapshot,
    fixture_wall: Duration,
}

impl RamHammingBooleanityCpuEvalFixture {
    pub fn new(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        seed: u64,
    ) -> Result<Self, RamHammingBooleanityEvalError> {
        if !(4..=28).contains(&log_t) {
            return Err(eval_error(
                "RAM Hamming booleanity evaluator geometry is outside the supported range",
            ));
        }
        let fixture_started = Instant::now();
        let cycles = 1usize << log_t;
        let mut source_session = ProofSession::default();
        let columns =
            RamAccessColumns::shared(&mut source_session, witness, log_t).map_err(kernel_error)?;

        let census_started = Instant::now();
        let accesses = columns
            .addresses
            .iter()
            .filter(|&&address| address != NO_ACCESS)
            .count();
        let access_runs = columns
            .addresses
            .iter()
            .map(|&address| address != NO_ACCESS)
            .fold((0usize, false), |(runs, previous), access| {
                (runs + usize::from(access && !previous), access)
            })
            .0;
        let mixed_parents_by_child_width = [1usize, 2, 4, 8, 16, 32]
            .into_iter()
            .map(|width| mixed_parent_count(&columns.addresses, width))
            .collect();
        let census_wall = census_started.elapsed();

        let stage1_cycle_binding = (0..log_t)
            .map(|index| challenge_field(seed ^ 0xa54f_f53a_5f1d_36f1 ^ index as u64))
            .collect::<Vec<_>>();
        let relation = RamHammingBooleanity::new(
            jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions::new(log_t),
            stage1_cycle_binding,
        );
        let claims = RamHammingBooleanityInputClaims::default();
        let points = RamHammingBooleanityInputClaims::default();
        let sumcheck_challenges = (0..log_t)
            .map(|round| challenge_field(seed ^ 0x9e37_79b9_7f4a_7c15 ^ round as u64))
            .collect::<Vec<_>>();
        let width_32_elements = cycles / 32;
        let shape = RamHammingBooleanityShapeSnapshot {
            cycles,
            accesses,
            no_access_cycles: cycles - accesses,
            access_runs,
            mixed_parents_by_child_width,
            census_wall,
            address_source_bytes: cycles * size_of::<u32>(),
            packed_access_bytes: cycles.div_ceil(u8::BITS as usize),
            dense_h_field_bytes: cycles * size_of::<AkitaField>(),
            width_32_dense_a_bytes: width_32_elements * size_of::<AkitaField>(),
            width_32_dense_b_bytes: width_32_elements * size_of::<AkitaField>() / 2,
        };
        Ok(Self {
            columns,
            metal: OnceCell::new(),
            relation,
            claims,
            points,
            sumcheck_challenges,
            shape,
            fixture_wall: fixture_started.elapsed(),
        })
    }

    pub fn log_t(&self) -> usize {
        self.relation.trace_dimensions().log_t()
    }

    pub fn cycles(&self) -> usize {
        self.shape.cycles
    }

    pub fn fixture_wall(&self) -> Duration {
        self.fixture_wall
    }

    pub fn shape(&self) -> &RamHammingBooleanityShapeSnapshot {
        &self.shape
    }

    pub const fn metal_route(&self) -> &'static str {
        "packed_access_width32_v1"
    }

    pub fn run_cpu(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<RamHammingBooleanityEvalSample, RamHammingBooleanityEvalError> {
        let relation_challenges = jolt_claims::NoChallenges::default();
        let inputs = || ProverInputs {
            relation: &self.relation,
            claims: &self.claims,
            points: &self.points,
            challenges: &relation_challenges,
        };
        let member_started = Instant::now();
        let prepare_started = Instant::now();
        let mut kernel = OptimizedRamHammingBooleanity
            .prepare(&mut ProofSession::default(), witness, inputs())
            .map_err(kernel_error)?;
        let prepare_wall = prepare_started.elapsed();

        let rounds_started = Instant::now();
        let mut previous_claim = AkitaField::zero();
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
            round_timings.push(RamHammingBooleanityRoundTiming {
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
            .ok_or_else(|| eval_error("RAM Hamming evaluator has no terminal challenge"))?;
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
                &relation_challenges,
            )
            .map_err(kernel_error)?;
        let output_claims = output_claims.opening_values();
        let output_wall = output_started.elapsed();

        Ok(RamHammingBooleanityEvalSample {
            result: RamHammingBooleanityEvalResult {
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
    ) -> Result<RamHammingBooleanityEvalSample, RamHammingBooleanityEvalError> {
        if self.metal.get().is_none() {
            let context = SolinasMetal::for_akita().map_err(kernel_error)?;
            self.metal
                .set(context)
                .map_err(|_| eval_error("RAM Hamming Metal context initialized twice"))?;
        }
        let context = self
            .metal
            .get()
            .ok_or_else(|| eval_error("RAM Hamming Metal context is unavailable"))?;
        let member_started = Instant::now();
        let prepare_started = Instant::now();
        let mut sequence = context
            .prepare_ram_hamming_sequence(
                Arc::clone(&self.columns),
                self.relation.stage1_cycle_binding(),
            )
            .map_err(kernel_error)?;
        let prepare_wall = prepare_started.elapsed();

        let rounds_started = Instant::now();
        let mut previous_claim = AkitaField::zero();
        let mut round_polynomials = Vec::with_capacity(self.sumcheck_challenges.len());
        let mut round_timings = Vec::with_capacity(self.sumcheck_challenges.len());
        for round in 0..self.sumcheck_challenges.len() {
            let round_started = Instant::now();
            let polynomial = match round.checked_sub(1) {
                Some(previous) => sequence
                    .bind_and_message(self.sumcheck_challenges[previous], previous_claim)
                    .map_err(kernel_error)?,
                None => sequence.message(previous_claim).map_err(kernel_error)?,
            };
            round_timings.push(RamHammingBooleanityRoundTiming {
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
            .ok_or_else(|| eval_error("RAM Hamming evaluator has no terminal challenge"))?;
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
        let expected_eq = self
            .relation
            .derive_output_term(
                &JoltDerivedId::from(RamHammingBooleanityPublic::EqCycle),
                &self.points,
                &output_points,
                &jolt_claims::NoChallenges::default(),
            )
            .map_err(kernel_error)?;
        if terminal.eq_cycle() != expected_eq {
            return Err(eval_error(
                "RAM Hamming Metal sequence produced a derived equality mismatch",
            ));
        }
        let output_claims = RamHammingBooleanityOutputClaims {
            ram_hamming_weight: terminal.hamming(),
        }
        .opening_values();
        let output_wall = output_started.elapsed();

        Ok(RamHammingBooleanityEvalSample {
            result: RamHammingBooleanityEvalResult {
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

fn mixed_parent_count(addresses: &[u32], child_width: usize) -> usize {
    addresses
        .chunks_exact(2 * child_width)
        .filter(|parent| {
            let first = parent[0] != NO_ACCESS;
            parent
                .iter()
                .skip(1)
                .any(|&address| (address != NO_ACCESS) != first)
        })
        .count()
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

fn kernel_error(error: impl ToString) -> RamHammingBooleanityEvalError {
    eval_error(error.to_string())
}

fn eval_error(message: impl Into<String>) -> RamHammingBooleanityEvalError {
    RamHammingBooleanityEvalError::Kernel(message.into())
}
