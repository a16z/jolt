use std::{mem::size_of, sync::Arc, time::Duration, time::Instant};

use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
use jolt_claims::protocols::jolt::relations::ram::RamRafEvaluationInputClaims;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRafEvaluationPublic};
use jolt_claims::{NoChallenges, OutputClaims as _};
use jolt_field::Zero as _;
use jolt_field::{CanonicalBytes as _, CanonicalEncoding as _, Prime128OffsetA7F7 as AkitaField};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::relations::{ConcreteSumcheck as _, SumcheckOutputPoints};
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_witness::JoltWitnessPlane;

use super::super::backend::MetalBackend;
use super::super::solinas::{
    DeviceInfo, MetalError, RamRafAddressPlane, RamRafAffineTail, RamRafConfig, RamRafCounters,
    RamRafSegmentedAddressPlane, RAM_RAF_ADDRESS_DOMAIN, RAM_RAF_INNER_LENGTH, RAM_RAF_TILE_COUNT,
};
use crate::metal::ram_records::{RamAccessColumns, RamAccessValues, NO_ACCESS};
use crate::optimized::OptimizedBackend;
use crate::ram_access::RamAccessTape;
use crate::{PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

#[derive(Debug, thiserror::Error)]
pub enum RamRafEvaluationEvalError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("RAM RAF evaluation evaluator failed: {0}")]
    Kernel(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamRafEvaluationEvalResult {
    round_polynomials: Vec<Vec<AkitaField>>,
    final_claim: AkitaField,
    ram_ra: AkitaField,
    unmap_address: AkitaField,
}

impl RamRafEvaluationEvalResult {
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
        write(&self.ram_ra.to_bytes_le_vec());
        write(&self.unmap_address.to_bytes_le_vec());
        hash
    }

    pub fn rounds(&self) -> usize {
        self.round_polynomials.len()
    }

    pub const fn output_claims(&self) -> usize {
        1
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRafEvaluationRoundTiming {
    pub round: usize,
    pub wall: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamRafEvaluationShapeSnapshot {
    pub cycles: usize,
    pub addresses: usize,
    pub accesses: usize,
    pub no_access_cycles: usize,
    pub access_density_ppm: usize,
    pub source_bytes: usize,
    pub resident_address_bytes: usize,
    pub segmented_borrowed_bytes: usize,
    pub segmented_bounded_addresses: usize,
    pub segmented_hot_addresses: usize,
    pub segmented_hot_chunks: usize,
    pub current_address_passes: usize,
    pub current_threadgroups: usize,
    pub current_address_loads: usize,
    pub current_address_load_bytes: usize,
    pub current_threadgroup_clear_bytes: usize,
    pub current_threadgroup_read_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct RamRafEvaluationCpuEvalSample {
    pub result: RamRafEvaluationEvalResult,
    pub member_wall: Duration,
    pub prepare_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
    pub round_timings: Vec<RamRafEvaluationRoundTiming>,
}

#[derive(Clone, Debug)]
pub struct RamRafEvaluationMetalEvalSample {
    pub result: RamRafEvaluationEvalResult,
    pub member_wall: Duration,
    pub sequence_setup_wall: Duration,
    pub sequence_wall: Duration,
    pub sequence_gpu_active: Duration,
    pub tail_setup_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
    pub counters: RamRafCounters,
    pub round_timings: Vec<RamRafEvaluationRoundTiming>,
}

pub struct RamRafEvaluationCpuMetalEvalFixture {
    backend: MetalBackend,
    columns: Arc<RamAccessColumns>,
    address_plane: Option<RamRafAddressPlane>,
    segmented_plane: Option<RamRafSegmentedAddressPlane>,
    relation: RamRafEvaluation<AkitaField>,
    claims: RamRafEvaluationInputClaims<AkitaField>,
    challenges: Vec<AkitaField>,
    shape: RamRafEvaluationShapeSnapshot,
    metal_route: &'static str,
    fixture_wall: Duration,
}

impl RamRafEvaluationCpuMetalEvalFixture {
    /// Collects the shared RAM source and resident address plane outside both arms.
    pub fn new(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        log_k: usize,
        lowest_address: u64,
        seed: u64,
    ) -> Result<Self, RamRafEvaluationEvalError> {
        if !(15..=28).contains(&log_t) || !(1..=32).contains(&log_k) {
            return Err(RamRafEvaluationEvalError::Kernel(
                "RAM RAF evaluator geometry is outside the supported range".to_owned(),
            ));
        }
        let fixture_started = Instant::now();
        let backend = MetalBackend::production()?;
        let cycles = 1usize << log_t;
        let addresses = 1usize << log_k;
        let mut source_session = ProofSession::default();
        let columns =
            RamAccessColumns::shared(&mut source_session, witness, log_t).map_err(kernel_error)?;
        columns
            .validate_addresses::<AkitaField>(addresses)
            .map_err(kernel_error)?;
        let tau_low = (0..log_t)
            .map(|index| challenge_field(seed ^ 0x3c6e_f372_fe94_f82b ^ index as u64))
            .collect::<Vec<_>>();
        let dimensions = ReadWriteDimensions::new(log_t, log_k, log_t, log_k);
        let raf_dimensions = RamRafEvaluationDimensions::try_from(dimensions)
            .map_err(|error| RamRafEvaluationEvalError::Kernel(error.to_string()))?;
        let relation =
            RamRafEvaluation::new(dimensions, raf_dimensions, log_k, lowest_address, tau_low);
        let challenges = (0..log_k)
            .map(|round| challenge_field(seed ^ 0x9e37_79b9_7f4a_7c15 ^ round as u64))
            .collect::<Vec<_>>();
        let claims = RamRafEvaluationInputClaims {
            ram_address: probe_input_claim(witness, &columns, &relation)?,
        };
        let accesses = columns
            .addresses
            .iter()
            .filter(|&&address| address != NO_ACCESS)
            .count();
        let (segmented_plane, segmented_route) = {
            let config = backend.config.ram_read_write;
            let (qualified, retained) = {
                let tape = source_session.state::<RamAccessTape>().ok_or_else(|| {
                    RamRafEvaluationEvalError::Kernel(
                        "RAM RAF evaluator lost the shared access certificate".to_owned(),
                    )
                })?;
                let qualified = cycles >= config.trace_cutoff_elements
                    && accesses >= config.minimum_accesses
                    && tape.increment_compatible()
                    && tape.ram_ra_compatible()
                    && tape.hamming_exact();
                let retained = tape.records().map(|records| {
                    records
                        .iter()
                        .map(|record| (record.cycle, record.address))
                        .unzip::<_, _, Vec<_>, Vec<_>>()
                });
                (qualified, retained)
            };
            if qualified {
                let values = source_session.state::<RamAccessValues>().ok_or_else(|| {
                    RamRafEvaluationEvalError::Kernel(
                        "RAM RAF evaluator lost the shared value columns".to_owned(),
                    )
                })?;
                let sequence = backend.context.prepare_ram_read_write_sequence(
                    &columns.addresses,
                    &values.pre_values,
                    &values.post_values,
                    log_t,
                    addresses,
                )?;
                (
                    Some(sequence.ram_raf_segmented_address_plane()),
                    Some("borrowed_address_segmented_v1"),
                )
            } else if let Some((cycle_ids, address_ids)) =
                retained.filter(|(ids, _)| !ids.is_empty())
            {
                (
                    Some(backend.context.prepare_ram_raf_segmented_accesses(
                        cycles,
                        addresses,
                        &cycle_ids,
                        &address_ids,
                    )?),
                    Some("retained_access_segmented_v1"),
                )
            } else {
                (None, None)
            }
        };
        let address_plane =
            if segmented_plane.is_none() && log_k == RAM_RAF_ADDRESS_DOMAIN.ilog2() as usize {
                Some(backend.context.prepare_ram_raf_addresses(
                    &columns.addresses,
                    backend.config.ram_raf_evaluation.dispatch,
                )?)
            } else {
                None
            };
        let metal_route = segmented_route.unwrap_or(if address_plane.is_some() {
            "six_tile_split_equality_v1"
        } else {
            "optimized_cpu_fallback_unsupported_address_domain"
        });
        let dense_route = usize::from(address_plane.is_some());
        let threadgroups = dense_route * (cycles / RAM_RAF_INNER_LENGTH) * RAM_RAF_TILE_COUNT;
        let current_address_loads = dense_route * cycles * RAM_RAF_TILE_COUNT;
        let accumulator_entries = dense_route * (cycles / RAM_RAF_INNER_LENGTH) * addresses;
        let accumulator_bytes = accumulator_entries * 5 * size_of::<u32>();
        let shape = RamRafEvaluationShapeSnapshot {
            cycles,
            addresses,
            accesses,
            no_access_cycles: cycles - accesses,
            access_density_ppm: accesses.saturating_mul(1_000_000) / cycles,
            source_bytes: cycles * size_of::<u32>(),
            resident_address_bytes: address_plane
                .as_ref()
                .map_or(0, RamRafAddressPlane::resident_bytes),
            segmented_borrowed_bytes: segmented_plane
                .as_ref()
                .map_or(0, RamRafSegmentedAddressPlane::borrowed_bytes),
            segmented_bounded_addresses: segmented_plane
                .as_ref()
                .map_or(0, RamRafSegmentedAddressPlane::bounded_address_count),
            segmented_hot_addresses: segmented_plane
                .as_ref()
                .map_or(0, RamRafSegmentedAddressPlane::hot_address_count),
            segmented_hot_chunks: segmented_plane
                .as_ref()
                .map_or(0, RamRafSegmentedAddressPlane::hot_message_chunk_count),
            current_address_passes: dense_route * RAM_RAF_TILE_COUNT,
            current_threadgroups: threadgroups,
            current_address_loads,
            current_address_load_bytes: current_address_loads * size_of::<u32>(),
            current_threadgroup_clear_bytes: accumulator_bytes,
            current_threadgroup_read_bytes: accumulator_bytes,
        };
        Ok(Self {
            backend,
            columns,
            address_plane,
            segmented_plane,
            relation,
            claims,
            challenges,
            shape,
            metal_route,
            fixture_wall: fixture_started.elapsed(),
        })
    }

    pub fn log_t(&self) -> usize {
        self.relation.read_write_dimensions().log_t()
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

    pub fn shape(&self) -> &RamRafEvaluationShapeSnapshot {
        &self.shape
    }

    pub fn device_info(&self) -> DeviceInfo {
        self.backend.context.device_info()
    }

    pub fn dispatch_config(&self) -> RamRafConfig {
        self.backend.config.ram_raf_evaluation.dispatch
    }

    pub fn metal_route(&self) -> &'static str {
        self.metal_route
    }

    pub fn run_cpu(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<RamRafEvaluationCpuEvalSample, RamRafEvaluationEvalError> {
        let points = RamRafEvaluationInputClaims::<Vec<AkitaField>>::default();
        let relation_challenges = NoChallenges::default();
        let inputs = || ProverInputs {
            relation: &self.relation,
            claims: &self.claims,
            points: &points,
            challenges: &relation_challenges,
        };
        let mut session = ProofSession::default();
        session.park(Arc::clone(&self.columns));
        let member_started = Instant::now();
        let prepare_started = Instant::now();
        let mut kernel = OptimizedBackend
            .prepare(&mut session, witness, inputs())
            .map_err(kernel_error)?;
        let prepare_wall = prepare_started.elapsed();
        let (round_polynomials, final_claim, round_timings, rounds_wall) =
            run_kernel_rounds(kernel.as_mut(), self.claims.ram_address, &self.challenges)?;

        let final_challenge = terminal_challenge(&self.challenges)?;
        let finish_started = Instant::now();
        kernel
            .finish_rounds(final_challenge)
            .map_err(kernel_error)?;
        let finish_wall = finish_started.elapsed();

        let output_started = Instant::now();
        let output_points = self
            .relation
            .derive_opening_points(&self.challenges, &points)
            .map_err(kernel_error)?;
        let output_claims = kernel
            .output_claims(&self.claims)
            .map_err(kernel_error)?
            .opening_values();
        kernel
            .validate_derived_tables(
                &self.relation,
                &points,
                &output_points,
                &relation_challenges,
            )
            .map_err(kernel_error)?;
        let unmap_address = expected_unmap(
            &self.relation,
            &points,
            &output_points,
            &relation_challenges,
        )?;
        let ram_ra = output_claims.first().copied().ok_or_else(|| {
            RamRafEvaluationEvalError::Kernel("CPU RAF output claim is missing".to_owned())
        })?;
        let output_wall = output_started.elapsed();
        Ok(RamRafEvaluationCpuEvalSample {
            result: RamRafEvaluationEvalResult {
                round_polynomials,
                final_claim,
                ram_ra,
                unmap_address,
            },
            member_wall: member_started.elapsed(),
            prepare_wall,
            rounds_wall,
            finish_wall,
            output_wall,
            round_timings,
        })
    }

    pub fn run_metal(&self) -> Result<RamRafEvaluationMetalEvalSample, RamRafEvaluationEvalError> {
        let member_started = Instant::now();
        let (sequence_setup_wall, sequence_wall, observation) = if let Some(source) =
            self.segmented_plane.clone()
        {
            let setup_started = Instant::now();
            let sequence = self
                .backend
                .context
                .prepare_ram_raf_segmented_sequence(source, self.relation.tau_low())?;
            let setup_wall = setup_started.elapsed();
            let sequence_started = Instant::now();
            let observation = sequence.execute_timed()?;
            (setup_wall, sequence_started.elapsed(), observation)
        } else {
            let address_plane = self.address_plane.clone().ok_or_else(|| {
                RamRafEvaluationEvalError::Kernel(
                    "production Metal RAF route does not support this address domain".to_owned(),
                )
            })?;
            let setup_started = Instant::now();
            let sequence = self.backend.context.prepare_ram_raf_sequence(
                address_plane,
                self.relation.tau_low(),
                self.backend.config.ram_raf_evaluation.dispatch,
            )?;
            let setup_wall = setup_started.elapsed();
            let sequence_started = Instant::now();
            let observation = sequence.execute_timed()?;
            (setup_wall, sequence_started.elapsed(), observation)
        };

        let tail_setup_started = Instant::now();
        let mut tail = RamRafAffineTail::new(observation.masses, self.relation.lowest_address())
            .map_err(|error| RamRafEvaluationEvalError::Kernel(error.to_string()))?;
        if tail.input_claim() != self.claims.ram_address {
            return Err(RamRafEvaluationEvalError::Kernel(
                "Metal RAF pushforward changed the input claim".to_owned(),
            ));
        }
        let tail_setup_wall = tail_setup_started.elapsed();

        let rounds_started = Instant::now();
        let mut previous_claim = self.claims.ram_address;
        let mut round_polynomials = Vec::with_capacity(self.challenges.len());
        let mut round_timings = Vec::with_capacity(self.challenges.len());
        for (round, &challenge) in self.challenges.iter().enumerate() {
            let round_started = Instant::now();
            if round != 0 {
                tail.bind(self.challenges[round - 1])
                    .map_err(|error| RamRafEvaluationEvalError::Kernel(error.to_string()))?;
            }
            let coefficients = tail
                .message(previous_claim)
                .map_err(|error| RamRafEvaluationEvalError::Kernel(error.to_string()))?
                .coefficients();
            let polynomial = UnivariatePoly::new(coefficients.to_vec());
            previous_claim = polynomial.evaluate(challenge);
            round_polynomials.push(coefficients.to_vec());
            round_timings.push(RamRafEvaluationRoundTiming {
                round,
                wall: round_started.elapsed(),
            });
        }
        let rounds_wall = rounds_started.elapsed();

        let final_challenge = terminal_challenge(&self.challenges)?;
        let finish_started = Instant::now();
        tail.bind(final_challenge)
            .map_err(|error| RamRafEvaluationEvalError::Kernel(error.to_string()))?;
        let output = tail
            .output()
            .map_err(|error| RamRafEvaluationEvalError::Kernel(error.to_string()))?;
        let finish_wall = finish_started.elapsed();

        let output_started = Instant::now();
        let points = RamRafEvaluationInputClaims::<Vec<AkitaField>>::default();
        let relation_challenges = NoChallenges::default();
        let output_points = self
            .relation
            .derive_opening_points(&self.challenges, &points)
            .map_err(kernel_error)?;
        let expected = expected_unmap(
            &self.relation,
            &points,
            &output_points,
            &relation_challenges,
        )?;
        if output.unmap_address != expected {
            return Err(RamRafEvaluationEvalError::Kernel(
                "Metal RAF affine tail changed the derived unmap evaluation".to_owned(),
            ));
        }
        let output_wall = output_started.elapsed();
        Ok(RamRafEvaluationMetalEvalSample {
            result: RamRafEvaluationEvalResult {
                round_polynomials,
                final_claim: previous_claim,
                ram_ra: output.ram_ra,
                unmap_address: output.unmap_address,
            },
            member_wall: member_started.elapsed(),
            sequence_setup_wall,
            sequence_wall,
            sequence_gpu_active: observation.gpu_active,
            tail_setup_wall,
            rounds_wall,
            finish_wall,
            output_wall,
            counters: observation.counters,
            round_timings,
        })
    }
}

type KernelRoundsResult = (
    Vec<Vec<AkitaField>>,
    AkitaField,
    Vec<RamRafEvaluationRoundTiming>,
    Duration,
);

fn run_kernel_rounds(
    kernel: &mut dyn SumcheckKernel<AkitaField, Relation = RamRafEvaluation<AkitaField>>,
    input_claim: AkitaField,
    challenges: &[AkitaField],
) -> Result<KernelRoundsResult, RamRafEvaluationEvalError> {
    let rounds_started = Instant::now();
    let mut previous_claim = input_claim;
    let mut round_polynomials = Vec::with_capacity(challenges.len());
    let mut round_timings = Vec::with_capacity(challenges.len());
    for (round, &challenge) in challenges.iter().enumerate() {
        let round_started = Instant::now();
        let bind = round.checked_sub(1).map(|previous| challenges[previous]);
        let polynomial = kernel
            .prove_round(bind, round, previous_claim)
            .map_err(kernel_error)?;
        previous_claim = polynomial.evaluate(challenge);
        round_polynomials.push(polynomial.coefficients().to_vec());
        round_timings.push(RamRafEvaluationRoundTiming {
            round,
            wall: round_started.elapsed(),
        });
    }
    Ok((
        round_polynomials,
        previous_claim,
        round_timings,
        rounds_started.elapsed(),
    ))
}

fn expected_unmap(
    relation: &RamRafEvaluation<AkitaField>,
    input_points: &RamRafEvaluationInputClaims<Vec<AkitaField>>,
    output_points: &SumcheckOutputPoints<AkitaField, RamRafEvaluation<AkitaField>>,
    challenges: &NoChallenges<AkitaField>,
) -> Result<AkitaField, RamRafEvaluationEvalError> {
    relation
        .derive_output_term(
            &JoltDerivedId::from(RamRafEvaluationPublic::UnmapAddress),
            input_points,
            output_points,
            challenges,
        )
        .map_err(kernel_error)
}

fn probe_input_claim(
    witness: &dyn JoltWitnessPlane<AkitaField>,
    columns: &Arc<RamAccessColumns>,
    relation: &RamRafEvaluation<AkitaField>,
) -> Result<AkitaField, RamRafEvaluationEvalError> {
    let claims = RamRafEvaluationInputClaims {
        ram_address: AkitaField::zero(),
    };
    let points = RamRafEvaluationInputClaims::<Vec<AkitaField>>::default();
    let challenges = NoChallenges::default();
    let mut session = ProofSession::default();
    session.park(Arc::clone(columns));
    let mut kernel = OptimizedBackend
        .prepare(
            &mut session,
            witness,
            ProverInputs {
                relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            },
        )
        .map_err(kernel_error)?;
    match kernel.prove_round(None, 0, AkitaField::zero()) {
        Ok(_) => Ok(AkitaField::zero()),
        Err(SumcheckError::RoundCheckFailed { actual, .. }) => Ok(actual),
        Err(error) => Err(RamRafEvaluationEvalError::Kernel(error.to_string())),
    }
}

fn terminal_challenge(challenges: &[AkitaField]) -> Result<AkitaField, RamRafEvaluationEvalError> {
    challenges.last().copied().ok_or_else(|| {
        RamRafEvaluationEvalError::Kernel("RAM RAF evaluator has no terminal challenge".to_owned())
    })
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

fn kernel_error(error: impl ToString) -> RamRafEvaluationEvalError {
    RamRafEvaluationEvalError::Kernel(error.to_string())
}
