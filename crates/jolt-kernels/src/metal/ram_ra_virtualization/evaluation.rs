use std::{mem::size_of, sync::Arc, time::Duration, time::Instant};

use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::geometry::ram::RamRaVirtualizationDimensions;
use jolt_claims::protocols::jolt::relations::ram::RamRaVirtualizationInputClaims;
use jolt_claims::{NoChallenges, OutputClaims as _};
use jolt_field::{AkitaField, FixedBytes, TranscriptChallenge};
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;
use jolt_witness::JoltWitnessPlane;

use crate::metal::backend::MetalBackend;
use crate::metal::solinas::{
    DeviceInfo, MetalError, RamRafAddressPlane, RamRafConfig, RamRafError,
};
use crate::optimized::ram_trace::{RamAccessColumns, NO_ACCESS};
use crate::optimized::OptimizedBackend;
use crate::{PrepareKernel, ProofSession, ProverInputs};

/// Failure from the fixed CPU/Metal RAM RA virtualization evaluator.
#[derive(Debug, thiserror::Error)]
pub enum RamRaVirtualizationEvalError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("RAM RA virtualization evaluator failed: {0}")]
    Kernel(String),
}

/// Canonical outputs compared between the optimized CPU and Metal arms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamRaVirtualizationEvalResult {
    round_polynomials: Vec<Vec<AkitaField>>,
    final_claim: AkitaField,
    output_claims: Vec<AkitaField>,
}

impl RamRaVirtualizationEvalResult {
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

/// One timed sumcheck call. Rounds after zero include the preceding bind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaVirtualizationRoundTiming {
    pub round: usize,
    pub wall: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamRaVirtualizationShapeSnapshot {
    pub cycles: usize,
    pub accesses: usize,
    pub no_access_cycles: usize,
    pub maximum_address: Option<u32>,
    pub active_blocks_by_lazy_round: Vec<usize>,
    pub committed_factors: usize,
    pub committed_chunk_bits: usize,
    pub chunk_table_entries: usize,
    pub source_bytes: usize,
    pub resident_address_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct RamRaVirtualizationCpuEvalSample {
    pub result: RamRaVirtualizationEvalResult,
    pub member_wall: Duration,
    pub prepare_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
    pub round_timings: Vec<RamRaVirtualizationRoundTiming>,
}

pub type RamRaVirtualizationMetalEvalSample = RamRaVirtualizationCpuEvalSample;

/// Real-witness fixture shared by the isolated optimized CPU and Metal arms.
pub struct RamRaVirtualizationCpuMetalEvalFixture {
    backend: MetalBackend,
    columns: Arc<RamAccessColumns>,
    address_plane: Option<RamRafAddressPlane>,
    relation: RamRaVirtualization<AkitaField>,
    claims: RamRaVirtualizationInputClaims<AkitaField>,
    challenges: Vec<AkitaField>,
    shape: RamRaVirtualizationShapeSnapshot,
    fixture_wall: Duration,
}

impl RamRaVirtualizationCpuMetalEvalFixture {
    /// Collects the production RAM address source and resident plane once.
    pub fn new(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        log_k: usize,
        committed_chunk_bits: usize,
        seed: u64,
    ) -> Result<Self, RamRaVirtualizationEvalError> {
        if !(4..=28).contains(&log_t)
            || !(1..=32).contains(&log_k)
            || !(1..=32).contains(&committed_chunk_bits)
        {
            return Err(RamRaVirtualizationEvalError::Kernel(
                "RAM RA virtualization evaluator geometry is outside the supported range"
                    .to_owned(),
            ));
        }
        let fixture_started = Instant::now();
        let backend = MetalBackend::production()?;
        let cycles = 1usize << log_t;
        let address_count = 1usize << log_k;
        let mut source_session = ProofSession::default();
        let columns = RamAccessColumns::shared(&mut source_session, witness, log_t)
            .map_err(|error| RamRaVirtualizationEvalError::Kernel(error.to_string()))?;
        columns
            .validate_addresses::<AkitaField>(address_count)
            .map_err(|error| RamRaVirtualizationEvalError::Kernel(error.to_string()))?;
        let address_plane = match backend
            .context
            .prepare_ram_raf_addresses(&columns.addresses, RamRafConfig::default())
        {
            Ok(plane) => Some(plane),
            Err(MetalError::RamRaf(RamRafError::AddressOutsideDomain { .. })) => None,
            Err(error) => return Err(error.into()),
        };

        let r_address = (0..log_k)
            .map(|index| challenge_field(seed ^ 0xa54f_f53a_5f1d_36f1 ^ index as u64))
            .collect::<Vec<_>>();
        let r_cycle = (0..log_t)
            .map(|index| challenge_field(seed ^ 0x3c6e_f372_fe94_f82b ^ index as u64))
            .collect::<Vec<_>>();
        let chunks = committed_address_chunks(&r_address, committed_chunk_bits);
        let dimensions = RamRaVirtualizationDimensions::new(log_t, chunks.len());
        let relation =
            RamRaVirtualization::new(dimensions, r_address, r_cycle, committed_chunk_bits);
        let challenges = (0..log_t)
            .map(|round| challenge_field(seed ^ 0x9e37_79b9_7f4a_7c15 ^ round as u64))
            .collect::<Vec<_>>();
        let claims = RamRaVirtualizationInputClaims {
            ram_ra_reduced: probe_input_claim(witness, &columns, &relation)?,
        };
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
        let shape = RamRaVirtualizationShapeSnapshot {
            cycles,
            accesses,
            no_access_cycles: cycles - accesses,
            maximum_address,
            active_blocks_by_lazy_round: lazy_active_blocks(&columns.addresses),
            committed_factors: chunks.len(),
            committed_chunk_bits,
            chunk_table_entries: chunks.iter().map(|chunk| 1usize << chunk.len()).sum(),
            source_bytes: columns.addresses.len() * size_of::<u32>(),
            resident_address_bytes: address_plane
                .as_ref()
                .map_or(0, RamRafAddressPlane::resident_bytes),
        };
        Ok(Self {
            backend,
            columns,
            address_plane,
            relation,
            claims,
            challenges,
            shape,
            fixture_wall: fixture_started.elapsed(),
        })
    }

    pub fn log_t(&self) -> usize {
        self.relation.dimensions().log_t()
    }

    pub fn log_k(&self) -> usize {
        self.relation.ram_reduced_address().len()
    }

    pub fn cycles(&self) -> usize {
        self.shape.cycles
    }

    pub fn fixture_wall(&self) -> Duration {
        self.fixture_wall
    }

    pub fn shape(&self) -> &RamRaVirtualizationShapeSnapshot {
        &self.shape
    }

    pub fn device_info(&self) -> DeviceInfo {
        self.backend.context.device_info()
    }

    pub fn metal_route(&self) -> &'static str {
        let _ = self.address_plane.as_ref();
        if self.shape.cycles
            >= self
                .backend
                .config
                .ram_ra_virtualization
                .trace_cutoff_elements
            && (2..=3).contains(&self.shape.committed_factors)
            && self.shape.committed_chunk_bits == 8
        {
            "device_cycle_sequence_v1"
        } else {
            "optimized_cpu_host"
        }
    }

    pub fn run_cpu(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<RamRaVirtualizationCpuEvalSample, RamRaVirtualizationEvalError> {
        self.run(witness, false)
    }

    pub fn run_metal(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<RamRaVirtualizationMetalEvalSample, RamRaVirtualizationEvalError> {
        self.run(witness, true)
    }

    fn run(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        metal: bool,
    ) -> Result<RamRaVirtualizationCpuEvalSample, RamRaVirtualizationEvalError> {
        let points = RamRaVirtualizationInputClaims::<Vec<AkitaField>>::default();
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
        let mut kernel = if metal {
            self.backend.prepare(&mut session, witness, inputs())
        } else {
            OptimizedBackend.prepare(&mut session, witness, inputs())
        }
        .map_err(|error| RamRaVirtualizationEvalError::Kernel(error.to_string()))?;
        let prepare_wall = prepare_started.elapsed();

        let rounds_started = Instant::now();
        let mut previous_claim = self.claims.ram_ra_reduced;
        let mut round_polynomials = Vec::with_capacity(self.challenges.len());
        let mut round_timings = Vec::with_capacity(self.challenges.len());
        for round in 0..self.challenges.len() {
            let round_started = Instant::now();
            let bind = round
                .checked_sub(1)
                .map(|previous| self.challenges[previous]);
            let polynomial = kernel
                .prove_round(bind, round, previous_claim)
                .map_err(|error| RamRaVirtualizationEvalError::Kernel(error.to_string()))?;
            round_timings.push(RamRaVirtualizationRoundTiming {
                round,
                wall: round_started.elapsed(),
            });
            previous_claim = polynomial.evaluate(self.challenges[round]);
            round_polynomials.push(polynomial.coefficients().to_vec());
        }
        let rounds_wall = rounds_started.elapsed();

        let final_challenge = self.challenges.last().copied().ok_or_else(|| {
            RamRaVirtualizationEvalError::Kernel(
                "RAM RA virtualization evaluator has no terminal challenge".to_owned(),
            )
        })?;
        let finish_started = Instant::now();
        kernel
            .finish_rounds(final_challenge)
            .map_err(|error| RamRaVirtualizationEvalError::Kernel(error.to_string()))?;
        let finish_wall = finish_started.elapsed();

        let output_started = Instant::now();
        let output_points = self
            .relation
            .derive_opening_points(&self.challenges, &points)
            .map_err(|error| RamRaVirtualizationEvalError::Kernel(error.to_string()))?;
        let output_claims = kernel
            .output_claims(&self.claims)
            .map_err(|error| RamRaVirtualizationEvalError::Kernel(error.to_string()))?;
        kernel
            .validate_derived_tables(
                &self.relation,
                &points,
                &output_points,
                &relation_challenges,
            )
            .map_err(|error| RamRaVirtualizationEvalError::Kernel(error.to_string()))?;
        let output_claims = output_claims.opening_values();
        let output_wall = output_started.elapsed();

        Ok(RamRaVirtualizationCpuEvalSample {
            result: RamRaVirtualizationEvalResult {
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

fn probe_input_claim(
    witness: &dyn JoltWitnessPlane<AkitaField>,
    columns: &Arc<RamAccessColumns>,
    relation: &RamRaVirtualization<AkitaField>,
) -> Result<AkitaField, RamRaVirtualizationEvalError> {
    let claims = RamRaVirtualizationInputClaims {
        ram_ra_reduced: AkitaField::zero(),
    };
    let points = RamRaVirtualizationInputClaims::<Vec<AkitaField>>::default();
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
        .map_err(|error| RamRaVirtualizationEvalError::Kernel(error.to_string()))?;
    match kernel.prove_round(None, 0, AkitaField::zero()) {
        Ok(_) => Ok(AkitaField::zero()),
        Err(SumcheckError::RoundCheckFailed { actual, .. }) => Ok(actual),
        Err(error) => Err(RamRaVirtualizationEvalError::Kernel(error.to_string())),
    }
}

fn challenge_field(seed: u64) -> AkitaField {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&splitmix(seed).to_le_bytes());
    bytes[8..].copy_from_slice(&splitmix(seed ^ 0xd1b5_4a32_d192_ed03).to_le_bytes());
    AkitaField::from_challenge_bytes(&bytes)
}

fn lazy_active_blocks(addresses: &[u32]) -> Vec<usize> {
    let mut counts = [0usize; 4];
    for block in addresses.chunks(16) {
        let mut pairs = [false; 8];
        for (pair, addresses) in pairs.iter_mut().zip(block.chunks(2)) {
            *pair = addresses.iter().any(|&address| address != NO_ACCESS);
        }
        counts[0] += pairs.iter().filter(|&&active| active).count();
        let quads = [
            pairs[0] || pairs[1],
            pairs[2] || pairs[3],
            pairs[4] || pairs[5],
            pairs[6] || pairs[7],
        ];
        counts[1] += quads.iter().filter(|&&active| active).count();
        let octets = [quads[0] || quads[1], quads[2] || quads[3]];
        counts[2] += octets.iter().filter(|&&active| active).count();
        counts[3] += usize::from(octets[0] || octets[1]);
    }
    counts.to_vec()
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
