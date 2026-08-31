use std::time::{Duration, Instant};

use jolt_claims::protocols::jolt::geometry::spartan::SpartanProductDimensions;
use jolt_claims::{NoChallenges, OutputClaims as _};
use jolt_field::{Field as _, One as _, Zero as _};
use jolt_field::{FixedBytes, Prime128OffsetA7F7 as AkitaField, TranscriptChallenge};
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage2::product_remainder::{
    product_remainder_input_values_from_uniskip_output, ProductRemainder,
    ProductRemainderInputClaims,
};
use jolt_witness::{witnesses::SpartanOuterRow, JoltWitnessPlane};

use super::{
    MetalBackend, MetalInstructionClaimResidentRows, OptimizedProductRemainder,
    OptimizedProductUniskip, ProductRemainderSequence,
};
use crate::metal::backend::MetalConfig;
use crate::metal::solinas::{
    DeviceInfo, MetalError, ProductRemainderStorageLayout, SpartanOuterUniskipRows,
};
use crate::optimized::spartan_outer::prepare_metal_spartan_outer_witness_rows;
use crate::uniskip::UniskipKernel;
use crate::{PrepareKernel, ProofSession, ProverInputs};

#[derive(Debug, thiserror::Error)]
pub enum ProductRemainderEvalError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("Product remainder evaluator failed: {0}")]
    Kernel(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductRemainderEvalResult {
    round_polynomials: Vec<Vec<AkitaField>>,
    final_claim: AkitaField,
    output_claims: Vec<AkitaField>,
}

impl ProductRemainderEvalResult {
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

#[derive(Clone, Copy, Debug)]
pub struct ProductRemainderRoundTiming {
    pub round: usize,
    pub wall: Duration,
}

#[derive(Clone, Debug)]
pub struct ProductRemainderCpuEvalSample {
    pub result: ProductRemainderEvalResult,
    pub uniskip_setup_wall: Duration,
    pub member_wall: Duration,
    pub prepare_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
    pub round_timings: Vec<ProductRemainderRoundTiming>,
}

#[derive(Clone, Debug)]
pub struct ProductRemainderMetalEvalSample {
    pub result: ProductRemainderEvalResult,
    pub upstream_storage_wall: Duration,
    pub sequence_setup_wall: Duration,
    pub uniskip_setup_wall: Duration,
    pub member_wall: Duration,
    pub prepare_wall: Duration,
    pub rounds_wall: Duration,
    pub finish_wall: Duration,
    pub output_wall: Duration,
    pub round_timings: Vec<ProductRemainderRoundTiming>,
}

impl ProductRemainderMetalEvalSample {
    pub fn charged_wall(&self) -> Duration {
        self.sequence_setup_wall
            .checked_add(self.member_wall)
            .unwrap_or(Duration::MAX)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductRemainderShapeSnapshot {
    pub cycles: usize,
    pub source_row_bytes: usize,
    pub source_bytes: u64,
    pub state_a_bytes: usize,
    pub state_b_bytes: usize,
    pub workspace_bytes: usize,
    pub cpu_tail_elements: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductRemainderNumericWidthSnapshot {
    pub samples: usize,
    pub left_zero: usize,
    pub left_u32: usize,
    pub left_lookup_zero: usize,
    pub left_lookup_u16: usize,
    pub left_lookup_u32: usize,
    pub lookup_zero: usize,
    pub lookup_u16: usize,
    pub lookup_u32: usize,
    pub right_zero: usize,
    pub right_u32: usize,
    pub right_u64: usize,
}

pub struct ProductRemainderCpuMetalEvalFixture {
    backend: MetalBackend,
    rows: SpartanOuterUniskipRows,
    log_t: usize,
    tau_low: Vec<AkitaField>,
    relation: ProductRemainder<AkitaField>,
    input_claim: AkitaField,
    challenges: Vec<AkitaField>,
    fixture_wall: Duration,
    shape: ProductRemainderShapeSnapshot,
    numeric_widths: ProductRemainderNumericWidthSnapshot,
    materialize_threads_per_threadgroup: usize,
    transition_threads_per_threadgroup: usize,
    openings_threads_per_threadgroup: usize,
}

impl ProductRemainderCpuMetalEvalFixture {
    pub fn new(
        witness: &dyn JoltWitnessPlane<AkitaField>,
        log_t: usize,
        seed: u64,
        materialize_threads_per_threadgroup: Option<usize>,
        transition_threads_per_threadgroup: Option<usize>,
        openings_threads_per_threadgroup: Option<usize>,
    ) -> Result<Self, ProductRemainderEvalError> {
        if !(4..=28).contains(&log_t) {
            return Err(ProductRemainderEvalError::Kernel(
                "log_t must be between 4 and 28".to_owned(),
            ));
        }
        let fixture_started = Instant::now();
        let cycles = 1usize << log_t;
        let numeric_widths = sample_numeric_widths(witness, cycles)?;
        let mut config = MetalConfig::production();
        if let Some(threads) = materialize_threads_per_threadgroup {
            config
                .spartan_product_remainder
                .dispatch
                .materialize_threads_per_threadgroup = Some(threads);
        }
        if let Some(threads) = transition_threads_per_threadgroup {
            config
                .spartan_product_remainder
                .dispatch
                .transition_threads_per_threadgroup = Some(threads);
        }
        if let Some(threads) = openings_threads_per_threadgroup {
            config
                .spartan_product_remainder
                .dispatch
                .openings_threads_per_threadgroup = Some(threads);
        }
        let materialize_threads_per_threadgroup = config
            .spartan_product_remainder
            .dispatch
            .materialize_threads_per_threadgroup
            .ok_or_else(|| {
                ProductRemainderEvalError::Kernel(
                    "evaluator requires a fixed materialize threadgroup width".to_owned(),
                )
            })?;
        let transition_threads_per_threadgroup = config
            .spartan_product_remainder
            .dispatch
            .transition_threads_per_threadgroup
            .ok_or_else(|| {
                ProductRemainderEvalError::Kernel(
                    "evaluator requires a fixed transition threadgroup width".to_owned(),
                )
            })?;
        let openings_threads_per_threadgroup = config
            .spartan_product_remainder
            .dispatch
            .openings_threads_per_threadgroup
            .ok_or_else(|| {
                ProductRemainderEvalError::Kernel(
                    "evaluator requires a fixed openings threadgroup width".to_owned(),
                )
            })?;
        let cpu_tail_elements = config.spartan_product_remainder.cpu_tail_elements;
        let backend = MetalBackend::new(config)?;
        let rows = prepare_metal_spartan_outer_witness_rows(&backend.context, witness, cycles)
            .map_err(kernel_error)?;
        let tau_low = (0..log_t)
            .map(|index| challenge_field(seed ^ 0x3c6e_f372_fe94_f82b ^ index as u64))
            .collect::<Vec<_>>();
        let tau_high = challenge_field(seed ^ 0xa54f_f53a_5f1d_36f1);
        let uniskip_challenge = challenge_field(seed ^ 0x510e_527f_ade6_82d1);
        let relation = ProductRemainder::new(
            SpartanProductDimensions::new(log_t),
            uniskip_challenge,
            tau_high,
            tau_low.clone(),
        );
        let input_claim = challenge_field(seed ^ 0xbb67_ae85_84ca_a73b);
        if input_claim == AkitaField::zero() {
            return Err(ProductRemainderEvalError::Kernel(
                "evaluator seed produced a zero input claim".to_owned(),
            ));
        }
        let challenges = (0..log_t)
            .map(|round| challenge_field(seed ^ 0x9e37_79b9_7f4a_7c15 ^ round as u64))
            .collect::<Vec<_>>();
        let e_in_capacity = 1usize << (log_t / 2);
        let e_out_capacity = cycles / e_in_capacity;
        let layout = ProductRemainderStorageLayout::new(cycles, e_in_capacity, e_out_capacity)
            .map_err(|error| ProductRemainderEvalError::Kernel(error.to_string()))?;
        let shape = ProductRemainderShapeSnapshot {
            cycles,
            source_row_bytes: 160,
            source_bytes: 160u64 * cycles as u64,
            state_a_bytes: layout.state_a_fields() * 16,
            state_b_bytes: layout.state_b_fields() * 16,
            workspace_bytes: layout.workspace_bytes(),
            cpu_tail_elements,
        };
        Ok(Self {
            backend,
            rows,
            log_t,
            tau_low,
            relation,
            input_claim,
            challenges,
            fixture_wall: fixture_started.elapsed(),
            shape,
            numeric_widths,
            materialize_threads_per_threadgroup,
            transition_threads_per_threadgroup,
            openings_threads_per_threadgroup,
        })
    }

    pub fn fixture_wall(&self) -> Duration {
        self.fixture_wall
    }

    pub fn log_t(&self) -> usize {
        self.log_t
    }

    pub fn cycles(&self) -> usize {
        self.shape.cycles
    }

    pub const fn shape(&self) -> ProductRemainderShapeSnapshot {
        self.shape
    }

    pub const fn numeric_widths(&self) -> ProductRemainderNumericWidthSnapshot {
        self.numeric_widths
    }

    pub const fn transition_threads_per_threadgroup(&self) -> usize {
        self.transition_threads_per_threadgroup
    }

    pub const fn materialize_threads_per_threadgroup(&self) -> usize {
        self.materialize_threads_per_threadgroup
    }

    pub const fn openings_threads_per_threadgroup(&self) -> usize {
        self.openings_threads_per_threadgroup
    }

    pub fn device_info(&self) -> DeviceInfo {
        self.backend.context.device_info()
    }

    pub const fn metal_route(&self) -> &'static str {
        "resident_stage1_standalone_no_prefetch"
    }

    pub fn run_cpu(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<ProductRemainderCpuEvalSample, ProductRemainderEvalError> {
        let setup_started = Instant::now();
        let mut session = ProofSession::default();
        OptimizedProductUniskip
            .prepare(&mut session, self.log_t, &self.tau_low, witness)
            .map_err(kernel_error)?;
        let uniskip_setup_wall = setup_started.elapsed();

        let claims = product_remainder_input_values_from_uniskip_output(self.input_claim);
        let points = ProductRemainderInputClaims::<Vec<AkitaField>>::default();
        let relation_challenges = NoChallenges::<AkitaField>::default();
        let member_started = Instant::now();
        let prepare_started = Instant::now();
        let kernel = OptimizedProductRemainder
            .prepare(
                &mut session,
                witness,
                ProverInputs {
                    relation: &self.relation,
                    claims: &claims,
                    points: &points,
                    challenges: &relation_challenges,
                },
            )
            .map_err(kernel_error)?;
        let prepare_wall = prepare_started.elapsed();
        let execution = self.execute(kernel, &claims, &points, &relation_challenges)?;
        Ok(ProductRemainderCpuEvalSample {
            result: execution.result,
            uniskip_setup_wall,
            member_wall: member_started.elapsed(),
            prepare_wall,
            rounds_wall: execution.rounds_wall,
            finish_wall: execution.finish_wall,
            output_wall: execution.output_wall,
            round_timings: execution.round_timings,
        })
    }

    pub fn run_metal(
        &self,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<ProductRemainderMetalEvalSample, ProductRemainderEvalError> {
        let mut session = ProofSession::default();
        session.park(self.rows.clone());
        let upstream_storage_started = Instant::now();
        let outer_storage = self
            .backend
            .context
            .prepare_outer_remainder_sequence_storage(
                self.cycles(),
                self.backend.config.spartan_outer_remainder.dispatch,
            )?;
        let upstream_storage_wall = upstream_storage_started.elapsed();
        session.park(outer_storage);
        let sequence_setup_started = Instant::now();
        self.backend
            .prepare_product_remainder_witness(&mut session, self.log_t, witness)
            .map_err(kernel_error)?;
        let sequence_setup_wall = sequence_setup_started.elapsed();
        let sequence = session.state::<ProductRemainderSequence>().ok_or_else(|| {
            ProductRemainderEvalError::Kernel(
                "Metal evaluator did not admit the resident sequence".to_owned(),
            )
        })?;
        if sequence.storage_layout().rows() != self.cycles() {
            return Err(ProductRemainderEvalError::Kernel(
                "Metal evaluator admitted the wrong sequence shape".to_owned(),
            ));
        }

        let uniskip_started = Instant::now();
        <MetalBackend as UniskipKernel<AkitaField, ProductRemainder<AkitaField>>>::prepare(
            &self.backend,
            &mut session,
            self.log_t,
            &self.tau_low,
            witness,
        )
        .map_err(kernel_error)?;
        let uniskip_setup_wall = uniskip_started.elapsed();
        drop(session.take::<MetalInstructionClaimResidentRows>());

        let claims = product_remainder_input_values_from_uniskip_output(self.input_claim);
        let points = ProductRemainderInputClaims::<Vec<AkitaField>>::default();
        let relation_challenges = NoChallenges::<AkitaField>::default();
        let member_started = Instant::now();
        let prepare_started = Instant::now();
        let kernel =
            <MetalBackend as PrepareKernel<AkitaField, ProductRemainder<AkitaField>>>::prepare(
                &self.backend,
                &mut session,
                witness,
                ProverInputs {
                    relation: &self.relation,
                    claims: &claims,
                    points: &points,
                    challenges: &relation_challenges,
                },
            )
            .map_err(kernel_error)?;
        let prepare_wall = prepare_started.elapsed();
        let execution = self.execute(kernel, &claims, &points, &relation_challenges)?;
        Ok(ProductRemainderMetalEvalSample {
            result: execution.result,
            upstream_storage_wall,
            sequence_setup_wall,
            uniskip_setup_wall,
            member_wall: member_started.elapsed(),
            prepare_wall,
            rounds_wall: execution.rounds_wall,
            finish_wall: execution.finish_wall,
            output_wall: execution.output_wall,
            round_timings: execution.round_timings,
        })
    }

    fn execute(
        &self,
        mut kernel: Box<
            dyn crate::SumcheckKernel<AkitaField, Relation = ProductRemainder<AkitaField>>,
        >,
        claims: &jolt_verifier::stages::relations::SumcheckInputClaims<
            AkitaField,
            ProductRemainder<AkitaField>,
        >,
        points: &ProductRemainderInputClaims<Vec<AkitaField>>,
        relation_challenges: &NoChallenges<AkitaField>,
    ) -> Result<ProductRemainderExecution, ProductRemainderEvalError> {
        let rounds_started = Instant::now();
        let mut previous_claim = self.input_claim;
        let mut round_polynomials = Vec::with_capacity(self.challenges.len());
        let mut round_timings = Vec::with_capacity(self.challenges.len());
        for (round, &challenge) in self.challenges.iter().enumerate() {
            let round_started = Instant::now();
            let bind = round
                .checked_sub(1)
                .map(|previous| self.challenges[previous]);
            let polynomial = kernel
                .prove_round(bind, round, previous_claim)
                .map_err(kernel_error)?;
            round_timings.push(ProductRemainderRoundTiming {
                round,
                wall: round_started.elapsed(),
            });
            previous_claim = polynomial.evaluate(challenge);
            round_polynomials.push(polynomial.coefficients().to_vec());
        }
        let rounds_wall = rounds_started.elapsed();

        let final_challenge = self.challenges.last().copied().ok_or_else(|| {
            ProductRemainderEvalError::Kernel(
                "Product remainder evaluator has no terminal challenge".to_owned(),
            )
        })?;
        let finish_started = Instant::now();
        kernel
            .finish_rounds(final_challenge)
            .map_err(kernel_error)?;
        let finish_wall = finish_started.elapsed();

        let output_started = Instant::now();
        let output_points = self
            .relation
            .derive_opening_points(&self.challenges, points)
            .map_err(kernel_error)?;
        let output_claims = kernel.output_claims(claims).map_err(kernel_error)?;
        kernel
            .validate_derived_tables(&self.relation, points, &output_points, relation_challenges)
            .map_err(kernel_error)?;
        let output_claims = output_claims.opening_values();
        let output_wall = output_started.elapsed();

        Ok(ProductRemainderExecution {
            result: ProductRemainderEvalResult {
                round_polynomials,
                final_claim: previous_claim,
                output_claims,
            },
            rounds_wall,
            finish_wall,
            output_wall,
            round_timings,
        })
    }
}

struct ProductRemainderExecution {
    result: ProductRemainderEvalResult,
    rounds_wall: Duration,
    finish_wall: Duration,
    output_wall: Duration,
    round_timings: Vec<ProductRemainderRoundTiming>,
}

fn kernel_error(error: impl ToString) -> ProductRemainderEvalError {
    ProductRemainderEvalError::Kernel(error.to_string())
}

fn sample_numeric_widths(
    witness: &dyn JoltWitnessPlane<AkitaField>,
    cycles: usize,
) -> Result<ProductRemainderNumericWidthSnapshot, ProductRemainderEvalError> {
    const MAX_SAMPLES: usize = 1 << 18;
    let owned = witness.owned_rows().ok_or_else(|| {
        ProductRemainderEvalError::Kernel(
            "numeric-width sampling requires slice-backed witness rows".to_owned(),
        )
    })?;
    if owned.cycles() < cycles {
        return Err(ProductRemainderEvalError::Kernel(
            "numeric-width sampling witness is shorter than the Product domain".to_owned(),
        ));
    }
    let rows = owned.view();
    let samples = cycles.min(MAX_SAMPLES);
    let mut snapshot = ProductRemainderNumericWidthSnapshot {
        samples,
        left_zero: 0,
        left_u32: 0,
        left_lookup_zero: 0,
        left_lookup_u16: 0,
        left_lookup_u32: 0,
        lookup_zero: 0,
        lookup_u16: 0,
        lookup_u32: 0,
        right_zero: 0,
        right_u32: 0,
        right_u64: 0,
    };
    for sample in 0..samples {
        let index = sample * (cycles / samples);
        let row = rows
            .window::<SpartanOuterRow>(index)
            .map_err(kernel_error)?;
        let left = row.left_instruction_input.0;
        let left_lookup = row.left_lookup_operand.0;
        let lookup = row.lookup_output.0;
        let right = row.right_instruction_input.0.unsigned_abs();
        snapshot.left_zero += usize::from(left == 0);
        snapshot.left_u32 += usize::from(u32::try_from(left).is_ok());
        snapshot.left_lookup_zero += usize::from(left_lookup == 0);
        snapshot.left_lookup_u16 += usize::from(u16::try_from(left_lookup).is_ok());
        snapshot.left_lookup_u32 += usize::from(u32::try_from(left_lookup).is_ok());
        snapshot.lookup_zero += usize::from(lookup == 0);
        snapshot.lookup_u16 += usize::from(u16::try_from(lookup).is_ok());
        snapshot.lookup_u32 += usize::from(u32::try_from(lookup).is_ok());
        snapshot.right_zero += usize::from(right == 0);
        snapshot.right_u32 += usize::from(u32::try_from(right).is_ok());
        snapshot.right_u64 += usize::from(u64::try_from(right).is_ok());
    }
    Ok(snapshot)
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
