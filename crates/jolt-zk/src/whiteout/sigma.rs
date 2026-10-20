//! The Σ-protocol for the residual relation (Whiteout Protocol
//! `wo:prot:sigma`; `specs/whiteout-sigma.md` §2).
//!
//! After the masked transcript, the prover holds `s` (laid out by the
//! segment table) with `N·s = t` and every lifted row satisfied. It:
//!
//! 1. checks, before any challenge: the segment alphabets, every row, `ct(g)`,
//!    and `N·s = t`;
//! 2. squeezes ℓ aggregation challenges `ρ` and asserts `ct(h) = 0` for
//!    `h = Q_μ(1, s)`;
//! 3. squeezes a fork seed `σ`, then per attempt samples a Gaussian mask `y`,
//!    commits to `V = (N y, L y + Q_μ(0,y), L s + B_μ((1,s),(0,y)), h)` as
//!    `t_D`, draws a secret 256-bit nonce `n`, derives `c` from the attempt
//!    fork `(σ, n, t_D)`, and applies the joint rejection rule to
//!    `z = y + c·s`;
//! 4. sends only the accepted attempt's `t_D` and `n`, then proves the
//!    response statement.
//!
//! Everything an attempt computes before its accept bit is constant time in
//! the attempt's secrets (y, n, c and what derives from them): the Gaussian
//! sampler, the Ajtai map, the branch-free lifts, the dense challenge
//! product, and the joint rejection rule. A rejected attempt leaves nothing
//! in the transcript.

use jolt_algebra::{CyclotomicRing, JoltField};
use jolt_crypto::challenge::{
    sample_indexed_challenges_from_seed, OperatorNormRejection, D64_SELECTIVE_L2_CHALLENGE_CONFIG,
};
use jolt_crypto::lattice::{
    AjtaiLevel, GadgetDigits, MatrixShape, MatrixView, PreparedMatrix, CHALLENGE_RING_DIM,
};
use jolt_transcript::{
    Channel, Fork, Sponge, VerifierTranscript, ATTEMPT_NONCE_LEN, FORK_SEED_LEN,
};

use super::lift::{ChallengeRing, LiftedRelation};
use super::linear::{LRow, RingLinearMap};
use super::params::{SigmaParams, RESPONSE_LOG_BASIS};
use super::response::{
    ResponseLayout, ResponseProver, ResponseSegment, ResponseStatement, FIELD_LEVELS,
};
use super::segments::SegmentTable;
use super::WhiteoutError;
use crate::masking::ZkError;

/// The public setup of one Σ-protocol shape.
pub struct SigmaSetup<F: JoltField, const D: usize> {
    pub table: SegmentTable,
    pub params: SigmaParams,
    /// `N` and its targets: the commitment openings.
    pub linear: RingLinearMap<F, D>,
    pub l_row: LRow<F, D>,
    /// `A_D`, binding on the radix-16 digits of `V`, over the challenge ring.
    pub v_level: AjtaiLevel,
    pub v_matrix: PreparedMatrix<F, CHALLENGE_RING_DIM>,
}

impl<F: JoltField, const D: usize> SigmaSetup<F, D> {
    /// The setup over `table`, with `A_D` of rank `v_rank` cut from
    /// `v_stream`. Every segment must have native dimension `D`, so that `N`
    /// is linear over the challenge ring.
    pub fn new(
        table: SegmentTable,
        linear: RingLinearMap<F, D>,
        l_row: LRow<F, D>,
        v_rank: usize,
        v_stream: &[F],
    ) -> Result<Self, WhiteoutError> {
        if !D.is_multiple_of(CHALLENGE_RING_DIM)
            || table
                .segments()
                .iter()
                .any(|segment| segment.native_dim != D)
        {
            return Err(WhiteoutError::InvalidRelation(
                "every segment must have the setup's native ring dimension",
            ));
        }
        let message_rings = (linear.num_outputs() * D) / CHALLENGE_RING_DIM + 3;
        let v_level = AjtaiLevel {
            matrix: MatrixShape {
                rows: v_rank,
                width: message_rings * FIELD_LEVELS,
                ring_dim: CHALLENGE_RING_DIM,
            },
            digits: GadgetDigits {
                log_basis: RESPONSE_LOG_BASIS,
                levels: FIELD_LEVELS,
            },
            message_rings,
            randomness_columns: 0,
        };
        let v_matrix = PreparedMatrix::new(&MatrixView::new(v_level.matrix, v_stream)?)?;
        Ok(Self {
            params: SigmaParams::derive(&table)?,
            table,
            linear,
            l_row,
            v_level,
            v_matrix,
        })
    }

    /// The layout of `e`.
    pub fn response_layout(&self) -> ResponseLayout {
        let mut segments: Vec<ResponseSegment> = self
            .table
            .segments()
            .iter()
            .zip(self.params.response_digits())
            .enumerate()
            .map(|(segment, (entry, &digits))| ResponseSegment::Response {
                segment,
                coefficients: entry.num_coefficients(),
                digits,
            })
            .collect();
        segments.push(ResponseSegment::V {
            coefficients: self.v_level.message_rings * CHALLENGE_RING_DIM,
        });
        segments.push(ResponseSegment::Theta);
        ResponseLayout { segments }
    }

    /// A hard bound on every coefficient of a response with its certified
    /// digits: the negative end of the widest balanced range.
    pub fn response_bound(&self) -> u64 {
        let radix = 1u64 << RESPONSE_LOG_BASIS;
        self.params
            .response_digits()
            .iter()
            .map(|&digits| (radix / 2) * (radix.saturating_pow(digits as u32) - 1) / (radix - 1))
            .max()
            .unwrap_or(0)
    }
}

/// The attempt seed: `(σ, n, t_D)` through the attempt fork.
fn attempt_seed<H: Sponge, F: JoltField>(
    sigma: &[u8; FORK_SEED_LEN],
    nonce: &[u8; ATTEMPT_NONCE_LEN],
    t_d: &[ChallengeRing<F>],
) -> [u8; 32] {
    let mut fork = Fork::<H>::attempt(sigma, nonce);
    let mut bytes = vec![0u8; F::NUM_BYTES];
    for coefficient in t_d.iter().flat_map(|ring| ring.coefficients()) {
        coefficient.to_bytes_le(&mut bytes);
        fork.absorb(&bytes);
    }
    fork.squeeze()
}

/// Verifies the Σ-protocol: squeezes `ρ` and `σ`, reads `t_D` and the
/// nonce, rebuilds `c`, and has `response` verify the response statement.
pub fn verify_sigma<F: JoltField, const D: usize, H: Sponge, P: ResponseProver<F, D>>(
    setup: &SigmaSetup<F, D>,
    relation: &LiftedRelation<F>,
    transcript: &mut VerifierTranscript<'_, H>,
    response: &P,
) -> Result<(), ZkError> {
    let rho: Vec<F> = (0..relation.labels().num_challenges())
        .map(|_| transcript.challenge::<F>())
        .collect();
    let sigma: [u8; FORK_SEED_LEN] = transcript.challenge_bytes();
    let t_d: Vec<F> = transcript.receive_n(setup.v_level.matrix.rows * CHALLENGE_RING_DIM)?;
    let nonce: [u8; ATTEMPT_NONCE_LEN] = transcript
        .receive_bytes(ATTEMPT_NONCE_LEN)?
        .try_into()
        .map_err(|_| WhiteoutError::ResponseRejected("nonce length"))?;
    let v_image: Vec<ChallengeRing<F>> = t_d
        .chunks_exact(CHALLENGE_RING_DIM)
        .map(CyclotomicRing::from_slice)
        .collect();
    let seed = attempt_seed::<H, F>(&sigma, &nonce, &v_image);
    let challenge = sample_indexed_challenges_from_seed(
        &seed,
        CHALLENGE_RING_DIM,
        1,
        &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
        Some(OperatorNormRejection::D64_SELECTIVE_L2),
    )
    .map_err(WhiteoutError::from)?
    .into_iter()
    .next()
    .ok_or(WhiteoutError::ResponseRejected("no challenge sampled"))?;
    let statement = ResponseStatement {
        setup,
        relation,
        challenge,
        rho,
        v_image,
    };
    response.verify_response(&statement, transcript)
}

#[cfg(feature = "whiteout-prover")]
pub use prover::prove_sigma;

#[cfg(feature = "whiteout-prover")]
mod prover {
    use jolt_algebra::{CyclotomicRing, JoltField};
    use std::ops::Range;

    use jolt_crypto::challenge::{sample_challenge_ct, SparseChallenge};
    use jolt_crypto::lattice::{commit, CHALLENGE_RING_DIM};
    use jolt_crypto::sample::{GaussianSampler, GaussianWidth, JointRejection, SampleError};
    use jolt_transcript::{Channel, ProverTranscript, Sponge, ATTEMPT_NONCE_LEN, FORK_SEED_LEN};
    use rand_core::{CryptoRng, RngCore};

    use super::super::challenge::{challenge_ring, challenge_times_i64};
    use super::super::lift::{
        constant_ring, AggregatedRelation, ChallengeRing, LiftedRelation, LinearParts,
    };
    use super::super::response::{ResponseProver, ResponseStatement, ResponseWitness};
    use super::super::WhiteoutError;
    use super::{attempt_seed, SigmaSetup};
    use crate::masking::ZkError;

    /// Every pad and witness digit is at most 2^7 in magnitude.
    const DIGIT_BOUND: u32 = 1 << 7;

    /// What an accepted attempt hands on.
    struct Accepted<F: JoltField> {
        t_d: Vec<ChallengeRing<F>>,
        nonce: [u8; ATTEMPT_NONCE_LEN],
        challenge: SparseChallenge,
        z: Vec<i64>,
        v: Vec<F>,
    }

    /// Per-proof state shared by the attempts.
    struct Prover<'a, F: JoltField, const D: usize> {
        setup: &'a SigmaSetup<F, D>,
        aggregated: AggregatedRelation<'a, F>,
        s: &'a [i64],
        parts_s: LinearParts<F>,
        l_s: ChallengeRing<F>,
        h: ChallengeRing<F>,
        /// `(coefficient range, sampler)` per Gaussian block.
        samplers: Vec<(Range<usize>, GaussianSampler)>,
        tail_bound: u32,
        sigma: [u8; FORK_SEED_LEN],
    }

    /// Proves the residual relation for `s`, returning the number of
    /// attempts (independent of the witness; the only attempt-dependent
    /// value that may leave).
    pub fn prove_sigma<F, const D: usize, H, R, P>(
        setup: &SigmaSetup<F, D>,
        relation: &LiftedRelation<F>,
        s: &[i64],
        transcript: &mut ProverTranscript<H>,
        rng: &mut R,
        response: &P,
    ) -> Result<usize, ZkError>
    where
        F: JoltField,
        H: Sponge,
        R: RngCore + CryptoRng,
        P: ResponseProver<F, D>,
    {
        // Before any challenge: alphabets, every row, ct(g), N·s = t.
        setup.table.check_private_vector(s)?;
        let s_view = setup.table.challenge_view(s)?;
        if let Some(row) = relation.first_violation(&s_view)? {
            return Err(WhiteoutError::ResidualUnsatisfied { row }.into());
        }
        if !setup
            .linear
            .apply(s, DIGIT_BOUND)?
            .iter()
            .eq(setup.linear.targets())
        {
            return Err(WhiteoutError::ResidualOpeningMismatch.into());
        }

        let rho: Vec<F> = (0..relation.labels().num_challenges())
            .map(|_| transcript.challenge::<F>())
            .collect();
        let aggregated = relation.aggregate(&rho)?;
        let parts_s = aggregated.linear_parts(&s_view)?;
        let h = aggregated.quadratic(&constant_ring(F::one()), &parts_s);
        if !h.constant_term().is_zero() {
            return Err(WhiteoutError::InvariantViolation("ct(h) ≠ 0").into());
        }
        let starts = setup.table.starts();
        let samplers = setup
            .params
            .blocks()
            .iter()
            .map(|block| {
                let (start, _) = starts
                    .get(block.segment)
                    .copied()
                    .ok_or(WhiteoutError::InvalidRelation("block segment missing"))?;
                Ok((
                    start..start + block.len,
                    GaussianSampler::new(GaussianWidth::from_multiplier(block.width)?),
                ))
            })
            .collect::<Result<Vec<_>, WhiteoutError>>()?;
        let tail_bound = samplers
            .iter()
            .map(|(_, sampler)| sampler.width().tail_bound())
            .max()
            .unwrap_or(0);
        let prover = Prover {
            setup,
            l_s: setup.l_row.apply(s, DIGIT_BOUND)?,
            aggregated,
            s,
            parts_s,
            h,
            samplers,
            tail_bound: u32::try_from(tail_bound)
                .map_err(|_| WhiteoutError::InvariantViolation("tail bound exceeds u32"))?,
            sigma: transcript.challenge_bytes(),
        };
        let rejection = setup.params.rejection()?;
        let max_attempts = setup.params.max_attempts()?;

        for attempt in 1..=max_attempts {
            let Some(accepted) = prover.attempt::<H, R>(&rejection, rng)? else {
                continue;
            };
            let flat: Vec<F> = accepted
                .t_d
                .iter()
                .flat_map(|ring| ring.coefficients().iter().copied())
                .collect();
            transcript.send_all(&flat);
            transcript.send_bytes(&accepted.nonce);
            let c = accepted
                .challenge
                .to_dense_ct()
                .map_err(WhiteoutError::from)?;
            let c_ring: ChallengeRing<F> = challenge_ring(&c);
            let parts_z = prover
                .aggregated
                .linear_parts(&setup.table.challenge_view(&accepted.z)?)?;
            let product = prover.aggregated.product_polynomial(&c_ring, &parts_z);
            let theta = product.get(CHALLENGE_RING_DIM..).unwrap_or_default();
            let witness = ResponseWitness::new(setup, &accepted.z, &accepted.v, theta)?;
            let statement = ResponseStatement {
                setup,
                relation,
                challenge: accepted.challenge,
                rho,
                v_image: accepted.t_d,
            };
            response.prove_response(&statement, &witness, transcript)?;
            return Ok(attempt);
        }
        Err(WhiteoutError::AttemptsExhausted.into())
    }

    impl<F: JoltField, const D: usize> Prover<'_, F, D> {
        /// One attempt. Every step before the accept bit is constant time in
        /// the attempt's secrets; only that bit is branched on.
        #[inline(never)]
        fn attempt<H: Sponge, R: RngCore + CryptoRng>(
            &self,
            rejection: &JointRejection,
            rng: &mut R,
        ) -> Result<Option<Accepted<F>>, WhiteoutError> {
            let table = &self.setup.table;
            let mut y = vec![0i64; table.num_coefficients()];
            for (range, sampler) in &self.samplers {
                sampler.fill(
                    rng,
                    y.get_mut(range.clone())
                        .ok_or(WhiteoutError::InvalidRelation("block outside s"))?,
                );
            }
            let parts_y = self.aggregated.linear_parts(&table.challenge_view(&y)?)?;
            let f = self.setup.linear.apply(&y, self.tail_bound)?;
            let e0 = self.setup.l_row.apply(&y, self.tail_bound)?
                + self.aggregated.quadratic_at_zero(&parts_y);
            let e1 = self.l_s + self.aggregated.polar(&self.parts_s, &parts_y);
            let mut v: Vec<F> = f
                .iter()
                .flat_map(|ring| ring.coefficients().iter().copied())
                .collect();
            for ring in [e0, e1, self.h] {
                v.extend_from_slice(ring.coefficients());
            }
            let v_rings: Vec<ChallengeRing<F>> = v
                .chunks_exact(CHALLENGE_RING_DIM)
                .map(CyclotomicRing::from_slice)
                .collect();
            let t_d = commit(&self.setup.v_level, &self.setup.v_matrix, &v_rings, &[])?;

            let mut nonce = [0u8; ATTEMPT_NONCE_LEN];
            rng.fill_bytes(&mut nonce);
            let challenge = sample_challenge_ct(&attempt_seed::<H, F>(&self.sigma, &nonce, &t_d))?;
            let c = challenge.to_dense_ct()?;

            // z = y + c·s on every coordinate; the shift on the Gaussian ones.
            let mut z = y;
            let mut z_gaussian = Vec::new();
            let mut v_gaussian = Vec::new();
            let mut offset = 0;
            for segment in table.segments() {
                let range = offset..offset + segment.num_coefficients();
                offset = range.end;
                let shift = challenge_times_i64(
                    &c,
                    segment.native_dim,
                    self.s
                        .get(range.clone())
                        .ok_or(WhiteoutError::InvalidRelation("segment outside s"))?,
                )?;
                let target = z
                    .get_mut(range.clone())
                    .ok_or(WhiteoutError::InvalidRelation("segment outside s"))?;
                for (dst, &value) in target.iter_mut().zip(&shift) {
                    *dst += value;
                }
                if self.samplers.iter().any(|(block, _)| *block == range) {
                    z_gaussian.extend_from_slice(target);
                    v_gaussian.extend(shift);
                }
            }
            let accept = match rejection.accept(&z_gaussian, &v_gaussian, rng) {
                Ok(accept) => accept,
                Err(SampleError::ShiftOutOfRange) => {
                    return Err(WhiteoutError::InvariantViolation(
                        "shift outside its bound: malformed witness",
                    ))
                }
                Err(error) => return Err(error.into()),
            };
            if !bool::from(accept) {
                return Ok(None);
            }
            Ok(Some(Accepted {
                t_d,
                nonce,
                challenge,
                z,
                v,
            }))
        }
    }
}
