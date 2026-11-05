//! Timing harness for `examples/sigma_dudect.rs` (feature
//! `whiteout-diagnostics`): each secret-dependent step of a Σ-protocol
//! attempt, on a synthetic instance, callable on chosen inputs. Diagnostics
//! only; not a protocol API.

use std::ops::Range;

use jolt_crypto::lattice::{commit, CHALLENGE_RING_DIM};
use jolt_crypto::sample::GaussianWidth;
use jolt_field::Ring;
use jolt_transcript::{
    protocol_tag, prover_transcript, Blake2b512, BytesMsg, ProverState, VerifierState,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use super::challenge::{challenge_times_i64_into, DenseChallenge};
use super::compression::ImageCompression;
pub use super::dit::with_dit;
use super::fixture::{Instance, RawImages, F};
use super::lift::{AggregatedRelation, ChallengeRing, ClaimTables, LiftedRelation, LinearParts};
use super::response::{ResponseProver, ResponseStatement, ResponseWitness};
use super::sigma::{attempt_seed, prove_sigma_with, SelfChecks};
use super::transport::{send_values, FORK_SEED_LEN};
use super::{SigmaSetup, WhiteoutError};
use crate::masking::ZkError;

/// A synthetic instance with its setup, relation and aggregation challenges.
pub struct AttemptHarness<const D: usize> {
    instance: Instance<D>,
    setup: SigmaSetup<F, D>,
    lifted: LiftedRelation<F>,
    rho: Vec<F>,
}

/// The linear parts of a mask, opaque.
pub struct Parts(LinearParts<F>);

/// The attempt's steps, with the per-proof state built once.
pub struct AttemptSteps<'a, const D: usize> {
    harness: &'a AttemptHarness<D>,
    aggregated: AggregatedRelation<'a, F>,
    tables: ClaimTables<F>,
    parts_s: LinearParts<F>,
}

impl<const D: usize> AttemptHarness<D> {
    pub fn new(seed: u64) -> Result<Self, WhiteoutError> {
        Self::scaled(seed, 1)
    }

    /// The instance with a witness `scale` times the base length.
    pub fn scaled(seed: u64, scale: usize) -> Result<Self, WhiteoutError> {
        let instance = Instance::<D>::scaled(seed, scale);
        let setup = instance.setup();
        let lifted = setup.lift(&instance.relation, &instance.claims)?;
        let rho = (0..lifted.labels().num_challenges())
            .map(|index| F::from_u64(index as u64 + 3))
            .collect();
        Ok(Self {
            instance,
            setup,
            lifted,
            rho,
        })
    }

    /// Coefficients of `s`.
    pub fn len(&self) -> usize {
        self.instance.s.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instance.s.is_empty()
    }

    /// The Gaussian coefficient ranges of `s`, where a mask is nonzero.
    pub fn gaussian_ranges(&self) -> Vec<Range<usize>> {
        self.setup
            .params()
            .blocks()
            .iter()
            .filter_map(|block| self.setup.table().placed().get(block.segment))
            .map(|placed| placed.coefficients())
            .collect()
    }

    /// The prover's hard bound on a mask coefficient, the sampler's tail cut.
    pub fn tail_bound(&self) -> Result<u64, WhiteoutError> {
        Ok(GaussianWidth::from_multiplier(self.setup.params().width())?.tail_bound())
    }

    /// Ring elements of `V`.
    pub fn v_rings(&self) -> usize {
        self.setup.v_level().message_rings
    }

    /// One whole Σ-protocol proof, the prover's self-checks included and
    /// the response proof left out; returns the attempt count.
    pub fn prove(&self, seed: u64) -> Result<usize, ZkError> {
        let mut transcript = prover_transcript::<Blake2b512>(
            &protocol_tag(b"jolt-zk/whiteout-diagnostics"),
            &BytesMsg(b"sigma".as_slice()),
        );
        for (_, payload) in self.setup.compressed_targets() {
            send_values(&mut transcript, payload);
        }
        prove_sigma_with(
            &self.setup,
            &self.lifted,
            &self.instance.s,
            &mut transcript,
            &mut ChaCha20Rng::seed_from_u64(seed),
            &NoResponse,
            SelfChecks::Run,
        )
    }

    /// Runs `run` on the steps, under DIT as the prover's attempts are.
    pub fn steps<R: Send>(
        &self,
        run: impl FnOnce(&AttemptSteps<'_, D>) -> Result<R, WhiteoutError> + Send,
    ) -> Result<R, WhiteoutError> {
        with_dit(|| {
            let aggregated = self.lifted.aggregate(&self.rho)?;
            let tables = self.lifted.claim_tables()?;
            let parts_s = aggregated.linear_parts(&tables, &self.instance.s[..], false)?;
            run(&AttemptSteps {
                harness: self,
                aggregated,
                tables,
                parts_s,
            })
        })?
    }
}

impl<const D: usize> AttemptSteps<'_, D> {
    /// The linear parts of the mask `y` (public coordinates skipped).
    pub fn linear_parts(&self, y: &[i64]) -> Result<Parts, WhiteoutError> {
        Ok(Parts(self.aggregated.linear_parts(
            &self.tables,
            y,
            true,
        )?))
    }

    pub fn quadratic_at_zero(&self, y: &Parts) -> ChallengeRing<F> {
        self.aggregated.quadratic_at_zero(&y.0)
    }

    pub fn polar(&self, y: &Parts) -> ChallengeRing<F> {
        self.aggregated.polar(&self.parts_s, &y.0)
    }

    /// `N·y`, the gadget block included, at the hard bound `bound`.
    pub fn ajtai(&self, y: &[i64], bound: u64) -> Result<usize, WhiteoutError> {
        Ok(self.harness.setup.linear().apply(y, bound)?.len())
    }

    pub fn l_row(&self, y: &[i64], bound: u64) -> Result<ChallengeRing<F>, WhiteoutError> {
        self.harness.setup.l_row().apply(y, bound)
    }

    /// `t_D = A_D G⁻¹(V)`.
    pub fn commit_v(&self, v: &[ChallengeRing<F>]) -> Result<Vec<ChallengeRing<F>>, WhiteoutError> {
        let setup = &self.harness.setup;
        Ok(commit(setup.v_level(), setup.v_matrix(), v, &[])?)
    }

    /// The attempt fork's seed for `(σ, n, cm(t_D))`, with raw images.
    pub fn seed(&self, nonce: &[u8; 32], t_d: &[ChallengeRing<F>]) -> [u8; 32] {
        let payload: Vec<F> = t_d
            .iter()
            .flat_map(|ring| ring.coefficients().iter().copied())
            .collect();
        attempt_seed::<Blake2b512, F>(&[5; FORK_SEED_LEN], nonce, &payload)
    }

    /// `z = y + c·s` and `v = c·s` on the Gaussian coordinates, into the
    /// caller's buffers.
    pub fn assemble(
        &self,
        c: &DenseChallenge,
        y: &[i64],
        z: &mut [i64],
        v: &mut [i64],
    ) -> Result<(), WhiteoutError> {
        let mut offset = 0;
        for range in self.harness.gaussian_ranges() {
            let end = offset + range.len();
            let (Some(z), Some(v), Some(s), Some(y)) = (
                z.get_mut(offset..end),
                v.get_mut(offset..end),
                self.harness.instance.s.get(range.clone()),
                y.get(range),
            ) else {
                return Err(WhiteoutError::InvalidRelation("assembly buffers"));
            };
            challenge_times_i64_into(c, D, s, v)?;
            for ((z, &y), &v) in z.iter_mut().zip(y).zip(v.iter()) {
                *z = y + v;
            }
            offset = end;
        }
        Ok(())
    }
}

/// Proves nothing about `e`: timing covers the Σ-protocol alone.
struct NoResponse;

impl<const D: usize> ResponseProver<F, D> for NoResponse {
    type Sponge = Blake2b512;

    fn compression(&self) -> &dyn ImageCompression<F> {
        &RawImages
    }

    fn prove_response(
        &self,
        _statement: &ResponseStatement<'_, F, D>,
        _witness: &ResponseWitness,
        _transcript: &mut ProverState<Blake2b512>,
    ) -> Result<(), ZkError> {
        Ok(())
    }

    fn verify_response(
        &self,
        _statement: &ResponseStatement<'_, F, D>,
        _transcript: &mut VerifierState<'_, Blake2b512>,
    ) -> Result<(), ZkError> {
        Err(WhiteoutError::ResponseRejected("diagnostics prove only").into())
    }
}

/// Ring elements of the challenge ring with `CHALLENGE_RING_DIM`
/// coefficients, for callers building `V`.
pub const RING: usize = CHALLENGE_RING_DIM;
