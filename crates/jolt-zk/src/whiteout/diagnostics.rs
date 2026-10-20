//! Timing harness for `examples/sigma_dudect.rs` (feature
//! `whiteout-diagnostics`): each secret-dependent step of a Σ-protocol
//! attempt, on a synthetic instance, callable on chosen inputs. Diagnostics
//! only; not a protocol API.

use std::ops::Range;

use jolt_algebra::Ring;
use jolt_crypto::lattice::{commit, CHALLENGE_RING_DIM};
use jolt_transcript::{
    Blake2b512, ProtocolId, ProverTranscript, Sponge, VerifierTranscript, ATTEMPT_NONCE_LEN,
    FORK_SEED_LEN,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use super::challenge::{challenge_times_i64_into, DenseChallenge};
use super::fixture::{Instance, F};
use super::lift::{AggregatedRelation, ChallengeRing, ClaimTables, LiftedRelation, LinearParts};
use super::response::{ResponseProver, ResponseStatement, ResponseWitness};
use super::sigma::{attempt_seed, prove_sigma_with, SelfChecks};
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

    /// Ring elements of `V`.
    pub fn v_rings(&self) -> usize {
        self.setup.v_level().message_rings
    }

    /// One whole Σ-protocol proof, the prover's self-checks included and
    /// the response proof (W5) left out; returns the attempt count.
    pub fn prove(&self, seed: u64) -> Result<usize, ZkError> {
        let protocol = ProtocolId::new::<Blake2b512>("jolt-zk/whiteout-diagnostics");
        let mut transcript = ProverTranscript::<Blake2b512>::new(&protocol, b"sigma");
        for ring in self.instance.u.iter().chain(&self.instance.t_p) {
            transcript.send_all(ring.coefficients());
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

    pub fn steps<R>(
        &self,
        run: impl FnOnce(&AttemptSteps<'_, D>) -> Result<R, WhiteoutError>,
    ) -> Result<R, WhiteoutError> {
        let aggregated = self.lifted.aggregate(&self.rho)?;
        let tables = self.lifted.claim_tables()?;
        let parts_s = aggregated.linear_parts(&tables, &self.instance.s, false)?;
        run(&AttemptSteps {
            harness: self,
            aggregated,
            tables,
            parts_s,
        })
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
    pub fn ajtai(&self, y: &[i64], bound: u32) -> Result<usize, WhiteoutError> {
        Ok(self.harness.setup.linear().apply(y, bound)?.len())
    }

    pub fn l_row(&self, y: &[i64], bound: u32) -> Result<ChallengeRing<F>, WhiteoutError> {
        self.harness.setup.l_row().apply(y, bound)
    }

    /// `t_D = A_D G⁻¹(V)`.
    pub fn commit_v(&self, v: &[ChallengeRing<F>]) -> Result<Vec<ChallengeRing<F>>, WhiteoutError> {
        let setup = &self.harness.setup;
        Ok(commit(setup.v_level(), setup.v_matrix(), v, &[])?)
    }

    /// The attempt fork's seed for `(σ, n, t_D)`.
    pub fn seed(&self, nonce: &[u8; ATTEMPT_NONCE_LEN], t_d: &[ChallengeRing<F>]) -> [u8; 32] {
        attempt_seed::<Blake2b512, F>(&[5; FORK_SEED_LEN], nonce, t_d)
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
    fn prove_response<H: Sponge>(
        &self,
        _statement: &ResponseStatement<'_, F, D>,
        _witness: &ResponseWitness,
        _transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        Ok(())
    }

    fn verify_response<H: Sponge>(
        &self,
        _statement: &ResponseStatement<'_, F, D>,
        _transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        Err(WhiteoutError::ResponseRejected("diagnostics prove only").into())
    }
}

/// Ring elements of the challenge ring with `CHALLENGE_RING_DIM`
/// coefficients, for callers building `V`.
pub const RING: usize = CHALLENGE_RING_DIM;
