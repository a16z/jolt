//! Synthetic Σ-protocol statements and the statement's own checker, for
//! tests of [`ResponseProver`] implementations (feature
//! `whiteout-test-support`). A response prover sees its statement only
//! inside `prove_response`, so these run the whole Σ-protocol on the
//! synthetic instance of the crate's own tests. Test support only.

use jolt_algebra::One;
use jolt_transcript::{Blake2b512, ProtocolId, ProverTranscript, Sponge, VerifierTranscript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use super::compression::ImageCompression;
use super::fixture::Instance;
pub use super::fixture::RawImages;
pub use super::fixture::F;
use super::response::{ResponseProver, ResponseStatement, ResponseWitness};
use super::sigma::{prove_sigma_with, SelfChecks};
use super::{verify_sigma, BlockTarget, LiftedRelation, SigmaSetup};
use crate::masking::{ResidualRelation, ZkError};

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-zk/whiteout-sigma");
const LABEL: &[u8] = b"whiteout-sigma";

/// Sends `e` in the clear; the verifier checks the statement directly.
pub struct CheckingResponseProver;

impl<const D: usize> ResponseProver<F, D> for CheckingResponseProver {
    fn compression(&self) -> &dyn ImageCompression<F> {
        &RawImages
    }

    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        statement.check_witness(witness)?;
        SendingResponseProver.prove_response(statement, witness, transcript)
    }

    fn verify_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        let bytes = transcript.receive_bytes(statement.layout()?.len())?;
        let witness = ResponseWitness {
            digits: bytes.iter().map(|&byte| byte as i8).collect(),
        };
        Ok(statement.check_witness(&witness)?)
    }
}

/// Sends `e` without checking it, so the verifier's check is what rejects.
pub(crate) struct SendingResponseProver;

impl<const D: usize> ResponseProver<F, D> for SendingResponseProver {
    fn compression(&self) -> &dyn ImageCompression<F> {
        &RawImages
    }

    fn prove_response<H: Sponge>(
        &self,
        _statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        let bytes: Vec<u8> = witness.digits.iter().map(|&digit| digit as u8).collect();
        transcript.send_bytes(&bytes);
        Ok(())
    }

    fn verify_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        CheckingResponseProver.verify_response(statement, transcript)
    }
}

/// One synthetic Σ-protocol instance over `R_D`: a residual witness under a
/// two-level commitment, five masked messages and one masked product.
pub struct SyntheticSigma<const D: usize> {
    instance: Instance<D>,
    checks: SelfChecks,
}

impl<const D: usize> SyntheticSigma<D> {
    /// An honest instance, proved with every prover self-check.
    pub fn new(seed: u64, scale: usize) -> Self {
        Self {
            instance: Instance::scaled(seed, scale),
            checks: SelfChecks::Run,
        }
    }

    /// An instance whose first affine row is off by one, proved with the
    /// prover's self-checks skipped: its response statement fails Item 5,
    /// `ct(h) = 0`, and nothing else.
    pub fn violating_item_five(seed: u64, scale: usize) -> Self {
        let mut instance = Instance::scaled(seed, scale);
        if let Some(row) = instance.relation.affine.first_mut() {
            row.constant += F::one();
        }
        Self {
            instance,
            checks: SelfChecks::Skip,
        }
    }

    /// The length of `e`, the response statement's committed vector, under
    /// `compression`: only its witness lengths are read.
    pub fn response_len(&self, compression: &dyn ImageCompression<F>) -> usize {
        let unknown = || BlockTarget::Compressed {
            payload: Vec::new(),
            image: None,
        };
        self.instance
            .setup_with([unknown(), unknown()])
            .response_layout(compression)
            .map_or(0, |layout| layout.len())
    }

    /// The images, then the Σ-protocol with `response` proving the
    /// response statement; `seed` drives the prover's randomness.
    pub fn prove<P: ResponseProver<F, D> + Sync>(
        &self,
        response: &P,
        seed: u64,
    ) -> Result<Vec<u8>, ZkError> {
        let setup = self
            .instance
            .setup_with(self.instance.targets(response.compression()));
        let lifted = setup.lift(&self.instance.relation, &self.instance.claims)?;
        prove_instance(&self.instance, &setup, &lifted, response, self.checks, seed)
            .map(|(proof, _)| proof)
    }

    /// Verifies `proof`, reading the images from it, with `response`
    /// verifying the response statement.
    pub fn verify<P: ResponseProver<F, D>>(
        &self,
        proof: &[u8],
        response: &P,
    ) -> Result<(), ZkError> {
        verify_instance(&self.instance, &self.instance.relation, proof, response)
    }
}

/// Proves `instance` under `setup`, sending the images first; returns the
/// proof and the attempt count.
pub(crate) fn prove_instance<const D: usize, P: ResponseProver<F, D> + Sync>(
    instance: &Instance<D>,
    setup: &SigmaSetup<F, D>,
    lifted: &LiftedRelation<F>,
    response: &P,
    checks: SelfChecks,
    seed: u64,
) -> Result<(Vec<u8>, usize), ZkError> {
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, LABEL);
    for (_, payload) in setup.compressed_targets() {
        transcript.send_all(payload);
    }
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let attempts = prove_sigma_with(
        setup,
        lifted,
        &instance.s,
        &mut transcript,
        &mut rng,
        response,
        checks,
    )?;
    Ok((transcript.finish(), attempts))
}

/// Verifies a proof of `relation`, reading the images from the proof.
pub(crate) fn verify_instance<const D: usize, P: ResponseProver<F, D>>(
    instance: &Instance<D>,
    relation: &ResidualRelation<F>,
    proof: &[u8],
    response: &P,
) -> Result<(), ZkError> {
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, LABEL, proof);
    let compression = response.compression();
    let mut receive = |rings: usize| -> Result<BlockTarget<F, D>, ZkError> {
        Ok(BlockTarget::Compressed {
            payload: transcript.receive_n::<F>(compression.payload_len(rings * D)?)?,
            image: None,
        })
    };
    let targets = [receive(instance.u.len())?, receive(instance.t_p.len())?];
    let setup = instance.setup_with(targets);
    let lifted = setup.lift(relation, &instance.claims)?;
    verify_sigma(&setup, &lifted, &mut transcript, response)?;
    Ok(transcript.finish()?)
}
