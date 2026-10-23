//! The pads as Whiteout commits them: the lattice pad commitment, sent
//! compressed at the transcript position of the pad commitment
//! (`specs/whiteout-jolt-integration.md` §2, row 2), as the masking front
//! end's [`CommittedPads`] and [`CommittedPadsView`].

#[cfg(feature = "whiteout-prover")]
use jolt_algebra::CyclotomicRing;
use jolt_algebra::JoltField;
#[cfg(feature = "whiteout-prover")]
use jolt_crypto::lattice::{sample_box_digits, CHALLENGE_RING_DIM};
#[cfg(feature = "whiteout-prover")]
use jolt_transcript::ProverTranscript;
use jolt_transcript::{Sponge, VerifierTranscript};
#[cfg(feature = "whiteout-prover")]
use rand_core::{CryptoRng, RngCore};

#[cfg(feature = "whiteout-prover")]
use super::pad_commitment::{LatticePadOpening, RANDOMNESS_LOG_BASIS};
#[cfg(feature = "whiteout-prover")]
use super::sample::sample_zero_constant_term;
#[cfg(feature = "whiteout-prover")]
use super::WhiteoutError;
use super::{ImageCompression, LatticePadCommitment};
#[cfg(feature = "whiteout-prover")]
use crate::masking::CommittedPads;
use crate::masking::{CommittedPadsView, PadLayout, ZkError};

/// What the verifier holds of the pad commitment: the compressed payload of
/// `t_p = A_p · s_p`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticePadPayload<F> {
    pub payload: Vec<F>,
}

/// The prover's opening of the pad commitment: the committed digits, and
/// the image and its compressed payload, which the Σ-protocol states.
#[cfg(feature = "whiteout-prover")]
pub struct WhiteoutPadOpening<F> {
    opening: LatticePadOpening,
    image: Vec<F>,
    payload: Vec<F>,
}

#[cfg(feature = "whiteout-prover")]
impl<F: JoltField> WhiteoutPadOpening<F> {
    /// The committed digits.
    pub fn opening(&self) -> &LatticePadOpening {
        &self.opening
    }

    /// The compressed payload sent for the image.
    pub fn payload(&self) -> &[F] {
        &self.payload
    }

    /// The image `t_p` as elements of `R_D`.
    ///
    /// # Errors
    ///
    /// [`WhiteoutError::InvalidRelation`] if the image is not whole `R_D`
    /// elements.
    pub fn image<const D: usize>(&self) -> Result<Vec<CyclotomicRing<F, D>>, WhiteoutError> {
        if !self.image.len().is_multiple_of(D) {
            return Err(WhiteoutError::InvalidRelation(
                "pad image is not whole ring elements",
            ));
        }
        self.image
            .chunks_exact(D)
            .map(|ring| {
                let coefficients: [F; D] = ring
                    .try_into()
                    .map_err(|_| WhiteoutError::InvalidRelation("pad image ring"))?;
                Ok(CyclotomicRing::from_coefficients(coefficients))
            })
            .collect()
    }
}

/// Samples the pads, the mask `g` and the hiding randomness `r_p` from
/// `rng`, commits them under `commitment`, and sends the compressed image.
/// Called at the pad commitment's transcript position, before the first
/// challenge.
///
/// # Errors
///
/// [`ZkError`] if `layout` does not fit `commitment` or compression fails.
#[cfg(feature = "whiteout-prover")]
pub fn commit_lattice_pads<F, const D: usize, H, R>(
    commitment: &LatticePadCommitment<F, D>,
    compression: &dyn ImageCompression<F>,
    layout: PadLayout,
    transcript: &mut ProverTranscript<H>,
    rng: &mut R,
) -> Result<CommittedPads<F, WhiteoutPadOpening<F>>, ZkError>
where
    F: JoltField,
    H: Sponge,
    R: RngCore + CryptoRng,
{
    let pads: Vec<F> = (0..layout.num_pads).map(|_| F::random(rng)).collect();
    let g = sample_zero_constant_term::<F, CHALLENGE_RING_DIM, _>(rng);
    let randomness = sample_box_digits(
        rng,
        commitment.randomness_columns() * D,
        RANDOMNESS_LOG_BASIS,
    )
    .map_err(WhiteoutError::from)?;
    let (image, opening) = commitment.commit(&pads, g, &randomness)?;
    let image: Vec<F> = image
        .iter()
        .flat_map(|ring| ring.coefficients().iter().copied())
        .collect();
    let compressed = compression.compress(&image)?;
    transcript.send_all(&compressed.payload);
    CommittedPads::new(
        layout,
        pads,
        WhiteoutPadOpening {
            opening,
            image,
            payload: compressed.payload,
        },
    )
}

/// Reads the compressed pad commitment [`commit_lattice_pads`] sends. The
/// verifier's masked channel can only be built from the returned view.
///
/// # Errors
///
/// [`ZkError`] on a truncated payload.
pub fn receive_lattice_pads<F, const D: usize, H>(
    commitment: &LatticePadCommitment<F, D>,
    compression: &dyn ImageCompression<F>,
    layout: PadLayout,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<CommittedPadsView<LatticePadPayload<F>>, ZkError>
where
    F: JoltField,
    H: Sponge,
{
    let payload = transcript.receive_n(compression.payload_len(commitment.image_len())?)?;
    Ok(CommittedPadsView::new(
        layout,
        LatticePadPayload { payload },
    ))
}
