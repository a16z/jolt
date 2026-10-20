//! BlindFold as a [`ZkBackend`]: Pedersen row commitments to the pads, and
//! the folded relaxed-R1CS proof of the residual relation.

use std::marker::PhantomData;

use jolt_algebra::{CanonicalDecode, Field, Zero};
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_transcript::{ProverTranscript, Sponge, VerifierTranscript};
use rand_core::{CryptoRng, RngCore};

use super::protocol::pad_row_count;
use super::{prove, BlindFoldProtocol, BlindFoldWitness};
use crate::masking::{
    CommittedPads, CommittedPadsView, PadLayout, ResidualRelation, ZkBackend, ZkError,
};

/// The curve back end over the vector commitment `VC`.
pub struct BlindFold<VC>(PhantomData<VC>);

/// The pad commitment's opening: one independent blind per pad row (a shared
/// blind would make `C_i − C_j` an unblinded commitment to `p_i − p_j`), and
/// the row commitments the instance reuses.
pub struct BlindFoldPadOpening<F, Com> {
    pub row_blinds: Vec<F>,
    pub row_commitments: Vec<Com>,
}

/// The residual witness: the evaluation y inside the PCS's hiding evaluation
/// commitment, and its blind ρ.
#[derive(Clone, Copy, Debug)]
pub struct BlindFoldResidualWitness<F> {
    pub evaluation: F,
    pub blind: F,
}

impl<VC: VectorCommitment> BlindFold<VC> {
    /// The pad-row length: the setup's capacity, a power of two.
    fn row_len(setup: &VC::Setup) -> Result<usize, ZkError> {
        let row_len = VC::capacity(setup);
        if row_len == 0 || !row_len.is_power_of_two() {
            return Err(ZkError::InvalidLayout {
                reason: "vector-commitment capacity is not a non-zero power of two",
            });
        }
        Ok(row_len)
    }
}

impl<VC> ZkBackend<VC::Field> for BlindFold<VC>
where
    VC: VectorCommitment,
    VC::Output: Copy + HomomorphicCommitment<VC::Field> + CanonicalDecode,
{
    type Setup = VC::Setup;
    type PadCommitment = Vec<VC::Output>;
    type PadOpening = BlindFoldPadOpening<VC::Field, VC::Output>;
    type ResidualCommitment = VC::Output;
    type Witness = BlindFoldResidualWitness<VC::Field>;

    fn commit_pads<H: Sponge, R: RngCore + CryptoRng>(
        setup: &VC::Setup,
        layout: &PadLayout,
        transcript: &mut ProverTranscript<H>,
        rng: &mut R,
    ) -> Result<CommittedPads<VC::Field, Self::PadOpening>, ZkError> {
        let row_len = Self::row_len(setup)?;
        let pads: Vec<VC::Field> = (0..layout.num_pads)
            .map(|_| VC::Field::random(rng))
            .collect();
        let row_blinds: Vec<VC::Field> = (0..pad_row_count(layout.num_pads, row_len))
            .map(|_| VC::Field::random(rng))
            .collect();
        let row_commitments: Vec<VC::Output> = pads
            .chunks(row_len)
            .zip(&row_blinds)
            .map(|(row, blind)| VC::commit(setup, row, blind))
            .collect();
        transcript.send_all(&row_commitments);
        CommittedPads::new(
            layout.clone(),
            pads,
            BlindFoldPadOpening {
                row_blinds,
                row_commitments,
            },
        )
    }

    fn receive_pad_commitment<H: Sponge>(
        setup: &VC::Setup,
        layout: &PadLayout,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<CommittedPadsView<Vec<VC::Output>>, ZkError> {
        let row_len = Self::row_len(setup)?;
        let commitments = transcript.receive_n(pad_row_count(layout.num_pads, row_len))?;
        Ok(CommittedPadsView::new(layout.clone(), commitments))
    }

    fn prove<H: Sponge, R: RngCore + CryptoRng>(
        setup: &VC::Setup,
        relation: &ResidualRelation<VC::Field>,
        pads: CommittedPads<VC::Field, Self::PadOpening>,
        residual: &VC::Output,
        witness: &Self::Witness,
        transcript: &mut ProverTranscript<H>,
        rng: &mut R,
    ) -> Result<(), ZkError> {
        pads.finish()?;
        if relation
            .first_violation(pads.pads(), &[witness.evaluation])?
            .is_some()
        {
            return Err(ZkError::InvalidRelation {
                reason: "the masked transcript does not satisfy its residual rows",
            });
        }
        let row_len = Self::row_len(setup)?;
        let opening = pads.opening();
        let protocol = BlindFoldProtocol::new(
            relation,
            row_len,
            opening.row_commitments.clone(),
            *residual,
        )?;

        let dimensions = &protocol.dimensions;
        let mut rows: Vec<Vec<VC::Field>> = pads
            .pads()
            .chunks(row_len)
            .map(|row| {
                let mut row = row.to_vec();
                row.resize(row_len, VC::Field::zero());
                row
            })
            .collect();
        let mut blindings = opening.row_blinds.clone();
        for value in [witness.evaluation, witness.blind] {
            let mut row = vec![VC::Field::zero(); row_len];
            if let Some(first) = row.first_mut() {
                *first = value;
            }
            rows.push(row);
            blindings.push(VC::Field::random(rng));
        }
        rows.resize(
            dimensions.witness.row_count,
            vec![VC::Field::zero(); row_len],
        );
        blindings.resize(dimensions.witness.row_count, VC::Field::zero());

        prove::<VC::Field, VC, H, R>(
            setup,
            &protocol,
            transcript,
            BlindFoldWitness {
                rows: &rows,
                blindings: &blindings,
                eval_output: witness.evaluation,
                eval_blinding: witness.blind,
            },
            rng,
        )
        .map_err(ZkError::from)
    }

    fn verify<H: Sponge>(
        setup: &VC::Setup,
        relation: &ResidualRelation<VC::Field>,
        pads: &CommittedPadsView<Vec<VC::Output>>,
        residual: &VC::Output,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        pads.finish()?;
        let protocol = BlindFoldProtocol::new(
            relation,
            Self::row_len(setup)?,
            pads.commitment().clone(),
            *residual,
        )?;
        protocol
            .verify::<VC, H>(setup, transcript)
            .map_err(ZkError::from)
    }
}
