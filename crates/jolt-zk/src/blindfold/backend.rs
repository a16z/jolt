//! BlindFold as a [`ZkBackend`]: Pedersen row commitments to the pads, and
//! the folded relaxed-R1CS proof of the residual relation.

use std::marker::PhantomData;

#[cfg(feature = "prover")]
use jolt_algebra::{Field, Zero};
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
#[cfg(feature = "prover")]
use jolt_transcript::ProverTranscript;
use jolt_transcript::{Encoding, FieldTranscript, Transcript};
#[cfg(feature = "prover")]
use rand_core::{CryptoRng, RngCore};

use super::protocol::pad_row_count;
#[cfg(feature = "prover")]
use super::protocol::product_intermediates;
#[cfg(feature = "prover")]
use super::{prove, BlindFoldWitness};
use super::{BlindFoldProof, BlindFoldProtocol, RelaxedError};
#[cfg(feature = "prover")]
use crate::masking::CommittedPads;
use crate::masking::{
    CommittedPadsView, OperandRule, PadLayout, ResidualRelation, ZkBackend, ZkError,
};
use crate::secret::SecretVec;

/// The curve back end over the vector commitment `VC`.
pub struct BlindFold<VC>(PhantomData<VC>);

/// The pad commitment's opening: one independent blind per pad row (a shared
/// blind would make `C_i − C_j` an unblinded commitment to `p_i − p_j`), and
/// the row commitments the instance reuses. The blinds are zeroed when it is
/// dropped.
pub struct BlindFoldPadOpening<F, Com> {
    #[cfg_attr(
        not(feature = "prover"),
        expect(dead_code, reason = "only the prover builds and opens the pad rows")
    )]
    row_blinds: SecretVec<F>,
    pub row_commitments: Vec<Com>,
}

/// The residual witness: the evaluation y inside the PCS's hiding evaluation
/// commitment, and its blind ρ.
#[derive(Clone, Copy, Debug)]
pub struct BlindFoldResidualWitness<F> {
    pub evaluation: F,
    pub blind: F,
}

impl<VC> BlindFold<VC> {
    /// The masking lowering's operand rule for BlindFold: an R1CS
    /// constraint's A side is any linear combination. The lattice back end
    /// keeps the default `MaskedSlot` (W6).
    pub const OPERANDS: OperandRule = OperandRule::Affine;
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
    VC::Field: Encoding,
    VC::Output: Copy + HomomorphicCommitment<VC::Field> + Encoding,
{
    type Setup = VC::Setup;
    type PadCommitment = Vec<VC::Output>;
    type PadOpening = BlindFoldPadOpening<VC::Field, VC::Output>;
    type ResidualCommitment = VC::Output;
    type Witness = BlindFoldResidualWitness<VC::Field>;
    type Proof = BlindFoldProof<VC::Field, VC::Output>;

    #[cfg(feature = "prover")]
    fn commit_pads<T: Transcript, R: RngCore + CryptoRng>(
        setup: &VC::Setup,
        layout: &PadLayout,
        transcript: &mut T,
        rng: &mut R,
    ) -> Result<(CommittedPads<VC::Field, Self::PadOpening>, Vec<VC::Output>), ZkError> {
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
        transcript.append_values(&row_commitments);
        let pads = CommittedPads::new(
            layout.clone(),
            pads,
            BlindFoldPadOpening {
                row_blinds: row_blinds.into(),
                row_commitments: row_commitments.clone(),
            },
        )?;
        Ok((pads, row_commitments))
    }

    fn receive_pad_commitment<T: Transcript>(
        setup: &VC::Setup,
        layout: &PadLayout,
        commitment: Vec<VC::Output>,
        transcript: &mut T,
    ) -> Result<CommittedPadsView<Vec<VC::Output>>, ZkError> {
        let row_len = Self::row_len(setup)?;
        let expected = pad_row_count(layout.num_pads, row_len);
        if commitment.len() != expected {
            return Err(ZkError::BlindFoldShape(RelaxedError::LengthMismatch {
                name: "pad row commitments",
                expected,
                actual: commitment.len(),
            }));
        }
        transcript.append_values(&commitment);
        Ok(CommittedPadsView::new(layout.clone(), commitment))
    }

    #[cfg(feature = "prover")]
    fn prove<T: FieldTranscript<VC::Field> + ProverTranscript, R: RngCore + CryptoRng>(
        setup: &VC::Setup,
        relation: &ResidualRelation<VC::Field>,
        pads: CommittedPads<VC::Field, Self::PadOpening>,
        residual: &VC::Output,
        witness: &Self::Witness,
        transcript: &mut T,
        rng: &mut R,
    ) -> Result<Self::Proof, ZkError> {
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
        let mut blindings = opening.row_blinds.to_vec();
        for value in [witness.evaluation, witness.blind] {
            let mut row = vec![VC::Field::zero(); row_len];
            if let Some(first) = row.first_mut() {
                *first = value;
            }
            rows.push(row);
            blindings.push(VC::Field::random(rng));
        }
        let intermediates = product_intermediates(relation, pads.pads(), &[witness.evaluation])?;
        for chunk in intermediates.chunks(row_len) {
            let mut row = chunk.to_vec();
            row.resize(row_len, VC::Field::zero());
            rows.push(row);
            blindings.push(VC::Field::random(rng));
        }
        if rows.len() != dimensions.witness_rows.products.end {
            return Err(ZkError::InvalidRelation {
                reason: "product intermediates differ from the BlindFold layout",
            });
        }
        rows.resize(
            dimensions.witness.row_count,
            vec![VC::Field::zero(); row_len],
        );
        blindings.resize(dimensions.witness.row_count, VC::Field::zero());

        prove::<VC::Field, VC, T, R>(
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

    fn verify<T: FieldTranscript<VC::Field>>(
        setup: &VC::Setup,
        relation: &ResidualRelation<VC::Field>,
        pads: &CommittedPadsView<Vec<VC::Output>>,
        residual: &VC::Output,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), ZkError> {
        pads.finish()?;
        let protocol = BlindFoldProtocol::new(
            relation,
            Self::row_len(setup)?,
            pads.commitment().clone(),
            *residual,
        )?;
        protocol
            .verify::<VC, T>(setup, proof, transcript)
            .map_err(ZkError::from)
    }
}
