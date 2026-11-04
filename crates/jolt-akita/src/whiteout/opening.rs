//! [`AkitaScheme`]'s masked group opening: the verifier seam of jolt-zk and
//! the prover's counterpart on [`TraceOneHotCommitment`].

use akita_transcript::TranscriptSponge;
use jolt_transcript::VerifierState;
use jolt_zk::masking::{OperandShapeDomain, ResidualRows};
use jolt_zk::whiteout::{MaskedGroupClaim, MaskedGroupOpening};

use super::*;
use crate::adapters::{reverse_point, AKITA_ONE_HOT_K16};
use crate::{AkitaChunkProfile, AkitaCommitment, AkitaScheme, AkitaVerifierSetup};

/// The Whiteout parameters of `setup`.
pub(crate) fn whiteout(setup: &AkitaVerifierSetup) -> Result<&WhiteoutSetup, ZkError> {
    setup.whiteout().ok_or_else(|| {
        AkitaError::InvalidSetup("the Akita setup has no Whiteout parameters".into()).into()
    })
}

/// Whether a setup of one-hot `one_hot_k` and `chunk_profile` has a masked
/// opening: one single-chunk K=16 trace group.
pub(crate) fn opens_whiteout(one_hot_k: usize, chunk_profile: AkitaChunkProfile) -> bool {
    one_hot_k == AKITA_ONE_HOT_K16 && chunk_profile == AkitaChunkProfile::default()
}

/// The masked opening of `groups`: a K=16 one-hot trace group on a
/// single-chunk setup, the setup's own trace shape, beside precommitted
/// groups the setup admits. Both roles pass the statement's shape.
pub(crate) fn opening_shape(
    setup: &AkitaVerifierSetup,
    groups: &MaskedOpeningShape,
) -> Result<Arc<WhiteoutShape>, ZkError> {
    if !opens_whiteout(setup.one_hot_k, setup.akita_chunk_profile()) {
        return Err(AkitaError::UnsupportedSchedule(
            "Whiteout opens one single-chunk K=16 trace group".into(),
        )
        .into());
    }
    if groups.num_vars != setup.max_num_vars
        || groups.num_polys != setup.max_num_polys_per_commitment_group
    {
        return Err(AkitaError::UnsupportedSchedule(format!(
            "Whiteout opens the setup's trace group ({} variables, {} polynomials), not {} and {}",
            setup.max_num_vars,
            setup.max_num_polys_per_commitment_group,
            groups.num_vars,
            groups.num_polys
        ))
        .into());
    }
    whiteout(setup)?.shape(groups)
}

/// Splits `groups` into the precommitted groups and the trace group, the
/// last.
fn split_trace<T>(mut groups: Vec<T>) -> Result<(Vec<T>, T), ZkError> {
    let trace = groups.pop().ok_or_else(|| {
        ZkError::from(AkitaError::InvalidInput(
            "a Whiteout opening has a trace group".into(),
        ))
    })?;
    Ok((groups, trace))
}

/// The decoded commitments at their points in Akita's order, and the
/// claims' forms in group order. Every group's point is reversed, as in the
/// clear grouped opening (`native_batching`): dense and one-hot sources both
/// keep Jolt's row-major order.
type DecodedGroups = (Vec<(CommittedGroup<F>, Vec<F>)>, Vec<AffineRow<F>>);

fn decode(
    shape: &WhiteoutShape,
    groups: Vec<MaskedGroupClaim<F, AkitaCommitment>>,
) -> Result<DecodedGroups, ZkError> {
    let (precommitted, trace) = split_trace(groups)?;
    let (precommitted_backend, trace_backend) = shape.row().backend_commitments(
        &precommitted
            .iter()
            .map(|group| &group.commitment)
            .collect::<Vec<_>>(),
        &trace.commitment,
    )?;
    let mut decoded = Vec::with_capacity(precommitted.len() + 1);
    let mut forms = Vec::new();
    for (group, backend) in precommitted.into_iter().zip(precommitted_backend) {
        decoded.push((backend, reverse_point(&group.point)));
        forms.extend(group.evaluations);
    }
    decoded.push((trace_backend, reverse_point(&trace.point)));
    forms.extend(trace.evaluations);
    Ok((decoded, forms))
}

impl MaskedGroupOpening<AkitaField> for AkitaScheme {
    type Setup = AkitaVerifierSetup;
    type GroupCommitment = AkitaCommitment;
    type Shape = Arc<WhiteoutShape>;
    type Prefix = MaskedPrefix<F, F>;
    type Sponge = TranscriptSponge;

    fn shape(
        setup: &AkitaVerifierSetup,
        groups: &MaskedOpeningShape,
    ) -> Result<Arc<WhiteoutShape>, ZkError> {
        opening_shape(setup, groups)
    }

    fn receive_pads(
        shape: &Arc<WhiteoutShape>,
        layout: PadLayout,
        transcript: &mut VerifierState<'_, TranscriptSponge>,
    ) -> Result<CommittedPadsView<LatticePadPayload<F>>, ZkError> {
        shape.tail.receive_pads(layout, transcript)
    }

    fn verify_masked<P>(
        shape: &Arc<WhiteoutShape>,
        groups: Vec<MaskedGroupClaim<F, AkitaCommitment>>,
        pads: &mut CommittedPadsView<P>,
        transcript: &mut VerifierState<'_, TranscriptSponge>,
    ) -> Result<MaskedPrefix<F, F>, ZkError> {
        let (groups, forms) = decode(shape, groups)?;
        Ok(shape
            .row()
            .verify_masked(&groups, forms, pads, transcript)?)
    }

    fn count_masked(
        shape: &Arc<WhiteoutShape>,
        groups: Vec<MaskedGroupClaim<F, AkitaCommitment>>,
        view: &mut CommittedPadsView<()>,
        transcript: &mut VerifierState<'_, TranscriptSponge>,
    ) -> Result<MaskedPrefix<F, F>, ZkError> {
        let (groups, forms) = decode(shape, groups)?;
        Ok(shape.row().count_masked(&groups, forms, view, transcript)?)
    }

    fn prefix_shape(
        prefix: &MaskedPrefix<F, F>,
        shape_domain: &mut OperandShapeDomain,
    ) -> Result<(), ZkError> {
        prefix.shape(shape_domain)
    }

    fn append_prefix_rows(
        prefix: &MaskedPrefix<F, F>,
        masked_auxiliary: &[F],
        relation: &mut ResidualRelation<F>,
    ) -> Result<(), ZkError> {
        prefix.rows(masked_auxiliary).append_rows(relation)
    }

    fn verify_tail(
        shape: &Arc<WhiteoutShape>,
        prefix: &MaskedPrefix<F, F>,
        relation: &ResidualRelation<F>,
        pads: &CommittedPadsView<LatticePadPayload<F>>,
        transcript: &mut VerifierState<'_, TranscriptSponge>,
    ) -> Result<(), ZkError> {
        shape.tail.verify(relation, prefix, pads, transcript)
    }
}

/// The prover's residual opening at the cut, which the Σ-protocol proves.
#[cfg(feature = "zk-prover")]
pub struct WhiteoutResidual {
    pub(crate) opening: ResidualOpening<F, F>,
}
