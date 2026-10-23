//! [`AkitaScheme`]'s masked group opening: the verifier seam of jolt-zk and
//! the prover's counterpart on [`TraceOneHotCommitment`].

use jolt_zk::masking::{OperandShapeDomain, ResidualRows};
use jolt_zk::whiteout::{MaskedGroupClaim, MaskedGroupOpening};

use super::*;
use crate::adapters::{reverse_point, AkitaBackendFlavor, AKITA_ONE_HOT_K16};
use crate::{AkitaChunkProfile, AkitaCommitment, AkitaScheme, AkitaVerifierSetup};

/// The Whiteout parameters of `setup`.
fn whiteout(setup: &AkitaVerifierSetup) -> Result<&WhiteoutSetup, ZkError> {
    setup.whiteout.as_ref().ok_or_else(|| {
        AkitaError::InvalidSetup("the Akita setup has no Whiteout parameters".into()).into()
    })
}

/// The masked opening of a K=16 one-hot trace group of `num_polys`
/// polynomials in `num_vars` variables, on a single-chunk setup: the only
/// shape this prototype opens. The verifier passes statement-derived sizes;
/// the prover its own commitment's.
pub(crate) fn trace_shape(
    setup: &AkitaVerifierSetup,
    num_vars: usize,
    num_polys: usize,
) -> Result<Arc<WhiteoutShape>, ZkError> {
    if setup.one_hot_k != AKITA_ONE_HOT_K16
        || setup.akita_chunk_profile() != AkitaChunkProfile::Single
    {
        return Err(AkitaError::UnsupportedSchedule(
            "Whiteout opens one single-chunk K=16 trace group".into(),
        )
        .into());
    }
    // One shape per setup, so the setup's shape cache holds at most one.
    if num_vars != setup.max_num_vars || num_polys != setup.max_num_polys_per_commitment_group {
        return Err(AkitaError::UnsupportedSchedule(format!(
            "Whiteout opens the setup's trace group ({} variables, {} polynomials), not \
             {num_vars} and {num_polys}",
            setup.max_num_vars, setup.max_num_polys_per_commitment_group
        ))
        .into());
    }
    whiteout(setup)?.shape(num_vars, num_polys)
}

/// The prover's shape for its own `trace` commitment.
pub(crate) fn prover_trace_shape(
    setup: &AkitaVerifierSetup,
    trace: &AkitaCommitment,
) -> Result<Arc<WhiteoutShape>, ZkError> {
    if trace.backend_flavor() != AkitaBackendFlavor::OneHot
        || trace.one_hot_k() != AKITA_ONE_HOT_K16
    {
        return Err(AkitaError::UnsupportedSchedule(
            "Whiteout opens one K=16 one-hot trace group".into(),
        )
        .into());
    }
    trace_shape(setup, trace.num_vars, trace.poly_count)
}

/// The one group a masked opening opens.
fn single<T>(groups: Vec<T>) -> Result<T, ZkError> {
    let mut groups = groups.into_iter();
    match (groups.next(), groups.next()) {
        (Some(group), None) => Ok(group),
        (None, _) | (Some(_), Some(_)) => Err(AkitaError::UnsupportedSchedule(
            "Whiteout opens exactly one group (precommitted groups are W6d)".into(),
        )
        .into()),
    }
}

impl MaskedGroupOpening<AkitaField> for AkitaScheme {
    type Setup = AkitaVerifierSetup;
    type GroupCommitment = AkitaCommitment;
    type Shape = Arc<WhiteoutShape>;
    type Prefix = MaskedPrefix<F, F>;

    fn shape(
        setup: &AkitaVerifierSetup,
        num_vars: usize,
        num_polys: usize,
    ) -> Result<Arc<WhiteoutShape>, ZkError> {
        trace_shape(setup, num_vars, num_polys)
    }

    fn receive_pads<H: Sponge>(
        shape: &Arc<WhiteoutShape>,
        layout: PadLayout,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<CommittedPadsView<LatticePadPayload<F>>, ZkError> {
        shape.tail.receive_pads(&shape.row, layout, transcript)
    }

    fn verify_masked<H: Sponge, P>(
        shape: &Arc<WhiteoutShape>,
        groups: Vec<MaskedGroupClaim<F, AkitaCommitment>>,
        pads: &mut CommittedPadsView<P>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<MaskedPrefix<F, F>, ZkError> {
        let group = single(groups)?;
        let commitment = shape.row.backend_commitment(&group.commitment)?;
        Ok(shape.row.verify_masked(
            &commitment,
            reverse_point(&group.point),
            group.evaluations,
            pads,
            transcript,
        )?)
    }

    fn count_masked<H: Sponge>(
        shape: &Arc<WhiteoutShape>,
        groups: Vec<MaskedGroupClaim<F, AkitaCommitment>>,
        view: &mut CommittedPadsView<()>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<MaskedPrefix<F, F>, ZkError> {
        let group = single(groups)?;
        let commitment = shape.row.backend_commitment(&group.commitment)?;
        Ok(shape.row.count_masked(
            &commitment,
            reverse_point(&group.point),
            group.evaluations,
            view,
            transcript,
        )?)
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

    fn verify_tail<H: Sponge>(
        shape: &Arc<WhiteoutShape>,
        prefix: &MaskedPrefix<F, F>,
        relation: &ResidualRelation<F>,
        pads: &CommittedPadsView<LatticePadPayload<F>>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        shape
            .tail
            .verify(&shape.row, relation, prefix, pads, transcript)
    }
}

/// The prover's residual opening at the cut, which the Σ-protocol proves.
pub struct WhiteoutResidual {
    pub(crate) opening: ResidualOpening<F, F>,
}
