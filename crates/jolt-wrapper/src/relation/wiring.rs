//! Cross-stage wiring: every produced opening keyed by `JoltOpeningId` with
//! its opening point, the batch geometry (member offsets and per-round
//! emitted degrees) and the canonical output-claim order of a relation.

use std::collections::HashMap;

use jolt_claims::protocols::jolt::{JoltChallengeId, JoltDerivedId, JoltOpeningId};
use jolt_claims::{OutputClaims, SymbolicSumcheck};
use jolt_field::{Fr, Zero};

use super::ctx::{Ctx, Lc};
use super::lower::Sources;
use super::sumcheck::{absorb_openings, begin_batch, compressed_rounds, Batch, Member};
use super::RelationError;

#[derive(Default)]
pub(crate) struct Wires {
    pub sources: Sources,
    pub points: HashMap<JoltOpeningId, Vec<Lc>>,
}

impl Wires {
    pub(crate) fn set(&mut self, id: JoltOpeningId, lc: Lc, point: Vec<Lc>) {
        self.sources.opening(id, lc);
        drop(self.points.insert(id, point));
    }

    /// An aliased output is the same polynomial at the same (structurally
    /// identical) point as `source`: it shares the wire.
    pub(crate) fn alias(
        &mut self,
        aliased: JoltOpeningId,
        source: JoltOpeningId,
    ) -> Result<(), RelationError> {
        let lc = self.sources.opening_lc(&source)?;
        let point = self.point(&source)?.to_vec();
        self.set(aliased, lc, point);
        Ok(())
    }

    pub(crate) fn lc(&self, id: &JoltOpeningId) -> Result<Lc, RelationError> {
        self.sources.opening_lc(id)
    }

    pub(crate) fn point(&self, id: &JoltOpeningId) -> Result<&[Lc], RelationError> {
        self.points
            .get(id)
            .map(Vec::as_slice)
            .ok_or_else(|| RelationError::MissingSource {
                kind: "opening point",
                id: format!("{id:?}"),
            })
    }

    pub(crate) fn challenge(&mut self, id: impl Into<JoltChallengeId>, lc: Lc) {
        self.sources.challenge(id.into(), lc);
    }

    pub(crate) fn derived(&mut self, id: impl Into<JoltDerivedId>, lc: Lc) {
        self.sources.derived(id.into(), lc);
    }
}

/// One member's placement in a batch: `rounds` consecutive batch rounds from
/// `offset`, emitting `degrees[i]` coefficients in its `i`-th round.
pub(crate) struct Layout {
    pub rounds: usize,
    pub offset: usize,
    pub degrees: Vec<usize>,
}

impl Layout {
    pub(crate) fn uniform(rounds: usize, degree: usize, offset: usize) -> Self {
        Self {
            rounds,
            offset,
            degrees: vec![degree; rounds],
        }
    }

    /// The default placement: aligned to the end of a `max_rounds` batch.
    pub(crate) fn suffix(rounds: usize, degree: usize, max_rounds: usize) -> Self {
        Self::uniform(rounds, degree, max_rounds - rounds)
    }

    pub(crate) fn slice<'a>(&self, batch_point: &'a [Lc]) -> &'a [Lc] {
        &batch_point[self.offset..self.offset + self.rounds]
    }
}

/// Per batch round, the highest degree any active member emits.
pub(crate) fn batch_degrees(layouts: &[Layout]) -> Vec<usize> {
    let max_rounds = layouts
        .iter()
        .map(|layout| layout.offset + layout.rounds)
        .max()
        .unwrap_or(0);
    (0..max_rounds)
        .map(|round| {
            layouts
                .iter()
                .filter(|layout| layout.offset <= round && round < layout.offset + layout.rounds)
                .map(|layout| layout.degrees[round - layout.offset])
                .max()
                .unwrap_or(1)
        })
        .collect()
}

/// Batch head + rounds: returns the batch, the full batch point and the
/// final reduced claim.
pub(crate) fn run_batch(
    ctx: &mut Ctx,
    inputs: &[Lc],
    layouts: &[Layout],
) -> Result<(Batch, Vec<Lc>, Lc), RelationError> {
    let members: Vec<Member> = inputs
        .iter()
        .zip(layouts)
        .map(|(input, layout)| Member {
            input_claim: input.clone(),
            rounds: layout.rounds,
        })
        .collect();
    let batch = begin_batch(ctx, &members)?;
    let degrees = batch_degrees(layouts);
    let (point, final_claim) = compressed_rounds(ctx, &batch.claimed_sum, &degrees)?;
    Ok((batch, point, final_claim))
}

/// The wire order of a relation's produced openings: its `Outputs` struct in
/// field-declaration order, restricted to the ids the output expression (or
/// `extra`, for wire cells the expression never reads) references.
pub(crate) fn canonical_outputs<S>(
    relation: &S,
    extra: &[JoltOpeningId],
) -> Result<Vec<JoltOpeningId>, RelationError>
where
    S: SymbolicSumcheck<OpeningId = JoltOpeningId>,
    S::Outputs<Fr>: OutputClaims<Fr>,
{
    let referenced = relation.expected_output_openings::<Fr>();
    let outputs = S::Outputs::<Fr>::from_opening_values(|id| {
        (referenced.contains(id) || extra.contains(id)).then_some(Fr::zero())
    })
    .map_err(|error| RelationError::Geometry(format!("{error:?}")))?;
    Ok(outputs.canonical_order())
}

/// Absorbs one member's produced openings (aliased ids share their source
/// wire and are not absorbed) and registers each with its opening point.
pub(crate) fn absorb_member<S>(
    ctx: &mut Ctx,
    wires: &mut Wires,
    relation: &S,
    extra: &[JoltOpeningId],
    aliases: &[(JoltOpeningId, JoltOpeningId)],
    point_of: impl Fn(&JoltOpeningId) -> Vec<Lc>,
) -> Result<(), RelationError>
where
    S: SymbolicSumcheck<OpeningId = JoltOpeningId>,
    S::Outputs<Fr>: OutputClaims<Fr>,
{
    let order = canonical_outputs(relation, extra)?;
    let absorbed: Vec<JoltOpeningId> = order
        .iter()
        .copied()
        .filter(|id| !aliases.iter().any(|(aliased, _)| aliased == id))
        .collect();
    let values = absorb_openings(ctx, absorbed.len())?;
    for (id, lc) in absorbed.into_iter().zip(values) {
        let point = point_of(&id);
        wires.set(id, lc, point);
    }
    for &(aliased, source) in aliases {
        wires.alias(aliased, source)?;
    }
    Ok(())
}
