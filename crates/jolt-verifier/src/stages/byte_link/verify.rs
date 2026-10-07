//! The link verifier: replays the prover's messages through
//! [`super::transcript`] and checks every relation of the protocol.

use std::iter;

use jolt_claims::protocols::jolt::lattice::byte_link::{
    ByteLinkBatch, ByteLinkInputs, HistogramGroup, HistogramQuery, BYTE_BITS, BYTE_LINK_PACKS,
};
use jolt_claims::protocols::jolt::lattice::geometry::BalancedIncChunking;
use jolt_claims::protocols::jolt::lattice::ByteTraceLayoutPlan;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial as Poly, JoltRelationId};
use jolt_field::JoltField;
use jolt_poly::EqPolynomial;
use jolt_transcript::{AppendToTranscript, Transcript};

use super::transcript;
use super::{
    ByteLinkCompression, ByteLinkOpening, ByteLinkOpenings, ByteLinkProof, GkrLayerProof,
    ReductionProof,
};
use crate::error::ByteLinkError;
use crate::VerifierError;

/// Verifies the link over the stage-6b `inputs` against the byte trace `plan`
/// and returns the openings of `Q` and both `W` groups for stage 8.
#[jolt_verifier_derive::fs_scope(ByteLink)]
pub fn verify<F, C, T>(
    proof: &ByteLinkProof<F, C>,
    inputs: &ByteLinkInputs<F>,
    plan: &ByteTraceLayoutPlan,
    transcript: &mut T,
) -> Result<ByteLinkOpenings<F>, VerifierError>
where
    F: JoltField,
    C: AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let log_t = inputs.cycle_point().len();
    if plan.packing().logical_num_vars() != log_t {
        return Err(ByteLinkError::Shape {
            what: "stage-6b cycle point",
        }
        .into());
    }
    transcript::absorb_histogram_commitments(transcript, &proof.histogram_commitments);
    let compression = transcript::draw_compression(transcript);
    transcript::absorb_roots(transcript, &proof.trace_roots, &proof.table_roots);
    for (pack, ([p_trace, b_trace], [p_table, b_table])) in
        proof.trace_roots.iter().zip(&proof.table_roots).enumerate()
    {
        if *b_trace == F::zero() || *b_table == F::zero() {
            return Err(ByteLinkError::ZeroRoot { pack }.into());
        }
        if *p_trace * *b_table != *p_table * *b_trace {
            return Err(ByteLinkError::Roots { pack }.into());
        }
    }

    let [triple_roots @ .., ram_root] = &proof.table_roots;
    let (z, trace_leaves) = verify_gkr(
        transcript,
        ByteLinkBatch::Trace,
        &proof.trace,
        &proof.trace_roots,
        log_t,
    )?;
    let (y, triple_leaves) = verify_gkr(
        transcript,
        ByteLinkBatch::Triples,
        &proof.triples,
        triple_roots,
        log_t,
    )?;
    let (y_ram, ram_leaves) = verify_gkr(
        transcript,
        ByteLinkBatch::Ram,
        &proof.ram,
        iter::once(ram_root),
        log_t,
    )?;

    let eq_r_z = EqPolynomial::<F>::mle(inputs.cycle_point(), &z);
    if let Some(pack) = trace_leaves.iter().position(|[p, _]| *p != eq_r_z) {
        return Err(ByteLinkError::TraceLeaf { pack }.into());
    }
    let ByteLinkCompression { gamma, beta } = &compression;
    let tables = triple_leaves
        .iter()
        .map(|leaf| (HistogramGroup::Triples, y.as_slice(), leaf))
        .chain(
            ram_leaves
                .iter()
                .map(|leaf| (HistogramGroup::Ram, y_ram.as_slice(), leaf)),
        );
    for (pack, ((group, point, [_, b]), gamma)) in tables.zip(gamma).enumerate() {
        if group.table_denominator(point, gamma, *beta) != Some(*b) {
            return Err(ByteLinkError::TableLeaf { pack }.into());
        }
    }

    let queries = inputs.histogram_queries();
    let triples = verify_query(
        transcript,
        HistogramGroup::Triples,
        &proof.triple_query,
        &queries,
        &y,
        &triple_leaves,
    )?;
    let ram = verify_query(
        transcript,
        HistogramGroup::Ram,
        &proof.ram_query,
        &queries,
        &y_ram,
        &ram_leaves,
    )?;
    let source = verify_source(
        transcript,
        &proof.source,
        inputs,
        plan,
        &compression,
        &z,
        &trace_leaves,
    )?;
    Ok(ByteLinkOpenings {
        triples,
        ram,
        source,
    })
}

/// One batched GKR from `roots` down to the leaves; returns the leaf point and
/// every tree's leaf claims.
fn verify_gkr<'a, F, T>(
    transcript: &mut T,
    batch: ByteLinkBatch,
    layers: &[GkrLayerProof<F>],
    roots: impl IntoIterator<Item = &'a [F; 2]>,
    log_t: usize,
) -> Result<(Vec<F>, Vec<[F; 2]>), VerifierError>
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    if layers.len() != batch.num_vars(log_t) {
        return Err(ByteLinkError::Shape { what: "GKR layers" }.into());
    }
    let mut point = Vec::new();
    let mut claims = roots.into_iter().copied().collect::<Vec<_>>();
    for (layer, proof) in layers.iter().enumerate() {
        transcript::absorb_layer_claims(transcript, batch, layer, &point, &claims);
        let weights = transcript::draw_layer_weights(transcript, claims.len());
        let claimed_sum = claims
            .iter()
            .zip(&weights)
            .map(|([p, b], [w_p, w_b])| *w_p * *p + *w_b * *b)
            .sum();
        let sumcheck = transcript::verify_rounds(transcript, claimed_sum, &proof.rounds, layer)?;
        if proof.children.len() != claims.len() {
            return Err(ByteLinkError::Shape {
                what: "GKR children",
            }
            .into());
        }
        transcript::absorb_children(transcript, &proof.children);
        let parent = sumcheck
            .point
            .as_slice()
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let gates = proof
            .children
            .iter()
            .zip(&weights)
            .map(|([p_0, b_0, p_1, b_1], [w_p, w_b])| {
                *w_p * (*p_0 * *b_1 + *p_1 * *b_0) + *w_b * *b_0 * *b_1
            })
            .sum::<F>();
        if sumcheck.value != EqPolynomial::<F>::mle(&point, &parent) * gates {
            return Err(ByteLinkError::Gate { batch, layer }.into());
        }
        let mu = transcript::draw_child_selector(transcript);
        claims = proof
            .children
            .iter()
            .map(|[p_0, b_0, p_1, b_1]| [*p_0 + mu * (*p_1 - *p_0), *b_0 + mu * (*b_1 - *b_0)])
            .collect();
        point = parent;
        point.push(mu);
    }
    Ok((point, claims))
}

/// `group`'s histogram query reduction: per pack its marginal queries, then
/// its table leaf `W(y)`, reduced to one point of every `W` of the group.
fn verify_query<F, T>(
    transcript: &mut T,
    group: HistogramGroup,
    proof: &ReductionProof<F>,
    queries: &[HistogramQuery<F>],
    leaf_point: &[F],
    leaves: &[[F; 2]],
) -> Result<ByteLinkOpening<F>, VerifierError>
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    let terms = group
        .packs()
        .zip(leaves)
        .flat_map(|(pack, [w, _])| {
            queries
                .iter()
                .filter(move |query| query.pack == pack)
                .map(move |query| (pack, query.point.as_slice(), query.value))
                .chain(iter::once((pack, leaf_point, *w)))
        })
        .collect::<Vec<_>>();
    let values = terms.iter().map(|(_, _, value)| *value).collect::<Vec<_>>();
    transcript::absorb_query_values(transcript, group, &values);
    let weights = transcript::draw_query_weights(transcript, values.len());
    let claimed_sum = values.iter().zip(&weights).map(|(v, a)| *v * *a).sum();
    let sumcheck =
        transcript::verify_rounds(transcript, claimed_sum, &proof.rounds, group.num_vars())?;
    if proof.finals.len() != group.packs().len() {
        return Err(ByteLinkError::Shape { what: "W finals" }.into());
    }
    transcript::absorb_query_finals(transcript, &proof.finals);
    let point = sumcheck
        .point
        .as_slice()
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let expected = group
        .packs()
        .zip(&proof.finals)
        .map(|(pack, w)| {
            *w * terms
                .iter()
                .zip(&weights)
                .filter(|((term_pack, _, _), _)| *term_pack == pack)
                .map(|((_, query_point, _), a)| *a * EqPolynomial::<F>::mle(query_point, &point))
                .sum::<F>()
        })
        .sum::<F>();
    if sumcheck.value != expected {
        return Err(ByteLinkError::Query { group }.into());
    }
    Ok(ByteLinkOpening {
        point,
        values: proof.finals.clone(),
    })
}

/// The `Q` reduction of the seven trace denominator leaves, `F(r)` and both
/// zero slots to every column of `Q` at one point.
fn verify_source<F, T>(
    transcript: &mut T,
    proof: &ReductionProof<F>,
    inputs: &ByteLinkInputs<F>,
    plan: &ByteTraceLayoutPlan,
    compression: &ByteLinkCompression<F>,
    leaf_point: &[F],
    trace_leaves: &[[F; 2]],
) -> Result<ByteLinkOpening<F>, VerifierError>
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    let r = inputs.cycle_point();
    let denominators = trace_leaves.iter().map(|[_, b]| *b).collect::<Vec<_>>();
    transcript::absorb_source_claims(transcript, leaf_point, &denominators);
    let theta = transcript::draw_zero_slot_point(transcript, r.len());
    let weights = transcript::draw_source_weights(transcript);
    let claimed_sum = denominators
        .iter()
        .zip(&weights.denominators)
        .map(|(b, a)| *a * (compression.beta - *b))
        .sum::<F>()
        + weights.fused_inc * inputs.fused_inc();
    let sumcheck = transcript::verify_rounds(transcript, claimed_sum, &proof.rounds, r.len())?;
    if proof.finals.len() != plan.packing().ids().len() {
        return Err(ByteLinkError::Shape { what: "Q finals" }.into());
    }
    transcript::absorb_source_finals(transcript, &proof.finals);
    let point = sumcheck
        .point
        .as_slice()
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();

    let column = |column: Poly| {
        plan.packing()
            .slot_index(&column)
            .and_then(|slot| proof.finals.get(slot))
            .copied()
            .ok_or(ByteLinkError::Shape { what: "Q slot" })
    };
    let mut packs = F::zero();
    for ((columns, gamma), weight) in BYTE_LINK_PACKS
        .iter()
        .zip(&compression.gamma)
        .zip(&weights.denominators)
    {
        let mut tuple = F::zero();
        for (pack_column, gamma) in columns.iter().zip(gamma) {
            tuple += *gamma * column(*pack_column)?;
        }
        packs += *weight * tuple;
    }
    let chunking = BalancedIncChunking::new(BYTE_BITS).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::ByteLink,
            reason: error.to_string(),
        }
    })?;
    let increment_columns = (0..chunking.chunk_count())
        .map(Poly::BalancedIncDigit)
        .chain(iter::once(Poly::BalancedIncCarry));
    let mut fused_inc = F::zero();
    for (place_value, increment_column) in
        chunking.column_place_values::<F>().zip(increment_columns)
    {
        fused_inc += place_value * column(increment_column)?;
    }
    let mut zero_slots = F::zero();
    for (index, weight) in weights.zero_slots.iter().enumerate() {
        zero_slots += *weight * column(Poly::ZeroSlot(index))?;
    }
    let expected = EqPolynomial::<F>::mle(leaf_point, &point) * packs
        + EqPolynomial::<F>::mle(r, &point) * weights.fused_inc * fused_inc
        + EqPolynomial::<F>::mle(&theta, &point) * zero_slots;
    if sumcheck.value != expected {
        return Err(ByteLinkError::Source.into());
    }
    Ok(ByteLinkOpening {
        point,
        values: proof.finals.clone(),
    })
}
