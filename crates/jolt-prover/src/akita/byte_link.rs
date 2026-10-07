//! The byte link stage between stage 7 and the final opening: the histogram
//! commitments are absorbed, the compression challenges drawn, and the link
//! proved against the byte trace through [`LinkTranscript`].

use jolt_claims::protocols::jolt::lattice::byte_link::{
    ByteLinkBatch, ByteLinkInputs, HistogramGroup,
};
use jolt_field::JoltField;
use jolt_kernels::byte_link::reference::{self, ByteTrace, Histograms};
use jolt_kernels::byte_link::{ByteLinkDraw, ByteLinkMessage, ByteLinkTranscript};
use jolt_poly::UnivariatePoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::byte_link::{
    transcript as schedule, ByteLinkOpenings, ByteLinkProof, GkrLayerProof, ReductionProof,
};

use crate::ProverError;

/// The link's wire and the openings it leaves for stage 8.
pub struct ProvedByteLink<F: JoltField, C> {
    pub proof: ByteLinkProof<F, C>,
    pub openings: ByteLinkOpenings<F>,
}

/// Proves the link over the stage-6b `inputs`: absorbs the committed
/// histograms, draws the compression challenges and runs the reference prover.
pub fn prove_byte_link<F, C, T>(
    trace: &ByteTrace<'_>,
    histograms: &Histograms<F>,
    inputs: &ByteLinkInputs<F>,
    histogram_commitments: [C; 2],
    transcript: &mut T,
) -> Result<ProvedByteLink<F, C>, ProverError<F>>
where
    F: JoltField,
    C: AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    schedule::absorb_histogram_commitments(transcript, &histogram_commitments);
    let compression = schedule::draw_compression(transcript);
    let mut link = LinkTranscript::new(transcript);
    let openings = reference::prove(trace, histograms, inputs, &compression, &mut link)?;
    Ok(ProvedByteLink {
        proof: link.finish(histogram_commitments)?,
        openings,
    })
}

/// The link's Fiat–Shamir side over the proof transcript: every message is
/// absorbed and every challenge drawn through the verifier's schedule, and the
/// sent messages are kept as the wire.
pub struct LinkTranscript<'a, F: JoltField, T> {
    transcript: &'a mut T,
    /// `(P, B)` of the seven trace roots, then of the seven table roots.
    roots: Option<[[F; 2]; 14]>,
    trace: Vec<GkrLayerProof<F>>,
    triples: Vec<GkrLayerProof<F>>,
    ram: Vec<GkrLayerProof<F>>,
    triple_query: ReductionProof<F>,
    ram_query: ReductionProof<F>,
    source: ReductionProof<F>,
    /// A message out of protocol order or shape, which the wire cannot carry.
    malformed: bool,
}

impl<'a, F: JoltField, T: Transcript<Challenge = F>> LinkTranscript<'a, F, T> {
    pub fn new(transcript: &'a mut T) -> Self {
        let reduction = || ReductionProof {
            rounds: Vec::new(),
            finals: Vec::new(),
        };
        Self {
            transcript,
            roots: None,
            trace: Vec::new(),
            triples: Vec::new(),
            ram: Vec::new(),
            triple_query: reduction(),
            ram_query: reduction(),
            source: reduction(),
            malformed: false,
        }
    }

    /// The recorded wire with the histogram commitments the stage absorbed.
    pub fn finish<C>(
        self,
        histogram_commitments: [C; 2],
    ) -> Result<ByteLinkProof<F, C>, ProverError<F>> {
        let (Some(roots), false) = (self.roots, self.malformed) else {
            return Err(ProverError::InvariantViolation {
                reason: "the byte-link prover sent messages out of the protocol's order or shape",
            });
        };
        Ok(ByteLinkProof {
            histogram_commitments,
            trace_roots: std::array::from_fn(|pack| roots[pack]),
            table_roots: std::array::from_fn(|pack| roots[7 + pack]),
            trace: self.trace,
            triples: self.triples,
            ram: self.ram,
            triple_query: self.triple_query,
            ram_query: self.ram_query,
            source: self.source,
        })
    }

    fn layers(&mut self, batch: ByteLinkBatch) -> &mut Vec<GkrLayerProof<F>> {
        match batch {
            ByteLinkBatch::Trace => &mut self.trace,
            ByteLinkBatch::Triples => &mut self.triples,
            ByteLinkBatch::Ram => &mut self.ram,
        }
    }

    fn query(&mut self, group: HistogramGroup) -> &mut ReductionProof<F> {
        match group {
            HistogramGroup::Triples => &mut self.triple_query,
            HistogramGroup::Ram => &mut self.ram_query,
        }
    }
}

/// A round polynomial of degree `D` without its linear coefficient, the form
/// the verifier reads back.
fn compressed<F: JoltField, const D: usize>(poly: &UnivariatePoly<F>) -> Option<[F; D]> {
    match poly.coefficients() {
        [constant, _, higher @ ..] if higher.len() + 1 == D => Some(std::array::from_fn(|k| {
            if k == 0 {
                *constant
            } else {
                higher[k - 1]
            }
        })),
        _ => None,
    }
}

impl<F: JoltField, T: Transcript<Challenge = F>> ByteLinkTranscript<F>
    for LinkTranscript<'_, F, T>
{
    fn append(&mut self, message: ByteLinkMessage<'_, F>) {
        let pairs = |pairs: &[(F, F)]| pairs.iter().map(|&(p, b)| [p, b]).collect::<Vec<_>>();
        match message {
            ByteLinkMessage::Roots(roots) => match <[[F; 2]; 14]>::try_from(pairs(roots)) {
                Ok(roots) => {
                    let (trace, table) = roots.split_at(7);
                    schedule::absorb_roots(self.transcript, trace, table);
                    self.roots = Some(roots);
                }
                Err(_) => self.malformed = true,
            },
            ByteLinkMessage::LayerClaims {
                batch,
                layer,
                point,
                claims,
            } => {
                schedule::absorb_layer_claims(self.transcript, batch, layer, point, &pairs(claims));
                let layers = self.layers(batch);
                let in_order = layers.len() == layer;
                layers.push(GkrLayerProof {
                    rounds: Vec::new(),
                    children: Vec::new(),
                });
                self.malformed |= !in_order;
            }
            ByteLinkMessage::LayerRound { batch, poly, .. } => {
                schedule::absorb_round(self.transcript, poly);
                match (compressed(poly), self.layers(batch).last_mut()) {
                    (Some(round), Some(layer)) => layer.rounds.push(round),
                    _ => self.malformed = true,
                }
            }
            ByteLinkMessage::Children {
                batch, children, ..
            } => {
                schedule::absorb_children(self.transcript, children);
                match self.layers(batch).last_mut() {
                    Some(layer) => layer.children = children.to_vec(),
                    None => self.malformed = true,
                }
            }
            ByteLinkMessage::QueryValues { group, values } => {
                schedule::absorb_query_values(self.transcript, group, values);
            }
            ByteLinkMessage::QueryRound { group, poly, .. } => {
                schedule::absorb_round(self.transcript, poly);
                match compressed(poly) {
                    Some(round) => self.query(group).rounds.push(round),
                    None => self.malformed = true,
                }
            }
            ByteLinkMessage::QueryFinals { group, values } => {
                schedule::absorb_query_finals(self.transcript, values);
                self.query(group).finals = values.to_vec();
            }
            ByteLinkMessage::SourceClaims {
                point,
                denominators,
            } => schedule::absorb_source_claims(self.transcript, point, denominators),
            ByteLinkMessage::SourceRound { poly, .. } => {
                schedule::absorb_round(self.transcript, poly);
                match compressed(poly) {
                    Some(round) => self.source.rounds.push(round),
                    None => self.malformed = true,
                }
            }
            ByteLinkMessage::SourceFinals(values) => {
                schedule::absorb_source_finals(self.transcript, values);
                self.source.finals = values.to_vec();
            }
        }
    }

    fn challenges(&mut self, draw: ByteLinkDraw, count: usize) -> Vec<F> {
        let transcript = &mut *self.transcript;
        match draw {
            ByteLinkDraw::LayerWeights { .. } => {
                self.malformed |= !count.is_multiple_of(2);
                schedule::draw_layer_weights(transcript, count / 2)
                    .into_iter()
                    .flatten()
                    .collect()
            }
            ByteLinkDraw::LayerChallenge { .. }
            | ByteLinkDraw::QueryChallenge { .. }
            | ByteLinkDraw::SourceChallenge { .. } => (0..count)
                .map(|_| schedule::draw_round_challenge(transcript))
                .collect(),
            ByteLinkDraw::ChildSelector { .. } => (0..count)
                .map(|_| schedule::draw_child_selector(transcript))
                .collect(),
            ByteLinkDraw::QueryWeights { .. } => schedule::draw_query_weights(transcript, count),
            ByteLinkDraw::ZeroSlotPoint => schedule::draw_zero_slot_point(transcript, count),
            ByteLinkDraw::SourceWeights => {
                let weights = schedule::draw_source_weights(transcript);
                self.malformed |= count != 10;
                weights
                    .denominators
                    .into_iter()
                    .chain([weights.fused_inc])
                    .chain(weights.zero_slots)
                    .collect()
            }
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::panic, reason = "fixture link proofs")]
mod tests {
    use jolt_akita::AkitaField as F;
    use jolt_claims::protocols::jolt::lattice::byte_link::BYTE_LINK_PACKS;
    use jolt_claims::protocols::jolt::JoltCommittedPolynomial as Poly;
    use jolt_field::{One, Ring};
    use jolt_kernels::byte_link::fixtures::{scaled_active, SyntheticTrace};
    use jolt_poly::{eq_index_msb, EqPolynomial};
    use jolt_verifier::error::ByteLinkError;
    use jolt_verifier::stages::byte_link::verify;
    use jolt_verifier::VerifierError;

    use super::*;
    use crate::akita::preprocessing::AkitaTranscript;

    const LOG_ROWS: usize = 16;

    fn trace() -> SyntheticTrace {
        SyntheticTrace::new(LOG_ROWS, scaled_active(LOG_ROWS), 5, false)
    }

    fn prove(
        trace: &SyntheticTrace,
        histograms: &Histograms<F>,
        inputs: &ByteLinkInputs<F>,
    ) -> ProvedByteLink<F, F> {
        let mut transcript = AkitaTranscript::new(b"byte-link-test");
        let commitments = [F::from_u64(1), F::from_u64(2)];
        prove_byte_link(
            &trace.trace(),
            histograms,
            inputs,
            commitments,
            &mut transcript,
        )
        .unwrap()
    }

    fn verify_link(
        trace: &SyntheticTrace,
        proof: &ByteLinkProof<F, F>,
        inputs: &ByteLinkInputs<F>,
    ) -> Result<ByteLinkOpenings<F>, VerifierError> {
        let mut transcript = AkitaTranscript::new(b"byte-link-test");
        verify(proof, inputs, &trace.plan, &mut transcript)
    }

    fn rejection(result: Result<ByteLinkOpenings<F>, VerifierError>) -> ByteLinkError {
        match result {
            Err(VerifierError::ByteLink(error)) => error,
            other => panic!("expected a byte-link rejection, got {other:?}"),
        }
    }

    /// Every column of `Q` and every `W` at the openings' points, evaluated
    /// from the bytes.
    fn assert_opens_the_trace(
        trace: &SyntheticTrace,
        inputs: &ByteLinkInputs<F>,
        openings: &ByteLinkOpenings<F>,
    ) {
        let rows = trace.rows();
        let eq_x = EqPolynomial::<F>::evals(&openings.source.point, None);
        for (slot, value) in openings.source.values.iter().enumerate() {
            let expected = (0..rows)
                .map(|t| eq_x[t] * F::from_i64(trace.bytes[slot * rows + t].into()))
                .sum::<F>();
            assert_eq!(*value, expected, "Q slot {slot}");
        }
        let eq_r = EqPolynomial::<F>::evals(inputs.cycle_point(), None);
        for (group, opening) in [
            (HistogramGroup::Triples, &openings.triples),
            (HistogramGroup::Ram, &openings.ram),
        ] {
            for (pack, value) in group.packs().zip(&opening.values) {
                let expected = (0..rows)
                    .map(|t| {
                        let codes = BYTE_LINK_PACKS[pack].map(|column| trace.at(column, t) as u8);
                        eq_r[t] * eq_index_msb(&opening.point, group.table_index(codes) as u128)
                    })
                    .sum::<F>();
                assert_eq!(*value, expected, "W of pack {pack}");
            }
        }
    }

    /// The honest link verifies and opens `Q` and `W` at their true values; an
    /// altered root, round polynomial or `W` final, and an altered stage-6b
    /// marginal, are each rejected by the check that owns them.
    #[test]
    fn byte_link_round_trips_and_opens_the_trace() {
        let trace = trace();
        let inputs = trace.inputs::<F>(11);
        let histograms = reference::histograms(&trace.trace(), &inputs).unwrap();
        let ProvedByteLink { proof, openings } = prove(&trace, &histograms, &inputs);
        assert_eq!(verify_link(&trace, &proof, &inputs).unwrap(), openings);
        assert_opens_the_trace(&trace, &inputs, &openings);

        let altered = |alter: fn(&mut ByteLinkProof<F, F>)| {
            let mut proof = proof.clone();
            alter(&mut proof);
            rejection(verify_link(&trace, &proof, &inputs))
        };
        assert_eq!(
            altered(|proof| proof.trace_roots[0][0] += F::one()),
            ByteLinkError::Roots { pack: 0 }
        );
        assert_eq!(
            altered(|proof| proof.trace[14].rounds[3][0] += F::one()),
            ByteLinkError::Gate {
                batch: ByteLinkBatch::Trace,
                layer: 14
            }
        );
        assert_eq!(
            altered(|proof| proof.triple_query.finals[2] += F::one()),
            ByteLinkError::Query {
                group: HistogramGroup::Triples
            }
        );

        let (mut claims, fused_inc) = trace.claims::<F>(11);
        claims.get_mut(&Poly::InstructionRa(4)).unwrap().value += F::one();
        let wrong = ByteLinkInputs::new(&claims, &fused_inc).unwrap();
        assert_eq!(
            rejection(verify_link(&trace, &proof, &wrong)),
            ByteLinkError::Query {
                group: HistogramGroup::Triples
            }
        );
    }

    /// Proofs over a byte trace that disagrees with the stage-6b claims — a
    /// changed instruction byte or increment digit, a non-Boolean activity
    /// byte, a nonzero zero slot — and over a changed `W` cell are rejected.
    #[test]
    fn byte_link_rejects_bytes_or_histograms_that_disagree() {
        let trace = trace();
        let inputs = trace.inputs::<F>(11);
        assert_eq!(trace.at(Poly::RamActivity, 1), 1);
        for (column, byte, expected) in [
            (
                Poly::InstructionRa(5),
                trace.at(Poly::InstructionRa(5), 1).wrapping_add(1),
                ByteLinkError::Query {
                    group: HistogramGroup::Triples,
                },
            ),
            (
                Poly::BalancedIncDigit(4),
                trace.at(Poly::BalancedIncDigit(4), 1).wrapping_add(1),
                ByteLinkError::Source,
            ),
            (Poly::RamActivity, 2, ByteLinkError::Roots { pack: 6 }),
            (Poly::ZeroSlot(1), 1, ByteLinkError::Source),
        ] {
            let mut tampered = trace.clone();
            *tampered.at_mut(column, 1) = byte;
            let histograms = reference::histograms(&tampered.trace(), &inputs).unwrap();
            let proof = prove(&tampered, &histograms, &inputs).proof;
            assert_eq!(
                rejection(verify_link(&tampered, &proof, &inputs)),
                expected,
                "{column:?}"
            );
        }

        let mut histograms = reference::histograms(&trace.trace(), &inputs).unwrap();
        histograms.tables[1][5] += F::one();
        let proof = prove(&trace, &histograms, &inputs).proof;
        assert_eq!(
            rejection(verify_link(&trace, &proof, &inputs)),
            ByteLinkError::Roots { pack: 1 }
        );
    }
}
