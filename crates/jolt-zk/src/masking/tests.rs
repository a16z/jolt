//! A synthetic masked sumcheck: Σ_x a(x)·b(x) = C over three variables,
//! closed by the product of the two output claims, plus a uni-skip round.
//! The honest masked transcript satisfies its residual rows, and on this
//! instance changing any one masked value breaks some row.

use jolt_algebra::{Field, Fr, Ring};
use jolt_claims::{challenge, constant, opening, Expr, FactoredExpr, Source};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    receive_masked_rounds, MaskedRound, RoundWire, SumcheckDomain, SumcheckDomainSpec,
    SumcheckRecorder, SumcheckStatement,
};
use jolt_transcript::{
    Blake2b512, PrivateChannel, ProtocolId, ProverTranscript, VerifierTranscript,
};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use super::*;

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-zk/masking-tests");
const LABEL: &[u8] = b"masking";
const NUM_VARS: usize = 3;
const DEGREE: usize = 2;
const UNISKIP_DOMAIN: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Opening {
    A,
    B,
    Uniskip,
}

type Product = FactoredExpr<Fr, Opening, (), ()>;

fn product() -> Product {
    FactoredExpr::product(vec![opening(Opening::A), opening(Opening::B)])
}

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn fold(table: &[Fr], r: Fr) -> Vec<Fr> {
    let half = table.len() / 2;
    (0..half)
        .map(|j| table[j] + r * (table[j + half] - table[j]))
        .collect()
}

/// The round polynomial Σ_rest a(X, rest)·b(X, rest) in coefficient form.
fn round_poly(a: &[Fr], b: &[Fr]) -> UnivariatePoly<Fr> {
    let eval = |x: Fr| -> Fr {
        let (a, b) = (fold(a, x), fold(b, x));
        a.iter().zip(&b).map(|(a, b)| *a * *b).sum()
    };
    let (e0, e1, e2) = (eval(fr(0)), eval(fr(1)), eval(fr(2)));
    let c2 = (e2 - e1 - e1 + e0) * fr(2).inverse().unwrap();
    UnivariatePoly::new(vec![e0, e1 - e0 - c2, c2])
}

/// The uni-skip round: `coefficients` whose centered-domain sum is the input.
fn uniskip_poly() -> UnivariatePoly<Fr> {
    UnivariatePoly::new(vec![fr(3), fr(5), fr(7)])
}

fn uniskip_input() -> Fr {
    let poly = uniskip_poly();
    let weights: Vec<Fr> = SumcheckDomainSpec::centered_integer(UNISKIP_DOMAIN)
        .round_sum_coefficients(DEGREE)
        .unwrap();
    poly.coefficients()
        .iter()
        .zip(&weights)
        .map(|(c, w)| *c * *w)
        .sum()
}

fn layout() -> PadLayout {
    // Uni-skip: DEGREE + 1 coefficients and its output claim; the sumcheck:
    // DEGREE wire values per round and the two output claims.
    let base = DEGREE + 1 + 1 + NUM_VARS * DEGREE + 2;
    // The two output claims take the last base slots.
    let mut shape = OperandShapeDomain::new(base);
    let _ = lower_factored(
        &product(),
        &mut shape,
        &mut |id| {
            Ok(Some(PadIndex(match id {
                Opening::A => base - 2,
                Opening::B => base - 1,
                Opening::Uniskip => unreachable!(),
            })))
        },
        &mut |_| Ok(Fr::from_u64(0)),
    )
    .unwrap();
    shape.layout().unwrap()
}

struct Proof {
    narg: Vec<u8>,
    pads: Vec<Fr>,
    claimed_sum: Fr,
}

fn prove() -> Proof {
    let a: Vec<Fr> = (1..=8).map(fr).collect();
    let b: Vec<Fr> = (11..=18).map(fr).collect();
    let claimed_sum: Fr = a.iter().zip(&b).map(|(a, b)| *a * *b).sum();
    let layout = layout();
    let pads: Vec<Fr> = (0..layout.num_pads).map(|i| fr(1000 + i as u64)).collect();
    let mut committed = CommittedPads::new(layout, pads.clone(), ()).unwrap();
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, LABEL);

    let mut uniskip = MaskingRecorder::new(&mut committed);
    let r0 = uniskip
        .absorb_full_round(&uniskip_poly(), DEGREE, &mut transcript)
        .unwrap();
    uniskip
        .finish(&[uniskip_poly().evaluate(r0)], &mut transcript)
        .unwrap();

    let mut recorder = MaskingRecorder::new(&mut committed);
    let (mut a, mut b) = (a, b);
    for _ in 0..NUM_VARS {
        let r = recorder
            .absorb_round(&round_poly(&a, &b), DEGREE, &mut transcript)
            .unwrap();
        a = fold(&a, r);
        b = fold(&b, r);
    }
    recorder.finish(&[a[0], b[0]], &mut transcript).unwrap();

    let mut values = ValuesDomain::default();
    let _ = lower_factored(
        &product(),
        &mut values,
        &mut |id| {
            Ok(match id {
                Opening::A => a[0],
                Opening::B => b[0],
                Opening::Uniskip => unreachable!(),
            })
        },
        &mut |_: &Source<Opening, (), ()>| -> Result<Fr, ZkError> { unreachable!() },
    )
    .unwrap();
    let _ = committed.send(&values.auxiliary, &mut transcript).unwrap();
    committed.finish().unwrap();

    Proof {
        narg: transcript.finish(),
        pads,
        claimed_sum,
    }
}

/// What the verifier records from a masked transcript.
struct Received {
    uniskip: Vec<MaskedRound<Fr>>,
    uniskip_claim: (Fr, usize),
    rounds: Vec<MaskedRound<Fr>>,
    claims: [(Fr, usize); 2],
    auxiliary: Vec<Fr>,
}

fn receive(narg: &[u8]) -> Received {
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, LABEL, narg);
    let layout = layout();
    let auxiliary_len = layout.auxiliary.len();
    let mut view = CommittedPadsView::new(layout, ());
    let mut channel = view.channel(&mut transcript);
    let uniskip = receive_masked_rounds::<Fr, _>(
        SumcheckStatement::new(1, DEGREE),
        RoundWire::Full,
        &mut channel,
    )
    .unwrap();
    let mut uniskip_claim = [Fr::from_u64(0)];
    let uniskip_slot = channel.exchange_private_base(&mut uniskip_claim).unwrap();
    let rounds = receive_masked_rounds::<Fr, _>(
        SumcheckStatement::new(NUM_VARS, DEGREE),
        RoundWire::Compressed,
        &mut channel,
    )
    .unwrap();
    let mut claims = [Fr::from_u64(0); 2];
    let slots = channel.exchange_private_base(&mut claims).unwrap();
    let mut auxiliary = vec![Fr::from_u64(0); auxiliary_len];
    let _ = channel.exchange_private_base(&mut auxiliary).unwrap();
    view.finish().unwrap();
    Received {
        uniskip,
        uniskip_claim: (uniskip_claim[0], uniskip_slot.start),
        rounds,
        claims: [(claims[0], slots.start), (claims[1], slots.start + 1)],
        auxiliary,
    }
}

fn relation(received: &Received, claimed_sum: Fr) -> ResidualRelation<Fr> {
    let mut relation = ResidualRelation::new(layout(), 0);

    let uniskip_claim =
        AffineRow::masked(received.uniskip_claim.0, PadIndex(received.uniskip_claim.1));
    let reduced = sumcheck_rows(
        &mut relation,
        AffineRow::constant(uniskip_input()),
        &received.uniskip,
        DEGREE,
        RoundFormat::Full(SumcheckDomainSpec::centered_integer(UNISKIP_DOMAIN)),
    )
    .unwrap();
    relation
        .affine
        .push(reduced.add_scaled(&uniskip_claim, -Fr::from_u64(1)));

    let running = sumcheck_rows(
        &mut relation,
        AffineRow::constant(claimed_sum),
        &received.rounds,
        DEGREE,
        RoundFormat::Compressed,
    )
    .unwrap();
    let [(a, slot_a), (b, slot_b)] = received.claims;
    let mut rows = RowsDomain::new(&mut relation, &received.auxiliary);
    let output = lower_factored(
        &product(),
        &mut rows,
        &mut |id| {
            Ok(match id {
                Opening::A => AffineRow::masked(a, PadIndex(slot_a)),
                Opening::B => AffineRow::masked(b, PadIndex(slot_b)),
                Opening::Uniskip => unreachable!(),
            })
        },
        &mut |_: &Source<Opening, (), ()>| -> Result<Fr, ZkError> { unreachable!() },
    )
    .unwrap();
    rows.finish().unwrap();
    relation
        .affine
        .push(running.add_scaled(&output, -Fr::from_u64(1)));
    relation
}

#[test]
fn honest_masked_transcript_satisfies_its_rows() {
    let proof = prove();
    let received = receive(&proof.narg);
    let relation = relation(&received, proof.claimed_sum);
    assert_eq!(relation.products.len(), 1);
    assert_eq!(relation.first_violation(&proof.pads, &[]).unwrap(), None);
}

type Perturbation = Box<dyn Fn(&mut Received)>;

/// On this instance, changing any one masked value while the challenges stay
/// those of the honest transcript breaks a row. The challenges are not
/// re-derived, so this checks that every masked value reaches a row, not
/// soundness against a prover who chooses its messages before the
/// challenges.
#[test]
fn changing_any_masked_value_breaks_a_row() {
    let proof = prove();
    let honest = receive(&proof.narg);
    let one = Fr::from_u64(1);
    let mut perturbations: Vec<Perturbation> = Vec::new();
    for round in 0..NUM_VARS {
        for index in 0..DEGREE {
            perturbations.push(Box::new(move |r| r.rounds[round].wire[index] += one));
        }
    }
    for index in 0..=DEGREE {
        perturbations.push(Box::new(move |r| r.uniskip[0].wire[index] += one));
    }
    perturbations.push(Box::new(move |r| r.uniskip_claim.0 += one));
    perturbations.push(Box::new(move |r| r.claims[0].0 += one));
    perturbations.push(Box::new(move |r| r.claims[1].0 += one));
    for index in 0..honest.auxiliary.len() {
        perturbations.push(Box::new(move |r| r.auxiliary[index] += one));
    }
    for perturb in &perturbations {
        let mut received = receive(&proof.narg);
        perturb(&mut received);
        let relation = relation(&received, proof.claimed_sum);
        assert!(relation
            .first_violation(&proof.pads, &[])
            .unwrap()
            .is_some());
    }
}

#[test]
fn a_wrong_claimed_sum_breaks_a_row() {
    let proof = prove();
    let received = receive(&proof.narg);
    let relation = relation(&received, proof.claimed_sum + Fr::from_u64(1));
    assert!(relation
        .first_violation(&proof.pads, &[])
        .unwrap()
        .is_some());
}

#[test]
fn shape_and_values_domains_allocate_alike() {
    let three_factors: FactoredExpr<Fr, Opening, (), ()> =
        FactoredExpr::product(vec![
            opening(Opening::A),
            opening(Opening::B) + opening(Opening::Uniskip),
            opening(Opening::Uniskip),
        ]) + FactoredExpr::from(opening::<Fr, Opening, (), ()>(Opening::A))
            + FactoredExpr::product(vec![opening(Opening::A), opening(Opening::B)])
                .scale(Fr::from_u64(0));
    let mut shape = OperandShapeDomain::new(3);
    let _ = lower_factored(
        &three_factors,
        &mut shape,
        &mut |id| Ok(Some(PadIndex(*id as usize))),
        &mut |_| Ok(Fr::from_u64(0)),
    )
    .unwrap();
    let shape = shape.layout().unwrap();
    let mut values = ValuesDomain::default();
    let value = lower_factored(
        &three_factors,
        &mut values,
        &mut |id| {
            Ok(match id {
                Opening::A => fr(2),
                Opening::B => fr(3),
                Opening::Uniskip => fr(5),
            })
        },
        &mut |_| Ok(Fr::from_u64(0)),
    )
    .unwrap();
    // A zero coefficient still allocates its product slot.
    assert_eq!(shape.auxiliary.len(), 3);
    assert_eq!(values.auxiliary, vec![fr(16), fr(80), fr(6)]);
    assert_eq!(value, fr(80) + fr(2));
}

type RandomExpr = Expr<Fr, usize, (), usize>;

const RANDOM_OPENINGS: usize = 4;
const RANDOM_CHALLENGES: usize = 3;

fn pick(rng: &mut ChaCha20Rng, bound: usize) -> usize {
    rng.next_u32() as usize % bound
}

/// One factor of every kind `lower_factored` classifies: public, a single
/// scaled opening, or affine in several openings.
fn random_factor(rng: &mut ChaCha20Rng) -> RandomExpr {
    let opening_at =
        |rng: &mut ChaCha20Rng| opening::<Fr, usize, (), usize>(pick(rng, RANDOM_OPENINGS));
    let challenge_at =
        |rng: &mut ChaCha20Rng| challenge::<Fr, usize, (), usize>(pick(rng, RANDOM_CHALLENGES));
    match pick(rng, 5) {
        0 => challenge_at(rng),
        1 => constant(Fr::random(rng)),
        2 => opening_at(rng),
        3 => opening_at(rng) * challenge_at(rng),
        _ => opening_at(rng) * challenge_at(rng) + opening_at(rng) + constant(Fr::random(rng)),
    }
}

/// Sums of up to four products of up to four factors; a quarter of the
/// coefficients are zero, which still allocate their slots.
fn random_factored(rng: &mut ChaCha20Rng) -> FactoredExpr<Fr, usize, (), usize> {
    (0..=pick(rng, 4)).fold(FactoredExpr::zero(), |sum, _| {
        let factors = (0..pick(rng, 5)).map(|_| random_factor(rng)).collect();
        let coefficient = if pick(rng, 4) == 0 {
            Fr::from_u64(0)
        } else {
            Fr::random(rng)
        };
        sum + FactoredExpr::product(factors).scale(coefficient)
    })
}

/// The shape, values and rows domains agree on random factored formulas:
/// equal auxiliary counts, the values domain computes the expanded
/// formula's value, and the rows domain's output equals that value over
/// pads whose every row holds.
#[test]
fn lowering_domains_agree_on_random_formulas() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6d61_736b);
    for _ in 0..256 {
        let expr = random_factored(&mut rng);
        let openings: Vec<Fr> = (0..RANDOM_OPENINGS).map(|_| Fr::random(&mut rng)).collect();
        let challenges: Vec<Fr> = (0..RANDOM_CHALLENGES)
            .map(|_| Fr::random(&mut rng))
            .collect();
        let public = |source: &Source<usize, (), usize>| match source {
            Source::Challenge(id) => Ok(challenges[*id]),
            Source::Opening(_) | Source::Derived(()) => unreachable!(),
        };
        let expected = expr.clone().expand().evaluate(
            |id| openings[*id],
            |id| challenges[*id],
            |()| unreachable!(),
        );

        let mut shape = OperandShapeDomain::new(RANDOM_OPENINGS);
        let _ = lower_factored(
            &expr,
            &mut shape,
            &mut |id| Ok(Some(PadIndex(*id))),
            &mut public.clone(),
        )
        .unwrap();
        let layout = shape.layout().unwrap();

        let mut values = ValuesDomain::default();
        let value = lower_factored(
            &expr,
            &mut values,
            &mut |id| Ok(openings[*id]),
            &mut public.clone(),
        )
        .unwrap();
        assert_eq!(value, expected);
        assert_eq!(values.auxiliary.len(), layout.auxiliary.len());
        let pads: Vec<Fr> = (0..layout.num_pads).map(|_| Fr::random(&mut rng)).collect();
        let masked_auxiliary: Vec<Fr> = values
            .auxiliary
            .iter()
            .zip(&pads[layout.auxiliary.clone()])
            .map(|(value, pad)| *value + *pad)
            .collect();
        let mut relation = ResidualRelation::new(layout, 0);
        let mut rows = RowsDomain::new(&mut relation, &masked_auxiliary);
        let output = lower_factored(
            &expr,
            &mut rows,
            &mut |id| Ok(AffineRow::masked(openings[*id] + pads[*id], PadIndex(*id))),
            &mut public.clone(),
        )
        .unwrap();
        rows.finish().unwrap();
        relation
            .affine
            .push(output.add_scaled(&AffineRow::constant(expected), -Fr::from_u64(1)));
        assert_eq!(relation.first_violation(&pads, &[]).unwrap(), None);
    }
}

#[test]
fn pad_exhaustion_is_an_error() {
    let layout = PadLayout::new(1, 0).unwrap();
    let mut committed = CommittedPads::new(layout, vec![fr(1)], ()).unwrap();
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, LABEL);
    assert!(matches!(
        committed.send(&[fr(1), fr(2)], &mut transcript),
        Err(ZkError::PadsExhausted { .. })
    ));
}
