//! A synthetic masked sumcheck: Σ_x a(x)·b(x) = C over three variables,
//! closed by the product of the two output claims, plus a uni-skip round.
//! The honest masked transcript satisfies its residual rows, and on this
//! instance changing any one masked value breaks some row.
#![cfg(feature = "masking-prover")]

use jolt_claim_expr::{challenge, constant, opening, Expr, FactoredExpr, Source};
use jolt_field::{Field, Fr, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    receive_masked_rounds, DegreeSchedule, MaskedRound, MemberDegrees, PadSlots, RoundWire,
    SumcheckDomain, SumcheckDomainSpec, SumcheckRecorder,
};
use jolt_transcript::{
    protocol_tag, prover_transcript, verifier_transcript, Blake2b512, BytesMsg, ClaimDomain,
    TAG_BYTES,
};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use super::*;

const PROTOCOL: [u8; TAG_BYTES] = protocol_tag(b"jolt-zk/masking-tests");
const LABEL: &[u8] = b"masking";
const NUM_VARS: usize = 3;
const DEGREE: usize = 2;
const UNISKIP_DOMAIN: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

/// The slots of `layout`'s whole auxiliary block.
fn auxiliary_slots(layout: &PadLayout) -> PadSlots {
    PadSlots {
        start: layout.auxiliary.start,
        len: layout.auxiliary.len(),
    }
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
    lower_factored(
        &product(),
        &mut shape,
        &mut ProductMemo::default(),
        &mut |id| {
            Ok(Some(PadIndex(match id {
                Opening::A => base - 2,
                Opening::B => base - 1,
                Opening::Uniskip => unreachable!(),
            })))
        },
        &mut |_| Ok(Fr::from_u64(0)),
    )
    .unwrap()
    .check_equals::<Fr, _>(None, &mut shape)
    .unwrap();
    shape.layout().unwrap()
}

struct Proof {
    tape: MaskedMessages<Fr>,
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
    let mut tape = MaskedMessages::default();
    let mut transcript = prover_transcript::<Blake2b512>(&PROTOCOL, &BytesMsg(LABEL));

    let _ = prove_uniskip_masked(
        &mut committed,
        &mut tape,
        &uniskip_poly(),
        uniskip_input(),
        DEGREE,
        UNISKIP_DOMAIN,
        &mut transcript,
    )
    .unwrap();

    let mut recorder = MaskingRecorder::new(&mut committed, &mut tape, InputClaims::Derived);
    let (mut a, mut b) = (a, b);
    for _ in 0..NUM_VARS {
        let r = recorder
            .absorb_round(&round_poly(&a, &b), DEGREE, &mut transcript)
            .unwrap();
        a = fold(&a, r);
        b = fold(&b, r);
    }
    let _ = recorder.finish(&[a[0], b[0]], &mut transcript).unwrap();

    let mut values = ValuesDomain::default();
    let _ = lower_factored(
        &product(),
        &mut values,
        &mut ProductMemo::default(),
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
    let _ = committed
        .send(&values.auxiliary, &mut tape, &mut transcript)
        .unwrap();
    committed.finish().unwrap();

    Proof {
        tape,
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
    auxiliary_slots: PadSlots,
}

fn receive(tape: &MaskedMessages<Fr>) -> Received {
    let mut transcript = verifier_transcript::<Blake2b512>(&PROTOCOL, &BytesMsg(LABEL), &[]);
    let layout = layout();
    let auxiliary_len = layout.auxiliary.len();
    let mut view = CommittedPadsView::new(layout, ());
    let mut reader = MaskedReader::new(tape);
    let mut channel = view.channel(&mut reader, &mut transcript);
    let uniskip = receive_masked_rounds::<Fr, _>(
        &DegreeSchedule::uniform(1, DEGREE),
        RoundWire::Full,
        &mut channel,
    )
    .unwrap();
    let (uniskip_claim, uniskip_slot) = channel.receive_private(1).unwrap();
    let rounds = receive_masked_rounds::<Fr, _>(
        &DegreeSchedule::uniform(NUM_VARS, DEGREE),
        RoundWire::Compressed,
        &mut channel,
    )
    .unwrap();
    let (claims, slots) = channel.receive_private(2).unwrap();
    let (auxiliary, auxiliary_slots) = channel.receive_private(auxiliary_len).unwrap();
    reader.finish().unwrap();
    view.finish().unwrap();
    Received {
        uniskip,
        uniskip_claim: (uniskip_claim[0], uniskip_slot.start),
        rounds,
        claims: [(claims[0], slots.start), (claims[1], slots.start + 1)],
        auxiliary,
        auxiliary_slots,
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
        &DegreeSchedule::uniform(1, DEGREE),
        &RoundFormat::Full(SumcheckDomainSpec::centered_integer(UNISKIP_DOMAIN)),
    )
    .unwrap();
    relation
        .affine
        .push(reduced.add_scaled(&uniskip_claim, -Fr::from_u64(1)));

    let running = sumcheck_rows(
        &mut relation,
        AffineRow::constant(claimed_sum),
        &received.rounds,
        &DegreeSchedule::uniform(NUM_VARS, DEGREE),
        &RoundFormat::Compressed,
    )
    .unwrap();
    let [(a, slot_a), (b, slot_b)] = received.claims;
    let mut rows =
        RowsDomain::new(&mut relation, &received.auxiliary, received.auxiliary_slots).unwrap();
    let output = lower_factored(
        &product(),
        &mut rows,
        &mut ProductMemo::default(),
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
    output.check_equals(running, &mut rows).unwrap();
    rows.finish().unwrap();
    relation
}

/// Under [`InputClaims::Send`] each input claim goes out masked, as its own
/// message, before the first round; a send that runs out of pads fails the
/// next round.
#[test]
fn sent_input_claims_go_out_masked_before_the_rounds() {
    let claims = [fr(7), fr(9)];
    let poly = UnivariatePoly::new(vec![fr(1), fr(2), fr(3)]);
    let layout = PadLayout::new(claims.len() + DEGREE, 0).unwrap();
    let pads: Vec<Fr> = (0..layout.num_pads).map(|i| fr(100 + i as u64)).collect();
    let mut committed = CommittedPads::new(layout, pads.clone(), ()).unwrap();
    let mut tape = MaskedMessages::default();
    let mut transcript = prover_transcript::<Blake2b512>(&PROTOCOL, &BytesMsg(LABEL));
    let mut recorder = MaskingRecorder::new(&mut committed, &mut tape, InputClaims::Send);
    recorder.absorb_input_claims(&claims, &mut transcript);
    let _ = recorder
        .absorb_round(&poly, DEGREE, &mut transcript)
        .unwrap();
    assert_eq!(tape.values[..2], [claims[0] + pads[0], claims[1] + pads[1]]);
    committed.finish().unwrap();

    let layout = PadLayout::new(1, 0).unwrap();
    let mut committed = CommittedPads::new(layout, vec![fr(1)], ()).unwrap();
    let mut tape = MaskedMessages::default();
    let mut recorder = MaskingRecorder::new(&mut committed, &mut tape, InputClaims::Send);
    recorder.absorb_input_claims(&claims, &mut transcript);
    assert!(matches!(
        recorder
            .absorb_round(&poly, DEGREE, &mut transcript)
            .map_err(ZkError::from),
        Err(ZkError::PadsExhausted { .. })
    ));
}

#[test]
fn honest_masked_transcript_satisfies_its_rows() {
    let proof = prove();
    let received = receive(&proof.tape);
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
    let honest = receive(&proof.tape);
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
        let mut received = receive(&proof.tape);
        perturb(&mut received);
        let relation = relation(&received, proof.claimed_sum);
        assert!(relation
            .first_violation(&proof.pads, &[])
            .unwrap()
            .is_some());
    }
}

/// The lowering reads each round at its scheduled width: a round sent wider
/// or narrower than its degree, or a schedule of another length, is a typed
/// error at that round, not a row.
#[test]
fn rounds_off_their_degree_schedule_are_rejected() {
    let proof = prove();
    let received = receive(&proof.tape);
    let mut relation = ResidualRelation::new(layout(), 0);
    let input = AffineRow::constant(proof.claimed_sum);
    let schedule = DegreeSchedule::batched::<Fr>(
        NUM_VARS,
        [MemberDegrees {
            offset: 0,
            degree: DEGREE + 1,
            round_degrees: vec![DEGREE, DEGREE + 1, DEGREE],
        }],
    )
    .unwrap();
    assert!(matches!(
        sumcheck_rows(
            &mut relation,
            input.clone(),
            &received.rounds,
            &schedule,
            &RoundFormat::Compressed,
        ),
        Err(ZkError::RoundWidthMismatch {
            round: 1,
            expected: 3,
            got: 2
        })
    ));
    assert!(matches!(
        sumcheck_rows(
            &mut relation,
            input,
            &received.rounds,
            &DegreeSchedule::uniform(NUM_VARS - 1, DEGREE),
            &RoundFormat::Compressed,
        ),
        Err(ZkError::RoundCountMismatch {
            expected: 2,
            got: 3
        })
    ));
}

#[test]
fn a_wrong_claimed_sum_breaks_a_row() {
    let proof = prove();
    let received = receive(&proof.tape);
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
                .scale(Fr::from_u64(0))
            + FactoredExpr::product(vec![
                opening(Opening::A),
                opening(Opening::Uniskip),
                opening(Opening::B),
            ])
            + FactoredExpr::product(vec![
                opening(Opening::A),
                opening(Opening::Uniskip),
                opening(Opening::A),
            ]);
    let mut shape = OperandShapeDomain::new(3);
    let _ = lower_factored(
        &three_factors,
        &mut shape,
        &mut ProductMemo::default(),
        &mut |id| Ok(Some(PadIndex(*id as usize))),
        &mut |_| Ok(Fr::from_u64(0)),
    )
    .unwrap();
    let shape = shape.layout().unwrap();
    let mut values = ValuesDomain::default();
    let value = lower_factored(
        &three_factors,
        &mut values,
        &mut ProductMemo::default(),
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
    // A·(B + U) takes a slot and its product with U stays pending, as does
    // the zero-coefficient A·B. A·U takes one slot for both terms that
    // repeat it.
    assert_eq!(shape.auxiliary.len(), 2);
    assert_eq!(values.auxiliary, vec![fr(16), fr(10)]);
    assert_eq!(
        value.pairs,
        vec![
            (fr(16), fr(5)),
            (fr(2), fr(0)),
            (fr(10), fr(3)),
            (fr(10), fr(2))
        ]
    );
    assert_eq!(evaluate(&value), fr(80) + fr(2) + fr(30) + fr(20));
}

/// `Σ_k x_k · y_k + affine`.
fn evaluate(value: &ProductSum<Fr>) -> Fr {
    value
        .pairs
        .iter()
        .fold(value.affine, |sum, (x, y)| sum + *x * *y)
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
        lower_factored(
            &expr,
            &mut shape,
            &mut ProductMemo::default(),
            &mut |id| Ok(Some(PadIndex(*id))),
            &mut public.clone(),
        )
        .unwrap()
        .check_equals::<Fr, _>(None, &mut shape)
        .unwrap();
        let layout = shape.layout().unwrap();

        let mut values = ValuesDomain::default();
        let value = lower_factored(
            &expr,
            &mut values,
            &mut ProductMemo::default(),
            &mut |id| Ok(openings[*id]),
            &mut public.clone(),
        )
        .unwrap();
        assert_eq!(evaluate(&value), expected);
        assert_eq!(values.auxiliary.len(), layout.auxiliary.len());
        let pads: Vec<Fr> = (0..layout.num_pads).map(|_| Fr::random(&mut rng)).collect();
        let masked_auxiliary: Vec<Fr> = values
            .auxiliary
            .iter()
            .zip(&pads[layout.auxiliary.clone()])
            .map(|(value, pad)| *value + *pad)
            .collect();
        let slots = auxiliary_slots(&layout);
        let mut relation = ResidualRelation::new(layout, 0);
        let mut rows = RowsDomain::new(&mut relation, &masked_auxiliary, slots).unwrap();
        let output = lower_factored(
            &expr,
            &mut rows,
            &mut ProductMemo::default(),
            &mut |id| Ok(AffineRow::masked(openings[*id] + pads[*id], PadIndex(*id))),
            &mut public.clone(),
        )
        .unwrap();
        output
            .check_equals(AffineRow::constant(expected), &mut rows)
            .unwrap();
        rows.finish().unwrap();
        let pinned: Vec<PadIndex> = relation
            .products
            .iter()
            .flat_map(|row| row.pairs().map(|pair| pair.x.masked().unwrap().pad))
            .collect();
        assert!(pinned
            .iter()
            .all(|pad| relation.layout.operand_pads.contains(pad)));
        assert_eq!(relation.first_violation(&pads, &[]).unwrap(), None);
    }
}

fn no_public(_: &Source<usize, (), usize>) -> Result<Fr, ZkError> {
    unreachable!()
}

/// `(a + b)·(c + a)` has no single-opening factor: under `MaskedSlot` it first
/// gives `a + b` a defining slot, and `MaskedSlot` rows reject `a + b` as
/// an operand; under `Affine` it takes no slot and the pair's operand is
/// `a + b`. Either
/// way the honest values satisfy the rows.
#[test]
fn operand_rules_decide_the_defining_slot() {
    let expr: FactoredExpr<Fr, usize, (), usize> = FactoredExpr::product(vec![
        opening(0usize) + opening(1usize),
        opening(2usize) + opening(0usize),
    ]);
    let openings = [fr(2), fr(3), fr(5)];
    let pads = [fr(101), fr(103), fr(107), fr(109)];
    for (rule, slots) in [(OperandRule::MaskedSlot, 1), (OperandRule::Affine, 0)] {
        let mut shape = OperandShapeDomain::new(3).with_operands(rule);
        lower_factored(
            &expr,
            &mut shape,
            &mut ProductMemo::default(),
            &mut |id| Ok(Some(PadIndex(*id))),
            &mut no_public,
        )
        .unwrap()
        .check_equals::<Fr, _>(None, &mut shape)
        .unwrap();
        let layout = shape.layout().unwrap();
        assert_eq!(layout.auxiliary.len(), slots);

        let mut values = ValuesDomain::with_operands(rule);
        let value = lower_factored(
            &expr,
            &mut values,
            &mut ProductMemo::default(),
            &mut |id| Ok(openings[*id]),
            &mut no_public,
        )
        .unwrap();
        assert_eq!(evaluate(&value), fr(5) * fr(7));
        let masked_auxiliary: Vec<Fr> = values.auxiliary.iter().map(|v| *v + pads[3]).collect();

        let mut relation = ResidualRelation::new(layout.clone(), 0);
        let mut rows = RowsDomain::new(&mut relation, &masked_auxiliary, auxiliary_slots(&layout))
            .unwrap()
            .with_operands(rule);
        lower_factored(
            &expr,
            &mut rows,
            &mut ProductMemo::default(),
            &mut |id| Ok(AffineRow::masked(openings[*id] + pads[*id], PadIndex(*id))),
            &mut no_public,
        )
        .unwrap()
        .check_equals(AffineRow::constant(fr(35)), &mut rows)
        .unwrap();
        rows.finish().unwrap();
        assert_eq!(relation.first_violation(&pads, &[]).unwrap(), None);
        let operand = &relation.products.last().unwrap().first.x;
        assert_eq!(
            matches!(operand, ProductOperand::Affine(_)),
            rule == OperandRule::Affine
        );
        if let ProductOperand::Affine(x) = operand {
            let mut relation = ResidualRelation::new(layout.clone(), 0);
            assert!(matches!(
                RowsDomain::new(&mut relation, &[], auxiliary_slots(&layout))
                    .unwrap()
                    .products_into(
                        (x.clone(), AffineRow::constant(fr(1))),
                        Vec::new(),
                        AffineRow::constant(fr(0))
                    ),
                Err(ZkError::InvalidRelation { .. })
            ));
        }
    }
}

/// A multi-pair row checks the sum of its pairs.
#[test]
fn product_rows_check_the_sum_of_their_pairs() {
    let pads = [fr(11), fr(13), fr(17), fr(19)];
    let slot = |value: u64, pad: usize| AffineRow::masked(fr(value) + pads[pad], PadIndex(pad));
    let pair = |x: u64, xp: usize, y: u64, yp: usize| ProductPair {
        x: ProductOperand::Masked(slot(x, xp).as_masked_operand().unwrap()),
        y: slot(y, yp),
    };
    let row = ProductRow {
        first: pair(2, 0, 3, 1),
        rest: vec![pair(5, 2, 7, 3)],
        z: AffineRow::constant(fr(2 * 3 + 5 * 7)),
    };
    assert_eq!(row.evaluate(&pads, &[]).unwrap(), fr(0));
    let mut tampered = row.clone();
    tampered.rest[0].y = slot(8, 3);
    assert_eq!(tampered.evaluate(&pads, &[]).unwrap(), fr(5));
}

#[test]
fn pad_exhaustion_is_an_error() {
    let layout = PadLayout::new(1, 0).unwrap();
    let mut committed = CommittedPads::new(layout, vec![fr(1)], ()).unwrap();
    assert!(matches!(
        committed.mask(&[fr(1), fr(2)]),
        Err(ZkError::PadsExhausted { .. })
    ));
}

mod claim_checks {
    //! `product_into` and `check_zero` in every domain, on Akita's stage-1
    //! leaf check as Akita `specs/whiteout-akita.md` §4.4 lowers it:
    //! `u = r·r`, then `u·(c₄u + c₃r + c₂) = claim − c₁r − c₀` lands on the
    //! claim with no slot of its own.

    use jolt_transcript::{ClaimDomain, ClaimMismatch};

    use super::*;

    fn leaf<D: ClaimDomain<Fr, Error = ZkError>>(
        domain: &mut D,
        r: D::Value,
        c: [Fr; 5],
        claim: D::Value,
    ) -> Result<(), ZkError> {
        let u = domain.product(r.clone(), r.clone())?;
        let y = domain.add(
            &domain.add(&domain.scale(&u, c[4]), &domain.scale(&r, c[3])),
            &domain.constant(c[2]),
        );
        let target = domain.add(
            &claim,
            &domain.add(&domain.scale(&r, -c[1]), &domain.constant(-c[0])),
        );
        domain.product_into(u, y, target)?;
        domain.check_zero(domain.add(&claim, &domain.scale(&claim, -fr(1))))
    }

    /// Pads 0 and 1 hold r and the claim; the leaf takes one auxiliary slot.
    #[test]
    fn domains_agree_on_product_into_and_check_zero() {
        let c = [3, 5, 7, 11, 13].map(fr);
        let r = fr(17);
        let claim = c.iter().rev().fold(fr(0), |sum, &c| sum * r + c);

        let mut shape = OperandShapeDomain::new(2);
        leaf(&mut shape, Some(PadIndex(0)), c, Some(PadIndex(1))).unwrap();
        let layout = shape.layout().unwrap();
        assert_eq!((layout.num_pads, layout.auxiliary.len()), (3, 1));
        assert_eq!(layout.operand_pads, vec![PadIndex(0), PadIndex(2)]);

        let mut values = ValuesDomain::default();
        leaf(&mut values, r, c, claim).unwrap();
        assert_eq!(values.auxiliary, vec![r * r]);
        assert!(matches!(
            leaf(&mut ValuesDomain::default(), r, c, claim + fr(1)),
            Err(ZkError::ClaimMismatch(ClaimMismatch))
        ));

        let pads = [fr(101), fr(103), fr(107)];
        let rows_for = |claim: Fr| {
            let masked_auxiliary = [r * r + pads[2]];
            let mut relation = ResidualRelation::new(layout.clone(), 0);
            let mut rows =
                RowsDomain::new(&mut relation, &masked_auxiliary, auxiliary_slots(&layout))
                    .unwrap();
            leaf(
                &mut rows,
                AffineRow::masked(r + pads[0], PadIndex(0)),
                c,
                AffineRow::masked(claim + pads[1], PadIndex(1)),
            )
            .unwrap();
            rows.finish().unwrap();
            relation
        };
        let honest = rows_for(claim);
        assert_eq!((honest.products.len(), honest.affine.len()), (2, 1));
        assert_eq!(honest.first_violation(&pads, &[]).unwrap(), None);
        assert!(rows_for(claim + fr(1))
            .first_violation(&pads, &[])
            .unwrap()
            .is_some());
    }
}

mod single_instance {
    //! Masked single-instance rounds as Akita's masked levels lower them
    //! (Akita `specs/whiteout-akita.md` §4.5): `prove_single_with` sends
    //! each round through the masked prover's round channel, the output
    //! claims a(r), b(r) go out masked, `visit_single_rounds` reads the
    //! rounds back through the masked verifier's round channel, and
    //! `sumcheck_rows` lowers them. The product of the output claims lands
    //! on the running claim.

    use jolt_field::{Ext2, ExtField, Prime128Offset275 as Fp};
    use jolt_poly::{Point, UnivariatePoly};
    use jolt_sumcheck::{
        prove_single_with, visit_single_rounds, FullRounds, ProveRounds, SingleFormat,
        SingleInstance,
    };
    use jolt_transcript::ClaimDomain;

    use super::*;

    type H = Blake2b512;
    const ROUNDS: usize = 3;
    const DEGREE: usize = 2;

    fn fp(value: u64) -> Fp {
        Fp::from_u64(value)
    }

    /// `Σ_x w(x) a(x) b(x)`, binding the lowest bit first; with `tau`,
    /// `w = eq(tau, ·)` and each round returns the full `q`.
    struct Kernel {
        a: Vec<Fp>,
        b: Vec<Fp>,
        tau: Option<Vec<Fp>>,
    }

    fn eq(point: &[Fp], x: usize) -> Fp {
        point
            .iter()
            .enumerate()
            .map(|(k, &p)| if x >> k & 1 == 1 { p } else { fp(1) - p })
            .product()
    }

    fn bind(table: &mut Vec<Fp>, r: Fp) {
        let half = table.len() / 2;
        for j in 0..half {
            table[j] = table[2 * j] + r * (table[2 * j + 1] - table[2 * j]);
        }
        table.truncate(half);
    }

    impl Kernel {
        fn new(eq_factored: bool) -> Self {
            Self {
                a: (1..=8).map(fp).collect(),
                b: (21..=28).map(fp).collect(),
                tau: eq_factored.then(|| vec![fp(5), fp(7), fp(9)]),
            }
        }

        fn instance(&self) -> SingleInstance<Fp> {
            let weight = |x| self.tau.as_ref().map_or(fp(1), |tau| eq(tau, x));
            SingleInstance {
                input_claim: (0..self.a.len())
                    .map(|x| weight(x) * self.a[x] * self.b[x])
                    .sum(),
                rounds: ROUNDS,
                degree: DEGREE,
                format: self.tau.clone().map_or(SingleFormat::Compressed, |tau| {
                    SingleFormat::EqFactored {
                        equality_point: Point::low_to_high(tau),
                    }
                }),
            }
        }
    }

    impl<Err> ProveRounds<Fp, Err> for Kernel {
        fn num_rounds(&self) -> usize {
            ROUNDS
        }

        fn prove_round(
            &mut self,
            bind_challenge: Option<Fp>,
            round: usize,
            _previous_claim: Fp,
        ) -> Result<UnivariatePoly<Fp>, Err> {
            if let Some(r) = bind_challenge {
                bind(&mut self.a, r);
                bind(&mut self.b, r);
            }
            let mut coefficients = vec![fp(0); DEGREE + 1];
            for j in 0..self.a.len() / 2 {
                let weight = self
                    .tau
                    .as_ref()
                    .map_or(fp(1), |tau| eq(&tau[round + 1..], j));
                let (a0, a1) = (self.a[2 * j], self.a[2 * j + 1] - self.a[2 * j]);
                let (b0, b1) = (self.b[2 * j], self.b[2 * j + 1] - self.b[2 * j]);
                coefficients[0] += weight * a0 * b0;
                coefficients[1] += weight * (a0 * b1 + a1 * b0);
                coefficients[2] += weight * a1 * b1;
            }
            Ok(UnivariatePoly::new(coefficients))
        }

        fn finish_rounds(&mut self, bind_challenge: Fp) -> Result<(), Err> {
            bind(&mut self.a, bind_challenge);
            bind(&mut self.b, bind_challenge);
            Ok(())
        }
    }

    fn layout() -> PadLayout {
        PadLayout::new(ROUNDS * DEGREE + 2, 0).unwrap()
    }

    fn pads() -> Vec<Fp> {
        (0..layout().num_pads)
            .map(|i| fp(1000 + i as u64))
            .collect()
    }

    /// Proves one instance over the masked prover's round channel and sends
    /// the output claims masked, then reads both back over the masked
    /// verifier's round channel: the received rounds and the masked output
    /// claims with their slots.
    fn prove(eq_factored: bool) -> (Vec<MaskedRound<Fp>>, [(Fp, usize); 2]) {
        let mut kernel = Kernel::new(eq_factored);
        let instance = kernel.instance();
        let mut committed = CommittedPads::new(layout(), pads(), ()).unwrap();
        let mut tape = MaskedMessages::default();
        let mut transcript = prover_transcript::<H>(&PROTOCOL, &BytesMsg(LABEL));
        let mut channel = committed.channel(&mut tape, &mut transcript);
        let reduction =
            prove_single_with(&mut FullRounds::new(&mut kernel), &instance, &mut channel).unwrap();
        let _: PadSlots = channel.send_private(&[kernel.a[0], kernel.b[0]]).unwrap();
        committed.finish().unwrap();

        let mut transcript = verifier_transcript::<H>(&PROTOCOL, &BytesMsg(LABEL), &[]);
        let mut view = CommittedPadsView::new(layout(), ());
        let mut reader = MaskedReader::new(&tape);
        let mut channel = view.channel(&mut reader, &mut transcript);
        let mut rounds = Vec::new();
        visit_single_rounds(
            instance.shape().unwrap(),
            &mut channel,
            |_, wire, slots, challenge| {
                rounds.push(MaskedRound {
                    wire: wire.into_owned(),
                    slots,
                    challenge,
                });
                Ok(())
            },
        )
        .unwrap();
        let (claims, slots) = channel.receive_private(2).unwrap();
        reader.finish().unwrap();
        view.finish().unwrap();
        let challenges: Vec<Fp> = rounds.iter().map(|round| round.challenge).collect();
        assert_eq!(challenges, reduction.challenges);
        (
            rounds,
            [(claims[0], slots.start), (claims[1], slots.start + 1)],
        )
    }

    /// The relation over the masked rounds, from the clear input claim.
    fn relation(eq_factored: bool, input_claim: Fp) -> ResidualRelation<Fp> {
        let instance = Kernel::new(eq_factored).instance();
        let (rounds, [(a, slot_a), (b, slot_b)]) = prove(eq_factored);
        let format = match instance.format {
            SingleFormat::Compressed => RoundFormat::Compressed,
            SingleFormat::EqFactored { equality_point } => {
                RoundFormat::EqFactored { equality_point }
            }
        };
        let mut relation = ResidualRelation::new(layout(), 0);
        let running = sumcheck_rows(
            &mut relation,
            AffineRow::constant(input_claim),
            &rounds,
            &DegreeSchedule::uniform(ROUNDS, DEGREE),
            &format,
        )
        .unwrap();
        let mut rows = RowsDomain::new(&mut relation, &[], auxiliary_slots(&layout())).unwrap();
        rows.product_into(
            AffineRow::masked(a, PadIndex(slot_a)),
            AffineRow::masked(b, PadIndex(slot_b)),
            running,
        )
        .unwrap();
        rows.finish().unwrap();
        relation
    }

    #[test]
    fn masked_single_instance_satisfies_its_rows() {
        for eq_factored in [false, true] {
            let claim = Kernel::new(eq_factored).instance().input_claim;
            let honest = relation(eq_factored, claim);
            assert_eq!((honest.affine.len(), honest.products.len()), (0, 1));
            assert_eq!(honest.first_violation(&pads(), &[]).unwrap(), None);
            let wrong = relation(eq_factored, claim + fp(1));
            assert!(wrong.first_violation(&pads(), &[]).unwrap().is_some());
        }
    }

    /// An extension element takes one pad per base coordinate, in the order
    /// the verifier's view takes them.
    #[test]
    fn extension_messages_mask_each_coordinate() {
        type E = Ext2<Fp>;
        let value = E::from_base_fn(|index| fp(3 + index as u64));
        let layout = PadLayout::new(2, 0).unwrap();
        let pads = vec![fp(11), fp(13)];
        let mut committed = CommittedPads::new(layout.clone(), pads.clone(), ()).unwrap();
        let coordinates: Vec<Fp> = (0..E::DEGREE)
            .map(|index| value.base_coefficient(index))
            .collect();
        let (slots, masked) = committed.mask(&coordinates).unwrap();
        committed.finish().unwrap();

        let mut view = CommittedPadsView::new(layout, ());
        assert_eq!(view.take_ext(1, E::DEGREE).unwrap(), slots);
        view.finish().unwrap();
        for (index, pad) in pads.iter().enumerate() {
            assert_eq!(masked[index] - *pad, value.base_coefficient(index));
        }
    }

    #[test]
    fn eq_factored_rounds_need_one_coordinate_each() {
        let instance = Kernel::new(true).instance();
        let (rounds, _) = prove(true);
        assert!(matches!(
            sumcheck_rows(
                &mut ResidualRelation::new(layout(), 0),
                AffineRow::constant(instance.input_claim),
                &rounds,
                &DegreeSchedule::uniform(ROUNDS, DEGREE),
                &RoundFormat::EqFactored {
                    equality_point: Point::low_to_high(vec![fp(5), fp(7)]),
                },
            ),
            Err(ZkError::RoundCountMismatch {
                expected: 3,
                got: 2
            })
        ));
    }
}
