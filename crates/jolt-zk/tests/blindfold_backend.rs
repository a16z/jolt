//! BlindFold over committed pads, end to end on a synthetic masked sumcheck:
//! Σ_x a(x)·b(x) = C closed by a·b at the output point, plus a residual
//! evaluation y = a(r) + b(r) inside a hiding evaluation commitment.

#![cfg(feature = "prover")]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "integration tests should fail loudly"
)]

use jolt_algebra::{Field, Fr, Ring};
use jolt_claims::{opening, FactoredExpr, Source};
use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{receive_masked_rounds, DegreeSchedule, RoundWire, SumcheckRecorder};
use jolt_transcript::{
    protocol_tag, prover_transcript, verifier_transcript, Blake2b512, TAG_BYTES,
};
use jolt_zk::blindfold::{BlindFold, BlindFoldProof, BlindFoldResidualWitness};
use jolt_zk::masking::{
    lower_factored, sumcheck_rows, AffineRow, CommittedPadsView, MaskedMessages, MaskedReader,
    MaskingRecorder, OperandShapeDomain, PadIndex, PadLayout, ProductMemo, ProductOperand,
    ResidualRelation, ResidualVar, RoundFormat, RowsDomain, ValuesDomain, ZkBackend, ZkError,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type Backend = BlindFold<Pedersen<Bn254G1>>;

const SESSION: [u8; TAG_BYTES] = protocol_tag(b"jolt-zk/blindfold-backend");
const STATEMENT: [u8; 8] = *b"bfbacknd";

/// Everything the synthetic statement's proof carries.
#[derive(Clone, Debug)]
struct Proof {
    pad_commitment: Vec<Bn254G1>,
    masked: MaskedMessages<Fr>,
    blindfold: BlindFoldProof<Fr, Bn254G1>,
}
const NUM_VARS: usize = 3;
const DEGREE: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Opening {
    A,
    B,
}

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

/// `a·b`, as `3·a·b − 2·a·b + (a + b)·(a − b) − a·a + b·b`: five terms whose
/// products land on one five-pair row, which BlindFold lowers with four
/// intermediates. Under BlindFold's affine operands, `(a + b)` is an operand
/// with no defining slot.
fn product() -> FactoredExpr<Fr, Opening, (), ()> {
    let a = || opening(Opening::A);
    let b = || opening(Opening::B);
    let ab = || FactoredExpr::product(vec![a(), b()]);
    ab().scale(fr(3))
        + ab().scale(-fr(2))
        + FactoredExpr::product(vec![a() + b(), a() - b()])
        + FactoredExpr::product(vec![a(), a()]).scale(-fr(1))
        + FactoredExpr::product(vec![b(), b()])
}

fn setup() -> PedersenSetup<Bn254G1> {
    let generator = Bn254::g1_generator();
    let message_generators = (1..=4).map(|i| generator.scalar_mul(&fr(i))).collect();
    PedersenSetup::new(message_generators, generator.scalar_mul(&fr(99)))
}

fn layout() -> PadLayout {
    let base = NUM_VARS * DEGREE + 2;
    let mut shape = OperandShapeDomain::new(base).with_operands(Backend::OPERANDS);
    lower_factored(
        &product(),
        &mut shape,
        &mut ProductMemo::default(),
        &mut |id| Ok(Some(PadIndex(base - 2 + *id as usize))),
        &mut |_| Ok(fr(0)),
    )
    .unwrap()
    .check_equals::<Fr, _>(None, &mut shape)
    .unwrap();
    shape.layout().unwrap()
}

fn fold(table: &[Fr], r: Fr) -> Vec<Fr> {
    let half = table.len() / 2;
    (0..half)
        .map(|j| table[j] + r * (table[j + half] - table[j]))
        .collect()
}

fn round_poly(a: &[Fr], b: &[Fr]) -> UnivariatePoly<Fr> {
    let eval = |x: Fr| -> Fr {
        let (a, b) = (fold(a, x), fold(b, x));
        a.iter().zip(&b).map(|(a, b)| *a * *b).sum()
    };
    let (e0, e1, e2) = (eval(fr(0)), eval(fr(1)), eval(fr(2)));
    let c2 = (e2 - e1 - e1 + e0) * fr(2).inverse().unwrap();
    UnivariatePoly::new(vec![e0, e1 - e0 - c2, c2])
}

fn tables() -> (Vec<Fr>, Vec<Fr>, Fr) {
    let a: Vec<Fr> = (1..=8).map(fr).collect();
    let b: Vec<Fr> = (11..=18).map(fr).collect();
    let sum = a.iter().zip(&b).map(|(a, b)| *a * *b).sum();
    (a, b, sum)
}

/// The verifier's half: reads the masked transcript after the pad
/// commitment and lowers its checks; leaves the transcript at the tail.
fn residual<T: jolt_transcript::FieldTranscript<Fr>>(
    claimed_sum: Fr,
    view: &mut CommittedPadsView<Vec<Bn254G1>>,
    tape: &mut MaskedReader<'_, Fr>,
    transcript: &mut T,
) -> Result<ResidualRelation<Fr>, ZkError> {
    let auxiliary_len = view.layout().auxiliary.len();
    let mut channel = view.channel(tape, transcript);
    let rounds = receive_masked_rounds::<Fr, _>(
        &DegreeSchedule::uniform(NUM_VARS, DEGREE),
        RoundWire::Compressed,
        &mut channel,
    )?;
    let (claims, slots) = channel.receive_private(2)?;
    let (auxiliary, _) = channel.receive_private(auxiliary_len)?;

    let mut relation = ResidualRelation::new(view.layout().clone(), 1);
    let running = sumcheck_rows(
        &mut relation,
        AffineRow::constant(claimed_sum),
        &rounds,
        &DegreeSchedule::uniform(NUM_VARS, DEGREE),
        &RoundFormat::Compressed,
    )
    .unwrap();
    let a = AffineRow::masked(claims[0], PadIndex(slots.start));
    let b = AffineRow::masked(claims[1], PadIndex(slots.start + 1));
    let mut rows = RowsDomain::new(&mut relation, &auxiliary).with_operands(Backend::OPERANDS);
    let output = lower_factored(
        &product(),
        &mut rows,
        &mut ProductMemo::default(),
        &mut |id| {
            Ok(match id {
                Opening::A => a.clone(),
                Opening::B => b.clone(),
            })
        },
        &mut |_: &Source<Opening, (), ()>| -> Result<Fr, ZkError> { unreachable!() },
    )
    .unwrap();
    output.check_equals(running, &mut rows).unwrap();
    rows.finish().unwrap();
    let pairs: Vec<_> = relation.products[0].pairs().collect();
    assert_eq!(pairs.len(), 5);
    assert!(matches!(pairs[2].x, ProductOperand::Affine(_)));
    assert!(view.layout().auxiliary.is_empty());
    relation.affine.push(
        AffineRow::variable(ResidualVar::Witness(0))
            .add_scaled(&a, -fr(1))
            .add_scaled(&b, -fr(1)),
    );
    Ok(relation)
}

/// Proves the synthetic statement; `y_offset` shifts the residual evaluation
/// away from its claimed relation.
fn prove(y_offset: Fr) -> Result<Proof, ZkError> {
    let setup = setup();
    let (mut a, mut b, claimed_sum) = tables();
    let mut rng = ChaCha20Rng::from_seed([7; 32]);
    let mut transcript = prover_transcript::<Blake2b512>(&SESSION, &STATEMENT);
    let (mut pads, pad_commitment) =
        Backend::commit_pads(&setup, &layout(), &mut transcript, &mut rng)?;
    let mut masked = MaskedMessages::default();

    let mut recorder = MaskingRecorder::new(&mut pads, &mut masked);
    for _ in 0..NUM_VARS {
        let r = recorder
            .absorb_round(&round_poly(&a, &b), DEGREE, &mut transcript)
            .unwrap();
        a = fold(&a, r);
        b = fold(&b, r);
    }
    let _ = recorder.finish(&[a[0], b[0]], &mut transcript).unwrap();
    let mut values = ValuesDomain::with_operands(Backend::OPERANDS);
    let _ = lower_factored(
        &product(),
        &mut values,
        &mut ProductMemo::default(),
        &mut |id| {
            Ok(match id {
                Opening::A => a[0],
                Opening::B => b[0],
            })
        },
        &mut |_: &Source<Opening, (), ()>| -> Result<Fr, ZkError> { unreachable!() },
    )?;
    let _ = pads.send(&values.auxiliary, &mut masked, &mut transcript)?;

    let witness = BlindFoldResidualWitness {
        evaluation: a[0] + b[0] + y_offset,
        blind: fr(41),
    };
    let residual_commitment =
        Pedersen::<Bn254G1>::commit(&setup, &[witness.evaluation], &witness.blind);

    // The prover's verifier replay over its own messages.
    let mut replay = verifier_transcript::<Blake2b512>(&SESSION, &STATEMENT, &[]);
    let mut view =
        Backend::receive_pad_commitment(&setup, &layout(), pad_commitment.clone(), &mut replay)?;
    let mut tape = MaskedReader::new(&masked);
    let relation = residual(claimed_sum, &mut view, &mut tape, &mut replay)?;

    let blindfold = Backend::prove(
        &setup,
        &relation,
        pads,
        &residual_commitment,
        &witness,
        &mut transcript,
        &mut rng,
    )?;
    Ok(Proof {
        pad_commitment,
        masked,
        blindfold,
    })
}

fn verify(proof: &Proof, residual_commitment: Bn254G1) -> Result<(), ZkError> {
    let setup = setup();
    let (_, _, claimed_sum) = tables();
    let mut transcript = verifier_transcript::<Blake2b512>(&SESSION, &STATEMENT, &[]);
    let mut view = Backend::receive_pad_commitment(
        &setup,
        &layout(),
        proof.pad_commitment.clone(),
        &mut transcript,
    )?;
    let mut tape = MaskedReader::new(&proof.masked);
    let relation = residual(claimed_sum, &mut view, &mut tape, &mut transcript)?;
    tape.finish()?;
    Backend::verify(
        &setup,
        &relation,
        &view,
        &residual_commitment,
        &proof.blindfold,
        &mut transcript,
    )
}

fn honest_residual_commitment(proof: &Proof) -> Bn254G1 {
    let (mut a, mut b, _) = tables();
    // Replay the honest challenges to recover a(r) + b(r).
    let setup = setup();
    let mut transcript = verifier_transcript::<Blake2b512>(&SESSION, &STATEMENT, &[]);
    let mut view = Backend::receive_pad_commitment(
        &setup,
        &layout(),
        proof.pad_commitment.clone(),
        &mut transcript,
    )
    .unwrap();
    let mut tape = MaskedReader::new(&proof.masked);
    let mut channel = view.channel(&mut tape, &mut transcript);
    let rounds = receive_masked_rounds::<Fr, _>(
        &DegreeSchedule::uniform(NUM_VARS, DEGREE),
        RoundWire::Compressed,
        &mut channel,
    )
    .unwrap();
    for round in &rounds {
        a = fold(&a, round.challenge);
        b = fold(&b, round.challenge);
    }
    Pedersen::<Bn254G1>::commit(&setup, &[a[0] + b[0]], &fr(41))
}

/// Every single-element change to the proof: each field value shifted by
/// one, each commitment moved by the generator.
fn tampers(proof: &Proof) -> Vec<(String, Proof)> {
    let g = Bn254::g1_generator();
    let mut out = Vec::new();
    let mut push = |name: String, edit: &dyn Fn(&mut Proof)| {
        let mut tampered = proof.clone();
        edit(&mut tampered);
        out.push((name, tampered));
    };
    for i in 0..proof.pad_commitment.len() {
        push(format!("pad commitment {i}"), &|p| {
            p.pad_commitment[i] += g;
        });
    }
    for i in 0..proof.masked.values.len() {
        push(format!("masked value {i}"), &|p| {
            p.masked.values[i] += fr(1);
        });
    }
    let bf = &proof.blindfold;
    for i in 0..bf.folding.auxiliary_rows.len() {
        push(format!("auxiliary row {i}"), &|p| {
            p.blindfold.folding.auxiliary_rows[i] += g;
        });
    }
    push("random u".into(), &|p| {
        p.blindfold.folding.random_u += fr(1);
    });
    for i in 0..bf.folding.random_pad_rows.len() {
        push(format!("random pad row {i}"), &|p| {
            p.blindfold.folding.random_pad_rows[i] += g;
        });
    }
    for i in 0..bf.folding.random_error_rows.len() {
        push(format!("random error row {i}"), &|p| {
            p.blindfold.folding.random_error_rows[i] += g;
        });
    }
    push("random eval".into(), &|p| {
        p.blindfold.folding.random_eval += g;
    });
    for i in 0..bf.folding.cross_term_error_rows.len() {
        push(format!("cross-term row {i}"), &|p| {
            p.blindfold.folding.cross_term_error_rows[i] += g;
        });
    }
    push("folded output".into(), &|p| {
        p.blindfold.folded_evaluation.output += fr(1);
    });
    push("folded blinding".into(), &|p| {
        p.blindfold.folded_evaluation.blinding += fr(1);
    });
    push("folded output opening".into(), &|p| {
        p.blindfold
            .folded_evaluation
            .output_opening
            .combined_blinding += fr(1);
    });
    push("folded blinding opening".into(), &|p| {
        p.blindfold
            .folded_evaluation
            .blinding_opening
            .combined_vector[0] += fr(1);
    });
    for round in 0..bf.outer_sumcheck.round_polynomials.len() {
        push(format!("outer round {round}"), &|p| {
            let poly = &mut p.blindfold.outer_sumcheck.round_polynomials[round];
            let mut coeffs = poly.coeffs_except_linear_term().to_vec();
            coeffs[0] += fr(1);
            *poly = jolt_poly::CompressedPoly::new(coeffs);
        });
    }
    for k in 0..3 {
        push(format!("outer claim {k}"), &|p| {
            p.blindfold.outer_claims.abc[k] += fr(1);
        });
    }
    push("error opening".into(), &|p| {
        p.blindfold.outer_claims.error_opening.combined_blinding += fr(1);
    });
    for round in 0..bf.inner_sumcheck.round_polynomials.len() {
        push(format!("inner round {round}"), &|p| {
            let poly = &mut p.blindfold.inner_sumcheck.round_polynomials[round];
            let mut coeffs = poly.coeffs_except_linear_term().to_vec();
            coeffs[0] += fr(1);
            *poly = jolt_poly::CompressedPoly::new(coeffs);
        });
    }
    push("witness opening".into(), &|p| {
        p.blindfold.witness_opening.combined_vector[0] += fr(1);
    });
    push("extra masked value".into(), &|p| {
        p.masked.values.push(fr(0));
    });
    push("short witness opening".into(), &|p| {
        let _ = p.blindfold.witness_opening.combined_vector.pop();
    });
    out
}

#[test]
fn honest_proof_verifies() {
    let proof = prove(fr(0)).unwrap();
    verify(&proof, honest_residual_commitment(&proof)).unwrap();
}

#[test]
fn prover_rejects_a_relation_its_witness_violates() {
    assert!(matches!(prove(fr(1)), Err(ZkError::InvalidRelation { .. })));
}

#[test]
fn verifier_rejects_a_wrong_residual_commitment() {
    let proof = prove(fr(0)).unwrap();
    let wrong = Pedersen::<Bn254G1>::commit(&setup(), &[fr(5)], &fr(41));
    assert!(verify(&proof, wrong).is_err());
}

#[test]
fn verifier_rejects_every_tampered_message() {
    let proof = prove(fr(0)).unwrap();
    let commitment = honest_residual_commitment(&proof);
    // The pad commitment, every masked value and every BlindFold message.
    for (name, tampered) in tampers(&proof) {
        let result = std::panic::catch_unwind(|| verify(&tampered, commitment));
        assert!(
            !matches!(result, Ok(Ok(()))),
            "tampering {name} was accepted"
        );
    }
}
