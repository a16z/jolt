//! BlindFold over committed pads, end to end on a synthetic masked sumcheck:
//! Σ_x a(x)·b(x) = C closed by a·b at the output point, plus a residual
//! evaluation y = a(r) + b(r) inside a hiding evaluation commitment.

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
    Blake2b512, PrivateChannel, ProtocolId, ProverTranscript, VerifierTranscript,
};
use jolt_zk::blindfold::{BlindFold, BlindFoldResidualWitness};
use jolt_zk::masking::{
    lower_factored, sumcheck_rows, AffineRow, CommittedPadsView, MaskingRecorder,
    OperandShapeDomain, PadIndex, PadLayout, ResidualRelation, ResidualVar, RoundFormat,
    RowsDomain, ValuesDomain, ZkBackend, ZkError,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type Backend = BlindFold<Pedersen<Bn254G1>>;

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-zk/blindfold-backend");
const LABEL: &[u8] = b"blindfold-backend";
const NUM_VARS: usize = 3;
const DEGREE: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Opening {
    A,
    B,
}

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn product() -> FactoredExpr<Fr, Opening, (), ()> {
    FactoredExpr::product(vec![opening(Opening::A), opening(Opening::B)])
}

fn setup() -> PedersenSetup<Bn254G1> {
    let generator = Bn254::g1_generator();
    let message_generators = (1..=4).map(|i| generator.scalar_mul(&fr(i))).collect();
    PedersenSetup::new(message_generators, generator.scalar_mul(&fr(99)))
}

fn layout() -> PadLayout {
    let base = NUM_VARS * DEGREE + 2;
    let mut shape = OperandShapeDomain::new(base);
    let _ = lower_factored(
        &product(),
        &mut shape,
        &mut |id| Ok(Some(PadIndex(base - 2 + *id as usize))),
        &mut |_| Ok(fr(0)),
    )
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
fn residual(
    claimed_sum: Fr,
    view: &mut CommittedPadsView<Vec<Bn254G1>>,
    transcript: &mut VerifierTranscript<'_, Blake2b512>,
) -> ResidualRelation<Fr> {
    let auxiliary_len = view.layout().auxiliary.len();
    let mut channel = view.channel(transcript);
    let rounds = receive_masked_rounds::<Fr, _>(
        &DegreeSchedule::uniform(NUM_VARS, DEGREE),
        RoundWire::Compressed,
        &mut channel,
    )
    .unwrap();
    let mut claims = [fr(0); 2];
    let slots = channel.exchange_private_base(&mut claims).unwrap();
    let mut auxiliary = vec![fr(0); auxiliary_len];
    let _ = channel.exchange_private_base(&mut auxiliary).unwrap();

    let mut relation = ResidualRelation::new(view.layout().clone(), 1);
    let running = sumcheck_rows(
        &mut relation,
        AffineRow::constant(claimed_sum),
        &rounds,
        &DegreeSchedule::uniform(NUM_VARS, DEGREE),
        RoundFormat::Compressed,
    )
    .unwrap();
    let a = AffineRow::masked(claims[0], PadIndex(slots.start));
    let b = AffineRow::masked(claims[1], PadIndex(slots.start + 1));
    let mut rows = RowsDomain::new(&mut relation, &auxiliary);
    let output = lower_factored(
        &product(),
        &mut rows,
        &mut |id| {
            Ok(match id {
                Opening::A => a.clone(),
                Opening::B => b.clone(),
            })
        },
        &mut |_: &Source<Opening, (), ()>| -> Result<Fr, ZkError> { unreachable!() },
    )
    .unwrap();
    rows.finish().unwrap();
    relation.affine.push(running.add_scaled(&output, -fr(1)));
    relation.affine.push(
        AffineRow::variable(ResidualVar::Witness(0))
            .add_scaled(&a, -fr(1))
            .add_scaled(&b, -fr(1)),
    );
    relation
}

/// Proves the synthetic statement; `y_offset` shifts the residual evaluation
/// away from its claimed relation.
fn prove(y_offset: Fr) -> Result<Vec<u8>, ZkError> {
    let setup = setup();
    let (mut a, mut b, claimed_sum) = tables();
    let mut rng = ChaCha20Rng::from_seed([7; 32]);
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, LABEL);
    let mut pads = Backend::commit_pads(&setup, &layout(), &mut transcript, &mut rng)?;

    let mut recorder = MaskingRecorder::new(&mut pads);
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
            })
        },
        &mut |_: &Source<Opening, (), ()>| -> Result<Fr, ZkError> { unreachable!() },
    )?;
    let _ = pads.send(&values.auxiliary, &mut transcript)?;

    let witness = BlindFoldResidualWitness {
        evaluation: a[0] + b[0] + y_offset,
        blind: fr(41),
    };
    let residual_commitment =
        Pedersen::<Bn254G1>::commit(&setup, &[witness.evaluation], &witness.blind);

    // The prover's verifier replay over its own argument string.
    let narg = transcript.narg().to_vec();
    let mut replay = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, LABEL, &narg);
    let mut view = Backend::receive_pad_commitment(&setup, &layout(), &mut replay)?;
    let relation = residual(claimed_sum, &mut view, &mut replay);

    Backend::prove(
        &setup,
        &relation,
        pads,
        &residual_commitment,
        &witness,
        &mut transcript,
        &mut rng,
    )?;
    Ok(transcript.finish())
}

fn verify(narg: &[u8], residual_commitment: Bn254G1) -> Result<(), ZkError> {
    let setup = setup();
    let (_, _, claimed_sum) = tables();
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, LABEL, narg);
    let mut view = Backend::receive_pad_commitment(&setup, &layout(), &mut transcript)?;
    let relation = residual(claimed_sum, &mut view, &mut transcript);
    Backend::verify(
        &setup,
        &relation,
        &view,
        &residual_commitment,
        &mut transcript,
    )?;
    transcript.finish().map_err(ZkError::from)
}

fn honest_residual_commitment() -> Bn254G1 {
    let (mut a, mut b, _) = tables();
    // Replay the honest challenges to recover a(r) + b(r).
    let narg = prove(fr(0)).unwrap();
    let setup = setup();
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, LABEL, &narg);
    let mut view = Backend::receive_pad_commitment(&setup, &layout(), &mut transcript).unwrap();
    let mut channel = view.channel(&mut transcript);
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

#[test]
fn honest_proof_verifies() {
    let narg = prove(fr(0)).unwrap();
    verify(&narg, honest_residual_commitment()).unwrap();
}

#[test]
fn prover_rejects_a_relation_its_witness_violates() {
    assert!(matches!(prove(fr(1)), Err(ZkError::InvalidRelation { .. })));
}

#[test]
fn verifier_rejects_a_wrong_residual_commitment() {
    let narg = prove(fr(0)).unwrap();
    let wrong = Pedersen::<Bn254G1>::commit(&setup(), &[fr(5)], &fr(41));
    assert!(verify(&narg, wrong).is_err());
}

#[test]
fn verifier_rejects_any_tampered_byte() {
    let narg = prove(fr(0)).unwrap();
    let commitment = honest_residual_commitment();
    // Every 97th byte keeps the sweep short while touching the pad
    // commitment, the masked messages and the BlindFold tail.
    for index in (0..narg.len()).step_by(97) {
        let mut tampered = narg.clone();
        tampered[index] ^= 1;
        let result = std::panic::catch_unwind(|| verify(&tampered, commitment));
        assert!(
            !matches!(result, Ok(Ok(()))),
            "flipping byte {index} was accepted"
        );
    }
}
