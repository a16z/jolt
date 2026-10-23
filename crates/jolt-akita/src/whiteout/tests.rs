//! A Whiteout opening of one hiding trace group, end to end on jolt-akita's
//! API: the pad commitment, masked statement claims, Akita's masked levels,
//! the auxiliary block, the Σ-protocol and the response proof, in the
//! transcript order of `specs/whiteout-jolt-integration.md` §2. The claims
//! are sent masked here; in Jolt they are affine forms of stage messages.

use jolt_algebra::Ring;
use jolt_poly::{MultilinearPoly, OneHotPolynomial};
use jolt_transcript::Blake2b512;
use jolt_transcript::PrivateChannel;
use jolt_zk::masking::{
    AffineRow, CommittedPadsView, OperandShapeDomain, PadIndex, PadLayout, ResidualRelation,
    ResidualRows,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

use super::*;
use crate::adapters::{one_hot_polynomial, reverse_point};
use crate::test_transcripts::{new_prover_transcript, new_verifier_transcript};
use crate::AKITA_ONE_HOT_K16;

const NUM_VARS: usize = 16;
/// Muldiv's trace group shape.
const NUM_POLYS: usize = 56;
const SESSION: &[u8] = b"whiteout-opening";

struct Fixture {
    row: WhiteoutRow,
    tail: WhiteoutTail,
    commitment: CommittedGroup<F>,
    handle: AkitaBackendHint,
    /// Akita's point order.
    point: Vec<F>,
    evaluations: Vec<F>,
}

fn fixture() -> Fixture {
    let row = WhiteoutRow::plan(NUM_VARS, NUM_POLYS).unwrap();
    let rows = 1usize << (NUM_VARS - AKITA_ONE_HOT_K16.ilog2() as usize);
    let polynomials = (0..NUM_POLYS)
        .map(|poly| {
            OneHotPolynomial::new(
                AKITA_ONE_HOT_K16,
                (0..rows)
                    .map(|row| Some(((3 * row + poly) % AKITA_ONE_HOT_K16) as u8))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let source = row
        .backend()
        .unwrap()
        .import_source(
            polynomials
                .iter()
                .map(|poly| {
                    one_hot_polynomial(poly, AKITA_ONE_HOT_K16)
                        .unwrap()
                        .unwrap()
                })
                .collect(),
        )
        .unwrap();
    let (commitment, handle) = row.commit_hiding(&source).unwrap();
    let jolt_point: Vec<F> = (0..NUM_VARS)
        .map(|i| F::from_u64(5 * i as u64 + 2))
        .collect();
    let evaluations = polynomials
        .iter()
        .map(|poly| poly.evaluate(&jolt_point))
        .collect();
    Fixture {
        row,
        tail: WhiteoutTail::new(),
        commitment,
        handle,
        point: reverse_point(&jolt_point),
        evaluations,
    }
}

/// Each claim over its own pad, in send order.
fn own_pads(masked: &[F], first: usize) -> Vec<AffineRow<F>> {
    masked
        .iter()
        .enumerate()
        .map(|(slot, &claim)| AffineRow::masked(claim, PadIndex(first + slot)))
        .collect()
}

/// The pad layout from the counting replay over zero bytes: the claims'
/// slots, Akita's masked messages, then its auxiliary slots.
fn layout(fixture: &Fixture) -> PadLayout {
    let zeros = vec![0u8; 1 << 22];
    let mut transcript = new_verifier_transcript(SESSION, &zeros);
    let mut view = CommittedPadsView::counting();
    let mut claims = vec![F::default(); NUM_POLYS];
    let slots = view
        .channel(&mut transcript)
        .exchange_private_base(&mut claims)
        .unwrap();
    let prefix = fixture
        .row
        .count_masked(
            &fixture.commitment,
            fixture.point.clone(),
            own_pads(&claims, slots.start),
            &mut view,
            &mut transcript,
        )
        .unwrap();
    let mut shape = OperandShapeDomain::new(view.used());
    prefix.shape(&mut shape).unwrap();
    shape.layout().unwrap()
}

/// Verifies through the auxiliary block: the masked prefix and the
/// relation its rows state.
fn replay(
    fixture: &Fixture,
    layout: &PadLayout,
    transcript: &mut VerifierTranscript<'_, Blake2b512>,
) -> (
    MaskedPrefix<F, F>,
    CommittedPadsView<LatticePadPayload<F>>,
    ResidualRelation<F>,
) {
    let mut pads = WhiteoutTail::receive_pads(&fixture.row, layout.clone(), transcript).unwrap();
    let mut claims = vec![F::default(); NUM_POLYS];
    let slots = pads
        .channel(transcript)
        .exchange_private_base(&mut claims)
        .unwrap();
    let prefix = fixture
        .row
        .verify_masked(
            &fixture.commitment,
            fixture.point.clone(),
            own_pads(&claims, slots.start),
            &mut pads,
            transcript,
        )
        .unwrap();
    let mut auxiliary = vec![F::default(); layout.auxiliary.len()];
    let _ = pads
        .channel(transcript)
        .exchange_private_base(&mut auxiliary)
        .unwrap();
    pads.finish().unwrap();
    let mut relation = ResidualRelation::new(layout.clone(), 1);
    prefix.rows(&auxiliary).append_rows(&mut relation).unwrap();
    (prefix, pads, relation)
}

fn prove(fixture: &Fixture, seed: u64) -> Vec<u8> {
    let layout = layout(fixture);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut transcript = new_prover_transcript(SESSION);
    let mut pads = fixture
        .tail
        .commit_pads(&fixture.row, layout.clone(), &mut transcript, &mut rng)
        .unwrap();
    let masked = pads.send(&fixture.evaluations, &mut transcript).unwrap();
    let forms = own_pads(&masked, 0);
    let (residual, auxiliary) = fixture
        .row
        .prove_masked(
            fixture.commitment.clone(),
            fixture.handle.clone(),
            fixture.point.clone(),
            fixture.evaluations.clone(),
            &forms,
            &mut pads,
            &mut transcript,
        )
        .unwrap();
    let _ = pads.send(&auxiliary, &mut transcript).unwrap();
    pads.finish().unwrap();
    let narg = transcript.narg().to_vec();
    let mut replayed = new_verifier_transcript(SESSION, &narg);
    let (_, _, relation) = replay(fixture, &layout, &mut replayed);
    fixture
        .tail
        .prove(
            &fixture.row,
            &relation,
            &residual,
            &pads,
            &mut transcript,
            &mut rng,
        )
        .unwrap();
    transcript.finish()
}

fn verify(fixture: &Fixture, proof: &[u8]) -> Result<(), ZkError> {
    let layout = layout(fixture);
    let mut transcript = new_verifier_transcript(SESSION, proof);
    let (prefix, pads, relation) = replay(fixture, &layout, &mut transcript);
    fixture
        .tail
        .verify(&fixture.row, &relation, &prefix, &pads, &mut transcript)?;
    transcript.finish()?;
    Ok(())
}

#[test]
fn whiteout_opening_verifies_and_rejects_a_flipped_response_byte() {
    let fixture = fixture();
    let proof = prove(&fixture, 7);
    verify(&fixture, &proof).unwrap();
    let mut tampered = proof;
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    assert!(verify(&fixture, &tampered).is_err());
}

/// The honest bound, pinned to Akita's frozen tables: muldiv's `A_p` (d 64,
/// width 898) takes rank 18, whose Σ-binding bracket is 179 < width ≤ 2463,
/// so rank 17 is under the 128-bit target there, as the A40 harness's rank
/// 2 is; `A_D` (key Open, d 64, ℓ∞ 15) takes rank 1 at width 4736, inside
/// the q128 cell's rank-1 width of 93,935. The residual's own ranks are
/// pinned by akita-pcs (`admission_prices_the_residual_at_the_sigma_radius`).
#[test]
fn whiteout_ranks_sit_at_the_tables_minimum() {
    let fixture = fixture();
    let layout = layout(&fixture);
    with_ring_dim!(fixture.row.ring_dim(), D, {
        let pads = WhiteoutTail::pad_commitment::<D>(&layout).unwrap();
        let width =
            LatticePadCommitment::<F, D>::width(pads.placement(), pads.randomness_columns())
                .unwrap();
        assert_eq!((D, width, pads.image_len() / D), (64, 898, 18));
        Ok::<(), ZkError>(())
    })
    .unwrap();
    assert_eq!(sigma_matrix_rank(64, 179).unwrap(), 17);
    assert_eq!(sigma_matrix_rank(64, 180).unwrap(), 18);
    assert_eq!(sigma_matrix_rank(64, 2463).unwrap(), 18);
    assert!(sigma_matrix_rank(64, 2464).unwrap() > 18);

    assert_eq!(digit_matrix_rank(4736).unwrap(), 1);
    assert_eq!(digit_matrix_rank(93_935).unwrap(), 1);
    assert!(digit_matrix_rank(93_936).unwrap() > 1);
}
