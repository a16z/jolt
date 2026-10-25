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
    shape: Arc<WhiteoutShape>,
    commitment: CommittedGroup<F>,
    handle: AkitaBackendHint,
    /// Akita's point order.
    point: Vec<F>,
    evaluations: Vec<F>,
}

/// Muldiv's masked opening shape: the trace group alone.
fn opening() -> MaskedOpeningShape {
    MaskedOpeningShape {
        precommitted: Vec::new(),
        num_vars: NUM_VARS,
        num_polys: NUM_POLYS,
    }
}

/// A fresh setup of the trace group alone, which plans its catalog.
fn fresh() -> WhiteoutSetup {
    WhiteoutSetup::new(None, WhiteoutSource::Plan, Some((NUM_VARS, NUM_POLYS))).unwrap()
}

impl Fixture {
    /// The same commitment and claims under another setup's `shape`.
    fn with_shape(&self, shape: Arc<WhiteoutShape>) -> Self {
        Self {
            shape,
            commitment: self.commitment.clone(),
            handle: self.handle.clone(),
            point: self.point.clone(),
            evaluations: self.evaluations.clone(),
        }
    }
}

fn fixture() -> Fixture {
    fixture_on(&fresh())
}

fn fixture_on(setup: &WhiteoutSetup) -> Fixture {
    let shape = setup.shape(&opening()).unwrap();
    let row = shape.row();
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
        .catalog()
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
        shape,
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
        .shape
        .row()
        .count_masked(
            &[(fixture.commitment.clone(), fixture.point.clone())],
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
    let mut pads = fixture
        .shape
        .tail()
        .receive_pads(fixture.shape.row(), layout.clone(), transcript)
        .unwrap();
    let mut claims = vec![F::default(); NUM_POLYS];
    let slots = pads
        .channel(transcript)
        .exchange_private_base(&mut claims)
        .unwrap();
    let prefix = fixture
        .shape
        .row()
        .verify_masked(
            &[(fixture.commitment.clone(), fixture.point.clone())],
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
        .shape
        .tail()
        .commit_pads(
            fixture.shape.row(),
            layout.clone(),
            &mut transcript,
            &mut rng,
        )
        .unwrap();
    let masked = pads.send(&fixture.evaluations, &mut transcript).unwrap();
    let forms = own_pads(&masked, 0);
    let compression = fixture
        .shape
        .tail()
        .compression(fixture.shape.row(), &layout)
        .unwrap();
    let (residual, auxiliary) = fixture
        .shape
        .row()
        .prove_masked(
            vec![ProverGroup {
                commitment: fixture.commitment.clone(),
                handle: fixture.handle.clone(),
                point: fixture.point.clone(),
                evaluations: fixture.evaluations.clone(),
            }],
            &forms,
            &mut pads,
            &mut transcript,
            compression.as_ref(),
        )
        .unwrap();
    let _ = pads.send(&auxiliary, &mut transcript).unwrap();
    pads.finish().unwrap();
    let narg = transcript.narg().to_vec();
    let mut replayed = new_verifier_transcript(SESSION, &narg);
    let (_, _, relation) = replay(fixture, &layout, &mut replayed);
    fixture
        .shape
        .tail()
        .prove(
            fixture.shape.row(),
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
    fixture.shape.tail().verify(
        fixture.shape.row(),
        &relation,
        &prefix,
        &pads,
        &mut transcript,
    )?;
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
    with_ring_dim!(fixture.shape.row().ring_dim(), D, {
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

/// `A_p`'s rank and randomness fixed point across Akita's Σ-binding brackets
/// at d 64 (`sigma_binding_infinity_width.csv`: rank 16 to width 65, 17 to
/// 179, 18 to 2463, 19 to 26,544). A layout of `operands` operand pads has
/// base width `32 + 16·operands`, and `r_p` takes 578 columns at rank 18 and
/// 610 at rank 19 (Akita's `hiding_randomness_columns`), so:
/// - the empty layout starts at rank 16 and climbs past rank 17's bracket to
///   rank 18 (width 610); no pad commitment at d 64 lands below rank 18;
/// - 115 operands sit at width 2450, the last step inside rank 18's bracket;
/// - 116 operands would take width 2466 at rank 18, past 2463, so the rank
///   steps to 19 and the wider `r_p` keeps it there (width 2498).
#[test]
fn whiteout_pad_rank_steps_at_the_sigma_binding_brackets() {
    for (operands, rank, randomness_columns, width) in [
        (0, 18, 578, 610),
        (115, 18, 578, 2450),
        (116, 19, 610, 2498),
    ] {
        let mut layout = PadLayout::new(operands, 0).unwrap();
        layout.operand_pads = (0..operands).map(PadIndex).collect();
        let placement = PadPlacement::new(&layout).unwrap();
        assert_eq!(
            WhiteoutTail::pad_rank::<64>(&placement).unwrap(),
            (rank, randomness_columns),
            "{operands} operand pads"
        );
        assert_eq!(
            LatticePadCommitment::<F, 64>::width(&placement, randomness_columns).unwrap(),
            width
        );
    }
}

/// A setup of the trace group alone that loads `plans`, as a setup decoded
/// in another process does.
fn loading(plans: WhiteoutPlans) -> WhiteoutSetup {
    WhiteoutSetup::from(WhiteoutSetupWire {
        grouped: None,
        plans: Some(plans),
    })
}

/// The [`WhiteoutPlanError`] `error` carries, if any.
fn plan_error(error: &ZkError) -> Option<&WhiteoutPlanError> {
    let ZkError::Whiteout(source) = error else {
        return None;
    };
    source.downcast_ref()
}

/// The error a setup loading `plans` refuses `opening` with.
fn refusal(plans: WhiteoutPlans, opening: &MaskedOpeningShape) -> ZkError {
    loading(plans).shape(opening).err().unwrap()
}

/// The byte range and value of the first integer field `name` at or after
/// `from` in `text`.
fn integer_field(text: &str, from: usize, name: &str) -> (usize, usize, usize) {
    let key = format!("\"{name}\":");
    let start = from + text[from..].find(&key).unwrap() + key.len();
    let end = start + text[start..].find(|c: char| !c.is_ascii_digit()).unwrap();
    (start, end, text[start..end].parse().unwrap())
}

/// `plans`' trace catalog with the residual commitment's `A_a` (the inner
/// matrix of the cut successor's one group) one rank lower and `B_a`'s
/// width narrowed to match, so the row stays geometrically consistent and
/// only the rank is short; edited in place in Akita's canonical bytes.
fn lower_residual_rank(plans: &WhiteoutPlans) -> Vec<u8> {
    let text = std::str::from_utf8(&plans.catalog.trace).unwrap();
    let folds = text.find("\"recursive_folds\":").unwrap();
    let cut = ZK_CUT_LEVEL as usize;
    let (mut depth, mut element) = (0usize, 0usize);
    let mut successor = None;
    for (offset, byte) in text[folds..].bytes().enumerate() {
        match byte {
            b'[' | b'{' => {
                depth += 1;
                if depth == 2 && element == cut {
                    successor = Some(folds + offset);
                    break;
                }
            }
            b']' | b'}' => depth -= 1,
            b',' if depth == 1 => element += 1,
            _ => {}
        }
    }
    let successor = successor.unwrap();
    let (rank_start, rank_end, rank) = integer_field(text, successor, "output_rank");
    let outer = successor + text[successor..].find("\"outer\":").unwrap();
    let (width_start, width_end, width) = integer_field(text, outer, "input_width");
    let (_, _, randomness) = integer_field(text, successor, "randomness_columns");
    let lowered_width = width - (width - randomness) / rank;
    assert!(rank_end < width_start);
    format!(
        "{}{}{}{}{}",
        &text[..rank_start],
        rank - 1,
        &text[rank_end..width_start],
        lowered_width,
        &text[width_end..]
    )
    .into_bytes()
}

/// A fresh setup persists its catalog's plan; a setup decoded from it
/// loads exactly that catalog and re-serializes to the same bytes, as does
/// an independent fresh setup. Every tampered or foreign plan is refused
/// with its [`WhiteoutPlanError`] rather than replanned.
#[test]
fn whiteout_persisted_catalog_is_validated_on_load() {
    let encode = |setup: &WhiteoutSetup| {
        bincode::serde::encode_to_vec(setup, bincode::config::standard()).unwrap()
    };
    let planned = fresh();
    let digest = planned
        .catalog(NUM_VARS, NUM_POLYS)
        .unwrap()
        .scheme
        .schedules()
        .catalog_digest();
    let plans = planned.plans().unwrap();
    assert_eq!(plans.responses, None);
    let bytes = encode(&planned);
    let (decoded, _): (WhiteoutSetup, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    assert_eq!(
        decoded
            .catalog(NUM_VARS, NUM_POLYS)
            .unwrap()
            .scheme
            .schedules()
            .catalog_digest(),
        digest
    );
    assert_eq!(encode(&decoded), bytes);
    assert_eq!(encode(&fresh()), bytes);

    // A residual `A_a` one rank below its Σ-binding rank, in an otherwise
    // consistent row, is refused by Akita's admission.
    let mut lowered = plans.clone();
    lowered.catalog.trace = lower_residual_rank(&plans);
    let error = refusal(lowered, &opening());
    assert!(matches!(
        plan_error(&error),
        Some(WhiteoutPlanError::Inadmissible(_))
    ));
    assert!(
        error.to_string().contains("below its Σ-binding rank"),
        "{error}"
    );

    // Another statement: the plan of a narrower trace group, relabelled or
    // not, has no row for a wider one; a trace catalog in a precommitted
    // group's place.
    let wider = MaskedOpeningShape {
        num_polys: NUM_POLYS + 1,
        ..opening()
    };
    let mut relabelled = plans.clone();
    relabelled.catalog.num_polys += 1;
    for foreign in [relabelled, plans.clone()] {
        assert!(matches!(
            plan_error(&refusal(foreign, &wider)),
            Some(WhiteoutPlanError::Statement(_))
        ));
    }
    let mut swapped = plans.clone();
    swapped
        .catalog
        .precommitted
        .push(plans.catalog.trace.clone());
    assert!(matches!(
        plan_error(&refusal(swapped, &opening())),
        Some(WhiteoutPlanError::Statement(_))
    ));

    let mut versioned = plans;
    versioned.version += 1;
    assert!(matches!(
        plan_error(&refusal(versioned, &opening())),
        Some(WhiteoutPlanError::Format { .. })
    ));
}

/// The error the pads of `fixture`'s proof are refused with.
fn pads_refusal(fixture: &Fixture, proof: &[u8]) -> ZkError {
    let mut transcript = new_verifier_transcript(SESSION, proof);
    fixture
        .shape
        .tail()
        .receive_pads(fixture.shape.row(), layout(fixture), &mut transcript)
        .err()
        .unwrap()
}

/// A proof verifies against a setup decoded with the prover's exported
/// plans, whose response set is then exhaustive: a length it lacks is
/// refused, not planned, and a response plan that is not a response
/// schedule is refused, both at the pads.
#[test]
fn whiteout_persisted_response_plan_verifies_and_is_validated() {
    let setup = fresh();
    let fixture = fixture_on(&setup);
    let proof = prove(&fixture, 7);
    verify(&fixture, &proof).unwrap();

    let plans = setup.export().unwrap();
    assert_eq!(plans.responses.as_ref().map(BTreeMap::len), Some(1));
    let bytes =
        bincode::serde::encode_to_vec(loading(plans.clone()), bincode::config::standard()).unwrap();
    let (decoded, _): (WhiteoutSetup, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    verify(
        &fixture.with_shape(decoded.shape(&opening()).unwrap()),
        &proof,
    )
    .unwrap();

    let mut missing = plans.clone();
    missing.responses = Some(BTreeMap::new());
    let error = pads_refusal(
        &fixture.with_shape(loading(missing).shape(&opening()).unwrap()),
        &proof,
    );
    assert!(matches!(
        plan_error(&error),
        Some(WhiteoutPlanError::Statement(_))
    ));

    let mut swapped = plans.clone();
    for response in swapped.responses.iter_mut().flat_map(BTreeMap::values_mut) {
        response.clone_from(&plans.catalog.trace);
    }
    let error = pads_refusal(
        &fixture.with_shape(loading(swapped).shape(&opening()).unwrap()),
        &proof,
    );
    assert!(matches!(
        plan_error(&error),
        Some(WhiteoutPlanError::Inadmissible(_))
    ));
}
