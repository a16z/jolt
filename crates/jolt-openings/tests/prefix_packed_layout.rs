#![expect(clippy::unwrap_used, reason = "tests exercise successful layouts")]

use jolt_field::{Fr, Ring};
use jolt_openings::{OpeningsError, PrefixPackedClaims, PrefixPackedLayout};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

#[test]
fn fixed_capacity_layout_preserves_protocol_order() {
    let layout = PrefixPackedLayout::new(2, 4, [30_u64, 10, 20]).unwrap();

    assert_eq!(layout.logical_num_vars(), 2);
    assert_eq!(layout.selector_num_vars(), 2);
    assert_eq!(layout.packed_num_vars(), 4);
    assert_eq!(layout.slot_capacity(), 4);
    assert_eq!(layout.ids(), [30, 10, 20]);
    assert_eq!(layout.slot_index(&10), Some(1));
    assert_eq!(layout.packed_index(&20, 3), Ok(11));
}

#[test]
fn fixed_capacity_layout_rejects_ambiguous_shapes() {
    assert!(matches!(
        PrefixPackedLayout::new(2, 0, [0_u64]),
        Err(OpeningsError::InvalidSetup(_))
    ));
    assert!(matches!(
        PrefixPackedLayout::new(2, 3, [0_u64]),
        Err(OpeningsError::InvalidSetup(_))
    ));
    assert!(matches!(
        PrefixPackedLayout::new(2, 2, [0_u64, 1, 2]),
        Err(OpeningsError::InvalidSetup(_))
    ));
    assert!(matches!(
        PrefixPackedLayout::new(2, 4, [7_u64, 7]),
        Err(OpeningsError::InvalidSetup(_))
    ));
}

#[test]
fn selector_reduction_matches_the_materialized_packed_polynomial() {
    let layout = PrefixPackedLayout::new(2, 4, [0_u64, 1, 2]).unwrap();
    let logical = [
        Polynomial::new(vec![fr(1), fr(2), fr(3), fr(5)]),
        Polynomial::new(vec![fr(7), fr(11), fr(13), fr(17)]),
        Polynomial::new(vec![fr(19), fr(23), fr(29), fr(31)]),
    ];
    let mut packed_evaluations = vec![fr(0); 1 << layout.packed_num_vars()];
    for (slot, polynomial) in logical.iter().enumerate() {
        for (local, value) in polynomial.evaluations().iter().enumerate() {
            packed_evaluations[(slot << layout.logical_num_vars()) | local] = *value;
        }
    }
    let packed = Polynomial::new(packed_evaluations);
    let logical_point = [fr(37), fr(41)];
    let selector = [fr(43), fr(47)];
    let evaluations = logical
        .iter()
        .map(|polynomial| polynomial.evaluate(&logical_point))
        .collect::<Vec<_>>();

    let packed_point = layout.pack_point(&selector, &logical_point).unwrap();
    let packed_evaluation = layout.reduce_evaluations(&selector, &evaluations).unwrap();
    assert_eq!(packed.evaluate(&packed_point), packed_evaluation);

    let unused_slot = [fr(1), fr(1)];
    assert_eq!(
        layout
            .reduce_evaluations(&unused_slot, &evaluations)
            .unwrap(),
        fr(0)
    );
}

#[test]
fn claim_reduction_binds_statement_before_drawing_selector() {
    let layout = PrefixPackedLayout::new(2, 4, [0_u64, 1, 2]).unwrap();
    let claims =
        PrefixPackedClaims::new([9_u8; 32], vec![fr(3), fr(5)], vec![fr(7), fr(11), fr(13)]);
    let mut transcript = Blake2bTranscript::new(b"prefix-packed-claim");
    let reduced = layout.reduce_claims(&claims, &mut transcript).unwrap();

    assert_eq!(reduced.point.len(), layout.packed_num_vars());
    assert_eq!(
        &reduced.point.as_slice()[layout.selector_num_vars()..],
        claims.point()
    );
    assert_eq!(
        reduced.value,
        layout
            .reduce_evaluations(
                &reduced.point.as_slice()[..layout.selector_num_vars()],
                claims.evaluations(),
            )
            .unwrap()
    );

    let mut changed_transcript = Blake2bTranscript::new(b"prefix-packed-claim");
    let changed =
        PrefixPackedClaims::new([9_u8; 32], vec![fr(3), fr(5)], vec![fr(7), fr(11), fr(17)]);
    let changed_reduced = layout
        .reduce_claims(&changed, &mut changed_transcript)
        .unwrap();
    assert_ne!(
        &reduced.point.as_slice()[..layout.selector_num_vars()],
        &changed_reduced.point.as_slice()[..layout.selector_num_vars()]
    );
}
