use super::super::tests::{fixed_backend_from_rows, uniskip_input_claims};
use super::*;
use crate::reference::spartan_product::{ReferenceProductRemainder, SpartanProductKernel};
use crate::ReferenceBackend;
use jolt_field::One;
use jolt_verifier::stages::stage2::product_remainder::product_remainder_input_values_from_uniskip_output;
use jolt_witness::testing::with_sample_backend;

#[test]
#[expect(clippy::unwrap_used, reason = "test oracle")]
fn native_product_matches_reference_rounds_and_openings() {
    // Reuse the existing product reference suite's native rows. This adds the
    // concrete Metal field and exercises every stage, rather than a second fixture.
    let log_t = TILE_ROWS.ilog2() as usize + 1;
    let prefix = TILE_ROWS + 1;
    let mut rows = super::super::tests::synthetic_rows(log_t, 2);
    rows[1].right_instruction_input.0 = i128::MIN;
    rows[1].left_instruction_input.0 = u64::MAX;
    rows[2].lookup_output.0 = u64::MAX;
    let tail = rows[rows.len() - 1];
    rows[prefix..].fill(tail);
    let mut point: Vec<_> = (0..log_t)
        .map(|i| AkitaField::from_u64(31 + 7 * i as u64))
        .collect();
    point[log_t - 1] = AkitaField::zero(); // cubic recovery needs the explicit q(1)
    point[log_t - 2] = AkitaField::one();
    let gpu = MetalProductUniskip::new().unwrap();
    assert!(gpu.admits(prefix, log_t));
    let store = BundleStore::Retained(rows.clone());
    let extension = gpu
        .extension_values(&store.access(), prefix, &point, &extension_coefficients())
        .unwrap();
    drop(store);
    let backend = fixed_backend_from_rows::<AkitaField>(
        log_t,
        &rows,
        #[cfg(feature = "field-inline")]
        &[],
    );
    let uniskip_inputs = uniskip_input_claims::<AkitaField>(
        &rows,
        #[cfg(feature = "field-inline")]
        &[],
        &point,
    );
    let mut reference_session = ProofSession::default();
    reference_session
        .park(SpartanProductKernel::<AkitaField>::prepare(log_t, &point, &backend).unwrap());
    drop(backend);
    let r0 = AkitaField::from_u64(101);
    let tau_high = AkitaField::from_u64(59);
    let expected = ReferenceBackend
        .first_round_poly(&mut reference_session, &[tau_high], &uniskip_inputs)
        .unwrap();
    let mut metal_session = ProofSession::default();
    metal_session.park(SpartanProductCarry {
        log_t,
        prefix_cycles: prefix,
        tau_low: point.clone(),
        rows: BundleStore::Retained(rows),
        #[cfg(feature = "field-inline")]
        field_rows: Vec::new(),
        extension_values: extension,
        compute: Some(Arc::new(gpu)),
    });
    let actual = OptimizedProductUniskip
        .first_round_poly(&mut metal_session, &[tau_high], &uniskip_inputs)
        .unwrap();
    assert_eq!(actual, expected, "uniskip polynomial");
    let mut previous = expected.evaluate(r0);
    let relation = ProductRemainder::new(SpartanProductDimensions::new(log_t), r0, tau_high, point);
    let claims = product_remainder_input_values_from_uniskip_output(previous);
    let points = SumcheckInputPoints::<AkitaField, ProductRemainder<AkitaField>>::default();
    let no_challenges = NoChallenges::<AkitaField>::default();
    let inputs = ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &no_challenges,
    };
    let mut reference = with_sample_backend(|dummy| {
        ReferenceProductRemainder
            .prepare(
                &mut reference_session,
                dummy,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &no_challenges,
                },
            )
            .unwrap()
    });
    let mut metal =
        ProductRemainderKernel::prepare(metal_session.take().unwrap(), &inputs).unwrap();
    assert!(matches!(metal.columns, ProductColumns::Device(_)));
    let mut bind = None;
    for round in 0..log_t {
        let expected = reference.prove_round(bind, round, previous).unwrap();
        let actual = metal.prove_round(bind, round, previous).unwrap();
        assert_eq!(actual, expected, "round {round}");
        let challenge = AkitaField::from_u64(if round < 2 {
            round as u64
        } else {
            73 + 11 * round as u64
        });
        previous = expected.evaluate(challenge);
        bind = Some(challenge);
    }
    let last = bind.unwrap();
    reference.finish_rounds(last).unwrap();
    metal.finish_rounds(last).unwrap();
    assert!(matches!(metal.columns, ProductColumns::Cpu { .. }));
    assert_eq!(
        reference.output_claims(&claims).unwrap(),
        metal.output_claims(&claims).unwrap()
    );
}
