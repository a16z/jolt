use super::*;
use jolt_verifier::stages::stage2::product_remainder::product_remainder_input_values_from_uniskip_output;

#[test]
#[expect(clippy::unwrap_used, reason = "test oracle")]
fn native_product_matches_cpu_rounds_and_openings() {
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
    let expected = extension_t1_values(
        &store.access(),
        prefix,
        &point,
        #[cfg(feature = "field-inline")]
        &[],
    )
    .unwrap();
    assert_eq!(extension, expected);
    let r0 = AkitaField::from_u64(101);
    let tau_high = AkitaField::from_u64(59);
    let relation = ProductRemainder::new(
        SpartanProductDimensions::new(log_t),
        r0,
        tau_high,
        point.clone(),
    );
    let weights = centered_lagrange_evals(DOMAIN, r0).unwrap();
    let scale = centered_lagrange_kernel(DOMAIN, tau_high, r0).unwrap();
    let eq = EqPolynomial::evals(&point, None);
    let mut previous = rows
        .iter()
        .zip(eq)
        .map(|(row, e)| {
            let [left, right] = factors(row, &weights);
            e * left * right
        })
        .sum::<AkitaField>()
        * scale;
    let claims = product_remainder_input_values_from_uniskip_output(previous);
    let points = SumcheckInputPoints::<AkitaField, ProductRemainder<AkitaField>>::default();
    let no_challenges = NoChallenges::<AkitaField>::default();
    let inputs = ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &no_challenges,
    };
    let carry = |compute| SpartanProductCarry {
        log_t,
        prefix_cycles: prefix,
        tau_low: point.clone(),
        rows: BundleStore::Retained(rows.clone()),
        #[cfg(feature = "field-inline")]
        field_rows: Vec::new(),
        extension_values: extension.clone(),
        compute,
    };
    let mut cpu = ProductRemainderKernel::prepare(carry(None), &inputs).unwrap();
    let mut metal = ProductRemainderKernel::prepare(carry(Some(Arc::new(gpu))), &inputs).unwrap();
    assert!(matches!(metal.columns, ProductColumns::Device(_)));
    let mut bind = None;
    for round in 0..log_t {
        let expected = cpu.prove_round(bind, round, previous).unwrap();
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
    cpu.finish_rounds(last).unwrap();
    metal.finish_rounds(last).unwrap();
    assert!(matches!(metal.columns, ProductColumns::Cpu { .. }));
    assert_eq!(
        cpu.output_claims(&claims).unwrap(),
        metal.output_claims(&claims).unwrap()
    );
}
