use super::GruenSplitEqPolynomial;
use crate::{BindingOrder, EqPolynomial, PointRef, UnivariatePoly};
use jolt_field::{Field, FpExt4, Fr, Prime128Offset275, Prime32Offset99, Ring};
use num_traits::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[test]
fn prepared_recovery_obeys_round_products_and_claims() {
    recovery_identities::<Fr>();
    recovery_identities::<Prime128Offset275>();
    recovery_identities::<FpExt4<Prime32Offset99>>();
}

fn recovery_identities<F: Field>() {
    let mut saw_zero_at_one = false;
    let mut saw_zero_factor = false;
    for coords in [
        vec![F::from_u64(3); 5],
        vec![F::zero(), F::one(), F::from_u64(7), F::one(), F::zero()],
    ] {
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            for scale in [F::one(), F::zero()] {
                let mut split = GruenSplitEqPolynomial::new_with_scaling(
                    &PointRef::low_to_high(&coords),
                    order,
                    Some(scale),
                );
                for round in 0..coords.len() {
                    let (l0, l1) = split.round().unwrap().linear_evals();
                    assert_eq!(
                        split.round().unwrap().prepare_linear_q_recovery().is_some(),
                        !l1.is_zero()
                    );
                    if let Some(prepared) = split.round().unwrap().prepare_linear_q_recovery() {
                        for degree in 1..=4 {
                            let coefficients: Vec<F> = (0..=degree)
                                .map(|i| F::from_u64(3 + i as u64 * 4))
                                .collect();
                            let q = UnivariatePoly::new(coefficients.clone());
                            let claim = l0 * q.evaluate(F::zero()) + l1 * q.evaluate(F::one());
                            let mut except_linear = coefficients.clone();
                            let _ = except_linear.remove(1);
                            let recovered = prepared
                                .gruen_poly_from_coeffs_except_linear(&except_linear, claim);
                            for x in 0..8 {
                                let x = F::from_u64(x);
                                assert_eq!(
                                    recovered.evaluate(x),
                                    (l0 * (F::one() - x) + l1 * x) * q.evaluate(x)
                                );
                            }
                        }
                        for claim in [F::zero(), F::one(), F::from_u64(17)] {
                            let cubic =
                                prepared.gruen_poly_deg_3(F::from_u64(5), F::from_u64(4), claim);
                            assert_eq!(cubic.evaluate(F::zero()) + cubic.evaluate(F::one()), claim);
                            let general = prepared.gruen_poly_from_coeffs_except_linear(
                                &[F::from_u64(5), F::from_u64(4)],
                                claim,
                            );
                            for x in 0..8 {
                                let x = F::from_u64(x);
                                assert_eq!(cubic.evaluate(x), general.evaluate(x));
                            }
                        }
                        assert_eq!(
                            prepared
                                .gruen_poly_from_coeffs_except_linear(&[], F::one())
                                .coefficients(),
                            &[F::zero()]
                        );
                        let untrimmed = prepared
                            .gruen_poly_from_coeffs_except_linear(&[F::zero(); 2], F::zero());
                        assert_eq!(untrimmed.coefficients(), &[F::zero(); 4]);
                        let zero = prepared.gruen_poly_deg_3(F::zero(), F::zero(), F::zero());
                        assert_eq!(zero.coefficients(), &[F::zero()]);
                    } else {
                        assert!(l1.is_zero());
                        saw_zero_at_one = true;
                        saw_zero_factor |= l0.is_zero();
                    }
                    let q =
                        UnivariatePoly::new(vec![F::from_u64(5), F::from_u64(9), F::from_u64(4)]);
                    let s = split.round().unwrap().gruen_mul(q.coefficients());
                    for x in 0..10 {
                        let x = F::from_u64(x);
                        assert_eq!(
                            s.evaluate(x),
                            (l0 * (F::one() - x) + l1 * x) * q.evaluate(x)
                        );
                    }
                    split
                        .round_mut()
                        .unwrap()
                        .bind(F::from_u64(round as u64 + 11));
                }
            }
        }
    }
    assert!(saw_zero_at_one && saw_zero_factor);
}

#[test]
fn little_endian_gruen_tables_reconstruct_full_eq_at_every_round() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xBB);
    for n in 1..=20 {
        let tau: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let mut full = EqPolynomial::evals(&PointRef::low_to_high(&tau), None);
        let mut split = GruenSplitEqPolynomial::new_with_scaling(
            &PointRef::low_to_high(&tau),
            BindingOrder::LowToHigh,
            None,
        );
        for (round, &coordinate) in tau.iter().enumerate() {
            let (inner, outer) = split.remaining_tables_after(0).unwrap();
            let (l0, l1) = split.round().unwrap().linear_evals();
            assert_eq!(split.round().unwrap().coordinate(), coordinate);
            for j in 0..full.len() / 2 {
                let remaining = inner[j % inner.len()] * outer[j / inner.len()];
                assert_eq!(l0 * remaining, full[2 * j], "n={n}, round={round}, j={j}");
                assert_eq!(
                    l1 * remaining,
                    full[2 * j + 1],
                    "n={n}, round={round}, j={j}"
                );
            }
            let r = Fr::random(&mut rng);
            for j in 0..full.len() / 2 {
                full[j] = full[2 * j] + r * (full[2 * j + 1] - full[2 * j]);
            }
            full.truncate(full.len() / 2);
            split.round_mut().unwrap().bind(r);
        }
        assert_eq!(split.current_scalar(), full[0]);
    }
}

#[test]
fn remaining_tables_after_matches_bound_tables_in_both_orders() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xBC);
    for n in 1..=20 {
        let tau: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            let mut state = GruenSplitEqPolynomial::new(&tau, order);
            for round in 0..n {
                let mut future = state.clone();
                for binds in 0..n - round {
                    assert_eq!(
                        state.remaining_tables_after(binds),
                        Some((future.e_in_current(), future.e_out_current()))
                    );
                    future.round_mut().unwrap().bind(Fr::random(&mut rng));
                }
                assert!(state.remaining_tables_after(n - round).is_none());
                assert!(state.remaining_tables_after(usize::MAX).is_none());
                state.round_mut().unwrap().bind(Fr::random(&mut rng));
            }
        }
    }
}

#[test]
fn finished_states_equal_empty_at_the_final_scalar() {
    for n in 0..=20 {
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            for scale in [Fr::one(), Fr::zero(), Fr::from_u64(7)] {
                for coordinate in [Fr::zero(), Fr::from_u64(3)] {
                    let point = vec![coordinate; n];
                    let mut state =
                        GruenSplitEqPolynomial::new_with_scaling(&point, order, Some(scale));
                    for _ in 0..n {
                        state.round_mut().unwrap().bind(Fr::one());
                    }
                    let expected_scalar = (0..n).fold(scale, |product, _| product * coordinate);
                    assert_eq!(state.current_scalar(), expected_scalar);
                    let scalar = state.current_scalar();
                    let empty =
                        GruenSplitEqPolynomial::new_with_scaling(&[] as &[Fr], order, Some(scalar));
                    for finished in [&state, &empty] {
                        assert_eq!(finished.remaining_rounds(), 0);
                        assert_eq!(finished.current_scalar(), scalar);
                        assert_eq!(finished.e_in_current(), &[Fr::one()]);
                        assert_eq!(finished.e_out_current(), &[Fr::one()]);
                        assert_eq!(finished.merge().evaluations(), &[scalar]);
                        for binds in [0, 1, usize::MAX] {
                            assert!(finished.remaining_tables_after(binds).is_none());
                        }
                        let mut copy = finished.clone();
                        assert!(copy.round().is_none());
                        assert!(copy.round_mut().is_none());
                        assert_eq!(&copy, finished);
                    }
                }
            }
        }
    }
}

#[test]
fn finished_round_handles_are_absent_in_both_orders() {
    for n in 0..=20 {
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            let mut state = GruenSplitEqPolynomial::<Fr>::new(&vec![Fr::one(); n], order);
            for remaining in (1..=n).rev() {
                assert_eq!(state.remaining_rounds(), remaining);
                let handle = state.round_mut().unwrap();
                assert_eq!(handle.round().coordinate(), Fr::one());
                handle.bind(Fr::one());
            }
            assert!(state.round().is_none());
            assert!(state.round_mut().is_none());
        }
    }
}

fn generic_callers<F: Field>() {
    let point = [F::one(); 3];
    let mut s = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let _ = GruenSplitEqPolynomial::new_with_scaling(&point, BindingOrder::LowToHigh, None);
    let _ = GruenSplitEqPolynomial::new_with_scaling(&point, BindingOrder::LowToHigh, None);
    let _ = s.current_scalar();
    let _ = s.remaining_rounds();
    let _ = s.round().unwrap().coordinate();
    let _ = s.round().unwrap().linear_evals();
    let _ = s.remaining_tables_after(0);
    let prepared = s.round().unwrap().prepare_linear_q_recovery().unwrap();
    let _ = prepared.gruen_poly_deg_3(F::zero(), F::zero(), F::zero());
    let _ = prepared.gruen_poly_from_coeffs_except_linear(&[], F::zero());
    let _ = s.round().unwrap().gruen_mul(&[]);
    let _ = s
        .round()
        .unwrap()
        .gruen_poly_deg_2(F::zero(), F::zero(), F::zero)
        .unwrap();
    let _ = s
        .round()
        .unwrap()
        .gruen_poly_deg_3(F::zero(), F::zero(), F::zero(), F::zero)
        .unwrap();
    let _ = s
        .round()
        .unwrap()
        .gruen_poly_from_evals(&[F::zero(); 2], F::zero(), F::zero)
        .unwrap();
    let _ = s.merge();
    let _ = s.e_out_in_for_window(1);
    let _ = s.e_active_for_window(1);
    let _ = s.e_in_current();
    let _ = s.e_out_current();
    let _ = s.e_in_current_len();
    let _ = s.e_out_current_len();
    let _ = s.group_index(0, 0);
    let _ = s.suffix_weight(0);
    let _ = GruenSplitEqPolynomial::<F>::low_to_high_bytes(3);
    let _ = s.par_fold_out_in(F::zero, |a, _, _, w| *a += w, |_, w, a| w * a, |a, b| a + b);
    let _ = s.par_fold_out_in_prefix(
        1,
        F::zero,
        |a, _, _, w| *a += w,
        |_, w, a| w * a,
        |a, b| a + b,
    );
    s.round_mut().unwrap().bind(F::one());
    s.round_mut().unwrap().bind(F::one());
}

#[test]
fn gruen_methods_require_only_field() {
    generic_callers::<Prime128Offset275>();
    generic_callers::<FpExt4<Prime32Offset99>>();
}
