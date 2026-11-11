use jolt_field::{Field, FpExt4, Fr, Prime128Offset6A23801F, Prime32Offset99, Ring};
use num_traits::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use std::cell::Cell;

#[test]
fn gruen_prefix_and_suffix_cover_the_remaining_equality_domain() {
    for bits in 1..9 {
        let point = random_point(bits, 319 + bits as u64);
        let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);
        for round in 0..bits {
            let weights = EqPolynomial::<Fr>::evals(&point[..bits - round - 1], None);
            for len in 0..=weights.len() {
                let prefix = split.par_fold_out_in_prefix(
                    len,
                    Fr::zero,
                    |sum, _, _, weight| *sum += weight,
                    |_, weight, sum| weight * sum,
                    |a, b| a + b,
                );
                assert_eq!(prefix, weights[..len].iter().copied().sum::<Fr>());
                assert_eq!(
                    split.suffix_weight(len),
                    weights[len..].iter().copied().sum::<Fr>()
                );
            }
            split.bind(Fr::from_u64(5 + round as u64));
        }
    }
}

use super::*;
use crate::{eq_index, Endianness, PointRef, TensorEqTable, LOW_TO_HIGH};

fn random_point(len: usize, seed: u64) -> Vec<Fr> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..len).map(|_| Fr::random(&mut rng)).collect()
}

#[test]
#[should_panic(expected = "split-eq suffix weights are only defined for low-to-high")]
fn suffix_weight_rejects_high_to_low_binding() {
    let split = GruenSplitEqPolynomial::<Fr>::new(&random_point(4, 581), BindingOrder::HighToLow);
    let _ = split.suffix_weight(1);
}

#[test]
fn tensor_eq_table_factors_full_eq_table() {
    for vars in 0..=10 {
        let point = random_point(vars, 100 + vars as u64);
        for scale in [Fr::zero(), Fr::one(), Fr::from_u64(7)] {
            let full = EqPolynomial::<Fr>::evals(&point, Some(scale));
            let tensors = std::iter::once(TensorEqTable::new_with_scaling(&point, scale)).chain(
                (0..=vars)
                    .map(|split| TensorEqTable::new_with_scaling_and_split(&point, scale, split)),
            );
            for tensor in tensors {
                assert_eq!(tensor.len(), full.len());
                for x_out in 0..tensor.e_out().len() {
                    for x_in in 0..tensor.e_in().len() {
                        let row = tensor.group_index(x_out, x_in);
                        assert_eq!(tensor.e_out()[x_out] * tensor.e_in()[x_in], full[row]);
                    }
                }
            }
        }
    }
}

#[test]
fn tensor_eq_fold_matches_full_table_dot_product() {
    for point in [
        random_point(9, 211),
        vec![Fr::zero(); 9],
        vec![Fr::one(); 9],
    ] {
        let values = random_point(1 << point.len(), 307);
        let tensor = TensorEqTable::<Fr>::new(&point);
        let eq = EqPolynomial::<Fr>::evals(&point, None);
        for prefix in 0..=values.len() {
            let folded = tensor.par_fold_out_in_prefix(
                prefix,
                Fr::zero,
                |inner, row, _x_in, e_in| {
                    *inner += e_in * values[row];
                },
                |_x_out, e_out, inner| e_out * inner,
                |left, right| left + right,
            );
            let full: Fr = eq[..prefix]
                .iter()
                .zip(&values)
                .map(|(eq, value)| *eq * *value)
                .sum();
            assert_eq!(folded, full);
        }
    }
}

#[test]
fn tensor_eq_evaluates_slices_in_one_fold() {
    let point = random_point(8, 811);
    let values = [
        random_point(1 << point.len(), 907),
        random_point(1 << point.len(), 1009),
        random_point(1 << point.len(), 1103),
    ];
    let slices = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let tensor = TensorEqTable::<Fr>::new(&point);
    let actual = tensor.evaluate_slices(&slices);
    let eq = EqPolynomial::<Fr>::evals(&point, None);
    let expected = values
        .iter()
        .map(|values| {
            eq.iter()
                .zip(values)
                .map(|(&eq, &value)| eq * value)
                .sum::<Fr>()
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn gruen_low_to_high_merge_matches_bound_eq() {
    let point = random_point(10, 401);
    let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);
    let mut dense = Polynomial::new(EqPolynomial::<Fr>::evals(&point, None));
    assert_eq!(split.merge(), dense);

    let challenges = random_point(point.len(), 509);
    for challenge in challenges {
        split.bind(challenge);
        dense.bind_with_order(challenge, BindingOrder::LowToHigh);
        assert_eq!(split.merge(), dense);
    }
}

#[test]
fn gruen_high_to_low_merge_matches_bound_eq() {
    let point = random_point(10, 601);
    let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::HighToLow);
    let mut dense = Polynomial::new(EqPolynomial::<Fr>::evals(&point, None));
    assert_eq!(split.merge(), dense);

    let challenges = random_point(point.len(), 709);
    for challenge in challenges {
        split.bind(challenge);
        dense.bind_with_order(challenge, BindingOrder::HighToLow);
        assert_eq!(split.merge(), dense);
    }
}

fn eq_factor(w: Fr, c: Fr) -> Fr {
    (Fr::one() - w) * (Fr::one() - c) + w * c
}

/// The current round's eq factor `l(X) = scalar * ((1-w)(1-X) + wX)`,
/// built by hand from the point coordinate and an independently tracked
/// scalar, never from the struct's internals.
fn hand_built_linear(scalar: Fr, w: Fr) -> UnivariatePoly<Fr> {
    UnivariatePoly::new(vec![scalar * (Fr::one() - w), scalar * (w + w - Fr::one())])
}

fn current_round_variable(
    split: &GruenSplitEqPolynomial<Fr>,
    point: &[Fr],
    order: BindingOrder,
) -> Fr {
    match order {
        BindingOrder::LowToHigh => point[split.current_index() - 1],
        BindingOrder::HighToLow => point[split.current_index()],
    }
}

#[test]
fn gruen_poly_deg_3_equals_hand_built_linear_times_quadratic() {
    for (order, seed) in [
        (BindingOrder::LowToHigh, 1301u64),
        (BindingOrder::HighToLow, 1409),
    ] {
        let point = random_point(6, seed);
        let challenges = random_point(3, seed + 1);
        let q = UnivariatePoly::new(random_point(3, seed + 2));

        let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, order);
        let mut hand_scalar = Fr::one();
        for (round, challenge) in challenges.into_iter().enumerate() {
            let w = current_round_variable(&split, &point, order);
            let l = hand_built_linear(hand_scalar, w);
            let hint = l.evaluate(Fr::zero()) * q.evaluate(Fr::zero())
                + l.evaluate(Fr::one()) * q.evaluate(Fr::one());
            let s = split
                .gruen_poly_deg_3(q.coefficients()[0], q.coefficients()[2], hint, || {
                    q.evaluate(Fr::one())
                })
                .unwrap();
            assert_eq!(s.coefficients().len(), 4, "{order:?} round {round}");
            // s and l*q both have degree <= 3, so agreement on 8 points
            // plus a random one forces polynomial equality
            for x in
                (0..8u64)
                    .map(Fr::from_u64)
                    .chain([Fr::random(&mut ChaCha20Rng::seed_from_u64(
                        seed + 3 + round as u64,
                    ))])
            {
                assert_eq!(
                    s.evaluate(x),
                    l.evaluate(x) * q.evaluate(x),
                    "{order:?} round {round}"
                );
            }
            hand_scalar *= eq_factor(w, challenge);
            split.bind(challenge);
        }
    }
}

#[test]
fn gruen_poly_from_evals_equals_hand_built_linear_times_q() {
    for (order, seed) in [
        (BindingOrder::LowToHigh, 2003u64),
        (BindingOrder::HighToLow, 2087),
    ] {
        // gruen_poly_from_evals requires degree >= 2: q_evals[0] must be q(1)
        for degree in [2usize, 3] {
            let point = random_point(5, seed + degree as u64);
            let challenges = random_point(2, seed + 10 + degree as u64);
            let q = UnivariatePoly::new(random_point(degree + 1, seed + 20 + degree as u64));

            let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, order);
            let mut hand_scalar = Fr::one();
            for &challenge in &challenges {
                let w = current_round_variable(&split, &point, order);
                hand_scalar *= eq_factor(w, challenge);
                split.bind(challenge);
            }

            let w = current_round_variable(&split, &point, order);
            let l = hand_built_linear(hand_scalar, w);
            // Toom layout: [q(1), ..., q(degree-1), leading coefficient]
            let mut q_evals: Vec<Fr> = (1..degree as u64)
                .map(|x| q.evaluate(Fr::from_u64(x)))
                .collect();
            q_evals.push(q.coefficients()[degree]);
            let hint = l.evaluate(Fr::zero()) * q.evaluate(Fr::zero())
                + l.evaluate(Fr::one()) * q.evaluate(Fr::one());

            let s = split
                .gruen_poly_from_evals(&q_evals, hint, || q.evaluate(Fr::zero()))
                .unwrap();
            assert_eq!(s.coefficients().len(), degree + 2, "{order:?} deg {degree}");
            for x in (0..2 * degree as u64 + 3).map(Fr::from_u64) {
                assert_eq!(
                    s.evaluate(x),
                    l.evaluate(x) * q.evaluate(x),
                    "{order:?} deg {degree}"
                );
            }
        }
    }
}

fn exceptional_rounds_match_products<F: JoltField>() {
    for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
        for coordinate in [F::zero(), F::one(), F::from_u64(7)] {
            for zero_prefix in [false, true] {
                let mut split = GruenSplitEqPolynomial::new_with_scaling(
                    &[coordinate],
                    order,
                    Some(if zero_prefix {
                        F::zero()
                    } else {
                        F::from_u64(11)
                    }),
                );
                for killed_prefix in [false, true] {
                    if killed_prefix {
                        let point = match order {
                            BindingOrder::LowToHigh => vec![coordinate, F::zero()],
                            BindingOrder::HighToLow => vec![F::zero(), coordinate],
                        };
                        split = GruenSplitEqPolynomial::new_with_scaling(
                            &point,
                            order,
                            Some(F::from_u64(11)),
                        );
                        split.bind(F::one());
                    }
                    let scalar = if zero_prefix || killed_prefix {
                        F::zero()
                    } else {
                        F::from_u64(11)
                    };
                    let l_zero = scalar * (F::one() - coordinate);
                    let l_one = scalar * coordinate;
                    for degree in [1usize, 2, 3] {
                        let q = UnivariatePoly::new(
                            (0..=degree)
                                .map(|i| F::from_u64(2 + 3 * i as u64))
                                .collect(),
                        );
                        let hint = l_zero * q.evaluate(F::zero()) + l_one * q.evaluate(F::one());
                        let called = Cell::new(false);
                        let polynomial = if degree == 1 {
                            split
                                .gruen_poly_deg_2(q.evaluate(F::one()), hint, || {
                                    called.set(true);
                                    q.evaluate(F::zero())
                                })
                                .unwrap()
                        } else {
                            let mut samples: Vec<F> = (1..degree)
                                .map(|i| q.evaluate(F::from_u64(i as u64)))
                                .collect();
                            samples.push(q.coefficients()[degree]);
                            split
                                .gruen_poly_from_evals(&samples, hint, || {
                                    called.set(true);
                                    q.evaluate(F::zero())
                                })
                                .unwrap()
                        };
                        assert_eq!(called.get(), !scalar.is_zero() && l_zero.is_zero());
                        if !scalar.is_zero() && l_zero.is_zero() {
                            let bad_hint = hint + F::one();
                            if degree == 1 {
                                assert_eq!(
                                    split.gruen_poly_deg_2(q.evaluate(F::one()), bad_hint, || q
                                        .evaluate(F::zero())),
                                    Err(hint)
                                );
                            } else {
                                let mut samples: Vec<F> = (1..degree)
                                    .map(|i| q.evaluate(F::from_u64(i as u64)))
                                    .collect();
                                samples.push(q.coefficients()[degree]);
                                assert_eq!(
                                    split.gruen_poly_from_evals(&samples, bad_hint, || q
                                        .evaluate(F::zero())),
                                    Err(hint)
                                );
                            }
                        }
                        assert_eq!(polynomial.coefficients().len(), degree + 2);
                        for x in (0..8).map(F::from_u64) {
                            let linear = l_zero + (l_one - l_zero) * x;
                            assert_eq!(polynomial.evaluate(x), linear * q.evaluate(x));
                        }
                        if degree == 2 {
                            called.set(false);
                            let cubic = split
                                .gruen_poly_deg_3(
                                    q.coefficients()[0],
                                    q.coefficients()[2],
                                    hint,
                                    || {
                                        called.set(true);
                                        q.evaluate(F::one())
                                    },
                                )
                                .unwrap();
                            assert_eq!(called.get(), !scalar.is_zero() && l_one.is_zero());
                            assert_eq!(cubic.coefficients().len(), 4);
                            assert_eq!(cubic.coefficients(), polynomial.coefficients());
                            if !scalar.is_zero() && l_one.is_zero() {
                                assert_eq!(
                                    split.gruen_poly_deg_3(
                                        q.coefficients()[0],
                                        q.coefficients()[2],
                                        hint + F::one(),
                                        || q.evaluate(F::one())
                                    ),
                                    Err(hint)
                                );
                            }
                        }
                    }
                    if scalar.is_zero() {
                        assert_eq!(
                            split.gruen_poly_deg_3(F::one(), F::one(), F::one(), || panic!(
                                "zero factor must not scan"
                            )),
                            Err(F::zero())
                        );
                        assert_eq!(
                            split.gruen_poly_from_evals(
                                &[F::one(), F::one()],
                                F::one(),
                                || panic!("zero factor must not scan")
                            ),
                            Err(F::zero())
                        );
                        assert_eq!(
                            split.gruen_poly_deg_2(F::one(), F::one(), || panic!(
                                "zero factor must not scan"
                            )),
                            Err(F::zero())
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn exceptional_gruen_rounds_match_true_products_bn254() {
    exceptional_rounds_match_products::<Fr>();
}

#[test]
fn exceptional_gruen_rounds_match_true_products_akita_field() {
    exceptional_rounds_match_products::<Prime128Offset6A23801F>();
}

#[test]
fn e_out_in_for_window_factors_naive_head_eq_table() {
    let point = random_point(9, 1601);
    let challenges = random_point(9, 1607);
    let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);

    for &challenge in &challenges {
        let current = split.current_index();
        for window in 1..=current {
            let (e_out, e_in) = split.e_out_in_for_window(window);
            let head_len = current - window;
            let head = EqPolynomial::<Fr>::evals(&point[..head_len], None);
            let in_bits = e_in.len().trailing_zeros() as usize;
            assert_eq!(
                e_out.len() * e_in.len(),
                head.len(),
                "window {window} at index {current}"
            );
            for x_out in 0..e_out.len() {
                for x_in in 0..e_in.len() {
                    assert_eq!(
                        e_out[x_out] * e_in[x_in],
                        head[(x_out << in_bits) | x_in],
                        "window {window} at index {current}, out {x_out}, in {x_in}"
                    );
                }
            }
        }
        let (e_out, e_in) = split.e_out_in_for_window(current + 3);
        assert_eq!((e_out, e_in), (&[Fr::one()][..], &[Fr::one()][..]));
        split.bind(challenge);
    }
}

#[test]
fn e_active_for_window_reconstructs_naive_eq_table_with_linear_factor() {
    let point = random_point(8, 1701);
    let challenges = random_point(3, 1709);
    let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);
    let mut hand_scalar = Fr::one();

    for stage in 0..=challenges.len() {
        if stage > 0 {
            let w = point[split.current_index() - 1];
            hand_scalar *= eq_factor(w, challenges[stage - 1]);
            split.bind(challenges[stage - 1]);
        }
        let current = split.current_index();
        let w_current = point[current - 1];
        let (lin_0, lin_1) = split.current_linear_evals();
        assert_eq!(
            (lin_0, lin_1),
            (
                hand_scalar * (Fr::one() - w_current),
                hand_scalar * w_current
            ),
            "stage {stage}"
        );

        assert_eq!(split.e_active_for_window(0), vec![Fr::one()]);
        assert_eq!(split.e_active_for_window(1), vec![Fr::one()]);
        assert_eq!(split.e_active_for_window(current + 1), vec![Fr::one()]);

        let full = EqPolynomial::<Fr>::evals(&point[..current], None);
        for window in 2..=current {
            let (e_out, e_in) = split.e_out_in_for_window(window);
            let active = split.e_active_for_window(window);
            assert_eq!(active.len(), 1 << (window - 1), "stage {stage}");
            let in_bits = e_in.len().trailing_zeros() as usize;
            for head_index in 0..e_out.len() * e_in.len() {
                let head =
                    e_out[head_index >> in_bits] * e_in[head_index & ((1usize << in_bits) - 1)];
                for (active_index, &active_value) in active.iter().enumerate() {
                    for last_bit in 0..2usize {
                        let index = (((head_index << (window - 1)) | active_index) << 1) | last_bit;
                        let last = if last_bit == 0 {
                            Fr::one() - w_current
                        } else {
                            w_current
                        };
                        assert_eq!(
                            head * active_value * last,
                            full[index],
                            "stage {stage} window {window} index {index}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn new_with_scaling_scales_bound_and_merged_results_by_factor() {
    let scaling = random_point(1, 3001)[0];
    for (order, seed) in [
        (BindingOrder::LowToHigh, 3103u64),
        (BindingOrder::HighToLow, 3203),
    ] {
        let point = random_point(7, seed);
        let challenges = random_point(7, seed + 1);
        let mut scaled =
            GruenSplitEqPolynomial::<Fr>::new_with_scaling(&point, order, Some(scaling));
        let mut plain = GruenSplitEqPolynomial::<Fr>::new(&point, order);

        assert_eq!(scaled.current_scalar(), scaling);
        assert_eq!(scaled.merge(), plain.merge() * scaling);
        for &challenge in &challenges {
            let (scaled_0, scaled_1) = scaled.current_linear_evals();
            let (plain_0, plain_1) = plain.current_linear_evals();
            assert_eq!(
                (scaled_0, scaled_1),
                (plain_0 * scaling, plain_1 * scaling),
                "{order:?}"
            );
            scaled.bind(challenge);
            plain.bind(challenge);
            assert_eq!(
                scaled.current_scalar(),
                plain.current_scalar() * scaling,
                "{order:?}"
            );
            assert_eq!(scaled.merge(), plain.merge() * scaling, "{order:?}");
        }
    }

    let empty =
        GruenSplitEqPolynomial::<Fr>::new_with_scaling(&[], BindingOrder::LowToHigh, Some(scaling));
    assert_eq!(empty.current_scalar(), scaling);
    assert_eq!(empty.merge().evaluations(), &[scaling]);
}

#[test]
fn gruen_par_fold_out_in_matches_sequential_reference_fold() {
    for (order, seed) in [
        (BindingOrder::LowToHigh, 4001u64),
        (BindingOrder::HighToLow, 4103),
    ] {
        let point = random_point(7, seed);
        let challenges = random_point(4, seed + 1);
        let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, order);

        for stage in 0..=challenges.len() {
            if stage > 0 {
                split.bind(challenges[stage - 1]);
            }
            let e_out = split.e_out_current().to_vec();
            let e_in = split.e_in_current().to_vec();
            let values = random_point(e_out.len() * e_in.len(), seed + 10 + stage as u64);
            let out_weights = random_point(e_out.len(), seed + 20 + stage as u64);
            let in_weights = random_point(e_in.len(), seed + 30 + stage as u64);

            let folded = split.par_fold_out_in(
                Fr::zero,
                |inner, row, x_in, e_in_value| {
                    *inner += e_in_value * in_weights[x_in] * values[row];
                },
                |x_out, e_out_value, inner| e_out_value * out_weights[x_out] * inner,
                |left, right| left + right,
            );

            let mut expected = Fr::zero();
            for (x_out, &e_out_value) in e_out.iter().enumerate() {
                let mut inner = Fr::zero();
                for (x_in, &e_in_value) in e_in.iter().enumerate() {
                    inner += e_in_value * in_weights[x_in] * values[x_out * e_in.len() + x_in];
                }
                expected += e_out_value * out_weights[x_out] * inner;
            }
            assert_eq!(folded, expected, "{order:?} stage {stage}");
        }
    }
}

#[test]
fn gruen_current_linear_factor_matches_merged_sumcheck_pair() {
    for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
        let point = random_point(8, 811);
        let challenges = random_point(4, 919);
        let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, order);
        for (round, challenge) in challenges.into_iter().enumerate() {
            let merged = split.merge();
            let (linear_0, linear_1) = split.current_linear_evals();
            for x_out in 0..split.e_out_current_len() {
                for x_in in 0..split.e_in_current_len() {
                    let row = split.group_index(x_out, x_in);
                    let dense_row = match order {
                        BindingOrder::LowToHigh => row,
                        BindingOrder::HighToLow => {
                            let out_bits = split.e_out_current_len().trailing_zeros() as usize;
                            (x_in << out_bits) | x_out
                        }
                    };
                    let head = split.e_out_current()[x_out] * split.e_in_current()[x_in];
                    let (dense_0, dense_1) = merged.sumcheck_eval_pair(dense_row, order);
                    assert_eq!(
                        head * linear_0,
                        dense_0,
                        "{order:?} round {round} row {row} eval 0"
                    );
                    assert_eq!(
                        head * linear_1,
                        dense_1,
                        "{order:?} round {round} row {row} eval 1"
                    );
                }
            }
            split.bind(challenge);
        }
    }
}

fn gruen_constructor_identity<F: Field, const E: Endianness>() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xe12_6775);
    for n in 0..=20 {
        let coords: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
        let point = PointRef::<E, _>::new(&coords);
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            for scaling_factor in [None, Some(F::zero()), Some(F::from_u64(7))] {
                let table =
                    GruenSplitEqPolynomial::<F>::new_with_scaling(&point, order, scaling_factor);
                let scale = scaling_factor.unwrap_or(F::one());
                assert_eq!(table.current_scalar, scale);
                assert_eq!(
                    table.current_index,
                    match order {
                        BindingOrder::LowToHigh => n,
                        BindingOrder::HighToLow => 0,
                    }
                );
                for (k, coord) in table.point.iter().enumerate() {
                    assert_eq!(*coord, point.msb(k));
                }
                assert_eq!(table.e_out_vec[0], [F::one()]);
                assert_eq!(table.e_in_vec[0], [F::one()]);
                let remaining = if n == 0 {
                    point.msb_range(0..0)
                } else {
                    match order {
                        BindingOrder::LowToHigh => point.msb_range(0..n - 1),
                        BindingOrder::HighToLow => point.msb_range(1..n),
                    }
                };
                let outer = table.e_out_vec.last().unwrap();
                let inner = table.e_in_vec.last().unwrap();
                let len = 1 << remaining.num_vars();
                assert_eq!(outer.len() * inner.len(), len);
                for index in [0, len / 3, len - 1] {
                    let factored = match order {
                        BindingOrder::LowToHigh => {
                            outer[index / inner.len()] * inner[index % inner.len()]
                        }
                        BindingOrder::HighToLow => {
                            inner[index / outer.len()] * outer[index % outer.len()]
                        }
                    };
                    assert_eq!(
                        scale * factored,
                        scale * eq_index(&remaining, index as u128),
                        "n={n}, endianness={E}, order={order:?}, index={index}"
                    );
                }
            }
        }
    }
}

#[test]
fn gruen_constructor_obeys_equality_in_both_orders() {
    gruen_constructor_identity::<Fr, HIGH_TO_LOW>();
    gruen_constructor_identity::<Fr, LOW_TO_HIGH>();
    gruen_constructor_identity::<FpExt4<Prime32Offset99>, HIGH_TO_LOW>();
    gruen_constructor_identity::<FpExt4<Prime32Offset99>, LOW_TO_HIGH>();
}
