//! Lower [`CheckpointRule`]s into compiled [`CheckpointEvalAction`]s.
//!
//! At compile time we know `(j, suffix_len, r_x, r_y)`, so the guard selection,
//! weight exponents, and special-case constants can all be baked into the op.
//! The runtime then only needs to evaluate a list of monomials — no interpretation
//! of the rule structure.

use crate::module::{
    BilinearExpr, ChallengeIdx, CheckpointAction, CheckpointEvalAction, CheckpointRule, DefaultVal,
    Monomial, RoundGuard, ScalarExpr, ValueSource, WeightFn,
};

/// Lower one [`CheckpointRule`] at a specific round for the given challenges.
///
/// Returns `None` when the rule is a no-op for this round (e.g. `Passthrough`
/// — the interpreter preserves the slot, so the batch can simply skip it).
pub fn lower_checkpoint_rule(
    rule: &CheckpointRule,
    self_idx: usize,
    r_x: ChallengeIdx,
    r_y: ChallengeIdx,
    j: usize,
    suffix_len: usize,
) -> Option<CheckpointEvalAction> {
    let action = rule
        .cases
        .iter()
        .find(|(g, _)| guard_matches(*g, j, suffix_len))
        .map_or(&rule.fallback, |(_, a)| a);
    if matches!(action, CheckpointAction::Passthrough) {
        return None;
    }
    Some(lower_action(action, rule.default, self_idx, r_x, r_y, j))
}

fn guard_matches(g: RoundGuard, j: usize, suffix_len: usize) -> bool {
    match g {
        RoundGuard::JEq(v) => j == v,
        RoundGuard::JLt(v) => j < v,
        RoundGuard::JGe(v) => j >= v,
        RoundGuard::JGt(v) => j > v,
        RoundGuard::SuffixLenNonZero => suffix_len != 0,
    }
}

fn lower_action(
    action: &CheckpointAction,
    rule_default: DefaultVal,
    self_idx: usize,
    r_x: ChallengeIdx,
    r_y: ChallengeIdx,
    j: usize,
) -> CheckpointEvalAction {
    let cp_self = |d: DefaultVal| ValueSource::Checkpoint {
        idx: self_idx,
        default: d,
    };
    let cp_dep = |idx: usize, d: DefaultVal| ValueSource::Checkpoint { idx, default: d };

    match action {
        CheckpointAction::Mul(expr) => {
            let mut out = ScalarExpr::new();
            for m in bilinear_monomials(*expr, r_x, r_y) {
                out.push(prepend_factor(m, cp_self(rule_default)));
            }
            CheckpointEvalAction::Set(out)
        }
        CheckpointAction::AddWeighted { weight, expr } => {
            let weight_src = weight_value(*weight, j);
            let mut out = vec![Monomial {
                coeff: 1,
                factors: vec![cp_self(rule_default)],
            }];
            for m in bilinear_monomials(*expr, r_x, r_y) {
                out.push(prepend_factor(m, weight_src.clone()));
            }
            CheckpointEvalAction::Set(out)
        }
        CheckpointAction::AddTwoTerm { x_weight, y_weight } => CheckpointEvalAction::Set(vec![
            Monomial {
                coeff: 1,
                factors: vec![cp_self(rule_default)],
            },
            Monomial {
                coeff: 1,
                factors: vec![weight_value(*x_weight, j), ValueSource::Challenge(r_x)],
            },
            Monomial {
                coeff: 1,
                factors: vec![weight_value(*y_weight, j), ValueSource::Challenge(r_y)],
            },
        ]),
        CheckpointAction::DepAdd {
            dep,
            dep_default,
            expr,
        } => {
            let mut out = vec![Monomial {
                coeff: 1,
                factors: vec![cp_self(rule_default)],
            }];
            for m in bilinear_monomials(*expr, r_x, r_y) {
                out.push(prepend_factor(m, cp_dep(*dep, *dep_default)));
            }
            CheckpointEvalAction::Set(out)
        }
        CheckpointAction::DepAddWeighted {
            dep,
            dep_default,
            weight,
            expr,
        } => {
            let weight_src = weight_value(*weight, j);
            let mut out = vec![Monomial {
                coeff: 1,
                factors: vec![cp_self(rule_default)],
            }];
            for m in bilinear_monomials(*expr, r_x, r_y) {
                let mut factors = m.factors;
                factors.push(cp_dep(*dep, *dep_default));
                factors.push(weight_src.clone());
                out.push(Monomial {
                    coeff: m.coeff,
                    factors,
                });
            }
            CheckpointEvalAction::Set(out)
        }
        CheckpointAction::Hybrid { mul, add } => {
            let mut out = ScalarExpr::new();
            for m in bilinear_monomials(*mul, r_x, r_y) {
                out.push(prepend_factor(m, cp_self(rule_default)));
            }
            out.extend(bilinear_monomials(*add, r_x, r_y));
            CheckpointEvalAction::Set(out)
        }
        CheckpointAction::SignExtAccum { dep, final_j } => {
            // Interpreter uses F::zero() for missing self, not rule.default.
            let self_zero = ValueSource::Checkpoint {
                idx: self_idx,
                default: DefaultVal::Zero,
            };
            // val = self + 2^(j/2) * (1 - r_y)
            // if j == final_j: val *= checkpoints[dep]  (interpreter uses .unwrap())
            let mut out = vec![
                Monomial {
                    coeff: 1,
                    factors: vec![self_zero.clone()],
                },
                Monomial {
                    coeff: 1,
                    factors: vec![
                        ValueSource::Pow2((j / 2) as u32),
                        ValueSource::OneMinusChallenge(r_y),
                    ],
                },
            ];
            if j == *final_j {
                // Multiply the whole sum by dep checkpoint.
                let dep_factor = cp_dep(*dep, DefaultVal::Zero);
                for m in &mut out {
                    m.factors.push(dep_factor.clone());
                }
            }
            CheckpointEvalAction::Set(out)
        }
        CheckpointAction::Rev8WAdd { xlen } => {
            let self_zero = ValueSource::Checkpoint {
                idx: self_idx,
                default: DefaultVal::Zero,
            };
            let mut out = vec![Monomial {
                coeff: 1,
                factors: vec![self_zero],
            }];
            let ry_bit_index = 2 * xlen - 1 - j;
            if ry_bit_index < 64 {
                out.push(Monomial {
                    coeff: crate::module::rev8w(1u64 << ry_bit_index) as i128,
                    factors: vec![ValueSource::Challenge(r_y)],
                });
            }
            let rx_bit_index = ry_bit_index + 1;
            if rx_bit_index < 64 {
                out.push(Monomial {
                    coeff: crate::module::rev8w(1u64 << rx_bit_index) as i128,
                    factors: vec![ValueSource::Challenge(r_x)],
                });
            }
            CheckpointEvalAction::Set(out)
        }
        CheckpointAction::Pow2DoubleMul { xlen } => {
            // val = cp * (1 + (shift_x - 1) * r_x) * (1 + (shift_y - 1) * r_y)
            // Interpreter uses .unwrap(); we use Zero default (trusts invariant).
            let self_zero = ValueSource::Checkpoint {
                idx: self_idx,
                default: DefaultVal::Zero,
            };
            let exp_x = 2 * xlen - j;
            let shift_x: u64 = 1u64.wrapping_shl(1u32.wrapping_shl(exp_x as u32));
            let exp_y = 2 * xlen - j - 1;
            let shift_y: u64 = 1u64.wrapping_shl(1u32.wrapping_shl(exp_y as u32));
            let sx: i128 = (shift_x as i128) - 1;
            let sy: i128 = (shift_y as i128) - 1;
            // (cp) * (1 + sx*r_x) * (1 + sy*r_y)
            //   = cp + sx*cp*r_x + sy*cp*r_y + sx*sy*cp*r_x*r_y
            CheckpointEvalAction::Set(vec![
                Monomial {
                    coeff: 1,
                    factors: vec![self_zero.clone()],
                },
                Monomial {
                    coeff: sx,
                    factors: vec![self_zero.clone(), ValueSource::Challenge(r_x)],
                },
                Monomial {
                    coeff: sy,
                    factors: vec![self_zero.clone(), ValueSource::Challenge(r_y)],
                },
                Monomial {
                    coeff: sx.wrapping_mul(sy),
                    factors: vec![
                        self_zero,
                        ValueSource::Challenge(r_x),
                        ValueSource::Challenge(r_y),
                    ],
                },
            ])
        }
        CheckpointAction::Pow2Init { half_pow } => {
            let shift: u64 = 1u64 << *half_pow;
            CheckpointEvalAction::Set(vec![
                Monomial {
                    coeff: 1,
                    factors: vec![],
                },
                Monomial {
                    coeff: (shift as i128) - 1,
                    factors: vec![ValueSource::Challenge(r_y)],
                },
            ])
        }
        CheckpointAction::Set(expr) => {
            CheckpointEvalAction::Set(bilinear_monomials(*expr, r_x, r_y))
        }
        CheckpointAction::SetScaled { coeff, expr } => {
            let mut out = bilinear_monomials(*expr, r_x, r_y);
            for m in &mut out {
                m.coeff = m.coeff.wrapping_mul(*coeff);
            }
            CheckpointEvalAction::Set(out)
        }
        CheckpointAction::Const(val) => match val {
            DefaultVal::Zero => CheckpointEvalAction::Set(vec![]),
            DefaultVal::One => CheckpointEvalAction::Set(vec![Monomial {
                coeff: 1,
                factors: vec![],
            }]),
            DefaultVal::Custom(v) => CheckpointEvalAction::Set(vec![Monomial {
                coeff: *v,
                factors: vec![],
            }]),
        },
        CheckpointAction::Passthrough => {
            unreachable!("Passthrough is handled by lower_checkpoint_rule returning None")
        }
        CheckpointAction::Null => CheckpointEvalAction::Clear,
    }
}

fn weight_value(w: WeightFn, j: usize) -> ValueSource {
    let shift: u32 = match w {
        WeightFn::Positional {
            rotation,
            word_bits,
            j_offset,
        } => {
            let pos = (j - j_offset) / 2;
            let rotated = (pos + rotation as usize) % word_bits as usize;
            (word_bits as usize - 1 - rotated) as u32
        }
        WeightFn::LinearJ { base } => (base - j) as u32,
        WeightFn::LinearJMinusOne { base } => (base - j - 1) as u32,
        WeightFn::HalfJ => (j / 2) as u32,
    };
    ValueSource::Pow2(shift)
}

fn prepend_factor(m: Monomial, f: ValueSource) -> Monomial {
    let mut factors = Vec::with_capacity(m.factors.len() + 1);
    factors.push(f);
    factors.extend(m.factors);
    Monomial {
        coeff: m.coeff,
        factors,
    }
}

/// Expand a bilinear expression `expr(r_x, r_y)` into monomials over challenges
/// and their (1 - challenge) complements.
///
/// Using `OneMinusChallenge` keeps each resulting monomial to at most two factors.
fn bilinear_monomials(expr: BilinearExpr, r_x: ChallengeIdx, r_y: ChallengeIdx) -> ScalarExpr {
    use BilinearExpr::{
        AntiXY, AntiYX, EqBit, NorBit, OneMinusX, OneMinusY, OnePlusY, OrBit, Product, XorBit, X, Y,
    };
    match expr {
        Product => vec![Monomial {
            coeff: 1,
            factors: vec![ValueSource::Challenge(r_x), ValueSource::Challenge(r_y)],
        }],
        AntiXY => vec![Monomial {
            coeff: 1,
            factors: vec![
                ValueSource::OneMinusChallenge(r_x),
                ValueSource::Challenge(r_y),
            ],
        }],
        AntiYX => vec![Monomial {
            coeff: 1,
            factors: vec![
                ValueSource::Challenge(r_x),
                ValueSource::OneMinusChallenge(r_y),
            ],
        }],
        NorBit => vec![Monomial {
            coeff: 1,
            factors: vec![
                ValueSource::OneMinusChallenge(r_x),
                ValueSource::OneMinusChallenge(r_y),
            ],
        }],
        EqBit => vec![
            Monomial {
                coeff: 1,
                factors: vec![ValueSource::Challenge(r_x), ValueSource::Challenge(r_y)],
            },
            Monomial {
                coeff: 1,
                factors: vec![
                    ValueSource::OneMinusChallenge(r_x),
                    ValueSource::OneMinusChallenge(r_y),
                ],
            },
        ],
        XorBit => vec![
            Monomial {
                coeff: 1,
                factors: vec![
                    ValueSource::OneMinusChallenge(r_x),
                    ValueSource::Challenge(r_y),
                ],
            },
            Monomial {
                coeff: 1,
                factors: vec![
                    ValueSource::Challenge(r_x),
                    ValueSource::OneMinusChallenge(r_y),
                ],
            },
        ],
        OrBit => vec![
            Monomial {
                coeff: 1,
                factors: vec![ValueSource::Challenge(r_x)],
            },
            Monomial {
                coeff: 1,
                factors: vec![ValueSource::Challenge(r_y)],
            },
            Monomial {
                coeff: -1,
                factors: vec![ValueSource::Challenge(r_x), ValueSource::Challenge(r_y)],
            },
        ],
        OneMinusX => vec![Monomial {
            coeff: 1,
            factors: vec![ValueSource::OneMinusChallenge(r_x)],
        }],
        OneMinusY => vec![Monomial {
            coeff: 1,
            factors: vec![ValueSource::OneMinusChallenge(r_y)],
        }],
        OnePlusY => vec![
            Monomial {
                coeff: 1,
                factors: vec![],
            },
            Monomial {
                coeff: 1,
                factors: vec![ValueSource::Challenge(r_y)],
            },
        ],
        X => vec![Monomial {
            coeff: 1,
            factors: vec![ValueSource::Challenge(r_x)],
        }],
        Y => vec![Monomial {
            coeff: 1,
            factors: vec![ValueSource::Challenge(r_y)],
        }],
    }
}
