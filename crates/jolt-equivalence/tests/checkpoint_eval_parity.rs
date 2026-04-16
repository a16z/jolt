//! Parity test for the data-driven lowering of `CheckpointRule`s.
//!
//! For each `CheckpointAction` variant, we construct a small synthetic rule,
//! run both the interpretive `eval_checkpoint_rule` and the compiled
//! `lower_checkpoint_rule` → `eval_scalar_expr` pipeline over random
//! snapshots/challenges at several `(j, suffix_len)` pairs, and assert field
//! equality. A mismatch indicates the lowering drifted from the interpreter.
//!
//! The action-by-action coverage is intentional: this test is the contract for
//! what each lowering variant should produce. End-to-end parity (transcript
//! divergence) is covered by the muldiv test.

use std::collections::HashMap;

use ark_ff::UniformRand;
use jolt_compiler::checkpoint_lowering::lower_checkpoint_rule;
use jolt_compiler::module::{
    BilinearExpr, ChallengeIdx, CheckpointAction, CheckpointEvalAction, CheckpointRule, DefaultVal,
    RoundGuard, WeightFn,
};
use jolt_compiler::PolynomialId;
use jolt_field::Fr;
use jolt_zkvm::scalar_expr::eval_scalar_expr;

mod common;
use common::eval_checkpoint_rule;

type F = Fr;

const RX: ChallengeIdx = ChallengeIdx(0);
const RY: ChallengeIdx = ChallengeIdx(1);

fn run_both(
    rule: &CheckpointRule,
    self_idx: usize,
    snapshot: &[Option<F>],
    rx: F,
    ry: F,
    j: usize,
    suffix_len: usize,
) -> (Option<F>, Option<F>) {
    let old = eval_checkpoint_rule(rule, self_idx, snapshot, rx, ry, j, suffix_len);
    let lowered = lower_checkpoint_rule(rule, self_idx, RX, RY, j, suffix_len);
    let empty: HashMap<PolynomialId, &[F]> = HashMap::new();
    let new = match lowered {
        // Skipped (Passthrough): slot stays at its pre-batch value.
        None => snapshot[self_idx],
        Some(CheckpointEvalAction::Clear) => None,
        Some(CheckpointEvalAction::Set(e)) => {
            Some(eval_scalar_expr(&e, &[rx, ry], snapshot, 0, &empty))
        }
    };
    (old, new)
}

fn assert_parity(
    rule: &CheckpointRule,
    label: &str,
    // `old_is_none_means_skip` lets Passthrough-ish rules treat Old=None as equivalent.
    passthrough_semantics: bool,
) {
    assert_parity_inner(
        rule,
        label,
        passthrough_semantics,
        &[0, 1, 2, 3, 5, 10, 30, 60],
    );
}

fn assert_parity_restricted(rule: &CheckpointRule, label: &str, js: &[usize]) {
    assert_parity_inner(rule, label, false, js);
}

fn assert_parity_inner(
    rule: &CheckpointRule,
    label: &str,
    passthrough_semantics: bool,
    js: &[usize],
) {
    let mut rng = ark_std::test_rng();
    for _ in 0..8 {
        let rx = F::rand(&mut rng);
        let ry = F::rand(&mut rng);
        let snapshot: Vec<Option<F>> = (0..8)
            .map(|i| {
                if i == 7 {
                    None
                } else {
                    Some(F::rand(&mut rng))
                }
            })
            .collect();
        for &j in js {
            for suffix_len in [0usize, 1, 8, 32] {
                let (old, new) = run_both(rule, 0, &snapshot, rx, ry, j, suffix_len);
                let _ = passthrough_semantics;
                assert_eq!(
                    old, new,
                    "[{label}] j={j} suffix={suffix_len} rx={rx:?} ry={ry:?}"
                );
            }
        }
    }
}

fn single_rule(action: CheckpointAction, default: DefaultVal) -> CheckpointRule {
    CheckpointRule {
        default,
        cases: vec![],
        fallback: action,
    }
}

#[test]
fn parity_mul() {
    for expr in [
        BilinearExpr::Product,
        BilinearExpr::EqBit,
        BilinearExpr::XorBit,
        BilinearExpr::NorBit,
    ] {
        assert_parity(
            &single_rule(CheckpointAction::Mul(expr), DefaultVal::One),
            "Mul",
            false,
        );
    }
}

#[test]
fn parity_add_weighted() {
    for weight in [
        WeightFn::Positional {
            rotation: 0,
            word_bits: 64,
            j_offset: 0,
        },
        WeightFn::Positional {
            rotation: 16,
            word_bits: 32,
            j_offset: 0,
        },
        WeightFn::LinearJ { base: 63 },
        WeightFn::LinearJMinusOne { base: 64 },
        WeightFn::HalfJ,
    ] {
        for expr in [BilinearExpr::Product, BilinearExpr::OrBit] {
            assert_parity(
                &single_rule(
                    CheckpointAction::AddWeighted { weight, expr },
                    DefaultVal::Zero,
                ),
                "AddWeighted",
                false,
            );
        }
    }
}

#[test]
fn parity_add_two_term() {
    // LinearJMinusOne{base=64} requires j < 64; use base that leaves headroom.
    assert_parity_restricted(
        &single_rule(
            CheckpointAction::AddTwoTerm {
                x_weight: WeightFn::LinearJ { base: 63 },
                y_weight: WeightFn::LinearJMinusOne { base: 63 },
            },
            DefaultVal::Zero,
        ),
        "AddTwoTerm",
        &[0, 1, 5, 10, 30, 60, 62],
    );
}

#[test]
fn parity_dep_add() {
    assert_parity(
        &single_rule(
            CheckpointAction::DepAdd {
                dep: 3,
                dep_default: DefaultVal::One,
                expr: BilinearExpr::Product,
            },
            DefaultVal::Zero,
        ),
        "DepAdd",
        false,
    );
    // Dep slot is None in snapshot → uses dep_default.
    assert_parity(
        &single_rule(
            CheckpointAction::DepAdd {
                dep: 7,
                dep_default: DefaultVal::Custom(-5),
                expr: BilinearExpr::XorBit,
            },
            DefaultVal::Zero,
        ),
        "DepAdd(None)",
        false,
    );
}

#[test]
fn parity_dep_add_weighted() {
    assert_parity(
        &single_rule(
            CheckpointAction::DepAddWeighted {
                dep: 2,
                dep_default: DefaultVal::One,
                weight: WeightFn::HalfJ,
                expr: BilinearExpr::AntiXY,
            },
            DefaultVal::Zero,
        ),
        "DepAddWeighted",
        false,
    );
}

#[test]
fn parity_hybrid() {
    assert_parity(
        &single_rule(
            CheckpointAction::Hybrid {
                mul: BilinearExpr::Product,
                add: BilinearExpr::Y,
            },
            DefaultVal::Zero,
        ),
        "Hybrid",
        false,
    );
}

#[test]
fn parity_sign_ext_accum() {
    for final_j in [1usize, 5, 10] {
        assert_parity(
            &single_rule(
                CheckpointAction::SignExtAccum { dep: 2, final_j },
                DefaultVal::Zero,
            ),
            "SignExtAccum",
            false,
        );
    }
}

#[test]
fn parity_rev8w_add() {
    for xlen in [32usize, 64] {
        assert_parity(
            &single_rule(CheckpointAction::Rev8WAdd { xlen }, DefaultVal::Zero),
            "Rev8WAdd",
            false,
        );
    }
}

#[test]
fn parity_pow2_double_mul() {
    // Pow2DoubleMul uses .unwrap() — test only with snapshots where the
    // self-slot is Some (we stage that by using slot index 0 which is Some
    // in our fixture).
    assert_parity(
        &single_rule(
            CheckpointAction::Pow2DoubleMul { xlen: 32 },
            DefaultVal::Zero,
        ),
        "Pow2DoubleMul",
        false,
    );
}

#[test]
fn parity_pow2_init() {
    for half_pow in [1u32, 8, 16, 32] {
        assert_parity(
            &single_rule(CheckpointAction::Pow2Init { half_pow }, DefaultVal::One),
            "Pow2Init",
            false,
        );
    }
}

#[test]
fn parity_set_and_set_scaled() {
    assert_parity(
        &single_rule(
            CheckpointAction::Set(BilinearExpr::Product),
            DefaultVal::Zero,
        ),
        "Set",
        false,
    );
    assert_parity(
        &single_rule(
            CheckpointAction::SetScaled {
                coeff: -7,
                expr: BilinearExpr::EqBit,
            },
            DefaultVal::Zero,
        ),
        "SetScaled",
        false,
    );
}

#[test]
fn parity_const_and_null() {
    for d in [
        DefaultVal::Zero,
        DefaultVal::One,
        DefaultVal::Custom(42),
        DefaultVal::Custom(-3),
    ] {
        assert_parity(
            &single_rule(CheckpointAction::Const(d), DefaultVal::Zero),
            "Const",
            false,
        );
    }
    assert_parity(
        &single_rule(CheckpointAction::Null, DefaultVal::Zero),
        "Null",
        false,
    );
}

#[test]
fn parity_passthrough() {
    assert_parity(
        &single_rule(CheckpointAction::Passthrough, DefaultVal::One),
        "Passthrough",
        true,
    );
    assert_parity(
        &single_rule(CheckpointAction::Passthrough, DefaultVal::Zero),
        "Passthrough(Zero)",
        true,
    );
}

#[test]
fn parity_guarded_rule() {
    // Exercise guard selection: J < 4 → Mul(Product); J == 4 → Const(One);
    // else → AddWeighted.
    let rule = CheckpointRule {
        default: DefaultVal::One,
        cases: vec![
            (
                RoundGuard::JLt(4),
                CheckpointAction::Mul(BilinearExpr::Product),
            ),
            (RoundGuard::JEq(4), CheckpointAction::Const(DefaultVal::One)),
            (
                RoundGuard::SuffixLenNonZero,
                CheckpointAction::AddWeighted {
                    weight: WeightFn::LinearJ { base: 63 },
                    expr: BilinearExpr::OrBit,
                },
            ),
        ],
        fallback: CheckpointAction::Null,
    };
    assert_parity(&rule, "guarded", false);
}
