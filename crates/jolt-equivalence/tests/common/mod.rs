//! Test-only interpretive reference for `CheckpointRule` evaluation.
//!
//! Lives here (not in jolt-zkvm) because runtime does not interpret
//! CheckpointActions — it evaluates pre-lowered `ScalarExpr`s via
//! `jolt_compiler::checkpoint_lowering::lower_checkpoint_rule` +
//! `jolt_zkvm::scalar_expr::eval_scalar_expr`. This module exists
//! purely to provide a ground-truth reference for parity tests.

use jolt_compiler::module::{
    rev8w, BilinearExpr, CheckpointAction, CheckpointRule, DefaultVal, RoundGuard, WeightFn,
};
use jolt_field::Field;

pub fn default_to_field<F: Field>(d: DefaultVal) -> F {
    match d {
        DefaultVal::Zero => F::zero(),
        DefaultVal::One => F::one(),
        DefaultVal::Custom(v) => {
            if v >= 0 {
                F::from_u128(v as u128)
            } else {
                F::zero() - F::from_u128((-v) as u128)
            }
        }
    }
}

fn bilinear_eval<F: Field>(expr: BilinearExpr, rx: F, ry: F) -> F {
    match expr {
        BilinearExpr::Product => rx * ry,
        BilinearExpr::AntiXY => (F::one() - rx) * ry,
        BilinearExpr::AntiYX => rx * (F::one() - ry),
        BilinearExpr::NorBit => (F::one() - rx) * (F::one() - ry),
        BilinearExpr::EqBit => rx * ry + (F::one() - rx) * (F::one() - ry),
        BilinearExpr::XorBit => (F::one() - rx) * ry + rx * (F::one() - ry),
        BilinearExpr::OrBit => rx + ry - rx * ry,
        BilinearExpr::OneMinusX => F::one() - rx,
        BilinearExpr::OneMinusY => F::one() - ry,
        BilinearExpr::OnePlusY => F::one() + ry,
        BilinearExpr::X => rx,
        BilinearExpr::Y => ry,
    }
}

fn weight_eval(w: WeightFn, j: usize) -> u128 {
    match w {
        WeightFn::Positional {
            rotation,
            word_bits,
            j_offset,
        } => {
            let pos = (j - j_offset) / 2;
            let rotated = (pos + rotation as usize) % word_bits as usize;
            let shift = word_bits as usize - 1 - rotated;
            1u128 << shift
        }
        WeightFn::LinearJ { base } => 1u128 << (base - j),
        WeightFn::LinearJMinusOne { base } => 1u128 << (base - j - 1),
        WeightFn::HalfJ => 1u128 << (j / 2),
    }
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

pub fn eval_checkpoint_rule<F: Field>(
    rule: &CheckpointRule,
    self_idx: usize,
    checkpoints: &[Option<F>],
    rx: F,
    ry: F,
    j: usize,
    suffix_len: usize,
) -> Option<F> {
    let action = rule
        .cases
        .iter()
        .find(|(guard, _)| guard_matches(*guard, j, suffix_len))
        .map_or(&rule.fallback, |(_, action)| action);
    apply_action(rule, action, self_idx, checkpoints, rx, ry, j)
}

fn apply_action<F: Field>(
    rule: &CheckpointRule,
    action: &CheckpointAction,
    self_idx: usize,
    checkpoints: &[Option<F>],
    rx: F,
    ry: F,
    j: usize,
) -> Option<F> {
    let cp = || checkpoints[self_idx].unwrap_or(default_to_field(rule.default));
    match action {
        CheckpointAction::Mul(expr) => Some(cp() * bilinear_eval(*expr, rx, ry)),
        CheckpointAction::AddWeighted { weight, expr } => {
            Some(cp() + F::from_u128(weight_eval(*weight, j)) * bilinear_eval(*expr, rx, ry))
        }
        CheckpointAction::AddTwoTerm { x_weight, y_weight } => Some(
            cp() + F::from_u128(weight_eval(*x_weight, j)) * rx
                + F::from_u128(weight_eval(*y_weight, j)) * ry,
        ),
        CheckpointAction::DepAdd {
            dep,
            dep_default,
            expr,
        } => {
            let dep_cp = checkpoints[*dep].unwrap_or(default_to_field(*dep_default));
            Some(cp() + dep_cp * bilinear_eval(*expr, rx, ry))
        }
        CheckpointAction::DepAddWeighted {
            dep,
            dep_default,
            weight,
            expr,
        } => {
            let dep_cp = checkpoints[*dep].unwrap_or(default_to_field(*dep_default));
            Some(
                cp() + dep_cp
                    * F::from_u128(weight_eval(*weight, j))
                    * bilinear_eval(*expr, rx, ry),
            )
        }
        CheckpointAction::Hybrid { mul, add } => {
            Some(cp() * bilinear_eval(*mul, rx, ry) + bilinear_eval(*add, rx, ry))
        }
        CheckpointAction::SignExtAccum { dep, final_j } => {
            let mut val = checkpoints[self_idx].unwrap_or(F::zero());
            val += F::from_u128(1u128 << (j / 2)) * (F::one() - ry);
            if j == *final_j {
                val *= checkpoints[*dep].unwrap();
            }
            Some(val)
        }
        CheckpointAction::Rev8WAdd { xlen } => {
            let mut val = checkpoints[self_idx].unwrap_or(F::zero());
            let ry_bit_index = 2 * xlen - 1 - j;
            if ry_bit_index < 64 {
                val += ry * F::from_u64(rev8w(1u64 << ry_bit_index));
            }
            let rx_bit_index = ry_bit_index + 1;
            if rx_bit_index < 64 {
                val += rx * F::from_u64(rev8w(1u64 << rx_bit_index));
            }
            Some(val)
        }
        CheckpointAction::Pow2DoubleMul { xlen } => {
            let mut val = checkpoints[self_idx].unwrap();
            let exp_x = 2 * xlen - j;
            let shift_x: u64 = 1u64.wrapping_shl(1u32.wrapping_shl(exp_x as u32));
            val *= F::one() + F::from_u64(shift_x - 1) * rx;
            let exp_y = 2 * xlen - j - 1;
            let shift_y: u64 = 1u64.wrapping_shl(1u32.wrapping_shl(exp_y as u32));
            val *= F::one() + F::from_u64(shift_y - 1) * ry;
            Some(val)
        }
        CheckpointAction::Pow2Init { half_pow } => {
            let shift: u64 = 1u64 << *half_pow;
            Some(F::one() + F::from_u64(shift - 1) * ry)
        }
        CheckpointAction::Set(expr) => Some(bilinear_eval(*expr, rx, ry)),
        CheckpointAction::SetScaled { coeff, expr } => {
            let c: F = if *coeff >= 0 {
                F::from_u128(*coeff as u128)
            } else {
                F::zero() - F::from_u128((-*coeff) as u128)
            };
            Some(c * bilinear_eval(*expr, rx, ry))
        }
        CheckpointAction::Const(val) => Some(default_to_field(*val)),
        CheckpointAction::Passthrough => checkpoints[self_idx],
        CheckpointAction::Null => None,
    }
}
