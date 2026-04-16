use std::collections::HashMap;

use jolt_compiler::module::{
    rev8w, BilinearExpr, ChallengeIdx, CheckpointAction, CheckpointRule, CombineEntry, DefaultVal,
    PrefixMleRule, RoundGuard, WeightFn,
};
use jolt_compiler::prefix_mle_lowering::{
    compute_mask_value, lower_prefix_mle, mask_roles_for, LoweringCtx,
};
use jolt_compiler::PolynomialId;
use jolt_field::Field;

use crate::scalar_expr::eval_scalar_expr;

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

/// Evaluate a checkpoint update rule, producing the new checkpoint value.
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

/// Data-driven compute_read_checking via compiler lowerings.
///
/// Replaces `eval_prefix_mle` with `lower_prefix_mle` + `eval_scalar_expr`:
/// each rule is lowered to a ScalarExpr at c=0 and c=1, mask buffers are
/// precomputed, and prefix values are evaluated as linear expressions.
/// The c=2 side uses multilinearity: prefix_c2 = 2·prefix_c1 − prefix_c0.
#[allow(clippy::too_many_arguments)]
pub fn compute_read_checking_from_lowered<F: Field>(
    prefix_rules: &[PrefixMleRule],
    combine_entries: &[CombineEntry],
    round: usize,
    suffix_polys: &[Vec<Vec<F>>],
    checkpoints: &[Option<F>],
    r_x: Option<F>,
    total_bits: usize,
) -> [F; 2] {
    let current_len = suffix_polys
        .first()
        .and_then(|table_polys| table_polys.first())
        .map_or(0, |sp| sp.len());
    let half = current_len / 2;
    let log_len = current_len.trailing_zeros() as usize;
    let b_len = log_len - 1;

    let rx_idx_used = r_x.map(|_| ChallengeIdx(0));
    let challenges: Vec<F> = vec![r_x.unwrap_or(F::zero())];

    let mut lowered: Vec<[Vec<jolt_compiler::module::Monomial>; 2]> =
        Vec::with_capacity(prefix_rules.len());
    let mut mask_bufs: Vec<[HashMap<PolynomialId, Vec<F>>; 2]> =
        Vec::with_capacity(prefix_rules.len());
    for rule in prefix_rules {
        let mut exprs: [Vec<jolt_compiler::module::Monomial>; 2] = Default::default();
        let mut bufs: [HashMap<PolynomialId, Vec<F>>; 2] = Default::default();
        for c_side in 0..2u32 {
            let ctx = LoweringCtx {
                j: round,
                b_len,
                total_bits,
                r_x: rx_idx_used,
                c: c_side,
            };
            let roles = mask_roles_for(rule, ctx);
            let expr = lower_prefix_mle(rule, ctx, |role| {
                roles.iter().find(|(r, _)| *r == role).unwrap().1
            });
            let mut buf_map: HashMap<PolynomialId, Vec<F>> = HashMap::with_capacity(roles.len());
            for (role, pid) in &roles {
                let vals: Vec<F> = (0..half as u128)
                    .map(|b| F::from_u128(compute_mask_value(*role, b, b_len)))
                    .collect();
                let _ = buf_map.insert(*pid, vals);
            }
            exprs[c_side as usize] = expr;
            bufs[c_side as usize] = buf_map;
        }
        lowered.push(exprs);
        mask_bufs.push(bufs);
    }

    let coeffs: Vec<F> = combine_entries
        .iter()
        .map(|e| {
            if e.coefficient >= 0 {
                F::from_u128(e.coefficient as u128)
            } else {
                F::zero() - F::from_u128((-e.coefficient) as u128)
            }
        })
        .collect();

    let mut eval_0 = F::zero();
    let mut eval_2_left = F::zero();
    let mut eval_2_right = F::zero();

    for b_val in 0..half {
        let prefixes_c0: Vec<F> = (0..prefix_rules.len())
            .map(|i| {
                let buffers: HashMap<PolynomialId, &[F]> = mask_bufs[i][0]
                    .iter()
                    .map(|(k, v)| (*k, v.as_slice()))
                    .collect();
                eval_scalar_expr(&lowered[i][0], &challenges, checkpoints, b_val, &buffers)
            })
            .collect();
        let prefixes_c1: Vec<F> = (0..prefix_rules.len())
            .map(|i| {
                let buffers: HashMap<PolynomialId, &[F]> = mask_bufs[i][1]
                    .iter()
                    .map(|(k, v)| (*k, v.as_slice()))
                    .collect();
                eval_scalar_expr(&lowered[i][1], &challenges, checkpoints, b_val, &buffers)
            })
            .collect();
        let prefixes_c2: Vec<F> = (0..prefix_rules.len())
            .map(|i| prefixes_c1[i] + prefixes_c1[i] - prefixes_c0[i])
            .collect();

        for (i, entry) in combine_entries.iter().enumerate() {
            let p_c0 = entry.prefix_idx.map_or(F::one(), |p| prefixes_c0[p]);
            let p_c2 = entry.prefix_idx.map_or(F::one(), |p| prefixes_c2[p]);
            let s_left = suffix_polys[entry.table_idx][entry.suffix_local_idx][b_val];
            let s_right = suffix_polys[entry.table_idx][entry.suffix_local_idx][b_val + half];
            eval_0 += coeffs[i] * p_c0 * s_left;
            eval_2_left += coeffs[i] * p_c2 * s_left;
            eval_2_right += coeffs[i] * p_c2 * s_right;
        }
    }

    [eval_0, eval_2_right + eval_2_right - eval_2_left]
}
