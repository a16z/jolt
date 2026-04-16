//! Prefix-suffix decomposition types for instruction lookup sumchecks.
//!
//! All evaluation logic is data-driven via `InstanceConfig` fields
//! (prefix_lowered, checkpoint_rules, combine_entries, suffix_ops,
//! suffix_at_empty). No protocol-specific evaluator trait needed.

use std::collections::HashMap;

use jolt_compiler::module::{CombineEntry, PrefixLoweredRound};
use jolt_compiler::prefix_mle_lowering::compute_mask_value;
use jolt_compiler::PolynomialId;
use jolt_field::Field;

pub use jolt_compute::LookupTraceData;

use crate::scalar_expr::eval_scalar_expr;

/// Per-phase buffer data produced by the suffix scatter handler.
pub struct PhaseBuffers<F: Field> {
    pub suffix_polys: Vec<Vec<Vec<F>>>,
    pub q_left: [Vec<F>; 2],
    pub q_right: [Vec<F>; 2],
    pub q_identity: [Vec<F>; 2],
    pub p_left: [Option<Vec<F>>; 2],
    pub p_right: [Option<Vec<F>>; 2],
    pub p_identity: [Option<Vec<F>>; 2],
}

/// Data-driven read-checking reduction via compiler-baked lowerings.
///
/// The compiler pre-lowers every `(round, rule, c-side)` into a `ScalarExpr`
/// plus mask-role bindings; the runtime only evaluates these expressions
/// against the current suffix polys and checkpoint scalars.
///
/// The c=2 prefix evaluations follow from multilinearity in c:
/// `prefix_c2 = 2·prefix_c1 − prefix_c0`. The combine matrix weights the
/// per-entry `(prefix × suffix)` contributions into the two sumcheck evaluations
/// returned.
pub fn compute_read_checking_from_lowered<F: Field>(
    lowered: &PrefixLoweredRound,
    combine_entries: &[CombineEntry],
    suffix_polys: &[Vec<Vec<F>>],
    checkpoints: &[Option<F>],
    r_x: Option<F>,
) -> [F; 2] {
    let current_len = suffix_polys
        .first()
        .and_then(|table_polys| table_polys.first())
        .map_or(0, |sp| sp.len());
    let half = current_len / 2;
    let b_len = lowered.b_len;

    let challenges: Vec<F> = vec![r_x.unwrap_or(F::zero())];
    let num_rules = lowered.c0.len();

    // Per-rule mask buffers for c=0 and c=1 (fresh allocator per rule means
    // ids in c0 and c1 are scoped to that rule — independent HashMaps).
    let mut mask_bufs_c0: Vec<HashMap<PolynomialId, Vec<F>>> = Vec::with_capacity(num_rules);
    let mut mask_bufs_c1: Vec<HashMap<PolynomialId, Vec<F>>> = Vec::with_capacity(num_rules);
    for rule_idx in 0..num_rules {
        let mut buf_c0: HashMap<PolynomialId, Vec<F>> =
            HashMap::with_capacity(lowered.masks_c0[rule_idx].len());
        for (pid, role) in &lowered.masks_c0[rule_idx] {
            let vals: Vec<F> = (0..half as u128)
                .map(|b| F::from_u128(compute_mask_value(*role, b, b_len)))
                .collect();
            let _ = buf_c0.insert(*pid, vals);
        }
        let mut buf_c1: HashMap<PolynomialId, Vec<F>> =
            HashMap::with_capacity(lowered.masks_c1[rule_idx].len());
        for (pid, role) in &lowered.masks_c1[rule_idx] {
            let vals: Vec<F> = (0..half as u128)
                .map(|b| F::from_u128(compute_mask_value(*role, b, b_len)))
                .collect();
            let _ = buf_c1.insert(*pid, vals);
        }
        mask_bufs_c0.push(buf_c0);
        mask_bufs_c1.push(buf_c1);
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
        let prefixes_c0: Vec<F> = (0..num_rules)
            .map(|i| {
                let buffers: HashMap<PolynomialId, &[F]> = mask_bufs_c0[i]
                    .iter()
                    .map(|(k, v)| (*k, v.as_slice()))
                    .collect();
                eval_scalar_expr(&lowered.c0[i], &challenges, checkpoints, b_val, &buffers)
            })
            .collect();
        let prefixes_c1: Vec<F> = (0..num_rules)
            .map(|i| {
                let buffers: HashMap<PolynomialId, &[F]> = mask_bufs_c1[i]
                    .iter()
                    .map(|(k, v)| (*k, v.as_slice()))
                    .collect();
                eval_scalar_expr(&lowered.c1[i], &challenges, checkpoints, b_val, &buffers)
            })
            .collect();
        let prefixes_c2: Vec<F> = (0..num_rules)
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
