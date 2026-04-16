//! Runtime evaluation of compiled [`ScalarExpr`]s.
//!
//! Used by [`Op::CheckpointEvalBatch`] — the compiler lowers every checkpoint
//! update rule into a sum of monomials over challenges and snapshot reads; the
//! runtime just walks the list.
//!
//! The `index` + `buffers` parameters extend the evaluator to per-`b`
//! expressions (e.g. prefix-MLE materialization) where factors may read a
//! precomputed device buffer at the current evaluation point.

use std::collections::HashMap;

use jolt_compiler::module::{DefaultVal, Monomial, ValueSource};
use jolt_compiler::PolynomialId;
use jolt_field::Field;

/// Evaluate a scalar expression against current challenges, a checkpoint
/// snapshot, an evaluation index, and optional indexed buffers.
///
/// For simple checkpoint-only expressions pass `index = 0` and an empty
/// `buffers` map.
#[allow(clippy::implicit_hasher)]
pub fn eval_scalar_expr<F: Field>(
    expr: &[Monomial],
    challenges: &[F],
    checkpoints: &[Option<F>],
    index: usize,
    buffers: &HashMap<PolynomialId, &[F]>,
) -> F {
    let mut acc = F::zero();
    for m in expr {
        let mut term = coeff_to_field::<F>(m.coeff);
        for f in &m.factors {
            term *= eval_value_source::<F>(f, challenges, checkpoints, index, buffers);
        }
        acc += term;
    }
    acc
}

#[allow(clippy::implicit_hasher)]
pub fn eval_value_source<F: Field>(
    src: &ValueSource,
    challenges: &[F],
    checkpoints: &[Option<F>],
    index: usize,
    buffers: &HashMap<PolynomialId, &[F]>,
) -> F {
    match src {
        ValueSource::Pow2(k) => pow2_field::<F>(*k),
        ValueSource::Challenge(ci) => challenges[ci.0],
        ValueSource::OneMinusChallenge(ci) => F::one() - challenges[ci.0],
        ValueSource::Checkpoint { idx, default } => {
            checkpoints[*idx].unwrap_or(default_as_field::<F>(*default))
        }
        ValueSource::IndexedPoly(poly) => {
            let buf = buffers.get(poly).unwrap_or_else(|| {
                panic!("eval_value_source: IndexedPoly({poly:?}) missing from buffers map")
            });
            buf[index]
        }
        ValueSource::SelectByIndex { index_poly, values } => {
            let buf = buffers.get(index_poly).unwrap_or_else(|| {
                panic!("eval_value_source: SelectByIndex index_poly({index_poly:?}) missing")
            });
            // Indexing poly holds small integers as field elements (preprocessed).
            let k = field_to_usize::<F>(buf[index]);
            coeff_to_field::<F>(values[k])
        }
    }
}

pub fn pow2_field<F: Field>(k: u32) -> F {
    if k < 127 {
        F::from_u128(1u128 << k)
    } else {
        let mut result = F::from_u128(1u128 << 126);
        for _ in 126..k {
            result += result;
        }
        result
    }
}

pub fn coeff_to_field<F: Field>(c: i128) -> F {
    if c >= 0 {
        F::from_u128(c as u128)
    } else {
        F::zero() - F::from_u128((-c) as u128)
    }
}

pub fn default_as_field<F: Field>(d: DefaultVal) -> F {
    match d {
        DefaultVal::Zero => F::zero(),
        DefaultVal::One => F::one(),
        DefaultVal::Custom(v) => coeff_to_field::<F>(v),
    }
}

/// Small-integer reverse conversion: interpret a field element that was
/// loaded from a buffer of `u32`/`usize` indices.
fn field_to_usize<F: Field>(f: F) -> usize {
    f.to_u64()
        .expect("SelectByIndex: index poly value does not fit in u64") as usize
}
