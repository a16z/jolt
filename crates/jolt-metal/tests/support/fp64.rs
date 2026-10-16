//! A model of the `jolt::Fp64` reductions, shared by the Fp64 and Ext2
//! suites: which branch of `fold2_canonicalize` a reduction takes, and
//! operands whose products reach the rare branches.

use super::field::{modulus, TestField};

const WORD: u128 = u64::MAX as u128;

/// Which branch `fold2_canonicalize(t, t2)` takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold2 {
    /// `t + C·t2 < p`: the sum is already canonical.
    Plain,
    /// `p ≤ t + C·t2 < 2^64`: one conditional add of `C`.
    Canonicalize,
    /// `t + C·t2 ≥ 2^64`: the wrap is corrected by adding `C`.
    Overflow,
}

pub const BRANCHES: [Fold2; 3] = [Fold2::Plain, Fold2::Canonicalize, Fold2::Overflow];

/// The fold-2 branch of `reduce_sum` over `products`, or of
/// `reduce_product` for one product. The products' sum, below `2^130`, is
/// `low + top·2^128`; the first fold of `low` gives `t` and a carry, and
/// `t2` is that carry plus `top·C`.
pub fn fold2_branch<F: TestField>(products: &[u128]) -> Fold2 {
    let (low, top) = products.iter().fold((0u128, 0u128), |(low, top), &x| {
        let (low, carry) = low.overflowing_add(x);
        (low, top + u128::from(carry))
    });
    let c = F::OFFSET;
    let first = (low & WORD) + c * (low >> 64);
    let v = (first & WORD) + c * ((first >> 64) + top * c);
    if v > WORD {
        Fold2::Overflow
    } else if v >= modulus::<F>() {
        Fold2::Canonicalize
    } else {
        Fold2::Plain
    }
}

/// `⌊((k + 1) · 2^64 − 1) / d⌋`, the largest `m` with `m · d` below
/// `(k + 1) · 2^64`.
fn window(k: u128, d: u128) -> u128 {
    ((k + 1) << 64).div_ceil(d) - 1
}

/// Operands whose product reaches the rare fold-2 branches.
///
/// With `a = 2^63` and `b = 2m`, the product is `m · 2^64`, so the first
/// fold gives `C·m`. Taking `m = window(k, C)` puts `C·m` in
/// `[(k + 1) 2^64 − C, (k + 1) 2^64)`: for `k = 0` that is `[p, 2^64)`
/// (canonicalize), and for `k ≥ 1` it is `t2 = k` with
/// `t ≥ 2^64 − C·k` (overflow). `b` is below `2^64` for every `k < C/2`,
/// so `mul_u64` reaches the branches with the same operands.
pub fn windows<F: TestField>() -> Vec<(u128, u128)> {
    let p = modulus::<F>();
    [0, 1, 2, 3]
        .into_iter()
        .map(|k| 2 * window(k, F::OFFSET))
        .filter(|&b| b < p)
        .flat_map(|b| [(1 << 63, b), (b, 1 << 63)])
        .collect()
}
