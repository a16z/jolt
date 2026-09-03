//! Column packing for HyperKZG: `k` columns of `rows` entries share one
//! polynomial of `rows·k` entries with the column index in the high
//! variables (`P_g[j·rows + row] = column_{g·k+j}[row]`), so a table of `c`
//! columns costs `⌈c/k⌉` commitments and one opening of size `rows·k`.

use jolt_crypto::{HomomorphicCommitment, PairingGroup};
use jolt_field::{Fr, Zero};
use jolt_hyperkzg::{HyperKZGCommitment, HyperKZGProverSetup};
use jolt_poly::EqPolynomial;
use rayon::prelude::*;

pub fn groups(columns: usize, k: usize) -> usize {
    columns.div_ceil(k)
}

/// `Σ_g weights[g]·P_g` as a dense polynomial of `rows·k` entries; `column`
/// returns entry `(column index, row)` and is not called for padding columns.
pub fn combine<C>(rows: usize, k: usize, columns: usize, weights: &[Fr], column: C) -> Vec<Fr>
where
    C: Fn(usize, usize) -> Fr + Sync,
{
    (0..rows * k)
        .into_par_iter()
        .map(|index| {
            let (j, row) = (index / rows, index % rows);
            weights
                .iter()
                .enumerate()
                .filter(|(g, _)| g * k + j < columns)
                .fold(Fr::zero(), |acc, (g, w)| acc + *w * column(g * k + j, row))
        })
        .collect()
}

/// Verifier side of [`combine`] at the point `(s_lo, r)`: `Σ_g weights[g]·Σ_j eq(s_lo, j)·claims[g·k + j]`.
pub fn combined_claim(claims: &[Fr], k: usize, weights: &[Fr], s_lo: &[Fr]) -> Fr {
    let eq_lo = EqPolynomial::<Fr>::evals(s_lo, None);
    assert_eq!(eq_lo.len(), k);
    weights.iter().enumerate().fold(Fr::zero(), |acc, (g, w)| {
        let inner = eq_lo
            .iter()
            .enumerate()
            .filter_map(|(j, e)| claims.get(g * k + j).map(|c| *e * *c))
            .fold(Fr::zero(), |a, v| a + v);
        acc + *w * inner
    })
}

/// HyperKZG opening point of a packed polynomial: the column variables
/// (`s_lo`, big-endian) ahead of the row variables (`row_point`, big-endian).
pub fn point(s_lo: &[Fr], row_point: &[Fr]) -> Vec<Fr> {
    s_lo.iter().chain(row_point).copied().collect()
}

/// `2^k` bases per group at `rows·k` entries; the polynomial length must be a power of two.
pub fn log2_exact(k: usize) -> usize {
    assert!(k.is_power_of_two(), "pack factor must be a power of two");
    k.trailing_zeros() as usize
}

/// SRS powers of slot `j = column mod k`: `[j·rows, (j+1)·rows)`.
pub fn slot_bases<P: PairingGroup>(
    setup: &HyperKZGProverSetup<P>,
    rows: usize,
    k: usize,
    column: usize,
) -> &[P::G1Affine] {
    let j = column % k;
    &setup.g1_powers()[j * rows..(j + 1) * rows]
}

/// Sums the per-column commitments of each group of `k` consecutive columns.
pub fn group_commitments<P: PairingGroup>(
    columns: &[HyperKZGCommitment<P>],
    k: usize,
) -> Vec<HyperKZGCommitment<P>> {
    columns
        .chunks(k)
        .map(|group| {
            group.iter().skip(1).fold(group[0], |acc, c| {
                <HyperKZGCommitment<P> as HomomorphicCommitment<Fr>>::add(&acc, c)
            })
        })
        .collect()
}
