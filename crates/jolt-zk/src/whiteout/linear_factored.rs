//! `N·x` for the factored vector `x` of [`super::weights`]: the verifier's
//! `A(r)`, where `x` is `eq(r, ·)` read through `e`'s layout.
//!
//! Every native element `E` of `x` is `S_E·Ê`, so a block of `N` takes one
//! column combination of its matrix and one ring product per row. Akita's
//! residual stacks one copy of the inner matrix `A_a` per live block, and
//! each block's column scales are a sum of windows of one table
//! `T = eq(r_hi, ·)`, one window per digit plane. `T` is a tensor product,
//! `T[K·2^m + y] = top[K]·low[y]`, so a window starting at
//! `base = K·2^m + c` scales column `l` by `low[(c + l) mod 2^m]` times
//! `top[K + (c + l) / 2^m]`: one shared column vector per residue `c`, times
//! a scalar per bucket `(c + l) / 2^m`. Blocks that share a matrix therefore
//! share its pass, once per residue rather than once per block. On Akita's
//! layouts every block has the same residue, and `A_a` is read once. The
//! value is the same field element as one pass per block; only the order of
//! the field sums changes.

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use jolt_algebra::{CyclotomicRing, JoltField};
use jolt_crypto::lattice::PreparedMatrix;
use rayon::prelude::*;

use super::linear::RingLinearMap;
use super::weights::eq_table;
use super::WhiteoutError;

/// A window of `T = eq(point, ·)` on a factored vector's scales: native
/// element `E` in `elements` takes `weight·T[start + (E − elements.start)]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScaleWindow<F> {
    pub elements: Range<usize>,
    pub start: usize,
    pub weight: F,
}

/// The scales of a factored vector over `s`, one per native element, as
/// values and as windows of `T = eq(point, ·)` (`point[0]` the least
/// significant bit): `dense[E]` is the sum of the windows holding `E`.
pub(crate) struct ColumnScales<'a, F> {
    pub dense: &'a [F],
    pub windows: &'a [ScaleWindow<F>],
    pub point: &'a [F],
}

/// One window of a block, seen from the block's column 0: column `l` takes
/// `weight·top[top + (residue + l) / 2^m]·low[(residue + l) mod 2^m]`.
struct Term<F> {
    residue: usize,
    top: usize,
    weight: F,
}

/// The windows of the block reading native elements `columns`, or `None`
/// unless every window that meets the block holds all of it.
fn block_terms<F: Copy>(
    windows: &[ScaleWindow<F>],
    columns: Range<usize>,
    low_bits: usize,
) -> Option<Vec<Term<F>>> {
    let mut terms = Vec::new();
    for window in windows {
        if window.elements.end <= columns.start || columns.end <= window.elements.start {
            continue;
        }
        if columns.start < window.elements.start || window.elements.end < columns.end {
            return None;
        }
        let base = window.start + (columns.start - window.elements.start);
        terms.push(Term {
            residue: base & ((1 << low_bits) - 1),
            top: base >> low_bits,
            weight: window.weight,
        });
    }
    (!terms.is_empty()).then_some(terms)
}

/// The buckets of residue `residue` over `width` columns: for bucket `q`,
/// its columns and the entries of `low` they read, in order.
fn buckets(
    residue: usize,
    width: usize,
    low_bits: usize,
) -> impl Iterator<Item = (Range<usize>, Range<usize>)> {
    let size = 1usize << low_bits;
    (0..(residue + width).div_ceil(size)).map(move |q| {
        let first = (q * size).max(residue);
        let end = ((q + 1) * size).min(residue + width);
        (
            first - residue..end - residue,
            first - q * size..end - q * size,
        )
    })
}

/// One read of a matrix: `(Σ_{l ∈ columns} scales[l − columns.start]·M[·][l])·lo`,
/// added with weight `w` to the rows of every block `(block, w)` in
/// `recipients`.
struct Pass<'m, F: JoltField, const D: usize> {
    matrix: &'m PreparedMatrix<F, D>,
    columns: Range<usize>,
    scales: Vec<F>,
    recipients: Vec<(usize, F)>,
}

/// The passes that evaluate every block of `map` on `scales`. Blocks that
/// share a matrix, whose windows hold them whole and take fewer residues
/// than there are such blocks, share one pass per (residue, bucket): its
/// scales are a run of `low`, and each block takes the bucket's `top`
/// entries through its windows. Every other block takes one pass of its own
/// over its dense scales, with weight one.
fn plan<'m, F: JoltField, const D: usize>(
    map: &'m RingLinearMap<F, D>,
    scales: &ColumnScales<'_, F>,
) -> Result<Vec<Pass<'m, F, D>>, WhiteoutError> {
    let blocks = map.blocks();
    let mut groups: Vec<(&Arc<PreparedMatrix<F, D>>, Vec<usize>)> = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        match groups
            .iter_mut()
            .find(|(matrix, _)| Arc::ptr_eq(matrix, &block.matrix))
        {
            Some((_, members)) => members.push(index),
            None => groups.push((&block.matrix, vec![index])),
        }
    }
    let mut passes = Vec::new();
    for (matrix, members) in groups {
        let width = matrix.shape().width;
        let low_bits =
            (width.next_power_of_two().trailing_zeros() as usize).min(scales.point.len());
        let mut own = Vec::new();
        let mut shared: Vec<(usize, Vec<Term<F>>)> = Vec::new();
        for index in members {
            let block = blocks.get(index).ok_or(WhiteoutError::InvariantViolation(
                "a group names a block outside N",
            ))?;
            let columns = block.input.start / D..block.input.end / D;
            match block_terms(scales.windows, columns, low_bits) {
                Some(terms) => shared.push((index, terms)),
                None => own.push(index),
            }
        }
        let residues: BTreeSet<usize> = shared
            .iter()
            .flat_map(|(_, terms)| terms.iter().map(|term| term.residue))
            .collect();
        if residues.len() < shared.len() {
            let (low_point, top_point) = scales.point.split_at(low_bits);
            let (low, top) = (eq_table(low_point), eq_table(top_point));
            for &residue in &residues {
                for (q, (columns, entries)) in buckets(residue, width, low_bits).enumerate() {
                    let recipients = shared
                        .iter()
                        .filter(|(_, terms)| terms.iter().any(|term| term.residue == residue))
                        .map(|(index, terms)| {
                            let weight = terms.iter().filter(|term| term.residue == residue).fold(
                                F::zero(),
                                |sum, term| {
                                    sum + term.weight
                                        * top.get(term.top + q).copied().unwrap_or_default()
                                },
                            );
                            (*index, weight)
                        })
                        .collect();
                    let scales = low
                        .get(entries)
                        .ok_or(WhiteoutError::InvariantViolation(
                            "a bucket reads outside eq's low table",
                        ))?
                        .to_vec();
                    passes.push(Pass {
                        matrix,
                        columns,
                        scales,
                        recipients,
                    });
                }
            }
        } else {
            own.extend(shared.into_iter().map(|(index, _)| index));
        }
        for index in own {
            let block = blocks.get(index).ok_or(WhiteoutError::InvariantViolation(
                "a group names a block outside N",
            ))?;
            let dense = scales
                .dense
                .get(block.input.start / D..block.input.end / D)
                .ok_or(WhiteoutError::InvalidRelation(
                    "linear block reads outside s",
                ))?;
            passes.push(Pass {
                matrix,
                columns: 0..width,
                scales: dense.to_vec(),
                recipients: vec![(index, F::one())],
            });
        }
    }
    Ok(passes)
}

impl<F: JoltField, const D: usize> RingLinearMap<F, D> {
    /// `N·x` for `x` laid out like `s` whose native element `E` is
    /// `scales.dense[E]·lo`: the passes of [`plan`], each a column
    /// combination and one ring product per row, with no product per matrix
    /// entry (the module documentation). The gadget term takes one `lo`
    /// multiple per row: `Σ_p S_p·b^p·lo`.
    pub fn apply_factored(
        &self,
        scales: &ColumnScales<'_, F>,
        lo: &CyclotomicRing<F, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, WhiteoutError> {
        let passes = plan(self, scales)?;
        let products: Vec<Vec<CyclotomicRing<F, D>>> = passes
            .par_iter()
            .map(|pass| -> Result<_, WhiteoutError> {
                let label = pass
                    .recipients
                    .first()
                    .map_or("unknown", |&(index, _)| self.profile_label(index));
                let _span = tracing::info_span!("whiteout_nz_product", block = label).entered();
                Ok(pass
                    .matrix
                    .combine_column_range(pass.columns.clone(), &pass.scales)?
                    .into_iter()
                    .map(|row| row * *lo)
                    .collect())
            })
            .collect::<Result<_, _>>()?;
        let mut blocks: Vec<Vec<CyclotomicRing<F, D>>> = self
            .blocks()
            .iter()
            .map(|block| vec![CyclotomicRing::zero(); block.matrix.shape().rows])
            .collect();
        for (pass, product) in passes.iter().zip(&products) {
            for &(index, weight) in &pass.recipients {
                let rows = blocks
                    .get_mut(index)
                    .ok_or(WhiteoutError::InvariantViolation(
                        "a pass names a block outside N",
                    ))?;
                for (dst, ring) in rows.iter_mut().zip(product) {
                    ring.scale_accumulate_into(dst, weight);
                }
            }
        }
        for (block, rows) in self.blocks().iter().zip(&mut blocks) {
            let Some(gadget) = &block.gadget else {
                continue;
            };
            // Each plane is one native element, `S_p·lo`, so row `i` loses
            // `(Σ_p S_p·b^p)·lo`.
            let mut planes = vec![F::zero(); rows.len()];
            for (row, power, range) in gadget.planes::<F>(D) {
                let scale = scales
                    .dense
                    .get(range.start / D)
                    .ok_or(WhiteoutError::InvalidRelation("gadget reads outside s"))?;
                *planes.get_mut(row).ok_or(WhiteoutError::InvalidRelation(
                    "gadget rows differ from M's",
                ))? += *scale * power;
            }
            for (row, sum) in rows.iter_mut().zip(&planes) {
                *row -= lo.scale(sum);
            }
        }
        Ok(blocks.into_iter().flatten().collect())
    }
}

#[cfg(all(test, feature = "whiteout-prover"))]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests should fail loudly"
)]
mod tests {
    use std::sync::Arc;

    use jolt_algebra::{CyclotomicRing, Field, One, Ring, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::super::fixture::{prepared, F};
    use super::super::linear::{BlockTarget, GadgetTerm, LinearBlock, RingLinearMap};
    use super::super::weights::eq_table;
    use super::{plan, ColumnScales, ScaleWindow};

    const D: usize = 64;
    const ROWS: usize = 2;
    const LEVELS: usize = 3;

    /// Scales written out from `windows` over `T = eq(point, ·)`, and
    /// random scales on the elements no window holds (the gadget digits).
    fn written(
        windows: &[ScaleWindow<F>],
        point: &[F],
        len: usize,
        rng: &mut ChaCha20Rng,
    ) -> Vec<F> {
        let table = eq_table(point);
        let mut dense: Vec<F> = (0..len).map(|_| F::random(&mut *rng)).collect();
        let held = |e: usize| windows.iter().any(|w| w.elements.contains(&e));
        for (e, value) in dense.iter_mut().enumerate() {
            if held(e) {
                *value = F::zero();
            }
        }
        for window in windows {
            for e in window.elements.clone() {
                let at = window.start + e - window.elements.start;
                dense[e] += window.weight * table.get(at).copied().unwrap_or_default();
            }
        }
        dense
    }

    /// `⟨ω, N·x⟩` for random output weights `ω`, two ways: through
    /// `apply_factored`, and through `N`'s adjoint (`matrix_adjoint`, entry
    /// by entry, with the gadget) against `x` written out,
    /// `x[p] = dense[p / D]·lo[p mod D]`.
    fn assert_dual(
        map: &RingLinearMap<F, D>,
        scales: &ColumnScales<'_, F>,
        lo: &CyclotomicRing<F, D>,
        rng: &mut ChaCha20Rng,
    ) {
        let image = map.apply_factored(scales, lo).unwrap();
        let omega: Vec<F> = (0..map.num_outputs() * D)
            .map(|_| F::random(&mut *rng))
            .collect();
        let forward = image
            .iter()
            .flat_map(|ring| ring.coefficients().iter().copied())
            .zip(&omega)
            .fold(F::zero(), |sum, (value, weight)| sum + value * *weight);
        let mut weights = vec![F::zero(); scales.dense.len() * D];
        map.adjoint(&omega, &mut weights).unwrap();
        let backward = weights
            .iter()
            .enumerate()
            .fold(F::zero(), |sum, (p, weight)| {
                sum + *weight * scales.dense[p / D] * lo.coefficients()[p % D]
            });
        assert_eq!(forward, backward);
    }

    /// A residual-shaped `N` over one shared `A` of width `width`, plus an
    /// unshared outer block: block `b` reads elements `[b·width, (b+1)·width)`
    /// with `planes` windows starting at `starts[b] + p·width`, then the
    /// gadget digits of every block, then the outer block's columns. Block
    /// `split` is cut into two runs, so its windows do not hold it whole.
    /// The plan reads `reads` columns of `A`; without windows, every block
    /// reads all of `A`.
    fn check(
        width: usize,
        planes: usize,
        starts: &[usize],
        split: Option<usize>,
        reads: usize,
        seed: u64,
    ) {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let blocks = starts.len();
        let a = prepared::<D>(ROWS, width, 3);
        let gadget_len = ROWS * LEVELS;
        let gadget_start = blocks * width;
        let outer_start = gadget_start + blocks * gadget_len;
        let outer_width = 5;
        let b = prepared::<D>(1, outer_width, 4);
        let mut linear: Vec<LinearBlock<F, D>> = (0..blocks)
            .map(|block| LinearBlock {
                matrix: Arc::clone(&a),
                input: block * width * D..(block + 1) * width * D,
                gadget: Some(GadgetTerm {
                    input: (gadget_start + block * gadget_len) * D
                        ..(gadget_start + (block + 1) * gadget_len) * D,
                    log_basis: 4,
                    levels: LEVELS,
                }),
                target: BlockTarget::Public(vec![CyclotomicRing::zero(); ROWS]),
            })
            .collect();
        linear.push(LinearBlock {
            matrix: b,
            input: outer_start * D..(outer_start + outer_width) * D,
            gadget: None,
            target: BlockTarget::Public(vec![CyclotomicRing::zero()]),
        });
        let map = RingLinearMap::new(linear).unwrap();

        let radix = F::from_u64(16);
        let mut windows = Vec::new();
        for (block, &start) in starts.iter().enumerate() {
            let first = block * width;
            let runs = if split == Some(block) {
                vec![(first, width / 3), (first + width / 3, width - width / 3)]
            } else {
                vec![(first, width)]
            };
            let mut at = start;
            for (run, len) in runs {
                let mut power = F::one();
                for _ in 0..planes {
                    windows.push(ScaleWindow {
                        elements: run..run + len,
                        start: at,
                        weight: power,
                    });
                    at += len;
                    power *= radix;
                }
            }
        }
        windows.push(ScaleWindow {
            elements: outer_start..outer_start + outer_width,
            start: 3,
            weight: F::one(),
        });
        let point: Vec<F> = (0..9).map(|_| F::random(&mut rng)).collect();
        let len = outer_start + outer_width;
        let dense = written(&windows, &point, len, &mut rng);
        let lo = CyclotomicRing::from_coefficients(std::array::from_fn(|_| F::random(&mut rng)));

        let factored = ColumnScales {
            dense: &dense,
            windows: &windows,
            point: &point,
        };
        let unfactored = ColumnScales {
            windows: &[],
            ..factored
        };
        let reads_of_a = |scales: &ColumnScales<'_, F>| -> usize {
            plan(&map, scales)
                .unwrap()
                .iter()
                .filter(|pass| std::ptr::eq(pass.matrix, Arc::as_ptr(&a)))
                .map(|pass| pass.columns.len())
                .sum()
        };
        assert_eq!(reads_of_a(&factored), reads);
        assert_eq!(reads_of_a(&unfactored), blocks * width);
        assert_dual(&map, &factored, &lo, &mut rng);
        assert_dual(&map, &unfactored, &lo, &mut rng);
    }

    /// Akita's layout: blocks back to back, `planes·width` digits each, so
    /// every window starts on a multiple of `width` and one residue serves
    /// all blocks, which read `A` once; the last block's windows run past
    /// `T`'s end.
    #[test]
    fn aligned_blocks_read_the_matrix_once() {
        let starts: Vec<usize> = (0..5).map(|b| b * 3 * 16 + 288).collect();
        check(16, 3, &starts, None, 16, 1);
    }

    /// Unaligned starts: two residues across five blocks, each straddling a
    /// bucket boundary, and a block cut into two runs, which takes its own
    /// pass; then a width that is not a power of two, two residues per
    /// block.
    #[test]
    fn unaligned_and_split_blocks_match_the_adjoint() {
        check(16, 3, &[5, 53, 101, 7, 55], Some(1), 3 * 16, 2);
        check(12, 2, &[0, 32, 64, 96, 3, 35], Some(5), 5 * 12, 3);
    }
}
