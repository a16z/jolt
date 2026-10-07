//! The byte link: the packs, W histogram groups, and S6b claims that tie the
//! retained one-hot claims to the signed-byte trace `Q`.
//!
//! A pack's table index is its tuple of unsigned byte codes, MSB-first, then
//! the RAM activity bit. With `W_j(h) = Σ_{t: tuple_j(t) = h} eq(r, t)` over
//! the S6b cycle point `r`, every retained one-hot claim is a marginal of one
//! W: `v_c = 2^16 · W̃_j(k_c at c's byte, ½ elsewhere)` for the triples and
//! `v_c = 2^8 · W̃_R(k_c at c's byte, ½ at the other byte, 1)` for RAM.

use std::collections::BTreeMap;
use std::ops::Range;

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::Field;
use jolt_openings::{EvaluationClaim, OpeningsError, PrecommittedRole};

use super::super::geometry::claim_reductions::bytecode::MAX_COMMITTED_BYTECODE_CHUNK_COUNT;
use super::super::JoltCommittedPolynomial as Poly;
use super::geometry::LatticeGeometryError;
use super::strategy::{append_trace_column, append_usize};

/// Bits of one signed byte and of one one-hot address chunk.
const BYTE_BITS: usize = 8;

/// The link packs in canonical order: five instruction triples, the last
/// instruction column with both bytecode columns, then the RAM pair with the
/// activity bit. Flattened without the activity bit, they list the one-hot
/// columns in `Q` slot order.
pub const BYTE_LINK_PACKS: [[Poly; 3]; 7] = [
    [
        Poly::InstructionRa(0),
        Poly::InstructionRa(1),
        Poly::InstructionRa(2),
    ],
    [
        Poly::InstructionRa(3),
        Poly::InstructionRa(4),
        Poly::InstructionRa(5),
    ],
    [
        Poly::InstructionRa(6),
        Poly::InstructionRa(7),
        Poly::InstructionRa(8),
    ],
    [
        Poly::InstructionRa(9),
        Poly::InstructionRa(10),
        Poly::InstructionRa(11),
    ],
    [
        Poly::InstructionRa(12),
        Poly::InstructionRa(13),
        Poly::InstructionRa(14),
    ],
    [
        Poly::InstructionRa(15),
        Poly::BytecodeRa(0),
        Poly::BytecodeRa(1),
    ],
    [Poly::RamRa(0), Poly::RamRa(1), Poly::RamActivity],
];

/// Index of the RAM pack in [`BYTE_LINK_PACKS`].
pub const RAM_PACK: usize = 6;

/// Order of the first histogram role: past the program image, whose order is
/// at most `2 + MAX_COMMITTED_BYTECODE_CHUNK_COUNT` (`packing.rs`).
const HISTOGRAM_ROLE_ORDER: u64 = 3 + MAX_COMMITTED_BYTECODE_CHUNK_COUNT as u64;

/// One committed W histogram group, opened jointly with `Q` at stage 8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistogramGroup {
    /// The six triple histograms, one 24-variable polynomial per pack.
    Triples,
    /// The 17-variable RAM histogram.
    Ram,
}

impl HistogramGroup {
    pub const ALL: [Self; 2] = [Self::Triples, Self::Ram];

    pub const fn of_pack(pack: usize) -> Self {
        if pack == RAM_PACK {
            Self::Ram
        } else {
            Self::Triples
        }
    }

    /// The packs whose histograms the group commits, in group order.
    pub const fn packs(self) -> Range<usize> {
        match self {
            Self::Triples => 0..RAM_PACK,
            Self::Ram => RAM_PACK..RAM_PACK + 1,
        }
    }

    pub fn polynomials(self) -> impl Iterator<Item = Poly> {
        self.packs().map(Poly::LinkHistogram)
    }

    /// Variables of one histogram: three bytes, or two bytes and activity.
    pub const fn num_vars(self) -> usize {
        match self {
            Self::Triples => 3 * BYTE_BITS,
            Self::Ram => 2 * BYTE_BITS + 1,
        }
    }

    /// The group's role in the joint opening, after every advice and
    /// committed-program role.
    pub const fn role(self) -> PrecommittedRole {
        match self {
            Self::Triples => PrecommittedRole::new(
                HISTOGRAM_ROLE_ORDER,
                b"byte_link_triple_histograms",
                "byte-link-triple-histograms",
            ),
            Self::Ram => PrecommittedRole::new(
                HISTOGRAM_ROLE_ORDER + 1,
                b"byte_link_ram_histogram",
                "byte-link-ram-histogram",
            ),
        }
    }

    /// Digest binding the group's histograms and arity; the group commitment's
    /// layout metadata must equal it.
    pub fn layout_digest(self) -> Result<[u8; 32], OpeningsError> {
        let mut hasher = Blake2b::<U32>::new();
        hasher.update(b"jolt/akita/byte-link/v1/histogram-group");
        self.append_shape(&mut hasher)?;
        Ok(hasher.finalize().into())
    }

    fn append_shape(self, hasher: &mut Blake2b<U32>) -> Result<(), OpeningsError> {
        append_usize(hasher, self.num_vars());
        append_usize(hasher, self.packs().len());
        for polynomial in self.polynomials() {
            append_trace_column(hasher, polynomial)?;
        }
        Ok(())
    }
}

/// Digest of the pack map and histogram groups, bound into `Q`'s layout
/// digest.
pub(super) fn byte_link_catalog_digest() -> Result<[u8; 32], OpeningsError> {
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(b"jolt/akita/byte-link/v1/catalog");
    append_usize(&mut hasher, BYTE_LINK_PACKS.len());
    for pack in &BYTE_LINK_PACKS {
        for column in pack {
            append_trace_column(&mut hasher, *column)?;
        }
    }
    for group in HistogramGroup::ALL {
        group.append_shape(&mut hasher)?;
    }
    Ok(hasher.finalize().into())
}

/// The W̃ evaluation one retained one-hot claim fixes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistogramQuery<F> {
    /// Index into [`BYTE_LINK_PACKS`].
    pub pack: usize,
    /// Table point, MSB-first.
    pub point: Vec<F>,
    pub value: F,
}

/// Every one-hot column of the link as `(pack, byte position, column)`, in
/// `Q` slot order.
fn one_hot_slots() -> impl Iterator<Item = (usize, usize, Poly)> {
    BYTE_LINK_PACKS
        .iter()
        .enumerate()
        .flat_map(|(pack, columns)| {
            columns
                .iter()
                .enumerate()
                .filter(|(_, column)| **column != Poly::RamActivity)
                .map(move |(position, column)| (pack, position, *column))
        })
}

/// The S6b claims the byte link authenticates against `Q`: every retained
/// one-hot opening and the fused increment `F(r)`, all at one cycle point `r`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteLinkInputs<F> {
    cycle_point: Vec<F>,
    /// `(address chunk k_c, value)` per one-hot column, in `one_hot_slots` order.
    one_hot: Vec<(Vec<F>, F)>,
    fused_inc: F,
}

impl<F: Field> ByteLinkInputs<F> {
    /// Takes every one-hot column's claim from `claims`, each at
    /// `(address chunk ‖ r)` where `r` is the fused increment's point.
    ///
    /// Rejects a missing column and a claim off `r`: one W per pack
    /// authenticates claims at one cycle point only.
    pub fn new(
        claims: &BTreeMap<Poly, EvaluationClaim<F>>,
        fused_inc: &EvaluationClaim<F>,
    ) -> Result<Self, LatticeGeometryError> {
        let cycle_point = fused_inc.point.as_slice();
        let one_hot = one_hot_slots()
            .map(|(_, _, column)| {
                let claim = claims
                    .get(&column)
                    .ok_or(LatticeGeometryError::ByteLinkMissingClaim { column })?;
                match claim.point.as_slice().split_at_checked(BYTE_BITS) {
                    Some((address, cycle)) if cycle == cycle_point => {
                        Ok((address.to_vec(), claim.value))
                    }
                    _ => Err(LatticeGeometryError::ByteLinkPointMismatch { column }),
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            cycle_point: cycle_point.to_vec(),
            one_hot,
            fused_inc: fused_inc.value,
        })
    }

    /// The shared S6b cycle point `r`, MSB-first.
    pub fn cycle_point(&self) -> &[F] {
        &self.cycle_point
    }

    pub fn fused_inc(&self) -> F {
        self.fused_inc
    }

    /// Every one-hot claim's histogram query, in `Q` slot order.
    pub fn histogram_queries(&self) -> Vec<HistogramQuery<F>> {
        let half = F::two_inv();
        one_hot_slots()
            .zip(&self.one_hot)
            .map(|((pack, position, _), (address, value))| {
                let bytes = if pack == RAM_PACK { 2 } else { 3 };
                let mut point = Vec::with_capacity(HistogramGroup::of_pack(pack).num_vars());
                for byte in 0..bytes {
                    if byte == position {
                        point.extend_from_slice(address);
                    } else {
                        point.extend(std::iter::repeat_n(half, BYTE_BITS));
                    }
                }
                if pack == RAM_PACK {
                    point.push(F::one());
                }
                let value = (0..BYTE_BITS * (bytes - 1)).fold(*value, |value, _| value.half());
                HistogramQuery { pack, point, value }
            })
            .collect()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use crate::protocols::jolt::lattice::packing::{
        byte_trace_columns, precommitted_packing_plan, OneHotTraceShape, PrecommittedPackingShape,
    };
    use crate::protocols::jolt::TracePolynomialOrder;
    use jolt_field::{Fr, Ring};
    use jolt_poly::eq_index_msb;

    const LOG_T: usize = 3;

    fn shape() -> OneHotTraceShape {
        OneHotTraceShape {
            ra_layout: JoltRaPolynomialLayout::new(16, 2, 2).unwrap(),
            log_t: LOG_T,
            log_k_chunk: 8,
        }
    }

    fn one_hot_columns() -> Vec<Poly> {
        one_hot_slots().map(|(_, _, column)| column).collect()
    }

    fn point(seed: u64, len: usize) -> Vec<Fr> {
        (0..len as u64)
            .map(|i| Fr::from_u64(seed * 97 + i * 13 + 5))
            .collect()
    }

    /// Selected-row code of column `c` at cycle `t`, with zero rows at `t = 0`.
    fn code(c: usize, t: usize) -> usize {
        if t == 0 {
            0
        } else {
            (t * 37 + c * 11) % 256
        }
    }

    fn active(t: usize) -> bool {
        t % 3 != 1
    }

    #[test]
    fn packs_list_the_one_hot_slots_of_q_in_order() {
        let expected = byte_trace_columns(&shape())
            .unwrap()
            .into_iter()
            .filter(|column| {
                matches!(
                    column,
                    Poly::InstructionRa(_) | Poly::BytecodeRa(_) | Poly::RamRa(_)
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(one_hot_columns(), expected);
        assert_eq!(BYTE_LINK_PACKS[RAM_PACK][2], Poly::RamActivity);
    }

    /// One claim per one-hot column at `(address chunk ‖ r)`, valued by `value`.
    fn claims_at(
        r: &[Fr],
        value: impl Fn(usize, Poly, &[Fr]) -> Fr,
    ) -> BTreeMap<Poly, EvaluationClaim<Fr>> {
        one_hot_columns()
            .into_iter()
            .enumerate()
            .map(|(c, column)| {
                let address = point(10 + c as u64, BYTE_BITS);
                let value = value(c, column, &address);
                (
                    column,
                    EvaluationClaim::new([address, r.to_vec()].concat(), value),
                )
            })
            .collect()
    }

    #[test]
    fn histogram_queries_are_inclusive_marginals_of_the_tuple_histograms() {
        let r = point(1, LOG_T);
        let columns = one_hot_columns();
        let codes = |column: Poly| -> Vec<usize> {
            let c = columns.iter().position(|other| *other == column).unwrap();
            (0..1 << LOG_T)
                .map(|t| {
                    if matches!(column, Poly::RamRa(_)) && !active(t) {
                        0
                    } else {
                        code(c, t)
                    }
                })
                .collect()
        };
        let claims = claims_at(&r, |_, column, address| {
            let is_ram = matches!(column, Poly::RamRa(_));
            (0..1 << LOG_T)
                .filter(|t| !is_ram || active(*t))
                .map(|t| {
                    eq_index_msb(&r, t as u128) * eq_index_msb(address, codes(column)[t] as u128)
                })
                .fold(Fr::from_u64(0), |sum, term| sum + term)
        });
        let fused_inc = EvaluationClaim::new(r.clone(), Fr::from_u64(0));
        let inputs = ByteLinkInputs::new(&claims, &fused_inc).unwrap();

        let queries = inputs.histogram_queries();
        assert_eq!(queries.len(), columns.len());
        for query in queries {
            let pack = BYTE_LINK_PACKS[query.pack];
            let tuple = |t: usize| -> u128 {
                if query.pack == RAM_PACK {
                    ((codes(pack[0])[t] as u128) << 9)
                        | ((codes(pack[1])[t] as u128) << 1)
                        | u128::from(active(t))
                } else {
                    pack.iter()
                        .fold(0, |index, column| (index << 8) | codes(*column)[t] as u128)
                }
            };
            let histogram_at_point = (0..1 << LOG_T)
                .map(|t| eq_index_msb(&r, t as u128) * eq_index_msb(&query.point, tuple(t)))
                .fold(Fr::from_u64(0), |sum, term| sum + term);
            assert_eq!(
                query.point.len(),
                HistogramGroup::of_pack(query.pack).num_vars()
            );
            assert_eq!(query.value, histogram_at_point);
        }
    }

    #[test]
    fn inputs_reject_misrouted_claims() {
        let r = point(1, LOG_T);
        let fused_inc = EvaluationClaim::new(r.clone(), Fr::from_u64(0));
        let mut honest = claims_at(&r, |c, _, _| Fr::from_u64(c as u64));
        let unrelated = EvaluationClaim::new(point(3, 2), Fr::from_u64(1));
        let _ = honest.insert(Poly::BalancedIncDigit(0), unrelated);
        let inputs = ByteLinkInputs::new(&honest, &fused_inc).unwrap();
        assert_eq!(inputs.cycle_point(), r.as_slice());

        let mut missing = honest.clone();
        let _ = missing.remove(&Poly::RamRa(1));
        assert_eq!(
            ByteLinkInputs::new(&missing, &fused_inc),
            Err(LatticeGeometryError::ByteLinkMissingClaim {
                column: Poly::RamRa(1),
            })
        );
        for point in [point(2, BYTE_BITS + LOG_T), point(2, BYTE_BITS - 1)] {
            let mut off_point = honest.clone();
            let _ = off_point.insert(
                Poly::BytecodeRa(1),
                EvaluationClaim::new(point, Fr::from_u64(1)),
            );
            assert_eq!(
                ByteLinkInputs::new(&off_point, &fused_inc),
                Err(LatticeGeometryError::ByteLinkPointMismatch {
                    column: Poly::BytecodeRa(1),
                })
            );
        }
    }

    #[test]
    fn histogram_roles_follow_every_program_role() {
        let program = precommitted_packing_plan(&PrecommittedPackingShape {
            bytecode_chunks: MAX_COMMITTED_BYTECODE_CHUNK_COUNT,
            log_bytecode_rows: 0,
            trace_order: TracePolynomialOrder::CycleMajor,
            program_image_log_words: Some(1),
        })
        .unwrap();
        let last_program_order = program
            .objects()
            .map(|object| object.precommitted_role().order())
            .max()
            .unwrap();
        let [triples, ram] = HistogramGroup::ALL;
        assert!(last_program_order < triples.role().order());
        assert!(triples.role().order() < ram.role().order());
        assert_ne!(
            triples.layout_digest().unwrap(),
            ram.layout_digest().unwrap()
        );
    }
}
