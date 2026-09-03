//! Copy identities of the fixed wiring (no extra rounds): for every copied
//! column `C_i` — the operand columns `X_s = κ·Z_ξ(src)`, the `Y_s` that are
//! not looked up, and the fingerprint columns `f_±` — with public source map
//! `src_i(x)` and public weight `w_i(x)`,
//! `Σ_x eq(τ,x)·C_i(x) = Σ_v K_i(τ,v)·Z_ξ(v)`, `K_i(τ,v) = Σ_{x: src_i(x)=v} eq(τ,x)·w_i(x)`.
//! The row member batches them with the copy weights `β_i`: the prover
//! table is `B(v) = Σ_i β_i·K_i(τ,v)`, the verifier evaluates
//! `Σ_i β_i·K_i(τ, r)` from the layout's kernels ([`super::verifier`]).

use jolt_field::{Fr, Ring, Zero};
use jolt_hyperkzg::VerifierObserver;

use super::layout::{Factor, Side};
use super::program::Program;
use super::relation::{RowRelation, FP_SLOTS_G1, FP_SLOTS_G2, FP_SLOTS_GT, SLOTS};
use super::template::ElemWiring;
use super::verifier::Evaluator;

/// What a row looks up: nothing, or a GT / G1 / G2 table entry whose first
/// `fp_slots` operand slots are fingerprinted instead of copied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReadKind {
    #[default]
    None,
    Gt,
    G1,
    G2,
}

impl ReadKind {
    pub const fn fp_slots(self) -> usize {
        match self {
            Self::None => 0,
            Self::Gt => FP_SLOTS_GT,
            Self::G1 => FP_SLOTS_G1,
            Self::G2 => FP_SLOTS_G2,
        }
    }
}

/// One fingerprint contribution on a table row: coordinate `src` enters slot
/// `slot` of the reading pattern; `conjugated` coordinates are negated in
/// `f_neg`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableRead {
    pub row: u32,
    pub slot: u8,
    pub src: u32,
    pub conjugated: bool,
}

/// The kernels of the fingerprint columns over one table region: table rows
/// read their own cell through the reading template's `Y` maps of the
/// fingerprinted slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FingerprintGroup {
    pub cell: Vec<Factor>,
    /// `(slot, map)` for `f_pos`; `f_neg` uses `conj_maps`.
    pub maps: Vec<(u8, Factor)>,
    pub conj_maps: Vec<(u8, Factor)>,
}

/// Prover table `B(v) = Σ_i β_i·K_i(τ, v)` over the source rows.
pub fn copy_kernel_table(
    program: &Program,
    kinds: &[ReadKind],
    reads: &[TableRead],
    eq_tau: &[Fr],
    relation: &RowRelation,
) -> Vec<Fr> {
    let mut table = vec![Fr::zero(); eq_tau.len()];
    for (row, spec) in program.rows.iter().enumerate() {
        let skip = kinds[row].fp_slots();
        for (s, slot) in spec.slots.iter().enumerate() {
            let x_weight = relation.copy_weight(s) * Fr::from_i64(i64::from(slot.kappa));
            table[slot.x as usize] += eq_tau[row] * x_weight;
            if s >= skip {
                table[slot.y as usize] += eq_tau[row] * relation.copy_weight(SLOTS + s);
            }
        }
    }
    let pos = relation.copy_weight(2 * SLOTS);
    let neg = relation.copy_weight(2 * SLOTS + 1);
    for read in reads {
        let eq = eq_tau[read.row as usize];
        let sign = if read.conjugated {
            -Fr::from_u64(1)
        } else {
            Fr::from_u64(1)
        };
        table[read.src as usize] +=
            eq * relation.fingerprint_weight(usize::from(read.slot)) * (pos + neg * sign);
    }
    table
}

/// The fingerprint columns `f_pos`, `f_neg` over the rows.
pub fn fingerprint_columns(
    reads: &[TableRead],
    z_xi: &[Fr],
    relation: &RowRelation,
) -> (Vec<Fr>, Vec<Fr>) {
    let mut pos = vec![Fr::zero(); z_xi.len()];
    let mut neg = vec![Fr::zero(); z_xi.len()];
    for read in reads {
        let term = relation.fingerprint_weight(usize::from(read.slot)) * z_xi[read.src as usize];
        pos[read.row as usize] += term;
        if read.conjugated {
            neg[read.row as usize] -= term;
        } else {
            neg[read.row as usize] += term;
        }
    }
    (pos, neg)
}

/// Verifier: `Σ_i β_i·K_i(τ, r)` over the fixed copies and the fingerprint
/// kernels, summing each weight's kernels before the one multiplication.
pub fn copy_kernel_eval<O: VerifierObserver>(
    evaluator: &mut Evaluator<'_, O>,
    copies: &[ElemWiring],
    fingerprints: &[FingerprintGroup],
    relation: &RowRelation,
) -> Fr {
    // Buckets: `2·SLOTS` copy weights, then the plain and conjugated
    // fingerprint slots.
    let fp_base = 2 * SLOTS;
    let mut buckets = vec![Fr::zero(); fp_base + 2 * FP_SLOTS_GT];
    for group in copies {
        let maps: Vec<(usize, &Factor)> = group
            .maps
            .iter()
            .map(|(slot, side, map)| {
                let index = match side {
                    Side::X => usize::from(*slot),
                    Side::Y => SLOTS + usize::from(*slot),
                };
                (index, map)
            })
            .collect();
        evaluator.group_into(&group.cell, &maps, &mut buckets);
    }
    for group in fingerprints {
        let mut maps: Vec<(usize, &Factor)> = group
            .maps
            .iter()
            .map(|(slot, map)| (fp_base + usize::from(*slot), map))
            .collect();
        maps.extend(
            group
                .conj_maps
                .iter()
                .map(|(slot, map)| (fp_base + FP_SLOTS_GT + usize::from(*slot), map)),
        );
        evaluator.group_into(&group.cell, &maps, &mut buckets);
    }
    let (pos, neg) = (
        relation.copy_weight(2 * SLOTS),
        relation.copy_weight(2 * SLOTS + 1),
    );
    let mut total = Fr::zero();
    for (i, bucket) in buckets[..fp_base].iter().enumerate() {
        total += evaluator.mul(relation.copy_weight(i), *bucket);
    }
    for slot in 0..FP_SLOTS_GT {
        let plain = evaluator.mul(pos, buckets[fp_base + slot]);
        let conj = evaluator.mul(neg, buckets[fp_base + FP_SLOTS_GT + slot]);
        total += evaluator.mul(relation.fingerprint_weight(slot), plain + conj);
    }
    total
}
