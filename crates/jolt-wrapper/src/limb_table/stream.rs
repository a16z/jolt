//! T2 on the wrapper stream: the verifying key (the layout and its six
//! verifier-key column groups committed once), the prover's column groups in
//! phase order, the two stage-A members, and the `TermExporter` mapping the
//! table's local terms to physical column ids.
//!
//! Protocol order (prover and verifier alike): the stream draws the offset
//! challenge `θ` after T1's phase-1a commitments → T2's phase 1b groups →
//! `ξ, α` → phase 2a → `fp_root` → phase 2b → `β, fp_combine, copy_root` →
//! phase 2c → `τ (LOG_ROWS), γ, λ, λ_lookup, constancy_root` → stage A
//! ([`commitment_phases`] declares the group and challenge counts). The
//! digit-link member's input claim is R's `DoryScalarLink` claim plus
//! `ρ^K + ρ^{K+1}·θ` ([`link_input_claim`]), `ρ` being that link's
//! challenge; the six verifier-key groups are the last of T2's block and
//! [`LimbTableKey::pinned_commitments`] go into
//! `AssemblyStatement::pinned_commitments`.

use std::ops::Range;

use ark_ff::PrimeField;
use jolt_crypto::Bn254;
use jolt_field::{Fr, Ring};
use jolt_hyperkzg::HyperKZGProverSetup;

use crate::stream::{
    commit_packed, AffineForm as StreamAffineForm, Column, ColumnId as StreamColumnId, Commitment,
    CommitmentPhase, StreamError, Term as StreamTerm, TermContext as StreamTermContext,
    TermExporter, TermObserver,
};

use super::columns::{Columns, CHUNK_COLUMNS, LIMBS};
use super::digit_link::{link_term, LinkMember};
use super::export::{columns, exact_column, free_column, phases, pin_columns, ClaimedColumns};
use super::layout::LOG_ROWS;
use super::lookup::{omega_column, omega_eval, public_evals, DIGIT_BITS};
use super::relation::{Challenges, Col, LookupConstants, RowRelation, RowSumcheck};
use super::schedule::Layout;
use super::terms::{plain, powers_with, Mul};

/// The challenges drawn after each committed phase (`phases()` order).
pub const PHASE_CHALLENGES: [usize; 4] = [2, 1, 3, LOG_ROWS + 4];

/// T2's transcript challenges: the offset challenge (drawn before phase
/// 1b), the per-phase challenges, and the digit link's `ρ` (R's link
/// challenge).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct T2Challenges {
    pub theta: Fr,
    pub row: Challenges,
    pub rho: Fr,
}

impl T2Challenges {
    /// Total per-phase challenge count.
    pub const fn count() -> usize {
        PHASE_CHALLENGES[0] + PHASE_CHALLENGES[1] + PHASE_CHALLENGES[2] + PHASE_CHALLENGES[3]
    }

    /// # Panics
    ///
    /// Panics unless `phase_challenges.len() == Self::count()`.
    pub fn from_challenges(theta: Fr, phase_challenges: &[Fr], rho: Fr) -> Self {
        assert_eq!(phase_challenges.len(), Self::count(), "T2 challenge count");
        let (after_1b, rest) = phase_challenges.split_at(PHASE_CHALLENGES[0]);
        let (after_2a, rest) = rest.split_at(PHASE_CHALLENGES[1]);
        let (after_2b, after_2c) = rest.split_at(PHASE_CHALLENGES[2]);
        let (tau, stage) = after_2c.split_at(LOG_ROWS);
        Self {
            theta,
            row: Challenges {
                tau: tau.to_vec(),
                xi: after_1b[0],
                alpha: after_1b[1],
                fp_root: after_2a[0],
                beta: after_2b[0],
                fp_combine: after_2b[1],
                copy_root: after_2b[2],
                gamma: stage[0],
                lambda: stage[1],
                lambda_lookup: stage[2],
                constancy_root: stage[3],
            },
            rho,
        }
    }

    /// Little-endian `τ` (the kernels' row point).
    pub fn tau_le(&self) -> Vec<Fr> {
        self.row.tau.iter().rev().copied().collect()
    }
}

/// The digit link's input claim from R's link claim `Σ_s ρ^s·scalar_s`: the
/// constant-one base (`ρ^K`) and the offset base (`ρ^{K+1}·θ`) are added,
/// `K` the number of named wires.
pub fn link_input_claim(r_link_claim: Fr, rho: Fr, theta: Fr, named_wires: usize) -> Fr {
    link_input_claim_with(r_link_claim, rho, theta, named_wires, &mut plain)
}

pub fn link_input_claim_with(
    r_link_claim: Fr,
    rho: Fr,
    theta: Fr,
    named_wires: usize,
    mul: Mul<'_>,
) -> Fr {
    let powers = powers_with(rho, named_wires + 2, mul);
    r_link_claim + powers[named_wires] + mul(powers[named_wires + 1], theta)
}

/// Group count of one phase's columns at packing `packing`.
fn phase_groups(columns: Range<usize>, packing: usize) -> usize {
    columns.len().div_ceil(packing)
}

/// T2's committed phases: group counts at `packing` and the challenges drawn
/// after each.
pub fn commitment_phases(packing: usize) -> [CommitmentPhase; 4] {
    let specs = phases();
    std::array::from_fn(|i| CommitmentPhase {
        group_count: phase_groups(specs[i].columns.clone(), packing),
        challenge_count: PHASE_CHALLENGES[i],
    })
}

/// Groups of T2's prover-committed columns.
pub fn prover_group_count(packing: usize) -> usize {
    phases()
        .iter()
        .map(|spec| phase_groups(spec.columns.clone(), packing))
        .sum()
}

/// Absolute indices of T2's verifier-key groups when its block starts at
/// `group_offset`.
pub fn vk_group_range(packing: usize, group_offset: usize) -> Range<usize> {
    let start = group_offset + prover_group_count(packing);
    start..start + phase_groups(Col::COMMITTED..Col::CLAIMED, packing)
}

/// The verifier-key columns of a layout as stream columns (the `pin` and
/// `free`/`exact` bits, the pinned limbs), padded to whole groups.
fn vk_columns(layout: &Layout, packing: usize) -> Vec<Column> {
    let (pin, limbs) = pin_columns(layout);
    let rows = pin.len();
    let bits = |values: &[Fr]| -> Column {
        Column::Bits(
            values
                .iter()
                .map(|v| u8::from(*v == Fr::from_u64(1)))
                .collect(),
        )
    };
    let mut out = vec![bits(&pin)];
    out.extend(limbs.into_iter().map(Column::Fr));
    out.push(bits(&free_column(layout)));
    out.push(bits(&exact_column(layout)));
    debug_assert_eq!(out.len(), Col::CLAIMED - Col::COMMITTED);
    pad(&mut out, packing, || Column::Bits(vec![0; rows]));
    out
}

/// T2's verifying key: the layout and its verifier-key groups committed once.
pub struct LimbTableKey {
    layout: Layout,
    packing: usize,
    commitments: Vec<Commitment>,
}

impl LimbTableKey {
    pub fn new(
        layout: Layout,
        packing: usize,
        setup: &HyperKZGProverSetup<Bn254>,
    ) -> Result<Self, StreamError> {
        let packed = commit_packed(&vk_columns(&layout, packing), packing, setup)?;
        if packed.commitments.len() != vk_group_range(packing, 0).len() {
            return Err(StreamError::StageCount);
        }
        Ok(Self {
            layout,
            packing,
            commitments: packed.commitments,
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn packing(&self) -> usize {
        self.packing
    }

    pub fn commitments(&self) -> &[Commitment] {
        &self.commitments
    }

    /// `(group index, commitment)` of every verifier-key group when T2's block
    /// starts at `group_offset`.
    pub fn pinned_commitments(&self, group_offset: usize) -> Vec<(usize, Commitment)> {
        vk_group_range(self.packing, group_offset)
            .zip(self.commitments.iter().copied())
            .collect()
    }
}

/// T2's columns as stream groups in phase order, then the verifier-key
/// groups. Chunks are `u16`, digit bits / flags / indicators are bits,
/// multiplicities `u32`, everything else full field elements.
pub struct StreamColumns {
    pub columns: Vec<Column>,
    /// Physical id of every local column (`Col::CLAIMED`).
    pub ids: Vec<StreamColumnId>,
    pub group_count: usize,
    pub vk_groups: Range<usize>,
}

impl StreamColumns {
    /// # Panics
    ///
    /// Panics unless `packing` is a power of two.
    pub fn new(
        claimed: &ClaimedColumns,
        chunks: &Columns,
        layout: &Layout,
        packing: usize,
        group_offset: usize,
    ) -> Self {
        assert!(packing.is_power_of_two());
        let rows = 1usize << LOG_ROWS;
        let mut out: Vec<Column> = Vec::new();
        let mut ids = vec![StreamColumnId { group: 0, slot: 0 }; Col::CLAIMED];
        let kinds = columns();
        for spec in phases() {
            for local in spec.columns.clone() {
                ids[local] = id(group_offset, out.len(), packing);
                out.push(stream_column(
                    local,
                    &claimed.columns[local],
                    chunks,
                    &kinds,
                ));
            }
            pad(&mut out, packing, || Column::Bits(vec![0; rows]));
        }
        debug_assert_eq!(out.len() / packing, prover_group_count(packing));
        for (i, column) in vk_columns(layout, packing).into_iter().enumerate() {
            if i < Col::CLAIMED - Col::COMMITTED {
                ids[Col::COMMITTED + i] = id(group_offset, out.len(), packing);
            }
            out.push(column);
        }
        let group_count = out.len() / packing;
        let vk_groups = vk_group_range(packing, group_offset);
        debug_assert_eq!(vk_groups.end, group_offset + group_count);
        Self {
            columns: out,
            ids,
            group_count,
            vk_groups,
        }
    }
}

/// The small-scalar encoding of local column `local`.
fn stream_column(
    local: usize,
    values: &[Fr],
    chunks: &Columns,
    kinds: &[super::export::ColumnSpec],
) -> Column {
    let spec = kinds
        .iter()
        .find(|c| c.first <= local && local < c.first + c.count)
        .unwrap_or_else(|| unreachable!("every claimed column has a spec"));
    match spec.name {
        "chunk" => Column::U16(chunks.chunk_column(local - Col::CHUNKS)),
        "digit_bit" | "sign_flag" => Column::Bits(values.iter().map(|v| small(*v) as u8).collect()),
        "lookup_mult_pos" | "lookup_mult_neg" | "range_mult" => {
            Column::U32(values.iter().map(|v| small(*v) as u32).collect())
        }
        _ => Column::Fr(values.to_vec()),
    }
}

/// A small nonnegative field element as an integer.
fn small(value: Fr) -> u64 {
    let limbs = ark_bn254::Fr::from(value).into_bigint();
    debug_assert!(limbs.0[1..].iter().all(|l| *l == 0));
    limbs.0[0]
}

fn id(group_offset: usize, column: usize, packing: usize) -> StreamColumnId {
    StreamColumnId {
        group: group_offset + column / packing,
        slot: column % packing,
    }
}

fn pad(columns: &mut Vec<Column>, packing: usize, zero: impl Fn() -> Column) {
    while !columns.len().is_multiple_of(packing) {
        columns.push(zero());
    }
}

/// The prover's two stage-A members for one draw of the challenges: the row
/// member (input claim zero) and the digit link.
pub struct Members<'a> {
    pub rows: RowSumcheck<'a>,
    pub link: LinkMember,
}

impl<'a> Members<'a> {
    /// `matrix`: the row member's `Col::WIDTH` columns (claimed then public).
    pub fn new(
        relation: &'a RowRelation,
        matrix: &[Vec<Fr>],
        layout: &Layout,
        digit_values: &[Fr],
        rho: Fr,
    ) -> Self {
        Self {
            rows: RowSumcheck::new(relation, matrix),
            link: LinkMember::new(omega_column(layout, rho), digit_values),
        }
    }
}

/// T2's `TermExporter`: derives the row relation from the phase challenges
/// (`challenge_offset` into `TermContext::challenges`, `θ` and `ρ` at their
/// own offsets), evaluates the public multilinears at the stage point and
/// maps the local terms to physical ids, every field multiplication of the
/// derivation observed.
pub struct StreamTermExporter<'a> {
    pub layout: &'a Layout,
    pub challenge_offset: usize,
    pub theta_offset: usize,
    pub rho_offset: usize,
    pub columns: &'a [StreamColumnId],
    pub row_member: usize,
    pub link_member: usize,
}

impl StreamTermExporter<'_> {
    pub fn challenges(&self, challenges: &[Fr]) -> T2Challenges {
        let count = T2Challenges::count();
        T2Challenges::from_challenges(
            challenges[self.theta_offset],
            &challenges[self.challenge_offset..self.challenge_offset + count],
            challenges[self.rho_offset],
        )
    }

    fn export(
        &self,
        context: &StreamTermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<StreamTerm> {
        let challenges = self.challenges(context.challenges);
        let tau_le = challenges.tau_le();
        let lookup = LookupConstants {
            one_row: self.layout.one_cell * 16,
        };
        let relation =
            RowRelation::new_with(challenges.row, lookup, &mut |a, b| observer.fr_mul(a, b));
        let public = public_evals(self.layout, &relation, &tau_le, context.row_point, observer);
        let omega = omega_eval(self.layout, challenges.rho, context.row_point, observer);
        let mut mul = |a, b| observer.fr_mul(a, b);
        let rho_rows = context.batching_coefficients[self.row_member];
        let rho_link = context.batching_coefficients[self.link_member];
        let mut local = relation.terms_with(&public, &mut mul);
        for term in &mut local {
            term.coefficient = mul(term.coefficient, rho_rows);
        }
        let mut link = link_term(omega);
        link.coefficient = mul(link.coefficient, rho_link);
        local.push(link);
        local
            .into_iter()
            .map(|term| StreamTerm {
                coefficient: term.coefficient,
                factors: term
                    .factors
                    .into_iter()
                    .map(|form| StreamAffineForm {
                        constant: form.constant,
                        weights: form
                            .weights
                            .into_iter()
                            .map(|(column, weight)| (self.columns[column.0 as usize], weight))
                            .collect(),
                    })
                    .collect(),
            })
            .collect()
    }
}

impl TermExporter for StreamTermExporter<'_> {
    fn terms(&self, context: &StreamTermContext<'_>) -> Vec<StreamTerm> {
        self.export(context, &mut jolt_hyperkzg::NoopVerifierObserver)
    }

    fn terms_observed(
        &self,
        context: &StreamTermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<StreamTerm> {
        self.export(context, observer)
    }
}

const _: () = assert!(CHUNK_COLUMNS + DIGIT_BITS + 5 == Col::PHASE_1B_END);
const _: () = assert!(LIMBS + 3 == Col::CLAIMED - Col::COMMITTED);
