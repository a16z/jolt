//! T2 on the wrapper stream: the verifying key (the layout and its six
//! verifier-key column groups committed once), the prover's column groups
//! built phase by phase ([`StreamBuilder`]), the two stage-A members, and the
//! `TermExporter` mapping the table's local terms to physical column ids.
//!
//! Protocol order (prover and verifier alike): the stream draws the offset
//! challenge `θ` after T1's phase-1a commitments → T2's phase 1b groups →
//! `ξ, α` → phase 2a → `fp_root` → phase 2b → `β, fp_combine, copy_root` →
//! phase 2c → `τ (LOG_ROWS), γ, λ, λ_lookup, constancy_root` → stage A
//! ([`commitment_phases`] declares the group and challenge counts). The
//! digit-link member's input claim is R's scalar-link claim `Σ_k W_k(ρ)·s_k`
//! plus the constant-one and offset bases' terms ([`link_input_claim`]),
//! `ρ` being that link's challenge and `W_k` the per-wire weights
//! [`link_weights`] sums over the wire's chain occurrences; the six
//! verifier-key groups are the last of T2's block and
//! [`LimbTableKey::pinned_commitments`] go into
//! `AssemblyStatement::pinned_commitments`.

use std::ops::Range;

use jolt_crypto::Bn254;
use jolt_field::{Fr, Ring};
use jolt_hyperkzg::{HyperKZGProverSetup, NoopVerifierObserver};

use crate::stream::{
    commit_packed, AffineForm as StreamAffineForm, Column, ColumnId as StreamColumnId, Commitment,
    CommitmentPhase, StreamError, Term as StreamTerm, TermContext as StreamTermContext,
    TermExporter, TermObserver,
};

use super::columns::{operand_columns, Columns, CHUNK_COLUMNS, LIMBS};
use super::digit_link::{link_terms, LinkMember};
use super::export::{exact_column, free_column, phases, pin_columns};
use super::layout::LOG_ROWS;
use super::lookup::{public_and_link_evals, LinkPowers, LookupColumns, PublicColumns, DIGIT_BITS};
use super::relation::{
    eq_tau_column, Challenges, Col, LookupConstants, RowMatrix, RowRelation, RowSumcheck, SLOTS,
};
use super::schedule::Layout;
use super::terms::{plain, powers_with, Mul};
use super::wiring::{copy_kernel_table, fingerprint_columns};

/// The challenges drawn after each committed phase (`phases()` order).
pub const PHASE_CHALLENGES: [usize; 4] = [2, 1, 3, LOG_ROWS + 4];

/// T2's transcript challenges: the offset challenge, the per-phase challenges,
/// and the digit link's `ρ`, which follows the phase-1b commitment.
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
        Self {
            theta,
            row: row_challenges(phase_challenges),
            rho,
        }
    }

    /// Extracts T2's phase challenges, skipping the wrapper link-challenge block
    /// after phase 1b when present.
    pub(crate) fn from_transcript(
        theta: Fr,
        challenges: &[Fr],
        challenge_offset: usize,
        rho_offset: usize,
    ) -> Self {
        let phase_challenges = if rho_offset == challenge_offset + Self::count() {
            challenges[challenge_offset..rho_offset].to_vec()
        } else {
            let first_end = challenge_offset + PHASE_CHALLENGES[0];
            let remaining = Self::count() - PHASE_CHALLENGES[0];
            let mut phase_challenges = challenges[challenge_offset..first_end].to_vec();
            phase_challenges
                .extend_from_slice(&challenges[rho_offset + 1..rho_offset + 1 + remaining]);
            phase_challenges
        };
        Self::from_challenges(theta, &phase_challenges, challenges[rho_offset])
    }

    /// Little-endian `τ` (the kernels' row point).
    pub fn tau_le(&self) -> Vec<Fr> {
        self.row.tau.iter().rev().copied().collect()
    }
}

/// The row member's challenges from the per-phase challenges in transcript
/// order (`PHASE_CHALLENGES`: `ξ, α | fp_root | β, fp_combine, copy_root | τ,
/// γ, λ, λ_lookup, constancy_root`).
///
/// # Panics
///
/// Panics unless `phase_challenges.len() == T2Challenges::count()`.
pub fn row_challenges(phase_challenges: &[Fr]) -> Challenges {
    assert_eq!(
        phase_challenges.len(),
        T2Challenges::count(),
        "T2 challenge count"
    );
    let (after_1b, rest) = phase_challenges.split_at(PHASE_CHALLENGES[0]);
    let (after_2a, rest) = rest.split_at(PHASE_CHALLENGES[1]);
    let (after_2b, after_2c) = rest.split_at(PHASE_CHALLENGES[2]);
    let (tau, stage) = after_2c.split_at(LOG_ROWS);
    Challenges {
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
    }
}

/// The digit link's input claim from R's scalar-link claim `Σ_k W_k(ρ)·s_k`
/// (`W = link_weights(layout, ρ)`, the named wires in the published order):
/// the constant-one base (`W_K`), the offset base (`W_{K+1}·θ`) and the
/// recoding window checks' constant `WINDOW_BOUND·ρ^{M+256}·Σ_{o<256} ρ^o`
/// are added, `K` the number of named wires and `M` the occurrence count.
pub fn link_input_claim(r_link_claim: Fr, rho: Fr, theta: Fr, layout: &Layout) -> Fr {
    link_input_claim_with(r_link_claim, rho, theta, layout, &mut plain)
}

pub fn link_input_claim_with(
    r_link_claim: Fr,
    rho: Fr,
    theta: Fr,
    layout: &Layout,
    mul: Mul<'_>,
) -> Fr {
    let powers = LinkPowers::new_with(layout, rho, mul);
    let weights = powers.base_weights(layout);
    let named = layout.digit_bases as usize - 2;
    r_link_claim + weights[named] + mul(weights[named + 1], theta) + powers.window_constant(mul)
}

/// Group count of one phase's columns at packing `packing`.
fn phase_groups(columns: Range<usize>, packing: usize) -> usize {
    columns.len().div_ceil(packing)
}

/// Groups of the pinned verifier-key columns at `packing`.
fn vk_groups(packing: usize) -> usize {
    phase_groups(Col::COMMITTED..Col::CLAIMED, packing)
}

/// T2's committed phases: group counts at `packing` and the challenges drawn
/// after each. The phase list owns the whole group geometry of the block:
/// the verifier-key groups sit after the last phase's columns and are
/// counted with it, so the sizes sum to every emitted group.
pub fn commitment_phases(packing: usize) -> [CommitmentPhase; 4] {
    commitment_phases_with_final_fill(packing, 0)
}

pub(crate) fn commitment_phases_with_final_fill(
    packing: usize,
    final_fill: usize,
) -> [CommitmentPhase; 4] {
    let specs = phases();
    std::array::from_fn(|i| CommitmentPhase {
        group_count: (specs[i].columns.len() + if i + 1 == specs.len() { final_fill } else { 0 })
            .div_ceil(packing)
            + if i + 1 == specs.len() {
                vk_groups(packing)
            } else {
                0
            },
        challenge_count: PHASE_CHALLENGES[i],
    })
}

/// Groups of T2's prover-committed columns.
pub fn prover_group_count(packing: usize) -> usize {
    prover_group_count_with_final_fill(packing, 0)
}

fn prover_group_count_with_final_fill(packing: usize, final_fill: usize) -> usize {
    phases()
        .iter()
        .enumerate()
        .map(|(index, spec)| {
            (spec.columns.len()
                + if index + 1 == phases().len() {
                    final_fill
                } else {
                    0
                })
            .div_ceil(packing)
        })
        .sum()
}

/// Absolute indices of T2's verifier-key groups when its block starts at
/// `group_offset`.
pub fn vk_group_range(packing: usize, group_offset: usize) -> Range<usize> {
    vk_group_range_with_final_fill(packing, group_offset, 0)
}

fn vk_group_range_with_final_fill(
    packing: usize,
    group_offset: usize,
    final_fill: usize,
) -> Range<usize> {
    let start = group_offset + prover_group_count_with_final_fill(packing, final_fill);
    start..start + vk_groups(packing)
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
        self.pinned_commitments_with_final_fill(group_offset, 0)
    }

    pub(crate) fn pinned_commitments_with_final_fill(
        &self,
        group_offset: usize,
        final_fill: usize,
    ) -> Vec<(usize, Commitment)> {
        vk_group_range_with_final_fill(self.packing, group_offset, final_fill)
            .zip(self.commitments.iter().copied())
            .collect()
    }

    pub(crate) fn column_ids_with_final_fill(
        &self,
        group_offset: usize,
        final_fill: usize,
    ) -> Vec<StreamColumnId> {
        let mut ids = vec![StreamColumnId { group: 0, slot: 0 }; Col::CLAIMED];
        let mut position = 0;
        for (index, phase) in phases().into_iter().enumerate() {
            for local in phase.columns.clone() {
                ids[local] = physical_id(group_offset, position, self.packing);
                position += 1;
            }
            if index + 1 == phases().len() {
                position += final_fill;
            }
            position = position.div_ceil(self.packing) * self.packing;
        }
        for target in ids.iter_mut().take(Col::CLAIMED).skip(Col::COMMITTED) {
            *target = physical_id(group_offset, position, self.packing);
            position += 1;
        }
        ids
    }

    pub(crate) fn final_fill_column_ids(
        &self,
        group_offset: usize,
        count: usize,
    ) -> Vec<StreamColumnId> {
        let before_final = phases()
            .iter()
            .take(phases().len() - 1)
            .map(|phase| phase.columns.len().div_ceil(self.packing) * self.packing)
            .sum::<usize>();
        let start = before_final + phases()[phases().len() - 1].columns.len();
        (start..start + count)
            .map(|position| physical_id(group_offset, position, self.packing))
            .collect()
    }
}

/// T2's columns as stream groups in phase order, then the verifier-key
/// groups. Chunks are `u16`, digit bits / flags / indicators are bits,
/// multiplicities `u32`, everything else full field elements.
pub struct StreamColumns {
    /// Physical id of every local column (`Col::CLAIMED`).
    pub ids: Vec<StreamColumnId>,
    pub group_count: usize,
    pub vk_groups: Range<usize>,
}

/// What [`StreamBuilder::finish`] hands the prover: the row relation of the
/// drawn challenges, the row member's matrix (`Col::WIDTH` columns: the
/// claimed columns in `Col` order — `matrix[Col::D]` is the digit link's
/// digit column — then the public ones) and the stream columns.
pub struct StreamWitness {
    pub relation: RowRelation,
    pub matrix: RowMatrix,
    pub stream: StreamColumns,
}

/// T2's prover witness assembled in protocol order — the one owner of every
/// column computation. Each phase method consumes exactly the challenges
/// drawn before that phase and returns its stream columns, padded to whole
/// groups (phase 2c's are followed by the verifier-key columns, which
/// [`commitment_phases`] counts with it); [`StreamBuilder::finish`] derives
/// the row relation and the members' matrix from the cached phase values.
/// Phases are callable only in protocol order.
pub struct StreamBuilder<'a> {
    layout: &'a Layout,
    chunks: &'a Columns,
    packing: usize,
    public: PublicColumns,
    columns: Vec<Column>,
    /// Position in `columns` of every local column emitted so far.
    positions: Vec<usize>,
    phase_start: usize,
    /// Local columns emitted in the current phase (checked against `phases()`).
    phase_locals: Vec<usize>,
    /// The per-phase challenges in transcript order ([`PHASE_CHALLENGES`]).
    drawn: Vec<Fr>,
    final_fill: usize,
    stage: Stage,
}

/// The phase the builder expects next, carrying the phase value the next
/// phase consumes.
enum Stage {
    OneB,
    TwoA,
    /// `Z_ξ(v)` per row, for the fingerprints.
    TwoB {
        z_xi: Vec<Fr>,
    },
    /// The `fp_root` powers, for the lookup helpers.
    TwoC {
        fp_pow: Vec<Fr>,
    },
    Finish,
}

impl<'a> StreamBuilder<'a> {
    /// # Panics
    ///
    /// Panics unless `packing` is a power of two and `chunks` has
    /// `2^LOG_ROWS` rows.
    pub fn new(layout: &'a Layout, chunks: &'a Columns, packing: usize) -> Self {
        assert!(packing.is_power_of_two());
        assert_eq!(chunks.rows(), 1usize << LOG_ROWS);
        Self {
            layout,
            chunks,
            packing,
            public: PublicColumns::new(layout),
            columns: Vec::new(),
            positions: vec![0; Col::CLAIMED],
            phase_start: 0,
            phase_locals: Vec::new(),
            drawn: Vec::new(),
            final_fill: 0,
            stage: Stage::OneB,
        }
    }

    /// Phase 1b (after `θ`): the chunks, digit bits and values `D`, the
    /// lookup and range multiplicities, the sign flags.
    pub fn phase_1b(&mut self) -> &[Column] {
        assert!(
            matches!(self.stage, Stage::OneB),
            "phase 1b is the first phase"
        );
        self.begin();
        for j in 0..CHUNK_COLUMNS {
            let chunks = self.chunks.chunk_column(j);
            self.push(Col::CHUNKS + j, Column::U16(chunks));
        }
        for b in 0..DIGIT_BITS {
            let bits = self.public.digits[b].clone();
            self.push(Col::DIGITS + b, Column::Bits(bits));
        }
        let (digit_values, m_pos, m_neg) = (
            self.public.digit_values.clone(),
            self.public.m_pos.clone(),
            self.public.m_neg.clone(),
        );
        self.push(Col::D, Column::Fr(digit_values));
        self.push(Col::M_POS, Column::U32(m_pos));
        self.push(Col::M_NEG, Column::U32(m_neg));
        let range_mult = self.chunks.range_multiplicities(&self.public.digits);
        self.push(Col::MULT, Column::U32(range_mult));
        let flags = self.chunks.flags.clone();
        self.push(Col::FLAG, Column::Bits(flags));
        self.stage = Stage::TwoA;
        self.end(0)
    }

    /// Phase 2a (after `ξ, α`): the operand columns `X, Y`, the range helpers
    /// and the range inverse table.
    pub fn phase_2a(&mut self, xi: Fr, alpha: Fr) -> &[Column] {
        assert!(
            matches!(self.stage, Stage::TwoA),
            "phase 2a follows phase 1b"
        );
        self.drawn.extend([xi, alpha]);
        self.begin();
        let z_xi = self.chunks.xi_values(xi);
        let operands = operand_columns(&self.layout.program, &z_xi, SLOTS);
        for (s, operand) in operands.into_iter().enumerate() {
            self.push(Col::X + s, Column::Fr(operand));
        }
        let helpers = self.chunks.range_helpers(alpha, &self.public.digits);
        for (g, helper) in helpers.into_iter().enumerate() {
            self.push(Col::HELPERS + g, Column::Fr(helper));
        }
        self.push(Col::INV, Column::Fr(PublicColumns::inverse_table(alpha)));
        self.stage = Stage::TwoB { z_xi };
        self.end(1)
    }

    /// Phase 2b (after `fp_root`): the fingerprint columns `f_pos`, `f_neg`.
    pub fn phase_2b(&mut self, fp_root: Fr) -> &[Column] {
        assert!(
            matches!(self.stage, Stage::TwoB { .. }),
            "phase 2b follows phase 2a"
        );
        let Stage::TwoB { z_xi } = &mut self.stage else {
            unreachable!("asserted above")
        };
        let z_xi = std::mem::take(z_xi);
        self.drawn.push(fp_root);
        self.begin();
        let fp_pow = powers_with(fp_root, SLOTS, &mut plain);
        let (pos, neg) = fingerprint_columns(&self.layout.table_reads, &z_xi, &fp_pow);
        self.push(Col::F_POS, Column::Fr(pos));
        self.push(Col::F_NEG, Column::Fr(neg));
        self.stage = Stage::TwoC { fp_pow };
        self.end(2)
    }

    /// Phase 2c (after `β, fp_combine, copy_root`): the lookup helpers `h`,
    /// `g_pos`, `g_neg`, caller-owned fill columns, then the verifier-key columns.
    pub fn phase_2c(
        &mut self,
        beta: Fr,
        fp_combine: Fr,
        copy_root: Fr,
        fill: Vec<Column>,
    ) -> &[Column] {
        assert!(
            matches!(self.stage, Stage::TwoC { .. }),
            "phase 2c follows phase 2b"
        );
        let Stage::TwoC { fp_pow } = &mut self.stage else {
            unreachable!("asserted above")
        };
        let fp_pow = std::mem::take(fp_pow);
        self.drawn.extend([beta, fp_combine, copy_root]);
        self.begin();
        let lookup = {
            let y: Vec<&[Fr]> = (0..SLOTS).map(|s| self.full(Col::Y + s)).collect();
            LookupColumns::new(
                &self.public,
                &y,
                self.full(Col::F_POS),
                self.full(Col::F_NEG),
                &fp_pow,
                beta,
                fp_combine,
            )
        };
        self.push(Col::H, Column::Fr(lookup.h));
        self.push(Col::G_POS, Column::Fr(lookup.g_pos));
        self.push(Col::G_NEG, Column::Fr(lookup.g_neg));
        let rows = 1usize << LOG_ROWS;
        assert!(fill.iter().all(|column| column.len() == rows));
        self.final_fill = fill.len();
        self.columns.extend(fill);
        let _ = self.end(3);
        for (i, column) in vk_columns(self.layout, self.packing)
            .into_iter()
            .enumerate()
        {
            if i < Col::CLAIMED - Col::COMMITTED {
                self.positions[Col::COMMITTED + i] = self.columns.len();
            }
            self.columns.push(column);
        }
        self.stage = Stage::Finish;
        &self.columns[self.phase_start..]
    }

    /// After the stage-A challenges (`τ, γ, λ, λ_lookup, constancy_root`):
    /// the row relation, the row member's matrix and the stream columns with
    /// their physical ids for a block starting at `group_offset`.
    pub fn finish(
        mut self,
        tau: Vec<Fr>,
        gamma: Fr,
        lambda: Fr,
        lambda_lookup: Fr,
        constancy_root: Fr,
        group_offset: usize,
    ) -> StreamWitness {
        assert!(
            matches!(self.stage, Stage::Finish),
            "finish follows phase 2c"
        );
        self.drawn.extend(tau);
        self.drawn
            .extend([gamma, lambda, lambda_lookup, constancy_root]);
        let relation = RowRelation::new(
            row_challenges(&self.drawn),
            LookupConstants {
                one_row: self.layout.one_cell * 16,
            },
        );
        let eq_tau = eq_tau_column(&relation.challenges.tau);
        let copy = copy_kernel_table(
            &self.layout.program,
            &self.public.kinds,
            &self.layout.table_reads,
            &eq_tau,
            &relation,
        );
        let constancy = self.public.constancy_weights(&eq_tau);
        let (small, id) = PublicColumns::small_and_id();
        let PublicColumns {
            sel,
            is_gt,
            is_g1,
            is_g2,
            s0,
            coord,
            ..
        } = self.public;
        let group_count = self.columns.len() / self.packing;
        let vk_groups = vk_group_range_with_final_fill(self.packing, group_offset, self.final_fill);
        debug_assert_eq!(vk_groups.end, group_offset + group_count);
        let ids = self
            .positions
            .iter()
            .map(|&position| physical_id(group_offset, position, self.packing))
            .collect();
        let mut columns = self.columns.into_iter().map(Some).collect::<Vec<_>>();
        let mut matrix = self
            .positions
            .into_iter()
            .map(|position| {
                columns[position]
                    .take()
                    .unwrap_or_else(|| unreachable!("one position per T2 column"))
            })
            .collect::<Vec<_>>();
        matrix.extend([
            Column::Fr(eq_tau),
            Column::Fr(copy),
            Column::Fr(sel),
            Column::Fr(is_gt),
            Column::Fr(is_g1),
            Column::Fr(is_g2),
            Column::Fr(s0),
            Column::Fr(coord),
            Column::Fr(constancy),
            Column::Fr(small),
            Column::Fr(id),
        ]);
        StreamWitness {
            relation,
            matrix: RowMatrix::new(matrix),
            stream: StreamColumns {
                ids,
                group_count,
                vk_groups,
            },
        }
    }

    fn begin(&mut self) {
        self.phase_start = self.columns.len();
        self.phase_locals.clear();
    }

    /// Checks the phase emitted exactly the columns `phases()[phase]`
    /// declares (the geometry [`commitment_phases`] publishes), pads it to
    /// whole groups and returns its columns.
    fn end(&mut self, phase: usize) -> &[Column] {
        let declared: Vec<usize> = phases()[phase].columns.clone().collect();
        assert_eq!(
            self.phase_locals, declared,
            "phase {phase} columns match the declared geometry"
        );
        self.pad();
        &self.columns[self.phase_start..]
    }

    fn pad(&mut self) {
        let rows = 1usize << LOG_ROWS;
        pad(&mut self.columns, self.packing, || {
            Column::Bits(vec![0; rows])
        });
    }

    fn push(&mut self, local: usize, column: Column) {
        debug_assert_eq!(column.len(), 1usize << LOG_ROWS);
        self.positions[local] = self.columns.len();
        self.phase_locals.push(local);
        self.columns.push(column);
    }

    /// An emitted full-field column.
    fn full(&self, local: usize) -> &[Fr] {
        match &self.columns[self.positions[local]] {
            Column::Fr(values) => values,
            Column::Bits(_) | Column::U16(_) | Column::U32(_) => {
                unreachable!("column {local} is full-field")
            }
        }
    }
}

fn physical_id(group_offset: usize, column: usize, packing: usize) -> StreamColumnId {
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
    pub fn new(relation: &'a RowRelation, matrix: &'a RowMatrix, layout: &Layout, rho: Fr) -> Self {
        Self {
            rows: RowSumcheck::new_typed(relation, matrix),
            link: LinkMember::new_from_matrix(layout, rho, matrix),
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
        T2Challenges::from_transcript(
            challenges[self.theta_offset],
            challenges,
            self.challenge_offset,
            self.rho_offset,
        )
    }

    fn export(
        &self,
        context: &StreamTermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<StreamTerm> {
        let challenges = self.challenges(context.challenges);
        let tau_le = challenges.tau_le();
        // The stream's stage point is big-endian (the members bind the most
        // significant row bit first); the kernels read little-endian points.
        let r_le: Vec<Fr> = context.row_point.iter().rev().copied().collect();
        let lookup = LookupConstants {
            one_row: self.layout.one_cell * 16,
        };
        let relation =
            RowRelation::new_with(challenges.row, lookup, &mut |a, b| observer.fr_mul(a, b));
        let (public, link) = public_and_link_evals(
            self.layout,
            &relation,
            &tau_le,
            &r_le,
            challenges.rho,
            observer,
        );
        let mut mul = |a, b| observer.fr_mul(a, b);
        let rho_rows = context.batching_coefficients[self.row_member];
        let rho_link = context.batching_coefficients[self.link_member];
        let mut local = relation.batched_terms(&public, rho_rows, &mut mul);
        for mut term in link_terms(&link) {
            term.coefficient = mul(term.coefficient, rho_link);
            local.push(term);
        }
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
    fn max_factors(&self) -> usize {
        RowRelation::max_factors()
    }

    fn terms(&self, context: &StreamTermContext<'_>) -> Vec<StreamTerm> {
        self.export(context, &mut NoopVerifierObserver)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The phase sizes cover every emitted group, the verifier-key suffix
    /// included, at every packing the assembly uses.
    #[test]
    fn phases_cover_every_group() {
        for packing in [4, 16, 32] {
            let declared: usize = commitment_phases(packing)
                .iter()
                .map(|phase| phase.group_count)
                .sum();
            assert_eq!(
                declared,
                prover_group_count(packing) + vk_group_range(packing, 0).len(),
                "packing {packing}"
            );
        }
    }
}
