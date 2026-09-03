//! Digit-selected operands: the committed one-hot digit columns and the
//! operand-selection sumcheck that proves the selected part of the row
//! sumcheck's operand claims,
//! `Σ_{j,row} eq(r, row) · Σ_col Δ_col(j, u_col(row)) · F_col(j, row)`,
//! where `F_col` is the public multilinear "eq of the selected source"
//! (a product of kernel factors) the verifier evaluates in `O(bits)`.

use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use super::digits::CANDIDATES;
use super::layout::{eq_const, Bits, Factor, Side, LOG_CELLS, LOG_ROWS, ROWS};
use super::schedule::{ColumnSelectedPiece, DigitColumn, DigitEntry, Layout};
use super::template::{conjugated, DigitRule};
use super::wiring::SlotWeights;

pub const COLUMNS: [DigitColumn; 3] = [DigitColumn::Gt, DigitColumn::G1, DigitColumn::G2];
/// Bits of a digit column index: `j` (4) above the op index `u` (14).
pub const DIGIT_LOG: usize = LOG_CELLS + 4;

fn column_slot(column: DigitColumn) -> usize {
    match column {
        DigitColumn::Gt => 0,
        DigitColumn::G1 => 1,
        DigitColumn::G2 => 2,
    }
}

/// The three committed one-hot digit columns, `2^18` bits each: entry
/// `j·2^14 + u` is one iff op `u` of the column's group uses digit `j`.
pub struct DigitColumns {
    pub bits: [Vec<u8>; 3],
}

impl DigitColumns {
    pub fn from_entries(entries: &[DigitEntry]) -> Self {
        let mut bits: [Vec<u8>; 3] = std::array::from_fn(|_| vec![0u8; ROWS]);
        for entry in entries {
            let index = (usize::from(entry.j) << LOG_CELLS) | entry.index as usize;
            bits[column_slot(entry.column)][index] = 1;
        }
        Self { bits }
    }

    pub fn column(&self, column: DigitColumn) -> &[u8] {
        &self.bits[column_slot(column)]
    }
}

/// Pieces sharing one selected element relation, so the prover evaluates
/// the cell part once per row and the coordinate part once per `(c, j)`.
#[derive(Clone, Debug)]
pub struct SelectedGroup {
    pub column: DigitColumn,
    pub index_bits: Bits,
    pub factors: Vec<Factor>,
    pub entry_bits: Bits,
    pub rule: DigitRule,
    pub coord_bits: Bits,
    pub own_coord_bits: Bits,
    /// `(slot, side, own coordinate ↦ (element coordinate, κ))`.
    pub pieces: Vec<(u8, Side, CoordMap)>,
}

/// Own coordinate ↦ `(element coordinate, κ)`.
pub type CoordMap = Vec<Option<(u32, i32)>>;

pub fn group_pieces(selected: &[ColumnSelectedPiece]) -> Vec<SelectedGroup> {
    let mut groups: Vec<SelectedGroup> = Vec::new();
    for piece in selected {
        let p = &piece.piece;
        let existing = groups.iter_mut().find(|g| {
            g.column == piece.column
                && g.index_bits == piece.index_bits
                && g.factors == p.factors
                && g.entry_bits == p.entry_bits
                && g.rule == p.rule
                && g.coord_bits == p.coord_bits
                && g.own_coord_bits == p.own_coord_bits
        });
        let entry = (p.slot, p.side, p.coords.clone());
        match existing {
            Some(group) => group.pieces.push(entry),
            None => groups.push(SelectedGroup {
                column: piece.column,
                index_bits: piece.index_bits,
                factors: p.factors.clone(),
                entry_bits: p.entry_bits,
                rule: p.rule,
                coord_bits: p.coord_bits,
                own_coord_bits: p.own_coord_bits,
                pieces: vec![entry],
            }),
        }
    }
    groups
}

impl SelectedGroup {
    /// The table entry field value and coordinate sign rule for digit `j`;
    /// `None` is the GT identity digit.
    fn entry(&self, j: u8) -> Option<(u32, bool)> {
        match self.rule {
            DigitRule::Gt { .. } => {
                let d = i32::from(j) - 8;
                (d != 0).then(|| (d.unsigned_abs() - 1, d < 0))
            }
            DigitRule::Ec => Some((u32::from(j), false)),
        }
    }

    /// `Σ_pieces γ_piece · κ(c, j) · eq(r'_coord, target coordinate)` for own
    /// coordinate `c` and digit `j` (the identity digit reads the `one` row:
    /// only coordinate-0 targets, at that row's coordinate bits).
    fn coord_part(&self, c: u32, j: u8, weights: &SlotWeights, r_src: &[Fr]) -> Fr {
        let rv = &r_src[usize::from(self.coord_bits.lo)..usize::from(self.coord_bits.hi)];
        let mut sum = Fr::zero();
        for (slot, side, coords) in &self.pieces {
            let Some((target, kappa)) = coords[c as usize] else {
                continue;
            };
            let weight = weights.get(*slot, *side);
            match (self.rule, self.entry(j)) {
                (DigitRule::Gt { .. }, None) => {
                    // Identity digit: only coordinate-0 targets read the `one` row.
                    if target != 0 {
                        continue;
                    }
                    sum += weight * Fr::from_i64(i64::from(kappa));
                }
                (_, Some((_, negative))) => {
                    let sign = if negative && conjugated(target as u8) {
                        -1
                    } else {
                        1
                    };
                    sum += weight * Fr::from_i64(i64::from(kappa * sign)) * eq_const(rv, target);
                }
                (DigitRule::Ec, None) => unreachable!("EC digits always select an entry"),
            }
        }
        sum
    }

    /// `Π factors eq(r'_v, v(row))` for the non-identity digits, plus the
    /// identity source `eq(r', one)` for GT: returns `(cell part, one part)`.
    fn cell_part(&self, row: u32, r_src: &[Fr]) -> Option<Fr> {
        let mut product = Fr::one();
        for factor in &self.factors {
            let (v, w) = factor.apply(factor.u.extract(row))?;
            debug_assert_eq!(w, 1);
            if factor.v.width() > 0 {
                let rv = &r_src[usize::from(factor.v.lo)..usize::from(factor.v.hi)];
                product *= eq_const(rv, v);
            }
        }
        Some(product)
    }

    /// Verifier: the multilinear `F_group(j, row)` at `(ρ_j, ρ_row)`.
    pub fn f_mle(&self, rho_row: &[Fr], rho_j: &[Fr], weights: &SlotWeights, r_src: &[Fr]) -> Fr {
        // Cell factors: product over row fields (entry field handled with `j`).
        let mut cell = Fr::one();
        for factor in &self.factors {
            cell *= factor.mle(rho_row, r_src);
        }
        // (c, j) table: coordinate part × entry eq, identity digits separately.
        let own =
            &rho_row[usize::from(self.own_coord_bits.lo)..usize::from(self.own_coord_bits.hi)];
        let entry_r = &r_src[usize::from(self.entry_bits.lo)..usize::from(self.entry_bits.hi)];
        let mut table = Fr::zero();
        let mut identity = Fr::zero();
        for j in 0..CANDIDATES as u8 {
            let ej = eq_const(rho_j, u32::from(j));
            for c in 0..1u32 << self.own_coord_bits.width() {
                let cp = self.coord_part(c, j, weights, r_src);
                if cp.is_zero() {
                    continue;
                }
                let ec = eq_const(own, c);
                match self.entry(j) {
                    Some((entry, _)) => table += ej * ec * cp * eq_const(entry_r, entry),
                    None => identity += ej * ec * cp,
                }
            }
        }
        let one_part = match self.rule {
            DigitRule::Gt { one } => {
                // Domain factors only (no source fields) times eq(r', one).
                let mut domain = Fr::one();
                for factor in &self.factors {
                    if factor.v.width() == 0 {
                        domain *= factor.mle(rho_row, r_src);
                    }
                }
                domain * eq_const(r_src, one) * identity
            }
            DigitRule::Ec => Fr::zero(),
        };
        cell * table + one_part
    }
}

/// The digit link the verifier checks: `Δ̃_col` at `(ρ_j ∥ ρ_u)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DigitOpening {
    pub column: DigitColumn,
    /// Little-endian point over the column's 18 index bits (`u` low, `j` high).
    pub point: Vec<Fr>,
    pub value: Fr,
}

/// Prover of the operand-selection sumcheck over `(row, j)` (22 variables,
/// row bits first, degree 3). Tables are dense over the extended cube.
pub struct SelectionSumcheck {
    eq: Vec<Fr>,
    delta: [Vec<Fr>; 3],
    f: [Vec<Fr>; 3],
    size: usize,
    round: usize,
    pub cheat: bool,
}

const EXT_LOG: usize = LOG_ROWS + 4;

impl SelectionSumcheck {
    /// `r_be`: the row sumcheck's point (big-endian); `r_src_le`: not used by
    /// the prover — the source point `r'` is the wiring sumcheck's point,
    /// passed little-endian.
    pub fn new(
        layout: &Layout,
        digits: &DigitColumns,
        weights: &SlotWeights,
        r_be: &[Fr],
        r_src_le: &[Fr],
    ) -> Self {
        let groups = group_pieces(&layout.selected);
        let eq_rows = EqPolynomial::<Fr>::evals(r_be, None);
        let size = 1usize << EXT_LOG;
        let mut delta: [Vec<Fr>; 3] = std::array::from_fn(|_| vec![Fr::zero(); size]);
        let mut f: [Vec<Fr>; 3] = std::array::from_fn(|_| vec![Fr::zero(); size]);
        // Δ_col(j, u_col(row)) over the extended index (row low, j high).
        for (slot, column) in COLUMNS.iter().enumerate() {
            let bits = digits.column(*column);
            let index_bits = layout
                .selected
                .iter()
                .find(|p| p.column == *column)
                .map(|p| p.index_bits);
            let Some(index_bits) = index_bits else {
                continue;
            };
            delta[slot]
                .par_chunks_mut(ROWS)
                .enumerate()
                .for_each(|(j, chunk)| {
                    for (row, value) in chunk.iter_mut().enumerate() {
                        let u = index_bits.extract(row as u32) as usize;
                        if bits[(j << LOG_CELLS) | u] == 1 {
                            *value = Fr::one();
                        }
                    }
                });
        }
        for group in &groups {
            let slot = column_slot(group.column);
            let coord_width = 1u32 << group.own_coord_bits.width();
            let mut coord_table = vec![Fr::zero(); (CANDIDATES as u32 * coord_width) as usize];
            for j in 0..CANDIDATES as u8 {
                for c in 0..coord_width {
                    coord_table[usize::from(j) * coord_width as usize + c as usize] =
                        group.coord_part(c, j, weights, r_src_le);
                }
            }
            let entry_r =
                &r_src_le[usize::from(group.entry_bits.lo)..usize::from(group.entry_bits.hi)];
            let entry_eq: Vec<Fr> = (0..CANDIDATES as u8)
                .map(|j| {
                    group
                        .entry(j)
                        .map_or(Fr::zero(), |(e, _)| eq_const(entry_r, e))
                })
                .collect();
            let one_eq = match group.rule {
                DigitRule::Gt { one } => eq_const(r_src_le, one),
                DigitRule::Ec => Fr::zero(),
            };
            // Cell part per row (None outside the domain).
            let cell: Vec<Option<Fr>> = (0..ROWS as u32)
                .into_par_iter()
                .map(|row| group.cell_part(row, r_src_le))
                .collect();
            f[slot]
                .par_chunks_mut(ROWS)
                .enumerate()
                .for_each(|(j, chunk)| {
                    let j8 = j as u8;
                    for (row, value) in chunk.iter_mut().enumerate() {
                        let Some(cell_part) = cell[row] else {
                            continue;
                        };
                        let c = group.own_coord_bits.extract(row as u32);
                        let cp = coord_table[j * coord_width as usize + c as usize];
                        if cp.is_zero() {
                            continue;
                        }
                        *value += match group.entry(j8) {
                            Some(_) => cell_part * entry_eq[j] * cp,
                            None => one_eq * cp,
                        };
                    }
                });
        }
        let mut eq = vec![Fr::zero(); size];
        for chunk in eq.chunks_mut(ROWS) {
            chunk.copy_from_slice(&eq_rows);
        }
        Self {
            eq,
            delta,
            f,
            size,
            round: 0,
            cheat: false,
        }
    }

    /// `Σ_{j,row} eq·Σ_col Δ_col·F_col`: the selected part of the operand claims.
    pub fn input_claim(&self) -> Fr {
        (0..self.size)
            .into_par_iter()
            .map(|i| {
                let mut sum = Fr::zero();
                for col in 0..3 {
                    sum += self.delta[col][i] * self.f[col][i];
                }
                self.eq[i] * sum
            })
            .sum()
    }

    /// Final evaluations after every round: `(Δ_col(ρ), F_col(ρ))` per column.
    pub fn final_values(&self) -> [(Fr, Fr); 3] {
        assert_eq!(self.size, 1);
        std::array::from_fn(|col| (self.delta[col][0], self.f[col][0]))
    }

    fn round_poly(&self, claim: Fr) -> Vec<Fr> {
        let half = self.size / 2;
        let evals: [Fr; 4] = (0..half)
            .into_par_iter()
            .fold(
                || [Fr::zero(); 4],
                |mut acc, i| {
                    let (e0, e1) = (self.eq[2 * i], self.eq[2 * i + 1]);
                    let de = e1 - e0;
                    for col in 0..3 {
                        let (d0, d1) = (self.delta[col][2 * i], self.delta[col][2 * i + 1]);
                        let (f0, f1) = (self.f[col][2 * i], self.f[col][2 * i + 1]);
                        if d0.is_zero() && d1.is_zero() {
                            continue;
                        }
                        let (dd, df) = (d1 - d0, f1 - f0);
                        acc[0] += e0 * d0 * f0;
                        acc[1] += e1 * d1 * f1;
                        let (e2, d2, f2) = (e1 + de, d1 + dd, f1 + df);
                        acc[2] += e2 * d2 * f2;
                        let (e3, d3, f3) = (e2 + de, d2 + dd, f2 + df);
                        acc[3] += e3 * d3 * f3;
                    }
                    acc
                },
            )
            .reduce(
                || [Fr::zero(); 4],
                |a, b| std::array::from_fn(|i| a[i] + b[i]),
            );
        let mut coefficients = UnivariatePoly::from_evals(&evals).into_coefficients();
        if self.cheat {
            let tail: Fr = coefficients[1..].iter().fold(Fr::zero(), |acc, c| acc + *c);
            coefficients[0] = (claim - tail) * two_inverse();
        }
        coefficients
    }

    fn bind(&mut self, r: Fr) {
        let half = self.size / 2;
        let fold = |column: &mut Vec<Fr>| {
            for i in 0..half {
                column[i] = column[2 * i] + r * (column[2 * i + 1] - column[2 * i]);
            }
            column.truncate(half);
        };
        fold(&mut self.eq);
        for col in 0..3 {
            fold(&mut self.delta[col]);
            fold(&mut self.f[col]);
        }
        self.size = half;
        self.round += 1;
    }
}

fn two_inverse() -> Fr {
    Fr::from_u64(2)
        .inverse()
        .unwrap_or_else(|| unreachable!("2 is invertible"))
}

impl ProveRounds<Fr> for SelectionSumcheck {
    fn num_rounds(&self) -> usize {
        EXT_LOG
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        debug_assert_eq!(round, self.round);
        Ok(UnivariatePoly::new(self.round_poly(previous_claim)))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}

/// Verifier-side final check of the selection sumcheck: recomputes
/// `eq(r, ρ_row)·Σ_col Δ_col(ρ)·F̃_col(ρ)` from the claimed digit openings and
/// the public kernels; `rho_le` is the 22-variable point (row bits then `j`).
pub fn final_value(
    layout: &Layout,
    weights: &SlotWeights,
    r_be: &[Fr],
    r_src_le: &[Fr],
    rho_le: &[Fr],
    digit_values: &[(DigitColumn, Fr)],
) -> Fr {
    let (rho_row, rho_j) = rho_le.split_at(LOG_ROWS);
    let r_le: Vec<Fr> = r_be.iter().rev().copied().collect();
    let eq = EqPolynomial::<Fr>::mle(&r_le, rho_row);
    let groups = group_pieces(&layout.selected);
    let mut sum = Fr::zero();
    for column in COLUMNS {
        let Some(&(_, delta)) = digit_values.iter().find(|(c, _)| *c == column) else {
            continue;
        };
        let f: Fr = groups
            .iter()
            .filter(|g| g.column == column)
            .map(|g| g.f_mle(rho_row, rho_j, weights, r_src_le))
            .sum();
        sum += delta * f;
    }
    eq * sum
}

/// The digit openings the verifier needs: `Δ̃_col` at `(ρ_row[index bits] ∥ ρ_j)`.
pub fn digit_openings(layout: &Layout, rho_le: &[Fr]) -> Vec<(DigitColumn, Vec<Fr>)> {
    let (rho_row, rho_j) = rho_le.split_at(LOG_ROWS);
    let mut out = Vec::new();
    for column in COLUMNS {
        let Some(piece) = layout.selected.iter().find(|p| p.column == column) else {
            continue;
        };
        let bits = piece.index_bits;
        let mut point: Vec<Fr> = rho_row[usize::from(bits.lo)..usize::from(bits.hi)].to_vec();
        point.extend_from_slice(rho_j);
        out.push((column, point));
    }
    out
}
