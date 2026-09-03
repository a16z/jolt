//! Op templates: the row structure of one operation (slots over its own
//! rows and over input elements) and where each element's rows are relative
//! to the op's rows. A template placed on a family of ops yields both the
//! explicit rows the prover evaluates and the wiring kernels the verifier
//! evaluates; the two agree by construction because both resolve operands
//! through the same [`ElemRel`]s.

use ark_bn254::Fq;

use super::layout::{Bits, Factor, Kernel, Piece, Side, LOG_ROWS};
use super::program::{Program, RowId, RowSpec, Slot, Source};

/// Coordinate `coord` of element `elem`; element 0 is the op's own rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ref {
    pub elem: u8,
    pub coord: u8,
}

pub const fn own(coord: u8) -> Ref {
    Ref { elem: 0, coord }
}

pub const fn at(elem: u8, coord: u8) -> Ref {
    Ref { elem, coord }
}

/// A sum of products `Σ κ · a · b` over references.
pub type RefSlots = Vec<(Ref, Ref, i32)>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowKind {
    Compute,
    /// Prover witness `num / den` (a curve slope).
    Quotient {
        num: RefSlots,
        den: RefSlots,
    },
    /// Coordinate `coord` of the `Fq2` quotient `num / den`.
    QuotientFq2 {
        num: [RefSlots; 2],
        den: [RefSlots; 2],
        coord: u8,
    },
    /// Coordinate `coord` of the `Fq12` inverse of element 1.
    InverseFq12 {
        coord: u8,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemplateRow {
    pub slots: RefSlots,
    pub kind: RowKind,
    pub pin: Option<Fq>,
}

impl TemplateRow {
    pub fn compute(slots: RefSlots) -> Self {
        Self {
            slots,
            kind: RowKind::Compute,
            pin: None,
        }
    }

    pub fn pinned(slots: RefSlots, value: Fq) -> Self {
        Self {
            slots,
            kind: RowKind::Compute,
            pin: Some(value),
        }
    }

    pub fn witness(kind: RowKind) -> Self {
        Self {
            slots: Vec::new(),
            kind,
            pin: None,
        }
    }
}

/// Rows `0..rows.len()` of an op, referencing `elems` input elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Template {
    pub rows: Vec<TemplateRow>,
    pub elems: usize,
    /// Evaluation order of the rows (a row may sit before rows it reads).
    pub order: Vec<usize>,
}

impl Template {
    pub fn new(rows: Vec<TemplateRow>) -> Self {
        let elems = rows
            .iter()
            .flat_map(|row| row.slots.iter().flat_map(|(x, y, _)| [x.elem, y.elem]))
            .max()
            .map_or(0, usize::from);
        let order = (0..rows.len()).collect();
        Self { rows, elems, order }
    }

    pub fn with_order(mut self, order: Vec<usize>) -> Self {
        assert_eq!(order.len(), self.rows.len());
        self.order = order;
        self
    }

    pub fn max_slots(&self) -> usize {
        self.rows
            .iter()
            .map(|row| row.slots.len())
            .max()
            .unwrap_or(0)
    }
}

/// Where an element's rows are, relative to a row of the op that reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElemRel {
    /// Element rows at `coord_bits = offset + coord` of the source index whose
    /// other bits `factors` derive from the op's row index (op-level fields only).
    Structured {
        factors: Vec<Factor>,
        coord_bits: Bits,
        offset: u32,
    },
    /// A fixed list of rows (public constants, the `one` row): coordinate `i`
    /// is `rows[i]`.
    Rows(Vec<RowId>),
    /// A digit-selected element: `factors` fix every source bit except the
    /// table-entry field `entry_bits`, which the committed digit chooses.
    Selected {
        factors: Vec<Factor>,
        coord_bits: Bits,
        offset: u32,
        entry_bits: Bits,
        rule: DigitRule,
    },
}

/// How a digit `j ∈ [0,16)` (value `d = j − 8`) selects a table entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DigitRule {
    /// GT: entry `|d| − 1` holds `X^{|d|}`, negative digits read its conjugate
    /// (coordinates `≥ 6` negated), `d = 0` reads the identity row.
    Gt { one: RowId },
    /// EC: entry `j` holds `d·P + Z0`.
    Ec,
}

/// Coordinates of GT elements that a conjugation negates.
pub fn conjugated(coord: u8) -> bool {
    coord >= 6
}

impl ElemRel {
    pub fn structured(factors: Vec<Factor>, coord_bits: Bits, offset: u32) -> Self {
        Self::Structured {
            factors,
            coord_bits,
            offset,
        }
    }

    /// Concrete `(row, sign)` of `coord` for the op row `row`; `digit` is the
    /// selecting digit for [`ElemRel::Selected`]. `None`: the coordinate is
    /// structurally zero (identity coordinates) or the op row is outside the
    /// relation's domain.
    pub fn resolve(&self, row: u32, coord: u8, digit: Option<u8>) -> Option<(RowId, i32)> {
        match self {
            Self::Structured {
                factors,
                coord_bits,
                offset,
            } => {
                let mut src = apply_factors(factors, row)?;
                coord_bits.insert(&mut src, offset + u32::from(coord));
                Some((src, 1))
            }
            Self::Rows(rows) => Some((rows[usize::from(coord)], 1)),
            Self::Selected {
                factors,
                coord_bits,
                offset,
                entry_bits,
                rule,
            } => {
                let Some(j) = digit else {
                    unreachable!("selected element without a digit")
                };
                let d = i32::from(j) - 8;
                match rule {
                    DigitRule::Gt { one } => {
                        if d == 0 {
                            return (coord == 0).then_some((*one, 1));
                        }
                        let mut src = apply_factors(factors, row)?;
                        entry_bits.insert(&mut src, d.unsigned_abs() - 1);
                        coord_bits.insert(&mut src, offset + u32::from(coord));
                        let sign = if d < 0 && conjugated(coord) { -1 } else { 1 };
                        Some((src, sign))
                    }
                    DigitRule::Ec => {
                        let mut src = apply_factors(factors, row)?;
                        entry_bits.insert(&mut src, u32::from(j));
                        coord_bits.insert(&mut src, offset + u32::from(coord));
                        Some((src, 1))
                    }
                }
            }
        }
    }
}

fn apply_factors(factors: &[Factor], row: u32) -> Option<u32> {
    let mut src = 0u32;
    for factor in factors {
        let (v, _) = factor.apply(factor.u.extract(row))?;
        factor.v.insert(&mut src, v);
    }
    Some(src)
}

/// A digit-selected slot side: the sumcheck-proven part of the wiring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedPiece {
    pub slot: u8,
    pub side: Side,
    /// Source bits except the entry field, from the op's row index.
    pub factors: Vec<Factor>,
    pub entry_bits: Bits,
    pub rule: DigitRule,
    /// Own coordinate `c` ↦ `(element coordinate, κ)` for rows using this slot.
    pub coords: Vec<Option<(u32, i32)>>,
    pub coord_bits: Bits,
    pub own_coord_bits: Bits,
}

/// One family of ops sharing a template and element relations.
pub struct Family<'a> {
    pub name: &'static str,
    pub template: &'a Template,
    pub elems: Vec<ElemRel>,
    /// Row field holding the op's own coordinate; template row `i` sits at
    /// coordinate `own_offset + i`.
    pub own_bits: Bits,
    pub own_offset: u32,
    /// Restrictions selecting the family's ops among all rows with the same shape.
    pub domain: Vec<Factor>,
}

impl Family<'_> {
    /// Writes the op whose coordinate-0 row is `base` (digit for selected elements).
    pub fn emit(&self, program: &mut Program, base: RowId, digit: Option<u8>) {
        debug_assert_eq!(self.own_bits.extract(base), 0);
        for &i in &self.template.order {
            let row = &self.template.rows[i];
            let id = base + self.own_offset + i as u32;
            let resolve = |r: Ref| -> Option<(RowId, i32)> {
                if r.elem == 0 {
                    Some((base + self.own_offset + u32::from(r.coord), 1))
                } else {
                    self.elems[usize::from(r.elem) - 1].resolve(id, r.coord, digit)
                }
            };
            let slots_of = |refs: &RefSlots| -> Vec<Slot> {
                refs.iter()
                    .map(|(x, y, kappa)| match (resolve(*x), resolve(*y)) {
                        (Some((x, sx)), Some((y, sy))) => Slot {
                            x,
                            y,
                            kappa: kappa * sx * sy,
                        },
                        // Structurally zero operand: the slot keeps its
                        // position with a zero coefficient.
                        _ => Slot {
                            x: id,
                            y: id,
                            kappa: 0,
                        },
                    })
                    .collect()
            };
            let slots = slots_of(&row.slots);
            let source = match &row.kind {
                RowKind::Compute => Source::Compute,
                RowKind::Quotient { num, den } => Source::Quotient {
                    num: slots_of(num),
                    den: slots_of(den),
                },
                RowKind::QuotientFq2 { num, den, coord } => Source::QuotientFq2 {
                    num: [slots_of(&num[0]), slots_of(&num[1])],
                    den: [slots_of(&den[0]), slots_of(&den[1])],
                    coord: *coord,
                },
                RowKind::InverseFq12 { coord } => Source::InverseFq12 {
                    coords: Box::new(std::array::from_fn(|c| {
                        resolve(at(1, c as u8))
                            .map(|(row, sign)| vec![(row, sign)])
                            .unwrap_or_default()
                    })),
                    coord: *coord,
                },
            };
            program.write_row(
                id,
                RowSpec {
                    slots,
                    source,
                    pin: row.pin,
                },
            );
        }
    }

    /// The wiring kernels of this family: one piece per (slot, side, element).
    pub fn pieces(&self) -> (Vec<Piece>, Vec<SelectedPiece>) {
        let mut fixed = Vec::new();
        let mut selected = Vec::new();
        let width = 1usize << self.own_bits.width();
        for s in 0..self.template.max_slots() {
            for side in [Side::X, Side::Y] {
                let refs = |row: &TemplateRow| {
                    row.slots.get(s).map(|(x, y, k)| match side {
                        Side::X => (*x, *k),
                        Side::Y => (*y, 1),
                    })
                };
                let mut elems: Vec<u8> = self
                    .template
                    .rows
                    .iter()
                    .filter_map(|row| refs(row).map(|(r, _)| r.elem))
                    .collect();
                elems.sort_unstable();
                elems.dedup();
                for elem in elems {
                    let mut coords: Vec<Option<(u32, i32)>> = vec![None; width];
                    for (i, row) in self.template.rows.iter().enumerate() {
                        if let Some((r, k)) = refs(row) {
                            if r.elem == elem {
                                coords[self.own_offset as usize + i] =
                                    Some((u32::from(r.coord), k));
                            }
                        }
                    }
                    let piece = |factors: &[Factor], coord_bits: Bits, offset: u32| {
                        let map = coords
                            .iter()
                            .map(|entry| entry.map(|(c, k)| (offset + c, k)))
                            .collect();
                        let mut all = self.domain.clone();
                        all.extend_from_slice(factors);
                        all.push(Factor::map(self.own_bits, coord_bits, map));
                        Piece {
                            slot: s as u8,
                            side,
                            kernel: Kernel::new(all),
                        }
                    };
                    if elem == 0 {
                        let index = Bits::new(self.own_bits.hi, LOG_ROWS as u8);
                        fixed.push(piece(
                            &[Factor::same(index, index)],
                            self.own_bits,
                            self.own_offset,
                        ));
                        continue;
                    }
                    match &self.elems[usize::from(elem) - 1] {
                        ElemRel::Structured {
                            factors,
                            coord_bits,
                            offset,
                        } => fixed.push(piece(factors, *coord_bits, *offset)),
                        ElemRel::Rows(rows) => {
                            let map = coords
                                .iter()
                                .map(|entry| entry.map(|(c, k)| (rows[c as usize], k)))
                                .collect();
                            let mut all = self.domain.clone();
                            all.push(Factor::map(
                                self.own_bits,
                                Bits::new(0, LOG_ROWS as u8),
                                map,
                            ));
                            fixed.push(Piece {
                                slot: s as u8,
                                side,
                                kernel: Kernel::new(all),
                            });
                        }
                        ElemRel::Selected {
                            factors,
                            coord_bits,
                            offset,
                            entry_bits,
                            rule,
                        } => {
                            let mut all = self.domain.clone();
                            all.extend_from_slice(factors);
                            selected.push(SelectedPiece {
                                slot: s as u8,
                                side,
                                factors: all,
                                entry_bits: *entry_bits,
                                rule: *rule,
                                coords: coords
                                    .iter()
                                    .map(|entry| entry.map(|(c, k)| (offset + c, k)))
                                    .collect(),
                                coord_bits: *coord_bits,
                                own_coord_bits: self.own_bits,
                            });
                        }
                    }
                }
            }
        }
        (fixed, selected)
    }
}
