//! T1's verifier-side algebra for the stream's term stage: the Fiat–Shamir
//! randomizers of both members, the batched final relation as affine-form
//! terms over local column ids, the virtual value columns for the copy links
//! and the link row maps.
//!
//! Local column ids: `0..227` committed bits (`163..227` the canonicality
//! witness), `227..291` wired bits (`din`, `bin`), `291..307` wired words
//! (`WiredWord::ALL`), `307..313` verifier-key columns (`VkColumn::ALL`,
//! committed once in the key — `adapter::HashTableKey`).
//! `adapter::StreamTermExporter` maps them to the stream's physical ids.

use jolt_field::{CanonicalEncoding, Fr, One, Ring, Zero};

use super::eq::{
    eq_evals_with, eq_plus_one_with, eq_points_with, eq_zero_with, plain, pow2, powers_with, Mul,
};
use super::layout::{
    wired_columns, Relation, WiredWord, WordColumn, B_XOR, CANON, CANON_BITS, COMMITTED,
    CONSTRAINTS, D_XOR, LOG_COLUMNS, MESSAGE, WIRED_BITS, WIRED_WORDS, WORD_BITS,
};
use super::schedule::{ByteSource, Squeeze, SymbolicSchedule};
use super::wiring::{
    canonicality, source, PublicInputs, Source, VkColumn, Weights, WiringStatement, WordSlot,
    LOG_CELL, MODULUS_HI, WIRING_TERMS,
};

pub type ColumnId = usize;

pub const WIRED_BIT_BASE: ColumnId = COMMITTED;
pub const WIRED_WORD_BASE: ColumnId = COMMITTED + WIRED_BITS;
pub const VK_BASE: ColumnId = COMMITTED + WIRED_BITS + WIRED_WORDS;
pub const COLUMNS: usize = VK_BASE + VkColumn::ALL.len();

pub fn wired_word_id(word: WiredWord) -> ColumnId {
    WIRED_WORD_BASE + word.index()
}

pub fn vk_id(column: VkColumn) -> ColumnId {
    VK_BASE + VkColumn::ALL.iter().position(|c| *c == column).unwrap_or(0)
}

/// Column-space index of the relation (`layout`) → column id.
fn wired_id(column_space: usize) -> ColumnId {
    wired_columns()
        .iter()
        .position(|&j| j == column_space)
        .map_or(column_space, |i| WIRED_BIT_BASE + i)
}

/// The Fiat–Shamir randomizers of T1's two members, drawn by the stream after
/// T1's columns are committed: the row relation's `τ₁` and constraint
/// batching challenge, the wiring zero-check's `τ₂` and slot batching
/// challenge (`2 · log_rows + 2` challenges).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct T1Challenges {
    pub tau_rows: Vec<Fr>,
    pub tau_wiring: Vec<Fr>,
    pub relation_gammas: Vec<Fr>,
    pub wiring_gammas: Vec<Fr>,
}

impl T1Challenges {
    pub const fn count(log_rows: usize) -> usize {
        2 * log_rows + 2
    }

    /// # Panics
    ///
    /// Panics unless `challenges.len() == Self::count(log_rows)`.
    pub fn from_challenges(challenges: &[Fr], log_rows: usize) -> Self {
        Self::from_challenges_with(challenges, log_rows, &mut plain)
    }

    /// `from_challenges` with the coefficient powers routed through `mul`.
    ///
    /// # Panics
    ///
    /// Panics unless `challenges.len() == Self::count(log_rows)`.
    pub fn from_challenges_with(challenges: &[Fr], log_rows: usize, mul: Mul<'_>) -> Self {
        assert_eq!(
            challenges.len(),
            Self::count(log_rows),
            "T1 challenge count"
        );
        let (tau_rows, rest) = challenges.split_at(log_rows);
        let (tau_wiring, rest) = rest.split_at(log_rows);
        Self {
            tau_rows: tau_rows.to_vec(),
            tau_wiring: tau_wiring.to_vec(),
            relation_gammas: powers_with(rest[0], CONSTRAINTS, mul),
            wiring_gammas: powers_with(rest[1], WIRING_TERMS, mul),
        }
    }

    pub fn relation(&self) -> Relation {
        Relation::new(&self.relation_gammas)
    }

    pub fn wiring(&self) -> WiringStatement<'_> {
        WiringStatement {
            gammas: &self.wiring_gammas,
            log_rows: self.tau_rows.len(),
        }
    }

    /// The members' input claims for the stage statement: the row relation
    /// sums to zero, the wiring zero-check to its public constant.
    pub fn input_claims(&self, public: &PublicInputs) -> [Fr; 2] {
        self.input_claims_with(public, &mut plain)
    }

    /// `input_claims` with the verifier's multiplications routed through `mul`.
    pub fn input_claims_with(&self, public: &PublicInputs, mul: Mul<'_>) -> [Fr; 2] {
        [
            Fr::zero(),
            self.wiring()
                .input_claim_with(&self.tau_wiring, public, mul),
        ]
    }
}

/// `constant + Σ w_c · v_c` over column evaluations at the common point.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AffineForm {
    pub constant: Fr,
    pub weights: Vec<(ColumnId, Fr)>,
}

impl AffineForm {
    pub fn column(id: ColumnId) -> Self {
        Self {
            constant: Fr::zero(),
            weights: vec![(id, Fr::one())],
        }
    }

    /// Add `weight · v_id`, merging with an existing weight of the column.
    pub fn add(&mut self, id: ColumnId, weight: Fr) {
        if weight.is_zero() {
            return;
        }
        match self.weights.iter_mut().find(|(c, _)| *c == id) {
            Some((_, w)) => *w += weight,
            None => self.weights.push((id, weight)),
        }
    }

    pub fn evaluate(&self, eval: &dyn Fn(ColumnId) -> Fr) -> Fr {
        self.weights
            .iter()
            .fold(self.constant, |acc, (id, w)| acc + *w * eval(*id))
    }
}

/// `coefficient · Π_j factor_j`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    pub coefficient: Fr,
    pub factors: Vec<AffineForm>,
}

pub fn evaluate_terms(terms: &[Term], eval: &dyn Fn(ColumnId) -> Fr) -> Fr {
    terms.iter().fold(Fr::zero(), |acc, term| {
        acc + term
            .factors
            .iter()
            .fold(term.coefficient, |p, f| p * f.evaluate(eval))
    })
}

/// Everything the batched final relation depends on besides the column
/// evaluations: the members' randomizers and batch coefficients, the stage's
/// row point (round order) and the public inputs.
pub struct FinalContext<'a> {
    pub challenges: &'a T1Challenges,
    pub row_point: &'a [Fr],
    pub rho_rows: Fr,
    pub rho_wiring: Fr,
    pub public: &'a PublicInputs,
}

/// The batched final claim of T1's two members as terms, every field
/// multiplication routed through `mul` (the verifier's operation counter):
/// `ρ_rows · eq(τ₁, r) · Q(v) + ρ_wiring · (eq(τ₂, r) · [wired + pins](v) − Σ_κ K_κ · V_κ(v))`.
/// Degree ≤ 2: one linear term, 163 booleanity squares, 64 XOR operand terms
/// `v_j · (γ_sq v_j + γ_cross w_j)` (booleanity and cross term merged), two
/// half-word pin products, two canonicality products.
pub fn terms(ctx: &FinalContext<'_>, mul: Mul<'_>) -> Vec<Term> {
    let mut linear = AffineForm::default();
    let mut terms = Vec::with_capacity(COMMITTED + 5);

    // Row relation.
    let relation = Relation::new_with(&ctx.challenges.relation_gammas, mul);
    let eq_rows = eq_points_with(&ctx.challenges.tau_rows, ctx.row_point, mul);
    let e = mul(ctx.rho_rows, eq_rows);
    for j in 0..1 << LOG_COLUMNS {
        let (square, cross) = (relation.gamma_sq[j], relation.gamma_cross[j]);
        if !cross.is_zero() {
            terms.push(Term {
                coefficient: e,
                factors: vec![
                    AffineForm::column(j),
                    AffineForm {
                        constant: Fr::zero(),
                        weights: vec![(j, square), (wired_id(j), cross)],
                    },
                ],
            });
        } else if !square.is_zero() {
            terms.push(Term {
                coefficient: mul(e, square),
                factors: vec![AffineForm::column(j), AffineForm::column(j)],
            });
        }
        if !relation.l1[j].is_zero() {
            linear.add(j, mul(e, relation.l1[j]));
        }
        if !relation.l2[j].is_zero() {
            linear.add(wired_id(j), mul(e, relation.l2[j]));
        }
    }

    // Wiring: the same quantities as `WiringStatement::final_check`, as forms.
    let wiring = ctx.challenges.wiring();
    let gammas = wiring.gammas;
    let n = wiring.log_rows;
    let (tau_hi, tau_lo) = ctx.challenges.tau_wiring.split_at(n - LOG_CELL);
    let (r_hi, r_lo) = ctx.row_point.split_at(n - LOG_CELL);
    let eq_tau_lo = eq_evals_with(tau_lo, mul);
    let eq_r_lo = eq_evals_with(r_lo, mul);
    let same_cell = eq_points_with(tau_hi, r_hi, mul);
    let previous_cell = eq_plus_one_with(r_hi, tau_hi, mul);
    let next_cell = eq_plus_one_with(tau_hi, r_hi, mul);
    let eq_lo = eq_tau_lo
        .iter()
        .zip(&eq_r_lo)
        .fold(Fr::zero(), |acc, (a, b)| acc + mul(*a, *b));
    let eq_full = mul(same_cell, eq_lo);
    let r_first_cell = eq_zero_with(r_hi, mul);
    let scale = mul(ctx.rho_wiring, eq_full);
    for (k, gamma) in gammas.iter().enumerate().take(WIRED_BITS) {
        linear.add(WIRED_BIT_BASE + k, mul(scale, *gamma));
    }
    for word in WiredWord::ALL {
        linear.add(
            wired_word_id(word),
            mul(scale, gammas[WordSlot::Word(word).gamma_index()]),
        );
    }
    let gamma_lo = gammas[WIRED_BITS + WIRED_WORDS];
    let gamma_hi = gammas[WIRED_BITS + WIRED_WORDS + 1];
    let mut is_lo = AffineForm::column(vk_id(VkColumn::LoIsConst));
    let mut is_hi = AffineForm::column(vk_id(VkColumn::HiIsConst));
    linear.add(vk_id(VkColumn::LoConst), -mul(scale, gamma_lo));
    linear.add(vk_id(VkColumn::HiConst), -mul(scale, gamma_hi));
    for (word, high, value) in ctx.public.tail_halves() {
        let weight = mul(r_first_cell, eq_r_lo[word]);
        let (is, gamma) = if high {
            (&mut is_hi, gamma_hi)
        } else {
            (&mut is_lo, gamma_lo)
        };
        is.constant += weight;
        let scaled = mul(scale, gamma);
        let scaled_weight = mul(scaled, weight);
        linear.constant -= mul(scaled_weight, Fr::from_u64(u64::from(value)));
    }
    let half = |from: usize| {
        let mut form = AffineForm::default();
        for k in 0..WORD_BITS / 2 {
            form.add(MESSAGE + from + k, pow2(k));
        }
        form
    };
    terms.push(Term {
        coefficient: mul(scale, gamma_lo),
        factors: vec![is_lo, half(0)],
    });
    terms.push(Term {
        coefficient: mul(scale, gamma_hi),
        factors: vec![is_hi, half(WORD_BITS / 2)],
    });
    // Canonicality: sel · (Σ_k 2^k canon_k + w_hi) − (r_hi − 1) · sel.
    let bound = Fr::from_u64(MODULUS_HI - 1);
    for (shifted, selector) in [
        (false, VkColumn::WireAligned),
        (true, VkColumn::WireShifted),
    ] {
        let gamma = gammas[WIRED_BITS + WIRED_WORDS + 2 + usize::from(shifted)];
        let mut value = AffineForm::default();
        for k in 0..CANON_BITS {
            value.add(CANON + k, pow2(k));
        }
        for (column, weight) in canonicality(shifted) {
            let id = if column < COMMITTED {
                column
            } else {
                WIRED_WORD_BASE + column - COMMITTED
            };
            value.add(id, weight);
        }
        let scaled = mul(scale, gamma);
        linear.add(vk_id(selector), -mul(scaled, bound));
        terms.push(Term {
            coefficient: scaled,
            factors: vec![AffineForm::column(vk_id(selector)), value],
        });
    }
    // Sources: kernel weights summed per (slot, group, weights), then expanded
    // once per distinct triple. `cell_position[kind][p] = cell factor · eqτ[p]`
    // is shared by every slot read at position `p`.
    let cell_position: Vec<[Fr; 3]> = eq_tau_lo
        .iter()
        .map(|e| {
            [
                mul(same_cell, *e),
                mul(previous_cell, *e),
                mul(next_cell, *e),
            ]
        })
        .collect();
    let mut kernels: Vec<((WordSlot, WordColumn, Weights), Fr)> = Vec::new();
    for (p, cell_position) in cell_position.iter().enumerate() {
        for slot in WordSlot::all() {
            let (key, kind, from) = match source(p, slot) {
                Source::Cell {
                    group,
                    weights,
                    delta,
                } => (
                    (slot, group, weights),
                    0,
                    (p as isize - isize::from(delta)) as usize,
                ),
                Source::Previous {
                    group,
                    weights,
                    position,
                } => ((slot, group, weights), 1, usize::from(position)),
                Source::Next {
                    group,
                    weights,
                    position,
                } => ((slot, group, weights), 2, usize::from(position)),
                Source::Zero | Source::Const(_) => continue,
            };
            let weight = mul(cell_position[kind], eq_r_lo[from]);
            match kernels.iter_mut().find(|(k, _)| *k == key) {
                Some((_, acc)) => *acc += weight,
                None => kernels.push((key, weight)),
            }
        }
    }
    // `γ_slot · 2^k` once per (word slot, bit).
    let mut slot_weights: Vec<((WordSlot, usize), Fr)> = Vec::new();
    for ((slot, group, weights), kernel) in kernels {
        let base = group.base();
        let scaled = mul(ctx.rho_wiring, kernel);
        for j in 0..WORD_BITS {
            if let Some(k) = weights.coefficient(j) {
                let known = slot_weights.iter().find(|(key, _)| *key == (slot, k));
                let slot_weight = if let Some((_, weight)) = known {
                    *weight
                } else {
                    let weight = wiring.slot_weight_with(slot, k, mul);
                    slot_weights.push(((slot, k), weight));
                    weight
                };
                linear.add(base + j, -mul(scaled, slot_weight));
            }
        }
    }
    terms.push(Term {
        coefficient: Fr::one(),
        factors: vec![linear],
    });
    terms
}

/// The linear form of one decoder over the 128 challenge bits, from the
/// production decoder itself (bit `k` → `decode(2^k)`).
fn decoder_coefficients(decode: fn(&[u8]) -> Fr) -> Vec<Fr> {
    (0..128)
        .map(|k| {
            let mut bytes = [0u8; 16];
            bytes[k / 8] = 1 << (k % 8);
            decode(&bytes)
        })
        .collect()
}

/// The decoded `Transcript::challenge` value of a squeeze, at the first
/// challenge row of its cell: bits 0..64 are the row's `D'`, `B'`; the words
/// `out[10]`, `out[11] mod 2^29` are the wired `a_in`, `x_in`.
pub fn challenge125() -> AffineForm {
    let c = decoder_coefficients(Fr::from_challenge_bytes);
    let mut form = AffineForm::default();
    for k in 0..WORD_BITS {
        form.add(D_XOR + k, c[k]);
        form.add(B_XOR + k, c[WORD_BITS + k]);
    }
    form.add(wired_word_id(WiredWord::AIn), c[64]);
    form.add(wired_word_id(WiredWord::XIn), c[96]);
    form
}

/// The decoded `Transcript::challenge_scalar` value of a squeeze, at the
/// first challenge row of its cell (`bswap(out[10])`, `bswap(out[11])` are
/// the wired `y_in`, `z_in`).
pub fn challenge_scalar128() -> AffineForm {
    let c = decoder_coefficients(Fr::from_scalar_challenge_bytes);
    let mut form = AffineForm::default();
    for k in 0..WORD_BITS {
        form.add(D_XOR + k, c[k]);
        form.add(B_XOR + k, c[WORD_BITS + k]);
    }
    form.add(wired_word_id(WiredWord::YIn), c[88]);
    form.add(wired_word_id(WiredWord::ZIn), c[120]);
    form
}

/// The absorbed field element (32 bytes big-endian) starting at an aligned
/// wire row: `Σ_w 2^{8(28 − 4w)} · bswap(m(row + w))` with `bswap(m(row))`
/// over the row's `m` bits and the rest the wired `fr_next` words. Only
/// canonical encodings reach this value: the wiring member's canonicality
/// constraint rejects every `x + r`.
pub fn fr_word() -> AffineForm {
    let mut form = AffineForm::default();
    for j in 0..WORD_BITS {
        form.add(MESSAGE + j, pow2(224 + 8 * (3 - j / 8) + j % 8));
    }
    for i in 1..8u8 {
        form.add(
            wired_word_id(WiredWord::FrNext(i)),
            pow2(8 * (28 - 4 * usize::from(i))),
        );
    }
    form
}

/// The absorbed field element whose first two bytes are the high half of a
/// wire row's `m` (wires absorbed before the first squeeze sit two bytes into
/// their words, after the 22-byte preamble tail): the high half of `m` is
/// bytes 0–1, `fr_next` words bytes 2–29, `fr_tail` bytes 30–31.
pub fn fr_word_shifted() -> AffineForm {
    let mut form = AffineForm::default();
    for j in 16..WORD_BITS {
        form.add(MESSAGE + j, pow2(240 + 8 * (3 - j / 8) + j % 8));
    }
    for i in 1..8u8 {
        form.add(
            wired_word_id(WiredWord::FrNext(i)),
            pow2(8 * (30 - 4 * usize::from(i))),
        );
    }
    form.add(wired_word_id(WiredWord::FrTail), Fr::one());
    form
}

/// Link identities as row maps: `(wire index, row)` of every absorbed field
/// element by word alignment (byte 0 at word byte 0 → `fr_word`; at word
/// byte 2 → `fr_word_shifted`), `(squeeze, first challenge row)` and every
/// element / public byte `(source, row, byte)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LinkMap {
    pub wires: Vec<(u32, usize)>,
    pub wires_shifted: Vec<(u32, usize)>,
    pub challenges: Vec<(Squeeze, usize)>,
    pub bytes: Vec<(ByteSource, usize, u8)>,
}

impl LinkMap {
    pub fn new(schedule: &SymbolicSchedule) -> Self {
        let mut map = Self {
            challenges: schedule.challenge_rows(),
            ..Self::default()
        };
        for (index, row, shifted) in schedule.wire_rows() {
            if shifted {
                map.wires_shifted.push((index, row));
            } else {
                map.wires.push((index, row));
            }
        }
        for (source, row, byte) in schedule.byte_links() {
            if matches!(
                source,
                ByteSource::Public { .. } | ByteSource::Element { .. }
            ) {
                map.bytes.push((source, row, byte));
            }
        }
        map
    }
}
