//! The exported interface of T1 for the stream assembler (W5) and the copy
//! links (W6-RT): the column list in packing order, the two stage-A members,
//! the batched final relation as affine-form terms over column evaluations,
//! the virtual value columns and the link identities.
//!
//! Column ids: `0..163` committed bits, `163..227` wired bits (`din`, `bin`),
//! `227..241` wired words (`WiredWord::ALL`), `241..245` verifier-key columns
//! (`VkColumn::ALL`).

use jolt_field::{CanonicalEncoding, Fr, One, Ring, Zero};

use super::layout::{
    wired_columns, Relation, WiredWord, WordColumn, B_XOR, COMMITTED, DEGREE, D_XOR, LOG_COLUMNS,
    MESSAGE, WIRED_BITS, WIRED_WORDS, WORD_BITS,
};
use super::schedule::{ByteSource, Squeeze, SymbolicSchedule};
use super::wiring::{
    eq_points, source, PublicInputs, Source, VkColumn, Weights, WiringStatement, WordSlot,
    CELL_ROWS, LOG_CELL,
};
use jolt_poly::{EqPlusOnePolynomial, EqPolynomial};

pub type ColumnId = usize;

pub const WIRED_BIT_BASE: ColumnId = COMMITTED;
pub const WIRED_WORD_BASE: ColumnId = COMMITTED + WIRED_BITS;
pub const VK_BASE: ColumnId = COMMITTED + WIRED_BITS + WIRED_WORDS;
pub const COLUMNS: usize = VK_BASE + VkColumn::ALL.len();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnKind {
    Bit,
    U16,
    U32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnSpec {
    pub id: ColumnId,
    pub name: String,
    pub kind: ColumnKind,
    /// Prover-committed (packing order) or verifier-key column.
    pub vk: bool,
}

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

/// Every column, committed ones first in packing order.
pub fn column_specs() -> Vec<ColumnSpec> {
    let mut specs = Vec::with_capacity(COLUMNS);
    let bit = |id: ColumnId, name: String| ColumnSpec {
        id,
        name,
        kind: ColumnKind::Bit,
        vk: false,
    };
    let groups = [("a_out", 0), ("d_xor", 32), ("c_out", 64), ("b_xor", 96)];
    for (name, base) in groups {
        for k in 0..WORD_BITS {
            specs.push(bit(base + k, format!("{name}[{k}]")));
        }
    }
    for (id, name) in [(128, "carry_a_lo"), (129, "carry_a_hi"), (130, "carry_c")] {
        specs.push(bit(id, name.to_string()));
    }
    for k in 0..WORD_BITS {
        specs.push(bit(MESSAGE + k, format!("m[{k}]")));
    }
    for k in 0..WORD_BITS {
        specs.push(bit(WIRED_BIT_BASE + k, format!("din[{k}]")));
    }
    for k in 0..WORD_BITS {
        specs.push(bit(WIRED_BIT_BASE + WORD_BITS + k, format!("bin[{k}]")));
    }
    for word in WiredWord::ALL {
        specs.push(ColumnSpec {
            id: wired_word_id(word),
            name: format!("{word:?}"),
            kind: ColumnKind::U32,
            vk: false,
        });
    }
    for column in VkColumn::ALL {
        specs.push(ColumnSpec {
            id: vk_id(column),
            name: format!("vk_{column:?}"),
            kind: if column.is_bit() {
                ColumnKind::Bit
            } else {
                ColumnKind::U16
            },
            vk: true,
        });
    }
    specs
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

/// One stage-A member of T1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemberSpec {
    pub name: &'static str,
    pub degree: usize,
    pub rounds: usize,
    pub offset: usize,
}

/// The row relation and the wiring zero-check, head-aligned.
pub fn members(log_rows: usize) -> [MemberSpec; 2] {
    [
        MemberSpec {
            name: "t1-rows",
            degree: DEGREE,
            rounds: log_rows,
            offset: 0,
        },
        MemberSpec {
            name: "t1-wiring",
            degree: 3,
            rounds: log_rows,
            offset: 0,
        },
    ]
}

/// Everything the batched final relation depends on besides the column
/// evaluations: the members' `τ`s and batch coefficients, the stage's row
/// challenges (round order) and the public inputs.
pub struct FinalContext<'a> {
    pub relation: &'a Relation,
    pub wiring: &'a WiringStatement<'a>,
    pub tau_rows: &'a [Fr],
    pub tau_wiring: &'a [Fr],
    pub challenges: &'a [Fr],
    pub rho_rows: Fr,
    pub rho_wiring: Fr,
    pub public: &'a PublicInputs,
}

/// The batched final claim of T1's two members as terms:
/// `ρ_rows · eq(τ₁, r) · Q(v) + ρ_wiring · (eq(τ₂, r) · [wired + pins](v) − Σ_κ K_κ · V_κ(v))`.
/// Degree ≤ 2: one linear term, 163 booleanity squares, 64 XOR cross terms,
/// two half-word pin products.
pub fn terms(ctx: &FinalContext<'_>) -> Vec<Term> {
    let mut linear = AffineForm::default();
    let mut terms = Vec::with_capacity(COMMITTED + WIRED_BITS + 3);

    // Row relation.
    let e = ctx.rho_rows * super::layout::eq_rounds(ctx.tau_rows, ctx.challenges);
    let rel = ctx.relation;
    for j in 0..1 << LOG_COLUMNS {
        if !rel.gamma_sq[j].is_zero() {
            terms.push(Term {
                coefficient: e * rel.gamma_sq[j],
                factors: vec![AffineForm::column(j), AffineForm::column(j)],
            });
        }
        if !rel.gamma_cross[j].is_zero() {
            terms.push(Term {
                coefficient: e * rel.gamma_cross[j],
                factors: vec![AffineForm::column(j), AffineForm::column(wired_id(j))],
            });
        }
        linear.add(j, e * rel.l1[j]);
        linear.add(wired_id(j), e * rel.l2[j]);
    }

    // Wiring: the same quantities as `WiringStatement::final_check`, as forms.
    let w = ctx.wiring;
    let n = w.log_rows;
    let r: Vec<Fr> = ctx.challenges.iter().rev().copied().collect();
    let (tau_hi, tau_lo) = ctx.tau_wiring.split_at(n - LOG_CELL);
    let (r_hi, r_lo) = r.split_at(n - LOG_CELL);
    let eq_tau_lo = EqPolynomial::<Fr>::evals(tau_lo, None);
    let eq_r_lo = EqPolynomial::<Fr>::evals(r_lo, None);
    let same_cell = eq_points(tau_hi, r_hi);
    let previous_cell = EqPlusOnePolynomial::new(r_hi.to_vec()).evaluate(tau_hi);
    let next_cell = EqPlusOnePolynomial::new(tau_hi.to_vec()).evaluate(r_hi);
    let eq_full = same_cell
        * eq_tau_lo
            .iter()
            .zip(&eq_r_lo)
            .fold(Fr::zero(), |acc, (a, b)| acc + *a * *b);
    let r_first_cell = r_hi.iter().fold(Fr::one(), |acc, a| acc * (Fr::one() - *a));
    let scale = ctx.rho_wiring * eq_full;
    for k in 0..WIRED_BITS {
        linear.add(WIRED_BIT_BASE + k, scale * w.gammas[k]);
    }
    for word in WiredWord::ALL {
        linear.add(
            wired_word_id(word),
            scale * w.gammas[WordSlot::Word(word).gamma_index()],
        );
    }
    let gamma_lo = w.gammas[WIRED_BITS + WIRED_WORDS];
    let gamma_hi = w.gammas[WIRED_BITS + WIRED_WORDS + 1];
    let mut is_lo = AffineForm::column(vk_id(VkColumn::LoIsConst));
    let mut is_hi = AffineForm::column(vk_id(VkColumn::HiIsConst));
    linear.add(vk_id(VkColumn::LoConst), -scale * gamma_lo);
    linear.add(vk_id(VkColumn::HiConst), -scale * gamma_hi);
    for (word, high, value) in ctx.public.tail_halves() {
        let weight = r_first_cell * eq_r_lo[word];
        let (is, gamma) = if high {
            (&mut is_hi, gamma_hi)
        } else {
            (&mut is_lo, gamma_lo)
        };
        is.constant += weight;
        linear.constant -= scale * gamma * weight * Fr::from_u64(u64::from(value));
    }
    let half = |from: usize| {
        let mut form = AffineForm::default();
        for k in 0..WORD_BITS / 2 {
            form.add(MESSAGE + from + k, Fr::one().mul_pow_2(k));
        }
        form
    };
    terms.push(Term {
        coefficient: scale * gamma_lo,
        factors: vec![is_lo, half(0)],
    });
    terms.push(Term {
        coefficient: scale * gamma_hi,
        factors: vec![is_hi, half(WORD_BITS / 2)],
    });
    // Sources: kernel weights summed per (slot, group, weights), then expanded
    // once per distinct triple.
    let mut kernels: Vec<((WordSlot, WordColumn, Weights), Fr)> = Vec::new();
    for p in 0..CELL_ROWS {
        for slot in WordSlot::all() {
            let (key, weight) = match source(p, slot) {
                Source::Cell {
                    group,
                    weights,
                    delta,
                } => (
                    (slot, group, weights),
                    same_cell * eq_tau_lo[p] * eq_r_lo[(p as isize - isize::from(delta)) as usize],
                ),
                Source::Previous {
                    group,
                    weights,
                    position,
                } => (
                    (slot, group, weights),
                    previous_cell * eq_tau_lo[p] * eq_r_lo[usize::from(position)],
                ),
                Source::Next {
                    group,
                    weights,
                    position,
                } => (
                    (slot, group, weights),
                    next_cell * eq_tau_lo[p] * eq_r_lo[usize::from(position)],
                ),
                Source::Zero | Source::Const(_) => continue,
            };
            match kernels.iter_mut().find(|(k, _)| *k == key) {
                Some((_, acc)) => *acc += weight,
                None => kernels.push((key, weight)),
            }
        }
    }
    for ((slot, group, weights), kernel) in kernels {
        let base = group.base();
        for j in 0..WORD_BITS {
            if let Some(k) = weights.coefficient(j) {
                linear.add(base + j, -ctx.rho_wiring * kernel * w.slot_weight(slot, k));
            }
        }
    }
    terms.push(Term {
        coefficient: Fr::one(),
        factors: vec![linear],
    });
    terms
}

/// The verifier's kernel work: distinct kernels `(position, source row
/// offset, cell offset)`, the number of `(position, slot)` copy entries, and
/// the distinct `(slot, group, weights)` value forms the kernel weights are
/// summed into before the 32-bit expansion.
pub fn kernel_counts() -> (usize, usize, usize) {
    let mut kernels: Vec<(usize, isize, i8)> = Vec::new();
    let mut forms: Vec<(WordSlot, WordColumn, Weights)> = Vec::new();
    let mut entries = 0;
    for p in 0..CELL_ROWS {
        for slot in WordSlot::all() {
            let (key, form) = match source(p, slot) {
                Source::Cell {
                    group,
                    weights,
                    delta,
                } => ((p, isize::from(delta), 0), (slot, group, weights)),
                Source::Previous {
                    group,
                    weights,
                    position,
                } => ((p, isize::from(position), -1), (slot, group, weights)),
                Source::Next {
                    group,
                    weights,
                    position,
                } => ((p, isize::from(position), 1), (slot, group, weights)),
                Source::Zero | Source::Const(_) => continue,
            };
            entries += 1;
            if !kernels.contains(&key) {
                kernels.push(key);
            }
            if !forms.contains(&form) {
                forms.push(form);
            }
        }
    }
    (kernels.len(), entries, forms.len())
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

/// The absorbed field element (32 bytes big-endian) starting at a wire row:
/// `Σ_w 2^{8(28 − 4w)} · bswap(m(row + w))` with `bswap(m(row))` over the
/// row's `m` bits and the rest the wired `fr_next` words.
pub fn fr_word() -> AffineForm {
    let mut form = AffineForm::default();
    for j in 0..WORD_BITS {
        form.add(
            MESSAGE + j,
            Fr::one().mul_pow_2(224 + 8 * (3 - j / 8) + j % 8),
        );
    }
    for i in 1..8u8 {
        form.add(
            wired_word_id(WiredWord::FrNext(i)),
            Fr::one().mul_pow_2(8 * (28 - 4 * usize::from(i))),
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
        form.add(
            MESSAGE + j,
            Fr::one().mul_pow_2(240 + 8 * (3 - j / 8) + j % 8),
        );
    }
    for i in 1..8u8 {
        form.add(
            wired_word_id(WiredWord::FrNext(i)),
            Fr::one().mul_pow_2(8 * (30 - 4 * usize::from(i))),
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
        for (source, row, byte) in schedule.byte_links() {
            match source {
                ByteSource::Wire { index, byte: 0 } if byte == 0 => map.wires.push((index, row)),
                ByteSource::Wire { index, byte: 0 } if byte == 2 => {
                    map.wires_shifted.push((index, row));
                }
                ByteSource::Wire { .. } | ByteSource::Padding => {}
                ByteSource::Constant(_) => {}
                ByteSource::Public { .. } | ByteSource::Element { .. } => {
                    map.bytes.push((source, row, byte));
                }
            }
        }
        map
    }
}
