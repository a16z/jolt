//! The row relation and its stage-A member. Every row `x` satisfies
//! `Σ_s X_s·Y_s − z(ξ) − k(ξ)·q(ξ) − (B − ξ)·C(ξ) = 0` over `Fr` (the limb
//! polynomial identity at the challenge `ξ`, exact over the integers by the
//! chunk range checks), the public pins, the grouped LogUp range checks, the
//! digit-bit constraints and the operand-lookup constraints; the linear
//! copy, constancy and LogUp-sum identities ride along as `Σ_x (...) = 0`
//! terms without an `eq` factor. The prover folds a row-major matrix; the
//! verifier consumes the same relation as [`Term`]s over the column
//! evaluations at the stage point.

use jolt_field::{Fr, One, Ring, Zero};

use super::columns::{
    recompose, Constants, CANON_CHUNKS, CANON_SHIFT, CARRIES, CARRY_CHUNKS, CHUNK_BITS,
    CHUNK_COLUMNS, DIGIT_COLUMNS, GROUP_SIZE, HELPER_COLUMNS, K_CHUNKS, LIMBS, Q_HI, RANGE_COLUMNS,
    Z_CHUNKS,
};
use super::layout::LOG_ROWS;
use super::terms::{plain, powers_with, AffineForm, ColumnId, Mul, Term};

pub use super::row_sumcheck::{eq_tau_column, RowSumcheck};

/// Slots per row (the dense `Fq12` product form).
pub const SLOTS: usize = 22;

/// Column indices of the row member's matrix, in packing order: the
/// committed phases 1b, 2a, 2b, 2c (each committed before the challenges the
/// next one depends on, see [`super::export::phases`]), then the
/// VK-committed public columns.
pub struct Col;

impl Col {
    pub const CHUNKS: usize = 0;
    pub const DIGITS: usize = Self::CHUNKS + CHUNK_COLUMNS;
    pub const ZERO: usize = Self::DIGITS;
    pub const NEG: usize = Self::DIGITS + 1;
    pub const E0: usize = Self::DIGITS + 2;
    /// The digit value `d = (1 − zero)(1 − 2neg)(1 + e)` as a column.
    pub const D: usize = Self::DIGITS + DIGIT_COLUMNS;
    pub const M_POS: usize = Self::D + 1;
    pub const M_NEG: usize = Self::M_POS + 1;
    pub const MULT: usize = Self::M_NEG + 1;
    /// Sign flags `[y > −y]` of the byte-linked points, on their sign rows.
    pub const FLAG: usize = Self::MULT + 1;
    /// Phase 1b: every prover-chosen value column, committed before `ξ, α`.
    pub const PHASE_1B_END: usize = Self::FLAG + 1;

    pub const X: usize = Self::PHASE_1B_END;
    pub const Y: usize = Self::X + SLOTS;
    pub const HELPERS: usize = Self::Y + SLOTS;
    pub const INV: usize = Self::HELPERS + HELPER_COLUMNS;
    /// Phase 2a: functions of `ξ, α` and phase-1b values, committed before `fp_root`.
    pub const PHASE_2A_END: usize = Self::INV + 1;
    pub const F_POS: usize = Self::PHASE_2A_END;
    pub const F_NEG: usize = Self::F_POS + 1;
    /// Phase 2b: the table fingerprints (`fp_root`), committed before `β, fp_combine, copy_root`.
    pub const PHASE_2B_END: usize = Self::F_NEG + 1;
    pub const H: usize = Self::PHASE_2B_END;
    pub const G_POS: usize = Self::H + 1;
    pub const G_NEG: usize = Self::G_POS + 1;
    /// Phase 2c: the lookup helpers (`β, fp_combine`), committed before `τ, γ, λ, …`.
    pub const PHASE_2C_END: usize = Self::G_NEG + 1;
    pub const COMMITTED: usize = Self::PHASE_2C_END;

    pub const PIN: usize = Self::COMMITTED;
    pub const PIN_LIMBS: usize = Self::PIN + 1;
    /// Rows exempt from the limb identity (inputs, public constants).
    pub const FREE: usize = Self::PIN_LIMBS + LIMBS;
    /// Rows whose limb identity is exact (`k = 0`): sign gadgets.
    pub const EXACT: usize = Self::FREE + 1;
    pub const VK_END: usize = Self::EXACT + 1;
    /// Every column with an evaluation claim at the stage point.
    pub const CLAIMED: usize = Self::VK_END;

    /// Prover-only public columns appended to the matrix.
    pub const EQ_TAU: usize = Self::CLAIMED;
    pub const COPY_KERNEL: usize = Self::EQ_TAU + 1;
    pub const SEL: usize = Self::COPY_KERNEL + 1;
    pub const IS_GT: usize = Self::SEL + 1;
    pub const IS_G1: usize = Self::IS_GT + 1;
    pub const IS_G2: usize = Self::IS_G1 + 1;
    pub const S0: usize = Self::IS_G2 + 1;
    pub const COORD: usize = Self::S0 + 1;
    pub const CONSTANCY: usize = Self::COORD + 1;
    pub const SMALL: usize = Self::CONSTANCY + 1;
    pub const ID: usize = Self::SMALL + 1;
    pub const WIDTH: usize = Self::ID + 1;
    pub const PUBLIC: usize = Self::WIDTH - Self::CLAIMED;
}

/// Verifier-evaluated public multilinears at the stage point, in the order of
/// the prover-only matrix columns (`Col::EQ_TAU..Col::WIDTH`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicEvals {
    pub eq_tau: Fr,
    pub copy_kernel: Fr,
    pub sel: Fr,
    pub is_gt: Fr,
    pub is_g1: Fr,
    pub is_g2: Fr,
    pub s0: Fr,
    /// `Σ_x eq(r,x)·sel(x)·c(x)`: the own coordinate of the selected rows.
    pub coord: Fr,
    pub constancy: Fr,
    pub small: Fr,
    pub id: Fr,
}

impl PublicEvals {
    pub(super) fn as_array(&self) -> [Fr; Col::PUBLIC] {
        [
            self.eq_tau,
            self.copy_kernel,
            self.sel,
            self.is_gt,
            self.is_g1,
            self.is_g2,
            self.s0,
            self.coord,
            self.constancy,
            self.small,
            self.id,
        ]
    }
}

/// The five quantities every term of the final relation is linear in
/// (`eq(τ, r)`, `λ`, `λ_lookup`, the copy kernel, the constancy kernel): a
/// member's batching coefficient scales them once instead of every term.
struct TermScale {
    eq: Fr,
    lambda: Fr,
    lambda_lookup: Fr,
    copy_kernel: Fr,
    constancy: Fr,
}

/// Transcript challenges of the row member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Challenges {
    /// Big-endian row point (`tau[0]` the most significant bit, bound first).
    pub tau: Vec<Fr>,
    /// Limb-polynomial evaluation point.
    pub xi: Fr,
    /// LogUp range-check challenge.
    pub alpha: Fr,
    /// Batching root of the row constraints.
    pub gamma: Fr,
    /// Weight of the LogUp range sum.
    pub lambda: Fr,
    /// Operand lookup: key offset, fingerprint root, key/fingerprint combiner,
    /// sum weight.
    pub beta: Fr,
    pub fp_root: Fr,
    pub fp_combine: Fr,
    pub lambda_lookup: Fr,
    /// Copy-member weight root (per operand column and fingerprint column)
    /// and the digit constancy weight root.
    pub copy_root: Fr,
    pub constancy_root: Fr,
}

/// Row index of the `one` element cell (coordinate 0), the lookup key of the
/// GT identity digit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupConstants {
    pub one_row: u32,
}

/// Batching order of the row constraints: limb identity, pin, range groups,
/// digit booleanity (5), digit range, lookup read, lookup table (2), inverse
/// table.
const GAMMA_LIMB: usize = 0;
const GAMMA_PIN: usize = 1;
const GAMMA_RANGE: usize = 2;
const GAMMA_BOOL: usize = GAMMA_RANGE + HELPER_COLUMNS;
const GAMMA_DIGIT_RANGE: usize = GAMMA_BOOL + DIGIT_COLUMNS;
const GAMMA_DIGIT_VALUE: usize = GAMMA_DIGIT_RANGE + 1;
const GAMMA_READ: usize = GAMMA_DIGIT_VALUE + 1;
const GAMMA_TABLE: usize = GAMMA_READ + 1;
const GAMMA_INV: usize = GAMMA_TABLE + 2;
/// The range table lives on the first `2^16` rows: multiplicity and inverse
/// vanish elsewhere, so no prover value can extend the table.
const GAMMA_TABLE_GATE: usize = GAMMA_INV + 1;
/// Byte-linked inputs (free rows that are not pinned constants) are canonical:
/// `d + z_hi = q_hi − 1` with `d` the four low quotient chunks.
const GAMMA_CANON: usize = GAMMA_TABLE_GATE + 2;
/// Sign gadgets: the flag is a bit and exact rows have `k = 0`.
const GAMMA_FLAG_BOOL: usize = GAMMA_CANON + 1;
const GAMMA_EXACT: usize = GAMMA_FLAG_BOOL + 1;
pub const GAMMA_TERMS: usize = GAMMA_EXACT + 1;

/// Fingerprinted operand slots per selected op kind.
pub const FP_SLOTS_GT: usize = SLOTS;
pub const FP_SLOTS_G1: usize = 2;
pub const FP_SLOTS_G2: usize = 4;
/// Lookup key offset of conjugated (negative-digit) GT reads.
const NEG_KEY_OFFSET: u64 = 1 << LOG_ROWS;

/// Public parameters of the row member.
pub struct RowRelation {
    pub challenges: Challenges,
    pub lookup: LookupConstants,
    pub constants: Constants,
    gammas: Vec<Fr>,
    xi_pow: [Fr; LIMBS],
    q_xi: Fr,
    /// `2^256` in limb form: `2^64·ξ²` (the sign rows' flag term).
    flag_xi: Fr,
    fp_pow: Vec<Fr>,
    copy_pow: Vec<Fr>,
    constancy_pow: Vec<Fr>,
    /// `Z(ξ)`, `k(ξ)`, `C(ξ)` over the chunk columns.
    z_xi_form: AffineForm,
    k_xi_form: AffineForm,
    c_xi_form: AffineForm,
}

impl RowRelation {
    pub fn new(challenges: Challenges, lookup: LookupConstants) -> Self {
        Self::new_with(challenges, lookup, &mut plain)
    }

    /// [`Self::new`] with every challenge-derived product routed through
    /// `mul` (the verifier's statement derivation).
    pub fn new_with(challenges: Challenges, lookup: LookupConstants, mul: Mul<'_>) -> Self {
        assert_eq!(challenges.tau.len(), LOG_ROWS);
        let constants = Constants::new();
        let xi = challenges.xi;
        let xi_pow = [Fr::one(), xi, mul(xi, xi)];
        let q_xi = xi_pow
            .iter()
            .zip(&constants.q_limbs)
            .fold(Fr::zero(), |acc, (p, q)| acc + mul(*p, *q));
        let flag_xi = mul(Fr::pow2(64), xi_pow[2]);
        let chunk_form = |start: usize, count: usize| {
            let mut form = AffineForm::default();
            for j in 0..count {
                form.add_column(
                    ColumnId((Col::CHUNKS + start + j) as u32),
                    constants.pow_chunk[j],
                );
            }
            form
        };
        let z_xi_form = chunk_form(0, 6)
            .plus(&chunk_form(6, 6).scale_with(xi_pow[1], mul))
            .plus(&chunk_form(12, Z_CHUNKS - 12).scale_with(xi_pow[2], mul));
        let top = chunk_form(Z_CHUNKS + 12, K_CHUNKS - 12)
            .plus(&AffineForm::constant(-constants.k_offset_top_limb));
        let k_xi_form = chunk_form(Z_CHUNKS, 6)
            .plus(&chunk_form(Z_CHUNKS + 6, 6).scale_with(xi_pow[1], mul))
            .plus(&top.scale_with(xi_pow[2], mul));
        let mut c_xi_form = AffineForm::default();
        let mut power = Fr::one();
        for i in 0..CARRIES {
            let start = Z_CHUNKS + K_CHUNKS + CARRY_CHUNKS * i;
            let carry = chunk_form(start, CARRY_CHUNKS)
                .plus(&AffineForm::constant(-constants.carry_offset));
            c_xi_form.accumulate(&if i == 0 {
                carry
            } else {
                carry.scale_with(power, mul)
            });
            if i + 1 < CARRIES {
                power = mul(power, xi);
            }
        }
        Self {
            gammas: powers_with(challenges.gamma, GAMMA_TERMS, mul),
            flag_xi,
            xi_pow,
            q_xi,
            fp_pow: powers_with(challenges.fp_root, SLOTS, mul),
            // Operand columns then the two fingerprint columns.
            copy_pow: powers_with(challenges.copy_root, 2 * SLOTS + 2, mul),
            constancy_pow: powers_with(challenges.constancy_root, DIGIT_COLUMNS, mul),
            challenges,
            lookup,
            constants,
            z_xi_form,
            k_xi_form,
            c_xi_form,
        }
    }

    /// Round-polynomial degree: `eq · h · Π_3(α − c)`.
    pub const fn degree() -> usize {
        GROUP_SIZE + 2
    }

    /// Copy weight of operand column `X_s` / `Y_s` (`index < 2·SLOTS`) or of
    /// the fingerprint columns (`2·SLOTS`, `2·SLOTS + 1`).
    pub fn copy_weight(&self, index: usize) -> Fr {
        self.copy_pow[index]
    }

    pub fn constancy_weight(&self, bit: usize) -> Fr {
        self.constancy_pow[bit]
    }

    /// `z(ξ)` from the sixteen `z` chunks (each limb is six chunks; the top
    /// limb four).
    fn z_xi(&self, chunks: &[Fr]) -> Fr {
        let c = &self.constants;
        let limb_of = |start: usize, count: usize| recompose(&chunks[start..start + count], c);
        limb_of(0, 6) * self.xi_pow[0]
            + limb_of(6, 6) * self.xi_pow[1]
            + limb_of(12, Z_CHUNKS - 12) * self.xi_pow[2]
    }

    /// `k(ξ)`: the offset quotient's limbs with `2^75` removed from the top limb.
    fn k_xi(&self, chunks: &[Fr]) -> Fr {
        let c = &self.constants;
        let limb_of = |start: usize, count: usize| recompose(&chunks[start..start + count], c);
        let k = &chunks[Z_CHUNKS..Z_CHUNKS + K_CHUNKS];
        (limb_of(Z_CHUNKS, 6) + limb_of(Z_CHUNKS + 6, 6) * self.xi_pow[1])
            + (recompose(&k[12..], c) - c.k_offset_top_limb) * self.xi_pow[2]
    }

    /// `C(ξ) = Σ_i ξ^i·(c'_i − 2^111)`.
    fn c_xi(&self, chunks: &[Fr]) -> Fr {
        let c = &self.constants;
        let mut power = Fr::one();
        let mut sum = Fr::zero();
        for i in 0..CARRIES {
            let start = Z_CHUNKS + K_CHUNKS + CARRY_CHUNKS * i;
            sum += (recompose(&chunks[start..start + CARRY_CHUNKS], c) - c.carry_offset) * power;
            power *= self.challenges.xi;
        }
        sum
    }

    /// The `eq`-weighted part `Φ(row)` of the summand.
    fn phi(&self, v: &[Fr]) -> Fr {
        let ch = &self.challenges;
        let c = &self.constants;
        let chunks = &v[Col::CHUNKS..Col::CHUNKS + CHUNK_COLUMNS];
        let g = &self.gammas;
        // Limb identity.
        let mut products = Fr::zero();
        for s in 0..SLOTS {
            products += v[Col::X + s] * v[Col::Y + s];
        }
        let z_xi = self.z_xi(chunks);
        let k_xi = self.k_xi(chunks);
        let limb = products + v[Col::EXACT] * (Fr::one() - v[Col::FLAG]) * self.flag_xi
            - z_xi
            - k_xi * self.q_xi
            - (c.pow_limb - ch.xi) * self.c_xi(chunks);
        let mut phi = g[GAMMA_LIMB] * (Fr::one() - v[Col::FREE]) * limb;
        // Sign gadgets: a bit flag, exact rows without a quotient.
        phi += g[GAMMA_FLAG_BOOL] * v[Col::FLAG] * (v[Col::FLAG] - Fr::one());
        phi += g[GAMMA_EXACT] * v[Col::EXACT] * k_xi;
        // Pins.
        let pin_xi = (0..LIMBS).fold(Fr::zero(), |acc, a| {
            acc + v[Col::PIN_LIMBS + a] * self.xi_pow[a]
        });
        phi += g[GAMMA_PIN] * v[Col::PIN] * (z_xi - pin_xi);
        // Range groups.
        for grp in 0..HELPER_COLUMNS {
            let product = (0..GROUP_SIZE).fold(Fr::one(), |acc, i| {
                acc * (ch.alpha - Self::range_value(v, GROUP_SIZE * grp + i))
            });
            phi += g[GAMMA_RANGE + grp] * (v[Col::HELPERS + grp] * product - Fr::one());
        }
        // Digit bits.
        for b in 0..DIGIT_COLUMNS {
            let bit = v[Col::DIGITS + b];
            phi += g[GAMMA_BOOL + b] * bit * (bit - Fr::one());
        }
        phi += g[GAMMA_DIGIT_RANGE]
            * (Fr::one() - v[Col::NEG])
            * v[Col::E0]
            * v[Col::E0 + 1]
            * v[Col::E0 + 2];
        let (d, _) = Self::digit(v);
        phi += g[GAMMA_DIGIT_VALUE] * (v[Col::D] - v[Col::SEL] * d);
        // Operand lookup, reading side.
        let (key, fingerprint) = self.read_key(v);
        phi += g[GAMMA_READ]
            * (v[Col::H] * (ch.beta + key + ch.fp_combine * fingerprint) - v[Col::SEL]);
        // Operand lookup, table side.
        for (i, (gc, fc, mc)) in [
            (Col::G_POS, Col::F_POS, Col::M_POS),
            (Col::G_NEG, Col::F_NEG, Col::M_NEG),
        ]
        .into_iter()
        .enumerate()
        {
            let offset = if i == 1 {
                Fr::from_u64(NEG_KEY_OFFSET)
            } else {
                Fr::zero()
            };
            phi += g[GAMMA_TABLE + i]
                * (v[gc] * (ch.beta + v[Col::ID] + offset + ch.fp_combine * v[fc]) - v[mc]);
        }
        // Range inverse table, gated to its `2^16` rows.
        phi += g[GAMMA_INV] * v[Col::SMALL] * (v[Col::INV] * (ch.alpha - v[Col::ID]) - Fr::one());
        let outside = Fr::one() - v[Col::SMALL];
        phi += g[GAMMA_TABLE_GATE] * outside * v[Col::MULT];
        phi += g[GAMMA_TABLE_GATE + 1] * outside * v[Col::INV];
        phi +=
            g[GAMMA_CANON] * v[Col::FREE] * (Fr::one() - v[Col::PIN]) * Self::canonicality(chunks);
        phi
    }

    /// `d + z_hi − (q_hi − 1)` over a row's chunks (zero on canonical free rows).
    fn canonicality(chunks: &[Fr]) -> Fr {
        let mut sum = -Fr::from_u64(Q_HI - 1);
        for i in 0..CANON_CHUNKS {
            let weight = Fr::from_u64(1u64 << (CHUNK_BITS * i));
            sum += weight * (chunks[Z_CHUNKS + i] + chunks[CANON_SHIFT / CHUNK_BITS + i]);
        }
        sum
    }

    fn range_value(v: &[Fr], i: usize) -> Fr {
        if i < CHUNK_COLUMNS {
            v[Col::CHUNKS + i]
        } else {
            v[Col::DIGITS + i - CHUNK_COLUMNS]
        }
    }

    /// Digit value `d = (1 − zero)(1 − 2·neg)(1 + e)` from the bits, and the
    /// entry magnitude `e`.
    fn digit(v: &[Fr]) -> (Fr, Fr) {
        let two = Fr::from_u64(2);
        let e = v[Col::E0] + two * v[Col::E0 + 1] + two * two * v[Col::E0 + 2];
        let d = (Fr::one() - v[Col::ZERO]) * (Fr::one() - two * v[Col::NEG]) * (Fr::one() + e);
        (d, e)
    }

    /// Lookup key and fingerprint of a reading row:
    /// GT: `key = (1−zero)(S0 + 16e) + zero·(one_row + c) + 2^18·neg`,
    /// EC: `key = S0 + stride·d` (`16` for G1 cells, `8` for G2 half cells);
    /// `fingerprint = Σ_{s<n} fp^s·Y_s`.
    fn read_key(&self, v: &[Fr]) -> (Fr, Fr) {
        let (_, e) = Self::digit(v);
        let d = v[Col::D];
        let sixteen = Fr::from_u64(16);
        let one_row = Fr::from_u64(u64::from(self.lookup.one_row));
        let gt_key = (Fr::one() - v[Col::ZERO]) * (v[Col::S0] + sixteen * e)
            + v[Col::ZERO] * (one_row + v[Col::COORD])
            + Fr::from_u64(NEG_KEY_OFFSET) * v[Col::NEG];
        // `stride` vanishes on GT rows, so it needs no `(1 − is_gt)` factor.
        let stride = sixteen * v[Col::IS_G1] + Fr::from_u64(8) * v[Col::IS_G2];
        let key = v[Col::IS_GT] * gt_key + (Fr::one() - v[Col::IS_GT]) * v[Col::S0] + stride * d;
        let fp = |n: usize| (0..n).fold(Fr::zero(), |acc, s| acc + self.fp_pow[s] * v[Col::Y + s]);
        let fingerprint = v[Col::IS_GT] * fp(FP_SLOTS_GT)
            + v[Col::IS_G1] * fp(FP_SLOTS_G1)
            + v[Col::IS_G2] * fp(FP_SLOTS_G2);
        (key, fingerprint)
    }

    /// The `Σ_x`-only part: LogUp sums, copy identities, digit constancy.
    fn linear(&self, v: &[Fr]) -> Fr {
        let ch = &self.challenges;
        // Range LogUp: Σ_g h_g·e_{2,g} − mult·inv.
        let mut logup = Fr::zero();
        for grp in 0..HELPER_COLUMNS {
            let f: [Fr; GROUP_SIZE] =
                std::array::from_fn(|i| ch.alpha - Self::range_value(v, GROUP_SIZE * grp + i));
            let elementary = f[1] * f[2] + f[0] * f[2] + f[0] * f[1];
            logup += v[Col::HELPERS + grp] * elementary;
        }
        logup -= v[Col::MULT] * v[Col::INV];
        // Operand lookup sum.
        let lookup = v[Col::H] - v[Col::G_POS] - v[Col::G_NEG];
        // Copy identities: eq(τ,x)·Σ β_i·col_i(x) − B(x)·Z_ξ(x); looked-up
        // `Y_s` (the fingerprinted slots of selected rows) are not copies.
        let mut copied = Fr::zero();
        for s in 0..SLOTS {
            copied += self.copy_pow[s] * v[Col::X + s];
            copied += self.copy_pow[SLOTS + s] * Self::copy_mask(v, s) * v[Col::Y + s];
        }
        copied +=
            self.copy_pow[2 * SLOTS] * v[Col::F_POS] + self.copy_pow[2 * SLOTS + 1] * v[Col::F_NEG];
        let z_xi = self.z_xi(&v[Col::CHUNKS..Col::CHUNKS + CHUNK_COLUMNS]);
        let copy = v[Col::EQ_TAU] * copied - v[Col::COPY_KERNEL] * z_xi;
        // Digit constancy: W(x)·Σ_b β'_b·bit_b(x).
        let bits = (0..DIGIT_COLUMNS).fold(Fr::zero(), |acc, b| {
            acc + self.constancy_pow[b] * v[Col::DIGITS + b]
        });
        ch.lambda * logup + ch.lambda_lookup * lookup + copy + v[Col::CONSTANCY] * bits
    }

    /// `1 − Σ_kind is_kind(x)·[s < fp_slots(kind)]`: whether `Y_s` is a copy.
    fn copy_mask(v: &[Fr], s: usize) -> Fr {
        let mut mask = Fr::one() - v[Col::IS_GT];
        if s < FP_SLOTS_G1 {
            mask -= v[Col::IS_G1];
        }
        if s < FP_SLOTS_G2 {
            mask -= v[Col::IS_G2];
        }
        mask
    }

    /// The summand at one (possibly extrapolated) row of the matrix.
    pub fn summand(&self, v: &[Fr]) -> Fr {
        v[Col::EQ_TAU] * self.phi(v) + self.linear(v)
    }

    /// Every row-local constraint's unweighted value at a row (each is zero
    /// on an honest witness), for diagnostics and the per-constraint test.
    pub fn constraint_values(&self, v: &[Fr]) -> Vec<(&'static str, Fr)> {
        let ch = &self.challenges;
        let c = &self.constants;
        let chunks = &v[Col::CHUNKS..Col::CHUNKS + CHUNK_COLUMNS];
        let mut out = Vec::new();
        let products = (0..SLOTS).fold(Fr::zero(), |acc, s| acc + v[Col::X + s] * v[Col::Y + s]);
        let z_xi = self.z_xi(chunks);
        out.push((
            "limb",
            (Fr::one() - v[Col::FREE])
                * (products + v[Col::EXACT] * (Fr::one() - v[Col::FLAG]) * self.flag_xi
                    - z_xi
                    - self.k_xi(chunks) * self.q_xi
                    - (c.pow_limb - ch.xi) * self.c_xi(chunks)),
        ));
        out.push(("flag_bool", v[Col::FLAG] * (v[Col::FLAG] - Fr::one())));
        out.push(("exact_quotient", v[Col::EXACT] * self.k_xi(chunks)));
        let pin_xi = (0..LIMBS).fold(Fr::zero(), |acc, a| {
            acc + v[Col::PIN_LIMBS + a] * self.xi_pow[a]
        });
        out.push(("pin", v[Col::PIN] * (z_xi - pin_xi)));
        let mut range = Fr::zero();
        for grp in 0..HELPER_COLUMNS {
            let product = (0..GROUP_SIZE).fold(Fr::one(), |acc, i| {
                acc * (ch.alpha - Self::range_value(v, GROUP_SIZE * grp + i))
            });
            range += v[Col::HELPERS + grp] * product - Fr::one();
        }
        out.push(("range", range));
        let mut bools = Fr::zero();
        for b in 0..DIGIT_COLUMNS {
            let bit = v[Col::DIGITS + b];
            bools += bit * (bit - Fr::one());
        }
        out.push(("digit_bool", bools));
        out.push((
            "digit_range",
            (Fr::one() - v[Col::NEG]) * v[Col::E0] * v[Col::E0 + 1] * v[Col::E0 + 2],
        ));
        out.push(("digit_value", v[Col::D] - v[Col::SEL] * Self::digit(v).0));
        let (key, fingerprint) = self.read_key(v);
        out.push((
            "lookup_read",
            v[Col::H] * (ch.beta + key + ch.fp_combine * fingerprint) - v[Col::SEL],
        ));
        for (name, gc, fc, mc, offset) in [
            (
                "lookup_table_pos",
                Col::G_POS,
                Col::F_POS,
                Col::M_POS,
                Fr::zero(),
            ),
            (
                "lookup_table_neg",
                Col::G_NEG,
                Col::F_NEG,
                Col::M_NEG,
                Fr::from_u64(NEG_KEY_OFFSET),
            ),
        ] {
            out.push((
                name,
                v[gc] * (ch.beta + v[Col::ID] + offset + ch.fp_combine * v[fc]) - v[mc],
            ));
        }
        out.push((
            "inverse_table",
            v[Col::SMALL] * (v[Col::INV] * (ch.alpha - v[Col::ID]) - Fr::one()),
        ));
        let outside = Fr::one() - v[Col::SMALL];
        out.push(("range_mult_gate", outside * v[Col::MULT]));
        out.push(("range_inv_gate", outside * v[Col::INV]));
        out.push((
            "canonicality",
            v[Col::FREE] * (Fr::one() - v[Col::PIN]) * Self::canonicality(chunks),
        ));
        out
    }

    /// The `Σ_x`-only identities' per-row summands (each sums to zero over
    /// the rows of an honest witness): range LogUp, lookup LogUp, copies,
    /// constancy.
    pub fn linear_values(&self, v: &[Fr]) -> Vec<(&'static str, Fr)> {
        let ch = &self.challenges;
        let mut logup = Fr::zero();
        for grp in 0..HELPER_COLUMNS {
            let f: [Fr; GROUP_SIZE] =
                std::array::from_fn(|i| ch.alpha - Self::range_value(v, GROUP_SIZE * grp + i));
            logup += v[Col::HELPERS + grp] * (f[1] * f[2] + f[0] * f[2] + f[0] * f[1]);
        }
        logup -= v[Col::MULT] * v[Col::INV];
        let lookup = v[Col::H] - v[Col::G_POS] - v[Col::G_NEG];
        let mut copied = Fr::zero();
        for s in 0..SLOTS {
            copied += self.copy_pow[s] * v[Col::X + s];
            copied += self.copy_pow[SLOTS + s] * Self::copy_mask(v, s) * v[Col::Y + s];
        }
        copied +=
            self.copy_pow[2 * SLOTS] * v[Col::F_POS] + self.copy_pow[2 * SLOTS + 1] * v[Col::F_NEG];
        let z_xi = self.z_xi(&v[Col::CHUNKS..Col::CHUNKS + CHUNK_COLUMNS]);
        let copy = v[Col::EQ_TAU] * copied - v[Col::COPY_KERNEL] * z_xi;
        let bits = (0..DIGIT_COLUMNS).fold(Fr::zero(), |acc, b| {
            acc + self.constancy_pow[b] * v[Col::DIGITS + b]
        });
        vec![
            ("range_logup", logup),
            ("lookup_logup", lookup),
            ("copy", copy),
            ("constancy", v[Col::CONSTANCY] * bits),
        ]
    }

    /// The member's final relation as terms over the claimed column
    /// evaluations (`Col::CLAIMED` columns), given the public multilinears
    /// at the stage point. `Σ_t term_t(v) == summand(v ∥ public)`.
    pub fn terms(&self, public: &PublicEvals) -> Vec<Term> {
        self.terms_with(public, &mut plain)
    }

    /// The final relation as terms, every constant product observed by `mul`.
    pub fn terms_with(&self, public: &PublicEvals, mul: Mul<'_>) -> Vec<Term> {
        let scale = TermScale {
            eq: public.eq_tau,
            lambda: self.challenges.lambda,
            lambda_lookup: self.challenges.lambda_lookup,
            copy_kernel: public.copy_kernel,
            constancy: public.constancy,
        };
        self.terms_scaled(public, &scale, mul)
    }

    /// [`Self::terms_with`] with every term multiplied by `batching` (the
    /// member's batching coefficient): five products instead of one per term.
    pub fn batched_terms(&self, public: &PublicEvals, batching: Fr, mul: Mul<'_>) -> Vec<Term> {
        let scale = TermScale {
            eq: mul(public.eq_tau, batching),
            lambda: mul(self.challenges.lambda, batching),
            lambda_lookup: mul(self.challenges.lambda_lookup, batching),
            copy_kernel: mul(public.copy_kernel, batching),
            constancy: mul(public.constancy, batching),
        };
        self.terms_scaled(public, &scale, mul)
    }

    fn terms_scaled(&self, public: &PublicEvals, scale: &TermScale, mul: Mul<'_>) -> Vec<Term> {
        let ch = &self.challenges;
        let c = &self.constants;
        let g = &self.gammas;
        let eq = scale.eq;
        let column = |i: usize| AffineForm::column(ColumnId(i as u32));
        let one_minus = |i: usize| {
            AffineForm::constant(Fr::one())
                .plus(&AffineForm::scaled(ColumnId(i as u32), -Fr::one()))
        };
        let mut terms = Vec::new();
        // Limb identity, on non-free rows.
        let bound = one_minus(Col::FREE);
        let g_limb = mul(eq, g[GAMMA_LIMB]);
        for s in 0..SLOTS {
            terms.push(Term::new(
                g_limb,
                vec![bound.clone(), column(Col::X + s), column(Col::Y + s)],
            ));
        }
        let z_xi = self.z_xi_form.clone();
        let mut limb_linear = -z_xi.clone();
        limb_linear.accumulate(&self.k_xi_form.clone().scale_with(-self.q_xi, mul));
        limb_linear.accumulate(
            &self
                .c_xi_form
                .clone()
                .scale_with(-(c.pow_limb - ch.xi), mul),
        );
        terms.push(Term::new(g_limb, vec![bound.clone(), limb_linear]));
        terms.push(Term::new(
            mul(g_limb, self.flag_xi),
            vec![bound, column(Col::EXACT), one_minus(Col::FLAG)],
        ));
        // Sign gadgets.
        terms.push(Term::new(
            mul(eq, g[GAMMA_FLAG_BOOL]),
            vec![
                column(Col::FLAG),
                column(Col::FLAG).plus(&AffineForm::constant(-Fr::one())),
            ],
        ));
        terms.push(Term::new(
            mul(eq, g[GAMMA_EXACT]),
            vec![column(Col::EXACT), self.k_xi_form.clone()],
        ));
        // Pins.
        let mut pin_xi = AffineForm::default();
        for a in 0..LIMBS {
            pin_xi.add_column(ColumnId((Col::PIN_LIMBS + a) as u32), -self.xi_pow[a]);
        }
        terms.push(Term::new(
            mul(eq, g[GAMMA_PIN]),
            vec![column(Col::PIN), z_xi.clone().plus(&pin_xi)],
        ));
        // Range groups and the LogUp sum.
        for grp in 0..HELPER_COLUMNS {
            let f: Vec<AffineForm> = (0..GROUP_SIZE)
                .map(|i| {
                    AffineForm::constant(ch.alpha).plus(&AffineForm::scaled(
                        Self::range_column(GROUP_SIZE * grp + i),
                        -Fr::one(),
                    ))
                })
                .collect();
            let g_range = mul(eq, g[GAMMA_RANGE + grp]);
            let mut factors = vec![column(Col::HELPERS + grp)];
            factors.extend(f.iter().cloned());
            terms.push(Term::new(g_range, factors));
            terms.push(Term::new(-g_range, vec![]));
            for i in 0..GROUP_SIZE {
                let mut factors = vec![column(Col::HELPERS + grp)];
                factors.extend(
                    f.iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, x)| x.clone()),
                );
                terms.push(Term::new(scale.lambda, factors));
            }
        }
        terms.push(Term::new(
            -scale.lambda,
            vec![column(Col::MULT), column(Col::INV)],
        ));
        // Digit bits.
        for b in 0..DIGIT_COLUMNS {
            terms.push(Term::new(
                mul(eq, g[GAMMA_BOOL + b]),
                vec![
                    column(Col::DIGITS + b),
                    column(Col::DIGITS + b).plus(&AffineForm::constant(-Fr::one())),
                ],
            ));
        }
        terms.push(Term::new(
            mul(eq, g[GAMMA_DIGIT_RANGE]),
            vec![
                one_minus(Col::NEG),
                column(Col::E0),
                column(Col::E0 + 1),
                column(Col::E0 + 2),
            ],
        ));
        let e = Self::e_form();
        let one_plus_e = AffineForm::constant(Fr::one()).plus(&e);
        let one_minus_2neg = AffineForm::constant(Fr::one()).plus(&AffineForm::scaled(
            ColumnId(Col::NEG as u32),
            -Fr::from_u64(2),
        ));
        let g_digit = mul(eq, g[GAMMA_DIGIT_VALUE]);
        terms.push(Term::new(g_digit, vec![column(Col::D)]));
        terms.push(Term::new(
            -mul(g_digit, public.sel),
            vec![one_minus(Col::ZERO), one_minus_2neg, one_plus_e],
        ));
        // Lookup, reading side: h·(β + key + fp_combine·F) − sel.
        let sixteen = Fr::from_u64(16);
        let one_row = Fr::from_u64(u64::from(self.lookup.one_row));
        let gr = mul(eq, g[GAMMA_READ]);
        let h = column(Col::H);
        terms.push(Term::new(mul(gr, ch.beta), vec![h.clone()]));
        // GT key.
        let gr_gt = mul(gr, public.is_gt);
        terms.push(Term::new(
            mul(gr_gt, public.s0),
            vec![h.clone(), one_minus(Col::ZERO)],
        ));
        terms.push(Term::new(
            mul(gr_gt, sixteen),
            vec![h.clone(), one_minus(Col::ZERO), e.clone()],
        ));
        terms.push(Term::new(
            mul(gr_gt, one_row + public.coord),
            vec![h.clone(), column(Col::ZERO)],
        ));
        terms.push(Term::new(
            mul(gr_gt, Fr::from_u64(NEG_KEY_OFFSET)),
            vec![h.clone(), column(Col::NEG)],
        ));
        // EC key.
        let gr_ec = mul(gr, Fr::one() - public.is_gt);
        terms.push(Term::new(mul(gr_ec, public.s0), vec![h.clone()]));
        let ec_scale = mul(sixteen, public.is_g1) + mul(Fr::from_u64(8), public.is_g2);
        terms.push(Term::new(
            mul(gr, ec_scale),
            vec![h.clone(), column(Col::D)],
        ));
        // Fingerprints.
        let gr_combine = mul(gr, ch.fp_combine);
        for (indicator, n) in [
            (public.is_gt, FP_SLOTS_GT),
            (public.is_g1, FP_SLOTS_G1),
            (public.is_g2, FP_SLOTS_G2),
        ] {
            let mut fp = AffineForm::default();
            for s in 0..n {
                fp.add_column(ColumnId((Col::Y + s) as u32), self.fp_pow[s]);
            }
            terms.push(Term::new(mul(gr_combine, indicator), vec![h.clone(), fp]));
        }
        terms.push(Term::new(-mul(gr, public.sel), vec![]));
        // Lookup, table side.
        for (i, (gc, fc, mc)) in [
            (Col::G_POS, Col::F_POS, Col::M_POS),
            (Col::G_NEG, Col::F_NEG, Col::M_NEG),
        ]
        .into_iter()
        .enumerate()
        {
            let offset = if i == 1 {
                Fr::from_u64(NEG_KEY_OFFSET)
            } else {
                Fr::zero()
            };
            let gt = mul(eq, g[GAMMA_TABLE + i]);
            terms.push(Term::new(
                mul(gt, ch.beta + public.id + offset),
                vec![column(gc)],
            ));
            terms.push(Term::new(
                mul(gt, ch.fp_combine),
                vec![column(gc), column(fc)],
            ));
            terms.push(Term::new(-gt, vec![column(mc)]));
        }
        // Range inverse table.
        let g_inv = mul(eq, g[GAMMA_INV]);
        let gi = mul(g_inv, public.small);
        terms.push(Term::new(
            mul(gi, ch.alpha - public.id),
            vec![column(Col::INV)],
        ));
        terms.push(Term::new(-gi, vec![]));
        let outside = mul(eq, Fr::one() - public.small);
        terms.push(Term::new(
            mul(outside, g[GAMMA_TABLE_GATE]),
            vec![column(Col::MULT)],
        ));
        terms.push(Term::new(
            mul(outside, g[GAMMA_TABLE_GATE + 1]),
            vec![column(Col::INV)],
        ));
        // Canonicality of free rows.
        let mut canon = AffineForm::constant(-Fr::from_u64(Q_HI - 1));
        for i in 0..CANON_CHUNKS {
            let weight = Fr::from_u64(1u64 << (CHUNK_BITS * i));
            canon.add_column(ColumnId((Col::CHUNKS + Z_CHUNKS + i) as u32), weight);
            canon.add_column(
                ColumnId((Col::CHUNKS + CANON_SHIFT / CHUNK_BITS + i) as u32),
                weight,
            );
        }
        terms.push(Term::new(
            mul(eq, g[GAMMA_CANON]),
            vec![column(Col::FREE), one_minus(Col::PIN), canon],
        ));
        // Lookup sum.
        terms.push(Term::new(scale.lambda_lookup, vec![h.clone()]));
        terms.push(Term::new(-scale.lambda_lookup, vec![column(Col::G_POS)]));
        terms.push(Term::new(-scale.lambda_lookup, vec![column(Col::G_NEG)]));
        // Copy identities (looked-up `Y_s` masked out).
        let mut copied = AffineForm::default();
        for s in 0..SLOTS {
            copied.add_column(ColumnId((Col::X + s) as u32), self.copy_pow[s]);
            let mut mask = Fr::one() - public.is_gt;
            if s < FP_SLOTS_G1 {
                mask -= public.is_g1;
            }
            if s < FP_SLOTS_G2 {
                mask -= public.is_g2;
            }
            copied.add_column(
                ColumnId((Col::Y + s) as u32),
                mul(self.copy_pow[SLOTS + s], mask),
            );
        }
        copied.add_column(ColumnId(Col::F_POS as u32), self.copy_pow[2 * SLOTS]);
        copied.add_column(ColumnId(Col::F_NEG as u32), self.copy_pow[2 * SLOTS + 1]);
        terms.push(Term::new(eq, vec![copied]));
        terms.push(Term::new(-scale.copy_kernel, vec![z_xi]));
        // Digit constancy.
        let mut bits = AffineForm::default();
        for b in 0..DIGIT_COLUMNS {
            bits.add_column(ColumnId((Col::DIGITS + b) as u32), self.constancy_pow[b]);
        }
        terms.push(Term::new(scale.constancy, vec![bits]));
        terms
    }

    /// Range-checked column `i`: the chunks, then the digit bits.
    fn range_column(i: usize) -> ColumnId {
        if i < CHUNK_COLUMNS {
            ColumnId((Col::CHUNKS + i) as u32)
        } else {
            ColumnId((Col::DIGITS + i - CHUNK_COLUMNS) as u32)
        }
    }

    fn e_form() -> AffineForm {
        let mut e = AffineForm::default();
        for (i, weight) in [1u64, 2, 4].into_iter().enumerate() {
            e.add_column(ColumnId((Col::E0 + i) as u32), Fr::from_u64(weight));
        }
        e
    }
}

const _: () = assert!(RANGE_COLUMNS == CHUNK_COLUMNS + DIGIT_COLUMNS);
const _: () = assert!(CHUNK_BITS == 16);

impl RowRelation {
    /// Fingerprint weight `fp^s` of operand slot `s`.
    pub fn fingerprint_weight(&self, slot: usize) -> Fr {
        self.fp_pow[slot]
    }
}
