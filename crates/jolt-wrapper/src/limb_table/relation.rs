//! The row relation and its stage-A member. Every row `x` satisfies
//! `Σ_s X_s·Y_s − z(ξ) − k(ξ)·q(ξ) − (B − ξ)·C(ξ) = 0` over `Fr` (the limb
//! polynomial identity at the challenge `ξ`, exact over the integers by the
//! chunk range checks), the public pins, the grouped LogUp range checks, the
//! digit-bit constraints and the operand-lookup constraints; the linear
//! copy, constancy and LogUp-sum identities ride along as `Σ_x (...) = 0`
//! terms without an `eq` factor. The prover folds a row-major matrix; the
//! verifier consumes the same relation as [`Term`]s over the column
//! evaluations at the stage point.

use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use super::columns::{
    recompose, xi_powers, Constants, CARRIES, CARRY_CHUNKS, CHUNK_BITS, CHUNK_COLUMNS,
    DIGIT_COLUMNS, GROUP_SIZE, HELPER_COLUMNS, K_CHUNKS, LIMBS, RANGE_COLUMNS, Z_CHUNKS,
};
use super::layout::LOG_ROWS;
use super::terms::{fold_linear, AffineForm, ColumnId, Term};

/// Slots per row (the dense `Fq12` product form).
pub const SLOTS: usize = 22;

/// Column indices of the exported column list: committed phase 1, committed
/// phase 2 (after the challenges `ξ, α, β, γ` are drawn), then the
/// VK-committed public columns.
pub mod col {
    use super::*;

    pub const CHUNKS: usize = 0;
    pub const DIGITS: usize = CHUNKS + CHUNK_COLUMNS;
    pub const ZERO: usize = DIGITS;
    pub const NEG: usize = DIGITS + 1;
    pub const E0: usize = DIGITS + 2;
    /// The digit value `d = (1 − zero)(1 − 2neg)(1 + e)` as a column.
    pub const D: usize = DIGITS + DIGIT_COLUMNS;
    pub const M_POS: usize = D + 1;
    pub const M_NEG: usize = M_POS + 1;
    pub const MULT: usize = M_NEG + 1;
    pub const PHASE1_END: usize = MULT + 1;

    pub const X: usize = PHASE1_END;
    pub const Y: usize = X + SLOTS;
    pub const HELPERS: usize = Y + SLOTS;
    pub const INV: usize = HELPERS + HELPER_COLUMNS;
    pub const H: usize = INV + 1;
    pub const G_POS: usize = H + 1;
    pub const G_NEG: usize = G_POS + 1;
    pub const F_POS: usize = G_NEG + 1;
    pub const F_NEG: usize = F_POS + 1;
    pub const PHASE2_END: usize = F_NEG + 1;
    pub const COMMITTED: usize = PHASE2_END;

    pub const PIN: usize = COMMITTED;
    pub const PIN_LIMBS: usize = PIN + 1;
    /// Rows exempt from the limb identity (inputs, public constants).
    pub const FREE: usize = PIN_LIMBS + LIMBS;
    pub const VK_END: usize = FREE + 1;
    /// Every column with an evaluation claim at the stage point.
    pub const CLAIMED: usize = VK_END;

    /// Prover-only public columns appended to the matrix.
    pub const EQ_TAU: usize = CLAIMED;
    pub const COPY_KERNEL: usize = EQ_TAU + 1;
    pub const SEL: usize = COPY_KERNEL + 1;
    pub const IS_GT: usize = SEL + 1;
    pub const IS_G1: usize = IS_GT + 1;
    pub const IS_G2: usize = IS_G1 + 1;
    pub const S0: usize = IS_G2 + 1;
    pub const COORD: usize = S0 + 1;
    pub const CONSTANCY: usize = COORD + 1;
    pub const SMALL: usize = CONSTANCY + 1;
    pub const ID: usize = SMALL + 1;
    pub const WIDTH: usize = ID + 1;
    pub const PUBLIC: usize = WIDTH - CLAIMED;
}

/// Verifier-evaluated public multilinears at the stage point, in the order of
/// the prover-only matrix columns (`col::EQ_TAU..col::WIDTH`).
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
    fn as_array(&self) -> [Fr; col::PUBLIC] {
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

/// Transcript challenges of the row member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Challenges {
    /// Big-endian row point (`tau[0]` the most significant bit, bound last).
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
pub const GAMMA_TERMS: usize = GAMMA_INV + 1;

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
    fp_pow: Vec<Fr>,
    copy_pow: Vec<Fr>,
    constancy_pow: Vec<Fr>,
}

fn powers(root: Fr, count: usize) -> Vec<Fr> {
    let mut out = Vec::with_capacity(count);
    let mut power = Fr::one();
    for _ in 0..count {
        out.push(power);
        power *= root;
    }
    out
}

impl RowRelation {
    pub fn new(challenges: Challenges, lookup: LookupConstants) -> Self {
        assert_eq!(challenges.tau.len(), LOG_ROWS);
        let constants = Constants::new();
        let xi_pow = xi_powers(challenges.xi);
        let q_xi = constants.q_xi(&xi_pow);
        Self {
            gammas: powers(challenges.gamma, GAMMA_TERMS),
            xi_pow,
            q_xi,
            fp_pow: powers(challenges.fp_root, SLOTS),
            // Operand columns then the two fingerprint columns.
            copy_pow: powers(challenges.copy_root, 2 * SLOTS + 2),
            constancy_pow: powers(challenges.constancy_root, DIGIT_COLUMNS),
            challenges,
            lookup,
            constants,
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
        let chunks = &v[col::CHUNKS..col::CHUNKS + CHUNK_COLUMNS];
        let g = &self.gammas;
        // Limb identity.
        let mut products = Fr::zero();
        for s in 0..SLOTS {
            products += v[col::X + s] * v[col::Y + s];
        }
        let z_xi = self.z_xi(chunks);
        let limb = products
            - z_xi
            - self.k_xi(chunks) * self.q_xi
            - (c.pow_limb - ch.xi) * self.c_xi(chunks);
        let mut phi = g[GAMMA_LIMB] * (Fr::one() - v[col::FREE]) * limb;
        // Pins.
        let pin_xi = (0..LIMBS).fold(Fr::zero(), |acc, a| {
            acc + v[col::PIN_LIMBS + a] * self.xi_pow[a]
        });
        phi += g[GAMMA_PIN] * v[col::PIN] * (z_xi - pin_xi);
        // Range groups.
        for grp in 0..HELPER_COLUMNS {
            let product = (0..GROUP_SIZE).fold(Fr::one(), |acc, i| {
                acc * (ch.alpha - Self::range_value(v, GROUP_SIZE * grp + i))
            });
            phi += g[GAMMA_RANGE + grp] * (v[col::HELPERS + grp] * product - Fr::one());
        }
        // Digit bits.
        for b in 0..DIGIT_COLUMNS {
            let bit = v[col::DIGITS + b];
            phi += g[GAMMA_BOOL + b] * bit * (bit - Fr::one());
        }
        phi += g[GAMMA_DIGIT_RANGE]
            * (Fr::one() - v[col::NEG])
            * v[col::E0]
            * v[col::E0 + 1]
            * v[col::E0 + 2];
        let (d, _) = Self::digit(v);
        phi += g[GAMMA_DIGIT_VALUE] * (v[col::D] - v[col::SEL] * d);
        // Operand lookup, reading side.
        let (key, fingerprint) = self.read_key(v);
        phi += g[GAMMA_READ]
            * (v[col::H] * (ch.beta + key + ch.fp_combine * fingerprint) - v[col::SEL]);
        // Operand lookup, table side.
        for (i, (gc, fc, mc)) in [
            (col::G_POS, col::F_POS, col::M_POS),
            (col::G_NEG, col::F_NEG, col::M_NEG),
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
                * (v[gc] * (ch.beta + v[col::ID] + offset + ch.fp_combine * v[fc]) - v[mc]);
        }
        // Range inverse table.
        phi += g[GAMMA_INV] * v[col::SMALL] * (v[col::INV] * (ch.alpha - v[col::ID]) - Fr::one());
        phi
    }

    fn range_value(v: &[Fr], i: usize) -> Fr {
        if i < CHUNK_COLUMNS {
            v[col::CHUNKS + i]
        } else {
            v[col::DIGITS + i - CHUNK_COLUMNS]
        }
    }

    /// Digit value `d = (1 − zero)(1 − 2·neg)(1 + e)` from the bits, and the
    /// entry magnitude `e`.
    fn digit(v: &[Fr]) -> (Fr, Fr) {
        let two = Fr::from_u64(2);
        let e = v[col::E0] + two * v[col::E0 + 1] + two * two * v[col::E0 + 2];
        let d = (Fr::one() - v[col::ZERO]) * (Fr::one() - two * v[col::NEG]) * (Fr::one() + e);
        (d, e)
    }

    /// Lookup key and fingerprint of a reading row:
    /// GT: `key = (1−zero)(S0 + 16e) + zero·(one_row + c) + 2^18·neg`,
    /// EC: `key = S0 + stride·d` (`16` for G1 cells, `8` for G2 half cells);
    /// `fingerprint = Σ_{s<n} fp^s·Y_s`.
    fn read_key(&self, v: &[Fr]) -> (Fr, Fr) {
        let (_, e) = Self::digit(v);
        let d = v[col::D];
        let sixteen = Fr::from_u64(16);
        let one_row = Fr::from_u64(u64::from(self.lookup.one_row));
        let gt_key = (Fr::one() - v[col::ZERO]) * (v[col::S0] + sixteen * e)
            + v[col::ZERO] * (one_row + v[col::COORD])
            + Fr::from_u64(NEG_KEY_OFFSET) * v[col::NEG];
        // `stride` vanishes on GT rows, so it needs no `(1 − is_gt)` factor.
        let stride = sixteen * v[col::IS_G1] + Fr::from_u64(8) * v[col::IS_G2];
        let key = v[col::IS_GT] * gt_key + (Fr::one() - v[col::IS_GT]) * v[col::S0] + stride * d;
        let fp = |n: usize| (0..n).fold(Fr::zero(), |acc, s| acc + self.fp_pow[s] * v[col::Y + s]);
        let fingerprint = v[col::IS_GT] * fp(FP_SLOTS_GT)
            + v[col::IS_G1] * fp(FP_SLOTS_G1)
            + v[col::IS_G2] * fp(FP_SLOTS_G2);
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
            logup += v[col::HELPERS + grp] * elementary;
        }
        logup -= v[col::MULT] * v[col::INV];
        // Operand lookup sum.
        let lookup = v[col::H] - v[col::G_POS] - v[col::G_NEG];
        // Copy identities: eq(τ,x)·Σ β_i·col_i(x) − B(x)·Z_ξ(x); looked-up
        // `Y_s` (the fingerprinted slots of selected rows) are not copies.
        let mut copied = Fr::zero();
        for s in 0..SLOTS {
            copied += self.copy_pow[s] * v[col::X + s];
            copied += self.copy_pow[SLOTS + s] * Self::copy_mask(v, s) * v[col::Y + s];
        }
        copied +=
            self.copy_pow[2 * SLOTS] * v[col::F_POS] + self.copy_pow[2 * SLOTS + 1] * v[col::F_NEG];
        let z_xi = self.z_xi(&v[col::CHUNKS..col::CHUNKS + CHUNK_COLUMNS]);
        let copy = v[col::EQ_TAU] * copied - v[col::COPY_KERNEL] * z_xi;
        // Digit constancy: W(x)·Σ_b β'_b·bit_b(x).
        let bits = (0..DIGIT_COLUMNS).fold(Fr::zero(), |acc, b| {
            acc + self.constancy_pow[b] * v[col::DIGITS + b]
        });
        ch.lambda * logup + ch.lambda_lookup * lookup + copy + v[col::CONSTANCY] * bits
    }

    /// `1 − Σ_kind is_kind(x)·[s < fp_slots(kind)]`: whether `Y_s` is a copy.
    fn copy_mask(v: &[Fr], s: usize) -> Fr {
        let mut mask = Fr::one() - v[col::IS_GT];
        if s < FP_SLOTS_G1 {
            mask -= v[col::IS_G1];
        }
        if s < FP_SLOTS_G2 {
            mask -= v[col::IS_G2];
        }
        mask
    }

    /// The summand at one (possibly extrapolated) row of the matrix.
    pub fn summand(&self, v: &[Fr]) -> Fr {
        v[col::EQ_TAU] * self.phi(v) + self.linear(v)
    }

    /// Every row-local constraint's unweighted value at a row (each is zero
    /// on an honest witness), for diagnostics and the per-constraint test.
    pub fn constraint_values(&self, v: &[Fr]) -> Vec<(&'static str, Fr)> {
        let ch = &self.challenges;
        let c = &self.constants;
        let chunks = &v[col::CHUNKS..col::CHUNKS + CHUNK_COLUMNS];
        let mut out = Vec::new();
        let products = (0..SLOTS).fold(Fr::zero(), |acc, s| acc + v[col::X + s] * v[col::Y + s]);
        let z_xi = self.z_xi(chunks);
        out.push((
            "limb",
            (Fr::one() - v[col::FREE])
                * (products
                    - z_xi
                    - self.k_xi(chunks) * self.q_xi
                    - (c.pow_limb - ch.xi) * self.c_xi(chunks)),
        ));
        let pin_xi = (0..LIMBS).fold(Fr::zero(), |acc, a| {
            acc + v[col::PIN_LIMBS + a] * self.xi_pow[a]
        });
        out.push(("pin", v[col::PIN] * (z_xi - pin_xi)));
        let mut range = Fr::zero();
        for grp in 0..HELPER_COLUMNS {
            let product = (0..GROUP_SIZE).fold(Fr::one(), |acc, i| {
                acc * (ch.alpha - Self::range_value(v, GROUP_SIZE * grp + i))
            });
            range += v[col::HELPERS + grp] * product - Fr::one();
        }
        out.push(("range", range));
        let mut bools = Fr::zero();
        for b in 0..DIGIT_COLUMNS {
            let bit = v[col::DIGITS + b];
            bools += bit * (bit - Fr::one());
        }
        out.push(("digit_bool", bools));
        out.push((
            "digit_range",
            (Fr::one() - v[col::NEG]) * v[col::E0] * v[col::E0 + 1] * v[col::E0 + 2],
        ));
        out.push(("digit_value", v[col::D] - v[col::SEL] * Self::digit(v).0));
        let (key, fingerprint) = self.read_key(v);
        out.push((
            "lookup_read",
            v[col::H] * (ch.beta + key + ch.fp_combine * fingerprint) - v[col::SEL],
        ));
        for (name, gc, fc, mc, offset) in [
            (
                "lookup_table_pos",
                col::G_POS,
                col::F_POS,
                col::M_POS,
                Fr::zero(),
            ),
            (
                "lookup_table_neg",
                col::G_NEG,
                col::F_NEG,
                col::M_NEG,
                Fr::from_u64(NEG_KEY_OFFSET),
            ),
        ] {
            out.push((
                name,
                v[gc] * (ch.beta + v[col::ID] + offset + ch.fp_combine * v[fc]) - v[mc],
            ));
        }
        out.push((
            "inverse_table",
            v[col::SMALL] * (v[col::INV] * (ch.alpha - v[col::ID]) - Fr::one()),
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
            logup += v[col::HELPERS + grp] * (f[1] * f[2] + f[0] * f[2] + f[0] * f[1]);
        }
        logup -= v[col::MULT] * v[col::INV];
        let lookup = v[col::H] - v[col::G_POS] - v[col::G_NEG];
        let mut copied = Fr::zero();
        for s in 0..SLOTS {
            copied += self.copy_pow[s] * v[col::X + s];
            copied += self.copy_pow[SLOTS + s] * Self::copy_mask(v, s) * v[col::Y + s];
        }
        copied +=
            self.copy_pow[2 * SLOTS] * v[col::F_POS] + self.copy_pow[2 * SLOTS + 1] * v[col::F_NEG];
        let z_xi = self.z_xi(&v[col::CHUNKS..col::CHUNKS + CHUNK_COLUMNS]);
        let copy = v[col::EQ_TAU] * copied - v[col::COPY_KERNEL] * z_xi;
        let bits = (0..DIGIT_COLUMNS).fold(Fr::zero(), |acc, b| {
            acc + self.constancy_pow[b] * v[col::DIGITS + b]
        });
        vec![
            ("range_logup", logup),
            ("lookup_logup", lookup),
            ("copy", copy),
            ("constancy", v[col::CONSTANCY] * bits),
        ]
    }

    /// The member's final relation as terms over the claimed column
    /// evaluations (`col::CLAIMED` columns), given the public multilinears
    /// at the stage point. `Σ_t term_t(v) == summand(v ∥ public)`.
    pub fn terms(&self, public: &PublicEvals) -> Vec<Term> {
        let ch = &self.challenges;
        let c = &self.constants;
        let g = &self.gammas;
        let eq = public.eq_tau;
        let column = |i: usize| AffineForm::column(ColumnId(i as u32));
        let one_minus =
            |i: usize| AffineForm::constant(Fr::one()).plus(&column(i).scale(-Fr::one()));
        let mut terms = Vec::new();
        // Limb identity, on non-free rows.
        let bound = one_minus(col::FREE);
        for s in 0..SLOTS {
            terms.push(Term::new(
                eq * g[GAMMA_LIMB],
                vec![bound.clone(), column(col::X + s), column(col::Y + s)],
            ));
        }
        let z_xi = self.z_xi_form();
        let mut limb_linear = z_xi.clone().scale(-Fr::one());
        limb_linear.accumulate(&self.k_xi_form().scale(-self.q_xi));
        limb_linear.accumulate(&self.c_xi_form().scale(-(c.pow_limb - ch.xi)));
        terms.push(Term::new(eq * g[GAMMA_LIMB], vec![bound, limb_linear]));
        // Pins.
        let mut pin_xi = AffineForm::default();
        for a in 0..LIMBS {
            pin_xi.add_column(ColumnId((col::PIN_LIMBS + a) as u32), -self.xi_pow[a]);
        }
        terms.push(Term::new(
            eq * g[GAMMA_PIN],
            vec![column(col::PIN), z_xi.clone().plus(&pin_xi)],
        ));
        // Range groups and the LogUp sum.
        for grp in 0..HELPER_COLUMNS {
            let f: Vec<AffineForm> = (0..GROUP_SIZE)
                .map(|i| {
                    AffineForm::constant(ch.alpha)
                        .plus(&Self::range_form(GROUP_SIZE * grp + i).scale(-Fr::one()))
                })
                .collect();
            let mut factors = vec![column(col::HELPERS + grp)];
            factors.extend(f.iter().cloned());
            terms.push(Term::new(eq * g[GAMMA_RANGE + grp], factors));
            terms.push(Term::new(-eq * g[GAMMA_RANGE + grp], vec![]));
            for i in 0..GROUP_SIZE {
                let mut factors = vec![column(col::HELPERS + grp)];
                factors.extend(
                    f.iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, x)| x.clone()),
                );
                terms.push(Term::new(ch.lambda, factors));
            }
        }
        terms.push(Term::new(
            -ch.lambda,
            vec![column(col::MULT), column(col::INV)],
        ));
        // Digit bits.
        for b in 0..DIGIT_COLUMNS {
            terms.push(Term::new(
                eq * g[GAMMA_BOOL + b],
                vec![
                    column(col::DIGITS + b),
                    column(col::DIGITS + b).plus(&AffineForm::constant(-Fr::one())),
                ],
            ));
        }
        terms.push(Term::new(
            eq * g[GAMMA_DIGIT_RANGE],
            vec![
                one_minus(col::NEG),
                column(col::E0),
                column(col::E0 + 1),
                column(col::E0 + 2),
            ],
        ));
        let e = Self::e_form();
        let one_plus_e = AffineForm::constant(Fr::one()).plus(&e);
        let one_minus_2neg =
            AffineForm::constant(Fr::one()).plus(&column(col::NEG).scale(-Fr::from_u64(2)));
        terms.push(Term::new(eq * g[GAMMA_DIGIT_VALUE], vec![column(col::D)]));
        terms.push(Term::new(
            -eq * g[GAMMA_DIGIT_VALUE] * public.sel,
            vec![one_minus(col::ZERO), one_minus_2neg, one_plus_e],
        ));
        // Lookup, reading side: h·(β + key + fp_combine·F) − sel.
        let sixteen = Fr::from_u64(16);
        let one_row = Fr::from_u64(u64::from(self.lookup.one_row));
        let gr = eq * g[GAMMA_READ];
        let h = column(col::H);
        terms.push(Term::new(gr * ch.beta, vec![h.clone()]));
        // GT key.
        terms.push(Term::new(
            gr * public.is_gt * public.s0,
            vec![h.clone(), one_minus(col::ZERO)],
        ));
        terms.push(Term::new(
            gr * public.is_gt * sixteen,
            vec![h.clone(), one_minus(col::ZERO), e.clone()],
        ));
        terms.push(Term::new(
            gr * public.is_gt * (one_row + public.coord),
            vec![h.clone(), column(col::ZERO)],
        ));
        terms.push(Term::new(
            gr * public.is_gt * Fr::from_u64(NEG_KEY_OFFSET),
            vec![h.clone(), column(col::NEG)],
        ));
        // EC key.
        let is_ec = Fr::one() - public.is_gt;
        terms.push(Term::new(gr * is_ec * public.s0, vec![h.clone()]));
        terms.push(Term::new(
            gr * (sixteen * public.is_g1 + Fr::from_u64(8) * public.is_g2),
            vec![h.clone(), column(col::D)],
        ));
        // Fingerprints.
        for (indicator, n) in [
            (public.is_gt, FP_SLOTS_GT),
            (public.is_g1, FP_SLOTS_G1),
            (public.is_g2, FP_SLOTS_G2),
        ] {
            let mut fp = AffineForm::default();
            for s in 0..n {
                fp.add_column(ColumnId((col::Y + s) as u32), self.fp_pow[s]);
            }
            terms.push(Term::new(
                gr * ch.fp_combine * indicator,
                vec![h.clone(), fp],
            ));
        }
        terms.push(Term::new(-gr * public.sel, vec![]));
        // Lookup, table side.
        for (i, (gc, fc, mc)) in [
            (col::G_POS, col::F_POS, col::M_POS),
            (col::G_NEG, col::F_NEG, col::M_NEG),
        ]
        .into_iter()
        .enumerate()
        {
            let offset = if i == 1 {
                Fr::from_u64(NEG_KEY_OFFSET)
            } else {
                Fr::zero()
            };
            let gt = eq * g[GAMMA_TABLE + i];
            terms.push(Term::new(
                gt * (ch.beta + public.id + offset),
                vec![column(gc)],
            ));
            terms.push(Term::new(gt * ch.fp_combine, vec![column(gc), column(fc)]));
            terms.push(Term::new(-gt, vec![column(mc)]));
        }
        // Range inverse table.
        let gi = eq * g[GAMMA_INV] * public.small;
        terms.push(Term::new(
            gi * (ch.alpha - public.id),
            vec![column(col::INV)],
        ));
        terms.push(Term::new(-gi, vec![]));
        // Lookup sum.
        terms.push(Term::new(ch.lambda_lookup, vec![h.clone()]));
        terms.push(Term::new(-ch.lambda_lookup, vec![column(col::G_POS)]));
        terms.push(Term::new(-ch.lambda_lookup, vec![column(col::G_NEG)]));
        // Copy identities (looked-up `Y_s` masked out).
        let mut copied = AffineForm::default();
        for s in 0..SLOTS {
            copied.add_column(ColumnId((col::X + s) as u32), self.copy_pow[s]);
            let mut mask = Fr::one() - public.is_gt;
            if s < FP_SLOTS_G1 {
                mask -= public.is_g1;
            }
            if s < FP_SLOTS_G2 {
                mask -= public.is_g2;
            }
            copied.add_column(
                ColumnId((col::Y + s) as u32),
                self.copy_pow[SLOTS + s] * mask,
            );
        }
        copied.add_column(ColumnId(col::F_POS as u32), self.copy_pow[2 * SLOTS]);
        copied.add_column(ColumnId(col::F_NEG as u32), self.copy_pow[2 * SLOTS + 1]);
        terms.push(Term::new(public.eq_tau, vec![copied]));
        terms.push(Term::new(-public.copy_kernel, vec![z_xi]));
        // Digit constancy.
        let mut bits = AffineForm::default();
        for b in 0..DIGIT_COLUMNS {
            bits.add_column(ColumnId((col::DIGITS + b) as u32), self.constancy_pow[b]);
        }
        terms.push(Term::new(public.constancy, vec![bits]));
        fold_linear(terms)
    }

    fn chunk_form(&self, start: usize, count: usize) -> AffineForm {
        let mut form = AffineForm::default();
        for j in 0..count {
            form.add_column(
                ColumnId((col::CHUNKS + start + j) as u32),
                self.constants.pow_chunk[j],
            );
        }
        form
    }

    fn z_xi_form(&self) -> AffineForm {
        self.chunk_form(0, 6)
            .scale(self.xi_pow[0])
            .plus(&self.chunk_form(6, 6).scale(self.xi_pow[1]))
            .plus(&self.chunk_form(12, Z_CHUNKS - 12).scale(self.xi_pow[2]))
    }

    fn k_xi_form(&self) -> AffineForm {
        let top = self
            .chunk_form(Z_CHUNKS + 12, K_CHUNKS - 12)
            .plus(&AffineForm::constant(-self.constants.k_offset_top_limb));
        self.chunk_form(Z_CHUNKS, 6)
            .plus(&self.chunk_form(Z_CHUNKS + 6, 6).scale(self.xi_pow[1]))
            .plus(&top.scale(self.xi_pow[2]))
    }

    fn c_xi_form(&self) -> AffineForm {
        let mut form = AffineForm::default();
        let mut power = Fr::one();
        for i in 0..CARRIES {
            let start = Z_CHUNKS + K_CHUNKS + CARRY_CHUNKS * i;
            form.accumulate(
                &self
                    .chunk_form(start, CARRY_CHUNKS)
                    .plus(&AffineForm::constant(-self.constants.carry_offset))
                    .scale(power),
            );
            power *= self.challenges.xi;
        }
        form
    }

    fn range_form(i: usize) -> AffineForm {
        if i < CHUNK_COLUMNS {
            AffineForm::column(ColumnId((col::CHUNKS + i) as u32))
        } else {
            AffineForm::column(ColumnId((col::DIGITS + i - CHUNK_COLUMNS) as u32))
        }
    }

    fn e_form() -> AffineForm {
        let mut e = AffineForm::default();
        for (i, weight) in [1u64, 2, 4].into_iter().enumerate() {
            e.add_column(ColumnId((col::E0 + i) as u32), Fr::from_u64(weight));
        }
        e
    }
}

const _: () = assert!(RANGE_COLUMNS == CHUNK_COLUMNS + DIGIT_COLUMNS);
const _: () = assert!(CHUNK_BITS == 16);

/// Linear extrapolation of a row pair to `X = x`: `even + x·(odd − even)`.
fn extrapolate(out: &mut [Fr], even: &[Fr], odd: &[Fr], x: Fr) {
    for ((slot, &e), &o) in out.iter_mut().zip(even).zip(odd) {
        *slot = e + x * (o - e);
    }
}

/// The row member as a batch member: a row-major matrix of every claimed and
/// public column, folded round by round (low bit first). The `eq(τ,·)`
/// column is part of the matrix, so the round polynomial is the plain
/// degree-`d+1` evaluation of the summand at `d + 2` points.
pub struct RowSumcheck<'a> {
    relation: &'a RowRelation,
    matrix: Vec<Fr>,
    scratch: Vec<Fr>,
    rows: usize,
    round: usize,
    points: Vec<Fr>,
    /// A cheating prover forces every round check to pass so that rejection
    /// happens at the verifier's final relation check.
    pub cheat: bool,
}

impl<'a> RowSumcheck<'a> {
    /// `columns[i]` is matrix column `i` (`col::WIDTH` columns of `2^LOG_ROWS`
    /// rows: claimed columns then the public ones).
    pub fn new(relation: &'a RowRelation, columns: &[Vec<Fr>]) -> Self {
        assert_eq!(columns.len(), col::WIDTH);
        let rows = 1usize << LOG_ROWS;
        for column in columns {
            assert_eq!(column.len(), rows);
        }
        let mut matrix = vec![Fr::zero(); rows * col::WIDTH];
        matrix
            .par_chunks_mut(col::WIDTH)
            .enumerate()
            .for_each(|(row, slot)| {
                for (value, column) in slot.iter_mut().zip(columns) {
                    *value = column[row];
                }
            });
        Self {
            relation,
            matrix,
            scratch: Vec::new(),
            rows,
            round: 0,
            points: (0..=(Self::degree() + 1) as u64)
                .map(Fr::from_u64)
                .collect(),
            cheat: false,
        }
    }

    /// Summand degree in each variable (`eq` included).
    pub const fn degree() -> usize {
        RowRelation::degree()
    }

    /// `Σ_row summand(row)` — zero for an honest witness.
    pub fn input_claim(&self) -> Fr {
        self.matrix
            .par_chunks(col::WIDTH)
            .map(|row| self.relation.summand(row))
            .sum()
    }

    fn round_poly(&self, claim: Fr) -> Vec<Fr> {
        let relation = self.relation;
        let width = col::WIDTH;
        let evals: Vec<Fr> = self.matrix[..self.rows * width]
            .par_chunks(2 * width)
            .fold(
                || (vec![Fr::zero(); self.points.len()], vec![Fr::zero(); width]),
                |(mut acc, mut scratch), pair| {
                    let (even, odd) = pair.split_at(width);
                    acc[0] += relation.summand(even);
                    acc[1] += relation.summand(odd);
                    for (i, &point) in self.points.iter().enumerate().skip(2) {
                        extrapolate(&mut scratch, even, odd, point);
                        acc[i] += relation.summand(&scratch);
                    }
                    (acc, scratch)
                },
            )
            .map(|(acc, _)| acc)
            .reduce(
                || vec![Fr::zero(); self.points.len()],
                |a, b| a.iter().zip(&b).map(|(x, y)| *x + *y).collect(),
            );
        let mut coefficients = UnivariatePoly::from_evals(&evals).into_coefficients();
        while coefficients.len() > 2 && coefficients.last() == Some(&Fr::zero()) {
            let _ = coefficients.pop();
        }
        if self.cheat {
            let tail: Fr = coefficients[1..].iter().fold(Fr::zero(), |acc, c| acc + *c);
            coefficients[0] = (claim - tail) * two_inverse();
        }
        coefficients
    }

    fn bind(&mut self, r: Fr) {
        let width = col::WIDTH;
        let half = self.rows / 2;
        if self.scratch.len() < half * width {
            self.scratch.resize(half * width, Fr::zero());
        }
        self.scratch[..half * width]
            .par_chunks_mut(width)
            .zip(self.matrix[..self.rows * width].par_chunks(2 * width))
            .for_each(|(out, pair)| {
                let (even, odd) = pair.split_at(width);
                extrapolate(out, even, odd, r);
            });
        std::mem::swap(&mut self.matrix, &mut self.scratch);
        self.rows = half;
        self.round += 1;
    }

    /// Final evaluations after every round: the claimed columns, then the
    /// public columns (which the verifier recomputes).
    pub fn final_row(&self) -> &[Fr] {
        assert_eq!(self.rows, 1, "sumcheck not finished");
        &self.matrix[..col::WIDTH]
    }

    /// Claimed column evaluations at the point (`col::CLAIMED` values).
    pub fn claims(&self) -> Vec<Fr> {
        self.final_row()[..col::CLAIMED].to_vec()
    }

    /// The public columns' final values (the verifier's oracle in tests).
    pub fn public_evals(&self) -> PublicEvals {
        let p = &self.final_row()[col::CLAIMED..];
        PublicEvals {
            eq_tau: p[0],
            copy_kernel: p[1],
            sel: p[2],
            is_gt: p[3],
            is_g1: p[4],
            is_g2: p[5],
            s0: p[6],
            coord: p[7],
            constancy: p[8],
            small: p[9],
            id: p[10],
        }
    }
}

impl PublicEvals {
    /// Matrix row layout helper: the public part of a final row.
    pub fn to_row(&self) -> [Fr; col::PUBLIC] {
        self.as_array()
    }
}

fn two_inverse() -> Fr {
    Fr::from_u64(2)
        .inverse()
        .unwrap_or_else(|| unreachable!("2 is invertible"))
}

impl ProveRounds<Fr> for RowSumcheck<'_> {
    fn num_rounds(&self) -> usize {
        LOG_ROWS
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

/// `eq(τ, x)` over the rows for the big-endian `tau`.
pub fn eq_tau_column(tau: &[Fr]) -> Vec<Fr> {
    EqPolynomial::<Fr>::evals(tau, None)
}

impl RowRelation {
    /// Fingerprint weight `fp^s` of operand slot `s`.
    pub fn fingerprint_weight(&self, slot: usize) -> Fr {
        self.fp_pow[slot]
    }
}
