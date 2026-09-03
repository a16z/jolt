//! Committed witness columns of the table: per row the sixteen `u16` chunks
//! of `z`, seventeen of the offset quotient `k' = k + 2^267`, seven of each
//! offset carry `c_i + 2^111` of the limb-polynomial identity
//! `Σ κ·x(B')·y(B') − z(B') − k(B')·q(B') = (B − B')·C(B')`, then the
//! grouped LogUp helpers and the range-table multiplicities. Operands enter
//! the row relation as their evaluations at the challenge `ξ`:
//! `Z_ξ(v) = Σ_a ξ^a·limb_a(v)`.

use ark_bn254::{Fq, Fr as ArkFr};
use ark_ff::{BigInteger, PrimeField};
use jolt_field::{Fr, Ring};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Zero};
use rayon::prelude::*;
use std::sync::LazyLock;

use super::program::{half_plus_one, Program, Slot, Source};

pub const CHUNK_BITS: usize = 16;
pub const LIMB_BITS: usize = 96;
pub const LIMBS: usize = 3;
pub const Z_CHUNKS: usize = 16;
/// Canonicality of byte-linked inputs: the top 64 bits of `q` and the four
/// `u16` chunks of `d = (q_hi − 1) − z_hi` (an input is accepted only when
/// `z_hi ≤ q_hi − 1`, losing the `2^-62` fraction of canonical elements with
/// `z_hi = q_hi`).
pub const Q_HI: u64 = 0x3064_4e72_e131_a029;
pub const CANON_SHIFT: usize = 192;
pub const CANON_CHUNKS: usize = 4;
pub const K_CHUNKS: usize = 17;
/// Coefficients of the carry polynomial `C` (degree 3).
pub const CARRIES: usize = 4;
pub const CARRY_CHUNKS: usize = 7;
pub const CARRY_OFFSET_BITS: usize = 111;
/// `k` is signed (negative `κ` terms); `k' = k + 2^K_OFFSET_BITS` is committed.
pub const K_OFFSET_BITS: usize = 267;
pub const CHUNK_COLUMNS: usize = Z_CHUNKS + K_CHUNKS + CARRIES * CARRY_CHUNKS;
/// Committed digit bit columns (`zero, neg, e0, e1, e2`), range-checked with
/// the chunks so the LogUp groups divide evenly.
pub const DIGIT_COLUMNS: usize = 5;
/// Columns per LogUp helper (`h_g · Π_{i∈g}(α − c_i) = 1`).
pub const GROUP_SIZE: usize = 3;
pub const RANGE_COLUMNS: usize = CHUNK_COLUMNS + DIGIT_COLUMNS;
pub const HELPER_COLUMNS: usize = RANGE_COLUMNS / GROUP_SIZE;
/// Range-table size (`2^CHUNK_BITS` values) as a row count.
pub const TABLE_LOG: usize = CHUNK_BITS;

const _: () = assert!(RANGE_COLUMNS.is_multiple_of(GROUP_SIZE));

pub type RowChunks = [u16; CHUNK_COLUMNS];

pub fn q_biguint() -> BigUint {
    BigUint::from_bytes_le(&Fq::MODULUS.to_bytes_le())
}

pub fn fq_to_biguint(x: &Fq) -> BigUint {
    BigUint::from_bytes_le(&x.into_bigint().to_bytes_le())
}

pub fn fr_from_biguint(v: &BigUint) -> Fr {
    Fr::from(ArkFr::from_le_bytes_mod_order(&v.to_bytes_le()))
}

pub fn fr_from_bigint(v: &BigInt) -> Fr {
    let magnitude = fr_from_biguint(v.magnitude());
    if v.sign() == Sign::Minus {
        -magnitude
    } else {
        magnitude
    }
}

pub fn limb(v: &BigUint, index: usize) -> BigUint {
    (v >> (LIMB_BITS * index)) & ((BigUint::one() << LIMB_BITS) - BigUint::one())
}

/// The 96-bit limbs of an integer below `2^256` as field elements.
pub fn limbs_of(v: &BigUint) -> [Fr; LIMBS] {
    std::array::from_fn(|a| fr_from_biguint(&limb(v, a)))
}

pub fn fq_limbs(v: &Fq) -> [Fr; LIMBS] {
    limbs_of(&fq_to_biguint(v))
}

/// `Σ_a ξ^a·limb_a`.
pub fn xi_form(limbs: &[Fr; LIMBS], xi_powers: &[Fr; LIMBS]) -> Fr {
    limbs
        .iter()
        .zip(xi_powers)
        .fold(Fr::from_u64(0), |acc, (l, p)| acc + *l * *p)
}

pub fn xi_powers(xi: Fr) -> [Fr; LIMBS] {
    [Fr::from_u64(1), xi, xi * xi]
}

fn chunks_of(v: &BigUint, out: &mut [u16]) {
    assert!(
        v.bits() as usize <= CHUNK_BITS * out.len(),
        "value exceeds {} chunks",
        out.len()
    );
    let digits = v.to_u64_digits();
    for (j, chunk) in out.iter_mut().enumerate() {
        let digit = digits.get(j / 4).copied().unwrap_or(0);
        *chunk = (digit >> (16 * (j % 4))) as u16;
    }
}

fn exact_shift_right(value: &BigInt, bits: usize) -> BigInt {
    let modulus = BigInt::one() << bits;
    assert!(
        value.is_multiple_of(&modulus),
        "carry position not divisible by 2^{bits}"
    );
    value / modulus
}

/// The witness of one compute row: `Σ κ·X·Y = k·q + z` over the integers and
/// the carries `c_i = (d_i + c_{i−1}) / B` of the position sums `d_i`.
/// `Σ κ·x·y` of a row's slots over the integers.
fn slot_sum(slots: &[Slot], values: &[BigUint]) -> BigInt {
    slots.iter().fold(BigInt::zero(), |acc, slot| {
        acc + BigInt::from(slot.coefficient())
            * BigInt::from(&values[slot.x as usize] * &values[slot.y as usize])
    })
}

fn row_chunks(slots: &[Slot], values: &[BigUint], z: &BigUint, q: &BigUint) -> RowChunks {
    let q_limbs: [BigInt; LIMBS] = std::array::from_fn(|a| BigInt::from(limb(q, a)));
    let mut positions = vec![BigInt::zero(); 2 * LIMBS - 1];
    for slot in slots {
        let x = &values[slot.x as usize];
        let y = &values[slot.y as usize];
        let kappa = BigInt::from(slot.coefficient());
        for a in 0..LIMBS {
            for b in 0..LIMBS {
                positions[a + b] += &kappa * BigInt::from(limb(x, a) * limb(y, b));
            }
        }
    }
    let sum = slot_sum(slots, values);
    let z_int = BigInt::from(z.clone());
    let q_int = BigInt::from(q.clone());
    // Exact rows: `z = Σ + (1 − flag)·2^256` differs from the sum by a
    // multiple of `2^256 ≡ 2^256 (mod q)`; the flag term enters the identity
    // as the operand `exact·(1 − flag)·2^64·ξ²`, so it is not part of `k`.
    let flagged = &z_int - &sum == BigInt::one() << 256;
    let (k, remainder) = if flagged {
        (BigInt::zero(), BigInt::zero())
    } else {
        (&sum - &z_int).div_rem(&q_int)
    };
    assert!(remainder.is_zero(), "row value is not the sum modulo q");
    if flagged {
        positions[LIMBS - 1] += BigInt::one() << (256 - 2 * LIMB_BITS);
    }
    let k_offset = BigInt::one() << K_OFFSET_BITS;
    let k_prime = (&k + &k_offset)
        .to_biguint()
        .unwrap_or_else(|| unreachable!("k above -2^267"));
    // Limbs of the signed k: the two low limbs of k' and its top limb minus
    // the offset's contribution 2^(267-192).
    let k_limbs: [BigInt; LIMBS] = std::array::from_fn(|a| {
        let l = BigInt::from(limb(&k_prime, a));
        if a == LIMBS - 1 {
            l - (BigInt::one() << (K_OFFSET_BITS - 2 * LIMB_BITS))
        } else {
            l
        }
    });
    let z_limbs: [BigInt; LIMBS] = std::array::from_fn(|a| BigInt::from(limb(z, a)));
    let mut chunks = [0u16; CHUNK_COLUMNS];
    chunks_of(z, &mut chunks[..Z_CHUNKS]);
    chunks_of(&k_prime, &mut chunks[Z_CHUNKS..Z_CHUNKS + K_CHUNKS]);
    let mut carry_in = BigInt::zero();
    for c in 0..CARRIES {
        let mut position = positions[c].clone() + &carry_in;
        if c < LIMBS {
            position -= &z_limbs[c];
        }
        for a in 0..LIMBS {
            if c >= a && c - a < LIMBS {
                position -= &k_limbs[a] * &q_limbs[c - a];
            }
        }
        let carry = exact_shift_right(&position, LIMB_BITS);
        let offset = (&carry + (BigInt::one() << CARRY_OFFSET_BITS))
            .to_biguint()
            .unwrap_or_else(|| unreachable!("carry above -2^111"));
        let start = Z_CHUNKS + K_CHUNKS + CARRY_CHUNKS * c;
        chunks_of(&offset, &mut chunks[start..start + CARRY_CHUNKS]);
        carry_in = carry;
    }
    // Top position: `d_4 + c_3 = 0` is implied by the integer identity.
    let top = &positions[2 * LIMBS - 2] - &k_limbs[LIMBS - 1] * &q_limbs[LIMBS - 1] + &carry_in;
    assert!(top.is_zero(), "limb identity does not close");
    chunks
}

/// Chunk row of a padding row: value zero, `k = 0`, carries zero.
pub fn zero_row_chunks() -> RowChunks {
    let mut chunks = [0u16; CHUNK_COLUMNS];
    chunks_of(
        &(BigUint::one() << K_OFFSET_BITS),
        &mut chunks[Z_CHUNKS..Z_CHUNKS + K_CHUNKS],
    );
    for c in 0..CARRIES {
        let start = Z_CHUNKS + K_CHUNKS + CARRY_CHUNKS * c;
        chunks_of(
            &(BigUint::one() << CARRY_OFFSET_BITS),
            &mut chunks[start..start + CARRY_CHUNKS],
        );
    }
    chunks
}

/// The committed witness: `2^log_rows` chunk rows (program rows then zero
/// padding) and the row values they encode.
pub struct Columns {
    pub log_rows: usize,
    pub chunks: Vec<RowChunks>,
    /// Row values as integers below `2^256` (canonical `Fq` for an honest
    /// witness; the exact integer on exact rows).
    pub values: Vec<BigUint>,
    /// The values' 96-bit limbs.
    pub limbs: Vec<[Fr; LIMBS]>,
    /// The committed sign-flag column: `[of > −of]` on sign rows, one on the
    /// other exact rows (no `2^256` term), zero elsewhere.
    pub flags: Vec<u8>,
}

impl Columns {
    pub fn rows(&self) -> usize {
        1 << self.log_rows
    }

    /// Builds the chunk rows from the evaluated program; rows beyond the
    /// program are zero rows. Witness rows (quotients, inputs, constants)
    /// carry `z` with `k = 0` and zero carries.
    pub fn generate(program: &Program, values: &[Fq], log_rows: usize) -> Self {
        assert!(program.len() <= 1 << log_rows);
        assert!(log_rows >= TABLE_LOG);
        let q = q_biguint();
        let mut ints: Vec<BigUint> = values.par_iter().map(fq_to_biguint).collect();
        // Exact rows hold `Σ κ·x·y (+ (1 − flag)·2^256)` as an integer, which
        // may exceed `q`; their operands are never exact rows above `q`.
        let mut flags = vec![0u8; 1 << log_rows];
        let half = fq_to_biguint(&half_plus_one());
        for row in program.exact_rows() {
            let spec = &program.rows[row];
            let mut sum = slot_sum(&spec.slots, &ints);
            flags[row] = 1;
            if let Source::Sign { of } = spec.source {
                if ints[of as usize] < half {
                    flags[row] = 0;
                    sum += BigInt::one() << 256;
                }
            }
            ints[row] = sum
                .to_biguint()
                .unwrap_or_else(|| unreachable!("exact row value is nonnegative"));
            assert!(ints[row].bits() as usize <= CHUNK_BITS * Z_CHUNKS);
        }
        let mut chunks: Vec<RowChunks> = program
            .rows
            .par_iter()
            .zip(&ints)
            .map(|(spec, z)| {
                if spec.source == Source::Compute || spec.source.is_exact() {
                    row_chunks(&spec.slots, &ints, z, &q)
                } else {
                    let mut row = zero_row_chunks();
                    chunks_of(z, &mut row[..Z_CHUNKS]);
                    if let Source::Input(_) | Source::Window(_) = spec.source {
                        // Canonicality witness `d = (q_hi − 1) − z_hi` (top 64 bits)
                        // in the otherwise idle low quotient chunks of an input row.
                        // The `2^-62` fraction of canonical inputs with `z_hi = q_hi`
                        // has no witness: the saturated `d` fails the identity.
                        let z_hi: u64 = (z >> CANON_SHIFT).try_into().unwrap_or(u64::MAX);
                        let d = (Q_HI - 1).saturating_sub(z_hi);
                        chunks_of(
                            &BigUint::from(d),
                            &mut row[Z_CHUNKS..Z_CHUNKS + CANON_CHUNKS],
                        );
                    }
                    row
                }
            })
            .collect();
        let padding = (1usize << log_rows) - program.len();
        chunks.extend(std::iter::repeat_n(zero_row_chunks(), padding));
        ints.extend(std::iter::repeat_n(BigUint::zero(), padding));
        let limbs = ints.par_iter().map(limbs_of).collect();
        Self {
            log_rows,
            chunks,
            values: ints,
            limbs,
            flags,
        }
    }

    /// Committed chunk column `j` (`u16` scalars for the small-scalar MSM).
    pub fn chunk_column(&self, j: usize) -> Vec<u16> {
        self.chunks.iter().map(|row| row[j]).collect()
    }

    /// Chunk `(row, j)` as a field element.
    pub fn chunk(&self, row: usize, j: usize) -> Fr {
        Fr::from_u64(u64::from(self.chunks[row][j]))
    }

    /// `Z_ξ(v)` for every row.
    pub fn xi_values(&self, xi: Fr) -> Vec<Fr> {
        let powers = xi_powers(xi);
        self.limbs.par_iter().map(|l| xi_form(l, &powers)).collect()
    }

    /// Range-checked value `i` (chunks then digit bits) of `row`.
    fn range_value(&self, digit_bits: &[Vec<u8>], row: usize, i: usize) -> u64 {
        if i < CHUNK_COLUMNS {
            u64::from(self.chunks[row][i])
        } else {
            u64::from(digit_bits[i - CHUNK_COLUMNS][row])
        }
    }

    /// The LogUp multiplicity of every 16-bit value over the range-checked
    /// columns (as a `rows`-long column; challenge-free, committed in phase 1b).
    pub fn range_multiplicities(&self, digit_bits: &[Vec<u8>]) -> Vec<u32> {
        assert_eq!(digit_bits.len(), DIGIT_COLUMNS);
        let mut multiplicities = vec![0u32; self.rows()];
        for row in 0..self.rows() {
            for i in 0..RANGE_COLUMNS {
                multiplicities[self.range_value(digit_bits, row, i) as usize] += 1;
            }
        }
        multiplicities
    }

    /// The LogUp helper columns `h_g = 1/Π_{i∈g}(α − c_i)` over the
    /// range-checked columns for challenge `alpha` (phase 2a).
    pub fn range_helpers(&self, alpha: Fr, digit_bits: &[Vec<u8>]) -> Vec<Vec<Fr>> {
        assert_eq!(digit_bits.len(), DIGIT_COLUMNS);
        let value = |row, i| self.range_value(digit_bits, row, i);
        (0..HELPER_COLUMNS)
            .into_par_iter()
            .map(|g| {
                let mut products: Vec<ArkFr> = (0..self.rows())
                    .map(|row| {
                        let product = (0..GROUP_SIZE).fold(Fr::from_u64(1), |acc, i| {
                            acc * (alpha - Fr::from_u64(value(row, GROUP_SIZE * g + i)))
                        });
                        ArkFr::from(product)
                    })
                    .collect();
                ark_ff::batch_inversion(&mut products);
                products.into_iter().map(Fr::from).collect()
            })
            .collect()
    }
}

/// The committed operand columns `X_s = κ_s·Z_ξ(src_x)`, `Y_s = Z_ξ(src_y)`
/// (slot-major: all `X_s` then all `Y_s`), zero on rows without that slot.
pub fn operand_columns(program: &Program, z_xi: &[Fr], num_slots: usize) -> Vec<Vec<Fr>> {
    let rows = z_xi.len();
    let mut columns = vec![vec![Fr::from_u64(0); rows]; 2 * num_slots];
    for (row, spec) in program.rows.iter().enumerate() {
        for (s, slot) in spec.slots.iter().enumerate() {
            columns[s][row] = Fr::from_i64(i64::from(slot.kappa)) * z_xi[slot.x as usize];
            columns[num_slots + s][row] =
                Fr::from_i64(i64::from(slot.y_sign)) * z_xi[slot.y as usize];
        }
    }
    columns
}

/// Field constants of the row relation, shared by prover and verifier
/// (computed once: [`Constants::get`]; the verifier derivation does no
/// fixed-power arithmetic).
#[derive(Clone, Copy)]
pub struct Constants {
    pub q_limbs: [Fr; LIMBS],
    pub pow_limb: Fr,
    pub pow_chunk: [Fr; K_CHUNKS],
    pub carry_offset: Fr,
    /// `2^267` in top-limb form (`2^75`).
    pub k_offset_top_limb: Fr,
    /// `2^64` (the sign rows' `2^256` term in limb form is `2^64·ξ²`).
    pub pow_64: Fr,
}

static CONSTANTS: LazyLock<Constants> = LazyLock::new(Constants::new);

impl Default for Constants {
    fn default() -> Self {
        Self::new()
    }
}

impl Constants {
    pub fn new() -> Self {
        let q = q_biguint();
        Self {
            q_limbs: std::array::from_fn(|a| fr_from_biguint(&limb(&q, a))),
            pow_limb: Fr::pow2(LIMB_BITS),
            pow_chunk: std::array::from_fn(|j| Fr::pow2(CHUNK_BITS * j)),
            carry_offset: Fr::pow2(CARRY_OFFSET_BITS),
            k_offset_top_limb: Fr::pow2(K_OFFSET_BITS - 2 * LIMB_BITS),
            pow_64: Fr::pow2(64),
        }
    }

    /// The constants, computed once per process.
    pub fn get() -> &'static Self {
        &CONSTANTS
    }

    /// `q(ξ) = Σ_a ξ^a·q_a`.
    pub fn q_xi(&self, xi_powers: &[Fr; LIMBS]) -> Fr {
        xi_form(&self.q_limbs, xi_powers)
    }
}

/// `Σ_j 2^{16j} chunk_j`.
pub fn recompose(chunks: &[Fr], constants: &Constants) -> Fr {
    chunks
        .iter()
        .zip(&constants.pow_chunk)
        .fold(Fr::from_u64(0), |acc, (chunk, weight)| {
            acc + *chunk * *weight
        })
}

/// Bound check of the limb identity for a program: `Σ|κ| ≤ 2^7` keeps
/// `|k| < 2^266` and every carry below `2^105`.
pub fn kappa_bound_holds(program: &Program) -> bool {
    program.max_kappa_sum() <= 1 << 7
}
