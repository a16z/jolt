//! Committed witness columns of the table (lane M2's layout): per row the
//! sixteen `u16` chunks of `z`, seventeen of the offset quotient
//! `k' = k + 2^267`, seven of each offset limb carry `c + 2^111`, then the
//! grouped LogUp helpers and the range-table multiplicities.

use ark_bn254::Fq;
use ark_ff::{BigInteger, PrimeField};
use jolt_field::{Fr, Ring};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Zero};
use rayon::prelude::*;

use super::program::{Program, Slot, Source};

pub const CHUNK_BITS: usize = 16;
pub const LIMB_BITS: usize = 96;
pub const LIMBS: usize = 3;
pub const Z_CHUNKS: usize = 16;
pub const K_CHUNKS: usize = 17;
pub const CARRIES: usize = 3;
pub const CARRY_CHUNKS: usize = 7;
pub const CARRY_OFFSET_BITS: usize = 111;
/// `k` is signed (negative `κ` terms); `k' = k + 2^K_OFFSET_BITS` is committed.
pub const K_OFFSET_BITS: usize = 267;
pub const CHUNK_COLUMNS: usize = Z_CHUNKS + K_CHUNKS + CARRIES * CARRY_CHUNKS;
/// Chunk columns per LogUp helper (`h_g · Π_{i∈g}(α − c_i) = 1`).
pub const GROUP_SIZE: usize = 3;
pub const HELPER_COLUMNS: usize = CHUNK_COLUMNS / GROUP_SIZE;
/// Committed columns: chunks, helpers, one multiplicity column.
pub const COMMITTED_COLUMNS: usize = CHUNK_COLUMNS + HELPER_COLUMNS + 1;
/// Range-table size (`2^CHUNK_BITS` values) as a row count.
pub const TABLE_LOG: usize = CHUNK_BITS;

pub type RowChunks = [u16; CHUNK_COLUMNS];

pub fn q_biguint() -> BigUint {
    BigUint::from_bytes_le(&Fq::MODULUS.to_bytes_le())
}

pub fn fq_to_biguint(x: &Fq) -> BigUint {
    BigUint::from_bytes_le(&x.into_bigint().to_bytes_le())
}

pub fn fr_from_biguint(v: &BigUint) -> Fr {
    Fr::from(ark_bn254::Fr::from_le_bytes_mod_order(&v.to_bytes_le()))
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

/// The CRT witness of one row: `Σ κ·X·Y = k·q + z` over the integers with
/// `X = κ`-scaled limbs of the source rows, and the three low-position carries.
fn row_chunks(slots: &[Slot], values: &[BigUint], z: &BigUint, q: &BigUint) -> RowChunks {
    let q_limbs: [BigInt; LIMBS] = std::array::from_fn(|a| BigInt::from(limb(q, a)));
    let mut sum = BigInt::zero();
    let mut positions = [BigInt::zero(), BigInt::zero(), BigInt::zero()];
    for slot in slots {
        let x = &values[slot.x as usize];
        let y = &values[slot.y as usize];
        let kappa = BigInt::from(slot.kappa);
        sum += &kappa * BigInt::from(x * y);
        for a in 0..LIMBS {
            for b in 0..LIMBS - a {
                positions[a + b] += &kappa * BigInt::from(limb(x, a) * limb(y, b));
            }
        }
    }
    let z_int = BigInt::from(z.clone());
    let q_int = BigInt::from(q.clone());
    let (k, remainder) = (&sum - &z_int).div_rem(&q_int);
    assert!(remainder.is_zero(), "row value is not the sum modulo q");
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
        let mut position = positions[c].clone() - &z_limbs[c] + &carry_in;
        for a in 0..=c {
            position -= &k_limbs[a] * &q_limbs[c - a];
        }
        let carry = exact_shift_right(&position, LIMB_BITS);
        let offset = (&carry + (BigInt::one() << CARRY_OFFSET_BITS))
            .to_biguint()
            .unwrap_or_else(|| unreachable!("carry above -2^111"));
        let start = Z_CHUNKS + K_CHUNKS + CARRY_CHUNKS * c;
        chunks_of(&offset, &mut chunks[start..start + CARRY_CHUNKS]);
        carry_in = carry;
    }
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
    /// Row values as integers below `2^256` (canonical `Fq` for an honest witness).
    pub values: Vec<BigUint>,
    /// The values' 96-bit limbs, the virtual operands the wiring reads.
    pub limbs: Vec<[Fr; LIMBS]>,
}

impl Columns {
    pub fn rows(&self) -> usize {
        1 << self.log_rows
    }

    /// Builds the chunk rows from the evaluated program; rows beyond the
    /// program are zero rows.
    pub fn generate(program: &Program, values: &[Fq], log_rows: usize) -> Self {
        assert!(program.len() <= 1 << log_rows);
        assert!(log_rows >= TABLE_LOG);
        let q = q_biguint();
        let mut ints: Vec<BigUint> = values.par_iter().map(fq_to_biguint).collect();
        let mut chunks: Vec<RowChunks> = program
            .rows
            .par_iter()
            .zip(&ints)
            .map(|(spec, z)| {
                if spec.source == Source::Compute {
                    row_chunks(&spec.slots, &ints, z, &q)
                } else {
                    let mut row = zero_row_chunks();
                    chunks_of(z, &mut row[..Z_CHUNKS]);
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

    /// LogUp witness for challenge `alpha`: helper columns
    /// `h_g = 1/Π_{i∈g}(α − chunk_i)` and the multiplicity of every 16-bit
    /// value over all chunk columns (as a `rows`-long column).
    pub fn logup_columns(&self, alpha: Fr) -> (Vec<Vec<Fr>>, Vec<u32>) {
        let helpers: Vec<Vec<Fr>> = (0..HELPER_COLUMNS)
            .into_par_iter()
            .map(|g| {
                let mut products: Vec<ark_bn254::Fr> = self
                    .chunks
                    .iter()
                    .map(|row| {
                        let product = (0..GROUP_SIZE).fold(Fr::from_u64(1), |acc, i| {
                            acc * (alpha - Fr::from_u64(u64::from(row[GROUP_SIZE * g + i])))
                        });
                        ark_bn254::Fr::from(product)
                    })
                    .collect();
                ark_ff::batch_inversion(&mut products);
                products.into_iter().map(Fr::from).collect()
            })
            .collect();
        let mut multiplicities = vec![0u32; self.rows()];
        for row in &self.chunks {
            for &chunk in row {
                multiplicities[usize::from(chunk)] += 1;
            }
        }
        (helpers, multiplicities)
    }
}

/// Field constants of the row relation, shared by prover and verifier.
pub struct Constants {
    pub q: Fr,
    pub q_limbs: [Fr; LIMBS],
    pub pow_limb: [Fr; LIMBS],
    pub pow_chunk: [Fr; K_CHUNKS],
    pub carry_offset: Fr,
    /// `2^267` and its top-limb form `2^75`.
    pub k_offset: Fr,
    pub k_offset_top_limb: Fr,
}

impl Default for Constants {
    fn default() -> Self {
        Self::new()
    }
}

impl Constants {
    pub fn new() -> Self {
        let q = q_biguint();
        Self {
            q: fr_from_biguint(&q),
            q_limbs: std::array::from_fn(|a| fr_from_biguint(&limb(&q, a))),
            pow_limb: std::array::from_fn(|a| Fr::pow2(LIMB_BITS * a)),
            pow_chunk: std::array::from_fn(|j| Fr::pow2(CHUNK_BITS * j)),
            carry_offset: Fr::pow2(CARRY_OFFSET_BITS),
            k_offset: Fr::pow2(K_OFFSET_BITS),
            k_offset_top_limb: Fr::pow2(K_OFFSET_BITS - 2 * LIMB_BITS),
        }
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

/// Bound check of the CRT argument for a program: `Σ|κ|·2^512 < 2^519` keeps
/// `|k| < 2^266` and every carry below `2^105`.
pub fn kappa_bound_holds(program: &Program) -> bool {
    program.max_kappa_sum() <= 1 << 7
}
