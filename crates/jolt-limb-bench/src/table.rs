//! Witness table generation from real Fq data: per row `t` random operand
//! pairs, the quotient/remainder of `Σ x_i·y_i` by q, and the three 96-bit
//! limb carries of the low positions, all decomposed into 16-bit chunks.

use ark_bn254::Fq;
use ark_ff::{BigInteger, PrimeField};
use jolt_field::{Fr, Ring};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;

pub const CHUNK_BITS: usize = 16;
pub const LIMB_BITS: usize = 96;
pub const LIMBS: usize = 3;
pub const Z_CHUNKS: usize = 16;
pub const K_CHUNKS: usize = 17;
pub const CARRIES: usize = 3;
pub const CARRY_CHUNKS: usize = 7;
pub const CARRY_OFFSET_BITS: usize = 111;
pub const CHUNK_COLUMNS: usize = Z_CHUNKS + K_CHUNKS + CARRIES * CARRY_CHUNKS;
pub const OPERAND_CHUNKS: usize = 16;

pub struct Table {
    pub rows: usize,
    pub t: usize,
    /// Column-major 16-bit chunks: z (16), k (17), carries (3 × 7), then, when
    /// operands are committed, x_i (16) and y_i (16) for every i < t.
    pub chunks: Vec<Vec<u16>>,
    /// Row-major operand limbs as field elements: per row, for each i, x_i
    /// limbs 0..3 then y_i limbs 0..3.
    pub operand_rows: Vec<Fr>,
    /// Tampering hook: `(column, row, value)` replaces a chunk by an arbitrary
    /// field element (not representable as `u16`), as a cheating prover would.
    pub overrides: Vec<(usize, usize, Fr)>,
}

pub fn q_biguint() -> BigUint {
    BigUint::from_bytes_le(&Fq::MODULUS.to_bytes_le())
}

pub fn fr_from_biguint(v: &BigUint) -> Fr {
    Fr::from(ark_bn254::Fr::from_le_bytes_mod_order(&v.to_bytes_le()))
}

pub fn limb(v: &BigUint, index: usize) -> BigUint {
    (v >> (LIMB_BITS * index)) & ((BigUint::one() << LIMB_BITS) - BigUint::one())
}

fn chunks_of(v: &BigUint, count: usize) -> Vec<u16> {
    assert!(
        v.bits() as usize <= CHUNK_BITS * count,
        "value exceeds {count} chunks"
    );
    let digits = v.to_u64_digits();
    (0..count)
        .map(|j| {
            let digit = digits.get(j / 4).copied().unwrap_or(0);
            (digit >> (16 * (j % 4))) as u16
        })
        .collect()
}

fn exact_shift_right(value: &BigInt, bits: usize) -> BigInt {
    let modulus = BigInt::one() << bits;
    assert!(
        value.is_multiple_of(&modulus),
        "carry position not divisible by 2^{bits}"
    );
    value / modulus
}

fn signed(v: &BigUint) -> BigInt {
    BigInt::from_biguint(Sign::Plus, v.clone())
}

struct Row {
    chunks: Vec<u16>,
    operand_limbs: Vec<Fr>,
}

fn generate_row(rng: &mut ChaCha20Rng, t: usize, q: &BigUint, commit_operands: bool) -> Row {
    let random_fq = |rng: &mut ChaCha20Rng| {
        let mut bytes = [0u8; 48];
        rng.fill_bytes(&mut bytes);
        BigUint::from_bytes_le(&bytes) % q
    };
    let x: Vec<BigUint> = (0..t).map(|_| random_fq(rng)).collect();
    let y: Vec<BigUint> = (0..t).map(|_| random_fq(rng)).collect();
    let sum: BigUint = x.iter().zip(&y).map(|(a, b)| a * b).sum();
    let (k, z) = sum.div_rem(q);

    let limbs = |v: &BigUint| -> [BigInt; LIMBS] { std::array::from_fn(|a| signed(&limb(v, a))) };
    let ql = limbs(q);
    let kl = limbs(&k);
    let zl = limbs(&z);
    let xl: Vec<[BigInt; LIMBS]> = x.iter().map(limbs).collect();
    let yl: Vec<[BigInt; LIMBS]> = y.iter().map(limbs).collect();

    // P_c = Σ_i Σ_{a+b=c} x_{i,a} y_{i,b} − Σ_{a+b=c} k_a q_b − z_c for c = 0, 1, 2.
    let position = |c: usize| -> BigInt {
        let mut acc = BigInt::zero();
        for a in 0..=c.min(LIMBS - 1) {
            let b = c - a;
            if b >= LIMBS {
                continue;
            }
            for (xi, yi) in xl.iter().zip(&yl) {
                acc += &xi[a] * &yi[b];
            }
            acc -= &kl[a] * &ql[b];
        }
        acc - &zl[c]
    };
    let mut carries = Vec::with_capacity(CARRIES);
    let mut carry_in = BigInt::zero();
    for c in 0..CARRIES {
        let carry = exact_shift_right(&(position(c) + &carry_in), LIMB_BITS);
        carries.push(carry.clone());
        carry_in = carry;
    }

    let mut chunks = Vec::with_capacity(
        CHUNK_COLUMNS
            + if commit_operands {
                2 * t * OPERAND_CHUNKS
            } else {
                0
            },
    );
    chunks.extend(chunks_of(&z, Z_CHUNKS));
    chunks.extend(chunks_of(&k, K_CHUNKS));
    for carry in &carries {
        let offset = carry + (BigInt::one() << CARRY_OFFSET_BITS);
        let (sign, magnitude) = offset.into_parts();
        assert_ne!(sign, Sign::Minus, "carry below −2^{CARRY_OFFSET_BITS}");
        chunks.extend(chunks_of(&magnitude, CARRY_CHUNKS));
    }
    if commit_operands {
        for (xi, yi) in x.iter().zip(&y) {
            chunks.extend(chunks_of(xi, OPERAND_CHUNKS));
            chunks.extend(chunks_of(yi, OPERAND_CHUNKS));
        }
    }
    let mut operand_limbs = Vec::with_capacity(2 * LIMBS * t);
    for (xi, yi) in x.iter().zip(&y) {
        operand_limbs.extend((0..LIMBS).map(|a| fr_from_biguint(&limb(xi, a))));
        operand_limbs.extend((0..LIMBS).map(|a| fr_from_biguint(&limb(yi, a))));
    }
    Row {
        chunks,
        operand_limbs,
    }
}

impl Table {
    pub fn generate(rows: usize, t: usize, commit_operands: bool, seed: u64) -> Self {
        let q = q_biguint();
        let generated: Vec<Row> = (0..rows)
            .into_par_iter()
            .map(|row| {
                let mut rng = ChaCha20Rng::seed_from_u64(seed ^ row as u64);
                generate_row(&mut rng, t, &q, commit_operands)
            })
            .collect();
        let num_chunk_columns = generated[0].chunks.len();
        let chunks: Vec<Vec<u16>> = (0..num_chunk_columns)
            .into_par_iter()
            .map(|column| generated.iter().map(|row| row.chunks[column]).collect())
            .collect();
        let operand_rows: Vec<Fr> = generated
            .into_par_iter()
            .flat_map_iter(|row| row.operand_limbs)
            .collect();
        Self {
            rows,
            t,
            chunks,
            operand_rows,
            overrides: Vec::new(),
        }
    }

    /// Chunk `(column, row)` as the prover commits it (override-aware).
    pub fn chunk(&self, column: usize, row: usize) -> Fr {
        self.overrides
            .iter()
            .find(|(c, r, _)| *c == column && *r == row)
            .map_or_else(
                || Fr::from_u64(u64::from(self.chunks[column][row])),
                |(_, _, v)| *v,
            )
    }

    /// Chunk column as field elements, when an override forces the full-width path.
    pub fn chunk_column_fr(&self, column: usize) -> Vec<Fr> {
        (0..self.rows).map(|row| self.chunk(column, row)).collect()
    }

    pub fn column_has_override(&self, column: usize) -> bool {
        self.overrides.iter().any(|(c, _, _)| *c == column)
    }

    /// LogUp witness: one helper column `1/Π_{i∈g}(α − chunk_i)` per group of
    /// `group_size` chunk columns, and the multiplicity of every 16-bit value
    /// across all chunk columns.
    pub fn logup_columns(&self, alpha: Fr, group_size: usize) -> (Vec<Vec<Fr>>, Vec<u32>) {
        let groups: Vec<Vec<usize>> = (0..self.chunks.len())
            .collect::<Vec<_>>()
            .chunks(group_size)
            .map(<[usize]>::to_vec)
            .collect();
        let inverses: Vec<Vec<Fr>> = groups
            .par_iter()
            .map(|group| {
                let mut values: Vec<ark_bn254::Fr> = (0..self.rows)
                    .map(|row| {
                        let product = group.iter().fold(Fr::from_u64(1), |acc, &column| {
                            acc * (alpha - self.chunk(column, row))
                        });
                        ark_bn254::Fr::from(product)
                    })
                    .collect();
                ark_ff::batch_inversion(&mut values);
                values.into_iter().map(Fr::from).collect()
            })
            .collect();
        let mut multiplicities = vec![0u32; 1 << CHUNK_BITS];
        for column in &self.chunks {
            for &chunk in column {
                multiplicities[usize::from(chunk)] += 1;
            }
        }
        (inverses, multiplicities)
    }
}

/// Field constants shared by prover and verifier.
pub struct Constants {
    pub q: Fr,
    pub q_limbs: [Fr; LIMBS],
    pub pow_limb: [Fr; LIMBS],
    pub pow_chunk: [Fr; K_CHUNKS],
    pub carry_offset: Fr,
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
        }
    }
}

pub fn recompose(chunks: &[Fr], constants: &Constants) -> Fr {
    chunks
        .iter()
        .zip(&constants.pow_chunk)
        .fold(Fr::zero(), |acc, (chunk, weight)| acc + *chunk * *weight)
}
