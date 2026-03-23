//! End-to-end tests for RNS-Montgomery GPU reduce kernels.
//!
//! Validates that the RNS-Montgomery reduce kernel on Metal produces
//! the same results as CPU field arithmetic.

#![cfg(target_os = "macos")]

use std::ffi::c_void;

use jolt_field::{Field, Fr};
use jolt_ir::{KernelDescriptor, KernelShape};
use jolt_metal::rns::{BASIS_SIZE, SECONDARY_C, SECONDARY_PRIMES};
use metal::{Device, MTLResourceOptions, MTLSize};
use num_traits::Zero;
use rand::rngs::StdRng;
use rand::SeedableRng;

const NUM_PRIMES: usize = jolt_metal::rns::NUM_PRIMES;

fn reduce_mod(x: u64, p: u32, c: u32) -> u32 {
    let hi = x >> 31;
    let lo = x & 0x7FFF_FFFF;
    let r = lo + (c as u64) * hi;
    let hi2 = (r >> 31) as u32;
    let lo2 = (r & 0x7FFF_FFFF) as u32;
    let result = lo2 + c * hi2;
    if result >= p {
        result - p
    } else {
        result
    }
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }
    result
}

// --- Big integer arithmetic (minimal) ---
const BIG_LIMBS: usize = 10;
type BigInt = [u64; BIG_LIMBS];
const BIG_ZERO: BigInt = [0u64; BIG_LIMBS];

fn big_from_bytes_le(bytes: &[u8; 32]) -> BigInt {
    let mut r = BIG_ZERO;
    for i in 0..4 {
        r[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
    }
    r
}

fn big_mul(a: &BigInt, b: &BigInt) -> BigInt {
    let mut out = BIG_ZERO;
    for i in 0..BIG_LIMBS {
        if a[i] == 0 {
            continue;
        }
        let mut carry = 0u128;
        for j in 0..BIG_LIMBS {
            if i + j >= BIG_LIMBS {
                break;
            }
            let prod = (a[i] as u128) * (b[j] as u128) + (out[i + j] as u128) + carry;
            out[i + j] = prod as u64;
            carry = prod >> 64;
        }
    }
    out
}

fn big_sub(a: &BigInt, b: &BigInt) -> BigInt {
    let mut r = BIG_ZERO;
    let mut borrow = 0i64;
    for i in 0..BIG_LIMBS {
        let diff = (a[i] as i128) - (b[i] as i128) - (borrow as i128);
        r[i] = diff as u64;
        borrow = i64::from(diff < 0);
    }
    r
}

fn big_cmp(a: &BigInt, b: &BigInt) -> std::cmp::Ordering {
    for i in (0..BIG_LIMBS).rev() {
        let ord = a[i].cmp(&b[i]);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    std::cmp::Ordering::Equal
}

fn big_bit_len(a: &BigInt) -> usize {
    for i in (0..BIG_LIMBS).rev() {
        if a[i] != 0 {
            return i * 64 + (64 - a[i].leading_zeros() as usize);
        }
    }
    0
}

fn big_shl(a: &BigInt, shift: usize) -> BigInt {
    if shift == 0 {
        return *a;
    }
    let word_shift = shift / 64;
    let bit_shift = shift % 64;
    let mut r = BIG_ZERO;
    for i in 0..BIG_LIMBS {
        if i + word_shift >= BIG_LIMBS {
            break;
        }
        r[i + word_shift] |= if bit_shift == 0 {
            a[i]
        } else {
            a[i] << bit_shift
        };
        if bit_shift > 0 && i + word_shift + 1 < BIG_LIMBS {
            r[i + word_shift + 1] |= a[i] >> (64 - bit_shift);
        }
    }
    r
}

fn big_shr1(a: &BigInt) -> BigInt {
    let mut r = BIG_ZERO;
    for i in 0..BIG_LIMBS {
        r[i] = a[i] >> 1;
        if i + 1 < BIG_LIMBS {
            r[i] |= a[i + 1] << 63;
        }
    }
    r
}

fn big_mod(a: &BigInt, m: &BigInt) -> BigInt {
    if big_cmp(a, m) == std::cmp::Ordering::Less {
        return *a;
    }
    let a_bits = big_bit_len(a);
    let m_bits = big_bit_len(m);
    assert!(m_bits > 0);
    let mut remainder = *a;
    if a_bits <= m_bits {
        if big_cmp(&remainder, m) != std::cmp::Ordering::Less {
            remainder = big_sub(&remainder, m);
        }
        return remainder;
    }
    let shift = a_bits - m_bits;
    let mut shifted_m = big_shl(m, shift);
    for _ in 0..=shift {
        if big_cmp(&remainder, &shifted_m) != std::cmp::Ordering::Less {
            remainder = big_sub(&remainder, &shifted_m);
        }
        shifted_m = big_shr1(&shifted_m);
    }
    remainder
}

fn big_mod_pow(base: &BigInt, exp: &BigInt, modulus: &BigInt) -> BigInt {
    let mut result = BIG_ZERO;
    result[0] = 1;
    let mut b = big_mod(base, modulus);
    let bits = big_bit_len(exp);
    for bit in 0..bits {
        let word = bit / 64;
        let pos = bit % 64;
        if (exp[word] >> pos) & 1 == 1 {
            result = big_mod(&big_mul(&result, &b), modulus);
        }
        b = big_mod(&big_mul(&b, &b), modulus);
    }
    result
}

fn big_r() -> BigInt {
    let bytes: [u8; 32] = [
        0x01, 0x00, 0x00, 0xf0, 0x93, 0xf5, 0xe1, 0x43, 0x91, 0x70, 0xb9, 0x79, 0x48, 0xe8, 0x33,
        0x28, 0x5d, 0x58, 0x81, 0x81, 0xb6, 0x45, 0x50, 0xb8, 0x29, 0xa0, 0x31, 0xe1, 0x72, 0x4e,
        0x64, 0x30,
    ];
    big_from_bytes_le(&bytes)
}

fn big_m() -> BigInt {
    let primary = jolt_metal::rns::PRIMARY_PRIMES;
    let mut product = BIG_ZERO;
    product[0] = primary[0] as u64;
    for &p in &primary[1..] {
        let mut p_big = BIG_ZERO;
        p_big[0] = p as u64;
        product = big_mul(&product, &p_big);
    }
    product
}

/// Convert Fr to RNS-Montgomery residues (all 18 primes).
fn fr_to_rns_mont(fr: &Fr) -> [u32; NUM_PRIMES] {
    let primes = jolt_metal::rns::PRIMES;
    let c_values = jolt_metal::rns::C_VALUES;

    let bytes = fr.to_bytes();
    let a = big_from_bytes_le(&bytes);
    let r = big_r();
    let m_mod_r = big_mod(&big_m(), &r);
    let a_mont = big_mod(&big_mul(&a, &m_mod_r), &r);

    let mut u32_limbs = [0u32; 20];
    for i in 0..BIG_LIMBS {
        u32_limbs[2 * i] = a_mont[i] as u32;
        u32_limbs[2 * i + 1] = (a_mont[i] >> 32) as u32;
    }

    let mut residues = [0u32; NUM_PRIMES];
    for i in 0..NUM_PRIMES {
        let p = primes[i] as u64;
        let mut power = 1u64;
        let mut acc = 0u64;
        for (k, &limb) in u32_limbs.iter().enumerate() {
            let pow2k = (power % p) as u32;
            acc += (limb as u64) * (pow2k as u64);
            if k % 2 == 1 {
                acc = reduce_mod(acc, primes[i], c_values[i]) as u64;
            }
            power = (power * (1u64 << 32)) % p;
        }
        residues[i] = reduce_mod(acc, primes[i], c_values[i]);
    }
    residues
}

/// Reconstruct Fr from B' residues.
fn bp_residues_to_fr(bp: &[u32; BASIS_SIZE]) -> Fr {
    let mut garner = [[0u32; BASIS_SIZE]; BASIS_SIZE];
    for i in 1..BASIS_SIZE {
        let pi = SECONDARY_PRIMES[i] as u64;
        for j in 0..i {
            let pj = SECONDARY_PRIMES[j] as u64;
            garner[i][j] = mod_pow(pj, pi - 2, pi) as u32;
        }
    }

    let mut v = *bp;
    for i in 1..BASIS_SIZE {
        let pi = SECONDARY_PRIMES[i] as u64;
        for j in 0..i {
            let diff = if v[i] >= v[j] {
                (v[i] - v[j]) as u64
            } else {
                pi - (v[j] - v[i]) as u64
            };
            let inv = garner[i][j] as u64;
            v[i] = ((diff * inv) % pi) as u32;
        }
    }

    let mut result = [0u64; BIG_LIMBS];
    let mut factor = [0u64; BIG_LIMBS];
    factor[0] = 1;
    for i in 0..BASIS_SIZE {
        let vi = v[i] as u64;
        let mut carry = 0u128;
        for limb in 0..BIG_LIMBS {
            carry += (factor[limb] as u128) * (vi as u128) + (result[limb] as u128);
            result[limb] = carry as u64;
            carry >>= 64;
        }
        if i < BASIS_SIZE - 1 {
            let pi = SECONDARY_PRIMES[i] as u64;
            let mut carry_f = 0u128;
            for f in &mut factor {
                carry_f += (*f as u128) * (pi as u128);
                *f = carry_f as u64;
                carry_f >>= 64;
            }
        }
    }

    let r = big_r();
    let a_mont = big_mod(&result, &r);
    let m_mod_r = big_mod(&big_m(), &r);
    let mut exp = r;
    exp[0] -= 2;
    let m_inv = big_mod_pow(&m_mod_r, &exp, &r);
    let a = big_mod(&big_mul(&a_mont, &m_inv), &r);

    let mut le_bytes = [0u8; 32];
    for i in 0..4 {
        le_bytes[i * 8..(i + 1) * 8].copy_from_slice(&a[i].to_le_bytes());
    }
    Fr::from_le_bytes_mod_order(&le_bytes)
}

/// Build SoA u32 buffer: [prime_0: n vals | prime_1: n vals | ... | prime_17: n vals]
fn build_soa_buffer(elements: &[[u32; NUM_PRIMES]]) -> Vec<u32> {
    let n = elements.len();
    let mut buf = vec![0u32; NUM_PRIMES * n];
    for (i, elem) in elements.iter().enumerate() {
        for (r, &residue) in elem.iter().enumerate() {
            buf[r * n + i] = residue;
        }
    }
    buf
}

/// CPU reference: evaluate sumcheck reduce for ProductSum D=d, P=1.
///
/// `inputs` is a slice of D vectors (one per input polynomial), each of length 2*n_pairs
/// in interleaved layout: [lo_0, hi_0, lo_1, hi_1, ...].
///
/// At each eval point t, computes `weight × Π_k input_k(t)`.
fn cpu_reference_reduce(inputs: &[Vec<Fr>], weights: &[Fr], d: usize) -> Vec<Fr> {
    let n_elems = inputs[0].len();
    let n_pairs = n_elems / 2;
    assert_eq!(n_pairs, weights.len());

    let mut evals = vec![Fr::zero(); d];
    for i in 0..n_pairs {
        // Read lo/hi for each input
        let lo: Vec<Fr> = inputs.iter().map(|inp| inp[2 * i]).collect();
        let hi: Vec<Fr> = inputs.iter().map(|inp| inp[2 * i + 1]).collect();
        let diff: Vec<Fr> = lo.iter().zip(hi.iter()).map(|(&l, &h)| h - l).collect();

        // t=1: product of all hi values
        let mut cur = hi.clone();
        let product: Fr = cur.iter().copied().product();
        evals[0] += weights[i] * product;

        // t=2..D-1
        for t in 2..d {
            for k in 0..d {
                cur[k] += diff[k];
            }
            let product: Fr = cur.iter().copied().product();
            evals[t - 1] += weights[i] * product;
        }

        // t=∞: product of all diffs
        let product: Fr = diff.iter().copied().product();
        evals[d - 1] += weights[i] * product;
    }
    evals
}

/// Build interleaved SoA buffers for multiple input polynomials.
///
/// Each input is `[lo_0, hi_0, lo_1, hi_1, ...]` in interleaved layout.
/// Returns one SoA buffer per input polynomial.
fn build_input_soa_buffers(inputs_fr: &[Vec<Fr>]) -> Vec<Vec<u32>> {
    inputs_fr
        .iter()
        .map(|input| {
            let rns_elems: Vec<_> = input.iter().map(fr_to_rns_mont).collect();
            build_soa_buffer(&rns_elems)
        })
        .collect()
}

/// Dispatch RNS-Montgomery reduce with multiple input buffers.
fn dispatch_rns_reduce_multi(
    device: &Device,
    queue: &metal::CommandQueue,
    input_soas: &[Vec<u32>],
    weight_soa: &[u32],
    n_pairs: usize,
    d: usize,
) -> Vec<Fr> {
    let num_evals = d;
    let num_inputs = input_soas.len();

    let desc = KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: d,
            num_products: 1,
        },
        degree: d,
        tensor_split: None,
    };

    let kernel: jolt_metal::rns_compiler::RnsKernel<Fr> =
        jolt_metal::rns_compiler::compile_rns(device, &desc);

    let group_size = 256usize;
    let num_groups = n_pairs.div_ceil(group_size).min(256);

    // Upload input buffers
    let input_bufs: Vec<_> = input_soas
        .iter()
        .map(|soa| {
            device.new_buffer_with_data(
                soa.as_ptr().cast::<c_void>(),
                (soa.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        })
        .collect();

    let weight_buf = device.new_buffer_with_data(
        weight_soa.as_ptr().cast::<c_void>(),
        (weight_soa.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let partials_size = BASIS_SIZE * num_groups * num_evals;
    let partials_buf = device.new_buffer(
        (partials_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let params = [n_pairs as u32];
    let params_buf = device.new_buffer_with_data(
        params.as_ptr().cast::<c_void>(),
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&kernel.pipeline_l2h);

    // Bind input buffers at indices 0..num_inputs-1
    for (k, buf) in input_bufs.iter().enumerate() {
        enc.set_buffer(k as u64, Some(buf), 0);
    }
    // Weights at index num_inputs
    enc.set_buffer(num_inputs as u64, Some(&weight_buf), 0);
    // Partials at num_inputs + 1
    enc.set_buffer(num_inputs as u64 + 1, Some(&partials_buf), 0);
    // Params at num_inputs + 2
    enc.set_buffer(num_inputs as u64 + 2, Some(&params_buf), 0);

    enc.dispatch_thread_groups(
        MTLSize::new(num_groups as u64, 1, 1),
        MTLSize::new(group_size as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // SAFETY: shared memory is coherent after command buffer completion.
    let partials: &[u32] = unsafe {
        let ptr = partials_buf.contents().cast::<u32>();
        std::slice::from_raw_parts(ptr, partials_size)
    };

    let mut evals = Vec::with_capacity(num_evals);
    for d_idx in 0..num_evals {
        let mut bp_residues = [0u32; BASIS_SIZE];
        for j in 0..BASIS_SIZE {
            let base = j * num_groups * num_evals;
            let mut acc = 0u64;
            for g in 0..num_groups {
                acc += partials[base + g * num_evals + d_idx] as u64;
                if g % 4 == 3 {
                    acc = reduce_mod(acc, SECONDARY_PRIMES[j], SECONDARY_C[j]) as u64;
                }
            }
            bp_residues[j] = reduce_mod(acc, SECONDARY_PRIMES[j], SECONDARY_C[j]);
        }
        evals.push(bp_residues_to_fr(&bp_residues));
    }
    evals
}

#[test]
fn rns_mont_reduce_d2_weighted() {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let mut rng = StdRng::seed_from_u64(0x1234);

    let n_pairs = 64;
    let d = 2;

    // 2 input polynomials, each with 2*n_pairs elements (interleaved lo/hi)
    let inputs_fr: Vec<Vec<Fr>> = (0..d)
        .map(|_| (0..2 * n_pairs).map(|_| Fr::random(&mut rng)).collect())
        .collect();
    let weights: Vec<Fr> = (0..n_pairs).map(|_| Fr::random(&mut rng)).collect();

    let expected = cpu_reference_reduce(&inputs_fr, &weights, d);

    let input_soas = build_input_soa_buffers(&inputs_fr);
    let w_rns: Vec<_> = weights.iter().map(fr_to_rns_mont).collect();
    let weight_soa = build_soa_buffer(&w_rns);

    let got = dispatch_rns_reduce_multi(&device, &queue, &input_soas, &weight_soa, n_pairs, d);

    assert_eq!(expected.len(), got.len());
    for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
        assert_eq!(e, g, "RNS-Mont reduce D=2 random eval {i} mismatch");
    }
}

#[test]
fn rns_mont_reduce_d2_small() {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();

    let n_pairs = 4;
    let d = 2;

    // 2 inputs, each interleaved: [lo_0, hi_0, lo_1, hi_1, ...]
    let input_0 = vec![
        Fr::from_u64(1),
        Fr::from_u64(5),
        Fr::from_u64(2),
        Fr::from_u64(6),
        Fr::from_u64(3),
        Fr::from_u64(7),
        Fr::from_u64(4),
        Fr::from_u64(8),
    ];
    let input_1 = vec![
        Fr::from_u64(1),
        Fr::from_u64(1),
        Fr::from_u64(1),
        Fr::from_u64(1),
        Fr::from_u64(1),
        Fr::from_u64(1),
        Fr::from_u64(1),
        Fr::from_u64(1),
    ];
    let inputs_fr = vec![input_0, input_1];
    let weights = vec![Fr::from_u64(1); n_pairs];

    let expected = cpu_reference_reduce(&inputs_fr, &weights, d);

    let input_soas = build_input_soa_buffers(&inputs_fr);
    let w_rns: Vec<_> = weights.iter().map(fr_to_rns_mont).collect();
    let weight_soa = build_soa_buffer(&w_rns);

    let got = dispatch_rns_reduce_multi(&device, &queue, &input_soas, &weight_soa, n_pairs, d);

    assert_eq!(expected.len(), got.len());
    for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
        assert_eq!(e, g, "RNS-Mont reduce D=2 small eval {i} mismatch");
    }
}

#[test]
fn rns_mont_reduce_d3_weighted() {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let mut rng = StdRng::seed_from_u64(0xD3);

    let n_pairs = 32;
    let d = 3;

    let inputs_fr: Vec<Vec<Fr>> = (0..d)
        .map(|_| (0..2 * n_pairs).map(|_| Fr::random(&mut rng)).collect())
        .collect();
    let weights: Vec<Fr> = (0..n_pairs).map(|_| Fr::random(&mut rng)).collect();

    let expected = cpu_reference_reduce(&inputs_fr, &weights, d);

    let input_soas = build_input_soa_buffers(&inputs_fr);
    let w_rns: Vec<_> = weights.iter().map(fr_to_rns_mont).collect();
    let weight_soa = build_soa_buffer(&w_rns);

    let got = dispatch_rns_reduce_multi(&device, &queue, &input_soas, &weight_soa, n_pairs, d);

    assert_eq!(expected.len(), got.len());
    for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
        assert_eq!(e, g, "RNS-Mont reduce D=3 eval {i} mismatch");
    }
}

#[test]
fn rns_mont_reduce_d4_weighted() {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let mut rng = StdRng::seed_from_u64(0xD4);

    let n_pairs = 32;
    let d = 4;

    let inputs_fr: Vec<Vec<Fr>> = (0..d)
        .map(|_| (0..2 * n_pairs).map(|_| Fr::random(&mut rng)).collect())
        .collect();
    let weights: Vec<Fr> = (0..n_pairs).map(|_| Fr::random(&mut rng)).collect();

    let expected = cpu_reference_reduce(&inputs_fr, &weights, d);

    let input_soas = build_input_soa_buffers(&inputs_fr);
    let w_rns: Vec<_> = weights.iter().map(fr_to_rns_mont).collect();
    let weight_soa = build_soa_buffer(&w_rns);

    let got = dispatch_rns_reduce_multi(&device, &queue, &input_soas, &weight_soa, n_pairs, d);

    assert_eq!(expected.len(), got.len());
    for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
        assert_eq!(e, g, "RNS-Mont reduce D=4 eval {i} mismatch");
    }
}

// ---------------------------------------------------------------------------
// MetalBackend API reduce test
// ---------------------------------------------------------------------------

#[test]
fn rns_reduce_via_backend_d2() {
    use jolt_compute::BindingOrder;
    use jolt_metal::MetalBackend;

    let backend = MetalBackend::new();
    let mut rng = StdRng::seed_from_u64(0xBACE);

    let n_pairs = 64;
    let d = 2;

    let inputs_fr: Vec<Vec<Fr>> = (0..d)
        .map(|_| (0..2 * n_pairs).map(|_| Fr::random(&mut rng)).collect())
        .collect();
    let weights: Vec<Fr> = (0..n_pairs).map(|_| Fr::random(&mut rng)).collect();

    let expected = cpu_reference_reduce(&inputs_fr, &weights, d);

    let desc = KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: d,
            num_products: 1,
        },
        degree: d,
        tensor_split: None,
    };
    let kernel = backend.compile_rns_reduce(&desc);
    let input_bufs: Vec<_> = inputs_fr.iter().map(|v| backend.upload_rns(v)).collect();
    let weight_buf = backend.upload_rns(&weights);
    let input_refs: Vec<_> = input_bufs.iter().collect();

    let got =
        backend.rns_pairwise_reduce(&input_refs, &weight_buf, &kernel, BindingOrder::LowToHigh);

    assert_eq!(expected.len(), got.len());
    for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
        assert_eq!(e, g, "Backend RNS reduce D=2 eval {i} mismatch");
    }
}

// ---------------------------------------------------------------------------
// Bind kernel tests (via MetalBackend API)
// ---------------------------------------------------------------------------

#[test]
fn rns_bind_l2h_parity() {
    use jolt_compute::BindingOrder;
    use jolt_metal::MetalBackend;

    let backend = MetalBackend::new();
    let bind_kernel = backend.compile_rns_bind();
    let mut rng = StdRng::seed_from_u64(0xB10D);

    let n = 128;
    let data: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let challenge = Fr::random(&mut rng);

    // CPU reference: interleaved pairs, lo + challenge * (hi - lo)
    let expected: Vec<Fr> = (0..n / 2)
        .map(|i| {
            let lo = data[2 * i];
            let hi = data[2 * i + 1];
            lo + challenge * (hi - lo)
        })
        .collect();

    let mut rns_buf = backend.upload_rns(&data);
    assert_eq!(rns_buf.n_elements(), n);

    backend.rns_bind_inplace(
        &mut rns_buf,
        challenge,
        &bind_kernel,
        BindingOrder::LowToHigh,
    );
    assert_eq!(rns_buf.n_elements(), n / 2);

    // Download and reconstruct: read the RNS SoA buffer, CRT each element
    let half = n / 2;
    let raw: &[u32] = unsafe {
        let ptr = rns_buf.raw().contents().cast::<u32>();
        std::slice::from_raw_parts(ptr, NUM_PRIMES * half)
    };
    let got: Vec<Fr> = (0..half)
        .map(|i| {
            let mut bp = [0u32; BASIS_SIZE];
            for j in 0..BASIS_SIZE {
                bp[j] = raw[(BASIS_SIZE + j) * half + i];
            }
            bp_residues_to_fr(&bp)
        })
        .collect();

    for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
        assert_eq!(e, g, "RNS bind L2H element {i} mismatch");
    }
}

#[test]
fn rns_bind_h2l_parity() {
    use jolt_compute::BindingOrder;
    use jolt_metal::MetalBackend;

    let backend = MetalBackend::new();
    let bind_kernel = backend.compile_rns_bind();
    let mut rng = StdRng::seed_from_u64(0xB102);

    let n = 64;
    let data: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let challenge = Fr::random(&mut rng);

    // CPU reference: split-half pairs, lo=data[i], hi=data[i+n/2]
    let half = n / 2;
    let expected: Vec<Fr> = (0..half)
        .map(|i| {
            let lo = data[i];
            let hi = data[i + half];
            lo + challenge * (hi - lo)
        })
        .collect();

    // Upload data in H2L layout: same SoA but pairs are (buf[i], buf[i+half])
    let mut rns_buf = backend.upload_rns(&data);
    backend.rns_bind_inplace(
        &mut rns_buf,
        challenge,
        &bind_kernel,
        BindingOrder::HighToLow,
    );
    assert_eq!(rns_buf.n_elements(), half);

    let raw: &[u32] = unsafe {
        let ptr = rns_buf.raw().contents().cast::<u32>();
        std::slice::from_raw_parts(ptr, NUM_PRIMES * half)
    };
    let got: Vec<Fr> = (0..half)
        .map(|i| {
            let mut bp = [0u32; BASIS_SIZE];
            for j in 0..BASIS_SIZE {
                bp[j] = raw[(BASIS_SIZE + j) * half + i];
            }
            bp_residues_to_fr(&bp)
        })
        .collect();

    for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
        assert_eq!(e, g, "RNS bind H2L element {i} mismatch");
    }
}
