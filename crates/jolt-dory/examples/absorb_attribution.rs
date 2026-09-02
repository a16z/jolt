//! W10b attribution bench — where do the CPU absorb-lane µs/pair go?
//!
//! ```text
//! cargo run --release -p jolt-dory --example absorb_attribution
//! ```
//!
//! Decides ticket 1 (fresh CPU Miller schedule) and calibrates ticket 2
//! (Costello–Stebila n=2 merging) with numbers from THIS box:
//!
//! - **D** ate digit structure: dbl/add step counts, adjacent zero-zero
//!   iteration pairs = the CS10 n=2 merge coverage.
//! - **P** primitive rates, 1 thread: arkworks Fq/Fq2/Fq12 ops vs a fresh
//!   no-carry CIOS Montgomery mul and a lazy-reduction Fq2 mul (double-width
//!   accumulation, one REDC per output coordinate) — parity-checked.
//! - **S** structural probe: a fresh multi-Miller over prepared coefficients
//!   (no per-pair G2Prepared clone, ONE shared squaring ladder, pairwise
//!   line combining) built on arkworks field ops — value-checked against
//!   `Bn254::multi_miller_loop`, then timed 1-thread and all-core in the
//!   production absorb shape.

#![expect(
    clippy::print_stdout,
    clippy::cast_precision_loss,
    reason = "benchmark harness: report to stdout, fail loudly"
)]

use std::time::Instant;

use ark_bn254::{Bn254, Fq, Fq12, Fq2, Fq6, G1Affine, G1Projective, G2Projective};
use ark_ec::bn::BnConfig;
use ark_ec::pairing::Pairing;
use ark_ec::CurveGroup;
use ark_ff::{BigInt, Field, Fp6Config, MontConfig, One, PrimeField, UniformRand};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;

type ArkG2Prepared = <Bn254 as Pairing>::G2Prepared;

const N_PAIRS: usize = 8192;
const SHARD: usize = 512;

fn min_secs(passes: usize, mut f: impl FnMut()) -> f64 {
    f();
    (0..passes)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    println!(
        "absorb attribution | threads {} | pairs {N_PAIRS} | shard {SHARD}\n",
        rayon::current_num_threads()
    );
    digits();
    primitives();
    structural();
}

// --- D: ate digit structure ---------------------------------------------------

fn digits() {
    let ate = <ark_bn254::Config as BnConfig>::ATE_LOOP_COUNT;
    let iters = ate.len() - 1;
    // Iteration k (k = 0 outermost) doubles, then adds when digit ≠ 0 —
    // the digit consulted is ate[len − 1 − k] per the prepare/eval loops.
    let digit = |k: usize| ate[iters - 1 - k];
    let nonzero = (0..iters).filter(|&k| digit(k) != 0).count();

    // CS10 n=2 coverage: greedy left-to-right pairing of consecutive
    // iterations where BOTH digits are zero (merged step = ℓ₁²·ℓ₂ with the
    // simple degree-3 expansion; an add line anywhere in the window needs
    // the bigger triple-product table entry).
    let mut zz_merges = 0usize;
    let mut any_merges = 0usize;
    let mut k = 0;
    while k + 1 < iters {
        if digit(k) == 0 && digit(k + 1) == 0 {
            zz_merges += 1;
            any_merges += 1;
            k += 2;
        } else {
            // A uniform scheme merges every window regardless of digits.
            any_merges += 1;
            k += 2;
        }
    }
    println!("== D: ate structure ==");
    println!(
        "iterations {iters} | nonzero digits {nonzero} | ell steps/pair {} (dbl {iters} + add {nonzero} + frob 2)",
        iters + nonzero + 2
    );
    println!(
        "CS10 n=2: zero-zero merges {zz_merges} (cover {} of {iters} dbl steps), uniform windows {any_merges}\n",
        2 * zz_merges
    );
}

// --- P: primitives --------------------------------------------------------------
//
// The fresh Montgomery mul: CIOS with the no-carry optimization (the same
// algorithm arkworks' generic path uses — the probe asks what hand
// scheduling buys over the macro expansion on this core). Written from the
// algorithm statement; limb constants read from arkworks at runtime.

fn fq_modulus() -> [u64; 4] {
    Fq::MODULUS.0
}

fn fq_inv() -> u64 {
    <ark_bn254::FqConfig as MontConfig<4>>::INV
}

#[inline(always)]
fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let t = a as u128 + (b as u128) * (c as u128) + carry as u128;
    (t as u64, (t >> 64) as u64)
}

/// a·b·R⁻¹ mod p, canonical output. No-carry CIOS, N = 4.
#[expect(clippy::needless_range_loop, reason = "b indexes the outer CIOS word")]
#[inline(always)]
fn mont_mul(a: &[u64; 4], b: &[u64; 4], p: &[u64; 4], inv: u64) -> [u64; 4] {
    let mut t = [0u64; 4];
    for i in 0..4 {
        let (r0, mut c1) = mac(t[0], a[0], b[i], 0);
        let m = r0.wrapping_mul(inv);
        let (_, mut c2) = mac(r0, m, p[0], 0);
        for j in 1..4 {
            let (rj, c1n) = mac(t[j], a[j], b[i], c1);
            c1 = c1n;
            let (out, c2n) = mac(rj, m, p[j], c2);
            c2 = c2n;
            t[j - 1] = out;
        }
        t[3] = c1 + c2;
    }
    conditional_sub(&mut t, p);
    t
}

#[inline(always)]
fn conditional_sub(t: &mut [u64; 4], p: &[u64; 4]) {
    let mut borrow = 0u64;
    let mut r = [0u64; 4];
    for i in 0..4 {
        let (d, b1) = t[i].overflowing_sub(p[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        r[i] = d;
        borrow = u64::from(b1 | b2);
    }
    if borrow == 0 {
        *t = r;
    }
}

/// 4×4 → 8-limb schoolbook product (no reduction) — the lazy-Fq2 leg.
#[inline(always)]
fn mul_wide(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
    let mut t = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u64;
        for j in 0..4 {
            let (lo, hi) = mac(t[i + j], a[i], b[j], carry);
            t[i + j] = lo;
            carry = hi;
        }
        t[i + 4] = carry;
    }
    t
}

/// Montgomery reduction of an 8-limb value < R·p, canonical output.
#[inline(always)]
fn redc8(t: &[u64; 8], p: &[u64; 4], inv: u64) -> [u64; 4] {
    let mut t = *t;
    let mut carry_top = 0u64;
    for i in 0..4 {
        let m = t[i].wrapping_mul(inv);
        let (_, mut carry) = mac(t[i], m, p[0], 0);
        for j in 1..4 {
            let (lo, hi) = mac(t[i + j], m, p[j], carry);
            t[i + j] = lo;
            carry = hi;
        }
        let s = t[i + 4] as u128 + carry as u128 + carry_top as u128;
        t[i + 4] = s as u64;
        carry_top = (s >> 64) as u64;
    }
    let mut out = [t[4], t[5], t[6], t[7]];
    conditional_sub(&mut out, p);
    out
}

/// 8-limb add (lazy accumulation).
#[inline(always)]
fn add_wide(a: &[u64; 8], b: &[u64; 8]) -> [u64; 8] {
    let mut r = [0u64; 8];
    let mut carry = 0u64;
    for i in 0..8 {
        let s = a[i] as u128 + b[i] as u128 + carry as u128;
        r[i] = s as u64;
        carry = (s >> 64) as u64;
    }
    r
}

/// 8-limb subtract with a p²-multiple pre-added to keep it nonnegative.
#[inline(always)]
fn sub_wide(a: &[u64; 8], b: &[u64; 8], p2: &[u64; 8]) -> [u64; 8] {
    let shifted = add_wide(a, p2);
    let mut r = [0u64; 8];
    let mut borrow = 0u64;
    for i in 0..8 {
        let (d, b1) = shifted[i].overflowing_sub(b[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        r[i] = d;
        borrow = u64::from(b1 | b2);
    }
    assert_eq!(borrow, 0, "lazy sub underflow");
    r
}

fn limbs(x: Fq) -> [u64; 4] {
    x.0 .0
}

fn fq_from(limbs: [u64; 4]) -> Fq {
    Fq::new_unchecked(BigInt::new(limbs))
}

fn primitives() {
    let p = fq_modulus();
    let inv = fq_inv();
    let mut rng = ChaCha20Rng::seed_from_u64(0x7731);

    // Parity: fresh mont-mul and lazy Fq2 against arkworks on random values.
    for _ in 0..4096 {
        let a = Fq::rand(&mut rng);
        let b = Fq::rand(&mut rng);
        assert_eq!(
            fq_from(mont_mul(&limbs(a), &limbs(b), &p, inv)),
            a * b,
            "fresh mont-mul disagrees with arkworks"
        );
    }
    // p² as 8 limbs, for the lazy-subtraction shift.
    let p2 = mul_wide(&p, &p);
    for _ in 0..4096 {
        let a = Fq2::rand(&mut rng);
        let b = Fq2::rand(&mut rng);
        // Lazy Karatsuba: c0 = a0b0 − a1b1, c1 = (a0+a1)(b0+b1) − a0b0 − a1b1,
        // double-width until one REDC per coordinate.
        let v0 = mul_wide(&limbs(a.c0), &limbs(b.c0));
        let v1 = mul_wide(&limbs(a.c1), &limbs(b.c1));
        let s0 = limbs(a.c0 + a.c1);
        let s1 = limbs(b.c0 + b.c1);
        let cross = mul_wide(&s0, &s1);
        let c0 = redc8(&sub_wide(&v0, &v1, &p2), &p, inv);
        let c1 = redc8(&sub_wide(&sub_wide(&cross, &v0, &p2), &v1, &p2), &p, inv);
        let got = Fq2::new(fq_from(c0), fq_from(c1));
        assert_eq!(got, a * b, "lazy Fq2 disagrees with arkworks");
    }
    println!("== P: primitives (parity ✓, 1 thread) ==");

    let n = 1usize << 20;
    let a = Fq::rand(&mut rng);
    let b = Fq::rand(&mut rng);

    // Dependent chain: latency-bound, the Miller fold shape.
    let ark_chain = min_secs(5, || {
        let mut x = a;
        for _ in 0..n {
            x *= b;
        }
        let _ = std::hint::black_box(x);
    });
    let fresh_chain = min_secs(5, || {
        let mut x = limbs(a);
        let bl = limbs(b);
        for _ in 0..n {
            x = mont_mul(&x, &bl, &p, inv);
        }
        let _ = std::hint::black_box(x);
    });
    // 4-way independent: throughput-bound.
    let seeds: Vec<Fq> = (0..4).map(|_| Fq::rand(&mut rng)).collect();
    let ark_tp = min_secs(5, || {
        let mut xs = [seeds[0], seeds[1], seeds[2], seeds[3]];
        for _ in 0..n / 4 {
            for x in &mut xs {
                *x *= b;
            }
        }
        let _ = std::hint::black_box(xs);
    });
    let fresh_tp = min_secs(5, || {
        let mut xs = [
            limbs(seeds[0]),
            limbs(seeds[1]),
            limbs(seeds[2]),
            limbs(seeds[3]),
        ];
        let bl = limbs(b);
        for _ in 0..n / 4 {
            for x in &mut xs {
                *x = mont_mul(x, &bl, &p, inv);
            }
        }
        let _ = std::hint::black_box(xs);
    });
    println!(
        "fq mul chain: ark {:.2} ns vs fresh {:.2} ns ({:.2}×) | 4-way: ark {:.2} ns vs fresh {:.2} ns ({:.2}×)",
        ark_chain / n as f64 * 1e9,
        fresh_chain / n as f64 * 1e9,
        ark_chain / fresh_chain,
        ark_tp / n as f64 * 1e9,
        fresh_tp / n as f64 * 1e9,
        ark_tp / fresh_tp,
    );

    // Fq2: arkworks (3 mont muls) vs lazy (3 wide muls + 2 REDCs).
    let a2 = Fq2::rand(&mut rng);
    let b2 = Fq2::rand(&mut rng);
    let m = n / 8;
    let ark_fq2 = min_secs(5, || {
        let mut x = a2;
        for _ in 0..m {
            x *= b2;
        }
        let _ = std::hint::black_box(x);
    });
    let lazy_fq2 = min_secs(5, || {
        let mut x = (limbs(a2.c0), limbs(a2.c1));
        let b0 = limbs(b2.c0);
        let b1 = limbs(b2.c1);
        let bs = limbs(b2.c0 + b2.c1);
        for _ in 0..m {
            let v0 = mul_wide(&x.0, &b0);
            let v1 = mul_wide(&x.1, &b1);
            let s0 = fq_from(x.0) + fq_from(x.1);
            let cross = mul_wide(&limbs(s0), &bs);
            x.0 = redc8(&sub_wide(&v0, &v1, &p2), &p, inv);
            x.1 = redc8(&sub_wide(&sub_wide(&cross, &v0, &p2), &v1, &p2), &p, inv);
        }
        let _ = std::hint::black_box(x);
    });
    println!(
        "fq2 mul chain: ark {:.1} ns vs lazy {:.1} ns ({:.2}×)",
        ark_fq2 / m as f64 * 1e9,
        lazy_fq2 / m as f64 * 1e9,
        ark_fq2 / lazy_fq2,
    );

    // Fq12 fold ops (arkworks): the Miller inner loop's currency.
    let mut f = Fq12::rand(&mut rng);
    let c0 = Fq2::rand(&mut rng);
    let c3 = Fq2::rand(&mut rng);
    let c4 = Fq2::rand(&mut rng);
    let k = n / 32;
    let sqr = min_secs(5, || {
        let mut x = f;
        for _ in 0..k {
            let _ = x.square_in_place();
        }
        let _ = std::hint::black_box(x);
    });
    let m034 = min_secs(5, || {
        let mut x = f;
        for _ in 0..k {
            x.mul_by_034(&c0, &c3, &c4);
        }
        let _ = std::hint::black_box(x);
    });
    let full = min_secs(5, || {
        let mut x = f;
        for _ in 0..k {
            x *= f;
        }
        let _ = std::hint::black_box(x);
    });
    let _ = f.square_in_place();
    println!(
        "fq12 ark: sqr {:.0} ns | mul_by_034 {:.0} ns | full mul {:.0} ns\n",
        sqr / k as f64 * 1e9,
        m034 / k as f64 * 1e9,
        full / k as f64 * 1e9,
    );
}

// --- S: structural probe --------------------------------------------------------
//
// Fresh multi-Miller over prepared coefficients, arkworks field ops:
// no per-pair clone, one squaring ladder for the whole slice, pairwise
// line combining (the fq12.metal formulas, Rust twins). Value equals
// arkworks' multi_miller_loop by partition invariance of the Miller
// product — asserted below before timing.

fn mul_by_v(x: &Fq6) -> Fq6 {
    Fq6::new(
        x.c2 * <ark_bn254::Fq6Config as Fp6Config>::NONRESIDUE,
        x.c0,
        x.c1,
    )
}

struct LinePair {
    c0: Fq6,
    c1a: Fq2,
    c1b: Fq2,
}

fn combine_lines(l1: &(Fq2, Fq2, Fq2), l2: &(Fq2, Fq2, Fq2)) -> LinePair {
    let (a1, b1, c1) = *l1;
    let (a2, b2, c2) = *l2;
    let aa = a1 * a2;
    let bb = b1 * b2;
    let cc = c1 * c2;
    let ab = (a1 + b1) * (a2 + b2) - aa - bb;
    let ac = (a1 + c1) * (a2 + c2) - aa - cc;
    let bc = (b1 + c1) * (b2 + c2) - bb - cc;
    LinePair {
        c0: Fq6::new(
            aa + cc * <ark_bn254::Fq6Config as Fp6Config>::NONRESIDUE,
            bb,
            bc,
        ),
        c1a: ab,
        c1b: ac,
    }
}

fn mul_by_line_pair(f: &mut Fq12, l: &LinePair) {
    let v0 = f.c0 * l.c0;
    let mut v1 = f.c1;
    v1.mul_by_01(&l.c1a, &l.c1b);
    let sum = Fq6::new(l.c0.c0 + l.c1a, l.c0.c1 + l.c1b, l.c0.c2);
    let c1 = (f.c0 + f.c1) * sum - v0 - v1;
    f.c0 = v0 + mul_by_v(&v1);
    f.c1 = c1;
}

/// One coefficient step over the slice: evaluate each live pair's line at
/// its G1 point, fold pairwise (odd tail via mul_by_034).
fn fold_step(f: &mut Fq12, points: &[G1Affine], coeffs: &[&ArkG2Prepared], step: usize) {
    let mut held: Option<(Fq2, Fq2, Fq2)> = None;
    for (pt, prep) in points.iter().zip(coeffs) {
        let c = &prep.ell_coeffs[step];
        let line = (
            Fq2::new(c.0.c0 * pt.y, c.0.c1 * pt.y),
            Fq2::new(c.1.c0 * pt.x, c.1.c1 * pt.x),
            c.2,
        );
        match held.take() {
            Some(h) => mul_by_line_pair(f, &combine_lines(&h, &line)),
            None => held = Some(line),
        }
    }
    if let Some((c0, c3, c4)) = held {
        f.mul_by_034(&c0, &c3, &c4);
    }
}

/// The fresh loop: `∏ᵢ miller(points[i], coeffs[i])` over one shared ladder.
fn fresh_multi_miller(points: &[G1Affine], coeffs: &[&ArkG2Prepared]) -> Fq12 {
    let ate = <ark_bn254::Config as BnConfig>::ATE_LOOP_COUNT;
    let iters = ate.len() - 1;
    let mut f = Fq12::one();
    let mut step = 0usize;
    for k in 0..iters {
        if k != 0 {
            let _ = f.square_in_place();
        }
        fold_step(&mut f, points, coeffs, step);
        step += 1;
        if ate[iters - 1 - k] != 0 {
            fold_step(&mut f, points, coeffs, step);
            step += 1;
        }
    }
    fold_step(&mut f, points, coeffs, step);
    fold_step(&mut f, points, coeffs, step + 1);
    f
}

/// Step-major coefficient slab (the device table's layout): entry
/// `step · n_rows + row` — at each step the slice's rows read
/// near-sequentially instead of scattering across per-row Vecs.
fn build_slab(preps: &[ArkG2Prepared]) -> Vec<(Fq2, Fq2, Fq2)> {
    let steps = preps[0].ell_coeffs.len();
    let n_rows = preps.len();
    let zero: Fq2 = ark_ff::Zero::zero();
    let mut slab = vec![(zero, zero, zero); steps * n_rows];
    for (row, prep) in preps.iter().enumerate() {
        for (step, c) in prep.ell_coeffs.iter().enumerate() {
            slab[step * n_rows + row] = *c;
        }
    }
    slab
}

fn fold_step_slab(
    f: &mut Fq12,
    points: &[G1Affine],
    rows: &[u32],
    slab: &[(Fq2, Fq2, Fq2)],
    n_rows: usize,
    step: usize,
) {
    let base = &slab[step * n_rows..(step + 1) * n_rows];
    let mut held: Option<(Fq2, Fq2, Fq2)> = None;
    for (pt, &row) in points.iter().zip(rows) {
        let c = &base[row as usize];
        let line = (
            Fq2::new(c.0.c0 * pt.y, c.0.c1 * pt.y),
            Fq2::new(c.1.c0 * pt.x, c.1.c1 * pt.x),
            c.2,
        );
        match held.take() {
            Some(h) => mul_by_line_pair(f, &combine_lines(&h, &line)),
            None => held = Some(line),
        }
    }
    if let Some((c0, c3, c4)) = held {
        f.mul_by_034(&c0, &c3, &c4);
    }
}

fn fresh_multi_miller_slab(
    points: &[G1Affine],
    rows: &[u32],
    slab: &[(Fq2, Fq2, Fq2)],
    n_rows: usize,
) -> Fq12 {
    let ate = <ark_bn254::Config as BnConfig>::ATE_LOOP_COUNT;
    let iters = ate.len() - 1;
    let mut f = Fq12::one();
    let mut step = 0usize;
    for k in 0..iters {
        if k != 0 {
            let _ = f.square_in_place();
        }
        fold_step_slab(&mut f, points, rows, slab, n_rows, step);
        step += 1;
        if ate[iters - 1 - k] != 0 {
            fold_step_slab(&mut f, points, rows, slab, n_rows, step);
            step += 1;
        }
    }
    fold_step_slab(&mut f, points, rows, slab, n_rows, step);
    fold_step_slab(&mut f, points, rows, slab, n_rows, step + 1);
    f
}

fn structural() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x7732);
    let g1_seeds: Vec<G1Affine> = (0..128)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    // DISTINCT preps: recycled G2 seeds leave the coefficient table
    // cache-resident and flatter every CPU arm (production streams a
    // 137 MB table in row windows).
    let g2_scalars: Vec<ark_bn254::Fr> = (0..N_PAIRS)
        .map(|_| ark_bn254::Fr::rand(&mut rng))
        .collect();
    let g2_gen = G2Projective::rand(&mut rng);
    let preps: Vec<ArkG2Prepared> = g2_scalars
        .par_iter()
        .map(|s| (g2_gen * s).into_affine().into())
        .collect();
    let ps: Vec<G1Affine> = (0..N_PAIRS).map(|i| g1_seeds[(i * 31 + 7) % 128]).collect();

    // Parity: fresh loop ≡ arkworks on a shard.
    let refs: Vec<&ArkG2Prepared> = preps[..SHARD].iter().collect();
    let fresh = fresh_multi_miller(&ps[..SHARD], &refs);
    let ark = Bn254::multi_miller_loop(ps[..SHARD].iter().copied(), preps[..SHARD].iter().cloned());
    assert_eq!(fresh, ark.0, "fresh loop disagrees with arkworks");
    println!("== S: structural probe (parity ✓ over {SHARD} pairs) ==");

    // 1-thread shard: arkworks prepared / absorb-real (clone) / fresh.
    let ark_1t = min_secs(3, || {
        let f =
            Bn254::multi_miller_loop(ps[..SHARD].iter().copied(), preps[..SHARD].iter().cloned());
        let _ = std::hint::black_box(f.0);
    });
    let clone_1t = min_secs(3, || {
        let qc: Vec<ArkG2Prepared> = (0..SHARD).map(|i| preps[i].clone()).collect();
        let f = Bn254::multi_miller_loop(ps[..SHARD].iter().copied(), qc);
        let _ = std::hint::black_box(f.0);
    });
    let fresh_1t = min_secs(3, || {
        let refs: Vec<&ArkG2Prepared> = preps[..SHARD].iter().collect();
        let f = fresh_multi_miller(&ps[..SHARD], &refs);
        let _ = std::hint::black_box(f);
    });
    println!(
        "1-thread {SHARD} pairs: ark prepared {:.2} µs/pair | ark absorb-real {:.2} | fresh {:.2} ({:.2}× vs absorb-real)",
        ark_1t / SHARD as f64 * 1e6,
        clone_1t / SHARD as f64 * 1e6,
        fresh_1t / SHARD as f64 * 1e6,
        clone_1t / fresh_1t,
    );

    // Step-major slab variant (the device table layout, host-shared).
    let slab = build_slab(&preps);
    let all_rows: Vec<u32> = (0..N_PAIRS as u32).collect();
    let slab_check = fresh_multi_miller_slab(&ps[..SHARD], &all_rows[..SHARD], &slab, N_PAIRS);
    assert_eq!(slab_check, ark.0, "slab loop disagrees with arkworks");
    let slab_1t = min_secs(3, || {
        let f = fresh_multi_miller_slab(&ps[..SHARD], &all_rows[..SHARD], &slab, N_PAIRS);
        let _ = std::hint::black_box(f);
    });
    println!(
        "1-thread slab (step-major): {:.2} µs/pair ({:.2}× vs absorb-real)",
        slab_1t / SHARD as f64 * 1e6,
        clone_1t / slab_1t,
    );

    // All-core, production absorb shape: the pass sweeps 32 row WINDOWS
    // (superchunks); within each, all 40 columns absorb the same 256-row
    // window in parallel (par_iter over columns) — ~328k pair-evals, the
    // @2^22 scale, true table locality for every arm.
    let columns = 40usize;
    let window = 256usize;
    let windows = N_PAIRS / window;
    let total_evals = columns * N_PAIRS;
    // Per (window, column): that column's G1 points for the window's rows.
    let col_points: Vec<Vec<G1Affine>> = (0..windows * columns)
        .map(|i| {
            let w = i / columns;
            (0..window)
                .map(|r| g1_seeds[(w * window + r + i * 31) % 128])
                .collect()
        })
        .collect();
    let window_rows: Vec<Vec<u32>> = (0..windows)
        .map(|w| ((w * window) as u32..((w + 1) * window) as u32).collect())
        .collect();

    let ark_cols = min_secs(2, || {
        let mut f = Fq12::one();
        for w in 0..windows {
            let acc = (0..columns)
                .into_par_iter()
                .map(|c| {
                    let rows = &window_rows[w];
                    let qc: Vec<ArkG2Prepared> = rows
                        .iter()
                        .map(|&row| preps[row as usize].clone())
                        .collect();
                    Bn254::multi_miller_loop(col_points[w * columns + c].iter().copied(), qc).0
                })
                .reduce(Fq12::one, |a, b| a * b);
            f *= acc;
        }
        let _ = std::hint::black_box(f);
    });
    let fresh_cols = min_secs(2, || {
        let mut f = Fq12::one();
        for w in 0..windows {
            let acc = (0..columns)
                .into_par_iter()
                .map(|c| {
                    let refs: Vec<&ArkG2Prepared> = window_rows[w]
                        .iter()
                        .map(|&row| &preps[row as usize])
                        .collect();
                    fresh_multi_miller(&col_points[w * columns + c], &refs)
                })
                .reduce(Fq12::one, |a, b| a * b);
            f *= acc;
        }
        let _ = std::hint::black_box(f);
    });
    let slab_cols = min_secs(2, || {
        let mut f = Fq12::one();
        for w in 0..windows {
            let acc = (0..columns)
                .into_par_iter()
                .map(|c| {
                    fresh_multi_miller_slab(
                        &col_points[w * columns + c],
                        &window_rows[w],
                        &slab,
                        N_PAIRS,
                    )
                })
                .reduce(Fq12::one, |a, b| a * b);
            f *= acc;
        }
        let _ = std::hint::black_box(f);
    });
    println!(
        "all-core {windows} windows × {columns} cols × {window} rows ({total_evals} pair-evals):"
    );
    println!(
        "  ark absorb-real {:.2} µs/pair | fresh refs {:.2} ({:.2}×) | fresh slab {:.2} ({:.2}×)\n",
        ark_cols / total_evals as f64 * 1e6,
        fresh_cols / total_evals as f64 * 1e6,
        ark_cols / fresh_cols,
        slab_cols / total_evals as f64 * 1e6,
        ark_cols / slab_cols,
    );
}
