//! Test/bench support for the Metal tier: the cross-process GPU lock,
//! deterministic field-element fixtures (shared by the parity tests below
//! and the `metal_microbench` example), and the device-path probe counters
//! the parity gates assert against.

use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU64, Ordering};

use jolt_field::{CanonicalEncoding, Fr, Ring};

use super::montgomery::MontgomeryConstants;

use super::field::fr_from_u32_limbs;

/// Successful device round dispatches (process lifetime). Threshold gates
/// fall back SILENTLY by design, so a test that means to exercise the device
/// must assert this advanced — otherwise it green-lights the CPU fallback.
static DEVICE_ROUNDS: AtomicU64 = AtomicU64::new(0);

/// Kernel dispatches encoded by test/bench builds.
static DEVICE_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Buffers whose construction fell back to allocate+copy (no-copy declined).
static COPIED_BUFFERS: AtomicU64 = AtomicU64::new(0);

/// Committed tier-2 Miller dispatches (the commit slot's hybrid absorb).
static MILLER_DISPATCHES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn note_device_round() {
    let _ = DEVICE_ROUNDS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(any(test, feature = "bench-utils"))]
pub(crate) fn note_device_dispatch() {
    let _ = DEVICE_DISPATCHES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn note_miller_dispatch() {
    let _ = MILLER_DISPATCHES.fetch_add(1, Ordering::Relaxed);
}

/// How many tier-2 Miller dispatches the commit slot has committed in this
/// process — the parity tests' probe that the device arm really ran (the
/// gate falls back SILENTLY by design).
pub fn miller_dispatch_count() -> u64 {
    MILLER_DISPATCHES.load(Ordering::Relaxed)
}

pub(crate) fn note_copied_buffers(count: u64) {
    if count > 0 {
        let _ = COPIED_BUFFERS.fetch_add(count, Ordering::Relaxed);
    }
}

/// How many sumcheck rounds have run on the device in this process.
pub fn device_probe_count() -> u64 {
    DEVICE_ROUNDS.load(Ordering::Relaxed)
}

/// How many Metal kernel dispatches test/bench code has encoded.
pub fn device_dispatch_count() -> u64 {
    DEVICE_DISPATCHES.load(Ordering::Relaxed)
}

/// How many slot buffers fell back to allocate+copy in this process.
pub fn copied_buffer_count() -> u64 {
    COPIED_BUFFERS.load(Ordering::Relaxed)
}

/// Exclusive advisory lock on `/tmp/jolt-gpu.lock`, the campaign convention
/// for serializing GPU-touching tests (nextest runs one process per test;
/// macOS has no flock(1) binary, so the lock is taken via the flock(2)
/// syscall INSIDE tests/benches — never wrap the outer nextest invocation).
/// Held for the guard's lifetime; released on drop (fd close).
pub struct GpuLock(#[expect(dead_code, reason = "held for the flock")] File);

#[expect(
    clippy::expect_used,
    reason = "test/bench support: fail loudly on lock setup"
)]
pub fn gpu_lock() -> GpuLock {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/jolt-gpu.lock")
        .expect("open /tmp/jolt-gpu.lock");
    // SAFETY: flock on an owned, open fd; blocks until the lock is granted.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
    assert_eq!(rc, 0, "flock(/tmp/jolt-gpu.lock) failed");
    GpuLock(file)
}

/// Deterministic uniform-ish field elements (splitmix64 stream reduced
/// mod p) — dependency-free so the microbench example can share it.
pub fn seeded_frs(seed: u64, n: usize) -> Vec<Fr> {
    let mut state = seed;
    let mut next_u64 = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    (0..n)
        .map(|_| {
            let mut bytes = [0u8; 64];
            for chunk in bytes.chunks_exact_mut(8) {
                chunk.copy_from_slice(&next_u64().to_le_bytes());
            }
            Fr::from_bytes_le_reduced(&bytes)
        })
        .collect()
}

/// Arithmetic edge cases: additive/multiplicative identities, the wrap
/// boundary, R and R² (values whose Montgomery forms are the seam
/// constants), and the largest canonical limb pattern.
pub fn edge_frs() -> Vec<Fr> {
    let zero = Fr::from_u64(0);
    let one = Fr::from_u64(1);
    let minus_one = zero - one;
    // r2_u32 is R² mod p = the Montgomery form of the VALUE R.
    let r = fr_from_u32_limbs(&u32_seam_limbs(Fr::r2_u32()));
    let r_squared = r * r;
    // Montgomery limbs p-1: the largest canonical representation.
    let mut pm1 = u32_seam_limbs(Fr::modulus_u32());
    pm1[0] -= 1; // p is odd, no borrow
    let max_repr = fr_from_u32_limbs(&pm1);
    let two = Fr::from_u64(2);
    vec![zero, one, minus_one, r, r_squared, max_repr, two]
}

fn u32_seam_limbs(limbs: &[u32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    out.copy_from_slice(limbs);
    out
}

/// Host reference for the bind fold: `out[i] = a[2i] + r·(a[2i+1] − a[2i])`.
pub fn host_bind(a: &[Fr], r: Fr) -> Vec<Fr> {
    a.chunks_exact(2)
        .map(|p| p[0] + r * (p[1] - p[0]))
        .collect()
}

/// Host reference for bind_eval's partial-sum output:
/// `s(t_j) = Σ_i a[2i] + t_j·(a[2i+1] − a[2i])`.
pub fn host_bind_eval_sums(a: &[Fr], points: &[Fr]) -> Vec<Fr> {
    points
        .iter()
        .map(|t| {
            a.chunks_exact(2)
                .map(|p| p[0] + *t * (p[1] - p[0]))
                .fold(Fr::from_u64(0), |acc, v| acc + v)
        })
        .collect()
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests: fail loudly"
)]
mod tests {
    use jolt_dory::DoryScheme;
    use jolt_poly::UnivariatePoly;

    use super::super::field::{fr_as_u32s, fr_to_u32_limbs, FR_U32_LIMBS};
    use super::super::runtime::{KernelId, MetalContext, MAX_EVAL_POINTS, THREADGROUP_SIZE};
    use super::super::{metal_gate, PageAlignedVec};
    use super::*;
    use crate::JoltBackend;

    /// The layout contract behind every device view: Fr's bytes viewed as
    /// [u64; 4] LE and as [u32; 8] LE are the same bytes, and the u32 view
    /// round-trips through the seam recombination.
    #[test]
    fn fr_layout_matches_u32_view() {
        let mut elems = edge_frs();
        elems.extend(seeded_frs(0xf00d, 1000));

        let u32_view = fr_as_u32s(&elems);
        assert_eq!(u32_view.len(), elems.len() * FR_U32_LIMBS);

        // Byte-identity of the two views ("hash both ways").
        // SAFETY: plain byte views of initialized memory.
        let fr_bytes = unsafe {
            std::slice::from_raw_parts(elems.as_ptr().cast::<u8>(), size_of_val(elems.as_slice()))
        };
        // SAFETY: as above.
        let u32_bytes = unsafe {
            std::slice::from_raw_parts(u32_view.as_ptr().cast::<u8>(), size_of_val(u32_view))
        };
        assert_eq!(fr_bytes, u32_bytes);

        for (i, &e) in elems.iter().enumerate() {
            let device_limbs: [u32; FR_U32_LIMBS] = u32_view[i * FR_U32_LIMBS..][..FR_U32_LIMBS]
                .try_into()
                .unwrap();
            // The in-memory u32 view IS the u64 Montgomery limbs, split.
            assert_eq!(device_limbs, fr_to_u32_limbs(e));
            // And recombination lands on the identical element.
            assert_eq!(fr_from_u32_limbs(&device_limbs), e);
            // Explicit u64↔u32 split check against the inner limbs.
            for (w, pair) in e.inner_limbs().iter().zip(device_limbs.chunks_exact(2)) {
                assert_eq!(*w as u32, pair[0]);
                assert_eq!((*w >> 32) as u32, pair[1]);
            }
        }
    }

    /// Pairs covering every edge×edge combination, padded to `n` with
    /// seeded random elements.
    fn paired_inputs(n: usize) -> (Vec<Fr>, Vec<Fr>) {
        let edges = edge_frs();
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for &x in &edges {
            for &y in &edges {
                a.push(x);
                b.push(y);
            }
        }
        assert!(a.len() <= n);
        a.extend(seeded_frs(1, n - a.len()));
        b.extend(seeded_frs(2, n - b.len()));
        (a, b)
    }

    fn run_elemwise(kernel: KernelId, a: &[Fr], b: &[Fr]) -> Vec<Fr> {
        let ctx = MetalContext::global().expect("metal context");
        let a_buf = ctx.wrap_slice(fr_as_u32s(a)).unwrap();
        let b_buf = ctx.wrap_slice(fr_as_u32s(b)).unwrap();
        let out_buf = ctx.alloc_u32s(a.len() * FR_U32_LIMBS).unwrap();
        ctx.run_once(
            kernel,
            &[a.len() as u32],
            &[&a_buf, &b_buf, &out_buf],
            a.len(),
        )
        .unwrap();
        read_frs(&out_buf, a.len())
    }

    fn read_frs(buffer: &super::super::DeviceBuffer<'_>, n: usize) -> Vec<Fr> {
        let mut words = vec![0u32; n * FR_U32_LIMBS];
        buffer.copy_to_u32s(&mut words);
        words
            .chunks_exact(FR_U32_LIMBS)
            .map(|c| fr_from_u32_limbs(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn device_mont_mul_matches_host() {
        let _lock = gpu_lock();
        let (a, b) = paired_inputs(1 << 16);
        let device = run_elemwise(KernelId::FrMul, &a, &b);
        for i in 0..a.len() {
            assert_eq!(device[i], a[i] * b[i], "mul mismatch at {i}");
        }
    }

    #[test]
    fn device_add_matches_host() {
        let _lock = gpu_lock();
        let (a, b) = paired_inputs(1 << 16);
        let device = run_elemwise(KernelId::FrAdd, &a, &b);
        for i in 0..a.len() {
            assert_eq!(device[i], a[i] + b[i], "add mismatch at {i}");
        }
    }

    #[test]
    fn device_sub_matches_host() {
        let _lock = gpu_lock();
        let (a, b) = paired_inputs(1 << 16);
        let device = run_elemwise(KernelId::FrSub, &a, &b);
        for i in 0..a.len() {
            assert_eq!(device[i], a[i] - b[i], "sub mismatch at {i}");
        }
    }

    #[test]
    fn device_pow2k_matches_host() {
        let _lock = gpu_lock();
        let mut a = edge_frs();
        a.extend(seeded_frs(3, (1 << 12) - a.len()));
        let k = 16u32;

        let ctx = MetalContext::global().expect("metal context");
        let a_buf = ctx.wrap_slice(fr_as_u32s(&a)).unwrap();
        let out_buf = ctx.alloc_u32s(a.len() * FR_U32_LIMBS).unwrap();
        ctx.run_once(
            KernelId::FrPow2k,
            &[a.len() as u32, k],
            &[&a_buf, &out_buf],
            a.len(),
        )
        .unwrap();
        let device = read_frs(&out_buf, a.len());

        for i in 0..a.len() {
            let mut expected = a[i];
            for _ in 0..k {
                expected = expected * expected;
            }
            assert_eq!(device[i], expected, "pow2k mismatch at {i}");
        }
    }

    fn bind_params(n_out: usize, r: Fr) -> Vec<u32> {
        let mut params = vec![n_out as u32];
        params.extend_from_slice(&fr_to_u32_limbs(r));
        params
    }

    fn bind4_weights(z: Fr) -> [Fr; 4] {
        std::array::from_fn(|index| UnivariatePoly::evaluate_basis(4, index, z))
    }

    fn bind4_params(n_out: usize, weights: [Fr; 4]) -> Vec<u32> {
        let mut params = vec![n_out as u32];
        for weight in &weights[1..] {
            params.extend_from_slice(&fr_to_u32_limbs(*weight));
        }
        params
    }

    fn run_bind4(ctx: &MetalContext, values: &[Fr], z: Fr) -> Vec<Fr> {
        let n_out = values.len() / 4;
        let values_buf = ctx.wrap_slice(fr_as_u32s(values)).unwrap();
        let out_buf = ctx.alloc_u32s(n_out * FR_U32_LIMBS).unwrap();
        ctx.run_once(
            KernelId::FrBind4,
            &bind4_params(n_out, bind4_weights(z)),
            &[&values_buf, &out_buf],
            n_out,
        )
        .unwrap();
        read_frs(&out_buf, n_out)
    }

    #[test]
    fn device_bind_matches_host() {
        let _lock = gpu_lock();
        let ctx = MetalContext::global().expect("metal context");
        // 2^16 elements plus a ragged size exercising the bounds check.
        for n_in in [1usize << 16, 2 * 5] {
            let mut a = edge_frs();
            a.extend(seeded_frs(4, n_in - a.len().min(n_in)));
            a.truncate(n_in);
            let n_out = n_in / 2;
            let r = seeded_frs(5, 1)[0];

            let a_buf = ctx.wrap_slice(fr_as_u32s(&a)).unwrap();
            let out_buf = ctx.alloc_u32s(n_out * FR_U32_LIMBS).unwrap();
            ctx.run_once(
                KernelId::FrBind,
                &bind_params(n_out, r),
                &[&a_buf, &out_buf],
                n_out,
            )
            .unwrap();
            let device = read_frs(&out_buf, n_out);

            assert_eq!(device, host_bind(&a, r), "bind mismatch at n_in={n_in}");
        }
    }

    #[test]
    fn device_bind4_matches_quaternary_lagrange_host() {
        let _lock = gpu_lock();
        let ctx = MetalContext::global().expect("metal context");
        for n_in in [1usize << 16, 4 * 5] {
            let mut a = edge_frs();
            a.extend(seeded_frs(0x44, n_in - a.len().min(n_in)));
            a.truncate(n_in);
            let n_out = n_in / 4;
            let z = seeded_frs(0x45, 1)[0];
            let weights = bind4_weights(z);

            let a_buf = ctx.wrap_slice(fr_as_u32s(&a)).unwrap();
            let out_buf = ctx.alloc_u32s(n_out * FR_U32_LIMBS).unwrap();
            ctx.run_once(
                KernelId::FrBind4,
                &bind4_params(n_out, weights),
                &[&a_buf, &out_buf],
                n_out,
            )
            .unwrap();
            let device = read_frs(&out_buf, n_out);
            let host: Vec<Fr> = a
                .chunks_exact(4)
                .map(|values| {
                    values
                        .iter()
                        .zip(weights)
                        .fold(Fr::from_u64(0), |acc, (&value, weight)| {
                            acc + value * weight
                        })
                })
                .collect();

            assert_eq!(device, host, "bind4 mismatch at n_in={n_in}");
        }
    }

    #[test]
    fn radix4_degree_two_sumcheck_shape_accepts_and_rejects_tampering() {
        let _lock = gpu_lock();
        let ctx = MetalContext::global().expect("metal context");
        let n_in = 1usize << 14;
        let left = seeded_frs(0x4a, n_in);
        let right = seeded_frs(0x4b, n_in);

        let q_evals: Vec<Fr> = (0..=6)
            .map(|point| {
                let weights = bind4_weights(Fr::from_u64(point));
                left.chunks_exact(4)
                    .zip(right.chunks_exact(4))
                    .map(|(left_values, right_values)| {
                        let left_eval = left_values
                            .iter()
                            .zip(weights)
                            .fold(Fr::from_u64(0), |acc, (&value, weight)| {
                                acc + value * weight
                            });
                        let right_eval = right_values
                            .iter()
                            .zip(weights)
                            .fold(Fr::from_u64(0), |acc, (&value, weight)| {
                                acc + value * weight
                            });
                        left_eval * right_eval
                    })
                    .fold(Fr::from_u64(0), |acc, value| acc + value)
            })
            .collect();
        let q = UnivariatePoly::from_evals(&q_evals);
        assert!(q.coefficients().len() <= 7, "degree exceeds 3d=6");

        let current_claim = left
            .iter()
            .zip(&right)
            .fold(Fr::from_u64(0), |acc, (&l, &r)| acc + l * r);
        let four_point_sum = (0..4)
            .map(|point| q.evaluate(Fr::from_u64(point)))
            .fold(Fr::from_u64(0), |acc, value| acc + value);
        assert_eq!(four_point_sum, current_claim);

        let z = seeded_frs(0x4c, 1)[0];
        let left_bound = run_bind4(ctx, &left, z);
        let right_bound = run_bind4(ctx, &right, z);
        let terminal_claim = left_bound
            .iter()
            .zip(&right_bound)
            .fold(Fr::from_u64(0), |acc, (&l, &r)| acc + l * r);
        assert_eq!(q.evaluate(z), terminal_claim);

        // This degree-4 perturbation vanishes at all four digit nodes, so the
        // claim check still passes; the random-point terminal check must bind it.
        let c = seeded_frs(0x4d, 1)[0];
        let mut tampered_coefficients = q.coefficients().to_vec();
        tampered_coefficients.resize(7, Fr::from_u64(0));
        for (coefficient, delta) in tampered_coefficients.iter_mut().zip([0i64, -6, 11, -6, 1]) {
            *coefficient += if delta < 0 {
                -c * Fr::from_u64(delta.unsigned_abs())
            } else {
                c * Fr::from_u64(delta as u64)
            };
        }
        let tampered = UnivariatePoly::new(tampered_coefficients);
        let tampered_sum = (0..4)
            .map(|point| tampered.evaluate(Fr::from_u64(point)))
            .fold(Fr::from_u64(0), |acc, value| acc + value);
        assert_eq!(tampered_sum, current_claim);
        assert_ne!(tampered.evaluate(z), terminal_claim);
    }

    #[test]
    fn device_bind_eval_matches_host() {
        let _lock = gpu_lock();
        let ctx = MetalContext::global().expect("metal context");
        let n_in = 1usize << 16;
        let mut a = edge_frs();
        a.extend(seeded_frs(6, n_in - a.len()));
        let n_out = n_in / 2;
        let r = seeded_frs(7, 1)[0];

        // Degree-3 shape {0,2,3}, plus single-point and 4-point sets to
        // exercise the runtime point count.
        let point_sets: [&[u64]; 3] = [&[0, 2, 3], &[5], &[0, 2, 3, 4]];
        for raw_points in point_sets {
            let points: Vec<Fr> = raw_points.iter().map(|&t| Fr::from_u64(t)).collect();
            let num_tgs = n_out.div_ceil(THREADGROUP_SIZE);

            let mut params = vec![n_out as u32, points.len() as u32, num_tgs as u32];
            params.extend_from_slice(&fr_to_u32_limbs(r));
            for j in 0..MAX_EVAL_POINTS {
                let t = points.get(j).copied().unwrap_or_else(|| Fr::from_u64(0));
                params.extend_from_slice(&fr_to_u32_limbs(t));
            }

            let a_buf = ctx.wrap_slice(fr_as_u32s(&a)).unwrap();
            let out_buf = ctx.alloc_u32s(n_out * FR_U32_LIMBS).unwrap();
            let partials_buf = ctx
                .alloc_u32s(points.len() * num_tgs * FR_U32_LIMBS)
                .unwrap();
            ctx.run_once(
                KernelId::FrBindEval,
                &params,
                &[&a_buf, &out_buf, &partials_buf],
                n_out,
            )
            .unwrap();

            let device_out = read_frs(&out_buf, n_out);
            assert_eq!(device_out, host_bind(&a, r), "bind output mismatch");

            let partials = read_frs(&partials_buf, points.len() * num_tgs);
            let device_sums: Vec<Fr> = partials
                .chunks_exact(num_tgs)
                .map(|tg_sums| tg_sums.iter().fold(Fr::from_u64(0), |acc, v| acc + *v))
                .collect();
            assert_eq!(
                device_sums,
                host_bind_eval_sums(&a, &points),
                "partial sums mismatch for points {raw_points:?}"
            );
        }
    }

    #[test]
    fn wrap_slice_nocopy_eligibility() {
        let _lock = gpu_lock();
        let ctx = MetalContext::global().expect("metal context");

        // PageAlignedVec: no-copy at any length via its capacity guarantee.
        let small = PageAlignedVec::<u32>::from_elem(7, 100);
        assert!(!small.device_buffer(ctx).unwrap().was_copied());

        // Large Vec allocations (malloc large zone) are page-aligned and
        // page-granular — the documented no-copy invariant.
        let big = vec![0u64; 1 << 16]; // 512 KiB
        assert!(ctx.wrap_slice_nocopy(&big).is_some());
        assert!(!ctx.wrap_slice(&big).unwrap().was_copied());

        // A small heap slice is (at best) 16-byte aligned: declined, copy
        // fallback engages.
        let tiny = vec![1u32; 16];
        assert!(ctx.wrap_slice_nocopy(&tiny).is_none());
        assert!(ctx.wrap_slice(&tiny).unwrap().was_copied());

        // A large-but-unaligned view: offset into a large allocation.
        let offset = &big[1..];
        if !(offset.as_ptr() as usize).is_multiple_of(super::super::PAGE_SIZE) {
            assert!(ctx.wrap_slice_nocopy(offset).is_none());
        }
    }

    /// No-copy round trip through a real kernel: results written by the GPU
    /// land in host memory with zero copies either way.
    #[test]
    fn nocopy_bind_writes_host_memory() {
        let _lock = gpu_lock();
        let ctx = MetalContext::global().expect("metal context");
        let n_in = 1usize << 12;
        let n_out = n_in / 2;
        let input = PageAlignedVec::from_slice(&seeded_frs(8, n_in));
        let mut output = PageAlignedVec::from_elem(Fr::from_u64(0), n_out);
        let r = seeded_frs(9, 1)[0];

        {
            let a_buf = input.device_buffer(ctx).unwrap();
            let out_buf = output.device_buffer_mut(ctx).unwrap();
            assert!(!a_buf.was_copied() && !out_buf.was_copied());
            ctx.run_once(
                KernelId::FrBind,
                &bind_params(n_out, r),
                &[&a_buf, &out_buf],
                n_out,
            )
            .unwrap();
        }
        assert_eq!(&*output, host_bind(&input, r).as_slice());
    }

    #[test]
    fn metal_gate_env_convention() {
        // nextest gives each test its own process, so env mutation is safe.
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::remove_var("JOLT_METAL_MIN_TERMS");
        assert!(!metal_gate("bind", super::super::DEFAULT_MIN_TERMS - 1));
        assert!(metal_gate("bind", super::super::DEFAULT_MIN_TERMS));

        std::env::set_var("JOLT_METAL_MIN_TERMS", "100");
        assert!(metal_gate("bind", 100));
        assert!(!metal_gate("bind", 99));

        // Per-slot override wins over the global; kind is normalized.
        std::env::set_var("JOLT_METAL_MIN_TERMS_RAM_RW", "10");
        assert!(metal_gate("ram-rw", 10));
        assert!(!metal_gate("ram-rw", 9));
        assert!(!metal_gate("bind", 99));

        std::env::set_var("JOLT_METAL_DISABLE", "1");
        assert!(!metal_gate("bind", usize::MAX));
        std::env::set_var("JOLT_METAL_DISABLE", "0");
        assert!(metal_gate("bind", 100));
    }

    /// The constructor prewarms the global context and fails closed.
    #[test]
    fn backend_metal_constructs() {
        let _lock = gpu_lock();
        let backend = JoltBackend::<Fr, DoryScheme>::metal().expect("metal backend");
        // Zero slots overwritten in W1: a session opens like any other.
        let _session = backend.begin_proof();
    }
}
