//! W7 probe microbench: M4 integer-multiply rates, small-limb Montgomery,
//! and [0,2p) lazy reduction. Measurement-only — none of these kernels ship;
//! they exist to calibrate (or refute) conversion waves.
//!
//! ```text
//! cargo run --release -p jolt-kernels --example mulrate_microbench --features metal
//! ```
//!
//! Sections (protocol: min over warm passes, sync-bracketed, GPU lock held):
//!
//! - **P1** raw ALU rates from ~10-line dependent-chain kernels (ILP-8):
//!   u32 mul-lo / mulhi / mad, the lo+hi *pair* (one full 32×32→64 product,
//!   both as a `ulong` widening mul and as an explicit `mul`+`mulhi` pair),
//!   u32/u64 add for carry-chain calibration, and 16-bit mul/mad. Turner's
//!   4-vs-8-cycle numbers are M1-era; this pins the M4 roof that decides P2.
//! - **P2** small-limb Montgomery (R1 #3): w=13-bit limbs, n=20, u32 column
//!   accumulation with zero carry logic in the hot loops (ZPrize'23 WebGPU
//!   shape), chained-squaring D3 geometry, head-to-head against the
//!   production 8×u32 CIOS `jk_fr_pow2k` on the same data.
//! - **P3** [0,2p) domain (R1 #6): production-twin CIOS vs a lazy variant
//!   that drops the final conditional subtraction and canonicalizes once per
//!   64-squaring chain.
//!
//! All probe kernels compile into a private library so the production
//! pipeline registry is untouched; the production comparison arm runs the
//! real `KernelId::FrPow2k` through `MetalContext::global()`.

#![expect(
    clippy::print_stdout,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "benchmark harness: report to stdout, fail loudly"
)]

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    bench::run();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("mulrate_microbench requires --features metal on macOS");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod bench {
    use std::ffi::c_void;
    use std::fmt::Write as _;
    use std::ptr::NonNull;
    use std::time::Instant;

    use jolt_field::{CanonicalBytes, Fr, FromPrimitiveInt, MontgomeryConstants, MulPow2};
    use jolt_kernels::metal::testing::{edge_frs, gpu_lock, seeded_frs};
    use jolt_kernels::metal::{fr_as_u32s, KernelId, MetalContext, PageAlignedVec, FR_U32_LIMBS};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary, MTLResourceOptions, MTLSize,
    };

    /// Threadgroup width, matching the production dispatch geometry.
    const TG: usize = 256;

    /// Small-limb parameters: w-bit limbs, n limbs, R' = 2^(w·n).
    const SL_W: u32 = 13;
    const SL_N: usize = 20;
    const SL_MASK: u32 = (1 << SL_W) - 1;

    // ---------------------------------------------------------------- MSL --

    /// The probe shader library. Rate kernels follow one pattern: 8
    /// independent dependent-chains (ILP-8) of the pure op under test, the
    /// loop body unrolled 4× by macro, seeds data-derived so nothing
    /// constant-folds, results stored so nothing dead-code-eliminates.
    /// Odd×odd=odd keeps multiply chains from collapsing to zero (integer
    /// ALU rate is data-oblivious, but live values keep the probe honest).
    fn probe_source() -> String {
        let mut src = String::with_capacity(16 * 1024);

        // Constants for the CIOS twins (production values, probe-local names)
        // and the small-limb domain.
        let p_u32 = <Fr as MontgomeryConstants>::modulus_u32();
        let _ = writeln!(src, "#include <metal_stdlib>\nusing namespace metal;");
        let _ = writeln!(src, "constant uint FRM[8] = {};", limb_array(p_u32));
        let _ = writeln!(src, "constant uint FRINV = {:#010x}u;", Fr::inv32());
        let _ = writeln!(src, "#define SL_W {SL_W}u");
        let _ = writeln!(src, "#define SL_N {SL_N}u");
        let _ = writeln!(src, "#define SL_MASK {SL_MASK:#x}u");
        let _ = writeln!(
            src,
            "constant uint SL_P[SL_N] = {};",
            limb_array(&p_small_limbs())
        );
        let _ = writeln!(src, "constant uint SL_INV = {:#06x}u;", sl_neg_inv());

        src.push_str(RATE_KERNELS);
        src.push_str(SMALL_LIMB_KERNELS);
        src.push_str(CIOS_KERNELS);
        src
    }

    fn limb_array(limbs: &[u32]) -> String {
        let body: Vec<String> = limbs.iter().map(|l| format!("{l:#010x}u")).collect();
        format!("{{ {} }}", body.join(", "))
    }

    /// The modulus p chopped into `SL_N` w-bit little-endian limbs.
    fn p_small_limbs() -> Vec<u32> {
        let p_u32 = <Fr as MontgomeryConstants>::modulus_u32();
        let mut bytes = [0u8; 32];
        for (chunk, limb) in bytes.chunks_exact_mut(4).zip(p_u32) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        chop_bits(&bytes)
    }

    /// Little-endian bit-chop of a 32-byte value into SL_N limbs of SL_W bits.
    fn chop_bits(bytes: &[u8; 32]) -> Vec<u32> {
        (0..SL_N)
            .map(|i| {
                let bit = i * SL_W as usize;
                let byte = bit / 8;
                // 13 bits span at most 3 bytes.
                let window = u32::from(bytes[byte])
                    | (u32::from(*bytes.get(byte + 1).unwrap_or(&0)) << 8)
                    | (u32::from(*bytes.get(byte + 2).unwrap_or(&0)) << 16);
                (window >> (bit % 8)) & SL_MASK
            })
            .collect()
    }

    /// -p^{-1} mod 2^w (Newton iteration; p is odd).
    fn sl_neg_inv() -> u32 {
        let p0 = <Fr as MontgomeryConstants>::modulus_u32()[0];
        let mut inv = 1u32;
        for _ in 0..5 {
            inv = inv.wrapping_mul(2u32.wrapping_sub(p0.wrapping_mul(inv)));
        }
        inv.wrapping_neg() & SL_MASK
    }

    /// P1: pure-op rate kernels. Ops per thread per loop iteration: 32 for
    /// the scalar chains, 16 products for the pair chains, 16 for add64.
    const RATE_KERNELS: &str = r"
struct RateParams { uint iters; uint salt; };

#define SEED8(T, S) \
    T x0 = T((S) * 0x9e3779b9u) | T(1), x1 = T((S) * 0x85ebca6bu) | T(1), \
      x2 = T((S) * 0xc2b2ae35u) | T(1), x3 = T((S) * 0x27d4eb2fu) | T(1), \
      x4 = T((S) * 0x165667b1u) | T(1), x5 = T((S) * 0xd3a2646cu) | T(1), \
      x6 = T((S) * 0xfd7046c5u) | T(1), x7 = T((S) * 0xb55a4f09u) | T(1);

#define R4(S) S S S S

#define RATE_KERNEL(NAME, T, BODY) \
kernel void NAME( \
    device uint* io [[buffer(0)]], \
    constant RateParams& p [[buffer(1)]], \
    uint gid [[thread_position_in_grid]]) \
{ \
    uint s = io[gid] ^ p.salt; \
    SEED8(T, s) \
    for (uint i = 0; i < p.iters; i++) { R4(BODY) } \
    io[gid] = uint(x0^x1^x2^x3^x4^x5^x6^x7); \
}

RATE_KERNEL(pr_rate_mul32, uint,
    x0*=x0; x1*=x1; x2*=x2; x3*=x3; x4*=x4; x5*=x5; x6*=x6; x7*=x7;)
RATE_KERNEL(pr_rate_mulhi32, uint,
    x0=mulhi(x0,x0); x1=mulhi(x1,x1); x2=mulhi(x2,x2); x3=mulhi(x3,x3);
    x4=mulhi(x4,x4); x5=mulhi(x5,x5); x6=mulhi(x6,x6); x7=mulhi(x7,x7);)
RATE_KERNEL(pr_rate_mad32, uint,
    x0=x0*x0+s; x1=x1*x1+s; x2=x2*x2+s; x3=x3*x3+s;
    x4=x4*x4+s; x5=x5*x5+s; x6=x6*x6+s; x7=x7*x7+s;)
RATE_KERNEL(pr_rate_add32, uint,
    x0+=x1; x1+=x2; x2+=x3; x3+=x4; x4+=x5; x5+=x6; x6+=x7; x7+=x0;)
RATE_KERNEL(pr_rate_mul16, ushort,
    x0=ushort(x0*x0); x1=ushort(x1*x1); x2=ushort(x2*x2); x3=ushort(x3*x3);
    x4=ushort(x4*x4); x5=ushort(x5*x5); x6=ushort(x6*x6); x7=ushort(x7*x7);)
RATE_KERNEL(pr_rate_mad16, ushort,
    x0=ushort(x0*x0+ushort(s)); x1=ushort(x1*x1+ushort(s));
    x2=ushort(x2*x2+ushort(s)); x3=ushort(x3*x3+ushort(s));
    x4=ushort(x4*x4+ushort(s)); x5=ushort(x5*x5+ushort(s));
    x6=ushort(x6*x6+ushort(s)); x7=ushort(x7*x7+ushort(s));)

// One full 32×32→64 product per step, both halves consumed (the CIOS shape).
// 4 chains × 4 unrolls = 16 products per loop iteration.
kernel void pr_rate_pair32(
    device uint* io [[buffer(0)]],
    constant RateParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint s = io[gid] ^ p.salt;
    SEED8(uint, s)
    for (uint i = 0; i < p.iters; i++) {
        R4({ ulong w = (ulong)x0*(ulong)x1; x0 = (uint)w; x1 = (uint)(w >> 32); }
           { ulong w = (ulong)x2*(ulong)x3; x2 = (uint)w; x3 = (uint)(w >> 32); }
           { ulong w = (ulong)x4*(ulong)x5; x4 = (uint)w; x5 = (uint)(w >> 32); }
           { ulong w = (ulong)x6*(ulong)x7; x6 = (uint)w; x7 = (uint)(w >> 32); })
    }
    io[gid] = x0^x1^x2^x3^x4^x5^x6^x7;
}

// Same product count via explicit mul + mulhi on the same operands.
kernel void pr_rate_pair32x(
    device uint* io [[buffer(0)]],
    constant RateParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint s = io[gid] ^ p.salt;
    SEED8(uint, s)
    for (uint i = 0; i < p.iters; i++) {
        R4({ uint lo = x0*x1; x1 = mulhi(x0,x1); x0 = lo; }
           { uint lo = x2*x3; x3 = mulhi(x2,x3); x2 = lo; }
           { uint lo = x4*x5; x5 = mulhi(x4,x5); x4 = lo; }
           { uint lo = x6*x7; x7 = mulhi(x6,x7); x6 = lo; })
    }
    io[gid] = x0^x1^x2^x3^x4^x5^x6^x7;
}

// u64 dependent adds: the carry-chain currency of the 8×u32 CIOS.
// 4 chains × 4 unrolls = 16 u64 adds per loop iteration.
kernel void pr_rate_add64(
    device uint* io [[buffer(0)]],
    constant RateParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint s = io[gid] ^ p.salt;
    ulong w = (ulong)s | ((ulong)(s ^ 0x9e3779b9u) << 32);
    ulong z0 = w | 1, z1 = (w ^ 0x85ebca6bu) | 1, z2 = (w ^ 0xc2b2ae35u) | 1,
          z3 = (w ^ 0x27d4eb2fu) | 1;
    for (uint i = 0; i < p.iters; i++) {
        R4(z0 += z1; z1 += z2; z2 += z3; z3 += z0;)
    }
    io[gid] = uint(z0 ^ z1 ^ z2 ^ z3) ^ uint((z0 ^ z1 ^ z2 ^ z3) >> 32);
}
";

    /// P2: w=13 small-limb Montgomery squaring chain (SOS form).
    ///
    /// Value domain: [0, 2p), Montgomery radix R' = 2^260. Inputs ≤ 2p give
    /// T = (x² + Σmᵢp·2^(13i))/R' ≤ 4p²/R' + p < 1.05p, so the chain is
    /// self-stabilizing with no conditional subtraction anywhere.
    ///
    /// Column headroom (the whole point of w=13): every product < 2^26; a
    /// column accumulates ≤ 20 mul products + ≤ 20 reduction products + a
    /// ≤ 2^19 carry < 2^31.4, so u32 accumulators never overflow and the hot
    /// loops are pure 32-bit mads — zero carry logic. The generic (i,j)
    /// product grid is kept (no squaring shortcut) to mirror the production
    /// `fr_mont_mul(x, x)` chain it is benchmarked against.
    const SMALL_LIMB_KERNELS: &str = r"
struct SlPowParams { uint n; uint k; };

inline void sl13_mont_sq(thread uint* x) {
    uint t[2u * SL_N];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2u * SL_N; i++) {
        t[i] = 0u;
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < SL_N; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SL_N; j++) {
            t[i + j] += x[i] * x[j];
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < SL_N; i++) {
        uint m = ((t[i] & SL_MASK) * SL_INV) & SL_MASK;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SL_N; j++) {
            t[i + j] += m * SL_P[j];
        }
        t[i + 1] += t[i] >> SL_W;
    }
    uint c = 0u;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < SL_N; j++) {
        uint v = t[SL_N + j] + c;
        x[j] = v & SL_MASK;
        c = v >> SL_W;
    }
}

kernel void pr_sl13_pow2k(
    device const uint* a [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant SlPowParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    uint x[SL_N];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < SL_N; j++) {
        x[j] = a[gid * SL_N + j];
    }
    for (uint i = 0; i < p.k; i++) {
        sl13_mont_sq(x);
    }
    #pragma clang loop unroll(full)
    for (uint j = 0; j < SL_N; j++) {
        out[gid * SL_N + j] = x[j];
    }
}
";

    /// P3: 8×u32 CIOS twins. `canon` mirrors the production `fr_mont_mul`
    /// (final branchless trial subtraction, canonical in/out); `lazy` drops
    /// that subtraction entirely — sound for BN254 because p < 2^254 = R/4
    /// gives T = (ab + mp)/R < 4p²/R + p < 2p for inputs < 2p (Walter) — and
    /// canonicalizes once after the whole squaring chain.
    const CIOS_KERNELS: &str = r"
struct CiosPowParams { uint n; uint k; };

struct Cios8 { uint v[8]; };

inline Cios8 cios_mul_core(Cios8 a, Cios8 b, thread uint* top) {
    uint t[10];
    for (uint i = 0; i < 10u; i++) {
        t[i] = 0u;
    }
    for (uint i = 0; i < 8u; i++) {
        ulong carry = 0;
        for (uint j = 0; j < 8u; j++) {
            ulong cur = (ulong)t[j] + (ulong)a.v[i] * (ulong)b.v[j] + carry;
            t[j] = (uint)cur;
            carry = cur >> 32;
        }
        ulong cur = (ulong)t[8] + carry;
        t[8] = (uint)cur;
        t[9] = (uint)(cur >> 32);

        uint m = t[0] * FRINV;
        cur = (ulong)t[0] + (ulong)m * (ulong)FRM[0];
        carry = cur >> 32;
        for (uint j = 1; j < 8u; j++) {
            cur = (ulong)t[j] + (ulong)m * (ulong)FRM[j] + carry;
            t[j - 1] = (uint)cur;
            carry = cur >> 32;
        }
        cur = (ulong)t[8] + carry;
        t[7] = (uint)cur;
        t[8] = t[9] + (uint)(cur >> 32);
        t[9] = 0u;
    }
    *top = t[8];
    Cios8 r;
    for (uint i = 0; i < 8u; i++) {
        r.v[i] = t[i];
    }
    return r;
}

// Branchless conditional subtraction of p (production tail).
inline Cios8 cios_cond_sub(Cios8 sum, uint top) {
    Cios8 diff;
    ulong borrow = 0;
    for (uint i = 0; i < 8u; i++) {
        ulong d = (ulong)sum.v[i] - (ulong)FRM[i] - borrow;
        diff.v[i] = (uint)d;
        borrow = (d >> 32) & 1u;
    }
    bool take_diff = (top != 0u) || (borrow == 0);
    uint mask = take_diff ? 0xffffffffu : 0u;
    Cios8 r;
    for (uint i = 0; i < 8u; i++) {
        r.v[i] = (diff.v[i] & mask) | (sum.v[i] & ~mask);
    }
    return r;
}

inline Cios8 cios_load(device const uint* p, uint idx) {
    Cios8 r;
    for (uint i = 0; i < 8u; i++) {
        r.v[i] = p[idx * 8u + i];
    }
    return r;
}

inline void cios_store(device uint* p, uint idx, Cios8 x) {
    for (uint i = 0; i < 8u; i++) {
        p[idx * 8u + i] = x.v[i];
    }
}

kernel void pr_pow2k_canon(
    device const uint* a [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant CiosPowParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    Cios8 x = cios_load(a, gid);
    for (uint i = 0; i < p.k; i++) {
        uint top;
        x = cios_cond_sub(cios_mul_core(x, x, &top), top);
    }
    cios_store(out, gid, x);
}

kernel void pr_pow2k_lazy(
    device const uint* a [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant CiosPowParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    Cios8 x = cios_load(a, gid);
    for (uint i = 0; i < p.k; i++) {
        uint top;
        x = cios_mul_core(x, x, &top);
    }
    cios_store(out, gid, cios_cond_sub(x, 0u));
}
";

    // ------------------------------------------------------------ harness --

    struct Probe {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        library: Retained<ProtocolObject<dyn MTLLibrary>>,
    }

    type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
    type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

    impl Probe {
        fn new(source: &str) -> Self {
            let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(source), None)
                .unwrap_or_else(|e| panic!("probe MSL compile: {}", e.localizedDescription()));
            let queue = device.newCommandQueue().expect("command queue");
            Self {
                device,
                queue,
                library,
            }
        }

        fn pipeline(&self, name: &str) -> Pipeline {
            let function = self
                .library
                .newFunctionWithName(&NSString::from_str(name))
                .unwrap_or_else(|| panic!("missing kernel {name}"));
            self.device
                .newComputePipelineStateWithFunction_error(&function)
                .unwrap_or_else(|e| panic!("pipeline {name}: {}", e.localizedDescription()))
        }

        fn buffer_from(&self, words: &[u32]) -> Buffer {
            let bytes: NonNull<c_void> = NonNull::from(&words[0]).cast();
            // SAFETY: `bytes` points at `words.len() * 4` readable bytes;
            // Metal copies them into a fresh shared-storage buffer.
            unsafe {
                self.device.newBufferWithBytes_length_options(
                    bytes,
                    words.len() * 4,
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .expect("buffer")
        }
    }

    fn read_buffer(buffer: &Buffer, words: usize) -> Vec<u32> {
        // SAFETY: shared-storage buffer of at least `words * 4` bytes, read
        // after waitUntilCompleted.
        unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u32>(), words) }
            .to_vec()
    }

    impl Probe {
        /// One synchronous dispatch: encode, commit, wait.
        fn run(&self, pipeline: &Pipeline, params: &[u32], buffers: &[&Buffer], threads: usize) {
            let cb = self.queue.commandBuffer().expect("command buffer");
            let encoder = cb.computeCommandEncoder().expect("encoder");
            encoder.setComputePipelineState(pipeline);
            for (index, buffer) in buffers.iter().enumerate() {
                // SAFETY: live MTLBuffer, index in range for the pipeline.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
                }
            }
            if !params.is_empty() {
                let bytes: NonNull<c_void> = NonNull::from(&params[0]).cast();
                // SAFETY: `bytes` points at `params.len() * 4` readable
                // bytes; Metal copies them during this call.
                unsafe {
                    encoder.setBytes_length_atIndex(bytes, size_of_val(params), buffers.len());
                }
            }
            let groups = MTLSize {
                width: threads.div_ceil(TG).max(1),
                height: 1,
                depth: 1,
            };
            let per_group = MTLSize {
                width: TG,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, per_group);
            encoder.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            assert_eq!(
                cb.status(),
                MTLCommandBufferStatus::Completed,
                "dispatch failed"
            );
        }
    }

    /// Minimum of `passes` timed runs of `f`, after one warm pass.
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

    // ------------------------------------------------------------ sections --

    pub fn run() {
        let _lock = gpu_lock();
        let ctx = MetalContext::global().expect("metal context");
        println!("device: {}", ctx.device_name());
        println!("protocol: min over warm passes, sync-bracketed, GPU otherwise idle\n");

        let probe = Probe::new(&probe_source());

        let mul32 = p1_rates(&probe);
        let (prod_rate, sl_rate) = p2_small_limb(&probe, ctx);
        p3_lazy_domain(&probe, ctx, prod_rate);

        println!("== verdict inputs ==");
        println!(
            "production CIOS mont-mul: {prod_rate:.2} Gmul/s = {:.0} G 32×32→64 products/s (128/mul)",
            prod_rate * 128.0
        );
        println!(
            "small-limb w=13: {sl_rate:.2} Gmul/s → {:.2}× production",
            sl_rate / prod_rate
        );
        println!(
            "mul32 roof: {mul32:.0} G op/s (see P1 table for the pair roof the CIOS sits under)"
        );
    }

    /// P1: the rates table. Returns the mul32 rate for the summary.
    fn p1_rates(probe: &Probe) -> f64 {
        println!("== P1: M4 integer ALU rates (dependent chains, ILP-8, 2^18 threads) ==");
        let threads = 1usize << 18;
        let iters = 1024u32;
        let seeds: Vec<u32> = (0..threads as u32)
            .map(|i| i.wrapping_mul(0x9e37_79b9))
            .collect();

        // (kernel, ops per thread per iteration, unit)
        let kernels: &[(&str, u32, &str)] = &[
            ("pr_rate_mul32", 32, "mul32 (lo)"),
            ("pr_rate_mulhi32", 32, "mulhi32"),
            ("pr_rate_mad32", 32, "mad32 (a*b+c)"),
            ("pr_rate_pair32", 16, "pair32 (ulong widen)"),
            ("pr_rate_pair32x", 16, "pair32 (mul+mulhi)"),
            ("pr_rate_add32", 32, "add32"),
            ("pr_rate_add64", 16, "add64"),
            ("pr_rate_mul16", 32, "mul16"),
            ("pr_rate_mad16", 32, "mad16"),
        ];

        let mut mul32_rate = 0.0f64;
        let mut rows = Vec::new();
        for &(name, ops_per_iter, label) in kernels {
            let pipeline = probe.pipeline(name);
            let io = probe.buffer_from(&seeds);
            let time = |it: u32| {
                min_secs(7, || {
                    probe.run(&pipeline, &[it, 0xa5a5_5a5a], &[&io], threads);
                })
            };
            let full = time(iters);
            let half = time(iters / 2);
            let scale = full / half;
            let ops = threads as f64 * f64::from(iters) * f64::from(ops_per_iter);
            let rate = ops / full / 1e9;
            if name == "pr_rate_mul32" {
                mul32_rate = rate;
            }
            let flag = if !(1.6..=2.6).contains(&scale) {
                "  ⚠ non-linear (loop folded?)"
            } else {
                ""
            };
            rows.push((label, rate, full, scale, flag));
        }

        let base = rows[0].1;
        println!(
            "{:<22} {:>10} {:>12} {:>10}",
            "op", "G op/s", "vs mul32", "iters×2"
        );
        for (label, rate, _t, scale, flag) in &rows {
            println!(
                "{label:<22} {rate:>10.1} {:>11.2}× {scale:>9.2}×{flag}",
                rate / base
            );
        }
        println!();
        mul32_rate
    }

    /// P2: small-limb Montgomery vs the production CIOS kernel, same data,
    /// same chained-squaring geometry. Returns (production, small-limb) in
    /// Gmul/s.
    fn p2_small_limb(probe: &Probe, ctx: &MetalContext) -> (f64, f64) {
        println!("== P2: small-limb Montgomery (w=13, n=20, R'=2^260) vs production CIOS ==");
        let n = 1usize << 18;
        let k = 64u32;
        let muls = n as f64 * f64::from(k);

        // Shared inputs: seeded field elements (+ edge cases up front).
        let mut inputs = edge_frs();
        inputs.extend(seeded_frs(7, n - inputs.len()));

        // Production arm: the real pipeline, same D3 shape.
        let prod_in = PageAlignedVec::from_slice(&inputs);
        let mut prod_out = PageAlignedVec::from_elem(Fr::from_u64(0), n);
        let prod = {
            let a = prod_in.device_buffer(ctx).unwrap();
            let out = prod_out.device_buffer_mut(ctx).unwrap();
            min_secs(7, || {
                ctx.run_once(KernelId::FrPow2k, &[n as u32, k], &[&a, &out], n)
                    .unwrap();
            })
        };
        let prod_rate = muls / prod / 1e9;
        println!(
            "production jk_fr_pow2k (8×u32 CIOS): {:.2} ms → {prod_rate:.2} Gmul/s",
            prod * 1e3
        );

        // Small-limb arm: convert values into the R'=2^260 domain.
        let sl_in: Vec<u32> = inputs.iter().flat_map(|x| to_small_limbs(*x)).collect();
        let a = probe.buffer_from(&sl_in);
        let out = probe.buffer_from(&vec![0u32; n * SL_N]);
        let pipeline = probe.pipeline("pr_sl13_pow2k");
        let sl = min_secs(7, || {
            probe.run(&pipeline, &[n as u32, k], &[&a, &out], n);
        });
        let sl_rate = muls / sl / 1e9;
        println!(
            "small-limb pr_sl13_pow2k (u32 columns): {:.2} ms → {sl_rate:.2} Gmul/s",
            sl * 1e3
        );
        println!(
            "ratio small-limb/production: {:.2}×  (mads/mul: {} vs 128 wide products/mul)",
            sl_rate / prod_rate,
            2 * SL_N * SL_N + SL_N,
        );

        // Correctness: reassembled device values must equal x^(2^k)·2^260.
        let sl_out = read_buffer(&out, n * SL_N);
        let check = 256.min(n);
        for (i, (x, limbs)) in inputs
            .iter()
            .zip(sl_out.chunks_exact(SL_N))
            .enumerate()
            .take(check)
        {
            let mut expect = *x;
            for _ in 0..k {
                expect = expect * expect;
            }
            assert_eq!(
                from_small_limbs(limbs),
                shift_to_sl_domain(expect),
                "small-limb mismatch at element {i}"
            );
        }
        println!("correctness: first {check} elements (incl. edge cases) match host chain ✓\n");
        (prod_rate, sl_rate)
    }

    /// P3: canonical-twin CIOS vs the [0,2p) lazy variant.
    fn p3_lazy_domain(probe: &Probe, ctx: &MetalContext, prod_rate: f64) {
        println!("== P3: [0,2p) lazy reduction (drop final conditional subtraction) ==");
        let n = 1usize << 18;
        let k = 64u32;
        let muls = n as f64 * f64::from(k);

        let mut inputs = edge_frs();
        inputs.extend(seeded_frs(11, n - inputs.len()));
        let in_words = fr_as_u32s(&inputs).to_vec();
        let a = probe.buffer_from(&in_words);
        let out_canon = probe.buffer_from(&vec![0u32; n * FR_U32_LIMBS]);
        let out_lazy = probe.buffer_from(&vec![0u32; n * FR_U32_LIMBS]);

        let canon_pipe = probe.pipeline("pr_pow2k_canon");
        let lazy_pipe = probe.pipeline("pr_pow2k_lazy");
        let canon = min_secs(7, || {
            probe.run(&canon_pipe, &[n as u32, k], &[&a, &out_canon], n);
        });
        let lazy = min_secs(7, || {
            probe.run(&lazy_pipe, &[n as u32, k], &[&a, &out_lazy], n);
        });

        let canon_rate = muls / canon / 1e9;
        let lazy_rate = muls / lazy / 1e9;
        println!(
            "canon twin: {:.2} ms → {canon_rate:.2} Gmul/s ({:+.1}% vs production kernel — twin sanity)",
            canon * 1e3,
            (canon_rate / prod_rate - 1.0) * 100.0
        );
        println!(
            "lazy [0,2p): {:.2} ms → {lazy_rate:.2} Gmul/s → {:+.1}% vs canon twin",
            lazy * 1e3,
            (lazy_rate / canon_rate - 1.0) * 100.0
        );

        // Correctness: canon twin must equal the production kernel bytes;
        // lazy (canonicalized once at the tail) must equal canon.
        let prod_in = PageAlignedVec::from_slice(&inputs);
        let mut prod_out = PageAlignedVec::from_elem(Fr::from_u64(0), n);
        {
            let a = prod_in.device_buffer(ctx).unwrap();
            let out = prod_out.device_buffer_mut(ctx).unwrap();
            ctx.run_once(KernelId::FrPow2k, &[n as u32, k], &[&a, &out], n)
                .unwrap();
        }
        let canon_words = read_buffer(&out_canon, n * FR_U32_LIMBS);
        let lazy_words = read_buffer(&out_lazy, n * FR_U32_LIMBS);
        assert_eq!(
            canon_words,
            fr_as_u32s(&prod_out),
            "canon twin diverged from production kernel"
        );
        assert_eq!(lazy_words, canon_words, "lazy result not canonical-equal");
        println!("correctness: canon ≡ production bytes, lazy ≡ canon ✓\n");
    }

    // ----------------------------------------------- small-limb conversion --

    /// value·2^260 mod p (mul_pow_2 caps its shift at 255, so split it).
    fn shift_to_sl_domain(x: Fr) -> Fr {
        let half = SL_W as usize * SL_N / 2;
        x.mul_pow_2(half).mul_pow_2(half)
    }

    /// x (as a field value) → 20×13-bit limbs of x·2^260 mod p.
    fn to_small_limbs(x: Fr) -> Vec<u32> {
        let mut bytes = [0u8; 32];
        shift_to_sl_domain(x).to_bytes_le(&mut bytes);
        chop_bits(&bytes)
    }

    /// 20×13-bit limbs (possibly redundant, value < 2p) → field value.
    fn from_small_limbs(limbs: &[u32]) -> Fr {
        limbs.iter().rev().fold(Fr::from_u64(0), |acc, &l| {
            acc.mul_pow_2(SL_W as usize) + Fr::from_u64(u64::from(l))
        })
    }
}
