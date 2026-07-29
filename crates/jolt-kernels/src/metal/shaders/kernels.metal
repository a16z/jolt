// Compute kernels over BN254 Fr arrays (see fr.metal for the arithmetic).
//
// Buffer convention (mirrored by `metal::runtime::ComputePass::dispatch`):
// data buffers occupy [[buffer(0..k)]] in argument order, the parameter
// struct is bound with setBytes at the next index. All parameter structs are
// sequences of `uint` (field elements as FR_LIMBS little-endian u32 limbs),
// so the Rust side builds them as flat `[u32]` slices with no padding.

struct ElemwiseParams {
    uint n;
};

struct PowParams {
    uint n;
    uint k;
};

struct BindParams {
    uint n_out;
    uint r[FR_LIMBS];
};

struct BindEvalParams {
    uint n_out;
    uint num_points;
    uint num_tgs;
    uint r[FR_LIMBS];
    uint points[FR_LIMBS * JK_MAX_EVAL_POINTS];
};

// Empty kernel for dispatch-latency measurement.
kernel void jk_noop() {}

kernel void jk_fr_mul(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant ElemwiseParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    fr_store(out, gid, fr_mont_mul(fr_load(a, gid), fr_load(b, gid)));
}

kernel void jk_fr_add(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant ElemwiseParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    fr_store(out, gid, fr_add(fr_load(a, gid), fr_load(b, gid)));
}

kernel void jk_fr_sub(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant ElemwiseParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    fr_store(out, gid, fr_sub(fr_load(a, gid), fr_load(b, gid)));
}

// out[i] = a[i]^(2^k): k dependent Montgomery squarings per element.
// Compute-bound probe — one load, one store, k muls.
kernel void jk_fr_pow2k(
    device const uint* a [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant PowParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    Fr256 x = fr_load(a, gid);
    for (uint i = 0; i < p.k; i++) {
        x = fr_mont_mul(x, x);
    }
    fr_store(out, gid, x);
}

// The sumcheck fold primitive: out[i] = a[2i] + r·(a[2i+1] − a[2i]).
kernel void jk_fr_bind(
    device const uint* a [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant BindParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n_out) {
        return;
    }
    Fr256 lo = fr_load(a, 2 * gid);
    Fr256 diff = fr_sub(fr_load(a, 2 * gid + 1), lo);
    Fr256 r = fr_load_const(p.r, 0);
    fr_store(out, gid, fr_add(lo, fr_mont_mul(r, diff)));
}

// Fused fold + round-poly evaluation: the bind above, plus per-threadgroup
// partial sums of v_j(i) = a[2i] + t_j·(a[2i+1] − a[2i]) for each runtime
// eval point t_j. One kernel serves every point set — t is data, so the
// shape stays uniform (t = 0 still runs its multiply). Partials are written
// point-major: partials[j * num_tgs + tg], reduced on the host.
kernel void jk_fr_bind_eval(
    device const uint* a [[buffer(0)]],
    device uint* out [[buffer(1)]],
    device uint* partials [[buffer(2)]],
    constant BindEvalParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];

    // Inactive lanes contribute zero to every partial sum but still execute
    // the uniform arithmetic below (no early return: they must reach the
    // threadgroup barriers).
    bool active = gid < p.n_out;
    Fr256 lo = fr_zero();
    Fr256 diff = fr_zero();
    if (active) {
        lo = fr_load(a, 2 * gid);
        diff = fr_sub(fr_load(a, 2 * gid + 1), lo);
        Fr256 r = fr_load_const(p.r, 0);
        fr_store(out, gid, fr_add(lo, fr_mont_mul(r, diff)));
    }

    for (uint j = 0; j < p.num_points; j++) {
        Fr256 t = fr_load_const(p.points, j);
        Fr256 v = fr_add(lo, fr_mont_mul(t, diff));
        fr_store_tg(scratch, lid, v);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = JK_TG_SIZE / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Fr256 s = fr_add(fr_load_tg(scratch, lid), fr_load_tg(scratch, lid + stride));
                fr_store_tg(scratch, lid, s);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0) {
            fr_store(partials, j * p.num_tgs + tg, fr_load_tg(scratch, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
