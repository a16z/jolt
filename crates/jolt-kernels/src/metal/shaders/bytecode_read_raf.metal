// Stage-6b bytecode read+RAF device kernels.

#define JK_BYTECODE_STAGES 5u
#define JK_BYTECODE_MAX_FACTORS 16u

struct BytecodeInitParams {
    uint n;
    uint lo_bits;
    uint in_len;
    uint out_len;
    uint entry[FR_LIMBS];
};

kernel void jk_bytecode_init(
    device const uint* e_hi [[buffer(0)]],
    device const uint* e_lo [[buffer(1)]],
    device const uint* weights [[buffer(2)]],
    device uint* combined [[buffer(3)]],
    constant BytecodeInitParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    uint hi = gid >> p.lo_bits;
    uint lo = gid & (p.in_len - 1u);
    Fr256 value = fr_zero();
    for (uint stage = 0u; stage < JK_BYTECODE_STAGES; stage++) {
        Fr256 eq = fr_mont_mul(fr_load(e_hi, stage * p.out_len + hi),
                               fr_load(e_lo, stage * p.in_len + lo));
        value = fr_add(value, fr_mont_mul(fr_load(weights, stage), eq));
    }
    if (gid == 0u) {
        value = fr_add(value, fr_load_const(p.entry, 0));
    }
    fr_store(combined, gid, value);
}

inline Fr256 jk_bytecode_gather(device const uint* rows, device const uint* table,
                                uint width, uint k_entries, uint shift, uint mask, uint j)
{
    Fr256 sum = fr_zero();
    for (uint off = 0u; off < width; off++) {
        uint mapped = rows[(j * width + off) * 2u + 1u];
        if (mapped != 0xFFFFFFFFu) {
            uint idx = (uint)(((ulong)mapped >> shift) & (ulong)mask);
            sum = fr_add(sum, fr_load(table + off * k_entries * FR_LIMBS, idx));
        }
    }
    return sum;
}

struct BytecodeLazyParams {
    uint groups;
    uint do_bind;
    uint num_tgs;
    uint width;
    uint num_ra;
    uint k_entries;
    uint mask;
    uint len;
    uint r[FR_LIMBS];
};

kernel void jk_bytecode_lazy_round(
    device const uint* rows [[buffer(0)]],
    device const uint* shifts [[buffer(1)]],
    device const uint* tables [[buffer(2)]],
    device const uint* cur [[buffer(3)]],
    device uint* nxt [[buffer(4)]],
    device uint* partials [[buffer(5)]],
    constant BytecodeLazyParams& p [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    bool bind = p.do_bind != 0u;
    Fr256 r = fr_load_const(p.r, 0);
    Fr256 evals[JK_BYTECODE_MAX_FACTORS];
    Fr256 steps[JK_BYTECODE_MAX_FACTORS];
    Fr256 lo[JK_BYTECODE_MAX_FACTORS];
    Fr256 c_lo, c_hi;
    jk_round_pair(cur, nxt, bind, r, gid, active, c_lo, c_hi);
    lo[0] = c_lo;
    evals[0] = c_hi;
    steps[0] = fr_sub(c_hi, c_lo);
    uint per_poly = p.width * p.k_entries * FR_LIMBS;
    for (uint i = 0u; i < p.num_ra; i++) {
        Fr256 ra_lo = fr_zero();
        Fr256 ra_hi = fr_zero();
        if (active) {
            device const uint* table = tables + i * per_poly;
            ra_lo = jk_bytecode_gather(rows, table, p.width, p.k_entries, shifts[i], p.mask,
                                        2u * gid);
            ra_hi = jk_bytecode_gather(rows, table, p.width, p.k_entries, shifts[i], p.mask,
                                        2u * gid + 1u);
        }
        lo[i + 1u] = ra_lo;
        evals[i + 1u] = ra_hi;
        steps[i + 1u] = fr_sub(ra_hi, ra_lo);
    }
    uint factors = p.num_ra + 1u;
    Fr256 product = lo[0];
    for (uint factor = 1u; factor < factors; factor++) {
        product = fr_mont_mul(product, lo[factor]);
    }
    jk_tg_sum(scratch, lid, tg, product, partials, 0u, p.num_tgs);
    for (uint lane = 1u; lane < factors; lane++) {
        for (uint factor = 0u; factor < factors; factor++) {
            evals[factor] = fr_add(evals[factor], steps[factor]);
        }
        product = evals[0];
        for (uint factor = 1u; factor < factors; factor++) {
            product = fr_mont_mul(product, evals[factor]);
        }
        jk_tg_sum(scratch, lid, tg, product, partials, lane, p.num_tgs);
    }
}

struct BytecodeAdoptParams {
    uint len;
    uint num_ra;
    uint k_entries;
    uint mask;
    uint old_len;
    uint r[FR_LIMBS];
};

kernel void jk_bytecode_adopt(
    device const uint* rows [[buffer(0)]],
    device const uint* shifts [[buffer(1)]],
    device const uint* tables [[buffer(2)]],
    device const uint* combined [[buffer(3)]],
    device uint* out [[buffer(4)]],
    constant BytecodeAdoptParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint factors = p.num_ra + 1u;
    if (gid >= factors * p.len) {
        return;
    }
    uint factor = gid / p.len;
    uint j = gid - factor * p.len;
    Fr256 value;
    if (factor == 0u) {
        Fr256 lo = fr_load(combined, 2u * j);
        value = fr_add(lo, fr_mont_mul(fr_load_const(p.r, 0),
                                      fr_sub(fr_load(combined, 2u * j + 1u), lo)));
    } else {
        uint ra = factor - 1u;
        device const uint* table = tables + ra * 8u * p.k_entries * FR_LIMBS;
        value = jk_bytecode_gather(rows, table, 8u, p.k_entries, shifts[ra], p.mask, j);
    }
    ulong out_word = ((ulong)factor * (ulong)p.len + (ulong)j) * (ulong)FR_LIMBS;
    fr_store(out + out_word, 0u, value);
}

struct BytecodeDenseParams {
    uint groups;
    uint do_bind;
    uint num_tgs;
    uint factors;
    uint len;
    uint r[FR_LIMBS];
};

kernel void jk_bytecode_dense_round(
    device const uint* cur [[buffer(0)]],
    device uint* nxt [[buffer(1)]],
    device uint* partials [[buffer(2)]],
    constant BytecodeDenseParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    bool bind = p.do_bind != 0u;
    Fr256 r = fr_load_const(p.r, 0);
    Fr256 lo[JK_BYTECODE_MAX_FACTORS];
    Fr256 evals[JK_BYTECODE_MAX_FACTORS];
    Fr256 steps[JK_BYTECODE_MAX_FACTORS];
    for (uint factor = 0u; factor < p.factors; factor++) {
        ulong cur_word = (ulong)factor * (ulong)p.len * (ulong)FR_LIMBS;
        ulong nxt_word = (ulong)factor * (ulong)(p.len >> 1) * (ulong)FR_LIMBS;
        Fr256 factor_lo, factor_hi;
        jk_round_pair(cur + cur_word, nxt + nxt_word, bind, r, gid, active,
                      factor_lo, factor_hi);
        lo[factor] = factor_lo;
        evals[factor] = factor_hi;
        steps[factor] = fr_sub(factor_hi, factor_lo);
    }
    Fr256 product = lo[0];
    for (uint factor = 1u; factor < p.factors; factor++) {
        product = fr_mont_mul(product, lo[factor]);
    }
    jk_tg_sum(scratch, lid, tg, product, partials, 0u, p.num_tgs);
    for (uint lane = 1u; lane < p.factors; lane++) {
        for (uint factor = 0u; factor < p.factors; factor++) {
            evals[factor] = fr_add(evals[factor], steps[factor]);
        }
        product = evals[0];
        for (uint factor = 1u; factor < p.factors; factor++) {
            product = fr_mont_mul(product, evals[factor]);
        }
        jk_tg_sum(scratch, lid, tg, product, partials, lane, p.num_tgs);
    }
}

struct BytecodeOffsetProbeParams {
    uint factor;
    uint len;
    uint element;
};

// Geometry-only probe for the production flat-factor rebase. No large
// allocation: expose the 64-bit word offset as two u32 words.
kernel void jk_bytecode_offset_probe(
    device uint* out [[buffer(0)]],
    constant BytecodeOffsetProbeParams& p [[buffer(1)]])
{
    ulong word = ((ulong)p.factor * (ulong)p.len + (ulong)p.element) * (ulong)FR_LIMBS;
    out[0] = (uint)word;
    out[1] = (uint)(word >> 32);
}
