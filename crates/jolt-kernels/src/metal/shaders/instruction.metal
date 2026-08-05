// Stage-5 instruction read-RAF phase scans (see fr.metal for the field
// arithmetic and optimized/instruction_read_raf.rs for the host twin).
//
// Rows are the repr(C) InstructionCycleRow view: 12 uint words per row —
// lookup index limbs LE at [0..4), PC/RAM columns at [4..8) (unused here),
// table byte and RAF flag packed in word 8 (offsets pinned host-side).
//
// Accumulation spaces: cells fed by plain `u` additions stay in Montgomery
// form; cells fed by scalar products accumulate mont_mul(u, scalar) — the
// RAW-space value u·scalar — and the host multiplies the reduced cell by R
// once (value-space ×2^256), landing on the exact field element the CPU's
// deferred accumulator produces. Fr sums are exact, so any regrouping
// across lanes/simdgroups is byte-neutral after that fix-up.
//
// Conflict resolution: each simdgroup owns a private device-memory bucket
// row (zeroed at kernel start), so races only exist between lanes of one
// simdgroup. A tile whose active lanes share one chunk (the dominant case
// in early phases — high index bits are mostly zero on real traces)
// reduces lane values with a shuffle-xor butterfly and lane 0 adds once;
// mixed-chunk tiles fall back to lanes taking turns (masked serial adds).

// --- u128 / bit helpers -----------------------------------------------------

// (val >> shift) & 255 for a 128-bit value in (lo, hi), shift <= 120.
inline uint jk_chunk8(ulong lo, ulong hi, uint shift) {
    ulong w;
    if (shift == 0u) {
        w = lo;
    } else if (shift < 64u) {
        w = (lo >> shift) | (hi << (64u - shift));
    } else {
        w = hi >> (shift - 64u);
    }
    return (uint)(w & 255ul);
}

// Low `len` bits of (lo, hi), len <= 120.
inline void jk_mask128(ulong lo, ulong hi, uint len, thread ulong& olo, thread ulong& ohi) {
    if (len >= 64u) {
        olo = lo;
        ohi = (len == 64u) ? 0ul : (hi & ((1ul << (len - 64u)) - 1ul));
    } else {
        olo = (len == 0u) ? 0ul : (lo & ((1ul << len) - 1ul));
        ohi = 0ul;
    }
}

inline uint jk_clz64(ulong v) {
    uint hi = (uint)(v >> 32);
    return (hi != 0u) ? clz(hi) : (32u + clz((uint)v));
}

inline uint jk_ctz64(ulong v) {
    uint lo = (uint)v;
    return (lo != 0u) ? ctz(lo) : (32u + ctz((uint)(v >> 32)));
}

// Rust `unbounded_shr`/`unbounded_shl` (zero past the width) and the
// release-mode wrapping shift (amount masked to the width).
inline ulong jk_shr64_unbounded(ulong x, uint s) { return (s >= 64u) ? 0ul : (x >> s); }
inline ulong jk_shl64_unbounded(ulong x, uint s) { return (s >= 64u) ? 0ul : (x << s); }
inline uint jk_shr32_unbounded(uint x, uint s) { return (s >= 32u) ? 0u : (x >> s); }
inline uint jk_shl32_unbounded(uint x, uint s) { return (s >= 32u) ? 0u : (x << s); }
inline uint jk_shl32_wrapping(uint x, uint s) { return x << (s & 31u); }

inline uint jk_bswap32(uint x) {
    return (x << 24) | ((x & 0xFF00u) << 8) | ((x >> 8) & 0xFF00u) | (x >> 24);
}

inline ulong jk_rotr64(ulong v, uint r) { return (v >> r) | (v << (64u - r)); }
inline uint jk_rotr32(uint v, uint r) { return (v >> r) | (v << (32u - r)); }

// Compact the even-position bits of v into its low 32 bits (Morton
// half-extract; the mask cascade mirrors uninterleave_bits in
// jolt-lookup-tables/src/interleave.rs).
inline ulong jk_compact_even(ulong v) {
    v &= 0x5555555555555555ul;
    v = (v | (v >> 1)) & 0x3333333333333333ul;
    v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0Ful;
    v = (v | (v >> 4)) & 0x00FF00FF00FF00FFul;
    v = (v | (v >> 8)) & 0x0000FFFF0000FFFFul;
    v = (v | (v >> 16)) & 0x00000000FFFFFFFFul;
    return v;
}

// LookupBits::uninterleave on a masked 128-bit value: odd positions -> x,
// even positions -> y (both at most 60 bits for len <= 120).
inline void jk_uninterleave(ulong s_lo, ulong s_hi, thread ulong& x, thread ulong& y) {
    ulong xs_lo = (s_lo >> 1) | (s_hi << 63);
    ulong xs_hi = s_hi >> 1;
    x = jk_compact_even(xs_lo) | (jk_compact_even(xs_hi) << 32);
    y = jk_compact_even(s_lo) | (jk_compact_even(s_hi) << 32);
}

// LookupBits::leading_ones for a value of `len` bits, len <= 64.
inline uint jk_leading_ones(ulong v, uint len) {
    if (len == 0u) {
        return 0u;
    }
    return jk_clz64(~(v << (64u - len)));
}

// LookupBits::trailing_zeros: min(value tz, len). jk_ctz64(0) = 64 >= len.
inline uint jk_trailing_zeros(ulong v, uint len) {
    return min(jk_ctz64(v), len);
}

// --- suffix MLEs ------------------------------------------------------------
//
// One case per Suffixes variant, ids = the Rust enum's repr(u8)
// discriminants (declaration order — jolt-lookup-tables
// src/tables/suffixes/mod.rs). The `suffix_probe` parity test pins every
// case against the Rust implementation, so a reordered enum or a drifted
// port fails loudly. XLEN = 64 is baked in (the host slot asserts it).

#define JK_SUF_ONE 0u
#define JK_SUF_AND 1u
#define JK_SUF_ANDNOT 2u
#define JK_SUF_XOR 3u
#define JK_SUF_OR 4u
#define JK_SUF_RIGHT_OPERAND 5u
#define JK_SUF_RIGHT_OPERAND_W 6u
#define JK_SUF_CHANGE_DIVISOR 7u
#define JK_SUF_CHANGE_DIVISOR_W 8u
#define JK_SUF_UPPER_WORD 9u
#define JK_SUF_LOWER_WORD 10u
#define JK_SUF_LOWER_HALF_WORD 11u
#define JK_SUF_LESS_THAN 12u
#define JK_SUF_GREATER_THAN 13u
#define JK_SUF_EQ 14u
#define JK_SUF_LEFT_IS_ZERO 15u
#define JK_SUF_RIGHT_IS_ZERO 16u
#define JK_SUF_LSB 17u
#define JK_SUF_DIV_BY_ZERO 18u
#define JK_SUF_POW2 19u
#define JK_SUF_POW2_W 20u
#define JK_SUF_REV8W 21u
#define JK_SUF_RIGHT_SHIFT_PADDING 22u
#define JK_SUF_RIGHT_SHIFT 23u
#define JK_SUF_RIGHT_SHIFT_HELPER 24u
#define JK_SUF_SIGN_EXTENSION 25u
#define JK_SUF_LEFT_SHIFT 26u
#define JK_SUF_TWO_LSB 27u
#define JK_SUF_SIGN_EXTENSION_UPPER_HALF 28u
#define JK_SUF_SIGN_EXTENSION_RIGHT_OPERAND 29u
#define JK_SUF_RIGHT_SHIFT_W 30u
#define JK_SUF_RIGHT_SHIFT_W_HELPER 31u
#define JK_SUF_LEFT_SHIFT_W_HELPER 32u
#define JK_SUF_LEFT_SHIFT_W 33u
#define JK_SUF_OVERFLOW_BITS_ZERO 34u
#define JK_SUF_XOR_ROT_16 35u
#define JK_SUF_XOR_ROT_24 36u
#define JK_SUF_XOR_ROT_32 37u
#define JK_SUF_XOR_ROT_63 38u
#define JK_SUF_XOR_ROT_W_16 39u
#define JK_SUF_XOR_ROT_W_12 40u
#define JK_SUF_XOR_ROT_W_8 41u
#define JK_SUF_XOR_ROT_W_7 42u

inline ulong jk_suffix_mle(uint id, ulong s_lo, ulong s_hi, uint len) {
    ulong x = 0ul;
    ulong y = 0ul;
    uint yl = len - (len / 2u);
    switch (id) {
        case JK_SUF_ONE:
            return 1ul;
        case JK_SUF_AND:
            jk_uninterleave(s_lo, s_hi, x, y);
            return x & y;
        case JK_SUF_ANDNOT:
            jk_uninterleave(s_lo, s_hi, x, y);
            return x & ~y;
        case JK_SUF_XOR:
            jk_uninterleave(s_lo, s_hi, x, y);
            return x ^ y;
        case JK_SUF_OR:
            jk_uninterleave(s_lo, s_hi, x, y);
            return x | y;
        case JK_SUF_RIGHT_OPERAND:
            jk_uninterleave(s_lo, s_hi, x, y);
            return y;
        case JK_SUF_RIGHT_OPERAND_W:
            jk_uninterleave(s_lo, s_hi, x, y);
            return y & 0xFFFFFFFFul;
        case JK_SUF_CHANGE_DIVISOR:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (((1ul << yl) - 1ul) == y && x == 0ul) ? 1ul : 0ul;
        case JK_SUF_CHANGE_DIVISOR_W: {
            jk_uninterleave(s_lo, s_hi, x, y);
            uint yl2 = min(yl, 32u);
            ulong x32 = x & 0xFFFFFFFFul;
            ulong y32 = y & 0xFFFFFFFFul;
            return (((1ul << yl2) - 1ul) == y32 && x32 == 0ul) ? 1ul : 0ul;
        }
        case JK_SUF_UPPER_WORD:
            return s_hi;
        case JK_SUF_LOWER_WORD:
            return s_lo;
        case JK_SUF_LOWER_HALF_WORD:
            return s_lo & 0xFFFFFFFFul;
        case JK_SUF_LESS_THAN:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (x < y) ? 1ul : 0ul;
        case JK_SUF_GREATER_THAN:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (x > y) ? 1ul : 0ul;
        case JK_SUF_EQ:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (x == y) ? 1ul : 0ul;
        case JK_SUF_LEFT_IS_ZERO:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (x == 0ul) ? 1ul : 0ul;
        case JK_SUF_RIGHT_IS_ZERO:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (y == 0ul) ? 1ul : 0ul;
        case JK_SUF_LSB:
            return (len == 0u) ? 1ul : (s_lo & 1ul);
        case JK_SUF_DIV_BY_ZERO:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (x == 0ul && y == ((1ul << yl) - 1ul)) ? 1ul : 0ul;
        case JK_SUF_POW2:
            // split(log2(XLEN) = 6): shift = the low 6 bits.
            return (len == 0u) ? 1ul : (1ul << (s_lo & 63ul));
        case JK_SUF_POW2_W:
            return (len == 0u) ? 1ul : (1ul << (s_lo & 31ul));
        case JK_SUF_REV8W: {
            uint lo32 = jk_bswap32((uint)s_lo);
            uint hi32 = jk_bswap32((uint)(s_lo >> 32));
            return (ulong)lo32 | ((ulong)hi32 << 32);
        }
        case JK_SUF_RIGHT_SHIFT_PADDING: {
            if (len == 0u) {
                return 1ul;
            }
            uint shift = (uint)(s_lo & 63ul);
            return 1ul << (63u - shift);
        }
        case JK_SUF_RIGHT_SHIFT:
            jk_uninterleave(s_lo, s_hi, x, y);
            return jk_shr64_unbounded(x, jk_trailing_zeros(y, yl));
        case JK_SUF_RIGHT_SHIFT_HELPER:
            jk_uninterleave(s_lo, s_hi, x, y);
            return 1ul << jk_leading_ones(y, yl);
        case JK_SUF_SIGN_EXTENSION: {
            jk_uninterleave(s_lo, s_hi, x, y);
            uint padding = jk_trailing_zeros(y, yl);
            // ((1u128 << 64) - (1u128 << (64 - padding))) as u64
            return (padding == 0u) ? 0ul : (~0ul << (64u - padding));
        }
        case JK_SUF_LEFT_SHIFT:
            jk_uninterleave(s_lo, s_hi, x, y);
            return jk_shl64_unbounded(x & ~y, jk_leading_ones(y, yl));
        case JK_SUF_TWO_LSB:
            return (len == 0u || (s_lo & 3ul) == 0ul) ? 1ul : 0ul;
        case JK_SUF_SIGN_EXTENSION_UPPER_HALF:
            if (len >= 32u) {
                return ((s_lo >> 31) & 1ul) != 0ul ? 0xFFFFFFFF00000000ul : 0ul;
            }
            return 1ul;
        case JK_SUF_SIGN_EXTENSION_RIGHT_OPERAND:
            if (len >= 64u) {
                return ((s_lo >> 62) & 1ul) != 0ul ? 0xFFFFFFFF00000000ul : 0ul;
            }
            return 1ul;
        case JK_SUF_RIGHT_SHIFT_W:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (ulong)jk_shr32_unbounded((uint)x, min(jk_trailing_zeros(y, yl), 32u));
        case JK_SUF_RIGHT_SHIFT_W_HELPER: {
            jk_uninterleave(s_lo, s_hi, x, y);
            uint yl2 = min(yl, 32u);
            ulong y2 = (yl2 == 0u) ? 0ul : (y & ((1ul << yl2) - 1ul));
            return 1ul << jk_leading_ones(y2, yl2);
        }
        case JK_SUF_LEFT_SHIFT_W_HELPER:
            jk_uninterleave(s_lo, s_hi, x, y);
            // Rust `1u32 << leading_ones` in release mode: wrapping shift.
            return (ulong)jk_shl32_wrapping(1u, jk_leading_ones(y, yl));
        case JK_SUF_LEFT_SHIFT_W: {
            jk_uninterleave(s_lo, s_hi, x, y);
            uint yl2 = min(yl, 32u);
            ulong y2 = (yl2 == 0u) ? 0ul : (y & ((1ul << yl2) - 1ul));
            uint x32 = (uint)x & ~(uint)y2;
            return (ulong)jk_shl32_unbounded(x32, jk_leading_ones(y2, yl2));
        }
        case JK_SUF_OVERFLOW_BITS_ZERO:
            return (s_hi == 0ul) ? 1ul : 0ul;
        case JK_SUF_XOR_ROT_16:
            jk_uninterleave(s_lo, s_hi, x, y);
            return jk_rotr64(x ^ y, 16u);
        case JK_SUF_XOR_ROT_24:
            jk_uninterleave(s_lo, s_hi, x, y);
            return jk_rotr64(x ^ y, 24u);
        case JK_SUF_XOR_ROT_32:
            jk_uninterleave(s_lo, s_hi, x, y);
            return jk_rotr64(x ^ y, 32u);
        case JK_SUF_XOR_ROT_63:
            jk_uninterleave(s_lo, s_hi, x, y);
            return jk_rotr64(x ^ y, 63u);
        case JK_SUF_XOR_ROT_W_16:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (ulong)jk_rotr32((uint)x ^ (uint)y, 16u);
        case JK_SUF_XOR_ROT_W_12:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (ulong)jk_rotr32((uint)x ^ (uint)y, 12u);
        case JK_SUF_XOR_ROT_W_8:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (ulong)jk_rotr32((uint)x ^ (uint)y, 8u);
        case JK_SUF_XOR_ROT_W_7:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (ulong)jk_rotr32((uint)x ^ (uint)y, 7u);
        default:
            return 0ul;
    }
}

// --- simdgroup accumulation helpers -----------------------------------------

inline Fr256 jk_fr_from_u64(ulong v) {
    Fr256 r = fr_zero();
    r.v[0] = (uint)v;
    r.v[1] = (uint)(v >> 32);
    return r;
}

inline Fr256 jk_fr_from_u128(ulong lo, ulong hi) {
    Fr256 r = fr_zero();
    r.v[0] = (uint)lo;
    r.v[1] = (uint)(lo >> 32);
    r.v[2] = (uint)hi;
    r.v[3] = (uint)(hi >> 32);
    return r;
}

// Butterfly sum across the simdgroup; every lane ends with the total.
inline Fr256 jk_simd_sum(Fr256 v, uint simd_size) {
    for (uint delta = simd_size / 2u; delta > 0u; delta >>= 1u) {
        Fr256 other;
        for (uint i = 0; i < FR_LIMBS; i++) {
            other.v[i] = simd_shuffle_xor(v.v[i], (ushort)delta);
        }
        v = fr_add(v, other);
    }
    return v;
}

inline void jk_cell_add(device uint* cells, uint cell, Fr256 v) {
    fr_store(cells, cell, fr_add(fr_load(cells, cell), v));
}

// --- kernels ----------------------------------------------------------------

#define JK_IRR_RAF_CELLS (6u * 256u)
#define JK_IRR_SUF_CELLS (8u * 256u)

struct IrrPhaseScanParams {
    uint n;                  // row count
    uint rows_per_sg;
    uint num_sgs;
    uint suffix_len;         // this phase's suffix width (0..120)
    uint prev_shift;         // previous phase's suffix width (condensation)
    uint do_condense;
    uint canonical;          // CANONICAL_INSTRUCTION_ADDRESS
    uint upper_suffix_bits;  // suffix_len.saturating_sub(64)
};

inline Fr256 jk_simd_shuffle_fr(Fr256 v, uint source) {
    Fr256 out;
    for (uint i = 0u; i < FR_LIMBS; i++) {
        out.v[i] = simd_shuffle(v.v[i], (ushort)source);
    }
    return out;
}

// Reduce equal scatter keys inside the simdgroup, then let only each key's
// first lane update device memory. Leaders have distinct cells, so the
// read-modify-writes are independent without 32 lane-turn barriers.
inline void jk_simd_scatter3(
    device uint* cells,
    uint key,
    bool active,
    Fr256 v0,
    Fr256 v1,
    Fr256 v2,
    uint lane,
    uint simd_size)
{
    bool collision = false;
    bool leader = active;
    for (uint source = 0u; source < simd_size; source++) {
        bool source_active = simd_shuffle((uint)active, (ushort)source) != 0u;
        uint source_key = simd_shuffle(key, (ushort)source);
        bool same = source_active && source_key == key && source != lane;
        collision = collision || same;
        leader = leader && !(same && source < lane);
    }
    Fr256 sum0 = fr_zero();
    Fr256 sum1 = fr_zero();
    Fr256 sum2 = fr_zero();
    for (uint source = 0u; source < simd_size; source++) {
        bool source_collision =
            simd_shuffle((uint)(active && collision), (ushort)source) != 0u;
        if (source_collision) {
            uint source_key = simd_shuffle(key, (ushort)source);
            Fr256 source0 = jk_simd_shuffle_fr(v0, source);
            Fr256 source1 = jk_simd_shuffle_fr(v1, source);
            Fr256 source2 = jk_simd_shuffle_fr(v2, source);
            if (collision && source_key == key) {
                sum0 = fr_add(sum0, source0);
                sum1 = fr_add(sum1, source1);
                sum2 = fr_add(sum2, source2);
            }
        }
    }
    if (!active || (collision && !leader)) {
        return;
    }
    if (!collision) {
        sum0 = v0;
        sum1 = v1;
        sum2 = v2;
    }
    uint family_base = (key >> 8u) * 3u * 256u;
    uint chunk = key & 255u;
    jk_cell_add(cells, family_base + chunk, sum0);
    jk_cell_add(cells, family_base + 256u + chunk, sum1);
    jk_cell_add(cells, family_base + 512u + chunk, sum2);
}

// Fused condensation + RAF scan. Per-simdgroup cells, quantity-major:
// [0] shift_half, [1] left, [2] right (interleaved rows);
// [3] shift_full, [4] identity, [5] upper-all-ones (RAF rows);
// cell = quantity * 256 + chunk. Cells 1/2/4 are RAW space (host fix-up).
kernel void jk_irr_phase_scan(
    device const uint* rows [[buffer(0)]],
    device uint* u_evals [[buffer(1)]],
    device const uint* v_prev [[buffer(2)]],
    device uint* partials [[buffer(3)]],
    constant IrrPhaseScanParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    uint sg = gid / simd_size;
    if (sg >= p.num_sgs) {
        return;
    }
    device uint* my = partials + sg * JK_IRR_RAF_CELLS * FR_LIMBS;
    for (uint i = lane; i < JK_IRR_RAF_CELLS * FR_LIMBS; i += simd_size) {
        my[i] = 0u;
    }
    simdgroup_barrier(mem_flags::mem_device);

    uint row_start = sg * p.rows_per_sg;
    uint row_end = min(row_start + p.rows_per_sg, p.n);
    for (uint base = row_start; base < row_end; base += simd_size) {
        uint j = base + lane;
        bool active = j < row_end;
        uint chunk = 0u;
        bool flag = false;
        Fr256 v0 = fr_zero();
        Fr256 v1 = fr_zero();
        Fr256 v2 = fr_zero();
        if (active) {
            device const uint* row = rows + j * 12u;
            ulong lo = (ulong)row[0] | ((ulong)row[1] << 32);
            ulong hi = (ulong)row[2] | ((ulong)row[3] << 32);
            flag = ((row[8] >> 8) & 0xFFu) != 0u;
            Fr256 u = fr_load(u_evals, j);
            if (p.do_condense != 0u) {
                u = fr_mont_mul(u, fr_load(v_prev, jk_chunk8(lo, hi, p.prev_shift)));
                fr_store(u_evals, j, u);
            }
            chunk = jk_chunk8(lo, hi, p.suffix_len);
            ulong s_lo, s_hi;
            jk_mask128(lo, hi, p.suffix_len, s_lo, s_hi);
            v0 = u;
            if (!flag) {
                ulong x, y;
                jk_uninterleave(s_lo, s_hi, x, y);
                v1 = fr_mont_mul(u, jk_fr_from_u64(x));
                v2 = fr_mont_mul(u, jk_fr_from_u64(y));
            } else {
                v1 = fr_mont_mul(u, jk_fr_from_u128(s_lo, s_hi));
                bool upper_ok = (p.canonical != 0u)
                    && (p.upper_suffix_bits == 0u
                        || s_hi == ((1ul << p.upper_suffix_bits) - 1ul));
                v2 = upper_ok ? u : fr_zero();
            }
        }

        // Family split without divergence in the emit structure: RAF rows
        // land in cells 3..5, interleaved rows in 0..2.
        bool tile_uniform = simd_all(active)
            && simd_all(chunk == simd_broadcast_first(chunk))
            && simd_size == 32u;
        if (tile_uniform) {
            uint c = simd_broadcast_first(chunk);
            Fr256 a0 = jk_simd_sum(flag ? fr_zero() : v0, simd_size);
            Fr256 a1 = jk_simd_sum(flag ? fr_zero() : v1, simd_size);
            Fr256 a2 = jk_simd_sum(flag ? fr_zero() : v2, simd_size);
            Fr256 b0 = jk_simd_sum(flag ? v0 : fr_zero(), simd_size);
            Fr256 b1 = jk_simd_sum(flag ? v1 : fr_zero(), simd_size);
            Fr256 b2 = jk_simd_sum(flag ? v2 : fr_zero(), simd_size);
            if (lane == 0u) {
                jk_cell_add(my, 0u * 256u + c, a0);
                jk_cell_add(my, 1u * 256u + c, a1);
                jk_cell_add(my, 2u * 256u + c, a2);
                jk_cell_add(my, 3u * 256u + c, b0);
                jk_cell_add(my, 4u * 256u + c, b1);
                jk_cell_add(my, 5u * 256u + c, b2);
            }
        } else {
            uint key = (flag ? 256u : 0u) + chunk;
            jk_simd_scatter3(my, key, active, v0, v1, v2, lane, simd_size);
        }
    }
}

// Ablation arm for JOLT_IRR_PHASE_SCAN_LEGACY: identical arithmetic and
// schedule, retaining the barrier-serialized scatter this lane replaced.
kernel void jk_irr_phase_scan_legacy(
    device const uint* rows [[buffer(0)]],
    device uint* u_evals [[buffer(1)]],
    device const uint* v_prev [[buffer(2)]],
    device uint* partials [[buffer(3)]],
    constant IrrPhaseScanParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    uint sg = gid / simd_size;
    if (sg >= p.num_sgs) {
        return;
    }
    device uint* my = partials + sg * JK_IRR_RAF_CELLS * FR_LIMBS;
    for (uint i = lane; i < JK_IRR_RAF_CELLS * FR_LIMBS; i += simd_size) {
        my[i] = 0u;
    }
    simdgroup_barrier(mem_flags::mem_device);

    uint row_start = sg * p.rows_per_sg;
    uint row_end = min(row_start + p.rows_per_sg, p.n);
    for (uint base = row_start; base < row_end; base += simd_size) {
        uint j = base + lane;
        bool active = j < row_end;
        uint chunk = 0u;
        bool flag = false;
        Fr256 v0 = fr_zero();
        Fr256 v1 = fr_zero();
        Fr256 v2 = fr_zero();
        if (active) {
            device const uint* row = rows + j * 12u;
            ulong lo = (ulong)row[0] | ((ulong)row[1] << 32);
            ulong hi = (ulong)row[2] | ((ulong)row[3] << 32);
            flag = ((row[8] >> 8) & 0xFFu) != 0u;
            Fr256 u = fr_load(u_evals, j);
            if (p.do_condense != 0u) {
                u = fr_mont_mul(u, fr_load(v_prev, jk_chunk8(lo, hi, p.prev_shift)));
                fr_store(u_evals, j, u);
            }
            chunk = jk_chunk8(lo, hi, p.suffix_len);
            ulong s_lo, s_hi;
            jk_mask128(lo, hi, p.suffix_len, s_lo, s_hi);
            v0 = u;
            if (!flag) {
                ulong x, y;
                jk_uninterleave(s_lo, s_hi, x, y);
                v1 = fr_mont_mul(u, jk_fr_from_u64(x));
                v2 = fr_mont_mul(u, jk_fr_from_u64(y));
            } else {
                v1 = fr_mont_mul(u, jk_fr_from_u128(s_lo, s_hi));
                bool upper_ok = (p.canonical != 0u)
                    && (p.upper_suffix_bits == 0u
                        || s_hi == ((1ul << p.upper_suffix_bits) - 1ul));
                v2 = upper_ok ? u : fr_zero();
            }
        }

        uint cell_base = (flag ? 3u : 0u) * 256u + chunk;
        bool tile_uniform = simd_all(active)
            && simd_all(chunk == simd_broadcast_first(chunk))
            && simd_size == 32u;
        if (tile_uniform) {
            uint c = simd_broadcast_first(chunk);
            Fr256 a0 = jk_simd_sum(flag ? fr_zero() : v0, simd_size);
            Fr256 a1 = jk_simd_sum(flag ? fr_zero() : v1, simd_size);
            Fr256 a2 = jk_simd_sum(flag ? fr_zero() : v2, simd_size);
            Fr256 b0 = jk_simd_sum(flag ? v0 : fr_zero(), simd_size);
            Fr256 b1 = jk_simd_sum(flag ? v1 : fr_zero(), simd_size);
            Fr256 b2 = jk_simd_sum(flag ? v2 : fr_zero(), simd_size);
            if (lane == 0u) {
                jk_cell_add(my, 0u * 256u + c, a0);
                jk_cell_add(my, 1u * 256u + c, a1);
                jk_cell_add(my, 2u * 256u + c, a2);
                jk_cell_add(my, 3u * 256u + c, b0);
                jk_cell_add(my, 4u * 256u + c, b1);
                jk_cell_add(my, 5u * 256u + c, b2);
            }
        } else {
            for (uint k = 0u; k < simd_size; k++) {
                if (lane == k && active) {
                    jk_cell_add(my, cell_base, v0);
                    jk_cell_add(my, cell_base + 256u, v1);
                    jk_cell_add(my, cell_base + 512u, v2);
                }
                simdgroup_barrier(mem_flags::mem_device);
            }
        }
    }
}

struct IrrSuffixScanParams {
    uint num_sgs;
    uint suffix_len;
};

// Per-table suffix scan over the host-built simdgroup schedule: simdgroup
// sg gathers bucket_flat[sg_range[2sg]..sg_range[2sg+1]] rows of table
// slot sg_slot[sg]; cell = suffix_position * 256 + chunk. suffix_meta rows
// are 9 words per slot: count, then ids packed id | is_01 << 8 (the 0/1
// flag picks Montgomery adds over raw-space products).
kernel void jk_irr_suffix_scan(
    device const uint* rows [[buffer(0)]],
    device const uint* u_evals [[buffer(1)]],
    device const uint* bucket_flat [[buffer(2)]],
    device const uint* sg_slot [[buffer(3)]],
    device const uint* sg_range [[buffer(4)]],
    device const uint* suffix_meta [[buffer(5)]],
    device uint* partials [[buffer(6)]],
    constant IrrSuffixScanParams& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    uint sg = gid / simd_size;
    if (sg >= p.num_sgs) {
        return;
    }
    device uint* my = partials + sg * JK_IRR_SUF_CELLS * FR_LIMBS;
    for (uint i = lane; i < JK_IRR_SUF_CELLS * FR_LIMBS; i += simd_size) {
        my[i] = 0u;
    }
    simdgroup_barrier(mem_flags::mem_device);

    uint slot = sg_slot[sg];
    uint count = suffix_meta[slot * 9u];
    uint start = sg_range[2u * sg];
    uint end = sg_range[2u * sg + 1u];
    for (uint base = start; base < end; base += simd_size) {
        uint i = base + lane;
        bool active = i < end;
        uint chunk = 0u;
        ulong s_lo = 0ul;
        ulong s_hi = 0ul;
        Fr256 u = fr_zero();
        if (active) {
            uint j = bucket_flat[i];
            device const uint* row = rows + j * 12u;
            ulong lo = (ulong)row[0] | ((ulong)row[1] << 32);
            ulong hi = (ulong)row[2] | ((ulong)row[3] << 32);
            u = fr_load(u_evals, j);
            chunk = jk_chunk8(lo, hi, p.suffix_len);
            jk_mask128(lo, hi, p.suffix_len, s_lo, s_hi);
        }
        bool tile_uniform = simd_all(active)
            && simd_all(chunk == simd_broadcast_first(chunk))
            && simd_size == 32u;
        for (uint s = 0u; s < count; s++) {
            uint meta = suffix_meta[slot * 9u + 1u + s];
            uint id = meta & 0xFFu;
            bool is01 = (meta & 0x100u) != 0u;
            Fr256 v = fr_zero();
            if (active) {
                ulong m = jk_suffix_mle(id, s_lo, s_hi, p.suffix_len);
                if (m != 0ul) {
                    v = is01 ? u : fr_mont_mul(u, jk_fr_from_u64(m));
                }
            }
            if (tile_uniform) {
                Fr256 total = jk_simd_sum(v, simd_size);
                if (lane == 0u) {
                    jk_cell_add(my, s * 256u + simd_broadcast_first(chunk), total);
                }
            } else {
                for (uint k = 0u; k < simd_size; k++) {
                    if (lane == k && active) {
                        jk_cell_add(my, s * 256u + chunk, v);
                    }
                    simdgroup_barrier(mem_flags::mem_device);
                }
            }
        }
    }
}

struct IrrReduceParams {
    uint total_cells;      // out length
    uint cells_per_group;  // out cells per group (= partials row cells used)
    uint stride;           // partials row stride, in cells
};

// Tree-finish of the per-simdgroup rows: out[cell] sums its group's rows.
// Groups are contiguous simdgroup ranges (one group for the RAF scan; one
// per table slot for the suffix scan).
kernel void jk_irr_reduce(
    device const uint* partials [[buffer(0)]],
    device const uint* group_range [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant IrrReduceParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.total_cells) {
        return;
    }
    uint group = gid / p.cells_per_group;
    uint cell = gid % p.cells_per_group;
    Fr256 acc = fr_zero();
    uint sg_end = group_range[2u * group + 1u];
    for (uint sg = group_range[2u * group]; sg < sg_end; sg++) {
        acc = fr_add(acc, fr_load(partials + sg * p.stride * FR_LIMBS, cell));
    }
    fr_store(out, gid, acc);
}

struct IrrCycleInitParams {
    uint n;
    uint phase_begin;  // first bound-challenge table of this ra product
    uint phase_count;  // 0 selects combined_val mode
    uint address_bits;
    uint out_base;     // element offset of this table in `out`
    uint raf_interleaved[FR_LIMBS];
    uint raf_identity[FR_LIMBS];
};

// Address→cycle handoff materialization, one output table per dispatch:
// combined_val (phase_count = 0) is the collapsed lookup-table value plus
// the RAF constant selected by the row's flag; ra_i (phase_count > 0) is
// the product of its phases' bound-challenge chunk weights. Pure map — a
// failed command buffer leaves nothing to recover.
kernel void jk_irr_cycle_init(
    device const uint* rows [[buffer(0)]],
    device const uint* v_tables [[buffer(1)]],
    device const uint* table_values [[buffer(2)]],
    device uint* out [[buffer(3)]],
    constant IrrCycleInitParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    device const uint* row = rows + gid * 12u;
    ulong lo = (ulong)row[0] | ((ulong)row[1] << 32);
    ulong hi = (ulong)row[2] | ((ulong)row[3] << 32);
    ulong out_word = ((ulong)p.out_base + (ulong)gid) * (ulong)FR_LIMBS;
    if (p.phase_count == 0u) {
        uint table_plus_one = row[8] & 0xFFu;
        bool flag = ((row[8] >> 8) & 0xFFu) != 0u;
        Fr256 value =
            (table_plus_one != 0u) ? fr_load(table_values, table_plus_one - 1u) : fr_zero();
        Fr256 raf = flag ? fr_load_const(p.raf_identity, 0) : fr_load_const(p.raf_interleaved, 0);
        fr_store(out + out_word, 0u, fr_add(value, raf));
        return;
    }
    uint phase = p.phase_begin;
    uint shift = p.address_bits - (phase + 1u) * 8u;
    Fr256 product = fr_load(v_tables, phase * 256u + jk_chunk8(lo, hi, shift));
    for (uint k = 1u; k < p.phase_count; k++) {
        phase += 1u;
        shift -= 8u;
        product = fr_mont_mul(product, fr_load(v_tables, phase * 256u + jk_chunk8(lo, hi, shift)));
    }
    fr_store(out + out_word, 0u, product);
}

#define JK_IRR_MAX_FACTORS 16u

struct IrrCycleRoundParams {
    uint groups;      // post-bind per-table pair count = active threads
    uint do_bind;
    uint num_tgs;     // partials stride
    uint log_in;      // log2(e_in length)
    uint num_tables;  // F = 1 + ra_count (≤ JK_IRR_MAX_FACTORS)
    uint len;         // per-table CURRENT (pre-bind) length = cur stride
    uint r[FR_LIMBS];
};

// Stage-5 cycle product round over the flat cycle tables (combined_val then
// the ra products; cur compact at stride len, nxt written compact at stride
// len/2). Folds the pending challenge per table (jk_round_pair) and
// accumulates the product-grid evaluations q(t) = Σ_y eq(y)·Π_f tbl_f(t, y)
// for t ∈ {1, …, F−1, ∞} as lane-major per-threadgroup partials. eq =
// e_out·e_in rides in factor 0, matching the CPU's e_in-in-Val fold by
// distributivity (exact, so the host's gruen assembly lands byte-identical).
kernel void jk_irr_cycle_round(
    device const uint* cur [[buffer(0)]],
    device uint* nxt [[buffer(1)]],
    device const uint* e_in [[buffer(2)]],
    device const uint* e_out [[buffer(3)]],
    device uint* partials [[buffer(4)]],
    constant IrrCycleRoundParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    bool bind = p.do_bind != 0u;
    Fr256 r = fr_load_const(p.r, 0);
    uint f_count = min(p.num_tables, JK_IRR_MAX_FACTORS);

    // Per-factor linear evaluations over the folded pair: value at t = 1 and
    // the per-step increment (inactive lanes hold zeros and contribute zero
    // to every lane sum, but still reach the barriers in jk_tg_sum).
    Fr256 evals[JK_IRR_MAX_FACTORS];
    Fr256 steps[JK_IRR_MAX_FACTORS];
    for (uint f = 0u; f < f_count; f++) {
        Fr256 lo, hi;
        ulong cur_word = (ulong)f * (ulong)p.len * (ulong)FR_LIMBS;
        ulong nxt_word = (ulong)f * (ulong)(p.len >> 1) * (ulong)FR_LIMBS;
        jk_round_pair(cur + cur_word, nxt + nxt_word,
                      bind, r, gid, active, lo, hi);
        evals[f] = hi;
        steps[f] = fr_sub(hi, lo);
    }
    Fr256 eq = fr_zero();
    if (active) {
        eq = fr_mont_mul(fr_load(e_out, gid >> p.log_in),
                         fr_load(e_in, gid & ((1u << p.log_in) - 1u)));
    }
    evals[0] = fr_mont_mul(eq, evals[0]);
    steps[0] = fr_mont_mul(eq, steps[0]);

    for (uint t = 1u; t < f_count; t++) {
        Fr256 prod = evals[0];
        for (uint f = 1u; f < f_count; f++) {
            prod = fr_mont_mul(prod, evals[f]);
        }
        jk_tg_sum(scratch, lid, tg, prod, partials, t - 1u, p.num_tgs);
        if (t + 1u < f_count) {
            for (uint f = 0u; f < f_count; f++) {
                evals[f] = fr_add(evals[f], steps[f]);
            }
        }
    }
    Fr256 lead = steps[0];
    for (uint f = 1u; f < f_count; f++) {
        lead = fr_mont_mul(lead, steps[f]);
    }
    jk_tg_sum(scratch, lid, tg, lead, partials, f_count - 1u, p.num_tgs);
}

struct SuffixProbeParams {
    uint n;
};

// Test-only: evaluate suffix MLEs on explicit cases (6 words each: bits as
// 4 LE limbs, id, len) so the Rust side can pin every variant.
kernel void jk_suffix_probe(
    device const uint* cases [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant SuffixProbeParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    device const uint* c = cases + gid * 6u;
    ulong lo = (ulong)c[0] | ((ulong)c[1] << 32);
    ulong hi = (ulong)c[2] | ((ulong)c[3] << 32);
    ulong v = jk_suffix_mle(c[4], lo, hi, c[5]);
    out[2u * gid] = (uint)v;
    out[2u * gid + 1u] = (uint)(v >> 32);
}

// --- Stage-3 instruction input-virtualization (W9) ---------------------------
//
// Summand q = (is_rs2·rs2 + is_imm·imm) + γ·(is_rs1·rs1 + is_pc·upc) over
// eight tables, table-major in one ping-pong pair (table i's length-`len`
// region at element i·len). Two kernels: the first bind materializes the
// dense tables straight from the trace record's native lanes (u64 values,
// i128 immediates, packed flag words — promoted in-register, never a
// T-sized dense table in memory) fused with that round's q sums; later
// rounds are the standard fused fold+eval. Partial sums land in 4 slots
// (q at t = 0..3); the host applies ℓ(t) and assembles the wire polynomial
// through the optimized tier's own recipe.

// v·R mod p for a u64 — the same canonical Montgomery residue the host's
// `Field::from_u64` produces.
inline Fr256 jk_fr_mont_from_u64(ulong v) {
    return fr_mont_mul(jk_fr_from_u64(v), fr_load_const(FR_R2, 0));
}

// Flag bit `bit` of packed word `w` as 0 or 1 in Montgomery form
// (`Field::from_bool`), mask-selected so warps stay uniform.
inline Fr256 jk_fr_flag(uint w, uint bit) {
    uint mask = 0u - ((w >> bit) & 1u);
    Fr256 r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.v[i] = FR_ONE[i] & mask;
    }
    return r;
}

inline Fr256 jk_ii_select(bool selected, Fr256 value) {
    uint mask = 0u - uint(selected);
    Fr256 out;
    for (uint i = 0; i < FR_LIMBS; i++) {
        out.v[i] = value.v[i] & mask;
    }
    return out;
}

// Quadratic coefficient of f(t)·v(t), where f is Boolean at t=0,1.
inline Fr256 jk_ii_flag_times_slope(bool f0, bool f1, Fr256 v0, Fr256 v1) {
    Fr256 slope = fr_sub(v1, v0);
    Fr256 rising = jk_ii_select(!f0 && f1, slope);
    Fr256 falling = jk_ii_select(f0 && !f1, slope);
    return fr_sub(rising, falling);
}

struct InstrInputQ0Params {
    uint groups;        // native row pairs = T/2
    uint num_tgs;
    uint log_in;        // log2(e_in length)
    uint flag_bits[4];  // packed-lane bit of is_rs1, is_pc, is_rs2, is_imm
    uint gamma[FR_LIMBS];
};

// Native round-0 message. Each operand product is quadratic in t. Boolean
// endpoint selection gives q(0), q(1) without a field multiplication; the
// flag-transition times operand slope gives the quadratic coefficient.
// Three weighted reductions reconstruct q(2), q(3) on the host.
kernel void jk_instr_input_q0(
    device const uint* flags [[buffer(0)]],
    device const ulong* rs1 [[buffer(1)]],
    device const ulong* upc [[buffer(2)]],
    device const ulong* rs2 [[buffer(3)]],
    device const uint* imm [[buffer(4)]],  // 4 LE u32 words per cycle
    device const uint* e_in [[buffer(5)]],
    device const uint* e_out [[buffer(6)]],
    device uint* partials [[buffer(7)]],
    constant InstrInputQ0Params& p [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    Fr256 q0 = fr_zero();
    Fr256 q1 = fr_zero();
    Fr256 qa = fr_zero();
    Fr256 eq = fr_zero();
    if (active) {
        uint j = 2u * gid;
        uint w0 = flags[j], w1 = flags[j + 1u];
        bool rs1_f0 = ((w0 >> p.flag_bits[0]) & 1u) != 0u;
        bool rs1_f1 = ((w1 >> p.flag_bits[0]) & 1u) != 0u;
        bool pc_f0 = ((w0 >> p.flag_bits[1]) & 1u) != 0u;
        bool pc_f1 = ((w1 >> p.flag_bits[1]) & 1u) != 0u;
        bool rs2_f0 = ((w0 >> p.flag_bits[2]) & 1u) != 0u;
        bool rs2_f1 = ((w1 >> p.flag_bits[2]) & 1u) != 0u;
        bool imm_f0 = ((w0 >> p.flag_bits[3]) & 1u) != 0u;
        bool imm_f1 = ((w1 >> p.flag_bits[3]) & 1u) != 0u;

        Fr256 rs1_0 = jk_fr_mont_from_u64(rs1[j]);
        Fr256 rs1_1 = jk_fr_mont_from_u64(rs1[j + 1u]);
        Fr256 upc_0 = jk_fr_mont_from_u64(upc[j]);
        Fr256 upc_1 = jk_fr_mont_from_u64(upc[j + 1u]);
        Fr256 rs2_0 = jk_fr_mont_from_u64(rs2[j]);
        Fr256 rs2_1 = jk_fr_mont_from_u64(rs2[j + 1u]);
        uint k = 4u * j;
        Fr256 imm_0 = fr_from_i128(imm[k], imm[k + 1u], imm[k + 2u], imm[k + 3u]);
        Fr256 imm_1 = fr_from_i128(imm[k + 4u], imm[k + 5u], imm[k + 6u], imm[k + 7u]);

        Fr256 left0 = fr_add(jk_ii_select(rs1_f0, rs1_0), jk_ii_select(pc_f0, upc_0));
        Fr256 left1 = fr_add(jk_ii_select(rs1_f1, rs1_1), jk_ii_select(pc_f1, upc_1));
        Fr256 right0 = fr_add(jk_ii_select(rs2_f0, rs2_0), jk_ii_select(imm_f0, imm_0));
        Fr256 right1 = fr_add(jk_ii_select(rs2_f1, rs2_1), jk_ii_select(imm_f1, imm_1));
        Fr256 left_a = fr_add(jk_ii_flag_times_slope(rs1_f0, rs1_f1, rs1_0, rs1_1),
                              jk_ii_flag_times_slope(pc_f0, pc_f1, upc_0, upc_1));
        Fr256 right_a = fr_add(jk_ii_flag_times_slope(rs2_f0, rs2_f1, rs2_0, rs2_1),
                               jk_ii_flag_times_slope(imm_f0, imm_f1, imm_0, imm_1));
        Fr256 gamma = fr_load_const(p.gamma, 0);
        q0 = fr_add(right0, fr_mont_mul(gamma, left0));
        q1 = fr_add(right1, fr_mont_mul(gamma, left1));
        qa = fr_add(right_a, fr_mont_mul(gamma, left_a));
        eq = fr_mont_mul(fr_load(e_out, gid >> p.log_in),
                         fr_load(e_in, gid & ((1u << p.log_in) - 1u)));
    }
    jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, q0), partials, 0u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, q1), partials, 1u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, qa), partials, 2u, p.num_tgs);
}

// Fold one table's quad (e0, e1) → lo, (e2, e3) → hi with r, store the pair
// at nxt[2y], nxt[2y+1], and hand back (value at t=0, step) of the pair's
// linear extension.
inline void jk_ii_fold_store(Fr256 e0, Fr256 e1, Fr256 e2, Fr256 e3, Fr256 r,
                             device uint* table, uint y,
                             thread Fr256& v, thread Fr256& s)
{
    Fr256 lo = fr_add(e0, fr_mont_mul(r, fr_sub(e1, e0)));
    Fr256 hi = fr_add(e2, fr_mont_mul(r, fr_sub(e3, e2)));
    fr_store(table, 2u * y, lo);
    fr_store(table, 2u * y + 1u, hi);
    v = lo;
    s = fr_sub(hi, lo);
}

// q(t)·eq at t = 0..3 into partials slots 0..3. `v` mutates in place
// (v += s per point); every thread participates in the reductions.
inline void jk_ii_eval(thread Fr256* v, thread const Fr256* s, Fr256 eq,
                       Fr256 gamma, threadgroup uint* scratch, uint lid,
                       uint tg, device uint* partials, uint num_tgs)
{
    for (uint t = 0; t < 4u; t++) {
        if (t != 0u) {
            for (uint i = 0; i < 8u; i++) {
                v[i] = fr_add(v[i], s[i]);
            }
        }
        Fr256 right = fr_add(fr_mont_mul(v[4], v[5]), fr_mont_mul(v[6], v[7]));
        Fr256 left = fr_add(fr_mont_mul(v[0], v[1]), fr_mont_mul(v[2], v[3]));
        Fr256 q = fr_add(right, fr_mont_mul(gamma, left));
        jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, q), partials, t, num_tgs);
    }
}

struct InstrInputBindNativeParams {
    uint groups;        // post-bind pair count = T/4
    uint num_tgs;
    uint log_in;        // log2(e_in length)
    uint out_len;       // dense per-table length = T/2
    uint flag_bits[4];  // packed-lane bit of is_rs1, is_pc, is_rs2, is_imm
    uint r[FR_LIMBS];
    uint gamma[FR_LIMBS];
};

// First bind: native rows 4y..4y+3 → dense rows 2y, 2y+1 of all eight
// tables, plus the round's q sums over the folded pair. Table order is the
// output-claim declaration order [is_rs1, rs1, is_pc, upc, is_rs2, rs2,
// is_imm, imm]. Lanes are promoted per table so at most one table's four
// residues are live besides the running (v, s) pairs.
kernel void jk_instr_input_bind_native(
    device const uint* flags [[buffer(0)]],
    device const ulong* rs1 [[buffer(1)]],
    device const ulong* upc [[buffer(2)]],
    device const ulong* rs2 [[buffer(3)]],
    device const uint* imm [[buffer(4)]],  // 4 LE u32 words per cycle
    device uint* dense [[buffer(5)]],      // 8·out_len, table-major
    device const uint* e_in [[buffer(6)]],
    device const uint* e_out [[buffer(7)]],
    device uint* partials [[buffer(8)]],
    constant InstrInputBindNativeParams& p [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    Fr256 r = fr_load_const(p.r, 0);
    Fr256 gamma = fr_load_const(p.gamma, 0);

    Fr256 v[8];
    Fr256 s[8];
    Fr256 eq = fr_zero();
    if (active) {
        uint j = 4u * gid;
        uint stride = p.out_len * FR_LIMBS;
        uint w0 = flags[j], w1 = flags[j + 1u], w2 = flags[j + 2u], w3 = flags[j + 3u];
        jk_ii_fold_store(jk_fr_flag(w0, p.flag_bits[0]), jk_fr_flag(w1, p.flag_bits[0]),
                         jk_fr_flag(w2, p.flag_bits[0]), jk_fr_flag(w3, p.flag_bits[0]),
                         r, dense, gid, v[0], s[0]);
        jk_ii_fold_store(jk_fr_mont_from_u64(rs1[j]), jk_fr_mont_from_u64(rs1[j + 1u]),
                         jk_fr_mont_from_u64(rs1[j + 2u]), jk_fr_mont_from_u64(rs1[j + 3u]),
                         r, dense + stride, gid, v[1], s[1]);
        jk_ii_fold_store(jk_fr_flag(w0, p.flag_bits[1]), jk_fr_flag(w1, p.flag_bits[1]),
                         jk_fr_flag(w2, p.flag_bits[1]), jk_fr_flag(w3, p.flag_bits[1]),
                         r, dense + 2u * stride, gid, v[2], s[2]);
        jk_ii_fold_store(jk_fr_mont_from_u64(upc[j]), jk_fr_mont_from_u64(upc[j + 1u]),
                         jk_fr_mont_from_u64(upc[j + 2u]), jk_fr_mont_from_u64(upc[j + 3u]),
                         r, dense + 3u * stride, gid, v[3], s[3]);
        jk_ii_fold_store(jk_fr_flag(w0, p.flag_bits[2]), jk_fr_flag(w1, p.flag_bits[2]),
                         jk_fr_flag(w2, p.flag_bits[2]), jk_fr_flag(w3, p.flag_bits[2]),
                         r, dense + 4u * stride, gid, v[4], s[4]);
        jk_ii_fold_store(jk_fr_mont_from_u64(rs2[j]), jk_fr_mont_from_u64(rs2[j + 1u]),
                         jk_fr_mont_from_u64(rs2[j + 2u]), jk_fr_mont_from_u64(rs2[j + 3u]),
                         r, dense + 5u * stride, gid, v[5], s[5]);
        jk_ii_fold_store(jk_fr_flag(w0, p.flag_bits[3]), jk_fr_flag(w1, p.flag_bits[3]),
                         jk_fr_flag(w2, p.flag_bits[3]), jk_fr_flag(w3, p.flag_bits[3]),
                         r, dense + 6u * stride, gid, v[6], s[6]);
        uint k = 4u * j;
        jk_ii_fold_store(fr_from_i128(imm[k], imm[k + 1u], imm[k + 2u], imm[k + 3u]),
                         fr_from_i128(imm[k + 4u], imm[k + 5u], imm[k + 6u], imm[k + 7u]),
                         fr_from_i128(imm[k + 8u], imm[k + 9u], imm[k + 10u], imm[k + 11u]),
                         fr_from_i128(imm[k + 12u], imm[k + 13u], imm[k + 14u], imm[k + 15u]),
                         r, dense + 7u * stride, gid, v[7], s[7]);
        eq = fr_mont_mul(fr_load(e_out, gid >> p.log_in),
                         fr_load(e_in, gid & ((1u << p.log_in) - 1u)));
    } else {
        for (uint i = 0; i < 8u; i++) {
            v[i] = fr_zero();
            s[i] = fr_zero();
        }
    }
    jk_ii_eval(v, s, eq, gamma, scratch, lid, tg, partials, p.num_tgs);
}

struct InstrInputRoundParams {
    uint groups;   // post-bind pair count = len/4
    uint num_tgs;
    uint log_in;   // log2(e_in length)
    uint len;      // per-table PRE-bind length
    uint r[FR_LIMBS];
    uint gamma[FR_LIMBS];
};

// Dense rounds after the first bind: fold all eight tables (always binding —
// round 0 is the native kernel's) and accumulate the four q sums.
kernel void jk_instr_input_round(
    device const uint* cur [[buffer(0)]],
    device uint* nxt [[buffer(1)]],
    device const uint* e_in [[buffer(2)]],
    device const uint* e_out [[buffer(3)]],
    device uint* partials [[buffer(4)]],
    constant InstrInputRoundParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    Fr256 r = fr_load_const(p.r, 0);
    Fr256 gamma = fr_load_const(p.gamma, 0);
    uint half_len = p.len >> 1;

    Fr256 v[8];
    Fr256 s[8];
    for (uint i = 0; i < 8u; i++) {
        Fr256 lo, hi;
        jk_round_pair(cur + i * p.len * FR_LIMBS, nxt + i * half_len * FR_LIMBS,
                      true, r, gid, active, lo, hi);
        v[i] = lo;
        s[i] = fr_sub(hi, lo);
    }
    Fr256 eq = fr_zero();
    if (active) {
        eq = fr_mont_mul(fr_load(e_out, gid >> p.log_in),
                         fr_load(e_in, gid & ((1u << p.log_in) - 1u)));
    }
    jk_ii_eval(v, s, eq, gamma, scratch, lid, tg, partials, p.num_tgs);
}
