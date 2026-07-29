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
            // Lanes take turns; the (uniform) barrier orders lane k's
            // read-modify-write before lane k+1 may touch the same cell.
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
