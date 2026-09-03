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
// simdgroup, and those are settled in registers — a shuffle-xor butterfly
// for a tile whose active lanes share one chunk (the dominant case in early
// phases — high index bits are mostly zero on real traces), a sort +
// segmented scan otherwise — before one lane per key adds to device memory.

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

inline uint jk_popcount64(ulong v) {
    return popcount((uint)v) + popcount((uint)(v >> 32));
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

// pext(x, y) (suffixes/pext.rs): pack x's bits at y's set positions toward
// bit 0, lowest first.
inline ulong jk_pext(ulong x, ulong y) {
    if (y == 0ul) {
        return 0ul;
    }
    uint tz = jk_ctz64(y);
    ulong normalized = y >> tz;
    if ((normalized & (normalized + 1ul)) == 0ul) {
        // Contiguous mask: shift plus truncate.
        return (x >> tz) & normalized;
    }
    ulong bits = y;
    ulong out = 0ul;
    uint k = 0u;
    while (bits != 0ul) {
        out |= ((x >> jk_ctz64(bits)) & 1ul) << k;
        k += 1u;
        bits &= bits - 1ul;
    }
    return out;
}

// window_sign_bit (suffixes/window_sign.rs): x's bit at y's most significant
// set bit, 0 if y is zero. ilog2(y) = 63 - clz64(y).
inline ulong jk_window_sign_bit(ulong x, ulong y) {
    return (y == 0ul) ? 0ul : ((x >> (63u - jk_clz64(y))) & 1ul);
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
#define JK_SUF_UPPER_WORD 7u
#define JK_SUF_LOWER_WORD 8u
#define JK_SUF_LOWER_HALF_WORD 9u
#define JK_SUF_LESS_THAN 10u
#define JK_SUF_GREATER_THAN 11u
#define JK_SUF_EQ 12u
#define JK_SUF_LEFT_IS_ZERO 13u
#define JK_SUF_RIGHT_IS_ZERO 14u
#define JK_SUF_LSB 15u
#define JK_SUF_DIV_BY_ZERO 16u
#define JK_SUF_POW2 17u
#define JK_SUF_POW2_W 18u
#define JK_SUF_REV8W 19u
#define JK_SUF_RIGHT_SHIFT_PADDING 20u
#define JK_SUF_RIGHT_SHIFT 21u
#define JK_SUF_RIGHT_SHIFT_HELPER 22u
#define JK_SUF_SIGN_EXTENSION 23u
#define JK_SUF_LEFT_SHIFT 24u
#define JK_SUF_TWO_LSB 25u
#define JK_SUF_SIGN_EXTENSION_UPPER_HALF 26u
#define JK_SUF_SIGN_EXTENSION_RIGHT_OPERAND 27u
#define JK_SUF_RIGHT_SHIFT_W 28u
#define JK_SUF_RIGHT_SHIFT_W_HELPER 29u
#define JK_SUF_LEFT_SHIFT_W_HELPER 30u
#define JK_SUF_LEFT_SHIFT_W 31u
#define JK_SUF_OVERFLOW_BITS_ZERO 32u
#define JK_SUF_XOR_ROT_16 33u
#define JK_SUF_XOR_ROT_24 34u
#define JK_SUF_XOR_ROT_32 35u
#define JK_SUF_XOR_ROT_63 36u
#define JK_SUF_XOR_ROT_W_16 37u
#define JK_SUF_XOR_ROT_W_12 38u
#define JK_SUF_XOR_ROT_W_8 39u
#define JK_SUF_XOR_ROT_W_7 40u
#define JK_SUF_POW2_OFFSET_W 41u
#define JK_SUF_PEXT 42u
#define JK_SUF_PEXT_HELPER 43u
#define JK_SUF_WINDOW_SIGN 44u
#define JK_SUF_WINDOW_SIGN_POW2 45u
#define JK_SUF_XOR_ROT_W_22 46u
#define JK_SUF_XOR_ROT_W_19 47u
#define JK_SUF_XOR_ROT_W_6 48u
#define JK_SUF_SIGN_EXTENSION_W 49u
#define JK_SUF_X31_Y0 50u
#define JK_SUF_POW2_OFFSET_B 51u
#define JK_SUF_POW2_OFFSET_H 52u
#define JK_SUF_ALIGN_ADDR 53u
#define JK_SUF_SHIFT_DATA_B 54u
#define JK_SUF_SHIFT_DATA_H 55u
#define JK_SUF_SHIFT_DATA_W 56u
#define JK_SUF_OFFSET_SCALE_B 57u
#define JK_SUF_OFFSET_SCALE_H 58u
#define JK_SUF_OFFSET_SCALE_W 59u
#define JK_SUF_XOR_ROT_L1_PAIRS 60u
#define JK_SUF_TOP_Y_BIT 61u
#define JK_SUF_BOTTOM_X_BIT 62u

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
        case JK_SUF_POW2_OFFSET_W:
            // 2^(32·bit2) of the non-interleaved window; below 3 bits the
            // offset bit is prefix-owned and the suffix factor is 1.
            return (len < 3u) ? 1ul : (1ul << (32u * (uint)((s_lo >> 2) & 1ul)));
        case JK_SUF_PEXT:
            jk_uninterleave(s_lo, s_hi, x, y);
            return jk_pext(x, y);
        case JK_SUF_PEXT_HELPER:
            jk_uninterleave(s_lo, s_hi, x, y);
            return 1ul << jk_popcount64(y);
        case JK_SUF_WINDOW_SIGN:
            jk_uninterleave(s_lo, s_hi, x, y);
            return jk_window_sign_bit(x, y);
        case JK_SUF_WINDOW_SIGN_POW2:
            jk_uninterleave(s_lo, s_hi, x, y);
            return jk_window_sign_bit(x, y) << jk_popcount64(y);
        case JK_SUF_XOR_ROT_W_22:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (ulong)jk_rotr32((uint)x ^ (uint)y, 22u);
        case JK_SUF_XOR_ROT_W_19:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (ulong)jk_rotr32((uint)x ^ (uint)y, 19u);
        case JK_SUF_XOR_ROT_W_6:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (ulong)jk_rotr32((uint)x ^ (uint)y, 6u);
        case JK_SUF_SIGN_EXTENSION_W: {
            if (len == 0u) {
                return 0ul;
            }
            jk_uninterleave(s_lo, s_hi, x, y);
            ulong fill = 0ul;
            if (len >= 64u) {
                if (((x >> 31) & 1ul) == 0ul) {
                    return 0ul;
                }
                // (1u128 << 64) - (1u128 << 32) as u64
                fill = 0xFFFFFFFF00000000ul;
            }
            // The Rust loop walks y's low min(yl, 32) bits MSB-first: bit j
            // contributes (1 - y_j) << (31 - j) and position 0 (j = 31) is
            // skipped, so the index set is exactly j in [0, min(yl, 31)).
            uint count = min(yl, 31u);
            for (uint j = 0u; j < count; j++) {
                fill += (1ul - ((y >> j) & 1ul)) << (31u - j);
            }
            return fill;
        }
        case JK_SUF_X31_Y0:
            if (len < 64u) {
                return 0ul;
            }
            jk_uninterleave(s_lo, s_hi, x, y);
            return ((x >> 31) & 1ul) * (y & 1ul);
        case JK_SUF_POW2_OFFSET_B:
            // 2^(8·(ea mod 8)); a sub-3-bit window is the partial factor for
            // the offset bits it owns (masked s_lo covers that).
            return 1ul << (8u * (uint)(s_lo & 7ul));
        case JK_SUF_POW2_OFFSET_H:
            return 1ul << (8u * (uint)(s_lo & 6ul));
        case JK_SUF_ALIGN_ADDR:
            return s_lo & ~7ul;
        case JK_SUF_SHIFT_DATA_B:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (x & 0xFFul) << (8u * (uint)(y & 7ul));
        case JK_SUF_SHIFT_DATA_H:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (x & 0xFFFFul) << (8u * (uint)(y & 6ul));
        case JK_SUF_SHIFT_DATA_W:
            jk_uninterleave(s_lo, s_hi, x, y);
            return (x & 0xFFFFFFFFul) << (8u * (uint)(y & 4ul));
        case JK_SUF_OFFSET_SCALE_B:
            // 2^(8·(y mod 8)) over the offset bits y_0..y_2 (even index
            // positions); the masked y half is the partial factor for the
            // offset bits the suffix owns.
            jk_uninterleave(s_lo, s_hi, x, y);
            return 1ul << (8u * (uint)(y & 7ul));
        case JK_SUF_OFFSET_SCALE_H:
            jk_uninterleave(s_lo, s_hi, x, y);
            return 1ul << (8u * (uint)(y & 6ul));
        case JK_SUF_OFFSET_SCALE_W:
            jk_uninterleave(s_lo, s_hi, x, y);
            return 1ul << (8u * (uint)(y & 4ul));
        case JK_SUF_XOR_ROT_L1_PAIRS: {
            jk_uninterleave(s_lo, s_hi, x, y);
            uint pairs = len / 2u;
            return (x ^ (y << 1)) & ((1ul << pairs) - 1ul) & ~1ul;
        }
        case JK_SUF_TOP_Y_BIT: {
            uint pairs = len / 2u;
            if (pairs == 0u) {
                return 0ul;
            }
            jk_uninterleave(s_lo, s_hi, x, y);
            return (y >> (pairs - 1u)) & 1ul;
        }
        case JK_SUF_BOTTOM_X_BIT:
            jk_uninterleave(s_lo, s_hi, x, y);
            return x & 1ul;
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

inline Fr256 jk_fr_shuffle_v4(Fr256 v, uint source) {
    uint4 a = uint4(v.v[0], v.v[1], v.v[2], v.v[3]);
    uint4 b = uint4(v.v[4], v.v[5], v.v[6], v.v[7]);
    uint4 sa = simd_shuffle(a, (ushort)source);
    uint4 sb = simd_shuffle(b, (ushort)source);
    Fr256 out;
    out.v[0] = sa.x;
    out.v[1] = sa.y;
    out.v[2] = sa.z;
    out.v[3] = sa.w;
    out.v[4] = sb.x;
    out.v[5] = sb.y;
    out.v[6] = sb.z;
    out.v[7] = sb.w;
    return out;
}

inline Fr256 jk_fr_shuffle_up_v4(Fr256 v, ushort delta) {
    uint4 a = uint4(v.v[0], v.v[1], v.v[2], v.v[3]);
    uint4 b = uint4(v.v[4], v.v[5], v.v[6], v.v[7]);
    uint4 sa = simd_shuffle_up(a, delta);
    uint4 sb = simd_shuffle_up(b, delta);
    Fr256 out;
    out.v[0] = sa.x;
    out.v[1] = sa.y;
    out.v[2] = sa.z;
    out.v[3] = sa.w;
    out.v[4] = sb.x;
    out.v[5] = sb.y;
    out.v[6] = sb.z;
    out.v[7] = sb.w;
    return out;
}

// Bitonic sort of `(key << 5) | lane` across the simdgroup (inactive lanes
// carry 0xFFFF and sort to the end); ascending, unique by the lane bits.
inline uint jk_sort_key_lane(uint packed, uint lane) {
    for (uint k = 2u; k <= 32u; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            uint partner = simd_shuffle_xor(packed, (ushort)j);
            bool up = ((lane & k) == 0u);
            bool keep_min = ((lane & j) == 0u) == up;
            packed = keep_min ? min(packed, partner) : max(packed, partner);
        }
    }
    return packed;
}

// Run structure of the sorted keys, from one key shuffle + one ballot:
// per-lane offset into its equal-key run, the longest offset in the tile,
// and the per-run tail lane. Invalid lanes (0xFFFF-packed) sort to the top,
// keep `run_off = 0`, and never emit; the first of them still marks a run
// start so the last valid lane detects its tail. A lane at offset `o`
// merges at scan step `d` iff `o >= d` (sorted keys make "the key `d` lanes
// up equals mine" and "my run extends `d` past its start" the same
// predicate), so the scan below can stop after the longest run is covered
// (the skipped steps would have performed no adds).
inline void jk_sorted_runs(
    uint skey,
    bool valid,
    uint lane,
    uint simd_size,
    thread uint& run_off,
    thread uint& max_off,
    thread bool& tail)
{
    uint pkey = simd_shuffle_up(skey, 1u);
    bool start = lane == 0u || pkey != skey;
    uint start_mask = (uint)(ulong)simd_ballot(start);
    uint below = start_mask & (0xFFFFFFFFu >> (31u - lane));
    run_off = valid ? (lane - (31u - clz(below))) : 0u;
    max_off = simd_max(run_off);
    tail = valid && (lane == simd_size - 1u || ((start_mask >> (lane + 1u)) & 1u) != 0u);
}

// Flush the held (key, 3xFr) accumulators: sort lanes by key, gather each
// lane's sorted-slot values with one vec4 shuffle triple, segmented
// inclusive scan over the (now contiguous) equal-key runs, segment tails
// add the per-key totals to device cells. The run-offset ballot
// (jk_sorted_runs) bounds the <= 5 scan steps by the longest equal-key run
// — production tiles carry 8-18 distinct keys, so most flushes finish in
// 1-3 steps.
inline void jk_flush_sorted3(
    device uint* cells,
    uint key,
    bool flush,
    Fr256 h0,
    Fr256 h1,
    Fr256 h2,
    uint lane,
    uint simd_size)
{
    uint packed = jk_sort_key_lane(flush ? ((key << 5u) | lane) : 0xFFFFu, lane);
    uint src = packed & 31u;
    uint skey = packed >> 5u;
    bool valid = packed != 0xFFFFu;
    Fr256 g0 = jk_fr_shuffle_v4(h0, src);
    Fr256 g1 = jk_fr_shuffle_v4(h1, src);
    Fr256 g2 = jk_fr_shuffle_v4(h2, src);
    uint run_off;
    uint max_off;
    bool tail;
    jk_sorted_runs(skey, valid, lane, simd_size, run_off, max_off, tail);
    for (uint d = 1u; d <= max_off; d <<= 1u) {
        Fr256 p0 = jk_fr_shuffle_up_v4(g0, (ushort)d);
        Fr256 p1 = jk_fr_shuffle_up_v4(g1, (ushort)d);
        Fr256 p2 = jk_fr_shuffle_up_v4(g2, (ushort)d);
        if (run_off >= d) {
            g0 = fr_add(g0, p0);
            g1 = fr_add(g1, p1);
            g2 = fr_add(g2, p2);
        }
    }
    if (tail) {
        uint family_base = (skey >> 8u) * 3u * 256u;
        uint chunk = skey & 255u;
        jk_cell_add(cells, family_base + chunk, g0);
        jk_cell_add(cells, family_base + 256u + chunk, g1);
        jk_cell_add(cells, family_base + 512u + chunk, g2);
    }
}

// Fused condensation + RAF scan with per-lane run-length accumulation.
// Per-simdgroup cells, quantity-major:
// [0] shift_half, [1] left, [2] right (interleaved rows);
// [3] shift_full, [4] identity, [5] upper-all-ones (RAF rows);
// cell = quantity * 256 + chunk. Cells 1/2/4 are RAW space (host fix-up).
//
// Equal-key runs accumulate in per-lane registers; on any key change the
// tile flushes through jk_flush_sorted3 (sort + segmented scan). No
// uniform-tile special case: run-length + sorted flush handles uniform runs
// cheaper than per-tile butterflies.
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

    uint held_key = 0xFFFFFFFFu;
    Fr256 h0 = fr_zero();
    Fr256 h1 = fr_zero();
    Fr256 h2 = fr_zero();

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

        uint key = (flag ? 256u : 0u) + chunk;
        bool same = active && key == held_key;
        if (same) {
            h0 = fr_add(h0, v0);
            h1 = fr_add(h1, v1);
            h2 = fr_add(h2, v2);
        }
        bool take = active && !same;
        bool flush = take && held_key != 0xFFFFFFFFu;
        if (simd_any(flush)) {
            jk_flush_sorted3(my, held_key, flush, h0, h1, h2, lane, simd_size);
        }
        if (take) {
            held_key = key;
            h0 = v0;
            h1 = v1;
            h2 = v2;
        }
    }
    bool have = held_key != 0xFFFFFFFFu;
    if (simd_any(have)) {
        jk_flush_sorted3(my, held_key, have, h0, h1, h2, lane, simd_size);
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
//
// The chunk keys are suffix-invariant, so a tile sorts (chunk, lane) ONCE
// and every suffix reuses the mapping: one vec4 gather + 5-step segmented
// scan per suffix (zero-valued lanes contribute exact zeros) instead of
// per-suffix collision scatters. Uniform tiles skip the sort and keep the
// single-butterfly emit.
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
    uint slot = sg_slot[sg];
    uint count = suffix_meta[slot * 9u];
    device uint* my = partials + sg * JK_IRR_SUF_CELLS * FR_LIMBS;
    for (uint i = lane; i < count * 256u * FR_LIMBS; i += simd_size) {
        my[i] = 0u;
    }
    simdgroup_barrier(mem_flags::mem_device);

    // Suffix descriptors are per-slot constants: read them once, not per
    // tile-iteration x suffix.
    uint metas[8];
    for (uint s = 0u; s < count; s++) {
        metas[s] = suffix_meta[slot * 9u + 1u + s];
    }

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

        if (tile_uniform) {
            uint ucell = simd_broadcast_first(chunk);
            for (uint s = 0u; s < count; s++) {
                uint meta = metas[s];
                uint id = meta & 0xFFu;
                bool is01 = (meta & 0x100u) != 0u;
                Fr256 v = fr_zero();
                ulong m = jk_suffix_mle(id, s_lo, s_hi, p.suffix_len);
                if (m != 0ul) {
                    v = is01 ? u : fr_mont_mul(u, jk_fr_from_u64(m));
                }
                Fr256 total = jk_simd_sum(v, simd_size);
                if (lane == 0u) {
                    jk_cell_add(my, s * 256u + ucell, total);
                }
            }
        } else {
            // Sort the tile once (chunk keys are suffix-invariant), pull each
            // lane's sorted-slot suffix bits + weight through the mapping, and
            // hoist the run structure: every suffix then scans in place with
            // no per-suffix gathers or key shuffles, in as many steps as the
            // longest run needs.
            uint packed =
                jk_sort_key_lane(active ? ((chunk << 5u) | lane) : 0xFFFFu, lane);
            uint src = packed & 31u;
            uint skey = packed >> 5u;
            bool valid = packed != 0xFFFFu;
            uint4 sbits = simd_shuffle(
                uint4((uint)s_lo, (uint)(s_lo >> 32), (uint)s_hi, (uint)(s_hi >> 32)),
                (ushort)src);
            ulong g_lo = (ulong)sbits.x | ((ulong)sbits.y << 32);
            ulong g_hi = (ulong)sbits.z | ((ulong)sbits.w << 32);
            Fr256 gu = jk_fr_shuffle_v4(u, src);
            uint run_off;
            uint max_off;
            bool tail;
            jk_sorted_runs(skey, valid, lane, simd_size, run_off, max_off, tail);

            for (uint s = 0u; s < count; s++) {
                uint meta = metas[s];
                uint id = meta & 0xFFu;
                bool is01 = (meta & 0x100u) != 0u;
                Fr256 g = fr_zero();
                if (valid) {
                    ulong m = jk_suffix_mle(id, g_lo, g_hi, p.suffix_len);
                    if (m != 0ul) {
                        g = is01 ? gu : fr_mont_mul(gu, jk_fr_from_u64(m));
                    }
                }
                for (uint d = 1u; d <= max_off; d <<= 1u) {
                    Fr256 pv = jk_fr_shuffle_up_v4(g, (ushort)d);
                    if (run_off >= d) {
                        g = fr_add(g, pv);
                    }
                }
                if (tail) {
                    jk_cell_add(my, s * 256u + skey, g);
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
struct IrrEqOuterParams {
    uint n;
    uint lo_bits;
};

// eq(r, ·) as the outer product of its high/low half tables:
// `u_evals[j] = hi[j >> lo_bits] · lo[j & mask]` — exact by distributivity
// (each half entry is the exact partial product; Montgomery mul of
// canonical elements is the canonical full product), so the table lands
// byte-identical to the host `EqPolynomial::evals` fill it replaces. Runs
// as dispatch 0 of the phase-0 scan command buffer.
kernel void jk_irr_eq_outer(
    device uint* u_evals [[buffer(0)]],
    device const uint* hi [[buffer(1)]],
    device const uint* lo [[buffer(2)]],
    constant IrrEqOuterParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    Fr256 h = fr_load(hi, gid >> p.lo_bits);
    Fr256 l = fr_load(lo, gid & ((1u << p.lo_bits) - 1u));
    fr_store(u_evals, gid, fr_mont_mul(h, l));
}

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
    ulong out_word = (ulong)gid * (ulong)FR_LIMBS;
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

struct IrrCycleInitFusedParams {
    uint n;
    uint ra_count;
    uint phases_per_ra;
    uint address_bits;
    uint raf_interleaved[FR_LIMBS];
    uint raf_identity[FR_LIMBS];
};

// Fused twin of jk_irr_cycle_init for the adopting (flat ping-pong) path:
// one dispatch writes combined_val AND every ra product at stride n, reading
// each row once instead of once per output table.
kernel void jk_irr_cycle_init_fused(
    device const uint* rows [[buffer(0)]],
    device const uint* v_tables [[buffer(1)]],
    device const uint* table_values [[buffer(2)]],
    device uint* out [[buffer(3)]],
    constant IrrCycleInitFusedParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    device const uint* row = rows + gid * 12u;
    ulong lo = (ulong)row[0] | ((ulong)row[1] << 32);
    ulong hi = (ulong)row[2] | ((ulong)row[3] << 32);
    uint table_plus_one = row[8] & 0xFFu;
    bool flag = ((row[8] >> 8) & 0xFFu) != 0u;
    Fr256 value =
        (table_plus_one != 0u) ? fr_load(table_values, table_plus_one - 1u) : fr_zero();
    Fr256 raf = flag ? fr_load_const(p.raf_identity, 0) : fr_load_const(p.raf_interleaved, 0);
    fr_store(out + (ulong)gid * (ulong)FR_LIMBS, 0u, fr_add(value, raf));
    uint phase = 0u;
    for (uint i = 0u; i < p.ra_count; i++) {
        uint shift = p.address_bits - (phase + 1u) * 8u;
        Fr256 product = fr_load(v_tables, phase * 256u + jk_chunk8(lo, hi, shift));
        for (uint k = 1u; k < p.phases_per_ra; k++) {
            phase += 1u;
            shift -= 8u;
            product =
                fr_mont_mul(product, fr_load(v_tables, phase * 256u + jk_chunk8(lo, hi, shift)));
        }
        phase += 1u;
        ulong out_word = ((ulong)(i + 1u) * (ulong)p.n + (ulong)gid) * (ulong)FR_LIMBS;
        fr_store(out + out_word, 0u, product);
    }
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

// --- Stage-3 instruction input-virtualization --------------------------------
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
