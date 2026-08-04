#define SFX_XLEN 64
#define SFX_LOG_XLEN 6

typedef struct {
    u128 bits;
    unsigned int len;
} sfx_bits;

__device__ __forceinline__ u128 sfx_mask(unsigned int len) {
    if (len >= 128) return ~(u128)0;
    if (len == 0) return 0;
    return (((u128)1) << len) - 1;
}

__device__ __forceinline__ sfx_bits sfx_new(u128 bits, unsigned int len) {
    sfx_bits out;
    out.len = len;
    out.bits = bits & sfx_mask(len);
    return out;
}

__device__ __forceinline__ unsigned long long sfx_u64(sfx_bits b) {
    return (unsigned long long)b.bits;
}

__device__ __forceinline__ unsigned int sfx_ctz128(u128 v, unsigned int len) {
    if (v == 0) return len;
    unsigned long long lo = (unsigned long long)v;
    unsigned int tz = (lo != 0) ? (unsigned int)__ffsll((long long)lo) - 1u
                                : 64u + (unsigned int)__ffsll((long long)(unsigned long long)(v >> 64)) - 1u;
    return tz < len ? tz : len;
}

__device__ __forceinline__ unsigned int sfx_trailing_zeros(sfx_bits b) {
    return sfx_ctz128(b.bits, b.len);
}

__device__ __forceinline__ unsigned int sfx_leading_ones(sfx_bits b) {
    if (b.len == 0) return 0;
    u128 shifted = b.len >= 128 ? b.bits : (b.bits << (128 - b.len));
    unsigned int count = 0;
    for (unsigned int i = 0; i < 128; i++) {
        if (((shifted >> (127 - i)) & 1) == 0) break;
        count++;
    }
    return count;
}

__device__ __forceinline__ void sfx_uninterleave(sfx_bits b, sfx_bits *x_out, sfx_bits *y_out) {
    const u128 M0 = ((u128)0x5555555555555555ULL << 64) | 0x5555555555555555ULL;
    const u128 M1 = ((u128)0x3333333333333333ULL << 64) | 0x3333333333333333ULL;
    const u128 M2 = ((u128)0x0F0F0F0F0F0F0F0FULL << 64) | 0x0F0F0F0F0F0F0F0FULL;
    const u128 M3 = ((u128)0x00FF00FF00FF00FFULL << 64) | 0x00FF00FF00FF00FFULL;
    const u128 M4 = ((u128)0x0000FFFF0000FFFFULL << 64) | 0x0000FFFF0000FFFFULL;
    const u128 M5 = ((u128)0x00000000FFFFFFFFULL << 64) | 0x00000000FFFFFFFFULL;
    const u128 M6 = (u128)0xFFFFFFFFFFFFFFFFULL;

    u128 x = (b.bits >> 1) & M0;
    u128 y = b.bits & M0;
    x = (x | (x >> 1)) & M1;
    x = (x | (x >> 2)) & M2;
    x = (x | (x >> 4)) & M3;
    x = (x | (x >> 8)) & M4;
    x = (x | (x >> 16)) & M5;
    x = (x | (x >> 32)) & M6;
    y = (y | (y >> 1)) & M1;
    y = (y | (y >> 2)) & M2;
    y = (y | (y >> 4)) & M3;
    y = (y | (y >> 8)) & M4;
    y = (y | (y >> 16)) & M5;
    y = (y | (y >> 32)) & M6;

    unsigned int x_len = b.len / 2;
    *x_out = sfx_new(x, x_len);
    *y_out = sfx_new(y, b.len - x_len);
}

__device__ __forceinline__ void sfx_split(sfx_bits b,
                                          unsigned int suffix_len,
                                          sfx_bits *prefix,
                                          sfx_bits *suffix) {
    *suffix = sfx_new(b.bits & sfx_mask(suffix_len), suffix_len);
    u128 upper = suffix_len >= 128 ? 0 : (b.bits >> suffix_len);
    *prefix = sfx_new(upper, b.len - suffix_len);
}

__device__ __forceinline__ unsigned long long sfx_shl_u64(unsigned long long v, unsigned int s) {
    return s >= 64 ? 0ULL : (v << s);
}

__device__ __forceinline__ unsigned long long sfx_shr_u64(unsigned long long v, unsigned int s) {
    return s >= 64 ? 0ULL : (v >> s);
}

__device__ __forceinline__ unsigned int sfx_shl_u32(unsigned int v, unsigned int s) {
    return s >= 32 ? 0u : (v << s);
}

__device__ __forceinline__ unsigned int sfx_shr_u32(unsigned int v, unsigned int s) {
    return s >= 32 ? 0u : (v >> s);
}

__device__ __forceinline__ unsigned int sfx_rotr_u64(unsigned long long v,
                                                     unsigned int r,
                                                     unsigned long long *out) {
    r &= 63u;
    *out = r == 0 ? v : ((v >> r) | (v << (64 - r)));
    return 0;
}

__device__ __forceinline__ unsigned int sfx_rotr_u32(unsigned int v, unsigned int r) {
    r &= 31u;
    return r == 0 ? v : ((v >> r) | (v << (32 - r)));
}

__device__ __forceinline__ unsigned long long sfx_rev8w(unsigned long long v) {
    unsigned int lo = __byte_perm((unsigned int)v, 0, 0x0123);
    unsigned int hi = __byte_perm((unsigned int)(v >> 32), 0, 0x0123);
    return (unsigned long long)lo + ((unsigned long long)hi << 32);
}

__device__ unsigned long long sfx_eval(unsigned int suffix, sfx_bits b) {
    sfx_bits x, y;
    switch (suffix) {
        case 0:  // One
            return 1ULL;
        case 1:  // And
            sfx_uninterleave(b, &x, &y);
            return sfx_u64(x) & sfx_u64(y);
        case 2:  // AndNot
            sfx_uninterleave(b, &x, &y);
            return sfx_u64(x) & ~sfx_u64(y);
        case 3:  // Xor
            sfx_uninterleave(b, &x, &y);
            return sfx_u64(x) ^ sfx_u64(y);
        case 4:  // Or
            sfx_uninterleave(b, &x, &y);
            return sfx_u64(x) | sfx_u64(y);
        case 5:  // RightOperand
            sfx_uninterleave(b, &x, &y);
            return sfx_u64(y);
        case 6:  // RightOperandW
            sfx_uninterleave(b, &x, &y);
            return (unsigned long long)(unsigned int)sfx_u64(y);
        case 7: {  // ChangeDivisor
            sfx_uninterleave(b, &x, &y);
            unsigned long long yv = sfx_u64(y);
            unsigned long long ones = sfx_shl_u64(1ULL, y.len) - 1ULL;
            return (ones == yv && sfx_u64(x) == 0) ? 1ULL : 0ULL;
        }
        case 8: {  // ChangeDivisorW
            sfx_uninterleave(b, &x, &y);
            unsigned int y_len = y.len < (SFX_XLEN / 2) ? y.len : (SFX_XLEN / 2);
            unsigned long long xv = (unsigned long long)(unsigned int)sfx_u64(x);
            unsigned long long yv = (unsigned long long)(unsigned int)sfx_u64(y);
            unsigned long long ones = sfx_shl_u64(1ULL, y_len) - 1ULL;
            return (ones == yv && xv == 0) ? 1ULL : 0ULL;
        }
        case 9:  // UpperWord
            return (unsigned long long)(b.bits >> SFX_XLEN);
        case 10:  // LowerWord
            return (unsigned long long)(b.bits & sfx_mask(SFX_XLEN));
        case 11:  // LowerHalfWord
            return (unsigned long long)(b.bits & sfx_mask(SFX_XLEN / 2));
        case 12:  // LessThan
            sfx_uninterleave(b, &x, &y);
            return sfx_u64(x) < sfx_u64(y) ? 1ULL : 0ULL;
        case 13:  // GreaterThan
            sfx_uninterleave(b, &x, &y);
            return sfx_u64(x) > sfx_u64(y) ? 1ULL : 0ULL;
        case 14:  // Eq
            sfx_uninterleave(b, &x, &y);
            return (x.bits == y.bits) ? 1ULL : 0ULL;
        case 15:  // LeftOperandIsZero
            sfx_uninterleave(b, &x, &y);
            return sfx_u64(x) == 0 ? 1ULL : 0ULL;
        case 16:  // RightOperandIsZero
            sfx_uninterleave(b, &x, &y);
            return sfx_u64(y) == 0 ? 1ULL : 0ULL;
        case 17:  // Lsb
            return b.len == 0 ? 1ULL : (unsigned long long)(b.bits & 1);
        case 18: {  // DivByZero
            sfx_uninterleave(b, &x, &y);
            bool divisor_zero = sfx_u64(x) == 0;
            unsigned long long ones = sfx_shl_u64(1ULL, y.len) - 1ULL;
            return (divisor_zero && sfx_u64(y) == ones) ? 1ULL : 0ULL;
        }
        case 19: {  // Pow2
            if (b.len == 0) return 1ULL;
            sfx_bits hi, shift;
            sfx_split(b, SFX_LOG_XLEN, &hi, &shift);
            return sfx_shl_u64(1ULL, (unsigned int)sfx_u64(shift));
        }
        case 20: {  // Pow2W
            if (b.len == 0) return 1ULL;
            sfx_bits hi, shift;
            sfx_split(b, 5, &hi, &shift);
            return sfx_shl_u64(1ULL, (unsigned int)sfx_u64(shift));
        }
        case 21:  // Rev8W
            return sfx_rev8w((unsigned long long)b.bits);
        case 22: {  // RightShiftPadding
            if (b.len == 0) return 1ULL;
            sfx_bits hi, shift;
            sfx_split(b, SFX_LOG_XLEN, &hi, &shift);
            return sfx_shl_u64(1ULL, SFX_XLEN - 1u - (unsigned int)sfx_u64(shift));
        }
        case 23:  // RightShift
            sfx_uninterleave(b, &x, &y);
            return sfx_shr_u64(sfx_u64(x), sfx_trailing_zeros(y));
        case 24:  // RightShiftHelper
            sfx_uninterleave(b, &x, &y);
            return sfx_shl_u64(1ULL, sfx_leading_ones(y));
        case 25: {  // SignExtension
            sfx_uninterleave(b, &x, &y);
            unsigned int tz = sfx_ctz128(y.bits, 128);
            unsigned int padding = tz < y.len ? tz : y.len;
            u128 full = ((u128)1 << SFX_XLEN);
            u128 keep = ((u128)1 << (SFX_XLEN - padding));
            return (unsigned long long)(full - keep);
        }
        case 26: {  // LeftShift
            sfx_uninterleave(b, &x, &y);
            unsigned long long xv = sfx_u64(x) & ~sfx_u64(y);
            return sfx_shl_u64(xv, sfx_leading_ones(y));
        }
        case 27:  // TwoLsb
            return (b.len == 0 || sfx_ctz128(b.bits, 128) >= 2) ? 1ULL : 0ULL;
        case 28: {  // SignExtensionUpperHalf
            unsigned int half = SFX_XLEN / 2;
            if (b.len >= half) {
                unsigned long long sign = (unsigned long long)((b.bits >> (half - 1)) & 1);
                if (sign == 1) {
                    return (sfx_shl_u64(1ULL, half) - 1ULL) << half;
                }
                return 0ULL;
            }
            return 1ULL;
        }
        case 29: {  // SignExtensionRightOperand
            if (b.len >= SFX_XLEN) {
                unsigned long long sign =
                    (unsigned long long)((b.bits >> (SFX_XLEN - 2)) & 1);
                if (sign == 1) {
                    u128 full = ((u128)1 << SFX_XLEN);
                    u128 keep = ((u128)1 << (SFX_XLEN / 2));
                    return (unsigned long long)(full - keep);
                }
                return 0ULL;
            }
            return 1ULL;
        }
        case 30: {  // RightShiftW
            sfx_uninterleave(b, &x, &y);
            unsigned int tz = sfx_trailing_zeros(y);
            unsigned int limit = SFX_XLEN / 2;
            unsigned int shift = tz < limit ? tz : limit;
            return (unsigned long long)sfx_shr_u32((unsigned int)sfx_u64(x), shift);
        }
        case 31: {  // RightShiftWHelper
            sfx_uninterleave(b, &x, &y);
            unsigned int y_len = y.len < (SFX_XLEN / 2) ? y.len : (SFX_XLEN / 2);
            sfx_bits y_trunc = sfx_new(y.bits, y_len);
            return sfx_shl_u64(1ULL, sfx_leading_ones(y_trunc));
        }
        case 32:  // LeftShiftWHelper
            sfx_uninterleave(b, &x, &y);
            return (unsigned long long)sfx_shl_u32(1u, sfx_leading_ones(y));
        case 33: {  // LeftShiftW
            sfx_uninterleave(b, &x, &y);
            unsigned int y_len = y.len < (SFX_XLEN / 2) ? y.len : (SFX_XLEN / 2);
            sfx_bits y_trunc = sfx_new(y.bits, y_len);
            unsigned int xv = (unsigned int)sfx_u64(x);
            unsigned int yv = (unsigned int)sfx_u64(y_trunc);
            xv &= ~yv;
            return (unsigned long long)sfx_shl_u32(xv, sfx_leading_ones(y_trunc));
        }
        case 34:  // OverflowBitsZero
            return ((b.bits >> SFX_XLEN) == 0) ? 1ULL : 0ULL;
        case 35:
        case 36:
        case 37:
        case 38: {  // XorRot16 / 24 / 32 / 63
            const unsigned int rots[4] = {16u, 24u, 32u, 63u};
            sfx_uninterleave(b, &x, &y);
            unsigned long long xr = sfx_u64(x) ^ sfx_u64(y);
            unsigned long long out;
            sfx_rotr_u64(xr, rots[suffix - 35], &out);
            return out;
        }
        case 39:
        case 40:
        case 41:
        case 42: {  // XorRotW16 / W12 / W8 / W7
            const unsigned int rots[4] = {16u, 12u, 8u, 7u};
            sfx_uninterleave(b, &x, &y);
            unsigned int xr = (unsigned int)sfx_u64(x) ^ (unsigned int)sfx_u64(y);
            return (unsigned long long)sfx_rotr_u32(xr, rots[suffix - 39]);
        }
        default:
            return 0ULL;
    }
}

extern "C" __global__ void sfx_eval_batch_kernel(const unsigned long long *__restrict__ bits,
                                                 const unsigned char *__restrict__ lens,
                                                 unsigned int suffix,
                                                 unsigned long long *__restrict__ out,
                                                 unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sfx_bits b;
    b.bits = ((u128)bits[2 * i + 1] << 64) | (u128)bits[2 * i];
    b.len = lens[i];
    b.bits &= sfx_mask(b.len);
    out[i] = sfx_eval(suffix, b);
}
