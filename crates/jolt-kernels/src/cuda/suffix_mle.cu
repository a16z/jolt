typedef unsigned int u32;

__device__ __forceinline__ u64 sm_shl_u64(u64 x, unsigned int s) {
    return s >= 64 ? 0ULL : (x << s);
}

__device__ __forceinline__ u64 sm_shr_u64(u64 x, unsigned int s) {
    return s >= 64 ? 0ULL : (x >> s);
}

__device__ __forceinline__ u32 sm_shl_u32(u32 x, unsigned int s) {
    return s >= 32 ? 0u : (x << s);
}

__device__ __forceinline__ u32 sm_shr_u32(u32 x, unsigned int s) {
    return s >= 32 ? 0u : (x >> s);
}

__device__ __forceinline__ void sm_uninterleave(u128 v, u64 *x, u64 *y) {
    uninterleave_u128(v, x, y);
}

__device__ __forceinline__ unsigned int sm_tz_raw(u128 v) {
    if (v == 0) return 128;
    u64 lo = (u64)v;
    return (lo != 0) ? (unsigned int)__ffsll((long long)lo) - 1
                     : 64 + (unsigned int)__ffsll((long long)(u64)(v >> 64)) - 1;
}

__device__ __forceinline__ unsigned int sm_trailing_zeros(u128 v, unsigned int len) {
    unsigned int tz = sm_tz_raw(v);
    return tz < len ? tz : len;
}

__device__ __forceinline__ unsigned int sm_leading_ones(u64 v, unsigned int len) {
    unsigned int shift = (128u - len) & 127u;
    u128 shifted = ((u128)v) << shift;
    unsigned int count = 0;
    for (int i = 127; i >= 0; i--) {
        if ((shifted >> i) & (u128)1) count++;
        else break;
    }
    return count;
}

__device__ __forceinline__ u64 sm_rev8w(u64 v) {
    u32 lo = __byte_perm((u32)v, 0, 0x0123);
    u32 hi = __byte_perm((u32)(v >> 32), 0, 0x0123);
    return (u64)lo + ((u64)hi << 32);
}

enum {
    SM_ONE = 0, SM_AND, SM_ANDNOT, SM_OR, SM_XOR,
    SM_RIGHT_OPERAND, SM_RIGHT_OPERAND_W, SM_CHANGE_DIVISOR, SM_CHANGE_DIVISOR_W,
    SM_UPPER_WORD, SM_LOWER_WORD, SM_LOWER_HALF_WORD,
    SM_LESS_THAN, SM_GREATER_THAN, SM_EQ,
    SM_LEFT_IS_ZERO, SM_RIGHT_IS_ZERO, SM_LSB, SM_DIV_BY_ZERO,
    SM_POW2, SM_POW2_W, SM_REV8W, SM_RIGHT_SHIFT_PADDING,
    SM_RIGHT_SHIFT, SM_RIGHT_SHIFT_HELPER, SM_SIGN_EXTENSION, SM_LEFT_SHIFT,
    SM_TWO_LSB, SM_SIGN_EXTENSION_UPPER_HALF, SM_SIGN_EXTENSION_RIGHT_OPERAND,
    SM_RIGHT_SHIFT_W, SM_RIGHT_SHIFT_W_HELPER, SM_LEFT_SHIFT_W_HELPER, SM_LEFT_SHIFT_W,
    SM_OVERFLOW_BITS_ZERO,
    SM_XOR_ROT16, SM_XOR_ROT24, SM_XOR_ROT32, SM_XOR_ROT63,
    SM_XOR_ROTW7, SM_XOR_ROTW8, SM_XOR_ROTW12, SM_XOR_ROTW16
};

#define SM_XLEN 64u

__device__ __forceinline__ u64 sm_xor_rot(u128 bits, unsigned int len, unsigned int rot) {
    u64 x, y;
    sm_uninterleave(bits, &x, &y);
    u64 r = x ^ y;
    return (r >> rot) | sm_shl_u64(r, 64 - rot);
}

__device__ __forceinline__ u64 sm_xor_rotw(u128 bits, unsigned int len, unsigned int rot) {
    u64 x, y;
    sm_uninterleave(bits, &x, &y);
    u32 r = (u32)x ^ (u32)y;
    u32 out = (r >> rot) | sm_shl_u32(r, 32 - rot);
    return (u64)out;
}

__device__ __forceinline__ u64 suffix_mle_eval(
    u128 bits,
    unsigned int len,
    unsigned int variant
) {
    switch (variant) {
        case SM_ONE:
            return 1;
        case SM_AND: {
            u64 x, y; sm_uninterleave(bits, &x, &y); return x & y;
        }
        case SM_ANDNOT: {
            u64 x, y; sm_uninterleave(bits, &x, &y); return x & ~y;
        }
        case SM_OR: {
            u64 x, y; sm_uninterleave(bits, &x, &y); return x | y;
        }
        case SM_XOR: {
            u64 x, y; sm_uninterleave(bits, &x, &y); return x ^ y;
        }
        case SM_RIGHT_OPERAND: {
            u64 x, y; sm_uninterleave(bits, &x, &y); return y;
        }
        case SM_RIGHT_OPERAND_W: {
            u64 x, y; sm_uninterleave(bits, &x, &y); return (u64)(u32)y;
        }
        case SM_CHANGE_DIVISOR: {
            u64 x, y; sm_uninterleave(bits, &x, &y);
            unsigned int ylen = len - (len / 2);
            u64 ones = sm_shl_u64(1, ylen) - 1;
            return (ones == y && x == 0) ? 1 : 0;
        }
        case SM_CHANGE_DIVISOR_W: {
            u64 x, y; sm_uninterleave(bits, &x, &y);
            unsigned int ylen = len - (len / 2);
            unsigned int ycap = ylen < (SM_XLEN / 2) ? ylen : (SM_XLEN / 2);
            u32 xw = (u32)x; u32 yw = (u32)y;
            u64 ones = sm_shl_u64(1, ycap) - 1;
            return (ones == (u64)yw && xw == 0) ? 1 : 0;
        }
        case SM_UPPER_WORD:
            return (u64)(bits >> SM_XLEN);
        case SM_LOWER_WORD:
            return (u64)bits;
        case SM_LOWER_HALF_WORD: {
            unsigned int hw = SM_XLEN / 2;
            return (u64)(bits & ((((u128)1) << hw) - 1));
        }
        case SM_LESS_THAN: {
            u64 x, y; sm_uninterleave(bits, &x, &y); return (x < y) ? 1 : 0;
        }
        case SM_GREATER_THAN: {
            u64 x, y; sm_uninterleave(bits, &x, &y); return (x > y) ? 1 : 0;
        }
        case SM_EQ: {
            u64 x, y; sm_uninterleave(bits, &x, &y); return (x == y) ? 1 : 0;
        }
        case SM_LEFT_IS_ZERO: {
            u64 x, y; sm_uninterleave(bits, &x, &y); return (x == 0) ? 1 : 0;
        }
        case SM_RIGHT_IS_ZERO: {
            u64 x, y; sm_uninterleave(bits, &x, &y); return (y == 0) ? 1 : 0;
        }
        case SM_LSB:
            return (len == 0) ? 1 : (u64)(bits & 1);
        case SM_DIV_BY_ZERO: {
            u64 divisor, quotient; sm_uninterleave(bits, &divisor, &quotient);
            unsigned int qlen = len - (len / 2);
            u64 ones = sm_shl_u64(1, qlen) - 1;
            return (divisor == 0 && quotient == ones) ? 1 : 0;
        }
        case SM_POW2: {
            if (len == 0) return 1;
            unsigned int log_xlen = 6;
            u64 shift = (u64)(bits & ((((u128)1) << log_xlen) - 1));
            return sm_shl_u64(1, (unsigned int)shift);
        }
        case SM_POW2_W: {
            if (len == 0) return 1;
            u64 shift = (u64)(bits & 0x1f);
            return sm_shl_u64(1, (unsigned int)shift);
        }
        case SM_REV8W:
            return sm_rev8w((u64)bits);
        case SM_RIGHT_SHIFT_PADDING: {
            if (len == 0) return 1;
            unsigned int log_xlen = 6;
            u64 shift = (u64)(bits & ((((u128)1) << log_xlen) - 1));
            return sm_shl_u64(1, SM_XLEN - 1 - (unsigned int)shift);
        }
        case SM_RIGHT_SHIFT: {
            u64 x, y; sm_uninterleave(bits, &x, &y);
            unsigned int ylen = len - (len / 2);
            return sm_shr_u64(x, sm_trailing_zeros((u128)y, ylen));
        }
        case SM_RIGHT_SHIFT_HELPER: {
            u64 x, y; sm_uninterleave(bits, &x, &y);
            unsigned int ylen = len - (len / 2);
            return sm_shl_u64(1, sm_leading_ones(y, ylen));
        }
        case SM_SIGN_EXTENSION: {
            u64 x, y; sm_uninterleave(bits, &x, &y);
            unsigned int ylen = len - (len / 2);
            unsigned int tz = sm_trailing_zeros((u128)y, ylen);
            u128 top = ((u128)1) << SM_XLEN;
            u128 low = ((u128)1) << (SM_XLEN - tz);
            return (u64)(top - low);
        }
        case SM_LEFT_SHIFT: {
            u64 x, y; sm_uninterleave(bits, &x, &y);
            unsigned int ylen = len - (len / 2);
            u64 xx = x & ~y;
            return sm_shl_u64(xx, sm_leading_ones(y, ylen));
        }
        case SM_TWO_LSB:
            return (len == 0 || sm_tz_raw(bits) >= 2) ? 1 : 0;
        case SM_SIGN_EXTENSION_UPPER_HALF: {
            unsigned int hw = SM_XLEN / 2;
            if (len >= hw) {
                u64 sign = (u64)((bits >> (hw - 1)) & 1);
                return sign ? ((sm_shl_u64(1, hw) - 1) << hw) : 0;
            }
            return 1;
        }
        case SM_SIGN_EXTENSION_RIGHT_OPERAND: {
            if (len >= SM_XLEN) {
                u64 sign = (u64)((bits >> (SM_XLEN - 2)) & 1);
                if (sign) {
                    u128 top = ((u128)1) << SM_XLEN;
                    u128 low = ((u128)1) << (SM_XLEN / 2);
                    return (u64)(top - low);
                }
                return 0;
            }
            return 1;
        }
        case SM_RIGHT_SHIFT_W: {
            u64 x, y; sm_uninterleave(bits, &x, &y);
            unsigned int ylen = len - (len / 2);
            unsigned int tz = sm_trailing_zeros((u128)y, ylen);
            unsigned int cap = SM_XLEN / 2;
            if (tz > cap) tz = cap;
            return (u64)sm_shr_u32((u32)x, tz);
        }
        case SM_RIGHT_SHIFT_W_HELPER: {
            u64 x, y; sm_uninterleave(bits, &x, &y);
            unsigned int ylen = len - (len / 2);
            unsigned int ycap = ylen < (SM_XLEN / 2) ? ylen : (SM_XLEN / 2);
            u64 yw = (u64)y & ((sm_shl_u64(1, ycap)) - 1);
            return sm_shl_u64(1, sm_leading_ones(yw, ycap));
        }
        case SM_LEFT_SHIFT_W_HELPER: {
            u64 x, y; sm_uninterleave(bits, &x, &y);
            unsigned int ylen = len - (len / 2);
            return (u64)sm_shl_u32(1u, sm_leading_ones(y, ylen));
        }
        case SM_LEFT_SHIFT_W: {
            u64 x, y; sm_uninterleave(bits, &x, &y);
            unsigned int ylen = len - (len / 2);
            unsigned int ycap = ylen < (SM_XLEN / 2) ? ylen : (SM_XLEN / 2);
            u64 yw_full = (u64)y & ((sm_shl_u64(1, ycap)) - 1);
            u32 xw = (u32)x; u32 yw = (u32)yw_full;
            u32 xx = xw & ~yw;
            return (u64)sm_shl_u32(xx, sm_leading_ones(yw_full, ycap));
        }
        case SM_OVERFLOW_BITS_ZERO:
            return ((bits >> SM_XLEN) == 0) ? 1 : 0;
        case SM_XOR_ROT16: return sm_xor_rot(bits, len, 16);
        case SM_XOR_ROT24: return sm_xor_rot(bits, len, 24);
        case SM_XOR_ROT32: return sm_xor_rot(bits, len, 32);
        case SM_XOR_ROT63: return sm_xor_rot(bits, len, 63);
        case SM_XOR_ROTW7: return sm_xor_rotw(bits, len, 7);
        case SM_XOR_ROTW8: return sm_xor_rotw(bits, len, 8);
        case SM_XOR_ROTW12: return sm_xor_rotw(bits, len, 12);
        case SM_XOR_ROTW16: return sm_xor_rotw(bits, len, 16);
    }
    return 0;
}

extern "C" __global__ void suffix_mle_probe(
    u64 *__restrict__ out,
    const u64 *__restrict__ bits_lo,
    const u64 *__restrict__ bits_hi,
    const unsigned int *__restrict__ len,
    const unsigned int *__restrict__ variant,
    unsigned long n
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u128 bits = ((u128)bits_hi[i] << 64) | (u128)bits_lo[i];
    out[i] = suffix_mle_eval(bits, len[i], variant[i]);
}
