#define PFX_XLEN 64
#define PFX_LOG_K 128
#define PFX_COUNT 46

#define PFX_LOWER_WORD 0
#define PFX_LOWER_HALF_WORD 1
#define PFX_UPPER_WORD 2
#define PFX_EQ 3
#define PFX_AND 4
#define PFX_ANDN 5
#define PFX_OR 6
#define PFX_XOR 7
#define PFX_LESS_THAN 8
#define PFX_LEFT_IS_ZERO 9
#define PFX_RIGHT_IS_ZERO 10
#define PFX_LEFT_MSB 11
#define PFX_RIGHT_MSB 12
#define PFX_DIV_BY_ZERO 13
#define PFX_POS_REM_EQ_DIV 14
#define PFX_POS_REM_LT_DIV 15
#define PFX_NEG_DIV_ZERO_REM 16
#define PFX_NEG_DIV_EQ_REM 17
#define PFX_NEG_DIV_GT_REM 18
#define PFX_LSB 19
#define PFX_POW2 20
#define PFX_POW2W 21
#define PFX_REV8W 22
#define PFX_RIGHT_SHIFT 23
#define PFX_SIGN_EXTENSION 24
#define PFX_LEFT_SHIFT 25
#define PFX_LEFT_SHIFT_HELPER 26
#define PFX_TWO_LSB 27
#define PFX_SIGN_EXT_UPPER_HALF 28
#define PFX_CHANGE_DIVISOR 29
#define PFX_CHANGE_DIVISOR_W 30
#define PFX_RIGHT_OPERAND 31
#define PFX_RIGHT_OPERAND_W 32
#define PFX_SIGN_EXT_RIGHT_OPERAND 33
#define PFX_RIGHT_SHIFT_W 34
#define PFX_LEFT_SHIFT_W_HELPER 35
#define PFX_LEFT_SHIFT_W 36
#define PFX_OVERFLOW_BITS_ZERO 37
#define PFX_XOR_ROT16 38
#define PFX_XOR_ROT24 39
#define PFX_XOR_ROT32 40
#define PFX_XOR_ROT63 41
#define PFX_XOR_ROTW7 42
#define PFX_XOR_ROTW8 43
#define PFX_XOR_ROTW12 44
#define PFX_XOR_ROTW16 45

__device__ __forceinline__ void pfx_load(const u64 *__restrict__ checkpoints,
                                         unsigned int index,
                                         u64 *out) {
    load4(checkpoints + (unsigned long long)index * LIMBS, out);
}

__device__ __forceinline__ void pfx_zero(u64 *out) {
    for (int l = 0; l < LIMBS; l++) out[l] = 0;
}

__device__ __forceinline__ void pfx_one(u64 *out) {
    load4(FR_ONE, out);
}

__device__ __forceinline__ void pfx_from_u64(unsigned long long v, u64 *out) {
    u64 raw[LIMBS] = {v, 0, 0, 0};
    fr_to_mont(raw, out);
}

__device__ __forceinline__ void pfx_from_u128(u128 v, u64 *out) {
    u64 raw[LIMBS] = {(unsigned long long)v, (unsigned long long)(v >> 64), 0, 0};
    fr_to_mont(raw, out);
}

__device__ __forceinline__ unsigned long long pfx_ones_u64(unsigned int len) {
    return len >= 64 ? ~0ULL : ((1ULL << len) - 1ULL);
}

__device__ __forceinline__ unsigned int pfx_msb(unsigned long long v, unsigned int len) {
    return len == 0 ? 0u : (unsigned int)((v >> (len - 1)) & 1ULL);
}

__device__ void pfx_eval(unsigned int prefix,
                         const u64 *__restrict__ checkpoints,
                         sfx_bits b,
                         unsigned int suffix_len,
                         u64 *out) {
    unsigned int j_start = PFX_LOG_K - suffix_len - b.len;
    sfx_bits x, y;

    switch (prefix) {
        case PFX_LOWER_WORD: {
            if (j_start < PFX_XLEN) { pfx_zero(out); return; }
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, PFX_LOWER_WORD, base);
            pfx_from_u128(b.bits << suffix_len, term);
            fr_add(base, term, out);
            return;
        }
        case PFX_LOWER_HALF_WORD: {
            if (j_start < PFX_XLEN + PFX_XLEN / 2) { pfx_zero(out); return; }
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, PFX_LOWER_HALF_WORD, base);
            pfx_from_u128(b.bits << suffix_len, term);
            fr_add(base, term, out);
            return;
        }
        case PFX_UPPER_WORD: {
            u64 base[LIMBS];
            pfx_load(checkpoints, PFX_UPPER_WORD, base);
            if (j_start >= PFX_XLEN) { store4(out, base); return; }
            u64 term[LIMBS];
            if (suffix_len > PFX_XLEN) {
                pfx_from_u64(sfx_shl_u64((unsigned long long)b.bits, suffix_len - PFX_XLEN), term);
            } else {
                sfx_bits hi, lo;
                sfx_split(b, PFX_XLEN - suffix_len, &hi, &lo);
                pfx_from_u64(sfx_u64(hi), term);
            }
            fr_add(base, term, out);
            return;
        }
        case PFX_EQ: {
            sfx_uninterleave(b, &x, &y);
            if (x.bits == y.bits) { pfx_load(checkpoints, PFX_EQ, out); } else { pfx_zero(out); }
            return;
        }
        case PFX_AND: {
            sfx_uninterleave(b, &x, &y);
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, PFX_AND, base);
            pfx_from_u64(sfx_shl_u64(sfx_u64(x) & sfx_u64(y), suffix_len / 2), term);
            fr_add(base, term, out);
            return;
        }
        case PFX_ANDN: {
            sfx_uninterleave(b, &x, &y);
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, PFX_ANDN, base);
            pfx_from_u64(sfx_shl_u64(sfx_u64(x) & ~sfx_u64(y), suffix_len / 2), term);
            fr_add(base, term, out);
            return;
        }
        case PFX_OR: {
            sfx_uninterleave(b, &x, &y);
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, PFX_OR, base);
            pfx_from_u64(sfx_shl_u64(sfx_u64(x) | sfx_u64(y), suffix_len / 2), term);
            fr_add(base, term, out);
            return;
        }
        case PFX_XOR: {
            sfx_uninterleave(b, &x, &y);
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, PFX_XOR, base);
            pfx_from_u64(sfx_shl_u64(sfx_u64(x) ^ sfx_u64(y), suffix_len / 2), term);
            fr_add(base, term, out);
            return;
        }
        case PFX_LESS_THAN: {
            sfx_uninterleave(b, &x, &y);
            u64 lt[LIMBS];
            pfx_load(checkpoints, PFX_LESS_THAN, lt);
            if (sfx_u64(x) < sfx_u64(y)) {
                u64 eq[LIMBS];
                pfx_load(checkpoints, PFX_EQ, eq);
                fr_add(lt, eq, out);
            } else {
                store4(out, lt);
            }
            return;
        }
        case PFX_LEFT_IS_ZERO: {
            sfx_uninterleave(b, &x, &y);
            if (sfx_u64(x) != 0) { pfx_zero(out); } else { pfx_load(checkpoints, PFX_LEFT_IS_ZERO, out); }
            return;
        }
        case PFX_RIGHT_IS_ZERO: {
            sfx_uninterleave(b, &x, &y);
            if (sfx_u64(y) != 0) { pfx_zero(out); } else { pfx_load(checkpoints, PFX_RIGHT_IS_ZERO, out); }
            return;
        }
        case PFX_LEFT_MSB: {
            if (j_start > 0) { pfx_load(checkpoints, PFX_LEFT_MSB, out); return; }
            sfx_uninterleave(b, &x, &y);
            pfx_from_u64((unsigned long long)pfx_msb(sfx_u64(x), x.len), out);
            return;
        }
        case PFX_RIGHT_MSB: {
            if (j_start > 0) { pfx_load(checkpoints, PFX_RIGHT_MSB, out); return; }
            sfx_uninterleave(b, &x, &y);
            pfx_from_u64((unsigned long long)pfx_msb(sfx_u64(y), y.len), out);
            return;
        }
        case PFX_DIV_BY_ZERO: {
            sfx_uninterleave(b, &x, &y);
            if (sfx_u64(x) != 0 || sfx_u64(y) != pfx_ones_u64(y.len)) { pfx_zero(out); }
            else { pfx_load(checkpoints, PFX_DIV_BY_ZERO, out); }
            return;
        }
        case PFX_POS_REM_EQ_DIV: {
            sfx_uninterleave(b, &x, &y);
            if (x.bits != y.bits) { pfx_zero(out); return; }
            if (j_start == 0 && b.len != 0) {
                if (pfx_msb(sfx_u64(x), x.len) != 0 || pfx_msb(sfx_u64(y), y.len) != 0) {
                    pfx_zero(out); return;
                }
            }
            pfx_load(checkpoints, PFX_POS_REM_EQ_DIV, out);
            return;
        }
        case PFX_POS_REM_LT_DIV: {
            sfx_uninterleave(b, &x, &y);
            if (j_start == 0 && b.len != 0) {
                if (pfx_msb(sfx_u64(x), x.len) != 0 || pfx_msb(sfx_u64(y), y.len) != 0) {
                    pfx_zero(out); return;
                }
            }
            u64 lt[LIMBS];
            pfx_load(checkpoints, PFX_POS_REM_LT_DIV, lt);
            if (sfx_u64(x) < sfx_u64(y)) {
                u64 eq[LIMBS];
                pfx_load(checkpoints, PFX_POS_REM_EQ_DIV, eq);
                fr_add(lt, eq, out);
            } else {
                store4(out, lt);
            }
            return;
        }
        case PFX_NEG_DIV_ZERO_REM: {
            sfx_uninterleave(b, &x, &y);
            if (sfx_u64(x) != 0) { pfx_zero(out); return; }
            if (j_start == 0 && b.len != 0 && pfx_msb(sfx_u64(y), y.len) != 1) {
                pfx_zero(out); return;
            }
            pfx_load(checkpoints, PFX_NEG_DIV_ZERO_REM, out);
            return;
        }
        case PFX_NEG_DIV_EQ_REM: {
            sfx_uninterleave(b, &x, &y);
            if (x.bits != y.bits) { pfx_zero(out); return; }
            if (j_start == 0 && b.len != 0) {
                if (pfx_msb(sfx_u64(x), x.len) != 1 || pfx_msb(sfx_u64(y), y.len) != 1) {
                    pfx_zero(out); return;
                }
            }
            pfx_load(checkpoints, PFX_NEG_DIV_EQ_REM, out);
            return;
        }
        case PFX_NEG_DIV_GT_REM: {
            sfx_uninterleave(b, &x, &y);
            if (j_start == 0 && b.len != 0) {
                if (pfx_msb(sfx_u64(x), x.len) != 1 || pfx_msb(sfx_u64(y), y.len) != 1) {
                    pfx_zero(out); return;
                }
            }
            u64 gt[LIMBS];
            pfx_load(checkpoints, PFX_NEG_DIV_GT_REM, gt);
            if (sfx_u64(x) > sfx_u64(y)) {
                u64 eq[LIMBS];
                pfx_load(checkpoints, PFX_NEG_DIV_EQ_REM, eq);
                fr_add(gt, eq, out);
            } else {
                store4(out, gt);
            }
            return;
        }
        case PFX_LSB: {
            if (suffix_len == 0) {
                pfx_from_u64((unsigned long long)(b.bits & 1), out);
            } else {
                pfx_one(out);
            }
            return;
        }
        case PFX_POW2: {
            if (suffix_len != 0) { pfx_one(out); return; }
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, PFX_POW2, base);
            pfx_from_u64(sfx_shl_u64(1ULL, (unsigned int)(b.bits & (PFX_XLEN - 1))), term);
            fr_mul(base, term, out);
            return;
        }
        case PFX_POW2W: {
            if (suffix_len != 0) { pfx_one(out); return; }
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, PFX_POW2W, base);
            pfx_from_u64(sfx_shl_u64(1ULL, (unsigned int)(b.bits & 0x1F)), term);
            fr_mul(base, term, out);
            return;
        }
        case PFX_REV8W: {
            if (suffix_len >= 64) { pfx_zero(out); return; }
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, PFX_REV8W, base);
            pfx_from_u64(sfx_rev8w(sfx_shl_u64((unsigned long long)b.bits, suffix_len)), term);
            fr_add(base, term, out);
            return;
        }
        case PFX_RIGHT_SHIFT:
        case PFX_RIGHT_SHIFT_W: {
            if (prefix == PFX_RIGHT_SHIFT_W && j_start < PFX_XLEN) { pfx_zero(out); return; }
            sfx_uninterleave(b, &x, &y);
            unsigned long long xv = sfx_u64(x), yv = sfx_u64(y);
            unsigned int n = y.len;
            u64 result[LIMBS];
            pfx_load(checkpoints, prefix, result);
            for (unsigned int i = 0; i < n; i++) {
                unsigned int x_i = (unsigned int)((xv >> (n - 1 - i)) & 1ULL);
                unsigned int y_i = (unsigned int)((yv >> (n - 1 - i)) & 1ULL);
                if (y_i == 1) {
                    u64 doubled[LIMBS], bit[LIMBS], sum[LIMBS];
                    fr_add(result, result, doubled);
                    pfx_from_u64((unsigned long long)x_i, bit);
                    fr_add(doubled, bit, sum);
                    store4(result, sum);
                }
            }
            store4(out, result);
            return;
        }
        case PFX_SIGN_EXTENSION: {
            sfx_uninterleave(b, &x, &y);
            unsigned long long yv = sfx_u64(y);
            unsigned int y_len = y.len;
            if (j_start == 0) {
                if (pfx_msb(sfx_u64(x), x.len) == 0) { pfx_zero(out); return; }
                unsigned long long sum = 0ULL;
                for (unsigned int i = 1; i < y_len; i++) {
                    if (((yv >> (y_len - 1 - i)) & 1ULL) == 0) sum += (1ULL << i);
                }
                pfx_from_u64(sum, out);
                return;
            }
            u64 sign[LIMBS], acc[LIMBS];
            pfx_load(checkpoints, PFX_LEFT_MSB, sign);
            pfx_zero(acc);
            unsigned int base_index = j_start / 2;
            for (unsigned int i = 0; i < y_len; i++) {
                if (((yv >> (y_len - 1 - i)) & 1ULL) == 0) {
                    u64 term[LIMBS], sum[LIMBS];
                    pfx_from_u64(sfx_shl_u64(1ULL, base_index + i), term);
                    fr_add(acc, term, sum);
                    store4(acc, sum);
                }
            }
            u64 base[LIMBS], scaled[LIMBS];
            pfx_load(checkpoints, PFX_SIGN_EXTENSION, base);
            fr_mul(sign, acc, scaled);
            fr_add(base, scaled, out);
            return;
        }
        case PFX_LEFT_SHIFT:
        case PFX_LEFT_SHIFT_W: {
            unsigned int helper = prefix == PFX_LEFT_SHIFT ? PFX_LEFT_SHIFT_HELPER
                                                           : PFX_LEFT_SHIFT_W_HELPER;
            if (prefix == PFX_LEFT_SHIFT_W && j_start < PFX_XLEN) { pfx_zero(out); return; }
            sfx_uninterleave(b, &x, &y);
            unsigned long long xv = sfx_u64(x), yv = sfx_u64(y);
            unsigned int n = y.len;
            u64 result[LIMBS], prod[LIMBS];
            pfx_load(checkpoints, prefix, result);
            pfx_load(checkpoints, helper, prod);
            unsigned int bit_base = PFX_XLEN - 1 - j_start / 2;
            for (unsigned int i = 0; i < n; i++) {
                unsigned int x_i = (unsigned int)((xv >> (n - 1 - i)) & 1ULL);
                unsigned int y_i = (unsigned int)((yv >> (n - 1 - i)) & 1ULL);
                if (y_i == 0 && x_i == 1) {
                    u64 pow[LIMBS], term[LIMBS], sum[LIMBS];
                    pfx_from_u64(sfx_shl_u64(1ULL, bit_base - i), pow);
                    fr_mul(prod, pow, term);
                    fr_add(result, term, sum);
                    store4(result, sum);
                }
                if (y_i == 1) {
                    u64 doubled[LIMBS];
                    fr_add(prod, prod, doubled);
                    store4(prod, doubled);
                }
            }
            store4(out, result);
            return;
        }
        case PFX_LEFT_SHIFT_HELPER: {
            sfx_uninterleave(b, &x, &y);
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, PFX_LEFT_SHIFT_HELPER, base);
            pfx_from_u64(sfx_shl_u64(1ULL, (unsigned int)__popcll((long long)sfx_u64(y))), term);
            fr_mul(base, term, out);
            return;
        }
        case PFX_LEFT_SHIFT_W_HELPER: {
            if (j_start < PFX_XLEN) { pfx_one(out); return; }
            sfx_uninterleave(b, &x, &y);
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, PFX_LEFT_SHIFT_W_HELPER, base);
            pfx_from_u64(sfx_shl_u64(1ULL, (unsigned int)__popcll((long long)sfx_u64(y))), term);
            fr_mul(base, term, out);
            return;
        }
        case PFX_TWO_LSB: {
            if (suffix_len == 0) {
                unsigned int v = (unsigned int)b.bits;
                unsigned int tz = v == 0 ? 32u : (unsigned int)(__ffs((int)v) - 1);
                if (tz >= 2) { pfx_one(out); } else { pfx_zero(out); }
            } else {
                pfx_load(checkpoints, PFX_TWO_LSB, out);
            }
            return;
        }
        case PFX_SIGN_EXT_UPPER_HALF: {
            unsigned int half = PFX_XLEN / 2;
            if (suffix_len >= half) { pfx_one(out); return; }
            unsigned int sign_round = PFX_XLEN + half;
            if (j_start <= sign_round && sign_round < j_start + b.len) {
                sfx_uninterleave(b, &x, &y);
                unsigned int sign = pfx_msb(sfx_u64(x), x.len);
                u64 mask[LIMBS], bit[LIMBS];
                pfx_from_u128((((u128)1 << half) - 1) << half, mask);
                pfx_from_u64((unsigned long long)sign, bit);
                fr_mul(mask, bit, out);
            } else {
                pfx_load(checkpoints, PFX_SIGN_EXT_UPPER_HALF, out);
            }
            return;
        }
        case PFX_CHANGE_DIVISOR: {
            sfx_uninterleave(b, &x, &y);
            unsigned long long xv = sfx_u64(x), yv = sfx_u64(y);
            if (j_start == 0) {
                if (pfx_msb(xv, x.len) == 0) { pfx_zero(out); return; }
                unsigned long long rest = xv & pfx_ones_u64(x.len - 1);
                if (rest != 0 || yv != pfx_ones_u64(y.len)) { pfx_zero(out); return; }
                pfx_load(checkpoints, PFX_CHANGE_DIVISOR, out);
                return;
            }
            if (xv != 0 || yv != pfx_ones_u64(y.len)) { pfx_zero(out); return; }
            pfx_load(checkpoints, PFX_CHANGE_DIVISOR, out);
            return;
        }
        case PFX_CHANGE_DIVISOR_W: {
            if (j_start < PFX_XLEN) { pfx_zero(out); return; }
            sfx_uninterleave(b, &x, &y);
            unsigned long long xv = sfx_u64(x), yv = sfx_u64(y);
            if (j_start == PFX_XLEN) {
                if (pfx_msb(xv, x.len) == 0) { pfx_zero(out); return; }
                unsigned long long rest = xv & pfx_ones_u64(x.len - 1);
                if (rest != 0 || yv != pfx_ones_u64(y.len)) { pfx_zero(out); return; }
                u64 two[LIMBS], big[LIMBS];
                pfx_from_u64(2ULL, two);
                pfx_from_u128((u128)1 << PFX_XLEN, big);
                fr_sub(two, big, out);
                return;
            }
            if (xv != 0 || yv != pfx_ones_u64(y.len)) { pfx_zero(out); return; }
            pfx_load(checkpoints, PFX_CHANGE_DIVISOR_W, out);
            return;
        }
        case PFX_RIGHT_OPERAND: {
            sfx_uninterleave(b, &x, &y);
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, PFX_RIGHT_OPERAND, base);
            pfx_from_u128(y.bits << (suffix_len / 2), term);
            fr_add(base, term, out);
            return;
        }
        case PFX_RIGHT_OPERAND_W: {
            u64 base[LIMBS];
            pfx_load(checkpoints, PFX_RIGHT_OPERAND_W, base);
            if (suffix_len < PFX_XLEN) {
                sfx_uninterleave(b, &x, &y);
                u64 term[LIMBS];
                pfx_from_u128(y.bits << (suffix_len / 2), term);
                fr_add(base, term, out);
            } else {
                store4(out, base);
            }
            return;
        }
        case PFX_SIGN_EXT_RIGHT_OPERAND: {
            if (suffix_len >= PFX_XLEN) { pfx_one(out); return; }
            if (j_start >= PFX_XLEN + 2) {
                pfx_load(checkpoints, PFX_SIGN_EXT_RIGHT_OPERAND, out);
                return;
            }
            sfx_uninterleave(b, &x, &y);
            unsigned int sign = pfx_msb(sfx_u64(y), y.len);
            u64 mask[LIMBS], bit[LIMBS];
            pfx_from_u128(((u128)1 << PFX_XLEN) - ((u128)1 << (PFX_XLEN / 2)), mask);
            pfx_from_u64((unsigned long long)sign, bit);
            fr_mul(mask, bit, out);
            return;
        }
        case PFX_OVERFLOW_BITS_ZERO: {
            if (j_start >= 128 - PFX_XLEN) {
                pfx_load(checkpoints, PFX_OVERFLOW_BITS_ZERO, out);
                return;
            }
            u128 overflow = suffix_len >= PFX_XLEN ? b.bits : (b.bits >> (PFX_XLEN - suffix_len));
            if (overflow != 0) { pfx_zero(out); } else {
                pfx_load(checkpoints, PFX_OVERFLOW_BITS_ZERO, out);
            }
            return;
        }
        case PFX_XOR_ROT16:
        case PFX_XOR_ROT24:
        case PFX_XOR_ROT32:
        case PFX_XOR_ROT63: {
            const unsigned int rots[4] = {16u, 24u, 32u, 63u};
            unsigned int rotation = rots[prefix - PFX_XOR_ROT16];
            sfx_uninterleave(b, &x, &y);
            unsigned long long xor_val = sfx_u64(x) ^ sfx_u64(y);
            unsigned int half = suffix_len / 2;
            unsigned int shift = half >= rotation ? half - rotation
                                                  : PFX_XLEN + half - rotation;
            unsigned long long rotated;
            shift &= 63u;
            rotated = shift == 0 ? xor_val
                                 : ((xor_val << shift) | (xor_val >> (64 - shift)));
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, prefix, base);
            pfx_from_u64(rotated, term);
            fr_add(base, term, out);
            return;
        }
        case PFX_XOR_ROTW7:
        case PFX_XOR_ROTW8:
        case PFX_XOR_ROTW12:
        case PFX_XOR_ROTW16: {
            if (j_start < PFX_XLEN) { pfx_zero(out); return; }
            const unsigned int rots[4] = {7u, 8u, 12u, 16u};
            unsigned int rotation = rots[prefix - PFX_XOR_ROTW7];
            sfx_uninterleave(b, &x, &y);
            unsigned int xor_val = (unsigned int)sfx_u64(x) ^ (unsigned int)sfx_u64(y);
            unsigned int half = suffix_len / 2;
            unsigned int shift = half >= rotation ? half - rotation : 32u + half - rotation;
            shift &= 31u;
            unsigned int rotated = shift == 0 ? xor_val
                                              : ((xor_val << shift) | (xor_val >> (32 - shift)));
            u64 base[LIMBS], term[LIMBS];
            pfx_load(checkpoints, prefix, base);
            pfx_from_u64((unsigned long long)rotated, term);
            fr_add(base, term, out);
            return;
        }
        default:
            pfx_zero(out);
            return;
    }
}

extern "C" __global__ void pfx_default_checkpoints_kernel(u64 *__restrict__ out,
                                                          unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 value[LIMBS];
    switch (i) {
        case PFX_EQ:
        case PFX_LEFT_IS_ZERO:
        case PFX_RIGHT_IS_ZERO:
        case PFX_DIV_BY_ZERO:
        case PFX_POS_REM_EQ_DIV:
        case PFX_NEG_DIV_ZERO_REM:
        case PFX_NEG_DIV_EQ_REM:
        case PFX_LSB:
        case PFX_POW2:
        case PFX_POW2W:
        case PFX_LEFT_SHIFT_HELPER:
        case PFX_TWO_LSB:
        case PFX_SIGN_EXT_UPPER_HALF:
        case PFX_LEFT_SHIFT_W_HELPER:
        case PFX_OVERFLOW_BITS_ZERO:
            pfx_one(value);
            break;
        case PFX_CHANGE_DIVISOR: {
            u64 two[LIMBS], big[LIMBS];
            pfx_from_u64(2ULL, two);
            pfx_from_u128((u128)1 << PFX_XLEN, big);
            fr_sub(two, big, value);
            break;
        }
        default:
            pfx_zero(value);
            break;
    }
    store4(out + (unsigned long long)i * LIMBS, value);
}

extern "C" __global__ void pfx_eval_batch_kernel(const u64 *__restrict__ checkpoints,
                                                 const unsigned long long *__restrict__ bits,
                                                 const unsigned char *__restrict__ lens,
                                                 unsigned int prefix,
                                                 unsigned int suffix_len,
                                                 u64 *__restrict__ out,
                                                 unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sfx_bits b;
    b.bits = ((u128)bits[2 * i + 1] << 64) | (u128)bits[2 * i];
    b.len = lens[i];
    b.bits &= sfx_mask(b.len);
    u64 value[LIMBS];
    pfx_eval(prefix, checkpoints, b, suffix_len, value);
    store4(out + (unsigned long long)i * LIMBS, value);
}
