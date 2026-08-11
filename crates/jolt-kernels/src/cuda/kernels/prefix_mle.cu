__device__ __forceinline__ unsigned int pml_pop_msb(sfx_bits *b) {
    if (b->len == 0u) return 0u;
    unsigned int bit = (unsigned int)((b->bits >> (b->len - 1u)) & (u128)1);
    b->len -= 1u;
    b->bits &= sfx_mask(b->len);
    return bit;
}

__device__ __forceinline__ void pml_from_u32(unsigned int v, u64 *out) {
    pfx_from_u64((unsigned long long)v, out);
}

__device__ __forceinline__ void pml_one_minus(const u64 *v, u64 *out) {
    u64 one[LIMBS];
    load4(FR_ONE, one);
    fr_sub(one, v, out);
}

__device__ __forceinline__ unsigned int pml_leading_ones(unsigned long long v, unsigned int len) {
    unsigned int count = 0;
    for (unsigned int i = 0; i < len; i++) {
        if (((v >> (len - 1 - i)) & 1ULL) == 0ULL) break;
        count++;
    }
    return count;
}

__device__ __forceinline__ unsigned int pml_trailing_zeros(unsigned long long v, unsigned int len) {
    if (len == 0) return 0;
    unsigned long long masked = v & pfx_ones_u64(len);
    if (masked == 0ULL) return len;
    unsigned int count = 0;
    while (((masked >> count) & 1ULL) == 0ULL) count++;
    return count;
}

__device__ __forceinline__ unsigned long long pml_rotate_left(unsigned long long v,
                                                             unsigned int shift) {
    shift &= 63u;
    if (shift == 0u) return v;
    return (v << shift) | (v >> (64u - shift));
}

__device__ __forceinline__ unsigned long long pml_shl(unsigned long long v, unsigned int shift) {
    return shift >= 64u ? 0ULL : (v << shift);
}

__device__ __forceinline__ void pml_eq_pair(const u64 *a, const u64 *b, u64 *out) {
    u64 ab[LIMBS], na[LIMBS], nb[LIMBS], nanb[LIMBS];
    fr_mul(a, b, ab);
    pml_one_minus(a, na);
    pml_one_minus(b, nb);
    fr_mul(na, nb, nanb);
    fr_add(ab, nanb, out);
}

__device__ __forceinline__ void pml_xor_pair(const u64 *a, const u64 *b, u64 *out) {
    u64 na[LIMBS], nb[LIMBS], t0[LIMBS], t1[LIMBS];
    pml_one_minus(a, na);
    pml_one_minus(b, nb);
    fr_mul(na, b, t0);
    fr_mul(a, nb, t1);
    fr_add(t0, t1, out);
}

struct pml_args {
    const u64 *checkpoints;
    const u64 *r_x;
    unsigned int has_r_x;
    unsigned int c;
    sfx_bits b;
    unsigned int round;
};

__device__ __forceinline__ unsigned int pml_suffix_len(const pml_args *a) {
    return PFX_LOG_K - a->round - a->b.len - 1u;
}

__device__ int pml_eval(unsigned int prefix, const pml_args *a, u64 *out) {
    unsigned int suffix_len = pml_suffix_len(a);
    unsigned int j = a->round;
    u64 c_f[LIMBS];
    pml_from_u32(a->c, c_f);

    switch (prefix) {
        case PFX_UPPER_WORD: {
            u64 result[LIMBS];
            pfx_load(a->checkpoints, PFX_UPPER_WORD, result);
            if (j >= PFX_XLEN) { store4(out, result); return 1; }
            sfx_bits b = a->b;
            u64 term[LIMBS], scaled[LIMBS], sum[LIMBS];
            if (a->has_r_x) {
                pfx_from_u64(1ULL << (PFX_XLEN - j), term);
                fr_mul(term, a->r_x, scaled);
                fr_add(result, scaled, sum);
                store4(result, sum);
                pfx_from_u64(1ULL << (PFX_XLEN - j - 1), term);
                fr_mul(term, c_f, scaled);
                fr_add(result, scaled, sum);
                store4(result, sum);
            } else {
                unsigned int y_msb = pml_pop_msb(&b);
                pfx_from_u64(1ULL << (PFX_XLEN - j - 1), term);
                fr_mul(term, c_f, scaled);
                fr_add(result, scaled, sum);
                store4(result, sum);
                u64 y_f[LIMBS];
                pml_from_u32(y_msb, y_f);
                pfx_from_u64(1ULL << (PFX_XLEN - j - 2), term);
                fr_mul(term, y_f, scaled);
                fr_add(result, scaled, sum);
                store4(result, sum);
            }
            if (suffix_len > PFX_XLEN) {
                pfx_from_u64(pml_shl(sfx_u64(b), suffix_len - PFX_XLEN), term);
            } else {
                sfx_bits hi, lo;
                sfx_split(b, PFX_XLEN - suffix_len, &hi, &lo);
                pfx_from_u64(sfx_u64(hi), term);
            }
            fr_add(result, term, sum);
            store4(out, sum);
            return 1;
        }
        case PFX_EQ: {
            u64 result[LIMBS];
            pfx_load(a->checkpoints, PFX_EQ, result);
            sfx_bits b = a->b;
            u64 factor[LIMBS], product[LIMBS];
            if (a->has_r_x) {
                pml_eq_pair(a->r_x, c_f, factor);
            } else {
                u64 y_f[LIMBS];
                pml_from_u32(pml_pop_msb(&b), y_f);
                pml_eq_pair(c_f, y_f, factor);
            }
            fr_mul(result, factor, product);
            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            if (sfx_u64(x) != sfx_u64(y)) { pfx_zero(out); return 1; }
            store4(out, product);
            return 1;
        }
        case PFX_AND:
        case PFX_ANDN:
        case PFX_OR:
        case PFX_XOR: {
            u64 result[LIMBS];
            pfx_load(a->checkpoints, prefix, result);
            sfx_bits b = a->b;
            unsigned int shift = PFX_XLEN - 1u - j / 2u;
            u64 bit[LIMBS];
            if (a->has_r_x) {
                if (prefix == PFX_AND) {
                    fr_mul(a->r_x, c_f, bit);
                } else if (prefix == PFX_ANDN) {
                    u64 nc[LIMBS];
                    pml_one_minus(c_f, nc);
                    fr_mul(a->r_x, nc, bit);
                } else if (prefix == PFX_OR) {
                    u64 prod[LIMBS], sum[LIMBS];
                    fr_mul(a->r_x, c_f, prod);
                    fr_add(a->r_x, c_f, sum);
                    fr_sub(sum, prod, bit);
                } else {
                    pml_xor_pair(a->r_x, c_f, bit);
                }
            } else {
                u64 y_f[LIMBS];
                pml_from_u32(pml_pop_msb(&b), y_f);
                if (prefix == PFX_AND) {
                    fr_mul(c_f, y_f, bit);
                } else if (prefix == PFX_ANDN) {
                    u64 ny[LIMBS];
                    pml_one_minus(y_f, ny);
                    fr_mul(c_f, ny, bit);
                } else if (prefix == PFX_OR) {
                    u64 prod[LIMBS], sum[LIMBS];
                    fr_mul(c_f, y_f, prod);
                    fr_add(c_f, y_f, sum);
                    fr_sub(sum, prod, bit);
                } else {
                    pml_xor_pair(c_f, y_f, bit);
                }
            }
            u64 pow[LIMBS], scaled[LIMBS], sum[LIMBS];
            pfx_from_u64(1ULL << shift, pow);
            fr_mul(pow, bit, scaled);
            fr_add(result, scaled, sum);
            store4(result, sum);

            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            unsigned long long xv = sfx_u64(x), yv = sfx_u64(y);
            unsigned long long combined;
            if (prefix == PFX_AND) {
                combined = xv & yv;
            } else if (prefix == PFX_ANDN) {
                combined = xv & ~yv;
            } else if (prefix == PFX_OR) {
                combined = xv | yv;
            } else {
                combined = xv ^ yv;
            }
            u64 tail[LIMBS];
            pfx_from_u64(pml_shl(combined, suffix_len / 2u), tail);
            fr_add(result, tail, sum);
            store4(out, sum);
            return 1;
        }
        case PFX_LESS_THAN: {
            u64 lt[LIMBS], eq[LIMBS];
            pfx_load(a->checkpoints, PFX_LESS_THAN, lt);
            pfx_load(a->checkpoints, PFX_EQ, eq);
            sfx_bits b = a->b;

            u64 lhs[LIMBS], rhs[LIMBS];
            if (a->has_r_x) {
                pml_one_minus(a->r_x, lhs);
                store4(rhs, c_f);
            } else {
                u64 y_f[LIMBS];
                pml_from_u32(pml_pop_msb(&b), y_f);
                pml_one_minus(c_f, lhs);
                store4(rhs, y_f);
            }

            u64 first[LIMBS], term[LIMBS], sum[LIMBS];
            fr_mul(lhs, rhs, first);
            fr_mul(eq, first, term);
            fr_add(lt, term, sum);
            store4(lt, sum);

            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            if (sfx_u64(x) < sfx_u64(y)) {
                u64 factor[LIMBS], updated[LIMBS];
                if (a->has_r_x) {
                    pml_eq_pair(a->r_x, c_f, factor);
                } else {
                    pml_eq_pair(c_f, rhs, factor);
                }
                fr_mul(eq, factor, updated);
                fr_add(lt, updated, sum);
                store4(lt, sum);
            }
            store4(out, lt);
            return 1;
        }
        case PFX_RIGHT_SHIFT: {
            u64 result[LIMBS];
            pfx_load(a->checkpoints, PFX_RIGHT_SHIFT, result);
            sfx_bits b = a->b;
            u64 factor[LIMBS], scaled[LIMBS], addend[LIMBS], sum[LIMBS];
            if (a->has_r_x) {
                pml_from_u32(1u + a->c, factor);
                fr_mul(result, factor, scaled);
                fr_mul(a->r_x, c_f, addend);
                fr_add(scaled, addend, sum);
                store4(result, sum);
            } else {
                unsigned int y_msb = pml_pop_msb(&b);
                pml_from_u32(1u + y_msb, factor);
                fr_mul(result, factor, scaled);
                pml_from_u32(a->c * y_msb, addend);
                fr_add(scaled, addend, sum);
                store4(result, sum);
            }
            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            unsigned int half = b.len / 2u;
            pfx_from_u64(1ULL << pml_leading_ones(sfx_u64(y), half), factor);
            fr_mul(result, factor, scaled);
            pfx_from_u64(sfx_u64(x) >> pml_trailing_zeros(sfx_u64(y), half), addend);
            fr_add(scaled, addend, sum);
            store4(out, sum);
            return 1;
        }
        case PFX_LEFT_SHIFT_HELPER: {
            u64 result[LIMBS];
            pfx_load(a->checkpoints, PFX_LEFT_SHIFT_HELPER, result);
            sfx_bits b = a->b;
            u64 factor[LIMBS], scaled[LIMBS];
            if (a->has_r_x) {
                pml_from_u32(1u + a->c, factor);
            } else {
                pml_from_u32(1u + pml_pop_msb(&b), factor);
            }
            fr_mul(result, factor, scaled);
            store4(result, scaled);
            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            pfx_from_u64(1ULL << pml_leading_ones(sfx_u64(y), b.len / 2u), factor);
            fr_mul(result, factor, scaled);
            store4(out, scaled);
            return 1;
        }
        case PFX_XOR_ROT16:
        case PFX_XOR_ROT24:
        case PFX_XOR_ROT32:
        case PFX_XOR_ROT63: {
            unsigned int rotation;
            if (prefix == PFX_XOR_ROT16) rotation = 16u;
            else if (prefix == PFX_XOR_ROT24) rotation = 24u;
            else if (prefix == PFX_XOR_ROT32) rotation = 32u;
            else rotation = 63u;

            u64 result[LIMBS];
            pfx_load(a->checkpoints, prefix, result);
            sfx_bits b = a->b;
            u64 bit[LIMBS];
            if (a->has_r_x) {
                pml_xor_pair(a->r_x, c_f, bit);
            } else {
                u64 y_f[LIMBS];
                pml_from_u32(pml_pop_msb(&b), y_f);
                pml_xor_pair(c_f, y_f, bit);
            }
            unsigned int rotated_pos = (j / 2u + rotation) % PFX_XLEN;
            unsigned int shift = PFX_XLEN - 1u - rotated_pos;
            u64 pow[LIMBS], scaled[LIMBS], sum[LIMBS];
            pfx_from_u64(1ULL << shift, pow);
            fr_mul(pow, bit, scaled);
            fr_add(result, scaled, sum);
            store4(result, sum);

            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            int half = (int)(suffix_len / 2u);
            unsigned int tail_shift;
            if (half - (int)rotation >= 0) {
                tail_shift = (unsigned int)(half - (int)rotation);
            } else {
                tail_shift = (unsigned int)((int)PFX_XLEN + (half - (int)rotation));
            }
            u64 tail[LIMBS];
            pfx_from_u64(pml_rotate_left(sfx_u64(x) ^ sfx_u64(y), tail_shift), tail);
            fr_add(result, tail, sum);
            store4(out, sum);
            return 1;
        }
        case PFX_LOWER_WORD:
        case PFX_LOWER_HALF_WORD: {
            unsigned int floor = prefix == PFX_LOWER_WORD ? PFX_XLEN
                                                          : PFX_XLEN + PFX_XLEN / 2u;
            if (j < floor) { pfx_zero(out); return 1; }
            u64 result[LIMBS];
            pfx_load(a->checkpoints, prefix, result);
            sfx_bits b = a->b;
            u64 term[LIMBS], scaled[LIMBS], sum[LIMBS];
            if (a->has_r_x) {
                pfx_from_u128((u128)1 << (2u * PFX_XLEN - j), term);
                fr_mul(term, a->r_x, scaled);
                fr_add(result, scaled, sum);
                store4(result, sum);
                pfx_from_u128((u128)1 << (2u * PFX_XLEN - j - 1u), term);
                fr_mul(term, c_f, scaled);
                fr_add(result, scaled, sum);
                store4(result, sum);
            } else {
                unsigned int y_msb = pml_pop_msb(&b);
                pfx_from_u128((u128)1 << (2u * PFX_XLEN - j - 1u), term);
                fr_mul(term, c_f, scaled);
                fr_add(result, scaled, sum);
                store4(result, sum);
                u64 y_f[LIMBS];
                pml_from_u32(y_msb, y_f);
                pfx_from_u128((u128)1 << (2u * PFX_XLEN - j - 2u), term);
                fr_mul(term, y_f, scaled);
                fr_add(result, scaled, sum);
                store4(result, sum);
            }
            pfx_from_u128(b.bits << suffix_len, term);
            fr_add(result, term, sum);
            store4(out, sum);
            return 1;
        }
        case PFX_LEFT_IS_ZERO: {
            sfx_bits x, y;
            sfx_uninterleave(a->b, &x, &y);
            if (sfx_u64(x) != 0ULL) { pfx_zero(out); return 1; }
            u64 result[LIMBS], factor[LIMBS], product[LIMBS];
            pfx_load(a->checkpoints, PFX_LEFT_IS_ZERO, result);
            pml_one_minus(a->has_r_x ? a->r_x : c_f, factor);
            fr_mul(result, factor, product);
            store4(out, product);
            return 1;
        }
        case PFX_RIGHT_IS_ZERO: {
            sfx_bits b = a->b;
            u64 factor[LIMBS];
            if (a->has_r_x) {
                pml_one_minus(c_f, factor);
            } else {
                u64 y_f[LIMBS];
                pml_from_u32(pml_pop_msb(&b), y_f);
                pml_one_minus(y_f, factor);
            }
            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            if (sfx_u64(y) != 0ULL) { pfx_zero(out); return 1; }
            u64 result[LIMBS], product[LIMBS];
            pfx_load(a->checkpoints, PFX_RIGHT_IS_ZERO, result);
            fr_mul(result, factor, product);
            store4(out, product);
            return 1;
        }
        case PFX_LEFT_MSB: {
            if (j == 0u) { store4(out, c_f); return 1; }
            if (j == 1u) { store4(out, a->r_x); return 1; }
            pfx_load(a->checkpoints, PFX_LEFT_MSB, out);
            return 1;
        }
        case PFX_RIGHT_MSB: {
            if (j == 0u) {
                sfx_bits b = a->b;
                pml_from_u32(pml_pop_msb(&b), out);
                return 1;
            }
            if (j == 1u) { store4(out, c_f); return 1; }
            pfx_load(a->checkpoints, PFX_RIGHT_MSB, out);
            return 1;
        }
        case PFX_DIV_BY_ZERO: {
            sfx_bits b = a->b;
            u64 factor[LIMBS];
            if (a->has_r_x) {
                u64 nr[LIMBS];
                pml_one_minus(a->r_x, nr);
                fr_mul(nr, c_f, factor);
            } else {
                u64 y_f[LIMBS], nc[LIMBS];
                pml_from_u32(pml_pop_msb(&b), y_f);
                pml_one_minus(c_f, nc);
                fr_mul(nc, y_f, factor);
            }
            sfx_bits divisor, quotient;
            sfx_uninterleave(b, &divisor, &quotient);
            if (sfx_u64(divisor) != 0ULL ||
                sfx_u64(quotient) != pfx_ones_u64(quotient.len)) {
                pfx_zero(out);
                return 1;
            }
            u64 result[LIMBS], product[LIMBS];
            pfx_load(a->checkpoints, PFX_DIV_BY_ZERO, result);
            fr_mul(result, factor, product);
            store4(out, product);
            return 1;
        }
        case PFX_POS_REM_EQ_DIV:
        case PFX_NEG_DIV_EQ_REM: {
            unsigned int negative = prefix == PFX_NEG_DIV_EQ_REM;
            sfx_bits b = a->b;
            if (j == 0u) {
                u64 sign[LIMBS];
                pml_from_u32(pml_pop_msb(&b), sign);
                sfx_bits rem, div;
                sfx_uninterleave(b, &rem, &div);
                if (sfx_u64(rem) != sfx_u64(div)) { pfx_zero(out); return 1; }
                if (negative) {
                    fr_mul(c_f, sign, out);
                } else {
                    u64 nc[LIMBS], nsign[LIMBS];
                    pml_one_minus(c_f, nc);
                    pml_one_minus(sign, nsign);
                    fr_mul(nc, nsign, out);
                }
                return 1;
            }
            if (j == 1u) {
                sfx_bits rem, div;
                sfx_uninterleave(b, &rem, &div);
                if (sfx_u64(rem) != sfx_u64(div)) { pfx_zero(out); return 1; }
                if (negative) {
                    fr_mul(a->r_x, c_f, out);
                } else {
                    u64 nr[LIMBS], nc[LIMBS];
                    pml_one_minus(a->r_x, nr);
                    pml_one_minus(c_f, nc);
                    fr_mul(nr, nc, out);
                }
                return 1;
            }
            u64 factor[LIMBS];
            if (a->has_r_x) {
                pml_eq_pair(a->r_x, c_f, factor);
            } else {
                u64 y_f[LIMBS];
                pml_from_u32(pml_pop_msb(&b), y_f);
                pml_eq_pair(c_f, y_f, factor);
            }
            sfx_bits rem, div;
            sfx_uninterleave(b, &rem, &div);
            if (sfx_u64(rem) != sfx_u64(div)) { pfx_zero(out); return 1; }
            u64 checkpoint[LIMBS], product[LIMBS];
            pfx_load(a->checkpoints, prefix, checkpoint);
            fr_mul(checkpoint, factor, product);
            store4(out, product);
            return 1;
        }
        case PFX_POS_REM_LT_DIV:
        case PFX_NEG_DIV_GT_REM: {
            unsigned int negative = prefix == PFX_NEG_DIV_GT_REM;
            unsigned int eq_index = negative ? PFX_NEG_DIV_EQ_REM : PFX_POS_REM_EQ_DIV;
            sfx_bits b = a->b;
            if (j == 0u) {
                u64 sign[LIMBS];
                pml_from_u32(pml_pop_msb(&b), sign);
                sfx_bits rem, div;
                sfx_uninterleave(b, &rem, &div);
                unsigned long long rv = sfx_u64(rem), dv = sfx_u64(div);
                int fail = negative ? (rv <= dv) : (rv >= dv);
                if (fail) { pfx_zero(out); return 1; }
                if (negative) {
                    fr_mul(c_f, sign, out);
                } else {
                    u64 nc[LIMBS], nsign[LIMBS];
                    pml_one_minus(c_f, nc);
                    pml_one_minus(sign, nsign);
                    fr_mul(nc, nsign, out);
                }
                return 1;
            }
            if (j == 1u) {
                sfx_bits rem, div;
                sfx_uninterleave(b, &rem, &div);
                unsigned long long rv = sfx_u64(rem), dv = sfx_u64(div);
                int fail = negative ? (rv <= dv) : (rv >= dv);
                if (fail) { pfx_zero(out); return 1; }
                if (negative) {
                    fr_mul(a->r_x, c_f, out);
                } else {
                    u64 nr[LIMBS], nc[LIMBS];
                    pml_one_minus(a->r_x, nr);
                    pml_one_minus(c_f, nc);
                    fr_mul(nr, nc, out);
                }
                return 1;
            }

            u64 acc[LIMBS], eq[LIMBS];
            pfx_load(a->checkpoints, prefix, acc);
            pfx_load(a->checkpoints, eq_index, eq);

            u64 lhs[LIMBS], rhs[LIMBS], first[LIMBS], sum[LIMBS];
            if (a->has_r_x) {
                if (negative) {
                    store4(lhs, a->r_x);
                    pml_one_minus(c_f, rhs);
                } else {
                    pml_one_minus(a->r_x, lhs);
                    store4(rhs, c_f);
                }
            } else {
                u64 y_f[LIMBS];
                pml_from_u32(pml_pop_msb(&b), y_f);
                if (negative) {
                    store4(lhs, c_f);
                    pml_one_minus(y_f, rhs);
                } else {
                    pml_one_minus(c_f, lhs);
                    store4(rhs, y_f);
                }
            }
            fr_mul(lhs, rhs, first);

            if (j == 2u || j == 3u) {
                u64 scaled[LIMBS];
                fr_mul(acc, first, scaled);
                store4(acc, scaled);
            } else {
                u64 term[LIMBS];
                fr_mul(eq, first, term);
                fr_add(acc, term, sum);
                store4(acc, sum);
            }

            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            unsigned long long xv = sfx_u64(x), yv = sfx_u64(y);
            int advance = negative ? (xv > yv) : (xv < yv);
            if (advance) {
                u64 factor[LIMBS], updated[LIMBS];
                if (a->has_r_x) {
                    pml_eq_pair(a->r_x, c_f, factor);
                } else {
                    u64 y_f[LIMBS];
                    sfx_bits tmp = a->b;
                    pml_from_u32(pml_pop_msb(&tmp), y_f);
                    pml_eq_pair(c_f, y_f, factor);
                }
                fr_mul(eq, factor, updated);
                fr_add(acc, updated, sum);
                store4(acc, sum);
            }
            store4(out, acc);
            return 1;
        }
        case PFX_NEG_DIV_ZERO_REM: {
            sfx_bits b = a->b;
            if (j == 0u) {
                u64 sign[LIMBS];
                pml_from_u32(pml_pop_msb(&b), sign);
                sfx_bits rem, div;
                sfx_uninterleave(b, &rem, &div);
                if (sfx_u64(rem) != 0ULL) { pfx_zero(out); return 1; }
                u64 nc[LIMBS];
                pml_one_minus(c_f, nc);
                fr_mul(nc, sign, out);
                return 1;
            }
            if (j == 1u) {
                sfx_bits rem, div;
                sfx_uninterleave(b, &rem, &div);
                if (sfx_u64(rem) != 0ULL) { pfx_zero(out); return 1; }
                u64 nr[LIMBS];
                pml_one_minus(a->r_x, nr);
                fr_mul(nr, c_f, out);
                return 1;
            }
            u64 factor[LIMBS];
            if (a->has_r_x) {
                pml_one_minus(a->r_x, factor);
            } else {
                (void)pml_pop_msb(&b);
                pml_one_minus(c_f, factor);
            }
            sfx_bits rem, div;
            sfx_uninterleave(b, &rem, &div);
            if (sfx_u64(rem) != 0ULL) { pfx_zero(out); return 1; }
            u64 checkpoint[LIMBS], product[LIMBS];
            pfx_load(a->checkpoints, PFX_NEG_DIV_ZERO_REM, checkpoint);
            fr_mul(checkpoint, factor, product);
            store4(out, product);
            return 1;
        }
        case PFX_LSB: {
            if (j == 2u * PFX_XLEN - 1u) { store4(out, c_f); return 1; }
            if (suffix_len == 0u) {
                pml_from_u32((unsigned int)(sfx_u64(a->b) & 1ULL), out);
                return 1;
            }
            pfx_one(out);
            return 1;
        }
        case PFX_TWO_LSB: {
            if (j == 2u * PFX_XLEN - 1u) {
                u64 nc[LIMBS], nr[LIMBS];
                pml_one_minus(c_f, nc);
                pml_one_minus(a->r_x, nr);
                fr_mul(nc, nr, out);
                return 1;
            }
            if (j == 2u * PFX_XLEN - 2u) {
                u64 bit0[LIMBS], nb0[LIMBS], nc[LIMBS];
                pml_from_u32((unsigned int)(sfx_u64(a->b) & 1ULL), bit0);
                pml_one_minus(bit0, nb0);
                pml_one_minus(c_f, nc);
                fr_mul(nb0, nc, out);
                return 1;
            }
            if (suffix_len == 0u) {
                if ((sfx_u64(a->b) & 3ULL) == 0ULL) { pfx_one(out); } else { pfx_zero(out); }
                return 1;
            }
            pfx_one(out);
            return 1;
        }
        case PFX_POW2:
        case PFX_POW2W: {
            unsigned int bits_needed = prefix == PFX_POW2 ? 6u : 5u;
            unsigned int mask = prefix == PFX_POW2 ? (PFX_XLEN - 1u) : 31u;
            if (suffix_len != 0u) { pfx_one(out); return 1; }
            sfx_bits b = a->b;
            unsigned long long low = sfx_u64(b) & (unsigned long long)mask;
            u64 result[LIMBS];
            pfx_from_u64(1ULL << low, result);
            if (b.len >= bits_needed) { store4(out, result); return 1; }

            unsigned int num_bits = b.len;
            unsigned long long shift = 1ULL << (1ULL << num_bits);
            u64 factor[LIMBS], scaled[LIMBS];
            pfx_from_u64(1ULL + (shift - 1ULL) * (unsigned long long)a->c, factor);
            fr_mul(result, factor, scaled);
            store4(result, scaled);
            if (b.len == bits_needed - 1u) { store4(out, result); return 1; }

            num_bits += 1u;
            shift = 1ULL << (1ULL << num_bits);
            if (a->has_r_x) {
                u64 term[LIMBS], one[LIMBS], sum[LIMBS];
                pfx_from_u64(shift - 1ULL, term);
                fr_mul(term, a->r_x, scaled);
                load4(FR_ONE, one);
                fr_add(one, scaled, sum);
                fr_mul(result, sum, scaled);
                store4(result, scaled);
            }
            u64 checkpoint[LIMBS];
            pfx_load(a->checkpoints, prefix, checkpoint);
            fr_mul(result, checkpoint, scaled);
            store4(out, scaled);
            return 1;
        }
        case PFX_REV8W: {
            if (suffix_len >= 64u) { pfx_zero(out); return 1; }
            u64 eval[LIMBS];
            pfx_load(a->checkpoints, PFX_REV8W, eval);
            sfx_bits b = a->b;
            unsigned int c_bit_index = suffix_len + b.len;
            u64 sum[LIMBS];
            if (c_bit_index < 64u) {
                unsigned long long rev = sfx_rev8w(1ULL << c_bit_index);
                unsigned int shift = 0u;
                while (((rev >> shift) & 1ULL) == 0ULL) shift++;
                u64 term[LIMBS];
                pfx_from_u128((u128)a->c << shift, term);
                fr_add(eval, term, sum);
                store4(eval, sum);
            }
            if (c_bit_index + 1u < 64u && a->has_r_x) {
                unsigned long long rev_pow2 = sfx_rev8w(1ULL << (c_bit_index + 1u));
                u64 term[LIMBS], scaled[LIMBS];
                pfx_from_u64(rev_pow2, term);
                fr_mul(a->r_x, term, scaled);
                fr_add(eval, scaled, sum);
                store4(eval, sum);
            }
            u64 tail[LIMBS];
            pfx_from_u64(sfx_rev8w(pml_shl(sfx_u64(b), suffix_len)), tail);
            fr_add(eval, tail, sum);
            store4(out, sum);
            return 1;
        }
        case PFX_SIGN_EXTENSION: {
            sfx_bits b = a->b;
            if (j == 0u) {
                if (a->c == 0u) { pfx_zero(out); return 1; }
                (void)pml_pop_msb(&b);
                sfx_bits x, y;
                sfx_uninterleave(b, &x, &y);
                u64 result[LIMBS], term[LIMBS], sum[LIMBS];
                pfx_zero(result);
                unsigned int y_len = y.len;
                for (unsigned int index = 1u; index <= y_len; index++) {
                    unsigned int y_i = pml_pop_msb(&y);
                    pfx_from_u64((1ULL - (unsigned long long)y_i) << index, term);
                    fr_add(result, term, sum);
                    store4(result, sum);
                }
                fr_mul(result, c_f, sum);
                store4(out, sum);
                return 1;
            }
            if (j == 1u) {
                sfx_bits x, y;
                sfx_uninterleave(b, &x, &y);
                u64 result[LIMBS], term[LIMBS], sum[LIMBS];
                pfx_zero(result);
                unsigned int y_len = y.len;
                for (unsigned int index = 1u; index <= y_len; index++) {
                    unsigned int y_i = pml_pop_msb(&y);
                    pfx_from_u64((1ULL - (unsigned long long)y_i) << index, term);
                    fr_add(result, term, sum);
                    store4(result, sum);
                }
                fr_mul(result, a->r_x, sum);
                store4(out, sum);
                return 1;
            }
            u64 sign_bit[LIMBS], result[LIMBS];
            pfx_load(a->checkpoints, PFX_LEFT_MSB, sign_bit);
            pfx_load(a->checkpoints, PFX_SIGN_EXTENSION, result);
            u64 term[LIMBS], sum[LIMBS];
            if (a->has_r_x) {
                u64 nc[LIMBS], scaled[LIMBS];
                pml_one_minus(c_f, nc);
                pfx_from_u64(1ULL << (j / 2u), term);
                fr_mul(term, nc, scaled);
                fr_add(result, scaled, sum);
                store4(result, sum);
            } else {
                if (pml_pop_msb(&b) == 0u) {
                    pfx_from_u64(1ULL << (j / 2u), term);
                    fr_add(result, term, sum);
                    store4(result, sum);
                }
            }
            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            unsigned int index = j / 2u;
            unsigned int len = y.len;
            for (unsigned int i = 0u; i < len; i++) {
                index += 1u;
                if (pml_pop_msb(&y) == 0u) {
                    pfx_from_u64(1ULL << index, term);
                    fr_add(result, term, sum);
                    store4(result, sum);
                }
            }
            fr_mul(result, sign_bit, sum);
            store4(out, sum);
            return 1;
        }
        case PFX_SIGN_EXT_UPPER_HALF: {
            unsigned int half = PFX_XLEN / 2u;
            if (suffix_len >= half) { pfx_one(out); return 1; }
            u64 mask[LIMBS];
            pfx_from_u128((((u128)1 << half) - 1) << half, mask);
            if (j == PFX_XLEN + half) {
                fr_mul(mask, c_f, out);
                return 1;
            }
            if (j == PFX_XLEN + half + 1u) {
                fr_mul(mask, a->r_x, out);
                return 1;
            }
            pfx_load(a->checkpoints, PFX_SIGN_EXT_UPPER_HALF, out);
            return 1;
        }
        case PFX_SIGN_EXT_RIGHT_OPERAND: {
            if (suffix_len >= PFX_XLEN) { pfx_one(out); return 1; }
            u64 mask[LIMBS];
            pfx_from_u128(((u128)1 << PFX_XLEN) - ((u128)1 << (PFX_XLEN / 2u)), mask);
            if (j == PFX_XLEN) {
                sfx_bits b = a->b;
                u64 sign[LIMBS];
                pml_from_u32(pml_pop_msb(&b), sign);
                fr_mul(mask, sign, out);
                return 1;
            }
            if (j == PFX_XLEN + 1u) {
                fr_mul(mask, c_f, out);
                return 1;
            }
            pfx_load(a->checkpoints, PFX_SIGN_EXT_RIGHT_OPERAND, out);
            return 1;
        }
        case PFX_CHANGE_DIVISOR:
        case PFX_CHANGE_DIVISOR_W: {
            unsigned int wide = prefix == PFX_CHANGE_DIVISOR_W;
            unsigned int base_round = wide ? PFX_XLEN : 0u;
            if (wide && j < PFX_XLEN) { pfx_zero(out); return 1; }

            u64 result[LIMBS];
            if (!wide) {
                pfx_load(a->checkpoints, PFX_CHANGE_DIVISOR, result);
            } else if (j == PFX_XLEN || j == PFX_XLEN + 1u) {
                u64 two[LIMBS], big[LIMBS];
                pfx_from_u64(2ULL, two);
                pfx_from_u128((u128)1 << PFX_XLEN, big);
                fr_sub(two, big, result);
            } else {
                pfx_load(a->checkpoints, PFX_CHANGE_DIVISOR_W, result);
            }

            sfx_bits b = a->b;
            if (j == base_round) {
                if (pml_pop_msb(&b) == 0u) { pfx_zero(out); return 1; }
                sfx_bits x, y;
                sfx_uninterleave(b, &x, &y);
                if (sfx_u64(x) != 0ULL || sfx_u64(y) != pfx_ones_u64(y.len)) {
                    pfx_zero(out);
                    return 1;
                }
                u64 scaled[LIMBS];
                fr_mul(result, c_f, scaled);
                store4(out, scaled);
                return 1;
            }
            if (a->has_r_x) {
                sfx_bits x, y;
                sfx_uninterleave(b, &x, &y);
                if (sfx_u64(x) != 0ULL || sfx_u64(y) != pfx_ones_u64(y.len) || a->c == 0u) {
                    pfx_zero(out);
                    return 1;
                }
                u64 lhs[LIMBS], factor[LIMBS], scaled[LIMBS];
                if (j == base_round + 1u) {
                    store4(lhs, a->r_x);
                } else {
                    pml_one_minus(a->r_x, lhs);
                }
                fr_mul(lhs, c_f, factor);
                fr_mul(result, factor, scaled);
                store4(out, scaled);
                return 1;
            }
            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            if ((b.len > 0u && sfx_u64(x) != 0ULL) || sfx_u64(y) != pfx_ones_u64(y.len)) {
                pfx_zero(out);
                return 1;
            }
            u64 nc[LIMBS], scaled[LIMBS];
            pml_one_minus(c_f, nc);
            fr_mul(result, nc, scaled);
            store4(out, scaled);
            return 1;
        }
        case PFX_RIGHT_OPERAND:
        case PFX_RIGHT_OPERAND_W: {
            unsigned int wide = prefix == PFX_RIGHT_OPERAND_W;
            u64 result[LIMBS];
            pfx_load(a->checkpoints, prefix, result);
            u64 term[LIMBS], sum[LIMBS];
            if (j % 2u == 1u && (!wide || j > PFX_XLEN)) {
                unsigned int shift = PFX_XLEN - 1u - j / 2u;
                pfx_from_u128((u128)a->c << shift, term);
                fr_add(result, term, sum);
                store4(result, sum);
            }
            if (!wide || suffix_len < PFX_XLEN) {
                sfx_bits x, y;
                sfx_uninterleave(a->b, &x, &y);
                pfx_from_u128((u128)sfx_u64(y) << (suffix_len / 2u), term);
                fr_add(result, term, sum);
                store4(result, sum);
            }
            store4(out, result);
            return 1;
        }
        case PFX_RIGHT_SHIFT_W: {
            if (j < PFX_XLEN) { pfx_zero(out); return 1; }
            u64 result[LIMBS];
            pfx_load(a->checkpoints, PFX_RIGHT_SHIFT_W, result);
            sfx_bits b = a->b;
            u64 factor[LIMBS], scaled[LIMBS], addend[LIMBS], sum[LIMBS];
            if (a->has_r_x) {
                pml_from_u32(1u + a->c, factor);
                fr_mul(result, factor, scaled);
                fr_mul(a->r_x, c_f, addend);
                fr_add(scaled, addend, sum);
                store4(result, sum);
            } else {
                unsigned int y_msb = pml_pop_msb(&b);
                pml_from_u32(1u + y_msb, factor);
                fr_mul(result, factor, scaled);
                pml_from_u32(a->c * y_msb, addend);
                fr_add(scaled, addend, sum);
                store4(result, sum);
            }
            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            unsigned int half = b.len / 2u;
            pfx_from_u64(1ULL << pml_leading_ones(sfx_u64(y), half), factor);
            fr_mul(result, factor, scaled);
            pfx_from_u64(sfx_u64(x) >> pml_trailing_zeros(sfx_u64(y), half), addend);
            fr_add(scaled, addend, sum);
            store4(out, sum);
            return 1;
        }
        case PFX_LEFT_SHIFT_W_HELPER: {
            if (j < PFX_XLEN) { pfx_one(out); return 1; }
            u64 result[LIMBS];
            pfx_load(a->checkpoints, PFX_LEFT_SHIFT_W_HELPER, result);
            sfx_bits b = a->b;
            u64 factor[LIMBS], scaled[LIMBS];
            if (a->has_r_x) {
                pml_from_u32(1u + a->c, factor);
            } else {
                pml_from_u32(1u + pml_pop_msb(&b), factor);
            }
            fr_mul(result, factor, scaled);
            store4(result, scaled);
            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            pfx_from_u64(1ULL << pml_leading_ones(sfx_u64(y), b.len / 2u), factor);
            fr_mul(result, factor, scaled);
            store4(out, scaled);
            return 1;
        }
        case PFX_LEFT_SHIFT:
        case PFX_LEFT_SHIFT_W: {
            unsigned int wide = prefix == PFX_LEFT_SHIFT_W;
            if (wide && j < PFX_XLEN) { pfx_zero(out); return 1; }
            unsigned int helper_index = wide ? PFX_LEFT_SHIFT_W_HELPER : PFX_LEFT_SHIFT_HELPER;
            u64 result[LIMBS], helper[LIMBS];
            pfx_load(a->checkpoints, prefix, result);
            pfx_load(a->checkpoints, helper_index, helper);

            unsigned int bit_index = PFX_XLEN - 1u - j / 2u;
            sfx_bits b = a->b;
            u64 pow[LIMBS], term[LIMBS], scaled[LIMBS], sum[LIMBS];
            pfx_from_u64(bit_index >= 64u ? 0ULL : (1ULL << bit_index), pow);
            if (a->has_r_x) {
                u64 nc[LIMBS];
                pml_one_minus(c_f, nc);
                fr_mul(a->r_x, nc, term);
                fr_mul(term, helper, scaled);
                fr_mul(scaled, pow, term);
                fr_add(result, term, sum);
                store4(result, sum);
                u64 factor[LIMBS], updated[LIMBS];
                pml_from_u32(1u + a->c, factor);
                fr_mul(helper, factor, updated);
                store4(helper, updated);
            } else {
                unsigned int y_msb = pml_pop_msb(&b);
                pml_from_u32(a->c * (1u - y_msb), term);
                fr_mul(term, helper, scaled);
                fr_mul(scaled, pow, term);
                fr_add(result, term, sum);
                store4(result, sum);
                u64 factor[LIMBS], updated[LIMBS];
                pml_from_u32(1u + y_msb, factor);
                fr_mul(helper, factor, updated);
                store4(helper, updated);
            }

            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            unsigned long long xv = sfx_u64(x) & ~sfx_u64(y);
            unsigned int half = b.len / 2u;
            unsigned int shift = pml_leading_ones(sfx_u64(y), half) + bit_index - half;
            pfx_from_u64(pml_shl(xv, shift), term);
            fr_mul(term, helper, scaled);
            fr_add(result, scaled, sum);
            store4(out, sum);
            return 1;
        }
        case PFX_OVERFLOW_BITS_ZERO: {
            u64 result[LIMBS];
            pfx_load(a->checkpoints, PFX_OVERFLOW_BITS_ZERO, result);
            if (j >= PFX_LOG_K - PFX_XLEN) { store4(out, result); return 1; }
            sfx_bits b = a->b;
            u64 factor[LIMBS];
            if (a->has_r_x) {
                u64 nr[LIMBS], nc[LIMBS];
                pml_one_minus(a->r_x, nr);
                pml_one_minus(c_f, nc);
                fr_mul(nr, nc, factor);
            } else {
                u64 y_f[LIMBS], nc[LIMBS], ny[LIMBS];
                pml_from_u32(pml_pop_msb(&b), y_f);
                pml_one_minus(c_f, nc);
                pml_one_minus(y_f, ny);
                fr_mul(nc, ny, factor);
            }
            u64 scaled[LIMBS];
            fr_mul(result, factor, scaled);
            store4(result, scaled);
            u128 rest = b.bits;
            unsigned int zero = ((rest << suffix_len) >> PFX_XLEN) == 0 ? 1u : 0u;
            if (zero == 0u) { pfx_zero(out); return 1; }
            store4(out, result);
            return 1;
        }
        case PFX_XOR_ROTW7:
        case PFX_XOR_ROTW8:
        case PFX_XOR_ROTW12:
        case PFX_XOR_ROTW16: {
            if (j < PFX_XLEN) { pfx_zero(out); return 1; }
            unsigned int rotation;
            if (prefix == PFX_XOR_ROTW7) rotation = 7u;
            else if (prefix == PFX_XOR_ROTW8) rotation = 8u;
            else if (prefix == PFX_XOR_ROTW12) rotation = 12u;
            else rotation = 16u;

            u64 result[LIMBS];
            pfx_load(a->checkpoints, prefix, result);
            sfx_bits b = a->b;
            u64 bit[LIMBS];
            if (a->has_r_x) {
                pml_xor_pair(a->r_x, c_f, bit);
            } else {
                u64 y_f[LIMBS];
                pml_from_u32(pml_pop_msb(&b), y_f);
                pml_xor_pair(c_f, y_f, bit);
            }
            unsigned int position = (j - PFX_XLEN) / 2u;
            unsigned int rotated = (position + rotation) % 32u;
            rotated = 32u - 1u - rotated;
            u64 pow[LIMBS], scaled[LIMBS], sum[LIMBS];
            pfx_from_u64(1ULL << rotated, pow);
            fr_mul(pow, bit, scaled);
            fr_add(result, scaled, sum);
            store4(result, sum);

            sfx_bits x, y;
            sfx_uninterleave(b, &x, &y);
            unsigned int xor32 = (unsigned int)(sfx_u64(x) ^ sfx_u64(y));
            int half = (int)(suffix_len / 2u);
            unsigned int tail_shift;
            if (half - (int)rotation >= 0) {
                tail_shift = (unsigned int)(half - (int)rotation);
            } else {
                tail_shift = (unsigned int)(32 + (half - (int)rotation));
            }
            tail_shift &= 31u;
            unsigned int rotated_tail =
                tail_shift == 0u ? xor32 : ((xor32 << tail_shift) | (xor32 >> (32u - tail_shift)));
            u64 tail[LIMBS];
            pfx_from_u64((unsigned long long)rotated_tail, tail);
            fr_add(result, tail, sum);
            store4(out, sum);
            return 1;
        }
        default:
            return 0;
    }
}

extern "C" __global__ void pfx_mle_batch_kernel(const u64 *__restrict__ checkpoints,
                                                const unsigned long long *__restrict__ bits,
                                                const unsigned char *__restrict__ lens,
                                                unsigned int prefix,
                                                const u64 *__restrict__ r_x,
                                                unsigned int has_r_x,
                                                unsigned int c,
                                                unsigned int round,
                                                unsigned int suffix_len,
                                                u64 *__restrict__ out,
                                                unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    pml_args args;
    args.checkpoints = checkpoints;
    args.r_x = r_x;
    args.has_r_x = has_r_x;
    args.c = c;
    args.round = round;
    args.b.bits = ((u128)bits[2 * i + 1] << 64) | (u128)bits[2 * i];
    args.b.len = lens[i];
    args.b.bits &= sfx_mask(args.b.len);

    u64 value[LIMBS];
    if (!pml_eval(prefix, &args, value)) {
        pfx_eval(prefix, checkpoints, args.b, suffix_len, value);
    }
    store4(out + (unsigned long long)i * LIMBS, value);
}

__device__ __forceinline__ void pml_default(unsigned int prefix, u64 *out) {
    switch (prefix) {
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
            pfx_one(out);
            return;
        case PFX_CHANGE_DIVISOR: {
            u64 two[LIMBS], big[LIMBS];
            pfx_from_u64(2ULL, two);
            pfx_from_u128((u128)1 << PFX_XLEN, big);
            fr_sub(two, big, out);
            return;
        }
        default:
            pfx_zero(out);
            return;
    }
}

__device__ __forceinline__ void pml_eq_rxry(const u64 *r_x, const u64 *r_y, u64 *out) {
    pml_eq_pair(r_x, r_y, out);
}

__device__ void pml_update(unsigned int prefix,
                           const u64 *__restrict__ checkpoints,
                           const u64 *r_x,
                           const u64 *r_y,
                           unsigned int j,
                           unsigned int suffix_len,
                           u64 *out) {
    u64 current[LIMBS];
    pfx_load(checkpoints, prefix, current);
    u64 one[LIMBS];
    load4(FR_ONE, one);
    u64 nrx[LIMBS], nry[LIMBS];
    pml_one_minus(r_x, nrx);
    pml_one_minus(r_y, nry);

    switch (prefix) {
        case PFX_LOWER_WORD:
        case PFX_LOWER_HALF_WORD: {
            unsigned int floor = prefix == PFX_LOWER_WORD ? PFX_XLEN
                                                          : PFX_XLEN + PFX_XLEN / 2u;
            if (j < floor) { pml_default(prefix, out); return; }
            u64 term[LIMBS], scaled[LIMBS], sum[LIMBS];
            pfx_from_u128((u128)1 << (2u * PFX_XLEN - j), term);
            fr_mul(term, r_x, scaled);
            fr_add(current, scaled, sum);
            store4(current, sum);
            pfx_from_u128((u128)1 << (2u * PFX_XLEN - j - 1u), term);
            fr_mul(term, r_y, scaled);
            fr_add(current, scaled, sum);
            store4(out, sum);
            return;
        }
        case PFX_UPPER_WORD: {
            if (j >= PFX_XLEN) { store4(out, current); return; }
            u64 term[LIMBS], scaled[LIMBS], sum[LIMBS];
            pfx_from_u64(1ULL << (PFX_XLEN - j), term);
            fr_mul(term, r_x, scaled);
            fr_add(current, scaled, sum);
            store4(current, sum);
            pfx_from_u64(1ULL << (PFX_XLEN - j - 1u), term);
            fr_mul(term, r_y, scaled);
            fr_add(current, scaled, sum);
            store4(out, sum);
            return;
        }
        case PFX_EQ:
        case PFX_POS_REM_EQ_DIV:
        case PFX_NEG_DIV_EQ_REM: {
            if (prefix != PFX_EQ && j == 1u) {
                if (prefix == PFX_POS_REM_EQ_DIV) {
                    fr_mul(nrx, nry, out);
                } else {
                    fr_mul(r_x, r_y, out);
                }
                return;
            }
            u64 factor[LIMBS], product[LIMBS];
            pml_eq_rxry(r_x, r_y, factor);
            fr_mul(current, factor, product);
            store4(out, product);
            return;
        }
        case PFX_AND:
        case PFX_ANDN:
        case PFX_OR:
        case PFX_XOR: {
            unsigned int shift = PFX_XLEN - 1u - j / 2u;
            u64 bit[LIMBS];
            if (prefix == PFX_AND) {
                fr_mul(r_x, r_y, bit);
            } else if (prefix == PFX_ANDN) {
                fr_mul(r_x, nry, bit);
            } else if (prefix == PFX_OR) {
                u64 prod[LIMBS], sum[LIMBS];
                fr_mul(r_x, r_y, prod);
                fr_add(r_x, r_y, sum);
                fr_sub(sum, prod, bit);
            } else {
                pml_xor_pair(r_x, r_y, bit);
            }
            u64 pow[LIMBS], scaled[LIMBS], sum[LIMBS];
            pfx_from_u64(1ULL << shift, pow);
            fr_mul(pow, bit, scaled);
            fr_add(current, scaled, sum);
            store4(out, sum);
            return;
        }
        case PFX_LESS_THAN: {
            u64 eq[LIMBS], term[LIMBS], scaled[LIMBS], sum[LIMBS];
            pfx_load(checkpoints, PFX_EQ, eq);
            fr_mul(nrx, r_y, term);
            fr_mul(eq, term, scaled);
            fr_add(current, scaled, sum);
            store4(out, sum);
            return;
        }
        case PFX_LEFT_IS_ZERO: {
            u64 product[LIMBS];
            fr_mul(current, nrx, product);
            store4(out, product);
            return;
        }
        case PFX_RIGHT_IS_ZERO: {
            u64 product[LIMBS];
            fr_mul(current, nry, product);
            store4(out, product);
            return;
        }
        case PFX_LEFT_MSB: {
            if (j == 1u) { store4(out, r_x); } else { store4(out, current); }
            return;
        }
        case PFX_RIGHT_MSB: {
            if (j == 1u) { store4(out, r_y); } else { store4(out, current); }
            return;
        }
        case PFX_DIV_BY_ZERO: {
            u64 term[LIMBS], product[LIMBS];
            fr_mul(nrx, r_y, term);
            fr_mul(current, term, product);
            store4(out, product);
            return;
        }
        case PFX_POS_REM_LT_DIV:
        case PFX_NEG_DIV_GT_REM: {
            unsigned int negative = prefix == PFX_NEG_DIV_GT_REM;
            unsigned int eq_index = negative ? PFX_NEG_DIV_EQ_REM : PFX_POS_REM_EQ_DIV;
            if (j == 1u) {
                if (negative) { fr_mul(r_x, r_y, out); } else { fr_mul(nrx, nry, out); }
                return;
            }
            u64 factor[LIMBS];
            if (negative) { fr_mul(r_x, nry, factor); } else { fr_mul(nrx, r_y, factor); }
            if (j == 3u) {
                u64 product[LIMBS];
                fr_mul(current, factor, product);
                store4(out, product);
                return;
            }
            u64 eq[LIMBS], scaled[LIMBS], sum[LIMBS];
            pfx_load(checkpoints, eq_index, eq);
            fr_mul(eq, factor, scaled);
            fr_add(current, scaled, sum);
            store4(out, sum);
            return;
        }
        case PFX_NEG_DIV_ZERO_REM: {
            if (j == 1u) {
                fr_mul(nrx, r_y, out);
                return;
            }
            u64 product[LIMBS];
            fr_mul(current, nrx, product);
            store4(out, product);
            return;
        }
        case PFX_LSB: {
            if (j == 2u * PFX_XLEN - 1u) { store4(out, r_y); } else { pfx_one(out); }
            return;
        }
        case PFX_TWO_LSB: {
            if (j == 2u * PFX_XLEN - 1u) {
                fr_mul(nrx, nry, out);
            } else {
                store4(out, current);
            }
            return;
        }
        case PFX_POW2:
        case PFX_POW2W: {
            unsigned int bits_needed = prefix == PFX_POW2 ? 6u : 5u;
            if (suffix_len != 0u) { pfx_one(out); return; }
            if (j == 2u * PFX_XLEN - bits_needed) {
                unsigned long long shift =
                    prefix == PFX_POW2 ? (1ULL << (PFX_XLEN / 2u)) : (1ULL << 16);
                u64 term[LIMBS], scaled[LIMBS], sum[LIMBS];
                pfx_from_u64(shift - 1ULL, term);
                fr_mul(term, r_y, scaled);
                fr_add(one, scaled, sum);
                store4(out, sum);
                return;
            }
            if (2u * PFX_XLEN - j < bits_needed) {
                u64 term[LIMBS], scaled[LIMBS], sum[LIMBS], product[LIMBS];
                unsigned long long shift = 1ULL << (1ULL << (2u * PFX_XLEN - j));
                pfx_from_u64(shift - 1ULL, term);
                fr_mul(term, r_x, scaled);
                fr_add(one, scaled, sum);
                fr_mul(current, sum, product);
                store4(current, product);
                shift = 1ULL << (1ULL << (2u * PFX_XLEN - j - 1u));
                pfx_from_u64(shift - 1ULL, term);
                fr_mul(term, r_y, scaled);
                fr_add(one, scaled, sum);
                fr_mul(current, sum, product);
                store4(out, product);
                return;
            }
            pfx_one(out);
            return;
        }
        case PFX_REV8W: {
            u64 sum[LIMBS];
            unsigned int r_y_bit_index = 2u * PFX_XLEN - 1u - j;
            if (r_y_bit_index < 64u) {
                u64 term[LIMBS], scaled[LIMBS];
                pfx_from_u64(sfx_rev8w(1ULL << r_y_bit_index), term);
                fr_mul(r_y, term, scaled);
                fr_add(current, scaled, sum);
                store4(current, sum);
            }
            if (r_y_bit_index + 1u < 64u) {
                u64 term[LIMBS], scaled[LIMBS];
                pfx_from_u64(sfx_rev8w(1ULL << (r_y_bit_index + 1u)), term);
                fr_mul(r_x, term, scaled);
                fr_add(current, scaled, sum);
                store4(current, sum);
            }
            store4(out, current);
            return;
        }
        case PFX_RIGHT_SHIFT:
        case PFX_RIGHT_SHIFT_W: {
            if (prefix == PFX_RIGHT_SHIFT_W && j < PFX_XLEN) { pfx_zero(out); return; }
            u64 factor[LIMBS], scaled[LIMBS], addend[LIMBS], sum[LIMBS];
            fr_add(one, r_y, factor);
            fr_mul(current, factor, scaled);
            fr_mul(r_x, r_y, addend);
            fr_add(scaled, addend, sum);
            store4(out, sum);
            return;
        }
        case PFX_SIGN_EXTENSION: {
            if (j == 1u) { pml_default(PFX_SIGN_EXTENSION, out); return; }
            u64 term[LIMBS], scaled[LIMBS], sum[LIMBS];
            pfx_from_u64(1ULL << (j / 2u), term);
            fr_mul(term, nry, scaled);
            fr_add(current, scaled, sum);
            store4(current, sum);
            if (j == 2u * PFX_XLEN - 1u) {
                u64 msb[LIMBS], product[LIMBS];
                pfx_load(checkpoints, PFX_LEFT_MSB, msb);
                fr_mul(current, msb, product);
                store4(current, product);
            }
            store4(out, current);
            return;
        }
        case PFX_SIGN_EXT_UPPER_HALF: {
            unsigned int half = PFX_XLEN / 2u;
            if (j == PFX_XLEN + half + 1u) {
                u64 mask[LIMBS], product[LIMBS];
                pfx_from_u128((((u128)1 << half) - 1) << half, mask);
                fr_mul(mask, r_x, product);
                store4(out, product);
            } else {
                store4(out, current);
            }
            return;
        }
        case PFX_SIGN_EXT_RIGHT_OPERAND: {
            if (j == PFX_XLEN + 1u) {
                u64 mask[LIMBS], product[LIMBS];
                pfx_from_u128(((u128)1 << PFX_XLEN) - ((u128)1 << (PFX_XLEN / 2u)), mask);
                fr_mul(mask, r_y, product);
                store4(out, product);
            } else {
                store4(out, current);
            }
            return;
        }
        case PFX_LEFT_SHIFT:
        case PFX_LEFT_SHIFT_W: {
            unsigned int wide = prefix == PFX_LEFT_SHIFT_W;
            if (wide && j < PFX_XLEN) { pfx_zero(out); return; }
            unsigned int helper_index = wide ? PFX_LEFT_SHIFT_W_HELPER : PFX_LEFT_SHIFT_HELPER;
            u64 helper[LIMBS];
            pfx_load(checkpoints, helper_index, helper);
            unsigned int bit_index = PFX_XLEN - 1u - j / 2u;
            u64 term[LIMBS], scaled[LIMBS], pow[LIMBS], sum[LIMBS];
            fr_mul(r_x, nry, term);
            fr_mul(term, helper, scaled);
            pfx_from_u64(bit_index >= 64u ? 0ULL : (1ULL << bit_index), pow);
            fr_mul(scaled, pow, term);
            fr_add(current, term, sum);
            store4(out, sum);
            return;
        }
        case PFX_LEFT_SHIFT_HELPER:
        case PFX_LEFT_SHIFT_W_HELPER: {
            if (prefix == PFX_LEFT_SHIFT_W_HELPER && j < PFX_XLEN) { pfx_one(out); return; }
            u64 factor[LIMBS], product[LIMBS];
            fr_add(one, r_y, factor);
            fr_mul(current, factor, product);
            store4(out, product);
            return;
        }
        case PFX_CHANGE_DIVISOR: {
            u64 factor[LIMBS], product[LIMBS];
            if (j == 1u) { fr_mul(r_x, r_y, factor); } else { fr_mul(nrx, r_y, factor); }
            fr_mul(current, factor, product);
            store4(out, product);
            return;
        }
        case PFX_CHANGE_DIVISOR_W: {
            if (j < PFX_XLEN) { pfx_zero(out); return; }
            u64 factor[LIMBS], product[LIMBS];
            if (j == PFX_XLEN + 1u) {
                u64 two[LIMBS], big[LIMBS], base[LIMBS];
                pfx_from_u64(2ULL, two);
                pfx_from_u128((u128)1 << PFX_XLEN, big);
                fr_sub(two, big, base);
                fr_mul(r_x, r_y, factor);
                fr_mul(base, factor, product);
            } else {
                fr_mul(nrx, r_y, factor);
                fr_mul(current, factor, product);
            }
            store4(out, product);
            return;
        }
        case PFX_RIGHT_OPERAND:
        case PFX_RIGHT_OPERAND_W: {
            if (prefix == PFX_RIGHT_OPERAND_W && j <= PFX_XLEN) { store4(out, current); return; }
            unsigned int shift = PFX_XLEN - 1u - j / 2u;
            u64 pow[LIMBS], scaled[LIMBS], sum[LIMBS];
            pfx_from_u64(1ULL << shift, pow);
            fr_mul(pow, r_y, scaled);
            fr_add(current, scaled, sum);
            store4(out, sum);
            return;
        }
        case PFX_OVERFLOW_BITS_ZERO: {
            if (j >= PFX_LOG_K - PFX_XLEN) { store4(out, current); return; }
            u64 term[LIMBS], product[LIMBS];
            fr_mul(nrx, nry, term);
            fr_mul(current, term, product);
            store4(out, product);
            return;
        }
        case PFX_XOR_ROT16:
        case PFX_XOR_ROT24:
        case PFX_XOR_ROT32:
        case PFX_XOR_ROT63:
        case PFX_XOR_ROTW7:
        case PFX_XOR_ROTW8:
        case PFX_XOR_ROTW12:
        case PFX_XOR_ROTW16: {
            unsigned int wide = prefix >= PFX_XOR_ROTW7;
            if (wide && j < PFX_XLEN) { pfx_zero(out); return; }
            unsigned int rotation;
            switch (prefix) {
                case PFX_XOR_ROT16: rotation = 16u; break;
                case PFX_XOR_ROT24: rotation = 24u; break;
                case PFX_XOR_ROT32: rotation = 32u; break;
                case PFX_XOR_ROT63: rotation = 63u; break;
                case PFX_XOR_ROTW7: rotation = 7u; break;
                case PFX_XOR_ROTW8: rotation = 8u; break;
                case PFX_XOR_ROTW12: rotation = 12u; break;
                default: rotation = 16u; break;
            }
            unsigned int width = wide ? 32u : PFX_XLEN;
            unsigned int position = wide ? (j - PFX_XLEN) / 2u : j / 2u;
            unsigned int shift = width - 1u - (position + rotation) % width;
            u64 bit[LIMBS], pow[LIMBS], scaled[LIMBS], sum[LIMBS];
            pml_xor_pair(r_x, r_y, bit);
            pfx_from_u64(1ULL << shift, pow);
            fr_mul(pow, bit, scaled);
            fr_add(current, scaled, sum);
            store4(out, sum);
            return;
        }
        default:
            store4(out, current);
            return;
    }
}

extern "C" __global__ void pfx_update_checkpoints_kernel(const u64 *__restrict__ checkpoints,
                                                        const u64 *__restrict__ r_x,
                                                        const u64 *__restrict__ r_y,
                                                        unsigned int round,
                                                        unsigned int suffix_len,
                                                        u64 *__restrict__ out,
                                                        unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 value[LIMBS];
    pml_update(i, checkpoints, r_x, r_y, round, suffix_len, value);
    store4(out + (unsigned long long)i * LIMBS, value);
}

extern "C" __global__ void pfx_mle_round_kernel(const u64 *__restrict__ checkpoints,
                                                const u64 *__restrict__ r_x,
                                                unsigned int has_r_x,
                                                unsigned int round,
                                                unsigned int b_len,
                                                unsigned int half,
                                                u64 *__restrict__ out) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= half) return;
    unsigned int prefix = blockIdx.y;
    unsigned int point = blockIdx.z;

    pml_args args;
    args.checkpoints = checkpoints;
    args.r_x = r_x;
    args.has_r_x = has_r_x;
    args.c = point == 0u ? 0u : 2u;
    args.round = round;
    args.b = sfx_new((u128)b, b_len);

    u64 value[LIMBS];
    if (!pml_eval(prefix, &args, value)) {
        pfx_eval(prefix, checkpoints, args.b, PFX_LOG_K - round - b_len - 1u, value);
    }
    unsigned long long slot =
        ((unsigned long long)point * PFX_COUNT + prefix) * half + b;
    store4(out + slot * LIMBS, value);
}
