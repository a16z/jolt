#include <metal_stdlib>
using namespace metal;

#ifndef SOLINAS_OFFSET
#define SOLINAS_OFFSET 275u
#endif

struct SolinasFp128 {
    uint4 limb;
};

struct SolinasWide256 {
    uint limb[8];
};

struct SolinasCorrection {
    SolinasFp128 value;
    uint carry;
};

inline SolinasFp128 solinas_zero() {
    SolinasFp128 result;
    for (uint i = 0; i < 4; i++) {
        result.limb[i] = 0;
    }
    return result;
}

inline SolinasFp128 solinas_select(bool take_lhs, SolinasFp128 lhs, SolinasFp128 rhs) {
    SolinasFp128 result;
    uint mask = take_lhs ? 0xffffffffu : 0u;
    for (uint i = 0; i < 4; i++) {
        result.limb[i] = (lhs.limb[i] & mask) | (rhs.limb[i] & ~mask);
    }
    return result;
}

inline SolinasCorrection solinas_add_offset(SolinasFp128 value) {
    SolinasCorrection result;
    ulong sum = (ulong)value.limb[0] + (ulong)SOLINAS_OFFSET;
    result.value.limb[0] = (uint)sum;
    ulong carry = sum >> 32;
    for (uint i = 1; i < 4; i++) {
        sum = (ulong)value.limb[i] + carry;
        result.value.limb[i] = (uint)sum;
        carry = sum >> 32;
    }
    result.carry = (uint)carry;
    return result;
}

inline SolinasFp128 solinas_sub_offset(SolinasFp128 value) {
    SolinasFp128 result;
    ulong subtrahend = (ulong)SOLINAS_OFFSET;
    for (uint i = 0; i < 4; i++) {
        ulong word = (ulong)value.limb[i];
        result.limb[i] = (uint)(word - subtrahend);
        subtrahend = word < subtrahend ? 1u : 0u;
    }
    return result;
}

inline SolinasFp128 solinas_add(SolinasFp128 lhs, SolinasFp128 rhs) {
    SolinasFp128 sum;
    ulong carry = 0;
    for (uint i = 0; i < 4; i++) {
        ulong word = (ulong)lhs.limb[i] + (ulong)rhs.limb[i] + carry;
        sum.limb[i] = (uint)word;
        carry = word >> 32;
    }

    SolinasCorrection corrected = solinas_add_offset(sum);
    return solinas_select(carry != 0 || corrected.carry != 0, corrected.value, sum);
}

inline SolinasFp128 solinas_sub(SolinasFp128 lhs, SolinasFp128 rhs) {
    SolinasFp128 difference;
    ulong borrow = 0;
    for (uint i = 0; i < 4; i++) {
        ulong subtrahend = (ulong)rhs.limb[i] + borrow;
        ulong word = (ulong)lhs.limb[i];
        difference.limb[i] = (uint)(word - subtrahend);
        borrow = word < subtrahend ? 1u : 0u;
    }

    SolinasFp128 corrected = solinas_sub_offset(difference);
    return solinas_select(borrow != 0, corrected, difference);
}

inline SolinasWide256 solinas_product_wide(SolinasFp128 lhs, SolinasFp128 rhs) {
    SolinasWide256 product;
    for (uint i = 0; i < 8; i++) {
        product.limb[i] = 0;
    }
    for (uint i = 0; i < 4; i++) {
        ulong carry = 0;
        for (uint j = 0; j < 4; j++) {
            uint k = i + j;
            ulong word = (ulong)lhs.limb[i] * (ulong)rhs.limb[j]
                + (ulong)product.limb[k]
                + carry;
            product.limb[k] = (uint)word;
            carry = word >> 32;
        }
        product.limb[i + 4] = (uint)carry;
    }
    return product;
}

inline SolinasFp128 solinas_reduce(SolinasWide256 product) {
    SolinasFp128 folded;
    ulong carry = 0;
    for (uint i = 0; i < 4; i++) {
        ulong word = (ulong)product.limb[i + 4] * (ulong)SOLINAS_OFFSET
            + (ulong)product.limb[i]
            + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32;
    }

    ulong word = (ulong)folded.limb[0] + carry * (ulong)SOLINAS_OFFSET;
    folded.limb[0] = (uint)word;
    carry = word >> 32;
    for (uint i = 1; i < 4; i++) {
        word = (ulong)folded.limb[i] + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32;
    }

    SolinasCorrection corrected = solinas_add_offset(folded);
    return solinas_select(carry != 0 || corrected.carry != 0, corrected.value, folded);
}

inline SolinasFp128 solinas_mul_wide(SolinasFp128 lhs, SolinasFp128 rhs) {
    return solinas_reduce(solinas_product_wide(lhs, rhs));
}
