#ifndef JOLT_METAL_COMMON_H
#define JOLT_METAL_COMMON_H

#include <metal_stdlib>
using namespace metal;

// Add with carry: returns (sum, carry_out).
inline uint2 adc(uint a, uint b, uint carry_in) {
    ulong s = ulong(a) + ulong(b) + ulong(carry_in);
    return uint2(uint(s), uint(s >> 32));
}

// Subtract with borrow: returns (diff, borrow_out).
// borrow_out is 1 if underflow occurred.
inline uint2 sbb(uint a, uint b, uint borrow_in) {
    ulong s = ulong(a) - ulong(b) - ulong(borrow_in);
    return uint2(uint(s), uint(s >> 63));
}

#endif // JOLT_METAL_COMMON_H
