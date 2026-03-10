#ifndef JOLT_METAL_COMMON_H
#define JOLT_METAL_COMMON_H

#include <metal_stdlib>
using namespace metal;

// Add with carry: returns (sum, carry_out). Pure 32-bit — no ulong emulation.
inline uint2 adc(uint a, uint b, uint carry_in) {
    uint s1 = a + b;
    uint c1 = uint(s1 < a);
    uint s2 = s1 + carry_in;
    uint c2 = uint(s2 < s1);
    return uint2(s2, c1 + c2);
}

// Subtract with borrow: returns (diff, borrow_out). Pure 32-bit.
// borrow_out is 1 if underflow occurred.
inline uint2 sbb(uint a, uint b, uint borrow_in) {
    uint d1 = a - b;
    uint b1 = uint(a < b);
    uint d2 = d1 - borrow_in;
    uint b2 = uint(d1 < borrow_in);
    return uint2(d2, b1 + b2);
}

#endif // JOLT_METAL_COMMON_H
