struct NativeCore { ulong rs1, pc, rs2, rd, imm_lo, imm_hi, lookup; uint flags, reserved; };
struct NativeLookup { ulong left, right_lo, right_hi; };
struct InputTerms { uint left, right; uint terms[5][2]; };
template <typename F> F native_operand(NativeCore row, uint column) {
    switch (column) {
        case 0: return F::from_u64(row.rs1);
        case 1: return F::from_u64(row.pc);
        case 2: return F::from_u64(row.rs2);
        case 3: return jolt::from_i128<F>(ulong2(row.imm_lo, row.imm_hi));
        default: return F::from_u64(row.rd);
    }
}
template <typename F>
void native_inputs(NativeCore row, constant InputTerms& terms, thread F& left, thread F& right) {
    left = F::zero(); right = F::zero();
    for (uint i = 0; i < terms.left + terms.right; i++) {
        if (row.flags & (1u << terms.terms[i][0])) {
            F value = native_operand<F>(row, terms.terms[i][1]);
            if (i < terms.left) left = left + value; else right = right + value;
        }
    }
}
template <typename F> F native_lookup_right(NativeLookup row) {
    return F::from_u128(ulong2(row.right_lo, row.right_hi));
}
