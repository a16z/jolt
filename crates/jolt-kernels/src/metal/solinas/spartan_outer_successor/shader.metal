// Append after outer_remainder/shader.metal.

struct SpartanSigned320 {
    uint limb[10];
};

inline SpartanSigned320 spartan_s320_zero() {
    SpartanSigned320 value;
    for (uint i = 0u; i < 10u; i++) {
        value.limb[i] = 0u;
    }
    return value;
}

inline SpartanSigned320 spartan_s320_negate(SpartanSigned320 value) {
    ulong carry = 1ul;
    for (uint i = 0u; i < 10u; i++) {
        ulong word = (ulong)(~value.limb[i]) + carry;
        value.limb[i] = (uint)word;
        carry = word >> 32;
    }
    return value;
}

inline void spartan_s320_add(
    thread SpartanSigned320& accumulator,
    SpartanSigned320 value)
{
    ulong carry = 0ul;
    for (uint i = 0u; i < 10u; i++) {
        ulong word = (ulong)accumulator.limb[i] + (ulong)value.limb[i] + carry;
        accumulator.limb[i] = (uint)word;
        carry = word >> 32;
    }
}

inline void spartan_s320_fmadd(
    thread SpartanSigned320& accumulator,
    SolinasFp128 weight,
    SpartanSigned192 value)
{
    bool negative = (value.limb[5] & 0x80000000u) != 0u;
    if (negative) {
        value = spartan_s192_negate(value);
    }

    SpartanSigned320 product = spartan_s320_zero();
    for (uint i = 0u; i < 5u; i++) {
        ulong carry = 0ul;
        for (uint j = 0u; j < 4u; j++) {
            uint k = i + j;
            ulong word = (ulong)value.limb[i] * (ulong)weight.limb[j]
                + (ulong)product.limb[k]
                + carry;
            product.limb[k] = (uint)word;
            carry = word >> 32;
        }
        product.limb[i + 4u] = (uint)carry;
    }
    spartan_s320_add(
        accumulator,
        negative ? spartan_s320_negate(product) : product);
}

inline void spartan_s320_add_carry(
    thread SolinasWide256& value,
    uint index,
    ulong carry)
{
    for (uint i = index; i < 8u && carry != 0ul; i++) {
        ulong word = (ulong)value.limb[i] + carry;
        value.limb[i] = (uint)word;
        carry = word >> 32;
    }
}

inline SolinasFp128 spartan_s320_reduce(SpartanSigned320 value) {
    bool negative = (value.limb[9] & 0x80000000u) != 0u;
    if (negative) {
        value = spartan_s320_negate(value);
    }

    SolinasWide256 folded;
    for (uint i = 0u; i < 8u; i++) {
        folded.limb[i] = i < 4u ? value.limb[i] : 0u;
    }

    ulong carry = 0ul;
    for (uint i = 0u; i < 4u; i++) {
        ulong word = (ulong)value.limb[i + 4u] * (ulong)SOLINAS_OFFSET
            + (ulong)folded.limb[i]
            + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32;
    }
    spartan_s320_add_carry(folded, 4u, carry);

    ulong offset_squared = (ulong)SOLINAS_OFFSET * (ulong)SOLINAS_OFFSET;
    uint factor[2] = {
        (uint)offset_squared,
        (uint)(offset_squared >> 32),
    };
    for (uint i = 0u; i < 2u; i++) {
        carry = 0ul;
        for (uint j = 0u; j < 2u; j++) {
            uint k = i + j;
            ulong word = (ulong)value.limb[i + 8u] * (ulong)factor[j]
                + (ulong)folded.limb[k]
                + carry;
            folded.limb[k] = (uint)word;
            carry = word >> 32;
        }
        spartan_s320_add_carry(folded, i + 2u, carry);
    }

    SolinasFp128 reduced = solinas_reduce(folded);
    return negative ? solinas_sub(solinas_zero(), reduced) : reduced;
}

inline SolinasFp128 outer_fold_b_deferred(
    device const InstructionInputRow& compact,
    device const SpartanOuterUniskipResidualRow& residual,
    device const SolinasFp128* lagrange,
    bool second_stream)
{
    uint count = second_stream ? 9u : 10u;
    SpartanSigned320 sum = spartan_s320_zero();
    for (uint row = 0u; row < count; row++) {
        spartan_s320_fmadd(
            sum,
            lagrange[row],
            outer_b_row(compact, residual, row, second_stream));
    }
    return spartan_s320_reduce(sum);
}

kernel void solinas_outer_remainder_deferred_b_materialize(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterUniskipResidualRow* residual_rows [[buffer(1)]],
    device const SolinasFp128* lagrange [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* b_state [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant OuterRemainderPhaseParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    bool accumulate = false;
    for (uint x_out = block;
         x_out < params.e_out_length;
         x_out += params.blocks) {
        SolinasFp128 q_zero = solinas_zero();
        SolinasFp128 q_infinity = solinas_zero();
        for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
            uint cycle = x_out * params.e_in_length + x_in;
            SolinasFp128 az_0 = outer_fold_a(compact_rows[cycle], lagrange, false);
            SolinasFp128 az_1 = outer_fold_a(compact_rows[cycle], lagrange, true);
            SolinasFp128 bz_0 = outer_fold_b_deferred(
                compact_rows[cycle], residual_rows[cycle], lagrange, false);
            SolinasFp128 bz_1 = outer_fold_b_deferred(
                compact_rows[cycle], residual_rows[cycle], lagrange, true);
            b_state[2u * cycle] = bz_0;
            b_state[2u * cycle + 1u] = bz_1;
            SolinasFp128 weight = e_in[x_in];
            q_zero = solinas_add(
                q_zero,
                solinas_mul_wide(weight, solinas_mul_wide(az_0, bz_0)));
            q_infinity = solinas_add(
                q_infinity,
                solinas_mul_wide(
                    weight,
                    solinas_mul_wide(
                        solinas_sub(az_1, az_0),
                        solinas_sub(bz_1, bz_0))));
        }
        outer_finish_two_columns(
            q_zero,
            q_infinity,
            e_out[x_out],
            partials,
            block,
            shared,
            tid,
            lane,
            simdgroup,
            threads,
            accumulate);
        accumulate = true;
    }
}

inline SolinasFp128 outer_fold_a_collapsed_probe(
    device const InstructionInputRow& compact,
    constant const SolinasFp128* lookup)
{
    ulong flags = instruction_input_row_word(compact, 5u);
    uint low = (uint)(flags & 31ul);
    uint middle = (uint)((flags >> 5) & 31ul);
    uint high = (uint)((flags >> 10) & 31ul);
    return solinas_add(
        solinas_add(lookup[low], lookup[32u + middle]),
        lookup[64u + high]);
}

kernel void solinas_outer_remainder_collapsed_a_stream_bind_probe(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SolinasFp128* b_source [[buffer(1)]],
    device SolinasFp128* destination [[buffer(2)]],
    constant const SolinasFp128* a_lookup [[buffer(3)]],
    device const SolinasFp128* e_in [[buffer(4)]],
    device const SolinasFp128* e_out [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant SolinasFp128& challenge [[buffer(7)]],
    constant OuterRemainderPhaseParams& params [[buffer(8)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    bool accumulate = false;
    for (uint x_out = block;
         x_out < params.e_out_length;
         x_out += params.blocks) {
        SolinasFp128 q_zero = solinas_zero();
        SolinasFp128 q_infinity = solinas_zero();
        for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
            uint pair = x_out * params.e_in_length + x_in;
            uint cycle_0 = 2u * pair;
            uint cycle_1 = cycle_0 + 1u;
            SolinasFp128 az_0 = outer_fold_a_collapsed_probe(
                compact_rows[cycle_0], a_lookup);
            SolinasFp128 az_1 = outer_fold_a_collapsed_probe(
                compact_rows[cycle_1], a_lookup);
            SolinasFp128 bz_0 = outer_bind(
                b_source[2u * cycle_0],
                b_source[2u * cycle_0 + 1u],
                challenge);
            SolinasFp128 bz_1 = outer_bind(
                b_source[2u * cycle_1],
                b_source[2u * cycle_1 + 1u],
                challenge);
            destination[2u * cycle_0] = az_0;
            destination[2u * cycle_0 + 1u] = bz_0;
            destination[2u * cycle_1] = az_1;
            destination[2u * cycle_1 + 1u] = bz_1;

            SolinasFp128 weight = e_in[x_in];
            q_zero = solinas_add(
                q_zero,
                solinas_mul_wide(weight, solinas_mul_wide(az_0, bz_0)));
            q_infinity = solinas_add(
                q_infinity,
                solinas_mul_wide(
                    weight,
                    solinas_mul_wide(
                        solinas_sub(az_1, az_0),
                        solinas_sub(bz_1, bz_0))));
        }
        outer_finish_two_columns(
            q_zero,
            q_infinity,
            e_out[x_out],
            partials,
            block,
            shared,
            tid,
            lane,
            simdgroup,
            threads,
            accumulate);
        accumulate = true;
    }
}
