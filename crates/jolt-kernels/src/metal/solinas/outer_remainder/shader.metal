// Candidate source. Concatenate after fp128.metal, simd_reduce.metal, and
// spartan_outer_common.metal.

#define OUTER_REMAINDER_CANONICAL_OPENINGS 35u
#define OUTER_REMAINDER_TILE_ROWS 64u
#define OUTER_REMAINDER_SIMD_WIDTH 32u
#define OUTER_REMAINDER_MAX_COLUMNS_PER_SIMDGROUP 9u
#define OUTER_REMAINDER_REGISTERS_COMPONENTS 3u
#define OUTER_REMAINDER_FIRST_B_OFFSET 202u
#define OUTER_REMAINDER_SECOND_B_OFFSET 215u

constant uint OUTER_REMAINDER_BOOLEAN_FLAG_BITS[OUTER_REMAINDER_CANONICAL_OPENINGS] = {
    0u, 0u, 0u, 6u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 11u, 12u, 0u, 8u, 2u, 3u, 4u,
    0u, 1u, 5u, 14u, 9u, 7u, 15u, 13u, 16u, 24u, 10u
};

struct OuterRemainderPhaseParams {
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint blocks;
};

struct OuterRemainderOpeningParams {
    uint columns;
    uint e_in_length;
    uint e_out_length;
    uint blocks;
    uint source_elements;
    uint reserved_0;
    uint reserved_1;
    uint reserved_2;
};

struct OuterRemainderReduceParams {
    uint input_count;
    uint columns;
    uint reserved_0;
    uint reserved_1;
};

inline int outer_flag(ulong flags, uint bit) {
    return (int)((flags >> bit) & 1ul);
}

inline SolinasFp128 outer_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32);
    return result;
}

inline SolinasFp128 outer_from_signed_u128(
    ulong low,
    ulong high,
    bool positive)
{
    SpartanSigned192 value = spartan_scaled_u128(low, high, 1u);
    if (!positive) {
        value = spartan_s192_negate(value);
    }
    return spartan_small_times_s192(1, value);
}

inline SolinasFp128 outer_bind(
    SolinasFp128 low,
    SolinasFp128 high,
    SolinasFp128 challenge)
{
    return solinas_add(
        low,
        solinas_mul_wide(challenge, solinas_sub(high, low)));
}

inline void outer_row_memory(
    device const InstructionInputRow& compact,
    device const SpartanOuterSuccessorRow& successor,
    device const SpartanOuterColdRow& cold,
    thread ulong& ram_address,
    thread ulong& rs2,
    thread ulong& rd_write,
    thread ulong& ram_read,
    thread ulong& ram_write)
{
    ulong flags = instruction_input_row_word(compact, 5u);
    bool load = outer_flag(flags, 0u) != 0;
    bool store = outer_flag(flags, 1u) != 0;
    ulong memory_0 = spartan_outer_residual_word(successor, cold, 6u);
    ulong memory_1 = spartan_outer_residual_word(successor, cold, 7u);
    rs2 = instruction_input_row_word(compact, 2u);
    ram_address = load || store ? memory_0 : 0ul;
    rd_write = store ? 0ul : (load ? memory_1 : memory_0);
    ram_read = load || store ? memory_1 : 0ul;
    ram_write = load ? memory_1 : (store ? rs2 : 0ul);
}

inline SolinasFp128 outer_fold_a_lookup(
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

struct OuterDeferredSigned320 {
    uint limb[10];
};

inline OuterDeferredSigned320 outer_deferred_s320_zero() {
    OuterDeferredSigned320 value;
    for (uint i = 0u; i < 10u; i++) {
        value.limb[i] = 0u;
    }
    return value;
}

inline OuterDeferredSigned320 outer_deferred_s320_negate(
    OuterDeferredSigned320 value)
{
    ulong carry = 1ul;
    for (uint i = 0u; i < 10u; i++) {
        ulong word = (ulong)(~value.limb[i]) + carry;
        value.limb[i] = (uint)word;
        carry = word >> 32;
    }
    return value;
}

inline void outer_deferred_s320_add(
    thread OuterDeferredSigned320& accumulator,
    OuterDeferredSigned320 value)
{
    ulong carry = 0ul;
    for (uint i = 0u; i < 10u; i++) {
        ulong word = (ulong)accumulator.limb[i] + (ulong)value.limb[i] + carry;
        accumulator.limb[i] = (uint)word;
        carry = word >> 32;
    }
}

inline void outer_deferred_s320_fmadd_u64(
    thread OuterDeferredSigned320& accumulator,
    SolinasFp128 weight,
    ulong value)
{
    if (value == 0ul) {
        return;
    }
    uint source[2] = {
        (uint)value,
        (uint)(value >> 32),
    };
    OuterDeferredSigned320 product = outer_deferred_s320_zero();
    for (uint i = 0u; i < 2u; i++) {
        ulong carry = 0ul;
        for (uint j = 0u; j < 4u; j++) {
            uint k = i + j;
            ulong word = (ulong)source[i] * (ulong)weight.limb[j]
                + (ulong)product.limb[k]
                + carry;
            product.limb[k] = (uint)word;
            carry = word >> 32;
        }
        product.limb[i + 4u] = (uint)carry;
    }
    outer_deferred_s320_add(accumulator, product);
}

inline void outer_deferred_s320_fmadd_signed_u128(
    thread OuterDeferredSigned320& accumulator,
    SolinasFp128 weight,
    ulong low,
    ulong high,
    bool positive)
{
    if (low == 0ul && high == 0ul) {
        return;
    }
    uint source[4] = {
        (uint)low,
        (uint)(low >> 32),
        (uint)high,
        (uint)(high >> 32),
    };
    OuterDeferredSigned320 product = outer_deferred_s320_zero();
    for (uint i = 0u; i < 4u; i++) {
        ulong carry = 0ul;
        for (uint j = 0u; j < 4u; j++) {
            uint k = i + j;
            ulong word = (ulong)source[i] * (ulong)weight.limb[j]
                + (ulong)product.limb[k]
                + carry;
            product.limb[k] = (uint)word;
            carry = word >> 32;
        }
        product.limb[i + 4u] = (uint)carry;
    }
    outer_deferred_s320_add(
        accumulator,
        positive ? product : outer_deferred_s320_negate(product));
}

inline void outer_deferred_s320_add_field(
    thread OuterDeferredSigned320& accumulator,
    SolinasFp128 value)
{
    OuterDeferredSigned320 extended = outer_deferred_s320_zero();
    for (uint i = 0u; i < 4u; i++) {
        extended.limb[i] = value.limb[i];
    }
    outer_deferred_s320_add(accumulator, extended);
}

inline void outer_deferred_s320_add_carry(
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

inline SolinasFp128 outer_deferred_s320_reduce(OuterDeferredSigned320 value) {
    bool negative = (value.limb[9] & 0x80000000u) != 0u;
    if (negative) {
        value = outer_deferred_s320_negate(value);
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
    outer_deferred_s320_add_carry(folded, 4u, carry);

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
        outer_deferred_s320_add_carry(folded, i + 2u, carry);
    }

    SolinasFp128 reduced = solinas_reduce(folded);
    return negative ? solinas_sub(solinas_zero(), reduced) : reduced;
}

inline SolinasFp128 outer_fold_b_first(
    device const SpartanOuterSuccessorRow& successor,
    device const SpartanOuterColdRow& cold,
    ulong flags,
    ulong ram_address,
    ulong rs2,
    ulong rd_write,
    ulong ram_read,
    ulong ram_write,
    constant const SolinasFp128* coefficients)
{
    OuterDeferredSigned320 sum = outer_deferred_s320_zero();
    outer_deferred_s320_fmadd_u64(sum, coefficients[0], ram_address);
    outer_deferred_s320_fmadd_u64(sum, coefficients[1], ram_read);
    outer_deferred_s320_fmadd_u64(sum, coefficients[2], ram_write);
    outer_deferred_s320_fmadd_u64(sum, coefficients[3], rd_write);
    outer_deferred_s320_fmadd_u64(sum, coefficients[4], rs2);
    outer_deferred_s320_fmadd_u64(
        sum, coefficients[5], spartan_outer_residual_word(successor, cold, 8u));
    outer_deferred_s320_fmadd_u64(
        sum, coefficients[6], spartan_outer_residual_word(successor, cold, 0u));
    outer_deferred_s320_fmadd_u64(
        sum, coefficients[7], spartan_outer_residual_word(successor, cold, 13u));
    outer_deferred_s320_fmadd_u64(
        sum, coefficients[8], spartan_outer_residual_word(successor, cold, 11u));
    outer_deferred_s320_fmadd_u64(
        sum, coefficients[9], spartan_outer_residual_word(successor, cold, 12u));
    outer_deferred_s320_fmadd_u64(
        sum, coefficients[10], spartan_outer_residual_word(successor, cold, 5u));
    outer_deferred_s320_add_field(sum, coefficients[11]);
    if (outer_flag(flags, 15u) != 0) {
        outer_deferred_s320_add_field(sum, coefficients[12]);
    }
    return outer_deferred_s320_reduce(sum);
}

inline SolinasFp128 outer_fold_b_second(
    device const InstructionInputRow& compact,
    device const SpartanOuterSuccessorRow& successor,
    device const SpartanOuterColdRow& cold,
    ulong flags,
    ulong ram_address,
    ulong rd_write,
    constant const SolinasFp128* coefficients)
{
    OuterDeferredSigned320 sum = outer_deferred_s320_zero();
    outer_deferred_s320_fmadd_u64(sum, coefficients[0], ram_address);
    outer_deferred_s320_fmadd_u64(
        sum, coefficients[1], instruction_input_row_word(compact, 0u));
    outer_deferred_s320_fmadd_signed_u128(
        sum,
        coefficients[2],
        instruction_input_row_word(compact, 3u),
        instruction_input_row_word(compact, 4u),
        outer_flag(flags, 18u) != 0);
    outer_deferred_s320_fmadd_signed_u128(
        sum,
        coefficients[3],
        spartan_outer_residual_word(successor, cold, 9u),
        spartan_outer_residual_word(successor, cold, 10u),
        true);
    outer_deferred_s320_fmadd_u64(
        sum, coefficients[4], spartan_outer_residual_word(successor, cold, 0u));
    outer_deferred_s320_fmadd_signed_u128(
        sum,
        coefficients[5],
        spartan_outer_residual_word(successor, cold, 1u),
        spartan_outer_residual_word(successor, cold, 2u),
        outer_flag(flags, 17u) != 0);
    outer_deferred_s320_fmadd_signed_u128(
        sum,
        coefficients[6],
        spartan_outer_residual_word(successor, cold, 3u),
        spartan_outer_residual_word(successor, cold, 4u),
        outer_flag(flags, 19u) != 0);
    outer_deferred_s320_fmadd_u64(sum, coefficients[7], rd_write);
    outer_deferred_s320_fmadd_u64(
        sum, coefficients[8], spartan_outer_residual_word(successor, cold, 13u));
    outer_deferred_s320_fmadd_u64(
        sum, coefficients[9], instruction_input_row_word(compact, 1u));
    outer_deferred_s320_fmadd_u64(
        sum, coefficients[10], spartan_outer_residual_word(successor, cold, 11u));
    outer_deferred_s320_add_field(sum, coefficients[11]);
    outer_deferred_s320_add_field(sum, coefficients[12]);
    if (outer_flag(flags, 16u) != 0) {
        outer_deferred_s320_add_field(sum, coefficients[13]);
    }
    if (outer_flag(flags, 15u) != 0) {
        outer_deferred_s320_add_field(sum, coefficients[14]);
    }
    return outer_deferred_s320_reduce(sum);
}

inline void outer_finish_two_columns(
    SolinasFp128 q_zero,
    SolinasFp128 q_infinity,
    SolinasFp128 outer_weight,
    device SolinasFp128* partials,
    uint block,
    threadgroup SolinasFp128* shared,
    uint tid,
    uint lane,
    uint simdgroup,
    uint threads,
    bool accumulate)
{
    q_zero = solinas_simd_sum_32(q_zero);
    q_infinity = solinas_simd_sum_32(q_infinity);
    if (lane == 0u) {
        shared[2u * simdgroup] = q_zero;
        shared[2u * simdgroup + 1u] = q_infinity;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u && lane < 2u) {
        uint simdgroups = threads / OUTER_REMAINDER_SIMD_WIDTH;
        SolinasFp128 sum = solinas_zero();
        for (uint group = 0; group < simdgroups; group++) {
            sum = solinas_add(sum, shared[2u * group + lane]);
        }
        SolinasFp128 weighted = solinas_mul_wide(outer_weight, sum);
        partials[2u * block + lane] = accumulate
            ? solinas_add(partials[2u * block + lane], weighted)
            : weighted;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void solinas_outer_remainder_materialize_b_and_message(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterSuccessorRow* successor_rows [[buffer(1)]],
    device const SpartanOuterColdRow* cold_rows [[buffer(2)]],
    constant const SolinasFp128* a_lookup [[buffer(3)]],
    device const SolinasFp128* e_in [[buffer(4)]],
    device const SolinasFp128* e_out [[buffer(5)]],
    device SolinasFp128* b_state [[buffer(6)]],
    device SolinasFp128* partials [[buffer(7)]],
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
            uint cycle = x_out * params.e_in_length + x_in;
            SolinasFp128 az_0 = outer_fold_a_lookup(
                compact_rows[cycle], a_lookup + 10u);
            SolinasFp128 az_1 = outer_fold_a_lookup(
                compact_rows[cycle], a_lookup + 106u);
            ulong flags = instruction_input_row_word(compact_rows[cycle], 5u);
            ulong ram_address;
            ulong rs2;
            ulong rd_write;
            ulong ram_read;
            ulong ram_write;
            outer_row_memory(
                compact_rows[cycle],
                successor_rows[cycle],
                cold_rows[cycle],
                ram_address,
                rs2,
                rd_write,
                ram_read,
                ram_write);
            SolinasFp128 bz_0 = outer_fold_b_first(
                successor_rows[cycle],
                cold_rows[cycle],
                flags,
                ram_address,
                rs2,
                rd_write,
                ram_read,
                ram_write,
                a_lookup + OUTER_REMAINDER_FIRST_B_OFFSET);
            SolinasFp128 bz_1 = outer_fold_b_second(
                compact_rows[cycle],
                successor_rows[cycle],
                cold_rows[cycle],
                flags,
                ram_address,
                rd_write,
                a_lookup + OUTER_REMAINDER_SECOND_B_OFFSET);
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

inline SolinasFp128 outer_fold_a_collapsed(
    device const InstructionInputRow& compact,
    constant const SolinasFp128* lookup)
{
    return outer_fold_a_lookup(compact, lookup);
}

kernel void solinas_outer_remainder_collapsed_a_stream_bind(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device SolinasFp128* state [[buffer(1)]],
    constant const SolinasFp128* a_lookup [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant SolinasFp128& challenge [[buffer(6)]],
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
            uint pair = x_out * params.e_in_length + x_in;
            uint cycle_0 = 2u * pair;
            uint cycle_1 = cycle_0 + 1u;
            SolinasFp128 az_0 = outer_fold_a_collapsed(
                compact_rows[cycle_0], a_lookup);
            SolinasFp128 az_1 = outer_fold_a_collapsed(
                compact_rows[cycle_1], a_lookup);
            SolinasFp128 bz_0 = outer_bind(
                state[2u * cycle_0],
                state[2u * cycle_0 + 1u],
                challenge);
            SolinasFp128 bz_1 = outer_bind(
                state[2u * cycle_1],
                state[2u * cycle_1 + 1u],
                challenge);
            state[2u * cycle_0] = az_0;
            state[2u * cycle_0 + 1u] = bz_0;
            state[2u * cycle_1] = az_1;
            state[2u * cycle_1 + 1u] = bz_1;

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

kernel void solinas_outer_remainder_bind_and_message(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant SolinasFp128& challenge [[buffer(5)]],
    constant OuterRemainderPhaseParams& params [[buffer(6)]],
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
        uint source_cell = 4u * pair;
        uint destination_cell = 2u * pair;
        SolinasFp128 az_0 = outer_bind(
            source[2u * source_cell], source[2u * (source_cell + 1u)], challenge);
        SolinasFp128 bz_0 = outer_bind(
            source[2u * source_cell + 1u],
            source[2u * (source_cell + 1u) + 1u],
            challenge);
        SolinasFp128 az_1 = outer_bind(
            source[2u * (source_cell + 2u)],
            source[2u * (source_cell + 3u)],
            challenge);
        SolinasFp128 bz_1 = outer_bind(
            source[2u * (source_cell + 2u) + 1u],
            source[2u * (source_cell + 3u) + 1u],
            challenge);
        destination[2u * destination_cell] = az_0;
        destination[2u * destination_cell + 1u] = bz_0;
        destination[2u * (destination_cell + 1u)] = az_1;
        destination[2u * (destination_cell + 1u) + 1u] = bz_1;

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

inline ulong outer_staged_word(
    threadgroup const ulong* row_words,
    uint row,
    uint word)
{
    return row_words[row * 20u + word];
}

inline SolinasFp128 outer_product_uniskip_endpoint(
    SolinasFp128 left_input,
    SolinasFp128 right_input,
    SolinasFp128 lookup_output,
    ulong flags,
    bool plus_two)
{
    SolinasFp128 one = outer_from_u64(1ul);
    SolinasFp128 three = solinas_add(one, solinas_add(one, one));
    SolinasFp128 three_lookup = solinas_add(
        lookup_output,
        solinas_add(lookup_output, lookup_output));
    SolinasFp128 three_left = solinas_add(
        left_input,
        solinas_add(left_input, left_input));
    SolinasFp128 left = plus_two
        ? solinas_sub(left_input, three_lookup)
        : solinas_sub(three_left, three_lookup);
    if (outer_flag(flags, 5u) != 0) {
        left = solinas_add(left, plus_two ? three : one);
    }

    SolinasFp128 three_right = solinas_add(
        right_input,
        solinas_add(right_input, right_input));
    SolinasFp128 right = plus_two ? right_input : three_right;
    if (outer_flag(flags, 25u) != 0) {
        right = solinas_sub(right, three);
    }
    if (outer_flag(flags, 26u) == 0) {
        right = solinas_add(right, plus_two ? three : one);
    }
    return solinas_mul_wide(left, right);
}

inline SolinasFp128 outer_opening_value(
    threadgroup const ulong* row_words,
    uint row,
    uint column)
{
    ulong flags = outer_staged_word(row_words, row, 5u);

    switch (column) {
        case 0u: return outer_from_u64(outer_staged_word(row_words, row, 6u));
        case 1u:
            return outer_from_signed_u128(
                outer_staged_word(row_words, row, 7u),
                outer_staged_word(row_words, row, 8u),
                outer_flag(flags, 17u) != 0);
        case 2u:
            return outer_from_signed_u128(
                outer_staged_word(row_words, row, 9u),
                outer_staged_word(row_words, row, 10u),
                outer_flag(flags, 19u) != 0);
        case 3u: return outer_from_u64((ulong)outer_flag(flags, 6u));
        case 4u: return outer_from_u64(outer_staged_word(row_words, row, 11u));
        case 5u: return outer_from_u64(outer_staged_word(row_words, row, 1u));
        case 6u:
            return outer_from_signed_u128(
                outer_staged_word(row_words, row, 3u),
                outer_staged_word(row_words, row, 4u),
                outer_flag(flags, 18u) != 0);
        case 7u: {
            bool access = outer_flag(flags, 0u) != 0 ||
                outer_flag(flags, 1u) != 0;
            ulong value = access ? outer_staged_word(row_words, row, 12u) : 0ul;
            return outer_from_u64(value);
        }
        case 8u: return outer_from_u64(outer_staged_word(row_words, row, 0u));
        case 9u: return outer_from_u64(outer_staged_word(row_words, row, 2u));
        case 10u: {
            bool load = outer_flag(flags, 0u) != 0;
            bool store = outer_flag(flags, 1u) != 0;
            ulong value = store
                ? 0ul
                : outer_staged_word(row_words, row, load ? 13u : 12u);
            return outer_from_u64(value);
        }
        case 11u: {
            bool access = outer_flag(flags, 0u) != 0 ||
                outer_flag(flags, 1u) != 0;
            ulong value = access ? outer_staged_word(row_words, row, 13u) : 0ul;
            return outer_from_u64(value);
        }
        case 12u: {
            bool load = outer_flag(flags, 0u) != 0;
            bool store = outer_flag(flags, 1u) != 0;
            ulong value = load
                ? outer_staged_word(row_words, row, 13u)
                : (store ? outer_staged_word(row_words, row, 2u) : 0ul);
            return outer_from_u64(value);
        }
        case 13u: return outer_from_u64(outer_staged_word(row_words, row, 14u));
        case 14u:
            return outer_from_signed_u128(
                outer_staged_word(row_words, row, 15u),
                outer_staged_word(row_words, row, 16u),
                true);
        case 15u: return outer_from_u64(outer_staged_word(row_words, row, 17u));
        case 16u: return outer_from_u64(outer_staged_word(row_words, row, 18u));
        case 17u: return outer_from_u64((ulong)outer_flag(flags, 11u));
        case 18u: return outer_from_u64((ulong)outer_flag(flags, 12u));
        case 19u: return outer_from_u64(outer_staged_word(row_words, row, 19u));
        case 20u: return outer_from_u64((ulong)outer_flag(flags, 8u));
        case 21u: return outer_from_u64((ulong)outer_flag(flags, 2u));
        case 22u: return outer_from_u64((ulong)outer_flag(flags, 3u));
        case 23u: return outer_from_u64((ulong)outer_flag(flags, 4u));
        case 24u: return outer_from_u64((ulong)outer_flag(flags, 0u));
        case 25u: return outer_from_u64((ulong)outer_flag(flags, 1u));
        case 26u: return outer_from_u64((ulong)outer_flag(flags, 5u));
        case 27u: return outer_from_u64((ulong)outer_flag(flags, 14u));
        case 28u: return outer_from_u64((ulong)outer_flag(flags, 9u));
        case 29u: return outer_from_u64((ulong)outer_flag(flags, 7u));
        case 30u: return outer_from_u64((ulong)outer_flag(flags, 15u));
        case 31u: return outer_from_u64((ulong)outer_flag(flags, 13u));
        case 32u: return outer_from_u64((ulong)outer_flag(flags, 16u));
        case 33u: return outer_from_u64((ulong)outer_flag(flags, 24u));
        case 34u: return outer_from_u64((ulong)outer_flag(flags, 10u));
        case 35u:
            return outer_product_uniskip_endpoint(
                outer_from_u64(outer_staged_word(row_words, row, 6u)),
                outer_from_signed_u128(
                    outer_staged_word(row_words, row, 7u),
                    outer_staged_word(row_words, row, 8u),
                    outer_flag(flags, 17u) != 0),
                outer_from_u64(outer_staged_word(row_words, row, 19u)),
                flags,
                false);
        default:
            return outer_product_uniskip_endpoint(
                outer_from_u64(outer_staged_word(row_words, row, 6u)),
                outer_from_signed_u128(
                    outer_staged_word(row_words, row, 7u),
                    outer_staged_word(row_words, row, 8u),
                    outer_flag(flags, 17u) != 0),
                outer_from_u64(outer_staged_word(row_words, row, 19u)),
                flags,
                true);
    }
}

inline bool outer_opening_is_boolean(uint column) {
    return column == 3u || column == 17u || column == 18u ||
        column == 20u ||
        (column >= 21u && column < OUTER_REMAINDER_CANONICAL_OPENINGS);
}

inline bool outer_opening_boolean(
    threadgroup const ulong* row_words,
    uint row,
    uint column)
{
    ulong flags = outer_staged_word(row_words, row, 5u);
    return outer_flag(flags, OUTER_REMAINDER_BOOLEAN_FLAG_BITS[column]) != 0;
}

inline bool outer_opening_is_u64(uint column) {
    return column == 0u || (column >= 4u && column <= 5u) ||
        (column >= 7u && column <= 13u) ||
        (column >= 15u && column <= 16u) || column == 19u;
}

inline bool outer_opening_is_replaced_by_registers_claim(
    constant OuterRemainderOpeningParams& params,
    uint column)
{
    return params.reserved_0 != 0u &&
        (column == 8u || column == 9u || column == 10u);
}

inline ulong outer_opening_u64(
    threadgroup const ulong* row_words,
    uint row,
    uint column)
{
    ulong flags = outer_staged_word(row_words, row, 5u);
    switch (column) {
        case 0u: return outer_staged_word(row_words, row, 6u);
        case 4u: return outer_staged_word(row_words, row, 11u);
        case 5u: return outer_staged_word(row_words, row, 1u);
        case 7u: {
            bool access = outer_flag(flags, 0u) != 0 ||
                outer_flag(flags, 1u) != 0;
            return access ? outer_staged_word(row_words, row, 12u) : 0ul;
        }
        case 8u: return outer_staged_word(row_words, row, 0u);
        case 9u: return outer_staged_word(row_words, row, 2u);
        case 10u: {
            bool load = outer_flag(flags, 0u) != 0;
            bool store = outer_flag(flags, 1u) != 0;
            return store
                ? 0ul
                : outer_staged_word(row_words, row, load ? 13u : 12u);
        }
        case 11u: {
            bool access = outer_flag(flags, 0u) != 0 ||
                outer_flag(flags, 1u) != 0;
            return access ? outer_staged_word(row_words, row, 13u) : 0ul;
        }
        case 12u: {
            bool load = outer_flag(flags, 0u) != 0;
            bool store = outer_flag(flags, 1u) != 0;
            return load
                ? outer_staged_word(row_words, row, 13u)
                : (store ? outer_staged_word(row_words, row, 2u) : 0ul);
        }
        case 13u: return outer_staged_word(row_words, row, 14u);
        case 15u: return outer_staged_word(row_words, row, 17u);
        case 16u: return outer_staged_word(row_words, row, 18u);
        default: return outer_staged_word(row_words, row, 19u);
    }
}

kernel void solinas_outer_remainder_opening_tiles(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterSuccessorRow* successor_rows [[buffer(1)]],
    device const SpartanOuterColdRow* cold_rows [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant OuterRemainderOpeningParams& params [[buffer(6)]],
    threadgroup ulong* row_words [[threadgroup(0)]],
    threadgroup SolinasFp128* tile_weights [[threadgroup(1)]],
    threadgroup SolinasFp128* shard_sums [[threadgroup(2)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    (void)shard_sums;
    uint simdgroups = threads / OUTER_REMAINDER_SIMD_WIDTH;
    if (tid < params.columns) {
        partials[block * params.columns + tid] = solinas_zero();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint x_out = block;
         x_out < params.e_out_length;
         x_out += params.blocks) {
        SolinasFp128 sums[OUTER_REMAINDER_MAX_COLUMNS_PER_SIMDGROUP];
        for (uint slot = 0u;
             slot < OUTER_REMAINDER_MAX_COLUMNS_PER_SIMDGROUP;
             slot++) {
            sums[slot] = solinas_zero();
        }
        uint block_start = x_out * params.e_in_length;
        uint block_rows = min(
            params.e_in_length,
            params.source_elements - block_start);
        for (uint tile_start = 0u;
             tile_start < block_rows;
             tile_start += OUTER_REMAINDER_TILE_ROWS) {
            uint tile_count = min(
                OUTER_REMAINDER_TILE_ROWS,
                block_rows - tile_start);
            for (uint flat = tid; flat < tile_count * 20u; flat += threads) {
                uint tile_row = flat / 20u;
                uint word = flat - tile_row * 20u;
                uint source_row = block_start + tile_start + tile_row;
                uint residual_word = word - 6u;
                row_words[flat] = word < 6u
                    ? instruction_input_row_word(compact_rows[source_row], word)
                    : spartan_outer_residual_word(
                        successor_rows[source_row],
                        cold_rows[source_row],
                        residual_word);
            }
            for (uint tile_row = tid; tile_row < tile_count; tile_row += threads) {
                tile_weights[tile_row] = e_in[tile_start + tile_row];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint slot = 0u;
                 slot < OUTER_REMAINDER_MAX_COLUMNS_PER_SIMDGROUP;
                 slot++) {
                uint column = simdgroup + slot * simdgroups;
                if (column < params.columns &&
                    !outer_opening_is_replaced_by_registers_claim(params, column)) {
                    SolinasFp128 sum = sums[slot];
                    if (outer_opening_is_boolean(column)) {
                        for (uint tile_row = lane;
                             tile_row < tile_count;
                             tile_row += OUTER_REMAINDER_SIMD_WIDTH) {
                            if (outer_opening_boolean(row_words, tile_row, column)) {
                                sum = solinas_add(sum, tile_weights[tile_row]);
                            }
                        }
                    } else if (outer_opening_is_u64(column)) {
                        for (uint tile_row = lane;
                             tile_row < tile_count;
                             tile_row += OUTER_REMAINDER_SIMD_WIDTH) {
                            sum = solinas_add(
                                sum,
                                solinas_half_width_mul_u64(
                                    tile_weights[tile_row],
                                    outer_opening_u64(
                                        row_words, tile_row, column)));
                        }
                    } else {
                        for (uint tile_row = lane;
                             tile_row < tile_count;
                             tile_row += OUTER_REMAINDER_SIMD_WIDTH) {
                            SolinasFp128 value = outer_opening_value(
                                row_words, tile_row, column);
                            sum = solinas_add(
                                sum,
                                solinas_mul_wide(tile_weights[tile_row], value));
                        }
                    }
                    sums[slot] = sum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint slot = 0u;
             slot < OUTER_REMAINDER_MAX_COLUMNS_PER_SIMDGROUP;
             slot++) {
            uint column = simdgroup + slot * simdgroups;
            if (column < params.columns &&
                !outer_opening_is_replaced_by_registers_claim(params, column)) {
                SolinasFp128 column_sum = solinas_simd_sum_32(sums[slot]);
                if (lane == 0u) {
                    uint output = block * params.columns + column;
                    partials[output] = solinas_add(
                        partials[output],
                        solinas_mul_wide(e_out[x_out], column_sum));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void solinas_outer_remainder_build_registers_claim(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterSuccessorRow* successor_rows [[buffer(1)]],
    device const SpartanOuterColdRow* cold_rows [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* q_partials [[buffer(4)]],
    device ulong* rd_write_value [[buffer(5)]],
    constant OuterRemainderOpeningParams& params [[buffer(6)]],
    threadgroup SolinasFp128* outer_weights [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint low_groups = (params.e_in_length + threads - 1u) / threads;
    uint block = group / low_groups;
    uint low_group = group - block * low_groups;
    uint x_lo = low_group * threads + tid;
    uint high_count =
        (params.e_out_length + params.blocks - 1u - block) / params.blocks;
    for (uint index = tid; index < high_count; index += threads) {
        outer_weights[index] = e_out[block + index * params.blocks];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (x_lo >= params.e_in_length) {
        return;
    }

    SolinasFp128 sums[OUTER_REMAINDER_REGISTERS_COMPONENTS];
    for (uint component = 0u;
         component < OUTER_REMAINDER_REGISTERS_COMPONENTS;
         component++) {
        sums[component] = solinas_zero();
    }
    for (uint high = 0u; high < high_count; high++) {
        uint x_hi = block + high * params.blocks;
        uint row = x_hi * params.e_in_length + x_lo;
        ulong rs1 = 0ul;
        ulong rs2 = 0ul;
        ulong rd = 0ul;
        if (row < params.source_elements) {
            device const InstructionInputRow& compact = compact_rows[row];
            device const SpartanOuterSuccessorRow& successor = successor_rows[row];
            device const SpartanOuterColdRow& cold = cold_rows[row];
            ulong flags = instruction_input_row_word(compact, 5u);
            bool load = outer_flag(flags, 0u) != 0;
            bool store = outer_flag(flags, 1u) != 0;
            rs1 = instruction_input_row_word(compact, 0u);
            rs2 = instruction_input_row_word(compact, 2u);
            rd = store
                ? 0ul
                : spartan_outer_residual_word(
                    successor, cold, load ? 7u : 6u);
        }
        rd_write_value[row] = rd;
        SolinasFp128 weight = outer_weights[high];
        sums[0] = solinas_add(
            sums[0], solinas_half_width_mul_u64(weight, rd));
        sums[1] = solinas_add(
            sums[1], solinas_half_width_mul_u64(weight, rs1));
        sums[2] = solinas_add(
            sums[2], solinas_half_width_mul_u64(weight, rs2));
    }
    for (uint component = 0u;
         component < OUTER_REMAINDER_REGISTERS_COMPONENTS;
         component++) {
        uint output =
            (component * params.blocks + block) * params.e_in_length + x_lo;
        q_partials[output] = sums[component];
    }
}

kernel void solinas_outer_remainder_reduce_registers_claim(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* components [[buffer(1)]],
    constant OuterRemainderOpeningParams& params [[buffer(2)]],
    uint output [[thread_position_in_grid]])
{
    uint outputs =
        OUTER_REMAINDER_REGISTERS_COMPONENTS * params.e_in_length;
    if (output >= outputs) {
        return;
    }
    uint component = output / params.e_in_length;
    uint x_lo = output - component * params.e_in_length;
    SolinasFp128 sum = solinas_zero();
    for (uint block = 0u; block < params.blocks; block++) {
        uint index =
            (component * params.blocks + block) * params.e_in_length + x_lo;
        sum = solinas_add(sum, partials[index]);
    }
    components[output] = sum;
}

kernel void solinas_outer_remainder_dot_registers_claim(
    device const SolinasFp128* components [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device SolinasFp128* openings [[buffer(2)]],
    constant OuterRemainderOpeningParams& params [[buffer(3)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint component [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sum = solinas_zero();
    uint offset = component * params.e_in_length;
    for (uint x_lo = tid; x_lo < params.e_in_length; x_lo += threads) {
        sum = solinas_add(
            sum,
            solinas_mul_wide(e_in[x_lo], components[offset + x_lo]));
    }
    sum = solinas_simd_sum_32(sum);
    if (lane == 0u) {
        shared[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        uint simdgroups = threads / OUTER_REMAINDER_SIMD_WIDTH;
        sum = lane < simdgroups ? shared[lane] : solinas_zero();
        sum = solinas_simd_sum_32(sum);
        if (lane == 0u) {
            uint column = component == 0u ? 10u : (component == 1u ? 8u : 9u);
            openings[column] = sum;
        }
    }
}

kernel void solinas_outer_remainder_reduce_columns(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant OuterRemainderReduceParams& params [[buffer(2)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint column [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sum = solinas_zero();
    for (uint block = tid; block < params.input_count; block += threads) {
        sum = solinas_add(sum, partials[block * params.columns + column]);
    }
    sum = solinas_simd_sum_32(sum);
    if (lane == 0u) {
        shared[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        uint simdgroups = threads / OUTER_REMAINDER_SIMD_WIDTH;
        sum = lane < simdgroups ? shared[lane] : solinas_zero();
        sum = solinas_simd_sum_32(sum);
        if (lane == 0u) {
            output[column] = sum;
        }
    }
}
