#define ADDRESS_CYCLE_PHASE_BINS 256u
#define ADDRESS_CYCLE_RA_FACTORS 4u

struct AddressCycleLookup {
    ulong2 limbs;
};

struct AddressCycleParams {
    uint rows;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

inline uint address_cycle_lookup_byte(AddressCycleLookup lookup, uint shift) {
    return shift < 64
        ? (uint)(lookup.limbs[0] >> shift) & 0xffu
        : (uint)(lookup.limbs[1] >> (shift - 64)) & 0xffu;
}

inline void address_cycle_factors(
    uint cycle,
    device const uchar* packed_rows,
    device const AddressCycleLookup* lookups,
    device const uint* cycle_to_table_major,
    device const SolinasFp128* phase_tables,
    device const SolinasFp128* table_values,
    SolinasFp128 raf_interleaved,
    SolinasFp128 raf_identity,
    thread SolinasFp128* factors)
{
    uint row = cycle_to_table_major[cycle];
    uchar packed = packed_rows[row];
    uint table = (uint)packed & 0x7fu;
    factors[0] = table == 0
        ? solinas_zero()
        : table_values[table - 1];
    factors[0] = solinas_add(
        factors[0],
        (packed & 0x80u) == 0 ? raf_interleaved : raf_identity);

    AddressCycleLookup lookup = lookups[row];
    for (uint factor = 0; factor < ADDRESS_CYCLE_RA_FACTORS; factor++) {
        uint phase = factor * 4u;
        uint shift = 120u - phase * 8u;
        SolinasFp128 value = phase_tables[
            phase * ADDRESS_CYCLE_PHASE_BINS + address_cycle_lookup_byte(lookup, shift)];
        for (uint offset = 1; offset < 4; offset++) {
            phase += 1;
            shift -= 8u;
            value = solinas_mul_wide(
                value,
                phase_tables[
                    phase * ADDRESS_CYCLE_PHASE_BINS
                    + address_cycle_lookup_byte(lookup, shift)]);
        }
        factors[factor + 1] = value;
    }
}

kernel void solinas_address_cycle_message(
    device const uchar* packed_rows [[buffer(0)]],
    device const AddressCycleLookup* lookups [[buffer(1)]],
    device const uint* cycle_to_table_major [[buffer(2)]],
    device const SolinasFp128* phase_tables [[buffer(3)]],
    device const SolinasFp128* table_values [[buffer(4)]],
    device const SolinasFp128* e_in [[buffer(5)]],
    device const SolinasFp128* e_out [[buffer(6)]],
    device SolinasFp128* partials [[buffer(7)]],
    constant SolinasFp128& raf_interleaved [[buffer(8)]],
    constant SolinasFp128& raf_identity [[buffer(9)]],
    constant AddressCycleParams& params [[buffer(10)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[PRODUCT5_FACTORS];
    for (uint sample = 0; sample < PRODUCT5_FACTORS; sample++) {
        lanes[sample] = solinas_zero();
    }

    for (uint x_in = x_in_thread; x_in < params.e_in_length;
         x_in += threads_per_threadgroup) {
        uint pair = x_out * params.e_in_length + x_in;
        SolinasFp128 lo[PRODUCT5_FACTORS];
        SolinasFp128 hi[PRODUCT5_FACTORS];
        address_cycle_factors(
            2u * pair,
            packed_rows,
            lookups,
            cycle_to_table_major,
            phase_tables,
            table_values,
            raf_interleaved,
            raf_identity,
            lo);
        address_cycle_factors(
            2u * pair + 1u,
            packed_rows,
            lookups,
            cycle_to_table_major,
            phase_tables,
            table_values,
            raf_interleaved,
            raf_identity,
            hi);

        SolinasFp128 evals[PRODUCT5_FACTORS];
        SolinasFp128 steps[PRODUCT5_FACTORS];
        for (uint factor = 0; factor < PRODUCT5_FACTORS; factor++) {
            if (factor == 0) {
                lo[factor] = solinas_mul_wide(e_in[x_in], lo[factor]);
                hi[factor] = solinas_mul_wide(e_in[x_in], hi[factor]);
            }
            evals[factor] = hi[factor];
            steps[factor] = solinas_sub(hi[factor], lo[factor]);
        }
        for (uint sample = 0; sample < PRODUCT5_FACTORS - 1; sample++) {
            lanes[sample] = solinas_add(lanes[sample], product5_product(evals));
            if (sample + 1 < PRODUCT5_FACTORS - 1) {
                for (uint factor = 0; factor < PRODUCT5_FACTORS; factor++) {
                    evals[factor] = solinas_add(evals[factor], steps[factor]);
                }
            }
        }
        lanes[PRODUCT5_FACTORS - 1] = solinas_add(
            lanes[PRODUCT5_FACTORS - 1],
            product5_product(steps));
    }

    product5_finish_block(
        lanes,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads_per_threadgroup / 32);
}

kernel void solinas_address_cycle_bind(
    device const uchar* packed_rows [[buffer(0)]],
    device const AddressCycleLookup* lookups [[buffer(1)]],
    device const uint* cycle_to_table_major [[buffer(2)]],
    device const SolinasFp128* phase_tables [[buffer(3)]],
    device const SolinasFp128* table_values [[buffer(4)]],
    device SolinasFp128* bound [[buffer(5)]],
    constant SolinasFp128& raf_interleaved [[buffer(6)]],
    constant SolinasFp128& raf_identity [[buffer(7)]],
    constant SolinasFp128& challenge [[buffer(8)]],
    constant AddressCycleParams& params [[buffer(9)]],
    uint position [[thread_position_in_grid]])
{
    uint bound_elements = params.rows / 2u;
    if (position >= bound_elements) {
        return;
    }
    SolinasFp128 lo[PRODUCT5_FACTORS];
    SolinasFp128 hi[PRODUCT5_FACTORS];
    address_cycle_factors(
        2u * position,
        packed_rows,
        lookups,
        cycle_to_table_major,
        phase_tables,
        table_values,
        raf_interleaved,
        raf_identity,
        lo);
    address_cycle_factors(
        2u * position + 1u,
        packed_rows,
        lookups,
        cycle_to_table_major,
        phase_tables,
        table_values,
        raf_interleaved,
        raf_identity,
        hi);
    for (uint factor = 0; factor < PRODUCT5_FACTORS; factor++) {
        bound[factor * bound_elements + position] = solinas_add(
            lo[factor],
            solinas_mul_wide(challenge, solinas_sub(hi[factor], lo[factor])));
    }
}

kernel void solinas_address_cycle_fused_transition(
    device const uchar* packed_rows [[buffer(0)]],
    device const AddressCycleLookup* lookups [[buffer(1)]],
    device const uint* cycle_to_table_major [[buffer(2)]],
    device const SolinasFp128* phase_tables [[buffer(3)]],
    device const SolinasFp128* table_values [[buffer(4)]],
    device SolinasFp128* bound [[buffer(5)]],
    device const SolinasFp128* e_in [[buffer(6)]],
    device const SolinasFp128* e_out [[buffer(7)]],
    device SolinasFp128* partials [[buffer(8)]],
    constant SolinasFp128& raf_interleaved [[buffer(9)]],
    constant SolinasFp128& raf_identity [[buffer(10)]],
    constant SolinasFp128& challenge [[buffer(11)]],
    constant AddressCycleParams& params [[buffer(12)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[PRODUCT5_FACTORS];
    for (uint sample = 0; sample < PRODUCT5_FACTORS; sample++) {
        lanes[sample] = solinas_zero();
    }
    uint bound_elements = params.rows / 2u;

    for (uint x_in = x_in_thread; x_in < params.e_in_length;
         x_in += threads_per_threadgroup) {
        uint pair = x_out * params.e_in_length + x_in;
        SolinasFp128 rows[4][PRODUCT5_FACTORS];
        for (uint row = 0; row < 4; row++) {
            address_cycle_factors(
                4u * pair + row,
                packed_rows,
                lookups,
                cycle_to_table_major,
                phase_tables,
                table_values,
                raf_interleaved,
                raf_identity,
                rows[row]);
        }

        SolinasFp128 evals[PRODUCT5_FACTORS];
        SolinasFp128 steps[PRODUCT5_FACTORS];
        for (uint factor = 0; factor < PRODUCT5_FACTORS; factor++) {
            SolinasFp128 bound_0 = solinas_add(
                rows[0][factor],
                solinas_mul_wide(
                    challenge,
                    solinas_sub(rows[1][factor], rows[0][factor])));
            SolinasFp128 bound_1 = solinas_add(
                rows[2][factor],
                solinas_mul_wide(
                    challenge,
                    solinas_sub(rows[3][factor], rows[2][factor])));
            uint destination = factor * bound_elements + 2u * pair;
            bound[destination] = bound_0;
            bound[destination + 1u] = bound_1;
            if (factor == 0) {
                bound_0 = solinas_mul_wide(e_in[x_in], bound_0);
                bound_1 = solinas_mul_wide(e_in[x_in], bound_1);
            }
            evals[factor] = bound_1;
            steps[factor] = solinas_sub(bound_1, bound_0);
        }
        for (uint sample = 0; sample < PRODUCT5_FACTORS - 1; sample++) {
            lanes[sample] = solinas_add(lanes[sample], product5_product(evals));
            if (sample + 1 < PRODUCT5_FACTORS - 1) {
                for (uint factor = 0; factor < PRODUCT5_FACTORS; factor++) {
                    evals[factor] = solinas_add(evals[factor], steps[factor]);
                }
            }
        }
        lanes[PRODUCT5_FACTORS - 1] = solinas_add(
            lanes[PRODUCT5_FACTORS - 1],
            product5_product(steps));
    }

    product5_finish_block(
        lanes,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads_per_threadgroup / 32);
}
