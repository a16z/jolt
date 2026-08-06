#define PRODUCT5_FACTORS 5u

struct Product5Params {
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

struct Product5ReductionParams {
    uint input_count;
    uint output_count;
    uint2 reserved;
};

inline SolinasFp128 product5_product(thread const SolinasFp128* factors) {
    SolinasFp128 product = factors[0];
    for (uint factor = 1; factor < PRODUCT5_FACTORS; factor++) {
        product = solinas_mul_wide(product, factors[factor]);
    }
    return product;
}

inline void product5_finish_block(
    thread SolinasFp128* lanes,
    SolinasFp128 outer_weight,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint x_out,
    uint e_out_length,
    uint lane_in_simd,
    uint simdgroup,
    uint simdgroups)
{
    for (uint sample = 0; sample < PRODUCT5_FACTORS; sample++) {
        SolinasFp128 sum = solinas_simd_sum_32(lanes[sample]);
        if (lane_in_simd == 0) {
            shared[sample * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        for (uint sample = 0; sample < PRODUCT5_FACTORS; sample++) {
            SolinasFp128 sum = lane_in_simd < simdgroups
                ? shared[sample * simdgroups + lane_in_simd]
                : solinas_zero();
            sum = solinas_simd_sum_32(sum);
            if (lane_in_simd == 0) {
                partials[sample * e_out_length + x_out] =
                    solinas_mul_wide(outer_weight, sum);
            }
        }
    }
}

kernel void solinas_product5_message(
    device const SolinasFp128* tables [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant Product5Params& params [[buffer(4)]],
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
        SolinasFp128 evals[PRODUCT5_FACTORS];
        SolinasFp128 steps[PRODUCT5_FACTORS];
        for (uint factor = 0; factor < PRODUCT5_FACTORS; factor++) {
            uint base = factor * params.source_elements + 2 * pair;
            SolinasFp128 lo = tables[base];
            SolinasFp128 hi = tables[base + 1];
            if (factor == 0) {
                lo = solinas_mul_wide(e_in[x_in], lo);
                hi = solinas_mul_wide(e_in[x_in], hi);
            }
            evals[factor] = hi;
            steps[factor] = solinas_sub(hi, lo);
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

kernel void solinas_product5_fused_transition(
    device const SolinasFp128* tables [[buffer(0)]],
    device SolinasFp128* bound [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant SolinasFp128& challenge [[buffer(5)]],
    constant Product5Params& params [[buffer(6)]],
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
    uint bound_elements = params.source_elements / 2;

    for (uint x_in = x_in_thread; x_in < params.e_in_length;
         x_in += threads_per_threadgroup) {
        uint pair = x_out * params.e_in_length + x_in;
        SolinasFp128 evals[PRODUCT5_FACTORS];
        SolinasFp128 steps[PRODUCT5_FACTORS];
        for (uint factor = 0; factor < PRODUCT5_FACTORS; factor++) {
            uint source = factor * params.source_elements + 4 * pair;
            SolinasFp128 lo_0 = tables[source];
            SolinasFp128 hi_0 = tables[source + 1];
            SolinasFp128 lo_1 = tables[source + 2];
            SolinasFp128 hi_1 = tables[source + 3];
            SolinasFp128 bound_0 = solinas_add(
                lo_0,
                solinas_mul_wide(challenge, solinas_sub(hi_0, lo_0)));
            SolinasFp128 bound_1 = solinas_add(
                lo_1,
                solinas_mul_wide(challenge, solinas_sub(hi_1, lo_1)));
            uint destination = factor * bound_elements + 2 * pair;
            bound[destination] = bound_0;
            bound[destination + 1] = bound_1;
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

kernel void solinas_product5_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant Product5ReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]])
{
    for (uint sample = 0; sample < PRODUCT5_FACTORS; sample++) {
        SolinasFp128 value = gid < params.input_count
            ? input[sample * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane_in_simd == 0) {
            output[sample * params.output_count + gid / 32] = value;
        }
    }
}
