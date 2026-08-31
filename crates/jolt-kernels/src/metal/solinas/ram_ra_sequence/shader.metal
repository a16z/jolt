// Direct cycle-order RAM RA virtualization. The fourth reduction lane stays
// zero so this sequence can share the Instruction RA column reducer.

#define RAM_RA_MAX_FACTORS 3u
#define RAM_RA_BINS 256u

struct RamRaMessageParams {
    uint e_in_length;
    uint e_out_length;
    uint factor_count;
    uint reserved;
};

struct RamRaBranchParams {
    uint branch_width;
    uint factor_count;
    uint2 reserved;
};

struct RamRaMaterializeParams {
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint factor_count;
};

constant uint ram_ra_branch_width [[function_constant(20)]];

inline uint ram_ra_address_byte(uint address, uint factor, uint factor_count) {
    return (address >> ((factor_count - 1u - factor) * 8u)) & 0xffu;
}

inline bool ram_ra_gather(
    uint pair,
    uint branch_width,
    uint factor_count,
    device const uint* addresses,
    device const SolinasFp128* branches,
    thread InstructionRaLinear* factors)
{
    SolinasFp128 lo[RAM_RA_MAX_FACTORS];
    SolinasFp128 hi[RAM_RA_MAX_FACTORS];
    for (uint factor = 0; factor < factor_count; factor++) {
        lo[factor] = solinas_zero();
        hi[factor] = solinas_zero();
    }

    bool active = false;
    uint original = 2u * pair * branch_width;
    for (uint offset = 0; offset < branch_width; offset++) {
        uint lo_address = addresses[original + offset];
        uint hi_address = addresses[original + branch_width + offset];
        if (lo_address != 0xffffffffu) {
            active = true;
            for (uint factor = 0; factor < factor_count; factor++) {
                uint table = (factor * branch_width + offset) * RAM_RA_BINS;
                lo[factor] = solinas_add(
                    lo[factor],
                    branches[table + ram_ra_address_byte(lo_address, factor, factor_count)]);
            }
        }
        if (hi_address != 0xffffffffu) {
            active = true;
            for (uint factor = 0; factor < factor_count; factor++) {
                uint table = (factor * branch_width + offset) * RAM_RA_BINS;
                hi[factor] = solinas_add(
                    hi[factor],
                    branches[table + ram_ra_address_byte(hi_address, factor, factor_count)]);
            }
        }
    }

    for (uint factor = 0; factor < factor_count; factor++) {
        factors[factor].at_one = hi[factor];
        factors[factor].at_infinity = solinas_sub(hi[factor], lo[factor]);
    }
    return active;
}

inline void ram_ra_product(
    thread const InstructionRaLinear* factors,
    uint factor_count,
    thread SolinasFp128* q)
{
    if (factor_count == 2u) {
        q[0] = solinas_mul_wide(factors[0].at_one, factors[1].at_one);
        q[1] = solinas_mul_wide(factors[0].at_infinity, factors[1].at_infinity);
        q[2] = solinas_zero();
        q[3] = solinas_zero();
        return;
    }
    InstructionRaQuadratic first = instruction_ra_quadratic(factors[0], factors[1]);
    q[0] = solinas_mul_wide(first.at_one, factors[2].at_one);
    q[1] = solinas_mul_wide(
        first.at_two,
        solinas_add(factors[2].at_one, factors[2].at_infinity));
    q[2] = solinas_mul_wide(first.at_infinity, factors[2].at_infinity);
    q[3] = solinas_zero();
}

kernel void solinas_ram_ra_lazy_message(
    device const uint* addresses [[buffer(0)]],
    device const SolinasFp128* branches [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant RamRaMessageParams& params [[buffer(5)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[INSTRUCTION_RA_SAMPLES];
    for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    for (uint x_in = x_in_thread; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        InstructionRaLinear factors[RAM_RA_MAX_FACTORS];
        if (!ram_ra_gather(
                pair,
                ram_ra_branch_width,
                params.factor_count,
                addresses,
                branches,
                factors)) {
            continue;
        }
        SolinasFp128 q[INSTRUCTION_RA_SAMPLES];
        ram_ra_product(factors, params.factor_count, q);
        for (uint sample = 0; sample < params.factor_count; sample++) {
            lanes[sample] = solinas_add(
                lanes[sample],
                solinas_mul_wide(e_in[x_in], q[sample]));
        }
    }

    instruction_ra_finish_block(
        lanes,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}

kernel void solinas_ram_ra_double_branches(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    constant SolinasFp128& challenge [[buffer(2)]],
    constant RamRaBranchParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint per_factor = params.branch_width * RAM_RA_BINS;
    uint elements = params.factor_count * per_factor;
    if (gid >= elements) {
        return;
    }
    uint factor = gid / per_factor;
    uint within = gid - factor * per_factor;
    uint destination_base = factor * 2u * per_factor;
    SolinasFp128 value = source[gid];
    SolinasFp128 one = solinas_zero();
    one.limb[0] = 1u;
    destination[destination_base + within] = solinas_mul_wide(
        solinas_sub(one, challenge), value);
    destination[destination_base + per_factor + within] =
        solinas_mul_wide(challenge, value);
}

kernel void solinas_ram_ra_materialize_width_16(
    device const uint* addresses [[buffer(0)]],
    device const SolinasFp128* branches [[buffer(1)]],
    device SolinasFp128* dense [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant RamRaMaterializeParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[INSTRUCTION_RA_SAMPLES];
    for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    for (uint x_in = x_in_thread; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        InstructionRaLinear factors[RAM_RA_MAX_FACTORS];
        bool active = ram_ra_gather(
            pair, 16u, params.factor_count, addresses, branches, factors);
        for (uint factor = 0; factor < params.factor_count; factor++) {
            uint destination = factor * params.source_elements + 2u * pair;
            dense[destination] = solinas_sub(
                factors[factor].at_one,
                factors[factor].at_infinity);
            dense[destination + 1u] = factors[factor].at_one;
        }
        if (!active) {
            continue;
        }
        SolinasFp128 q[INSTRUCTION_RA_SAMPLES];
        ram_ra_product(factors, params.factor_count, q);
        for (uint sample = 0; sample < params.factor_count; sample++) {
            lanes[sample] = solinas_add(
                lanes[sample],
                solinas_mul_wide(e_in[x_in], q[sample]));
        }
    }

    instruction_ra_finish_block(
        lanes,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}

kernel void solinas_ram_ra_dense_transition(
    device const SolinasFp128* tables [[buffer(0)]],
    device SolinasFp128* bound [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant SolinasFp128& challenge [[buffer(5)]],
    constant RamRaMaterializeParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[INSTRUCTION_RA_SAMPLES];
    for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }
    uint bound_elements = params.source_elements / 2u;

    for (uint x_in = x_in_thread; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        InstructionRaLinear factors[RAM_RA_MAX_FACTORS];
        for (uint factor = 0; factor < params.factor_count; factor++) {
            uint source = factor * params.source_elements + 4u * pair;
            SolinasFp128 lo_0 = tables[source];
            SolinasFp128 hi_0 = tables[source + 1u];
            SolinasFp128 lo_1 = tables[source + 2u];
            SolinasFp128 hi_1 = tables[source + 3u];
            SolinasFp128 bound_0 = solinas_add(
                lo_0,
                solinas_mul_wide(challenge, solinas_sub(hi_0, lo_0)));
            SolinasFp128 bound_1 = solinas_add(
                lo_1,
                solinas_mul_wide(challenge, solinas_sub(hi_1, lo_1)));
            uint destination = factor * bound_elements + 2u * pair;
            bound[destination] = bound_0;
            bound[destination + 1u] = bound_1;
            factors[factor].at_one = bound_1;
            factors[factor].at_infinity = solinas_sub(bound_1, bound_0);
        }
        SolinasFp128 q[INSTRUCTION_RA_SAMPLES];
        ram_ra_product(factors, params.factor_count, q);
        for (uint sample = 0; sample < params.factor_count; sample++) {
            lanes[sample] = solinas_add(
                lanes[sample],
                solinas_mul_wide(e_in[x_in], q[sample]));
        }
    }

    instruction_ra_finish_block(
        lanes,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}
