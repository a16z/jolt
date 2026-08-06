kernel void solinas_instruction_ra_first_message(
    device const InstructionRaLookup* lookups [[buffer(0)]],
    device const uint* cycle_to_table_major [[buffer(1)]],
    device const SolinasFp128* chunk_tables [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant InstructionRaFirstMessageParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[INSTRUCTION_RA_SAMPLES];
    for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    for (uint x_in = x_in_thread; x_in < params.e_in_length;
         x_in += threads_per_threadgroup) {
        uint pair = x_out * params.e_in_length + x_in;
        uint lo_row = cycle_to_table_major[2u * pair];
        uint hi_row = cycle_to_table_major[2u * pair + 1u];
        InstructionRaLookup lo_lookup = lookups[lo_row];
        InstructionRaLookup hi_lookup = lookups[hi_row];

        SolinasFp128 q[INSTRUCTION_RA_SAMPLES];
        for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
            q[sample] = solinas_zero();
        }
        for (uint group = 0; group < INSTRUCTION_RA_GROUPS; group++) {
            instruction_ra_accumulate_group(
                group,
                lo_lookup,
                hi_lookup,
                chunk_tables,
                q);
        }
        for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
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
        threads_per_threadgroup / 32u);
}

kernel void solinas_instruction_ra_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant InstructionRaReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]])
{
    for (uint sample = 0; sample < INSTRUCTION_RA_SAMPLES; sample++) {
        SolinasFp128 value = gid < params.input_count
            ? input[sample * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane_in_simd == 0) {
            output[sample * params.output_count + gid / 32u] = value;
        }
    }
}
