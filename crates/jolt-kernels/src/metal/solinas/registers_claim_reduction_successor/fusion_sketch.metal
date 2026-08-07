// Unregistered schedule sketch. This is not source-assembled or compiled.

struct RegistersClaimFusionParams {
    uint prefix_elements;
    uint suffix_elements;
    uint blocks;
    uint reserved;
};

// Integrate this loop shape into the OuterRemainder opening shader rather than
// dispatching it as a second full-domain row scan. The real implementation
// retains the existing 32 non-register scalar-opening lanes around this core.
kernel void registers_claim_fused_opening_sketch(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterUniskipResidualRow* residual_rows [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* q_partials [[buffer(4)]],
    device ulong* rd_plane [[buffer(5)]],
    constant RegistersClaimFusionParams& params [[buffer(6)]],
    threadgroup ulong* staged_words [[threadgroup(0)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    for (uint tile = 0u; tile < params.prefix_elements; tile += 32u) {
        SolinasFp128 q_lane = solinas_zero();

        for (uint x_hi = block; x_hi < params.suffix_elements;
             x_hi += params.blocks) {
            uint row_start = x_hi * params.prefix_elements + tile;

            // Reuse the real opening shader's coalesced row staging and its
            // exact rd/load/store reconstruction. No global row reread occurs.
            outer_stage_rows(
                compact_rows,
                residual_rows,
                row_start,
                staged_words,
                tid,
                threads);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lane < 32u && tile + lane < params.prefix_elements) {
                ulong value = 0ul;
                if (simdgroup == 0u) {
                    value = outer_staged_rd(staged_words, lane);
                    rd_plane[row_start + lane] = value;
                } else if (simdgroup == 1u) {
                    value = outer_staged_rs1(staged_words, lane);
                } else if (simdgroup == 2u) {
                    value = outer_staged_rs2(staged_words, lane);
                }
                if (simdgroup < 3u) {
                    q_lane = solinas_add(
                        q_lane,
                        solinas_half_width_mul_u64(e_out[x_hi], value));
                }
            }

            // The production body also accumulates the other 32 scalar
            // openings here, weighted by e_in[tile + lane] and e_out[x_hi].
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simdgroup < 3u && lane < 32u && tile + lane < params.prefix_elements) {
            uint column = simdgroup;
            uint output =
                (column * params.blocks + block) * params.prefix_elements
                + tile + lane;
            q_partials[output] = q_lane;
        }
    }
}

// A following dispatch in the same command buffer reduces the B partials for
// each (column, x_lo), writes the immutable 3P component carrier, and dots it
// with e_in to fill stage-1 opening columns 8, 9, and 10.
