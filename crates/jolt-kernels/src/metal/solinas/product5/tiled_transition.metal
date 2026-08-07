struct Product5TiledTransitionParams {
    uint source_elements;
    uint destination_elements;
    uint e_in_length;
    uint e_out_length;
    uint tile_pairs;
    uint tiles_per_out;
    uint total_tiles;
    uint reserved;
};

template <uint tile_pairs>
inline void product5_tiled_factor_sample_impl(
    device const SolinasFp128* source,
    device SolinasFp128* destination,
    device const SolinasFp128* e_in,
    device SolinasFp128* tile_partials,
    constant SolinasFp128& challenge,
    constant Product5TiledTransitionParams& params,
    threadgroup SolinasFp128* scratch,
    uint tile,
    uint lane,
    uint simdgroup)
{
    if (params.tile_pairs != tile_pairs || tile >= params.total_tiles) {
        return;
    }

    if (simdgroup < 5u) {
        uint factor = simdgroup;
        for (uint local_pair = lane; local_pair < tile_pairs; local_pair += 32u) {
            uint pair = tile * tile_pairs + local_pair;
            uint x_in = pair % params.e_in_length;
            uint source_index = factor * params.source_elements + 4u * pair;
            SolinasFp128 lo_0 = source[source_index];
            SolinasFp128 hi_0 = source[source_index + 1u];
            SolinasFp128 lo_1 = source[source_index + 2u];
            SolinasFp128 hi_1 = source[source_index + 3u];
            SolinasFp128 bound_0 = solinas_add(
                lo_0,
                solinas_mul_wide(challenge, solinas_sub(hi_0, lo_0)));
            SolinasFp128 bound_1 = solinas_add(
                lo_1,
                solinas_mul_wide(challenge, solinas_sub(hi_1, lo_1)));
            uint destination_index = factor * params.destination_elements + 2u * pair;
            destination[destination_index] = bound_0;
            destination[destination_index + 1u] = bound_1;
            if (factor == 0u) {
                SolinasFp128 weight = e_in[x_in];
                bound_0 = solinas_mul_wide(weight, bound_0);
                bound_1 = solinas_mul_wide(weight, bound_1);
            }
            scratch[(2u * factor) * tile_pairs + local_pair] = bound_0;
            scratch[(2u * factor + 1u) * tile_pairs + local_pair] = bound_1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup < 5u) {
        uint sample = simdgroup;
        SolinasFp128 sum = solinas_zero();
        for (uint local_pair = lane; local_pair < tile_pairs; local_pair += 32u) {
            SolinasFp128 product = solinas_zero();
            for (uint factor = 0u; factor < 5u; factor++) {
                SolinasFp128 lo = scratch[(2u * factor) * tile_pairs + local_pair];
                SolinasFp128 hi = scratch[(2u * factor + 1u) * tile_pairs + local_pair];
                SolinasFp128 step = solinas_sub(hi, lo);
                SolinasFp128 value = step;
                if (sample < 4u) {
                    value = hi;
                    for (uint offset = 0u; offset < sample; offset++) {
                        value = solinas_add(value, step);
                    }
                }
                product = factor == 0u ? value : solinas_mul_wide(product, value);
            }
            sum = solinas_add(sum, product);
        }
        sum = solinas_simd_sum_32(sum);
        if (lane == 0u) {
            tile_partials[sample * params.total_tiles + tile] = sum;
        }
    }
}

kernel void solinas_product5_tiled_factor_sample_32(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device SolinasFp128* tile_partials [[buffer(3)]],
    constant SolinasFp128& challenge [[buffer(4)]],
    constant Product5TiledTransitionParams& params [[buffer(5)]],
    threadgroup SolinasFp128* scratch [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    product5_tiled_factor_sample_impl<32u>(
        source, destination, e_in, tile_partials, challenge, params,
        scratch, tile, lane, simdgroup);
}

kernel void solinas_product5_tiled_factor_sample_64(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device SolinasFp128* tile_partials [[buffer(3)]],
    constant SolinasFp128& challenge [[buffer(4)]],
    constant Product5TiledTransitionParams& params [[buffer(5)]],
    threadgroup SolinasFp128* scratch [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    product5_tiled_factor_sample_impl<64u>(
        source, destination, e_in, tile_partials, challenge, params,
        scratch, tile, lane, simdgroup);
}

kernel void solinas_product5_tiled_factor_sample_128(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device SolinasFp128* tile_partials [[buffer(3)]],
    constant SolinasFp128& challenge [[buffer(4)]],
    constant Product5TiledTransitionParams& params [[buffer(5)]],
    threadgroup SolinasFp128* scratch [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    product5_tiled_factor_sample_impl<128u>(
        source, destination, e_in, tile_partials, challenge, params,
        scratch, tile, lane, simdgroup);
}

kernel void solinas_product5_tiled_weight_tiles(
    device const SolinasFp128* tile_partials [[buffer(0)]],
    device const SolinasFp128* e_out [[buffer(1)]],
    device SolinasFp128* outer_partials [[buffer(2)]],
    constant Product5TiledTransitionParams& params [[buffer(3)]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (x_out >= params.e_out_length) {
        return;
    }
    uint tile_start = x_out * params.tiles_per_out;
    for (uint sample = 0u; sample < 5u; sample++) {
        SolinasFp128 sum = solinas_zero();
        for (uint tile = lane; tile < params.tiles_per_out; tile += 32u) {
            sum = solinas_add(
                sum,
                tile_partials[sample * params.total_tiles + tile_start + tile]);
        }
        sum = solinas_simd_sum_32(sum);
        if (lane == 0u) {
            outer_partials[sample * params.e_out_length + x_out] =
                solinas_mul_wide(e_out[x_out], sum);
        }
    }
}
