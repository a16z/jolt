extern "C" __global__ void read_table_round_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ prefix_polys,
    const u64 *__restrict__ suffix_blob,
    const unsigned int *__restrict__ table_variant,
    const unsigned int *__restrict__ table_suffix_offset,
    const unsigned int *__restrict__ table_suffix_count,
    unsigned long len,
    unsigned long half,
    unsigned long items
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * (3 * 4);
    for (int k = 0; k < 3 * 4; k++) acc[k] = 0;

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < items) {
        unsigned int t = (unsigned int)(i / half);
        unsigned long row = i % half;
        unsigned int variant = table_variant[t];
        unsigned long suffix_offset = table_suffix_offset[t];
        unsigned int suffix_count = table_suffix_count[t];

        u64 sleft[4 * 4], sright[4 * 4];
        for (unsigned int s = 0; s < suffix_count; s++) {
            const u64 *poly = suffix_blob + (suffix_offset + s) * len * 4;
            load4(poly + row * 4, sleft + s * 4);
            load4(poly + (row + half) * 4, sright + s * 4);
        }

        u64 prefixes[46 * 4];
        for (int p = 0; p < 46; p++) {
            load4(prefix_polys + ((unsigned long)p * len + row) * 4, prefixes + p * 4);
        }
        u64 lane[4];
        combine_eval(variant, prefixes, sleft, suffix_count, lane);
        for (int k = 0; k < 4; k++) acc[0 * 4 + k] = lane[k];

        for (int p = 0; p < 46; p++) {
            u64 low[4], high[4], t2[4];
            load4(prefix_polys + ((unsigned long)p * len + row) * 4, low);
            load4(prefix_polys + ((unsigned long)p * len + row + half) * 4, high);
            fr_add(high, high, t2);
            fr_sub(t2, low, prefixes + p * 4);
        }
        combine_eval(variant, prefixes, sleft, suffix_count, lane);
        for (int k = 0; k < 4; k++) acc[1 * 4 + k] = lane[k];
        combine_eval(variant, prefixes, sright, suffix_count, lane);
        for (int k = 0; k < 4; k++) acc[2 * 4 + k] = lane[k];
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * (3 * 4);
            for (int e = 0; e < 3; e++) {
                u64 s[4];
                fr_add(acc + e * 4, other + e * 4, s);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = s[k];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 3 * 4; k++) out[blockIdx.x * (3 * 4) + k] = acc[k];
    }
}
