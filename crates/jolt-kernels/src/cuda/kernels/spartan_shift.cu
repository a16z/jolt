extern "C" __global__ void ss_packed_columns_kernel(
    const unsigned int *__restrict__ pc_words,
    const unsigned int *__restrict__ flags, u64 *__restrict__ packed,
    unsigned int virtual_bit, unsigned int first_bit, unsigned int noop_bit,
    unsigned int flag_base, unsigned long long *__restrict__ unmapped,
    unsigned int cycles) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cycles) return;

    unsigned int word = pc_words[t];
    if (word == 0xFFFFFFFFu) {
        atomicMin(unmapped, (unsigned long long)t);
        return;
    }

    unsigned int mask = flags[t];
    u64 out = (u64)word;
    out |= (u64)((mask >> virtual_bit) & 1u) << flag_base;
    out |= (u64)((mask >> first_bit) & 1u) << (flag_base + 1u);
    out |= (u64)((mask >> noop_bit) & 1u) << (flag_base + 2u);
    packed[t] = out;
}

extern "C" __global__ void ss_fold_suffix_kernel(
    const u64 *__restrict__ unexpanded_pc, const u64 *__restrict__ packed,
    const u64 *__restrict__ weights, u64 *__restrict__ out0,
    u64 *__restrict__ out1, u64 *__restrict__ out2,
    u64 *__restrict__ out3, u64 *__restrict__ out4,
    unsigned int sum_len, unsigned int flag_base) {
    extern __shared__ u64 scratch[];
    unsigned int a = blockIdx.x, tid = threadIdx.x;
    unsigned long long base = (unsigned long long)a * sum_len;
    u64 acc[5][LIMBS] = {};
    unsigned long long b = tid;
    while (b < sum_len) {
        u64 folded[5][2 * UNR_SLOTS] = {};
        // One native product or field addition per slot per term; batching
        // at 4096 terms bounds each split accumulator lane below 2^44.
        for (unsigned int n = 0; n < 4096 && b < sum_len; n++, b += blockDim.x) {
            u64 weight[LIMBS];
            load4(weights + b * LIMBS, weight);
            u64 pc = unexpanded_pc[base + b];
            u64 row = packed[base + b];
            u64 expanded = row & 0xFFFFFFFFULL;
            unr_mul_words(weight, &pc, 1, folded[0]);
            unr_mul_words(weight, &expanded, 1, folded[1]);
            #pragma unroll
            for (int c = 0; c < 3; c++) {
                if ((row >> (flag_base + c)) & 1ULL) unr_add_field(folded[c + 2], weight);
            }
        }
        #pragma unroll
        for (int c = 0; c < 5; c++) {
            u64 partial[LIMBS], sum[LIMBS];
            unr_finalize(folded[c], partial);
            fr_add(acc[c], partial, sum);
            #pragma unroll
            for (int l = 0; l < LIMBS; l++) acc[c][l] = sum[l];
        }
    }
    u64 *outputs[5] = {out0, out1, out2, out3, out4};
    #pragma unroll
    for (int c = 0; c < 5; c++) {
        store4(scratch + tid * LIMBS, acc[c]);
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                u64 x[LIMBS], y[LIMBS], sum[LIMBS];
                load4(scratch + tid * LIMBS, x);
                load4(scratch + (tid + stride) * LIMBS, y);
                fr_add(x, y, sum);
                store4(scratch + tid * LIMBS, sum);
            }
            __syncthreads();
        }
        if (tid == 0) {
            u64 sum[LIMBS];
            load4(scratch, sum);
            store4(outputs[c] + (unsigned long long)a * LIMBS, sum);
        }
        __syncthreads();
    }
}
