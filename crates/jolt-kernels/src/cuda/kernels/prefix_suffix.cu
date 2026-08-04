#define PS_LANES 5

__device__ __forceinline__ void ps_to_mont(unsigned long long lo,
                                           unsigned long long hi,
                                           u64 *out) {
    u64 raw[LIMBS] = {lo, hi, 0, 0};
    fr_to_mont(raw, out);
}

extern "C" __global__ void ps_init_q_raf_kernel(const unsigned int *__restrict__ chunks,
                                                const unsigned long long *__restrict__ suffix_left,
                                                const unsigned long long *__restrict__ suffix_right,
                                                const unsigned long long *__restrict__ suffix_value,
                                                const unsigned char *__restrict__ raf_flags,
                                                const u64 *__restrict__ u_evals,
                                                u64 *__restrict__ buckets,
                                                unsigned int rows,
                                                unsigned int chunk_count) {
    extern __shared__ u64 scratch[];
    unsigned int bucket = blockIdx.x;
    unsigned int tid = threadIdx.x;

    u64 acc[PS_LANES][LIMBS];
    for (int lane = 0; lane < PS_LANES; lane++) {
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;
    }

    for (unsigned int r = tid; r < rows; r += blockDim.x) {
        if (chunks[r] != bucket) continue;
        u64 u[LIMBS];
        load4(u_evals + (unsigned long long)r * LIMBS, u);

        if (raf_flags[r]) {
            fr_add(acc[1], u, acc[1]);
            unsigned long long lo = suffix_value[2 * r];
            unsigned long long hi = suffix_value[2 * r + 1];
            if (lo != 0 || hi != 0) {
                u64 mont[LIMBS], term[LIMBS];
                ps_to_mont(lo, hi, mont);
                fr_mul(u, mont, term);
                fr_add(acc[4], term, acc[4]);
            }
        } else {
            fr_add(acc[0], u, acc[0]);
            if (suffix_left[r] != 0) {
                u64 mont[LIMBS], term[LIMBS];
                ps_to_mont(suffix_left[r], 0, mont);
                fr_mul(u, mont, term);
                fr_add(acc[2], term, acc[2]);
            }
            if (suffix_right[r] != 0) {
                u64 mont[LIMBS], term[LIMBS];
                ps_to_mont(suffix_right[r], 0, mont);
                fr_mul(u, mont, term);
                fr_add(acc[3], term, acc[3]);
            }
        }
    }

    for (int lane = 0; lane < PS_LANES; lane++) {
        store4(scratch + tid * LIMBS, acc[lane]);
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                u64 a[LIMBS], b[LIMBS], sum[LIMBS];
                load4(scratch + tid * LIMBS, a);
                load4(scratch + (tid + stride) * LIMBS, b);
                fr_add(a, b, sum);
                store4(scratch + tid * LIMBS, sum);
            }
            __syncthreads();
        }
        if (tid == 0) {
            u64 total[LIMBS];
            load4(scratch, total);
            store4(buckets + ((unsigned long long)lane * (unsigned long long)chunk_count +
                              (unsigned long long)bucket) *
                                 LIMBS,
                   total);
        }
        __syncthreads();
    }
}

extern "C" __global__ void ps_scale_shift_kernel(u64 *__restrict__ buckets,
                                                 const u64 *__restrict__ half_scale,
                                                 const u64 *__restrict__ full_scale,
                                                 unsigned int chunk_count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunk_count) return;
    u64 half[LIMBS], full[LIMBS], value[LIMBS], scaled[LIMBS];
    load4(half_scale, half);
    load4(full_scale, full);
    load4(buckets + (unsigned long long)i * LIMBS, value);
    fr_mul(value, half, scaled);
    store4(buckets + (unsigned long long)i * LIMBS, scaled);
    load4(buckets + ((unsigned long long)chunk_count + (unsigned long long)i) * LIMBS, value);
    fr_mul(value, full, scaled);
    store4(buckets + ((unsigned long long)chunk_count + (unsigned long long)i) * LIMBS, scaled);
}
