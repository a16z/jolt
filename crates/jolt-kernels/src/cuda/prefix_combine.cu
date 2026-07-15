__device__ __constant__ u64 PC_R2[4] = {
    0x1bb8e645ae216da7ULL, 0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL, 0x0216d0b17f4e44a5ULL
};

__device__ __forceinline__ void pc_u64_to_mont(u64 raw, u64 *out) {
    u64 limbs[4] = {raw, 0, 0, 0};
    fr_mul(limbs, PC_R2, out);
}

__device__ __forceinline__ void pc_madd(const u64 *a, const u64 *b, u64 *acc) {
    u64 prod[4], sum[4];
    fr_mul(a, b, prod);
    fr_add(acc, prod, sum);
    for (int k = 0; k < 4; k++) acc[k] = sum[k];
}

__device__ __forceinline__ void pc_msub(const u64 *a, const u64 *b, u64 *acc) {
    u64 prod[4], diff[4];
    fr_mul(a, b, prod);
    fr_sub(acc, prod, diff);
    for (int k = 0; k < 4; k++) acc[k] = diff[k];
}

__device__ __forceinline__ void pc_add(const u64 *a, u64 *acc) {
    u64 sum[4];
    fr_add(acc, a, sum);
    for (int k = 0; k < 4; k++) acc[k] = sum[k];
}

__device__ __forceinline__ void combine_eval(
    unsigned int variant,
    const u64 *__restrict__ prefixes,
    const u64 *__restrict__ suffixes,
    unsigned int suffix_count,
    u64 *out
) {
    for (int k = 0; k < 4; k++) out[k] = 0;
    const u64 *P0 = prefixes;
    const u64 *S0 = suffixes;
    #define PFX(i) (P0 + (i) * 4)
    #define SFX(i) (S0 + (i) * 4)

    switch (variant) {
        case 0: {
            pc_madd(PFX(0), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 1: {
            pc_madd(PFX(0), SFX(0), out); pc_add(SFX(1), out);
            pc_msub(PFX(19), SFX(2), out);
            break;
        }
        case 2: {
            pc_madd(PFX(4), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 3: {
            pc_madd(PFX(5), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 4: {
            pc_madd(PFX(6), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 5: {
            pc_madd(PFX(7), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 6: {
            pc_madd(PFX(3), SFX(0), out);
            break;
        }
        case 7: {
            pc_add(SFX(0), out);
            pc_madd(PFX(12), SFX(0), out);
            pc_msub(PFX(11), SFX(0), out);
            pc_msub(PFX(8), SFX(0), out);
            pc_msub(PFX(3), SFX(1), out);
            break;
        }
        case 8: {
            pc_add(SFX(0), out);
            pc_msub(PFX(8), SFX(0), out);
            pc_msub(PFX(3), SFX(1), out);
            break;
        }
        case 9: {
            pc_add(SFX(0), out);
            pc_msub(PFX(3), SFX(1), out);
            break;
        }
        case 10: {
            pc_madd(PFX(11), SFX(0), out);
            pc_msub(PFX(12), SFX(0), out);
            pc_madd(PFX(8), SFX(0), out);
            pc_madd(PFX(3), SFX(1), out);
            break;
        }
        case 11: {
            pc_madd(PFX(8), SFX(0), out);
            pc_madd(PFX(3), SFX(1), out);
            break;
        }
        case 12: {
            u64 ones[4], t[4];
            pc_u64_to_mont(0xFFFFFFFFFFFFFFFFULL, ones);
            fr_mul(ones, PFX(11), t);
            fr_mul(t, SFX(0), out);
            break;
        }
        case 13: {
            pc_madd(PFX(2), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 14: {
            pc_madd(PFX(8), SFX(0), out);
            pc_madd(PFX(3), SFX(1), out);
            pc_madd(PFX(3), SFX(2), out);
            break;
        }
        case 15: {
            pc_madd(PFX(10), SFX(2), out);
            pc_madd(PFX(8), SFX(0), out);
            pc_madd(PFX(3), SFX(1), out);
            break;
        }
        case 16: {
            pc_add(SFX(0), out);
            pc_msub(PFX(9), SFX(1), out);
            pc_madd(PFX(13), SFX(2), out);
            break;
        }
        case 17: {
            pc_add(SFX(0), out);
            pc_msub(PFX(19), SFX(1), out);
            break;
        }
        case 18: {
            pc_madd(PFX(27), SFX(0), out);
            break;
        }
        case 19: {
            pc_madd(PFX(1), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 20: {
            pc_madd(PFX(1), SFX(0), out); pc_add(SFX(1), out);
            pc_madd(PFX(28), SFX(2), out);
            break;
        }
        case 21: {
            pc_madd(PFX(20), SFX(0), out);
            break;
        }
        case 22: {
            pc_madd(PFX(21), SFX(0), out);
            break;
        }
        case 23: {
            u64 two64[4], t[4];
            u64 raw[4] = {0, 1, 0, 0};
            fr_mul(raw, PC_R2, two64);
            fr_mul(two64, SFX(0), t);
            for (int k = 0; k < 4; k++) out[k] = t[k];
            pc_msub(PFX(20), SFX(1), out);
            break;
        }
        case 24: {
            pc_madd(PFX(22), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 25: {
            pc_madd(PFX(23), SFX(1), out); pc_add(SFX(0), out);
            break;
        }
        case 26: {
            pc_madd(PFX(23), SFX(2), out);
            pc_add(SFX(1), out);
            pc_madd(PFX(11), SFX(3), out);
            pc_madd(PFX(24), SFX(0), out);
            break;
        }
        case 27: {
            pc_madd(PFX(23), SFX(0), out);
            pc_add(SFX(1), out);
            pc_madd(PFX(26), SFX(2), out);
            pc_madd(PFX(25), SFX(3), out);
            break;
        }
        case 28: {
            pc_madd(PFX(34), SFX(0), out);
            pc_add(SFX(1), out);
            pc_madd(PFX(35), SFX(2), out);
            pc_madd(PFX(36), SFX(3), out);
            break;
        }
        case 29: {
            pc_madd(PFX(31), SFX(0), out);
            pc_add(SFX(1), out);
            pc_madd(PFX(29), SFX(2), out);
            break;
        }
        case 30: {
            pc_madd(PFX(32), SFX(0), out);
            pc_add(SFX(1), out);
            pc_madd(PFX(30), SFX(2), out);
            pc_madd(PFX(33), SFX(3), out);
            break;
        }
        case 31: {
            pc_madd(PFX(37), SFX(0), out);
            break;
        }
        case 32: {
            pc_madd(PFX(40), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 33: {
            pc_madd(PFX(39), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 34: {
            pc_madd(PFX(38), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 35: {
            pc_madd(PFX(41), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 36: {
            pc_madd(PFX(45), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 37: {
            pc_madd(PFX(44), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 38: {
            pc_madd(PFX(43), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        case 39: {
            pc_madd(PFX(42), SFX(0), out); pc_add(SFX(1), out);
            break;
        }
        default: break;
    }
    #undef PFX
    #undef SFX
}

extern "C" __global__ void prefix_combine_probe(
    u64 *__restrict__ out,
    const u64 *__restrict__ prefixes,
    const u64 *__restrict__ suffixes,
    const unsigned int *__restrict__ suffix_count,
    const unsigned int *__restrict__ variant,
    unsigned long n
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    combine_eval(
        variant[i],
        prefixes + i * 46 * 4,
        suffixes + i * 4 * 4,
        suffix_count[i],
        out + i * 4
    );
}
