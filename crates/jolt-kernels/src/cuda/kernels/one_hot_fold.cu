#define OHF_COLD 0xFFFFFFFFu
#define OHF_INSTRUCTION 0u
#define OHF_BYTECODE 1u
#define OHF_RAM 2u
#define OHF_SLOTS 4
#define OHF_LANES 8
#define OHF_MAX_LANES 2

__device__ __forceinline__ unsigned int ohf_chunk_u128(u64 lo, u64 hi, unsigned int shift,
                                                      unsigned int mask) {
    u64 value;
    if (shift >= 64) {
        value = hi >> (shift - 64);
    } else {
        value = lo >> shift;
        if (shift > 0) value |= hi << (64 - shift);
    }
    return (unsigned int)(value & (u64)mask);
}

__device__ __forceinline__ void ohf_family(unsigned int p, unsigned int instruction_polys,
                                           unsigned int bytecode_polys, unsigned int ram_polys,
                                           unsigned int chunk_bits, unsigned int *family,
                                           unsigned int *shift) {
    if (p < instruction_polys) {
        *family = OHF_INSTRUCTION;
        *shift = chunk_bits * (instruction_polys - 1 - p);
        return;
    }
    if (p < instruction_polys + bytecode_polys) {
        unsigned int local = p - instruction_polys;
        *family = OHF_BYTECODE;
        *shift = chunk_bits * (bytecode_polys - 1 - local);
        return;
    }
    unsigned int local = p - instruction_polys - bytecode_polys;
    *family = OHF_RAM;
    *shift = chunk_bits * (ram_polys - 1 - local);
}

__device__ __forceinline__ bool ohf_index(const u64 *__restrict__ lookup,
                                          const unsigned int *__restrict__ pc,
                                          const unsigned int *__restrict__ ram,
                                          unsigned int family, unsigned int shift,
                                          unsigned int mask, unsigned long long cycle,
                                          unsigned int *out) {
    if (family == OHF_INSTRUCTION) {
        *out = ohf_chunk_u128(lookup[2 * cycle], lookup[2 * cycle + 1], shift, mask);
        return true;
    }
    unsigned int word = (family == OHF_BYTECODE) ? pc[cycle] : ram[cycle];
    if (word == OHF_COLD) return false;
    *out = (word >> shift) & mask;
    return true;
}

__device__ __forceinline__ void ohf_atomic_add_field(u64 *lanes, const u64 *value) {
    for (int i = 0; i < LIMBS; i++) {
        unsigned long long piece = (unsigned long long)value[i];
        unsigned long long low = piece & 0xFFFFFFFFULL;
        unsigned long long high = piece >> 32;
        if (low != 0ULL) atomicAdd((unsigned long long *)&lanes[2 * i], low);
        if (high != 0ULL) atomicAdd((unsigned long long *)&lanes[2 * i + 1], high);
    }
}

__device__ __forceinline__ void ohf_finalize(const u64 *lanes, u64 *out) {
    u64 limbs[OHF_SLOTS + 2];
    u128 carry = 0;
    for (int i = 0; i < OHF_SLOTS; i++) {
        u128 t = (u128)lanes[2 * i] + ((u128)lanes[2 * i + 1] << 32) + carry;
        limbs[i] = (u64)t;
        carry = t >> 64;
    }
    limbs[OHF_SLOTS] = (u64)carry;
    limbs[OHF_SLOTS + 1] = (u64)(carry >> 64);

    u64 acc[LIMBS] = {0, 0, 0, 0};
    u64 scale[LIMBS];
    load4(UNR_MONT_2_64, scale);
    for (int i = OHF_SLOTS + 1; i >= 0; i--) {
        u64 scaled[LIMBS];
        fr_mul(acc, scale, scaled);
        u64 addend[LIMBS] = {limbs[i], 0, 0, 0};
        fr_add(scaled, addend, acc);
    }
    store4(out, acc);
}

__device__ __forceinline__ void ohf_weight(const u64 *__restrict__ e_in, unsigned int e_in_len,
                                           const u64 *__restrict__ e_out,
                                           unsigned int num_x_in_bits, unsigned long long g,
                                           u64 *combined) {
    u64 weight[LIMBS];
    if (e_in_len <= 1) {
        load4(FR_ONE, weight);
    } else {
        unsigned long long x_in = g & ((1ull << num_x_in_bits) - 1ull);
        load4(e_in + x_in * LIMBS, weight);
    }
    u64 e_out_eval[LIMBS];
    load4(e_out + (g >> num_x_in_bits) * LIMBS, e_out_eval);
    fr_mul(weight, e_out_eval, combined);
}

__device__ __forceinline__ void ohf_block_reduce(u64 *scratch, unsigned int lanes,
                                                u64 acc[OHF_MAX_LANES][LIMBS],
                                                u64 *__restrict__ partials) {
    unsigned int tid = threadIdx.x;
    for (unsigned int lane = 0; lane < lanes; lane++) {
        store4(scratch + tid * LIMBS, acc[lane]);
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                u64 a[LIMBS], b[LIMBS], s[LIMBS];
                load4(scratch + tid * LIMBS, a);
                load4(scratch + (tid + stride) * LIMBS, b);
                fr_add(a, b, s);
                store4(scratch + tid * LIMBS, s);
            }
            __syncthreads();
        }
        if (tid == 0) {
            u64 total[LIMBS];
            load4(scratch, total);
            store4(partials + ((unsigned long long)lane * gridDim.x + blockIdx.x) * LIMBS, total);
        }
        __syncthreads();
    }
}

extern "C" __global__ void ohf_fold_kernel(
    const u64 *__restrict__ lookup, const unsigned int *__restrict__ pc,
    const unsigned int *__restrict__ ram, unsigned int instruction_polys,
    unsigned int bytecode_polys, unsigned int ram_polys, unsigned int chunk_bits,
    unsigned int addresses, unsigned int cycles, const u64 *__restrict__ eq,
    u64 *__restrict__ slots, unsigned int polys_per_block, unsigned int use_shared) {
    extern __shared__ u64 histogram[];

    unsigned int polys = instruction_polys + bytecode_polys + ram_polys;
    unsigned int base = blockIdx.y * polys_per_block;
    unsigned int count = polys - base;
    if (count > polys_per_block) count = polys_per_block;
    unsigned int mask = (1u << chunk_bits) - 1u;
    unsigned int lanes = count * addresses * OHF_LANES;

    if (use_shared) {
        for (unsigned int i = threadIdx.x; i < lanes; i += blockDim.x) histogram[i] = 0ULL;
        __syncthreads();
    }

    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long j = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         j < cycles; j += stride) {
        u64 value[LIMBS];
        load4(eq + j * LIMBS, value);
        for (unsigned int local = 0; local < count; local++) {
            unsigned int family, shift;
            ohf_family(base + local, instruction_polys, bytecode_polys, ram_polys, chunk_bits,
                       &family, &shift);
            unsigned int a;
            if (!ohf_index(lookup, pc, ram, family, shift, mask, j, &a)) continue;
            if (use_shared) {
                ohf_atomic_add_field(
                    histogram + ((unsigned long long)local * addresses + a) * OHF_LANES, value);
            } else {
                ohf_atomic_add_field(
                    slots + ((unsigned long long)(base + local) * addresses + a) * OHF_LANES,
                    value);
            }
        }
    }

    if (!use_shared) return;
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < lanes; i += blockDim.x) {
        unsigned long long lane = (unsigned long long)histogram[i];
        if (lane == 0ULL) continue;
        unsigned int bucket = i / OHF_LANES;
        unsigned int within = i - bucket * OHF_LANES;
        unsigned int local = bucket / addresses;
        unsigned int a = bucket - local * addresses;
        atomicAdd((unsigned long long *)&slots[((unsigned long long)(base + local) * addresses + a) *
                                                  OHF_LANES +
                                              within],
                  lane);
    }
}

extern "C" __global__ void ohf_reduce_kernel(const u64 *__restrict__ slots,
                                            u64 *__restrict__ out, unsigned int buckets) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= buckets) return;
    ohf_finalize(slots + (unsigned long long)b * OHF_LANES, out + (unsigned long long)b * LIMBS);
}

extern "C" __global__ void ohf_affine_kernel(unsigned long long base, unsigned long long stride,
                                             u64 *__restrict__ out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 raw[LIMBS] = {base + stride * (unsigned long long)i, 0, 0, 0};
    u64 value[LIMBS];
    fr_to_mont(raw, value);
    store4(out + (unsigned long long)i * LIMBS, value);
}
