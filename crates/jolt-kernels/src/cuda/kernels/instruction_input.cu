#define II_KIND_NARROW 0u
#define II_KIND_WIDE 1u
#define II_KIND_FLAG 2u

#define II_NARROW 3
#define II_WIDE 1
#define II_GATHER_BITS 4
#define II_EXTRA_WORDS 10
#define II_EXTRA_RS1 0
#define II_EXTRA_RS2 1
#define II_EXTRA_IMM_LO 5
#define II_EXTRA_IMM_HI 6

extern "C" __global__ void ii_gather_kernel(
    const u64 *__restrict__ extras, const u64 *__restrict__ address,
    const unsigned int *__restrict__ canonical, const unsigned int *__restrict__ bit_sources,
    unsigned int sign_base, u64 *__restrict__ narrow, u64 *__restrict__ wide,
    unsigned int *__restrict__ flags, unsigned int cycles) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cycles) return;

    const u64 *words = extras + (size_t)t * II_EXTRA_WORDS;
    u64 *row = narrow + (size_t)t * II_NARROW;
    row[0] = words[II_EXTRA_RS1];
    row[1] = address[t];
    row[2] = words[II_EXTRA_RS2];

    unsigned int mask = 0u;
    u128 imm = ((u128)words[II_EXTRA_IMM_HI] << 64) | (u128)words[II_EXTRA_IMM_LO];
    bool negative = ((__int128)imm) < 0;
    u128 magnitude = negative ? (~imm + (u128)1) : imm;
    u64 *limbs = wide + (size_t)t * (II_WIDE * 2);
    limbs[0] = (u64)magnitude;
    limbs[1] = (u64)(magnitude >> 64);
    if (negative) mask |= 1u << sign_base;

    unsigned int source = canonical[t];
    for (unsigned int bit = 0u; bit < II_GATHER_BITS; ++bit) {
        mask |= ((source >> bit_sources[bit]) & 1u) << bit;
    }
    flags[t] = mask;
}

extern "C" __global__ void ii_columns_kernel(const u64 *__restrict__ narrow,
                                            const u64 *__restrict__ wide,
                                            const unsigned int *__restrict__ flags,
                                            unsigned int narrow_width, unsigned int wide_width,
                                            unsigned int kind, unsigned int slot,
                                            unsigned int sign_base, unsigned int cycles,
                                            u64 *__restrict__ out) {
    unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cycles) return;

    unsigned int mask = flags[t];
    u64 value[LIMBS] = {0, 0, 0, 0};

    if (kind == II_KIND_FLAG) {
        if ((mask >> slot) & 1u) load4(FR_ONE, value);
    } else if (kind == II_KIND_NARROW) {
        u64 raw[LIMBS] = {narrow[t * narrow_width + slot], 0, 0, 0};
        fr_to_mont(raw, value);
    } else {
        const u64 *row = wide + t * (wide_width * 2);
        u64 raw[LIMBS] = {row[2 * slot], row[2 * slot + 1], 0, 0};
        u64 promoted[LIMBS];
        fr_to_mont(raw, promoted);
        if ((mask >> (sign_base + slot)) & 1u) {
            u64 zero[LIMBS] = {0, 0, 0, 0};
            fr_sub(zero, promoted, value);
        } else {
            for (int l = 0; l < LIMBS; l++) value[l] = promoted[l];
        }
    }

    store4(out + t * LIMBS, value);
}
