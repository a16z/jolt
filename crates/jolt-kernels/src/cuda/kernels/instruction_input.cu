#define II_KIND_NARROW 0u
#define II_KIND_WIDE 1u
#define II_KIND_FLAG 2u

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
