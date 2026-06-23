use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use cudarc::driver::{
    result as cuda_result, CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use jolt_field::Fr;

const LIMBS: usize = 4;
const BLOCK: u32 = 256;

const KERNEL_SRC: &str = r#"
typedef unsigned long long u64;
typedef unsigned __int128 u128;

__device__ __constant__ u64 MODULUS[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
__device__ __constant__ u64 INV = 0xc2e1f593efffffffULL;

__device__ __forceinline__ u64 mac(u64 a, u64 b, u64 c, u64 *carry) {
    u128 t = (u128)a + (u128)b * (u128)c + (u128)(*carry);
    *carry = (u64)(t >> 64);
    return (u64)t;
}

__device__ __forceinline__ u64 adc(u64 a, u64 b, u64 *carry) {
    u128 t = (u128)a + (u128)b + (u128)(*carry);
    *carry = (u64)(t >> 64);
    return (u64)t;
}

__device__ __forceinline__ u64 sbb(u64 a, u64 b, u64 *borrow) {
    u128 t = ((u128)1 << 64) + (u128)a - (u128)b - (u128)(*borrow);
    *borrow = (t >> 64) == 0 ? 1 : 0;
    return (u64)t;
}

__device__ __forceinline__ int geq_modulus(const u64 *a) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] != MODULUS[i]) return a[i] > MODULUS[i];
    }
    return 1;
}

__device__ __forceinline__ void sub_modulus(u64 *a) {
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) a[i] = sbb(a[i], MODULUS[i], &borrow);
}

__device__ __forceinline__ void load4(const u64 *__restrict__ p, u64 *r) {
    ulonglong4 v = *reinterpret_cast<const ulonglong4 *>(p);
    r[0] = v.x; r[1] = v.y; r[2] = v.z; r[3] = v.w;
}

__device__ __forceinline__ void store4(u64 *p, const u64 *r) {
    ulonglong4 v;
    v.x = r[0]; v.y = r[1]; v.z = r[2]; v.w = r[3];
    *reinterpret_cast<ulonglong4 *>(p) = v;
}

__device__ void fr_add(const u64 *a, const u64 *b, u64 *out) {
    u64 carry = 0;
    for (int i = 0; i < 4; i++) out[i] = adc(a[i], b[i], &carry);
    if (carry != 0 || geq_modulus(out)) sub_modulus(out);
}

__device__ void fr_sub(const u64 *a, const u64 *b, u64 *out) {
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) out[i] = sbb(a[i], b[i], &borrow);
    if (borrow != 0) {
        u64 carry = 0;
        for (int i = 0; i < 4; i++) out[i] = adc(out[i], MODULUS[i], &carry);
    }
}

__device__ void fr_mul(const u64 *a, const u64 *b, u64 *out) {
    u64 t[6];
    for (int i = 0; i < 6; i++) t[i] = 0;

    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) t[j] = mac(t[j], a[j], b[i], &carry);
        u64 c = 0;
        t[4] = adc(t[4], carry, &c);
        t[5] = adc(t[5], 0, &c);

        u64 m = t[0] * INV;
        u64 c2 = 0;
        mac(t[0], m, MODULUS[0], &c2);
        for (int j = 1; j < 4; j++) t[j - 1] = mac(t[j], m, MODULUS[j], &c2);
        u64 c3 = 0;
        t[3] = adc(t[4], c2, &c3);
        t[4] = adc(t[5], 0, &c3);
        t[5] = 0;
    }

    for (int i = 0; i < 4; i++) out[i] = t[i];
    if (t[4] != 0 || geq_modulus(out)) sub_modulus(out);
}

extern "C" __global__ void add_kernel(u64 *__restrict__ io, const u64 *__restrict__ b, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u64 x[4], y[4];
        load4(io + i * 4, x);
        load4(b + i * 4, y);
        fr_add(x, y, x);
        store4(io + i * 4, x);
    }
}

extern "C" __global__ void sub_kernel(u64 *__restrict__ io, const u64 *__restrict__ b, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u64 x[4], y[4];
        load4(io + i * 4, x);
        load4(b + i * 4, y);
        fr_sub(x, y, x);
        store4(io + i * 4, x);
    }
}

extern "C" __global__ void mul_kernel(u64 *__restrict__ io, const u64 *__restrict__ b, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u64 x[4], y[4];
        load4(io + i * 4, x);
        load4(b + i * 4, y);
        fr_mul(x, y, x);
        store4(io + i * 4, x);
    }
}

extern "C" __global__ void fma_kernel(u64 *__restrict__ io, const u64 *__restrict__ b, const u64 *__restrict__ c, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u64 x[4], y[4], z[4];
        load4(io + i * 4, x);
        load4(b + i * 4, y);
        load4(c + i * 4, z);
        fr_mul(x, y, x);
        fr_add(x, z, x);
        store4(io + i * 4, x);
    }
}

extern "C" __global__ void bind_kernel(u64 *__restrict__ out, const u64 *__restrict__ values, const u64 *__restrict__ challenge, unsigned long half) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half) {
        u64 lo[4], hi[4], c[4];
        load4(values + (i * 2) * 4, lo);
        load4(values + (i * 2 + 1) * 4, hi);
        load4(challenge, c);
        u64 diff[4];
        fr_sub(hi, lo, diff);
        fr_mul(diff, c, diff);
        fr_add(lo, diff, lo);
        store4(out + i * 4, lo);
    }
}

extern "C" __global__ void eq_double(u64 *__restrict__ out, const u64 *__restrict__ in, const u64 *__restrict__ challenge, unsigned long size_in) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size_in) {
        u64 scalar[4], c[4], hi[4], lo[4];
        load4(in + i * 4, scalar);
        load4(challenge, c);
        fr_mul(scalar, c, hi);
        fr_sub(scalar, hi, lo);
        store4(out + (i * 2 + 1) * 4, hi);
        store4(out + (i * 2) * 4, lo);
    }
}

__device__ __forceinline__ void group_matvec(
    const u64 *__restrict__ dots,
    const u64 *__restrict__ weights,
    const unsigned int *__restrict__ rows,
    unsigned int group_len,
    u64 *acc
) {
    for (int k = 0; k < 4; k++) acc[k] = 0;
    for (unsigned int j = 0; j < group_len; j++) {
        u64 d[4], w[4], p[4];
        load4(dots + rows[j] * 4, d);
        load4(weights + j * 4, w);
        fr_mul(w, d, p);
        u64 t[4];
        fr_add(acc, p, t);
        for (int k = 0; k < 4; k++) acc[k] = t[k];
    }
}

__device__ __forceinline__ void csr_row_dot(
    const u64 *__restrict__ coeffs,
    const unsigned int *__restrict__ vars,
    unsigned int start,
    unsigned int end,
    const u64 *__restrict__ witness_row,
    u64 *acc
) {
    for (int k = 0; k < 4; k++) acc[k] = 0;
    for (unsigned int k = start; k < end; k++) {
        u64 coeff[4], w[4], p[4];
        load4(coeffs + k * 4, coeff);
        load4(witness_row + vars[k] * 4, w);
        fr_mul(coeff, w, p);
        u64 t[4];
        fr_add(acc, p, t);
        for (int i = 0; i < 4; i++) acc[i] = t[i];
    }
}

extern "C" __global__ void row_dots_kernel(
    u64 *__restrict__ a_out,
    u64 *__restrict__ b_out,
    const u64 *__restrict__ witness,
    const unsigned int *__restrict__ a_offsets,
    const unsigned int *__restrict__ a_vars,
    const u64 *__restrict__ a_coeffs,
    const unsigned int *__restrict__ b_offsets,
    const unsigned int *__restrict__ b_vars,
    const u64 *__restrict__ b_coeffs,
    unsigned long row_count,
    unsigned long num_vars_padded,
    unsigned long total
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        unsigned long cycle = i / row_count;
        unsigned long row = i % row_count;
        const u64 *witness_row = witness + cycle * num_vars_padded * 4;

        u64 acc[4];
        csr_row_dot(a_coeffs, a_vars, a_offsets[row], a_offsets[row + 1], witness_row, acc);
        store4(a_out + i * 4, acc);
        csr_row_dot(b_coeffs, b_vars, b_offsets[row], b_offsets[row + 1], witness_row, acc);
        store4(b_out + i * 4, acc);
    }
}

__device__ __forceinline__ void csr_group_dot(
    const u64 *__restrict__ coeffs,
    const unsigned int *__restrict__ vars,
    const unsigned int *__restrict__ offsets,
    unsigned int entry_start,
    unsigned int entry_end,
    const u64 *__restrict__ witness_row,
    u64 *acc
) {
    for (int k = 0; k < 4; k++) acc[k] = 0;
    for (unsigned int e = entry_start; e < entry_end; e++) {
        unsigned int start = offsets[e];
        unsigned int end = offsets[e + 1];
        for (unsigned int k = start; k < end; k++) {
            u64 coeff[4], w[4], p[4];
            load4(coeffs + k * 4, coeff);
            load4(witness_row + vars[k] * 4, w);
            fr_mul(coeff, w, p);
            u64 t[4];
            fr_add(acc, p, t);
            for (int i = 0; i < 4; i++) acc[i] = t[i];
        }
    }
}

extern "C" __global__ void dense_outer_fused_kernel(
    u64 *__restrict__ eq_out,
    u64 *__restrict__ az_out,
    u64 *__restrict__ bz_out,
    const u64 *__restrict__ eq_evals,
    const u64 *__restrict__ scale,
    const u64 *__restrict__ witness,
    const unsigned int *__restrict__ a_offsets,
    const unsigned int *__restrict__ a_vars,
    const u64 *__restrict__ a_coeffs,
    const unsigned int *__restrict__ b_offsets,
    const unsigned int *__restrict__ b_vars,
    const u64 *__restrict__ b_coeffs,
    unsigned int split,
    unsigned int total_entries,
    unsigned long num_vars_padded,
    unsigned long cycles
) {
    unsigned long cycle = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cycle < cycles) {
        unsigned long index = cycle * 2;
        const u64 *witness_row = witness + cycle * num_vars_padded * 4;

        u64 s[4];
        load4(scale, s);
        u64 e0[4], e1[4], r[4];
        load4(eq_evals + index * 4, e0);
        load4(eq_evals + (index + 1) * 4, e1);
        fr_mul(e0, s, r);
        store4(eq_out + index * 4, r);
        fr_mul(e1, s, r);
        store4(eq_out + (index + 1) * 4, r);

        u64 az0[4], az1[4], bz0[4], bz1[4];
        csr_group_dot(a_coeffs, a_vars, a_offsets, 0, split, witness_row, az0);
        csr_group_dot(a_coeffs, a_vars, a_offsets, split, total_entries, witness_row, az1);
        csr_group_dot(b_coeffs, b_vars, b_offsets, 0, split, witness_row, bz0);
        csr_group_dot(b_coeffs, b_vars, b_offsets, split, total_entries, witness_row, bz1);
        store4(az_out + index * 4, az0);
        store4(az_out + (index + 1) * 4, az1);
        store4(bz_out + index * 4, bz0);
        store4(bz_out + (index + 1) * 4, bz1);
    }
}

extern "C" __global__ void dense_outer_kernel(
    u64 *__restrict__ eq_out,
    u64 *__restrict__ az_out,
    u64 *__restrict__ bz_out,
    const u64 *__restrict__ eq_evals,
    const u64 *__restrict__ scale,
    const u64 *__restrict__ weights,
    const u64 *__restrict__ row_dots_a,
    const u64 *__restrict__ row_dots_b,
    const unsigned int *__restrict__ first_rows,
    const unsigned int *__restrict__ second_rows,
    unsigned int first_len,
    unsigned int second_len,
    unsigned long row_count,
    unsigned long cycles
) {
    unsigned long cycle = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cycle < cycles) {
        unsigned long index = cycle * 2;
        const u64 *a = row_dots_a + cycle * row_count * 4;
        const u64 *b = row_dots_b + cycle * row_count * 4;

        u64 s[4];
        load4(scale, s);
        u64 e0[4], e1[4], r[4];
        load4(eq_evals + index * 4, e0);
        load4(eq_evals + (index + 1) * 4, e1);
        fr_mul(e0, s, r);
        store4(eq_out + index * 4, r);
        fr_mul(e1, s, r);
        store4(eq_out + (index + 1) * 4, r);

        u64 az0[4], bz0[4], az1[4], bz1[4];
        group_matvec(a, weights, first_rows, first_len, az0);
        group_matvec(b, weights, first_rows, first_len, bz0);
        group_matvec(a, weights, second_rows, second_len, az1);
        group_matvec(b, weights, second_rows, second_len, bz1);
        store4(az_out + index * 4, az0);
        store4(bz_out + index * 4, bz0);
        store4(az_out + (index + 1) * 4, az1);
        store4(bz_out + (index + 1) * 4, bz1);
    }
}

__device__ __forceinline__ void cubic_coeffs(
    const u64 *eq0, const u64 *eq1,
    const u64 *az0, const u64 *az1,
    const u64 *bz0, const u64 *bz1,
    u64 *c
) {
    u64 eqd[4], azd[4], bzd[4];
    fr_sub(eq1, eq0, eqd);
    fr_sub(az1, az0, azd);
    fr_sub(bz1, bz0, bzd);

    u64 az0bz0[4], azdbz0[4], az0bzd[4], azdbzd[4];
    fr_mul(az0, bz0, az0bz0);
    fr_mul(azd, bz0, azdbz0);
    fr_mul(az0, bzd, az0bzd);
    fr_mul(azd, bzd, azdbzd);

    u64 t[4], s[4];

    fr_mul(eq0, az0bz0, c + 0);

    fr_mul(eqd, az0bz0, s);
    fr_mul(eq0, azdbz0, t);
    fr_add(s, t, s);
    fr_mul(eq0, az0bzd, t);
    fr_add(s, t, c + 4);

    fr_mul(eqd, azdbz0, s);
    fr_mul(eqd, az0bzd, t);
    fr_add(s, t, s);
    fr_mul(eq0, azdbzd, t);
    fr_add(s, t, c + 8);

    fr_mul(eqd, azdbzd, c + 12);
}

__device__ __forceinline__ void cubic_tuple_add(u64 *a, const u64 *b) {
    u64 t[4];
    fr_add(a + 0, b + 0, t);  for (int k = 0; k < 4; k++) a[k] = t[k];
    fr_add(a + 4, b + 4, t);  for (int k = 0; k < 4; k++) a[4 + k] = t[k];
    fr_add(a + 8, b + 8, t);  for (int k = 0; k < 4; k++) a[8 + k] = t[k];
    fr_add(a + 12, b + 12, t); for (int k = 0; k < 4; k++) a[12 + k] = t[k];
}

extern "C" __global__ void cubic_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ eq,
    const u64 *__restrict__ az,
    const u64 *__restrict__ bz,
    unsigned long pairs
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 16;

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) {
        cubic_coeffs(
            eq + (i * 2) * 4, eq + (i * 2 + 1) * 4,
            az + (i * 2) * 4, az + (i * 2 + 1) * 4,
            bz + (i * 2) * 4, bz + (i * 2 + 1) * 4,
            acc
        );
    } else {
        for (int k = 0; k < 16; k++) acc[k] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            cubic_tuple_add(acc, sdata + (threadIdx.x + s) * 16);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 16; k++) out[blockIdx.x * 16 + k] = acc[k];
    }
}

extern "C" __global__ void cubic_tuple_reduce(
    u64 *__restrict__ out,
    const u64 *__restrict__ in,
    unsigned long n
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 16;

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (int k = 0; k < 16; k++) acc[k] = in[i * 16 + k];
    } else {
        for (int k = 0; k < 16; k++) acc[k] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            cubic_tuple_add(acc, sdata + (threadIdx.x + s) * 16);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 16; k++) out[blockIdx.x * 16 + k] = acc[k];
    }
}

extern "C" __global__ void round_poly_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ factors,
    const u64 *__restrict__ points,
    const u64 *__restrict__ term_coeffs,
    const unsigned int *__restrict__ term_offsets,
    const unsigned int *__restrict__ term_indices,
    unsigned long pair_stride,
    unsigned int num_terms,
    unsigned int degree,
    unsigned long half
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * (degree * 4);
    for (unsigned int e = 0; e < degree; e++) {
        for (int k = 0; k < 4; k++) acc[e * 4 + k] = 0;
    }

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        for (unsigned int e = 0; e < degree; e++) {
            u64 x[4];
            load4(points + e * 4, x);
            u64 eval[4];
            for (int k = 0; k < 4; k++) eval[k] = 0;
            for (unsigned int t = 0; t < num_terms; t++) {
                u64 prod[4];
                load4(term_coeffs + t * 4, prod);
                unsigned int start = term_offsets[t];
                unsigned int end = term_offsets[t + 1];
                for (unsigned int s = start; s < end; s++) {
                    const u64 *factor = factors + term_indices[s] * pair_stride * 2 * 4;
                    u64 lo[4], hi[4];
                    load4(factor + (row * 2) * 4, lo);
                    load4(factor + (row * 2 + 1) * 4, hi);
                    u64 diff[4], linear[4];
                    fr_sub(hi, lo, diff);
                    fr_mul(diff, x, linear);
                    fr_add(lo, linear, linear);
                    fr_mul(prod, linear, prod);
                }
                fr_add(eval, prod, eval);
            }
            for (int k = 0; k < 4; k++) acc[e * 4 + k] = eval[k];
        }
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * (degree * 4);
            for (unsigned int e = 0; e < degree; e++) {
                u64 t[4];
                fr_add(acc + e * 4, other + e * 4, t);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = t[k];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (unsigned int k = 0; k < degree * 4; k++) {
            out[blockIdx.x * (degree * 4) + k] = acc[k];
        }
    }
}

extern "C" __global__ void round_poly_reduce(
    u64 *__restrict__ out,
    const u64 *__restrict__ in,
    unsigned int degree,
    unsigned long n
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * (degree * 4);

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (unsigned int k = 0; k < degree * 4; k++) acc[k] = in[i * (degree * 4) + k];
    } else {
        for (unsigned int k = 0; k < degree * 4; k++) acc[k] = 0;
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * (degree * 4);
            for (unsigned int e = 0; e < degree; e++) {
                u64 t[4];
                fr_add(acc + e * 4, other + e * 4, t);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = t[k];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (unsigned int k = 0; k < degree * 4; k++) {
            out[blockIdx.x * (degree * 4) + k] = acc[k];
        }
    }
}

extern "C" __global__ void sum_reduce(u64 *__restrict__ out, const u64 *__restrict__ in, unsigned long n) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 4;

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        load4(in + i * 4, acc);
    } else {
        for (int k = 0; k < 4; k++) acc[k] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            u64 *other = sdata + (threadIdx.x + s) * 4;
            u64 tmp[4];
            fr_add(acc, other, tmp);
            for (int k = 0; k < 4; k++) acc[k] = tmp[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        store4(out + blockIdx.x * 4, acc);
    }
}

extern "C" __global__ void product_reduce(u64 *__restrict__ out, const u64 *__restrict__ in, unsigned long n, const u64 *__restrict__ one) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 4;

    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        load4(in + i * 4, acc);
    } else {
        for (int k = 0; k < 4; k++) acc[k] = one[k];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            u64 *other = sdata + (threadIdx.x + s) * 4;
            u64 tmp[4];
            fr_mul(acc, other, tmp);
            for (int k = 0; k < 4; k++) acc[k] = tmp[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        store4(out + blockIdx.x * 4, acc);
    }
}
"#;

#[derive(Debug)]
pub enum CudaError {
    Compile(cudarc::nvrtc::CompileError),
    Driver(cudarc::driver::DriverError),
    Pool,
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::Compile(e) => write!(f, "nvrtc compile error: {e:?}"),
            CudaError::Driver(e) => write!(f, "cuda driver error: {e:?}"),
            CudaError::Pool => write!(f, "pinned staging pool invariant violated"),
        }
    }
}

impl std::error::Error for CudaError {}

impl From<cudarc::nvrtc::CompileError> for CudaError {
    fn from(e: cudarc::nvrtc::CompileError) -> Self {
        CudaError::Compile(e)
    }
}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        CudaError::Driver(e)
    }
}

pub struct CudaKernelContext {
    stream: Arc<CudaStream>,
    add: CudaFunction,
    sub: CudaFunction,
    mul: CudaFunction,
    fma: CudaFunction,
    bind: CudaFunction,
    eq_double: CudaFunction,
    dense_outer: CudaFunction,
    dense_outer_fused: CudaFunction,
    row_dots: CudaFunction,
    cubic_pairs: CudaFunction,
    cubic_tuple_reduce: CudaFunction,
    round_poly_pairs: CudaFunction,
    round_poly_reduce: CudaFunction,
    sum_reduce: CudaFunction,
    product_reduce: CudaFunction,
    one_dev: CudaSlice<u64>,
    staging: PinnedStaging,
}

pub struct DeviceFrVec {
    stream: Arc<CudaStream>,
    buf: CudaSlice<u64>,
    len: usize,
    staging: PinnedStaging,
}

impl DeviceFrVec {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn to_host(&self) -> Result<Vec<Fr>, CudaError> {
        if self.len == 0 {
            return Ok(Vec::new());
        }
        let n = self.len * LIMBS;
        let mut pool = lock_pool(&self.staging);
        let staging = pool.ensure(self.stream.context(), n)?;
        self.stream
            .memcpy_dtoh(&self.buf.slice(0..n), staging.as_mut_slice(n))?;
        self.stream.synchronize()?;
        Ok(unflatten(staging.as_slice(n)))
    }

    pub fn try_clone(&self) -> Result<Self, CudaError> {
        Ok(Self {
            stream: self.stream.clone(),
            buf: self.stream.clone_dtod(&self.buf)?,
            len: self.len,
            staging: self.staging.clone(),
        })
    }

    pub fn first(&self) -> Result<Fr, CudaError> {
        let raw = self.stream.clone_dtoh(&self.buf.slice(0..LIMBS))?;
        Ok(limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]))
    }
}

#[inline]
fn fr_to_limbs(f: Fr) -> [u64; LIMBS] {
    f.inner_limbs().0
}

#[inline]
fn limbs_to_fr(limbs: [u64; LIMBS]) -> Fr {
    Fr::from_bigint_unchecked(jolt_field::Limbs(limbs))
}

fn unflatten(raw: &[u64]) -> Vec<Fr> {
    raw.chunks_exact(LIMBS)
        .map(|c| limbs_to_fr([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct PinnedBuf {
    ctx: Arc<CudaContext>,
    ptr: *mut u64,
    cap: usize,
}

impl PinnedBuf {
    fn with_capacity(ctx: &Arc<CudaContext>, cap: usize) -> Result<Self, CudaError> {
        ctx.bind_to_thread()?;
        // SAFETY: malloc_host returns uninitialized page-locked memory; callers
        // only read limbs they have written or that a DMA has filled.
        let ptr = unsafe { cuda_result::malloc_host(cap * std::mem::size_of::<u64>(), 0)? };
        Ok(Self {
            ctx: ctx.clone(),
            ptr: ptr.cast::<u64>(),
            cap,
        })
    }

    fn as_slice(&self, len: usize) -> &[u64] {
        debug_assert!(len <= self.cap);
        // SAFETY: `ptr` points at `cap >= len` u64s of page-locked memory owned by self.
        unsafe { std::slice::from_raw_parts(self.ptr, len) }
    }

    fn as_mut_slice(&mut self, len: usize) -> &mut [u64] {
        debug_assert!(len <= self.cap);
        // SAFETY: `ptr` points at `cap >= len` u64s of page-locked memory owned by
        // self, and `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        // SAFETY: `ptr` was returned by malloc_host and is freed exactly once.
        self.ctx
            .record_err(unsafe { cuda_result::free_host(self.ptr.cast()) });
    }
}

// SAFETY: the allocation is owned solely by this `PinnedBuf` and only accessed
// behind a borrow (and, when shared, a Mutex), like cudarc's `PinnedHostSlice`.
unsafe impl Send for PinnedBuf {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for PinnedBuf {}

#[derive(Default)]
struct PinnedPool {
    buf: Option<PinnedBuf>,
}

impl PinnedPool {
    fn ensure(&mut self, ctx: &Arc<CudaContext>, len: usize) -> Result<&mut PinnedBuf, CudaError> {
        let grow = match &self.buf {
            Some(b) => b.cap < len,
            None => true,
        };
        if grow {
            self.buf = Some(PinnedBuf::with_capacity(ctx, len)?);
        }
        match self.buf.as_mut() {
            Some(b) => Ok(b),
            None => Err(CudaError::Pool),
        }
    }
}

type PinnedStaging = Arc<Mutex<PinnedPool>>;

fn lock_pool(pool: &PinnedStaging) -> MutexGuard<'_, PinnedPool> {
    pool.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Inputs to the fused dense-outer construction: builds `eq/az/bz` directly from
/// the witness, so the large per-cycle row-dots never cross PCIe. `a`/`b` are
/// weighted-CSR sparse matrices (the lagrange weights pre-folded into the coeffs)
/// over both row groups concatenated; `split` is the CSR-row index where the
/// second group begins.
pub struct FusedOuterInputs<'a> {
    pub eq_evals: &'a [Fr],
    pub scale: Fr,
    pub witness: &'a [Fr],
    pub a_offsets: &'a [u32],
    pub a_vars: &'a [u32],
    pub a_coeffs: &'a [Fr],
    pub b_offsets: &'a [u32],
    pub b_vars: &'a [u32],
    pub b_coeffs: &'a [Fr],
    pub split: usize,
    pub num_vars_padded: usize,
}

pub struct RoundPolyTerms<'a> {
    pub factors: &'a [&'a DeviceFrVec],
    pub term_coeffs: &'a [Fr],
    pub term_factor_offsets: &'a [u32],
    pub term_factor_indices: &'a [u32],
    pub degree: usize,
}

impl CudaKernelContext {
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let opts = CompileOptions {
            options: vec!["--device-int128".to_string()],
            ..Default::default()
        };
        let module = ctx.load_module(compile_ptx_with_opts(KERNEL_SRC, opts)?)?;
        let one_dev = stream.clone_htod(&fr_to_limbs(<Fr as num_traits::One>::one()))?;
        Ok(Self {
            add: module.load_function("add_kernel")?,
            sub: module.load_function("sub_kernel")?,
            mul: module.load_function("mul_kernel")?,
            fma: module.load_function("fma_kernel")?,
            bind: module.load_function("bind_kernel")?,
            eq_double: module.load_function("eq_double")?,
            dense_outer: module.load_function("dense_outer_kernel")?,
            dense_outer_fused: module.load_function("dense_outer_fused_kernel")?,
            row_dots: module.load_function("row_dots_kernel")?,
            cubic_pairs: module.load_function("cubic_pairs")?,
            cubic_tuple_reduce: module.load_function("cubic_tuple_reduce")?,
            round_poly_pairs: module.load_function("round_poly_pairs")?,
            round_poly_reduce: module.load_function("round_poly_reduce")?,
            sum_reduce: module.load_function("sum_reduce")?,
            product_reduce: module.load_function("product_reduce")?,
            stream,
            one_dev,
            staging: Arc::new(Mutex::new(PinnedPool::default())),
        })
    }

    pub fn upload(&self, values: &[Fr]) -> Result<DeviceFrVec, CudaError> {
        let buf = if values.is_empty() {
            self.stream.alloc_zeros(0)?
        } else {
            let n = values.len() * LIMBS;
            let mut pool = lock_pool(&self.staging);
            let staging = pool.ensure(self.stream.context(), n)?;
            for (slot, &v) in staging.as_mut_slice(n).chunks_exact_mut(LIMBS).zip(values) {
                slot.copy_from_slice(&fr_to_limbs(v));
            }
            let dev = self.stream.clone_htod(staging.as_slice(n))?;
            self.stream.synchronize()?;
            dev
        };
        Ok(DeviceFrVec {
            stream: self.stream.clone(),
            buf,
            len: values.len(),
            staging: self.staging.clone(),
        })
    }

    #[expect(clippy::too_many_arguments)]
    pub fn dense_outer_construct(
        &self,
        eq_evals: &[Fr],
        scale: Fr,
        weights: &[Fr],
        row_dots_a: &[Fr],
        row_dots_b: &[Fr],
        row_count: usize,
        first_group_rows: &[u32],
        second_group_rows: &[u32],
    ) -> Result<(DeviceFrVec, DeviceFrVec, DeviceFrVec), CudaError> {
        let len = eq_evals.len();
        let cycles = len / 2;

        let mut eq: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;
        let mut az: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;
        let mut bz: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;

        if cycles > 0 {
            let eq_evals = self.upload(eq_evals)?;
            let scale = self.upload(&[scale])?;
            let weights = self.upload(weights)?;
            let row_dots_a = self.upload(row_dots_a)?;
            let row_dots_b = self.upload(row_dots_b)?;
            let first = self.stream.clone_htod(first_group_rows)?;
            let second = self.stream.clone_htod(second_group_rows)?;

            let first_len = first_group_rows.len() as u32;
            let second_len = second_group_rows.len() as u32;
            let row_count_arg = row_count as u64;
            let cycles_arg = cycles as u64;
            let cfg = LaunchConfig {
                grid_dim: ((cycles as u32).div_ceil(BLOCK), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let f = self.dense_outer.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch
                .arg(&mut eq)
                .arg(&mut az)
                .arg(&mut bz)
                .arg(&eq_evals.buf)
                .arg(&scale.buf)
                .arg(&weights.buf)
                .arg(&row_dots_a.buf)
                .arg(&row_dots_b.buf)
                .arg(&first)
                .arg(&second)
                .arg(&first_len)
                .arg(&second_len)
                .arg(&row_count_arg)
                .arg(&cycles_arg);
            // SAFETY: one thread per cycle reads eq_evals[2c..2c+2], the scale, the
            // weights, and the per-cycle row_count dots from a/b, writing the two
            // output pairs at 2c/2c+1 in eq/az/bz. All device buffers are sized for
            // `len` field elements (cycles * row_count for the dot tables).
            let _ = unsafe { launch.launch(cfg) }?;
            self.stream.synchronize()?;
        }

        let make = |buf, len| DeviceFrVec {
            stream: self.stream.clone(),
            buf,
            len,
            staging: self.staging.clone(),
        };
        Ok((make(eq, len), make(az, len), make(bz, len)))
    }

    pub fn dense_outer_fused(
        &self,
        inputs: FusedOuterInputs<'_>,
    ) -> Result<(DeviceFrVec, DeviceFrVec, DeviceFrVec), CudaError> {
        let len = inputs.eq_evals.len();
        let cycles = len / 2;

        let mut eq: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;
        let mut az: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;
        let mut bz: CudaSlice<u64> = self.stream.alloc_zeros(len * LIMBS)?;

        if cycles > 0 {
            let eq_evals = self.upload(inputs.eq_evals)?;
            let scale = self.upload(&[inputs.scale])?;
            let witness = self.upload(inputs.witness)?;
            let a_coeffs = self.upload(inputs.a_coeffs)?;
            let b_coeffs = self.upload(inputs.b_coeffs)?;
            let a_offsets = self.stream.clone_htod(inputs.a_offsets)?;
            let a_vars = self.stream.clone_htod(inputs.a_vars)?;
            let b_offsets = self.stream.clone_htod(inputs.b_offsets)?;
            let b_vars = self.stream.clone_htod(inputs.b_vars)?;

            let split = inputs.split as u32;
            let total_entries = (inputs.a_offsets.len() - 1) as u32;
            let num_vars_padded_arg = inputs.num_vars_padded as u64;
            let cycles_arg = cycles as u64;
            let cfg = LaunchConfig {
                grid_dim: ((cycles as u32).div_ceil(BLOCK), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let f = self.dense_outer_fused.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch
                .arg(&mut eq)
                .arg(&mut az)
                .arg(&mut bz)
                .arg(&eq_evals.buf)
                .arg(&scale.buf)
                .arg(&witness.buf)
                .arg(&a_offsets)
                .arg(&a_vars)
                .arg(&a_coeffs.buf)
                .arg(&b_offsets)
                .arg(&b_vars)
                .arg(&b_coeffs.buf)
                .arg(&split)
                .arg(&total_entries)
                .arg(&num_vars_padded_arg)
                .arg(&cycles_arg);
            // SAFETY: one thread per cycle reads eq_evals[2c..2c+2], the scale, its
            // witness slice, and the two weighted-CSR groups (entries 0..split and
            // split..total), writing the output pairs at 2c/2c+1 in eq/az/bz.
            let _ = unsafe { launch.launch(cfg) }?;
            self.stream.synchronize()?;
        }

        let make = |buf, len| DeviceFrVec {
            stream: self.stream.clone(),
            buf,
            len,
            staging: self.staging.clone(),
        };
        Ok((make(eq, len), make(az, len), make(bz, len)))
    }

    #[expect(clippy::too_many_arguments)]
    pub fn compute_row_dots(
        &self,
        witness: &[Fr],
        a_offsets: &[u32],
        a_vars: &[u32],
        a_coeffs: &[Fr],
        b_offsets: &[u32],
        b_vars: &[u32],
        b_coeffs: &[Fr],
        row_count: usize,
        num_vars_padded: usize,
        num_cycles: usize,
    ) -> Result<(Vec<Fr>, Vec<Fr>), CudaError> {
        let total = num_cycles * row_count;
        if total == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let witness_dev = self.upload(witness)?;
        let a_coeffs_dev = self.upload(a_coeffs)?;
        let b_coeffs_dev = self.upload(b_coeffs)?;
        let a_offsets_dev = self.stream.clone_htod(a_offsets)?;
        let a_vars_dev = self.stream.clone_htod(a_vars)?;
        let b_offsets_dev = self.stream.clone_htod(b_offsets)?;
        let b_vars_dev = self.stream.clone_htod(b_vars)?;
        let mut a_out: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;
        let mut b_out: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;

        let row_count_arg = row_count as u64;
        let num_vars_padded_arg = num_vars_padded as u64;
        let total_arg = total as u64;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = self.row_dots.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut a_out)
            .arg(&mut b_out)
            .arg(&witness_dev.buf)
            .arg(&a_offsets_dev)
            .arg(&a_vars_dev)
            .arg(&a_coeffs_dev.buf)
            .arg(&b_offsets_dev)
            .arg(&b_vars_dev)
            .arg(&b_coeffs_dev.buf)
            .arg(&row_count_arg)
            .arg(&num_vars_padded_arg)
            .arg(&total_arg);
        // SAFETY: one thread per (cycle, row) of `total` reads its CSR row's
        // nonzeros (bounded by the offsets) from the coeff/var arrays and the
        // matching witness slice, writing one element to a_out/b_out (total each).
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;

        let a_raw = self.stream.clone_dtoh(&a_out)?;
        let b_raw = self.stream.clone_dtoh(&b_out)?;
        Ok((unflatten(&a_raw), unflatten(&b_raw)))
    }

    fn map(&self, func: &CudaFunction, a: &mut DeviceFrVec, b: &DeviceFrVec) -> Result<(), CudaError> {
        assert_eq!(a.len, b.len, "map operands must have equal length");
        let n = a.len;
        if n == 0 {
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_arg = n as u64;
        let mut launch = self.stream.launch_builder(func);
        let _ = launch.arg(&mut a.buf).arg(&b.buf).arg(&n_arg);
        // SAFETY: each thread i reads a[i] and b[i] and writes a[i] in place; the
        // elementwise access pattern means the in/out alias on `a` is hazard-free.
        // Both buffers hold n * LIMBS u64s.
        let _ = unsafe { launch.launch(cfg) }?;

        Ok(())
    }

    pub fn add(&self, a: &mut DeviceFrVec, b: &DeviceFrVec) -> Result<(), CudaError> {
        let f = self.add.clone();
        self.map(&f, a, b)
    }

    pub fn sub(&self, a: &mut DeviceFrVec, b: &DeviceFrVec) -> Result<(), CudaError> {
        let f = self.sub.clone();
        self.map(&f, a, b)
    }

    pub fn mul(&self, a: &mut DeviceFrVec, b: &DeviceFrVec) -> Result<(), CudaError> {
        let f = self.mul.clone();
        self.map(&f, a, b)
    }

    pub fn fma(
        &self,
        a: &mut DeviceFrVec,
        b: &DeviceFrVec,
        c: &DeviceFrVec,
    ) -> Result<(), CudaError> {
        assert_eq!(a.len, b.len, "fma operands must have equal length");
        assert_eq!(a.len, c.len, "fma operands must have equal length");
        let n = a.len;
        if n == 0 {
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_arg = n as u64;
        let f = self.fma.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch.arg(&mut a.buf).arg(&b.buf).arg(&c.buf).arg(&n_arg);
        // SAFETY: elementwise kernel; the in/out alias on `a` is hazard-free, and
        // all three buffers hold n * LIMBS u64s.
        let _ = unsafe { launch.launch(cfg) }?;

        Ok(())
    }

    pub fn bind(
        &self,
        values: &mut DeviceFrVec,
        scratch: &mut DeviceFrVec,
        challenge: Fr,
    ) -> Result<(), CudaError> {
        let half = values.len / 2;
        if scratch.buf.len() < half * LIMBS {
            scratch.buf = self.stream.alloc_zeros(half * LIMBS)?;
        }
        scratch.len = half;
        if half == 0 {
            std::mem::swap(values, scratch);
            return Ok(());
        }

        let challenge_dev = self.stream.clone_htod(&fr_to_limbs(challenge))?;
        let cfg = LaunchConfig {
            grid_dim: ((half as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let half_arg = half as u64;
        let f = self.bind.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut scratch.buf)
            .arg(&values.buf)
            .arg(&challenge_dev)
            .arg(&half_arg);
        // SAFETY: each thread i reads values[2i], values[2i+1] and the single
        // challenge, writing scratch[i]; scratch and values are distinct buffers
        // holding >= half and 2*half elements respectively.
        let _ = unsafe { launch.launch(cfg) }?;
        self.stream.synchronize()?;

        std::mem::swap(values, scratch);
        Ok(())
    }

    pub fn cubic_accumulate(
        &self,
        eq: &DeviceFrVec,
        az: &DeviceFrVec,
        bz: &DeviceFrVec,
    ) -> Result<[Fr; 4], CudaError> {
        use num_traits::Zero;
        assert_eq!(eq.len, az.len, "cubic operands must have equal length");
        assert_eq!(eq.len, bz.len, "cubic operands must have equal length");
        let pairs = eq.len / 2;
        if pairs == 0 {
            return Ok([Fr::zero(); 4]);
        }

        const TUPLE: usize = 4 * LIMBS;
        let shared = BLOCK * TUPLE as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (pairs as u32).div_ceil(BLOCK);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * TUPLE)?;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: shared,
        };
        let pairs_arg = pairs as u64;
        let f = self.cubic_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&eq.buf)
            .arg(&az.buf)
            .arg(&bz.buf)
            .arg(&pairs_arg);
        // SAFETY: each block reads up to BLOCK pairs from eq/az/bz (2*pairs elements
        // each) and writes one 4-coefficient tuple to `buf` (blocks tuples). Shared
        // memory holds BLOCK tuples as configured above.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(BLOCK);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * TUPLE)?;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: shared,
            };
            let len_arg = len as u64;
            let f = self.cubic_tuple_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&len_arg);
            // SAFETY: each block reads up to BLOCK tuples from `buf` (len tuples) and
            // writes one tuple to `out` (out_blocks tuples). Shared memory holds
            // BLOCK tuples as configured above.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..TUPLE))?;
        Ok(core::array::from_fn(|i| {
            limbs_to_fr([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]])
        }))
    }

    pub fn sum_of_products_round_poly(
        &self,
        terms: RoundPolyTerms<'_>,
    ) -> Result<Vec<Fr>, CudaError> {
        use jolt_field::Field;
        use num_traits::Zero;
        let degree = terms.degree;
        assert!(degree >= 1, "round poly degree must be >= 1");
        let pair_stride = terms.factors[0].len() / 2;
        for factor in terms.factors {
            assert_eq!(
                factor.len(),
                pair_stride * 2,
                "round poly factors must have equal length"
            );
        }
        if pair_stride == 0 {
            return Ok(vec![Fr::zero(); degree]);
        }

        let mut packed: CudaSlice<u64> =
            self.stream.alloc_zeros(terms.factors.len() * pair_stride * 2 * LIMBS)?;
        for (index, factor) in terms.factors.iter().enumerate() {
            let offset = index * pair_stride * 2 * LIMBS;
            self.stream.memcpy_dtod(
                &factor.buf,
                &mut packed.slice_mut(offset..offset + pair_stride * 2 * LIMBS),
            )?;
        }

        // Evaluation points {0, 2, 3, ..., degree}: index 0 is x=0, index e>=1 is x=e+1.
        let points: Vec<Fr> = (0..degree)
            .map(|e| Fr::from_u64(if e == 0 { 0 } else { (e + 1) as u64 }))
            .collect();
        let points_dev = self.upload(&points)?;
        let coeffs_dev = self.upload(terms.term_coeffs)?;
        let offsets_dev = self.stream.clone_htod(terms.term_factor_offsets)?;
        let indices_dev = self.stream.clone_htod(terms.term_factor_indices)?;

        let tuple = degree * LIMBS;
        // Cap the block so the per-thread tuple accumulators fit in 48 KB shared memory;
        // the tree reduce needs a power-of-two block, so take the largest such <= cap.
        let max_block = (48 * 1024 / (tuple * std::mem::size_of::<u64>())).max(1) as u32;
        let capped = BLOCK.min(max_block);
        let block = 1u32 << (u32::BITS - 1 - capped.leading_zeros());
        let shared = block * tuple as u32 * std::mem::size_of::<u64>() as u32;
        let blocks = (pair_stride as u32).div_ceil(block);
        let mut buf: CudaSlice<u64> = self.stream.alloc_zeros(blocks as usize * tuple)?;

        let pair_stride_arg = pair_stride as u64;
        let num_terms_arg = terms.term_coeffs.len() as u32;
        let degree_arg = degree as u32;
        let half_arg = pair_stride as u64;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let f = self.round_poly_pairs.clone();
        let mut launch = self.stream.launch_builder(&f);
        let _ = launch
            .arg(&mut buf)
            .arg(&packed)
            .arg(&points_dev.buf)
            .arg(&coeffs_dev.buf)
            .arg(&offsets_dev)
            .arg(&indices_dev)
            .arg(&pair_stride_arg)
            .arg(&num_terms_arg)
            .arg(&degree_arg)
            .arg(&half_arg);
        // SAFETY: one thread per row reads its pair from each packed factor and the
        // term tables (bounded by offsets), writing one `degree`-tuple per block;
        // shared memory holds `block` tuples as configured above.
        let _ = unsafe { launch.launch(cfg) }?;

        let mut len = blocks as usize;
        while len > 1 {
            let out_blocks = (len as u32).div_ceil(block);
            let mut out: CudaSlice<u64> = self.stream.alloc_zeros(out_blocks as usize * tuple)?;
            let len_arg = len as u64;
            let cfg = LaunchConfig {
                grid_dim: (out_blocks, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            };
            let f = self.round_poly_reduce.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch.arg(&mut out).arg(&buf).arg(&degree_arg).arg(&len_arg);
            // SAFETY: each block reads up to `block` tuples from `buf` (len total) and
            // writes one tuple to `out`; shared memory holds `block` tuples.
            let _ = unsafe { launch.launch(cfg) }?;
            buf = out;
            len = out_blocks as usize;
        }

        let raw = self.stream.clone_dtoh(&buf.slice(0..tuple))?;
        self.stream.synchronize()?;
        Ok((0..degree)
            .map(|e| limbs_to_fr([raw[e * 4], raw[e * 4 + 1], raw[e * 4 + 2], raw[e * 4 + 3]]))
            .collect())
    }

    pub fn eq_evals(&self, r: &[Fr], scaling_factor: Option<Fr>) -> Result<DeviceFrVec, CudaError> {
        use num_traits::One;
        let scaling = scaling_factor.unwrap_or_else(Fr::one);
        let total = 1usize << r.len();

        let mut cur: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;
        self.stream
            .memcpy_htod(&fr_to_limbs(scaling), &mut cur.slice_mut(0..LIMBS))?;
        let mut next: CudaSlice<u64> = self.stream.alloc_zeros(total * LIMBS)?;

        let mut size_in = 1usize;
        for &r_j in r {
            let challenge_dev = self.stream.clone_htod(&fr_to_limbs(r_j))?;
            let size_in_arg = size_in as u64;
            let cfg = LaunchConfig {
                grid_dim: ((size_in as u32).div_ceil(BLOCK), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let f = self.eq_double.clone();
            let mut launch = self.stream.launch_builder(&f);
            let _ = launch
                .arg(&mut next)
                .arg(&cur)
                .arg(&challenge_dev)
                .arg(&size_in_arg);
            // SAFETY: each thread i reads cur[i] (size_in elements) and the single
            // challenge, writing next[2i], next[2i+1]; cur and next are distinct
            // buffers holding >= 2*size_in elements after this round.
            let _ = unsafe { launch.launch(cfg) }?;
            std::mem::swap(&mut cur, &mut next);
            size_in *= 2;
        }
        self.stream.synchronize()?;

        Ok(DeviceFrVec {
            stream: self.stream.clone(),
            buf: cur,
            len: total,
            staging: self.staging.clone(),
        })
    }

    fn reduce_to_device(&self, sum: bool, values: &DeviceFrVec) -> Result<DeviceFrVec, CudaError> {
        use num_traits::{One, Zero};
        if values.len == 0 {
            return self.upload(&[if sum { Fr::zero() } else { Fr::one() }]);
        }
        if values.len == 1 {
            return values.try_clone();
        }

        let mut owned: Option<CudaSlice<u64>> = None;
        let mut len = values.len;

        while len > 1 {
            let input: &CudaSlice<u64> = owned.as_ref().unwrap_or(&values.buf);
            let blocks = (len as u32).div_ceil(BLOCK);
            let mut out_dev: CudaSlice<u64> =
                self.stream.alloc_zeros(blocks as usize * LIMBS)?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * std::mem::size_of::<u64>() as u32,
            };
            let len_arg = len as u64;
            let func = if sum {
                self.sum_reduce.clone()
            } else {
                self.product_reduce.clone()
            };
            let mut launch = self.stream.launch_builder(&func);
            let _ = launch.arg(&mut out_dev).arg(input).arg(&len_arg);
            if !sum {
                let _ = launch.arg(&self.one_dev);
            }
            // SAFETY: each block reads up to BLOCK elements from `input` (len total)
            // and writes one element per block to `out_dev` (blocks total). Shared
            // memory holds BLOCK field elements as configured above.
            let _ = unsafe { launch.launch(cfg) }?;

            owned = Some(out_dev);
            len = blocks as usize;
        }

        match owned {
            Some(buf) => Ok(DeviceFrVec {
                stream: self.stream.clone(),
                buf,
                len: 1,
                staging: self.staging.clone(),
            }),
            None => Err(CudaError::Pool),
        }
    }

    pub fn sum(&self, values: &DeviceFrVec) -> Result<Fr, CudaError> {
        let host = self.reduce_to_device(true, values)?.to_host()?;
        Ok(host[0])
    }

    pub fn product(&self, values: &DeviceFrVec) -> Result<Fr, CudaError> {
        let host = self.reduce_to_device(false, values)?.to_host()?;
        Ok(host[0])
    }

    pub fn sum_device(&self, values: &DeviceFrVec) -> Result<DeviceFrVec, CudaError> {
        self.reduce_to_device(true, values)
    }

    pub fn product_device(&self, values: &DeviceFrVec) -> Result<DeviceFrVec, CudaError> {
        self.reduce_to_device(false, values)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::Field;
    use num_traits::{One, Zero};
    use proptest::prelude::*;

    fn ctx() -> CudaKernelContext {
        CudaKernelContext::new(0).unwrap()
    }

    fn fr_strategy() -> impl Strategy<Value = Fr> {
        any::<[u8; 32]>().prop_map(|bytes| Fr::from_bytes(&bytes))
    }

    fn fr_vec_strategy(max: usize) -> impl Strategy<Value = Vec<Fr>> {
        prop::collection::vec(fr_strategy(), 0..max)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 16, .. ProptestConfig::default()
        })]

        #[test]
        fn add_matches_cpu(a in fr_vec_strategy(300)) {
            let b = a.iter().rev().copied().collect::<Vec<_>>();
            let expected: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();
            let c = ctx();
            let mut a_dev = c.upload(&a).unwrap();
            c.add(&mut a_dev, &c.upload(&b).unwrap()).unwrap();
            prop_assert_eq!(a_dev.to_host().unwrap(), expected);
        }

        #[test]
        fn sub_matches_cpu(a in fr_vec_strategy(300)) {
            let b = a.iter().rev().copied().collect::<Vec<_>>();
            let expected: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| *x - *y).collect();
            let c = ctx();
            let mut a_dev = c.upload(&a).unwrap();
            c.sub(&mut a_dev, &c.upload(&b).unwrap()).unwrap();
            prop_assert_eq!(a_dev.to_host().unwrap(), expected);
        }

        #[test]
        fn mul_matches_cpu(a in fr_vec_strategy(300)) {
            let b = a.iter().rev().copied().collect::<Vec<_>>();
            let expected: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| *x * *y).collect();
            let c = ctx();
            let mut a_dev = c.upload(&a).unwrap();
            c.mul(&mut a_dev, &c.upload(&b).unwrap()).unwrap();
            prop_assert_eq!(a_dev.to_host().unwrap(), expected);
        }

        #[test]
        fn fma_matches_cpu(a in fr_vec_strategy(300)) {
            let b = a.iter().rev().copied().collect::<Vec<_>>();
            let cc = a.iter().map(|x| *x + Fr::from_u64(1)).collect::<Vec<_>>();
            let expected: Vec<Fr> = a
                .iter()
                .zip(&b)
                .zip(&cc)
                .map(|((x, y), z)| *x * *y + *z)
                .collect();
            let c = ctx();
            let mut a_dev = c.upload(&a).unwrap();
            c.fma(&mut a_dev, &c.upload(&b).unwrap(), &c.upload(&cc).unwrap())
                .unwrap();
            prop_assert_eq!(a_dev.to_host().unwrap(), expected);
        }

        #[test]
        fn sum_matches_cpu(a in fr_vec_strategy(2000)) {
            let expected: Fr = a.iter().copied().sum();
            let c = ctx();
            let got = c.sum(&c.upload(&a).unwrap()).unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn product_matches_cpu(a in fr_vec_strategy(2000)) {
            let expected: Fr = a.iter().copied().product();
            let c = ctx();
            let got = c.product(&c.upload(&a).unwrap()).unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn sum_device_matches_cpu(a in fr_vec_strategy(2000)) {
            let expected: Fr = a.iter().copied().sum();
            let c = ctx();
            let got = c.sum_device(&c.upload(&a).unwrap()).unwrap();
            prop_assert_eq!(got.len(), 1);
            prop_assert_eq!(got.to_host().unwrap(), vec![expected]);
        }

        #[test]
        fn product_device_matches_cpu(a in fr_vec_strategy(2000)) {
            let expected: Fr = a.iter().copied().product();
            let c = ctx();
            let got = c.product_device(&c.upload(&a).unwrap()).unwrap();
            prop_assert_eq!(got.len(), 1);
            prop_assert_eq!(got.to_host().unwrap(), vec![expected]);
        }

        #[test]
        fn cubic_accumulate_matches_cpu(
            (eq, az, bz) in (1usize..600).prop_flat_map(|pairs| {
                let len = pairs * 2;
                (
                    prop::collection::vec(fr_strategy(), len),
                    prop::collection::vec(fr_strategy(), len),
                    prop::collection::vec(fr_strategy(), len),
                )
            })
        ) {
            let expected = cpu_cubic(&eq, &az, &bz);
            let c = ctx();
            let got = c
                .cubic_accumulate(
                    &c.upload(&eq).unwrap(),
                    &c.upload(&az).unwrap(),
                    &c.upload(&bz).unwrap(),
                )
                .unwrap();
            prop_assert_eq!(got, expected);
        }
    }

    fn cpu_cubic(eq: &[Fr], az: &[Fr], bz: &[Fr]) -> [Fr; 4] {
        let mut c = [Fr::zero(); 4];
        for ((eq_pair, az_pair), bz_pair) in eq
            .chunks_exact(2)
            .zip(az.chunks_exact(2))
            .zip(bz.chunks_exact(2))
        {
            let eq0 = eq_pair[0];
            let eqd = eq_pair[1] - eq_pair[0];
            let az0 = az_pair[0];
            let azd = az_pair[1] - az_pair[0];
            let bz0 = bz_pair[0];
            let bzd = bz_pair[1] - bz_pair[0];
            let az0bz0 = az0 * bz0;
            let azdbz0 = azd * bz0;
            let az0bzd = az0 * bzd;
            let azdbzd = azd * bzd;
            c[0] += eq0 * az0bz0;
            c[1] += eqd * az0bz0 + eq0 * azdbz0 + eq0 * az0bzd;
            c[2] += eqd * azdbz0 + eqd * az0bzd + eq0 * azdbzd;
            c[3] += eqd * azdbzd;
        }
        c
    }

    struct Csr {
        offsets: Vec<u32>,
        vars: Vec<u32>,
        coeffs: Vec<Fr>,
    }

    // `rows` lists each row's (var, coeff) nonzeros.
    fn build_csr(rows: &[Vec<(u32, Fr)>]) -> Csr {
        let mut offsets = vec![0u32];
        let mut vars = Vec::new();
        let mut coeffs = Vec::new();
        for row in rows {
            for &(var, coeff) in row {
                vars.push(var);
                coeffs.push(coeff);
            }
            offsets.push(vars.len() as u32);
        }
        Csr {
            offsets,
            vars,
            coeffs,
        }
    }

    fn cpu_row_dots(
        witness: &[Fr],
        csr: &Csr,
        row_count: usize,
        num_vars_padded: usize,
        num_cycles: usize,
    ) -> Vec<Fr> {
        let mut out = vec![Fr::zero(); num_cycles * row_count];
        for cycle in 0..num_cycles {
            let base = cycle * num_vars_padded;
            for row in 0..row_count {
                let start = csr.offsets[row] as usize;
                let end = csr.offsets[row + 1] as usize;
                let mut acc = Fr::zero();
                for k in start..end {
                    acc += csr.coeffs[k] * witness[base + csr.vars[k] as usize];
                }
                out[cycle * row_count + row] = acc;
            }
        }
        out
    }

    const ROW_DOTS_NUM_VARS: usize = 32;

    fn row_strategy() -> impl Strategy<Value = Vec<(u32, Fr)>> {
        prop::collection::vec(
            (0u32..ROW_DOTS_NUM_VARS as u32, fr_strategy()),
            0..6,
        )
    }

    proptest! {
        #[test]
        fn compute_row_dots_matches_cpu(
            log_cycles in 0usize..8,
            a_rows in prop::collection::vec(row_strategy(), 1..24),
            b_rows in prop::collection::vec(row_strategy(), 1..24),
            seed in fr_strategy(),
        ) {
            let row_count = a_rows.len().min(b_rows.len());
            let a = build_csr(&a_rows[..row_count]);
            let b = build_csr(&b_rows[..row_count]);

            let num_cycles = 1usize << log_cycles;
            let num_vars_padded = ROW_DOTS_NUM_VARS.next_power_of_two();
            let witness: Vec<Fr> = (0..num_cycles * num_vars_padded)
                .map(|i| seed + Fr::from_u64(i as u64))
                .collect();

            let expected_a = cpu_row_dots(&witness, &a, row_count, num_vars_padded, num_cycles);
            let expected_b = cpu_row_dots(&witness, &b, row_count, num_vars_padded, num_cycles);

            let c = ctx();
            let (got_a, got_b) = c
                .compute_row_dots(
                    &witness,
                    &a.offsets,
                    &a.vars,
                    &a.coeffs,
                    &b.offsets,
                    &b.vars,
                    &b.coeffs,
                    row_count,
                    num_vars_padded,
                    num_cycles,
                )
                .unwrap();
            prop_assert_eq!(got_a, expected_a);
            prop_assert_eq!(got_b, expected_b);
        }
    }

    fn flatten_terms(terms: &[(Fr, Vec<u32>)]) -> (Vec<Fr>, Vec<u32>, Vec<u32>) {
        let mut coeffs = Vec::new();
        let mut offsets = vec![0u32];
        let mut indices = Vec::new();
        for (coeff, factor_indices) in terms {
            coeffs.push(*coeff);
            indices.extend_from_slice(factor_indices);
            offsets.push(indices.len() as u32);
        }
        (coeffs, offsets, indices)
    }

    proptest! {
        #[test]
        fn sum_of_products_round_poly_matches_cpu(
            log_pairs in 0usize..10,
            num_factors in 1usize..6,
            terms_spec in prop::collection::vec(
                (fr_strategy(), prop::collection::vec(0u32..6, 1..4)),
                1..5,
            ),
            seed in fr_strategy(),
        ) {
            let half = 1usize << log_pairs;
            let len = half * 2;
            let terms_spec: Vec<(Fr, Vec<u32>)> = terms_spec
                .into_iter()
                .map(|(coeff, idxs)| {
                    (coeff, idxs.into_iter().map(|i| i % num_factors as u32).collect())
                })
                .collect();
            let degree = terms_spec.iter().map(|(_, idxs)| idxs.len()).max().unwrap();

            let factors: Vec<Vec<Fr>> = (0..num_factors)
                .map(|f| (0..len).map(|i| seed + Fr::from_u64((f * len + i) as u64)).collect())
                .collect();

            let cpu_terms: Vec<crate::stage6::DenseTerm<Fr>> = terms_spec
                .iter()
                .map(|(coeff, idxs)| crate::stage6::DenseTerm {
                    coefficient: *coeff,
                    factors: idxs.iter().map(|&i| i as usize).collect(),
                })
                .collect();
            let mut expected = vec![Fr::zero(); degree];
            let mut row_evals = vec![Fr::zero(); degree];
            for row in 0..half {
                for e in &mut row_evals {
                    *e = Fr::zero();
                }
                crate::stage6::accumulate_dense_row_evaluations(
                    &factors,
                    &cpu_terms,
                    row,
                    &mut row_evals,
                );
                for (acc, e) in expected.iter_mut().zip(&row_evals) {
                    *acc += *e;
                }
            }

            let (term_coeffs, term_factor_offsets, term_factor_indices) =
                flatten_terms(&terms_spec);
            let c = ctx();
            let factor_devs: Vec<DeviceFrVec> =
                factors.iter().map(|f| c.upload(f).unwrap()).collect();
            let factor_refs: Vec<&DeviceFrVec> = factor_devs.iter().collect();
            let got = c
                .sum_of_products_round_poly(RoundPolyTerms {
                    factors: &factor_refs,
                    term_coeffs: &term_coeffs,
                    term_factor_offsets: &term_factor_offsets,
                    term_factor_indices: &term_factor_indices,
                    degree,
                })
                .unwrap();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn eq_evals_matches_cpu(
            r in prop::collection::vec(fr_strategy(), 0..12),
            scaling in prop::option::of(fr_strategy()),
        ) {
            let expected = jolt_poly::EqPolynomial::<Fr>::evals(&r, scaling);
            let c = ctx();
            let got = c.eq_evals(&r, scaling).unwrap().to_host().unwrap();
            prop_assert_eq!(got, expected);
        }
    }

    #[test]
    fn edge_cases() {
        let c = ctx();
        let zero = Fr::zero();
        let one = Fr::one();

        let mut empty = c.upload(&[]).unwrap();
        c.add(&mut empty, &c.upload(&[]).unwrap()).unwrap();
        assert_eq!(empty.to_host().unwrap(), Vec::<Fr>::new());
        assert_eq!(c.sum(&empty).unwrap(), zero);
        assert_eq!(c.product(&empty).unwrap(), one);

        let single = c.upload(&[Fr::from_u64(42)]).unwrap();
        assert_eq!(c.sum(&single).unwrap(), Fr::from_u64(42));
        assert_eq!(c.product(&single).unwrap(), Fr::from_u64(42));

        let b = c.upload(&[one, one, Fr::from_u64(6)]).unwrap();

        let mut a = c.upload(&[zero, one, Fr::from_u64(7)]).unwrap();
        c.add(&mut a, &b).unwrap();
        assert_eq!(
            a.to_host().unwrap(),
            vec![one, Fr::from_u64(2), Fr::from_u64(13)]
        );

        let mut a = c.upload(&[zero, one, Fr::from_u64(7)]).unwrap();
        c.mul(&mut a, &b).unwrap();
        assert_eq!(
            a.to_host().unwrap(),
            vec![zero, one, Fr::from_u64(42)]
        );
    }

    #[test]
    fn try_clone_is_independent() {
        let c = ctx();
        let original = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let a = c.upload(&original).unwrap();
        let mut cloned = a.try_clone().unwrap();
        let ones = c.upload(&[Fr::one(); 3]).unwrap();
        c.add(&mut cloned, &ones).unwrap();

        assert_eq!(a.to_host().unwrap(), original);
        assert_eq!(
            cloned.to_host().unwrap(),
            vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(4)]
        );
    }

    #[test]
    fn reduce_preserves_input() {
        let c = ctx();
        // Larger than BLOCK so the multi-pass loop runs.
        let original: Vec<Fr> = (1..=1000u64).map(Fr::from_u64).collect();
        let a = c.upload(&original).unwrap();

        let expected_sum: Fr = original.iter().copied().sum();
        assert_eq!(c.sum(&a).unwrap(), expected_sum);
        assert_eq!(a.to_host().unwrap(), original);

        let expected_product: Fr = original.iter().copied().product();
        assert_eq!(c.product(&a).unwrap(), expected_product);
        assert_eq!(a.to_host().unwrap(), original);
    }
}
