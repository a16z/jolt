#define REG_COLD 0xFFFFFFFFu
#define REG_SLOTS 3

#define REG_RA_ZERO 0u
#define REG_RA_GAMMA 1u
#define REG_RA_GAMMA_SQ 2u
#define REG_RA_BOTH 3u

__device__ __forceinline__ void reg_swap_slots(unsigned int *col, unsigned long long *value,
                                              unsigned long long *prev, unsigned long long *next,
                                              unsigned int *ra_kind, unsigned int *wa_flag,
                                              unsigned int a, unsigned int b) {
    unsigned int c = col[a];
    col[a] = col[b];
    col[b] = c;
    unsigned long long v = value[a];
    value[a] = value[b];
    value[b] = v;
    v = prev[a];
    prev[a] = prev[b];
    prev[b] = v;
    v = next[a];
    next[a] = next[b];
    next[b] = v;
    unsigned int k = ra_kind[a];
    ra_kind[a] = ra_kind[b];
    ra_kind[b] = k;
    k = wa_flag[a];
    wa_flag[a] = wa_flag[b];
    wa_flag[b] = k;
}

__device__ __forceinline__ unsigned int reg_build_slots(
    unsigned int rs1, unsigned long long rs1_value, unsigned int rs2,
    unsigned long long rs2_value, unsigned int rd, unsigned long long rd_pre,
    unsigned long long rd_post, unsigned int *col, unsigned long long *value,
    unsigned long long *prev, unsigned long long *next, unsigned int *ra_kind,
    unsigned int *wa_flag) {
    unsigned int len = 0;

    if (rs1 != REG_COLD) {
        col[len] = rs1;
        value[len] = rs1_value;
        prev[len] = rs1_value;
        next[len] = rs1_value;
        ra_kind[len] = REG_RA_GAMMA;
        wa_flag[len] = 0u;
        len++;
    }

    if (rs2 != REG_COLD) {
        int found = -1;
        for (unsigned int i = 0; i < len; i++) {
            if (col[i] == rs2) {
                found = (int)i;
                break;
            }
        }
        if (found >= 0) {
            ra_kind[found] = REG_RA_BOTH;
        } else {
            col[len] = rs2;
            value[len] = rs2_value;
            prev[len] = rs2_value;
            next[len] = rs2_value;
            ra_kind[len] = REG_RA_GAMMA_SQ;
            wa_flag[len] = 0u;
            len++;
        }
    }

    if (rd != REG_COLD) {
        int found = -1;
        for (unsigned int i = 0; i < len; i++) {
            if (col[i] == rd) {
                found = (int)i;
                break;
            }
        }
        if (found >= 0) {
            wa_flag[found] = 1u;
            next[found] = rd_post;
        } else {
            col[len] = rd;
            value[len] = rd_pre;
            prev[len] = rd_pre;
            next[len] = rd_post;
            ra_kind[len] = REG_RA_ZERO;
            wa_flag[len] = 1u;
            len++;
        }
    }

    if (len == 2) {
        if (col[0] > col[1]) reg_swap_slots(col, value, prev, next, ra_kind, wa_flag, 0, 1);
    } else if (len == 3) {
        if (col[0] > col[1]) reg_swap_slots(col, value, prev, next, ra_kind, wa_flag, 0, 1);
        if (col[1] > col[2]) reg_swap_slots(col, value, prev, next, ra_kind, wa_flag, 1, 2);
        if (col[0] > col[1]) reg_swap_slots(col, value, prev, next, ra_kind, wa_flag, 0, 1);
    }
    return len;
}

extern "C" __global__ void reg_count_kernel(
    const unsigned int *__restrict__ rs1_address, const unsigned int *__restrict__ rs2_address,
    const unsigned int *__restrict__ rd_address, unsigned int cycles,
    unsigned int *__restrict__ counts) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cycles) return;
    unsigned int col[REG_SLOTS], ra_kind[REG_SLOTS], wa_flag[REG_SLOTS];
    unsigned long long value[REG_SLOTS], prev[REG_SLOTS], next[REG_SLOTS];
    counts[j] = reg_build_slots(rs1_address[j], 0ULL, rs2_address[j], 0ULL, rd_address[j], 0ULL,
                               0ULL, col, value, prev, next, ra_kind, wa_flag);
}

extern "C" __global__ void reg_scatter_kernel(
    const unsigned int *__restrict__ rs1_address, const u64 *__restrict__ rs1_value,
    const unsigned int *__restrict__ rs2_address, const u64 *__restrict__ rs2_value,
    const unsigned int *__restrict__ rd_address, const u64 *__restrict__ rd_pre_value,
    const u64 *__restrict__ rd_post_value, const unsigned int *__restrict__ offsets,
    unsigned int cycles, unsigned int *__restrict__ out_rows,
    unsigned int *__restrict__ out_cols, u64 *__restrict__ out_val, u64 *__restrict__ out_prev,
    u64 *__restrict__ out_next, unsigned short *__restrict__ out_coeff_index) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cycles) return;

    unsigned int col[REG_SLOTS], ra_kind[REG_SLOTS], wa_flag[REG_SLOTS];
    unsigned long long value[REG_SLOTS], prev[REG_SLOTS], next[REG_SLOTS];
    unsigned int len =
        reg_build_slots(rs1_address[j], rs1_value[j], rs2_address[j], rs2_value[j],
                        rd_address[j], rd_pre_value[j], rd_post_value[j], col, value, prev,
                        next, ra_kind, wa_flag);
    if (len == 0) return;

    unsigned int k = offsets[j];
    for (unsigned int i = 0; i < len; i++) {
        out_rows[k + i] = j;
        out_cols[k + i] = col[i];
        out_prev[k + i] = prev[i];
        out_next[k + i] = next[i];

        u64 raw[LIMBS] = {value[i], 0, 0, 0};
        u64 val[LIMBS];
        fr_to_mont(raw, val);
        store4(out_val + (unsigned long long)(k + i) * LIMBS, val);

        out_coeff_index[(unsigned long long)(k + i) * 2 + 0] = (unsigned short)ra_kind[i];
        out_coeff_index[(unsigned long long)(k + i) * 2 + 1] = (unsigned short)wa_flag[i];
    }
}
