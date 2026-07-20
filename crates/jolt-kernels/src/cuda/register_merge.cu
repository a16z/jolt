extern "C" __global__ void register_merge_count(
    unsigned int *__restrict__ counts,
    const int *__restrict__ rs1_addr,
    const int *__restrict__ rs2_addr,
    const int *__restrict__ rd_addr,
    unsigned long n
) {
    unsigned long c = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    int a1 = rs1_addr[c];
    int a2 = rs2_addr[c];
    int a3 = rd_addr[c];

    unsigned int count = 0;
    if (a1 >= 0) count++;
    if (a2 >= 0 && a2 != a1) count++;
    if (a3 >= 0 && a3 != a1 && a3 != a2) count++;
    counts[c] = count;
}

extern "C" __global__ void register_merge_scatter(
    unsigned int *__restrict__ rows_out,
    unsigned int *__restrict__ cols_out,
    u64 *__restrict__ prev_out,
    u64 *__restrict__ next_out,
    u64 *__restrict__ rs1_flag_out,
    u64 *__restrict__ rs2_flag_out,
    u64 *__restrict__ rd_flag_out,
    const unsigned int *__restrict__ offsets,
    const int *__restrict__ rs1_addr,
    const u64 *__restrict__ rs1_val,
    const int *__restrict__ rs2_addr,
    const u64 *__restrict__ rs2_val,
    const int *__restrict__ rd_addr,
    const u64 *__restrict__ rd_pre,
    const u64 *__restrict__ rd_post,
    unsigned long n
) {
    unsigned long c = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    int col[3];
    u64 prev[3];
    u64 next[3];
    unsigned char f1[3];
    unsigned char f2[3];
    unsigned char f3[3];
    int m = 0;

    int a1 = rs1_addr[c];
    if (a1 >= 0) {
        col[m] = a1;
        prev[m] = rs1_val[c];
        next[m] = rs1_val[c];
        f1[m] = 1; f2[m] = 0; f3[m] = 0;
        m++;
    }

    int a2 = rs2_addr[c];
    if (a2 >= 0) {
        int hit = -1;
        for (int j = 0; j < m; j++) if (col[j] == a2) hit = j;
        if (hit >= 0) {
            f2[hit] = 1;
        } else {
            col[m] = a2;
            prev[m] = rs2_val[c];
            next[m] = rs2_val[c];
            f1[m] = 0; f2[m] = 1; f3[m] = 0;
            m++;
        }
    }

    int a3 = rd_addr[c];
    if (a3 >= 0) {
        int hit = -1;
        for (int j = 0; j < m; j++) if (col[j] == a3) hit = j;
        if (hit >= 0) {
            f3[hit] = 1;
            next[hit] = rd_post[c];
        } else {
            col[m] = a3;
            prev[m] = rd_pre[c];
            next[m] = rd_post[c];
            f1[m] = 0; f2[m] = 0; f3[m] = 1;
            m++;
        }
    }

    // Insertion sort the (at most 3) entries by column.
    for (int i = 1; i < m; i++) {
        int cc = col[i];
        u64 pp = prev[i], nn = next[i];
        unsigned char g1 = f1[i], g2 = f2[i], g3 = f3[i];
        int j = i - 1;
        while (j >= 0 && col[j] > cc) {
            col[j + 1] = col[j];
            prev[j + 1] = prev[j];
            next[j + 1] = next[j];
            f1[j + 1] = f1[j];
            f2[j + 1] = f2[j];
            f3[j + 1] = f3[j];
            j--;
        }
        col[j + 1] = cc;
        prev[j + 1] = pp;
        next[j + 1] = nn;
        f1[j + 1] = g1;
        f2[j + 1] = g2;
        f3[j + 1] = g3;
    }

    unsigned int base = offsets[c];
    for (int j = 0; j < m; j++) {
        unsigned int o = base + j;
        rows_out[o] = (unsigned int)c;
        cols_out[o] = (unsigned int)col[j];
        prev_out[o] = prev[j];
        next_out[o] = next[j];
        rs1_flag_out[o] = f1[j];
        rs2_flag_out[o] = f2[j];
        rd_flag_out[o] = f3[j];
    }
}
