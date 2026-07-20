extern "C" __global__ void schedule_round_count(
    unsigned int *__restrict__ counts,
    const unsigned int *__restrict__ cur_rows,
    const unsigned int *__restrict__ cur_cols,
    unsigned long len
) {
    unsigned long k = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= len) return;

    unsigned int p = cur_rows[k] / 2;
    bool group_start = (k == 0) || (cur_rows[k - 1] / 2 != p);
    if (!group_start) {
        counts[k] = 0;
        return;
    }

    unsigned int even_row = 2 * p;
    unsigned int odd_row = even_row + 1;
    unsigned long mid = k;
    while (mid < len && cur_rows[mid] == even_row) mid++;
    unsigned long gend = mid;
    while (gend < len && cur_rows[gend] == odd_row) gend++;

    unsigned long i = k;
    unsigned long j = mid;
    unsigned int out = 0;
    while (i < mid || j < gend) {
        if (j >= gend || (i < mid && cur_cols[i] < cur_cols[j])) {
            i++;
        } else if (i >= mid || cur_cols[j] < cur_cols[i]) {
            j++;
        } else {
            i++;
            j++;
        }
        out++;
    }
    counts[k] = out;
}

extern "C" __global__ void schedule_round_emit(
    int *__restrict__ even_idx,
    int *__restrict__ odd_idx,
    unsigned int *__restrict__ pair_out,
    unsigned int *__restrict__ next_rows,
    unsigned int *__restrict__ next_cols,
    const unsigned int *__restrict__ offsets,
    const unsigned int *__restrict__ cur_rows,
    const unsigned int *__restrict__ cur_cols,
    unsigned long len
) {
    unsigned long k = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= len) return;

    unsigned int p = cur_rows[k] / 2;
    bool group_start = (k == 0) || (cur_rows[k - 1] / 2 != p);
    if (!group_start) return;

    unsigned int even_row = 2 * p;
    unsigned int odd_row = even_row + 1;
    unsigned long mid = k;
    while (mid < len && cur_rows[mid] == even_row) mid++;
    unsigned long gend = mid;
    while (gend < len && cur_rows[gend] == odd_row) gend++;

    unsigned long i = k;
    unsigned long j = mid;
    unsigned int o = offsets[k];
    while (i < mid || j < gend) {
        int ei;
        int oi;
        unsigned int col;
        if (j >= gend || (i < mid && cur_cols[i] < cur_cols[j])) {
            ei = (int)i;
            oi = -1;
            col = cur_cols[i];
            i++;
        } else if (i >= mid || cur_cols[j] < cur_cols[i]) {
            ei = -1;
            oi = (int)j;
            col = cur_cols[j];
            j++;
        } else {
            ei = (int)i;
            oi = (int)j;
            col = cur_cols[i];
            i++;
            j++;
        }
        even_idx[o] = ei;
        odd_idx[o] = oi;
        pair_out[o] = p;
        next_rows[o] = p;
        next_cols[o] = col;
        o++;
    }
}
