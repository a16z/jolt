extern "C" __global__ void ram_rw_cycle_round_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ val_coeff,
    const u64 *__restrict__ ra_coeff,
    const u64 *__restrict__ prev_val,
    const u64 *__restrict__ next_val,
    const int *__restrict__ even_idx,
    const int *__restrict__ odd_idx,
    const unsigned int *__restrict__ pair,
    const u64 *__restrict__ inc,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    const u64 *__restrict__ gamma,
    unsigned int in_pairs,
    unsigned long items
) {
    (void)out;
    (void)val_coeff;
    (void)ra_coeff;
    (void)prev_val;
    (void)next_val;
    (void)even_idx;
    (void)odd_idx;
    (void)pair;
    (void)inc;
    (void)e_in;
    (void)e_out;
    (void)gamma;
    (void)in_pairs;
    (void)items;
}
