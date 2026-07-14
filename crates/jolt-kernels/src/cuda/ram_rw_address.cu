extern "C" __global__ void ram_rw_address_round_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ ra_coeff,
    const u64 *__restrict__ val_coeff,
    const u64 *__restrict__ val_init,
    const int *__restrict__ even_idx,
    const int *__restrict__ odd_idx,
    const unsigned int *__restrict__ pair,
    const u64 *__restrict__ one_plus_gamma_in,
    const u64 *__restrict__ gc_in,
    unsigned long num_groups
) {
}
