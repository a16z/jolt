//! Public transcript-scalar slots exposed to Dory-assist semantics.

pub const DORY_REDUCE_ROUND_TRANSCRIPT_SCALARS: usize = 8;
pub const DORY_FINAL_TRANSCRIPT_SCALARS: usize = 4;

pub const fn opening_point_coordinate(index: usize) -> usize {
    index
}

pub const fn dory_reduce_beta(point_len: usize, round: usize) -> usize {
    point_len + round * DORY_REDUCE_ROUND_TRANSCRIPT_SCALARS
}

pub const fn dory_reduce_beta_inverse(point_len: usize, round: usize) -> usize {
    dory_reduce_beta(point_len, round) + 1
}

pub const fn dory_reduce_alpha(point_len: usize, round: usize) -> usize {
    dory_reduce_beta(point_len, round) + 2
}

pub const fn dory_reduce_alpha_inverse(point_len: usize, round: usize) -> usize {
    dory_reduce_beta(point_len, round) + 3
}

pub const fn dory_reduce_alpha_beta(point_len: usize, round: usize) -> usize {
    dory_reduce_beta(point_len, round) + 4
}

pub const fn dory_reduce_alpha_inverse_beta_inverse(point_len: usize, round: usize) -> usize {
    dory_reduce_beta(point_len, round) + 5
}

pub const fn dory_reduce_s1_fold_factor(point_len: usize, round: usize) -> usize {
    dory_reduce_beta(point_len, round) + 6
}

pub const fn dory_reduce_s2_fold_factor(point_len: usize, round: usize) -> usize {
    dory_reduce_beta(point_len, round) + 7
}

pub const fn dory_gamma(point_len: usize, reduce_rounds: usize) -> usize {
    point_len + reduce_rounds * DORY_REDUCE_ROUND_TRANSCRIPT_SCALARS
}

pub const fn dory_gamma_inverse(point_len: usize, reduce_rounds: usize) -> usize {
    dory_gamma(point_len, reduce_rounds) + 1
}

pub const fn dory_scalar_product_sigma_c(point_len: usize, reduce_rounds: usize) -> usize {
    dory_gamma(point_len, reduce_rounds) + 2
}

pub const fn dory_final_d(point_len: usize, reduce_rounds: usize, has_sigma_c: bool) -> usize {
    dory_gamma(point_len, reduce_rounds) + 2 + has_sigma_c as usize
}

pub const fn dory_final_d_inverse(
    point_len: usize,
    reduce_rounds: usize,
    has_sigma_c: bool,
) -> usize {
    dory_final_d(point_len, reduce_rounds, has_sigma_c) + 1
}

pub const fn dory_final_d_squared(
    point_len: usize,
    reduce_rounds: usize,
    has_sigma_c: bool,
) -> usize {
    dory_final_d(point_len, reduce_rounds, has_sigma_c) + 2
}

pub const fn transcript_scalar_count(
    point_len: usize,
    reduce_rounds: usize,
    has_sigma_c: bool,
) -> usize {
    dory_final_d(point_len, reduce_rounds, has_sigma_c) + DORY_FINAL_TRANSCRIPT_SCALARS - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_scalar_indices_are_contiguous_after_opening_point() {
        assert_eq!(opening_point_coordinate(0), 0);
        assert_eq!(opening_point_coordinate(1), 1);
        assert_eq!(dory_reduce_beta(2, 0), 2);
        assert_eq!(dory_reduce_beta_inverse(2, 0), 3);
        assert_eq!(dory_reduce_alpha(2, 0), 4);
        assert_eq!(dory_reduce_alpha_inverse(2, 0), 5);
        assert_eq!(dory_reduce_alpha_beta(2, 0), 6);
        assert_eq!(dory_reduce_alpha_inverse_beta_inverse(2, 0), 7);
        assert_eq!(dory_reduce_s1_fold_factor(2, 0), 8);
        assert_eq!(dory_reduce_s2_fold_factor(2, 0), 9);
        assert_eq!(dory_reduce_beta(2, 1), 10);
        assert_eq!(dory_reduce_alpha(2, 1), 12);
        assert_eq!(dory_gamma(2, 2), 18);
        assert_eq!(dory_gamma_inverse(2, 2), 19);
        assert_eq!(dory_final_d(2, 2, false), 20);
        assert_eq!(dory_final_d_inverse(2, 2, false), 21);
        assert_eq!(dory_final_d_squared(2, 2, false), 22);
        assert_eq!(transcript_scalar_count(2, 2, false), 23);
    }

    #[test]
    fn zk_scalar_product_challenge_occupies_the_slot_before_final_d() {
        assert_eq!(dory_gamma(3, 1), 11);
        assert_eq!(dory_gamma_inverse(3, 1), 12);
        assert_eq!(dory_scalar_product_sigma_c(3, 1), 13);
        assert_eq!(dory_final_d(3, 1, true), 14);
        assert_eq!(dory_final_d_inverse(3, 1, true), 15);
        assert_eq!(dory_final_d_squared(3, 1, true), 16);
        assert_eq!(transcript_scalar_count(3, 1, true), 17);
    }
}
