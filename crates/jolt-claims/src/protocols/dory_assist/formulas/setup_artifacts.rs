//! Public verifier-setup artifact slots exposed to Dory-assist semantics.

use super::artifacts::{G1_ARTIFACT_COORDS, G2_ARTIFACT_COORDS, GT_ARTIFACT_COEFFS};

pub const DORY_SETUP_GT_FAMILIES: usize = 5;

pub const fn dory_setup_chi_start(round: usize) -> usize {
    round * GT_ARTIFACT_COEFFS
}

pub const fn dory_setup_delta_1l_start(max_rounds: usize, round: usize) -> usize {
    setup_gt_family_start(max_rounds, 1) + round * GT_ARTIFACT_COEFFS
}

pub const fn dory_setup_delta_1r_start(max_rounds: usize, round: usize) -> usize {
    setup_gt_family_start(max_rounds, 2) + round * GT_ARTIFACT_COEFFS
}

pub const fn dory_setup_delta_2l_start(max_rounds: usize, round: usize) -> usize {
    setup_gt_family_start(max_rounds, 3) + round * GT_ARTIFACT_COEFFS
}

pub const fn dory_setup_delta_2r_start(max_rounds: usize, round: usize) -> usize {
    setup_gt_family_start(max_rounds, 4) + round * GT_ARTIFACT_COEFFS
}

pub const fn dory_setup_g1_0_start(max_rounds: usize) -> usize {
    setup_gt_family_start(max_rounds, DORY_SETUP_GT_FAMILIES)
}

pub const fn dory_setup_g2_0_start(max_rounds: usize) -> usize {
    dory_setup_g1_0_start(max_rounds) + G1_ARTIFACT_COORDS
}

pub const fn dory_setup_h1_start(max_rounds: usize) -> usize {
    dory_setup_g2_0_start(max_rounds) + G2_ARTIFACT_COORDS
}

pub const fn dory_setup_h2_start(max_rounds: usize) -> usize {
    dory_setup_h1_start(max_rounds) + G1_ARTIFACT_COORDS
}

pub const fn dory_setup_ht_start(max_rounds: usize) -> usize {
    dory_setup_h2_start(max_rounds) + G2_ARTIFACT_COORDS
}

pub const fn dory_setup_artifact_count(max_rounds: usize) -> usize {
    dory_setup_ht_start(max_rounds) + GT_ARTIFACT_COEFFS
}

const fn setup_gt_family_start(max_rounds: usize, family: usize) -> usize {
    family * (max_rounds + 1) * GT_ARTIFACT_COEFFS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_artifact_layout_is_contiguous() {
        let max_rounds = 2;

        assert_eq!(dory_setup_chi_start(0), 0);
        assert_eq!(dory_setup_chi_start(2), 32);
        assert_eq!(dory_setup_delta_1l_start(max_rounds, 0), 48);
        assert_eq!(dory_setup_delta_1r_start(max_rounds, 0), 96);
        assert_eq!(dory_setup_delta_2l_start(max_rounds, 0), 144);
        assert_eq!(dory_setup_delta_2r_start(max_rounds, 0), 192);
        assert_eq!(dory_setup_g1_0_start(max_rounds), 240);
        assert_eq!(dory_setup_g2_0_start(max_rounds), 243);
        assert_eq!(dory_setup_h1_start(max_rounds), 248);
        assert_eq!(dory_setup_h2_start(max_rounds), 251);
        assert_eq!(dory_setup_ht_start(max_rounds), 256);
        assert_eq!(dory_setup_artifact_count(max_rounds), 272);
    }
}
