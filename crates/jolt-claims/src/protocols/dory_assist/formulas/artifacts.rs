//! Public artifact slots exposed by the Dory-assist checked-input relation.

pub const GT_ARTIFACT_COEFFS: usize = 16;
pub const G1_ARTIFACT_COORDS: usize = 3;
pub const G2_ARTIFACT_COORDS: usize = 5;

pub const DORY_PROOF_DIGEST_INDEX: usize = 0;
pub const DORY_VMV_C_START: usize = DORY_PROOF_DIGEST_INDEX + 1;
pub const DORY_VMV_D2_START: usize = DORY_VMV_C_START + GT_ARTIFACT_COEFFS;
pub const DORY_VMV_E1_START: usize = DORY_VMV_D2_START + GT_ARTIFACT_COEFFS;
pub const DORY_ZK_E2_START: usize = DORY_VMV_E1_START + G1_ARTIFACT_COORDS;
pub const DORY_ZK_Y_COM_START: usize = DORY_ZK_E2_START + G2_ARTIFACT_COORDS;
pub const DORY_SCALAR_PRODUCT_P1_START: usize = DORY_ZK_Y_COM_START + G1_ARTIFACT_COORDS;
pub const DORY_SCALAR_PRODUCT_P2_START: usize = DORY_SCALAR_PRODUCT_P1_START + GT_ARTIFACT_COEFFS;
pub const DORY_SCALAR_PRODUCT_Q_START: usize = DORY_SCALAR_PRODUCT_P2_START + GT_ARTIFACT_COEFFS;
pub const DORY_SCALAR_PRODUCT_R_START: usize = DORY_SCALAR_PRODUCT_Q_START + GT_ARTIFACT_COEFFS;
pub const DORY_SCALAR_PRODUCT_E1_START: usize = DORY_SCALAR_PRODUCT_R_START + GT_ARTIFACT_COEFFS;
pub const DORY_SCALAR_PRODUCT_E2_START: usize = DORY_SCALAR_PRODUCT_E1_START + G1_ARTIFACT_COORDS;
pub const DORY_SCALAR_PRODUCT_R1_INDEX: usize = DORY_SCALAR_PRODUCT_E2_START + G2_ARTIFACT_COORDS;
pub const DORY_SCALAR_PRODUCT_R2_INDEX: usize = DORY_SCALAR_PRODUCT_R1_INDEX + 1;
pub const DORY_SCALAR_PRODUCT_R3_INDEX: usize = DORY_SCALAR_PRODUCT_R2_INDEX + 1;
pub const DORY_REDUCE_ROUNDS_START: usize = DORY_SCALAR_PRODUCT_R3_INDEX + 1;

pub const FIRST_REDUCE_ARTIFACT_COORDS: usize =
    4 * GT_ARTIFACT_COEFFS + G1_ARTIFACT_COORDS + G2_ARTIFACT_COORDS;
pub const SECOND_REDUCE_ARTIFACT_COORDS: usize =
    2 * GT_ARTIFACT_COEFFS + 2 * G1_ARTIFACT_COORDS + 2 * G2_ARTIFACT_COORDS;
pub const REDUCE_ROUND_ARTIFACT_COORDS: usize =
    FIRST_REDUCE_ARTIFACT_COORDS + SECOND_REDUCE_ARTIFACT_COORDS;

pub const fn reduce_round_start(round: usize) -> usize {
    DORY_REDUCE_ROUNDS_START + round * REDUCE_ROUND_ARTIFACT_COORDS
}

pub const fn reduce_first_d1_left_start(round: usize) -> usize {
    reduce_round_start(round)
}

pub const fn reduce_first_d1_right_start(round: usize) -> usize {
    reduce_first_d1_left_start(round) + GT_ARTIFACT_COEFFS
}

pub const fn reduce_first_d2_left_start(round: usize) -> usize {
    reduce_first_d1_right_start(round) + GT_ARTIFACT_COEFFS
}

pub const fn reduce_first_d2_right_start(round: usize) -> usize {
    reduce_first_d2_left_start(round) + GT_ARTIFACT_COEFFS
}

pub const fn reduce_first_e1_beta_start(round: usize) -> usize {
    reduce_first_d2_right_start(round) + GT_ARTIFACT_COEFFS
}

pub const fn reduce_first_e2_beta_start(round: usize) -> usize {
    reduce_first_e1_beta_start(round) + G1_ARTIFACT_COORDS
}

pub const fn reduce_second_c_plus_start(round: usize) -> usize {
    reduce_first_e2_beta_start(round) + G2_ARTIFACT_COORDS
}

pub const fn reduce_second_c_minus_start(round: usize) -> usize {
    reduce_second_c_plus_start(round) + GT_ARTIFACT_COEFFS
}

pub const fn reduce_second_e1_plus_start(round: usize) -> usize {
    reduce_second_c_minus_start(round) + GT_ARTIFACT_COEFFS
}

pub const fn reduce_second_e1_minus_start(round: usize) -> usize {
    reduce_second_e1_plus_start(round) + G1_ARTIFACT_COORDS
}

pub const fn reduce_second_e2_plus_start(round: usize) -> usize {
    reduce_second_e1_minus_start(round) + G1_ARTIFACT_COORDS
}

pub const fn reduce_second_e2_minus_start(round: usize) -> usize {
    reduce_second_e2_plus_start(round) + G2_ARTIFACT_COORDS
}

pub const fn final_e1_start(reduce_rounds: usize) -> usize {
    DORY_REDUCE_ROUNDS_START + reduce_rounds * REDUCE_ROUND_ARTIFACT_COORDS
}

pub const fn final_e2_start(reduce_rounds: usize) -> usize {
    final_e1_start(reduce_rounds) + G1_ARTIFACT_COORDS
}
