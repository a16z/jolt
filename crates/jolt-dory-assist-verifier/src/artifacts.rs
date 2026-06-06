//! Layout helpers for public Dory proof artifacts staged into `Fq`.

use std::ops::Range;

use jolt_dory::DoryProof;
use jolt_field::Fq;

pub use jolt_claims::protocols::dory_assist::formulas::artifacts::{
    DORY_PROOF_DIGEST_INDEX, DORY_REDUCE_ROUNDS_START, DORY_SCALAR_PRODUCT_E1_START,
    DORY_SCALAR_PRODUCT_E2_START, DORY_SCALAR_PRODUCT_P1_START, DORY_SCALAR_PRODUCT_P2_START,
    DORY_SCALAR_PRODUCT_Q_START, DORY_SCALAR_PRODUCT_R1_INDEX, DORY_SCALAR_PRODUCT_R2_INDEX,
    DORY_SCALAR_PRODUCT_R3_INDEX, DORY_SCALAR_PRODUCT_R_START, DORY_VMV_C_START, DORY_VMV_D2_START,
    DORY_VMV_E1_START, DORY_ZK_E2_START, DORY_ZK_Y_COM_START, FIRST_REDUCE_ARTIFACT_COORDS,
    G1_ARTIFACT_COORDS, G2_ARTIFACT_COORDS, GT_ARTIFACT_COEFFS, REDUCE_ROUND_ARTIFACT_COORDS,
    SECOND_REDUCE_ARTIFACT_COORDS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoryProofArtifactLayout {
    reduce_rounds: usize,
}

impl DoryProofArtifactLayout {
    pub const fn new(reduce_rounds: usize) -> Self {
        Self { reduce_rounds }
    }

    pub fn for_proof(proof: &DoryProof) -> Self {
        Self::new(proof.reduce_round_count())
    }

    pub const fn reduce_rounds(self) -> usize {
        self.reduce_rounds
    }

    pub const fn expected_len(self) -> usize {
        self.final_e2_start() + G2_ARTIFACT_COORDS
    }

    pub fn vmv_c(self) -> Range<usize> {
        gt_range(DORY_VMV_C_START)
    }

    pub fn vmv_d2(self) -> Range<usize> {
        gt_range(DORY_VMV_D2_START)
    }

    pub fn vmv_e1(self) -> Range<usize> {
        g1_range(DORY_VMV_E1_START)
    }

    pub fn zk_e2(self) -> Range<usize> {
        g2_range(DORY_ZK_E2_START)
    }

    pub fn zk_y_com(self) -> Range<usize> {
        g1_range(DORY_ZK_Y_COM_START)
    }

    pub fn scalar_product_p1(self) -> Range<usize> {
        gt_range(DORY_SCALAR_PRODUCT_P1_START)
    }

    pub fn scalar_product_p2(self) -> Range<usize> {
        gt_range(DORY_SCALAR_PRODUCT_P2_START)
    }

    pub fn scalar_product_q(self) -> Range<usize> {
        gt_range(DORY_SCALAR_PRODUCT_Q_START)
    }

    pub fn scalar_product_r(self) -> Range<usize> {
        gt_range(DORY_SCALAR_PRODUCT_R_START)
    }

    pub fn scalar_product_e1(self) -> Range<usize> {
        g1_range(DORY_SCALAR_PRODUCT_E1_START)
    }

    pub fn scalar_product_e2(self) -> Range<usize> {
        g2_range(DORY_SCALAR_PRODUCT_E2_START)
    }

    pub const fn scalar_product_r1(self) -> usize {
        let _ = self;
        DORY_SCALAR_PRODUCT_R1_INDEX
    }

    pub const fn scalar_product_r2(self) -> usize {
        let _ = self;
        DORY_SCALAR_PRODUCT_R2_INDEX
    }

    pub const fn scalar_product_r3(self) -> usize {
        let _ = self;
        DORY_SCALAR_PRODUCT_R3_INDEX
    }

    pub const fn reduce_round_start(self, round: usize) -> usize {
        DORY_REDUCE_ROUNDS_START + round * REDUCE_ROUND_ARTIFACT_COORDS
    }

    pub fn reduce_round(self, round: usize) -> DoryReduceRoundArtifactRanges {
        DoryReduceRoundArtifactRanges::new(self.reduce_round_start(round))
    }

    pub const fn final_e1_start(self) -> usize {
        DORY_REDUCE_ROUNDS_START + self.reduce_rounds * REDUCE_ROUND_ARTIFACT_COORDS
    }

    pub const fn final_e2_start(self) -> usize {
        self.final_e1_start() + G1_ARTIFACT_COORDS
    }

    pub fn final_e1(self) -> Range<usize> {
        g1_range(self.final_e1_start())
    }

    pub fn final_e2(self) -> Range<usize> {
        g2_range(self.final_e2_start())
    }

    pub fn gt_at(self, artifacts: &[Fq], range: Range<usize>) -> Option<[Fq; GT_ARTIFACT_COEFFS]> {
        let _ = self;
        copy_artifact(artifacts, range)
    }

    pub fn g1_at(self, artifacts: &[Fq], range: Range<usize>) -> Option<[Fq; G1_ARTIFACT_COORDS]> {
        let _ = self;
        copy_artifact(artifacts, range)
    }

    pub fn g2_at(self, artifacts: &[Fq], range: Range<usize>) -> Option<[Fq; G2_ARTIFACT_COORDS]> {
        let _ = self;
        copy_artifact(artifacts, range)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoryReduceRoundArtifactRanges {
    start: usize,
}

impl DoryReduceRoundArtifactRanges {
    pub const fn new(start: usize) -> Self {
        Self { start }
    }

    pub fn first_d1_left(self) -> Range<usize> {
        gt_range(self.start)
    }

    pub fn first_d1_right(self) -> Range<usize> {
        gt_range(self.first_d1_left().end)
    }

    pub fn first_d2_left(self) -> Range<usize> {
        gt_range(self.first_d1_right().end)
    }

    pub fn first_d2_right(self) -> Range<usize> {
        gt_range(self.first_d2_left().end)
    }

    pub fn first_e1_beta(self) -> Range<usize> {
        g1_range(self.first_d2_right().end)
    }

    pub fn first_e2_beta(self) -> Range<usize> {
        g2_range(self.first_e1_beta().end)
    }

    pub fn second_c_plus(self) -> Range<usize> {
        gt_range(self.first_e2_beta().end)
    }

    pub fn second_c_minus(self) -> Range<usize> {
        gt_range(self.second_c_plus().end)
    }

    pub fn second_e1_plus(self) -> Range<usize> {
        g1_range(self.second_c_minus().end)
    }

    pub fn second_e1_minus(self) -> Range<usize> {
        g1_range(self.second_e1_plus().end)
    }

    pub fn second_e2_plus(self) -> Range<usize> {
        g2_range(self.second_e1_minus().end)
    }

    pub fn second_e2_minus(self) -> Range<usize> {
        g2_range(self.second_e2_plus().end)
    }

    pub fn full(self) -> Range<usize> {
        self.start..self.second_e2_minus().end
    }
}

pub fn gt_range(start: usize) -> Range<usize> {
    start..start + GT_ARTIFACT_COEFFS
}

pub fn g1_range(start: usize) -> Range<usize> {
    start..start + G1_ARTIFACT_COORDS
}

pub fn g2_range(start: usize) -> Range<usize> {
    start..start + G2_ARTIFACT_COORDS
}

pub fn copy_artifact<const N: usize>(artifacts: &[Fq], range: Range<usize>) -> Option<[Fq; N]> {
    if range.len() != N {
        return None;
    }
    let slice = artifacts.get(range)?;
    let mut artifact = [Fq::default(); N];
    artifact.copy_from_slice(slice);
    Some(artifact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_artifact_ranges_are_contiguous() {
        let layout = DoryProofArtifactLayout::new(0);

        assert_eq!(layout.vmv_c(), 1..17);
        assert_eq!(layout.vmv_d2(), 17..33);
        assert_eq!(layout.vmv_e1(), 33..36);
        assert_eq!(layout.zk_e2(), 36..41);
        assert_eq!(layout.zk_y_com(), 41..44);
        assert_eq!(layout.scalar_product_p1(), 44..60);
        assert_eq!(layout.scalar_product_p2(), 60..76);
        assert_eq!(layout.scalar_product_q(), 76..92);
        assert_eq!(layout.scalar_product_r(), 92..108);
        assert_eq!(layout.scalar_product_e1(), 108..111);
        assert_eq!(layout.scalar_product_e2(), 111..116);
        assert_eq!(layout.scalar_product_r1(), 116);
        assert_eq!(layout.scalar_product_r2(), 117);
        assert_eq!(layout.scalar_product_r3(), 118);
        assert_eq!(DORY_REDUCE_ROUNDS_START, 119);
    }

    #[test]
    fn reduce_round_ranges_are_contiguous() {
        let round = DoryReduceRoundArtifactRanges::new(DORY_REDUCE_ROUNDS_START);

        assert_eq!(round.first_d1_left(), 119..135);
        assert_eq!(round.first_d1_right(), 135..151);
        assert_eq!(round.first_d2_left(), 151..167);
        assert_eq!(round.first_d2_right(), 167..183);
        assert_eq!(round.first_e1_beta(), 183..186);
        assert_eq!(round.first_e2_beta(), 186..191);
        assert_eq!(round.second_c_plus(), 191..207);
        assert_eq!(round.second_c_minus(), 207..223);
        assert_eq!(round.second_e1_plus(), 223..226);
        assert_eq!(round.second_e1_minus(), 226..229);
        assert_eq!(round.second_e2_plus(), 229..234);
        assert_eq!(round.second_e2_minus(), 234..239);
        assert_eq!(round.full().len(), REDUCE_ROUND_ARTIFACT_COORDS);
    }

    #[test]
    fn expected_len_accounts_for_reduce_rounds_and_final_pair() {
        assert_eq!(
            DoryProofArtifactLayout::new(0).expected_len(),
            DORY_REDUCE_ROUNDS_START + G1_ARTIFACT_COORDS + G2_ARTIFACT_COORDS
        );
        assert_eq!(
            DoryProofArtifactLayout::new(2).expected_len(),
            DORY_REDUCE_ROUNDS_START
                + 2 * REDUCE_ROUND_ARTIFACT_COORDS
                + G1_ARTIFACT_COORDS
                + G2_ARTIFACT_COORDS
        );
    }

    #[test]
    fn copy_artifact_rejects_wrong_width() {
        let artifacts = vec![Fq::default(); GT_ARTIFACT_COEFFS];

        assert!(
            copy_artifact::<{ GT_ARTIFACT_COEFFS }>(&artifacts, 0..GT_ARTIFACT_COEFFS).is_some()
        );
        assert!(
            copy_artifact::<{ G1_ARTIFACT_COORDS }>(&artifacts, 0..GT_ARTIFACT_COEFFS).is_none()
        );
    }
}
