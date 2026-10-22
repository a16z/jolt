use jolt_algebra::Ring;

use super::{BytecodeReadRafCycleShape, BytecodeReadRafInputClaims, BytecodeReadRafOutputClaims};
use crate::protocols::jolt::geometry::bytecode::{
    bytecode_read_raf_address_phase_opening, read_raf_cycle_output,
};
use crate::protocols::jolt::{
    BytecodeReadRafChallenge, JoltChallengeId, JoltDerivedId, JoltExpr, JoltFactoredExpr,
    JoltOpeningId, JoltRelationId,
};
use crate::{opening, SumcheckChallenges, SymbolicSumcheck};

#[derive(Clone, Copy, Debug, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct BytecodeReadRafCyclePhaseChallenges<F> {
    #[challenge(BytecodeReadRafChallenge::Gamma)]
    pub gamma: F,
}

/// The cycle phase of the bytecode read-RAF sumcheck: starts from the staged
/// address-phase opening and reduces to the bytecode-table cycle output.
#[derive(Clone)]
pub struct ReadRafCyclePhase {
    shape: BytecodeReadRafCycleShape,
}

impl SymbolicSumcheck for ReadRafCyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafCycleShape;
    type Challenges<F> = BytecodeReadRafCyclePhaseChallenges<F>;
    type Inputs<C> = BytecodeReadRafInputClaims<C>;
    type Outputs<C> = BytecodeReadRafOutputClaims<C>;

    fn new(shape: BytecodeReadRafCycleShape) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeReadRaf
    }

    fn rounds(&self) -> usize {
        self.shape.0.log_t()
    }

    fn degree(&self) -> usize {
        self.shape.0.num_committed_ra_polys() + 1
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        self.output_factored().expand()
    }

    fn output_factored<F: Ring>(&self) -> JoltFactoredExpr<F> {
        read_raf_cycle_output(self.shape.0, self.shape.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::bytecode::{bytecode_ra, BytecodeReadRafDimensions};
    use crate::protocols::jolt::BytecodeReadRafPublic;
    use jolt_algebra::{Fr, Ring};

    /// The cycle output is `w · ∏_i ra_i` with
    /// `w = Σ_s γ^s val_s + γ^n raf_outer + γ^(n+1) raf_shift + γ^(n+2) entry`,
    /// stated as one term whose RA factors appear once.
    #[test]
    fn cycle_output_is_weight_sum_times_ra_product() {
        let num_val_stages = 5;
        let relation =
            ReadRafCyclePhase::new((BytecodeReadRafDimensions::new(10, 12, 3), num_val_stages));
        let factored = relation.output_factored::<Fr>();
        assert_eq!(factored.terms.len(), 1);
        assert_eq!(factored.terms[0].factors.len(), 4);

        let gamma = Fr::from_u64(3);
        let ra = [5, 7, 11].map(Fr::from_u64);
        let public = |id: &JoltDerivedId| match *id {
            JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::StageValue(s)) => {
                Fr::from_u64(13 + s as u64)
            }
            JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::SpartanOuterRaf) => {
                Fr::from_u64(19)
            }
            JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::SpartanShiftRaf) => {
                Fr::from_u64(23)
            }
            JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::Entry) => Fr::from_u64(29),
            _ => Fr::from_u64(0),
        };
        let output = relation.output_expression::<Fr>().evaluate(
            |id| {
                (0..3)
                    .find(|&i| *id == bytecode_ra(i))
                    .map_or(Fr::from_u64(0), |i| ra[i])
            },
            |_| gamma,
            public,
        );

        let mut weight = Fr::from_u64(0);
        let mut power = Fr::from_u64(1);
        for value in [13, 14, 15, 16, 17, 19, 23, 29] {
            weight += power * Fr::from_u64(value);
            power *= gamma;
        }
        assert_eq!(output, weight * ra[0] * ra[1] * ra[2]);
    }
}
