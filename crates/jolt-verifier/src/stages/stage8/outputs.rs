#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineOpeningId;
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::Field;
use jolt_openings::{BatchOpeningStatement, VerifierOpeningClaim};
use jolt_poly::{Point, HIGH_TO_LOW};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage8OpeningId {
    Jolt(JoltOpeningId),
    #[cfg(feature = "field-inline")]
    FieldInline(FieldInlineOpeningId),
}

impl From<JoltOpeningId> for Stage8OpeningId {
    fn from(id: JoltOpeningId) -> Self {
        Self::Jolt(id)
    }
}

#[cfg(feature = "field-inline")]
impl From<FieldInlineOpeningId> for Stage8OpeningId {
    fn from(id: FieldInlineOpeningId) -> Self {
        Self::FieldInline(id)
    }
}

pub type Stage8OpeningStatement<F, C, Claim> =
    BatchOpeningStatement<F, C, Stage8OpeningId, Stage8OpeningId, Claim>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage8ClaimMode {
    Clear,
    Committed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8LogicalOpening<F: Field> {
    pub id: Stage8OpeningId,
    pub point: Vec<F>,
    pub claim: Option<F>,
    pub scale: F,
}

impl<F: Field> Stage8LogicalOpening<F> {
    pub fn claim_mode(&self) -> Stage8ClaimMode {
        if self.claim.is_some() {
            Stage8ClaimMode::Clear
        } else {
            Stage8ClaimMode::Committed
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8LogicalManifest<F: Field> {
    pub openings: Vec<Stage8LogicalOpening<F>>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
}

impl<F: Field> Stage8LogicalManifest<F> {
    pub fn opening_ids(&self) -> Vec<Stage8OpeningId> {
        self.openings.iter().map(|opening| opening.id).collect()
    }
}

#[derive(Clone, Debug)]
pub struct Stage8ClearBatchStatement<F: Field, C> {
    pub logical_manifest: Stage8LogicalManifest<F>,
    pub opening_ids: Vec<Stage8OpeningId>,
    pub opening_claims: Vec<VerifierOpeningClaim<F, C>>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub statement: Stage8OpeningStatement<F, C, F>,
}

#[derive(Clone, Debug)]
pub struct Stage8ZkBatchStatement<F: Field, C> {
    pub logical_manifest: Stage8LogicalManifest<F>,
    pub opening_ids: Vec<Stage8OpeningId>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub statement: Stage8OpeningStatement<F, C, ()>,
}

#[derive(Clone, Debug)]
pub enum Stage8BatchStatement<F: Field, C> {
    Clear(Stage8ClearBatchStatement<F, C>),
    Zk(Stage8ZkBatchStatement<F, C>),
}

#[derive(Clone, Debug)]
pub struct Stage8ClearOutput<F: Field, C> {
    pub opening_claims: Vec<VerifierOpeningClaim<F, C>>,
    pub opening_ids: Vec<Stage8OpeningId>,
    pub constraint_coefficients: Vec<F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_claim: F,
    pub joint_commitment: C,
}

#[derive(Clone, Debug)]
pub struct Stage8ZkOutput<F: Field, C, H> {
    pub opening_ids: Vec<Stage8OpeningId>,
    pub constraint_coefficients: Vec<F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_commitment: C,
    pub hiding_evaluation_commitment: H,
}

#[derive(Clone, Debug)]
pub enum Stage8Output<F: Field, C, H> {
    Clear(Stage8ClearOutput<F, C>),
    Zk(Stage8ZkOutput<F, C, H>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn logical_manifest_reports_ids_and_claim_modes() {
        let id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::IncClaimReduction,
        ));
        let manifest = Stage8LogicalManifest {
            openings: vec![
                Stage8LogicalOpening {
                    id,
                    point: vec![Fr::from_u64(1)],
                    claim: Some(Fr::from_u64(2)),
                    scale: Fr::from_u64(3),
                },
                Stage8LogicalOpening {
                    id,
                    point: vec![Fr::from_u64(4)],
                    claim: None,
                    scale: Fr::from_u64(5),
                },
            ],
            pcs_opening_point: Point::high_to_low(vec![Fr::from_u64(6)]),
        };

        assert_eq!(manifest.opening_ids(), vec![id, id]);
        assert_eq!(manifest.openings[0].claim_mode(), Stage8ClaimMode::Clear);
        assert_eq!(
            manifest.openings[1].claim_mode(),
            Stage8ClaimMode::Committed
        );
    }
}
