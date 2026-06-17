#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineOpeningId;
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::Field;
use jolt_openings::{BatchOpeningStatement, PhysicalView, VerifierOpeningClaim};
use jolt_poly::{Point, HIGH_TO_LOW};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8PhysicalOpening<F: Field> {
    pub id: Stage8OpeningId,
    pub relation: Stage8OpeningId,
    pub view: PhysicalView<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8PhysicalManifest<F: Field> {
    pub openings: Vec<Stage8PhysicalOpening<F>>,
    pub layout_digest: [u8; 32],
}

impl<F: Field> Stage8PhysicalManifest<F> {
    pub fn direct(logical: &Stage8LogicalManifest<F>, layout_digest: [u8; 32]) -> Self {
        Self {
            openings: logical
                .openings
                .iter()
                .map(|opening| Stage8PhysicalOpening {
                    id: opening.id,
                    relation: opening.id,
                    view: PhysicalView::Direct,
                })
                .collect(),
            layout_digest,
        }
    }

    #[cfg(feature = "akita")]
    pub fn from_packed_view_catalog(
        logical: &Stage8LogicalManifest<F>,
        layout: &jolt_akita::PackedWitnessLayout,
        catalog: &jolt_akita::PackedViewCatalog<Stage8OpeningId, Stage8OpeningId, F>,
    ) -> Result<Self, jolt_akita::PackedViewError> {
        let openings = logical
            .openings
            .iter()
            .map(|opening| {
                let formula = catalog.lookup(&opening.id, &opening.id)?;
                Ok(Stage8PhysicalOpening {
                    id: opening.id,
                    relation: opening.id,
                    view: formula.physical_view_at(layout, &opening.point)?,
                })
            })
            .collect::<Result<Vec<_>, jolt_akita::PackedViewError>>()?;

        Ok(Self {
            openings,
            layout_digest: layout.digest,
        })
    }

    #[cfg(feature = "akita")]
    pub fn from_jolt_lattice_view_formulas(
        logical: &Stage8LogicalManifest<F>,
        layout: &jolt_akita::PackedWitnessLayout,
        formulas: impl IntoIterator<Item = super::JoltLatticeViewFormulaWithRowPoint<F>>,
    ) -> Result<Self, jolt_akita::PackedViewError> {
        let entries = formulas
            .into_iter()
            .map(|(id, formula, row_point)| {
                let id = Stage8OpeningId::from(id);
                Ok((
                    id,
                    id,
                    super::lattice::akita_packed_view_formula(&formula)?,
                    row_point,
                ))
            })
            .collect::<Result<Vec<_>, jolt_akita::PackedViewError>>()?;

        let openings = logical
            .openings
            .iter()
            .map(|opening| {
                let (_, relation, formula, row_point) = entries
                    .iter()
                    .find(|(id, relation, _, _)| *id == opening.id && *relation == opening.id)
                    .ok_or(jolt_akita::PackedViewError::MissingView)?;
                Ok(Stage8PhysicalOpening {
                    id: opening.id,
                    relation: *relation,
                    view: formula.physical_view_at(layout, row_point)?,
                })
            })
            .collect::<Result<Vec<_>, jolt_akita::PackedViewError>>()?;

        Ok(Self {
            openings,
            layout_digest: layout.digest,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Stage8ClearBatchStatement<F: Field, C> {
    pub logical_manifest: Stage8LogicalManifest<F>,
    pub physical_manifest: Stage8PhysicalManifest<F>,
    pub opening_ids: Vec<Stage8OpeningId>,
    pub opening_claims: Vec<VerifierOpeningClaim<F, C>>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub statement: Stage8OpeningStatement<F, C, F>,
}

#[derive(Clone, Debug)]
pub struct Stage8ZkBatchStatement<F: Field, C> {
    pub logical_manifest: Stage8LogicalManifest<F>,
    pub physical_manifest: Stage8PhysicalManifest<F>,
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
    #![cfg_attr(
        feature = "akita",
        expect(clippy::panic, reason = "tests fail loudly on setup errors")
    )]

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

    #[test]
    fn physical_manifest_direct_resolver_uses_logical_ids() {
        let id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::IncClaimReduction,
        ));
        let logical = Stage8LogicalManifest {
            openings: vec![Stage8LogicalOpening {
                id,
                point: vec![Fr::from_u64(1)],
                claim: Some(Fr::from_u64(2)),
                scale: Fr::from_u64(3),
            }],
            pcs_opening_point: Point::high_to_low(vec![Fr::from_u64(4)]),
        };
        let physical = Stage8PhysicalManifest::direct(&logical, [9; 32]);

        assert_eq!(physical.layout_digest, [9; 32]);
        assert_eq!(physical.openings[0].id, id);
        assert_eq!(physical.openings[0].relation, id);
        assert!(matches!(physical.openings[0].view, PhysicalView::Direct));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn physical_manifest_resolves_akita_packed_views_from_catalog() {
        use jolt_akita::{
            PackedAlphabet, PackedFactDomain, PackedFamilyId, PackedFamilySpec, PackedViewCatalog,
            PackedViewEntry, PackedViewFormula, PackedWitnessLayout,
        };

        let id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::IncClaimReduction,
        ));
        let logical = Stage8LogicalManifest {
            openings: vec![Stage8LogicalOpening {
                id,
                point: vec![Fr::from_u64(1)],
                claim: Some(Fr::from_u64(2)),
                scale: Fr::from_u64(3),
            }],
            pcs_opening_point: Point::high_to_low(vec![Fr::from_u64(4)]),
        };
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::IncSign,
            PackedFactDomain::TraceRows { log_t: 0 },
            1,
            PackedAlphabet::Bit,
        )])
        .unwrap_or_else(|error| panic!("test layout should be valid: {error}"));
        let catalog = PackedViewCatalog::new(
            &layout,
            [PackedViewEntry::new(
                id,
                id,
                PackedViewFormula::direct(PackedFamilyId::IncSign, 0, 1),
            )],
        )
        .unwrap_or_else(|error| panic!("test catalog should be valid: {error}"));

        let physical =
            Stage8PhysicalManifest::from_packed_view_catalog(&logical, &layout, &catalog)
                .unwrap_or_else(|error| panic!("packed view resolution should succeed: {error}"));

        assert_eq!(physical.layout_digest, layout.digest);
        assert_eq!(physical.openings[0].id, id);
        assert_eq!(physical.openings[0].relation, id);
        assert!(matches!(
            &physical.openings[0].view,
            PhysicalView::PackedLinear {
                layout_digest,
                terms
            } if *layout_digest == layout.digest
                && terms.len() == 1
                && terms[0].coefficient == Fr::from_u64(1)
                && terms[0].family == PackedFamilyId::IncSign.physical_ref()
                && terms[0].symbol == 1
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn physical_manifest_resolves_jolt_lattice_view_formulas() {
        use jolt_akita::{
            PackedAlphabet, PackedFactDomain, PackedFamilyId, PackedFamilySpec, PackedWitnessLayout,
        };
        use jolt_claims::protocols::jolt::{
            byte_decode_terms, LatticePackedFamilyId, LatticePackedViewFormula,
        };

        let jolt_id = JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeChunk(0),
            JoltRelationId::BytecodeClaimReduction,
        );
        let id = Stage8OpeningId::from(jolt_id);
        let logical = Stage8LogicalManifest {
            openings: vec![Stage8LogicalOpening {
                id,
                point: vec![Fr::from_u64(1)],
                claim: Some(Fr::from_u64(2)),
                scale: Fr::from_u64(3),
            }],
            pcs_opening_point: Point::high_to_low(vec![Fr::from_u64(4)]),
        };
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::BytecodeChunk { index: 0 },
            PackedFactDomain::BytecodeRows { log_bytecode: 1 },
            8,
            PackedAlphabet::Byte,
        )])
        .unwrap_or_else(|error| panic!("test layout should be valid: {error}"));

        let physical = Stage8PhysicalManifest::from_jolt_lattice_view_formulas(
            &logical,
            &layout,
            [(
                jolt_id,
                LatticePackedViewFormula::linear_decoded(byte_decode_terms::<Fr>(
                    LatticePackedFamilyId::BytecodeChunk { index: 0 },
                    3,
                )),
                vec![Fr::from_u64(1)],
            )],
        )
        .unwrap_or_else(|error| panic!("lattice formula should resolve: {error}"));

        assert_eq!(physical.layout_digest, layout.digest);
        assert_eq!(physical.openings[0].id, id);
        assert_eq!(physical.openings[0].relation, id);
        assert!(matches!(
            &physical.openings[0].view,
            PhysicalView::PackedLinear {
                layout_digest,
                terms
            } if *layout_digest == layout.digest
                && terms.len() == 256
                && terms[7].coefficient == Fr::from_u64(7)
                && terms[7].family == (PackedFamilyId::BytecodeChunk { index: 0 }).physical_ref()
                && terms[7].limb == 3
                && terms[7].symbol == 7
                && terms[7].row_point == vec![Fr::from_u64(1)]
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn physical_manifest_rejects_missing_jolt_lattice_formula() {
        use jolt_akita::{
            PackedAlphabet, PackedFactDomain, PackedFamilyId, PackedFamilySpec, PackedViewError,
            PackedWitnessLayout,
        };
        use jolt_claims::protocols::jolt::{LatticePackedFamilyId, LatticePackedViewFormula};

        let expected_id = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::IncClaimReduction,
        );
        let supplied_id = JoltOpeningId::committed(
            JoltCommittedPolynomial::RdInc,
            JoltRelationId::IncClaimReduction,
        );
        let logical = Stage8LogicalManifest {
            openings: vec![Stage8LogicalOpening {
                id: Stage8OpeningId::from(expected_id),
                point: vec![Fr::from_u64(1)],
                claim: Some(Fr::from_u64(2)),
                scale: Fr::from_u64(3),
            }],
            pcs_opening_point: Point::high_to_low(vec![Fr::from_u64(4)]),
        };
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::IncSign,
            PackedFactDomain::TraceRows { log_t: 0 },
            1,
            PackedAlphabet::Bit,
        )])
        .unwrap_or_else(|error| panic!("test layout should be valid: {error}"));

        assert!(matches!(
            Stage8PhysicalManifest::from_jolt_lattice_view_formulas(
                &logical,
                &layout,
                [(
                    supplied_id,
                    LatticePackedViewFormula::<Fr>::direct(LatticePackedFamilyId::IncSign, 0, 1,),
                    vec![Fr::from_u64(1)],
                )],
            ),
            Err(PackedViewError::MissingView)
        ));
    }

    #[cfg(feature = "akita")]
    #[test]
    fn physical_manifest_rejects_masked_jolt_lattice_formula() {
        use jolt_akita::{
            PackedAlphabet, PackedFactDomain, PackedFamilyId, PackedFamilySpec, PackedViewError,
            PackedWitnessLayout,
        };
        use jolt_claims::protocols::jolt::LatticePackedViewFormula;

        let jolt_id = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::IncClaimReduction,
        );
        let logical = Stage8LogicalManifest {
            openings: vec![Stage8LogicalOpening {
                id: Stage8OpeningId::from(jolt_id),
                point: vec![Fr::from_u64(1)],
                claim: Some(Fr::from_u64(2)),
                scale: Fr::from_u64(3),
            }],
            pcs_opening_point: Point::high_to_low(vec![Fr::from_u64(4)]),
        };
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::IncSign,
            PackedFactDomain::TraceRows { log_t: 0 },
            1,
            PackedAlphabet::Bit,
        )])
        .unwrap_or_else(|error| panic!("test layout should be valid: {error}"));

        assert!(matches!(
            Stage8PhysicalManifest::from_jolt_lattice_view_formulas(
                &logical,
                &layout,
                [(
                    jolt_id,
                    LatticePackedViewFormula::<Fr>::masked_decoded(
                        JoltRelationId::FusedIncrementTranslation,
                    ),
                    vec![Fr::from_u64(1)],
                )],
            ),
            Err(PackedViewError::MaskedViewRequiresTranslation)
        ));
    }
}
