use crate::protocols::field_inline::FieldInlineOpeningId;
use crate::protocols::jolt::JoltOpeningId;

/// An id in a protocol family defined outside the workspace.
///
/// The family and the verifier composing it must uphold these soundness
/// requirements; the composite arms perform no runtime validation:
///
/// - Use one constant `family` string, distinct from every other family hosted
///   by the same verifier.
/// - Make `From` conversions into each composite enum injective per id kind.
/// - Make `TryFrom<ComposedOpeningId>` the exact inverse of the opening-id
///   embedding. Return the original composite unchanged in `Err` for every
///   other arm, other family, and own-family index outside the embedding's image.
/// - Uphold the alias invariants of
///   `ConcreteSumcheck::aliased_output_openings` in `jolt-verifier`:
///   each alias is owned and expression-referenced by the declaring relation,
///   has one canonical source absorbed by another batch member, and binds the
///   identical point slice as its source.
///
/// Member resolvers trust these conversions and take the first member that
/// resolves an id. Violating the requirements can make alias validation compare
/// the wrong claims without an error. Composite ids are neither serialized nor
/// absorbed into the transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExternalId {
    pub family: &'static str,
    pub index: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComposedOpeningId {
    Jolt(JoltOpeningId),
    FieldInline(FieldInlineOpeningId),
    External(ExternalId),
}

impl From<JoltOpeningId> for ComposedOpeningId {
    fn from(id: JoltOpeningId) -> Self {
        Self::Jolt(id)
    }
}

impl From<FieldInlineOpeningId> for ComposedOpeningId {
    fn from(id: FieldInlineOpeningId) -> Self {
        Self::FieldInline(id)
    }
}

/// Downcast from the composite [`ComposedOpeningId`] to one protocol family's
/// opening id: returns the original id on a family mismatch. The inverse
/// of the `From` embeddings above, letting family-typed resolvers (each batch
/// member's claim struct speaks its own id family) participate in one
/// composite-keyed lookup.
impl TryFrom<ComposedOpeningId> for JoltOpeningId {
    type Error = ComposedOpeningId;

    fn try_from(id: ComposedOpeningId) -> Result<Self, Self::Error> {
        match id {
            ComposedOpeningId::Jolt(id) => Ok(id),
            ComposedOpeningId::FieldInline(_) | ComposedOpeningId::External(_) => Err(id),
        }
    }
}

impl TryFrom<ComposedOpeningId> for FieldInlineOpeningId {
    type Error = ComposedOpeningId;

    fn try_from(id: ComposedOpeningId) -> Result<Self, Self::Error> {
        match id {
            ComposedOpeningId::FieldInline(id) => Ok(id),
            ComposedOpeningId::Jolt(_) | ComposedOpeningId::External(_) => Err(id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::field_inline::{FieldInlineRelationId, FieldInlineVirtualPolynomial};
    use crate::protocols::jolt::JoltRelationId;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum AlphaId {
        Left,
        Right,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum BetaId {
        First,
        Second,
    }

    impl From<AlphaId> for ComposedOpeningId {
        fn from(id: AlphaId) -> Self {
            Self::External(ExternalId {
                family: "alpha",
                index: match id {
                    AlphaId::Left => 0,
                    AlphaId::Right => 7,
                },
            })
        }
    }

    impl TryFrom<ComposedOpeningId> for AlphaId {
        type Error = ComposedOpeningId;

        fn try_from(id: ComposedOpeningId) -> Result<Self, Self::Error> {
            match id {
                ComposedOpeningId::External(ExternalId {
                    family: "alpha",
                    index: 0,
                }) => Ok(Self::Left),
                ComposedOpeningId::External(ExternalId {
                    family: "alpha",
                    index: 7,
                }) => Ok(Self::Right),
                ComposedOpeningId::Jolt(_)
                | ComposedOpeningId::FieldInline(_)
                | ComposedOpeningId::External(_) => Err(id),
            }
        }
    }

    impl From<BetaId> for ComposedOpeningId {
        fn from(id: BetaId) -> Self {
            Self::External(ExternalId {
                family: "beta",
                index: match id {
                    BetaId::First => 0,
                    BetaId::Second => 9,
                },
            })
        }
    }

    impl TryFrom<ComposedOpeningId> for BetaId {
        type Error = ComposedOpeningId;

        fn try_from(id: ComposedOpeningId) -> Result<Self, Self::Error> {
            match id {
                ComposedOpeningId::External(ExternalId {
                    family: "beta",
                    index: 0,
                }) => Ok(Self::First),
                ComposedOpeningId::External(ExternalId {
                    family: "beta",
                    index: 9,
                }) => Ok(Self::Second),
                ComposedOpeningId::Jolt(_)
                | ComposedOpeningId::FieldInline(_)
                | ComposedOpeningId::External(_) => Err(id),
            }
        }
    }

    #[test]
    fn external_family_conversions_are_disjoint() {
        for (id, index) in [(AlphaId::Left, 0), (AlphaId::Right, 7)] {
            let composite = ComposedOpeningId::from(id);
            assert_eq!(
                composite,
                ComposedOpeningId::External(ExternalId {
                    family: "alpha",
                    index,
                })
            );
            assert_eq!(AlphaId::try_from(composite), Ok(id));
            assert_eq!(BetaId::try_from(composite), Err(composite));
        }
        for (id, index) in [(BetaId::First, 0), (BetaId::Second, 9)] {
            let composite = ComposedOpeningId::from(id);
            assert_eq!(
                composite,
                ComposedOpeningId::External(ExternalId {
                    family: "beta",
                    index,
                })
            );
            assert_eq!(BetaId::try_from(composite), Ok(id));
            assert_eq!(AlphaId::try_from(composite), Err(composite));
        }
        for composite in [
            ComposedOpeningId::Jolt(JoltOpeningId::trusted_advice(JoltRelationId::SpartanOuter)),
            ComposedOpeningId::FieldInline(FieldInlineOpeningId::virtual_polynomial(
                FieldInlineVirtualPolynomial::FieldRdValue,
                FieldInlineRelationId::FieldRegistersSpartanOuter,
            )),
            ComposedOpeningId::External(ExternalId {
                family: "alpha",
                index: 1,
            }),
            ComposedOpeningId::External(ExternalId {
                family: "beta",
                index: 1,
            }),
        ] {
            assert_eq!(AlphaId::try_from(composite), Err(composite));
            assert_eq!(BetaId::try_from(composite), Err(composite));
        }
    }

    #[test]
    fn external_opening_ids_sort_after_workspace_ids() {
        let jolt =
            ComposedOpeningId::Jolt(JoltOpeningId::trusted_advice(JoltRelationId::SpartanOuter));
        let field_inline =
            ComposedOpeningId::FieldInline(FieldInlineOpeningId::virtual_polynomial(
                FieldInlineVirtualPolynomial::FieldRdValue,
                FieldInlineRelationId::FieldRegistersSpartanOuter,
            ));
        let external = ComposedOpeningId::External(ExternalId {
            family: "alpha",
            index: 0,
        });
        assert!(jolt < field_inline);
        assert!(jolt < external);
        assert!(field_inline < external);
        assert_eq!(JoltOpeningId::try_from(external), Err(external));
        assert_eq!(FieldInlineOpeningId::try_from(external), Err(external));
        assert!(
            external
                < ComposedOpeningId::External(ExternalId {
                    family: "alpha",
                    index: 1
                })
        );
        assert!(
            ComposedOpeningId::External(ExternalId {
                family: "alpha",
                index: u64::MAX
            }) < ComposedOpeningId::External(ExternalId {
                family: "beta",
                index: 0
            })
        );
    }
}
