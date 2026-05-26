use crate::{OracleRef, WitnessDimensions, WitnessNamespace};

use super::PolynomialEncoding;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaterializationPolicy {
    #[default]
    BackendChoice,
    Borrowed,
    Materialized,
    Streaming,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RetentionHint {
    #[default]
    Ephemeral,
    ThroughStage8,
    ThroughBlindFold,
    Permanent,
}

#[derive(Debug, PartialEq, Eq)]
pub struct OracleDescriptor<N: WitnessNamespace> {
    pub reference: OracleRef<N>,
    pub dimensions: WitnessDimensions,
    pub encoding: PolynomialEncoding,
}

impl<N: WitnessNamespace> Clone for OracleDescriptor<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N: WitnessNamespace> Copy for OracleDescriptor<N> {}

impl<N: WitnessNamespace> OracleDescriptor<N> {
    pub const fn new(
        reference: OracleRef<N>,
        dimensions: WitnessDimensions,
        encoding: PolynomialEncoding,
    ) -> Self {
        Self {
            reference,
            dimensions,
            encoding,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ViewRequirement<N: WitnessNamespace> {
    pub oracle: OracleRef<N>,
    pub encoding: PolynomialEncoding,
    pub materialization: MaterializationPolicy,
    pub retention: RetentionHint,
}

impl<N: WitnessNamespace> Clone for ViewRequirement<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N: WitnessNamespace> Copy for ViewRequirement<N> {}

impl<N: WitnessNamespace> ViewRequirement<N> {
    pub const fn new(
        oracle: OracleRef<N>,
        encoding: PolynomialEncoding,
        materialization: MaterializationPolicy,
        retention: RetentionHint,
    ) -> Self {
        Self {
            oracle,
            encoding,
            materialization,
            retention,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleViewRequest<N: WitnessNamespace> {
    pub requirement: ViewRequirement<N>,
}

impl<N: WitnessNamespace> OracleViewRequest<N> {
    pub const fn new(requirement: ViewRequirement<N>) -> Self {
        Self { requirement }
    }

    pub const fn oracle(&self) -> OracleRef<N> {
        self.requirement.oracle
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolynomialView<'a, F, N: WitnessNamespace> {
    Borrowed {
        descriptor: OracleDescriptor<N>,
        values: &'a [F],
    },
    Owned {
        descriptor: OracleDescriptor<N>,
        values: Vec<F>,
    },
    Deferred {
        descriptor: OracleDescriptor<N>,
    },
}

impl<'a, F, N: WitnessNamespace> PolynomialView<'a, F, N> {
    pub const fn borrowed(descriptor: OracleDescriptor<N>, values: &'a [F]) -> Self {
        Self::Borrowed { descriptor, values }
    }

    pub const fn owned(descriptor: OracleDescriptor<N>, values: Vec<F>) -> Self {
        Self::Owned { descriptor, values }
    }

    pub const fn deferred(descriptor: OracleDescriptor<N>) -> Self {
        Self::Deferred { descriptor }
    }

    pub const fn descriptor(&self) -> &OracleDescriptor<N> {
        match self {
            Self::Borrowed { descriptor, .. }
            | Self::Owned { descriptor, .. }
            | Self::Deferred { descriptor } => descriptor,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Borrowed { values, .. } => values.len(),
            Self::Owned { values, .. } => values.len(),
            Self::Deferred { descriptor } => descriptor.dimensions.rows,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub const fn encoding(&self) -> PolynomialEncoding {
        self.descriptor().encoding
    }

    pub const fn as_slice(&self) -> Option<&[F]> {
        match self {
            Self::Borrowed { values, .. } => Some(values),
            Self::Owned { values, .. } => Some(values.as_slice()),
            Self::Deferred { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NamespaceId, WitnessNamespace};

    #[derive(Clone, Copy, Debug)]
    enum TestNamespace {}

    impl WitnessNamespace for TestNamespace {
        type ChallengeId = u8;
        type CommittedId = u8;
        type OpeningId = u8;
        type PublicId = u8;
        type VirtualId = u8;

        const ID: NamespaceId = NamespaceId::new("test");
    }

    fn descriptor(rows: usize) -> OracleDescriptor<TestNamespace> {
        OracleDescriptor::new(
            OracleRef::committed(3),
            WitnessDimensions::new(rows, 2),
            PolynomialEncoding::Dense,
        )
    }

    #[test]
    fn borrowed_view_reports_backing_slice() {
        let values = [1_u64, 2, 3, 4];
        let view = PolynomialView::borrowed(descriptor(values.len()), &values);

        assert_eq!(view.len(), values.len());
        assert_eq!(view.as_slice(), Some(values.as_slice()));
        assert_eq!(view.encoding(), PolynomialEncoding::Dense);
    }

    #[test]
    fn deferred_view_reports_declared_rows_without_materializing() {
        let view = PolynomialView::<u64, TestNamespace>::deferred(descriptor(8));

        assert_eq!(view.len(), 8);
        assert_eq!(view.as_slice(), None);
    }
}
