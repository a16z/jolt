use crate::{OracleRef, PolynomialEncoding, WitnessDimensions, WitnessNamespace};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamespaceId;

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

    #[test]
    fn descriptor_reports_declared_rows_and_encoding() {
        let descriptor = OracleDescriptor::<TestNamespace>::new(
            OracleRef::committed(3),
            WitnessDimensions::new(3),
            PolynomialEncoding::Dense,
        );

        assert_eq!(descriptor.dimensions.rows(), 8);
        assert_eq!(descriptor.encoding, PolynomialEncoding::Dense);
    }
}
