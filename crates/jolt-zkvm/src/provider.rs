//! Composite polynomial data provider for the prover runtime.
//!
//! [`ProverData`] routes [`materialize`](BufferProvider::materialize) calls
//! to the appropriate source based on [`PolynomialDescriptor::source`]:
//!
//! | `PolySource` | Source | Examples |
//! |---|---|---|
//! | `Witness` | [`Polynomials<F>`] | RdInc, RamInc, one-hot RAs |
//! | `R1cs(_)` | [`R1csSource`] | Az, Bz, Cz, Variable columns |
//! | `Derived` | [`DerivedSource`] | ProductLeft, ProductRight |
//! | `Preprocessed` | [`PreprocessedSource`] | IoMask, RamInit, BytecodeTable |

use std::borrow::Cow;

use jolt_compiler::PolySource;
use jolt_compute::BufferProvider;
use jolt_field::Field;
use jolt_r1cs::R1csSource;
use jolt_witness::{PolynomialId, Polynomials};

use crate::derived::DerivedSource;
use crate::preprocessed::PreprocessedSource;

/// Composite [`BufferProvider`] unifying all prover data sources.
///
/// The runtime sees a single provider; internally it dispatches on
/// [`poly_id.descriptor().source`](PolynomialId::descriptor) to
/// the matching source.
pub struct ProverData<'a, F: Field> {
    witness: &'a mut Polynomials<F>,
    r1cs: R1csSource<'a, F>,
    derived: DerivedSource<'a, F>,
    preprocessed: PreprocessedSource<F>,
}

impl<'a, F: Field> ProverData<'a, F> {
    pub fn new(
        witness: &'a mut Polynomials<F>,
        r1cs: R1csSource<'a, F>,
        derived: DerivedSource<'a, F>,
        preprocessed: PreprocessedSource<F>,
    ) -> Self {
        Self {
            witness,
            r1cs,
            derived,
            preprocessed,
        }
    }

    /// Mutable access to the R1CS source for setting Spartan challenges.
    ///
    /// The runtime calls this before materializing `CombinedRow`, which
    /// needs the outer sumcheck challenge point and rho coefficients.
    /// This is orthogonal to the `BufferProvider` trait — it's protocol-
    /// specific state that the generic trait doesn't model.
    pub fn r1cs_mut(&mut self) -> &mut R1csSource<'a, F> {
        &mut self.r1cs
    }
}

impl<F: Field> BufferProvider<F> for ProverData<'_, F> {
    fn materialize(&self, poly_id: PolynomialId) -> Cow<'_, [F]> {
        match poly_id.descriptor().source {
            PolySource::Witness => Cow::Borrowed(self.witness.get(poly_id)),
            PolySource::R1cs(column) => Cow::Owned(self.r1cs.compute(column)),
            PolySource::Derived => self.derived.compute(poly_id),
            PolySource::Preprocessed => Cow::Borrowed(self.preprocessed.get(poly_id)),
        }
    }

    fn release(&mut self, poly_id: PolynomialId) {
        match poly_id.descriptor().source {
            PolySource::Witness => self.witness.release(poly_id),
            PolySource::R1cs(_) | PolySource::Derived => {}
            PolySource::Preprocessed => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_r1cs::{ConstraintMatrices, R1csKey};
    use jolt_witness::{CycleInput, PolynomialConfig};

    use jolt_field::Fr;
    use num_traits::One;

    #[test]
    fn routes_all_sources() {
        let config = PolynomialConfig::new(4, 8, 4, 4);
        let mut polys = Polynomials::<Fr>::new(config);
        polys.push(&[CycleInput::PADDING; 4]);
        polys.finish();

        let one = Fr::one();
        let matrices = ConstraintMatrices::new(
            1,
            2,
            vec![vec![(0, one)]],
            vec![vec![(0, one)]],
            vec![vec![(1, one)]],
        );
        let key = R1csKey::new(matrices, 2);
        let witness = vec![Fr::one(); 2 * key.num_vars_padded];
        let r1cs = R1csSource::new(&key, &witness);
        let derived = DerivedSource::new(&witness, 2, key.num_vars_padded);
        let preprocessed = PreprocessedSource::new();

        let provider = ProverData::new(&mut polys, r1cs, derived, preprocessed);

        // Witness source
        let rd_data = provider.materialize(PolynomialId::RdInc);
        assert_eq!(rd_data.len(), 4);

        // R1cs source
        assert!(matches!(
            PolynomialId::Az.descriptor().source,
            PolySource::R1cs(_)
        ));
        let az_data = provider.materialize(PolynomialId::Az);
        assert!(!az_data.is_empty());

        // Witness source (another poly)
        let ram_data = provider.materialize(PolynomialId::RamInc);
        assert_eq!(ram_data.len(), 4);
    }
}
