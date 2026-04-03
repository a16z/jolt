//! Composite buffer provider for the prover runtime.
//!
//! [`ProverBuffers`] routes [`BufferProvider::load`] calls to the appropriate
//! data source based on [`PolynomialDescriptor::source`]: witness polynomials
//! come from [`Polynomials`], R1CS-derived polynomials come from [`R1csProvider`].

use jolt_compiler::PolySource;
use jolt_compute::{Buf, BufferProvider, ComputeBackend, DeviceBuffer};
use jolt_field::Field;
use jolt_r1cs::R1csProvider;
use jolt_witness::{PolynomialId, Polynomials};

/// Composite [`BufferProvider`] that unifies witness and R1CS data sources.
///
/// The runtime sees a single provider; internally it dispatches on
/// [`descriptor().source`](PolynomialId::descriptor):
/// - [`PolySource::R1cs`] → [`R1csProvider`]
/// - Everything else → [`Polynomials<F>`]
pub struct ProverBuffers<'a, F: Field> {
    polys: &'a mut Polynomials<F>,
    r1cs: R1csProvider<'a, F>,
}

impl<'a, F: Field> ProverBuffers<'a, F> {
    pub fn new(polys: &'a mut Polynomials<F>, r1cs: R1csProvider<'a, F>) -> Self {
        Self { polys, r1cs }
    }

    /// Mutable access to the R1CS provider (e.g. to set Spartan challenges mid-proving).
    pub fn r1cs_mut(&mut self) -> &mut R1csProvider<'a, F> {
        &mut self.r1cs
    }
}

impl<B: ComputeBackend, F: Field> BufferProvider<B, F> for ProverBuffers<'_, F> {
    fn load(&mut self, poly_id: PolynomialId, backend: &B) -> Buf<B, F> {
        match poly_id.descriptor().source {
            PolySource::R1cs(_) => self.r1cs.load(poly_id, backend),
            _ => DeviceBuffer::Field(backend.upload(self.polys.get(poly_id))),
        }
    }

    fn as_slice(&self, poly_id: PolynomialId) -> &[F] {
        match poly_id.descriptor().source {
            PolySource::R1cs(_) => {
                panic!(
                    "R1CS polynomial {poly_id:?} is computed on-the-fly; \
                     it should never appear in CollectOpeningClaim",
                )
            }
            _ => self.polys.get(poly_id),
        }
    }

    fn release(&mut self, poly_id: PolynomialId) {
        match poly_id.descriptor().source {
            PolySource::R1cs(_) => {} // on-the-fly, nothing to release
            _ => self.polys.release(poly_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_compute::ComputeBackend;
    use jolt_cpu::CpuBackend;
    use jolt_r1cs::{ConstraintMatrices, R1csKey};
    use jolt_witness::{CycleInput, PolynomialConfig};

    use jolt_field::Fr;
    use num_traits::One;

    #[test]
    fn routes_witness_and_r1cs() {
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
        let r1cs = R1csProvider::new(&key, &witness);

        let mut provider = ProverBuffers::new(&mut polys, r1cs);
        let backend = CpuBackend;

        // Load witness poly
        let rd_buf = provider.load(PolynomialId::RdInc, &backend);
        let rd_data = backend.download(rd_buf.as_field());
        assert_eq!(rd_data.len(), 4);

        // Load R1CS poly — dispatches through descriptor().source
        assert!(matches!(
            PolynomialId::Az.descriptor().source,
            PolySource::R1cs(_)
        ));
        let az_buf = provider.load(PolynomialId::Az, &backend);
        let az_data = backend.download(az_buf.as_field());
        assert!(!az_data.is_empty());

        // Load another witness poly
        let ram_buf = provider.load(PolynomialId::RamInc, &backend);
        let ram_data = backend.download(ram_buf.as_field());
        assert_eq!(ram_data.len(), 4);
    }
}
