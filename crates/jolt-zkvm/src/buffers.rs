//! Composite buffer provider for the prover runtime.
//!
//! [`ProverBuffers`] routes [`BufferProvider::load`] calls to the appropriate
//! data source based on [`PolynomialId`]: witness polynomials come from
//! [`Polynomials`], R1CS-derived polynomials (Az, Bz, Cz, combined row)
//! come from [`R1csProvider`].

use jolt_compute::{Buf, BufferProvider, ComputeBackend, DeviceBuffer};
use jolt_field::Field;
use jolt_r1cs::{R1csProvider, POLY_AZ, POLY_BZ, POLY_COMBINED_ROW, POLY_CZ};
use jolt_witness::{PolynomialId, Polynomials};

/// Composite [`BufferProvider`] that unifies witness and R1CS data sources.
///
/// The runtime sees a single provider; internally it dispatches to:
/// - [`Polynomials<F>`] for committed witness polynomials (RdInc, RamInc, RA, etc.)
/// - [`R1csProvider`] for Spartan's R1CS-derived polynomials (Az, Bz, Cz, combined row)
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

impl<B: ComputeBackend, F: Field> BufferProvider<PolynomialId, B, F> for ProverBuffers<'_, F> {
    fn load(&mut self, poly_id: PolynomialId, backend: &B) -> Buf<B, F> {
        match poly_id {
            PolynomialId::Az => self.r1cs.load(POLY_AZ, backend),
            PolynomialId::Bz => self.r1cs.load(POLY_BZ, backend),
            PolynomialId::Cz => self.r1cs.load(POLY_CZ, backend),
            PolynomialId::CombinedRow => self.r1cs.load(POLY_COMBINED_ROW, backend),
            id => DeviceBuffer::Field(backend.upload(self.polys.get(id))),
        }
    }

    fn as_slice(&self, poly_id: PolynomialId) -> &[F] {
        match poly_id {
            PolynomialId::Az | PolynomialId::Bz | PolynomialId::Cz | PolynomialId::CombinedRow => {
                panic!(
                    "R1CS polynomial {poly_id:?} is computed on-the-fly; \
                 it should never appear in CollectOpeningClaim",
                )
            }
            id => self.polys.get(id),
        }
    }

    fn release(&mut self, poly_id: PolynomialId) {
        match poly_id {
            // R1CS polys are computed on-the-fly, nothing to release
            PolynomialId::Az | PolynomialId::Bz | PolynomialId::Cz | PolynomialId::CombinedRow => {}
            id => self.polys.release(id),
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
        // Minimal witness: 4 padding cycles
        let config = PolynomialConfig::new(4, 8, 4, 4);
        let mut polys = Polynomials::<Fr>::new(config);
        polys.push(&[CycleInput::PADDING; 4]);
        polys.finish();

        // Minimal R1CS: 1 constraint, 2 vars, 2 cycles
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

        // Load R1CS poly
        let az_buf = provider.load(PolynomialId::Az, &backend);
        let az_data = backend.download(az_buf.as_field());
        assert!(!az_data.is_empty());

        // Load another witness poly
        let ram_buf = provider.load(PolynomialId::RamInc, &backend);
        let ram_data = backend.download(ram_buf.as_field());
        assert_eq!(ram_data.len(), 4);
    }
}
