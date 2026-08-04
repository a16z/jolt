use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
use jolt_program::execution::TraceSource;

use super::registers::invalid_register_address;
use super::{checked_pow2, TraceBackend};
use crate::backend::OneHotSource;
use crate::witnesses::ram_access_address;
use crate::{WitnessError, JOLT_VM_LABEL};

pub(crate) const ONE_HOT_REASON: &str =
    "not a one-hot polynomial with a per-cycle hot address; use oracle_table";

impl<T: TraceSource + Clone> TraceBackend<'_, T> {
    fn ram_hot_indices(&self) -> Result<Vec<Option<usize>>, WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
        let mut indices = Vec::with_capacity(cycles);
        let mut trace = self.trace.trace.clone();
        for _ in 0..cycles {
            let Some(row) = trace.next_row() else {
                indices.push(None);
                continue;
            };
            let hot = match ram_access_address(row.ram_access) {
                Some(raw) => self.remapped_ram_address(raw)?,
                None => None,
            };
            indices.push(hot);
        }
        Ok(indices)
    }

    fn register_hot_indices(
        &self,
        id: JoltVirtualPolynomial,
    ) -> Result<Vec<Option<usize>>, WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
        let register_count = checked_pow2(REGISTER_ADDRESS_BITS)?;
        let mut indices = Vec::with_capacity(cycles);
        let mut trace = self.trace.trace.clone();
        for _ in 0..cycles {
            let Some(row) = trace.next_row() else {
                indices.push(None);
                continue;
            };
            let register = match id {
                JoltVirtualPolynomial::Rs1Ra => row.registers.rs1.map(|read| read.register),
                JoltVirtualPolynomial::Rs2Ra => row.registers.rs2.map(|read| read.register),
                JoltVirtualPolynomial::RdWa => row.registers.rd.map(|write| write.register),
                _ => {
                    return Err(WitnessError::UnknownOracle {
                        label: JOLT_VM_LABEL,
                    })
                }
            };
            let hot = match register {
                Some(register) => {
                    let register = usize::from(register);
                    if register >= register_count {
                        return Err(invalid_register_address(register as u8));
                    }
                    Some(register)
                }
                None => None,
            };
            indices.push(hot);
        }
        Ok(indices)
    }
}

impl<T: TraceSource + Clone> OneHotSource for TraceBackend<'_, T> {
    fn hot_indices(&self, id: JoltPolynomialId) -> Result<Vec<Option<usize>>, WitnessError> {
        match id {
            JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamRa) => self.ram_hot_indices(),
            JoltPolynomialId::Virtual(
                virtual_id @ (JoltVirtualPolynomial::Rs1Ra
                | JoltVirtualPolynomial::Rs2Ra
                | JoltVirtualPolynomial::RdWa),
            ) => self.register_hot_indices(virtual_id),
            _ => Err(WitnessError::NotServed {
                oracle: format!("{id:?}"),
                reason: ONE_HOT_REASON,
            }),
        }
    }

    fn hot_address_bits(&self, id: JoltPolynomialId) -> Result<usize, WitnessError> {
        match id {
            JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamRa) => self.ram_log_k(),
            JoltPolynomialId::Virtual(
                JoltVirtualPolynomial::Rs1Ra
                | JoltVirtualPolynomial::Rs2Ra
                | JoltVirtualPolynomial::RdWa,
            ) => Ok(REGISTER_ADDRESS_BITS),
            _ => Err(WitnessError::NotServed {
                oracle: format!("{id:?}"),
                reason: ONE_HOT_REASON,
            }),
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_field::{Fr, FromPrimitiveInt};

    use crate::backend::OneHotSource;
    use crate::testing::{with_ram_sized_backend, with_sample_backend};
    use crate::JoltWitnessOracle;

    fn assert_views_agree<B>(backend: &B, id: JoltPolynomialId)
    where
        B: JoltWitnessOracle<Fr> + OneHotSource,
    {
        let dense: Vec<Fr> = JoltWitnessOracle::<Fr>::oracle_table(backend, id).unwrap();
        let indices = backend.hot_indices(id).unwrap();
        let log_k = backend.hot_address_bits(id).unwrap();
        let cycles = indices.len();
        assert_eq!(dense.len(), cycles << log_k, "grid shape for {id:?}");

        let mut rebuilt = vec![Fr::from_u64(0); dense.len()];
        for (cycle, hot) in indices.iter().enumerate() {
            if let Some(address) = hot {
                rebuilt[address * cycles + cycle] = Fr::from_u64(1);
            }
        }
        assert_eq!(rebuilt, dense, "sparse and dense views disagree for {id:?}");
    }

    #[test]
    fn hot_indices_reconstruct_the_dense_grid() {
        with_sample_backend(|backend| {
            for id in [
                JoltVirtualPolynomial::Rs1Ra,
                JoltVirtualPolynomial::Rs2Ra,
                JoltVirtualPolynomial::RdWa,
            ] {
                assert_views_agree(backend, JoltPolynomialId::Virtual(id));
            }
        });
    }

    #[test]
    fn ram_hot_indices_reconstruct_the_dense_grid() {
        with_ram_sized_backend(|backend| {
            assert_views_agree(
                backend,
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamRa),
            );
        });
    }

    #[test]
    fn non_one_hot_ids_are_rejected() {
        with_sample_backend(|backend| {
            let id = JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamVal);
            assert!(backend.hot_indices(id).is_err());
            assert!(backend.hot_address_bits(id).is_err());
        });
    }
}
