//! Register-domain virtual polynomials.

use super::*;

impl<T: TraceSource + Clone> TraceBackedJoltVmWitness<'_, T> {
    pub(crate) fn materialize_register_read_write_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        match id {
            JoltVirtualPolynomial::RegistersVal
            | JoltVirtualPolynomial::Rs1Ra
            | JoltVirtualPolynomial::Rs2Ra
            | JoltVirtualPolynomial::RdWa => {}
            _ => {
                return Err(WitnessError::UnknownOracle {
                    namespace: JOLT_VM_NAMESPACE.name,
                });
            }
        }

        let cycles = checked_pow2(self.config.log_t)?;
        let register_count = checked_pow2(REGISTER_ADDRESS_BITS)?;
        let mut values = vec![F::zero(); register_count * cycles];
        let mut trace = self.trace.trace.clone();

        if id == JoltVirtualPolynomial::RegistersVal {
            let mut state = vec![0u64; register_count];
            for cycle in 0..cycles {
                for (register, value) in state.iter().copied().enumerate() {
                    values[register * cycles + cycle] = F::from_u64(value);
                }

                let Some(row) = trace.next_row() else {
                    continue;
                };
                if let Some(write) = row.registers.rd {
                    let register = usize::from(write.register);
                    if register >= register_count {
                        return Err(invalid_register_address(write.register));
                    }
                    state[register] = write.post_value;
                }
            }
            return Ok(values);
        }

        for cycle in 0..cycles {
            let Some(row) = trace.next_row() else {
                break;
            };
            let register = match id {
                JoltVirtualPolynomial::Rs1Ra => row.registers.rs1.map(|read| read.register),
                JoltVirtualPolynomial::Rs2Ra => row.registers.rs2.map(|read| read.register),
                JoltVirtualPolynomial::RdWa => row.registers.rd.map(|write| write.register),
                _ => None,
            };
            if let Some(register) = register {
                let register = usize::from(register);
                if register >= register_count {
                    return Err(invalid_register_address(register as u8));
                }
                values[register * cycles + cycle] = F::one();
            }
        }

        Ok(values)
    }
}

pub(crate) fn invalid_register_address(register: u8) -> WitnessError {
    WitnessError::InvalidWitnessData {
        namespace: JOLT_VM_NAMESPACE.name,
        reason: format!(
            "register index {register} exceeds {}-bit register read-write domain",
            REGISTER_ADDRESS_BITS
        ),
    }
}
