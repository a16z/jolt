use super::*;

impl TraceBackend {
    pub(crate) fn materialize_register_read_write_virtual<F: JoltField>(
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
                    label: JOLT_VM_LABEL,
                });
            }
        }

        let rows = self.trace_rows()?;
        let cycles = rows.cycles();
        let register_count = checked_pow2(REGISTER_ADDRESS_BITS)?;
        let mut values = jolt_utils::unsafe_allocate_zero_vec(checked_dense_grid_len::<F>(
            register_count,
            cycles,
        )?);

        // Padding rows touch no register, so the walk covers the whole domain.
        let mut cycle = 0;
        if id == JoltVirtualPolynomial::RegistersVal {
            let mut state = vec![0u64; register_count];
            rows.visit(0..cycles, WALK_ROWS, &mut |chunk, _, _| {
                for row in chunk {
                    for (register, value) in state.iter().copied().enumerate() {
                        values[register * cycles + cycle] = F::from_u64(value);
                    }
                    if let Some(register) = row.rd_index() {
                        let register = usize::from(register);
                        if register >= register_count {
                            return Err(invalid_register_address(register as u8));
                        }
                        state[register] = row.rd_write_value();
                    }
                    cycle += 1;
                }
                Ok(())
            })?;
            return Ok(values);
        }

        rows.visit(0..cycles, WALK_ROWS, &mut |chunk, _, _| {
            for row in chunk {
                let register = match id {
                    JoltVirtualPolynomial::Rs1Ra => row.rs1_index(),
                    JoltVirtualPolynomial::Rs2Ra => row.rs2_index(),
                    JoltVirtualPolynomial::RdWa => row.rd_index(),
                    _ => None,
                };
                if let Some(register) = register {
                    let register = usize::from(register);
                    if register >= register_count {
                        return Err(invalid_register_address(register as u8));
                    }
                    values[register * cycles + cycle] = F::one();
                }
                cycle += 1;
            }
            Ok(())
        })?;
        Ok(values)
    }
}

pub(crate) fn invalid_register_address(register: u8) -> WitnessError {
    WitnessError::InvalidWitnessData {
        label: JOLT_VM_LABEL,
        reason: format!(
            "register index {register} exceeds {}-bit register read-write domain",
            REGISTER_ADDRESS_BITS
        ),
    }
}
