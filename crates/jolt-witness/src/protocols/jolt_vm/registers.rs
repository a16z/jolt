//! Register read-write rows and register-domain virtual polynomials.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoltVmRegisterRead {
    pub register: u8,
    pub value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoltVmRegisterWrite {
    pub register: u8,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JoltVmRegisterReadWriteRow {
    pub rs1: Option<JoltVmRegisterRead>,
    pub rs2: Option<JoltVmRegisterRead>,
    pub rd: Option<JoltVmRegisterWrite>,
    pub rd_increment: i128,
}

pub trait JoltVmRegisterReadWriteRows {
    fn register_read_write_rows(&self) -> Result<Vec<JoltVmRegisterReadWriteRow>, WitnessError>;
}

impl<T: TraceSource + Clone> JoltVmRegisterReadWriteRows for TraceBackedJoltVmWitness<'_, T> {
    fn register_read_write_rows(&self) -> Result<Vec<JoltVmRegisterReadWriteRow>, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        let register_count = checked_pow2(REGISTER_ADDRESS_BITS)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        for _ in 0..rows {
            let Some(row) = trace.next_row() else {
                values.push(JoltVmRegisterReadWriteRow::default());
                continue;
            };
            let rs1 = row
                .registers
                .rs1
                .map(|read| register_read(read.register, read.value, register_count))
                .transpose()?;
            let rs2 = row
                .registers
                .rs2
                .map(|read| register_read(read.register, read.value, register_count))
                .transpose()?;
            let rd = row
                .registers
                .rd
                .map(|write| {
                    register_write(
                        write.register,
                        write.pre_value,
                        write.post_value,
                        register_count,
                    )
                })
                .transpose()?;
            values.push(JoltVmRegisterReadWriteRow {
                rs1,
                rs2,
                rd,
                rd_increment: JoltVmIncrementStreamKind::RdInc.value_from_row(&row),
            });
        }
        Ok(values)
    }
}

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

    pub(crate) fn evaluate_register_read_write_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
        point: &[F],
    ) -> Result<F, WitnessError> {
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
        let expected_vars = REGISTER_ADDRESS_BITS
            .checked_add(self.config.log_t)
            .ok_or_else(|| WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: "register read-write point length overflow".to_owned(),
            })?;
        if point.len() != expected_vars {
            return Err(WitnessError::InvalidDimensions {
                namespace: JOLT_VM_NAMESPACE.name,
                reason: format!(
                    "register read-write point has {} variables, expected {expected_vars}",
                    point.len()
                ),
            });
        }

        let cycles = checked_pow2(self.config.log_t)?;
        let (register_point, cycle_point) = point.split_at(REGISTER_ADDRESS_BITS);
        let register_eq = eq_evals_msb(register_point)?;
        let cycle_eq = eq_evals_msb(cycle_point)?;
        let register_count = checked_pow2(REGISTER_ADDRESS_BITS)?;

        if id == JoltVirtualPolynomial::RegistersVal {
            let mut state = vec![0u64; register_count];
            let mut state_eval = F::zero();
            let mut trace = self.trace.trace.clone();
            let mut result = F::zero();
            for cycle_weight in cycle_eq.iter().copied().take(cycles) {
                result += cycle_weight * state_eval;
                let Some(row) = trace.next_row() else {
                    continue;
                };
                if let Some(write) = row.registers.rd {
                    let register = usize::from(write.register);
                    if register >= register_count {
                        return Err(invalid_register_address(write.register));
                    }
                    let previous = state[register];
                    state_eval += register_eq[register]
                        * (F::from_u64(write.post_value) - F::from_u64(previous));
                    state[register] = write.post_value;
                }
            }
            return Ok(result);
        }

        let mut trace = self.trace.trace.clone();
        let mut result = F::zero();
        for &cycle_weight in cycle_eq.iter().take(cycles) {
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
                result += cycle_weight * register_eq[register];
            }
        }
        Ok(result)
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

pub(crate) fn register_read(
    register: u8,
    value: u64,
    register_count: usize,
) -> Result<JoltVmRegisterRead, WitnessError> {
    if usize::from(register) >= register_count {
        return Err(invalid_register_address(register));
    }
    Ok(JoltVmRegisterRead { register, value })
}

pub(crate) fn register_write(
    register: u8,
    pre_value: u64,
    post_value: u64,
    register_count: usize,
) -> Result<JoltVmRegisterWrite, WitnessError> {
    if usize::from(register) >= register_count {
        return Err(invalid_register_address(register));
    }
    Ok(JoltVmRegisterWrite {
        register,
        pre_value,
        post_value,
    })
}
