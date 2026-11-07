//! Address-first RAM and register rounds over a shared device matrix.
//!
//! Address rounds retain sparse entries, initial-value checkpoints, and a
//! dense cycle-eq table. After the last address bind, `finish_address` produces
//! dense read/write/value cycle columns and releases that equality table;
//! the cycle tail uses split-eq reductions.
//!
//! Both callers place this state on GPU 0. In particular, registers use the
//! same sparse address-matrix path as RAM, whereas optimized registers scan
//! compact access chunks with raw increment checkpoints.

use jolt_field::{Fr, JoltField};
use jolt_poly::{BindingOrder, UnivariatePoly};

use super::address_major_matrix::DeviceAddressMajorMatrix;
use super::context::CudaKernelContext;
use super::device::{fr_into, require_fr, DeviceFrVec};
use super::error::CudaError;
use super::half_fold::FoldColumn;
use super::read_write_matrix::DeviceReadWriteMatrix;
use super::split_eq::DeviceSplitEq;
use super::sum_of_products::{DeviceSumOfProducts, SumOfProducts};

pub(crate) struct AddressFirst<F: JoltField> {
    context: &'static CudaKernelContext,
    matrix: Option<DeviceAddressMajorMatrix>,
    dense: Option<[DeviceFrVec; 3]>,
    inc: DeviceFrVec,
    eq: DeviceSplitEq<F>,
    merged_eq: Option<DeviceFrVec>,
    form: DeviceSumOfProducts,
    log_k: usize,
    cycles: usize,
    bound: usize,
}

impl<F: JoltField> AddressFirst<F> {
    #[cfg(feature = "allocative")]
    pub(crate) fn device_bytes(&self) -> usize {
        self.matrix
            .as_ref()
            .map_or(0, DeviceAddressMajorMatrix::device_bytes)
            + self.dense.as_ref().map_or(0, |columns| {
                columns.iter().map(DeviceFrVec::device_bytes).sum()
            })
            + self.inc.device_bytes()
            + self.eq.device_bytes()
            + self.merged_eq.as_ref().map_or(0, DeviceFrVec::device_bytes)
            + self.form.device_bytes()
    }

    pub(crate) fn new(
        context: &'static CudaKernelContext,
        matrix: DeviceReadWriteMatrix,
        inc: DeviceFrVec,
        val_init: &[Fr],
        cycle_point: &[F],
        write_scale: F,
    ) -> Result<Self, CudaError> {
        let log_k = val_init.len().ilog2() as usize;
        let eq = DeviceSplitEq::new(context, cycle_point, BindingOrder::LowToHigh)?;
        let mut form = SumOfProducts::new();
        form.push(F::one(), &[0, 2])?;
        form.push(write_scale, &[1, 2])?;
        form.push(write_scale, &[1, 3])?;
        let mut state = Self {
            context,
            matrix: Some(matrix.into_address_first(context, val_init)?),
            dense: None,
            cycles: inc.len(),
            inc,
            merged_eq: Some(eq.merge(context)?),
            eq,
            form: form.upload(context)?,
            log_k,
            bound: 0,
        };
        if log_k == 0 {
            state.finish_address()?;
        }
        Ok(state)
    }

    fn missing() -> CudaError {
        CudaError::InvariantViolation {
            reason: "the address-first read/write phase has no device tables",
        }
    }

    fn finish_address(&mut self) -> Result<(), CudaError> {
        self.dense = Some(self.matrix.take().ok_or_else(Self::missing)?.materialize(
            self.context,
            1,
            self.cycles,
        )?);
        self.merged_eq = None;
        Ok(())
    }

    pub(crate) fn bind(&mut self, challenge: F) -> Result<(), CudaError> {
        if self.bound < self.log_k {
            self.matrix
                .as_mut()
                .ok_or_else(Self::missing)?
                .bind(self.context, challenge)?;
            self.bound += 1;
            if self.bound == self.log_k {
                self.finish_address()?;
            }
        } else {
            let r = require_fr(challenge)?;
            for column in self.dense.as_mut().ok_or_else(Self::missing)? {
                *column = self.context.bind(column, r, BindingOrder::LowToHigh)?;
            }
            self.inc = self.context.bind(&self.inc, r, BindingOrder::LowToHigh)?;
            self.eq.bind(challenge);
            self.bound += 1;
        }
        Ok(())
    }

    pub(crate) fn round(&self, claim: F) -> Result<UnivariatePoly<F>, CudaError> {
        if self.bound < self.log_k {
            let evals = self
                .matrix
                .as_ref()
                .ok_or_else(Self::missing)?
                .round_evals(
                    self.context,
                    &self.inc,
                    self.merged_eq.as_ref().ok_or_else(Self::missing)?,
                )?;
            return Ok(UnivariatePoly::from_evals_and_hint(claim, &evals));
        }
        let columns = self.dense.as_ref().ok_or_else(Self::missing)?;
        let handles = [
            FoldColumn::Field(&columns[0]),
            FoldColumn::Field(&columns[1]),
            FoldColumn::Field(&columns[2]),
            FoldColumn::Field(&self.inc),
        ];
        let (q0, leading, q1) = self.form.round_gruen_endpoints(
            self.context,
            &handles,
            self.inc.len() / 2,
            &self.eq,
        )?;
        self.eq.gruen_poly_deg_3(q0, leading, claim, q1)
    }

    pub(crate) fn final_values(&self) -> Result<[F; 4], CudaError> {
        let columns = self.dense.as_ref().ok_or_else(Self::missing)?;
        let mut values = [F::zero(); 4];
        for (out, column) in values.iter_mut().zip(columns.iter().chain([&self.inc])) {
            if column.len() != 1 {
                return Err(Self::missing());
            }
            *out = fr_into(column.first()?).ok_or_else(Self::missing)?;
        }
        Ok(values)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "device test failures must fail the test"
)]
mod tests {
    use jolt_field::Fr;
    use jolt_poly::EqPolynomial;

    use super::AddressFirst;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::read_write_matrix::{DeviceReadWriteMatrix, MatrixEntry};

    fn fr(value: u64) -> Fr {
        Fr::from(value)
    }

    #[test]
    fn address_first_rounds_match_dense_polynomials_at_boolean_and_field_challenges() {
        let Some(context) = shared_context() else {
            return;
        };
        const LOG_T: usize = 5;
        const LOG_K: usize = 3;
        for mode in 0..3 {
            let mut cycles = 1usize << LOG_T;
            let mut addresses = 1usize << LOG_K;
            let mut memory: Vec<u64> = (0..addresses).map(|k| 100 + k as u64).collect();
            let initial: Vec<_> = memory.iter().map(|&value| fr(value)).collect();
            let mut val = vec![fr(0); addresses * cycles];
            let mut ra = val.clone();
            let mut inc = vec![fr(0); cycles];
            let mut entries = Vec::new();
            for j in 0..cycles {
                for k in 0..addresses {
                    val[k * cycles + j] = fr(memory[k]);
                }
                if j % 7 == 0 {
                    continue;
                }
                let k = (j * 3 + j / 8) % (addresses - 1);
                let before = memory[k];
                let after = (before as i64 + j as i64 % 5 - 2) as u64;
                memory[k] = after;
                inc[j] = fr(after) - fr(before);
                ra[k * cycles + j] = fr(1);
                entries.push(MatrixEntry {
                    row: j as u32,
                    col: k as u32,
                    val_coeff: fr(before),
                    prev_val: before,
                    next_val: after,
                    coeffs: [fr(1), fr(0)],
                });
            }
            let gamma = fr(17);
            let point: Vec<_> = (0..LOG_T)
                .map(|i| fr(if mode < 2 { mode } else { 31 + i as u64 }))
                .collect();
            let mut eq = EqPolynomial::evals(&point, None);
            let matrix =
                DeviceReadWriteMatrix::new(context, &entries, 1, Some(gamma)).expect("matrix");
            let mut state = AddressFirst::new(
                context,
                matrix,
                context.upload(&inc).expect("increments"),
                &initial,
                &point,
                gamma,
            )
            .expect("address-first state");
            let mut claim = fr(0);
            for k in 0..addresses {
                for j in 0..cycles {
                    let v = val[k * cycles + j];
                    claim += eq[j] * ra[k * cycles + j] * (v + gamma * (v + inc[j]));
                }
            }
            for round in 0..LOG_K + LOG_T {
                let poly = state.round(claim).expect("round polynomial");
                for x in [fr(0), fr(1), fr(2), fr(3)] {
                    let mut expected = fr(0);
                    if addresses > 1 {
                        for k in 0..addresses / 2 {
                            for j in 0..cycles {
                                let low = 2 * k * cycles + j;
                                let high = low + cycles;
                                let r = ra[low] + x * (ra[high] - ra[low]);
                                let v = val[low] + x * (val[high] - val[low]);
                                expected += eq[j] * r * (v + gamma * (v + inc[j]));
                            }
                        }
                    } else {
                        for j in 0..cycles / 2 {
                            let r = ra[2 * j] + x * (ra[2 * j + 1] - ra[2 * j]);
                            let v = val[2 * j] + x * (val[2 * j + 1] - val[2 * j]);
                            let delta = inc[2 * j] + x * (inc[2 * j + 1] - inc[2 * j]);
                            let e = eq[2 * j] + x * (eq[2 * j + 1] - eq[2 * j]);
                            expected += e * r * (v + gamma * (v + delta));
                        }
                    }
                    assert_eq!(poly.evaluate(x), expected, "mode {mode}, round {round}");
                }
                let challenge = fr(if mode < 2 { mode } else { 53 + round as u64 });
                claim = poly.evaluate(challenge);
                state.bind(challenge).expect("bind");
                if addresses > 1 {
                    for table in [&mut ra, &mut val] {
                        *table = (0..addresses / 2 * cycles)
                            .map(|i| {
                                let low = (2 * (i / cycles)) * cycles + i % cycles;
                                table[low] + challenge * (table[low + cycles] - table[low])
                            })
                            .collect();
                    }
                    addresses /= 2;
                } else {
                    for table in [&mut ra, &mut val, &mut inc, &mut eq] {
                        *table = table
                            .chunks_exact(2)
                            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                            .collect();
                    }
                    cycles /= 2;
                }
            }
            assert_eq!(
                state.final_values().expect("final values"),
                [ra[0], ra[0], val[0], inc[0]]
            );
        }
    }
}
