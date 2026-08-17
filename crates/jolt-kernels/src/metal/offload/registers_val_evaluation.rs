//! Metal offload seams for the stage-5 registers value-evaluation kernel:
//! ready-state construction from device readbacks and the device-resident
//! round state machine.

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_field::Field;
use jolt_poly::{EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckError;

use crate::optimized::registers_val_evaluation::{IncSource, ValEvaluationKernel, WaState};
use crate::optimized::support::SplitLt;
use crate::KernelError;

impl<F: Field> ValEvaluationKernel<F> {
    pub(crate) fn new_ready(
        inc: Vec<F>,
        rd: Vec<Option<u8>>,
        r_address: &[F],
        r_cycle: &[F],
    ) -> Result<Self, KernelError<F>> {
        let cycles = inc.len();
        if cycles < 2 || !cycles.is_power_of_two() || r_cycle.len() != cycles.ilog2() as usize {
            return Err(KernelError::InvariantViolation {
                reason: "registers value ready state has inconsistent cycle geometry",
            });
        }
        if rd.len() != cycles || r_address.len() != REGISTER_ADDRESS_BITS {
            return Err(KernelError::InvariantViolation {
                reason: "registers value ready state has inconsistent address geometry",
            });
        }
        if rd
            .iter()
            .flatten()
            .any(|&index| index as usize >= 1 << REGISTER_ADDRESS_BITS)
        {
            return Err(KernelError::InvariantViolation {
                reason: "registers value ready state has an invalid register index",
            });
        }
        Ok(Self {
            rounds: r_cycle.len(),
            inc: IncSource::Ready(Polynomial::new(inc)),
            wa: WaState::Indices {
                rd,
                eq_address: EqPolynomial::<F>::evals(r_address, None),
            },
            lt: SplitLt::new(r_cycle),
            rounds_bound: 0,
        })
    }

    pub(crate) fn new_offloaded(r_cycle: &[F]) -> Self {
        Self {
            rounds: r_cycle.len(),
            inc: IncSource::Offloaded,
            wa: WaState::Offloaded,
            lt: SplitLt::new(r_cycle),
            rounds_bound: 0,
        }
    }

    pub(crate) fn metal_bind_offloaded(&mut self, challenge: F) -> Result<&[F], SumcheckError<F>> {
        if !matches!(&self.inc, IncSource::Offloaded) || !matches!(&self.wa, WaState::Offloaded) {
            return Err(SumcheckError::ComputeBackend {
                backend: "metal",
                message: "registers value CPU tail was restored before the device bind".to_owned(),
            });
        }
        self.lt.bind(challenge);
        self.rounds_bound += 1;
        self.lt
            .split_lo()
            .ok_or_else(|| SumcheckError::ComputeBackend {
                backend: "metal",
                message: "registers value Metal prefix crossed the split-LT boundary".to_owned(),
            })
    }

    pub(crate) fn metal_message(
        &self,
        evals: [F; 3],
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if !matches!(&self.inc, IncSource::Offloaded) || !matches!(&self.wa, WaState::Offloaded) {
            return Err(SumcheckError::ComputeBackend {
                backend: "metal",
                message: "registers value device message arrived after CPU restoration".to_owned(),
            });
        }
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    pub(crate) fn metal_restore_dense(&mut self, rows: &[[F; 2]]) -> Result<(), SumcheckError<F>> {
        let remaining = self.rounds.checked_sub(self.rounds_bound).ok_or_else(|| {
            SumcheckError::ComputeBackend {
                backend: "metal",
                message: "registers value device bound too many rounds".to_owned(),
            }
        })?;
        let shift = u32::try_from(remaining).map_err(|_| SumcheckError::ComputeBackend {
            backend: "metal",
            message: "registers value CPU-tail length overflow".to_owned(),
        })?;
        let expected = 1usize
            .checked_shl(shift)
            .ok_or_else(|| SumcheckError::ComputeBackend {
                backend: "metal",
                message: "registers value CPU-tail length overflow".to_owned(),
            })?;
        if rows.len() != expected
            || self.lt.current_len() != expected
            || !matches!(&self.inc, IncSource::Offloaded)
            || !matches!(&self.wa, WaState::Offloaded)
        {
            return Err(SumcheckError::ComputeBackend {
                backend: "metal",
                message: "registers value CPU-tail state does not match the bound device state"
                    .to_owned(),
            });
        }
        self.inc = IncSource::Ready(Polynomial::new(rows.iter().map(|row| row[0]).collect()));
        self.wa = WaState::Dense(rows.iter().map(|row| row[1]).collect());
        Ok(())
    }
}
