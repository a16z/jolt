use jolt_field::Field;

use crate::protocols::jolt::geometry::dimensions::JoltFormulaPointError;

pub const FIELD_REGISTERS_ADDRESS_BITS: usize = super::super::FIELD_REGISTERS_LOG_K;

// field_inline shares the protocol-agnostic crate-root sumcheck spec; it gains a
// `domain` field, always `BooleanHypercube` via `::boolean`.
pub use crate::SumcheckSpec as FieldInlineSumcheckSpec;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FieldRegistersTraceDimensions {
    log_t: usize,
}

impl FieldRegistersTraceDimensions {
    pub const fn new(log_t: usize) -> Self {
        Self { log_t }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn sumcheck(self, degree: usize) -> FieldInlineSumcheckSpec {
        FieldInlineSumcheckSpec::boolean(self.log_t, degree)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FieldRegistersReadWriteDimensions {
    log_t: usize,
    log_k: usize,
    phase1_num_rounds: usize,
    phase2_num_rounds: usize,
}

impl FieldRegistersReadWriteDimensions {
    pub const fn new(
        log_t: usize,
        log_k: usize,
        phase1_num_rounds: usize,
        phase2_num_rounds: usize,
    ) -> Self {
        Self {
            log_t,
            log_k,
            phase1_num_rounds,
            phase2_num_rounds,
        }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn log_k(self) -> usize {
        self.log_k
    }

    pub const fn phase1_num_rounds(self) -> usize {
        self.phase1_num_rounds
    }

    pub const fn phase2_num_rounds(self) -> usize {
        self.phase2_num_rounds
    }

    pub const fn read_write_sumcheck(self) -> FieldInlineSumcheckSpec {
        FieldInlineSumcheckSpec::boolean(self.log_t + self.log_k, 3)
    }

    pub fn read_write_opening_point<F: Field>(
        self,
        challenges: &[F],
    ) -> Result<FieldRegistersReadWriteOpeningPoint<F>, JoltFormulaPointError> {
        self.validate_phase_split()?;
        let expected = self.log_t + self.log_k;
        if challenges.len() != expected {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected,
                got: challenges.len(),
            });
        }

        let (phase1, rest) = challenges.split_at(self.phase1_num_rounds);
        let (phase2, rest) = rest.split_at(self.phase2_num_rounds);
        let (phase3_cycle, phase3_address) = rest.split_at(self.log_t - self.phase1_num_rounds);

        let r_cycle = phase3_cycle
            .iter()
            .rev()
            .copied()
            .chain(phase1.iter().rev().copied())
            .collect::<Vec<_>>();
        let r_address = phase3_address
            .iter()
            .rev()
            .copied()
            .chain(phase2.iter().rev().copied())
            .collect::<Vec<_>>();
        let opening_point = [r_address.as_slice(), r_cycle.as_slice()].concat();

        Ok(FieldRegistersReadWriteOpeningPoint {
            r_address,
            r_cycle,
            opening_point,
        })
    }

    const fn validate_phase_split(self) -> Result<(), JoltFormulaPointError> {
        if self.phase1_num_rounds > self.log_t || self.phase2_num_rounds > self.log_k {
            return Err(JoltFormulaPointError::InvalidReadWritePhaseSplit {
                phase1_num_rounds: self.phase1_num_rounds,
                log_t: self.log_t,
                phase2_num_rounds: self.phase2_num_rounds,
                log_k: self.log_k,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldRegistersReadWriteOpeningPoint<F: Field> {
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub opening_point: Vec<F>,
}
