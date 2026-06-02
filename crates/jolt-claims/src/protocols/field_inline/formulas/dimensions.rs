use serde::{Deserialize, Serialize};

pub const FIELD_REGISTERS_ADDRESS_BITS: usize = super::super::FIELD_REGISTERS_LOG_K;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineSumcheckSpec {
    pub rounds: usize,
    pub degree: usize,
}

impl FieldInlineSumcheckSpec {
    pub const fn boolean(rounds: usize, degree: usize) -> Self {
        Self { rounds, degree }
    }
}

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
}
