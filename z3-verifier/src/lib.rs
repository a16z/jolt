mod cpu_constraints;
mod virtual_sequences;

/// Bounds a single `Solver::check`. Every obligation in this crate discharges in
/// milliseconds, so a call that hits this is a blowup rather than a slow proof.
/// Surfacing it as `SatResult::Unknown` keeps the failure attributable to a named
/// test instead of hanging the run.
#[cfg(test)]
pub(crate) const Z3_TIMEOUT_MS: u32 = 30_000;

/// Fixed so a counterexample search is reproducible. Has no bearing on validity:
/// `Unsat` is the only passing verdict and does not depend on the seed.
#[cfg(test)]
pub(crate) const Z3_RANDOM_SEED: u32 = 42;

#[macro_export]
macro_rules! template_format {
    (FormatR) => {
        FormatR {
            rd: 1,
            rs1: 2,
            rs2: 3,
        }
    };
    (FormatI) => {
        FormatI {
            rd: 1,
            rs1: 2,
            imm: 1234,
        }
    };
    (FormatU) => {
        FormatU { rd: 1, imm: 1234 }
    };
    (FormatB) => {
        FormatB {
            rs1: 2,
            rs2: 3,
            imm: 1234,
        }
    };
    (FormatJ) => {
        FormatJ { rd: 1, imm: 1234 }
    };
    (FormatLoad) => {
        FormatLoad {
            rd: 1,
            rs1: 2,
            imm: 1234,
        }
    };
    (FormatS) => {
        FormatS {
            rs1: 2,
            rs2: 3,
            imm: 1234,
        }
    };
    (FormatAssert) => {
        FormatAssert { rs1: 2, imm: 1234 }
    };
    (FormatT) => {
        FormatT { rd: 1, rs1: 2 }
    };
    (FormatVirtualRightShiftI) => {
        FormatVirtualRightShiftI {
            rd: 1,
            rs1: 2,
            imm: 1234,
        }
    };
    (FormatVirtualRightShiftR) => {
        FormatVirtualRightShiftR {
            rd: 1,
            rs1: 2,
            rs2: 3,
        }
    };
    (FormatFence) => {
        FormatFence {}
    };
}
