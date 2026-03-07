//! Opaque tags identifying polynomials and sumcheck instances.
//!
//! Tags are `u64` values passed to [`jolt_ir::OpeningBinding`] to connect
//! symbolic expression variables to concrete Jolt polynomials and sumcheck
//! stages. `jolt-ir` treats them as opaque — all interpretation happens here.
//!
//! # Layout
//!
//! - [`poly`]: Polynomial tags — one per committed or virtual polynomial.
//! - [`sumcheck`]: Sumcheck instance tags — one per sumcheck invocation.

/// Polynomial identity tags.
///
/// Each constant identifies a distinct polynomial whose opening appears in
/// one or more sumcheck claim formulas.
pub mod poly {
    // Committed polynomials (owned, have evaluation tables).
    pub const RAM_INC: u64 = 100;
    pub const RD_INC: u64 = 101;
    pub const INSTRUCTION_RA: u64 = 200;

    // Virtual polynomials (derived during proving, no owned table).
    pub const RAM_READ_VALUE: u64 = 1000;
    pub const RAM_WRITE_VALUE: u64 = 1001;
    pub const RAM_RA: u64 = 1002;
    pub const RAM_VAL: u64 = 1003;
    pub const RAM_VAL_FINAL: u64 = 1004;
    pub const RAM_ADDRESS: u64 = 1005;
    pub const RAM_HAMMING_WEIGHT: u64 = 1006;

    pub const RD_WRITE_VALUE: u64 = 1100;
    pub const RS1_VALUE: u64 = 1101;
    pub const RS2_VALUE: u64 = 1102;
    pub const REGISTERS_VAL: u64 = 1103;
    pub const RD_WA: u64 = 1104;
    pub const RS1_RA: u64 = 1105;
    pub const RS2_RA: u64 = 1106;

    pub const LOOKUP_OUTPUT: u64 = 1200;
    pub const LEFT_LOOKUP_OPERAND: u64 = 1201;
    pub const RIGHT_LOOKUP_OPERAND: u64 = 1202;
    pub const LEFT_INSTRUCTION_INPUT: u64 = 1203;
    pub const RIGHT_INSTRUCTION_INPUT: u64 = 1204;

    pub const NEXT_PC: u64 = 1300;
    pub const NEXT_UNEXPANDED_PC: u64 = 1301;
    pub const NEXT_IS_VIRTUAL: u64 = 1302;
    pub const NEXT_IS_FIRST_IN_SEQUENCE: u64 = 1303;
    pub const NEXT_IS_NOOP: u64 = 1304;

    pub const UNIVARIATE_SKIP: u64 = 1400;
    pub const IO_MASK: u64 = 1401;

    /// Parameterized committed RA polynomial: `INSTRUCTION_RA_BASE + index`.
    #[inline]
    pub const fn instruction_ra(index: usize) -> u64 {
        INSTRUCTION_RA + index as u64
    }
}

/// Sumcheck instance tags.
///
/// Each constant identifies a sumcheck invocation. A polynomial may be opened
/// in multiple sumcheck instances (at different points), so the `(polynomial_tag,
/// sumcheck_tag)` pair uniquely identifies an opening.
pub mod sumcheck {
    pub const SPARTAN_OUTER: u64 = 1;
    pub const SPARTAN_PRODUCT_VIRTUAL: u64 = 2;
    pub const SHIFT: u64 = 3;
    pub const INSTRUCTION_INPUT_VIRTUAL: u64 = 4;
    pub const RAM_READ_WRITE_CHECKING: u64 = 10;
    pub const RAM_VAL_CHECK: u64 = 11;
    pub const RAM_RAF_EVALUATION: u64 = 12;
    pub const RAM_OUTPUT_CHECK: u64 = 13;
    pub const RAM_HAMMING_BOOLEANITY: u64 = 14;
    pub const RAM_RA_VIRTUAL: u64 = 15;
    pub const REGISTERS_READ_WRITE_CHECKING: u64 = 20;
    pub const REGISTERS_VAL_EVALUATION: u64 = 21;
    pub const REGISTERS_CLAIM_REDUCTION: u64 = 30;
    pub const INSTRUCTION_CLAIM_REDUCTION: u64 = 31;
    pub const INSTRUCTION_READ_RAF: u64 = 32;
    pub const INSTRUCTION_RA_VIRTUAL: u64 = 33;
    pub const HAMMING_WEIGHT_CLAIM_REDUCTION: u64 = 40;
    pub const INC_CLAIM_REDUCTION: u64 = 41;
    pub const RAM_RA_CLAIM_REDUCTION: u64 = 42;
    pub const ADVICE_CLAIM_REDUCTION: u64 = 50;
    pub const ADVICE_CLAIM_REDUCTION_CYCLE_PHASE: u64 = 51;
}
