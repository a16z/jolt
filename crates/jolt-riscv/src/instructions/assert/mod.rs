//! Virtual assert instructions used inside Jolt's tracer-emitted virtual sequences.

pub mod assert_eq;
pub mod assert_halfword_alignment;
pub mod assert_lte;
pub mod assert_mulu_no_overflow;
pub mod assert_valid_div0;
pub mod assert_valid_unsigned_remainder;
pub mod assert_word_alignment;

pub use assert_eq::AssertEq;
pub use assert_halfword_alignment::AssertHalfwordAlignment;
pub use assert_lte::AssertLte;
pub use assert_mulu_no_overflow::AssertMulUNoOverflow;
pub use assert_valid_div0::AssertValidDiv0;
pub use assert_valid_unsigned_remainder::AssertValidUnsignedRemainder;
pub use assert_word_alignment::AssertWordAlignment;
