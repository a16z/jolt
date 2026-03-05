//! Virtual assertion instructions used by the Jolt VM for constraint checking.

use crate::macros::define_instruction;
use crate::opcodes;

define_instruction!(
    /// Virtual ASSERT_EQ: returns 1 if operands are equal, 0 otherwise.
    AssertEq, opcodes::ASSERT_EQ, "ASSERT_EQ",
    |x, y| u64::from(x == y)
);

define_instruction!(
    /// Virtual ASSERT_LTE: returns 1 if `x <= y` (unsigned), 0 otherwise.
    AssertLte, opcodes::ASSERT_LTE, "ASSERT_LTE",
    |x, y| u64::from(x <= y)
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn assert_eq_basic() {
        assert_eq!(AssertEq.execute(5, 5), 1);
        assert_eq!(AssertEq.execute(5, 6), 0);
    }

    #[test]
    fn assert_lte_basic() {
        assert_eq!(AssertLte.execute(3, 5), 1);
        assert_eq!(AssertLte.execute(5, 5), 1);
        assert_eq!(AssertLte.execute(6, 5), 0);
    }
}
