use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::AssertValidUnsignedRemainder;
use jolt_trace::JoltCycle;

impl_lookup_table!(AssertValidUnsignedRemainder, Some(ValidUnsignedRemainder));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for AssertValidUnsignedRemainder<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            (self.0.rs2_val().unwrap_or(0) & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (remainder, divisor) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (divisor == 0 || remainder < divisor as u64).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_inputs_match_constraint_test, materialize_entry_test};

    #[test]
    fn materialize_entry_virtualassertvalidunsignedremainder() {
        materialize_entry_test!(AssertValidUnsignedRemainder, tracer::instruction::virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualassertvalidunsignedremainder() {
        instruction_inputs_match_constraint_test!(AssertValidUnsignedRemainder, tracer::instruction::virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder);
    }
}
