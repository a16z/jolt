//! BN254 Fr coprocessor instructions — no RV lookup tables.
//!
//! FR ops (FieldOp / FieldAssertEq / FieldMov / FieldSLL*) are handled by
//! the FR Twist sumcheck machinery in the prover, not the RV instruction
//! lookup path. The `InstructionLookupTable` impls return `None` so that
//! generic dispatch over `Instruction` (via `with_isa_struct!`) compiles
//! cleanly across FR variants without special-casing each call site.

use crate::traits::impl_lookup_table;
use jolt_trace::instructions::{
    FieldAdd, FieldAssertEq, FieldInv, FieldMov, FieldMul, FieldSLL128, FieldSLL192, FieldSLL64,
    FieldSub,
};

impl_lookup_table!(FieldMul, None);
impl_lookup_table!(FieldAdd, None);
impl_lookup_table!(FieldSub, None);
impl_lookup_table!(FieldInv, None);
impl_lookup_table!(FieldAssertEq, None);
impl_lookup_table!(FieldMov, None);
impl_lookup_table!(FieldSLL64, None);
impl_lookup_table!(FieldSLL128, None);
impl_lookup_table!(FieldSLL192, None);
