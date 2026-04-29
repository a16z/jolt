//! Per-instruction test helpers.

use jolt_trace::JoltCycle;
use rand::prelude::*;

use crate::{InstructionLookupTable, LookupQuery, XLEN};

/// Internal helper for [`materialize_entry_test!`]. Use the macro at call sites
/// — it picks up the verbose `Foo<RISCVCycle<TracerType>>` / `RISCVCycle<TracerType>`
/// type pair from a Jolt struct ident and a tracer instruction path.
#[doc(hidden)]
#[expect(clippy::unwrap_used)]
pub fn materialize_entry_test_fn<
    T: InstructionLookupTable<XLEN> + LookupQuery<XLEN> + From<C> + core::fmt::Debug,
    C: JoltCycle,
>() {
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10_000 {
        let cycle: T = T::from(C::random(&mut rng));
        let table = cycle.lookup_table().unwrap();
        assert_eq!(
            cycle.to_lookup_output(),
            table.materialize_entry(cycle.to_lookup_index()),
            "{cycle:?}",
        );
    }
}

/// Fuzz-check that an instruction's `to_lookup_output` agrees with the
/// corresponding lookup table's `materialize_entry(to_lookup_index)` across a
/// batch of random cycles. Pass the Jolt instruction newtype and the tracer
/// instruction path; the macro builds the `Foo<RISCVCycle<TracerType>>` /
/// `RISCVCycle<TracerType>` type pair.
///
/// ```ignore
/// materialize_entry_test!(Add, tracer::instruction::add::ADD);
/// ```
#[macro_export]
macro_rules! materialize_entry_test {
    ($jolt:ident, $tracer:path $(,)?) => {
        $crate::instructions::test::materialize_entry_test_fn::<
            $jolt<tracer::instruction::RISCVCycle<$tracer>>,
            tracer::instruction::RISCVCycle<$tracer>,
        >()
    };
}
