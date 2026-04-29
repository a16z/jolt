//! Per-instruction test helpers.

use jolt_trace::JoltCycle;
use rand::prelude::*;

use crate::{InstructionLookupTable, LookupQuery, XLEN};

/// Fuzz-check that an instruction's `to_lookup_output` agrees with the
/// corresponding lookup table's `materialize_entry(to_lookup_index)`
/// across a batch of random cycles.
#[expect(clippy::unwrap_used)]
pub fn materialize_entry_test<
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
