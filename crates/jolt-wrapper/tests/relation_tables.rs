//! The 54 lookup-table gadgets against `LookupTableKind::evaluate_mle` on
//! random 128-coordinate points.
//!
//! `cargo nextest run -p jolt-wrapper --cargo-quiet --test relation_tables`

#![expect(clippy::unwrap_used)]

use jolt_field::{Field, Fr};
use jolt_lookup_tables::LookupTableKind;
use jolt_wrapper::relation::table_gadget_values;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn table_gadgets_match_native_mles() {
    let mut rng = StdRng::seed_from_u64(0x7ab1e);
    for _ in 0..200 {
        let point: Vec<Fr> = (0..128).map(|_| Fr::random(&mut rng)).collect();
        let values = table_gadget_values(&point).unwrap();
        let mismatches: Vec<String> = LookupTableKind::<64>::iter()
            .zip(&values)
            .filter(|(kind, value)| **value != kind.evaluate_mle::<Fr, Fr>(&point))
            .map(|(kind, _)| format!("{kind:?}"))
            .collect();
        assert!(mismatches.is_empty(), "{mismatches:?}");
    }
}
