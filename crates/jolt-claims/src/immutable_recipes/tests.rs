#![expect(clippy::unwrap_used, reason = "literal recipe fixtures must validate")]

use super::{find, ordered_fold, EvaluationError, Phase, RecipeError, Record, Table, Terms};
use crate::{Expr, Source, Term};
use jolt_field::{Prime128OffsetA7F7 as F, Ring, IMMUTABLE_RECIPE_MODULUS};
use std::cell::RefCell;

type Id = usize;
static SOURCES: [Source<Id, Id, Id>; 3] =
    [Source::Opening(4), Source::Challenge(5), Source::Derived(6)];

fn record<'a>(
    headers: &'a [u64],
    factors: &'a [u64],
    coefficients: &'a [u128],
) -> Result<Record<'a, Id, Id, Id>, RecipeError> {
    Record::new(
        &[1, 1],
        Phase::Output,
        headers,
        factors,
        coefficients,
        &SOURCES,
        &[],
    )
}

#[test]
fn both_views_obey_literal_callback_trace_even_after_zero() {
    let compact = record(&[3, 16, 34], &[0x0002_0001_0000, 2], &[0, 7, 2]).unwrap();
    let canonical = Expr {
        terms: vec![
            Term {
                coefficient: F::from_u64(0),
                factors: SOURCES.to_vec(),
            },
            Term {
                coefficient: F::from_u64(7),
                factors: vec![],
            },
            Term {
                coefficient: F::from_u64(2),
                factors: vec![Source::Derived(6), Source::Opening(4)],
            },
        ],
    };
    for terms in [
        Terms::Compact {
            record: &compact,
            term: 0,
            cursor: 0,
        },
        Terms::Canonical(canonical.terms.iter()),
    ] {
        let seen = RefCell::new(Vec::new());
        let value = ordered_fold(
            terms,
            false,
            |id| {
                seen.borrow_mut().push((0, *id));
                Ok::<_, ()>(F::from_u64(3))
            },
            |id| {
                seen.borrow_mut().push((1, *id));
                Ok(F::from_u64(5))
            },
            |id| {
                seen.borrow_mut().push((2, *id));
                Ok(F::from_u64(11))
            },
        )
        .unwrap();
        assert_eq!(value, F::from_u64(73));
        assert_eq!(*seen.borrow(), [(0, 4), (1, 5), (2, 6), (2, 6), (0, 4)]);
    }
}

#[test]
fn first_resolver_error_prevents_later_callbacks() {
    let compact = record(&[3], &[0x0002_0001_0000], &[0]).unwrap();
    let seen = RefCell::new(Vec::new());
    let result = ordered_fold(
        Terms::Compact {
            record: &compact,
            term: 0,
            cursor: 0,
        },
        false,
        |_| {
            seen.borrow_mut().push(0);
            Ok(F::from_u64(0))
        },
        |_| {
            seen.borrow_mut().push(1);
            Err("first")
        },
        |_| {
            seen.borrow_mut().push(2);
            Err("later")
        },
    );
    assert_eq!(result, Err(EvaluationError::Resolver("first")));
    assert_eq!(*seen.borrow(), [0, 1]);
}

#[test]
fn malformed_records_never_expose_a_view() {
    assert!(matches!(
        record(&[1], &[1 << 16], &[1]),
        Err(RecipeError::Structure)
    ));
    assert!(matches!(
        record(&[1], &[3], &[1]),
        Err(RecipeError::Identifier)
    ));
    assert!(matches!(
        record(&[13], &[0, 0, 0, 0], &[1]),
        Err(RecipeError::Structure)
    ));
    assert!(matches!(
        record(&[1 << 20], &[], &[1]),
        Err(RecipeError::Structure)
    ));
    assert!(matches!(
        record(&[1], &[0], &[IMMUTABLE_RECIPE_MODULUS]),
        Err(RecipeError::Coefficient)
    ));
    let records = [
        record(&[1], &[0], &[1]).unwrap(),
        record(&[1], &[0], &[1]).unwrap(),
    ];
    assert!(matches!(Table::new(&records), Err(RecipeError::Duplicate)));
}

#[test]
fn packed_factor_boundaries_and_unit_signs_keep_every_callback() {
    let coefficients = [1, IMMUTABLE_RECIPE_MODULUS - 1, 7];
    let compact = record(&[12, 17, 32], &[0, 0, 0, 0], &coefficients).unwrap();
    let mut count = 0;
    let result = ordered_fold(
        Terms::Compact {
            record: &compact,
            term: 0,
            cursor: 0,
        },
        true,
        |_| {
            count += 1;
            Ok::<_, ()>(F::from_u64(2))
        },
        |_| Err(()),
        |_| Err(()),
    )
    .unwrap();
    assert_eq!(count, 13);
    assert_eq!(result, F::from_u64(4101));
}

#[test]
fn native_field_never_visits_keys() {
    #[cfg(not(target_arch = "riscv64"))]
    {
        let mut visited = false;
        let table = Table::empty();
        let result = find::<F, Id, Id, Id>(&table, Phase::Input, false, |_| {
            visited = true;
            true
        });
        assert!(matches!(result, Ok(None)));
        assert!(!visited);
    }
}

#[test]
fn coefficient_import_rejects_modulus_without_reduction() {
    assert_eq!(F::immutable_recipe_import(IMMUTABLE_RECIPE_MODULUS), None);
    assert_eq!(
        F::immutable_recipe_import(17).and_then(|v| v.immutable_recipe_export()),
        Some(17)
    );
}

#[test]
fn full_shape_key_distinguishes_order_payloads_and_affine_flag() {
    use super::{KeySink, KeyVisit};
    use crate::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
    use crate::protocols::jolt::JoltVirtualPolynomial;
    use jolt_riscv::CircuitFlags;
    use std::collections::BTreeSet;

    struct Words(Vec<u64>);
    impl KeySink for Words {
        fn word(&mut self, value: u64) -> bool {
            self.0.push(value);
            true
        }
    }
    let address = JoltVirtualPolynomial::InstructionRa(7);
    let load = JoltVirtualPolynomial::OpFlags(CircuitFlags::Load);
    let shapes = [
        (12, vec![address, load], false),
        (13, vec![address, load], false),
        (12, vec![address], false),
        (12, vec![load, address], false),
        (
            12,
            vec![JoltVirtualPolynomial::InstructionRa(8), load],
            false,
        ),
        (
            12,
            vec![address, JoltVirtualPolynomial::OpFlags(CircuitFlags::Store)],
            false,
        ),
        (12, vec![address, load], true),
    ];
    let mut keys = BTreeSet::new();
    for (index, (log_t, variables, affine)) in shapes.into_iter().enumerate() {
        let shape = SpartanOuterDimensions::new(log_t, variables, affine).unwrap();
        let mut words = Words(Vec::new());
        assert!(shape.visit(&mut words));
        if index == 0 {
            assert_eq!(words.0, [12, 2, 25, 7, 36, 3, 0]);
        }
        assert!(keys.insert(words.0));
    }
    assert_eq!(keys.len(), 7);
}
