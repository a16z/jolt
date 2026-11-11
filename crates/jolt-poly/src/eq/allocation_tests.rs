use super::EqPolynomial;
use crate::{eq_index, Budget, PointRef, TensorEqTable};
use jolt_field::{impl_ring_ops, Field, Fr, Ring};
use jolt_utils::alloc::{try_zeroed_vec, zeroed_vec, ZeroAllocError};
use num_traits::Zero;
use rand_core::RngCore;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::num::NonZeroU8;

/// This field's representation deliberately forbids all-zero bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct NonZeroRepr {
    tag: NonZeroU8,
    value: Fr,
}

impl NonZeroRepr {
    fn new(value: Fr) -> Self {
        Self {
            tag: NonZeroU8::new(1).unwrap(),
            value,
        }
    }
}

impl Default for NonZeroRepr {
    fn default() -> Self {
        Self::new(Fr::zero())
    }
}

impl Display for NonZeroRepr {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Display::fmt(&self.value, f)
    }
}

impl_ring_ops!(impl[] NonZeroRepr {
    add(a, b): Self::new(a.value + b.value),
    sub(a, b): Self::new(a.value - b.value),
    mul(a, b): Self::new(a.value * b.value),
    neg(a): Self::new(-a.value),
    zero: Self::default(),
    one: Self::new(Fr::from_u64(1)),
});

impl Ring for NonZeroRepr {
    fn from_u64(v: u64) -> Self {
        Self::new(Fr::from_u64(v))
    }
    fn from_i64(v: i64) -> Self {
        Self::new(Fr::from_i64(v))
    }
    fn from_u128(v: u128) -> Self {
        Self::new(Fr::from_u128(v))
    }
    fn from_i128(v: i128) -> Self {
        Self::new(Fr::from_i128(v))
    }
}

impl Field for NonZeroRepr {
    fn inverse(&self) -> Option<Self> {
        self.value.inverse().map(Self::new)
    }
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new(Fr::random(rng))
    }
}

fn generic_callers<F: Field>(r: &[F]) {
    let point = PointRef::low_to_high(r);
    let table = EqPolynomial::<F>::evals_parallel(&point, None);
    let tensor =
        TensorEqTable::<F>::new_with_scaling_and_split(&point, F::from_u64(1), r.len() / 2);
    for (i, value) in table.iter().enumerate() {
        assert_eq!(*value, tensor.evaluate_index(i));
    }
    assert_eq!(table.iter().copied().sum::<F>(), F::from_u64(1));
    let budgeted = EqPolynomial::<F>::evals_in(
        &point,
        None,
        Budget {
            max_bytes: usize::MAX,
        },
    )
    .unwrap();
    for (i, value) in budgeted.iter().enumerate() {
        assert_eq!(*value, eq_index(&point, i as u128));
    }
}

#[test]
fn non_marker_fields_initialize_values_safely() {
    let zeros = NonZeroRepr::zeroed_vec(8);
    assert!(zeros.iter().all(Zero::is_zero));
    assert_eq!(NonZeroRepr::try_zeroed_vec(8).unwrap(), zeros);
    assert_eq!(NonZeroRepr::zeroed_vec(0), [] as [NonZeroRepr; 0]);
    assert!(NonZeroRepr::try_zeroed_vec(usize::MAX).is_err());
    generic_callers(&[NonZeroRepr::from_u64(2), NonZeroRepr::from_u64(3)]);
    generic_callers::<NonZeroRepr>(&[]);
}

#[test]
fn zeroed_allocations_distinguish_overflow_from_empty_requests() {
    assert!(matches!(
        try_zeroed_vec::<Fr>(usize::MAX),
        Err(ZeroAllocError::CapacityOverflow)
    ));
    assert!(std::panic::catch_unwind(|| zeroed_vec::<Fr>(usize::MAX)).is_err());
    assert_eq!(try_zeroed_vec::<Fr>(0), Ok(Vec::new()));
}
