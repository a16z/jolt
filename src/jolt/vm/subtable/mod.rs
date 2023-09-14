use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use std::any::TypeId;

use crate::jolt::vm::subtable::{eq::EQSubtable, xor::XORSubtable};
use crate::jolt::vm::test_vm::TestSubtables;

#[enum_dispatch]
pub trait LassoSubtable<F: PrimeField>: 'static {
  fn subtable_id(&self) -> TypeId {
    TypeId::of::<Self>()
  }
  fn materialize(&self, M: usize) -> Vec<F>;
  fn evaluate_mle(&self, point: &[F]) -> F;
}

// TODO(moodlezoup): Consider replacing From<TypeId> and Into<usize> with
//     combined trait/function to_enum_index(subtable: &dyn LassoSubtable<F>) => usize
#[macro_export]
macro_rules! subtable_enum {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[repr(usize)]
        #[enum_dispatch(LassoSubtable<F>)]
        #[derive(EnumCountMacro, EnumIter)]
        pub enum $enum_name<F: PrimeField> { $($alias($struct)),+ }
        impl<F: PrimeField> From<TypeId> for $enum_name<F> {
          fn from(subtable_id: TypeId) -> Self {
            $(
              if subtable_id == TypeId::of::<$struct>() {
                $enum_name::from(<$struct>::new())
              } else
            )+
            { panic!("Unexpected subtable id") }
          }
        }

        impl<F: PrimeField> Into<usize> for $enum_name<F> {
          fn into(self) -> usize {
            unsafe { *<*const _>::from(&self).cast::<usize>() }
          }
        }
    };
}

pub mod eq;
pub mod xor;
