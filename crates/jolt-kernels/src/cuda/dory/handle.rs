use std::ops::{Add, Mul, Neg, Sub};

use ark_bn254::{G1Projective, G2Projective};
use ark_serialize::{Read, Write};
use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2};
use dory::primitives::arithmetic::Group as DoryGroup;
use dory::primitives::{Compress, DoryDeserialize, DorySerialize, SerializationError, Validate};

use crate::cuda::common::error::CudaError;

use super::arena::{self, Family, G1_WORDS, G2_WORDS};

macro_rules! device_handle {
    (
        $name:ident,
        $family:expr,
        $words:expr,
        $projective:ty,
        $ark:ident,
        $limbs:path,
        $point:path
    ) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            offset: u32,
        }

        impl $name {
            pub(super) fn store(point: &$projective) -> Self {
                let limbs = $limbs(point);
                let stored = arena::reserve($family, 1).and_then(|offset| {
                    arena::write($family, offset, &limbs)?;
                    u32::try_from(offset).map_err(|_| CudaError::InvariantViolation {
                        reason: "a Dory arena offset exceeded the handle's 32-bit width",
                    })
                });
                if let Ok(offset) = stored {
                    Self { offset }
                } else {
                    arena::poison(concat!(
                        "a ",
                        stringify!($name),
                        " handle could not be stored"
                    ));
                    Self { offset: 0 }
                }
            }

            pub(super) fn overwrite(&self, point: &$projective) {
                if arena::write($family, self.offset as usize, &$limbs(point)).is_err() {
                    arena::poison(concat!(
                        "a ",
                        stringify!($name),
                        " handle could not be overwritten"
                    ));
                }
            }

            pub(super) fn load(self) -> $projective {
                if let Ok(limbs) = arena::read($family, self.offset as usize, 1) {
                    $point(&limbs)
                } else {
                    arena::poison(concat!(
                        "a ",
                        stringify!($name),
                        " handle could not be read"
                    ));
                    <$projective>::default()
                }
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.offset == other.offset || self.load() == other.load()
            }
        }

        impl Add for $name {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self::store(&(self.load() + rhs.load()))
            }
        }

        impl Add<&Self> for $name {
            type Output = Self;
            fn add(self, rhs: &Self) -> Self {
                Self::store(&(self.load() + rhs.load()))
            }
        }

        impl Sub for $name {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Self::store(&(self.load() - rhs.load()))
            }
        }

        impl Sub<&Self> for $name {
            type Output = Self;
            fn sub(self, rhs: &Self) -> Self {
                Self::store(&(self.load() - rhs.load()))
            }
        }

        impl Neg for $name {
            type Output = Self;
            fn neg(self) -> Self {
                Self::store(&(-self.load()))
            }
        }

        impl Mul<$name> for ArkFr {
            type Output = $name;
            fn mul(self, rhs: $name) -> $name {
                $name::store(&(rhs.load() * self.0))
            }
        }

        impl Mul<&$name> for ArkFr {
            type Output = $name;
            fn mul(self, rhs: &$name) -> $name {
                $name::store(&(rhs.load() * self.0))
            }
        }

        impl DorySerialize for $name {
            fn serialize_with_mode<W: Write>(
                &self,
                writer: W,
                compress: Compress,
            ) -> Result<(), SerializationError> {
                $ark(self.load()).serialize_with_mode(writer, compress)
            }

            fn serialized_size(&self, compress: Compress) -> usize {
                $ark(self.load()).serialized_size(compress)
            }
        }

        impl DoryDeserialize for $name {
            fn deserialize_with_mode<R: Read>(
                reader: R,
                compress: Compress,
                validate: Validate,
            ) -> Result<Self, SerializationError> {
                let point = $ark::deserialize_with_mode(reader, compress, validate)?;
                Ok(Self::store(&point.0))
            }
        }

        impl DoryGroup for $name {
            type Scalar = ArkFr;

            fn identity() -> Self {
                Self::store(&<$projective>::default())
            }

            fn add(&self, rhs: &Self) -> Self {
                Self::store(&(self.load() + rhs.load()))
            }

            fn neg(&self) -> Self {
                Self::store(&(-self.load()))
            }

            fn scale(&self, k: &Self::Scalar) -> Self {
                Self::store(&(self.load() * k.0))
            }

            fn random() -> Self {
                arena::poison(
                    "resident handles cannot outlive their arena, so they cannot carry a setup",
                );
                Self::store(&<$projective>::default())
            }
        }
    };
}

device_handle!(
    DeviceG1,
    Family::G1,
    G1_WORDS,
    G1Projective,
    ArkG1,
    arena::g1_limbs,
    arena::g1_point
);

device_handle!(
    DeviceG2,
    Family::G2,
    G2_WORDS,
    G2Projective,
    ArkG2,
    arena::g2_limbs,
    arena::g2_point
);

const _: () = assert!(G1_WORDS == 12);
const _: () = assert!(G2_WORDS == 24);

macro_rules! device_handle_bulk {
    ($span:ident, $load_all:ident, $store_all:ident, $store_frozen:ident, $store_all_with:ident, $rehome:ident, $name:ident, $family:expr, $words:expr, $projective:ty, $limbs:path, $point:path) => {
        pub(super) fn $span(handles: &[$name]) -> Option<usize> {
            let first = handles.first()?.offset as usize;
            handles
                .iter()
                .enumerate()
                .all(|(index, handle)| handle.offset as usize == first + index)
                .then_some(first)
        }

        pub(super) fn $load_all(handles: &[$name]) -> Vec<$projective> {
            if handles.is_empty() {
                return Vec::new();
            }
            if let Some(first) = $span(handles) {
                if let Ok(limbs) = arena::read($family, first, handles.len()) {
                    return limbs.chunks_exact($words).map($point).collect();
                }
                arena::poison(concat!(
                    "a contiguous ",
                    stringify!($name),
                    " span could not be read"
                ));
            }
            handles.iter().map(|handle| handle.load()).collect()
        }

        pub(super) fn $store_all(points: &[$projective]) -> Vec<$name> {
            $store_all_with(points, false)
        }

        pub(super) fn $rehome(handles: &mut [$name], points: &[$projective]) {
            if handles.len() != points.len() {
                arena::poison(concat!(
                    "a ",
                    stringify!($name),
                    " rehoming was handed a mismatched point count"
                ));
                return;
            }
            for (handle, fresh) in handles.iter_mut().zip($store_all(points)) {
                *handle = fresh;
            }
        }

        pub(super) fn $store_frozen(points: &[$projective]) -> Vec<$name> {
            $store_all_with(points, true)
        }

        fn $store_all_with(points: &[$projective], freeze: bool) -> Vec<$name> {
            if points.is_empty() {
                return Vec::new();
            }
            let mut limbs = Vec::with_capacity(points.len() * $words);
            for point in points {
                limbs.extend_from_slice(&$limbs(point));
            }
            let stored = arena::reserve($family, points.len()).and_then(|offset| {
                arena::write($family, offset, &limbs)?;
                if freeze && offset == 0 {
                    arena::freeze($family, points.len(), &limbs)?;
                }
                match (
                    u32::try_from(offset),
                    u32::try_from(offset + points.len() - 1),
                ) {
                    (Ok(first), Ok(_)) => Ok(first),
                    _ => Err(CudaError::InvariantViolation {
                        reason: "a Dory arena offset exceeded the handle's 32-bit width",
                    }),
                }
            });
            if let Ok(first) = stored {
                (0..points.len())
                    .map(|index| $name {
                        offset: first + index as u32,
                    })
                    .collect()
            } else {
                arena::poison(concat!(
                    "a ",
                    stringify!($name),
                    " handle span could not be stored"
                ));
                vec![$name { offset: 0 }; points.len()]
            }
        }
    };
}

device_handle_bulk!(
    span,
    load_all,
    store_all,
    store_frozen,
    store_all_with,
    rehome,
    DeviceG1,
    Family::G1,
    G1_WORDS,
    G1Projective,
    arena::g1_limbs,
    arena::g1_point
);

device_handle_bulk!(
    span_g2,
    load_all_g2,
    store_all_g2,
    store_frozen_g2,
    store_all_with_g2,
    rehome_g2,
    DeviceG2,
    Family::G2,
    G2_WORDS,
    G2Projective,
    arena::g2_limbs,
    arena::g2_point
);

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::cuda::common::context::shared_context;

    fn scalar(rng: &mut ChaCha20Rng) -> ArkFr {
        ArkFr(ark_bn254::Fr::rand(rng))
    }

    #[test]
    fn device_g1_group_matches_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let guard = match arena::open(1_024, 16) {
            Ok(guard) => guard,
            Err(_) => return,
        };
        let mut rng = ChaCha20Rng::seed_from_u64(9_100);

        for _ in 0..16 {
            let left = G1Projective::rand(&mut rng);
            let right = G1Projective::rand(&mut rng);
            let weight = scalar(&mut rng);

            let a = DeviceG1::store(&left);
            let b = DeviceG1::store(&right);

            assert_eq!(a.load(), left, "the handle did not round-trip");
            assert_eq!((a + b).load(), left + right, "add diverged");
            assert_eq!((a - b).load(), left - right, "sub diverged");
            assert_eq!((-a).load(), -left, "neg diverged");
            assert_eq!(
                DoryGroup::scale(&a, &weight).load(),
                left * weight.0,
                "scale diverged"
            );
            assert_eq!((weight * a).load(), left * weight.0, "scalar mul diverged");
            assert_eq!(
                <DeviceG1 as DoryGroup>::identity().load(),
                G1Projective::default(),
                "identity diverged"
            );
            assert!(!arena::poisoned(), "the arena poisoned during valid use");
        }
        drop(guard);
    }

    #[test]
    fn device_g2_group_matches_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let guard = match arena::open(16, 1_024) {
            Ok(guard) => guard,
            Err(_) => return,
        };
        let mut rng = ChaCha20Rng::seed_from_u64(9_300);

        for _ in 0..16 {
            let left = G2Projective::rand(&mut rng);
            let right = G2Projective::rand(&mut rng);
            let weight = scalar(&mut rng);

            let a = DeviceG2::store(&left);
            let b = DeviceG2::store(&right);

            assert_eq!(a.load(), left, "the handle did not round-trip");
            assert_eq!((a + b).load(), left + right, "add diverged");
            assert_eq!((a - b).load(), left - right, "sub diverged");
            assert_eq!((-a).load(), -left, "neg diverged");
            assert_eq!(
                DoryGroup::scale(&a, &weight).load(),
                left * weight.0,
                "scale diverged"
            );
            assert_eq!((weight * a).load(), left * weight.0, "scalar mul diverged");
            assert!(!arena::poisoned(), "the arena poisoned during valid use");
        }
        drop(guard);
    }

    #[test]
    fn store_frozen_serves_reads_without_the_device() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(9_400);
        let setup: Vec<G2Projective> = (0..8).map(|_| G2Projective::rand(&mut rng)).collect();
        let mutable = G2Projective::rand(&mut rng);

        let guard = match arena::open(16, 16) {
            Ok(guard) => guard,
            Err(_) => return,
        };
        let frozen = store_frozen_g2(&setup);
        let live = DeviceG2::store(&mutable);

        assert_eq!(load_all_g2(&frozen), setup, "the frozen prefix diverged");
        assert_eq!(live.load(), mutable, "the live handle diverged");
        assert!(!arena::poisoned(), "the arena poisoned during valid use");

        frozen[0].overwrite(&mutable);
        assert!(
            arena::poisoned(),
            "writing into the frozen prefix must poison"
        );
        drop(guard);
    }

    #[test]
    fn random_poisons_instead_of_returning_identity() {
        if shared_context().is_none() {
            return;
        }
        for generated in [
            || {
                let _ = DeviceG1::random();
            },
            || {
                let _ = DeviceG2::random();
            },
        ] {
            let guard = match arena::open(16, 16) {
                Ok(guard) => guard,
                Err(_) => return,
            };
            assert!(!arena::poisoned(), "a fresh arena must not be poisoned");
            generated();
            assert!(
                arena::poisoned(),
                "random() must poison rather than hand back the identity"
            );
            drop(guard);
        }
    }

    #[test]
    fn device_g1_poisons_when_the_arena_is_closed() {
        if shared_context().is_none() {
            return;
        }
        {
            let _guard = match arena::open(4, 4) {
                Ok(guard) => guard,
                Err(_) => return,
            };
        }
        let mut rng = ChaCha20Rng::seed_from_u64(9_200);
        let _ = DeviceG1::store(&G1Projective::rand(&mut rng));
        assert!(
            arena::poisoned(),
            "storing into a closed arena must poison rather than return a wrong point"
        );
    }
}
