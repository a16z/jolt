use core::{fmt::{Formatter, Result as FmtResult}, marker::PhantomData, mem::size_of};
use bincode::error::DecodeError;
use serde::de::{
    DeserializeOwned, DeserializeSeed, Deserializer, EnumAccess, Error, MapAccess,
    SeqAccess, VariantAccess, Visitor,
};

// Bincode's serde sequence decoder forwards untrusted size hints without
// charging its byte limit for the collection allocation. Budget those hints
// separately, and hide them from visitors until their element type is known.
const ALLOCATION_BUDGET: usize = 16 * 1024 * 1024;

pub(super) fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<(T, usize), DecodeError> {
    let mut budget = ALLOCATION_BUDGET;
    bincode::serde::seed_decode_from_slice(
        Bounded { inner: PhantomData::<T>, budget: &mut budget },
        bytes,
        bincode::config::standard().with_limit::<ALLOCATION_BUDGET>(),
    )
}

struct Bounded<'a, T> {
    inner: T,
    budget: &'a mut usize,
}

fn charge<E: Error>(budget: &mut usize, bytes: usize) -> Result<(), E> {
    *budget = budget.checked_sub(bytes).ok_or_else(|| E::custom("record allocation limit"))?;
    Ok(())
}

fn collection<E: Error, T>(budget: &mut usize, count: usize) -> Result<(), E> {
    // A Vec with no size hint may grow geometrically and has a minimum capacity.
    let bytes = count.max(4).checked_mul(size_of::<T>().max(1))
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| E::custom("record allocation overflow"))?;
    charge(budget, bytes)
}

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for Bounded<'_, S> {
    type Value = S::Value;

    fn deserialize<D: Deserializer<'de>>(self, inner: D) -> Result<Self::Value, D::Error> {
        self.inner.deserialize(Bounded { inner, budget: self.budget })
    }
}

macro_rules! deserialize {
    ($($method:ident($($arg:ident: $ty:ty),*));* $(;)?) => {$(
        fn $method<V: Visitor<'de>>(self, $($arg: $ty,)* visitor: V) -> Result<V::Value, Self::Error> {
            self.inner.$method($($arg,)* Bounded { inner: visitor, budget: self.budget })
        }
    )*};
}

impl<'de, D: Deserializer<'de>> Deserializer<'de> for Bounded<'_, D> {
    type Error = D::Error;
    deserialize! {
        deserialize_any(); deserialize_bool();
        deserialize_i8(); deserialize_i16(); deserialize_i32(); deserialize_i64(); deserialize_i128();
        deserialize_u8(); deserialize_u16(); deserialize_u32(); deserialize_u64(); deserialize_u128();
        deserialize_f32(); deserialize_f64(); deserialize_char();
        deserialize_str(); deserialize_bytes(); deserialize_option();
        deserialize_unit(); deserialize_unit_struct(name: &'static str);
        deserialize_newtype_struct(name: &'static str); deserialize_seq();
        deserialize_tuple(len: usize); deserialize_tuple_struct(name: &'static str, len: usize);
        deserialize_map(); deserialize_struct(name: &'static str, fields: &'static [&'static str]);
        deserialize_enum(name: &'static str, variants: &'static [&'static str]);
        deserialize_identifier(); deserialize_ignored_any();
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.deserialize_bytes(visitor)
    }

    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        self.deserialize_str(visitor)
    }

    fn is_human_readable(&self) -> bool {
        self.inner.is_human_readable()
    }
}

macro_rules! visit {
    ($($method:ident($($arg:ident: $ty:ty),*));* $(;)?) => {$(
        fn $method<E: Error>(self, $($arg: $ty),*) -> Result<Self::Value, E> {
            self.inner.$method($($arg),*)
        }
    )*};
}

impl<'de, V: Visitor<'de>> Visitor<'de> for Bounded<'_, V> {
    type Value = V::Value;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        self.inner.expecting(formatter)
    }

    visit! {
        visit_bool(v: bool); visit_i8(v: i8); visit_i16(v: i16); visit_i32(v: i32);
        visit_i64(v: i64); visit_i128(v: i128); visit_u8(v: u8); visit_u16(v: u16);
        visit_u32(v: u32); visit_u64(v: u64); visit_u128(v: u128);
        visit_f32(v: f32); visit_f64(v: f64); visit_char(v: char); visit_unit(); visit_none();
    }

    fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
        charge::<E>(self.budget, value.len())?;
        self.inner.visit_str(value)
    }

    fn visit_borrowed_str<E: Error>(self, value: &'de str) -> Result<Self::Value, E> {
        charge::<E>(self.budget, value.len())?;
        self.inner.visit_borrowed_str(value)
    }

    fn visit_bytes<E: Error>(self, value: &[u8]) -> Result<Self::Value, E> {
        charge::<E>(self.budget, value.len())?;
        self.inner.visit_bytes(value)
    }

    fn visit_borrowed_bytes<E: Error>(self, value: &'de [u8]) -> Result<Self::Value, E> {
        charge::<E>(self.budget, value.len())?;
        self.inner.visit_borrowed_bytes(value)
    }

    fn visit_some<D: Deserializer<'de>>(self, inner: D) -> Result<Self::Value, D::Error> {
        self.inner.visit_some(Bounded { inner, budget: self.budget })
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(self, inner: D) -> Result<Self::Value, D::Error> {
        self.inner.visit_newtype_struct(Bounded { inner, budget: self.budget })
    }

    fn visit_seq<S: SeqAccess<'de>>(self, inner: S) -> Result<Self::Value, S::Error> {
        let count = inner.size_hint().ok_or_else(|| S::Error::custom("missing sequence length"))?;
        self.inner.visit_seq(Sequence { inner, budget: self.budget, count: Some(count) })
    }

    fn visit_map<M: MapAccess<'de>>(self, inner: M) -> Result<Self::Value, M::Error> {
        let count = inner.size_hint().ok_or_else(|| M::Error::custom("missing map length"))?;
        // Include per-entry bookkeeping, in addition to key/value storage below.
        collection::<M::Error, [usize; 8]>(self.budget, count)?;
        self.inner.visit_map(Bounded { inner, budget: self.budget })
    }

    fn visit_enum<E: EnumAccess<'de>>(self, inner: E) -> Result<Self::Value, E::Error> {
        self.inner.visit_enum(Bounded { inner, budget: self.budget })
    }
}

struct Sequence<'a, S> {
    inner: S,
    budget: &'a mut usize,
    count: Option<usize>,
}

impl<'de, S: SeqAccess<'de>> SeqAccess<'de> for Sequence<'_, S> {
    type Error = S::Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error> {
        if let Some(count) = self.count.take() {
            collection::<Self::Error, T::Value>(self.budget, count)?;
        }
        self.inner.next_element_seed(Bounded { inner: seed, budget: self.budget })
    }

    fn size_hint(&self) -> Option<usize> {
        Some(0)
    }
}

impl<'de, M: MapAccess<'de>> MapAccess<'de> for Bounded<'_, M> {
    type Error = M::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error> {
        charge::<Self::Error>(self.budget, size_of::<K::Value>())?;
        self.inner.next_key_seed(Bounded { inner: seed, budget: self.budget })
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, Self::Error> {
        charge::<Self::Error>(self.budget, size_of::<V::Value>())?;
        self.inner.next_value_seed(Bounded { inner: seed, budget: self.budget })
    }

    fn size_hint(&self) -> Option<usize> {
        Some(0)
    }
}

impl<'a, 'de, E: EnumAccess<'de>> EnumAccess<'de> for Bounded<'a, E> {
    type Error = E::Error;
    type Variant = Bounded<'a, E::Variant>;

    fn variant_seed<V: DeserializeSeed<'de>>(self, seed: V) -> Result<(V::Value, Self::Variant), Self::Error> {
        let (value, inner) = self.inner.variant_seed(Bounded { inner: seed, budget: &mut *self.budget })?;
        Ok((value, Bounded { inner, budget: self.budget }))
    }
}

impl<'de, V: VariantAccess<'de>> VariantAccess<'de> for Bounded<'_, V> {
    type Error = V::Error;

    fn unit_variant(self) -> Result<(), Self::Error> {
        self.inner.unit_variant()
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, Self::Error> {
        self.inner.newtype_variant_seed(Bounded { inner: seed, budget: self.budget })
    }

    fn tuple_variant<T: Visitor<'de>>(self, len: usize, visitor: T) -> Result<T::Value, Self::Error> {
        self.inner.tuple_variant(len, Bounded { inner: visitor, budget: self.budget })
    }

    fn struct_variant<T: Visitor<'de>>(self, fields: &'static [&'static str], visitor: T) -> Result<T::Value, Self::Error> {
        self.inner.struct_variant(fields, Bounded { inner: visitor, budget: self.budget })
    }
}
