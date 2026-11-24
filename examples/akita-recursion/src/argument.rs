use core::fmt::{Formatter, Result as FmtResult};
use jolt_verifier::proof::PROOF_ARGUMENT_NEWTYPE;
use serde::de::{
    DeserializeSeed, Deserializer, EnumAccess, Error, MapAccess, SeqAccess, VariantAccess, Visitor,
};

/// Decoder wrapper that hands a proof's argument string over as one byte
/// string. Everything else, size hints included, is forwarded unchanged.
///
/// Only wrap a decoder that reads from a slice: such a decoder checks the
/// length the input announces against the bytes that remain before it lends
/// them, which is what makes the byte-string request safe on untrusted input.
pub(crate) struct ArgumentAsBytes<T>(pub(crate) T);

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for ArgumentAsBytes<S> {
    type Value = S::Value;

    fn deserialize<D: Deserializer<'de>>(self, inner: D) -> Result<Self::Value, D::Error> {
        self.0.deserialize(ArgumentAsBytes(inner))
    }
}

macro_rules! deserialize {
    ($($method:ident($($arg:ident: $ty:ty),*));* $(;)?) => {$(
        fn $method<V: Visitor<'de>>(self, $($arg: $ty,)* visitor: V) -> Result<V::Value, Self::Error> {
            self.0.$method($($arg,)* ArgumentAsBytes(visitor))
        }
    )*};
}

impl<'de, D: Deserializer<'de>> Deserializer<'de> for ArgumentAsBytes<D> {
    type Error = D::Error;
    deserialize! {
        deserialize_any(); deserialize_bool();
        deserialize_i8(); deserialize_i16(); deserialize_i32(); deserialize_i64(); deserialize_i128();
        deserialize_u8(); deserialize_u16(); deserialize_u32(); deserialize_u64(); deserialize_u128();
        deserialize_f32(); deserialize_f64(); deserialize_char();
        deserialize_str(); deserialize_string(); deserialize_bytes(); deserialize_byte_buf();
        deserialize_option(); deserialize_unit(); deserialize_unit_struct(name: &'static str);
        deserialize_seq();
        deserialize_tuple(len: usize); deserialize_tuple_struct(name: &'static str, len: usize);
        deserialize_map(); deserialize_struct(name: &'static str, fields: &'static [&'static str]);
        deserialize_enum(name: &'static str, variants: &'static [&'static str]);
        deserialize_identifier(); deserialize_ignored_any();
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        if name == PROOF_ARGUMENT_NEWTYPE {
            return self.0.deserialize_bytes(visitor);
        }
        self.0
            .deserialize_newtype_struct(name, ArgumentAsBytes(visitor))
    }

    fn is_human_readable(&self) -> bool {
        self.0.is_human_readable()
    }
}

macro_rules! visit {
    ($($method:ident($($arg:ident: $ty:ty),*));* $(;)?) => {$(
        fn $method<E: Error>(self, $($arg: $ty),*) -> Result<Self::Value, E> {
            self.0.$method($($arg),*)
        }
    )*};
}

impl<'de, V: Visitor<'de>> Visitor<'de> for ArgumentAsBytes<V> {
    type Value = V::Value;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        self.0.expecting(formatter)
    }

    visit! {
        visit_bool(v: bool); visit_i8(v: i8); visit_i16(v: i16); visit_i32(v: i32);
        visit_i64(v: i64); visit_i128(v: i128); visit_u8(v: u8); visit_u16(v: u16);
        visit_u32(v: u32); visit_u64(v: u64); visit_u128(v: u128);
        visit_f32(v: f32); visit_f64(v: f64); visit_char(v: char);
        visit_str(v: &str); visit_borrowed_str(v: &'de str); visit_string(v: String);
        visit_bytes(v: &[u8]); visit_borrowed_bytes(v: &'de [u8]); visit_byte_buf(v: Vec<u8>);
        visit_unit(); visit_none();
    }

    fn visit_some<D: Deserializer<'de>>(self, inner: D) -> Result<Self::Value, D::Error> {
        self.0.visit_some(ArgumentAsBytes(inner))
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(self, inner: D) -> Result<Self::Value, D::Error> {
        self.0.visit_newtype_struct(ArgumentAsBytes(inner))
    }

    fn visit_seq<S: SeqAccess<'de>>(self, inner: S) -> Result<Self::Value, S::Error> {
        self.0.visit_seq(ArgumentAsBytes(inner))
    }

    fn visit_map<M: MapAccess<'de>>(self, inner: M) -> Result<Self::Value, M::Error> {
        self.0.visit_map(ArgumentAsBytes(inner))
    }

    fn visit_enum<E: EnumAccess<'de>>(self, inner: E) -> Result<Self::Value, E::Error> {
        self.0.visit_enum(ArgumentAsBytes(inner))
    }
}

impl<'de, S: SeqAccess<'de>> SeqAccess<'de> for ArgumentAsBytes<S> {
    type Error = S::Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, Self::Error> {
        self.0.next_element_seed(ArgumentAsBytes(seed))
    }

    fn size_hint(&self) -> Option<usize> {
        self.0.size_hint()
    }
}

impl<'de, M: MapAccess<'de>> MapAccess<'de> for ArgumentAsBytes<M> {
    type Error = M::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, Self::Error> {
        self.0.next_key_seed(ArgumentAsBytes(seed))
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> Result<V::Value, Self::Error> {
        self.0.next_value_seed(ArgumentAsBytes(seed))
    }

    fn size_hint(&self) -> Option<usize> {
        self.0.size_hint()
    }
}

impl<'de, E: EnumAccess<'de>> EnumAccess<'de> for ArgumentAsBytes<E> {
    type Error = E::Error;
    type Variant = ArgumentAsBytes<E::Variant>;

    fn variant_seed<V: DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant), Self::Error> {
        let (value, inner) = self.0.variant_seed(ArgumentAsBytes(seed))?;
        Ok((value, ArgumentAsBytes(inner)))
    }
}

impl<'de, V: VariantAccess<'de>> VariantAccess<'de> for ArgumentAsBytes<V> {
    type Error = V::Error;

    fn unit_variant(self) -> Result<(), Self::Error> {
        self.0.unit_variant()
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<T::Value, Self::Error> {
        self.0.newtype_variant_seed(ArgumentAsBytes(seed))
    }

    fn tuple_variant<T: Visitor<'de>>(
        self,
        len: usize,
        visitor: T,
    ) -> Result<T::Value, Self::Error> {
        self.0.tuple_variant(len, ArgumentAsBytes(visitor))
    }

    fn struct_variant<T: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: T,
    ) -> Result<T::Value, Self::Error> {
        self.0.struct_variant(fields, ArgumentAsBytes(visitor))
    }
}
