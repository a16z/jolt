// Raw traces predate a feature-independent INLINE tag. Preserve both
// encodings; new catalog tags reserve the entire range in either build.
pub(super) const INLINE_CYCLE_TAG: u32 = if cfg!(feature = "field-inline") {
    164
} else {
    154
};

// Serde's derive uses declaration order for binary enum tags. Explicit tags
// keep raw traces readable when the instruction catalog grows.
macro_rules! impl_trace_serde {
    (
        $name:ident {
            units: [$($unit:ident => $unit_tag:expr),* $(,)?],
            payloads: [$($(#[$meta:meta])* $variant:ident => $tag:expr),* $(,)?]
        }
    ) => {
        impl ::serde::Serialize for $name {
            fn serialize<S: ::serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                match self {
                    $(Self::$unit => serializer.serialize_unit_variant(
                        stringify!($name), $unit_tag, stringify!($unit)
                    ),)*
                    $($(#[$meta])* Self::$variant(value) => serializer.serialize_newtype_variant(
                        stringify!($name), $tag, stringify!($variant), value
                    ),)*
                }
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $name {
            fn deserialize<D: ::serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                enum Variant {
                    $($unit,)*
                    $($(#[$meta])* $variant,)*
                }

                const VARIANTS: &[&str] = &[
                    $(stringify!($unit),)*
                    $($(#[$meta])* stringify!($variant),)*
                ];

                impl<'de> ::serde::Deserialize<'de> for Variant {
                    fn deserialize<D: ::serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                        struct TagVisitor;

                        impl ::serde::de::Visitor<'_> for TagVisitor {
                            type Value = Variant;

                            fn expecting(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                                f.write_str(concat!(stringify!($name), " variant"))
                            }

                            fn visit_u64<E: ::serde::de::Error>(self, tag: u64) -> Result<Variant, E> {
                                match tag {
                                    $(x if x == u64::from($unit_tag as u32) => Ok(Variant::$unit),)*
                                    $($(#[$meta])* x if x == u64::from($tag as u32) => Ok(Variant::$variant),)*
                                    _ => Err(E::invalid_value(::serde::de::Unexpected::Unsigned(tag), &self)),
                                }
                            }

                            fn visit_str<E: ::serde::de::Error>(self, tag: &str) -> Result<Variant, E> {
                                match tag {
                                    $(stringify!($unit) => Ok(Variant::$unit),)*
                                    $($(#[$meta])* stringify!($variant) => Ok(Variant::$variant),)*
                                    _ => Err(E::unknown_variant(tag, VARIANTS)),
                                }
                            }

                            fn visit_bytes<E: ::serde::de::Error>(self, tag: &[u8]) -> Result<Variant, E> {
                                match ::std::str::from_utf8(tag) {
                                    Ok(tag) => self.visit_str(tag),
                                    Err(_) => Err(E::invalid_value(::serde::de::Unexpected::Bytes(tag), &self)),
                                }
                            }
                        }

                        deserializer.deserialize_identifier(TagVisitor)
                    }
                }

                struct EnumVisitor;

                impl<'de> ::serde::de::Visitor<'de> for EnumVisitor {
                    type Value = $name;

                    fn expecting(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                        f.write_str(concat!("enum ", stringify!($name)))
                    }

                    fn visit_enum<A: ::serde::de::EnumAccess<'de>>(self, value: A) -> Result<$name, A::Error> {
                        let (variant, payload) = value.variant::<Variant>()?;
                        match variant {
                            $(Variant::$unit => {
                                ::serde::de::VariantAccess::unit_variant(payload)?;
                                Ok($name::$unit)
                            },)*
                            $($(#[$meta])* Variant::$variant =>
                                ::serde::de::VariantAccess::newtype_variant(payload).map($name::$variant),)*
                        }
                    }
                }

                deserializer.deserialize_enum(stringify!($name), VARIANTS, EnumVisitor)
            }
        }
    };
}

pub(super) use impl_trace_serde;

#[cfg(test)]
mod tests {
    use super::super::{Cycle, Instruction};
    use blake3::Hasher;
    use serde::{de::DeserializeOwned, Serialize};
    use std::fmt::Debug;
    use strum::IntoEnumIterator;

    fn legacy_digest<T: Serialize + DeserializeOwned + PartialEq + Debug>(
        items: impl Iterator<Item = T>,
        cutoff: u32,
    ) -> String {
        let mut hash = Hasher::new();
        for item in items {
            let bytes = postcard::to_stdvec(&item).unwrap();
            let (tag, _) = postcard::take_from_bytes::<u32>(&bytes).unwrap();
            assert_eq!(postcard::from_bytes::<T>(&bytes).unwrap(), item);
            let json = serde_json::to_vec(&item).unwrap();
            assert_eq!(serde_json::from_slice::<T>(&json).unwrap(), item);
            if tag < cutoff {
                hash.update(&(bytes.len() as u64).to_le_bytes());
                hash.update(&bytes);
                hash.update(&(json.len() as u64).to_le_bytes());
                hash.update(&json);
            }
        }
        hash.finalize().to_hex().to_string()
    }

    #[test]
    fn raw_trace_wire_compatibility() {
        // Frozen from main 9407f93: all legacy variants' Postcard and JSON encodings.
        let (cycle, instruction) = if cfg!(feature = "field-inline") {
            (
                "bcd765625cd35af25ae03dfb98f967544c2610107ac4cbe7804631deb962ebda",
                "93c9ea8fc20e339f82c659078ccb9a6bf680951a715c46cedb136e584fb3dd54",
            )
        } else {
            (
                "5bc69bf942ba2d2646314433872d75a7e93893cca109ee364c2533d7258f9358",
                "3483564131cf1602c4f8e08b16ce53592786d70a5a26f4a730ed455b5efcfe13",
            )
        };
        assert_eq!(legacy_digest(Cycle::iter(), 165), cycle);
        assert_eq!(legacy_digest(Instruction::iter(), 166), instruction);

        for (value, tag) in [
            (Cycle::VirtualXORROT3(Default::default()), 165),
            (Cycle::VirtualXORROT61(Default::default()), 185),
        ] {
            let bytes = postcard::to_stdvec(&value).unwrap();
            assert_eq!(postcard::take_from_bytes::<u32>(&bytes).unwrap().0, tag);
        }
        for (value, tag) in [
            (Instruction::VirtualXORROT3(Default::default()), 166),
            (Instruction::VirtualXORROT61(Default::default()), 186),
        ] {
            let bytes = postcard::to_stdvec(&value).unwrap();
            assert_eq!(postcard::take_from_bytes::<u32>(&bytes).unwrap().0, tag);
        }
        assert!(postcard::from_bytes::<Cycle>(&[0x90, 0x4e]).is_err());
        assert!(postcard::from_bytes::<Instruction>(&[0x90, 0x4e]).is_err());
    }
}
