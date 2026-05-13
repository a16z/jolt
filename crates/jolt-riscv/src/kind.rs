#[cfg(feature = "serialization")]
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JoltInstructionTag(pub u16);

macro_rules! define_source_instruction_kind {
    (
        instructions: [$($instr:ident => $marker:ident => $canonical_name:expr),* $(,)?]
    ) => {
        #[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
        /// Instruction kind decoded from program bytes before Jolt bytecode expansion.
        ///
        /// This includes ordinary RV64 instructions plus Jolt custom source
        /// opcodes, so source identity is the canonical namespaced string, not
        /// the compact final-bytecode tag.
        pub enum SourceInstructionKind {
            #[default]
            NoOp,
            Unimpl,
            $(
                $instr,
            )*
            Inline,
        }

        impl SourceInstructionKind {
            pub const ALL: &'static [Self] = &[
                Self::NoOp,
                Self::Unimpl,
                $(
                    Self::$instr,
                )*
                Self::Inline,
            ];

            pub const fn name(self) -> &'static str {
                match self {
                    Self::NoOp => "NoOp",
                    Self::Unimpl => "Unimpl",
                    $(
                        Self::$instr => stringify!($instr),
                    )*
                    Self::Inline => "Inline",
                }
            }

            pub const fn canonical_name(self) -> &'static str {
                match self {
                    Self::NoOp => "jolt.pseudo.noop",
                    Self::Unimpl => "jolt.pseudo.unimpl",
                    $(
                        Self::$instr => $canonical_name,
                    )*
                    Self::Inline => "jolt.inline.dispatch",
                }
            }

            pub fn from_canonical_name(name: &str) -> Option<Self> {
                match name {
                    "jolt.pseudo.noop" => Some(Self::NoOp),
                    "jolt.pseudo.unimpl" => Some(Self::Unimpl),
                    $(
                        $canonical_name => Some(Self::$instr),
                    )*
                    "jolt.inline.dispatch" => Some(Self::Inline),
                    _ => None,
                }
            }

            pub const fn expands_to_jolt(self) -> bool {
                !matches!(self, Self::NoOp | Self::Unimpl)
            }

            pub const fn has_side_effects(self) -> bool {
                matches!(
                    self,
                    Self::AdviceLB
                        | Self::AdviceLD
                        | Self::AdviceLH
                        | Self::AdviceLW
                        | Self::AMOADDD
                        | Self::AMOADDW
                        | Self::AMOANDD
                        | Self::AMOANDW
                        | Self::AMOMAXD
                        | Self::AMOMAXUD
                        | Self::AMOMAXUW
                        | Self::AMOMAXW
                        | Self::AMOMIND
                        | Self::AMOMINUD
                        | Self::AMOMINUW
                        | Self::AMOMINW
                        | Self::AMOORD
                        | Self::AMOORW
                        | Self::AMOSWAPD
                        | Self::AMOSWAPW
                        | Self::AMOXORD
                        | Self::AMOXORW
                        | Self::BEQ
                        | Self::BGE
                        | Self::BGEU
                        | Self::BLT
                        | Self::BLTU
                        | Self::BNE
                        | Self::CSRRS
                        | Self::CSRRW
                        | Self::EBREAK
                        | Self::ECALL
                        | Self::Inline
                        | Self::JAL
                        | Self::JALR
                        | Self::LB
                        | Self::LBU
                        | Self::LD
                        | Self::LH
                        | Self::LHU
                        | Self::LRD
                        | Self::LRW
                        | Self::LW
                        | Self::LWU
                        | Self::MRET
                        | Self::SB
                        | Self::SCD
                        | Self::SCW
                        | Self::SD
                        | Self::SH
                        | Self::SW
                        | Self::VirtualAdviceLoad
                        | Self::VirtualHostIO
                        | Self::VirtualSW
                )
            }
        }
    };
}

macro_rules! define_jolt_instruction_kind {
    (
        instructions: [$($instr:ident => $marker:ident => ($tag:expr, $canonical_name:expr)),* $(,)?]
    ) => {
        #[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
        #[repr(u16)]
        /// Final instruction kind admitted into Jolt bytecode and proving tables.
        ///
        /// This enum is intentionally smaller than [`SourceInstructionKind`].
        /// Its compact tag is a final-bytecode serialization/indexing identity,
        /// not a source ISA identity.
        pub enum JoltInstructionKind {
            #[default]
            NoOp,
            $(
                $instr,
            )*
        }

        impl SourceInstructionKind {
            pub const fn from_jolt_kind(kind: JoltInstructionKind) -> Option<Self> {
                match kind {
                    JoltInstructionKind::NoOp => Some(Self::NoOp),
                    $(
                        JoltInstructionKind::$instr => Some(Self::$instr),
                    )*
                }
            }

            pub const fn jolt_kind(self) -> Option<JoltInstructionKind> {
                match self {
                    Self::NoOp => Some(JoltInstructionKind::NoOp),
                    $(
                        Self::$instr => Some(JoltInstructionKind::$instr),
                    )*
                    _ => None,
                }
            }
        }

        impl JoltInstructionKind {
            pub const ALL: &'static [Self] = &[
                Self::NoOp,
                $(
                    Self::$instr,
                )*
            ];

            pub const fn name(self) -> &'static str {
                match self {
                    Self::NoOp => "NoOp",
                    $(
                        Self::$instr => stringify!($instr),
                    )*
                }
            }

            pub const fn canonical_name(self) -> &'static str {
                match self {
                    Self::NoOp => "jolt.pseudo.noop",
                    $(
                        Self::$instr => $canonical_name,
                    )*
                }
            }

            pub const fn tag(self) -> JoltInstructionTag {
                match self {
                    Self::NoOp => JoltInstructionTag(0x0000),
                    $(
                        Self::$instr => JoltInstructionTag($tag),
                    )*
                }
            }

            pub const fn from_tag(tag: JoltInstructionTag) -> Option<Self> {
                match tag.0 {
                    0x0000 => Some(Self::NoOp),
                    $(
                        $tag => Some(Self::$instr),
                    )*
                    _ => None,
                }
            }

            pub const fn has_side_effects(self) -> bool {
                matches!(
                    self,
                    Self::BEQ
                        | Self::BGE
                        | Self::BGEU
                        | Self::BLT
                        | Self::BLTU
                        | Self::BNE
                        | Self::FENCE
                        | Self::JAL
                        | Self::JALR
                        | Self::LD
                        | Self::SD
                        | Self::VirtualAdviceLoad
                        | Self::VirtualHostIO
                )
            }
        }
    };
}

crate::for_each_instruction_kind!(define_source_instruction_kind);
crate::for_each_jolt_instruction_kind!(define_jolt_instruction_kind);

#[cfg(feature = "serialization")]
impl CanonicalSerialize for SourceInstructionKind {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.canonical_name()
            .as_bytes()
            .to_vec()
            .serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.canonical_name()
            .as_bytes()
            .to_vec()
            .serialized_size(compress)
    }
}

#[cfg(feature = "serialization")]
impl CanonicalDeserialize for SourceInstructionKind {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let bytes = Vec::<u8>::deserialize_with_mode(reader, compress, validate)?;
        let name = std::str::from_utf8(&bytes).map_err(|_| SerializationError::InvalidData)?;
        Self::from_canonical_name(name).ok_or(SerializationError::InvalidData)
    }
}

#[cfg(feature = "serialization")]
impl Valid for SourceInstructionKind {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[cfg(feature = "serialization")]
impl CanonicalSerialize for JoltInstructionKind {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.tag().0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.tag().0.serialized_size(compress)
    }
}

#[cfg(feature = "serialization")]
impl CanonicalDeserialize for JoltInstructionKind {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u16::deserialize_with_mode(reader, compress, validate)?;
        Self::from_tag(JoltInstructionTag(value)).ok_or(SerializationError::InvalidData)
    }
}

#[cfg(feature = "serialization")]
impl Valid for JoltInstructionKind {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{JoltInstructionKind, JoltInstructionTag, SourceInstructionKind};
    use std::collections::HashSet;

    #[test]
    fn tags_are_stable_for_representative_final_rows() {
        assert_eq!(JoltInstructionKind::NoOp.tag(), JoltInstructionTag(0x0000));
        assert_eq!(JoltInstructionKind::ADD.tag(), JoltInstructionTag(0x0002));
        assert_eq!(
            JoltInstructionKind::VirtualHostIO.tag(),
            JoltInstructionTag(0x0068)
        );
        assert_eq!(
            JoltInstructionKind::VirtualXORROTW7.tag(),
            JoltInstructionTag(0x0088)
        );
    }

    #[test]
    fn final_tags_round_trip_and_are_unique() {
        let mut seen = HashSet::new();
        for kind in JoltInstructionKind::ALL {
            let tag = kind.tag();
            assert!(seen.insert(tag), "duplicate tag {tag:?} for {kind:?}");
            assert_eq!(JoltInstructionKind::from_tag(tag), Some(*kind));
        }
    }

    #[test]
    fn source_kinds_use_canonical_names_instead_of_tags() {
        assert_eq!(SourceInstructionKind::ADD.canonical_name(), "rv64.add");
        assert_eq!(
            SourceInstructionKind::Inline.canonical_name(),
            "jolt.inline.dispatch"
        );
        assert_eq!(
            SourceInstructionKind::from_canonical_name("rv64.addw"),
            Some(SourceInstructionKind::ADDW)
        );
    }

    #[test]
    fn source_canonical_names_are_unique_and_non_empty() {
        let mut seen = HashSet::new();
        for kind in SourceInstructionKind::ALL {
            let name = kind.canonical_name();
            assert!(!name.is_empty(), "empty canonical name for {kind:?}");
            assert!(seen.insert(name), "duplicate canonical name {name:?}");
        }
    }

    #[test]
    fn source_to_final_mapping_is_partial() {
        assert_eq!(
            SourceInstructionKind::ADD.jolt_kind(),
            Some(JoltInstructionKind::ADD)
        );
        assert_eq!(
            SourceInstructionKind::VirtualHostIO.jolt_kind(),
            Some(JoltInstructionKind::VirtualHostIO)
        );
        assert_eq!(SourceInstructionKind::ADDW.jolt_kind(), None);
        assert_eq!(SourceInstructionKind::Inline.jolt_kind(), None);
        assert_eq!(SourceInstructionKind::Unimpl.jolt_kind(), None);
    }

    #[test]
    fn final_canonical_names_are_unique_and_non_empty() {
        let mut seen = HashSet::new();
        for kind in JoltInstructionKind::ALL {
            let name = kind.canonical_name();
            assert!(!name.is_empty(), "empty canonical name for {kind:?}");
            assert!(seen.insert(name), "duplicate canonical name {name:?}");
        }
    }
}
