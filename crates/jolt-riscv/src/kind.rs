#[cfg(feature = "serialization")]
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JoltInstructionTag(pub u16);

macro_rules! define_instruction_kind {
    (
        instructions: [$($instr:ident => ($tag:expr, $canonical_name:expr)),* $(,)?]
    ) => {
        #[derive(
            Default,
            Debug,
            Clone,
            Copy,
            PartialEq,
            Eq,
            Hash,
        )]
        #[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
        #[repr(u16)]
        /// Instruction kind decoded from program bytes before Jolt bytecode expansion.
        ///
        /// This includes ordinary RV64 instructions plus Jolt custom source
        /// opcodes, so it is intentionally named "source" rather than "RISC-V".
        pub enum SourceInstructionKind {
            #[default]
            NoOp,
            Unimpl,
            $(
                $instr,
            )*
            Inline,
        }

        #[derive(
            Default,
            Debug,
            Clone,
            Copy,
            PartialEq,
            Eq,
            Hash,
        )]
        #[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
        #[repr(u16)]
        pub enum JoltInstructionKind {
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

            pub const fn tag(self) -> JoltInstructionTag {
                match self {
                    Self::NoOp => JoltInstructionTag(0x0000),
                    Self::Unimpl => JoltInstructionTag(0x0001),
                    $(
                        Self::$instr => JoltInstructionTag($tag),
                    )*
                    Self::Inline => JoltInstructionTag(0x0089),
                }
            }

            pub const fn from_tag(tag: JoltInstructionTag) -> Option<Self> {
                match tag.0 {
                    0x0000 => Some(Self::NoOp),
                    0x0001 => Some(Self::Unimpl),
                    $(
                        $tag => Some(Self::$instr),
                    )*
                    0x0089 => Some(Self::Inline),
                    _ => None,
                }
            }

            pub const fn from_jolt_kind(kind: JoltInstructionKind) -> Option<Self> {
                match kind {
                    JoltInstructionKind::NoOp => Some(Self::NoOp),
                    JoltInstructionKind::Unimpl => Some(Self::Unimpl),
                    $(
                        JoltInstructionKind::$instr => Some(Self::$instr),
                    )*
                    JoltInstructionKind::Inline => Some(Self::Inline),
                }
            }

            pub const fn jolt_kind(self) -> JoltInstructionKind {
                match self {
                    Self::NoOp => JoltInstructionKind::NoOp,
                    Self::Unimpl => JoltInstructionKind::Unimpl,
                    $(
                        Self::$instr => JoltInstructionKind::$instr,
                    )*
                    Self::Inline => JoltInstructionKind::Inline,
                }
            }

            pub const fn expands_to_jolt(self) -> bool {
                !matches!(self, Self::NoOp | Self::Unimpl)
            }
        }

        #[cfg(feature = "serialization")]
        impl CanonicalSerialize for SourceInstructionKind {
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
        impl CanonicalDeserialize for SourceInstructionKind {
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

        impl JoltInstructionKind {
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

            pub const fn tag(self) -> JoltInstructionTag {
                match self {
                    Self::NoOp => JoltInstructionTag(0x0000),
                    Self::Unimpl => JoltInstructionTag(0x0001),
                    $(
                        Self::$instr => JoltInstructionTag($tag),
                    )*
                    Self::Inline => JoltInstructionTag(0x0089),
                }
            }

            pub const fn from_tag(tag: JoltInstructionTag) -> Option<Self> {
                match tag.0 {
                    0x0000 => Some(Self::NoOp),
                    0x0001 => Some(Self::Unimpl),
                    $(
                        $tag => Some(Self::$instr),
                    )*
                    0x0089 => Some(Self::Inline),
                    _ => None,
                }
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

crate::for_each_instruction_kind!(define_instruction_kind);

#[cfg(test)]
mod tests {
    use super::{JoltInstructionKind, JoltInstructionTag, SourceInstructionKind};
    use std::collections::HashSet;

    #[test]
    fn tags_are_stable_for_representative_rows() {
        assert_eq!(JoltInstructionKind::NoOp.tag(), JoltInstructionTag(0x0000));
        assert_eq!(
            JoltInstructionKind::Unimpl.tag(),
            JoltInstructionTag(0x0001)
        );
        assert_eq!(JoltInstructionKind::ADD.tag(), JoltInstructionTag(0x0002));
        assert_eq!(
            JoltInstructionKind::VirtualHostIO.tag(),
            JoltInstructionTag(0x0068)
        );
        assert_eq!(
            JoltInstructionKind::VirtualXORROTW7.tag(),
            JoltInstructionTag(0x0088)
        );
        assert_eq!(
            JoltInstructionKind::Inline.tag(),
            JoltInstructionTag(0x0089)
        );
    }

    #[test]
    fn tags_round_trip_and_are_unique() {
        let mut seen = HashSet::new();
        for kind in JoltInstructionKind::ALL {
            let tag = kind.tag();
            assert!(seen.insert(tag), "duplicate tag {tag:?} for {kind:?}");
            assert_eq!(JoltInstructionKind::from_tag(tag), Some(*kind));
        }
    }

    #[test]
    fn canonical_names_are_stable_for_representative_rows() {
        assert_eq!(JoltInstructionKind::ADD.canonical_name(), "rv64.add");
        assert_eq!(
            JoltInstructionKind::VirtualHostIO.canonical_name(),
            "jolt.virtual.host_io"
        );
        assert_eq!(
            JoltInstructionKind::Inline.canonical_name(),
            "jolt.inline.dispatch"
        );
    }

    #[test]
    fn canonical_names_are_unique_and_non_empty() {
        let mut seen = HashSet::new();
        for kind in JoltInstructionKind::ALL {
            let name = kind.canonical_name();
            assert!(!name.is_empty(), "empty canonical name for {kind:?}");
            assert!(seen.insert(name), "duplicate canonical name {name:?}");
        }
    }

    #[test]
    fn source_tags_are_stable_for_representative_rows() {
        assert_eq!(
            SourceInstructionKind::NoOp.tag(),
            JoltInstructionTag(0x0000)
        );
        assert_eq!(
            SourceInstructionKind::Unimpl.tag(),
            JoltInstructionTag(0x0001)
        );
        assert_eq!(SourceInstructionKind::ADD.tag(), JoltInstructionTag(0x0002));
        assert_eq!(
            SourceInstructionKind::VirtualHostIO.tag(),
            JoltInstructionTag(0x0068)
        );
        assert_eq!(
            SourceInstructionKind::Inline.tag(),
            JoltInstructionTag(0x0089)
        );
    }

    #[test]
    fn source_tags_round_trip_and_are_unique() {
        let mut seen = HashSet::new();
        for kind in SourceInstructionKind::ALL {
            let tag = kind.tag();
            assert!(
                seen.insert(tag),
                "duplicate source tag {tag:?} for {kind:?}"
            );
            assert_eq!(SourceInstructionKind::from_tag(tag), Some(*kind));
        }
    }

    #[test]
    fn source_and_final_tags_currently_share_values_for_mapped_rows() {
        assert_eq!(
            SourceInstructionKind::ADD.tag(),
            JoltInstructionKind::ADD.tag()
        );
        assert_eq!(
            SourceInstructionKind::Inline.tag(),
            JoltInstructionKind::Inline.tag()
        );
    }
}
