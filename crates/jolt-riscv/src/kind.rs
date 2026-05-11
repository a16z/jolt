#[cfg(feature = "serialization")]
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

macro_rules! define_instruction_kind {
    (
        instructions: [$($instr:ident),* $(,)?]
    ) => {
        /// Decoded guest-program instruction kind before bytecode expansion.
        ///
        /// This is the source side of the program pipeline. It includes RV64
        /// ISA opcodes decoded from ELF text and Jolt custom source opcodes
        /// such as registered inlines, advice loads, and host I/O markers.
        /// Expansion maps these source kinds into final [`JoltInstructionKind`]
        /// rows consumed by bytecode preprocessing, tracing, and proof code.
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

            pub const fn is_source_only(self) -> bool {
                matches!(
                    self,
                    Self::Inline
                        | Self::ADDIW
                        | Self::ADDW
                        | Self::SUBW
                        | Self::MULH
                        | Self::MULHSU
                        | Self::MULW
                        | Self::LB
                        | Self::LBU
                        | Self::LH
                        | Self::LHU
                        | Self::LW
                        | Self::LWU
                        | Self::AdviceLB
                        | Self::AdviceLH
                        | Self::AdviceLW
                        | Self::AdviceLD
                        | Self::AMOADDD
                        | Self::AMOANDD
                        | Self::AMOORD
                        | Self::AMOXORD
                        | Self::AMOSWAPD
                        | Self::AMOMAXD
                        | Self::AMOMAXUD
                        | Self::AMOMIND
                        | Self::AMOMINUD
                        | Self::AMOADDW
                        | Self::AMOANDW
                        | Self::AMOORW
                        | Self::AMOXORW
                        | Self::AMOSWAPW
                        | Self::AMOMAXW
                        | Self::AMOMAXUW
                        | Self::AMOMINW
                        | Self::AMOMINUW
                        | Self::LRD
                        | Self::LRW
                        | Self::DIV
                        | Self::DIVU
                        | Self::DIVW
                        | Self::DIVUW
                        | Self::REM
                        | Self::REMU
                        | Self::REMW
                        | Self::REMUW
                        | Self::SB
                        | Self::SCD
                        | Self::SCW
                        | Self::SH
                        | Self::SW
                        | Self::CSRRW
                        | Self::CSRRS
                        | Self::EBREAK
                        | Self::ECALL
                        | Self::MRET
                        | Self::SLL
                        | Self::SLLI
                        | Self::SLLW
                        | Self::SLLIW
                        | Self::SRL
                        | Self::SRLI
                        | Self::SRA
                        | Self::SRAI
                        | Self::SRLIW
                        | Self::SRAIW
                        | Self::SRLW
                        | Self::SRAW
                )
            }

            pub const fn final_kind(self) -> Option<JoltInstructionKind> {
                if self.is_source_only() {
                    None
                } else {
                    Some(self.jolt_kind())
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

            pub const fn handles_rd_zero_internally(self) -> bool {
                matches!(
                    self,
                    Self::ECALL | Self::MRET | Self::EBREAK | Self::CSRRW | Self::CSRRS
                )
            }
        }

        impl From<JoltInstructionKind> for SourceInstructionKind {
            fn from(value: JoltInstructionKind) -> Self {
                match value {
                    JoltInstructionKind::NoOp => Self::NoOp,
                    JoltInstructionKind::Unimpl => Self::Unimpl,
                    $(
                        JoltInstructionKind::$instr => Self::$instr,
                    )*
                    JoltInstructionKind::Inline => Self::Inline,
                }
            }
        }

        #[cfg(feature = "serialization")]
        impl CanonicalSerialize for JoltInstructionKind {
            fn serialize_with_mode<W: Write>(
                &self,
                writer: W,
                compress: Compress,
            ) -> Result<(), SerializationError> {
                (*self as u16).serialize_with_mode(writer, compress)
            }

            fn serialized_size(&self, compress: Compress) -> usize {
                (*self as u16).serialized_size(compress)
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
                match value {
                    x if x == Self::NoOp as u16 => Ok(Self::NoOp),
                    x if x == Self::Unimpl as u16 => Ok(Self::Unimpl),
                    $(
                        x if x == Self::$instr as u16 => Ok(Self::$instr),
                    )*
                    x if x == Self::Inline as u16 => Ok(Self::Inline),
                    _ => Err(SerializationError::InvalidData),
                }
            }
        }

        #[cfg(feature = "serialization")]
        impl Valid for JoltInstructionKind {
            fn check(&self) -> Result<(), SerializationError> {
                Ok(())
            }
        }

        impl JoltInstructionKind {
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
