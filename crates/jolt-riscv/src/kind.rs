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
                        | Self::FieldMul
                        | Self::FieldAdd
                        | Self::FieldSub
                        | Self::FieldInv
                        | Self::FieldAssertEq
                        | Self::FieldMov
                        | Self::FieldSLL64
                        | Self::FieldSLL128
                        | Self::FieldSLL192
                )
            }
        }
    };
}

crate::for_each_instruction_kind!(define_instruction_kind);

impl JoltInstructionKind {
    /// Returns `(reads_frs1, reads_frs2, writes_frd)` for this instruction
    /// kind. Single source of truth for FR-coprocessor access classification;
    /// shared between the host-side `fr_bytecode_from_trace` and the
    /// kernel-side `stage6_bytecode_entries` so the two cannot drift.
    ///
    /// Soundness: Stage 4's FR Twist and Stage 6's bytecode-RAF anchor both
    /// depend on this classification. Divergence between the two consumers
    /// would break the cycle-vs-bytecode binding that closes the C-A4
    /// drop-event gap.
    #[inline]
    pub const fn fr_access_flags(self) -> (bool, bool, bool) {
        match self {
            Self::FieldMul | Self::FieldAdd | Self::FieldSub => (true, true, true),
            Self::FieldInv => (true, false, true),
            Self::FieldAssertEq => (true, true, false),
            Self::FieldMov | Self::FieldSLL64 | Self::FieldSLL128 | Self::FieldSLL192 => {
                (false, false, true)
            }
            _ => (false, false, false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fr_access_flags_match_per_kind_contract() {
        // Locks the FR access classification used by Stage 4 RW + Stage 6
        // bytecode-RAF anchor. Any future change to FR instruction semantics
        // requires updating this table — which prevents silent classifier
        // drift between the host and kernel call sites.
        type Kind = JoltInstructionKind;
        let cases: &[(Kind, (bool, bool, bool))] = &[
            (Kind::FieldMul, (true, true, true)),
            (Kind::FieldAdd, (true, true, true)),
            (Kind::FieldSub, (true, true, true)),
            (Kind::FieldInv, (true, false, true)),
            (Kind::FieldAssertEq, (true, true, false)),
            (Kind::FieldMov, (false, false, true)),
            (Kind::FieldSLL64, (false, false, true)),
            (Kind::FieldSLL128, (false, false, true)),
            (Kind::FieldSLL192, (false, false, true)),
            (Kind::ADD, (false, false, false)),
            (Kind::NoOp, (false, false, false)),
        ];
        for (kind, expected) in cases {
            assert_eq!(
                kind.fr_access_flags(),
                *expected,
                "fr_access_flags drift on {kind:?}"
            );
        }
    }
}
