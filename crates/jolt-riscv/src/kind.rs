use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
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
            Serialize,
            Deserialize,
        )]
        #[repr(u16)]
        pub enum InstructionKind {
            #[default]
            NoOp,
            Unimpl,
            $(
                $instr,
            )*
            Inline,
        }

        impl CanonicalSerialize for InstructionKind {
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

        impl CanonicalDeserialize for InstructionKind {
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

        impl Valid for InstructionKind {
            fn check(&self) -> Result<(), SerializationError> {
                Ok(())
            }
        }
    };
}

crate::for_each_instruction_kind!(define_instruction_kind);
