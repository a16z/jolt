#[macro_export]
macro_rules! declare_riscv_instr {
    (
      name    = $name:ident,
      mask    = $mask:expr,
      match   = $match_:expr,
      format  = $format:ty,
      ram     = $ram:ty
  ) => {
        #[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
        pub struct $name {
            pub address: u64,
            pub operands: $format,
            pub virtual_sequence_remaining: Option<u16>,
            pub is_first_in_sequence: bool,
            /// Set if instruction is C-Type
            pub is_compressed: bool,
        }

        impl $crate::instruction::RISCVInstruction for $name {
            const MASK: u32 = $mask;
            const MATCH: u32 = $match_;

            type Format = $format;
            type RAMAccess = $ram;

            fn operands(&self) -> &Self::Format {
                &self.operands
            }

            fn new(word: u32, address: u64, validate: bool, compressed: bool) -> Self {
                if validate {
                    debug_assert_eq!(
                        word & Self::MASK,
                        Self::MATCH,
                        "word: {:x}, mask: {:x}, word & mask: {:x}, match: {:x}",
                        word,
                        Self::MASK,
                        word & Self::MASK,
                        Self::MATCH
                    );
                }
                Self {
                    address,
                    operands: <$format as $crate::instruction::format::InstructionFormat>::parse(
                        word,
                    ),
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed: compressed,
                }
            }

            #[cfg(any(feature = "test-utils", test))]
            fn random(rng: &mut rand::rngs::StdRng) -> Self {
                Self {
                    address: rand::RngCore::next_u64(rng),
                    operands: <$format as $crate::instruction::format::InstructionFormat>::random(
                        rng,
                    ),
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed: false,
                }
            }

            fn execute(&self, cpu: &mut $crate::emulator::cpu::Cpu, ram: &mut Self::RAMAccess) {
                self.exec(cpu, ram)
            }
        }

        impl From<$crate::instruction::NormalizedInstruction> for $name {
            fn from(ni: $crate::instruction::NormalizedInstruction) -> Self {
                Self {
                    address: ni.address as u64,
                    operands: ni.operands.into(),
                    virtual_sequence_remaining: ni.virtual_sequence_remaining,
                    is_first_in_sequence: ni.is_first_in_sequence,
                    is_compressed: ni.is_compressed,
                }
            }
        }

        impl From<$name> for $crate::instruction::NormalizedInstruction {
            fn from(instr: $name) -> $crate::instruction::NormalizedInstruction {
                $crate::instruction::NormalizedInstruction {
                    address: instr.address as usize,
                    operands: instr.operands.into(),
                    is_compressed: instr.is_compressed,
                    virtual_sequence_remaining: instr.virtual_sequence_remaining,
                    is_first_in_sequence: instr.is_first_in_sequence,
                }
            }
        }
    };
}
