#[macro_export]
macro_rules! declare_riscv_instr {
    (
      name    = $name:ident,
      mask    = $mask:expr,
      match   = $match_:expr,
      format  = $format:ty,
      ram     = $ram:ty
      $(, is_virtual = $virt:tt)?
        $(,)?
  ) => {
        #[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
        pub struct $name {
            pub address: u64,
            pub operands: $format,
            pub inline_sequence_remaining: Option<u16>,
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
                debug_assert_eq!(word & Self::MASK, Self::MATCH, "word: {:x}, mask: {:x}, word & mask: {:x}, match: {:x}", word, Self::MASK, word & Self::MASK, Self::MATCH);
              }
              if declare_riscv_instr!(@is_virtual $( $virt )?) {
                    panic!(
                        "virtual instruction `{}` cannot be built from a machine word",
                        stringify!($name)
                    );
                }
                Self {
                    address,
                    operands: <$format as $crate::instruction::format::InstructionFormat>::parse(word),
                    inline_sequence_remaining: None,
                    is_compressed: compressed,
                }
            }

            #[cfg(any(feature = "test-utils", test))]
            fn random(rng: &mut rand::rngs::StdRng) -> Self {
                Self {
                    address: rand::RngCore::next_u64(rng),
                    operands: <$format as $crate::instruction::format::InstructionFormat>::random(rng),
                    inline_sequence_remaining: None,
                    is_compressed: false,
                }
            }

            fn execute(&self, cpu: &mut Cpu, ram: &mut Self::RAMAccess) {
                self.exec(cpu, ram)
            }
        }

        impl From<$crate::instruction::NormalizedInstruction> for $name {
            fn from(ni: $crate::instruction::NormalizedInstruction) -> Self {
                Self {
                    address: ni.address as u64,
                    operands: ni.operands.into(),
                    inline_sequence_remaining: None,
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
                    inline_sequence_remaining: instr.inline_sequence_remaining,
                }
            }
        }
    };

    (@is_virtual true) => {
        true
    };
    (@is_virtual) => {
        false
    };
}
