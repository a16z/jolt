#[macro_export]
macro_rules! declare_riscv_instr {
    (
      name    = $name:ident,
      mask    = $mask:expr,
      match   = $match_:expr,
      format  = $format:ty,
      ram     = $ram:ty $(, { $($extra:item)* })? $(,)?
  ) => {
        $crate::declare_riscv_instr!(@inner $name, $mask, $match_, $format, $ram $(, { $($extra)* })?);
    };
    (@source_kind $name:ident) => {
        match ::jolt_riscv::SourceInstructionKind::from_name(stringify!($name)) {
            Some(kind) => kind,
            None => unreachable!("unknown tracer instruction source kind"),
        }
    };
    (@inner $name:ident, $mask:expr, $match_:expr, $format:ty, $ram:ty $(, { $($extra:item)* })?) => {
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

            fn source_kind(&self) -> ::jolt_riscv::SourceInstructionKind {
                $crate::declare_riscv_instr!(@source_kind $name)
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

            $($($extra)*)?
        }

        impl From<$crate::instruction::SourceInstructionRow> for $name {
            fn from(row: $crate::instruction::SourceInstructionRow) -> Self {
                Self {
                    address: row.address as u64,
                    operands: <$format as $crate::instruction::format::InstructionFormat>::from_source(
                        row.operands,
                        $crate::declare_riscv_instr!(@source_kind $name),
                    ),
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed: row.is_compressed,
                }
            }
        }

    };
}
