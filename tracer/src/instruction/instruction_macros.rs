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
            pub virtual_sequence_remaining: Option<u16>,
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
              if declare_riscv_instr!(@is_virtual $( $virt )?) {
                    panic!(
                        "virtual instruction `{}` cannot be built from a machine word",
                        stringify!($name)
                    );
                }
                if validate {
                    debug_assert_eq!(word & Self::MASK, Self::MATCH);
                }
                Self {
                    address,
                    operands: <$format>::parse(word),
                    virtual_sequence_remaining: None,
                    is_compressed: compressed,
                }
            }

            fn random(rng: &mut rand::rngs::StdRng) -> Self {
                Self {
                    address: rand::RngCore::next_u64(rng),
                    operands: <$format>::random(rng),
                    virtual_sequence_remaining: None,
                    is_compressed: false,
                }
            }

            fn execute(&self, cpu: &mut Cpu, ram: &mut Self::RAMAccess) {
                self.exec(cpu, ram)
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
