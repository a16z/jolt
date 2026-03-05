//! Internal macros for concise instruction definitions.

/// Defines a unit-struct instruction with its `Instruction` trait implementation.
///
/// Usage:
/// ```ignore
/// define_instruction!(Add, opcodes::ADD, "ADD", |x, y| x.wrapping_add(y));
/// ```
macro_rules! define_instruction {
    ($(#[$meta:meta])* $name:ident, $opcode:expr, $label:expr, |$x:ident, $y:ident| $body:expr) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        #[derive(serde::Serialize, serde::Deserialize)]
        pub struct $name;

        impl $crate::Instruction for $name {
            #[inline]
            fn opcode(&self) -> u32 {
                $opcode
            }

            #[inline]
            fn name(&self) -> &'static str {
                $label
            }

            #[inline]
            fn execute(&self, $x: u64, $y: u64) -> u64 {
                $body
            }

            #[inline]
            fn lookups(&self, _x: u64, _y: u64) -> Vec<$crate::LookupQuery> {
                Vec::new()
            }
        }
    };
}

pub(crate) use define_instruction;
