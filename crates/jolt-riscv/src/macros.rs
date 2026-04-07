//! Internal macros for concise instruction definitions.

/// Defines a unit-struct instruction with `Instruction` and `Flags` trait implementations.
///
/// # Syntax
///
/// ```ignore
/// define_instruction!(
///     /// Doc comment.
///     Add, "ADD",
///     |x, y| x.wrapping_add(y),
///     circuit: [AddOperands, WriteLookupOutputToRD],
///     instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
///     table: RangeCheck,
/// );
/// ```
///
/// All three trailing sections (`circuit`, `instruction`, `table`) are optional.
/// Omitting `circuit` or `instruction` produces an all-false flag array.
/// Omitting `table` makes `lookup_table()` return `None`.
macro_rules! define_instruction {
    (
        $(#[$meta:meta])*
        $name:ident, $label:expr,
        |$x:ident, $y:ident| $body:expr
        $(, circuit: [$($cflag:ident),* $(,)?])?
        $(, instruction: [$($iflag:ident),* $(,)?])?
        $(, table: $table:ident)?
        $(,)?
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        #[derive(serde::Serialize, serde::Deserialize)]
        pub struct $name;

        impl $crate::Instruction for $name {
            #[inline]
            fn name(&self) -> &'static str {
                $label
            }

            #[inline]
            fn execute(&self, $x: u64, $y: u64) -> u64 {
                $body
            }

            #[inline]
            fn lookup_table(&self) -> Option<$crate::LookupTableKind> {
                define_instruction!(@table $($table)?)
            }
        }

        impl $crate::Flags for $name {
            #[inline]
            fn circuit_flags(&self) -> $crate::CircuitFlagSet {
                define_instruction!(@cflags $(, $($cflag),*)?)
            }

            #[inline]
            fn instruction_flags(&self) -> $crate::InstructionFlagSet {
                define_instruction!(@iflags $(, $($iflag),*)?)
            }
        }
    };

    (@cflags) => { $crate::CircuitFlagSet::default() };
    (@cflags, $($flag:ident),+) => {
        $crate::CircuitFlagSet::default()$(.set($crate::CircuitFlags::$flag))+
    };

    (@iflags) => { $crate::InstructionFlagSet::default() };
    (@iflags, $($flag:ident),+) => {
        $crate::InstructionFlagSet::default()$(.set($crate::InstructionFlags::$flag))+
    };

    // Internal: resolve optional table to Some(Kind) or None.
    (@table $table:ident) => { Some($crate::LookupTableKind::$table) };
    (@table) => { None };
}
