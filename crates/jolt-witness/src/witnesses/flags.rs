use jolt_field::Field;

/// Whether the successor row is a no-op. The last cycle's missing successor
/// counts as a no-op: the product/shift family requires `NextIsNoop = 1` at
/// `T - 1` (legacy forces `not_next_noop = false` there — "EqPlusOne does not
/// do overflow").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextIsNoop(pub bool);

/// Whether the successor row is a virtual instruction; false at the last
/// cycle and for undecodable successors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextIsVirtual(pub bool);

/// Whether the successor row starts a virtual sequence; false at the last
/// cycle and for undecodable successors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextIsFirstInSequence(pub bool);

/// Jump instruction actually taken (jump flag set and the successor is a
/// real instruction; unlike [`NextIsNoop`], a missing successor does NOT
/// count as a no-op here). At `T - 1` constraint 21
/// (`ShouldJump = Jump · (1 − NextIsNoop)`) forces `ShouldJump = 0`, which
/// holds under either convention only because the padded trace ends in a
/// NoOp row (`Jump = 0` there) — the padding every prover config guarantees.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShouldJump(pub bool);

/// Branch instruction whose comparison output is 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShouldBranch(pub bool);

/// Set when the instruction's lookup operands are NOT interleaved (the RAF
/// address decomposition applies).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InstructionRafFlag(pub bool);

/// One circuit flag of the instruction; which flag is bound at the use site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpFlag(pub bool);

/// One instruction flag of the instruction; which flag is bound at the use
/// site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InstructionFlag(pub bool);

/// Whether the instruction's lookup targets the table bound at the use site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookupTableFlag(pub bool);

macro_rules! bool_to_field {
    ($($name:ident),* $(,)?) => {
        $(impl $name {
            pub fn to_field<F: Field>(self) -> F {
                F::from_bool(self.0)
            }
        })*
    };
}

bool_to_field!(
    NextIsNoop,
    NextIsVirtual,
    NextIsFirstInSequence,
    ShouldJump,
    ShouldBranch,
    InstructionRafFlag,
    OpFlag,
    InstructionFlag,
    LookupTableFlag,
);
