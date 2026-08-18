//! Shared relation *shapes*: the register-Twist family's input/output algebra
//! as generic expression builders, instantiated by each protocol family with
//! its own ids.
//!
//! Ownership rule: this module owns the algebra only — it references NO
//! protocol id types and mints no ids. Each protocol module
//! (`protocols::jolt`, `protocols::field_inline`) owns its id families,
//! geometry, and claim structs, and instantiates a shape by implementing that
//! shape's id-supplier trait with its own constructors. Composition of the two
//! protocols happens only in `jolt-verifier`; the protocol modules never
//! import each other.
//!
//! Term-order stability is load-bearing: the BlindFold lowering turns these
//! expressions into R1CS rows term by term, so every builder reproduces the
//! exact term sequence of the pre-shape hand-written expressions (pinned by
//! this module's structural tests).

pub mod claim_reductions;
pub mod registers;

#[cfg(test)]
pub(crate) mod test_ids {
    //! Toy id families for the structural term-order pins. Framework-only:
    //! deliberately not any protocol's ids.

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Opening {
        In(usize),
        Out(usize),
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Derived {
        Eq(usize),
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Challenge {
        Gamma,
    }
}
