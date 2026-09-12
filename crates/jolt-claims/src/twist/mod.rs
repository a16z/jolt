//! The Twist memory-checking relations, generic over the checked memory.
//!
//! Twist proves a register file or RAM consistent over time with a fixed
//! pipeline of sumcheck relations: read/write checking, value evaluation, and
//! the value / increment claim reductions that route its openings. Every
//! memory the protocol checks — the ordinary x-register file, the field-inline
//! FR register file — proves the *same identities* over its own polynomial
//! family, so this module states each identity once as an expression builder
//! generic over an id-binding trait, and each protocol module binds it to a
//! concrete memory instance.
//!
//! Ownership rule: this module owns the sumcheck identities only — it
//! references NO protocol id types and mints no ids. Each protocol module
//! (`protocols::jolt`, `protocols::field_inline`) owns its id families,
//! geometry, and claim structs, and binds an identity by implementing its
//! id-binding trait. Composition of the two protocols happens only in
//! `jolt-verifier`; the protocol modules never import each other.
//!
//! Term-order stability is load-bearing: the BlindFold lowering turns these
//! expressions into R1CS rows term by term, so every builder reproduces the
//! exact term sequence of the previously hand-written per-protocol
//! expressions (pinned by this module's structural tests).

pub mod claim_reductions;
pub mod memory_checking;

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
