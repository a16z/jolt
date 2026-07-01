use jolt_field::RingCore;

use crate::{Expr, SumcheckDomain};

/// Pure symbolic description of one sumcheck relation: its id, sumcheck spec, and
/// input/output algebra over the relation's id types. The expression methods are
/// generic over the field `F`, so the implementing object is field-independent (it
/// holds only its [`Shape`](Self::Shape)) and emits expressions for any field on
/// demand. See `specs/symbolic-sumcheck.md`.
pub trait SymbolicSumcheck {
    type RelationId;
    type OpeningId;
    type DerivedId;
    type ChallengeId;

    /// The construction input that fully determines this relation's structure (its
    /// expressions and sumcheck spec). Field-independent for every relation.
    type Shape;

    /// The relation's drawn Fiat-Shamir challenges, parameterized by the field
    /// (challenges carry no opening point). A relation with no challenges uses
    /// [`NoChallenges`](crate::NoChallenges).
    type Challenges<F>;

    /// The relation's consumed-claim struct, generic over the opening *cell*,
    /// instantiated at `F` (the serialized wire value) or `Vec<F>` (the derived
    /// opening point). A symbolic-only relation uses [`NoInputs`](crate::NoInputs).
    type Inputs<C>;
    /// The relation's produced-claim struct, generic over the opening *cell*
    /// (`F` value | `Vec<F>` point). A symbolic-only relation uses
    /// [`NoOutputs`](crate::NoOutputs).
    type Outputs<C>;

    fn new(shape: Self::Shape) -> Self;

    /// The relation this sumcheck belongs to. A type-level constant; NOT a unique
    /// key — several sumchecks can share one relation id (address/cycle-phase
    /// splits, the full/committed bytecode modes, the Spartan uni-skip/remainder
    /// pairs).
    fn id() -> Self::RelationId;

    /// The domain this sumcheck runs over. Defaults to the Boolean hypercube;
    /// only the univariate-skip relations override it (centered-integer domain).
    /// A fixed constant per relation, independent of the [`Shape`](Self::Shape);
    /// it takes `&self` only so it reads like its `rounds`/`degree` siblings at
    /// the (instance) call sites.
    fn domain(&self) -> SumcheckDomain {
        SumcheckDomain::BooleanHypercube
    }

    /// The sumcheck round count, derived from [`Shape`](Self::Shape).
    fn rounds(&self) -> usize;

    /// The per-round degree bound, derived from [`Shape`](Self::Shape).
    fn degree(&self) -> usize;

    fn input_expression<F: RingCore>(
        &self,
    ) -> Expr<F, Self::OpeningId, Self::DerivedId, Self::ChallengeId>;
    fn output_expression<F: RingCore>(
        &self,
    ) -> Expr<F, Self::OpeningId, Self::DerivedId, Self::ChallengeId>;
}
