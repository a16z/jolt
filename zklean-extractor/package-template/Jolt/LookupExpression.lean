import Mathlib.Algebra.Ring.Defs
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic.Ring

/-!
# Checked lookup expressions

The extractor represents a lookup evaluator with this small expression
language before printing it as Lean code. The central theorem proves that a
simple structural check is enough to establish that the interpreted expression
is affine in every variable.
-/

namespace Jolt.LookupExpression

/-- An algebraic lookup expression with `n` input variables. -/
inductive Expr (n : Nat) where
  | const (value : Nat)
  | var (index : Fin n)
  | add (left right : Expr n)
  | sub (left right : Expr n)
  | mul (left right : Expr n)
  deriving DecidableEq

/-- Evaluate an algebraic lookup expression over a commutative ring. -/
def Expr.eval
    {n : Nat} {F : Type*} [CommRing F]
    (expression : Expr n) (point : Fin n → F) : F :=
  match expression with
  | .const value => value
  | .var index => point index
  | .add left right => left.eval point + right.eval point
  | .sub left right => left.eval point - right.eval point
  | .mul left right => left.eval point * right.eval point

/-- The variables that occur in an expression. -/
def Expr.variables {n : Nat} : Expr n → Finset (Fin n)
  | .const _ => ∅
  | .var index => {index}
  | .add left right => left.variables ∪ right.variables
  | .sub left right => left.variables ∪ right.variables
  | .mul left right => left.variables ∪ right.variables

/-- Check that every multiplication uses disjoint sets of variables. -/
def Expr.isSyntacticallyMultilinear {n : Nat} : Expr n → Bool
  | .const _ => true
  | .var _ => true
  | .add left right =>
      left.isSyntacticallyMultilinear && right.isSyntacticallyMultilinear
  | .sub left right =>
      left.isSyntacticallyMultilinear && right.isSyntacticallyMultilinear
  | .mul left right =>
      left.isSyntacticallyMultilinear &&
        right.isSyntacticallyMultilinear &&
          decide (Disjoint left.variables right.variables)

/--
An expression is syntactically multilinear when the structural check returns
`true`.
-/
def Expr.IsSyntacticallyMultilinear {n : Nat} (expression : Expr n) : Prop :=
  expression.isSyntacticallyMultilinear = true

instance Expr.instDecidableIsSyntacticallyMultilinear
    {n : Nat} (expression : Expr n) :
    Decidable expression.IsSyntacticallyMultilinear :=
  inferInstanceAs (Decidable (expression.isSyntacticallyMultilinear = true))

/-- A function is affine in each input coordinate separately. -/
def IsMultiAffine
    {n : Nat} {F : Type*} [CommRing F]
    (polynomial : (Fin n → F) → F) : Prop :=
  ∀ (index : Fin n) (point : Fin n → F) (value : F),
    polynomial (Function.update point index value) =
      (1 - value) * polynomial (Function.update point index 0) +
        value * polynomial (Function.update point index 1)

/-- Embed a Boolean bit as zero or one in a type with those constants. -/
def boolCast {F : Type*} [Zero F] [One F] (bit : Bool) : F :=
  if bit then 1 else 0

/-- Embedding a Boolean bit into a semiring agrees with its natural number
value. -/
theorem boolCast_eq_natCast
    {F : Type*} [Semiring F] (bit : Bool) :
    boolCast (F := F) bit = (bit.toNat : F) := by
  cases bit <;> simp [boolCast]

/-- Boolean conjunction becomes multiplication after embedding into a
semiring. -/
theorem natCast_and
    {F : Type*} [Semiring F] (left right : Bool) :
    ((left && right).toNat : F) = left.toNat * right.toNat := by
  cases left <;> cases right <;> simp

/-- A polynomial agrees with a natural-number lookup table at every Boolean
input. -/
def IsBooleanExtension
    {n : Nat} {F : Type*} [CommRing F]
    (polynomial : (Fin n → F) → F)
    (table : (Fin n → Bool) → Nat) : Prop :=
  ∀ point, polynomial (fun i => boolCast (point i)) = table point

/-- A polynomial is the multilinear extension of a lookup table when it is
affine in every coordinate and agrees with the table on every Boolean input. -/
def IsLookupTableMLE
    {n : Nat} {F : Type*} [CommRing F]
    (polynomial : (Fin n → F) → F)
    (table : (Fin n → Bool) → Nat) : Prop :=
  IsMultiAffine polynomial ∧ IsBooleanExtension polynomial table

private theorem Expr.eval_update_of_not_mem
    {n : Nat} {F : Type*} [CommRing F]
    (expression : Expr n) (index : Fin n) (point : Fin n → F) (value : F)
    (hindex : index ∉ expression.variables) :
    expression.eval (Function.update point index value) =
      expression.eval point := by
  induction expression with
  | const constant => simp [Expr.eval]
  | var j =>
      simp only [Expr.variables, Finset.mem_singleton] at hindex
      have hji : j ≠ index := Ne.symm hindex
      simp [Expr.eval, Function.update, hji]
  | add left right left_ih right_ih =>
      simp only [Expr.variables, Finset.mem_union, not_or] at hindex
      simp [Expr.eval, left_ih hindex.1, right_ih hindex.2]
  | sub left right left_ih right_ih =>
      simp only [Expr.variables, Finset.mem_union, not_or] at hindex
      simp [Expr.eval, left_ih hindex.1, right_ih hindex.2]
  | mul left right left_ih right_ih =>
      simp only [Expr.variables, Finset.mem_union, not_or] at hindex
      simp [Expr.eval, left_ih hindex.1, right_ih hindex.2]

/-- The structural multilinearity check implies the mathematical affine
property for the interpreted expression over every commutative ring. -/
theorem Expr.isMultiAffine
    {n : Nat} {F : Type*} [CommRing F]
    (expression : Expr n)
    (h : expression.IsSyntacticallyMultilinear) :
    IsMultiAffine (expression.eval (F := F)) := by
  induction expression with
  | const constant =>
      intro index point value
      simp [Expr.eval]
      ring
  | var j =>
      intro index point value
      by_cases hindex : index = j
      · subst j
        simp [Expr.eval]
      · have hji : j ≠ index := Ne.symm hindex
        simp [Expr.eval, Function.update, hji]
        ring
  | add left right left_ih right_ih =>
      simp only [Expr.IsSyntacticallyMultilinear,
        Expr.isSyntacticallyMultilinear, Bool.and_eq_true] at h
      rcases h with ⟨hleft, hright⟩
      intro index point value
      change left.eval (Function.update point index value) +
          right.eval (Function.update point index value) =
        (1 - value) *
            (left.eval (Function.update point index 0) +
              right.eval (Function.update point index 0)) +
          value *
            (left.eval (Function.update point index 1) +
              right.eval (Function.update point index 1))
      rw [left_ih hleft index point value, right_ih hright index point value]
      ring
  | sub left right left_ih right_ih =>
      simp only [Expr.IsSyntacticallyMultilinear,
        Expr.isSyntacticallyMultilinear, Bool.and_eq_true] at h
      rcases h with ⟨hleft, hright⟩
      intro index point value
      change left.eval (Function.update point index value) -
          right.eval (Function.update point index value) =
        (1 - value) *
            (left.eval (Function.update point index 0) -
              right.eval (Function.update point index 0)) +
          value *
            (left.eval (Function.update point index 1) -
              right.eval (Function.update point index 1))
      rw [left_ih hleft index point value, right_ih hright index point value]
      ring
  | mul left right left_ih right_ih =>
      simp only [Expr.IsSyntacticallyMultilinear,
        Expr.isSyntacticallyMultilinear, Bool.and_eq_true,
        decide_eq_true_eq] at h
      rcases h with ⟨⟨hleft, hright⟩, hdisjoint⟩
      intro index point value
      change left.eval (Function.update point index value) *
          right.eval (Function.update point index value) =
        (1 - value) *
            (left.eval (Function.update point index 0) *
              right.eval (Function.update point index 0)) +
          value *
            (left.eval (Function.update point index 1) *
              right.eval (Function.update point index 1))
      by_cases hindex : index ∈ left.variables
      · have hnot : index ∉ right.variables :=
          Finset.disjoint_left.mp hdisjoint hindex
        rw [right.eval_update_of_not_mem index point value hnot,
          right.eval_update_of_not_mem index point 0 hnot,
          right.eval_update_of_not_mem index point 1 hnot,
          left_ih hleft index point value]
        ring
      · rw [left.eval_update_of_not_mem index point value hindex,
          left.eval_update_of_not_mem index point 0 hindex,
          left.eval_update_of_not_mem index point 1 hindex,
          right_ih hright index point value]
        ring

end Jolt.LookupExpression
