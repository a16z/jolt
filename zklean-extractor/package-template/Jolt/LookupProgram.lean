import Jolt.LookupGraphExpression

/-!
# Certified lookup programs

A lookup program contains the field expression used by the verifier and the
natural number expression used to materialize the table. A decidable check
establishes that the field expression is multi-affine and is the arithmetic
form of the materializer.
-/

namespace Jolt.LookupExpression

/-- A Boolean expression over the input bits of a lookup table. -/
inductive BoolExpr (n : Nat) where
  | input (index : Nat)
  | conj (left right : BoolExpr n)
  deriving DecidableEq

/-- Check that every Boolean materializer input is in range. -/
def BoolExpr.wellFormed {n : Nat} : BoolExpr n → Bool
  | .input index => index < n
  | .conj left right => left.wellFormed && right.wellFormed

/-- Evaluate a Boolean materializer expression. -/
def BoolExpr.eval {n : Nat} (expression : BoolExpr n) (point : Fin n → Bool) : Bool :=
  match expression with
  | .input index => if h : index < n then point ⟨index, h⟩ else false
  | .conj left right => left.eval point && right.eval point

/-- Convert a Boolean materializer expression to a field expression. -/
def BoolExpr.arithmetize {n : Nat} : BoolExpr n → Expr n
  | .input index => if h : index < n then .var ⟨index, h⟩ else .const 0
  | .conj left right => .mul left.arithmetize right.arithmetize

/-- Arithmetization preserves a Boolean expression on Boolean inputs. -/
theorem BoolExpr.boolCast_eval
    {n : Nat} {F : Type*} [CommRing F]
    (expression : BoolExpr n) (point : Fin n → Bool) :
    boolCast (F := F) (expression.eval point) =
      expression.arithmetize.eval (fun i => boolCast (point i)) := by
  induction expression with
  | input index =>
      simp only [BoolExpr.eval, BoolExpr.arithmetize]
      split <;> simp [boolCast, Expr.eval]
  | conj left right left_ih right_ih =>
      rw [BoolExpr.eval, boolCast_eq_natCast, natCast_and,
        ← boolCast_eq_natCast, ← boolCast_eq_natCast, left_ih, right_ih]
      rfl

/-- A natural number expression computed from lookup input bits. -/
inductive NatExpr (n : Nat) where
  | ofBitsBE (bits : List (BoolExpr n))
  deriving DecidableEq

/-- Check that every input used by a natural number materializer is in range. -/
def NatExpr.wellFormed {n : Nat} : NatExpr n → Bool
  | .ofBitsBE bits => bits.all BoolExpr.wellFormed

/-- A materializer is well formed when all of its input indices are valid. -/
def NatExpr.WellFormed {n : Nat} (expression : NatExpr n) : Prop :=
  expression.wellFormed = true

instance NatExpr.instDecidableWellFormed {n : Nat} (expression : NatExpr n) :
    Decidable expression.WellFormed :=
  inferInstanceAs (Decidable (expression.wellFormed = true))

/-- Read a most-significant-bit-first Boolean list as a natural number. -/
def BoolExpr.bitsToNat {n : Nat} : List (BoolExpr n) → (Fin n → Bool) → Nat
  | [], _ => 0
  | bit :: bits, point =>
      2 ^ bits.length * (bit.eval point).toNat + BoolExpr.bitsToNat bits point

/-- Convert most-significant-bit-first expressions to their field expression. -/
def BoolExpr.bitsArithmetize {n : Nat} : List (BoolExpr n) → Expr n
  | [] => .const 0
  | [bit] => .mul (.const 1) bit.arithmetize
  | bit :: next :: bits =>
      .add
        (.mul (.const (2 ^ (next :: bits).length)) bit.arithmetize)
        (BoolExpr.bitsArithmetize (next :: bits))

/-- Evaluate the natural number expression represented by a materializer. -/
def NatExpr.eval {n : Nat} (expression : NatExpr n) (point : Fin n → Bool) : Nat :=
  match expression with
  | .ofBitsBE bits => BoolExpr.bitsToNat bits point

/-- Convert a natural number materializer to a field expression. -/
def NatExpr.arithmetize {n : Nat} : NatExpr n → Expr n
  | .ofBitsBE bits => BoolExpr.bitsArithmetize bits

private theorem BoolExpr.cast_bitsToNat
    {n : Nat} {F : Type*} [CommRing F]
    (bits : List (BoolExpr n)) (point : Fin n → Bool) :
    (BoolExpr.bitsToNat bits point : F) =
      (BoolExpr.bitsArithmetize bits).eval (fun i => boolCast (point i)) := by
  induction bits with
  | nil => rfl
  | cons bit bits ih =>
      cases bits with
      | nil =>
          simpa only [bitsToNat, bitsArithmetize, Expr.eval, List.length_nil,
            pow_zero, Nat.cast_add, Nat.cast_mul, Nat.cast_one, Nat.cast_zero,
            add_zero, one_mul, boolCast_eq_natCast] using
            bit.boolCast_eval (F := F) point
      | cons next bits =>
          change
            ((2 ^ (next :: bits).length * (bit.eval point).toNat +
                BoolExpr.bitsToNat (next :: bits) point : Nat) : F) =
              Expr.eval
                (.add
                  (.mul (.const (2 ^ (next :: bits).length)) bit.arithmetize)
                  (BoolExpr.bitsArithmetize (next :: bits)))
                (fun i => boolCast (point i))
          rw [Nat.cast_add, Nat.cast_mul, Expr.eval, ← boolCast_eq_natCast,
            bit.boolCast_eval, ih]
          rfl

/-- Arithmetization preserves a natural number expression on Boolean inputs. -/
theorem NatExpr.cast_eval
    {n : Nat} {F : Type*} [CommRing F]
    (expression : NatExpr n) (point : Fin n → Bool) :
    (expression.eval point : F) =
      expression.arithmetize.eval (fun i => boolCast (point i)) := by
  cases expression with
  | ofBitsBE bits => exact BoolExpr.cast_bitsToNat (F := F) bits point

/-- Prove that a concrete extracted graph has the same polynomial semantics as a materializer. -/
macro "prove_lookup_program_correspondence" graph:ident materializer:ident : tactic =>
  `(tactic|
    (unfold $graph $materializer) <;>
      simp (config := { maxSteps := 1000000 }) [
        Jolt.LookupGraph.Graph.toExpr,
        Jolt.LookupGraph.Graph.expressionNodes,
        Jolt.LookupGraph.Node.toExpr,
        Jolt.LookupGraph.getExpression,
        NatExpr.arithmetize,
        BoolExpr.bitsArithmetize,
        BoolExpr.arithmetize,
        Expr.eval] <;>
      ring)

/-- The extracted field evaluator and materializer for one lookup table. -/
structure LookupProgram (n : Nat) where
  mle : Jolt.LookupGraph.Graph n
  mleWellFormed : mle.WellFormed
  materializer : NatExpr n
  materializerWellFormed : materializer.WellFormed

/-- Check that the materializer's arithmetic form is syntactically multilinear. -/
def LookupProgram.check {n : Nat} (program : LookupProgram n) : Bool :=
  program.materializer.arithmetize.isSyntacticallyMultilinear

/-- Evaluate a lookup program on a vector of field elements. -/
def LookupProgram.evalVector
    {n : Nat} {F : Type*} [CommRing F]
    (program : LookupProgram n) (point : Vector F n) : F :=
  program.mle.evalVector program.mleWellFormed point

@[simp]
theorem LookupProgram.evalVector_ofFn
    {n : Nat} {F : Type*} [CommRing F]
    (program : LookupProgram n) (point : Fin n → F) :
    program.evalVector (Vector.ofFn point) = program.mle.toExpr.eval point := by
  unfold LookupProgram.evalVector Jolt.LookupGraph.Graph.evalVector
  rw [program.mle.eval_eq_toExpr_eval]
  congr 1
  funext index
  exact Vector.getElem_ofFn index.isLt

/-- A checked lookup program is the multilinear extension of its materializer. -/
theorem LookupProgram.isLookupTableMLE
    {n : Nat} {F : Type*} [CommRing F]
    (program : LookupProgram n) (h : program.check = true)
    (mleCorrespondence : ∀ point : Fin n → F,
      program.mle.toExpr.eval point = program.materializer.arithmetize.eval point) :
    IsLookupTableMLE
      (fun point : Fin n → F => program.evalVector (Vector.ofFn point))
      program.materializer.eval := by
  constructor
  · rw [show (fun point : Fin n → F => program.evalVector (Vector.ofFn point)) =
        program.materializer.arithmetize.eval from by
          funext point
          rw [program.evalVector_ofFn]
          exact mleCorrespondence point]
    exact program.materializer.arithmetize.isMultiAffine (F := F) h
  · intro point
    change program.evalVector (Vector.ofFn (fun i => boolCast (point i))) = _
    rw [LookupProgram.evalVector_ofFn, mleCorrespondence]
    exact (program.materializer.cast_eval (F := F) point).symm

end Jolt.LookupExpression
