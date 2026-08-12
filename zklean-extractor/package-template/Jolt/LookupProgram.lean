import Jolt.LookupCorrespondenceCertificate

/-!
# Certified lookup programs

A lookup program contains the field graph used by the verifier and the shared
materializer graph used during preprocessing. Lean checks both graphs and
establishes their correspondence over every commutative ring.
-/

namespace Jolt.LookupExpression

/-- The extracted field evaluator and materializer for one lookup table. -/
structure LookupProgram (n : Nat) where
  mle : Jolt.LookupGraph.Graph n
  mleWellFormed : mle.WellFormed
  materializer : MaterializerGraph n
  materializerWellFormed : materializer.WellFormed

/-- Check that the verifier's extracted field expression is syntactically multilinear. -/
def LookupProgram.check {n : Nat} (program : LookupProgram n) : Bool :=
  program.mle.checkMultilinear

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

@[simp]
theorem LookupProgram.evalVector_ofFn_graph
    {n : Nat} {F : Type*} [CommRing F]
    (program : LookupProgram n) (point : Fin n → F) :
    program.evalVector (Vector.ofFn point) =
      program.mle.eval program.mleWellFormed point := by
  unfold LookupProgram.evalVector Jolt.LookupGraph.Graph.evalVector
  congr 1
  funext index
  exact Vector.getElem_ofFn index.isLt

/-- A checked lookup program is the multilinear extension of its materializer. -/
theorem LookupProgram.isLookupTableMLE
    {n : Nat} {F : Type*} [CommRing F]
    (program : LookupProgram n) (h : program.check = true)
    (mleCorrespondence : ∀ point : Fin n → F,
      program.mle.eval program.mleWellFormed point = program.materializer.arithEval point) :
    IsLookupTableMLE
      (fun point : Fin n → F => program.evalVector (Vector.ofFn point))
      program.materializer.eval := by
  constructor
  · rw [show (fun point : Fin n → F => program.evalVector (Vector.ofFn point)) =
        program.mle.toExpr.eval from by
          funext point
          exact program.evalVector_ofFn point]
    change program.mle.checkMultilinear = true at h
    rw [program.mle.checkMultilinear_eq_toExpr] at h
    exact program.mle.toExpr.isMultiAffine (F := F) h
  · intro point
    change program.evalVector (Vector.ofFn (fun i => boolCast (point i))) = _
    rw [LookupProgram.evalVector_ofFn_graph, mleCorrespondence]
    exact (program.materializer.cast_eval (F := F) point).symm

end Jolt.LookupExpression
