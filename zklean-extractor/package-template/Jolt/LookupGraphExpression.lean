import Jolt.LookupExpression
import Jolt.LookupGraph
import Mathlib.Data.Fintype.Basic

/-!
# Algebraic meaning of lookup graphs

This file connects the compact graph evaluator to the algebraic expression
language used by lookup certificates. The main theorem states that direct
graph evaluation and evaluation of the reconstructed expression agree.
-/

namespace Jolt.LookupGraph

open Jolt.LookupExpression

/-- Read a reconstructed expression, returning zero when the index is out of bounds. -/
def getExpression {arity : Nat} (expressions : Array (Expr arity))
    (index : Nat) : Expr arity :=
  expressions[index]?.getD (.const 0)

/-- Reconstruct one algebraic expression from the expressions at earlier graph nodes. -/
def Node.toExpr {arity : Nat} (node : Node) (expressions : Array (Expr arity)) :
    Expr arity :=
  match node with
  | .constant value => .const value
  | .input index => if h : index < arity then .var ⟨index, h⟩ else .const 0
  | .add left right => .add (getExpression expressions left) (getExpression expressions right)
  | .sub left right => .sub (getExpression expressions left) (getExpression expressions right)
  | .mul left right => .mul (getExpression expressions left) (getExpression expressions right)

/-- Reconstruct the expression represented at every graph node. -/
def Graph.expressionNodes {arity : Nat} (graph : Graph arity) : Array (Expr arity) :=
  graph.nodeChunks.foldl
    (fun expressions nodes =>
      nodes.foldl (fun expressions node => expressions.push (node.toExpr expressions)) expressions)
    #[]

/-- Reconstruct the algebraic expression at the root of a graph. -/
def Graph.toExpr {arity : Nat} (graph : Graph arity) : Expr arity :=
  getExpression graph.expressionNodes graph.root

/-- The variable support and multilinearity result for one graph node. -/
structure ExprSummary (arity : Nat) where
  support : Nat
  multilinear : Bool
  deriving DecidableEq

/-- Interpret the low `arity` bits of a natural number as variable indices. -/
def bitsToFinset {arity : Nat} (bits : Nat) : Finset (Fin arity) :=
  (Finset.univ : Finset (Fin arity)).filter fun index => bits.testBit index

@[simp]
private theorem bitsToFinset_zero {arity : Nat} :
    bitsToFinset (arity := arity) 0 = ∅ := by
  ext index
  simp [bitsToFinset]

@[simp]
private theorem bitsToFinset_or {arity : Nat} (left right : Nat) :
    bitsToFinset (arity := arity) (left ||| right) =
      bitsToFinset left ∪ bitsToFinset right := by
  ext index
  simp [bitsToFinset]

@[simp]
private theorem bitsToFinset_singleton {arity : Nat} (index : Fin arity) :
    bitsToFinset (arity := arity) (1 <<< index.val) = {index} := by
  rw [Nat.shiftLeft_eq, one_mul]
  ext candidate
  simp [bitsToFinset, Nat.testBit_two_pow, Fin.ext_iff, eq_comm]

private theorem singleton_lt {arity : Nat} (index : Fin arity) :
    1 <<< index.val < 2 ^ arity := by
  rw [Nat.shiftLeft_eq, one_mul]
  exact Nat.pow_lt_pow_of_lt (by omega) index.isLt

private theorem bitAnd_eq_zero_iff_disjoint {arity left right : Nat}
    (hleft : left < 2 ^ arity) (hright : right < 2 ^ arity) :
    left &&& right = 0 ↔
      Disjoint (bitsToFinset (arity := arity) left) (bitsToFinset right) := by
  rw [Finset.disjoint_left]
  constructor
  · intro h index hleftMem hrightMem
    have hbit := congrArg (fun bits : Nat => bits.testBit index) h
    simp [Nat.testBit_and] at hbit
    simp [bitsToFinset] at hleftMem hrightMem
    simp_all
  · intro h
    apply Nat.eq_of_testBit_eq
    intro i
    by_cases hi : i < arity
    · let index : Fin arity := ⟨i, hi⟩
      simp [bitsToFinset] at h
      cases hl : left.testBit i <;> cases hr : right.testBit i <;> simp_all
      have hfalse := h (a := index) (by simpa [index] using hl)
      simp_all [index]
    · have hi' : arity ≤ i := Nat.le_of_not_gt hi
      have hlefti : left < 2 ^ i :=
        hleft.trans_le (Nat.pow_le_pow_right (by omega) hi')
      have hrighti : right < 2 ^ i :=
        hright.trans_le (Nat.pow_le_pow_right (by omega) hi')
      simp [Nat.testBit_and, Nat.testBit_lt_two_pow hlefti,
        Nat.testBit_lt_two_pow hrighti]

/-- Compute an expression's support and multilinearity without duplicating shared subexpressions. -/
def _root_.Jolt.LookupExpression.Expr.summary
    {arity : Nat} : Expr arity → ExprSummary arity
  | .const _ => ⟨0, true⟩
  | .var index => ⟨1 <<< index.val, true⟩
  | .add left right | .sub left right =>
      let left := left.summary
      let right := right.summary
      ⟨left.support ||| right.support, left.multilinear && right.multilinear⟩
  | .mul left right =>
      let left := left.summary
      let right := right.summary
      ⟨left.support ||| right.support,
        left.multilinear && right.multilinear &&
          decide (left.support &&& right.support = 0)⟩

private theorem _root_.Jolt.LookupExpression.Expr.summary_support_lt
    {arity : Nat} (expression : Expr arity) :
    expression.summary.support < 2 ^ arity := by
  induction expression with
  | const => simp [Expr.summary]
  | var index => exact singleton_lt index
  | add left right hleft hright =>
      exact Nat.or_lt_two_pow hleft hright
  | sub left right hleft hright =>
      exact Nat.or_lt_two_pow hleft hright
  | mul left right hleft hright =>
      exact Nat.or_lt_two_pow hleft hright

@[simp]
theorem _root_.Jolt.LookupExpression.Expr.summary_variables
    {arity : Nat} (expression : Expr arity) :
    bitsToFinset expression.summary.support = expression.variables := by
  induction expression <;> simp_all [Expr.summary, Expr.variables]

@[simp]
theorem _root_.Jolt.LookupExpression.Expr.summary_multilinear
    {arity : Nat} (expression : Expr arity) :
    expression.summary.multilinear = expression.isSyntacticallyMultilinear := by
  induction expression with
  | const => rfl
  | var => rfl
  | add left right hleft hright | sub left right hleft hright =>
      simp only [Expr.summary, Expr.isSyntacticallyMultilinear, hleft, hright]
  | mul left right hleft hright =>
      simp only [Expr.summary, Expr.isSyntacticallyMultilinear, hleft, hright]
      simp only [bitAnd_eq_zero_iff_disjoint left.summary_support_lt
        right.summary_support_lt, left.summary_variables, right.summary_variables]

/-- Read a graph summary, returning the constant-zero summary out of bounds. -/
def getSummary {arity : Nat} (summaries : Array (ExprSummary arity))
    (index : Nat) : ExprSummary arity :=
  summaries[index]?.getD ⟨0, true⟩

/-- Summarize one node from summaries of the preceding nodes. -/
def Node.summary {arity : Nat} (node : Node)
    (summaries : Array (ExprSummary arity)) : ExprSummary arity :=
  match node with
  | .constant _ => ⟨0, true⟩
  | .input index =>
      if index < arity then ⟨1 <<< index, true⟩ else ⟨0, true⟩
  | .add left right | .sub left right =>
      let left := getSummary summaries left
      let right := getSummary summaries right
      ⟨left.support ||| right.support, left.multilinear && right.multilinear⟩
  | .mul left right =>
      let left := getSummary summaries left
      let right := getSummary summaries right
      ⟨left.support ||| right.support,
        left.multilinear && right.multilinear &&
          decide (left.support &&& right.support = 0)⟩

/-- Extend preceding summaries with the summaries of one graph chunk. -/
def Graph.summaryChunk {arity : Nat} (summaries : Array (ExprSummary arity))
    (nodes : List Node) : Array (ExprSummary arity) :=
  nodes.foldl (fun summaries node => summaries.push (node.summary summaries)) summaries

/-- Summarize every graph node while preserving graph sharing. -/
def Graph.summaryNodes {arity : Nat} (graph : Graph arity) : Array (ExprSummary arity) :=
  graph.nodeChunks.foldl Graph.summaryChunk #[]

/-- Check syntactic multilinearity directly on the shared graph. -/
def Graph.checkMultilinear {arity : Nat} (graph : Graph arity) : Bool :=
  (getSummary graph.summaryNodes graph.root).multilinear

private theorem getSummary_map_summary
    {arity : Nat} (expressions : Array (Expr arity)) (index : Nat) :
    getSummary (expressions.map Expr.summary) index = (getExpression expressions index).summary := by
  unfold getSummary getExpression
  rw [Array.getElem?_map]
  cases expressions[index]? <;> simp [Expr.summary]

private theorem Node.summary_map_toExpr
    {arity : Nat} (node : Node) (expressions : Array (Expr arity)) :
    node.summary (expressions.map Expr.summary) = (node.toExpr expressions).summary := by
  cases node with
  | constant value => rfl
  | input index =>
      simp only [Node.summary, Node.toExpr]
      split <;> rfl
  | add left right | sub left right | mul left right =>
      simp only [Node.summary, Node.toExpr, Expr.summary,
        getSummary_map_summary]

private theorem summaryChunk_eq_map_expressionChunk_summary
    {arity : Nat} (nodes : List Node) (expressions : Array (Expr arity)) :
    nodes.foldl
        (fun summaries node => summaries.push (node.summary summaries))
        (expressions.map Expr.summary) =
      (nodes.foldl
          (fun expressions node => expressions.push (node.toExpr expressions))
          expressions).map Expr.summary := by
  exact List.foldl_hom
    (l := nodes)
    (init := expressions)
    (g₁ := fun expressions node => expressions.push (node.toExpr expressions))
    (g₂ := fun summaries node => summaries.push (node.summary summaries))
    (fun expressions : Array (Expr arity) => expressions.map Expr.summary)
    (by intro values node; simp only [Array.map_push, Node.summary_map_toExpr])

theorem Graph.summaryNodes_eq_map_expressionNodes_summary
    {arity : Nat} (graph : Graph arity) :
    graph.summaryNodes = graph.expressionNodes.map Expr.summary := by
  unfold Graph.summaryNodes Graph.expressionNodes
  simpa only [Array.map_empty] using
    (List.foldl_hom
      (l := graph.nodeChunks)
      (fun expressions : Array (Expr arity) => expressions.map Expr.summary)
      (g₁ := fun expressions nodes =>
        nodes.foldl
          (fun expressions node => expressions.push (node.toExpr expressions))
          expressions)
      (g₂ := fun summaries nodes =>
        nodes.foldl
          (fun summaries node => summaries.push (node.summary summaries))
          summaries)
      (init := #[])
      (by
        intro expressions nodes
        exact summaryChunk_eq_map_expressionChunk_summary nodes expressions))

theorem Graph.checkMultilinear_eq_toExpr
    {arity : Nat} (graph : Graph arity) :
    graph.checkMultilinear = graph.toExpr.isSyntacticallyMultilinear := by
  unfold Graph.checkMultilinear Graph.toExpr
  rw [graph.summaryNodes_eq_map_expressionNodes_summary, getSummary_map_summary,
    Expr.summary_multilinear]

private theorem getExpression_eval
    {arity : Nat} {F : Type*} [CommRing F]
    (expressions : Array (Expr arity)) (point : Fin arity → F) (index : Nat) :
    (getExpression expressions index).eval point =
      getValue (expressions.map fun expression => expression.eval point) index := by
  unfold getExpression getValue
  rw [Array.getElem?_map]
  cases h : expressions[index]? <;> simp [Expr.eval]

private theorem Node.eval_map_toExpr
    {arity : Nat} {F : Type*} [CommRing F]
    (node : Node) (expressions : Array (Expr arity)) (point : Fin arity → F) :
    node.eval (expressions.map fun expression => expression.eval point) point =
      (node.toExpr expressions).eval point := by
  cases node with
  | constant value => rfl
  | input index =>
      simp only [Node.eval, Node.toExpr]
      split <;> simp [Expr.eval]
  | add left right =>
      simp only [Node.eval, Node.toExpr, Expr.eval]
      rw [getExpression_eval, getExpression_eval]
  | sub left right =>
      simp only [Node.eval, Node.toExpr, Expr.eval]
      rw [getExpression_eval, getExpression_eval]
  | mul left right =>
      simp only [Node.eval, Node.toExpr, Expr.eval]
      rw [getExpression_eval, getExpression_eval]

/-- Direct graph evaluation computes the values of the reconstructed expressions. -/
private theorem evalChunk_eq_map_expressionChunk_eval
    {arity : Nat} {F : Type*} [CommRing F]
    (nodes : List Node) (expressions : Array (Expr arity)) (point : Fin arity → F) :
    nodes.foldl
        (fun values node => values.push (node.eval values point))
        (expressions.map fun expression => expression.eval point) =
      (nodes.foldl
          (fun expressions node => expressions.push (node.toExpr expressions))
          expressions).map (fun expression => expression.eval point) := by
  exact List.foldl_hom
    (l := nodes)
    (init := expressions)
    (g₁ := fun expressions node => expressions.push (node.toExpr expressions))
    (g₂ := fun values node => values.push (node.eval values point))
    (fun expressions : Array (Expr arity) =>
      expressions.map fun expression => expression.eval point)
    (by
      intro values node
      simp only [Array.map_push, Node.eval_map_toExpr])

/-- Direct graph evaluation computes the values of the reconstructed expressions. -/
theorem Graph.evalNodes_eq_map_expressionNodes_eval
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : Graph arity) (point : Fin arity → F) :
    graph.evalNodes point =
      graph.expressionNodes.map (fun expression => expression.eval point) := by
  unfold Graph.evalNodes Graph.expressionNodes
  simpa only [Array.map_empty] using
    (List.foldl_hom
      (l := graph.nodeChunks)
      (fun expressions => expressions.map fun expression => expression.eval point)
      (g₁ := fun expressions nodes =>
        nodes.foldl
          (fun expressions node => expressions.push (node.toExpr expressions))
          expressions)
      (g₂ := fun values nodes =>
        nodes.foldl
          (fun values node => values.push (node.eval values point))
          values)
      (init := #[])
      (by
        intro expressions nodes
        exact evalChunk_eq_map_expressionChunk_eval nodes expressions point))

/-- Direct graph evaluation agrees with the reconstructed algebraic expression. -/
theorem Graph.eval_eq_toExpr_eval
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : Graph arity) (wellFormed : graph.WellFormed)
    (point : Fin arity → F) :
    graph.eval wellFormed point = graph.toExpr.eval point := by
  unfold Graph.eval Graph.toExpr
  rw [graph.evalNodes_eq_map_expressionNodes_eval]
  exact (getExpression_eval graph.expressionNodes point graph.root).symm

end Jolt.LookupGraph
