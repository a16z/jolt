import Jolt.LookupExpression
import Jolt.LookupGraph

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
