import Jolt.MaterializerGraph
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Data.List.Sort

/-!
# Checked algebraic correspondence

The two extracted graphs use the same recurrence but may associate or order
additions and multiplications differently. This file normalizes only those
commutative-ring identities. It deliberately does not distribute products, so
shared recurrence nodes stay compact.
-/

namespace Jolt.LookupAC

/-- An algebraic expression with flattened, sorted sums and products. -/
inductive ACExpr where
  | constant (value : Nat)
  | input (index : Nat)
  | add (terms : List ACExpr)
  | sub (left right : ACExpr)
  | mul (factors : List ACExpr)

mutual
  /-- Structural equality for normalized expressions. -/
  def ACExpr.equal : ACExpr → ACExpr → Bool
    | .constant left, .constant right => left == right
    | .input left, .input right => left == right
    | .add left, .add right => ACExpr.listEqual left right
    | .sub left₁ right₁, .sub left₂ right₂ =>
        left₁.equal left₂ && right₁.equal right₂
    | .mul left, .mul right => ACExpr.listEqual left right
    | _, _ => false

  /-- Structural equality for lists of normalized expressions. -/
  def ACExpr.listEqual : List ACExpr → List ACExpr → Bool
    | [], [] => true
    | left :: lefts, right :: rights => left.equal right && listEqual lefts rights
    | _, _ => false
end

mutual
  theorem ACExpr.equal_eq_true_iff (left right : ACExpr) :
      left.equal right = true ↔ left = right := by
    cases left <;> cases right <;>
      simp [ACExpr.equal, ACExpr.equal_eq_true_iff,
        ACExpr.listEqual_eq_true_iff]

  theorem ACExpr.listEqual_eq_true_iff (left right : List ACExpr) :
      ACExpr.listEqual left right = true ↔ left = right := by
    cases left <;> cases right <;>
      simp [ACExpr.listEqual, ACExpr.equal_eq_true_iff,
        ACExpr.listEqual_eq_true_iff]
end

/-- Constructor order used by the deterministic operand ordering. -/
def ACExpr.tag : ACExpr → Nat
  | .constant _ => 0
  | .input _ => 1
  | .add _ => 2
  | .sub _ _ => 3
  | .mul _ => 4

mutual
  /-- A deterministic structural comparison. -/
  def ACExpr.cmp (left right : ACExpr) : Ordering :=
    match compare left.tag right.tag with
    | .lt => .lt
    | .gt => .gt
    | .eq =>
        match left, right with
        | .constant left, .constant right => compare left right
        | .input left, .input right => compare left right
        | .add left, .add right => ACExpr.listCmp left right
        | .sub left₁ right₁, .sub left₂ right₂ =>
            match left₁.cmp left₂ with
            | .eq => right₁.cmp right₂
            | result => result
        | .mul left, .mul right => ACExpr.listCmp left right
        | _, _ => .eq

  /-- Lexicographic comparison for normalized expression lists. -/
  def ACExpr.listCmp : List ACExpr → List ACExpr → Ordering
    | [], [] => .eq
    | [], _ :: _ => .lt
    | _ :: _, [] => .gt
    | left :: lefts, right :: rights =>
        match left.cmp right with
        | .eq => listCmp lefts rights
        | result => result
end

/-- Read an input coordinate, returning zero for an invalid index. -/
def getInput {arity : Nat} {F : Type*} [Zero F]
    (point : Fin arity → F) (index : Nat) : F :=
  if h : index < arity then point ⟨index, h⟩ else 0

/-- Evaluate a normalized expression. -/
def ACExpr.eval
    {arity : Nat} {F : Type*} [CommRing F]
    (expression : ACExpr) (point : Fin arity → F) : F :=
  match expression with
  | .constant value => value
  | .input index => getInput point index
  | .add terms => (terms.map fun term => term.eval point).sum
  | .sub left right => left.eval point - right.eval point
  | .mul factors => (factors.map fun factor => factor.eval point).prod

/-- A deterministic ordering for sum and product operands. -/
def ACExpr.sort (expressions : List ACExpr) : List ACExpr :=
  expressions.mergeSort fun left right => left.cmp right ≠ Ordering.gt

/-- Flatten one level of a sum. -/
def ACExpr.addTerms : ACExpr → List ACExpr
  | .add terms => terms
  | expression => [expression]

/-- Flatten one level of a product. -/
def ACExpr.mulFactors : ACExpr → List ACExpr
  | .mul factors => factors
  | expression => [expression]

/-- Canonical addition modulo associativity and commutativity. -/
def ACExpr.smartAdd (left right : ACExpr) : ACExpr :=
  match left, right with
  | .constant 0, expression | expression, .constant 0 => expression
  | left, right => .add (ACExpr.sort (left.addTerms ++ right.addTerms))

/-- Canonical multiplication modulo associativity and commutativity. -/
def ACExpr.smartMul (left right : ACExpr) : ACExpr :=
  match left, right with
  | .constant 0, _ | _, .constant 0 => .constant 0
  | .constant 1, expression | expression, .constant 1 => expression
  | left, right => .mul (ACExpr.sort (left.mulFactors ++ right.mulFactors))

@[simp]
private theorem ACExpr.sum_eval_sort
    {arity : Nat} {F : Type*} [CommRing F]
    (expressions : List ACExpr) (point : Fin arity → F) :
    ((ACExpr.sort expressions).map fun expression => expression.eval point).sum =
      (expressions.map fun expression => expression.eval point).sum := by
  exact ((List.mergeSort_perm expressions _).map _).sum_eq

@[simp]
private theorem ACExpr.prod_eval_sort
    {arity : Nat} {F : Type*} [CommRing F]
    (expressions : List ACExpr) (point : Fin arity → F) :
    ((ACExpr.sort expressions).map fun expression => expression.eval point).prod =
      (expressions.map fun expression => expression.eval point).prod := by
  exact ((List.mergeSort_perm expressions _).map _).prod_eq

theorem ACExpr.eval_smartAdd
    {arity : Nat} {F : Type*} [CommRing F]
    (left right : ACExpr) (point : Fin arity → F) :
    (left.smartAdd right).eval point = left.eval point + right.eval point := by
  cases left <;> cases right <;>
    simp [ACExpr.smartAdd, ACExpr.addTerms, ACExpr.eval,
      ACExpr.sum_eval_sort, List.sum_append] <;>
    split <;> simp_all [ACExpr.eval]

theorem ACExpr.eval_smartMul
    {arity : Nat} {F : Type*} [CommRing F]
    (left right : ACExpr) (point : Fin arity → F) :
    (left.smartMul right).eval point = left.eval point * right.eval point := by
  cases left <;> cases right <;>
    simp [ACExpr.smartMul, ACExpr.mulFactors, ACExpr.eval,
      ACExpr.prod_eval_sort, List.prod_append] <;>
    split <;> simp_all [ACExpr.eval]

/-- Read a normalized expression, returning zero outside the array. -/
def getExpression (values : Array ACExpr) (index : Nat) : ACExpr :=
  values[index]?.getD (.constant 0)

theorem getExpression_eval
    {arity : Nat} {F : Type*} [CommRing F]
    (values : Array ACExpr) (point : Fin arity → F) (index : Nat) :
    (getExpression values index).eval point =
      Jolt.LookupGraph.getValue (values.map fun expression => expression.eval point) index := by
  unfold getExpression Jolt.LookupGraph.getValue
  rw [Array.getElem?_map]
  cases h : values[index]? <;> simp [ACExpr.eval]

/-- Normalize one verifier graph node from its preceding nodes. -/
def _root_.Jolt.LookupGraph.Node.toACExpr
    (node : Jolt.LookupGraph.Node) (values : Array ACExpr) : ACExpr :=
  match node with
  | .constant value => .constant value
  | .input index => .input index
  | .add left right => (getExpression values left).smartAdd (getExpression values right)
  | .sub left right => .sub (getExpression values left) (getExpression values right)
  | .mul left right => (getExpression values left).smartMul (getExpression values right)

/-- Normalize every verifier graph node while preserving graph sharing. -/
def _root_.Jolt.LookupGraph.Graph.acNodes
    {arity : Nat} (graph : Jolt.LookupGraph.Graph arity) : Array ACExpr :=
  graph.nodeChunks.foldl
    (fun values nodes => nodes.foldl
      (fun values node => values.push (node.toACExpr values)) values)
    #[]

/-- The canonical associative-commutative expression denoted by a verifier graph. -/
def _root_.Jolt.LookupGraph.Graph.toACExpr
    {arity : Nat} (graph : Jolt.LookupGraph.Graph arity) : ACExpr :=
  getExpression graph.acNodes graph.root

private theorem _root_.Jolt.LookupGraph.Node.eval_map_toACExpr
    {arity : Nat} {F : Type*} [CommRing F]
    (node : Jolt.LookupGraph.Node) (values : Array ACExpr) (point : Fin arity → F) :
    node.eval (values.map fun expression => expression.eval point) point =
      (node.toACExpr values).eval point := by
  cases node with
  | constant value => simp [Jolt.LookupGraph.Node.eval,
      Jolt.LookupGraph.Node.toACExpr, ACExpr.eval]
  | input index =>
      simp only [Jolt.LookupGraph.Node.eval, Jolt.LookupGraph.Node.toACExpr, ACExpr.eval]
      unfold getInput
      split <;> simp_all
  | add left right =>
      simp only [Jolt.LookupGraph.Node.eval, Jolt.LookupGraph.Node.toACExpr]
      rw [ACExpr.eval_smartAdd, getExpression_eval, getExpression_eval]
  | sub left right =>
      simp only [Jolt.LookupGraph.Node.eval, Jolt.LookupGraph.Node.toACExpr, ACExpr.eval]
      rw [getExpression_eval, getExpression_eval]
  | mul left right =>
      simp only [Jolt.LookupGraph.Node.eval, Jolt.LookupGraph.Node.toACExpr]
      rw [ACExpr.eval_smartMul, getExpression_eval, getExpression_eval]

private theorem evalGraphChunk_eq_map_acChunk_eval
    {arity : Nat} {F : Type*} [CommRing F]
    (nodes : List Jolt.LookupGraph.Node) (values : Array ACExpr)
    (point : Fin arity → F) :
    nodes.foldl (fun values node => values.push (node.eval values point))
        (values.map fun expression => expression.eval point) =
      (nodes.foldl (fun values node => values.push (node.toACExpr values)) values).map
        (fun expression => expression.eval point) := by
  exact List.foldl_hom
    (l := nodes)
    (init := values)
    (g₁ := fun values node => values.push (node.toACExpr values))
    (g₂ := fun values node => values.push (node.eval values point))
    (fun values : Array ACExpr => values.map fun expression => expression.eval point)
    (by intro values node; simp only [Array.map_push, Jolt.LookupGraph.Node.eval_map_toACExpr])

theorem _root_.Jolt.LookupGraph.Graph.eval_eq_acEval
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : Jolt.LookupGraph.Graph arity) (wellFormed : graph.WellFormed)
    (point : Fin arity → F) :
    graph.eval wellFormed point = graph.toACExpr.eval point := by
  unfold Jolt.LookupGraph.Graph.eval Jolt.LookupGraph.Graph.toACExpr
  have hnodes : graph.evalNodes point =
      graph.acNodes.map (fun expression => expression.eval point) := by
    unfold Jolt.LookupGraph.Graph.evalNodes Jolt.LookupGraph.Graph.acNodes
    simpa only [Array.map_empty] using
      (List.foldl_hom
        (l := graph.nodeChunks)
        (fun values : Array ACExpr => values.map fun expression => expression.eval point)
        (g₁ := fun values nodes => nodes.foldl
          (fun values node => values.push (node.toACExpr values)) values)
        (g₂ := fun values nodes => nodes.foldl
          (fun values node => values.push (node.eval values point)) values)
        (init := #[])
        (by intro values nodes; exact evalGraphChunk_eq_map_acChunk_eval nodes values point))
  rw [hnodes, getExpression_eval]

/-- Normalize one Boolean materializer node. -/
def _root_.Jolt.LookupExpression.MaterializerBoolNode.toACExpr
    (node : Jolt.LookupExpression.MaterializerBoolNode)
    (values : Array ACExpr) : ACExpr :=
  match node with
  | .input index => .input index
  | .conj left right => (getExpression values left).smartMul (getExpression values right)
  | .neg value => .sub (.constant 1) (getExpression values value)

/-- Normalize one natural-number materializer node. -/
def _root_.Jolt.LookupExpression.MaterializerNatNode.toACExpr
    (node : Jolt.LookupExpression.MaterializerNatNode)
    (boolValues values : Array ACExpr) : ACExpr :=
  match node with
  | .constant value => .constant value
  | .ofBit value => getExpression boolValues value
  | .add left right => (getExpression values left).smartAdd (getExpression values right)
  | .mul left right => (getExpression values left).smartMul (getExpression values right)

/-- Normalize every Boolean materializer node. -/
def _root_.Jolt.LookupExpression.MaterializerGraph.boolACNodes
    {arity : Nat} (graph : Jolt.LookupExpression.MaterializerGraph arity) : Array ACExpr :=
  graph.boolNodeChunks.foldl
    (fun values nodes => nodes.foldl
      (fun values node => values.push (node.toACExpr values)) values)
    #[]

/-- Normalize every natural-number materializer node. -/
def _root_.Jolt.LookupExpression.MaterializerGraph.natACNodes
    {arity : Nat} (graph : Jolt.LookupExpression.MaterializerGraph arity)
    (boolValues : Array ACExpr) : Array ACExpr :=
  graph.natNodeChunks.foldl
    (fun values nodes => nodes.foldl
      (fun values node => values.push (node.toACExpr boolValues values)) values)
    #[]

/-- The canonical associative-commutative expression denoted by a materializer. -/
def _root_.Jolt.LookupExpression.MaterializerGraph.toACExpr
    {arity : Nat} (graph : Jolt.LookupExpression.MaterializerGraph arity) : ACExpr :=
  getExpression (graph.natACNodes graph.boolACNodes) graph.root

private theorem _root_.Jolt.LookupExpression.MaterializerBoolNode.evalArith_map_toACExpr
    {arity : Nat} {F : Type*} [CommRing F]
    (node : Jolt.LookupExpression.MaterializerBoolNode)
    (values : Array ACExpr) (point : Fin arity → F) :
    node.evalArith (values.map fun expression => expression.eval point) point =
      (node.toACExpr values).eval point := by
  cases node with
  | input index =>
      simp only [Jolt.LookupExpression.MaterializerBoolNode.evalArith,
        Jolt.LookupExpression.MaterializerBoolNode.toACExpr, ACExpr.eval]
      unfold getInput
      split <;> simp_all
  | conj left right =>
      simp only [Jolt.LookupExpression.MaterializerBoolNode.evalArith,
        Jolt.LookupExpression.MaterializerBoolNode.toACExpr]
      rw [ACExpr.eval_smartMul, getExpression_eval, getExpression_eval]
  | neg value =>
      simp only [Jolt.LookupExpression.MaterializerBoolNode.evalArith,
        Jolt.LookupExpression.MaterializerBoolNode.toACExpr, ACExpr.eval]
      rw [getExpression_eval]
      simp only [Nat.cast_one]

private theorem evalMaterializerBoolChunk_eq_map_acChunk_eval
    {arity : Nat} {F : Type*} [CommRing F]
    (nodes : List Jolt.LookupExpression.MaterializerBoolNode)
    (values : Array ACExpr) (point : Fin arity → F) :
    nodes.foldl (fun values node => values.push (node.evalArith values point))
        (values.map fun expression => expression.eval point) =
      (nodes.foldl (fun values node => values.push (node.toACExpr values)) values).map
        (fun expression => expression.eval point) := by
  exact List.foldl_hom
    (l := nodes)
    (init := values)
    (g₁ := fun values node => values.push (node.toACExpr values))
    (g₂ := fun values node => values.push (node.evalArith values point))
    (fun values : Array ACExpr => values.map fun expression => expression.eval point)
    (by intro values node; simp only [Array.map_push,
      Jolt.LookupExpression.MaterializerBoolNode.evalArith_map_toACExpr])

private theorem _root_.Jolt.LookupExpression.MaterializerNatNode.evalArith_map_toACExpr
    {arity : Nat} {F : Type*} [CommRing F]
    (node : Jolt.LookupExpression.MaterializerNatNode)
    (boolValues values : Array ACExpr) (point : Fin arity → F) :
    node.evalArith
        (boolValues.map fun expression => expression.eval point)
        (values.map fun expression => expression.eval point) =
      (node.toACExpr boolValues values).eval point := by
  cases node with
  | constant value => simp [Jolt.LookupExpression.MaterializerNatNode.evalArith,
      Jolt.LookupExpression.MaterializerNatNode.toACExpr, ACExpr.eval]
  | ofBit value =>
      simp only [Jolt.LookupExpression.MaterializerNatNode.evalArith,
        Jolt.LookupExpression.MaterializerNatNode.toACExpr]
      rw [getExpression_eval]
  | add left right =>
      simp only [Jolt.LookupExpression.MaterializerNatNode.evalArith,
        Jolt.LookupExpression.MaterializerNatNode.toACExpr]
      rw [ACExpr.eval_smartAdd, getExpression_eval, getExpression_eval]
  | mul left right =>
      simp only [Jolt.LookupExpression.MaterializerNatNode.evalArith,
        Jolt.LookupExpression.MaterializerNatNode.toACExpr]
      rw [ACExpr.eval_smartMul, getExpression_eval, getExpression_eval]

private theorem evalMaterializerNatChunk_eq_map_acChunk_eval
    {arity : Nat} {F : Type*} [CommRing F]
    (nodes : List Jolt.LookupExpression.MaterializerNatNode)
    (boolValues values : Array ACExpr) (point : Fin arity → F) :
    nodes.foldl
        (fun values node => values.push
          (node.evalArith (boolValues.map fun expression => expression.eval point) values))
        (values.map fun expression => expression.eval point) =
      (nodes.foldl
          (fun values node => values.push (node.toACExpr boolValues values)) values).map
        (fun expression => expression.eval point) := by
  exact List.foldl_hom
    (l := nodes)
    (init := values)
    (g₁ := fun values node => values.push (node.toACExpr boolValues values))
    (g₂ := fun values node => values.push
      (node.evalArith (boolValues.map fun expression => expression.eval point) values))
    (fun values : Array ACExpr => values.map fun expression => expression.eval point)
    (by intro values node; simp only [Array.map_push,
      Jolt.LookupExpression.MaterializerNatNode.evalArith_map_toACExpr])

theorem _root_.Jolt.LookupExpression.MaterializerGraph.arithEval_eq_acEval
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : Jolt.LookupExpression.MaterializerGraph arity) (point : Fin arity → F) :
    graph.arithEval point = graph.toACExpr.eval point := by
  unfold Jolt.LookupExpression.MaterializerGraph.arithEval
    Jolt.LookupExpression.MaterializerGraph.toACExpr
  have hbool : graph.evalBoolNodesArith point =
      graph.boolACNodes.map (fun expression => expression.eval point) := by
    unfold Jolt.LookupExpression.MaterializerGraph.evalBoolNodesArith
      Jolt.LookupExpression.MaterializerGraph.boolACNodes
    simpa only [Array.map_empty] using
      (List.foldl_hom
        (l := graph.boolNodeChunks)
        (fun values : Array ACExpr => values.map fun expression => expression.eval point)
        (g₁ := fun values nodes => nodes.foldl
          (fun values node => values.push (node.toACExpr values)) values)
        (g₂ := fun values nodes => nodes.foldl
          (fun values node => values.push (node.evalArith values point)) values)
        (init := #[])
        (by intro values nodes; exact
          evalMaterializerBoolChunk_eq_map_acChunk_eval nodes values point))
  rw [hbool]
  have hnat : graph.evalNatNodesArith
        (graph.boolACNodes.map fun expression => expression.eval point) =
      (graph.natACNodes graph.boolACNodes).map
        (fun expression => expression.eval point) := by
    unfold Jolt.LookupExpression.MaterializerGraph.evalNatNodesArith
      Jolt.LookupExpression.MaterializerGraph.natACNodes
    simpa only [Array.map_empty] using
      (List.foldl_hom
        (l := graph.natNodeChunks)
        (fun values : Array ACExpr => values.map fun expression => expression.eval point)
        (g₁ := fun values nodes => nodes.foldl
          (fun values node => values.push (node.toACExpr graph.boolACNodes values)) values)
        (g₂ := fun values nodes => nodes.foldl
          (fun values node => values.push
            (node.evalArith
              (graph.boolACNodes.map (fun expression => expression.eval point)) values)) values)
        (init := #[])
        (by
          intro values nodes
          exact evalMaterializerNatChunk_eq_map_acChunk_eval
            nodes graph.boolACNodes values point))
  rw [hnat, getExpression_eval]

end Jolt.LookupAC
