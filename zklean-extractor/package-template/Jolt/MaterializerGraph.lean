import Jolt.LookupGraphExpression

/-!
# Shared lookup materializer graphs

The extractor records Boolean and natural-number materializer operations as
topologically ordered graphs. Shared recurrence values occur once. The main
theorem proves that Boolean execution and field arithmetic agree for every
generated graph.
-/

namespace Jolt.LookupExpression

/-- One Boolean operation in a materializer graph. -/
inductive MaterializerBoolNode where
  | input (index : Nat)
  | conj (left right : Nat)
  | neg (value : Nat)
  deriving DecidableEq

/-- Check the bounds of one Boolean node. -/
def MaterializerBoolNode.wellFormedAt (arity position : Nat) : MaterializerBoolNode → Bool
  | .input index => index < arity
  | .conj left right => left < position && right < position
  | .neg value => value < position

private def boolNodesWellFormed (arity : Nat) : List MaterializerBoolNode → Nat → Bool
  | [], _ => true
  | node :: nodes, position =>
      node.wellFormedAt arity position && boolNodesWellFormed arity nodes (position + 1)

/-- Read a Boolean node value, returning false outside the array. -/
def getBoolValue (values : Array Bool) (index : Nat) : Bool :=
  values[index]?.getD false

/-- Evaluate one Boolean node. -/
def MaterializerBoolNode.eval
    {arity : Nat} (node : MaterializerBoolNode)
    (values : Array Bool) (point : Fin arity → Bool) : Bool :=
  match node with
  | .input index => if h : index < arity then point ⟨index, h⟩ else false
  | .conj left right => getBoolValue values left && getBoolValue values right
  | .neg value => !getBoolValue values value

/-- Evaluate one Boolean node after arithmetization. -/
def MaterializerBoolNode.evalArith
    {arity : Nat} {F : Type*} [CommRing F]
    (node : MaterializerBoolNode) (values : Array F) (point : Fin arity → F) : F :=
  match node with
  | .input index => if h : index < arity then point ⟨index, h⟩ else 0
  | .conj left right =>
      Jolt.LookupGraph.getValue values left * Jolt.LookupGraph.getValue values right
  | .neg value => 1 - Jolt.LookupGraph.getValue values value

/-- One natural-number operation in a materializer graph. -/
inductive MaterializerNatNode where
  | constant (value : Nat)
  | ofBit (value : Nat)
  | add (left right : Nat)
  | mul (left right : Nat)
  deriving DecidableEq

/-- Check the bounds of one natural-number node. -/
def MaterializerNatNode.wellFormedAt
    (boolCount position : Nat) : MaterializerNatNode → Bool
  | .constant _ => true
  | .ofBit value => value < boolCount
  | .add left right | .mul left right => left < position && right < position

private def natNodesWellFormed (boolCount : Nat) : List MaterializerNatNode → Nat → Bool
  | [], _ => true
  | node :: nodes, position =>
      node.wellFormedAt boolCount position && natNodesWellFormed boolCount nodes (position + 1)

/-- Read a natural-number node value, returning zero outside the array. -/
def getNatValue (values : Array Nat) (index : Nat) : Nat :=
  values[index]?.getD 0

/-- Evaluate one natural-number node. -/
def MaterializerNatNode.eval
    (node : MaterializerNatNode) (boolValues : Array Bool) (values : Array Nat) : Nat :=
  match node with
  | .constant value => value
  | .ofBit value => (getBoolValue boolValues value).toNat
  | .add left right => getNatValue values left + getNatValue values right
  | .mul left right => getNatValue values left * getNatValue values right

/-- Evaluate one natural-number node in a commutative ring. -/
def MaterializerNatNode.evalArith
    {F : Type*} [CommRing F]
    (node : MaterializerNatNode) (boolValues values : Array F) : F :=
  match node with
  | .constant value => value
  | .ofBit value => Jolt.LookupGraph.getValue boolValues value
  | .add left right =>
      Jolt.LookupGraph.getValue values left + Jolt.LookupGraph.getValue values right
  | .mul left right =>
      Jolt.LookupGraph.getValue values left * Jolt.LookupGraph.getValue values right

/-- A compact materializer with separate Boolean and natural-number graphs. -/
structure MaterializerGraph (arity : Nat) where
  boolNodeChunks : List (List MaterializerBoolNode)
  natNodeChunks : List (List MaterializerNatNode)
  root : Nat
  deriving DecidableEq

/-- Boolean nodes in execution order. -/
def MaterializerGraph.boolNodes {arity : Nat} (graph : MaterializerGraph arity) :
    List MaterializerBoolNode :=
  graph.boolNodeChunks.flatten

/-- Natural-number nodes in execution order. -/
def MaterializerGraph.natNodes {arity : Nat} (graph : MaterializerGraph arity) :
    List MaterializerNatNode :=
  graph.natNodeChunks.flatten

/-- Check input bounds, backward references, and the root bound. -/
def MaterializerGraph.wellFormed {arity : Nat} (graph : MaterializerGraph arity) : Bool :=
  boolNodesWellFormed arity graph.boolNodes 0 &&
    natNodesWellFormed graph.boolNodes.length graph.natNodes 0 &&
    graph.root < graph.natNodes.length

/-- A materializer graph is well formed when its bounds check succeeds. -/
def MaterializerGraph.WellFormed {arity : Nat} (graph : MaterializerGraph arity) : Prop :=
  graph.wellFormed = true

instance MaterializerGraph.instDecidableWellFormed
    {arity : Nat} (graph : MaterializerGraph arity) : Decidable graph.WellFormed :=
  inferInstanceAs (Decidable (graph.wellFormed = true))

/-- Evaluate all Boolean nodes. -/
def MaterializerGraph.evalBoolNodes
    {arity : Nat} (graph : MaterializerGraph arity) (point : Fin arity → Bool) : Array Bool :=
  graph.boolNodeChunks.foldl
    (fun values nodes => nodes.foldl (fun values node => values.push (node.eval values point)) values)
    #[]

/-- Evaluate all natural-number nodes. -/
def MaterializerGraph.evalNatNodes
    {arity : Nat} (graph : MaterializerGraph arity) (boolValues : Array Bool) : Array Nat :=
  graph.natNodeChunks.foldl
    (fun values nodes => nodes.foldl (fun values node => values.push (node.eval boolValues values)) values)
    #[]

/-- Evaluate the materializer on Boolean input. -/
def MaterializerGraph.eval
    {arity : Nat} (graph : MaterializerGraph arity) (point : Fin arity → Bool) : Nat :=
  getNatValue (graph.evalNatNodes (graph.evalBoolNodes point)) graph.root

/-- Evaluate all arithmetized Boolean nodes. -/
def MaterializerGraph.evalBoolNodesArith
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : MaterializerGraph arity) (point : Fin arity → F) : Array F :=
  graph.boolNodeChunks.foldl
    (fun values nodes =>
      nodes.foldl (fun values node => values.push (node.evalArith values point)) values)
    #[]

/-- Evaluate all arithmetized natural-number nodes. -/
def MaterializerGraph.evalNatNodesArith
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : MaterializerGraph arity) (boolValues : Array F) : Array F :=
  graph.natNodeChunks.foldl
    (fun values nodes =>
      nodes.foldl (fun values node => values.push (node.evalArith boolValues values)) values)
    #[]

/-- Evaluate the materializer's arithmetic interpretation. -/
def MaterializerGraph.arithEval
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : MaterializerGraph arity) (point : Fin arity → F) : F :=
  Jolt.LookupGraph.getValue
    (graph.evalNatNodesArith (graph.evalBoolNodesArith point)) graph.root

private theorem getBoolValue_cast
    {F : Type*} [CommRing F] (values : Array Bool) (index : Nat) :
    boolCast (F := F) (getBoolValue values index) =
      Jolt.LookupGraph.getValue (values.map fun value => boolCast (F := F) value) index := by
  unfold getBoolValue Jolt.LookupGraph.getValue
  rw [Array.getElem?_map]
  cases h : values[index]? <;> simp [boolCast]

private theorem getNatValue_cast
    {F : Type*} [CommRing F] (values : Array Nat) (index : Nat) :
    (getNatValue values index : F) =
      Jolt.LookupGraph.getValue (values.map fun (value : Nat) => (value : F)) index := by
  unfold getNatValue Jolt.LookupGraph.getValue
  rw [Array.getElem?_map]
  cases h : values[index]? <;> simp

private theorem MaterializerBoolNode.evalArith_map
    {arity : Nat} {F : Type*} [CommRing F]
    (node : MaterializerBoolNode) (values : Array Bool) (point : Fin arity → Bool) :
    node.evalArith (values.map fun value => boolCast (F := F) value)
        (fun index => boolCast (F := F) (point index)) =
      boolCast (F := F) (node.eval values point) := by
  cases node with
  | input index =>
      simp only [MaterializerBoolNode.evalArith, MaterializerBoolNode.eval]
      split <;> simp [boolCast]
  | conj left right =>
      simp only [MaterializerBoolNode.evalArith, MaterializerBoolNode.eval]
      rw [← getBoolValue_cast, ← getBoolValue_cast]
      cases getBoolValue values left <;> cases getBoolValue values right <;> simp [boolCast]
  | neg value =>
      simp only [MaterializerBoolNode.evalArith, MaterializerBoolNode.eval]
      rw [← getBoolValue_cast]
      cases getBoolValue values value <;> simp [boolCast]

private theorem evalBoolChunkArith_eq_map_eval
    {arity : Nat} {F : Type*} [CommRing F]
    (nodes : List MaterializerBoolNode) (values : Array Bool) (point : Fin arity → Bool) :
    nodes.foldl
        (fun values node => values.push
          (node.evalArith values (fun index => boolCast (F := F) (point index))))
        (values.map fun value => boolCast (F := F) value) =
      (nodes.foldl (fun values node => values.push (node.eval values point)) values).map
        (fun value => boolCast (F := F) value) := by
  exact List.foldl_hom
    (l := nodes)
    (init := values)
    (g₁ := fun values node => values.push (node.eval values point))
    (g₂ := fun values node => values.push
      (node.evalArith values (fun index => boolCast (F := F) (point index))))
    (fun values : Array Bool => values.map fun value => boolCast (F := F) value)
    (by
      intro values node
      simp only [Array.map_push, MaterializerBoolNode.evalArith_map])

private theorem MaterializerGraph.evalBoolNodesArith_eq_map_eval
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : MaterializerGraph arity) (point : Fin arity → Bool) :
    graph.evalBoolNodesArith (F := F) (fun index => boolCast (point index)) =
      (graph.evalBoolNodes point).map fun value => boolCast value := by
  unfold MaterializerGraph.evalBoolNodesArith MaterializerGraph.evalBoolNodes
  simpa only [Array.map_empty] using
    (List.foldl_hom
      (l := graph.boolNodeChunks)
      (fun values : Array Bool => values.map fun value => boolCast (F := F) value)
      (g₁ := fun values nodes =>
        nodes.foldl (fun values node => values.push (node.eval values point)) values)
      (g₂ := fun values nodes => nodes.foldl
        (fun values node => values.push
          (node.evalArith values (fun index => boolCast (F := F) (point index)))) values)
      (init := #[])
      (by
        intro values nodes
        exact evalBoolChunkArith_eq_map_eval nodes values point))

private theorem MaterializerNatNode.evalArith_map
    {F : Type*} [CommRing F]
    (node : MaterializerNatNode) (boolValues : Array Bool) (values : Array Nat) :
    node.evalArith
        (boolValues.map fun value => boolCast (F := F) value)
        (values.map fun (value : Nat) => (value : F)) =
      (node.eval boolValues values : F) := by
  cases node with
  | constant => rfl
  | ofBit value =>
      simpa only [MaterializerNatNode.evalArith, MaterializerNatNode.eval,
        boolCast_eq_natCast] using (getBoolValue_cast (F := F) boolValues value).symm
  | add left right =>
      simp only [MaterializerNatNode.evalArith, MaterializerNatNode.eval, Nat.cast_add]
      rw [← getNatValue_cast, ← getNatValue_cast]
  | mul left right =>
      simp only [MaterializerNatNode.evalArith, MaterializerNatNode.eval, Nat.cast_mul]
      rw [← getNatValue_cast, ← getNatValue_cast]

private theorem evalNatChunkArith_eq_map_eval
    {F : Type*} [CommRing F]
    (nodes : List MaterializerNatNode) (boolValues : Array Bool) (values : Array Nat) :
    nodes.foldl
        (fun values node => values.push
          (node.evalArith (boolValues.map fun value => boolCast (F := F) value) values))
        (values.map fun (value : Nat) => (value : F)) =
      (nodes.foldl (fun values node => values.push (node.eval boolValues values)) values).map
        (fun (value : Nat) => (value : F)) := by
  exact List.foldl_hom
    (l := nodes)
    (init := values)
    (g₁ := fun values node => values.push (node.eval boolValues values))
    (g₂ := fun values node => values.push
      (node.evalArith (boolValues.map fun value => boolCast (F := F) value) values))
    (fun values : Array Nat => values.map fun (value : Nat) => (value : F))
    (by
      intro values node
      simp only [Array.map_push, MaterializerNatNode.evalArith_map])

private theorem MaterializerGraph.evalNatNodesArith_eq_map_eval
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : MaterializerGraph arity) (boolValues : Array Bool) :
    graph.evalNatNodesArith (F := F)
        (boolValues.map fun value => boolCast (F := F) value) =
      (graph.evalNatNodes boolValues).map fun (value : Nat) => (value : F) := by
  unfold MaterializerGraph.evalNatNodesArith MaterializerGraph.evalNatNodes
  simpa only [Array.map_empty] using
    (List.foldl_hom
      (l := graph.natNodeChunks)
      (fun values : Array Nat => values.map fun (value : Nat) => (value : F))
      (g₁ := fun values nodes =>
        nodes.foldl (fun values node => values.push (node.eval boolValues values)) values)
      (g₂ := fun values nodes => nodes.foldl
        (fun values node => values.push
          (node.evalArith (boolValues.map fun value => boolCast (F := F) value) values)) values)
      (init := #[])
      (by
        intro values nodes
        exact evalNatChunkArith_eq_map_eval nodes boolValues values))

/-- Boolean materializer execution agrees with arithmetic execution in every ring. -/
theorem MaterializerGraph.cast_eval
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : MaterializerGraph arity) (point : Fin arity → Bool) :
    (graph.eval point : F) =
      graph.arithEval (fun index => boolCast (F := F) (point index)) := by
  unfold MaterializerGraph.eval MaterializerGraph.arithEval
  rw [graph.evalBoolNodesArith_eq_map_eval,
    graph.evalNatNodesArith_eq_map_eval, getNatValue_cast]

end Jolt.LookupExpression
