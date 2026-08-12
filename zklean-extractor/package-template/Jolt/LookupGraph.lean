import Mathlib.Algebra.Ring.Defs

/-!
# Shared lookup expression graphs

The extractor writes lookup polynomials as data in this small language. Each
node may refer only to an earlier node, so a graph records a shared
subexpression once. One static interpreter evaluates every generated table.
-/

namespace Jolt.LookupGraph

/-- One operation in a shared algebraic expression graph. -/
inductive Node where
  | constant (value : Nat)
  | input (index : Nat)
  | add (left right : Nat)
  | sub (left right : Nat)
  | mul (left right : Nat)
  deriving DecidableEq

/-- Check the bounds of one node at its position in a graph. -/
def Node.wellFormedAt (arity position : Nat) : Node → Bool
  | .constant _ => true
  | .input index => index < arity
  | .add left right | .sub left right | .mul left right =>
      left < position && right < position

private def nodesWellFormed (arity : Nat) : List Node → Nat → Bool
  | [], _ => true
  | node :: nodes, position =>
      node.wellFormedAt arity position && nodesWellFormed arity nodes (position + 1)

/-- A topologically ordered algebraic expression graph with `arity` inputs. -/
structure Graph (arity : Nat) where
  nodeChunks : List (List Node)
  root : Nat
  deriving DecidableEq

/-- The graph nodes in execution order. -/
def Graph.nodes {arity : Nat} (graph : Graph arity) : List Node :=
  graph.nodeChunks.flatten

/-- Check input bounds, backward references, and the root bound. -/
def Graph.wellFormed {arity : Nat} (graph : Graph arity) : Bool :=
  nodesWellFormed arity graph.nodes 0 && graph.root < graph.nodes.length

/-- A graph is well formed when its executable bounds check returns true. -/
def Graph.WellFormed {arity : Nat} (graph : Graph arity) : Prop :=
  graph.wellFormed = true

instance Graph.instDecidableWellFormed {arity : Nat} (graph : Graph arity) :
    Decidable graph.WellFormed :=
  inferInstanceAs (Decidable (graph.wellFormed = true))

/-- Read an array value, returning zero when the index is out of bounds. -/
def getValue {F : Type*} [Zero F] (values : Array F) (index : Nat) : F :=
  values[index]?.getD 0

/-- Evaluate one node from the values of the preceding nodes. -/
def Node.eval
    {arity : Nat} {F : Type*} [CommRing F]
    (node : Node) (values : Array F) (point : Fin arity → F) : F :=
  match node with
  | .constant value => value
  | .input index => if h : index < arity then point ⟨index, h⟩ else 0
  | .add left right => getValue values left + getValue values right
  | .sub left right => getValue values left - getValue values right
  | .mul left right => getValue values left * getValue values right

/-- Evaluate all nodes in topological order. -/
def Graph.evalNodes
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : Graph arity) (point : Fin arity → F) : Array F :=
  graph.nodeChunks.foldl
    (fun values nodes =>
      nodes.foldl (fun values node => values.push (node.eval values point)) values)
    #[]

/-- Evaluate the root of a shared lookup expression graph. -/
def Graph.eval
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : Graph arity) (_wellFormed : graph.WellFormed)
    (point : Fin arity → F) : F :=
  getValue (graph.evalNodes point) graph.root

/-- Evaluate a shared lookup expression graph on a fixed-length vector. -/
def Graph.evalVector
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : Graph arity) (wellFormed : graph.WellFormed)
    (point : Vector F arity) : F :=
  graph.eval wellFormed (fun index => point[index])

end Jolt.LookupGraph
