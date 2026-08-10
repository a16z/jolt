import Jolt.LookupAC
import Mathlib.Algebra.BigOperators.Group.List.Basic

/-!
# Proof-producing lookup correspondence certificates

Generated certificates map verifier and materializer nodes into one compact,
hash-consed associative-commutative DAG. Lean checks each source node locally
and proves that both roots have the same interpretation over every commutative
ring.
-/

namespace Jolt.LookupCorrespondence

/-- One node in the shared associative-commutative certificate DAG. -/
inductive CanonNode where
  | constant (value : Nat)
  | input (index : Nat)
  | add (terms : List Nat)
  | sub (left right : Nat)
  | mul (factors : List Nat)
  deriving DecidableEq

/-- Every reference in a canonical node points backward. -/
def CanonNode.WellFormedAt (arity position : Nat) : CanonNode → Prop
  | .constant _ => True
  | .input index => index < arity
  | .add terms | .mul terms => ∀ index ∈ terms, index < position
  | .sub left right => left < position ∧ right < position

instance CanonNode.instDecidableWellFormedAt
    (arity position : Nat) (node : CanonNode) : Decidable (node.WellFormedAt arity position) := by
  cases node <;> simp only [CanonNode.WellFormedAt] <;> infer_instance

/-- A consecutive range of the canonical oracle is topologically well formed. -/
def CanonRangeValid (arity : Nat) (oracle : Nat → CanonNode) : Nat → Nat → Prop
  | _, 0 => True
  | position, count + 1 =>
      (oracle position).WellFormedAt arity position ∧
        CanonRangeValid arity oracle (position + 1) count

instance instDecidableCanonRangeValid
    (arity : Nat) (oracle : Nat → CanonNode) (position count : Nat) :
    Decidable (CanonRangeValid arity oracle position count) := by
  induction count generalizing position with
  | zero => simp only [CanonRangeValid]; infer_instance
  | succ count ih => simp only [CanonRangeValid]; infer_instance

theorem CanonRangeValid.at {arity : Nat} {oracle : Nat → CanonNode}
    {start count index : Nat} (hvalid : CanonRangeValid arity oracle start count)
    (hlower : start ≤ index) (hupper : index < start + count) :
    (oracle index).WellFormedAt arity index := by
  induction count generalizing start with
  | zero => omega
  | succ count ih =>
      simp only [CanonRangeValid] at hvalid
      by_cases hindex : index = start
      · simpa [hindex] using hvalid.1
      · exact ih hvalid.2 (by omega) (by omega)

theorem CanonRangeValid.append {arity : Nat} {oracle : Nat → CanonNode}
    {start leftCount rightCount : Nat}
    (left : CanonRangeValid arity oracle start leftCount)
    (right : CanonRangeValid arity oracle (start + leftCount) rightCount) :
    CanonRangeValid arity oracle start (leftCount + rightCount) := by
  induction leftCount generalizing start with
  | zero => simpa using right
  | succ count ih =>
      rw [Nat.succ_add]
      simp only [CanonRangeValid] at left ⊢
      constructor
      · exact left.1
      · have right' :
            CanonRangeValid arity oracle (start + 1 + count) rightCount := by
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using right
        exact ih left.2 right'

/-- Evaluate one canonical node from preceding canonical values. -/
def CanonNode.eval
    {arity : Nat} {F : Type*} [CommRing F]
    (node : CanonNode) (values : Array F) (point : Fin arity → F) : F :=
  match node with
  | .constant value => value
  | .input index => if h : index < arity then point ⟨index, h⟩ else 0
  | .add terms => (terms.map fun index => Jolt.LookupGraph.getValue values index).sum
  | .sub left right =>
      Jolt.LookupGraph.getValue values left - Jolt.LookupGraph.getValue values right
  | .mul factors => (factors.map fun index => Jolt.LookupGraph.getValue values index).prod

theorem CanonNode.WellFormedAt.mono {arity left right : Nat} {node : CanonNode}
    (hvalid : node.WellFormedAt arity left) (hle : left ≤ right) :
    node.WellFormedAt arity right := by
  cases node with
  | constant => trivial
  | input => exact hvalid
  | add terms | mul terms =>
      intro index hindex
      exact (hvalid index hindex).trans_le hle
  | sub => exact ⟨hvalid.1.trans_le hle, hvalid.2.trans_le hle⟩

/-- Evaluate the first `count` canonical nodes. -/
def canonValues
    {arity : Nat} {F : Type*} [CommRing F]
    (oracle : Nat → CanonNode) (point : Fin arity → F) : Nat → Array F
  | 0 => #[]
  | count + 1 =>
      let values := canonValues oracle point count
      values.push ((oracle count).eval values point)

@[simp]
theorem canonValues_size
    {arity : Nat} {F : Type*} [CommRing F]
    (oracle : Nat → CanonNode) (point : Fin arity → F) (count : Nat) :
    (canonValues oracle point count).size = count := by
  induction count with
  | zero => rfl
  | succ count ih => simp [canonValues, ih]

private theorem getValue_push_of_lt
    {F : Type*} [Zero F] (values : Array F) (value : F) {index : Nat}
    (hindex : index < values.size) :
    Jolt.LookupGraph.getValue (values.push value) index =
      Jolt.LookupGraph.getValue values index := by
  unfold Jolt.LookupGraph.getValue
  rw [Array.getElem?_push_lt hindex]
  rw [Array.getElem?_eq_getElem hindex]

private theorem CanonNode.eval_push
    {arity : Nat} {F : Type*} [CommRing F]
    (node : CanonNode) (values : Array F) (value : F) (point : Fin arity → F)
    (hvalid : node.WellFormedAt arity values.size) :
    node.eval (values.push value) point = node.eval values point := by
  cases node with
  | constant => rfl
  | input => rfl
  | add terms | mul terms =>
      simp only [CanonNode.eval]
      congr 1
      apply List.map_congr_left
      intro index hindex
      exact getValue_push_of_lt values value (hvalid index hindex)
  | sub left right =>
      simp only [CanonNode.WellFormedAt] at hvalid
      simp only [CanonNode.eval]
      rw [getValue_push_of_lt values value hvalid.1,
        getValue_push_of_lt values value hvalid.2]

/-- Every well-formed canonical node evaluates according to its oracle declaration. -/
private theorem CanonRangeValid.prefix {arity : Nat} {oracle : Nat → CanonNode}
    {start count : Nat} (hvalid : CanonRangeValid arity oracle start (count + 1)) :
    CanonRangeValid arity oracle start count := by
  induction count generalizing start with
  | zero => trivial
  | succ count ih =>
      simp only [CanonRangeValid] at hvalid ⊢
      exact ⟨hvalid.1, ih hvalid.2⟩

theorem canonValues_sound
    {arity : Nat} {F : Type*} [CommRing F]
    (oracle : Nat → CanonNode) (point : Fin arity → F) {count index : Nat}
    (hvalid : CanonRangeValid arity oracle 0 count) (hindex : index < count) :
    Jolt.LookupGraph.getValue (canonValues oracle point count) index =
      (oracle index).eval (canonValues oracle point count) point := by
  induction count generalizing index with
  | zero => omega
  | succ count ih =>
      have hprefix : CanonRangeValid arity oracle 0 count := hvalid.prefix
      have hnode := hvalid.at (index := index) (by omega) (by omega)
      simp only [canonValues]
      by_cases hlast : index = count
      · subst index
        rw [show Jolt.LookupGraph.getValue
            ((canonValues oracle point count).push
              ((oracle count).eval (canonValues oracle point count) point)) count =
            (oracle count).eval (canonValues oracle point count) point by
          unfold Jolt.LookupGraph.getValue
          rw [Array.getElem?_push]
          simp [canonValues_size]]
        exact (CanonNode.eval_push _ _ _ _ (by simpa [canonValues_size] using hnode)).symm
      · rw [getValue_push_of_lt _ _ (by simpa [canonValues_size] using (show index < count by omega))]
        rw [ih hprefix (by omega)]
        exact (CanonNode.eval_push _ _ _ _ (by
          simpa [canonValues_size] using hnode.mono (show index ≤ count by omega))).symm

/-- The result of one canonicalization step is either an existing node or a claimed node shape. -/
inductive CanonStep where
  | existing (index : Nat)
  | node (value : CanonNode)
  deriving DecidableEq

/-- Executable check that a claimed canonical ID realizes one canonicalization step. -/
def CanonStep.matches
    (oracle : Nat → CanonNode) (count claimed : Nat) : CanonStep → Bool
  | .existing index => claimed < count && claimed == index
  | .node value => claimed < count && oracle claimed == value

/-- A claimed canonical ID realizes one canonicalization step. -/
def CanonStep.Matches
    (oracle : Nat → CanonNode) (count claimed : Nat) (step : CanonStep) : Prop :=
  step.matches oracle count claimed = true

instance CanonStep.instDecidableMatches
    (oracle : Nat → CanonNode) (count claimed : Nat) (step : CanonStep) :
    Decidable (step.Matches oracle count claimed) :=
  inferInstanceAs (Decidable (step.matches oracle count claimed = true))

theorem CanonStep.Matches.claimed_lt
    {oracle : Nat → CanonNode} {count claimed : Nat} {step : CanonStep}
    (hmatch : step.Matches oracle count claimed) : claimed < count := by
  cases step <;>
    simp only [CanonStep.Matches, CanonStep.matches, Bool.and_eq_true,
      decide_eq_true_eq, beq_iff_eq] at hmatch <;>
    exact hmatch.1

/-- Flatten a canonical sum and remove zero. -/
def addTerms (oracle : Nat → CanonNode) (index : Nat) : List Nat :=
  match oracle index with
  | .constant 0 => []
  | .add terms => terms
  | _ => [index]

/-- Flatten a canonical product and remove one. -/
def mulFactors (oracle : Nat → CanonNode) (index : Nat) : List Nat :=
  match oracle index with
  | .constant 1 => []
  | .mul factors => factors
  | _ => [index]

/-- Insert one canonical ID into a sorted list. -/
def insertId (index : Nat) : List Nat → List Nat
  | [] => [index]
  | first :: rest =>
      if index ≤ first then index :: first :: rest
      else first :: insertId index rest

/-- Deterministically order canonical IDs with a transparent certificate checker. -/
def sortIds : List Nat → List Nat
  | [] => []
  | index :: indices => insertId index (sortIds indices)

/-- Canonicalize addition modulo associativity, commutativity, and zero. -/
def smartAdd (oracle : Nat → CanonNode) (left right : Nat) : CanonStep :=
  match sortIds (addTerms oracle left ++ addTerms oracle right) with
  | [] => .node (.constant 0)
  | [index] => .existing index
  | terms => .node (.add terms)

/-- Canonicalize multiplication modulo associativity, commutativity, zero, and one. -/
def smartMul (oracle : Nat → CanonNode) (left right : Nat) : CanonStep :=
  if oracle left = .constant 0 ∨ oracle right = .constant 0 then
    .node (.constant 0)
  else
    match sortIds (mulFactors oracle left ++ mulFactors oracle right) with
    | [] => .node (.constant 1)
    | [index] => .existing index
    | factors => .node (.mul factors)

private theorem sum_addTerms
    {arity : Nat} {F : Type*} [CommRing F]
    (oracle : Nat → CanonNode) (point : Fin arity → F) {count index : Nat}
    (hvalid : CanonRangeValid arity oracle 0 count) (hindex : index < count) :
    ((addTerms oracle index).map fun id =>
        Jolt.LookupGraph.getValue (canonValues oracle point count) id).sum =
      Jolt.LookupGraph.getValue (canonValues oracle point count) index := by
  have hsound := canonValues_sound oracle point hvalid hindex
  cases hnode : oracle index with
  | constant value =>
      rw [addTerms, hnode]
      by_cases hzero : value = 0
      · subst value
        simpa [CanonNode.eval, hnode] using hsound.symm
      · simp [hzero]
  | input | sub | mul => simp [addTerms, hnode]
  | add terms => simpa [addTerms, hnode, CanonNode.eval] using hsound.symm

private theorem prod_mulFactors
    {arity : Nat} {F : Type*} [CommRing F]
    (oracle : Nat → CanonNode) (point : Fin arity → F) {count index : Nat}
    (hvalid : CanonRangeValid arity oracle 0 count) (hindex : index < count) :
    ((mulFactors oracle index).map fun id =>
        Jolt.LookupGraph.getValue (canonValues oracle point count) id).prod =
      Jolt.LookupGraph.getValue (canonValues oracle point count) index := by
  have hsound := canonValues_sound oracle point hvalid hindex
  cases hnode : oracle index with
  | constant value =>
      rw [mulFactors, hnode]
      by_cases hone : value = 1
      · subst value
        simpa [CanonNode.eval, hnode] using hsound.symm
      · simp [hone]
  | input | add | sub => simp [mulFactors, hnode]
  | mul factors => simpa [mulFactors, hnode, CanonNode.eval] using hsound.symm

private theorem sum_insertId
    {F : Type*} [AddCommMonoid F] (index : Nat) (indices : List Nat) (value : Nat → F) :
    ((insertId index indices).map value).sum = value index + (indices.map value).sum := by
  induction indices with
  | nil => simp [insertId]
  | cons first rest ih =>
      simp only [insertId]
      split
      · simp
      · simp only [List.map_cons, List.sum_cons, ih]
        ac_rfl

private theorem sum_sortIds
    {F : Type*} [AddCommMonoid F] (indices : List Nat) (value : Nat → F) :
    ((sortIds indices).map value).sum = (indices.map value).sum := by
  induction indices with
  | nil => rfl
  | cons index indices ih =>
      simp only [sortIds, List.map_cons, List.sum_cons, sum_insertId, ih]

private theorem prod_insertId
    {F : Type*} [CommMonoid F] (index : Nat) (indices : List Nat) (value : Nat → F) :
    ((insertId index indices).map value).prod = value index * (indices.map value).prod := by
  induction indices with
  | nil => simp [insertId]
  | cons first rest ih =>
      simp only [insertId]
      split
      · simp
      · simp only [List.map_cons, List.prod_cons, ih]
        ac_rfl

private theorem prod_sortIds
    {F : Type*} [CommMonoid F] (indices : List Nat) (value : Nat → F) :
    ((sortIds indices).map value).prod = (indices.map value).prod := by
  induction indices with
  | nil => rfl
  | cons index indices ih =>
      simp only [sortIds, List.map_cons, List.prod_cons, prod_insertId, ih]

theorem smartAdd_sound
    {arity : Nat} {F : Type*} [CommRing F]
    (oracle : Nat → CanonNode) (point : Fin arity → F) {count left right claimed : Nat}
    (hvalid : CanonRangeValid arity oracle 0 count)
    (hleft : left < count) (hright : right < count)
    (hmatch : (smartAdd oracle left right).Matches oracle count claimed) :
    Jolt.LookupGraph.getValue (canonValues oracle point count) claimed =
      Jolt.LookupGraph.getValue (canonValues oracle point count) left +
        Jolt.LookupGraph.getValue (canonValues oracle point count) right := by
  let value := fun index =>
    Jolt.LookupGraph.getValue (canonValues oracle point count) index
  generalize hterms : sortIds (addTerms oracle left ++ addTerms oracle right) = terms
  have hadd : (terms.map value).sum = value left + value right := by
    rw [← hterms, sum_sortIds, List.map_append, List.sum_append,
      sum_addTerms oracle point hvalid hleft,
      sum_addTerms oracle point hvalid hright]
  rw [smartAdd, hterms] at hmatch
  cases terms with
  | nil =>
      simp only [CanonStep.Matches, CanonStep.matches, Bool.and_eq_true,
        decide_eq_true_eq, beq_iff_eq] at hmatch
      have hsound := canonValues_sound oracle point hvalid hmatch.1
      rw [hmatch.2] at hsound
      simp only [CanonNode.eval, Nat.cast_zero] at hsound
      exact hsound.trans (by simpa using hadd)
  | cons first rest =>
      cases rest with
      | nil =>
          simp only [CanonStep.Matches, CanonStep.matches, Bool.and_eq_true,
            decide_eq_true_eq, beq_iff_eq] at hmatch
          rw [hmatch.2]
          simpa [value] using hadd
      | cons second rest =>
          simp only [CanonStep.Matches, CanonStep.matches, Bool.and_eq_true,
            decide_eq_true_eq, beq_iff_eq] at hmatch
          have hsound := canonValues_sound oracle point hvalid hmatch.1
          rw [hmatch.2] at hsound
          simp only [CanonNode.eval] at hsound
          exact hsound.trans hadd

theorem smartMul_sound
    {arity : Nat} {F : Type*} [CommRing F]
    (oracle : Nat → CanonNode) (point : Fin arity → F) {count left right claimed : Nat}
    (hvalid : CanonRangeValid arity oracle 0 count)
    (hleft : left < count) (hright : right < count)
    (hmatch : (smartMul oracle left right).Matches oracle count claimed) :
    Jolt.LookupGraph.getValue (canonValues oracle point count) claimed =
      Jolt.LookupGraph.getValue (canonValues oracle point count) left *
        Jolt.LookupGraph.getValue (canonValues oracle point count) right := by
  let value := fun index =>
    Jolt.LookupGraph.getValue (canonValues oracle point count) index
  by_cases hzero : oracle left = .constant 0 ∨ oracle right = .constant 0
  · rw [smartMul, if_pos hzero] at hmatch
    simp only [CanonStep.Matches, CanonStep.matches, Bool.and_eq_true,
      decide_eq_true_eq, beq_iff_eq] at hmatch
    have hclaimed := canonValues_sound oracle point hvalid hmatch.1
    rw [hmatch.2] at hclaimed
    simp only [CanonNode.eval, Nat.cast_zero] at hclaimed
    rcases hzero with hleftZero | hrightZero
    · have hleftValue := canonValues_sound oracle point hvalid hleft
      rw [hleftZero] at hleftValue
      simp only [CanonNode.eval, Nat.cast_zero] at hleftValue
      rw [hclaimed, hleftValue, zero_mul]
    · have hrightValue := canonValues_sound oracle point hvalid hright
      rw [hrightZero] at hrightValue
      simp only [CanonNode.eval, Nat.cast_zero] at hrightValue
      rw [hclaimed, hrightValue, mul_zero]
  · generalize hterms : sortIds (mulFactors oracle left ++ mulFactors oracle right) = factors
    have hmul : (factors.map value).prod = value left * value right := by
      rw [← hterms, prod_sortIds, List.map_append, List.prod_append,
        prod_mulFactors oracle point hvalid hleft,
        prod_mulFactors oracle point hvalid hright]
    rw [smartMul, if_neg hzero, hterms] at hmatch
    cases factors with
    | nil =>
        simp only [CanonStep.Matches, CanonStep.matches, Bool.and_eq_true,
          decide_eq_true_eq, beq_iff_eq] at hmatch
        have hsound := canonValues_sound oracle point hvalid hmatch.1
        rw [hmatch.2] at hsound
        simp only [CanonNode.eval, Nat.cast_one] at hsound
        exact hsound.trans (by simpa using hmul)
    | cons first rest =>
        cases rest with
        | nil =>
            simp only [CanonStep.Matches, CanonStep.matches, Bool.and_eq_true,
              decide_eq_true_eq, beq_iff_eq] at hmatch
            rw [hmatch.2]
            simpa [value] using hmul
        | cons second rest =>
            simp only [CanonStep.Matches, CanonStep.matches, Bool.and_eq_true,
              decide_eq_true_eq, beq_iff_eq] at hmatch
            have hsound := canonValues_sound oracle point hvalid hmatch.1
            rw [hmatch.2] at hsound
            simp only [CanonNode.eval] at hsound
            exact hsound.trans hmul

private theorem CanonStep.node_sound
    {arity : Nat} {F : Type*} [CommRing F]
    (oracle : Nat → CanonNode) (point : Fin arity → F) {count claimed : Nat}
    (hvalid : CanonRangeValid arity oracle 0 count) (node : CanonNode)
    (hmatch : (CanonStep.node node).Matches oracle count claimed) :
    Jolt.LookupGraph.getValue (canonValues oracle point count) claimed =
      node.eval (canonValues oracle point count) point := by
  simp only [CanonStep.Matches, CanonStep.matches, Bool.and_eq_true,
    decide_eq_true_eq, beq_iff_eq] at hmatch
  have hsound := canonValues_sound oracle point hvalid hmatch.1
  rwa [hmatch.2] at hsound

/-- Canonicalization step claimed for one verifier-graph node. -/
def _root_.Jolt.LookupGraph.Node.canonStep
    (oracle : Nat → CanonNode) (ids : Nat → Nat) : Jolt.LookupGraph.Node → CanonStep
  | .constant value => .node (.constant value)
  | .input index => .node (.input index)
  | .add left right => smartAdd oracle (ids left) (ids right)
  | .sub left right => .node (.sub (ids left) (ids right))
  | .mul left right => smartMul oracle (ids left) (ids right)

/-- One verifier node is well formed and has the claimed canonical ID. -/
def _root_.Jolt.LookupGraph.Node.ValidCanonAt
    (arity canonCount : Nat) (oracle : Nat → CanonNode) (ids : Nat → Nat)
    (position : Nat) (node : Jolt.LookupGraph.Node) : Prop :=
  node.wellFormedAt arity position = true ∧
    (node.canonStep oracle ids).Matches oracle canonCount (ids position)

instance _root_.Jolt.LookupGraph.Node.instDecidableValidCanonAt
    (arity canonCount : Nat) (oracle : Nat → CanonNode) (ids : Nat → Nat)
    (position : Nat) (node : Jolt.LookupGraph.Node) :
    Decidable (node.ValidCanonAt arity canonCount oracle ids position) := by
  unfold Jolt.LookupGraph.Node.ValidCanonAt
  infer_instance

/-- Consecutive verifier nodes have the claimed canonical IDs. -/
def GraphNodesValid
    (arity canonCount : Nat) (oracle : Nat → CanonNode) (ids : Nat → Nat) :
    Nat → List Jolt.LookupGraph.Node → Prop
  | _, [] => True
  | position, node :: nodes =>
      node.ValidCanonAt arity canonCount oracle ids position ∧
        GraphNodesValid arity canonCount oracle ids (position + 1) nodes

instance instDecidableGraphNodesValid
    (arity canonCount : Nat) (oracle : Nat → CanonNode) (ids : Nat → Nat)
    (position : Nat) (nodes : List Jolt.LookupGraph.Node) :
    Decidable (GraphNodesValid arity canonCount oracle ids position nodes) := by
  induction nodes generalizing position with
  | nil => simp only [GraphNodesValid]; infer_instance
  | cons node nodes ih => simp only [GraphNodesValid]; infer_instance

/-- Chunked verifier nodes have the claimed canonical IDs. -/
def GraphChunksValid
    (arity canonCount : Nat) (oracle : Nat → CanonNode) (ids : Nat → Nat) :
    Nat → List (List Jolt.LookupGraph.Node) → Prop
  | _, [] => True
  | position, nodes :: chunks =>
      GraphNodesValid arity canonCount oracle ids position nodes ∧
        GraphChunksValid arity canonCount oracle ids (position + nodes.length) chunks

instance instDecidableGraphChunksValid
    (arity canonCount : Nat) (oracle : Nat → CanonNode) (ids : Nat → Nat)
    (position : Nat) (chunks : List (List Jolt.LookupGraph.Node)) :
    Decidable (GraphChunksValid arity canonCount oracle ids position chunks) := by
  induction chunks generalizing position with
  | nil => simp only [GraphChunksValid]; infer_instance
  | cons nodes chunks ih => simp only [GraphChunksValid]; infer_instance

private def ValuesAgree
    {F : Type*} [Zero F] (values : Array F) (ids : Nat → Nat)
    (canon : Array F) (canonCount : Nat) : Prop :=
  ∀ (index : Nat) (hindex : index < values.size),
    ids index < canonCount ∧ values[index] = Jolt.LookupGraph.getValue canon (ids index)

private theorem getValue_eq_of_agrees
    {F : Type*} [Zero F] {values canon : Array F} {ids : Nat → Nat} {canonCount index : Nat}
    (hagrees : ValuesAgree values ids canon canonCount) (hindex : index < values.size) :
    Jolt.LookupGraph.getValue values index = Jolt.LookupGraph.getValue canon (ids index) := by
  unfold Jolt.LookupGraph.getValue
  rw [Array.getElem?_eq_getElem hindex]
  simpa using (hagrees index hindex).2

private theorem valuesAgree_push
    {F : Type*} [Zero F] {values canon : Array F} {ids : Nat → Nat} {canonCount : Nat}
    (hagrees : ValuesAgree values ids canon canonCount) (value : F)
    (hid : ids values.size < canonCount)
    (hvalue : value = Jolt.LookupGraph.getValue canon (ids values.size)) :
    ValuesAgree (values.push value) ids canon canonCount := by
  intro index hindex
  rw [Array.getElem_push]
  split
  · exact hagrees index (by assumption)
  · have : index = values.size := by
      rw [Array.size_push] at hindex
      omega
    subst index
    exact ⟨hid, hvalue⟩

private theorem graphNode_sound
    {arity : Nat} {F : Type*} [CommRing F]
    (canonOracle : Nat → CanonNode) (canonCount : Nat) (ids : Nat → Nat)
    (point : Fin arity → F) (values : Array F) (node : Jolt.LookupGraph.Node)
    (hcanon : CanonRangeValid arity canonOracle 0 canonCount)
    (hvalid : node.ValidCanonAt arity canonCount canonOracle ids values.size)
    (hagrees : ValuesAgree values ids (canonValues canonOracle point canonCount) canonCount) :
    node.eval values point =
      Jolt.LookupGraph.getValue (canonValues canonOracle point canonCount) (ids values.size) := by
  rcases hvalid with ⟨hwellFormed, hmatch⟩
  cases node with
  | constant value =>
      exact (CanonStep.node_sound canonOracle point hcanon (.constant value) hmatch).symm
  | input index =>
      simp only [Jolt.LookupGraph.Node.wellFormedAt] at hwellFormed
      have hsound := CanonStep.node_sound canonOracle point hcanon (.input index) hmatch
      rw [hsound]
      simp [Jolt.LookupGraph.Node.eval, CanonNode.eval]
  | add left right =>
      simp only [Jolt.LookupGraph.Node.wellFormedAt, Bool.and_eq_true] at hwellFormed
      have hleft : left < values.size := by simpa using hwellFormed.1
      have hright : right < values.size := by simpa using hwellFormed.2
      rw [Jolt.LookupGraph.Node.eval,
        getValue_eq_of_agrees hagrees hleft,
        getValue_eq_of_agrees hagrees hright]
      exact (smartAdd_sound canonOracle point hcanon
        (hagrees left hleft).1 (hagrees right hright).1 hmatch).symm
  | sub left right =>
      simp only [Jolt.LookupGraph.Node.wellFormedAt, Bool.and_eq_true] at hwellFormed
      have hleft : left < values.size := by simpa using hwellFormed.1
      have hright : right < values.size := by simpa using hwellFormed.2
      have hsound := CanonStep.node_sound canonOracle point hcanon
        (.sub (ids left) (ids right)) hmatch
      simp only [Jolt.LookupGraph.Node.eval, CanonNode.eval] at hsound ⊢
      rw [getValue_eq_of_agrees hagrees hleft,
        getValue_eq_of_agrees hagrees hright]
      exact hsound.symm
  | mul left right =>
      simp only [Jolt.LookupGraph.Node.wellFormedAt, Bool.and_eq_true] at hwellFormed
      have hleft : left < values.size := by simpa using hwellFormed.1
      have hright : right < values.size := by simpa using hwellFormed.2
      rw [Jolt.LookupGraph.Node.eval,
        getValue_eq_of_agrees hagrees hleft,
        getValue_eq_of_agrees hagrees hright]
      exact (smartMul_sound canonOracle point hcanon
        (hagrees left hleft).1 (hagrees right hright).1 hmatch).symm

private theorem foldGraphNodes_valid
    {arity : Nat} {F : Type*} [CommRing F]
    (canonOracle : Nat → CanonNode) (canonCount : Nat) (ids : Nat → Nat)
    (point : Fin arity → F) (nodes : List Jolt.LookupGraph.Node) (position : Nat)
    (values : Array F) (hcanon : CanonRangeValid arity canonOracle 0 canonCount)
    (hvalid : GraphNodesValid arity canonCount canonOracle ids position nodes)
    (hsize : values.size = position)
    (hagrees : ValuesAgree values ids (canonValues canonOracle point canonCount) canonCount) :
    let result := nodes.foldl (fun values node => values.push (node.eval values point)) values
    result.size = values.size + nodes.length ∧
      ValuesAgree result ids (canonValues canonOracle point canonCount) canonCount := by
  induction nodes generalizing position values with
  | nil =>
      simp only [List.foldl, List.length_nil, Nat.add_zero]
      exact ⟨trivial, hagrees⟩
  | cons node nodes ih =>
      simp only [GraphNodesValid] at hvalid
      rcases hvalid with ⟨hnode, hrest⟩
      have hsound := graphNode_sound canonOracle canonCount ids point values node hcanon
        (by simpa [hsize] using hnode) hagrees
      have hagrees' := valuesAgree_push hagrees (node.eval values point)
        (by simpa [hsize] using hnode.2.claimed_lt) (by simpa [hsize] using hsound)
      have hsize' : (values.push (node.eval values point)).size = position + 1 := by
        simp [hsize]
      rcases ih (position + 1) (values.push (node.eval values point))
          hrest hsize' hagrees' with ⟨hresultSize, hresultAgrees⟩
      simp only [List.foldl]
      constructor
      · rw [hresultSize, Array.size_push]
        simp only [List.length_cons]
        omega
      · exact hresultAgrees

private theorem foldGraphChunks_valid
    {arity : Nat} {F : Type*} [CommRing F]
    (canonOracle : Nat → CanonNode) (canonCount : Nat) (ids : Nat → Nat)
    (point : Fin arity → F) (chunks : List (List Jolt.LookupGraph.Node)) (position : Nat)
    (values : Array F) (hcanon : CanonRangeValid arity canonOracle 0 canonCount)
    (hvalid : GraphChunksValid arity canonCount canonOracle ids position chunks)
    (hsize : values.size = position)
    (hagrees : ValuesAgree values ids (canonValues canonOracle point canonCount) canonCount) :
    let result := chunks.foldl
      (fun values nodes => nodes.foldl
        (fun values node => values.push (node.eval values point)) values) values
    result.size = values.size + chunks.flatten.length ∧
      ValuesAgree result ids (canonValues canonOracle point canonCount) canonCount := by
  induction chunks generalizing position values with
  | nil =>
      simp only [List.foldl, List.flatten_nil, List.length_nil, Nat.add_zero]
      exact ⟨trivial, hagrees⟩
  | cons nodes chunks ih =>
      simp only [GraphChunksValid] at hvalid
      rcases hvalid with ⟨hnodes, hchunks⟩
      have hnodesResult := foldGraphNodes_valid canonOracle canonCount ids point
        nodes position values hcanon hnodes hsize hagrees
      let next := nodes.foldl
        (fun values node => values.push (node.eval values point)) values
      have hnextSize : next.size = position + nodes.length := by
        simpa [next, hsize] using hnodesResult.1
      have hnextAgrees :
          ValuesAgree next ids (canonValues canonOracle point canonCount) canonCount := by
        simpa [next] using hnodesResult.2
      rcases ih (position + nodes.length) next hchunks hnextSize hnextAgrees with
        ⟨hresultSize, hresultAgrees⟩
      simp only [List.foldl]
      constructor
      · rw [hresultSize, hnextSize, hsize]
        simp only [List.flatten_cons, List.length_append]
        omega
      · exact hresultAgrees

/-- A locally checked verifier graph evaluates to its claimed canonical root. -/
theorem graph_eval_eq_canon
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : Jolt.LookupGraph.Graph arity) (wellFormed : graph.WellFormed)
    (canonOracle : Nat → CanonNode) (canonCount : Nat) (ids : Nat → Nat)
    (hcanon : CanonRangeValid arity canonOracle 0 canonCount)
    (hvalid : GraphChunksValid arity canonCount canonOracle ids 0 graph.nodeChunks)
    (hroot : graph.root < graph.nodes.length) (point : Fin arity → F) :
    graph.eval wellFormed point =
      Jolt.LookupGraph.getValue (canonValues canonOracle point canonCount) (ids graph.root) := by
  have hresult := foldGraphChunks_valid canonOracle canonCount ids point
    graph.nodeChunks 0 #[] hcanon hvalid rfl (by
      intro index hindex
      simp at hindex)
  have hagrees :
      ValuesAgree (graph.evalNodes point) ids
        (canonValues canonOracle point canonCount) canonCount := by
    simpa [Jolt.LookupGraph.Graph.evalNodes] using hresult.2
  unfold Jolt.LookupGraph.Graph.eval
  have hevalSize : (graph.evalNodes point).size = graph.nodes.length := by
    simpa [Jolt.LookupGraph.Graph.evalNodes, Jolt.LookupGraph.Graph.nodes] using hresult.1
  exact getValue_eq_of_agrees hagrees (by rw [hevalSize]; exact hroot)

/-- Canonicalization step claimed for one Boolean materializer node. -/
def _root_.Jolt.LookupExpression.MaterializerBoolNode.canonStep
    (oracle : Nat → CanonNode) (oneId : Nat) (ids : Nat → Nat) :
    Jolt.LookupExpression.MaterializerBoolNode → CanonStep
  | .input index => .node (.input index)
  | .conj left right => smartMul oracle (ids left) (ids right)
  | .neg value => .node (.sub oneId (ids value))

/-- Canonicalization step claimed for one natural-number materializer node. -/
def _root_.Jolt.LookupExpression.MaterializerNatNode.canonStep
    (oracle : Nat → CanonNode) (boolIds natIds : Nat → Nat) :
    Jolt.LookupExpression.MaterializerNatNode → CanonStep
  | .constant value => .node (.constant value)
  | .ofBit value => .existing (boolIds value)
  | .add left right => smartAdd oracle (natIds left) (natIds right)
  | .mul left right => smartMul oracle (natIds left) (natIds right)

/-- One Boolean materializer node has the claimed canonical ID. -/
def _root_.Jolt.LookupExpression.MaterializerBoolNode.ValidCanonAt
    (arity canonCount : Nat) (oracle : Nat → CanonNode) (oneId : Nat) (ids : Nat → Nat)
    (position : Nat) (node : Jolt.LookupExpression.MaterializerBoolNode) : Prop :=
  node.wellFormedAt arity position = true ∧
    (node.canonStep oracle oneId ids).Matches oracle canonCount (ids position)

instance _root_.Jolt.LookupExpression.MaterializerBoolNode.instDecidableValidCanonAt
    (arity canonCount : Nat) (oracle : Nat → CanonNode) (oneId : Nat) (ids : Nat → Nat)
    (position : Nat) (node : Jolt.LookupExpression.MaterializerBoolNode) :
    Decidable (node.ValidCanonAt arity canonCount oracle oneId ids position) := by
  unfold Jolt.LookupExpression.MaterializerBoolNode.ValidCanonAt
  infer_instance

/-- One natural-number materializer node has the claimed canonical ID. -/
def _root_.Jolt.LookupExpression.MaterializerNatNode.ValidCanonAt
    (boolCount canonCount : Nat) (oracle : Nat → CanonNode)
    (boolIds natIds : Nat → Nat) (position : Nat)
    (node : Jolt.LookupExpression.MaterializerNatNode) : Prop :=
  node.wellFormedAt boolCount position = true ∧
    (node.canonStep oracle boolIds natIds).Matches oracle canonCount (natIds position)

instance _root_.Jolt.LookupExpression.MaterializerNatNode.instDecidableValidCanonAt
    (boolCount canonCount : Nat) (oracle : Nat → CanonNode)
    (boolIds natIds : Nat → Nat) (position : Nat)
    (node : Jolt.LookupExpression.MaterializerNatNode) :
    Decidable (node.ValidCanonAt boolCount canonCount oracle boolIds natIds position) := by
  unfold Jolt.LookupExpression.MaterializerNatNode.ValidCanonAt
  infer_instance

def MaterializerBoolNodesValid
    (arity canonCount : Nat) (oracle : Nat → CanonNode) (oneId : Nat) (ids : Nat → Nat) :
    Nat → List Jolt.LookupExpression.MaterializerBoolNode → Prop
  | _, [] => True
  | position, node :: nodes =>
      node.ValidCanonAt arity canonCount oracle oneId ids position ∧
        MaterializerBoolNodesValid arity canonCount oracle oneId ids (position + 1) nodes

instance instDecidableMaterializerBoolNodesValid
    (arity canonCount : Nat) (oracle : Nat → CanonNode) (oneId : Nat) (ids : Nat → Nat)
    (position : Nat) (nodes : List Jolt.LookupExpression.MaterializerBoolNode) :
    Decidable (MaterializerBoolNodesValid arity canonCount oracle oneId ids position nodes) := by
  induction nodes generalizing position with
  | nil => simp only [MaterializerBoolNodesValid]; infer_instance
  | cons node nodes ih => simp only [MaterializerBoolNodesValid]; infer_instance

def MaterializerBoolChunksValid
    (arity canonCount : Nat) (oracle : Nat → CanonNode) (oneId : Nat) (ids : Nat → Nat) :
    Nat → List (List Jolt.LookupExpression.MaterializerBoolNode) → Prop
  | _, [] => True
  | position, nodes :: chunks =>
      MaterializerBoolNodesValid arity canonCount oracle oneId ids position nodes ∧
        MaterializerBoolChunksValid arity canonCount oracle oneId ids
          (position + nodes.length) chunks

instance instDecidableMaterializerBoolChunksValid
    (arity canonCount : Nat) (oracle : Nat → CanonNode) (oneId : Nat) (ids : Nat → Nat)
    (position : Nat) (chunks : List (List Jolt.LookupExpression.MaterializerBoolNode)) :
    Decidable (MaterializerBoolChunksValid arity canonCount oracle oneId ids position chunks) := by
  induction chunks generalizing position with
  | nil => simp only [MaterializerBoolChunksValid]; infer_instance
  | cons nodes chunks ih => simp only [MaterializerBoolChunksValid]; infer_instance

def MaterializerNatNodesValid
    (boolCount canonCount : Nat) (oracle : Nat → CanonNode)
    (boolIds natIds : Nat → Nat) :
    Nat → List Jolt.LookupExpression.MaterializerNatNode → Prop
  | _, [] => True
  | position, node :: nodes =>
      node.ValidCanonAt boolCount canonCount oracle boolIds natIds position ∧
        MaterializerNatNodesValid boolCount canonCount oracle boolIds natIds
          (position + 1) nodes

instance instDecidableMaterializerNatNodesValid
    (boolCount canonCount : Nat) (oracle : Nat → CanonNode)
    (boolIds natIds : Nat → Nat) (position : Nat)
    (nodes : List Jolt.LookupExpression.MaterializerNatNode) :
    Decidable (MaterializerNatNodesValid boolCount canonCount oracle boolIds natIds position nodes) := by
  induction nodes generalizing position with
  | nil => simp only [MaterializerNatNodesValid]; infer_instance
  | cons node nodes ih => simp only [MaterializerNatNodesValid]; infer_instance

def MaterializerNatChunksValid
    (boolCount canonCount : Nat) (oracle : Nat → CanonNode)
    (boolIds natIds : Nat → Nat) :
    Nat → List (List Jolt.LookupExpression.MaterializerNatNode) → Prop
  | _, [] => True
  | position, nodes :: chunks =>
      MaterializerNatNodesValid boolCount canonCount oracle boolIds natIds position nodes ∧
        MaterializerNatChunksValid boolCount canonCount oracle boolIds natIds
          (position + nodes.length) chunks

instance instDecidableMaterializerNatChunksValid
    (boolCount canonCount : Nat) (oracle : Nat → CanonNode)
    (boolIds natIds : Nat → Nat) (position : Nat)
    (chunks : List (List Jolt.LookupExpression.MaterializerNatNode)) :
    Decidable (MaterializerNatChunksValid boolCount canonCount oracle boolIds natIds position chunks) := by
  induction chunks generalizing position with
  | nil => simp only [MaterializerNatChunksValid]; infer_instance
  | cons nodes chunks ih => simp only [MaterializerNatChunksValid]; infer_instance

private theorem materializerBoolNode_sound
    {arity : Nat} {F : Type*} [CommRing F]
    (canonOracle : Nat → CanonNode) (canonCount oneId : Nat) (ids : Nat → Nat)
    (point : Fin arity → F) (values : Array F)
    (node : Jolt.LookupExpression.MaterializerBoolNode)
    (hcanon : CanonRangeValid arity canonOracle 0 canonCount)
    (honeId : oneId < canonCount) (hone : canonOracle oneId = .constant 1)
    (hvalid : node.ValidCanonAt arity canonCount canonOracle oneId ids values.size)
    (hagrees : ValuesAgree values ids (canonValues canonOracle point canonCount) canonCount) :
    node.evalArith values point =
      Jolt.LookupGraph.getValue (canonValues canonOracle point canonCount) (ids values.size) := by
  rcases hvalid with ⟨hwellFormed, hmatch⟩
  cases node with
  | input index =>
      have hsound := CanonStep.node_sound canonOracle point hcanon (.input index) hmatch
      rw [hsound]
      simp [Jolt.LookupExpression.MaterializerBoolNode.evalArith, CanonNode.eval]
  | conj left right =>
      simp only [Jolt.LookupExpression.MaterializerBoolNode.wellFormedAt,
        Bool.and_eq_true] at hwellFormed
      have hleft : left < values.size := by simpa using hwellFormed.1
      have hright : right < values.size := by simpa using hwellFormed.2
      rw [Jolt.LookupExpression.MaterializerBoolNode.evalArith,
        getValue_eq_of_agrees hagrees hleft,
        getValue_eq_of_agrees hagrees hright]
      exact (smartMul_sound canonOracle point hcanon
        (hagrees left hleft).1 (hagrees right hright).1 hmatch).symm
  | neg value =>
      simp only [Jolt.LookupExpression.MaterializerBoolNode.wellFormedAt] at hwellFormed
      have hvalue : value < values.size := by simpa using hwellFormed
      have hsound := CanonStep.node_sound canonOracle point hcanon
        (.sub oneId (ids value)) hmatch
      have honeValue := canonValues_sound canonOracle point hcanon honeId
      rw [hone] at honeValue
      simp only [CanonNode.eval, Nat.cast_one] at honeValue
      simp only [Jolt.LookupExpression.MaterializerBoolNode.evalArith, CanonNode.eval] at hsound ⊢
      rw [getValue_eq_of_agrees hagrees hvalue, ← honeValue]
      exact hsound.symm

private theorem foldMaterializerBoolNodes_valid
    {arity : Nat} {F : Type*} [CommRing F]
    (canonOracle : Nat → CanonNode) (canonCount oneId : Nat) (ids : Nat → Nat)
    (point : Fin arity → F) (nodes : List Jolt.LookupExpression.MaterializerBoolNode)
    (position : Nat) (values : Array F)
    (hcanon : CanonRangeValid arity canonOracle 0 canonCount)
    (honeId : oneId < canonCount) (hone : canonOracle oneId = .constant 1)
    (hvalid : MaterializerBoolNodesValid arity canonCount canonOracle oneId ids position nodes)
    (hsize : values.size = position)
    (hagrees : ValuesAgree values ids (canonValues canonOracle point canonCount) canonCount) :
    let result := nodes.foldl
      (fun values node => values.push (node.evalArith values point)) values
    result.size = values.size + nodes.length ∧
      ValuesAgree result ids (canonValues canonOracle point canonCount) canonCount := by
  induction nodes generalizing position values with
  | nil =>
      simp only [List.foldl, List.length_nil, Nat.add_zero]
      exact ⟨trivial, hagrees⟩
  | cons node nodes ih =>
      simp only [MaterializerBoolNodesValid] at hvalid
      rcases hvalid with ⟨hnode, hrest⟩
      have hsound := materializerBoolNode_sound canonOracle canonCount oneId ids
        point values node hcanon honeId hone (by simpa [hsize] using hnode) hagrees
      have hagrees' := valuesAgree_push hagrees (node.evalArith values point)
        (by simpa [hsize] using hnode.2.claimed_lt) (by simpa [hsize] using hsound)
      have hsize' : (values.push (node.evalArith values point)).size = position + 1 := by
        simp [hsize]
      rcases ih (position + 1) (values.push (node.evalArith values point))
          hrest hsize' hagrees' with ⟨hresultSize, hresultAgrees⟩
      simp only [List.foldl]
      constructor
      · rw [hresultSize, Array.size_push]
        simp only [List.length_cons]
        omega
      · exact hresultAgrees

private theorem foldMaterializerBoolChunks_valid
    {arity : Nat} {F : Type*} [CommRing F]
    (canonOracle : Nat → CanonNode) (canonCount oneId : Nat) (ids : Nat → Nat)
    (point : Fin arity → F)
    (chunks : List (List Jolt.LookupExpression.MaterializerBoolNode))
    (position : Nat) (values : Array F)
    (hcanon : CanonRangeValid arity canonOracle 0 canonCount)
    (honeId : oneId < canonCount) (hone : canonOracle oneId = .constant 1)
    (hvalid : MaterializerBoolChunksValid arity canonCount canonOracle oneId ids position chunks)
    (hsize : values.size = position)
    (hagrees : ValuesAgree values ids (canonValues canonOracle point canonCount) canonCount) :
    let result := chunks.foldl
      (fun values nodes => nodes.foldl
        (fun values node => values.push (node.evalArith values point)) values) values
    result.size = values.size + chunks.flatten.length ∧
      ValuesAgree result ids (canonValues canonOracle point canonCount) canonCount := by
  induction chunks generalizing position values with
  | nil =>
      simp only [List.foldl, List.flatten_nil, List.length_nil, Nat.add_zero]
      exact ⟨trivial, hagrees⟩
  | cons nodes chunks ih =>
      simp only [MaterializerBoolChunksValid] at hvalid
      rcases hvalid with ⟨hnodes, hchunks⟩
      have hnodesResult := foldMaterializerBoolNodes_valid canonOracle canonCount oneId ids
        point nodes position values hcanon honeId hone hnodes hsize hagrees
      let next := nodes.foldl
        (fun values node => values.push (node.evalArith values point)) values
      have hnextSize : next.size = position + nodes.length := by
        simpa [next, hsize] using hnodesResult.1
      have hnextAgrees :
          ValuesAgree next ids (canonValues canonOracle point canonCount) canonCount := by
        simpa [next] using hnodesResult.2
      rcases ih (position + nodes.length) next hchunks hnextSize hnextAgrees with
        ⟨hresultSize, hresultAgrees⟩
      simp only [List.foldl]
      constructor
      · rw [hresultSize, hnextSize, hsize]
        simp only [List.flatten_cons, List.length_append]
        omega
      · exact hresultAgrees

private theorem materializerNatNode_sound
    {arity : Nat} {F : Type*} [CommRing F]
    (canonOracle : Nat → CanonNode) (canonCount : Nat)
    (boolIds natIds : Nat → Nat) (point : Fin arity → F)
    (boolValues values : Array F) (node : Jolt.LookupExpression.MaterializerNatNode)
    (hcanon : CanonRangeValid arity canonOracle 0 canonCount)
    (hvalid : node.ValidCanonAt boolValues.size canonCount canonOracle
      boolIds natIds values.size)
    (hboolAgrees : ValuesAgree boolValues boolIds
      (canonValues canonOracle point canonCount) canonCount)
    (hagrees : ValuesAgree values natIds
      (canonValues canonOracle point canonCount) canonCount) :
    node.evalArith boolValues values =
      Jolt.LookupGraph.getValue (canonValues canonOracle point canonCount) (natIds values.size) := by
  rcases hvalid with ⟨hwellFormed, hmatch⟩
  cases node with
  | constant value =>
      exact (CanonStep.node_sound canonOracle point hcanon (.constant value) hmatch).symm
  | ofBit value =>
      simp only [Jolt.LookupExpression.MaterializerNatNode.wellFormedAt] at hwellFormed
      have hvalue : value < boolValues.size := by simpa using hwellFormed
      simp only [Jolt.LookupExpression.MaterializerNatNode.canonStep,
        CanonStep.Matches, CanonStep.matches, Bool.and_eq_true,
        decide_eq_true_eq, beq_iff_eq] at hmatch
      simp only [Jolt.LookupExpression.MaterializerNatNode.evalArith]
      rw [getValue_eq_of_agrees hboolAgrees hvalue, hmatch.2]
  | add left right =>
      simp only [Jolt.LookupExpression.MaterializerNatNode.wellFormedAt,
        Bool.and_eq_true] at hwellFormed
      have hleft : left < values.size := by simpa using hwellFormed.1
      have hright : right < values.size := by simpa using hwellFormed.2
      rw [Jolt.LookupExpression.MaterializerNatNode.evalArith,
        getValue_eq_of_agrees hagrees hleft,
        getValue_eq_of_agrees hagrees hright]
      exact (smartAdd_sound canonOracle point hcanon
        (hagrees left hleft).1 (hagrees right hright).1 hmatch).symm
  | mul left right =>
      simp only [Jolt.LookupExpression.MaterializerNatNode.wellFormedAt,
        Bool.and_eq_true] at hwellFormed
      have hleft : left < values.size := by simpa using hwellFormed.1
      have hright : right < values.size := by simpa using hwellFormed.2
      rw [Jolt.LookupExpression.MaterializerNatNode.evalArith,
        getValue_eq_of_agrees hagrees hleft,
        getValue_eq_of_agrees hagrees hright]
      exact (smartMul_sound canonOracle point hcanon
        (hagrees left hleft).1 (hagrees right hright).1 hmatch).symm

private theorem foldMaterializerNatNodes_valid
    {arity : Nat} {F : Type*} [CommRing F]
    (canonOracle : Nat → CanonNode) (canonCount : Nat)
    (boolIds natIds : Nat → Nat) (point : Fin arity → F)
    (boolValues : Array F) (nodes : List Jolt.LookupExpression.MaterializerNatNode)
    (position : Nat) (values : Array F)
    (hcanon : CanonRangeValid arity canonOracle 0 canonCount)
    (hvalid : MaterializerNatNodesValid boolValues.size canonCount
      canonOracle boolIds natIds position nodes)
    (hsize : values.size = position)
    (hboolAgrees : ValuesAgree boolValues boolIds
      (canonValues canonOracle point canonCount) canonCount)
    (hagrees : ValuesAgree values natIds
      (canonValues canonOracle point canonCount) canonCount) :
    let result := nodes.foldl
      (fun values node => values.push (node.evalArith boolValues values)) values
    result.size = values.size + nodes.length ∧
      ValuesAgree result natIds (canonValues canonOracle point canonCount) canonCount := by
  induction nodes generalizing position values with
  | nil =>
      simp only [List.foldl, List.length_nil, Nat.add_zero]
      exact ⟨trivial, hagrees⟩
  | cons node nodes ih =>
      simp only [MaterializerNatNodesValid] at hvalid
      rcases hvalid with ⟨hnode, hrest⟩
      have hsound := materializerNatNode_sound canonOracle canonCount boolIds natIds
        point boolValues values node hcanon (by simpa [hsize] using hnode) hboolAgrees hagrees
      have hagrees' := valuesAgree_push hagrees (node.evalArith boolValues values)
        (by simpa [hsize] using hnode.2.claimed_lt) (by simpa [hsize] using hsound)
      have hsize' : (values.push (node.evalArith boolValues values)).size = position + 1 := by
        simp [hsize]
      rcases ih (position + 1) (values.push (node.evalArith boolValues values))
          hrest hsize' hagrees' with ⟨hresultSize, hresultAgrees⟩
      simp only [List.foldl]
      constructor
      · rw [hresultSize, Array.size_push]
        simp only [List.length_cons]
        omega
      · exact hresultAgrees

private theorem foldMaterializerNatChunks_valid
    {arity : Nat} {F : Type*} [CommRing F]
    (canonOracle : Nat → CanonNode) (canonCount : Nat)
    (boolIds natIds : Nat → Nat) (point : Fin arity → F)
    (boolValues : Array F)
    (chunks : List (List Jolt.LookupExpression.MaterializerNatNode))
    (position : Nat) (values : Array F)
    (hcanon : CanonRangeValid arity canonOracle 0 canonCount)
    (hvalid : MaterializerNatChunksValid boolValues.size canonCount
      canonOracle boolIds natIds position chunks)
    (hsize : values.size = position)
    (hboolAgrees : ValuesAgree boolValues boolIds
      (canonValues canonOracle point canonCount) canonCount)
    (hagrees : ValuesAgree values natIds
      (canonValues canonOracle point canonCount) canonCount) :
    let result := chunks.foldl
      (fun values nodes => nodes.foldl
        (fun values node => values.push (node.evalArith boolValues values)) values) values
    result.size = values.size + chunks.flatten.length ∧
      ValuesAgree result natIds (canonValues canonOracle point canonCount) canonCount := by
  induction chunks generalizing position values with
  | nil =>
      simp only [List.foldl, List.flatten_nil, List.length_nil, Nat.add_zero]
      exact ⟨trivial, hagrees⟩
  | cons nodes chunks ih =>
      simp only [MaterializerNatChunksValid] at hvalid
      rcases hvalid with ⟨hnodes, hchunks⟩
      have hnodesResult := foldMaterializerNatNodes_valid canonOracle canonCount
        boolIds natIds point boolValues nodes position values hcanon hnodes hsize
        hboolAgrees hagrees
      let next := nodes.foldl
        (fun values node => values.push (node.evalArith boolValues values)) values
      have hnextSize : next.size = position + nodes.length := by
        simpa [next, hsize] using hnodesResult.1
      have hnextAgrees :
          ValuesAgree next natIds (canonValues canonOracle point canonCount) canonCount := by
        simpa [next] using hnodesResult.2
      rcases ih (position + nodes.length) next hchunks hnextSize hnextAgrees with
        ⟨hresultSize, hresultAgrees⟩
      simp only [List.foldl]
      constructor
      · rw [hresultSize, hnextSize, hsize]
        simp only [List.flatten_cons, List.length_append]
        omega
      · exact hresultAgrees

/-- A locally checked materializer evaluates to its claimed canonical root. -/
theorem materializer_eval_eq_canon
    {arity : Nat} {F : Type*} [CommRing F]
    (graph : Jolt.LookupExpression.MaterializerGraph arity)
    (canonOracle : Nat → CanonNode) (canonCount oneId : Nat)
    (boolIds natIds : Nat → Nat)
    (hcanon : CanonRangeValid arity canonOracle 0 canonCount)
    (honeId : oneId < canonCount) (hone : canonOracle oneId = .constant 1)
    (hbool : MaterializerBoolChunksValid arity canonCount canonOracle oneId boolIds
      0 graph.boolNodeChunks)
    (hnat : MaterializerNatChunksValid graph.boolNodes.length canonCount
      canonOracle boolIds natIds 0 graph.natNodeChunks)
    (hroot : graph.root < graph.natNodes.length) (point : Fin arity → F) :
    graph.arithEval point =
      Jolt.LookupGraph.getValue (canonValues canonOracle point canonCount) (natIds graph.root) := by
  have hboolResult := foldMaterializerBoolChunks_valid canonOracle canonCount oneId
    boolIds point graph.boolNodeChunks 0 #[] hcanon honeId hone hbool rfl (by
      intro index hindex
      simp at hindex)
  have hboolAgrees :
      ValuesAgree (graph.evalBoolNodesArith point) boolIds
        (canonValues canonOracle point canonCount) canonCount := by
    simpa [Jolt.LookupExpression.MaterializerGraph.evalBoolNodesArith] using hboolResult.2
  have hboolSize : (graph.evalBoolNodesArith point).size = graph.boolNodes.length := by
    simpa [Jolt.LookupExpression.MaterializerGraph.evalBoolNodesArith,
      Jolt.LookupExpression.MaterializerGraph.boolNodes] using hboolResult.1
  have hnat' : MaterializerNatChunksValid (graph.evalBoolNodesArith point).size canonCount
      canonOracle boolIds natIds 0 graph.natNodeChunks := by
    simpa [hboolSize] using hnat
  have hnatResult := foldMaterializerNatChunks_valid canonOracle canonCount boolIds natIds
    point (graph.evalBoolNodesArith point) graph.natNodeChunks 0 #[] hcanon hnat' rfl
    hboolAgrees (by
      intro index hindex
      simp at hindex)
  have hnatAgrees :
      ValuesAgree
        (graph.evalNatNodesArith (graph.evalBoolNodesArith point)) natIds
        (canonValues canonOracle point canonCount) canonCount := by
    simpa [Jolt.LookupExpression.MaterializerGraph.evalNatNodesArith] using hnatResult.2
  have hnatSize :
      (graph.evalNatNodesArith (graph.evalBoolNodesArith point)).size =
        graph.natNodes.length := by
    simpa [Jolt.LookupExpression.MaterializerGraph.evalNatNodesArith,
      Jolt.LookupExpression.MaterializerGraph.natNodes] using hnatResult.1
  unfold Jolt.LookupExpression.MaterializerGraph.arithEval
  exact getValue_eq_of_agrees hnatAgrees (by rw [hnatSize]; exact hroot)

end Jolt.LookupCorrespondence
