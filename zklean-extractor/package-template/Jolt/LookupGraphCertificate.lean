import Jolt.LookupGraphExpression

namespace Jolt.LookupGraph

/-- Summarize a node from an externally supplied summary oracle. -/
def Node.summaryFrom {arity : Nat} (oracle : Nat → ExprSummary arity) : Node → ExprSummary arity
  | .constant _ => ⟨0, true⟩
  | .input index => if index < arity then ⟨1 <<< index, true⟩ else ⟨0, true⟩
  | .add left right | .sub left right =>
      let left := oracle left
      let right := oracle right
      ⟨left.support ||| right.support, left.multilinear && right.multilinear⟩
  | .mul left right =>
      let left := oracle left
      let right := oracle right
      ⟨left.support ||| right.support,
        left.multilinear && right.multilinear &&
          decide (left.support &&& right.support = 0)⟩

/-- One node is well formed and has the summary claimed by the oracle. -/
def Node.ValidSummaryAt {arity : Nat} (oracle : Nat → ExprSummary arity)
    (position : Nat) (node : Node) : Prop :=
  node.wellFormedAt arity position = true ∧ node.summaryFrom oracle = oracle position

/-- A consecutive list of nodes has the summaries claimed by the oracle. -/
def NodesValid {arity : Nat} (oracle : Nat → ExprSummary arity) : Nat → List Node → Prop
  | _, [] => True
  | position, node :: nodes =>
      node.ValidSummaryAt oracle position ∧ NodesValid oracle (position + 1) nodes

/-- Chunked nodes have the summaries claimed by the oracle. -/
def ChunksValid {arity : Nat} (oracle : Nat → ExprSummary arity) :
    Nat → List (List Node) → Prop
  | _, [] => True
  | position, nodes :: chunks =>
      NodesValid oracle position nodes ∧
        ChunksValid oracle (position + nodes.length) chunks

/-- A graph-local summary certificate. -/
def Graph.ValidSummary {arity : Nat} (graph : Graph arity)
    (oracle : Nat → ExprSummary arity) : Prop :=
  ChunksValid oracle 0 graph.nodeChunks ∧ graph.root < graph.nodes.length

private def SummariesAgree {arity : Nat} (summaries : Array (ExprSummary arity))
    (oracle : Nat → ExprSummary arity) : Prop :=
  ∀ (index : Nat) (h : index < summaries.size), summaries[index] = oracle index

private theorem getSummary_eq_of_agrees {arity : Nat}
    {summaries : Array (ExprSummary arity)} {oracle : Nat → ExprSummary arity}
    (hagree : SummariesAgree summaries oracle) {index : Nat} (hindex : index < summaries.size) :
    getSummary summaries index = oracle index := by
  unfold getSummary
  rw [Array.getElem?_eq_getElem hindex]
  simp [hagree index hindex]

private theorem summariesAgree_push {arity : Nat}
    {summaries : Array (ExprSummary arity)} {oracle : Nat → ExprSummary arity}
    (hagree : SummariesAgree summaries oracle) (summary : ExprSummary arity)
    (hsummary : summary = oracle summaries.size) :
    SummariesAgree (summaries.push summary) oracle := by
  intro index hindex
  rw [Array.getElem_push]
  split
  · exact hagree index (by assumption)
  · have : index = summaries.size := by
      rw [Array.size_push] at hindex
      omega
    subst index
    exact hsummary

private theorem Node.summary_eq_summaryFrom {arity : Nat}
    (oracle : Nat → ExprSummary arity) (summaries : Array (ExprSummary arity))
    (node : Node) (hwellFormed : node.wellFormedAt arity summaries.size = true)
    (hagrees : SummariesAgree summaries oracle) :
    node.summary summaries = node.summaryFrom oracle := by
  cases node with
  | constant => rfl
  | input index =>
      simp only [Node.wellFormedAt] at hwellFormed
      simp [Node.summary, Node.summaryFrom]
  | add left right | sub left right | mul left right =>
      simp only [Node.wellFormedAt, Bool.and_eq_true] at hwellFormed
      have hleft : left < summaries.size := by simpa using hwellFormed.1
      have hright : right < summaries.size := by simpa using hwellFormed.2
      simp only [Node.summary, Node.summaryFrom]
      rw [getSummary_eq_of_agrees hagrees hleft,
        getSummary_eq_of_agrees hagrees hright]

private theorem foldNodes_valid {arity : Nat}
    (oracle : Nat → ExprSummary arity) (nodes : List Node) (position : Nat)
    (summaries : Array (ExprSummary arity))
    (hvalid : NodesValid oracle position nodes)
    (hsize : summaries.size = position)
    (hagrees : SummariesAgree summaries oracle) :
    let result := nodes.foldl
      (fun summaries node => summaries.push (node.summary summaries)) summaries
    result.size = summaries.size + nodes.length ∧ SummariesAgree result oracle := by
  induction nodes generalizing position summaries with
  | nil =>
      simp only [List.foldl, List.length_nil, Nat.add_zero]
      exact ⟨trivial, hagrees⟩
  | cons node nodes ih =>
      simp only [NodesValid] at hvalid
      rcases hvalid with ⟨⟨hwellFormed, hsummary⟩, hrest⟩
      have hnode := node.summary_eq_summaryFrom oracle summaries
        (by simpa [hsize] using hwellFormed) hagrees
      have hnew : node.summary summaries = oracle summaries.size := by
        rw [hnode, hsummary, hsize]
      have hagrees' := summariesAgree_push hagrees (node.summary summaries) hnew
      have hsize' : (summaries.push (node.summary summaries)).size = position + 1 := by
        simp [hsize]
      rcases ih (position + 1) (summaries.push (node.summary summaries))
          hrest hsize' hagrees' with ⟨hresultSize, hresultAgrees⟩
      simp only [List.foldl]
      constructor
      · rw [hresultSize, Array.size_push]
        simp only [List.length_cons]
        omega
      · exact hresultAgrees

private theorem foldChunks_valid {arity : Nat}
    (oracle : Nat → ExprSummary arity) (chunks : List (List Node)) (position : Nat)
    (summaries : Array (ExprSummary arity))
    (hvalid : ChunksValid oracle position chunks)
    (hsize : summaries.size = position)
    (hagrees : SummariesAgree summaries oracle) :
    let result := chunks.foldl Graph.summaryChunk summaries
    result.size = summaries.size + chunks.flatten.length ∧ SummariesAgree result oracle := by
  induction chunks generalizing position summaries with
  | nil =>
      simp only [List.foldl, List.flatten_nil, List.length_nil, Nat.add_zero]
      exact ⟨trivial, hagrees⟩
  | cons nodes chunks ih =>
      simp only [ChunksValid] at hvalid
      rcases hvalid with ⟨hnodes, hchunks⟩
      have hnodesResult := foldNodes_valid oracle nodes position summaries hnodes hsize hagrees
      let next := Graph.summaryChunk summaries nodes
      have hnextSize : next.size = position + nodes.length := by
        simpa [next, Graph.summaryChunk, hsize] using hnodesResult.1
      have hnextAgrees : SummariesAgree next oracle := by
        simpa [next, Graph.summaryChunk] using hnodesResult.2
      rcases ih (position + nodes.length) next hchunks hnextSize hnextAgrees with
        ⟨hresultSize, hresultAgrees⟩
      simp only [List.foldl]
      constructor
      · rw [hresultSize, hnextSize, hsize]
        simp only [List.flatten_cons, List.length_append]
        omega
      · exact hresultAgrees

/-- A local summary certificate determines the result of the shared graph checker. -/
theorem Graph.checkMultilinear_eq_of_validSummary {arity : Nat}
    (graph : Graph arity) (oracle : Nat → ExprSummary arity)
    (hvalid : graph.ValidSummary oracle) :
    graph.checkMultilinear = (oracle graph.root).multilinear := by
  rcases hvalid with ⟨hchunks, hroot⟩
  have hresult := foldChunks_valid oracle graph.nodeChunks 0 #[] hchunks rfl (by
    intro index hindex
    simp at hindex)
  have hresult' :
      graph.summaryNodes.size = graph.nodes.length ∧
        SummariesAgree graph.summaryNodes oracle := by
    simpa [Graph.summaryNodes, Graph.nodes] using hresult
  have hrootSize : graph.root < graph.summaryNodes.size := by
    rw [hresult'.1]
    exact hroot
  unfold Graph.checkMultilinear
  rw [getSummary_eq_of_agrees hresult'.2 hrootSize]

end Jolt.LookupGraph
