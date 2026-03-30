//! L0 analysis: compute derived properties of a validated protocol.

use crate::ir::expr::Factor;
use crate::ir::{ClaimId, Protocol, Vertex};

/// Derived properties of a validated L0 protocol. Consumed by staging
/// and lowering passes.
#[derive(Clone, Debug)]
pub struct IRInfo {
    /// Vertex indices in topological (dependency) order.
    pub topo_order: Vec<usize>,
    /// Longest dependency chain ending at each vertex.
    /// `depth[v] = 0` for source vertices (no predecessors).
    pub depth: Vec<usize>,
    /// Length of the longest chain in the DAG. Lower bound on stage count.
    pub critical_path: usize,
    /// Polynomial degree of each vertex's composition (max poly-factor
    /// count in any term). Determines sumcheck evaluation points per round.
    /// Evaluate vertices have degree 0.
    pub degree: Vec<usize>,
    /// `successors[v]` = vertex indices that consume claims produced by `v`.
    pub successors: Vec<Vec<usize>>,
    /// `predecessors[v]` = vertex indices that produce claims consumed by `v`.
    pub predecessors: Vec<Vec<usize>>,
}

/// Compute all derived properties. Caller must ensure the protocol has
/// already passed validation.
pub(crate) fn compute(protocol: &Protocol) -> IRInfo {
    let n = protocol.vertices.len();

    let (successors, predecessors) = build_adjacency(protocol);
    let topo_order = toposort(n, &successors, &predecessors);
    let depth = compute_depth(n, &topo_order, &predecessors);
    let critical_path = depth.iter().copied().max().unwrap_or(0);
    let degree = compute_degree(protocol);

    IRInfo {
        topo_order,
        depth,
        critical_path,
        degree,
        successors,
        predecessors,
    }
}

/// Build successor and predecessor adjacency lists from the claim graph.
fn build_adjacency(protocol: &Protocol) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = protocol.vertices.len();
    let mut successors: Vec<Vec<usize>> = vec![vec![]; n];
    let mut predecessors: Vec<Vec<usize>> = vec![vec![]; n];

    // Map claim_id -> producing vertex (sorted for binary search)
    let mut claim_to_vertex: Vec<(ClaimId, usize)> = protocol
        .claims
        .iter()
        .map(|c| (c.id, c.produced_by))
        .collect();
    claim_to_vertex.sort_unstable_by_key(|(id, _)| *id);

    for (vi, vertex) in protocol.vertices.iter().enumerate() {
        // Claim-based dependencies
        for &consumed in vertex.consumes() {
            if let Ok(pos) = claim_to_vertex.binary_search_by_key(&consumed, |(id, _)| *id) {
                let producer = claim_to_vertex[pos].1;
                if producer != vi {
                    successors[producer].push(vi);
                    predecessors[vi].push(producer);
                }
            }
        }
        // Evaluate vertex depends on its target vertex
        if let Vertex::Evaluate { at_vertex, .. } = vertex {
            if *at_vertex != vi {
                successors[*at_vertex].push(vi);
                predecessors[vi].push(*at_vertex);
            }
        }
    }

    // Dedup (a vertex may consume multiple claims from the same producer)
    for adj in successors.iter_mut().chain(predecessors.iter_mut()) {
        adj.sort_unstable();
        adj.dedup();
    }

    (successors, predecessors)
}

/// Kahn's algorithm. Protocol is already validated acyclic.
fn toposort(n: usize, successors: &[Vec<usize>], predecessors: &[Vec<usize>]) -> Vec<usize> {
    let mut in_degree: Vec<usize> = predecessors.iter().map(|p| p.len()).collect();
    let mut queue: Vec<usize> = (0..n).filter(|&v| in_degree[v] == 0).collect();
    let mut order = Vec::with_capacity(n);

    while let Some(v) = queue.pop() {
        order.push(v);
        for &succ in &successors[v] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push(succ);
            }
        }
    }

    order
}

/// Depth of each vertex: longest chain ending at that vertex.
/// DP on topological order: `depth[v] = max(depth[pred] + 1)` for all predecessors.
fn compute_depth(n: usize, topo_order: &[usize], predecessors: &[Vec<usize>]) -> Vec<usize> {
    let mut depth = vec![0usize; n];
    for &v in topo_order {
        for &pred in &predecessors[v] {
            let candidate = depth[pred] + 1;
            if candidate > depth[v] {
                depth[v] = candidate;
            }
        }
    }
    depth
}

/// Max number of Poly factors in any term of each vertex's composition.
/// Evaluate vertices have degree 0.
fn compute_degree(protocol: &Protocol) -> Vec<usize> {
    protocol
        .vertices
        .iter()
        .map(|vertex| match vertex {
            Vertex::Sumcheck { composition, .. } => composition
                .0
                .iter()
                .map(|term| {
                    term.factors
                        .iter()
                        .filter(|f| matches!(f, Factor::Poly(_)))
                        .count()
                })
                .max()
                .unwrap_or(0),
            Vertex::Evaluate { .. } => 0,
        })
        .collect()
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::ir::{PolyKind, PublicPoly};

    /// Linear chain: v0 -> v1 -> v2
    fn linear_chain() -> Protocol {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let rho = p.challenge("rho");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let c = p.poly("c", &[d], PolyKind::Virtual);

        let c0 = p.sumcheck(eq * a, 0, &[d]);
        let c1 = p.sumcheck(eq * b, rho * c0[0], &[d]);
        let _ = p.sumcheck(eq * c, rho * c1[0], &[d]);
        p
    }

    /// Diamond: v0 -> v1, v0 -> v2, v1 -> v3, v2 -> v3
    fn diamond() -> Protocol {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let rho = p.challenge("rho");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let c = p.poly("c", &[d], PolyKind::Virtual);
        let e = p.poly("e", &[d], PolyKind::Virtual);

        let c0 = p.sumcheck(eq * a, 0, &[d]);
        let c1 = p.sumcheck(eq * b, rho * c0[0], &[d]);
        let c2 = p.sumcheck(eq * c, rho * c0[0], &[d]);
        let _ = p.sumcheck(eq * e, rho * c1[0] + c2[0], &[d]);
        p
    }

    #[test]
    fn linear_topo_order() {
        let info = compute(&linear_chain());
        for (pos, &v) in info.topo_order.iter().enumerate() {
            for &pred in &info.predecessors[v] {
                let pred_pos = info.topo_order.iter().position(|&x| x == pred).unwrap();
                assert!(pred_pos < pos, "pred v{pred} should appear before v{v}");
            }
        }
    }

    #[test]
    fn linear_depth() {
        let info = compute(&linear_chain());
        assert_eq!(info.depth[0], 0);
        assert_eq!(info.depth[1], 1);
        assert_eq!(info.depth[2], 2);
        assert_eq!(info.critical_path, 2);
    }

    #[test]
    fn diamond_depth() {
        let info = compute(&diamond());
        assert_eq!(info.depth[0], 0); // v0: source
        assert_eq!(info.depth[1], 1); // v1: depends on v0
        assert_eq!(info.depth[2], 1); // v2: depends on v0
        assert_eq!(info.depth[3], 2); // v3: depends on v1 and v2
        assert_eq!(info.critical_path, 2);
    }

    #[test]
    fn diamond_adjacency() {
        let info = compute(&diamond());
        assert!(info.successors[0].contains(&1));
        assert!(info.successors[0].contains(&2));
        assert!(info.predecessors[3].contains(&1));
        assert!(info.predecessors[3].contains(&2));
    }

    #[test]
    fn degree_computation() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let gamma = p.challenge("gamma");

        // eq * (a * b): degree 3 (three Poly factors)
        let c0 = p.sumcheck(eq * (a * b), 0, &[d]);
        // eq * gamma * a: degree 2 (two Poly factors, gamma is Challenge)
        let _ = p.sumcheck(eq * gamma * a, gamma * c0[0] + c0[1], &[d]);

        let info = compute(&p);
        assert_eq!(info.degree[0], 3);
        assert_eq!(info.degree[1], 2);
    }

    #[test]
    fn independent_vertices() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);

        let _ = p.sumcheck(eq * a, 0, &[d]);
        let _ = p.sumcheck(eq * b, 0, &[d]);

        let info = compute(&p);
        assert_eq!(info.depth[0], 0);
        assert_eq!(info.depth[1], 0);
        assert_eq!(info.critical_path, 0);
        assert!(info.successors[0].is_empty());
        assert!(info.successors[1].is_empty());
    }

    #[test]
    fn empty_protocol() {
        let p = Protocol::new();
        let info = compute(&p);
        assert!(info.topo_order.is_empty());
        assert_eq!(info.critical_path, 0);
    }

    #[test]
    fn evaluate_in_dag() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let rho = p.challenge("rho");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let c = p.poly("c", &[d], PolyKind::Virtual);

        let upstream = p.sumcheck(eq * a, 0, &[d]); // v0
        let b_eval = p.evaluate(b, upstream[0]); // v1 (eval @ v0)
        let _ = p.sumcheck(eq * c, rho * b_eval, &[d]); // v2 consumes b_eval

        let info = compute(&p);
        // v0 -> v1 (eval depends on v0), v1 -> v2 (v2 consumes b_eval from v1)
        assert_eq!(info.depth[0], 0);
        assert_eq!(info.depth[1], 1);
        assert_eq!(info.depth[2], 2);
        assert_eq!(info.degree[1], 0); // evaluate vertex has degree 0
    }
}
