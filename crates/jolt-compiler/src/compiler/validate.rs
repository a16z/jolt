//! L0 validation: reject malformed protocols before compilation.

use std::collections::VecDeque;
use std::fmt;

use crate::ir::expr::Factor;
use crate::ir::{ClaimId, PolyKind, Protocol, PublicPoly, Vertex};

/// A compile-time diagnostic. Validation collects all diagnostics rather
/// than failing on the first — gives the user a complete picture.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub message: String,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

macro_rules! diag {
    ($($arg:tt)*) => {
        Diagnostic { message: format!($($arg)*) }
    };
}

/// Run all validation checks on an L0 protocol.
pub(crate) fn validate(protocol: &Protocol) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    check_indices(protocol, &mut diags);
    check_dimensions(protocol, &mut diags);
    check_compositions(protocol, &mut diags);
    check_public_poly_claims(protocol, &mut diags);
    check_acyclicity(protocol, &mut diags);
    diags
}

/// All factor indices, dim indices, and claim references are in bounds.
fn check_indices(protocol: &Protocol, diags: &mut Vec<Diagnostic>) {
    let num_polys = protocol.polynomials.len();
    let num_challenges = protocol.challenge_names.len();
    let num_dims = protocol.dim_names.len();

    let check_factors = |vi: usize, expr: &crate::ir::expr::Expr, diags: &mut Vec<Diagnostic>| {
        for term in &expr.0 {
            for factor in &term.factors {
                match factor {
                    Factor::Poly(idx) if *idx >= num_polys => {
                        diags.push(diag!("v{vi}: poly index {idx} out of bounds"));
                    }
                    Factor::Challenge(idx) if *idx >= num_challenges => {
                        diags.push(diag!("v{vi}: challenge index {idx} out of bounds"));
                    }
                    Factor::Claim(id) => {
                        if !protocol.claims.iter().any(|c| c.id == *id) {
                            diags.push(diag!("v{vi}: claim c{} does not exist", id.0));
                        }
                    }
                    Factor::Poly(_) | Factor::Challenge(_) => {}
                }
            }
        }
    };

    for (vi, vertex) in protocol.vertices.iter().enumerate() {
        match vertex {
            Vertex::Sumcheck {
                composition,
                input_sum,
                binding_order,
                ..
            } => {
                check_factors(vi, composition, diags);
                check_factors(vi, input_sum, diags);
                for &dim in binding_order {
                    if dim >= num_dims {
                        diags.push(diag!("v{vi}: dim index {dim} out of bounds"));
                    }
                }
            }
            Vertex::Evaluate {
                poly, at_vertex, ..
            } => {
                if *poly >= num_polys {
                    diags.push(diag!("v{vi}: poly index {} out of bounds", *poly));
                }
                if *at_vertex >= protocol.vertices.len() {
                    diags.push(diag!(
                        "v{vi}: evaluate target v{} out of bounds",
                        *at_vertex
                    ));
                }
            }
        }
    }
}

/// Every poly in a composition has its declared dims covered by binding_order.
fn check_dimensions(protocol: &Protocol, diags: &mut Vec<Diagnostic>) {
    for (vi, vertex) in protocol.vertices.iter().enumerate() {
        let (composition, binding_order) = match vertex {
            Vertex::Sumcheck {
                composition,
                binding_order,
                ..
            } => (composition, binding_order),
            Vertex::Evaluate { .. } => continue,
        };

        let mut poly_indices: Vec<usize> = composition
            .0
            .iter()
            .flat_map(|term| {
                term.factors.iter().filter_map(|f| match f {
                    Factor::Poly(idx) => Some(*idx),
                    _ => None,
                })
            })
            .collect();
        poly_indices.sort_unstable();
        poly_indices.dedup();

        for poly_idx in poly_indices {
            if poly_idx >= protocol.polynomials.len() {
                continue; // already reported by check_indices
            }
            let poly_dims = &protocol.polynomials[poly_idx].dims;
            // A poly may have dims already bound by a prior vertex in the DAG.
            // Only flag polys with ZERO dim overlap — that's truly invalid.
            if !poly_dims.is_empty() && !poly_dims.iter().any(|d| binding_order.contains(d)) {
                let missing: Vec<usize> = poly_dims
                    .iter()
                    .filter(|d| !binding_order.contains(d))
                    .copied()
                    .collect();
                diags.push(diag!(
                    "v{vi}: poly {poly_idx} has dims {missing:?} not covered by binding_order"
                ));
            }
        }
    }
}

/// PublicPoly point references are valid claims.
fn check_public_poly_claims(protocol: &Protocol, diags: &mut Vec<Diagnostic>) {
    let extract_point = |pp: &PublicPoly| -> Option<ClaimId> {
        match pp {
            PublicPoly::Eq(c) | PublicPoly::EqPlusOne(c) | PublicPoly::Lt(c) => *c,
            PublicPoly::Identity | PublicPoly::Preprocessed => None,
        }
    };
    for (pi, poly) in protocol.polynomials.iter().enumerate() {
        if let PolyKind::Public(ref pp) = poly.kind {
            if let Some(claim_id) = extract_point(pp) {
                if !protocol.claims.iter().any(|c| c.id == claim_id) {
                    diags.push(diag!(
                        "poly {pi}: PublicPoly references claim c{} which does not exist",
                        claim_id.0
                    ));
                }
            }
        }
    }
}

/// No empty compositions.
fn check_compositions(protocol: &Protocol, diags: &mut Vec<Diagnostic>) {
    for (vi, vertex) in protocol.vertices.iter().enumerate() {
        let comp = match vertex {
            Vertex::Sumcheck { composition, .. } => Some(composition),
            Vertex::Evaluate { .. } => None,
        };
        if let Some(composition) = comp {
            if composition.0.is_empty() {
                diags.push(diag!("v{vi}: empty composition"));
            }
        }
    }
}

/// Claim graph is acyclic (Kahn's algorithm).
fn check_acyclicity(protocol: &Protocol, diags: &mut Vec<Diagnostic>) {
    let n = protocol.vertices.len();
    if n == 0 {
        return;
    }

    let mut in_degree = vec![0usize; n];
    let mut successors: Vec<Vec<usize>> = vec![vec![]; n];

    // Map claim_id -> producing vertex
    let mut claim_to_vertex: Vec<(ClaimId, usize)> = protocol
        .claims
        .iter()
        .map(|c| (c.id, c.produced_by))
        .collect();
    claim_to_vertex.sort_unstable_by_key(|(id, _)| *id);

    for (vi, vertex) in protocol.vertices.iter().enumerate() {
        // Claim-based dependencies (sumcheck input_sum referencing upstream claims)
        for &consumed in vertex.consumes() {
            if let Ok(pos) = claim_to_vertex.binary_search_by_key(&consumed, |(id, _)| *id) {
                let producer = claim_to_vertex[pos].1;
                if producer != vi {
                    successors[producer].push(vi);
                    in_degree[vi] += 1;
                }
            }
        }
        if let Vertex::Evaluate { at_vertex, .. } = vertex {
            if *at_vertex < n && *at_vertex != vi {
                successors[*at_vertex].push(vi);
                in_degree[vi] += 1;
            }
        }
    }

    // Kahn's BFS
    let mut queue: VecDeque<usize> = (0..n).filter(|&v| in_degree[v] == 0).collect();
    let mut consumed = 0usize;

    while let Some(v) = queue.pop_front() {
        consumed += 1;
        for &succ in &successors[v] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push_back(succ);
            }
        }
    }

    if consumed < n {
        let stuck: Vec<usize> = (0..n).filter(|&v| in_degree[v] > 0).collect();
        diags.push(diag!("cycle detected involving vertices {stuck:?}"));
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::ir::expr::{Expr, Factor, Term};
    use crate::ir::{Claim, PolyKind, PublicPoly, Vertex};

    fn valid_protocol() -> Protocol {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let upstream = p.sumcheck(eq * (a * b), 0, &[d]);
        let c = p.poly("c", &[d], PolyKind::Virtual);
        let rho = p.challenge("rho");
        let _ = p.sumcheck(eq * c, rho * upstream[0] + upstream[1], &[d]);
        p
    }

    #[test]
    fn valid_protocol_passes() {
        let diags = validate(&valid_protocol());
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    }

    #[test]
    fn poly_out_of_bounds() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let _ = p.poly("a", &[d], PolyKind::Virtual);
        p.vertices.push(Vertex::Sumcheck {
            composition: Expr(vec![Term {
                coeff: 1,
                factors: vec![Factor::Poly(99)],
            }]),
            input_sum: Expr::from(0i64),
            produces: vec![],
            consumes: vec![],
            binding_order: vec![d],
            domain_size: None,
        });
        let diags = validate(&p);
        assert!(diags.iter().any(|d| d.message.contains("poly index 99")));
    }

    #[test]
    fn challenge_out_of_bounds() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let a = p.poly("a", &[d], PolyKind::Virtual);
        p.vertices.push(Vertex::Sumcheck {
            composition: Expr(vec![Term {
                coeff: 1,
                factors: vec![Factor::Poly(a.0), Factor::Challenge(42)],
            }]),
            input_sum: Expr::from(0i64),
            produces: vec![],
            consumes: vec![],
            binding_order: vec![d],
            domain_size: None,
        });
        let diags = validate(&p);
        assert!(diags
            .iter()
            .any(|d| d.message.contains("challenge index 42")));
    }

    #[test]
    fn dim_out_of_bounds() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let a = p.poly("a", &[d], PolyKind::Virtual);
        p.vertices.push(Vertex::Sumcheck {
            composition: Expr::from(a),
            input_sum: Expr::from(0i64),
            produces: vec![],
            consumes: vec![],
            binding_order: vec![77],
            domain_size: None,
        });
        let diags = validate(&p);
        assert!(diags.iter().any(|d| d.message.contains("dim index 77")));
    }

    #[test]
    fn dimension_mismatch() {
        let mut p = Protocol::new();
        let log_T = p.dim("log_T");
        let log_K = p.dim("log_K");

        // Partial overlap (poly has [log_T, log_K], vertex binds [log_T]) is OK:
        // log_K may have been bound by a prior vertex in the DAG.
        let a = p.poly("a", &[log_T, log_K], PolyKind::Virtual);
        p.vertices.push(Vertex::Sumcheck {
            composition: Expr::from(a),
            input_sum: Expr::from(0i64),
            produces: vec![],
            consumes: vec![],
            binding_order: vec![log_T],
            domain_size: None,
        });
        let diags = validate(&p);
        assert!(
            !diags
                .iter()
                .any(|d| d.message.contains("not covered by binding_order")),
            "partial overlap should not be flagged"
        );

        // Zero overlap (poly has [log_K], vertex binds [log_T]) IS an error.
        let mut p2 = Protocol::new();
        let log_T2 = p2.dim("log_T");
        let log_K2 = p2.dim("log_K");
        let b = p2.poly("b", &[log_K2], PolyKind::Virtual);
        p2.vertices.push(Vertex::Sumcheck {
            composition: Expr::from(b),
            input_sum: Expr::from(0i64),
            produces: vec![],
            consumes: vec![],
            binding_order: vec![log_T2],
            domain_size: None,
        });
        let diags2 = validate(&p2);
        assert!(diags2
            .iter()
            .any(|d| d.message.contains("not covered by binding_order")));
    }

    #[test]
    fn empty_composition() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        p.vertices.push(Vertex::Sumcheck {
            composition: Expr::from(0i64),
            input_sum: Expr::from(0i64),
            produces: vec![],
            consumes: vec![],
            binding_order: vec![d],
            domain_size: None,
        });
        let diags = validate(&p);
        assert!(diags
            .iter()
            .any(|d| d.message.contains("empty composition")));
    }

    #[test]
    fn cycle_detected() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);

        let c0 = ClaimId(0);
        let c1 = ClaimId(1);
        p.claims.push(Claim {
            id: c0,
            poly: a.0,
            produced_by: 0,
        });
        p.claims.push(Claim {
            id: c1,
            poly: b.0,
            produced_by: 1,
        });
        p.vertices.push(Vertex::Sumcheck {
            composition: Expr::from(a),
            input_sum: Expr::from(c1),
            produces: vec![c0],
            consumes: vec![c1],
            binding_order: vec![d],
            domain_size: None,
        });
        p.vertices.push(Vertex::Sumcheck {
            composition: Expr::from(b),
            input_sum: Expr::from(c0),
            produces: vec![c1],
            consumes: vec![c0],
            binding_order: vec![d],
            domain_size: None,
        });

        let diags = validate(&p);
        assert!(diags.iter().any(|d| d.message.contains("cycle detected")));
    }

    #[test]
    fn claim_out_of_bounds_in_input_sum() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let a = p.poly("a", &[d], PolyKind::Virtual);
        p.vertices.push(Vertex::Sumcheck {
            composition: Expr::from(a),
            input_sum: Expr::from(ClaimId(999)),
            produces: vec![],
            consumes: vec![ClaimId(999)],
            binding_order: vec![d],
            domain_size: None,
        });
        let diags = validate(&p);
        assert!(diags.iter().any(|d| d.message.contains("claim c999")));
    }

    #[test]
    fn multiple_errors_collected() {
        let mut p = Protocol::new();
        let _d = p.dim("d");
        p.vertices.push(Vertex::Sumcheck {
            composition: Expr::from(0i64),
            input_sum: Expr::from(0i64),
            produces: vec![],
            consumes: vec![],
            binding_order: vec![99],
            domain_size: None,
        });
        let diags = validate(&p);
        assert!(diags.len() >= 2, "expected multiple diagnostics: {diags:?}");
    }

    #[test]
    fn evaluate_valid() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let upstream = p.sumcheck(eq * a, 0, &[d]);
        let _ = p.evaluate(b, upstream[0]);
        let diags = validate(&p);
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    }
}
