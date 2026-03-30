//! Graphviz DOT output for protocol IR visualization.
//!
//! Renders an L0 [`Protocol`] as a Graphviz DOT digraph. Vertices become
//! rich HTML-label nodes showing composition formulas; claims become edges.
//!
//! ```text
//! # Render to SVG:
//! jolt-dot protocol.json | dot -Tsvg -o protocol.svg
//! ```

use std::fmt::Write;

use crate::ir::expr::{Expr, Factor};
use crate::ir::{ClaimId, PolyKind, Protocol};

const DEPTH_COLORS: &[&str] = &[
    "#d4edda", // 0: green  — sources
    "#cce5ff", // 1: blue
    "#e2d5f1", // 2: purple
    "#fff3cd", // 3: yellow
    "#f8d7da", // 4: red
    "#d1ecf1", // 5: cyan
    "#fde2e4", // 6: pink
];

const EDGE_COMMITTED: &str = "#2b6cb0";
const EDGE_VIRTUAL: &str = "#888888";

/// Render an L0 [`Protocol`] as a Graphviz DOT string.
///
/// Vertices are rendered as HTML-label nodes with composition formulas.
/// Claims become directed edges labeled with the claim ID and polynomial
/// name. Node background color encodes DAG depth.
pub fn protocol_to_dot(protocol: &Protocol) -> String {
    let depth = compute_depth(protocol);
    let max_depth = depth.iter().copied().max().unwrap_or(0);
    let mut dot = String::with_capacity(4096);

    // --- preamble ---
    let _ = writeln!(dot, "digraph Protocol {{");
    let _ = writeln!(dot, "    rankdir=TB;");
    let _ = writeln!(
        dot,
        "    graph [fontname=\"Helvetica\" fontsize=12 labeljust=l];"
    );
    let _ = writeln!(dot, "    node  [shape=plaintext fontname=\"Helvetica\"];");
    let _ = writeln!(dot, "    edge  [fontname=\"Helvetica\" fontsize=10];");
    let _ = writeln!(dot);
    let _ = writeln!(
        dot,
        "    label=<\
         <B>Protocol</B> &mdash; \
         {} vertices, {} polys, {} claims, {} dims, \
         critical path {}\
         >;",
        protocol.vertices.len(),
        protocol.polynomials.len(),
        protocol.claims.len(),
        protocol.dim_names.len(),
        max_depth,
    );
    let _ = writeln!(dot);

    // --- vertex nodes ---
    for (i, vertex) in protocol.vertices.iter().enumerate() {
        let d = depth[i];
        let bg = DEPTH_COLORS[d % DEPTH_COLORS.len()];

        let produces_items: Vec<String> = vertex
            .produces()
            .iter()
            .map(|cid| {
                let poly_idx = protocol.claims[cid.0 as usize].poly;
                let name = html_escape(&protocol.polynomials[poly_idx].name);
                let kind_tag = match &protocol.polynomials[poly_idx].kind {
                    PolyKind::Committed => " <FONT COLOR=\"#2b6cb0\">[C]</FONT>",
                    PolyKind::Virtual => "",
                    PolyKind::Public(_) => " <FONT COLOR=\"#38a169\">[P]</FONT>",
                };
                format!("c{}: {name}{kind_tag}", cid.0)
            })
            .collect();

        let produces_row = if produces_items.is_empty() {
            String::new()
        } else {
            format!(
                "<TR><TD COLSPAN=\"2\" ALIGN=\"LEFT\">\
                 <FONT POINT-SIZE=\"9\" COLOR=\"#666666\">\
                 &rarr; {}</FONT></TD></TR>\n",
                produces_items.join(", ")
            )
        };

        match vertex {
            crate::ir::Vertex::Sumcheck {
                composition,
                input_sum,
                binding_order,
                domain_size,
                ..
            } => {
                let dims = binding_order
                    .iter()
                    .map(|&di| protocol.dim_names[di].as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let comp = fmt_expr_html(protocol, composition);
                let input = fmt_expr_html(protocol, input_sum);
                let label = if let Some(d) = domain_size {
                    format!("uniskip D={d}, {dims}")
                } else {
                    dims
                };

                let _ = write!(
                    dot,
                    "    v{i} [label=<\n\
                     <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"5\">\n\
                       <TR><TD BGCOLOR=\"{bg}\"><B>v{i}</B></TD>\
                           <TD BGCOLOR=\"{bg}\">[{label}]</TD></TR>\n\
                       <TR><TD COLSPAN=\"2\" ALIGN=\"LEFT\">\
                           <FONT FACE=\"Courier\">{comp}</FONT> = \
                           <FONT FACE=\"Courier\">{input}</FONT></TD></TR>\n\
                       {produces_row}\
                     </TABLE>>];\n",
                );
            }
            crate::ir::Vertex::Evaluate {
                poly, at_vertex, ..
            } => {
                let poly_name = html_escape(&protocol.polynomials[*poly].name);
                let _ = write!(
                    dot,
                    "    v{i} [label=<\n\
                     <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"5\">\n\
                       <TR><TD BGCOLOR=\"{bg}\"><B>v{i}</B></TD>\
                           <TD BGCOLOR=\"{bg}\">[eval]</TD></TR>\n\
                       <TR><TD COLSPAN=\"2\" ALIGN=\"LEFT\">\
                           <FONT FACE=\"Courier\">{poly_name} @ v{at_vertex}</FONT></TD></TR>\n\
                       {produces_row}\
                     </TABLE>>];\n",
                );
            }
        }
    }

    let _ = writeln!(dot);

    // --- claim edges ---
    for (vi, vertex) in protocol.vertices.iter().enumerate() {
        for &claim_id in vertex.consumes() {
            if let Some(claim) = protocol.claims.iter().find(|c| c.id == claim_id) {
                let producer = claim.produced_by;
                let poly_name = html_escape(&protocol.polynomials[claim.poly].name);
                let color = match &protocol.polynomials[claim.poly].kind {
                    PolyKind::Committed => EDGE_COMMITTED,
                    _ => EDGE_VIRTUAL,
                };
                let penwidth = match &protocol.polynomials[claim.poly].kind {
                    PolyKind::Committed => "2.0",
                    _ => "1.0",
                };
                let _ = writeln!(
                    dot,
                    "    v{producer} -> v{vi} [\
                     label=<c{}: <I>{poly_name}</I>> \
                     color=\"{color}\" penwidth={penwidth}];",
                    claim_id.0,
                );
            }
        }
    }

    // --- legend ---
    let _ = writeln!(dot);
    let _ = write!(
        dot,
        "    legend [label=<\n\
         <TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"2\" CELLPADDING=\"3\">\n\
           <TR><TD COLSPAN=\"2\"><B>Legend</B></TD></TR>\n\
           <TR><TD BGCOLOR=\"{}\">&#9632;</TD><TD ALIGN=\"LEFT\">Source (depth 0)</TD></TR>\n\
           <TR><TD BGCOLOR=\"{}\">&#9632;</TD><TD ALIGN=\"LEFT\">Depth 1</TD></TR>\n\
           <TR><TD BGCOLOR=\"{}\">&#9632;</TD><TD ALIGN=\"LEFT\">Depth 2+</TD></TR>\n\
           <TR><TD><FONT COLOR=\"{EDGE_COMMITTED}\">&#9644;&#9644;</FONT></TD>\
               <TD ALIGN=\"LEFT\">Committed poly</TD></TR>\n\
           <TR><TD><FONT COLOR=\"{EDGE_VIRTUAL}\">&#9644;&#9644;</FONT></TD>\
               <TD ALIGN=\"LEFT\">Virtual poly</TD></TR>\n",
        DEPTH_COLORS[0], DEPTH_COLORS[1], DEPTH_COLORS[2],
    );

    if !protocol.dim_names.is_empty() {
        let dims = protocol.dim_names.join(", ");
        let _ = writeln!(
            dot,
            "           <TR><TD></TD><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"9\">\
             Dims: {dims}</FONT></TD></TR>"
        );
    }

    let committed: Vec<&str> = protocol
        .polynomials
        .iter()
        .filter(|p| matches!(p.kind, PolyKind::Committed))
        .map(|p| p.name.as_str())
        .collect();
    if !committed.is_empty() {
        let names = committed.join(", ");
        let _ = writeln!(
            dot,
            "           <TR><TD></TD><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"9\" COLOR=\"#2b6cb0\">\
             Committed: {names}</FONT></TD></TR>"
        );
    }

    let _ = writeln!(dot, "         </TABLE>>];");
    let _ = writeln!(dot, "}}");

    dot
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn compute_depth(protocol: &Protocol) -> Vec<usize> {
    let n = protocol.vertices.len();
    if n == 0 {
        return vec![];
    }

    let claim_to_vertex: Vec<(ClaimId, usize)> = protocol
        .claims
        .iter()
        .map(|c| (c.id, c.produced_by))
        .collect();

    let mut predecessors: Vec<Vec<usize>> = vec![vec![]; n];
    for (vi, vertex) in protocol.vertices.iter().enumerate() {
        for &consumed in vertex.consumes() {
            if let Some(&(_, producer)) = claim_to_vertex.iter().find(|(id, _)| *id == consumed) {
                if producer != vi && producer < n {
                    predecessors[vi].push(producer);
                }
            }
        }
        if let crate::ir::Vertex::Evaluate { at_vertex, .. } = &vertex {
            if *at_vertex != vi && *at_vertex < n {
                predecessors[vi].push(*at_vertex);
            }
        }
    }
    for preds in &mut predecessors {
        preds.sort_unstable();
        preds.dedup();
    }

    // Kahn's toposort
    let mut successors: Vec<Vec<usize>> = vec![vec![]; n];
    let mut in_degree = vec![0usize; n];
    for (vi, preds) in predecessors.iter().enumerate() {
        in_degree[vi] = preds.len();
        for &pred in preds {
            successors[pred].push(vi);
        }
    }
    let mut queue: Vec<usize> = (0..n).filter(|&v| in_degree[v] == 0).collect();
    let mut topo = Vec::with_capacity(n);
    while let Some(v) = queue.pop() {
        topo.push(v);
        for &succ in &successors[v] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push(succ);
            }
        }
    }

    // DP on topo order
    let mut depth = vec![0usize; n];
    for &v in &topo {
        for &pred in &predecessors[v] {
            let candidate = depth[pred] + 1;
            if candidate > depth[v] {
                depth[v] = candidate;
            }
        }
    }
    depth
}

/// Format an [`Expr`] using polynomial/challenge names with HTML entities.
fn fmt_expr_html(protocol: &Protocol, expr: &Expr) -> String {
    if expr.0.is_empty() {
        return "0".into();
    }

    let mut s = String::new();
    for (i, term) in expr.0.iter().enumerate() {
        let abs = term.coeff.unsigned_abs();

        if i > 0 {
            if term.coeff < 0 {
                s.push_str(" &minus; ");
            } else {
                s.push_str(" + ");
            }
        } else if term.coeff < 0 {
            s.push_str("&minus;");
        }

        if term.factors.is_empty() {
            let _ = write!(s, "{abs}");
        } else {
            if abs != 1 {
                let _ = write!(s, "{abs}&middot;");
            }
            for (j, factor) in term.factors.iter().enumerate() {
                if j > 0 {
                    s.push_str("&middot;");
                }
                match factor {
                    Factor::Poly(idx) => {
                        s.push_str(&html_escape(&protocol.polynomials[*idx].name));
                    }
                    Factor::Challenge(idx) => {
                        s.push_str(&html_escape(&protocol.challenge_names[*idx]));
                    }
                    Factor::Claim(id) => {
                        let _ = write!(s, "c{}", id.0);
                    }
                }
            }
        }
    }
    s
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{PolyKind, PublicPoly};

    #[test]
    fn empty_protocol_produces_valid_dot() {
        let p = Protocol::new();
        let dot = protocol_to_dot(&p);
        assert!(dot.contains("digraph Protocol"));
        assert!(dot.contains('}'));
    }

    #[test]
    fn single_vertex_no_edges() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let _ = p.sumcheck(eq * a, 0, &[d]);
        let dot = protocol_to_dot(&p);
        assert!(dot.contains("v0"));
        assert!(!dot.contains("->"));
    }

    #[test]
    fn chain_produces_edges() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let rho = p.challenge("rho");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let c0 = p.sumcheck(eq * a, 0, &[d]);
        let _ = p.sumcheck(eq * b, rho * c0[0], &[d]);
        let dot = protocol_to_dot(&p);
        assert!(dot.contains("v0 -> v1"));
    }

    #[test]
    fn committed_poly_gets_blue_edge() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let rho = p.challenge("rho");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let w = p.poly("w", &[d], PolyKind::Committed);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let c0 = p.sumcheck(eq * w, 0, &[d]);
        let _ = p.sumcheck(eq * b, rho * c0[0], &[d]);
        let dot = protocol_to_dot(&p);
        assert!(dot.contains(EDGE_COMMITTED));
    }

    #[test]
    fn depth_coloring_varies() {
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
        let dot = protocol_to_dot(&p);
        // v0 depth=0 green, v1 depth=1 blue, v2 depth=2 purple
        assert!(dot.contains(DEPTH_COLORS[0]));
        assert!(dot.contains(DEPTH_COLORS[1]));
        assert!(dot.contains(DEPTH_COLORS[2]));
    }

    #[test]
    fn html_escape_works() {
        assert_eq!(html_escape("a<b>&c"), "a&lt;b&gt;&amp;c");
    }
}
