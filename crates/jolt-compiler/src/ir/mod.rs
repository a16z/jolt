//! L0: Protocol — the mathematical specification of a SNARK.
//!
//! A protocol is a DAG of [`Vertex`] nodes connected by [`Claim`] edges.
//! Polynomials, challenges, and dimensions are declared on the protocol and
//! referenced by index. The protocol builds itself — no separate builder type.

pub mod expr;

use std::fmt;

use serde::{Deserialize, Serialize};

use expr::{Challenge, Expr, Factor, Poly};

/// Unique identifier for a claim. The only newtype ID in the system.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct ClaimId(pub u32);

/// Whether a polynomial is committed (PCS-backed), virtual (derived during
/// proving), or public (deterministic from challenges).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PolyKind {
    Committed,
    Virtual,
    Public(PublicPoly),
}

/// Public polynomials known to the protocol. Materialized from challenge
/// values by the kernel compiler — no witness data needed.
///
/// Point-parameterized variants take `Option<ClaimId>`:
/// - `None` — anchored at the containing sumcheck's own output point
/// - `Some(c)` — anchored at the point where claim `c` was produced
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PublicPoly {
    /// Equality polynomial: `eq(r, x) = Π (rᵢxᵢ + (1-rᵢ)(1-xᵢ))`.
    Eq(Option<ClaimId>),
    /// Successor polynomial: evaluates to 1 when `x = r + 1`.
    EqPlusOne(Option<ClaimId>),
    /// Less-than polynomial: `lt(r, x) = 1 iff x < r` (as integers).
    Lt(Option<ClaimId>),
    /// Identity polynomial: `id(x) = Σ xᵢ·2^(n-1-i)`. No point parameter.
    Identity,
    /// Preprocessed polynomial: determined at preprocessing time from
    /// program-specific data (bytecode, initial memory, instruction set).
    /// Known to both prover and verifier.
    Preprocessed,
}

/// A polynomial declared in the protocol.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PolyDef {
    pub name: String,
    pub dims: Vec<usize>,
    pub kind: PolyKind,
}

/// A claim: "polynomial P evaluates to some value at a point determined
/// by the producing vertex."
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Claim {
    pub id: ClaimId,
    pub poly: usize,
    pub produced_by: usize,
}

/// A vertex in the protocol DAG — an atomic proof step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Vertex {
    Sumcheck {
        composition: Expr,
        /// The claimed sum. Empty `Expr` = zero-check. May reference
        /// upstream claim evals via `Factor::Claim`.
        input_sum: Expr,
        produces: Vec<ClaimId>,
        /// Upstream claims consumed — derived from `Claim` factors in `input_sum`.
        consumes: Vec<ClaimId>,
        binding_order: Vec<usize>,
        /// When `Some(d)`, the first round uses an extended evaluation
        /// domain `{0, 1, …, d−1}` instead of `{0, 1}` (uni-skip).
        domain_size: Option<usize>,
    },
    /// Evaluate polynomial at the binding point of another vertex.
    /// Produces exactly one claim.
    Evaluate {
        poly: usize,
        at_vertex: usize,
        claim: ClaimId,
    },
}

impl Vertex {
    pub fn produces(&self) -> &[ClaimId] {
        match self {
            Vertex::Sumcheck { produces, .. } => produces,
            Vertex::Evaluate { claim, .. } => std::slice::from_ref(claim),
        }
    }

    pub fn consumes(&self) -> &[ClaimId] {
        match self {
            Vertex::Sumcheck { consumes, .. } => consumes,
            Vertex::Evaluate { .. } => &[],
        }
    }

    pub fn binding_order(&self) -> Option<&[usize]> {
        match self {
            Vertex::Sumcheck { binding_order, .. } => Some(binding_order),
            Vertex::Evaluate { .. } => None,
        }
    }

    pub fn composition(&self) -> Option<&Expr> {
        match self {
            Vertex::Sumcheck { composition, .. } => Some(composition),
            Vertex::Evaluate { .. } => None,
        }
    }

    pub fn input_sum(&self) -> Option<&Expr> {
        match self {
            Vertex::Sumcheck { input_sum, .. } => Some(input_sum),
            Vertex::Evaluate { .. } => None,
        }
    }

    pub fn domain_size(&self) -> Option<usize> {
        match self {
            Vertex::Sumcheck { domain_size, .. } => *domain_size,
            Vertex::Evaluate { .. } => None,
        }
    }
}

/// A SNARK protocol: DAG of vertices connected by claims.
///
/// Pure mathematical specification — no scheduling information.
/// The compiler transforms this into a [`Module`](crate::module::Module).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Protocol {
    pub dim_names: Vec<String>,
    pub challenge_names: Vec<String>,
    pub polynomials: Vec<PolyDef>,
    pub claims: Vec<Claim>,
    pub vertices: Vec<Vertex>,
    pub(crate) next_claim: u32,
}

impl Protocol {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dim(&mut self, name: &str) -> usize {
        let idx = self.dim_names.len();
        self.dim_names.push(name.into());
        idx
    }

    pub fn challenge(&mut self, name: &str) -> Challenge {
        let idx = self.challenge_names.len();
        self.challenge_names.push(name.into());
        Challenge(idx)
    }

    pub fn poly(&mut self, name: &str, dims: &[usize], kind: PolyKind) -> Poly {
        let idx = self.polynomials.len();
        self.polynomials.push(PolyDef {
            name: name.into(),
            dims: dims.to_vec(),
            kind,
        });
        Poly(idx)
    }

    /// Add a sumcheck vertex. `consumes` is derived from `Claim` factors
    /// in `input_sum`. Returns one claim per non-public poly in the composition.
    pub fn sumcheck(
        &mut self,
        composition: Expr,
        input_sum: impl Into<Expr>,
        binding_order: &[usize],
    ) -> Vec<ClaimId> {
        let input_sum = input_sum.into();
        let vertex_idx = self.vertices.len();

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

        let witness_polys: Vec<usize> = poly_indices
            .into_iter()
            .filter(|&idx| !matches!(self.polynomials[idx].kind, PolyKind::Public(_)))
            .collect();

        let mut produced = Vec::with_capacity(witness_polys.len());
        for &poly_idx in &witness_polys {
            let id = ClaimId(self.next_claim);
            self.next_claim += 1;
            self.claims.push(Claim {
                id,
                poly: poly_idx,
                produced_by: vertex_idx,
            });
            produced.push(id);
        }

        let consumes = input_sum.claim_deps();

        self.vertices.push(Vertex::Sumcheck {
            composition,
            input_sum,
            produces: produced.clone(),
            consumes,
            binding_order: binding_order.to_vec(),
            domain_size: None,
        });

        produced
    }

    /// Add a uni-skip first round. Splits a sumcheck into two steps:
    /// this round handles the first variable over an extended domain,
    /// producing an intermediate claim. Pass that claim as the continuation
    /// sumcheck's `input_sum`.
    ///
    /// ```ignore
    /// let mid = p.uniskip_round(eq * a * b, 0, &[log_T], 5);
    /// let claims = p.sumcheck(eq * a * b, mid, &[log_T]);
    /// ```
    pub fn uniskip_round(
        &mut self,
        composition: Expr,
        input_sum: impl Into<Expr>,
        binding_order: &[usize],
        domain_size: usize,
    ) -> ClaimId {
        let input_sum = input_sum.into();
        let vertex_idx = self.vertices.len();

        // Synthetic virtual polynomial for the intermediate claim.
        let intermediate_poly = self.poly(
            &format!("_uniskip_mid_v{vertex_idx}"),
            binding_order,
            PolyKind::Virtual,
        );

        let id = ClaimId(self.next_claim);
        self.next_claim += 1;
        self.claims.push(Claim {
            id,
            poly: intermediate_poly.0,
            produced_by: vertex_idx,
        });

        let consumes = input_sum.claim_deps();

        self.vertices.push(Vertex::Sumcheck {
            composition,
            input_sum,
            produces: vec![id],
            consumes,
            binding_order: binding_order.to_vec(),
            domain_size: Some(domain_size),
        });

        id
    }

    /// Evaluate `poly` at the binding point of the vertex that produced `at`.
    /// Returns a single claim for that evaluation.
    pub fn evaluate(&mut self, poly: Poly, at: ClaimId) -> ClaimId {
        let at_vertex = self.claims[at.0 as usize].produced_by;
        let vertex_idx = self.vertices.len();
        let id = ClaimId(self.next_claim);
        self.next_claim += 1;
        self.claims.push(Claim {
            id,
            poly: poly.0,
            produced_by: vertex_idx,
        });
        self.vertices.push(Vertex::Evaluate {
            poly: poly.0,
            at_vertex,
            claim: id,
        });
        id
    }
}

// --- Display ---

impl fmt::Display for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sumchecks = self
            .vertices
            .iter()
            .filter(|v| matches!(v, Vertex::Sumcheck { .. }))
            .count();
        let evals = self
            .vertices
            .iter()
            .filter(|v| matches!(v, Vertex::Evaluate { .. }))
            .count();
        writeln!(
            f,
            "Protocol ({} dims, {} challenges, {} polys, {} sumchecks, {} evals)",
            self.dim_names.len(),
            self.challenge_names.len(),
            self.polynomials.len(),
            sumchecks,
            evals,
        )?;

        for (i, v) in self.vertices.iter().enumerate() {
            match v {
                Vertex::Sumcheck {
                    composition,
                    input_sum,
                    produces,
                    consumes,
                    binding_order,
                    domain_size,
                } => {
                    let dims: Vec<&str> = binding_order
                        .iter()
                        .map(|&d| self.dim_names[d].as_str())
                        .collect();
                    if let Some(d) = domain_size {
                        write!(f, "\n  v{i} [uniskip D={d}, {}]: Σ ", dims.join(", "))?;
                    } else {
                        write!(f, "\n  v{i} [{}]: Σ ", dims.join(", "))?;
                    }
                    self.fmt_expr(composition, f)?;
                    write!(f, " = ")?;
                    self.fmt_expr(input_sum, f)?;

                    if !produces.is_empty() {
                        write!(f, "\n    -> [")?;
                        for (j, cid) in produces.iter().enumerate() {
                            if j > 0 {
                                write!(f, ", ")?;
                            }
                            let poly_idx = self.claims[cid.0 as usize].poly;
                            write!(f, "c{}: {}", cid.0, self.polynomials[poly_idx].name)?;
                        }
                        write!(f, "]")?;
                    }

                    if !consumes.is_empty() {
                        write!(f, "\n    <- [")?;
                        for (j, cid) in consumes.iter().enumerate() {
                            if j > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "c{}", cid.0)?;
                        }
                        write!(f, "]")?;
                    }
                }
                Vertex::Evaluate {
                    poly,
                    at_vertex,
                    claim,
                } => {
                    write!(
                        f,
                        "\n  v{i} [eval]: {} @ v{at_vertex}",
                        self.polynomials[*poly].name
                    )?;
                    write!(
                        f,
                        "\n    -> [c{}: {}]",
                        claim.0, self.polynomials[*poly].name
                    )?;
                }
            }
        }
        writeln!(f)?;
        Ok(())
    }
}

impl Protocol {
    fn fmt_factor(&self, factor: &Factor, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match factor {
            Factor::Poly(i) => write!(f, "{}", self.polynomials[*i].name),
            Factor::Challenge(i) => write!(f, "{}", self.challenge_names[*i]),
            Factor::Claim(id) => write!(f, "c{}", id.0),
        }
    }

    fn fmt_expr(&self, e: &Expr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if e.0.is_empty() {
            return write!(f, "0");
        }
        for (i, term) in e.0.iter().enumerate() {
            let abs = term.coeff.unsigned_abs();

            if i > 0 {
                if term.coeff < 0 {
                    write!(f, " - ")?;
                } else {
                    write!(f, " + ")?;
                }
            } else if term.coeff < 0 {
                write!(f, "-")?;
            }

            if term.factors.is_empty() {
                write!(f, "{abs}")?;
            } else {
                if abs != 1 {
                    write!(f, "{abs} · ")?;
                }
                for (j, factor) in term.factors.iter().enumerate() {
                    if j > 0 {
                        write!(f, " · ")?;
                    }
                    self.fmt_factor(factor, f)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn build_minimal_protocol() {
        let mut p = Protocol::new();
        let log_T = p.dim("log_T");
        let eq = p.poly("eq", &[log_T], PolyKind::Public(PublicPoly::Eq(None)));
        let az = p.poly("Az", &[log_T], PolyKind::Virtual);
        let bz = p.poly("Bz", &[log_T], PolyKind::Virtual);
        let cz = p.poly("Cz", &[log_T], PolyKind::Virtual);

        let claims = p.sumcheck(eq * (az * bz - cz), 0, &[log_T]);
        assert_eq!(claims.len(), 3);
        assert!(p.claims.iter().all(|c| c.poly != 0));
    }

    #[test]
    fn claim_ids_are_sequential() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);

        let c1 = p.sumcheck(a + b, 0, &[d]);
        let c2 = p.sumcheck(a * b, 0, &[d]);

        assert_eq!(c1[0].0, 0);
        assert_eq!(c1[1].0, 1);
        assert_eq!(c2[0].0, 2);
        assert_eq!(c2[1].0, 3);
    }

    #[test]
    fn committed_poly_produces_claims() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let w = p.poly("w", &[d], PolyKind::Committed);

        let claims = p.sumcheck(eq * w, 0, &[d]);
        assert_eq!(claims.len(), 1);
        assert_eq!(p.claims[claims[0].0 as usize].poly, w.0);
    }

    #[test]
    fn derived_input_sum() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let rho = p.challenge("rho");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);

        let upstream = p.sumcheck(eq * (a * b), 0, &[d]);

        let c = p.poly("c", &[d], PolyKind::Virtual);
        let downstream = p.sumcheck(eq * c, rho * upstream[0] + upstream[1], &[d]);
        assert_eq!(downstream.len(), 1);

        let last = &p.vertices[p.vertices.len() - 1];
        assert_eq!(last.consumes().len(), 2);
        assert!(last.consumes().contains(&upstream[0]));
        assert!(last.consumes().contains(&upstream[1]));
    }

    #[test]
    fn evaluate_produces_claim() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);

        let upstream = p.sumcheck(eq * a, 0, &[d]);
        let eval_claim = p.evaluate(b, upstream[0]);

        assert_eq!(p.claims[eval_claim.0 as usize].poly, b.0);
        assert!(matches!(
            p.vertices.last().unwrap(),
            Vertex::Evaluate { .. }
        ));
    }

    #[test]
    fn evaluate_feeds_downstream() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let rho = p.challenge("rho");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let c = p.poly("c", &[d], PolyKind::Virtual);

        let upstream = p.sumcheck(eq * a, 0, &[d]);
        let b_eval = p.evaluate(b, upstream[0]);

        let downstream = p.sumcheck(eq * c, rho * b_eval, &[d]);
        assert_eq!(downstream.len(), 1);
        assert!(p.vertices.last().unwrap().consumes().contains(&b_eval));
    }

    #[test]
    fn serde_roundtrip_json() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let _ = p.sumcheck(eq * a, 0, &[d]);

        let json = serde_json::to_string_pretty(&p).unwrap();
        let p2: Protocol = serde_json::from_str(&json).unwrap();
        assert_eq!(p2.polynomials, p.polynomials);
        assert_eq!(p2.vertices.len(), p.vertices.len());
        assert_eq!(p2.claims.len(), p.claims.len());
    }

    #[test]
    fn serde_roundtrip_ron() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let _ = p.sumcheck(eq * (a * b), 0, &[d]);

        let ron_str = ron::to_string(&p).unwrap();
        let p2: Protocol = ron::from_str(&ron_str).unwrap();
        assert_eq!(p2.polynomials, p.polynomials);
        assert_eq!(p2.vertices.len(), 1);
    }

    #[test]
    fn protocol_display_sumcheck() {
        let mut p = Protocol::new();
        let d = p.dim("log_T");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let az = p.poly("Az", &[d], PolyKind::Virtual);
        let bz = p.poly("Bz", &[d], PolyKind::Virtual);
        let cz = p.poly("Cz", &[d], PolyKind::Virtual);
        let _ = p.sumcheck(eq * (az * bz - cz), 0, &[d]);
        let s = p.to_string();
        assert!(s.contains("4 polys"));
        assert!(s.contains("sumchecks"));
        assert!(s.contains("eq · Az · Bz - eq · Cz = 0"));
        assert!(s.contains("c0: Az"));
    }

    #[test]
    fn protocol_display_evaluate() {
        let mut p = Protocol::new();
        let d = p.dim("log_T");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let upstream = p.sumcheck(eq * a, 0, &[d]);
        let _ = p.evaluate(b, upstream[0]);
        let s = p.to_string();
        assert!(s.contains("sumchecks"));
        assert!(s.contains("evals"));
        assert!(s.contains("[eval]: b @ v0"));
    }
}
