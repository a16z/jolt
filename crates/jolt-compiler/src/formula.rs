//! Kernel-level composition formula: normalized sum-of-products.
//!
//! [`CompositionFormula`] is the canonical form consumed by compute backends
//! for kernel compilation. Produced by the compiler from protocol-level
//! [`Expr`](crate::ir::expr::Expr).

use serde::{Deserialize, Serialize};

/// Variable reference in a composition formula term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Factor {
    /// Reference to an input polynomial buffer by index.
    Input(u32),
    /// Reference to a challenge slot, resolved at runtime.
    Challenge(u32),
}

/// A single term: `coefficient × factor₀ × factor₁ × …`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductTerm {
    pub coefficient: i128,
    pub factors: Vec<Factor>,
}

impl ProductTerm {
    /// Number of [`Factor::Input`] references.
    #[inline]
    pub fn input_degree(&self) -> usize {
        self.factors
            .iter()
            .filter(|f| matches!(f, Factor::Input(_)))
            .count()
    }
}

/// Normalized sum-of-products representation of a composition polynomial.
///
/// `Σᵢ coeffᵢ × ∏ⱼ factorᵢⱼ`
///
/// This is the canonical form consumed by compute backends for kernel
/// compilation. Field-dependent methods (evaluate, weight extraction) are
/// provided by the backends, not here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionFormula {
    pub terms: Vec<ProductTerm>,
    /// Number of distinct input polynomial slots.
    pub num_inputs: usize,
    /// Number of distinct challenge slots.
    pub num_challenges: usize,
}

impl CompositionFormula {
    /// Build from terms, computing `num_inputs` and `num_challenges`.
    pub fn from_terms(terms: Vec<ProductTerm>) -> Self {
        let mut max_input: Option<u32> = None;
        let mut max_challenge: Option<u32> = None;
        for term in &terms {
            for factor in &term.factors {
                match factor {
                    Factor::Input(id) => {
                        max_input = Some(max_input.map_or(*id, |m: u32| m.max(*id)));
                    }
                    Factor::Challenge(id) => {
                        max_challenge = Some(max_challenge.map_or(*id, |m: u32| m.max(*id)));
                    }
                }
            }
        }
        Self {
            terms,
            num_inputs: max_input.map_or(0, |m| m as usize + 1),
            num_challenges: max_challenge.map_or(0, |m| m as usize + 1),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Maximum input-variable degree across all terms.
    pub fn degree(&self) -> usize {
        self.terms
            .iter()
            .map(|t| t.input_degree())
            .max()
            .unwrap_or(0)
    }

    /// Detect a pure product-sum structure: every term is a product of exactly
    /// `d` input factors with coefficient 1 and no challenge factors.
    pub fn as_product_sum(&self) -> Option<(usize, usize)> {
        if self.terms.is_empty() {
            return None;
        }
        let d = self.terms[0].factors.len();
        if d == 0 {
            return None;
        }
        for term in &self.terms {
            if term.coefficient != 1 || term.factors.len() != d {
                return None;
            }
            if term
                .factors
                .iter()
                .any(|f| matches!(f, Factor::Challenge(_)))
            {
                return None;
            }
        }
        Some((d, self.terms.len()))
    }

    /// Every non-zero term has exactly one input factor and at least one
    /// challenge factor. Enables pre-combination optimization.
    pub fn is_linear_combination(&self) -> bool {
        let mut has_nonzero = false;
        for term in &self.terms {
            if term.coefficient == 0 {
                continue;
            }
            has_nonzero = true;
            let n_inputs = term
                .factors
                .iter()
                .filter(|f| matches!(f, Factor::Input(_)))
                .count();
            let n_challenges = term
                .factors
                .iter()
                .filter(|f| matches!(f, Factor::Challenge(_)))
                .count();
            if n_inputs != 1 || n_challenges < 1 {
                return false;
            }
        }
        has_nonzero
    }

    /// `eq(x) · g(x)` pattern: single term `Input(0) × Input(1)`.
    pub fn is_eq_product(&self) -> bool {
        self.terms.len() == 1 && {
            let t = &self.terms[0];
            t.coefficient == 1
                && t.factors.len() == 2
                && t.factors.contains(&Factor::Input(0))
                && t.factors.contains(&Factor::Input(1))
        }
    }

    /// Hamming booleanity pattern: `challenge × input × (input − 1)`.
    pub fn is_hamming_booleanity(&self) -> bool {
        if self.terms.len() != 2 {
            return false;
        }

        let input_factors = |idx: usize| -> Vec<u32> {
            self.terms[idx]
                .factors
                .iter()
                .filter_map(|f| match f {
                    Factor::Input(id) => Some(*id),
                    Factor::Challenge(_) => None,
                })
                .collect()
        };
        let challenge_count = |idx: usize| -> usize {
            self.terms[idx]
                .factors
                .iter()
                .filter(|f| matches!(f, Factor::Challenge(_)))
                .count()
        };

        let i0 = input_factors(0);
        let i1 = input_factors(1);

        let (sq_idx, lin_idx) = if i0.len() == 2 && i1.len() == 1 {
            (0, 1)
        } else if i0.len() == 1 && i1.len() == 2 {
            (1, 0)
        } else {
            return false;
        };

        let sq = input_factors(sq_idx);
        let lin = input_factors(lin_idx);

        sq[0] == sq[1]
            && sq[0] == lin[0]
            && challenge_count(sq_idx) == 1
            && challenge_count(lin_idx) == 1
    }
}

/// Variable binding direction during sumcheck.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BindingOrder {
    #[default]
    LowToHigh,
    HighToLow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_terms_counts() {
        let terms = vec![
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0), Factor::Input(2), Factor::Challenge(1)],
            },
            ProductTerm {
                coefficient: -1,
                factors: vec![Factor::Input(1)],
            },
        ];
        let f = CompositionFormula::from_terms(terms);
        assert_eq!(f.num_inputs, 3);
        assert_eq!(f.num_challenges, 2);
    }

    #[test]
    fn degree_computation() {
        let terms = vec![
            ProductTerm {
                coefficient: 1,
                factors: vec![
                    Factor::Input(0),
                    Factor::Input(0),
                    Factor::Challenge(0),
                ],
            },
            ProductTerm {
                coefficient: -1,
                factors: vec![Factor::Input(0), Factor::Challenge(0)],
            },
        ];
        let f = CompositionFormula::from_terms(terms);
        assert_eq!(f.degree(), 2);
    }

    #[test]
    fn product_sum_detected() {
        let terms = vec![
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0), Factor::Input(1)],
            },
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(2), Factor::Input(3)],
            },
        ];
        let f = CompositionFormula::from_terms(terms);
        assert_eq!(f.as_product_sum(), Some((2, 2)));
    }

    #[test]
    fn product_sum_rejected_with_challenge() {
        let terms = vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Challenge(0)],
        }];
        let f = CompositionFormula::from_terms(terms);
        assert_eq!(f.as_product_sum(), None);
    }

    #[test]
    fn linear_combination_detected() {
        let terms = vec![
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Challenge(0), Factor::Input(0)],
            },
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Challenge(1), Factor::Input(1)],
            },
        ];
        let f = CompositionFormula::from_terms(terms);
        assert!(f.is_linear_combination());
    }

    #[test]
    fn eq_product_detected() {
        let terms = vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1)],
        }];
        let f = CompositionFormula::from_terms(terms);
        assert!(f.is_eq_product());
    }

    #[test]
    fn hamming_booleanity_detected() {
        let terms = vec![
            ProductTerm {
                coefficient: 1,
                factors: vec![
                    Factor::Challenge(0),
                    Factor::Input(0),
                    Factor::Input(0),
                ],
            },
            ProductTerm {
                coefficient: -1,
                factors: vec![Factor::Challenge(0), Factor::Input(0)],
            },
        ];
        let f = CompositionFormula::from_terms(terms);
        assert!(f.is_hamming_booleanity());
    }

    #[test]
    fn empty_formula() {
        let f = CompositionFormula::from_terms(vec![]);
        assert!(f.is_empty());
        assert_eq!(f.degree(), 0);
        assert_eq!(f.num_inputs, 0);
        assert_eq!(f.num_challenges, 0);
    }
}
