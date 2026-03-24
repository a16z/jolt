//! Witness builders for sumcheck vertices.
//!
//! Each builder takes polynomial tables + evaluation cache + backend and
//! produces a `Box<dyn SumcheckCompute>`. These are dispatched by vertex
//! kind when the prover walks the protocol graph.
//!
//! The builders are the **execution layer** — they decide how to compute
//! each sumcheck efficiently. The protocol graph says what to prove; these
//! say how.

use std::sync::Arc;

use jolt_compute::ComputeBackend;
use jolt_field::Field;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::SumcheckCompute;

use crate::evaluators::catalog::{self, Term};
use crate::evaluators::kernel::KernelEvaluator;

/// Evaluates a multilinear polynomial at a point.
pub fn eval_poly<F: Field>(table: &[F], point: &[F]) -> F {
    jolt_poly::Polynomial::new(table.to_vec()).evaluate(point)
}

/// Builds a `SumcheckCompute` for `Σ w(x) · g(x)` where `w` is the weighting
/// polynomial and `g` is a sum-of-products formula over `polys`.
///
/// `formula_descriptor` wraps with `opening(0) · Σ_k c_k · Π opening(factor+1)`,
/// so `w` becomes `inputs[0]` and term factor indices are 0-based into `polys`.
pub fn formula_witness<F: Field, B: ComputeBackend>(
    w: &[F],
    polys: &[&[F]],
    terms: &[Term<F>],
    degree: usize,
    backend: &Arc<B>,
) -> Box<dyn SumcheckCompute<F>> {
    let (desc, challenges) = catalog::formula_descriptor(terms, polys.len(), degree);
    let kernel = backend.compile_kernel_with_challenges::<F>(&desc, &challenges);

    let mut inputs = vec![backend.upload(w)];
    inputs.extend(polys.iter().map(|t| backend.upload(t)));

    Box::new(KernelEvaluator::with_unit_weights(
        inputs,
        kernel,
        desc.degree + 1,
        Arc::clone(backend),
    ))
}

/// Builds a claim reduction witness: `Σ eq(r, x) · Σ_i c_i · p_i(x)`.
///
/// This is the most common pattern — a γ-weighted linear combination of
/// polynomials under an eq polynomial. Used by RegistersCR, InstrLookupsCR,
/// RamRaCR, IncCR, HammingWeightCR.
pub fn claim_reduction_witness<F: Field, B: ComputeBackend>(
    eq_point: &[F],
    polys: &[&[F]],
    coefficients: &[F],
    backend: &Arc<B>,
) -> Box<dyn SumcheckCompute<F>> {
    let eq = EqPolynomial::new(eq_point.to_vec()).evaluations();
    let terms: Vec<Term<F>> = coefficients
        .iter()
        .enumerate()
        .map(|(i, &c)| Term {
            coeff: c,
            factors: vec![i],
        })
        .collect();
    formula_witness(&eq, polys, &terms, 3, backend)
}

/// Builds a booleanity witness: `Σ eq(r, x) · (h(x)² − h(x))`.
///
/// Zero-check — the claimed sum is always zero.
pub fn booleanity_witness<F: Field, B: ComputeBackend>(
    eq_point: &[F],
    h_poly: &[F],
    backend: &Arc<B>,
) -> Box<dyn SumcheckCompute<F>> {
    let eq = EqPolynomial::new(eq_point.to_vec()).evaluations();
    let terms = vec![
        Term {
            coeff: F::one(),
            factors: vec![0, 0],
        },
        Term {
            coeff: -F::one(),
            factors: vec![0],
        },
    ];
    formula_witness(&eq, &[h_poly], &terms, 4, backend)
}

/// Builds a product virtual witness: `Σ eq(r, x) · Σ_k c_k · Π factors_k(x)`.
///
/// Used by ProductVirtual (S2). Each term is a product of factor polynomials
/// weighted by γ-powers.
pub fn product_virtual_witness<F: Field, B: ComputeBackend>(
    eq_point: &[F],
    factor_polys: &[&[F]],
    terms: &[Term<F>],
    degree: usize,
    backend: &Arc<B>,
) -> Box<dyn SumcheckCompute<F>> {
    let eq = EqPolynomial::new(eq_point.to_vec()).evaluations();
    formula_witness(&eq, factor_polys, terms, degree, backend)
}

/// Builds a shift witness: `Σ eq+1(r, x) · Σ_i γ^i · p_i(x)`.
///
/// Uses EqPlusOne instead of standard eq. The combined eq+1 table is
/// precomputed from outer and product points.
pub fn shift_witness<F: Field, B: ComputeBackend>(
    eq_plus_one_combined: &[F],
    shift_polys: &[&[F]],
    gamma_powers: &[F],
    backend: &Arc<B>,
) -> Box<dyn SumcheckCompute<F>> {
    let terms: Vec<Term<F>> = gamma_powers
        .iter()
        .enumerate()
        .map(|(i, &c)| Term {
            coeff: c,
            factors: vec![i],
        })
        .collect();
    formula_witness(eq_plus_one_combined, shift_polys, &terms, 3, backend)
}

/// Builds an instruction input witness:
/// `Σ eq(r, x) · (Σ_k c_k · flag_k(x) · value_k(x))`.
///
/// Each term is a flag×value product: is_rs2·rs2_v, is_imm·imm, etc.
pub fn instruction_input_witness<F: Field, B: ComputeBackend>(
    eq_point: &[F],
    flag_value_pairs: &[(&[F], &[F])],
    coefficients: &[F],
    backend: &Arc<B>,
) -> Box<dyn SumcheckCompute<F>> {
    let eq = EqPolynomial::new(eq_point.to_vec()).evaluations();
    let mut polys = Vec::new();
    let mut terms = Vec::new();
    for (i, (&(flag, value), &coeff)) in flag_value_pairs.iter().zip(coefficients).enumerate() {
        let base = i * 2;
        polys.push(flag);
        polys.push(value);
        terms.push(Term {
            coeff,
            factors: vec![base, base + 1],
        });
    }
    let poly_refs: Vec<&[F]> = polys;
    formula_witness(&eq, &poly_refs, &terms, 4, backend)
}

/// Builds a weighted product witness: `Σ w(x) · Π_i p_i(x)`.
///
/// Used for RamValCheck (w = eq·(LT+γ), polys = [inc, addr]) and
/// RegistersValEval (w = LT, polys = [inc, wa]).
pub fn weighted_product_witness<F: Field, B: ComputeBackend>(
    w: &[F],
    polys: &[&[F]],
    backend: &Arc<B>,
) -> Box<dyn SumcheckCompute<F>> {
    let factors: Vec<usize> = (0..polys.len()).collect();
    let terms = vec![Term {
        coeff: F::one(),
        factors,
    }];
    formula_witness(w, polys, &terms, polys.len() + 2, backend)
}

/// Builds a registers RW witness:
/// `Σ eq(r, x) · (wa·inc + wa·val + γ·ra1·val + γ²·ra2·val)`.
#[allow(clippy::too_many_arguments)]
pub fn registers_rw_witness<F: Field, B: ComputeBackend>(
    eq_point: &[F],
    rd_wa: &[F],
    rd_inc: &[F],
    val: &[F],
    rs1_ra: &[F],
    rs2_ra: &[F],
    gamma: F,
    backend: &Arc<B>,
) -> Box<dyn SumcheckCompute<F>> {
    let eq = EqPolynomial::new(eq_point.to_vec()).evaluations();
    let g2 = gamma * gamma;
    let terms = vec![
        Term {
            coeff: F::one(),
            factors: vec![0, 1],
        },
        Term {
            coeff: F::one(),
            factors: vec![0, 2],
        },
        Term {
            coeff: gamma,
            factors: vec![3, 2],
        },
        Term {
            coeff: g2,
            factors: vec![4, 2],
        },
    ];
    formula_witness(
        &eq,
        &[rd_wa, rd_inc, val, rs1_ra, rs2_ra],
        &terms,
        4,
        backend,
    )
}
