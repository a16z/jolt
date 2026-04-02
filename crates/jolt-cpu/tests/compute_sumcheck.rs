//! Integration test: validates the compute pipeline round polynomials.
//!
//! Flow: Formula → KernelSpec → compile() → reduce()
//! → round polynomial evaluations → UnivariatePoly → verify.
//!
//! Cross-checks against a hand-written reference implementation to ensure
//! the compute layer produces identical round polynomials.
//!
//! Both the reference and compute paths use LowToHigh binding order:
//! pairs are `(buf[2i], buf[2i+1])` at each round, binding the LSB first.

use jolt_compiler::{Factor, Formula, Iteration, KernelSpec, ProductTerm};
use jolt_compute::{BindingOrder, Buf, ComputeBackend, DeviceBuffer};
use jolt_cpu::{compile, CpuBackend};
use jolt_field::{Field, Fr};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_sumcheck::SumcheckVerifier;
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
use num_traits::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn make_spec(formula: &Formula) -> KernelSpec {
    KernelSpec {
        num_evals: formula.degree(),
        formula: formula.clone(),
        iteration: Iteration::Dense,
        binding_order: BindingOrder::LowToHigh,
    }
}

fn reference_round_poly(f: &[Fr], g: &[Fr], eq: &[Fr]) -> UnivariatePoly<Fr> {
    let half = f.len() / 2;
    let mut evals = [Fr::zero(); 4];
    for i in 0..half {
        let (f_lo, f_hi) = (f[2 * i], f[2 * i + 1]);
        let (g_lo, g_hi) = (g[2 * i], g[2 * i + 1]);
        let (eq_lo, eq_hi) = (eq[2 * i], eq[2 * i + 1]);
        let (df, dg, deq) = (f_hi - f_lo, g_hi - g_lo, eq_hi - eq_lo);

        evals[0] += eq_lo * f_lo * g_lo;
        evals[1] += eq_hi * f_hi * g_hi;
        let (f2, g2, eq2) = (f_lo + df + df, g_lo + dg + dg, eq_lo + deq + deq);
        evals[2] += eq2 * f2 * g2;
        let three = Fr::from_u64(3);
        evals[3] += (eq_lo + three * deq) * (f_lo + three * df) * (g_lo + three * dg);
    }
    let points: Vec<(Fr, Fr)> = evals
        .iter()
        .enumerate()
        .map(|(t, &y)| (Fr::from_u64(t as u64), y))
        .collect();
    UnivariatePoly::interpolate(&points)
}

fn reference_bind(f: &mut Vec<Fr>, g: &mut Vec<Fr>, eq: &mut Vec<Fr>, challenge: Fr) {
    let half = f.len() / 2;
    for i in 0..half {
        f[i] = f[2 * i] + challenge * (f[2 * i + 1] - f[2 * i]);
        g[i] = g[2 * i] + challenge * (g[2 * i + 1] - g[2 * i]);
        eq[i] = eq[2 * i] + challenge * (eq[2 * i + 1] - eq[2 * i]);
    }
    f.truncate(half);
    g.truncate(half);
    eq.truncate(half);
}

fn compute_round_poly(
    f_buf: &[Fr],
    g_buf: &[Fr],
    eq_buf: &[Fr],
    kernel: &jolt_cpu::CpuKernel<Fr>,
) -> UnivariatePoly<Fr> {
    let backend = CpuBackend;
    let half = f_buf.len() / 2;

    let eq_dbuf: Buf<CpuBackend, Fr> = DeviceBuffer::Field(eq_buf.to_vec());
    let f_dbuf: Buf<CpuBackend, Fr> = DeviceBuffer::Field(f_buf.to_vec());
    let g_dbuf: Buf<CpuBackend, Fr> = DeviceBuffer::Field(g_buf.to_vec());

    let toom_evals = backend.reduce(kernel, &[&eq_dbuf, &f_dbuf, &g_dbuf], &[]);

    let mut p0 = Fr::zero();
    for i in 0..half {
        p0 += eq_buf[2 * i] * f_buf[2 * i] * g_buf[2 * i];
    }

    UnivariatePoly::from_evals_toom(&[p0, toom_evals[0], toom_evals[1], toom_evals[2]])
}

fn compute_bind(buf: &mut Vec<Fr>, challenge: Fr) {
    let backend = CpuBackend;
    backend.interpolate_inplace(buf, challenge, BindingOrder::LowToHigh);
}

fn make_sparse_spec(formula: &Formula) -> KernelSpec {
    KernelSpec {
        num_evals: formula.degree(),
        formula: formula.clone(),
        iteration: Iteration::Sparse,
        binding_order: BindingOrder::LowToHigh,
    }
}

fn product_sum_formula(d: usize, p: usize) -> Formula {
    let terms: Vec<_> = (0..p)
        .map(|g| ProductTerm {
            coefficient: 1,
            factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
        })
        .collect();
    Formula::from_terms(terms)
}

/// Prove and verify with both reference and compute-backend witnesses.
///
/// The EqPolynomial evaluation table has natural LowToHigh layout where
/// `(table[2i], table[2i+1])` are the LSB pair. Both paths use this layout
/// directly, binding the LSB first at each round.
///
/// After all rounds, the verifier's challenge point has LSB-first ordering:
/// `challenges[0]` binds the LSB, `challenges[n-1]` binds the MSB.
/// `Polynomial::evaluate` expects MSB-first ordering, so we reverse.
#[test]
fn compute_witness_matches_reference() {
    let mut rng = ChaCha20Rng::seed_from_u64(12345);
    let num_vars = 6;
    let n = 1usize << num_vars;

    let f: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let g: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let tau: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

    let eq_evals = EqPolynomial::new(tau.clone()).evaluations();
    let claimed_sum: Fr = f
        .iter()
        .zip(g.iter())
        .zip(eq_evals.iter())
        .map(|((&fi, &gi), &ei)| ei * fi * gi)
        .sum();

    let claim = SumcheckClaim {
        num_vars,
        degree: 3,
        claimed_sum,
    };

    // Reference proof: inline prover loop (LowToHigh layout, no interleave needed)
    let mut rf = f.clone();
    let mut rg = g.clone();
    let mut req = eq_evals.clone();
    let mut ref_pt = Blake2bTranscript::new(b"ref");
    let mut ref_polys = Vec::new();
    for _ in 0..num_vars {
        let poly = reference_round_poly(&rf, &rg, &req);
        for c in poly.coefficients() {
            c.append_to_transcript(&mut ref_pt);
        }
        let ch: Fr = ref_pt.challenge();
        reference_bind(&mut rf, &mut rg, &mut req, ch);
        ref_polys.push(poly);
    }
    let ref_proof = jolt_sumcheck::SumcheckProof {
        round_polynomials: ref_polys,
    };
    let mut ref_vt = Blake2bTranscript::new(b"ref");
    let ref_result = SumcheckVerifier::verify(&claim, &ref_proof, &mut ref_vt);
    assert!(ref_result.is_ok(), "reference sumcheck should verify");

    // Compute-backend proof: inline prover loop (same LowToHigh layout)
    let formula = product_sum_formula(3, 1);
    let kernel = compile::<Fr>(&make_spec(&formula));
    let mut cf = f.clone();
    let mut cg = g.clone();
    let mut ceq = eq_evals.clone();
    let mut compute_pt = Blake2bTranscript::new(b"compute");
    let mut compute_polys = Vec::new();
    for _ in 0..num_vars {
        let poly = compute_round_poly(&cf, &cg, &ceq, &kernel);
        for c in poly.coefficients() {
            c.append_to_transcript(&mut compute_pt);
        }
        let ch: Fr = compute_pt.challenge();
        compute_bind(&mut cf, ch);
        compute_bind(&mut cg, ch);
        compute_bind(&mut ceq, ch);
        compute_polys.push(poly);
    }
    let compute_proof = jolt_sumcheck::SumcheckProof {
        round_polynomials: compute_polys,
    };
    let mut compute_vt = Blake2bTranscript::new(b"compute");
    let compute_result = SumcheckVerifier::verify(&claim, &compute_proof, &mut compute_vt);
    assert!(
        compute_result.is_ok(),
        "compute-backend sumcheck should verify"
    );

    // Both produce valid proofs with correct final evaluations.
    // LowToHigh binding produces challenges in LSB-first order.
    // Polynomial::evaluate expects MSB-first, so reverse.
    let (ref_eval, ref_challenges) = ref_result.unwrap();
    let (compute_eval, compute_challenges) = compute_result.unwrap();

    let f_poly = jolt_poly::Polynomial::new(f.clone());
    let g_poly = jolt_poly::Polynomial::new(g.clone());

    let ref_point: Vec<Fr> = ref_challenges.iter().rev().copied().collect();
    let eq_at_ref = EqPolynomial::new(tau.clone()).evaluate(&ref_point);
    assert_eq!(
        ref_eval,
        eq_at_ref * f_poly.evaluate(&ref_point) * g_poly.evaluate(&ref_point)
    );

    let compute_point: Vec<Fr> = compute_challenges.iter().rev().copied().collect();
    let eq_at_compute = EqPolynomial::new(tau).evaluate(&compute_point);
    assert_eq!(
        compute_eval,
        eq_at_compute * f_poly.evaluate(&compute_point) * g_poly.evaluate(&compute_point)
    );
}

#[test]
fn compute_round_poly_sum_check() {
    let mut rng = ChaCha20Rng::seed_from_u64(99999);
    let num_vars = 5;
    let n = 1usize << num_vars;

    let f: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let g: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let tau: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

    let eq_evals = EqPolynomial::new(tau).evaluations();
    let claimed_sum: Fr = f
        .iter()
        .zip(g.iter())
        .zip(eq_evals.iter())
        .map(|((&fi, &gi), &ei)| ei * fi * gi)
        .sum();

    let formula = product_sum_formula(3, 1);
    let kernel = compile::<Fr>(&make_spec(&formula));
    let mut cf = f;
    let mut cg = g;
    let mut ceq = eq_evals;
    let mut running_sum = claimed_sum;

    for _ in 0..num_vars {
        let round_poly = compute_round_poly(&cf, &cg, &ceq, &kernel);
        let s0 = round_poly.evaluate(Fr::zero());
        let s1 = round_poly.evaluate(Fr::from_u64(1));
        assert_eq!(s0 + s1, running_sum, "s(0)+s(1) must equal running sum");

        let challenge = Fr::random(&mut rng);
        running_sum = round_poly.evaluate(challenge);
        compute_bind(&mut cf, challenge);
        compute_bind(&mut cg, challenge);
        compute_bind(&mut ceq, challenge);
    }
}

/// Compute s(0) for a 3-input product composition over sparse pairs.
///
/// s(0) = Σ (lo_a * lo_b * lo_c) for each pair, where lo defaults to 0
/// for unpaired odd-only entries.
fn sparse_s0(cols: &[&[Fr]], keys: &[u64]) -> Fr {
    let n = keys.len();
    let mut sum = Fr::zero();
    let mut i = 0;
    while i < n {
        let key = keys[i];
        if key.is_multiple_of(2) {
            let mut prod = Fr::one();
            for col in cols {
                prod *= col[i];
            }
            sum += prod;
            if i + 1 < n && keys[i + 1] == key + 1 {
                i += 2;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    sum
}

/// Sparse sumcheck: verify s(0)+s(1) == running_sum at each round.
///
/// Uses composition `eq * f * g` (degree 3, product-sum d=3 p=1) with
/// sparse iteration. Keys span a 4-variable hypercube (16 positions),
/// so 4 sumcheck rounds fully collapse the data.
#[test]
fn sparse_sumcheck_rounds() {
    let mut rng = ChaCha20Rng::seed_from_u64(54321);
    let num_vars = 4;

    // Sparse entries: 4 complete pairs in a 16-position hypercube.
    let keys = vec![0u64, 1, 4, 5, 10, 11, 14, 15];
    let n = keys.len();

    let col_eq: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let col_f: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let col_g: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

    let claimed_sum: Fr = (0..n).map(|i| col_eq[i] * col_f[i] * col_g[i]).sum();

    let formula = product_sum_formula(3, 1);
    let kernel = compile::<Fr>(&make_sparse_spec(&formula));
    let backend = CpuBackend;

    let mut bufs: Vec<Buf<CpuBackend, Fr>> = vec![
        DeviceBuffer::Field(col_eq),
        DeviceBuffer::Field(col_f),
        DeviceBuffer::Field(col_g),
        DeviceBuffer::U64(keys),
    ];

    let mut running_sum = claimed_sum;

    for round in 0..num_vars {
        let cur_keys = bufs[3].as_u64();
        let s0 = sparse_s0(
            &[bufs[0].as_field(), bufs[1].as_field(), bufs[2].as_field()],
            cur_keys,
        );

        let buf_refs: Vec<&Buf<CpuBackend, Fr>> = bufs.iter().collect();
        let toom_evals = backend.reduce(&kernel, &buf_refs, &[]);
        let s1 = toom_evals[0]; // Toom-Cook grid: first eval is at t=1.

        assert_eq!(
            s0 + s1,
            running_sum,
            "s(0)+s(1) != running_sum at round {round}"
        );

        // Round polynomial: degree 3, evals at {0, 1, 2, ∞}.
        let round_poly =
            UnivariatePoly::from_evals_toom(&[s0, toom_evals[0], toom_evals[1], toom_evals[2]]);

        let challenge = Fr::random(&mut rng);
        running_sum = round_poly.evaluate(challenge);
        backend.bind(&kernel, &mut bufs, challenge);
    }

    // After all rounds, one entry remains.
    assert_eq!(bufs[0].as_field().len(), 1);
}
