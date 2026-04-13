//! Booleanity sumcheck state machine for the CPU backend.
//!
//! Replicates jolt-core's two-phase Gruen-based booleanity sumcheck:
//! - Phase 1 (address, log_k_chunk rounds): G_d projections + Gruen polynomial
//! - Phase 2 (cycle, log_t rounds): H_d polynomials + Gruen polynomial with eq_r_r scaling

#![allow(non_snake_case)]

use jolt_field::Field;
use jolt_poly::EqPolynomial;

/// Opaque booleanity state for one sumcheck instance.
///
/// Mirrors `BooleanitySumcheckProver` in jolt-core but uses flat iteration
/// instead of `GruenSplitEqPolynomial::par_fold_out_in_unreduced`.
pub struct CpuBooleanityState<F: Field> {
    /// G_d\[k\] = Σ_j eq(r_cycle, j) · ra_d(k, j).  One Vec per dimension.
    G: Vec<Vec<F>>,
    /// Expanding eq table for bound address variables (matches core's ExpandingTable).
    /// Layout: first half = (1−r) scaled, second half = r scaled.
    F_table: Vec<F>,
    /// RA data per dimension, AddressMajor: ra_d\[k * T + j\].
    /// Kept until Phase 2 transition to construct H.
    ra_data: Vec<Vec<F>>,

    addr_w: Vec<F>,
    addr_current_scalar: F,
    addr_current_index: usize,

    cycle_w: Vec<F>,
    cycle_current_scalar: F,
    cycle_current_index: usize,

    gamma_powers: Vec<F>,
    gamma_powers_square: Vec<F>,

    /// H_d\[j\] pre-scaled by γ^d.  Bound in-place each Phase 2 round.
    H: Option<Vec<Vec<F>>>,
    eq_r_r: F,

    /// Eq table over the "rest" (unbound, non-current) variables.
    /// Contracted (sum-of-pairs) after each round.
    eq_rest: Vec<F>,

    log_k_chunk: usize,
    log_t: usize,
    num_polys: usize,
    current_round: usize,
}

impl<F: Field> CpuBooleanityState<F> {
    /// Build the booleanity state from raw RA data and challenge values.
    ///
    /// * `ra_data` – per-dimension RA polynomial in AddressMajor layout.
    /// * `addr_ch` – r_address challenges (LE, length `log_k_chunk`).
    /// * `cycle_ch` – r_cycle challenges (LE, length `log_t`).
    pub fn new(
        ra_data: Vec<Vec<F>>,
        addr_ch: &[F],
        cycle_ch: &[F],
        gamma_powers: Vec<F>,
        gamma_powers_square: Vec<F>,
        log_k_chunk: usize,
        log_t: usize,
    ) -> Self {
        let k_chunk = 1usize << log_k_chunk;
        let t = 1usize << log_t;
        let num_polys = ra_data.len();

        // eq(r_cycle, ·) for G_d projection
        let cycle_eq: Vec<F> = EqPolynomial::<F>::evals(cycle_ch, None);

        // G_d[k] = Σ_j eq_cycle[j] · ra_d[k·T + j]
        let G: Vec<Vec<F>> = ra_data
            .iter()
            .map(|ra_d| {
                (0..k_chunk)
                    .map(|k| {
                        let base = k * t;
                        let mut acc = F::zero();
                        for j in 0..t {
                            acc += cycle_eq[j] * ra_d[base + j];
                        }
                        acc
                    })
                    .collect()
            })
            .collect();

        // eq_rest for Phase 1 round 0: over addr_w[0 .. log_k_chunk-2]
        let eq_rest = if log_k_chunk > 1 {
            EqPolynomial::<F>::evals(&addr_ch[..log_k_chunk - 1], None)
        } else {
            vec![F::one()]
        };

        Self {
            G,
            F_table: vec![F::one()],
            ra_data,
            addr_w: addr_ch.to_vec(),
            addr_current_scalar: F::one(),
            addr_current_index: log_k_chunk,
            cycle_w: cycle_ch.to_vec(),
            cycle_current_scalar: F::one(),
            cycle_current_index: log_t,
            gamma_powers,
            gamma_powers_square,
            H: None,
            eq_r_r: F::zero(),
            eq_rest,
            log_k_chunk,
            log_t,
            num_polys,
            current_round: 0,
        }
    }

    /// Compute the 4 round-polynomial evaluations \[s(0), s(1), s(2), s(3)\].
    pub fn compute_round(&self, previous_claim: F) -> Vec<F> {
        if self.current_round < self.log_k_chunk {
            self.compute_phase1_round(previous_claim)
        } else {
            self.compute_phase2_round(previous_claim)
        }
    }

    fn compute_phase1_round(&self, previous_claim: F) -> Vec<F> {
        let m = self.current_round + 1; // 1-indexed (matches core)
        let num_rest = self.eq_rest.len(); // 2^(n − m) groups
        let num_bound = 1usize << self.current_round; // 2^(m−1) = F_table.len()

        let mut q_c = F::zero();
        let mut q_e = F::zero();

        for rest in 0..num_rest {
            let eq_val = self.eq_rest[rest];
            if eq_val.is_zero() {
                continue;
            }

            let mut group_c = F::zero();
            let mut group_e = F::zero();

            for d in 0..self.num_polys {
                let gamma_sq = self.gamma_powers_square[d];
                let g_d = &self.G[d];

                let mut inner_c = F::zero();
                let mut inner_e = F::zero();

                for bound in 0..num_bound {
                    let f_k = self.F_table[bound];

                    // X = 0: addr = rest · 2^m + bound
                    let addr_0 = (rest << m) + bound;
                    let g0_f = g_d[addr_0] * f_k;
                    inner_c += g0_f * (f_k - F::one());
                    inner_e += g0_f * f_k;

                    // X = 1: addr = rest · 2^m + 2^(m−1) + bound
                    let addr_1 = (rest << m) + (1 << (m - 1)) + bound;
                    let g1_f = g_d[addr_1] * f_k;
                    inner_e += g1_f * f_k;
                }

                group_c += gamma_sq * inner_c;
                group_e += gamma_sq * inner_e;
            }

            q_c += eq_val * group_c;
            q_e += eq_val * group_e;
        }

        gruen_poly_deg3(
            q_c,
            q_e,
            previous_claim,
            self.addr_current_scalar,
            self.addr_w[self.addr_current_index - 1],
        )
    }

    fn compute_phase2_round(&self, previous_claim: F) -> Vec<F> {
        let h = self.H.as_ref().expect("H must be initialized in Phase 2");
        let half = h[0].len() / 2;

        let mut q_c = F::zero();
        let mut q_e = F::zero();

        for j in 0..half {
            let eq_val = self.eq_rest[j];
            if eq_val.is_zero() {
                continue;
            }

            let mut group_c = F::zero();
            let mut group_e = F::zero();

            for (d, h_d) in h.iter().enumerate().take(self.num_polys) {
                let h_lo = h_d[2 * j];
                let h_hi = h_d[2 * j + 1];
                let b = h_hi - h_lo;
                let rho = self.gamma_powers[d];
                group_c += h_lo * (h_lo - rho);
                group_e += b * b;
            }

            q_c += eq_val * group_c;
            q_e += eq_val * group_e;
        }

        let adjusted_claim = previous_claim * self.eq_r_r.inverse().expect("eq_r_r nonzero");

        let inner = gruen_poly_deg3(
            q_c,
            q_e,
            adjusted_claim,
            self.cycle_current_scalar,
            self.cycle_w[self.cycle_current_index - 1],
        );

        // Multiply back by eq_r_r
        inner.into_iter().map(|v| v * self.eq_r_r).collect()
    }

    /// Advance the state after a sumcheck challenge is squeezed.
    pub fn ingest_challenge(&mut self, r_j: F) {
        if self.current_round < self.log_k_chunk {
            self.ingest_phase1(r_j);
        } else {
            self.ingest_phase2(r_j);
        }
        self.current_round += 1;
    }

    fn ingest_phase1(&mut self, r_j: F) {
        // Update addr Gruen scalar: current_scalar *= eq(w_j, r_j)
        let w_j = self.addr_w[self.addr_current_index - 1];
        let prod = w_j * r_j;
        self.addr_current_scalar *= F::one() - w_j - r_j + prod + prod;
        self.addr_current_index -= 1;

        // Expand F_table (matches core's ExpandingTable::update LowToHigh)
        let old_len = self.F_table.len();
        self.F_table.resize(2 * old_len, F::zero());
        for i in 0..old_len {
            let scaled = self.F_table[i] * r_j;
            self.F_table[old_len + i] = scaled;
            self.F_table[i] -= scaled;
        }

        // Contract eq_rest (marginalize last variable)
        contract_eq_rest(&mut self.eq_rest);

        // Transition at end of Phase 1
        if self.current_round == self.log_k_chunk - 1 {
            self.transition_to_phase2();
        }
    }

    fn ingest_phase2(&mut self, r_j: F) {
        // Update cycle Gruen scalar
        let w_j = self.cycle_w[self.cycle_current_index - 1];
        let prod = w_j * r_j;
        self.cycle_current_scalar *= F::one() - w_j - r_j + prod + prod;
        self.cycle_current_index -= 1;

        // Bind H polynomials in-place
        if let Some(ref mut h) = self.H {
            for h_d in h.iter_mut() {
                let half = h_d.len() / 2;
                for j in 0..half {
                    h_d[j] = h_d[2 * j] + r_j * (h_d[2 * j + 1] - h_d[2 * j]);
                }
                h_d.truncate(half);
            }
        }

        // Contract eq_rest
        contract_eq_rest(&mut self.eq_rest);
    }

    fn transition_to_phase2(&mut self) {
        self.eq_r_r = self.addr_current_scalar;

        let k_chunk = 1usize << self.log_k_chunk;
        let t = 1usize << self.log_t;

        // H_d[j] = γ^d · Σ_k F[k] · ra_d[k·T + j]
        let h: Vec<Vec<F>> = (0..self.num_polys)
            .map(|d| {
                let rho = self.gamma_powers[d];
                let ra_d = &self.ra_data[d];
                (0..t)
                    .map(|j| {
                        let mut sum = F::zero();
                        for k in 0..k_chunk {
                            sum += self.F_table[k] * ra_d[k * t + j];
                        }
                        rho * sum
                    })
                    .collect()
            })
            .collect();

        self.H = Some(h);

        // eq_rest for Phase 2 round 0: over cycle_w[0 .. log_t − 2]
        self.eq_rest = if self.log_t > 1 {
            EqPolynomial::<F>::evals(&self.cycle_w[..self.log_t - 1], None)
        } else {
            vec![F::one()]
        };

        // Free Phase 1 data
        self.G.clear();
        self.ra_data.clear();
        self.F_table.clear();
    }

    /// After all rounds complete, return per-polynomial RA evaluations:
    /// `ra_d(r_addr, r_cycle) = H_d[0] / gamma_powers[d]`
    ///
    /// H_d was pre-scaled by γ^d during Phase 2 construction, so we unscale.
    pub fn final_ra_claims(&self) -> Vec<F> {
        let h = self
            .H
            .as_ref()
            .expect("final_ra_claims: Phase 2 must be initialized");
        h.iter()
            .zip(&self.gamma_powers)
            .map(|(h_d, &gamma_pow)| {
                assert_eq!(h_d.len(), 1, "final_ra_claims: H not fully bound");
                h_d[0] * gamma_pow.inverse().expect("gamma nonzero")
            })
            .collect()
    }
}

/// Gruen polynomial construction: s(X) = l(X) · q(X) evaluated at {0,1,2,3}.
///
/// Identical to `GruenSplitEqPolynomial::gruen_poly_deg_3` in jolt-core.
fn gruen_poly_deg3<F: Field>(
    q_constant: F,
    q_quadratic_coeff: F,
    s_0_plus_s_1: F,
    current_scalar: F,
    w_j: F,
) -> Vec<F> {
    // Linear eq factor
    let eq_eval_1 = current_scalar * w_j;
    let eq_eval_0 = current_scalar - eq_eval_1;
    let eq_m = eq_eval_1 - eq_eval_0;
    let eq_eval_2 = eq_eval_1 + eq_m;
    let eq_eval_3 = eq_eval_2 + eq_m;

    // Quadratic reconstruction
    let cubic_eval_0 = eq_eval_0 * q_constant;
    let cubic_eval_1 = s_0_plus_s_1 - cubic_eval_0;
    let quadratic_eval_1 = cubic_eval_1 / eq_eval_1;
    let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
    let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - q_constant + e_times_2;
    let quadratic_eval_3 = quadratic_eval_2 + quadratic_eval_1 - q_constant + e_times_2 + e_times_2;

    vec![
        cubic_eval_0,
        cubic_eval_1,
        eq_eval_2 * quadratic_eval_2,
        eq_eval_3 * quadratic_eval_3,
    ]
}

/// Contract an eq table by marginalizing out its last variable:
/// `new[j] = old[2j] + old[2j+1]`.
fn contract_eq_rest<F: Field>(eq: &mut Vec<F>) {
    if eq.len() > 1 {
        let new_len = eq.len() / 2;
        for j in 0..new_len {
            eq[j] = eq[2 * j] + eq[2 * j + 1];
        }
        eq.truncate(new_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    #[test]
    fn expanding_table_matches_core() {
        // Verify our F_table expansion matches ExpandingTable::update LowToHigh.
        // After updates with r0=0.3, r1=0.7 (as field elements):
        //   F = [(1-r0)(1-r1), r0(1-r1), (1-r0)r1, r0·r1]
        let r0 = Fr::from_u64(3) / Fr::from_u64(10);
        let r1 = Fr::from_u64(7) / Fr::from_u64(10);

        let mut f = vec![Fr::one()];

        // Update with r0
        let old_len = f.len();
        f.resize(2 * old_len, Fr::zero());
        for i in 0..old_len {
            let scaled = f[i] * r0;
            f[old_len + i] = scaled;
            f[i] -= scaled;
        }
        assert_eq!(f.len(), 2);
        assert_eq!(f[0], Fr::one() - r0);
        assert_eq!(f[1], r0);

        // Update with r1
        let old_len = f.len();
        f.resize(2 * old_len, Fr::zero());
        for i in 0..old_len {
            let scaled = f[i] * r1;
            f[old_len + i] = scaled;
            f[i] -= scaled;
        }
        assert_eq!(f.len(), 4);
        assert_eq!(f[0], (Fr::one() - r0) * (Fr::one() - r1));
        assert_eq!(f[1], r0 * (Fr::one() - r1));
        assert_eq!(f[2], (Fr::one() - r0) * r1);
        assert_eq!(f[3], r0 * r1);
    }

    #[test]
    fn eq_rest_contraction() {
        // EqPolynomial([w0, w1, w2]) contracted should give EqPolynomial([w0, w1])
        let w0 = Fr::from_u64(5);
        let w1 = Fr::from_u64(7);
        let w2 = Fr::from_u64(11);

        let mut eq3 = EqPolynomial::<Fr>::evals(&[w0, w1, w2], None);
        let eq2 = EqPolynomial::<Fr>::evals(&[w0, w1], None);

        contract_eq_rest(&mut eq3);
        assert_eq!(eq3.len(), eq2.len());
        for i in 0..eq2.len() {
            assert_eq!(eq3[i], eq2[i]);
        }
    }
}
