//! HammingWeight + Address Reduction sumcheck state for the CPU backend.
//!
//! Computes G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j), then runs a degree-2
//! sumcheck: Σ_k Σ_i G_i(k) · (γ_hw + γ_bool·eq_bool(k) + γ_virt·eq_virt_i(k)).

#![allow(non_snake_case)]

use jolt_field::Field;
use jolt_poly::EqPolynomial;

pub struct CpuHwReductionState<F: Field> {
    G: Vec<Vec<F>>,
    eq_bool: Vec<F>,
    eq_virt: Vec<Vec<F>>,
    gamma_powers: Vec<F>,
    hw_claims: Vec<F>,
    bool_claims: Vec<F>,
    virt_claims: Vec<F>,
    num_polys: usize,
}

impl<F: Field> CpuHwReductionState<F> {
    /// Build the HW reduction state from RA data and challenge values.
    ///
    /// * `ra_data` – per-dimension RA polynomial in AddressMajor layout: ra[k*T + j].
    /// * `cycle_ch_be` – r_cycle challenges (BE, length `log_t`).
    /// * `addr_bool_ch_be` – r_addr_bool challenges (BE, length `log_k_chunk`).
    /// * `addr_virt_ch_be` – per-RA r_addr_virt challenges (BE, length `log_k_chunk` each).
    /// * `gamma_powers` – γ^0, γ^1, ..., γ^{3N-1}.
    /// * `hw_claims` – per-RA HammingWeight claims (1 for inst/bc, ram_hw_factor for ram).
    /// * `bool_claims` – per-RA booleanity claims.
    /// * `virt_claims` – per-RA virtualization claims.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ra_data: &[Vec<F>],
        cycle_ch_be: &[F],
        addr_bool_ch_be: &[F],
        addr_virt_ch_be: &[Vec<F>],
        gamma_powers: Vec<F>,
        hw_claims: Vec<F>,
        bool_claims: Vec<F>,
        virt_claims: Vec<F>,
        log_k_chunk: usize,
        log_t: usize,
    ) -> Self {
        let k_chunk = 1usize << log_k_chunk;
        let t = 1usize << log_t;
        let num_polys = ra_data.len();

        // G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k·T + j)
        // cycle_ch_be is in BE (MSB first) — matches EqPolynomial::evals convention.
        let cycle_eq: Vec<F> = EqPolynomial::<F>::evals(cycle_ch_be, None);
        let G: Vec<Vec<F>> = ra_data
            .iter()
            .map(|ra| {
                let mut g = vec![F::zero(); k_chunk];
                for k in 0..k_chunk {
                    for j in 0..t {
                        let ra_val = ra[k * t + j];
                        if !ra_val.is_zero() {
                            g[k] += cycle_eq[j] * ra_val;
                        }
                    }
                }
                g
            })
            .collect();

        // eq_bool: shared across all families
        let eq_bool: Vec<F> = EqPolynomial::<F>::evals(addr_bool_ch_be, None);

        // eq_virt: one per RA polynomial
        let eq_virt: Vec<Vec<F>> = addr_virt_ch_be
            .iter()
            .map(|ch| EqPolynomial::<F>::evals(ch, None))
            .collect();

        Self {
            G,
            eq_bool,
            eq_virt,
            gamma_powers,
            hw_claims,
            bool_claims,
            virt_claims,
            num_polys,
        }
    }

    /// Compute the input claim: Σ_i (γ^{3i}·H_i + γ^{3i+1}·bool_i + γ^{3i+2}·virt_i).
    pub fn input_claim(&self) -> F {
        let mut claim = F::zero();
        for i in 0..self.num_polys {
            claim += self.gamma_powers[3 * i] * self.hw_claims[i];
            claim += self.gamma_powers[3 * i + 1] * self.bool_claims[i];
            claim += self.gamma_powers[3 * i + 2] * self.virt_claims[i];
        }
        claim
    }

    /// Compute round evaluations (3 values: eval_0, eval_2 for degree-2 polynomial).
    pub fn reduce(&self, previous_claim: F) -> Vec<F> {
        let half_n = self.G[0].len() / 2;
        let mut evals = [F::zero(); 2]; // degree 2: eval_0, eval_2

        for j in 0..half_n {
            let eq_b_lo = self.eq_bool[2 * j];
            let eq_b_hi = self.eq_bool[2 * j + 1];
            // eval_2 point: 2·hi - lo
            let eq_b_2 = eq_b_hi + eq_b_hi - eq_b_lo;

            for i in 0..self.num_polys {
                let g_lo = self.G[i][2 * j];
                let g_hi = self.G[i][2 * j + 1];
                let g_2 = g_hi + g_hi - g_lo;

                let eq_v_lo = self.eq_virt[i][2 * j];
                let eq_v_hi = self.eq_virt[i][2 * j + 1];
                let eq_v_2 = eq_v_hi + eq_v_hi - eq_v_lo;

                let gamma_hw = self.gamma_powers[3 * i];
                let gamma_bool = self.gamma_powers[3 * i + 1];
                let gamma_virt = self.gamma_powers[3 * i + 2];

                // eval_0: G_lo · (γ_hw + γ_bool·eq_b_lo + γ_virt·eq_v_lo)
                evals[0] += g_lo * (gamma_hw + gamma_bool * eq_b_lo + gamma_virt * eq_v_lo);
                // eval_2: G_2 · (γ_hw + γ_bool·eq_b_2 + γ_virt·eq_v_2)
                evals[1] += g_2 * (gamma_hw + gamma_bool * eq_b_2 + gamma_virt * eq_v_2);
            }
        }

        // UniPoly::from_evals_and_hint convention: [eval_0, eval_2]
        // eval_1 = previous_claim - eval_0
        // The runtime will reconstruct the full compressed polynomial.
        vec![evals[0], previous_claim - evals[0], evals[1]]
    }

    /// Bind all polynomials at a challenge value (LowToHigh order).
    pub fn bind(&mut self, r: F) {
        let one_minus_r = F::one() - r;
        for g in &mut self.G {
            let half = g.len() / 2;
            for j in 0..half {
                g[j] = g[2 * j] * one_minus_r + g[2 * j + 1] * r;
            }
            g.truncate(half);
        }
        {
            let half = self.eq_bool.len() / 2;
            for j in 0..half {
                self.eq_bool[j] = self.eq_bool[2 * j] * one_minus_r + self.eq_bool[2 * j + 1] * r;
            }
            self.eq_bool.truncate(half);
        }
        for eq in &mut self.eq_virt {
            let half = eq.len() / 2;
            for j in 0..half {
                eq[j] = eq[2 * j] * one_minus_r + eq[2 * j + 1] * r;
            }
            eq.truncate(half);
        }
    }

    /// Extract final G_i evaluations (after all rounds bound G to 1 element each).
    pub fn final_g_claims(&self) -> Vec<F> {
        self.G
            .iter()
            .map(|g| {
                assert_eq!(g.len(), 1, "G not fully bound");
                g[0]
            })
            .collect()
    }
}
