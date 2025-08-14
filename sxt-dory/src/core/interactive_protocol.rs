//! We implement the interactive protocol for prover <> verifier Dory proofs
//! This mainly involves the messages in the dory-reduce protocol
use rayon::prelude::*;

use crate::{
    arithmetic::{Field, Group, MultiScalarMul, Pairing},
    messages::{
        FirstReduceChallenge, FirstReduceMessage, FoldScalarsChallenge, ScalarProductMessage,
        SecondReduceChallenge, SecondReduceMessage,
    },
    setup::VerifierSetup,
    state::{DoryProverState, DoryVerifierState, VerifierState},
};

use super::{ProverSetup, ScalarProductChallenge};

/// Below is the **prover** side of the interactive protocol for Dory
/// We define the relevant message implementations in the order of communication
impl<E: Pairing> crate::ProverState for DoryProverState<E>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    type G1 = E::G1;
    type G2 = E::G2;
    type GT = E::GT;
    type Scalar = <E::G1 as Group>::Scalar;
    type Setup = ProverSetup<E>;

    /* ---------- First‑Reduce --------------------------------------- */
    #[tracing::instrument(skip_all)]
    fn compute_first_reduce_message<M1, M2>(
        &self,
        setup: &Self::Setup,
    ) -> FirstReduceMessage<Self::G1, Self::G2, Self::GT>
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>,
    {
        if self.nu == 0 {
            panic!("Not enough rounds left in prover state");
        }

        // n/2
        let n2 = 1usize << (self.nu - 1);

        let (v1_l, v1_r) = self.v1.split_at(n2);
        let (v2_l, v2_r) = self.v2.split_at(n2);

        /* ---------- COMPUTE D ---------- */
        // Collapsed Γ-vectors of length n/2 (Γ₁′, Γ₂′)
        let g2_prime = &setup.g2_vec()[..1 << (self.nu - 1)];
        let g1_prime = &setup.g1_vec()[..1 << (self.nu - 1)];

        let (d1_left, d1_right, d2_left, d2_right) =
                // Use cached G2 if available, always use runtime G1
                if setup.g2_cache.is_some() {
                    let g2_prime_count = 1 << (self.nu - 1);

                    // D₁L,R = ⟨v₁L/R , Γ₂′⟩ - v1 is runtime, g2_prime uses cache
                    let d1_left = E::multi_pair_cached(
                        Some(v1_l),
                        None,
                        None, // G1: use runtime points v1_l
                        None,
                        Some(g2_prime_count),
                        setup.g2_cache.as_ref(), // G2: use first 2^(nu-1) cached elements
                    );
                    let d1_right = E::multi_pair_cached(
                        Some(v1_r),
                        None,
                        None, // G1: use runtime points v1_r
                        None,
                        Some(g2_prime_count),
                        setup.g2_cache.as_ref(), // G2: use first 2^(nu-1) cached elements
                    );

                    // D₂L,R = ⟨Γ₁′ , v₂L/R⟩ - g1_prime is runtime, v2 is runtime
                    let d2_left = E::multi_pair(g1_prime, v2_l);
                    let d2_right = E::multi_pair(g1_prime, v2_r);
                    (d1_left, d1_right, d2_left, d2_right)
                } else {
                    // Fallback to regular multi-pairing when cache is not available
                    let d1_left = E::multi_pair(v1_l, g2_prime);
                    let d1_right = E::multi_pair(v1_r, g2_prime);
                    let d2_left = E::multi_pair(g1_prime, v2_l);
                    let d2_right = E::multi_pair(g1_prime, v2_r);
                    (d1_left, d1_right, d2_left, d2_right)
                };

        /* ---------- COMPUTE E (for extended protocol) ---------- */
        // E₁β = ⟨Γ₁ , s₂⟩
        let e1_beta = M1::msm(&setup.g1_vec()[..1 << self.nu], &self.s2);
        // E₂β = ⟨Γ₂ , s₁⟩
        let e2_beta = M2::msm(&setup.g2_vec()[..1 << self.nu], &self.s1);

        FirstReduceMessage {
            d1_left,
            d1_right,
            d2_left,
            d2_right,
            e1_beta,
            e2_beta,
        }
    }

    /* ---------- Reduce-Combine --------------------------------------- */
    #[tracing::instrument(skip_all)]
    fn reduce_combine<M1, M2>(
        mut self,
        setup: &Self::Setup,
        chall: FirstReduceChallenge<Self::Scalar>,
    ) -> Self
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>,
    {
        let beta = chall.beta;
        let beta_inv = chall.beta_inverse;

        let g1_prime = &setup.g1_vec()[..1 << self.nu];
        let g2_prime = &setup.g2_vec()[..1 << self.nu];

        // Prover work P(*):
        // ṽ₁ ← ṽ₁ + β·Γ₁
        // Use cached version if cache is available
        if setup.g1_cache.is_some() && setup.g2_cache.is_some() {
            M1::fixed_scalar_variable_with_add_cached(
                g1_prime.len(),
                setup.g1_cache.as_ref(),
                setup.g2_cache.as_ref(),
                &mut self.v1,
                &beta,
            );
        } else {
            M1::fixed_scalar_variable_with_add(g1_prime, &mut self.v1, &beta);
        }

        // ṽ₂ ← ṽ₂ + β⁻¹·Γ₂
        // Use cached version if cache is available
        if setup.g1_cache.is_some() && setup.g2_cache.is_some() {
            M2::fixed_scalar_variable_with_add_cached(
                g2_prime.len(),
                setup.g1_cache.as_ref(),
                setup.g2_cache.as_ref(),
                &mut self.v2,
                &beta_inv,
            );
        } else {
            M2::fixed_scalar_variable_with_add(g2_prime, &mut self.v2, &beta_inv);
        }

        self
    }

    /* ---------- Second‑Reduce -------------------------------------- */
    #[tracing::instrument(skip_all)]
    fn compute_second_reduce_message<M1, M2>(
        &self,
        _setup: &Self::Setup, // not used in this step
    ) -> SecondReduceMessage<Self::G1, Self::G2, Self::GT>
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>,
    {
        let n2 = 1usize << (self.nu - 1);

        let (v1_l, v1_r) = self.v1.split_at(n2);
        let (v2_l, v2_r) = self.v2.split_at(n2);
        let (s1_l, s1_r) = self.s1.split_at(n2);
        let (s2_l, s2_r) = self.s2.split_at(n2);

        // ---- C terms ----------------------------------------------------------
        let c_plus = E::multi_pair(v1_l, v2_r); // ⟨v₁L, v₂R⟩
        let c_minus = E::multi_pair(v1_r, v2_l); // ⟨v₁R, v₂L⟩

        // ---- E terms (extended protocol) ---------------------------------------
        let e1_plus = M1::msm(v1_l, s2_r); // ⟨v₁L, s₂R⟩
        let e1_minus = M1::msm(v1_r, s2_l); // ⟨v₁R, s₂L⟩
        let e2_plus = M2::msm(v2_r, s1_l); // ⟨v₂R, s₁L⟩
        let e2_minus = M2::msm(v2_l, s1_r); // ⟨v₂L, s₁R⟩

        SecondReduceMessage {
            c_plus,
            c_minus,
            e1_plus,
            e1_minus,
            e2_plus,
            e2_minus,
        }
    }

    /// On every round, cut the vector length in half and fold with the
    /// α-challenge:
    ///
    ///   v₁ ← α · v₁L + v₁R
    ///   v₂ ← α⁻¹· v₂L + v₂R
    ///   s₁ ← α · s₁L + s₁R
    ///   s₂ ← α⁻¹· s₂L + s₂R
    ///
    /// After folding, all four vectors are truncated to `n/2`.
    #[tracing::instrument(skip_all)]
    fn reduce_fold<M1, M2>(
        mut self,
        _setup: &Self::Setup,
        chall: SecondReduceChallenge<Self::Scalar>,
    ) -> Self
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>,
    {
        let (alpha, alpha_inv) = (chall.alpha, chall.alpha_inverse);
        let n2 = 1usize << (self.nu - 1);

        /* ─── fold v-vectors ────────────────────────────────────────────── */
        let (v1_l, v1_r_slice) = self.v1.split_at_mut(n2);
        let v1_r = &*v1_r_slice; // Convert mutable slice to immutable for par_iter()

        let (v2_l, v2_r_slice) = self.v2.split_at_mut(n2);
        let v2_r = &*v2_r_slice;

        M1::fixed_scalar_scale_with_add(v1_l, v1_r, &alpha);

        M2::fixed_scalar_scale_with_add(v2_l, v2_r, &alpha_inv);

        self.v1.truncate(n2);
        self.v2.truncate(n2);

        /* ─── fold s-vectors (extended protocol)──────────-────────────────── */
        let (s1_l, s1_r_slice) = self.s1.split_at_mut(n2);
        let s1_r = &*s1_r_slice;

        let (s2_l, s2_r_slice) = self.s2.split_at_mut(n2);
        let s2_r = s2_r_slice;

        s1_l.par_iter_mut()
            .zip(s1_r.par_iter())
            .for_each(|(s_l, s_r_val)| *s_l = s_l.mul(&alpha).add(s_r_val));

        s2_l.par_iter_mut()
            .zip(s2_r.par_iter())
            .for_each(|(s_l, s_r_val)| *s_l = s_l.mul(&alpha_inv).add(s_r_val));

        self.s1.truncate(n2);
        self.s2.truncate(n2);

        self.nu -= 1;

        self
    }

    /* ---------- Final Scalar‑Product input -------------------------- */
    // Note: we apply `fold-scalars` transform onto e1 and e1 before sending to verifier.
    // We apply it here as this is the last step of the IP protocol and the transformation mutates
    // the contents of `ScalarProductMessage`, anyways.
    #[tracing::instrument(skip_all)]
    fn compute_scalar_product_message<M1, M2>(
        self,
        setup: &Self::Setup,
        chall: FoldScalarsChallenge<Self::Scalar>,
    ) -> ScalarProductMessage<Self::G1, Self::G2>
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>,
    {
        debug_assert_eq!(self.nu, 0);
        debug_assert_eq!(self.v1.len(), 1);
        debug_assert_eq!(self.v2.len(), 1);
        debug_assert_eq!(self.s1.len(), 1);
        debug_assert_eq!(self.s2.len(), 1);

        let (gamma, gamma_inv) = (chall.gamma, chall.gamma_inverse);

        // Apply `fold-scalars`` transformation to the vectors:
        // v1' = v1 + γ * s1 * H1
        // v2' = v2 + γ^(-1) * s2 * H2

        let gamma_s1_product = gamma.mul(&self.s1[0]);
        let e1 = self.v1[0].add(&setup.h1().scale(&gamma_s1_product));

        let gamma_inv_s2_product = gamma_inv.mul(&self.s2[0]);
        let e2 = self.v2[0].add(&setup.h2().scale(&gamma_inv_s2_product));

        ScalarProductMessage {
            e1: e1.clone(),
            e2: e2.clone(),
        }
    }
}

/// Below is the **verifier** side of the interactive protocol for Dory
/// We define the relevant message implementations in the order of communication
impl<E: Pairing> VerifierState for DoryVerifierState<E>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
    E::GT: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    type G1 = E::G1;
    type G2 = E::G2;
    type GT = E::GT;
    type Scalar = <E::G1 as Group>::Scalar;
    type Setup = VerifierSetup<E>;

    /// This is the round i verifier algorithm of the extended Dory-Reduce algorithm in section 3.2 & 4.2 of the paper.
    /// This function should be called after messages are received and challenges are pulled from the transcript.
    fn dory_reduce_verify_round(
        &mut self,
        setup: &Self::Setup,
        first_msg: &FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
        second_msg: &SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
        alpha_pair: (Self::Scalar, Self::Scalar),
        beta_pair: (Self::Scalar, Self::Scalar),
    ) -> bool {
        let (alpha, alpha_inv) = alpha_pair;
        let (beta, beta_inv) = beta_pair;

        // Update C according to the protocol
        // C' <- C + χᵢ + β * D₂ + β⁻¹ * D₁ + α * C_plus + α⁻¹ * C_minus
        Self::dory_reduce_verify_update_c(
            self,
            setup,
            (second_msg.c_plus.clone(), second_msg.c_minus.clone()),
            (alpha.clone(), alpha_inv.clone()),
            (beta.clone(), beta_inv.clone()),
        );

        // Update D₁ and D₂ according to the protocol
        // D₁' <- α * D₁L + D₁R + α * β * Δ₁L + β * Δ₁R
        // D₂' <- α⁻¹ * D₂L + D₂R + α⁻¹ * β⁻¹ * Δ₂L + β⁻¹ * Δ₂R
        Self::dory_reduce_verify_update_ds(
            self,
            setup,
            (
                first_msg.d1_left.clone(),
                first_msg.d1_right.clone(),
                first_msg.d2_left.clone(),
                first_msg.d2_right.clone(),
            ),
            (alpha.clone(), alpha_inv.clone()),
            (beta.clone(), beta_inv.clone()),
        );

        // Update E₁ and E₂ for the **extended** protocol
        // E₁' <- E₁ + β * E₁β + α * E₁+ + α⁻¹ * E₁-
        // E₂' <- E₂ + β⁻¹ * E₂β + α * E₂+ + α⁻¹ * E₂-
        Self::dory_reduce_verify_update_es(
            self,
            (first_msg.e1_beta.clone(), first_msg.e2_beta.clone()),
            (
                second_msg.e1_plus.clone(),
                second_msg.e1_minus.clone(),
                second_msg.e2_plus.clone(),
                second_msg.e2_minus.clone(),
            ),
            (alpha.clone(), alpha_inv.clone()),
            (beta.clone(), beta_inv.clone()),
        );

        // We update s1_tensor and s2_tensor if they exist, mirroring `fold s-vectors` in `reduce_fold`:
        if let (Some(s1), Some(s2)) = (self.s1_tensor.as_mut(), self.s2_tensor.as_mut()) {
            let n2 = 1usize << (self.nu - 1);

            // Split s1 and s2 into left and right halves
            let (s1_l, s1_r) = s1.split_at_mut(n2);
            let (s2_l, s2_r) = s2.split_at_mut(n2);

            // Fold s1: s1' = α * s1_l + s1_r
            for i in 0..n2 {
                s1_l[i] = s1_l[i].mul(&alpha).add(&s1_r[i]);
            }

            // Fold s2: s2' = α⁻¹ * s2_l + s2_r
            for i in 0..n2 {
                s2_l[i] = s2_l[i].mul(&alpha_inv).add(&s2_r[i]);
            }

            // Truncate to keep only the folded half
            s1.truncate(n2);
            s2.truncate(n2);
        }

        // decrement the rounds
        self.nu -= 1;

        true
    }

    /// From the Dory-Reduce algorithm in section 3.2 of the paper.
    /// Updates C in the verifier state.
    /// C' <- C + χᵢ + β * D₂ + β⁻¹ * D₁ + α * C_plus + α⁻¹ * C_minus
    fn dory_reduce_verify_update_c(
        &mut self,
        setup: &Self::Setup,
        c_pair: (Self::GT, Self::GT),
        alpha_pair: (Self::Scalar, Self::Scalar),
        beta_pair: (Self::Scalar, Self::Scalar),
    ) {
        let (c_plus, c_minus) = c_pair;
        let (alpha, alpha_inv) = alpha_pair;
        let (beta, beta_inv) = beta_pair;

        let chi = &setup.chi[self.nu];

        // Update c with each component
        // C' <- C + χᵢ + β * D₂ + β⁻¹ * D₁ + α * C_plus + α⁻¹ * C_minus

        // Start with the original C
        let mut new_c = self.c.clone();

        // Add χᵢ
        new_c = new_c.add(chi);

        // Add β * D₂
        new_c = new_c.add(&self.d_2.scale(&beta));

        // Add β⁻¹ * D₁
        new_c = new_c.add(&self.d_1.scale(&beta_inv));

        // Add α * C_plus
        new_c = new_c.add(&c_plus.scale(&alpha));

        // Add α⁻¹ * C_minus
        new_c = new_c.add(&c_minus.scale(&alpha_inv));

        self.c = new_c;
    }

    /// From the Dory-Reduce algorithm in section 3.2 of the paper.
    /// Updates `D_1` and `D_2`
    /// * `D_1' <- alpha * D_1L + D_1R + alpha * beta * Delta_1L + beta * Delta_1R`
    /// * `D_2' <- alpha_inv * D_2L + D_2R + alpha_inv * beta_inv * Delta_2L + beta_inv * Delta_2R`
    fn dory_reduce_verify_update_ds(
        &mut self,
        setup: &Self::Setup,
        d_values: (Self::GT, Self::GT, Self::GT, Self::GT),
        alpha_pair: (Self::Scalar, Self::Scalar),
        beta_pair: (Self::Scalar, Self::Scalar),
    ) {
        let (d_1l, d_1r, d_2l, d_2r) = d_values;
        let (alpha, alpha_inv) = alpha_pair;
        let (beta, beta_inv) = beta_pair;

        // Get the precomputed values for the current round
        let delta_1l = &setup.delta_1l[self.nu];
        let delta_1r = &setup.delta_1r[self.nu];
        let delta_2l = &setup.delta_2l[self.nu];
        let delta_2r = &setup.delta_2r[self.nu];

        // D_1' <- alpha * D_1L + D_1R + alpha * beta * Delta_1L + beta * Delta_1R
        let mut new_d_1 = d_1l.scale(&alpha);
        new_d_1 = new_d_1.add(&d_1r);

        // alpha * beta * Delta_1L
        let alpha_beta = alpha.mul(&beta);
        new_d_1 = new_d_1.add(&delta_1l.scale(&alpha_beta));

        // beta * Delta_1R
        new_d_1 = new_d_1.add(&delta_1r.scale(&beta));

        // D_2' <- alpha_inv * D_2L + D_2R + alpha_inv * beta_inv * Delta_2L + beta_inv * Delta_2R
        let mut new_d_2 = d_2l.scale(&alpha_inv);
        new_d_2 = new_d_2.add(&d_2r);

        // alpha_inv * beta_inv * Delta_2L
        let alpha_inv_beta_inv = alpha_inv.mul(&beta_inv);
        new_d_2 = new_d_2.add(&delta_2l.scale(&alpha_inv_beta_inv));

        // beta_inv * Delta_2R
        new_d_2 = new_d_2.add(&delta_2r.scale(&beta_inv));

        self.d_1 = new_d_1;
        self.d_2 = new_d_2;
    }

    /// From the extended Dory-Reduce algorithm in section 4.2 of the paper.
    /// Updates `E_1` and `E_2`
    /// * `E_1' <- E_1 + beta * E_1beta + alpha * E_1plus + alpha_inv * E_1minus`
    /// * `E_2' <- E_2 + beta_inv * E_2beta + alpha * E_2plus + alpha_inv * E_2minus`
    fn dory_reduce_verify_update_es(
        &mut self,
        e_beta_pair: (Self::G1, Self::G2),
        e_values: (Self::G1, Self::G1, Self::G2, Self::G2),
        alpha_pair: (Self::Scalar, Self::Scalar),
        beta_pair: (Self::Scalar, Self::Scalar),
    ) {
        let (e_1beta, e_2beta) = e_beta_pair;
        let (e_1plus, e_1minus, e_2plus, e_2minus) = e_values;
        let (alpha, alpha_inv) = alpha_pair;
        let (beta, beta_inv) = beta_pair;

        // E_1' <- E_1 + beta * E_1beta + alpha * E_1plus + alpha_inv * E_1minus
        let mut new_e_1: Self::G1 = self.e_1.clone();
        new_e_1 = new_e_1.add(&e_1beta.scale(&beta));
        new_e_1 = new_e_1.add(&e_1plus.scale(&alpha));
        new_e_1 = new_e_1.add(&e_1minus.scale(&alpha_inv));

        // E_2' <- E_2 + beta_inv * E_2beta + alpha * E_2plus + alpha_inv * E_2minus
        let mut new_e_2 = self.e_2.clone();
        new_e_2 = new_e_2.add(&e_2beta.scale(&beta_inv));
        new_e_2 = new_e_2.add(&e_2plus.scale(&alpha));
        new_e_2 = new_e_2.add(&e_2minus.scale(&alpha_inv));

        self.e_1 = new_e_1;
        self.e_2 = new_e_2;
    }

    /// Verifier side of the Fold-Scalars protocol from the paper
    fn apply_fold_scalars(
        &mut self,
        setup: &VerifierSetup<E>,
        gamma_pair: FoldScalarsChallenge<Self::Scalar>,
    ) {
        let gamma = gamma_pair.gamma;
        let gamma_inv = gamma_pair.gamma_inverse;

        debug_assert_eq!(self.nu, 0, "nu should be 0 at the Fold-Scalars step.");

        let s1_final = self
            .s1_tensor
            .as_ref()
            .and_then(|v| v.get(0))
            .cloned()
            .unwrap();

        let s2_final = self
            .s2_tensor
            .as_ref()
            .and_then(|v| v.get(0))
            .cloned()
            .unwrap();

        // --- C' Update ---
        // C' ← C + ⟨s̃₁, s̃₂⟩HT + γ · e(H₁, E₂) + γ⁻¹ · e(E₁, H₂)
        // At nu=0, ⟨s̃₁, s̃₂⟩ becomes s1_final * s2_final

        let mut new_c = self.c.clone();

        // Term 1: s1_final * s2_final * setup.ht
        new_c = new_c.add(&setup.ht.scale(&s1_final.mul(&s2_final)));

        // Term 2: γ * e(setup.h1, self.e_2)
        let pairing_h1_e2 = E::pair(&setup.h1, &self.e_2);
        new_c = new_c.add(&pairing_h1_e2.scale(&gamma));

        // Term 3: γ⁻¹ * e(self.e_1, setup.h2)
        let pairing_e1_h2 = E::pair(&self.e_1, &setup.h2);
        new_c = new_c.add(&pairing_e1_h2.scale(&gamma_inv));

        self.c = new_c;

        // --- D₁' Update
        // D₁' ← D₁ + e(H₁, ⟨s̃₁, γΓ₂⟩)
        // At nu=0, ⟨s̃₁, γΓ₂⟩ becomes Γ₂₀·s̃₁γ which is setup.g2_0 * s1_final * gamma

        let scalar_for_g2_in_d1 = s1_final.mul(&gamma);
        // Scale setup.g2_0 (which is Γ₂₀) by scalar_for_g2_in_d1
        let g2_0_scaled = setup.g2_0.scale(&scalar_for_g2_in_d1);
        let pairing_h1_g2 = E::pair(&setup.h1, &g2_0_scaled);

        self.d_1 = self.d_1.add(&pairing_h1_g2);

        // --- D₂' Update
        // D₂' ← D₂ + e(γ⁻¹⟨Γ₁, s̃₂⟩, H₂)
        // At nu=0, γ⁻¹⟨Γ₁, s̃₂⟩ becomes Γ₁₀·s̃₂γ⁻¹ which is setup.g1_0 * s2_final * gamma_inv

        let scalar_for_g1_in_d2 = s2_final.mul(&gamma_inv);
        // Scale setup.g1_0 (which is Γ₁₀) by scalar_for_g1_in_d2
        let g1_0_scaled = setup.g1_0.scale(&scalar_for_g1_in_d2);
        let pairing_g1_h2 = E::pair(&g1_0_scaled, &setup.h2);

        self.d_2 = self.d_2.add(&pairing_g1_h2);
    }

    /// Final verification step for Extended Dory-Reduce
    /// This implements the final pairing check based on:
    /// 1. The scalar product verification from section 3.1 of the paper
    ///
    /// For nu = 0, we check:
    /// pairing(E_1 + Gamma_1_0 * d, E_2 + Gamma_2_0 * d_inv) ==
    /// (C + chi[0] + D_2 * d + D_1 * d_inv)
    fn verify_final_pairing(
        &self,
        setup: &Self::Setup,
        message: &ScalarProductMessage<Self::G1, Self::G2>,
        d_pair: ScalarProductChallenge<Self::Scalar>, // This should be a fresh challenge 'd', not gamma
    ) -> bool {
        // challenge
        let (d, d_inverse) = (d_pair.d, d_pair.d_inverse);

        // Assert that we're at the appropriate round (nu = 0)
        assert_eq!(self.nu, 0);

        // Left side of the equation: pairing(E_1 + Gamma_1_0 * gamma, E_2 + Gamma_2_0 * gamma_inv)
        let e1_modified = message.e1.add(&setup.g1_0.scale(&d));
        let e2_modified = message.e2.add(&setup.g2_0.scale(&d_inverse));

        let left_side = E::pair(&e1_modified, &e2_modified);

        // Right side of the equation: (C + chi[0] + D_2 * gamma + D_1 * gamma_inv)
        let mut right_side = self.c.clone();

        // Add chi[0]
        right_side = right_side.add(&setup.chi[0]);

        // Add D_2 * gamma
        right_side = right_side.add(&self.d_2.scale(&d));

        // Add D_1 * gamma_inv
        right_side = right_side.add(&self.d_1.scale(&d_inverse));

        // Compare the two sides
        left_side == right_side
    }
}
