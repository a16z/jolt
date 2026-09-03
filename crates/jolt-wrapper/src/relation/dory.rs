//! The Dory evaluation-proof transcript and its `Fr` scalar algebra. Group
//! elements are opaque prover bytes here; every scalar the pairing-side
//! verifier (T2) multiplies by is an R1CS output named in [`DoryLinks`].

use std::collections::HashMap;

use jolt_r1cs::Variable;

use super::ctx::{Ctx, Lc};
use super::gadgets::{one_minus, reversed};
use super::replay::SqueezeKind;
use super::RelationError;

const DORY_SERDE_LABEL: &[u8] = b"dory_serde";
/// Arkworks compressed encodings: `Fq12`, `G1Affine`, `G2Affine`.
const GT_BYTES: usize = 384;
const G1_BYTES: usize = 32;
const G2_BYTES: usize = 64;

/// A Dory verifier scalar, indexed by fold round `j` (0 = first fold) where
/// applicable. `Delta1R(k)` / `Delta2R(k)` use the folded coordinate index
/// `k = σ − 1 − j`; the native setup constant paired with them is
/// `setup.delta_*[k + 1]`. `Chi(k)` uses that same coordinate index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DoryScalar {
    /// `y`: the joint evaluation claim (scalar on `Γ2_0`).
    Evaluation,
    /// `ρ^i · β_0⁻¹`: the commitment-RLC weight of the `i`-th committed polynomial.
    CommitmentWeight(usize),
    Beta(usize),
    BetaInv(usize),
    Alpha(usize),
    AlphaInv(usize),
    Gamma,
    GammaInv,
    D,
    DInv,
    DSquared,
    /// `β_0 + d²`: the initial `D2` accumulator scalar.
    D2Init,
    /// `u_j`: `β_{j+1}⁻¹`, or `d⁻¹` in the last round.
    U(usize),
    /// `v_j`: `β_{j+1}`, or `d` in the last round.
    V(usize),
    /// `u_j · α_j` (D1L).
    UAlpha(usize),
    /// `v_j · α_j⁻¹` (D2L).
    VAlphaInv(usize),
    /// `1 + u_j α_j β_j + v_j α_j⁻¹ β_j⁻¹` (the `χ[k]` scalar).
    Chi(usize),
    /// `u_j · β_j` (Δ1R).
    Delta1R(usize),
    /// `v_j · β_j⁻¹` (Δ2R).
    Delta2R(usize),
    /// `Π_j (α_j (1 − y_j) + y_j)`.
    S1Acc,
    /// `Π_j (α_j⁻¹ (1 − x_j) + x_j)`.
    S2Acc,
    /// `s1_acc · s2_acc` (the `HT` scalar).
    Ht,
    /// `−γ · d⁻¹ · s1_acc`: the `Γ2_0` coefficient of the `e(H1, ·)` pairing input.
    PairingG2ZeroScalar,
    /// `−γ⁻¹ · d · s2_acc`: the `Γ1_0` coefficient of the `e(·, H2)` pairing input.
    PairingG1ZeroScalar,
}

impl DoryScalar {
    pub fn link_order(sigma: usize, commitments: usize) -> Vec<Self> {
        let mut order = Vec::with_capacity(commitments + 12 * sigma + 9);
        order.extend((0..commitments).map(Self::CommitmentWeight));
        order.push(Self::D2Init);
        for round in 0..sigma {
            let coordinate = sigma - 1 - round;
            order.extend([
                Self::Alpha(round),
                Self::AlphaInv(round),
                Self::UAlpha(round),
                Self::U(round),
                Self::VAlphaInv(round),
                Self::V(round),
                Self::Delta1R(coordinate),
                Self::Delta2R(coordinate),
            ]);
        }
        order.extend((0..sigma).map(Self::Chi));
        order.push(Self::Ht);
        order.extend((0..sigma).map(Self::Beta));
        order.extend([
            Self::GammaInv,
            Self::PairingG1ZeroScalar,
            Self::D,
            Self::DSquared,
            Self::DInv,
        ]);
        order.extend((0..sigma).map(Self::BetaInv));
        order.extend([Self::Evaluation, Self::Gamma, Self::PairingG2ZeroScalar]);
        order
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoryLinks {
    pub num_vars: usize,
    pub sigma: usize,
    pub scalars: Vec<(DoryScalar, Variable)>,
}

/// Runs the Dory transcript (opaque group elements, scalar challenges) for a
/// `num_vars`-variable opening and emits the verifier scalars.
pub(crate) fn walk(
    ctx: &mut Ctx,
    opening_point: &[Lc],
    evaluation: &Lc,
    rlc_powers: &[Lc],
) -> Result<DoryLinks, RelationError> {
    let num_vars = opening_point.len();
    let sigma = num_vars.div_ceil(2);
    let nu = num_vars - sigma;
    // Dory reads the point low-to-high: coordinate 0 is the last Jolt coordinate.
    let point = reversed(opening_point);
    let s1_coords = &point[..sigma];
    let mut s2_coords = point[sigma..sigma + nu].to_vec();
    s2_coords.resize(sigma, Lc::zero());

    let opaque = |ctx: &mut Ctx, len: usize| -> Result<(), RelationError> {
        ctx.absorb_label_count(DORY_SERDE_LABEL, len)?;
        ctx.absorb_opaque(len)
    };
    // vmv_c, vmv_d2 (GT), vmv_e1 (G1).
    opaque(ctx, GT_BYTES)?;
    opaque(ctx, GT_BYTES)?;
    opaque(ctx, G1_BYTES)?;
    let mut betas = Vec::with_capacity(sigma);
    let mut alphas = Vec::with_capacity(sigma);
    for _ in 0..sigma {
        for _ in 0..4 {
            opaque(ctx, GT_BYTES)?;
        }
        opaque(ctx, G1_BYTES)?;
        opaque(ctx, G2_BYTES)?;
        betas.push(ctx.squeeze(SqueezeKind::Scalar)?);
        opaque(ctx, GT_BYTES)?;
        opaque(ctx, GT_BYTES)?;
        opaque(ctx, G1_BYTES)?;
        opaque(ctx, G1_BYTES)?;
        opaque(ctx, G2_BYTES)?;
        opaque(ctx, G2_BYTES)?;
        alphas.push(ctx.squeeze(SqueezeKind::Scalar)?);
    }
    let gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    opaque(ctx, G1_BYTES)?;
    opaque(ctx, G2_BYTES)?;
    let d = ctx.squeeze(SqueezeKind::Scalar)?;

    let mut scalar_variables = HashMap::new();
    let mut emit = |ctx: &mut Ctx, scalar: DoryScalar, lc: &Lc| {
        let variable = ctx.materialize(lc);
        assert!(scalar_variables.insert(scalar, variable).is_none());
    };
    emit(ctx, DoryScalar::Evaluation, evaluation);
    let beta_inv: Vec<Lc> = betas.iter().map(|beta| ctx.inverse(beta)).collect();
    let alpha_inv: Vec<Lc> = alphas.iter().map(|alpha| ctx.inverse(alpha)).collect();
    let gamma_inv = ctx.inverse(&gamma);
    let d_inv = ctx.inverse(&d);
    let d_squared = ctx.mul(&d, &d);
    for (index, power) in rlc_powers.iter().enumerate() {
        let weight = ctx.mul(power, &beta_inv[0]);
        emit(ctx, DoryScalar::CommitmentWeight(index), &weight);
    }
    for j in 0..sigma {
        emit(ctx, DoryScalar::Beta(j), &betas[j]);
        emit(ctx, DoryScalar::BetaInv(j), &beta_inv[j]);
        emit(ctx, DoryScalar::Alpha(j), &alphas[j]);
        emit(ctx, DoryScalar::AlphaInv(j), &alpha_inv[j]);
    }
    emit(ctx, DoryScalar::Gamma, &gamma);
    emit(ctx, DoryScalar::GammaInv, &gamma_inv);
    emit(ctx, DoryScalar::D, &d);
    emit(ctx, DoryScalar::DInv, &d_inv);
    emit(ctx, DoryScalar::DSquared, &d_squared);
    emit(
        ctx,
        DoryScalar::D2Init,
        &(betas[0].clone() + d_squared.clone()),
    );

    let mut s1_acc = Lc::one();
    let mut s2_acc = Lc::one();
    for j in 0..sigma {
        let last = j + 1 == sigma;
        let u = if last {
            d_inv.clone()
        } else {
            beta_inv[j + 1].clone()
        };
        let v = if last {
            d.clone()
        } else {
            betas[j + 1].clone()
        };
        emit(ctx, DoryScalar::U(j), &u);
        emit(ctx, DoryScalar::V(j), &v);
        let u_alpha = ctx.mul(&u, &alphas[j]);
        emit(ctx, DoryScalar::UAlpha(j), &u_alpha);
        let v_alpha_inv = ctx.mul(&v, &alpha_inv[j]);
        emit(ctx, DoryScalar::VAlphaInv(j), &v_alpha_inv);
        let chi_left = ctx.mul(&u_alpha, &betas[j]);
        let chi_right = ctx.mul(&v_alpha_inv, &beta_inv[j]);
        emit(
            ctx,
            DoryScalar::Chi(sigma - 1 - j),
            &(Lc::one() + chi_left + chi_right),
        );
        let delta_1r = ctx.mul(&u, &betas[j]);
        emit(ctx, DoryScalar::Delta1R(sigma - 1 - j), &delta_1r);
        let delta_2r = ctx.mul(&v, &beta_inv[j]);
        emit(ctx, DoryScalar::Delta2R(sigma - 1 - j), &delta_2r);
        // Round j folds coordinate index σ − 1 − j.
        let coordinate = sigma - 1 - j;
        let y = &s1_coords[coordinate];
        let x = &s2_coords[coordinate];
        let s1_factor = ctx.mul(&alphas[j], &one_minus(y)) + y.clone();
        s1_acc = ctx.mul(&s1_acc, &s1_factor);
        let s2_factor = ctx.mul(&alpha_inv[j], &one_minus(x)) + x.clone();
        s2_acc = ctx.mul(&s2_acc, &s2_factor);
    }
    let ht = ctx.mul(&s1_acc, &s2_acc);
    emit(ctx, DoryScalar::Ht, &ht);
    let gamma_d_inv = ctx.mul(&gamma, &d_inv);
    let e1_scalar = ctx.mul(&gamma_d_inv, &s1_acc);
    emit(ctx, DoryScalar::PairingG2ZeroScalar, &-e1_scalar);
    let gamma_inv_d = ctx.mul(&gamma_inv, &d);
    let e2_scalar = ctx.mul(&gamma_inv_d, &s2_acc);
    emit(ctx, DoryScalar::PairingG1ZeroScalar, &-e2_scalar);

    let scalars = DoryScalar::link_order(sigma, rlc_powers.len())
        .into_iter()
        .map(|scalar| {
            let variable = scalar_variables
                .remove(&scalar)
                .unwrap_or_else(|| unreachable!("missing Dory scalar {scalar:?}"));
            (scalar, variable)
        })
        .collect();
    debug_assert!(scalar_variables.is_empty());
    Ok(DoryLinks {
        num_vars,
        sigma,
        scalars,
    })
}
