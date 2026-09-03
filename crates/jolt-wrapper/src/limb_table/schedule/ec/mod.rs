//! The G1 and G2 lanes: affine Straus chains over the flattened check's
//! MSMs with transcript-randomized offsets.
//!
//! Every main chain starts at `R = θ·G` and reads table entries `(d − 8)·P +
//! Z0` with `Z0 = φ(R) = θ·φ(G)`, `φ` the GLV endomorphism (`φ(G) = [λ]G`),
//! `θ` the wrapper's offset challenge drawn after the phase-1a commitments
//! (the points `P` and the digits of the proof scalars among them). The
//! offsets are removed by one extra base `−K` per chain with scalar `θ`,
//! `K = (16^64 + n·(16^64 − 1)/15·λ)·G` (`n` bases including itself), so
//! no correction depends on the prover. Processing order within a window:
//! four doublings, the proof bases, the correction base last.
//!
//! Exceptional affine cases. An add `acc + entry` with `x_entry = x_acc` is
//! either `entry = acc` (the slope pin `λ·(x2 − x1) = y2 − y1` is vacuous and
//! the output unconstrained) or `entry = −acc` (the pin reads `0 = −2·y1`:
//! no witness). The correction base's adds carry the completeness guard
//! ([`super::super::ops::g1_add_guard`]: `inv·(x2 − x1) = 1` from the add's
//! own rows), so they admit no exceptional witness at all. For a proof-base
//! add of window `w`, base `k`, the accumulator is `θ·A·G − 16·k_K·P_w·G +
//! H` with `A = A_{w,k} = 16^{w+1} + λ·(n·(16^{w+1} − 16)/15 + k)`; before
//! the window's doublings it is `θ·A_w·G − k_K·P_w·G + H` with `A_w = 16^w +
//! λ·n·(16^w − 1)/15` (the consumed correction prefix not yet doubled).
//! `K = k_K·G`, `P_w` is the integer formed by the `w` digits of `θ` the
//! correction base has consumed, and `H` the partial MSM, fixed before `θ`.
//! Writing `θ = 16^{64−w}·P_w + S_w`, `acc = ±entry` (or `acc = 0`) is
//! `(A ∓ λ)·S_w + ((A ∓ λ)·16^{64−w} − 16·k_K)·P_w ≡ c` for an add and
//! `A_w·S_w + (A_w·16^{64−w} − k_K)·P_w ≡ c` for a doubling, `c` fixed:
//! for each of the `≤ 16^w·16/15` prefixes there is at most one `S_w`, and
//! for each of the `≤ 16^{64−w}·16/15` suffixes at most one prefix, so at
//! most `2^129` values of `θ` are exceptional at any site, `2^{−125}` of
//! them, provided the two coefficients are nonzero —
//! [`tests::offsets_are_nondegenerate`] sweeps every `(w, k, n ≤ 64)`. The
//! one zero it finds is the last add's `acc = −entry`, i.e. a zero MSM, for
//! every `θ`: that case has no witness (an honest zero MSM has probability
//! `1/r`). Table entries `(d ∓ 1)·P + Z0 ± P` degenerate only when `θ·λ·G`
//! equals a fixed point, one root each; neither curve has 2-torsion.
//!
//! `R` itself is a fixed-base Straus chain of `θ` over `G` with constant
//! offsets `R'' = G`, `Z'' = 9G` and a constant correction: the accumulator
//! multiplier before the add of window `w` is the integer
//! `16^{w+1} + Σ_{i<w} (d_i + 9)·16^{w−i} ∈ (16^{w+1}, 2.07·16^{w+1})` and
//! the entry multiplier `d_w + 9 ∈ [1, 16]`, so `acc = ±entry` needs a wrap
//! modulo `r`, impossible below `2^254` (`w ≤ 62`) and a single-residue
//! event on the last window (`≤ 32` digit strings) and on the correction
//! (`θ ∈ {0, −2·(16^64 + 9(16^64 − 1)/15)}`). The ψ-chain adds (prover-chosen
//! operands, no `θ`) are guarded as well; the Miller-loop adds pair the
//! constant `Q` bases with points inside `G2`.

use ark_bn254::{
    g1::Config as G1Config, g2::Config as G2Config, Config as Bn254Config, Fq, G1Affine, G2Affine,
};
use ark_ec::bn::BnConfig;
use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ec::short_weierstrass::SWCurveConfig;
use ark_ec::{AffineRepr, CurveGroup};
use num_bigint::BigUint;
use std::collections::HashMap;
use std::ops::Range;

use super::super::digits::{digits, WINDOWS};
use super::super::dory::{FlattenedCheck, G1Base, G2Base, InputElement, Wire, WireValues};
use super::super::layout::{Bits, Factor, Rel};
use super::super::ops::{
    g1_add, g1_add_guard, g1_copy, g1_dbl, g1_endo, g1_on_curve, g1_sign, g2_add, g2_add_guard,
    g2_add_guarded, g2_copy, g2_dbl, g2_endo, g2_negation_pins, g2_on_curve, g2_psi, g2_sign,
    psi_coefficients,
};
use super::super::program::{half_plus_one, RowId};
use super::super::relation::{FP_SLOTS_G1, FP_SLOTS_G2};
use super::super::template::{DigitRule, ElemRel, Family, Template};
use super::super::wiring::ReadKind;
use super::{
    half_row, hi, naf, row, Builder, Cells, DigitOp, DorySetupInputs, KeyBase, SelectedFamily, B1,
    B1I, B2, B2I, C, C3, CELL, HI1I, HI1T, HI2, HI2I, HI2T, J1, J2, K1, K2, KM1, LOG_ROWS, M1, W1,
    W2,
};

mod g1;
mod g2;
mod psi;
mod sign;

/// The fixed-base chain's table offset `Z'' = 9·G` (every entry `d + 9 ≥ 1`).
const FIXED_TABLE_OFFSET: u64 = 9;

/// `(16^64 − 1)/15`: the offset count a base's 64 window adds accumulate to.
fn window_sum() -> BigUint {
    ((BigUint::from(1u32) << (4 * WINDOWS)) - BigUint::from(1u32)) / BigUint::from(15u32)
}

fn scale<A: AffineRepr>(point: A, scalar: &BigUint) -> A::Group {
    point.mul_bigint(scalar.to_u64_digits())
}

/// `−K = −(16^64·R'' + n·(16^64 − 1)/15·Z'')`: what a chain of `n` bases
/// accumulates from its offsets, negated.
fn offset_correction<A: AffineRepr>(r: A, z: A, bases: usize) -> A {
    let sixteen_pow = BigUint::from(1u32) << (4 * WINDOWS);
    let total = scale(r, &sixteen_pow) + scale(z, &(window_sum() * BigUint::from(bases)));
    (-total).into_affine()
}

/// A base of a chain: one of the check's MSM bases or a public constant point.
enum Operand<B, A> {
    Base(B),
    Constant(A),
}

/// One chain: bases with scalar wires, first `k` slot, accumulator start rows,
/// table offset rows, the constant correction of a fixed-base chain and
/// whether the last base's adds carry the completeness guard (the main
/// chains' `θ`-digit correction base).
struct Chain<B, A> {
    bases: Vec<(Operand<B, A>, Wire)>,
    kbase: u32,
    init: Vec<RowId>,
    z0: Vec<RowId>,
    correction: Option<A>,
    guard: bool,
}

impl<B, A> Chain<B, A> {
    /// `k` slots used: bases, four doublings, the correction, the guard.
    fn slots(&self) -> u32 {
        self.bases.len() as u32 + 4 + u32::from(self.correction.is_some()) + u32::from(self.guard)
    }
}

/// Shared state of a lane's chains: the next table index, the first cell of
/// every input element and the fresh inputs' table indices.
struct Lane {
    table_base: u32,
    first_input: HashMap<InputElement, u32>,
    /// Fresh inputs (table index for G1, half cell for G2) and their elements.
    fresh: Vec<u32>,
    fresh_elements: Vec<InputElement>,
    acc_output: Option<u32>,
}

impl Lane {
    fn new() -> Self {
        Self {
            table_base: 0,
            first_input: HashMap::new(),
            fresh: Vec::new(),
            fresh_elements: Vec::new(),
            acc_output: None,
        }
    }
}

/// The G1 templates of a lane.
struct G1Templates {
    copy: Template,
    copy_neg: Template,
    add: Template,
    sub: Template,
    dbl: Template,
    guard: Template,
}

impl G1Templates {
    fn new() -> Self {
        Self {
            copy: g1_copy(false),
            copy_neg: g1_copy(true),
            add: g1_add(false),
            sub: g1_add(true),
            dbl: g1_dbl(),
            guard: g1_add_guard(),
        }
    }
}

/// The G2 templates of a lane.
struct G2Templates {
    copy: Template,
    copy_neg: Template,
    add: Template,
    sub: Template,
    dbl: Template,
    guard: Template,
}

impl G2Templates {
    fn new() -> Self {
        Self {
            copy: g2_copy(false),
            copy_neg: g2_copy(true),
            add: g2_add(false),
            sub: g2_add(true),
            dbl: g2_dbl(),
            guard: g2_add_guard(),
        }
    }
}

// ----- G2 subgroup checks (ψ-chains) ----------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::{BigInteger, Field, One, PrimeField, Zero};

    fn modulus<A: AffineRepr>() -> BigUint {
        BigUint::from_bytes_be(&<A::ScalarField as PrimeField>::MODULUS.to_bytes_be())
    }

    /// `φ(G) = [λ]G` on both groups (the endomorphism the offsets rely on).
    #[test]
    fn glv_endomorphism_is_lambda() {
        let g1 = G1Affine::generator();
        assert_eq!(
            G1Config::endomorphism_affine(&g1),
            g1.mul_bigint(G1Config::LAMBDA.into_bigint()).into_affine()
        );
        let g2 = G2Affine::generator();
        assert_eq!(
            G2Config::endomorphism_affine(&g2),
            g2.mul_bigint(G2Config::LAMBDA.into_bigint()).into_affine()
        );
    }

    /// No `(window, add, chain size)` of a main chain makes an exceptional
    /// affine case θ-independent (module doc): both `A ∓ λ` and
    /// `(A ∓ λ)·16^{64−w} − 16·k_K` are nonzero for every add, and `A_w`,
    /// `A_w·16^{64−w} − k_K` for every doubling, `w < 64`, `k < n ≤ 64` —
    /// except the last add's `acc = −entry`, the zero MSM.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test")]
    fn offsets_are_nondegenerate() {
        let sixteen = Fr::from(16u64);
        let sixteen_pow = sixteen.pow([WINDOWS as u64]);
        let window_sum = Fr::from(window_sum());
        let fifteen_inverse = Fr::from(15u64).inverse().unwrap();
        for lambda in [G1Config::LAMBDA, G2Config::LAMBDA] {
            assert!(!lambda.is_zero());
            for n in 2..=64u64 {
                let k_k = sixteen_pow + Fr::from(n) * window_sum * lambda;
                let offsets = |power: Fr| Fr::from(n) * (power - Fr::one()) * fifteen_inverse;
                let mut power = Fr::one();
                for w in 0..WINDOWS as u64 {
                    let suffix = sixteen.pow([WINDOWS as u64 - w]);
                    let a_dbl = power + lambda * offsets(power);
                    assert_ne!(a_dbl, Fr::zero(), "doubling n={n} w={w}");
                    assert_ne!(
                        a_dbl * suffix - k_k,
                        Fr::zero(),
                        "doubling slope n={n} w={w}"
                    );
                    let before_window = power * sixteen;
                    for k in 0..n {
                        let a = before_window + lambda * (offsets(power) * sixteen + Fr::from(k));
                        for sign in [Fr::one(), -Fr::one()] {
                            let coefficient = a - sign * lambda;
                            let slope = coefficient * suffix - sixteen * k_k;
                            assert_ne!(coefficient, Fr::zero(), "add n={n} w={w} k={k}");
                            if w + 1 == WINDOWS as u64 && k + 1 == n && sign == -Fr::one() {
                                // The last add: `acc = −entry` is the zero MSM,
                                // θ-independent by construction (no witness).
                                assert_eq!(coefficient, k_k, "zero MSM n={n}");
                                assert_eq!(slope, Fr::zero(), "zero MSM slope n={n}");
                                continue;
                            }
                            assert_ne!(slope, Fr::zero(), "add slope n={n} w={w} k={k}");
                        }
                    }
                    power *= sixteen;
                }
            }
        }
    }

    /// The G2 membership identity the ψ-chains pin: `ψ²(P) + ψ([6x+3]P) +
    /// [6x+1]P = 0` on the subgroup, false on a random point of the twist's
    /// cofactor torsion.
    #[test]
    fn psi_identity_holds_on_g2_only() {
        use super::super::super::ops::psi_coefficients;
        use ark_bn254::Fq2;
        use ark_ff::UniformRand;
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;
        let psi = |p: G2Affine, power: usize| -> G2Affine {
            let (cx, cy) = psi_coefficients(power);
            let conj = |mut v: Fq2| {
                for _ in 0..power {
                    let _ = v.conjugate_in_place();
                }
                v
            };
            G2Affine::new_unchecked(conj(p.x) * cx, conj(p.y) * cy)
        };
        let x = u128::from(<ark_bn254::Config as ark_ec::bn::BnConfig>::X[0]);
        let limbs = |s: u128| [s as u64, (s >> 64) as u64];
        let check = |p: G2Affine| {
            let a = p.mul_bigint(limbs(6 * x + 3)).into_affine();
            let b = p.mul_bigint(limbs(6 * x + 1)).into_affine();
            (psi(p, 2).into_group() + psi(a, 1).into_group() + b.into_group()).is_zero()
        };
        let mut rng = ChaCha20Rng::seed_from_u64(0x9D1);
        for _ in 0..4 {
            assert!(check(G2Affine::rand(&mut rng)));
        }
        // A point of the twist outside G2: random x, solve for y, no cofactor clearing.
        let outside = loop {
            let x = Fq2::rand(&mut rng);
            if let Some(p) = G2Affine::get_point_from_x_unchecked(x, true) {
                if !p.is_in_correct_subgroup_assuming_on_curve() {
                    break p;
                }
            }
        };
        assert!(!check(outside));
    }

    /// The fixed-base chain's integer regime: the accumulator multiplier of
    /// window `w ≤ 62` stays below `r`, so no wrap can align it with an entry.
    #[test]
    fn fixed_base_multipliers_stay_below_the_modulus() {
        let r = modulus::<G1Affine>();
        let sixteen = BigUint::from(16u32);
        // Largest multiplier before the add of window 62: every digit `7`.
        let mut multiplier = BigUint::one();
        for _ in 0..62 {
            multiplier = &multiplier * &sixteen + BigUint::from(7u32 + FIXED_TABLE_OFFSET as u32);
        }
        multiplier *= &sixteen;
        assert!(multiplier < r);
        assert_eq!(
            offset_correction(G1Affine::generator(), G1Affine::generator(), 1),
            (-(scale(G1Affine::generator(), &(BigUint::one() << 256))
                + scale(G1Affine::generator(), &window_sum())))
            .into_affine()
        );
    }
}
