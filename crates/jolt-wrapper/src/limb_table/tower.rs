//! BN254 tower coordinates: an `Fq12` element as twelve `Fq` coordinates
//! (`c[6h + 2v + t]` = `Fq12.c{h}.c{v}.c{t}`), plus the bilinear multiplication
//! form and the Frobenius matrices, both probed from arkworks' tower
//! arithmetic so every row formula has arkworks as its single owner.

use std::sync::OnceLock;

use ark_bn254::{Fq, Fq12, Fq2, Fq6};
use ark_ff::{Field, One, Zero};

pub const FQ12_COORDS: usize = 12;

/// Coordinate positions of a `mul_by_034` line element `(c0, c3, c4)`:
/// `Fq12.c0.c0`, `Fq12.c1.c0`, `Fq12.c1.c1`.
pub const LINE_COORDS: [[usize; 2]; 3] = [[0, 1], [6, 7], [8, 9]];

pub fn fq2_coords(x: &Fq2) -> [Fq; 2] {
    [x.c0, x.c1]
}

pub fn fq2_from_coords(c: [Fq; 2]) -> Fq2 {
    Fq2::new(c[0], c[1])
}

pub fn fq12_coords(x: &Fq12) -> [Fq; FQ12_COORDS] {
    let mut out = [Fq::zero(); FQ12_COORDS];
    for (h, c6) in [x.c0, x.c1].iter().enumerate() {
        for (v, c2) in [c6.c0, c6.c1, c6.c2].iter().enumerate() {
            out[6 * h + 2 * v] = c2.c0;
            out[6 * h + 2 * v + 1] = c2.c1;
        }
    }
    out
}

pub fn fq12_from_coords(c: &[Fq; FQ12_COORDS]) -> Fq12 {
    let fq6 = |h: usize| {
        Fq6::new(
            Fq2::new(c[6 * h], c[6 * h + 1]),
            Fq2::new(c[6 * h + 2], c[6 * h + 3]),
            Fq2::new(c[6 * h + 4], c[6 * h + 5]),
        )
    };
    Fq12::new(fq6(0), fq6(1))
}

/// One product term `kappa · x[a] · y[b]` of an output coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Term {
    pub a: u8,
    pub b: u8,
    pub kappa: i8,
}

/// Small-integer value of a field element produced by tower arithmetic on
/// basis vectors (only `0, ±1, ±9` and their small sums occur).
fn small_int(x: Fq) -> Option<i8> {
    (-64i8..=64).find(|&k| {
        let candidate = if k < 0 {
            -Fq::from(k.unsigned_abs() as u64)
        } else {
            Fq::from(k as u64)
        };
        candidate == x
    })
}

fn basis(i: usize) -> Fq12 {
    let mut c = [Fq::zero(); FQ12_COORDS];
    c[i] = Fq::one();
    fq12_from_coords(&c)
}

/// `mul_form()[c]`: the terms of coordinate `c` of the product `x · y`.
pub fn mul_form() -> &'static [Vec<Term>; FQ12_COORDS] {
    static FORM: OnceLock<[Vec<Term>; FQ12_COORDS]> = OnceLock::new();
    FORM.get_or_init(|| {
        let mut form: [Vec<Term>; FQ12_COORDS] = Default::default();
        for a in 0..FQ12_COORDS {
            for b in 0..FQ12_COORDS {
                let product = fq12_coords(&(basis(a) * basis(b)));
                for (c, value) in product.iter().enumerate() {
                    if value.is_zero() {
                        continue;
                    }
                    let kappa = small_int(*value).unwrap_or_else(|| {
                        unreachable!("tower product of basis vectors is a small integer")
                    });
                    form[c].push(Term {
                        a: a as u8,
                        b: b as u8,
                        kappa,
                    });
                }
            }
        }
        form
    })
}

/// Per output coordinate, the `(a, constant)` pairs of a Frobenius power.
pub type FrobeniusForm = [Vec<(u8, Fq)>; FQ12_COORDS];

/// `frobenius_form(p)[c]`: `Frob^p(x)[c] = Σ constant · x[a]` as `(a, constant)`
/// pairs; the constants are full `Fq` values (Frobenius coefficients of the
/// tower), so a Frobenius output coordinate is a row with public-constant operands.
pub fn frobenius_form(power: usize) -> &'static FrobeniusForm {
    static FORMS: OnceLock<[FrobeniusForm; 4]> = OnceLock::new();
    &FORMS.get_or_init(|| {
        std::array::from_fn(|power| {
            let mut form: FrobeniusForm = Default::default();
            for a in 0..FQ12_COORDS {
                let mut image = basis(a);
                image.frobenius_map_in_place(power);
                for (c, value) in fq12_coords(&image).iter().enumerate() {
                    if !value.is_zero() {
                        form[c].push((a as u8, *value));
                    }
                }
            }
            form
        })
    })[power % 4]
}

/// Coordinates whose sign flips under conjugation (`Fq12.c1` negated): the
/// cyclotomic inverse of a unitary element.
pub fn conjugated(coord: usize) -> bool {
    coord >= 6
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests may fail loudly")]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn apply_mul_form(x: &[Fq; 12], y: &[Fq; 12]) -> [Fq; 12] {
        let mut out = [Fq::zero(); 12];
        for (c, terms) in mul_form().iter().enumerate() {
            for term in terms {
                let kappa = Fq::from(term.kappa.unsigned_abs() as u64);
                let product = x[term.a as usize] * y[term.b as usize] * kappa;
                if term.kappa < 0 {
                    out[c] -= product;
                } else {
                    out[c] += product;
                }
            }
        }
        out
    }

    #[test]
    fn mul_form_matches_arkworks() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        for _ in 0..4 {
            let x = Fq12::rand(&mut rng);
            let y = Fq12::rand(&mut rng);
            let z = apply_mul_form(&fq12_coords(&x), &fq12_coords(&y));
            assert_eq!(fq12_from_coords(&z), x * y);
        }
        let max_terms = mul_form().iter().map(Vec::len).max().unwrap();
        assert!(max_terms <= 24, "{max_terms}");
    }

    #[test]
    fn frobenius_form_matches_arkworks() {
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let x = Fq12::rand(&mut rng);
        let coords = fq12_coords(&x);
        for power in 1..4 {
            let mut expected = x;
            expected.frobenius_map_in_place(power);
            let mut out = [Fq::zero(); 12];
            for (c, terms) in frobenius_form(power).iter().enumerate() {
                for (a, constant) in terms {
                    out[c] += coords[*a as usize] * constant;
                }
            }
            assert_eq!(fq12_from_coords(&out), expected);
        }
    }

    #[test]
    fn conjugation_is_cyclotomic_inverse_on_pairing_outputs() {
        use ark_ec::pairing::Pairing;
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        let g1 = ark_bn254::G1Projective::rand(&mut rng);
        let g2 = ark_bn254::G2Projective::rand(&mut rng);
        let gt = ark_bn254::Bn254::pairing(g1, g2).0;
        let mut coords = fq12_coords(&gt);
        for (c, value) in coords.iter_mut().enumerate() {
            if conjugated(c) {
                *value = -*value;
            }
        }
        assert_eq!(fq12_from_coords(&coords), gt.inverse().unwrap());
    }
}
