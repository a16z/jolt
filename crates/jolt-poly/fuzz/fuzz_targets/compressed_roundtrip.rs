#![no_main]

//! Compression round-trip for sumcheck round polynomials: dropping the linear
//! coefficient and recovering it from the hint `h = p(0) + p(1)` must
//! reproduce the original polynomial and its evaluations. This is the wire
//! transformation every compressed sumcheck round goes through.

use jolt_field::{Fr, FromPrimitiveInt, ReducingBytes};
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use libfuzzer_sys::fuzz_target;

const SCALAR_BYTES: usize = 32;
const MAX_DEGREE: usize = 6;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    // A compressible polynomial needs at least the constant and linear terms.
    let coeff_count = (data[0] as usize % MAX_DEGREE) + 2; // 2..=7
    if data.len() < 1 + (coeff_count + 1) * SCALAR_BYTES {
        return;
    }
    let scalar_at = |index: usize| {
        let start = 1 + index * SCALAR_BYTES;
        <Fr as ReducingBytes>::from_le_bytes_mod_order(&data[start..start + SCALAR_BYTES])
    };
    let coefficients: Vec<Fr> = (0..coeff_count).map(scalar_at).collect();
    let x = scalar_at(coeff_count);

    let poly = UnivariatePoly::new(coefficients);
    let hint = poly.evaluate(Fr::from_u64(0)) + poly.evaluate(Fr::from_u64(1));
    let compressed = poly.compress();

    assert_eq!(
        compressed.evaluate_with_hint(hint, x),
        poly.evaluate(x),
        "hint-based evaluation disagrees with the uncompressed polynomial"
    );
    assert_eq!(
        compressed.decompress(hint),
        poly,
        "decompress did not recover the original polynomial"
    );
    assert_eq!(
        compressed.degree(),
        poly.degree(),
        "compression changed the polynomial degree"
    );
});
