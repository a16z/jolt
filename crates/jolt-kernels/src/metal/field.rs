//! Host↔device field-element representation.
//!
//! [`jolt_field::Fr`] is `#[repr(transparent)]` over `ark_bn254::Fr`, whose
//! storage is its Montgomery-form `[u64; 4]` little-endian limbs. On this
//! little-endian target those bytes ARE the `[u32; 8]` little-endian limbs
//! the shaders operate on, so an `&[Fr]` is device-visible as
//! `device uint*` with zero conversion — the layout assertions and the
//! `fr_layout` parity test in [`super::testing`] pin this.
//!
//! The shader-side constants come from the [`MontgomeryConstants`] seam
//! (`jolt-field/src/montgomery_constants.rs`), formatted into an MSL
//! preamble by [`constants_preamble`] — no limb value is hardcoded in any
//! `.metal` source.

use jolt_field::{Fq, Fr, Limbs, MontgomeryConstants};

use super::runtime::{MAX_EVAL_POINTS, THREADGROUP_SIZE};

/// u32 limbs per field element (8 for BN254).
pub const FR_U32_LIMBS: usize = <Fr as MontgomeryConstants>::NUM_U32_LIMBS;

// The reinterpret-cast contract: one Fr is exactly its 8 u32 limbs, aligned
// at least as strictly as u32.
const _: () = assert!(size_of::<Fr>() == FR_U32_LIMBS * 4);
const _: () = assert!(align_of::<Fr>() >= align_of::<u32>());
const _: () = assert!(size_of::<Fr>() == <Fr as MontgomeryConstants>::FIELD_BYTE_SIZE);
// The shaders reuse FR_LIMBS for base-field values (g1.metal's Fq256).
const _: () = assert!(<Fq as MontgomeryConstants>::NUM_U32_LIMBS == FR_U32_LIMBS);

/// u32 stride between consecutive `ark_bn254::G1Affine` elements when their
/// backing memory is viewed as a `uint` array on the device (x limbs at +0,
/// y at +[`FR_U32_LIMBS`]; the trailing `infinity` flag is padding to the
/// shader). Pinned by the layout assertions below and the
/// `g1_affine_layout_matches_u32_view` test in [`super::testing`].
pub const G1_AFFINE_U32_STRIDE: usize = size_of::<ark_bn254::G1Affine>() / 4;

// The zero-copy G1Affine view contract: x at offset 0, y right behind it,
// whole struct a u32 multiple. `offset_of` makes field reordering (legal
// for repr(Rust)) a compile error instead of silent garbage.
const _: () = assert!(std::mem::offset_of!(ark_bn254::G1Affine, x) == 0);
const _: () = assert!(std::mem::offset_of!(ark_bn254::G1Affine, y) == FR_U32_LIMBS * 4);
const _: () = assert!(size_of::<ark_bn254::G1Affine>().is_multiple_of(4));
const _: () = assert!(align_of::<ark_bn254::G1Affine>() >= align_of::<u32>());

/// The generated MSL preamble: field constants off the
/// [`MontgomeryConstants`] seam plus the dispatch-geometry defines shared
/// with [`super::runtime`]. Prepended to every shader source before
/// `newLibraryWithSource` compilation.
pub(super) fn constants_preamble() -> String {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(1024);
    out.push_str("// Generated from jolt_field::MontgomeryConstants — never hand-edit limbs.\n");
    let _ = writeln!(out, "#define FR_LIMBS {FR_U32_LIMBS}u");
    let _ = writeln!(out, "#define JK_TG_SIZE {THREADGROUP_SIZE}u");
    let _ = writeln!(out, "#define JK_MAX_EVAL_POINTS {MAX_EVAL_POINTS}u");
    let _ = writeln!(out, "#define JK_G1_AFFINE_STRIDE {G1_AFFINE_U32_STRIDE}u");
    let _ = writeln!(
        out,
        "constant uint FR_MOD[FR_LIMBS] = {};",
        limb_array(Fr::modulus_u32())
    );
    let _ = writeln!(out, "constant uint FR_INV32 = {:#010x}u;", Fr::inv32());
    let _ = writeln!(
        out,
        "constant uint FQ_MOD[FR_LIMBS] = {};",
        limb_array(Fq::modulus_u32())
    );
    let _ = writeln!(out, "constant uint FQ_INV32 = {:#010x}u;", Fq::inv32());
    let _ = writeln!(
        out,
        "constant uint FQ_ONE[FR_LIMBS] = {};",
        limb_array(Fq::one_u32())
    );
    out
}

fn limb_array(limbs: &[u32]) -> String {
    let body: Vec<String> = limbs.iter().map(|l| format!("{l:#010x}u")).collect();
    format!("{{ {} }}", body.join(", "))
}

/// View a field-element slice as its device representation.
pub fn fr_as_u32s(elems: &[Fr]) -> &[u32] {
    // SAFETY: the layout assertions above pin size_of::<Fr>() == 8 * 4 with
    // alignment ≥ u32's, and every bit pattern is a valid u32, so the
    // reinterpretation is in-bounds and valid for the same lifetime.
    unsafe { std::slice::from_raw_parts(elems.as_ptr().cast::<u32>(), elems.len() * FR_U32_LIMBS) }
}

/// View a mutable field-element slice as its device representation.
///
/// Writing limbs ≥ p through this view produces a non-canonical element —
/// memory-safe, but arithmetic on it is wrong. Device kernels only ever
/// write canonical residues (see `fr.metal`).
pub fn fr_as_u32s_mut(elems: &mut [Fr]) -> &mut [u32] {
    // SAFETY: same layout argument as `fr_as_u32s`; the &mut borrow is
    // exclusive, so no aliasing is introduced.
    unsafe {
        std::slice::from_raw_parts_mut(elems.as_mut_ptr().cast::<u32>(), elems.len() * FR_U32_LIMBS)
    }
}

/// Rebuild a field element from device limbs (canonical Montgomery form).
pub fn fr_from_u32_limbs(limbs: &[u32; FR_U32_LIMBS]) -> Fr {
    let mut words = [0u64; 4];
    for (i, w) in words.iter_mut().enumerate() {
        *w = u64::from(limbs[2 * i]) | (u64::from(limbs[2 * i + 1]) << 32);
    }
    Fr::from_bigint_unchecked(Limbs(words))
}

/// A field element's device limbs (Montgomery form, LE u32).
pub fn fr_to_u32_limbs(x: Fr) -> [u32; FR_U32_LIMBS] {
    let words = x.inner_limbs().0;
    let mut limbs = [0u32; FR_U32_LIMBS];
    for (i, w) in words.iter().enumerate() {
        limbs[2 * i] = *w as u32;
        limbs[2 * i + 1] = (*w >> 32) as u32;
    }
    limbs
}
