use super::commitment_scheme::CommitmentScheme;
use crate::{
    field::JoltField,
    msm::{Icicle, VariableBaseMSM},
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        sparse_matrix_polynomial::get_num_columns,
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        transcript::{AppendToTranscript, Transcript},
    },
};
use ark_bn254::{Bn254, Fr, G1Projective, G2Projective};
use ark_ec::bn::{G1Prepared, G2Prepared};
type BnG1Prepared = G1Prepared<ark_bn254::Config>;
type BnG2Prepared = G2Prepared<ark_bn254::Config>;
use ark_ec::{
    pairing::{MillerLoopOutput, Pairing as ArkPairing, PairingOutput},
    CurveGroup,
};
use ark_ff::{Field, One, PrimeField, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{rand::RngCore, Zero};
use rayon::prelude::*;
use std::{borrow::Borrow, marker::PhantomData};

use dory::{
    arithmetic::{
        Field as DoryField, Group as DoryGroup, MultiScalarMul as DoryMultiScalarMul,
        Pairing as DoryPairing,
    },
    commit, evaluate, setup_with_srs_file,
    transcript::Transcript as DoryTranscript,
    verify, DoryProof, DoryProofBuilder, Polynomial as DoryPolynomial, ProverSetup, VerifierSetup,
};

// NewType wrappers for Jolt + arkworks types to interop with Dory traits
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltFieldWrapper<F: JoltField>(pub F);

impl<F: JoltField> DoryField for JoltFieldWrapper<F> {
    fn zero() -> Self {
        JoltFieldWrapper(F::zero())
    }

    fn one() -> Self {
        JoltFieldWrapper(F::one())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn add(&self, rhs: &Self) -> Self {
        JoltFieldWrapper(self.0 + rhs.0)
    }

    fn sub(&self, rhs: &Self) -> Self {
        JoltFieldWrapper(self.0 - rhs.0)
    }

    fn mul(&self, rhs: &Self) -> Self {
        JoltFieldWrapper(self.0 * rhs.0)
    }

    fn inv(&self) -> Option<Self> {
        self.0.inverse().map(JoltFieldWrapper)
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        JoltFieldWrapper(F::random(rng))
    }

    fn from_u64(val: u64) -> Self {
        JoltFieldWrapper(F::from_u64(val))
    }

    fn from_i64(val: i64) -> Self {
        JoltFieldWrapper(F::from_i64(val))
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltGroupWrapper<G: CurveGroup>(pub G);

impl<G> DoryGroup for JoltGroupWrapper<G>
where
    G: CurveGroup + VariableBaseMSM,
    G::ScalarField: JoltField,
{
    type Scalar = JoltFieldWrapper<G::ScalarField>;

    fn identity() -> Self {
        Self(G::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0 + rhs.0)
    }

    fn neg(&self) -> Self {
        Self(-self.0)
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        Self(self.0 * k.0)
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self(G::rand(rng))
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltGTWrapper<P: ArkPairing>(pub P::TargetField);

impl<P: ArkPairing> From<PairingOutput<P>> for JoltGTWrapper<P> {
    fn from(value: PairingOutput<P>) -> Self {
        Self(value.0)
    }
}

impl<P: ArkPairing> Into<PairingOutput<P>> for JoltGTWrapper<P> {
    fn into(self) -> PairingOutput<P> {
        PairingOutput(self.0)
    }
}

impl<P> DoryGroup for JoltGTWrapper<P>
where
    P: ArkPairing,
    P::ScalarField: JoltField,
{
    type Scalar = JoltFieldWrapper<P::ScalarField>;

    fn identity() -> Self {
        Self(P::TargetField::one())
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0 * rhs.0)
    }

    fn neg(&self) -> Self {
        Self(self.0.inverse().expect("GT element should be invertible"))
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        Self(self.0.pow(k.0.into_bigint()))
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self(P::TargetField::rand(rng))
    }
}

impl<P: ArkPairing> Default for JoltGTWrapper<P> {
    fn default() -> Self {
        Self(P::TargetField::one())
    }
}

impl<P> std::iter::Sum for JoltGTWrapper<P>
where
    P: ArkPairing,
    P::ScalarField: JoltField,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::identity(), |acc, x| acc.add(&x))
    }
}

// Specialized MSM implementations for G1 and G2 to leverage `jolt-optimizations` in arkworks
pub struct JoltMsmG1;
pub struct JoltMsmG2;
pub struct JoltMSM;

// G1 MSM implementation with jolt-optimizations
impl DoryMultiScalarMul<JoltGroupWrapper<G1Projective>> for JoltMsmG1 {
    fn msm(
        bases: &[JoltGroupWrapper<G1Projective>],
        scalars: &[JoltFieldWrapper<Fr>],
    ) -> JoltGroupWrapper<G1Projective> {
        // # Safety
        // JoltGroupWrapper always has same memory layout as underlying G1Projective here.
        let projective_points: &[G1Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G1Projective, bases.len())
        };
        let affines = G1Projective::normalize_batch(projective_points);

        // # Safety
        // JoltFieldWrapper always has same memory layout as underlying Fr here.
        let raw_scalars: &[Fr] =
            unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, scalars.len()) };

        let result = G1Projective::msm_field_elements(&affines, None, raw_scalars, None, false)
            .expect("msm_field_elements should not fail");

        JoltGroupWrapper(result)
    }

    fn fixed_base_vector_msm(
        base: &JoltGroupWrapper<G1Projective>,
        scalars: &[JoltFieldWrapper<Fr>],
        _g1_cache: Option<&dory::curve::G1Cache>,
        _g2_cache: Option<&dory::curve::G2Cache>,
    ) -> Vec<JoltGroupWrapper<G1Projective>> {
        if scalars.is_empty() {
            return vec![];
        }

        // # Safety
        // JoltFieldWrapper always has same memory layout as underlying Fr
        let raw_scalars: &[Fr] =
            unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, scalars.len()) };

        // Use `jolt-optimizations`` fixed_base_vector_msm_g1
        let results_proj = jolt_optimizations::fixed_base_vector_msm_g1(&base.0, raw_scalars);

        results_proj.into_iter().map(JoltGroupWrapper).collect()
    }

    fn fixed_scalar_variable_with_add(
        bases: &[JoltGroupWrapper<G1Projective>],
        vs: &mut [JoltGroupWrapper<G1Projective>],
        scalar: &JoltFieldWrapper<Fr>,
    ) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must have same length");

        // # Safety
        // JoltGroupWrapper is repr(transparent) so has same memory layout as G1Projective
        let vs_proj: &mut [G1Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G1Projective, vs.len())
        };
        let bases_proj: &[G1Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G1Projective, bases.len())
        };

        // Use `jolt-optimizations`: v[i] = v[i] + scalar * generators[i]
        jolt_optimizations::vector_add_scalar_mul_g1_online(vs_proj, bases_proj, scalar.0);
    }

    fn fixed_scalar_variable_with_add_cached(
        bases_count: usize,
        g1_cache: Option<&dory::curve::G1Cache>,
        _g2_cache: Option<&dory::curve::G2Cache>,
        vs: &mut [JoltGroupWrapper<G1Projective>],
        scalar: &JoltFieldWrapper<Fr>,
    ) {
        assert_eq!(bases_count, vs.len(), "bases_count must equal vs length");

        if let Some(cache) = g1_cache {
            // Get precomputed data slice for zero-copy access
            // Assuming get_precomputed_slice now returns windowed data
            let precomputed = cache.get_windowed_data().expect("Cache data should exist");

            let subset_data = jolt_optimizations::Windowed2Signed2Data {
                windowed2_tables: precomputed.windowed2_tables[..bases_count].to_vec(),
            };

            // # Safety
            // JoltGroupWrapper is repr(transparent) so has same memory layout as G1Projective
            let vs_proj: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G1Projective, vs.len())
            };

            // Use memory-efficient windowed2_signed method
            jolt_optimizations::vector_add_scalar_mul_g1_windowed2_signed(
                vs_proj,
                scalar.0,
                &subset_data,
            );
        } else {
            panic!("G1 cache not available for cached operation");
        }
    }

    fn fixed_scalar_scale_with_add(
        vs: &mut [JoltGroupWrapper<G1Projective>],
        addends: &[JoltGroupWrapper<G1Projective>],
        scalar: &JoltFieldWrapper<Fr>,
    ) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have same length"
        );

        // # Safety
        // JoltGroupWrapper is repr(transparent) so has same memory layout as G1Projective
        let vs_proj: &mut [G1Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G1Projective, vs.len())
        };
        let addends_proj: &[G1Projective] = unsafe {
            std::slice::from_raw_parts(addends.as_ptr() as *const G1Projective, addends.len())
        };

        // Use `jolt-optimizations`: v[i] = scalar * v[i] + gamma[i]
        jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(vs_proj, scalar.0, addends_proj);
    }
}

// G2 MSM implementation with jolt-optimizations
impl DoryMultiScalarMul<JoltGroupWrapper<G2Projective>> for JoltMsmG2 {
    fn msm(
        bases: &[JoltGroupWrapper<G2Projective>],
        scalars: &[JoltFieldWrapper<Fr>],
    ) -> JoltGroupWrapper<G2Projective> {
        let projective_points: Vec<G2Projective> = bases.iter().map(|w| w.0).collect();
        let affines = G2Projective::normalize_batch(&projective_points);

        // # Safety
        // JoltFieldWrapper always has same memory layout as underlying Fr here.
        let raw_scalars: &[Fr] =
            unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, scalars.len()) };

        let result = G2Projective::msm_field_elements(&affines, None, raw_scalars, None, false)
            .expect("msm_field_elements should not fail");

        JoltGroupWrapper(result)
    }

    fn fixed_base_vector_msm(
        base: &JoltGroupWrapper<G2Projective>,
        scalars: &[JoltFieldWrapper<Fr>],
        _g1_cache: Option<&dory::curve::G1Cache>,
        g2_cache: Option<&dory::curve::G2Cache>,
    ) -> Vec<JoltGroupWrapper<G2Projective>> {
        if scalars.is_empty() {
            return vec![];
        }

        // # Safety
        // JoltFieldWrapper always has same memory layout as underlying Fr
        let raw_scalars: &[Fr] =
            unsafe { std::slice::from_raw_parts(scalars.as_ptr() as *const Fr, scalars.len()) };

        // Check if we have cached GLV tables for g_fin
        if let Some(glv_tables) = g2_cache.and_then(|cache| cache.get_g_fin_glv_tables()) {
            // Use precomputed GLV tables
            let results_proj: Vec<G2Projective> = raw_scalars
                .par_iter()
                .map(|&scalar| {
                    // glv_four_scalar_mul returns a vector, we need the first element
                    jolt_optimizations::glv_four_scalar_mul(glv_tables, scalar)[0]
                })
                .collect();

            results_proj.into_iter().map(JoltGroupWrapper).collect()
        } else {
            // Fall back to online computation
            let base_proj = base.0;
            let results_proj: Vec<G2Projective> = raw_scalars
                .par_iter()
                .map(|&scalar| {
                    jolt_optimizations::glv_four_scalar_mul_online(scalar, &[base_proj])[0]
                })
                .collect();

            results_proj.into_iter().map(JoltGroupWrapper).collect()
        }
    }

    fn fixed_scalar_variable_with_add(
        bases: &[JoltGroupWrapper<G2Projective>],
        vs: &mut [JoltGroupWrapper<G2Projective>],
        scalar: &JoltFieldWrapper<Fr>,
    ) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must have same length");

        // # Safety
        // JoltGroupWrapper is repr(transparent) so has same memory layout as G2Projective
        let vs_proj: &mut [G2Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G2Projective, vs.len())
        };
        let bases_proj: &[G2Projective] = unsafe {
            std::slice::from_raw_parts(bases.as_ptr() as *const G2Projective, bases.len())
        };

        // Use `jolt-optimizations`: v[i] = v[i] + scalar * generators[i]
        jolt_optimizations::vector_add_scalar_mul_g2_online(vs_proj, bases_proj, scalar.0);
    }

    fn fixed_scalar_variable_with_add_cached(
        bases_count: usize,
        _g1_cache: Option<&dory::curve::G1Cache>,
        g2_cache: Option<&dory::curve::G2Cache>,
        vs: &mut [JoltGroupWrapper<G2Projective>],
        scalar: &JoltFieldWrapper<Fr>,
    ) {
        assert_eq!(bases_count, vs.len(), "bases_count must equal vs length");

        if let Some(cache) = g2_cache {
            let precomputed = cache.get_windowed_data().expect("Cache data should exist");

            let subset_data = jolt_optimizations::Windowed2Signed4Data {
                windowed2_tables: precomputed.windowed2_tables[..bases_count].to_vec(),
            };

            // # Safety
            // JoltGroupWrapper is repr(transparent) so has same memory layout as G2Projective
            let vs_proj: &mut [G2Projective] = unsafe {
                std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G2Projective, vs.len())
            };

            // Use memory-efficient windowed2_signed method
            jolt_optimizations::vector_add_scalar_mul_g2_windowed2_signed(
                vs_proj,
                scalar.0,
                &subset_data,
            );
        } else {
            panic!("G2 cache not available for cached operation");
        }
    }

    fn fixed_scalar_scale_with_add(
        vs: &mut [JoltGroupWrapper<G2Projective>],
        addends: &[JoltGroupWrapper<G2Projective>],
        scalar: &JoltFieldWrapper<Fr>,
    ) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have same length"
        );

        // # Safety
        // JoltGroupWrapper is repr(transparent) so has same memory layout as G2Projective
        let vs_proj: &mut [G2Projective] = unsafe {
            std::slice::from_raw_parts_mut(vs.as_mut_ptr() as *mut G2Projective, vs.len())
        };
        let addends_proj: &[G2Projective] = unsafe {
            std::slice::from_raw_parts(addends.as_ptr() as *const G2Projective, addends.len())
        };

        // Use jolt-optimizations function: v[i] = scalar * v[i] + gamma[i]
        jolt_optimizations::vector_scalar_mul_add_gamma_g2_online(vs_proj, scalar.0, addends_proj);
    }
}

// We implement MSM for specifically GT (the other case handles G1 and G2)
impl<P> DoryMultiScalarMul<JoltGTWrapper<P>> for JoltMSM
where
    P: ArkPairing,
    P::ScalarField: JoltField,
{
    #[tracing::instrument(skip_all, name = "GT MSM")]
    fn msm(
        bases: &[JoltGTWrapper<P>],
        scalars: &[JoltFieldWrapper<P::ScalarField>],
    ) -> JoltGTWrapper<P> {
        let chunk_size = (scalars.len() / rayon::current_num_threads()).max(32);

        bases
            .par_chunks(chunk_size)
            .zip(scalars.par_chunks(chunk_size))
            .map(|(base_chunk, coeff_chunk)| {
                base_chunk
                    .iter()
                    .zip(coeff_chunk.iter())
                    .filter(|(_, coeff)| !coeff.0.is_zero())
                    .fold(JoltGTWrapper::<P>::identity(), |acc, (base, coeff)| {
                        acc.add(&base.scale(coeff))
                    })
            })
            .sum()
    }
}

#[derive(Clone, Debug)]
pub struct JoltPairing<E: ArkPairing>(PhantomData<E>);

impl<E> DoryPairing for JoltPairing<E>
where
    E: ArkPairing,
    E::ScalarField: JoltField,
    E::G1: Icicle,
    E::G2: Icicle,
{
    type G1 = JoltGroupWrapper<E::G1>;
    type G2 = JoltGroupWrapper<E::G2>;
    type GT = JoltGTWrapper<E>;

    #[tracing::instrument(skip_all)]
    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT {
        let gt = E::pairing(p.0, q.0).0;
        JoltGTWrapper(gt)
    }

    #[tracing::instrument(skip_all)]
    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        // Delegate to the cached version with no caches
        Self::multi_pair_cached(Some(ps), None, None, Some(qs), None, None)
    }

    fn multi_pair_cached(
        g1_points: Option<&[Self::G1]>,
        g1_count: Option<usize>,
        g1_cache: Option<&dory::curve::G1Cache>,
        g2_points: Option<&[Self::G2]>,
        g2_count: Option<usize>,
        g2_cache: Option<&dory::curve::G2Cache>,
    ) -> Self::GT {
        match (g1_points, g1_count, g1_cache, g2_points, g2_count, g2_cache) {
            // Case 1: Both G1 and G2 use cached prepared values (fully optimized)
            (None, Some(g1_c), Some(g1_cache), None, Some(g2_c), Some(g2_cache)) => {
                assert_eq!(g1_c, g2_c, "G1 and G2 counts must be equal");
                if g1_c == 0 {
                    return Self::GT::identity();
                }

                // Extract prepared values from caches
                let g1_prepared: Vec<&BnG1Prepared> = (0..g1_c)
                    .map(|i| {
                        g1_cache
                            .get_prepared(i)
                            .expect("Index out of bounds in G1 cache")
                    })
                    .collect();

                let g2_prepared: Vec<&BnG2Prepared> = (0..g2_c)
                    .map(|i| {
                        g2_cache
                            .get_prepared(i)
                            .expect("Index out of bounds in G2 cache")
                    })
                    .collect();

                // Use Bn254 directly since caches are BN254-specific
                let ml_result = Bn254::multi_miller_loop(g1_prepared, g2_prepared).0;

                let pairing_result = Bn254::final_exponentiation(MillerLoopOutput(ml_result))
                    .expect("Final exponentiation should not fail");

                // # Safety
                // When E = Bn254, JoltGTWrapper<Bn254> has same memory layout as JoltGTWrapper<E>
                // since JoltGTWrapper is repr(transparent) and E::TargetField = Bn254::TargetField
                let bn_result = JoltGTWrapper::<Bn254>(pairing_result.0);
                unsafe {
                    let ptr = &bn_result as *const JoltGTWrapper<Bn254> as *const Self::GT;
                    (*ptr).clone()
                }
            }

            // Case 2: G1 cached, G2 fresh points (partial optimization)
            (None, Some(g1_c), Some(g1_cache), Some(g2_points), _, _) => {
                assert_eq!(
                    g1_c,
                    g2_points.len(),
                    "G1 count must equal G2 points length"
                );
                if g1_c == 0 {
                    return Self::GT::identity();
                }

                // G1 from cache
                let g1_prepared: Vec<&BnG1Prepared> = (0..g1_c)
                    .map(|i| {
                        g1_cache
                            .get_prepared(i)
                            .expect("Index out of bounds in G1 cache")
                    })
                    .collect();

                // # Safety
                // JoltGroupWrapper is repr(transparent) so has same memory layout as E::G2
                // When E = Bn254, E::G2 = G2Projective
                let g2_inner: &[G2Projective] = unsafe {
                    std::slice::from_raw_parts(
                        g2_points.as_ptr() as *const G2Projective,
                        g2_points.len(),
                    )
                };

                let g2_affine = G2Projective::normalize_batch(g2_inner);
                let g2_prepared = g2_affine
                    .par_iter()
                    .map(BnG2Prepared::from)
                    .collect::<Vec<_>>();

                // Use Bn254 directly since caches are BN254-specific
                let ml_result = Bn254::multi_miller_loop(g1_prepared, g2_prepared).0;

                let pairing_result = Bn254::final_exponentiation(MillerLoopOutput(ml_result))
                    .expect("Final exponentiation should not fail");

                // # Safety
                // When E = Bn254, JoltGTWrapper<Bn254> has same memory layout as JoltGTWrapper<E>
                // since JoltGTWrapper is repr(transparent) and E::TargetField = Bn254::TargetField
                let bn_result = JoltGTWrapper::<Bn254>(pairing_result.0);
                unsafe {
                    let ptr = &bn_result as *const JoltGTWrapper<Bn254> as *const Self::GT;
                    (*ptr).clone()
                }
            }

            // Case 3: G1 fresh points, G2 cached (partial optimization)
            (Some(g1_points), _, _, None, Some(g2_c), Some(g2_cache)) => {
                assert_eq!(
                    g1_points.len(),
                    g2_c,
                    "G1 points length must equal G2 count"
                );
                if g2_c == 0 {
                    return Self::GT::identity();
                }

                // # Safety
                // JoltGroupWrapper is repr(transparent) so has same memory layout as E::G1
                // When E = Bn254, E::G1 = G1Projective
                let g1_inner: &[G1Projective] = unsafe {
                    std::slice::from_raw_parts(
                        g1_points.as_ptr() as *const G1Projective,
                        g1_points.len(),
                    )
                };

                let g1_affine = G1Projective::normalize_batch(g1_inner);
                let g1_prepared = g1_affine
                    .par_iter()
                    .map(BnG1Prepared::from)
                    .collect::<Vec<_>>();

                // G2 from cache
                let g2_prepared: Vec<&BnG2Prepared> = (0..g2_c)
                    .map(|i| {
                        g2_cache
                            .get_prepared(i)
                            .expect("Index out of bounds in G2 cache")
                    })
                    .collect();

                // Use Bn254 directly since caches are BN254-specific
                let ml_result = Bn254::multi_miller_loop(g1_prepared, g2_prepared).0;

                let pairing_result = Bn254::final_exponentiation(MillerLoopOutput(ml_result))
                    .expect("Final exponentiation should not fail");

                // # Safety
                // When E = Bn254, JoltGTWrapper<Bn254> has same memory layout as JoltGTWrapper<E>
                // since JoltGTWrapper is repr(transparent) and E::TargetField = Bn254::TargetField
                let bn_result = JoltGTWrapper::<Bn254>(pairing_result.0);
                unsafe {
                    let ptr = &bn_result as *const JoltGTWrapper<Bn254> as *const Self::GT;
                    (*ptr).clone()
                }
            }

            // Case 4: Both fresh points (no caching benefit)
            (Some(g1_points), _, _, Some(g2_points), _, _) => {
                assert_eq!(
                    g1_points.len(),
                    g2_points.len(),
                    "G1 and G2 vectors must have equal length"
                );
                if g1_points.is_empty() {
                    return Self::GT::identity();
                }

                // # Safety
                // JoltGroupWrapper is repr(transparent) so has same memory layout as inner types
                let g1_inner: &[E::G1] = unsafe {
                    std::slice::from_raw_parts(g1_points.as_ptr() as *const E::G1, g1_points.len())
                };
                let g2_inner: &[E::G2] = unsafe {
                    std::slice::from_raw_parts(g2_points.as_ptr() as *const E::G2, g2_points.len())
                };

                let aff_left = E::G1::normalize_batch(g1_inner);
                let aff_right = E::G2::normalize_batch(g2_inner);

                let left: Vec<_> = aff_left.par_iter().map(E::G1Prepared::from).collect();
                let right: Vec<_> = aff_right.par_iter().map(E::G2Prepared::from).collect();

                let num_chunks = rayon::current_num_threads();
                let chunk_size = (left.len() / num_chunks.max(1)).max(1);

                let ml_result = left
                    .par_chunks(chunk_size)
                    .zip(right.par_chunks(chunk_size))
                    .map(|(aa, bb)| E::multi_miller_loop(aa.iter().cloned(), bb.iter().cloned()).0)
                    .product();

                let pairing_result = E::final_exponentiation(MillerLoopOutput(ml_result))
                    .expect("Final exponentiation should not fail");

                JoltGTWrapper(pairing_result.0)
            }

            _ => panic!("Invalid combination of parameters provided to multi_pair_cached"),
        }
    }
}

impl<F, G> DoryPolynomial<JoltFieldWrapper<F>, JoltGroupWrapper<G>> for MultilinearPolynomial<F>
where
    F: JoltField + PrimeField,
    G: CurveGroup<ScalarField = F> + VariableBaseMSM,
{
    fn get(&self, index: usize) -> JoltFieldWrapper<F> {
        assert!(
            index < self.len(),
            "Polynomial index out of bounds: {} >= {}",
            index,
            self.len()
        );
        JoltFieldWrapper(self.get_coeff(index))
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn evaluate(&self, point: &[JoltFieldWrapper<F>]) -> JoltFieldWrapper<F> {
        let point: Vec<_> = point.iter().rev().map(|x| x.0).collect();
        JoltFieldWrapper(PolynomialEvaluation::evaluate(self, &point))
    }

    #[tracing::instrument(skip_all)]
    fn commit_rows<M1: DoryMultiScalarMul<JoltGroupWrapper<G>>>(
        &self,
        g1_generators: &[JoltGroupWrapper<G>],
        row_len: usize,
    ) -> Vec<JoltGroupWrapper<G>> {
        let bases: Vec<_> = g1_generators
            .par_iter()
            .map(|g| g.0.into_affine())
            .collect();
        debug_assert_eq!(get_num_columns(), row_len);

        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly
                .Z
                .par_chunks(row_len)
                .map(|row| {
                    JoltGroupWrapper(
                        VariableBaseMSM::msm_field_elements(&bases, None, row, None, false)
                            .unwrap(),
                    )
                })
                .collect(),
            MultilinearPolynomial::U8Scalars(poly) => poly
                .coeffs
                .par_chunks(row_len)
                .map(|row| JoltGroupWrapper(VariableBaseMSM::msm_u8(&bases, row, None).unwrap()))
                .collect(),
            MultilinearPolynomial::U16Scalars(poly) => poly
                .coeffs
                .par_chunks(row_len)
                .map(|row| {
                    JoltGroupWrapper(
                        VariableBaseMSM::msm_u16(&bases, None, row, None, false).unwrap(),
                    )
                })
                .collect(),
            MultilinearPolynomial::U32Scalars(poly) => poly
                .coeffs
                .par_chunks(row_len)
                .map(|row| {
                    JoltGroupWrapper(
                        VariableBaseMSM::msm_u32(&bases, None, row, None, false).unwrap(),
                    )
                })
                .collect(),
            MultilinearPolynomial::U64Scalars(poly) => poly
                .coeffs
                .par_chunks(row_len)
                .map(|row| {
                    JoltGroupWrapper(
                        VariableBaseMSM::msm_u64(&bases, None, row, None, false).unwrap(),
                    )
                })
                .collect(),
            MultilinearPolynomial::I64Scalars(poly) => poly
                .coeffs
                .par_chunks(row_len)
                .map(|row| {
                    // TODO(moodlezoup): This can be optimized
                    let scalars: Vec<_> = row.iter().map(|x| F::from_i64(*x)).collect();
                    JoltGroupWrapper(
                        VariableBaseMSM::msm_field_elements(&bases, None, &scalars, None, false)
                            .unwrap(),
                    )
                })
                .collect(),
            MultilinearPolynomial::Sparse(_) => todo!(),
            MultilinearPolynomial::OneHot(_) => todo!(),
        }
    }

    #[tracing::instrument(skip_all)]
    fn vector_matrix_product(
        &self,
        left_vec: &[JoltFieldWrapper<F>],
        sigma: usize,
        nu: usize,
    ) -> Vec<JoltFieldWrapper<F>> {
        // # Safety
        // JoltFieldWrapper always has same memory layout as underlying F here.
        let left_vec: &[F] =
            unsafe { std::slice::from_raw_parts(left_vec.as_ptr() as *const F, left_vec.len()) };

        let num_columns = get_num_columns();
        println!("sigma: {sigma}");
        println!("nu: {nu}");
        println!("num_columns: {num_columns}");

        match self {
            MultilinearPolynomial::LargeScalars(poly) => (0..num_columns)
                .into_par_iter()
                .map(|col_index| {
                    JoltFieldWrapper(
                        poly.Z
                            .iter()
                            .skip(col_index)
                            .step_by(num_columns)
                            .zip(left_vec.iter())
                            .map(|(&a, &b)| -> F { a * b })
                            .sum::<F>(),
                    )
                })
                .collect(),
            _ => todo!(),
        }
    }
}

// Note that we have this `Option<&'a mut T>` so that we can derive Default, which is required.
#[derive(Default)]
pub struct JoltToDoryTranscriptRef<'a, F: JoltField, T: Transcript> {
    transcript: Option<&'a mut T>,
    _phantom: PhantomData<F>,
}

impl<'a, F: JoltField, T: Transcript> JoltToDoryTranscriptRef<'a, F, T> {
    pub fn new(transcript: &'a mut T) -> Self {
        JoltToDoryTranscriptRef {
            transcript: Some(transcript),
            _phantom: PhantomData,
        }
    }
}

impl<'a, F: JoltField, T: Transcript> DoryTranscript for JoltToDoryTranscriptRef<'a, F, T> {
    type Scalar = JoltFieldWrapper<F>;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &Self::Scalar) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        transcript.append_scalar(&x.0);
    }

    fn append_group<G: CanonicalSerialize>(&mut self, _label: &[u8], g: &G) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        transcript.append_serializable(g);
    }

    fn append_serde<S: serde::Serialize>(&mut self, _label: &[u8], s: &S) {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        let bytes = postcard::to_allocvec(s).unwrap_or_default();
        transcript.append_bytes(&bytes);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> Self::Scalar {
        let transcript = self
            .transcript
            .as_mut()
            .expect("Transcript not initialized");
        JoltFieldWrapper(transcript.challenge_scalar::<F>())
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        panic!("Reset not supported for JoltToDoryTranscript")
    }
}

// BN254-specific Aliases
pub type JoltG1Wrapper = JoltGroupWrapper<G1Projective>;
pub type JoltG2Wrapper = JoltGroupWrapper<G2Projective>;
pub type JoltGTBn254 = JoltGTWrapper<Bn254>;

pub type JoltBn254 = JoltPairing<Bn254>;

#[derive(Clone, Debug)]
pub struct DoryCommitmentScheme<ProofTranscript: Transcript>(PhantomData<ProofTranscript>);

#[derive(Clone, Debug, Default, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryCommitment(pub JoltGTBn254);

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryProofData {
    pub sigma: usize,
    pub dory_proof_data: DoryProof<JoltG1Wrapper, JoltG2Wrapper, JoltGTBn254>,
}

#[derive(Default, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryBatchedProof {
    proofs: Vec<DoryProofData>,
}

impl<ProofTranscript> CommitmentScheme<ProofTranscript> for DoryCommitmentScheme<ProofTranscript>
where
    ProofTranscript: Transcript,
{
    type Field = Fr;
    type ProverSetup = ProverSetup<JoltBn254>;
    type VerifierSetup = VerifierSetup<JoltBn254>;
    type Commitment = DoryCommitment;
    type Proof = DoryProofData;
    type BatchedProof = DoryBatchedProof;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let srs_file_name = format!("dory_srs_{max_num_vars}_variables.srs");
        let (prover_setup, _) = setup_with_srs_file::<JoltBn254, _>(
            &mut ark_std::rand::thread_rng(),
            max_num_vars,
            Some(&srs_file_name), // Will load if exists, generate and save if not
        );

        prover_setup
    }

    fn setup_verifier(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        prover_setup.to_verifier_setup()
    }

    fn srs_size(_setup: &Self::ProverSetup) -> usize {
        todo!()
    }

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::commit")]
    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> Self::Commitment {
        let sigma = get_num_columns().log_2();

        let commitment_val = commit::<JoltBn254, JoltMsmG1, _>(poly, 0, sigma, setup);
        DoryCommitment(commitment_val)
    }

    fn batch_commit<U>(_polys: &[U], _setup: &Self::ProverSetup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        todo!("Batch commit not yet implemented for Dory")
    }

    // Note that Dory implementation sometimes uses the term 'evaluation'/'evaluate' -- this is same as 'opening'/'open'
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::prove")]
    fn prove(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        // Dory uses the opposite endian-ness as Jolt
        let point_dory: Vec<JoltFieldWrapper<Self::Field>> = opening_point
            .iter()
            .rev()
            .map(|&p| JoltFieldWrapper(p))
            .collect();

        let sigma = get_num_columns().log_2();
        let dory_transcript = JoltToDoryTranscriptRef::<Self::Field, _>::new(transcript);

        // dory evaluate returns the opening but in this case we don't use it, we pass directly the opening to verify()
        let (_claimed_evaluation, proof_builder) =
            evaluate::<
                JoltBn254,
                JoltToDoryTranscriptRef<'_, Self::Field, ProofTranscript>,
                JoltMsmG1,
                JoltMsmG2,
                _,
            >(poly, &point_dory, sigma, setup, dory_transcript);

        let dory_proof = proof_builder.build();

        DoryProofData {
            sigma,
            dory_proof_data: dory_proof,
        }
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        // Dory uses the opposite endian-ness as Jolt
        let opening_point_dory: Vec<JoltFieldWrapper<Self::Field>> = opening_point
            .iter()
            .rev()
            .map(|&p| JoltFieldWrapper(p))
            .collect();

        let claimed_opening = JoltFieldWrapper(*opening);
        let dory_transcript = JoltToDoryTranscriptRef::<Self::Field, _>::new(transcript);
        let verifier_builder =
            DoryProofBuilder::from_proof_no_transcript(proof.dory_proof_data.clone());

        let verify_result = verify::<
            JoltBn254,
            JoltToDoryTranscriptRef<'_, Self::Field, ProofTranscript>,
            JoltMsmG1,
            JoltMsmG2,
            JoltMSM,
        >(
            commitment.0.clone(),
            claimed_opening,
            &opening_point_dory,
            verifier_builder,
            proof.sigma,
            setup,
            dory_transcript,
        );

        match verify_result {
            Ok(()) => Ok(()),
            Err(e) => Err(ProofVerifyError::DoryError(format!("{e:?}"))),
        }
    }

    fn combine_commitments(
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let combined_commitment: PairingOutput<_> = commitments
            .iter()
            .zip(coeffs.iter())
            .map(|(commitment, coeff)| {
                let g: PairingOutput<_> = commitment.0.clone().into();
                g * coeff
            })
            .sum();
        DoryCommitment(JoltGTWrapper::from(combined_commitment))
    }

    fn protocol_name() -> &'static [u8] {
        b"dory_commitment_scheme"
    }
}

impl AppendToTranscript for DoryCommitment {
    fn append_to_transcript<PT: Transcript>(&self, transcript: &mut PT) {
        transcript.append_serializable(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::compact_polynomial::CompactPolynomial;
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::utils::transcript::KeccakTranscript;
    use ark_std::rand::thread_rng;
    use ark_std::UniformRand;
    use std::time::Instant;

    fn test_commitment_scheme_with_poly(
        poly: MultilinearPolynomial<Fr>,
        poly_type_name: &str,
        prover_setup: &ProverSetup<JoltBn254>,
        verifier_setup: &VerifierSetup<JoltBn254>,
    ) -> (
        std::time::Duration,
        std::time::Duration,
        std::time::Duration,
        std::time::Duration,
    ) {
        let num_vars = poly.get_num_vars();
        let num_coeffs = match &poly {
            MultilinearPolynomial::LargeScalars(dense) => dense.Z.len(),
            MultilinearPolynomial::U8Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::U16Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::U32Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::U64Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::I64Scalars(compact) => compact.coeffs.len(),
            MultilinearPolynomial::Sparse(_) => todo!(),
            MultilinearPolynomial::OneHot(_) => todo!(),
        };

        println!(
            "Testing Dory PCS ({poly_type_name}) with {num_vars} variables, {num_coeffs} coefficients"
        );

        let mut rng = thread_rng();
        let opening_point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

        let commit_start = Instant::now();
        let commitment = DoryCommitmentScheme::<KeccakTranscript>::commit(&poly, prover_setup);
        let commit_time = commit_start.elapsed();

        println!(" Commit time: {commit_time:?}");

        let mut reversed_opening_point = opening_point.clone();
        reversed_opening_point.reverse();
        let evaluation = <MultilinearPolynomial<Fr> as PolynomialEvaluation<Fr>>::evaluate(
            &poly,
            &reversed_opening_point,
        );

        let mut prove_transcript = KeccakTranscript::new(b"dory_test");
        let prove_start = Instant::now();
        let proof = DoryCommitmentScheme::<KeccakTranscript>::prove(
            prover_setup,
            &poly,
            &opening_point,
            &mut prove_transcript,
        );
        let prove_time = prove_start.elapsed();

        println!(" Prove time: {prove_time:?}");

        let mut verify_transcript = KeccakTranscript::new(b"dory_test");
        let verify_start = Instant::now();
        let verification_result = DoryCommitmentScheme::<KeccakTranscript>::verify(
            &proof,
            verifier_setup,
            &mut verify_transcript,
            &opening_point,
            &evaluation,
            &commitment,
        );
        let verify_time = verify_start.elapsed();

        println!(" Verify time: {verify_time:?}");
        let total_time = commit_time + prove_time + verify_time;
        println!(" Total time (without setup): {total_time:?}");

        assert!(
            verification_result.is_ok(),
            "Dory verification failed for {poly_type_name}: {verification_result:?}"
        );
        println!(" ✅ {poly_type_name} test passed!\n");

        (commit_time, prove_time, verify_time, total_time)
    }

    #[test]
    fn test_dory_commitment_scheme_all_polynomial_types() {
        let max_num_vars = 18;

        let num_vars = 18;

        let num_coeffs = 1 << num_vars;

        println!("Setting up Dory PCS with max_num_vars = {max_num_vars}");
        let setup_start = Instant::now();
        let prover_setup = DoryCommitmentScheme::<KeccakTranscript>::setup_prover(max_num_vars);
        let verifier_setup =
            DoryCommitmentScheme::<KeccakTranscript>::setup_verifier(&prover_setup);
        let setup_time = setup_start.elapsed();
        println!("Setup time: {setup_time:?}\n");

        let mut rng = thread_rng();

        // Test 1: LargeScalars (Field elements)
        let coeffs_large: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
        let poly_large = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs_large));
        let (commit_large, prove_large, verify_large, total_large) =
            test_commitment_scheme_with_poly(
                poly_large,
                "LargeScalars",
                &prover_setup,
                &verifier_setup,
            );

        // Test 2: U8Scalars
        let coeffs_u8: Vec<u8> = (0..num_coeffs).map(|_| rng.next_u32() as u8).collect();
        let poly_u8 = MultilinearPolynomial::U8Scalars(CompactPolynomial::from_coeffs(coeffs_u8));
        let (commit_u8, prove_u8, verify_u8, total_u8) =
            test_commitment_scheme_with_poly(poly_u8, "U8Scalars", &prover_setup, &verifier_setup);

        // Test 3: U16Scalars
        let coeffs_u16: Vec<u16> = (0..num_coeffs).map(|_| rng.next_u32() as u16).collect();
        let poly_u16 =
            MultilinearPolynomial::U16Scalars(CompactPolynomial::from_coeffs(coeffs_u16));
        let (commit_u16, prove_u16, verify_u16, total_u16) = test_commitment_scheme_with_poly(
            poly_u16,
            "U16Scalars",
            &prover_setup,
            &verifier_setup,
        );

        // Test 4: U32Scalars
        let coeffs_u32: Vec<u32> = (0..num_coeffs).map(|_| rng.next_u32()).collect();
        let poly_u32 =
            MultilinearPolynomial::U32Scalars(CompactPolynomial::from_coeffs(coeffs_u32));
        let (commit_u32, prove_u32, verify_u32, total_u32) = test_commitment_scheme_with_poly(
            poly_u32,
            "U32Scalars",
            &prover_setup,
            &verifier_setup,
        );

        // Test 5: U64Scalars
        let coeffs_u64: Vec<u64> = (0..num_coeffs).map(|_| rng.next_u64()).collect();
        let poly_u64 =
            MultilinearPolynomial::U64Scalars(CompactPolynomial::from_coeffs(coeffs_u64));
        let (commit_u64, prove_u64, verify_u64, total_u64) = test_commitment_scheme_with_poly(
            poly_u64,
            "U64Scalars",
            &prover_setup,
            &verifier_setup,
        );

        // Test 6: I64Scalars
        let coeffs_i64: Vec<i64> = (0..num_coeffs).map(|_| rng.next_u64() as i64).collect();
        let poly_i64 =
            MultilinearPolynomial::I64Scalars(CompactPolynomial::from_coeffs(coeffs_i64));
        let (commit_i64, prove_i64, verify_i64, total_i64) = test_commitment_scheme_with_poly(
            poly_i64,
            "I64Scalars",
            &prover_setup,
            &verifier_setup,
        );

        println!("========== PERFORMANCE SUMMARY ==========");

        println!("Setup time: {setup_time:?}\n");

        println!("Polynomial Type | Commit Time | Prove Time | Verify Time | Total Time");

        println!("----------------|-------------|-------------|-------------|------------");
        println!(
            "LargeScalars | {commit_large:>11?} | {prove_large:>11?} | {verify_large:>11?} | {total_large:>10?}"
        );
        println!(
            "U8Scalars | {commit_u8:>11?} | {prove_u8:>11?} | {verify_u8:>11?} | {total_u8:>10?}"
        );
        println!(
            "U16Scalars | {commit_u16:>11?} | {prove_u16:>11?} | {verify_u16:>11?} | {total_u16:>10?}"
        );
        println!(
            "U32Scalars | {commit_u32:>11?} | {prove_u32:>11?} | {verify_u32:>11?} | {total_u32:>10?}"
        );
        println!(
            "U64Scalars | {commit_u64:>11?} | {prove_u64:>11?} | {verify_u64:>11?} | {total_u64:>10?}"
        );
        println!(
            "I64Scalars | {commit_i64:>11?} | {prove_i64:>11?} | {verify_i64:>11?} | {total_i64:>10?}"
        );
        println!("==========================================");
    }

    #[test]
    fn test_dory_soundness() {
        use ark_std::UniformRand;

        let num_vars = 10;
        let max_num_vars = 10;
        let num_coeffs = 1 << num_vars;

        let mut rng = thread_rng();
        let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs.clone()));

        let opening_point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

        let prover_setup = DoryCommitmentScheme::<KeccakTranscript>::setup_prover(max_num_vars);
        let verifier_setup =
            DoryCommitmentScheme::<KeccakTranscript>::setup_verifier(&prover_setup);

        let commitment = DoryCommitmentScheme::<KeccakTranscript>::commit(&poly, &prover_setup);

        let mut prove_transcript =
            KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());

        // Compute the correct evaluation
        let opening_point_dory: Vec<JoltFieldWrapper<Fr>> =
            opening_point.iter().map(|&p| JoltFieldWrapper(p)).collect();
        let correct_evaluation = <MultilinearPolynomial<Fr> as DoryPolynomial<
            JoltFieldWrapper<Fr>,
            JoltGroupWrapper<G1Projective>,
        >>::evaluate(&poly, &opening_point_dory)
        .0;

        let proof = DoryCommitmentScheme::<KeccakTranscript>::prove(
            &prover_setup,
            &poly,
            &opening_point,
            &mut prove_transcript,
        );

        // Test 1: Tamper with the evaluation
        {
            let tampered_evaluation = Fr::rand(&mut rng);

            let mut verify_transcript =
                KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &proof,
                &verifier_setup,
                &mut verify_transcript,
                &opening_point,
                &tampered_evaluation,
                &commitment,
            );

            assert!(
                result.is_err(),
                "Verification should fail with tampered evaluation"
            );
            println!("✅ Test 1 passed: Tampered evaluation correctly rejected");
        }

        // Test 2: Tamper with the opening point
        {
            let tampered_opening_point: Vec<Fr> =
                (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

            let mut verify_transcript =
                KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &proof,
                &verifier_setup,
                &mut verify_transcript,
                &tampered_opening_point,
                &correct_evaluation,
                &commitment,
            );

            assert!(
                result.is_err(),
                "Verification should fail with tampered opening point"
            );
            println!("✅ Test 2 passed: Tampered opening point correctly rejected");
        }

        // Test 3: Use wrong commitment
        {
            // Create a different polynomial and its commitment
            let wrong_coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
            let wrong_poly =
                MultilinearPolynomial::LargeScalars(DensePolynomial::new(wrong_coeffs));
            let wrong_commitment =
                DoryCommitmentScheme::<KeccakTranscript>::commit(&wrong_poly, &prover_setup);

            let mut verify_transcript =
                KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &proof,
                &verifier_setup,
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
                &wrong_commitment,
            );

            assert!(
                result.is_err(),
                "Verification should fail with wrong commitment"
            );
            println!("✅ Test 3 passed: Wrong commitment correctly rejected");
        }

        // Test 4: Use wrong domain in transcript
        {
            let mut verify_transcript = KeccakTranscript::new(b"wrong_domain");
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &proof,
                &verifier_setup,
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
                &commitment,
            );

            assert!(
                result.is_err(),
                "Verification should fail with wrong transcript domain"
            );
            println!("✅ Test 4 passed: Wrong transcript domain correctly rejected");
        }

        // Test 5: Verify that correct proof still passes
        {
            let mut verify_transcript =
                KeccakTranscript::new(DoryCommitmentScheme::<KeccakTranscript>::protocol_name());
            let result = DoryCommitmentScheme::<KeccakTranscript>::verify(
                &proof,
                &verifier_setup,
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
                &commitment,
            );

            assert!(
                result.is_ok(),
                "Verification should succeed with correct proof"
            );
            println!("✅ Test 5 passed: Correct proof indeed verifies successfully");
        }
    }
}
