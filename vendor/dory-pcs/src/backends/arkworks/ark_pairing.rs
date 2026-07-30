//! BN254 pairing implementation with optimizations

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use super::ark_group::{ArkG1, ArkG2, ArkGT};
use crate::primitives::arithmetic::{Group, PairingCurve};
use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;

#[cfg(feature = "parallel")]
use ark_ff::One;

#[derive(Default, Clone, Debug)]
pub struct BN254;

mod pairing_helpers {
    use super::*;
    use super::{ArkG1, ArkG2, ArkGT};

    /// Determine optimal chunk size for parallel Miller loop computation
    #[cfg(feature = "parallel")]
    fn determine_chunk_size(total: usize) -> usize {
        const MIN_CHUNK: usize = 32;
        const MAX_CHUNK: usize = 128;

        if total < MIN_CHUNK {
            return total;
        }

        let num_threads = rayon::current_num_threads();
        let chunk = total.div_ceil(num_threads);
        chunk.clamp(MIN_CHUNK, MAX_CHUNK)
    }

    /// Sequential multi-pairing
    #[allow(dead_code)]
    #[tracing::instrument(skip_all, name = "multi_pair_sequential", fields(len = ps.len()))]
    pub(super) fn multi_pair_sequential(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        use ark_bn254::{G1Affine, G2Affine};

        let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> = ps
            .iter()
            .map(|p| {
                let affine: G1Affine = p.0.into();
                affine.into()
            })
            .collect();

        let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> = qs
            .iter()
            .map(|q| {
                let affine: G2Affine = q.0.into();
                affine.into()
            })
            .collect();

        multi_pair_with_prepared(ps_prep, &qs_prep)
    }

    /// Sequential multi-pairing with G2 from setup (uses cache if available)
    #[allow(dead_code)]
    #[tracing::instrument(skip_all, name = "multi_pair_g2_setup_sequential", fields(len = ps.len()))]
    pub(super) fn multi_pair_g2_setup_sequential(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        use ark_bn254::G1Affine;

        let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> = ps
            .iter()
            .map(|p| {
                let affine: G1Affine = p.0.into();
                affine.into()
            })
            .collect();

        #[cfg(feature = "cache")]
        {
            if let Some(cache) = crate::backends::arkworks::ark_cache::get_prepared_cache() {
                return multi_pair_with_prepared(ps_prep, &cache.g2_prepared[..qs.len()]);
            }
        }

        use ark_bn254::G2Affine;
        let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> = qs
            .iter()
            .map(|q| {
                let affine: G2Affine = q.0.into();
                affine.into()
            })
            .collect();
        multi_pair_with_prepared(ps_prep, &qs_prep)
    }

    /// Sequential multi-pairing with G1 from setup (uses cache if available)
    #[allow(dead_code)]
    #[tracing::instrument(skip_all, name = "multi_pair_g1_setup_sequential", fields(len = ps.len()))]
    pub(super) fn multi_pair_g1_setup_sequential(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        use ark_bn254::G2Affine;

        let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> = qs
            .iter()
            .map(|q| {
                let affine: G2Affine = q.0.into();
                affine.into()
            })
            .collect();

        #[cfg(feature = "cache")]
        {
            if let Some(cache) = crate::backends::arkworks::ark_cache::get_prepared_cache() {
                let ps_prep: Vec<_> = ps
                    .iter()
                    .enumerate()
                    .map(|(i, _)| cache.g1_prepared[i].clone())
                    .collect();
                return multi_pair_with_prepared(ps_prep, &qs_prep);
            }
        }

        use ark_bn254::G1Affine;
        let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> = ps
            .iter()
            .map(|p| {
                let affine: G1Affine = p.0.into();
                affine.into()
            })
            .collect();
        multi_pair_with_prepared(ps_prep, &qs_prep)
    }

    fn multi_pair_with_prepared(
        ps_prep: Vec<<Bn254 as Pairing>::G1Prepared>,
        qs_prep: &[<Bn254 as Pairing>::G2Prepared],
    ) -> ArkGT {
        let miller_output = Bn254::multi_miller_loop(ps_prep, qs_prep.to_vec());
        let result = Bn254::final_exponentiation(miller_output)
            .expect("Final exponentiation should not fail");
        ArkGT(result)
    }

    /// Parallel multi-pairing with chunked Miller loops (no caching assumptions)
    #[cfg(feature = "parallel")]
    #[tracing::instrument(skip_all, name = "multi_pair_parallel", fields(len = ps.len(), chunk_size = determine_chunk_size(ps.len())))]
    pub(super) fn multi_pair_parallel(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        use ark_bn254::{G1Affine, G2Affine};
        use rayon::prelude::*;

        let chunk_size = determine_chunk_size(ps.len());

        let combined = ps
            .par_chunks(chunk_size)
            .zip(qs.par_chunks(chunk_size))
            .map(|(ps_chunk, qs_chunk)| {
                let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> = ps_chunk
                    .iter()
                    .map(|p| {
                        let affine: G1Affine = p.0.into();
                        affine.into()
                    })
                    .collect();

                let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> = qs_chunk
                    .iter()
                    .map(|q| {
                        let affine: G2Affine = q.0.into();
                        affine.into()
                    })
                    .collect();

                Bn254::multi_miller_loop(ps_prep, qs_prep)
            })
            .reduce(
                || ark_ec::pairing::MillerLoopOutput(<<Bn254 as Pairing>::TargetField>::one()),
                |a, b| ark_ec::pairing::MillerLoopOutput(a.0 * b.0),
            );

        let result =
            Bn254::final_exponentiation(combined).expect("Final exponentiation should not fail");
        ArkGT(result)
    }

    /// Parallel multi-pairing with G2 from setup (uses cache if available)
    #[cfg(feature = "parallel")]
    #[tracing::instrument(skip_all, name = "multi_pair_g2_setup_parallel", fields(len = ps.len(), chunk_size = determine_chunk_size(ps.len())))]
    pub(super) fn multi_pair_g2_setup_parallel(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        use ark_bn254::G1Affine;
        use rayon::prelude::*;

        let chunk_size = determine_chunk_size(ps.len());

        #[cfg(feature = "cache")]
        let cache = crate::backends::arkworks::ark_cache::get_prepared_cache();

        let combined = ps
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, ps_chunk)| {
                let start_idx = chunk_idx * chunk_size;
                let end_idx = start_idx + ps_chunk.len();

                let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> = ps_chunk
                    .iter()
                    .map(|p| {
                        let affine: G1Affine = p.0.into();
                        affine.into()
                    })
                    .collect();

                #[cfg(feature = "cache")]
                let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> = if let Some(ref c) = cache {
                    c.g2_prepared[start_idx..end_idx].to_vec()
                } else {
                    use ark_bn254::G2Affine;
                    qs[start_idx..end_idx]
                        .iter()
                        .map(|q| {
                            let affine: G2Affine = q.0.into();
                            affine.into()
                        })
                        .collect()
                };
                #[cfg(not(feature = "cache"))]
                let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> = {
                    use ark_bn254::G2Affine;
                    qs[start_idx..end_idx]
                        .iter()
                        .map(|q| {
                            let affine: G2Affine = q.0.into();
                            affine.into()
                        })
                        .collect()
                };

                Bn254::multi_miller_loop(ps_prep, qs_prep)
            })
            .reduce(
                || ark_ec::pairing::MillerLoopOutput(<<Bn254 as Pairing>::TargetField>::one()),
                |a, b| ark_ec::pairing::MillerLoopOutput(a.0 * b.0),
            );

        let result =
            Bn254::final_exponentiation(combined).expect("Final exponentiation should not fail");
        ArkGT(result)
    }

    /// Parallel multi-pairing with G1 from setup (uses cache if available)
    #[cfg(feature = "parallel")]
    #[tracing::instrument(skip_all, name = "multi_pair_g1_setup_parallel", fields(len = ps.len(), chunk_size = determine_chunk_size(ps.len())))]
    pub(super) fn multi_pair_g1_setup_parallel(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        use ark_bn254::G2Affine;
        use rayon::prelude::*;

        let chunk_size = determine_chunk_size(ps.len());

        #[cfg(feature = "cache")]
        let cache = crate::backends::arkworks::ark_cache::get_prepared_cache();

        let combined = qs
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, qs_chunk)| {
                let start_idx = chunk_idx * chunk_size;
                let end_idx = start_idx + qs_chunk.len();

                let qs_prep: Vec<<Bn254 as Pairing>::G2Prepared> = qs_chunk
                    .iter()
                    .map(|q| {
                        let affine: G2Affine = q.0.into();
                        affine.into()
                    })
                    .collect();

                #[cfg(feature = "cache")]
                let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> = if let Some(ref c) = cache {
                    c.g1_prepared[start_idx..end_idx].to_vec()
                } else {
                    use ark_bn254::G1Affine;
                    ps[start_idx..end_idx]
                        .iter()
                        .map(|p| {
                            let affine: G1Affine = p.0.into();
                            affine.into()
                        })
                        .collect()
                };
                #[cfg(not(feature = "cache"))]
                let ps_prep: Vec<<Bn254 as Pairing>::G1Prepared> = {
                    use ark_bn254::G1Affine;
                    ps[start_idx..end_idx]
                        .iter()
                        .map(|p| {
                            let affine: G1Affine = p.0.into();
                            affine.into()
                        })
                        .collect()
                };

                Bn254::multi_miller_loop(ps_prep, qs_prep)
            })
            .reduce(
                || ark_ec::pairing::MillerLoopOutput(<<Bn254 as Pairing>::TargetField>::one()),
                |a, b| ark_ec::pairing::MillerLoopOutput(a.0 * b.0),
            );

        let result =
            Bn254::final_exponentiation(combined).expect("Final exponentiation should not fail");
        ArkGT(result)
    }

    /// Optimized multi-pairing dispatch
    pub(super) fn multi_pair_optimized(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        #[cfg(feature = "parallel")]
        {
            multi_pair_parallel(ps, qs)
        }
        #[cfg(not(feature = "parallel"))]
        {
            multi_pair_sequential(ps, qs)
        }
    }

    /// Optimized multi-pairing dispatch for G2 from setup
    pub(super) fn multi_pair_g2_setup_optimized(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        #[cfg(feature = "parallel")]
        {
            multi_pair_g2_setup_parallel(ps, qs)
        }
        #[cfg(not(feature = "parallel"))]
        {
            multi_pair_g2_setup_sequential(ps, qs)
        }
    }

    /// Optimized multi-pairing dispatch for G1 from setup
    pub(super) fn multi_pair_g1_setup_optimized(ps: &[ArkG1], qs: &[ArkG2]) -> ArkGT {
        #[cfg(feature = "parallel")]
        {
            multi_pair_g1_setup_parallel(ps, qs)
        }
        #[cfg(not(feature = "parallel"))]
        {
            multi_pair_g1_setup_sequential(ps, qs)
        }
    }
}

impl PairingCurve for BN254 {
    type G1 = ArkG1;
    type G2 = ArkG2;
    type GT = ArkGT;

    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT {
        ArkGT(Bn254::pairing(p.0, q.0))
    }

    #[tracing::instrument(skip_all, name = "BN254::multi_pair", fields(len = ps.len()))]
    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        pairing_helpers::multi_pair_optimized(ps, qs)
    }

    #[tracing::instrument(skip_all, name = "BN254::multi_pair_g2_setup", fields(len = ps.len()))]
    fn multi_pair_g2_setup(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair_g2_setup requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        pairing_helpers::multi_pair_g2_setup_optimized(ps, qs)
    }

    #[tracing::instrument(skip_all, name = "BN254::multi_pair_g1_setup", fields(len = ps.len()))]
    fn multi_pair_g1_setup(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair_g1_setup requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        pairing_helpers::multi_pair_g1_setup_optimized(ps, qs)
    }
}
