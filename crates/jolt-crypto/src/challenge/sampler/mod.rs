//! Sparse ring fold challenge sampling via Fiat-Shamir with PRG expansion.
//!
//! After the prover's folded witness message `v` is absorbed, the protocol
//! samples sparse ring elements `c` used to fold the witness toward the next
//! commitment. Every [`SparseChallengeConfig`] uses the signed-sparse sampler:
//! `count_pm1` coefficients at ±1 and `count_pm2` at ±2.

mod op_norm;
mod op_norm_accumulate;
mod position_sample;
mod signed_sparse;
mod xof;

pub(crate) use position_sample::MAX_STACK_RING_DIM;
pub(crate) use signed_sparse::SignedSparseScratch;
pub(crate) use xof::{IndexedXofPrefix, XofCursor};

use akita_error::AkitaError;
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use std::sync::{Arc, LazyLock};

use crate::{OperatorNormRejection, SparseChallenge, SparseChallengeConfig};

use op_norm::OpNormTable;

const OP_NORM_PREDICATE_SCALE: u32 = 48;
const MAX_OP_NORM_ATTEMPTS: usize = 4096;
#[cfg(feature = "parallel")]
const PARALLEL_SAMPLING_WORK_THRESHOLD: usize = 1 << 14;
static D64_SELECTIVE_L2_OP_NORM_TABLE: LazyLock<Result<Arc<OpNormTable>, &'static str>> =
    LazyLock::new(|| {
        let config = crate::D64_SELECTIVE_L2_CHALLENGE_CONFIG;
        let policy = OperatorNormRejection::D64_SELECTIVE_L2;
        let table = OpNormTable::new(
            64,
            OP_NORM_PREDICATE_SCALE,
            config.l1_norm() as u64,
            u64::from(policy.threshold),
        )
        .map_err(|_| "failed to initialize the D64 selective-L2 operator-norm table")?;
        if !table.strict_threshold_contains_shrunken_subset(
            policy.fractional_bits,
            policy.root_coordinate_error_units,
            u64::from(policy.threshold),
            policy.rounding_margin_units,
        ) {
            return Err("the D64 operator-norm table does not contain its certified subset");
        }
        Ok(Arc::new(table))
    });
static D128_SELECTIVE_L2_OP_NORM_TABLE: LazyLock<Result<Arc<OpNormTable>, &'static str>> =
    LazyLock::new(|| {
        let config = crate::D128_SELECTIVE_L2_CHALLENGE_CONFIG;
        let policy = OperatorNormRejection::D128_SELECTIVE_L2;
        let table = OpNormTable::new(
            128,
            OP_NORM_PREDICATE_SCALE,
            config.l1_norm() as u64,
            u64::from(policy.threshold),
        )
        .map_err(|_| "failed to initialize the D128 selective-L2 operator-norm table")?;
        if !table.strict_threshold_contains_shrunken_subset(
            policy.fractional_bits,
            policy.root_coordinate_error_units,
            u64::from(policy.threshold),
            policy.rounding_margin_units,
        ) {
            return Err("the D128 operator-norm table does not contain its certified subset");
        }
        Ok(Arc::new(table))
    });

#[derive(Clone)]
enum IndexedSamplingPolicy {
    Unrestricted,
    OperatorNorm {
        threshold: u32,
        table: Arc<OpNormTable>,
    },
}

impl IndexedSamplingPolicy {
    fn new(
        ring_d: usize,
        cfg: &SparseChallengeConfig,
        rejection: Option<OperatorNormRejection>,
    ) -> Result<Self, AkitaError> {
        let Some(rejection) = rejection else {
            return Ok(Self::Unrestricted);
        };
        rejection
            .validate(ring_d, cfg)
            .map_err(|error| AkitaError::InvalidSetup(error.into()))?;
        let table = match rejection {
            OperatorNormRejection::D64_SELECTIVE_L2 => &*D64_SELECTIVE_L2_OP_NORM_TABLE,
            OperatorNormRejection::D128_SELECTIVE_L2 => &*D128_SELECTIVE_L2_OP_NORM_TABLE,
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "unsupported operator-norm rejection policy".into(),
                ));
            }
        };
        let table = Arc::clone(
            table
                .as_ref()
                .map_err(|message| AkitaError::InvalidSetup((*message).into()))?,
        );
        Ok(Self::OperatorNorm {
            threshold: rejection.threshold,
            table,
        })
    }
}

struct IndexedChallengeWorker {
    prefix: IndexedXofPrefix,
    cursor: XofCursor,
    scratch: SignedSparseScratch,
    policy: IndexedSamplingPolicy,
}

impl IndexedChallengeWorker {
    fn new(
        prefix: &IndexedXofPrefix,
        cfg: &SparseChallengeConfig,
        policy: &IndexedSamplingPolicy,
    ) -> Self {
        Self {
            prefix: prefix.clone(),
            cursor: XofCursor::new(),
            scratch: SignedSparseScratch::new(cfg.count_pm1, cfg.count_pm2),
            policy: policy.clone(),
        }
    }

    fn sample(
        &mut self,
        coordinate_index: u64,
        ring_d: usize,
        cfg: &SparseChallengeConfig,
    ) -> Result<SparseChallenge, AkitaError> {
        self.cursor
            .reset_indexed_prefix(&self.prefix, coordinate_index);
        let IndexedSamplingPolicy::OperatorNorm { threshold, table } = &self.policy else {
            self.scratch
                .sample(&mut self.cursor, ring_d, cfg.count_pm1, cfg.count_pm2)?;
            return Ok(self.scratch.take_challenge());
        };
        for _ in 0..MAX_OP_NORM_ATTEMPTS {
            self.scratch
                .sample(&mut self.cursor, ring_d, cfg.count_pm1, cfg.count_pm2)?;
            if table.accept_strict_parts(
                self.scratch.positions(),
                self.scratch.coeffs(),
                u64::from(*threshold),
            )? {
                return Ok(self.scratch.take_challenge());
            }
        }
        Err(AkitaError::InvalidInput(format!(
            "operator-norm rejection exceeded {MAX_OP_NORM_ATTEMPTS} attempts at coordinate {coordinate_index}"
        )))
    }
}

pub(crate) fn sample_indexed_challenges_from_seed(
    seed: &[u8],
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
    rejection: Option<OperatorNormRejection>,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    u64::try_from(n - 1).map_err(|_| {
        AkitaError::InvalidSetup("sparse challenge coordinate index exceeds u64".into())
    })?;
    let policy = IndexedSamplingPolicy::new(ring_d, cfg, rejection)?;
    let prefix =
        IndexedXofPrefix::new(seed).map_err(|message| AkitaError::InvalidSetup(message.into()))?;
    #[cfg(feature = "parallel")]
    {
        let work = n
            .checked_mul(cfg.weight())
            .ok_or_else(|| AkitaError::InvalidSetup("sparse challenge work overflow".into()))?;
        if work >= PARALLEL_SAMPLING_WORK_THRESHOLD {
            return (0..n)
                .into_par_iter()
                .map_init(
                    || IndexedChallengeWorker::new(&prefix, cfg, &policy),
                    |worker, index| {
                        let coordinate_index = u64::try_from(index).map_err(|_| {
                            AkitaError::InvalidSetup(
                                "sparse challenge coordinate index exceeds u64".into(),
                            )
                        })?;
                        worker.sample(coordinate_index, ring_d, cfg)
                    },
                )
                .collect();
        }
    }
    sample_indexed_challenges_sequential(&prefix, ring_d, n, cfg, &policy)
}

fn sample_indexed_challenges_sequential(
    prefix: &IndexedXofPrefix,
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
    policy: &IndexedSamplingPolicy,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    sample_indexed_challenges_sequential_with(prefix, n, cfg, policy, |worker, index| {
        worker.sample(index, ring_d, cfg)
    })
}

fn sample_indexed_challenges_sequential_with(
    prefix: &IndexedXofPrefix,
    n: usize,
    cfg: &SparseChallengeConfig,
    policy: &IndexedSamplingPolicy,
    mut sample: impl FnMut(&mut IndexedChallengeWorker, u64) -> Result<SparseChallenge, AkitaError>,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    let mut worker = IndexedChallengeWorker::new(prefix, cfg, policy);
    (0..n)
        .map(|index| {
            let coordinate_index = u64::try_from(index).map_err(|_| {
                AkitaError::InvalidSetup("sparse challenge coordinate index exceeds u64".into())
            })?;
            sample(&mut worker, coordinate_index)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_draw_matches_individual_coordinate_streams() {
        let ring_d = 64;
        let cfg = SparseChallengeConfig::production_for_ring_dim(ring_d).unwrap();
        let seed = [11u8; 32];
        let challenge_count = 257;
        let indexed =
            sample_indexed_challenges_from_seed(&seed, ring_d, challenge_count, &cfg, None)
                .unwrap();
        let policy = IndexedSamplingPolicy::new(ring_d, &cfg, None).unwrap();
        let prefix = IndexedXofPrefix::new(&seed).unwrap();
        let expected = (0..challenge_count)
            .map(|index| {
                IndexedChallengeWorker::new(&prefix, &cfg, &policy)
                    .sample(index as u64, ring_d, &cfg)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(indexed, expected);
        assert_ne!(indexed[0], indexed[1]);
    }

    #[test]
    fn reprogramming_one_coordinate_leaves_every_other_coordinate_unchanged() {
        let ring_d = 64;
        let cfg = SparseChallengeConfig::production_for_ring_dim(ring_d).unwrap();
        let seed = [23u8; 32];
        let alternate_seed = [29u8; 32];
        let before = sample_indexed_challenges_from_seed(&seed, ring_d, 12, &cfg, None).unwrap();
        let policy = IndexedSamplingPolicy::new(ring_d, &cfg, None).unwrap();
        let prefix = IndexedXofPrefix::new(&seed).unwrap();
        let alternate_prefix = IndexedXofPrefix::new(&alternate_seed).unwrap();
        let alternate_answer = IndexedChallengeWorker::new(&alternate_prefix, &cfg, &policy)
            .sample(7, ring_d, &cfg)
            .unwrap();
        assert_ne!(alternate_answer, before[7]);

        let forked = sample_indexed_challenges_sequential_with(
            &prefix,
            before.len(),
            &cfg,
            &policy,
            |worker, index| {
                if index == 7 {
                    Ok(alternate_answer.clone())
                } else {
                    worker.sample(index, ring_d, &cfg)
                }
            },
        )
        .unwrap();
        for index in 0..forked.len() {
            if index == 7 {
                assert_ne!(forked[index], before[index]);
            } else {
                assert_eq!(forked[index], before[index]);
            }
        }
    }

    #[test]
    fn indexed_operator_rejection_uses_independent_coordinate_streams() {
        let seed = [31u8; 32];
        for (ring_d, cfg, rejection) in [
            (
                64,
                crate::D64_SELECTIVE_L2_CHALLENGE_CONFIG,
                OperatorNormRejection::D64_SELECTIVE_L2,
            ),
            (
                128,
                crate::D128_SELECTIVE_L2_CHALLENGE_CONFIG,
                OperatorNormRejection::D128_SELECTIVE_L2,
            ),
        ] {
            let indexed =
                sample_indexed_challenges_from_seed(&seed, ring_d, 8, &cfg, Some(rejection))
                    .unwrap();
            let policy = IndexedSamplingPolicy::new(ring_d, &cfg, Some(rejection)).unwrap();
            let prefix = IndexedXofPrefix::new(&seed).unwrap();
            let individual = (0..8)
                .map(|index| {
                    IndexedChallengeWorker::new(&prefix, &cfg, &policy)
                        .sample(index, ring_d, &cfg)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(indexed, individual);
            assert!(indexed.iter().all(|challenge| {
                challenge.positions.len() == cfg.weight()
                    && challenge.coeffs.iter().filter(|&&c| c.abs() == 1).count() == cfg.count_pm1
                    && challenge.coeffs.iter().filter(|&&c| c.abs() == 2).count() == cfg.count_pm2
            }));
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn indexed_parallel_and_sequential_sampling_match() {
        let ring_d = 64;
        let cfg = SparseChallengeConfig::production_for_ring_dim(ring_d).unwrap();
        let seed = [37u8; 32];
        let challenge_count = 400;
        let parallel =
            sample_indexed_challenges_from_seed(&seed, ring_d, challenge_count, &cfg, None)
                .unwrap();
        let policy = IndexedSamplingPolicy::new(ring_d, &cfg, None).unwrap();
        let prefix = IndexedXofPrefix::new(&seed).unwrap();
        let sequential =
            sample_indexed_challenges_sequential(&prefix, ring_d, challenge_count, &cfg, &policy)
                .unwrap();
        assert_eq!(parallel, sequential);
    }
}
