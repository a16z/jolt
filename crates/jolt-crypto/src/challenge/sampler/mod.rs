//! Sparse ring fold challenges decoded from one transcript message.
//!
//! After the prover's folded witness message `v` is absorbed, the protocol
//! samples sparse ring elements `c` used to fold the witness toward the next
//! commitment. Every [`SparseChallengeConfig`] is a fixed-weight signed shell:
//! `count_pm1` coefficients at ±1 and `count_pm2` at ±2. The decoder lives in
//! [`stream`]; this module holds the certified operator-norm predicates.

#[cfg(feature = "sample")]
mod constant_time;
mod indexed;
mod op_norm;
mod op_norm_accumulate;
mod position_sample;
mod signed_sparse;
mod stream;
mod xof;

#[cfg(feature = "sample")]
pub use constant_time::sample_challenge_ct;
#[cfg(feature = "sample-diagnostics")]
pub use constant_time::sample_challenge_ct_counted;
pub use indexed::sample_indexed_challenges_from_seed;
pub use stream::{SparseChallengeShell, SparseChallengeStream};

use crate::CryptoError;
use std::sync::{Arc, LazyLock};

use crate::challenge::{
    OperatorNormRejection, SparseChallengeConfig, D128_SELECTIVE_L2_CHALLENGE_CONFIG,
    D64_SELECTIVE_L2_CHALLENGE_CONFIG,
};

use op_norm::OpNormTable;

const OP_NORM_PREDICATE_SCALE: u32 = 48;
/// Draws an operator-norm-rejecting sampler makes per coordinate before it fails.
const MAX_OP_NORM_ATTEMPTS: usize = 4096;
static D64_SELECTIVE_L2_OP_NORM_TABLE: LazyLock<Result<Arc<OpNormTable>, &'static str>> =
    LazyLock::new(|| {
        let config = D64_SELECTIVE_L2_CHALLENGE_CONFIG;
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
        let config = D128_SELECTIVE_L2_CHALLENGE_CONFIG;
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
enum ChallengePolicy {
    Unrestricted,
    OperatorNorm {
        threshold: u32,
        table: Arc<OpNormTable>,
    },
}

impl ChallengePolicy {
    fn new(
        ring_d: usize,
        cfg: &SparseChallengeConfig,
        rejection: Option<OperatorNormRejection>,
    ) -> Result<Self, CryptoError> {
        let Some(rejection) = rejection else {
            return Ok(Self::Unrestricted);
        };
        rejection
            .validate(ring_d, cfg)
            .map_err(|error| CryptoError::InvalidSetup(error.into()))?;
        let table = match rejection {
            OperatorNormRejection::D64_SELECTIVE_L2 => &*D64_SELECTIVE_L2_OP_NORM_TABLE,
            OperatorNormRejection::D128_SELECTIVE_L2 => &*D128_SELECTIVE_L2_OP_NORM_TABLE,
            _ => {
                return Err(CryptoError::InvalidSetup(
                    "unsupported operator-norm rejection policy".into(),
                ));
            }
        };
        let table = Arc::clone(
            table
                .as_ref()
                .map_err(|message| CryptoError::InvalidSetup((*message).into()))?,
        );
        Ok(Self::OperatorNorm {
            threshold: rejection.threshold,
            table,
        })
    }
}
