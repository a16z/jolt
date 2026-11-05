//! Conditional inner verification and inherited setup obligations.
//!
//! A successful deferred replay is not acceptance. The enclosing execution
//! proof must publish the complete ordered statement and its verifier must
//! check that same statement in the outer Akita opening.

use std::fmt::{Debug, Formatter, Result as FormatResult};

use crate::adapters::{akita_error, invalid_batch};
use crate::AkitaField;
use akita_types::setup_claims::SetupBatchStatement;
pub use akita_verifier::{SetupClaimAccumulator, SetupClaimAdvice};
use jolt_openings::OpeningsError;

/// Setup verification mode carried by Jolt's owned verification context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupMode {
    Check,
    Defer,
}

/// Runtime setup-verification policy and obligations for a native batch.
///
/// Keys remain immutable and serializable. Deferred success is conditional:
/// callers must discharge every claim returned by the accumulator. Discard
/// this context after a failed verification.
#[must_use = "deferred setup obligations must be discharged before acceptance"]
pub struct AkitaVerificationContext<'a> {
    pub mode: SetupMode,
    pub accumulator: Option<SetupClaimAccumulator<'a, AkitaField>>,
    /// Optional diagnostic cycle marker; never part of the proof or transcript.
    pub cycle_marker: Option<&'static str>,
}

impl Default for AkitaVerificationContext<'_> {
    fn default() -> Self {
        Self {
            mode: SetupMode::Check,
            accumulator: None,
            cycle_marker: None,
        }
    }
}

#[derive(Clone)]
pub struct InheritedSetupClaims {
    pub(crate) level: u32,
    pub(crate) statement: SetupBatchStatement<AkitaField>,
}

impl Debug for InheritedSetupClaims {
    fn fmt(&self, f: &mut Formatter<'_>) -> FormatResult {
        f.debug_struct("InheritedSetupClaims")
            .field("level", &self.level)
            .field("claim_count", &self.statement.claims().len())
            .finish_non_exhaustive()
    }
}

impl InheritedSetupClaims {
    /// `public_output` is the canonical descriptor produced by the recursion
    /// guest, after removing the application's output serialization envelope.
    /// Checking it here prevents a host-selected subset from replacing the
    /// complete collection authenticated by the execution proof.
    pub fn from_public_output(
        level: u32,
        statement: SetupBatchStatement<AkitaField>,
        public_output: &[u8],
    ) -> Result<Self, OpeningsError> {
        if statement.claims().is_empty()
            || statement
                .descriptor_bytes::<AkitaField>()
                .map_err(akita_error)?
                != public_output
        {
            return Err(invalid_batch(
                "recursive public output differs from inherited setup claims",
            ));
        }
        Ok(Self { level, statement })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert valid setup claim construction"
)]
mod tests {
    use super::*;
    use akita_types::setup_claims::{SetupClaim, SetupPolynomialId, SetupQuery};
    use akita_types::{derive_public_matrix_prefix, AkitaExpandedSetup, AkitaSetupDescriptor};
    use jolt_field::Ring;

    #[test]
    fn inherited_setup_rejects_omitted_reordered_and_changed_public_claims() {
        let seed = [23; 32].into();
        let setup = AkitaExpandedSetup::<AkitaField>::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 4,
                max_num_batched_polys: 1,
                num_field_elements: 16,
                setup_seed: seed,
            },
            derive_public_matrix_prefix(16, &[23; 32].into()),
        );
        let domain = SetupPolynomialId::new::<AkitaField, AkitaField>(&setup, 16).unwrap();
        let claims = [2, 3].map(|value| {
            SetupClaim::new(
                domain.clone(),
                SetupQuery::evaluation(vec![AkitaField::from_u64(value); 4]).unwrap(),
                AkitaField::from_u64(value),
            )
            .unwrap()
        });
        let statement = SetupBatchStatement::new(domain.clone(), 2, claims.to_vec()).unwrap();
        let output = statement.descriptor_bytes::<AkitaField>().unwrap();
        assert!(InheritedSetupClaims::from_public_output(0, statement, &output).is_ok());
        for altered in [
            vec![],
            vec![claims[0].clone()],
            vec![claims[1].clone(), claims[0].clone()],
        ] {
            let statement = SetupBatchStatement::new(domain.clone(), 2, altered).unwrap();
            assert!(InheritedSetupClaims::from_public_output(0, statement, &output).is_err());
        }
        let mut changed = output;
        *changed.last_mut().unwrap() ^= 1;
        let statement = SetupBatchStatement::new(domain, 2, claims.to_vec()).unwrap();
        assert!(InheritedSetupClaims::from_public_output(0, statement, &changed).is_err());
    }
}
