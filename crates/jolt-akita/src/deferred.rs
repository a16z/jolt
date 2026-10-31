//! Conditional inner verification and inherited setup obligations.
//!
//! A successful deferred replay is not acceptance. The enclosing execution
//! proof must publish the complete ordered statement and its verifier must
//! check that same statement in the outer Akita opening.

use std::fmt::{Debug, Formatter, Result as FormatResult};
use std::sync::Arc;

use akita_params::FoldSchedule;
use akita_prover::backend::SetupBatchBackend;
use akita_types::setup_claims::{
    SetupBatchMemory, SetupBatchStatement, SetupPolynomialId, SetupQuery,
};
use akita_types::{derive_public_matrix_prefix, AkitaExpandedSetup};
pub use akita_verifier::{SetupClaimAccumulator, SetupClaimAdvice, SetupMode};

use crate::adapters::{
    akita_error, invalid_batch, one_hot_setup_prover, one_hot_setup_verifier, AkitaBackend,
};
use crate::one_hot_family::with_one_hot_family;
use crate::{AkitaBatchProof, AkitaError, AkitaField, AkitaProverSetup, AkitaVerifierSetup};
use jolt_openings::OpeningsError;

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

    pub fn accumulator(&self) -> SetupClaimAccumulator<'static, AkitaField> {
        SetupClaimAccumulator::from_batch(self.level, self.statement.clone())
    }
}

impl AkitaProverSetup {
    /// Provision an actual prefix before making any commitments.
    pub fn with_setup_domain(mut self, field_len: usize) -> Result<Self, OpeningsError> {
        if !field_len.is_power_of_two() {
            return Err(invalid_batch("setup domain must be a power of two"));
        }
        let mut prover = one_hot_setup_prover(&self.verifier)?;
        if prover.expanded.descriptor().num_field_elements < field_len {
            let mut descriptor = prover.expanded.descriptor().clone();
            descriptor.num_field_elements = field_len;
            let matrix = derive_public_matrix_prefix(field_len, &descriptor.setup_seed);
            prover.expanded = Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(descriptor, matrix),
            );
        }
        let public = one_hot_setup_verifier(&self.verifier, &prover)?;
        self.verifier.backend_cache = Default::default();
        self.one_hot_cpu_backend = Some(Arc::new(
            AkitaBackend::new(Arc::clone(&prover.expanded)).map_err(akita_error)?,
        ));
        self.one_hot_backend_prover_setup = Some(Arc::new(prover));
        self.verifier.prime_backend_cache(None, Some(public))?;
        Ok(self)
    }

    pub fn required_setup_domain_len(&self) -> Result<usize, OpeningsError> {
        self.one_hot_backend()?
            .0
            .expanded
            .descriptor()
            .num_field_elements
            .checked_next_power_of_two()
            .ok_or_else(|| invalid_batch("setup domain overflows"))
    }

    pub fn with_inherited_setup(mut self, claims: InheritedSetupClaims) -> Self {
        self.inherited_setup = Some(Arc::new(claims));
        self
    }

    /// Host-side advice generation. Its answers remain unauthenticated until
    /// the outer opening discharges the guest's exported obligations.
    pub fn prepare_setup_answer(
        &self,
        polynomial: &SetupPolynomialId,
        query: &SetupQuery<AkitaField>,
    ) -> Result<AkitaField, AkitaError> {
        let (_, backend) = self
            .one_hot_backend()
            .map_err(|e| AkitaError::InvalidInput(e.to_string()))?;
        backend
            .prepare_setup_query(polynomial, query, 2, SetupBatchMemory::Regenerate)
            .map(|(value, _)| value)
    }
}

impl AkitaVerifierSetup {
    pub fn setup_offloads(
        &self,
        proof: &AkitaBatchProof,
    ) -> Result<Vec<(u32, usize)>, OpeningsError> {
        let scheme = self.one_hot_scheme()?;
        with_one_hot_family!(scheme scheme, |scheme| {
            scheme.schedules().resolve_selection(proof.selection()).and_then(|row| setup_offloads(row.schedule()))
        }).map_err(akita_error)
    }

    pub fn materialized_setup_field_len(&self) -> Result<usize, OpeningsError> {
        let verifier = self.one_hot_verifier()?;
        Ok(
            with_one_hot_family!(verifier verifier, |verifier| verifier.setup().expanded().descriptor().num_field_elements),
        )
    }

    pub fn setup_polynomial(&self, field_len: usize) -> Result<SetupPolynomialId, OpeningsError> {
        let verifier = self.one_hot_verifier()?;
        with_one_hot_family!(verifier verifier, |verifier| {
            SetupPolynomialId::new::<AkitaField, AkitaField>(verifier.setup().expanded(), field_len)
        })
        .map_err(akita_error)
    }
}

fn setup_offloads(schedule: &FoldSchedule) -> Result<Vec<(u32, usize)>, AkitaError> {
    schedule
        .recursive_folds
        .iter()
        .enumerate()
        .filter_map(|(level, fold)| {
            fold.params
                .setup_prefix()
                .and_then(|prefix| prefix.slot_id())
                .map(|id| {
                    Ok((
                        u32::try_from(level).map_err(|_| AkitaError::InvalidProof)?,
                        id.n_prefix()?,
                    ))
                })
        })
        .collect()
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert valid setup claim construction"
)]
mod tests {
    use super::*;
    use akita_types::setup_claims::SetupClaim;
    use akita_types::AkitaSetupDescriptor;
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
