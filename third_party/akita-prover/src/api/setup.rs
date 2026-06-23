//! Prover setup artifact and config-free setup expansion helpers.

use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_serialization::{AkitaSerialize, SerializationError, Valid};
use akita_types::{
    derive_public_matrix_flat, sample_public_matrix_seed, AkitaExpandedSetup, AkitaSetupSeed,
    AkitaVerifierSetup, SetupMatrixEnvelope, SetupPrefixProverRegistry,
    SetupPrefixVerifierRegistry,
};
#[cfg(feature = "zk")]
use akita_types::{derive_zk_b_matrix, derive_zk_d_matrix};
use std::sync::Arc;

/// Prover setup artifact.
///
/// Backend-prepared compute state is intentionally not stored here. Host code
/// prepares a compute backend from the expanded setup when it wants to prove.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaProverSetup<F: FieldCore, const D: usize> {
    /// Expanded matrix stage used by both prover and verifier.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
    /// Preprocessed setup-prefix commitment slots for setup-claim offloading.
    pub prefix_slots: SetupPrefixProverRegistry<F, D>,
}

impl<F: FieldCore, const D: usize> AkitaProverSetup<F, D> {
    /// Generate a prover setup from already-computed setup capacity bounds.
    ///
    /// The caller supplies config-derived capacity bounds. This constructor
    /// owns only the concrete prover artifact: matrix expansion for the chosen
    /// capacity envelope.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity calculation overflows or the setup
    /// descriptor cannot be built.
    #[tracing::instrument(skip_all, name = "AkitaProverSetup::generate_with_capacity")]
    pub fn generate_with_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        setup_envelope: SetupMatrixEnvelope,
    ) -> Result<Self, AkitaError>
    where
        F: CanonicalField + RandomSampling + AkitaSerialize,
    {
        let public_matrix_seed = sample_public_matrix_seed();
        let seed = AkitaSetupSeed {
            max_num_vars,
            max_num_batched_polys,
            gen_ring_dim: D,
            max_setup_len: setup_envelope.max_setup_len,
            #[cfg(feature = "zk")]
            max_zk_b_len: setup_envelope.max_zk_b_len,
            #[cfg(feature = "zk")]
            max_zk_d_len: setup_envelope.max_zk_d_len,
            public_matrix_seed,
        };
        seed.check().map_err(|err| {
            AkitaError::InvalidSetup(format!("setup seed validation failed: {err}"))
        })?;

        let shared_flat =
            derive_public_matrix_flat::<F, D>(setup_envelope.max_setup_len, &public_matrix_seed);
        #[cfg(feature = "zk")]
        let zk_b_matrix =
            derive_zk_b_matrix::<F, D>(setup_envelope.max_zk_b_len, &public_matrix_seed);
        #[cfg(feature = "zk")]
        let zk_d_matrix =
            derive_zk_d_matrix::<F, D>(setup_envelope.max_zk_d_len, &public_matrix_seed);
        let expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                seed,
                shared_flat,
                #[cfg(feature = "zk")]
                zk_b_matrix,
                #[cfg(feature = "zk")]
                zk_d_matrix,
            ),
        );

        Ok(Self {
            expanded,
            prefix_slots: SetupPrefixProverRegistry::new(),
        })
    }

    /// Derive a verifier setup from this prover setup.
    ///
    /// # Errors
    ///
    /// Returns an error if prover prefix-slot metadata cannot be converted into
    /// verifier-visible prefix slots.
    pub fn verifier_setup(&self) -> Result<AkitaVerifierSetup<F>, AkitaError> {
        let mut prefix_slots = SetupPrefixVerifierRegistry::new();
        prefix_slots.replace_from_prover_registry(&self.prefix_slots)?;
        Ok(AkitaVerifierSetup {
            expanded: self.expanded.clone(),
            prefix_slots,
        })
    }

    /// Wrap an already-validated [`AkitaExpandedSetup`] in a prover setup.
    ///
    /// Use this when the caller has already run strict setup validation, for
    /// example through checked setup deserialization. This still re-checks
    /// seed-to-matrix derivation at the trust boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if the expanded setup does not match its seed.
    pub fn from_validated_expanded(expanded: AkitaExpandedSetup<F>) -> Result<Self, AkitaError>
    where
        F: CanonicalField + RandomSampling + Valid,
    {
        expanded.check().map_err(|err| {
            AkitaError::InvalidSetup(format!("expanded setup validation failed: {err}"))
        })?;
        Self::from_seed_validated_expanded(expanded)
    }

    /// Wrap a seed-validated [`AkitaExpandedSetup`] in a prover setup.
    ///
    /// This skips seed-to-matrix rederivation. Use it only when the caller
    /// just verified the matrix with `validate_public_matrix_matches_seed` in
    /// the same trust boundary, such as the disk-cache loader in
    /// `akita-setup`.
    ///
    /// # Errors
    ///
    /// Returns an error if the setup's generation dimension does not match
    /// `D` or its internal shape metadata is malformed.
    pub fn from_seed_validated_expanded(expanded: AkitaExpandedSetup<F>) -> Result<Self, AkitaError>
    where
        F: CanonicalField + Valid,
    {
        expanded.seed().check().map_err(|err| {
            AkitaError::InvalidSetup(format!("expanded setup seed validation failed: {err}"))
        })?;
        expanded.shared_matrix().check().map_err(|err| {
            AkitaError::InvalidSetup(format!("expanded setup matrix validation failed: {err}"))
        })?;
        if expanded.seed().gen_ring_dim != D {
            return Err(AkitaError::InvalidSetup(format!(
                "expanded setup ring dimension {} does not match prover D={D}",
                expanded.seed().gen_ring_dim
            )));
        }
        if expanded.shared_matrix().gen_ring_dim() != expanded.seed().gen_ring_dim {
            return Err(AkitaError::InvalidSetup(
                "expanded setup matrix generation dimension does not match setup seed".to_string(),
            ));
        }
        if expanded.shared_matrix().total_ring_elements() != expanded.seed().max_setup_len {
            return Err(AkitaError::InvalidSetup(
                "expanded setup matrix length does not match setup seed".to_string(),
            ));
        }
        #[cfg(feature = "zk")]
        {
            expanded.zk_b_matrix().check().map_err(|err| {
                AkitaError::InvalidSetup(format!(
                    "expanded setup zkB matrix validation failed: {err}"
                ))
            })?;
            expanded.zk_d_matrix().check().map_err(|err| {
                AkitaError::InvalidSetup(format!(
                    "expanded setup zkD matrix validation failed: {err}"
                ))
            })?;
            if expanded.zk_b_matrix().gen_ring_dim() != expanded.seed().gen_ring_dim {
                return Err(AkitaError::InvalidSetup(
                    "expanded setup zkB matrix generation dimension does not match setup seed"
                        .to_string(),
                ));
            }
            if expanded.zk_d_matrix().gen_ring_dim() != expanded.seed().gen_ring_dim {
                return Err(AkitaError::InvalidSetup(
                    "expanded setup zkD matrix generation dimension does not match setup seed"
                        .to_string(),
                ));
            }
            if expanded.zk_b_matrix().total_ring_elements() != expanded.seed().max_zk_b_len {
                return Err(AkitaError::InvalidSetup(
                    "expanded setup zkB matrix length does not match setup seed".to_string(),
                ));
            }
            if expanded.zk_d_matrix().total_ring_elements() != expanded.seed().max_zk_d_len {
                return Err(AkitaError::InvalidSetup(
                    "expanded setup zkD matrix length does not match setup seed".to_string(),
                ));
            }
        }
        let expanded = Arc::new(expanded);
        expanded.shared_matrix().total_ring_elements_at::<D>()?;
        #[cfg(feature = "zk")]
        {
            expanded.zk_b_matrix().total_ring_elements_at::<D>()?;
            expanded.zk_d_matrix().total_ring_elements_at::<D>()?;
        }
        Ok(Self {
            expanded,
            prefix_slots: SetupPrefixProverRegistry::new(),
        })
    }

    /// Wrap a pre-built [`AkitaExpandedSetup`] in a prover setup.
    ///
    /// # Errors
    ///
    /// Returns an error if the expanded setup is not valid for this field.
    pub fn from_expanded(expanded: AkitaExpandedSetup<F>) -> Result<Self, AkitaError>
    where
        F: CanonicalField + RandomSampling + Valid,
    {
        Self::from_validated_expanded(expanded)
    }
}

impl<F: FieldCore + RandomSampling + Valid + AkitaSerialize, const D: usize> Valid
    for AkitaProverSetup<F, D>
{
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()?;
        self.prefix_slots.check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;

    #[test]
    fn validated_expanded_setup_rejects_mismatched_ring_dimension() {
        let setup = AkitaProverSetup::<Prime128Offset275, 64>::generate_with_capacity(
            8,
            1,
            SetupMatrixEnvelope {
                max_setup_len: 1,
                #[cfg(feature = "zk")]
                max_zk_b_len: 1,
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
            },
        )
        .expect("generate D=64 setup");
        let expanded = (*setup.expanded).clone();

        let err = AkitaProverSetup::<Prime128Offset275, 32>::from_validated_expanded(expanded)
            .expect_err("D=64 setup must not be reinterpreted as D=32");

        assert!(err.to_string().contains("ring dimension 64"));
    }

    #[test]
    fn generate_with_capacity_rejects_zero_setup_len() {
        let zero_len = AkitaProverSetup::<Prime128Offset275, 32>::generate_with_capacity(
            8,
            1,
            SetupMatrixEnvelope {
                max_setup_len: 0,
                #[cfg(feature = "zk")]
                max_zk_b_len: 1,
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
            },
        )
        .expect_err("zero setup length must not produce an undecodable setup");
        assert!(zero_len.to_string().contains("max_setup_len"));
    }

    #[test]
    fn prover_setup_check_validates_prefix_slots() {
        use akita_algebra::CyclotomicRing;
        use akita_types::{
            AkitaCommitmentHint, FlatDigitBlocks, RingCommitment, SetupPrefixSlot,
            SetupPrefixSlotId,
        };

        let mut setup = AkitaProverSetup::<Prime128Offset275, 32>::generate_with_capacity(
            8,
            1,
            SetupMatrixEnvelope {
                max_setup_len: 1,
                #[cfg(feature = "zk")]
                max_zk_b_len: 1,
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
            },
        )
        .expect("generate setup");
        let decomposed = FlatDigitBlocks::<32>::from_blocks(vec![Vec::new()]);
        let recomposed = vec![Vec::new()];
        #[cfg(feature = "zk")]
        let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
            decomposed,
            recomposed,
            FlatDigitBlocks::empty(),
        );
        #[cfg(not(feature = "zk"))]
        let hint =
            AkitaCommitmentHint::singleton_with_recomposed_inner_rows(decomposed, recomposed);
        setup
            .prefix_slots
            .insert(SetupPrefixSlot {
                id: SetupPrefixSlotId {
                    setup_seed_digest: [1u8; 32],
                    d_setup: 32,
                    natural_len: 1,
                    n_prefix: 3,
                    level_params_digest: [2u8; 32],
                },
                natural_len: 1,
                padded_len: 3,
                commitment: RingCommitment {
                    u: vec![CyclotomicRing::zero()],
                },
                hint,
            })
            .expect("insert malformed slot");

        let err = setup
            .check()
            .expect_err("prover setup check must reject invalid prefix slots");
        assert!(err.to_string().contains("n_prefix"));
    }
}
