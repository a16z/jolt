use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaField, AkitaScheme, DenseSource, PrecommittedShape, WhiteoutCapacity};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::lattice::FieldIncLayout;
use jolt_claims::protocols::jolt::geometry::dimensions::{
    CommitmentMatrixShape, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::lattice::advice_packing_plan;
use jolt_claims::protocols::jolt::lattice::increments::FusedIncLayout;
use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_program::preprocess::{compute_max_ram_k, compute_min_ram_k};
use jolt_verifier::config::{JoltProtocolConfig, ZkConfig};
use jolt_verifier::proof::{JoltProof, JoltProofClaims, JoltStageProofs};
use jolt_verifier::stages::pow::StagePowNonces;
use jolt_verifier::verifier::validate_inputs;
use jolt_verifier::whiteout::{pad_layout, whiteout_frame};
use jolt_zk::masking::MaskedMessages;

use super::one_hot_trace_setup_shape;
use super::preprocessing::{AkitaProverPreprocessing, AkitaVc, AkitaVerifierPreprocessing};
use crate::{PreprocessingError, ProverConfig};

pub(crate) fn provision(
    preprocessing: &mut AkitaProverPreprocessing,
    config: &ProverConfig,
    untrusted: bool,
    trusted: bool,
) -> Result<(), PreprocessingError> {
    let failure = |error: String| PreprocessingError::InvalidConfiguration { reason: error };
    let program = &preprocessing.verifier.program;
    let minimum = compute_min_ram_k(
        program.min_bytecode_address(),
        program.program_image_len_words(),
        program.memory_layout(),
    )
    .map_err(|error| failure(error.to_string()))?;
    let maximum =
        compute_max_ram_k(program.memory_layout()).map_err(|error| failure(error.to_string()))?;
    let mut capacity = WhiteoutCapacity::default();
    let mut covered_requested = false;
    for log_k in minimum.ilog2()..=maximum.ilog2() {
        let Some(variant) =
            config.whiteout_ram_variant(1usize << log_k, program.max_padded_trace_length())?
        else {
            continue;
        };
        let (shape, digest, _) = one_hot_trace_setup_shape(&variant, program.bytecode_len())
            .map_err(|error| failure(error.to_string()))?;
        let setup = &preprocessing.verifier.pcs_setup;
        if shape.num_vars != setup.max_num_vars()
            || shape.num_polys != setup.max_num_polys_per_commitment_group()
            || digest != setup.default_layout_digest()
        {
            continue;
        }
        for advice in [false, true]
            .into_iter()
            .filter(|present| !present || untrusted)
        {
            capacity.include(
                replay(&preprocessing.verifier, &variant, advice, trusted).map_err(failure)?,
            );
        }
        covered_requested |= variant == *config;
    }
    if !covered_requested {
        return Err(failure(
            "Whiteout capacity replay did not cover the requested configuration: no valid RAM variant with its shape and layout digest".into(),
        ));
    }
    preprocessing.verifier.pcs_setup = preprocessing
        .pcs_setup
        .set_whiteout_capacity(capacity)
        .map_err(|error| failure(error.to_string()))?;
    Ok(())
}

impl ProverConfig {
    fn whiteout_ram_variant(
        &self,
        ram_k: usize,
        max_trace_length: usize,
    ) -> Result<Option<Self>, PreprocessingError> {
        let variant = if ram_k == self.ram_K {
            *self
        } else {
            let mut variant = Self::from_padded_dimensions::<AkitaField>(
                self.trace_length,
                ram_k,
                max_trace_length,
            )
            .map_err(|error| PreprocessingError::InvalidConfiguration {
                reason: error.to_string(),
            })?;
            variant.one_hot_config = self.one_hot_config;
            variant.trace_polynomial_order = self.trace_polynomial_order;
            variant.akita_chunk_profile = self.akita_chunk_profile;
            if self.rw_config
                != crate::config::read_write_config(
                    self.trace_length.ilog2() as usize,
                    self.ram_K.ilog2() as usize,
                )
            {
                variant.rw_config = self.rw_config;
            }
            variant
        };
        let log_t = variant.trace_length.ilog2() as usize;
        for dimensions in [
            variant
                .rw_config
                .ram_dimensions(log_t, ram_k.ilog2() as usize),
            variant
                .rw_config
                .register_dimensions(log_t, REGISTER_ADDRESS_BITS),
        ] {
            if dimensions.validate_phase_split().is_err() {
                return Ok(None);
            }
        }
        Ok(Some(variant))
    }
}

/// ShapeTranscript and Akita's counting replay ignore values: zero commitments
/// with the admitted profiles give the same pad layout as any proof of this shape.
fn replay(
    preprocessing: &AkitaVerifierPreprocessing,
    config: &ProverConfig,
    untrusted: bool,
    trusted: bool,
) -> Result<WhiteoutCapacity, String> {
    let program = &preprocessing.program;
    let memory = program.memory_layout();
    let mut requests = Vec::new();
    for (present, kind, bytes) in [
        (
            untrusted,
            JoltAdviceKind::Untrusted,
            memory.max_untrusted_advice_size,
        ),
        (
            trusted,
            JoltAdviceKind::Trusted,
            memory.max_trusted_advice_size,
        ),
    ] {
        if present {
            let max_bytes = usize::try_from(bytes).map_err(|error| error.to_string())?;
            let words = CommitmentMatrixShape::advice_from_max_bytes(max_bytes).total_vars();
            let plan = advice_packing_plan(kind, words).map_err(|error| error.to_string())?;
            requests.push((
                PrecommittedShape {
                    num_vars: plan.packing().packed_num_vars(),
                    source: DenseSource::Bounded,
                },
                plan.layout_digest(),
            ));
        }
    }
    let log_t = config.trace_length.ilog2() as usize;
    #[cfg(feature = "field-inline")]
    {
        let layout = FieldIncLayout::new(log_t);
        requests.push((
            PrecommittedShape {
                num_vars: layout.num_vars(),
                source: DenseSource::FullField,
            },
            layout.layout_digest(),
        ));
    }
    if let Some(committed) = program.committed() {
        for commitment in [
            &committed.bytecode_commitment,
            &committed.program_image_commitment,
        ] {
            requests.push((
                PrecommittedShape {
                    num_vars: commitment.num_vars(),
                    source: DenseSource::Bounded,
                },
                commitment.layout_digest(),
            ));
        }
    }
    let fused = FusedIncLayout::new(log_t);
    requests.push((
        PrecommittedShape {
            num_vars: fused.num_vars(),
            source: DenseSource::Bounded,
        },
        fused.layout_digest(),
    ));
    let (trace, mut objects) = preprocessing
        .pcs_setup
        .whiteout_zero_commitments(&requests)
        .map_err(|error| error.to_string())?;
    let fused = objects.pop().ok_or("missing fused increment shape")?;
    let mut objects = objects.into_iter();
    let untrusted_commitment = if untrusted {
        Some(objects.next().ok_or("missing untrusted advice shape")?)
    } else {
        None
    };
    let trusted_commitment = if trusted {
        Some(objects.next().ok_or("missing trusted advice shape")?)
    } else {
        None
    };
    #[cfg(feature = "field-inline")]
    let field = objects.next().ok_or("missing field increment shape")?;
    let proof: JoltProof<AkitaScheme, AkitaVc, (), ()> = JoltProof {
        protocol: JoltProtocolConfig::for_zk(ZkConfig::Whiteout),
        commitments: trace,
        fused_inc_commitment: Some(fused),
        #[cfg(feature = "field-inline")]
        field_inc_commitment: Some(field),
        stages: JoltStageProofs::masked(StagePowNonces::default()),
        joint_opening_proof: (),
        untrusted_advice_commitment: untrusted_commitment,
        claims: JoltProofClaims::Zk {
            masked: MaskedMessages::default(),
            zk_proof: (),
        },
        trace_length: config.trace_length,
        ram_K: config.ram_K,
        rw_config: config.rw_config,
        one_hot_config: config.one_hot_config,
        trace_polynomial_order: config.trace_polynomial_order,
        narg: Vec::new(),
    };
    let public = JoltDevice {
        memory_layout: memory.clone(),
        ..JoltDevice::default()
    };
    let checked = validate_inputs(preprocessing, &public, &proof, trusted)
        .map_err(|error| error.to_string())?;
    let frame = whiteout_frame(preprocessing, &proof, &checked, trusted_commitment.as_ref())
        .map_err(|error| error.to_string())?;
    let layout =
        pad_layout(preprocessing, &proof, &checked, &frame).map_err(|error| error.to_string())?;
    frame
        .shape
        .tail()
        .requirements(layout.layout())
        .map_err(|error| error.to_string())
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests assert configuration outcomes")]
mod tests {
    use super::*;

    #[test]
    fn custom_read_write_split_skips_only_incompatible_ram_variants() {
        let mut config =
            ProverConfig::from_padded_dimensions::<AkitaField>(1 << 16, 1 << 17, 1 << 16).unwrap();
        config.rw_config.ram_rw_phase2_num_rounds = 16;
        assert_eq!(config.whiteout_ram_variant(1 << 15, 1 << 16).unwrap(), None);
        assert_eq!(
            config.whiteout_ram_variant(1 << 17, 1 << 16).unwrap(),
            Some(config)
        );
        assert_eq!(
            config
                .whiteout_ram_variant(1 << 16, 1 << 16)
                .unwrap()
                .unwrap()
                .rw_config,
            config.rw_config,
        );
    }
}
