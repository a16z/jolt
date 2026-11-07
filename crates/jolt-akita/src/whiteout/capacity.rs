use super::*;
use crate::adapters::{serialize_akita, AkitaBackendFlavor, AkitaVerifierScheduleArtifacts};
use crate::{AkitaProverSetup, AkitaVerifierSetup};
use akita_params::sis::SIS_MAX_MODULE_RANK;
use akita_params::FlatMatrix;
use akita_types::{AkitaExpandedSetup, AkitaSetupDescriptor};
use jolt_crypto::lattice::MatrixShape;
use jolt_zk::whiteout::MAX_RESPONSE_NUM_VARS;

// The response contains every matrix input coefficient; allow the maximum SIS rank plus the pad L row.
const MAX_STREAM_COEFFICIENTS: usize =
    (SIS_MAX_MODULE_RANK as usize + 1) * (1 << MAX_RESPONSE_NUM_VARS);

/// Public coefficient limits, not an envelope of accepted proof headers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WhiteoutCapacity {
    pub pad: usize,
    pub v: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhiteoutStream {
    Pad,
    V,
}

#[derive(Debug, thiserror::Error)]
#[error("Whiteout {stream:?} stream needs {required} coefficients, capacity is {available}")]
pub struct WhiteoutCapacityError {
    pub stream: WhiteoutStream,
    pub required: usize,
    pub available: usize,
}

impl WhiteoutCapacity {
    pub(super) fn validate(self) -> Result<(), ZkError> {
        Self {
            pad: MAX_STREAM_COEFFICIENTS,
            v: MAX_STREAM_COEFFICIENTS,
        }
        .check(self)
    }

    pub fn include(&mut self, other: Self) {
        self.pad = self.pad.max(other.pad);
        self.v = self.v.max(other.v);
    }

    pub(super) fn check(self, required: Self) -> Result<(), ZkError> {
        for (stream, required, available) in [
            (WhiteoutStream::Pad, required.pad, self.pad),
            (WhiteoutStream::V, required.v, self.v),
        ] {
            if required > available {
                return Err(ZkError::Whiteout(Box::new(WhiteoutCapacityError {
                    stream,
                    required,
                    available,
                })));
            }
        }
        Ok(())
    }
}

pub(super) struct CapacityStreams {
    pub capacity: WhiteoutCapacity,
    pub pad_data: Arc<StreamCoefficients>,
    pub v_data: Arc<StreamCoefficients>,
    pub pad64: PreparedMatrix<F, 64>,
    pub pad128: PreparedMatrix<F, 128>,
    pub pad256: PreparedMatrix<F, 256>,
    pub v: PreparedMatrix<F, CHALLENGE_RING_DIM>,
}

pub(super) struct StreamCoefficients(pub AkitaExpandedSetup<F>);

impl AsRef<[F]> for StreamCoefficients {
    fn as_ref(&self) -> &[F] {
        self.0.shared_matrix().as_field_slice()
    }
}

impl CapacityStreams {
    pub fn new(capacity: WhiteoutCapacity) -> Result<Self, ZkError> {
        capacity.validate()?;
        let derive = |len, seed| -> Result<Arc<StreamCoefficients>, ZkError> {
            Ok(Arc::new(StreamCoefficients(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    AkitaSetupDescriptor {
                        max_num_vars: 0,
                        max_num_batched_polys: 1,
                        num_field_elements: len,
                        setup_seed: PublicMatrixSeed::shake256_paged_v1(seed),
                    },
                    FlatMatrix::from_flat_data(public_stream(len, seed)?),
                ),
            )))
        };
        Self::from_streams(
            capacity,
            derive(capacity.pad, PAD_STREAM_SEED)?,
            derive(capacity.v, V_STREAM_SEED)?,
        )
    }

    pub fn from_streams(
        capacity: WhiteoutCapacity,
        pad: Arc<StreamCoefficients>,
        v: Arc<StreamCoefficients>,
    ) -> Result<Self, ZkError> {
        for (stream, len, seed) in [
            (&pad, capacity.pad, PAD_STREAM_SEED),
            (&v, capacity.v, V_STREAM_SEED),
        ] {
            if stream.as_ref().as_ref().len() != len
                || stream.0.descriptor().setup_seed != PublicMatrixSeed::shake256_paged_v1(seed)
            {
                return Err(AkitaError::InvalidSetup(
                    "Whiteout stream identity or capacity mismatch".into(),
                )
                .into());
            }
        }
        fn root<const D: usize>(
            data: Arc<StreamCoefficients>,
        ) -> Result<PreparedMatrix<F, D>, ZkError> {
            PreparedMatrix::from_shared(
                MatrixShape {
                    rows: 1,
                    width: data.as_ref().as_ref().len() / D,
                    ring_dim: D,
                },
                data,
            )
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()).into())
        }
        let pad64 = root::<64>(Arc::clone(&pad))?;
        let v_matrix = pad64
            .with_shared(
                MatrixShape {
                    rows: 1,
                    width: capacity.v / CHALLENGE_RING_DIM,
                    ring_dim: CHALLENGE_RING_DIM,
                },
                Arc::clone(&v),
            )
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        Ok(Self {
            capacity,
            pad64,
            pad128: root(Arc::clone(&pad))?,
            pad256: root(Arc::clone(&pad))?,
            v: v_matrix,
            pad_data: pad,
            v_data: v,
        })
    }
}

impl AkitaVerifierSetup {
    pub fn whiteout_capacity(&self) -> Option<WhiteoutCapacity> {
        self.whiteout().map(|setup| setup.capacity)
    }

    pub fn set_whiteout_capacity(&mut self, capacity: WhiteoutCapacity) -> Result<(), ZkError> {
        match Arc::make_mut(&mut self.schedule_artifacts) {
            AkitaVerifierScheduleArtifacts::OneHotWhiteout { whiteout, .. } => {
                whiteout.set_capacity(capacity)?;
                self.backend_cache.whiteout = Default::default();
                Ok(())
            }
            AkitaVerifierScheduleArtifacts::Dense { .. }
            | AkitaVerifierScheduleArtifacts::OneHot { .. }
            | AkitaVerifierScheduleArtifacts::Both { .. } => {
                Err(AkitaError::InvalidSetup("setup has no Whiteout parameters".into()).into())
            }
        }
    }
}

impl AkitaProverSetup {
    pub fn set_whiteout_capacity(
        &mut self,
        capacity: WhiteoutCapacity,
    ) -> Result<AkitaVerifierSetup, ZkError> {
        self.verifier.set_whiteout_capacity(capacity)?;
        Ok(self.verifier.clone())
    }
}

impl AkitaVerifierSetup {
    pub fn whiteout_zero_commitments(
        &self,
        precommitted: &[(PrecommittedShape, [u8; 32])],
    ) -> Result<(AkitaCommitment, Vec<AkitaCommitment>), ZkError> {
        let groups = MaskedOpeningShape {
            precommitted: precommitted.iter().map(|(shape, _)| *shape).collect(),
            num_vars: self.max_num_vars,
            num_polys: self.max_num_polys_per_commitment_group,
        };
        let shape = opening::opening_shape(self, &groups)?;
        let resolved = shape
            .row()
            .catalog
            .scheme
            .schedules()
            .resolve_key(&shape.row().key)?;
        let package = |profile: GroupCommitPhaseParams,
                       digest,
                       one_hot_k|
         -> Result<AkitaCommitment, ZkError> {
            let group = CommittedGroup::<F>::zero(profile)
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
            Ok(AkitaCommitment {
                backend_flavor: if one_hot_k == 0 {
                    AkitaBackendFlavor::Dense
                } else {
                    AkitaBackendFlavor::OneHot
                },
                layout_digest: digest,
                num_vars: profile.group.num_vars(),
                poly_count: profile.group.num_polynomials(),
                one_hot_k,
                backend_coeff_len: group.rows().coeff_len(),
                serialized_backend_bytes: serialize_akita(group.commitment())
                    .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?,
            })
        };
        let profiles = resolved.profiles();
        let trace = package(
            profiles.final_group,
            self.default_layout_digest,
            self.one_hot_k,
        )?;
        let objects = profiles
            .precommitteds
            .iter()
            .zip(precommitted)
            .map(|(profile, (_, digest))| package(*profile, *digest, 0))
            .collect::<Result<_, _>>()?;
        Ok((trace, objects))
    }
}
