//! Byte expansion consumed by native FoldDraw's indexed sparse sampler.
use akita_challenges::SPARSE_CHALLENGE_STREAM_DOMAIN;
use jolt_field::Fr;
use jolt_r1cs::{bn254_bits::ByteVar, R1csBuilder};
use jolt_transcript::r1cs::{Blake2bR1csError, Blake2bStreamVar};

/// Fixed-schedule indexed sparse-challenge bytes for native Akita epoch 6.
///
/// Root bytes must come from the constrained/authenticated FoldDraw transcript;
/// allocating honest bytes alone does not establish that binding. Coordinate
/// and read lengths are public circuit shape. ONE and same-builder obligations
/// are inherited from Blake2bStreamVar. This does not constrain position/sign
/// sampling, rejected draws, routing, operator norms or the FoldDraw root itself.
#[derive(Clone, Debug)]
pub struct AkitaSparseStreamVar(Blake2bStreamVar);

impl AkitaSparseStreamVar {
    /// Bind exactly root[32] || coordinate LE-u64 to the native-owned domain.
    pub fn new(
        builder: &R1csBuilder<Fr>,
        root: &[ByteVar; 32],
        coordinate: u64,
    ) -> Result<Self, Blake2bR1csError> {
        let mut context = root.to_vec();
        context.extend(coordinate.to_le_bytes().map(ByteVar::constant));
        Ok(Self(Blake2bStreamVar::new(
            builder,
            SPARSE_CHALLENGE_STREAM_DOMAIN,
            &context,
        )?))
    }

    /// Return the next constrained prefix, preserving unused block bytes.
    pub fn read(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
        len: usize,
    ) -> Result<Vec<ByteVar>, Blake2bR1csError> {
        self.0.read(builder, len)
    }
}
