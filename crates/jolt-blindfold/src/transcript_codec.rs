use jolt_field::Field;
use jolt_transcript::FsAbsorb;

/// Absorbs a field vector using BlindFold's legacy count-prefixed payload shape:
/// an 8-byte little-endian element count followed by contiguous canonical field
/// bytes. Spongefish provides the outer message boundary; this prefix preserves
/// the old `LabelWithCount` transcript bytes inside that boundary.
pub fn absorb_legacy_field_vec<F, T>(transcript: &mut T, values: &[F])
where
    F: Field,
    T: FsAbsorb,
{
    let mut bytes = vec![0u8; 8 + values.len() * F::NUM_BYTES];
    bytes[..8].copy_from_slice(&(values.len() as u64).to_le_bytes());
    for (chunk, value) in bytes[8..].chunks_exact_mut(F::NUM_BYTES).zip(values) {
        value.to_bytes_le(chunk);
    }
    transcript.absorb_bytes(&bytes);
}
