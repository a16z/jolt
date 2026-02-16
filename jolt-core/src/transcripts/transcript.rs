use crate::curve::JoltGroupElement;
use crate::field::JoltField;
use ark_serialize::CanonicalSerialize;
use std::borrow::Borrow;

/// Maximum label length when packed with a length/count.
/// 32 bytes total - 8 bytes for u64 length = 24 bytes for label.
const MAX_LABEL_LEN_WITH_LENGTH: usize = 24;

pub trait Transcript: Default + Clone + Sync + Send + 'static {
    fn new(label: &'static [u8]) -> Self;
    #[cfg(test)]
    fn compare_to(&mut self, other: Self);

    // === Internal methods (implementors provide these) ===
    // These preserve EVM-compatible serialization logic

    #[doc(hidden)]
    fn raw_append_label(&mut self, label: &'static [u8]);

    /// Pack label (right-padded, 24 bytes) and length (big-endian, 8 bytes) into 32 bytes.
    /// Used only for length/count prefixes in variable-length methods.
    #[doc(hidden)]
    fn raw_append_label_with_len(&mut self, label: &'static [u8], len: u64) {
        assert!(
            label.len() <= MAX_LABEL_LEN_WITH_LENGTH,
            "Label too long for packed format: {} > {}",
            label.len(),
            MAX_LABEL_LEN_WITH_LENGTH
        );
        let mut packed = [0u8; 32];
        packed[..label.len()].copy_from_slice(label);
        // Zero-pad label portion (already zeroed)
        // Append length as big-endian in last 8 bytes
        packed[24..32].copy_from_slice(&len.to_be_bytes());
        self.raw_append_bytes(&packed);
    }

    #[doc(hidden)]
    fn raw_append_bytes(&mut self, bytes: &[u8]);

    #[doc(hidden)]
    fn raw_append_u64(&mut self, x: u64);

    #[doc(hidden)]
    fn raw_append_scalar<F: JoltField>(&mut self, scalar: &F);

    // === Public API - Labels required ===

    /// Append a domain-separation label with no associated data.
    fn append_label(&mut self, label: &'static [u8]) {
        self.raw_append_label(label);
    }

    /// Append raw bytes with a label.
    /// Variable-length: label and length packed into single 32-byte word.
    fn append_bytes(&mut self, label: &'static [u8], bytes: &[u8]) {
        self.raw_append_label_with_len(label, bytes.len() as u64);
        self.raw_append_bytes(bytes);
    }

    /// Append a u64 value with a label.
    /// Two separate 32-byte words: label (right-padded) + value (left-padded for EVM uint256).
    fn append_u64(&mut self, label: &'static [u8], x: u64) {
        self.raw_append_label(label);
        self.raw_append_u64(x);
    }

    /// Append a scalar field element with a label.
    /// Fixed-size: no length prefix needed.
    fn append_scalar<F: JoltField>(&mut self, label: &'static [u8], scalar: &F) {
        self.raw_append_label(label);
        self.raw_append_scalar(scalar);
    }

    /// Append a curve point with a label (compressed serialization).
    /// Fixed-size: no length prefix needed.
    fn append_point<G: JoltGroupElement>(&mut self, label: &'static [u8], point: &G) {
        self.raw_append_label(label);
        let mut bytes = Vec::new();
        point
            .serialize_compressed(&mut bytes)
            .expect("JoltGroupElement serialization should not fail");
        self.raw_append_bytes(&bytes);
    }

    /// Append a serializable value with a label.
    /// Variable-length: label and length packed into single 32-byte word.
    fn append_serializable<T: CanonicalSerialize>(&mut self, label: &'static [u8], data: &T) {
        let mut buf = vec![];
        data.serialize_uncompressed(&mut buf).unwrap();
        self.raw_append_label_with_len(label, buf.len() as u64);
        // Reverse for EVM big-endian compatibility
        buf.reverse();
        self.raw_append_bytes(&buf);
    }

    /// Append a slice of scalars with a label.
    /// Variable-length: label and count packed into single 32-byte word.
    fn append_scalars<F: JoltField>(&mut self, label: &'static [u8], scalars: &[impl Borrow<F>]) {
        self.raw_append_label_with_len(label, scalars.len() as u64);
        for s in scalars {
            self.raw_append_scalar(s.borrow());
        }
    }

    /// Append a slice of curve points with a label (compressed serialization).
    /// Variable-length: label and count packed into single 32-byte word.
    fn append_points<G: JoltGroupElement>(&mut self, label: &'static [u8], points: &[G]) {
        self.raw_append_label_with_len(label, points.len() as u64);
        for p in points {
            let mut bytes = Vec::new();
            p.serialize_compressed(&mut bytes)
                .expect("JoltGroupElement serialization should not fail");
            self.raw_append_bytes(&bytes);
        }
    }

    /// Append a slice of `CanonicalSerialize` points with a label (compressed serialization).
    /// Same layout as `append_points` but works with any serializable type (e.g. arkworks affine points).
    fn append_points_serializable<T: CanonicalSerialize>(
        &mut self,
        label: &'static [u8],
        points: &[T],
    ) {
        self.raw_append_label_with_len(label, points.len() as u64);
        for p in points {
            let mut bytes = Vec::new();
            p.serialize_compressed(&mut bytes)
                .expect("Point serialization should not fail");
            self.raw_append_bytes(&bytes);
        }
    }

    // === Challenge generation methods (signatures unchanged) ===

    fn challenge_u128(&mut self) -> u128;
    fn challenge_scalar<F: JoltField>(&mut self) -> F;
    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F;
    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F>;
    /// Compute powers of scalar q : (1, q, q^2, ..., q^(len-1))
    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F>;
    /// Optimized method that returns F::Challenge
    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge;
    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge>;
    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F>;
}
