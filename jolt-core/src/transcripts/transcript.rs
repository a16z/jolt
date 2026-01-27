use crate::field::JoltField;
use ark_ec::CurveGroup;
use ark_serialize::CanonicalSerialize;
use std::borrow::Borrow;

pub trait Transcript: Default + Clone + Sync + Send + 'static {
    fn new(label: &'static [u8]) -> Self;
    #[cfg(test)]
    fn compare_to(&mut self, other: Self);

    // === Internal methods (implementors provide these) ===
    // These preserve EVM-compatible serialization logic

    #[doc(hidden)]
    fn raw_append_message(&mut self, msg: &'static [u8]);

    #[doc(hidden)]
    fn raw_append_bytes(&mut self, bytes: &[u8]);

    #[doc(hidden)]
    fn raw_append_u64(&mut self, x: u64);

    #[doc(hidden)]
    fn raw_append_scalar<F: JoltField>(&mut self, scalar: &F);

    #[doc(hidden)]
    fn raw_append_point<G: CurveGroup>(&mut self, point: &G);

    // === Public API - Labels required ===

    /// Append raw bytes with a label.
    /// Variable-length: includes length prefix.
    fn append_bytes(&mut self, label: &'static [u8], bytes: &[u8]) {
        self.raw_append_message(label);
        self.raw_append_u64(bytes.len() as u64);
        self.raw_append_bytes(bytes);
    }

    /// Append a u64 value with a label.
    /// Fixed-size: no length prefix needed.
    fn append_u64(&mut self, label: &'static [u8], x: u64) {
        self.raw_append_message(label);
        self.raw_append_u64(x);
    }

    /// Append a scalar field element with a label.
    /// Fixed-size: no length prefix needed.
    fn append_scalar<F: JoltField>(&mut self, label: &'static [u8], scalar: &F) {
        self.raw_append_message(label);
        self.raw_append_scalar(scalar);
    }

    /// Append a curve point with a label.
    /// Fixed-size: no length prefix needed.
    fn append_point<G: CurveGroup>(&mut self, label: &'static [u8], point: &G) {
        self.raw_append_message(label);
        self.raw_append_point(point);
    }

    /// Append a serializable value with a label.
    /// Variable-length: includes length prefix.
    fn append_serializable<T: CanonicalSerialize>(&mut self, label: &'static [u8], data: &T) {
        self.raw_append_message(label);
        let mut buf = vec![];
        data.serialize_uncompressed(&mut buf).unwrap();
        self.raw_append_u64(buf.len() as u64);
        // Reverse for EVM big-endian compatibility
        buf.reverse();
        self.raw_append_bytes(&buf);
    }

    /// Append a slice of scalars with a label.
    /// Variable-length: includes count prefix.
    fn append_scalars<F: JoltField>(&mut self, label: &'static [u8], scalars: &[impl Borrow<F>]) {
        self.raw_append_message(label);
        self.raw_append_u64(scalars.len() as u64);
        for s in scalars {
            self.raw_append_scalar(s.borrow());
        }
    }

    /// Append a slice of curve points with a label.
    /// Variable-length: includes count prefix.
    fn append_points<G: CurveGroup>(&mut self, label: &'static [u8], points: &[G]) {
        self.raw_append_message(label);
        self.raw_append_u64(points.len() as u64);
        for p in points {
            self.raw_append_point(p);
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
