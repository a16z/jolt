//! Compressed commitment images (Whiteout §2, "Commitment compression"):
//! the verifier holds a short payload `cm(t)` in place of an image `t`, and
//! the response vector `e` carries the compression witness that reconstructs
//! `t` and opens the payload. The response prover owns the compression map
//! (`specs/whiteout-sigma.md` §3.1); the statement reaches it through
//! [`ImageCompression`], and Items 2 and 3 read `t` from the witness.

use jolt_field::JoltField;

use super::WhiteoutError;

/// A compression map and its witness, owned by the response prover.
/// Object-safe, so the statement holds it by reference.
pub trait ImageCompression<F: JoltField>: Sync {
    /// The payload length of an image of `image_len` coefficients.
    fn payload_len(&self, image_len: usize) -> Result<usize, WhiteoutError>;

    /// The witness length, in digits of `e`, of an image of `image_len`
    /// coefficients.
    fn witness_len(&self, image_len: usize) -> Result<usize, WhiteoutError>;

    /// `(cm(t), witness)` for the image `t`.
    fn compress(&self, image: &[F]) -> Result<CompressedImage<F>, WhiteoutError>;

    /// `t` from its witness: a linear map of the witness digits.
    fn reconstruct(&self, image_len: usize, witness: &[i8]) -> Result<Vec<F>, WhiteoutError>;

    /// The transpose of [`Self::reconstruct`]: the weight on each witness
    /// digit of `⟨weights, t⟩`.
    fn reconstruct_transpose(
        &self,
        image_len: usize,
        weights: &[F],
    ) -> Result<Vec<F>, WhiteoutError>;

    /// Whether `witness` opens `payload` as the compression of an image of
    /// `image_len` coefficients: its digits lie in the map's alphabet and
    /// its chain equations hold.
    fn check(&self, image_len: usize, payload: &[F], witness: &[i8]) -> Result<(), WhiteoutError>;
}

/// A compressed image and its witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedImage<F> {
    pub payload: Vec<F>,
    pub witness: Vec<i8>,
}
