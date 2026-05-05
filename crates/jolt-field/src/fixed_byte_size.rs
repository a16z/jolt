/// Fixed byte-size metadata for canonical encodings.
pub trait FixedByteSize {
    /// Byte length of the fixed-size encoding.
    const NUM_BYTES: usize;
}
