//! Framed Blake2b-512 counter expansion for fixed public read schedules.
use jolt_field::Fr;
use jolt_r1cs::{bn254_bits::ByteVar, R1csBuilder};

use super::{blake2b512, Blake2bR1csError};

/// Capacity of all 2^64 counter blocks, including block u64::MAX.
pub const BLAKE2B_STREAM_BYTES: u128 = 1u128 << 70;

/// Constrained H(u32le(domain length) || domain || u64le(context length) ||
/// context || u64le(block index)), starting at block zero.
///
/// Domain, lengths, positions and read schedule are public circuit shape.
/// Context bytes are constrained handles; callers must bind their provenance.
/// ONE must be fixed externally to one. All handles and calls must use the same
/// builder: index checks do not establish builder provenance. Returned bytes
/// must be bound to the consumer's relation. No sampling is implemented here.
#[derive(Clone, Debug)]
pub struct Blake2bStreamVar {
    prefix: Vec<ByteVar>,
    position: u128,
    block: Option<[ByteVar; 64]>,
}

impl Blake2bStreamVar {
    /// Frame a public domain and a fixed-length constrained context.
    pub fn new(
        builder: &R1csBuilder<Fr>,
        domain: &[u8],
        context: &[ByteVar],
    ) -> Result<Self, Blake2bR1csError> {
        let domain_len =
            u32::try_from(domain.len()).map_err(|_| Blake2bR1csError::MessageTooLong)?;
        let context_len =
            u64::try_from(context.len()).map_err(|_| Blake2bR1csError::MessageTooLong)?;
        for byte in context {
            byte.validate_indices(builder)?;
        }
        let capacity = domain
            .len()
            .checked_add(context.len())
            .and_then(|n| n.checked_add(20))
            .ok_or(Blake2bR1csError::MessageTooLong)?;
        let mut prefix = Vec::with_capacity(capacity);
        prefix.extend(domain_len.to_le_bytes().map(ByteVar::constant));
        prefix.extend(domain.iter().copied().map(ByteVar::constant));
        prefix.extend(context_len.to_le_bytes().map(ByteVar::constant));
        prefix.extend_from_slice(context);
        Ok(Self {
            prefix,
            position: 0,
            block: None,
        })
    }

    /// Return the next bytes, retaining unused block suffixes across calls.
    /// Empty reads are no-ops, including at exhaustion. An over-capacity read
    /// returns StreamExhausted before changing state or emitting constraints.
    pub fn read(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
        len: usize,
    ) -> Result<Vec<ByteVar>, Blake2bR1csError> {
        let end = self
            .position
            .checked_add(len as u128)
            .filter(|&end| end <= BLAKE2B_STREAM_BYTES)
            .ok_or(Blake2bR1csError::StreamExhausted)?;
        if len == 0 {
            return Ok(Vec::new());
        }
        for byte in self.prefix.iter().chain(self.block.iter().flatten()) {
            byte.validate_indices(builder)?;
        }
        let mut output = Vec::with_capacity(len);
        while output.len() < len {
            let offset = (self.position % 64) as usize;
            if offset == 0 {
                let index = u64::try_from(self.position / 64)
                    .map_err(|_| Blake2bR1csError::StreamExhausted)?;
                let mut message = self.prefix.clone();
                message.extend(index.to_le_bytes().map(ByteVar::constant));
                self.block = Some(blake2b512(builder, &message)?);
            }
            let block = self.block.as_ref().ok_or(Blake2bR1csError::InvalidLength)?;
            let take = (64 - offset).min(len - output.len());
            output.extend(block.iter().skip(offset).take(take).cloned());
            self.position += take as u128;
        }
        debug_assert_eq!(self.position, end);
        Ok(output)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "fixed vectors and explicit boundary state injection"
)]
mod tests {
    use super::*;
    use jolt_field::Ring;
    use jolt_r1cs::LinearCombination;

    #[test]
    fn final_counter_and_atomic_exhaustion() {
        let mut builder = R1csBuilder::new();
        let context = [ByteVar::allocate(&mut builder, Some(7))];
        let mut stream = Blake2bStreamVar::new(&builder, b"stream-boundary", &context).unwrap();
        stream.position = BLAKE2B_STREAM_BYTES - 64;
        let before = builder.num_vars();
        assert!(matches!(
            stream.read(&mut builder, 65),
            Err(Blake2bR1csError::StreamExhausted)
        ));
        assert_eq!(builder.num_vars(), before);
        assert_eq!(stream.position, BLAKE2B_STREAM_BYTES - 64);
        assert!(stream.block.is_none());
        let mut output = stream.read(&mut builder, 63).unwrap();
        assert!(stream.read(&mut builder, 0).unwrap().is_empty());
        let before = builder.num_vars();
        assert!(matches!(
            stream.read(&mut builder, 2),
            Err(Blake2bR1csError::StreamExhausted)
        ));
        assert_eq!(builder.num_vars(), before);
        output.extend(stream.read(&mut builder, 1).unwrap());
        assert!(stream.read(&mut builder, 0).unwrap().is_empty());
        assert!(matches!(
            stream.read(&mut builder, 1),
            Err(Blake2bR1csError::StreamExhausted)
        ));
        let hex = "b69ac7569b996cb9010b7a60f4d3dcf9d705283fea99cf901bc389ed53e93805ecc7ef1770ee6392353258f84dd9f8163422be9f5504994a8c8805de65540348";
        for (byte, pair) in output.iter().zip(hex.as_bytes().chunks_exact(2)) {
            let expected = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap();
            builder.assert_equal(
                byte.expression(),
                LinearCombination::constant(Fr::from_u64(expected.into())),
            );
        }
        let witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
    }

    #[test]
    fn wrong_builder_index_fails_before_emission() {
        let mut source = R1csBuilder::new();
        let input = [ByteVar::allocate(&mut source, Some(7))];
        let mut stream = Blake2bStreamVar::new(&source, b"domain", &input).unwrap();
        let mut other = R1csBuilder::new();
        assert!(stream.read(&mut other, 1).is_err());
        assert_eq!(other.num_vars(), 1);
        assert_eq!(stream.position, 0);
        assert!(Blake2bStreamVar::new(&other, b"domain", &input).is_err());
    }
}
