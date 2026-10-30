#[path = "bounded.rs"]
mod bounded;
use serde::de::DeserializeOwned;

// Bound decoded records and bincode's byte budget before allocating.
const MAX_RECORD_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub(crate) struct RecordError;

/// Length-prefixed records, padded with zeroes to eight bytes relative to base.
pub(crate) struct Records<'a> {
    buffer: &'a [u8],
    offset: usize,
    base: usize,
}

impl<'a> Records<'a> {
    pub(crate) fn new(buffer: &'a [u8]) -> Result<Self, RecordError> {
        let mut records = Self { buffer, offset: 0, base: 0 };
        let pad = records.len_prefix()?;
        if pad >= 8 {
            return Err(RecordError);
        }
        let end = records.offset.checked_add(pad).ok_or(RecordError)?;
        let padding = buffer.get(records.offset..end).ok_or(RecordError)?;
        if padding.iter().any(|byte| *byte != 0)
            || (buffer.as_ptr() as usize).checked_add(end).ok_or(RecordError)? % 8 != 0
        {
            return Err(RecordError);
        }
        records.offset = end;
        records.base = end;
        Ok(records)
    }

    fn len_prefix(&mut self) -> Result<usize, RecordError> {
        let end = self.offset.checked_add(8).ok_or(RecordError)?;
        let bytes = self.buffer.get(self.offset..end).ok_or(RecordError)?;
        let mut length = [0; 8];
        length.copy_from_slice(bytes);
        self.offset = end;
        usize::try_from(u64::from_le_bytes(length)).map_err(|_| RecordError)
    }

    pub(crate) fn raw(&mut self) -> Result<&'a [u8], RecordError> {
        let len = self.len_prefix()?;
        let end = self.offset.checked_add(len).ok_or(RecordError)?;
        let bytes = self.buffer.get(self.offset..end).ok_or(RecordError)?;
        let padded = end.checked_sub(self.base)
            .and_then(|len| len.checked_next_multiple_of(8))
            .and_then(|len| self.base.checked_add(len))
            .ok_or(RecordError)?;
        let padding = self.buffer.get(end..padded).ok_or(RecordError)?;
        if padding.iter().any(|byte| *byte != 0) {
            return Err(RecordError);
        }
        self.offset = padded;
        Ok(bytes)
    }

    pub(crate) fn record<T: DeserializeOwned>(&mut self) -> Result<T, RecordError> {
        let bytes = self.raw()?;
        if bytes.len() > MAX_RECORD_BYTES {
            return Err(RecordError);
        }
        let (value, consumed) = bounded::decode(bytes).map_err(|_| RecordError)?;
        if consumed != bytes.len() {
            return Err(RecordError);
        }
        Ok(value)
    }

    pub(crate) fn check_count(&self, count: u32, records_per_item: usize) -> Result<(), RecordError> {
        let minimum = usize::try_from(count).ok()
            .and_then(|count| count.checked_mul(records_per_item))
            .and_then(|count| count.checked_mul(8))
            .ok_or(RecordError)?;
        if minimum > self.buffer.len().saturating_sub(self.offset) {
            return Err(RecordError);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_sdk::JoltDevice;
    #[cfg(feature = "akita")]
    use jolt_akita::AkitaField as TestField;
    #[cfg(not(feature = "akita"))]
    use jolt_sdk::VerifierField as TestField;

    fn frame(body: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0; 23 + body.len().next_multiple_of(8)];
        let pad = (8 - (bytes.as_ptr() as usize + 8) % 8) % 8;
        bytes[..8].copy_from_slice(&(pad as u64).to_le_bytes());
        let base = 8 + pad;
        bytes[base..base + 8].copy_from_slice(&(body.len() as u64).to_le_bytes());
        bytes[base + 8..base + 8 + body.len()].copy_from_slice(body);
        bytes.truncate(base + 8 + body.len().next_multiple_of(8));
        bytes
    }

    fn verdict<T: DeserializeOwned>(bytes: &[u8]) -> u32 {
        Records::new(bytes).and_then(|mut records| records.record::<T>())
            .map(|_| 1).unwrap_or(0)
    }

    #[test]
    fn malformed_framing_rejects_without_panicking() {
        let valid = frame(&[42u8]);
        assert_eq!(verdict::<u8>(&valid), 1);
        for end in 0..valid.len() {
            assert_eq!(verdict::<u8>(&valid[..end]), 0, "truncation at {end}");
        }
        let base = Records::new(&valid).unwrap().base;
        let mut oversized = valid.clone();
        oversized[base..base + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(verdict::<u8>(&oversized), 0);
        oversized[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(verdict::<u8>(&oversized), 0);
        let mut padding = valid.clone();
        *padding.last_mut().unwrap() = 1;
        assert_eq!(verdict::<u8>(&padding), 0);
        assert_eq!(verdict::<u8>(&frame(&[42, 43])), 0);
        assert!(Records::new(&valid).unwrap().check_count(u32::MAX, 2).is_err());
    }

    #[test]
    fn nested_collection_lengths_are_bounded_before_allocation() {
        let mut body = vec![253];
        body.extend_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(verdict::<Vec<Vec<u8>>>(&frame(&body)), 0);
        assert_eq!(verdict::<Vec<()>>(&frame(&body)), 0);
        // A compact wire length must not reserve a large vector of large items.
        let body = bincode::serde::encode_to_vec(100_000usize, bincode::config::standard()).unwrap();
        assert_eq!(verdict::<Vec<[[u64; 32]; 32]>>(&frame(&body)), 0);
        let count = 400_000usize;
        let mut body = bincode::serde::encode_to_vec(count, bincode::config::standard()).unwrap();
        body.resize(body.len() + count, 0); // Valid empty vectors, excessive aggregate storage.
        assert_eq!(verdict::<Vec<Vec<u8>>>(&frame(&body)), 0);
        let nested = vec![vec![1u8, 2, 3], vec![], vec![4]];
        let body = bincode::serde::encode_to_vec(&nested, bincode::config::standard()).unwrap();
        let decoded: Vec<Vec<u8>> = Records::new(&frame(&body)).unwrap().record().unwrap();
        assert_eq!(decoded, nested);
    }

    #[test]
    fn noncanonical_field_and_malformed_devices_reject() {
        let size = if cfg!(feature = "akita") { 16 } else { 32 };
        assert_eq!(verdict::<TestField>(&frame(&vec![0; size])), 1);
        assert_eq!(verdict::<TestField>(&frame(&vec![255; size])), 0);
        // Four empty byte vectors followed by an invalid boolean.
        assert_eq!(verdict::<JoltDevice>(&frame(&[0, 0, 0, 0, 2])), 0);
        // A vector length that cannot fit in the body, including the u64 tag.
        let mut huge_vector = vec![253];
        huge_vector.extend_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(verdict::<JoltDevice>(&frame(&huge_vector)), 0);
        let device = bincode::serde::encode_to_vec(JoltDevice::default(), bincode::config::standard()).unwrap();
        assert_eq!(verdict::<JoltDevice>(&frame(&device)), 1);
        for end in 0..device.len() {
            assert_eq!(verdict::<JoltDevice>(&frame(&device[..end])), 0);
        }
    }
}
