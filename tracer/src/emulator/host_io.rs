//! Rules every tracer backend applies to a guest's host calls
//! (`VirtualHostIO`). A call this module rejects is logged and ignored, and
//! every backend rejects the same calls. Buffers whose size the guest
//! chooses are reserved fallibly. Out of host memory, a host call can still
//! abort in the bookkeeping of each guest load (the checkpoint and
//! sparse-memory maps in `MemoryData::access_u64`), as the guest's own loads
//! do, and in the `tracing` subscriber.

use core::fmt::{Display, Formatter, Result as FmtResult};
use std::io::{Result as IoResult, Write};

use jolt_platform::{JOLT_PRINT_LINE, JOLT_PRINT_STRING};

/// Exclusive end of the addresses a print or cycle-marker string can occupy:
/// the guest passes those pointers and lengths as `u32`.
pub const HOST_STRING_ADDRESS_LIMIT: u64 = 1 << 32;

/// Guest memory as a host call reads it.
pub trait HostMemory {
    /// Whether the backend's guest load returns normally at every address in
    /// `[start, end)`, for `start < end`.
    fn is_host_readable(&self, start: u64, end: u64) -> bool;

    /// Loads the byte at an address [`HostMemory::is_host_readable`]
    /// accepted.
    fn load_host_byte(&mut self, address: u64) -> Option<u8>;
}

/// Why a host call reads no guest bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostReadError {
    /// The range ends past its address limit or holds a byte a guest load
    /// would reject.
    Unreadable { ptr: u64, len: u64 },
    /// The buffer for the range cannot be allocated.
    Allocation { len: u64 },
}

impl Display for HostReadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Unreadable { ptr, len } => {
                write!(f, "guest range at {ptr:#x} of length {len} is not readable")
            }
            Self::Allocation { len } => write!(f, "cannot allocate {len} bytes"),
        }
    }
}

/// Appends the guest bytes `[ptr, ptr + len)` to `out` for a host call.
/// Loads nothing and leaves `out` unchanged when the range ends past
/// `address_limit`, when any byte in it is not readable, or when `out`
/// cannot grow to hold it.
pub fn read_host_bytes(
    memory: &mut impl HostMemory,
    ptr: u64,
    len: u64,
    address_limit: u64,
    out: &mut Vec<u8>,
) -> Result<(), HostReadError> {
    let unreadable = HostReadError::Unreadable { ptr, len };
    let end = ptr
        .checked_add(len)
        .filter(|&end| end <= address_limit)
        .ok_or(unreadable)?;
    if len == 0 {
        return Ok(());
    }
    if !memory.is_host_readable(ptr, end) {
        return Err(unreadable);
    }
    usize::try_from(len)
        .ok()
        .and_then(|len| out.try_reserve(len).ok())
        .ok_or(HostReadError::Allocation { len })?;
    let initial_len = out.len();
    for address in ptr..end {
        let Some(byte) = memory.load_host_byte(address) else {
            out.truncate(initial_len);
            return Err(unreadable);
        };
        out.push(byte);
    }
    Ok(())
}

/// `String::from_utf8_lossy(&bytes).into_owned()`, but returns an error
/// instead of aborting when the string with its replacement characters
/// cannot be allocated.
pub fn lossy_host_string(bytes: Vec<u8>) -> Result<String, HostReadError> {
    let bytes = match String::from_utf8(bytes) {
        Ok(string) => return Ok(string),
        Err(error) => error.into_bytes(),
    };
    let replacement = '\u{FFFD}';
    let len: usize = bytes
        .utf8_chunks()
        .map(|chunk| {
            let replaced = !chunk.invalid().is_empty();
            chunk.valid().len() + usize::from(replaced) * replacement.len_utf8()
        })
        .sum();
    let mut string = String::new();
    string
        .try_reserve_exact(len)
        .map_err(|_| HostReadError::Allocation {
            len: u64::try_from(len).unwrap_or(u64::MAX),
        })?;
    for chunk in bytes.utf8_chunks() {
        string.push_str(chunk.valid());
        if !chunk.invalid().is_empty() {
            string.push(replacement);
        }
    }
    Ok(string)
}

/// The text a print host call appends to its message, by event type, or
/// `None` for an unknown event type.
pub fn print_terminator(event_type: u32) -> Option<&'static str> {
    match event_type {
        JOLT_PRINT_STRING => Some(""),
        JOLT_PRINT_LINE => Some("\n"),
        _ => None,
    }
}

/// Writes the bytes `print!("{}{terminator}", String::from_utf8_lossy(message))`
/// writes, but returns a write error instead of panicking on it.
pub fn write_guest_print(out: &mut impl Write, message: &[u8], terminator: &str) -> IoResult<()> {
    for chunk in message.utf8_chunks() {
        out.write_all(chunk.valid().as_bytes())?;
        if !chunk.invalid().is_empty() {
            out.write_all("\u{FFFD}".as_bytes())?;
        }
    }
    out.write_all(terminator.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_prints_match_lossy_utf8_formatting() {
        let message = b"ok \xff\xfe \xc3\xa9t\xc3\xa9 \xe2\x82";
        for terminator in ["", "\n"] {
            let mut out = Vec::new();
            write_guest_print(&mut out, message, terminator).unwrap();
            let expected = format!("{}{terminator}", String::from_utf8_lossy(message));
            assert_eq!(out, expected.into_bytes());
        }
    }

    #[test]
    fn host_strings_match_lossy_utf8_conversion() {
        for message in [
            &b"ok \xff\xfe \xc3\xa9t\xc3\xa9 \xe2\x82"[..],
            b"valid",
            b"",
        ] {
            let expected = String::from_utf8_lossy(message).into_owned();
            assert_eq!(lossy_host_string(message.to_vec()).unwrap(), expected);
        }
    }
}
