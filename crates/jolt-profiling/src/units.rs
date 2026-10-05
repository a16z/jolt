/// Bytes per gibibyte (GiB, binary, 2^30).
pub const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// Bytes per mebibyte (MiB, binary, 2^20).
pub const BYTES_PER_MIB: f64 = 1_048_576.0;

/// Formats a memory size given in GiB to a human-readable string.
///
/// Uses GiB for values >= 1.0, otherwise MiB.
pub fn format_memory_size(gib: f64) -> String {
    if gib.abs() >= 1.0 {
        format!("{gib:.2} GiB")
    } else {
        format!("{:.2} MiB", gib * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_exactly_one_gib() {
        assert_eq!(format_memory_size(1.0), "1.00 GiB");
    }

    #[test]
    fn format_small_value_uses_mib() {
        assert_eq!(format_memory_size(0.5), "512.00 MiB");
    }
}
