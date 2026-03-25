//! Memory size unit constants and formatting helpers.

/// Bytes per gibibyte (GiB, binary, 2^30).
pub const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// Bytes per mebibyte (MiB, binary, 2^20).
pub const BYTES_PER_MIB: f64 = 1_048_576.0;

/// Formats a memory size given in GiB to a human-readable string.
///
/// Uses GiB for values >= 1.0, otherwise MiB.
pub fn format_memory_size(gib: f64) -> String {
    if gib >= 1.0 {
        format!("{gib:.2} GiB")
    } else {
        format!("{:.2} MiB", gib * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_large_value_uses_gib() {
        assert_eq!(format_memory_size(2.5), "2.50 GiB");
    }

    #[test]
    fn format_exactly_one_gib() {
        assert_eq!(format_memory_size(1.0), "1.00 GiB");
    }

    #[test]
    fn format_small_value_uses_mib() {
        assert_eq!(format_memory_size(0.5), "512.00 MiB");
    }

    #[test]
    fn format_zero() {
        assert_eq!(format_memory_size(0.0), "0.00 MiB");
    }

    #[test]
    fn format_tiny_value() {
        let result = format_memory_size(0.001);
        assert!(result.contains("MiB"));
    }

    #[test]
    fn constants_are_correct() {
        assert_eq!(BYTES_PER_GIB, (1u64 << 30) as f64);
        assert_eq!(BYTES_PER_MIB, (1u64 << 20) as f64);
    }
}
