//! The cap on the dense state a streaming kernel holds.

/// A cap on the dense state one streaming kernel holds: `bytes` bytes for
/// every `per_rows` rows of the trace before its padding. A kernel recomputes
/// its rounds from the rows until the state it would fill fits: tables whose
/// size the round fixes fill at [`Self::crossover`], and state whose size a
/// pass counts fills once [`Self::admits`] it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseCap {
    pub(crate) bytes: usize,
    pub(crate) per_rows: usize,
}

/// The cap every streaming kernel runs under: 4 bytes per trace row, a
/// sixteenth of a resident row. Being a density, it has a kernel fill at the
/// same round at every trace length, and each round before that costs one
/// pass over the rows before the padding. It bounds one kernel, not a stage:
/// an in-place bind keeps a table's storage for three rounds, so a stage
/// holds together the fills of its kernels that fill within three rounds of
/// each other.
pub(crate) const DENSE_CAP: DenseCap = DenseCap {
    bytes: 4,
    per_rows: 1,
};

impl DenseCap {
    /// Whether `bytes` bytes of state over a trace of `rows` rows before its
    /// padding fit the cap.
    pub(crate) fn admits(self, bytes: usize, rows: usize) -> bool {
        // Neither product overflows: each factor fits `usize`.
        bytes as u128 * self.per_rows as u128 <= self.bytes as u128 * rows as u128
    }

    /// The first bind depth, of `rounds`, at which tables that take
    /// `cycle_bytes` bytes per cycle before any bind fit the cap, or `rounds`
    /// when none does: each bind halves them, so after `depth` binds they
    /// take `cycle_bytes` bytes per `2^depth` cycles. Rounding a table up to
    /// whole entries is not counted.
    pub(crate) fn crossover(self, cycle_bytes: usize, rounds: usize) -> usize {
        let held = cycle_bytes as u128 * self.per_rows as u128;
        let mut admitted = self.bytes as u128;
        for depth in 0..rounds {
            if held <= admitted {
                return depth;
            }
            admitted = admitted.saturating_mul(2);
        }
        rounds
    }
}

#[cfg(test)]
mod tests {
    use super::DenseCap;

    /// The crossover is exact at the cap's boundary, agrees with
    /// [`DenseCap::admits`] over `2^depth` rows, is the round count when no
    /// depth fits, and does not wrap for sizes past `usize` bytes.
    #[test]
    fn crossover_is_exact_at_the_cap_and_past_usize_bytes() {
        let cap = DenseCap {
            bytes: 48,
            per_rows: 3,
        };
        assert_eq!(cap.crossover(16, 20), 0);
        assert_eq!(cap.crossover(17, 20), 1);
        assert_eq!(cap.crossover(16 << 10, 20), 10);
        assert_eq!(cap.crossover((16 << 10) + 1, 20), 11);
        assert_eq!(cap.crossover((16 << 10) + 1, 11), 11);
        assert_eq!(cap.crossover((16 << 10) + 1, 4), 4);
        assert_eq!(cap.crossover(0, 0), 0);
        assert!(cap.admits(16 << 10, 1 << 10));
        assert!(!cap.admits((16 << 10) + 1, 1 << 10));
        assert!(cap.admits(0, 0));
        assert!(!cap.admits(1, 0));

        let none = DenseCap {
            bytes: 0,
            per_rows: 1,
        };
        assert_eq!(none.crossover(1, 20), 20);
        assert_eq!(none.crossover(0, 20), 0);

        let wide = DenseCap {
            bytes: usize::MAX,
            per_rows: usize::MAX,
        };
        assert_eq!(wide.crossover(usize::MAX, 200), usize::BITS as usize);
        assert!(wide.admits(usize::MAX, usize::MAX));
        assert!(!wide.admits(usize::MAX, usize::MAX - 1));
    }
}
