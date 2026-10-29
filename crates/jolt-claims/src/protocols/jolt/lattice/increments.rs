//! Direct Akita commitment layout for the RV64 fused increment polynomial.

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::Field;
#[cfg(feature = "akita")]
use jolt_openings::CommitmentGroupRole;
use jolt_openings::OpeningsError;

use crate::lattice::MIN_DENSE_OBJECT_NUM_VARS;

/// The trace polynomial occupies the first slice of a dense commitment padded
/// to Akita's minimum supported arity. Opening the added coordinates at zero
/// selects that slice; no protocol relation consumes the remaining entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedIncLayout {
    log_t: usize,
}

impl FusedIncLayout {
    pub const fn new(log_t: usize) -> Self {
        Self { log_t }
    }

    pub fn num_vars(self) -> usize {
        self.log_t.max(MIN_DENSE_OBJECT_NUM_VARS)
    }

    pub fn layout_digest(self) -> [u8; 32] {
        let mut hasher = Blake2b::<U32>::new();
        hasher.update(b"jolt/akita/fused-inc/v1");
        hasher.update((self.log_t as u64).to_le_bytes());
        hasher.update((self.num_vars() as u64).to_le_bytes());
        hasher.finalize().into()
    }

    pub fn opening_point<F: Field>(self, cycle_point: &[F]) -> Result<Vec<F>, OpeningsError> {
        if cycle_point.len() != self.log_t {
            return Err(OpeningsError::InvalidBatch(format!(
                "fused increment point has {} variables, expected {}",
                cycle_point.len(),
                self.log_t,
            )));
        }
        let mut point = vec![F::zero(); self.num_vars() - self.log_t];
        point.extend_from_slice(cycle_point);
        Ok(point)
    }
}

/// Orders the bounded fused increment after advice and direct program objects.
#[cfg(feature = "akita")]
pub const fn fused_inc_group_role() -> CommitmentGroupRole {
    CommitmentGroupRole::new(4, b"fused_inc", "fused-inc")
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};
    use jolt_poly::Polynomial;

    #[test]
    fn opening_selects_the_trace_and_binds_its_logical_arity() {
        for log_t in [3, MIN_DENSE_OBJECT_NUM_VARS, MIN_DENSE_OBJECT_NUM_VARS + 1] {
            let layout = FusedIncLayout::new(log_t);
            let point = (0..log_t)
                .map(|i| Fr::from_u64(i as u64 + 2))
                .collect::<Vec<_>>();
            let values = (0..1usize << log_t)
                .map(|i| Fr::from_i128(if i % 2 == 0 { i as i128 } else { -(i as i128) }))
                .collect::<Vec<_>>();
            let expected = Polynomial::new(values.clone()).evaluate(&point);
            let mut padded = values;
            padded.resize(1 << layout.num_vars(), Fr::from_u64(97));
            assert_eq!(
                Polynomial::new(padded).evaluate(&layout.opening_point(&point).unwrap()),
                expected,
            );
            assert!(layout.opening_point(&point[1..]).is_err());
        }
        assert_ne!(
            FusedIncLayout::new(3).layout_digest(),
            FusedIncLayout::new(4).layout_digest()
        );
    }
}
