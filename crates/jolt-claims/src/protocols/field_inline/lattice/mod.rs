//! Direct Akita commitment layout for the field-register increment polynomial.

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::Field;
use jolt_openings::OpeningsError;
#[cfg(feature = "akita")]
use jolt_openings::PrecommittedRole;

use crate::lattice::MIN_DENSE_OBJECT_NUM_VARS;

/// The trace polynomial occupies the first slice of a dense commitment padded
/// to Akita's minimum supported arity. Opening the added coordinates at zero
/// selects that slice; no protocol relation consumes the remaining entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldIncLayout {
    log_t: usize,
}

impl FieldIncLayout {
    pub const fn new(log_t: usize) -> Self {
        Self { log_t }
    }

    pub fn num_vars(self) -> usize {
        self.log_t.max(MIN_DENSE_OBJECT_NUM_VARS)
    }

    pub fn layout_digest(self) -> [u8; 32] {
        let mut hasher = Blake2b::<U32>::new();
        hasher.update(b"jolt/field-inline/akita/inc/v1");
        hasher.update((self.log_t as u64).to_le_bytes());
        hasher.update((self.num_vars() as u64).to_le_bytes());
        hasher.finalize().into()
    }

    pub fn opening_point<F: Field>(self, cycle_point: &[F]) -> Result<Vec<F>, OpeningsError> {
        if cycle_point.len() != self.log_t {
            return Err(OpeningsError::InvalidBatch(format!(
                "field increment point has {} variables, expected {}",
                cycle_point.len(),
                self.log_t,
            )));
        }
        let mut point = vec![F::zero(); self.num_vars() - self.log_t];
        point.extend_from_slice(cycle_point);
        Ok(point)
    }
}

/// The field increment follows advice roles 0 and 1 in the joint opening.
#[cfg(feature = "akita")]
pub const fn field_inc_precommitted_role() -> PrecommittedRole {
    PrecommittedRole::new(2, b"field_inc", "field-inc")
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use jolt_field::{Fr, Ring, Zero};
    use jolt_poly::Polynomial;

    use super::*;

    #[test]
    fn padded_opening_preserves_the_trace_polynomial() {
        for log_t in [3, MIN_DENSE_OBJECT_NUM_VARS, MIN_DENSE_OBJECT_NUM_VARS + 1] {
            let layout = FieldIncLayout::new(log_t);
            let values: Vec<Fr> = (0..1u64 << log_t).map(Fr::from_u64).collect();
            let point: Vec<Fr> = (0..log_t).map(|i| Fr::from_u64(i as u64 + 3)).collect();
            let expected = Polynomial::new(values.clone()).evaluate(&point);
            let mut padded = values;
            padded.resize(1 << layout.num_vars(), Fr::from_u64(9));
            assert_eq!(
                Polynomial::new(padded).evaluate(&layout.opening_point(&point).unwrap()),
                expected,
            );
            assert!(layout.opening_point(&vec![Fr::zero(); log_t + 1]).is_err());
        }
    }

    #[test]
    fn layout_digest_binds_the_trace_arity() {
        let small = FieldIncLayout::new(3);
        let larger = FieldIncLayout::new(4);
        assert_eq!(small.num_vars(), larger.num_vars());
        assert_ne!(small.layout_digest(), larger.layout_digest());
    }
}
