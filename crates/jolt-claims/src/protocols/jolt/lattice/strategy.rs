//! The canonical native-Akita `W_jolt` commitment layout. Every committed
//! member is a strict `K x T` one-hot polynomial, in the order returned by
//! [`wjolt_members`](super::packing::wjolt_members), and all members open at
//! one common `(cycle || address)` point.

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::Field;
use jolt_openings::OpeningsError;

use super::super::JoltCommittedPolynomial;
use super::packing::{wjolt_members, WJoltShape};

/// Wjolt is always committed as one native Akita group of strict one-hot
/// members.
pub const W_JOLT_LAYOUT: WJoltLayout = WJoltLayout;

/// The one protocol layout for the per-proof `W_jolt` commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WJoltLayout;

/// The canonical member order and uniform arity for one proof shape.
pub struct WJoltLayoutPlan {
    pub members: Vec<JoltCommittedPolynomial>,
    pub member_arity: usize,
}

/// The commitment-object setup shape the layout requires.
pub struct WJoltSetupShape {
    pub num_vars: usize,
    pub num_polys: usize,
}

impl WJoltLayout {
    /// The canonical object layout for `shape`.
    pub fn plan(&self, shape: &WJoltShape) -> Result<WJoltLayoutPlan, OpeningsError> {
        let members =
            wjolt_members(shape).map_err(|error| OpeningsError::InvalidBatch(error.to_string()))?;
        Ok(WJoltLayoutPlan {
            members,
            member_arity: shape.log_k_chunk + shape.log_t,
        })
    }

    /// The commitment-object setup shape.
    pub fn setup_shape(&self, shape: &WJoltShape) -> Result<WJoltSetupShape, OpeningsError> {
        let plan = self.plan(shape)?;
        Ok(WJoltSetupShape {
            num_vars: plan.member_arity,
            num_polys: plan.members.len(),
        })
    }

    /// A protocol-owned digest of the exact native batch layout. This binds
    /// the commitment and verifier setup to the ordered member identities,
    /// dimensions, and layout version; it is never supplied by the proof.
    pub fn layout_digest(&self, shape: &WJoltShape) -> Result<[u8; 32], OpeningsError> {
        let WJoltLayoutPlan {
            members,
            member_arity,
        } = self.plan(shape)?;
        let mut hasher = Blake2b::<U32>::new();
        hasher.update(b"jolt/akita/w_jolt/native-one-hot/v1");
        append_usize(&mut hasher, member_arity);
        append_usize(&mut hasher, members.len());
        append_usize(&mut hasher, shape.log_t);
        append_usize(&mut hasher, shape.log_k_chunk);
        append_usize(&mut hasher, shape.ra_layout.instruction());
        append_usize(&mut hasher, shape.ra_layout.bytecode());
        append_usize(&mut hasher, shape.ra_layout.ram());
        for member in members {
            match member {
                JoltCommittedPolynomial::InstructionRa(index) => {
                    hasher.update([0]);
                    append_usize(&mut hasher, index);
                }
                JoltCommittedPolynomial::BytecodeRa(index) => {
                    hasher.update([1]);
                    append_usize(&mut hasher, index);
                }
                JoltCommittedPolynomial::RamRa(index) => {
                    hasher.update([2]);
                    append_usize(&mut hasher, index);
                }
                JoltCommittedPolynomial::UnsignedIncChunk(index) => {
                    hasher.update([3]);
                    append_usize(&mut hasher, index);
                }
                JoltCommittedPolynomial::UnsignedIncMsb => hasher.update([4]),
                other => {
                    return Err(OpeningsError::InvalidBatch(format!(
                        "non-Wjolt polynomial {other:?} in native one-hot layout"
                    )));
                }
            }
        }
        Ok(hasher.finalize().into())
    }

    /// Maps a column's leaf-claim point (its `(address || cycle)` cell order,
    /// high-to-low) onto the committed variable order.
    ///
    /// Members are row-major (`cycle || address`), while relation leaves use
    /// (`address || cycle`), so this moves the address block behind the cycle
    /// block. The MSB is treated identically: it is a full `K x T` one-hot
    /// member whose hot address is either zero or one.
    pub fn member_point<F: Field>(
        &self,
        polynomial: JoltCommittedPolynomial,
        chunk_width: usize,
        leaf_point: &[F],
    ) -> Result<Vec<F>, OpeningsError> {
        let invalid = |message: String| OpeningsError::InvalidBatch(message);
        match polynomial {
            JoltCommittedPolynomial::InstructionRa(_)
            | JoltCommittedPolynomial::BytecodeRa(_)
            | JoltCommittedPolynomial::RamRa(_)
            | JoltCommittedPolynomial::UnsignedIncChunk(_)
            | JoltCommittedPolynomial::UnsignedIncMsb => {
                if leaf_point.len() < chunk_width {
                    return Err(invalid(format!(
                        "{polynomial:?} leaf point has {} variables, below its \
                         {chunk_width}-variable address block",
                        leaf_point.len()
                    )));
                }
                let (address, cycle) = leaf_point.split_at(chunk_width);
                let mut point = Vec::with_capacity(leaf_point.len());
                point.extend_from_slice(cycle);
                point.extend_from_slice(address);
                Ok(point)
            }
            other => Err(invalid(format!(
                "polynomial {other:?} is not a per-proof packed column"
            ))),
        }
    }
}

fn append_usize(hasher: &mut Blake2b<U32>, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn shape(log_t: usize) -> WJoltShape {
        WJoltShape {
            ra_layout: JoltRaPolynomialLayout::new(2, 1, 1).unwrap(),
            log_t,
            log_k_chunk: 8,
        }
    }

    #[test]
    fn native_layout_is_uniform_and_digest_bound() {
        let WJoltLayoutPlan {
            members,
            member_arity,
        } = W_JOLT_LAYOUT.plan(&shape(5)).unwrap();
        assert_eq!(member_arity, 13);
        assert_eq!(
            members.last(),
            Some(&JoltCommittedPolynomial::UnsignedIncMsb)
        );

        let digest = W_JOLT_LAYOUT.layout_digest(&shape(5)).unwrap();
        assert_ne!(digest, [0; 32]);
        assert_ne!(digest, W_JOLT_LAYOUT.layout_digest(&shape(6)).unwrap());
    }

    #[test]
    fn every_member_uses_the_same_point_permutation() {
        let leaf = (0..5).map(Fr::from_u64).collect::<Vec<_>>();
        let expected = [2, 3, 4, 0, 1]
            .into_iter()
            .map(Fr::from_u64)
            .collect::<Vec<_>>();
        for polynomial in [
            JoltCommittedPolynomial::InstructionRa(0),
            JoltCommittedPolynomial::BytecodeRa(0),
            JoltCommittedPolynomial::RamRa(0),
            JoltCommittedPolynomial::UnsignedIncChunk(0),
            JoltCommittedPolynomial::UnsignedIncMsb,
        ] {
            assert_eq!(
                W_JOLT_LAYOUT.member_point(polynomial, 2, &leaf).unwrap(),
                expected
            );
        }
    }
}
