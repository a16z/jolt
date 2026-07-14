//! The `W_jolt` commitment strategy: every fact the prover and verifier must
//! agree on to commit the packed witness and settle its joint opening,
//! owned in one place. Selected at compile time by the `packed` cargo
//! feature (default: [`WJoltStrategy::Grouped`]). Consumers `match` on
//! [`W_JOLT_STRATEGY`], so both strategies stay compile-checked in every
//! build, and cargo feature unification makes a prover/verifier strategy
//! mismatch unrepresentable within one binary.

use jolt_field::Field;
use jolt_openings::{OpeningsError, PrefixPacking};

use super::super::JoltCommittedPolynomial;
use super::packing::{proof_packing, ProofPackingShape};

/// The strategy in force: `Grouped` by default, `Packed` under the `packed`
/// cargo feature.
#[cfg(feature = "packed")]
pub const W_JOLT_STRATEGY: WJoltStrategy = WJoltStrategy::Packed;
#[cfg(not(feature = "packed"))]
pub const W_JOLT_STRATEGY: WJoltStrategy = WJoltStrategy::Grouped;

/// How the per-proof packed witness (`W_jolt`) is committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WJoltStrategy {
    /// One prefix-packed sparse-unit polynomial over the canonical proof
    /// packing — every column scattered into a single committed object,
    /// opened as a singleton.
    Packed,
    /// One commitment group of strict row-major one-hot members — one
    /// `K = 2^log_k_chunk` member per committed column — reduced at a shared
    /// cell point and opened by a single native batch proof.
    Grouped,
}

/// The strategy's object layout for a proof shape.
pub enum WJoltPlan {
    /// The single packed object: its canonical multi-slot packing. Leaf
    /// claims keep their `(symbol ‖ cycle)` points; the packing's prefixes
    /// place them.
    Packed {
        packing: PrefixPacking<JoltCommittedPolynomial>,
    },
    /// The member group: the canonical column order and the shared committed
    /// arity every member polynomial has (the native batch opens them at a
    /// single point).
    Grouped {
        members: Vec<JoltCommittedPolynomial>,
        member_arity: usize,
    },
}

/// The commitment-object setup shape a strategy requires.
pub struct WJoltSetupShape {
    pub num_vars: usize,
    pub num_polys: usize,
    /// Whether the object only ever commits/opens through the backend's
    /// one-hot flavor (the full-flavor setup of the same shape is large and
    /// slow, and a grouped one-hot object never touches it).
    pub one_hot_only: bool,
}

impl WJoltStrategy {
    /// The canonical object layout for `shape`.
    pub fn plan(&self, shape: &ProofPackingShape) -> Result<WJoltPlan, OpeningsError> {
        let packing =
            proof_packing(shape).map_err(|error| OpeningsError::InvalidBatch(error.to_string()))?;
        Ok(match self {
            Self::Packed => WJoltPlan::Packed { packing },
            Self::Grouped => WJoltPlan::Grouped {
                members: packing
                    .iter()
                    .map(|(polynomial, _slot)| *polynomial)
                    .collect(),
                member_arity: shape.log_k_chunk + shape.log_t,
            },
        })
    }

    /// The commitment-object setup shape.
    pub fn setup_shape(&self, shape: &ProofPackingShape) -> Result<WJoltSetupShape, OpeningsError> {
        Ok(match self.plan(shape)? {
            WJoltPlan::Packed { packing } => WJoltSetupShape {
                num_vars: packing.packed_num_vars,
                num_polys: 1,
                one_hot_only: false,
            },
            WJoltPlan::Grouped {
                members,
                member_arity,
            } => WJoltSetupShape {
                num_vars: member_arity,
                num_polys: members.len(),
                one_hot_only: true,
            },
        })
    }

    /// Maps a column's leaf-claim point (its `(symbol ‖ cycle)` cell order,
    /// high-to-low) onto the committed variable order.
    ///
    /// [`Self::Packed`] commits the cell order directly — the leaf point is
    /// already the slot-local point the packing expects. For
    /// [`Self::Grouped`], members are row-major (`cycle ‖ lane`): `Ra` and
    /// unsigned-inc chunk columns move their `chunk_width`-variable symbol
    /// block behind the cycle block, and the msb column — committed as a
    /// `2^chunk_width`-lane member holding its bit on lanes 0/1 — gains the
    /// constant lane suffix selecting lane 1:
    /// `M(r_cycle) = member(r_cycle ‖ 0..0, 1)`.
    pub fn member_point<F: Field>(
        &self,
        polynomial: JoltCommittedPolynomial,
        chunk_width: usize,
        leaf_point: &[F],
    ) -> Result<Vec<F>, OpeningsError> {
        if let Self::Packed = self {
            return Ok(leaf_point.to_vec());
        }
        let invalid = |message: String| OpeningsError::InvalidBatch(message);
        match polynomial {
            JoltCommittedPolynomial::InstructionRa(_)
            | JoltCommittedPolynomial::BytecodeRa(_)
            | JoltCommittedPolynomial::RamRa(_)
            | JoltCommittedPolynomial::UnsignedIncChunk(_) => {
                if leaf_point.len() < chunk_width {
                    return Err(invalid(format!(
                        "{polynomial:?} leaf point has {} variables, below its \
                         {chunk_width}-variable symbol block",
                        leaf_point.len()
                    )));
                }
                let (symbol, cycle) = leaf_point.split_at(chunk_width);
                let mut point = Vec::with_capacity(leaf_point.len());
                point.extend_from_slice(cycle);
                point.extend_from_slice(symbol);
                Ok(point)
            }
            JoltCommittedPolynomial::UnsignedIncMsb => {
                let mut point = Vec::with_capacity(leaf_point.len() + chunk_width);
                point.extend_from_slice(leaf_point);
                point.extend(std::iter::repeat_n(F::zero(), chunk_width - 1));
                point.push(F::one());
                Ok(point)
            }
            other => Err(invalid(format!(
                "polynomial {other:?} is not a per-proof packed column"
            ))),
        }
    }

    /// The trivial single-slot packing a [`Self::Grouped`] member object
    /// presents to the joint opening (the member is its own whole domain —
    /// no prefix bits).
    pub fn member_packing(
        &self,
        polynomial: JoltCommittedPolynomial,
        member_arity: usize,
    ) -> Result<PrefixPacking<JoltCommittedPolynomial>, OpeningsError> {
        PrefixPacking::new([(polynomial, member_arity)])
    }
}
