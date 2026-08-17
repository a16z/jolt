//! Sparse cycle sequence for RAM Hamming-weight booleanity.
//!
//! Unlike the other members this one collapses an access-only topology (no
//! increment leaves) and binds cycle variables in stage-1 order, indexing the
//! cycle binding forward rather than reversed.

use std::sync::Arc;

use jolt_field::Field;

use super::frontier::{
    build_parent_weights, weight_level_bytes, FrontierDriver, RamCycleError, RamCycleMember,
};
use super::owner::RamCycleFamilyOwner;
use super::topology::RamBlockTopology;

const MEMBER: RamCycleMember = RamCycleMember::HammingBooleanity;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamHammingMessage<F> {
    coefficients: [F; 4],
}

impl<F> RamHammingMessage<F> {
    pub const fn coefficients(&self) -> &[F; 4] {
        &self.coefficients
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamHammingTerminal<F> {
    ram_hamming_weight: F,
    eq_cycle: F,
}

pub struct RamHammingSparsePlan {
    topology: RamBlockTopology,
    source_generation: u64,
    source_fingerprint: u64,
    log_t: usize,
    access_leaves: usize,
    parent_nodes: usize,
    middle_nodes: usize,
    estimated_products: u128,
    topology_bytes: usize,
}

impl RamHammingSparsePlan {
    pub fn new(owner: &RamCycleFamilyOwner) -> Result<Self, RamCycleError> {
        let receipt = owner.receipt();
        let topology = RamBlockTopology::build(receipt.log_t(), owner.access_records(), &[])?;
        let access_leaves = topology.leaf_cycles().len();
        let overflow = || RamCycleError::Overflow { member: MEMBER };
        let parent_nodes = topology
            .census()
            .iter()
            .skip(1)
            .try_fold(0usize, |sum, level| {
                sum.checked_add(usize::try_from(level.entries()).map_err(|_| overflow())?)
                    .ok_or_else(overflow)
            })?;
        let middle_nodes = topology
            .census()
            .iter()
            .skip(1)
            .take(receipt.log_t().saturating_sub(1))
            .try_fold(0usize, |sum, level| {
                sum.checked_add(usize::try_from(level.entries()).map_err(|_| overflow())?)
                    .ok_or_else(overflow)
            })?;
        let rounds = receipt.log_t() as u128;
        let estimated_products = (parent_nodes as u128)
            .checked_mul(7)
            .and_then(|value| value.checked_add(middle_nodes as u128))
            .and_then(|value| value.checked_add(rounds.checked_mul(10)?))
            .ok_or_else(overflow)?;
        let topology_bytes = topology.owned_heap_bytes();
        Ok(Self {
            topology,
            source_generation: receipt.source_generation(),
            source_fingerprint: receipt.fingerprint(),
            log_t: receipt.log_t(),
            access_leaves,
            parent_nodes,
            middle_nodes,
            estimated_products,
            topology_bytes,
        })
    }

    copy_field_getters! { pub, {
        source_generation: u64,
        source_fingerprint: u64,
        log_t: usize,
        access_leaves: usize,
        parent_nodes: usize,
        middle_nodes: usize,
        estimated_products: u128,
        topology_bytes: usize,
    }}
}

impl<F: Copy> RamHammingTerminal<F> {
    copy_field_getters! { pub, {
        ram_hamming_weight: F,
        eq_cycle: F,
    }}
}

pub struct HostSparseRamHammingBooleanity<F> {
    _owner: Arc<RamCycleFamilyOwner>,
    topology: RamBlockTopology,
    core: FrontierDriver<F>,
    parent_weights: Vec<Vec<F>>,
    cycle_binding: Vec<F>,
    eq_scale: F,
}

impl<F: Field> HostSparseRamHammingBooleanity<F> {
    pub fn new(
        owner: Arc<RamCycleFamilyOwner>,
        stage1_cycle_binding: &[F],
    ) -> Result<Self, RamCycleError> {
        let plan = RamHammingSparsePlan::new(&owner)?;
        Self::new_from_plan(owner, stage1_cycle_binding, plan)
    }

    pub(crate) fn new_from_plan(
        owner: Arc<RamCycleFamilyOwner>,
        stage1_cycle_binding: &[F],
        plan: RamHammingSparsePlan,
    ) -> Result<Self, RamCycleError> {
        let receipt = owner.receipt();
        if stage1_cycle_binding.len() != receipt.log_t() {
            return Err(RamCycleError::CyclePointLength {
                member: MEMBER,
                expected: receipt.log_t(),
                got: stage1_cycle_binding.len(),
            });
        }
        if plan.source_generation != receipt.source_generation()
            || plan.source_fingerprint != receipt.fingerprint()
            || plan.log_t != receipt.log_t()
            || plan.access_leaves != owner.access_records().len()
        {
            return Err(RamCycleError::PlanReceiptMismatch { member: MEMBER });
        }
        let topology = plan.topology;
        let blocks = topology
            .leaf_cycles()
            .iter()
            .map(|&cycle| u64::from(cycle))
            .collect::<Vec<_>>();
        let values = vec![F::one(); blocks.len()];
        let rounds = receipt.log_t();
        let parent_weights = build_parent_weights(
            MEMBER,
            &topology,
            F::zero(),
            F::one(),
            |&parent, high, round| {
                let coordinate = stage1_cycle_binding[round];
                parent
                    * if high {
                        coordinate
                    } else {
                        F::one() - coordinate
                    }
            },
        )?;
        let core = FrontierDriver::new(MEMBER, rounds, blocks, vec![values]);
        core.validate_frontier(topology.census())?;
        Ok(Self {
            _owner: owner,
            topology,
            core,
            parent_weights,
            cycle_binding: stage1_cycle_binding.to_vec(),
            eq_scale: F::one(),
        })
    }

    pub fn owned_heap_bytes(&self) -> usize {
        self.topology.owned_heap_bytes()
            + self.core.owned_heap_bytes()
            + weight_level_bytes(&self.parent_weights)
            + self.cycle_binding.capacity() * std::mem::size_of::<F>()
    }

    pub const fn num_rounds(&self) -> usize {
        self.core.num_rounds()
    }

    pub const fn round(&self) -> usize {
        self.core.round()
    }

    pub fn message(&mut self) -> Result<RamHammingMessage<F>, RamCycleError> {
        self.core.ensure_active()?;
        let mut q = [F::zero(); 3];
        self.core.prepare_round(
            &self.topology,
            &self.parent_weights,
            |weight, lows, _highs, slopes| {
                let low = lows[0];
                let slope = slopes[0];
                let q_0 = low * low - low;
                let q_1 = slope * (low + low - F::one());
                let q_2 = slope * slope;
                q[0] += weight * q_0;
                q[1] += weight * q_1;
                q[2] += weight * q_2;
            },
        )?;

        let coordinate = self.cycle_binding[self.core.round()];
        let l_0 = self.eq_scale * (F::one() - coordinate);
        let l_1 = self.eq_scale * (coordinate + coordinate - F::one());
        let coefficients = [
            l_0 * q[0],
            l_0 * q[1] + l_1 * q[0],
            l_0 * q[2] + l_1 * q[1],
            l_1 * q[2],
        ];
        Ok(RamHammingMessage { coefficients })
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), RamCycleError> {
        let round = self.core.round();
        let cycle_binding = &self.cycle_binding;
        let eq_scale = &mut self.eq_scale;
        self.core.bind_cached(&self.topology, challenge, || {
            let coordinate = cycle_binding[round];
            *eq_scale *= (F::one() - coordinate) + challenge * (coordinate + coordinate - F::one());
        })
    }

    pub fn terminal(&self) -> Result<RamHammingTerminal<F>, RamCycleError> {
        let lanes = self.core.terminal_values(self.topology.census())?;
        let ram_hamming_weight = lanes.map_or(F::zero(), |lanes| lanes[0][0]);
        Ok(RamHammingTerminal {
            ram_hamming_weight,
            eq_cycle: self.eq_scale,
        })
    }
}

pub fn estimated_ram_hamming_products(owner: &RamCycleFamilyOwner) -> Result<u128, RamCycleError> {
    Ok(RamHammingSparsePlan::new(owner)?.estimated_products())
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::AkitaField;

    use super::super::owner::{OwnerConfig, RamAccessRecord, RamIncrementRecord};
    use super::*;

    fn field(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    fn fixture_owner() -> RamCycleFamilyOwner {
        let config = OwnerConfig::new(3, 3, 47, 16).unwrap();
        let records = vec![
            RamAccessRecord::new(0, 1, 0, 2),
            RamAccessRecord::new(3, 5, 0, 3),
            RamAccessRecord::new(4, 1, 2, 2),
            RamAccessRecord::new(7, 5, 3, 1),
        ];
        let increments = vec![
            RamIncrementRecord::new(0, 2),
            RamIncrementRecord::new(2, -3),
            RamIncrementRecord::new(3, 3),
            RamIncrementRecord::new(6, 4),
            RamIncrementRecord::new(7, -2),
        ];
        RamCycleFamilyOwner::from_sparse_records(
            config,
            records,
            increments,
            vec![0, 2, 0, 0, 0, 1, 0, 0],
        )
        .unwrap()
    }

    fn dense_eq(binding: &[AkitaField]) -> Vec<AkitaField> {
        (0..1usize << binding.len())
            .map(|index| {
                binding
                    .iter()
                    .enumerate()
                    .fold(AkitaField::one(), |value, (bit, coordinate)| {
                        value
                            * if index & (1 << bit) == 0 {
                                AkitaField::one() - *coordinate
                            } else {
                                *coordinate
                            }
                    })
            })
            .collect()
    }

    fn dense_message(hamming: &[AkitaField], eq: &[AkitaField]) -> [AkitaField; 4] {
        let mut coefficients = [AkitaField::zero(); 4];
        for pair in 0..hamming.len() / 2 {
            let low = 2 * pair;
            let high = low + 1;
            let h_0 = hamming[low];
            let h_slope = hamming[high] - h_0;
            let q = [
                h_0 * h_0 - h_0,
                h_slope * (h_0 + h_0 - AkitaField::one()),
                h_slope * h_slope,
            ];
            let l = [eq[low], eq[high] - eq[low]];
            coefficients[0] += l[0] * q[0];
            coefficients[1] += l[0] * q[1] + l[1] * q[0];
            coefficients[2] += l[0] * q[2] + l[1] * q[1];
            coefficients[3] += l[1] * q[2];
        }
        coefficients
    }

    fn bind_dense(values: &mut Vec<AkitaField>, challenge: AkitaField) {
        let bound = values.len() / 2;
        for index in 0..bound {
            let low = values[2 * index];
            let high = values[2 * index + 1];
            values[index] = low + challenge * (high - low);
        }
        values.truncate(bound);
    }

    #[test]
    fn sparse_sequence_matches_independent_dense_relation() {
        let owner = Arc::new(fixture_owner());
        let binding = [field(7), field(11), field(13)];
        let mut sparse = HostSparseRamHammingBooleanity::new(Arc::clone(&owner), &binding).unwrap();
        let mut hamming = vec![AkitaField::zero(); owner.receipt().cycles()];
        for record in owner.access_records() {
            hamming[record.cycle() as usize] = AkitaField::one();
        }
        let mut eq = dense_eq(&binding);

        assert_eq!(sparse.core.frontier_len(), owner.access_records().len());
        assert!(estimated_ram_hamming_products(&owner).unwrap() < 1_000);
        for challenge in [field(17), field(19), field(23)] {
            assert_eq!(
                sparse.message().unwrap().coefficients(),
                &dense_message(&hamming, &eq)
            );
            sparse.bind(challenge).unwrap();
            bind_dense(&mut hamming, challenge);
            bind_dense(&mut eq, challenge);
        }
        let terminal = sparse.terminal().unwrap();
        assert_eq!(terminal.ram_hamming_weight(), hamming[0]);
        assert_eq!(terminal.eq_cycle(), eq[0]);
    }

    #[test]
    fn sparse_plan_freezes_one_topology_and_owner_receipt() {
        let owner = Arc::new(fixture_owner());
        let plan = RamHammingSparsePlan::new(&owner).unwrap();
        assert_eq!(plan.source_generation(), 47);
        assert_eq!(plan.source_fingerprint(), owner.receipt().fingerprint());
        assert_eq!(plan.log_t(), 3);
        assert_eq!(plan.access_leaves(), 4);
        assert_eq!(plan.parent_nodes(), 7);
        assert_eq!(plan.middle_nodes(), 6);
        assert_eq!(plan.estimated_products(), 85);
        assert_eq!(plan.topology_bytes(), 128);
        let topology_bytes = plan.topology_bytes();

        let sequence = HostSparseRamHammingBooleanity::new_from_plan(
            owner,
            &[field(7), field(11), field(13)],
            plan,
        )
        .unwrap();
        assert_eq!(sequence.num_rounds(), 3);
        assert!(sequence.owned_heap_bytes() > topology_bytes);
    }

    #[test]
    fn empty_support_remains_zero_without_dense_storage() {
        let config = OwnerConfig::new(3, 2, 53, 8).unwrap();
        let owner = Arc::new(
            RamCycleFamilyOwner::from_sparse_records(config, Vec::new(), Vec::new(), vec![0; 4])
                .unwrap(),
        );
        let mut sparse =
            HostSparseRamHammingBooleanity::new(owner, &[field(3), field(5), field(7)]).unwrap();
        for challenge in [field(11), field(13), field(17)] {
            assert_eq!(
                sparse.message().unwrap().coefficients(),
                &[AkitaField::zero(); 4]
            );
            sparse.bind(challenge).unwrap();
        }
        assert_eq!(
            sparse.terminal().unwrap().ram_hamming_weight(),
            AkitaField::zero()
        );
    }

    #[test]
    fn bind_requires_a_prepared_message() {
        let owner = Arc::new(fixture_owner());
        let mut sparse =
            HostSparseRamHammingBooleanity::new(owner, &[field(3), field(5), field(7)]).unwrap();
        assert_eq!(
            sparse.bind(field(11)),
            Err(RamCycleError::MessageNotPrepared {
                member: RamCycleMember::HammingBooleanity,
                round: 0
            })
        );
    }
}
