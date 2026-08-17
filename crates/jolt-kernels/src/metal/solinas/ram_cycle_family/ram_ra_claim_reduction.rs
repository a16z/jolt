use std::sync::Arc;

use jolt_field::Field;

use super::frontier::{
    build_parent_weights, eq_at_boolean_index, for_each_leaf_access, weight_level_bytes,
    FrontierDriver, RamCycleError, RamCycleMember,
};
use super::owner::RamCycleFamilyOwner;

const TERMS: usize = 3;
const MEMBER: RamCycleMember = RamCycleMember::RaClaimReduction;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimMessage<F> {
    at_zero: F,
    at_two: F,
}

impl<F: Field> RamRaClaimMessage<F> {
    pub const fn sampled_evaluations(self) -> [F; 2] {
        [self.at_zero, self.at_two]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimTerminal<F> {
    ram_ra: F,
    eq_cycles: [F; TERMS],
}

impl<F: Field> RamRaClaimTerminal<F> {
    copy_field_getters! { pub, {
        ram_ra: F,
        eq_cycles: [F; TERMS],
    }}
}

pub struct HostSparseRamRaClaimReduction<F> {
    owner: Arc<RamCycleFamilyOwner>,
    core: FrontierDriver<F>,
    parent_weights: Vec<Vec<[F; TERMS]>>,
    cycle_points: [Vec<F>; TERMS],
    scales: [F; TERMS],
}

impl<F: Field> HostSparseRamRaClaimReduction<F> {
    pub fn new(
        owner: Arc<RamCycleFamilyOwner>,
        r_address: &[F],
        cycle_points: [&[F]; TERMS],
        gamma: F,
    ) -> Result<Self, RamCycleError> {
        let receipt = owner.receipt();
        if r_address.len() != receipt.log_k() {
            return Err(RamCycleError::AddressPointLength {
                member: MEMBER,
                expected: receipt.log_k(),
                got: r_address.len(),
            });
        }
        for point in cycle_points {
            if point.len() != receipt.log_t() {
                return Err(RamCycleError::CyclePointLength {
                    member: MEMBER,
                    expected: receipt.log_t(),
                    got: point.len(),
                });
            }
        }

        let (blocks, values) = seed_frontier(&owner, r_address)?;
        let rounds = receipt.log_t();
        let cycle_points = cycle_points.map(<[F]>::to_vec);
        let gamma_powers = [F::one(), gamma, gamma * gamma];
        let parent_weights = build_parent_weights(
            MEMBER,
            owner.block_topology(),
            [F::zero(); TERMS],
            gamma_powers,
            |parent, high, round| {
                let mut weight = [F::zero(); TERMS];
                for term in 0..TERMS {
                    let coordinate = cycle_points[term][rounds - 1 - round];
                    weight[term] = parent[term]
                        * if high {
                            coordinate
                        } else {
                            F::one() - coordinate
                        };
                }
                weight
            },
        )?;
        let core = FrontierDriver::new(MEMBER, rounds, blocks, vec![values]);
        core.validate_frontier(owner.block_topology().census())?;
        Ok(Self {
            owner,
            core,
            parent_weights,
            cycle_points,
            scales: [F::one(); TERMS],
        })
    }

    pub fn owned_heap_bytes(&self) -> usize {
        self.core.owned_heap_bytes()
            + weight_level_bytes(&self.parent_weights)
            + self
                .cycle_points
                .iter()
                .map(|point| point.capacity() * std::mem::size_of::<F>())
                .sum::<usize>()
    }

    pub const fn num_rounds(&self) -> usize {
        self.core.num_rounds()
    }

    pub const fn round(&self) -> usize {
        self.core.round()
    }

    pub fn message(&mut self) -> Result<RamRaClaimMessage<F>, RamCycleError> {
        self.core.ensure_active()?;
        let round = self.core.round();
        let rounds = self.core.num_rounds();
        let mut c_zero = [F::zero(); TERMS];
        let mut c_two = [F::zero(); TERMS];
        for term in 0..TERMS {
            let coordinate = self.cycle_points[term][rounds - 1 - round];
            c_zero[term] = self.scales[term] * (F::one() - coordinate);
            c_two[term] = self.scales[term] * (coordinate + coordinate + coordinate - F::one());
        }

        let mut at_zero = F::zero();
        let mut at_two = F::zero();
        self.core.prepare_round(
            self.owner.block_topology(),
            &self.parent_weights,
            |weight, lows, highs, slopes| {
                let h_zero = lows[0];
                let h_one = highs[0];
                let slope = slopes[0];
                let mut g_zero = F::zero();
                let mut g_two = F::zero();
                for term in 0..TERMS {
                    g_zero += weight[term] * c_zero[term];
                    g_two += weight[term] * c_two[term];
                }
                at_zero += h_zero * g_zero;
                at_two += (h_one + slope) * g_two;
            },
        )?;
        Ok(RamRaClaimMessage { at_zero, at_two })
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), RamCycleError> {
        let round = self.core.round();
        let rounds = self.core.num_rounds();
        let cycle_points = &self.cycle_points;
        let scales = &mut self.scales;
        self.core
            .bind_cached(self.owner.block_topology(), challenge, || {
                for term in 0..TERMS {
                    let coordinate = cycle_points[term][rounds - 1 - round];
                    scales[term] *=
                        (F::one() - coordinate) + challenge * (coordinate + coordinate - F::one());
                }
            })
    }

    pub fn terminal(&self) -> Result<RamRaClaimTerminal<F>, RamCycleError> {
        let lanes = self
            .core
            .terminal_values(self.owner.block_topology().census())?;
        let ram_ra = lanes.map_or(F::zero(), |lanes| lanes[0][0]);
        Ok(RamRaClaimTerminal {
            ram_ra,
            eq_cycles: self.scales,
        })
    }
}

pub fn estimated_ram_ra_claim_products(owner: &RamCycleFamilyOwner) -> Result<u128, RamCycleError> {
    let overflow = || RamCycleError::Overflow { member: MEMBER };
    let receipt = owner.receipt();
    let census = owner.block_topology().census();
    let parent_nodes = census
        .iter()
        .skip(1)
        .try_fold(0u128, |sum, level| {
            sum.checked_add(u128::from(level.entries()))
        })
        .ok_or_else(overflow)?;
    let middle_nodes = census
        .iter()
        .skip(1)
        .take(receipt.log_t().saturating_sub(1))
        .try_fold(0u128, |sum, level| {
            sum.checked_add(u128::from(level.entries()))
        })
        .ok_or_else(overflow)?;
    let address = u128::try_from(receipt.access_count())
        .map_err(|_| overflow())?
        .checked_mul(u128::try_from(receipt.log_k()).map_err(|_| overflow())?)
        .ok_or_else(overflow)?;
    address
        .checked_add(1)
        .and_then(|value| value.checked_add(3 * middle_nodes))
        .and_then(|value| value.checked_add(9 * parent_nodes))
        .and_then(|value| value.checked_add(12 * u128::try_from(receipt.log_t()).ok()?))
        .ok_or_else(overflow)
}

fn seed_frontier<F: Field>(
    owner: &RamCycleFamilyOwner,
    r_address: &[F],
) -> Result<(Vec<u64>, Vec<F>), RamCycleError> {
    let leaves = owner.block_topology().leaf_cycles().len();
    let mut blocks = Vec::with_capacity(leaves);
    let mut values = Vec::with_capacity(leaves);
    for_each_leaf_access(MEMBER, owner, |cycle, address| {
        let value = match address {
            Some(address) => eq_at_boolean_index(MEMBER, r_address, u64::from(address))?,
            None => F::zero(),
        };
        blocks.push(u64::from(cycle));
        values.push(value);
        Ok(())
    })?;
    Ok((blocks, values))
}
