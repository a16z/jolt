use std::collections::BTreeMap;

use jolt_field::Field;

use super::{RamFamilyV2Error, SparseRamOwner};

pub(crate) const RAM_RA_CLAIM_TERMS: usize = 3;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RamRaClaimOracleResult<F: Field> {
    pub(crate) input_claim: F,
    /// Round evaluations at `t = 0` and `t = 2`, in low-to-high bind order.
    pub(crate) messages: Vec<[F; 2]>,
    pub(crate) ram_ra: F,
    pub(crate) derived_cycle_eq: [F; RAM_RA_CLAIM_TERMS],
    /// Address point followed by the reversed low-to-high challenges.
    pub(crate) output_point: Vec<F>,
}

pub(crate) struct RamRaClaimOracleInputs<'a, F: Field> {
    pub(crate) owner: &'a SparseRamOwner,
    pub(crate) r_address: &'a [F],
    pub(crate) cycle_points: [&'a [F]; RAM_RA_CLAIM_TERMS],
    pub(crate) gamma: F,
}

/// Evaluates the exact RAM RA claim-reduction transcript from sparse access
/// rows, without using the owner's merge topology.
pub(crate) fn ram_ra_claim_reduction<F: Field>(
    inputs: RamRaClaimOracleInputs<'_, F>,
    challenges: &[F],
) -> Result<RamRaClaimOracleResult<F>, RamFamilyV2Error> {
    validate_inputs(&inputs, challenges)?;
    inputs.owner.require_hamming_support()?;

    let gamma_powers = [F::one(), inputs.gamma, inputs.gamma * inputs.gamma];
    let address_weights = inputs
        .owner
        .accesses()
        .iter()
        .map(|access| eq_index(inputs.r_address, u64::from(access.address())))
        .collect::<Vec<_>>();

    let input_claim = inputs.owner.accesses().iter().zip(&address_weights).fold(
        F::zero(),
        |claim, (access, &h)| {
            let e = (0..RAM_RA_CLAIM_TERMS).fold(F::zero(), |sum, term| {
                sum + gamma_powers[term]
                    * eq_index(inputs.cycle_points[term], u64::from(access.cycle()))
            });
            claim + h * e
        },
    );

    let mut messages = Vec::with_capacity(inputs.owner.log_t());
    for round in 0..inputs.owner.log_t() {
        let mut h_by_parent = BTreeMap::<u64, [F; 2]>::new();
        for (access, &h) in inputs.owner.accesses().iter().zip(&address_weights) {
            let cycle = access.cycle();
            let cycle_bits = u64::from(cycle);
            let parent = cycle_bits >> (round + 1);
            let side = ((cycle_bits >> round) & 1) as usize;
            let bound_h = h * low_basis_weight(cycle, &challenges[..round]);
            h_by_parent.entry(parent).or_insert([F::zero(); 2])[side] += bound_h;
        }

        let mut message = [F::zero(); 2];
        for (parent, [h_0, h_1]) in h_by_parent {
            let h_2 = h_1 + h_1 - h_0;
            let even_block = parent << 1;
            let odd_block = even_block | 1;
            let mut e_0 = F::zero();
            let mut e_2 = F::zero();
            for (term, &gamma_power) in gamma_powers.iter().enumerate() {
                let term_0 =
                    bound_cycle_eq(inputs.cycle_points[term], &challenges[..round], even_block);
                let term_1 =
                    bound_cycle_eq(inputs.cycle_points[term], &challenges[..round], odd_block);
                e_0 += gamma_power * term_0;
                e_2 += gamma_power * (term_1 + term_1 - term_0);
            }
            message[0] += h_0 * e_0;
            message[1] += h_2 * e_2;
        }
        messages.push(message);
    }

    let ram_ra = inputs
        .owner
        .accesses()
        .iter()
        .zip(&address_weights)
        .fold(F::zero(), |sum, (access, &h)| {
            sum + h * low_basis_weight(access.cycle(), challenges)
        });
    let output_cycle = challenges.iter().rev().copied().collect::<Vec<_>>();
    let derived_cycle_eq = inputs
        .cycle_points
        .map(|point| eq_points(point, output_cycle.as_slice()));
    let output_point = [inputs.r_address, output_cycle.as_slice()].concat();

    Ok(RamRaClaimOracleResult {
        input_claim,
        messages,
        ram_ra,
        derived_cycle_eq,
        output_point,
    })
}

fn validate_inputs<F: Field>(
    inputs: &RamRaClaimOracleInputs<'_, F>,
    challenges: &[F],
) -> Result<(), RamFamilyV2Error> {
    if inputs.r_address.len() != inputs.owner.address_bits() {
        return Err(RamFamilyV2Error::PointLength {
            point: "address point",
            expected: inputs.owner.address_bits(),
            got: inputs.r_address.len(),
        });
    }
    if challenges.len() != inputs.owner.log_t() {
        return Err(RamFamilyV2Error::PointLength {
            point: "sumcheck challenges",
            expected: inputs.owner.log_t(),
            got: challenges.len(),
        });
    }
    for point in inputs.cycle_points {
        if point.len() != inputs.owner.log_t() {
            return Err(RamFamilyV2Error::PointLength {
                point: "cycle point",
                expected: inputs.owner.log_t(),
                got: point.len(),
            });
        }
    }
    Ok(())
}

fn eq_index<F: Field>(point: &[F], index: u64) -> F {
    point
        .iter()
        .enumerate()
        .fold(F::one(), |value, (coordinate, &r)| {
            let bit = point.len() - coordinate - 1;
            if index >> bit & 1 == 0 {
                value * (F::one() - r)
            } else {
                value * r
            }
        })
}

fn eq_points<F: Field>(lhs: &[F], rhs: &[F]) -> F {
    lhs.iter().zip(rhs).fold(F::one(), |value, (&lhs, &rhs)| {
        value * (lhs * rhs + (F::one() - lhs) * (F::one() - rhs))
    })
}

fn low_basis_weight<F: Field>(cycle: u32, challenges: &[F]) -> F {
    challenges
        .iter()
        .enumerate()
        .fold(F::one(), |weight, (bit, &challenge)| {
            if u64::from(cycle) >> bit & 1 == 0 {
                weight * (F::one() - challenge)
            } else {
                weight * challenge
            }
        })
}

fn bound_cycle_eq<F: Field>(point: &[F], challenges: &[F], remaining_index: u64) -> F {
    let remaining = point.len() - challenges.len();
    let high = eq_index(&point[..remaining], remaining_index);
    let low = challenges
        .iter()
        .enumerate()
        .fold(F::one(), |weight, (bound_bit, &challenge)| {
            let coordinate = point[point.len() - bound_bit - 1];
            weight * (coordinate * challenge + (F::one() - coordinate) * (F::one() - challenge))
        });
    high * low
}
