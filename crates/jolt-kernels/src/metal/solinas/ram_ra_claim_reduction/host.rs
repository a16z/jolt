use jolt_field::Field;
use jolt_poly::EqPolynomial;

use super::{RamRaClaimError, RamRaClaimShape, RAM_RA_CLAIM_TERMS};

pub struct RamRaClaimPrefixState<F: Field> {
    shape: RamRaClaimShape,
    p: [Vec<F>; RAM_RA_CLAIM_TERMS],
    q: [Vec<F>; RAM_RA_CLAIM_TERMS],
    eq_hi: [Vec<F>; RAM_RA_CLAIM_TERMS],
    r_cycle_lo: [Vec<F>; RAM_RA_CLAIM_TERMS],
    gamma_powers: [F; RAM_RA_CLAIM_TERMS],
    challenges: Vec<F>,
}

pub struct RamRaClaimSuffixSeed<F: Field> {
    shape: RamRaClaimShape,
    eq_hi: [Vec<F>; RAM_RA_CLAIM_TERMS],
    scales: [F; RAM_RA_CLAIM_TERMS],
    gamma_powers: [F; RAM_RA_CLAIM_TERMS],
    r_prefix: Vec<F>,
}

pub struct RamRaClaimSuffixState<F: Field> {
    h: Vec<F>,
    eq_hi: [Vec<F>; RAM_RA_CLAIM_TERMS],
    scales: [F; RAM_RA_CLAIM_TERMS],
    coefficients: [F; RAM_RA_CLAIM_TERMS],
    rounds_bound: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamRaClaimHostOutput<F: Field> {
    pub ram_ra: F,
    pub derived_cycle_eq: [F; RAM_RA_CLAIM_TERMS],
}

impl<F: Field> RamRaClaimPrefixState<F> {
    pub fn new(
        shape: RamRaClaimShape,
        q: [Vec<F>; RAM_RA_CLAIM_TERMS],
        cycle_points: [&[F]; RAM_RA_CLAIM_TERMS],
        gamma: F,
    ) -> Result<Self, RamRaClaimError> {
        for table in &q {
            require_length("Q", table.len(), shape.prefix_length())?;
        }
        for point in cycle_points {
            require_length("cycle point", point.len(), shape.log_t())?;
        }
        let p =
            cycle_points.map(|point| EqPolynomial::<F>::evals(&point[shape.suffix_bits()..], None));
        let eq_hi =
            cycle_points.map(|point| EqPolynomial::<F>::evals(&point[..shape.suffix_bits()], None));
        let r_cycle_lo = cycle_points.map(|point| point[shape.suffix_bits()..].to_vec());
        Ok(Self {
            shape,
            p,
            q,
            eq_hi,
            r_cycle_lo,
            gamma_powers: [F::one(), gamma, gamma * gamma],
            challenges: Vec::with_capacity(shape.prefix_bits()),
        })
    }

    pub fn message_evals(&self) -> Result<[F; 2], RamRaClaimError> {
        require_active_round(self.p[0].len())?;
        let mut evals = [F::zero(); 2];
        for term in 0..RAM_RA_CLAIM_TERMS {
            let mut sum = [F::zero(); 2];
            for pair in 0..self.p[term].len() / 2 {
                let p_0 = self.p[term][2 * pair];
                let p_1 = self.p[term][2 * pair + 1];
                let q_0 = self.q[term][2 * pair];
                let q_1 = self.q[term][2 * pair + 1];
                sum[0] += p_0 * q_0;
                sum[1] += (p_1 + p_1 - p_0) * (q_1 + q_1 - q_0);
            }
            evals[0] += self.gamma_powers[term] * sum[0];
            evals[1] += self.gamma_powers[term] * sum[1];
        }
        Ok(evals)
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), RamRaClaimError> {
        if self.challenges.len() >= self.shape.prefix_bits() {
            return Err(RamRaClaimError::OracleState {
                length: self.p[0].len(),
            });
        }
        for table in self.p.iter_mut().chain(&mut self.q) {
            bind_pairs(table, challenge)?;
        }
        self.challenges.push(challenge);
        Ok(())
    }

    pub fn finish(self) -> Result<RamRaClaimSuffixSeed<F>, RamRaClaimError> {
        if self.challenges.len() != self.shape.prefix_bits() || self.p[0].len() != 1 {
            return Err(RamRaClaimError::OracleState {
                length: self.p[0].len(),
            });
        }
        let r_prefix = self.challenges.iter().rev().copied().collect::<Vec<_>>();
        let scales =
            core::array::from_fn(|term| EqPolynomial::<F>::mle(&self.r_cycle_lo[term], &r_prefix));
        Ok(RamRaClaimSuffixSeed {
            shape: self.shape,
            eq_hi: self.eq_hi,
            scales,
            gamma_powers: self.gamma_powers,
            r_prefix,
        })
    }
}

impl<F: Field> RamRaClaimSuffixSeed<F> {
    pub fn r_prefix(&self) -> &[F] {
        &self.r_prefix
    }

    pub fn start(self, h: Vec<F>) -> Result<RamRaClaimSuffixState<F>, RamRaClaimError> {
        require_length("H-prime", h.len(), self.shape.suffix_length())?;
        let coefficients = core::array::from_fn(|term| self.gamma_powers[term] * self.scales[term]);
        Ok(RamRaClaimSuffixState {
            h,
            eq_hi: self.eq_hi,
            scales: self.scales,
            coefficients,
            rounds_bound: 0,
        })
    }
}

impl<F: Field> RamRaClaimSuffixState<F> {
    pub fn message_evals(&self) -> Result<[F; 2], RamRaClaimError> {
        require_active_round(self.h.len())?;
        let mut evals = [F::zero(); 2];
        for pair in 0..self.h.len() / 2 {
            let h_0 = self.h[2 * pair];
            let h_1 = self.h[2 * pair + 1];
            let mut eq_0 = F::zero();
            let mut eq_2 = F::zero();
            for term in 0..RAM_RA_CLAIM_TERMS {
                let e_0 = self.eq_hi[term][2 * pair];
                let e_1 = self.eq_hi[term][2 * pair + 1];
                eq_0 += self.coefficients[term] * e_0;
                eq_2 += self.coefficients[term] * (e_1 + e_1 - e_0);
            }
            evals[0] += h_0 * eq_0;
            evals[1] += (h_1 + h_1 - h_0) * eq_2;
        }
        Ok(evals)
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), RamRaClaimError> {
        bind_pairs(&mut self.h, challenge)?;
        for table in &mut self.eq_hi {
            bind_pairs(table, challenge)?;
        }
        self.rounds_bound += 1;
        Ok(())
    }

    pub fn finish(self) -> Result<RamRaClaimHostOutput<F>, RamRaClaimError> {
        if self.h.len() != 1 || self.eq_hi.iter().any(|table| table.len() != 1) {
            return Err(RamRaClaimError::OracleState {
                length: self.h.len(),
            });
        }
        Ok(RamRaClaimHostOutput {
            ram_ra: self.h[0],
            derived_cycle_eq: core::array::from_fn(|term| self.scales[term] * self.eq_hi[term][0]),
        })
    }
}

fn require_length(table: &'static str, got: usize, expected: usize) -> Result<(), RamRaClaimError> {
    if got == expected {
        Ok(())
    } else {
        Err(RamRaClaimError::TableLength {
            table,
            expected,
            got,
        })
    }
}

fn require_active_round(length: usize) -> Result<(), RamRaClaimError> {
    if length >= 2 && length.is_power_of_two() {
        Ok(())
    } else {
        Err(RamRaClaimError::OracleState { length })
    }
}

fn bind_pairs<F: Field>(table: &mut Vec<F>, challenge: F) -> Result<(), RamRaClaimError> {
    require_active_round(table.len())?;
    let half = table.len() / 2;
    for pair in 0..half {
        let even = table[2 * pair];
        table[pair] = even + challenge * (table[2 * pair + 1] - even);
    }
    table.truncate(half);
    Ok(())
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use checked fixtures")]
mod tests {
    use jolt_field::AkitaField;

    use super::*;
    use crate::metal::solinas::ram_ra_claim_reduction::{
        oracle::{self, RamRaClaimOracleInputs},
        RAM_RA_CLAIM_ADDRESS_DOMAIN, RAM_RA_CLAIM_NO_ACCESS,
    };

    #[test]
    fn interactive_host_states_match_the_independent_split_oracle() {
        let rows = 1 << 12;
        let shape = RamRaClaimShape::new(rows, RAM_RA_CLAIM_ADDRESS_DOMAIN).unwrap();
        let addresses = (0..rows)
            .map(|row| {
                if row % 5 == 0 {
                    (row % RAM_RA_CLAIM_ADDRESS_DOMAIN) as u32
                } else {
                    RAM_RA_CLAIM_NO_ACCESS
                }
            })
            .collect::<Vec<_>>();
        let r_address = point(RAM_RA_CLAIM_ADDRESS_DOMAIN.ilog2() as usize, 17);
        let cycle_points = [
            point(shape.log_t(), 29),
            point(shape.log_t(), 41),
            point(shape.log_t(), 53),
        ];
        let cycle_refs = cycle_points.each_ref().map(Vec::as_slice);
        let challenges = point(shape.log_t(), 71);
        let gamma = AkitaField::from_u64(83);
        let expected = oracle::split(
            RamRaClaimOracleInputs {
                addresses: &addresses,
                r_address: &r_address,
                cycle_points: cycle_refs,
                gamma,
            },
            &challenges,
        )
        .unwrap();
        let eq_address = EqPolynomial::<AkitaField>::evals(&r_address, None);
        let eq_hi = cycle_refs
            .map(|point| EqPolynomial::<AkitaField>::evals(&point[..shape.suffix_bits()], None));
        let q = oracle::build_q(&addresses, &eq_address, &eq_hi, shape.prefix_bits()).unwrap();

        let mut prefix = RamRaClaimPrefixState::new(shape, q, cycle_refs, gamma).unwrap();
        for (round, &challenge) in challenges[..shape.prefix_bits()].iter().enumerate() {
            assert_eq!(prefix.message_evals().unwrap(), expected.messages[round]);
            prefix.bind(challenge).unwrap();
        }
        let seed = prefix.finish().unwrap();
        let eq_prefix = EqPolynomial::<AkitaField>::evals(seed.r_prefix(), None);
        let h = oracle::gather_h(
            &addresses,
            &eq_address,
            &eq_prefix,
            shape.prefix_bits(),
            shape.suffix_bits(),
        )
        .unwrap();
        let mut suffix = seed.start(h).unwrap();
        for (offset, &challenge) in challenges[shape.prefix_bits()..].iter().enumerate() {
            let round = shape.prefix_bits() + offset;
            assert_eq!(suffix.message_evals().unwrap(), expected.messages[round]);
            suffix.bind(challenge).unwrap();
        }
        let output = suffix.finish().unwrap();
        assert_eq!(output.ram_ra, expected.ram_ra);
        assert_eq!(output.derived_cycle_eq, expected.derived_cycle_eq);
    }

    fn point(length: usize, seed: u64) -> Vec<AkitaField> {
        (0..length)
            .map(|index| AkitaField::from_u64(seed + 13 * index as u64))
            .collect()
    }
}
