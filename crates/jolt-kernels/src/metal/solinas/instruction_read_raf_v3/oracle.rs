use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::instruction::{
    upper_half_all_ones, CANONICAL_INSTRUCTION_ADDRESS,
};
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_poly::{
    IdentityPolynomial, MultilinearEvaluation, OperandPolynomial, OperandSide, UnivariatePoly,
};

use super::{InstructionReadRafV3Error, ADDRESS_BITS};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct InstructionReadRafRow {
    lookup_index: u128,
    table_index: Option<usize>,
    raf_flag: bool,
}

impl InstructionReadRafRow {
    pub(crate) fn new(
        lookup_index: u128,
        table_index: Option<usize>,
        raf_flag: bool,
    ) -> Result<Self, InstructionReadRafV3Error> {
        if let Some(index) = table_index {
            if index >= LookupTableKind::<RISCV_XLEN>::COUNT {
                return Err(InstructionReadRafV3Error::InvalidTable(index));
            }
        }
        Ok(Self {
            lookup_index,
            table_index,
            raf_flag,
        })
    }

    pub(crate) const fn lookup_index(self) -> u128 {
        self.lookup_index
    }

    pub(crate) const fn table_index(self) -> Option<usize> {
        self.table_index
    }

    pub(crate) const fn raf_flag(self) -> bool {
        self.raf_flag
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RoundMessage<F> {
    evaluations: Vec<F>,
}

impl<F: Field> RoundMessage<F> {
    fn new(evaluations: Vec<F>) -> Result<Self, InstructionReadRafV3Error> {
        if evaluations.len() < 2 {
            return Err(InstructionReadRafV3Error::EmptyRoundMessage);
        }
        Ok(Self { evaluations })
    }

    pub(crate) fn evaluations(&self) -> &[F] {
        &self.evaluations
    }

    pub(crate) const fn degree(&self) -> usize {
        self.evaluations.len() - 1
    }

    pub(crate) fn sum_at_boolean_points(&self) -> F {
        self.evaluations[0] + self.evaluations[1]
    }

    pub(crate) fn evaluate(&self, challenge: F) -> F {
        UnivariatePoly::interpolate_over_integers(&self.evaluations).evaluate(challenge)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReadRafOutputClaims<F> {
    pub(crate) lookup_table_flags: Vec<F>,
    pub(crate) instruction_ra: Vec<F>,
    pub(crate) instruction_raf_flag: F,
    pub(crate) output_expression: F,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CycleTables<F> {
    eq_reduction: Vec<F>,
    combined_value: Vec<F>,
    instruction_ra: Vec<Vec<F>>,
}

/// Direct dense relation oracle.
///
/// Address messages evaluate the point-mass relation from its definition.
/// Cycle messages bind dense tables in `(2i, 2i + 1)` order.  This deliberately
/// shares no prefix/suffix decomposition, phase condensation, Gruen split, or
/// Product5 implementation with either candidate backend.
#[derive(Clone, Debug)]
pub(crate) struct DenseReadRafOracle<F> {
    rows: Vec<InstructionReadRafRow>,
    gamma: F,
    r_reduction: Vec<F>,
    virtual_ra: usize,
    address_challenges: Vec<F>,
    cycle_challenges: Vec<F>,
    cycle_tables: Option<CycleTables<F>>,
}

impl<F: Field> DenseReadRafOracle<F> {
    pub(crate) fn new(
        rows: Vec<InstructionReadRafRow>,
        r_reduction: Vec<F>,
        gamma: F,
        virtual_ra: usize,
    ) -> Result<Self, InstructionReadRafV3Error> {
        if rows.is_empty() || !rows.len().is_power_of_two() || rows.len() > u32::MAX as usize {
            return Err(InstructionReadRafV3Error::InvalidCycles(rows.len()));
        }
        let log_t = rows.len().trailing_zeros() as usize;
        if r_reduction.len() != log_t {
            return Err(InstructionReadRafV3Error::ReductionPointLength {
                expected: log_t,
                got: r_reduction.len(),
            });
        }
        if virtual_ra == 0 || !ADDRESS_BITS.is_multiple_of(virtual_ra) {
            return Err(InstructionReadRafV3Error::InvalidVirtualRa(virtual_ra));
        }
        Ok(Self {
            rows,
            gamma,
            r_reduction,
            virtual_ra,
            address_challenges: Vec::with_capacity(ADDRESS_BITS),
            cycle_challenges: Vec::with_capacity(log_t),
            cycle_tables: None,
        })
    }

    pub(crate) fn input_claim(&self) -> F {
        self.rows
            .iter()
            .enumerate()
            .map(|(cycle, &row)| eq_index(&self.r_reduction, cycle) * input_value(row, self.gamma))
            .sum()
    }

    pub(crate) const fn address_rounds_bound(&self) -> usize {
        self.address_challenges.len()
    }

    pub(crate) const fn cycle_rounds_bound(&self) -> usize {
        self.cycle_challenges.len()
    }

    pub(crate) fn address_message(&self) -> Result<RoundMessage<F>, InstructionReadRafV3Error> {
        if self.address_challenges.len() == ADDRESS_BITS {
            return Err(InstructionReadRafV3Error::AddressPhaseComplete);
        }
        dense_address_message(
            &self.rows,
            &self.r_reduction,
            &self.address_challenges,
            self.gamma,
        )
    }

    pub(crate) fn bind_address(&mut self, challenge: F) -> Result<(), InstructionReadRafV3Error> {
        if self.address_challenges.len() == ADDRESS_BITS {
            return Err(InstructionReadRafV3Error::AddressPhaseComplete);
        }
        self.address_challenges.push(challenge);
        if self.address_challenges.len() == ADDRESS_BITS {
            self.cycle_tables = Some(self.materialize_cycle_tables());
        }
        Ok(())
    }

    pub(crate) fn cycle_message(&self) -> Result<RoundMessage<F>, InstructionReadRafV3Error> {
        if self.address_challenges.len() != ADDRESS_BITS {
            return Err(InstructionReadRafV3Error::CyclePhaseNotReady);
        }
        let tables = self
            .cycle_tables
            .as_ref()
            .ok_or(InstructionReadRafV3Error::CyclePhaseNotReady)?;
        if tables.eq_reduction.len() == 1 {
            return Err(InstructionReadRafV3Error::CyclePhaseComplete);
        }
        let degree = self.virtual_ra + 2;
        let half = tables.eq_reduction.len() / 2;
        let evaluations = (0..=degree)
            .map(|node| {
                let point = F::from_u64(node as u64);
                (0..half)
                    .map(|pair| {
                        let mut product = extension_pair(&tables.eq_reduction, pair, point)
                            * extension_pair(&tables.combined_value, pair, point);
                        for ra in &tables.instruction_ra {
                            product *= extension_pair(ra, pair, point);
                        }
                        product
                    })
                    .sum()
            })
            .collect();
        RoundMessage::new(evaluations)
    }

    pub(crate) fn bind_cycle(&mut self, challenge: F) -> Result<(), InstructionReadRafV3Error> {
        if self.address_challenges.len() != ADDRESS_BITS {
            return Err(InstructionReadRafV3Error::CyclePhaseNotReady);
        }
        let tables = self
            .cycle_tables
            .as_mut()
            .ok_or(InstructionReadRafV3Error::CyclePhaseNotReady)?;
        if tables.eq_reduction.len() == 1 {
            return Err(InstructionReadRafV3Error::CyclePhaseComplete);
        }
        bind_low_to_high(&mut tables.eq_reduction, challenge);
        bind_low_to_high(&mut tables.combined_value, challenge);
        for ra in &mut tables.instruction_ra {
            bind_low_to_high(ra, challenge);
        }
        self.cycle_challenges.push(challenge);
        Ok(())
    }

    pub(crate) fn final_claim(&self) -> Result<F, InstructionReadRafV3Error> {
        let tables = self
            .cycle_tables
            .as_ref()
            .ok_or(InstructionReadRafV3Error::CyclePhaseNotReady)?;
        if tables.eq_reduction.len() != 1 {
            return Err(InstructionReadRafV3Error::RoundsRemaining {
                remaining: tables.eq_reduction.len().trailing_zeros() as usize,
            });
        }
        Ok(tables.instruction_ra.iter().fold(
            tables.eq_reduction[0] * tables.combined_value[0],
            |value, ra| value * ra[0],
        ))
    }

    pub(crate) fn output_claims(
        &self,
    ) -> Result<ReadRafOutputClaims<F>, InstructionReadRafV3Error> {
        let tables = self
            .cycle_tables
            .as_ref()
            .ok_or(InstructionReadRafV3Error::CyclePhaseNotReady)?;
        let log_t = self.rows.len().trailing_zeros() as usize;
        if self.cycle_challenges.len() != log_t || tables.eq_reduction.len() != 1 {
            return Err(InstructionReadRafV3Error::RoundsRemaining {
                remaining: log_t.saturating_sub(self.cycle_challenges.len()),
            });
        }

        let mut lookup_table_flags = Vec::with_capacity(LookupTableKind::<RISCV_XLEN>::COUNT);
        for table in 0..LookupTableKind::<RISCV_XLEN>::COUNT {
            let values = self
                .rows
                .iter()
                .map(|row| F::from_u64(u64::from(row.table_index == Some(table))))
                .collect();
            lookup_table_flags.push(fold_cycle_claim(values, &self.cycle_challenges));
        }
        let instruction_raf_flag = fold_cycle_claim(
            self.rows
                .iter()
                .map(|row| F::from_u64(u64::from(row.raf_flag)))
                .collect(),
            &self.cycle_challenges,
        );
        let instruction_ra: Vec<F> = tables
            .instruction_ra
            .iter()
            .map(|values| values[0])
            .collect();

        let gamma2 = self.gamma * self.gamma;
        let r_address = &self.address_challenges;
        let left = OperandPolynomial::new(ADDRESS_BITS, OperandSide::Left).evaluate(r_address);
        let right = OperandPolynomial::new(ADDRESS_BITS, OperandSide::Right).evaluate(r_address);
        let identity = IdentityPolynomial::new(ADDRESS_BITS).evaluate(r_address);
        let upper = if CANONICAL_INSTRUCTION_ADDRESS {
            upper_half_all_ones(r_address)
        } else {
            F::zero()
        };
        let table_sum: F = LookupTableKind::<RISCV_XLEN>::iter()
            .zip(lookup_table_flags.iter().copied())
            .map(|(table, flag)| table.evaluate_mle::<F, F>(r_address) * flag)
            .sum();
        let raf_constant = self.gamma * left + gamma2 * right;
        let raf_flag_coefficient =
            gamma2 * identity - self.gamma * left - gamma2 * right + gamma2 * self.gamma * upper;
        let ra_product: F = instruction_ra.iter().copied().product();
        let output_expression = tables.eq_reduction[0]
            * ra_product
            * (table_sum + raf_constant + raf_flag_coefficient * instruction_raf_flag);

        Ok(ReadRafOutputClaims {
            lookup_table_flags,
            instruction_ra,
            instruction_raf_flag,
            output_expression,
        })
    }

    fn materialize_cycle_tables(&self) -> CycleTables<F> {
        let chunk_bits = ADDRESS_BITS / self.virtual_ra;
        let eq_reduction = (0..self.rows.len())
            .map(|cycle| eq_index(&self.r_reduction, cycle))
            .collect();
        let combined_value = self
            .rows
            .iter()
            .copied()
            .map(|row| combined_value(row, &self.address_challenges, self.gamma))
            .collect();
        let instruction_ra = (0..self.virtual_ra)
            .map(|factor| {
                let start = factor * chunk_bits;
                let end = start + chunk_bits;
                self.rows
                    .iter()
                    .map(|row| {
                        eq_address_range(
                            &self.address_challenges[start..end],
                            row.lookup_index,
                            start,
                        )
                    })
                    .collect()
            })
            .collect();
        CycleTables {
            eq_reduction,
            combined_value,
            instruction_ra,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AddressAtom<F> {
    row: InstructionReadRafRow,
    mass: F,
    cycles: u32,
}

impl<F: Copy> AddressAtom<F> {
    pub(crate) const fn row(&self) -> InstructionReadRafRow {
        self.row
    }

    pub(crate) const fn mass(&self) -> F {
        self.mass
    }

    pub(crate) const fn cycles(&self) -> u32 {
        self.cycles
    }
}

/// Exact distributive compression by `(table, RAF, raw lookup)`.  The raw
/// `u128` key prevents fp128 aliases from merging.
pub(crate) fn aggregate_address_atoms<F: Field>(
    rows: &[InstructionReadRafRow],
    r_reduction: &[F],
) -> Result<Vec<AddressAtom<F>>, InstructionReadRafV3Error> {
    if rows.is_empty() || !rows.len().is_power_of_two() {
        return Err(InstructionReadRafV3Error::InvalidCycles(rows.len()));
    }
    let log_t = rows.len().trailing_zeros() as usize;
    if r_reduction.len() != log_t {
        return Err(InstructionReadRafV3Error::ReductionPointLength {
            expected: log_t,
            got: r_reduction.len(),
        });
    }
    let mut atoms = BTreeMap::<InstructionReadRafRow, (F, u32)>::new();
    for (cycle, &row) in rows.iter().enumerate() {
        let entry = atoms.entry(row).or_insert((F::zero(), 0));
        entry.0 += eq_index(r_reduction, cycle);
        entry.1 = entry
            .1
            .checked_add(1)
            .ok_or(InstructionReadRafV3Error::SizeOverflow("atom cycle count"))?;
    }
    Ok(atoms
        .into_iter()
        .map(|(row, (mass, cycles))| AddressAtom { row, mass, cycles })
        .collect())
}

pub(crate) fn atom_address_message<F: Field>(
    atoms: &[AddressAtom<F>],
    address_challenges: &[F],
    gamma: F,
) -> Result<RoundMessage<F>, InstructionReadRafV3Error> {
    if address_challenges.len() == ADDRESS_BITS {
        return Err(InstructionReadRafV3Error::AddressPhaseComplete);
    }
    let evaluations = (0..=2)
        .map(|node| {
            let point = F::from_u64(node as u64);
            atoms
                .iter()
                .map(|atom| {
                    address_row_contribution(atom.row, atom.mass, address_challenges, point, gamma)
                })
                .sum()
        })
        .collect();
    RoundMessage::new(evaluations)
}

fn dense_address_message<F: Field>(
    rows: &[InstructionReadRafRow],
    r_reduction: &[F],
    address_challenges: &[F],
    gamma: F,
) -> Result<RoundMessage<F>, InstructionReadRafV3Error> {
    let evaluations = (0..=2)
        .map(|node| {
            let point = F::from_u64(node as u64);
            rows.iter()
                .enumerate()
                .map(|(cycle, &row)| {
                    address_row_contribution(
                        row,
                        eq_index(r_reduction, cycle),
                        address_challenges,
                        point,
                        gamma,
                    )
                })
                .sum()
        })
        .collect();
    RoundMessage::new(evaluations)
}

fn address_row_contribution<F: Field>(
    row: InstructionReadRafRow,
    cycle_mass: F,
    address_challenges: &[F],
    current: F,
    gamma: F,
) -> F {
    let round = address_challenges.len();
    let prefix_weight = eq_address_range(address_challenges, row.lookup_index, 0);
    let current_bit = address_bit(row.lookup_index, round);
    let current_eq = if current_bit {
        current
    } else {
        F::one() - current
    };
    let mut address = Vec::with_capacity(ADDRESS_BITS);
    address.extend_from_slice(address_challenges);
    address.push(current);
    address.extend(
        (round + 1..ADDRESS_BITS)
            .map(|coordinate| F::from_u64(u64::from(address_bit(row.lookup_index, coordinate)))),
    );
    cycle_mass * prefix_weight * current_eq * combined_value(row, &address, gamma)
}

fn combined_value<F: Field>(row: InstructionReadRafRow, address: &[F], gamma: F) -> F {
    let table = row.table_index.map_or_else(F::zero, |index| {
        LookupTableKind::<RISCV_XLEN>::iter()
            .find(|table| table.index() == index)
            .map_or_else(F::zero, |table| table.evaluate_mle::<F, F>(address))
    });
    let gamma2 = gamma * gamma;
    if row.raf_flag {
        let identity = IdentityPolynomial::new(ADDRESS_BITS).evaluate(address);
        let upper = if CANONICAL_INSTRUCTION_ADDRESS {
            upper_half_all_ones(address)
        } else {
            F::zero()
        };
        table + gamma2 * identity + gamma2 * gamma * upper
    } else {
        let left = OperandPolynomial::new(ADDRESS_BITS, OperandSide::Left).evaluate(address);
        let right = OperandPolynomial::new(ADDRESS_BITS, OperandSide::Right).evaluate(address);
        table + gamma * left + gamma2 * right
    }
}

fn input_value<F: Field>(row: InstructionReadRafRow, gamma: F) -> F {
    // The canonical-address term constrains the relation; it is not an
    // upstream lookup-operand opening and therefore is absent here.
    let address = boolean_address(row.lookup_index);
    let table = row.table_index.map_or_else(F::zero, |index| {
        LookupTableKind::<RISCV_XLEN>::iter()
            .find(|table| table.index() == index)
            .map_or_else(F::zero, |table| table.evaluate_mle::<F, F>(&address))
    });
    let gamma2 = gamma * gamma;
    if row.raf_flag {
        table + gamma2 * IdentityPolynomial::new(ADDRESS_BITS).evaluate(&address)
    } else {
        let left = OperandPolynomial::new(ADDRESS_BITS, OperandSide::Left).evaluate(&address);
        let right = OperandPolynomial::new(ADDRESS_BITS, OperandSide::Right).evaluate(&address);
        table + gamma * left + gamma2 * right
    }
}

fn boolean_address<F: Field>(lookup_index: u128) -> Vec<F> {
    (0..ADDRESS_BITS)
        .map(|coordinate| F::from_u64(u64::from(address_bit(lookup_index, coordinate))))
        .collect()
}

fn address_bit(lookup_index: u128, coordinate: usize) -> bool {
    lookup_index & (1u128 << (ADDRESS_BITS - 1 - coordinate)) != 0
}

fn eq_address_range<F: Field>(point: &[F], lookup_index: u128, offset: usize) -> F {
    point
        .iter()
        .enumerate()
        .fold(F::one(), |weight, (local, &coordinate)| {
            if address_bit(lookup_index, offset + local) {
                weight * coordinate
            } else {
                weight * (F::one() - coordinate)
            }
        })
}

fn eq_index<F: Field>(point: &[F], index: usize) -> F {
    point
        .iter()
        .enumerate()
        .fold(F::one(), |weight, (coordinate, &challenge)| {
            let bit = index & (1usize << (point.len() - 1 - coordinate)) != 0;
            if bit {
                weight * challenge
            } else {
                weight * (F::one() - challenge)
            }
        })
}

fn extension_pair<F: Field>(values: &[F], pair: usize, point: F) -> F {
    let lo = values[2 * pair];
    let hi = values[2 * pair + 1];
    lo + point * (hi - lo)
}

fn bind_low_to_high<F: Field>(values: &mut Vec<F>, challenge: F) {
    let half = values.len() / 2;
    for pair in 0..half {
        values[pair] = extension_pair(values, pair, challenge);
    }
    values.truncate(half);
}

fn fold_cycle_claim<F: Field>(mut values: Vec<F>, challenges: &[F]) -> F {
    for &challenge in challenges {
        bind_low_to_high(&mut values, challenge);
    }
    values[0]
}
