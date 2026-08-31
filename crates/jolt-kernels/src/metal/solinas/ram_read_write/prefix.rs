use jolt_field::{One as _, Zero as _};
use jolt_field::{Prime128OffsetA7F7 as AkitaField, Ring};

use crate::ram_access::RamAccessRecord;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PrefixWeight {
    weight: AkitaField,
    suffix: AkitaField,
}

impl PrefixWeight {
    fn initial() -> Self {
        Self {
            weight: AkitaField::one(),
            suffix: AkitaField::zero(),
        }
    }

    fn bind(&mut self, cycle: u32, round: usize, challenge: AkitaField) {
        if cycle & (1 << round) == 0 {
            let one_minus_challenge = AkitaField::one() - challenge;
            self.weight *= one_minus_challenge;
            self.suffix = challenge + one_minus_challenge * self.suffix;
        } else {
            self.weight *= challenge;
            self.suffix *= challenge;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AddressPrefixCell {
    address: u32,
    block: u32,
    previous: u64,
    next: u64,
    value: AkitaField,
    ra: AkitaField,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CyclePrefixCell {
    block: u32,
    hamming: AkitaField,
    increment: AkitaField,
}

fn address_frontier(
    records: &[RamAccessRecord],
    weights: &[PrefixWeight],
    rounds_bound: usize,
) -> Vec<AddressPrefixCell> {
    let mut order = (0..records.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&index| {
        let record = records[index];
        (record.address, record.cycle)
    });
    let mut cells = Vec::new();
    let mut cursor = 0;
    while cursor < order.len() {
        let first = records[order[cursor]];
        let block = first.cycle >> rounds_bound;
        let mut end = cursor + 1;
        while end < order.len() {
            let record = records[order[end]];
            if record.address != first.address || record.cycle >> rounds_bound != block {
                break;
            }
            end += 1;
        }
        let mut value = AkitaField::from_u64(first.pre_value);
        let mut ra = AkitaField::zero();
        for &index in &order[cursor..end] {
            let record = records[index];
            let delta = i128::from(record.post_value) - i128::from(record.pre_value);
            value += AkitaField::from_i128(delta) * weights[index].suffix;
            ra += weights[index].weight;
        }
        let last = records[order[end - 1]];
        cells.push(AddressPrefixCell {
            address: first.address,
            block,
            previous: first.pre_value,
            next: last.post_value,
            value,
            ra,
        });
        cursor = end;
    }
    cells
}

fn cycle_frontier(
    records: &[RamAccessRecord],
    weights: &[PrefixWeight],
    rounds_bound: usize,
) -> Vec<CyclePrefixCell> {
    let mut cells = Vec::new();
    let mut cursor = 0;
    while cursor < records.len() {
        let block = records[cursor].cycle >> rounds_bound;
        let mut end = cursor + 1;
        while end < records.len() && records[end].cycle >> rounds_bound == block {
            end += 1;
        }
        let mut hamming = AkitaField::zero();
        let mut increment = AkitaField::zero();
        for (record, weight) in records[cursor..end].iter().zip(&weights[cursor..end]) {
            let delta = i128::from(record.post_value) - i128::from(record.pre_value);
            hamming += weight.weight;
            increment += AkitaField::from_i128(delta) * weight.weight;
        }
        cells.push(CyclePrefixCell {
            block,
            hamming,
            increment,
        });
        cursor = end;
    }
    cells
}

fn address_quadratic(
    cells: &[AddressPrefixCell],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> [AkitaField; 2] {
    let in_bits = e_in.len().trailing_zeros();
    let in_mask = e_in.len() - 1;
    let mut result = [AkitaField::zero(); 2];
    let mut cursor = 0;
    while cursor < cells.len() {
        let first = cells[cursor];
        let parent = first.block >> 1;
        let paired = cells
            .get(cursor + 1)
            .is_some_and(|second| second.address == first.address && second.block >> 1 == parent);
        let (low_value, low_ra, high_value, high_ra) = if paired {
            let second = cells[cursor + 1];
            (first.value, first.ra, second.value, second.ra)
        } else if first.block.is_multiple_of(2) {
            (
                first.value,
                first.ra,
                AkitaField::from_u64(first.next),
                AkitaField::zero(),
            )
        } else {
            (
                AkitaField::from_u64(first.previous),
                AkitaField::zero(),
                first.value,
                first.ra,
            )
        };
        let head = e_out[(parent >> in_bits) as usize] * e_in[parent as usize & in_mask];
        result[0] += head * low_ra * low_value;
        result[1] += head * (high_ra - low_ra) * (high_value - low_value);
        cursor += usize::from(paired) + 1;
    }
    result
}

fn cycle_quadratic(
    cells: &[CyclePrefixCell],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> [AkitaField; 2] {
    let in_bits = e_in.len().trailing_zeros();
    let in_mask = e_in.len() - 1;
    let mut result = [AkitaField::zero(); 2];
    let mut cursor = 0;
    while cursor < cells.len() {
        let first = cells[cursor];
        let parent = first.block >> 1;
        let paired = cells
            .get(cursor + 1)
            .is_some_and(|second| second.block >> 1 == parent);
        let (low_hamming, low_increment, high_hamming, high_increment) = if paired {
            let second = cells[cursor + 1];
            (
                first.hamming,
                first.increment,
                second.hamming,
                second.increment,
            )
        } else if first.block.is_multiple_of(2) {
            (
                first.hamming,
                first.increment,
                AkitaField::zero(),
                AkitaField::zero(),
            )
        } else {
            (
                AkitaField::zero(),
                AkitaField::zero(),
                first.hamming,
                first.increment,
            )
        };
        let head = e_out[(parent >> in_bits) as usize] * e_in[parent as usize & in_mask];
        result[0] += head * low_hamming * low_increment;
        result[1] += head * (high_hamming - low_hamming) * (high_increment - low_increment);
        cursor += usize::from(paired) + 1;
    }
    result
}

fn prefix_table(challenges: &[AkitaField]) -> Vec<PrefixWeight> {
    let mut table = vec![PrefixWeight::initial()];
    for &challenge in challenges {
        let previous = table;
        let one_minus_challenge = AkitaField::one() - challenge;
        table = Vec::with_capacity(2 * previous.len());
        table.extend(previous.iter().map(|entry| PrefixWeight {
            weight: one_minus_challenge * entry.weight,
            suffix: challenge + one_minus_challenge * entry.suffix,
        }));
        table.extend(previous.iter().map(|entry| PrefixWeight {
            weight: challenge * entry.weight,
            suffix: challenge * entry.suffix,
        }));
    }
    table
}

#[cfg(test)]
mod tests {
    use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};

    use super::*;

    fn split_equality(
        point: &[AkitaField],
        outer_bits: usize,
    ) -> (Vec<AkitaField>, Vec<AkitaField>) {
        let split = outer_bits.min(point.len());
        (
            EqPolynomial::new(point[split..].to_vec()).evaluations(),
            EqPolynomial::new(point[..split].to_vec()).evaluations(),
        )
    }

    #[test]
    fn record_prefix_matches_dense_polynomials_through_six_challenges() {
        let log_t = 7;
        let cycles = 1 << log_t;
        let addresses = 5;
        let accesses = [
            (0, 1, 4, 9),
            (1, 1, 9, 9),
            (3, 2, 7, 2),
            (4, 1, 9, 12),
            (7, 4, 3, 8),
            (8, 2, 2, 2),
            (15, 1, 12, 5),
            (16, 4, 8, 1),
            (31, 0, 6, 11),
            (32, 0, 11, 10),
            (63, 2, 2, 13),
            (64, 1, 5, 14),
            (95, 4, 1, 4),
            (126, 2, 13, 6),
            (127, 2, 6, 7),
        ];
        let records = accesses
            .into_iter()
            .map(|(cycle, address, pre_value, post_value)| RamAccessRecord {
                cycle,
                address,
                pre_value,
                post_value,
            })
            .collect::<Vec<_>>();
        let initial = [6u64, 4, 7, 19, 3];
        let mut memory = initial;
        let mut dense_value = vec![AkitaField::zero(); addresses * cycles];
        let mut dense_ra = vec![AkitaField::zero(); addresses * cycles];
        let mut dense_hamming = vec![AkitaField::zero(); cycles];
        let mut dense_increment = vec![AkitaField::zero(); cycles];
        let mut record_cursor = 0;
        for cycle in 0..cycles {
            for address in 0..addresses {
                dense_value[address * cycles + cycle] = AkitaField::from_u64(memory[address]);
            }
            if records
                .get(record_cursor)
                .is_some_and(|record| record.cycle as usize == cycle)
            {
                let record = records[record_cursor];
                assert_eq!(memory[record.address as usize], record.pre_value);
                dense_ra[record.address as usize * cycles + cycle] = AkitaField::one();
                dense_hamming[cycle] = AkitaField::one();
                dense_increment[cycle] = AkitaField::from_i128(
                    i128::from(record.post_value) - i128::from(record.pre_value),
                );
                memory[record.address as usize] = record.post_value;
                record_cursor += 1;
            }
        }
        assert_eq!(record_cursor, records.len());

        let tau = (0..log_t)
            .map(|index| AkitaField::from_u64(17 + 29 * index as u64))
            .collect::<Vec<_>>();
        let mut dense_value = dense_value
            .chunks(cycles)
            .map(|values| Polynomial::new(values.to_vec()))
            .collect::<Vec<_>>();
        let mut dense_ra = dense_ra
            .chunks(cycles)
            .map(|values| Polynomial::new(values.to_vec()))
            .collect::<Vec<_>>();
        let mut dense_hamming = Polynomial::new(dense_hamming);
        let mut dense_increment = Polynomial::new(dense_increment);
        let mut weights = vec![PrefixWeight::initial(); records.len()];
        let mut challenges = Vec::new();

        for rounds_bound in 0..=6 {
            let table = prefix_table(&challenges);
            let table_mask = table.len() - 1;
            for (record, weight) in records.iter().zip(&weights) {
                assert_eq!(
                    *weight,
                    table[record.cycle as usize & table_mask],
                    "table weight after {rounds_bound} binds"
                );
            }
            let address_cells = address_frontier(&records, &weights, rounds_bound);
            let cycle_cells = cycle_frontier(&records, &weights, rounds_bound);
            let head_point = &tau[..log_t - rounds_bound - 1];
            let (e_in, e_out) = split_equality(head_point, log_t / 2);
            let expected_address = dense_value.iter().zip(&dense_ra).fold(
                [AkitaField::zero(); 2],
                |mut sum, (value, ra)| {
                    for parent in 0..value.len() / 2 {
                        let low = 2 * parent;
                        let high = low + 1;
                        let head = EqPolynomial::new(head_point.to_vec()).evaluations()[parent];
                        sum[0] += head * ra.evals()[low] * value.evals()[low];
                        sum[1] += head
                            * (ra.evals()[high] - ra.evals()[low])
                            * (value.evals()[high] - value.evals()[low]);
                    }
                    sum
                },
            );
            let mut expected_cycle = [AkitaField::zero(); 2];
            let eq = EqPolynomial::new(head_point.to_vec()).evaluations();
            for (parent, head) in eq.into_iter().enumerate() {
                let low = 2 * parent;
                let high = low + 1;
                expected_cycle[0] +=
                    head * dense_hamming.evals()[low] * dense_increment.evals()[low];
                expected_cycle[1] += head
                    * (dense_hamming.evals()[high] - dense_hamming.evals()[low])
                    * (dense_increment.evals()[high] - dense_increment.evals()[low]);
            }
            assert_eq!(
                address_quadratic(&address_cells, &e_in, &e_out),
                expected_address,
                "address quadratic after {rounds_bound} binds"
            );
            assert_eq!(
                cycle_quadratic(&cycle_cells, &e_in, &e_out),
                expected_cycle,
                "cycle quadratic after {rounds_bound} binds"
            );

            for cell in &address_cells {
                let index = cell.address as usize;
                assert_eq!(cell.value, dense_value[index].evals()[cell.block as usize]);
                assert_eq!(cell.ra, dense_ra[index].evals()[cell.block as usize]);
            }
            for cell in &cycle_cells {
                assert_eq!(cell.hamming, dense_hamming.evals()[cell.block as usize]);
                assert_eq!(cell.increment, dense_increment.evals()[cell.block as usize]);
            }
            if rounds_bound == 6 {
                break;
            }
            let challenge = AkitaField::from_u64(211 + 31 * rounds_bound as u64);
            challenges.push(challenge);
            for (record, weight) in records.iter().zip(&mut weights) {
                weight.bind(record.cycle, rounds_bound, challenge);
            }
            for polynomial in &mut dense_value {
                polynomial.bind_with_order(challenge, BindingOrder::LowToHigh);
            }
            for polynomial in &mut dense_ra {
                polynomial.bind_with_order(challenge, BindingOrder::LowToHigh);
            }
            dense_hamming.bind_with_order(challenge, BindingOrder::LowToHigh);
            dense_increment.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
    }
}
