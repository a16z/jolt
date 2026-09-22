//! Exact private sequential reads, finalized before a sampler result can escape.
use jolt_field::{CanonicalEncoding, Fr, Ring};
use jolt_r1cs::bn254_bits::ByteVar;
use jolt_r1cs::{LinearCombination, R1csBuilder};

use super::CandidateError;

type Expression = LinearCombination<Fr>;
type Record = [Expression; 3];

pub(super) struct ReadTape<'a> {
    tape: &'a [ByteVar],
    slots: usize,
    records: Vec<Record>,
    cursor: Expression,
}

impl ReadTape<'_> {
    /// Only this closure boundary can release outputs; routing is mandatory.
    pub(super) fn constrain<T, E: From<CandidateError>>(
        builder: &mut R1csBuilder<Fr>,
        tape: &[ByteVar],
        slots: usize,
        sample: impl FnOnce(&mut R1csBuilder<Fr>, &mut ReadTape<'_>) -> Result<T, E>,
    ) -> Result<T, E> {
        let n = slots
            .max(1)
            .checked_next_power_of_two()
            .ok_or(CandidateError::Shape)?;
        // BN254 Fr has characteristic greater than 2^64 and 255. This bounds
        // every prefix count/index and makes the packed Boolean bytes injective.
        let _ = u64::try_from(n).map_err(|_| CandidateError::Shape)?;
        if n > 1 {
            let _ = (n / 2)
                .checked_mul(2 * n.ilog2() as usize - 1)
                .ok_or(CandidateError::Shape)?;
        }
        let _ = u64::try_from(tape.len()).map_err(|_| CandidateError::Shape)?;
        for byte in tape {
            byte.validate_indices(builder)
                .map_err(CandidateError::from)?;
        }
        let mut reads = ReadTape {
            tape,
            slots,
            records: Vec::with_capacity(slots),
            cursor: Expression::zero(),
        };
        let output = sample(builder, &mut reads)?;
        reads.finish(builder, n)?;
        Ok(output)
    }

    pub(super) fn read(
        &mut self,
        builder: &mut R1csBuilder<Fr>,
        active: Expression,
    ) -> Result<[Expression; 8], CandidateError> {
        if self.records.len() == self.slots {
            return Err(CandidateError::Shape);
        }
        builder.assert_product(
            active.clone(),
            active.clone() - Expression::one(),
            Expression::zero(),
        );
        let known_active = builder.evaluate(&active).ok();
        let index = builder
            .evaluate(&self.cursor)
            .ok()
            .and_then(|v| v.to_u64_checked())
            .and_then(|v| usize::try_from(v).ok());
        let witness = match known_active {
            Some(value) if value == Fr::from_u64(0) => Some(0),
            Some(value) if value == Fr::from_u64(1) => index
                .and_then(|i| self.tape.get(i))
                .and_then(|b| builder.evaluate(&b.expression()).ok())
                .and_then(|v| v.to_u64_checked())
                .and_then(|v| u8::try_from(v).ok()),
            _ => None,
        };
        if known_active == Some(Fr::from_u64(1)) && index.is_some_and(|i| i >= self.tape.len()) {
            builder.assert_zero(active);
            return Err(CandidateError::CapacityExceeded);
        }
        let byte = ByteVar::allocate(builder, witness);
        builder.assert_product(
            Expression::one() - active.clone(),
            byte.expression(),
            Expression::zero(),
        );
        self.records
            .push([active.clone(), self.cursor.clone(), byte.expression()]);
        self.cursor = Self::materialize(builder, self.cursor.clone() + active);
        Ok(byte.bit_expressions())
    }

    pub(super) fn consumed(&self) -> Expression {
        self.cursor.clone()
    }

    /// Preserve the public ending-cursor API without a per-read linear scan.
    pub(super) fn one_hot_cursor(&self, builder: &mut R1csBuilder<Fr>) -> Vec<Expression> {
        let known = builder.evaluate(&self.cursor).ok();
        let mut sum = Expression::zero();
        let mut count = Expression::zero();
        let result = (0..=self.tape.len())
            .map(|i| {
                let index = Fr::from_u128(i as u128);
                let value =
                    builder.alloc_witness(known.map(|v| Fr::from_u64(u64::from(v == index))));
                let value = Expression::variable(value);
                builder.assert_product(
                    value.clone(),
                    value.clone() - Expression::one(),
                    Expression::zero(),
                );
                sum = sum.clone() + value.clone();
                count = count.clone() + value.clone().scale(index);
                value
            })
            .collect();
        builder.assert_equal(sum, Expression::one());
        builder.assert_equal(count, self.cursor.clone());
        result
    }

    fn materialize(builder: &mut R1csBuilder<Fr>, value: Expression) -> Expression {
        let variable = builder.alloc_witness(builder.evaluate(&value).ok());
        builder.assert_equal(variable, value);
        variable.into()
    }

    fn finish(mut self, builder: &mut R1csBuilder<Fr>, n: usize) -> Result<(), CandidateError> {
        if self.records.len() != self.slots {
            return Err(CandidateError::Shape);
        }
        self.records
            .resize_with(n, || std::array::from_fn(|_| Expression::zero()));
        let activities: Option<Vec<_>> = self
            .records
            .iter()
            .map(|r| builder.evaluate(&r[0]).ok().map(|x| x == Fr::from_u64(1)))
            .collect();
        let settings = activities
            .map(|active| {
                let mut next_active = 0;
                let mut next_inactive = active.iter().filter(|&&x| x).count();
                let permutation: Vec<_> = active
                    .into_iter()
                    .map(|a| {
                        let next = if a {
                            &mut next_active
                        } else {
                            &mut next_inactive
                        };
                        let result = *next;
                        *next += 1;
                        result
                    })
                    .collect();
                let mut settings = Vec::new();
                Self::route(&permutation, &mut settings)?;
                Ok::<_, CandidateError>(settings)
            })
            .transpose()?;
        let mut offset = 0;
        let output = Self::network(
            self.records,
            settings.as_deref(),
            &mut offset,
            &mut |left, right, swap| Self::switch(builder, left, right, swap),
        )?;
        for (j, record) in output.iter().enumerate() {
            if let Some(next) = output.get(j + 1) {
                builder.assert_product(
                    next[0].clone(),
                    Expression::one() - record[0].clone(),
                    Expression::zero(),
                );
            }
            builder.assert_product(
                record[0].clone(),
                record[1].clone() - Expression::constant(Fr::from_u128(j as u128)),
                Expression::zero(),
            );
            if let Some(byte) = self.tape.get(j) {
                builder.assert_product(
                    record[0].clone(),
                    record[2].clone() - byte.expression(),
                    Expression::zero(),
                );
            } else {
                builder.assert_zero(record[0].clone());
            }
        }
        Ok(())
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "array::from_fn indices are within all three-component records"
    )]
    fn switch(
        builder: &mut R1csBuilder<Fr>,
        left: Record,
        right: Record,
        swap: Option<bool>,
    ) -> (Record, Record) {
        let selector = builder.alloc_witness(swap.map(|s| Fr::from_u64(u64::from(s))));
        let selector = Expression::variable(selector);
        builder.assert_product(
            selector.clone(),
            selector.clone() - Expression::one(),
            Expression::zero(),
        );
        let first: Record = std::array::from_fn(|i| {
            let value = builder
                .evaluate(&left[i])
                .ok()
                .zip(builder.evaluate(&right[i]).ok())
                .zip(swap)
                .map(|((a, b), s)| if s { b } else { a });
            let variable = builder.alloc_witness(value);
            builder.assert_product(
                selector.clone(),
                right[i].clone() - left[i].clone(),
                Expression::variable(variable) - left[i].clone(),
            );
            variable.into()
        });
        let second = std::array::from_fn(|i| {
            Self::materialize(
                builder,
                left[i].clone() + right[i].clone() - first[i].clone(),
            )
        });
        (first, second)
    }

    // The pairing graph is the union of two perfect matchings, hence even cycles.
    // Two-coloring sends one member of each input/output pair into each subnet.
    #[expect(
        clippy::indexing_slicing,
        reason = "validated permutations and power-of-two pairing indices"
    )]
    fn route(permutation: &[usize], settings: &mut Vec<bool>) -> Result<(), CandidateError> {
        let n = permutation.len();
        if !n.is_power_of_two() {
            return Err(CandidateError::Shape);
        }
        let mut inverse = vec![None; n];
        for (i, &destination) in permutation.iter().enumerate() {
            if destination >= n || inverse[destination].is_some() {
                return Err(CandidateError::Shape);
            }
            inverse[destination] = Some(i);
        }
        if n == 1 {
            return Ok(());
        }
        if n == 2 {
            settings.push(permutation[0] == 1);
            return Ok(());
        }
        let mut color = vec![None; n];
        for seed in 0..n {
            if color[seed].is_some() {
                continue;
            }
            color[seed] = Some(false);
            let mut pending = vec![seed];
            while let Some(i) = pending.pop() {
                let c = color[i].ok_or(CandidateError::Shape)?;
                for neighbor in [
                    i ^ 1,
                    inverse[permutation[i] ^ 1].ok_or(CandidateError::Shape)?,
                ] {
                    match color[neighbor] {
                        None => {
                            color[neighbor] = Some(!c);
                            pending.push(neighbor);
                        }
                        Some(other) if other == c => return Err(CandidateError::Shape),
                        Some(_) => {}
                    }
                }
            }
        }
        let mut upper = vec![0; n / 2];
        let mut lower = vec![0; n / 2];
        for i in 0..n / 2 {
            let swap = color[2 * i].ok_or(CandidateError::Shape)?;
            settings.push(swap);
            upper[i] = permutation[2 * i + usize::from(swap)] / 2;
            lower[i] = permutation[2 * i + usize::from(!swap)] / 2;
        }
        Self::route(&upper, settings)?;
        Self::route(&lower, settings)?;
        for j in 0..n / 2 {
            settings.push(
                color[inverse[2 * j].ok_or(CandidateError::Shape)?].ok_or(CandidateError::Shape)?,
            );
        }
        Ok(())
    }

    fn network<T>(
        input: Vec<T>,
        settings: Option<&[bool]>,
        offset: &mut usize,
        switch: &mut impl FnMut(T, T, Option<bool>) -> (T, T),
    ) -> Result<Vec<T>, CandidateError> {
        let n = input.len();
        if !n.is_power_of_two() {
            return Err(CandidateError::Shape);
        }
        if n == 1 {
            return Ok(input);
        }
        let mut upper = Vec::with_capacity(n / 2);
        let mut lower = Vec::with_capacity(n / 2);
        let mut input = input.into_iter();
        while let Some(left) = input.next() {
            let right = input.next().ok_or(CandidateError::Shape)?;
            let setting = settings
                .map(|s| s.get(*offset).copied().ok_or(CandidateError::Shape))
                .transpose()?;
            *offset += 1;
            let (a, b) = switch(left, right, setting);
            upper.push(a);
            lower.push(b);
        }
        if n == 2 {
            upper.extend(lower);
            return Ok(upper);
        }
        let upper = Self::network(upper, settings, offset, switch)?;
        let lower = Self::network(lower, settings, offset, switch)?;
        let mut output = Vec::with_capacity(n);
        for (a, b) in upper.into_iter().zip(lower) {
            let setting = settings
                .map(|s| s.get(*offset).copied().ok_or(CandidateError::Shape))
                .transpose()?;
            *offset += 1;
            let (a, b) = switch(a, b, setting);
            output.push(a);
            output.push(b);
        }
        Ok(output)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "exhaustive topology and adversarial constraint tests"
)]
mod tests {
    use super::*;
    use jolt_r1cs::Variable;

    fn check_permutation(permutation: &[usize]) {
        let mut settings = Vec::new();
        ReadTape::route(permutation, &mut settings).unwrap();
        let mut offset = 0;
        let output = ReadTape::network(
            (0..permutation.len()).collect(),
            Some(&settings),
            &mut offset,
            &mut |a, b, s| if s.unwrap() { (b, a) } else { (a, b) },
        )
        .unwrap();
        assert_eq!(offset, settings.len());
        for (destination, &source) in output.iter().enumerate() {
            assert_eq!(permutation[source], destination);
        }
    }

    #[test]
    fn every_small_permutation_and_large_prescribed_routes() {
        fn enumerate(p: &mut [usize], start: usize) -> usize {
            if start == p.len() {
                check_permutation(p);
                return 1;
            }
            let mut count = 0;
            for i in start..p.len() {
                p.swap(start, i);
                count += enumerate(p, start + 1);
                p.swap(start, i);
            }
            count
        }
        for (n, count) in [(1, 1), (2, 2), (4, 24), (8, 40320)] {
            assert_eq!(enumerate(&mut (0..n).collect::<Vec<_>>(), 0), count);
        }
        for n in [16usize, 128, 1024] {
            check_permutation(&(0..n).map(|i| (i + 7) % n).collect::<Vec<_>>());
            check_permutation(
                &(0..n)
                    .map(|i| i.reverse_bits() >> (usize::BITS - n.ilog2()))
                    .collect::<Vec<_>>(),
            );
        }
        for bad in [vec![], vec![0, 0], vec![0, 2], vec![0, 1, 2]] {
            assert!(ReadTape::route(&bad, &mut Vec::new()).is_err());
        }
    }

    #[test]
    fn routing_constraints_padding_tampering_and_unknown_layout() {
        let emit = |known: bool| {
            let mut builder = R1csBuilder::new();
            let tape: Vec<_> = [0x81u8, 0x12, 0xfa, 0x33]
                .into_iter()
                .map(|x| ByteVar::allocate(&mut builder, known.then_some(x)))
                .collect();
            let records = ReadTape::constrain::<_, CandidateError>(
                &mut builder,
                &tape,
                5,
                |builder, reads| {
                    for active in [true, false, true, false, true] {
                        let var =
                            builder.alloc_witness(known.then_some(Fr::from_u64(u64::from(active))));
                        let _ = reads.read(builder, var.into())?;
                    }
                    Ok((reads.records.clone(), builder.num_vars()))
                },
            )
            .unwrap();
            (builder, tape, records)
        };
        let (builder, tape, (records, first_network)) = emit(true);
        for (record, expected) in records.iter().zip([0x81u64, 0, 0x12, 0, 0xfa]) {
            assert_eq!(
                builder.evaluate(&record[2]).unwrap(),
                Fr::from_u64(expected)
            );
        }
        let witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        let unknown = emit(false).0.into_matrices();
        assert_eq!(matrices.a, unknown.a);
        assert_eq!(matrices.b, unknown.b);
        assert_eq!(matrices.c, unknown.c);
        // Tape, active/inactive bytes, cursor, and activity all remain linked.
        for expression in [
            tape[0].bit_expressions()[0].clone(),
            records[0][2].clone(),
            records[1][2].clone(),
            records[2][1].clone(),
            records[1][0].clone(),
        ] {
            let variable = expression
                .terms
                .iter()
                .find(|(v, _)| *v != Variable::ONE)
                .unwrap()
                .0;
            let mut forged = witness.clone();
            forged[variable.index()] += Fr::from_u64(1);
            assert!(matrices.check_witness(&forged).is_err());
        }
        // Non-Boolean switches and all materialized internal outputs are constrained.
        for i in first_network..witness.len() {
            let mut forged = witness.clone();
            forged[i] += Fr::from_u64(2);
            assert!(matrices.check_witness(&forged).is_err(), "variable {i}");
        }
    }

    #[test]
    fn coherent_forged_byte_or_index_fails_after_rerouting() {
        for (index, value) in [(0u64, 8u8), (1, 7)] {
            let mut builder = R1csBuilder::new();
            let tape = [ByteVar::allocate(&mut builder, Some(7))];
            let byte = ByteVar::allocate(&mut builder, Some(value));
            let reads = ReadTape {
                tape: &tape,
                slots: 1,
                records: vec![[
                    Expression::one(),
                    Expression::constant(Fr::from_u64(index)),
                    byte.expression(),
                ]],
                cursor: Expression::one(),
            };
            reads.finish(&mut builder, 1).unwrap();
            let witness = builder.witness().unwrap();
            assert!(builder.into_matrices().check_witness(&witness).is_err());
        }
        let mut offset = 0;
        assert!(
            ReadTape::network(vec![0, 1], Some(&[]), &mut offset, &mut |a, b, _| (a, b)).is_err()
        );
    }

    #[test]
    fn empty_singleton_and_capacity_are_exact() {
        for (slots, active, tape_len, expected) in [
            (0, false, 0, 0u64),
            (1, false, 0, 0),
            (1, true, 1, 1),
            (3, false, 1, 0),
        ] {
            let mut builder = R1csBuilder::new();
            let tape: Vec<_> = (0..tape_len)
                .map(|_| ByteVar::allocate(&mut builder, Some(9)))
                .collect();
            let count =
                ReadTape::constrain::<_, CandidateError>(&mut builder, &tape, slots, |b, r| {
                    for _ in 0..slots {
                        let _ = r.read(b, Expression::constant(Fr::from_u64(u64::from(active))))?;
                    }
                    Ok(r.consumed())
                })
                .unwrap();
            assert_eq!(builder.evaluate(&count).unwrap(), Fr::from_u64(expected));
            let witness = builder.witness().unwrap();
            assert!(builder.into_matrices().check_witness(&witness).is_ok());
        }
        let mut builder = R1csBuilder::new();
        let failed = ReadTape::constrain::<_, CandidateError>(&mut builder, &[], 1, |b, r| {
            r.read(b, Expression::one())
        });
        assert!(matches!(failed, Err(CandidateError::CapacityExceeded)));
        let witness = builder.witness().unwrap();
        assert!(builder.into_matrices().check_witness(&witness).is_err());
        // Unknown construction retains the capacity row. Fill every auxiliary
        // coherently for one active zero-byte read from an empty tape.
        let mut builder = R1csBuilder::new();
        let (activity, count) =
            ReadTape::constrain::<_, CandidateError>(&mut builder, &[], 1, |b, r| {
                let activity = b.alloc_witness(None);
                let _ = r.read(b, activity.into())?;
                Ok((activity, r.consumed()))
            })
            .unwrap();
        let matrices = builder.into_matrices();
        let mut witness = vec![Fr::from_u64(0); matrices.num_vars];
        witness[Variable::ONE.index()] = Fr::from_u64(1);
        witness[activity.index()] = Fr::from_u64(1);
        witness[count.terms[0].0.index()] = Fr::from_u64(1);
        assert_eq!(
            matrices.check_witness(&witness),
            Err(matrices.num_constraints - 1)
        );
    }
}
