//! Independent address-phase oracle for the successor topology and handoff.
//!
//! It deliberately does not import Jolt field, polynomial, or sumcheck code.

pub const AKITA_OFFSET: u128 = 0xffff_a7f7;
pub const AKITA_MODULUS: u128 = u128::MAX - (AKITA_OFFSET - 1);
pub const STAGES: usize = 9;
pub const BASE_STAGES: usize = 5;
pub const VALUE_TABLES: usize = 6;
pub const SIGN_BIT: u32 = 1 << 31;
pub const INNER_MASK: u32 = (1 << 15) - 1;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp(u128);

impl Fp {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    pub fn new(value: u128) -> Result<Self, OracleError> {
        if value >= AKITA_MODULUS {
            return Err(OracleError::NonCanonicalField);
        }
        Ok(Self(value))
    }

    pub const fn canonical(self) -> u128 {
        self.0
    }

    pub fn from_u64(value: u64) -> Self {
        Self(u128::from(value))
    }

    pub fn from_signed_magnitude(magnitude: u64, negative: bool) -> Self {
        let value = Self::from_u64(magnitude);
        if negative { value.neg() } else { value }
    }

    pub fn add(self, rhs: Self) -> Self {
        let (mut value, mut carry) = self.0.overflowing_add(rhs.0);
        while carry {
            (value, carry) = value.overflowing_add(AKITA_OFFSET);
        }
        if value >= AKITA_MODULUS {
            value -= AKITA_MODULUS;
        }
        Self(value)
    }

    pub fn neg(self) -> Self {
        if self == Self::ZERO {
            self
        } else {
            Self(AKITA_MODULUS - self.0)
        }
    }

    pub fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }

    pub fn mul(self, rhs: Self) -> Self {
        let mut product = Self::ZERO;
        let mut addend = self;
        let mut scalar = rhs.0;
        while scalar != 0 {
            if scalar & 1 != 0 {
                product = product.add(addend);
            }
            scalar >>= 1;
            if scalar != 0 {
                addend = addend.add(addend);
            }
        }
        product
    }

    pub fn square(self) -> Self {
        self.mul(self)
    }

    pub fn pow(self, mut exponent: u128) -> Self {
        let mut result = Self::ONE;
        let mut base = self;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.mul(base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.square();
            }
        }
        result
    }

    pub fn inverse(self) -> Result<Self, OracleError> {
        if self == Self::ZERO {
            return Err(OracleError::DivisionByZero);
        }
        Ok(self.pow(AKITA_MODULUS - 2))
    }

    pub fn div(self, rhs: Self) -> Result<Self, OracleError> {
        Ok(self.mul(rhs.inverse()?))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Row {
    pub push_pc: usize,
    pub fused_inc_magnitude: u64,
    pub fused_inc_negative: bool,
}

impl Row {
    pub fn from_mapped_pc(mapped_pc: Option<usize>, fused_inc: i64) -> Self {
        Self {
            push_pc: mapped_pc.unwrap_or(0),
            fused_inc_magnitude: fused_inc.unsigned_abs(),
            fused_inc_negative: fused_inc.is_negative(),
        }
    }

    pub fn fused_inc(self) -> Fp {
        Fp::from_signed_magnitude(self.fused_inc_magnitude, self.fused_inc_negative)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedCell(u32);

impl PackedCell {
    pub fn new(start: usize, count: usize) -> Result<Self, OracleError> {
        let start = u16::try_from(start).map_err(|_| OracleError::InvalidTopology)?;
        let count = u16::try_from(count).map_err(|_| OracleError::InvalidTopology)?;
        Ok(Self(u32::from(start) | (u32::from(count) << 16)))
    }

    pub const fn start(self) -> usize {
        (self.0 & 0xffff) as usize
    }

    pub const fn count(self) -> usize {
        (self.0 >> 16) as usize
    }

    pub const fn words(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressMajorTopology {
    pub addresses: usize,
    pub outer_length: usize,
    pub inner_length: usize,
    pub layout: Vec<PackedCell>,
    pub inner_sign: Vec<u32>,
    pub magnitudes: Vec<u64>,
}

impl AddressMajorTopology {
    pub fn cell(&self, address: usize, outer: usize) -> Result<PackedCell, OracleError> {
        if address >= self.addresses || outer >= self.outer_length {
            return Err(OracleError::InvalidTopology);
        }
        Ok(self.layout[address * self.outer_length + outer])
    }

    pub fn rows(&self) -> Result<usize, OracleError> {
        self.outer_length
            .checked_mul(self.inner_length)
            .ok_or(OracleError::SizeOverflow)
    }

    pub fn validate(&self) -> Result<(), OracleError> {
        let rows = self.rows()?;
        if self.addresses == 0
            || self.inner_length == 0
            || self.inner_length > (INNER_MASK as usize + 1)
            || self.layout.len() != self.addresses * self.outer_length
            || self.inner_sign.len() != rows
            || self.magnitudes.len() != rows
        {
            return Err(OracleError::InvalidTopology);
        }
        for outer in 0..self.outer_length {
            let mut expected_start = 0;
            for address in 0..self.addresses {
                let cell = self.cell(address, outer)?;
                if cell.start() != expected_start {
                    return Err(OracleError::InvalidTopology);
                }
                expected_start = expected_start
                    .checked_add(cell.count())
                    .ok_or(OracleError::SizeOverflow)?;
            }
            if expected_start != self.inner_length {
                return Err(OracleError::InvalidTopology);
            }
            let base = outer * self.inner_length;
            for &inner_sign in &self.inner_sign[base..base + self.inner_length] {
                let inner = (inner_sign & INNER_MASK) as usize;
                if inner >= self.inner_length || inner_sign & !(INNER_MASK | SIGN_BIT) != 0 {
                    return Err(OracleError::InvalidTopology);
                }
            }
        }
        Ok(())
    }
}

pub fn build_topology(
    rows: &[Row],
    addresses: usize,
    inner_length: usize,
) -> Result<AddressMajorTopology, OracleError> {
    if rows.is_empty()
        || addresses == 0
        || inner_length == 0
        || rows.len() % inner_length != 0
        || inner_length > (INNER_MASK as usize + 1)
        || rows.iter().any(|row| row.push_pc >= addresses)
    {
        return Err(OracleError::InvalidShape);
    }
    let outer_length = rows.len() / inner_length;
    let mut topology = AddressMajorTopology {
        addresses,
        outer_length,
        inner_length,
        layout: vec![PackedCell::default(); addresses * outer_length],
        inner_sign: vec![0; rows.len()],
        magnitudes: vec![0; rows.len()],
    };
    let mut counts = vec![0_usize; addresses];
    let mut cursors = vec![0_usize; addresses];

    for outer in 0..outer_length {
        counts.fill(0);
        let base = outer * inner_length;
        for row in &rows[base..base + inner_length] {
            counts[row.push_pc] += 1;
        }

        let mut start = 0;
        for address in 0..addresses {
            topology.layout[address * outer_length + outer] =
                PackedCell::new(start, counts[address])?;
            cursors[address] = start;
            start += counts[address];
        }
        if start != inner_length {
            return Err(OracleError::InvalidTopology);
        }

        for (inner, row) in rows[base..base + inner_length].iter().enumerate() {
            let destination = base + cursors[row.push_pc];
            cursors[row.push_pc] += 1;
            topology.inner_sign[destination] =
                (inner as u32) | if row.fused_inc_negative { SIGN_BIT } else { 0 };
            topology.magnitudes[destination] = row.fused_inc_magnitude;
        }
    }
    topology.validate()?;
    Ok(topology)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitEqTables {
    pub lo: Vec<Vec<Fp>>,
    pub hi: Vec<Vec<Fp>>,
}

pub fn split_eq_tables(
    stage_points: &[Vec<Fp>],
    inner_log2: usize,
) -> Result<SplitEqTables, OracleError> {
    if stage_points.len() != STAGES
        || stage_points.is_empty()
        || stage_points[0].len() < inner_log2
        || stage_points
            .iter()
            .any(|point| point.len() != stage_points[0].len())
    {
        return Err(OracleError::InvalidPoint);
    }
    let hi_bits = stage_points[0].len() - inner_log2;
    let mut lo = Vec::with_capacity(STAGES);
    let mut hi = Vec::with_capacity(STAGES);
    for point in stage_points {
        hi.push(eq_table(&point[..hi_bits])?);
        lo.push(eq_table(&point[hi_bits..])?);
    }
    Ok(SplitEqTables { lo, hi })
}

pub fn eq_table(point: &[Fp]) -> Result<Vec<Fp>, OracleError> {
    let length = 1_usize
        .checked_shl(point.len() as u32)
        .ok_or(OracleError::SizeOverflow)?;
    let mut table = Vec::with_capacity(length);
    table.push(Fp::ONE);
    for &challenge in point {
        let one_minus = Fp::ONE.sub(challenge);
        let mut next = Vec::with_capacity(table.len() * 2);
        for value in table {
            next.push(value.mul(one_minus));
            next.push(value.mul(challenge));
        }
        table = next;
    }
    Ok(table)
}

pub fn direct_pushforwards(
    rows: &[Row],
    addresses: usize,
    inner_length: usize,
    eq: &SplitEqTables,
) -> Result<Vec<Vec<Fp>>, OracleError> {
    validate_pushforward_shape(rows.len(), addresses, inner_length, eq)?;
    if rows.iter().any(|row| row.push_pc >= addresses) {
        return Err(OracleError::InvalidShape);
    }
    let mut output = vec![vec![Fp::ZERO; addresses]; STAGES];
    for (index, row) in rows.iter().copied().enumerate() {
        let outer = index / inner_length;
        let inner = index % inner_length;
        for stage in 0..STAGES {
            let mut value = eq.hi[stage][outer].mul(eq.lo[stage][inner]);
            if stage >= BASE_STAGES {
                value = value.mul(row.fused_inc());
            }
            output[stage][row.push_pc] = output[stage][row.push_pc].add(value);
        }
    }
    Ok(output)
}

pub fn topology_pushforwards(
    topology: &AddressMajorTopology,
    eq: &SplitEqTables,
) -> Result<Vec<Vec<Fp>>, OracleError> {
    topology.validate()?;
    validate_pushforward_shape(
        topology.rows()?,
        topology.addresses,
        topology.inner_length,
        eq,
    )?;
    let mut output = vec![vec![Fp::ZERO; topology.addresses]; STAGES];
    for address in 0..topology.addresses {
        for outer in 0..topology.outer_length {
            let cell = topology.cell(address, outer)?;
            let base = outer * topology.inner_length + cell.start();
            let end = base + cell.count();
            let mut sums = [Fp::ZERO; STAGES];
            for slot in base..end {
                let inner_sign = topology.inner_sign[slot];
                let inner = (inner_sign & INNER_MASK) as usize;
                let increment = Fp::from_signed_magnitude(
                    topology.magnitudes[slot],
                    inner_sign & SIGN_BIT != 0,
                );
                for stage in 0..STAGES {
                    let mut value = eq.lo[stage][inner];
                    if stage >= BASE_STAGES {
                        value = value.mul(increment);
                    }
                    sums[stage] = sums[stage].add(value);
                }
            }
            for stage in 0..STAGES {
                output[stage][address] =
                    output[stage][address].add(sums[stage].mul(eq.hi[stage][outer]));
            }
        }
    }
    Ok(output)
}

fn validate_pushforward_shape(
    rows: usize,
    addresses: usize,
    inner_length: usize,
    eq: &SplitEqTables,
) -> Result<(), OracleError> {
    if rows == 0
        || addresses == 0
        || inner_length == 0
        || rows % inner_length != 0
        || eq.lo.len() != STAGES
        || eq.hi.len() != STAGES
        || eq.lo.iter().any(|table| table.len() != inner_length)
        || eq.hi.iter().any(|table| table.len() != rows / inner_length)
    {
        return Err(OracleError::InvalidShape);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageValueSource {
    Table(usize),
    Complement(usize),
}

pub const STAGE_VALUES: [StageValueSource; STAGES] = [
    StageValueSource::Table(0),
    StageValueSource::Table(1),
    StageValueSource::Table(2),
    StageValueSource::Table(3),
    StageValueSource::Table(4),
    StageValueSource::Table(5),
    StageValueSource::Table(5),
    StageValueSource::Complement(5),
    StageValueSource::Complement(5),
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AddressState {
    pub pushforwards: Vec<Vec<Fp>>,
    pub values: Vec<Vec<Fp>>,
    pub int_table: Vec<Fp>,
    pub entry_trace: Vec<Fp>,
    pub entry_expected: Vec<Fp>,
    pub stage_weights: [Fp; STAGES],
    pub raf_weights: [Fp; STAGES],
    pub entry_weight: Fp,
}

impl AddressState {
    pub fn validate(&self) -> Result<usize, OracleError> {
        if self.pushforwards.len() != STAGES || self.values.len() != VALUE_TABLES {
            return Err(OracleError::InvalidAddressState);
        }
        let length = self.int_table.len();
        if length == 0
            || !length.is_power_of_two()
            || self.entry_trace.len() != length
            || self.entry_expected.len() != length
            || self.pushforwards.iter().any(|table| table.len() != length)
            || self.values.iter().any(|table| table.len() != length)
        {
            return Err(OracleError::InvalidAddressState);
        }
        Ok(length)
    }

    pub fn boolean_claim(&self) -> Result<Fp, OracleError> {
        let length = self.validate()?;
        let mut claim = Fp::ZERO;
        for index in 0..length {
            claim = claim.add(self.summand(index, Fp::ZERO)?);
        }
        Ok(claim)
    }

    pub fn message(&self, previous_claim: Fp) -> Result<[Fp; 3], OracleError> {
        let length = self.validate()?;
        if length < 2 {
            return Err(OracleError::InvalidAddressState);
        }
        let mut at_zero = Fp::ZERO;
        let mut at_two = Fp::ZERO;
        for pair in 0..length / 2 {
            at_zero = at_zero.add(self.pair_summand(pair, Fp::ZERO)?);
            at_two = at_two.add(self.pair_summand(pair, Fp::from_u64(2))?);
        }
        Ok([at_zero, previous_claim.sub(at_zero), at_two])
    }

    pub fn bind(&mut self, challenge: Fp) -> Result<(), OracleError> {
        self.validate()?;
        for table in self
            .pushforwards
            .iter_mut()
            .chain(self.values.iter_mut())
            .chain([
                &mut self.int_table,
                &mut self.entry_trace,
                &mut self.entry_expected,
            ])
        {
            bind_table(table, challenge)?;
        }
        Ok(())
    }

    pub fn output(&self) -> Result<Fp, OracleError> {
        if self.validate()? != 1 {
            return Err(OracleError::InvalidAddressState);
        }
        self.summand(0, Fp::ZERO)
    }

    pub fn raw_bound_values(&self) -> Result<[Fp; VALUE_TABLES], OracleError> {
        if self.validate()? != 1 {
            return Err(OracleError::InvalidAddressState);
        }
        Ok(core::array::from_fn(|index| self.values[index][0]))
    }

    fn summand(&self, index: usize, _unused: Fp) -> Result<Fp, OracleError> {
        let mut sum = self
            .entry_weight
            .mul(self.entry_trace[index])
            .mul(self.entry_expected[index]);
        for stage in 0..STAGES {
            let value = stage_value(STAGE_VALUES[stage], &self.values, index)?;
            let with_raf = value.add(self.raf_weights[stage].mul(self.int_table[index]));
            sum = sum.add(
                self.stage_weights[stage]
                    .mul(self.pushforwards[stage][index])
                    .mul(with_raf),
            );
        }
        Ok(sum)
    }

    fn pair_summand(&self, pair: usize, point: Fp) -> Result<Fp, OracleError> {
        let interpolate = |table: &[Fp]| {
            let low = table[2 * pair];
            low.add(point.mul(table[2 * pair + 1].sub(low)))
        };
        let int = interpolate(&self.int_table);
        let mut sum = self
            .entry_weight
            .mul(interpolate(&self.entry_trace))
            .mul(interpolate(&self.entry_expected));
        for stage in 0..STAGES {
            let value = match STAGE_VALUES[stage] {
                StageValueSource::Table(index) => interpolate(&self.values[index]),
                StageValueSource::Complement(index) => {
                    Fp::ONE.sub(interpolate(&self.values[index]))
                }
            };
            let with_raf = value.add(self.raf_weights[stage].mul(int));
            sum = sum.add(
                self.stage_weights[stage]
                    .mul(interpolate(&self.pushforwards[stage]))
                    .mul(with_raf),
            );
        }
        Ok(sum)
    }
}

fn stage_value(
    source: StageValueSource,
    values: &[Vec<Fp>],
    index: usize,
) -> Result<Fp, OracleError> {
    match source {
        StageValueSource::Table(table) => values
            .get(table)
            .and_then(|values| values.get(index))
            .copied()
            .ok_or(OracleError::InvalidAddressState),
        StageValueSource::Complement(table) => values
            .get(table)
            .and_then(|values| values.get(index))
            .copied()
            .map(|value| Fp::ONE.sub(value))
            .ok_or(OracleError::InvalidAddressState),
    }
}

pub fn bind_table(table: &mut Vec<Fp>, challenge: Fp) -> Result<(), OracleError> {
    if table.len() < 2 || !table.len().is_power_of_two() {
        return Err(OracleError::InvalidAddressState);
    }
    let next = (0..table.len() / 2)
        .map(|pair| {
            let low = table[2 * pair];
            low.add(challenge.mul(table[2 * pair + 1].sub(low)))
        })
        .collect();
    *table = next;
    Ok(())
}

pub fn evaluate_quadratic(samples: [Fp; 3], point: Fp) -> Result<Fp, OracleError> {
    let two = Fp::from_u64(2);
    let l0 = point.sub(Fp::ONE).mul(point.sub(two)).div(two)?;
    let l1 = point.neg().mul(point.sub(two));
    let l2 = point.mul(point.sub(Fp::ONE)).div(two)?;
    Ok(samples[0]
        .mul(l0)
        .add(samples[1].mul(l1))
        .add(samples[2].mul(l2)))
}

pub fn gamma_weights(gamma: Fp) -> ([Fp; STAGES], [Fp; STAGES], Fp) {
    let mut powers = [Fp::ONE; STAGES + 3];
    for index in 1..powers.len() {
        powers[index] = powers[index - 1].mul(gamma);
    }
    let stage_weights = core::array::from_fn(|index| powers[index]);
    let mut raf_weights = [Fp::ZERO; STAGES];
    raf_weights[0] = powers[STAGES];
    raf_weights[2] = powers[STAGES - 1];
    (stage_weights, raf_weights, powers[STAGES + 2])
}

pub fn checksum(tables: &[Vec<Fp>]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in tables.iter().flatten() {
        for byte in value.canonical().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OracleError {
    NonCanonicalField,
    DivisionByZero,
    InvalidShape,
    InvalidPoint,
    InvalidTopology,
    InvalidAddressState,
    SizeOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(value: u64) -> Fp {
        Fp::from_u64(value)
    }

    fn fixture_rows() -> Vec<Row> {
        (0..16)
            .map(|index| Row {
                push_pc: match index % 5 {
                    0 => 0,
                    1 => 3,
                    _ => (index * 3 + 1) % 4,
                },
                fused_inc_magnitude: if index == 7 {
                    u64::MAX
                } else {
                    (index * 11) as u64
                },
                fused_inc_negative: index % 3 == 0,
            })
            .collect()
    }

    fn points() -> Vec<Vec<Fp>> {
        (0..STAGES)
            .map(|stage| {
                (0..4)
                    .map(|bit| f(3 + stage as u64 * 7 + bit as u64 * 5))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn near_modulus_arithmetic_is_canonical() {
        let high = Fp::new(AKITA_MODULUS - 1).unwrap();
        assert_eq!(high.add(Fp::ONE), Fp::ZERO);
        assert_eq!(high.mul(high), Fp::ONE);
        assert_eq!(f(17).div(f(17)).unwrap(), Fp::ONE);
    }

    #[test]
    fn absent_pc_pushes_to_zero() {
        assert_eq!(Row::from_mapped_pc(None, -9).push_pc, 0);
        assert_eq!(Row::from_mapped_pc(Some(0), 9).push_pc, 0);
    }

    #[test]
    fn address_major_topology_matches_direct_definition() {
        let rows = fixture_rows();
        let topology = build_topology(&rows, 4, 4).unwrap();
        let eq = split_eq_tables(&points(), 2).unwrap();
        let direct = direct_pushforwards(&rows, 4, 4, &eq).unwrap();
        let grouped = topology_pushforwards(&topology, &eq).unwrap();
        assert_eq!(grouped, direct);
        assert_ne!(checksum(&direct), 0);
    }

    #[test]
    fn packed_layout_is_address_major_and_outer_local() {
        let rows = fixture_rows();
        let topology = build_topology(&rows, 4, 4).unwrap();
        for outer in 0..4 {
            let cells = (0..4)
                .map(|address| topology.cell(address, outer).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(cells[0].start(), 0);
            assert_eq!(cells.iter().map(|cell| cell.count()).sum::<usize>(), 4);
            assert_eq!(cells[3].start() + cells[3].count(), 4);
        }
    }

    #[test]
    fn host_round_handoff_matches_direct_sumcheck() {
        let addresses = 4;
        let rows = fixture_rows();
        let eq = split_eq_tables(&points(), 2).unwrap();
        let pushforwards = direct_pushforwards(&rows, addresses, 4, &eq).unwrap();
        let values = (0..VALUE_TABLES)
            .map(|table| {
                (0..addresses)
                    .map(|index| f(11 + table as u64 * 13 + index as u64 * 17))
                    .collect()
            })
            .collect();
        let int_table = (0..addresses).map(|index| f(index as u64)).collect();
        let mut entry_trace = vec![Fp::ZERO; addresses];
        entry_trace[3] = Fp::ONE;
        let mut entry_expected = vec![Fp::ZERO; addresses];
        entry_expected[3] = Fp::ONE;
        let (stage_weights, raf_weights, entry_weight) = gamma_weights(f(19));
        let mut state = AddressState {
            pushforwards,
            values,
            int_table,
            entry_trace,
            entry_expected,
            stage_weights,
            raf_weights,
            entry_weight,
        };
        let challenges = [f(23), f(29)];
        let mut claim = state.boolean_claim().unwrap();
        for challenge in challenges {
            let message = state.message(claim).unwrap();
            assert_eq!(message[0].add(message[1]), claim);
            claim = evaluate_quadratic(message, challenge).unwrap();
            state.bind(challenge).unwrap();
        }
        assert_eq!(state.output().unwrap(), claim);
        assert_eq!(state.raw_bound_values().unwrap().len(), VALUE_TABLES);
    }
}
