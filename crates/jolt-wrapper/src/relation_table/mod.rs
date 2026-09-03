//! Plonkish row-table lowering of the verifier-algebra R1CS.

mod copy_link;
mod protocol;
mod prover;
mod scalar_link;
#[cfg(all(test, feature = "prover-fixtures"))]
mod tests;

use std::collections::BTreeMap;

use ark_bn254::Fr as ArkFr;
use ark_ff::batch_inversion;
use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_hyperkzg::{NoopVerifierObserver, VerifierObserver};
use jolt_r1cs::{ConstraintMatrices, SparseRow};
use thiserror::Error;

use crate::relation::{Relation, ScheduleEntry};
use crate::stream::{Column, StreamError};

pub use copy_link::{CopyLink, CopyLinkClaims, CopyLinkProver, CopyLinkSide, CopyLinkWitness};
pub use protocol::{
    prove, setup, verify, RelationTableProof, RelationTableProverKey, RelationTableVerifierKey,
};
pub use prover::RelationTableProver;
pub use scalar_link::{DoryScalarLink, DoryScalarLinkProver};

pub const WIRES: usize = 3;
pub const FIXED_COLUMNS: usize = 9;
pub const WITNESS_COLUMNS: usize = 5;
pub const TOTAL_COLUMNS: usize = FIXED_COLUMNS + WITNESS_COLUMNS;
pub const DEGREE: usize = 5;

pub const Q_L: usize = 0;
pub const Q_R: usize = 1;
pub const Q_O: usize = 2;
pub const Q_M: usize = 3;
pub const Q_C: usize = 4;
pub const SIGMA_A: usize = 5;
pub const SIGMA_B: usize = 6;
pub const SIGMA_C: usize = 7;
pub const ACTIVE: usize = 8;

pub const WIRE_A: usize = FIXED_COLUMNS;
pub const WIRE_B: usize = WIRE_A + 1;
pub const WIRE_C: usize = WIRE_B + 1;
pub const H_ID: usize = WIRE_C + 1;
pub const H_SIGMA: usize = H_ID + 1;

#[derive(Debug, Error)]
pub enum RelationTableError {
    #[error("common row domain must be a power of two and hold {minimum} gates, got {actual}")]
    RowDomain { minimum: usize, actual: usize },
    #[error("assignment has {actual} values, expected at least {expected}")]
    Assignment { expected: usize, actual: usize },
    #[error("copy denominator is zero")]
    ZeroDenominator,
    #[error("lowered gate {0} is not satisfied")]
    Gate(usize),
    #[error("copy relation is not satisfied")]
    Copy,
    #[error("claim count mismatch")]
    Claims,
    #[error("stream: {0}")]
    Stream(#[from] StreamError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Symbol {
    Variable(usize),
    Node(usize),
    Unused(usize),
}

#[derive(Clone, Copy, Debug)]
struct Gate {
    selectors: [Fr; 5],
    cells: [Symbol; WIRES],
    active: bool,
}

struct GateBuilder {
    gates: Vec<Gate>,
    next_node: usize,
    next_unused: usize,
}

impl GateBuilder {
    fn new(num_vars: usize) -> Self {
        Self {
            gates: Vec::new(),
            next_node: num_vars,
            next_unused: 0,
        }
    }

    fn unused(&mut self) -> Symbol {
        let symbol = Symbol::Unused(self.next_unused);
        self.next_unused += 1;
        symbol
    }

    fn node(&mut self) -> Symbol {
        let symbol = Symbol::Node(self.next_node);
        self.next_node += 1;
        symbol
    }

    fn linear(&mut self, row: &SparseRow<Fr>) -> Symbol {
        let constant = row
            .iter()
            .filter(|(variable, _)| *variable == 0)
            .map(|(_, coefficient)| *coefficient)
            .sum();
        let terms = row.iter().filter(|(variable, _)| *variable != 0);
        let mut accumulator = None;
        for &(variable, coefficient) in terms {
            let output = self.node();
            let (a, b, q_l, q_r, q_c) = match accumulator {
                Some(previous) => (
                    previous,
                    Symbol::Variable(variable),
                    Fr::one(),
                    coefficient,
                    Fr::zero(),
                ),
                None => (
                    Symbol::Variable(variable),
                    self.unused(),
                    coefficient,
                    Fr::zero(),
                    constant,
                ),
            };
            self.gates.push(Gate {
                selectors: [q_l, q_r, -Fr::one(), Fr::zero(), q_c],
                cells: [a, b, output],
                active: true,
            });
            accumulator = Some(output);
        }
        accumulator.unwrap_or_else(|| {
            let output = self.node();
            let unused_a = self.unused();
            let unused_b = self.unused();
            self.gates.push(Gate {
                selectors: [Fr::zero(), Fr::zero(), Fr::one(), Fr::zero(), -constant],
                cells: [unused_a, unused_b, output],
                active: true,
            });
            output
        })
    }

    fn constraint(&mut self, a: &SparseRow<Fr>, b: &SparseRow<Fr>, c: &SparseRow<Fr>) {
        let a = self.linear(a);
        let b = self.linear(b);
        let c = self.linear(c);
        self.gates.push(Gate {
            selectors: [Fr::zero(), Fr::zero(), -Fr::one(), Fr::one(), Fr::zero()],
            cells: [a, b, c],
            active: true,
        });
    }

    fn anchor(&mut self, variable: usize) {
        let unused_b = self.unused();
        let unused_c = self.unused();
        self.gates.push(Gate {
            selectors: [Fr::zero(); 5],
            cells: [Symbol::Variable(variable), unused_b, unused_c],
            active: true,
        });
    }

    fn inactive(&mut self) {
        let cells = std::array::from_fn(|_| self.unused());
        self.gates.push(Gate {
            selectors: [Fr::zero(); 5],
            cells,
            active: false,
        });
    }

    fn zero(&mut self) {
        let a = self.unused();
        let b = self.unused();
        let c = self.unused();
        self.gates.push(Gate {
            selectors: [Fr::one(), Fr::zero(), Fr::zero(), Fr::zero(), Fr::zero()],
            cells: [a, b, c],
            active: true,
        });
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RelationCellLayout {
    pub absorbed_word_base: usize,
    pub absorbed_words: usize,
    pub challenge_base: usize,
    pub challenges: usize,
    pub dory_scalar_base: usize,
    pub dory_scalars: usize,
    pub dory_scalar_capacity: usize,
}

/// Profile-fixed gate selectors and copy permutation.
pub struct RelationTable {
    gates: Vec<Gate>,
    fixed: [Vec<Fr>; FIXED_COLUMNS],
    rows: usize,
    num_vars: usize,
    node_count: usize,
    gate_rows: usize,
    cell_layout: RelationCellLayout,
}

impl RelationTable {
    pub fn new(matrices: &ConstraintMatrices<Fr>, rows: usize) -> Result<Self, RelationTableError> {
        Self::lower(matrices, &[], &[], &[], rows)
    }

    pub fn from_relation(relation: &Relation, rows: usize) -> Result<Self, RelationTableError> {
        let mut absorbed_words = Vec::new();
        let mut challenges = Vec::new();
        for entry in &relation.link.schedule {
            match entry {
                ScheduleEntry::Fr(variable) => absorbed_words.push(variable.index()),
                ScheduleEntry::Squeeze { var, .. } => challenges.push(var.index()),
                ScheduleEntry::Bytes(_) | ScheduleEntry::Opaque { .. } => {}
            }
        }
        let dory_scalars = relation
            .link
            .dory
            .scalars
            .iter()
            .map(|(_, variable)| variable.index())
            .collect::<Vec<_>>();
        Self::lower(
            &relation.matrices,
            &absorbed_words,
            &challenges,
            &dory_scalars,
            rows,
        )
    }

    fn lower(
        matrices: &ConstraintMatrices<Fr>,
        absorbed_words: &[usize],
        challenges: &[usize],
        dory_scalars: &[usize],
        rows: usize,
    ) -> Result<Self, RelationTableError> {
        let mut builder = GateBuilder::new(matrices.num_vars);
        for ((a, b), c) in matrices.a.iter().zip(&matrices.b).zip(&matrices.c) {
            builder.constraint(a, b, c);
        }
        let gate_rows = builder.gates.len();
        let absorbed_word_base = builder.gates.len();
        for &variable in absorbed_words {
            builder.anchor(variable);
        }
        let challenge_base = builder.gates.len();
        for &variable in challenges {
            builder.anchor(variable);
        }
        let dory_scalar_capacity = if dory_scalars.is_empty() {
            0
        } else {
            dory_scalars.len().next_power_of_two()
        };
        let dory_scalar_base = if dory_scalar_capacity == 0 {
            builder.gates.len()
        } else {
            builder.gates.len().next_multiple_of(dory_scalar_capacity)
        };
        while builder.gates.len() < dory_scalar_base {
            builder.inactive();
        }
        for &variable in dory_scalars {
            builder.anchor(variable);
        }
        for _ in dory_scalars.len()..dory_scalar_capacity {
            builder.zero();
        }
        if !rows.is_power_of_two() || rows < builder.gates.len() {
            return Err(RelationTableError::RowDomain {
                minimum: builder.gates.len(),
                actual: rows,
            });
        }
        let used_rows = builder.gates.len();
        let node_count = builder.next_node - matrices.num_vars;
        let mut fixed: [Vec<Fr>; FIXED_COLUMNS] = std::array::from_fn(|_| vec![Fr::zero(); rows]);
        for (row, gate) in builder.gates.iter().enumerate() {
            for (column, &selector) in gate.selectors.iter().enumerate() {
                fixed[column][row] = selector;
            }
            fixed[ACTIVE][row] = Fr::from_u64(u64::from(gate.active));
        }

        let mut occurrences = BTreeMap::<Symbol, Vec<(usize, usize)>>::new();
        for (row, gate) in builder.gates.iter().enumerate() {
            for (wire, &symbol) in gate.cells.iter().enumerate() {
                occurrences.entry(symbol).or_default().push((wire, row));
            }
        }
        for positions in occurrences.values() {
            for (current, next) in positions
                .iter()
                .zip(positions.iter().cycle().skip(1))
                .take(positions.len())
            {
                fixed[SIGMA_A + current.0][current.1] = cell_id(rows, next.0, next.1);
            }
        }
        for (wire, column) in fixed[SIGMA_A..=SIGMA_C].iter_mut().enumerate() {
            for (row, value) in column.iter_mut().enumerate().skip(used_rows) {
                *value = cell_id(rows, wire, row);
            }
        }
        Ok(Self {
            gates: builder.gates,
            fixed,
            rows,
            num_vars: matrices.num_vars,
            node_count,
            gate_rows,
            cell_layout: RelationCellLayout {
                absorbed_word_base,
                absorbed_words: absorbed_words.len(),
                challenge_base,
                challenges: challenges.len(),
                dory_scalar_base,
                dory_scalars: dory_scalars.len(),
                dory_scalar_capacity,
            },
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn gate_rows(&self) -> usize {
        self.gate_rows
    }

    pub fn cell_layout(&self) -> RelationCellLayout {
        self.cell_layout
    }

    pub fn fixed_column_count(&self) -> usize {
        FIXED_COLUMNS
    }

    pub fn witness_column_count(&self) -> usize {
        WITNESS_COLUMNS
    }

    pub fn fixed_columns(&self) -> Vec<Column> {
        self.fixed.iter().cloned().map(Column::Fr).collect()
    }

    fn masked_columns(&self, witness: &RelationTableWitness, part: WitnessPart) -> Vec<Column> {
        let mut columns: Vec<Column> = (0..TOTAL_COLUMNS)
            .map(|_| Column::Fr(vec![Fr::zero(); self.rows]))
            .collect();
        let range = match part {
            WitnessPart::Wires => 0..WIRES,
            WitnessPart::Helpers => WIRES..WITNESS_COLUMNS,
        };
        for index in range {
            columns[FIXED_COLUMNS + index] = Column::Fr(witness.columns[index].clone());
        }
        columns
    }

    fn fixed_masked_columns(&self) -> Vec<Column> {
        let mut columns = self.fixed_columns();
        columns.extend((0..WITNESS_COLUMNS).map(|_| Column::Fr(vec![Fr::zero(); self.rows])));
        columns
    }

    pub fn wire_witness(
        &self,
        assignment: &[Fr],
    ) -> Result<RelationTableWitness, RelationTableError> {
        if assignment.len() < self.num_vars {
            return Err(RelationTableError::Assignment {
                expected: self.num_vars,
                actual: assignment.len(),
            });
        }
        let mut nodes = vec![None; self.node_count];
        let mut columns: [Vec<Fr>; WITNESS_COLUMNS] =
            std::array::from_fn(|_| vec![Fr::zero(); self.rows]);
        for (row, gate) in self.gates.iter().enumerate() {
            let a = symbol_value(gate.cells[0], assignment, &nodes, self.num_vars);
            let b = symbol_value(gate.cells[1], assignment, &nodes, self.num_vars);
            let c = match gate.cells[2] {
                Symbol::Node(index) if nodes[index - self.num_vars].is_none() => {
                    let q = gate.selectors;
                    let value = -(q[Q_L] * a + q[Q_R] * b + q[Q_M] * a * b + q[Q_C])
                        * q[Q_O].inverse().ok_or(RelationTableError::Gate(row))?;
                    nodes[index - self.num_vars] = Some(value);
                    value
                }
                symbol => symbol_value(symbol, assignment, &nodes, self.num_vars),
            };
            columns[0][row] = a;
            columns[1][row] = b;
            columns[2][row] = c;
            if gate_value(gate.selectors, [a, b, c]) != Fr::zero() {
                return Err(RelationTableError::Gate(row));
            }
        }
        Ok(RelationTableWitness { columns })
    }

    pub fn add_copy_helpers(
        &self,
        witness: &mut RelationTableWitness,
        beta: Fr,
        gamma: Fr,
    ) -> Result<(), RelationTableError> {
        let mut denominators = Vec::with_capacity(2 * WIRES * self.gates.len());
        for row in 0..self.gates.len() {
            for wire in 0..WIRES {
                let value = witness.columns[wire][row];
                denominators.push(gamma + value + beta * cell_id(self.rows, wire, row));
            }
            for wire in 0..WIRES {
                let value = witness.columns[wire][row];
                denominators.push(gamma + value + beta * self.fixed[SIGMA_A + wire][row]);
            }
        }
        if denominators.iter().any(Zero::is_zero) {
            return Err(RelationTableError::ZeroDenominator);
        }
        let mut inverses: Vec<ArkFr> = denominators
            .iter()
            .copied()
            .map(ArkFr::from)
            .collect();
        batch_inversion(&mut inverses);
        for (row, chunk) in inverses.chunks_exact(2 * WIRES).enumerate() {
            let active = self.fixed[ACTIVE][row];
            witness.columns[H_ID - FIXED_COLUMNS][row] =
                active * chunk[..WIRES].iter().copied().map(Fr::from).sum::<Fr>();
            witness.columns[H_SIGMA - FIXED_COLUMNS][row] =
                active * chunk[WIRES..].iter().copied().map(Fr::from).sum::<Fr>();
        }
        Ok(())
    }

    pub fn witness(
        &self,
        assignment: &[Fr],
        beta: Fr,
        gamma: Fr,
    ) -> Result<RelationTableWitness, RelationTableError> {
        let mut witness = self.wire_witness(assignment)?;
        self.add_copy_helpers(&mut witness, beta, gamma)?;
        self.check_witness(&witness, beta, gamma)?;
        Ok(witness)
    }

    pub fn check_witness(
        &self,
        witness: &RelationTableWitness,
        beta: Fr,
        gamma: Fr,
    ) -> Result<(), RelationTableError> {
        let mut copy_sum = Fr::zero();
        for row in 0..self.rows {
            let values = std::array::from_fn(|wire| witness.columns[wire][row]);
            if gate_value(
                std::array::from_fn(|column| self.fixed[column][row]),
                values,
            ) != Fr::zero()
            {
                return Err(RelationTableError::Gate(row));
            }
            let ids = std::array::from_fn(|wire| cell_id(self.rows, wire, row));
            let sigmas = std::array::from_fn(|wire| self.fixed[SIGMA_A + wire][row]);
            let h_id = witness.columns[H_ID - FIXED_COLUMNS][row];
            let h_sigma = witness.columns[H_SIGMA - FIXED_COLUMNS][row];
            let active = self.fixed[ACTIVE][row];
            if grouped_relation(values, ids, active, h_id, beta, gamma) != Fr::zero()
                || grouped_relation(values, sigmas, active, h_sigma, beta, gamma) != Fr::zero()
            {
                return Err(RelationTableError::Copy);
            }
            copy_sum += h_id - h_sigma;
        }
        if !copy_sum.is_zero() {
            return Err(RelationTableError::Copy);
        }
        Ok(())
    }

    pub fn final_value(
        rows: usize,
        tau: &[Fr],
        beta: Fr,
        gamma: Fr,
        relation_weights: [Fr; 3],
        point: &[Fr],
        claims: &[Fr],
    ) -> Result<Fr, RelationTableError> {
        Self::final_value_observed(
            rows,
            tau,
            beta,
            gamma,
            relation_weights,
            point,
            claims,
            &mut NoopVerifierObserver,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the transcript challenges and stage-B claim inputs"
    )]
    pub fn final_value_observed<O: VerifierObserver>(
        rows: usize,
        tau: &[Fr],
        beta: Fr,
        gamma: Fr,
        relation_weights: [Fr; 3],
        point: &[Fr],
        claims: &[Fr],
        observer: &mut O,
    ) -> Result<Fr, RelationTableError> {
        if claims.len() != TOTAL_COLUMNS || point.len() != rows.trailing_zeros() as usize {
            return Err(RelationTableError::Claims);
        }
        let fixed: [Fr; FIXED_COLUMNS] = claims[..FIXED_COLUMNS]
            .try_into()
            .map_err(|_| RelationTableError::Claims)?;
        let wires = [claims[WIRE_A], claims[WIRE_B], claims[WIRE_C]];
        let identity = identity_mle_observed(point, observer);
        let ids = std::array::from_fn(|wire| Fr::from_u64(wire as u64 * rows as u64) + identity);
        let sigmas = [fixed[SIGMA_A], fixed[SIGMA_B], fixed[SIGMA_C]];
        let eq = eq_mle_observed(tau, point, observer);
        let selectors = fixed[..5]
            .try_into()
            .map_err(|_| RelationTableError::Claims)?;
        let gate_value = gate_value_observed(selectors, wires, observer);
        let gate = observer.fr_mul(eq, gate_value);
        let id_relation = grouped_relation_observed(
            wires,
            ids,
            fixed[ACTIVE],
            claims[H_ID],
            beta,
            gamma,
            observer,
        );
        let sigma_relation = grouped_relation_observed(
            wires,
            sigmas,
            fixed[ACTIVE],
            claims[H_SIGMA],
            beta,
            gamma,
            observer,
        );
        let eq_id = observer.fr_mul(eq, id_relation);
        let id_term = observer.fr_mul(relation_weights[0], eq_id);
        let eq_sigma = observer.fr_mul(eq, sigma_relation);
        let sigma_term = observer.fr_mul(relation_weights[1], eq_sigma);
        let sum_term = observer.fr_mul(relation_weights[2], claims[H_ID] - claims[H_SIGMA]);
        Ok(gate + id_term + sigma_term + sum_term)
    }
}

enum WitnessPart {
    Wires,
    Helpers,
}

pub struct RelationTableWitness {
    columns: [Vec<Fr>; WITNESS_COLUMNS],
}

impl RelationTableWitness {
    pub fn evaluations(&self) -> &[Vec<Fr>; WITNESS_COLUMNS] {
        &self.columns
    }
}

fn symbol_value(symbol: Symbol, assignment: &[Fr], nodes: &[Option<Fr>], num_vars: usize) -> Fr {
    match symbol {
        Symbol::Variable(index) => assignment[index],
        Symbol::Node(index) => nodes[index - num_vars].unwrap_or_else(Fr::zero),
        Symbol::Unused(_) => Fr::zero(),
    }
}

fn cell_id(rows: usize, wire: usize, row: usize) -> Fr {
    Fr::from_u64((wire * rows + row) as u64)
}

fn identity_mle_observed<O: VerifierObserver>(point: &[Fr], observer: &mut O) -> Fr {
    point
        .iter()
        .enumerate()
        .fold(Fr::zero(), |sum, (index, &coordinate)| {
            sum + observer.fr_mul(coordinate, Fr::from_u64(1 << (point.len() - index - 1)))
        })
}

fn eq_mle_observed<O: VerifierObserver>(left: &[Fr], right: &[Fr], observer: &mut O) -> Fr {
    left.iter()
        .zip(right)
        .fold(Fr::one(), |value, (&left, &right)| {
            let both = observer.fr_mul(left, right);
            let neither = observer.fr_mul(Fr::one() - left, Fr::one() - right);
            observer.fr_mul(value, both + neither)
        })
}

fn gate_value(selectors: [Fr; 5], wires: [Fr; WIRES]) -> Fr {
    gate_value_observed(selectors, wires, &mut NoopVerifierObserver)
}

fn gate_value_observed<O: VerifierObserver>(
    selectors: [Fr; 5],
    wires: [Fr; WIRES],
    observer: &mut O,
) -> Fr {
    let product = observer.fr_mul(wires[0], wires[1]);
    observer.fr_mul(selectors[Q_L], wires[0])
        + observer.fr_mul(selectors[Q_R], wires[1])
        + observer.fr_mul(selectors[Q_O], wires[2])
        + observer.fr_mul(selectors[Q_M], product)
        + selectors[Q_C]
}

fn grouped_relation(
    values: [Fr; WIRES],
    ids: [Fr; WIRES],
    active: Fr,
    helper: Fr,
    beta: Fr,
    gamma: Fr,
) -> Fr {
    grouped_relation_observed(
        values,
        ids,
        active,
        helper,
        beta,
        gamma,
        &mut NoopVerifierObserver,
    )
}

fn grouped_relation_observed<O: VerifierObserver>(
    values: [Fr; WIRES],
    ids: [Fr; WIRES],
    active: Fr,
    helper: Fr,
    beta: Fr,
    gamma: Fr,
    observer: &mut O,
) -> Fr {
    let denominators: [Fr; WIRES] =
        std::array::from_fn(|i| gamma + values[i] + observer.fr_mul(beta, ids[i]));
    let product01 = observer.fr_mul(denominators[0], denominators[1]);
    let product = observer.fr_mul(product01, denominators[2]);
    let elementary = observer.fr_mul(denominators[0], denominators[1])
        + observer.fr_mul(denominators[0], denominators[2])
        + observer.fr_mul(denominators[1], denominators[2]);
    observer.fr_mul(helper, product) - observer.fr_mul(active, elementary)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests fail on invalid evaluator shapes")]
mod cost_tests {
    use jolt_field::{Fr, Ring};

    use super::*;
    use crate::stream::VerifierCost;

    #[test]
    fn final_evaluator_cost_is_row_count_independent() {
        let rows = 1 << 18;
        let tau = (0..18)
            .map(|index| Fr::from_u64((index + 2) as u64))
            .collect::<Vec<_>>();
        let point = (0..18)
            .map(|index| Fr::from_u64((index + 23) as u64))
            .collect::<Vec<_>>();
        let claims = (0..TOTAL_COLUMNS)
            .map(|index| Fr::from_u64((index + 47) as u64))
            .collect::<Vec<_>>();
        let mut cost = VerifierCost::default();
        let _ = RelationTable::final_value_observed(
            rows,
            &tau,
            Fr::from_u64(67),
            Fr::from_u64(71),
            [Fr::from_u64(73), Fr::from_u64(79), Fr::from_u64(83)],
            &point,
            &claims,
            &mut cost,
        )
        .expect("valid evaluator shape");
        assert_eq!(cost.fr_mul, 103);
    }
}
