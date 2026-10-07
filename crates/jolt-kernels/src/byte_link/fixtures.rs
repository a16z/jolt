//! Synthetic byte traces and their honest stage-6b claims for the link's tests
//! and benches.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::lattice::byte_link::{ByteLinkInputs, BYTE_LINK_PACKS, RAM_PACK};
use jolt_claims::protocols::jolt::lattice::{ByteTraceLayoutPlan, OneHotTraceShape};
use jolt_claims::protocols::jolt::JoltCommittedPolynomial as Poly;
use jolt_field::JoltField;
use jolt_openings::EvaluationClaim;
use jolt_poly::EqPolynomial;

use super::reference::ByteTrace;

/// Executed cycles of the 2^29 target trace.
const ACTIVE_AT_29: usize = 402_654_183;

/// The 2^29 target's executed cycles scaled to `2^log_rows`.
pub fn scaled_active(log_rows: usize) -> usize {
    ACTIVE_AT_29.div_ceil(1 << (29 - log_rows))
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn row_hash(seed: u64, column: u64, t: usize) -> u64 {
    let mut state = seed ^ column.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (t as u64).rotate_left(29);
    splitmix(&mut state)
}

/// A byte trace `Q` owned with its layout; every slot is zero from cycle
/// `active` on.
#[derive(Clone)]
pub struct SyntheticTrace {
    pub plan: ByteTraceLayoutPlan,
    pub bytes: Vec<i8>,
    pub active: usize,
}

impl SyntheticTrace {
    /// Uniform bytes below `active` cycles, about a third of them RAM
    /// accesses, or with `hot` 256 tuples per pack; increment digits uniform,
    /// the carry in {-1, 0, 1}, the zero slots and every later cycle zero.
    /// Cycles 0..5 then single out byte extremes, an active RAM access to
    /// address bytes zero, zero non-RAM bytes, and inactive RAM cycles with
    /// zero and nonzero address bytes.
    #[expect(clippy::expect_used, reason = "the 16/2/2 byte geometry is fixed")]
    pub fn new(log_rows: usize, active: usize, seed: u64, hot: bool) -> Self {
        let plan = ByteTraceLayoutPlan::new(&OneHotTraceShape {
            ra_layout: JoltRaPolynomialLayout::new(16, 2, 2).expect("16/2/2 RA layout"),
            log_t: log_rows,
            log_k_chunk: 8,
        })
        .expect("byte trace layout");
        let rows = 1usize << log_rows;
        let mut bytes = vec![0i8; plan.packing().ids().len() * rows];
        let tuples = (0..BYTE_LINK_PACKS.len())
            .map(|pack| {
                (0..256)
                    .map(|k| {
                        let x = row_hash(seed, 2000 + pack as u64, k);
                        if pack == RAM_PACK && !x.is_multiple_of(3) {
                            [0, 0, 0]
                        } else {
                            [
                                x as i8,
                                (x >> 16) as i8,
                                if pack == RAM_PACK { 1 } else { (x >> 32) as i8 },
                            ]
                        }
                    })
                    .collect::<Vec<[i8; 3]>>()
            })
            .collect::<Vec<_>>();
        let ram_active = |t: usize| row_hash(seed, 1000, t).is_multiple_of(3);
        let columns = plan
            .packing()
            .ids()
            .iter()
            .zip(bytes.chunks_exact_mut(rows));
        for (slot, (column, values)) in columns.enumerate() {
            let position = BYTE_LINK_PACKS
                .iter()
                .enumerate()
                .find_map(|(pack, columns)| {
                    columns
                        .iter()
                        .position(|packed| packed == column)
                        .map(|position| (pack, position))
                });
            for (t, value) in values[..active].iter_mut().enumerate() {
                let hash = row_hash(seed, slot as u64, t);
                *value = match (*column, position) {
                    (_, Some((pack, position))) if hot => {
                        tuples[pack][(row_hash(seed, 3000 + pack as u64, t) & 255) as usize]
                            [position]
                    }
                    (Poly::RamActivity, _) => i8::from(ram_active(t)),
                    (Poly::RamRa(_), _) if !ram_active(t) => 0,
                    (Poly::BalancedIncCarry, _) => (hash % 3) as i8 - 1,
                    (Poly::ZeroSlot(_), _) => 0,
                    _ => hash as i8,
                };
            }
        }
        let mut trace = Self {
            plan,
            bytes,
            active,
        };
        trace.set_edge_cycles();
        trace
    }

    fn set_edge_cycles(&mut self) {
        let edges: [(i8, i8, i8); 5] = [
            (-128, -128, 1),
            (127, 127, 1),
            (0, 0, 1),
            (0, 0, 0),
            (-128, 127, 0),
        ];
        for (t, &(byte, ram, activity)) in edges.iter().enumerate() {
            for pack in &BYTE_LINK_PACKS[..RAM_PACK] {
                for column in pack {
                    *self.at_mut(*column, t) = if t == 2 { 0 } else { byte };
                }
            }
            *self.at_mut(Poly::RamRa(0), t) = ram;
            *self.at_mut(Poly::RamRa(1), t) = ram;
            *self.at_mut(Poly::RamActivity, t) = activity;
        }
    }

    pub fn rows(&self) -> usize {
        1 << self.plan.packing().logical_num_vars()
    }

    #[expect(
        clippy::expect_used,
        reason = "every link column has a byte-trace slot"
    )]
    fn slot(&self, column: Poly) -> usize {
        self.plan
            .packing()
            .slot_index(&column)
            .expect("byte-trace slot")
    }

    pub fn at(&self, column: Poly, t: usize) -> i8 {
        self.bytes[self.slot(column) * self.rows() + t]
    }

    pub fn at_mut(&mut self, column: Poly, t: usize) -> &mut i8 {
        let index = self.slot(column) * self.rows() + t;
        &mut self.bytes[index]
    }

    pub fn trace(&self) -> ByteTrace<'_> {
        ByteTrace {
            plan: &self.plan,
            bytes: &self.bytes,
            active_rows: self.active,
        }
    }

    /// The honest stage-6b claims of this trace at a pseudorandom cycle point
    /// and address chunks.
    #[expect(
        clippy::expect_used,
        reason = "the claims are built at one cycle point"
    )]
    pub fn inputs<F: JoltField>(&self, seed: u64) -> ByteLinkInputs<F> {
        let (claims, fused_inc) = self.claims(seed);
        ByteLinkInputs::new(&claims, &fused_inc).expect("honest byte-link claims")
    }

    /// Every one-hot column's inclusive evaluation at `(k_c ‖ r)`, RAM columns
    /// on active cycles only, and the fused increment's evaluation at `r`.
    pub fn claims<F: JoltField>(
        &self,
        seed: u64,
    ) -> (BTreeMap<Poly, EvaluationClaim<F>>, EvaluationClaim<F>) {
        let mut state = seed;
        let mut random = || F::from_u64(splitmix(&mut state));
        let log_rows = self.plan.packing().logical_num_vars();
        let cycle_point = (0..log_rows).map(|_| random()).collect::<Vec<_>>();
        let eq_r = EqPolynomial::<F>::evals(&cycle_point, None);
        let sigma = |byte: i8| F::from_i64(i64::from(byte));
        let mut claims = BTreeMap::new();
        for column in BYTE_LINK_PACKS.iter().flatten() {
            if *column == Poly::RamActivity {
                continue;
            }
            let address = (0..8).map(|_| random()).collect::<Vec<_>>();
            let eq_k = EqPolynomial::<F>::evals(&address, None);
            let ram = matches!(column, Poly::RamRa(_));
            let value = (0..self.rows())
                .filter(|&t| !ram || self.at(Poly::RamActivity, t) != 0)
                .map(|t| eq_r[t] * eq_k[self.at(*column, t) as u8 as usize])
                .sum::<F>();
            let _ = claims.insert(
                *column,
                EvaluationClaim::new([address, cycle_point.clone()].concat(), value),
            );
        }
        let fused = (0..self.rows())
            .map(|t| {
                let increment =
                    (0..8)
                        .rev()
                        .fold(sigma(self.at(Poly::BalancedIncCarry, t)), |acc, digit| {
                            acc * F::from_u64(256)
                                + sigma(self.at(Poly::BalancedIncDigit(digit), t))
                        });
                eq_r[t] * increment
            })
            .sum::<F>();
        (claims, EvaluationClaim::new(cycle_point, fused))
    }
}
