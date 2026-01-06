//! Branching Program Optimization for Jagged Polynomial Verifier
//!
//! This module implements a width-4 read-once branching program (ROBP) for efficiently
//! computing the multilinear extension of the function g(a, b, c, d) = 1 iff b < d AND b = a + c.
//!
//! The branching program reads the bits of a, b, c, d from LSB to MSB and maintains:
//! - A carry bit from the addition check (b = a + c)
//! - A comparison bit tracking whether b < d so far
//!
//! This optimization reduces the verifier's computation from O(2^n) to O(n) for n-bit inputs.

use crate::field::JoltField;
use ark_ff::{One, Zero};
use std::cmp::max;

/// Memory state of the branching program (2 bits = 4 possible states)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryState {
    /// Carry bit from the addition a + c
    pub carry: bool,
    /// Whether b < d so far in the comparison
    pub comparison_so_far: bool,
}

impl MemoryState {
    /// Initial state at the beginning of computation
    pub fn initial() -> Self {
        MemoryState {
            carry: false,
            comparison_so_far: false,
        }
    }

    /// Success state: no carry and b < d
    pub fn success() -> Self {
        MemoryState {
            carry: false,
            comparison_so_far: true,
        }
    }

    /// Convert state to index for array lookups
    pub fn to_index(&self) -> usize {
        (self.carry as usize) + ((self.comparison_so_far as usize) << 1)
    }

    /// Convert index to state
    pub fn from_index(idx: usize) -> Self {
        MemoryState {
            carry: (idx & 1) != 0,
            comparison_so_far: ((idx >> 1) & 1) != 0,
        }
    }
}

/// Represents either a valid state or a failure in the computation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateOrFail {
    State(MemoryState),
    Fail,
}

/// The four bits read at each layer of the branching program
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitState<T> {
    pub a_bit: T,
    pub b_bit: T,
    pub c_bit: T,
    pub d_bit: T,
}

/// Transition function: given current bits and state, compute next state
///
/// Checks two conditions:
/// 1. b_bit must equal (a_bit + c_bit + carry) mod 2 (addition check)
/// 2. Updates comparison based on b_bit vs d_bit
pub fn transition_function(bits: BitState<bool>, state: MemoryState) -> StateOrFail {
    // Check addition constraint: b = a + c (with carry)
    let sum = bits.a_bit as u8 + bits.c_bit as u8 + state.carry as u8;
    let expected_b = (sum & 1) != 0;
    let new_carry = sum >= 2;

    if bits.b_bit != expected_b {
        return StateOrFail::Fail;
    }

    // Update comparison b < d
    let new_comparison = if bits.b_bit != bits.d_bit {
        // If bits differ, d_bit must be 1 and b_bit must be 0 for b < d
        bits.d_bit && !bits.b_bit
    } else {
        // If bits are equal, maintain previous comparison
        state.comparison_so_far
    };

    StateOrFail::State(MemoryState {
        carry: new_carry,
        comparison_so_far: new_comparison,
    })
}

/// A point in the multilinear polynomial space
#[derive(Clone, Debug)]
pub struct Point<F> {
    coords: Vec<F>,
}

impl<F: JoltField> Point<F> {
    /// Create a point from a vector of field elements
    pub fn from(coords: Vec<F>) -> Self {
        Point { coords }
    }

    /// Create a point from a slice
    pub fn from_slice(slice: &[F]) -> Self {
        Point {
            coords: slice.to_vec(),
        }
    }

    /// Create a point representing a usize value in binary (LSB first)
    pub fn from_usize(val: usize, num_bits: usize) -> Self {
        let coords = (0..num_bits)
            .map(|i| {
                if (val >> i) & 1 == 1 {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect();
        Point { coords }
    }

    /// Get the dimension (number of coordinates)
    pub fn dimension(&self) -> usize {
        self.coords.len()
    }

    /// Get coordinate at index (LSB first ordering)
    pub fn get(&self, idx: usize) -> Option<&F> {
        self.coords.get(idx)
    }

    /// Get coordinate at index, returning zero if out of bounds
    pub fn get_or_zero(&self, idx: usize) -> F {
        self.coords.get(idx).cloned().unwrap_or_else(F::zero)
    }
}

/// Main branching program structure for computing g and its multilinear extension
pub struct JaggedBranchingProgram {
    num_bits: usize,
}

impl JaggedBranchingProgram {
    /// Create a new branching program for n-bit inputs
    pub fn new(num_bits: usize) -> Self {
        JaggedBranchingProgram { num_bits }
    }

    /// Evaluate g(a, b, c, d) on boolean inputs
    pub fn eval_boolean(&self, a: &[bool], b: &[bool], c: &[bool], d: &[bool]) -> bool {
        let mut state = MemoryState::initial();

        // Process bits from LSB to MSB
        for i in 0..self.num_bits {
            let bits = BitState {
                a_bit: a.get(i).copied().unwrap_or(false),
                b_bit: b.get(i).copied().unwrap_or(false),
                c_bit: c.get(i).copied().unwrap_or(false),
                d_bit: d.get(i).copied().unwrap_or(false),
            };

            match transition_function(bits, state) {
                StateOrFail::State(new_state) => state = new_state,
                StateOrFail::Fail => return false,
            }
        }

        // Accept if we end in success state (no carry, b < d)
        state == MemoryState::success()
    }

    /// Compute the multilinear extension ĝ(za, zb, zc, zd) efficiently
    ///
    /// Uses dynamic programming to compute the MLE in O(num_bits) time
    /// by working backwards from the success state
    pub fn eval_multilinear<F: JoltField>(
        &self,
        za: &Point<F>,
        zb: &Point<F>,
        zc: &Point<F>,
        zd: &Point<F>,
    ) -> F {
        // Ensure we have enough bits
        let max_dim = max(
            max(za.dimension(), zb.dimension()),
            max(zc.dimension(), zd.dimension()),
        );
        let num_bits = max(self.num_bits, max_dim);

        // state_values[i] = value starting from state i
        let mut state_values = [F::zero(); 4];
        state_values[MemoryState::success().to_index()] = F::one();

        // Process layers from MSB to LSB (backwards)
        for layer in (0..num_bits).rev() {
            let mut new_state_values = [F::zero(); 4];

            // Precompute equality polynomial values for this layer
            let eq_values = self.compute_eq_values_for_layer(za, zb, zc, zd, layer);

            // For each possible starting state
            for state_idx in 0..4 {
                let state = MemoryState::from_index(state_idx);

                // Sum over all possible bit combinations
                for bit_idx in 0..16 {
                    let bits = self.index_to_bits(bit_idx);

                    // Get next state
                    if let StateOrFail::State(next_state) = transition_function(bits, state) {
                        // Contribution = eq(bits, z_values) * value_from_next_state
                        let contribution = eq_values[bit_idx] * state_values[next_state.to_index()];
                        new_state_values[state_idx] += contribution;
                    }
                    // If transition fails, no contribution
                }
            }

            state_values = new_state_values;
        }

        // Return value starting from initial state
        state_values[MemoryState::initial().to_index()]
    }

    /// Compute equality polynomial values for all 16 bit combinations at a given layer
    fn compute_eq_values_for_layer<F: JoltField>(
        &self,
        za: &Point<F>,
        zb: &Point<F>,
        zc: &Point<F>,
        zd: &Point<F>,
        layer: usize,
    ) -> [F; 16] {
        let za_val = za.get_or_zero(layer);
        let zb_val = zb.get_or_zero(layer);
        let zc_val = zc.get_or_zero(layer);
        let zd_val = zd.get_or_zero(layer);

        let mut eq_values = [F::zero(); 16];

        for i in 0..16 {
            let bits = self.index_to_bits(i);

            // eq(bits, z) = ∏ eq_i(bit_i, z_i)
            let mut val = F::one();

            // eq(a_bit, za)
            val *= if bits.a_bit { za_val } else { F::one() - za_val };

            // eq(b_bit, zb)
            val *= if bits.b_bit { zb_val } else { F::one() - zb_val };

            // eq(c_bit, zc)
            val *= if bits.c_bit { zc_val } else { F::one() - zc_val };

            // eq(d_bit, zd)
            val *= if bits.d_bit { zd_val } else { F::one() - zd_val };

            eq_values[i] = val;
        }

        eq_values
    }

    /// Convert a 4-bit index to BitState
    fn index_to_bits(&self, idx: usize) -> BitState<bool> {
        BitState {
            a_bit: (idx & 1) != 0,
            b_bit: ((idx >> 1) & 1) != 0,
            c_bit: ((idx >> 2) & 1) != 0,
            d_bit: ((idx >> 3) & 1) != 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;
    use ark_ff::UniformRand;

    fn bool_vec(val: usize, len: usize) -> Vec<bool> {
        (0..len).map(|i| ((val >> i) & 1) != 0).collect()
    }

    #[test]
    fn test_memory_state_conversions() {
        for i in 0..4 {
            let state = MemoryState::from_index(i);
            assert_eq!(state.to_index(), i);
        }
    }

    #[test]
    fn test_boolean_evaluation() {
        let prog = JaggedBranchingProgram::new(4);

        // Test: g(2, 5, 3, 8) = true because 5 < 8 AND 5 = 2 + 3
        assert!(prog.eval_boolean(&bool_vec(2, 4), &bool_vec(5, 4), &bool_vec(3, 4), &bool_vec(8, 4)));

        // Test: g(2, 6, 3, 8) = false because 6 ≠ 2 + 3 (should be 5)
        assert!(!prog.eval_boolean(&bool_vec(2, 4), &bool_vec(6, 4), &bool_vec(3, 4), &bool_vec(8, 4)));

        // Test: g(2, 5, 3, 4) = false because 5 ≮ 4
        assert!(!prog.eval_boolean(&bool_vec(2, 4), &bool_vec(5, 4), &bool_vec(3, 4), &bool_vec(4, 4)));

        // Test: g(1, 3, 2, 10) = true because 3 < 10 AND 3 = 1 + 2
        assert!(prog.eval_boolean(&bool_vec(1, 4), &bool_vec(3, 4), &bool_vec(2, 4), &bool_vec(10, 4)));

        // Edge case: g(0, 0, 0, 1) = true because 0 < 1 AND 0 = 0 + 0
        assert!(prog.eval_boolean(&bool_vec(0, 4), &bool_vec(0, 4), &bool_vec(0, 4), &bool_vec(1, 4)));
    }

    #[test]
    fn test_multilinear_consistency() {
        let prog = JaggedBranchingProgram::new(3);
        let mut rng = ark_std::test_rng();

        // Random evaluation points
        let za = Point::from(vec![Fq::rand(&mut rng), Fq::rand(&mut rng), Fq::rand(&mut rng)]);
        let zb = Point::from(vec![Fq::rand(&mut rng), Fq::rand(&mut rng), Fq::rand(&mut rng)]);
        let zc = Point::from(vec![Fq::rand(&mut rng), Fq::rand(&mut rng), Fq::rand(&mut rng)]);
        let zd = Point::from(vec![Fq::rand(&mut rng), Fq::rand(&mut rng), Fq::rand(&mut rng)]);

        // Compute MLE using branching program
        let mle_result = prog.eval_multilinear(&za, &zb, &zc, &zd);

        // Compute MLE naively by summing over hypercube
        let mut naive_sum = Fq::zero();
        for a_val in 0..8 {
            for b_val in 0..8 {
                for c_val in 0..8 {
                    for d_val in 0..8 {
                        // Check if g(a,b,c,d) = 1
                        if prog.eval_boolean(&bool_vec(a_val, 3), &bool_vec(b_val, 3),
                                           &bool_vec(c_val, 3), &bool_vec(d_val, 3)) {
                            // Add eq(a, za) * eq(b, zb) * eq(c, zc) * eq(d, zd)
                            let mut term = Fq::one();

                            for i in 0..3 {
                                let a_bit = ((a_val >> i) & 1) != 0;
                                let b_bit = ((b_val >> i) & 1) != 0;
                                let c_bit = ((c_val >> i) & 1) != 0;
                                let d_bit = ((d_val >> i) & 1) != 0;

                                let za_i = za.get(i).unwrap();
                                let zb_i = zb.get(i).unwrap();
                                let zc_i = zc.get(i).unwrap();
                                let zd_i = zd.get(i).unwrap();

                                term *= if a_bit { *za_i } else { Fq::one() - za_i };
                                term *= if b_bit { *zb_i } else { Fq::one() - zb_i };
                                term *= if c_bit { *zc_i } else { Fq::one() - zc_i };
                                term *= if d_bit { *zd_i } else { Fq::one() - zd_i };
                            }

                            naive_sum += term;
                        }
                    }
                }
            }
        }

        assert_eq!(mle_result, naive_sum, "MLE computation mismatch");
    }

    #[test]
    fn test_point_creation() {
        let p1 = Point::<Fq>::from_usize(5, 4);
        assert_eq!(p1.dimension(), 4);
        assert_eq!(p1.get(0).unwrap(), &Fq::one());  // bit 0 = 1
        assert_eq!(p1.get(1).unwrap(), &Fq::zero()); // bit 1 = 0
        assert_eq!(p1.get(2).unwrap(), &Fq::one());  // bit 2 = 1
        assert_eq!(p1.get(3).unwrap(), &Fq::zero()); // bit 3 = 0

        let p2 = Point::from(vec![Fq::one(), Fq::zero()]);
        assert_eq!(p2.dimension(), 2);
        assert_eq!(p2.get_or_zero(0), Fq::one());
        assert_eq!(p2.get_or_zero(1), Fq::zero());
        assert_eq!(p2.get_or_zero(2), Fq::zero()); // Out of bounds
    }
}