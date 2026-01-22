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
//!
//! ## Jagged Assist Support
//!
//! This module also provides utilities for the Jagged Assist optimization (Lemma 4.6),
//! which enables efficient streaming evaluation of the ROBP's MLE during sumcheck.
//! Key utilities:
//! - `CoordType` and `get_coordinate_info`: Map sumcheck variables to (a,b,c,d) coordinates
//! - `transition_mle`: Compute the MLE of the transition matrix
//! - `precompute_backward`: Precompute backward vectors for forward-backward decomposition
//! - `update_forward`: Update forward state as sumcheck variables are bound

use crate::field::JoltField;
use std::cmp::max;

/// Identifies which coordinate (a, b, c, d) a sumcheck variable belongs to.
///
/// In the Jagged Assist sumcheck, variables use **interleaved ordering** to align
/// with the ROBP layer structure:
/// - Variables 0,1,2,3 → (a₀, b₀, c₀, d₀) = ROBP layer 0
/// - Variables 4,5,6,7 → (a₁, b₁, c₁, d₁) = ROBP layer 1
/// - etc.
///
/// This enables efficient forward-backward decomposition where we can update
/// the forward state after every 4 sumcheck rounds (one complete ROBP layer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordType {
    A,
    B,
    C,
    D,
}

/// Maps a sumcheck variable index to its coordinate type and ROBP layer (bit position).
///
/// Uses **interleaved ordering**: `(a₀,b₀,c₀,d₀, a₁,b₁,c₁,d₁, ...)`
///
/// This aligns with the ROBP structure where each layer processes one bit from
/// each of (a, b, c, d) simultaneously.
///
/// # Arguments
/// * `var_idx` - The sumcheck variable index (0 to 4*num_bits - 1)
/// * `_num_bits` - Number of bits per coordinate (unused but kept for API consistency)
///
/// # Returns
/// A tuple of (CoordType, layer_index) where layer_index is the ROBP layer (bit position)
#[inline]
pub fn get_coordinate_info(var_idx: usize, _num_bits: usize) -> (CoordType, usize) {
    let layer = var_idx / 4;
    let coord = var_idx % 4;
    let coord_type = match coord {
        0 => CoordType::A,
        1 => CoordType::B,
        2 => CoordType::C,
        _ => CoordType::D,
    };
    (coord_type, layer)
}

/// Compute eq(bit, z) = z if bit else (1 - z)
#[inline]
pub fn eq_bit<F: JoltField>(bit: bool, z: F) -> F {
    if bit {
        z
    } else {
        F::one() - z
    }
}

/// Convert a single bit (0 or 1) to a field element
#[inline]
pub fn bit_to_field<F: JoltField>(bit: usize) -> F {
    if bit == 1 {
        F::one()
    } else {
        F::zero()
    }
}

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
            val *= if bits.a_bit {
                za_val
            } else {
                F::one() - za_val
            };

            // eq(b_bit, zb)
            val *= if bits.b_bit {
                zb_val
            } else {
                F::one() - zb_val
            };

            // eq(c_bit, zc)
            val *= if bits.c_bit {
                zc_val
            } else {
                F::one() - zc_val
            };

            // eq(d_bit, zd)
            val *= if bits.d_bit {
                zd_val
            } else {
                F::one() - zd_val
            };

            eq_values[i] = val;
        }

        eq_values
    }

    /// Convert a 4-bit index to BitState
    pub fn index_to_bits(&self, idx: usize) -> BitState<bool> {
        BitState {
            a_bit: (idx & 1) != 0,
            b_bit: ((idx >> 1) & 1) != 0,
            c_bit: ((idx >> 2) & 1) != 0,
            d_bit: ((idx >> 3) & 1) != 0,
        }
    }

    // =========================================================================
    // Jagged Assist Support (Lemma 4.6 - Forward-Backward Decomposition)
    // =========================================================================

    /// Compute the MLE of the transition matrix T(za, zb, zc, zd) for a given starting state.
    ///
    /// Returns an array where `result[s']` is the "probability" of transitioning from
    /// `from_state` to state `s'` given the MLE-extended inputs `(za, zb, zc, zd)`.
    ///
    /// This is computed as:
    /// ```text
    /// T[from_state, to_state] = Σ_{bits ∈ {0,1}^4} eq(bits, z) · 1[transition(from_state, bits) = to_state]
    /// ```
    pub fn transition_mle<F: JoltField>(
        &self,
        za: F,
        zb: F,
        zc: F,
        zd: F,
        from_state: MemoryState,
    ) -> [F; 4] {
        let mut result = [F::zero(); 4];

        for bits_idx in 0..16 {
            let bits = self.index_to_bits(bits_idx);

            if let StateOrFail::State(to_state) = transition_function(bits, from_state) {
                // eq((a_bit, b_bit, c_bit, d_bit), (za, zb, zc, zd))
                let eq_val = eq_bit(bits.a_bit, za)
                    * eq_bit(bits.b_bit, zb)
                    * eq_bit(bits.c_bit, zc)
                    * eq_bit(bits.d_bit, zd);

                result[to_state.to_index()] += eq_val;
            }
            // Transitions to Fail state contribute nothing
        }

        result
    }

    /// Precompute backward vectors for forward-backward decomposition.
    ///
    /// For each ROBP layer `i`, `backward[i][s]` represents the "probability" of
    /// reaching the accept state from state `s`, given the suffix of the input
    /// from layer `i` onwards.
    ///
    /// The ROBP processes one bit from each of (a, b, c, d) at each layer,
    /// so there are `num_bits` layers total.
    ///
    /// # Arguments
    /// * `r_x` - The `a` coordinate values (field elements from constraint challenge)
    /// * `r_dense` - The `b` coordinate values (field elements from dense challenge)
    /// * `t_prev` - The `c` coordinate as an integer (t_{y-1}, converted to bits)
    /// * `t_curr` - The `d` coordinate as an integer (t_y, converted to bits)
    ///
    /// # Returns
    /// A vector of length `num_bits + 1`, where `backward[layer][state]` gives
    /// the backward value for that (layer, state) pair.
    pub fn precompute_backward<F: JoltField>(
        &self,
        r_x: &[F],
        r_dense: &[F],
        t_prev: usize,
        t_curr: usize,
    ) -> Vec<[F; 4]> {
        let num_layers = self.num_bits;
        let mut backward = vec![[F::zero(); 4]; num_layers + 1];

        // Initialize final layer: only accept state gets value 1
        backward[num_layers][MemoryState::success().to_index()] = F::one();

        // Work backwards through ROBP layers (MSB to LSB)
        for layer in (0..num_layers).rev() {
            // Get the coordinate values for this layer
            let za = r_x.get(layer).cloned().unwrap_or(F::zero());
            let zb = r_dense.get(layer).cloned().unwrap_or(F::zero());
            let zc = bit_to_field::<F>((t_prev >> layer) & 1);
            let zd = bit_to_field::<F>((t_curr >> layer) & 1);

            // backward[layer][s] = Σ_{s'} T(z)[s, s'] · backward[layer+1][s']
            for s in 0..4 {
                let transitions = self.transition_mle(za, zb, zc, zd, MemoryState::from_index(s));
                backward[layer][s] = (0..4)
                    .map(|s_prime| transitions[s_prime] * backward[layer + 1][s_prime])
                    .sum();
            }
        }

        backward
    }

    /// Update forward state after processing one ROBP layer.
    ///
    /// Given the current forward state and the coordinate values for this layer,
    /// computes the new forward state after the transition.
    ///
    /// # Arguments
    /// * `forward` - Current forward state: `forward[s]` = prob of being in state `s`
    /// * `za`, `zb`, `zc`, `zd` - Coordinate values for this layer
    ///
    /// # Returns
    /// New forward state after processing this layer
    pub fn update_forward<F: JoltField>(
        &self,
        forward: &[F; 4],
        za: F,
        zb: F,
        zc: F,
        zd: F,
    ) -> [F; 4] {
        let mut new_forward = [F::zero(); 4];

        for s in 0..4 {
            if forward[s].is_zero() {
                continue;
            }

            let transitions = self.transition_mle(za, zb, zc, zd, MemoryState::from_index(s));
            for s_prime in 0..4 {
                new_forward[s_prime] += forward[s] * transitions[s_prime];
            }
        }

        new_forward
    }

    /// Compute ĝ(z) using the forward-backward decomposition at a specific layer.
    ///
    /// Given forward state up to layer `layer`, and backward vectors from layer `layer+1`,
    /// compute the full MLE value by contracting through the transition at `layer`.
    ///
    /// # Arguments
    /// * `forward` - Forward state up to (but not including) this layer
    /// * `backward` - Backward vectors (precomputed)
    /// * `layer` - The layer to contract through
    /// * `za`, `zb`, `zc`, `zd` - Coordinate values for this layer
    ///
    /// # Returns
    /// The MLE value ĝ(prefix, z_layer, suffix)
    pub fn compute_mle_via_forward_backward<F: JoltField>(
        &self,
        forward: &[F; 4],
        backward: &[[F; 4]],
        layer: usize,
        za: F,
        zb: F,
        zc: F,
        zd: F,
    ) -> F {
        let mut result = F::zero();

        for s in 0..4 {
            if forward[s].is_zero() {
                continue;
            }

            let transitions = self.transition_mle(za, zb, zc, zd, MemoryState::from_index(s));
            for s_prime in 0..4 {
                result += forward[s] * transitions[s_prime] * backward[layer + 1][s_prime];
            }
        }

        result
    }

    /// Get the number of bits this branching program operates on
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Get the initial forward state (all mass in initial state)
    pub fn initial_forward_state<F: JoltField>() -> [F; 4] {
        let mut forward = [F::zero(); 4];
        forward[MemoryState::initial().to_index()] = F::one();
        forward
    }

    /// Compute the full 4×4 transition matrix T[from_state, to_state] at once.
    ///
    /// This is more efficient than calling transition_mle 4 times because we
    /// iterate over the 16 bit patterns only once.
    ///
    /// Returns T as a [[F; 4]; 4] where T[s][s'] = probability of s -> s' transition.
    #[inline]
    pub fn transition_matrix_full<F: JoltField>(&self, za: F, zb: F, zc: F, zd: F) -> [[F; 4]; 4] {
        let mut t_matrix = [[F::zero(); 4]; 4];

        // Precompute eq values for each bit
        let eq_a = [eq_bit(false, za), eq_bit(true, za)];
        let eq_b = [eq_bit(false, zb), eq_bit(true, zb)];
        let eq_c = [eq_bit(false, zc), eq_bit(true, zc)];
        let eq_d = [eq_bit(false, zd), eq_bit(true, zd)];

        for bits_idx in 0..16 {
            let bits = self.index_to_bits(bits_idx);

            // Compute eq value using precomputed values
            let eq_val = eq_a[bits.a_bit as usize]
                * eq_b[bits.b_bit as usize]
                * eq_c[bits.c_bit as usize]
                * eq_d[bits.d_bit as usize];

            // For each source state, compute the transition
            for s in 0..4 {
                let from_state = MemoryState::from_index(s);
                if let StateOrFail::State(to_state) = transition_function(bits, from_state) {
                    t_matrix[s][to_state.to_index()] += eq_val;
                }
            }
        }

        t_matrix
    }

    /// Apply a precomputed transition matrix to compute F · T · B.
    ///
    /// This computes: Σ_s Σ_s' forward[s] * T[s][s'] * backward[s']
    #[inline]
    pub fn apply_transition_matrix<F: JoltField>(
        forward: &[F; 4],
        t_matrix: &[[F; 4]; 4],
        backward: &[F; 4],
    ) -> F {
        let mut result = F::zero();

        for s in 0..4 {
            if forward[s].is_zero() {
                continue;
            }
            for s_prime in 0..4 {
                result += forward[s] * t_matrix[s][s_prime] * backward[s_prime];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;
    use ark_ff::UniformRand;
    use ark_ff::{One, Zero};

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
        assert!(prog.eval_boolean(
            &bool_vec(2, 4),
            &bool_vec(5, 4),
            &bool_vec(3, 4),
            &bool_vec(8, 4)
        ));

        // Test: g(2, 6, 3, 8) = false because 6 ≠ 2 + 3 (should be 5)
        assert!(!prog.eval_boolean(
            &bool_vec(2, 4),
            &bool_vec(6, 4),
            &bool_vec(3, 4),
            &bool_vec(8, 4)
        ));

        // Test: g(2, 5, 3, 4) = false because 5 ≮ 4
        assert!(!prog.eval_boolean(
            &bool_vec(2, 4),
            &bool_vec(5, 4),
            &bool_vec(3, 4),
            &bool_vec(4, 4)
        ));

        // Test: g(1, 3, 2, 10) = true because 3 < 10 AND 3 = 1 + 2
        assert!(prog.eval_boolean(
            &bool_vec(1, 4),
            &bool_vec(3, 4),
            &bool_vec(2, 4),
            &bool_vec(10, 4)
        ));

        // Edge case: g(0, 0, 0, 1) = true because 0 < 1 AND 0 = 0 + 0
        assert!(prog.eval_boolean(
            &bool_vec(0, 4),
            &bool_vec(0, 4),
            &bool_vec(0, 4),
            &bool_vec(1, 4)
        ));
    }

    #[test]
    fn test_multilinear_consistency() {
        let prog = JaggedBranchingProgram::new(3);
        let mut rng = ark_std::test_rng();

        // Random evaluation points
        let za = Point::from(vec![
            Fq::rand(&mut rng),
            Fq::rand(&mut rng),
            Fq::rand(&mut rng),
        ]);
        let zb = Point::from(vec![
            Fq::rand(&mut rng),
            Fq::rand(&mut rng),
            Fq::rand(&mut rng),
        ]);
        let zc = Point::from(vec![
            Fq::rand(&mut rng),
            Fq::rand(&mut rng),
            Fq::rand(&mut rng),
        ]);
        let zd = Point::from(vec![
            Fq::rand(&mut rng),
            Fq::rand(&mut rng),
            Fq::rand(&mut rng),
        ]);

        // Compute MLE using branching program
        let mle_result = prog.eval_multilinear(&za, &zb, &zc, &zd);

        // Compute MLE naively by summing over hypercube
        let mut naive_sum = Fq::zero();
        for a_val in 0..8 {
            for b_val in 0..8 {
                for c_val in 0..8 {
                    for d_val in 0..8 {
                        // Check if g(a,b,c,d) = 1
                        if prog.eval_boolean(
                            &bool_vec(a_val, 3),
                            &bool_vec(b_val, 3),
                            &bool_vec(c_val, 3),
                            &bool_vec(d_val, 3),
                        ) {
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
        assert_eq!(p1.get(0).unwrap(), &Fq::one()); // bit 0 = 1
        assert_eq!(p1.get(1).unwrap(), &Fq::zero()); // bit 1 = 0
        assert_eq!(p1.get(2).unwrap(), &Fq::one()); // bit 2 = 1
        assert_eq!(p1.get(3).unwrap(), &Fq::zero()); // bit 3 = 0

        let p2 = Point::from(vec![Fq::one(), Fq::zero()]);
        assert_eq!(p2.dimension(), 2);
        assert_eq!(p2.get_or_zero(0), Fq::one());
        assert_eq!(p2.get_or_zero(1), Fq::zero());
        assert_eq!(p2.get_or_zero(2), Fq::zero()); // Out of bounds
    }

    // =========================================================================
    // Tests for Jagged Assist utilities
    // =========================================================================

    #[test]
    fn test_get_coordinate_info() {
        let num_bits = 4;

        // Interleaved ordering: (a₀,b₀,c₀,d₀, a₁,b₁,c₁,d₁, ...)
        // Layer 0: variables 0,1,2,3
        assert_eq!(get_coordinate_info(0, num_bits), (CoordType::A, 0)); // a₀
        assert_eq!(get_coordinate_info(1, num_bits), (CoordType::B, 0)); // b₀
        assert_eq!(get_coordinate_info(2, num_bits), (CoordType::C, 0)); // c₀
        assert_eq!(get_coordinate_info(3, num_bits), (CoordType::D, 0)); // d₀

        // Layer 1: variables 4,5,6,7
        assert_eq!(get_coordinate_info(4, num_bits), (CoordType::A, 1)); // a₁
        assert_eq!(get_coordinate_info(5, num_bits), (CoordType::B, 1)); // b₁
        assert_eq!(get_coordinate_info(6, num_bits), (CoordType::C, 1)); // c₁
        assert_eq!(get_coordinate_info(7, num_bits), (CoordType::D, 1)); // d₁

        // Layer 2: variables 8,9,10,11
        assert_eq!(get_coordinate_info(8, num_bits), (CoordType::A, 2)); // a₂
        assert_eq!(get_coordinate_info(9, num_bits), (CoordType::B, 2)); // b₂
        assert_eq!(get_coordinate_info(10, num_bits), (CoordType::C, 2)); // c₂
        assert_eq!(get_coordinate_info(11, num_bits), (CoordType::D, 2)); // d₂

        // Layer 3: variables 12,13,14,15
        assert_eq!(get_coordinate_info(12, num_bits), (CoordType::A, 3)); // a₃
        assert_eq!(get_coordinate_info(13, num_bits), (CoordType::B, 3)); // b₃
        assert_eq!(get_coordinate_info(14, num_bits), (CoordType::C, 3)); // c₃
        assert_eq!(get_coordinate_info(15, num_bits), (CoordType::D, 3)); // d₃
    }

    #[test]
    fn test_eq_bit() {
        let z = Fq::from(7u64);

        // eq(0, z) = 1 - z
        assert_eq!(eq_bit(false, z), Fq::one() - z);

        // eq(1, z) = z
        assert_eq!(eq_bit(true, z), z);

        // Edge cases
        assert_eq!(eq_bit(false, Fq::zero()), Fq::one());
        assert_eq!(eq_bit(true, Fq::zero()), Fq::zero());
        assert_eq!(eq_bit(false, Fq::one()), Fq::zero());
        assert_eq!(eq_bit(true, Fq::one()), Fq::one());
    }

    #[test]
    fn test_bit_to_field() {
        assert_eq!(bit_to_field::<Fq>(0), Fq::zero());
        assert_eq!(bit_to_field::<Fq>(1), Fq::one());
    }

    #[test]
    fn test_transition_mle_on_boolean_inputs() {
        let prog = JaggedBranchingProgram::new(4);

        // On boolean inputs, transition_mle should give 1 for valid transitions, 0 otherwise
        let initial = MemoryState::initial();

        // Test with boolean inputs (0, 0, 0, 0)
        let transitions =
            prog.transition_mle(Fq::zero(), Fq::zero(), Fq::zero(), Fq::zero(), initial);

        // From initial state with bits (0,0,0,0):
        // sum = 0 + 0 + 0 = 0, expected_b = 0, new_carry = false
        // b_bit (0) == expected_b (0), so valid
        // b_bit == d_bit, so comparison stays false
        // New state: (carry=false, comparison=false) = initial state
        assert_eq!(transitions[MemoryState::initial().to_index()], Fq::one());
        assert_eq!(transitions[MemoryState::success().to_index()], Fq::zero());

        // Test with boolean inputs (0, 0, 0, 1) - d_bit = 1
        let transitions =
            prog.transition_mle(Fq::zero(), Fq::zero(), Fq::zero(), Fq::one(), initial);

        // b_bit (0) != d_bit (1), so comparison becomes (d_bit == 1) = true
        // New state: (carry=false, comparison=true) = success state
        assert_eq!(transitions[MemoryState::success().to_index()], Fq::one());
    }

    #[test]
    fn test_forward_backward_matches_eval_multilinear() {
        let prog = JaggedBranchingProgram::new(3);
        let mut rng = ark_std::test_rng();

        // Generate random evaluation point
        let r_x: Vec<Fq> = (0..3).map(|_| Fq::rand(&mut rng)).collect();
        let r_dense: Vec<Fq> = (0..3).map(|_| Fq::rand(&mut rng)).collect();
        let t_prev = 2usize; // Some cumulative height
        let t_curr = 5usize; // Next cumulative height

        // Compute using eval_multilinear
        let za = Point::from(r_x.clone());
        let zb = Point::from(r_dense.clone());
        let zc = Point::from_usize(t_prev, 3);
        let zd = Point::from_usize(t_curr, 3);
        let expected = prog.eval_multilinear(&za, &zb, &zc, &zd);

        // Compute using forward-backward decomposition
        let backward = prog.precompute_backward(&r_x, &r_dense, t_prev, t_curr);

        // Start with initial forward state
        let mut forward = JaggedBranchingProgram::initial_forward_state::<Fq>();

        // Process all layers
        for layer in 0..3 {
            let za_layer = r_x[layer];
            let zb_layer = r_dense[layer];
            let zc_layer = bit_to_field((t_prev >> layer) & 1);
            let zd_layer = bit_to_field((t_curr >> layer) & 1);

            forward = prog.update_forward(&forward, za_layer, zb_layer, zc_layer, zd_layer);
        }

        // Final result: forward · accept_indicator
        // backward[num_bits] has accept state = 1, others = 0
        let result: Fq = (0..4).map(|s| forward[s] * backward[3][s]).sum();

        assert_eq!(result, expected, "Forward-backward decomposition mismatch");
    }

    #[test]
    fn test_compute_mle_via_forward_backward() {
        let prog = JaggedBranchingProgram::new(3);
        let mut rng = ark_std::test_rng();

        // Generate random evaluation point
        let r_x: Vec<Fq> = (0..3).map(|_| Fq::rand(&mut rng)).collect();
        let r_dense: Vec<Fq> = (0..3).map(|_| Fq::rand(&mut rng)).collect();
        let t_prev = 1usize;
        let t_curr = 6usize;

        // Compute expected result
        let za = Point::from(r_x.clone());
        let zb = Point::from(r_dense.clone());
        let zc = Point::from_usize(t_prev, 3);
        let zd = Point::from_usize(t_curr, 3);
        let expected = prog.eval_multilinear(&za, &zb, &zc, &zd);

        // Precompute backward
        let backward = prog.precompute_backward(&r_x, &r_dense, t_prev, t_curr);

        // Compute via forward-backward at each layer and verify consistency
        let mut forward = JaggedBranchingProgram::initial_forward_state::<Fq>();

        for layer in 0..3 {
            let za_layer = r_x[layer];
            let zb_layer = r_dense[layer];
            let zc_layer = bit_to_field((t_prev >> layer) & 1);
            let zd_layer = bit_to_field((t_curr >> layer) & 1);

            // Compute MLE at this split point
            let mle_at_layer = prog.compute_mle_via_forward_backward(
                &forward, &backward, layer, za_layer, zb_layer, zc_layer, zd_layer,
            );

            // This should equal the full MLE (since we're using the actual coordinates)
            assert_eq!(
                mle_at_layer, expected,
                "Forward-backward at layer {} mismatch",
                layer
            );

            // Update forward for next iteration
            forward = prog.update_forward(&forward, za_layer, zb_layer, zc_layer, zd_layer);
        }
    }

    #[test]
    fn test_transition_mle_sums_to_one_or_less() {
        use rand::Rng;

        let prog = JaggedBranchingProgram::new(4);
        let mut rng = ark_std::test_rng();

        // On boolean inputs, transitions from any state should sum to at most 1
        // (exactly 1 if no failure, 0 if failure)
        for _ in 0..10 {
            let za = if rng.gen_bool(0.5) {
                Fq::one()
            } else {
                Fq::zero()
            };
            let zb = if rng.gen_bool(0.5) {
                Fq::one()
            } else {
                Fq::zero()
            };
            let zc = if rng.gen_bool(0.5) {
                Fq::one()
            } else {
                Fq::zero()
            };
            let zd = if rng.gen_bool(0.5) {
                Fq::one()
            } else {
                Fq::zero()
            };

            for s in 0..4 {
                let state = MemoryState::from_index(s);
                let transitions = prog.transition_mle(za, zb, zc, zd, state);
                let sum: Fq = transitions.iter().sum();

                // Sum should be 0 or 1 on boolean inputs
                assert!(
                    sum == Fq::zero() || sum == Fq::one(),
                    "Transition sum should be 0 or 1 on boolean inputs, got {:?}",
                    sum
                );
            }
        }
    }
}
