//! Backend-agnostic AST emission trait.
//!
//! [`AstEmitter`] is the pluggable codegen interface: gnark, Spartan+HyperKZG,
//! and future backends each implement this trait to produce their native
//! constraint representation from an [`AstBundle`](crate::bundle::AstBundle).

/// Trait for emitting constraints from an AST node graph.
///
/// Each method corresponds to an operation recorded in the arena. The `Wire`
/// associated type is the backend's handle to a value — a Go expression string
/// for gnark, an R1CS variable for Spartan, etc.
///
/// Implementations accumulate side effects (code generation, constraint
/// emission) in `&mut self`.
pub trait AstEmitter {
    /// A handle to a value in the target representation.
    type Wire;

    /// Emit a constant field element (BN254 limbs, little-endian).
    fn constant(&mut self, val: [u64; 4]) -> Self::Wire;

    /// Emit a named input variable.
    fn variable(&mut self, index: u32, name: &str) -> Self::Wire;

    /// Emit negation: `-inner`.
    fn neg(&mut self, inner: Self::Wire) -> Self::Wire;

    /// Emit multiplicative inverse: `inner^{-1}`.
    fn inv(&mut self, inner: Self::Wire) -> Self::Wire;

    /// Emit addition: `lhs + rhs`.
    fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;

    /// Emit subtraction: `lhs - rhs`.
    fn sub(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;

    /// Emit multiplication: `lhs * rhs`.
    fn mul(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;

    /// Emit division: `lhs / rhs`.
    fn div(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;

    /// Emit a Poseidon hash: `Poseidon(state, n_rounds, data)`.
    fn poseidon(&mut self, state: Self::Wire, n_rounds: Self::Wire, data: Self::Wire)
        -> Self::Wire;

    /// Emit byte reversal (LE ↔ BE) of a field element.
    fn byte_reverse(&mut self, inner: Self::Wire) -> Self::Wire;

    /// Emit truncation to low 128 bits.
    fn truncate_128(&mut self, inner: Self::Wire) -> Self::Wire;

    /// Emit multiplication by 2^192.
    fn mul_two_pow_192(&mut self, inner: Self::Wire) -> Self::Wire;

    /// Emit an assertion that `expr == 0`.
    fn assert_zero(&mut self, expr: Self::Wire);

    /// Emit an assertion that `lhs == rhs`.
    fn assert_equal(&mut self, lhs: Self::Wire, rhs: Self::Wire);
}
