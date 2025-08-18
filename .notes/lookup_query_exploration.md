# LookupQuery Exploration

This document details the role of `LookupQuery` in the Jolt proving system, focusing on its connection to instruction execution, sparse-dense SHOUT, and prefix-suffix decompositions.

## The `LookupQuery` Trait

The `LookupQuery` is a trait that defines a standard interface for converting the semantics of a RISC-V instruction into a query against a lookup table. It is the bridge between the processor's state and the cryptographic primitives.

```rust
pub trait LookupQuery<const WORD_SIZE: usize> {
    fn to_instruction_inputs(&self) -> (u64, i64);
    fn to_lookup_operands(&self) -> (u64, u64);
    fn to_lookup_index(&self) -> u64;
    fn to_lookup_output(&self) -> u64;
}
```

A `match` statement in `jolt-core/src/jolt/instruction/mod.rs` dispatches calls on the generic `RV32IMCycle` to instruction-specific implementations of this trait.

## Instruction-Specific Implementations

Here's how different instruction types implement the `LookupQuery` trait:

### `ADD` (Arithmetic)

- **Summary**: The `ADD` instruction transforms its operation into a query that asks, "Is the value `rs1 + rs2` a valid `WORD_SIZE`-bit integer?"
- **`to_instruction_inputs`**: Returns the values from `rs1` and `rs2`.
- **`to_lookup_operands`**: Overrides the default to return `(0, rs1 + rs2)`.
- **`to_lookup_index`**: Returns only the sum, `rs1 + rs2`.
- **`lookup_table`**: `RangeCheckTable`. This lookup proves that the addition did not overflow.
- **`to_lookup_output`**: Returns the actual result of the addition, `(rs1 + rs2) % 2^WORD_SIZE`.

### `LW` (Memory Load)

- **Summary**: `LW` (and `SW`) does **not** use the standard lookup mechanism. Its implementation is a no-op.
- **`to_instruction_inputs`**: Returns `(0, 0)`.
- **`lookup_table`**: `None`.
- **Note**: Memory operations are handled by a separate mechanism (the RAM subprotocol, "Twist"), which is activated by the `Load` and `Store` circuit flags.

### `JAL` (Jump and Link)

- **Summary**: `JAL` uses the lookup to verify that its computed target address is valid.
- **`to_instruction_inputs`**: Returns the Program Counter (`PC`) and the immediate value (`imm`).
- **`to_lookup_operands`**: Returns `(0, PC + imm)`.
- **`to_lookup_index`**: Returns the computed jump target address, `PC + imm`.
- **`lookup_table`**: `RangeCheckTable`. This proves the jump address is within a valid range.
- **`to_lookup_output`**: Returns the computed jump target address, which is then used to update the `PC`.

## Synthesis and Next Steps

The pattern is to transform an instruction's semantics into a single `u64` index (`to_lookup_index`), which is then proven to be a member of a specific lookup table.

- For operations on two operands (e.g., `XOR`), `to_lookup_index` uses bit-interleaving to combine two 32-bit operands into one 64-bit index.
- For operations that produce a single result to be checked (e.g., `ADD`, `JAL`), `to_lookup_index` returns that single result.

The next step in the pipeline is to take this `lookup_index` and use it in the **SHOUT** lookup argument. The large size of this index necessitates decomposing it into smaller limbs, which is where **sparse-dense SHOUT** and **prefix-suffix decompositions** come into play.

## The `sparse_dense_shout` Protocol

The `prove_sparse_dense_shout` function in `jolt-core/src/subprotocols/sparse_dense_shout.rs` is the engine that consumes the `lookup_index` values and proves their validity in a batch.

### 1. Index Collection

First, the function iterates through the entire execution trace and collects the `u64` `lookup_index` from every single instruction cycle. This creates a large vector of lookup values that need to be checked.

```rust
let lookup_indices: Vec<_> = trace
    .par_iter()
    .map(|cycle| LookupBits::new(LookupQuery::<WORD_SIZE>::to_lookup_index(cycle), log_K))
    .collect();
```

### 2. Limb Decomposition

A 64-bit lookup index is too large to be used directly. The protocol treats each index `k` as a concatenation of four 16-bit "limbs": `k = (k_1, k_2, k_3, k_4)`. The proof proceeds in 4 phases, where each phase is responsible for binding one of these limbs for all lookups simultaneously.

### 3. Prefix-Suffix Decomposition

This is the core optimization. Instead of working with a massive lookup table polynomial `T(k)`, the protocol decomposes it based on the limb structure. For a `k` split into a prefix `p` and suffix `s`, we have `T(k) = T(p, s)`. The protocol precomputes polynomials for all possible suffixes (`suffix_polys`). During the sumcheck, it combines evaluations from these smaller suffix polynomials with evaluations of dynamically generated prefix polynomials. This avoids ever having to build the full table polynomial `T` in memory.

### 4. `ra` (Read Address) Polynomials

At the end of each of the 4 phases, the prover materializes a polynomial called `ra_i`. The evaluations of `ra_i` correspond to the `i`-th limb of the `lookup_index` for each cycle in the trace. These four `ra` polynomials serve as the "read addresses" into the decomposed lookup table and are the key witnesses used in the final verification stage.

### Conclusion

The `LookupQuery` trait is the foundational layer that translates abstract CPU instructions into concrete numerical values (`lookup_index`). The `sparse_dense_shout` protocol then takes these values and uses a sophisticated, phased sumcheck argument with prefix-suffix optimizations to efficiently prove, in a batch, that every single instruction lookup performed during the program's execution was valid.

## Further Decomposition Examples

Here are a few more examples that illustrate the different decomposition strategies.

### `XOR` (Bitwise Operation)

- **Instruction (`xor.rs`)**: Uses the default `LookupQuery` implementation, meaning `to_lookup_index` calls `interleave_bits(rs1, rs2)` to combine both operands into a single 64-bit index. It points to the `XorTable`.
- **Table (`xor.rs`)**: The `XorTable` is designed to prove `materialize_entry(k) = x ^ y` where `(x, y) = uninterleave(k)`.
- **Decomposition**: It uses `One` and `Xor` suffixes. The `combine` function `prefixes[Prefixes::Xor] * one + xor` demonstrates how the full XOR evaluation is constructed limb by limb during the SHOUT protocol, using a precomputed prefix value combined with the evaluation of the current suffix.

### `SLT` (Comparison Operation)

- **Instruction (`slt.rs`)**: Also uses the default `interleave_bits` on `rs1` and `rs2` to create the lookup index. It points to the `SignedLessThanTable`.
- **Table (`signed_less_than.rs`)**: This table proves that `(x as i32) < (y as i32)`.
- **Decomposition**: Its `combine` function is more complex: `prefixes[LeftOperandMsb] * one - prefixes[RightOperandMsb] * one + prefixes[LessThan] * one + prefixes[Eq] * less_than`. This is a powerful example of decomposition, encoding the logic of a signed comparison by combining prefix evaluations for the sign bits (`Msb`), less-than, and equality on the remaining bits.

### `MUL` (Multiplication)

- **Instruction (`mul.rs`)**: This instruction follows the `ADD` pattern. It computes the product in `to_lookup_operands`, returning `(0, rs1 * rs2)`. The index is therefore just the 64-bit product.
- **Table**: It points to `RangeCheckTable`.
- **Decomposition**: This reveals a key design choice. The lookup does **not** prove the multiplication itself. The R1CS constraint `LeftInstructionInput * RightInstructionInput == Product` handles that. The lookup on the `RangeCheckTable` is only used to prove that the resulting `Product` is a valid 64-bit value.

## Connection to the Prefix-Suffix Inner Product Paper

The SHOUT protocol in Jolt is a direct implementation of the "Prefix-Suffix Inner Product Protocol" described in academic papers. Here’s how the concepts from the paper connect to the code.

### 1. The Core Sumcheck (Equation 14)

- **Paper**: Defines a sumcheck over an inner product: \(\sum_{x \in \{0,1\}^{\log N}} \tilde{u}(x) \cdot \tilde{a}(x)\).
- **Code**: This is the conceptual goal of `prove_sparse_dense_shout`.
    - `u_tilde(x)` is the **prover's witness**: a polynomial that specifies which lookup is performed at which cycle. This is built from `eq_r_prime_evals` (for the cycle) and the `ra` polynomials (for the lookup value).
    - `a_tilde(x)` is the **multilinear extension (MLE) of the lookup table**. For example, for `XorTable`, `a_tilde` is the polynomial that computes the XOR of the uninterleaved input bits.

The protocol proves this inner product, ensuring that every lookup the program performs is valid according to the table.

### 2. The Prefix-Suffix Structure (Definition A.1 & Equation 15)

- **Paper**: States that `a_tilde(x)` must be decomposable into a sum of prefix/suffix products:

$$\tilde{a}(x_1, \dots, x_{\log N}) = \sum_{j=1}^{k} \text{prefix}_j(x_1, \dots, x_i) \cdot \text{suffix}_j(x_{i+1}, \dots, x_{\log N})$$

- **Code**: This is precisely the `combine` function in every `JoltLookupTable`. The `combine` function *is* the code's implementation of this definition.

For example, the `SignedLessThanTable`'s `combine` function is a sum of 4 prefix-suffix products, directly matching the formula with `k=4`.

```rust
// A direct implementation of Equation (15) from the paper
fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
    let [one, less_than] = suffixes.try_into().unwrap();
    
    // Term 1: prefix_1 * suffix_1
    prefixes[Prefixes::LeftOperandMsb] * one 
    // Term 2: prefix_2 * suffix_2
    - prefixes[Prefixes::RightOperandMsb] * one 
    // Term 3: prefix_3 * suffix_3
    + prefixes[Prefixes::LessThan] * one 
    // Term 4: prefix_4 * suffix_4
    + prefixes[Prefixes::Eq] * less_than
}
```

### 3. Passes and Cutoffs (C and i)

- **Paper**: Describes an algorithm using `C` passes with different prefix/suffix "cutoffs" `i`.
- **Code**: This is the `for phase in 0..4` loop in `prove_sparse_dense_shout`.
    - The number of passes `C` is hardcoded to `4`.
    - The cutoff `i` changes in each phase by recalculating `suffix_len`. In each phase, a 16-bit limb of the lookup index is processed, growing the prefix and shrinking the suffix. This phased approach is the paper's multi-pass algorithm in action.

## Dissecting a Prefix: `LeftShift`

To make this more concrete, let's analyze a specific prefix implementation: `LeftShiftPrefix`.

- **Goal**: The comments state that this prefix computes `x << s`, where `s` is the number of **leading ones** in `y`. This is used to implement shift instructions like `SLL`.
- **Polynomial Formula**: The `update_prefix_checkpoint` function reveals the mathematical formula being constructed. At each step `i` of the sumcheck (for bits `x_i` and `y_i`), it adds a new term to the total, building a polynomial that approximates:
$$ \text{LeftShift}(x, y) \approx \sum_{i=0}^{W-1} x_i \cdot (\dots \text{logic involving } y_i \dots) \cdot (\text{term depending on } y_{0..i-1}) \cdot 2^{W-1-i} $$
The formula is complex because it must be expressed in a way that can be built up incrementally as the sumcheck protocol provides more randomness.
- **Key Idea**: The prefix uses a helper prefix, `LeftShiftHelper`, which computes \(\prod_{k=0}^{i-1} (1+y_k)\). This helper carries information about the preceding bits of `y` that is needed to compute the logic for the current bit `i`.
- **`prefix_mle` Function**: This function evaluates the partially-built polynomial during the sumcheck. It has two parts: one that uses the random challenges for bits that have already been bound, and another that computes the contribution of the remaining, unbound bits by performing a direct shift calculation on them.

This example shows that each "prefix" is not just a simple value but a potentially complex polynomial itself, designed to be built and evaluated efficiently within the iterative structure of the sumcheck protocol. 