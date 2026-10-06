# Multilinear Extensions

A function $f \colon \{0,1\}^v \to \mathbb{F}$ defined on the $v$-dimensional boolean hypercube has a unique **multilinear extension** (MLE) $\tilde{f} \colon \mathbb{F}^v \to \mathbb{F}$: the unique polynomial of degree at most 1 in each variable that agrees with $f$ on all $2^v$ boolean inputs. Concretely,

$$
\tilde{f}(x_1, \dots, x_v) = \sum_{b \in \{0,1\}^v} f(b) \cdot \prod_{i=1}^{v} \bigl(b_i \cdot x_i + (1 - b_i)(1 - x_i)\bigr)
$$

The product term is the **equality polynomial** $\widetilde{\textsf{eq}}(x, b)$, which equals 1 when $x = b$ on the boolean hypercube and interpolates smoothly elsewhere.

MLEs are the central data structure in sumcheck-based proof systems: the [sumcheck protocol](./sumcheck.md) operates over multilinear polynomials, and essentially all witness data in Jolt is represented as MLEs.

For a more formal introduction, see **Section 3.5** of [Proofs, Arguments, and Zero Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf).

## Representation

An MLE over $v$ variables is stored as a vector of $n = 2^v$ **evaluations** over the boolean hypercube $\{0,1\}^v$. The $i$-th entry of this vector stores $f(b)$ where $b$ is the binary representation of $i$. Jolt uses big-endian indexing: the most significant bit of $i$ corresponds to $x_1$.

## Variable binding

Given $\tilde{f}(x_1, \dots, x_v)$ represented by $n = 2^v$ evaluations, **binding** the most significant variable to a challenge $r \in \mathbb{F}$ produces a new MLE $\tilde{f}'(x_2, \dots, x_v) = \tilde{f}(r, x_2, \dots, x_v)$ represented by $n/2$ evaluations. For each index $i \in \{0, \dots, n/2 - 1\}$:

$$
E'[i] = E[i] + r \cdot (E[i + n/2] - E[i])
$$

where $E[i] = \tilde{f}(0, b)$ and $E[i + n/2] = \tilde{f}(1, b)$ for the same suffix $b$. This is a single $O(n)$ pass. Binding the *least* significant variable instead pairs even/odd entries: $E'[i] = E[2i] + r \cdot (E[2i+1] - E[2i])$.

Which variable gets bound in each round is determined by the `BindingOrder`: `HighToLow` (most significant first) or `LowToHigh` (least significant first).

## Implementations in Jolt

The MLE implementations live in `crates/jolt-poly/src/`. `Polynomial<T>` stores a Boolean-hypercube evaluation table, while the `MultilinearEvaluation`, `MultilinearBinding`, and `MultilinearPoly` traits separate point evaluation, sumcheck binding, and PCS access. The same generic type stores both full field elements and compact scalar types.

### Dense polynomials

`Polynomial<F>` (`dense.rs`) is the straightforward representation: a `Vec<F>` of $2^v$ field elements. This is the baseline MLE type used when coefficients are full-sized field elements.

**Binding** operates in-place on the evaluation vector. In `HighToLow` order, the vector is split into left and right halves (corresponding to the most significant variable being 0 or 1) and interpolated:

```rust
let (left, right) = self.Z.split_at_mut(n);
left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
    *a += r * (*b - *a);
});
```

In `LowToHigh` order, adjacent even/odd pairs are interpolated instead:

```rust
for i in 0..n {
    self.Z[i] = self.Z[2 * i] + r * (self.Z[2 * i + 1] - self.Z[2 * i]);
}
```

Both orders have parallel variants (`bind_parallel`) that use Rayon. The `HighToLow` parallel path binds in place, while the `LowToHigh` parallel path cannot.

### Compact polynomials

`Polynomial<T>` stores coefficients as small scalars such as `bool`, `u8`, or `i64` rather than full field elements. This is the "pay-per-bit" representation that makes [Dory](../dory.md) commitments efficient.

Binding promotes the compact evaluation table to field elements:

```rust
// For a pair (a, b) of small scalars:
match a.cmp(&b) {
    Ordering::Equal => a.to_field(),
    Ordering::Less  => a.to_field() + r * (b - a).to_field(),
    Ordering::Greater => a.to_field() - r * (a - b).to_field(),
}
```

Subsequent operations use `Polynomial<F>`. Concrete scalar types remain monomorphized, so small-scalar optimizations apply without an enum dispatch layer.

### Other specialized representations

Several other types in `crates/jolt-poly/src/` exploit common structure:

- **`OneHotPolynomial`** (`one_hot.rs`): Stores the optional nonzero index for each row rather than a dense vector of 0s and 1s. Used for $\widetilde{\textsf{ra}}$ and $\widetilde{\textsf{wa}}$ polynomials in [Twist and Shout](../twist-shout.md).
- **`RlcSource`** (`multilinear.rs`): Represents a lazy random linear combination of multiple PCS sources.
- **`EqPolynomial`** (`eq.rs`): The [equality MLE](./eq-polynomial.md) $\widetilde{\textsf{eq}}(r, \cdot)$, commonly used as a building block in sumcheck and MLE evaluation.
