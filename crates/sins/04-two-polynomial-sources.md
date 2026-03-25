# Sin 4: Two Competing Polynomial Data Sources With No Unifying Abstraction

## The Problem

Polynomial evaluation tables come from two places:

```rust
// Committed polynomials (dense tables from witness generation)
committed_store.get(poly_id) -> &[F]

// Virtual polynomials (computed on-the-fly from trace)
trace_polys.materialize(poly_id) -> Vec<F>
```

The prover checks `poly_id.is_committed()` at 3 different sites to decide which source:

```rust
// Site 1: S1 claim evaluations (prover.rs:239)
if poly_id.is_committed() && poly_id != PolynomialId::SpartanWitness {
    witness_builder::eval_poly(committed_store.get(poly_id), point)
} else {
    trace_polys.eval_at_point(poly_id, r_cycle)
}

// Site 2: S2-S7 claim evaluations (prover.rs:323)
if poly_id.is_committed() && poly_id != PolynomialId::SpartanWitness {
    witness_builder::eval_poly(committed_store.get(poly_id), point)
} else {
    trace_polys.eval_at_point(poly_id, &claim_point)
}

// Site 3: build_witness buffer materialization (prover.rs:511)
if poly_id.is_committed() && poly_id != PolynomialId::SpartanWitness {
    committed_store.get(poly_id).to_vec()
} else {
    trace_polys.materialize(poly_id)
}
```

Three copies of the same `if is_committed() && != SpartanWitness` branch.

## Why This Is a Sin

1. **Duplication**: The same routing logic appears 3 times
2. **Fragile exclusion**: `!= SpartanWitness` is a special case because Spartan's witness
   is "committed" but opened separately. If another polynomial gets similar treatment,
   every site needs updating.
3. **Unnecessary clone**: Site 3 does `.to_vec()` on committed tables — a full copy of
   data that the backend will copy again during `upload()`.
4. **Mixed return types**: `&[F]` (borrowed) vs `Vec<F>` (owned) forces callers to handle
   both.

## Proposed Solution: Unified PolynomialSource

### Trait

```rust
/// Unified source of polynomial evaluation tables.
///
/// Encapsulates the committed/virtual distinction so callers never
/// need to check `is_committed()`.
pub trait PolynomialSource<F: Field> {
    /// Get a polynomial's evaluation table as a slice.
    /// For committed polys: returns a reference to the stored table.
    /// For virtual polys: materializes on demand (cached after first call).
    fn get(&self, id: PolynomialId) -> &[F];

    /// Evaluate a polynomial at a point without materializing the full table.
    fn eval_at_point(&self, id: PolynomialId, point: &[F]) -> F;
}
```

### Implementation

```rust
pub struct UnifiedSource<'a, F: Field, R: CycleRow> {
    committed: &'a WitnessStore<F>,
    trace: &'a TracePolynomials<'a, R>,
    /// Cache for materialized virtual tables (populated on first access).
    materialized_cache: RefCell<HashMap<PolynomialId, Vec<F>>>,
}

impl<F: Field, R: CycleRow> PolynomialSource<F> for UnifiedSource<'_, F, R> {
    fn get(&self, id: PolynomialId) -> &[F] {
        if id.is_committed() && id != PolynomialId::SpartanWitness {
            self.committed.get(id)
        } else {
            // Materialize once, cache for reuse
            self.materialized_cache
                .borrow_mut()
                .entry(id)
                .or_insert_with(|| self.trace.materialize(id))
        }
    }

    fn eval_at_point(&self, id: PolynomialId, point: &[F]) -> F {
        if id.is_committed() && id != PolynomialId::SpartanWitness {
            let table = self.committed.get(id);
            eval_poly(table, point)
        } else {
            self.trace.eval_at_point(id, point)
        }
    }
}
```

### What changes in the prover

All 3 branch sites become single calls:

```rust
// Before (3 different sites):
if poly_id.is_committed() && poly_id != PolynomialId::SpartanWitness { ... } else { ... }

// After:
let eval = source.eval_at_point(poly_id, &point);
// or
let table = source.get(poly_id);
```

### Bonus: Backend-Aware Upload

The `UnifiedSource` can also own upload logic:

```rust
impl UnifiedSource {
    /// Upload a polynomial to the backend, with domain padding.
    fn upload_padded<B: ComputeBackend>(
        &self, id: PolynomialId, domain_size: usize, backend: &B
    ) -> B::Buffer<F> {
        let table = self.get(id);
        if table.len() == domain_size {
            backend.upload(table)  // Zero-copy path (no intermediate Vec)
        } else {
            let mut padded = table.to_vec();
            padded.resize(domain_size, F::zero());
            backend.upload(&padded)
        }
    }
}
```

This eliminates the `.to_vec()` + `.resize()` pattern in `build_witness()`.

### What This Eliminates

- 3 copies of `is_committed() && != SpartanWitness` branching
- Unnecessary `.to_vec()` clone on committed tables
- The need for callers to know which source to use for which polynomial
- The SpartanWitness exclusion at every call site (encapsulated once)

## Decisions — Unblocked by Analysis 11

With Spartan dissolved into graph vertices, the blocker is resolved.

- `PolynomialSource` trait unifies committed and virtual polynomial access
- Spartan's Az, Bz, Cz are virtual polynomials materialized by `jolt-witness` (same as trace tables)
- R1CS matrix MLEs are provided by `jolt-r1cs` — just another data source
- `is_committed()` branching eliminated — the source knows how to provide any polynomial by ID
- See [Analysis 11](11-unified-execution-model.md) for the unified model
