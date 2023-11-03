# Jolt todos
*We could track this in issues, but will be confusing for on-lookers. Items here should either be sorted before merge into `master` or moved to GH issues.*
- Multi-subtable: Fix `dim` / `read` / `final` size increase from `C` -> `num_memories` 
- `nz` is currently a `C`-dimensional set of `s`-sized vectors. Should be swapped to an instruction strategy. Fixing this should allow cleaning `memory_to_dimension_index` + `to_lookup_polys` + `Densified::from_...`
- `JoltStrategy::prove` as new entry point
```
    let proof = JoltStrategyImpl::prove(instructions: Vec<JoltStrategyImpl::Instruction>);
    proof.verify();
```
- `combine_lookups` / `combine_lookups_eq` – Currently hacky, can this be better?
- Derive `JoltStrategy::primary_poly_degree` rather than hardcode.
- Correct indexing for variable `memory_size` / `dimension` Subtables
- Fix all the tests / benches
- Move some values from dynamic functions to constants and enforce with test?
```
trait JoltStrategy {
    const NUM_MEMORIES: usize;
    const NUM_SUBTABLES: usize;
}
```
- Macros (TODO: Describe)
- `LassoStrategy` – Potentially can have single high level trait defining
    - `primary_poly_degree`
    - `combine_lookups`
    - `to_lookup_polys`
    - `materialize_subtables`
    - `evaluate_memory_mle`
    - `num_instructions`
    - `num_subtables` (potentially unused now?)
Then `JoltStrategy` extends and adds default impls for some and `instructions()` + `type Instruction`.
Note: This may not work due to necessity of Lasso code handling specific instructions.

