# Target Architecture

**Last updated**: 2026-04-13

## One-sentence summary

The compiler lowers protocol knowledge into a generic schedule of ops;
the runtime executes that schedule without knowing what protocol it represents.

## The Problem

Three subprotocol state machines (PrefixSuffix, Booleanity, HwReduction) leak
protocol-specific names and types through four abstraction layers:

| Layer | Current | Target |
|-------|---------|--------|
| Op enum (compiler) | 12 protocol-named variants | 4 generic: `InstanceInit/Bind/Reduce/Finalize` |
| ComputeBackend trait | 3 associated types + 12 methods | 1 type + 4 methods |
| RuntimeState | 3 separate HashMaps | 1 `instance_states` map |
| CpuBackend | 12 method impls | 4 method impls (internal enum dispatch) |

Plus 2 config types (`BooleanityConfig`, `HwReductionConfig`) that force the
runtime to import protocol-specific types.

## Target Types

### InstanceConfig (compiler → runtime → backend)

Lives in `jolt-compiler/src/module.rs`. Describes what data to gather and which
subprotocol the backend should instantiate. The runtime passes it through
without inspecting it.

```rust
/// Configuration for a stateful sumcheck instance.
///
/// Each variant describes a different algorithm, but the runtime doesn't
/// branch on it — it passes the config to the backend's instance_init().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstanceConfig {
    PrefixSuffix {
        kernel: usize,
    },
    Booleanity {
        ra_poly_ids: Vec<PolynomialId>,
        addr_challenges: Vec<usize>,
        cycle_challenges: Vec<usize>,
        gamma_powers: Vec<usize>,
        gamma_powers_square: Vec<usize>,
        log_k_chunk: usize,
        log_t: usize,
    },
    HwReduction {
        ra_poly_ids: Vec<PolynomialId>,
        cycle_challenges_be: Vec<usize>,
        addr_bool_challenges_be: Vec<usize>,
        addr_virt_challenges_be: Vec<Vec<usize>>,
        gamma_powers: Vec<usize>,
        hw_eval_challenge: usize,
        instruction_d: usize,
        bytecode_d: usize,
        ram_d: usize,
        log_k_chunk: usize,
        log_t: usize,
    },
}
```

> Note: The variants carry protocol names, but this is fine — the enum is
> defined in the compiler (which IS protocol-aware). The key invariant is
> that the runtime never matches on these variants.

### Generic Op variants (replace 12 with 4)

```rust
// These REPLACE:
//   PrefixSuffixInit/Bind/Reduce/Materialize
//   BooleanityInit/Bind/Reduce/CacheOpenings
//   HwReductionInit/Bind/Reduce/CacheOpenings

Op::InstanceInit {
    batch: usize,
    instance: usize,
    config: InstanceConfig,
},
Op::InstanceBind {
    batch: usize,
    instance: usize,
    challenge: usize,
},
Op::InstanceReduce {
    batch: usize,
    instance: usize,
},
Op::InstanceFinalize {
    batch: usize,
    instance: usize,
    /// Buffer outputs to insert into device_buffers
    output_buffers: Vec<PolynomialId>,
    /// Evaluation outputs to insert into evaluations cache
    output_evals: Vec<PolynomialId>,
},
```

### ComputeBackend (collapse 3+12 → 1+4)

```rust
pub trait ComputeBackend: Send + Sync + 'static {
    type Buffer<T: Scalar>: Send + Sync;
    type CompiledKernel<F: Field>: Send + Sync;
    type InstanceState<F: Field>: Send + Sync;  // NEW (replaces 3 types)

    // Existing generic methods (unchanged):
    fn compile<F: Field>(&self, spec: &KernelSpec) -> Self::CompiledKernel<F>;
    fn reduce<F: Field>(...) -> Vec<F>;
    fn bind<F: Field>(...);
    fn interpolate_inplace<F: Field>(...);
    fn upload/download/alloc/len/eq_table/lt_table/...  // buffer ops unchanged

    // NEW (replaces 12 protocol-specific methods):
    fn instance_init<F: Field>(
        &self,
        config: &InstanceConfig,
        challenges: &[F],
        provider: &mut impl BufferProvider<F>,
        lookup_trace: Option<&LookupTraceData>,
        kernels: &[KernelDef],       // for PS iteration
    ) -> Self::InstanceState<F>;

    fn instance_bind<F: Field>(
        &self,
        state: &mut Self::InstanceState<F>,
        challenge: F,
    );

    fn instance_reduce<F: Field>(
        &self,
        state: &Self::InstanceState<F>,
        previous_claim: F,
    ) -> Vec<F>;

    fn instance_finalize<F: Field>(
        &self,
        state: Self::InstanceState<F>,
    ) -> InstanceOutput<Self::Buffer<F>, F>;
}

/// Output from a stateful instance finalization.
pub struct InstanceOutput<Buf, F> {
    /// Polynomial buffers to insert into device_buffers (e.g., PS outputs).
    pub buffers: Vec<(PolynomialId, Buf)>,
    /// Scalar evaluations to insert into evaluations cache (e.g., bool/HW claims).
    pub evaluations: Vec<(PolynomialId, F)>,
}
```

### CpuBackend (internal dispatch)

```rust
// In jolt-cpu/src/backend.rs:
pub enum CpuInstanceState<F: Field> {
    PrefixSuffix(CpuPrefixSuffixState<F>),
    Booleanity(CpuBooleanityState<F>),
    HwReduction(CpuHwReductionState<F>),
}

impl ComputeBackend for CpuBackend {
    type InstanceState<F: Field> = CpuInstanceState<F>;

    fn instance_init<F: Field>(...) -> CpuInstanceState<F> {
        match config {
            InstanceConfig::PrefixSuffix { kernel } => {
                CpuInstanceState::PrefixSuffix(self.ps_init_internal(...))
            }
            InstanceConfig::Booleanity { .. } => {
                CpuInstanceState::Booleanity(self.bool_init_internal(...))
            }
            InstanceConfig::HwReduction { .. } => {
                CpuInstanceState::HwReduction(self.hw_init_internal(...))
            }
        }
    }

    fn instance_bind<F: Field>(&self, state: &mut CpuInstanceState<F>, challenge: F) {
        match state {
            CpuInstanceState::PrefixSuffix(s) => self.ps_bind_internal(s, challenge),
            CpuInstanceState::Booleanity(s) => self.bool_bind_internal(s, challenge),
            CpuInstanceState::HwReduction(s) => self.hw_bind_internal(s, challenge),
        }
    }
    // ... same pattern for reduce/finalize
}
```

The protocol-specific implementations (ps_init_internal, etc.) remain as
private methods or in submodules. The math doesn't change — only the dispatch surface.

### RuntimeState (collapse 3 → 1)

```rust
struct RuntimeState<B: ComputeBackend, F: Field, PCS: ...> {
    // ... all existing fields EXCEPT:
    // DELETE: prefix_suffix_states, booleanity_states, hw_reduction_states

    // REPLACE WITH:
    instance_states: HashMap<(usize, usize), B::InstanceState<F>>,
}
```

### Runtime dispatch (12 match arms → 4)

```rust
Op::InstanceInit { batch, instance, config } => {
    let s = backend.instance_init(config, &state.challenges, provider, ...);
    state.instance_states.insert((*batch, *instance), s);
}
Op::InstanceBind { batch, instance, challenge } => {
    let s = state.instance_states.get_mut(&(*batch, *instance)).unwrap();
    backend.instance_bind(s, state.challenges[*challenge]);
}
Op::InstanceReduce { batch, instance } => {
    let s = state.instance_states.get(&(*batch, *instance)).unwrap();
    let claim = state.batch_instance_claims[*batch][*instance];
    state.last_round_instance_evals[*instance] = backend.instance_reduce(s, claim);
}
Op::InstanceFinalize { batch, instance, output_buffers, output_evals } => {
    let s = state.instance_states.remove(&(*batch, *instance)).unwrap();
    let out = backend.instance_finalize(s);
    for (pid, buf) in out.buffers { device_buffers.insert(pid, DeviceBuffer::Field(buf)); }
    for (pid, val) in out.evaluations { state.evaluations.insert(pid, val); }
}
```

## What changes, what doesn't

### Changes
- `Op` enum: -12 variants, +4 generic
- `ComputeBackend` trait: -3 types, -12 methods, +1 type, +4 methods
- `RuntimeState`: -3 fields, +1 field
- `runtime.rs` dispatch: -~250 LOC of protocol-specific matching
- `jolt-compiler/emit.rs`: emit `InstanceInit` instead of `PrefixSuffixInit` etc.
- `BooleanityConfig`, `HwReductionConfig`: absorbed into `InstanceConfig` variants
- `jolt-metal/backend.rs`: same pattern as CPU (if Metal implements these)

### Doesn't change
- The actual math (prefix_suffix.rs, booleanity.rs, hw_reduction.rs internals)
- Transcript bytes (same values flow in same order)
- KernelSpec, Formula, Iteration (these are already generic)
- Buffer management, PCS ops, orchestration ops
- Test assertions or proof structure

## Verification

After migration, these greps should return 0 hits in the runtime:
```bash
grep -c "PrefixSuffix\|Booleanity\|HwReduction" crates/jolt-zkvm/src/runtime.rs
grep -c "ps_init\|ps_bind\|ps_reduce\|ps_materialize" crates/jolt-compute/src/traits.rs
grep -c "bool_init\|bool_bind\|bool_reduce\|bool_final" crates/jolt-compute/src/traits.rs
grep -c "hw_init\|hw_bind\|hw_reduce\|hw_final" crates/jolt-compute/src/traits.rs
```

And transcript parity + cross-system verification must pass:
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
cargo nextest run -p jolt-equivalence zkvm_proof_accepted --cargo-quiet
```
