# Whiteout verifier: public state built once (O3)

Parent: `specs/whiteout-optimization-campaign.md` (O3). Answers
`specs/whiteout-o0-profile.md`.

O3 changes no message, check or transcript operation. It keeps the
public state that a Whiteout verify rebuilt on every call, and it stops
transforming matrices that the verifier only reads over `F`.
`fs_obligations` passes without a bless.

## What is cached, and what keys it

Every cache below holds a deterministic function of public inputs. Its key
is the whole input, not a digest or a projection of it. Nothing derived
from proof bytes (`cm(u_a)`, the pad commitment's payload, u_a, any
challenge) is held across verifies. Those values enter each verify's Σ
setup as targets.

| state | held by | key | built from |
|---|---|---|---|
| `A`, `[B \| B′]` (`ResidualMatrices<D>`) | `WhiteoutRow::matrices` | the row (one cell per row) | the row's `ResidualShape` and the catalog's Akita setup stream |
| `A_p` and `L` (`LatticePadCommitment<D>`) | `WhiteoutTail::layouts` | (row, whole `PadLayout`) | the layout's `PadPlacement`, its Σ-binding rank, `PAD_STREAM_SEED` |
| `A_D` | `WhiteoutTail::layouts` | (row, whole `PadLayout`) | the shapes of the row's blocks and of `A_p`, the digit rank, `V_STREAM_SEED` |
| the response prover of `e`'s length | `WhiteoutTail::layouts`, then `ResponsePlans` by `num_vars` | (row, whole `PadLayout`) | `SigmaSetup::response_layout` over the setup above, then the setup's response plan for that length |

**Why each key is lossless.**

- *Row.* A `WhiteoutTail` owns its `WhiteoutRow`, and its methods read no
  other row, so the type system pairs each layout cache with one row.
  The setup's `PlannedSetup` builds one shape per admitted list of
  precommitted groups. The row's matrix cell is likewise a field of the
  row.
  - `ResidualShape::blocks` checks in release builds that the matrices it
    receives have this shape's dimensions.
  - Shape equality determines the matrices only because a catalog has one
    setup stream (`WhiteoutCatalog::stream`, Akita's expanded shared
    matrix), and every matrix is an offset-0 prefix of it (`MatrixView`).
    Two shapes with equal dimensions over different streams would pass the
    check. They cannot meet here: the row and its catalog are fixed
    together at construction.
  - `ResidualMatrices` has private fields, so `ResidualShape::matrices` is
    its only constructor. A caller cannot hand `blocks` matrices cut from
    anywhere else.
- *Layout.* The cache compares `PadLayout` with `==` on all of it:
  `num_pads`, the auxiliary range, and every operand pad. The pad
  commitment is `LatticePadCommitment::new(PadPlacement::new(layout), …)`
  at a rank and width priced from the placement, cut from a constant seed.
  `A_D`'s width is a function of the blocks' row counts and `A_p`'s. Its
  rank is a function of that width. `e`'s layout reads the segment table,
  the Σ parameters, `V`'s ring count and each compressed target's image
  length (`rows · D`). It never reads a target's payload. So the Σ setup
  over empty payloads sizes `e` exactly as a proof's does. That is why
  `build_layout_state` builds that setup once, instead of a second
  derivation of the length.
- *Ring dimension.* The cells are type-erased (`Arc<dyn Any + Send + Sync>`)
  because `D` is a const parameter chosen at runtime by
  `with_ring_dim!`. A row has one ring dimension, and a mismatched
  downcast is an `AkitaError::Internal`, never a reinterpretation.

**Growth.** Each tail keeps at most `MAX_CACHED_LAYOUTS` = 8 layouts. A
small bound cannot come from the setup alone: the row fixes the trace and
precommitted groups, but the layout also counts stages 1–7's masked
messages. Those depend on dimensions each proof's header carries (trace
length, RAM size, read-write and one-hot configuration). A verifier fed
headers that vary would otherwise grow the cache by one pad commitment and
`A_D` per header. Past the cap, a layout's state is built for its verify
and dropped. That is correct, only slower.

**Concurrency.** Each state is built outside its lock or cell: the builds
run on Rayon, and a worker waiting inside `get_or_init` could steal a job
that waits on it (the deadlock `PreparedMatrix` already documents).
Concurrent first callers may each build, and the first one stored is kept.
Failed builds are not stored, so a persisted response set that lacks a
length is refused again on every verify (D32's behaviour).

**Prover.** `commit_pads`, `compression` and `prove` read the same cells,
so the prover also builds each layout's state once. The state is public.
`prove_sigma` still runs inside `with_dit`, and the secrets it applies
the matrices to are unchanged.

## The NTT round trip

`PreparedMatrix` now keeps its entries over `F`, copied from the stream.
It builds the CRT+NTT form on its first constant-time product
(`apply_centered`, `apply_transposed`) and keeps it. The verifier reads
`A_a`, `[B_a | B′_a]`, `A_p`, `L` and `A_D` only through
`combine_columns`, so it never transforms them. Before O3 it converted
each matrix into CRT+NTT form and straight back
(`whiteout_nz_field_entries`, which this ticket removes from the
taxonomy).

A prover that applies a matrix holds both forms. That adds 16 B per
coefficient on top of the 24 B of the CRT+NTT form, for these public
matrices only. It is a one-time cost per matrix per process.

`apply_i32` takes the NTT entries before its parallel column loop. Its
workers therefore share one built form instead of racing to build it.

## Not cached

- **The pad-layout replay** (`jolt_verifier::whiteout::pad_layout`) reruns
  stages 1–7 and the masked prefix over the proof's seeded prefix. That
  prefix includes the proof's commitments. The replay is a function of the
  statement's dimensions in practice, but no key short of the seeded prefix
  is shown to determine it. A key on the full prefix would never hit across
  proofs. Removing it needs a statement-level derivation of the layout,
  which is a protocol-ownership change. O0 put it at 1% (one thread) to 4%
  (16 threads).
- **The per-verify Σ setup** (`whiteout_sigma_setup`): the segment table,
  the lifted parameters and the blocks with this proof's targets. It now
  clones `Arc`s and checks shapes.

## Results (LOADED, n = 3 per cell)

> **LOADED: not for absolute claims.** The 1-minute load was 60–150
> throughout. Before and after alternated within each round, and the one
> that went first alternated between rounds. Cells show the median and
> [min–max] verify time in ms, from O0's harness.

The binaries were built from the same tree, before and after the O3
commit, with `CARGO_PROFILE_RELEASE_DEBUG=0` and `--features
profiling,akita,zk`. The parsing is `scripts/whiteout_verify_profile.py`'s
`split` and `report`. The driver ran before/after instead of clear/Whiteout.

| workload | mode | before | after | ratio |
|---|---|---:|---:|---:|
| muldiv 2^16 | 16 threads | 1,112.7 [1,002.7–1,303.8] | 318.3 [240.0–467.1] | 3.5× |
| muldiv 2^16 | 1 thread | 3,937.6 [3,699.6–4,470.6] | 304.8 [238.5–447.4] | 12.9× |
| muldiv 2^16 | first verify, transported | 1,927.5 [1,404.5–2,355.0] | 586.2 [566.8–775.2] | 3.3× |
| fib 2^16 | 16 threads | 1,580.8 [963.9–2,759.8] | 394.6 [313.0–718.4] | 4.0× |
| fib 2^16 | 1 thread | 3,014.3 [2,877.3–3,915.1] | 297.1 [289.1–634.4] | 10.1× |
| fib 2^16 | first verify, transported | 2,990.8 [1,197.7–3,317.4] | 582.1 [537.6–773.7] | 5.1× |
| fib 2^20 | 16 threads | 1,898.5 [1,079.7–2,792.4] | 534.5 [425.9–588.6] | 3.6× |
| fib 2^20 | 1 thread | 3,658.6 [3,470.6–9,963.6] | 492.4 [417.9–589.8] | 7.4× |
| fib 2^20 | first verify, transported | 1,752.2 [1,212.1–3,240.4] | 748.2 [543.7–862.8] | 2.3× |

**Where a steady verify goes after O3** (1 thread, median share): the
response proof is 75–85%. Within it, `N·z` is 31–57%, the lifted rows
17–25%, and the eq dual 3–4%. The pad-layout replay is 7–9%.

The report's "public-state builds, any context" bucket sums
`whiteout_pad_commitment_setup`, `whiteout_residual_matrices`,
`whiteout_v_matrix_setup` and the pre-O3 `whiteout_nz_field_entries`,
wherever they nest. In a steady verify after O3 it is 0.0 ms in every cell,
so the replay is all that is recomputed. Before O3 it was 49–81%. On a
transported first verify, the one-time builds are 11–23%. They run under
`whiteout_response_lookup`, which is 11–24%.

**Thread scaling.** Under this load, 16 threads are no faster than 1. The
`N·z` products spend 3–6× more CPU with 16 threads than the 1-thread wall
time (A_a: 309–1,180 ms CPU against 62–206 ms). The eq dual runs 3–8×
slower with 16 threads than with 1. Both are fork-join overhead on a
saturated machine, as O0 found. O3 adds no parallelism. Whether the
remaining loops (`N·z`, lifted rows) gain from more, or from less nesting,
can be decided only on an idle machine.
