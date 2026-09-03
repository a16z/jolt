# Metal bench & attribution rigs

The intentional measurement tools retained from the Metal saturation
campaign (`.journals/metal-saturation.md`). Everything else the campaign
built was a one-off probe and was deleted at PR handoff once its question
was answered; the journal's kill list holds the verdicts. Run every timed
GPU rig under `/usr/bin/lockf -k /tmp/jolt-metal-gpu.lock` with the GPU
otherwise idle.

| rig | command | measures | used by |
|---|---|---|---|
| `metal_fr_bind` | `cargo bench -p jolt-eval --features metal --bench metal_fr_bind` | Single `FrBind` kernel dispatch (Criterion, sync-bracketed). Doubles as the ambient-window health probe: healthy ≈ 255 µs @2^20, certification requires < 350 µs. | every wave-gate certification |
| `irr_roof` | `IRR_ROOF_ROWS=<dump> cargo bench -p jolt-eval --features metal --bench irr_roof` | Stage-5 IRR phase/suffix scan kernels against candidate roofs on production-distribution rows: reduce share, simdgroup/width sweeps, quiet/loads floor probes, `fr_mont_mul` chain ladder, in-body fixed-step scan rung. Row dumps come from a prove run with `JOLT_IRR_DUMP_ROWS=<path>`. | `lane-reports/metal-w12-st5scan.md`, `metal-w17-scangap.md` |
| `miller_multipair` | `cargo bench -p jolt-eval --features metal --bench miller_multipair` | Stage-8 reduce-round `multi_pair_device` at production dispatch sizes; fly vs split-ladder arms (`JOLT_MILLER_FLY_SPLIT`), pairing TG cap per invocation (`JOLT_METAL_PAIRING_TG_CAP`). | `lane-reports/metal-w5-st8.md`, W4 fly bundle |
| `st0-contention` | `cargo run --release -p jolt-eval --bin st0-contention --features metal -- --scale 22 --iters 5 --legs commit,walk,corun` | Stage-0 in isolation, no full prove: Metal commit slot, hoisted record walk, and their co-run; `g1-N` single-superchunk tier-1 dispatches; `g1x` = the tier-1 `jk_g1_seg_sum` attribution matrix (width sweep, variant pipelines, mul-chain roofs); `soak` memory-pressure controls. | `lane-reports/metal-w12-st0attr.md`, wave-9 X9 |
| `metal_microbench` | `cargo run --release -p jolt-kernels --example metal_microbench --features metal` | Device characterization: dispatch latency floor, bind/reduce rates, unified-memory wrap costs. The evidence behind `DEFAULT_MIN_TERMS` and the per-slot gate convention. | W1 infrastructure, gate-threshold sweeps |
| `pairing_pipeline_stats` | `cargo run --release -p jolt-kernels --example pairing_pipeline_stats --features metal` | `maxTotalThreadsPerThreadgroup` per pairing-family pipeline — the only register-pressure reading available without the `metal` CLI toolchain. | W3/W4 fly spill pricing |

Supporting seams that stay with the rigs: the `bench-utils` feature
(fixture re-exports + `MetalContext::compile_variant`/`dispatch_variant`
ad-hoc kernel probes), `JOLT_IRR_DUMP_ROWS` (production row dump for
`irr_roof`), and `jolt-eval`'s `telemetry:*` objectives over
`jolt-prover profile --backend metal` runs.

Env-gated in-tree diagnostics (off by default, documented at their
definitions): `JOLT_METAL_CB_TRACE=1` — one stderr line per committed
command buffer with its GPU execution window and dispatch summary (the
per-kernel attribution mechanism behind the campaign's CB numbers);
`JOLT_LIFETIME_TRACE=1` — drop-site tracing for the trace-record
allocation family (RSS/footprint debugging).
