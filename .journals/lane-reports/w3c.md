# W3C — RegistersRWC legacy-table parallel build

## Decision and shape

Host-parallel, exact-output three-pass build:

1. Rayon counts variable-length rows in coarse cycle chunks.
2. The host scans counts into exact legacy CSR offsets.
3. Rayon scatters entries, `inc`, and operand-index lanes directly into
   disjoint spare-capacity slices of the final vectors.

The Metal slot had a separate serial builder, so the same shape was applied to
both optimized-host and Metal prepare paths. Message/bind kernels and every
round representation are unchanged. Production work is capped at four chunks;
eight chunks won the target more strongly but induced the same downstream
regression.

The serial builders remain as the equality oracle and as a same-binary timing
arm selected by `JOLT_REGISTERS_PREPARE_SERIAL`. Two fixture-scale tests compare
every entry and companion table, including Metal CSR offsets:

```text
cargo nextest run -p jolt-kernels --features metal --cargo-quiet parallel_prepare
2/2 passed (242 skipped)
```

No device builder was attempted. The four-chunk residue projects above 1 s at
2^27 (0.434 s at 2^25 times the measured 2.05² scaling), but an exact
variable-length device build needs a count/scan/scatter dependency before the
existing upload/round path. W2B's 0.30–0.36 s device result used a different
fixed-slot representation; its small 2^25 margin over this host result does not
justify another command-buffer boundary.

## 2^24 gate — four chunks

Locked, AC-powered, non-monitor same-binary A/B. A is the serial environment
arm; B is the parallel default.

| metric | serial A | parallel B | delta |
|---|---:|---:|---:|
| Registers prepare | 0.865 s | **0.330 s** | **−61.8%** |
| stage 4 | 1.661 s | **1.208 s** | **−27.2%** |
| stage 3 | 0.364 s | 0.375 s | +3.0% |
| stage 5 | 1.845 s | 1.891 s | +2.5% |
| stage 8 | 1.171 s | 1.204 s | +2.8% |

The prepare and stage-4 targets pass; the strict cross-stage gate does not.
Artifacts: `/tmp/w3c-chunk4-s24-{A,B}.json`.

## 2^25 cool confirmation

### Eight-chunk ABBA

Each arm started after at least three quiet minutes on AC with load below 6.

| metric | serial mean | parallel mean | delta |
|---|---:|---:|---:|
| Registers prepare | 1.302 s | **0.274 s** | **−78.9%** |
| stage 4 | 2.737 s | **1.689 s** | **−38.3%** |
| stage 8 | 4.301 s | 4.541 s | **+5.6% / +240 ms** |
| summed stages | 19.922 s | 18.978 s | −4.7% |

Artifacts: `/tmp/w3c-chunk8-s25-{A1,B1,B2,A2}.json`.

### Four-chunk independently cooled pair

The worker cap was reduced to four to test whether lower instantaneous host
pressure removed the spill. Both arms again started after at least three quiet
minutes on AC; control load was 1.08 and parallel-arm load was 0.98.

| metric | serial A | parallel B | delta |
|---|---:|---:|---:|
| Registers prepare | 1.299 s | **0.434 s** | **−66.6%** |
| stage 4 | 2.674 s | **1.835 s** | **−31.4%** |
| stage 5 | 3.156 s | 3.142 s | −0.4% |
| stage 6a | 0.052 s | 0.056 s | +8.5% / +4 ms |
| stage 6b | 1.483 s | 1.537 s | **+3.6% / +54 ms** |
| stage 7 | 0.227 s | 0.246 s | +8.3% / +19 ms |
| stage 8 | 4.215 s | 4.437 s | **+5.3% / +222 ms** |
| summed stages | 19.802 s | 19.127 s | −3.4% |

Prepare and stage 4 clear the −50%/−10% gates. Stage 6b and stage 8 reject the
candidate. Artifacts: `/tmp/w3c-chunk4-s25-{A,B}.json`.

## Mechanism and verdict

The exact table build parallelizes cleanly. Its saving projects to about 3.6 s
at 2^27 from the independently cooled pair, consistent with the lane prize.
The failure is shared-SoC pressure: replacing a 1.30 s low-core build with a
short four/eight-core burst leaves later Metal work slower. Stage 8 reproduced
at +5.3–5.6% under both task counts despite byte-identical tables and untouched
round code. Four chunks additionally moved stages 6b–7, so reducing Rayon
fan-out did not satisfy the no-stage-over-2% clause.

**Rejected.** The target win is real, but the candidate fails the cross-stage
retention gate at both scales. The full retention matrix was therefore not run;
the two direct equality tests, `cargo fmt --check`, and the default pre-commit
clippy passed. Feature-matrix clippy and suites were skipped. No 2^27 run was
made.

Experimental implementation commit:
`b1533470a7382a19f20434ec42b0b67905b7069e`.

Final experimental binary SHA-256:
`9291d5858a03d91b884b2a902725e162baea3adcc7515d5929d898b78fadf385`.
