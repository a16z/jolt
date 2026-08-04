# W3B — BytecodeReadRafCycle 2^27 re-certification

## Verdict

**GO: remove the 2^26 device cap.** W3A's lean-memory regime removes the
certification cliff. Two bench-locked 2^27 A/B pairs, in opposite orders,
put device-on st6b 2.194 s and 2.650 s ahead of the CPU arm. IncCR improves
in both pairs. Command-buffer timestamps show no other member's GPU execution
regressing.

## Lean-regime 2^27 A/B

Same release binary. `JOLT_METAL_MIN_TERMS_BYTECODE_READ_RAF_CYCLE=huge`
selects the CPU arm after the cap-removal edit; unset selects the device.
Pair 1 predates the edit and used the requested
`JOLT_METAL_MAX_TERMS_BYTECODE_READ_RAF_CYCLE=999999999999999` override.

| metric | pair 1 device | pair 1 CPU | pair 2 CPU | pair 2 device |
|---|---:|---:|---:|---:|
| whole prove | 78.18 s | 88.76 s | 82.56 s | 85.06 s |
| st6b | **16.937 s** | 19.131 s | 18.476 s | **15.826 s** |
| BRRC prepare | 0.267 s | 2.289 s | 2.053 s | 0.334 s |
| BRRC rounds | 4.978 s | 7.089 s | 3.454 s | 4.829 s |
| IncCR prepare | 1.828 s | 1.877 s | 1.637 s | 1.730 s |
| IncCR rounds | **2.720 s** | 2.787 s | 3.528 s | **2.467 s** |
| st6b-entry footprint | 54.20 GiB | 54.26 GiB | 54.18 GiB | 54.39 GiB |
| peak footprint | **75.31 GiB** | 79.24 GiB | 79.16 GiB | **75.49 GiB** |

Mean st6b: 16.382 s device vs 18.804 s CPU, **−2.422 s (−12.9%)**.
Whole-prove walls moved with pair order outside st6b; the stage-local gate is
stable in both directions.

The W3A §4 `+6 GiB` question is confirmed directionally but measures
**+3.80 GiB at lifetime peak** here: capped CPU re-hosting builds fresh BRRC
tables after the matched 54.2–54.4 GiB entry, while device-on retains its
existing buffers.

## Attribution

`JOLT_METAL_CB_TRACE=1` on the reverse-order pair separates kernel execution
from queue wait:

| non-BRRC device work | CPU arm GPU time | device arm GPU time |
|---|---:|---:|
| Bool lazy+dense | 1.005 s | 0.969 s |
| both RAV lazy+dense | 2.226 s | 1.701 s |
| RamHamming rounds | 0.175 s | 0.100 s |
| IncCR init+rounds | 0.227 s | 0.226 s |

No device member's execution regresses. RamRa's inclusive wrapper span grows
1.391→2.740 s because the span remains open across the batch's synchronous
collect phase; its RAV command buffers execute before BRRC on the single
global Metal queue. This is wait attribution, not slower RamRa kernels.

The four proposed cliff mechanisms discriminate as follows:

1. **Flat-factor/SLC blowout: no.** BRRC GPU execution is 0.667 s at 2^26
   and 1.279 s at 2^27: 1.92× for 2× rows. Dense+adopt is 0.068→0.125 s.
2. **64-bit MSL offsets: no.** Only dense/adopt use the wide factor rebase;
   their scaling is linear, and the synthetic `2^32`-word parity probe passes.
3. **Occupancy collapse: no.** Init, lazy, adoption, and dense command-buffer
   times show no super-linear tier discontinuity.
4. **Batch scheduling: historical cliff mechanism.** BRRC's lean 2^27 GPU
   work is 1.279 s, while its host-visible prepare+round span is 5.163 s;
   the balance is ordered queue wait. Under the old fat regime, compressor
   stalls and fresh-page demand inflated those waits and the capped CPU arm's
   tables by another 3.8–6 GiB. W3A removed that regime dependency.

At 2^26 the first CB-timestamp A/B shows no victim signature: st6b
3.218 s device vs 4.774 s CPU, with every other member span improving.

## Retained change and gates

- `BytecodeReadRafCycle` now uses the ordinary lower-bound `metal_gate`; the
  `MAX_DEVICE_ROWS` constant and now-unused `metal_gate_capped` helper are gone.
- Forced-device lockstep plus the synthetic 2^27 offset suite: 5/5.
- `byte_diff`, `prover-fixtures`: 12/12; `prover-fixtures,metal`: 12/12.
  The campaign's 11-fixture count became stale on this trunk.
- `jolt-kernels --features metal`: 239/239; `jolt-dory`: 46/46; legacy
  muldiv: 3/3 `host` and 3/3 `host,zk`.
- Clippy `-D warnings`: workspace `allocative,host`; workspace
  `allocative,host,zk`; Metal `jolt-kernels` + `jolt-prover`. Fmt clean.
- 2^27 run budget: 4/6. Artifacts:
  `/tmp/w3b-lean-s27-{device,capped}.{log,json}`,
  `/tmp/w3b-s2{6,7}-{device,cpu}-cb.{log,json}`.
