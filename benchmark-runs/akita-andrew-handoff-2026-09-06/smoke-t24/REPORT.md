# Four-workload Metal matrix

Verified observations: 4.

Rates below use padded trace rows, not hashes/second. M5 is an optional projection, not measured.

| Scale | Measured mean MHz | Measured >=10 MHz? | Projected M5 mean MHz (1.13x) |
|---|---:|---|---:|
| 2^24 | 1.9358 | no | 2.1874 |

Only complete four-workload scales receive a mean. One observation per cell; no uncertainty interval.
Individual times, actual rows and memory are in results.csv; machine/build/guest identity is in manifest.json.
Inspect events.jsonl and raw logs for failures. No omitted/failed cell is treated as a pass.
