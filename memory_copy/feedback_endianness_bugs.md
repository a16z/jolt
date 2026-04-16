---
name: endianness_bugs
description: Be especially vigilant about endianness bugs — they are very common in this codebase
type: feedback
---

Be especially aware of endianness bugs when working with polynomial indices, challenge orderings, and eq table construction.

**Why:** Endianness bugs are very common in this project. The codebase has multiple endianness conventions — jolt-core's EqPolynomial uses BIG-ENDIAN indexing (r[0] = MSB), binding order (LowToHigh/HighToLow) determines which variable is bound first, and challenge vectors can be in either BIG_ENDIAN or LITTLE_ENDIAN order depending on context. Mismatches between these conventions are a frequent source of subtle bugs that cause sumcheck divergence.

**How to apply:** When working with eq tables, challenge vectors, or polynomial index orderings:
1. Always verify which convention each system uses (BIG_ENDIAN vs LITTLE_ENDIAN)
2. Check that point[0] maps to the expected bit position when building eq tables
3. Verify that LowToHigh binding processes point[n-1] (LSB) first, not point[0]
4. When converting between OpeningPoint<BIG_ENDIAN> and round-order challenges, track the reversal
5. The GruenSplitEqPolynomial binds w[current_index-1] first (i.e., from the END of w), which means the last element of the tau vector is bound first in LowToHigh mode
