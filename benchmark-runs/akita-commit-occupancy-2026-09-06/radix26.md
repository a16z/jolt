# D22: bounded radix-26 root accumulation

Test five signed radix-26 digits per field coefficient, normalizing every two
eight-row tiles. This removes per-update carry propagation while retaining
the accepted two-task/four-coefficient mapping, 40 persistent source words
per thread, 32 KiB shared tile and original matrix traffic. This is an internal
arithmetic experiment, not a field, commitment, transcript or parameter change.

## Representation and normalization

Let B=2^26, Q=2^24, C=AKITA_OFFSET=2^32-22537 and p=2^128-C.
An accumulator represents S=sum(d_i*B^i,i=0..4) modulo p. Each d_i is signed
i32. Decode an input a in [0,p) into four digits in [0,B) and a top digit in
[0,Q); add or subtract these five digits according to the unchanged negacyclic
sign. Input word extraction is:

```
x0 = v0 & (B-1)
x1 = ((v0 >> 26) | (v1 << 6)) & (B-1)
x2 = ((v1 >> 20) | (v2 << 12)) & (B-1)
x3 = ((v2 >> 14) | (v3 << 18)) & (B-1)
x4 = v3 >> 8
```

All extraction operands are u32. Unsigned shift truncation discards only bits
outside the requested digit. Normalize after tile indices1,3,5,...; there
are at most16 selected entries per task between normalizations:

1. For i=0..3, carry=floor(d_i/B), replace d_i by its nonnegative remainder
   modulo B, and add carry to d_(i+1). This preserves S as an integer.
2. Let q=floor(d4/Q); replace d4 by its nonnegative remainder modulo Q.
3. Set d0-=22537*q and d1+=64*q. These two updates add q*C because
   C=64*B-22537. Hence the total change is -q*(2^128-C)=-q*p.

No u64 multiplication is needed for periodic reduction. The small constant
22537 is derived from AKITA_OFFSET, not a second owner of the modulus.

## Signed-intermediate bound

Inductive post-normalization invariant:

```
-17*22537 <= d0 <= B-1+17*22537
-17*64    <= d1 <= B-1+17*64
0 <= d2,d3 < B;  0 <= d4 < Q
```

Zero initialization satisfies it. Up to16 signed input digits change each
lower digit by at most16*(B-1), and the top by at most16*(Q-1).
The first carry is in [-17,17]. Including that carry, the second digit lies
in [-16*B-1089,17*B+1088], so its carry is also in [-17,17]. Each of the next
two digits lies in [-16*B-1,17*B], preserving this carry bound. The top lies
in [-16*Q-1,17*Q], so q is in [-17,17]. The fold restores the stated invariant.

Every intermediate is bounded in magnitude by17*B+17*22537=1141233817,
strictly below2^31. The products |22537*q|<=383129 and |64*q|<=1088 also fit.
These bounds include additions during carry propagation, not just stored
post-normalization digits. Fewer than16 updates, including an odd final tile,
are covered. Exactly one selected coefficient per source row is required;
the frozen K256/D128 kernel provides that property.

## Final canonical field value

Propagate the first four radix carries once more, but retain signed d4.
Pack normalized lower digits into u32 words:

```
w0 = u32(d0) | (u32(d1) << 26)
w1 = (u32(d1) >> 6) | (u32(d2) << 20)
w2 = (u32(d2) >> 12) | (u32(d3) << 14)
w3 = (u32(d3) >> 18) | (u32(d4) << 8)
```

Use the existing AkitaWideAccumulator/akita_reduce_wide final reducer:
low_digits=(w0&65535,w1&65535,w2&65535,w3&65535),
high_digits=(w0>>16,w1>>16,w2>>16,d4>>8). The last component remains signed
and includes the high quotient; truncating it to16bits would be incorrect.
This representation is the same integer S. The existing reducer canonicalizes
S modulo the unchanged p. No normalization/reduction formula is added to the
verifier or protocol layer.

## Cost and launch decision

Existing denominator: full P19 panel approximately12.0s, U=1.253317T useful
updates. Source accounting: digit extraction about14 scalar ops plus five
signed multiply-add expressions/update; periodic normalization about16ops
per coefficient per16 possible contributions (~1.8ops/update at observed
density). Compare the earlier naive36ops/update carry tally. Neither tally is
an ISA count; multiply-add fusion and temporary register allocation are open.

Unlike R4's eight digits and D21's one-task mapping, this retains five source
words/coefficient and349matrix sweeps. It adds extraction, periodic carry/fold
work, and final packing, but no source pass, metadata, synchronization, field
storage or extra matrix reads. Logical shared gathers remain20.053TB and
matrix requests1047GiB. Traffic floors are unchanged; no measured issue-rate
floor or occupancy percentage is claimed. The source-op reduction is large
enough to warrant one production-body test, not a predicted wall gain.

## Claim-to-code map and verification

| Claim | Diagnostic code unit to implement | Independent check |
|---|---|---|
| Five-digit exact decode and signed updates | radix26 accumulation helper | Existing U128 add/sub oracle; all final bytes |
| Normalization preserves S mod p and bounds | radix26 normalization helper | GPU state probe vs CPU weighted modular sum, with boundary states and post-bounds |
| At most16 updates between folds | kernel fold after odd tile indices | Fixed8rows/tile and2tile cadence; multi-tile dense fixtures |
| Signed top quotient survives packing | radix26 final packing | p-1 dense positive/negative cases and word-boundary matrices |
| Same work/output ownership | original root body and reducer | ExactH,65target samples,192MiB parent-output comparison |
| Performance and resources | full-panel controller | Frozen paired gates, watchdog/RSS/cooling checks |

Verification update09:24UTC: all four D22 observations passed512 scalar
normalizer states (32 corner combinations), ten P1024 field-oracle fixtures,
65 independent target partial samples and all192MiB final-output bytes.
The generated cadence is every two eight-row tiles. GPU reduction8.9695%
and complete-boundary saving1.077148s miss the fixed10% component gate;
no production promotion or full-proof gain is established.
Use P1024 reduced fixtures so periodic normalization actually executes, plus
allzero, selected-zero, odd-task tail, extremal matrix words, and dense p-1
inputs with shifts0 and127. CPU state-probe ground truth is a weighted modular
sum using U128 modular additions, not a second radix-normalization algorithm.

Epoch8: <=3transactions,checkpoint09:45UTC; D22 tooling<=25min and cooled
cohort<=12min. Run P,C, stop for futility if GPU saving<3% OR wall saving<0.3s;
otherwise finish C,P. Ranking still requires>=10% GPU reduction AND>=1s wall
saving, parent drift<=3%, all exact checks,5s percommand/180s process/88GiB,
zero swaps/watchdogs. No full proof or production change before that gate.
Security parameters, evaluator, transcript and verifier remain frozen. The
existing implementation already has witness-dependent selector loops; this
does not establish a new constant-time guarantee. No new external exposure
or witness storage is introduced.

Remaining unknowns: M4 lowering, physical register allocation, issue mix and
actual speed. Resolve through the frozen diagnostic, not invented counters.

## D23 preregistration: decode during shared staging

Keep the D22 arithmetic, final reducer, original selector slices,64tasks/group,
1024threads,16position partials,349matrix streams and44commands. Decode the
five digits once during cooperative global-to-shared copying. The hot loop
gathers five decoded digits and performs five signed multiply-add expressions.
Use8positions/4rows per tile and five1024-word planes:20480shared bytes.
Global A remains raw128-bit and3GiB; no expanded matrix or extra global pass.
Normalize every FOUR tiles, retaining the proved16-contribution bound exactly.
The original P19 geometry and P1024 fixtures divide this tile size exactly.
Compile-time constraints pin the digit/cadence and signed-intermediate bounds.

For U=1253317010304 useful coefficient updates,349*3*524288*128=
70262980608 matrix fields are staged. Thus each staged field supplies17.8375
useful updates on average. Moving approximately14 source decode operations
out of the hot loop avoids14*(U-70262980608)=16562756415744 source operations.
This is NOT an ISA count or a predicted proportional speedup. It adds4U=
5013268041216 bytes of logical shared reads and281051922432 shared-store
bytes,25% above D22. Shared storage drops32->20KiB, but no residency credit
is assumed. Tile count/barriers/control double2048->4096 per partial; D1's
raw half-tile experiment cost approximately10%, a qualified~1.2s penalty
proxy, not a transferable constant. Global matrix requests remain1047GiB.
These competing terms make this a discriminating experiment, not a promised
gain. There is no new measured occupancy or issue-rate floor.

Claim/code mapping: a new staged helper owns digit extraction and five-plane
gather; the existing D22 normalizer/final reducer are unchanged. Generated
candidate tile constants are distinct from accepted D128/D512 constants.
Resource checks require parent32768 and candidate20480 shared bytes. The
same512-state probe, ten P1024 independent field oracles,65target samples,
all final bytes and original padding checks must pass before any ranking.

Epoch8 transaction2; tooling<=15min,cooled cohort<=12min; checkpoint09:45UTC
or immediately after its in-flight fixed cohort completes. P,C,C,P with120s
cooling and immutable binaries/artifacts. Stop after first P,C if GPU saving
<3% OR wall saving<0.3s. Full ranking remains>=10% GPU reduction AND>=1s wall
saving, <=3% parent drift, zero swaps/watchdogs,5s command/180s process/88GiB.
No production/final-proof change before this gate. No protocol, parameter,
field, transcript, verifier or security change. A12-position tile or different
normalization cadence would require a separate bound and preregistration.
