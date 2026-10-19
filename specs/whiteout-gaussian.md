# Design note: `jolt-crypto::sample` Gaussian and joint rejection (M5a)

Design only. Labels refer to `akita-paper/sections/whiteout/` (read
2026-10-19). Numbers the text omits come from `scripts/whiteout-costs.py`
(headline: k=2, variant 1, `mqo+eroot`).

## 1. Distributions

- **Gaussian.** `D_σ(z) ∝ exp(−z²/(2σ²))` on Z, centre 0 (§2.5,
  `wo:sec:prelim-rejection`). `π_0` is the product law with width `σ_b` on
  every coordinate of block `b`.
- **Widths.** `σ_b = αΓγ_bω_b^{−1/2}`, `α = 16`, `Γ = 18`, `Σω_b = 1`
  (§6.2). The script gives six blocks, 6,220,582 coordinates, and two
  distinct widths: `σ ≈ 1.366·10⁷ (2^23.70)` for `w`, `t_hat`, `const_ops`
  and `σ ≈ 8.536·10⁵ (2^19.70)` for `anchor_salt`, `pads`, `pad_salt`.
  Realized: `k_b = 16,080,468` and `1,005,030`.
- **Realized widths.** The sampler (§2) needs `σ_b = k_b·σ_2` with integer
  `k_b` and `σ_2 = 1/√(2 ln 2)`. The parameter layer rounds `k_b` up. A wider
  width keeps `‖cs‖_σ ≤ 1/α`, and the box and digit counts are derived from
  the realized `σ_b`.
- **Vectors.** Each private coordinate of `s` is a ring element. Its `d`
  coefficients are i.i.d. `D_{σ_b}`, lifted to F_q by a constant-time
  select of `z` or `q − |z|`. Public coordinates get `y = 0`, are excluded
  from every block, and never enter the rejection step; that branch depends
  only on the public layout (§4.2 "Public coordinates").
- **Tail.** There is no separate tail cut; the sampler's support ends at
  `14k_b ≈ 16.5σ_b` (§2). The acceptance box `𝓑` is
  `|z_i| ≤ L_b = ⌊τσ_b⌋`, with `τ = √(2 ln(2N/(1−p_𝓑))) ≈ 6.82` for
  `p_𝓑 ≥ 0.999` (§2.5, bound after Thm `wo:thm:joint-rejection`). §4.3
  requires `𝓑 ⊆ 𝓑_cert`.

**Joint rejection rule** (Def. `wo:def:joint-rejection`). Let
`ε = e^{−t²/2} = 2^{−137}` and `M = exp(t/α + 1/(2α²)) ≈ 2.37`. Given
`z = y + v` with `v = cs` (Protocol `wo:prot:sigma`, step 4), the rule
accepts with probability `min{1, π_0(z)/(M π_v(z))}` if `z ∈ 𝓑` and rejects
otherwise. Expanding the ratio:

`π_0(z)/π_v(z) = exp(½‖v‖²_σ − Σ_i z_i v_i/σ_{b(i)}²)`.

With realized widths, `σ_b² = k_b²/(2 ln 2)`, so

`P_acc = min{1, 2^X / M}`, where `X = Σ_b (V_b − 2S_b)/k_b²`,
`S_b = Σ_{i∈b} z_i v_i`, and `V_b = Σ_{i∈b} v_i²`.

`X` is an exact rational. Acceptance per attempt is in
`[(p_𝓑−ε)/M, p_𝓑/M]` (Thm `wo:thm:joint-rejection`); `T_att = 163`
caps all-rejected at `2^{−128}` (§6.2, Thm `wo:thm:completeness`).

## 2. Sampler

A CDT for the target width is infeasible: `τσ ≈ 10⁸` entries. The sampler
is BLISS's binary-Gaussian method (Ducas et al. 2013, Alg. 10–11), made
isochronous as in Howe–Prest–Ricosset–Rossi 2020. One trial for block `b`:

1. Draw `x ← D⁺_{σ_2}` (`∝ 2^{−x²}`, `x ∈ [0,13]`) from 13 CDT
   thresholds `T_j = ⌊2^{192}·Pr[X ≤ j]⌋`, `j ∈ [0,12]`. The whole table is
   scanned; `x` is the sum of the `u ≥ T_j` bits, computed with
   constant-time compares. The dropped mass `Pr[X ≥ 14] ≈ 2^{−196.6}` is
   below the table's resolution.
2. Draw `y` uniform in `[0, k_b)` with the shared uniform rejection sampler.
3. Set `z = k_b x + y`. Accept with probability
   `exp(−(y² + 2k_b xy)/(2σ_b²)) = 2^{−N/k_b²}`, where `N = y² + 2k_b xy < 27k_b²`.
   `2^{−N/k_b²}` is computed as a product over all `ℓ_b = ⌈log₂ 27k_b²⌉`
   bits of `N` (53 and 45 bits here). Every factor is a constant-time select
   between 1 and a public 192-bit entry `2^{−2^i/k_b²}`. Then compare with
   one 192-bit uniform.
4. Draw a sign bit and output `±z`. Reject `z = 0` when the sign bit is 1.
   Steps 3 and 4 combine into one acceptance mask.

The expected number of trials is 1.47 (acceptance about 0.68, computed
numerically).

**Constant time.** The sampler takes no secret input; its output is its
only secret. Every trial runs a fixed instruction sequence, and the loop
repeats on the trial's accept bit. A rejection sampler's accept/reject
sequence is independent of the value it finally accepts (the HPRR20
isochrony argument), and `k_b` is public. So the running time depends on
independent coins only, never on the output or the witness. Secret data
gets no hardware division, no `f64`, and no table index.

**Constants.** Every runtime constant (`T_j`, `2^{−2^i/k_b²}`,
`2^{−2^i/Q}`, `M^{−1}`) is computed directly in `new` from public inputs by
a variable-time big-integer fixed-point routine at 320-bit working
precision (`ln 2` by its binary series, `exp` by Taylor series after range
reduction, square root by integer `sqrt`). Its error is below `2^{−300}`,
and each constant is then floored to 192 bits, so it lies in
`(x − 2^{−192}, x]`. Constants are never derived from one another; for
example, repeated squaring of `2^{−1/k²}` would amplify the error by about
`2^{52}`. Tests compare them with mpmath vectors (400-bit).

**Precision.** Per trial, the CDT floors move at most `26·2^{−192}` of mass,
the exp product has absolute error at most `(2ℓ_b+1)·2^{−192}` (`ℓ_b`
truncations, `ℓ_b` floored factors, and the initial `1 − 2^{−192}`), and
the dropped tail is `2^{−196.6}`. The trial's joint law of `(accept, z)` is
then within `δ ≤ 134·2^{−192} ≈ 2^{−184.9}`, and one output is within
`2δ/P_acc ≈ 2^{−183.3}` (`P_acc ≈ 0.68` per trial). Over every sample of
every attempt, `6.2·10⁶ × 163 ≈ 2^{29.9}` at the headline configuration
and about `2^{34}` at larger traces, the sampler adds at most `2^{−149}` to
Thm `wo:thm:zk`'s `2^{−128.8}` (§6.2 asks implementations to add this
term).

**Randomness.** `R: RngCore + CryptoRng` (rand_core 0.6, as
`whiteout-interfaces.md` §5). For parallel sampling, the caller draws one
32-byte seed from `R` per chunk and expands it with ChaCha20. This adds a
PRG assumption on ChaCha20 to the otherwise statistical zero-knowledge
argument (the prover's coins are pseudorandom, not uniform).

## 3. Rejection rule in constant time

- **Shift bound.** `v = cs` must be the exact centred-integer negacyclic
  product (not a reduction mod q). Each block carries a public bound
  `|v_i| ≤ V_b` (honestly `‖c‖₁ ·` max digit). `accept` checks it in
  constant time over all coordinates and returns a typed error only when
  it fails, which an honest prover never triggers.
- `S_b = Σ z_i v_i` and `V_b = Σ v_i²` are exact `i128` sums over `z`
  clamped (constant time) to the box, so arithmetic never overflows even
  for an out-of-box `z`, which is rejected anyway. At the headline
  configuration `|V_b − 2S_b| < 2^{66}` and `Q/k_b² < 2^{48}`, so
  `P = Σ_b (V_b − 2S_b)·Q/k_b²` fits `i128`. `new` checks the worst case
  `Σ_b (2L_bV_b + V_b²)·len_b·Q/k_b² < 2^{126}` and rejects the
  parameters otherwise.
- Box membership is an AND of constant-time compares across all
  coordinates, with no early exit.
- `X = P/Q` with `Q = Π k_b²` over the distinct widths (`2^{87.8}` here).
  A constant-time restoring division (128 fixed iterations of
  shift-compare-select; never `udiv`) gives the floor quotient `n` and
  `r ∈ [0, Q)`, with a constant-time sign fix for negative `P`.
- `2^X/M = 2^{n+1}·G` with `G = 2^{−(Q−r)/Q}·M^{−1} ∈ (1/8, 1/2)`.
  `2^{−(Q−r)/Q}` uses the bit-product method over `⌈log₂ Q⌉ + 1` public
  factors `2^{−2^i/Q}`. `n` is clamped to `[−256, 2]` in constant time.
  For `n + 1 ≤ 0`, `G` is shifted right by a barrel shifter of
  constant-time selects; for `n + 1 > 0` the result saturates at 1 when the
  top `n + 1` bits of `G` are nonzero, so nothing wraps.
- `M` is computed in `new` from `(α, ε bits)`: `t = √(2·ε_bits·ln 2)`,
  `M = exp(t/α + 1/(2α²))`. `M^{−1}` is floored to 192 bits, which is exact
  for some `M' ≥ M`; Thm `wo:thm:joint-rejection` holds for any larger `M`.
- Compare the result with a 192-bit uniform. Its absolute error is below
  `2^{−181}` per attempt, negligible beside `ε`.

Only the accept bit (`Choice`) leaves the step; `jolt-zk::whiteout`
branches on it. Its law depends on the state only within `ε/M` (H3 of Thm
`wo:thm:zk`).

## 4. API sketch (`jolt_crypto::sample`)

```rust
pub struct GaussianWidth { k: u64 }               // σ = k·σ_2, 1 ≤ k ≤ 2^31
impl GaussianWidth {
    pub fn from_multiplier(k: u64) -> Result<Self, SampleError>;
    pub fn at_least(sigma: f64) -> Result<Self, SampleError>;    // public parameter path only
    pub fn multiplier(&self) -> u64;
    pub fn sigma(&self) -> f64;
}
pub struct GaussianSampler { /* k, CDT, exp table */ }
impl GaussianSampler {
    pub fn new(width: GaussianWidth) -> Self;
    pub fn sample<R: RngCore + CryptoRng>(&self, rng: &mut R) -> i64;
    pub fn fill<R: RngCore + CryptoRng>(&self, rng: &mut R, out: &mut [i64]);
}
/// Public parameters, built once. Blocks cover only Gaussian-masked
/// coordinates; public coordinates are excluded.
pub struct BlockParams { pub width: GaussianWidth, pub box_bound: u64, pub shift_bound: u64, pub len: usize }
pub struct JointRejection { /* blocks, Q, exp table, M^{-1} */ }
impl JointRejection {
    /// Computes M = exp(t/α + 1/(2α²)), t = √(2·epsilon_bits·ln 2).
    pub fn new(blocks: Vec<BlockParams>, alpha: u32, epsilon_bits: u32) -> Result<Self, SampleError>;
    /// `z = y + v` and `v = cs` (exact centred-integer negacyclic product),
    /// block-major, lengths fixed by `blocks`.
    pub fn accept<R: RngCore + CryptoRng>(&self, z: &[i64], v: &[i64], rng: &mut R)
        -> Result<Choice, SampleError>;
}
pub enum SampleError { WidthOutOfRange, NoBlocks, BadRejectionParameters,
    ArithmeticBound, LengthMismatch { expected: usize, got: usize }, ShiftOutOfRange }
```

Errors cover public shape, plus `ShiftOutOfRange` (the release check on
`v`).

**Dependencies.** `subtle` (2.6.1, already in `Cargo.lock` through the
arkworks and sha3 stacks) for `Choice` and constant-time selects. The
192-bit arithmetic is a three-limb hand-written type, so `crypto-bigint`
(0.5.5, also already locked) is not needed. `num-bigint` (already an
optional dependency of jolt-crypto) becomes unconditional for the public
constant computation. `jolt-zk::whiteout` owns attempts,
nonces, and `T_att` (`whiteout-interfaces.md` §5).

## 5. Verification

- **Ground truth.** For `σ ≫ η_ε(Z) ≈ 5.6` (smoothing parameter in the
  `ρ_s(x) = e^{−πx²/s²}` convention, `ε = 2^{−140}`, which corresponds to
  width `σ = s/√(2π) ≈ 2.2` in this note's convention), the discrete Gaussian's
  binned mass equals the normal CDF to within f64 precision, by the Poisson
  summation formula. That gives the test oracle.
- **Chi-square tests.** Point-wise at small `k ∈ {1, 2, 3, 10}` against the
  exact pmf. At the production `k_b`, 10⁸ samples in 200 equiprobable bins.
- **KS test.** Against the normal CDF.
- **Moments.** Mean 0, variance `σ²`, kurtosis 3, each within 5 standard
  errors.
- **Binary Gaussian.** The CDT against the exact `2^{−x²}/Σ`.
- **Fixed-point kernels.** Frozen vectors from mpmath at 250 bits for
  `2^{−N/k²}` and `2^{r/Q}`.
- **Rejection rule.** On small instances (n ≤ 4, small σ), enumerate the
  exact acceptance rate and `π_0(·|𝓑)`. Check the Monte-Carlo rate within
  the theorem's bounds and chi-square the accepted `z`.
- **Timing tool.** An intentional diagnostic in
  `crates/jolt-crypto/examples/sample_dudect.rs`, with its command
  documented and not in CI (spec acceptance M5a). It runs Welch's t-test
  (dudect) on two targets: one sampler trial, with a fixed RNG stream
  against random streams, and `JointRejection::accept`, with fixed `(z,v)`
  against random `(z,v)`. It passes if `|t| < 4.5` after 10⁷ measurements.
- **Code review must check:** no `if`, `match`, `?`, early return, or slice
  index on secret data; no `/`, `%`, or `f64` on secrets; masks behind
  `subtle` or `black_box`; and the disassembly of `trial` and `accept`
  (`cargo asm`) has no secret-dependent branches.

## 6. Performance

Estimated: a trial is 13 three-limb compares plus 53 sequential 192-bit
multiplications (each a dependent 3×3-limb product), about 0.25–0.4 µs on
one core, so about 0.35–0.6 µs per sample at 1.47 trials. The cost script
charges `Ns · att · T_GAUSS = 6.22·10⁶ × 2.37 × 10 ns = 0.148 s` with no
thread count. Under the same accounting, 16 threads with perfect scaling
give 0.33–0.55 s, which moves the 4.7% prover overhead to about 5.5–6.5%;
serial accounting would give 5–9 s, which would dominate the overhead. The
sampler's cost is therefore a thread-count assumption the paper should
state.

## 7. Open questions

1. **Precision versus Rényi divergence.** Statistical distance forces
   192-bit arithmetic. A Rényi-divergence argument (Prest 2017) allows
   64-bit, about 5× faster, but changes Thm `wo:thm:zk`'s statement. Do the
   paper authors want that?
2. **Box `L_b`.** The paper fixes only `p_𝓑 ≥ 0.999`. The cost script's
   envelope mode charges digits at the digit envelope. Should `L_b` be
   `⌊τσ_b⌋` or the envelope? Either must satisfy `𝓑 ⊆ 𝓑_cert`.
3. **Block list.** The paper names no blocks; the script defines six. The
   paper should list them, or W4 should derive them from `PadLayout`.
Q1 and Q3 are paper-author decisions and do not block M5a: the API is
generic over the block list and the precision argument is statistical.
