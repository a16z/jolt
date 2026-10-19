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
 
- **Realized widths.** The sampler (§2) needs `σ_b = k_b·σ_2` with integer
  `k_b` and `σ_2 = 1/√(2 ln 2)`. The parameter layer rounds `k_b` up. A wider
  width keeps `‖cs‖_σ ≤ 1/α`, and the box and digit counts are derived from
  the realized `σ_b`.
- **Vectors.** Each private coordinate of `s` is a ring element. Its `d`
  coefficients are i.i.d. `D_{σ_b}`, lifted to F_q by a constant-time
  select of `z` or `q − |z|`. Public coordinates get `y = 0`, and that branch
  depends only on the public layout (§4.2 "Public coordinates").
- **Tail.** There is no separate tail cut; the sampler's support ends at
  `13k_b ≈ 15.3σ_b` (§2). The acceptance box `𝓑` is
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

1. Draw `x ← D⁺_{σ_2}` (`∝ 2^{−x²}`, `x ∈ [0,12]`) from a 13-entry CDT
   with 192-bit entries. The whole table is scanned; `x` is the sum of the
   `u ≥ T[j]` bits, computed with constant-time compares.
2. Draw `y` uniform in `[0, k_b)` with the shared uniform rejection sampler.
3. Set `z = k_b x + y`. Accept with probability
   `exp(−(y² + 2k_b xy)/(2σ_b²)) = 2^{−N/k_b²}`, where `N = y² + 2k_b xy < 25k_b²`.
   `2^{−N/k_b²}` is computed as a product over all `ℓ_b = ⌈log₂ 25k_b²⌉`
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

**Precision.** The CDT error is at most `13·2^{−192}`, the support cut costs
`2^{−169}`, and the exp product is accurate to `53·2^{−190}`. A trial is
then within `2^{−168}` of exact and an output within `2^{−167}`. Counting
every sample of every attempt, `6.2·10⁶ × 163 ≈ 2^{29.9}`, the sampler adds
at most `2^{−137}` to Thm `wo:thm:zk`'s `2^{−128.8}` (§6.2 asks
implementations to add this term).

**Randomness.** `R: RngCore + CryptoRng` (rand_core 0.6, as
`whiteout-interfaces.md` §5). For parallel sampling, the caller draws one
32-byte seed from `R` per chunk and expands it with ChaCha20.

## 3. Rejection rule in constant time

- `S_b` and `V_b` are exact `i128` multiply-accumulates. With `|z| < 2^{28}`,
  `|v| < 2^{13}` (`‖c‖₁ = 53` times digits ≤ 128), and `N_b < 2^{23}`, the
  sums stay below `2^{64}`.
- Box membership is an AND of constant-time compares across all coordinates,
  with no early exit.
- `X = P/Q` with `Q = Π k_b²` over the distinct widths (`< 2^{96}`) and
  `P` in `I256`. A constant-time `div_rem` by the public `Q` splits `X`
  into `n + r/Q` (crypto-bigint `Uint` shift-subtract; never `udiv`).
- `2^{r/Q}` uses the same bit-product method (`⌈log₂ Q⌉` public factors).
  `2^n` is a barrel shift by constant-time selects, clamped to `n ≥ −256`,
  where the probability is already below `2^{−200}`.
- `M^{−1}` is rounded down to 192 bits. That is exact for some `M' ≥ M`, and
  Thm `wo:thm:joint-rejection` holds for any larger `M`.
- Compare `P_acc` with a 192-bit uniform. The relative error is at most
  `2^{−182}` per attempt, negligible beside `ε`.

Only the accept bit (`Choice`) leaves the step; `jolt-zk::whiteout`
branches on it. Its law depends on the state only within `ε/M` (H3 of Thm
`wo:thm:zk`).

## 4. API sketch (`jolt_crypto::sample`)

```rust
pub struct GaussianWidth { k: u64 }               // σ = k·σ_2, k ≥ 1
impl GaussianWidth {
    pub fn at_least(sigma: f64) -> Result<Self, SampleError>;    // public parameter path only
    pub fn sigma(&self) -> f64;
}
pub struct GaussianSampler { width: GaussianWidth, exp_table: Box<[U192]> }
impl GaussianSampler {
    pub fn new(width: GaussianWidth) -> Self;
    pub fn sample<R: RngCore + CryptoRng>(&self, rng: &mut R) -> i64;
    pub fn fill<R: RngCore + CryptoRng>(&self, rng: &mut R, out: &mut [i64]);
}
/// Public: widths per block, box L_b, M (rounded up). Built once from parameters.
pub struct JointRejection { blocks: Vec<BlockParams>, m_inv: U192 }
pub struct BlockParams { pub width: GaussianWidth, pub bound: u64, pub len: usize }
impl JointRejection {
    pub fn new(blocks: Vec<BlockParams>, m: f64) -> Result<Self, SampleError>;
    /// `z`, `v` laid out block-major; lengths fixed by `blocks`.
    pub fn accept<R: RngCore + CryptoRng>(&self, z: &[i64], v: &[i64], rng: &mut R)
        -> Result<Choice, SampleError>;
}
pub enum SampleError { ZeroWidth, WidthTooLarge, LengthMismatch { expected: usize, got: usize }, BoxExceedsSupport }
```

Errors cover public shape only. `jolt-zk::whiteout` owns attempts,
nonces, and `T_att` (`whiteout-interfaces.md` §5).

## 5. Verification

- **Ground truth.** For `σ ≫ η_ε(Z) ≈ 5.6`, the discrete Gaussian's
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

Estimated, not measured: a trial is 13 three-limb compares plus up to 53
192-bit multiplications, about 100–200 ns on one core, so 150–300 ns per
sample. 6.22·10⁶ samples × 2.37 attempts on 16 threads take 0.14–0.28 s,
against the paper's 10 ns/sample assumption (0.15 s; §6 "Limitations of the
model"). The upper end adds about 0.13 s to the 1.15 s overhead.

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
4. **`|v|` bound.** The `i128` headroom assumes digits ≤ 128 (`const_ops`
   base 256). This must be rechecked if the alphabets change.
