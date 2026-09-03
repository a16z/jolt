# Lane Z: Zeromorph PCS

## Decision

The requested `ell + 1` G1 proof is not the KZG Zeromorph protocol. The non-hiding protocol in [Kohrita–Towa §6](https://eprint.iacr.org/2023/917.pdf) sends:

1. `ell` quotient commitments `C_k`;
2. the raised-degree batch commitment `C_q`;
3. the final KZG witness `pi`.

Proof shape: **`ell + 2` G1, zero Fr**. Removing `C_q` removes the individual bounds `deg(q_k) < 2^k`; retaining two pairings then gives no sound degree check. The implementation keeps `C_q`.

Likewise, [BDFG20 §4](https://eprint.iacr.org/2020/081.pdf) batches distinct *univariate* evaluation points. Distinct multilinear points produce distinct Zeromorph quotient families. The implemented multi-point extension shares the final KZG witness but retains each point's `ell + 1` pre-witness commitments: `t(ell + 1) + 1` G1.

## Encoding and quotient identity

For `N = 2^ell`, Jolt's evaluation table is committed verbatim:

```text
f_hat(X) = sum_{i=0}^{N-1} f(bits(i)) X^i
C_f      = [f_hat(tau)]_1
```

`bits(i)` is Jolt's high-to-low coordinate order. Paper variable `X_k` is Jolt coordinate `ell - 1 - k`; this makes coefficient `i` use paper bit `k` with weight `2^k`. The cross-scheme test pins both this ordering and equality with `HyperKZGScheme::commit`.

Successively eliminate paper variables high-to-low:

```text
q_k(X_0, ..., X_{k-1})
  = R_{k+1}(X_0, ..., X_{k-1}, 1)
  - R_{k+1}(X_0, ..., X_{k-1}, 0)

R_k = R_{k+1}(X_0, ..., X_{k-1}, u_k)

f - v = sum_{k=0}^{ell-1} (X_k - u_k) q_k
```

Commit `q_hat_k = U_k(q_k)`, whose degree bound is `d_k = 2^k - 1`. After challenge `y`:

```text
q_hat(X) = sum_k y^k X^(N - 2^k) q_hat_k(X)
C_q      = [q_hat(tau)]_1
```

Only the upper `N/2` coefficients of `q_hat` can be nonzero; its commitment uses that SRS slice.

## Combined degree/evaluation check

Define:

```text
Phi_m(T) = sum_{i=0}^{2^m-1} T^i

A_k(x) = x^(2^k) Phi_(ell-k-1)(x^(2^(k+1)))
         - u_k Phi_(ell-k)(x^(2^k))

zeta_x(X) = q_hat(X) - sum_k y^k x^(N-2^k) q_hat_k(X)
Z_x(X)    = f_hat(X) - v Phi_ell(x) - sum_k A_k(x) q_hat_k(X)
H(X)      = zeta_x(X) + z Z_x(X) = (X - x) Q(X)
```

The HyperKZG setup contains `N + 1` G1 powers, so §6's degree-enforcing shift is `N_max - (N - 1) = 2`:

```text
pi = [X^2 Q(X)]_1

e(C_H, [tau^2]_2) = e(pi, [tau - x]_2)
```

Verification uses one two-pair multi-pairing. `Phi` suffix products, powers `x^(2^k)`, and every `A_k(x)` take `O(ell)` field operations.

For points `j = 0..t-1`, all quotient and raised-degree commitments precede challenge `rho`; the final polynomial is:

```text
H_multi(X) = sum_j rho^j (zeta_j,x(X) + z Z_j,x(X)).
```

It has one shifted KZG witness and the same two-pair equation. A false component can cancel for at most `t - 1` values of `rho`.

## Proof bytes

BN254 compressed G1 is 32 bytes. Sizes exclude serializer container-length framing, matching the existing HyperKZG 2,560-byte accounting.

| Claim shape | Group elements | Bytes at `ell = 20` |
|---|---:|---:|
| 1 polynomial × 1 point | `ell + 2` | **704** |
| `m` polynomials × 1 point, external RLC | `ell + 2` | **704** |
| 1 polynomial × 2 points | `2(ell + 1) + 1` | **1,376** |
| 1 polynomial × 3 points | `3(ell + 1) + 1` | **2,048** |

No proof field elements. Single-point size is 72.5% below HyperKZG's 2,560 bytes.

## Timings

Criterion release build, 10 Rayon threads, 10 samples. Shared 16 GiB host; lower-contention runs retained. Peak process RSS across the `ell = 21` commit/open-1/open-3/verify run: **1.10 GB**.

| `ell` | Commit | Open 1 point | Open 3 points | Verify |
|---:|---:|---:|---:|---:|
| 20 | 0.622 s | **1.638 s** | **3.648 s** | 1.11 ms |
| 21 | 1.235 s | **3.084 s** | **6.744 s** | 0.85 ms |

HyperKZG comparison supplied for this campaign: 1.35 s / 2.60 s open at `ell = 20 / 21`; Zeromorph is 1.21× / 1.19× slower while removing the evaluation vector.

## EVM verifier sketch

- Single point: one MSM over `ell + 3` G1 inputs (`C_q`, `C_f`, `[1]_1`, and `ell` `C_k`); 23 inputs at `ell = 20`.
- `t` points: one MSM over `t(ell + 1) + 2` G1 inputs; 65 inputs for three points at `ell = 20`.
- G2: compute `[tau - x]_2`; `[tau^2]_2` comes from the verifier key.
- Pairing precompile: one product check containing two pairs.
