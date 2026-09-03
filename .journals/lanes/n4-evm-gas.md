# N4 — D4 EVM verifier gas

Measured 2026-09-02 in `/Volumes/Dev/scratch/wrapper-verifier-gas`.

## Result

The D4 verifier costs **546,555 execution gas on Cancun**. Its exact 1,960-byte dummy payload adds **51,328 intrinsic gas**, for **597,883 gas total**. Istanbul costs **640,990 gas total**.

| Item | Cancun | Istanbul |
|---|---:|---:|
| `vk_x` MSM: 8 `ecMul` + 8 `ecAdd` | 58,513 | 68,204 |
| Groth16 core: one 4-pair pairing | 183,353 | 183,959 |
| Deferred MSM: 24 `ecMul` + 22 `ecAdd` | 183,969 | 211,915 |
| Deferred check: one 2-pair pairing | 114,702 | 115,307 |
| Blake2b-F binding: 7 × 12-round compressions + Fr reduction | 5,443 | 9,696 |
| Verifier dispatch/glue residual | 575 | 581 |
| **Verifier execution** | **546,555** | **589,662** |
| Transaction base | 21,000 | 21,000 |
| Calldata byte charge: 86 zero + 1,874 nonzero | 30,328 | 30,328 |
| **1,960-byte intrinsic subtotal** | **51,328** | **51,328** |
| **Total** | **597,883** | **640,990** |

The component rows come from `gasleft()` spans inside `profileD4`; `verify` in Forge's contract gas report supplies the execution total. Their difference is the residual row. Forge method gas excludes transaction intrinsic gas.

Cancun saves 43,107 execution gas. D4 makes 71 precompile calls; most of the gap is Istanbul's 700-gas call charge versus Cancun's 100-gas warm-precompile access charge: `71 × 600 = 42,600`.

## Variants

The variant totals hold the 1,960-byte payload fixed, isolating verifier execution. Keccak's generated dummy payload has one fewer zero byte, so its intrinsic cost is 51,340 gas.

| Design | Cancun execution | Cancun total | Istanbul execution | Istanbul total |
|---|---:|---:|---:|---:|
| D4: 24-term MSM + Blake2b-F | 546,555 | 597,883 | 589,662 | 640,990 |
| 5-term MSM + Blake2b-F | 401,839 | 453,167 | 422,492 | 473,820 |
| 24-term MSM + Keccak | 541,864 | 593,204 | 580,724 | 632,064 |

The 5-term deferred MSM span is 39,239 gas on Cancun and 44,731 on Istanbul. The Keccak binding span is 745 gas on Cancun and 752 on Istanbul.

## Calldata

| Encoding | Bytes | Zero | Nonzero | Intrinsic gas |
|---|---:|---:|---:|---:|
| Stated packed D4 payload, measured dummy | 1,960 | 86 | 1,874 | 51,328 |
| Stated packed D4 payload, all-nonzero bound | 1,960 | 0 | 1,960 | 52,360 |
| Harness `verify(bytes)` ABI call | 2,052 | 171 | 1,881 | 51,780 |
| 12 KB layer-1 proof, all nonzero | 12,000 | 0 | 12,000 | 213,000 |
| 17 KB layer-1 proof, all nonzero | 17,000 | 0 | 17,000 | 293,000 |

`KB` is decimal. For binary sizes, 12 KiB costs 217,608 intrinsic gas and 17 KiB costs 299,528.

The prompt's counts do not fit one encoding: 24 G1 points plus 24 independent Fr scalars alone occupy 2,304 bytes, while its 1,960-byte breakdown allocates 21 G1 points and eight Fr values. The harness preserves the requested execution shape and 1,960-byte payload: 24 scalar-multiplication calls read 21 encoded G1 slots, with slots 9–11 reused across the two 12-term accumulators; 24 scalar uses are derived from the eight public Fr values. Changing these to 24 distinct calldata slots does not change the precompile or `calldataload` count, but it requires a larger payload.

## Harness

- Solidity `0.8.30+commit.73712a01`, optimizer enabled, 200 runs.
- Foundry `1.7.1` (`4072e48705af9d93e3c0f6e29e93b5e9a40caed8`).
- One contract: `src/WrapperVerifierGas.sol`; one test: `test/WrapperVerifierGas.t.sol`.
- Groth16: eight public-input multiplications/additions followed by one 4-pair call.
- Deferred check: 24 multiplications and 22 additions followed by one 2-pair call; the five-term variant uses five multiplications and four additions.
- Binding: 770 calldata bytes, seven Blake2b-F calls at 12 rounds, then a two-limb 512-bit reduction modulo Fr; Keccak covers the same 770 bytes.
- Every elliptic-curve input is a valid BN254 point. Pairing inputs are point/negated-point pairs, and every verifier call returns true.

`--evm-version` alone did not invalidate Forge's cached compiler target in this installation, so the commands also set `FOUNDRY_EVM_VERSION`; artifact metadata was checked after each clean build.

```sh
~/.config/.foundry/bin/forge init --no-git wrapper-verifier-gas

~/.config/.foundry/bin/forge clean
FOUNDRY_EVM_VERSION=cancun ~/.config/.foundry/bin/forge test --gas-report --evm-version cancun -vv

~/.config/.foundry/bin/forge clean
FOUNDRY_EVM_VERSION=istanbul ~/.config/.foundry/bin/forge test --gas-report --evm-version istanbul -vv

~/.config/.foundry/bin/forge clean
FOUNDRY_EVM_VERSION=cancun ~/.config/.foundry/bin/forge snapshot --evm-version cancun --snap /tmp/wrapper-gas-cancun.snapshot

~/.config/.foundry/bin/forge clean
FOUNDRY_EVM_VERSION=istanbul ~/.config/.foundry/bin/forge snapshot --evm-version istanbul --snap /tmp/wrapper-gas-istanbul.snapshot
```

Snapshot results:

```text
Cancun   WrapperVerifierGasTest:testGasProfiles() (gas: 3412196)
         WrapperVerifierGasTest:testIntrinsicCalldata() (gas: 1608779)
Istanbul WrapperVerifierGasTest:testGasProfiles() (gas: 3753901)
         WrapperVerifierGasTest:testIntrinsicCalldata() (gas: 1726018)
```

Snapshot rows measure the whole test harness; contract gas-report rows above are the verifier figures.
