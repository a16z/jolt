# Spartan SVO Component Benchmark Results

## SHA2-Chain Guest

This benchmark breaks down the performance of the Small Value Optimization (SVO) components within the Spartan sumcheck protocol.

### 8 Iterations

| Component             | Time        |
|-----------------------|-------------|
| 1. Precomputation     | 46.338 ms   |
| 2. SVO Rounds         | 83.947 µs   |
| 3. Streaming Round    | 27.262 ms   |
| 4. Remaining Rounds   | 22.771 ms   |

### 16 Iterations

| Component             | Time        |
|-----------------------|-------------|
| 1. Precomputation     | 103.83 ms   |
| 2. SVO Rounds         | 96.073 µs   |
| 3. Streaming Round    | 57.409 ms   |
| 4. Remaining Rounds   | 45.059 ms   |

### 32 Iterations

| Component             | Time        |
|-----------------------|-------------|
| 1. Precomputation     | 451.18 ms   |
| 2. SVO Rounds         | 136.44 µs   |
| 3. Streaming Round    | 105.00 ms   |
| 4. Remaining Rounds   | 83.140 ms   |

### 64 Iterations

| Component             | Time        |
|-----------------------|-------------|
| 1. Precomputation     | 2.8575 s    |
| 2. SVO Rounds         | 144.71 µs   |
| 3. Streaming Round    | 207.61 ms   |
| 4. Remaining Rounds   | 166.62 ms   |

### 128 Iterations

| Component             | Time        |
|-----------------------|-------------|
| 1. Precomputation     | 933.97 ms   |
| 2. SVO Rounds         | 246.54 µs   |
| 3. Streaming Round    | 4.1603 s    |
| 4. Remaining Rounds   | 327.75 ms   |

### 256 Iterations

| Component             | Time        |
|-----------------------|-------------|
| 1. Precomputation     | 2.1128 s    |
| 2. SVO Rounds         | 389.25 µs   |
| 3. Streaming Round    | 2.0125 s    |
| 4. Remaining Rounds   | 762.88 ms   |

### 512 Iterations

| Component             | Time        |
|-----------------------|-------------|
| 1. Precomputation     | 5.0340 s    |
| 2. SVO Rounds         | 537.11 µs   |
| 3. Streaming Round    | 5.3005 s    |
| 4. Remaining Rounds   | 1.6924 s    |

## SHA3-Chain Guest (Post-V4 Refactor of new_with_precompute)

This benchmark breaks down the performance of the Small Value Optimization (SVO) components within the Spartan sumcheck protocol, using the `sha3-chain-guest` program, after the V4 refactoring of `new_with_precompute`.

### 8 Iterations

| Component             | Time        | Change vs Baseline |
|-----------------------|-------------|--------------------|
| 1. Precomputation     | 60.242 ms   | -19.994%           |
| 2. SVO Rounds         | 96.676 µs   | -0.6778%           |
| 3. Streaming Round    | 59.274 ms   | -0.1207%           |
| 4. Remaining Rounds   | 50.328 ms   | +3.1764%           |

### 16 Iterations

| Component             | Time        | Change vs Baseline |
|-----------------------|-------------|--------------------|
| 1. Precomputation     | 2.5975 s    | +69.272%           |
| 2. SVO Rounds         | 134.39 µs   | -3.9968%           |
| 3. Streaming Round    | 127.73 ms   | -7.6858%           |
| 4. Remaining Rounds   | 107.99 ms   | +5.9765%           |

### 32 Iterations

| Component             | Time        | Change vs Baseline |
|-----------------------|-------------|--------------------|
| 1. Precomputation     | 1.2588 s    | -10.028%           |
| 2. SVO Rounds         | 161.20 µs   | -15.111%           |
| 3. Streaming Round    | 322.33 ms   | -4.3104%           |
| 4. Remaining Rounds   | 241.44 ms   | +0.8032%           |

### 64 Iterations

| Component             | Time        | Change vs Baseline |
|-----------------------|-------------|--------------------|
| 1. Precomputation     | 884.54 ms   | -35.544%           |
| 2. SVO Rounds         | 248.19 µs   | +65.971%           |
| 3. Streaming Round    | 2.3349 s    | +803.29%           |
| 4. Remaining Rounds   | 490.72 ms   | +194.52%           |

### 128 Iterations

| Component             | Time        | Change vs Baseline |
|-----------------------|-------------|--------------------|
| 1. Precomputation     | 2.1121 s    | +126.14%           |
| 2. SVO Rounds         | 324.56 µs   | +31.647%           |
| 3. Streaming Round    | 3.3788 s    | -18.787%           |
| 4. Remaining Rounds   | 1.2444 s    | +279.68%           |

### 256 Iterations

| Component             | Time        | Change vs Baseline |
|-----------------------|-------------|--------------------|
| 1. Precomputation     | 4.5529 s    | +115.49%           |
| 2. SVO Rounds         | 520.68 µs   | +33.767%           |
| 3. Streaming Round    | 6.5145 s    | +223.71%           |
| 4. Remaining Rounds   | 2.0154 s    | +164.18%           |

### 512 Iterations

| Component             | Time        | Change vs Baseline |
|-----------------------|-------------|--------------------|
| 1. Precomputation     | 8.5650 s    | +70.143%           |
| 2. SVO Rounds         | 854.65 µs   | +59.118%           |
| 3. Streaming Round    | 17.550 s    | +231.11%           |
| 4. Remaining Rounds   | 6.1683 s    | +264.46%           | 