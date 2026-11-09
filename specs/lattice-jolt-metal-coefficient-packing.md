# Resident trace coefficient packing

This candidate replaces the native trace's CPU coefficient-packing scan with
arithmetic over the selectors retained by the Metal commitment. The opening
point, transcript, extension-coordinate order, and CPU fallback remain owned by
the canonical Akita adapter.

For column `c`, block `b`, subring coefficient `j`, ring dimension `D`, and
embedding stride `h = D / s`, the output is

```
Y[c,b,j] = sum_u packing_weight[u]
                 * sum_m position_weight[m] * X[c,b*P+m,j*h+u].
```

`X` is the committed K16 one-hot ring: a row at offset `r` contributes at
coefficient `16*r + hot`. Encoded selector 255 denotes absence; encoded zero
is a committed digit zero. Rows beyond the stored prefix contribute zero.

The initial consumer admits D512/K16 and degree-one encodings that expose their
base scalar through `FpExtEncoding::degree_one_base`. Every valid subring divisor
of 512 uses the same formula. Other extensions and unsupported shapes use the
existing CPU traversal. Canonical output coordinates are
`[column][block][extension coordinate][subring coefficient]`; the sole admitted
extension coordinate is zero.

The proof-bound compute owner checks source identity and borrows its retained
selector windows. Packing does not consume the cache: root folding later takes
the same windows. If retention is absent or its block geometry differs, packing
declines before execution. Device errors after selection propagate.

A gather kernel sums position weights in 1024-position tiles, with one thread
per coefficient. A combine kernel reduces the tiles and multiplies once per
coefficient by its packing weight. This preserves the CPU optimization that
avoids a field multiplication per nonzero. Gather and combine dispatches reuse
one block of partials in command order. Both write their full addressed output;
untouched trailing logical blocks remain zero. No threadgroup scratch is used.

Each partial, output, and position-weight buffer is bounded by 64 MiB. At the
known frozen Fibonacci root geometry (39 columns, 64 blocks, 65,536 positions),
tile partials occupy 20,447,232 bytes. The selected packing subring dimension
will come from the next existing-span profile fields; it is not inferred from
K16 or the NTT geometry.

The gather performs up to sixteen selector comparisons per stored row, with
neighboring lanes reading the same selector. This is an implementation candidate,
not an established speedup. Acceptance requires inclusive same-workload timing,
including weight uploads, allocation, dispatch, and output transfer.

The focused companion test extends the existing one-hot test file using the
independent canonical CPU `OneHotBatchView` packing backend. It covers tile and
block boundaries, workspace reuse across windows, committed and absent zeros,
omitted suffixes, near-modulus challenges, and subring dimensions 64 and 512.
Normal and Metal-validation execution remain required before integration.
