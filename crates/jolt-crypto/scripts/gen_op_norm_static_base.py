#!/usr/bin/env python3
"""Generate src/challenge/sampler/op_norm/static_base.rs (feature `ont-static-base`).

Exact-integer transcription of `op_norm.rs` `build_tables` (with `pi_enclosure`,
`arctan_inv`, `taylor_cos`, `taylor_sin`, `octant_reduce`, `apply_sign`,
`downscale_round`, `imul`, `iadd`, `isub`, `fdiv`, `cdiv`, `fshr`, `cshr`) for
the D64 selective-L2 operator-norm table: ring degree d = 64 at scale q = 48.
Python integers are unbounded, so every intermediate that Rust keeps in `i128`
is range-checked; a `checked_mul` overflow in Rust is a loop exit here as well.

The output holds the 128-entry `base_cos` and `base_sin` tables as `i64` (the
production path downcasts them to `i64`; the generator asserts they fit) and
`eps_root`. The transcription is not trusted on its own: the
`static_base_equals_build_tables` unit test compares every entry and `eps_root`
with the Rust `build_tables` output.

Usage: gen_op_norm_static_base.py [OUT]   (default: ../src/challenge/sampler/op_norm/static_base.rs)
"""
import hashlib
import struct
import sys
from pathlib import Path

I128_MIN, I128_MAX = -(1 << 127), (1 << 127) - 1
TRIG_SCALE = 61
TRIG_ONE = 1 << TRIG_SCALE
NTERMS = 12
D, Q = 64, 48


def chk(x):
    assert I128_MIN <= x <= I128_MAX, "i128 overflow"
    return x


def fdiv(a, b):
    return chk(a // b)


def cdiv(a, b):
    return chk(-((-a) // b))


def fshr(x, sh):
    return x >> sh


def cshr(x, sh):
    return -((-x) >> sh)


def imul(a, b):
    assert a[0] >= 0 and b[0] >= 0
    return (fshr(chk(a[0] * b[0]), TRIG_SCALE), cshr(chk(a[1] * b[1]), TRIG_SCALE))


def iadd(a, b):
    return (chk(a[0] + b[0]), chk(a[1] + b[1]))


def isub(a, b):
    return (chk(a[0] - b[1]), chk(a[1] - b[0]))


def add_remainder(s, phi2, pw, fact, next_factor):
    next_pow = imul(pw, phi2)
    next_fact = chk(fact * next_factor * (next_factor + 1))
    rem = cdiv(next_pow[1], next_fact)
    return (s[0] - rem, s[1] + rem)


def taylor_cos(phi):
    phi2 = imul(phi, phi)
    s, pw, fact = (TRIG_ONE, TRIG_ONE), (TRIG_ONE, TRIG_ONE), 1
    for n in range(1, NTERMS + 1):
        pw = imul(pw, phi2)
        tn = 2 * n
        fact = chk(fact * (tn - 1) * tn)
        term = (fdiv(pw[0], fact), cdiv(pw[1], fact))
        s = isub(s, term) if n % 2 == 1 else iadd(s, term)
    return add_remainder(s, phi2, pw, fact, 2 * NTERMS + 1)


def taylor_sin(phi):
    phi2 = imul(phi, phi)
    s, pw, fact = phi, phi, 1
    for n in range(1, NTERMS + 1):
        pw = imul(pw, phi2)
        tn = 2 * n
        fact = chk(fact * tn * (tn + 1))
        term = (fdiv(pw[0], fact), cdiv(pw[1], fact))
        s = isub(s, term) if n % 2 == 1 else iadd(s, term)
    return add_remainder(s, phi2, pw, fact, 2 * NTERMS + 2)


def arctan_inv(x):
    total, x_pow, x2, count, n = 0, x, x * x, 0, 0
    while True:
        denom = (2 * n + 1) * x_pow  # Rust: (2n+1).checked_mul(x_pow), None ends the loop
        if not I128_MIN <= denom <= I128_MAX:
            break
        term = TRIG_ONE // denom  # denom > 0: truncation equals floor
        if term == 0:
            break
        total = total + term if n % 2 == 0 else total - term
        count += 1
        n += 1
        nxt = x_pow * x2  # Rust: x_pow.checked_mul(x2), None ends the loop
        if not I128_MIN <= nxt <= I128_MAX:
            break
        x_pow = nxt
    slack = count + 1
    return (total - slack, total + slack)


def pi_enclosure():
    a5, a239 = arctan_inv(5), arctan_inv(239)
    return (16 * a5[0] - 4 * a239[1], 16 * a5[1] - 4 * a239[0])


def octant_reduce(t, d):
    t %= 2 * d
    cos_sign, sin_sign = 1, 1
    if t > d:
        t = 2 * d - t
        sin_sign = -1
    if t > d // 2:
        t = d - t
        cos_sign = -cos_sign
    swap = t > d // 4
    if swap:
        t = d // 2 - t
    return t, swap, cos_sign, sin_sign


def apply_sign(v, sign):
    return (-v[1], -v[0]) if sign < 0 else v


def downscale_round(v, shift):
    lo, hi = fshr(v[0], shift), cshr(v[1], shift)
    entry = fdiv(lo + hi, 2)
    return entry, max(entry - lo, hi - entry)


def build_tables(d, q):
    pi_lo, pi_hi = pi_enclosure()
    shift = TRIG_SCALE - q
    first_octant = []
    for r in range(d // 4 + 1):
        phi = (fdiv(pi_lo * r, d), cdiv(pi_hi * r, d))
        first_octant.append((taylor_cos(phi), taylor_sin(phi)))
    base_cos, base_sin, eps_root = [0] * (2 * d), [0] * (2 * d), 0
    for t in range(2 * d):
        r, swap, cos_sign, sin_sign = octant_reduce(t, d)
        cos_phi, sin_phi = first_octant[r]
        cos_a, sin_a = (sin_phi, cos_phi) if swap else (cos_phi, sin_phi)
        c_val, c_err = downscale_round(apply_sign(cos_a, cos_sign), shift)
        s_val, s_err = downscale_round(apply_sign(sin_a, sin_sign), shift)
        base_cos[t], base_sin[t] = c_val, s_val
        eps_root = max(eps_root, c_err, s_err)
    return base_cos, base_sin, eps_root


def render(base_cos, base_sin, eps_root, digest):
    def table(name, values):
        body = "".join(f"    {v},\n" for v in values)
        return f"pub(super) static {name}: [i64; {len(values)}] = [\n{body}];\n"

    return (
        "//! Generated by `scripts/gen_op_norm_static_base.py`; do not edit.\n"
        "//!\n"
        f"//! Certified `base_cos` / `base_sin` tables of the D64 selective-L2\n"
        f"//! operator-norm predicate (`d = {D}`, `q = {Q}`), as `i64`, and their\n"
        "//! sound error bound. `static_base_equals_build_tables` checks every entry\n"
        "//! against `build_tables`.\n"
        f"//! SHA-256 of the little-endian `i64` entries (cos then sin): {digest}\n"
        "\n"
        f"pub(super) const D: usize = {D};\n"
        f"pub(super) const Q: u32 = {Q};\n"
        f"pub(super) const EPS_ROOT: i128 = {eps_root};\n"
        "\n"
        + table("BASE_COS", base_cos)
        + "\n"
        + table("BASE_SIN", base_sin)
    )


def main(argv):
    out = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "src/challenge/sampler/op_norm/static_base.rs"
    base_cos, base_sin, eps_root = build_tables(D, Q)
    assert all(-(1 << 63) <= v < (1 << 63) for v in base_cos + base_sin), "entry does not fit i64"
    blob = b"".join(struct.pack("<q", v) for v in base_cos + base_sin)
    digest = hashlib.sha256(blob).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(base_cos, base_sin, eps_root, digest))
    print(f"entries,{len(base_cos) + len(base_sin)}")
    print(f"eps_root,{eps_root}")
    print(f"base_tables_sha256_le_i64,{digest}")
    print(f"out,{out}")


if __name__ == "__main__":
    main(sys.argv)
