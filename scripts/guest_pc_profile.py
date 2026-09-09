#!/usr/bin/env python3
"""Symbolize a traced guest's PC profile (tracer `JOLT_PC_PROFILE`).

    # 1. trace with JOLT_PC_PROFILE=prof.txt (and JOLT_BACKTRACE=1 so the
    #    guest ELF keeps its symbols), then:
    guest_pc_profile.py report prof.txt path/to/guest.elf [--top 40]

    # 2. charge leaf functions (memcpy, hashing, ...) to their callers:
    guest_pc_profile.py ranges path/to/guest.elf memcpy memset '<blake2' > ranges.txt
    JOLT_PC_PROFILE=prof.txt JOLT_PC_PROFILE_RANGES=ranges.txt <trace again>
    guest_pc_profile.py callers prof.txt.ra path/to/guest.elf

`llvm-nm` comes from rustup's llvm-tools component.
"""
import argparse
import bisect
import collections
import glob
import os
import subprocess
import sys


def llvm_nm():
    for candidate in glob.glob(os.path.expanduser("~/.rustup/toolchains/*/lib/rustlib/*/bin/llvm-nm")):
        return candidate
    return "llvm-nm"


def symbols(elf):
    out = subprocess.run([llvm_nm(), "-S", "--defined-only", "--demangle", elf], capture_output=True, text=True, check=True).stdout
    syms = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[2].lower() in ("t", "w") and not parts[3].startswith(".L"):
            syms.append((int(parts[0], 16), int(parts[1], 16), parts[3]))
    syms.sort()
    return syms


class Resolver:
    def __init__(self, syms):
        self.syms = syms
        self.starts = [s[0] for s in syms]

    def name(self, pc):
        i = bisect.bisect_right(self.starts, pc) - 1
        return self.syms[i][2] if i >= 0 else "?"


def report(args):
    resolver = Resolver(symbols(args.elf))
    per = collections.Counter()
    total = 0
    for line in open(args.profile):
        pc, rows = line.split()
        rows = int(rows)
        total += rows
        per[resolver.name(int(pc, 16))] += rows
    print(f"total rows {total}")
    for name, rows in per.most_common(args.top):
        print(f"{rows:>12} {100 * rows / total:5.1f}%  {name[: args.width]}")


def ranges(args):
    for start, size, name in symbols(args.elf):
        if any(pattern in name for pattern in args.patterns):
            print(f"{start:x} {start + size:x}")


def callers(args):
    resolver = Resolver(symbols(args.elf))
    per = collections.Counter()
    totals = collections.Counter()
    for line in open(args.profile_ra):
        leaf, caller, rows = line.split()
        rows = int(rows)
        leaf = resolver.name(int(leaf, 16))[:60]
        per[(leaf, resolver.name(int(caller, 16))[: args.width])] += rows
        totals[leaf] += rows
    for leaf, rows in totals.most_common(args.top):
        print(f"== {leaf}: {rows}")
        for (l, c), n in per.most_common():
            if l == leaf and n >= args.min_share * rows:
                print(f"   {n:>11} {100 * n / rows:5.1f}%  {c}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("report")
    p.add_argument("profile")
    p.add_argument("elf")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--width", type=int, default=150)
    p.set_defaults(func=report)
    p = sub.add_parser("ranges")
    p.add_argument("elf")
    p.add_argument("patterns", nargs="+")
    p.set_defaults(func=ranges)
    p = sub.add_parser("callers")
    p.add_argument("profile_ra")
    p.add_argument("elf")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--width", type=int, default=150)
    p.add_argument("--min-share", type=float, default=0.02)
    p.set_defaults(func=callers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
