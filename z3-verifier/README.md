# Z3 verifier

This crate uses the Z3 SMT solver to try and verify (1) the R1CS/product constraints by the Jolt CPU, (2) the virtual sequences used by Jolt, and (3) the guest `memcpy` and `memmove` in `jolt-platform/src/memory.S`, for every length, address, overlap, and memory content.

The copy-runtime proofs (`copy_runtime::test_copy_runtime_*`) assemble `memory.S` with `clang` and link it with the toolchain's `rust-lld`, so they need `clang` on `PATH`. They take several minutes each.

This is structured as tests, but note that all tests are not expected to succeed. Some hang due to solver limitations, and some fail but due to reasons which are understood (e.g. `NextIsNoop` makes `JAL/JALR` jump destination underconstrained).

Consistency refers to a virtual sequence only allowing a single result for a given input, i.e it not being underconstrained. Correctness is a stronger statement which says that it only allows the correct computation, this may be harder to solve for which is why they are separated.

To see the found examples of contradictions, run `cargo nextest run --no-capture`.

To run you will need z3 installed on your system.
```
# ubuntu/debian
sudo apt-get install libz3-dev clang pkg-config
# fedora/RHEL
sudo dnf install z3-devel clang
# arch linux
sudo pacman -S z3 clang
# macos
brew install z3 pkg-config
```
