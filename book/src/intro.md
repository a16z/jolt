![Jolt Alpha](imgs/jolt_alpha.png)

[Jolt](https://eprint.iacr.org/2023/1217.pdf) is a RISC-V zkVM uniquely designed around **sumcheck** and **lookup arguments** (**J**ust **O**ne **L**ookup **T**able).

Jolt is **fast** (state-of-the-art performance on CPU) and relatively **simple** (thanks to its lookup-centric architecture).

This book has the following top-level sections:
1. Intro (you are here)
2. [Usage](./usage/usage.md) (how to use Jolt in your applications)
3. [How it works](./how/how-it-works.md) (how the proof system and implementation work under the hood)
4. [Roadmap](./roadmap/roadmap.md) (what's next for Jolt)

## Related reading

### Papers

[Jolt: SNARKs for Virtual Machines via Lookups](https://eprint.iacr.org/2023/1217) \
Arasu Arun, Srinath Setty, Justin Thaler

[Twist and Shout: Faster memory checking arguments via one-hot addressing and increments](https://eprint.iacr.org/2025/105) \
Srinath Setty, Justin Thaler

[Unlocking the lookup singularity with Lasso
](https://eprint.iacr.org/2023/1216) \
Srinath Setty, Justin Thaler, Riad Wahby


### Blog posts
Initial launch:
- [Releasing Jolt](https://a16zcrypto.com/posts/article/a-new-era-in-snark-design-releasing-jolt/)
- [FAQ on Jolt's initial implementation](https://a16zcrypto.com/posts/article/faqs-on-jolts-initial-implementation/)

Updates:
- Nov 12, 2024 [blog](https://a16zcrypto.com/posts/article/jolt-an-update/) [video](https://a16zcrypto.com/posts/videos/an-update-on-jolts-development-roadmap/)
- Aug 18, 2025 (Twist and Shout upgrade) [blog](https://a16zcrypto.com/posts/article/jolt-6x-speedup/)

### Background
- [Proofs, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)

## Credits
Jolt was initially forked from Srinath Setty's work on [`microsoft/Spartan`](https://github.com/microsoft/spartan), specifically the [`arkworks-rs/Spartan`](https://github.com/arkworks-rs/spartan) fork in order to use the excellent `arkworks-rs` field and curve implementations.
Jolt uses its own [fork](https://github.com/a16z/arkworks-algebra) of `arkworks-algebra` with certain optimizations, including some described [here](./how/optimizations/small-value.md).
Jolt's R1CS is also proven using a version of Spartan (forked from the [microsoft/Spartan2](https://github.com/microsoft/Spartan2) codebase) optimized for Jolt's uniform R1CS constraints.
Jolt uses Dory as its PCS, implemented in [`spaceandtimefdn/sxt-dory`](https://github.com/spaceandtimefdn/sxt-dory).
