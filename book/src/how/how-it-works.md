# How it works

This section dives into in the inner workings of Jolt, beyond what an application developer needs to know.
It covers the architecture, design decisions, and implementation details of Jolt.
Proceed if you:

- have been poking around the Jolt codebase and have questions about why something is implemented the way it is

- have read some papers (Jolt, Twist/Shout) and want to know how things fit together in practice

- would like to start contributing to Jolt

This section is comprised of the following subsections:

- [Architecture overview](./architecture/architecture.md)
    - [RISC-V emulation](./architecture/emulation.md)
    - [R1CS constraints](./architecture/r1cs_constraints.md)
    - [Registers](./architecture/registers.md)
    - [RAM](./architecture/ram.md)
    - [Instruction execution](./architecture/instruction_execution.md)
    - [Bytecode](./architecture/bytecode.md)
    - [Batched opening proof](./architecture/opening-proof.md)
- [Twist and Shout](./twist-shout.md)
- [Dory](./dory.md)
- [Optimizations](./optimizations/optimizations.md)
    - [Batched sumcheck](./optimizations/batched-sumcheck.md)
    - [Batched openings](./optimizations/batched-openings.md)
    - [Inlines](./optimizations/inlines.md)
    - [Leveraging uniformity in Spartan](./optimizations/uniform-spartan.md)
    - [Small value optimizations](./optimizations/small-value.md)
    - [EQ optimizations](./optimizations/eq.md)
- [Appendix](./appendix/appendix.md)
    - [Terminology and nomenclature](./appendix/terminology.md)
    - [Multilinear extensions](./appendix/multilinear-extensions.md)
    - [Sumcheck](./appendix/sumcheck.md)
    - [Polynomial commitment schemes](./appendix/pcs.md)
    - [Memory checking and lookup arguments](./appendix/memory-checking.md)
    - [RISC-V](./appendix/risc-v.md)
    - [Jolt 0.1.0](./appendix/jolt-classic.md)
    - [Additional resources](./appendix/resources.md)
