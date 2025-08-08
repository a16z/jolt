![Jolt Alpha](imgs/jolt_alpha.png)

[Jolt](https://eprint.iacr.org/2023/1217.pdf) is a zkVM framework built around the [Lasso](https://eprint.iacr.org/2023/1216.pdf) lookup argument.

Jolt powers succinct proofs of execution of programs written in any high-level language. Jolt's sumcheck-based SNARK makes extensive use of multivariate polynomials and commitment schemes. Jolt zkVMs have state-of-the-art prover performance and have substantial room for growth over the coming decades.

Jolt zkVMs have a simple programming model, requiring only 50-100 LOC to implement new VM instructions.

The Jolt codebase currently targets the RISC-V instruction set which is supported by most high-level language compilers, but the code is intended to be extensible and usable by any ISA.

The only property of the ISA that Jolt requires is that each
primitive instruction is "decomposable". This means that evaluating the instruction on one or two 32-bit or 64-bit inputs can be done
via a procedure of the following form:

- decompose each input into, say, 8-bit chunks
- apply one or more specified functions to these chunks (one from each input)
- reconstruct the output of the original instruction from the outputs of the functions operating on the chunks

## Related reading
- [Introducing Lasso and Jolt](https://a16zcrypto.com/posts/article/introducing-lasso-and-jolt/)
- [Understanding Lasso and Jolt](https://a16zcrypto.com/posts/article/building-on-lasso-and-jolt/)


## Background reading
- [Proofs, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)


## Credits
[Lasso](https://eprint.iacr.org/2023/1216.pdf) was written by Srinath Setty, Justin Thaler and Riad Wahby. [Jolt](https://eprint.iacr.org/2023/1217.pdf) was written by Arasu Arun, Srinath Setty, and Justin Thaler.

Jolt was initially forked from Srinath Setty's work on [microsoft/Spartan](https://github.com/microsoft/spartan), specifically the [arkworks-rs/Spartan](https://github.com/arkworks-rs/spartan) fork in order to use the excellent Arkworks-rs prime field arithmetic library. Jolt's R1CS is also proven using a version of Spartan (forked from the [microsoft/Spartan2](https://github.com/microsoft/Spartan2) codebase) optimized for the case of uniform R1CS constraints.
