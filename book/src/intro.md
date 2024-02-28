# Intro
![Jolt Alpha](imgs/jolt_alpha.png)

`TODO: Nice intro segment`

## Related reading
- [Introducing Lasso and Jolt](https://a16zcrypto.com/posts/article/introducing-lasso-and-jolt/)
- [Understanding Lasso and Jolt](https://a16zcrypto.com/posts/article/building-on-lasso-and-jolt/)


## Background reading
- [Proofs, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)


## Credits
[Lasso](https://people.cs.georgetown.edu/jthaler/Lasso-paper.pdf) was written by Srinath Setty, Justin Thaler and Riad Wahby. [Jolt](https://people.cs.georgetown.edu/jthaler/Jolt-paper.pdf) was written by Arasu Arun, Srinath Setty, and Justin Thaler.

Jolt was initially forked from Srinath Setty's work on [microsoft/Spartan](https://github.com/microsoft/spartan), specifically the [arkworks-rs/Spartan](https://github.com/arkworks-rs/spartan) fork in order to use the excellent Arkworks-rs prime field arthmetic library. For witness generation Jolt uses Phil Sippl's [circom-witness-rs library](https://github.com/philsippl/circom-witness-rs). The circuits are written with [Circom](https://github.com/iden3/circom). Jolt's R1CS is checked by [our fork](https://github.com/a16z/spartan2) of [microsoft/Spartan2](https://github.com/microsoft/Spartan2), which is optimized to the case of uniform R1CS constraints. Both implementations of Spartan2 use the EF's Privacy Scaling Exploration team's [halo2curves](https://github.com/privacy-scaling-explorations/halo2curves) for prime field arithmetic.