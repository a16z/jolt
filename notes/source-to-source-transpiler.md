# Source to source transpiler

## Background 

The two crates we are interested in are `crates/jolt-riscv` and `crates/jolt-program`.
The first crate clearly articulates, what is a native Jolt instruction, and what is an expandable risc-v instruction. 
That is, given a riscv instruction as input, jolt often expands given instruction into a sequence of jolt instructions, and then runs those instead.
This process is called bytecode expansion. 
It does this for risc-v instructions and some custom inlines. We do not worry with the inlines for now, lets get the riscv sorted. 
The full expansions are given in `jolt-program` but this crate depends on `riscv` to know when to further expand. 
For example, the risc-v instruction `lw` does not exist in the Jolt-ISA. 
It is replaced by a sequence that involves the instruction `srl`. 
That too is not a native and must be expanded. 
The knowledge of what is native and not native is in `jolt-riscv`.

## Problem Statement

We want to write a source to source transpiler. 
Rust code to do bytecode expansions already exists. 
But we do not want to parse arbitrary rust. 
We want either re-write the code using a structured language, or take the current rust code and load a fixed AST. This is our starting problem.
We will deal with the backends of what language we will transpile to is for later.

