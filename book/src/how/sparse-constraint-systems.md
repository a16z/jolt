The design of Jolt (the built system) leans heavily on a simple but powerful technique that is not described in the Jolt paper.

Jolt uses the technique to achieve what is sometimes called "a la carte" prover costs: for
each cycle of the VM, the Jolt prover only "pays" for the primitive instruction
that was actually executed at that cycle, regardless of how many instructions are
in the Instruction Set Architecture (in Jolt's case, RISC-V). 
This design will allow Jolt to be extended to richer instruction sets
(which is [equivalent to supporting different pre-compiles](https://a16zcrypto.com/posts/article/understanding-jolt-clarifications-and-reflections/#section--4)) 
without increases in per-cycle prover cost. 

The technique is already incorporated into Jolt in the context of lookups, but it's simplest
to explain in the context of pre-compiles. 
Here is the rough idea. Suppose a pre-compile is implemented via some constraint system (say, R1CS for concreteness). 
As a first attempt at supporting the pre-compile, we can include into Jolt a single, data-parallel constraint system that "assumes" that the pre-compile is executed
at every single cycle that the VM ran for. There are two problems with this. First,
the pre-compile will not actually be executed at every cycle, so we need a way to "turn off"
all the constraints for every cycle at which the pre-compile is not executed. Second, we do not want the prover to "pay" in runtime for 
the constraints that are turned off.

To address both of these issues, for each cycle $j$ we have the prover commit to a binary flag $b_j$ indicating
whether or not that pre-compile was actually executed at that cycle. 

For each cycle $j$ consider the $i$'th constraint that is part of the pre-compile, say: 
$$ \langle a_i, z \rangle \cdot \langle b_i, z \rangle  - \langle c_i, z \rangle = 0.$$

Here, one should think of $z$ as an execution trace for the VM, i.e., a list of everything
that happened at each and every cycle of VM. In Jolt, there are under 100 entries of $z$ per cycle
(though as we add pre-compiles this number might grow).

We modify each constraint by multiplying the left-hand side by the binary flag $b_j$, i.e., 
$$b_j \cdot \left(\langle a_i, z \rangle \cdot \langle b_i, z \rangle  - \langle c_i, z \rangle \right) = 0.$$

Now, for any cycle where this pre-compile is not executed, the prover can simply assign any variables 
in $z$ that are only "used" by the pre-compile to 0. This makes these values totally "free" to commit to
when using any commitment scheme based on elliptic curves (after we switch the commitment scheme to Binius,
committing to 0s will no longer be "free", but we still expect the commitment costs to be so low that it won't matter a huge amount).
These variables might not satisfy the original constraint but will satisfy the modified one
simply because $b_j$ will be set to 0. We say that $b_j$ "turned off'' the constraint. 

Moreover, in sum-check-based SNARKs for R1CS like Spartan, there are prover algorithms dating to [CTY11](https://arxiv.org/abs/1109.6882) and [CMT12](https://dl.acm.org/doi/pdf/10.1145/2090236.2090245?casa_token=HjA6caUU7n0AAAAA:i03m3k8MR9Hz3uPH-ZjZmPL6c0OuIfFJg2Q_zko4G5bh-wL2HdvqLI4M1T186F01DxDfVeMt2pdq5w) 
where the prover's runtime grows only with the number of constraints where the associated flag $b_j$ is not zero. 
So the prover effectively pays nothing at all (no commitment costs, no field work) for constraints
at any cycle where the pre-compile is not actually executed. We call these algorithms "streaming/sparse sum-check proving."
In a happy coincidence, these are the *same* algorithms we expect to use to achieve a streaming prover (i.e., to control prover
space *without* recursion). 

More precisely, the prover time in these algorithms is about $m$ field operations *per round of sum-check*, where $m$ is the number of constraints with associated binary flag
$b_j$ not equal to $0$. But if $n$ is the *total* number of constraints (including those that are turned off), 
we can "switch over" to the standard "dense" linear-time sum-check proving algorithm after enough rounds $i$ pass 
so that $n/2^i \approx m$. In Jolt, we expect this "switchover" to happen by round $4$ or $5$. 
In the end, the amount of extra field work done by the prover owing to the sparsity will only be a factor of $2$ or so.

Jolt uses this approach within Lasso as well. Across all of the primtive RISC-V instructions,
there are about 80 "subtables" that get used. Any particular primitive instruction only needs
to access between 4 and 10 of these subtables. We "pretend" that every primitive instruction
actually accesses all 80 of the subtables, but use binary flags to "turn off" any subtable
lookups that do not actually occur (i.e., that is, we tell the grand product 
in the guts of the Lasso lookup argument for that subtable to ``ignore'' that lookup). 
This is why about 93% of the factors multiplied in Jolt's various grand product arguments are 1. 
The same algorithmic approach ensures that the "turned off lookups" do not increase
commitment time, and that our grand product prover does not pay any field work for the "turned off" subtable lookups. 

There are alternative approaches we could take to achieve "a la carte" prover costs, e.g., [vRAM](https://web.eecs.umich.edu/~genkin/papers/vram.pdf)'s approach
of having the prover sort all cycles by which primitive operation or pre-compile was executed at that cycle
(see also the much more recent work [Ceno](https://eprint.iacr.org/2024/387)).
But the above approach is compatable with a streaming prover, avoids committing to the same data multiple times,
and has other benefits.

We call this technique (fast proving for) "sparse constraint systems". Note that the term sparse here
does not refer to there being the sparsity of the R1CS constraint matrices themselves, 
but rather to almost all of the left-hand sides of the constraints being $0$
when the constraints are evaluated on the valid witness vector $z$.

