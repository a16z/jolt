# Batched sumcheck

There is a standard technique for batching multiple sumcheck instances together to reduce verifier cost and proof size.

Consider a batch of two sumchecks:

- Sumcheck A is over $n$ variables, and has degree $j$
- Sumcheck B is over $m$ variables, and has degree $k$

If we were to prove them serially, we would have one proof of size $\Theta(nj)$ and one proof of size $\Theta(mk)$ (and the same asymptotics for verifier cost).
If we instead prove them in parallel (i.e. as a batched sumcheck), we would instead have a single proof of size $\Theta(\max(n, m) \cdot \max(j, k))$.

For details, refer to Jim Posen's ["Perspectives on Sumcheck batching"](https://hackmd.io/s/HyxaupAAA).

Our implementation of batched sumcheck is uses the `SumcheckInstance` trait to represent individual instances, and the `BatchedSumcheck` enum to house the batch `prove`/`verify` functions.

## "Bespoke" batching

🚧 These docs are under construction 🚧

👷If you are urgently interested in this specific page, open a Github issue and we'll try to expedite it.👷
