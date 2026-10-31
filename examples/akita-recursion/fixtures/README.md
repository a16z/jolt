# Saved accumulated-proof replay fixture

`outer-2p30.bundle` contains the saved outer proof, its public I/O, trusted
verifier preprocessing, candidate setup answers and the inherited/output setup
domains. The proof certifies a committed two-proof SHA2-chain verifier: both
inner traces are padded to `2^26`; the outer runs 955,732,363 cycles and is
padded to `2^30`. Its catalog is W8R4/K16 with eight witness units and setup
offloading.

The selected schedule has nine nonterminal folds, one offloading and eight
direct. The outer proof already proves the inherited twelve-claim batch in
batched Stage 3. The external guest recovers that batch only from actual outer
public output and exports eight direct-fold obligations. The expected encoded
output is `outer-2p30.output`.

The original proof and public I/O remain byte-identical. The verifier metadata
was converted through named fields to the current schedule schema, whose new
empty accumulated-prefix demand defaults to zero. Native Check, native Defer,
canonical-output equality and exact original proof/public-I/O byte comparisons
all pass. Hashes and selected-row identity are in `outer-2p30.json`.

Run `bash examples/akita-recursion/saved-outer-benchmark.sh` from the repository
root. Preparation derives the verifier material from the fixed setup seed and
embeds it in the committed external guest. It does not accept a replacement
verifier key or a second copy of the inherited batch through advice.
