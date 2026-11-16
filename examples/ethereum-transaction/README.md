# Ethereum transaction guest

The guest in `guest/` executes one Ethereum mainnet transaction as the first
transaction of its block, the way a stateless block prover would:

1. It decodes the block header, the parent header and the signed transaction
   from RLP, and checks that the parent header hashes to the header's parent
   hash.
2. It verifies the transaction's signature under a supplied public key with
   the secp256k1 inline and derives the sender from that key.
3. It runs the transaction in revm 43 with the mainnet precompiles, under the
   header's block environment. Every account, storage slot and contract code
   the run reads is resolved by keccak256 from the parent header's state root,
   through a sparse Merkle-Patricia trie over the witness.
4. It applies the transaction's changes to the trie and returns the block
   hash, transaction hash, sender, status, gas used, receipt hash and new state
   root.

It is the `ethereum-transaction` workload of the prover benchmark suite
(`scripts/bench/suites.json`). Its expanded bytecode is far larger than the
other suite guests', which is what the suite uses it for.

## What the guest does not check

- The block's system calls (EIP-4788, EIP-2935) run before its first
  transaction and are not executed here. The returned state root is the parent
  state plus this transaction alone, so it is not a value the chain records.
- The signature check does not read the signature's parity bit. Of the two
  keys a signature verifies under, Ethereum selects one by that bit; the guest
  accepts either.
- That the transaction is in the block. The outputs name both hashes, so a
  reader can check this against a node.
- `BLOCKHASH` resolves only the parent block. The guest fails on an older
  block number.
- It applies Osaka rules with the BPO2 blob parameters and rejects blocks
  before BPO2 (2026-01-07). It does not know later forks.

## Fixture

`fixtures/mainnet-26429593-0.rlp` is the witness for the first transaction of
mainnet block 26429593:

| | |
|---|---|
| Block hash | `0xa09b582ab29a09cb17205032e7bbbaf62678bdb0a8263480b38327ec7cbc862c` |
| Transaction | `0xf1a82aee7fdc4effd94853a2a7062c528dae7a3326216aab8ffc27809f82294f` |
| What it does | A contract call that swaps a token for WETH on a Uniswap V2 pair: two ERC-20 `Transfer` logs, `Sync` and `Swap` |
| Gas used | 106489, status 1 |
| Receipt hash | `0xc5e4a88f00abd49be76cd333afcceb842648a0f162f4a738d99a6b8ccab4c4b4` |

The file is the RLP list `[header, parent_header, transaction, public_key,
preimages]`: 67486 bytes, of which 93 trie nodes take 32578 and four contract
codes 33012. `fetch.py` read the block's `debug_executionWitness` and raw
header, transaction and receipt from `https://api.zan.top/eth-mainnet` on
2026-11-16. The generator kept the preimages the transaction resolves and
checked that the guest's receipt hashes to the chain's.

To build a fixture for another block:

```bash
python3 examples/ethereum-transaction/fetch.py RPC_URL BLOCK_NUMBER > capture.json
cargo run --release -p ethereum-transaction -- capture.json \
    examples/ethereum-transaction/fixtures/mainnet-BLOCK_NUMBER-0.rlp
```

The endpoint must serve the `debug` namespace. The generator stops if the
guest's logic does not reproduce the chain's receipt, or if the transaction
needs a trie node the block's witness does not carry. A new fixture also needs
its path in `crates/jolt-prover/src/profile.rs` and its expected outcome in
`guest/src/lib.rs`.

```bash
cargo nextest run -p ethereum-transaction-guest --cargo-quiet
```

runs the guest natively on the fixture and checks the trie against Ethereum's
secure-trie test vectors.
