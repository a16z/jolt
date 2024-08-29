import { describe, test } from 'node:test';
import {
  ComputeBudgetInstruction,
  ComputeBudgetProgram,
  PublicKey,
  SystemProgram,
  Transaction,
  TransactionInstruction
} from '@solana/web3.js';
import { assert } from 'chai';
import { start } from 'solana-bankrun';

describe('jolt-verifier', async () => {
  // load program in solana-bankrun
  const PROGRAM_ID = PublicKey.unique();
  const context = await start([{ name: 'jolt_svm_verifier', programId: PROGRAM_ID }], []);
  const client = context.banksClient;
  const payer = context.payer;

  test('test HyperKZG', async () => {
    const blockhash = context.lastBlockhash;
    const compute_ix = ComputeBudgetProgram.setComputeUnitLimit({units: 1_400_000});
    // We setup our instruction.
    const ix = new TransactionInstruction({
      keys: [{ pubkey: payer.publicKey, isSigner: true, isWritable: true }],
      programId: PROGRAM_ID,
      data: Buffer.from([0]), // hyperkzg instruction
    });

    const tx = new Transaction();
    tx.recentBlockhash = blockhash;
    tx.add(compute_ix);
    tx.add(ix).sign(payer);

    // Now we process the transaction
    const transaction = await client.processTransaction(tx);
  });

  test('test Grand Product', async () => {
    const blockhash = context.lastBlockhash;
    const compute_ix = ComputeBudgetProgram.setComputeUnitLimit({units: 1_400_000});
    // We setup our instruction.
    const ix = new TransactionInstruction({
      keys: [{ pubkey: payer.publicKey, isSigner: true, isWritable: true }],
      programId: PROGRAM_ID,
      data: Buffer.from([1]), // sumcheck instruction
    });

    const tx = new Transaction();
    tx.recentBlockhash = blockhash;
    tx.add(compute_ix);
    tx.add(ix).sign(payer);

    // Now we process the transaction
    const transaction = await client.processTransaction(tx);
  });
});
