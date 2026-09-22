#!/usr/bin/env python3
"""Independent hashlib vectors for the documented epoch-6 counter framing."""
import hashlib
import json
import struct


def block(domain, context, index):
    framed = (struct.pack('<I', len(domain)) + domain +
              struct.pack('<Q', len(context)) + context + struct.pack('<Q', index))
    return hashlib.blake2b(framed, digest_size=64).hexdigest()


if __name__ == '__main__':
    print(json.dumps({
        'sparse_zero_first_two': [block(b'akita/sparse-challenge/blake2b512/v1', bytes(40), j)
                                  for j in (0, 1)],
        'final_counter': block(b'stream-boundary', bytes([7]), 2**64 - 1),
    }, indent=2))
