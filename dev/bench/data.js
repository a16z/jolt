window.BENCHMARK_DATA = {
  "lastUpdate": 1740204084605,
  "repoUrl": "https://github.com/a16z/jolt",
  "entries": {
    "Benchmarks": [
      {
        "commit": {
          "author": {
            "email": "33655900+cre-mer@users.noreply.github.com",
            "name": "Jonas Merhej",
            "username": "cre-mer"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "783da5d32010e707f85085d59ae0451f6d8a6b25",
          "message": "fix dependencies in example Cargo.toml (#587)",
          "timestamp": "2025-02-11T14:27:48-08:00",
          "tree_id": "a63a65ad01534ccac2e1972689e6e1796f89462f",
          "url": "https://github.com/a16z/jolt/commit/783da5d32010e707f85085d59ae0451f6d8a6b25"
        },
        "date": 1739314498742,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.9042,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3400024,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7855,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3394324,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.719,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3394188,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6964,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3391632,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6481,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4610940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 63.0816,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 11033940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.55,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3392852,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.4869,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3392300,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.4405,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4659872,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "112157037+Roee-87@users.noreply.github.com",
            "name": "Roy Rotstein",
            "username": "Roee-87"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c93148dd651e2efb3a2a411dad92e88ae4cf7358",
          "message": "refactored from() method for MemoryOp array impl (#591)\n\n* refactored from() method for MemoryOp array impl\n\n* removed instruction type code\n\n* removed duplicate RV32IM::JAL instruction\n\n* ran cargo fmt\n\n* removed instruction_type variable\n\n* removed duplicate SW and replaced with SB",
          "timestamp": "2025-02-20T12:01:22-05:00",
          "tree_id": "c8f8ef8bc9cd514e12db6d3f75257d398853e95e",
          "url": "https://github.com/a16z/jolt/commit/c93148dd651e2efb3a2a411dad92e88ae4cf7358"
        },
        "date": 1740072523039,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8784,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3400004,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.764,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3392008,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.7203,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3394316,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.699,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3394292,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6511,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609676,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 63.1604,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 11042828,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.5387,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3394728,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.4926,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394612,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.4329,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4662520,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mchl.zhu.96@gmail.com",
            "name": "Michael Zhu",
            "username": "moodlezoup"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dae559d08f350797b93747b4a0c7dfa0baef9649",
          "message": "Twist, d=1 (#573)\n\n* Val-evaluation sumcheck\n\n* Read-checking sumcheck\n\n* Combined read/write-checking sumcheck\n\n* Twist e2e\n\n* Benchmark, tracing spans, and optimizations\n\n* use Zipf distribution in benchmark\n\n* Optimize ra/wa materialization and memory allocations\n\n* Switch binding order for second half of Twist read/write-checking sumcheck\n\n* Check for 0s in second half of sumcheck\n\n* Preemptively multiply eq(r, x) by z\n\n* Avoid unnecessary memcpy when materializing val",
          "timestamp": "2025-02-20T13:10:44-05:00",
          "tree_id": "597beed2854d874109035b5eb530d9d9adc82d73",
          "url": "https://github.com/a16z/jolt/commit/dae559d08f350797b93747b4a0c7dfa0baef9649"
        },
        "date": 1740076663614,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8988,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3400316,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7808,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3394336,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.7188,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3394704,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6841,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3394328,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.671,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4611316,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 63.1746,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 11248160,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.5479,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3394976,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.4919,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3392320,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.434,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4702328,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mchl.zhu.96@gmail.com",
            "name": "Michael Zhu",
            "username": "moodlezoup"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "107d93770df1249bb8fc6f0e47b4b3c9d1cdb0b4",
          "message": "Feat/hyrax returns (#594)\n\n* Bring back Hyrax\n\n* Delete BatchType\n\n* Delete CommitShape\n\n* add comment",
          "timestamp": "2025-02-21T09:57:15-05:00",
          "tree_id": "e0d2f6e6f4e7e7c33be1b1a6ef3528cc28afa881",
          "url": "https://github.com/a16z/jolt/commit/107d93770df1249bb8fc6f0e47b4b3c9d1cdb0b4"
        },
        "date": 1740151420991,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8752,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3395596,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7731,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3394196,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.7089,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3394324,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6804,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3394056,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6542,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609720,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 62.9582,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 11026756,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.5351,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3389996,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.4834,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394884,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.439,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4674296,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "benoit.razet@gmail.com",
            "name": "benoitrazet",
            "username": "benoitrazet"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9740195b4c310b5d0ba6cb0bc0102a5cf706e973",
          "message": "Remove unnecessary truncateOverflow subtable (#592)\n\n* Remove unnecessary truncateOverflow subtable\n\nThe truncateOverflow subtable always returns 0 with\nWORD_SIZE = 32 or 64 bits. All the calls to this subtable are\neliminated.\n\n* Add link to CI benchmark results in README\n\n---------\n\nCo-authored-by: James Parker <james@galois.com>",
          "timestamp": "2025-02-21T17:03:18-05:00",
          "tree_id": "6bd23a68d8fc135c5285a0ed3ec6576c3f6ff158",
          "url": "https://github.com/a16z/jolt/commit/9740195b4c310b5d0ba6cb0bc0102a5cf706e973"
        },
        "date": 1740176984046,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8478,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3399764,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7365,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3394308,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6758,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3394196,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6564,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393948,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6251,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609376,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 62.0686,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10928012,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.4958,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3392276,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.4232,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3392300,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.3884,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4675376,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "107961892+sagar-a16z@users.noreply.github.com",
            "name": "Sagar Dhawan",
            "username": "sagar-a16z"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "82dadd08098ab93d61b518f9a0e2f989ebb7155c",
          "message": "add another bench to more closely measure icicle (#595)\n\n* add another bench to more closely measure icicle\n\n* fix: typos",
          "timestamp": "2025-02-21T21:34:59-08:00",
          "tree_id": "5bdded2dac4299abd1dddb3a3d1a2b046f4014a4",
          "url": "https://github.com/a16z/jolt/commit/82dadd08098ab93d61b518f9a0e2f989ebb7155c"
        },
        "date": 1740204084173,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8494,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3399736,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7364,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3394084,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6792,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3394084,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6576,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3394200,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6241,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4611704,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 62.2213,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 11019732,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.5034,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3394740,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.4227,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394616,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.3773,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4684180,
            "unit": "KB",
            "extra": ""
          }
        ]
      }
    ]
  }
}