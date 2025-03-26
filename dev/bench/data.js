window.BENCHMARK_DATA = {
  "lastUpdate": 1743003372817,
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
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "42a210aec1995fac7e1d4ecfc1b2e6afbeb1cd94",
          "message": "Bump openssl from 0.10.68 to 0.10.70 in /jolt-evm-verifier/script (#584)\n\nBumps [openssl](https://github.com/sfackler/rust-openssl) from 0.10.68 to 0.10.70.\n- [Release notes](https://github.com/sfackler/rust-openssl/releases)\n- [Commits](https://github.com/sfackler/rust-openssl/compare/openssl-v0.10.68...openssl-v0.10.70)\n\n---\nupdated-dependencies:\n- dependency-name: openssl\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2025-02-22T16:18:13-05:00",
          "tree_id": "56ef8fca309d6680e38562a2ded5953813a47c91",
          "url": "https://github.com/a16z/jolt/commit/42a210aec1995fac7e1d4ecfc1b2e6afbeb1cd94"
        },
        "date": 1740260705721,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8446,
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
            "value": 1.7313,
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
            "value": 1.6815,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3393944,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6603,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3391744,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6225,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609244,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 62.3933,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10898076,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.5087,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3392296,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.4141,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394496,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.3901,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4652860,
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
          "id": "2cac3800a416a949644f5b5d908b90ebe45c382e",
          "message": "Fix size parameters for sha2-chain (#598)",
          "timestamp": "2025-02-26T15:16:05-05:00",
          "tree_id": "00d637d2cb953b9a898dac1c73074f46b8e270bd",
          "url": "https://github.com/a16z/jolt/commit/2cac3800a416a949644f5b5d908b90ebe45c382e"
        },
        "date": 1740602564714,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8448,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3399924,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7338,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3391760,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.681,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3394068,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6584,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393668,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6207,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609468,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 62.4934,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 11044684,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.5007,
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
            "value": 3.4142,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394496,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.4179,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4676868,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "way-scarp4u@icloud.com",
            "name": "hexcow",
            "username": "hexcow"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "170ae18f8f148e0247cfab60f6df356e9685223a",
          "message": "chore: fix issue with missing mode parameter in the function (#599)\n\n* chore: fix issue with missing mode parameter in the function\n\n* Update jolt-core/benches/commit.rs\n\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>\n\n---------\n\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>",
          "timestamp": "2025-02-27T15:11:43-05:00",
          "tree_id": "702a2ad34a62a80c15bcc42582638b4fe843030c",
          "url": "https://github.com/a16z/jolt/commit/170ae18f8f148e0247cfab60f6df356e9685223a"
        },
        "date": 1740688689265,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8456,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3399932,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7391,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3393936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6739,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3393816,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6449,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393552,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.617,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609264,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 62.1843,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10997508,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.4905,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3394592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.4239,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3389724,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.3893,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4679100,
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
          "id": "d5d65251424ee888a9affc7b66d26f307969ced9",
          "message": "Adding precompile fields for RvTraceRow struct (#600)\n\n* updated instructions with new fields for RvTraceRow struct\n\n* fixed clippy errors\n\n* fixed lhu.rs test",
          "timestamp": "2025-02-28T08:15:40-05:00",
          "tree_id": "0e7c68b7bb0f98418b483f999437df99ad8decdc",
          "url": "https://github.com/a16z/jolt/commit/d5d65251424ee888a9affc7b66d26f307969ced9"
        },
        "date": 1740750197968,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8632,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3397664,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7405,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3393692,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6882,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3391500,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6821,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6408,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4613112,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 62.4986,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10836220,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.5087,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3394596,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.4286,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394376,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.3831,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4673092,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "8093171+sergey-melnychuk@users.noreply.github.com",
            "name": "Sergey Melnychuk",
            "username": "sergey-melnychuk"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ba28fea5e7909477a28f3bd426c5bd423c288291",
          "message": "chore: fix typos in code & docs, add CI job (#597)\n\n* chore(typos): add exclusions & run `typos -w`\n\n* chore(typos): fix remaining typos manually\n\n* chore(ci): add `typos` job\n\n* fixup: undo `numer -> number` fix\n\n* fixup: undo `FLE -> FILE` fix\n\n* chore(ci): fix `typos` job\n\n---------\n\nCo-authored-by: sergey-melnychuk <sergey-melnychuk@users.noreply.github.com>",
          "timestamp": "2025-03-03T13:34:53-05:00",
          "tree_id": "650295e2301b77cba53b8c4eafd34616f8749b97",
          "url": "https://github.com/a16z/jolt/commit/ba28fea5e7909477a28f3bd426c5bd423c288291"
        },
        "date": 1741028487937,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8867,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3400032,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7316,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3394180,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6829,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3394200,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6608,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3394180,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6372,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609536,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 62.095,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 11135804,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.5735,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3387288,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.4184,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394728,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.3729,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4676488,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "8504041+arasuarun@users.noreply.github.com",
            "name": "Arasu Arun",
            "username": "arasuarun"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ad83ce7d998d14440ccfb449822cb65ace808de6",
          "message": "The Return of the King (#530)\n\n* port eq_plus_one; sumcheck test\n\n* finished\n\n* cleanup\n\n* minor\n\n* clippy\n\n* cargo fmt\n\n* cargo clippy and fmt\n\n* unused import doh\n\n* Optimize computation of bind_z, bind_shift_z, and bind_z_ry_var (#605)\n\n* Optimize computation of bind_z, bind_shift_z, and bind_z_ry_var\n\n* Refactor\n\n* review nits\n\n* more cleanup\n\n* cargo fmt\n\n---------\n\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>",
          "timestamp": "2025-03-05T13:15:26-05:00",
          "tree_id": "f092d537a8f7cbff110186ba63abf9e6bbe18571",
          "url": "https://github.com/a16z/jolt/commit/ad83ce7d998d14440ccfb449822cb65ace808de6"
        },
        "date": 1741200135365,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8349,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3399848,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7304,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3393936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6813,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3394272,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6552,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6267,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609788,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 53.9525,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8912324,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.4083,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3394724,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.2045,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3391924,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.3092,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4676980,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "8504041+arasuarun@users.noreply.github.com",
            "name": "Arasu Arun",
            "username": "arasuarun"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b22e78773cb84817137a510fc0fe06ffa928d138",
          "message": "Rename non-uniform constraints to cross-step constraints (#609)\n\n* rename non-uniform constraints to cross-step constraints\n\n* cargo fmt\n\n* change NonUniform structs",
          "timestamp": "2025-03-07T12:40:16-05:00",
          "tree_id": "8356920f8b1f94689f28407ee7c21b1930bebe27",
          "url": "https://github.com/a16z/jolt/commit/b22e78773cb84817137a510fc0fe06ffa928d138"
        },
        "date": 1741370825806,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8347,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3399932,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7315,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3394076,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6841,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3391796,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.654,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393808,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6363,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609380,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 54.6179,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8781516,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.4023,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3391924,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.206,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394352,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.3002,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4672992,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "89341729c51142036a12392315c4fcb9ea85fbe1",
          "message": "Bump ring from 0.17.8 to 0.17.13 in /jolt-evm-verifier/script (#610)\n\nBumps [ring](https://github.com/briansmith/ring) from 0.17.8 to 0.17.13.\n- [Changelog](https://github.com/briansmith/ring/blob/main/RELEASES.md)\n- [Commits](https://github.com/briansmith/ring/commits)\n\n---\nupdated-dependencies:\n- dependency-name: ring\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2025-03-11T12:29:49-04:00",
          "tree_id": "3e94946d0c079b83883ff0e9a1964fd2d3e4340d",
          "url": "https://github.com/a16z/jolt/commit/89341729c51142036a12392315c4fcb9ea85fbe1"
        },
        "date": 1741712195266,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8333,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3399920,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7451,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3394320,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6925,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3394248,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6544,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393924,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6293,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609300,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 54.4592,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8847568,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.4086,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3394340,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.2084,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394480,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.3014,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4676408,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "0xminds@gmail.com",
            "name": "minds",
            "username": "0xminds"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dd81340637a51ddfb382c57237e9cd05f548555f",
          "message": "fix: Fix incorrect bit shift in read_doubleword function (#614)",
          "timestamp": "2025-03-11T20:39:10-04:00",
          "tree_id": "87be11e02a02a4d1d9f6525fc88a6c30a8658113",
          "url": "https://github.com/a16z/jolt/commit/dd81340637a51ddfb382c57237e9cd05f548555f"
        },
        "date": 1741741536812,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.84,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3399784,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7251,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3394076,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6728,
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
            "value": 1.6523,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393780,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6343,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609408,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 53.9587,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8983640,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.4092,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3392184,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.1986,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394720,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.3034,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4673156,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e4e0a73391d166e968dea578f5077ffb9a4f31c1",
          "message": "Bump ring from 0.17.8 to 0.17.13 (#615)\n\nBumps [ring](https://github.com/briansmith/ring) from 0.17.8 to 0.17.13.\n- [Changelog](https://github.com/briansmith/ring/blob/main/RELEASES.md)\n- [Commits](https://github.com/briansmith/ring/commits)\n\n---\nupdated-dependencies:\n- dependency-name: ring\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2025-03-13T10:39:14-04:00",
          "tree_id": "ac263b002e407e39b224e0b1024ec2a97ea21f62",
          "url": "https://github.com/a16z/jolt/commit/e4e0a73391d166e968dea578f5077ffb9a4f31c1"
        },
        "date": 1741878361869,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8295,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3399788,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.728,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3394188,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6768,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3393940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6661,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393088,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6169,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4611328,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 54.7757,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8928052,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.4107,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3394456,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.2164,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394748,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.3073,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4674988,
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
          "id": "8f88cabab4be25ea2004bfee75f77001a4f4654c",
          "message": "Import Serializable trait in wasm verify (#620)",
          "timestamp": "2025-03-21T12:34:42-04:00",
          "tree_id": "d89cc16955a83d8b24c75e8c2edcc70f2f61911d",
          "url": "https://github.com/a16z/jolt/commit/8f88cabab4be25ea2004bfee75f77001a4f4654c"
        },
        "date": 1742576525671,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.8284,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3399640,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.7341,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3393948,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6749,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3394304,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6544,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3392416,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6214,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609516,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 54.2692,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8750376,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.4155,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3394520,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.2238,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394484,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.3124,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4675248,
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
          "id": "137552a6c583f43a079e88898c2322ff5dc66a7e",
          "message": "Removed unused compiler features (#619)\n\n* Removed unused compiler features\n\n* Unpin nightly version\n\n* clippy\n\n* cargo fmt\n\n* clippy\n\n* cargo update",
          "timestamp": "2025-03-21T13:27:32-04:00",
          "tree_id": "fa06bbd238c174158c5c8527439e16b48c8a83f3",
          "url": "https://github.com/a16z/jolt/commit/137552a6c583f43a079e88898c2322ff5dc66a7e"
        },
        "date": 1742579677172,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7888,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3400560,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6958,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3393736,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6409,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3393692,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6185,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393700,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5914,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4610948,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 52.8942,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8725640,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.3684,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3394204,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.1358,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3393940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.2636,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4657892,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@jeff.org",
            "name": "Noah Citron",
            "username": "ncitron"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "cbefafda952192a7a55e207d94559d854f1a0a89",
          "message": "fix: undefined symbols when compiling guest in host environemnt (#624)",
          "timestamp": "2025-03-25T15:32:08-04:00",
          "tree_id": "f7684d9fcba508d8213078fd8ab2f849a8461426",
          "url": "https://github.com/a16z/jolt/commit/cbefafda952192a7a55e207d94559d854f1a0a89"
        },
        "date": 1742932780425,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7948,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3400184,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6897,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3391544,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6381,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3394112,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6338,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393816,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5969,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4613248,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 52.8008,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8792276,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.3458,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3394480,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.1323,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3391656,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.2592,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4675180,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "g1684774@gmail.com",
            "name": "g1684774",
            "username": "g1684774"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "291e63001ebad5f7679b0811d273f25ba615f8d4",
          "message": "feat: add bmmtv code from ripp (#611)\n\n* feat: add bmmtv code from ripp\n\n* refactor: remove code not used by polynomial commitment\n\n* refactor: rename afgho commitment\n\n* refactor: remove ark-poly and use unipoly from jolt\n\n* refactor: remove more unused code\n\n* refactor: use Transcript instead of Digest\n\n* feat: use jolt srs\n\n* refactor: remove generic gipa, tipa and tipa with ssm\n\n* refactor: remove dummy param\n\n* refactor: remove ssm commitment\n\n* refactor: remove identity and cleanup unused code\n\n* refactor: remove srs\n\n* refactor: remove dhc\n\n* refactor: remove unused code\n\n* refactor: remove inner product trait\n\n* refactor: remove identity commitment\n\n* refactor: remove use copy instead of reference\n\n* refactor: rename functions and clean up\n\n* refactor: simplify files\n\n* refactor: use jolt kzg\n\n* refactor: use variable msm for everything\n\n* style: cargo fmt\n\n* style: cargo clippy\n\n* refactor: update code with requested changes\n\n* fix: negate coeff in extend unipoly\n\n* refactor: move commitment_keys downstream\n\n* refactor: remove rayon feature gate from inner product\n\n* refactor: use PairingOutput for afgho\n\n* refactor: move gipa methods to gipa proof\n\n* refactor: change gipa prove arguments\n\n* refactor: remove comments\n\n* refactor: add symmetry to UnivariatePolynimalCommitment\n\n* docs: update update kzg opening docs\n\n* refactor: use already calculated transcript_inverse\n\n* refactor: use tracing for profiling\n\n* chore: add NOTICE.md to bmmtv\n\n* style: fix clippy and typos\n\n* refactor: use div_ceil",
          "timestamp": "2025-03-26T11:09:20-04:00",
          "tree_id": "bbef6e5184fc6464b63e80e56a66d4e97e1145ee",
          "url": "https://github.com/a16z/jolt/commit/291e63001ebad5f7679b0811d273f25ba615f8d4"
        },
        "date": 1743003371852,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7813,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3400196,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6822,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3394076,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6565,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3393696,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6179,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393564,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5819,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4609020,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 52.7228,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8966260,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.359,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3394304,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.1315,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3394308,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.271,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4671324,
            "unit": "KB",
            "extra": ""
          }
        ]
      }
    ]
  }
}