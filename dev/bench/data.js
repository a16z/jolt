window.BENCHMARK_DATA = {
  "lastUpdate": 1754679891025,
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
          "id": "e2a68681835b0d32e2150cc3029bd1b2efc91079",
          "message": "(feat) Sparse-dense shout, part 1 (#625)\n\n* temp\n\n* temp2\n\n* temp3\n\n* bug fixes\n\n* verification\n\n* Tracing spans\n\n* Optimize v, w, x binding\n\n* Optimizations\n\n* Progress towards supporting gamma=1\n\n* Optimize tree computation\n\n* Somewhat sketchy hashmap optimization\n\n* temp\n\n* temp2\n\n* Interleave operands to form lookup index\n\n* holy shit it works\n\n* Remove gamma=1 codepaths\n\n* refactor\n\n* Implement sparse-dense methods for OR\n\n* AssertAlignedMemoryAccessInstruction -> AssertHalfwordAlignmentInstruction\n\n* Support eta > 1\n\n* Remove OFFSET const generic from LowBitSubtable\n\n* Add test macros\n\n* SLTU\n\n* New algorithm, works with oR\n\n* SLTU\n\n* Clean up profiling code\n\n* temp\n\n* Add LookupBits\n\n* Slowly but surely\n\n* mulhu\n\n* sub\n\n* xor\n\n* Refactor suffixes\n\n* Refactor prefixes\n\n* Delete old algorithm\n\n* Implement materialize_entry\n\n* Add evaluate_mle tests\n\n* Implement PrefixSuffixDecomposition for more instructions\n\n* More instructions\n\n* movsign and assert_valid_signed_remainder\n\n* Pow2\n\n* Optimize suffix poly computation\n\n* Virtual sequences for shifts\n\n* Optimize SRA virtual sequence using RightShiftPaddingInstruction\n\n* appease CI\n\n* Cleanup + comments\n\n* Add comments\n\n* Fix rem sequence_output",
          "timestamp": "2025-03-26T21:09:28-04:00",
          "tree_id": "52ff7f03fa39b61b4d6a5ba2b90a3dd860b7e765",
          "url": "https://github.com/a16z/jolt/commit/e2a68681835b0d32e2150cc3029bd1b2efc91079"
        },
        "date": 1743039401067,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7796,
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
            "value": 1.6801,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3393696,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6324,
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
            "value": 1.6124,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393540,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5799,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4610700,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 53.5323,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8907496,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.3567,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3393984,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.1378,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3393824,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.2604,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4670304,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jprider63@users.noreply.github.com",
            "name": "JP",
            "username": "jprider63"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4fb71f8366215a2bcda872f3412577d5eaf7e6a2",
          "message": "Streaming polynomial commitments (#629)\n\n* Introduce `StreamingCommitmentScheme` trait with HyperKZG and Hyrax implementations\n\n* Add test for streaming HyperKZG\n\n* Fix offset error in streaming HyperKZG\n\n* Formatting\n\n* Fix warnings and clippy",
          "timestamp": "2025-04-03T14:55:30-04:00",
          "tree_id": "9d1fa18e9816827c359936dbd8f0d851ca6a7762",
          "url": "https://github.com/a16z/jolt/commit/4fb71f8366215a2bcda872f3412577d5eaf7e6a2"
        },
        "date": 1743708195389,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7731,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 3399840,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6741,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 3393676,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6303,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 3393572,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6079,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 3393568,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5793,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 4608692,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 53.4478,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8752508,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.343,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 3391428,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.1217,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 3393824,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.2434,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 4672980,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "marsella@users.noreply.github.com",
            "name": "Marcella Hastings",
            "username": "marsella"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c13b5122dea2e8743c3b15b7af40c0fd2a7496ad",
          "message": "fix prover input in fibonacci (#636)",
          "timestamp": "2025-04-08T13:11:45-04:00",
          "tree_id": "fbfb013e6d3da22ddd06fea6111275b1ff9f7a83",
          "url": "https://github.com/a16z/jolt/commit/c13b5122dea2e8743c3b15b7af40c0fd2a7496ad"
        },
        "date": 1744134488740,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7674,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4606108,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6733,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4598540,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6251,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4600464,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6008,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4598564,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5776,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6982684,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 53.2455,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9969328,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.3212,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4599388,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.1023,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4598856,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.2491,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7034800,
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
          "id": "473bd10fd68229a88899859e86e11570a73810f1",
          "message": "Bump openssl from 0.10.71 to 0.10.72 (#633)\n\nBumps [openssl](https://github.com/sfackler/rust-openssl) from 0.10.71 to 0.10.72.\n- [Release notes](https://github.com/sfackler/rust-openssl/releases)\n- [Commits](https://github.com/sfackler/rust-openssl/compare/openssl-v0.10.71...openssl-v0.10.72)\n\n---\nupdated-dependencies:\n- dependency-name: openssl\n  dependency-version: 0.10.72\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2025-04-10T12:11:42-04:00",
          "tree_id": "40a178a4cefa3116fb339ed3131a833b10e0637f",
          "url": "https://github.com/a16z/jolt/commit/473bd10fd68229a88899859e86e11570a73810f1"
        },
        "date": 1744303776214,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7665,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4609168,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6778,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4598148,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6216,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4598416,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5968,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4598236,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5813,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6984252,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 53.5179,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10183952,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.329,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4599400,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.1023,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4598844,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.2443,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7071056,
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
          "id": "945c0328a313c25d61975c576a17e368bed2421b",
          "message": "Bump openssl from 0.10.71 to 0.10.72 in /jolt-evm-verifier/script (#639)\n\nBumps [openssl](https://github.com/sfackler/rust-openssl) from 0.10.71 to 0.10.72.\n- [Release notes](https://github.com/sfackler/rust-openssl/releases)\n- [Commits](https://github.com/sfackler/rust-openssl/compare/openssl-v0.10.71...openssl-v0.10.72)\n\n---\nupdated-dependencies:\n- dependency-name: openssl\n  dependency-version: 0.10.72\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2025-04-10T12:56:06-04:00",
          "tree_id": "ec4d13e454d9b9b1ccf2b856a2f6a5f626b71c88",
          "url": "https://github.com/a16z/jolt/commit/945c0328a313c25d61975c576a17e368bed2421b"
        },
        "date": 1744306430912,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7709,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4605084,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.689,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4602628,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6155,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4598540,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.6059,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4598824,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5755,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6984844,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 53.5626,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10074900,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.3242,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4599488,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.1189,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4599100,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.2437,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7049884,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "marsella@users.noreply.github.com",
            "name": "Marcella Hastings",
            "username": "marsella"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a92f5df4a2cc3c4d1f7baf75b791e9a87e6e072c",
          "message": "Fix clippy errors (#646)\n\n* fix clippy errors\n\nThese fixes were made with\n$ cargo clippy --fix --all\n$ cargo fmt\n\n* fix clippy errors in generated code\n\nThis fixes new format errors from clippy in the generated code and in\nthe documentation.\n\n* remove deprecated action from CI\n\nThe `actions/cargo` action was deprecated in October 2023. This replaces\nthe two uses with direct calls to `cargo`, which is already available\nvia the `setup-rust-toolchain` action.",
          "timestamp": "2025-04-29T13:53:09-04:00",
          "tree_id": "93dc02a22d772a1f7a9c0d8cb4847db38fad8656",
          "url": "https://github.com/a16z/jolt/commit/a92f5df4a2cc3c4d1f7baf75b791e9a87e6e072c"
        },
        "date": 1745951835573,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7816,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4604872,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.8011,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4600964,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6259,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4598388,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5985,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4601064,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5787,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6978604,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 53.1062,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9918140,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.3207,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4599636,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.1347,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4598900,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.2367,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7038004,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrew@tretyakov.xyz",
            "name": "Andrew Tretyakov",
            "username": "0xAndoroid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ddd152b14e6110a4ed7ca89105d81f9950a90f61",
          "message": "Implement binding scratch space for compact polynomial (#648)\n\n* implement binding scratch space for compact polynomial\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* cargo fmt\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* cargo fmt & clippy\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>",
          "timestamp": "2025-04-29T14:05:59-04:00",
          "tree_id": "25e132283f3b776160ce3f17a1a2aee296527887",
          "url": "https://github.com/a16z/jolt/commit/ddd152b14e6110a4ed7ca89105d81f9950a90f61"
        },
        "date": 1745952629371,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7751,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4604672,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6806,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4598816,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6292,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4600464,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5934,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4598468,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.585,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6993236,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 53.4059,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10132468,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.3284,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4601676,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.1219,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4598964,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.2427,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7032856,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "53157953+markosg04@users.noreply.github.com",
            "name": "Markos",
            "username": "markosg04"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "edce59e951725973f425ff4d04f1ac19de0ece8e",
          "message": "Feat/dense binding scratch space (#649)\n\n* added scratch space optimization for dense poly binding\n\n* small fix\n\n---------\n\nCo-authored-by: Markos Georghiades <53157953+Markos-The-G@users.noreply.github.com>\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>",
          "timestamp": "2025-04-29T14:34:33-04:00",
          "tree_id": "d28fc3c8e6f9744dcdf3365ed740242c2b376998",
          "url": "https://github.com/a16z/jolt/commit/edce59e951725973f425ff4d04f1ac19de0ece8e"
        },
        "date": 1745954330998,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7616,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4610380,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6774,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4598904,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6258,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4598368,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5969,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4598364,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5816,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6978568,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 55.6847,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9278412,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.3347,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4601520,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.097,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4599036,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.2392,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7065332,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "marsella@users.noreply.github.com",
            "name": "Marcella Hastings",
            "username": "marsella"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "28896dc3d5eab0927e01c3e89658b11f154e900c",
          "message": "add CI job to test the `jolt` SDK (#644)\n\n* add CI job to test the `jolt` SDK\n\n* add breadcrumb for book maintainers\n\nThis will hopefully help keep the CI correct by making sure that changes\nto the SDK are correctly propagated.",
          "timestamp": "2025-04-30T22:05:41-04:00",
          "tree_id": "aaf3ef1c6a0e0e09fd8e21a30cfec4f8a3d3ec32",
          "url": "https://github.com/a16z/jolt/commit/28896dc3d5eab0927e01c3e89658b11f154e900c"
        },
        "date": 1746067384002,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7578,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4604540,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6748,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4602908,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.6172,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4598460,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5976,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4598508,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5712,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6984968,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 53.1133,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10052816,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.3237,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4601144,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 3.0937,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4599000,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.2338,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7050264,
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
          "id": "c073e866bb259100d2fa02890835959f96cfd691",
          "message": "fix: guest programs using stdlib (#672)",
          "timestamp": "2025-05-08T17:53:46-04:00",
          "tree_id": "2423898dbc9c87bfef0f10d59508aacc077d715a",
          "url": "https://github.com/a16z/jolt/commit/c073e866bb259100d2fa02890835959f96cfd691"
        },
        "date": 1746743594744,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7418,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4596636,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6289,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4594240,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5759,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4590588,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5695,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4590364,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5359,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6970572,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 48.7172,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10198836,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.2258,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4591280,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.9537,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4594896,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.1542,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7032664,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zouguangxian@gmail.com",
            "name": "Zou Guangxian",
            "username": "zouguangxian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "42ff741ac1fe9d20f2c4ba6d3856fd3432101b37",
          "message": "fix: use paste crate to resolve enum_dispatch inner scope issue (#675)\n\nThe error \"cannot find value `inner` in this scope\" occurs because enum_dispatch\ncreates an inner identifier during macro expansion, but it's not properly scoped.\nThe paste crate helps by creating hygienic identifiers during macro expansion,\nensuring the inner value is available in the correct scope.\n\nThis is a more robust solution than relying on trait definition order, as it\nproperly handles macro hygiene and identifier scoping.",
          "timestamp": "2025-05-11T18:19:18-04:00",
          "tree_id": "16413f7b45771a9ca4ce5cc9e06c8bb819e37b8e",
          "url": "https://github.com/a16z/jolt/commit/42ff741ac1fe9d20f2c4ba6d3856fd3432101b37"
        },
        "date": 1747004363002,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.717,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4596412,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6528,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4590372,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.598,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4590408,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5646,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4596940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5562,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6966020,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 49.7096,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10123424,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.2343,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4594620,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.938,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4590680,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.1155,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7039940,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zouguangxian@gmail.com",
            "name": "Zou Guangxian",
            "username": "zouguangxian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "22306a1181167c835e4d0d9f35340efac257df13",
          "message": "refactor: remove associated_type_defaults feature flag (#674)\n\nThis commit removes the associated_type_defaults feature flag and makes all associated types explicit in the codebase. The changes include:\n\n- Remove #![feature(associated_type_defaults)] from lib.rs\n- Remove default value (= ()) from JoltField::SmallValueLookupTables\n- Remove default values from MemoryCheckingProver associated types:\n  - ReadWriteGrandProduct\n  - InitFinalGrandProduct\n  - ExogenousOpenings\n  - Preprocessing\n  - MemoryTuple\n- Make all implementations of MemoryCheckingProver explicitly specify their associated types\n- Reorder associated type declarations for better readability\n\nThese changes make the code more explicit and remove the dependency on the unstable associated_type_defaults feature, improving compatibility with stable Rust.",
          "timestamp": "2025-05-12T09:10:48-04:00",
          "tree_id": "2290ad7c3ff181fd465837fbbb131268c5619856",
          "url": "https://github.com/a16z/jolt/commit/22306a1181167c835e4d0d9f35340efac257df13"
        },
        "date": 1747057833801,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7203,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4601004,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6383,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4590692,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5758,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4590660,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5607,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4590064,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.563,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6967912,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 49.4199,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10037248,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.2182,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4591492,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.9408,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4590880,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.1376,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7025440,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "amaloz@galois.com",
            "name": "Alex J. Malozemoff",
            "username": "amaloz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f1d25c14cb7d1ad91314b3315dcd7179b4e379ef",
          "message": "Make the `common` crate `no_std`-compliant (#677)\n\nThis is Step 1 in a long line of steps towards making `jolt` `no_std` compliant.",
          "timestamp": "2025-05-13T20:48:28-04:00",
          "tree_id": "fa18681f7e7d07a2a50e3589737c56dc1ff776d0",
          "url": "https://github.com/a16z/jolt/commit/f1d25c14cb7d1ad91314b3315dcd7179b4e379ef"
        },
        "date": 1747186097522,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7099,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4597060,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6323,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4590324,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5884,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4590444,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5623,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4592004,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5348,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6968852,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 48.8934,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9976192,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.218,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4595080,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.9474,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4590644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 2.1299,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7033288,
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
          "id": "2b7f942411c5061656a4b414a91fc7c036d01320",
          "message": "update guest toolchain to rust 1.86.0 (#673)",
          "timestamp": "2025-05-13T20:50:10-04:00",
          "tree_id": "0d98ea3b66253c7bbda1468b8ad9625da62815f3",
          "url": "https://github.com/a16z/jolt/commit/2b7f942411c5061656a4b414a91fc7c036d01320"
        },
        "date": 1747186169608,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7129,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4597596,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6359,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4594592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.584,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4590300,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5886,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4590080,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5348,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6966596,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 48.8975,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10119648,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.2297,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4593064,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.9454,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4590804,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.9028,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7041540,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zouguangxian@gmail.com",
            "name": "Zou Guangxian",
            "username": "zouguangxian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8cb7674941968f163fe07a2bca0e72b917895d39",
          "message": "chore: add devcontainer configuration for Rust development (#664)\n\n* chore: add devcontainer configuration for Rust development\n\n* chore: update Rust devcontainer image from bullseye to bookworm",
          "timestamp": "2025-05-14T12:46:09-04:00",
          "tree_id": "7916ed56aa476d7cc0d0d260591fc889e825fd19",
          "url": "https://github.com/a16z/jolt/commit/8cb7674941968f163fe07a2bca0e72b917895d39"
        },
        "date": 1747243566105,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7346,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4596868,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6274,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4590520,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5873,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4590380,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5594,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4592424,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5481,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6968460,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 48.7414,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10127884,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.2283,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4591244,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.9362,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4592752,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.9018,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7028664,
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
          "id": "dadf839bbc6c8cc9ed2a965e2661d0673f386a6f",
          "message": "fix: remove codegen-units release override to fix aarch64 prover (#679)",
          "timestamp": "2025-05-14T13:54:51-04:00",
          "tree_id": "f91bf4c96d031a54951c1136795266a80c561f8c",
          "url": "https://github.com/a16z/jolt/commit/dadf839bbc6c8cc9ed2a965e2661d0673f386a6f"
        },
        "date": 1747247509471,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7119,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4598584,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6394,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4592632,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.577,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4593904,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5721,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4591936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5475,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6972220,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 48.4542,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10066956,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.2211,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4592996,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.9209,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4592556,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8981,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7044032,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "incio.gusta@gmail.com",
            "name": "Gustavo Inacio",
            "username": "gusinacio"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9372e6ab7fb7735141bfc5a12f65ee25c9a6cf93",
          "message": "feat: add hyperbmmtv (#670)\n\n* feat: add hyperbmmtv\n\n* test: add bmmtv hyper test\n\n* feat: add tracing to hyperbmmtv\n\n* feat: create proof for hyperbmmtv\n\n* feat: verify hyperbmmtv\n\n* fix: clippy and typos\n\n* refactor: use poly length for bivarite split\n\n* feat: add consistency check\n\n* refactor: fix requested changes",
          "timestamp": "2025-05-15T15:41:17-04:00",
          "tree_id": "857d27efda1ec8719fdc590378a5299b1e86f67a",
          "url": "https://github.com/a16z/jolt/commit/9372e6ab7fb7735141bfc5a12f65ee25c9a6cf93"
        },
        "date": 1747340278129,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7117,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4600376,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6349,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4592092,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5786,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4594080,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.563,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4592152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5252,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6974708,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 48.3582,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10048880,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.2205,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4592980,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.9212,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4596860,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8924,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7033444,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "qvd@andrew.cmu.edu",
            "name": "Quang Dao",
            "username": "quangvdao"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3b37a374a93398dbaac8a106394c3a6e3ccb68c2",
          "message": "feat: implement gruen's optimization & more for spartan (#683)\n\n* implement gruen's optimization & more for spartan\n\n* fmt\n\n* fix hyperbmmtv error\n\n* further fix hyperbmmtv\n\n* small change in split eq poly\n\n* fix icicle clippy\n\n* added Arc optimization to eliminate collection time\n\n* fmt + remove in-function tracing spans\n\n* fix test\n\n* fmt\n\n* addressed comments",
          "timestamp": "2025-05-20T17:12:27-04:00",
          "tree_id": "2450e57d58b3a96016678399c2cff8f69a04a264",
          "url": "https://github.com/a16z/jolt/commit/3b37a374a93398dbaac8a106394c3a6e3ccb68c2"
        },
        "date": 1747777815948,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7366,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4598500,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6414,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4593972,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5716,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4592244,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5728,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4591716,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5375,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6968312,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 46.0595,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9315168,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.192,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4593424,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8562,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4592620,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8885,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7037176,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "amaloz@galois.com",
            "name": "Alex J. Malozemoff",
            "username": "amaloz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8c607644fa091c05571a04f21db17361c005848e",
          "message": "Make `tracer` `no_std` compatible (#680)\n\nThis requires a slight change to the API that is exposed to `jolt-core`: The\n`trace` function no longer takes a `PathBuf` (since that's not available in a\n`no_std` environment) and instead takes the elf contents as a `Vec<u8>`. This\nrequires a slight change to the `jolt-core` `host` module to make that API\nchange compatible.",
          "timestamp": "2025-05-23T09:33:09-04:00",
          "tree_id": "820b87547d95699d4f54cd38c764bbeff95dae5b",
          "url": "https://github.com/a16z/jolt/commit/8c607644fa091c05571a04f21db17361c005848e"
        },
        "date": 1748009501258,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7061,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4598620,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6514,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4594312,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5792,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4591956,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5696,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4594008,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5394,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6972188,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 46.8225,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9321060,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.2159,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4592972,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8893,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4592036,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8956,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7027748,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "4633847+protoben@users.noreply.github.com",
            "name": "Ben Hamlin",
            "username": "protoben"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0d2a747aae90646cc66236baa2732f81b972268c",
          "message": "Use lazy_static for ark SMALL_VALUE_LOOKUP_TABLES initialization (#689)",
          "timestamp": "2025-05-27T10:29:00-04:00",
          "tree_id": "ca3472311ad49210abdb70318ea47a31781fd571",
          "url": "https://github.com/a16z/jolt/commit/0d2a747aae90646cc66236baa2732f81b972268c"
        },
        "date": 1748358364524,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.6926,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4598204,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6303,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4587984,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5766,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4590272,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5554,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4588664,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5381,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6970816,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 46.1958,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9308976,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1976,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4594044,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8692,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4588296,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.903,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7039552,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "qvd@andrew.cmu.edu",
            "name": "Quang Dao",
            "username": "quangvdao"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ffea13a9c0a465e8ab33dc25336a5cec7633354c",
          "message": "feat: small value optimization for Spartan (#690)\n\n* initial commit of everything\n\n* clippy + fmt\n\n* update README and book\n\n* more clippy\n\n* fmt\n\n* fix capacity issue\n\n* removed all in-function timing & tracing stuff\n\n* graphite suggestion\n\n* long lines\n\n* fix regression in streaming round\n\n* revised estimate\n\n* fix typo & clippy",
          "timestamp": "2025-05-27T20:35:06-04:00",
          "tree_id": "6f3b646b4496249394361163d0f7bfc08c727dc0",
          "url": "https://github.com/a16z/jolt/commit/ffea13a9c0a465e8ab33dc25336a5cec7633354c"
        },
        "date": 1748394734417,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7036,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6433,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4588664,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5749,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4587956,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.558,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4590124,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5313,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6968012,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 45.2085,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9228320,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1814,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4593736,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8509,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4588132,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8859,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7027808,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "4633847+protoben@users.noreply.github.com",
            "name": "Ben Hamlin",
            "username": "protoben"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2d8ba50ea8c4e8d64038b119e3029c205a9ca751",
          "message": "Add a tool for Lean frontend extraction (#684)\n\n* Extract ASTs for each subtable MLE and output a representation compatible with the ZkLean library.\n\nWe do this by implementing `JoltField` using a type, `MleAst`, that constructs an AST of the computation computed. We then run the `evaluate_mle` function for each implementor of `LassoSubtable`. We print the results as a Lean4 file.\n\nWe test that the ASTs are generated correctly by creating an evaluate function for the AST, then writing property tests that compare the result of extracting and evaluating the AST against running `evaluate_mle` on an existing `JoltField` directly.\n\n* Update subtable output to accommodate new ZkLean API\n\n* Resolve \"Extract ASTs for combining tables\"\n\n* Resolve \"Extract R1CS constraints\"\n\n* Confirmed that we can treat auxiliaries the same as inputs for R1CS\n\n* Resolve \"Extract virtual instructions\"\n\n* Fix lookup_step with latest zkLean\n\nThe changes are:\n- composed tables needed _32 at the end\n- comma interspersed in the array of pair instruction_flags,composed_tables\n- add a constrainEq between result of mux_lookup and witness LookupOutput\n\n* Generate Witnessable instance for JoltR1CSInputs\n\n* Resolve \"Split extraction up into multiple modules\"\n\n* Add derive(Debug) to LassoSubtable impls\n\n* Bump ark package versions\n\n* Migrates most of the extraction to using JoltInstruction and LassoSubtable\n\nThis commit gets us most of the way to extracting via JoltInstruction and\nLassoSubtable. However, pretty-printing valid Lean4 identifiers for the\nnames is still todo.\n\n* Remove binius dependency from zklean-extractor\n\n* Add static string conversions for JoltInstruction and LassoSubtable\n\n* Make extracted subtable and instruction names into valid Lean identifiers\n\n* Eliminate proc_macro stuff from zklean-extractor and bump Cargo.lock\n\n* Extract constants from Jolt codebase\n\nJolt uses certain constants for a given instruction set / decomposition\nstrategy / memory layout. Currently, only one set of constants is supported,\nbut there may be more sets in the future (e.g., for 64-bit risc-v). This adds\nthe ability to generalize over those constants and pulls them from the Jolt\ncodebase, rather than hard-coding them in the extractor.\n\n* Add the jolt step function\n\n* Refactor and add docs\n\n* One more clarifying comment\n\n* Rename JoltField to ZKField for zkLean\n\n* Resolve \"Integration tests\"\n\n* Fix clippy errors\n\n* Fix some errors caught by graphite\n\n* Run cargo fmt\n\n* Update Cargo.lock\n\n* Fix some typos and linter errors\n\n* Update zklean-extractor README\n\n* Remove unnecessary phantoms\n\n* Remove ark field initialization from tests\n\nThe lazy_static library is now being used to intialize the small-value lookup\ntables for ark_bn254::Fr, so we don't need to call the initialization function\nmanually in our tests.\n\n---------\n\nCo-authored-by: Benoit Razet <benoit.razet@galois.com>\nCo-authored-by: James Parker <james@galois.com>",
          "timestamp": "2025-05-27T23:09:10-04:00",
          "tree_id": "b357c109fcfc24596bf0f7f299e7b929af9fb83f",
          "url": "https://github.com/a16z/jolt/commit/2d8ba50ea8c4e8d64038b119e3029c205a9ca751"
        },
        "date": 1748404003381,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.6955,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4595340,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.625,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4587776,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5876,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4588056,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5646,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4587988,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5463,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6973420,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 44.6532,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9313320,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1972,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4589740,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.847,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4588424,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8805,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7038056,
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
          "id": "1a6227a98784d9ceaab5433ae1635feda2b67c40",
          "message": "fix: improve emulator memory safety and error reporting (#694)\n\n* fix: improve emulator memory safety and error reporting\n\n* Update tracer/src/emulator/cpu.rs\n\nCo-authored-by: graphite-app[bot] <96075541+graphite-app[bot]@users.noreply.github.com>\n\n* clippy\n\n---------\n\nCo-authored-by: graphite-app[bot] <96075541+graphite-app[bot]@users.noreply.github.com>",
          "timestamp": "2025-05-27T23:22:28-04:00",
          "tree_id": "25192252ca1ebb24634ecb5bb8afe0b1752bc79e",
          "url": "https://github.com/a16z/jolt/commit/1a6227a98784d9ceaab5433ae1635feda2b67c40"
        },
        "date": 1748404846853,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7187,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594912,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6317,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4588056,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5857,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4587892,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5611,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4593948,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.547,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6965960,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 45.6293,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9306008,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.2036,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4588496,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8486,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4591832,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8797,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7047216,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "4633847+protoben@users.noreply.github.com",
            "name": "Ben Hamlin",
            "username": "protoben"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b2ca76cdb3b48a409c8b4e37f19cd32ee13b8031",
          "message": "Update extractor to use new memory config struct (#702)",
          "timestamp": "2025-06-08T17:25:42-04:00",
          "tree_id": "32b1052bc8cf9259593ec59369b06dd7be31b847",
          "url": "https://github.com/a16z/jolt/commit/b2ca76cdb3b48a409c8b4e37f19cd32ee13b8031"
        },
        "date": 1749420178225,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7011,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4596176,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6272,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4587596,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5704,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4587756,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5568,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4588340,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5335,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6974632,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 44.8125,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9301680,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1711,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4594976,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.816,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4590288,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8928,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7028912,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zouguangxian@gmail.com",
            "name": "Zou Guangxian",
            "username": "zouguangxian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "574dc265cea8ae525cfb039acd7a64a9840fced9",
          "message": "feat: add RISC-V emulator with ELF file execution and signature generation (#686)\n\n* feat: add RISC-V emulator with ELF file execution and signature generation\n\nThis commit introduces a new RISC-V emulator in the `tracer` crate, allowing execution of ELF files and optional signature file generation. Key changes include:\n\n- New `jolt-emu` binary defined in `Cargo.toml`.\n- Implementation of the emulator logic in `src/main.rs`, including command-line argument parsing using `clap`.\n- Updates to `Cargo.lock` to include new dependencies: `clap` and `tracing-subscriber`.\n- Refactoring of the `Emulator` struct to manage signature addresses and implement a method for writing signatures.\n- Enhancements to ELF section handling in `elf_analyzer.rs` for improved section address retrieval.\n\nThis feature expands the functionality of the project, enabling users to run RISC-V programs and capture execution signatures.\n\n* refactor: remove unused `find_section_addr` method from `elf_analyzer.rs`\n\n* doc: explain signatures in emulator",
          "timestamp": "2025-06-08T17:27:27-04:00",
          "tree_id": "b09e594434944d384c09ec31b59f5fbbe418bf5d",
          "url": "https://github.com/a16z/jolt/commit/574dc265cea8ae525cfb039acd7a64a9840fced9"
        },
        "date": 1749420301113,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.6908,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594464,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6216,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4587816,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5706,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4589932,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5516,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4587556,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5275,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6970480,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 44.4251,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9274692,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1735,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4590824,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8143,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4588664,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8771,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7058272,
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
          "id": "592805d626d29376547ef97ae872e699a4bc125b",
          "message": "fix: get clippy back to life (#705)",
          "timestamp": "2025-06-10T12:28:49-04:00",
          "tree_id": "bccba5ddeca69b1ca7c1a435f352915d720d6075",
          "url": "https://github.com/a16z/jolt/commit/592805d626d29376547ef97ae872e699a4bc125b"
        },
        "date": 1749575175786,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.694,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594300,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6219,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4590308,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5726,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4589704,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5511,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4587892,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5268,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6968188,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 44.6253,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9312212,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1845,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4588720,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8266,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4588236,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8806,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7027824,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "1141361+emlazzarin@users.noreply.github.com",
            "name": "Eddy Lazzarin",
            "username": "emlazzarin"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "eb070a2d1e751938499fa64922eb064165161791",
          "message": "updates license from MIT to MIT + Apache 2.0 (#706)\n\n* updates license from MIT to MIT + Apache 2.0\n\n* restores Microsoft to MIT license",
          "timestamp": "2025-06-11T09:02:30-04:00",
          "tree_id": "baffab3e5862c126ba4a3381275b1c7f70138043",
          "url": "https://github.com/a16z/jolt/commit/eb070a2d1e751938499fa64922eb064165161791"
        },
        "date": 1749649221228,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7111,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594384,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6334,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4587604,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5731,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4591512,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5615,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4588048,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.537,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6963912,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 44.5917,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9239776,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1855,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4588652,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8357,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4592360,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8827,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7057468,
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
          "id": "e8c245607a4ab881f40713bae7cc4119dc117ae4",
          "message": "fix: SH virtual instruction alignment check (#704)\n\n* fix: SH virtual instruction alignment check\n\n* clippy",
          "timestamp": "2025-06-12T12:55:22-04:00",
          "tree_id": "bfadf3f4ae9727cd5c0c717b834ae7dbdba4abe6",
          "url": "https://github.com/a16z/jolt/commit/e8c245607a4ab881f40713bae7cc4119dc117ae4"
        },
        "date": 1749749605977,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7008,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594552,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6223,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4588996,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5747,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4587728,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.552,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4587820,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5308,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6965428,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 44.8731,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9306844,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1638,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4588772,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8377,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4588196,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8956,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7023344,
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
          "id": "be82b55aec0a599af2edf6771d64033b117db30e",
          "message": "fix: option to skip building guest for host architecture (#714)",
          "timestamp": "2025-06-18T17:49:37-04:00",
          "tree_id": "31c5921913e272495e3a8322fd82fbce85ddcf0c",
          "url": "https://github.com/a16z/jolt/commit/be82b55aec0a599af2edf6771d64033b117db30e"
        },
        "date": 1750285654691,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.6991,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594448,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6195,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4587876,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5817,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4592116,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5533,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4587432,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.545,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6967664,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 44.6932,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9304672,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1789,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4590188,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8195,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4590296,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8833,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7035328,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zouguangxian@gmail.com",
            "name": "Zou Guangxian",
            "username": "zouguangxian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e9caa23565dbb13019afe61a2c95f51d1999e286",
          "message": "fix: make jolt pass riscv-arch-tests tests (#715)\n\n* fix: increase TEST_MEMORY_CAPACITY for running the riscv-arch-test\n\n* fix: correct memory address validation in Jolt emulator\n\nThe issue occurred because memory is allocated as 64-bit words (8 bytes each), but the validation\nwas comparing byte addresses directly against the word count. Now properly converts byte addresses\nto word indices using (address >> 3) before comparison.",
          "timestamp": "2025-06-19T09:21:08-04:00",
          "tree_id": "4f10f4796319969c052a3eea28bb9a92684132b3",
          "url": "https://github.com/a16z/jolt/commit/e9caa23565dbb13019afe61a2c95f51d1999e286"
        },
        "date": 1750341529470,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.6902,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4597120,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6207,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4587648,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5731,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4589544,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5605,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4589696,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5503,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6963864,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 44.8656,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9259832,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1964,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4588596,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8936,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4589816,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8874,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7024072,
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
          "id": "85bf51da10efa9c679c35ffc1a8d45cc6cb1c788",
          "message": "Add constraint for LUI (#721)",
          "timestamp": "2025-06-20T12:11:50-04:00",
          "tree_id": "12ceebcfdb6ec4ad8e63fc47bae728315d710a9f",
          "url": "https://github.com/a16z/jolt/commit/85bf51da10efa9c679c35ffc1a8d45cc6cb1c788"
        },
        "date": 1750438138852,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.6957,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594792,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6339,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4590376,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5774,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4587628,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5629,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4587772,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5396,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6974640,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 45.0248,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9307396,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1873,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4588728,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8266,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4588304,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8843,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7029900,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "1141361+emlazzarin@users.noreply.github.com",
            "name": "Eddy Lazzarin",
            "username": "emlazzarin"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bde91811fd1e0bf0d3966c174547ba428af3dee4",
          "message": "Merge pull request #724 from a16z/license-update\n\nadds a licensing section to README",
          "timestamp": "2025-06-24T13:49:06-07:00",
          "tree_id": "ef8b720e7effa0455bedbe74a724b97803284bc2",
          "url": "https://github.com/a16z/jolt/commit/bde91811fd1e0bf0d3966c174547ba428af3dee4"
        },
        "date": 1750800377139,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.6917,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594996,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6265,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4587432,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5752,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4587908,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5644,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4592096,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5325,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6964112,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 44.7496,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9315208,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1866,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4590800,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8371,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4588596,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8865,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7033800,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "1141361+emlazzarin@users.noreply.github.com",
            "name": "Eddy Lazzarin",
            "username": "emlazzarin"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b70c724f026ac9e0b20388e1ac2764a37bdbaeff",
          "message": "corrects various typos and errata in documentation (#725)\n\n* two typos\n\n* corrects various errata\n\n* fix GKR link\n\n* Update book/src/future/groth-16.md\n\nCo-authored-by: graphite-app[bot] <96075541+graphite-app[bot]@users.noreply.github.com>\n\n* Update book/src/future/groth-16.md\n\nCo-authored-by: graphite-app[bot] <96075541+graphite-app[bot]@users.noreply.github.com>\n\n* broken link\n\n---------\n\nCo-authored-by: graphite-app[bot] <96075541+graphite-app[bot]@users.noreply.github.com>",
          "timestamp": "2025-06-25T10:06:02-04:00",
          "tree_id": "0f399e8e05c916c1ca0a6e0670a32476f5c1c9af",
          "url": "https://github.com/a16z/jolt/commit/b70c724f026ac9e0b20388e1ac2764a37bdbaeff"
        },
        "date": 1750862642070,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7117,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594264,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6332,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4591828,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5838,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4592312,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5848,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4587836,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5442,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6963996,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 45.9901,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9253040,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1849,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4589356,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.9223,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4588240,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8873,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7031492,
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
          "id": "54d86e9a78358d4fad43c19b46c10b7702e71484",
          "message": "fix: docs update to reflect real status of gpu support (#730)",
          "timestamp": "2025-06-27T13:07:05-04:00",
          "tree_id": "722a3d446f0787d401b551b7cbc00436a9947837",
          "url": "https://github.com/a16z/jolt/commit/54d86e9a78358d4fad43c19b46c10b7702e71484"
        },
        "date": 1751046278116,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.7106,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594500,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6309,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4587996,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5773,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4588032,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5571,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4587752,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5388,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6963836,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 45.6199,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9307788,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1856,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4589024,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8538,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4590512,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8872,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7024584,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "4633847+protoben@users.noreply.github.com",
            "name": "Ben Hamlin",
            "username": "protoben"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2343e36fc541b9b890317c5e155000430a01a72d",
          "message": "Make example commands in extractor readme run out of the box (#736)\n\nPreviously, the example commands in the zklean extractor readme required some\nmodifications to their arguments in order to run, which was confusing. In\naddition, the readme included some non-standard ways of running the extractor\nand didn't include the required Lean commands. This commit clarifies the readme\nby\n* including only the standard way of running the extractor;\n* adding in the Lean commands required for compilation; and\n* ensuring that all commands run when copied verbatim into the command line.",
          "timestamp": "2025-07-01T21:27:43-04:00",
          "tree_id": "be632c06c28275d8db0b3d91761139b8e190efca",
          "url": "https://github.com/a16z/jolt/commit/2343e36fc541b9b890317c5e155000430a01a72d"
        },
        "date": 1751421898626,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.6909,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594292,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6263,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4587368,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5793,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4586696,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5554,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4589848,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5223,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6970496,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 45.0891,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9310032,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1826,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4588688,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8317,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4588736,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8853,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7036044,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "125606576+donatik27@users.noreply.github.com",
            "name": "Noisy",
            "username": "donatik27"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "42de0ca1f581dd212dda7ff44feee806556531d2",
          "message": "update actions/checkout  to v4 (#752)\n\n* Update deploy-mdbook.yml\n\n* Update rust.yml",
          "timestamp": "2025-07-08T10:39:55-04:00",
          "tree_id": "ee1d0af4cf213cb4592893f7ee8f300fe73def3f",
          "url": "https://github.com/a16z/jolt/commit/42de0ca1f581dd212dda7ff44feee806556531d2"
        },
        "date": 1751987953284,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 1.6988,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 4594344,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 1.6591,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 4590152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 1.5958,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 4588500,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 1.5692,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 4587772,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.5623,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 6963748,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 45.8586,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9313592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 2.1944,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 4590860,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 2.8723,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 4588392,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 1.8989,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 7007436,
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
          "id": "fd3ffb6a35313518175f164c5ac515869df52390",
          "message": "Merge pull request #809 from a16z/refactor/jolt-trait\n\nTwist and Shout",
          "timestamp": "2025-07-30T11:24:07-04:00",
          "tree_id": "16c9c9ed83d33a84a49ea6e0390acd3dbb43414d",
          "url": "https://github.com/a16z/jolt/commit/fd3ffb6a35313518175f164c5ac515869df52390"
        },
        "date": 1753890799944,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.0983,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 333244,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8378,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 329076,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 330232,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0306,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 330604,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6982,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 389492,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 120.8774,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 13396508,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.5758,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 335452,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.5841,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 348200,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 321368,
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
          "id": "0395728f94aeb3808bb3c8ade387c7264de506fe",
          "message": "Merge pull request #812 from a16z/fix/ci-binary-check\n\n[JOLT-150] Fix example project templates",
          "timestamp": "2025-07-30T11:58:11-04:00",
          "tree_id": "08281d8671e32b63a22d71f37727dca33661200f",
          "url": "https://github.com/a16z/jolt/commit/0395728f94aeb3808bb3c8ade387c7264de506fe"
        },
        "date": 1753892749835,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.3739,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 330704,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.6551,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 330608,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 328076,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.5121,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 327444,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6796,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 394312,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 121.8559,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 13469896,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.741,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 337596,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 10.0424,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 388820,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 318408,
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
          "id": "eb2f00510ba74e93e3bb40e97a513312cdf4e435",
          "message": "Merge pull request #808 from a16z/feat/booleanity-logt-gruen\n\nBooleanity log(T) rounds Gruen",
          "timestamp": "2025-07-30T13:03:51-04:00",
          "tree_id": "fc6a3a61604872ec6647c0a67885584e9c48b39a",
          "url": "https://github.com/a16z/jolt/commit/eb2f00510ba74e93e3bb40e97a513312cdf4e435"
        },
        "date": 1753896700698,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 2.9151,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 328376,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.9936,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 327908,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 329056,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0374,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 333688,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.1876,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 395392,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 120.6366,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 13355624,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.804,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 344460,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.1856,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 382144,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 324476,
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
          "id": "eaeeb9a2f2b9962b45df71b65f205d2652749575",
          "message": "Merge pull request #813 from a16z/fix/postcard-deserialize\n\nFix guest return value deserialization",
          "timestamp": "2025-07-30T13:54:59-04:00",
          "tree_id": "62609a932a0763ebe79d618e4a468184aae68305",
          "url": "https://github.com/a16z/jolt/commit/eaeeb9a2f2b9962b45df71b65f205d2652749575"
        },
        "date": 1753899812883,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 2.9252,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 334968,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8367,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 330052,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1083,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 331832,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0566,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 329612,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6938,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 335612,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 121.201,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 13331916,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.771,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 331140,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.0683,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 388140,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 325240,
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
          "id": "799c369584815bc1925db13e553264ecc5b8bd0b",
          "message": "Merge pull request #816 from a16z/fix/stdlib-example\n\nFix UNIMPL instruction address handling",
          "timestamp": "2025-07-30T19:45:05-04:00",
          "tree_id": "ddc80de913f4e373e37f3c2116eddaa4ea34e18c",
          "url": "https://github.com/a16z/jolt/commit/799c369584815bc1925db13e553264ecc5b8bd0b"
        },
        "date": 1753920821229,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 2.9055,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 334040,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8167,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 329972,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.0921,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 326472,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0585,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 330376,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.1907,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 390728,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 120.1896,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 13399412,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.7578,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 331600,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 10.037,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 407592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.3984,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 380708,
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
          "id": "a2eb0ad5bc1b96b73480b2dc4d95199e2efe3a7a",
          "message": "fix: Assumptions about stack and heap locations in MemoryLayout (#815)",
          "timestamp": "2025-07-31T11:37:28-04:00",
          "tree_id": "1a73e0574f5a1253fe3b96bc40d0737f32e45958",
          "url": "https://github.com/a16z/jolt/commit/a2eb0ad5bc1b96b73480b2dc4d95199e2efe3a7a"
        },
        "date": 1753977997964,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.5551,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 339296,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8433,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 323028,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1005,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 327172,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0492,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 331600,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.707,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 334748,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 121.0045,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 13396836,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.7753,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 339344,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 8.9916,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 386688,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.7071,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 336676,
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
          "id": "f520156981cefd97adfa03ca1eeee5c698434d2c",
          "message": "Misceallaneous cleanup (#818)\n\n* clean up print statements\n\n* Remove unused code\n\n* Remove unused codepaths in Dory multi_pair_cached\n\n* Simplify eq(k_m, c) stuff\n\n* NUM_RA_I_VARS -> DTH_ROOT_OF_K\n\n* aesthetic improvements to CommittedPolynomial\n\n* Make use of setup_prover parameter consistent\n\n* Refactor twist sumcheck switch index",
          "timestamp": "2025-07-31T15:24:56-04:00",
          "tree_id": "a56afa2b910bbea22cf164198f425f992e2d636e",
          "url": "https://github.com/a16z/jolt/commit/f520156981cefd97adfa03ca1eeee5c698434d2c"
        },
        "date": 1753991581744,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 2.9113,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 329100,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.2773,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 332772,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1687,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 331128,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0306,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 329020,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6808,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 335864,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 120.547,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 13326676,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.6133,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 336508,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 10.0166,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 363488,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2851,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 384852,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "atretyakov@a16z.com",
            "name": "Andrew Tretyakov",
            "username": "0xAndoroid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bab0aac40b2c1657c12f69c00ceb87365bccfd3e",
          "message": "check claim in bytecode (#820)\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-08-01T11:52:47-04:00",
          "tree_id": "dd40dc10e3ddee57fe0652f262ca0e4640c8bb56",
          "url": "https://github.com/a16z/jolt/commit/bab0aac40b2c1657c12f69c00ceb87365bccfd3e"
        },
        "date": 1754065239750,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.5412,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 339572,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8054,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 322812,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.0854,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 331064,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.031,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 331880,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6821,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 385980,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 120.6528,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 13398256,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.7357,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 338168,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.7982,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 395152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.3542,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 338932,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "atretyakov@a16z.com",
            "name": "Andrew Tretyakov",
            "username": "0xAndoroid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ea3b85ee7ecc4336e380b460c82c6420f28c28a2",
          "message": "[JOLT-175] Fix/verify virtual in proven fully (and prove rd in bytecode) (#821)\n\n* prove rd claim\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* add checks that everything is provern\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-08-01T12:29:48-04:00",
          "tree_id": "2177dd867a91b2df24c9b49e0baf24e0a6385ffb",
          "url": "https://github.com/a16z/jolt/commit/ea3b85ee7ecc4336e380b460c82c6420f28c28a2"
        },
        "date": 1754067461928,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.7556,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 330748,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.4847,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 326672,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.0607,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 332952,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0399,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 329160,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6638,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 339328,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 120.4619,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 13396964,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.7294,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 340268,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.9124,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 346656,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.3297,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 329804,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42178850+0xAndoroid@users.noreply.github.com",
            "name": "Andrew Tretyakov",
            "username": "0xAndoroid"
          },
          "committer": {
            "email": "42178850+0xAndoroid@users.noreply.github.com",
            "name": "Andrew Tretyakov",
            "username": "0xAndoroid"
          },
          "distinct": true,
          "id": "70c77337426615b67191b301e9175e2bb093830d",
          "message": "fix: typo\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-08-01T12:45:27-04:00",
          "tree_id": "718d666c1a80dbba51e14caf2fd03fd12c052d1a",
          "url": "https://github.com/a16z/jolt/commit/70c77337426615b67191b301e9175e2bb093830d"
        },
        "date": 1754068394640,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.504,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 343872,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.4173,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 332736,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.0724,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 327468,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0294,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 328308,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.67,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 341484,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 120.1299,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 13467812,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.1791,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 341684,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.996,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 351604,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2972,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 336608,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "atretyakov@a16z.com",
            "name": "Andrew Tretyakov",
            "username": "0xAndoroid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "20ac6eb526af383e7b597273990b5e4b783cc2a6",
          "message": "fix ram k being too small (#823)\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-08-01T13:39:20-04:00",
          "tree_id": "303e08eb01d38801e9ed9ee56d4237ec1929822f",
          "url": "https://github.com/a16z/jolt/commit/20ac6eb526af383e7b597273990b5e4b783cc2a6"
        },
        "date": 1754071641882,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.501,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 335320,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.7127,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 326516,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.5842,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 332864,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0227,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 329884,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6663,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 336228,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 119.9976,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 13462288,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.7485,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 334764,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 10.0373,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 347516,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.3166,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 347016,
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
          "id": "1d52b0da75475ebe111fc8e295f8e18293004ace",
          "message": "feat: add wrapper crate \"jolt-verifier\" (#822)\n\n* rework features and jolt-core default binary\n\n* add wrapper crate \"jolt-verifier\" to expose verifier api\n\n* fix: tests\n\n* typo",
          "timestamp": "2025-08-01T14:17:10-04:00",
          "tree_id": "738f18f87f7f930b457fcd024b34120c5ed9fb59",
          "url": "https://github.com/a16z/jolt/commit/1d52b0da75475ebe111fc8e295f8e18293004ace"
        },
        "date": 1754074006571,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.8137,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 339972,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8636,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 325764,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.3739,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 328832,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0604,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 324420,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.7657,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 336536,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 120.1488,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 13348088,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.7778,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 332788,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.0635,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 388040,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.6035,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 398120,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "53157953+markosg04@users.noreply.github.com",
            "name": "Markos",
            "username": "markosg04"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b399ff6371a142e7b11a2b6f81c1c25f0cb023a8",
          "message": "Drop sumcheck data structures (#814)\n\n* drop data structures\n\n* trace clone optimization and more drops\n\n* comments\n\n* batching first round log T mem optimization\n\n* pr comments\n\n* cleanup\n\n* booleanity remove option wrappers\n\n* one hot Option wrappers removed\n\n* clippy\n\n* typo",
          "timestamp": "2025-08-01T14:32:06-04:00",
          "tree_id": "a9c13dbcdf219bb7fe9c24ebb73c792ba905bbf6",
          "url": "https://github.com/a16z/jolt/commit/b399ff6371a142e7b11a2b6f81c1c25f0cb023a8"
        },
        "date": 1754074811659,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 2.9114,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 331068,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.7745,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 328920,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.4377,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 327644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0278,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 324192,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.2056,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 335068,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 119.1373,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8750408,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.7869,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 329636,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 10.0653,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 352444,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.3485,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 342620,
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
          "id": "57ea518d6d9872fb221bf6ac97df1456a5494cf2",
          "message": "chore: add the jolt-verifier tests to CI (#825)",
          "timestamp": "2025-08-01T14:45:04-04:00",
          "tree_id": "7f6f2f934ee5e70e3eb2c2a7f89dca5c6c87f800",
          "url": "https://github.com/a16z/jolt/commit/57ea518d6d9872fb221bf6ac97df1456a5494cf2"
        },
        "date": 1754075609683,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.5555,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 339752,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.7855,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 327860,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.0846,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 331008,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.7026,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 332292,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6732,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 383884,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 119.3416,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8743208,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.7444,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 337744,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.0653,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 348216,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.3003,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 345048,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "atretyakov@a16z.com",
            "name": "Andrew Tretyakov",
            "username": "0xAndoroid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0582b2aa4a33944506d75ce891db7cf090814ff6",
          "message": "fix virtual seq for remu (#826)\n\n* fix virtual seq for remu\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* pad ocdesize to dth root of k\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* make sure bases and row have the same length\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* temp: modify muldiv example\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* Add constraint check\n\n* Add muldiv test\n\n* Fix MSM edge case for RLC poly\n\n* Fix SDK panic bug\n\n* Revert \"temp: modify muldiv example\"\n\nThis reverts commit 6738606d1f53a89863557186925e0011f7a40cc6.\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>",
          "timestamp": "2025-08-01T18:05:08-04:00",
          "tree_id": "86d0beb5b75138c7f0d55066867e2537b6cd953b",
          "url": "https://github.com/a16z/jolt/commit/0582b2aa4a33944506d75ce891db7cf090814ff6"
        },
        "date": 1754087606119,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 2.9228,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 329512,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8001,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 326828,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.0891,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 331252,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0254,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 330128,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6775,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 340044,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 118.8718,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8743444,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.5508,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 328596,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 8.9875,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 354612,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.3238,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 389604,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "atretyakov@a16z.com",
            "name": "Andrew Tretyakov",
            "username": "0xAndoroid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "895d94b39e2f04fd7ad1bab6389a8b7a8b2072e5",
          "message": "fix ci (#827)\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-08-01T19:57:53-04:00",
          "tree_id": "935dc1076168d4f0759fc93010a15fb196063ccc",
          "url": "https://github.com/a16z/jolt/commit/895d94b39e2f04fd7ad1bab6389a8b7a8b2072e5"
        },
        "date": 1754094376242,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.8725,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 339832,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.805,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 330296,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1104,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 330364,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0409,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 323648,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6682,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 389920,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 119.1305,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8610400,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.7351,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 339888,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.7431,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 351480,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.3398,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 349152,
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
          "id": "1ddad9e1d81c2a14e1a37042a33cbdf9dd3c70fb",
          "message": "Feat/optimize msm parallelization (#831)\n\n* Point to local\n\n* Dory commit bench\n\n* Use serial MSM within commit_rows\n\n* cargo update",
          "timestamp": "2025-08-04T12:53:54-04:00",
          "tree_id": "cdfd138051e00b0a741f67a3514d54fdc22089e4",
          "url": "https://github.com/a16z/jolt/commit/1ddad9e1d81c2a14e1a37042a33cbdf9dd3c70fb"
        },
        "date": 1754328132175,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 2.9116,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 334816,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.7802,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 328392,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.0889,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 328740,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0166,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 331900,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6609,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 391676,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 118.4067,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 8612172,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.7401,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 332412,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.9456,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 345108,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2971,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 395580,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zouguangxian@gmail.com",
            "name": "Zou Guangxian",
            "username": "zouguangxian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d59219a0633d91dc5dbe19ade5f66f179c27c834",
          "message": "refactor: Refactor MMU to handle JoltDevice absence gracefully (#835)\n\n- Ensured assertions related to effective addresses are only performed when `jolt_device` is available.",
          "timestamp": "2025-08-05T08:07:42-07:00",
          "tree_id": "34192783a76be45d03cedd0838643ca28f1a0673",
          "url": "https://github.com/a16z/jolt/commit/d59219a0633d91dc5dbe19ade5f66f179c27c834"
        },
        "date": 1754408175005,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.8537,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 460764,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.6064,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 466208,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1441,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 454152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0836,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 475104,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.7214,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 671428,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 110.7777,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9643956,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.9361,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 475504,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.0285,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 495408,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2977,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 620920,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zouguangxian@gmail.com",
            "name": "Zou Guangxian",
            "username": "zouguangxian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0369981446471c2ed2c4a4d2f24d61205a2d0853",
          "message": "Fix RISC-V instruction bugs in division, remainder, and multiplication (#834)\n\n* fix: correct remainder sign handling in DIV and REM instructions\n\n* fix: remove incorrect value assignment in VirtualMovsign exec\n\n- Remove the line that overwrites the calculated sign-based value with\n  the original rs1 value\n- This was causing VirtualMovsign to return the original input instead\n  of the intended sign indicator (-1 for negative, 0 for non-negative)\n\n* fix: correct MULHSU virtual sequence\n\n- Document the two's complement encoding relationship:\n  MULHSU(rs1, rs2) = MULHU(rs1_unsigned, rs2) - rs2\n- Replace MULHU with MUL instruction  because we need the lower bits of (-rs2), not the upper bits",
          "timestamp": "2025-08-05T11:22:01-04:00",
          "tree_id": "0be13ae581ba601fcceae227e66b8ec11e7cac5c",
          "url": "https://github.com/a16z/jolt/commit/0369981446471c2ed2c4a4d2f24d61205a2d0853"
        },
        "date": 1754409003735,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.6088,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 475152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8171,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 482056,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1223,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 462800,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.683,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 458504,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6918,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 632336,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 110.0427,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9563368,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.5853,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 494736,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.4478,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 494088,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.275,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 607172,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zouguangxian@gmail.com",
            "name": "Zou Guangxian",
            "username": "zouguangxian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "eafe3176caa9b09a72cdaf2e9d9a40d2903d4083",
          "message": "feat: add btreemap example (#836)\n\nAdd BTreeMap workload with complex memory access patterns (insertions,\ndeletions, range scans) for profiling.",
          "timestamp": "2025-08-06T14:18:39-04:00",
          "tree_id": "1e182f74bc4f7cfc2c4e90f3402d504e900ee448",
          "url": "https://github.com/a16z/jolt/commit/eafe3176caa9b09a72cdaf2e9d9a40d2903d4083"
        },
        "date": 1754506317030,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 2.9125,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 450976,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "btreemap-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "btreemap-mem",
            "value": 3509692,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.0151,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 482676,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.4031,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 469296,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0594,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 475280,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.7117,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 615220,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 110.9966,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9529772,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.5678,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 476600,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.4915,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 496628,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2787,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 603940,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "53157953+markosg04@users.noreply.github.com",
            "name": "Markos",
            "username": "markosg04"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "91b96a4761215a5ede711afc34d9b2b35c64ccc3",
          "message": "Merge pull request #848 from a16z/sagar/fix_cycle_tracking\n\nfix: cycle tracking marker id length",
          "timestamp": "2025-08-07T17:19:17-04:00",
          "tree_id": "61d66ae6b866c47bfd6c49efeb197c0d651708e2",
          "url": "https://github.com/a16z/jolt/commit/91b96a4761215a5ede711afc34d9b2b35c64ccc3"
        },
        "date": 1754603688315,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.4244,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 475216,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "btreemap-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "btreemap-mem",
            "value": 3683588,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.3882,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 464948,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1376,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 476080,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.7072,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 476500,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.7789,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 631464,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 110.6959,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9527616,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.6456,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 491484,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.5923,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 494216,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.3023,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 608244,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "53157953+markosg04@users.noreply.github.com",
            "name": "Markos",
            "username": "markosg04"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "63989bcdb500e3410b5152ae50366da7ce45f0ac",
          "message": "Better batching sumcheck prep parallelism (#845)\n\n* feat: prepare_sumcheck parallelism\n\n* style: use challenge_vector instead\n\n* fix: fixtures\n\n* fix: binaries",
          "timestamp": "2025-08-07T20:03:21-04:00",
          "tree_id": "356d46638d58daf7ebe7cc08d46efdd2575f4dde",
          "url": "https://github.com/a16z/jolt/commit/63989bcdb500e3410b5152ae50366da7ce45f0ac"
        },
        "date": 1754613372194,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.006,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 467744,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "btreemap-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "btreemap-mem",
            "value": 3728508,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.443,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 477104,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1064,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 464080,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.051,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 461020,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.1705,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 655880,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 110.9098,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9429944,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.5771,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 495712,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.5104,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 496348,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2915,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 554880,
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
          "id": "d7c47be8a1b1553cc56e16befca13c90a7ba1922",
          "message": "Feat/verifier memory layout checks (#847)\n\n* Add verifier memory layout checks\n\n* gitignore spike tarballs\n\n* fmt",
          "timestamp": "2025-08-07T20:32:24-04:00",
          "tree_id": "b45a77031ee91e1801bc6016c6a7ccda90226e5a",
          "url": "https://github.com/a16z/jolt/commit/d7c47be8a1b1553cc56e16befca13c90a7ba1922"
        },
        "date": 1754615150411,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.6238,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 459088,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "btreemap-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "btreemap-mem",
            "value": 3646900,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8152,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 478084,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.7716,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 482176,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0865,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 478152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.7239,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 603036,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 111.0941,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9412364,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.5773,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 487812,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.278,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 497812,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2341,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 586188,
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
          "id": "1c64eb5f1e3694ad85a49eda9298e8d815d24d82",
          "message": "Revert \"Better batching sumcheck prep parallelism (#845)\" (#851)\n\nThis reverts commit 63989bcdb500e3410b5152ae50366da7ce45f0ac.",
          "timestamp": "2025-08-07T23:08:18-04:00",
          "tree_id": "ac9de7e5371444f9808b4ee555819794b1418bbe",
          "url": "https://github.com/a16z/jolt/commit/1c64eb5f1e3694ad85a49eda9298e8d815d24d82"
        },
        "date": 1754624501687,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.6454,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 452536,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "btreemap-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "btreemap-mem",
            "value": 3918744,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8377,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 470116,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.124,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 470592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.085,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 467320,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.9249,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 634544,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 110.9942,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9595976,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.6113,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 489940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 8.7474,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 499712,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.3365,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 548592,
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
          "id": "91c0748a1bf3b5d16ff1f8e3dbe12b120473d0d4",
          "message": "fix: add a stack canary to prevent stack overflows  (#849)\n\n* fix: add a stack canary to prevent stack overflows\n\n* fixes writes to bss\n\n* fix: tracer tests\n\n* fix: wasm\n\n* fix: test",
          "timestamp": "2025-08-08T10:41:56-04:00",
          "tree_id": "11b67aec32fa11962697bd210a7ca0b67af44330",
          "url": "https://github.com/a16z/jolt/commit/91c0748a1bf3b5d16ff1f8e3dbe12b120473d0d4"
        },
        "date": 1754666063359,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.539,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 453340,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "btreemap-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "btreemap-mem",
            "value": 3774340,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.4879,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 468812,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1161,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 477700,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0532,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 473604,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.713,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 640712,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 111.9156,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9503936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.6776,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 490324,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.4898,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 497288,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2641,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 599956,
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
          "id": "1c669b0edea32fde173a972f4338cc3cd81e9389",
          "message": "fix: optimize Cycle size to improve tracer memory usage (#852)\n\n* fix: add a stack canary to prevent stack overflows\n\n* fixes writes to bss\n\n* fix: tracer tests\n\n* fix: wasm\n\n* fix: test\n\n* fix: optimize Cycle size to improve tracer memory usage\n\n* fix fmt\n\n* fix: verifier test fixtures\n\n---------\n\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>",
          "timestamp": "2025-08-08T10:53:19-04:00",
          "tree_id": "b4d1de23da1c6050d89132e0ee1f3b8d74e0db37",
          "url": "https://github.com/a16z/jolt/commit/1c669b0edea32fde173a972f4338cc3cd81e9389"
        },
        "date": 1754666779634,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.334,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 466860,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "btreemap-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "btreemap-mem",
            "value": 3829896,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8616,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 483856,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.2153,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 462460,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.053,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 482936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6883,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 650668,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 111.3668,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9258836,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.5377,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 488196,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 8.5775,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 497780,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2212,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 585272,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42227752+omibo@users.noreply.github.com",
            "name": "Omid Bodaghi",
            "username": "omibo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c65a8000e18e072af9eb5126da9e3c0b976ca0e8",
          "message": "Omid/inlines dispatch (#828)\n\n* Implement Precompile instruction and registry to store precompiles\n\n* Add trace generator in VirtualInstructionSequence to the registry, and make it precompile specific\n\n* Test exec() and trace() correct execution from the external crate\n\n* Add sha256 inline as an extended instruction\n\n* Clean up the code\n\n* Rename precompile to inline\n\n* Add sha256_init to inline instruction\n\n* Fix tests in precompile-example\n\n* Implement inlines in an external crate out of the Jolt's workspace\n\n* Set cpu xlen to Xlen::Bit32 in test cases\n\n* Add inlines crate to the project and remove sha256 instruction\n\n* Restructure inlines dir to support separate crates for each inline\n\n* Import sha2_inline crate in sha2_e2d_dory() test\n\n* Merge main into omid/dispatch\n\n* Change variable names and types\n\n---------\n\nCo-authored-by: Omid Bodaghi <omid.bodaghi@Omids-MacBook-Pro.local>",
          "timestamp": "2025-08-08T14:18:56-04:00",
          "tree_id": "3053f2ab55ef19edb45fccbd13514350664cfffc",
          "url": "https://github.com/a16z/jolt/commit/c65a8000e18e072af9eb5126da9e3c0b976ca0e8"
        },
        "date": 1754679085978,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.6353,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 467860,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "btreemap-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "btreemap-mem",
            "value": 3544660,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.4079,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 475780,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.7304,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 471484,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.6867,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 467848,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6914,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 646028,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 111.1575,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9363304,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.5862,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 477700,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.487,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 494628,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.259,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 594396,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "aribiswas3@gmail.com",
            "name": "Ari",
            "username": "abiswas3"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1a98afecb28e491f856d33327f7b3bac38b5df0c",
          "message": "Optimisations/polynomial evaluations (#843)\n\n* feat: ability to track large scalar multiplications\n\n* feat: ability to track large scalar multiplications\n\nChanging the batch polynomial interface before implementing speedups\n\n* feat: ability to track large scalar multiplications\n\nChanging the batch polynomial interface before implementing speedups.\nSplitEq implementation brought in -- adding in examples for counting\nmultiplications, and benches to measure time next.\n\n* feat: ability to track large scalar multiplications\n\nChanging the batch polynomial interface before implementing speedups.\nSplitEq implementation brought in -- adding in examples for counting\nmultiplications, and benches to measure time next.\n\nFor single polynomial evals -- and very sparse, split-eq definitely\nmakes sense. For big batch sizses, it makes no sense.\n\n* feat: ability to track large scalar multiplications\n\nChanging the batch polynomial interface before implementing speedups.\nSplitEq implementation brought in -- adding in examples for counting\nmultiplications, and benches to measure time next.\n\nFor single polynomial evals -- and very sparse, split-eq definitely\nmakes sense. For big batch sizses, it makes no sense.\n\n* feat: ability to track large scalar multiplications\n\nChanging the batch polynomial interface before implementing speedups.\nSplitEq implementation brought in -- adding in examples for counting\nmultiplications, and benches to measure time next.\n\nFor single polynomial evals -- and very sparse, split-eq definitely\nmakes sense. For big batch sizses, it makes no sense.\n\n* feat: ability to track large scalar multiplications\n\nHad the sums flattened which defeats the purpose of the optimisations\n\n* feat: ability to track large scalar multiplications\n\nHad the sums flattened which defeats the purpose of the optimisations.\nMore parallelism concerns -- e2 needs to be parallel.\n\n* feat: ability to track large scalar multiplications\n\nEverything works; except have not changed names\n\n* feat: ability to track large scalar multiplications\n\nEverything works;speedups are done\n\n* feat: ability to track large scalar multiplications\n\nEverything works;speedups are done\n\n* Update jolt-core/src/poly/multilinear_polynomial.rs\r\n\r\nAdding in One hot support\n\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>\n\n* mv from examples to benches\n\nMoved polynomial evaluation to benches as per Michaels suggestion\n\n* tests for tracked mult\n\n* michaels recommendations based on verbosity\n\n* moving inside out evaluate into benches only\n\nWe're using split_eq\n\n* moving unused functions to benches\n\n* fixed the filed mul in ark.rs\n\n* fix doc strings\n\n---------\n\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>",
          "timestamp": "2025-08-08T14:32:44-04:00",
          "tree_id": "3f5b050d003e00550bd7d21e4694e3a23f9b1f04",
          "url": "https://github.com/a16z/jolt/commit/1a98afecb28e491f856d33327f7b3bac38b5df0c"
        },
        "date": 1754679890205,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.0612,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 452332,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "btreemap-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "btreemap-mem",
            "value": 3529860,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.7923,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 468328,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1232,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 470640,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0791,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 461048,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.9294,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 625932,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 111.3158,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9326644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.5948,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 482336,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.4606,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 493336,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.3092,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 599584,
            "unit": "KB",
            "extra": ""
          }
        ]
      }
    ]
  }
}