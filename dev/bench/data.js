window.BENCHMARK_DATA = {
  "lastUpdate": 1761840727421,
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
          "id": "ef6af78440704c13ebd6a92fae12be02b94c0f22",
          "message": "optimize/Shift right bitmask evaluate MLE (#855)\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-08-08T15:11:36-04:00",
          "tree_id": "b9d9835ff4a2f684808d3de82b96d36ea883a8ea",
          "url": "https://github.com/a16z/jolt/commit/ef6af78440704c13ebd6a92fae12be02b94c0f22"
        },
        "date": 1754682296452,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.0765,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 466044,
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
            "value": 3671648,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.4135,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 481936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1175,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 482784,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.048,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 475824,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.7142,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 603532,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 111.2916,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9457216,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.5658,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 482308,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 8.7117,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 498468,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.281,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 619740,
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
          "id": "15848445a200b608f45678a5e3a927031601e2cf",
          "message": "feat: add utility to write traces to a file (#854)\n\n* feat: add utility to write traces to a file\n\n* clippy\n\n* fix tracer tests\n\n---------\n\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>",
          "timestamp": "2025-08-08T15:56:22-04:00",
          "tree_id": "78e8b10088fff318c1e5872100643fb889a751a6",
          "url": "https://github.com/a16z/jolt/commit/15848445a200b608f45678a5e3a927031601e2cf"
        },
        "date": 1754684961017,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.4783,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 480796,
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
            "value": 3610932,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8578,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 470380,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.141,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 477628,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0673,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 470868,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.7307,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 649304,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 111.6749,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9396756,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.565,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 488668,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.469,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 496124,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2649,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 579240,
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
          "id": "21a8d919cd0964a654b7632f3b8b1e71e62caf00",
          "message": "Merge pull request #853 from a16z/feat/prepare-sumcheck-opt\n\nfeat: prepare sumcheck parallelism (fixed)",
          "timestamp": "2025-08-08T16:15:54-04:00",
          "tree_id": "abb52330de067b102003f89c754c16a8c442fc77",
          "url": "https://github.com/a16z/jolt/commit/21a8d919cd0964a654b7632f3b8b1e71e62caf00"
        },
        "date": 1754686104654,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.5468,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 465752,
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
            "value": 3423604,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.6456,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 477264,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1097,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 482020,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0788,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 473324,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.2042,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 636860,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 111.4842,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9343580,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.5609,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 487388,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 8.5558,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 493284,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2973,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 585436,
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
          "id": "1687134d117a19d1f6c6bd03fd23191013c53d1b",
          "message": "feat: enhance devcontainer with oh-my-zsh plugins and dependencies (#856)\n\n- Add dpkg configuration to include examples directory (required by fzf plugins)\n- Install python3 and fzf packages for oh-my-zsh plugin support\n- Configure oh-my-zsh with git, git-prompt, and fzf plugins\n- git-prompt plugin requires python3 to run git status scripts\n- fzf plugin requires fzf binary and examples for proper functionality",
          "timestamp": "2025-08-10T01:02:42-04:00",
          "tree_id": "5ba7b85a9a90525127dfdc9d5c8e205f8c74562e",
          "url": "https://github.com/a16z/jolt/commit/1687134d117a19d1f6c6bd03fd23191013c53d1b"
        },
        "date": 1754804152102,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 2.9282,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 452724,
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
            "value": 3821284,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8101,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 474688,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1539,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 478768,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.3096,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 468576,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.7235,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 602040,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 111.583,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9232272,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.5891,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 483588,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 8.4838,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 496584,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2636,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 567268,
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
          "id": "05ca5e7f8d9592cde92e3f51bb41401278cbbb9d",
          "message": "update actions version (#864)\n\nSigned-off-by: Andrew Tretyakov",
          "timestamp": "2025-08-12T09:59:40-04:00",
          "tree_id": "ca58c030ccf620741a698bf58ae4c7add6954235",
          "url": "https://github.com/a16z/jolt/commit/05ca5e7f8d9592cde92e3f51bb41401278cbbb9d"
        },
        "date": 1755009209977,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 2.9881,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 460084,
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
            "value": 3696096,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.2047,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 473964,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1164,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 471724,
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
            "value": 474156,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.7108,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 642712,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 111.3698,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9320588,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.5701,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 487792,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 8.5249,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 495672,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2877,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 641228,
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
          "id": "47d78f897990d1280eaa391eaf6dcd5e219d006b",
          "message": "Bump slab from 0.4.10 to 0.4.11 (#868)\n\nBumps [slab](https://github.com/tokio-rs/slab) from 0.4.10 to 0.4.11.\n- [Release notes](https://github.com/tokio-rs/slab/releases)\n- [Changelog](https://github.com/tokio-rs/slab/blob/master/CHANGELOG.md)\n- [Commits](https://github.com/tokio-rs/slab/compare/v0.4.10...v0.4.11)\n\n---\nupdated-dependencies:\n- dependency-name: slab\n  dependency-version: 0.4.11\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2025-08-12T15:55:20-04:00",
          "tree_id": "ac57a471edb43606de702119748feafb755d9397",
          "url": "https://github.com/a16z/jolt/commit/47d78f897990d1280eaa391eaf6dcd5e219d006b"
        },
        "date": 1755030538758,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.8112,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 457324,
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
            "value": 3385236,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.831,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 480676,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.2089,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 476168,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0811,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 473716,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.7378,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 603948,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 111.1674,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9387240,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.4957,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 488176,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.578,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 492832,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2849,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 586352,
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
          "id": "f57a31eacd28ece0d1bd7e88305534985c2ea077",
          "message": "chore: update gitignore rules (#871)",
          "timestamp": "2025-08-13T13:12:57-04:00",
          "tree_id": "1d43eb863a708c42198407d3f2462bab46d2e671",
          "url": "https://github.com/a16z/jolt/commit/f57a31eacd28ece0d1bd7e88305534985c2ea077"
        },
        "date": 1755107183873,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.4039,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 455840,
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
            "value": 3172640,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.7928,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 473204,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1475,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 482324,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.6119,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 483808,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.9897,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 664120,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 111.487,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9268376,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.5839,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 466184,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.5481,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 497200,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2875,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 589968,
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
          "id": "35a4df4df70e82408b899478e531ad6328405c86",
          "message": "Feat: transcript optimizations (#859)\n\n* feat: 128 bit transcript scalars\n\n* testing 128 bit scalars\n\n* remove powers in prepare_sumcheck\n\n* feat: blake2b transcript\n\n* style: transcript -> transcripts\n\n* tests passing\n\n* feat: KeccakTranscript -> Blake2bTranscript\n\n* fix: clippy\n\n* fix: fixtures\n\n* fix: bench fib number\n\n* fix: delete old transcript file\n\n* fix: from_bytes PR comment",
          "timestamp": "2025-08-15T06:15:33-07:00",
          "tree_id": "dcbe2ba44e571d4b1960f524c91df0fc45c0a81d",
          "url": "https://github.com/a16z/jolt/commit/35a4df4df70e82408b899478e531ad6328405c86"
        },
        "date": 1755265742973,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.4042,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 453440,
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
            "value": 3941168,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.8574,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 474268,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.488,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 476224,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.4778,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 473508,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6983,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 624984,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 110.2791,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9239252,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.5316,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 484628,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.5497,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 493312,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.1927,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 581364,
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
          "id": "06bfc4c021b121ab05ea839cbace032f27072e04",
          "message": "feat: support both nightly and stable guest toolchains (#865)\n\n* feat: support both nightly and stable guest toolchains\n* added support for print statements inside guest programs\n* added support for getrandom\n* upgraded to rust 1.89.0\n* support for nightly toolchain via the \"nightly\" #provable attribute\n\n* fix wasm build\n\n* fmt\n\n* typo\n\n* add a print warning when sys_rand is used",
          "timestamp": "2025-08-15T09:43:20-07:00",
          "tree_id": "2186598989652a1b7b2a7726e1b335e246fac82d",
          "url": "https://github.com/a16z/jolt/commit/06bfc4c021b121ab05ea839cbace032f27072e04"
        },
        "date": 1755278229709,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 2.9276,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 458024,
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
            "value": 3443636,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 2.7796,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 463600,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.7469,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 469340,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.0646,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 452756,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.6926,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 674372,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 110.8138,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9331496,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.5236,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 482412,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.1898,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 497548,
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
            "value": 587692,
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
          "id": "df7779d0e4db1062053a55d423eddf7c1d5c0dc3",
          "message": "refactor: remove BMNTV (#876)",
          "timestamp": "2025-08-15T18:06:26-04:00",
          "tree_id": "6ad3b4d6c29b01ddbfa235d67e7d4b5c70627d07",
          "url": "https://github.com/a16z/jolt/commit/df7779d0e4db1062053a55d423eddf7c1d5c0dc3"
        },
        "date": 1755297603708,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 2.9253,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 452120,
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
            "value": 3672356,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.2143,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 460056,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.1097,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 469252,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.6639,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 477112,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 1.7027,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 644708,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 110.486,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9355980,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.8012,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 481156,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 8.8154,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 492576,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.2119,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 584228,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "juliuszh@stanford.edu",
            "name": "Jules",
            "username": "JuI3s"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c87fbc8d1c70b7cd827c5df884d9e123fddf862c",
          "message": "feat: Large d multiplication optimization (#832)\n\n* Large d optimization basic eq sanity check\n\n* Temp first round sumcheck pass\n\n* Large d optimization correctness\n\n* Sumcheck test and cleanup\n\n* Multiple tests for large d optimization\n\n* Clean up unused test\n\n* Remove unused eq\n\n* Renamed to MLE\n\n* temp, added naive sumcheck benchmark\n\n* Remove previous_claim from argument\n\n* Benchmarking\n\n* Bug fix\n\n* Optimization caching stuff\n\n* Naive sumcheck bug fix\n\n* Test case for very large d\n\n* Turn vec into array\n\n* Large d opimization bench\n\n* temp\n\n* temp1\n\n* optimize 2\n\n* temp\n\n* temp\n\n* Turn vec into arr\n\n* Temp remove mle+\n\n* debug temp\n\n* Debug temp\n\n* Appendix C Optimization correctness\n\n* Code cleanup\n\n* Minor cleanup\n\n* Generic cleanup\n\n* Optimization outer sum over b\n\n* Zach's optimization\n\n* Vec to array\n\n* Before fix+\n\n* Fix round bug\n\n* Commented out test validation\n\n* Karatsuba d=16 bench\n\n* refactor in progress\n\n* Generic compute mle product (in progress\n\n* Generic mle product evaluation\n\n* Minor changes\n\n* Karatsuba bench\n\n* Quang's optimization\n\n* cargo fmt\n\n* cargo clippy\n\n* Large d bench\n\n* clippy\n\n* Removed debugging trace and added bench for d=4\n\n* Fix benchmark fresh data issue\n\n* d=4 benchmark\n\n* Update karatsuba\n\n* Toom multiplication benchmark\n\n* Delete unused files\n\n* Rename benchmarks\n\n* Rename sumcheck proof structs\n\n* Toom multiplication interpolation\n\n* Toom multiplication integration in large d sumcheck\n\n* fmt and clippy\n\n* Update typos.toml\n\n* Remove unnecessary benchmarks\n\n* Make typo bot happy\n\n* Make graphite bot happy\n\n* Instruction ra virtualization in progress\n\n* clippy\n\n* read raf ra virtualization\n\n* Adapt ra virtual to not use small field mul trait\n\n* Read Raf ra virtualization integration\n\n* clippy\n\n* Fix large d sumcheck benchmark\n\n* temp\n\n* Make sure stand-alone optimization correctness\n\n* Ra i poly checkpoint\n\n* Draft virtual ra claim working+\n\n* temp\n\n* Temp stage 3 verification passed\n\n* fib_e2e_dory passed\n\n* Remove debub prints\n\n* clippy\n\n* Remove debug comment\n\n* Remove debug check\n\n* Change type issues\n\n* Fix typo\n\n* Add new virtual poly\n\n* minor\n\n* Update test fixture\n\n* updated test fixtures\n\n* Minor changes\n\n* Code cleanup\n\n* temp resolving PR issues\n\n* minor\n\n* Move to only one expanding table\n\n* Update jolt-core/src/subprotocols/karatsuba.rs\n\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>\n\n* Update jolt-core/src/subprotocols/karatsuba.rs\n\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>\n\n* rename large d sumcheck optimization\n\n* Resolve PR issues\n\n* remove verifier files\n\n* Generate test fixtures\n\n* minor\n\n* clippy\n\n* delete print\n\n* Resolve merge conflicts\n\n---------\n\nCo-authored-by: julius <jzhang@a16z.com>\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>",
          "timestamp": "2025-08-19T14:16:54-04:00",
          "tree_id": "5be34d3f6887856281c0807549aa33bc4166f14d",
          "url": "https://github.com/a16z/jolt/commit/c87fbc8d1c70b7cd827c5df884d9e123fddf862c"
        },
        "date": 1755629765923,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.405,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 453464,
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
            "value": 3678340,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.6011,
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
            "value": 2.6604,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 455300,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 3.3118,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 461932,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 3.0893,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 584948,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 114.4131,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9334512,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.1143,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 471092,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 10.0371,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 478772,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.0277,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 564708,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "verifier-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "verifier-mem",
            "value": 10721416,
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
          "id": "f23b95527683f3a935afba18f4368bf1cdac12ed",
          "message": "fix (#889)\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-08-19T15:47:32-04:00",
          "tree_id": "8f896496cc1006ee0dbd438362ec9beec57f5af3",
          "url": "https://github.com/a16z/jolt/commit/f23b95527683f3a935afba18f4368bf1cdac12ed"
        },
        "date": 1755635165151,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 4.0655,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 453056,
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
            "value": 3437380,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.8245,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 464400,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 3.0906,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 452168,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 3.1156,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 466372,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.3369,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 597384,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 114.7136,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9370260,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.6677,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 473836,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.902,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 476644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.1347,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 555792,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "verifier-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "verifier-mem",
            "value": 10926388,
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
          "id": "e426299e256525d21c314df1eb4043bdd19c81dc",
          "message": "fix: exclude the verifier example from benchmarks (#895)",
          "timestamp": "2025-08-20T09:28:08-07:00",
          "tree_id": "c6caabc3fcc5e889a8ef007d127f633daa9aa8cf",
          "url": "https://github.com/a16z/jolt/commit/e426299e256525d21c314df1eb4043bdd19c81dc"
        },
        "date": 1755709315180,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.3734,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 450692,
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
            "value": 3852856,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.8421,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 449600,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.5259,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 447308,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.6098,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 452012,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.3291,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 621372,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 113.2741,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9276488,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.0705,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 469692,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 8.8527,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 478200,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.9681,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 559636,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dmitriymozir@gmail.com",
            "name": "MozirDmitriy",
            "username": "MozirDmitriy"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bdbed9c066fe9c43e360ec99e234cb331ee9926a",
          "message": "fix: prevent overflow in negative i64/i128 field conversions (#894)",
          "timestamp": "2025-08-20T13:17:30-04:00",
          "tree_id": "d1abe669ccf2e324774a946bbf4af90d7d162e1c",
          "url": "https://github.com/a16z/jolt/commit/bdbed9c066fe9c43e360ec99e234cb331ee9926a"
        },
        "date": 1755712330259,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.4063,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 451036,
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
            "value": 3518576,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.3538,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 459680,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.5947,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 460260,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.4932,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 465000,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.4002,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 613156,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 114.6411,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9293816,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.0587,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 470748,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.9989,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 478556,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.6832,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 539096,
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
          "id": "692e9cd3954b47d939bcc3c07693a728405ff7ce",
          "message": "fix div and rem sequences (#898)\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-08-20T17:48:55-04:00",
          "tree_id": "37c84e44ce93e41c7ab5eaf2370d91735e941b0b",
          "url": "https://github.com/a16z/jolt/commit/692e9cd3954b47d939bcc3c07693a728405ff7ce"
        },
        "date": 1755728750106,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.5712,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 456372,
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
            "value": 3795264,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.3956,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 463608,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.9945,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 447984,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 3.3985,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 457968,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.5787,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 657136,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 114.8266,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9271616,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.2973,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 465176,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 10.1563,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 484872,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.8858,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 576172,
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
          "id": "6765db557b1207bba67c320b95af79f3304ff731",
          "message": "feat: better parallelism and miscellaneous optimizations (#870)\n\n* fix: Arc for RLC\n\n* testing better witness gen\n\n* feat: 15% e2e prover speedup at high T\n\n* fix: bench numbers\n\n* fix: call sites for generate_witness_batch\n\n* fix: pr comments\n\n* fix: remove arc from one hot poly in enum\n\n* fix: bench number\n\n* fix: split linear_combination\n\n* fix: small nits\n\n* fix: few more nits\n\n* fix: added back allocative print\n\n* fix: move the print heap mem usage",
          "timestamp": "2025-08-21T10:37:57-04:00",
          "tree_id": "c02ed48c690654b4f5b4540524802356975f0362",
          "url": "https://github.com/a16z/jolt/commit/6765db557b1207bba67c320b95af79f3304ff731"
        },
        "date": 1755789160563,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.7736,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 448372,
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
            "value": 3860828,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.1767,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 467656,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.5667,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 459912,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.5214,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 463884,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.9367,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 620816,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 122.1326,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 11858920,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.5918,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 464808,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.8658,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 514644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.2113,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 589264,
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
          "id": "007ae942e615c6379ca15c8af8e406c9a3476da8",
          "message": "Revert \"fix div and rem sequences (#898)\" (#899)\n\nThis reverts commit 692e9cd3954b47d939bcc3c07693a728405ff7ce.",
          "timestamp": "2025-08-21T10:40:43-04:00",
          "tree_id": "6a339f4e3f60237d734cf752ee536ae7276079a2",
          "url": "https://github.com/a16z/jolt/commit/007ae942e615c6379ca15c8af8e406c9a3476da8"
        },
        "date": 1755789676985,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 4.9073,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 448364,
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
            "value": 3678640,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.5855,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 457100,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.8199,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 445084,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.89,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 461564,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 3.4876,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 617040,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 129.9137,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9242300,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.7299,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 471944,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 10.7192,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 508524,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.4725,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 545920,
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
          "id": "1a62a0d73fb84ededb27e72422ebc112e96262da",
          "message": "fix: cargo.lock (#901)",
          "timestamp": "2025-08-21T09:07:37-07:00",
          "tree_id": "9c1a689fead39cf198165da64e20e7b543835b87",
          "url": "https://github.com/a16z/jolt/commit/1a62a0d73fb84ededb27e72422ebc112e96262da"
        },
        "date": 1755794456542,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 4.1431,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 451828,
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
            "value": 3614416,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.1735,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 459932,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.4868,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 458988,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.6797,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 453636,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.2622,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 620380,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 118.114,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10117036,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.9759,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 461748,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.9073,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 491916,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.2526,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 536900,
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
          "id": "1d267f9e75d96c9527254e38af21099a26d03a24",
          "message": "refactor: remove unused poly utils (#896)",
          "timestamp": "2025-08-22T12:54:23-04:00",
          "tree_id": "e25f9abc7b7a193ea4b06b6bfab594d0c895cae7",
          "url": "https://github.com/a16z/jolt/commit/1d267f9e75d96c9527254e38af21099a26d03a24"
        },
        "date": 1755883700720,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.9237,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 450772,
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
            "value": 3664928,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 4.1022,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 457844,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.5238,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 464608,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.4752,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 461136,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.26,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 624236,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 115.6982,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9129836,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.0315,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 473404,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 8.8724,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 475692,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.9309,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 577828,
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
          "id": "f980d80cc19b080afb0eb4cabd00a859231f8834",
          "message": "Feat/docs revamp (#897)\n\n* Reorganize docs\n\n* Update SUMMARY.md\n\n* Add placeholders\n\n* Refresh Usage section\n\n* Make Optimizations a subdirectory\n\n* rv64\n\n* Refresh architecture overview\n\n* Update intro\n\n* Add end-to-end DAG diagram to architecture overview\n\n* r1cs and emulation sections\n\n* batched sumcheck\n\n* refresh readme\n\n* Describe wv virtualization\n\n* ram\n\n* Update construction message\n\n* registers\n\n* typo\n\n* instruction execution",
          "timestamp": "2025-08-25T10:00:00-04:00",
          "tree_id": "91fab14b22d6ffae0480286d81e11c63cc0387de",
          "url": "https://github.com/a16z/jolt/commit/f980d80cc19b080afb0eb4cabd00a859231f8834"
        },
        "date": 1756132583617,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.8574,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 446264,
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
            "value": 3626308,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.2586,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 463676,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 3.1457,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 451056,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.5377,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 461420,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 3.0431,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 646604,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 118.4185,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9767808,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.2083,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 476680,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.4224,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 522748,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.1145,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 550828,
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
          "id": "d6fc00441c2377ba0b681b5de0eeed8643668c7f",
          "message": "chore: implement the suggestions in PR-891 (#902)\n\n* chore: implement the suggestions in PR-891\n\n* feat: add proof generation and verification commands for recursion example\n\n- Introduced new CLI commands for generating and verifying proofs for the Fibonacci and Muldiv examples.\n- Added functionality to save and load proof data from specified directories.\n\n* chore: update Cargo.lock",
          "timestamp": "2025-08-25T08:41:20-07:00",
          "tree_id": "56d179ce7d0a202ac5aa3b3a623b385bb0d3db66",
          "url": "https://github.com/a16z/jolt/commit/d6fc00441c2377ba0b681b5de0eeed8643668c7f"
        },
        "date": 1756138535012,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.337,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 447608,
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
            "value": 3795440,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.1891,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 456192,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.6508,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 468068,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.4893,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 461920,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.3742,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 636812,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 117.2024,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9764724,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.5206,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 471204,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.9388,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 526136,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.994,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 569252,
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
          "id": "92f79bbd16ab502510f0c0cda620dee3ee5030e0",
          "message": "refactor: streamline cargo command execution and enhance error reporting (#908)\n\n- Introduced a new function `compose_command_line` to format the command line for better readability and error handling.\n- Simplified the argument passing to the `cargo` command by using an array.\n- Improved error reporting by displaying the full command line when a build fails.",
          "timestamp": "2025-08-25T12:21:24-04:00",
          "tree_id": "c6566271e4de09fb635a9c98cc847e8a57219169",
          "url": "https://github.com/a16z/jolt/commit/92f79bbd16ab502510f0c0cda620dee3ee5030e0"
        },
        "date": 1756140932284,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.3078,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 448584,
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
            "value": 3528564,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.1782,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 459824,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.5097,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 458492,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.4783,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 463448,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.5405,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 639228,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 115.8254,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9255480,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.3774,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 493656,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.9878,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 529568,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.0632,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 576700,
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
          "id": "80f6c9935c5a328a6133e9963d3a128436edf6a9",
          "message": "docs/inlines (#909)\n\n* docs/inlines\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fix comments\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* note on custom instructions\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* docs: add note about similarity between inlines and virtual sequences\n\nAddressed PR feedback by adding a sentence that points out the similarity\nbetween inlines and the virtual sequences already used for certain RISC-V\ninstructions, with a link to the relevant documentation section.\n\n* docs: address all PR review comments for inlines documentation\n\n- Changed all Appendix references to link to prefix-suffix sumcheck in instruction-execution.html\n- Fixed incorrect \"Appendix C\" reference to also link to prefix-suffix sumcheck\n- Emphasized that custom instructions with structured MLEs are the core innovation\n- Added clarification that extra register usage comes with ~0 cost to the prover\n- Enhanced explanation of why structured MLEs make instructions \"lookupable\"\n- Clarified that this is fundamentally different from traditional assembly optimization\n- Noted that user-defined custom instructions are only available in core Jolt codebase\n- Maintained \"register pressure\" term per reviewer's defense of its legitimacy\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-08-26T15:55:40-04:00",
          "tree_id": "23b67fdbbd30d7b3d16f94ee404c2e6acc561937",
          "url": "https://github.com/a16z/jolt/commit/80f6c9935c5a328a6133e9963d3a128436edf6a9"
        },
        "date": 1756240251931,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.3209,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 446016,
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
            "value": 3576940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.7162,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 458176,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.6506,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 458960,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.4261,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 459476,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.8545,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 618448,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 117.2929,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9725392,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.8105,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 470004,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.953,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 513148,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.3233,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 541296,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "radikpadik76@gmail.com",
            "name": "radik878",
            "username": "radik878"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2343e4b39c7fed5ec398dd4776bdac360aa76373",
          "message": "fix(msm): avoid i64::MIN overflow in I64Scalars MSM (#917)\n\n* fix(msm): avoid i64::MIN overflow in I64Scalars MSM\n\n* delete comments",
          "timestamp": "2025-08-27T09:33:19-07:00",
          "tree_id": "a36982159652f2f3e5153af3810350a229cdedfd",
          "url": "https://github.com/a16z/jolt/commit/2343e4b39c7fed5ec398dd4776bdac360aa76373"
        },
        "date": 1756314494212,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.8183,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 445928,
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
            "value": 3796696,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.187,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 460772,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 3.0534,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 459112,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.5242,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 452388,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.7432,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 602152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 118.9091,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10227496,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.0243,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 465380,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.8564,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 478252,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.1365,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 594472,
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
          "id": "94d24fc945842d123ef1382b8d82e5d35de9fd61",
          "message": "docs: Round out architecture section (#918)\n\n* Round out architecture section\n\n* typo\n\n* delete Spartan section from Optimizations",
          "timestamp": "2025-08-29T13:39:38-04:00",
          "tree_id": "50ed3a13fe283744f992a0b5fdef620e878c8ddd",
          "url": "https://github.com/a16z/jolt/commit/94d24fc945842d123ef1382b8d82e5d35de9fd61"
        },
        "date": 1756491316716,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 4.264,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 456724,
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
            "value": 3606524,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.1939,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 461560,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.5492,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 458080,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.8283,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 459056,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.5532,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 577092,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 115.759,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9164436,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.1869,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 485628,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.9885,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 550624,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.9436,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 551048,
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
          "id": "f5f9ae19e0b4309f9276d3423f86c0a73e8d60c5",
          "message": "Fix latex typo (#919)",
          "timestamp": "2025-08-29T13:50:23-04:00",
          "tree_id": "1fd2624db6a4998c4339d8ce180e26665ad70e2c",
          "url": "https://github.com/a16z/jolt/commit/f5f9ae19e0b4309f9276d3423f86c0a73e8d60c5"
        },
        "date": 1756491936937,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.3436,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 450004,
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
            "value": 3832408,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.5183,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 457840,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.5922,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 456592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.4617,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 470680,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.923,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 620976,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 118.9674,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10250952,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.991,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 468900,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.8994,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 491632,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.6042,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 556632,
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
          "id": "3c3a65e69f81e332f29e3fd7a2b870c0a8d61b4f",
          "message": "fix typo (#920)",
          "timestamp": "2025-08-29T13:52:30-04:00",
          "tree_id": "7086fc0f65f5766206bb6957cfd6a6544fb9145a",
          "url": "https://github.com/a16z/jolt/commit/3c3a65e69f81e332f29e3fd7a2b870c0a8d61b4f"
        },
        "date": 1756492088425,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.3281,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 449228,
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
            "value": 3774808,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 4.0442,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 460064,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.57,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 464700,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.5626,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 462956,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.3372,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 614104,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 116.315,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9278436,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.051,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 469328,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.958,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 507084,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.9682,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 540192,
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
          "id": "2e7888813e7c876db6e30ff1b2cee52755f89708",
          "message": "chore(deps): bump tracing-subscriber from 0.3.19 to 0.3.20 (#925)\n\nBumps [tracing-subscriber](https://github.com/tokio-rs/tracing) from 0.3.19 to 0.3.20.\n- [Release notes](https://github.com/tokio-rs/tracing/releases)\n- [Commits](https://github.com/tokio-rs/tracing/compare/tracing-subscriber-0.3.19...tracing-subscriber-0.3.20)\n\n---\nupdated-dependencies:\n- dependency-name: tracing-subscriber\n  dependency-version: 0.3.20\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2025-09-02T16:02:55-04:00",
          "tree_id": "e14335d5ff33cf58848134b9337950c7ae063418",
          "url": "https://github.com/a16z/jolt/commit/2e7888813e7c876db6e30ff1b2cee52755f89708"
        },
        "date": 1756845573283,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.4392,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 470660,
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
            "value": 3593696,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.3718,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 458100,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.6058,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 454652,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.9144,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 455716,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.6296,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 595152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 119.2432,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10359168,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.249,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 491328,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 10.0023,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 476596,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.1707,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 556336,
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
          "id": "87f4432c9ba75bbcc94a7fda965047a555ae269c",
          "message": "address Justin's comments (#924)",
          "timestamp": "2025-09-02T16:15:24-04:00",
          "tree_id": "afb0b7078f134b0d27f0629e9f02868a67281e77",
          "url": "https://github.com/a16z/jolt/commit/87f4432c9ba75bbcc94a7fda965047a555ae269c"
        },
        "date": 1756846262650,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.3445,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 470004,
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
            "value": 3351172,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.9252,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 456916,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.8622,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 454888,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.6632,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 460432,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.3871,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 622748,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 118.4964,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10270064,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.6508,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 465464,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.0034,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 532808,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.4469,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 576184,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39494992+GUJustin@users.noreply.github.com",
            "name": "Justin Thaler",
            "username": "GUJustin"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "21b9eebcad3bcedf5b0ad4401bb461b39d972389",
          "message": "Update twist-shout.md",
          "timestamp": "2025-09-08T16:12:40-04:00",
          "tree_id": "e4ec12b15c51b487bb2f1c0174971c274dd83822",
          "url": "https://github.com/a16z/jolt/commit/21b9eebcad3bcedf5b0ad4401bb461b39d972389"
        },
        "date": 1757364405474,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.2497,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 446328,
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
            "value": 3789296,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.1526,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 464680,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.4664,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 462448,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.5253,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 461664,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.2333,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 605892,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 115.3999,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9140208,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 3.9475,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 475040,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.8106,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 507772,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.8462,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 576880,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39494992+GUJustin@users.noreply.github.com",
            "name": "Justin Thaler",
            "username": "GUJustin"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e582bdea432bcc7a7c658e54d1aa36d5f1d59a29",
          "message": "Update twist-shout.md",
          "timestamp": "2025-09-08T16:17:08-04:00",
          "tree_id": "069647cdd8a34939e930294760c8f2c6c249cfc3",
          "url": "https://github.com/a16z/jolt/commit/e582bdea432bcc7a7c658e54d1aa36d5f1d59a29"
        },
        "date": 1757364768391,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.9891,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 447436,
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
            "value": 3539512,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 4.1507,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 459404,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.557,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 462224,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 3.0019,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 457664,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.3984,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 661240,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 116.9784,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9300680,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 5.0807,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 471600,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.1776,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 508580,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.8633,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 540564,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39494992+GUJustin@users.noreply.github.com",
            "name": "Justin Thaler",
            "username": "GUJustin"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b410e36ddb03c133f9ec63499ffdcf89cc72b420",
          "message": "Update twist-shout.md",
          "timestamp": "2025-09-08T16:42:58-04:00",
          "tree_id": "941a96bc0e103a743ecef641dbd871bf39ad4eca",
          "url": "https://github.com/a16z/jolt/commit/b410e36ddb03c133f9ec63499ffdcf89cc72b420"
        },
        "date": 1757366333934,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.3416,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 446064,
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
            "value": 3205332,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.2331,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 464932,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.5353,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 449944,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 3.0459,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 452464,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.3557,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 617312,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 120.357,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10590540,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.1125,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 462768,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.163,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 509288,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.9315,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 597332,
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
          "id": "bed3217f3e154e0249724c8843a0dc4f7c945c7d",
          "message": "fix: typos\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-09-08T18:27:34-04:00",
          "tree_id": "6f0b1bda65a86f31e7a30b0fde8f5355a05ee47c",
          "url": "https://github.com/a16z/jolt/commit/bed3217f3e154e0249724c8843a0dc4f7c945c7d"
        },
        "date": 1757372544026,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.3679,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 447460,
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
            "value": 3667432,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.2809,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 461660,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 3.1671,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 456856,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.4505,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 466756,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.3932,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 592748,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 116.2486,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9289688,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.0719,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 484188,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.5531,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 480892,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.4282,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 571828,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39494992+GUJustin@users.noreply.github.com",
            "name": "Justin Thaler",
            "username": "GUJustin"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a80555130273992031d6958a5effd6d0f466bb79",
          "message": "Update streaming.md",
          "timestamp": "2025-09-09T13:40:17-04:00",
          "tree_id": "aea75b0d878638810f9677fe285da55657d5295c",
          "url": "https://github.com/a16z/jolt/commit/a80555130273992031d6958a5effd6d0f466bb79"
        },
        "date": 1757441703778,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.9757,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 460000,
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
            "value": 3611704,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.1614,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 456960,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.545,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 457352,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.5059,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 462244,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.433,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 600340,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 115.0832,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9283936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.1602,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 472536,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.8211,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 512092,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.3835,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 549088,
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
          "id": "13e96f9bf3060ec47618422e423c34952b8e415b",
          "message": "fix: typos\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-09-10T11:10:20-04:00",
          "tree_id": "6a37cc59f8ee8fde32daec6d59afc8e095afb363",
          "url": "https://github.com/a16z/jolt/commit/13e96f9bf3060ec47618422e423c34952b8e415b"
        },
        "date": 1757519133059,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.3205,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 448064,
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
            "value": 3936636,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.9357,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 459152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.4838,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 457180,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.5032,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 467152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.996,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 618344,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 118.8431,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10179756,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.0566,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 465588,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 9.3288,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 579944,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 4.9561,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 544520,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "60488569+tcoratger@users.noreply.github.com",
            "name": "Thomas Coratger",
            "username": "tcoratger"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c27879edc18557dc2f7acd2237718a7ad865180c",
          "message": "clippy: small nits (#942)",
          "timestamp": "2025-09-17T11:56:20-04:00",
          "tree_id": "e75e795eba723050dd9185ff2908f20181bb5d7c",
          "url": "https://github.com/a16z/jolt/commit/c27879edc18557dc2f7acd2237718a7ad865180c"
        },
        "date": 1758126785212,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 3.4786,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 465116,
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
            "value": 3782796,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 3.4961,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 462612,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.6392,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 459332,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.81,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 450092,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 3.0144,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 611400,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 119.3976,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 10396572,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 5.0535,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 474208,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 10.0252,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 514568,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.4484,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 555976,
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
          "id": "87af8fe6b84c47c82299df87b59e7dd2caf67f20",
          "message": "Chore/rm fixtures (#952)\n\n* rm fixtures\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* update gitignore\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* update ci\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fully exclude fixture rs files\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fix ci\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-09-19T14:55:46-04:00",
          "tree_id": "f9af61816caa450cfc80a05da27867ca0f0574e5",
          "url": "https://github.com/a16z/jolt/commit/87af8fe6b84c47c82299df87b59e7dd2caf67f20"
        },
        "date": 1758310315063,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 4.216,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 447508,
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
            "value": 3682132,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 4.1458,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 460036,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "memory-ops-time",
            "value": 2.6463,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "memory-ops-mem",
            "value": 460988,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 2.5603,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 450524,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 2.4262,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 571980,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 115.2882,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 9218364,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 4.121,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 469156,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 8.9693,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 520872,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "stdlib-time",
            "value": 5.2343,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "stdlib-mem",
            "value": 564324,
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
          "id": "5101ad2143039de6f279613810414c3d071d1f8f",
          "message": "fix: fix performange benchmarks\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-09-30T15:40:26-04:00",
          "tree_id": "3251230aa66b7d6e69d6f950dac6aece11f992bb",
          "url": "https://github.com/a16z/jolt/commit/5101ad2143039de6f279613810414c3d071d1f8f"
        },
        "date": 1759264386145,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 752008,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 784768,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 768596,
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
            "value": 3387436,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 780168,
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
            "value": 766552,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 798324,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 839652,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 853444,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 349340,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 7021344,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 805128,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 771104,
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
            "value": 83644,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3aa111c877f8ba9453a4df0095e084c60f7d9df4",
          "message": "Only materialize ra_i polynomials after 2nd sumcheck round (#968)",
          "timestamp": "2025-10-01T16:56:13-04:00",
          "tree_id": "d82384d2b76be78bbe99f4f84aace3c05423b303",
          "url": "https://github.com/a16z/jolt/commit/3aa111c877f8ba9453a4df0095e084c60f7d9df4"
        },
        "date": 1759355239742,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 758504,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 761012,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 778860,
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
            "value": 3769860,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 778792,
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
            "value": 763020,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 741976,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 847768,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 865376,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 350188,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6420720,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 807940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 778456,
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
            "value": 82820,
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
          "id": "e1c961ce2ff69c78c6ab1677bc029d39daf5689f",
          "message": "standardize tracing instrumentation for sumchecks (#977)",
          "timestamp": "2025-10-01T18:47:23-04:00",
          "tree_id": "a03740b86737f71d7ed206cec6687832004a5772",
          "url": "https://github.com/a16z/jolt/commit/e1c961ce2ff69c78c6ab1677bc029d39daf5689f"
        },
        "date": 1759361860324,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 772092,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 769776,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 772936,
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
            "value": 3741512,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 763704,
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
            "value": 770552,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 762936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 864052,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 835712,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 344784,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6100544,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 806344,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 764880,
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
            "value": 82840,
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
          "id": "4087cc965ed8240e2399e00525da5674f9d82171",
          "message": "chore: add tracing onto the JoltDAG prove and verify methods\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-02T14:51:43-04:00",
          "tree_id": "8a5aec6cf1926dd24b8459a01f85b50040bd673f",
          "url": "https://github.com/a16z/jolt/commit/4087cc965ed8240e2399e00525da5674f9d82171"
        },
        "date": 1759434405145,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 761136,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 766856,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 759528,
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
            "value": 3714136,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 775288,
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
            "value": 765916,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 774248,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 848560,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 838824,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 341248,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6971228,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 778648,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 782140,
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
            "value": 82452,
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
          "id": "6e289f39ed5d96a0b3cb564e3504ca975f574df6",
          "message": "Reword tracing of Jolt prove function (#978)\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-02T15:17:00-04:00",
          "tree_id": "b39ab2e9b6b43047b7ad949ea31fd6dd1d8201ba",
          "url": "https://github.com/a16z/jolt/commit/6e289f39ed5d96a0b3cb564e3504ca975f574df6"
        },
        "date": 1759435712149,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 779056,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 768364,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 775060,
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
            "value": 3737592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 772104,
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
            "value": 766984,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 761768,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 850604,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 818784,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 348532,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6917628,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 790628,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 811012,
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
            "value": 82336,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "codygunton@gmail.com",
            "name": "Cody Gunton",
            "username": "codygunton"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "97c1b40ad10f12d1c0e7fd6847c9ff5cd5a7eafb",
          "message": "fix: run two-byte alignment tests (#979)\n\n* rename patch\n\n* update a patching to use only I ext and no privs",
          "timestamp": "2025-10-02T17:51:48-04:00",
          "tree_id": "693e8db25b5100acda1c25dc2539ad21fb928e46",
          "url": "https://github.com/a16z/jolt/commit/97c1b40ad10f12d1c0e7fd6847c9ff5cd5a7eafb"
        },
        "date": 1759444932216,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 772124,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 770992,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 771324,
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
            "value": 3729008,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 764348,
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
            "value": 760084,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 790456,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 858972,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 857692,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 342980,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6095420,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 782404,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 780040,
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
            "value": 83836,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5a07c7a127f860274a5eac133ee81c8a8a24aec4",
          "message": "Create new crate for Jolt fields (#984)",
          "timestamp": "2025-10-06T14:10:47-07:00",
          "tree_id": "c5f91c2eff291f4f4f781d210592a3c6633cf3ac",
          "url": "https://github.com/a16z/jolt/commit/5a07c7a127f860274a5eac133ee81c8a8a24aec4"
        },
        "date": 1759788219847,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 759360,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 769388,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 767408,
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
            "value": 3928476,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 760820,
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
            "value": 772436,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 766132,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 844016,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 828024,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 345456,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6252472,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 792592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 767228,
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
            "value": 856104,
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
            "email": "mchl.zhu.96@gmail.com",
            "name": "Michael Zhu",
            "username": "moodlezoup"
          },
          "distinct": true,
          "id": "43d7a05622edc13f3875e1cc9c5b0ec8b8666e09",
          "message": "Revert \"Create new crate for Jolt fields (#984)\"\n\nThis reverts commit 5a07c7a127f860274a5eac133ee81c8a8a24aec4.",
          "timestamp": "2025-10-06T16:47:43-07:00",
          "tree_id": "693e8db25b5100acda1c25dc2539ad21fb928e46",
          "url": "https://github.com/a16z/jolt/commit/43d7a05622edc13f3875e1cc9c5b0ec8b8666e09"
        },
        "date": 1759797661543,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 768416,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 762880,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 754416,
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
            "value": 3443756,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 762280,
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
            "value": 772904,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 756152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 863856,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 822656,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 346732,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6088460,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 781460,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 756760,
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
            "value": 850056,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "github@randomwalks.xyz",
            "name": "Ari",
            "username": "abiswas3"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c5d0eb0a37cbb70fe6f4a8e620c441501e61d5c0",
          "message": "perf: F::Challenge optimized binding and MLE evals\n\n* feat: small_scalar_binding\n\nOps are generic but associated type was being rv64, so merging\n\n* feat: small_scalar_binding\n\nCode builds.\nNeed to bring changes to transcripts -- and then if all tests pass,\nwe can change call sites\n\n* feat: small_scalar_binding\n\nCode builds.\nNeed to bring changes to transcripts -- and then if all tests pass,\nwe can change call sites\n\n* feat: small_scalar_binding\n\nTranscript tests also pass. Checking allocative\n\n* feat: small_scalar_binding\n\nMultipliplication with native operations is still faster (as expected).\n\n* Merge branch 'feat/rv64' of https://github.com/abiswas3/jolt into feat/rv64-small-binding\n\n* feat: small_binding\n\nnearly all errors are gone.\nI need to comment out the tests to make things pass cos MLE_field is not\nyet written\n\n* feat: small_binding\n\nJust commitments and lookuptables remain.\nOf course mle_field to make_the whole thing work\n\n* feat: small_binding\n\nJust tests and ambiguous challenge type (which I will figure out how to\ndeal with)\n\n* feat: small_binding\n\nIdeally just lookup tables left, and all tests will pass.\nBut that is to be seen\n\n* feat: small_binding\n\nPassing these tests are such a nightmare\n\n* feat: small_binding\n\nBringing in the field versions of mle lookup tables\n\n* feat: small_binding\n\nMLE lookuptable done, but no prefix_field yet\n\n* feat: small_binding\n\nupdate fields and then it's all done\n\n* feat: small_binding\n\nupdate fields and then it's all done\n\n* feat: small_binding\n\nChanged to warns.\n\n* feat: small_binding\n\nAll compiler errors gone!\n\n* feat: small_binding\n\nClippy + Old Benches passing!\n\n* feat: small_binding\n\nClippy + Old Benches passing!\n\n* feat: small_binding\n\nTidying up code structure\n\n* feat: small_binding\n\nReady for PR\n\n* chore: reducing size of diff\n\nThe multiple variants of mle_field and mle_half_and_half are not needed\nit seems. Just tell the rust compiler [X] and [Y] are types that have\nthe a certain property.\n\n* chore: reducing size of diff\n\nTidying up things\n\n1. Removing comments for standard operations\n2. Challenge now implemnents ark_ff::UniformRand\n3. eq_poly mle is generic (the remainder will be generic as well)\n4. benches now use feaures so we don't need to comment out things and\nrun again\n\n* chore: reducing size of diff\n\nTidying up things\n\n1. Removing comments for standard operations\n2. Challenge now implemnents ark_ff::UniformRand\n3. eq_poly mle is generic (the remainder will be generic as well)\n4. benches now use feaures so we don't need to comment out things and\nrun again\n5. Preparing to pull the lastest feat/rv64 with no trait objects and\nmake into generics\n\n* chore: reducing size of diff\n\nPreparing the merge\n\n* chore: reducing size of diff\n\nPreparing the merge\n\n* Merge remote-tracking branch 'upstream/feat/rv64' into feat/rv64-small-binding\n\nMerge seems to be sucessful. These merges can be a bit of a nightmare,\nwith the changeds being so subtle. But i think i've gotten everything\n in.\n\n* chore: reducing diff size\n\nGeneric polynomial evals pass\n\n* chore: reducing diff size\n\nLookuptable MLE generics almost done\n\n* chore: reducing diff size\n\nLookuptable MLE tests errors gone\n\n* chore: reducing diff size\n\nPrefix MLE and Checkpoint erros half gone\n\n* chore: reducing diff size\n\nPrefix MLE and Checkpoint erros 3/4 gone\n\n* chore: reducing diff size\n\nNow all errors seem to be gone. Testing\n\n* chore: reducing diff size\n\nAll tests pass as well.\n\n* chore: reducing diff size\n\nCosmetics -- making sure all trait bounds are consistent\n\n* chore: reducing diff size\n\nAll remnants of field removed!\n\n* chore: reducing diff size\n\nAll remnants of field removed!\n\n* chore: reducing diff size\n\nNearly ready for PR\n\n* optimisation: polynomial batch and single\n\nMore cache-efficient sparsity accounting polynomial evaluations.\n\n* optimisation: polynomial batch and single\n\nAll tests pass. Formatting works. One more review before PR\n\n* optimisation: polynomial batch and single\n\nAddressing Markos stylistic comments\n\n* optimisation: polynomial batch and single\n\nAddressing Markos stylistic comments\n\n* Update jolt-core/src/poly/multilinear_polynomial.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* Update jolt-core/src/poly/eq_poly.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* chore: cleaning up comment and formatting\n\n* chore: cleaning up comment and formatting\n\nBenches documented. The theory docs will be updated soon\n\n* chore: cleaning up comment and formatting\n\n1. Removed *r + F::zero() with (*r).into()\n2. Macros for ops generation documented\n3. Cleaner imports for mle in eq_poly\n\n* chore: cleaning up comment and formatting\n\n1. Removed *r + F::zero() with (*r).into()\n2. Macros for ops generation documented\n3. Cleaner imports for mle in eq_poly\n\n* chore: cleaning up comment and formatting\n\n1. Removed *r + F::zero() with (*r).into()\n2. Macros for ops generation documented\n3. Cleaner imports for mle in eq_poly\n4. Making MontU128 the default\n\n* chore: cleaning up comment and formatting\n\n1. Removed *r + F::zero() with (*r).into()\n2. Macros for ops generation documented\n3. Cleaner imports for mle in eq_poly\n4. Making MontU128 the default\n5. Cleaning up all the Challenge::randoms\n\n* Update jolt-core/src/transcripts/blake2b.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* Update jolt-core/src/subprotocols/mles_product_sum.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* Update jolt-core/src/subprotocols/mles_product_sum.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* Update jolt-core/src/subprotocols/mles_product_sum.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* Update jolt-core/src/field/mod.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* Update jolt-core/src/field/challenge/mont_ark_u128.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* Update jolt-core/benches/poly_bench.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* Update jolt-core/benches/poly_bench.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* Update jolt-core/benches/poly_bench.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* chore: renames\n\n* chore: renames\n\n* Update jolt-core/src/field/mod.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* Update jolt-core/src/field/mod.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* fix: cargo toml and srs -> urs\n\n---------\n\nCo-authored-by: Ari <software@randomwalks.xyz>\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\nCo-authored-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\nCo-authored-by: markosg04 <mgeorghiades@a16z.com>",
          "timestamp": "2025-10-06T17:42:48-07:00",
          "tree_id": "7801fb28f108d60d24e163910442be2382800b34",
          "url": "https://github.com/a16z/jolt/commit/c5d0eb0a37cbb70fe6f4a8e620c441501e61d5c0"
        },
        "date": 1759800910269,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 788576,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 781924,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 755648,
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
            "value": 3715716,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 798120,
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
            "value": 777576,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 755884,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 887004,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 861472,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 345008,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6455152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 809056,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 760496,
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
            "value": 859040,
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
          "id": "710e678bfa3bcdd3f3a229485308ac7a3841f5ee",
          "message": "Update arkworks-algebra overrides to twist-shout branch (#989)",
          "timestamp": "2025-10-07T10:06:27-04:00",
          "tree_id": "0d8c19da464c93f67ce4c82b6a70571ba69fa77e",
          "url": "https://github.com/a16z/jolt/commit/710e678bfa3bcdd3f3a229485308ac7a3841f5ee"
        },
        "date": 1759849108017,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 778592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 780012,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 773548,
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
            "value": 3569216,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 773308,
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
            "value": 774828,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 771888,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 847360,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 849116,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 348252,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6232120,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 780376,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 758212,
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
            "value": 846064,
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
          "id": "60c33cb289b8244d6249787caf2920d33367913e",
          "message": "fix: Add sumcheck input and opening claims to transcript (#981)\n\n* Add sumcheck input claims to transcript\n\n* Add ProofTranscript generic to SumcheckInstance\n\n* Append opening claims to transcript",
          "timestamp": "2025-10-07T11:18:51-04:00",
          "tree_id": "628b5da5f1bafeae8402230b7ff12ae39dd06e83",
          "url": "https://github.com/a16z/jolt/commit/60c33cb289b8244d6249787caf2920d33367913e"
        },
        "date": 1759853421723,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 762160,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 784476,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 778560,
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
            "value": 3552928,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 763736,
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
            "value": 763860,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 755492,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 857516,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 829668,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 347240,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6489688,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 787004,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 770256,
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
            "value": 848712,
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
          "id": "9bd495d37a9947b473a7c9efb0c3ada9da97e9d8",
          "message": "fix: elf path for recursion example (#991)",
          "timestamp": "2025-10-07T08:39:24-07:00",
          "tree_id": "afa61117b9e1912eac9a53264676ff287cbd8be5",
          "url": "https://github.com/a16z/jolt/commit/9bd495d37a9947b473a7c9efb0c3ada9da97e9d8"
        },
        "date": 1759854729375,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 779696,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 794568,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 774724,
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
            "value": 3427688,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 768232,
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
            "value": 780996,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 814396,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 836976,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 826752,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 349572,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6089020,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 783020,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 778152,
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
            "value": 843676,
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
            "email": "mchl.zhu.96@gmail.com",
            "name": "Michael Zhu",
            "username": "moodlezoup"
          },
          "distinct": true,
          "id": "3108c7e14fb16a11320ee7753d6369485511a60e",
          "message": "Revert \"Feat/product virtualization (#986)\"\n\nThis reverts commit daf6bf8b653b96728d8bc59c20037c7d6e913647.",
          "timestamp": "2025-10-07T09:12:03-07:00",
          "tree_id": "afa61117b9e1912eac9a53264676ff287cbd8be5",
          "url": "https://github.com/a16z/jolt/commit/3108c7e14fb16a11320ee7753d6369485511a60e"
        },
        "date": 1759856766214,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 776556,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 772320,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 796972,
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
            "value": 3826580,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 753900,
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
            "value": 757200,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 793148,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 861436,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 884364,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 346852,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6422180,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 774604,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 768176,
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
            "value": 866028,
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
          "id": "1175cf8908a61887fc1ce2eae556a23fb180d977",
          "message": "[JOLT-218] Feat/product virtualization (#992)\n\n* Make Product a virtual polynomial\n\n* Product virtualization sumcheck\n\n* Remove product constraint\n\n* rebase fix\n\n* Add Product to ALL_VIRTUAL_POLYNOMIALS\n\n* Delete comment\n\n* fix rebase",
          "timestamp": "2025-10-07T14:00:38-04:00",
          "tree_id": "06f8b7baa7bc5efdd29d2b9a96d29142379c4bc5",
          "url": "https://github.com/a16z/jolt/commit/1175cf8908a61887fc1ce2eae556a23fb180d977"
        },
        "date": 1759863124578,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 764944,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 775736,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 751816,
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
            "value": 3650144,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 756036,
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
            "value": 759452,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 757428,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 863328,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 827644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 347948,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6153164,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 805108,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 771520,
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
            "value": 871464,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "183745461+danielwlz@users.noreply.github.com",
            "name": "danielwlz",
            "username": "danielwlz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "58cd8e0622839418a181e44e9595658c2909fc7d",
          "message": "feat: add pprof CPU profiling support (#988)\n\n* feat: add pprof CPU profiling support\n\nAdd feature-gated pprof support for detailed CPU profiling of Jolt proof generation and verification.\n\nFeatures:\n- PprofGuard RAII struct with automatic profile generation on drop\n- pprof_scope! macro for easy integration\n- Automatic file naming based on benchmark name (e.g., sha3_prove.pb)\n- Configurable sampling frequency via PPROF_FREQ env var (default: 100 Hz)\n- Configurable output path prefix via PPROF_PREFIX env var\n  (default: benchmark-runs/pprof/<benchmark>_)\n- Integration points: trace, prove, and verify operations\n- Performance metrics: logs prove duration and throughput (kHz)\n\nUsage:\n  cargo run --release --features pprof -p jolt-core profile --name sha3\n  go tool pprof -http=:8080 benchmark-runs/pprof/sha3_prove.pb\n\nEnvironment variables:\n- PPROF_PREFIX: Output path prefix (default: benchmark-runs/pprof/<benchmark>_)\n  Examples: \"pprof/\" for pprof/prove.pb, \"custom/path/test_\" for custom/path/test_prove.pb\n- PPROF_FREQ: Sampling frequency in Hz (default: 100)\n\nFiles changed:\n- jolt-core/Cargo.toml: Add pprof feature and dependencies\n- jolt-core/src/zkvm/mod.rs: Add PprofGuard, macro, and integration\n- jolt-core/src/bin/jolt_core.rs: Auto-set PPROF_PREFIX from benchmark name\n- jolt-core/benches/e2e_profiling.rs: Add Display derive for BenchType\n- .gitignore: Ignore pprof output files and benchmark-runs/\n- README.md: Document pprof usage with examples\n\n* Enable frame pointer feature for pprof to avoid SIGTRAP on Mac\n\n* Amend println to tracing::info\n\n* Add binary path to go tool pprof command\n\n* Fix typo\n\n* cargo fmt\n\n* Add documentation to jolt book and screenshots",
          "timestamp": "2025-10-08T13:45:44-04:00",
          "tree_id": "89c07e076d89b75fdc90b096f551b501a208d90b",
          "url": "https://github.com/a16z/jolt/commit/58cd8e0622839418a181e44e9595658c2909fc7d"
        },
        "date": 1759948592860,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 759936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 759796,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 754228,
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
            "value": 3695468,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 757404,
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
            "value": 781600,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 766536,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 857904,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 829956,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 348428,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6001488,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 784824,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 777560,
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
            "value": 867452,
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
          "id": "4aa82ae420b3f7fc428d6194ea2286eb42a1a7d9",
          "message": "Monitor process memory using memory_stats instead of system memory (#998)",
          "timestamp": "2025-10-09T10:35:09-04:00",
          "tree_id": "b9e93d2be29c729aaadc2e389e2d26c3be4677e7",
          "url": "https://github.com/a16z/jolt/commit/4aa82ae420b3f7fc428d6194ea2286eb42a1a7d9"
        },
        "date": 1760023772900,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 806680,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 789744,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 767984,
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
            "value": 3438804,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 768944,
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
            "value": 755140,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 780120,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 852936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 838708,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 348428,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6164348,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 784008,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 764268,
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
            "value": 869124,
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
          "id": "3300eb3b13ee7a4030641d07675a82ecaea499be",
          "message": "feat: memory reduction [JOLT-219] (#994)\n\n* refactcor ra_polynomial\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* refactor instruction booleanity to use ra_poly\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* bytecode booleanity\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* one hot polynomial to use RaPoly\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* three rounds of ra_poly\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* parallel unsafe initialization\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* rebase\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* no unsafe stuff\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* tracing\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* Revert \"three rounds of ra_poly\"\n\nThis reverts commit eae0fdc43b3ae610e772c5abc624409c631ac234.\n\n* use u8 for indices\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* Reapply \"three rounds of ra_poly\"\n\nThis reverts commit 5a8ce8f7627505c343f226fb4c1cbc4357e1bf52.\n\n* wow...\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* replace par chunks with par iter\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* add assert check\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fix sequential loop\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fix warning in the cargo\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* par_iter everywhere\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* add extra tracing\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* allocative for prefix registry\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* try again\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-09T16:59:31-04:00",
          "tree_id": "d5afc2fb8625d02a66871e642d232788807c7b66",
          "url": "https://github.com/a16z/jolt/commit/3300eb3b13ee7a4030641d07675a82ecaea499be"
        },
        "date": 1760046714314,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 763024,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 765540,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 757040,
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
            "value": 3545768,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 759288,
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
            "value": 793756,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 767072,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 852376,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 886324,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 346932,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5588244,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 790612,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 790416,
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
            "value": 844756,
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
          "id": "66ec47903666a59db48ee1d6b9f0e0b61c7e75e2",
          "message": "feat: benchmark utils (#1000)\n\n* initial master bench\n\n* output summary\n\n* sha bench broken\n\n* put into a bash instead\n\n* jolt benchmark\n\n* only count prover time\n\n* sha -> chain\n\n* rv64bench benchmarks\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* merge\n\n* single thread?\n\n* enable all benchmarks\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* continuous graph. Let's make it sawtooth\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* sawtooth pattern\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fix btree bench\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fix?\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fix all committed polynomials being uninitialized\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* plot\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* put the uncompressed in legend\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fix scale on mem chart\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* final plot adjustments\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* Move remaining benchmarks from src/benches/bench.rs to benches/e2e_profiling.rs\n\n- Moved master_benchmark function and helper functions (get_fib_input, get_sha2_chain_iterations, etc.)\n- Moved plot creation functions (create_benchmark_plot, create_proof_size_plot, plot_from_csv)\n- Moved prove_example_with_trace function\n- Removed shout and twist benchmark functions as they are no longer needed\n- Updated prove API calls to use new signature with elf_contents parameter\n- Fixed all imports and removed unused dependencies\n\n* update scale\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* update the title of the plot\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* remove unused file\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* update title\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* don't count tracing time\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* special trace ln calc\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* add scripts\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* up singlethreaded number to 2^24\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* put the command in the numa thing\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* remove numa from script\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* add special rust flags\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* cleanup: benchmark rv64 (#993)\n\n* Step 1: BenchmarkArgs\n\n* Add empirical cycle calculations for iterations\n\n* Refactor master_benchmark to deduplicate code\n\n* Refactor to use arguments instead of env\n\n* Refactor master_benchmark to accept single benchmark type and scale as parameters\n\n* Remove dead code, fix build\n\n* cargo fmt\n\n* Enhance optimize_machine.sh with memory overcommit and max map count settings, improve ulimit error handling, and add system info output\n\n* Add cargo.lock\n\n* Refactor jolt_benchmarks.sh to support variable trace lengths and customizable benchmarks; update output paths and CSV handling for results.\n\n* Fixes per AI\n\n* Move to scripts to directory\n\n* Rename run_benchmarks.sh to avoid ambiguity\n\n* Python scripts to display benchmarks and plot\n\n* Update README\n\n* Clean up plotting script\n\n* Flip args order\n\n* Fix script bug, cleanup unused\n\n* cargo fmt\n\n* Hz -> kHz\n\n* flake8\n\n* Bugfix\n\n* Add resume option to jolt_benchmarks.sh for skipping existing benchmarks, clean up script, error handling\n\n* Fixes and cleanup\n\n* Add error handling for missing plotly dependency in plotting script\n\n* Plotting fixes\n\n* Refactor and cleanup summary generator\n\n* Address outdated comment\n\n* clippy\n\n* cargo fmt\n\n* Ignore benchmark runs\n\n* Fix SDK verifier test regression\n\n* Regenerate Cargo.lock\n\n* Fix regressions\n\n* Fix merge regression\n\n* Fix AllCommittedPolynomials properly\n\n* Apply AI suggestion\n\n* clippy\n\n* cargo fmt\n\n* Clean up output\n\n* Infer scale from max-trace-length if not supplied\n\n* Run scale 18-20 in benchmarks\n\n* fmt, clippy\n\n* Update scripts/jolt_benchmarks.sh\n\nCo-authored-by: graphite-app[bot] <96075541+graphite-app[bot]@users.noreply.github.com>\n\n* Use cache in bench\n\n* Remove from README.md\n\nCo-authored-by: Andrew Tretyakov <andrew@tretyakov.xyz>\n\n* Remove from CI\n\n* Cleanup bash scripts\n\n* Add documentation\n\n---------\n\nCo-authored-by: graphite-app[bot] <96075541+graphite-app[bot]@users.noreply.github.com>\nCo-authored-by: Andrew Tretyakov <andrew@tretyakov.xyz>\n\n* rm plotly\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\nCo-authored-by: markosg04 <mgeorghiades@a16z.com>\nCo-authored-by: danielwlz <183745461+danielwlz@users.noreply.github.com>\nCo-authored-by: graphite-app[bot] <96075541+graphite-app[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-09T17:06:37-04:00",
          "tree_id": "edf6a3fcf185e0ea466da1a7fe1877cce16a7ec5",
          "url": "https://github.com/a16z/jolt/commit/66ec47903666a59db48ee1d6b9f0e0b61c7e75e2"
        },
        "date": 1760047086918,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 796308,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 774624,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 799592,
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
            "value": 3685516,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 781300,
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
            "value": 762848,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 756868,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 838856,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 854336,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 349644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5570332,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 787052,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 773452,
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
            "value": 845804,
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
          "id": "b12fd69a293aa67b17b84db217417b4a7e7bd036",
          "message": "Stage 3 memory optimizations (#1001)\n\n* Use u8 for H\n\n* Avoid 8-byte alignment for LookupBits struct",
          "timestamp": "2025-10-09T17:10:35-04:00",
          "tree_id": "abb31b36ee999d4d8434f5da0dc5efa80422ba99",
          "url": "https://github.com/a16z/jolt/commit/b12fd69a293aa67b17b84db217417b4a7e7bd036"
        },
        "date": 1760047324840,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 762324,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 798316,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 762000,
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
            "value": 3711108,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 774020,
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
            "value": 766944,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 764384,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 870276,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 861792,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 343336,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5714472,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 782980,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 771296,
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
            "value": 867644,
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
          "id": "10ba26fa779aba860f5ce8b83e1a8ae3d76bebf9",
          "message": "feat: delayed reduction for all sumchecks (#976)\n\n* feat: delayed reduction for bytecode sumchecks\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* feat: optimize instruction_lookups sumchecks with delayed reduction\n\n* feat: optimize registers val_evaluation sumcheck with delayed reduction\n\n* feat: optimize RAM sumchecks with delayed reduction\n\n* feat: optimize more RAM sumchecks with delayed reduction\n\n* feat: optimize RAM hamming_booleanity sumcheck with delayed reduction\n\n* chore: cleanup\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* feat: optimize init_suffix_polys in the instructions read-raf\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* perf: optimize instruction_lookups/booleanity with delayed reduction\n\nApply delayed reduction patterns from bytecode/booleanity:\n- Use fold_with with F::Unreduced<5> accumulation in phase1\n- Apply mul_trunc<4, 9> for multiplication with B_evals\n- Use mul_unreduced<9> for D_evals multiplication in phase2\n- Minimize Montgomery reductions throughout compute_prover_message\n\n* perf: optimize RAM booleanity with delayed reduction\n\nFixed the delayed reduction pattern by properly handling gamma powers\naccumulation inside the inner loop to avoid incorrect use of\nfrom_montgomery_reduce on Unreduced<5> types. Now uses mul_trunc\nfor final multiplication similar to instruction_lookups pattern.\n\n* perf: keep RAM booleanity sumcheck optimized with barrett_reduce for 5-limb values\n\n- RAM booleanity: Use F::from_barrett_reduce for Unreduced<5> values\n- Keeps gamma multiplication outside inner loop to avoid additional field muls\n- RAM ra_virtual left unoptimized due to variable-length multiplication chain\n\n* perf: optimize RAM ra_virtual sumcheck with delayed reduction\n\n- Multiply all ra evaluations in field arithmetic first\n- Convert to Unreduced<5> for accumulation\n- Use barrett_reduce for final reduction from 5 limbs\n- Properly handles generic field types using Zero trait\n\n* perf: optimize instructions ra_virtual sumcheck with delayed reduction\n\n- Optimize compute_mles_product_sum to use delayed reduction\n- Multiply in field arithmetic first, then convert to Unreduced<5>\n- Accumulate as Unreduced<5> in parallel reduction\n- Use barrett_reduce for final conversion from 5 limbs\n\n* chore: cleanup\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* perf: optimize registers read-write checking with delayed reduction\n\n- Apply delayed reduction to both E_in bound and unbound branches\n- Use mul_unreduced<9> for eq_r_prime_eval and E_out_eval multiplications\n- Accumulate as Unreduced<9> and use montgomery_reduce for final conversion\n- Consistent optimization pattern across all branches\n\n* perf: optimize read-write checking for registers\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-09T17:22:37-04:00",
          "tree_id": "3ee3f722ec8a133d34780bd5bb25180ff79201bf",
          "url": "https://github.com/a16z/jolt/commit/10ba26fa779aba860f5ce8b83e1a8ae3d76bebf9"
        },
        "date": 1760048130361,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 972572,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 956020,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 992564,
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
            "value": 3516260,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 975040,
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
            "value": 995164,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 948792,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1083368,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1059660,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 345540,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 6299148,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 972616,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 998516,
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
            "value": 1048720,
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
          "id": "3957d594166e7e00872ddfd1b2aecf5ae49ff273",
          "message": "Modify postprocess_trace.py command in README\n\nUpdated the postprocess_trace.py command to use benchmark-runs directory.",
          "timestamp": "2025-10-09T18:07:40-04:00",
          "tree_id": "693b7b41f05ad83f28d2e888c11e73d510a22b3d",
          "url": "https://github.com/a16z/jolt/commit/3957d594166e7e00872ddfd1b2aecf5ae49ff273"
        },
        "date": 1760050966624,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 960572,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 953224,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 949424,
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
            "value": 3548616,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 944648,
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
            "value": 956172,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 999920,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1040584,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1057212,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 342484,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5483156,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1008012,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 953632,
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
            "value": 1044116,
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
          "id": "59a0a4b9a7eaec9b76b44865fa34474615943c11",
          "message": "Drop unused trace in profiling (#1004)",
          "timestamp": "2025-10-10T11:07:51-04:00",
          "tree_id": "40c94d1cf8a59104d0acf69576435970acaeefc9",
          "url": "https://github.com/a16z/jolt/commit/59a0a4b9a7eaec9b76b44865fa34474615943c11"
        },
        "date": 1760112119251,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 940928,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 996456,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 964440,
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
            "value": 3487380,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 981016,
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
            "value": 953276,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 937764,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1063768,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1018236,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 345304,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5482888,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1004916,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 958248,
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
            "value": 1056356,
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
          "id": "ef60e5e9ec8cdeb717122bc11a45412a0e62f3d7",
          "message": "feat: batch operations when committing to polys (#1003)\n\n* feat: batch operations when committing to polys\n\n* fix dependencies and patches\n\n* fix index\n\n* clippy",
          "timestamp": "2025-10-10T13:37:24-04:00",
          "tree_id": "3081ced04bbf86d21ca13a681498cc25187ef2f9",
          "url": "https://github.com/a16z/jolt/commit/ef60e5e9ec8cdeb717122bc11a45412a0e62f3d7"
        },
        "date": 1760121061783,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 985568,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 933684,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 937568,
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
            "value": 3581064,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 935264,
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
            "value": 933820,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 962944,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1052772,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 986308,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 349724,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5419808,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 968844,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 942028,
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
            "value": 1055428,
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
          "id": "3812be1ae65587b527c64388d6aeafe87090c26d",
          "message": "feat: more mem reduction (#1006)\n\n* booleanity for ram\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* support any type for indices\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* ram sumchecks\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* bytecode read raf\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* registers val\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-10T16:07:23-04:00",
          "tree_id": "3526e1aa06aaae8b2ddc42b34ceeff51a736b399",
          "url": "https://github.com/a16z/jolt/commit/3812be1ae65587b527c64388d6aeafe87090c26d"
        },
        "date": 1760130116329,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 998492,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 978956,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 934092,
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
            "value": 3673188,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 921852,
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
            "value": 953444,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 923860,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1037132,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1022404,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 343064,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5259840,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 954620,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 956620,
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
            "value": 1045200,
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
          "id": "54a037d4694a7de3d9d2f91e06a2611b04f5106e",
          "message": "feat: Untrusted/Trusted Advice (#987)\n\n* Add private inputs\n\n* Add private_input to the memory\n\n* Changed private input region\n\n* Update tracer\n\n* Update zkvm to include private input\n\n* Left 4 sumchecks to complete the verification\n\n* Verification passes with cheating (Giving the private_input value to the verifier)\n\n* Commitment matches with the evaluation\n\n* Verify the commitment\n\n* More changes\n\n* Refactor code and move evaluation proof and verification to the latest stage\n\n* Refactoring state_manager\n\n* Add dory commit with no hint\n\n* Refactor jolt_dag\n\n* Generate only one evaluation point proof for advice commitment\n\n* Refactor provable macro\n\n* Refactor code\n\n* Add TrustedAdvice to macros and tracer\n\n* Add TrustedAdvice to jolt zkvm\n\n* Commit only to a vector of length equal to untrusted_advice\n\n* Increase untrusted_advice_len by 1\n\n* Fixing untrusted_advice issue when remapping to trusted_advice_start\n\n* More log\n\n* All sumchecks pass correctly\n\n* Add more logs\n\n* Using selectors\n\n* Pass all tests\n\n* Combine advice_openning proofs\n\n* Remove transcript.clone() for advice prove/verify\n\n* Use specifc dory params for each advice type",
          "timestamp": "2025-10-10T19:03:12-04:00",
          "tree_id": "56e0e303b13b93322413f53c241d9d2f824f4ec1",
          "url": "https://github.com/a16z/jolt/commit/54a037d4694a7de3d9d2f91e06a2611b04f5106e"
        },
        "date": 1760140814113,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 956504,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 938064,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 968508,
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
            "value": 3691840,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 928368,
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
            "value": 920360,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1065812,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 997260,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1061800,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 995308,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 347040,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5239356,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1005708,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 946976,
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
            "value": 1036680,
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
          "id": "98bb67ee6c3867f9969e8710ab65f76c260c6003",
          "message": "fix: revert fib example input (#1009)",
          "timestamp": "2025-10-13T09:35:47-07:00",
          "tree_id": "c27d6d93098cba503a71ea24f5e4c49cbcc4033b",
          "url": "https://github.com/a16z/jolt/commit/98bb67ee6c3867f9969e8710ab65f76c260c6003"
        },
        "date": 1760376885026,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 921720,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 928688,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 949404,
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
            "value": 3425012,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 973000,
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
            "value": 976248,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1095416,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 971472,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1069048,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1007124,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 349292,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5172696,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 929852,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 947312,
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
            "value": 1078148,
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
          "id": "de7c8ea14862f29102980a17057d15d0d48c82ce",
          "message": "Fix/reduce virtual registers usage in amo instructinos (#1010)\n\n* reduce virtual register usage\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fix virtual instructinos tests\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* install bottom and zellij in setup machine\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* rename format\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-13T14:39:47-04:00",
          "tree_id": "c89caed8226821dea6f47dfe1469929cab18e438",
          "url": "https://github.com/a16z/jolt/commit/de7c8ea14862f29102980a17057d15d0d48c82ce"
        },
        "date": 1760384387106,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 938644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 931864,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 929660,
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
            "value": 3526416,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 963616,
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
            "value": 929504,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1029272,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 933184,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1064112,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 985372,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 348032,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5219196,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 943316,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 937536,
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
            "value": 1041600,
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
          "id": "f7ab8952d0b781732d1d4fe742f0935b8a47bd92",
          "message": "Change HammingBooleanity to use split eq poly (#1011)\n\n* change hamming booleanity to use split eq poly\n\n* fmt",
          "timestamp": "2025-10-13T14:51:41-04:00",
          "tree_id": "948ae264c8fac0916ba4eae2d0c52dd7c771b76b",
          "url": "https://github.com/a16z/jolt/commit/f7ab8952d0b781732d1d4fe742f0935b8a47bd92"
        },
        "date": 1760384959031,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 921428,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 976072,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 954912,
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
            "value": 3495612,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 950232,
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
            "value": 963976,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1030972,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 925624,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1045048,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1005456,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 343548,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4981536,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 963592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 953464,
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
            "value": 1034272,
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
          "id": "7ea0ebbd587bc7ca1145e6ad0d4a07800571ab5b",
          "message": "feat: delayed reduction for HammingBooleanity & split eq poly for bytecode Booleanity (#1012)\n\n* change hamming booleanity to use split eq poly\n\n* fmt\n\n* add delayed reduction\n\n* merge main\n\n* add split eq poly for bytecode booleanity as well\n\n* add docstring to booleanity\n\n* fix\n\n* small cleanup",
          "timestamp": "2025-10-13T18:09:45-04:00",
          "tree_id": "579c3dfd4e8502754903d520a3011bd7fb77bb82",
          "url": "https://github.com/a16z/jolt/commit/7ea0ebbd587bc7ca1145e6ad0d4a07800571ab5b"
        },
        "date": 1760396817696,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 957108,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 1013220,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 965692,
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
            "value": 3385912,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 956600,
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
            "value": 926740,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1034956,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 921828,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1056908,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 984180,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 347552,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5173192,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 952380,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 929684,
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
            "value": 1038740,
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
          "id": "8f911bb11dd5588a67c4b0919464ab5ef5cf9a48",
          "message": "fix: print padded kHz metrics\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-14T14:57:30-04:00",
          "tree_id": "b4fae1475e6971228ece8f03b33312fed0e32b49",
          "url": "https://github.com/a16z/jolt/commit/8f911bb11dd5588a67c4b0919464ab5ef5cf9a48"
        },
        "date": 1760471876400,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 925472,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 956284,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 934736,
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
            "value": 3412320,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 926920,
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
            "value": 929868,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1051556,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 943056,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1094440,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1121248,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 344936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5163916,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 959080,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 996604,
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
            "value": 1084012,
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
          "id": "11ace0f5581ee14e6f43de26b06002e79ad20291",
          "message": "fix: add RUSTFLAGS and RUST_LOG to benchmarks script\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-14T15:19:53-04:00",
          "tree_id": "b332b0ccf60bf7211c7a8914fa5bd42511a569d4",
          "url": "https://github.com/a16z/jolt/commit/11ace0f5581ee14e6f43de26b06002e79ad20291"
        },
        "date": 1760473141787,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 953008,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 938504,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 980988,
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
            "value": 3562636,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 930560,
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
            "value": 922436,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1121028,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 912960,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1063696,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1083152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 344760,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 5087080,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 941612,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 965364,
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
            "value": 1039828,
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
          "id": "b9b494e55fc9f5ba4d0a44431be7c50072dafacb",
          "message": "feat: virtualize products (#995)\n\n* feat: generalize product sumcheck\n\n* test: integration of ShouldBranch sumcheck, failing\n\n* fix: stage 3 passes, stage 4 has endian-ness problem with the stage 2 eq\n\n* feat: extra stages and val polynomials for bytecode\n\n* feat: ShouldBranch virtualization, requires bespoke instruction batching\n\n* feat: remaining product terms virtualized\n\n* style: various cleanups\n\n* feat: prove should branch in the instruction execution sumcheck (#997)\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* feat: add instruction input sumcheck (#1005)\n\n* perf: initial stage balancing and better address-major batch addition\n\n* perf: shuffling sumchecks\n\n* perf: address major better batch addition\n\n* style: fmt\n\n* Update registers to handle stage 3 claim (#1015)\n\n* fix: test adjustments post virtualization\n\n* style: clippy\n\n* style: small nits during review\n\n* fix: indexing for fixtures\n\n* style: gammas and iterators\n\n* chore: remove unused R1CS inputs\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\nCo-authored-by: Andrew Tretyakov <atretyakov@a16z.com>\nCo-authored-by: Andrew Milson <andrewmilson@users.noreply.github.com>\nCo-authored-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-14T19:30:23-04:00",
          "tree_id": "914d458b613090346b297bd4d926223c5a0fe87f",
          "url": "https://github.com/a16z/jolt/commit/b9b494e55fc9f5ba4d0a44431be7c50072dafacb"
        },
        "date": 1760488252367,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 919912,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 928436,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 947164,
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
            "value": 3545228,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 965052,
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
            "value": 921940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1085408,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 925820,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1035852,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 974332,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 345964,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4441636,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 955980,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 956380,
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
            "value": 1040468,
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
          "id": "b853ad0b53f37f27148f6f58a96c1e979098391f",
          "message": "fix: add more ticks to benchmark plot\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-14T19:31:10-04:00",
          "tree_id": "df33a2a9cef6396fe9b8ead46e3144bc81c0ca6d",
          "url": "https://github.com/a16z/jolt/commit/b853ad0b53f37f27148f6f58a96c1e979098391f"
        },
        "date": 1760488281583,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 984080,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 939564,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 931632,
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
            "value": 3551108,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 933156,
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
            "value": 923388,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1092840,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 907344,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1058132,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 976592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 346920,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4324832,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 943920,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 984848,
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
            "value": 1041448,
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
          "id": "588b159cd8a8d738a162ed637a979b533a30e4c0",
          "message": "fix: hide ticks on proof size plot to avoid overlapping\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-15T14:23:06-04:00",
          "tree_id": "af637c39ad3bc0dd98c6a974d1d28df71dfa7762",
          "url": "https://github.com/a16z/jolt/commit/588b159cd8a8d738a162ed637a979b533a30e4c0"
        },
        "date": 1760556236065,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 963484,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 940756,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 955024,
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
            "value": 3467152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 917532,
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
            "value": 964180,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1037320,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 930128,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1075320,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1003084,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 345592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4339540,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 973564,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 921868,
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
            "value": 1071764,
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
          "id": "eaa6ee35ae34a2184cd83b5910b6fe2c8adf9c93",
          "message": "[JOLT-157] Optimize instruction `read_raf_checking` memory (#1019)\n\n* Avoid copying LookupBits\n\n* Defer computation of ra polynomial + drop stuff in init_log_t_rounds\n\n* Use GruenSplitEqPolynomial for last log(T) rounds of instruction read/raf sumcheck\n\n* Use delayed reduction",
          "timestamp": "2025-10-15T19:06:24-04:00",
          "tree_id": "82bad3aa59953b3f56dfe50ee40540129b0580e9",
          "url": "https://github.com/a16z/jolt/commit/eaa6ee35ae34a2184cd83b5910b6fe2c8adf9c93"
        },
        "date": 1760573154965,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 951856,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 961696,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 948008,
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
            "value": 3590932,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 958924,
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
            "value": 933316,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1087228,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 952116,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1092348,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1010960,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 348948,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4403620,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 980088,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 963716,
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
            "value": 1059036,
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
          "id": "f211d749eafb3f0664eba8a3e40a9356dc3f3c11",
          "message": "fix: update virtual registers constant (#1022)\n\n* update constant\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fix amo instructions\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-17T11:14:53-04:00",
          "tree_id": "ba828dbc98a965a792cae621267b84039ea75886",
          "url": "https://github.com/a16z/jolt/commit/f211d749eafb3f0664eba8a3e40a9356dc3f3c11"
        },
        "date": 1760717831008,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 967024,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 989484,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 953328,
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
            "value": 3415632,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 948040,
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
            "value": 929032,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1072812,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 1006776,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1064340,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1040744,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 346708,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4329752,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 968548,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 994696,
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
            "value": 1067720,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c8be6f128f4360739ef3977df0f30933abdff88b",
          "message": "Reduce memory in bytecode/read_raf_checking.rs (#1024)",
          "timestamp": "2025-10-17T14:27:38-04:00",
          "tree_id": "cc3fc062ff9959628f2f59b2b373022a738e9764",
          "url": "https://github.com/a16z/jolt/commit/c8be6f128f4360739ef3977df0f30933abdff88b"
        },
        "date": 1760729172264,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 1004616,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 945512,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 949452,
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
            "value": 3626736,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 969804,
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
            "value": 949004,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1083516,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 953020,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1119524,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1021144,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 347296,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4378840,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 980576,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 942092,
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
            "value": 1075960,
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
          "id": "8fef76e41dfc237a0a92b8ff3440b8c1248c43f5",
          "message": "Omid/fix advice (#1023)\n\n* Dynamic ordering of trusted and untrusted advice in memory\n\n* Update merkle-tree example\n\n* Remove a comment\n\n* Fix a nit",
          "timestamp": "2025-10-17T16:04:03-04:00",
          "tree_id": "5f3a9cd4eec6aeebe92d11fffc0a0698a9dbaf0a",
          "url": "https://github.com/a16z/jolt/commit/8fef76e41dfc237a0a92b8ff3440b8c1248c43f5"
        },
        "date": 1760734999645,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 978556,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 975780,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 957124,
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
            "value": 3492152,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 996644,
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
            "value": 957432,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1129832,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 962808,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1098644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1019272,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 347772,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4353116,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 987216,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 983444,
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
            "value": 1081972,
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
          "id": "80101ed579f5eb7873029e2c0fddfc9382b9512e",
          "message": "Miscellaneous memory optimizations (#1021)\n\n* Fix stage 7 flamegraph file name\n\n* Use i128 instead of F for inc accumulator in I\n\n* Simplify append_dense interface\n\n* Shared dense polynomial\n\n* Further simplification of opening proof logic\n\n* dense_polynomial_map\n\n* Drop stage 7 sumchecks before PCS::prove\n\n* clippy\n\n* Add nested par_iter to compute flag_claims\n\n* fix tests\n\n* fix CI error",
          "timestamp": "2025-10-17T17:12:01-04:00",
          "tree_id": "15c4a0bc4bcf5a4c15c03b697c55e5aa470efab8",
          "url": "https://github.com/a16z/jolt/commit/80101ed579f5eb7873029e2c0fddfc9382b9512e"
        },
        "date": 1760738996103,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 967844,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 982484,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 945028,
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
            "value": 3787048,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 985728,
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
            "value": 981624,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1183180,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 951604,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1083068,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 993612,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 350328,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4325572,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 974828,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 997772,
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
            "value": 1104856,
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
          "id": "906aaf89c1732e799d15748997f64491079c7bb6",
          "message": "Fix/virtual sequence constraint (#1025)\n\n* Add is_first_in_sequence\n\n* rename stuff\n\n* Add R1CS inputs\n\n* Add constraint\n\n* rename PCSumcheck -> ShiftSumcheck\n\n* Wire up claims to shift sumcheck\n\n* Wire up claims to bytecode read-checking sumcheck\n\n* Fix cycle size test\n\n* graphite suggestions",
          "timestamp": "2025-10-17T19:11:55-04:00",
          "tree_id": "7932d0d2c0cb2ab8f36fb01a909c94dca4704838",
          "url": "https://github.com/a16z/jolt/commit/906aaf89c1732e799d15748997f64491079c7bb6"
        },
        "date": 1760746233652,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 966520,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake2-ex-mem",
            "value": 954584,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "blake3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "blake3-ex-mem",
            "value": 960828,
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
            "value": 3606072,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 937832,
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
            "value": 1001720,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1209640,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 1005936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1082580,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1029796,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 341516,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4435540,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1038704,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 955120,
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
            "value": 1069736,
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
          "id": "9f8a49cb29edcb659c367e00fc9b220822522500",
          "message": "fix: CI\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-20T15:46:13-04:00",
          "tree_id": "8393c9099c7b8dd7fcabfae7d9bdaf92683bd87c",
          "url": "https://github.com/a16z/jolt/commit/9f8a49cb29edcb659c367e00fc9b220822522500"
        },
        "date": 1760992770831,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 984940,
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
            "value": 3600352,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 972964,
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
            "value": 1010120,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1142844,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 945848,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1090068,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1014016,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 348412,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4289200,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1021408,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 980020,
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
            "value": 1069016,
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
          "id": "8487e42d662dd4a19e5d8e6ac37f804f29d49cdd",
          "message": "feat: Univariate skip for Spartan (#971)\n\n* add interpolation stuff\n\n* delete old svo stuff to make way for univariate skip\n\n* delete old spartan quadratic, move outer sumcheck to separate file\n\n* consolidate (split) eq poly\n\n* name changes, merge all to spartan/outer\n\n* fixing stuff\n\n* reorg + more progress\n\n* reorged outer sumcheck proof\n\n* update first round verify univariate skip\n\n* more chnages\n\n* updated r1cs key evaluations\n\n* more changes + fmt + clippy\n\n* some small fixes, tests still fail\n\n* small fixes\n\n* fix verification\n\n* Merge upstream/main into univariate-skip per policy: upstream OpeningProof + 26 constraints; keep branch changes elsewhere; delete interleaved poly and small_value; resolve conflicts\n\n* fixing challenge change\n\n* finish porting challenge\n\n* fixed all errors\n\n* more changes & refactor\n\n* refactor, WIP\n\n* more progress with new staging\n\n* fleshed out uni skip interface, but verification still fails (for remaining sumcheck)\n\n* more refactoring\n\n* added uni skip error enum, reworked to dense az/bz/cz\n\n* WIP, prepare for merge from main\n\n* small fix\n\n* move all constraints to eq-cond, remove all Cz contributions\n\n* fmt + clippy\n\n* parallelize collection of bound az/bz\n\n* fix uni skip logic, more removal of cz\n\n* move NOTICE to spartan folder, small fixes\n\n* further tightening the constraints\n\n* more cleanup\n\n* WIP, starting on uni skip for product virtual\n\n* more progress but still not done\n\n* finally fixed outer sumcheck!\n\n* IT WORKS!!!!\n\n* initial impl of product with uni skip\n\n* fmt + clippy\n\n* fully replaced old product virtualization\n\n* clippy + fmt\n\n* update on product virtualization\n\n* more progress, still debugging\n\n* refactored accumulation primitives\n\n* debugged succesffully the stage 2 uni skip round\n\n* small stuff\n\n* readded append transcript for sumcheck input claim\n\n* fix product sumcheck\n\n* fixed errors from merge\n\n* change coeff type to i128 in ops\n\n* small cleanup\n\n* fix bytecode read raf checking for fused product virtual\n\n* changes to product & cleanup\n\n* reworked accumulation API\n\n* found the damn error\n\n* more shuffling around\n\n* bunch of changes, move booleanity stage 2 => stage 6\n\n* add bool variant to multilinear polynomial\n\n* convert some u8 multilinear to bool multilinear\n\n* two more places to turn u8 into bool for multilinear\n\n* small changes\n\n* fmt + clippy\n\n* final commit for the day...\n\n* add cfg test to signed reduce\n\n* add tracing spans\n\n* another one\n\n* FOUND AND FIX THE BUG!!!\n\n* clippy\n\n* fmt\n\n* fixed another accumulation bug\n\n* clippy\n\n* another one\n\n* docstring\n\n* clippy\n\n* efficiency improvements\n\n* small change: optimized gamma in booleanity\n\n* fmt\n\n* move RAM stuff from stage 3 to stage 4\n\n* cleanup, remove round compression\n\n* addressed mzhu's comments\n\n* delete deferred products as they are not used\n\n* add if-else\n\n* Update jolt-core/src/field/mod.rs\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>\n\n* small fix to uniskip_targets\n\n* cleanup\n\n* change build uniskip first round to have option base\n\n* fmt\n\n* revert G\n\n---------\n\nCo-authored-by: Markos <53157953+markosg04@users.noreply.github.com>",
          "timestamp": "2025-10-20T22:09:55-04:00",
          "tree_id": "c8b56740c43802fd4a48008cd096186b51600633",
          "url": "https://github.com/a16z/jolt/commit/8487e42d662dd4a19e5d8e6ac37f804f29d49cdd"
        },
        "date": 1761015833110,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 971844,
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
            "value": 3432188,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 960300,
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
            "value": 939644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1280544,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 951936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1064072,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1036400,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 346560,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4782648,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 978672,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 981692,
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
            "value": 1077652,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0e75cad23c628f11bd41c04132848a80e62434c4",
          "message": "Reduce mem in registers/val_evaluation.rs by computing LT on-the-fly (#1028)\n\nCo-authored-by: Michael Zhu <mchl.zhu.96@gmail.com>",
          "timestamp": "2025-10-20T22:27:04-04:00",
          "tree_id": "9da0c2d535206eeb282642e0132345ae9e985a37",
          "url": "https://github.com/a16z/jolt/commit/0e75cad23c628f11bd41c04132848a80e62434c4"
        },
        "date": 1761016854863,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 951812,
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
            "value": 3627740,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 930232,
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
            "value": 963228,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1219828,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 945576,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1101936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1019076,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 345120,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4884848,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1041120,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 985248,
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
            "value": 1059956,
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
          "id": "cdcdeee2f63f49106bfebd95d54a5276b65729f8",
          "message": "small fix to outer to avoid overflow (#1032)",
          "timestamp": "2025-10-21T15:25:16-04:00",
          "tree_id": "4914c342ad64b7a59451b662ed9c4e05f7ffd7ce",
          "url": "https://github.com/a16z/jolt/commit/cdcdeee2f63f49106bfebd95d54a5276b65729f8"
        },
        "date": 1761077952979,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 958828,
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
            "value": 3443972,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 959776,
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
            "value": 966984,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1157484,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 955896,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1097188,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1087668,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 343668,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4789976,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1019856,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 952984,
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
            "value": 1071828,
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
            "email": "42178850+0xAndoroid@users.noreply.github.com",
            "name": "Andrew Tretyakov",
            "username": "0xAndoroid"
          },
          "distinct": true,
          "id": "1dd8086b6d03c8f290e887c8eede8cdbdf63ad9c",
          "message": "Use combined poly for shift sumcheck (#1030)",
          "timestamp": "2025-10-21T15:46:31-04:00",
          "tree_id": "bd626b32b177aac2f1e8219d419030cc910c8a20",
          "url": "https://github.com/a16z/jolt/commit/1dd8086b6d03c8f290e887c8eede8cdbdf63ad9c"
        },
        "date": 1761079272035,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 969924,
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
            "value": 3530904,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 984268,
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
            "value": 934836,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1219056,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 930388,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1099168,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1038276,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 345256,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4785432,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 990120,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 957660,
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
            "value": 1097440,
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
          "id": "448384700b3431e3149d28c42dd5a31b4e13a162",
          "message": "ref: remove virtual move (#1033)\n\n* remove virtual move\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fix arch tests\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-21T16:31:49-04:00",
          "tree_id": "08ea1be84c13f7d8aaa5a0cd56a0c7465a521b47",
          "url": "https://github.com/a16z/jolt/commit/448384700b3431e3149d28c42dd5a31b4e13a162"
        },
        "date": 1761082036290,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 942092,
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
            "value": 3437964,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 957192,
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
            "value": 938824,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1209912,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 935956,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1099332,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1024820,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 351848,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4763072,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1012584,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 955224,
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
            "value": 1053120,
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
          "id": "2cc4b6a08e292b6c9c72f8a5c8897270eb79db70",
          "message": "Fix:  Keccak SDK (#1036)\n\n* Fix Keccak256 issue on unaligned inputs\n\n* Fix CI",
          "timestamp": "2025-10-21T18:57:25-04:00",
          "tree_id": "af118e2255d177627e22c5ea3720e06d650ed97d",
          "url": "https://github.com/a16z/jolt/commit/2cc4b6a08e292b6c9c72f8a5c8897270eb79db70"
        },
        "date": 1761090772850,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 949312,
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
            "value": 3615592,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 949500,
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
            "value": 958776,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1239576,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 979000,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1090112,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1089120,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 340036,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4767184,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1008712,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 1007872,
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
            "value": 1066484,
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
          "id": "34a9dda9cebdfa7f4d1a4236811b06bfc1a1ee08",
          "message": "refactor: pass opening accumulator into input claim [JOLT-165] (#1035)\n\n* opening accumulator trait\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* add argument to the function\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* wrap accumulator in an option\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* some sumchecks\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* all remaining sumchecks\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-21T16:09:49-07:00",
          "tree_id": "2190a052c56538ed25f4bc8a6e9b85fb290d36f1",
          "url": "https://github.com/a16z/jolt/commit/34a9dda9cebdfa7f4d1a4236811b06bfc1a1ee08"
        },
        "date": 1761091449626,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 936320,
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
            "value": 3411064,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 972628,
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
            "value": 957192,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1212092,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 949408,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1073184,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1057308,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 347228,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4790208,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1007708,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 962028,
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
            "value": 1065616,
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
          "id": "918b5f46ba3b4d49475ff3fcaf90cd8bd4508cd0",
          "message": "refactor: remove +1 in remap address (#1034)\n\n* ref: remove +1 in remap address\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* change the raf polynomial\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-22T15:42:41-04:00",
          "tree_id": "c53694a51b9c7b35450fbfc2439fe4baa56b5490",
          "url": "https://github.com/a16z/jolt/commit/918b5f46ba3b4d49475ff3fcaf90cd8bd4508cd0"
        },
        "date": 1761165531053,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 942504,
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
            "value": 3569464,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 953376,
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
            "value": 956968,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1199020,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 971936,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1074316,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1035224,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 347156,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4793716,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1030304,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 976064,
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
            "value": 1103192,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "38e78d66242444c238a418997b92403023df89fe",
          "message": "Change bind order of shift and instruction input sumchecks (#1041)",
          "timestamp": "2025-10-22T20:31:32-04:00",
          "tree_id": "741a97d8ca9325de901cf2265b82a9aa47497841",
          "url": "https://github.com/a16z/jolt/commit/38e78d66242444c238a418997b92403023df89fe"
        },
        "date": 1761182744979,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 965360,
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
            "value": 3534932,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 961604,
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
            "value": 1012448,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1229940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 931684,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1097360,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1007484,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 351084,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4753988,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 994312,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 970356,
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
            "value": 1082656,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8f8c4dcfe2f606a6564ae89af4c26ab6d45fc103",
          "message": "refactor: change bind order of RAM val and val final sumchecks (#1045)",
          "timestamp": "2025-10-23T09:29:12-07:00",
          "tree_id": "46f54e8af0916ca55127114e165c707757f68a68",
          "url": "https://github.com/a16z/jolt/commit/8f8c4dcfe2f606a6564ae89af4c26ab6d45fc103"
        },
        "date": 1761240173381,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 987608,
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
            "value": 3540332,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 1011188,
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
            "value": 932284,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1189000,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 950060,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1087784,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1075792,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 345112,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4874712,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1007600,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 977284,
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
            "value": 1060448,
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
          "id": "f80cc2ff97c1551e5ec20dd708211c7a3da88913",
          "message": "docs: add RV64 post (#1046)",
          "timestamp": "2025-10-23T12:00:38-07:00",
          "tree_id": "a665b265d81f22af5593072c2523692d1a19003e",
          "url": "https://github.com/a16z/jolt/commit/f80cc2ff97c1551e5ec20dd708211c7a3da88913"
        },
        "date": 1761249223139,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 984564,
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
            "value": 3527452,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 965312,
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
            "value": 959416,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1241828,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 944788,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1085668,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1047988,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 349716,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4927864,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1008980,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 944304,
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
            "value": 1095456,
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
          "id": "1a47fc01692879d0e215bc5d21daa899a6875531",
          "message": "docs: add inline performance metrics to Jolt Book (#1043)\n\n* Add inline performance metrics to Jolt Book\n\n* Update proving time doc for inlines",
          "timestamp": "2025-10-23T12:43:33-07:00",
          "tree_id": "682aa3f966a10014c47372b4514c4f5cee8694aa",
          "url": "https://github.com/a16z/jolt/commit/1a47fc01692879d0e215bc5d21daa899a6875531"
        },
        "date": 1761251773492,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 950700,
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
            "value": 3566740,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 1038236,
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
            "value": 950052,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1173940,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 926772,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1080080,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1010212,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 345304,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4853132,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 996204,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 988264,
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
            "value": 1060352,
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
          "id": "0017138804dc69240b0d4f4aaf4f40dbd4101f4a",
          "message": "feat: boolean Az in spartan [JOLT-222] (#1047)\n\n* new instruction flags\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* is rd not zero\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* fix bug\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n* bytecode changes\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>\n\n---------\n\nSigned-off-by: Andrew Tretyakov <42178850+0xAndoroid@users.noreply.github.com>",
          "timestamp": "2025-10-23T18:04:26-04:00",
          "tree_id": "e28ee3701b0dd46ccfe0a5ac71d78d3cc117a4a6",
          "url": "https://github.com/a16z/jolt/commit/0017138804dc69240b0d4f4aaf4f40dbd4101f4a"
        },
        "date": 1761260322086,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 1007888,
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
            "value": 3581644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 936168,
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
            "value": 940500,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1218584,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 952596,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1074276,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1010568,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 349328,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4896896,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 979432,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 964996,
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
            "value": 1054872,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "juliuszh@stanford.edu",
            "name": "Jules",
            "username": "JuI3s"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8552465b19113829b61580944cf770a76b3ee867",
          "message": "Compression doc (#1014)\n\n* compression doc\n\n* minor\n\n* fix typo\n\n* fix typo\n\n* basis conversion\n\n* typo\n\n* Compression\n\n* Fix latex render and add code impl\n\n* Improved documentation\n\n* Fix typo",
          "timestamp": "2025-10-27T11:56:13-04:00",
          "tree_id": "dffd1be51885442e3502d2521f52e165aa126e6a",
          "url": "https://github.com/a16z/jolt/commit/8552465b19113829b61580944cf770a76b3ee867"
        },
        "date": 1761583768339,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 956396,
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
            "value": 3614636,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 936840,
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
            "value": 969852,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1173040,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 961060,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1095228,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1034784,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 346788,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4984388,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 994260,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 952148,
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
            "value": 1055672,
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
          "id": "451da72d58568b5cd0da7f2a636f92c4639ad859",
          "message": "Docstring for sumcheck description (#1017)\n\n* docstring for booleanity ram and bytecode\n\n* added docstring for instruction booleanity\n\n* docstring for bytecode read raf\n\n* docstring for instruction read raf\n\n* added docstrings for all sumchecks (except spartan outer & product)\n\n* fix docstring for read raf checking\n\n* add back comments for read raf checking\n\n* more comments for instruction read raf\n\n* clippy\n\n* more clippy",
          "timestamp": "2025-10-27T11:56:50-04:00",
          "tree_id": "8bd71bc3a41e2dc61921562946d02795b8ac9f59",
          "url": "https://github.com/a16z/jolt/commit/451da72d58568b5cd0da7f2a636f92c4639ad859"
        },
        "date": 1761583785342,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 967708,
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
            "value": 3515300,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 952196,
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
            "value": 926464,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1152644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 938080,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1061644,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1036376,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 345560,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4872184,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 966548,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 949680,
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
            "value": 1069104,
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
          "id": "d470bcc00e610c9f33829cbc3a87d98a23ea977f",
          "message": "refactor: DRY booleanity and hamming (#1040)\n\n* refactor: booleanity trait\n\n* style: comments\n\n* style: booleanity trait is conceptually not an extension\n\n* fix: MaybeAllocative\n\n* refactor: hamming trait\n\n* fix: conflicting blanket impls\n\n* style: comments\n\n* fix: instrument traits\n\n* refactor: trait -> struct booleanity\n\n* refactor: trait -> struct hamming\n\n* style: comments and helpers\n\n* refactor: move setup to caller\n\n* refactor: hamming caller data",
          "timestamp": "2025-10-29T15:13:56-04:00",
          "tree_id": "69267d263f76947fa037e85a0f666035a911f0b5",
          "url": "https://github.com/a16z/jolt/commit/d470bcc00e610c9f33829cbc3a87d98a23ea977f"
        },
        "date": 1761768694220,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 999284,
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
            "value": 3662652,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 982624,
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
            "value": 964380,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1191036,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 943780,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1070524,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1027924,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 345584,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4846520,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1016720,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 958844,
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
            "value": 1085068,
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
          "id": "2b71181f52c75aa6b4a0d11608f9339b13cbd778",
          "message": "Omid/test lookup trace (#1048)\n\n* Initial template for tests\n\n* Update JALR instruction\n\n* Remove lookup_output_matches_trace() test for JALR\n\n* Revert JALR back to its original implementation\n\n* Removed unused imports\n\n* Refactor code\n\n* Fix ChangeDivisor Prefix\n\n* Fix VirtualChangeDivisorW\n\n* Refactor\n\n* Refactor\n\n* Fix jalr lookup\n\n* Rename remaineder to dividend\n\n* Fix random rd value",
          "timestamp": "2025-10-29T20:42:42-04:00",
          "tree_id": "d56823545ffbf8892b036912e02fe906ec2e0b72",
          "url": "https://github.com/a16z/jolt/commit/2b71181f52c75aa6b4a0d11608f9339b13cbd778"
        },
        "date": 1761788082442,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 958528,
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
            "value": 3467884,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 957896,
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
            "value": 975884,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1212692,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 983428,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1082788,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1072460,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 344696,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4766064,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 1035244,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 974948,
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
            "value": 1088408,
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
          "id": "968bc7c32d811df99504367d4d6dea7573c2787d",
          "message": "Small fixes to docstrings (#1055)\n\n* fix docstring for inc description\n\n* fix wrong gamma power in bytecode read raf docstring",
          "timestamp": "2025-10-29T20:42:23-04:00",
          "tree_id": "071e0d9baa31fde3e9adf89394a385a15770b263",
          "url": "https://github.com/a16z/jolt/commit/968bc7c32d811df99504367d4d6dea7573c2787d"
        },
        "date": 1761788103195,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 988808,
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
            "value": 3630128,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 941120,
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
            "value": 1001872,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1190576,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 943848,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1078260,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1020712,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 344140,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4759700,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 973704,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 960116,
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
            "value": 1071868,
            "unit": "KB",
            "extra": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "github@randomwalks.xyz",
            "name": "Ari",
            "username": "abiswas3"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ce843b9d933d510f0037711612e16b62e16bcc74",
          "message": "optimistaion/better_parallels (#1056)\n\n- There are few more, but as the code changes I'll have to double check\nthem. I'll keep PR'ing as I find them in these mini PRs.\n\nCo-authored-by: Ari <software@randomwalks.xyz>",
          "timestamp": "2025-10-30T11:18:39-04:00",
          "tree_id": "96fb72191a35fe52deafbd0f28c5ed6d7696a112",
          "url": "https://github.com/a16z/jolt/commit/ce843b9d933d510f0037711612e16b62e16bcc74"
        },
        "date": 1761840725985,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "alloc-mem",
            "value": 976996,
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
            "value": 3567272,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "fibonacci-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "fibonacci-mem",
            "value": 988060,
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
            "value": 963664,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "merkle-tree-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "merkle-tree-mem",
            "value": 1217772,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "muldiv-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "muldiv-mem",
            "value": 949360,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "multi-function-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "multi-function-mem",
            "value": 1075656,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "random-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "random-mem",
            "value": 1034952,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "recover-ecdsa-mem",
            "value": 348988,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-chain-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-chain-mem",
            "value": 4878488,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha2-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha2-ex-mem",
            "value": 995448,
            "unit": "KB",
            "extra": ""
          },
          {
            "name": "sha3-ex-time",
            "value": 0,
            "unit": "s",
            "extra": ""
          },
          {
            "name": "sha3-ex-mem",
            "value": 969008,
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
            "value": 1081368,
            "unit": "KB",
            "extra": ""
          }
        ]
      }
    ]
  }
}