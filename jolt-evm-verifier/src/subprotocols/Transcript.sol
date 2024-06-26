// SPDX-License-Identifier: MIT

pragma solidity >=0.8.0;

// An implementation of a Fiat Shamir Public Coin protocol which matches the one from the Jolt rust repo
// We first define an object and memory region (the max memory limit of writes is defined on init),
// then users can write data to this trascript or pull determistic randoms values.
// Care should be taken to ensure that all writes are done with consistent amounts of data.
contract FiatShamirTranscript {

    // We wrap this memory region mostly to discourage downsteam touching of it
    struct Transcript {
        // A laid out memory region of [32 byte seed][uint256 n rounds][working memory for hashes]
        bytes32[] region;
    }

    function new_transcript(bytes32 encocdedName, uint256 maxSize) internal returns(Transcript memory) {
        // We have to write at least 32 bytes for the constant string labels in Jolt
        assert(maxSize > 1);
        // Allocates our transcript memory safely
        bytes32[] memory internal_data = new bytes32[](maxSize+2);


        bytes32 seed = keccak256(bytes(encocdedName));
        internal_data[0] = seed;
        return (Transcript(internal_data));
    }
}