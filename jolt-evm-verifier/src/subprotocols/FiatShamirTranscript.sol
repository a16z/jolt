// SPDX-License-Identifier: MIT

pragma solidity >=0.8.0;

// We wrap this memory region mostly to discourage downstream touching of it
// Note - Always init this via the new_transcript function as this hashes seed data and
//        appends the protocol name
// Note - We don't clean the data in the memory region as we always only hash up to the point we stored in each
//        function context. Direct access of the region of memory inside of the transcript may contain dirty bits.
struct Transcript {
    // A laid out memory region of [32 byte seed][uint256 n rounds][working memory for hashes]
    bytes32[] region;
}

// An implementation of a Fiat Shamir Public Coin protocol which matches the one from the Jolt rust repo
// We first define an object and memory region (the max memory limit of writes is defined on init),
// then users can write data to this transcript or pull deterministic randoms values.
// Care should be taken to ensure that all writes are done with consistent amounts of data.
library FiatShamirTranscript {
    /// Generates a new transcript held in memory by initializing the region in memory before hashing the protocol
    /// name into the first position
    /// @param encodedName A string of up to 32 bytes encoded as a bytes32 in solidity
    /// @param maxSize The number of 32 byte memory slots to reserve as the max input to the hashed region, MUST be > 1.
    function new_transcript(bytes32 encodedName, uint256 maxSize) internal pure returns (Transcript memory) {
        // We have to write at least 32 bytes for the constant string labels in Jolt
        assert(maxSize > 1);
        // Allocates our transcript memory safely
        bytes32[] memory internal_data = new bytes32[](maxSize + 2);

        assembly ("memory-safe") {
            // Hashes over the protocol name at the start of the array and stores into first array position
            let dataPtr := add(internal_data, 0x20)
            mstore(dataPtr, encodedName)
            mstore(dataPtr, keccak256(dataPtr, 0x20))
        }
        return (Transcript(internal_data));
    }

    /// Appends 32 bytes of data to a transcript which is already running. Increments the n round and resets seed.
    /// @param transcript The transcript we are hashing the value into
    /// @param added The data which is hashed into the public coin's seed.
    function append_bytes32(Transcript memory transcript, bytes32 added) internal pure {
        bytes32[] memory region = transcript.region;
        assembly ("memory-safe") {
            // Hashes over the protocol name at the start of the array and stores into first array position
            let seedPtr := add(region, 0x20)
            let nRoundPtr := add(seedPtr, 0x20)
            // Overwrite store the data to the start of the memory region
            mstore(add(nRoundPtr, 0x20), added)
            // Hash starting at the internal seed over three words
            let hashed := keccak256(seedPtr, 0x60)
            // Update the seed and the nrounds
            mstore(nRoundPtr, add(mload(nRoundPtr), 1))
            mstore(seedPtr, hashed)
        }
    }

    /// Appends a u64 following the conventions of the rust version (namely that the rust version also packs this to u256)
    /// Mostly we have this function to maintain compatibility with the rust version.
    /// @param transcript The transcript we are hashing the value into
    /// @param added The data which is hashed into the public coin's seed.
    function append_u64(Transcript memory transcript, uint64 added) internal pure {
        /// Because we are passing in a solidity type checked uint64 we can just cast to 32 bytes and
        /// this ensures the top bits are clean
        append_bytes32(transcript, bytes32(uint256(added)));
    }

    /// Appends a u256 scalar value to the transcript
    /// WARN - This function assumes that the caller has done the mod to ensure the top bits are zero
    /// @param transcript The transcript we are hashing the value into
    /// @param added The data which is hashed into the public coin's seed.
    function append_scalar(Transcript memory transcript, uint256 added) internal pure {
        append_bytes32(transcript, bytes32(added));
    }

    /// We append a vector of scalars with this function, and follow the pattern of rust where this is done
    /// as an append of singular scalars of vector length
    /// WARN - This function assumes that the caller has done the mod to ensure the top bits are zero
    /// @param transcript The transcript we are hashing the value into
    /// @param added The data which is hashed into the public coin's seed.
    function append_vector(Transcript memory transcript, uint256[] memory added) internal pure {
        append_bytes32(transcript, "begin_append_vector");
        for (uint256 i = 0; i < added.length; i++) {
            append_bytes32(transcript, bytes32(added[i]));
        }
        append_bytes32(transcript, "end_append_vector");
        /// TODO (aleph_v) we may want a calldata pointer version of this
        /// TODO (aleph_v) After looking at gas performance we might want to change this to append a whole memory region instead
        ///      of stepping
    }

    /// We append a point of a N/pN x N/pN where p is less than 2^256 and the point is encoded as (32 bytes, 32 bytes)
    /// On eth mainnet this will be a point on the bn256 pairing curve which there is a precompile for
    /// WARN - This function assumes that the caller has done the mod to ensure the top bits are zero for x and y
    /// @param transcript The transcript we are hashing the value into
    /// @param added_x The point's x value which is hashed into the public coin's seed.
    /// @param added_y The points y value which hashed into the public coin's seed.
    function append_point(Transcript memory transcript, uint256 added_x, uint256 added_y) internal pure {
        // Here because we want to append two values without incrementing the counter we don't use
        // the default append_bytes32
        bytes32[] memory region = transcript.region;
        // You can only call this function if the region was initialized with at least max_size 2
        assert(region.length >= 4);
        assembly ("memory-safe") {
            // Hashes over the protocol name at the start of the array and stores into first array position
            let seedPtr := add(region, 0x20)
            let nRoundPtr := add(seedPtr, 0x20)
            // Overwrite store the data to the memory region
            mstore(add(nRoundPtr, 0x20), added_x)
            mstore(add(nRoundPtr, 0x40), added_y)
            // Hash starting at the internal seed over four words
            let hashed := keccak256(seedPtr, 0x80)
            // Update the seed and the nrounds
            mstore(nRoundPtr, add(mload(nRoundPtr), 1))
            mstore(seedPtr, hashed)
        }
    }

    /// We append an array of points to the the transcript via a uint256 vector in which the x coordinates
    /// are the even points and the y coordinates are the odd points
    /// Matching the behavior of the rust code we do this by evoking point addition for each point
    /// @param transcript The transcript we are hashing the value into
    /// @param added The points encoded as [x0, y0, x1, y1, ... xn, yn]
    function append_points(Transcript memory transcript, uint256[] memory added) internal pure {
        // We wrap vector addition in messages to ensure that it is not possible to misattribute
        // the end of the array with a normal point addition.
        append_bytes32(transcript, "begin_append_vector");
        for (uint256 i = 0; i < added.length; i += 2) {
            append_point(transcript, added[i], added[i + 1]);
        }
        append_bytes32(transcript, "end_append_vector");
        // TODO - Similar comments as the vector append for scalars
    }

    /// We include a bytes append method but require the use of it to be encoded in a secure way to a multiple of
    /// 32 bytes and represented as a bytes32 array
    /// @param transcript The transcript we are hashing the value into
    /// @param added The bytes array we add to the transcript
    // TODO - This whole routine is complex, can we just get rid of it and have a 32 byte array version? or none?
    function append_bytes(Transcript memory transcript, bytes32[] memory added) internal pure {
        bytes32[] memory region = transcript.region;
        // The length checks are done in high level sol so we don't assert the memory length
        for (uint256 i = 0; i < added.length; i++) {
            region[i + 2] = added[i];
        }

        // A mem copy routine for a byte array
        assembly ("memory-safe") {
            // Hashes over the protocol name at the start of the array and stores into first array position
            let seedPtr := add(region, 0x20)
            let nRoundPtr := add(seedPtr, 0x20)
            // Hash starting at the internal seed over four words
            let hashed := keccak256(seedPtr, add(0x40, mul(0x20, mload(added))))
            // Update the seed and the nrounds
            mstore(nRoundPtr, add(mload(nRoundPtr), 1))
            mstore(seedPtr, hashed)
        }
    }

    /// Loads a 32 byte deterministic random from a transcript by hashing the internal seed and round constant
    /// Then it updates the seed and round constant
    /// @param transcript The transcript which is a running hash of previous assigned data
    function challenge_bytes32(Transcript memory transcript) internal pure returns (bytes32 challenge) {
        // Loads the pointer to the data field
        bytes32[] memory region = transcript.region;
        // Hash just the seed and round constant
        assembly ("memory-safe") {
            let dataPtr := add(region, 0x20)
            // the hash starts at the seed value and goes for 32 bytes plus the bytes in added
            let hashed := keccak256(dataPtr, 0x40)
            // Update the seed and the nrounds
            let nroundsPtr := add(dataPtr, 0x20)
            mstore(nroundsPtr, add(mload(nroundsPtr), 1))
            mstore(dataPtr, hashed)
            challenge := hashed
        }
    }

    /// Returns a scalar prime field element by loading a 32 byte random region then modding with the
    /// provided constant. This is known to introduce some bias, and such values should not be used
    /// for applications which are highly sensitive to slight deviations from the uniform distribution (eg private keys)
    /// @param transcript The transcript which is a running hash of previous assigned data
    /// @param order The value which we mod the result by
    function challenge_scalar(Transcript memory transcript, uint256 order) internal pure returns (uint256) {
        return (uint256(challenge_bytes32(transcript)) % order);
    }

    /// Returns an array of scalars by repeatedly calling the challenge_scalar function, the same security considerations
    /// apply to the produced numbers. This function allocates new memory.
    /// @param transcript The transcript which is a running hash of previous assigned data
    /// @param numb The number of scalars we want.
    /// @param order The value which we mod the result by
    function challenge_scalars(Transcript memory transcript, uint256 numb, uint256 order)
        internal
        pure
        returns (uint256[] memory challenges)
    {
        challenges = new uint256[](numb);
        for (uint256 i = 0; i < numb; i++) {
            challenges[i] = challenge_scalar(transcript, order);
        }
    }
}
