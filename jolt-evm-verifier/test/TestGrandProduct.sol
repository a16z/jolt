// SPDX-License-Identifier: MIT

pragma solidity ^0.8.21;

import {TestBase} from "./base/TestBase.sol";
import {GrandProductVerifier, GrandProductProof} from "../src/subprotocols/GrandProductVerifier.sol";
import {Transcript, FiatShamirTranscript} from "../src/subprotocols/FiatShamirTranscript.sol";
import {Fr} from "../src/subprotocols/Fr.sol";

import "forge-std/console.sol";

contract TestGrandProduct is TestBase {
    function testValidGrandProductProof() public {
        // TODO(moodlezoup): Update GrandProductVerifier.sol for new batching protocol
        vm.skip(true);
        // Inits the transcript with the same string label as the rust code
        Transcript memory transcript = FiatShamirTranscript.new_transcript("test_transcript", 4);

        (GrandProductProof memory proof, uint256[] memory claims, uint256[] memory r) = getGrandProductExample();

        Fr[] memory claims_fr;
        assembly ("memory-safe") {
            claims_fr := claims
        }

        Fr[] memory verifierRGrandProduct = GrandProductVerifier.verifyGrandProduct(proof, claims_fr, transcript);

        for (uint256 i = 0; i < verifierRGrandProduct.length; i++) {
            assertTrue(r[i] == Fr.unwrap(verifierRGrandProduct[i]));
        }
    }
}
