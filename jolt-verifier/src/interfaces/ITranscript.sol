// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {Fr} from "../reference/Fr.sol";

interface ITranscript {
//    function challengeVector(string memory label) external returns (bytes[] memory transcript);
// TODO: modify
    function challengeScalar(string memory label, uint256 index) external returns (Fr challenge);
    //function appendScalar(string memory label, Fr scalar) external;
    //function appendMessage(string memory label, string memory message) external;
}
